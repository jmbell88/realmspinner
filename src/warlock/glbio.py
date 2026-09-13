"""GLB container reading and rewriting.

Factored out of ``pipelines.postprocess`` so the viewer's loader and the
transform-insertion code share one implementation of the binary format. The
split/rebuild pair is deliberately lossless in the second half: ``rest`` is
every byte after the JSON chunk, verbatim, so rewriting the JSON never touches
the binary payload (which is the only thing large enough to be worth not
copying wrongly).
"""

from __future__ import annotations

import json
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

GLB_MAGIC = 0x46546C67  # 'glTF'
CHUNK_JSON = 0x4E4F534A  # 'JSON'
CHUNK_BIN = 0x004E4942  # 'BIN\0'


def split_glb(data: bytes) -> tuple[bytes, dict, bytes]:
    """-> (12-byte header, parsed JSON chunk, every following byte verbatim)."""
    # struct.error is not what callers key on: every other refusal in this
    # module is a ValueError, and a file too short to hold its own header is a
    # refusal like any other rather than an unpacking accident.
    try:
        magic, version, _length = struct.unpack_from("<III", data, 0)
    except struct.error:
        raise ValueError("truncated GLB: shorter than its 12-byte header") from None
    if magic != GLB_MAGIC:
        raise ValueError("not a GLB file")
    if version != 2:
        raise ValueError(f"unsupported GLB version {version}")
    try:
        chunk_len, chunk_type = struct.unpack_from("<II", data, 12)
    except struct.error:
        raise ValueError("truncated GLB: missing its first chunk header") from None
    if chunk_type != CHUNK_JSON:
        raise ValueError("first GLB chunk is not JSON")
    if 20 + chunk_len > len(data):
        # A slice past the end is silently short in Python, so a body truncated
        # mid-JSON would reach json.loads as a partial document and fail there
        # with a decoder's error rather than the format's. Callers key on
        # ValueError either way; this one says what actually went wrong.
        raise ValueError("truncated GLB: the JSON chunk overruns the file")
    start = 20
    doc = json.loads(data[start : start + chunk_len])
    # The 2026-09-11 audit, finding clay-03: a JSON chunk that parses cleanly
    # but is not an object (a bare array, say) used to reach every downstream
    # ``.get(...)`` call -- gltf.load's own, and clay/glbimport.py's
    # _declared_budget, which runs *before* glb_to_claydoc's protective
    # try/except -- as a bare AttributeError instead of the named refusal
    # every other malformed-GLB boundary in this module raises. Checked once
    # here so every caller inherits the fix.
    if not isinstance(doc, dict):
        raise ValueError("not a GLB file: its JSON chunk is not an object")
    return data[:12], doc, data[start + chunk_len :]


def rebuild_glb(header: bytes, gltf: dict, rest: bytes) -> bytes:
    # The JSON chunk pads with spaces (0x20) per the glTF spec, not zeros.
    payload = json.dumps(gltf, separators=(",", ":")).encode()
    payload += b" " * (-len(payload) % 4)
    total = len(header) + 8 + len(payload) + len(rest)
    return (
        header[:8]
        + struct.pack("<I", total)
        + struct.pack("<II", len(payload), CHUNK_JSON)
        + payload
        + rest
    )


def read_glb(path: Path | bytes) -> tuple[dict, bytes]:
    """-> (glTF JSON, the BIN chunk's payload).

    What a loader wants, as opposed to what a rewriter wants: the binary buffer
    unwrapped from its chunk header. A GLB with no BIN chunk yields ``b""``
    rather than raising -- the accessors then have to name an external buffer,
    which the loader refuses on its own terms with a better message.
    """
    data = path if isinstance(path, bytes) else Path(path).read_bytes()
    _header, gltf, rest = split_glb(data)
    offset = 0
    while offset + 8 <= len(rest):
        chunk_len, chunk_type = struct.unpack_from("<II", rest, offset)
        body = rest[offset + 8 : offset + 8 + chunk_len]
        if chunk_type == CHUNK_BIN:
            return gltf, body
        offset += 8 + chunk_len
    return gltf, b""


# --- root extras ---------------------------------------------------------

# Both helpers below exist for the character pipeline's export step: it copies
# the served ``animated.glb`` -- never rewrites it in place -- and stamps or
# renames things onto that copy, so ``rest`` (the BIN chunk, the mesh and
# animation-sampler data) must come out byte-identical or a re-export would
# silently redo the vertex/keyframe compression each time.


def root_extras(data: bytes) -> dict:
    """-> the glTF root's ``extras`` object, or ``{}`` if it carries none."""
    _header, gltf, _rest = split_glb(data)
    return gltf.get("extras", {})


def set_root_extras(data: bytes, key: str, value: Any) -> bytes:
    """Set ``gltf["extras"][key] = value`` on the root JSON and re-emit the GLB.

    The host stamps a clip-library digest here so a stale bake -- an
    ``animated.glb`` exported before its source animations last changed -- can
    be detected without re-parsing every keyframe. ``extras`` is created if
    absent; any other key already in it is preserved, and the BIN chunk is
    carried through ``rest`` untouched (see ``rebuild_glb``).
    """
    if not isinstance(key, str) or not key:
        raise ValueError("root extras key must be a non-empty string")
    try:
        json.dumps(value)
    except TypeError as exc:
        raise ValueError(f"root extras value for {key!r} is not JSON-serialisable") from exc
    header, gltf, rest = split_glb(data)
    new_gltf = dict(gltf)
    extras = dict(new_gltf.get("extras", {}))
    extras[key] = value
    new_gltf["extras"] = extras
    return rebuild_glb(header, new_gltf, rest)


# --- animations ------------------------------------------------------------


def animation_names(data: bytes) -> list[str]:
    """-> each animation's ``name``, in file order; ``""`` for an unnamed one."""
    _header, gltf, _rest = split_glb(data)
    return [anim.get("name", "") for anim in gltf.get("animations", [])]


def rename_animations(data: bytes, mapping: Mapping[str, str]) -> bytes:
    """Rename ``animations[i].name`` per ``mapping`` and re-emit the GLB.

    Godot reads the ``-loop`` suffix on a clip's name as "loop this
    animation", so a later export step renames looping clips on a *copy* of
    the served ``animated.glb`` rather than on the file this app keeps
    editing. Renaming is by current name because that is what an export step
    has in hand (the clip list, not its indices), and an animation missing
    from ``mapping`` is untouched -- both its name and its samplers.

    Refused rather than silently applied: a ``mapping`` key naming an
    animation this file does not have (a stale clip list), a target name that
    collides with another animation's name after the rename (mapped or not),
    and an empty target name (glTF allows an unnamed animation but not by
    renaming one into it, since ``""`` participates in the collision check
    like any other name).
    """
    header, gltf, rest = split_glb(data)
    animations = gltf.get("animations", [])
    current_names = [anim.get("name", "") for anim in animations]
    current_set = set(current_names)
    for old_name in mapping:
        if old_name not in current_set:
            raise ValueError(f"no animation named {old_name!r} in this file")

    new_names = list(current_names)
    for i, name in enumerate(current_names):
        if name in mapping:
            new_name = mapping[name]
            if not new_name:
                raise ValueError("a renamed animation's name must not be empty")
            new_names[i] = new_name

    seen: set[str] = set()
    for name in new_names:
        if name in seen:
            raise ValueError(f"renaming would leave two animations named {name!r}")
        seen.add(name)

    new_animations = []
    for anim, new_name in zip(animations, new_names, strict=True):
        if anim.get("name", "") == new_name:
            new_animations.append(anim)
        else:
            renamed = dict(anim)
            renamed["name"] = new_name
            new_animations.append(renamed)

    new_gltf = dict(gltf)
    new_gltf["animations"] = new_animations
    return rebuild_glb(header, new_gltf, rest)
