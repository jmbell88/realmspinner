"""Skeleton templates, rig/pose validation, and the Blender subprocess boundary.

This module is the *host* side of rigging and never imports ``bpy``. That split
is deliberate and load-bearing:

* ``bpy`` is process-global and not thread-safe, and it hard-crashes (not
  raises) on some degenerate geometry -- which trellis output frequently is.
  Importing it into the app process would put a segfault between the user and
  every other job. So Blender work runs out-of-process, mirroring how
  ``pipelines/trellis.py`` treats ``trellis-server.exe``.
* Everything decidable without Blender -- which templates exist, whether a pose
  payload is well-formed, where a template's joints land on a given bbox -- is
  decidable here, in plain Python, under test, with no 400 MB import.

``pipelines/blender_worker.py`` is the other side: it runs as
``python -m warlock.pipelines.blender_worker`` in this same interpreter and
imports :func:`fit_template` from here, so host and worker can never disagree
about where a joint goes.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import winjob

log = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"

# Same shape and generator as a job id (uuid4().hex[:12]), and validated for the
# same reason: config.job_dir() and every path built under it do no sanitisation.
RESOURCE_ID_RE = re.compile(r"^[0-9a-f]{12}$")

# A template's key is interpolated into filenames -- the Poser preview cache
# (``poselib.preview_path``) and its digest both build ``<key>.glb`` /
# ``<key>.json`` paths from it -- so the key must be a safe path component,
# and it comes out of the JSON body, not the filename.
TEMPLATE_KEY_RE = re.compile(r"^[a-z0-9_]+$")

# Progress lines the worker prints. Anything else on its stdout is log noise.
PROGRESS_PREFIX = "[blender]"
RE_PROGRESS = re.compile(r"^\[blender\]\s+([\d.]+)\s+(.*)$")

# A bone shorter than this fraction of the mesh's largest dimension is one
# Blender silently deletes on leaving edit mode, taking its children with it.
MIN_BONE_FRACTION = 0.002

# Equal to ``studio.viewer.programs.MAX_JOINTS``: the viewer's skinning shader
# has a fixed-size joint-matrix uniform array, and a skin with more joints than
# that draws at rest (see ``viewer.scene``'s warning) rather than failing --
# silently, since nothing about a rig job says "this skin will not animate".
# Refusing a skeleton editor build over that ceiling, here, is what keeps that
# silent failure unreachable from the one door that can grow a rig past 64
# bones. ``tests/test_rigging.py`` pins the two constants equal so they cannot
# drift apart independently.
MAX_SKELETON_BONES = 64

# A hand-editable skeleton's bone name: Blender truncates a bone name at 63
# bytes and silently renames a duplicate with a numeric suffix -- either of
# which would make the name this validator accepted not the name Blender ends
# up building, so both are refused here instead of discovered after a bake.
BONE_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,63}$")

# Rigging a 300k-face mesh with automatic weights is minutes of CPU, not hours.
BLENDER_TIMEOUT = 1800.0

# Tail of the worker's output kept for the error message when it fails.
ERROR_TAIL_CHARS = 2000


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def is_valid_id(value: str) -> bool:
    return bool(RESOURCE_ID_RE.match(value))


# --- template registry ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Template:
    key: str
    label: str
    root: str
    bones: tuple[dict[str, Any], ...]
    mirror_pairs: tuple[tuple[str, str], ...]


_templates: dict[str, Template] | None = None

# read_record's stat-before-read guard (see MAX_RECORD_BYTES, far below), but
# these three loaders each predate it and never got one: the 2026-09-11 audit
# (poser-03) found _load_templates/_load_pose_library/_load_clip_library each
# doing a bare ``json.loads(path.read_text(...))`` with no size check at all,
# so a file dropped in TEMPLATE_DIR, PRESET_DIR/BATTERY_DIR, or -- the one
# genuinely user-editable directory of the three, per user_clip_dir()'s own
# docstring -- CLIP_DIR/user_clip_dir() was read into memory in full before
# anything could refuse it.
#
# A shipped template is 1-3 KB and a shipped pose library 2-4 KB (checked on
# disk before choosing this); 1 MiB is the same three-orders-of-magnitude
# headroom MAX_RECORD_BYTES already uses for a single pose/rig record, and
# plenty for a hand-authored template or preset file that will never approach
# it.
MAX_TEMPLATE_BYTES = 1 << 20

# A shipped clip library is up to ~44 KB today, but unlike a template it is
# also something a user edits and re-saves through service.clips.save, whose
# own write-door caps (MAX_LIBRARY_KEYS=256 poses, MAX_KEYS=64 keys/clip) allow
# a file substantially larger once every pose carries a full skeleton's worth
# of bones at JSON's verbosity. 4 MiB leaves real headroom above that
# legitimate maximum while still refusing anything that is not a hand-authored
# or program-written clip library.
MAX_CLIP_LIBRARY_BYTES = 4 << 20


def _read_json_capped(path: Path, ceiling: int) -> Any:
    """One JSON file, refusing anything over ``ceiling`` before it is parsed.

    Raises on any problem -- oversized, unreadable, not valid JSON -- so every
    caller's existing ``except Exception: log.exception(...)`` ("a malformed
    file costs you that entry, not the app") already covers this the same way
    it covers a bad body; this only moves the guard in front of the read
    instead of leaving it absent. Not ``read_record``: that one also demands
    the document be a dict, which the pose/rig sidecars it serves need but
    these three registries do not -- their own parsing already raises a
    specific, more useful error on the wrong shape.
    """
    size = path.stat().st_size
    if size > ceiling:
        raise ValueError(f"{path} is {size} bytes, over the {ceiling}-byte ceiling")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_templates() -> dict[str, Template]:
    found: dict[str, Template] = {}
    for path in sorted(TEMPLATE_DIR.glob("*.json")):
        try:
            raw = _read_json_capped(path, MAX_TEMPLATE_BYTES)
            template = _parse_template(raw)
            # The key <-> filename convention is enforced here rather than
            # assumed downstream: ``poselib.template_digest`` reads
            # ``TEMPLATE_DIR/<key>.json`` back, and a key that named a
            # different file would make the preview cache hash the wrong
            # bytes -- or nothing at all.
            if template.key != path.stem:
                raise ValueError(
                    f"template key {template.key!r} does not match filename {path.name}"
                )
            found[template.key] = template
        except Exception:
            # A malformed template must cost you that template, not the app.
            log.exception("skipping unusable skeleton template %s", path)
    return found


def _parse_template(raw: dict[str, Any]) -> Template:
    if not TEMPLATE_KEY_RE.match(str(raw["key"])):
        raise ValueError(f"template key {raw['key']!r} is not a safe path component")
    bones = raw["bones"]
    names = {b["name"] for b in bones}
    if len(names) != len(bones):
        raise ValueError("duplicate bone names")
    for b in bones:
        if b["parent"] is not None and b["parent"] not in names:
            raise ValueError(f"bone {b['name']!r} has unknown parent {b['parent']!r}")
        for end in ("head", "tail"):
            if len(b[end]) != 3:
                raise ValueError(f"bone {b['name']!r} {end} is not a 3-vector")
    if raw["root"] not in names:
        raise ValueError(f"root {raw['root']!r} is not a bone")
    # The root must be parentless: rig.json's ``root`` is this bone, and
    # ``blender_worker._apply_root_translation`` inverts only the bone's own
    # rest frame -- sound exactly while no parent's *pose* sits above it. Every
    # shipped template already satisfies this; enforcing it keeps the premise a
    # fact rather than a habit.
    root_parent = next(b["parent"] for b in bones if b["name"] == raw["root"])
    if root_parent is not None:
        raise ValueError(f"root {raw['root']!r} must be parentless")
    return Template(
        key=raw["key"],
        label=raw["label"],
        root=raw["root"],
        bones=tuple(
            {
                "name": b["name"],
                "parent": b["parent"],
                "head": [float(v) for v in b["head"]],
                "tail": [float(v) for v in b["tail"]],
            }
            for b in bones
        ),
        mirror_pairs=tuple((a, b) for a, b in raw.get("mirror_pairs", [])),
    )


def templates() -> dict[str, Template]:
    global _templates
    if _templates is None:
        _templates = _load_templates()
    return _templates


def get_template(key: str) -> Template:
    try:
        return templates()[key]
    except KeyError:
        raise ValueError(f"unknown skeleton template {key!r}") from None


def catalog() -> list[dict[str, str]]:
    """The template table in the same {key, label} shape the UI selects use."""
    return [{"key": t.key, "label": t.label} for t in templates().values()]


# --- shipped pose libraries -------------------------------------------------
#
# A pose is a bone-name -> local-quaternion map, and fit_template puts a given
# template's bones in the same place on every mesh. So a pose authored against
# one humanoid rig applies to every other humanoid rig -- which is what makes a
# shipped library possible at all, and why these live next to the templates
# rather than being seeded into each job's poses/ directory.

PRESET_DIR = TEMPLATE_DIR / "poses"

# The deformation battery: the poses a *rig* is reviewed in. Same format and
# the same loader as the shipped presets, deliberately -- they are rendered
# through the same sheet pipeline -- but a separate directory, because these
# are not poses anybody would ship an asset in and they have no business in the
# pose library the user picks from.
BATTERY_DIR = TEMPLATE_DIR / "deform_qa"

_presets: dict[str, list[dict[str, Any]]] | None = None
_batteries: dict[str, list[dict[str, Any]]] | None = None


def _load_pose_library(
    directory: Path, *, with_ids: bool = False
) -> dict[str, list[dict[str, Any]]]:
    found: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            raw = _read_json_capped(path, MAX_TEMPLATE_BYTES)
            rows = []
            for i, pose in enumerate(raw["poses"]):
                row = {"name": str(pose["name"]), "bones": pose["bones"]}
                if with_ids:
                    # A synthetic id, and only where the library is rendered
                    # rather than offered: a sheet row is keyed by (pose id,
                    # frame), so rows that all had no id would put the first
                    # pose in every row of the sheet. A preset the user picks
                    # keeps having none -- it is not a saved pose.
                    row["id"] = f"{path.stem}_{i}"
                rows.append(row)
            found[path.stem] = rows
        except Exception:
            # A malformed library costs you that library, not the app -- the
            # same rule _load_templates follows.
            log.exception("skipping unusable pose library %s", path)
    return found


def preset_poses(template_key: str) -> list[dict[str, Any]]:
    """The shipped poses for a template. Raises ValueError on an unknown key."""
    global _presets
    get_template(template_key)  # validates the key against the registry
    if _presets is None:
        _presets = _load_pose_library(PRESET_DIR)
    return [dict(p) for p in _presets.get(template_key, [])]


def deform_battery(template_key: str) -> list[dict[str, Any]]:
    """The review poses for a template, or [] if it has no battery yet.

    Empty is a normal answer and the caller renders nothing: a battery is
    authored per skeleton (a squat means nothing to a fish), and a template
    without one should cost the QA sheet, never the rig.
    """
    global _batteries
    get_template(template_key)  # validates the key against the registry
    if _batteries is None:
        _batteries = _load_pose_library(BATTERY_DIR, with_ids=True)
    return [dict(p) for p in _batteries.get(template_key, [])]


# The Troupe clip library: the same argument as the pose presets one directory
# up, one level higher. A *clip* is an ordered list of those poses plus how many
# frames each step holds -- so the thing that is portable across rigs is now a
# whole walk cycle rather than a single silhouette, and authoring cost moves
# from 256 frames per character to ~22 keyframes once ever.
CLIP_DIR = TEMPLATE_DIR / "clips"

_clips: dict[str, dict[str, Any]] | None = None
#: The same shape, read from ``user_clip_dir()``. A template present here
#: *replaces* the shipped entry rather than merging with it: a clip library
#: is internally consistent by construction (every clip names poses the same
#: file carries), and half of one file's poses under another file's clips is
#: exactly the hole ``parse_clip_library`` refuses within a single file.
_user_clips: dict[str, dict[str, Any]] | None = None

#: ``template key -> str(exc)`` for a user clip library file that exists but
#: failed to parse, filled by the same read that builds ``_user_clips``. A
#: parse failure there still costs that template only -- ``_load_clip_library``
#: logs and skips it exactly as it always has, and the merged, rendering-facing
#: ``clip_library()`` still falls back to the shipped library for it. But
#: without this, ``service.clips.library()`` had no way to tell "no edits"
#: from "edits this build can no longer read" and presented the fallback as
#: the user's own, ``edited: True`` -- so the next Poser Save would silently
#: overwrite the very file that failed to parse. :func:`user_clip_error` is
#: the honest answer that door needs; nothing else reads this dict.
_user_clip_errors: dict[str, str] | None = None


# Mirrors service.clips.MAX_LIBRARY_KEYS/MAX_KEYS -- this module may import no
# more of ``warlock`` than ``winjob`` (this file's own docstring;
# ``tests/test_poser_imports.py`` pins it), so the write door's caps cannot be
# imported here and are restated as their own constants instead. The two must
# be kept in sync by hand: the 2026-09-11 audit (poser-04) found this parser
# applied neither, so a hand-edited library under user_clip_dir() (writable by
# any program, exactly like a pose file) with no count ceiling at all parsed
# in full where service.clips.save would have refused it at the write door.
#
# Raised from 256 to 1024 alongside service.clips.MAX_LIBRARY_KEYS when the
# clip-library schema moved to v3 (design decision D2): a library that can
# hold any number of named clips, not just the shipped five, wants more room
# to author key poses in without a hand-authored or agent-generated set
# hitting a ceiling picked for the original five-animation table.
# MAX_CLIP_LIBRARY_BYTES was checked against this: the humanoid library
# averages ~1.5 KB/pose at 19 bones and the largest authored skeleton (bird)
# carries 20, so 1024 poses lands around 1.6 MB -- comfortably inside the 4 MiB
# ceiling, which is left alone.
MAX_CLIP_LIBRARY_POSES = 1024
MAX_CLIP_KEYS = 64

#: The versions :func:`parse_clip_library` understands. 2 is the shipped shape
#: (no ``duration_ms``; timing lived in ``pipelines.charsheet.ANIMATIONS``
#: instead) and 3 adds it per clip -- see :data:`LEGACY_CLIP_DURATION_MS``.
#: A file naming any other version is refused the same way a malformed one
#: always was: ``ValueError`` out of the parser, caught and logged by
#: ``_load_clip_library`` for the shipped/user read paths and turned into an
#: ``Invalid`` by ``service.clips.save`` for the write door.
CLIP_LIBRARY_VERSIONS = (2, 3)

#: Per-rendered-frame duration, in whole milliseconds. The floor is a guard on
#: a typo (a clip cannot render faster than one frame a tick), the ceiling on
#: a clip so slow it is really a stall, and the step keeps every value evenly
#: divisible into ``warlock.clips.ANIMATION_FPS``'s 10 ms scene tick -- this
#: module cannot import ``warlock.clips`` (see ``CLIP_LIBRARY_VERSIONS`` above
#: on the import pin), so that relationship is only asserted, not imported;
#: ``tests/test_clip_library_v3.py::test_the_clip_duration_step_divides_the_animation_timebase``
#: pins ``1000 / clips.ANIMATION_FPS`` (10) against this constant so the two
#: cannot drift apart silently.
MIN_CLIP_DURATION_MS = 10
MAX_CLIP_DURATION_MS = 1000
CLIP_DURATION_STEP_MS = 10

#: A v2 clip carries no ``duration_ms`` of its own, so migrating one to v3
#: needs somewhere to read the time it was actually rendered at. Restated
#: from ``pipelines.charsheet.ANIMATIONS`` rather than imported, for the same
#: reason ``LEGACY_CLIP_DURATION_MS`` -- sorry, the same reason as above: this
#: module's outward ``warlock`` import is pinned to ``{winjob}`` and
#: ``charsheet`` is not in it.
#: ``tests/test_clip_library_v3.py`` (the legacy-frame-times test) imports
#: ``charsheet`` itself and pins this dict to ``ANIMATIONS``' own
#: ``(name, frames, loop, duration_ms)`` rows, so a change to one without the
#: other fails loudly instead of silently re-timing every migrated library.
LEGACY_CLIP_DURATION_MS: dict[str, int] = {
    "idle": 150,
    "walk": 100,
    "run": 60,
    "attack": 80,
    "jump": 100,
}

#: A clip named for one of Troupe's 16 facing directions -- or, worse, ending
#: in ``_<direction>`` -- collides with the tag Inker's importer already reads
#: off a filename: ``fall_back.png`` is read as clip ``fall`` facing ``back``.
#: Once any clip name can exist (not just the shipped five), authoring one
#: that *looks* like ``<clip>_<direction>`` is a trap the parser refuses
#: instead of leaving for Inker to silently mis-tag. Restated here from
#: ``pipelines.charsheet._DIRECTIONS_16`` for this module's import pin (see
#: ``LEGACY_CLIP_DURATION_MS`` above); pinned to it by ``tests/test_clip_library_v3.py``
#: (the restated-direction-keys test).
TROUPE_DIRECTION_KEYS: tuple[str, ...] = (
    "front",
    "front_front_left",
    "front_left",
    "left_front_left",
    "left",
    "left_back_left",
    "back_left",
    "back_back_left",
    "back",
    "back_back_right",
    "back_right",
    "right_back_right",
    "right",
    "right_front_right",
    "front_right",
    "front_front_right",
)

#: A clip's optional ``source`` is a small provenance note (e.g. ``{"file":
#: ..., "map": ..., "imported": ...}``) an importer or an agent can leave on a
#: clip it wrote, not free-form data -- so it is capped and type-checked like
#: every other field a clip carries, not merely round-tripped as opaque JSON.
MAX_CLIP_SOURCE_KEYS = 16
MAX_CLIP_SOURCE_BYTES = 2048


def validate_clip_duration_ms(value: Any, label: str) -> int:
    """One clip's ``duration_ms``, or raise. Shared by :func:`parse_clip_library`
    (the renderer's own check) and ``service.clips._check_shape`` (the write
    door's field-addressed one), so the two cannot quietly disagree about what
    a valid duration is -- the same argument ``validate_bones`` already makes
    for a pose's quaternions.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f'clip {label!r} "duration_ms" must be a whole number of milliseconds')
    if not (MIN_CLIP_DURATION_MS <= value <= MAX_CLIP_DURATION_MS):
        raise ValueError(
            f'clip {label!r} "duration_ms" must be {MIN_CLIP_DURATION_MS}-'
            f"{MAX_CLIP_DURATION_MS}, not {value}"
        )
    if value % CLIP_DURATION_STEP_MS != 0:
        raise ValueError(
            f'clip {label!r} "duration_ms" must be a multiple of '
            f"{CLIP_DURATION_STEP_MS}, not {value}"
        )
    return value


def reject_direction_named_clip(name: str) -> None:
    """Raise if *name* ends in ``_<direction>`` for one of Troupe's 16 facings.

    See :data:`TROUPE_DIRECTION_KEYS`. Suffix only -- a clip named exactly
    ``back`` is not read as anything facing anywhere, because there is no clip
    name before it for the tag parser to split off.
    """
    for direction in TROUPE_DIRECTION_KEYS:
        if name.endswith(f"_{direction}"):
            raise ValueError(
                f'clip {name!r} ends in "_{direction}"; Troupe reads a name '
                f'like that as clip {name[: -len(direction) - 1]!r} facing {direction!r}'
            )


def validate_clip_source(value: Any, label: str) -> dict[str, Any]:
    """A clip's optional ``source`` object, or raise. See its cap comment above."""
    if not isinstance(value, dict):
        raise ValueError(f'clip {label!r} "source" must be an object')
    if len(value) > MAX_CLIP_SOURCE_KEYS:
        raise ValueError(
            f'clip {label!r} "source" holds at most {MAX_CLIP_SOURCE_KEYS} fields'
        )
    out: dict[str, Any] = {}
    for k, v in value.items():
        if not isinstance(k, str):
            raise ValueError(f'clip {label!r} "source" keys must be strings')
        if isinstance(v, bool) or not isinstance(v, (str, int, float)):
            raise ValueError(f'clip {label!r} "source.{k}" must be a string or a number')
        out[k] = v
    if len(json.dumps(out)) > MAX_CLIP_SOURCE_BYTES:
        raise ValueError(f'clip {label!r} "source" is too large')
    return out


def parse_clip_library(raw: dict[str, Any]) -> dict[str, Any]:
    """One clip library file's contents, validated. Raises on a bad one.

    Split out of :func:`_load_clip_library` so the shipped library and a
    user-authored one go through exactly one parser: the editor writes files
    this reads back, and a second, laxer parse on the authoring side is how an
    editor comes to save something the renderer cannot open.

    v3 added per-clip ``duration_ms``: a v2 file (or one naming no version at
    all -- every file shipped before this parser existed) is migrated at read
    time from :data:`LEGACY_CLIP_DURATION_MS`, defaulting to 100 ms for a name
    that table does not carry. A v3 file states its own timing and a v3 clip
    that omits ``duration_ms`` is refused outright: unlike v2, there is no
    table left to fall back to once any clip name can exist.

    **The direction-suffix guard (** :func:`reject_direction_named_clip` **)
    applies to v3 files only.** It exists because once any clip name can
    exist, one that looks like ``<clip>_<direction>`` is a trap for Inker's
    tag parser -- but a v2 file predates that guard by definition, and a
    2026-09-13 fix found this parser applying it to v2 reads too: an existing,
    previously-legal user library naming a clip like ``turn_left`` was refused
    *whole* on read, logged and silently skipped, with rendering falling back
    to the shipped clips and the editor showing them marked as the user's own
    edited copy. A v2 file must read exactly as it always did; the save door
    (``service.clips._check_shape``, which always writes v3) is the one place
    such a name is still refused, because every save is a v3 write.
    """
    raw_version = raw.get("version")
    version = 2 if raw_version is None else int(raw_version)
    if version not in CLIP_LIBRARY_VERSIONS:
        raise ValueError(f"unknown clip library version {version}")
    raw_poses = raw["poses"]
    if len(raw_poses) > MAX_CLIP_LIBRARY_POSES:
        raise ValueError(
            f"a clip library holds at most {MAX_CLIP_LIBRARY_POSES} key poses, "
            f"not {len(raw_poses)}"
        )
    poses = {}
    pose_names = [str(pose["name"]) for pose in raw["poses"]]
    if len(set(pose_names)) != len(pose_names):
        raise ValueError("duplicate pose names")
    for pose in raw["poses"]:
        # **Through ``validate_pose``, like every other door a pose comes in
        # by.** A clip library is a file -- shipped, or authored in the editor
        # and written back -- and its quaternions were taken verbatim: a
        # 3-element list raised out of the middle of ``sheet._blend`` with no
        # mention of the file or the bone, and a NaN was interpolated into a
        # clip and written to disk. The bone names are *not* checked against a
        # rig here: a library is authored against a template and applied to
        # whatever is loaded, and the worker already reports an unknown bone
        # without failing the bake.
        row = {
            "name": str(pose["name"]),
            "bones": validate_bones(pose["bones"]),
        }
        if pose.get("root_translation"):
            # Function-level: ``poselib`` imports this module, so a top-level
            # import here would be circular. Shared rather than a bare
            # ``float(v)`` -- the 2026-09-07 audit (poser-05) found this door
            # (and ``service.clips._check_shape``) round-tripped
            # ``[nan, 1e30, 0.0]`` with no finite or magnitude check, unlike
            # the identical field on a library pose.
            from . import poselib

            row["root_translation"] = poselib.validate_root_translation(
                pose["root_translation"]
            )
        poses[row["name"]] = row
    # The frame every clip in this file is authored in. Per file and not per
    # clip: a library mixing the two would be one edit away from a clip that
    # silently means the other thing.
    space = str(raw.get("space") or "node")
    clips = []
    clip_names = [str(clip["name"]) for clip in raw["clips"]]
    if len(set(clip_names)) != len(clip_names):
        raise ValueError("duplicate clip names")
    for clip in raw["clips"]:
        name = str(clip["name"])
        if version >= 3:
            # v2-only libraries predate this guard (see the docstring above):
            # a name like ``turn_left`` was legal before any file could name a
            # clip beyond the shipped five, and a v2 file must keep reading
            # exactly as it always did. Every *save* still refuses it, because
            # ``service.clips.save`` always writes v3.
            reject_direction_named_clip(name)
        keys = [str(k) for k in clip["keys"]]
        if len(keys) > MAX_CLIP_KEYS:
            raise ValueError(
                f"clip {clip['name']!r} holds at most {MAX_CLIP_KEYS} keys, not {len(keys)}"
            )
        missing = [k for k in keys if k not in poses]
        if missing:
            raise ValueError(f"clip {clip['name']!r} names {missing}")
        if version >= 3:
            if "duration_ms" not in clip:
                raise ValueError(f'clip {name!r} needs "duration_ms" in a version 3 library')
            duration_ms = clip["duration_ms"]
        else:
            duration_ms = LEGACY_CLIP_DURATION_MS.get(name, 100)
        duration_ms = validate_clip_duration_ms(duration_ms, name)
        record: dict[str, Any] = {
            "name": name,
            "keys": keys,
            "segments": [int(n) for n in clip["segments"]],
            "closed": bool(clip.get("closed", False)),
            "easing": str(clip.get("easing") or "linear"),
            "space": space,
            "duration_ms": duration_ms,
        }
        if "provisional" in clip:
            provisional = clip["provisional"]
            if not isinstance(provisional, bool):
                raise ValueError(f'clip {name!r} "provisional" must be true or false')
            record["provisional"] = provisional
        if clip.get("source") is not None:
            record["source"] = validate_clip_source(clip["source"], name)
        clips.append(record)
    return {"poses": poses, "clips": clips, "space": space}


def _load_clip_library(
    directory: Path, *, errors: dict[str, str] | None = None
) -> dict[str, dict[str, Any]]:
    """``template key -> {"poses": {name: record}, "clips": [record]}``.

    A malformed library costs you that library and not the app, the rule
    ``_load_templates`` and ``_load_pose_library`` both follow -- and a clip
    naming a pose the file does not carry is exactly that: the file is
    internally inconsistent, and half-loading it would hand the renderer a clip
    with a hole in it.

    ``errors``, when given, is filled with ``path.stem -> str(exc)`` for every
    file this drops -- :func:`clip_library` passes its own ``_user_clip_errors``
    here so :func:`user_clip_error` can answer honestly without a second read
    of the same directory.
    """
    found: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            found[path.stem] = parse_clip_library(
                _read_json_capped(path, MAX_CLIP_LIBRARY_BYTES)
            )
        except Exception as exc:
            log.exception("skipping unusable clip library %s", path)
            if errors is not None:
                errors[path.stem] = str(exc)
    return found


#: Where an *edited* clip library lives, or None when nothing has said.
#:
#: **Told, never discovered.** This module is the one the host and the worker
#: share and it has deliberately never depended on the app's configuration --
#: ``tests/test_poser_imports.py`` pins its whole ``warlock`` import set to
#: ``{winjob}``, and reaching for ``config`` here (even inside a function body)
#: would be a real architectural change dressed up as a convenience. So the
#: layer that *has* a config sets this: ``service.core.WarlockService`` does it
#: on construction, which is the one object every app process builds before
#: anything asks for a clip.
#:
#: None means "the shipped library and nothing else", which is the correct
#: answer for a bare import, a test that never built a service, and the worker.
_user_clip_root: Path | None = None


def set_user_clip_dir(path: Any) -> None:
    """Point the loader at the user's editable clip libraries.

    Drops the caches, because the directory changing means every answer this
    module has already given about clips may now be the wrong one -- which is
    exactly what happens in a test suite that builds one service per test.
    """
    global _user_clip_root
    new = None if path is None else Path(path)
    if new != _user_clip_root:
        _user_clip_root = new
        invalidate_clips()


def user_clip_dir() -> Path | None:
    """Where an *edited* clip library lives, or None with no configured home.

    Beside the pose library under ``data_dir/poser/``, because it is the same
    kind of thing: authored content that belongs to the user rather than to the
    build. The shipped library under ``templates/clips/`` stays a factory
    default and is never written to -- which is what keeps an installed build
    (where the package tree may be read-only, and is replaced wholesale by the
    next installer) working the same way a checkout does.
    """
    return _user_clip_root


def invalidate_clips() -> None:
    """Drop the cached libraries so the next read sees a just-saved edit.

    The cache is a module global filled once, which is right for a library that
    ships with the build and wrong the moment one can be authored. Called by the
    save door rather than by a timestamp check: the writer knows.
    """
    global _clips, _user_clips, _user_clip_errors
    _clips = None
    _user_clips = None
    _user_clip_errors = None


def clip_library(template_key: str) -> dict[str, Any]:
    """The shipped clips for a template, or an empty library if it has none.

    Empty is a normal answer for the same reason ``deform_battery``'s is: a
    walk cycle authored for a humanoid means nothing to a fish, and a template
    without one should cost the character sheet, never the rig.
    """
    global _clips, _user_clips, _user_clip_errors
    get_template(template_key)
    if _clips is None:
        _clips = _load_clip_library(CLIP_DIR)
    if _user_clips is None:
        directory = user_clip_dir()
        _user_clip_errors = {}
        _user_clips = (
            _load_clip_library(directory, errors=_user_clip_errors)
            if directory is not None and directory.is_dir()
            else {}
        )
    # The user's file wins whole, never field by field -- see ``_user_clips``.
    # An unusable one has already been logged and dropped by the loader, so
    # this falls back to the shipped library rather than to nothing, which is
    # the same "costs you that library and not the app" rule one level up.
    # ``service.clips.library()`` is what tells the two cases apart for the
    # user, through :func:`user_clip_error` -- this function stays the
    # renderer's own "give me something to draw" door and keeps falling back.
    library = _user_clips.get(template_key) or _clips.get(template_key)
    return library or {"poses": {}, "clips": [], "space": "node"}


def user_clip_error(template_key: str) -> str | None:
    """The parse error for *template_key*'s user clip library file, or None.

    None both when the user has no file for this template and when the one
    they have parsed cleanly -- callers that need to tell those two apart
    already have ``poselib.clip_path(...).is_file()`` for the first half, the
    same test :func:`~warlock.service.clips.library`'s own ``edited`` flag
    uses. This is the answer for the other half: a file that exists but this
    build can no longer read, which used to make ``clip_library`` fall back to
    the shipped clips *silently* -- so the editor's ``edited`` flag lied,
    showing the shipped library marked as the user's own, and the next Poser
    Save would overwrite the very file that failed to parse.

    Populated by the same read :func:`clip_library` already does, so this
    never re-reads the directory and can never disagree with what that
    function actually served.
    """
    clip_library(template_key)  # ensures _user_clip_errors is populated
    return (_user_clip_errors or {}).get(template_key)


def shipped_clip_library(template_key: str) -> dict[str, Any]:
    """The template's *shipped* clips only -- never a user edit.

    :func:`clip_library` is the renderer's own door, and the user's file wins
    whole there by design (see ``_user_clips``' own comment): a hand-edited
    library is meant to change what a sheet renders the moment it is saved.
    An agent's movement vocabulary needs the opposite promise -- the catalogue
    it was told about at connect time must still be the catalogue a call
    against it means, for as long as the connection lasts -- so this reads
    straight out of the shipped ``_clips`` cache and never consults
    ``user_clip_dir()`` at all. Same empty-library fallback as
    :func:`clip_library`, and the same ``ValueError`` for an unknown template,
    both through :func:`get_template`.
    """
    global _clips
    get_template(template_key)
    # Bound to a local once and read only through it below: a concurrent
    # ``invalidate_clips()`` (Poser's save door, on another thread) between
    # the None check and the read used to be able to set the module global
    # back to None in the gap, and the second read raised
    # ``AttributeError: 'NoneType' object has no attribute 'get'`` instead of
    # simply serving the library this call had already committed to loading.
    clips = _clips
    if clips is None:
        clips = _load_clip_library(CLIP_DIR)
        _clips = clips
    library = clips.get(template_key)
    return library or {"poses": {}, "clips": [], "space": "node"}


def shipped_clip_templates() -> tuple[str, ...]:
    """Every template with at least one shipped clip, in :func:`catalog` order."""
    return tuple(
        row["key"] for row in catalog() if shipped_clip_library(row["key"])["clips"]
    )


def shipped_clip_names(template_key: str) -> tuple[str, ...]:
    """A template's shipped clip names, library order."""
    return tuple(str(c["name"]) for c in shipped_clip_library(template_key)["clips"])


def clip_keys(template_key: str, clip_name: str) -> list[dict[str, Any]]:
    """One clip's key poses, in order. Raises KeyError for an unknown clip."""
    library = clip_library(template_key)
    for clip in library["clips"]:
        if clip["name"] == clip_name:
            return [dict(library["poses"][k]) for k in clip["keys"]]
    raise KeyError(f"{clip_name!r} is not a clip of the {template_key} template")


# --- fitting ----------------------------------------------------------------

Vec3 = Sequence[float]


def fit_template(
    template: Template, bounds_min: Vec3, bounds_max: Vec3
) -> list[dict[str, Any]]:
    """Scale a template's normalized landmarks onto a mesh's bounding box.

    Normalized coordinates use Blender axes: x and y span -0.5..0.5 about the
    bbox centre, z spans 0 (bbox floor) to 1 (bbox ceiling). This is
    bbox-proportional, not learned, so it is approximate on unusual
    silhouettes -- the fitted positions are written into ``rig.json`` precisely
    so a later "adjust joints" pass can correct them without re-rigging from
    scratch.
    """
    size = [float(hi) - float(lo) for lo, hi in zip(bounds_min, bounds_max, strict=True)]
    largest = max(size) if max(size) > 0 else 1.0
    # A flat axis (a billboard-thin mesh, or a subject with no measurable depth)
    # would collapse every bone onto a plane and produce zero-length bones.
    # Borrowing the largest dimension keeps the skeleton three-dimensional; the
    # rig is approximate either way and a zero-length bone is not recoverable.
    size = [s if s > largest * 1e-4 else largest for s in size]
    centre = [(float(lo) + float(hi)) / 2.0 for lo, hi in zip(bounds_min, bounds_max, strict=True)]
    floor_z = float(bounds_min[2])
    min_len = largest * MIN_BONE_FRACTION

    def place(p: Vec3) -> list[float]:
        return [
            centre[0] + p[0] * size[0],
            centre[1] + p[1] * size[1],
            floor_z + p[2] * size[2],
        ]

    fitted: list[dict[str, Any]] = []
    for bone in template.bones:
        head = place(bone["head"])
        tail = place(bone["tail"])
        if _distance(head, tail) < min_len:
            tail = [head[0], head[1], head[2] + min_len]
        fitted.append({"name": bone["name"], "parent": bone["parent"], "head": head, "tail": tail})
    return fitted


def _distance(a: Vec3, b: Vec3) -> float:
    return sum((float(x) - float(y)) ** 2 for x, y in zip(a, b, strict=True)) ** 0.5


def validate_joints(
    payload: dict[str, Any],
    template: Template | Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize a corrected skeleton, or raise ValueError.

    The whole skeleton, not a patch: a partial correction would leave the caller
    and the worker disagreeing about which joints came from the fit and which
    from the user, and the fitted positions are already in rig.json for the
    editor to start from. Parentage comes from ``template`` and is never
    caller-supplied -- a client cannot restructure the hierarchy, only move it.

    ``template`` is usually a shipped :class:`Template`, but a *custom*
    skeleton (rig.json's ``skeleton == "custom"``) has no template to check
    against -- so this also accepts the bare structure, a sequence of
    ``{"name", "parent"}`` (extra keys ignored), which is what a rig's own
    ``bones`` list already is. Both shapes name every bone this payload must
    supply, in the order the output preserves.

    A zero-length bone is rejected rather than nudged: Blender silently deletes
    one on leaving edit mode and takes its children with it, so accepting it
    would produce a rig missing limbs with nothing to explain why.
    """
    structure = template.bones if isinstance(template, Template) else template
    raw = payload.get("bones")
    if not isinstance(raw, list) or not raw:
        raise ValueError("joints payload requires a non-empty 'bones' list")
    by_name: dict[str, dict[str, Any]] = {}
    for entry in raw:
        name = str(entry.get("name") or "")
        points: dict[str, list[float]] = {}
        for end in ("head", "tail"):
            point = entry.get(end)
            if not isinstance(point, (list, tuple)) or len(point) != 3:
                raise ValueError(f"bone {name!r} {end} must be a 3-vector")
            try:
                points[end] = [float(v) for v in point]
            except (TypeError, ValueError):
                raise ValueError(f"bone {name!r} {end} is not numeric") from None
            # NaN/inf sail through every comparison below -- the zero-length
            # check at the end is False for NaN, so a NaN joint would be
            # written into rig.json and only fail inside Blender.
            if not all(math.isfinite(v) for v in points[end]):
                raise ValueError(f"bone {name!r} {end} is not numeric")
        by_name[name] = points

    expected = [b["name"] for b in structure]
    missing = [n for n in expected if n not in by_name]
    if missing:
        raise ValueError(f"joints payload is missing bone(s): {missing}")
    unknown = [n for n in by_name if n not in expected]
    if unknown:
        raise ValueError(f"joints payload names unknown bone(s): {unknown}")

    # The skeleton's own extent, so the minimum bone length scales with the
    # mesh: MIN_BONE_FRACTION is a fraction of the subject, not of a metre.
    heads = [by_name[n]["head"] for n in expected]
    span = max(
        (hi - lo)
        for lo, hi in (
            (min(p[axis] for p in heads), max(p[axis] for p in heads)) for axis in range(3)
        )
    ) or 1.0
    out = []
    for bone in structure:
        entry = by_name[bone["name"]]
        if _distance(entry["head"], entry["tail"]) < span * MIN_BONE_FRACTION:
            raise ValueError(f"bone {bone['name']!r} would be zero-length")
        out.append(
            {
                "name": bone["name"],
                "parent": bone["parent"],
                "head": entry["head"],
                "tail": entry["tail"],
            }
        )
    return out


# --- skeleton editing ---------------------------------------------------------
#
# ``validate_joints`` only lets a caller move a template's own named joints.
# Everything below lets Poser's skeleton editor change the *structure* too:
# add or remove a pivot, split a bone, graft a limb preset -- the backend for
# a feature the UI is built on top of, per the plan this module was extended
# for on 2026-09-12. Two halves:
#
# * :func:`validate_skeleton` is the door a whole edited skeleton comes back
#   through, exactly once, before it is queued as a re-rig -- ``service.rig.
#   edit_skeleton``'s validator, ``validate_joints``'s shape for a skeleton
#   that may have a different shape than the template it started from.
# * The rest are pure functions over ``bones: list[dict]`` (``{name, parent,
#   head, tail}``) and never mutate their input -- an editor undo stack is
#   just the list of dicts each edit returned, and that only works if an
#   earlier one is never touched in place. Every one of them raises
#   :class:`RigError` naming a ``field`` on a bad edit, ``validate_skeleton``'s
#   convention, so a UI building an undo/redo stack on top of these gets the
#   same field-addressed refusals a first-time submit does.
#
# rig.json's ``skeleton`` field records which of the two shapes came out the
# other end: ``"template"`` when the edited skeleton still has exactly the
# base template's names and parents (a joint move, or ``adjusted: True`` from
# the older joints-only door), ``"custom"`` the moment either differs. A
# custom skeleton keeps ``rig.json["template"]`` naming the *base* template it
# started from -- ``clip_coverage`` and the pose library both need to know
# which template's poses/clips this rig might still play, even after its
# shape has diverged from it.


def _check_acyclic(by_name: Mapping[str, Mapping[str, Any]], *, field: str) -> None:
    """Raise :class:`RigError` if any bone's parent chain loops.

    ``_build_armature`` parents every bone in one pass with no cycle check of
    its own -- ``eb.parent = created[parent]`` -- so a cyclic parent list sent
    across the pipe would not fail until Blender's own edit-bone graph did
    something undefined with it. Walked from every bone, not just the
    apparent root(s): a cycle with no bone at all pointing to ``None`` would
    otherwise have no starting point this ever visits.
    """
    for start in by_name:
        seen: set[str] = set()
        cur: str | None = start
        while cur is not None:
            if cur in seen:
                raise RigError(f"the bone hierarchy has a cycle at {cur!r}", field=field)
            seen.add(cur)
            cur = by_name[cur]["parent"]


def _validate_mirror_pairs(
    raw: Any, names: Sequence[str]
) -> list[tuple[str, str]]:
    """A ``mirror_pairs`` list, checked against a skeleton's own bone names."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise RigError("mirror_pairs must be a list", field="mirror_pairs")
    name_set = set(names)
    seen: set[str] = set()
    pairs: list[tuple[str, str]] = []
    for pair in raw:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise RigError("each mirror pair must be [a, b]", field="mirror_pairs")
        a, b = str(pair[0]), str(pair[1])
        if a == b:
            raise RigError(f"a bone cannot mirror itself: {a!r}", field="mirror_pairs")
        if a not in name_set or b not in name_set:
            raise RigError(
                f"mirror pair names a bone this skeleton does not have: {(a, b)!r}",
                field="mirror_pairs",
            )
        if a in seen or b in seen:
            raise RigError(
                f"bone appears in more than one mirror pair: {(a, b)!r}", field="mirror_pairs"
            )
        seen.add(a)
        seen.add(b)
        pairs.append((a, b))
    return pairs


def _same_structure(by_name: Mapping[str, Mapping[str, Any]], base: Template) -> bool:
    """Whether an edited skeleton still has the base template's names and parents.

    Positions are deliberately not compared -- moving a joint is what
    ``adjusted: True`` already records, and it is not what turns a skeleton
    "custom".
    """
    base_names = {b["name"] for b in base.bones}
    if set(by_name) != base_names:
        return False
    base_parent = {b["name"]: b["parent"] for b in base.bones}
    return all(by_name[n]["parent"] == base_parent[n] for n in by_name)


def check_skeleton_structure(bones: Sequence[Mapping[str, Any]]) -> str:
    """Bare structural sanity for a caller-supplied bone list, returning its
    root's name or raising :class:`RigError`.

    Names are legal and unique, every parent resolves, there is exactly one
    root, and the parent graph is acyclic -- ``validate_skeleton``'s
    structural half only, with no position or bounds check. Its second caller
    is ``blender_worker._rig_bones``: a custom skeleton's structure was
    already checked once, host-side, by ``validate_skeleton`` before it was
    queued -- but the spec crosses a pipe as plain JSON to get to the worker,
    and re-trusting it there would mean a malformed spec (a bug in the queue
    row, a hand-edited one) builds an armature whose parents do not resolve
    instead of falling back to the bbox fit the way an untrusted
    ``template_bones`` list already does (see ``_rig_bones``).
    """
    names: list[str] = []
    by_parent: dict[str, str | None] = {}
    for entry in bones:
        if not isinstance(entry, Mapping):
            raise RigError("each bone must be an object", field="bones")
        name = str(entry.get("name") or "")
        if not BONE_NAME_RE.match(name):
            raise RigError(f"bone name {name!r} is not usable", field="bones")
        if name in by_parent:
            raise RigError(f"duplicate bone name {name!r}", field="bones")
        parent = entry.get("parent")
        by_parent[name] = None if parent is None else str(parent)
        names.append(name)
    for name, parent in by_parent.items():
        if parent is not None and parent not in by_parent:
            raise RigError(f"bone {name!r} has unknown parent {parent!r}", field="bones")
    roots = [n for n, p in by_parent.items() if p is None]
    if len(roots) != 1:
        raise RigError("a skeleton must have exactly one root bone", field="root")
    _check_acyclic({n: {"parent": p} for n, p in by_parent.items()}, field="bones")
    return roots[0]


def validate_skeleton(
    payload: dict[str, Any], *, base: Template, bounds: Mapping[str, Any]
) -> dict[str, Any]:
    """Normalize a whole edited skeleton, or raise :class:`RigError`.

    Returns ``{"bones", "root", "mirror_pairs", "skeleton"}`` -- everything
    ``service.rig.edit_skeleton`` needs to build a re-rig's params and
    everything the worker's ``_rig_meta`` needs to write into rig.json.
    ``base`` is the rig's *original* template (``rig.json["template"]``, even
    on a rig that is already custom), used only to decide ``skeleton``; it
    imposes no shape on the payload the way ``validate_joints``'s template
    argument does.

    ``bounds`` is the rig's own bounding box (rig.json's, Blender axes) --
    used to catch a gross unit mistake (a joint placed metres from the mesh
    when the mesh is centimetres across), not to constrain placement to the
    box: a deliberately extended limb (a tail, a reaching wing tip) is
    expected to reach outside it, so the box is expanded by its own diagonal
    before anything is checked against it.
    """
    raw = payload.get("bones")
    if not isinstance(raw, list) or not raw:
        raise RigError("a skeleton requires a non-empty 'bones' list", field="bones")
    if len(raw) > MAX_SKELETON_BONES:
        raise RigError(
            f"a skeleton may hold at most {MAX_SKELETON_BONES} bones, not {len(raw)}",
            field="bones",
        )

    names: list[str] = []
    by_name: dict[str, dict[str, Any]] = {}
    for entry in raw:
        if not isinstance(entry, Mapping):
            raise RigError("each bone must be an object", field="bones")
        name = str(entry.get("name") or "")
        if not BONE_NAME_RE.match(name):
            raise RigError(f"bone name {name!r} is not usable", field="bones")
        if name in by_name:
            raise RigError(f"duplicate bone name {name!r}", field="bones")
        parent = entry.get("parent")
        parent = None if parent is None else str(parent)
        points: dict[str, list[float]] = {}
        for end in ("head", "tail"):
            point = entry.get(end)
            if not isinstance(point, (list, tuple)) or len(point) != 3:
                raise RigError(f"bone {name!r} {end} must be a 3-vector", field="bones")
            try:
                values = [float(v) for v in point]
            except (TypeError, ValueError):
                raise RigError(f"bone {name!r} {end} is not numeric", field="bones") from None
            if not all(math.isfinite(v) for v in values):
                raise RigError(f"bone {name!r} {end} is not numeric", field="bones")
            points[end] = values
        by_name[name] = {"parent": parent, **points}
        names.append(name)

    for name, bone in by_name.items():
        if bone["parent"] is not None and bone["parent"] not in by_name:
            raise RigError(
                f"bone {name!r} has unknown parent {bone['parent']!r}", field="bones"
            )

    roots = [n for n, b in by_name.items() if b["parent"] is None]
    if len(roots) != 1:
        raise RigError("a skeleton must have exactly one root bone", field="root")
    root = roots[0]

    _check_acyclic(by_name, field="bones")

    # Checked against the rig's own bounds -- not the bones' own spread, which
    # a single stray joint would corrupt -- and *before* the zero-length check
    # below: a joint placed a continent away inflates every span computed from
    # the bone heads, which would otherwise flag ordinary short bones (a
    # spine, a finger) as zero-length before the real problem is ever named.
    lo = [float(v) for v in bounds["min"]]
    hi = [float(v) for v in bounds["max"]]
    margin = _distance(lo, hi) or 1.0
    box_lo = [lo[i] - margin for i in range(3)]
    box_hi = [hi[i] + margin for i in range(3)]
    for name, bone in by_name.items():
        head = bone["head"]
        if any(head[i] < box_lo[i] or head[i] > box_hi[i] for i in range(3)):
            raise RigError(f"bone {name!r} head is far outside the mesh", field="bones")

    span = max(
        (hi_ - lo_)
        for lo_, hi_ in (
            (
                min(b["head"][axis] for b in by_name.values()),
                max(b["head"][axis] for b in by_name.values()),
            )
            for axis in range(3)
        )
    ) or 1.0
    min_len = span * MIN_BONE_FRACTION
    for name, bone in by_name.items():
        if _distance(bone["head"], bone["tail"]) < min_len:
            raise RigError(f"bone {name!r} would be zero-length", field="bones")

    pairs = _validate_mirror_pairs(payload.get("mirror_pairs"), names)
    skeleton = "template" if _same_structure(by_name, base) else "custom"

    bones = [
        {
            "name": n,
            "parent": by_name[n]["parent"],
            "head": by_name[n]["head"],
            "tail": by_name[n]["tail"],
        }
        for n in names
    ]
    return {"bones": bones, "root": root, "mirror_pairs": pairs, "skeleton": skeleton}


def add_bone(
    bones: list[dict[str, Any]], parent: str | None, name: str, head: Vec3, tail: Vec3
) -> list[dict[str, Any]]:
    """A new list with one bone appended. Raises :class:`RigError`."""
    names = {b["name"] for b in bones}
    if not BONE_NAME_RE.match(name):
        raise RigError(f"bone name {name!r} is not usable", field="name")
    if name in names:
        raise RigError(f"duplicate bone name {name!r}", field="name")
    if parent is not None and parent not in names:
        raise RigError(f"unknown parent {parent!r}", field="parent")
    new_bone = {
        "name": name,
        "parent": parent,
        "head": [float(v) for v in head],
        "tail": [float(v) for v in tail],
    }
    return [dict(b) for b in bones] + [new_bone]


def split_bone(
    bones: list[dict[str, Any]], name: str, new_name: str
) -> list[dict[str, Any]]:
    """Cut ``name`` at its midpoint, inserting ``new_name`` as its new tail half.

    ``name`` keeps its head and gains the midpoint as its tail; ``new_name`` is
    a new child of ``name`` running from the midpoint to ``name``'s old tail;
    every bone that used to parent off ``name`` now parents off ``new_name``,
    since that is the half of the original bone their heads used to coincide
    with.
    """
    by_name = {b["name"]: b for b in bones}
    if name not in by_name:
        raise RigError(f"unknown bone {name!r}", field="name")
    if not BONE_NAME_RE.match(new_name):
        raise RigError(f"bone name {new_name!r} is not usable", field="new_name")
    if new_name in by_name:
        raise RigError(f"duplicate bone name {new_name!r}", field="new_name")
    bone = by_name[name]
    mid = [(h + t) / 2.0 for h, t in zip(bone["head"], bone["tail"], strict=True)]
    out = []
    for b in bones:
        nb = dict(b)
        if nb["name"] == name:
            nb["tail"] = list(mid)
        elif nb["parent"] == name:
            nb["parent"] = new_name
        out.append(nb)
    out.append({"name": new_name, "parent": name, "head": list(mid), "tail": list(bone["tail"])})
    return out


def remove_pivot(bones: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    """Remove one bone, reparenting its children to its own parent.

    Children keep their world-space heads/tails exactly as they are -- only
    ``parent`` changes -- so this is a hierarchy edit, not a placement one.
    Removing the root is refused unless it has exactly one child, in which
    case that child is promoted to root; a root with more than one child has
    no single bone to inherit the role, and a root with none would leave an
    empty skeleton.
    """
    by_name = {b["name"]: b for b in bones}
    if name not in by_name:
        raise RigError(f"unknown bone {name!r}", field="name")
    target = by_name[name]
    children = [b["name"] for b in bones if b["parent"] == name]
    if target["parent"] is None:
        if len(children) != 1:
            raise RigError(
                "the root can only be removed when it has exactly one child",
                field="root",
            )
        only_child = children[0]
        return [
            dict(b, parent=None) if b["name"] == only_child else dict(b)
            for b in bones
            if b["name"] != name
        ]
    new_parent = target["parent"]
    return [
        dict(b, parent=new_parent) if b["parent"] == name else dict(b)
        for b in bones
        if b["name"] != name
    ]


def remove_subtree(bones: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    """Remove ``name`` and everything beneath it. Refuses the root."""
    by_name = {b["name"]: b for b in bones}
    if name not in by_name:
        raise RigError(f"unknown bone {name!r}", field="name")
    if by_name[name]["parent"] is None:
        raise RigError("the root cannot be removed as a subtree", field="root")
    doomed = {name}
    changed = True
    while changed:
        changed = False
        for b in bones:
            if b["parent"] in doomed and b["name"] not in doomed:
                doomed.add(b["name"])
                changed = True
    return [dict(b) for b in bones if b["name"] not in doomed]


def rename_bone(
    bones: list[dict[str, Any]], pairs: Sequence[Sequence[str]], old: str, new: str
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """Rename one bone everywhere it is named: its own entry, every child's
    ``parent``, and either side of a mirror pair."""
    names = {b["name"] for b in bones}
    if old not in names:
        raise RigError(f"unknown bone {old!r}", field="old")
    if not BONE_NAME_RE.match(new):
        raise RigError(f"bone name {new!r} is not usable", field="new")
    if new != old and new in names:
        raise RigError(f"duplicate bone name {new!r}", field="new")
    out_bones = []
    for b in bones:
        nb = dict(b)
        if nb["name"] == old:
            nb["name"] = new
        if nb["parent"] == old:
            nb["parent"] = new
        out_bones.append(nb)
    out_pairs = [(new if a == old else a, new if b == old else b) for a, b in pairs]
    return out_bones, out_pairs


def prune_pairs(
    bones: Sequence[Mapping[str, Any]], pairs: Sequence[Sequence[str]]
) -> list[tuple[str, str]]:
    """Drop any mirror pair naming a bone that is no longer in ``bones``.

    The cleanup every removal above owes its caller: ``remove_pivot``/
    ``remove_subtree`` do not touch ``mirror_pairs`` themselves (they only see
    ``bones``), so a caller that keeps a pairs list beside its bones list runs
    this after any removal, the same way ``validate_skeleton`` already refuses
    a pair naming an unknown bone on the way in.
    """
    names = {b["name"] for b in bones}
    return [(a, b) for a, b in pairs if a in names and b in names]


def mirror_partner_name(name: str) -> str | None:
    """The ``.L``/``.R`` counterpart of a bone name, or ``None`` if it has none."""
    if name.endswith(".L"):
        return name[:-2] + ".R"
    if name.endswith(".R"):
        return name[:-2] + ".L"
    return None


def unique_name(bones: Sequence[Mapping[str, Any]], stem: str) -> str:
    """``stem``, or ``stem.2``, ``stem.3``, ... -- the first one not already used."""
    names = {b["name"] for b in bones}
    if stem not in names:
        return stem
    i = 2
    while f"{stem}.{i}" in names:
        i += 1
    return f"{stem}.{i}"


def _limb_frame(
    target: Mapping[str, Any], side: str | None
) -> tuple[list[float], list[float], list[float], list[float], float]:
    """The local frame a limb preset is authored in, anchored on ``target``.

    Returns ``(origin, e_f, e_s, e_u, length)``: ``origin`` is ``target``'s
    tail; ``e_f`` continues ``target``'s own head->tail direction; ``e_u`` is
    world +Z orthogonalized against ``e_f`` (world +Y instead, when the two are
    nearly parallel, so a target bone that already points up does not collapse
    the frame); ``e_s`` is ``e_f x e_u``, negated for ``side="R"`` so a preset
    written once mirrors by construction. ``length`` is ``target``'s own bone
    length, which is the unit every preset coordinate is written in.
    """
    head, tail = target["head"], target["tail"]
    length = _distance(head, tail)
    if length <= 0:
        length = 1.0
    f = [(t - h) / length for h, t in zip(head, tail, strict=True)]
    up = [0.0, 0.0, 1.0]
    dot = sum(a * b for a, b in zip(up, f, strict=True))
    if abs(dot) > 0.999:
        up = [0.0, 1.0, 0.0]
        dot = sum(a * b for a, b in zip(up, f, strict=True))
    u = [up[i] - dot * f[i] for i in range(3)]
    ulen = sum(v * v for v in u) ** 0.5 or 1.0
    u = [v / ulen for v in u]
    s = [
        f[1] * u[2] - f[2] * u[1],
        f[2] * u[0] - f[0] * u[2],
        f[0] * u[1] - f[1] * u[0],
    ]
    if side == "R":
        s = [-v for v in s]
    return list(tail), f, s, u, length


def _limb_point(
    origin: Sequence[float],
    e_f: Sequence[float],
    e_s: Sequence[float],
    e_u: Sequence[float],
    length: float,
    local: Sequence[float],
) -> list[float]:
    x, y, z = (float(v) for v in local)
    return [
        origin[i] + length * (x * e_s[i] + y * e_f[i] + z * e_u[i]) for i in range(3)
    ]


def attach_limb(
    bones: list[dict[str, Any]],
    pairs: Sequence[Sequence[str]],
    preset: str,
    parent: str,
    side: str | None,
    mirror: bool,
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """Graft a limb preset (``templates/limbs/<preset>.json``) onto ``parent``.

    ``side`` is ``"L"``, ``"R"`` or ``None`` (a centre limb: a tail, a spine
    continuation); the grafted bones are named ``f"{preset_bone}.{side}"``, or
    the bare preset name for a centre limb, deduplicated with
    :func:`unique_name` against every bone already in play. ``mirror=True``
    attaches both sides at once from the one preset and adds a mirror pair for
    each grafted bone; it is refused for ``side=None`` (a centre limb) since
    there is no second side to put a copy on. See :func:`_limb_frame` for the
    coordinate frame every preset is authored in.
    """
    if side is not None and side not in ("L", "R"):
        raise RigError(f"side must be 'L', 'R' or None, not {side!r}", field="side")
    if mirror and side is None:
        raise RigError("a centre limb has no side to mirror", field="side")
    preset_data = get_limb_preset(preset)
    by_name = {b["name"]: b for b in bones}
    if parent not in by_name:
        raise RigError(f"unknown parent {parent!r}", field="parent")

    def _attach_one(
        one_side: str | None, existing: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        origin, e_f, e_s, e_u, length = _limb_frame(by_name[parent], one_side)
        name_map: dict[str, str] = {}
        added: list[dict[str, Any]] = []
        for b in preset_data["bones"]:
            stem = f"{b['name']}.{one_side}" if one_side else b["name"]
            final = unique_name(existing + added, stem)
            name_map[b["name"]] = final
            real_parent = parent if b["parent"] is None else name_map[b["parent"]]
            added.append(
                {
                    "name": final,
                    "parent": real_parent,
                    "head": _limb_point(origin, e_f, e_s, e_u, length, b["head"]),
                    "tail": _limb_point(origin, e_f, e_s, e_u, length, b["tail"]),
                }
            )
        return added, name_map

    out_bones = [dict(b) for b in bones]
    out_pairs = [(str(a), str(b)) for a, b in pairs]
    if not mirror:
        added, _ = _attach_one(side, out_bones)
        return out_bones + added, out_pairs
    added_l, map_l = _attach_one("L", out_bones)
    added_r, map_r = _attach_one("R", out_bones + added_l)
    out_bones = out_bones + added_l + added_r
    out_pairs = out_pairs + [(map_l[base], map_r[base]) for base in map_l]
    return out_bones, out_pairs


def clip_coverage(rig: Mapping[str, Any], template_key: str) -> list[str]:
    """Bone names ``template_key``'s clip library poses that ``rig`` lacks.

    What a UI needs before it lets an edited skeleton play a clip authored for
    its base template: a clip keys some subset of the template's bones, and a
    custom skeleton may have renamed, removed, or never had one of them. Empty
    means every bone any clip pose animates survives, under its original name,
    in this rig -- whatever else about its shape changed.
    """
    library = clip_library(template_key)
    animated: set[str] = set()
    for pose in library["poses"].values():
        animated.update(pose["bones"])
    have = {b["name"] for b in rig.get("bones", []) if isinstance(b, Mapping)}
    return sorted(animated - have)


# --- limb presets -------------------------------------------------------------

LIMB_DIR = TEMPLATE_DIR / "limbs"

_limb_presets: dict[str, dict[str, Any]] | None = None


def _parse_limb_preset(raw: dict[str, Any]) -> dict[str, Any]:
    key = str(raw["key"])
    if not TEMPLATE_KEY_RE.match(key):
        raise ValueError(f"limb preset key {key!r} is not a safe path component")
    bones = raw["bones"]
    if not bones:
        raise ValueError("a limb preset needs at least one bone")
    names = {b["name"] for b in bones}
    if len(names) != len(bones):
        raise ValueError("duplicate bone names")
    for b in bones:
        if b["parent"] is not None and b["parent"] not in names:
            raise ValueError(f"bone {b['name']!r} has unknown parent {b['parent']!r}")
        for end in ("head", "tail"):
            if len(b[end]) != 3:
                raise ValueError(f"bone {b['name']!r} {end} is not a 3-vector")
    roots = [b for b in bones if b["parent"] is None]
    if len(roots) != 1:
        raise ValueError("a limb preset needs exactly one bone with parent None")
    return {
        "key": key,
        "label": str(raw["label"]),
        "bones": [
            {
                "name": str(b["name"]),
                "parent": None if b["parent"] is None else str(b["parent"]),
                "head": [float(v) for v in b["head"]],
                "tail": [float(v) for v in b["tail"]],
            }
            for b in bones
        ],
    }


def _load_limb_presets() -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    # Non-recursive, like ``_load_templates``'s own ``glob("*.json")`` on
    # ``TEMPLATE_DIR`` -- ``LIMB_DIR`` is a subdirectory of it precisely so a
    # preset never collides with, or is mistaken for, a skeleton template.
    for path in sorted(LIMB_DIR.glob("*.json")):
        try:
            raw = _read_json_capped(path, MAX_TEMPLATE_BYTES)
            preset = _parse_limb_preset(raw)
            if preset["key"] != path.stem:
                raise ValueError(
                    f"limb preset key {preset['key']!r} does not match filename {path.name}"
                )
            found[preset["key"]] = preset
        except Exception:
            # A malformed preset must cost you that preset, not the app --
            # ``_load_templates``'s rule, restated for the same reason.
            log.exception("skipping unusable limb preset %s", path)
    return found


def limb_presets() -> dict[str, dict[str, Any]]:
    global _limb_presets
    if _limb_presets is None:
        _limb_presets = _load_limb_presets()
    return _limb_presets


def get_limb_preset(key: str) -> dict[str, Any]:
    try:
        return limb_presets()[key]
    except KeyError:
        raise ValueError(f"unknown limb preset {key!r}") from None


# --- pose payloads ----------------------------------------------------------

MAX_POSE_NAME = 64
QUAT_EPSILON = 1e-3


def validate_pose(
    payload: dict[str, Any], known_bones: Sequence[str] | None = None
) -> dict[str, Any]:
    """Normalize a client pose payload, or raise ValueError.

    A pose is a bone -> local rotation quaternion map. Quaternions are stored
    XYZW, matching three.js's ``Quaternion.toArray()``, because the browser is
    the only thing that ever produces one; the Blender worker converts to
    WXYZ on the way in.
    """
    name = str(payload.get("name") or "").strip()
    if not name:
        raise ValueError("pose requires a name")
    if len(name) > MAX_POSE_NAME:
        raise ValueError(f"pose name must be at most {MAX_POSE_NAME} characters")

    raw = payload.get("bones")
    if not isinstance(raw, dict) or not raw:
        raise ValueError("pose requires a non-empty 'bones' map")
    return {"name": name, "bones": validate_bones(raw, known_bones)}


def validate_bones(
    raw: Any, known_bones: Sequence[str] | None = None
) -> dict[str, list[float]]:
    """One bone -> quaternion map, checked and renormalised, or ValueError.

    Split out of :func:`validate_pose` so a *clip library* can use it: a clip's
    key poses are read from a file like any other pose and their quaternions
    were taken verbatim, so a 3-element list raised out of the middle of
    ``sheet._blend`` naming neither the file nor the bone, and a NaN was
    interpolated into a clip and written to disk.

    **An empty map is legal here and is not in ``validate_pose``**, which is
    the whole reason for the split rather than a flag: a saved pose with no
    bones is a pose that says nothing and is a mistake, while a clip's rest key
    is exactly that map and is the ordinary case.
    """
    if not isinstance(raw, dict):
        raise ValueError("a bone map must be an object")
    allowed = set(known_bones) if known_bones is not None else None

    bones: dict[str, list[float]] = {}
    for bone, quat in raw.items():
        if allowed is not None and bone not in allowed:
            raise ValueError(f"unknown bone {bone!r}")
        if not isinstance(quat, (list, tuple)) or len(quat) != 4:
            raise ValueError(f"bone {bone!r} rotation must be a 4-element quaternion")
        try:
            values = [float(v) for v in quat]
        except (TypeError, ValueError):
            raise ValueError(f"bone {bone!r} rotation is not numeric") from None
        # Before the norm check, which a NaN would pass: abs(nan - 1.0) > eps
        # is False, so the raw NaN quaternion used to be stored verbatim.
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"bone {bone!r} rotation is not numeric")
        norm = sum(v * v for v in values) ** 0.5
        if abs(norm - 1.0) > QUAT_EPSILON:
            # Renormalize rather than reject: a browser accumulating gizmo
            # drags drifts off the unit sphere by ~1e-7 per drag, and refusing
            # the save over float noise would be indefensible. Only a
            # degenerate zero quaternion is unrecoverable.
            if norm < QUAT_EPSILON:
                raise ValueError(f"bone {bone!r} rotation is degenerate")
            values = [v / norm for v in values]
        bones[bone] = values
    return bones


# --- the two rotation frames --------------------------------------------------
#
# ``node`` is parent-relative and absolute -- what ``Model.set_rotation`` writes
# and what the pose editor is unconditionally in. ``delta`` is a rotation from
# the bone's own rest, which is what a clip library is authored in. The two are
# related by ``node = rest * delta``, and **that one sentence is the whole of
# the conversion** -- which is why it lives here rather than being written out
# at each boundary. It was written twice: once in ``studio/poser_mode`` against
# the viewer's rest quaternions and once in ``pipelines/blender_worker``
# against Blender's, and neither knew about the other. The multiply is one
# line; the *order* and which side is conjugated are the parts that drift, and
# a drifted one contorts a skeleton in a way nothing raises about.
#
# Quaternions here are XYZW throughout, this package's order everywhere.


def _quat_mul(a: Sequence[float], b: Sequence[float]) -> list[float]:
    ax, ay, az, aw = (float(v) for v in a)
    bx, by, bz, bw = (float(v) for v in b)
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def node_from_delta(rest: Sequence[float], delta: Sequence[float]) -> list[float]:
    """A rotation from rest, as a parent-relative absolute one."""

    return _quat_mul(rest, delta)


def delta_from_node(rest: Sequence[float], node: Sequence[float]) -> list[float]:
    """A parent-relative absolute rotation, as one from rest."""

    x, y, z, w = (float(v) for v in rest)
    return _quat_mul([-x, -y, -z, w], node)


def mirror_quaternion(q: Sequence[float]) -> list[float]:
    """Reflect a local joint rotation across the subject's YZ plane.

    Every template is symmetric about X (the mirror normal), and reflecting a
    rotation conjugates it: the components perpendicular to the normal flip
    sign and the one along it does not. So (x, y, z, w) -> (x, -y, -z, w).

    Lives here rather than only in the browser because it is the kind of sign
    convention that is wrong in a way you cannot see -- a mirrored arm that
    rotates the wrong way about one axis still looks plausible in a static
    pose. The JS copy in app.js must stay identical to this.
    """
    x, y, z, w = (float(v) for v in q)
    return [x, -y, -z, w]


def mirror_pose(
    bones: dict[str, Any], pairs: Sequence[Sequence[str]]
) -> dict[str, list[float]]:
    """Copy every posed bone onto its mirror partner, reflected.

    Bones with no partner (a spine, a tail) are left exactly as they are: they
    sit on the mirror plane, so reflecting them would rotate a centred limb off
    centre.
    """
    partner: dict[str, str] = {}
    for a, b in pairs:
        partner[a] = b
        partner[b] = a
    out = {name: [float(v) for v in q] for name, q in bones.items()}
    for name, quat in bones.items():
        other = partner.get(name)
        if other is not None:
            out[other] = mirror_quaternion(quat)
    return out


# --- the Blender subprocess -------------------------------------------------


class BlenderError(RuntimeError):
    """The worker exited non-zero, timed out, or produced no result file."""


def _terminate_worker(proc: subprocess.Popen[str], timeout: float = 10.0) -> None:
    """Kill and reap a Blender child without ever introducing another hang."""
    with contextlib.suppress(Exception):
        proc.kill()
    with contextlib.suppress(Exception):
        proc.wait(timeout=timeout)


def run_worker(
    spec: dict[str, Any],
    *,
    on_progress: Callable[[float, str], None] | None = None,
    on_start: Callable[[subprocess.Popen[str]], None] | None = None,
    timeout: float = BLENDER_TIMEOUT,
    module: str = "warlock.pipelines.blender_worker",
    marker: str = "blender",
    name: str = "Blender worker",
) -> dict[str, Any]:
    """Run one Blender operation out-of-process and return its result JSON.

    ``module``/``marker``/``name`` generalise the contract to any child that
    speaks it -- the LoRA trainer (``pipelines.lora_train_worker``) is the
    second -- without a second copy of the deadline, the drain threads and
    the staged-result rules below. The defaults keep every Blender caller
    exactly what it was.

    Synchronous and blocking -- every caller in the app dispatches it through
    ``asyncio.to_thread``, like every other multi-second call in this codebase.

    ``spec`` is handed over on stdin; the worker writes its result to
    ``spec["result_path"]`` rather than stdout, so a stray print from bpy (and
    it does print) can never corrupt the payload.

    ``on_start`` receives the live ``Popen`` so a cancel can kill it. Like
    trellis-server, there is no polite abort: bpy is inside a C weighting solve
    and checks nothing, so killing the process is the only thing that stops it.
    """
    result_path = Path(spec["result_path"])
    # The worker stages its result beside the served name and renames it in, so
    # the same three cleanup sites that clear a stale result have to clear the
    # staging file too -- a worker killed mid-write leaves the .tmp, not the
    # result, and it would otherwise sit in the source job's directory forever.
    result_tmp = result_path.with_name(result_path.name + ".tmp")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.unlink(missing_ok=True)
    result_tmp.unlink(missing_ok=True)

    re_progress = (
        RE_PROGRESS
        if marker == "blender"
        else re.compile(rf"^\[{re.escape(marker)}\]\s+([\d.]+)\s+(.*)$")
    )
    proc = subprocess.Popen(
        [sys.executable, "-m", module],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    # Same kill-on-close job as trellis-server: a bpy solve holds multiple GB,
    # and a parent that dies mid-rig used to leave it running indefinitely.
    winjob.assign(proc.pid)
    winjob.track(proc.pid, name.lower())
    tail: list[str] = []
    if on_start is not None:
        try:
            on_start(proc)
        except BaseException:
            _terminate_worker(proc)
            winjob.untrack(proc.pid)
            raise
    # stdout is drained on a helper thread so the *whole* run has a deadline, not
    # just the wait() after EOF: a bpy process that hangs mid-solve (or mid-render)
    # produces no further output and never closes stdout, and reading it inline
    # would block past any timeout and wedge the serial job queue forever.
    lines: queue.Queue[str | None] = queue.Queue()

    def _pump(stream: Any) -> None:
        try:
            for raw in stream:
                lines.put(raw)
        finally:
            lines.put(None)

    assert proc.stdin is not None and proc.stdout is not None
    reader = threading.Thread(target=_pump, args=(proc.stdout,), daemon=True)
    reader.start()

    # The spec goes out on a thread of its own for the same reason stdout is
    # drained on one: it has to be inside the deadline. A sheet spec carries
    # every cell's bones inline, and a 200-cell clip is far larger than the OS
    # pipe buffer -- so a worker that dies before draining stdin (a failed
    # `import bpy`, a driver that will not load) made this write block
    # indefinitely, wedging the serial GPU queue well past `timeout`. The
    # BrokenPipeError the other outcome raises is swallowed here too: the exit
    # code and the captured tail below say far more about what went wrong.
    def _send() -> None:
        try:
            proc.stdin.write(json.dumps(spec))
            proc.stdin.close()
        except OSError:
            pass

    writer = threading.Thread(target=_send, daemon=True)
    writer.start()

    deadline = time.monotonic() + timeout
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                # The deadline can elapse in the same queue-poll tick the
                # child's stdout closes -- drain whatever is already queued
                # (the EOF sentinel, or a final line) before calling this a
                # timeout, or a job that legitimately finished gets its
                # result deleted and a spurious BlenderError raised.
                try:
                    raw = lines.get_nowait()
                except queue.Empty:
                    raise subprocess.TimeoutExpired(proc.args, timeout) from None
            else:
                try:
                    raw = lines.get(timeout=min(remaining, 1.0))
                except queue.Empty:
                    continue
            if raw is None:
                break
            line = raw.rstrip()
            m = re_progress.match(line)
            if m and on_progress is not None:
                on_progress(float(m.group(1)), m.group(2))
            tail.append(line)
            if len(tail) > 200:
                del tail[:100]
        # A small floor on the final wait: the worker has already signalled
        # done (EOF/sentinel seen above), but its exit may still be
        # finalising -- deadline - now can be ~0 here and give the OS no
        # grace to reap it, re-raising the very timeout we just avoided.
        code = proc.wait(timeout=max(deadline - time.monotonic(), 1.0))
    except subprocess.TimeoutExpired:
        _terminate_worker(proc)
        result_path.unlink(missing_ok=True)
        result_tmp.unlink(missing_ok=True)
        raise BlenderError(f"{name} timed out after {timeout:.0f}s") from None
    finally:
        if proc.poll() is None:
            _terminate_worker(proc)
        # Every exit path leaves the child dead or killed, so the registry
        # entry comes out with it -- winjob's contract is that entries are
        # removed on reap, or terminate_tracked later opens a recycled pid
        # with PROCESS_TERMINATE and can kill an unrelated process.
        winjob.untrack(proc.pid)

    output = "\n".join(tail)[-ERROR_TAIL_CHARS:]
    if code != 0:
        # A killed-late worker may still have written the handoff file; it
        # would otherwise sit in the source job's directory until the next run.
        result_path.unlink(missing_ok=True)
        result_tmp.unlink(missing_ok=True)
        raise BlenderError(f"{name} exited with code {code}:\n{output}")
    if not result_path.exists():
        raise BlenderError(f"{name} wrote no result:\n{output}")
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        # An exit 0 with an unreadable result is the one way this could still
        # raise a raw decoder error at the caller. Typed like every other way
        # the worker can disappoint, and carrying the worker's own tail.
        result_path.unlink(missing_ok=True)
        raise BlenderError(f"{name} wrote an unreadable result:\n{output}") from exc
    # The handoff file has served its purpose; a rig writes it into the source
    # job's directory, where it would otherwise sit next to model.glb forever.
    result_path.unlink(missing_ok=True)
    result_tmp.unlink(missing_ok=True)
    return payload


# The worker writes to these temp names; the queue renames them onto the
# served names on success (finalize_rig). Two reasons, both load-bearing:
# Blender's GLB export writes in place over seconds, and rig.json -- the
# completion gate _attach_files and the file route key on -- is already
# satisfied by an earlier rig during a re-rig, so pointing the worker at the
# served names would let a concurrent GET read a truncated rig.glb. It also
# makes cancellation naturally non-destructive: discarding a half-written
# re-rig deletes the temps, never the previous successful rig's artifacts.
# The suffixes are load-bearing: Blender's glTF exporter appends ".glb" to a
# filepath that does not already end in it, so a ".tmp" suffix would make the
# worker write ".rig.tmp.glb.glb" and finalize_rig find nothing.
RIG_GLB_TMP = ".rig.tmp.glb"
RIG_JSON_TMP = ".rig.tmp.json"

# Where a re-texture assembles its replacement mesh before it becomes the one
# being served. Named here beside the rig temps because it is the same rule for
# the same reason -- a job writing into a *different* job's directory publishes
# by rename, so a cancel deletes a temp and never a finished artifact -- and
# because both the worker that writes it and ``_discard_artifacts`` need the
# one spelling. Nothing goes near Blender's exporter here, so the suffix is
# free; it keeps the rig temps' shape anyway.
RETEXTURE_GLB_TMP = ".retexture.tmp.glb"


#: Where a rig's joints come from when the user has not moved them by hand.
#: ``template`` is the bbox-proportional fit every rig used before this existed
#: and is still the default; ``measured`` reads them off the mesh.
JOINT_SOURCES = ("template", "measured")


def rig_spec(
    job_dir: Path,
    template_key: str,
    bones: list[dict[str, Any]] | None = None,
    *,
    template_bones: list[dict[str, Any]] | None = None,
    fit: dict[str, Any] | None = None,
    joints: str | None = None,
    skeleton: str | None = None,
    root: str | None = None,
    mirror_pairs: list[Sequence[str]] | None = None,
) -> dict[str, Any]:
    """The worker spec for rigging a finished job's mesh.

    Three sources of joints, in a fixed order of preference, and the order is
    the design:

    * ``bones`` -- world-space joints the *user* moved, from the adjust-joints
      pass. Already validated against the template by ``validate_joints``, and
      they win over everything: a correction is the last word by definition.
    * ``template_bones`` -- the template's own landmarks, replaced with ones
      measured off the reference image (``pipelines.pose2d``). Still normalized
      and still the template's shape, so the worker fits them onto the mesh
      bbox with exactly the ``fit_template`` it uses for the shipped ones --
      the scaling stays owned by the worker and this stays a *better template*
      rather than a second fitter.
    * nothing -- the shipped template, scaled bbox-proportionally, which is
      what every rig did before landmarks existed and is still right for a
      reference that really is standing in a T-pose.

    ``fit`` is what the host knows about how the second of those was found and
    the worker cannot: which model, how confident. It is recorded in rig.json
    and read by nothing that has to work, which is why it is a free-form dict.

    ``joints="measured"`` asks the worker to take ``pipelines.jointfit`` to the
    mesh's own vertices and use the result as if the user had corrected the
    joints by hand. It sits *below* ``bones`` in the same order of preference --
    a real correction is still the last word -- and above the template, because
    a measurement of the mesh in front of you beats a template scaled to its
    bounding box whenever the two disagree. The two supplied base meshes are
    exactly where they disagree: the shipped humanoid template is an A-pose and
    a T-pose mesh fits it badly.

    ``skeleton``/``root``/``mirror_pairs`` carry a *custom* skeleton's own
    structure across the pipe -- ``service.rig.edit_skeleton``'s
    ``validate_skeleton`` output -- so the worker can build the armature from
    the caller's own parents instead of the template's. All three are written
    only when given, which is what keeps a spec built for an ordinary template
    rig, or for an old ``adjust_joints`` joint move, byte-identical to what it
    always was: the pin in ``tests/test_rigging.py`` for exactly that.
    """
    get_template(template_key)  # fail here, not three seconds into a subprocess
    spec = {
        "op": "rig",
        "source_glb": str(job_dir / "model.glb"),
        "out_glb": str(job_dir / RIG_GLB_TMP),
        "out_json": str(job_dir / RIG_JSON_TMP),
        "result_path": str(job_dir / ".blender_result.json"),
        "template": template_key,
    }
    if bones is not None:
        spec["bones"] = bones
    if template_bones is not None:
        spec["template_bones"] = template_bones
    if fit is not None:
        spec["fit"] = fit
    if joints is not None:
        if joints not in JOINT_SOURCES:
            raise ValueError(f"joints must be one of {list(JOINT_SOURCES)}")
        spec["joints"] = joints
    if skeleton is not None:
        spec["skeleton"] = skeleton
    if root is not None:
        spec["root"] = root
    if mirror_pairs is not None:
        spec["mirror_pairs"] = [list(p) for p in mirror_pairs]
    return spec


def finalize_rig(job_dir: Path) -> None:
    """Rename the worker's temp artifacts onto the served names.

    GLB first, json second: rig.json is the completion marker, so at the
    moment it lands the rig.glb beside it is already the matching one.

    The renames retry briefly: on Windows, replacing rig.glb while a
    FileResponse is mid-stream from it raises PermissionError (files open
    without FILE_SHARE_DELETE), and failing an otherwise successful re-rig
    at the last step over a transient reader is the wrong trade.

    The pair is not atomic, and the failure that matters is the second half:
    once the GLB has landed, a JSON rename that exhausts its retries would leave
    the *new* rig.glb beside the *old* rig.json -- a completion marker
    advertising a skeleton the mesh no longer has. So that failure takes the
    stale marker with it: the directory then reads as "not rigged", which is
    true, and the caller's ``discard_rig_temps`` removes the unpublished JSON.

    A pose baked under the *previous* skeleton (``poses/<id>.glb``) or an
    ``animated.glb`` baked from the previous rig.glb's clips depicts joints
    this rig no longer has -- existence is each artifact's whole freshness
    test (``derive.get_file``'s ``animated.glb`` branch, ``rig.posed_model``),
    so left alone either would go on being served, silently wrong, forever.
    Removed here, before the renames below land the new rig.json: a crash in
    between costs a rebake under the *old* rig on next request, never a stale
    one served under the new rig.json's authority. The pose *records*
    (``poses/<id>.json``) are untouched -- a pose is still the same rotation
    request, only its cached bake is invalidated.
    """
    (job_dir / "animated.glb").unlink(missing_ok=True)
    poses_dir = job_dir / POSE_DIR_NAME
    if poses_dir.is_dir():
        for stale in poses_dir.glob("*.glb"):
            stale.unlink(missing_ok=True)

    for src, dest in ((RIG_GLB_TMP, "rig.glb"), (RIG_JSON_TMP, "rig.json")):
        for attempt in range(10):
            try:
                os.replace(job_dir / src, job_dir / dest)
                break
            except PermissionError:
                if attempt == 9:
                    if dest == "rig.json":
                        with contextlib.suppress(OSError):
                            (job_dir / "rig.json").unlink()
                    raise
                time.sleep(0.5)


def discard_rig_temps(job_dir: Path) -> None:
    """Remove whatever a failed or cancelled rig run left behind. Idempotent."""
    (job_dir / RIG_GLB_TMP).unlink(missing_ok=True)
    (job_dir / RIG_JSON_TMP).unlink(missing_ok=True)


def read_rig(job_dir: Path) -> dict[str, Any] | None:
    return read_record(job_dir / "rig.json", "rig.json")


class RigError(ValueError):
    """A rig.json this reads cannot answer for, naming the field.

    ``poselib.RecordError``'s shape, for a rig record instead of a pose one:
    a ``ValueError`` subclass so nothing that already catches ``ValueError``
    changes, with ``field`` carried so ``service.errors.invalid_from`` can
    point the refusal at a control instead of the message arriving addressed
    to nothing.
    """

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field


def validate_rig_bones(bones: Any) -> list[str]:
    """A rig.json's ``bones`` list, name-checked. Raises :class:`RigError`.

    docs/INVARIANTS.md states a pose *or rig* JSON is validated at the read
    door, not only at the write door -- but until the 2026-09-08 audit
    (poser-02) that was only ever built for pose records: a rig.json passed
    ``read_record``'s three file-level guards (valid JSON, valid dict, under
    the byte ceiling) and then a bare ``[b["name"] for b in bones]`` crashed
    on a bone with no ``name`` key with an uncaught ``KeyError``, one field
    deeper than the case the invariant already names as fixed. This is that
    same door for rig.json's bone list, tolerant like ``poselib.validate_
    record``'s own bone check: a bone entry that is not a dict, or whose name
    is not a non-empty string, is refused by name rather than by traceback.
    """
    if not isinstance(bones, list):
        raise RigError("rig.json's bones must be a list", field="bones")
    names: list[str] = []
    for entry in bones:
        name = entry.get("name") if isinstance(entry, dict) else None
        if not isinstance(name, str) or not name:
            raise RigError("rig.json has a bone with no name", field="bones")
        names.append(name)
    return names


def rig_bone_names(job_dir: Path) -> list[str] | None:
    rig = read_rig(job_dir)
    if rig is None:
        return None
    return validate_rig_bones(rig.get("bones", []))


# --- pose storage -----------------------------------------------------------
#
# One file per pose under <job_dir>/poses/, not a single poses.json. A pose is
# saved from the browser while the queue may be writing that same job's
# directory, and per-file writes need no read-modify-write and so no lock; the
# derived <pose_id>.glb sits beside its json for the same reason.

POSE_DIR_NAME = "poses"

# A ceiling so a scripted client can't turn a job directory into a million
# files. Far above any plausible hand-authored set.
MAX_POSES = 500

# A rig or pose record is a few KB of bone names and quaternions; a megabyte is
# already three orders of magnitude of headroom. Checked by stat *before* the
# read so a blob dropped into a job directory costs a log line rather than the
# RAM to parse it. Shared with poselib, which stores the same shape of record.
MAX_RECORD_BYTES = 1 << 20


def read_record(path: Path, what: str) -> dict[str, Any] | None:
    """One JSON sidecar, or ``None`` -- with the three guards every one needs.

    Written once because it was written four times and skipped three: the sheet,
    the pixel-sheet and the sprite-draft readers each had the two-line version
    of this, and each was missing all three guards.

    - **The ceiling is checked by ``stat`` before the read.** A blob dropped
      into a job directory should cost a log line, not the RAM to parse it.
    - **``ValueError``, not ``JSONDecodeError``.** ``UnicodeDecodeError`` is a
      ``ValueError`` and is *not* a ``JSONDecodeError``, so a sidecar holding
      bytes that are not UTF-8 raised straight out of the reader, through the
      listing, and into the pane -- instead of costing one record.
    - **The document has to be an object.** A valid-JSON array passes the
      parse and then fails at the first ``record.get(...)``, somewhere else
      entirely.
    """
    if not path.exists():
        return None
    try:
        size = path.stat().st_size
        if size > MAX_RECORD_BYTES:
            log.warning("ignoring %s at %s: %d bytes, over the ceiling", what, path, size)
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log.exception("unreadable %s at %s", what, path)
        return None
    if not isinstance(record, dict):
        log.warning("ignoring %s at %s: the document is not an object", what, path)
        return None
    return record


def write_json_staged(path: Path, payload: dict[str, Any], *, prefix: str) -> None:
    """Write one JSON document by staging it beside its destination.

    The Settings.flush rule, held in one place because three callers had a
    byte-identical copy of it: an overwrite that died mid-write used to leave
    the existing record truncated, and a pose file is the only copy of its
    rotations. ``prefix`` names the temp after what it will become, so a
    stranded one is attributable; ``BaseException`` rather than ``Exception``
    because a KeyboardInterrupt between the write and the rename must still
    take the dotfile with it.
    """
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=prefix, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def pose_dir(job_dir: Path) -> Path:
    return job_dir / POSE_DIR_NAME


def pose_path(job_dir: Path, pose_id: str) -> Path:
    """The record for one pose. Raises ValueError on an id that isn't ours.

    The guard is the point: this is the only place a caller-supplied pose id is
    turned into a path, and ``job_dir / pose_id`` does no sanitising of its own.
    """
    if not is_valid_id(pose_id):
        raise ValueError(f"malformed pose id {pose_id!r}")
    return pose_dir(job_dir) / f"{pose_id}.json"


def pose_glb_path(job_dir: Path, pose_id: str) -> Path:
    return pose_path(job_dir, pose_id).with_suffix(".glb")


def save_pose(
    job_dir: Path,
    pose: dict[str, Any],
    pose_id: str | None = None,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a validated pose payload and return the stored record.

    Passing an existing ``pose_id`` overwrites it in place -- that is how the
    editor saves an edit rather than accumulating near-duplicates. The derived
    GLB is dropped on overwrite, since it no longer depicts the pose.

    ``extra`` is merged into the record before the atomic write -- how a
    library-pose snapshot carries ``root_translation`` and ``source_pose``
    without every plain save learning those fields exist. It may not touch the
    keys this function owns: an ``extra`` that renamed the pose or swapped its
    bones would be a second writer of the same fact.
    """
    pose_id = pose_id or new_id()
    path = pose_path(job_dir, pose_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "id": pose_id,
        "name": pose["name"],
        "bones": pose["bones"],
        "created": time.time(),
    }
    if extra:
        protected = {"id", "name", "bones", "created"} & set(extra)
        if protected:
            raise ValueError(f"extra may not override {sorted(protected)}")
        record.update(extra)
    # The GLB is dropped *before* the JSON is staged in, not after. The two
    # statements are not atomic together, and a crash between them is real:
    # the 2026-09-08 audit's poser-04 found the old order -- write the JSON,
    # then drop the GLB -- left a crash between them pairing the *new* pose
    # record with the *old* baked GLB on disk, which posed_model's
    # ``if not path.exists(): ... bake ...`` then served as fresh forever,
    # with no error or staleness signal. This order's worst case is a crash
    # leaving the *old* JSON with no cached GLB -- costing only a redundant
    # rebake next time the pose is fetched, never stale content.
    pose_glb_path(job_dir, pose_id).unlink(missing_ok=True)
    write_json_staged(path, record, prefix=f".{pose_id}.")
    return record


def read_pose(job_dir: Path, pose_id: str) -> dict[str, Any] | None:
    return read_record(pose_path(job_dir, pose_id), "pose")


def _created_key(record: dict[str, Any]) -> float:
    """Sort key for ``created``, tolerant of a hand-edited file.

    A string timestamp sorted against floats raises and costs the whole list;
    an unusable one sorts as 0.0 instead, and a stable sort leaves equal keys
    in the filename order they were globbed in.
    """
    value = record.get("created", 0.0)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def list_poses(job_dir: Path) -> list[dict[str, Any]]:
    """Every saved pose, oldest first. A corrupt file costs itself, not the list."""
    directory = pose_dir(job_dir)
    if not directory.is_dir():
        return []
    poses = []
    for path in sorted(directory.glob("*.json")):
        if not is_valid_id(path.stem):
            continue
        record = read_pose(job_dir, path.stem)
        if record is None:
            continue
        if record.get("id") != path.stem:
            # The filename is the address every caller uses -- read, delete and
            # overwrite all go through pose_path(stem). A record whose internal
            # id disagrees is listed under its stem so it stays reachable; the
            # next save writes the reconciled id back.
            log.warning(
                "pose %s records id %r; listing it under its filename", path, record.get("id")
            )
            record = dict(record, id=path.stem)
        poses.append(record)
    poses.sort(key=_created_key)
    return poses


def delete_pose(job_dir: Path, pose_id: str) -> bool:
    path = pose_path(job_dir, pose_id)
    if not path.exists():
        return False
    # The derived GLB goes before the source JSON it depends on, matching
    # save_pose's ordering above and for the same reason (poser-05, the
    # 2026-09-08 audit): the old order -- delete the JSON, then the GLB --
    # left a crash between them stranding an orphaned <pose_id>.glb that
    # nothing lists, sweeps, or ever deletes, since posed_model raises
    # NotFound on the missing JSON before it would ever look for the GLB.
    # This order's worst case is a crash that leaves the JSON undeleted
    # (delete_pose raises, the pose still "exists") with no orphan behind it.
    pose_glb_path(job_dir, pose_id).unlink(missing_ok=True)
    path.unlink()
    return True


# --- sprite sheets ----------------------------------------------------------
#
# Same storage shape as poses, and beside them, for the same reasons: a sheet
# belongs to the mesh it depicts, one file per sheet needs no locking, and the
# id guard is the only thing between a caller and an arbitrary path.

SHEET_DIR_NAME = "sheets"

MAX_SHEETS = 200

# Same cap as MAX_POSE_NAME, for the same reason: a label the UI has to render.
MAX_SHEET_NAME = 64


def sheet_dir(job_dir: Path) -> Path:
    return job_dir / SHEET_DIR_NAME


# --- the deformation QA sheet -----------------------------------------------
#
# One per rig, overwritten by the next rig, and deliberately *not* in sheets/:
# it is a review artifact about the rig, not a sprite sheet the user asked for,
# so it must not appear in list_sheets, count against MAX_SHEETS or be deleted
# by delete_sheet. It lands in the source job's directory beside rig.glb for
# the reason the rig itself does -- it describes that mesh.

RIG_QA_PNG = "rig_qa.png"
RIG_QA_JSON = "rig_qa.json"


def rig_qa_png_path(job_dir: Path) -> Path:
    return job_dir / RIG_QA_PNG


def rig_qa_path(job_dir: Path) -> Path:
    return job_dir / RIG_QA_JSON


def sheet_path(job_dir: Path, sheet_id: str) -> Path:
    if not is_valid_id(sheet_id):
        raise ValueError(f"malformed sheet id {sheet_id!r}")
    return sheet_dir(job_dir) / f"{sheet_id}.json"


def sheet_png_path(job_dir: Path, sheet_id: str) -> Path:
    return sheet_path(job_dir, sheet_id).with_suffix(".png")


def sheet_pixel_path(job_dir: Path, sheet_id: str) -> Path:
    """The pixel restyle's sidecar, beside the render's.

    ``<id>.pixel.json`` rather than a second directory, and it is invisible to
    ``list_sheets`` for free: ``<id>.pixel`` fails ``is_valid_id``, so the
    listing skips it without needing to know this feature exists.
    """
    return sheet_path(job_dir, sheet_id).with_suffix(".pixel.json")


def sheet_pixel_png_path(job_dir: Path, sheet_id: str) -> Path:
    return sheet_path(job_dir, sheet_id).with_suffix(".pixel.png")


def read_sheet_pixel(job_dir: Path, sheet_id: str) -> dict[str, Any] | None:
    return read_record(sheet_pixel_path(job_dir, sheet_id), "pixel sheet")


def read_sheet(job_dir: Path, sheet_id: str) -> dict[str, Any] | None:
    return read_record(sheet_path(job_dir, sheet_id), "sheet")


def list_sheets(job_dir: Path) -> list[dict[str, Any]]:
    """Every finished sheet, oldest first.

    A sidecar with no PNG beside it is skipped: the queue writes the PNG first
    and the sidecar last, so the sidecar is the completion marker -- but a
    half-cleaned directory should not advertise an image that isn't there.
    """
    directory = sheet_dir(job_dir)
    if not directory.is_dir():
        return []
    sheets = []
    for path in sorted(directory.glob("*.json")):
        # .is_file(), not .exists(): the 2026-09-07/2026-09-08 audits fixed the
        # identical presence check at every other site in this area (sheet.pack,
        # pixelize.reduce_frames, troupe_mode.scores/atlas_texture) because
        # .exists() is also True for a directory, which a completed sheet's PNG
        # name never is but a hand-dropped one could be -- and this reader was
        # the one site the 2026-09-11 audit (troupe-06) found still unfixed.
        if not is_valid_id(path.stem) or not path.with_suffix(".png").is_file():
            continue
        record = read_sheet(job_dir, path.stem)
        if record is not None:
            sheets.append(record)
    sheets.sort(key=lambda s: s.get("created", 0.0))
    return sheets


def delete_sheet(job_dir: Path, sheet_id: str) -> bool:
    """Both files, and the pixel restyle's pair if one was ever made.

    The restyle is derived from this render and depicts nothing else, so
    leaving it behind would leave a sprite sheet of a sheet that is gone.
    """
    paths = [
        sheet_path(job_dir, sheet_id),
        sheet_png_path(job_dir, sheet_id),
        sheet_pixel_path(job_dir, sheet_id),
        sheet_pixel_png_path(job_dir, sheet_id),
    ]
    if not any(p.exists() for p in paths):
        return False
    for path in paths:
        path.unlink(missing_ok=True)
    return True


# --- sprite sheet drafts ----------------------------------------------------
#
# Same storage shape again, and for the same reasons -- but with one rule the
# others do not have: a draft is **write-once**. Every generate run mints a new
# id, so no draft is ever rewritten in place, and the pane can therefore cache
# this listing behind the directory's mtime without a stale record surviving an
# edit that never happens. Deleting is the only mutation.
#
# A draft is a *pair* of candidates from one run, so it is three files: the two
# PNGs, then the sidecar last, which is the completion marker -- the same order
# and the same reason as sheets.

SPRITE_DIR_NAME = "sprites"

# Two files per draft and a thumbnail per candidate in the sidebar; well under
# MAX_SHEETS because these accumulate per *attempt* rather than per keeper.
MAX_SPRITE_DRAFTS = 50

SPRITE_CANDIDATES = ("a", "b")


def sprite_dir(job_dir: Path) -> Path:
    return job_dir / SPRITE_DIR_NAME


def sprite_draft_path(job_dir: Path, draft_id: str) -> Path:
    if not is_valid_id(draft_id):
        raise ValueError(f"malformed sprite draft id {draft_id!r}")
    return sprite_dir(job_dir) / f"{draft_id}.json"


def sprite_draft_png_path(job_dir: Path, draft_id: str, candidate: str) -> Path:
    """``<id>.a.png`` / ``<id>.b.png``, beside the sidecar.

    The candidate letter is checked here rather than trusted: it arrives from a
    UI route, and ``<id>.<anything>.png`` would otherwise be a path a caller
    chooses. ``<id>.a`` also fails ``is_valid_id``, so ``list_sprite_drafts``
    skips the PNGs for free without knowing this naming exists.
    """
    if candidate not in SPRITE_CANDIDATES:
        raise ValueError(f"unknown sprite candidate {candidate!r}")
    return sprite_draft_path(job_dir, draft_id).with_suffix(f".{candidate}.png")


def read_sprite_draft(job_dir: Path, draft_id: str) -> dict[str, Any] | None:
    return read_record(sprite_draft_path(job_dir, draft_id), "sprite draft")


def list_sprite_drafts(job_dir: Path) -> list[dict[str, Any]]:
    """Every finished draft, oldest first. A draft missing a PNG is not one.

    Missing *which* PNG is the sidecar's answer and not this function's, which
    is the one thing here that is not obvious. It used to demand both letters of
    :data:`SPRITE_CANDIDATES`, and that was a hidden cap: a big sheet is drawn
    as a single candidate (``spritesynth.default_candidates``), so every
    eight-direction draft ever made would have been complete on disk, correct in
    its sidecar, and invisible in the pane. The record is read first and its own
    ``candidates`` list says which images it claims -- which is a stricter check
    than the old one as well as a truer one, since it also catches a draft
    claiming a candidate whose PNG never landed.
    """
    directory = sprite_dir(job_dir)
    if not directory.is_dir():
        return []
    drafts = []
    for path in sorted(directory.glob("*.json")):
        if not is_valid_id(path.stem):
            continue
        record = read_sprite_draft(job_dir, path.stem)
        if record is None:
            continue
        claimed = record.get("candidates")
        # A record with no usable list falls back to demanding both, which is
        # what every draft written before the count was a choice carries anyway
        # -- untrusted JSON does not get to shrink the check to nothing.
        letters = (
            SPRITE_CANDIDATES[: len(claimed)]
            if isinstance(claimed, list) and claimed
            else SPRITE_CANDIDATES
        )
        if not all(path.with_suffix(f".{c}.png").exists() for c in letters):
            continue
        drafts.append(record)
    drafts.sort(key=lambda d: d.get("created", 0.0))
    return drafts


def delete_sprite_draft(job_dir: Path, draft_id: str) -> bool:
    """The trio. A draft is one run, so its two candidates go together."""
    paths = [sprite_draft_path(job_dir, draft_id)] + [
        sprite_draft_png_path(job_dir, draft_id, c) for c in SPRITE_CANDIDATES
    ]
    if not any(p.exists() for p in paths):
        return False
    for path in paths:
        path.unlink(missing_ok=True)
    return True


def sheet_spec(
    source_glb: Path,
    frames_dir: Path,
    cells: list[dict[str, Any]],
    *,
    frame_size: int,
    elevation: float,
    lighting: str,
    margin: float | None = None,
    sockets: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """The worker spec for rendering one frame per sheet cell.

    ``cells`` carries each cell's pose bones inline rather than a pose id: the
    worker has no access to the job directory's pose files, and shipping the
    rotations with the cell keeps it a pure renderer.

    ``margin`` widens the ortho window past ``sheet.FRAME_MARGIN``. Its one
    writer is ``_q_troupe``'s reframe retry, and it is written **only when
    given** -- a spec without it is byte-identical to the one this function
    produced before the key existed, which is what keeps every sheet that does
    not clip rendering exactly as it did.

    ``sockets`` is ``[{"name", "bone", "offset": [along, lateral, up],
    "reach"}]``: attachment points to project per cell, offsets in bone-length
    units and reach in character heights so a socket list survives a re-fit
    onto a character of another size. Written only when given, for the same
    reason, and its presence is what makes the worker emit a ``sockets`` block
    at all.
    """
    spec: dict[str, Any] = {
        "op": "sheet",
        "source_glb": str(source_glb),
        "frames_dir": str(frames_dir),
        "result_path": str(frames_dir / "result.json"),
        "frame_size": frame_size,
        "elevation": elevation,
        "lighting": lighting,
        "cells": cells,
    }
    if margin is not None:
        spec["margin"] = float(margin)
    if sockets is not None:
        spec["sockets"] = [dict(s) for s in sockets]
    return spec


def views_spec(
    source_glb: Path,
    views_dir: Path,
    views: list[tuple[float, float]],
    *,
    size: int,
    depth: bool = False,
) -> dict[str, Any]:
    """The worker spec for rendering one flat view per direction.

    Half of a re-texture. The other half (``project_spec``) runs *after* the
    host has restyled these renders, which is why it is two ops rather than
    one: the restyle is an SDXL pass and Blender is a separate interpreter with
    no way to call back into this one.

    ``depth`` adds a second render per view -- the camera-depth encoding the
    visibility test and the ControlNet hint both decode. Off by default so
    every other caller keeps meaning exactly what it meant before the depth
    pass existed.
    """
    return {
        "op": "views",
        "source_glb": str(source_glb),
        "views_dir": str(views_dir),
        "result_path": str(views_dir / ".views_result.json"),
        "size": size,
        "views": [list(v) for v in views],
        "depth": bool(depth),
    }


def project_spec(
    source_glb: Path,
    views_dir: Path,
    out_dir: Path,
    views: list[tuple[float, float]],
    *,
    size: int,
    texture_size: int,
    depth: bool = False,
) -> dict[str, Any]:
    """The worker spec for baking each restyled view into the mesh's atlas.

    ``size`` is the render size again rather than a second knob: it is what
    frames the camera, and a camera framed differently from the one that drew
    the views would land every projection somewhere the colours are not.

    ``depth`` adds a third bake per view: the depth the camera recorded at
    each texel's pixel, next to the texel's own depth, which is everything the
    host's visibility compare needs. It requires the ``depth`` renders from
    ``views_spec`` to exist, and it defaults off for ``views_spec``'s reason.
    """
    return {
        "op": "project",
        "source_glb": str(source_glb),
        "views_dir": str(views_dir),
        "out_dir": str(out_dir),
        "result_path": str(out_dir / ".project_result.json"),
        "size": size,
        "texture_size": texture_size,
        "views": [list(v) for v in views],
        "depth": bool(depth),
    }


def fbx_spec(source_glb: Path, out_fbx: Path, result_dir: Path) -> dict[str, Any]:
    """The worker spec for converting a GLB to FBX.

    Blender is the converter because it is already here and already
    out-of-process; adding an FBX library to the app process would mean a second
    importer disagreeing with the one that produces every other artifact.
    """
    return {
        "op": "fbx",
        "source_glb": str(source_glb),
        "out_fbx": str(out_fbx),
        "result_path": str(result_dir / ".fbx_result.json"),
    }


def pose_spec(
    job_dir: Path,
    pose_id: str,
    bones: dict[str, Any],
    *,
    root_bone: str | None = None,
    root_offset: Sequence[float] | None = None,
) -> dict[str, Any]:
    """The worker spec for baking one saved pose into its own GLB.

    ``root_bone``/``root_offset`` carry a library pose's root translation
    (world units, Blender axes) into the bake. Both keys are added only when
    there is a bone *and* a nonzero offset, so every spec built before the
    fields existed -- and every pose without an offset -- is byte-identical to
    what it always was: backward compatibility is structural, not versioned.
    """
    spec = {
        "op": "pose",
        "rig_glb": str(job_dir / "rig.glb"),
        "out_glb": str(pose_glb_path(job_dir, pose_id)),
        "result_path": str(pose_dir(job_dir) / f".{pose_id}.result.json"),
        "bones": bones,
    }
    if root_bone is not None and root_offset is not None and any(float(v) for v in root_offset):
        spec["root_bone"] = root_bone
        spec["root_offset"] = [float(v) for v in root_offset]
    return spec


def armature_spec(template_key: str, out_glb: Path, result_dir: Path) -> dict[str, Any]:
    """The worker spec for exporting one template's armature with no mesh.

    What the Poser preview stands on: the skeleton is built and exported by the
    *same* Blender code path a real rig uses (``_build_armature`` + ``_export``),
    fitted over the canonical unit box, so the bone frames the editor rotates
    are the frames every bake will see. ``out_glb`` must end in ``.glb`` -- the
    exporter appends one to any path that does not (the RIG_GLB_TMP rule).
    """
    get_template(template_key)  # fail here, not three seconds into a subprocess
    return {
        "op": "armature",
        "template": template_key,
        "out_glb": str(out_glb),
        # Named per template: two previews building concurrently share the
        # previews directory, and ``run_worker`` unlinks and then watches the
        # result path -- one shared name would let each build eat the other's
        # answer. The per-template lock in ``template_preview`` serializes one
        # template's builds; this is what keeps two *different* ones apart.
        "result_path": str(result_dir / f".{template_key}.armature_result.json"),
    }


def clip_sample_spec(
    source: Path, template: str, result_path: Path, *, max_frames: int = 900
) -> dict[str, Any]:
    """The worker spec for "Import clip": sampling an external animation's
    world bone transforms so a later, Blender-free step can convert them onto
    ``template``'s own bone frames.

    **Never called "retarget"** -- see ``clipmaps``'s module docstring for why.
    ``candidates``/``strip`` are every shipped clip map's chain names and
    strip pattern, not just the one that will eventually match: the worker
    only has to recognise a source skeleton's *naming* well enough to pick the
    right armature and bones out of the file, and score which map actually
    resolves against ``source_bones``/``all_bone_names`` runs host-side,
    later, through ``clipmaps.match`` -- the same split ``fit_template``'s
    bbox arithmetic keeps clear of ``_build_armature``'s bpy calls.

    ``result_path`` is the caller's to make unique (``new_id()``, the same
    rule a pose or sheet cell follows) -- unlike ``armature_spec``'s
    per-template name, two "Import clip" runs against the same template have
    no lock serialising them and must never share a result file.
    """
    get_template(template)  # fail here, not three seconds into a subprocess
    from . import clipmaps  # local: clipmaps imports this module at its own top

    candidates: set[str] = set()
    strips: set[str] = set()
    for clip_map in clipmaps.load_clip_maps().values():
        strips.add(clip_map.strip.pattern)
        for chain in clip_map.bones.values():
            candidates.update(chain)
    return {
        "op": "clip_sample",
        "source": str(source),
        "template": template,
        "candidates": sorted(candidates),
        "strip": sorted(strips),
        "max_frames": int(max_frames),
        "result_path": str(result_path),
    }


def root_offset_world(root_translation: Sequence[float], bounds: dict[str, Any]) -> list[float]:
    """A library pose's root offset, scaled from character heights to world units.

    ``root_translation`` is stored in character-height units because the Poser
    armature is exactly one character tall (``poselib.UNIT_LO``/``UNIT_HI``);
    the target rig's height comes from rig.json's ``bounds``, which are Blender
    world coordinates, Z up. A degenerate height falls back to the largest
    extent -- the ``fit_template`` rule for a flat axis -- and to 1.0 when the
    whole box is a point, so the offset degrades to "as authored" rather than
    collapsing to zero.
    """
    lo = [float(v) for v in bounds["min"]]
    hi = [float(v) for v in bounds["max"]]
    h = hi[2] - lo[2]
    if h <= 0:
        h = max(b - a for a, b in zip(lo, hi, strict=True))
        if h <= 0:
            h = 1.0
    return [float(u) * h for u in root_translation]


# --- remesh --------------------------------------------------------------------

#: The remesh's staging name, beside the ``model.glb`` it is a candidate
#: replacement for -- ``RETEXTURE_GLB_TMP``'s rule exactly: published by
#: rename, discarded by a cancel, one spelling for the writer and the sweeper.
REMESH_GLB_TMP = ".remesh.tmp.glb"


def remesh_spec(
    source_glb: Path,
    out_glb: Path,
    result_dir: Path,
    *,
    target_faces: int,
    texture_size: int,
    close_holes: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    """The worker spec for remeshing a GLB to a quad budget and rebaking it.

    One op rather than the re-texture's two, because nothing on the host sits
    between the steps: the remesh, the unwrap and every bake are Blender's, and
    the host's only part is the publish afterwards. ``texture_size`` is
    resolved before this is built -- the worker never reads the atlas to decide
    it, so the spec is the whole record of what was asked.
    """
    return {
        "op": "remesh",
        "source_glb": str(source_glb),
        "out_glb": str(out_glb),
        "result_path": str(result_dir / ".remesh_result.json"),
        "target_faces": int(target_faces),
        "texture_size": int(texture_size),
        "close_holes": bool(close_holes),
        "seed": int(seed),
    }
