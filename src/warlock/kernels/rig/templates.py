"""The skeleton template registry, and the limb presets grafted onto one.

Split out of the former ``rigging.py`` (P4 of ``dev/RESTRUCTURE.md``): this is
the "read shipped JSON under ``templates/``" concern -- a skeleton template and
a limb preset are the same file shape (named bones with head/tail landmarks,
a parent graph, checked for one root and no cycle) read by the same capped
JSON loader, so they stay one module rather than two that would each need the
other's parser.

Pure and Blender-free, like every other module this split produced: nothing
here imports ``bpy``, a service, or the studio.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .store import RigError

log = logging.getLogger(__name__)

# Depth-sensitive, the way ``kernels/manual/loader.py`` is: this file now
# lives three levels under ``src/warlock`` (``kernels/rig/templates.py``), so
# reaching the package-data directory needs three ``.parent`` climbs, not the
# one ``rigging.py`` needed when it sat directly in ``src/warlock``. Getting
# this wrong does not raise -- ``TEMPLATE_DIR.glob("*.json")`` on a directory
# that happens not to exist, or exists but is empty, silently yields nothing,
# and every template load then "succeeds" with an empty registry. That failure
# mode is exactly why ``test_template_dir_resolves_to_the_real_directory``
# below pins this to a directory that actually contains ``humanoid.json``.
TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "templates"

# A template's key is interpolated into filenames -- the Poser preview cache
# (``poselib.preview_path``) and its digest both build ``<key>.glb`` /
# ``<key>.json`` paths from it -- so the key must be a safe path component,
# and it comes out of the JSON body, not the filename.
TEMPLATE_KEY_RE = re.compile(r"^[a-z0-9_]+$")


@dataclass(frozen=True, slots=True)
class Template:
    key: str
    label: str
    root: str
    bones: tuple[dict[str, Any], ...]
    mirror_pairs: tuple[tuple[str, str], ...]
    #: True for a template that :func:`catalog` omits -- not a normal
    #: skeleton a user picks to auto-fit, but still a real, fully valid
    #: template that :func:`get_template`/``create_rig`` resolve exactly
    #: like any other. 2026-09-16's ``blank`` (Poser's manual-rig bootstrap,
    #: one root bone) is the first and, for now, the only one.
    hidden: bool = False


_templates: dict[str, Template] | None = None

# read_record's stat-before-read guard (see store.MAX_RECORD_BYTES), but
# these loaders each predate it and never got one: the 2026-09-11 audit
# (poser-03) found _load_templates/_load_pose_library/_load_clip_library each
# doing a bare ``json.loads(path.read_text(...))`` with no size check at all,
# so a file dropped in TEMPLATE_DIR, PRESET_DIR/BATTERY_DIR, or -- the one
# genuinely user-editable directory of the three, per user_clip_dir()'s own
# docstring -- CLIP_DIR/user_clip_dir() was read into memory in full before
# anything could refuse it.
#
# A shipped template is 1-3 KB and a shipped pose library 2-4 KB (checked on
# disk before choosing this); 1 MiB is the same three-orders-of-magnitude
# headroom store.MAX_RECORD_BYTES already uses for a single pose/rig record,
# and plenty for a hand-authored template or preset file that will never
# approach it. Shared by :mod:`.poses`' pose-library loader and
# :mod:`.cliplib`'s limb/clip-library ceilings restate their own multiple of
# it rather than importing this one, for the reasons each states locally.
MAX_TEMPLATE_BYTES = 1 << 20


def _read_json_capped(path: Path, ceiling: int) -> Any:
    """One JSON file, refusing anything over ``ceiling`` before it is parsed.

    Raises on any problem -- oversized, unreadable, not valid JSON -- so every
    caller's existing ``except Exception: log.exception(...)`` ("a malformed
    file costs you that entry, not the app") already covers this the same way
    it covers a bad body; this only moves the guard in front of the read
    instead of leaving it absent. Not ``store.read_record``: that one also
    demands the document be a dict, which the pose/rig sidecars it serves need
    but these registries do not -- their own parsing already raises a
    specific, more useful error on the wrong shape.

    Shared by :mod:`.poses` (the shipped pose libraries) and :mod:`.cliplib`
    (the clip libraries) rather than duplicated: both read the same
    ``templates/`` tree this module owns, and importing it from here is what
    keeps the cap-before-parse rule in one place.
    """
    size = path.stat().st_size
    if size > ceiling:
        raise ValueError(f"{path} is {size} bytes, over the {ceiling}-byte ceiling")
    return json.loads(path.read_text(encoding="utf-8"))


def _check_acyclic(by_name: dict[str, dict[str, Any]], *, field: str) -> None:
    """Raise :class:`~.store.RigError` if any bone's parent chain loops.

    Lives here rather than in :mod:`.skeleton` (whose own "skeleton editing"
    section this walk was lifted from) because both this module's parsers --
    :func:`_parse_template` and :func:`_parse_limb_preset` -- need it at load
    time, and :mod:`.skeleton` already imports :class:`Template` and
    :func:`get_limb_preset` from here; putting the acyclic check the other way
    round would make the two modules import each other. :mod:`.skeleton`
    imports it back from here for its own structural checks
    (``check_skeleton_structure``, ``validate_skeleton``).

    ``_build_armature`` (the Blender worker) parents every bone in one pass
    with no cycle check of its own -- ``eb.parent = created[parent]`` -- so a
    cyclic parent list sent across the pipe would not fail until Blender's own
    edit-bone graph did something undefined with it. Walked from every bone,
    not just the apparent root(s): a cycle with no bone at all pointing to
    ``None`` would otherwise have no starting point this ever visits.
    """
    for start in by_name:
        seen: set[str] = set()
        cur: str | None = start
        while cur is not None:
            if cur in seen:
                raise RigError(f"the bone hierarchy has a cycle at {cur!r}", field=field)
            seen.add(cur)
            cur = by_name[cur]["parent"]


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
    # The 2026-09-14 audit, finding poser-02: everything above checks each
    # bone's own parent resolves and that there is one root, but never that
    # walking those parent links from every bone actually reaches that root --
    # a schema-valid file naming a disconnected parent cycle would pass all
    # of it and only fail later, uncaught, wherever a consumer walks the
    # chain (``_build_armature`` parents bones in one pass with no cycle
    # check of its own). Caught here, at load, so a malformed template costs
    # only that template.
    try:
        _check_acyclic({b["name"]: {"parent": b["parent"]} for b in bones}, field="bones")
    except RigError as exc:
        raise ValueError(str(exc)) from exc
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
        hidden=bool(raw.get("hidden", False)),
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
    """The template table in the same {key, label} shape the UI selects use.

    A hidden template (``blank``, so far) is a real, fully working template
    -- ``get_template``/``create_rig`` resolve it exactly like any other --
    but not one a user should be offered from a normal skeleton picker,
    since fitting it automatically produces one bare bone and nothing to
    pose. It reaches a mesh only through Poser's own "Rig manually" door,
    which names it by key directly rather than through this list.
    """
    return [{"key": t.key, "label": t.label} for t in templates().values() if not t.hidden]


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
    # The 2026-09-14 audit, finding poser-02: nothing above checked that the
    # parent graph is actually a tree reachable from that one root, and
    # ``attach_limb``'s ``_attach_one`` walks ``preset_data["bones"]`` in file
    # order assuming each bone's parent was already grafted -- ``real_parent
    # = ... name_map[b["parent"]]`` -- so a schema-valid preset listing a
    # child before its parent throws an uncaught KeyError on the frame thread
    # (Poser's ``_skeleton_call`` catches only RigError) instead of being
    # skipped at load like every other malformed preset. Refusing
    # out-of-order bones here is what guarantees ``_attach_one``'s name_map
    # already holds a bone's parent by the time it is needed, and it also
    # rules out any cycle: a bone in a cycle can never be preceded by its own
    # parent. ``_check_acyclic`` is added too, for the same reachable-from-
    # root guarantee ``_parse_template`` now gets, rather than relying on the
    # order check alone to prove it by construction.
    seen: set[str] = set()
    for b in bones:
        if b["parent"] is not None and b["parent"] not in seen:
            raise ValueError(
                f"bone {b['name']!r} is listed before its parent {b['parent']!r}; "
                "a limb preset must list a parent before its children"
            )
        seen.add(b["name"])
    try:
        _check_acyclic({b["name"]: {"parent": b["parent"]} for b in bones}, field="bones")
    except RigError as exc:
        raise ValueError(str(exc)) from exc
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
