"""A pose payload -- a bone -> local rotation quaternion map -- and the two
rotation frames Poser and a clip library each speak.

Split out of the former ``rigging.py`` (P4 of ``dev/RESTRUCTURE.md``). The
shipped pose libraries (the picker's presets and the deformation-QA battery)
live here too rather than in :mod:`.templates`: both are read through
:func:`validate_bones`, this module's own door, and a pose library is "a pose,
several times" in a way a limb preset is not, so keeping the two together
reads truer than splitting a pose concept across two files.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..sheet import POSE_SPACES
from . import templates
from .templates import TEMPLATE_DIR, _read_json_capped, get_template

log = logging.getLogger(__name__)

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

    Split out of :func:`validate_pose` so a *clip library* (:mod:`.cliplib`)
    can use it: a clip's key poses are read from a file like any other pose
    and their quaternions were taken verbatim, so a 3-element list raised out
    of the middle of ``sheet._blend`` naming neither the file nor the bone,
    and a NaN was interpolated into a clip and written to disk.

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

    It is the kind of sign convention that is wrong in a way you cannot see --
    a mirrored arm that rotates the wrong way about one axis still looks
    plausible in a static pose. The 2026-09-15 audit, finding poser-05: this
    docstring used to warn that a JS copy in ``app.js`` had to be kept
    identical to this function; that browser-side copy is retired and no such
    file exists in this tree any more, so the warning named a maintenance
    burden that is no longer real.
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


# --- shipped pose libraries -------------------------------------------------
#
# A pose is a bone-name -> local-quaternion map, and skeleton.fit_template puts
# a given template's bones in the same place on every mesh. So a pose authored
# against one humanoid rig applies to every other humanoid rig -- which is what
# makes a shipped library possible at all, and why these read from
# ``templates/`` rather than being seeded into each job's poses/ directory.

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
            raw = _read_json_capped(path, templates.MAX_TEMPLATE_BYTES)
            # The frame the whole file's poses are authored in -- the same
            # top-level field ``cliplib.parse_clip_library`` reads,
            # and the same reason: a battery authored as deltas from rest
            # says so once, for every pose, rather than per bone. Left off
            # a row entirely when it is the default, so a preset file with
            # no opinion renders exactly as it always has.
            space = str(raw.get("space") or "node")
            # The 2026-09-20 audit's poser-02: this read no membership test
            # while ``sheet.parse_pose``/``parse_clip`` (the same top-level
            # field, the same reason) both check ``space not in POSE_SPACES``.
            # An unrecognised value fell through to "anything but the exact
            # string 'delta' is node" in ``blender_worker._apply_pose`` and
            # every pose in the file applied in the wrong rotation frame,
            # silently -- the "character was lying down" incident's own read
            # door. Raising here costs the file, not the app: the per-file
            # ``try`` below already skips a malformed library and logs it.
            if space not in POSE_SPACES:
                raise ValueError(f"space must be one of {list(POSE_SPACES)}, not {space!r}")
            rows = []
            for i, pose in enumerate(raw["poses"]):
                # The 2026-09-18 audit's second-run poser-01: this stored
                # ``pose["bones"]`` verbatim, contradicting this function's
                # own docstring and the read-door rule ``cliplib.
                # parse_clip_library`` follows for the identically shaped
                # case -- a malformed shipped quaternion (a NaN, a non-unit
                # norm past repair, a non-numeric value) reached
                # ``blender_worker._apply_pose`` unchecked. ``validate_bones``
                # raises inside this file's own per-file ``try``, so one bad
                # pose costs that pose's file, not the app.
                row = {"name": str(pose["name"]), "bones": validate_bones(pose["bones"])}
                if space != "node":
                    row["space"] = space
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
