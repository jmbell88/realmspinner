"""Fitting a template onto a mesh's bounding box, and validating a skeleton's
structure once it is on there.

Split out of the former ``rigging.py`` (P4 of the 2026-09 restructure).
:func:`fit_template` only ever scales a template's own normalized landmarks;
everything below the "skeleton validation" banner is the door a whole skeleton
comes back through. The structural *editors* that used to sit beside it
(``add_bone``, ``split_bone``, ``attach_limb`` and the rest) served the
viewer's skeleton editor, which was removed on 2026-10-14 and had no
production caller (the 2026-10-14 audit, finding poser-rig-03); Clay builds
bones on its own document.

* :func:`validate_skeleton` takes a whole edited skeleton exactly once, before
  it is queued as a re-rig -- :func:`validate_joints`'s shape for a skeleton
  that may have a different shape than the template it started from.
* :func:`check_skeleton_structure` is its structural half alone.

rig.json's ``skeleton`` field records which of the two shapes came out the
other end: ``"template"`` when the edited skeleton still has exactly the
base template's names and parents (a joint move, or ``adjusted: True`` from
the older joints-only door), ``"custom"`` the moment either differs. A
custom skeleton keeps ``rig.json["template"]`` naming the *base* template it
started from -- the pose library needs to know which template's poses/clips
this rig might still play, even after its shape has diverged from it.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .store import RigError
from .templates import Template, _check_acyclic

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
# Anchored with ``\Z``, not ``$``: ``$`` also matches before a trailing newline, so
# ``"arm\n"`` was a legal name (the 2026-10-14 audit, finding poser-rig-06), the
# mistake ``store.RESOURCE_ID_RE`` was already moved off for the same reason.
BONE_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,63}\Z")

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


def _bounds_box(bounds: Mapping[str, Any]) -> tuple[list[float], list[float]]:
    """The rig's bounds grown by one diagonal on every side -- how far outside
    the mesh a joint may sit before it is a unit mistake, not a pose."""
    lo = [float(v) for v in bounds["min"]]
    hi = [float(v) for v in bounds["max"]]
    margin = _distance(lo, hi) or 1.0
    return [lo[i] - margin for i in range(3)], [hi[i] + margin for i in range(3)]


def _outside_box(point: Sequence[float], box: tuple[list[float], list[float]]) -> bool:
    box_lo, box_hi = box
    return any(point[i] < box_lo[i] or point[i] > box_hi[i] for i in range(3))


def validate_joints(
    payload: dict[str, Any],
    template: Template | Sequence[Mapping[str, Any]],
    *,
    bounds: Mapping[str, Any] | None = None,
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

    ``bounds`` (the rig's own ``{"min", "max"}``) switches on the gross-unit-
    mistake guard :func:`validate_skeleton` has: a head *or tail* farther than a
    diagonal outside the mesh is refused by name, before the span below is
    computed from it -- the 2026-10-03 audit, finding poser-35: one head at 1e9 m
    inflated the span and was refused as "bone 'hips' would be zero-length",
    while a far tail was accepted. Callers with no rig in hand omit it and get
    the older checks alone.
    """
    structure = template.bones if isinstance(template, Template) else template
    raw = payload.get("bones")
    if not isinstance(raw, list) or not raw:
        raise ValueError("joints payload requires a non-empty 'bones' list")
    by_name: dict[str, dict[str, Any]] = {}
    for entry in raw:
        # The 2026-09-26 audit, finding poser-rig-02: a non-object entry (a
        # bare bone name string, say) raised ``AttributeError`` out of the
        # ``.get`` below instead of the ``ValueError`` this function's own
        # docstring and every other refusal here promise -- and
        # ``service.rig.adjust_joints`` only catches ``(ValueError, KeyError,
        # TypeError)``, so the raw ``AttributeError`` reached its caller
        # unwrapped instead of becoming a field-addressed ``Invalid``.
        if not isinstance(entry, Mapping):
            raise ValueError(f"joints payload entries must be objects, not {entry!r}")
        name = str(entry.get("name") or "")
        # The 2026-09-26 audit, finding poser-rig-03: two entries naming the
        # same bone used to both land in ``by_name`` with the last silently
        # winning -- so a payload correcting one joint and, by a client bug,
        # repeating another bone's name twice would drop one of the two
        # corrections with no error at all.
        if name in by_name:
            raise ValueError(f"joints payload names bone {name!r} more than once")
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

    if bounds is not None:
        box = _bounds_box(bounds)
        for n in expected:
            for end in ("head", "tail"):
                if _outside_box(by_name[n][end], box):
                    raise ValueError(f"bone {n!r} {end} is far outside the mesh")

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


# --- skeleton validation ------------------------------------------------------
#
# ``validate_joints`` only lets a caller move a template's own named joints.
# Everything below validates a skeleton whose *structure* may differ -- see the
# module docstring above.


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


def _root_count_message(roots: Sequence[str]) -> str:
    """The refusal for a bone list without exactly one root, naming the roots it found so
    the author knows which bones to parent under which."""
    if len(roots) > 1:
        return (
            f"a skeleton must have exactly one root bone; this has {len(roots)} "
            f"({', '.join(roots)}). Parent the others under one bone first."
        )
    return "a skeleton must have exactly one root bone"


def check_skeleton_structure(bones: Sequence[Mapping[str, Any]]) -> str:
    """Bare structural sanity for a caller-supplied bone list, returning its
    root's name or raising :class:`~.store.RigError`.

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
        raise RigError(_root_count_message(roots), field="root")
    _check_acyclic({n: {"parent": p} for n, p in by_parent.items()}, field="bones")
    return roots[0]


def validate_skeleton(
    payload: dict[str, Any], *, base: Template, bounds: Mapping[str, Any]
) -> dict[str, Any]:
    """Normalize a whole edited skeleton, or raise :class:`~.store.RigError`.

    Returns ``{"bones", "root", "mirror_pairs", "skeleton"}`` -- everything
    a re-rig's params need and everything the worker's ``_rig_meta`` needs to
    write into rig.json.
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
        raise RigError(_root_count_message(roots), field="root")
    root = roots[0]

    _check_acyclic(by_name, field="bones")

    # Checked against the rig's own bounds -- not the bones' own spread, which
    # a single stray joint would corrupt -- and *before* the zero-length check
    # below: a joint placed a continent away inflates every span computed from
    # the bone heads, which would otherwise flag ordinary short bones (a
    # spine, a finger) as zero-length before the real problem is ever named.
    # Tails too (the 2026-10-03 audit, finding poser-35): a tail at 1e9 m built a
    # skeleton kilometres across because only heads were looked at.
    box = _bounds_box(bounds)
    for name, bone in by_name.items():
        for end in ("head", "tail"):
            if _outside_box(bone[end], box):
                raise RigError(f"bone {name!r} {end} is far outside the mesh", field="bones")

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
