"""Fitting a template onto a mesh's bounding box, and editing a skeleton's
structure once it is on there.

Split out of the former ``rigging.py`` (P4 of ``dev/RESTRUCTURE.md``).
:func:`fit_template` only ever scales a template's own normalized landmarks;
everything below the "skeleton editing" banner lets Poser's skeleton editor
change the *structure* too -- add or remove a pivot, split a bone, graft a
limb preset -- the backend for a feature the UI is built on top of, per the
plan this module was extended for on 2026-09-12. Two halves:

* :func:`validate_skeleton` is the door a whole edited skeleton comes back
  through, exactly once, before it is queued as a re-rig -- ``service.rig.
  edit_skeleton``'s validator, :func:`validate_joints`'s shape for a skeleton
  that may have a different shape than the template it started from.
* The rest are pure functions over ``bones: list[dict]`` (``{name, parent,
  head, tail}``) and never mutate their input -- an editor undo stack is
  just the list of dicts each edit returned, and that only works if an
  earlier one is never touched in place. Every one of them raises
  :class:`~.store.RigError` naming a ``field`` on a bad edit,
  :func:`validate_skeleton`'s convention, so a UI building an undo/redo stack
  on top of these gets the same field-addressed refusals a first-time submit
  does.

rig.json's ``skeleton`` field records which of the two shapes came out the
other end: ``"template"`` when the edited skeleton still has exactly the
base template's names and parents (a joint move, or ``adjusted: True`` from
the older joints-only door), ``"custom"`` the moment either differs. A
custom skeleton keeps ``rig.json["template"]`` naming the *base* template it
started from -- :func:`clip_coverage` and the pose library both need to know
which template's poses/clips this rig might still play, even after its
shape has diverged from it.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .cliplib import clip_library
from .store import RigError
from .templates import Template, _check_acyclic, get_limb_preset

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
# Everything below lets Poser's skeleton editor change the *structure* too --
# see the module docstring above.


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
        raise RigError("a skeleton must have exactly one root bone", field="root")
    _check_acyclic({n: {"parent": p} for n, p in by_parent.items()}, field="bones")
    return roots[0]


def validate_skeleton(
    payload: dict[str, Any], *, base: Template, bounds: Mapping[str, Any]
) -> dict[str, Any]:
    """Normalize a whole edited skeleton, or raise :class:`~.store.RigError`.

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
    """A new list with one bone appended. Raises :class:`~.store.RigError`."""
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
