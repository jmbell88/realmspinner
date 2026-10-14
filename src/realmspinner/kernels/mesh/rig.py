"""Bones and skins in a Clay document: the data, the caps and the questions.

**A bone is an object that was marked as one** (``Obj.bone``), BlockBench's idea that a
group is a bone and not a second kind of thing. Its pivot is its origin, its rest pose is
its own transform, and the bone hierarchy is the parent chain restricted to bones. A group
that only organises a scene stays a group and burns no joint slot, which is why the flag is
explicit and not derived from "has children".

**Binding comes in two kinds, and only one is stored.**

* *Rigid* binding is derived, never stored: a shape whose :func:`nearest_bone` exists and
  which carries no :class:`Skin` moves with that bone at weight 1.0, whatever the shape is
  parented to. Delete the bone and the shape is unbound again; nothing can go stale.
* *Weighted* binding is a :class:`Skin` on the object plus ``joints``/``weights`` on its
  mesh -- the shape Blender's heat weights, or a skinned glTF, arrive in. The skin names
  its joints by bone **uid**, never by index, for the rule every reference in this package
  follows (``edits``' opening paragraph), and carries the inverse bind matrices exactly as
  they arrived so that a round trip through Blender reads back what it wrote.

Everything here is pure data and questions over a ``uid -> object`` mapping: no op, no
selection, no history. The doors that change a bone or a skin are :class:`~.document.ClayDoc`'s.

**The caps are the reader's and the writer's both** (``serialize`` holds a document to the
same numbers): :data:`MAX_BONES` is the joint count the GPU skinning programs have room for
(``programs.MAX_JOINTS``) and the skeleton editor's ``MAX_SKELETON_BONES``, pinned equal to
both by ``tests/modes/clay/test_rig_caps.py``; :data:`MAX_SKIN_INFLUENCES` is the four joints
per vertex glTF's ``JOINTS_0``/``WEIGHTS_0`` carry.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "MAX_BONES",
    "MAX_SKIN_INFLUENCES",
    "Skin",
    "bone_parent",
    "bones",
    "nearest_bone",
    "skin_problems",
]

MAX_BONES = 64
MAX_SKIN_INFLUENCES = 4


@dataclass(frozen=True, eq=False)
class Skin:
    """Which bones move a mesh's vertices, and the matrices that undo their rest pose.

    ``joints`` is a bone uid per skin joint, in the order a vertex's ``joints`` indices
    count. ``inverse_bind`` is one 4x4 per joint, column-vector convention (``M @ v``),
    in the object's own frame -- stored as the file gave it, never recomputed, because a
    heat-weighted mesh is only correct against the matrices it was weighted with.
    """

    joints: tuple[int, ...]
    inverse_bind: Any

    def __post_init__(self) -> None:
        joints = tuple(int(j) for j in self.joints)
        if not joints:
            raise ValueError("a skin must name at least one joint")
        if len(set(joints)) != len(joints):
            raise ValueError("a skin names the same bone twice")
        if len(joints) > MAX_BONES:
            raise ValueError(f"a skin names {len(joints)} bones, past the {MAX_BONES} Clay holds")
        matrices = np.array(self.inverse_bind, dtype="f8", copy=True)
        if matrices.shape != (len(joints), 4, 4):
            raise ValueError(
                f"a skin needs one 4x4 inverse bind matrix per joint ({len(joints)}), "
                f"got {matrices.shape}"
            )
        if not np.isfinite(matrices).all():
            raise ValueError("a skin has an inverse bind matrix that is not a finite number")
        matrices.setflags(write=False)
        object.__setattr__(self, "joints", joints)
        object.__setattr__(self, "inverse_bind", matrices)


def bones(objects: Iterable[Any]) -> list[Any]:
    """Every marked bone, in document order."""
    return [obj for obj in objects if getattr(obj, "bone", False)]


def nearest_bone(by_uid: Mapping[int, Any], uid: int) -> int | None:
    """The uid of the bone *uid* moves with: itself if it is one, else the closest
    ancestor that is, else ``None``.

    An iterative walk with a visited set, for ``ClayDoc.ancestors``' reason: a parent
    chain of thousands of objects must not recurse, and a cycle (unreachable through any
    live edit, possible in a hand-edited file before it is refused) must not hang.
    """
    seen: set[int] = set()
    cursor: int | None = uid
    while cursor is not None and cursor not in seen:
        seen.add(cursor)
        obj = by_uid.get(cursor)
        if obj is None:
            return None
        if getattr(obj, "bone", False):
            return cursor
        cursor = obj.parent
    return None


def bone_parent(by_uid: Mapping[int, Any], uid: int) -> int | None:
    """The bone above bone *uid* in the hierarchy -- the nearest marked ancestor -- or
    ``None`` for a root bone. Groups between two bones are skipped."""
    obj = by_uid.get(uid)
    if obj is None or obj.parent is None:
        return None
    return nearest_bone(by_uid, obj.parent)


def skin_problems(objects: Sequence[Any]) -> list[str]:
    """Every way the skins in *objects* disagree with their meshes or the bones, as
    sentences. Empty is healthy.

    One function for the reader, the writer's check and ``ClayDoc.selection_problems``'
    sibling, so they cannot answer differently about what a consistent rig is.
    """
    by_uid = {obj.uid: obj for obj in objects}
    problems: list[str] = []
    marked = sum(1 for obj in objects if getattr(obj, "bone", False))
    if marked > MAX_BONES:
        problems.append(f"{marked} bones, past the {MAX_BONES} Clay holds")
    for obj in objects:
        skin = getattr(obj, "skin", None)
        has_weights = obj.mesh.joints is not None
        if skin is None and not has_weights:
            continue
        if (skin is None) != (not has_weights):
            problems.append(
                f"{obj.name!r} has "
                + ("a skin but no vertex weights" if skin is not None else "weights but no skin")
            )
            continue
        for joint in skin.joints:
            bone = by_uid.get(joint)
            if bone is None:
                problems.append(f"{obj.name!r}'s skin names uid {joint}, which does not exist")
            elif not getattr(bone, "bone", False):
                problems.append(f"{obj.name!r}'s skin names {bone.name!r}, which is not a bone")
        if len(obj.mesh.positions) and int(obj.mesh.joints.max()) >= len(skin.joints):
            problems.append(f"{obj.name!r} weights a joint its skin does not have")
    return problems
