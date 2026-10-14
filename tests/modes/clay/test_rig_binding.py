"""Bones and rigid binding as pure questions over the object list, and the door that marks them.

Tranche 2 of the 2026-10-13 Clay rig plan. A bone is an object marked as one; rigid binding is
*derived* from the outliner (a shape moves with its nearest bone ancestor) and stored nowhere,
so these claims are about the questions and about ``ClayDoc.set_bone``.
"""

from __future__ import annotations

import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import rig

from ._rigkit import rigged_doc


def _obj(doc: bd.ClayDoc, name: str, *, parent: int | None = None, bone: bool = False) -> int:
    return doc.add_object(
        bd.Obj(uid=bd.new_uid(), name=name, mesh=bp.box(), parent=parent, bone=bone)
    ).uid


def test_a_shape_under_a_bone_moves_with_it_and_a_bone_moves_with_itself() -> None:
    doc = bd.ClayDoc()
    hip = _obj(doc, "Hip", bone=True)
    leg = _obj(doc, "Leg", parent=hip)
    free = _obj(doc, "Free")
    by_uid = doc._uid_map()
    assert rig.nearest_bone(by_uid, hip) == hip
    assert rig.nearest_bone(by_uid, leg) == hip
    assert rig.nearest_bone(by_uid, free) is None


def test_groups_between_bones_are_skipped_and_the_nearest_bone_wins() -> None:
    doc = bd.ClayDoc()
    spine = _obj(doc, "Spine", bone=True)
    group = _obj(doc, "Group", parent=spine)
    arm = _obj(doc, "Arm", parent=group, bone=True)
    hand = _obj(doc, "Hand", parent=arm)
    by_uid = doc._uid_map()
    assert rig.nearest_bone(by_uid, hand) == arm
    assert rig.bone_parent(by_uid, arm) == spine, "the group between them is not a bone"
    assert rig.bone_parent(by_uid, spine) is None
    assert [o.name for o in rig.bones(doc.objects)] == ["Spine", "Arm"]


def test_deleting_a_bone_unbinds_its_shapes_because_nothing_was_stored() -> None:
    doc = bd.ClayDoc()
    hip = _obj(doc, "Hip", bone=True)
    leg = _obj(doc, "Leg", parent=hip)
    assert rig.nearest_bone(doc._uid_map(), leg) == hip
    doc.set_bone([hip], False)
    assert rig.nearest_bone(doc._uid_map(), leg) is None


def test_a_parent_cycle_does_not_hang_the_walk() -> None:
    a = bd.Obj(uid=1, name="A", mesh=bp.box(), parent=2)
    b = bd.Obj(uid=2, name="B", mesh=bp.box(), parent=1)
    assert rig.nearest_bone({1: a, 2: b}, 1) is None


def test_marking_bones_is_one_undo_step_for_the_whole_selection() -> None:
    doc = bd.ClayDoc()
    uids = [_obj(doc, f"B{i}") for i in range(3)]
    steps = len(doc.history)
    assert doc.set_bone(uids, True) is True
    assert all(doc.by_uid(u).bone for u in uids)
    assert len(doc.history) == steps + 1
    assert doc.undo()
    assert not any(doc.by_uid(u).bone for u in uids)
    assert doc.redo()
    assert all(doc.by_uid(u).bone for u in uids)


def test_marking_what_is_already_marked_pushes_nothing() -> None:
    doc = bd.ClayDoc()
    uid = _obj(doc, "Hip", bone=True)
    steps = len(doc.history)
    assert doc.set_bone([uid], True) is False
    assert doc.set_bone([_obj(doc, "Other")], False) is False
    assert len(doc.history) == steps + 1, "only the add of 'Other' pushed"


def test_the_bone_count_is_capped_when_marking_and_when_adding() -> None:
    doc = bd.ClayDoc()
    uids = [_obj(doc, f"B{i}") for i in range(rig.MAX_BONES + 1)]
    with pytest.raises(el.OpError, match="past the 64"):
        doc.set_bone(uids, True)
    assert not any(o.bone for o in doc.objects), "a refused mark changes nothing"
    assert doc.set_bone(uids[: rig.MAX_BONES], True) is True
    with pytest.raises(el.OpError, match="past the 64"):
        doc.set_bone(uids[rig.MAX_BONES :], True)
    with pytest.raises(el.OpError, match="past the 64"):
        doc.add_object(bd.Obj(uid=bd.new_uid(), name="Extra", mesh=bp.box(), bone=True))


def test_the_cap_is_the_gpu_skinning_and_skeleton_editor_cap() -> None:
    from realmspinner.kernels.rig import skeleton
    from realmspinner.studio.viewer import programs

    assert rig.MAX_BONES == skeleton.MAX_SKELETON_BONES == programs.MAX_JOINTS


def test_a_duplicate_of_a_bone_is_a_bone_and_obeys_the_cap() -> None:
    doc, bone, body = rigged_doc()
    copy = bd.Obj(uid=bd.new_uid(), name="Hip.001", mesh=bp.box(), bone=True)
    doc.add_object(copy)
    assert sum(o.bone for o in doc.objects) == 2
