"""The 2026-10-14 audit, fixer F11: Clay's document rig doors, the timeline and animation import.

Each test is named for the claim of one finding and was run against the unfixed tree first.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.kernels.mesh import animimport as ai
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import rig, serialize
from realmspinner.kernels.mesh.elements import OpError
from realmspinner.studio.modes.clay import timeline as tl
from realmspinner.studio.modes.clay.ui._view_pose import PoseOps

from ._rigkit import rigged_doc, weighted


def _angle(q) -> float:
    return float(np.degrees(2 * np.arctan2(q[2], q[3])))


def _quat(degrees: float):
    return m3.quat_from_axis_angle([0, 0, 1], np.radians(degrees))


# -- clay-44 (clay-rig-06) --------------------------------------------------------------------


def test_a_parked_playhead_past_a_looping_clips_end_holds_the_last_key() -> None:
    doc = bd.ClayDoc()
    hip = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Hip", mesh=bp.box(), bone=True)).uid
    doc.set_pose_mode(True)
    doc.add_animation("Wave")
    assert doc.animation().loop
    for t, deg in ((0.0, 0.0), (1.0, 90.0)):
        doc.set_key(hip, "rotation", t, _quat(deg))
    doc.set_rig_time(1.0)
    tl.step(doc, 1)
    assert doc.rig_time > 1.0
    assert _angle(doc.pose_overrides()[hip]["rotation"]) == pytest.approx(90.0, abs=1e-3)
    # Playback still wraps: it folds the time itself, and a playing playhead past the end does too.
    doc.playing = True
    doc.rig_time = 1.25
    assert _angle(doc.pose_overrides()[hip]["rotation"]) == pytest.approx(
        _angle(_quat(22.5)), abs=1e-3
    )


# -- clay-40 (clay-rig-02) --------------------------------------------------------------------


def test_two_different_non_ascii_bone_names_are_never_matched() -> None:
    assert ai.canonical("左腕") != ai.canonical("右腕")
    pairs, ambiguous = ai.match_bones({1: "左腕"}, {10: "右腕"})
    assert pairs == {} and ambiguous == []
    # Accented names keep their accented letters, so they do not collapse onto a neighbour.
    pairs, ambiguous = ai.match_bones({1: "Hüfte", 2: "Hfte"}, {10: "Hüfte"})
    assert pairs == {10: 1} and ambiguous == []
    # A name that is nothing but separators has no key at all, and matches nothing.
    pairs, ambiguous = ai.match_bones({1: "__"}, {10: "--"})
    assert pairs == {} and ambiguous == []
    # The ASCII behaviour the module promises is untouched.
    assert ai.canonical("mixamorig:LeftUpLeg") == ai.canonical("upleg.L") == "uplegl"
    assert ai.match_bones({1: "Épaule.L"}, {10: "Left_Épaule"}) == ({10: 1}, [])


# -- clay-02 (clay-document-01) ---------------------------------------------------------------


def test_a_static_export_of_a_document_with_a_weighted_mesh_puts_every_shape_on_its_own_node() -> (
    None
):
    doc, _bone, _body = rigged_doc()
    model = bd.to_model(doc, skinned=False)
    assert model.skins == []
    assert len(model.nodes) == 2
    assert all(node.mesh is not None for node in model.nodes if node.name == "Body")
    skinned = bd.to_model(doc)
    assert len(skinned.skins) == 1


# -- clay-03 (clay-document-02) ---------------------------------------------------------------


def _keyed_bone_and_other():
    doc = bd.ClayDoc()
    bone = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Hip", mesh=bp.box(), bone=True)).uid
    other = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Other", mesh=bp.box())).uid
    doc.add_animation("Walk")
    doc.set_key(bone, "translation", 0.0, [0, 1, 0])
    return doc, bone, other


def test_joining_or_separating_a_bone_never_leaves_a_skin_or_a_key_naming_a_missing_bone() -> None:
    # join: an animated bone is absorbed -> its keys go with it, in the same step
    doc, bone, other = _keyed_bone_and_other()
    doc.join_objects(other, bp.uv_sphere(), [bone])
    assert doc.rig_problems() == []
    assert all(bone not in a.channels for a in doc.rig_state.animations)
    serialize.rblk_bytes(doc)
    assert doc.undo()
    assert bone in doc.rig_state.animations[0].channels
    assert doc.by_uid(bone).bone

    # join: a bone a surviving skin names is refused, nothing changes
    doc, bone, body = rigged_doc()
    other = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Other", mesh=bp.box())).uid
    steps = len(doc.history)
    with pytest.raises(OpError, match="Unbind Skin"):
        doc.join_objects(other, bp.uv_sphere(), [bone])
    assert len(doc.history) == steps and doc.by_uid(bone).bone and doc.rig_problems() == []

    # separate: the same two cases
    doc, bone, _other = _keyed_bone_and_other()
    doc.separate(bone, [bp.box(), bp.box()])
    assert doc.rig_problems() == []
    assert all(bone not in a.channels for a in doc.rig_state.animations)
    serialize.rblk_bytes(doc)
    assert doc.undo()
    assert bone in doc.rig_state.animations[0].channels

    doc, bone, body = rigged_doc()
    with pytest.raises(OpError, match="Unbind Skin"):
        doc.separate(bone, [bp.box(), bp.box()])
    assert doc.by_uid(bone).bone and doc.rig_problems() == []


# -- clay-04 (clay-document-03) ---------------------------------------------------------------


def _hip_arm(arm_key=(0.0, 0.3, 0.0)):
    doc = bd.ClayDoc()
    hip = doc.add_bone(None).uid
    doc.set_transform(hip, translation=[5.0, 0.0, 0.0])
    arm = doc.add_bone(hip).uid
    doc.add_animation("Walk")
    doc.set_rig_time(0.0)
    doc.set_key(arm, "translation", 0.0, list(arm_key))
    doc.set_key(arm, "translation", 1.0, [arm_key[0] + 1.0, arm_key[1], arm_key[2]])
    doc.set_key(arm, "rotation", 0.0, _quat(30.0))
    doc.set_pose_mode(True)
    return doc, hip, arm


def _posed(doc, uid, time):
    doc.set_rig_time(time)
    return doc.posed_world(uid, doc.pose_overrides())


def _same_world(doc, arm, before):
    for time, expected in before.items():
        assert np.allclose(_posed(doc, arm, time), expected, atol=1e-6), time


@pytest.mark.parametrize("door", ["set_parent", "delete the parent", "set_origin", "group"])
def test_reparenting_a_keyed_bone_keeps_its_posed_world_position(door: str) -> None:
    doc, hip, arm = _hip_arm()
    before = {t: _posed(doc, arm, t) for t in (0.0, 0.5, 1.0)}
    original = doc.animation().channels[arm]["translation"].values.copy()
    steps = len(doc.history)
    if door == "set_parent":
        doc.set_parent(arm, None)
    elif door == "delete the parent":
        doc.remove_objects([hip])
    elif door == "set_origin":
        doc.set_origin(hip, [2.0, 3.0, 1.0])
    else:
        doc.group([arm])
    assert len(doc.history) > steps
    _same_world(doc, arm, before)
    # the keys were re-expressed with the rest transform, and one Ctrl+Z puts both back
    assert not np.array_equal(doc.animation().channels[arm]["translation"].values, original)
    assert doc.undo()
    assert np.array_equal(doc.animation().channels[arm]["translation"].values, original)
    _same_world(doc, arm, before)
    assert doc.redo()
    _same_world(doc, arm, before)


def test_reparenting_a_keyed_bone_under_a_nonuniform_scale_is_refused_not_sheared() -> None:
    doc, hip, arm = _hip_arm()
    other = doc.add_bone(None).uid
    doc.set_transform(other, scale=[2.0, 1.0, 1.0], rotation=_quat(45.0))
    before = (doc.by_uid(arm).parent, doc.animation().channels[arm]["rotation"].values.copy())
    with pytest.raises(OpError, match="keys"):
        doc.set_parent(arm, other)
    assert doc.by_uid(arm).parent == before[0]
    assert np.array_equal(doc.animation().channels[arm]["rotation"].values, before[1])


# -- clay-14 (clay-document-04) ---------------------------------------------------------------


def test_apply_pose_as_rest_reshapes_a_mesh_weighted_only_to_children_of_the_posed_bone() -> None:
    doc = bd.ClayDoc()
    shoulder = doc.add_bone(None).uid
    forearm = doc.add_bone(shoulder).uid
    doc.set_transform(forearm, translation=[1.0, 0.0, 0.0])
    skin = rig.Skin((forearm,), np.array([np.linalg.inv(doc.world_matrix(forearm))]))
    limb = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="Limb", mesh=weighted(bp.box()), skin=skin)
    ).uid
    doc.set_pose_mode(True)
    doc.auto_key = False
    doc.pose_bones({shoulder: {"rotation": _quat(90.0)}})
    overrides = doc.pose_overrides()
    obj = doc.by_uid(limb)
    expected = rig.skin_positions(
        obj.mesh.positions, obj.mesh.joints, obj.mesh.weights, doc.skin_palette(obj, overrides),
        doc._world_in(limb, doc._uid_map()),
    )
    assert not np.allclose(expected, obj.mesh.positions)  # the pose does carry the limb
    assert doc.apply_pose_as_rest()
    assert np.allclose(doc.by_uid(limb).mesh.positions, expected, atol=1e-5)


# -- clay-15 (clay-document-05) ---------------------------------------------------------------


def test_a_bone_with_a_zero_scale_axis_is_refused_by_name_instead_of_raising_singular_matrix() -> (
    None
):
    doc, bone, body = rigged_doc()
    with pytest.raises(OpError, match="Hip"):
        doc.set_transform(bone, scale=[0.0, 1.0, 1.0])
    # A document that already carries one (a file written before the refusal) still names it.
    doc.by_uid(bone).scale = np.array([0.0, 1.0, 1.0])
    with pytest.raises(OpError, match="Hip"):
        doc.skin_palette(doc.by_uid(body), {})
    with pytest.raises(OpError, match="Hip"):
        bd.to_model(doc)

    # ...and the viewport's frame-thread skinning step does not raise it.
    toasts: list[str] = []
    obj = doc.by_uid(body)
    doc.pose_mode = True
    fake = SimpleNamespace(
        _cache={body: SimpleNamespace(mesh=obj.mesh)},
        _skin_stamps={},
        _uid_index=lambda d: d._uid_map(),
        _preview_positions=lambda *a, **k: pytest.fail("a singular palette skinned a mesh"),
        _toast=toasts.append,
    )
    PoseOps._pose_skin(fake, doc)
    PoseOps._pose_skin(fake, doc)
    assert len(toasts) == 1 and "Hip" in toasts[0]


# -- clay-58 (clay-document-08) ---------------------------------------------------------------


def test_sliding_keys_past_the_end_of_the_timeline_clamps_instead_of_raising() -> None:
    doc = bd.ClayDoc()
    hip = doc.add_bone(None).uid
    doc.add_animation("Long")
    doc.set_key(hip, "translation", 0.0, [0, 0, 0])
    doc.set_key(hip, "translation", 599.0, [0, 1, 0])
    assert doc.move_bone_keys([(hip, 599.0)], 50.0)
    assert list(doc.animation().channels[hip]["translation"].times) == [0.0, 600.0]
    # already at the end: nothing to slide, and nothing raised
    assert not doc.move_bone_keys([(hip, 600.0)], 5.0)


# -- clay-60 (clay-document-11) ---------------------------------------------------------------


def test_join_objects_refuses_a_merge_that_takes_the_document_past_the_triangle_ceiling(
    monkeypatch,
) -> None:
    from realmspinner.kernels.mesh import glbimport

    doc = bd.ClayDoc()
    a = doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box())).uid
    b = doc.add_object(bd.Obj(uid=bd.new_uid(), name="B", mesh=bp.box())).uid
    monkeypatch.setattr(glbimport, "MAX_TRIANGLES", 30)
    steps = len(doc.history)
    before = doc.by_uid(a).mesh
    with pytest.raises(OpError, match="triangles"):
        doc.join_objects(a, bp.uv_sphere(0.5, 32, 16), [b])
    assert doc.by_uid(a).mesh is before and doc.by_uid(b) and len(doc.history) == steps
    # a merge that fits is still fine
    assert doc.join_objects(a, bp.box(), [b])


def test_join_objects_refuses_an_invalid_merged_mesh() -> None:
    import dataclasses

    doc = bd.ClayDoc()
    a = doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box())).uid
    b = doc.add_object(bd.Obj(uid=bd.new_uid(), name="B", mesh=bp.box())).uid
    good = bp.box()
    broken = dataclasses.replace(good, loops=good.loops + 10_000)
    with pytest.raises(OpError, match="not valid"):
        doc.join_objects(a, broken, [b])
    assert doc.by_uid(b)


# -- clay-62 (clay-document-13) ---------------------------------------------------------------


def test_the_weighted_mesh_refusal_names_the_real_reason() -> None:
    doc, bone, body = rigged_doc()
    with pytest.raises(OpError) as caught:
        doc.set_transform(body, translation=[1.0, 0.0, 0.0])
    text = str(caught.value)
    assert "ignored" not in text
    assert "bones" in text and "weights" in text
    # the exported vertices do carry the object's world matrix: the old reason was wrong
    doc.by_uid(body).translation = np.array([1.0, 0.0, 0.0])
    skinned = bd.to_model(doc)
    baked = skinned.meshes[-1][0].positions
    assert baked[:, 0].min() == pytest.approx(0.5, abs=1e-6)
    from realmspinner.kernels.mesh import document

    doc_text = document.ClayDoc._rest_to_world.__doc__ or ""
    assert "ignored by every reader" not in doc_text


# -- clay-82 (clay-panes-08) ------------------------------------------------------------------


def test_deleting_a_bone_drops_its_held_pose() -> None:
    doc = bd.ClayDoc()
    hip = doc.add_bone(None).uid
    doc.add_bone(hip)
    doc.set_pose_mode(True)
    doc.auto_key = False
    doc.pose_bones({hip: {"rotation": _quat(40.0)}})
    assert doc.pose_overrides()
    doc.remove_objects([hip])
    assert doc.pose_overrides() == {}
    from realmspinner.studio.modes.clay import ops

    assert not ops._can_apply_pose(doc)
    assert not doc.apply_pose_as_rest()
    # unmarking a bone does the same
    doc = bd.ClayDoc()
    hip = doc.add_bone(None).uid
    doc.set_pose_mode(True)
    doc.auto_key = False
    doc.pose_bones({hip: {"rotation": _quat(40.0)}})
    doc.set_bone([hip], False)
    assert doc.pose_overrides() == {}
