"""A Clay rig's animations in and out of a GLB.

Tranche 3 of the 2026-10-13 Clay rig plan. ``to_model`` writes the document's animations, a posed
export writes one frame of them into the joint nodes, and ``rigimport`` reads a file's channels
back onto its bones -- carried across the nodes that are not bones, resampled when cubic, and
refused by name past Clay's caps. Each claim fails on the tree before the change: nothing wrote or
read ``"animations"``.
"""

from __future__ import annotations

import numpy as np
import pytest

from realmspinner.kernels.geom3d import glbwrite, gltf
from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.kernels.mesh import anim, glbimport
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import primitives as bp

Q0 = [0.0, 0.0, 0.0, 1.0]
Q90 = [0.0, 0.0, float(np.sin(np.pi / 4)), float(np.cos(np.pi / 4))]


def _add(doc: bd.ClayDoc, name: str, **kw) -> int:
    kw.setdefault("mesh", bp.box())
    return doc.add_object(bd.Obj(uid=bd.new_uid(), name=name, **kw)).uid


def _animated() -> tuple[bd.ClayDoc, int, int]:
    doc = bd.ClayDoc()
    hip = _add(doc, "Hip", bone=True, translation=(0, 1, 0))
    knee = _add(doc, "Knee", bone=True, parent=hip, translation=(0, -0.5, 0))
    _add(doc, "Leg", parent=knee)
    doc.add_animation("Walk")
    doc.set_key(hip, "translation", 0.0, [0, 1, 0])
    doc.set_key(hip, "translation", 1.0, [0, 1.5, 0])
    doc.set_key(knee, "rotation", 0.0, Q0)
    doc.set_key(knee, "rotation", 0.5, Q90)
    doc.set_key(knee, "rotation", 1.0, Q0)
    doc.add_animation("Pose")
    doc.set_key(knee, "rotation", 0.0, Q90, animation=1)
    doc.set_animation_loop(1, False)
    doc.set_animation_interpolation(1, "step")
    return doc, hip, knee


def _round_trip(doc: bd.ClayDoc) -> bd.ClayDoc:
    return glbimport.glb_to_claydoc(glbwrite.write_glb(bd.to_model(doc)))


def _bone(doc: bd.ClayDoc, name: str) -> bd.Obj:
    return next(o for o in doc.objects if o.name == name and o.bone)


# -- writing -----------------------------------------------------------------------------------


def test_to_model_writes_one_glb_animation_per_clay_animation_with_node_targets() -> None:
    doc, hip, knee = _animated()
    model = bd.to_model(doc)
    assert [a.name for a in model.animations] == ["Walk", "Pose"]
    walk, pose = model.animations
    targets = {(c.node, c.path) for c in walk.channels}
    hip_node = next(i for i, n in enumerate(model.nodes) if n.name == "Hip" and n.skin is None)
    knee_node = next(i for i, n in enumerate(model.nodes) if n.name == "Knee" and n.skin is None)
    assert targets == {(hip_node, "translation"), (knee_node, "rotation")}
    assert {c.interpolation for c in walk.channels} == {"LINEAR"}
    assert {c.interpolation for c in pose.channels} == {"STEP"}
    assert walk.extras == {} and pose.extras == {"loop": False}


def test_a_document_without_animations_writes_none_and_a_static_model_never_does() -> None:
    doc, _hip, _knee = _animated()
    assert bd.to_model(doc, skinned=False).animations == []
    plain = bd.ClayDoc()
    _add(plain, "Box")
    assert bd.to_model(plain).animations == []


def test_place_carries_a_root_bones_translation_and_scale_keys_like_its_node() -> None:
    doc, hip, _knee = _animated()
    place = np.eye(4)
    place[:3, 3] = [10.0, 0.0, 0.0]
    placed = bd.to_model(doc, place=place)
    plain = bd.to_model(doc)
    a = next(c for c in plain.animations[0].channels if c.path == "translation")
    b = next(c for c in placed.animations[0].channels if c.path == "translation")
    assert np.allclose(b.values - a.values, [10, 0, 0]), "the root's keys are in placed space"
    rot_a = next(c for c in plain.animations[0].channels if c.path == "rotation")
    rot_b = next(c for c in placed.animations[0].channels if c.path == "rotation")
    assert np.allclose(rot_a.values, rot_b.values), "a child's keys are relative and unmoved"


def test_the_written_glb_round_trips_through_the_loader_with_its_keys() -> None:
    doc, _hip, _knee = _animated()
    out = gltf.load(glbwrite.write_glb(bd.to_model(doc)))
    assert [a.name for a in out.animations] == ["Walk", "Pose"]
    assert out.animations[1].extras == {"loop": False}


def test_a_posed_export_writes_the_pose_into_the_joint_nodes_and_no_animations() -> None:
    doc, hip, knee = _animated()
    doc.set_active_animation(0)
    rest = bd.to_model(doc)
    posed = bd.to_model(doc, at_time=0.5)
    assert posed.animations == []
    knee_node = next(i for i, n in enumerate(posed.nodes) if n.name == "Knee" and n.skin is None)
    assert np.allclose(posed.nodes[knee_node].rotation, Q90, atol=1e-6)
    assert np.allclose(rest.nodes[knee_node].rotation, Q0), "at rest the bone is unturned"
    hip_node = next(i for i, n in enumerate(posed.nodes) if n.name == "Hip" and n.skin is None)
    assert np.allclose(posed.nodes[hip_node].translation, [0, 1.25, 0])
    # The skin still describes the REST pose: the meshes are baked there and the joints
    # carry the pose.
    assert np.allclose(posed.skins[0].inverse_bind, rest.skins[0].inverse_bind)


def test_a_posed_export_of_a_document_with_no_selected_animation_is_the_rest_pose() -> None:
    doc, _hip, _knee = _animated()
    doc.set_active_animation(None)
    posed = bd.to_model(doc, at_time=0.5)
    rest = bd.to_model(doc)
    for a, b in zip(posed.nodes, rest.nodes, strict=True):
        assert np.allclose(a.rotation, b.rotation) and np.allclose(a.translation, b.translation)


# -- reading: the round trip --------------------------------------------------------------------


def test_clays_own_animations_come_back_with_their_keys_names_loop_and_interpolation() -> None:
    doc, hip, knee = _animated()
    out = _round_trip(doc)
    assert [a.name for a in out.rig_state.animations] == ["Walk", "Pose"]
    walk, pose = out.animation(0), out.animation(1)
    assert walk.loop and not pose.loop
    assert walk.interpolation == "linear" and pose.interpolation == "step"
    new_hip, new_knee = _bone(out, "Hip"), _bone(out, "Knee")
    src_knee = doc.animation(0).channels[knee]["rotation"]
    got = walk.channels[new_knee.uid]["rotation"]
    assert np.allclose(got.times, src_knee.times) and np.allclose(got.values, src_knee.values)
    assert np.allclose(walk.channels[new_hip.uid]["translation"].values[-1], [0, 1.5, 0])
    assert out.active_animation == 0
    assert out.rig_problems() == []


def test_the_frame_rate_is_guessed_from_the_smallest_gap_between_keys() -> None:
    doc, _hip, knee = _animated()
    # Keys every 1/30 s: the guess lands on 30, not on the default.
    doc.add_animation("Fine")
    for i in range(4):
        doc.set_key(knee, "rotation", i / 30.0, Q0 if i % 2 == 0 else Q90)
    assert _round_trip(doc).rig_state.fps == 30


def test_an_animation_that_names_nothing_gets_a_default_name_and_names_stay_unique() -> None:
    doc, hip, _knee = _animated()
    model = bd.to_model(doc)
    model.animations[0].name = ""
    model.animations[1].name = ""
    out = glbimport.glb_to_claydoc(glbwrite.write_glb(model))
    assert [a.name for a in out.rig_state.animations] == ["Animation", "Animation.001"]


# -- reading: what other tools write -----------------------------------------------------------


def _blender_shaped(animation: gltf.Animation) -> bytes:
    """Armature(0,1,0, scaled) -> Hip -> Knee, a quad skinned to both, one animation."""
    prim = gltf.Primitive(
        positions=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype="f4"),
        indices=np.array([0, 1, 2, 1, 3, 2], dtype="u4"),
        joints=np.array([[0, 1, 0, 0]] * 4, dtype="i4"),
        weights=np.array([[0.5, 0.5, 0, 0]] * 4, dtype="f4"),
    )
    nodes = [
        gltf.Node(name="Armature", translation=m3.vec3(0, 1, 0), children=[1]),
        gltf.Node(name="Hip", translation=m3.vec3(0, 1, 0), children=[2]),
        gltf.Node(name="Knee", translation=m3.vec3(0, 1, 0)),
        gltf.Node(name="Body", mesh=0, skin=0),
    ]
    bind = np.stack([np.eye(4), np.eye(4)])
    bind[0][:3, 3] = [0, -2, 0]
    bind[1][:3, 3] = [0, -3, 0]
    model = gltf.Model(
        nodes,
        [0, 3],
        [[prim]],
        [gltf.Skin(joints=[1, 2], inverse_bind=bind)],
        animations=[animation],
    )
    return glbwrite.write_glb(model)


def test_keys_on_a_joint_under_a_non_joint_node_are_carried_into_the_bones_frame() -> None:
    # Hip's glTF parent is the Armature empty (translated up one); its bone has no parent,
    # so its keys are world-relative: the Armature's offset must be added to every key.
    times = np.array([0.0, 1.0])
    translation = gltf.AnimChannel(
        1, "translation", times, np.array([[0, 1, 0], [0, 3, 0]]), "LINEAR"
    )
    out = glbimport.glb_to_claydoc(_blender_shaped(gltf.Animation("Bob", [translation])))
    hip = _bone(out, "Hip")
    channels = out.animation(0).channels[hip.uid]
    assert np.allclose(channels["translation"].values, [[0, 2, 0], [0, 4, 0]]), "Armature at y=1"
    assert set(channels) == {"translation", "rotation", "scale"}, "the fold keys all three"
    assert np.allclose(channels["rotation"].values, [Q0, Q0])


def test_keys_on_a_joint_under_a_joint_pass_through_unchanged() -> None:
    times = np.array([0.0, 1.0])
    rotation = gltf.AnimChannel(2, "rotation", times, np.array([Q0, Q90]), "LINEAR")
    out = glbimport.glb_to_claydoc(_blender_shaped(gltf.Animation("Bend", [rotation])))
    channels = out.animation(0).channels[_bone(out, "Knee").uid]
    assert set(channels) == {"rotation"}, "no fold, so only what the file animates"
    assert np.allclose(channels["rotation"].values, [Q0, Q90])


def test_channels_on_something_that_is_not_a_bone_are_dropped_and_said() -> None:
    times = np.array([0.0, 1.0])
    stray = gltf.AnimChannel(3, "translation", times, np.array([[0, 0, 0], [1, 0, 0]]))
    rotation = gltf.AnimChannel(2, "rotation", times, np.array([Q0, Q90]))
    out = glbimport.glb_to_claydoc(_blender_shaped(gltf.Animation("A", [stray, rotation])))
    assert len(out.animation(0).channels) == 1
    assert any("not a bone" in notice for notice in out.notices)


def test_a_cubic_spline_channel_is_resampled_to_linear_keys_and_the_notice_says_so() -> None:
    times = np.array([0.0, 1.0])
    values = np.zeros((6, 3))
    values[4] = [0, 1, 0]
    cubic = gltf.AnimChannel(2, "translation", times, values, "CUBICSPLINE")
    out = glbimport.glb_to_claydoc(_blender_shaped(gltf.Animation("Smooth", [cubic])))
    channel = out.animation(0).channels[_bone(out, "Knee").uid]["translation"]
    assert len(channel) > 2 and np.allclose(channel.values[-1], [0, 1, 0])
    assert out.animation(0).interpolation == "linear"
    assert any("CUBICSPLINE" in notice and "linear keys" in notice for notice in out.notices)


def test_a_file_with_no_animations_imports_with_none() -> None:
    doc, _hip, _knee = _animated()
    while doc.rig_state.animations:
        doc.delete_animation(0)
    out = _round_trip(doc)
    assert out.rig_state.animations == () and out.active_animation is None


# -- the caps are refusals by name ---------------------------------------------------------------


def test_more_keys_on_one_channel_than_clay_holds_is_refused_naming_the_bone() -> None:
    n = anim.MAX_KEYS_PER_CHANNEL + 1
    times = np.arange(n) * 0.001
    big = gltf.AnimChannel(2, "rotation", times, np.tile(Q0, (n, 1)))
    with pytest.raises(el.OpError, match="'Knee'.*4,096"):
        glbimport.glb_to_claydoc(_blender_shaped(gltf.Animation("Dense", [big])))


def test_a_clip_longer_than_the_cap_is_refused() -> None:
    times = np.array([0.0, anim.MAX_ANIMATION_SECONDS + 10])
    long = gltf.AnimChannel(2, "rotation", times, np.array([Q0, Q90]))
    with pytest.raises(el.OpError, match="seconds"):
        glbimport.glb_to_claydoc(_blender_shaped(gltf.Animation("Long", [long])))
