"""Opening a skinned GLB: joints become bones, weighted meshes keep their weights.

Tranche 2 of the 2026-10-13 Clay rig plan. Clay used to refuse a rigged file ("Clay has no
skinning"); every claim here fails on the tree before the change for that reason. The
acceptance test is the round trip: Clay's own rigid export reopens as the structure it was.
"""

from __future__ import annotations

import numpy as np
import pytest

from realmspinner.kernels.geom3d import glbwrite, gltf
from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import glbimport, rig
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as bp

from ._rigkit import skin_for, weighted


def _add(doc: bd.ClayDoc, name: str, **kw) -> int:
    kw.setdefault("mesh", bp.box())
    return doc.add_object(bd.Obj(uid=bd.new_uid(), name=name, **kw)).uid


def _round_trip(doc: bd.ClayDoc) -> bd.ClayDoc:
    return glbimport.glb_to_claydoc(glbwrite.write_glb(bd.to_model(doc)))


def _by_name(doc: bd.ClayDoc, name: str) -> bd.Obj:
    return next(o for o in doc.objects if o.name == name)


def _world_points(doc: bd.ClayDoc, obj: bd.Obj) -> np.ndarray:
    matrix = doc.world_matrix(obj.uid)
    points = np.hstack([obj.mesh.positions.astype("f8"), np.ones((len(obj.mesh.positions), 1))])
    return np.unique(np.round((matrix @ points.T).T[:, :3], 5), axis=0)


# -- Clay's own export reopens as the structure it was --------------------------------------


def test_a_rigid_export_reopens_with_the_same_bones_hierarchy_and_shapes_in_place() -> None:
    doc = bd.ClayDoc()
    hip = _add(doc, "Hip", bone=True, translation=(0, 1, 0))
    knee = _add(doc, "Knee", bone=True, mesh=bd._empty_mesh(), translation=(0, -0.5, 0), parent=hip)
    _add(doc, "Leg", parent=knee, translation=(0, -0.3, 0), scale=(1, 2, 1))
    _add(doc, "Free", translation=(3, 0, 0))

    out = _round_trip(doc)
    assert sorted(o.name for o in out.objects) == ["Free", "Hip", "Knee", "Leg"]
    assert [o.name for o in rig.bones(out.objects)] == ["Hip", "Knee"]
    assert out.by_uid(_by_name(out, "Knee").parent).name == "Hip"
    assert _by_name(out, "Hip").parent is None
    assert out.by_uid(_by_name(out, "Leg").parent).name == "Knee", "a rigid part is a bone's child"
    assert _by_name(out, "Free").parent is None and not _by_name(out, "Free").bone
    for name in ("Hip", "Knee", "Leg", "Free"):
        before = next(o for o in doc.objects if o.name == name)
        after = _by_name(out, name)
        assert bd._triangles_of(before.mesh) == bd._triangles_of(after.mesh), name
        if len(before.mesh.positions):
            assert np.allclose(_world_points(doc, before), _world_points(out, after), atol=1e-4), (
                name
            )
    assert out.rig_problems() == [] and out.notices == ()


def test_a_bones_own_mesh_is_folded_back_into_the_bone_not_made_a_second_object() -> None:
    doc = bd.ClayDoc()
    _add(doc, "Hip", bone=True)
    out = _round_trip(doc)
    assert [o.name for o in out.objects] == ["Hip"]
    assert out.objects[0].bone and bd._triangles_of(out.objects[0].mesh) == 12


def test_a_weighted_mesh_reopens_weighted_to_the_same_named_bones() -> None:
    doc = bd.ClayDoc()
    a = _add(doc, "A", bone=True, mesh=bd._empty_mesh())
    b = _add(doc, "B", bone=True, mesh=bd._empty_mesh(), translation=(0, 1, 0))
    mesh = weighted(bp.box(), 2)
    weights = mesh.weights.copy()
    joints = mesh.joints.copy()
    joints[:, 0] = np.arange(len(joints)) % 2
    joints[:, 1] = 1 - joints[:, 0]
    weights[:, 0], weights[:, 1] = 0.75, 0.25
    mesh = bm.Mesh(
        positions=mesh.positions,
        loops=mesh.loops,
        starts=mesh.starts,
        material=mesh.material,
        smooth=mesh.smooth,
        uv=mesh.uv,
        joints=joints,
        weights=weights,
    )
    _add(doc, "Body", mesh=mesh, skin=skin_for(a, b))

    out = _round_trip(doc)
    body = _by_name(out, "Body")
    assert body.skin is not None and body.parent is None
    assert [out.by_uid(j).name for j in body.skin.joints] == ["A", "B"]

    # Per source position, the (joint name, weight) pairs survive the export's vertex split.
    def influence(document: bd.ClayDoc, obj: bd.Obj) -> dict[tuple, set]:
        names = {j: document.by_uid(j).name for j in obj.skin.joints}
        table: dict[tuple, set] = {}
        for p, js, ws in zip(obj.mesh.positions, obj.mesh.joints, obj.mesh.weights, strict=True):
            row = frozenset(
                (names[obj.skin.joints[j]], round(float(w), 4))
                for j, w in zip(js, ws, strict=True)
                if w > 0
            )
            table.setdefault(tuple(np.round(p, 5)), set()).add(row)
        return table

    assert influence(out, body) == influence(
        doc, doc.by_uid(next(o.uid for o in doc.objects if o.name == "Body"))
    )
    # Stored inverse bind matrices are inv(world rest).
    for joint, matrix in zip(body.skin.joints, body.skin.inverse_bind, strict=True):
        assert np.allclose(matrix, np.linalg.inv(out.world_matrix(joint)), atol=1e-6)
    assert out.rig_problems() == []


# -- what other tools write ---------------------------------------------------------------


def _hand_model(*, knee_rotation=(0.0, 0.0, 0.0, 1.0), skin_extra_joint: bool = False):
    """Armature -> Hip -> Knee, a quad skinned half-and-half to the two, at the scene root."""
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype="f4")
    prim = gltf.Primitive(
        positions=positions,
        indices=np.array([0, 1, 2, 1, 3, 2], dtype="u4"),
        normals=np.tile(np.array([0, 0, 1], dtype="f4"), (4, 1)),
        joints=np.array([[0, 1, 0, 0]] * 4, dtype="i4"),
        weights=np.array([[0.5, 0.5, 0, 0]] * 4, dtype="f4"),
    )
    nodes = [
        gltf.Node(name="Armature", translation=m3.vec3(0, 1, 0), children=[1]),
        gltf.Node(name="Hip", translation=m3.vec3(0, 1, 0), children=[2]),
        gltf.Node(name="Knee", translation=m3.vec3(0, 1, 0), rotation=np.array(knee_rotation)),
        gltf.Node(name="Body", mesh=0, skin=0),
    ]
    # Bind pose: Hip at y = 2 and Knee at y = 3 in world space, whatever the knee is turned to now.
    bind = np.stack([_translation(0, -2, 0), _translation(0, -3, 0)])
    return gltf.Model(nodes, [0, 3], [[prim]], [gltf.Skin(joints=[1, 2], inverse_bind=bind)])


def _translation(x: float, y: float, z: float) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, 3] = [x, y, z]
    return matrix


def _import(model: gltf.Model) -> bd.ClayDoc:
    return glbimport.glb_to_claydoc(glbwrite.write_glb(model))


def test_a_blender_shaped_rig_skips_the_armature_empty_and_places_bones_in_world_space() -> None:
    out = _import(_hand_model())
    assert sorted(o.name for o in out.objects) == ["Body", "Hip", "Knee"]
    hip, knee = _by_name(out, "Hip"), _by_name(out, "Knee")
    assert hip.bone and hip.parent is None, "the Armature empty is not an object"
    assert knee.parent == hip.uid
    assert np.allclose(out.world_matrix(hip.uid)[:3, 3], [0, 2, 0])
    assert np.allclose(out.world_matrix(knee.uid)[:3, 3], [0, 3, 0])
    body = _by_name(out, "Body")
    assert (
        body.skin is not None and np.allclose(body.translation, 0) and body.mesh.joints is not None
    )
    # Skinned at the bind pose: the points are exactly the file's.
    assert np.allclose(
        np.unique(body.mesh.positions, axis=0),
        np.unique(_hand_model().meshes[0][0].positions, axis=0),
    )


def test_a_file_saved_in_a_pose_is_baked_at_that_pose() -> None:
    quarter_turn = (0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4))
    out = _import(_hand_model(knee_rotation=quarter_turn))
    body = _by_name(out, "Body")
    # Half the weight follows the knee, which turned 90 degrees about Z around its pivot at
    # (0, 3, 0); the other half follows the hip, which has not moved.
    pivot = np.array([0.0, 3.0, 0.0])

    def pose(v: np.ndarray) -> np.ndarray:
        turned = pivot + np.array([-(v[1] - pivot[1]), v[0] - pivot[0], v[2] - pivot[2]])
        return 0.5 * v + 0.5 * turned

    raw = _hand_model().meshes[0][0].positions.astype("f8")
    expected = np.unique(np.round([pose(v) for v in raw], 5), axis=0)
    assert np.allclose(np.unique(np.round(body.mesh.positions, 5), axis=0), expected, atol=1e-4)
    knee = _by_name(out, "Knee")
    assert np.allclose(out.world_matrix(knee.uid)[:3, 3], [0, 3, 0])
    # The stored inverse bind matrices describe the rest pose the bones were imported in.
    for joint, matrix in zip(body.skin.joints, body.skin.inverse_bind, strict=True):
        assert np.allclose(matrix, np.linalg.inv(out.world_matrix(joint)), atol=1e-6)


def test_more_than_the_bone_cap_of_joints_is_refused_by_name() -> None:
    count = rig.MAX_BONES + 1
    nodes = [
        gltf.Node(name=f"J{i}", children=[i + 1] if i < count - 1 else []) for i in range(count)
    ]
    nodes.append(gltf.Node(name="Body", mesh=0, skin=0))
    prim = gltf.Primitive(
        positions=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="f4"),
        indices=np.array([0, 1, 2], dtype="u4"),
        joints=np.zeros((3, 4), dtype="i4"),
        weights=np.tile(np.array([1, 0, 0, 0], dtype="f4"), (3, 1)),
    )
    skin = gltf.Skin(joints=list(range(count)), inverse_bind=np.tile(np.eye(4), (count, 1, 1)))
    model = gltf.Model(nodes, [0, count], [[prim]], [skin])
    with pytest.raises(el.OpError, match="skeleton has 65 joints"):
        _import(model)


def test_a_sheared_bone_is_refused_by_name() -> None:
    model = _hand_model()
    # A non-joint node with a non-uniform scale between a bone and a turned child makes the
    # child's matrix relative to its parent bone sheared.
    model.nodes[1].children = [4]
    model.nodes.append(gltf.Node(name="Squash", scale=m3.vec3(1, 3, 1), children=[2]))
    model.nodes[2].rotation = np.array([0.0, 0.0, np.sin(np.pi / 8), np.cos(np.pi / 8)])
    with pytest.raises(el.OpError, match="Knee"):
        _import(model)


def test_an_unskinned_mesh_under_a_joint_becomes_a_child_of_that_bone() -> None:
    model = _hand_model()
    prim = gltf.Primitive(
        positions=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="f4"),
        indices=np.array([0, 1, 2], dtype="u4"),
    )
    model.meshes.append([prim])
    model.nodes.append(gltf.Node(name="Sword", mesh=1, translation=m3.vec3(0, 0, 1)))
    model.nodes[2].children = [len(model.nodes) - 1]
    out = _import(model)
    sword = _by_name(out, "Sword")
    assert sword.parent == _by_name(out, "Knee").uid and not sword.bone
    assert np.allclose(out.world_matrix(sword.uid)[:3, 3], [0, 3, 1])


def test_two_vertices_at_one_position_with_different_weights_stay_two_points() -> None:
    model = _hand_model()
    prim = model.meshes[0][0]
    positions = np.vstack([prim.positions, prim.positions[:1]])
    joints = np.vstack([prim.joints, [[1, 0, 0, 0]]])
    weights = np.vstack([prim.weights, [[1, 0, 0, 0]]])
    model.meshes[0][0] = gltf.Primitive(
        positions=positions.astype("f4"),
        indices=np.array([0, 1, 2, 1, 3, 2, 4, 1, 2], dtype="u4"),
        joints=joints,
        weights=weights.astype("f4"),
    )
    body = _by_name(_import(model), "Body")
    at_origin = np.all(body.mesh.positions == 0, axis=1)
    assert at_origin.sum() == 2, "same place, different weights"


def test_a_joint_the_scene_does_not_place_is_refused() -> None:
    model = _hand_model()
    model.nodes.append(gltf.Node(name="Stray"))  # in no scene
    model.skins[0] = gltf.Skin(
        joints=[1, 2, len(model.nodes) - 1], inverse_bind=np.tile(np.eye(4), (3, 1, 1))
    )
    prim = model.meshes[0][0]
    model.meshes[0][0] = gltf.Primitive(
        positions=prim.positions, indices=prim.indices, joints=prim.joints, weights=prim.weights
    )
    with pytest.raises(el.OpError, match="Stray"):
        _import(model)


def test_a_skeleton_with_no_meshes_opens_as_bones_alone() -> None:
    nodes = [
        gltf.Node(name="Hip", children=[1]),
        gltf.Node(name="Knee", translation=m3.vec3(0, -1, 0)),
    ]
    skin = gltf.Skin(joints=[0, 1], inverse_bind=np.tile(np.eye(4), (2, 1, 1)))
    out = _import(gltf.Model(nodes, [0], [], [skin]))
    assert [(o.name, o.bone) for o in out.objects] == [("Hip", True), ("Knee", True)]


def test_an_imported_rig_saves_and_reopens_as_a_clay_document() -> None:
    from realmspinner.kernels.mesh import serialize

    out = _import(_hand_model())
    again = serialize.read_rblk(serialize.rblk_bytes(out))
    assert [o.name for o in again.objects] == [o.name for o in out.objects]
    assert again.rig_problems() == []
