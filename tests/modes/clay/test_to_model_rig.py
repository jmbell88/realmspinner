"""``to_model`` of a rigged Clay document: one skin, bones as nodes, skinned meshes at the roots.

Tranche 2 of the 2026-10-13 Clay rig plan. The shape is Blender's own: a skinned node's
transform is ignored by the glTF spec, so the geometry of a shape that moves with a bone goes to
an identity scene-root node with its vertices baked into world rest space. Each claim fails on
the tree before the change: ``to_model`` returned no skins and left every shape on its own node.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from realmspinner.kernels.geom3d import glbwrite, gltf
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as bp

from ._rigkit import skin_for, weighted


def _add(doc: bd.ClayDoc, name: str, **kw) -> int:
    kw.setdefault("mesh", bp.box())
    return doc.add_object(bd.Obj(uid=bd.new_uid(), name=name, **kw)).uid


def _named(model: gltf.Model, name: str) -> list[int]:
    return [i for i, n in enumerate(model.nodes) if n.name == name]


def test_a_document_without_bones_exports_exactly_as_it_always_did() -> None:
    doc = bd.ClayDoc()
    parent = _add(doc, "A")
    _add(doc, "B", parent=parent)
    model = bd.to_model(doc)
    assert model.skins == [] and [n.skin for n in model.nodes] == [None, None]
    assert all(p.joints is None and p.weights is None for prims in model.meshes for p in prims)
    first = glbwrite.write_glb(model)
    assert (
        hashlib.sha256(first).hexdigest() == hashlib.sha256(glbwrite.write_glb(model)).hexdigest()
    )


def test_bones_are_ordinary_nodes_listed_in_document_order_in_one_skin() -> None:
    doc = bd.ClayDoc()
    hip = _add(doc, "Hip", bone=True, translation=(0, 1, 0))
    knee = _add(doc, "Knee", bone=True, translation=(0, -0.5, 0), parent=hip)
    model = bd.to_model(doc)
    assert len(model.skins) == 1
    hip_node, knee_node = _named(model, "Hip")[0], _named(model, "Knee")[0]
    assert model.skins[0].joints == [hip_node, knee_node]
    assert knee_node in model.nodes[hip_node].children
    # inverse bind = inv(world rest): the knee sits at y = 0.5 in world space.
    assert np.allclose(model.skins[0].inverse_bind[0], np.linalg.inv(doc.world_matrix(hip)))
    assert np.allclose(model.skins[0].inverse_bind[1], np.linalg.inv(doc.world_matrix(knee)))
    assert np.allclose(doc.world_matrix(knee)[:3, 3], [0, 0.5, 0])


def test_a_shape_under_a_bone_becomes_an_identity_root_node_skinned_rigidly_at_weight_one() -> None:
    doc = bd.ClayDoc()
    hip = _add(doc, "Hip", bone=True, translation=(0, 1, 0))
    _add(doc, "Leg", parent=hip, translation=(0, -1, 0))
    model = bd.to_model(doc)

    skinned = [i for i, n in enumerate(model.nodes) if n.skin is not None]
    assert len(skinned) == 2, "the bone's own box and the leg: both move with the hip"
    for i in skinned:
        node = model.nodes[i]
        assert i in model.roots
        assert np.allclose(node.translation, 0) and np.allclose(node.rotation, [0, 0, 0, 1])
        assert np.allclose(node.scale, 1) and node.skin == 0 and node.mesh is not None
    leg_node = model.nodes[next(i for i in skinned if model.nodes[i].name == "Leg")]
    prim = model.meshes[leg_node.mesh][0]
    assert prim.joints.shape == (len(prim.positions), 4)
    assert (prim.joints[:, 0] == 0).all() and (prim.joints[:, 1:] == 0).all()
    assert np.allclose(prim.weights, [1, 0, 0, 0])
    # World rest space: the leg's box sits at the origin (hip at y=1, leg at y=-1 under it).
    assert np.allclose(prim.positions.mean(axis=0), [0, 0, 0], atol=1e-6)
    # The original nodes stay, without a mesh, so every other world matrix is unchanged.
    original = _named(model, "Leg")[0]
    assert model.nodes[original].mesh is None and model.nodes[original].skin is None


def test_a_free_shape_keeps_its_node_and_its_mesh() -> None:
    doc = bd.ClayDoc()
    _add(doc, "Hip", bone=True)
    free = _add(doc, "Free", translation=(5, 0, 0))
    model = bd.to_model(doc)
    node = model.nodes[_named(model, "Free")[0]]
    assert node.mesh is not None and node.skin is None
    assert np.allclose(node.translation, [5, 0, 0])
    assert free


def test_a_weighted_object_has_its_joint_indices_remapped_onto_the_document_skin() -> None:
    doc = bd.ClayDoc()
    a = _add(doc, "A", bone=True)
    b = _add(doc, "B", bone=True, translation=(0, 1, 0))
    mesh = weighted(bp.box(), 2)  # joint 0 at weight 1; joint 1 present with weight 0
    _add(doc, "Body", mesh=mesh, skin=skin_for(b, a))  # the skin lists B first
    model = bd.to_model(doc)
    body = model.nodes[
        next(i for i, n in enumerate(model.nodes) if n.skin is not None and n.name == "Body")
    ]
    prim = model.meshes[body.mesh][0]
    # Skin joint 0 is B (document joint 1) and skin joint 1 is A (document joint 0).
    assert (prim.joints[:, 0] == 1).all() and (prim.joints[:, 1] == 0).all()
    assert np.allclose(prim.weights[:, 0], 1.0)


def test_a_weighted_object_is_baked_into_world_rest_space() -> None:
    doc = bd.ClayDoc()
    bone = _add(doc, "Hip", bone=True)
    body = _add(
        doc,
        "Body",
        mesh=weighted(bp.box()),
        skin=skin_for(bone),
        translation=(0, 2, 0),
        scale=(2, 2, 2),
    )
    model = bd.to_model(doc)
    node = model.nodes[
        next(i for i, n in enumerate(model.nodes) if n.skin is not None and n.name == "Body")
    ]
    positions = model.meshes[node.mesh][0].positions
    lo, hi = positions.min(axis=0), positions.max(axis=0)
    assert np.allclose(lo, [-1, 1, -1]) and np.allclose(hi, [1, 3, 1])
    assert body


def test_a_hidden_bone_and_its_ancestors_still_export_as_joints() -> None:
    doc = bd.ClayDoc()
    group = _add(doc, "Group")
    hip = _add(doc, "Hip", bone=True, parent=group)
    doc.set_visibility({hip: False, group: False})
    model = bd.to_model(doc)
    assert model.skins and len(model.skins[0].joints) == 1
    joint = model.nodes[model.skins[0].joints[0]]
    assert joint.name == "Hip" and joint.mesh is None, "hidden: a joint with no mesh"
    assert _named(model, "Group"), "its ancestor survives so the joint keeps its world matrix"
    assert not [n for n in model.nodes if n.skin is not None], "nothing visible to skin"


def test_a_hidden_shape_under_a_bone_is_not_exported() -> None:
    doc = bd.ClayDoc()
    hip = _add(doc, "Hip", bone=True)
    leg = _add(doc, "Leg", parent=hip)
    doc.set_visibility({leg: False})
    model = bd.to_model(doc)
    assert [n.name for n in model.nodes if n.skin is not None] == ["Hip"]


def test_the_written_glb_loads_back_with_one_skin_and_skinned_primitives() -> None:
    doc = bd.ClayDoc()
    hip = _add(doc, "Hip", bone=True, translation=(0, 1, 0))
    _add(doc, "Leg", parent=hip, translation=(0, -1, 0))
    out = gltf.load(glbwrite.write_glb(bd.to_model(doc)))
    assert len(out.skins) == 1 and len(out.skins[0].joints) == 1
    skinned = [n for n in out.nodes if n.skin is not None]
    assert len(skinned) == 2
    for node in skinned:
        assert out.meshes[node.mesh][0].joints is not None


def test_an_inconsistent_rig_refuses_to_export_by_name() -> None:
    doc = bd.ClayDoc()
    bone = _add(doc, "Hip", bone=True)
    _add(doc, "Body", mesh=weighted(bp.box()), skin=skin_for(bone))
    doc.by_uid(bone).bone = False
    with pytest.raises(ValueError, match="rig is not consistent"):
        bd.to_model(doc)


def test_a_smooth_or_textured_weighted_mesh_carries_weights_through_the_vertex_split() -> None:
    # Flat shading splits vertices per corner; weights must follow each emitted vertex to
    # the source vertex it came from, not stay one row per source vertex.
    doc = bd.ClayDoc()
    first = _add(doc, "A", bone=True)
    second = _add(doc, "B", bone=True)
    mesh = weighted(bp.box(), 2)
    joints = mesh.joints.copy()
    joints[:, 0] = np.arange(len(joints)) % 2  # alternate joints by source vertex
    mesh = bm.Mesh(
        positions=mesh.positions,
        loops=mesh.loops,
        starts=mesh.starts,
        material=mesh.material,
        smooth=np.zeros(bm.face_count(mesh), dtype=bool),
        uv=None,
        joints=joints,
        weights=mesh.weights,
    )
    _add(doc, "Body", mesh=mesh, skin=skin_for(first, second))
    model = bd.to_model(doc)
    node = next(n for n in model.nodes if n.skin is not None and n.name == "Body")
    prim = model.meshes[node.mesh][0]
    assert len(prim.positions) == 24, "a flat box splits to one vertex per corner"
    for k in range(len(prim.positions)):
        source = int(np.flatnonzero((mesh.positions == prim.positions[k]).all(axis=1))[0])
        assert prim.joints[k, 0] == joints[source, 0], k
