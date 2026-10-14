"""The GLB writer's skins: joints, weights and column-major inverse bind matrices.

Tranche 2 of the 2026-10-13 Clay rig plan removed the writer's old refusal of a skin. Verified
the way the rest of the writer is -- by reading the bytes back through the independent loader --
plus the container facts a tolerant loader cannot see (component types, column-major layout).
Every claim fails on the tree before the change: ``write_glb`` raised "does not write a skin".
"""

from __future__ import annotations

import numpy as np
import pytest

from realmspinner.kernels.geom3d import glbwrite, gltf
from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.kernels.geom3d.glbio import read_glb


def _skinned_model(
    *, joints: np.ndarray | None = None, weights: np.ndarray | None = None
) -> gltf.Model:
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype="f4")
    joints = np.array([[0, 1, 0, 0]] * 4, dtype="i4") if joints is None else joints
    weights = np.array([[0.75, 0.25, 0, 0]] * 4, dtype="f4") if weights is None else weights
    prim = gltf.Primitive(
        positions=positions,
        indices=np.array([0, 1, 2, 1, 3, 2], dtype="u4"),
        normals=np.tile(np.array([0, 0, 1], dtype="f4"), (4, 1)),
        joints=joints,
        weights=weights,
    )
    moved = np.eye(4)
    moved[:3, 3] = [1.0, 2.0, 3.0]
    nodes = [
        gltf.Node(name="root", children=[1]),
        gltf.Node(name="hip", translation=m3.vec3(0, 1, 0), children=[2]),
        gltf.Node(name="knee", translation=m3.vec3(0, 1, 0)),
        gltf.Node(name="body", mesh=0, skin=0),
    ]
    skin = gltf.Skin(joints=[1, 2], inverse_bind=np.stack([np.eye(4), moved]))
    return gltf.Model(nodes, [0, 3], [[prim]], [skin])


def test_a_skin_round_trips_through_the_loader() -> None:
    model = _skinned_model()
    out = gltf.load(glbwrite.write_glb(model))
    assert len(out.skins) == 1
    assert out.skins[0].joints == [1, 2]
    assert np.allclose(out.skins[0].inverse_bind, model.skins[0].inverse_bind)
    assert out.nodes[3].skin == 0 and out.nodes[3].mesh == 0
    prim = out.meshes[0][0]
    assert np.array_equal(prim.joints, model.meshes[0][0].joints)
    assert np.array_equal(prim.weights, model.meshes[0][0].weights)


def test_joints_are_unsigned_bytes_and_weights_are_floats_in_the_json() -> None:
    doc, _ = read_glb(glbwrite.write_glb(_skinned_model()))
    attrs = doc["meshes"][0]["primitives"][0]["attributes"]
    assert doc["accessors"][attrs["JOINTS_0"]]["componentType"] == 5121
    assert doc["accessors"][attrs["JOINTS_0"]]["type"] == "VEC4"
    assert doc["accessors"][attrs["WEIGHTS_0"]]["componentType"] == 5126
    skin = doc["skins"][0]
    assert skin["joints"] == [1, 2]
    assert doc["accessors"][skin["inverseBindMatrices"]]["type"] == "MAT4"
    assert doc["nodes"][3]["skin"] == 0


def test_inverse_bind_matrices_are_written_column_major() -> None:
    doc, binary = read_glb(glbwrite.write_glb(_skinned_model()))
    accessor = doc["accessors"][doc["skins"][0]["inverseBindMatrices"]]
    view = doc["bufferViews"][accessor["bufferView"]]
    raw = np.frombuffer(binary, dtype="<f4", count=32, offset=view["byteOffset"]).reshape(2, 16)
    # The second matrix translates by (1, 2, 3): column-major puts that at elements 12..14.
    assert raw[1, 12:15].tolist() == [1.0, 2.0, 3.0]
    assert raw[1, 3] == 0.0 and raw[1, 15] == 1.0


def test_a_model_without_skins_writes_no_skin_keys_at_all() -> None:
    model = _skinned_model()
    model.skins = []
    model.nodes[3].skin = None
    prim = model.meshes[0][0]
    model.meshes[0][0] = gltf.Primitive(
        positions=prim.positions, indices=prim.indices, normals=prim.normals
    )
    doc, _ = read_glb(glbwrite.write_glb(model))
    assert "skins" not in doc and "skin" not in doc["nodes"][3]
    attrs = doc["meshes"][0]["primitives"][0]["attributes"]
    assert "JOINTS_0" not in attrs and "WEIGHTS_0" not in attrs


def test_a_skin_naming_a_missing_node_is_refused_by_name() -> None:
    model = _skinned_model()
    model.skins[0].joints = [1, 9]
    with pytest.raises(ValueError, match="names node 9"):
        glbwrite.write_glb(model)


def test_a_skin_with_the_wrong_number_of_matrices_is_refused() -> None:
    model = _skinned_model()
    model.skins[0] = gltf.Skin(joints=[1, 2], inverse_bind=np.eye(4)[None])
    with pytest.raises(ValueError, match="inverse bind matrices"):
        glbwrite.write_glb(model)


def test_a_skin_with_no_joints_or_a_non_finite_matrix_is_refused() -> None:
    model = _skinned_model()
    model.skins[0] = gltf.Skin(joints=[], inverse_bind=np.zeros((0, 4, 4)))
    with pytest.raises(ValueError, match="no joints"):
        glbwrite.write_glb(model)
    model = _skinned_model()
    bad = np.stack([np.eye(4), np.full((4, 4), np.nan)])
    model.skins[0] = gltf.Skin(joints=[1, 2], inverse_bind=bad)
    with pytest.raises(ValueError, match="non-finite"):
        glbwrite.write_glb(model)


def test_joints_without_weights_or_a_joint_past_255_are_refused() -> None:
    model = _skinned_model()
    prim = model.meshes[0][0]
    model.meshes[0][0] = gltf.Primitive(
        positions=prim.positions, indices=prim.indices, joints=prim.joints
    )
    with pytest.raises(ValueError, match="together"):
        glbwrite.write_glb(model)
    with pytest.raises(ValueError, match="0..255"):
        glbwrite.write_glb(_skinned_model(joints=np.full((4, 4), 300, dtype="i4")))


def test_a_skin_on_a_node_whose_mesh_was_dropped_loses_the_reference() -> None:
    # A node's ``skin`` means nothing without a mesh, and an empty mesh is dropped.
    model = _skinned_model()
    model.meshes[0][0] = gltf.Primitive(
        positions=np.zeros((0, 3), dtype="f4"), indices=np.zeros(0, dtype="u4")
    )
    doc, _ = read_glb(glbwrite.write_glb(model))
    assert "skin" not in doc["nodes"][3] and "mesh" not in doc["nodes"][3]
