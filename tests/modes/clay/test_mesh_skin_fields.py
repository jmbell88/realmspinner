"""Per-vertex skin weights on a ``Mesh``, and what the document does with them.

Tranche 2 of the 2026-10-13 Clay rig plan. The policy these claims pin is "freeze topology,
carry the rest": an op that keeps a weighted object's vertices carries its weights, an op that
changes the vertex count is refused with a sentence naming Unbind Skin, and the one place that
decides is ``ClayDoc.set_mesh``. Every claim fails on the tree before the fields exist (``Mesh``
has no ``joints``).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio.modes.clay import ops as clay_ops

from ._rigkit import rigged_doc, skin_for, weighted


class _Ctx:
    def __init__(self) -> None:
        self.toasts: list[tuple[str, str]] = []

    def toast(self, message: str, level: str = "info") -> None:
        self.toasts.append((message, level))


# -- the fields -----------------------------------------------------------------------------


def test_a_mesh_without_weights_has_none_and_validates() -> None:
    mesh = bp.box()
    assert mesh.joints is None and mesh.weights is None
    bm.validate(mesh)


def test_a_weighted_mesh_validates_and_freezes_its_arrays() -> None:
    mesh = weighted(bp.box())
    bm.validate(mesh)
    assert mesh.joints.shape == (len(mesh.positions), 4) and mesh.weights.dtype == np.float32
    assert not mesh.joints.flags.writeable and not mesh.weights.flags.writeable


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda m: dataclasses.replace(m, weights=None), "together"),
        (lambda m: dataclasses.replace(m, joints=m.joints[:-1]), "one row of four"),
        (lambda m: dataclasses.replace(m, joints=m.joints - 1), "negative"),
        (lambda m: dataclasses.replace(m, weights=m.weights * 2.0), "sum to one"),
        (lambda m: dataclasses.replace(m, weights=np.where(m.weights == 1, np.nan, 0.0)), "finite"),
    ],
)
def test_validate_refuses_weights_that_are_not_a_skin(change, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        bm.validate(change(weighted(bp.box())))


def test_replace_and_transformed_carry_the_weights() -> None:
    mesh = weighted(bp.box())
    moved = bm.transformed(mesh, np.diag([2.0, 2.0, 2.0, 1.0]))
    assert np.array_equal(moved.joints, mesh.joints) and np.array_equal(moved.weights, mesh.weights)
    smooth = dataclasses.replace(mesh, smooth=np.ones(bm.face_count(mesh), dtype=bool))
    assert smooth.joints is not None


def test_the_undo_budget_counts_the_weights() -> None:
    from realmspinner.kernels.mesh import edits

    plain, skinned = bp.box(), weighted(bp.box())
    assert edits.mesh_bytes(skinned) - edits.mesh_bytes(plain) == (
        skinned.joints.nbytes + skinned.weights.nbytes
    )


# -- the carry table: ops that keep the vertices keep the weights ---------------------------


def _faces_picked(doc: bd.ClayDoc, uid: int, faces: list[int]) -> None:
    doc.select([uid])
    doc.set_element_mode("face")
    doc.set_element_sel(uid, el.ElementSel(faces=faces))


@pytest.mark.parametrize(
    "op",
    ["triangulate", "flip", "shade-smooth", "mirror-x"],
)
def test_ops_that_keep_the_vertices_carry_a_weighted_objects_weights(op: str) -> None:
    doc, bone, body = rigged_doc()
    before = doc.by_uid(body).mesh
    _faces_picked(doc, body, [0, 1])
    if op in ("mirror-x", "shade-smooth"):
        doc.set_element_mode("object")
        doc.select([body])
    ctx = _Ctx()
    assert clay_ops.run(ctx, doc, clay_ops.get(op)) is True, ctx.toasts
    after = doc.by_uid(body).mesh
    assert after is not before
    assert after.joints is not None, op
    assert np.array_equal(after.joints, before.joints) and np.array_equal(
        after.weights, before.weights
    )
    assert doc.by_uid(body).skin is not None


def test_a_moved_point_keeps_its_weights() -> None:
    doc, bone, body = rigged_doc()
    mesh = doc.by_uid(body).mesh
    moved = bm.Mesh(
        positions=mesh.positions + 0.1,
        loops=mesh.loops,
        starts=mesh.starts,
        material=mesh.material,
        smooth=mesh.smooth,
        uv=mesh.uv,
    )
    assert doc.set_mesh(body, moved) is True
    assert doc.by_uid(body).mesh.joints is not None


# -- the refuse table: ops that change the vertex count are refused by name ------------------


@pytest.mark.parametrize("op", ["extrude", "inset", "subdivide", "bevel"])
def test_ops_that_change_the_vertex_count_are_refused_naming_unbind_skin(op: str) -> None:
    doc, bone, body = rigged_doc()
    before = doc.by_uid(body).mesh
    if op == "bevel":
        doc.select([body])
        doc.set_element_mode("edge")
        found = el.convert(before, el.ElementSel(faces=[0]), "edge")
        doc.set_element_sel(body, el.ElementSel(edges=[tuple(found.edges[0])]))
    else:
        _faces_picked(doc, body, [0])
    steps, ctx = len(doc.history), _Ctx()
    assert clay_ops.run(ctx, doc, clay_ops.get(op)) is False
    assert any("Unbind Skin" in message for message, _ in ctx.toasts), ctx.toasts
    assert doc.by_uid(body).mesh is before and len(doc.history) == steps


def test_set_mesh_is_the_backstop_for_a_door_that_bypasses_the_ops() -> None:
    doc, bone, body = rigged_doc()
    steps = len(doc.history)
    with pytest.raises(el.OpError, match="Unbind Skin"):
        doc.set_mesh(body, bp.icosphere())
    assert len(doc.history) == steps


# -- unbinding ------------------------------------------------------------------------------


def test_unbind_skin_drops_the_weights_and_the_skin_as_one_undo_step() -> None:
    doc, bone, body = rigged_doc()
    steps = len(doc.history)
    assert doc.unbind_skin(body) is True
    obj = doc.by_uid(body)
    assert obj.skin is None and obj.mesh.joints is None and obj.mesh.weights is None
    assert len(doc.history) == steps + 1
    assert doc.undo()
    assert doc.by_uid(body).skin is not None and doc.by_uid(body).mesh.joints is not None
    assert doc.redo()
    assert doc.by_uid(body).skin is None


def test_unbinding_an_unweighted_object_pushes_nothing() -> None:
    doc, bone, body = rigged_doc()
    steps = len(doc.history)
    assert doc.unbind_skin(bone) is False
    assert len(doc.history) == steps


def test_after_unbinding_the_topology_ops_work_again() -> None:
    doc, bone, body = rigged_doc()
    doc.unbind_skin(body)
    _faces_picked(doc, body, [0])
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("extrude")) is True


# -- joining, separating, deleting -----------------------------------------------------------


def test_a_weighted_object_cannot_be_joined_or_separated_without_losing_its_weights() -> None:
    doc, bone, body = rigged_doc()
    other = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Other", mesh=bp.box())).uid
    with pytest.raises(el.OpError, match="Unbind Skin"):
        doc.join_objects(other, bp.box(), [body])
    with pytest.raises(el.OpError, match="Unbind Skin"):
        doc.join_objects(body, bp.box(), [other])
    with pytest.raises(el.OpError, match="Unbind Skin"):
        doc.separate(body, [bp.box(), bp.box()])


def test_a_bone_a_skin_names_cannot_be_deleted_or_unmarked() -> None:
    doc, bone, body = rigged_doc()
    with pytest.raises(el.OpError, match="Unbind Skin"):
        doc.remove_object(bone)
    refusals = doc.remove_objects([bone])
    assert refusals and "Unbind Skin" in refusals[0]
    assert {o.uid for o in doc.objects} == {bone, body}
    with pytest.raises(el.OpError, match="Unbind Skin"):
        doc.set_bone([bone], False)
    assert doc.by_uid(bone).bone is True


def test_deleting_the_weighted_object_and_its_bone_together_is_allowed() -> None:
    doc, bone, body = rigged_doc()
    assert doc.remove_objects([bone, body]) == []
    assert doc.objects == []


def test_a_skin_that_names_a_bone_that_is_gone_is_a_rig_problem() -> None:
    doc = bd.ClayDoc()
    doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="Body", mesh=weighted(bp.box()), skin=skin_for(999))
    )
    assert any("does not exist" in problem for problem in doc.rig_problems())


def test_a_healthy_rig_has_no_problems() -> None:
    doc, bone, body = rigged_doc()
    assert doc.rig_problems() == []


def test_skin_validation_rejects_a_duplicate_joint_and_a_bad_matrix() -> None:
    with pytest.raises(ValueError, match="twice"):
        skin_for(1, 1)
    from realmspinner.kernels.mesh import rig

    with pytest.raises(ValueError, match="4x4"):
        rig.Skin((1,), np.zeros((2, 4, 4)))
    with pytest.raises(ValueError, match="finite"):
        rig.Skin((1,), np.full((1, 4, 4), np.nan))
