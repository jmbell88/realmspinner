"""Typed move / scale / rotate of the selected elements (finding car-01).

The viewport's G/S/R drags were the only way to move picked vertices, edges or
faces by an exact amount, so an agent -- which has no pointer -- could not taper a
box's top. Three Clay ops carry the same transform by typed numbers, and they are
built on :mod:`element_move` so the drag and the typed door cannot disagree.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import select
from realmspinner.studio.modes.clay import element_move as em
from realmspinner.studio.modes.clay import ops as clay_ops
from realmspinner.studio.modes.clay.agent import dispatch as agent_clay

NEW_OPS = ("move-elements", "scale-elements", "rotate-elements")


class _Ctx:
    def __init__(self) -> None:
        self.toasts: list[str] = []

    def toast(self, text: str, level: str = "info") -> None:
        self.toasts.append(text)


def _top_face_doc(*, translation=(0.0, 0.0, 0.0)) -> tuple[bd.ClayDoc, bd.Obj, np.ndarray]:
    """A box with its +Y face picked in face mode. -> doc, object, the face's vertices."""
    doc = bd.ClayDoc()
    obj = bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box(), translation=np.array(translation))
    doc.add_object(obj)
    doc.set_element_mode("face")
    sel = el.ElementSel(faces=select.faces_by_normal(obj.mesh, [0.0, 1.0, 0.0], 10.0))
    doc.set_element_sel(obj.uid, sel)
    return doc, obj, el.affected_verts(obj.mesh, sel)


def _positions(doc: bd.ClayDoc, uid: int) -> np.ndarray:
    return np.array(doc.by_uid(uid).mesh.positions, dtype="f8")


def test_move_elements_op_shifts_only_the_selected_face_vertices_by_the_typed_distance() -> None:
    doc, obj, top = _top_face_doc()
    before = _positions(doc, obj.uid)
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("move-elements"), dx=0.5, dy=-0.25, dz=2.0)
    after = _positions(doc, obj.uid)
    moved = np.flatnonzero(np.any(after != before, axis=1))
    assert set(moved) == set(top.tolist())
    np.testing.assert_allclose(after[top] - before[top], [[0.5, -0.25, 2.0]] * len(top), atol=1e-6)


def test_scale_elements_op_about_selection_centre_tapers_a_boxs_top_face() -> None:
    doc, obj, top = _top_face_doc()
    before = _positions(doc, obj.uid)
    centre = before[top].mean(axis=0)
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("scale-elements"), sx=0.5, sz=0.5)
    after = _positions(doc, obj.uid)
    for axis in (0, 2):
        halved = centre[axis] + (before[top][:, axis] - centre[axis]) * 0.5
        np.testing.assert_allclose(after[top][:, axis], halved, atol=1e-6)
    np.testing.assert_allclose(after[top][:, 1], before[top][:, 1], atol=1e-6)
    rest = np.setdiff1d(np.arange(len(before)), top)
    np.testing.assert_array_equal(after[rest], before[rest])


def test_scale_elements_op_about_world_origin_scales_the_distance_from_it() -> None:
    doc, obj, top = _top_face_doc()
    before = _positions(doc, obj.uid)
    op = clay_ops.get("scale-elements")
    assert clay_ops.run(_Ctx(), doc, op, sx=2.0, sy=2.0, sz=2.0, pivot=1)
    after = _positions(doc, obj.uid)
    np.testing.assert_allclose(after[top], before[top] * 2.0, atol=1e-6)


def test_scale_elements_refuses_a_negative_factor_instead_of_flipping_winding() -> None:
    doc, obj, _top = _top_face_doc()
    before = doc.by_uid(obj.uid).mesh
    head = doc.history.head
    ctx = _Ctx()
    assert clay_ops.run(ctx, doc, clay_ops.get("scale-elements"), sx=-1.0) is False
    assert doc.by_uid(obj.uid).mesh is before
    assert doc.history.head == head
    assert len(ctx.toasts) == 1 and "Mirror" in ctx.toasts[0]


def test_scale_elements_allows_zero_which_flattens_without_inverting() -> None:
    doc, obj, top = _top_face_doc()
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("scale-elements"), sx=0.0)
    after = _positions(doc, obj.uid)
    assert np.allclose(after[top][:, 0], after[top][:, 0][0])


def test_scale_elements_function_refuses_negative_factor_with_the_shared_sentence() -> None:
    doc, obj, _top = _top_face_doc()
    errors: list[Exception] = []
    assert em.scale_elements(doc, [-1.0, 1.0, 1.0], refused=errors) is False
    assert errors and isinstance(errors[0], el.OpError) and "Mirror" in str(errors[0])


def test_rotate_elements_op_about_world_origin() -> None:
    doc, obj, top = _top_face_doc()
    before = _positions(doc, obj.uid)
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("rotate-elements"), ry=90.0, pivot=1)
    after = _positions(doc, obj.uid)
    # +90 degrees about world Y sends (x, y, z) to (z, y, -x).
    expected = np.stack([before[top][:, 2], before[top][:, 1], -before[top][:, 0]], axis=1)
    np.testing.assert_allclose(after[top], expected, atol=1e-5)
    rest = np.setdiff1d(np.arange(len(before)), top)
    np.testing.assert_array_equal(after[rest], before[rest])


def test_rotate_elements_about_selection_centre_keeps_the_centre_fixed() -> None:
    doc, obj, top = _top_face_doc(translation=(3.0, 1.0, -2.0))
    centre = em.element_median_world(doc)
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("rotate-elements"), rx=30.0, ry=45.0, rz=10.0)
    np.testing.assert_allclose(em.element_median_world(doc), centre, atol=1e-5)


@pytest.mark.parametrize(
    ("name", "params"),
    [
        ("move-elements", {"dx": 1.0}),
        ("scale-elements", {"sx": 2.0}),
        ("rotate-elements", {"rz": 20.0}),
    ],
)
def test_element_ops_are_one_undo_step(name: str, params: dict) -> None:
    doc, obj, _top = _top_face_doc()
    original = _positions(doc, obj.uid)
    head = doc.history.head
    depth = len(doc.history.history())
    assert clay_ops.run(_Ctx(), doc, clay_ops.get(name), **params)
    assert doc.history.head != head
    assert len(doc.history.history()) == depth + 1
    assert not np.array_equal(_positions(doc, obj.uid), original)
    doc.history.undo(doc)
    assert doc.history.head == head
    np.testing.assert_array_equal(_positions(doc, obj.uid), original)


@pytest.mark.parametrize("name", NEW_OPS)
def test_element_ops_that_change_nothing_push_no_step(name: str) -> None:
    doc, _obj, _top = _top_face_doc()
    head = doc.history.head
    assert clay_ops.run(_Ctx(), doc, clay_ops.get(name)) is False
    assert doc.history.head == head


@pytest.mark.parametrize("name", NEW_OPS)
def test_element_ops_are_refused_in_object_mode_with_a_reason(name: str) -> None:
    doc, obj, _top = _top_face_doc()
    doc.set_element_mode("object")
    doc.selection.add(obj.uid)
    op = clay_ops.get(name)
    assert op.enabled(doc) is False
    assert clay_ops.reason_for(op, doc) == "Switch to Points, Edges or Faces mode first."
    ctx = _Ctx()
    head = doc.history.head
    assert clay_ops.run(ctx, doc, op, **({"dx": 1.0} if name == "move-elements" else {})) is False
    assert doc.history.head == head
    assert ctx.toasts


@pytest.mark.parametrize("name", NEW_OPS)
def test_element_ops_are_refused_with_nothing_picked(name: str) -> None:
    doc, obj, _top = _top_face_doc()
    doc.set_element_sel(obj.uid, None)
    op = clay_ops.get(name)
    assert op.enabled(doc) is False
    assert clay_ops.reason_for(op, doc) == "Select something first."


def test_the_new_ops_are_element_only_have_no_key_and_say_what_they_are_for() -> None:
    for name in NEW_OPS:
        op = clay_ops.get(name)
        assert op.modes == clay_ops.ELEMENT_MODES
        assert op.key == ""
        assert op.label.endswith("...")
        assert op.hint and op.params


def test_affine_about_is_the_one_pivot_conjugation_the_drag_uses() -> None:
    from realmspinner.studio.modes.clay.ui import _view_drag

    assert not hasattr(_view_drag, "_about")
    assert _view_drag.affine_about is em.affine_about
    scale = np.diag([2.0, 2.0, 2.0, 1.0])
    matrix = em.affine_about(np.array([1.0, 0.0, 0.0]), scale)
    probe = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    np.testing.assert_allclose(em.apply_affine(matrix, probe), [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])


# --- the agent door ----------------------------------------------------------


def _agent_ctx():
    from .test_agent_clay import _Ctx as AgentCtx

    return AgentCtx()


def _payload(result: dict):
    return json.loads(result["content"][0]["text"])


def test_the_agent_clay_op_door_runs_scale_elements() -> None:
    ctx = _agent_ctx()
    session = agent_clay.Session()
    added = agent_clay.call(ctx, session, "clay_add_primitive", {"generator": "box"})
    assert added["isError"] is False, added
    uid = _payload(added)["uid"]
    assert agent_clay.call(ctx, session, "clay_element_mode", {"mode": "face"})["isError"] is False
    picked = agent_clay.call(
        ctx, session, "clay_select_by", {"uid": uid, "query": "normal", "direction": [0, 1, 0]}
    )
    assert picked["isError"] is False, picked
    from realmspinner.studio.modes.clay import mode as clay_mode

    doc = clay_mode.ensure(ctx).get(session.tab_uid).doc
    before = _positions(doc, uid)
    result = agent_clay.call(
        ctx, session, "clay_op", {"name": "scale-elements", "params": {"sx": 0.5, "sz": 0.5}}
    )
    assert result["isError"] is False, result
    after = _positions(doc, uid)
    top = np.flatnonzero(before[:, 1] > 0)
    xz = [0, 2]
    assert len(top) and np.all(np.abs(after[top][:, xz]) < np.abs(before[top][:, xz]) + 1e-9)
    assert np.any(after[top] != before[top])
    bottom = np.flatnonzero(before[:, 1] < 0)
    np.testing.assert_array_equal(after[bottom], before[bottom])


def test_clay_catalog_ops_lists_the_three_new_rows() -> None:
    ctx = _agent_ctx()
    result = agent_clay.call(ctx, agent_clay.Session(), "clay_catalog", {"topic": "ops"})
    assert result["isError"] is False, result
    catalogue = _payload(result)["catalogue"]
    for name in NEW_OPS:
        assert name in catalogue
