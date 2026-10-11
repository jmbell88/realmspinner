"""Regression tests for the kernel half of the 2026-10-11 Clay audit.

Findings clay-12 (a sheared reparent leaves the object's own children behind),
clay-13 (a topology-changing ``set_mesh`` with no ``select`` leaves picks past
the mesh), clay-21 (``with_descendants`` is quadratic), clay-22 and clay-23
(docstrings that describe code that is gone), clay-28 (the ``merge_groups``
docstring), clay-36 (an identity UV transform returns a new mesh) and clay-41
(retyping a reopened box's own size dirties the document).
"""

from __future__ import annotations

import dataclasses
import inspect
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import measure, ops_dissolve, ops_topo, selection, serialize, uvtools
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import scratch as clay_scratch

TESTS = Path(__file__).resolve().parent


def _obj(name: str, **kwargs: Any) -> bd.Obj:
    kwargs.setdefault("mesh", bp.box())
    return bd.Obj(uid=bd.new_uid(), name=name, **kwargs)


def _world_positions(doc: bd.ClayDoc, uid: int) -> np.ndarray:
    pts = doc.by_uid(uid).mesh.positions.astype("f8")
    world = doc.world_matrix(uid)
    return (np.c_[pts, np.ones(len(pts))] @ world.T)[:, :3]


def _turn_z_45() -> np.ndarray:
    half = math.radians(45.0) / 2.0
    return np.array([0.0, 0.0, math.sin(half), math.cos(half)])


def _sheared_family() -> tuple[bd.ClayDoc, bd.Obj, bd.Obj, bd.Obj]:
    """A parent scaled (3, 1, 1), a root shape turned 45 degrees about Z, and
    that shape's own child: parenting the shape under the parent shears it."""
    doc = bd.ClayDoc()
    parent = doc.add_object(_obj("Wide", scale=np.array([3.0, 1.0, 1.0])))
    turned = doc.add_object(
        _obj("Turned", rotation=_turn_z_45(), translation=np.array([0.0, 2.0, 0.0]))
    )
    leaf = doc.add_object(
        _obj("Leaf", parent=turned.uid, translation=np.array([1.0, 0.0, 0.0]))
    )
    return doc, parent, turned, leaf


# --- clay-12 -----------------------------------------------------------------


def test_parenting_a_rotated_object_with_children_under_an_unevenly_scaled_parent_is_refused() -> (
    None
):
    doc, parent, turned, leaf = _sheared_family()
    steps = len(doc.history)
    leaf_before = _world_positions(doc, leaf.uid)

    with pytest.raises(el.OpError, match="Turned") as refused:
        doc.set_parent(turned.uid, parent.uid)

    assert "children" in str(refused.value)
    assert len(doc.history) == steps
    assert doc.by_uid(turned.uid).parent is None
    assert np.allclose(_world_positions(doc, leaf.uid), leaf_before)


def test_grouping_a_sheared_object_that_has_children_is_refused_and_adds_no_empty() -> None:
    doc, parent, turned, leaf = _sheared_family()
    # The already-sheared state, reached through the back door: the turned
    # shape sits under the wide parent with its child beneath it.
    doc.by_uid(turned.uid).parent = parent.uid
    steps, count = len(doc.history), len(doc.objects)
    leaf_before = _world_positions(doc, leaf.uid)

    with pytest.raises(el.OpError, match="Turned"):
        doc.group([turned.uid])

    assert len(doc.history) == steps and len(doc.objects) == count
    assert np.allclose(_world_positions(doc, leaf.uid), leaf_before)


def test_clearing_the_parent_of_a_sheared_object_with_children_is_refused() -> None:
    doc, parent, turned, leaf = _sheared_family()
    # Build the already-sheared state through the document's own back door:
    # the turned shape lives under the wide parent with its child beneath it.
    doc.by_uid(turned.uid).parent = parent.uid
    steps = len(doc.history)
    with pytest.raises(el.OpError, match="Turned"):
        doc.set_parent(turned.uid, None)
    assert len(doc.history) == steps and doc.by_uid(turned.uid).parent == parent.uid


def test_a_sheared_object_with_no_children_is_still_reparented_with_its_shear_baked() -> None:
    doc, parent, turned, leaf = _sheared_family()
    doc.set_parent(leaf.uid, None)
    before = _world_positions(doc, turned.uid)
    assert doc.set_parent(turned.uid, parent.uid)
    assert np.allclose(_world_positions(doc, turned.uid), before, atol=1e-5)


# --- clay-13 -----------------------------------------------------------------


def _icosphere_doc() -> tuple[bd.ClayDoc, bd.Obj]:
    doc = bd.ClayDoc()
    defaults, build = bp.GENERATORS["icosphere"]
    params = dict(defaults, subdivisions=3)
    obj = doc.add_object(_obj("Ball", mesh=build(**params), generator="icosphere", params=params))
    return doc, obj


def test_applying_a_preview_that_shrinks_a_mesh_in_face_mode_leaves_no_pick_past_the_mesh() -> None:
    doc, obj = _icosphere_doc()
    n_faces = bd.bm.face_count(obj.mesh)
    assert n_faces >= 300
    doc.select([obj.uid])
    doc.set_element_mode("face")
    doc.set_element_sel(obj.uid, el.ElementSel(faces=np.arange(n_faces - 10, n_faces)))
    assert doc.selection_problems() == []

    scratch = clay_scratch.clone(doc)
    shrunk, _ = ops_topo.delete_faces(obj.mesh, el.ElementSel(faces=np.arange(40)))
    scratch.set_mesh(obj.uid, shrunk, keep_generator=True)
    diff = clay_scratch.diff(doc, scratch)
    assert clay_scratch.transplant(doc, scratch, diff)

    assert bd.bm.face_count(doc.by_uid(obj.uid).mesh) == n_faces - 40
    assert doc.selection_problems() == []


def test_set_mesh_with_no_select_keeps_picks_the_new_mesh_still_has() -> None:
    doc, obj = _icosphere_doc()
    n_faces = bd.bm.face_count(obj.mesh)
    doc.select([obj.uid])
    doc.set_element_mode("face")
    doc.set_element_sel(obj.uid, el.ElementSel(faces=np.array([0, 1, n_faces - 1])))
    shrunk, _ = ops_topo.delete_faces(obj.mesh, el.ElementSel(faces=np.arange(40, 80)))

    doc.set_mesh(obj.uid, shrunk)

    assert doc.element_sel_of(obj.uid).faces.tolist() == [0, 1]
    assert doc.selection_problems() == []


def test_set_mesh_with_the_same_counts_leaves_picks_alone() -> None:
    """An element drag moves positions only: its picks must survive it."""
    doc, obj = _icosphere_doc()
    doc.select([obj.uid])
    doc.set_element_mode("face")
    pick = el.ElementSel(faces=np.array([3, 4, 5]))
    doc.set_element_sel(obj.uid, pick)
    moved = dataclasses.replace(obj.mesh, positions=obj.mesh.positions + np.float32(0.1))

    doc.set_mesh(obj.uid, moved)

    assert doc.element_sel_of(obj.uid).same_as(pick)


# --- clay-21 -----------------------------------------------------------------


def test_with_descendants_walks_the_document_a_constant_number_of_times(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    original = bd.ClayDoc.descendants

    def counted(self: bd.ClayDoc, uid: int) -> list[int]:
        calls.append(uid)
        return original(self, uid)

    monkeypatch.setattr(bd.ClayDoc, "descendants", counted)
    totals = []
    for n in (24, 96):
        doc = bd.ClayDoc()
        objs = [_obj(f"B{i}") for i in range(n)]
        for i, child in enumerate(objs[n // 2 :]):
            child.parent = objs[i].uid
        doc.add_objects(objs)
        calls.clear()
        found = selection.with_descendants(doc, [o.uid for o in objs[: n // 2]])
        assert found == {o.uid for o in objs}
        totals.append(len(calls))
    assert totals[1] <= totals[0] + 1, f"descendants() called {totals} times for 24 and 96 objects"


def test_with_descendants_still_answers_every_subtree_and_survives_a_cycle() -> None:
    doc = bd.ClayDoc()
    a, b, c, d = (doc.add_object(_obj(n)) for n in "abcd")
    b.parent, c.parent = a.uid, b.uid
    assert selection.with_descendants(doc, [a.uid]) == {a.uid, b.uid, c.uid}
    assert selection.with_descendants(doc, [d.uid]) == {d.uid}
    assert selection.with_descendants(doc, []) == set()
    a.parent = c.uid  # a hand-built cycle must terminate
    assert selection.with_descendants(doc, [a.uid]) == {a.uid, b.uid, c.uid}


# --- clay-22 / clay-23 / clay-28: docstrings ---------------------------------


def test_delete_selected_docstring_names_remove_objects_not_a_gesture() -> None:
    doc_text = selection.delete_selected.__doc__ or ""
    object_branch = doc_text.split("The element-mode branch")[0]
    assert "remove_objects" in object_branch
    assert "collapse_since" not in object_branch and "descending" not in object_branch
    pin = (TESTS / "test_clay_imports.py").read_text(encoding="utf-8")
    assert "``remove_object`` itself, per uid" not in pin


def test_is_closed_docstring_does_not_cite_a_door_that_no_longer_reports_closed() -> None:
    text = measure.is_closed.__doc__ or ""
    assert "clay_add_mesh" not in text


def test_merge_faces_documents_which_material_a_mixed_block_keeps() -> None:
    text = inspect.getdoc(ops_dissolve.merge_groups) or ""
    assert "lowest-numbered" in text
    assert "material" in text and "smooth" in text


# --- clay-36 -----------------------------------------------------------------


def _uv_quad() -> bm.Mesh:
    positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
    uv = [(0.1, 0.2), (0.7, 0.2), (0.7, 0.9), (0.1, 0.9)]
    return bm.from_faces(positions, [[0, 1, 2, 3]], [uv])


def test_transform_islands_with_identity_arguments_returns_the_same_mesh() -> None:
    mesh = _uv_quad()
    assert uvtools.transform_islands(mesh, [0]) is mesh
    assert uvtools.transform_islands(mesh, [0], pivot="origin") is mesh


def test_transform_islands_with_a_real_move_still_returns_a_new_mesh() -> None:
    mesh = _uv_quad()
    out = uvtools.transform_islands(mesh, [0], translate=(0.25, 0.0))
    assert out is not mesh and not np.array_equal(out.uv, mesh.uv)


# --- clay-41 -----------------------------------------------------------------


def test_retyping_a_reopened_boxs_own_size_pushes_no_step_and_leaves_the_document_clean() -> None:
    doc = bd.ClayDoc()
    defaults, build = bp.GENERATORS["box"]
    params = dict(defaults, size=(2.0, 1.0, 3.0))
    doc.add_object(_obj("Crate", mesh=build(**params), generator="box", params=params))
    reopened = serialize.read_rblk(serialize.rblk_bytes(doc))
    obj = reopened.objects[0]
    assert isinstance(obj.params["size"], list)  # the file hands back a list
    reopened.mark_saved()
    steps, rev = len(reopened.history), reopened.rev

    typed = dict(obj.params, size=(2.0, 1.0, 3.0))  # the widget hands back a tuple
    rebuilt = build(**bp.clamp_params("box", dict(typed)))
    changed = reopened.set_generator_params(
        obj.uid, typed, rebuilt, was={"params": dict(obj.params)}
    )

    assert changed is False
    assert not reopened.dirty and len(reopened.history) == steps and reopened.rev == rev


def test_retyping_a_different_size_on_a_reopened_box_still_pushes_a_step() -> None:
    doc = bd.ClayDoc()
    defaults, build = bp.GENERATORS["box"]
    params = dict(defaults, size=(2.0, 1.0, 3.0))
    doc.add_object(_obj("Crate", mesh=build(**params), generator="box", params=params))
    reopened = serialize.read_rblk(serialize.rblk_bytes(doc))
    obj = reopened.objects[0]
    reopened.mark_saved()
    typed = dict(obj.params, size=(2.0, 1.0, 4.0))
    rebuilt = build(**bp.clamp_params("box", dict(typed)))
    assert reopened.set_generator_params(obj.uid, typed, rebuilt, was={"params": dict(obj.params)})
    assert reopened.dirty
