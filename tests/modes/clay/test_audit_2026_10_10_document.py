"""The 2026-10-10 audit's document-model findings: clay-04, 05, 06, 32, 33.

Each test name is the claim; each one failed against the code it was written
for. Everything here is headless: a ``ClayDoc``, the undo stack under it and the
selection kernel beside it.
"""

from __future__ import annotations

import numpy as np

from realmspinner.core import undo
from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import selection, serialize


def _box_obj(name: str = "A", **kwargs: object) -> bd.Obj:
    d, build = bp.GENERATORS["box"]
    kwargs.setdefault("generator", "box")
    kwargs.setdefault("params", dict(d))
    return bd.Obj(
        uid=bd.new_uid(),
        name=name,
        mesh=build(**bp.clamp_params("box", dict(d))),
        **kwargs,  # type: ignore[arg-type]
    )


def _plain(name: str = "A", **kwargs: object) -> bd.Obj:
    return bd.Obj(uid=bd.new_uid(), name=name, mesh=bp.box(), **kwargs)  # type: ignore[arg-type]


def _z_quat(degrees: float) -> np.ndarray:
    half = np.deg2rad(degrees) / 2.0
    return np.array([0.0, 0.0, np.sin(half), np.cos(half)])


def _world_points(doc: bd.ClayDoc, uid: int) -> np.ndarray:
    """The object's vertices where they stand in the world."""
    obj = doc.by_uid(uid)
    pos = obj.mesh.positions.astype("f8")
    homo = np.hstack([pos, np.ones((len(pos), 1))])
    return (doc.world_matrix(uid) @ homo.T).T[:, :3]


def _sheared_pair() -> tuple[bd.ClayDoc, bd.Obj, bd.Obj]:
    """A (2, 1, 1) parent over a child turned 45 degrees: the child's world
    matrix is sheared, which no translation/rotation/scale can state."""
    doc = bd.ClayDoc()
    parent = doc.add_object(_plain("P", scale=np.array([2.0, 1.0, 1.0])))
    child = doc.add_object(_box_obj("C", parent=parent.uid, rotation=_z_quat(45.0)))
    return doc, parent, child


# --- clay-04 -------------------------------------------------------------------


def test_remove_object_under_a_non_uniformly_scaled_parent_never_stores_a_non_unit_rotation() -> (
    None
):
    doc, parent, child = _sheared_pair()
    before = _world_points(doc, child.uid)

    doc.remove_object(parent.uid)

    stored = doc.by_uid(child.uid)
    assert abs(float(np.linalg.norm(stored.rotation)) - 1.0) < 1e-9
    assert np.abs(_world_points(doc, child.uid) - before).max() < 1e-5, (
        "the child's geometry moved on screen when its sheared parent was deleted"
    )


def test_set_parent_to_none_under_a_non_uniformly_scaled_parent_keeps_the_geometry_in_place() -> (
    None
):
    doc, _parent, child = _sheared_pair()
    before = _world_points(doc, child.uid)

    doc.set_parent(child.uid, None)

    assert abs(float(np.linalg.norm(doc.by_uid(child.uid).rotation)) - 1.0) < 1e-9
    assert np.abs(_world_points(doc, child.uid) - before).max() < 1e-5


def test_grouping_an_already_sheared_child_keeps_it_where_it_stands() -> None:
    doc, _parent, child = _sheared_pair()
    before = _world_points(doc, child.uid)

    doc.group([child.uid])

    assert abs(float(np.linalg.norm(doc.by_uid(child.uid).rotation)) - 1.0) < 1e-9
    assert np.abs(_world_points(doc, child.uid) - before).max() < 1e-5


def test_join_objects_absorbing_a_sheared_parent_keeps_its_child_in_place() -> None:
    doc, parent, child = _sheared_pair()
    keeper = doc.add_object(_plain("K"))
    before = _world_points(doc, child.uid)

    doc.join_objects(keeper.uid, keeper.mesh, [parent.uid])

    assert abs(float(np.linalg.norm(doc.by_uid(child.uid).rotation)) - 1.0) < 1e-9
    assert np.abs(_world_points(doc, child.uid) - before).max() < 1e-5


def test_separate_a_sheared_parent_keeps_its_child_in_place() -> None:
    doc, parent, child = _sheared_pair()
    before = _world_points(doc, child.uid)
    half_a, half_b = bp.box(), bp.box()

    doc.separate(parent.uid, [half_a, half_b])

    assert abs(float(np.linalg.norm(doc.by_uid(child.uid).rotation)) - 1.0) < 1e-9
    assert np.abs(_world_points(doc, child.uid) - before).max() < 1e-5


def test_a_baked_shear_freezes_the_generator_and_undo_restores_mesh_generator_and_transform() -> (
    None
):
    doc, parent, child = _sheared_pair()
    mesh_before = child.mesh
    trs_before = tuple(np.array(v, copy=True) for v in child.trs())

    doc.remove_object(parent.uid)
    assert doc.by_uid(child.uid).generator is None, "baked geometry is no longer a box"

    assert doc.undo()
    restored = doc.by_uid(child.uid)
    assert restored.mesh is mesh_before
    assert restored.generator == "box"
    assert restored.parent == parent.uid
    for a, b in zip(restored.trs(), trs_before, strict=True):
        assert np.array_equal(a, b)


def test_a_clean_basis_is_not_baked_and_keeps_its_generator() -> None:
    doc = bd.ClayDoc()
    parent = doc.add_object(_plain("P", scale=np.array([2.0, 2.0, 2.0])))
    child = doc.add_object(_box_obj("C", parent=parent.uid, rotation=_z_quat(45.0)))
    mesh_before = child.mesh

    doc.remove_object(parent.uid)

    kept = doc.by_uid(child.uid)
    assert kept.mesh is mesh_before, "an orthogonal basis needs no baking"
    assert kept.generator == "box"


def test_a_baked_shear_survives_a_save_and_reopen_unchanged() -> None:
    doc, parent, child = _sheared_pair()
    doc.remove_object(parent.uid)
    on_screen = _world_points(doc, child.uid)

    back = serialize.read_rblk(serialize.rblk_bytes(doc))

    assert np.abs(_world_points(back, child.uid) - on_screen).max() < 1e-4


# --- clay-05 / clay-06: the undo head across eviction ---------------------------


class _Step(undo.Edit):
    cost = 1

    def undo(self, doc: object) -> None:
        pass

    def redo(self, doc: object) -> None:
        pass


def test_a_document_reads_dirty_after_eviction_even_when_every_remaining_step_is_undone() -> None:
    doc = serialize.read_rblk(serialize.rblk_bytes(bd.ClayDoc()))  # opened: empty history
    assert not doc.dirty
    for _ in range(undo.UNDO_MAX_DEPTH + 6):
        doc.add_object(_plain())
    assert doc.dirty
    while doc.undo():
        pass

    assert len(doc.objects) == 6, "the evicted work is still in the document"
    assert doc.dirty, "a document holding six unsaved objects read clean"


def test_a_stack_that_evicted_nothing_still_reads_zero_at_the_bottom() -> None:
    stack = undo.UndoStack()
    assert stack.head == 0
    stack.push(_Step())
    stack.undo(None)
    assert stack.head == 0


def test_the_undo_head_at_the_bottom_of_an_evicted_stack_is_the_last_evicted_serial() -> None:
    stack = undo.UndoStack()
    steps = [_Step() for _ in range(undo.UNDO_MAX_DEPTH + 3)]
    for step in steps:
        stack.push(step)
    while stack.undo(None):
        pass

    assert stack.head == steps[2].serial
    stack.clear()
    assert stack.head == 0, "clear forgets the floor with the history"


def test_a_checkpoint_taken_on_an_empty_history_reports_gone_after_the_stack_evicted_past_it() -> (
    None
):
    doc = bd.ClayDoc()
    doc.set_checkpoint("start")
    for _ in range(undo.UNDO_MAX_DEPTH + 16):
        doc.add_object(_plain())

    assert doc.checkpoint_status("start") == "gone"
    count = len(doc.objects)
    assert doc.restore_checkpoint("start") is False
    assert len(doc.objects) == count, "a gone checkpoint must not half-restore"


def test_a_checkpoint_on_an_empty_history_is_still_reachable_when_nothing_was_evicted() -> None:
    doc = bd.ClayDoc()
    doc.set_checkpoint("start")
    for _ in range(5):
        doc.add_object(_plain())

    assert doc.checkpoint_status("start") == "reachable"
    assert doc.restore_checkpoint("start") is True
    assert len(doc.objects) == 0


# --- clay-32 -------------------------------------------------------------------


def test_deleting_every_object_of_a_four_thousand_object_document_stays_under_a_budget(
    monkeypatch,
) -> None:
    n = 4000
    doc = bd.ClayDoc()
    box = bp.box()
    doc.add_objects([bd.Obj(uid=bd.new_uid(), name=f"B{i}", mesh=box) for i in range(n)])
    selection.select_all(doc)

    # Counted rather than timed: a wall-clock bound flakes under a loaded
    # machine, and the defect is the shape (one full scan per deleted object).
    scans = {"n": 0}
    for name in ("_uid_map", "index_of", "children_of"):
        original = getattr(bd.ClayDoc, name)

        def counted(self, *args, __original=original, **kwargs):
            scans["n"] += 1
            return __original(self, *args, **kwargs)

        monkeypatch.setattr(bd.ClayDoc, name, counted)

    refusals = selection.delete_selected(doc)

    assert refusals == []
    assert len(doc.objects) == 0
    assert scans["n"] < 50, f"{scans['n']} whole-document scans to delete {n} objects"
    monkeypatch.undo()
    assert doc.undo()
    assert len(doc.objects) == n, "the whole batch is one undo step"


def test_a_batch_delete_reparents_the_children_of_a_deleted_chain_like_one_at_a_time() -> None:
    doc = bd.ClayDoc()
    top = doc.add_object(_plain("top", translation=np.array([1.0, 0.0, 0.0])))
    mid = doc.add_object(_plain("mid", parent=top.uid, translation=np.array([0.0, 2.0, 0.0])))
    leaf = doc.add_object(_plain("leaf", parent=mid.uid, translation=np.array([0.0, 0.0, 3.0])))
    before = _world_points(doc, leaf.uid)

    assert doc.remove_objects([top.uid, mid.uid]) == []

    assert [o.name for o in doc.objects] == ["leaf"]
    assert doc.by_uid(leaf.uid).parent is None
    assert np.abs(_world_points(doc, leaf.uid) - before).max() < 1e-9
    assert doc.undo()
    assert [o.name for o in doc.objects] == ["top", "mid", "leaf"]
    assert doc.by_uid(leaf.uid).parent == mid.uid


# --- clay-33 -------------------------------------------------------------------


def test_delete_in_edge_mode_with_a_lone_edge_refuses_in_edge_terms() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_plain("A"))
    doc.set_element_mode("edge")
    doc.set_element_sel(a.uid, el.ElementSel(edges=[[0, 1]]))
    depth = len(doc.history)

    refusals = selection.delete_selected(doc)

    assert len(refusals) == 1
    assert "Select at least one face" not in refusals[0]
    assert "edges" in refusals[0] and "Faces mode" in refusals[0]
    assert len(doc.history) == depth


def test_delete_in_vertex_mode_with_a_lone_vertex_refuses_in_vertex_terms() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_plain("A"))
    doc.set_element_mode("vertex")
    doc.set_element_sel(a.uid, el.ElementSel(verts=[0]))

    refusals = selection.delete_selected(doc)

    assert len(refusals) == 1
    assert "Select at least one face" not in refusals[0]
    assert "corners" in refusals[0] and "Faces mode" in refusals[0]


def test_delete_in_face_mode_with_nothing_picked_keeps_the_face_sentence() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_plain("A"))
    doc.set_element_mode("face")
    doc.set_element_sel(a.uid, el.ElementSel(faces=[]))

    assert selection.delete_selected(doc) in (["Select at least one face to delete."], [])


def test_m3_is_still_the_decompose_other_callers_use() -> None:
    # decompose's behaviour for other callers is unchanged: a clean basis round-trips.
    t = np.array([1.0, 2.0, 3.0])
    q = _z_quat(30.0)
    s = np.array([2.0, 1.0, 3.0])
    t2, q2, s2 = m3.decompose(m3.compose(t, q, s))
    assert np.allclose(t2, t) and np.allclose(s2, s)
    assert np.allclose(m3.compose(t2, q2, s2), m3.compose(t, q, s))
