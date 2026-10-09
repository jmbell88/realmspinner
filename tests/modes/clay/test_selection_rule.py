"""The selection rule: the shape being edited stays selected in an element mode.

``ClayDoc.selection`` used to be *derived* in Points/Edges/Faces -- exactly the
objects with a non-empty ``element_sel`` -- so entering a mode deselected the
shape, the inspector went blank and a shape with nothing picked yet was not
"selected" at all. The rule now (see the ``document`` module docstring):

* object mode holds no element picks;
* in an element mode every pick is on a selected object;
* ``selection`` is written only by ``select`` (which prunes the picks of what
  it deselects), ``set_element_sel`` (which adds, never removes), ``separate``
  and the sites that make an object stop existing or being visible.

``test_selection_rule_fuzz`` walks the doors at random; these are the named cases.
"""

from __future__ import annotations

import numpy as np

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import ops as mesh_ops
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio.modes.clay import ops as clay_ops


class _Ctx:
    def __init__(self) -> None:
        self.toasts: list[str] = []

    def toast(self, message: str, level: str = "info") -> None:
        self.toasts.append(message)


def _doc(n: int = 2) -> tuple[bd.ClayDoc, list[bd.Obj]]:
    doc = bd.ClayDoc()
    objs = [
        doc.add_object(
            bd.Obj(
                uid=bd.new_uid(),
                name=f"B{i}",
                mesh=bp.box(),
                translation=np.array([3.0 * i, 0, 0]),
            )
        )
        for i in range(n)
    ]
    return doc, objs


def _faces(*idx: int) -> el.ElementSel:
    return el.ElementSel(faces=list(idx))


# --- the checker itself ---------------------------------------------------------


def test_selection_problems_names_each_rule_it_enforces() -> None:
    doc, (a, b) = _doc()
    assert doc.selection_problems() == []

    doc.selection = {a.uid, 999_999}
    assert any("not in the document" in p for p in doc.selection_problems())

    doc.selection = {a.uid}
    doc.element_sel = {a.uid: _faces(0)}  # picks in object mode
    assert any("object mode holds element picks" in p for p in doc.selection_problems())

    doc.element_mode = "face"
    assert doc.selection_problems() == []
    doc.element_sel = {b.uid: _faces(0)}  # a pick on an object that is not being edited
    assert any("not selected" in p for p in doc.selection_problems())

    doc.element_sel = {a.uid: _faces(99)}  # past the box's six faces
    assert any("past the mesh" in p for p in doc.selection_problems())

    doc.element_sel = {}
    a.visible = False
    assert any("hidden" in p for p in doc.selection_problems())


# --- entering and leaving a mode -------------------------------------------------


def test_entering_an_element_mode_with_several_shapes_selected_edits_all_of_them() -> None:
    doc, (a, b) = _doc()
    doc.select([a.uid, b.uid])
    doc.set_element_mode("face")
    assert doc.selection == {a.uid, b.uid}
    assert doc.element_sel == {}


def test_entering_an_element_mode_with_nothing_selected_enters_with_an_empty_edited_set() -> None:
    doc, (a, _b) = _doc()
    doc.set_element_mode("vertex")
    assert doc.element_mode == "vertex"
    assert doc.selection == set()


def test_the_first_pick_adds_its_object_to_the_edited_set() -> None:
    doc, (a, _b) = _doc()
    doc.set_element_mode("face")
    doc.set_element_sel(a.uid, _faces(0))
    assert doc.selection == {a.uid}


def test_switching_between_element_modes_leaves_the_selection_alone() -> None:
    doc, (a, b) = _doc()
    doc.select([a.uid])
    doc.set_element_mode("face")
    doc.set_element_sel(a.uid, _faces(0))
    for mode in ("vertex", "edge", "face", "object"):
        doc.set_element_mode(mode)
        assert doc.selection == {a.uid}, mode
        assert doc.selection_problems() == []


# --- select() in an element mode --------------------------------------------------


def test_select_in_an_element_mode_drops_the_picks_of_what_it_deselects() -> None:
    doc, (a, b) = _doc()
    doc.select([a.uid, b.uid])
    doc.set_element_mode("face")
    doc.set_element_sel(a.uid, _faces(0))
    doc.set_element_sel(b.uid, _faces(1))

    doc.select([b.uid])

    assert doc.selection == {b.uid}
    assert set(doc.element_sel) == {b.uid}
    assert doc.selection_problems() == []


def test_set_element_sel_never_removes_the_object_from_the_selection() -> None:
    doc, (a, _b) = _doc()
    doc.set_element_mode("face")
    doc.set_element_sel(a.uid, _faces(0))
    doc.set_element_sel(a.uid, None)
    assert doc.selection == {a.uid}
    assert doc.element_sel == {}


# --- the doors that remove an object ----------------------------------------------


def test_hiding_the_edited_shape_leaves_an_empty_edited_set_in_the_same_mode() -> None:
    doc, (a, _b) = _doc()
    doc.select([a.uid])
    doc.set_element_mode("face")
    doc.set_element_sel(a.uid, _faces(0))
    doc.set_visibility({a.uid: False})
    assert doc.element_mode == "face"
    assert doc.selection == set()
    assert doc.element_sel == {}


def test_undoing_the_creation_of_the_edited_shape_keeps_the_mode_and_redo_does_not_reselect() -> (
    None
):
    doc, (a,) = _doc(1)
    doc.set_element_mode("face")
    doc.set_element_sel(a.uid, _faces(0))
    assert doc.undo()
    assert doc.element_mode == "face"
    assert doc.selection == set()
    assert doc.element_sel == {}
    assert doc.redo()
    assert doc.selection == set(), "redo brings the shape back but does not select it"


def test_undoing_a_mesh_edit_keeps_the_shape_selected_and_drops_only_its_picks() -> None:
    doc, (a,) = _doc(1)
    doc.set_element_mode("face")
    doc.set_mesh(a.uid, bp.cylinder(), select=_faces(1))
    assert doc.undo()
    assert doc.selection == {a.uid}
    assert doc.element_sel == {}


def test_an_undone_show_that_hides_the_selected_shape_deselects_it() -> None:
    doc, (a, b) = _doc()
    doc.set_visibility({a.uid: False})
    doc.set_visibility({a.uid: True})
    doc.select([a.uid])
    assert doc.undo()  # the show is undone, so a is hidden again
    assert doc.selection == set()
    assert doc.selection_problems() == []


def test_joining_in_object_mode_selection_still_holds_the_target_not_the_absorbed() -> None:
    doc, (a, b) = _doc()
    merged = mesh_ops.join(
        [a, b], eps=1e-4, world=[doc.world_matrix(a.uid), doc.world_matrix(b.uid)]
    )
    doc.select([a.uid, b.uid])
    doc.join_objects(a.uid, merged, [b.uid])
    assert doc.selection == {a.uid}


def test_separating_in_an_element_mode_swaps_the_source_for_its_pieces() -> None:
    from realmspinner.kernels.mesh import separate as sep

    doc, (a, b) = _doc()
    joined = mesh_ops.join(
        [a, b], eps=1e-4, world=[doc.world_matrix(a.uid), doc.world_matrix(b.uid)]
    )
    doc.join_objects(a.uid, joined, [b.uid])
    doc.select([a.uid])
    doc.set_element_mode("vertex")
    pieces = doc.separate(a.uid, sep.by_loose_parts(doc.by_uid(a.uid).mesh))
    assert doc.selection == {p.uid for p in pieces}
    assert doc.element_sel == {}


# --- the gates that read the selection --------------------------------------------


def test_delete_is_greyed_in_an_element_mode_while_nothing_is_picked() -> None:
    """Delete there deletes picks. The edited shape stays selected, so a gate on
    the object selection enabled it, ran over nothing and answered ``ran``."""
    doc, (a, _b) = _doc()
    doc.select([a.uid])
    doc.set_element_mode("face")
    delete = clay_ops.get("delete")
    assert not delete.enabled(doc)
    assert clay_ops.reason_for(delete, doc) == "Select something in the viewport first."
    head = doc.history.head
    assert clay_ops.run(_Ctx(), doc, delete) is False
    assert doc.history.head == head


def test_delete_in_object_mode_is_still_gated_on_the_object_selection() -> None:
    doc, (a, _b) = _doc()
    delete = clay_ops.get("delete")
    assert not delete.enabled(doc)
    assert clay_ops.reason_for(delete, doc) == "Select something in the viewport first."
    doc.select([a.uid])
    assert delete.enabled(doc)


def test_separate_selection_splits_only_the_shapes_that_have_picks() -> None:
    doc, (a, b) = _doc()
    doc.select([a.uid, b.uid])
    doc.set_element_mode("face")
    doc.set_element_sel(a.uid, _faces(0, 1))
    ctx = _Ctx()

    assert clay_ops.run(ctx, doc, clay_ops.get("separate-selection")) is True

    assert ctx.toasts == [], "the shape with nothing picked was not asked to separate"
    assert b.uid in {o.uid for o in doc.objects}, "and was left whole"
    assert len(doc.objects) == 3
    assert doc.selection_problems() == []


def test_select_all_in_an_element_mode_picks_everything_on_the_shapes_being_edited() -> None:
    doc, (a, b) = _doc()
    doc.select([a.uid])
    doc.set_element_mode("face")
    clay_ops.run(_Ctx(), doc, clay_ops.get("select-all"))
    assert set(doc.element_sel) == {a.uid}, "the other shape is not pulled into the edit"
    assert doc.selection == {a.uid}
    assert doc.element_sel_of(a.uid).faces.tolist() == list(range(bm.face_count(a.mesh)))


def test_invert_in_an_element_mode_stays_on_the_shapes_being_edited() -> None:
    doc, (a, b) = _doc()
    doc.select([a.uid])
    doc.set_element_mode("face")
    doc.set_element_sel(a.uid, _faces(0))
    clay_ops.run(_Ctx(), doc, clay_ops.get("select-invert"))
    assert set(doc.element_sel) == {a.uid}
    assert doc.element_sel_of(a.uid).faces.tolist() == [1, 2, 3, 4, 5]
    assert doc.selection == {a.uid}


def test_select_all_with_no_shape_being_edited_still_picks_every_visible_shape() -> None:
    doc, (a, b) = _doc()
    doc.set_element_mode("face")
    clay_ops.run(_Ctx(), doc, clay_ops.get("select-all"))
    assert set(doc.element_sel) == {a.uid, b.uid}
    assert doc.selection == {a.uid, b.uid}


def test_shade_flat_in_face_mode_on_flat_faces_reports_that_it_did_not_run() -> None:
    """Found by the fuzz: the face branch returned ``None`` (read as ran) for a
    press that pushed no step, the object branch's clay-38 over again."""
    doc, (a, _b) = _doc()
    doc.set_element_mode("face")
    doc.set_element_sel(a.uid, _faces(0, 1))
    head = doc.history.head
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("shade-flat")) is False
    assert doc.history.head == head
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("shade-smooth")) is True
    assert doc.history.head != head
