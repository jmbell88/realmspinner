"""The 2026-10-07 Clay audit, outliner and palette-strip panes.

Each test's name is the claim. Panes cannot be driven headlessly, so the
decisions are pure helpers beside the draw calls and the document-level
outcomes go through the pane's non-imgui functions.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio.modes.clay.ui.panes import outliner as ol
from realmspinner.studio.modes.clay.ui.panes import palette_strip as ps
from realmspinner.studio.modes.clay.ui.panes import swatches as sw


class _Ctx:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def toast(self, message: str, level: str = "info", **_kw: Any) -> None:
        if level == "error":
            self.errors.append(message)


def _obj(name: str, uid: int | None = None, parent: int | None = None) -> bd.Obj:
    return bd.Obj(uid=bd.new_uid() if uid is None else uid, name=name, mesh=bp.box(), parent=parent)


def _state() -> SimpleNamespace:
    return SimpleNamespace(outliner_collapsed=set(), outliner_anchor=0, renaming=0)


# --- clay-23 (outliner half) ---------------------------------------------------


def test_collapsing_an_object_in_one_tab_does_not_collapse_the_same_uid_in_another_tab() -> None:
    state = _state()
    ol.set_collapsed(state, "bd1", 1, True)
    assert ol.is_collapsed(state, "bd1", 1)
    assert not ol.is_collapsed(state, "bd2", 1), "two files both carry uid 1"
    ol.set_collapsed(state, "bd1", 1, False)
    assert not ol.is_collapsed(state, "bd1", 1)


def test_a_collapsed_parent_hides_its_children_only_in_the_tab_that_collapsed_it() -> None:
    def doc_with_a_pair() -> bd.ClayDoc:
        doc = bd.ClayDoc()
        doc.add_object(_obj("P", uid=1))
        doc.add_object(_obj("K", uid=2, parent=1))
        return doc

    state = _state()
    ol.set_collapsed(state, "bd1", 1, True)
    assert [o.name for o, _d, _h in ol._visible_rows(state, "bd1", doc_with_a_pair(), "")] == ["P"]
    assert [o.name for o, _d, _h in ol._visible_rows(state, "bd2", doc_with_a_pair(), "")] == [
        "P",
        "K",
    ]


def test_a_shift_click_does_not_range_from_another_documents_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ol.imgui, "get_io", lambda: SimpleNamespace(key_shift=True, key_ctrl=False)
    )
    state = _state()
    other = bd.ClayDoc()
    other.add_object(_obj("x", uid=1))
    ol._click(state, other, other.objects[0], tab_uid="bd1")  # anchor = (bd1, 1)

    doc = bd.ClayDoc()
    for uid in (1, 2, 3):
        doc.add_object(_obj(f"n{uid}", uid=uid))
    ol._click(state, doc, doc.by_uid(3), tab_uid="bd2")
    assert doc.selection == {3}, "the anchor belonged to the other tab; the click is a plain one"
    ol._click(state, doc, doc.by_uid(1), tab_uid="bd2")  # now bd2 has its own anchor (3)
    assert doc.selection == {1, 2, 3}


def test_a_rename_in_progress_ends_when_the_active_tab_changes() -> None:
    state = _state()
    ol.begin_rename(state, "bd1", 1)
    assert ol.renaming_uid(state, "bd1") == 1
    assert ol.renaming_uid(state, "bd2") == 0, "the same uid in another tab is not being renamed"
    ol.settle_rename(state, "bd2")
    assert ol.renaming_uid(state, "bd1") == 0, "switching tabs ended the rename for good"


# --- clay-37 -----------------------------------------------------------------


def test_a_row_in_rename_asks_for_keyboard_focus_on_its_first_frame_and_escape_ends_it() -> None:
    state = _state()
    ol.begin_rename(state, "bd1", 4)
    assert ol._first_rename_frame(state, ("bd1", 4)) is True
    assert ol._first_rename_frame(state, ("bd1", 4)) is False, "only the first frame"
    ol.end_rename(state)
    ol.begin_rename(state, "bd1", 4)
    assert ol._first_rename_frame(state, ("bd1", 4)) is True, "a second rename asks again"

    # Escape: ends the rename and commits nothing, even with text typed.
    assert ol._rename_outcome("typed", "old", escaped=True, deactivated=True) == (None, True)
    # Click away / Enter after an edit commits; a click away untouched just ends.
    assert ol._rename_outcome("typed", "old", escaped=False, deactivated=True) == ("typed", True)
    assert ol._rename_outcome("old", "old", escaped=False, deactivated=True) == (None, True)
    # Still typing: nothing yet.
    assert ol._rename_outcome("old", "old", escaped=False, deactivated=False) == (None, False)


# --- clay-38 -----------------------------------------------------------------


def _two_boxes() -> tuple[bd.ClayDoc, bd.Obj, bd.Obj]:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    b = doc.add_object(_obj("B"))
    doc.add_material()
    doc.add_material()
    return doc, a, b


def test_the_palette_hint_does_not_promise_ctrl_click_editing_without_exactly_one_object() -> None:
    doc, a, b = _two_boxes()
    # Object mode, two selected.
    doc.select([a.uid, b.uid])
    assert "Ctrl+click" not in ps.click_hint(doc)
    assert "Select one object to edit its slot" in ps.click_hint(doc)
    # Face mode, faces on both objects.
    doc.set_element_mode("face")
    for o in (a, b):
        doc.set_element_sel(o.uid, el.ElementSel(faces=np.array([0], dtype="i4")))
    assert "Ctrl+click" not in ps.click_hint(doc)
    assert "Select one object to edit its slot" in ps.click_hint(doc)
    # Vertex mode, nothing picked and both shapes still being edited: no
    # "Click picks the slot" promise either.
    doc.set_element_mode("vertex")
    doc.clear_element_sel()
    hint = ps.click_hint(doc)
    assert "Click picks the slot" not in hint
    assert "Select one object to edit its slot" in hint
    # Exactly one object keeps the promise. ``clear_element_sel`` no longer
    # deselects, so narrow the edited set to the one shape first.
    doc.select([a.uid])
    doc.set_element_sel(a.uid, el.ElementSel(verts=np.array([0], dtype="i4")))
    assert doc.selection == {a.uid}
    assert "Click picks the slot Properties edits" in ps.click_hint(doc)
    doc.set_element_mode("object")
    doc.select([a.uid])
    assert "Ctrl+click" in ps.click_hint(doc)


# --- clay-39 (docs only: the behaviour is the contract) ------------------------


def test_the_palette_strip_repaints_every_selected_object_in_object_mode_as_one_undo_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sw, "_ctrl_held", lambda: False)
    doc, a, b = _two_boxes()
    doc.select([a.uid, b.uid])
    steps = len(doc.history)

    sw.click_slot(_Ctx(), doc, 2)

    assert [doc.by_uid(o.uid).material for o in (a, b)] == [2, 2]
    assert len(doc.history) == steps + 1, "one click, one undo step"
    assert doc.undo()
    assert [doc.by_uid(o.uid).material for o in (a, b)] == [0, 0]


# --- clay-80 -----------------------------------------------------------------


def _tree() -> tuple[bd.ClayDoc, bd.Obj, bd.Obj, bd.Obj, bd.Obj]:
    doc = bd.ClayDoc()
    b = doc.add_object(_obj("B"))
    c1 = doc.add_object(_obj("C1", parent=b.uid))
    c2 = doc.add_object(_obj("C2", parent=b.uid))
    a = doc.add_object(_obj("A"))
    return doc, a, b, c1, c2


def _names(doc: bd.ClayDoc) -> list[tuple[str, int]]:
    return [(o.name, d) for o, d, _h in ol._tree_rows(doc)]


def test_a_between_rows_drop_beside_a_child_changes_what_the_outliner_draws_or_is_refused() -> None:
    doc, a, b, c1, _c2 = _tree()
    before = _names(doc)
    steps = len(doc.history)

    ol._apply_drop(_Ctx(), doc, a.uid, c1, doc.index_of(c1.uid), "before")

    assert doc.by_uid(a.uid).parent == b.uid, "dropped beside a child, it joins that child's parent"
    assert _names(doc) != before
    assert ("A", 1) in _names(doc)
    assert len(doc.history) == steps + 1, "reparent plus reorder is one undo step"
    assert doc.undo()
    assert _names(doc) == before and doc.by_uid(a.uid).parent is None


def test_a_between_rows_drop_that_would_make_a_cycle_is_toasted_and_pushes_nothing() -> None:
    doc, a, b, c1, _c2 = _tree()
    # B beside its own child C1 would parent B under B.
    ctx = _Ctx()
    steps = len(doc.history)
    before = _names(doc)

    ol._apply_drop(ctx, doc, b.uid, c1, doc.index_of(c1.uid), "after")

    assert ctx.errors, "the refusal was never shown"
    assert len(doc.history) == steps
    assert _names(doc) == before and doc.by_uid(b.uid).parent is None


# --- clay-81 -----------------------------------------------------------------


def test_the_outliner_row_menu_acts_on_the_selection_when_the_row_is_selected_or_shows_no_selection_shortcut() -> None:  # noqa: E501
    doc, a, b, c1, _c2 = _tree()
    doc.select([a.uid, b.uid])
    # Right-clicking a selected row targets the whole selection...
    assert ol._menu_targets(doc, a) == sorted([a.uid, b.uid])
    # ...and the Del key hint is honest because the key would do the same.
    assert ol._delete_hint(doc, ol._menu_targets(doc, a)) == "Del"
    # A row outside the selection targets itself only, and the key would not.
    assert ol._menu_targets(doc, c1) == [c1.uid]
    assert ol._delete_hint(doc, [c1.uid]) == ""
    # In an element mode the Del key deletes elements, never these objects.
    doc.set_element_mode("face")
    assert ol._delete_hint(doc, [a.uid]) == ""


def test_the_outliner_row_menu_delete_removes_the_whole_selection_as_one_undo_step() -> None:
    doc, a, b, c1, c2 = _tree()
    doc.select([a.uid, c1.uid])
    steps = len(doc.history)

    assert ol._remove_objects(_Ctx(), doc, ol._menu_targets(doc, a)) == []

    assert sorted(o.name for o in doc.objects) == ["B", "C2"]
    assert len(doc.history) == steps + 1
    assert doc.undo()
    assert len(doc.objects) == 4
