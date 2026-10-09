"""The 2026-10-09 second audit's colour/material edit-slot and Inspector findings:
clay-57 (the UI half), 65, 66, 67, 68, 122 (state.py) and 129.

The first five are one design. Ctrl+clicking a swatch makes it the slot the
Colours & textures page edits (``ClayTab.edit_slots``), and every pane that names
"the shape's colour" -- the Colour card's outline, its "custom" swatch, the header
dot, the line under the grid, the page itself -- has to name the same slot, and
that slot has to stop being the Ctrl+clicked one the moment something else changes
what the shape wears.
"""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from _ui_context import imgui_context

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio.modes.clay import starter_colours as SC
from realmspinner.studio.modes.clay import state as clay_state
from realmspinner.studio.modes.clay.state import ClayState, ClayTab
from realmspinner.studio.modes.clay.ui import inspector_size
from realmspinner.studio.modes.clay.ui.panes import inspector as clay_inspector
from realmspinner.studio.modes.clay.ui.panes import outliner as clay_outliner
from realmspinner.studio.modes.clay.ui.panes import props as clay_props
from realmspinner.studio.modes.clay.ui.panes import swatches
from realmspinner.studio.modes.clay.ui.panes import tools as clay_tools

_CLAY = Path(clay_state.__file__).parent


class _Ctx:
    def __init__(self, state: Any = None) -> None:
        self.toasts: list[str] = []
        self.state = state

    def toast(self, message: str, level: str = "info") -> None:
        self.toasts.append(message)


def _tab_doc(kind: str = "box") -> tuple[ClayTab, bd.ClayDoc, _Ctx, Any]:
    """A tab over a document with one *kind* shape wearing the starter grey (slot 0)."""
    doc = bd.ClayDoc()
    state = ClayState()
    tab = state.add(ClayTab(doc=doc))
    ctx = _Ctx(SimpleNamespace(clay=state))
    obj = clay_tools.add_primitive(ctx, doc, kind)
    return tab, doc, ctx, obj


def _red() -> Any:
    return replace(bd.default_material("Red"), base_color_factor=SC.by_name("Red").factor)


def _ctrl_click(monkeypatch: pytest.MonkeyPatch, ctx: Any, doc: Any, obj: Any, slot: int) -> None:
    monkeypatch.setattr(swatches, "_ctrl_held", lambda: True)
    swatches._pick_slot(ctx, doc, obj, slot)


def _plain_click(monkeypatch: pytest.MonkeyPatch, ctx: Any, doc: Any, obj: Any, slot: int) -> None:
    monkeypatch.setattr(swatches, "_ctrl_held", lambda: False)
    swatches._pick_slot(ctx, doc, obj, slot)


def _frame(imgui: Any, draw: Any) -> None:
    imgui.new_frame()
    imgui.begin("##host")
    draw()
    imgui.end()
    imgui.end_frame()


# --- clay-66: the override expires ----------------------------------------------------


def test_a_starter_colour_click_makes_the_page_edit_the_new_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tab, doc, ctx, obj = _tab_doc()
    blue = doc.add_material()
    _ctrl_click(monkeypatch, ctx, doc, obj, blue)
    assert tab.edit_slot(doc, doc.by_uid(obj.uid)) == blue, "the premise: the page edits blue"

    painted = SC.paint(ctx, doc, [obj.uid], SC.by_name("Red"))

    assert tab.edit_slot(doc, doc.by_uid(obj.uid)) == painted, "the page kept editing the old slot"


def test_a_starter_colour_click_that_changes_nothing_still_hands_the_page_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shape already wears Red, the user Ctrl+clicked the spare slot, then clicked
    the Red starter: ``obj.material`` does not move, so expiry on it alone would keep
    the page on the spare slot while the card shows Red freshly clicked."""
    tab, doc, ctx, obj = _tab_doc()
    red = SC.paint(ctx, doc, [obj.uid], SC.by_name("Red"))
    _ctrl_click(monkeypatch, ctx, doc, obj, 0)
    assert tab.edit_slot(doc, doc.by_uid(obj.uid)) == 0

    SC.paint(ctx, doc, [obj.uid], SC.by_name("Red"))

    assert tab.edit_slot(doc, doc.by_uid(obj.uid)) == red


def test_the_pages_add_button_makes_the_page_edit_the_new_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tab, doc, ctx, obj = _tab_doc()
    spare = doc.add_material()
    _ctrl_click(monkeypatch, ctx, doc, obj, spare)

    fresh = doc.add_material_and_assign(obj.uid, repaint=True)

    assert tab.edit_slot(doc, doc.by_uid(obj.uid)) == fresh


def test_undoing_a_repaint_hands_the_page_back_to_the_slot_the_shape_wears_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tab, doc, ctx, obj = _tab_doc()
    other = doc.add_material()
    spare = doc.add_material()
    _plain_click(monkeypatch, ctx, doc, obj, other)
    _ctrl_click(monkeypatch, ctx, doc, obj, spare)
    assert tab.edit_slot(doc, doc.by_uid(obj.uid)) == spare

    assert doc.undo()  # the repaint

    assert doc.by_uid(obj.uid).material == 0
    page = tab.edit_slot(doc, doc.by_uid(obj.uid))
    assert page == 0, "the page kept editing the Ctrl+clicked slot"


def test_an_edit_slot_expires_when_the_palette_shrinks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``remove_material`` renumbers every slot above the one it drops, so an
    override recorded against the longer palette names nothing, or the wrong colour."""
    tab, doc, ctx, obj = _tab_doc()
    doc.add_material()
    last = doc.add_material()
    _ctrl_click(monkeypatch, ctx, doc, obj, last)
    assert tab.edit_slot(doc, doc.by_uid(obj.uid)) == last

    assert doc.remove_material(1)

    assert tab.edit_slot(doc, doc.by_uid(obj.uid)) == doc.by_uid(obj.uid).material
    assert obj.uid not in tab.edit_slots, "an expired override is dropped, not kept"


def test_a_ctrl_click_still_holds_while_nothing_the_shape_wears_has_changed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tab, doc, ctx, obj = _tab_doc()
    spare = doc.add_material()
    _ctrl_click(monkeypatch, ctx, doc, obj, spare)
    doc.set_props(obj.uid, name="Renamed")
    doc.select([obj.uid])
    assert tab.edit_slot(doc, doc.by_uid(obj.uid)) == spare


# --- clay-57 (UI half): Remove on the slot the page edits -------------------------------


def _remove_clicked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the page's Remove button report a press and nothing else."""
    monkeypatch.setattr(
        clay_props.widgets,
        "disabled_button",
        lambda label, enabled=True, *a, **k: label.startswith("Remove") and bool(enabled),
    )
    monkeypatch.setattr(clay_props.controls, "small_button", lambda *a, **k: False)


def test_removing_a_ctrl_clicked_unused_slot_leaves_the_objects_default_slot_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tab, doc, ctx, obj = _tab_doc("cylinder")
    red = SC.paint(ctx, doc, [obj.uid], SC.by_name("Red"))
    doc.add_material()
    _ctrl_click(monkeypatch, ctx, doc, obj, 0)  # the starter grey, now unused
    index = clay_props._page_slot(tab, doc, doc.by_uid(obj.uid))
    assert index == 0
    wearing = doc.materials[red].base_color_factor
    _remove_clicked(monkeypatch)

    with imgui_context(monkeypatch) as imgui:
        _frame(imgui, lambda: clay_props._palette_row(doc, doc.by_uid(obj.uid), index))

    after = doc.by_uid(obj.uid)
    assert len(doc.materials) == 2, "the premise: the unused slot went"
    assert doc.materials[after.material].base_color_factor == wearing, "the default slot moved"
    assert {int(s) for s in after.mesh.material} == {after.material}


def test_after_removing_the_slot_the_page_edits_the_page_edits_what_the_shape_wears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tab, doc, ctx, obj = _tab_doc("cylinder")
    SC.paint(ctx, doc, [obj.uid], SC.by_name("Red"))
    doc.add_material()
    _ctrl_click(monkeypatch, ctx, doc, obj, 0)
    _remove_clicked(monkeypatch)

    with imgui_context(monkeypatch) as imgui:
        _frame(imgui, lambda: clay_props._palette_row(doc, doc.by_uid(obj.uid), 0))

    after = doc.by_uid(obj.uid)
    page = tab.edit_slot(doc, after)
    assert doc.materials[page].base_color_factor == SC.by_name("Red").factor, (
        "the page now edits a slot renumbered into the removed one's place"
    )


# --- clay-65: one slot, named the same by every pane -----------------------------------


def _draw_card(
    monkeypatch: pytest.MonkeyPatch, tab: ClayTab, doc: bd.ClayDoc, ctx: Any, uids: list[int]
) -> tuple[Any, dict[str, bool]]:
    """Draw the Colour card; -> what ``draw_slots`` was told is current, and each
    starter/custom swatch's selected flag."""
    seen: dict[str, Any] = {}
    flags: dict[str, bool] = {}

    def fake_slots(ctx_: Any, doc_: Any, *, current: Any, saving: bool, **kw: Any) -> None:
        seen["current"] = current
        seen["kw"] = kw

    def fake_swatch(label: str, colour: Any, side: float, **kw: Any) -> bool:
        flags[label] = bool(kw.get("selected"))
        return False

    monkeypatch.setattr(swatches, "draw_slots", fake_slots)
    monkeypatch.setattr(swatches, "_swatch", fake_swatch)
    state = SimpleNamespace(inspector_page="")
    with imgui_context(monkeypatch) as imgui:
        _frame(imgui, lambda: clay_inspector._colour_card(ctx, state, tab, doc, uids))
    return seen, flags


def test_the_palette_outline_follows_the_slot_a_ctrl_click_chose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tab, doc, ctx, obj = _tab_doc()
    red = doc.add_material(_red())
    _ctrl_click(monkeypatch, ctx, doc, obj, red)
    assert tab.edit_slot(doc, doc.by_uid(obj.uid)) == red

    seen, flags = _draw_card(monkeypatch, tab, doc, ctx, [obj.uid])

    assert seen["current"] == red, "the grid outlines the default, not the slot the page edits"
    assert flags["##inspsw/Red"], "the starter that holds the edited slot's colour is outlined"


def test_the_header_dot_shows_the_slot_a_ctrl_click_chose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tab, doc, ctx, obj = _tab_doc()
    red = doc.add_material(_red())
    _ctrl_click(monkeypatch, ctx, doc, obj, red)
    drawn: list[Any] = []
    real = swatches._swatch_colour
    monkeypatch.setattr(
        swatches, "_swatch_colour", lambda material: drawn.append(material) or real(material)
    )

    with imgui_context(monkeypatch) as imgui:
        _frame(imgui, lambda: clay_inspector._header_row(doc, doc.by_uid(obj.uid), tab))

    assert drawn == [doc.materials[red]]


# --- clay-67: what the faces wear -------------------------------------------------------


def test_a_shape_whose_every_face_wears_one_slot_shows_that_slot_in_the_colour_card_and_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tab, doc, ctx, obj = _tab_doc()
    red = doc.add_material(_red())
    doc.paint_faces(obj.uid, list(range(len(obj.mesh.material))), red)  # face mode paint
    shape = doc.by_uid(obj.uid)
    assert shape.material == 0, "the premise: the default for new faces is untouched"

    assert tab.edit_slot(doc, shape) == red, "the page edits the default, not what renders"
    seen, flags = _draw_card(monkeypatch, tab, doc, ctx, [obj.uid])
    assert seen["current"] == red
    assert flags["##inspsw/Red"]

    drawn: list[Any] = []
    real = swatches._swatch_colour
    monkeypatch.setattr(
        swatches, "_swatch_colour", lambda material: drawn.append(material) or real(material)
    )
    with imgui_context(monkeypatch) as imgui:
        _frame(imgui, lambda: clay_inspector._header_row(doc, shape))
    assert drawn == [doc.materials[red]], "the header dot is grey on a red shape"


def test_a_shape_with_faces_in_two_slots_is_mixed_and_the_page_edits_its_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tab, doc, ctx, obj = _tab_doc()
    red = doc.add_material(_red())
    doc.paint_faces(obj.uid, [0, 1], red)
    shape = doc.by_uid(obj.uid)

    assert tab.shown_slot(doc, shape) is None, "mixed names no one slot"
    assert tab.edit_slot(doc, shape) == shape.material == 0
    seen, _flags = _draw_card(monkeypatch, tab, doc, ctx, [obj.uid])
    assert seen["current"] is None
    assert seen["kw"].get("mixed") is True, "the line under the grid says so"


def test_a_ctrl_click_overrides_what_the_faces_wear() -> None:
    tab, doc, ctx, obj = _tab_doc()
    red = doc.add_material(_red())
    doc.paint_faces(obj.uid, list(range(len(obj.mesh.material))), red)
    tab.choose_edit_slot(doc, doc.by_uid(obj.uid), 0)
    assert tab.edit_slot(doc, doc.by_uid(obj.uid)) == 0
    assert tab.shown_slot(doc, doc.by_uid(obj.uid)) == 0


# --- clay-68: the Size card measures a mesh once ----------------------------------------


def test_the_size_card_measures_a_large_mesh_once_not_every_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc = bd.ClayDoc()
    params = {**bp.GENERATORS["uv_sphere"][0], "segments": 600, "rings": 300}
    mesh = bp.GENERATORS["uv_sphere"][1](**params)
    assert len(mesh.positions) > 130_000
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Big", mesh=mesh))
    calls = 0
    real = bm.bounds

    def counted(m: Any) -> Any:
        nonlocal calls
        if len(m.positions) > 1000:  # a full pass, not the two-point stand-in's
            calls += 1
        return real(m)

    monkeypatch.setattr(bm, "bounds", counted)
    state, ctx = ClayState(), _Ctx()
    with imgui_context(monkeypatch) as imgui:
        for _ in range(100):
            _frame(imgui, lambda: inspector_size.draw(ctx, state, doc, obj))
        for _ in range(100):
            _frame(imgui, lambda: clay_props._dimensions(doc, obj, ctx=ctx, state=state))

    assert calls == 1, f"{calls} full passes over {len(mesh.positions)} vertices in 200 frames"


def test_the_size_card_re_measures_when_the_mesh_is_replaced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
    first = clay_props.measured(obj, obj.mesh, doc.world_matrix(obj.uid))
    assert first is not None
    doc.set_mesh(obj.uid, bp.box(size=(2.0, 2.0, 2.0)))
    after = clay_props.measured(
        doc.by_uid(obj.uid), doc.by_uid(obj.uid).mesh, doc.world_matrix(obj.uid)
    )
    assert after is not None
    assert after[1].tolist() == [2.0, 2.0, 2.0]
    assert first[1].tolist() == [1.0, 1.0, 1.0]


# --- clay-129: a hidden outliner row ------------------------------------------------------


def _hidden_and_selected() -> tuple[bd.ClayDoc, Any, Any]:
    doc, ctx = bd.ClayDoc(), _Ctx()
    keep = clay_tools.add_primitive(ctx, doc, "box")
    gone = clay_tools.add_primitive(ctx, doc, "box")
    doc.set_visibility({gone.uid: False})
    doc.select([keep.uid])
    return doc, keep, gone


def test_clicking_a_hidden_outliner_row_keeps_the_current_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc, keep, gone = _hidden_and_selected()
    state = ClayState()
    with imgui_context(monkeypatch):
        clay_outliner._click(state, doc, doc.by_uid(gone.uid), "t1")
    assert doc.selection == {keep.uid}, "the click replaced the selection with nothing"


def test_clicking_a_hidden_row_in_an_element_mode_does_not_leave_the_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc, keep, gone = _hidden_and_selected()
    doc.set_element_mode("face")
    state = ClayState()
    with imgui_context(monkeypatch):
        clay_outliner._click(state, doc, doc.by_uid(gone.uid), "t1")
    assert doc.element_mode == "face"
    assert keep.uid in doc.selection


def test_the_row_menu_greys_duplicate_for_a_hidden_row_and_says_why() -> None:
    doc, keep, gone = _hidden_and_selected()
    why = clay_outliner._duplicate_reason(doc, doc.by_uid(gone.uid))
    assert why and "hidden" in why.lower()
    assert clay_outliner._duplicate_reason(doc, doc.by_uid(keep.uid)) == ""


def test_the_row_menu_opens_on_a_hidden_row_without_selecting_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc, keep, gone = _hidden_and_selected()
    monkeypatch.setattr(clay_outliner.imgui, "begin_popup_context_item", lambda *a, **k: True)
    monkeypatch.setattr(clay_outliner.imgui, "end_popup", lambda: None)
    monkeypatch.setattr(clay_outliner.widgets, "popup_chrome", lambda **k: None)
    monkeypatch.setattr(clay_outliner.widgets, "divider", lambda: None)
    shown: dict[str, bool] = {}

    def menu_item(label: str, shortcut: str = "", selected: bool = False,
                  enabled: bool = True, **kw: Any) -> tuple[bool, bool]:
        shown[label] = enabled
        return (False, False)

    monkeypatch.setattr(clay_outliner.controls, "menu_item", menu_item)
    clay_outliner._context_menu(_Ctx(), ClayState(), doc, doc.by_uid(gone.uid), "t1")
    assert doc.selection == {keep.uid}
    duplicate = next(label for label in shown if "Duplicate" in label)
    assert shown[duplicate] is False


# --- clay-122 (state.py): the annotation names a pane that was deleted -------------------


def test_the_clay_tab_annotations_do_not_name_the_deleted_document_tab() -> None:
    text = (_CLAY / "state.py").read_text(encoding="utf-8")
    ast.parse(text)
    assert "the Document tab names it" not in text
