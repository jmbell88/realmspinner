"""The 2026-10-07 Clay audit, panes slice: the Material tab, the texture link to
Inker, and the pane tooltips and greyed buttons that explained nothing.

Each test's name is the claim, and each failed against the code it was written
for before the fix landed.
"""

from __future__ import annotations

import dataclasses
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from _ui_context import imgui_context

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.pixel import Document
from realmspinner.studio.modes.clay import texture_link as tl
from realmspinner.studio.modes.clay.state import ClayState, ClayTab
from realmspinner.studio.modes.clay.ui.panes import bridge as clay_bridge
from realmspinner.studio.modes.clay.ui.panes import header as clay_header
from realmspinner.studio.modes.clay.ui.panes import outliner as clay_outliner
from realmspinner.studio.modes.clay.ui.panes import props as clay_props
from realmspinner.studio.modes.inker import opening as inker_open
from realmspinner.studio.state import AppState

RED = (255, 0, 77, 255)


class _Inker:
    def __init__(self) -> None:
        self.docs: list[Any] = []

    def activate(self, uid: str) -> None:
        pass


class _Ctx:
    def __init__(self) -> None:
        self.state = AppState()
        self.state.clay = ClayState()
        self.state.inker = _Inker()
        self.toasts: list[tuple[str, str]] = []

    def toast(self, text: str, level: str = "info", **_kw: Any) -> None:
        self.toasts.append((text, level))

    def busy(self, key: str) -> bool:
        return False

    def submit(self, *args: Any, **kwargs: Any) -> bool:
        return True


def _tab(ctx: _Ctx, doc: bd.ClayDoc | None = None) -> ClayTab:
    tab = ClayTab(doc=doc if doc is not None else bd.ClayDoc(), title="Crate")
    ctx.state.clay.add(tab)
    return tab


def _textured(ctx: _Ctx) -> ClayTab:
    doc = bd.ClayDoc()
    assert doc.add_texture(0, 32)
    return _tab(ctx, doc)


def _open_in_inker(monkeypatch: pytest.MonkeyPatch, ctx: _Ctx, tab: ClayTab) -> Any:
    """Press Edit texture in Inker for real and let the open land: -> the Inker tab.

    ``open_pixels`` is the one seam stubbed. Its task thread builds the document
    palette-locked to PICO-8 (``set_palette`` snaps every visible pixel) before
    it is adopted, so the stand-in does the same and hands it to ``on_open``.
    """
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    monkeypatch.setattr(inker_open, "open_pixels", lambda *a, **k: calls.append((a, k)))
    tl.edit_in_inker(ctx, tab, 0)
    assert len(calls) == 1
    args, kw = calls[0]
    doc = Document.from_pixels(np.array(args[1], dtype=np.uint8), name="T")
    doc.set_palette([tuple(c) for c in kw["palette"]])
    inker = SimpleNamespace(uid="ink", title="Crate - texture", doc=doc)
    ctx.state.inker.docs.append(inker)
    kw["on_open"](inker)
    return inker


# --- clay-06 -----------------------------------------------------------------


def test_a_texture_painted_in_inker_and_closed_before_clay_is_drawn_still_lands_or_says_it_did_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _Ctx()
    tab = _textured(ctx)
    inker = _open_in_inker(monkeypatch, ctx, tab)
    tl.pull(ctx, tab)  # the open's own return (clay-50) is not what is under test
    before = tab.doc.materials[0].base_color
    assert inker.doc.fill((0, 0), RED, thresh=0)
    ctx.state.inker.docs.remove(inker)  # the user closes the texture's tab

    tl.pull(ctx, tab)

    landed = tab.doc.materials[0].base_color != before
    said = any("closed" in text or "dropped" in text for text, _level in ctx.toasts)
    assert landed or said, "a painting vanished with no word"
    assert landed
    assert tab.inker_links == [], "and the link is let go once it has landed"


def test_closing_an_untouched_inker_tab_drops_the_link_without_a_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _Ctx()
    tab = _textured(ctx)
    inker = _open_in_inker(monkeypatch, ctx, tab)
    tl.pull(ctx, tab)
    steps = len(tab.doc.history)
    ctx.state.inker.docs.remove(inker)

    tl.pull(ctx, tab)

    assert tab.inker_links == []
    assert len(tab.doc.history) == steps
    assert ctx.toasts == []


# --- clay-20 -----------------------------------------------------------------


def test_edit_texture_in_inker_refuses_a_texture_the_pull_would_later_refuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _Ctx()
    doc = bd.ClayDoc()
    side = tl.MAX_TEXTURE_SIDE + 76
    picture = (side, side, bytes([200, 200, 200, 255]) * (side * side))
    doc.set_material(0, replace(doc.materials[0], base_color=picture))
    tab = _tab(ctx, doc)
    opened: list[Any] = []
    monkeypatch.setattr(inker_open, "open_pixels", lambda *a, **k: opened.append((a, k)))

    tl.edit_in_inker(ctx, tab, 0)

    assert opened == [], "a document the first stroke would drop the link from"
    assert len(ctx.toasts) == 1 and ctx.toasts[0][1] == "error"
    assert str(side) in ctx.toasts[0][0] and str(tl.MAX_TEXTURE_SIDE) in ctx.toasts[0][0]


# --- clay-21 / clay-79 -------------------------------------------------------


def _uvless_box_tab(ctx: _Ctx) -> ClayTab:
    doc = bd.ClayDoc()
    box = dataclasses.replace(bp.box(), uv=None)
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=box, generator="box", params={}))
    return _tab(ctx, doc)


def _assigned(tab: ClayTab, side: int, index: int = 0) -> SimpleNamespace:
    image = {"width": side, "height": side, "rgba": bytes([255, 255, 255, 255]) * (side * side)}
    return SimpleNamespace(
        key=f"clay-mattex:{tab.uid}:{index}:base_color", result=image, tag=tab.doc.materials[index]
    )


def test_assigning_a_png_sets_the_colour_factor_to_white_and_unwraps_a_uvless_object() -> None:
    ctx = _Ctx()
    tab = _uvless_box_tab(ctx)
    doc = tab.doc
    assert doc.objects[0].mesh.uv is None
    assert doc.materials[0].base_color_factor[:3] != (1.0, 1.0, 1.0)
    # a slot that had been crisp: the assigned picture must not inherit that
    doc.set_material(0, replace(doc.materials[0], nearest=True))
    steps = len(doc.history)

    clay_props.on_task_done(ctx, _assigned(tab, 2))

    material = doc.materials[0]
    assert material.base_color[:2] == (2, 2)
    assert tuple(material.base_color_factor[:3]) == (1.0, 1.0, 1.0)
    assert material.base_color_factor[3] == pytest.approx(doc.materials[0].base_color_factor[3])
    assert material.nearest is False, "a smooth PNG is sampled smooth"
    assert doc.objects[0].mesh.uv is not None, "a texture on a mesh with no UVs is one texel"
    assert len(doc.history) == steps + 1, "one undo step"
    assert doc.undo()
    assert doc.materials[0].base_color is None and doc.objects[0].mesh.uv is None


def test_assigning_a_png_leaves_an_object_that_already_has_uvs_alone() -> None:
    ctx = _Ctx()
    doc = bd.ClayDoc()
    box = bp.box()
    assert box.uv is not None
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=box, generator="box", params={}))
    tab = _tab(ctx, doc)

    clay_props.on_task_done(ctx, _assigned(tab, 2))

    assert doc.objects[0].mesh.uv is box.uv


def test_assigning_a_png_past_the_texture_ceiling_is_refused_by_name() -> None:
    ctx = _Ctx()
    tab = _uvless_box_tab(ctx)
    side = tl.MAX_TEXTURE_SIDE + 1
    before = tab.doc.materials[0]

    clay_props.on_task_done(ctx, _assigned(tab, side))

    assert tab.doc.materials[0] is before
    assert len(ctx.toasts) == 1 and ctx.toasts[0][1] == "error"
    assert str(side) in ctx.toasts[0][0] and str(tl.MAX_TEXTURE_SIDE) in ctx.toasts[0][0]


def test_assigning_a_png_at_the_ceiling_still_lands() -> None:
    ctx = _Ctx()
    tab = _uvless_box_tab(ctx)

    clay_props.on_task_done(ctx, _assigned(tab, tl.MAX_TEXTURE_SIDE))

    assert tab.doc.materials[0].base_color[:2] == (tl.MAX_TEXTURE_SIDE,) * 2
    assert ctx.toasts == []


# --- clay-22 -----------------------------------------------------------------


def _two_slot_box() -> tuple[bd.ClayDoc, bd.Obj]:
    doc = bd.ClayDoc()
    doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box(), generator="box", params={})
    )
    doc.add_material(replace(bd.default_material("red"), base_color_factor=(1.0, 0.0, 0.0, 1.0)))
    obj = doc.objects[0]
    doc.paint_faces(obj.uid, [2, 3], 1)
    return doc, doc.by_uid(obj.uid)


def test_the_material_tab_names_the_slot_it_edits_and_the_uv_views_under_a_face_selection() -> None:
    doc, obj = _two_slot_box()
    doc.set_element_mode("face")
    doc.set_element_sel(obj.uid, el.ElementSel(faces=[2, 3]))
    index = int(obj.material)
    assert index == 0

    note = clay_props._edit_slot_note(doc, obj, index)

    assert "slot 0" in note, "the slot the fields edit is named"
    assert "slot 1" in note and "UV" in note, "and the slot the UV view is drawing"


def test_the_material_tab_says_only_the_slot_it_edits_when_the_uv_view_shows_the_same_one() -> None:
    doc, obj = _two_slot_box()
    note_object_mode = clay_props._edit_slot_note(doc, obj, 0)
    assert "slot 0" in note_object_mode and "UV" not in note_object_mode

    doc.set_element_mode("face")
    doc.set_element_sel(obj.uid, el.ElementSel(faces=[0, 1]))  # faces still on slot 0
    note = clay_props._edit_slot_note(doc, obj, 0)
    assert "slot 0" in note and "UV" not in note


# --- clay-50 -----------------------------------------------------------------


def test_the_first_return_from_inker_lands_the_palette_snap_even_with_no_stroke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _Ctx()
    tab = _textured(ctx)
    original = tab.doc.materials[0].base_color
    steps = len(tab.doc.history)

    inker = _open_in_inker(monkeypatch, ctx, tab)
    tl.pull_all(ctx)

    snapped = inker.doc.flatten(matte=False)
    assert snapped.tobytes() != original[2], "the fixture's grey is not on the PICO-8 table"
    assert tab.doc.materials[0].base_color == (32, 32, snapped.tobytes())
    assert len(tab.doc.history) == steps + 1, "one step for the snap, not a first-stroke recolour"

    assert inker.doc.fill((0, 0), RED, thresh=0)
    tl.pull_all(ctx)
    assert len(tab.doc.history) == steps + 2


# --- clay-73 -----------------------------------------------------------------


@pytest.mark.parametrize("mode", ["vertex", "edge"])
def test_a_swatch_click_in_vertex_mode_does_not_repaint_faces_outside_the_selection(
    mode: str,
) -> None:
    doc, obj = _two_slot_box()
    doc.set_props(obj.uid, material=0)
    doc.set_element_mode(mode)
    sel = el.ElementSel(verts=[0]) if mode == "vertex" else el.ElementSel(edges=[(0, 1)])
    doc.set_element_sel(obj.uid, sel)
    before = list(doc.by_uid(obj.uid).mesh.material)

    clay_props._pick_slot(_Ctx(), doc, doc.by_uid(obj.uid), 1)

    after = doc.by_uid(obj.uid)
    assert list(after.mesh.material) == before, "no face outside the selection was repainted"
    assert after.material == 1, "the click picked the slot the fields edit"


# --- clay-74 -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "noun"), [("face", "face"), ("vertex", "vertex"), ("edge", "edge")]
)
def test_properties_in_face_mode_with_nothing_selected_says_to_select_a_face(
    mode: str, noun: str
) -> None:
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="B", mesh=bp.box(), generator="box", params={}))
    doc.set_element_mode(mode)
    assert not doc.selection

    title, hint = clay_props._nothing_selected_text(doc)

    assert noun in hint
    assert "Click an object" not in hint


def test_properties_in_object_mode_with_nothing_selected_still_says_to_click_an_object() -> None:
    doc = bd.ClayDoc()
    assert clay_props._nothing_selected_text(doc) == (
        "Nothing selected",
        "Click an object in the viewport.",
    )


# --- clay-75 -----------------------------------------------------------------


def test_the_clear_texture_button_and_apply_scale_explain_why_they_are_greyed() -> None:
    assert clay_props._clear_texture_reason(None)
    assert clay_props._clear_texture_reason((1, 1, b"\0" * 4)) == ""
    assert clay_props._clear_parent_reason(SimpleNamespace(parent=None))
    assert clay_props._clear_parent_reason(SimpleNamespace(parent=3)) == ""
    assert clay_outliner._solo_reason(bd.ClayDoc())
    assert clay_outliner._solo_reason(SimpleNamespace(selection={1})) == ""
    assert clay_outliner._show_all_reason(0)
    assert clay_outliner._show_all_reason(2) == ""


def test_the_greyed_clear_texture_and_clear_parent_buttons_hand_their_reason_to_the_widget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, dict[str, Any]] = {}

    def spy(label, enabled, *a, **kw):
        seen[label] = {"enabled": enabled, **kw}
        return False

    monkeypatch.setattr(clay_props.widgets, "disabled_button", spy)
    monkeypatch.setattr(clay_props.controls, "small_button", lambda *a, **kw: False)
    monkeypatch.setattr(clay_outliner.widgets, "disabled_button", spy)
    doc, obj = _two_slot_box()
    with imgui_context(monkeypatch) as imgui:
        imgui.new_frame()
        imgui.begin("##host")
        clay_props._texture_slots(_Ctx(), _tab(_Ctx(), doc), doc, 0, doc.materials[0])
        clay_outliner._visibility_row(doc)
        imgui.end()
        imgui.end_frame()

    def one(needle: str) -> dict[str, Any]:
        (found,) = [v for k, v in seen.items() if needle in k]
        return found

    clear = one("texclear")
    assert not clear["enabled"] and clear["reason"], "greyed with its reason"
    assert clear.get("tooltip"), "and a glyph-only button names itself"
    assert not one("Solo")["enabled"] and one("Solo")["reason"]
    assert not one("Show all")["enabled"] and one("Show all")["reason"]


# --- clay-77 / clay-78 -------------------------------------------------------


def test_the_export_obj_tooltip_names_the_texture_pngs_it_writes() -> None:
    tip = clay_bridge.EXPORT_OBJ_TOOLTIP
    assert ".mtl" in tip and "PNG" in tip


def test_overlay_tooltips_describe_the_overlay_not_its_history() -> None:
    for key, _label, tooltip in clay_header.OVERLAY_ROWS:
        assert "before the overlay" not in tooltip, key
        assert "was unavailable" not in tooltip, key
