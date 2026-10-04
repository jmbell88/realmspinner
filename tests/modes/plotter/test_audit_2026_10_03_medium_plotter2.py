"""Closing plotter-13..18, -21 and -22 from the 2026-10-03 audit of Plotter.

One test per finding, named for the claim it proves.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from _ui_context import imgui_context

from realmspinner.kernels.grid2d import gid as gidlib
from realmspinner.kernels.grid2d.tileset import Tileset
from realmspinner.studio.modes.plotter import mode as plotter_mode
from realmspinner.studio.modes.plotter import setup as plotter_setup
from realmspinner.studio.modes.plotter.engine import project
from realmspinner.studio.modes.plotter.engine.tilemap import MapDoc
from realmspinner.studio.modes.plotter.ui.panes import canvas as plotter_canvas
from realmspinner.studio.modes.plotter.ui.panes import layers as plotter_layers
from realmspinner.studio.modes.plotter.ui.panes import menu as plotter_menu
from realmspinner.studio.modes.plotter.ui.panes import objects as plotter_objects
from realmspinner.studio.modes.plotter.ui.panes import tileset_editor as editor
from realmspinner.studio.modes.plotter.ui.panes import tools as plotter_tools


@pytest.fixture
def ui(monkeypatch):
    with imgui_context(monkeypatch) as imgui:
        yield imgui


def _ctx():
    toasts: list[tuple] = []
    return SimpleNamespace(
        state=SimpleNamespace(plotter=None, mode="plotter", preview={}, list_filters={}),
        settings=SimpleNamespace(get=lambda _k: None, set=lambda _k, _v: None),
        toast=lambda *a, **_k: toasts.append(a),
        toast_once=lambda *a, **_k: toasts.append(a),
        viewer=None,
        toasts=toasts,
    )


def _tileset(tiles: int = 4) -> Tileset:
    pixels = np.zeros((16, 16 * tiles, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    return Tileset(name="t", pixels=pixels, tile_w=16, tile_h=16)


def _infinite(ctx):
    tab = plotter_mode.new_document(ctx, (4, 4, 16, 16), infinite=True)
    tab.doc.add_tileset(_tileset())
    tab.doc.mark_saved()
    return tab, plotter_mode.ensure(ctx)


# --- plotter-13 -----------------------------------------------------------------


@pytest.mark.parametrize("tool", ["pick", "erase"])
def test_picking_or_erasing_in_the_void_of_an_infinite_map_does_not_grow_it(tool):
    ctx = _ctx()
    tab, state = _infinite(ctx)
    state.tool = tool
    head = tab.doc.history.head

    for cell in ((-3, -2), (9, 9)):
        plotter_canvas._apply(ctx, state, tab, cell)

    assert (tab.doc.origin_x, tab.doc.origin_y) == (0, 0)
    assert (tab.doc.width, tab.doc.height) == (4, 4)
    assert tab.doc.history.head == head, "a probe pushed an undo step"
    assert not tab.doc.dirty


@pytest.mark.parametrize("tool", ["stamp", "fill"])
def test_a_stamp_or_fill_with_nothing_in_the_hand_refuses_before_the_map_grows(tool):
    ctx = _ctx()
    tab, state = _infinite(ctx)
    state.tool = tool
    state.brush = None
    state.terrain = None
    head = tab.doc.history.head

    plotter_canvas._apply(ctx, state, tab, (-3, -2))

    assert (tab.doc.width, tab.doc.height) == (4, 4)
    assert tab.doc.history.head == head
    assert [t[0] for t in ctx.toasts] == ["Pick a tile from the tileset first."]


# --- plotter-14 -----------------------------------------------------------------


@pytest.mark.parametrize("tab_name", ["Collision", "Animation", "Terrain"])
def test_opening_the_sheet_on_a_smaller_tileset_resets_the_selected_tile_for_every_tab(
    ui, monkeypatch, tab_name
):
    ctx = _ctx()
    tab, state = _infinite(ctx)  # four tiles
    state.editing_tileset = 0
    state.editing_tile = 40
    state.tileset_tab = tab_name
    seen: list[int] = []
    for name in ("_tiles_tab", "_collision_tab", "_terrain_tab", "_animation_tab"):
        monkeypatch.setattr(
            editor, name, lambda *_a, **_k: seen.append(int(state.editing_tile))
        )
    monkeypatch.setattr(editor.manual_render, "help_button", lambda *_a, **_k: None)

    ui.new_frame()
    ui.begin("##host")
    editor.draw(ctx)
    ui.end()
    ui.end_frame()

    assert seen == [0], f"the {tab_name} tab was handed tile {seen} on a 4-tile set"
    assert tab.doc.tilesets[0].tileset.meta_of(0) is not None


# --- plotter-15 -----------------------------------------------------------------


def _record_rows(monkeypatch):
    rows: list[tuple[str, bool, str]] = []

    def row(label, key="", *, enabled=True, reason=""):
        rows.append((label, enabled, reason))
        return False

    monkeypatch.setattr(plotter_menu, "_row", row)
    monkeypatch.setattr(plotter_menu.controls, "menu_separator", lambda: None)
    return rows


def test_map_and_tileset_menu_rows_say_nothing_is_open_when_no_map_is_open(monkeypatch):
    rows = _record_rows(monkeypatch)
    ctx = SimpleNamespace()
    state = SimpleNamespace(resize_pending=False)

    plotter_menu._map_rows(ctx, state, None)
    plotter_menu._tileset_rows(ctx, state, None)

    greyed = [(label, reason) for label, enabled, reason in rows if not enabled]
    assert len(greyed) >= 9
    for label, reason in greyed:
        assert reason != plotter_menu.BUSY, f"{label} claims a write is under way"
        assert "open" in reason.lower() or "first" in reason.lower(), (label, reason)


def test_a_busy_map_still_says_it_is_being_written(monkeypatch):
    rows = _record_rows(monkeypatch)
    doc = MapDoc(4, 4, 16, 16)
    doc.add_tileset(_tileset())
    tab = SimpleNamespace(doc=doc, busy=True)

    plotter_menu._map_rows(SimpleNamespace(), SimpleNamespace(), tab)

    save = next(r for r in rows if r[0] == "Save")
    assert save[1] is False and save[2] == plotter_menu.BUSY


# --- plotter-16 -----------------------------------------------------------------


def test_x_chooses_insert_text_on_an_object_layer_even_with_a_brush_in_hand():
    import pygame

    ctx = _ctx()
    tab = plotter_mode.new_document(ctx, (4, 4, 16, 16))
    state = plotter_mode.ensure(ctx)
    objects = tab.doc.add_object_layer()
    tab.doc.set_active_layer(objects.uid)
    brush = np.array([[1, 2]], gidlib.DTYPE)
    state.brush = brush.copy()
    state.tool = "object"

    event = SimpleNamespace(type=pygame.KEYDOWN, key=pygame.K_x, mod=0)
    assert plotter_mode.handle_key(ctx, event) is True

    assert state.tool == "object_text"
    assert np.array_equal(state.brush, brush), "X flipped a brush nobody can see"


def test_x_still_flips_the_brush_on_a_tile_layer():
    import pygame

    ctx = _ctx()
    plotter_mode.new_document(ctx, (4, 4, 16, 16))
    state = plotter_mode.ensure(ctx)
    state.brush = np.array([[1, 2]], gidlib.DTYPE)

    plotter_mode.handle_key(ctx, SimpleNamespace(type=pygame.KEYDOWN, key=pygame.K_x, mod=0))

    # Flipping reverses the columns and sets the gid's flip bit.
    assert (state.brush & 0x0FFFFFFF).tolist() == [[2, 1]]


# --- plotter-17 -----------------------------------------------------------------


def test_choosing_hexagonal_in_the_combo_gets_a_nonzero_hex_side():
    form = plotter_setup.blank_form()
    form["projection"] = project.HEXAGONAL  # what the Projection combo writes

    plotter_setup.clamp(form)

    assert form["hex_side"] > 0

    ctx = _ctx()
    form["next"] = plotter_setup.NEXT_EMPTY
    plotter_canvas._create(ctx, form)
    assert plotter_mode.ensure(ctx).active.doc.hex_side > 0


def test_leaving_the_hexagonal_preset_for_another_projection_drops_its_hex_side():
    form = plotter_setup.blank_form()
    plotter_setup.apply_preset(form, "Hexagonal, 32 px")
    form["projection"] = project.ORTHOGONAL

    plotter_setup.clamp(form)

    assert form["hex_side"] == 0


# --- plotter-18 -----------------------------------------------------------------


@pytest.mark.parametrize("tool", ["object_rect", "object_point", "object_text"])
def test_delete_removes_the_selected_object_with_an_insert_tool_in_hand(tool):
    ctx = _ctx()
    tab = plotter_mode.new_document(ctx, (4, 4, 16, 16))
    state = plotter_mode.ensure(ctx)
    layer = tab.doc.add_object_layer()
    obj = plotter_layers.add_object(tab.doc, tab.doc.layer(layer.uid), "rect", 0, 0, 16, 16)
    tab.doc.set_active_layer(layer.uid)
    state.select_object(obj.uid)
    state.tool = tool

    plotter_mode._delete(ctx, state, tab)

    assert tab.doc.layer(layer.uid).objects == []
    assert not any("Select some cells" in t[0] for t in ctx.toasts)


def test_duplicate_acts_on_the_selected_object_with_an_insert_tool_in_hand():
    ctx = _ctx()
    tab = plotter_mode.new_document(ctx, (4, 4, 16, 16))
    state = plotter_mode.ensure(ctx)
    layer = tab.doc.add_object_layer()
    obj = plotter_layers.add_object(tab.doc, tab.doc.layer(layer.uid), "rect", 0, 0, 16, 16)
    tab.doc.set_active_layer(layer.uid)
    state.select_object(obj.uid)
    state.tool = "object_rect"

    plotter_mode._duplicate_object(ctx, state, tab)

    assert len(tab.doc.layer(layer.uid).objects) == 2


# --- plotter-21 -----------------------------------------------------------------


def test_the_objects_dock_submits_only_the_rows_the_pane_can_show(ui, monkeypatch):
    ctx = _ctx()
    tab = plotter_mode.new_document(ctx, (4, 4, 16, 16))
    layer = tab.doc.add_object_layer()
    live = tab.doc.layer(layer.uid)
    for i in range(3000):
        plotter_layers.add_object(tab.doc, live, "point", float(i), 0.0, 0.0, 0.0)
    calls: list[int] = []

    def row(*_a, **_k):
        # A real item of the real row height, so the clipper's cursor jumps
        # have something to extend the window from.
        calls.append(1)
        ui.dummy((10.0, ui.get_frame_height()))

    monkeypatch.setattr(plotter_objects, "_row", row)
    monkeypatch.setattr(plotter_objects.manual_render, "help_button", lambda *_a, **_k: None)

    # Two frames: a window's first frame has no size to clip against.
    for _frame in range(2):
        calls.clear()
        ui.new_frame()
        ui.set_next_window_size((400.0, 300.0))
        ui.begin("##host")
        ui.begin_child("##scroll", (380.0, 260.0))
        plotter_objects.draw(ctx)
        # What follows the dock in a real pane; list_row ends on a bare cursor move.
        ui.dummy((1.0, 1.0))
        ui.end_child()
        ui.end()
        ui.end_frame()

    assert 0 < len(calls) < 200, f"drew {len(calls)} rows of 3000 objects"


def test_the_object_property_combo_options_are_not_rebuilt_while_the_document_is_unchanged():
    ctx = _ctx()
    tab = plotter_mode.new_document(ctx, (4, 4, 16, 16))
    layer = tab.doc.add_object_layer()
    plotter_layers.add_object(tab.doc, tab.doc.layer(layer.uid), "point", 0, 0, 0, 0)

    first = plotter_layers.object_options(tab.doc)
    assert plotter_layers.object_options(tab.doc) is first

    plotter_layers.add_object(tab.doc, tab.doc.layer(layer.uid), "point", 1, 0, 0, 0)
    assert len(plotter_layers.object_options(tab.doc)) == len(first) + 1


# --- plotter-22 -----------------------------------------------------------------


def test_merge_down_onto_a_locked_layer_is_refused():
    ctx = _ctx()
    doc = MapDoc(4, 4, 16, 16)
    doc.add_tileset(_tileset())
    floor = doc.add_tile_layer("Floor")
    top = doc.add_tile_layer("Top")
    doc.write_region(top.uid, 0, 0, np.array([[1]], gidlib.DTYPE))
    doc.set_layer_props(floor.uid, locked=True)
    before = doc.layer(floor.uid).data.copy()

    plotter_layers._merge_down(ctx, doc, doc.layer(top.uid))

    assert doc.layer(top.uid) is not None, "the layer was merged away"
    assert np.array_equal(doc.layer(floor.uid).data, before)
    assert any("locked" in t[0] for t in ctx.toasts)


@pytest.mark.parametrize("scope", ["layer", "map"])
def test_offset_of_a_locked_layer_is_refused(ui, monkeypatch, scope):
    ctx = _ctx()
    doc = MapDoc(4, 4, 16, 16)
    doc.add_tileset(_tileset())
    layer = doc.add_tile_layer("Floor")
    doc.write_region(layer.uid, 0, 0, np.array([[1]], gidlib.DTYPE))
    doc.set_layer_props(layer.uid, locked=True)
    doc.set_active_layer(layer.uid)
    tab = SimpleNamespace(doc=doc, uid=1)
    ctx.state.preview["plotter_offset:1"] = {"dx": 1, "dy": 0, "wrap": True, "scope": scope}
    before = doc.layer(layer.uid).data.copy()
    monkeypatch.setattr(
        plotter_tools.controls, "button", lambda label, *a, **k: label == "Offset##apply"
    )

    ui.new_frame()
    ui.begin("##host")
    plotter_tools._offset_form(ctx, tab)
    ui.end()
    ui.end_frame()

    assert np.array_equal(doc.layer(layer.uid).data, before), "a locked layer was moved"
    assert any("locked" in t[0] for t in ctx.toasts)
