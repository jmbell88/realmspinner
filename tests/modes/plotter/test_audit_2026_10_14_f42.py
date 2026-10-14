"""Regression tests for the 2026-10-14 audit's Plotter pane and mode findings (round R).

One test per finding id, named for the claim. Each was run against the unfixed
tree first and failed there.
"""

from __future__ import annotations

import contextlib
import inspect
from types import SimpleNamespace

import numpy as np
import pytest
from _ui_context import imgui_context

from realmspinner.kernels.grid2d import gid
from realmspinner.kernels.grid2d.tileset import TileFrame, TileMeta, Tileset
from realmspinner.studio.modes.plotter import mode as plotter_mode
from realmspinner.studio.modes.plotter.engine import rmap
from realmspinner.studio.modes.plotter.engine.tilemap import (
    MapDoc,
    MapObject,
    TileShape,
    new_uid,
)
from realmspinner.studio.modes.plotter.ui.panes import canvas as plotter_canvas
from realmspinner.studio.modes.plotter.ui.panes import layers as plotter_layers
from realmspinner.studio.modes.plotter.ui.panes import objects as plotter_objects
from realmspinner.studio.modes.plotter.ui.panes import tileset_editor
from realmspinner.studio.modes.plotter.ui.panes import tools as plotter_tools
from realmspinner.studio.shell import paintview

from ._drive import Scene
from .test_plotter_mode import FakeCtx, _Done, _infinite, _tab, _tileset


@pytest.fixture
def ui(monkeypatch):
    with imgui_context(monkeypatch) as imgui:
        yield imgui


# --- plotter-16: a marquee drag in flight survives a growth ------------------------


def test_a_growth_with_a_marquee_drag_in_flight_shifts_the_remembered_selection_without_raising():
    """``select_before`` is ``(rect, mask)`` and ``capture_select_before`` the same;
    ``shift_cells`` treated the first as a 4-tuple rect and raised."""
    ctx = FakeCtx()
    tab, state = _infinite(ctx)
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True
    state.select_before = ((1, 1, 2, 2), mask)
    state.capture_select_before = ((0, 0, 1, 1), None)

    plotter_canvas._apply(ctx, state, tab, (-2, -1))

    rect, grown = state.select_before
    assert rect == (3, 2, 4, 3)
    assert grown.shape == (tab.doc.height, tab.doc.width) and bool(grown[2, 3])
    assert state.capture_select_before == ((2, 1, 3, 2), None)

    # The other shape the press stores: nothing was selected when it began.
    state.select_before = (None, None)
    state.capture_select_before = (None, None)
    plotter_canvas._apply(ctx, state, tab, (-2, -1))
    assert state.select_before == (None, None)
    assert state.capture_select_before == (None, None)


# --- plotter-30: a tileset landing on a background tab -----------------------------


def test_a_tileset_landing_on_a_background_tab_leaves_the_foreground_brush_alone():
    ctx = FakeCtx()
    background = _tab(ctx, tileset=False)
    foreground = _tab(ctx)
    state = plotter_mode.ensure(ctx)
    assert state.active is foreground
    brush = np.array([[foreground.doc.tilesets[0].firstgid]], gid.DTYPE)
    state.brush = brush
    state.tileset_index = 0
    state.terrain = (0, 1)

    plotter_mode.on_task_done(
        ctx, _Done(f"plotter-tileset:{background.uid}", {"tileset": _tileset("late"), "source": ""})
    )

    assert len(background.doc.tilesets) == 1, "the tileset still lands where it was asked for"
    assert state.brush is brush
    assert state.tileset_index == 0
    assert state.terrain == (0, 1)

    # A ruled sheet for a background tab is parked under a uid the popup never
    # opens for, and ``_switched`` drops it silently: say so instead.
    ctx.toasts.clear()
    sheet = (background.uid, "late", "", np.zeros((16, 16, 4), np.uint8), None)
    plotter_mode.on_task_done(ctx, _Done(f"plotter-tileset:{background.uid}", {"sheet": sheet}))
    assert state.sheet_import is None
    assert ctx.toasts, "the dropped sheet vanished with no toast"


# --- plotter-08: the canvas right-click acts on the object under the pointer -------


def test_right_clicking_an_unselected_object_selects_it_and_the_menu_acts_on_it(monkeypatch):
    scene = Scene(monkeypatch)
    first = scene.add(x=0.0, y=0.0, w=16.0, h=16.0, name="a")
    second = scene.add(x=64.0, y=64.0, w=16.0, h=16.0, name="b")
    scene.state.select_object(first.uid)
    opened: list[str] = []

    from imgui_bundle import imgui

    monkeypatch.setattr(imgui, "open_popup", lambda name: opened.append(name))

    @contextlib.contextmanager
    def _closed(_name):
        yield False

    monkeypatch.setattr(plotter_canvas.controls, "menu_popup", _closed)

    def right_click(at):
        scene.mouse.at = (float(at[0]), float(at[1]))
        scene.mouse.clicked = {0: False, 1: True, 2: False}
        plotter_canvas._object_menu(scene.ctx, scene.state, scene.tab, True, (0.0, 0.0))

    right_click((70.0, 70.0))
    assert scene.state.selected_object == second.uid
    assert opened == [plotter_canvas.OBJECT_MENU]

    # Empty space opens nothing and leaves the selection alone.
    opened.clear()
    right_click((200.0, 5.0))
    assert opened == []
    assert scene.state.selected_object == second.uid

    # With nothing selected at all, an object under the pointer still gets the menu.
    scene.state.select_object(None)
    right_click((4.0, 4.0))
    assert scene.state.selected_object == first.uid
    assert opened == [plotter_canvas.OBJECT_MENU]


# --- plotter-07: the Objects dock's popup ------------------------------------------


def test_right_clicking_an_objects_dock_row_opens_the_popup_that_menu_begins(ui, monkeypatch):
    ctx = SimpleNamespace(
        state=SimpleNamespace(plotter=None, preview={}, list_filters={}),
        settings=SimpleNamespace(get=lambda *_a, **_k: None, set=lambda *_a, **_k: None),
        toast=lambda *_a, **_k: None,
    )
    doc = MapDoc(16, 12, 16, 16)
    doc.add_tile_layer("Ground")
    layer = doc.add_object_layer("Objects")
    doc.add_object(layer.uid, MapObject(uid=new_uid(), name="door", w=16.0, h=16.0))
    tab = plotter_mode.adopt(ctx, doc, title="Level")
    assert tab is not None
    monkeypatch.setattr(plotter_objects.manual_render, "help_button", lambda *_a, **_k: None)
    monkeypatch.setattr(ui, "is_item_hovered", lambda *_a, **_k: True)
    monkeypatch.setattr(ui, "is_mouse_clicked", lambda button, *_a, **_k: button == 1)

    seen: list[bool] = []
    for _frame in range(2):
        ui.new_frame()
        ui.set_next_window_size((400.0, 300.0))
        ui.begin("##host")
        plotter_objects.draw(ctx)
        # Where ``_menu`` calls ``begin_popup``: the window's own id level.
        seen.append(bool(ui.is_popup_open(plotter_objects.MENU)))
        ui.end()
        ui.end_frame()

    assert any(seen), "the popup the dock opened is not the one its menu begins"


# --- plotter-15: the offset-lattice grid walks only what the viewport can see ------


def test_the_offset_lattice_grid_only_walks_cells_the_viewport_can_see(monkeypatch):
    import imgui_bundle

    from .test_canvas_grid import _fake_flag, _FakeDrawList

    monkeypatch.setattr(
        imgui_bundle,
        "imgui",
        SimpleNamespace(
            get_color_u32=lambda _rgba: 0xFFFFFFFF,
            ImDrawFlags_=SimpleNamespace(closed=_fake_flag()),
        ),
    )
    doc = MapDoc(128, 128, 16, 16, projection="hexagonal")
    view = paintview.PaintView()
    region = (200.0, 120.0)
    draw_list = _FakeDrawList()

    plotter_canvas._grid(draw_list, doc, view, (0.0, 0.0), region)

    c0, r0, c1, r1 = plotter_canvas._visible_range(view, doc, (0.0, 0.0), region)
    visible = (c1 - c0 + 1) * (r1 - r0 + 1)
    drawn = len(draw_list.polyline_calls)
    assert drawn >= visible, "a cell the viewport can see was not outlined"
    assert drawn <= visible + 4 * (c1 - c0 + r1 - r0 + 6), f"outlined {drawn} of 16384 cells"


# --- plotter-19 / plotter-tiles-02: animation frames name tiles the set holds ------


def _four_tiles(**extra) -> Tileset:
    return Tileset(name="t", pixels=_tileset().pixels, tile_w=16, tile_h=16, **extra)


def test_a_tileset_refuses_an_animation_frame_that_names_a_tile_it_does_not_have():
    """Dropped rather than refused whole, as the Wang tiles are: the file keeps its
    other frames and the map still opens."""
    meta = TileMeta(animation=(TileFrame(-5, 100), TileFrame(1, 100), TileFrame(99, 100)))
    held = _four_tiles(tiles={0: meta, 50: TileMeta(class_name="ghost")})

    assert [f.local_id for f in held.tiles[0].animation] == [1]
    assert 50 not in held.tiles, "a record for a tile the set lacks survives"

    all_bad = _four_tiles(tiles={0: TileMeta(animation=(TileFrame(-1, 100),))})
    assert 0 not in all_bad.tiles

    # The editor's own door, which does not re-run the constructor.
    edited = _four_tiles().with_meta(0, TileMeta(animation=(TileFrame(-5, 100),)))
    assert 0 not in edited.tiles
    edited.uv(0)  # the Animation preview's call; it raised IndexError for a bad id


def test_substitute_animated_ignores_a_frame_naming_a_tile_the_set_does_not_hold():
    doc = MapDoc(4, 4, 16, 16)
    doc.add_tileset(
        _four_tiles(tiles={0: TileMeta(animation=(TileFrame(-5, 100), TileFrame(0, 100)))})
    )
    for clock in (0, 150):
        subs = plotter_canvas.animated_substitutions(doc, clock)
        block = np.array([[1, 2], [3, 4]], gid.DTYPE)
        out = plotter_canvas.substitute_animated(block, subs)  # OverflowError before
        assert int(out[1, 1]) == 4

    # Defence in depth: a set whose record slipped past the constructor.
    ref = doc.tilesets[0]
    bad = dict(ref.tileset.tiles)
    bad[0] = TileMeta(animation=(TileFrame(-5, 100),))
    object.__setattr__(ref.tileset, "tiles", bad)
    assert plotter_canvas.animated_substitutions(doc, 0) == {}


# --- plotter-31: a group drag left open by a missed release ------------------------


def test_a_group_drag_left_open_by_a_missed_release_is_closed_on_the_next_idle_frame():
    ctx = FakeCtx()
    tab = _tab(ctx)
    layer = tab.doc.add_object_layer("Objects")
    uids = [
        tab.doc.add_object(layer.uid, MapObject(uid=new_uid(), x=float(i), w=4.0, h=4.0)).uid
        for i in range(2)
    ]
    tab.doc.mark_saved()
    tab.doc.begin_group_edit(layer.uid, uids)
    tab.doc.move_group(8.0, 0.0)
    assert tab.doc.editing_group
    state = plotter_mode.ensure(ctx)

    plotter_canvas._close_missed_sessions(state)

    assert not tab.doc.editing_group
    assert tab.doc.dirty, "the moved objects stayed ahead of the history head"
    # And the pane's per-frame sweep is the thing that calls it.
    assert "_close_missed_sessions(" in inspect.getsource(plotter_canvas.draw)


# --- plotter-01: a tile object's gid must resolve ----------------------------------


def _tile_object_map():
    doc = MapDoc(4, 4, 16, 16)
    doc.add_tileset(_tileset())
    layer = doc.add_object_layer("Objects")
    obj = doc.add_object(
        layer.uid, MapObject(uid=new_uid(), name="tile1", shape=TileShape(gid=1, w=16, h=16))
    )
    return doc, layer, obj


def test_a_tile_object_gid_no_tileset_covers_is_never_written_and_a_saved_map_always_reopens(
    ui, monkeypatch
):
    doc, layer, obj = _tile_object_map()

    # The Properties pane's Tile field, stepped past the last tile.
    monkeypatch.setattr(plotter_layers.controls, "input_int", lambda *_a, **_k: (True, 9999))
    monkeypatch.setattr(
        plotter_layers.controls, "checkbox", lambda _label, value, **_k: (False, value)
    )
    ui.new_frame()
    ui.begin("##host")
    ui.begin_table("##fields", 2)
    plotter_layers._shape_fields(doc, doc.layer(layer.uid), doc.layer(layer.uid).objects[0])
    ui.end_table()
    ui.end()
    ui.end_frame()
    assert doc.layer(layer.uid).objects[0].shape.gid == 1

    # Ctrl+V of a copied tile object after its tileset was removed.
    ctx = FakeCtx()
    tab = _tab(ctx)
    state = plotter_mode.ensure(ctx)
    pasted = tab.doc.add_object_layer("Objects")
    tab.doc.set_active_layer(pasted.uid)
    state.clipboard = plotter_mode.ObjectClip(
        obj=MapObject(uid=new_uid(), name="tile1", shape=TileShape(gid=9999, w=16, h=16))
    )
    state.clipboard_doc = tab.uid
    plotter_mode._paste_object(ctx, state, tab)
    assert tab.doc.layer(pasted.uid).objects == []
    assert ctx.toasts, "a refused paste said nothing"

    # And a clip whose tileset an undo took away is pruned like the tile clipboard.
    state.clipboard = plotter_mode.ObjectClip(
        obj=MapObject(uid=new_uid(), shape=TileShape(gid=9999, w=16, h=16))
    )
    state.clipboard_doc = tab.uid
    plotter_mode._prune_stale_brush(ctx, tab)
    assert state.clipboard is None

    # Neither map holds a gid its tilesets lack, so both save and reopen.
    reopened = rmap.read_rmap(rmap.rmap_bytes(doc))
    assert [o.name for layer_ in reopened.all_layers() for o in getattr(layer_, "objects", ())] == [
        "tile1"
    ]
    assert rmap.read_rmap(rmap.rmap_bytes(tab.doc)).width == tab.doc.width


# --- plotter-33: a new property cannot silently replace an existing one ------------


def test_adding_a_property_with_an_existing_name_is_refused_with_a_reason(ui, monkeypatch):
    from realmspinner.studio.modes.plotter.engine.props import Prop

    calls: list[tuple[str, bool, str]] = []

    def spy(label, enabled, *args, **kwargs):
        calls.append((label, bool(enabled), str(kwargs.get("reason", ""))))
        return False

    monkeypatch.setattr(plotter_layers.widgets, "disabled_button", spy)
    props = {"hp": Prop(type="int", value=7)}
    for typed, allowed in (("hp", False), ("mana", True)):
        calls.clear()
        form = {"name": typed, "type": "int", "selected": ""}
        ui.new_frame()
        ui.begin("##host")
        plotter_layers._prop_footer(None, form, props, lambda _values: None)
        ui.end()
        ui.end_frame()
        add = next(call for call in calls if "add-prop" in call[0])
        assert add[1] is allowed, (typed, add)
        if not allowed:
            assert "hp" in add[2] and "exists" in add[2]


# --- plotter-34: deleting inside the only group ------------------------------------


def test_deleting_a_layer_inside_the_only_group_is_not_refused_as_the_last_layer():
    doc = MapDoc(4, 4, 16, 16)
    first = doc.add_tile_layer("One")
    second = doc.add_tile_layer("Two")
    group = doc.add_group_layer("Folder")
    doc.move_layer(first.uid, 0, parent_uid=group.uid)
    doc.move_layer(second.uid, 0, parent_uid=group.uid)
    assert len(doc.layers) == 1, "the group is the only root entry"

    assert plotter_layers.layer_delete_blocker(doc, first.uid) == ""
    assert plotter_layers.layer_delete_blocker(doc, second.uid) == ""
    # The folder itself would take every layer with it: that one is the refusal.
    assert "at least one layer" in plotter_layers.layer_delete_blocker(doc, group.uid)

    lone = MapDoc(4, 4, 16, 16)
    only = lone.add_tile_layer("Only")
    assert "at least one layer" in plotter_layers.layer_delete_blocker(lone, only.uid)


# --- plotter-37: the Wang colour in hand follows the renumbering -------------------


def _wang_tab():
    from realmspinner.kernels.grid2d import wang as wanglib

    ctx = FakeCtx()
    tab = _tab(ctx)
    state = SimpleNamespace(tileset_wang_colour=1)
    wangset = wanglib.WangSet(
        colours=tuple(wanglib.WangColour(name=f"c{n}") for n in range(3)), tiles={}
    )
    tileset_editor._write_wangsets(tab, 0, (wangset,))
    return tab, state


def test_removing_an_earlier_wang_colour_keeps_the_colour_in_hand():
    tab, state = _wang_tab()
    state.tileset_wang_colour = 3  # the third colour, slot 2
    tileset_editor.remove_wang_colour(state, tab, 0, 0, 0)
    assert state.tileset_wang_colour == 2, "the hand now holds a different colour"

    # A later colour removed leaves the hand alone; its own colour resets it.
    tab, state = _wang_tab()
    state.tileset_wang_colour = 1
    tileset_editor.remove_wang_colour(state, tab, 0, 0, 2)
    assert state.tileset_wang_colour == 1
    tab, state = _wang_tab()
    state.tileset_wang_colour = 2
    tileset_editor.remove_wang_colour(state, tab, 0, 0, 1)
    assert state.tileset_wang_colour == 1
    # Unset stays unset.
    tab, state = _wang_tab()
    state.tileset_wang_colour = 0
    tileset_editor.remove_wang_colour(state, tab, 0, 0, 0)
    assert state.tileset_wang_colour == 0


# --- plotter-35: Resize and Autocrop honour the layer lock --------------------------


def _locked_floor():
    doc = MapDoc(4, 4, 16, 16)
    doc.add_tileset(_tileset())
    layer = doc.add_tile_layer("Floor")
    doc.write_region(layer.uid, 1, 1, np.array([[1]], gid.DTYPE))
    doc.set_layer_props(layer.uid, locked=True)
    doc.set_active_layer(layer.uid)
    return doc, layer


@pytest.mark.parametrize(
    ("form", "pressed"),
    [
        ({"w": 4, "h": 4, "dx": 1, "dy": 0}, "Resize##apply"),
        ({"w": 3, "h": 4, "dx": 0, "dy": 0}, "Resize##apply"),
        (None, "Autocrop to content"),
    ],
)
def test_resize_with_an_offset_refuses_while_a_tile_layer_is_locked(ui, monkeypatch, form, pressed):
    doc, layer = _locked_floor()
    before = doc.layer(layer.uid).data.copy()
    shape = (doc.width, doc.height)
    toasts: list[str] = []
    ctx = SimpleNamespace(
        state=SimpleNamespace(preview={}), toast=lambda message, *_a, **_k: toasts.append(message)
    )
    tab = SimpleNamespace(doc=doc, uid=1, view=SimpleNamespace(fitted=True))
    if form is not None:
        ctx.state.preview["plotter_resize:1"] = {"for": shape, **form}
    monkeypatch.setattr(
        plotter_tools.widgets, "primary_button", lambda label, *a, **k: label == pressed
    )
    monkeypatch.setattr(plotter_tools.controls, "button", lambda label, *a, **k: label == pressed)

    ui.new_frame()
    ui.begin("##host")
    plotter_tools._resize_form(ctx, tab)
    ui.end()
    ui.end_frame()

    assert (doc.width, doc.height) == shape, "a locked layer was resized away"
    assert np.array_equal(doc.layer(layer.uid).data, before)
    assert any("locked" in message for message in toasts)

    # A plain growth moves nothing, so the lock does not stand in its way.
    ctx.state.preview["plotter_resize:1"] = {"for": shape, "w": 6, "h": 6, "dx": 0, "dy": 0}
    monkeypatch.setattr(
        plotter_tools.widgets, "primary_button", lambda label, *a, **k: label == "Resize##apply"
    )
    monkeypatch.setattr(plotter_tools.controls, "button", lambda *a, **k: False)
    ui.new_frame()
    ui.begin("##host")
    plotter_tools._resize_form(ctx, tab)
    ui.end()
    ui.end_frame()
    assert (doc.width, doc.height) == (6, 6)

