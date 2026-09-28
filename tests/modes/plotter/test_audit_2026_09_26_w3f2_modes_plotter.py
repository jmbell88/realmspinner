"""Closing findings plotter-mode-04..-18 from the 2026-09-26 audit of Plotter.

One test per finding, named for the claim it proves. Every one of these failed
against the unfixed code (either run directly against the old source, or run
before its matching fix landed in this same change) before its fix was made;
see the fixer's own return for the pasted failing output.

Excluded here: plotter-mode-01..-03 (closed by an earlier wave), plotter-mode-07,
-08 and -20 (manual-only, no code to test), and plotter-mode-19 (needs
``engine/render.py``, owned by another fixer).
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

import numpy as np

from realmspinner.kernels.grid2d import gid as gidlib
from realmspinner.kernels.grid2d.tileset import Tileset
from realmspinner.studio.modes.plotter import mode as plotter_mode
from realmspinner.studio.modes.plotter import state as plotter_state
from realmspinner.studio.modes.plotter.engine.tilemap import MapDoc


def _tileset(name: str = "t", *, tile: int = 16, tiles: int = 2) -> Tileset:
    pixels = np.zeros((tile, tile * tiles, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    return Tileset(name=name, pixels=pixels, tile_w=tile, tile_h=tile)


# --- plotter-mode-04: one toast per gesture, not per cell --------------------


class _ToastCtx:
    """A minimal stand-in for ``state.toast``/``state.toast_once``'s real
    dedupe (text + level, over the live toasts) -- exact enough for one
    gesture inside one test, where nothing expires."""

    def __init__(self) -> None:
        self.toasts: list[tuple[str, str]] = []

    def toast(self, text: str, level: str = "info", *_a: Any, **_k: Any) -> None:
        self.toasts.append((text, level))

    def toast_once(self, text: str, level: str = "info", *_a: Any, **_k: Any) -> bool:
        if any(t == text and lvl == level for t, lvl in self.toasts):
            return False
        self.toast(text, level)
        return True


def test_a_drag_on_a_locked_layer_raises_one_toast() -> None:
    from realmspinner.studio.modes.plotter.ui.panes import canvas as plotter_canvas

    doc = MapDoc(4, 4, 8, 8)
    doc.add_tile_layer("Ground")
    layer = doc.layers[0]
    layer.locked = True
    doc.set_active_layer(layer.uid)
    tab = SimpleNamespace(doc=doc)
    ctx = _ToastCtx()

    for x in range(5):
        plotter_canvas._apply(ctx, None, tab, (x, 0))

    assert len(ctx.toasts) == 1, (
        f"one refusal per gesture, not per cell entered: got {ctx.toasts}"
    )
    assert ctx.toasts[0][1] == "error"


# --- plotter-mode-05: the tileset sheet's view state does not survive a switch


def test_switching_tabs_closes_the_tileset_sheet_and_resets_the_selected_tile() -> None:
    state = plotter_state.PlotterState()
    doc1 = MapDoc(4, 4, 8, 8)
    doc1.add_tileset(_tileset(tiles=8))
    tab1 = plotter_state.PlotterDoc(doc=doc1, title="one")
    state.add(tab1)
    doc2 = MapDoc(4, 4, 8, 8)
    tab2 = plotter_state.PlotterDoc(doc=doc2, title="two")
    state.add(tab2)

    state.activate(tab1.uid)
    state.editing_tileset = 0
    state.editing_tile = 5
    state.tileset_shape = 2

    state.activate(tab2.uid)

    assert state.editing_tileset is None, "the sheet must close on switch"
    assert state.editing_tile == 0, "a stale local id must not survive the switch"
    assert state.tileset_shape is None


# --- plotter-mode-06: renaming a stamp by typing is one undo step -----------


def test_renaming_a_stamp_by_typing_is_one_undo_step() -> None:
    """The manual promises ``storing, renaming and clearing a slot are each
    one undo step`` (32:577-579); ``rename_stamp`` pushes an unconditional
    ``history.push``, so the field must fold like every other one-gesture
    text field in this pane rather than call it straight off ``input_text``.
    """
    from realmspinner.studio.modes.plotter.ui.panes import stamps as plotter_stamps

    source = inspect.getsource(plotter_stamps._slot_row)
    after_field = source.split('"##name"', 1)[1]
    before_write = after_field.split("tab.doc.rename_stamp(", 1)[0]
    assert "controls.fold_undo(" in before_write, (
        "the name field is typed with no fold before rename_stamp is called"
    )


# --- plotter-mode-09: a reparent past MAX_GROUP_DEPTH is refused, not raised


def test_moving_a_layer_into_a_maximally_deep_group_is_refused_by_name() -> None:
    from realmspinner.studio.modes.plotter.ui.panes import layers as plotter_layers

    doc = MapDoc(4, 4, 16, 16)
    doc.add_tile_layer("Ground")
    layer = doc.layers[0]
    parent = None
    deepest = None
    for _ in range(65):  # depths 0..64: the deepest group MAX_GROUP_DEPTH allows
        deepest = doc.add_group_layer(parent_uid=parent)
        parent = deepest.uid

    toasts: list[tuple[str, str]] = []
    ctx = SimpleNamespace(toast=lambda text, kind="info": toasts.append((text, kind)))

    # Reparenting the plain tile layer one level past the deepest group nests
    # it 65 deep, past MAX_GROUP_DEPTH (64) -- ``move_layer`` raises
    # ``ValueError``, and nothing wrapped this door before the fix.
    plotter_layers._move_layer(ctx, doc, layer.uid, 0, parent_uid=deepest.uid)

    assert doc.layers[0] is layer, "refused, not moved -- it must stay at the root"
    assert toasts and toasts[-1][1] == "error"
    assert "deep" in toasts[-1][0]


# --- plotter-mode-10: the Layer menu's Raise/Lower follow can_shift_layer ---


def test_layer_menu_raise_and_lower_follow_can_shift_layer(monkeypatch) -> None:
    from realmspinner.studio import controls
    from realmspinner.studio.modes.plotter.ui.panes import menu as plotter_menu

    doc = MapDoc(4, 4, 16, 16)
    group = doc.add_group_layer("G")
    lone = doc.add_tile_layer("Lone", parent_uid=group.uid)
    doc.add_tile_layer("Other")  # a second *root* layer: len(doc.layers) > 1
    doc.set_active_layer(lone.uid)
    tab = SimpleNamespace(doc=doc, busy=False)

    seen: dict[str, tuple[bool, str]] = {}

    def fake_menu_item(label, shortcut="", selected=False, enabled=True, **kwargs):
        for name in ("Raise layer", "Lower layer"):
            if label.startswith(f"{name}##"):
                seen[name] = (bool(enabled), str(kwargs.get("reason", "")))
        return (False, selected)

    monkeypatch.setattr(controls, "menu_item", fake_menu_item)
    monkeypatch.setattr(controls, "menu_separator", lambda: None)

    plotter_menu._layer_rows(None, None, tab)

    # ``lone`` is the *only* child of ``group``: the document has more than one
    # layer overall (``len(doc.layers) == 2``), but this layer has no sibling to
    # trade places with in either direction.
    assert seen["Raise layer"][0] is False, seen["Raise layer"]
    assert seen["Lower layer"][0] is False, seen["Lower layer"]
    assert "end of its group" in seen["Raise layer"][1]
    assert "end of its group" in seen["Lower layer"][1]


# --- plotter-mode-11: a right-click capture must not destroy the marquee ---


def test_a_right_click_capture_leaves_the_selection_alone(monkeypatch) -> None:
    from realmspinner.studio.modes.plotter.ui.panes import canvas as plotter_canvas

    from ._drive import Mouse

    doc = MapDoc(4, 4, 8, 8)
    doc.add_tile_layer("Ground")
    layer = doc.layers[0]
    doc.write_region(layer.uid, 0, 0, np.full((2, 2), 3, dtype=gidlib.DTYPE))
    doc.set_active_layer(layer.uid)
    tab = SimpleNamespace(doc=doc)
    state = plotter_state.PlotterState()
    # A marquee a painting tool is honouring, made before the capture starts.
    state.set_selection((0, 0, 1, 1))
    ctx = SimpleNamespace(toast=lambda *a, **k: None)

    mouse = Mouse()
    mouse.install(monkeypatch)

    mouse.clicked = {0: False, 1: True, 2: False}
    plotter_canvas._capture_input(ctx, state, tab, (2, 2), True)

    mouse.clicked = {0: False, 1: False, 2: False}
    mouse.down = {0: False, 1: True, 2: False}
    plotter_canvas._capture_input(ctx, state, tab, (3, 3), True)

    mouse.down = {0: False, 1: False, 2: False}
    mouse.released = {0: False, 1: True, 2: False}
    plotter_canvas._capture_input(ctx, state, tab, (3, 3), True)

    assert state.select == (0, 0, 1, 1), (
        "the marquee from before the capture must survive it, not end as None"
    )


# --- plotter-mode-12: the terrain field re-arms state, not just the combo --


def test_the_terrain_field_rearms_a_stale_selection(monkeypatch) -> None:
    from realmspinner.studio import controls
    from realmspinner.studio.modes.plotter.ui.panes import tools as plotter_tools

    from ._terrainset import terrain_tileset

    doc = MapDoc(4, 4, 8, 8)
    doc.add_tileset(terrain_tileset(count=3))
    tab = SimpleNamespace(doc=doc)
    state = plotter_state.PlotterState()
    state.terrain = (0, 99)  # a rank this tileset does not carry

    monkeypatch.setattr(
        controls, "combo", lambda control_id, current, options, **k: (False, current)
    )

    field = plotter_tools._terrain_field(state, tab)
    field.draw(False)

    assert state.terrain == (0, 0), (
        "a stale state.terrain must be re-armed, not just re-drawn as options[0]"
    )


# --- plotter-mode-13: the self-referential frame button is gone -------------


def test_add_frame_from_selection_is_removed_not_self_referential() -> None:
    """The Tile form always shows the *picked* tile's own metadata
    (``_picked_local``), so "Add frame from selection" could only ever append
    that tile to its own animation -- there was never a second tile to pick.
    Removed rather than patched to take one, which the tileset editor's own
    Animation tab already does (its docstring names this same incident).
    """
    from realmspinner.studio.modes.plotter.ui.panes import tileset as plotter_tileset

    source = inspect.getsource(plotter_tileset._tile_form)
    assert 'controls.button("Add frame from selection"' not in source
    assert not hasattr(plotter_tileset, "_frame"), "the dead helper should go with it"


# --- plotter-mode-14: the tileset sheet's keys must not reach the hidden map


def test_the_tileset_sheet_being_open_blocks_delete_and_other_map_keys(
    plotter_ctx, monkeypatch
) -> None:
    import pygame

    ctx, state = plotter_ctx
    tab = state.active
    tab.doc.add_tileset(_tileset())
    layer = tab.doc.tile_layers()[0]
    tab.doc.write_region(layer.uid, 0, 0, np.array([[1]], gidlib.DTYPE))
    state.set_selection((0, 0, 0, 0))
    state.editing_tileset = 0  # the sheet covers the canvas now

    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0)
    event = SimpleNamespace(type=pygame.KEYDOWN, key=pygame.K_DELETE, mod=0)
    consumed = plotter_mode.handle_key(ctx, event)

    assert consumed is False
    assert int(tab.doc.layer(layer.uid).data[0, 0]) == 1, (
        "Delete must not touch the map hidden behind the tileset sheet"
    )


# --- plotter-mode-15: a tab's own preview drafts, and only its own ----------


def test_closing_a_tab_drops_only_its_own_preview_drafts() -> None:
    """Four holes in one cleanup, all from this finding: a bare
    ``str.startswith`` matched ``pl10`` while closing ``pl1``, and the
    goto-coordinate draft, the tileset tab-strip memo and the map-properties
    draft were never matched by any prefix at all.
    """
    doc = MapDoc(2, 2, 8, 8)
    doc.add_tile_layer("Ground")
    layer_uid = doc.layers[0].uid
    tab = SimpleNamespace(uid="pl1", doc=doc)
    ctx = SimpleNamespace(
        state=SimpleNamespace(
            preview={
                "plotter_minimap:pl1": 1,
                "plotter_minimap:pl10": 1,
                "plotter_goto:pl1": {},
                "plotter_tabsel:pl1": 0,
                "plotter_map_prop:pl1": {},
                f"plotter_rename:{layer_uid}": "typed",
            }
        )
    )

    plotter_mode._forget_preview(ctx, tab)

    assert set(ctx.state.preview) == {"plotter_minimap:pl10"}, ctx.state.preview


def test_the_map_properties_draft_is_keyed_by_tab_uid_not_id_of_doc() -> None:
    from realmspinner.studio.modes.plotter.ui.panes import layers as plotter_layers

    source = inspect.getsource(plotter_layers.map_rows)
    assert "{id(doc)}" not in source
    assert 'f"plotter_map_prop:{tab.uid}"' in source


# --- plotter-mode-16: Save As/Export must not eat a dotted stem ------------


def test_with_required_suffix_does_not_treat_a_dotted_stem_as_an_extension() -> None:
    from pathlib import Path

    from realmspinner.studio.modes.plotter import fileio as plotter_io

    path = Path("level 1.5")
    result = plotter_io._with_required_suffix(path, ".rmap")

    assert result.name == "level 1.5.rmap", (
        f"the '.5' is part of the title, not a stale extension: got {result.name!r}"
    )


def test_with_required_suffix_does_nothing_when_already_correct() -> None:
    from pathlib import Path

    from realmspinner.studio.modes.plotter import fileio as plotter_io

    path = Path("level.rmap")
    assert plotter_io._with_required_suffix(path, ".rmap") == path


# --- plotter-mode-17: on_task_failed clears saving only for a save task ----


def test_on_task_failed_only_clears_saving_for_a_save_task() -> None:
    state = plotter_state.PlotterState()
    doc = MapDoc(2, 2, 8, 8)
    tab = plotter_state.PlotterDoc(doc=doc, title="m")
    state.add(tab)
    ctx = SimpleNamespace(state=SimpleNamespace(plotter=state))
    tab.saving = True

    done = SimpleNamespace(key=f"plotter-tileset-image:{tab.uid}")
    plotter_mode.on_task_failed(ctx, done)
    assert tab.saving is True, (
        "a tileset-image task never set saving and must not clear it out from "
        "under a real save still in flight"
    )

    done = SimpleNamespace(key=f"plotter-save:{tab.uid}")
    plotter_mode.on_task_failed(ctx, done)
    assert tab.saving is False


# --- plotter-mode-18: the reload door resolves by firstgid, not a stale index


def test_the_reload_door_refuses_when_its_tileset_is_gone_even_if_a_firstgid_matches() -> None:
    """``ask_replace_tileset`` used to capture the tileset's own *index*; a
    remove-and-reinsert while the picker was up could put a different tileset
    at that index by the time the reload landed, repainting the wrong art. A
    firstgid alone is not a safe replacement either -- ``next_firstgid``
    restarts at 1 once the tileset list empties, so removing the map's only
    tileset and adding a fresh one hands it the very firstgid the reload was
    picked for. The door holds the ``TilesetRef`` object itself instead, so
    identity is the test, not a number that can recur.
    """
    from realmspinner.studio.modes.plotter import tilesets as plotter_tilesets

    doc = MapDoc(4, 4, 16, 16)
    doc.add_tile_layer("Tiles")
    original = doc.add_tileset(_tileset())
    doc.remove_tileset(0)
    replacement = doc.add_tileset(_tileset(name="different"))
    # The exact collision this fix has to survive: the empty-list reset means
    # the replacement's firstgid can coincide with the removed original's.
    assert replacement.firstgid == original.firstgid

    tab = SimpleNamespace(doc=doc)
    toasts: list[tuple[str, str]] = []
    ctx = SimpleNamespace(toast=lambda text, kind="info": toasts.append((text, kind)))
    fresh = np.zeros((32, 32, 4), dtype=np.uint8)
    fresh[..., 3] = 255

    plotter_tilesets.land_tileset_image(
        ctx, tab, {"replace": (original, "atlas.png", fresh)}
    )

    assert doc.tilesets[0].tileset is replacement.tileset, (
        "the removed tileset's own ref must not repaint whatever replaced it"
    )
    assert toasts and toasts[-1][1] == "error"
