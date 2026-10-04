"""Two Plotter items left over from the 2026-10-03 audit.

plotter-20, the texture half: a tile-metadata drag re-uploaded the tileset's GPU
texture every frame, because ``tileset_epoch`` -- which the metadata hook moves,
correctly, for the palette and every other reader -- was also the texture's
change stamp. ``tileset_pixel_epoch`` is the stamp that only a pixel or
tile-list change moves.

plotter-10, defence in depth: the readers cap layer-group nesting, but
``Duplicate layer`` had no frame around the engine's ``ValueError``.
"""

from __future__ import annotations

import inspect
import re
from types import SimpleNamespace

import numpy as np
import pytest

from realmspinner.kernels.grid2d.tileset import TileMeta, TileRect, Tileset
from realmspinner.studio.modes.plotter.engine import _map_layers
from realmspinner.studio.modes.plotter.engine.tilemap import MapDoc
from realmspinner.studio.modes.plotter.ui.panes import canvas as plotter_canvas
from realmspinner.studio.modes.plotter.ui.panes import layers as plotter_layers
from realmspinner.studio.modes.plotter.ui.panes import menu as plotter_menu
from realmspinner.studio.modes.plotter.ui.panes import stamps as plotter_stamps
from realmspinner.studio.modes.plotter.ui.panes import textures as plotter_textures
from realmspinner.studio.modes.plotter.ui.panes import tileset as plotter_tileset
from realmspinner.studio.modes.plotter.ui.panes import (
    tileset_editor as plotter_tileset_editor,
)


def _tileset(name: str = "t", *, tiles: int = 4, shade: int = 0) -> Tileset:
    pixels = np.full((16, 16 * tiles, 4), shade, dtype=np.uint8)
    pixels[..., 3] = 255
    return Tileset(name=name, pixels=pixels, tile_w=16, tile_h=16)


def _doc() -> MapDoc:
    doc = MapDoc(4, 4, 16, 16)
    doc.add_tileset(_tileset())
    doc.add_tile_layer("Ground")
    return doc


# --- plotter-20: the texture stamp ---------------------------------------------


def test_a_metadata_edit_does_not_move_the_texture_stamp() -> None:
    doc = _doc()
    pixel_before = doc.tileset_pixel_epoch
    epoch_before = doc.tileset_epoch

    doc.set_tile_meta(0, 1, TileMeta(class_name="Water"))
    assert doc.tileset_epoch > epoch_before, "other readers still need this one"
    assert doc.tileset_pixel_epoch == pixel_before

    # The drag path is the one that ran sixty times a second.
    doc.begin_tile_meta_edit(0, 2)
    for width in range(1, 6):
        assert doc.live_tile_meta(TileMeta(collision=(TileRect(0, 0, width, 4),)))
    doc.end_tile_meta_edit()
    assert doc.tileset_pixel_epoch == pixel_before

    # And so is undoing and redoing the metadata: the pixels never moved.
    doc.undo()
    doc.undo()
    doc.redo()
    assert doc.tileset_pixel_epoch == pixel_before


def test_every_way_the_pixels_or_the_tile_list_change_moves_the_texture_stamp() -> None:
    doc = _doc()
    seen = doc.tileset_pixel_epoch

    def moved(step: str) -> None:
        nonlocal seen
        assert doc.tileset_pixel_epoch > seen, f"{step} left the texture stamp alone"
        seen = doc.tileset_pixel_epoch

    doc.replace_tileset(0, _tileset("repainted", shade=90))
    moved("painting into the tileset")
    doc.undo()
    moved("undoing the repaint")
    doc.redo()
    moved("redoing the repaint")

    ref = doc.add_tileset(_tileset("second", shade=10))
    moved("adding a tileset")
    doc.undo()
    moved("undoing the add")
    doc.redo()
    moved("redoing the add")

    doc.remove_tileset(doc.tilesets.index(ref))
    moved("removing a tileset")
    doc.undo()
    moved("undoing the removal")


def test_a_tileset_epoch_that_tracks_metadata_still_moves_for_every_pixel_change() -> None:
    """``tileset_epoch`` is a superset of the pixel stamp: nothing that moved it
    before may stop moving it."""
    doc = _doc()
    for act in (
        lambda: doc.replace_tileset(0, _tileset("again", shade=3)),
        lambda: doc.add_tileset(_tileset("more")),
        lambda: doc.undo(),
    ):
        both = (doc.tileset_epoch, doc.tileset_pixel_epoch)
        act()
        assert doc.tileset_epoch > both[0] and doc.tileset_pixel_epoch > both[1]


class _Texture:
    def __init__(self) -> None:
        self.filter = None
        self.released = False

    def release(self) -> None:
        self.released = True


class _GL:
    NEAREST = 0x2600

    def __init__(self) -> None:
        self.uploads = 0

    def texture(self, size, components, data) -> _Texture:
        self.uploads += 1
        return _Texture()


def _ctx() -> tuple[SimpleNamespace, _GL]:
    gl = _GL()
    return SimpleNamespace(viewer=SimpleNamespace(ctx=gl), state=SimpleNamespace(preview={})), gl


def test_a_collision_drag_uploads_the_tileset_texture_once_and_a_repaint_uploads_again() -> None:
    ctx, gl = _ctx()
    doc = _doc()

    def draw() -> None:
        plotter_textures.tileset_texture(
            ctx, "tab", 0, doc.tilesets[0].tileset, doc.tileset_pixel_epoch
        )

    draw()
    doc.begin_tile_meta_edit(0, 1)
    for width in range(1, 30):  # a drag: one write per frame, one draw per write
        doc.live_tile_meta(TileMeta(collision=(TileRect(0, 0, width, 4),)))
        draw()
    doc.end_tile_meta_edit()
    draw()
    assert gl.uploads == 1, f"{gl.uploads} uploads for one drag"

    doc.replace_tileset(0, _tileset("repainted", shade=200))
    draw()
    assert gl.uploads == 2


def test_no_pane_stamps_a_tileset_texture_with_the_metadata_counter() -> None:
    """The five readers the audit named. Pinned by source because a pane draw
    needs an imgui frame; the behaviour is the test above."""
    for module in (
        plotter_canvas,
        plotter_stamps,
        plotter_tileset,
        plotter_tileset_editor,
    ):
        source = inspect.getsource(module)
        calls = re.findall(r"tileset_texture\((.*?)\)\s*\n", source, flags=re.S)
        assert calls, module.__name__
        for call in calls:
            assert "tileset_epoch" not in call, (module.__name__, call)
        assert "tileset_pixel_epoch" in source, module.__name__
    # The picker takes the stamp as an argument; its caller is what decides.
    assert "doc.tileset_pixel_epoch" in inspect.getsource(plotter_tileset.draw)


# --- plotter-10: Duplicate's ValueError ----------------------------------------


def _toasts() -> tuple[SimpleNamespace, list[tuple[str, str]]]:
    seen: list[tuple[str, str]] = []
    return SimpleNamespace(toast=lambda text, kind="info": seen.append((text, kind))), seen


def _too_deep_doc(monkeypatch) -> tuple[MapDoc, object]:
    """A tree already past the ceiling -- what a file read before the readers
    capped nesting leaves behind."""
    doc = MapDoc(4, 4, 16, 16)
    parent = None
    for _ in range(5):
        group = doc.add_group_layer(parent_uid=parent)
        parent = group.uid
    outer = doc.layers[0]
    monkeypatch.setattr(_map_layers, "MAX_GROUP_DEPTH", 2)
    return doc, outer


def test_duplicating_a_group_past_the_nesting_cap_is_refused_by_name(monkeypatch) -> None:
    doc, outer = _too_deep_doc(monkeypatch)
    ctx, toasts = _toasts()
    depth = len(doc.history)
    count = len(doc.all_layers())

    plotter_layers._duplicate_layer(ctx, doc, outer.uid)  # must not raise

    assert len(doc.all_layers()) == count and len(doc.history) == depth
    assert toasts and toasts[-1][1] == "error" and "deep" in toasts[-1][0]


def test_the_layer_menu_duplicate_row_frames_the_engine_refusal(monkeypatch) -> None:
    from realmspinner.studio import controls

    doc, outer = _too_deep_doc(monkeypatch)
    doc.set_active_layer(outer.uid)
    tab = SimpleNamespace(doc=doc, busy=False)
    ctx, toasts = _toasts()
    count = len(doc.all_layers())

    def menu_item(label, shortcut="", selected=False, enabled=True, **kwargs):
        return (label.startswith("Duplicate layer##"), selected)

    monkeypatch.setattr(controls, "menu_item", menu_item)
    monkeypatch.setattr(controls, "menu_separator", lambda: None)

    plotter_menu._layer_rows(ctx, None, tab)  # must not raise

    assert len(doc.all_layers()) == count
    assert toasts and toasts[-1][1] == "error" and "deep" in toasts[-1][0]


def test_a_refused_duplicate_does_not_burn_layer_or_object_ids(monkeypatch) -> None:
    doc, outer = _too_deep_doc(monkeypatch)
    layer_id, object_id = doc.next_layer_id, doc.next_object_id
    with pytest.raises(ValueError):
        doc.duplicate_layer(outer.uid)
    assert (doc.next_layer_id, doc.next_object_id) == (layer_id, object_id)


def test_no_duplicate_door_calls_the_engine_unframed() -> None:
    for module in (plotter_layers, plotter_menu):
        source = inspect.getsource(module)
        stripped = re.sub(r'"""[\s\S]*?"""', "", source)
        offenders = [
            line.strip()
            for line in stripped.splitlines()
            if "doc.duplicate_layer(" in line and not line.lstrip().startswith("#")
        ]
        # The one legitimate call sits inside ``_duplicate_layer``'s try.
        assert len(offenders) <= (1 if module is plotter_layers else 0), offenders
