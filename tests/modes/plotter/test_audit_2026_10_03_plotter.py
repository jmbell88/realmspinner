"""Closing plotter-01..07 from the 2026-10-03 audit of Plotter.

One test per finding, named for the claim it proves.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from realmspinner.kernels.grid2d import gid as gidlib
from realmspinner.kernels.grid2d.tileset import TileFrame, TileMeta, Tileset, compose_collection
from realmspinner.studio.modes.plotter import mode as plotter_mode
from realmspinner.studio.modes.plotter.engine import project, render, tmx
from realmspinner.studio.modes.plotter.engine.tilemap import MapDoc, MapObject, new_uid


def _solid(colour: tuple[int, int, int], w: int = 8, h: int = 8) -> np.ndarray:
    array = np.zeros((h, w, 4), dtype=np.uint8)
    array[..., :3] = colour
    array[..., 3] = 255
    return array


def _ctx():
    toasts: list[tuple] = []
    ctx = SimpleNamespace(
        state=SimpleNamespace(plotter=None, mode="plotter", preview={}),
        settings=SimpleNamespace(get=lambda _k: None, set=lambda _k, _v: None),
        toast=lambda *a, **_k: toasts.append(a),
        viewer=None,
        toasts=toasts,
    )
    return ctx


@pytest.fixture(autouse=True)
def _no_pygame_display(monkeypatch):
    import pygame

    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0)


# --- plotter-01 -----------------------------------------------------------------


def test_paste_refuses_a_clipboard_naming_a_tileset_the_map_no_longer_holds():
    ctx = _ctx()
    tab = plotter_mode.new_document(ctx, (4, 4, 8, 8))
    state = plotter_mode.ensure(ctx)
    ref = tab.doc.add_tileset(Tileset(name="G", pixels=_solid((9, 9, 9)), tile_w=8, tile_h=8))
    layer = tab.doc.tile_layers()[0]
    tab.doc.write_region(layer.uid, 0, 0, np.array([[ref.firstgid]], gidlib.DTYPE))
    state.clipboard = np.array([[ref.firstgid]], dtype=gidlib.DTYPE)
    state.clipboard_doc = tab.uid
    # The route in the audit: undo the cells and the tileset add, then paste.
    plotter_mode.undo(ctx, tab)
    plotter_mode.undo(ctx, tab)
    assert ref not in tab.doc.tilesets

    plotter_mode._paste(ctx, state, tab)

    assert state.brush is None, "a gid no tileset accounts for must not reach the brush"
    assert state.clipboard is None


def test_removing_a_tileset_drops_a_clipboard_that_named_it():
    ctx = _ctx()
    tab = plotter_mode.new_document(ctx, (4, 4, 8, 8))
    state = plotter_mode.ensure(ctx)
    tab.doc.add_tileset(Tileset(name="G", pixels=_solid((9, 9, 9)), tile_w=8, tile_h=8))
    state.clipboard = np.array([[1]], dtype=gidlib.DTYPE)
    state.clipboard_doc = tab.uid
    plotter_mode.remove_tileset(ctx, 0)
    assert state.clipboard is None


# --- plotter-02 -----------------------------------------------------------------


def _animated_set() -> Tileset:
    pixels = np.concatenate([_solid((255, 0, 0)), _solid((0, 0, 255))], axis=1)
    # Tile 0 is red but its animation starts on tile 1 (blue).
    return Tileset(
        name="anim",
        pixels=pixels,
        tile_w=8,
        tile_h=8,
        tiles={0: TileMeta(animation=(TileFrame(1, 100), TileFrame(0, 100)))},
    )


def test_render_map_draws_an_animated_tile_as_its_first_frame():
    doc = MapDoc(1, 1, 8, 8)
    doc.add_tileset(_animated_set())
    layer = doc.add_tile_layer("L")
    doc.write_region(layer.uid, 0, 0, np.array([[1]], gidlib.DTYPE))

    out = render.render_map(doc)

    assert tuple(out[0, 0]) == (0, 0, 255, 255)
    assert tuple(render.minimap(doc)[0, 0])[:3] == (0, 0, 255)


# --- plotter-03 -----------------------------------------------------------------


def test_a_minimap_of_a_sparse_image_collection_draws_instead_of_raising():
    atlas, collection = compose_collection(
        {0: _solid((255, 0, 0), 8, 8), 5: _solid((0, 255, 0), 8, 8), 9: _solid((0, 0, 255), 8, 8)}
    )
    tileset = Tileset(name="c", pixels=atlas, tile_w=8, tile_h=8, collection=collection)
    doc = MapDoc(3, 1, 8, 8)
    ref = doc.add_tileset(tileset)
    layer = doc.add_tile_layer("L")
    row = [ref.firstgid, ref.firstgid + 5, ref.firstgid + 9]
    doc.write_region(layer.uid, 0, 0, np.array([row], gidlib.DTYPE))

    mini = render.minimap(doc)

    assert tuple(mini[0, 0, :3]) == (255, 0, 0)
    assert tuple(mini[0, 1, :3]) == (0, 255, 0)
    assert tuple(mini[0, 2, :3]) == (0, 0, 255)


# --- plotter-04 -----------------------------------------------------------------


@pytest.mark.parametrize("projection", [project.STAGGERED, project.HEXAGONAL])
def test_a_resize_with_no_shift_leaves_a_staggered_or_hexagonal_maps_objects_where_they_were(
    projection,
):
    doc = MapDoc(8, 8, 32, 16, projection=projection)
    if projection == project.HEXAGONAL:
        doc.hex_side = 16
    layer = doc.add_object_layer()
    points = [(40.0, 20.0), (70.0, 33.0), (5.0, 3.0)]
    for x, y in points:
        doc.add_object(layer.uid, MapObject(uid=new_uid(), name="a", kind="point", x=x, y=y))

    doc.resize(9, 8)

    got = [(o.x, o.y) for o in doc.layer(layer.uid).objects]
    assert got == points


# --- plotter-05 -----------------------------------------------------------------


def test_growing_an_isometric_maps_height_keeps_its_objects_on_their_cells():
    doc = MapDoc(6, 6, 32, 16, projection=project.ISOMETRIC)
    x, y = project.cell_corner(doc._lattice(), 2, 3)
    layer = doc.add_object_layer()
    doc.add_object(layer.uid, MapObject(uid=new_uid(), name="a", kind="point", x=x, y=y))

    doc.resize(6, 10)

    moved = doc.layer(layer.uid).objects[0]
    column, row = project.cell_point(doc._lattice(), moved.x, moved.y)
    assert (round(column), round(row)) == (2, 3)


# --- plotter-06 -----------------------------------------------------------------


def test_chunks_in_two_layers_that_span_past_the_extent_cap_together_are_refused():
    body = (
        '<layer id="1" name="A"><data encoding="csv">'
        '<chunk x="0" y="0" width="1" height="1">3</chunk></data></layer>'
        '<layer id="2" name="B"><data encoding="csv">'
        '<chunk x="100000" y="0" width="1" height="1">3</chunk></data></layer>'
    )
    data = (
        '<map version="1.10" orientation="orthogonal" width="2" height="2" '
        'tilewidth="16" tileheight="16" infinite="1">'
        f'<tileset firstgid="1" source="t.tsx"/>{body}</map>'
    ).encode()

    def tsx_loader(_s: str) -> Tileset:
        return Tileset(name="t", pixels=_solid((1, 1, 1), 32, 32), tile_w=16, tile_h=16)

    with pytest.raises(ValueError, match="past the 4096 a side this build reads"):
        tmx.read_tmx(data, image_loader=lambda _s: _solid((1, 1, 1), 32, 32), tsx_loader=tsx_loader)


# --- plotter-07 -----------------------------------------------------------------


def _locked_group_with(doc, layer):
    group = doc.add_group_layer("Locked")
    doc.move_layer(layer.uid, 0, parent_uid=group.uid)
    doc.set_layer_props(group.uid, locked=True)
    doc.set_active_layer(layer.uid)


def test_a_lock_on_a_group_stops_delete_cut_paste_and_duplicate_inside_it():
    ctx = _ctx()
    tab = plotter_mode.new_document(ctx, (4, 4, 8, 8))
    state = plotter_mode.ensure(ctx)
    doc = tab.doc
    ref = doc.add_tileset(Tileset(name="G", pixels=_solid((9, 9, 9)), tile_w=8, tile_h=8))
    tiles = doc.tile_layers()[0]
    doc.write_region(tiles.uid, 0, 0, np.full((2, 2), ref.firstgid, gidlib.DTYPE))
    objs = doc.add_object_layer("Things")
    obj = MapObject(uid=new_uid(), name="o", kind="rect", x=0, y=0, w=8, h=8)
    doc.add_object(objs.uid, obj)
    _locked_group_with(doc, tiles)
    _locked_group_with(doc, objs)

    # Cells: Delete and Cut.
    doc.set_active_layer(tiles.uid)
    state.tool = "select"
    state.select = (0, 0, 1, 1)
    depth = len(doc.history)
    plotter_mode._delete(ctx, state, tab)
    plotter_mode._copy(ctx, state, tab, cut=True)
    assert (doc.layer(tiles.uid).data == ref.firstgid).sum() == 4
    assert len(doc.history) == depth

    # Objects: Cut, Paste, Duplicate, Delete, and the menu rows' own gate.
    doc.set_active_layer(objs.uid)
    state.tool = "object"
    state.select_object(obj.uid)
    depth = len(doc.history)
    plotter_mode._copy_object(ctx, state, tab, cut=True)
    state.clipboard = plotter_mode.ObjectClip(obj=obj)
    state.clipboard_doc = tab.uid
    plotter_mode._paste_object(ctx, state, tab)
    plotter_mode._duplicate_object(ctx, state, tab)
    plotter_mode._delete(ctx, state, tab)
    assert len(doc.layer(objs.uid).objects) == 1
    assert len(doc.history) == depth
    assert plotter_mode.layer_locked(doc, doc.layer(objs.uid))
