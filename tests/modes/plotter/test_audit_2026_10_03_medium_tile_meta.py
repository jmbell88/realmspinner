"""plotter-20 (2026-10-03 audit): dragging a collision handle must not copy the atlas."""

from __future__ import annotations

import numpy as np

from realmspinner.kernels.grid2d.tileset import TileMeta, Tileset
from realmspinner.studio.modes.plotter.engine.tilemap import MapDoc


def test_a_live_metadata_drag_keeps_the_same_pixel_buffer():
    pixels = np.zeros((64, 64, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    doc = MapDoc(4, 4, 16, 16)
    doc.add_tileset(Tileset(name="t", pixels=pixels, tile_w=16, tile_h=16))
    atlas = doc.tilesets[0].tileset.pixels

    doc.begin_tile_meta_edit(0, 2)
    for step in range(5):
        assert doc.live_tile_meta(TileMeta(class_name=f"c{step}")) is True
    doc.end_tile_meta_edit()

    assert doc.tilesets[0].tileset.pixels is atlas
