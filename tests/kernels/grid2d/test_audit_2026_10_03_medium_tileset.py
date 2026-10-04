"""plotter-20 (2026-10-03 audit): a metadata-only edit re-copied the whole atlas."""

from __future__ import annotations

import numpy as np

from realmspinner.kernels.grid2d.tileset import TileMeta, Tileset


def _tileset() -> Tileset:
    pixels = np.zeros((64, 64, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    return Tileset(name="t", pixels=pixels, tile_w=16, tile_h=16)


def test_with_meta_shares_the_frozen_pixels_instead_of_copying_the_atlas():
    base = _tileset()
    edited = base.with_meta(3, TileMeta(class_name="Water"))
    cleared = edited.with_meta(3, None)

    assert edited.pixels is base.pixels
    assert cleared.pixels is base.pixels
    # Still frozen, and still the edit the caller asked for.
    assert not edited.pixels.flags.writeable
    assert edited.tiles[3].class_name == "Water"
    assert 3 not in cleared.tiles and 3 not in base.tiles


def test_with_meta_still_refuses_an_empty_record_into_the_store():
    edited = _tileset().with_meta(1, TileMeta())
    assert edited.tiles == {}
