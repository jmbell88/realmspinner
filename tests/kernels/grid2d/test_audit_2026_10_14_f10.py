"""The 2026-10-14 audit, fixer F10: the grid2d kernel's three claims.

plotter-tiles-01: a Wang set may not name a tile its tileset does not have.
plotter-tiles-04: a collection's id-to-slot lookup is not a scan.
plotter-tiles-05: gid.py's reason for omitting the hex bit is the real one.
"""

from __future__ import annotations

import inspect

import numpy as np

from realmspinner.kernels.grid2d import gid as gidlib
from realmspinner.kernels.grid2d.tileset import Collection, Tileset, compose_collection
from realmspinner.kernels.grid2d.wang import WangColour, WangSet


def _pixels(size: int = 32) -> np.ndarray:
    out = np.zeros((size, size, 4), dtype=np.uint8)
    out[..., 3] = 255
    return out


def _wang(*local_ids: int) -> WangSet:
    return WangSet(
        name="W",
        kind="corner",
        colours=(WangColour("a", "#00ff00"),),
        tiles={local: (0, 1, 0, 1, 0, 1, 0, 1) for local in local_ids},
    )


def test_a_wang_set_naming_a_tile_the_tileset_does_not_have_is_trimmed_at_the_kernel() -> None:
    # 32px / 16px = four tiles, local ids 0..3.
    tileset = Tileset(
        name="t",
        pixels=_pixels(),
        tile_w=16,
        tile_h=16,
        wangsets=(_wang(0, 3, 4, 500, -1),),
    )
    assert sorted(tileset.wangsets[0].tiles) == [0, 3]


def test_an_in_range_wang_set_is_kept_as_the_same_object() -> None:
    wangset = _wang(0, 1, 2, 3)
    tileset = Tileset(
        name="t", pixels=_pixels(), tile_w=16, tile_h=16, wangsets=(wangset,)
    )
    assert tileset.wangsets[0] is wangset


def test_a_collection_wang_set_may_only_name_ids_the_collection_has() -> None:
    images = {2: _pixels(16), 7: _pixels(16)}
    atlas, collection = compose_collection(images)
    tileset = Tileset(
        name="c",
        pixels=atlas,
        tile_w=16,
        tile_h=16,
        collection=collection,
        wangsets=(_wang(2, 3, 7, 8),),
    )
    assert sorted(tileset.wangsets[0].tiles) == [2, 7]


class _NoScan(tuple):
    def index(self, *args, **kwargs):  # noqa: D401 - a tripwire
        raise AssertionError("slot_of scanned ids")


def test_a_collection_tile_lookup_does_not_scan_every_id() -> None:
    ids = tuple(range(0, 6000, 3))
    collection = Collection(
        ids=ids, sizes=tuple((8, 8) for _ in ids), columns=10, cell_w=8, cell_h=8
    )
    object.__setattr__(collection, "ids", _NoScan(ids))
    assert collection.slot_of(ids[-1]) == len(ids) - 1
    assert collection.slot_of(ids[0]) == 0
    assert collection.slot_of(1) is None
    assert collection.slot_of(-3) is None


def test_slot_of_docstring_no_longer_claims_a_linear_scan_is_fine() -> None:
    doc = inspect.getdoc(Collection.slot_of) or ""
    assert "linear scan" not in doc.lower()


def test_gid_comment_does_not_claim_tmx_refuses_hexagonal_grids() -> None:
    source = inspect.getsource(gidlib)
    assert "still refuses at the door" not in source
    assert "both projections it draws" not in source
    # The real reason: hex maps load, only the rotation flag is refused.
    assert "hexagonal projection loads" in source
