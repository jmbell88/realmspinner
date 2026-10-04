"""The 2026-10-03 audit's Medium findings inker-57, -58, -59, -60 and -66.

Tiles, sheets-in, palettes and opening: five defects with workarounds, each
pinned by a test whose name is the claim.
"""

from __future__ import annotations

import tracemalloc

import numpy as np
from PIL import Image

from realmspinner.kernels.pixel import tiling
from realmspinner.kernels.pixel.sheetin import document_from_sheet
from realmspinner.kernels.pixel.tiles import TilemapCel, TilesetSlot
from realmspinner.pipelines import pixel

MIB = 1 << 20


def _peak(fn) -> int:
    """Peak bytes numpy allocated while ``fn`` ran."""
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        fn()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


# --- inker-57 ---------------------------------------------------------------


def test_map_palette_fallback_peak_allocation_is_bounded_whatever_the_palette_size(
    monkeypatch,
) -> None:
    """The numpy fallback searched in fixed 65,536-row chunks, so its
    ``(chunk, entries, 3)`` float64 temporary grew with the palette: 4 MiB per
    entry, ~16 GiB at the 4096-entry ceiling. The chunk is sized from the entry
    count now, so the temporary is a fixed byte budget at any palette size."""
    monkeypatch.setattr(pixel, "_nearest_native", lambda flat, plab: None)
    rng = np.random.default_rng(3)
    for side, entries in ((256, 64), (128, 512)):
        image = Image.fromarray(
            rng.integers(0, 256, (side, side, 4), dtype=np.uint8), "RGBA"
        )
        palette = tuple(
            tuple(int(c) for c in row)
            for row in rng.integers(0, 256, (entries, 3), dtype=np.uint8)
        )
        peak = _peak(lambda i=image, p=palette: pixel.map_palette(i, p))
        assert peak < 64 * MIB, (side, entries, peak / MIB)


def test_map_palette_fallback_chunking_does_not_change_a_pick(monkeypatch) -> None:
    """Chunk size must not move the answer: a tiny budget gives the same pixels
    as a roomy one."""
    monkeypatch.setattr(pixel, "_nearest_native", lambda flat, plab: None)
    rng = np.random.default_rng(5)
    image = Image.fromarray(
        rng.integers(0, 256, (40, 40, 4), dtype=np.uint8), "RGBA"
    )
    palette = tuple(
        tuple(int(c) for c in row)
        for row in rng.integers(0, 256, (24, 3), dtype=np.uint8)
    )
    roomy = np.asarray(pixel.map_palette(image, palette))
    monkeypatch.setattr(pixel, "_SEARCH_CHUNK_BYTES", 24 * 24 * 7)
    tight = np.asarray(pixel.map_palette(image, palette))
    assert np.array_equal(roomy, tight)


# --- inker-58 ---------------------------------------------------------------


def _float_dominance(pixels: np.ndarray) -> tuple[float, float]:
    """The float64 reference this statistic used to be computed with."""
    rgb = pixels[:, :, :3].astype(np.float64)

    def one(a: np.ndarray) -> float:
        edge = float(np.abs(a[:, 0] - a[:, -1]).mean())
        interior = float(np.abs(np.diff(a, axis=1)).mean(axis=(0, 2)).max())
        if interior <= 0.0:
            return 0.0 if edge <= 0.0 else float("inf")
        return edge / interior

    return (one(rgb), one(rgb.transpose(1, 0, 2)))


def test_seam_dominance_matches_the_float_reference_and_stays_inside_the_seam_ratio_budget() -> None:  # noqa: E501
    """``seam_dominance`` ran on the frame thread after every stroke of a tiled
    document and still converted the whole flatten to float64 -- the cost
    ``seam_ratio``'s int16 rewrite removed. Same numbers, int16 memory."""
    rng = np.random.default_rng(7)
    for shape in ((32, 32), (48, 20), (20, 48)):
        tile = rng.integers(0, 256, (*shape, 4), dtype=np.uint8)
        got = tiling.seam_dominance(tile)
        assert got == _float_dominance(tile)

    big = rng.integers(0, 256, (768, 768, 4), dtype=np.uint8)
    ratio_peak = _peak(lambda: tiling.seam_ratio(big))
    dominance_peak = _peak(lambda: tiling.seam_dominance(big))
    assert dominance_peak <= ratio_peak * 1.5, (dominance_peak / MIB, ratio_peak / MIB)


# --- inker-59 ---------------------------------------------------------------


def _cel() -> TilemapCel:
    return TilemapCel(
        pixels=np.zeros((4, 4, 4), dtype=np.uint8),
        refs=np.zeros((2, 2), dtype=np.uint32),
    )


def test_a_tilemap_cel_compares_and_hashes_by_identity_like_every_layer() -> None:
    """``TilemapCel`` re-declared a bare ``@dataclass`` under ``Layer``'s
    ``eq=False``, so it regained a generated ``__eq__`` (array truth-value
    errors) and lost ``__hash__``."""
    a, b = _cel(), _cel()
    assert a == a and a != b
    assert b in [a, b] and a not in [b]
    assert [a, b].index(b) == 1
    stack = [a, b]
    stack.remove(b)
    assert stack == [a]
    assert len({a, b, a}) == 2


def test_a_tileset_slot_compares_and_hashes_by_identity() -> None:
    from realmspinner.kernels.pixel.tiles import blank_strip

    a = TilesetSlot(tileset=blank_strip(4, 4))
    b = TilesetSlot(tileset=blank_strip(4, 4))
    assert a == a and a != b
    assert b in [a, b] and a not in [b]
    assert len({a, b}) == 2


# --- inker-60 ---------------------------------------------------------------

_BOM = "﻿"


def test_every_pipelines_palette_reader_accepts_a_bom_prefixed_file() -> None:
    """``service.palettes`` reads with ``utf-8``, which leaves a Notepad BOM in
    place: ``parse_hex`` refused it as 'not a hex colour' and ``parse_gpl`` for
    a missing header, while ``parse_pal`` and ``parse_txt`` stripped it."""
    texts = {
        ".hex": "ff0000\n00ff00\n",
        ".gpl": "GIMP Palette\nName: x\n#\n255 0 0 red\n0 255 0 green\n",
        ".pal": "JASC-PAL\n0100\n2\n255 0 0\n0 255 0\n",
        ".txt": "ffff0000\nff00ff00\n",
    }
    assert set(texts) == set(pixel.PARSERS)
    for suffix, text in texts.items():
        plain = pixel.parse_palette(text, suffix)
        assert pixel.parse_palette(_BOM + text, suffix) == plain, suffix
        assert plain == ((255, 0, 0), (0, 255, 0))


# --- inker-66 ---------------------------------------------------------------


def test_a_pixel_restyle_sheet_opens_with_no_render_base_to_merge_against(
    tmp_path, monkeypatch
) -> None:
    """A pixel restyle opened with the *render's* ``{job, sheet}`` source, so
    Merge re-render loaded the newest raw render and swapped the cleaned pixel
    art for it. A pixel sheet has no re-renderer behind it, so no base."""
    from realmspinner.studio.modes.inker import opening

    size = 8
    atlas = np.zeros((size, 2 * size, 4), dtype=np.uint8)
    atlas[..., 3] = 255
    atlas[:, size:, 0] = 90
    Image.fromarray(atlas, "RGBA").save(tmp_path / "sheet.png")
    record = {
        "name": "walker",
        "frame_size": size,
        "columns": 2,
        "rows": 1,
        "cells": [
            {"x": 0, "y": 0, "w": size, "h": size},
            {"x": size, "y": 0, "w": size, "h": size},
        ],
        "animation": {
            "tags": [{"name": "walk_front", "start": 0, "end": 1, "loop": True}],
            "frames": [],
        },
    }
    for name in ("get_sheet", "get_pixel_sheet"):
        monkeypatch.setattr(
            f"realmspinner.service.sheets.{name}", lambda svc, job, sid: record
        )
    for name in ("sheet_png", "sheet_pixel_png"):
        monkeypatch.setattr(
            f"realmspinner.service.sheets.{name}",
            lambda svc, job, sid: tmp_path / "sheet.png",
        )

    rendered = opening._load_rendered_sheet(object(), "j1", "s1", False)["doc"]
    restyled = opening._load_rendered_sheet(object(), "j1", "s1", True)["doc"]

    assert rendered.sheet_base is not None
    assert rendered.sheet_base.source == {"job": "j1", "sheet": "s1"}
    assert restyled.sheet_base is None
    # The document is otherwise the same sheet: tags and frames survive.
    assert len(restyled.anim.frames) == 2

    # And the ordinary door still records a base when asked, so record_base is a
    # real switch rather than a removal.
    direct = document_from_sheet(atlas, record["cells"], record["animation"])
    assert direct.sheet_base is not None


# --- the extra: Auto-fit nine-slice centre on a keyed frame ------------------


def _keyed_fit_doc():
    from types import SimpleNamespace

    from realmspinner.kernels.pixel.document import Document
    from realmspinner.kernels.pixel.slices import SliceKey

    doc = Document.blank(12, 12)
    plane = doc.stack[0].pixels
    plane[:] = (200, 60, 60, 255)
    plane[:3] = plane[-3:] = (20, 20, 20, 255)
    plane[:, :3] = plane[:, -3:] = (20, 20, 20, 255)
    doc.invalidate_all()
    doc.ensure_animation()
    frame = doc.anim.frames[0].uid
    entry = doc.add_slice((0, 0, 4, 4), name="panel")
    assert doc.set_slice_key(entry.uid, frame, key=SliceKey(bounds=(0, 0, 12, 12)))
    tab = SimpleNamespace(uid="t", doc=doc, frame_uid=frame, busy=False)
    state = SimpleNamespace(slice_uid=entry.uid, transforming=False)
    ctx = SimpleNamespace(state=SimpleNamespace(inker=state))
    return doc, tab, state, ctx, entry, frame


def test_auto_fit_nine_slice_centre_on_a_keyed_frame_writes_that_frames_key():
    """``_run_nineslice_fit`` read the keyed frame's bounds but wrote the centre
    to the *base* slice -- the defect inker-52 repaired for the Nine-slice tick.
    The base (and every unkeyed frame) silently gained a centre while the frame
    the user was looking at stayed without one."""
    from realmspinner.studio.modes.inker import ops as inker_ops

    doc, tab, state, ctx, entry, frame = _keyed_fit_doc()
    assert inker_ops._run_nineslice_fit(ctx, tab)
    assert entry.center is None, "the base must be left alone"
    assert entry.keys[frame].center is not None
    assert entry.keys[frame].bounds == (0, 0, 12, 12)
    # One undo step takes it back.
    doc.undo()
    assert entry.keys[frame].center is None


def test_a_centre_on_a_keyed_frame_alone_still_enables_the_nine_slice_export():
    """``_has_nineslice`` looked only at the base centre, so a centre that now
    lives on a key would leave Export nine-slice panels greyed."""
    from realmspinner.studio.modes.inker import ops as inker_ops

    doc, tab, state, ctx, entry, frame = _keyed_fit_doc()
    assert not inker_ops._has_nineslice(state, tab)
    assert inker_ops._run_nineslice_fit(ctx, tab)
    assert inker_ops._has_nineslice(state, tab)
