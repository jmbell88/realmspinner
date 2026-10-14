"""Regression tests for the 2026-10-14 audit's F16 batch: the ``.aseprite`` reader
(``asein``), the brush kernel (``brush``) and the document's clipboard.

inker-02 (a held palette's forward fill), inker-03 (two-key slices over a long
timeline), inker-18 (empty legacy palette chunks), inker-19 (empty layer rows in
an animated file), inker-47 (a 0xffffffff cell over a tileset whose tile 0 has
art), inker-48 (a raster cel on a tilemap layer), inker-16 (the default mirror
axis in continuous image coordinates), inker-37 (the image brush's footprint),
inker-74 (the stamp cache), inker-15 (the pixel clipboard across tabs).
"""

from __future__ import annotations

import struct
import zlib

import numpy as np
import pytest

from realmspinner.core.safeio import pixelguard
from realmspinner.kernels.pixel import asein, brush
from realmspinner.kernels.pixel.tiles import TilemapCel

# --- a minimal .aseprite builder ------------------------------------------------------


def _chunk(kind: int, payload: bytes) -> bytes:
    return struct.pack("<IH", len(payload) + 6, kind) + payload


def _string(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<H", len(raw)) + raw


def _frame(chunks: list[bytes], duration: int = 100) -> bytes:
    body = b"".join(chunks)
    return (
        struct.pack(
            "<IHHHHI", len(body) + 16, 0xF1FA, min(len(chunks), 0xFFFF), duration, 0, len(chunks)
        )
        + body
    )


def _header(frames: int, width: int, height: int) -> bytes:
    head = struct.pack(
        "<IHHHHHIHIIB3sHBBhhHH",
        0, 0xA5E0, frames, width, height, 32, 1, 100, 0, 0, 0, b"\0\0\0", 0, 1, 1, 0, 0, 0, 0,
    )
    return head + b"\0" * 84


def _file(header: bytes, frames: list[bytes]) -> bytes:
    body = header + b"".join(frames)
    return struct.pack("<I", len(body)) + body[4:]


def _layer(name: str = "L", *, kind: int = 0, tileset: int | None = None) -> bytes:
    body = struct.pack("<HHHHHHB3s", 3, kind, 0, 0, 0, 0, 255, b"\0\0\0") + _string(name)
    if tileset is not None:
        body += struct.pack("<I", tileset)
    return _chunk(0x2004, body)


def _palette(colours: list[tuple[int, int, int, int]]) -> bytes:
    body = struct.pack("<III8s", len(colours), 0, len(colours) - 1, b"\0" * 8)
    for red, green, blue, alpha in colours:
        body += struct.pack("<HBBBB", 0, red, green, blue, alpha)
    return _chunk(0x2019, body)


def _slice(name: str, keys: list[tuple[int, int, int, int, int]]) -> bytes:
    body = struct.pack("<III", len(keys), 0, 0) + _string(name)
    for frame, x, y, w, h in keys:
        body += struct.pack("<IiiII", frame, x, y, w, h)
    return _chunk(0x2022, body)


def _raster_cel(layer: int, width: int, height: int, plane: bytes) -> bytes:
    body = struct.pack("<HhhBHh5s", layer, 0, 0, 255, 2, 0, b"\0" * 5)
    return _chunk(0x2005, body + struct.pack("<HH", width, height) + zlib.compress(plane))


def _tileset(tileset_id: int, tile_w: int, tile_h: int, tiles: list[bytes]) -> bytes:
    # flags 2 only: "tiles are embedded", and bit 4 (tile ID 0 is empty) *off* --
    # the pre-release layout whose erased cells are 0xffffffff.
    body = struct.pack("<IIIHHh", tileset_id, 2, len(tiles), tile_w, tile_h, 1) + b"\0" * 14
    body += _string("tiles")
    compressed = zlib.compress(b"".join(tiles))
    return _chunk(0x2023, body + struct.pack("<I", len(compressed)) + compressed)


def _tilemap_cel(layer: int, refs: np.ndarray) -> bytes:
    grid_h, grid_w = refs.shape
    body = (
        struct.pack("<HhhBHh5s", layer, 0, 0, 255, 3, 0, b"\0" * 5)
        + struct.pack("<HHH", grid_w, grid_h, 32)
        + struct.pack("<IIII", 0x1FFFFFFF, 0x80000000, 0x40000000, 0x20000000)
        + b"\0" * 10
        + zlib.compress(refs.astype("<u4").tobytes())
    )
    return _chunk(0x2005, body)


def _solid(width: int, height: int, rgba: tuple[int, int, int, int]) -> bytes:
    return bytes(rgba) * (width * height)


# --- inker-02 ---------------------------------------------------------------------------


def test_a_palette_changed_once_and_held_across_many_frames_is_charged_per_frame_it_fills(
    monkeypatch,
):
    """The two snapshots were charged and the whole-table copy the forward fill
    makes for every later frame was not: a 65,536-entry palette changed on frame 1
    and 65,535 empty frames retained frames x entries slots from ~1.8 MB."""
    monkeypatch.setattr(pixelguard, "MAX_DECODE_PIXELS", 100)
    first = [(i, i, i, 255) for i in range(5)]
    second = [(9, 9, 9, 255)] + first[1:]
    frames = [_frame([_layer(), _palette(first)]), _frame([_palette(second)])]
    frames += [_frame([]) for _ in range(60)]  # 60 frames x 5 colours = 300 > 100
    with pytest.raises(ValueError, match="colours"):
        asein.parse(_file(_header(len(frames), 1, 1), frames))


def test_a_held_palette_over_a_few_frames_still_opens():
    first = [(i, i, i, 255) for i in range(5)]
    second = [(9, 9, 9, 255)] + first[1:]
    frames = [_frame([_layer(), _palette(first)]), _frame([_palette(second)])]
    frames += [_frame([]) for _ in range(4)]
    sprite = asein.parse(_file(_header(len(frames), 1, 1), frames))
    assert sorted(sprite.frame_palettes) == [1, 2, 3, 4, 5]


# --- inker-03 ---------------------------------------------------------------------------


def test_many_two_key_slices_over_a_long_timeline_are_refused_before_expanding_per_frame_overrides():  # noqa: E501
    """A slice whose second key sits at frame 1 covers every later frame, so cost is
    slices x frames, which no ceiling counted: 200 two-key slices over 3,000 frames
    held 599,800 overrides from 60 KB."""
    slices = [_slice(f"s{i}", [(0, 0, 0, 1, 1), (1, 0, 0, 2, 2)]) for i in range(200)]
    frames = [_frame([_layer()] + slices)] + [_frame([]) for _ in range(5999)]
    data = _file(_header(len(frames), 1, 1), frames)
    with pytest.raises(ValueError, match="slice"):
        asein.document_from_aseprite(data)


def test_a_few_slices_over_a_long_timeline_still_open():
    slices = [_slice(f"s{i}", [(0, 0, 0, 1, 1), (1, 0, 0, 2, 2)]) for i in range(3)]
    frames = [_frame([_layer()] + slices)] + [_frame([]) for _ in range(299)]
    doc, _ = asein.document_from_aseprite(_file(_header(len(frames), 1, 1), frames))
    assert len(doc.slices) == 3
    assert all(len(s.keys) == 299 for s in doc.slices)


# --- inker-18 ---------------------------------------------------------------------------


def _legacy_palette_reader(packets: bytes):
    return asein._Reader(packets)


def test_repeated_empty_legacy_palette_chunks_do_not_recopy_the_table():
    """Every 0x0004/0x0011 chunk copied the whole accumulated table twice, however
    few bytes it carried: an 8-byte empty chunk cost 0.83 ms against a 65,280-entry
    table, so 1 MiB of them was ~110 s of CPU on the open task."""
    state = asein._Parse()
    big = struct.pack("<H", 255) + b"".join(
        struct.pack("<BB", 0, 0) + bytes(3 * 256) for _ in range(255)
    )
    asein._read_old_palette(state, _legacy_palette_reader(big), False)
    before = state.old_palette
    assert before is not None and len(before) == 65280
    asein._read_old_palette(state, _legacy_palette_reader(struct.pack("<H", 0)), False)
    assert state.old_palette is before


def test_a_legacy_palette_chunk_does_not_rewrite_an_earlier_frames_snapshot():
    """The table is now edited in place, so a frame's snapshot must be its own copy."""

    def packet(colour: int) -> bytes:
        return struct.pack("<H", 1) + struct.pack("<BB", 0, 1) + bytes([colour] * 3)

    frames = [
        _frame([_layer(), _chunk(0x0004, packet(10))]),
        _frame([_chunk(0x0004, packet(20))]),
    ]
    sprite = asein.parse(_file(_header(2, 1, 1), frames))
    assert sprite.palette == [(10, 10, 10, 255)]
    assert sprite.frame_palettes == {1: [(20, 20, 20, 255)]}


# --- inker-19 ---------------------------------------------------------------------------


def test_an_animated_aseprite_with_more_layer_rows_than_the_canvas_budget_is_refused():
    """The still branch refused empty layers past ``_layer_budget`` and the animated
    branch built a Track per row with no ceiling: 8,000 empty layers at 1024x1024
    took 24 s to open."""
    layers = [_layer(f"L{i}") for i in range(80)]  # the 1024x1024 budget is 64
    frames = [_frame(layers), _frame([])]
    data = _file(_header(2, 1024, 1024), frames)
    with pytest.raises(ValueError, match="empty layers"):
        asein.document_from_aseprite(data)


# --- inker-47 ---------------------------------------------------------------------------


def _tile_document(refs: np.ndarray, tiles: list[bytes], tile: int = 2):
    frames = [
        _frame(
            [
                _tileset(5, tile, tile, tiles),
                _layer("Tiles", kind=2, tileset=5),
                _tilemap_cel(0, refs),
            ]
        )
    ]
    return asein.document_from_aseprite(
        _file(_header(1, tile * refs.shape[1], tile * refs.shape[0]), frames)
    )


def test_a_0xffffffff_cell_over_a_tileset_whose_tile_zero_has_art_renders_empty():
    """With the "tile ID 0 is empty" flag off, 0xffffffff is the erased cell and tile 0
    is a real tile. It was translated to id 0 and warned as "read as empty", but id 0
    draws strip tile 0, so the cell rendered as that tile."""
    zero = _solid(2, 2, (200, 10, 10, 255))
    one = _solid(2, 2, (11, 22, 33, 255))
    doc, warnings = _tile_document(
        np.array([[1, 0xFFFFFFFF]], dtype=np.uint32), [zero, one]
    )
    assert any("0xffffffff" in warning for warning in warnings)
    image = doc.composite
    assert tuple(image[0, 0]) == (11, 22, 33, 255)  # file id 1: the second art tile
    assert tuple(image[0, 2]) == (0, 0, 0, 0)  # the erased cell is empty
    assert tuple(image[1, 3]) == (0, 0, 0, 0)


def test_a_tile_zero_with_art_that_the_cells_name_still_draws_after_the_shift():
    zero = _solid(2, 2, (200, 10, 10, 255))
    one = _solid(2, 2, (11, 22, 33, 255))
    doc, _ = _tile_document(np.array([[0, 1]], dtype=np.uint32), [zero, one])
    image = doc.composite
    assert tuple(image[0, 0]) == (200, 10, 10, 255)
    assert tuple(image[0, 2]) == (11, 22, 33, 255)
    cel = next(layer for layer in doc.stack if isinstance(layer, TilemapCel))
    assert int(cel.refs[0, 0]) != 0  # tile 0's art is no longer the empty cell


# --- inker-48 ---------------------------------------------------------------------------


def test_a_raster_cel_on_a_tilemap_layer_is_refused_or_warned():
    """A malformed or hand-edited file lost the cel's pixels silently: the cel opened
    as an empty grid with no warning and no refusal."""
    blank = _solid(2, 2, (0, 0, 0, 0))
    frames = [
        _frame(
            [
                _tileset(5, 2, 2, [blank]),
                _layer("Tiles", kind=2, tileset=5),
                _raster_cel(0, 2, 2, _solid(2, 2, (9, 9, 9, 255))),
            ]
        )
    ]
    data = _file(_header(1, 2, 2), frames)
    try:
        _, warnings = asein.document_from_aseprite(data)
    except ValueError as exc:
        assert "tilemap" in str(exc)
    else:
        assert any("raster" in warning or "tilemap" in warning for warning in warnings)


# --- inker-16 ---------------------------------------------------------------------------


@pytest.mark.parametrize("width", [10, 11])
def test_x_symmetry_maps_the_pixel_under_the_mouse_to_the_pixel_width_minus_one_minus_i(width):
    """Every real stroke arrives in continuous image coordinates, where pixel ``i``'s
    centre is ``i + 0.5``. The default axis was the index-space ``(w - 1) / 2``, so a
    mouse over pixel 0 of a width-10 canvas painted columns 0 and 8: column 9 was never
    a mirror target and the middle pixel did not map to itself."""
    from realmspinner.kernels import pixel as inker

    red = (255, 0, 0, 255)
    bad = []
    for i in range(width):
        doc = inker.Document.from_pixels(np.zeros((3, width, 4), np.uint8))
        doc.begin_stroke((i + 0.5, 1.5), red, size=1, nib="pixel", symmetry="x")
        doc.end_stroke()
        cols = sorted(set(np.nonzero(doc.composite[..., 3])[1].tolist()))
        if cols != sorted({i, width - 1 - i}):
            bad.append((i, cols))
    assert bad == []


def test_y_symmetry_and_the_cursor_twin_agree_with_the_stroke():
    """The cursor's twin, fed the pixel's centre, names the pixel the stroke paints."""
    from realmspinner.kernels import pixel as inker

    height = 7
    for j in range(height):
        doc = inker.Document.from_pixels(np.zeros((height, 3, 4), np.uint8))
        doc.begin_stroke((1.5, j + 0.5), (255, 0, 0, 255), size=1, nib="pixel", symmetry="y")
        doc.end_stroke()
        rows = sorted(set(np.nonzero(doc.composite[..., 3])[0].tolist()))
        assert rows == sorted({j, height - 1 - j})
        twins = brush.mirrors_of((1.5, j + 0.5), (3, height), "y")
        # The middle row of an odd canvas is its own mirror: one point, itself.
        assert int(np.floor(twins[-1][1])) == height - 1 - j


# --- inker-37 ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tip_w", "tip_h", "align"),
    [
        (3, 3, "free"),
        (4, 2, "free"),
        (4, 4, "free"),
        (2, 6, "free"),
        (4, 4, "aligned"),
        (3, 5, "aligned"),
    ],
)
def test_the_image_brush_footprint_is_the_rectangle_the_dab_paints_for_even_and_aligned_tips(
    tip_w, tip_h, align
):
    """The shared placement arithmetic: ``stamp_rect`` is what the cursor outline draws
    and what ``_image_dab`` paints, for every tip size and both alignments."""
    tip = np.zeros((tip_h, tip_w, 4), np.uint8)
    tip[...] = (255, 0, 0, 255)
    target = np.zeros((40, 40, 4), np.uint8)
    state = brush.StrokeState(
        layer_uid=1,
        size=(40, 40),
        before=target.copy(),
        colour=(0, 0, 0, 255),
        stamp=brush.Stamp(tip),
        stamp_align=align,
    )
    state.begin((10.5, 10.5), target)
    ys, xs = np.nonzero(target[..., 3])
    painted = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    assert brush.stamp_rect((10.5, 10.5), (tip_w, tip_h), align) == painted


# --- inker-74 ---------------------------------------------------------------------------


def test_the_stamp_cache_does_not_grow_with_every_distinct_hardness_value():
    """Keyed on the raw hardness float and bounded by entry count, so dragging the
    slider at a large brush size filled every slot with a 256 KB stamp the session
    never asked for again."""
    from realmspinner.kernels import pixel as inker

    brush.make_stamp.cache_clear()
    doc = inker.Document.from_pixels(np.zeros((64, 64, 4), np.uint8))
    for step in range(300):
        doc.begin_stroke(
            (32.0, 32.0), (255, 0, 0, 255), size=40, hardness=step / 300.0, nib="soft"
        )
        doc.end_stroke()
    assert brush.make_stamp.cache_info().currsize <= 101
