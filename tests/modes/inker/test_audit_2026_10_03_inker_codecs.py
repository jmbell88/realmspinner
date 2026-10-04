"""Regressions for the 2026-10-03 audit's inker codec findings (inker-01, 05,
09, 10).

Each test's name is its claim; the shape every one shares is "a document this
build can save is one it can reopen, and a file it opens costs memory in
proportion to its bytes, not to what it names".
"""

from __future__ import annotations

import io
import zipfile

import pytest

from realmspinner.core.safeio import pixelguard
from realmspinner.kernels.pixel import asein, aseout, ora
from realmspinner.kernels.pixel.document import Document


def _png(width: int, height: int) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGBA", (width, height), (9, 9, 9, 255)).save(buf, "PNG")
    return buf.getvalue()


def test_write_ora_refuses_an_animation_with_more_tracks_than_the_reader_will_open(
    monkeypatch,
):
    # inker-01: three tracks over ONE distinct cel passed the cel-only write
    # guard, then _read_animation refused the track count and the whole
    # timeline silently fell back to a flat 1-frame drawing.
    monkeypatch.setattr(ora, "MAX_ORA_LAYERS", 2)
    doc = Document.blank(8, 8)
    doc.ensure_animation()
    for _ in range(2):
        doc.add_layer()
    for _ in range(2):
        doc.add_frame(link=True)
    assert len(doc.anim.tracks) == 3
    assert len(list(doc.anim.unique_cel_layers())) == 1
    with pytest.raises(ValueError, match="layers|tracks"):
        ora.ora_bytes(doc)


def test_a_still_document_with_more_layers_than_this_build_can_reopen_is_refused_at_save(
    monkeypatch,
):
    # inker-05: both writers applied the reopen budget to animations only.
    monkeypatch.setattr(ora, "MAX_ORA_LAYERS", 2)
    doc = Document.blank(4, 4)
    for _ in range(2):
        doc.add_layer()
    with pytest.raises(ValueError, match="layers"):
        ora.ora_bytes(doc)

    # The .aseprite budget is pixels: layers x canvas against the decode cap.
    monkeypatch.setattr(pixelguard, "MAX_DECODE_PIXELS", 40)
    doc = Document.blank(4, 4)
    for layer in list(doc.stack):
        layer.pixels[:, :] = (200, 0, 0, 255)
    for _ in range(2):
        added = doc.add_layer()
        added.pixels[:, :] = (0, 200, 0, 255)
    with pytest.raises(ValueError, match="layers|pixels"):
        aseout.aseprite_bytes(doc)


def test_a_flat_ora_naming_one_member_many_times_does_not_hold_a_copy_per_layer(
    monkeypatch, tmp_path
):
    # inker-09: planes_data kept one raw copy of the member per <layer>
    # element, and re-read it each time. An RGB document needs none of them.
    decoded: list[bytes] = []
    real = pixelguard.opened

    def spy(fp, what, *a, **k):
        decoded.append(fp.getvalue())
        return real(fp, what, *a, **k)

    monkeypatch.setattr(pixelguard, "opened", spy)
    layers = "".join(f'<layer src="data/a.png" name="L{i}"/>' for i in range(8))
    xml = f'<image w="2" h="2"><stack>{layers}</stack></image>'
    path = tmp_path / "many.ora"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", b"image/openraster")
        zf.writestr("stack.xml", xml)
        zf.writestr("data/a.png", _png(2, 2))
    doc = ora.read_ora(path)
    assert len(doc.stack) == 8
    assert len(decoded) == 1, "one member must be decoded once, not once per <layer>"


def test_many_flourish_asset_ids_over_one_png_are_charged_by_pixels_not_count(
    monkeypatch,
):
    # inker-10: the assets map was bounded by count only; 4096 ids over one
    # 256x256 PNG retained 4096 arrays.
    monkeypatch.setattr(pixelguard, "MAX_DECODE_PIXELS", 4 * 4 * 3)
    from types import SimpleNamespace

    members = {f"id{i}": "data/tex.png" for i in range(10)}
    state = SimpleNamespace(assets={}, _asset_members=members)
    doc = SimpleNamespace(flourish={1: state})
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("data/tex.png", _png(4, 4))
    with zipfile.ZipFile(io.BytesIO(buf.getvalue())) as zf:
        ora._read_flourish_assets(doc, zf)
    arrays = list(state.assets.values())
    # Shared, not copied per id: ten ids over one member is one decode.
    assert len({id(a) for a in arrays}) <= 1
    # And a distinct-texture total past the budget stops decoding.
    state2 = SimpleNamespace(
        assets={}, _asset_members={f"id{i}": f"data/t{i}.png" for i in range(10)}
    )
    doc2 = SimpleNamespace(flourish={1: state2})
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for i in range(10):
            zf.writestr(f"data/t{i}.png", _png(4, 4))
    with zipfile.ZipFile(io.BytesIO(buf.getvalue())) as zf:
        ora._read_flourish_assets(doc2, zf)
    assert len(state2.assets) <= 3


def test_aseprite_round_trips_a_per_frame_table_on_frame_zero():
    # inker-06: _write_frame_palettes started at frame 1, so a recoloured
    # frame 0 reopened in the document's base colours.
    from tests.modes.inker.test_frame_palettes import BLUE, HOLE, RED, _at, _indexed

    doc = _indexed(frames=2)
    assert doc.set_frame_palette([HOLE, BLUE], 0) is True
    back, _warnings = asein.document_from_aseprite(aseout.aseprite_bytes(doc))
    assert back.palette_for(back.anim.frames[0]) == [HOLE, BLUE]
    assert back.palette_for(back.anim.frames[1]) == [HOLE, RED]
    back.set_current_frame(0)
    assert _at(back) == (0.0, 0.0, 255.0, 255.0)


def test_dropped_by_aseprite_reports_a_group_blend_mode():
    # inker-07: a Multiply group saved as Normal with the lossy-save prompt
    # silent, because only group opacity was listed.
    doc = Document.blank(4, 4)
    doc.stack[0].pixels[:, :] = (255, 0, 0, 255)
    node = doc.group_layers([0])
    assert node is not None
    assert not any("blend" in item for item in aseout.dropped_by_aseprite(doc))
    assert doc.set_group_props(node.uid, blend="multiply")
    assert any("group blend" in item for item in aseout.dropped_by_aseprite(doc))


def test_many_large_tilemap_cels_are_refused_before_they_are_inflated(monkeypatch):
    # inker-08: tilemap cels were not charged to the cel-pixel running total,
    # so a file naming many big grids was inflated cel by cel. The payload
    # here is not valid zlib: only a refusal that comes *before* the inflate
    # can say "pixels".
    import struct

    monkeypatch.setattr(pixelguard, "MAX_DECODE_PIXELS", 10)
    from tests.modes.inker.test_audit_2026_09_26_w1f2_modes_inker import (
        _ase_file,
        _ase_frame,
        _ase_header,
        _ase_layer_named,
        _chunk,
    )

    body = struct.pack("<HhhBHh5s", 0, 0, 0, 255, 3, 0, b"\0" * 5)
    masks = (0x1FFFFFFF, 0x80000000, 0x40000000, 0x20000000)
    body += struct.pack("<HHHIIII10s", 4, 4, 32, *masks, b"\0" * 10)
    body += b"not zlib at all"
    frames = [_ase_frame([_ase_layer_named("Art"), _chunk(0x2005, body)])]
    data = _ase_file(_ase_header(1, 2, 2), frames)
    with pytest.raises(ValueError, match="pixels"):
        asein.document_from_aseprite(data)


def _tilemap_file(
    canvas: int, tile: int, grid_w: int, grid_h: int, payload: bytes, *, x=0, y=0
):
    """A one-tilemap-layer file whose one cel declares ``grid_w``x``grid_h``."""
    import struct

    from tests.modes.inker.test_asein import _chunk as chunk
    from tests.modes.inker.test_asein import (
        _file,
        _frame,
        _header,
        _layer,
        _rgba,
        _tileset_chunk,
    )

    body = struct.pack("<HhhBHh5s", 0, x, y, 255, 3, 0, b"\0" * 5)
    masks = (0x1FFFFFFF, 0x80000000, 0x40000000, 0x20000000)
    body += struct.pack("<HHHIIII10s", grid_w, grid_h, 32, *masks, b"\0" * 10)
    body += payload
    blank = _rgba(tile, tile, (0, 0, 0, 0))
    chunks = [
        _tileset_chunk(7, tile, tile, [blank]),
        _layer("Map", kind=2, tileset=7),
        chunk(0x2005, body),
    ]
    return _file(_header(1, canvas, canvas), [_frame(chunks)])


def test_one_tilemap_cel_larger_than_the_canvas_can_place_is_refused_before_it_is_inflated(
    monkeypatch,
):
    # inker-08 (residual): the running total bounds the *sum* of the grids, so
    # one cel declaring a 16384x16384 grid (a 1 GiB inflate from ~1 MB of
    # zlib) was only bounded by that shared ceiling, not by its own
    # plausibility. A 2x2 canvas can place a 1x1 grid of 2px tiles; this cel
    # names 4000x4000 of them, and zlib must never be asked to open it.
    # The tileset chunk legitimately inflates; only the cel's own call counts.
    asked: list[str] = []
    real = asein._inflate

    def spy(raw, expected, what):
        asked.append(what)
        return real(raw, expected, what)

    monkeypatch.setattr(asein, "_inflate", spy)
    data = _tilemap_file(2, 2, 4000, 4000, b"not zlib at all")
    with pytest.raises(ValueError, match="larger than"):
        asein.document_from_aseprite(data)
    assert not [w for w in asked if "tilemap cel" in w], (
        "the grid was refused only after its inflate was asked for"
    )


def test_a_tilemap_cel_reaching_a_canvas_width_past_each_edge_still_opens():
    # The same bound's other side: a legitimate cel is allowed to overhang the
    # canvas (it is cropped, with a warning), up to one canvas on each side.
    # 4x4 canvas, 2x2 tiles: a 6x6 grid at (-4, -4) spans -4..8 on both axes.
    import zlib

    import numpy as np

    from realmspinner.kernels.pixel.tiles import TilemapCel

    refs = np.zeros((6, 6), dtype="<u4")
    payload = zlib.compress(refs.tobytes())
    data = _tilemap_file(4, 2, 6, 6, payload, x=-4, y=-4)
    doc, warnings = asein.document_from_aseprite(data)
    assert isinstance(doc.stack[0], TilemapCel)
    assert any("past the canvas" in w for w in warnings)

    # And one tile past that bound, on one axis alone, is refused.
    refs = np.zeros((6, 7), dtype="<u4")
    payload = zlib.compress(refs.tobytes())
    data = _tilemap_file(4, 2, 7, 6, payload, x=-4, y=-4)
    with pytest.raises(ValueError, match="larger than"):
        asein.document_from_aseprite(data)
