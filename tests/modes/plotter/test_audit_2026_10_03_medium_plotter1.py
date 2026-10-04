"""Closing plotter-08 through plotter-12 of the 2026-10-03 audit.

One test per claim, named for it. Each failed against the unfixed code before
its fix landed (the fixer's return carries the pasted failures).
"""

from __future__ import annotations

import base64
import gzip
import io
import json
import math
import zipfile

import numpy as np
import pytest

from realmspinner.kernels.grid2d import gid as gidlib
from realmspinner.kernels.grid2d.tileset import Tileset
from realmspinner.studio.modes.plotter.engine import rmap, tmx
from realmspinner.studio.modes.plotter.engine._map_layers import MAX_GROUP_DEPTH
from realmspinner.studio.modes.plotter.engine.tilemap import MapDoc, MapObject


def _pixels(w: int = 8, h: int = 8) -> np.ndarray:
    array = np.zeros((h, w, 4), dtype=np.uint8)
    array[..., 3] = 255
    return array


def _image_loader(_path: str) -> np.ndarray:
    return _pixels(16, 16)


def _tsx_loader(_path: str) -> Tileset:
    return Tileset(name="t", pixels=_pixels(16, 16), tile_w=16, tile_h=16)


def _read_tmx(xml: bytes) -> MapDoc:
    return tmx.read_tmx(xml, image_loader=_image_loader, tsx_loader=_tsx_loader)


def _read_tmj(payload: dict) -> MapDoc:
    return tmx.read_tmj(
        json.dumps(payload).encode(), image_loader=_image_loader, tsx_loader=_tsx_loader
    )


def _tmj_map(**extra) -> dict:
    return {
        "type": "map",
        "orientation": "orthogonal",
        "width": 1,
        "height": 1,
        "tilewidth": 16,
        "tileheight": 16,
        "layers": [],
        **extra,
    }


def _rewrite(doc: MapDoc, mutate) -> bytes:
    original = rmap.rmap_bytes(doc)
    with zipfile.ZipFile(io.BytesIO(original)) as zf:
        members = {name: zf.read(name) for name in zf.namelist()}
    manifest = json.loads(members[rmap.MANIFEST])
    mutate(manifest)
    members[rmap.MANIFEST] = json.dumps(manifest).encode()
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as zf:
        for name, blob in members.items():
            zf.writestr(name, blob)
    return out.getvalue()


def _doc() -> MapDoc:
    doc = MapDoc(4, 4, 8, 8)
    doc.add_tileset(Tileset(name="t", pixels=_pixels(), tile_w=8, tile_h=8))
    return doc


def _infinite() -> MapDoc:
    doc = MapDoc(8, 8, 8, 8)
    doc.add_tile_layer("Ground")
    doc.set_infinite(True)
    return doc


# --- plotter-08 -----------------------------------------------------------------


def test_an_infinite_whole_map_offset_moves_true_coordinates_the_way_the_layer_scope_does():
    """True = stored + origin, so a content slide by +dx has to *raise* the origin
    by dx. The whole-map branch lowered it, which moved the exported chunk and
    object coordinates the opposite way from the layer scope on the same form."""

    whole = _infinite()
    layer = whole.tile_layers()[0]
    whole.write_region(layer.uid, 0, 0, np.array([[5]], gidlib.DTYPE))

    def true_cell(doc: MapDoc) -> tuple[int, int]:
        data = np.asarray(doc.tile_layers()[0].data)
        ys, xs = np.nonzero(data)
        return int(xs[0]) + doc.origin_x, int(ys[0]) + doc.origin_y

    before = true_cell(whole)
    assert whole.offset(3, 2, wrap=False) is True
    after = true_cell(whole)
    assert (after[0] - before[0], after[1] - before[1]) == (3, 2)

    # The finite map's offset is the reference: +3,+2 moves content +3,+2.
    finite = MapDoc(8, 8, 8, 8)
    finite.add_tile_layer("Ground")
    flayer = finite.tile_layers()[0]
    finite.write_region(flayer.uid, 0, 0, np.array([[5]], gidlib.DTYPE))
    finite.offset(3, 2, wrap=False)
    ys, xs = np.nonzero(np.asarray(finite.tile_layers()[0].data))
    assert (int(xs[0]), int(ys[0])) == (3, 2)


# --- plotter-09 -----------------------------------------------------------------


def _tmx_with_layer_data(encoding: str, compression: str, text: str) -> bytes:
    return (
        '<map version="1.10" orientation="orthogonal" width="1" height="1" '
        'tilewidth="16" tileheight="16">'
        '<layer id="1" name="L" width="1" height="1">'
        f'<data encoding="{encoding}" compression="{compression}">{text}</data>'
        "</layer></map>"
    ).encode()


@pytest.mark.parametrize(
    "compression, payload",
    [
        ("zlib", b"this is not a zlib stream"),
        ("gzip", b"this is not a gzip stream"),
        # truncated -> EOFError; mtime=0 because the test id is built from these bytes
        ("gzip", gzip.compress(b"\x00" * 4, mtime=0)[:12]),
    ],
)
def test_a_corrupt_compressed_layer_or_non_finite_property_value_is_refused_as_a_value_error(
    compression, payload
):
    """A corrupt zlib or gzip layer payload left ``_decompress`` as zlib.error,
    BadGzipFile or EOFError, none of which the open doors frame as a refusal."""

    text = base64.b64encode(payload).decode()
    with pytest.raises(ValueError):
        _read_tmx(_tmx_with_layer_data("base64", compression, text))
    # And the same bytes through the JSON spelling.
    layer = {
        "type": "tilelayer",
        "name": "L",
        "width": 1,
        "height": 1,
        "compression": compression,
        "data": text,
    }
    with pytest.raises(ValueError):
        _read_tmj(_tmj_map(layers=[layer]))


@pytest.mark.parametrize("kind", ["int", "object"])
def test_a_tmx_int_or_object_property_of_infinity_is_refused_as_a_value_error(kind):
    xml = (
        '<map version="1.10" orientation="orthogonal" width="1" height="1" '
        'tilewidth="16" tileheight="16"><properties>'
        f'<property name="p" type="{kind}" value="1e999"/>'
        "</properties></map>"
    ).encode()
    with pytest.raises(ValueError):
        _read_tmx(xml)


@pytest.mark.parametrize("kind", ["int", "object"])
def test_a_tmj_int_or_object_property_of_infinity_is_refused_as_a_value_error(kind):
    payload = _tmj_map(properties=[{"name": "p", "type": kind, "value": float("inf")}])
    with pytest.raises(ValueError):
        _read_tmj(payload)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.__setitem__("properties", [5]),
        lambda p: p.__setitem__("properties", {"a": 1}),
        lambda p: p.__setitem__("width", [1]),
        lambda p: p.__setitem__("tilewidth", {"x": 1}),
    ],
)
def test_a_malformed_tmj_map_header_or_properties_block_is_refused_as_a_value_error(mutate):
    payload = _tmj_map()
    mutate(payload)
    with pytest.raises(ValueError):
        _read_tmj(payload)


def test_an_rmap_tilesets_entry_that_is_not_an_object_is_refused_as_a_value_error():
    data = _rewrite(_doc(), lambda m: m.__setitem__("tilesets", [5]))
    with pytest.raises(ValueError):
        rmap.read_rmap(data)


def test_an_rmap_object_property_of_infinity_is_refused_as_a_value_error():
    doc = _doc()
    doc.add_object_layer("Objs")

    def mutate(manifest: dict) -> None:
        manifest["properties"] = {"p": {"type": "object", "value": float("inf")}}

    with pytest.raises(ValueError):
        rmap.read_rmap(_rewrite(doc, mutate))


# --- plotter-10 -----------------------------------------------------------------


def _nested_groups(depth: int) -> list[dict]:
    """``depth`` groups, each inside the last, as Tiled JSON layer entries."""
    entry: dict = {"type": "group", "name": f"g{depth}", "layers": []}
    for level in range(depth - 1, 0, -1):
        entry = {"type": "group", "name": f"g{level}", "layers": [entry]}
    return [entry]


def test_a_tmj_group_tree_deeper_than_the_layer_depth_cap_is_refused_at_the_door():
    """The editor refuses a layer nested past ``MAX_GROUP_DEPTH`` groups; the
    readers did not, so a foreign file opened and then Duplicate on its outer
    group raised from the layers pane."""

    # The editor's own limit: the innermost layer may sit MAX_GROUP_DEPTH deep.
    ok = _read_tmj(_tmj_map(layers=_nested_groups(MAX_GROUP_DEPTH + 1)))
    assert len(ok.all_layers()) == MAX_GROUP_DEPTH + 1
    with pytest.raises(ValueError, match="deep"):
        _read_tmj(_tmj_map(layers=_nested_groups(MAX_GROUP_DEPTH + 2)))
    with pytest.raises(ValueError, match="deep"):
        _read_tmj(_tmj_map(layers=_nested_groups(100)))


def test_a_tmx_group_tree_deeper_than_the_layer_depth_cap_is_refused_at_the_door():
    inner = ""
    for level in range(MAX_GROUP_DEPTH + 2, 0, -1):
        inner = f'<group id="{level}" name="g{level}">{inner}</group>'
    xml = (
        '<map version="1.10" orientation="orthogonal" width="1" height="1" '
        f'tilewidth="16" tileheight="16">{inner}</map>'
    ).encode()
    with pytest.raises(ValueError, match="deep"):
        _read_tmx(xml)


def test_an_rmap_group_tree_deeper_than_the_layer_depth_cap_is_refused_at_the_door():
    def mutate(manifest: dict) -> None:
        manifest["layers"] = _nested_groups(MAX_GROUP_DEPTH + 2)
        for entry in _walk_entries(manifest["layers"]):
            entry.setdefault("id", 0)

    with pytest.raises(ValueError, match="deep"):
        rmap.read_rmap(_rewrite(_doc(), mutate))


def _walk_entries(entries):
    for entry in entries:
        yield entry
        yield from _walk_entries(entry.get("layers", []))


# --- plotter-11 -----------------------------------------------------------------


def test_set_layer_props_refuses_a_bad_object_layer_colour_before_it_pushes_a_step():
    doc = _doc()
    layer = doc.add_object_layer("Objs")
    head, dirty = doc.history.head, doc.dirty
    with pytest.raises(ValueError):
        doc.set_layer_props(layer.uid, color="purple")
    assert doc.history.head == head, "a step describing a change never made was pushed"
    assert doc.dirty == dirty
    assert doc.layer(layer.uid).color is None


def test_set_object_refuses_a_coordinate_that_is_not_a_number_before_it_pushes_a_step():
    doc = _doc()
    layer = doc.add_object_layer("Objs")
    obj = doc.add_object(layer.uid, MapObject(uid=0, name="o", x=1.0, y=2.0))
    head = doc.history.head
    with pytest.raises(ValueError):
        doc.set_object(layer.uid, obj.uid, x="abc")
    assert doc.history.head == head
    assert (obj.x, obj.y) == (1.0, 2.0)


# --- plotter-12 -----------------------------------------------------------------


@pytest.mark.parametrize("name", ["offset_x", "offset_y", "parallax_x", "parallax_y"])
@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
def test_set_layer_props_refuses_a_non_finite_offset_or_parallax(name, bad):
    doc = _doc()
    layer = doc.add_tile_layer("L")
    head = doc.history.head
    with pytest.raises(ValueError, match="finite"):
        doc.set_layer_props(layer.uid, **{name: bad})
    assert doc.history.head == head
    # What the setter holds, the reader holds: the saved file reopens.
    rmap.read_rmap(rmap.rmap_bytes(doc))


@pytest.mark.parametrize("name", ["x", "y"])
@pytest.mark.parametrize("bad", [math.inf, math.nan])
def test_set_object_refuses_a_non_finite_coordinate(name, bad):
    doc = _doc()
    layer = doc.add_object_layer("Objs")
    obj = doc.add_object(layer.uid, MapObject(uid=0, name="o", x=1.0, y=2.0))
    head = doc.history.head
    with pytest.raises(ValueError, match="finite"):
        doc.set_object(layer.uid, obj.uid, **{name: bad})
    assert doc.history.head == head
    rmap.read_rmap(rmap.rmap_bytes(doc))
