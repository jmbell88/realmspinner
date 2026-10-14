"""The 2026-10-14 audit, fixer F40: the Plotter map engine's remaining defects.

One file for ten findings (plotter-map-02, -03, -04, -05, -07, -08, -09, -12,
-13, -17). Each test's name is the claim it makes and each was run against the
unfixed code first.
"""

from __future__ import annotations

import gc
import io
import json
import tracemalloc
import zipfile

import numpy as np
import pytest
from PIL import Image

from realmspinner.kernels.grid2d.tileset import Tileset
from realmspinner.kernels.grid2d.wang import WangColour, WangSet
from realmspinner.studio.modes.plotter.engine import _map_tilesets, project, render, rmap, tmx, tsx
from realmspinner.studio.modes.plotter.engine.tilemap import MapDoc

_NO_LOADERS = {"image_loader": lambda _s: None, "tsx_loader": lambda _s: None}


def _png(side: int = 16, seed: int = 0) -> bytes:
    pixels = np.zeros((side, side, 4), np.uint8)
    pixels[0, 0, 0] = seed % 256
    pixels[0, 0, 1] = seed // 256
    pixels[..., 3] = 255
    out = io.BytesIO()
    Image.fromarray(pixels, "RGBA").save(out, "PNG")
    return out.getvalue()


def _archive(manifest: dict, members: dict[str, bytes] | None = None) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("map.json", json.dumps(manifest))
        for name, data in (members or {}).items():
            zf.writestr(name, data, zipfile.ZIP_STORED)
    return out.getvalue()


def _manifest(**over) -> dict:
    base = {
        "version": 3,
        "width": 1,
        "height": 1,
        "tile_w": 16,
        "tile_h": 16,
        "layers": [],
        "tilesets": [],
    }
    base.update(over)
    return base


def _tileset_entry(index: int, firstgid: int, **over) -> dict:
    entry = {
        "name": f"t{index}",
        "firstgid": firstgid,
        "image": f"tilesets/{index}.png",
        "tile_w": 16,
        "tile_h": 16,
    }
    entry.update(over)
    return entry


def _solid(side: int = 32, value: int = 200) -> np.ndarray:
    pixels = np.zeros((side, side, 4), np.uint8)
    pixels[..., 0] = value
    pixels[..., 3] = 255
    return pixels


# --- plotter-27 (plotter-map-17) ---------------------------------------------


def test_duplicate_layer_copies_each_tile_array_once_whatever_the_nesting() -> None:
    peaks: dict[int, int] = {}
    array_bytes = 0
    for depth in (1, 8):
        doc = MapDoc(512, 512, 8, 8)
        parent = None
        for _ in range(depth):
            parent = doc.add_group_layer("g", parent_uid=parent).uid
        leaf = doc.add_tile_layer("leaf", parent_uid=parent)
        array_bytes = int(np.asarray(leaf.data).nbytes)
        top = doc.layers[0].uid
        gc.collect()
        tracemalloc.start()
        try:
            doc.duplicate_layer(top)
            _, peaks[depth] = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
    # One copy of the leaf is the whole bill; the unfixed recursion copied it
    # once per level, so depth 8 peaked near nine times the array.
    assert peaks[8] < 2.5 * array_bytes, peaks


def test_duplicate_layer_still_gives_the_copy_fresh_identity_and_its_own_arrays() -> None:
    doc = MapDoc(4, 4, 8, 8)
    group = doc.add_group_layer("g")
    inner = doc.add_group_layer("inner", parent_uid=group.uid)
    leaf = doc.add_tile_layer("leaf", parent_uid=inner.uid)
    copy = doc.duplicate_layer(group.uid)
    copied_leaf = copy.children[0].children[0]
    assert copy.uid != group.uid and copied_leaf.uid != leaf.uid
    assert copied_leaf.id != leaf.id
    assert copied_leaf.data is not leaf.data
    assert copy.children is not group.children
    assert len(group.children) == 1 and group.children[0] is inner


# --- plotter-05 (plotter-map-04) ---------------------------------------------


def test_set_map_settings_refuses_a_skew_that_collapses_the_oblique_grid() -> None:
    doc = MapDoc(8, 8, 32, 32, projection=project.OBLIQUE)
    with pytest.raises(ValueError, match="skew"):
        doc.set_map_settings(skew_x=32, skew_y=32)  # 32*32 - 32*32 == 0
    assert (doc.skew_x, doc.skew_y) == (0, 0)
    doc.cell_at(10, 10)  # the canvas hit test still answers


def test_set_tile_size_refuses_a_size_that_collapses_an_oblique_skew() -> None:
    doc = MapDoc(8, 8, 16, 16, projection=project.OBLIQUE)
    doc.set_map_settings(skew_x=32, skew_y=32)
    with pytest.raises(ValueError, match="skew"):
        doc.set_tile_size(32, 32)
    assert (doc.tile_w, doc.tile_h) == (16, 16)


def test_the_readers_refuse_a_skew_that_collapses_the_oblique_grid() -> None:
    xml = (
        b'<map version="1.10" orientation="oblique" width="8" height="8"'
        b' tilewidth="32" tileheight="32" skewx="32" skewy="32"/>'
    )
    with pytest.raises(ValueError, match="skew"):
        tmx.read_tmx(xml, **_NO_LOADERS)
    payload = {
        "type": "map",
        "orientation": "oblique",
        "width": 8,
        "height": 8,
        "tilewidth": 32,
        "tileheight": 32,
        "skewx": 32,
        "skewy": 32,
        "layers": [],
    }
    with pytest.raises(ValueError, match="skew"):
        tmx.read_tmj(json.dumps(payload).encode(), **_NO_LOADERS)
    manifest = _manifest(projection="oblique", tile_w=32, tile_h=32, skew=[32, 32], version=4)
    with pytest.raises(ValueError, match="skew"):
        rmap.read_rmap(_archive(manifest))


# --- plotter-23 (plotter-map-13) ---------------------------------------------


def test_render_map_clips_a_layer_offset_far_off_the_canvas() -> None:
    doc = MapDoc(2, 2, 16, 16)
    doc.add_tileset(Tileset(name="t", pixels=_solid(16), tile_w=16, tile_h=16))
    layer = doc.add_tile_layer("ground")
    doc.write_region(layer.uid, 0, 0, np.ones((2, 2), np.uint32))
    doc.set_layer_props(layer.uid, offset_x=1e30)
    # Was ``OverflowError`` out of the native blit's ``int64`` conversion.
    image = render.render_map(doc)
    assert not image[..., 3].any()  # the layer is wholly off the canvas
    doc.set_layer_props(layer.uid, offset_x=-1e30, offset_y=1e30)
    assert not render.render_map(doc)[..., 3].any()


# --- plotter-03 (plotter-map-02) ---------------------------------------------


def test_read_rmap_refuses_a_manifest_whose_distinct_pictures_decode_past_a_document_budget(
    monkeypatch,
) -> None:
    sides = 32
    count = 4
    monkeypatch.setattr(
        _map_tilesets, "MAX_DOCUMENT_PIXELS", sides * sides * (count - 1), raising=False
    )
    manifest = _manifest(
        tilesets=[_tileset_entry(i, 1 + i * 4) for i in range(count)],
    )
    members = {f"tilesets/{i}.png": _png(sides, seed=i) for i in range(count)}
    with pytest.raises(ValueError, match="pixels"):
        rmap.read_rmap(_archive(manifest, members))


def test_read_rmap_charges_a_shared_picture_once_not_per_reference(monkeypatch) -> None:
    monkeypatch.setattr(_map_tilesets, "MAX_DOCUMENT_PIXELS", 32 * 32 * 2, raising=False)
    manifest = _manifest(
        tilesets=[_tileset_entry(0, 1 + i * 4, image="tilesets/0.png") for i in range(6)],
    )
    doc = rmap.read_rmap(_archive(manifest, {"tilesets/0.png": _png(32)}))
    assert len(doc.tilesets) == 6


def test_the_tiled_readers_refuse_tilesets_whose_images_decode_past_a_document_budget(
    monkeypatch,
) -> None:
    monkeypatch.setattr(_map_tilesets, "MAX_DOCUMENT_PIXELS", 32 * 32 * 2, raising=False)
    calls = {"n": 0}

    def image_loader(_source: str) -> np.ndarray:
        calls["n"] += 1
        return _solid(32, value=calls["n"])

    sets = "".join(
        f'<tileset firstgid="{1 + i * 4}" name="t{i}" tilewidth="16" tileheight="16"'
        f' tilecount="4" columns="2"><image source="i{i}.png" width="32" height="32"/></tileset>'
        for i in range(4)
    )
    xml = (
        '<map version="1.10" orientation="orthogonal" width="1" height="1"'
        f' tilewidth="16" tileheight="16">{sets}</map>'
    ).encode()
    with pytest.raises(ValueError, match="pixels"):
        tmx.read_tmx(xml, image_loader=image_loader, tsx_loader=lambda _s: None)

    payload = {
        "type": "map",
        "orientation": "orthogonal",
        "width": 1,
        "height": 1,
        "tilewidth": 16,
        "tileheight": 16,
        "layers": [],
        "tilesets": [
            {
                "firstgid": 1 + i * 4,
                "name": f"t{i}",
                "image": f"i{i}.png",
                "imagewidth": 32,
                "imageheight": 32,
                "tilewidth": 16,
                "tileheight": 16,
                "tilecount": 4,
                "columns": 2,
            }
            for i in range(4)
        ],
    }
    calls["n"] = 0
    with pytest.raises(ValueError, match="pixels"):
        tmx.read_tmj(
            json.dumps(payload).encode(),
            image_loader=image_loader,
            tsx_loader=lambda _s: None,
        )


# --- plotter-04 (plotter-map-03) ---------------------------------------------


def test_read_rmap_refuses_a_manifest_member_past_its_own_ceiling(monkeypatch) -> None:
    monkeypatch.setattr(rmap, "MAX_MANIFEST_BYTES", 4096, raising=False)
    head = json.dumps(_manifest())[:-1]
    padded = (head + ',"junk":[' + "[]," * 4000 + "[]]}").encode()
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("map.json", padded)
    with pytest.raises(ValueError, match="manifest"):
        rmap.read_rmap(out.getvalue())


def test_a_manifest_under_its_ceiling_still_opens() -> None:
    doc = MapDoc(4, 4, 16, 16)
    doc.add_tile_layer("ground")
    assert len(rmap.read_rmap(rmap.rmap_bytes(doc)).layers) == 1


# --- plotter-12 (plotter-map-07) ---------------------------------------------


def test_rmap_layer_arrays_and_manifest_are_deflated() -> None:
    doc = MapDoc(512, 512, 16, 16)
    doc.add_tileset(Tileset(name="t", pixels=_solid(16), tile_w=16, tile_h=16))
    doc.add_tile_layer("ground")
    data = rmap.rmap_bytes(doc)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        kinds = {info.filename: info.compress_type for info in zf.infolist()}
    assert kinds["map.json"] == zipfile.ZIP_DEFLATED
    assert kinds["layers/0.npy"] == zipfile.ZIP_DEFLATED
    assert kinds["tilesets/0.png"] == zipfile.ZIP_STORED  # a PNG is already compressed
    assert len(data) < 200_000  # an empty 512x512 layer was ~1 MB stored
    assert data == rmap.rmap_bytes(doc)  # fixed timestamps: still deterministic
    assert rmap.read_rmap(data).width == 512


# --- plotter-13 (plotter-map-08) ---------------------------------------------


def test_rmap_round_trip_keeps_a_wang_sets_and_colours_tile_and_class() -> None:
    wangset = WangSet(
        name="W",
        kind="corner",
        colours=(WangColour("grass", "#00ff00", 1.0, tile=1, klass="Meadow"),),
        tiles={0: (0, 1, 0, 1, 0, 1, 0, 1)},
        tile=1,
        klass="Biome",
    )
    doc = MapDoc(1, 1, 16, 16)
    doc.add_tileset(
        Tileset(
            name="t",
            pixels=np.full((16, 32, 4), 255, np.uint8),
            tile_w=16,
            tile_h=16,
            wangsets=(wangset,),
        )
    )
    data = rmap.rmap_bytes(doc)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        declared = json.loads(zf.read("map.json"))["version"]
    back = rmap.read_rmap(data).tilesets[0].tileset.wangsets[0]
    assert (back.tile, back.klass) == (1, "Biome")
    assert (back.colours[0].tile, back.colours[0].klass) == (1, "Meadow")
    # A reader that predates the two fields would drop them in silence, so the
    # gate has to name them: the version is past the one that dropped them.
    assert declared > rmap.STAMPS_VERSION


def test_a_wang_set_with_no_tile_or_class_keeps_its_old_version() -> None:
    wangset = WangSet(
        name="W",
        kind="corner",
        colours=(WangColour("grass", "#00ff00"),),
        tiles={0: (0, 1, 0, 1, 0, 1, 0, 1)},
    )
    doc = MapDoc(1, 1, 16, 16)
    doc.add_tileset(
        Tileset(
            name="t",
            pixels=np.full((16, 32, 4), 255, np.uint8),
            tile_w=16,
            tile_h=16,
            wangsets=(wangset,),
        )
    )
    assert json.loads(rmap.manifest_json(doc))["version"] == rmap.WANGSET_VERSION


# --- plotter-14 (plotter-map-09) ---------------------------------------------


def _overlap_loader(source: str) -> Tileset:
    return Tileset(name=source, pixels=_solid(32), tile_w=16, tile_h=16)  # 4 tiles


def test_read_tmx_refuses_tilesets_whose_gid_ranges_overlap() -> None:
    xml = (
        b'<map version="1.10" orientation="orthogonal" width="1" height="1"'
        b' tilewidth="16" tileheight="16">'
        b'<tileset firstgid="1" source="a.tsx"/><tileset firstgid="3" source="b.tsx"/></map>'
    )
    with pytest.raises(ValueError, match="overlap"):
        tmx.read_tmx(xml, image_loader=lambda _s: None, tsx_loader=_overlap_loader)


def test_read_tmj_refuses_tilesets_whose_gid_ranges_overlap() -> None:
    payload = {
        "type": "map",
        "orientation": "orthogonal",
        "width": 1,
        "height": 1,
        "tilewidth": 16,
        "tileheight": 16,
        "layers": [],
        "tilesets": [
            {"firstgid": 1, "source": "a.tsx"},
            {"firstgid": 3, "source": "b.tsx"},
        ],
    }
    with pytest.raises(ValueError, match="overlap"):
        tmx.read_tmj(
            json.dumps(payload).encode(),
            image_loader=lambda _s: None,
            tsx_loader=_overlap_loader,
        )


def test_read_rmap_refuses_tilesets_whose_gid_ranges_overlap() -> None:
    # 32x32 at 16px is four tiles, so a second set at gid 3 overlaps the first.
    manifest = _manifest(
        tilesets=[
            _tileset_entry(0, 1, image="tilesets/0.png"),
            _tileset_entry(1, 3, image="tilesets/0.png"),
        ]
    )
    with pytest.raises(ValueError, match="overlap"):
        rmap.read_rmap(_archive(manifest, {"tilesets/0.png": _png(32)}))


def test_contiguous_tilesets_still_open_in_every_reader() -> None:
    xml = (
        b'<map version="1.10" orientation="orthogonal" width="1" height="1"'
        b' tilewidth="16" tileheight="16">'
        b'<tileset firstgid="1" source="a.tsx"/><tileset firstgid="5" source="b.tsx"/></map>'
    )
    doc = tmx.read_tmx(xml, image_loader=lambda _s: None, tsx_loader=_overlap_loader)
    assert [ref.firstgid for ref in doc.tilesets] == [1, 5]


# --- plotter-22 (plotter-map-12) ---------------------------------------------


_SHORT_ARRAYS = [
    ("a tileset offset of one number", {"offset": [5]}),
    ("an animation pair of one number", {"tiles": {"0": {"animation": [[1]]}}}),
    (
        "a collision point of no numbers",
        {"tiles": {"0": {"collision": [{"kind": "polygon", "points": [[]]}]}}},
    ),
    (
        "a collection cell of one number",
        {"collection": {"ids": [0], "sizes": [[16, 16]], "columns": 1, "cell": [5]}},
    ),
]


@pytest.mark.parametrize("patch", [p for _, p in _SHORT_ARRAYS], ids=[n for n, _ in _SHORT_ARRAYS])
def test_read_rmap_frames_a_short_array_as_a_value_error(patch: dict) -> None:
    entry = _tileset_entry(0, 1, **patch)
    manifest = _manifest(version=9, tilesets=[entry])
    with pytest.raises(ValueError):
        rmap.read_rmap(_archive(manifest, {"tilesets/0.png": _png(16)}))


def test_read_tmj_frames_a_collision_point_with_no_x_as_a_value_error() -> None:
    payload = {
        "type": "map",
        "orientation": "orthogonal",
        "width": 1,
        "height": 1,
        "tilewidth": 16,
        "tileheight": 16,
        "layers": [],
        "tilesets": [
            {
                "firstgid": 1,
                "image": "a.png",
                "tilewidth": 16,
                "tileheight": 16,
                "tiles": [{"id": 0, "objectgroup": {"objects": [{"polygon": [{"y": 1}]}]}}],
            }
        ],
    }
    with pytest.raises(ValueError):
        tmx.read_tmj(
            json.dumps(payload).encode(),
            image_loader=lambda _s: np.full((16, 16, 4), 255, np.uint8),
            tsx_loader=lambda _s: None,
        )


def test_read_tsj_frames_a_collision_point_with_no_x_as_a_value_error() -> None:
    entry = {
        "name": "t",
        "image": "a.png",
        "tilewidth": 16,
        "tileheight": 16,
        "tiles": [{"id": 0, "objectgroup": {"objects": [{"polygon": [{"y": 1}]}]}}],
    }
    with pytest.raises(ValueError):
        tsx.read_tsj(json.dumps(entry).encode(), np.full((16, 16, 4), 255, np.uint8))


# --- plotter-06 (plotter-map-05) ---------------------------------------------


def test_a_non_finite_parallax_origin_is_refused_at_every_door_that_would_store_it() -> None:
    doc = MapDoc(2, 2, 16, 16)
    with pytest.raises(ValueError, match="parallax"):
        doc.set_map_settings(parallax_origin=(float("nan"), 0.0))
    with pytest.raises(ValueError, match="parallax"):
        doc.set_map_settings(parallax_origin=(0.0, float("inf")))
    assert tuple(doc.parallax_origin) == (0.0, 0.0)

    xml = (
        b'<map version="1.10" orientation="orthogonal" width="2" height="2"'
        b' tilewidth="16" tileheight="16" parallaxoriginx="inf" parallaxoriginy="nan"/>'
    )
    with pytest.raises(ValueError, match="parallax"):
        tmx.read_tmx(xml, **_NO_LOADERS)
    payload = json.dumps(
        {
            "type": "map",
            "orientation": "orthogonal",
            "width": 2,
            "height": 2,
            "tilewidth": 16,
            "tileheight": 16,
            "parallaxoriginx": float("inf"),
            "layers": [],
        }
    ).encode()
    with pytest.raises(ValueError, match="parallax"):
        tmx.read_tmj(payload, **_NO_LOADERS)


def test_read_rmap_names_the_map_not_a_layer_for_a_non_finite_parallax_origin() -> None:
    manifest = _manifest(version=4, parallax_origin=[float("inf"), 0.0])
    with pytest.raises(ValueError) as caught:
        rmap.read_rmap(_archive(manifest))
    assert "layer" not in str(caught.value)
    assert "parallax" in str(caught.value)
