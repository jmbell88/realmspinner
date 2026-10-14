"""Regression tests for the 2026-10-14 audit's Plotter engine and mode findings.

One test (or a small family) per finding id, named for the claim. Each was run
against the unfixed tree first and failed there.
"""

from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath

import numpy as np
import pytest
from PIL import Image

from realmspinner.kernels.grid2d import gid
from realmspinner.kernels.grid2d import tileset as tileset_module
from realmspinner.kernels.grid2d.tileset import Tileset
from realmspinner.studio.modes.plotter import fileio as plotter_io
from realmspinner.studio.modes.plotter import mode as plotter_mode
from realmspinner.studio.modes.plotter.engine import (
    _map_layers,
    _map_tilesets,
    rmap,
    tmx,
    tools,
    tsx,
)
from realmspinner.studio.modes.plotter.engine.tilemap import MapDoc

from .test_plotter_mode import FakeCtx, _Done, _grouped, _stamping


def _atlas(size: int = 16) -> np.ndarray:
    array = np.zeros((size, size, 4), np.uint8)
    array[..., 3] = 255
    return array


def _tileset(name: str = "t") -> Tileset:
    return Tileset(name=name, pixels=_atlas(), tile_w=16, tile_h=16)


def _loaders(counter: list[str] | None = None) -> dict:
    def tsx_loader(source: str) -> Tileset:
        if counter is not None:
            counter.append(source)
        return _tileset()

    return {"image_loader": lambda _s: _atlas(), "tsx_loader": tsx_loader}


# --- plotter-02: an infinite layer's dense window is charged ----------------------


def _infinite_tmj(layers: int) -> bytes:
    chunk = lambda x, y: {"x": x, "y": y, "width": 16, "height": 16, "data": [1] * 256}  # noqa: E731
    payload = {
        "type": "map",
        "orientation": "orthogonal",
        "infinite": True,
        "width": 16,
        "height": 16,
        "tilewidth": 16,
        "tileheight": 16,
        "tilesets": [{"firstgid": 1, "image": "a.png", "tilewidth": 16, "tileheight": 16}],
        "layers": [
            {
                "type": "tilelayer",
                "name": f"L{i}",
                "id": i + 1,
                "width": 16,
                "height": 16,
                "chunks": [chunk(0, 0), chunk(4080, 4080)],
            }
            for i in range(layers)
        ],
    }
    return json.dumps(payload).encode()


def test_read_tmj_charges_an_infinite_layers_dense_window_to_the_cell_budget(monkeypatch):
    # Two 16x16 chunks are 512 cells of charge, but the bounding box held is
    # 4096x4096. A ceiling under the window must refuse; before the fix the
    # layer was charged 512 and the 64 MiB array was allocated regardless.
    monkeypatch.setattr(tmx, "MAX_DECODED_CELLS", 4_000_000)
    with pytest.raises(ValueError, match="decode to more than"):
        tmx.read_tmj(_infinite_tmj(1), **_loaders())


def test_read_tmj_sums_the_dense_windows_of_every_infinite_layer(monkeypatch):
    # One window fits the ceiling; the second layer's does not.
    monkeypatch.setattr(tmx, "MAX_DECODED_CELLS", 20_000_000)
    assert tmx.read_tmj(_infinite_tmj(1), **_loaders()).tile_layers()
    with pytest.raises(ValueError, match="decode to more than"):
        tmx.read_tmj(_infinite_tmj(2), **_loaders())


def test_read_tmx_charges_an_infinite_layers_dense_window_to_the_cell_budget(monkeypatch):
    csv = ",".join(["1"] * 256)
    chunk = lambda x, y: (  # noqa: E731
        f'<chunk x="{x}" y="{y}" width="16" height="16">{csv}</chunk>'
    )
    xml = (
        '<map version="1.10" orientation="orthogonal" infinite="1" width="16" height="16" '
        'tilewidth="16" tileheight="16"><tileset firstgid="1" source="t.tsx"/>'
        '<layer id="1" name="L" width="16" height="16"><data encoding="csv">'
        f"{chunk(0, 0)}{chunk(4080, 4080)}</data></layer></map>"
    ).encode()
    monkeypatch.setattr(tmx, "MAX_DECODED_CELLS", 4_000_000)
    with pytest.raises(ValueError, match="decode to more than"):
        tmx.read_tmx(xml, **_loaders())


# --- plotter-11: the ceilings hold at every door --------------------------------


def test_add_tileset_refuses_past_the_ceiling_read_rmap_reopens():
    doc = MapDoc(2, 2, 16, 16)
    shared = _tileset()
    for index in range(_map_tilesets.MAX_TILESETS):
        doc.add_tileset(shared, source=f"{index}.tsx")
    depth = len(doc.history)
    with pytest.raises(ValueError, match="most this build keeps"):
        doc.add_tileset(shared)
    assert len(doc.history) == depth, "a refusal leaves no half-made step"
    assert len(doc.tilesets) == rmap.MAX_TILESETS
    # And what *was* allowed is a state the .rmap reader will take back.
    assert len(rmap.read_rmap(rmap.rmap_bytes(doc)).tilesets) == rmap.MAX_TILESETS


def test_add_layer_refuses_past_the_ceiling_read_rmap_reopens():
    doc = MapDoc(2, 2, 16, 16)
    while len(doc.all_layers()) < _map_layers.MAX_LAYER_NODES:
        doc.add_object_layer("o")
    with pytest.raises(ValueError, match="most this build keeps"):
        doc.add_object_layer("one too many")
    with pytest.raises(ValueError, match="most this build keeps"):
        doc.add_tile_layer("one too many")
    with pytest.raises(ValueError, match="most this build keeps"):
        doc.add_group_layer("one too many")
    with pytest.raises(ValueError, match="most this build keeps"):
        doc.duplicate_layer(doc.layers[0].uid)
    assert len(doc.all_layers()) == rmap.MAX_LAYER_NODES
    assert len(rmap.read_rmap(rmap.rmap_bytes(doc)).all_layers()) == rmap.MAX_LAYER_NODES


def test_the_rmap_ceilings_are_the_editing_doors_numbers():
    assert rmap.MAX_TILESETS == _map_tilesets.MAX_TILESETS
    assert rmap.MAX_LAYER_NODES == _map_layers.MAX_LAYER_NODES


def test_read_tmx_refuses_a_tileset_count_before_calling_the_loader_past_it():
    calls: list[str] = []
    refs = "".join(
        f'<tileset firstgid="{1 + i}" source="same.tsx"/>' for i in range(tmx.MAX_TILESETS + 1)
    )
    xml = (
        '<map version="1.10" orientation="orthogonal" width="1" height="1" '
        f'tilewidth="16" tileheight="16">{refs}</map>'
    ).encode()
    with pytest.raises(ValueError, match="tilesets this build reads"):
        tmx.read_tmx(xml, **_loaders(calls))
    assert len(calls) == tmx.MAX_TILESETS, "the 1001st loader call never happens"


def test_read_tmj_refuses_a_tileset_count_before_calling_the_loader_past_it():
    calls: list[str] = []
    payload = {
        "type": "map",
        "orientation": "orthogonal",
        "width": 1,
        "height": 1,
        "tilewidth": 16,
        "tileheight": 16,
        "tilesets": [
            {"firstgid": 1 + i, "source": "same.tsj"} for i in range(tmx.MAX_TILESETS + 1)
        ],
        "layers": [],
    }
    with pytest.raises(ValueError, match="tilesets this build reads"):
        tmx.read_tmj(json.dumps(payload).encode(), **_loaders(calls))
    assert len(calls) == tmx.MAX_TILESETS


# --- plotter-24: a non-finite extent or vertex is refused at the door ----------


def _object_map(body: str, attrs: str = "") -> bytes:
    return (
        '<map version="1.10" orientation="orthogonal" width="2" height="2" tilewidth="16" '
        f'tileheight="16"{attrs}><objectgroup id="1" name="o">{body}</objectgroup></map>'
    ).encode()


def test_read_tmx_refuses_a_non_finite_object_extent():
    with pytest.raises(ValueError, match="finite"):
        tmx.read_tmx(
            _object_map('<object id="1" x="0" y="0" width="inf" height="nan"/>'), **_loaders()
        )


def test_read_tmx_refuses_a_non_finite_polygon_vertex():
    body = '<object id="1" x="0" y="0"><polygon points="0,0 inf,1 2,2"/></object>'
    with pytest.raises(ValueError, match="finite"):
        tmx.read_tmx(_object_map(body), **_loaders())


def test_read_tmj_refuses_a_non_finite_object_extent_and_vertex():
    def payload(obj: dict) -> bytes:
        return json.dumps(
            {
                "type": "map",
                "orientation": "orthogonal",
                "width": 2,
                "height": 2,
                "tilewidth": 16,
                "tileheight": 16,
                "tilesets": [],
                "layers": [{"type": "objectgroup", "id": 1, "name": "o", "objects": [obj]}],
            }
        ).encode()

    # ``json`` writes float('inf') as the non-standard ``Infinity`` literal and
    # reads it back, which is how a real file carries one.
    with pytest.raises(ValueError, match="finite"):
        tmx.read_tmj(
            payload({"id": 1, "x": 0, "y": 0, "width": float("inf"), "height": 1}), **_loaders()
        )
    with pytest.raises(ValueError, match="finite"):
        tmx.read_tmj(
            payload(
                {
                    "id": 1,
                    "x": 0,
                    "y": 0,
                    "polyline": [{"x": 0, "y": 0}, {"x": float("nan"), "y": 1}],
                }
            ),
            **_loaders(),
        )


def test_a_negative_hexsidelength_is_clamped_by_both_tiled_readers():
    xml = (
        b'<map version="1.10" orientation="hexagonal" width="2" height="2" tilewidth="16" '
        b'tileheight="16" hexsidelength="-9"/>'
    )
    assert tmx.read_tmx(xml, **_loaders()).hex_side == 0
    payload = {
        "type": "map",
        "orientation": "hexagonal",
        "width": 2,
        "height": 2,
        "tilewidth": 16,
        "tileheight": 16,
        "hexsidelength": -9,
        "tilesets": [],
        "layers": [],
    }
    assert tmx.read_tmj(json.dumps(payload).encode(), **_loaders()).hex_side == 0


# --- plotter-25: the image layer cannot spell its way onto the atlas -----------


def test_an_image_layer_source_spelled_dot_slash_tilesets_does_not_overwrite_the_atlas():
    doc = MapDoc(2, 2, 16, 16)
    doc.add_tileset(_tileset("grass"))
    sky = np.zeros((4, 4, 4), np.uint8)
    sky[..., 2] = 255
    sky[..., 3] = 255
    for spelling in ("./tilesets/00-grass.png", "tilesets//00-grass.png", "./map.tmx"):
        fresh = MapDoc(2, 2, 16, 16)
        fresh.add_tileset(_tileset("grass"))
        fresh.add_image_layer("sky", pixels=sky, source=spelling)
        files = tmx.tmx_export(fresh)
        landing: dict[str, list[str]] = {}
        for key in files:
            landing.setdefault(PureWindowsPath(key).as_posix(), []).append(key)
        assert all(len(keys) == 1 for keys in landing.values()), (spelling, landing)
        # The image went under images/, and the atlas bytes are the atlas.
        assert any(key.startswith("images/") for key in files), spelling
        atlas_key = next(
            key for key in files if key.startswith("tilesets/") and key.endswith(".png")
        )
        assert files[atlas_key] != files[next(key for key in files if key.startswith("images/"))]


# --- plotter-26: the writer refuses what this build's reader would -------------


def test_tmx_export_refuses_a_name_the_xml_reader_would_not_read_back():
    doc = MapDoc(2, 2, 16, 16)
    doc.add_tile_layer("bad" + chr(0x0B) + "name")
    with pytest.raises(tsx.TiledUnsupported) as raised:
        tmx.tmx_export(doc)
    assert raised.value.exporting is True
    assert "U+000B" in str(raised.value)


def test_tmx_export_refuses_a_lone_surrogate_in_a_name_too():
    doc = MapDoc(2, 2, 16, 16)
    doc.add_tile_layer("half" + chr(0xD800))
    with pytest.raises(tsx.TiledUnsupported):
        tmx.tmx_export(doc)


def test_tmx_export_still_writes_ordinary_and_non_ascii_names():
    doc = MapDoc(2, 2, 16, 16)
    doc.add_tile_layer("Ground\tlayer é中\U0001f600")
    out = tmx.tmx_export(doc)["map.tmx"]
    assert tmx.read_tmx(out, **_loaders()).layers[0].name == "Ground\tlayer é中\U0001f600"


# --- plotter-28: the dangling-gid warning cannot fire from a reader ------------


def test_a_tile_object_naming_an_unaccounted_gid_is_refused_not_warned():
    body = '<object id="1" gid="99" x="0" y="0" width="16" height="16"/>'
    xml = (
        _object_map(body)
        .replace(b"</map>", b"</map>")
        .replace(b"<objectgroup", b'<tileset firstgid="1" source="t.tsx"/><objectgroup', 1)
    )
    warnings: list = []
    with pytest.raises(ValueError, match="accounts for"):
        tmx.read_tmx(xml, import_warnings=warnings, **_loaders())
    assert warnings == [], "refused outright, so there is nothing left to warn about"


def test_the_dangling_tile_object_docstring_says_finish_refuses_first():
    doc = tmx._warn_dangling_tile_objects.__doc__ or ""
    assert "_finish" in doc and "cannot fire" in doc
    assert "Logged rather than refused" not in doc


# --- plotter-20 / plotter-tiles-03: a non-finite probability -------------------


def test_random_stamp_survives_an_infinite_tile_probability():
    brush = np.array([[1, 2, 3]], gid.DTYPE)
    # Infinity: the normalised vector used to be NaN and ``choice`` raised.
    weights = {1: float("inf"), 2: 1.0, 3: 1.0}
    data = np.zeros((1, 3), gid.DTYPE)
    region = tools.random_stamp(
        data, 0, 0, brush, rng=np.random.default_rng(0), weights=lambda g: weights[g]
    )
    assert region is not None


def test_random_choices_rescales_weights_whose_sum_overflows():
    brush = np.array([[1, 2]], gid.DTYPE)
    _choices, probabilities = tools.random_choices(brush, lambda _g: 1e308)
    assert probabilities is not None
    assert np.isfinite(probabilities).all()
    assert probabilities.sum() == pytest.approx(1.0)
    # And ``choice`` takes it.
    np.random.default_rng(0).choice(2, p=probabilities)


def test_random_choices_leaves_ordinary_weights_exactly_as_they_were():
    brush = np.array([[1, 2, 3]], gid.DTYPE)
    weights = {1: 1.0, 2: 3.0, 3: 0.0}
    _choices, probabilities = tools.random_choices(brush, lambda g: weights[g])
    assert probabilities.tolist() == [0.25, 0.75, 0.0]


def test_a_tsx_with_an_infinite_tile_probability_is_refused_when_read():
    xml = (
        b'<tileset version="1.10" name="t" tilewidth="16" tileheight="16" '
        b'tilecount="1" columns="1">'
        b'<image source="a.png" width="16" height="16"/>'
        b'<tile id="0" probability="inf"/></tileset>'
    )
    with pytest.raises(ValueError, match="finite"):
        tsx.read_tsx(xml, _atlas())


# --- plotter-mode-08: an image collection decodes under one budget ------------


def _collection_tsx(directory: Path, names: list[str]) -> Path:
    tiles = "".join(
        f'<tile id="{i}"><image source="{name}" width="16" height="16"/></tile>'
        for i, name in enumerate(names)
    )
    path = directory / "t.tsx"
    path.write_text(
        '<?xml version="1.0"?><tileset version="1.10" name="many" tilewidth="16" tileheight="16" '
        f'tilecount="{len(names)}" columns="0">'
        '<grid orientation="orthogonal" width="1" height="1"/>'
        f"{tiles}</tileset>",
        encoding="utf-8",
    )
    return path


def _counting_decode(monkeypatch) -> list[Path]:
    calls: list[Path] = []
    real = plotter_io._decode

    def counting(path: Path) -> np.ndarray:
        calls.append(Path(path))
        return real(path)

    monkeypatch.setattr(plotter_io, "_decode", counting)
    return calls


def test_an_image_collection_tsx_is_refused_once_its_decoded_pixels_pass_one_budget(
    tmp_path, monkeypatch
):
    names = [f"tile{i}.png" for i in range(10)]
    for name in names:
        Image.fromarray(_atlas()).save(tmp_path / name)
    _collection_tsx(tmp_path, names)
    calls = _counting_decode(monkeypatch)
    # Three 16x16 tiles' worth of budget: the fourth decode crosses it.
    monkeypatch.setattr(tileset_module, "MAX_COLLECTION_PIXELS", 3 * 16 * 16)

    loader = plotter_io._loaders(tmp_path)["tsx_loader"]
    with pytest.raises(ValueError, match="pixels this build will hold"):
        loader("t.tsx")
    assert len(calls) == 4, "refused before the fifth decode, not after all ten"


def test_an_image_collection_decodes_a_repeated_file_once(tmp_path, monkeypatch):
    Image.fromarray(_atlas()).save(tmp_path / "a.png")
    _collection_tsx(tmp_path, ["a.png"] * 50)
    calls = _counting_decode(monkeypatch)
    loaded = plotter_io._loaders(tmp_path)["tsx_loader"]("t.tsx")
    assert loaded.tile_count == 50
    assert len(calls) == 1


def test_an_image_collection_with_too_many_entries_is_refused_before_any_decode(
    tmp_path, monkeypatch
):
    sources = {i: "a.png" for i in range(plotter_io.MAX_COLLECTION_ENTRIES + 1)}
    calls = _counting_decode(monkeypatch)
    with pytest.raises(ValueError, match="tile images"):
        plotter_io._decode_collection(tmp_path, sources)
    assert calls == []


# --- plotter-mode-09: a failed open is forgotten -------------------------------


def test_a_map_that_fails_to_open_is_forgotten_from_the_recent_list(tmp_path):
    ctx = FakeCtx(accept=False)
    path = tmp_path / "gone.rmap"
    path.write_bytes(b"")
    plotter_mode.remember_path(ctx, path)
    assert plotter_mode.recent_paths(ctx), "precondition: it is on the list"

    plotter_mode.open_path(ctx, path)
    key = ctx.submitted[-1]
    assert key.startswith("plotter-open:") and str(path) in key, "the path is in the key"

    plotter_mode.on_task_failed(ctx, _Done(key))
    assert plotter_mode.recent_paths(ctx) == []


def test_a_failed_edit_asset_open_forgets_nothing_it_does_not_hold(tmp_path):
    ctx = FakeCtx()
    path = tmp_path / "keep.rmap"
    path.write_bytes(b"")
    plotter_mode.remember_path(ctx, path)
    plotter_mode.on_task_failed(ctx, _Done("plotter-open:some-job-id"))
    assert plotter_mode.recent_paths(ctx), "a job id names no recents row"


# --- plotter-mode-07: the Unlock offered for an inherited lock works ------------


def test_unlock_on_a_group_inherited_lock_unlocks_the_group_or_names_it():
    ctx = FakeCtx()
    tab, state, layer = _stamping(ctx)
    group = _grouped(tab.doc, layer)
    ctx.state.plotter = state
    assert not layer.locked and plotter_mode.layer_locked(tab.doc, layer)

    state.select = (0, 0, 1, 1)
    plotter_mode._delete(ctx, state, tab)

    message, _kind, action, arg = ctx.toasts[-1]
    assert group.name in message, "the toast names the group that holds the lock"
    assert action == "unlock" and arg == str(group.uid)

    plotter_mode.unlock_layer(ctx, arg)
    assert not tab.doc.layer(group.uid).locked
    assert not plotter_mode.layer_locked(tab.doc, layer), "the button did what it said"


def test_the_leafs_own_lock_still_reads_as_its_own():
    ctx = FakeCtx()
    tab, state, layer = _stamping(ctx)
    tab.doc.set_layer_props(layer.uid, locked=True)
    ctx.state.plotter = state
    state.select = (0, 0, 1, 1)
    plotter_mode._delete(ctx, state, tab)
    message, _kind, action, arg = ctx.toasts[-1]
    assert message == f"{layer.name} is locked."
    assert arg == str(layer.uid)
