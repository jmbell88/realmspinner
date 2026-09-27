"""Closing ten findings from the 2026-09-26 audit of Plotter.

One test per finding, named for the claim it proves. Every one of these
failed against the unfixed code before its matching fix landed; see the
fixer's own return for the pasted failing output, since a probe against
pre-fix source (loaded as a throwaway module, never checked out over the
tree -- ``dev/INVARIANTS.md``'s rule) is not itself a regression test.
"""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest

from realmspinner.kernels.grid2d import gid as gidlib
from realmspinner.kernels.grid2d.tileset import Tileset
from realmspinner.studio.modes.plotter import mode as plotter_mode
from realmspinner.studio.modes.plotter.engine import project, props, rmap, tmx, tsx
from realmspinner.studio.modes.plotter.engine.tilemap import MapDoc, MapObject, new_uid
from realmspinner.studio.modes.plotter.ui.panes import tileset_editor


def _pixels(w: int = 8, h: int = 8) -> np.ndarray:
    array = np.zeros((h, w, 4), dtype=np.uint8)
    array[..., 3] = 255
    return array


def _tsx_loader(_source: str) -> Tileset:
    return Tileset(name="t", pixels=_pixels(32, 32), tile_w=16, tile_h=16)


def _image_loader(_source: str) -> np.ndarray:
    return _pixels(32, 32)


# --- plotter-map-01: rmap decoded-bytes budget --------------------------------


def test_read_rmap_refuses_many_layers_naming_one_large_member(monkeypatch):
    """A manifest can name one small ``.npy`` member from many tile-layer
    entries; the directory-claimed-bytes precheck sums each *distinct* member
    once, so it never notices. Each entry is still decoded in full and
    independently (tile arrays carry no picture-style cache), so the real cost
    is the entry count times one decode -- reproduced against the unfixed
    reader at a 4.5 KB archive, 40 entries, 168 MB decoded at a 1024-square
    map. ``_ReadBudget.decoded`` charges the running total instead.
    """
    pixels = np.zeros((8, 8, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    doc = MapDoc(100, 100, 8, 8)
    doc.add_tileset(Tileset(name="t", pixels=pixels, tile_w=8, tile_h=8))
    doc.add_tile_layer("Ground")
    doc.mark_saved()

    original = rmap.rmap_bytes(doc)
    with zipfile.ZipFile(io.BytesIO(original)) as zf:
        members = {name: zf.read(name) for name in zf.namelist()}
    manifest = json.loads(members[rmap.MANIFEST])
    one = manifest["layers"][0]
    assert one["type"] == "tile"
    # Ten entries, all naming the *same* "layers/0.npy" member -- the zip
    # directory only ever lists that member once.
    manifest["layers"] = [dict(one) for _ in range(10)]
    members[rmap.MANIFEST] = json.dumps(manifest).encode()
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as zf:
        for name, blob in members.items():
            zf.writestr(name, blob)
    data = out.getvalue()

    claimed = sum(int(info.file_size) for info in zipfile.ZipFile(io.BytesIO(data)).infolist())
    one_decode = doc.width * doc.height * 4
    # The ceiling sits above what the archive *claims* (so the old, unfixed
    # precheck would let this through) but below what ten decodes actually
    # cost -- which is the gap this fix closes.
    ceiling = one_decode * 5
    assert claimed < ceiling < one_decode * 10
    monkeypatch.setattr(rmap, "MAX_DECOMPRESSED_BYTES", ceiling)

    with pytest.raises(ValueError, match="bytes this build reads"):
        rmap.read_rmap(data)


# --- plotter-map-02: tmx/tmj cumulative decoded-cell budget -------------------


def _tmx_with_layers(count: int) -> bytes:
    body = "".join(
        f'<layer id="{i}" name="L{i}" width="4" height="4">'
        '<data encoding="csv">' + ",".join(["0"] * 16) + "</data></layer>"
        for i in range(1, count + 1)
    )
    return (
        '<map version="1.10" orientation="orthogonal" width="4" height="4" '
        'tilewidth="16" tileheight="16">'
        f'<tileset firstgid="1" source="t.tsx"/>{body}</map>'
    ).encode()


def test_read_tmx_refuses_total_decoded_cells_past_a_document_budget(monkeypatch):
    """``_Budget`` counted layers, chunks and objects, but never the *cells*
    each layer decodes to -- a document with many small, legally-declared
    layers costs one full array allocation per layer with nothing summing
    them, which is the same shape of hole a zlib-zeros payload exploits at
    ~750:1 (the audit's own repro: a 222 KB ``.tmx`` decoding to 168 MB).
    ``_Budget.cell_block`` closes it.
    """
    data = _tmx_with_layers(20)
    # 20 layers of 16 cells each = 320 cells total; the ceiling sits between
    # one layer's worth and the full document's.
    monkeypatch.setattr(tmx, "MAX_DECODED_CELLS", 16 * 5)
    with pytest.raises(ValueError, match="cells this build reads"):
        tmx.read_tmx(data, image_loader=_image_loader, tsx_loader=_tsx_loader)


# --- plotter-map-03: isometric resize keeps objects on their cells ------------


def test_resize_keeps_an_isometric_maps_objects_on_their_cells():
    """``resize``/``offset``/``autocrop``/``grow_to_hold`` used to shift every
    object by ``dx * tile_w, dy * tile_h`` -- the orthogonal pixel step -- so
    an object at cell (2, 3) on an isometric map detached from its tile after
    a resize. Recomputing through the lattice (``project.shift_by_cells``)
    keeps it on the same cell, shifted by the same whole-cell amount as the
    grid itself.
    """
    doc = MapDoc(6, 6, 32, 16, projection=project.ISOMETRIC)
    lat = doc._lattice()
    x, y = project.cell_corner(lat, 2, 3)
    layer = doc.add_object_layer()
    doc.add_object(
        layer.uid, MapObject(uid=new_uid(), name="a", kind="point", x=x, y=y)
    )

    # Same size, offset only: isolates the anchor shift itself from an
    # isometric map's origin also moving when its *height* changes (a
    # separate concern this finding is not about).
    doc.resize(6, 6, offset_x=-1, offset_y=2)

    moved = doc.layer(layer.uid).objects[0]
    column, row = project.cell_point(doc._lattice(), moved.x, moved.y)
    assert (round(column), round(row)) == (1, 5)


# --- plotter-mode-01: the tileset editor sheet over a real Tileset -----------


def test_the_tileset_sheet_draws_for_a_real_tileset(monkeypatch):
    """``len(ref.tileset)`` in the sheet's header and its tile grid raised
    ``TypeError`` on every real ``Tileset`` (it has no ``__len__``), which the
    stand-in the picker-clipping test uses papered over: that stand-in
    supplies ``__len__`` itself. ``tile_count`` is the real property both call
    sites now read.
    """
    from _ui_context import imgui_context

    with imgui_context(monkeypatch) as ui:
        from types import SimpleNamespace

        from realmspinner.studio.modes.plotter import state as plotter_state

        doc = MapDoc(4, 4, 8, 8)
        pixels = np.zeros((8, 32, 4), dtype=np.uint8)
        pixels[..., 3] = 255
        doc.add_tileset(Tileset(name="Ground", pixels=pixels, tile_w=8, tile_h=8))
        tab = plotter_state.PlotterDoc(doc=doc, title="m")
        state = plotter_state.PlotterState()
        state.add(tab)
        state.editing_tileset = 0
        ctx = SimpleNamespace(
            state=SimpleNamespace(plotter=state, preview={}),
            toast=lambda *a, **k: None,
        )
        assert tileset_editor.active(ctx) is True

        ui.new_frame()
        ui.begin("##host")
        # This is the crash: before the fix, drawing the sheet over a real
        # Tileset raised TypeError out of ``len(ref.tileset)`` before "Back to
        # the map" was ever reachable.
        tileset_editor.draw(ctx)
        ui.end()
        ui.end_frame()


# --- plotter-mode-02: undo of a tileset add drops the stale brush ------------


def test_undoing_a_tileset_add_drops_the_brush_that_named_it(plotter_ctx):
    """``undo``/``redo``/``step_history`` pruned a stale *object* selection but
    left a brush or terrain naming a tileset the undo just removed -- painting
    with the survivor writes a gid nothing in the document accounts for, and
    ``rmap`` refuses the very next open of the file.
    """
    ctx, state = plotter_ctx
    tab = state.active
    pixels = np.zeros((8, 8, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    ref = tab.doc.add_tileset(Tileset(name="Ground", pixels=pixels, tile_w=8, tile_h=8))
    state.brush = np.array([[ref.firstgid]], dtype=gidlib.DTYPE)

    plotter_mode.undo(ctx, tab)

    assert ref not in tab.doc.tilesets
    assert state.brush is None


# --- plotter-mode-03: a partial hex colour does not raise --------------------


def test_typing_a_partial_text_colour_does_not_raise():
    """The text object's Colour field, and the object layer's Outline colour
    field, wrote straight through to ``replace(Text, ...)``/
    ``set_layer_props`` with no validation, so the first partial keystroke
    ("#", "#4") raised ``ValueError`` out of a pane draw -- and the pane guard
    trips the whole panel after three of those. A half-typed value is kept
    pending instead, the Map properties dialog's background field's own rule.
    """
    import inspect

    from realmspinner.kernels.grid2d.tileset import colour_text
    from realmspinner.studio.modes.plotter.engine.tilemap import Text
    from realmspinner.studio.modes.plotter.ui.panes import layers as plotter_layers

    # The construction door itself: this is what every keystroke used to
    # reach unguarded, straight out of ``replace(shape, color=...)`` /
    # ``set_layer_props(color=...)``.
    with pytest.raises(ValueError):
        Text(color="#4")
    with pytest.raises(ValueError):
        colour_text("#4", "an object layer colour")

    # The text object's Colour field: bounded to the gap between the field
    # being read and the ``values`` dict being built, the same positional rule
    # ``test_tileset_editor_tile_class_and_duration_and_wang_name_typing_is_one_undo_step``
    # already uses in this suite -- not merely "a guard exists somewhere".
    shape_source = inspect.getsource(plotter_layers._shape_fields)
    after_color_field = shape_source.split('"##text-color"', 1)[1]
    before_values = after_color_field.split("values = {", 1)[0]
    assert "colour_text(" in before_values and "color_pending" in before_values

    # The object layer's Outline colour field: bounded to the gap between the
    # field being read and the next row this table draws.
    table_source = inspect.getsource(plotter_layers._layer_table)
    after_outline_field = table_source.split('"##object-layer-color"', 1)[1]
    before_next_field = after_outline_field.split("elif isinstance(layer, ImageLayer):", 1)[0]
    assert "except ValueError:" in before_next_field


# --- plotter-map-04: a property's value is checked against its type ---------


def test_a_property_whose_value_does_not_match_its_type_is_refused_at_the_reader():
    """Scalar property values (``string``/``int``/``float``/``bool``/``color``/
    ``file``) were never checked against the type the property claims to be.
    The JSON codec is the door this bites hardest: it hands a raw JSON value
    straight to ``Prop`` with no coercion, so a ``.tmj`` declaring ``"type":
    "int", "value": "not a number"`` built one silently -- and a
    ``"type": "bool", "value": "false"`` built a ``Prop`` whose value was the
    *string* ``"false"``, which ``_value_text``'s plain ``bool(...)`` test
    reads as truthy and re-exports as ``"true"``.
    """
    with pytest.raises(ValueError):
        props.read_json_properties([{"name": "n", "type": "int", "value": "not a number"}])

    result = props.read_json_properties([{"name": "flag", "type": "bool", "value": "false"}])
    assert result["flag"].value is False


# --- plotter-map-05: an unknown render order falls back at the reader -------


def test_reading_an_unknown_render_order_is_refused_or_falls_back_at_the_door():
    """All three readers stored ``renderorder`` straight from the file with no
    check -- ``render_map`` raised on the first draw instead, and once a
    document held a bad value ``MapDoc.set_map_settings`` could not repair it
    either (it re-validates the unchanged ``before`` on its own revert step).
    Falling back at the door means a bad value never gets in.
    """
    xml = (
        b'<map version="1.10" orientation="orthogonal" width="2" height="2" '
        b'tilewidth="16" tileheight="16" renderorder="sideways">'
        b'<tileset firstgid="1" source="t.tsx"/>'
        b'<layer id="1" name="L" width="2" height="2">'
        b'<data encoding="csv">0,0,0,0</data></layer></map>'
    )
    doc = tmx.read_tmx(xml, image_loader=_image_loader, tsx_loader=_tsx_loader)
    assert doc.renderorder in project.RENDER_ORDERS

    payload = {
        "width": 2, "height": 2, "tilewidth": 16, "tileheight": 16,
        "orientation": "orthogonal", "renderorder": "sideways",
        "tilesets": [{"firstgid": 1, "source": "t.tsx"}],
        "layers": [
            {"type": "tilelayer", "name": "L", "width": 2, "height": 2, "data": [0, 0, 0, 0]}
        ],
    }
    doc2 = tmx.read_tmj(
        json.dumps(payload).encode(), image_loader=_image_loader, tsx_loader=_tsx_loader
    )
    assert doc2.renderorder in project.RENDER_ORDERS

    base = MapDoc(4, 4, 8, 8)
    pixels = np.zeros((8, 8, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    base.add_tileset(Tileset(name="t", pixels=pixels, tile_w=8, tile_h=8))
    base.add_tile_layer("Ground")
    base.mark_saved()
    with zipfile.ZipFile(io.BytesIO(rmap.rmap_bytes(base))) as zf:
        members = {name: zf.read(name) for name in zf.namelist()}
    manifest = json.loads(members[rmap.MANIFEST])
    manifest["renderorder"] = "sideways"
    members[rmap.MANIFEST] = json.dumps(manifest).encode()
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as zf:
        for name, blob in members.items():
            zf.writestr(name, blob)
    doc3 = rmap.read_rmap(out.getvalue())
    assert doc3.renderorder in project.RENDER_ORDERS


# --- plotter-map-06: a JSON Wang colour/set tile of 0 survives --------------


def test_a_json_wang_colour_tile_of_zero_survives_the_read():
    """``int(x.get("tile", -1) or -1)`` treats a stored ``0`` as missing and
    turns a genuine wang tile 0 into -1 (unset), while the XML twin
    (``_int_attr``, no ``or`` idiom) keeps it.
    """
    entries = [
        {
            "name": "Terrain",
            "type": "corner",
            "colors": [{"name": "grass", "color": "#00ff00", "tile": 0}],
            "wangtiles": [],
            "tile": 0,
        }
    ]
    sets = tsx.read_wang_model_json(entries)
    assert sets[0].tile == 0
    assert sets[0].colours[0].tile == 0


# --- plotter-map-07: non-finite values are refused at the reader -----------


def test_a_non_finite_layer_offset_is_refused_at_the_reader():
    """``float("inf")`` parses cleanly out of Tiled's own text spelling, and
    Python's ``json`` module accepts ``Infinity``/``NaN`` by default, so both
    readers built a document carrying one -- and ``render_map`` discovered it
    later as a bare ``OverflowError`` out of ``round(inf)`` deep inside a
    render, rather than at the door the file came in through.
    """
    xml = (
        b'<map version="1.10" orientation="orthogonal" width="2" height="2" '
        b'tilewidth="16" tileheight="16">'
        b'<tileset firstgid="1" source="t.tsx"/>'
        b'<layer id="1" name="L" width="2" height="2" offsetx="inf">'
        b'<data encoding="csv">0,0,0,0</data></layer></map>'
    )
    with pytest.raises(ValueError, match="finite"):
        tmx.read_tmx(xml, image_loader=_image_loader, tsx_loader=_tsx_loader)

    payload = {
        "width": 2, "height": 2, "tilewidth": 16, "tileheight": 16,
        "orientation": "orthogonal",
        "tilesets": [{"firstgid": 1, "source": "t.tsx"}],
        "layers": [
            {
                "type": "tilelayer", "name": "L", "width": 2, "height": 2,
                "data": [0, 0, 0, 0], "opacity": float("nan"),
            }
        ],
    }
    with pytest.raises(ValueError, match="finite"):
        tmx.read_tmj(
            json.dumps(payload).encode(), image_loader=_image_loader, tsx_loader=_tsx_loader
        )
