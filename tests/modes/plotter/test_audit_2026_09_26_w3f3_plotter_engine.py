"""Closing plotter-map-08/09/10 and two new findings the w3f3 fixer surfaced.

One test per finding, named for the claim it proves. Every one of these failed
against the unfixed code before its matching fix landed -- proven by loading
the pre-fix source from ``git show HEAD:<path>`` as a throwaway package copy
in the fixer's scratchpad and running it there, never by checking the old text
out over the tree (``dev/INVARIANTS.md``'s rule). See the fixer's own return
for the pasted failing output.
"""

from __future__ import annotations

import inspect
import io
import json
import zipfile

import numpy as np
import pytest

from realmspinner.kernels.grid2d.tileset import Tileset
from realmspinner.studio.modes.plotter.engine import project, rmap, tmx, tsx
from realmspinner.studio.modes.plotter.engine.tilemap import MapDoc


def _pixels(w: int = 8, h: int = 8) -> np.ndarray:
    array = np.zeros((h, w, 4), dtype=np.uint8)
    array[..., 3] = 255
    return array


def _image_loader(_path: str) -> np.ndarray:
    return _pixels(16, 16)


def _tsx_loader(_path: str) -> Tileset:
    return Tileset(name="t", pixels=_pixels(16, 16), tile_w=16, tile_h=16)


def _doc() -> MapDoc:
    doc = MapDoc(4, 4, 8, 8)
    doc.add_tileset(Tileset(name="t", pixels=_pixels(), tile_w=8, tile_h=8))
    return doc


def _rewrite(doc: MapDoc, mutate) -> bytes:
    """Rebuild an ``.rmap`` archive with a mutated manifest, same shape as
    ``test_rmap.py``'s own private helper -- kept local rather than imported
    since that one is not part of this module's public surface."""
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


# --- plotter-map-08: malformed-but-parseable files raise ValueError ----------


def test_a_csv_cell_too_large_for_int64_is_refused_as_a_value_error():
    """``int("999...")`` parses fine as an arbitrary-precision Python int; only
    the cast to ``int64`` inside ``_gid_array`` discovers it does not fit, and
    that discovery used to leave as a bare ``OverflowError``."""
    xml = (
        b'<map version="1.10" orientation="orthogonal" width="1" height="1" '
        b'tilewidth="16" tileheight="16">'
        b'<tileset firstgid="1" source="t.tsx"/>'
        b'<layer id="1" name="L" width="1" height="1">'
        b'<data encoding="csv">99999999999999999999</data></layer></map>'
    )
    with pytest.raises(ValueError, match="not a number"):
        tmx.read_tmx(xml, image_loader=_image_loader, tsx_loader=_tsx_loader)


def test_a_tile_layers_data_as_a_bare_scalar_is_refused_as_a_value_error():
    """Tiled's own writer always emits a list; a ``"data": 5`` used to reach
    ``list(5)`` before ``_gid_array`` ever saw the value, raising a bare
    ``TypeError`` instead of the refusal every other malformed shape gets."""
    payload = {
        "width": 2, "height": 2, "tilewidth": 16, "tileheight": 16,
        "orientation": "orthogonal",
        "tilesets": [{"firstgid": 1, "source": "t.tsx"}],
        "layers": [
            {"type": "tilelayer", "name": "L", "width": 2, "height": 2, "data": 5}
        ],
    }
    with pytest.raises(ValueError, match="not a number"):
        tmx.read_tmj(
            json.dumps(payload).encode(), image_loader=_image_loader, tsx_loader=_tsx_loader
        )


def test_a_tilesets_entry_that_is_not_an_object_is_refused_as_a_value_error():
    """``"tilesets": [5]`` used to reach ``entry.get(...)`` on a bare int and
    raise ``AttributeError``, the layer loop's own long-fixed refusal by name a
    few hundred lines down from this one having never been mirrored here."""
    payload = {
        "width": 2, "height": 2, "tilewidth": 16, "tileheight": 16,
        "orientation": "orthogonal",
        "tilesets": [5],
        "layers": [
            {"type": "tilelayer", "name": "L", "width": 2, "height": 2, "data": [0, 0, 0, 0]}
        ],
    }
    with pytest.raises(ValueError, match="tileset reference is not an object"):
        tmx.read_tmj(
            json.dumps(payload).encode(), image_loader=_image_loader, tsx_loader=_tsx_loader
        )


def test_an_rmap_manifest_field_holding_json_null_is_refused_not_a_type_error():
    """``manifest.get("width", 1)`` hands back the stored ``None`` for a
    present-but-null key rather than the default, and plain ``int(None)`` used
    to raise ``TypeError`` out of this reader (the twin of plotter-map-05's
    ``renderorder`` gap, for the numeric header fields)."""
    doc = _doc()
    doc.add_tile_layer("Ground")
    data = _rewrite(doc, lambda m: m.update(width=None))
    with pytest.raises(ValueError):
        rmap.read_rmap(data)


def test_a_tilesets_own_tiles_entry_that_is_not_an_object_is_skipped():
    """``check_tileset_features_json`` and ``collection_sources_json`` already
    skip a non-object ``tiles`` entry; ``read_tile_meta_json`` reached
    ``tile.get("id", ...)`` on a bare int instead and raised ``AttributeError``."""
    assert tsx.read_tile_meta_json({"tiles": [5]}) == {}


# --- plotter-map-09: a duplicated image layer's pixels stay frozen ----------


def test_duplicating_an_image_layer_refreezes_its_pixels():
    """``copy.deepcopy`` on a numpy array calls its own ``.copy()``, which does
    not carry the ``writeable=False`` flag over -- the frozen picture
    ``ImageLayer.__post_init__`` set at construction came back writable on the
    far side of a duplicate."""
    doc = _doc()
    layer = doc.add_image_layer("Pic", pixels=_pixels(4, 4))
    assert layer.pixels.flags.writeable is False
    dup = doc.duplicate_layer(layer.uid)
    assert dup.pixels.flags.writeable is False


def test_an_undone_and_redone_image_layer_duplicate_stays_frozen():
    """Undo/redo replay the same layer object ``LayerAddEdit`` holds, so the
    refreeze has to happen once at duplication rather than at every edit site
    that might otherwise see the writable copy."""
    doc = _doc()
    layer = doc.add_image_layer("Pic", pixels=_pixels(4, 4))
    dup = doc.duplicate_layer(layer.uid)
    doc.undo()
    doc.redo()
    found = doc._locate(dup.uid)
    assert found is not None
    assert found[0].pixels.flags.writeable is False


# --- plotter-map-10: stale docstrings ---------------------------------------


def test_project_docstring_no_longer_claims_offset_cell_at_is_unimplemented():
    """``_offset_cell_at`` has existed since the offset lattices landed; the
    module docstring still said the nearest-centre test "is not implemented
    here" well after that. The check is the exact original contiguous claim,
    not a bare substring: the fix's own comment names the incident by quoting
    that old wording, so a substring check would fail against the fixed
    source too."""
    assert project.__doc__ is not None
    assert "is not an affine inverse and is not implemented here" not in project.__doc__
    assert hasattr(project, "_offset_cell_at")


def test_projections_comment_no_longer_calls_staggered_and_hexagonal_refused():
    """Staggered and hexagonal are in :data:`project.PROJECTIONS`, which is
    exactly what ``tmx._check_orientation`` accepts -- neither is refused at
    that door, unlike what the comment above the tuple used to claim. Checked
    as the exact original phrase for the same reason as the test above."""
    assert "remain explicit refusals at the Tiled door" not in inspect.getsource(project)
    assert project.STAGGERED in project.PROJECTIONS
    assert project.HEXAGONAL in project.PROJECTIONS


# Each pair is (module, the exact stale backtick-quoted citation it used to
# carry) -- checked for that literal old code reference rather than a bare
# "plotter_io" substring, since the fix's own comments explain the rename by
# naming the retired module in plain prose right beside the corrected one.
_RETIRED_CITATIONS = [
    (rmap, "``studio/plotter_io._encoded``"),
    (rmap, "``studio/plotter_mode.export_library``"),
    (tmx, "``plotter_io._load``"),
    (tmx, "``plotter_io._write``"),
]


@pytest.mark.parametrize("module,stale", _RETIRED_CITATIONS)
def test_engine_modules_no_longer_cite_the_retired_plotter_prefixed_names(module, stale):
    assert stale not in inspect.getsource(module)


def _flat(module) -> str:
    """Source with comment markers dropped and runs of whitespace collapsed,
    so a check for one of the original sentences does not depend on exactly
    where a multi-line ``#`` comment wraps."""
    return " ".join(inspect.getsource(module).replace("#", " ").split())


def test_tilemap_and_map_model_no_longer_cite_the_retired_plotter_prefixed_names():
    from realmspinner.studio.modes.plotter.engine import _map_geometry, _map_model, tilemap

    assert "the new-map form's ``plotter_setup.MAX_TILES``" not in _flat(tilemap)
    assert "because that is the size ``plotter_tilesets`` slices at" not in _flat(_map_geometry)
    assert "exactly why ``plotter_mode._paste`` refuses" not in _flat(_map_model)


# --- new finding: set_map_settings's revert re-validates every field --------


def test_fixing_the_one_invalid_field_a_document_holds_no_longer_refuses():
    """``set_map_settings`` used to apply ``after``, read back the normalized
    result, then revert to ``before`` to decide whether anything had changed --
    and that revert step, ``_apply_map_settings(before)``, re-validates every
    field of ``before``, not just the one being changed. A document already
    holding an invalid field (an unvalidated ``stagger_axis`` a pre-fallback
    reader once let through, say) could apply a fix to that very field --
    ``after`` is valid -- and then have the revert raise on the ``before`` it
    was only ever going to discard, undoing the fix along with it."""
    doc = _doc()
    doc.stagger_axis = "sideways"  # simulates a document holding a bad value
    doc.set_map_settings(stagger_axis="x")
    assert doc.map_settings()["stagger_axis"] == "x"


def test_set_map_settings_no_longer_reverts_and_reapplies():
    """The apply-revert-reapply dance is gone: ``_apply_map_settings`` is
    called once now, not three times. Matched as ``self._apply_map_settings(``
    -- the actual call -- rather than a bare substring, since the method's own
    docstring now names the old ``_apply_map_settings(before)`` call in prose
    to explain the fix."""
    source = inspect.getsource(MapDoc.set_map_settings)
    assert source.count("self._apply_map_settings(") == 1


# --- new finding: rmap.py's float parses accept non-finite values -----------


def test_a_non_finite_object_coordinate_is_refused():
    """The twin of plotter-map-07's ``tmx`` fix: Python's ``json`` module
    accepts the non-standard ``Infinity`` literal by default, and ``_read_object``
    used to hand it straight to ``MapObject`` with a bare ``float()``."""

    doc = _doc()
    doc.add_object_layer("Objs")
    data = _rewrite(
        doc,
        lambda m: m["layers"][-1].__setitem__(
            "objects",
            [{"id": 1, "name": "o", "x": float("inf"), "y": 0.0, "shape": {"kind": "point"}}],
        ),
    )
    with pytest.raises(ValueError, match="finite"):
        rmap.read_rmap(data)


def test_a_non_finite_layer_offset_is_refused():
    doc = _doc()
    doc.add_tile_layer("Ground")
    data = _rewrite(doc, lambda m: m["layers"][-1].__setitem__("offset", [float("nan"), 0.0]))
    with pytest.raises(ValueError, match="finite"):
        rmap.read_rmap(data)
