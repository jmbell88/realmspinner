"""Whether ``export_tileset`` can ever write a tileset that is missing a float.

The question (F17-7, left open by the 2026-10-10 audit): ``export_tileset``
never calls ``_settle``, so it never lands a floating buffer, and it writes
only tileset data. That is a bug exactly when something can rebuild a tileset
*from* a float, because the export would then hold the pre-float atlas. Traced:
nothing can. Every door that raises a float (``lift``, ``paste``,
``place_pixels``) refuses a tilemap layer by name, so a float is always bound
to a raster layer; and every door that builds or rewrites a tileset out of
layer pixels (``add_tilemap_layer``, ``convert_layer_to_tilemap``,
``convert_layer_to_raster``) lands the float first. These tests pin each half,
so the day one of them stops being true this file is what says export has to
start settling.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from realmspinner.kernels.pixel.document import Document
from realmspinner.kernels.pixel.selection import SelectionMask
from realmspinner.kernels.pixel.tiles import strip
from realmspinner.studio.modes.inker import mode as inker_mode
from realmspinner.studio.modes.inker.state import InkerDoc, InkerState

RED = (255, 0, 0, 255)


def _tile(colour: tuple[int, int, int, int]) -> np.ndarray:
    tile = np.zeros((4, 4, 4), dtype=np.uint8)
    tile[..., 0], tile[..., 1], tile[..., 2], tile[..., 3] = colour
    return tile


def _floating_doc() -> tuple[Document, Any]:
    """A raster layer with a lifted block, plus a tileset the export can name."""
    doc = Document.blank(8, 8)
    doc.stack.active.pixels[0:4, 0:4] = np.array(RED, dtype=np.uint8)
    slot = doc.add_tileset(strip(np.stack([np.zeros((4, 4, 4), np.uint8), _tile(RED)])))
    doc.mask = SelectionMask.from_rect(doc.size, (0, 0, 4, 4))
    assert doc.lift() is True
    assert doc.floating is not None
    return doc, slot


class _Ctx:
    def __init__(self, state: InkerState) -> None:
        self.state = type("S", (), {"inker": state, "plotter": None})()
        self.toasts: list[tuple[str, str]] = []

    def toast(self, message: str, kind: str = "info", **_kw: Any) -> None:
        self.toasts.append((message, kind))

    def submit(self, key: str, run: Any, *args: Any) -> bool:
        run(*args)
        return True


def test_a_float_cannot_be_raised_over_a_tilemap_layer_so_it_never_lands_in_a_tileset():
    doc = Document.blank(8, 8)
    slot = doc.add_tileset(strip(np.stack([np.zeros((4, 4, 4), np.uint8), _tile(RED)])))
    doc.add_tilemap_layer(slot.uid)
    doc.mask = SelectionMask.from_rect(doc.size, (0, 0, 4, 4))

    with pytest.raises(ValueError, match="tilemap"):
        doc.lift()
    assert doc.floating is None


def test_building_a_tileset_from_a_layer_lands_an_open_float_first():
    """The only doors that turn layer pixels into tileset tiles commit the float
    before they read the layer, so an export after them already has it."""
    doc, _slot = _floating_doc()
    layer = doc.stack.active

    assert doc.convert_layer_to_tilemap(layer.uid, 4, 4) is True

    assert doc.floating is None
    assert doc.tilesets[-1].tileset.tile_count >= 2


def test_export_tileset_with_a_float_open_writes_the_tileset_and_leaves_the_float(
    monkeypatch, tmp_path
):
    from PIL import Image

    from realmspinner.studio import dialogs

    doc, slot = _floating_doc()
    tab = InkerDoc(doc=doc, title="atlas.ora")
    state = InkerState()
    state.add(tab)
    dest = tmp_path / "atlas.tsx"
    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: dest)
    history_before = len(doc.history)

    inker_mode.export_tileset(_Ctx(state), tab, index=0)

    image = np.asarray(Image.open(dest.with_suffix(".png")).convert("RGBA"))
    assert np.array_equal(image, slot.tileset.pixels)
    # Not a settle: the export neither landed the float nor wrote a history step.
    assert doc.floating is not None
    assert len(doc.history) == history_before
