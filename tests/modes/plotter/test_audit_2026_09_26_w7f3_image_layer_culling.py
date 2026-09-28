"""plotter-mode-19: a repeating image layer drew every copy the *map* could
hold, not every copy the *pane* could show.

``canvas._image_layer`` tiles a small picture across an ``ImageLayer``'s
``repeat_x``/``repeat_y`` flags by walking ``render.repeats``, whose span is
the map's own pixel extent -- correct for ``render.render_image``, which
composites a still with no camera to cull against, but the canvas draws to a
finite pane every frame. Before this fix the canvas walked the very same
uncapped copy list and called ``draw_list.add_image`` for each one, so a small
picture repeating across a large map cost draw calls proportional to the
*map's* size rather than the *window's* -- thousands of quads a zoomed-in pane
could never show a single pixel of.
"""

from __future__ import annotations

from types import SimpleNamespace

import imgui_bundle
import numpy as np

from realmspinner.studio.modes.plotter.engine import scene as plotter_scene
from realmspinner.studio.modes.plotter.engine.tilemap import MapDoc
from realmspinner.studio.modes.plotter.ui.panes import canvas as canvas
from realmspinner.studio.shell import paintview


class _FakeDrawList:
    def __init__(self) -> None:
        self.image_calls: list[tuple] = []

    def add_image(self, texture, p0, p1, uv0, uv1, tint) -> None:
        self.image_calls.append((texture, p0, p1, uv0, uv1, tint))


def _picture(w: int = 4, h: int = 4) -> np.ndarray:
    pixels = np.zeros((h, w, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    return pixels


def _setup(monkeypatch):
    fake_imgui = SimpleNamespace(get_color_u32=lambda _rgba: 0xFFFFFFFF)
    monkeypatch.setattr(imgui_bundle, "imgui", fake_imgui)
    monkeypatch.setattr(canvas.plotter_textures, "image_texture", lambda *a, **k: "tex")
    monkeypatch.setattr(canvas.widgets, "texture_ref", lambda tex: tex)


def test_a_repeating_image_layer_only_draws_copies_the_pane_can_see(monkeypatch):
    _setup(monkeypatch)

    # 1000 columns of 16px tiles: a 16000px-wide map, so an uncapped tiling of
    # a 4px-wide picture is 4000 copies -- and every one of them used to reach
    # ``draw_list.add_image`` regardless of where the view was looking.
    doc = MapDoc(1000, 1, 16, 16)
    layer = doc.add_image_layer("sky", pixels=_picture(4, 4), repeat_x=True, repeat_y=False)
    entry = plotter_scene.resolved_for(doc, layer.uid)
    assert entry is not None

    view = paintview.PaintView()
    tab = SimpleNamespace(doc=doc, view=view, uid="tab")
    origin = (0.0, 0.0)
    region = (100.0, 100.0)
    shift = canvas._layer_shift(view, origin, entry)

    draw_list = _FakeDrawList()
    canvas._image_layer(None, tab, draw_list, origin, region, layer, entry, shift)

    # Only the copies overlapping [0, 100) can show on a pane that size --
    # px in {0, 4, ..., 96}, 25 of them -- however many thousand the map holds.
    assert len(draw_list.image_calls) == 25
    for _texture, p0, _p1, _uv0, _uv1, _tint in draw_list.image_calls:
        assert p0[0] < region[0] + 4.0 * view.zoom


def test_panning_far_from_the_origin_still_only_draws_the_visible_copies(monkeypatch):
    """The same map, panned 8000px right: a different 25 copies draw, not the
    thousands between the origin and the new viewport, and not the whole map."""
    _setup(monkeypatch)

    doc = MapDoc(1000, 1, 16, 16)
    layer = doc.add_image_layer("sky", pixels=_picture(4, 4), repeat_x=True, repeat_y=False)
    entry = plotter_scene.resolved_for(doc, layer.uid)
    assert entry is not None

    view = paintview.PaintView(pan=(-8000.0, 0.0))
    tab = SimpleNamespace(doc=doc, view=view, uid="tab")
    origin = (0.0, 0.0)
    region = (100.0, 100.0)
    shift = canvas._layer_shift(view, origin, entry)

    draw_list = _FakeDrawList()
    canvas._image_layer(None, tab, draw_list, origin, region, layer, entry, shift)

    assert len(draw_list.image_calls) == 25
