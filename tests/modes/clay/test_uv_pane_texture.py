"""The UV pane draws the material's texture behind the islands (Phase 3, item 6).

Two halves, tested apart the way ``ui/_uv_texture.py`` is built: the pure choice
of *which* material's texture to show, and the one-texture-per-tab GL cache
around it, driven with a fake texture factory (the repo has no GL context under
test). The draw order is read off a recording draw list.
"""

from __future__ import annotations

import inspect
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from _ui_context import imgui_context

from realmspinner.kernels.geom3d import gltf
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio.modes.clay.ui import _uv_texture
from realmspinner.studio.modes.clay.ui.panes import uv as clay_uv


def _image(seed: int = 0, size: int = 2) -> tuple[int, int, bytes]:
    return (size, size, bytes([seed % 256]) * (size * size * 4))


def _textured(seed: int = 0, **fields) -> gltf.Material:
    return gltf.Material(name=f"m{seed}", base_color=_image(seed), **fields)


def _doc_with_box() -> tuple[bd.ClayDoc, bd.Obj]:
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
    return doc, doc.by_uid(obj.uid)


# --- which material's texture --------------------------------------------------


def test_the_pane_shows_the_objects_default_slot_when_no_face_is_selected() -> None:
    doc, obj = _doc_with_box()
    slot = doc.add_material(_textured(1))
    doc.set_props(obj.uid, material=slot)
    assert _uv_texture.material_for_pane(doc, doc.by_uid(obj.uid)) is doc.materials[slot]


def test_the_pane_shows_the_first_selected_faces_slot_over_the_default() -> None:
    doc, obj = _doc_with_box()
    default = doc.add_material(_textured(1))
    painted = doc.add_material(_textured(2))
    doc.set_props(obj.uid, material=default)
    per_face = np.zeros(len(obj.mesh.material), dtype="i4")
    per_face[3] = painted
    doc.set_mesh(obj.uid, replace(obj.mesh, material=per_face))
    doc.set_element_mode("face")
    doc.set_element_sel(obj.uid, el.ElementSel(faces=[3, 5]))
    assert _uv_texture.material_for_pane(doc, doc.by_uid(obj.uid)) is doc.materials[painted]


def test_a_material_without_a_texture_shows_nothing_extra() -> None:
    doc, obj = _doc_with_box()
    assert doc.materials[obj.material].base_color is None
    assert _uv_texture.material_for_pane(doc, obj) is None


def test_a_slot_outside_the_palette_shows_nothing_rather_than_raising() -> None:
    """A stale selection against a mesh whose slots no longer exist."""
    doc, obj = _doc_with_box()
    per_face = np.full(len(obj.mesh.material), 9, dtype="i4")
    doc.set_mesh(obj.uid, replace(obj.mesh, material=per_face))
    doc.set_element_mode("face")
    doc.set_element_sel(obj.uid, el.ElementSel(faces=[0]))
    assert _uv_texture.material_for_pane(doc, doc.by_uid(obj.uid)) is None


# --- the cache: one texture per tab, rebuilt on a new source -------------------


class _Texture:
    def __init__(self, size, data) -> None:
        self.size = size
        self.data = data
        self.filter = None
        self.released = False
        self.glo = id(self) & 0xFFFF

    def release(self) -> None:
        self.released = True


class _GL:
    NEAREST = 9728
    LINEAR = 9729

    def __init__(self) -> None:
        self.made: list[_Texture] = []

    def texture(self, size, components, data) -> _Texture:
        assert components == 4
        made = _Texture(size, data)
        self.made.append(made)
        return made


def _ctx(with_viewer: bool = True) -> SimpleNamespace:
    gl = _GL()
    return SimpleNamespace(
        viewer=SimpleNamespace(ctx=gl) if with_viewer else None,
        state=SimpleNamespace(preview={}),
        gl=gl,
    )


def test_the_texture_is_uploaded_once_while_the_source_is_unchanged() -> None:
    ctx = _ctx()
    material = _textured(1)
    first = _uv_texture.pane_texture(ctx, "bd1", material)
    assert _uv_texture.pane_texture(ctx, "bd1", material) is first
    assert len(ctx.gl.made) == 1
    assert first.size == (2, 2) and first.data == material.base_color[2]


def test_a_replaced_material_is_a_new_texture_and_the_old_one_is_released() -> None:
    """The Inker pull's shape: ``replace(material, base_color=...)`` lands a new
    tuple under the same slot, and one pull per frame must not leak one GL
    texture each."""
    ctx = _ctx()
    material = _textured(1)
    first = _uv_texture.pane_texture(ctx, "bd1", material)
    pulled = replace(material, base_color=_image(2))
    second = _uv_texture.pane_texture(ctx, "bd1", pulled)
    assert second is not first
    assert first.released and not second.released
    for seed in range(3, 8):
        _uv_texture.pane_texture(ctx, "bd1", replace(material, base_color=_image(seed)))
    live = [t for t in ctx.gl.made if not t.released]
    assert len(live) == 1, "one texture per tab, whatever the number of pulls"


def test_equal_bytes_in_a_new_tuple_still_rebuild_because_identity_is_the_stamp() -> None:
    ctx = _ctx()
    material = _textured(1)
    first = _uv_texture.pane_texture(ctx, "bd1", material)
    same_bytes = replace(material, base_color=(2, 2, bytes(material.base_color[2])))
    assert same_bytes.base_color == material.base_color
    assert _uv_texture.pane_texture(ctx, "bd1", same_bytes) is not first


def test_a_nearest_material_samples_nearest_and_the_flag_flips_the_sampler() -> None:
    ctx = _ctx()
    crisp = _textured(1, nearest=True)
    soft = replace(crisp, nearest=False)
    nearest_tex = _uv_texture.pane_texture(ctx, "bd1", crisp)
    assert nearest_tex.filter == (ctx.gl.NEAREST, ctx.gl.NEAREST)
    linear_tex = _uv_texture.pane_texture(ctx, "bd1", soft)
    assert linear_tex is not nearest_tex and nearest_tex.released
    assert linear_tex.filter == (ctx.gl.LINEAR, ctx.gl.LINEAR)


def test_a_material_with_no_texture_releases_what_the_tab_held() -> None:
    ctx = _ctx()
    held = _uv_texture.pane_texture(ctx, "bd1", _textured(1))
    assert _uv_texture.pane_texture(ctx, "bd1", gltf.Material()) is None
    assert held.released
    assert not [k for k in ctx.state.preview if k.startswith(_uv_texture.PREFIX)]
    assert _uv_texture.pane_texture(ctx, "bd1", None) is None


def test_closing_one_tab_releases_its_texture_and_not_a_neighbours() -> None:
    """``bd1`` is a prefix of ``bd10``: the sweep must not take both."""
    ctx = _ctx()
    one = _uv_texture.pane_texture(ctx, "bd1", _textured(1))
    ten = _uv_texture.pane_texture(ctx, "bd10", _textured(2))
    _uv_texture.release_doc(ctx, "bd1")
    assert one.released and not ten.released
    _uv_texture.release_all(ctx)
    assert ten.released
    assert ctx.state.preview == {}


def test_no_gl_context_draws_nothing_and_holds_nothing() -> None:
    ctx = _ctx(with_viewer=False)
    assert _uv_texture.pane_texture(ctx, "bd1", _textured(1)) is None
    assert ctx.state.preview == {}
    assert _uv_texture.pane_texture(None, "bd1", _textured(1)) is None


def test_a_malformed_image_is_not_uploaded() -> None:
    ctx = _ctx()
    short = gltf.Material(base_color=(4, 4, b"\x00" * 10))
    assert _uv_texture.pane_texture(ctx, "bd1", short) is None
    assert ctx.gl.made == []


# --- the draw: under the islands -------------------------------------------------


class _RecordingDrawList:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __getattr__(self, name: str):
        def record(*args, **kwargs) -> None:
            self.calls.append(name)

        return record


@pytest.fixture
def ui(monkeypatch):
    with imgui_context(monkeypatch) as imgui:
        yield imgui


def test_the_backdrop_image_is_drawn_between_the_square_and_its_border(ui) -> None:
    draw = _RecordingDrawList()
    texture = SimpleNamespace(glo=7)
    view = clay_uv.UvPaneState().view
    ui.new_frame()
    try:
        clay_uv._backdrop(draw, view, (0.0, 0.0), texture)
    finally:
        ui.end_frame()
    assert draw.calls == ["add_rect_filled", "add_image", "add_rect"]


def test_a_backdrop_without_a_texture_adds_no_image(ui) -> None:
    draw = _RecordingDrawList()
    view = clay_uv.UvPaneState().view
    ui.new_frame()
    try:
        clay_uv._backdrop(draw, view, (0.0, 0.0))
    finally:
        ui.end_frame()
    assert draw.calls == ["add_rect_filled", "add_rect"]


def test_the_canvas_draws_the_backdrop_before_the_island_fills_and_outlines() -> None:
    source = inspect.getsource(clay_uv._canvas)
    backdrop = source.index("_backdrop(draw_list")
    assert backdrop < source.index("_faces(draw_list")
    assert backdrop < source.index("_island_outlines(")


def test_the_pane_has_no_seam_stretch_or_texel_overlay() -> None:
    source = inspect.getsource(clay_uv)
    for gone in ("seam", "stretch", "texel"):
        assert f"def _{gone}" not in source
