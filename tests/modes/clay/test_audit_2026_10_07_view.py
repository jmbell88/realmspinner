"""The 2026-10-07 audit's viewport findings (clay-27 .. clay-31, clay-62 .. clay-65).

Most of these need a real GL context (a ``ClayView`` owns a renderer), so they
ride the session ``gl`` fixture and skip where there is no GPU, like every
sibling in this package. The two documentation claims and the hint-line claim
read source text and need none.
"""

from __future__ import annotations

import inspect
import io
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from realmspinner.kernels.geom3d import gltf
from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio import viewport_hints as clay_hints
from realmspinner.studio.modes.clay.ui import _view_drag
from realmspinner.studio.modes.clay.ui import view as clay_view
from realmspinner.studio.viewer import scene as scenelib

RECT = (0.0, 0.0, 128.0, 96.0)


class _State:
    def __init__(self, tool: str = "select") -> None:
        self.tool = tool
        self.snap = False
        self.snap_translate = 0.125
        self.snap_rotate = 15.0


class _Ctx:
    def __init__(self, tool: str = "select") -> None:
        self.state = type("S", (), {"clay": _State(tool)})()


@pytest.fixture(autouse=True)
def _headless_mods(monkeypatch):
    """``pygame.key.get_mods`` raises with no display; stub it at zero."""
    import pygame

    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0, raising=False)


@pytest.fixture
def view(gl):
    v = clay_view.ClayView(gl, _Ctx())
    yield v
    v.release()


def _doc(count: int = 1) -> bd.ClayDoc:
    doc = bd.ClayDoc()
    for i in range(count):
        doc.add_object(
            bd.Obj(
                uid=bd.new_uid(),
                name=f"obj{i}",
                mesh=bp.box(),
                translation=m3.vec3(float(i) * 3.0, 0.0, 0.0),
            )
        )
    return doc


def _alive(texture: Any) -> bool:
    """A released moderngl texture's ``mglo`` becomes an ``InvalidObject``."""
    try:
        texture.read()
    except AttributeError:
        return False
    return True


def _textures(view: Any) -> set[Any]:
    return {
        texture
        for entry in view._cache.values()
        for material in entry.gpu.materials
        for texture in material.textures.values()
    }


def _event(kind: int, **kwargs: Any) -> Any:
    import pygame

    return pygame.event.Event(kind, **kwargs)


# --- clay-27: the world pass is O(N) on a steady frame ------------------------


def test_a_steady_frame_over_four_thousand_objects_does_not_scan_the_document_per_object(
    view, monkeypatch
) -> None:
    """``_world`` resolved the ancestor chain through ``doc.ancestors`` and
    ``doc.by_uid`` -- each a linear ``index_of`` scan -- *before* its memo, so a
    frame of N objects was N scans of N even when every memo entry hit
    (607 ms at the 4,096-object import ceiling). The pass now builds one
    uid -> object map per composite; the document's own scan is never entered,
    on a warm frame or a cold one."""
    doc = bd.ClayDoc()
    for i in range(4096):
        doc.add_object(
            bd.Obj(
                uid=bd.new_uid(),
                name=f"o{i}",
                mesh=bp.box(),
                translation=m3.vec3(float(i), 0, 0),
            )
        )
    # The composite reads only ``entry.gpu.draws``; a stand-in entry keeps this
    # test off four thousand real uploads.
    for obj in doc.objects:
        node = gltf.Node(name=obj.name, mesh=0)
        view._cache[obj.uid] = SimpleNamespace(
            gpu=SimpleNamespace(draws=[(node, object())], release=lambda: None)
        )

    scans = {"n": 0}
    real = bd.ClayDoc.index_of

    def counting(self: Any, uid: int) -> int:
        scans["n"] += 1
        return real(self, uid)

    monkeypatch.setattr(bd.ClayDoc, "index_of", counting)

    view._composite(doc)  # cold: every memo entry is a miss
    assert scans["n"] == 0, "a cold frame scanned the document per object"
    view._composite(doc)  # warm: every memo entry is a hit
    assert scans["n"] == 0, "a steady frame scanned the document per object"


def test_the_indexed_world_matches_the_documents_for_a_parented_chain(view) -> None:
    doc = _doc(count=3)
    a, b, c = doc.objects
    b.parent, c.parent = a.uid, b.uid
    a.rotation = m3.quat_from_axis_angle(m3.vec3(0, 1, 0), 0.7)
    a.scale = m3.vec3(2.0, 2.0, 2.0)
    index = {o.uid: o for o in doc.objects}
    for obj in doc.objects:
        assert np.array_equal(view._world(doc, obj, index), doc.world_matrix(obj.uid))
        assert np.array_equal(view._world(doc, obj), doc.world_matrix(obj.uid))


# --- clay-28: one palette picture, one GL texture -----------------------------


def _textured_doc(count: int, side: int = 8) -> bd.ClayDoc:
    doc = bd.ClayDoc()
    tex = (side, side, np.zeros((side, side, 4), np.uint8).tobytes())
    doc.materials[0:1] = [replace(doc.materials[0], base_color=tex, nearest=True)]
    for i in range(count):
        doc.add_object(
            bd.Obj(
                uid=bd.new_uid(),
                name=f"o{i}",
                mesh=bp.box(),
                translation=m3.vec3(float(i) * 2, 0, 0),
            )
        )
    return doc


def test_objects_sharing_one_palette_texture_share_one_gl_texture(view) -> None:
    doc = _textured_doc(6)
    view.sync(doc)
    textures = _textures(view)
    assert len(textures) == 1, f"{len(textures)} GL textures for one palette picture"
    shared = next(iter(textures))

    # Releasing all but one object must not free what the survivor still draws.
    keep = doc.objects[0].uid
    for obj in doc.objects:
        obj.visible = obj.uid == keep
    doc.touch()
    view.sync(doc)
    assert _alive(shared), "a borrower's release freed the texture the survivor samples"

    # And the last one out frees it: nothing is left for the view to leak.
    for obj in doc.objects:
        obj.visible = False
    doc.touch()
    view.sync(doc)
    assert not _alive(shared), "the last user left the texture uploaded"


def test_an_op_drag_preview_swap_does_not_reupload_the_shared_texture(view) -> None:
    doc = _textured_doc(2)
    view.sync(doc)
    before = _textures(view)
    uid = doc.objects[0].uid
    entry = view._cache[uid]
    view._swap_preview(doc, uid, bp.box())
    assert _textures(view) == before, "a preview frame uploaded a texture again"
    assert all(_alive(t) for t in before)
    assert view._cache[uid] is entry


def test_clearing_the_view_releases_every_shared_texture(view) -> None:
    doc = _textured_doc(3)
    view.sync(doc)
    textures = _textures(view)
    assert textures
    view.clear()
    assert not any(_alive(t) for t in textures)


# --- clay-29: an offscreen render shows what is really there ------------------


def _pixels(png: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=int)


def test_render_png_draws_an_object_a_pending_preview_would_remove(view) -> None:
    """``_composite`` omitted every object the Familiar preview would remove --
    right for the live viewport, where ``_ghost_draws`` tints it instead, but
    ``render_png``/``render_ids`` draw no ghost, so a screenshot or a
    ``clay_render`` call showed an empty scene and reported the object
    occluded."""
    doc = _doc(count=1)
    obj = doc.objects[0]
    before = _pixels(view.render_png(doc, size=96, view="front", shading="lit"))
    ids_before = view.render_ids(doc, size=96, view="front")[1]
    assert not (before == 255).all(), "the baseline picture is empty"

    view.set_preview(
        SimpleNamespace(added=frozenset(), mesh_changed=frozenset(), removed=frozenset({obj.uid})),
        doc,
    )
    after = _pixels(view.render_png(doc, size=96, view="front", shading="lit"))
    ids_after = view.render_ids(doc, size=96, view="front")[1]

    assert np.array_equal(before, after), "a pending preview erased a real object from the render"
    assert ids_after == ids_before and ids_after[0][2] > 0


def test_the_interactive_draw_still_hides_what_the_preview_would_remove(view) -> None:
    doc = _doc(count=2)
    gone = doc.objects[0].uid
    view.sync(doc)
    view.set_preview(
        SimpleNamespace(added=frozenset(), mesh_changed=frozenset(), removed=frozenset({gone})),
        doc,
    )
    assert gone not in view._composite(doc).uids
    assert gone in view._composite(doc, hide_removed=False).uids


# --- clay-30: a second button does not hijack a live marquee ------------------


def test_a_middle_press_during_a_marquee_neither_pans_nor_strands_the_rectangle(view) -> None:
    import pygame

    doc = _doc(count=1)
    doc.set_element_mode("face")
    view.frame_selection(doc)
    view.draw(doc, RECT, 0.0)
    far = (int(RECT[2]) - 1, int(RECT[3]) - 1)

    view.handle_event(doc, _event(pygame.MOUSEBUTTONDOWN, button=1, pos=(1, 1)), True)
    view.handle_event(doc, _event(pygame.MOUSEMOTION, pos=far), True)
    assert view._grab == "marquee"

    view.handle_event(doc, _event(pygame.MOUSEBUTTONDOWN, button=2, pos=far), True)
    assert view._grab == "marquee", "the middle press switched the grab to a pan"

    view.handle_event(doc, _event(pygame.MOUSEBUTTONUP, button=1, pos=far), True)
    assert view.marquee is None, "the rectangle stayed on screen"
    assert view._grab is None
    assert len(doc.element_sel_of(doc.objects[0].uid).faces) == 6, "the sweep was dropped"


# --- clay-31: a G drag reads in metres under any tool -------------------------


@pytest.mark.parametrize("tool", ["select", "move", "rotate", "scale"])
def test_a_g_drag_reads_in_metres_whatever_tool_is_held(gl, tool) -> None:
    v = clay_view.ClayView(gl, _Ctx(tool))
    try:
        doc = _doc(count=1)
        doc.select([doc.objects[0].uid])
        v.draw(doc, (0.0, 0.0, 200.0, 200.0), 0.0)
        v._last_mouse = (100.0, 100.0)
        assert v.begin_keyboard_drag(doc, "move")
        v._drag_keyboard(doc, (140.0, 120.0))
        hud = v.drag_hud
        assert hud.rstrip().endswith(" m"), f"{tool!r}: {hud!r}"
        assert "deg" not in hud
        assert "X " in hud and "Y " in hud and "Z " in hud, f"{tool!r}: {hud!r}"
        v.cancel_drag(doc)
    finally:
        v.release()


# --- clay-62: a marquee released after the mode left an element mode ----------


def test_a_marquee_released_after_the_mode_left_an_element_mode_selects_nothing(view) -> None:
    doc = _doc(count=1)
    doc.set_element_mode("face")
    view.draw(doc, (0.0, 0.0, 200.0, 200.0), 0.0)
    view._grab = "marquee"
    view._marquee_from = (0.0, 0.0)
    view.marquee = (0.0, 0.0, 200.0, 200.0)
    view._marquee_add = "replace"
    doc.set_element_mode("object")  # the 4 key, pressed mid-marquee

    view._release_drag(doc, 1)

    assert view.marquee is None
    assert doc.element_mode == "object"
    assert not doc.element_sel, "an element selection leaked into object mode"


# --- clay-63: the drag legend ---------------------------------------------------


def test_the_drag_legend_names_only_what_drag_input_does() -> None:
    """``hint(dragging=True)`` promised "X/Y/Z lock (again: local, again: off)",
    but a second press of the locked axis only clears it -- ``DragInput`` has no
    local space -- and nothing called the branch at all: the live line is
    ``drag_readout``'s. The dead branch is gone rather than corrected."""
    from realmspinner.kernels.mesh import drag as bdrag

    entry = bdrag.DragInput()
    assert entry.key("x") and entry.axis == "x"
    assert entry.key("x") and entry.axis == "", "a second press must clear, not go local"

    assert "dragging" not in inspect.signature(clay_hints.hint).parameters
    assert "drag_kind" not in inspect.signature(clay_hints.hint).parameters
    assert not hasattr(clay_hints, "_DRAGGING")
    assert "again: local" not in inspect.getsource(clay_hints)


# --- clay-64: documentation that named things that are gone -------------------


def test_clay_view_docs_name_no_removed_caller_or_snap() -> None:
    from realmspinner.kernels.mesh import pick as bp_pick

    view_src = inspect.getsource(clay_view)
    assert "_render_clay_reference" not in view_src
    assert "trellis_path_already_got" not in view_src
    assert "Trellis caller" not in view_src
    assert "build-to-trellis" not in view_src
    drag_doc = _view_drag.__doc__ or ""
    assert "vertex snapping" not in drag_doc
    assert "exists for snapping" not in (inspect.getdoc(bp_pick.nearest_vertex) or "")
    # The renamed pin exists, and the old name does not.
    from pathlib import Path

    pin = (Path(__file__).parent / "test_clay_view.py").read_text(encoding="utf-8")
    assert "def test_render_png_defaults_are_the_picture_an_unparameterised_call_draws" in pin
    assert "trellis_path_already_got" not in pin


# --- clay-65: a crisp material's texture (needs GL) ---------------------------


@pytest.mark.parametrize("shared", ["dict", "texture_cache"])
def test_a_nearest_material_uploads_an_unmipped_nearest_texture_and_does_not_share_it_with_a_smooth_one(  # noqa: E501
    gl, shared
) -> None:
    """Palette pictures are pixel art, so ``Material.nearest`` has to reach the
    GL texture as NEAREST/NEAREST with no mip chain -- and a smooth material
    over the *same* pixels object must get its own mipmapped texture rather than
    borrow the crisp one. Asserted against the real driver's filter state; the
    fake-context tests in ``test_gpumaterial_nearest`` cannot see it."""
    import moderngl

    pixels = bytes([255, 0, 0, 255] * 16)
    image = (4, 4, pixels)
    cache: Any = {} if shared == "dict" else scenelib.TextureCache()
    crisp = scenelib.GpuMaterial(gl, gltf.Material(base_color=image, nearest=True), cache)
    smooth = scenelib.GpuMaterial(gl, gltf.Material(base_color=image, nearest=False), cache)
    try:
        crisp_tex = crisp.textures["base_color"]
        smooth_tex = smooth.textures["base_color"]
        assert crisp_tex.filter == (moderngl.NEAREST, moderngl.NEAREST)
        assert smooth_tex.filter == (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        assert crisp_tex is not smooth_tex
    finally:
        crisp.release()
        smooth.release()
