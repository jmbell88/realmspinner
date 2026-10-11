"""The 2026-10-10 audit's Clay viewport findings: clay-01, 02, 25, 55, 56, 57.

The drag and pick tests ride the session ``gl`` fixture (a real ``ClayView``
owns a renderer) and skip where there is no GPU, like every sibling here.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio import viewport_hints as clay_hints
from realmspinner.studio.modes.clay.state import ClayState
from realmspinner.studio.modes.clay.ui import view as clay_view

RECT = (0.0, 0.0, 400.0, 400.0)


class _Ctx:
    def __init__(self) -> None:
        self.state = type("S", (), {"clay": ClayState()})()
        self.clay_view = None

    def toast(self, message: str, level: str = "info") -> None:
        pass


@pytest.fixture(autouse=True)
def _headless_mods(monkeypatch):
    import pygame

    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0, raising=False)


@pytest.fixture
def ctx():
    return _Ctx()


@pytest.fixture
def view(gl, ctx):
    v = clay_view.ClayView(gl, ctx)
    ctx.clay_view = v
    v._rect = RECT
    yield v
    v.release()


def _screen_of_point(view: Any, point: Any) -> tuple[float, float]:
    mvp = view.camera.projection() @ view.camera.view()
    clip = mvp @ np.append(np.asarray(point, dtype="f8"), 1.0)
    ndc = clip[:3] / clip[3]
    return ((ndc[0] + 1) * 0.5 * RECT[2], (1 - ndc[1]) * 0.5 * RECT[3])


def _ortho_front(view: Any) -> None:
    view.camera.orthographic = True
    view.camera.look_along("front")
    view.camera.set_target(m3.vec3())
    view.camera.distance = view.camera._goal_distance = 5.0
    view.camera.update(100.0)
    view.camera.aspect = 1.0


def _side_box_doc(x: float = 3.0) -> tuple[bd.ClayDoc, Any]:
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="origin", mesh=bp.box()))
    obj = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="side", mesh=bp.box(), translation=m3.vec3(x, 0.0, 0.0))
    )
    doc.select([obj.uid])
    return doc, obj


# --- clay-01: orthographic vertex pick and occlusion --------------------------


def test_orthographic_element_pick_rejects_a_vertex_on_the_far_side_of_a_closed_box(
    view,
) -> None:
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="box", mesh=bp.box((2, 2, 2))))
    doc.element_mode = "vertex"
    hidden_clicks = {}
    for ortho in (False, True):
        view.camera.orthographic = ortho
        view.frame_selection(doc)
        view.camera.update(100.0)
        view.camera.aspect = 1.0
        screen = view.screen_of(doc, obj.uid)
        hidden = []
        for i in range(len(obj.mesh.positions)):
            click = tuple(screen.xy[i])
            hit = view.pick_face(doc, click)
            if hit is None:
                continue
            origin, direction = view._ray(click)
            surface = np.linalg.norm(origin + direction * hit.t - view.camera.position)
            if screen.depth[i] > surface + 0.05:  # the vertex is behind the visible wall
                hidden.append((i, click))
        assert hidden, "the 3/4 frame must leave a vertex behind a near wall"
        hidden_clicks[ortho] = [
            (i, view.pick_element(doc, click)) for i, click in hidden
        ]
    for ortho, picks in hidden_clicks.items():
        for i, picked in picks:
            assert picked is None or picked[1] != i, (
                f"ortho={ortho}: the hidden far-side vertex {i} was picked through the box"
            )


# --- clay-02: G/R/S measure on the view plane ---------------------------------


def test_a_horizontal_mouse_move_during_g_in_an_ortho_front_view_does_not_change_depth(
    view,
) -> None:
    doc, obj = _side_box_doc()
    _ortho_front(view)
    start = _screen_of_point(view, obj.translation)
    view._last_mouse = start
    assert view.begin_keyboard_drag(doc, "move")
    view._drag_keyboard(doc, (start[0] + 40.0, start[1]))
    z = float(obj.translation[2])
    x = float(obj.translation[0])
    view.cancel_drag(doc)
    assert x > 3.0, "the object follows the pointer across the view"
    assert abs(z) < 1e-6, f"a purely horizontal move slid the object {z:+.4f} m in depth"


def test_r_in_an_ortho_front_view_turns_about_the_view_axis_not_the_eye_line(view) -> None:
    doc, obj = _side_box_doc()
    _ortho_front(view)
    start = _screen_of_point(view, obj.translation)
    view._last_mouse = (start[0] + 60.0, start[1])
    assert view.begin_keyboard_drag(doc, "rotate")
    view._drag_keyboard(doc, (start[0], start[1] - 60.0))
    quat = np.array(obj.rotation, dtype="f8")
    view.cancel_drag(doc)
    assert abs(quat[3]) < 0.999, "the drag must have turned the object"
    # About the view axis (world Z) the quaternion's x and y parts are zero; an
    # axis along the eye-to-pivot line, which leans toward X, would not be.
    assert abs(quat[0]) < 1e-6 and abs(quat[1]) < 1e-6, quat


# --- clay-25: element hover is throttled like object hover --------------------


def test_element_hover_picks_once_per_three_pixels_of_pointer_travel(view, monkeypatch) -> None:
    doc, _obj = _side_box_doc(x=0.0)
    doc.element_mode = "vertex"
    view.frame_selection(doc)
    view.camera.update(100.0)
    calls: list[tuple[float, float]] = []
    real = view.pick_element

    def counting(d: Any, local: tuple[float, float], hit: Any = None) -> Any:
        calls.append(local)
        return real(d, local, hit)

    monkeypatch.setattr(view, "pick_element", counting)
    view._last_mouse = (100.0, 100.0)
    for i in range(10):
        view._motion(doc, (100.0 + i, 100.0))
    assert 1 <= len(calls) <= 4, f"{len(calls)} full element picks for 10 px of travel"
    # And the throttle stands down for a change the pointer did not make.
    view.clear_hover(doc)
    before = len(calls)
    view._motion(doc, (110.0, 100.0))
    assert len(calls) == before + 1, "the first motion after a hover is cleared must pick"


# --- clay-55: the readout shows what lands ------------------------------------


def test_the_drag_readout_shows_the_snapped_amount_when_snap_is_on(view, ctx, monkeypatch) -> None:
    doc, obj = _side_box_doc(x=0.0)
    obj.translation = m3.vec3(0.0, 0.0, 0.0)
    ctx.state.clay.snap = True
    ctx.state.clay.snap_translate = 0.125
    _ortho_front(view)
    view._last_mouse = (200.0, 200.0)
    assert view.begin_keyboard_drag(doc, "move")
    pivot = np.array(view._drag_origin, dtype="f8")
    monkeypatch.setattr(
        view, "_view_plane_point", lambda local, centre: pivot + np.array([0.37, 0.0, 0.0])
    )
    view._key_anchor = pivot
    view._drag_keyboard(doc, (210.0, 200.0))
    landed = float(obj.translation[0])
    hud = view.drag_hud
    view.cancel_drag(doc)
    assert math.isclose(landed, 0.375), landed
    assert "0.375" in hud and "0.370" not in hud, hud


def test_the_rotate_readout_shows_the_snapped_angle_when_snap_is_on(view, ctx, monkeypatch) -> None:
    doc, obj = _side_box_doc(x=0.0)
    ctx.state.clay.snap = True
    ctx.state.clay.snap_rotate = 15.0
    _ortho_front(view)
    view._last_mouse = (300.0, 200.0)
    assert view.begin_keyboard_drag(doc, "rotate")
    pivot = np.array(view._drag_origin, dtype="f8")
    anchor = pivot + np.array([1.0, 0.0, 0.0])
    angle = math.radians(37.0)
    now = pivot + np.array([math.cos(angle), math.sin(angle), 0.0])
    view._key_anchor = anchor
    monkeypatch.setattr(view, "_view_plane_point", lambda local, centre: now)
    view._drag_keyboard(doc, (300.0, 100.0))
    hud = view.drag_hud
    view.cancel_drag(doc)
    assert "30.0 deg" in hud or "-30.0 deg" in hud, hud


# --- clay-56: the module header tells the truth -------------------------------


def test_the_drag_module_docstring_does_not_promise_one_step_per_object() -> None:
    from realmspinner.studio.modes.clay.ui import _view_drag

    doc = " ".join((_view_drag.__doc__ or "").split())
    assert "one history step per object" not in doc
    assert "one history step for the whole drag" in doc


# --- clay-57: the statistics line is memoised on the document's rev -----------


def test_stats_is_not_recomputed_for_a_document_whose_rev_has_not_moved(monkeypatch) -> None:
    doc = bd.ClayDoc()
    for i in range(3):
        doc.add_object(bd.Obj(uid=bd.new_uid(), name=f"o{i}", mesh=bp.box()))
    first = clay_hints.stats(doc)
    walked: list[int] = []
    real = clay_hints._faces_of
    monkeypatch.setattr(clay_hints, "_faces_of", lambda mesh: walked.append(1) or real(mesh))
    assert clay_hints.stats(doc) == first
    assert not walked, "an unchanged document walked its objects again"
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="more", mesh=bp.box()))
    again = clay_hints.stats(doc)
    assert walked and again != first, "a changed document must recount"
