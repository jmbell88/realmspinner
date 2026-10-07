"""Clay's interaction pass: the G/R/S key scheme, F that keeps the angle, the wheel
that zooms toward the pointer, an outline on what is selected, a grid with axes,
the translate gizmo's plane handles, H to hide, and a vertex that lands on the
grid.

Each test's name is the claim; each fails against the code it replaced. The ones
that need a window take the ``gl`` fixture (skipped where there is no GPU); the
rest are headless.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pygame
import pytest

from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio import _view_frame
from realmspinner.studio.modes.clay import element_move
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay.ui import view as clay_view
from realmspinner.studio.viewer import gizmo as gz
from realmspinner.studio.viewer import grid
from realmspinner.studio.viewer.camera import Camera, screen_ray

from .test_clay_mode import FakeCtx

RECT = (0.0, 0.0, 200.0, 150.0)


def _key(key: int, mod: int = 0) -> Any:
    return pygame.event.Event(pygame.KEYDOWN, key=key, mod=mod)


def _two_boxes() -> bd.ClayDoc:
    doc = bd.ClayDoc()
    for i in range(2):
        doc.add_object(
            bd.Obj(
                uid=bd.new_uid(),
                name=f"box{i}",
                mesh=bp.box(),
                translation=m3.vec3(float(i) * 3.0, 0.0, 0.0),
            )
        )
    return doc


def _adopt(doc: bd.ClayDoc | None = None) -> tuple[FakeCtx, Any, bd.ClayDoc]:
    ctx = FakeCtx()
    doc = doc or _two_boxes()
    tab = clay_mode.adopt(ctx, doc, title="T")
    return ctx, tab, doc


class _State:
    def __init__(self, tool: str = "select") -> None:
        self.tool = tool
        self.snap = False
        self.snap_translate = 0.125
        self.snap_rotate = 15.0


class _AppCtx:
    def __init__(self, tool: str = "select") -> None:
        self.state = SimpleNamespace(clay=_State(tool))


@pytest.fixture
def view(gl):
    v = clay_view.ClayView(gl, _AppCtx())
    yield v
    v.release()


# --- the key scheme ------------------------------------------------------------


def test_g_r_and_s_set_the_tool_and_begin_the_drag_with_a_selection() -> None:
    ctx, _tab, doc = _adopt()
    doc.select([doc.objects[0].uid])
    started: list[str] = []
    ctx.clay_view = SimpleNamespace(
        dragging=False, begin_keyboard_drag=lambda d, kind: started.append(kind) or True
    )
    state = clay_mode.ensure(ctx)
    for key, tool in ((pygame.K_g, "move"), (pygame.K_r, "rotate"), (pygame.K_s, "scale")):
        assert clay_mode.handle_key(ctx, _key(key)) is True
        assert state.tool == tool
    assert started == ["move", "rotate", "scale"]


def test_g_r_and_s_still_set_the_tool_when_there_is_nothing_to_drag() -> None:
    ctx, _tab, _doc = _adopt()
    state = clay_mode.ensure(ctx)
    for key, tool in ((pygame.K_g, "move"), (pygame.K_s, "scale"), (pygame.K_r, "rotate")):
        clay_mode.handle_key(ctx, _key(key))
        assert state.tool == tool


def test_e_in_object_mode_starts_nothing_and_leaves_the_tool_alone() -> None:
    ctx, _tab, doc = _adopt()
    doc.select([doc.objects[0].uid])
    started: list[str] = []
    ctx.clay_view = SimpleNamespace(
        dragging=False, begin_keyboard_drag=lambda d, kind: started.append(kind) or True
    )
    state = clay_mode.ensure(ctx)
    before = len(doc.history)
    assert clay_mode.handle_key(ctx, _key(pygame.K_e)) is True
    assert started == []
    assert state.tool == "select", "E was Rotate's letter"
    assert len(doc.history) == before


def test_w_is_no_longer_a_tool_key() -> None:
    ctx, _tab, _doc = _adopt()
    state = clay_mode.ensure(ctx)
    clay_mode.handle_key(ctx, _key(pygame.K_w))
    assert state.tool == "select"


# --- the camera --------------------------------------------------------------------


def test_frame_with_keep_angles_keeps_the_view_direction_and_defaults_do_not() -> None:
    cam = Camera()
    cam.look_along("front")
    cam.frame(np.array([-1.0, 0.0, -1.0]), np.array([1.0, 2.0, 1.0]), keep_angles=True)
    assert cam.theta == pytest.approx(0.0)
    assert cam.phi == pytest.approx(math.pi / 2)
    assert cam._goal_theta == pytest.approx(0.0)

    other = Camera()
    other.look_along("front")
    other.frame(np.array([-1.0, 0.0, -1.0]), np.array([1.0, 2.0, 1.0]))
    assert other.theta == pytest.approx(math.atan2(0.62, 0.62)), "the default still resets"


def test_frame_with_keep_angles_is_a_function_of_the_box_not_the_old_target() -> None:
    lo, hi = np.array([-1.0, 0.0, -1.0]), np.array([1.0, 2.0, 1.0])
    a, b = Camera(), Camera()
    a.look_along("right")
    b.look_along("right")
    b.set_target(np.array([50.0, 9.0, -30.0]))
    a.frame(lo, hi, keep_angles=True)
    b.frame(lo, hi, keep_angles=True)
    assert np.allclose(a.position, b.position)


def test_frame_min_zoom_lets_the_camera_come_closer_than_the_viewer_default() -> None:
    lo, hi = np.array([-1.0, 0.0, -1.0]), np.array([1.0, 2.0, 1.0])
    viewer, clay = Camera(), Camera()
    radius = viewer.frame(lo, hi)
    clay.frame(lo, hi, min_zoom=0.02)
    assert viewer.min_distance == pytest.approx(radius * 0.5)
    assert clay.min_distance == pytest.approx(radius * 0.02)


def test_f_in_clay_keeps_the_view_direction(view) -> None:
    doc = _two_boxes()
    view.camera.look_along("top")
    view.frame_selection(doc)
    assert view.camera.phi == pytest.approx(1e-6)
    view.camera.look_along("right")
    view.frame_selection(doc)
    assert view.camera.theta == pytest.approx(math.pi / 2)
    assert view.camera.phi == pytest.approx(math.pi / 2)
    assert view.camera.min_distance < view.radius * 0.1


@pytest.mark.parametrize("ortho", [False, True])
def test_wheel_dolly_with_a_cursor_keeps_the_world_point_under_it(ortho: bool) -> None:
    cam = Camera(aspect=4.0 / 3.0)
    cam.orthographic = ortho
    cam.set_target(np.array([1.0, 0.5, -2.0]))
    cam.distance = cam._goal_distance = 6.0
    cam.min_distance, cam.max_distance = 0.01, 100.0
    size = (400.0, 300.0)
    cursor = (310.0, 90.0)

    origin, direction = screen_ray(cam, *cursor, int(size[0]), int(size[1]))
    forward = -cam.view()[2, :3]
    t = float(np.dot(cam.target - origin, forward)) / float(np.dot(direction, forward))
    point = origin + direction * t

    cam.dolly(-4, cursor=cursor, size=size)  # a negative step is a zoom in
    cam.dolly(-2, cursor=cursor, size=size)
    for _ in range(600):
        cam.update(1.0 / 60.0)
    assert cam.distance < 6.0 * 0.8, "it did zoom in"

    origin, direction = screen_ray(cam, *cursor, int(size[0]), int(size[1]))
    miss = np.linalg.norm(np.cross(point - origin, direction))
    assert miss < 1e-3, "the point the cursor was over slid away"


def test_wheel_dolly_without_a_cursor_is_the_plain_dolly() -> None:
    plain, cursor = Camera(), Camera()
    plain.dolly(3)
    cursor.dolly(3, cursor=None, size=None)
    assert plain._goal_distance == cursor._goal_distance
    assert np.allclose(plain._goal_target, cursor._goal_target)


def test_the_clay_wheel_zooms_toward_the_pointer(view) -> None:
    doc = _two_boxes()
    view.frame_selection(doc)
    view._rect = RECT
    for _ in range(300):
        view.camera.update(1.0 / 60.0)
    before = view.camera._goal_target.copy()
    event = pygame.event.Event(pygame.MOUSEWHEEL, y=3, pos=(190, 20))
    # ``_local`` reads ``pos``; a wheel event carries none in pygame, so the
    # last mouse position stands in.
    view._last_mouse = (190.0, 20.0)
    assert view.handle_event(doc, event, True) is True
    assert not np.allclose(view.camera._goal_target, before), "a plain dolly never moves the target"


def test_a_keypad_digit_in_brackets_is_the_same_axis_view_as_the_row_digit() -> None:
    cam = Camera()
    assert _view_frame.axis_view_key(cam, "[1]", False) is True
    assert (cam.theta, cam.phi) == (0.0, math.pi / 2)
    assert _view_frame.axis_view_key(cam, "[3]", True) is True
    assert cam.theta == pytest.approx(-math.pi / 2), "Shift is the opposite side"
    assert _view_frame.axis_view_key(cam, "[5]", False) is True
    assert cam.orthographic
    assert _view_frame.axis_view_key(cam, "[2]", False) is False
    assert set(clay_mode.AXIS_VIEW_KEYS) == {"1", "3", "7"}


def test_bare_keypad_1_3_7_5_and_period_drive_the_view_in_clay() -> None:
    ctx, _tab, _doc = _adopt()
    ctx.clay_view = SimpleNamespace(dragging=False, camera=Camera())
    state = clay_mode.ensure(ctx)

    clay_mode.handle_key(ctx, _key(pygame.K_KP1))
    assert ctx.clay_view.camera.theta == 0.0
    clay_mode.handle_key(ctx, _key(pygame.K_KP7))
    assert ctx.clay_view.camera.phi < 0.01
    clay_mode.handle_key(ctx, _key(pygame.K_KP3, pygame.KMOD_SHIFT))
    assert ctx.clay_view.camera.theta == pytest.approx(-math.pi / 2)
    clay_mode.handle_key(ctx, _key(pygame.K_KP5))
    assert ctx.clay_view.camera.orthographic
    assert not state.frame_pending
    clay_mode.handle_key(ctx, _key(pygame.K_KP_PERIOD))
    assert state.frame_pending


# --- element_move --------------------------------------------------------------------


def _face_selected(doc: bd.ClayDoc, objects: list[Any]) -> None:
    doc.set_element_mode("face")
    for obj in objects:
        doc.set_element_sel(obj.uid, el.ElementSel(faces=[0]))


def test_move_elements_moves_every_objects_selection_as_one_undo_step() -> None:
    doc = _two_boxes()
    _face_selected(doc, doc.objects)
    before = [o.mesh.positions.copy() for o in doc.objects]
    depth = len(doc.history)

    assert element_move.move_elements(doc, np.array([0.0, 1.0, 0.0])) is True
    for obj, old in zip(doc.objects, before, strict=True):
        assert not np.allclose(obj.mesh.positions, old)
    assert len(doc.history) == depth + 1, "two objects, one step"

    doc.undo()
    for obj, old in zip(doc.objects, before, strict=True):
        assert np.allclose(obj.mesh.positions, old)


def test_move_elements_moves_by_the_world_delta_through_the_objects_placement() -> None:
    doc = _two_boxes()
    obj = doc.objects[1]
    turn = m3.quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), math.pi / 2)
    doc.set_transform(obj.uid, rotation=turn)
    _face_selected(doc, [obj])
    before = element_move.element_median_world(doc)
    element_move.move_elements(doc, np.array([2.0, 0.0, 0.0]))
    after = element_move.element_median_world(doc)
    assert np.allclose(after - before, [2.0, 0.0, 0.0])


def test_a_zero_move_pushes_nothing() -> None:
    doc = _two_boxes()
    _face_selected(doc, doc.objects)
    depth = len(doc.history)
    assert element_move.move_elements(doc, np.zeros(3)) is False
    assert element_move.move_elements(doc, np.zeros(3), snap_step=0.125) is False
    assert len(doc.history) == depth


def test_move_elements_skips_a_hidden_object() -> None:
    doc = _two_boxes()
    _face_selected(doc, doc.objects)
    hidden = doc.objects[1]
    doc.set_visibility({hidden.uid: False})
    old = hidden.mesh.positions.copy()
    element_move.move_elements(doc, np.array([0.0, 1.0, 0.0]))
    assert np.allclose(hidden.mesh.positions, old)
    assert not np.allclose(doc.objects[0].mesh.positions, bp.box().positions)


def test_move_elements_with_a_snap_step_puts_each_vertex_on_the_world_grid() -> None:
    doc = _two_boxes()
    doc.set_transform(doc.objects[0].uid, translation=m3.vec3(0.03, 0.0, 0.0))
    _face_selected(doc, [doc.objects[0]])
    element_move.move_elements(doc, np.array([0.1, 0.1, 0.1]), snap_step=0.125)
    obj = doc.objects[0]
    from realmspinner.kernels.mesh.elements import affected_verts

    verts = affected_verts(obj.mesh, doc.element_sel_of(obj.uid))
    local = obj.mesh.positions[verts].astype("f8")
    world = element_move.apply_affine(doc.world_matrix(obj.uid), local)
    assert np.allclose(world / 0.125, np.round(world / 0.125), atol=1e-5)


def test_the_vectorised_snap_breaks_ties_away_from_zero_like_snap_value() -> None:
    from realmspinner.kernels.mesh.ops import snap_value

    values = np.array([-1.5, -0.5, -0.0625, 0.0625, 0.5, 1.5, 2.5, 0.34, -0.34])
    got = element_move.snap_points(values, 0.125 * 4)
    want = [snap_value(float(v), 0.5) for v in values]
    assert np.allclose(got, want)
    assert np.array_equal(element_move.snap_points(values, 0.0), values)


def test_element_median_world_is_the_gizmo_centre(view) -> None:
    doc = _two_boxes()
    _face_selected(doc, doc.objects)
    assert np.allclose(element_move.element_median_world(doc), view.element_centre(doc))
    doc.clear_element_sel()
    assert element_move.element_median_world(doc) is None


def test_a_snapped_element_drag_lands_each_moved_vertex_on_the_grid(view) -> None:
    doc = _two_boxes()
    obj = doc.objects[0]
    doc.set_transform(obj.uid, translation=m3.vec3(0.03, 0.0, 0.0))
    _face_selected(doc, [obj])
    state = view.app_ctx.state.clay
    state.tool, state.snap = "move", True
    view._rect = RECT
    view._begin_gizmo_drag(doc)
    centre = view._element_centre
    view._preview_element_drag(doc, centre + np.array([0.07, 0.11, 0.0]), state)

    drag = view._element_drags[obj.uid]
    moved = drag.preview[drag.verts].astype("f8")
    world = element_move.apply_affine(drag.matrix, moved)
    assert np.allclose(world / 0.125, np.round(world / 0.125), atol=1e-4)

    state.snap = False
    view._preview_element_drag(doc, centre + np.array([0.07, 0.11, 0.0]), state)
    world = element_move.apply_affine(drag.matrix, drag.preview[drag.verts].astype("f8"))
    assert not np.allclose(world / 0.125, np.round(world / 0.125), atol=1e-4)
    view.cancel_drag(doc)


# --- selection feedback -------------------------------------------------------------------


def test_a_selected_object_draws_an_outline_in_object_mode(view) -> None:
    doc = _two_boxes()
    view.frame_selection(doc)
    view.sync(doc)
    assert view._object_overlays(doc) == [], "nothing selected, nothing outlined"

    doc.select([doc.objects[0].uid])
    items = view._object_overlays(doc)
    assert items, "a selected object draws an outline"
    assert all(tuple(i.color) == tuple(clay_view_overlay().OBJECT_SEL_COLOR) for i in items)

    doc.select([])
    assert view._object_overlays(doc) == []
    assert view._obj_overlays == {}, "the outline's GL objects are released with it"


def clay_view_overlay() -> Any:
    from realmspinner.studio.modes.clay.ui import _view_overlay

    return _view_overlay


def test_an_element_mode_draws_no_object_outline(view) -> None:
    doc = _two_boxes()
    doc.select([doc.objects[0].uid])
    view.sync(doc)
    assert view._object_overlays(doc)
    doc.set_element_mode("face")
    assert view._object_overlays(doc) == []
    assert view._obj_overlays == {}


def test_the_outline_survives_a_full_draw(view) -> None:
    doc = _two_boxes()
    doc.select([doc.objects[0].uid])
    view.frame_selection(doc)
    view.draw(doc, RECT, 0.0)
    assert view._obj_overlays


def test_hover_object_follows_the_pointer_and_redraws_only_when_the_uid_changes(view) -> None:
    doc = _two_boxes()
    view.frame_selection(doc)
    view.draw(doc, RECT, 0.0)
    uid = doc.objects[0].uid

    # Find where the first box sits on screen by sweeping a row of pixels.
    spots = [
        (x, RECT[3] / 2)
        for x in range(0, int(RECT[2]), 4)
        if view.pick(doc, (float(x), RECT[3] / 2)) == uid
    ]
    assert spots, "the framed box is on screen"
    x0, y0 = spots[len(spots) // 2]

    view._render_dirty = False
    view._motion(doc, (float(x0), y0))
    assert view.hover_object == uid
    assert view._render_dirty is True

    # Still over the same object, and inside the throttle step: no redraw.
    view._render_dirty = False
    view._motion(doc, (float(x0) + 1.0, y0))
    assert view.hover_object == uid
    assert view._render_dirty is False

    # Off the object into empty space: the hover clears and that is a redraw.
    view._render_dirty = False
    view._motion(doc, (RECT[2] - 1.0, 1.0))
    assert view.hover_object is None
    assert view._render_dirty is True

    view.draw(doc, RECT, 0.0)  # hover drawing must not raise either way
    view.hover_object = uid
    view._render_dirty = True
    view.draw(doc, RECT, 0.0)
    assert view._obj_overlays


def test_a_select_tool_gizmo_drag_commits_as_a_move(view) -> None:
    doc = _two_boxes()
    obj = doc.objects[0]
    doc.select([obj.uid])
    view.app_ctx.state.clay.tool = "select"
    view._rect = RECT
    assert view.active_gizmo(doc) is view.translate_gizmo

    view._begin_gizmo_drag(doc)
    obj.translation = np.array([1.0, 0.0, 0.0])
    view._release_drag(doc)
    assert doc.history.top is not None
    assert doc.history.top.label == "Move"


# --- the grid and the gizmo -----------------------------------------------------------------


def test_axes_geometry_is_a_red_x_line_and_a_blue_z_line_and_an_origin_marker() -> None:
    lines, marker = grid.axes_geometry(100.0)
    red, blue = grid._rgb(gz.AXIS_COLORS["x"]), grid._rgb(gz.AXIS_COLORS["z"])
    assert np.allclose(lines[:2, :3], [[-50.0, 0, 0], [50.0, 0, 0]])
    assert np.allclose(lines[:2, 3:], [red, red])
    assert np.allclose(lines[2:, :3], [[0, 0, -50.0], [0, 0, 50.0]])
    assert np.allclose(lines[2:, 3:], [blue, blue])
    assert len(marker) >= 4 and np.allclose(marker[:, :3].mean(axis=0), 0.0)


def test_set_axes_adds_axis_draws_without_changing_build(gl) -> None:
    before = grid.build(100.0, 100)
    g = grid.Grid(gl, _programs(gl))
    try:
        g.set_span(100.0, 100)
        assert g.axes is False and g._axes_vao is None
        g.set_axes(True)
        view = m3.identity()
        proj = m3.perspective(45.0, 1.0, 0.1, 100.0)
        g.render(view, proj)
        assert g._axes_vao is not None and g._marker_vao is not None
        after = grid.build(100.0, 100)
        assert np.array_equal(before[0], after[0]) and np.array_equal(before[1], after[1])
    finally:
        g.release()
    assert g._axes_vao is None


def _programs(gl: Any) -> Any:
    from realmspinner.studio.viewer.programs import ProgramCache

    return ProgramCache(gl)


def test_the_clay_viewport_turns_axes_on_for_its_draw_only(view) -> None:
    doc = _two_boxes()
    view.frame_selection(doc)
    seen: list[bool] = []
    original = view.renderer.grid.render

    def spy(v: Any, p: Any) -> None:
        seen.append(view.renderer.grid.axes)
        original(v, p)

    view.renderer.grid.render = spy
    view.draw(doc, RECT, 0.0)
    assert seen == [True]
    assert view.renderer.grid.axes is False, "render_png draws the grid it always did"


def _plane_gizmo(planes: bool) -> gz.TranslateGizmo:
    g = gz.TranslateGizmo(None, {}, planes=planes)
    g.origin = np.zeros(3)
    g.scale = 1.0
    return g


def _ray_at(point: Any, direction: Any = (0.0, 0.0, -1.0)) -> tuple[np.ndarray, np.ndarray]:
    d = np.array(direction, dtype="f8")
    return np.array(point, dtype="f8") - d * 20.0, d


def test_a_ray_through_a_plane_handle_hits_it_only_when_planes_are_on() -> None:
    inside = (0.4, 0.4, 0.0)
    assert _plane_gizmo(True).hit(*_ray_at(inside)) == "xy"
    assert _plane_gizmo(False).hit(*_ray_at(inside)) is None, "Mason and Poser are unchanged"
    assert _plane_gizmo(True).hit(*_ray_at((0.4, 0.0, 0.4), (0.0, -1.0, 0.0))) == "xz"
    assert _plane_gizmo(True).hit(*_ray_at((0.0, 0.4, 0.4), (-1.0, 0.0, 0.0))) == "yz"
    assert _plane_gizmo(True).hit(*_ray_at((0.2, 0.2, 0.0))) is None, "the corner by the origin"
    assert _plane_gizmo(True).hit(*_ray_at((0.9, 0.9, 0.0))) is None, "past the square"


def test_the_arrow_wins_over_the_plane_handle_it_borders() -> None:
    assert _plane_gizmo(True).hit(*_ray_at((0.5, 0.0, 0.0), (0.0, -1.0, 0.0))) == "x"


def test_a_plane_handle_drag_returns_the_gizmos_new_position_in_that_plane() -> None:
    g = _plane_gizmo(True)
    assert g.begin("xy", *_ray_at((0.4, 0.4, 0.0)))
    moved = g.update(*_ray_at((1.4, 2.4, 0.0)))
    assert np.allclose(moved, [1.0, 2.0, 0.0])
    assert np.allclose(g.origin, [1.0, 2.0, 0.0])
    g.end_drag()
    assert g.drag is None


# --- hide ---------------------------------------------------------------------------------


def test_h_hides_the_selected_objects_and_clears_the_selection() -> None:
    ctx, _tab, doc = _adopt()
    a, b = doc.objects
    doc.select([a.uid])
    depth = len(doc.history)
    assert clay_mode.handle_key(ctx, _key(pygame.K_h)) is True
    assert not a.visible and b.visible
    assert not doc.selection
    assert len(doc.history) == depth + 1, "hiding is one undo step"


def test_shift_h_isolates_the_selection_and_alt_h_shows_everything() -> None:
    ctx, _tab, doc = _adopt()
    a, b = doc.objects
    doc.select([a.uid])
    clay_mode.handle_key(ctx, _key(pygame.K_h, pygame.KMOD_SHIFT))
    assert a.visible and not b.visible
    assert doc.selection == {a.uid}, "isolating keeps what was isolated selected"

    clay_mode.handle_key(ctx, _key(pygame.K_h, pygame.KMOD_ALT))
    assert a.visible and b.visible


def test_alt_h_works_with_nothing_selected_and_h_with_nothing_selected_is_a_no_op() -> None:
    ctx, _tab, doc = _adopt()
    a, b = doc.objects
    doc.set_visibility({a.uid: False, b.uid: False})
    depth = len(doc.history)
    clay_mode.handle_key(ctx, _key(pygame.K_h))
    assert len(doc.history) == depth
    clay_mode.handle_key(ctx, _key(pygame.K_h, pygame.KMOD_ALT))
    assert a.visible and b.visible


def test_hide_keys_are_refused_while_the_tab_is_saving() -> None:
    ctx, tab, doc = _adopt()
    doc.select([doc.objects[0].uid])
    tab.saving = True
    for mod in (0, pygame.KMOD_SHIFT, pygame.KMOD_ALT):
        assert clay_mode.handle_key(ctx, _key(pygame.K_h, mod)) is True
    assert all(o.visible for o in doc.objects)
