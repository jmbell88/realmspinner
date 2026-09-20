"""Clay's knife: arming a bare op fire, the click-drag line, and the plane
it cuts with.

``kernels/mesh/ops_model.knife`` (untouched by this file) already cuts a
selection with a ``point``/``normal`` plane -- see ``test_ops_model.py``.
What is new here is everything between a user firing the "Knife" row with no
plane in hand and that call landing: the row arms the viewport
(``clay_ops._knife`` reaching ``ctx.clay_view``) instead of refusing,
``ClayView`` runs a click-drag gesture (``ui/_view_drag.py``'s
``DragOps.begin_knife``/``_press_knife``/``_commit_knife``), and the plane
that gesture computes is converted into each selected object's own local
frame before the cut ever runs.

Driven the same headless way ``test_clay_view.py``'s own drag tests are: a
real ``ClayView`` over the ``gl`` fixture's context, fed synthetic pygame
events, with a fake app ctx standing in for the window.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio.modes.clay import ops as clay_ops
from realmspinner.studio.modes.clay.ui import view as clay_view

RECT = (0.0, 0.0, 128.0, 96.0)


class _Toasts:
    def __init__(self) -> None:
        self.errors: list[str] = []


class _State:
    def __init__(self, tool: str = "select") -> None:
        self.tool = tool
        self.snap = False
        self.snap_translate = 0.125
        self.snap_rotate = 15.0
        self.snap_vertex = False


class _Ctx:
    """The app ctx every op fires through. ``clay_view`` starts ``None`` and
    is wired onto the real instance right after it is built -- the same
    two-step the running app follows (a ``ClayView`` is constructed with the
    ctx before anything can point back at it) -- so ``clay_ops._knife``'s own
    ``getattr(ctx, "clay_view", None)`` door resolves to it.
    """

    def __init__(self) -> None:
        self.state = type("S", (), {"clay": _State()})()
        self.toasts = _Toasts()
        self.clay_view: Any = None

    def toast(self, message: str, level: str = "info") -> None:
        if level == "error":
            self.toasts.errors.append(message)


@pytest.fixture
def view(gl):
    ctx = _Ctx()
    v = clay_view.ClayView(gl, ctx)
    ctx.clay_view = v
    yield v
    v.release()


@pytest.fixture(autouse=True)
def headless_mods(monkeypatch):
    """``pygame.key.get_mods`` raises with no display; stub it at zero. Not
    read anywhere on the knife's own path, but every other drag in this
    package needs it, so it is included for the same reason
    ``test_clay_view.py`` makes it autouse: a test that forgot would exercise
    the fallback rather than the code.
    """
    import pygame

    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0, raising=False)
    return monkeypatch


def _doc_with_box() -> tuple[bd.ClayDoc, int]:
    doc = bd.ClayDoc()
    obj = bd.Obj(uid=bd.new_uid(), name="box", mesh=bp.box())
    doc.add_object(obj)
    return doc, obj.uid


def _select_all_faces(doc: bd.ClayDoc, uid: int) -> None:
    doc.set_element_mode("face")
    n = bm.face_count(doc.by_uid(uid).mesh)
    doc.set_element_sel(uid, el.ElementSel(faces=np.arange(n, dtype="i4")))


def _centre() -> tuple[int, int]:
    return (int(RECT[2] * 0.5), int(RECT[3] * 0.5))


def _press(view, doc, pos, button: int = 1) -> None:
    import pygame

    view.handle_event(
        doc, pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=button, pos=pos), True
    )


def _motion(view, doc, pos) -> None:
    import pygame

    view.handle_event(doc, pygame.event.Event(pygame.MOUSEMOTION, pos=pos), True)


def _release(view, doc, pos, button: int = 1) -> None:
    import pygame

    view.handle_event(
        doc, pygame.event.Event(pygame.MOUSEBUTTONUP, button=button, pos=pos), True
    )


# --- arming -------------------------------------------------------------


def test_firing_knife_with_no_line_arms_the_viewport_instead_of_refusing(view) -> None:
    doc, uid = _doc_with_box()
    _select_all_faces(doc, uid)
    depth = len(doc.history)

    ran = clay_ops.run(view.app_ctx, doc, clay_ops.get("knife"))

    # Armed, not refused: no toast, nothing pushed, and ``run`` reports "did
    # not run" (there is nothing yet to undo) rather than an error.
    assert ran is False
    assert view.app_ctx.toasts.errors == []
    assert view._knife_armed is True
    assert view._grab is None
    assert view._knife_from is None and view._knife_to is None
    assert len(doc.history) == depth


def test_a_second_bare_fire_while_armed_does_not_restart_the_gesture(view) -> None:
    """``begin_knife`` refuses to arm over itself -- see its own docstring on
    why stomping a live grab is unsound. Firing the row twice in a row (a
    double-click on the tools-pane button) should not silently reset
    whatever the first arm was doing."""
    doc, uid = _doc_with_box()
    _select_all_faces(doc, uid)

    assert clay_ops.run(view.app_ctx, doc, clay_ops.get("knife")) is False
    assert view._knife_armed is True

    ran_again = clay_ops.run(view.app_ctx, doc, clay_ops.get("knife"))
    assert ran_again is False
    # Still armed (not un-armed by the second fire), and still toast-free --
    # ``begin_knife`` returning False is swallowed the same way the first
    # call's True was, not raised as a fresh refusal.
    assert view._knife_armed is True


def test_firing_the_op_with_no_view_still_toasts_the_existing_sentence() -> None:
    """No ``clay_view`` at all -- a headless script driving Clay, or exactly
    the fake ctx every other refusal in ``test_clay_ops.py`` uses -- and the
    row has nothing to arm, so it keeps the refusal it has always raised."""
    doc, uid = _doc_with_box()
    _select_all_faces(doc, uid)
    depth = len(doc.history)

    ctx = _Ctx()  # .clay_view left None

    ok = clay_ops.run(ctx, doc, clay_ops.get("knife"))

    assert ok is False
    assert ctx.toasts.errors == ["Draw a knife cut across the selected faces first."]
    assert len(doc.history) == depth


# --- the drag itself ------------------------------------------------------


def test_a_drag_across_selected_faces_cuts_them_in_one_undo_step(view) -> None:
    doc, uid = _doc_with_box()
    _select_all_faces(doc, uid)
    view.frame_selection(doc)
    view.draw(doc, RECT, 0.0)

    before_faces = bm.face_count(doc.by_uid(uid).mesh)
    depth = len(doc.history)

    assert clay_ops.run(view.app_ctx, doc, clay_ops.get("knife")) is False
    assert view._knife_armed is True

    centre = _centre()
    _press(view, doc, centre)
    assert view._grab == "knife"
    assert view._knife_armed is False

    _motion(view, doc, (centre[0] + 30, centre[1] + 20))
    _release(view, doc, (centre[0] + 30, centre[1] + 20))

    assert view._grab is None
    after = doc.by_uid(uid)
    assert bm.face_count(after.mesh) > before_faces, "the drag actually cut the box"
    assert len(doc.history) == depth + 1, "one drag, one Ctrl+Z"


def test_a_zero_length_drag_cancels_rather_than_cutting(view) -> None:
    doc, uid = _doc_with_box()
    _select_all_faces(doc, uid)
    view.frame_selection(doc)
    view.draw(doc, RECT, 0.0)

    before_mesh = doc.by_uid(uid).mesh
    depth = len(doc.history)

    clay_ops.run(view.app_ctx, doc, clay_ops.get("knife"))
    centre = _centre()
    _press(view, doc, centre)
    _release(view, doc, centre)  # press and release in the same spot

    assert view._grab is None
    assert view._knife_armed is False
    assert doc.by_uid(uid).mesh is before_mesh, "nothing was cut"
    assert len(doc.history) == depth, "nothing was pushed"


def test_escape_cancels_the_gesture_with_nothing_pushed(view) -> None:
    doc, uid = _doc_with_box()
    _select_all_faces(doc, uid)
    view.frame_selection(doc)
    view.draw(doc, RECT, 0.0)

    before_mesh = doc.by_uid(uid).mesh
    depth = len(doc.history)

    # Armed, not yet pressed -- ``cancel_drag`` is the door a right-click
    # (``_press_knife``) and the real Esc key (once ``dragging`` -- see its
    # own docstring -- has actually gone live) both go through.
    clay_ops.run(view.app_ctx, doc, clay_ops.get("knife"))
    assert view.cancel_drag(doc) is True
    assert view._knife_armed is False
    assert view._grab is None

    # And mid-line: armed, pressed, dragged, then cancelled before release.
    clay_ops.run(view.app_ctx, doc, clay_ops.get("knife"))
    centre = _centre()
    _press(view, doc, centre)
    _motion(view, doc, (centre[0] + 40, centre[1]))
    assert view._grab == "knife"
    assert view.cancel_drag(doc) is True

    assert view._grab is None
    assert view._knife_from is None and view._knife_to is None
    assert doc.by_uid(uid).mesh is before_mesh
    assert len(doc.history) == depth


def test_a_right_click_cancels_instead_of_opening_the_context_menu(view) -> None:
    doc, uid = _doc_with_box()
    _select_all_faces(doc, uid)
    view.frame_selection(doc)
    view.draw(doc, RECT, 0.0)
    before_mesh = doc.by_uid(uid).mesh

    clay_ops.run(view.app_ctx, doc, clay_ops.get("knife"))
    centre = _centre()
    _press(view, doc, centre)
    _motion(view, doc, (centre[0] + 40, centre[1]))
    assert view._grab == "knife"

    _press(view, doc, centre, button=3)
    assert view._grab is None
    assert view._knife_armed is False
    assert view.menu_request is None, "a knife-cancelling right-click is not a menu request"

    # The button-3 *release* that follows is the ordinary context-menu path
    # (``studio/_view_frame.py``'s ``_rmb_release``); with ``_rmb_at`` never
    # set by ``_press_knife``, it still opens nothing.
    _release(view, doc, centre, button=3)
    assert view.menu_request is None
    assert doc.by_uid(uid).mesh is before_mesh


# --- the plane, for a rotated and parented object --------------------------


def test_the_plane_is_right_for_a_rotated_parented_object(view) -> None:
    """The independent check: transform the *cut* box's own vertices to
    world space through ``doc.world_matrix`` -- the document's own generic
    door, not anything this feature added -- and confirm they straddle the
    plane a front-on, purely horizontal screen drag defines by camera
    geometry alone (``Camera.position``'s own formula: the eye sits
    ``distance`` back from ``target`` along ``-forward``, so at screen
    centre the unprojected point is exactly ``target``, and a horizontal
    drag's world direction is exactly ``right`` -- see ``Camera.AXIS_VIEWS``
    for the front view used here). That sidesteps re-deriving the local-
    frame conversion :func:`~.clay.ops._knife` itself performs: whether the
    parent's rotation or its translation was dropped, composed by hand, or
    the wrong half (point vs. normal) used the wrong door, the cut vertices
    would land somewhere other than on *this* independently-known plane.
    """
    parent = bd.Obj(
        uid=bd.new_uid(),
        name="parent",
        mesh=bp.box(),
        translation=m3.vec3(3.0, 0.0, 5.0),
        rotation=m3.quat_from_axis_angle(m3.vec3(0.0, 0.0, 1.0), math.pi / 2),
    )
    child = bd.Obj(uid=bd.new_uid(), name="child", mesh=bp.box(), parent=parent.uid)
    doc = bd.ClayDoc()
    doc.add_object(parent)
    doc.add_object(child)
    _select_all_faces(doc, child.uid)

    # Front view, by hand: theta=0, phi=pi/2 puts the eye on +Z from the
    # target (``Camera.position``'s own formula collapses to
    # ``target + (0, 0, distance)`` there), looking down -Z, with
    # world +X as "right" and world +Y as "up" -- the same simple case
    # ``Camera.AXIS_VIEWS["front"]`` names.
    view.camera.target = m3.vec3(3.0, 0.0, 5.0)
    view.camera.distance = 5.0
    view.camera.theta = 0.0
    view.camera.phi = math.pi / 2
    view.draw(doc, RECT, 0.0)

    clay_ops.run(view.app_ctx, doc, clay_ops.get("knife"))
    centre = _centre()  # exactly screen centre: ndc (0, 0)
    _press(view, doc, centre)
    _motion(view, doc, (centre[0] + 40, centre[1]))  # purely horizontal
    _release(view, doc, (centre[0] + 40, centre[1]))

    assert view._grab is None, "the drag committed"

    expected_point = np.array([3.0, 0.0, 5.0])  # == camera.target, by the geometry above
    expected_normal = np.array([0.0, 1.0, 0.0])  # "up", regardless of its sign

    world = np.asarray(doc.world_matrix(child.uid), dtype="f8")
    positions = doc.by_uid(child.uid).mesh.positions.astype("f8")
    world_positions = (world[:3, :3] @ positions.T).T + world[:3, 3]
    signed = (world_positions - expected_point) @ expected_normal

    assert np.any(signed > 1e-4) and np.any(signed < -1e-4), (
        "the cut split the geometry across the expected plane"
    )
    assert np.min(np.abs(signed)) < 1e-4, (
        "some vertex the cut introduced lands exactly on the expected plane"
    )
    assert bm.face_count(doc.by_uid(child.uid).mesh) > bm.face_count(bp.box())


# --- multiple objects -------------------------------------------------------


def test_every_object_with_a_face_selection_is_cut_by_the_same_world_plane() -> None:
    """The honest reading the op's own hint and docstring settle on: one
    line drawn, one plane, applied to every selected object's own local
    frame -- not one plane per object. A horizontal world plane through
    ``y=0`` crosses both boxes' own vertical middle regardless of where each
    sits along X, which is what lets this run with no viewport at all.
    """
    doc = bd.ClayDoc()
    a = bd.Obj(uid=bd.new_uid(), name="a", mesh=bp.box(), translation=m3.vec3(-2.0, 0.0, 0.0))
    b = bd.Obj(uid=bd.new_uid(), name="b", mesh=bp.box(), translation=m3.vec3(2.0, 0.0, 0.0))
    doc.add_object(a)
    doc.add_object(b)
    _select_all_faces(doc, a.uid)
    doc.set_element_sel(b.uid, el.ElementSel(faces=np.arange(bm.face_count(b.mesh), dtype="i4")))

    before_a = bm.face_count(doc.by_uid(a.uid).mesh)
    before_b = bm.face_count(doc.by_uid(b.uid).mesh)
    depth = len(doc.history)

    ctx = _Ctx()
    ok = clay_ops.run(
        ctx, doc, clay_ops.get("knife"), point=(0.0, 0.0, 0.0), normal=(0.0, 1.0, 0.0)
    )

    assert ok is True
    assert bm.face_count(doc.by_uid(a.uid).mesh) > before_a
    assert bm.face_count(doc.by_uid(b.uid).mesh) > before_b
    assert len(doc.history) == depth + 1, "both cuts fold into the one op call"
