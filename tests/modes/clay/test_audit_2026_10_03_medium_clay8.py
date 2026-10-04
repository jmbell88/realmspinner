"""The 2026-10-03 audit's Medium Clay findings, batch 8 (clay-70..75): Clay's
viewport -- the gizmo centre, an element drag on a modified object, the bounds
memo, a commit refused mid-drag, the wheel against a keyboard drag, and Alt+click
loop select.
"""

from __future__ import annotations

# ruff: noqa: E501 - a regression test's name is the claim, and these are long
import numpy as np
import pytest

from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import modifiers as mods
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio.modes.clay.ui import view as clay_view

RECT = (0.0, 0.0, 128.0, 96.0)


class _State:
    def __init__(self, tool: str = "select") -> None:
        self.tool = tool
        self.snap = False
        self.snap_translate = 0.125
        self.snap_rotate = 15.0
        self.snap_vertex = False


class _Ctx:
    def __init__(self, tool: str = "select") -> None:
        self.state = type("S", (), {"clay": _State(tool)})()
        self.toasts: list[str] = []

    def toast(self, text: str, level: str = "info") -> None:
        self.toasts.append(text)


@pytest.fixture
def view(gl):
    v = clay_view.ClayView(gl, _Ctx())
    yield v
    v.release()


def _doc(*, count: int = 2) -> bd.ClayDoc:
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


def _face_drag_ready(view, doc, *, uids=None) -> list[int]:
    """Face mode, face 0 selected on each of *uids*, move tool, drawn once."""
    doc.set_element_mode("face")
    uids = [o.uid for o in doc.objects] if uids is None else uids
    for uid in uids:
        doc.set_element_sel(uid, el.ElementSel(faces=[0]))
    view.app_ctx.state.clay.tool = "move"
    view.frame_selection(doc)
    view.draw(doc, RECT, 0.0)
    return uids


def _world_centroid(doc, uid) -> np.ndarray:
    obj = doc.by_uid(uid)
    verts = el.affected_verts(obj.mesh, doc.element_sel_of(uid))
    return obj.mesh.positions[verts].astype("f8").mean(axis=0) + np.asarray(obj.translation, "f8")


# --- clay-70: a hidden (or collider) object's elements are not the gizmo's -----


def test_the_gizmo_centre_ignores_a_hidden_objects_element_selection(view) -> None:
    doc = _doc(count=2)
    a, b = _face_drag_ready(view, doc)
    both = view.element_centre(doc)
    assert both is not None and both[0] > 0.5, "sanity: both objects pull the centre"

    doc.set_props(b, visible=False)
    centre = view.element_centre(doc)
    assert np.allclose(centre, _world_centroid(doc, a), atol=1e-5), (
        "the pivot sits off the vertices that actually move"
    )


def test_the_gizmo_centre_ignores_a_colliders_element_selection(view) -> None:
    doc = _doc(count=2)
    a, b = _face_drag_ready(view, doc)
    doc.by_uid(b).role = "collider"
    centre = view.element_centre(doc)
    assert np.allclose(centre, _world_centroid(doc, a), atol=1e-5)


def test_an_element_drag_never_moves_the_vertices_of_a_hidden_object(view) -> None:
    doc = _doc(count=2)
    a, b = _face_drag_ready(view, doc)
    doc.set_props(b, visible=False)
    view._begin_gizmo_drag(doc)
    view._preview_element_drag(
        doc, view.element_centre(doc) + np.array([0.0, 1.0, 0.0]), view.state
    )
    assert b not in view._element_drags
    view._grab = "gizmo"
    view._release_drag(doc)
    assert np.allclose(doc.by_uid(b).mesh.positions, bp.box().positions)
    assert not np.allclose(doc.by_uid(a).mesh.positions, bp.box().positions)


# --- clay-71: a drag on an object whose modifier changes the vertex count ------


def test_an_element_drag_on_a_mirrored_object_updates_the_selection_overlay_and_the_drawn_mesh(view) -> None:
    doc = _doc(count=1)
    uid = doc.objects[0].uid
    doc.set_modifiers(uid, (mods.make("array", {"count": 2.0, "offset_x": 2.0}, id=1),))
    _face_drag_ready(view, doc)
    entry = view._cache[uid]
    assert len(doc.evaluated(uid).positions) > len(doc.by_uid(uid).mesh.positions)
    gpu = entry.gpu.draws[0][1]
    overlay = view._overlays[uid]
    gpu_before = gpu.vbo.read()
    drawn_before = np.frombuffer(gpu_before, dtype="f4").reshape(-1, 8)[:, :3].copy()
    overlay_before = overlay.pos_vbo.read()

    view._begin_gizmo_drag(doc)
    view._preview_element_drag(
        doc, view.element_centre(doc) + np.array([0.0, 1.0, 0.0]), view.state
    )

    assert overlay.pos_vbo.read() != overlay_before, "the selection overlay froze"
    entry = view._cache[uid]
    assert entry.gpu.draws[0][1].vbo.read() != gpu_before, "the drawn mesh froze"
    # Both copies of the moved face travelled, not only the first's.
    drawn = np.frombuffer(entry.gpu.draws[0][1].vbo.read(), dtype="f4").reshape(-1, 8)[:, :3]
    changed = (drawn != drawn_before).any(axis=1)
    half = len(drawn) // 2
    assert changed[:half].sum() > 0 and changed[half:].sum() == changed[:half].sum()


# --- clay-72: the bounds memo follows a reparent ------------------------------


def test_world_bounds_follows_a_reparent_that_keeps_the_local_transform(view) -> None:
    doc = bd.ClayDoc()
    child = doc.add_object(bd.Obj(uid=bd.new_uid(), name="child", mesh=bp.box()))
    parent = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="parent", mesh=bp.box(), translation=m3.vec3(10.0, 0.0, 0.0))
    )
    doc.select([child.uid])
    lo, hi = view.world_bounds(doc, selected_only=True)
    assert abs(float((lo + hi)[0] * 0.5)) < 1e-6
    assert abs(float(view.selection_centre(doc)[0])) < 1e-6

    assert doc.set_parent(child.uid, parent.uid, keep_world=False)
    centre = view.selection_centre(doc)
    assert abs(float(centre[0]) - 10.0) < 1e-6, "served the box measured under the old parent"

    doc.history.undo(doc)
    assert abs(float(view.selection_centre(doc)[0])) < 1e-6, "and its undo, the same way"


# --- clay-73: a commit refused by a lock taken mid-drag -----------------------


def test_a_commit_refused_by_a_mid_drag_lock_restores_the_object_and_closes_the_history_gesture(view) -> None:
    # The element path (``set_mesh`` refuses a locked object).
    doc = _doc(count=1)
    [uid] = _face_drag_ready(view, doc)
    view._begin_gizmo_drag(doc)
    view._preview_element_drag(
        doc, view.element_centre(doc) + np.array([0.0, 1.0, 0.0]), view.state
    )
    doc.set_props(uid, locked=True)  # a step of its own: the depth is taken after it
    depth = len(doc.history)

    view._grab = "gizmo"
    view._release_drag(doc)

    assert doc.history._open_gestures == 0, "the gesture was left open"
    assert len(doc.history) == depth
    assert np.allclose(doc.by_uid(uid).mesh.positions, bp.box().positions)
    assert uid not in view._cache, "the previewed buffers outlived the refused commit"
    assert view.app_ctx.toasts, "a refusal says why"

    # The object path (``set_transform`` refuses it).
    doc = _doc(count=1)
    uid = doc.objects[0].uid
    doc.select([uid])
    view.draw(doc, RECT, 0.0)
    view._last_mouse = (64.0, 48.0)
    before = np.array(doc.by_uid(uid).translation)
    view.begin_keyboard_drag(doc, "move")
    view._motion(doc, (100.0, 48.0))
    doc.set_props(uid, locked=True)

    view._release_drag(doc, 1)

    assert doc.history._open_gestures == 0
    assert np.allclose(doc.by_uid(uid).translation, before)


# --- clay-74: a wheel notch is not a cancel ------------------------------------


def test_a_wheel_notch_does_not_cancel_a_keyboard_drag(view) -> None:
    import pygame

    doc = _doc(count=1)
    uid = doc.objects[0].uid
    doc.select([uid])
    view.draw(doc, RECT, 0.0)
    view._last_mouse = (64.0, 48.0)
    view.begin_keyboard_drag(doc, "move")
    view._motion(doc, (100.0, 48.0))
    moved = np.array(doc.by_uid(uid).translation)
    assert not np.allclose(moved, [0.0, 0.0, 0.0])

    for button in (4, 5):
        event = pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=button, pos=(100, 48))
        view.handle_event(doc, event, True)
        assert view._grab == "keydrag", f"button {button} cancelled the drag"
        assert np.allclose(doc.by_uid(uid).translation, moved)

    # Left still commits and right still cancels: only the wheel is exempt.
    event = pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=3, pos=(100, 48))
    view.handle_event(doc, event, True)
    assert view._grab is None
    assert np.allclose(doc.by_uid(uid).translation, [0.0, 0.0, 0.0])


# --- clay-75: Alt+click loop select replaces, it does not accumulate -----------


def test_alt_click_loop_select_replaces_the_selection_in_other_objects(view) -> None:
    doc = bd.ClayDoc()
    far = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="far", mesh=bp.grid(), translation=m3.vec3(100.0, 0.0, 0.0))
    )
    near = doc.add_object(bd.Obj(uid=bd.new_uid(), name="near", mesh=bp.grid()))
    doc.element_mode = "edge"
    doc.select([far.uid, near.uid])
    doc.set_element_sel(far.uid, el.select_all(far.mesh, "edge"))
    view.camera.set_target(m3.vec3())
    view.camera.set_position(m3.vec3(0.0, 0.0, 8.0))
    view.draw(doc, RECT, 0.0)

    assert view.select_loop_at(doc, (64.0, 48.0))

    assert set(doc.selection) == {near.uid}
    assert set(doc.element_sel) == {near.uid}, (
        "an object left out of the selection still holds red edges that move with the next drag"
    )
