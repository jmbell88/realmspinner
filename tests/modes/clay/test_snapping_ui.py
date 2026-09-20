"""Tranche 3 (scene structure): edge and face snap targets.

Both reuse the same screen-space picking and ray casting the viewport already
does for hover and click -- ``_snap_edge`` finds *which* edge under the
cursor the way ``pick_element`` does, then lands on the closest point along
it; ``_snap_face`` is a plain object-mode ray hit. Neither existed before this
tranche, so both tests fail with an ``AttributeError`` against the unfixed
view.
"""

from __future__ import annotations

import numpy as np
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh.adjacency import adjacency
from realmspinner.studio.modes.clay.ui import view as clay_view


class _State:
    def __init__(self) -> None:
        self.tool = "select"
        self.snap = False
        self.snap_translate = 0.125
        self.snap_rotate = 15.0
        self.snap_vertex = False
        self.snap_edge = False
        self.snap_face = False


class _Ctx:
    def __init__(self) -> None:
        self.state = type("S", (), {"clay": _State()})()


@pytest.fixture
def view(gl):
    v = clay_view.ClayView(gl, _Ctx())
    yield v
    v.release()


RECT = (0.0, 0.0, 256.0, 192.0)


def _framed_box(view) -> tuple[bd.ClayDoc, bd.Obj]:
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box()))
    view._rect = RECT
    view.frame_selection(doc)
    view.draw(doc, RECT, 0.0)
    return doc, obj


def test_face_snap_lands_exactly_on_the_ray_hit(view) -> None:
    """``_snap_face`` reuses the same ray ``pick_face`` casts, so the point it
    reports must be exactly where that ray meets the surface -- not merely
    somewhere on the object."""
    doc, obj = _framed_box(view)
    centre = (RECT[2] * 0.5, RECT[3] * 0.5)

    hit = view.pick_face(doc, centre, evaluated=True)
    assert hit is not None, "the box must be under the cursor for this test to mean anything"

    point = view._snap_face(doc, centre)

    origin, direction = view._ray(centre)
    expected = np.asarray(origin, dtype="f8") + np.asarray(direction, dtype="f8") * hit.t
    assert point is not None
    assert np.allclose(point, expected, atol=1e-6)


def test_face_snap_finds_nothing_over_empty_space(view) -> None:
    doc, _obj = _framed_box(view)
    assert view._snap_face(doc, (2.0, 2.0)) is None


def test_face_snap_excludes_the_object_being_dragged(view) -> None:
    """The worst failure available: a dragged object's own face tracking the
    cursor and reporting a snap on its own moving geometry."""
    doc, obj = _framed_box(view)
    doc.select([obj.uid])
    view.app_ctx.state.clay.tool = "move"
    view._begin_gizmo_drag(doc)

    centre = (RECT[2] * 0.5, RECT[3] * 0.5)
    assert view._snap_face(doc, centre) is None


def test_edge_snap_lands_on_the_segment_between_its_two_endpoints(view) -> None:
    """However the projection distorts the cursor's aim, the answer must be a
    point *on* the picked edge -- collinear with, and between, its own two
    endpoints -- never off either end of it."""
    doc, obj = _framed_box(view)
    a_idx, b_idx = (int(v) for v in adjacency(obj.mesh).edge_verts[0])
    screen = view.screen_of(doc, obj.uid)
    # Aimed at the screen-space midpoint of one known edge, so *some* edge is
    # unambiguously under the cursor.
    at = tuple(((screen.xy[a_idx] + screen.xy[b_idx]) * 0.5).tolist())

    point = view._snap_edge(doc, at)
    assert point is not None

    world = view._world(doc, obj)
    a_world = (world @ np.append(obj.mesh.positions[a_idx].astype("f8"), 1.0))[:3]
    b_world = (world @ np.append(obj.mesh.positions[b_idx].astype("f8"), 1.0))[:3]
    axis = b_world - a_world
    length = float(np.linalg.norm(axis))
    t = float(np.dot(point - a_world, axis) / (length * length))
    on_segment = a_world + axis * max(0.0, min(1.0, t))
    assert np.allclose(point, on_segment, atol=1e-6), "the point must lie on the segment"
    assert -1e-6 <= t <= 1 + 1e-6, "the point must not fall off either end of the edge"


def test_edge_snap_near_the_projected_midpoint_lands_near_the_world_midpoint(view) -> None:
    """An orthographic camera makes the screen-space and world-space
    midpoints coincide (no perspective foreshortening to distort one against
    the other), which is what makes this a meaningful numeric check rather
    than only a "some point on the segment" one."""
    doc, obj = _framed_box(view)
    view.camera.orthographic = True
    view.draw(doc, RECT, 0.0)
    a_idx, b_idx = (int(v) for v in adjacency(obj.mesh).edge_verts[0])
    screen = view.screen_of(doc, obj.uid)
    at = tuple(((screen.xy[a_idx] + screen.xy[b_idx]) * 0.5).tolist())

    point = view._snap_edge(doc, at)

    world = view._world(doc, obj)
    a_world = (world @ np.append(obj.mesh.positions[a_idx].astype("f8"), 1.0))[:3]
    b_world = (world @ np.append(obj.mesh.positions[b_idx].astype("f8"), 1.0))[:3]
    midpoint = (a_world + b_world) * 0.5
    assert point is not None
    assert np.allclose(point, midpoint, atol=1e-3)


def test_edge_snap_excludes_the_edges_a_drag_is_moving(view) -> None:
    from realmspinner.kernels.mesh import elements as el

    doc, obj = _framed_box(view)
    doc.set_element_mode("face")
    doc.set_element_sel(obj.uid, el.ElementSel(faces=[0]))
    view.app_ctx.state.clay.tool = "move"
    view._begin_gizmo_drag(doc)
    view._begin_element_drag(doc)

    a_idx, b_idx = (int(v) for v in adjacency(obj.mesh).edge_verts[0])
    moving = set(int(v) for v in view._element_drags[obj.uid].verts)
    if a_idx not in moving or b_idx not in moving:
        pytest.skip("edge 0 is not wholly inside the dragged face on this box layout")

    screen = view.screen_of(doc, obj.uid)
    at = tuple(((screen.xy[a_idx] + screen.xy[b_idx]) * 0.5).tolist())
    assert view._snap_edge(doc, at) is None
