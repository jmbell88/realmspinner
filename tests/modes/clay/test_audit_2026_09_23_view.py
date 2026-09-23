"""Regression tests for the 2026-09-23 audit's clay-06 and create-03.

clay-06: the pick, hover, marquee and snap loops in ``_view_pick.py`` and
``_view_drag.py`` skipped invisible and locked objects but not
``role == "collider"``, so a collider fit around a source object -- which
``view.py``'s own ``_composite`` never draws through the opaque path -- still
won every ray cast at the surface it was fit to. A click on the modelled
object landed on its own collider instead.

create-03: ``Camera.look_angles`` (so ``look_along``, so the Ctrl+1/3/7 axis
keys) and the ``orthographic`` toggle (Ctrl+5) both *snap* rather than ease,
so ``Camera.settled()`` reads true again the instant they land --
``FrameOps._frame_unchanged`` saw an unmoved camera and kept serving the
pre-snap texture until some unrelated input forced a redraw. Both are closed
by one mechanism, ``Camera.revision``, which ``_frame_unchanged`` now
compares alongside the render key, so every viewport this mixin serves
(Clay, Mason, Poser) is fixed without every ``mode.py`` caller having to
remember to set ``_render_dirty`` by hand.
"""

from __future__ import annotations

from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.kernels.mesh import colliders as cl
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio import _view_frame
from realmspinner.studio.modes.clay.ui import view as clay_view
from realmspinner.studio.viewer.camera import Camera

RECT = (0.0, 0.0, 128.0, 96.0)


class _State:
    def __init__(self) -> None:
        self.tool = "select"
        self.snap = False
        self.snap_translate = 0.125
        self.snap_rotate = 15.0
        self.snap_vertex = False


class _Ctx:
    def __init__(self) -> None:
        self.state = type("S", (), {"clay": _State()})()


# --- clay-06: colliders never win a pick, hover, marquee or snap -------------


def test_clicking_an_object_with_a_fitted_collider_still_selects_the_object_not_the_collider(
    gl,
) -> None:
    """The documented collider workflow (Clay chapter): fit a collider, then
    keep working with the object it protects. A sphere fit around a cone
    (``kernels.mesh.colliders.fit_sphere``) sits strictly outside the cone's
    own surface -- the same shape the audit's probe (clay-view-01.py) used --
    so any ray aimed at the cone hits the sphere first unless the pick loop
    skips ``role == "collider"``."""
    view = clay_view.ClayView(gl, _Ctx())
    try:
        doc = bd.ClayDoc()
        source = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Prop", mesh=bp.cone()))
        collider = doc.add_collider(source.uid, cl.fit_sphere(source.mesh))

        view._rect = RECT
        view.camera.set_target(m3.vec3())
        view.camera.set_position(m3.vec3(0.0, 0.0, 8.0))
        view.camera.aspect = RECT[2] / RECT[3]

        hit = view.pick(doc, (RECT[2] * 0.5, RECT[3] * 0.5))
        assert hit == source.uid, (
            f"pick_face returned {hit!r} -- the enclosing collider "
            f"({collider.uid}) won the ray instead of its source"
        )
    finally:
        view.release()


def test_hovering_over_a_fitted_collider_reports_the_source_face_not_the_colliders(
    gl,
) -> None:
    """``pick_element`` (used by hover as well as element-mode clicks) has its
    own object loop -- the 2026-09-22 audit's clay-02 already taught it to
    skip a locked object; a collider was still missing from that skip."""
    view = clay_view.ClayView(gl, _Ctx())
    try:
        doc = bd.ClayDoc()
        source = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Prop", mesh=bp.cone()))
        doc.add_collider(source.uid, cl.fit_sphere(source.mesh))
        doc.element_mode = "face"

        view._rect = RECT
        view.camera.set_target(m3.vec3())
        view.camera.set_position(m3.vec3(0.0, 0.0, 8.0))
        view.camera.aspect = RECT[2] / RECT[3]

        picked = view.pick_element(doc, (RECT[2] * 0.5, RECT[3] * 0.5))
        assert picked is not None, "the ray should hit the source's own face"
        uid, _index = picked
        assert uid == source.uid, f"pick_element reported the collider ({uid}), not the source"
    finally:
        view.release()


# --- create-03: an axis-view snap (Ctrl+1/3/7) and the ortho toggle ----------
# (Ctrl+5) both mark the frame dirty via Camera.revision -----------------------


class _Host:
    """Only what ``FrameOps._frame_unchanged`` reads -- the field set every
    ``_view_frame`` host already carries, per that module's own docstring."""

    def __init__(self, camera: Camera) -> None:
        self.camera = camera
        self._render_dirty = False
        self._last_render_key = "key"
        self.viewport = type("V", (), {"texture": object()})()


def test_an_axis_view_key_snap_marks_the_viewport_dirty() -> None:
    camera = Camera()
    host = _Host(camera)
    # Prime the cache: this exact key has already "rendered" once, with the
    # camera settled and nothing dirty -- the ordinary skip-this-frame case.
    assert _view_frame.FrameOps._frame_unchanged(host, "key") is True

    changed = _view_frame.axis_view_key(camera, "1", False)
    assert changed is True
    assert _view_frame.FrameOps._frame_unchanged(host, "key") is False, (
        "Ctrl+1 snaps the camera straight to its goal, so settled() reads "
        "true again the instant it lands -- the pre-fix _frame_unchanged "
        "saw an identical key and a settled camera and kept serving the "
        "pre-snap texture (2026-09-23 audit, create-03)"
    )

    # Prime again, then check the other snap path: Ctrl+5's orthographic
    # toggle, which never goes through look_angles at all.
    assert _view_frame.FrameOps._frame_unchanged(host, "key") is True
    changed_ortho = _view_frame.axis_view_key(camera, "5", False)
    assert changed_ortho is True
    assert _view_frame.FrameOps._frame_unchanged(host, "key") is False, (
        "Ctrl+5 toggles Camera.orthographic directly -- the same dirty-less "
        "snap create-03 found for the axis-view digits"
    )
