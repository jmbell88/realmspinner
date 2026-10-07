"""Regression test for the 2026-09-23 audit's create-03.

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

from realmspinner.studio import _view_frame
from realmspinner.studio.viewer.camera import Camera

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
