"""The 2026-10-04 audit's viewer findings create-19, create-33 and create-48.

All three are pure logic, so no GL context is needed: the camera and the grid's
line list are numpy, and the viewport's allocation is driven through a fake
context that fails on demand.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from realmspinner.studio.viewer import glctx, grid
from realmspinner.studio.viewer.camera import Camera

# --- create-19: Camera.frame ------------------------------------------------

_TALL = (np.array([-0.5, 0.0, -0.5]), np.array([0.5, 2.0, 0.5]))
_SMALL = (np.array([-0.05, 0.0, -0.05]), np.array([0.05, 0.1, 0.05]))


def _state(camera: Camera):
    return (
        camera.distance,
        camera.theta,
        camera.phi,
        *camera.target,
        *camera.position,
    )


def test_camera_frame_is_idempotent_and_ignores_the_previous_target():
    fresh = Camera()
    fresh.frame(*_SMALL)

    # A small asset adopted after a tall one: the old target (y = 1.0) must not
    # leak into where the eye goes, or the small asset is framed from below the
    # ground (the eye lands at 1.0 - 0.47 * distance < 0).
    after_tall = Camera()
    after_tall.frame(*_TALL)
    after_tall.frame(*_SMALL)
    assert _state(after_tall) == pytest.approx(_state(fresh))
    assert after_tall.position[1] > 0.0

    # Pressing F twice on one asset must not change the view.
    twice = Camera()
    twice.frame(*_TALL)
    once = _state(twice)
    twice.frame(*_TALL)
    assert _state(twice) == pytest.approx(once)

    # And the damping goals agree, or the next update() eases somewhere else.
    assert after_tall._goal_distance == pytest.approx(fresh._goal_distance)
    assert after_tall._goal_phi == pytest.approx(fresh._goal_phi)
    assert after_tall._goal_theta == pytest.approx(fresh._goal_theta)
    assert np.allclose(after_tall._goal_target, fresh._goal_target)


def test_camera_frame_keeps_the_fixed_three_quarter_offset_from_the_target():
    camera = Camera()
    radius = camera.frame(*_TALL)
    distance = radius / math.sin(math.radians(camera.fov * 0.5)) * 1.25
    expected = camera.target + distance * np.array([0.62, 0.47, 0.62])
    # The offset the app has always opened a model at, measured from the
    # target now rather than from wherever the camera happened to look.
    assert camera.position == pytest.approx(expected)


# --- create-33: an odd grid ---------------------------------------------------


def _centre_lines(positions, colors):
    centre = grid._rgb(grid.CENTRE_COLOR)
    return positions[np.all(np.isclose(colors, centre), axis=1)]


@pytest.mark.parametrize("size", [5, 101, 100, 16])
def test_an_odd_grid_still_draws_its_centre_line_through_the_origin(size):
    positions, colors = grid.build(float(size), size)
    centre = _centre_lines(positions, colors)
    # One line along X (z == 0) and one along Z (x == 0): two segments, four
    # vertices, each passing through the origin.
    assert len(centre) == 4
    along_x = centre[np.isclose(centre[:, 2], 0.0)]
    along_z = centre[np.isclose(centre[:, 0], 0.0)]
    assert len(along_x) == 2 and len(along_z) == 2
    assert np.all(np.isclose(centre[:, 1], 0.0))


def test_an_odd_one_metre_grid_still_draws_its_major_lines():
    positions, colors = grid.build(101.0, 101)
    major = grid._rgb(grid.MAJOR_COLOR)
    majors = positions[np.all(np.isclose(colors, major), axis=1)]
    # Lines at +-10, 20, 30, 40, 50 on each axis: 10 lines per axis, 2 verts.
    assert len(majors) == 10 * 2 * 2
    offsets = {round(abs(float(v)), 6) for v in majors[:, 0]} | {
        round(abs(float(v)), 6) for v in majors[:, 2]
    }
    assert {10.0, 20.0, 30.0, 40.0, 50.0} <= offsets


def test_an_odd_grid_stays_inside_its_span():
    positions, _ = grid.build(101.0, 101)
    assert np.abs(positions[:, [0, 2]]).max() <= 50.5 + 1e-6


# --- create-48: Viewport.resize after a failed allocation --------------------


class _Obj:
    def release(self):
        pass


class _Texture(_Obj):
    filter = None
    repeat_x = repeat_y = None


class _FlakyCtx:
    """Allocates normally until ``fail_on`` names the call that raises."""

    def __init__(self):
        self.fail_on: str | None = None
        self.calls: list[str] = []

    def _go(self, name, obj):
        self.calls.append(name)
        if self.fail_on == name:
            raise MemoryError(name)
        return obj

    def texture(self, size, comps):
        return self._go("texture", _Texture())

    def framebuffer(self, **kw):
        return self._go("framebuffer", _Obj())

    def renderbuffer(self, size, comps, samples=0):
        return self._go("renderbuffer", _Obj())

    def depth_renderbuffer(self, size, samples=0):
        return self._go("depth_renderbuffer", _Obj())


def test_a_viewport_whose_allocation_failed_retries_at_the_same_size():
    ctx = _FlakyCtx()
    vp = glctx.Viewport(ctx, (64, 64))
    assert vp.size == (64, 64)

    ctx.fail_on = "depth_renderbuffer"
    with pytest.raises(MemoryError):
        vp.resize((128, 128))
    # Nothing is allocated, so the viewport must not claim a size it has no
    # target for.
    assert vp.size != (128, 128)

    ctx.fail_on = None
    ctx.calls.clear()
    assert vp.resize((128, 128)) is True
    assert vp.size == (128, 128)
    assert vp.texture is not None and vp.draw_target is not None
