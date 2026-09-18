"""Clay's fixed-size grid (Task A) and god-light ground plane (Task C).

The grid module (``viewer/grid.py``) and the environment/renderer pieces
(``viewer/env.py``, ``viewer/render.py``) are shared with Mason, Poser and the
asset viewer, so what is pinned here is split the same way the source is:
pure-numpy geometry (no GL needed, always runs) and the GPU-side pieces
(skipped without a GL 3.3 context, the ``gl`` fixture's own rule).
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.viewer import env as envlib
from warlock.studio.viewer import glctx, grid
from warlock.studio.viewer.camera import Camera
from warlock.studio.viewer.render import Renderer

# --- grid.build: pure geometry, no GL --------------------------------------


def test_build_100m_span_spans_plus_minus_50():
    positions, _colors = grid.build(100.0, 100)
    assert positions[:, 0].min() == pytest.approx(-50.0)
    assert positions[:, 0].max() == pytest.approx(50.0)
    assert positions[:, 2].min() == pytest.approx(-50.0)
    assert positions[:, 2].max() == pytest.approx(50.0)
    # y is always the ground plane's own coordinate, never offset by the
    # grid itself -- only the ground quad (``render._Ground``) is nudged.
    assert np.all(positions[:, 1] == 0.0)


def test_build_100m_span_colours_every_ten_metres_as_major():
    """1 m cells (``span / divisions == 1.0``) get a brighter line every ten
    -- the moire fix -- and nothing else does."""
    positions, colors = grid.build(100.0, 100)
    major = grid._rgb(grid.MAJOR_COLOR)
    ordinary = grid._rgb(grid.GRID_COLOR)
    centre = grid._rgb(grid.CENTRE_COLOR)
    for pos, color in zip(positions, colors, strict=True):
        offset = pos[2] if pos[0] in (-50.0, 50.0) else pos[0]
        if offset == 0.0:
            assert tuple(color) == pytest.approx(centre)
        elif abs(offset) % 10.0 == 0.0:
            assert tuple(color) == pytest.approx(major), offset
        else:
            assert tuple(color) == pytest.approx(ordinary), offset


def test_a_non_1m_span_draws_no_major_lines():
    """Mason's, Poser's and the asset viewer's own grids (a power-of-ten span
    over ``DIVISIONS`` cells) never land on a 1 m cell, so they must never
    gain the major-line colour Task A introduced for Clay alone."""
    _positions, colors = grid.build(4.0, grid.DIVISIONS)
    major = grid._rgb(grid.MAJOR_COLOR)
    assert not any(tuple(c) == pytest.approx(major) for c in colors)


# --- Grid.set_span: divisions join the identity, defaults are unchanged ----


class _FakeProgram(dict):
    def __contains__(self, key):
        return True

    def __getitem__(self, key):
        return self.setdefault(key, _FakeUniform())


class _FakeUniform:
    value = None

    def write(self, data):
        self.value = data


class _FakePrograms:
    def get(self, name, defines=()):
        return _FakeProgram()


def test_set_span_with_no_divisions_keeps_the_module_default():
    """Mason's and the asset viewer's one-argument call must see exactly the
    grid they always have -- ``divisions=None`` is not a fourth behaviour."""

    class _Releasable:
        def release(self):
            pass

    class _FakeCtx:
        def buffer(self, data):
            return _Releasable()

        def vertex_array(self, program, attrs):
            return _Releasable()

    g = grid.Grid.__new__(grid.Grid)
    g.ctx = _FakeCtx()
    g.program = _FakeProgram()
    g.span = 0.0
    g.divisions = grid.DIVISIONS
    g._vbo = None
    g._vao = None
    g.set_span(4.0)
    assert g.divisions == grid.DIVISIONS
    g.set_span(100.0, divisions=100)
    assert g.divisions == 100
    # Same span, ``divisions`` reverting to the default: this must still
    # rebuild, or a Clay grid switched back to a plain viewer would keep
    # Clay's division count forever.
    g.set_span(100.0, divisions=None)
    assert g.divisions == grid.DIVISIONS


# --- env.bind: the light override -------------------------------------------


def test_bind_with_no_override_writes_the_key_light(gl):
    env = envlib.Environment(gl)
    program = _FakeProgram()
    env.bind(program)
    assert program["u_light_dir"].value == pytest.approx(env.key_direction)
    assert program["u_light_color"].value == pytest.approx(env.key_color)
    env.release()


def test_bind_with_an_override_writes_it_instead_and_mutates_nothing(gl):
    env = envlib.Environment(gl)
    before_dir, before_color = env.key_direction, env.key_color
    program = _FakeProgram()
    env.bind(program, light=((0.0, 1.0, 0.0), (2.0, 2.0, 2.0)))
    assert program["u_light_dir"].value == (0.0, 1.0, 0.0)
    assert program["u_light_color"].value == (2.0, 2.0, 2.0)
    # Never written back onto the shared instance -- the very next bind with
    # no override must still answer with the ordinary key light.
    assert env.key_direction == before_dir
    assert env.key_color == before_color
    program2 = _FakeProgram()
    env.bind(program2)
    assert program2["u_light_dir"].value == pytest.approx(before_dir)
    env.release()


def test_god_light_points_straight_down_at_the_key_intensity(gl):
    env = envlib.Environment(gl)
    direction, color = env.god_light
    assert direction == (0.0, 1.0, 0.0)
    assert color == env.key_color
    env.release()


# --- Renderer: the ground plane and the light override ----------------------


@pytest.fixture(scope="session")
def renderer(gl):
    r = Renderer(gl)
    yield r
    r.release()


@pytest.fixture
def viewport(gl):
    vp = glctx.Viewport(gl, (64, 64))
    yield vp
    vp.release()


def test_god_light_renders_a_different_picture_than_the_ordinary_key_light(
    renderer, viewport
):
    """The one observable effect a caller with no model at all can still
    check: the ground plane and the light-override paint different pixels
    even into an otherwise-empty scene."""
    camera = Camera()
    camera.frame(np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0]))
    renderer.grid.set_span(10.0, divisions=10)

    renderer.light_override = None
    renderer.draw(viewport, camera, None, show_grid=False, ground=False)
    without = viewport.read_rgba().copy()

    renderer.light_override = renderer.env.god_light
    renderer.draw(viewport, camera, None, show_grid=False, ground=True)
    with_ground = viewport.read_rgba().copy()
    renderer.light_override = None

    assert not np.array_equal(without, with_ground)


def test_camera_far_is_clamped_to_at_least_the_grid_size_plus_distance(gl):
    """``Camera.frame`` sizes the far plane off the *subject*'s own radius,
    which cuts a large fixed grid in half for a small prop -- the clamp
    ``ClayView.draw`` applies every frame."""
    from warlock.kernels.mesh import document as bd
    from warlock.kernels.mesh import primitives as bp
    from warlock.studio.modes.clay.ui import view as clay_view

    view = clay_view.ClayView(gl, None)
    try:
        view.grid_size = 100.0
        doc = bd.ClayDoc()
        doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
        view.frame_selection(doc)
        assert view.camera.far < view.camera.distance + view.grid_size
        view.draw(doc, (0.0, 0.0, 64.0, 64.0), 1.0 / 60.0)
        assert view.camera.far >= view.camera.distance + view.grid_size
    finally:
        view.release()


def test_render_png_restores_the_live_grid_span_and_divisions(gl):
    """An agent's render must not resize the grid the user is looking at in
    the live viewport -- ``render_png``'s ``finally`` restores both halves of
    the identity ``Grid.set_span`` rebuilds on."""
    from warlock.kernels.mesh import document as bd
    from warlock.kernels.mesh import primitives as bp
    from warlock.studio.modes.clay.ui import view as clay_view

    view = clay_view.ClayView(gl, None)
    try:
        view.grid_size = 100.0
        view.renderer.grid.set_span(100.0, divisions=100)
        doc = bd.ClayDoc()
        doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))

        view.render_png(doc, grid=True)

        assert view.renderer.grid.span == 100.0
        assert view.renderer.grid.divisions == 100
        assert view.renderer.light_override is None
    finally:
        view.release()
