"""The ground grid, as a line list.

three's GridHelper in ten lines: a square of ``divisions`` cells, the two
centre lines in the brighter colour. Most callers (Mason, Poser, the asset
viewer) still let the span follow the model -- a power-of-ten that comfortably
contains its footprint, through :func:`span_for` and
:meth:`~.render.Renderer.fit_grid` -- because a fixed 4 m grid frames a 1 m
prop and nothing else.

Clay is the one caller with a fixed-size grid instead: a user-set
``grid_size`` in metres (default 100), 1 m cells always (``divisions`` picked
to make ``span / divisions == 1.0``), with a brighter line every ten of them
(``MAJOR_COLOR``) so a hundred 1 m cells do not moire into solid grey.
"""

from __future__ import annotations

import math

import moderngl
import numpy as np

from ...kernels.geom3d import math3d as m3

DIVISIONS = 16
CENTRE_COLOR = 0x2C2F3A
GRID_COLOR = 0x232530
#: Every tenth line, when a cell is exactly 1 m (Clay's fixed-size grid,
#: ``clay_state.ClayState.grid_size``): a hundred-plus 1 m cells across the
#: whole grid moire into flat grey at any zoom, and a brighter line every 10 m
#: is the same fix three's own GridHelper uses for a dense grid. Between
#: ``GRID_COLOR`` and ``CENTRE_COLOR`` on purpose -- distinct from both, and
#: still dimmer than the two centre lines.
MAJOR_COLOR = 0x282B37
#: Metres between major lines. Only ever checked when the cell size is
#: exactly 1 m; a Mason/Poser/asset-viewer grid (a power-of-ten span over
#: ``DIVISIONS`` cells) never has a 1 m cell and never draws one.
MAJOR_STEP = 10.0


def span_for(width: float, depth: float) -> float:
    """A power-of-ten span that comfortably contains the footprint."""
    extent = max(width, depth) * 2.5
    if extent <= 0 or not math.isfinite(extent):
        return 1.0
    return 10.0 ** math.ceil(math.log10(extent))


def _rgb(value: int) -> tuple[float, float, float]:
    return tuple(((value >> shift) & 0xFF) / 255.0 for shift in (16, 8, 0))


def build(span: float, divisions: int = DIVISIONS) -> tuple[np.ndarray, np.ndarray]:
    """-> (positions (n, 3), colors (n, 3)) for a GL_LINES draw."""
    half = span * 0.5
    step = span / divisions
    # 1 m cells only: Clay's ``grid_size`` picks ``divisions`` so that
    # ``step`` always comes out to 1.0 (see ``clay_view.ClayView.draw``), and
    # every offset is then an exact integer -- no epsilon needed to test
    # "is this a multiple of ten".
    major_lines = math.isclose(step, 1.0, abs_tol=1e-6)
    positions: list[list[float]] = []
    colors: list[tuple[float, float, float]] = []
    # Lines sit at whole multiples of ``step``, so the centre line (k == 0)
    # and the majors are placed by *value*, not by index. The 2026-10-04
    # audit's create-33: with an odd ``divisions`` (Clay's Grid size field
    # takes any integer 1-1000) the index-based layout ran from -half in
    # steps, so every offset was a half-step -- no line through the origin and
    # ``centre = divisions // 2`` crowned the line at -0.5 m instead. An even
    # count yields exactly the lines it always has; an odd one has one fewer
    # than ``divisions + 1`` and stops half a cell short of each edge, which
    # is the price of keeping the origin a line.
    reach = int(math.floor(divisions / 2 + 1e-9))
    for k in range(-reach, reach + 1):
        offset = k * step
        if k == 0:
            color = _rgb(CENTRE_COLOR)
        elif major_lines and abs(offset) % MAJOR_STEP < 1e-6:
            color = _rgb(MAJOR_COLOR)
        else:
            color = _rgb(GRID_COLOR)
        positions += [[-half, 0.0, offset], [half, 0.0, offset]]
        positions += [[offset, 0.0, -half], [offset, 0.0, half]]
        colors += [color] * 4
    return np.array(positions, dtype="f4"), np.array(colors, dtype="f4")


def axes_geometry(span: float) -> tuple[np.ndarray, np.ndarray]:
    """-> (axis lines, origin marker), each ``(n, 6)`` f4 of ``x y z r g b``.

    The two lines run the grid's full width, X in the gizmo's red and Z in its
    blue (Y is up, so it has no line on the ground). The marker is a small
    three-axis cross at the origin: ``lines`` has no point-size uniform, so a
    dot would be one pixel, and a cross is legible at any zoom with no shader.
    Pure, so a test reads what the grid will draw without a GL readback.
    """
    from .gizmo import AXIS_COLORS

    half = span * 0.5
    red, green, blue = (_rgb(AXIS_COLORS[a]) for a in "xyz")
    lines = np.array(
        [
            [-half, 0.0, 0.0, *red], [half, 0.0, 0.0, *red],
            [0.0, 0.0, -half, *blue], [0.0, 0.0, half, *blue],
        ],
        dtype="f4",
    )
    tick = min(max(span * 0.01, 0.02), 0.15)
    marker = np.array(
        [
            [-tick, 0.0, 0.0, *red], [tick, 0.0, 0.0, *red],
            [0.0, -tick, 0.0, *green], [0.0, tick, 0.0, *green],
            [0.0, 0.0, -tick, *blue], [0.0, 0.0, tick, *blue],
        ],
        dtype="f4",
    )
    return lines, marker


class Grid:
    """The grid's GPU buffers, rebuilt only when the span changes."""

    # Class-level defaults for the axis overlay, so a grid built without
    # ``__init__`` (the tests' fake-context one) still releases cleanly.
    axes = False
    _axes_vbo = _axes_vao = _marker_vbo = _marker_vao = None
    _axes_span: float | None = None

    def __init__(self, ctx, programs) -> None:
        self.ctx = ctx
        self.program = programs.get("lines")
        self.span = 0.0
        self.divisions = DIVISIONS
        self._vbo = None
        self._vao = None
        # The axis lines and origin dot are a second buffer, drawn over the grid
        # only when a caller asks (``set_axes``); ``build`` never includes them.
        self.set_span(4.0)

    def set_axes(self, on: bool) -> None:
        """Draw the X (red) and Z (blue) axis lines and an origin marker over the grid.

        Off by default so Mason, Poser and the asset viewer keep exactly the
        grid they always had; Clay turns it on, because a modeller placing
        things on a metre grid needs to know which way is which and where the
        origin is. A *second* pair of vertex arrays rather than recolouring
        ``build``'s centre lines: ``build`` is what the grid's tests pin, and
        the axis colours are the gizmo's (``gizmo.AXIS_COLORS``), so a red line
        on the ground and a red arrow on the handle are the same axis.
        """
        self.axes = bool(on)

    def _ensure_axes(self) -> None:
        """(Re)build the axis buffers for the current span. A no-op while current."""
        if self._axes_span == self.span and self._axes_vao is not None:
            return
        self._release_axes()
        lines, marker = axes_geometry(self.span)
        self._axes_vbo = self.ctx.buffer(np.ascontiguousarray(lines).tobytes())
        self._axes_vao = self.ctx.vertex_array(
            self.program, [(self._axes_vbo, "3f 3f", "a_position", "a_color")]
        )
        self._marker_vbo = self.ctx.buffer(np.ascontiguousarray(marker).tobytes())
        self._marker_vao = self.ctx.vertex_array(
            self.program, [(self._marker_vbo, "3f 3f", "a_position", "a_color")]
        )
        self._axes_span = self.span

    def _release_axes(self) -> None:
        for obj in (self._axes_vao, self._axes_vbo, self._marker_vao, self._marker_vbo):
            if obj is not None:
                obj.release()
        self._axes_vao = self._axes_vbo = self._marker_vao = self._marker_vbo = None
        self._axes_span = None

    def set_span(self, span: float, divisions: int | None = None) -> None:
        """Rebuild for a new (span, divisions) pair, skipping an unchanged one.

        ``divisions=None`` keeps ``DIVISIONS`` -- Mason's, Poser's and the
        asset viewer's own calls (through :func:`~.render.Renderer.fit_grid`)
        pass one argument and must see exactly the grid they always have.
        Clay is the only caller that passes an explicit count, derived from
        its metre-sized ``grid_size`` (``clay_view.ClayView.draw``).
        """
        divisions = DIVISIONS if divisions is None else max(1, int(divisions))
        if span == self.span and divisions == self.divisions:
            return
        self.release()
        self.span = span
        self.divisions = divisions
        positions, colors = build(span, divisions)
        data = np.concatenate([positions, colors], axis=1)
        self._vbo = self.ctx.buffer(np.ascontiguousarray(data).tobytes())
        self._vao = self.ctx.vertex_array(
            self.program, [(self._vbo, "3f 3f", "a_position", "a_color")]
        )

    def render(self, view, proj) -> None:
        self.program["u_view"].write(m3.gl_bytes(view))
        self.program["u_proj"].write(m3.gl_bytes(proj))
        self.program["u_exposure"].value = 1.0
        self.program["u_alpha"].value = 1.0
        self._vao.render(mode=moderngl.LINES)
        if self.axes:
            self._ensure_axes()
            self._axes_vao.render(mode=moderngl.LINES)
            self._marker_vao.render(mode=moderngl.LINES)

    def release(self) -> None:
        for obj in (self._vao, self._vbo):
            if obj is not None:
                obj.release()
        self._vao = self._vbo = None
        self._release_axes()
        self.span = 0.0
        self.divisions = DIVISIONS
