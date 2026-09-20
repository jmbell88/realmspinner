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
    centre = divisions // 2
    # 1 m cells only: Clay's ``grid_size`` picks ``divisions`` so that
    # ``step`` always comes out to 1.0 (see ``clay_view.ClayView.draw``), and
    # every offset is then an exact integer -- no epsilon needed to test
    # "is this a multiple of ten".
    major_lines = math.isclose(step, 1.0, abs_tol=1e-6)
    positions: list[list[float]] = []
    colors: list[tuple[float, float, float]] = []
    for i in range(divisions + 1):
        offset = -half + i * step
        if i == centre:
            color = _rgb(CENTRE_COLOR)
        elif major_lines and abs(offset) % MAJOR_STEP < 1e-6:
            color = _rgb(MAJOR_COLOR)
        else:
            color = _rgb(GRID_COLOR)
        positions += [[-half, 0.0, offset], [half, 0.0, offset]]
        positions += [[offset, 0.0, -half], [offset, 0.0, half]]
        colors += [color] * 4
    return np.array(positions, dtype="f4"), np.array(colors, dtype="f4")


class Grid:
    """The grid's GPU buffers, rebuilt only when the span changes."""

    def __init__(self, ctx, programs) -> None:
        self.ctx = ctx
        self.program = programs.get("lines")
        self.span = 0.0
        self.divisions = DIVISIONS
        self._vbo = None
        self._vao = None
        self.set_span(4.0)

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

    def release(self) -> None:
        for obj in (self._vao, self._vbo):
            if obj is not None:
                obj.release()
        self._vao = self._vbo = None
        self.span = 0.0
        self.divisions = DIVISIONS
