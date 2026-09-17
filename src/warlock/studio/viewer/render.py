"""Drawing a model into a viewport.

The renderer owns the things that outlive any one model -- the program cache,
the environment probe, the grid -- and nothing else. A frame is: clear, draw
the grid, draw every primitive with the lit (or unlit) program, then draw the
overlays with depth testing off, which is how the frontend's ``renderOrder =
999`` markers stayed visible through the mesh.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import moderngl
import numpy as np

from ...kernels.geom3d import math3d as m3
from .env import Environment
from .glctx import Viewport
from .grid import Grid, span_for
from .programs import ProgramCache
from .scene import GpuModel

#: What the wire overlay's lines are drawn in: a near-black at three quarters
#: alpha. Chrome rather than material, so it is one colour whatever is under it
#: -- an overlay that took the material's colour would be invisible on exactly
#: the meshes a wireframe is reached for.
WIRE_TINT = (0.05, 0.05, 0.06, 0.75)

#: A hair below y=0, so the ground plane and the grid -- both nominally at
#: y=0 -- are not exactly coplanar. Without it the two z-fight: which one wins
#: a given pixel flickers with the camera angle, and a grid that flickers
#: reads as broken rather than as "on top of something". Pushed *down* rather
#: than pulling the grid up, so ``grid.py`` -- shared with Mason, Poser and
#: the asset viewer, none of which ever draw a ground plane under it -- needs
#: no change at all for this.
GROUND_Y_OFFSET = -0.001


def _ground_geometry(span: float) -> tuple[np.ndarray, np.ndarray]:
    """-> (positions (6, 3), normals (6, 3)): two triangles, one quad."""
    half = max(span, 0.0) * 0.5
    y = GROUND_Y_OFFSET
    positions = np.array(
        [
            [-half, y, -half], [half, y, -half], [half, y, half],
            [-half, y, -half], [half, y, half], [-half, y, half],
        ],
        dtype="f4",
    )
    normals = np.tile(np.array([0.0, 1.0, 0.0], dtype="f4"), (6, 1))
    return positions, normals


class _Ground:
    """The god-light ground plane's GPU buffers, rebuilt only when the span
    changes -- ``grid.Grid``'s own rule, restated here because a quad is a
    different enough shape (two triangles, a normal per vertex, no colour)
    that sharing the class would mean branching it in half."""

    def __init__(self, ctx: moderngl.Context, programs: ProgramCache) -> None:
        self.ctx = ctx
        self.program = programs.get("ground")
        self.span = -1.0
        self._vbo = None
        self._vao = None

    def set_span(self, span: float) -> None:
        if span == self.span:
            return
        self.release()
        self.span = span
        positions, normals = _ground_geometry(span)
        data = np.concatenate([positions, normals], axis=1)
        self._vbo = self.ctx.buffer(np.ascontiguousarray(data).tobytes())
        self._vao = self.ctx.vertex_array(
            self.program, [(self._vbo, "3f 3f", "a_position", "a_normal")]
        )

    def render(self, view: np.ndarray, proj: np.ndarray) -> None:
        if self._vao is None:
            return
        self.program["u_view"].write(m3.gl_bytes(view))
        self.program["u_proj"].write(m3.gl_bytes(proj))
        self._vao.render(mode=moderngl.TRIANGLES)

    def release(self) -> None:
        for obj in (self._vao, self._vbo):
            if obj is not None:
                obj.release()
        self._vao = self._vbo = None
        self.span = -1.0


@dataclass
class DrawItem:
    """One overlay draw: a vertex array, a colour and how to draw it.

    Markers, gizmo handles and Clay's element-selection overlays are the users.
    They are described rather than drawn directly so the renderer can order
    them without each of them knowing that it is an overlay.

    ``depth`` is the one thing they genuinely disagree about. A joint marker
    inside a mesh must draw through it -- one you cannot see is one you cannot
    grab -- but a selected vertex on the *far* side of a closed mesh must not,
    or the user is looking at a cloud of dots with no way to tell which are in
    front. So depth-tested items draw first, into the depth buffer the model
    left behind, and depth-off items draw over them, exactly as before.

    ``point_size`` is only read for ``POINTS``; zero leaves whatever the
    context had, which is what every existing caller wants.
    """

    vao: Any
    color: tuple[float, float, float, float]
    model: np.ndarray = field(default_factory=m3.identity)
    mode: int = moderngl.TRIANGLES
    vertices: int = -1
    depth: bool = False
    point_size: float = 0.0


class Renderer:
    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self.programs = ProgramCache(ctx)
        self.env = Environment(ctx)
        self.grid = Grid(ctx, self.programs)
        self.ground = _Ground(ctx, self.programs)
        self.exposure = 1.0
        # ``(direction, color)`` in place of the environment's own key light,
        # or ``None`` for the ordinary key light. Clay's god-light mode is the
        # one setter (``ClayView.draw``, from ``self.env.god_light``); nothing
        # else in this class reads Clay's state, so it arrives as a plain
        # value rather than a flag this module would have to know the name of.
        self.light_override: (
            tuple[tuple[float, float, float], tuple[float, float, float]] | None
        ) = None

    # -- frame -------------------------------------------------------------

    def draw(
        self,
        viewport: Viewport,
        camera: Any,
        gpu: GpuModel | None,
        *,
        model_matrix: np.ndarray | None = None,
        wireframe: bool = False,
        flat: bool = False,
        show_grid: bool = True,
        background: tuple[float, float, float, float] | None = None,
        overlays: list[DrawItem] | None = None,
        wire_overlay: bool = False,
        alpha: float = 1.0,
        ground: bool = False,
    ) -> None:
        """One frame: the grid, the model, and the overlays over it.

        ``wireframe`` *replaces* the fill and ``wire_overlay`` draws over it,
        and the two are different questions: the first is a shading mode -- show
        me the edges instead of the surface -- and the second is an overlay,
        show me the edges as well. Blender has both and calls them that.

        ``alpha`` below 1 is X-ray: the surface goes see-through so an element
        behind it can be picked. It also turns the depth *write* off, because a
        translucent surface that still wrote depth would hide exactly what it
        was made transparent to reveal -- the far side of its own mesh.

        ``ground`` draws a flat, lit quad at y=0 sized to the current grid
        span, before the grid -- Clay's god-light mode, and nobody else's: it
        is what the god light (``self.light_override``) has to fall on for
        the effect to be visible at all, since without it the light shines
        past every object onto the clear colour and only the model itself
        shows the change. Not pickable and not counted in any bounds -- it is
        drawn straight from ``self.grid.span`` rather than through
        :class:`DrawItem` or the scene it lights.
        """
        ctx = self.ctx
        viewport.use()
        clear = background if background is not None else (*self.env.background, 1.0)
        viewport.draw_target.clear(*clear, depth=1.0)

        camera.aspect = viewport.size[0] / max(viewport.size[1], 1)
        view = camera.view()
        proj = camera.projection()

        ctx.enable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        if ground and self.grid.span > 0.0:
            self.ground.set_span(self.grid.span)
            program = self.ground.program
            program["u_exposure"].value = self.exposure
            self.env.bind(program, light=self.light_override)
            self.ground.render(view, proj)

        if show_grid:
            ctx.wireframe = False
            self.grid.render(view, proj)

        if gpu is not None:
            model_matrix = m3.identity() if model_matrix is None else model_matrix
            # Desktop-only: GLES has no glPolygonMode, so a GLES port would
            # need a barycentric shader instead. Left as-is deliberately --
            # this app ships a window, not a web page.
            translucent = alpha < 0.999
            if translucent:
                ctx.depth_mask = False
            ctx.wireframe = wireframe
            self._draw_model(gpu, camera, view, proj, model_matrix, flat, alpha)
            ctx.wireframe = False
            if translucent:
                ctx.depth_mask = True
            if wire_overlay and not wireframe:
                # A second pass over the same geometry, edges only, pulled a
                # hair toward the eye. Without the polygon offset the lines sit
                # in exactly the plane of the faces they outline and z-fighting
                # makes them dashed -- which reads as a broken mesh rather than
                # as a wireframe. moderngl has no polygon-offset enable flag --
                # the ``polygon_offset`` setter below is itself the switch (a
                # non-zero pair enables, ``(0.0, 0.0)`` disables), so the
                # ``ctx.enable(moderngl.POLYGON_OFFSET_FILL)`` that used to sit
                # here raised AttributeError and tripped Clay's viewport for
                # every user of the Wireframe overlay from v0.0.30 until 2026-09-06.
                ctx.wireframe = True
                ctx.polygon_offset = (-1.0, -1.0)
                self._draw_model(
                    gpu, camera, view, proj, model_matrix, True, 1.0,
                    tint=WIRE_TINT,
                )
                ctx.polygon_offset = (0.0, 0.0)
                ctx.wireframe = False

        if overlays:
            # Depth-tested first, then depth-off over the top -- see
            # ``DrawItem.depth``. Everything goes through the one "solid"
            # program, the gizmo idiom: an overlay is a coloured vertex array,
            # and a second shader would be a second place to keep the tone
            # mapping in step.
            program = self.programs.get("solid")
            program["u_view"].write(m3.gl_bytes(view))
            program["u_proj"].write(m3.gl_bytes(proj))
            program["u_exposure"].value = self.exposure
            for tested in (True, False):
                items = [item for item in overlays if item.depth is tested]
                if not items:
                    continue
                if tested:
                    ctx.enable(moderngl.DEPTH_TEST)
                else:
                    ctx.disable(moderngl.DEPTH_TEST)
                for item in items:
                    if item.point_size:
                        ctx.point_size = item.point_size
                    program["u_model"].write(m3.gl_bytes(item.model))
                    program["u_color"].value = item.color
                    item.vao.render(mode=item.mode, vertices=item.vertices)
            ctx.enable(moderngl.DEPTH_TEST)

        viewport.resolve()

    def draw_ids(
        self,
        viewport: Viewport,
        camera: Any,
        composite: Any | None,
        *,
        id_colors: dict[int, tuple[int, int, int]],
    ) -> None:
        """One object-id pass: every primitive flat-shaded in its own
        object's colour, no lighting, no tone map, no blending and no
        culling -- the properties a caller decoding pixels back into uids by
        exact match needs. Blending and MSAA both mix an edge pixel's colour
        with its neighbour's or the background's, which is a pixel neither
        colour can claim afterwards; ``viewport`` being single-sample is the
        caller's job (see ``ClayView.render_ids``, the only one), and turning
        blending off here is this method's own half of that promise. Culling
        is off too: the id an interior or a back face reads back as does not
        depend on which way it faces, so there is nothing to gain by leaving
        gaps a normal draw would cull for fill-rate reasons alone.

        ``composite.uids`` is walked in lockstep with ``composite.draws`` --
        see :class:`~.._view_frame.Composite`'s own docstring -- so a draw
        whose uid is not in *id_colors* (an object the caller chose not to
        colour-code) is skipped rather than drawn in whatever the last
        uniform write left behind.
        """
        ctx = self.ctx
        viewport.use()
        viewport.draw_target.clear(1.0, 1.0, 1.0, 1.0, depth=1.0)
        camera.aspect = viewport.size[0] / max(viewport.size[1], 1)
        view = camera.view()
        proj = camera.projection()
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.BLEND)
        ctx.disable(moderngl.CULL_FACE)
        if composite is not None and composite.draws:
            program = self.programs.get("id")
            program["u_view"].write(m3.gl_bytes(view))
            program["u_proj"].write(m3.gl_bytes(proj))
            uids = composite.uids or ()
            for (node, primitive), uid in zip(composite.draws, uids, strict=True):
                color = id_colors.get(uid)
                if color is None:
                    continue
                program["u_model"].write(m3.gl_bytes(node.world))
                program["u_color"].value = (
                    color[0] / 255.0, color[1] / 255.0, color[2] / 255.0,
                )
                primitive.vao(program).render()
        viewport.resolve()

    def _draw_model(
        self,
        gpu: GpuModel,
        camera: Any,
        view: np.ndarray,
        proj: np.ndarray,
        model_matrix: np.ndarray,
        flat: bool,
        alpha: float = 1.0,
        *,
        tint: tuple[float, float, float, float] | None = None,
    ) -> None:
        """One pass over every primitive. ``tint`` overrides the material.

        The override exists for the wire overlay and for nothing else: those
        lines are chrome, so they must not take the colour of whatever material
        happens to be under them -- a dark wireframe over a dark material is a
        wireframe nobody can see. It is written *after* ``material.bind`` for
        the same reason it is a parameter rather than a mutation of the
        material: the material is the document's and this is the view's.
        """
        name = "unlit" if flat else "pbr"
        # The per-frame constants -- view, projection, exposure, camera, the
        # environment -- are written once per *program* rather than once per
        # primitive (B15), mirroring the overlay pass. A frame draws dozens of
        # primitives through a handful of programs.
        view_bytes = m3.gl_bytes(view)
        proj_bytes = m3.gl_bytes(proj)
        camera_pos = tuple(camera.position)
        seen: set[int] = set()
        # The normal-matrix cache lives on GpuModel; Clay's _Composite has
        # none, so those draws fall back to the inline computation.
        normal_bytes = getattr(gpu, "normal_matrix_bytes", None)
        for node, primitive in gpu.draws:
            program = self.programs.get(name, primitive.defines)
            world = model_matrix @ node.world
            program["u_model"].write(m3.gl_bytes(world))
            if id(program) not in seen:
                seen.add(id(program))
                program["u_view"].write(view_bytes)
                program["u_proj"].write(proj_bytes)
                program["u_exposure"].value = self.exposure
                if "u_alpha" in program:
                    program["u_alpha"].value = float(alpha)
                if "u_camera_pos" in program:
                    program["u_camera_pos"].value = camera_pos
                    self.env.bind(program, light=self.light_override)
            if "u_normal_matrix" in program:
                # The inverse transpose, so a non-uniform scale does not tilt
                # the normals. Per node because the placement transform is
                # uniform but a glTF node's need not be; cached per node on
                # the model (B15) because a node's world only moves on a pose
                # change or a placement change.
                #
                # A zero on any scale axis makes the 3x3 singular, and an
                # unguarded inverse raised out of the *draw* -- one flattened
                # object took the whole viewport down. Its normals are
                # undefined either way, so the identity is as good an answer
                # as exists and the rest of the scene still renders.
                if normal_bytes is not None:
                    program["u_normal_matrix"].write(normal_bytes(node, world))
                else:
                    try:
                        normal_matrix = np.linalg.inv(world[:3, :3]).T
                    except np.linalg.LinAlgError:
                        normal_matrix = np.eye(3)
                    program["u_normal_matrix"].write(
                        np.ascontiguousarray(normal_matrix.T, dtype="f4").tobytes()
                    )
            primitive.material.bind(program)
            if tint is not None and "u_base_color_factor" in program:
                program["u_base_color_factor"].value = tint
            if len(primitive.material.textures) >= 5 and "u_camera_pos" in program:
                # A five-texture material walks its units up to 4, which is
                # the environment probe's slot -- rebind it, since the hoist
                # above only bound it once per program.
                self.env.bind(program, light=self.light_override)
            if primitive.skinned and "u_joints" in program:
                palette = gpu.palette(node)
                if palette:
                    program["u_joints"].write(palette)
            if primitive.material.material.double_sided:
                self.ctx.disable(moderngl.CULL_FACE)
            else:
                # Matches three: single-sided means back faces are culled, and
                # trellis output is genuinely closed often enough for it to
                # matter to fill rate.
                self.ctx.enable(moderngl.CULL_FACE)
            primitive.vao(program).render()
        self.ctx.disable(moderngl.CULL_FACE)

    # -- helpers -----------------------------------------------------------

    def fit_grid(self, lo: np.ndarray, hi: np.ndarray) -> None:
        size = np.asarray(hi) - np.asarray(lo)
        self.grid.set_span(span_for(float(size[0]), float(size[2])))

    def release(self) -> None:
        self.grid.release()
        self.ground.release()
        self.env.release()
        self.programs.release()
