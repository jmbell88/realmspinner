"""Wire markers for the things in a scene that have no geometry of their own.

A :class:`~.mason.nodes.LightNode` and a :class:`~.mason.nodes.CameraNode`
resolve to a ``Placed`` with ``ref is None``, and ``mason_view._composite``
draws only placements that have a ref -- correctly, since the GPU cache is
keyed on one. So before this module existed a user could place a point light,
watch the node land in the document and the row appear in the outliner, and see
**nothing at all** in the viewport. :mod:`.viewer.markers` is the precedent the
plan named for the way out: describe them as :class:`~.viewer.render.DrawItem`
overlays and hand them to ``Renderer.draw(overlays=...)``, the path the gizmo
already takes.

**Lines, not surfaces.** A light is not an object in the scene; it is a symbol
for one, and a shaded blob would read as a prop that exports. Wire shapes also
cost nothing to build and need no normals, so the whole of this module is four
static vertex arrays and a matrix per node.

**Fixed world size, not fixed screen size.** The opposite choice from the
gizmo's, and deliberately: the gizmo is a *handle*, so it has to stay grabbable
at any zoom, while these are map symbols whose job is to say where in the scene
a light is. Screen-constant markers in a scene editor tile the display the
moment the user zooms out to look at the layout -- which is exactly when the
layout is what they are looking at. :data:`MARK_SIZE` is also the radius
:func:`~.mason.pick.ray_scene` tests a marker click against, imported from
there rather than restated, so what is drawn and what is clickable cannot
disagree.

**A spot's cone and a camera's frustum are drawn at their own angles**, by
scaling a unit shape rather than by rebuilding vertices: the shapes are the
only thing on the GPU and the angles are a matrix, so editing a cone angle in
Properties changes the picture with no upload at all.
"""

from __future__ import annotations

import math
from typing import Any

import moderngl
import numpy as np

from .....kernels.geom3d import math3d as m3
from ....viewer.render import DrawItem
from ..engine.pick import MARK_SIZE

#: Marker colours, as 0xRRGGBB -- :mod:`.viewer.markers`' own spelling, and
#: literals for its reason: ``theme.py`` is imgui-bearing and a viewport overlay
#: is drawn with no imgui frame in hand. Warm for a light, cool for a camera,
#: and one accent for "this is what is selected" shared by both, so the colour
#: answers "what kind" and the brightness answers "is it mine".
LIGHT = 0xF0C96C
CAMERA = 0x7C9CF0
#: The selected colour, matching ``viewer.markers.ACTIVE`` so selection means
#: the same green in every 3-D view in this app.
ACTIVE = 0x4CC38A

#: How far out a spot cone and a camera frustum are drawn, in metres. Short on
#: purpose: these say *which way* a light or a camera faces, and a cone drawn to
#: its real range would cross the whole scene.
THROW = 2.0

#: A camera marker's assumed aspect, for the frustum's width only. The real
#: aspect is the viewport's at export time and is not a property of the node
#: (glTF's ``aspectRatio`` is optional and this document does not carry one), so
#: the marker draws the common one rather than inventing a field for the sake of
#: a symbol's proportions.
MARK_ASPECT = 1.5

_CIRCLE_SEGMENTS = 24


def _circle(plane: tuple[int, int], radius: float = 1.0) -> list[list[float]]:
    """A closed ring of line segments in the two axes of ``plane``."""
    out: list[list[float]] = []
    for index in range(_CIRCLE_SEGMENTS):
        for step in (index, index + 1):
            angle = 2.0 * math.pi * (step % _CIRCLE_SEGMENTS) / _CIRCLE_SEGMENTS
            point = [0.0, 0.0, 0.0]
            point[plane[0]] = radius * math.cos(angle)
            point[plane[1]] = radius * math.sin(angle)
            out.append(point)
    return out


def point_shape() -> np.ndarray:
    """A bulb: three unit rings, one per plane. -> (n, 3) line vertices."""
    return np.array(
        _circle((0, 1)) + _circle((1, 2)) + _circle((0, 2)), dtype="f4"
    )


def cone_shape() -> np.ndarray:
    """A unit cone opening along ``-Z``: four ribs to a rim of radius 1 at
    ``z = -1``, plus the rim. Scaled per node so the drawn half-angle is the
    node's own ``outer_cone_angle``."""
    rim = [[x, y, -1.0] for x, y, _z in _circle((0, 1))]
    ribs: list[list[float]] = []
    for angle in (0.0, math.pi / 2, math.pi, 3 * math.pi / 2):
        ribs += [[0.0, 0.0, 0.0], [math.cos(angle), math.sin(angle), -1.0]]
    return np.array(ribs + rim, dtype="f4")


def sun_shape() -> np.ndarray:
    """Parallel rays: a unit ring with four lines running ``-Z`` from it, which
    is how a directional light is drawn everywhere -- it has a direction and no
    position to fall off from, and the parallel lines are the only part of the
    symbol that says so."""
    ring = _circle((0, 1), 0.45)
    rays: list[list[float]] = []
    for angle in (0.25, 0.75, 1.25, 1.75):
        radians = math.pi * angle
        x, y = 0.32 * math.cos(radians), 0.32 * math.sin(radians)
        rays += [[x, y, 0.0], [x, y, -1.0]]
    return np.array(ring + rays, dtype="f4")


def frustum_shape() -> np.ndarray:
    """A unit view frustum looking down ``-Z``: four corner rays to a
    ``1 x 1`` rect at ``z = -1``, that rect, and a triangle over its top edge
    saying which way up the camera is."""
    corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
    out: list[list[float]] = []
    for x, y in corners:
        out += [[0.0, 0.0, 0.0], [x, y, -1.0]]
    for index, (x, y) in enumerate(corners):
        nx, ny = corners[(index + 1) % 4]
        out += [[x, y, -1.0], [nx, ny, -1.0]]
    apex = [0.0, 1.6, -1.0]
    out += [[-1.0, 1.0, -1.0], apex, apex, [1.0, 1.0, -1.0]]
    return np.array(out, dtype="f4")


class SceneMarks:
    """The four marker buffers, and the draw-list builder over a frame's
    placements.

    One GL object set for the whole mode rather than one per node: the shapes
    are static and the difference between two lights is entirely a matrix, so
    there is nothing per-node to upload and nothing to invalidate when a light
    moves or its cone angle changes.
    """

    def __init__(self, ctx: Any, programs: Any) -> None:
        self.ctx = ctx
        self.program = programs.get("solid")
        self._buffers: list[Any] = []
        self._vaos: dict[str, Any] = {}
        for name, data in (
            ("point", point_shape()),
            ("cone", cone_shape()),
            ("sun", sun_shape()),
            ("frustum", frustum_shape()),
        ):
            vbo = ctx.buffer(data.tobytes())
            self._buffers.append(vbo)
            self._vaos[name] = ctx.vertex_array(self.program, [(vbo, "3f", "a_position")])

    def draws(self, placed: Any, selection: Any = ()) -> list[DrawItem]:
        """One :class:`DrawItem` per light and camera in ``placed``.

        Hidden placements are skipped -- a hidden node "does not render, export
        or pick", the sentence the outliner's own eye button promises -- and
        every other placement is ignored rather than filtered for by type here,
        so a sixth node kind costs this method nothing until it wants a symbol.
        """
        chosen = set(selection or ())
        items: list[DrawItem] = []
        for item in placed:
            if not item.visible:
                continue
            shape = _shape_for(item.node)
            if shape is None:
                continue
            name, scale = shape
            colour = ACTIVE if item.owner in chosen else _colour_for(item.node)
            items.append(
                DrawItem(
                    vao=self._vaos[name],
                    color=(*_rgb(colour), 1.0),
                    # ``item.world`` with its own scale divided back out would
                    # be a second rule to keep in step with the pick radius;
                    # the marker is placed by the node's world *position* and
                    # *rotation* alone, which is also what makes it the right
                    # size to click at any node scale.
                    model=_placement(item.world) @ m3.scaling(scale),
                    mode=moderngl.LINES,
                    # Depth-tested, ``viewer.markers``' opposite choice and for
                    # the reason its own docstring gives for making the call
                    # either way: a joint handle inside a mesh must be grabbable
                    # through it, but a light behind a wall that drew over the
                    # wall would make a scene editor's viewport unreadable about
                    # what is in front of what.
                    depth=True,
                )
            )
        return items

    def release(self) -> None:
        for vao in self._vaos.values():
            vao.release()
        for vbo in self._buffers:
            vbo.release()
        self._vaos.clear()
        self._buffers.clear()


def _shape_for(node: Any) -> tuple[str, tuple[float, float, float]] | None:
    """Which shape draws ``node``, and at what scale -- or ``None`` for a node
    that has geometry of its own and needs no symbol.

    Imported lazily for ``mason_view._terrain_node``'s stated reason: this
    module reaches into the engine for two class names on one line each, and a
    marker drawer is not a thing the node package should appear to depend on.
    """
    from ..engine.nodes import CameraNode, LightNode

    if isinstance(node, LightNode):
        if node.kind == "spot":
            # ``outer_cone_angle`` is the half-angle from the axis, per
            # KHR_lights_punctual -- so the rim radius is its tangent times the
            # throw, and the unit cone's own rim is at radius 1.
            radius = THROW * math.tan(max(0.01, min(float(node.outer_cone_angle), 1.55)))
            return "cone", (radius, radius, THROW)
        if node.kind == "directional":
            return "sun", (MARK_SIZE, MARK_SIZE, THROW)
        return "point", (MARK_SIZE, MARK_SIZE, MARK_SIZE)
    if isinstance(node, CameraNode):
        half_h = THROW * math.tan(max(0.01, min(float(node.yfov) * 0.5, 1.55)))
        return "frustum", (half_h * MARK_ASPECT, half_h, THROW)
    return None


def _colour_for(node: Any) -> int:
    from ..engine.nodes import LightNode

    return LIGHT if isinstance(node, LightNode) else CAMERA


def _placement(world: np.ndarray) -> np.ndarray:
    """``world`` with its scale normalized away, its translation kept.

    A light scaled to 5 is not a bigger light -- ``intensity`` is the field that
    says how much light there is -- so a marker that grew with the node's scale
    would be saying something the document does not mean, and would stop
    matching the fixed radius a click is tested against. A column scaled to zero
    leaves a zero-length basis vector; that is left as it is rather than
    replaced with an identity, because a marker drawn flat is a true picture of
    a node whose own axis has been collapsed.
    """
    matrix = np.array(world, dtype="f8", copy=True)
    for column in range(3):
        length = float(np.linalg.norm(matrix[:3, column]))
        if length > 0.0:
            matrix[:3, column] /= length
    return matrix


def _rgb(value: int) -> tuple[float, float, float]:
    return tuple(((value >> shift) & 0xFF) / 255.0 for shift in (16, 8, 0))
