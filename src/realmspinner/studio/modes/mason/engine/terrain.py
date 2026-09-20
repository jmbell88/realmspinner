"""The ground -- a document-singleton height field, its brushes, and the one
conversion out to a drawable mesh.

``MasonDoc.terrain`` holds at most one of these (see the plan's "a document
singleton plus one node"): the array is large enough that copying it into
every prefab instance would be its own memory problem, and two terrains is two
ground planes, which nobody asked for. The node that refers to it is
``nodes.py``'s business; this module only knows about the array, the brushes
that reshape it, and the mesh those brushes end up drawn as.

Written the same way ``plotter/tools.py`` writes its layer tools: a brush is a
pure function of the array it reads, never the array it is handed to mutate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .....kernels.geom3d import gltf

#: The largest side (cells per edge; the array itself is one larger) a
#: ``Terrain`` will hold. Measured 2026-09-11 (see ``dev/measurements/`` for
#: the dated write-up this number is read off): a full ``terrain_mesh()``
#: rebuild -- what a sculpt drag pays every frame it is open, since a brush
#: has no partial-mesh update -- costs 5.7 ms at side 256 and 28.2 ms at side
#: 512, against a 16.7 ms budget for the whole frame at 60 Hz. 256 is the
#: largest of the measured sides that leaves the rebuild inside that budget
#: with room for the rest of the frame; 512 alone is already past it. The
#: array itself is not the constraint -- 256's height field is 0.26 MB and
#: 512's is 1.05 MB, both nothing -- and neither is a single brush stroke,
#: which costs a few hundredths of a millisecond at every size tested because
#: it is local to the brush radius, not the array. At a sensible 1
#: vertex-per-metre density this caps a terrain at 256x256 m, which the scene
#: editor's own brief (place and light assets, not simulate open-world
#: terrain) does not need past.
MAX_TERRAIN_SIDE = 256

#: The floor a side must clear. Below this a "terrain" is a handful of
#: vertices with no sculpting question worth a brush, and letting it through
#: only means every brush has to cope with a 1x1 or 0x0 cell grid for no
#: document anyone would save.
MIN_TERRAIN_SIDE = 1

#: ``(x0, y0, x1, y1)`` in *vertex* indices, half-open -- one dimension over
#: from ``plotter/tools.py``'s ``Region``, which is ``(x0, y0, ndarray)`` over
#: a tile grid. ``x`` walks columns (the world-space X axis), ``y`` walks rows
#: (world-space Z); the array is square so the two never need distinguishing
#: by shape, only by convention, and this is the convention.
Rect = tuple[int, int, int, int]

Falloff = Literal["smooth", "linear"]


@dataclass(eq=False)
class Terrain:
    """A regular height-field grid, in local (unrotated, unscaled-by-anything-
    but-its-own-fields) space.

    ``heights`` is ``(n+1, n+1)`` f4, one vertex height in metres per cell
    corner, indexed ``heights[row, col]`` where ``row`` is the world-space Z
    direction and ``col`` is X -- the same axis order ``terrain_mesh`` reads it
    in.

    **Every brush rebinds ``heights`` to a new array; nothing ever writes into
    it in place.** Two things rest on that and both break silently if it is
    violated: :func:`terrain_mesh`'s memo below is keyed on
    ``cached_array is terrain.heights``, so an in-place write leaves the old
    mesh in the cache describing a ground that no longer exists under it; and
    an undo step's recorded ``before`` sub-array is a slice taken *before* the
    brush ran, which a write into the live array would then edit retroactively
    out from under the history stack (the exact trap ``core/undo.py``'s
    module docstring names for a shared buffer generally). A brush that wants
    to change the ground hands back ``(rect, new_sub_array)`` and something
    else -- not this module -- rebinds the full array around that patch.

    ``eq=False`` because the memo above needs the instance to be hashable by
    identity: a dataclass ``__eq__`` compares field values, which would make
    two terrains with identical heights collide as dict keys and is not what
    "this particular ground, right now" means anyway.
    """

    heights: np.ndarray
    size_x: float
    size_z: float
    material: gltf.Material
    #: ``(array, primitive)`` the last :func:`terrain_mesh` call produced for
    #: this instance, or ``None`` before the first call. Private to this
    #: module; see :func:`terrain_mesh` for why it is not keyed on ``id()``.
    _mesh_cache: tuple[np.ndarray, gltf.Primitive] | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        arr = np.asarray(self.heights)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError(
                f"a terrain's heights must be a square 2D array, got shape {arr.shape}"
            )
        side = arr.shape[0] - 1
        if side < MIN_TERRAIN_SIDE:
            raise ValueError(
                f"a terrain needs at least {MIN_TERRAIN_SIDE + 1} vertices per edge, "
                f"got {arr.shape[0]}"
            )
        if side > MAX_TERRAIN_SIDE:
            raise ValueError(
                f"a terrain side of {side} cells exceeds MAX_TERRAIN_SIDE ({MAX_TERRAIN_SIDE})"
            )
        if not (self.size_x > 0 and self.size_z > 0):
            raise ValueError(
                f"a terrain's size_x and size_z must both be positive, got "
                f"{self.size_x!r}, {self.size_z!r}"
            )
        if not np.isfinite(arr).all():
            raise ValueError("a terrain's heights must all be finite")
        # Coerce to f4 *and* own the buffer in one copy, unconditionally --
        # not just "when it looks like a view". A view reports only its own
        # small ``nbytes`` while pinning the whole base array alive (the trap
        # ``core/undo.py`` names), and an array that already happens to be
        # f4 and contiguous is not proof it is not itself a slice of something
        # bigger the caller still holds a reference to.
        self.heights = np.array(arr, dtype=np.float32, copy=True)

    @property
    def side(self) -> int:
        """Cells per edge (``n``): the mesh is ``n*n`` quads, ``(n+1)**2`` verts."""
        return self.heights.shape[0] - 1


# --- shared brush geometry ----------------------------------------------------


def _clip_circle(shape: tuple[int, int], cx: float, cz: float, radius: float) -> Rect | None:
    """The bounding rect of a circular brush, clipped to ``shape``.

    A brush dragged off the edge is a legitimate stroke whose visible part
    must land -- ``plotter/tools.py``'s rule, restated here in two dimensions
    of vertex index instead of one of tile. Only a placement *entirely*
    outside the array returns ``None``.
    """
    rows, cols = shape
    x0 = max(0, int(np.floor(cx - radius)))
    x1 = min(cols, int(np.ceil(cx + radius)) + 1)
    y0 = max(0, int(np.floor(cz - radius)))
    y1 = min(rows, int(np.ceil(cz + radius)) + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    return (x0, y0, x1, y1)


def _falloff_weight(
    rect: Rect, cx: float, cz: float, radius: float, falloff: Falloff
) -> np.ndarray:
    """Per-cell brush strength over ``rect``, 1 at the centre and 0 at ``radius``.

    ``t`` is the normalized radial distance, clamped to ``[0, 1]`` so a cell
    inside the rect's square corner past the circle reads a flat zero rather
    than an extrapolated (and possibly negative) weight.
    """
    x0, y0, x1, y1 = rect
    cols = np.arange(x0, x1, dtype=np.float32) - np.float32(cx)
    rows = np.arange(y0, y1, dtype=np.float32) - np.float32(cz)
    dist = np.sqrt(rows[:, None] ** 2 + cols[None, :] ** 2)
    t = np.clip(dist / np.float32(radius), 0.0, 1.0)
    if falloff == "smooth":
        return 1.0 - (3 * t**2 - 2 * t**3)
    if falloff == "linear":
        return 1.0 - t
    raise ValueError(f"unknown brush falloff {falloff!r}")


def _box_blur(heights: np.ndarray, rect: Rect) -> np.ndarray:
    """A 3x3 box blur of ``heights``, evaluated only over ``rect``.

    No scipy, and no padding of the *whole* array either: a radius-16 brush on
    a 2048-side terrain has no business touching the three-and-a-half million
    cells it never returns. Each of the nine taps is the rect's own row/column
    range shifted by one and clamped with ``np.clip`` -- edge-replicated
    against the true array boundary, not the rect's, which is what keeps a
    brush that clips against the terrain edge from blurring in a seam that
    was never there.
    """
    rows_n, cols_n = heights.shape
    x0, y0, x1, y1 = rect
    cols = np.arange(x0, x1)
    rows = np.arange(y0, y1)
    acc = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
    for dy in (-1, 0, 1):
        r = np.clip(rows + dy, 0, rows_n - 1)
        for dx in (-1, 0, 1):
            c = np.clip(cols + dx, 0, cols_n - 1)
            acc += heights[np.ix_(r, c)]
    return acc / np.float32(9.0)


# --- the brushes ----------------------------------------------------------
#
# Each takes the array it reads (never writes) and returns ``(rect,
# new_sub_array)`` or ``None``. Nothing here mutates, nothing pushes a step,
# and a computation that changed nothing returns ``None`` rather than an
# empty write -- ``plotter/tools.py``'s three rules, restated because a brush
# is a tool one dimension over.


def _unusable(cx: float, cz: float, radius: float, *scalars: float) -> bool:
    """True when this call would plant a NaN or an inf into the ground, or
    raise trying.

    ``_falloff_weight`` divides by ``radius``: at ``radius == 0`` the cell
    sitting exactly on the centre is ``0 / 0``, which is NaN, and
    ``np.clip`` passes a NaN straight through rather than clamping it away.
    That NaN then rides the falloff into ``after`` and out through the
    returned sub-array -- ``Terrain.__post_init__`` would refuse it, but a
    brush's region is written back into the live array directly, never
    through ``__post_init__`` again, so nothing re-checks it on the way in.
    A non-finite ``amount``/``strength``/``level``/``amplitude`` reaches the
    ground the same way, with no division needed to get there.

    ``cx``/``cz`` have their own real path to a non-finite value, upstream of
    any of this: a brush centre is usually the mouse ray hitting the ground,
    and ``viewer/picking.ray_plane`` divides by the ray's dot product with the
    plane normal, which is exactly zero when the ray is parallel to the plane
    -- an eye level with a flat horizon, not an edge case anyone has to
    provoke on purpose. A NaN or inf centre never even reaches the division
    above: ``_clip_circle`` does ``int(np.floor(cx))``, and Python's ``int()``
    raises ``ValueError`` on a NaN float and ``OverflowError`` on an infinite
    one, straight out of a mouse-move handler.

    A brush-size slider sitting at zero is an ordinary thing for a UI to
    hand over mid-drag, not a mistake -- the same call Clay's own
    ``snap_value`` makes about a step of zero being the off switch rather
    than a division -- so this returns a bool for the caller to turn into
    ``None``, never a ``ValueError``: a brush runs once a frame inside a
    gesture, and a refusal there has nothing to show.
    """
    if not (np.isfinite(cx) and np.isfinite(cz)):
        return True
    if not np.isfinite(radius) or radius <= 0:
        return True
    return not all(np.isfinite(s) for s in scalars)


def raise_lower(
    heights: np.ndarray,
    cx: float,
    cz: float,
    radius: float,
    amount: float,
    *,
    falloff: Falloff = "smooth",
) -> tuple[Rect, np.ndarray] | None:
    """Add ``amount`` metres at the centre, tapering to zero at ``radius``."""
    if _unusable(cx, cz, radius, amount):
        return None
    rect = _clip_circle(heights.shape, cx, cz, radius)
    if rect is None:
        return None
    x0, y0, x1, y1 = rect
    weight = _falloff_weight(rect, cx, cz, radius, falloff)
    before = heights[y0:y1, x0:x1]
    after = (before + weight * np.float32(amount)).astype(np.float32)
    if np.array_equal(before, after):
        # amount == 0 is the common case (a click that never dragged), but a
        # weight that is exactly zero everywhere the rect clips to would land
        # here too -- either way, nothing changed and no undo step is honest.
        return None
    return rect, np.ascontiguousarray(after)


def smooth(
    heights: np.ndarray, cx: float, cz: float, radius: float, strength: float
) -> tuple[Rect, np.ndarray] | None:
    """Blend each cell toward its 3x3 box-blurred neighbourhood.

    ``strength`` is 0..1, the fraction of the way from the original height to
    the blurred one; the brush's own circular falloff scales that fraction
    again, so a drag's edge feathers instead of stepping.
    """
    if _unusable(cx, cz, radius, strength):
        return None
    rect = _clip_circle(heights.shape, cx, cz, radius)
    if rect is None:
        return None
    x0, y0, x1, y1 = rect
    weight = _falloff_weight(rect, cx, cz, radius, "smooth") * np.float32(strength)
    before = heights[y0:y1, x0:x1]
    blurred = _box_blur(heights, rect)
    after = (before + (blurred - before) * weight).astype(np.float32)
    if np.array_equal(before, after):
        return None
    return rect, np.ascontiguousarray(after)


def flatten(
    heights: np.ndarray, cx: float, cz: float, radius: float, level: float, strength: float
) -> tuple[Rect, np.ndarray] | None:
    """Pull each cell toward ``level``, by ``strength`` scaled by the falloff."""
    if _unusable(cx, cz, radius, level, strength):
        return None
    rect = _clip_circle(heights.shape, cx, cz, radius)
    if rect is None:
        return None
    x0, y0, x1, y1 = rect
    weight = _falloff_weight(rect, cx, cz, radius, "smooth") * np.float32(strength)
    before = heights[y0:y1, x0:x1]
    after = (before + (np.float32(level) - before) * weight).astype(np.float32)
    if np.array_equal(before, after):
        # The ground already at ``level`` under the whole brush is the case
        # that matters: a user flattening a plateau that is already flat must
        # not spend an undo step (and a byte cost) saying so.
        return None
    return rect, np.ascontiguousarray(after)


def noise(
    heights: np.ndarray, cx: float, cz: float, radius: float, amplitude: float, *, seed: int
) -> tuple[Rect, np.ndarray] | None:
    """Add falloff-shaped noise, deterministic from ``seed``.

    ``numpy.random.default_rng(seed)`` rather than the legacy global state:
    two runs of the same document -- the whole point of a seeded brush -- must
    agree, and a generator seeded fresh here cannot have been advanced by
    anything else the session did first.
    """
    if _unusable(cx, cz, radius, amplitude):
        return None
    rect = _clip_circle(heights.shape, cx, cz, radius)
    if rect is None:
        return None
    x0, y0, x1, y1 = rect
    weight = _falloff_weight(rect, cx, cz, radius, "smooth")
    rng = np.random.default_rng(seed)
    field_noise = rng.uniform(-1.0, 1.0, size=(y1 - y0, x1 - x0)).astype(np.float32)
    before = heights[y0:y1, x0:x1]
    after = (before + field_noise * weight * np.float32(amplitude)).astype(np.float32)
    if np.array_equal(before, after):
        return None
    return rect, np.ascontiguousarray(after)


def set_height(heights: np.ndarray, rect: Rect, level: float) -> tuple[Rect, np.ndarray] | None:
    """Set every cell in ``rect`` to exactly ``level`` -- a hard edit, no falloff."""
    x0, y0, x1, y1 = rect
    if not all(np.isfinite(v) for v in (x0, y0, x1, y1, level)):
        # No division here to produce a NaN on its own, but ``rect`` reaches
        # this the same way a brush centre reaches ``_clip_circle``: a rect
        # built from a ray pick that came back non-finite would hit
        # ``int(x0)`` below and raise the identical ``ValueError``/
        # ``OverflowError`` a NaN or infinite mouse-ray-parallel-to-the-plane
        # centre does. A non-finite ``level`` would still write a NaN or inf
        # straight into the ground even with a perfectly good rect.
        return None
    rows_n, cols_n = heights.shape
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(cols_n, int(x1)), min(rows_n, int(y1))
    if x0 >= x1 or y0 >= y1:
        return None
    before = heights[y0:y1, x0:x1]
    after = np.full_like(before, np.float32(level))
    if np.array_equal(before, after):
        return None
    return (x0, y0, x1, y1), np.ascontiguousarray(after)


# --- the one conversion out -------------------------------------------------


def terrain_mesh(terrain: Terrain) -> gltf.Primitive:
    """A regular grid over ``terrain``, as a drawable :class:`gltf.Primitive`.

    ``(n+1)**2`` vertices at ``x = (i/n - 0.5) * size_x``, ``z = (j/n - 0.5) *
    size_z``, ``y = heights[j, i]``; ``2*n**2`` triangles.

    **Winding.** Per cell, the two triangles are ``(v00, v01, v10)`` and
    ``(v10, v01, v11)`` where ``v00``/``v10``/``v01``/``v11`` are the corners
    at ``(i, j)``/``(i+1, j)``/``(i, j+1)``/``(i+1, j+1)``. For a flat terrain
    that winds ``cross(v01 - v00, v10 - v00)`` to ``+Y``: this project's scene
    is Y-up and right-handed, the viewport looks down at the ground from
    above, and a triangle wound the other way would cull as backfacing the
    instant back-face culling is ever turned on for terrain the way it already
    is for library meshes.

    **Normals** are central differences of the height field converted to the
    classic heightfield normal ``normalize(-dh/dx, 1, -dh/dz)``, one-sided at
    the four border rows/columns. A wrapped neighbour there (index ``-1``
    read as "the far edge") would compute a slope across the seam between a
    terrain's two ends that do not actually meet, which is a lighting crease
    along the whole border -- so the edges get a forward/backward difference
    instead of no difference at all, not a wrapped one.

    **Memoized on the instance, not on ``id(heights)``.** An id is an address
    CPython is free to hand to a *different* array the moment the old one is
    collected, so a memo keyed on it can validate against the wrong object --
    exactly the bug ``viewer/picking.cached_bvh`` was written to stop
    repeating, and the plan's own first draft asked for it again. Holding
    ``(array, primitive)`` on the ``Terrain`` instance and checking
    ``cached_array is terrain.heights`` compares against an object this
    method is itself keeping alive, so there is no address to recycle out
    from under it. This is exactly ``gltf.Primitive._box``'s trick one level
    up the stack.
    """
    cached = terrain._mesh_cache
    if cached is not None and cached[0] is terrain.heights:
        return cached[1]
    primitive = _build_mesh(terrain)
    terrain._mesh_cache = (terrain.heights, primitive)
    return primitive


def _build_mesh(terrain: Terrain) -> gltf.Primitive:
    n = terrain.side
    heights = terrain.heights
    dx = terrain.size_x / n
    dz = terrain.size_z / n

    i = np.arange(n + 1, dtype=np.float32)
    x = (i / n - 0.5) * np.float32(terrain.size_x)
    z = (i / n - 0.5) * np.float32(terrain.size_z)
    xs = np.broadcast_to(x[None, :], (n + 1, n + 1))
    zs = np.broadcast_to(z[:, None], (n + 1, n + 1))
    # positions[j, i] == (x[i], heights[j, i], z[j]) -- height in the middle
    # slot because the mesh is Y-up.
    positions = np.stack([xs, heights, zs], axis=-1).reshape(-1, 3).astype(np.float32)

    dhdx = np.empty_like(heights)
    dhdx[:, 1:-1] = (heights[:, 2:] - heights[:, :-2]) / np.float32(2 * dx)
    dhdx[:, 0] = (heights[:, 1] - heights[:, 0]) / np.float32(dx)
    dhdx[:, -1] = (heights[:, -1] - heights[:, -2]) / np.float32(dx)

    dhdz = np.empty_like(heights)
    dhdz[1:-1, :] = (heights[2:, :] - heights[:-2, :]) / np.float32(2 * dz)
    dhdz[0, :] = (heights[1, :] - heights[0, :]) / np.float32(dz)
    dhdz[-1, :] = (heights[-1, :] - heights[-2, :]) / np.float32(dz)

    normals = np.stack([-dhdx, np.ones_like(dhdx), -dhdz], axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals.reshape(-1, 3).astype(np.float32)

    uvs = np.stack(np.broadcast_arrays(i[None, :] / n, i[:, None] / n), axis=-1)
    uvs = uvs.reshape(-1, 2).astype(np.float32)

    row = np.arange(n)
    col = np.arange(n)
    rr, cc = np.meshgrid(row, col, indexing="ij")
    v00 = (rr * (n + 1) + cc).ravel()
    v10 = (rr * (n + 1) + cc + 1).ravel()
    v01 = ((rr + 1) * (n + 1) + cc).ravel()
    v11 = ((rr + 1) * (n + 1) + cc + 1).ravel()
    indices = np.empty(n * n * 6, dtype=np.uint32)
    indices[0::6] = v00
    indices[1::6] = v01
    indices[2::6] = v10
    indices[3::6] = v10
    indices[4::6] = v01
    indices[5::6] = v11

    return gltf.Primitive(
        positions=positions,
        indices=indices,
        normals=normals,
        uvs=uvs,
        material=terrain.material,
    )


# --- sampling ----------------------------------------------------------------


def height_at(terrain: Terrain, x: float, z: float) -> float:
    """Bilinear height at local ``(x, z)``, clamped to the terrain's extent.

    A query off the edge answers with the edge rather than raising: this is
    what a "drop object to ground" op calls once per object in
    ``ops.py``, and a refusal partway through placing a multi-select has
    nothing sensible to show -- the object lands at the nearest ground the
    terrain actually has.
    """
    n = terrain.side
    heights = terrain.heights
    u = (x / terrain.size_x + 0.5) * n
    v = (z / terrain.size_z + 0.5) * n
    u = min(max(u, 0.0), float(n))
    v = min(max(v, 0.0), float(n))
    i0 = min(int(u), n - 1)
    j0 = min(int(v), n - 1)
    fu = u - i0
    fv = v - j0
    h00 = heights[j0, i0]
    h10 = heights[j0, i0 + 1]
    h01 = heights[j0 + 1, i0]
    h11 = heights[j0 + 1, i0 + 1]
    top = h00 + (h10 - h00) * fu
    bottom = h01 + (h11 - h01) * fu
    return float(top + (bottom - top) * fv)
