"""Facts about one or more Clay objects: bounds, mass properties, contact and
overlap -- measured, never selected.

**Analyze is not diagnose wearing a new name.** :mod:`~.diagnose` answers "is
this mesh broken" with rows a panel can draw and a click can select --
:class:`~.diagnose.Finding` carries an :class:`~.elements.ElementSel` for
exactly that reason. This module answers a different family of question --
"how big is it", "is it touching the ground", "do these two overlap, and by
how much" -- and every answer here is a number or a boolean, never a
selection. The two modules share exactly one thing,
:func:`~.adjacency.check_manifold`, for the one fact both need: whether a mesh
is closed. Nothing here knows about :class:`~.document.ClayDoc` or a session
either -- it takes the objects it is asked about and nothing else, so a later
program evaluator that wants to check a relationship between two named
objects mid-script can call :func:`analyze` directly rather than going through
a tool call.

**Bounds are exact here, and conservative in ``clay_scene``.**
:func:`~.ops.world_box` transforms the mesh's own *local* box's eight corners,
which is O(1) and correct for an axis-aligned object but over-reports a
rotated one -- a box tilted 45 degrees gets the box around its own tilted
box. That is the right answer for a properties panel asking "how much room
does this take up," and the wrong one for "what is this object's own extent" --
so this module transforms every *vertex* into world space instead, at the
O(V) :func:`~.ops.world_positions` already pays, and takes the box of that.

**Objects are duck-typed** on ``uid``, ``name``, ``mesh``, ``translation``,
``rotation`` and ``scale`` -- :class:`~.document.Obj`'s shape, but never
imported, for :func:`~.diagnose.scene_findings`'s own reason: this reads a
document without importing one.

**Pairs are two-phase, like every broad-narrow collision scheme.** A
sort-and-sweep over *near*-expanded world boxes finds which object pairs are
worth looking at closely; the ones that pass are triangle-grid-binned and
tested for real -- Moller-style separating-axis intersection, and (when they
do not overlap) the exact minimum distance, taken over every vertex-to-triangle
and edge-to-edge candidate the grid turned up. That is the same "broad phase
narrows what the narrow phase pays for" shape :mod:`~.ops_boolean` and every
other real-time collision library uses, just applied to a *report* instead of
a response.

**Overlap volume reuses the CSG kernel, not a second one.** ``manifold3d`` via
``trimesh.boolean`` is already how :func:`~.ops_boolean.boolean` computes a
real intersection; :func:`_overlap` below builds the same
``trimesh.Trimesh`` conversion :func:`~.ops_boolean._run` already does (over
world-space copies of the two meshes) and calls it, rather than a second
private CSG path that would need its own bugs found. It only runs when both
meshes are closed and the cheap SAT test already says they intersect, and it
is capped at :data:`MAX_OVERLAP_BOOLEANS` calls and
``ops_boolean.MAX_BOOLEAN_TRIANGLES`` triangles per call -- the existing
boolean ceiling, not a new lower one, for the same "no worse than one import
already allows" reasoning :mod:`~.ops_boolean` states for its own cap.

**Three ceilings, all refusals except the middle one.**
:data:`MAX_ANALYZE_OBJECTS` and :data:`MAX_ANALYZE_TRIANGLES` refuse outright,
the same shape :func:`~.ops_boolean._refuse_complexity` uses -- a call this
expensive would stall the frame it ran on for a fact-finding read, which is a
worse trade than telling the caller to narrow the selection.
:data:`MAX_TRIANGLE_PAIRS` does not refuse: past it, a pair's distance is
answered from a nearest-neighbour query over the raw vertices instead of the
full narrow phase, marked ``exact=False``, and ``intersects`` is reported as
``None`` (unknown) rather than guessed, because a whole-document call is
still worth answering approximately rather than not at all. See
``dev/INVARIANTS.md``'s accepted-stall list for why that approximate path
still exists rather than a fourth refusal: a big analysis is a deliberate
one-shot action, like a big ``clay_boolean``, not something that should be
unreachable.

**Floating is a graph over objects and one ground node**, built only for a
whole-document call (``pairs_among=None``): an edge for every ``contact``
pair, an edge from an object straight to the ground node when its own
``ground.contact`` is true, and anything the resulting search cannot reach
from the ground is floating. Object granularity, not sub-object component
granularity, even though :attr:`ObjectAnalysis.components` measures the
latter -- pairwise contact is already only computed between whole objects, and
extending it to test every component of one object against every component of
another would multiply the narrow-phase cost for a case this module's own
callers have not needed yet.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..geom3d import math3d as m3
from .adjacency import adjacency as mesh_adjacency
from .adjacency import cached_triangulation, check_manifold
from .elements import OpError
from .mesh import face_count

__all__ = [
    "MAX_ANALYZE_OBJECTS",
    "MAX_ANALYZE_TRIANGLES",
    "MAX_OVERLAP_BOOLEANS",
    "MAX_TRIANGLE_PAIRS",
    "Analysis",
    "GroundInfo",
    "ObjectAnalysis",
    "OverlapInfo",
    "PairAnalysis",
    "analyze",
]

MAX_ANALYZE_OBJECTS = 64
"""The most objects one :func:`analyze` call may be asked about at once.
Past it, every pairwise cost below scales with the square of this number, and
a caller with more than this many objects to compare should be narrowing with
``uids`` rather than asking for one answer over the whole document."""

MAX_ANALYZE_TRIANGLES = 200_000
"""The most triangles, summed across every given object, one call may face.
Bounds and mass properties are O(V) per object regardless, but the pairwise
narrow phase below is what this really guards -- a document at this ceiling
already gives the grid-binned triangle search real work to do."""

MAX_TRIANGLE_PAIRS = 500_000
"""Past this many candidate triangle pairs for one object pair (after the
grid-binning narrow phase, not before it), the exact vertex/edge distance
search -- and the SAT intersection test alongside it -- is skipped in favour
of a nearest-neighbour query over the two objects' raw vertices -- cheaper,
and honestly reported as ``exact=False``. ``intersects`` is reported as
``None`` (unknown) on this path rather than ``False``: the 2026-09-14 audit's
clay-01 found two heavily-overlapping 20,000-face spheres reading as
``intersects=False`` here, indistinguishable from an honest "checked and
clear" -- ``exact=False`` was documented as covering ``distance`` only, so a
caller had no way to tell "not checked" from "checked and found nothing."""

MAX_OVERLAP_BOOLEANS = 16
"""The most ``manifold3d`` intersections one :func:`analyze` call may run.
Each one is already bounded per-call by ``ops_boolean.MAX_BOOLEAN_TRIANGLES``;
this bounds how many of them a single whole-document call may queue, so a
scene with many mutually-overlapping objects cannot turn one read into
dozens of CSG kernel calls."""

_GRID_MIN_CELL = 1e-3
"""A floor under the triangle-grid cell size, so a *near* of 0.0 (a caller
asking only about touching objects) does not divide the world into
infinitely many cells."""

_VERTEX_SAMPLE_CAP = 2000
"""How many vertices the nearest-neighbour fallback (past
:data:`MAX_TRIANGLE_PAIRS`, or when the grid found no candidate at all) reads
from each side. A resampled subsequence, not a random one, so the same two
meshes give the same answer on every call."""

_MAX_CELLS_PER_TRIANGLE_AXIS = 64
"""A ceiling on how many grid cells one triangle's own AABB may span along
any axis in :func:`_grid_candidates`, enforced by growing the shared cell
size (never shrinking it, and never per-triangle) rather than bounding the
registration loop itself. The 2026-09-14 audit's clay-02 found nothing
bounded this: one large, thin triangle (an 80 m floor plate at the default
0.05 m cell) registered into millions of cells in the pure-Python loop below
-- 14 triangles took 2.35 s, a 160 m floor 10.6 s -- with none of
MAX_ANALYZE_OBJECTS / MAX_ANALYZE_TRIANGLES / MAX_TRIANGLE_PAIRS bounding it,
because the blow-up happens before any of them are even checked. Growing the
cell keeps :func:`_grid_candidates` exact for "candidate" -- the module
docstring's "for any cell size" proof holds for a *larger* cell just as well,
so this can only add candidates, never drop a pair whose boxes truly
overlap."""

_MAX_GRID_REGISTRATIONS = 2_000_000
"""A ceiling on the *total* number of triangle-into-cell registrations one
:func:`_grid_candidates` call may perform, summed across every triangle on
both sides, from each triangle's own cell-span product -- checked before
either bucket loop below runs, not after. :data:`_MAX_CELLS_PER_TRIANGLE_AXIS`
only bounds what *one* triangle can cost; the 2026-09-16 audit found nothing
bounded the total across *many* triangles that are each individually under
that cap -- an ordinary blockout floor tiled from 20,000 separate 10 m
quads (each on its own saturating the per-triangle cap, the way an authored
level's floor plates routinely do) measured 5.8 s in the loops below, at
only 10% of MAX_ANALYZE_TRIANGLES, with the per-triangle cap doing nothing
to stop it because no *single* triangle ever crossed it. Past this ceiling
:func:`_grid_candidates` returns ``None`` and the pair falls back to the same
vertex-sampled approximation :data:`MAX_TRIANGLE_PAIRS` already uses --
honestly reported ``exact=False`` -- rather than a fourth outright refusal,
for the same "a whole-document call is still worth answering approximately"
reasoning that ceiling's own docstring states."""


@dataclass(frozen=True)
class GroundInfo:
    """Where one object sits relative to the ground plane, ``y = 0``."""

    min_y: float
    contact: bool
    penetration: float


@dataclass(frozen=True)
class OverlapInfo:
    """The solid a pair's intersection encloses, when both sides are closed
    and the cheap test already says they cross."""

    volume: float
    depth: float


@dataclass(frozen=True)
class ObjectAnalysis:
    """Everything :func:`analyze` says about one object.

    ``bounds`` is ``(lo, hi)`` in world space, or ``None`` for an object with
    no vertices. ``volume`` is ``None`` unless ``closed``. ``symmetry`` is
    ``(x, y, z)``, the fraction of vertices with a mirror partner across the
    plane through the object's own bounds centre, perpendicular to that axis.
    """

    uid: int
    name: str
    bounds: tuple[np.ndarray, np.ndarray] | None
    area: float
    volume: float | None
    closed: bool
    components: int
    ground: GroundInfo | None
    symmetry: tuple[float, float, float]


@dataclass(frozen=True)
class PairAnalysis:
    """Everything :func:`analyze` says about one pair of objects.

    ``distance`` is the exact minimum surface distance when ``exact`` is
    true, or a vertex-only nearest-neighbour estimate when it is not.
    ``distance`` is ``0.0`` whenever ``intersects`` is true. ``overlap`` is
    ``None`` unless both objects are closed and ``intersects`` is true and
    the boolean budget had room left.

    ``intersects`` is ``None`` -- unknown, not "no" -- when the pair cleared
    :data:`MAX_TRIANGLE_PAIRS` and the SAT test was skipped along with the
    exact distance search (see that constant's docstring: the 2026-09-14
    audit's clay-01). Every other path answers ``True``/``False`` for real.
    """

    uids: tuple[int, int]
    distance: float | None
    intersects: bool | None
    contact: bool
    overlap: OverlapInfo | None
    exact: bool


@dataclass(frozen=True)
class Analysis:
    """The whole answer: every requested object, every pair worth reporting,
    and (only for a whole-document call) which objects are floating."""

    objects: tuple[ObjectAnalysis, ...]
    pairs: tuple[PairAnalysis, ...]
    floating: tuple[int, ...] | None
    truncated: bool


@dataclass
class _Geom:
    """Per-object working state, computed once and shared between the
    object-level and pair-level passes below."""

    world_pos: np.ndarray  # (V, 3) f8
    tris: np.ndarray  # (T, 3) i8, indices into world_pos
    lo: np.ndarray | None  # (3,) f8, or None for an empty mesh
    hi: np.ndarray | None
    diag: float


def _world_positions(obj: Any) -> np.ndarray:
    matrix = m3.compose(obj.translation, obj.rotation, obj.scale)
    positions = np.asarray(obj.mesh.positions, dtype="f8")
    if len(positions) == 0:
        return np.zeros((0, 3), dtype="f8")
    homogeneous = np.hstack([positions, np.ones((len(positions), 1))])
    return (matrix @ homogeneous.T).T[:, :3]


def _geometry(obj: Any) -> _Geom:
    world_pos = _world_positions(obj)
    tris, _face = cached_triangulation(obj.mesh)
    if len(tris):
        tris = np.asarray(tris, dtype="i8").reshape(-1, 3)
    else:
        tris = np.zeros((0, 3), dtype="i8")
    if len(world_pos) == 0:
        return _Geom(world_pos=world_pos, tris=tris, lo=None, hi=None, diag=0.0)
    lo = world_pos.min(axis=0)
    hi = world_pos.max(axis=0)
    diag = float(np.linalg.norm(hi - lo))
    return _Geom(world_pos=world_pos, tris=tris, lo=lo, hi=hi, diag=diag)


def _is_closed(mesh: Any) -> bool:
    """No hole, no non-manifold edge -- the same reading ``clay_add_mesh``'s
    own ``closed`` uses, and the one fact this module shares with
    :mod:`~.diagnose` rather than recomputing its own version of."""
    if len(mesh.starts) <= 1:
        return False
    report = check_manifold(mesh)
    return len(report.boundary_edges) == 0 and len(report.nonmanifold_edges) == 0


def _triangle_areas(tri_pts: np.ndarray) -> np.ndarray:
    if len(tri_pts) == 0:
        return np.zeros(0, dtype="f8")
    e1 = tri_pts[:, 1] - tri_pts[:, 0]
    e2 = tri_pts[:, 2] - tri_pts[:, 0]
    return 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)


def _signed_volume(tri_pts: np.ndarray) -> float:
    """The divergence-theorem volume of a closed, consistently-wound
    triangle soup, computed directly in whatever space *tri_pts* is given
    in -- world space here, so a non-uniform scale is already accounted for
    without a separate determinant correction."""
    if len(tri_pts) == 0:
        return 0.0
    v0, v1, v2 = tri_pts[:, 0], tri_pts[:, 1], tri_pts[:, 2]
    return float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum()) / 6.0


def _ground(world_pos: np.ndarray, contact_tol: float) -> GroundInfo:
    min_y = float(world_pos[:, 1].min())
    penetration = float(max(0.0, -min_y))
    contact = abs(min_y) <= contact_tol
    return GroundInfo(min_y=min_y, contact=bool(contact), penetration=penetration)


def _symmetry(
    world_pos: np.ndarray, lo: np.ndarray, hi: np.ndarray, tol_abs: float
) -> tuple[float, float, float]:
    """The fraction of vertices with a mirror partner, per axis, mirrored
    about the object's own bounds centre -- via a single ``cKDTree`` built
    once and queried three times, one per candidate mirror plane."""
    from scipy.spatial import cKDTree

    n = len(world_pos)
    if n == 0:
        return (0.0, 0.0, 0.0)
    center = (lo + hi) * 0.5
    tree = cKDTree(world_pos)
    out = []
    for axis in range(3):
        mirrored = world_pos.copy()
        mirrored[:, axis] = 2.0 * center[axis] - mirrored[:, axis]
        dist, _ = tree.query(mirrored, k=1)
        out.append(float(np.mean(dist <= tol_abs)))
    return (out[0], out[1], out[2])


def _components(world_pos: np.ndarray, mesh: Any, diag: float) -> int:
    """Connected components after welding vertices within ``1e-6 * diag`` of
    each other -- a mesh made of several disjoint pieces (an object built
    from more than one primitive and never merged) reports more than one.

    The weld is a lattice quantisation rather than a distance query: every
    vertex is rounded to the nearest multiple of the weld tolerance, and
    consecutive vertices sharing a rounded key (after sorting on it) are
    chained together with a graph edge -- enough to fully connect a bucket
    without needing every pair inside it, the same trick a union-find over a
    sorted key column always uses. Mesh topology (:attr:`~.adjacency.
    Adjacency.edge_verts`) supplies the rest of the graph, and
    ``scipy.sparse.csgraph.connected_components`` does the counting.
    """
    n = len(world_pos)
    if n == 0:
        return 0
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components as _cc

    edges = np.asarray(mesh_adjacency(mesh).edge_verts, dtype="i8").reshape(-1, 2)
    eps = max(1e-6 * diag, 1e-9)
    keys = np.round(world_pos / eps).astype("i8")
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    sorted_keys = keys[order]
    if n > 1:
        same = np.all(sorted_keys[1:] == sorted_keys[:-1], axis=1)
        weld_src = order[1:][same]
        weld_dst = order[:-1][same]
    else:
        weld_src = weld_dst = np.zeros(0, dtype="i8")
    row = np.concatenate([edges[:, 0], edges[:, 1], weld_src, weld_dst]).astype("i8")
    col = np.concatenate([edges[:, 1], edges[:, 0], weld_dst, weld_src]).astype("i8")
    if len(row) == 0:
        return n
    data = np.ones(len(row), dtype="i1")
    graph = csr_matrix((data, (row, col)), shape=(n, n))
    n_comp, _labels = _cc(graph, directed=False)
    return int(n_comp)


def _object_analysis(
    obj: Any, geom: _Geom, contact_tol: float, symmetry_tol: float
) -> ObjectAnalysis:
    if geom.lo is None:
        return ObjectAnalysis(
            uid=obj.uid,
            name=obj.name,
            bounds=None,
            area=0.0,
            volume=None,
            closed=False,
            components=0,
            ground=None,
            symmetry=(0.0, 0.0, 0.0),
        )
    closed = _is_closed(obj.mesh)
    tri_pts = geom.world_pos[geom.tris] if len(geom.tris) else np.zeros((0, 3, 3), dtype="f8")
    area = float(_triangle_areas(tri_pts).sum())
    volume = abs(_signed_volume(tri_pts)) if closed else None
    components = _components(geom.world_pos, obj.mesh, geom.diag)
    ground = _ground(geom.world_pos, contact_tol)
    symmetry = _symmetry(geom.world_pos, geom.lo, geom.hi, symmetry_tol * geom.diag)
    return ObjectAnalysis(
        uid=obj.uid,
        name=obj.name,
        bounds=(geom.lo, geom.hi),
        area=area,
        volume=volume,
        closed=closed,
        components=components,
        ground=ground,
        symmetry=symmetry,
    )


# --- broad phase: which object pairs are worth a close look -----------------


def _broad_phase_pairs(
    uids: Sequence[int], boxes: Sequence[tuple[np.ndarray, np.ndarray]]
) -> list[tuple[int, int]]:
    """Sort-and-sweep on *near*-expanded world boxes. -> uid pairs whose
    boxes overlap, low uid first.

    Sorted once on the low X corner; the sweep only has to look forward from
    each box until a later one's low X passes this one's high X, because
    nothing past that point can overlap on X either. The remaining two axes
    are then a plain interval check -- classic sweep-and-prune, worth doing
    even at :data:`MAX_ANALYZE_OBJECTS`'s modest scale because the
    alternative is a same-shaped nested loop with no early exit at all.
    """
    n = len(uids)
    order = sorted(range(n), key=lambda i: boxes[i][0][0])
    out: list[tuple[int, int]] = []
    for a_pos in range(n):
        i = order[a_pos]
        lo_i, hi_i = boxes[i]
        for b_pos in range(a_pos + 1, n):
            j = order[b_pos]
            lo_j, hi_j = boxes[j]
            if lo_j[0] > hi_i[0]:
                break
            if lo_j[1] > hi_i[1] or lo_i[1] > hi_j[1]:
                continue
            if lo_j[2] > hi_i[2] or lo_i[2] > hi_j[2]:
                continue
            a, b = uids[i], uids[j]
            out.append((a, b) if a < b else (b, a))
    return out


# --- narrow phase: triangle-grid binning, SAT, exact distance ---------------


def _grid_candidates(
    tri_a: np.ndarray, tri_b: np.ndarray, cell: float
) -> tuple[np.ndarray, np.ndarray] | None:
    """Candidate triangle index pairs whose axis-aligned boxes could overlap.

    Each triangle is registered in *every* cell its own box touches, not
    just the one its centroid falls in -- so two triangles whose boxes truly
    overlap always share at least one cell, for any cell size: the
    intersection region (if the boxes overlap at all) contains a point, and
    that point's cell lies inside both triangles' own registered range. That
    is what makes this exact for "candidate," unlike a centroid-plus-
    neighbour scheme, which misses a pair whenever one triangle's own box is
    wider than the neighbour search reaches -- measured on this module's own
    fixture: a slab's single top-face triangle, spanning metres, sharing no
    cell with a small object resting on it at the default 0.05 m cell.

    The smaller side is binned into a dict first, so the dict this builds is
    no bigger than it has to be; the larger side then only ever *looks up*.

    Returns ``None`` -- instead of candidate arrays -- when
    :data:`_MAX_GRID_REGISTRATIONS` would be crossed; see that constant for
    why a per-triangle cap alone does not bound this. The caller treats that
    exactly like the :data:`MAX_TRIANGLE_PAIRS` overflow it already handles.
    """
    if len(tri_a) == 0 or len(tri_b) == 0:
        return np.zeros(0, dtype="i8"), np.zeros(0, dtype="i8")

    # See _MAX_CELLS_PER_TRIANGLE_AXIS: grow (never shrink) the cell so no
    # single triangle's own AABB can register into more than roughly
    # (K + 1)^3 cells, however small *cell* (derived from *near*) is.
    max_extent = float(
        max(
            (tri_a.max(axis=1) - tri_a.min(axis=1)).max(),
            (tri_b.max(axis=1) - tri_b.min(axis=1)).max(),
        )
    )
    if max_extent > 0.0:
        cell = max(cell, max_extent / _MAX_CELLS_PER_TRIANGLE_AXIS)

    if len(tri_b) <= len(tri_a):
        small, large, swapped = tri_b, tri_a, False
    else:
        small, large, swapped = tri_a, tri_b, True

    lo_s = np.floor(small.min(axis=1) / cell).astype("i8")
    hi_s = np.floor(small.max(axis=1) / cell).astype("i8")
    lo_l = np.floor(large.min(axis=1) / cell).astype("i8")
    hi_l = np.floor(large.max(axis=1) / cell).astype("i8")

    # See _MAX_GRID_REGISTRATIONS: estimate what both loops below will cost
    # -- one cell-span product per triangle, summed over both sides -- from
    # lo/hi alone, vectorised, before either loop runs a single Python
    # iteration; the same "refuse before the allocation" shape
    # ops_boolean._refuse_complexity uses for the triangle count it would
    # hand the CSG kernel.
    span_s = (hi_s - lo_s + 1).astype(np.int64)
    span_l = (hi_l - lo_l + 1).astype(np.int64)
    total_registrations = int(
        (span_s[:, 0] * span_s[:, 1] * span_s[:, 2]).sum()
        + (span_l[:, 0] * span_l[:, 1] * span_l[:, 2]).sum()
    )
    if total_registrations > _MAX_GRID_REGISTRATIONS:
        return None

    buckets: dict[tuple[int, int, int], list[int]] = {}
    for k in range(len(small)):
        for ix in range(int(lo_s[k, 0]), int(hi_s[k, 0]) + 1):
            for iy in range(int(lo_s[k, 1]), int(hi_s[k, 1]) + 1):
                for iz in range(int(lo_s[k, 2]), int(hi_s[k, 2]) + 1):
                    buckets.setdefault((ix, iy, iz), []).append(k)

    out_small: list[int] = []
    out_large: list[int] = []
    for m in range(len(large)):
        seen: set[int] = set()
        for ix in range(int(lo_l[m, 0]), int(hi_l[m, 0]) + 1):
            for iy in range(int(lo_l[m, 1]), int(hi_l[m, 1]) + 1):
                for iz in range(int(lo_l[m, 2]), int(hi_l[m, 2]) + 1):
                    for k in buckets.get((ix, iy, iz), ()):
                        if k not in seen:
                            seen.add(k)
                            out_small.append(k)
                            out_large.append(m)

    small_idx = np.asarray(out_small, dtype="i8")
    large_idx = np.asarray(out_large, dtype="i8")
    # swapped=True means small==tri_a, large==tri_b, so out_small already
    # indexes tri_a and out_large already indexes tri_b -- (ia, ib) as is.
    # swapped=False means small==tri_b, large==tri_a, so it is the other
    # way around.
    return (small_idx, large_idx) if swapped else (large_idx, small_idx)


def _tri_tri_intersect(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Batched triangle-triangle intersection via separating-axis test.

    Eleven candidate axes -- each triangle's own face normal, plus the nine
    cross products of one edge from each triangle -- checked in turn; a pair
    intersects unless some axis separates their projected intervals. A
    degenerate (near-zero-length) axis is skipped rather than treated as
    separating, which is what a pair of edges that happen to be parallel
    would otherwise produce.
    """
    k = len(a)
    intersecting = np.ones(k, dtype=bool)
    if k == 0:
        return intersecting
    n1 = np.cross(a[:, 1] - a[:, 0], a[:, 2] - a[:, 0])
    n2 = np.cross(b[:, 1] - b[:, 0], b[:, 2] - b[:, 0])
    edges_a = [a[:, 1] - a[:, 0], a[:, 2] - a[:, 1], a[:, 0] - a[:, 2]]
    edges_b = [b[:, 1] - b[:, 0], b[:, 2] - b[:, 1], b[:, 0] - b[:, 2]]
    axes = [n1, n2]
    for ea in edges_a:
        for eb in edges_b:
            axes.append(np.cross(ea, eb))

    eps = 1e-9
    for axis in axes:
        length = np.linalg.norm(axis, axis=1)
        valid = length > 1e-12
        safe_len = np.where(valid, length, 1.0)
        axis_n = axis / safe_len[:, None]
        proj_a = np.einsum("kij,kj->ki", a, axis_n)
        proj_b = np.einsum("kij,kj->ki", b, axis_n)
        a_min, a_max = proj_a.min(axis=1), proj_a.max(axis=1)
        b_min, b_max = proj_b.min(axis=1), proj_b.max(axis=1)
        separated = valid & ((a_max < b_min - eps) | (b_max < a_min - eps))
        intersecting &= ~separated
    return intersecting


def _closest_point_triangle(p: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Ericson's ``ClosestPtPointTriangle``, batched. *p*: (K, 3); *tri*:
    (K, 3, 3). -> distance (K,). Only the distance is kept -- no caller here
    wants the point itself."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = p - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp = p - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)

    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4

    k = len(p)
    out = np.empty((k, 3), dtype="f8")
    done = np.zeros(k, dtype=bool)

    with np.errstate(invalid="ignore", divide="ignore"):
        case = (d1 <= 0) & (d2 <= 0)
        out[case] = a[case]
        done |= case

        case = (~done) & (d3 >= 0) & (d4 <= d3)
        out[case] = b[case]
        done |= case

        case = (~done) & (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        denom = d1 - d3
        v = np.where(denom != 0, d1 / denom, 0.0)
        out[case] = (a + v[:, None] * ab)[case]
        done |= case

        case = (~done) & (d6 >= 0) & (d5 <= d6)
        out[case] = c[case]
        done |= case

        case = (~done) & (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        denom = d2 - d6
        w = np.where(denom != 0, d2 / denom, 0.0)
        out[case] = (a + w[:, None] * ac)[case]
        done |= case

        bc_num = (d4 - d3)
        bc_num2 = (d5 - d6)
        case = (~done) & (va <= 0) & (bc_num >= 0) & (bc_num2 >= 0)
        denom = bc_num + bc_num2
        w2 = np.where(denom != 0, bc_num / denom, 0.0)
        out[case] = (b + w2[:, None] * (c - b))[case]
        done |= case

        inside = ~done
        denom = va + vb + vc
        inv = np.where(denom != 0, 1.0 / denom, 0.0)
        v_in = vb * inv
        w_in = vc * inv
        out[inside] = (a + ab * v_in[:, None] + ac * w_in[:, None])[inside]

    return np.linalg.norm(p - out, axis=1)


def _segment_segment_distance(
    p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray
) -> np.ndarray:
    """Ericson's ``ClosestPtSegmentSegment``, batched, distance only."""
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = np.einsum("ij,ij->i", d1, d1)
    e = np.einsum("ij,ij->i", d2, d2)
    f = np.einsum("ij,ij->i", d2, r)
    c = np.einsum("ij,ij->i", d1, r)
    b = np.einsum("ij,ij->i", d1, d2)

    k = len(p1)
    eps = 1e-12
    a_deg = a <= eps
    e_deg = e <= eps

    s = np.zeros(k, dtype="f8")
    t = np.zeros(k, dtype="f8")

    with np.errstate(invalid="ignore", divide="ignore"):
        only_a = a_deg & ~e_deg
        t[only_a] = np.clip(np.where(e > eps, f / e, 0.0), 0.0, 1.0)[only_a]

        only_e = e_deg & ~a_deg
        s[only_e] = np.clip(np.where(a > eps, -c / a, 0.0), 0.0, 1.0)[only_e]

        general = ~a_deg & ~e_deg
        denom = a * e - b * b
        s_gen = np.where(denom > eps, np.clip((b * f - c * e) / denom, 0.0, 1.0), 0.0)
        t_gen = np.where(e > eps, (b * s_gen + f) / e, 0.0)

        below = t_gen < 0.0
        above = t_gen > 1.0
        t_gen = np.clip(t_gen, 0.0, 1.0)
        s_below = np.where(a > eps, np.clip(-c / a, 0.0, 1.0), 0.0)
        s_above = np.where(a > eps, np.clip((b - c) / a, 0.0, 1.0), 0.0)
        s_gen = np.where(below, s_below, np.where(above, s_above, s_gen))

        s[general] = s_gen[general]
        t[general] = t_gen[general]

    closest1 = p1 + d1 * s[:, None]
    closest2 = p2 + d2 * t[:, None]
    return np.linalg.norm(closest1 - closest2, axis=1)


def _min_triangle_distance(cand_a: np.ndarray, cand_b: np.ndarray) -> float:
    """The minimum distance over every candidate triangle pair: each side's
    three vertices against the other's triangle, plus every edge-to-edge
    pair -- fifteen vectorised passes over the whole candidate set rather
    than a Python loop per pair."""
    best = np.inf
    for v in range(3):
        d = _closest_point_triangle(cand_a[:, v], cand_b)
        if len(d):
            best = min(best, float(d.min()))
    for v in range(3):
        d = _closest_point_triangle(cand_b[:, v], cand_a)
        if len(d):
            best = min(best, float(d.min()))
    corners = ((0, 1), (1, 2), (2, 0))
    for ea0, ea1 in corners:
        for eb0, eb1 in corners:
            d = _segment_segment_distance(
                cand_a[:, ea0], cand_a[:, ea1], cand_b[:, eb0], cand_b[:, eb1]
            )
            if len(d):
                best = min(best, float(d.min()))
    return best


def _vertex_sampled_distance(pos_a: np.ndarray, pos_b: np.ndarray) -> float | None:
    """A cheap, approximate fallback: nearest-neighbour distance over a
    resampled subsequence of each side's raw vertices, via one ``cKDTree``.
    Ignores triangle interiors, so it can only over-report the true
    distance -- which is why every caller of this marks its result
    ``exact=False``."""
    if len(pos_a) == 0 or len(pos_b) == 0:
        return None
    from scipy.spatial import cKDTree

    def _sample(pos: np.ndarray) -> np.ndarray:
        if len(pos) <= _VERTEX_SAMPLE_CAP:
            return pos
        idx = np.linspace(0, len(pos) - 1, _VERTEX_SAMPLE_CAP).astype("i8")
        return pos[idx]

    a = _sample(pos_a)
    b = _sample(pos_b)
    tree = cKDTree(b)
    dist, _ = tree.query(a, k=1)
    return float(np.min(dist))


def _overlap(geom_a: _Geom, geom_b: _Geom, mesh_a: Any, mesh_b: Any) -> OverlapInfo | None:
    """The volume and depth of *A* intersect *B*, in world space, via the
    same ``manifold3d``-backed conversion :func:`~.ops_boolean._run` already
    uses -- built over world-space copies of both meshes rather than the
    frame ``ops_boolean.boolean`` itself works in, since there is no "first
    object's frame" to prefer for a read that reports about both equally.

    ``depth`` is the overlap solid's own smallest world-axis extent -- for
    the case this exists to answer ("how far is A sunk into B"), the overlap
    region is thin along exactly the axis of penetration and wide along the
    others, so its own thinnest dimension *is* the sinking distance.
    """
    from . import ops_boolean
    from .mesh import Mesh, face_count

    total = (len(mesh_a.loops) - 2 * face_count(mesh_a)) + (
        len(mesh_b.loops) - 2 * face_count(mesh_b)
    )
    if total > ops_boolean.MAX_BOOLEAN_TRIANGLES:
        return None
    world_a = Mesh(
        positions=geom_a.world_pos.astype("f4"),
        loops=mesh_a.loops,
        starts=mesh_a.starts,
        material=mesh_a.material,
        smooth=mesh_a.smooth,
        uv=None,
    )
    world_b = Mesh(
        positions=geom_b.world_pos.astype("f4"),
        loops=mesh_b.loops,
        starts=mesh_b.starts,
        material=mesh_b.material,
        smooth=mesh_b.smooth,
        uv=None,
    )
    try:
        result = ops_boolean._run([world_a, world_b], ["A", "B"], kind="intersection")
    except Exception:
        # Best-effort: an overlap the kernel cannot compute (a degenerate
        # sliver, say) is reported as no overlap rather than failing the
        # whole analysis over one pair.
        return None
    verts = np.asarray(result.vertices, dtype="f8")
    if len(verts) == 0:
        return None
    volume = float(abs(result.volume))
    extent = verts.max(axis=0) - verts.min(axis=0)
    return OverlapInfo(volume=volume, depth=float(extent.min()))


def _pair_analysis(
    obj_a: Any,
    obj_b: Any,
    geom_a: _Geom,
    geom_b: _Geom,
    closed_a: bool,
    closed_b: bool,
    contact_tol: float,
    near: float,
    overlap_budget: list[int],
) -> tuple[PairAnalysis, bool]:
    uid_a, uid_b = obj_a.uid, obj_b.uid
    if geom_a.lo is None or geom_b.lo is None or len(geom_a.tris) == 0 or len(geom_b.tris) == 0:
        return (
            PairAnalysis(
                uids=(uid_a, uid_b),
                distance=None,
                intersects=False,
                contact=False,
                overlap=None,
                exact=True,
            ),
            False,
        )

    tri_a = geom_a.world_pos[geom_a.tris]
    tri_b = geom_b.world_pos[geom_b.tris]
    cell = max(near, _GRID_MIN_CELL)
    candidates = _grid_candidates(tri_a, tri_b, cell)

    if candidates is None:
        # See _MAX_GRID_REGISTRATIONS: the 2026-09-16 audit found many
        # separately-large triangles -- each individually under
        # _MAX_CELLS_PER_TRIANGLE_AXIS's own cap -- could still make this
        # pair's total registration cost stall the frame thread, because no
        # single triangle ever crossed the per-triangle cap that would have
        # caught it. Same fallback as the MAX_TRIANGLE_PAIRS overflow below:
        # an honest "unknown" rather than a hard refusal or a guessed "no".
        distance = _vertex_sampled_distance(geom_a.world_pos, geom_b.world_pos)
        contact = distance is not None and distance <= contact_tol
        return (
            PairAnalysis(
                uids=(uid_a, uid_b),
                distance=distance,
                intersects=None,
                contact=contact,
                overlap=None,
                exact=False,
            ),
            True,
        )

    ia, ib = candidates
    if len(ia) > MAX_TRIANGLE_PAIRS:
        # 2026-09-14 audit, clay-01: this used to hard-code intersects=False
        # here, so two heavily-overlapping 20,000-face meshes read as "not
        # touching" -- indistinguishable from an honest SAT "no". The SAT
        # test itself is skipped on this path (not just the exact distance
        # search), so the honest answer is "unknown", not "no".
        distance = _vertex_sampled_distance(geom_a.world_pos, geom_b.world_pos)
        contact = distance is not None and distance <= contact_tol
        return (
            PairAnalysis(
                uids=(uid_a, uid_b),
                distance=distance,
                intersects=None,
                contact=contact,
                overlap=None,
                exact=False,
            ),
            True,
        )

    if len(ia) == 0:
        # The boxes passed the broad phase, but no triangle pair shares a
        # grid cell -- still worth an approximate answer rather than none.
        distance = _vertex_sampled_distance(geom_a.world_pos, geom_b.world_pos)
        contact = distance is not None and distance <= contact_tol
        return (
            PairAnalysis(
                uids=(uid_a, uid_b),
                distance=distance,
                intersects=False,
                contact=contact,
                overlap=None,
                exact=True,
            ),
            False,
        )

    cand_a = tri_a[ia]
    cand_b = tri_b[ib]
    intersects = bool(_tri_tri_intersect(cand_a, cand_b).any())
    distance = 0.0 if intersects else _min_triangle_distance(cand_a, cand_b)
    contact = intersects or distance <= contact_tol

    overlap = None
    if intersects and closed_a and closed_b and overlap_budget[0] > 0:
        overlap = _overlap(geom_a, geom_b, obj_a.mesh, obj_b.mesh)
        overlap_budget[0] -= 1

    return (
        PairAnalysis(
            uids=(uid_a, uid_b),
            distance=distance,
            intersects=intersects,
            contact=contact,
            overlap=overlap,
            exact=True,
        ),
        False,
    )


def _floating(
    objs: Sequence[Any], object_rows: Sequence[ObjectAnalysis], pair_rows: Sequence[PairAnalysis]
) -> tuple[int, ...]:
    grounded = {row.uid for row in object_rows if row.ground is not None and row.ground.contact}
    graph: dict[int, set[int]] = {obj.uid: set() for obj in objs}
    for pair in pair_rows:
        if pair.contact:
            a, b = pair.uids
            graph.setdefault(a, set()).add(b)
            graph.setdefault(b, set()).add(a)

    reachable: set[int] = set(grounded)
    stack = list(grounded)
    while stack:
        uid = stack.pop()
        for neighbour in graph.get(uid, ()):
            if neighbour not in reachable:
                reachable.add(neighbour)
                stack.append(neighbour)

    return tuple(sorted(obj.uid for obj in objs if obj.uid not in reachable))


def analyze(
    objects: Sequence[Any],
    *,
    pairs_among: Sequence[int] | None = None,
    contact_tol: float = 0.001,
    near: float = 0.05,
    symmetry_tol: float = 0.002,
) -> Analysis:
    """Bounds, mass properties, ground contact and symmetry for every object
    in *objects*, plus pairwise distance/contact/overlap and (for a
    whole-document call) which objects are floating.

    *pairs_among*, given, restricts which of *objects* take part in the
    pairwise pass and turns floating off -- the shape a caller with a
    specific ``uids`` request wants: report fully on the named objects, but
    do not pay for (or claim to know) the whole scene's contact graph.
    ``None`` -- the default -- means every object in *objects* takes part and
    floating is computed, which is what a whole-document call wants and is
    also the only case in which "floating" means anything: an object outside
    *pairs_among*'s restriction could easily be its only support, and
    reporting on it as floating would be answering a question this call was
    never asked.

    Raises :class:`~.elements.OpError` past :data:`MAX_ANALYZE_OBJECTS` or
    :data:`MAX_ANALYZE_TRIANGLES` -- refused, not truncated, because either
    ceiling exists to keep this call from stalling the frame it runs on, and
    a stalled read is a worse answer than a refusal naming the ceiling.
    """
    objs = list(objects)
    if len(objs) > MAX_ANALYZE_OBJECTS:
        raise OpError(
            f"This analysis would need to look at {len(objs)} objects at "
            f"once, past the {MAX_ANALYZE_OBJECTS} Clay works with. Narrow "
            "the selection with uids."
        )

    # 2026-09-14 audit, clay-04: this used to sum len(geom.tris) after
    # _geometry(obj) -- via cached_triangulation -- had already triangulated
    # every object, so a call this refuses still paid the O(n^2) ear-clip
    # cost on every concave mesh first. An n-cornered face always
    # triangulates into n - 2 triangles (mesh.triangulate's own docstring),
    # the same trick ops_boolean._refuse_complexity uses, so the total this
    # call would face is knowable from mesh.loops and face_count alone.
    total_tris = sum(len(obj.mesh.loops) - 2 * face_count(obj.mesh) for obj in objs)
    if total_tris > MAX_ANALYZE_TRIANGLES:
        raise OpError(
            f"This analysis would need {total_tris:,} triangles at once, "
            f"past the {MAX_ANALYZE_TRIANGLES:,} Clay works with. Narrow "
            "the selection with uids."
        )

    geoms: dict[int, _Geom] = {obj.uid: _geometry(obj) for obj in objs}
    object_rows = [_object_analysis(obj, geoms[obj.uid], contact_tol, symmetry_tol) for obj in objs]
    closed_by_uid = {row.uid: row.closed for row in object_rows}

    if pairs_among is not None:
        restrict = set(pairs_among)
        pair_objs = [obj for obj in objs if obj.uid in restrict]
    else:
        pair_objs = objs

    boxes = [
        (geoms[obj.uid].lo - near, geoms[obj.uid].hi + near)
        for obj in pair_objs
        if geoms[obj.uid].lo is not None
    ]
    boxed_objs = [obj for obj in pair_objs if geoms[obj.uid].lo is not None]
    uid_pairs = _broad_phase_pairs([obj.uid for obj in boxed_objs], boxes)

    by_uid = {obj.uid: obj for obj in objs}
    truncated = False
    overlap_budget = [MAX_OVERLAP_BOOLEANS]
    pair_rows: list[PairAnalysis] = []
    for uid_a, uid_b in uid_pairs:
        obj_a, obj_b = by_uid[uid_a], by_uid[uid_b]
        row, was_truncated = _pair_analysis(
            obj_a,
            obj_b,
            geoms[uid_a],
            geoms[uid_b],
            closed_by_uid[uid_a],
            closed_by_uid[uid_b],
            contact_tol,
            near,
            overlap_budget,
        )
        truncated = truncated or was_truncated
        pair_rows.append(row)

    floating = _floating(objs, object_rows, pair_rows) if pairs_among is None else None

    return Analysis(
        objects=tuple(object_rows),
        pairs=tuple(pair_rows),
        floating=floating,
        truncated=truncated,
    )
