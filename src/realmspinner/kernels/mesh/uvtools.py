"""Islands, packing and overlap over an already-assigned uv.

Everything a UV editor pane needs once a mesh has ``uv`` set -- by :mod:`.uv`'s
box projection: which faces form one connected patch, moving a patch around in
uv space, packing several into the unit square, and the overlap check a pane's
overlay reads.

**A seam is not stored.** ``Mesh.uv`` already carries it implicitly -- two
corners at one vertex with different uvs -- so an island is always *derived*
(:func:`islands`). Nothing here adds a field to :class:`~.mesh.Mesh`.

Everything is pure numpy over :mod:`.mesh` and :mod:`.adjacency`; the one lazy
exception is ``scipy.sparse``/``scipy.sparse.csgraph`` for connected
components, imported inside the function that needs it -- the package's own
``LAZY_ONLY`` rule (see ``tests/modes/clay/test_clay_imports.py``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from typing import Literal

import numpy as np

from .adjacency import adjacency
from .earclip import corner_triangles
from .elements import OpError
from .mesh import Mesh, face_count, face_normals

__all__ = [
    "MAX_OVERLAP_TRIANGLES",
    "MAX_OVERLAP_BUCKET",
    "MAX_OVERLAP_REGISTRATIONS",
    "MAX_OVERLAP_PAIRS",
    "MAX_UV_ISLANDS",
    "islands",
    "transform_islands",
    "pack_islands",
    "overlap_faces",
]

#: Grid-bucketed :func:`overlap_faces` refuses a mesh with more uv triangles
#: than this rather than silently paying for it -- see the function's own
#: docstring for what "paying for it" means in the pathological case.
MAX_OVERLAP_TRIANGLES = 50_000

#: The other half of that guard: a bucket this full means the grid did not
#: help (every triangle's uv extent covers most of the square), and pairwise
#: testing everything in it is the O(n^2) the grid exists to avoid.
MAX_OVERLAP_BUCKET = 512

#: A ceiling on the *total* number of triangle-into-cell registrations one
#: :func:`overlap_faces` call may perform, summed across every triangle --
#: :data:`MAX_OVERLAP_BUCKET` alone only bounds one over-full cell, not the
#: aggregate cost of *many* triangles that each individually stay under it.
#: The 2026-09-19 audit's clay-12 found this grid
#: has no per-triangle span cap at all, so a layout of many uniformly-narrow
#: islands -- each one's own triangles keeping the *same* wide extent along
#: one uv axis regardless of how many islands there are -- makes the grid's
#: resolution (``sqrt(triangle count)``) shrink the cell size while the
#: triangle's own span does not, so its registration count grows with the
#: island count instead of staying flat. Measured on exactly that shape (this
#: module's own worst case, full-width horizontal-strip islands): 264,064
#: registrations at 4,000 triangles (2.3s), 736,020 at 8,000 (7.8s),
#: 2,064,004 at 16,000 (28.7s) -- all far under :data:`MAX_OVERLAP_TRIANGLES`.
#: An *ordinary* packed layout of similarly-sized islands stays well clear of
#: this even at the 50,000-triangle ceiling (measured ~289,000 registrations
#: there), so this only ever refuses the pathological shape.
MAX_OVERLAP_REGISTRATIONS = 500_000

#: A ceiling on the *total* number of pairwise ``_tri_tri_overlap_2d`` SAT
#: tests one :func:`overlap_faces` call may run, summed across every bucket.
#: The 2026-09-26 audit's clay-mesh-uv-02 found the other three ceilings
#: above all bound a *different* shape of unbounded cost: MAX_OVERLAP_BUCKET
#: alone bounds one over-full cell, not the total pair count of *many* cells
#: each individually under it. Twenty-four uv-stacked clusters of a few
#: hundred triangles each -- one bucket per cluster, none of them anywhere
#: near MAX_OVERLAP_BUCKET, and nowhere near MAX_OVERLAP_REGISTRATIONS either
#: since each triangle registers into only its own cell -- measured 61.6s in
#: one call with neither of the other two ceilings ever firing. Measured on
#: this machine, the identical stacked-cluster shape (uniform per-pair cost,
#: so the total pair count is what matters, not how it is distributed across
#: buckets): 249,500 pairs in 16.35s, 499,000 pairs in 38.64s -- roughly 70us
#: a pair, crossing a second somewhere around 14,000-15,000 pairs. Set well
#: under that crossing, the same margin every sibling ceiling in this package
#: keeps.
MAX_OVERLAP_PAIRS = 8_000

#: Past this many uv islands, :func:`pack_islands` and
#: :func:`transform_islands` refuse rather than pay their own per-island
#: numpy scan (an O(total corners) mask build, once per island) -- an
#: O(islands * corners) cost that accelerates as island count grows (the
#: 2026-09-19 audit's clay-11). Measured on this module's own
#: many-small-islands fixture: :func:`pack_islands` at
#: 0.041s/0.144s/0.539s/2.074s for 500/2,000/4,000/8,000 islands. Any
#: hard-surface prop unwrapped per-part reaches this.
MAX_UV_ISLANDS = 2_000


def _require_uv(mesh: Mesh, op: str) -> np.ndarray:
    if mesh.uv is None:
        raise OpError(
            f"{op} needs texture coordinates -- unwrap this object first."
        )
    return mesh.uv


def _two_sided_corners(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """``(c, twin[c])`` for every proper two-face edge, each edge counted once.

    ``adjacency.twin`` is already -1 on a boundary, a non-manifold edge and a
    flipped pair (see :mod:`.adjacency`'s own docstring), so filtering on it
    once here is what every function below reads instead of re-deriving the
    same three exclusions.
    """
    a = adjacency(mesh)
    c = np.flatnonzero(a.twin >= 0)
    c = c[c < a.twin[c]]
    return c, a.twin[c]


# --- seams and islands ----------------------------------------------------


def _face_components(n_faces: int, fc: np.ndarray, fd: np.ndarray) -> np.ndarray:
    """Raw (arbitrarily-numbered) connected components over face edges
    ``(fc, fd)``. Every face not named in either array is its own component."""
    if n_faces == 0:
        return np.zeros(0, dtype="i8")
    if len(fc) == 0:
        return np.arange(n_faces, dtype="i8")
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components as _cc

    data = np.ones(len(fc), dtype="i1")
    graph = csr_matrix((data, (fc, fd)), shape=(n_faces, n_faces))
    _n, labels = _cc(graph, directed=False)
    return labels.astype("i8")


def _renumber_by_lowest_face(labels: np.ndarray, n_faces: int) -> np.ndarray:
    """Relabel components ``0..k-1`` in order of each one's lowest face index,
    so the same mesh always gets the same island ids however scipy happened
    to number its raw components."""
    if n_faces == 0:
        return np.zeros(0, dtype="i4")
    n_comp = int(labels.max()) + 1
    min_face = np.full(n_comp, n_faces, dtype="i8")
    np.minimum.at(min_face, labels, np.arange(n_faces, dtype="i8"))
    order = np.argsort(min_face, kind="stable")
    remap = np.empty(n_comp, dtype="i4")
    remap[order] = np.arange(n_comp, dtype="i4")
    return remap[labels]


def islands(mesh: Mesh) -> np.ndarray:
    """``(F,)`` island id: faces connected across edges whose uv agrees on
    both corners.

    This reads whatever uv the mesh already carries -- it is the "what is
    one draggable patch right now" question a UV editor's click-to-select
    asks.

    Deterministic numbering: island 0 is the component containing the
    lowest-index face, island 1 the next lowest, and so on -- so a reload of
    the same mesh selects the same island by id.
    """
    uv = _require_uv(mesh, "islands")
    n_faces = face_count(mesh)
    if n_faces == 0:
        return np.zeros(0, dtype="i4")
    a = adjacency(mesh)
    c, d = _two_sided_corners(mesh)
    if len(c) == 0:
        return _renumber_by_lowest_face(np.arange(n_faces, dtype="i8"), n_faces)
    nc, nd = a.next_corner[c], a.next_corner[d]
    u_agree = np.all(np.abs(uv[c] - uv[nd]) <= 1e-6, axis=1)
    v_agree = np.all(np.abs(uv[nc] - uv[d]) <= 1e-6, axis=1)
    connect = u_agree & v_agree
    fc = a.corner_face[c[connect]].astype("i8")
    fd = a.corner_face[d[connect]].astype("i8")
    labels = _face_components(n_faces, fc, fd)
    return _renumber_by_lowest_face(labels, n_faces)


# --- moving islands around in uv space ------------------------------------


def _face_of_corner(mesh: Mesh) -> np.ndarray:
    counts = np.diff(mesh.starts.astype("i8"))
    return np.repeat(np.arange(face_count(mesh), dtype="i8"), counts)


def transform_islands(
    mesh: Mesh,
    island_ids: Sequence[int] | np.ndarray,
    *,
    translate: tuple[float, float] = (0.0, 0.0),
    rotate_deg: float = 0.0,
    scale: float = 1.0,
    pivot: Literal["centre", "origin"] = "centre",
    ids: np.ndarray | None = None,
) -> Mesh:
    """Rotate/scale each chosen island about its own uv-bbox centre, then
    slide the whole selection by *translate*.

    Each island spins and scales around **its own** centre rather than a
    shared one -- Blender's "individual origins" pivot -- because a batch of
    islands unwrapped independently (a box unwrap's six squares, an LSCM
    cut's several patches) has no shared centre that means anything; a
    common-origin rotate would fling every island but the first off across
    the square. ``translate`` is the one part applied to the selection as a
    whole, because nudging several islands together is the ordinary case a
    UV editor's arrow keys want. ``pivot="origin"`` rotates/scales about uv
    ``(0, 0)`` instead, for a caller that wants the square's own corner.

    An id in *island_ids* that :func:`islands` does not currently report is
    silently a no-op for that id, the same tolerance ``elements.combine``'s
    subtract has for an element that is not selected.

    *ids* lets a caller that already has :func:`islands`' own per-face array
    pass it straight through instead of paying this function's own adjacency-
    plus-connected-components pass again. Omitted (the default), this computes
    its own: a one-shot caller (the toolbar's typed field, a live drag) has no
    array of its own to hand in. See :data:`MAX_UV_ISLANDS` for the ceiling.
    """
    uv = _require_uv(mesh, "transform_islands")
    if ids is None:
        ids = islands(mesh)
    wanted = np.unique(np.asarray(list(island_ids), dtype="i4"))
    if len(wanted) > MAX_UV_ISLANDS:
        raise OpError(
            f"This would transform {len(wanted)} uv islands at once, past the "
            f"{MAX_UV_ISLANDS} this op reads -- check a smaller selection."
        )
    face_mask = np.isin(ids, wanted)
    if not face_mask.any():
        return mesh

    foc = _face_of_corner(mesh)
    corner_island = ids[foc]
    corner_mask = face_mask[foc]

    new_uv = np.array(uv, dtype="f8", copy=True)
    theta = math.radians(float(rotate_deg))
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    # Row-vector convention (p @ R): a CCW rotation by theta.
    rot = np.array([[cos_t, sin_t], [-sin_t, cos_t]])

    for label in wanted.tolist():
        sel = corner_mask & (corner_island == label)
        if not sel.any():
            continue
        pts = new_uv[sel]
        centre = np.zeros(2) if pivot == "origin" else (pts.min(axis=0) + pts.max(axis=0)) / 2.0
        pts = (pts - centre) * float(scale)
        pts = pts @ rot
        new_uv[sel] = pts + centre

    new_uv[corner_mask] += np.asarray(translate, dtype="f8")
    return replace(mesh, uv=new_uv.astype("f4"))


def pack_islands(mesh: Mesh, *, margin: float = 0.005, rotate: bool = False) -> Mesh:
    """Shelf-pack every island into the unit square, preserving each
    island's scale relative to the others.

    **Shelf packing**, not a general bin packer: islands are sorted tallest
    first (the standard heuristic -- placing the tallest islands first fixes
    each shelf's height once, so a shorter island placed later never reopens
    a shelf that has already been closed off) and laid left to right until
    the next one would cross a target width, then a new shelf starts below.
    The target width is ``sqrt(total island area)``, which aims the raw
    layout at roughly square before the uniform rescale below, and is
    widened to the single widest island if that alone would exceed it.

    **The whole layout is then scaled uniformly** to fit ``[0, 1] x [0, 1]``
    -- one factor for every island, so a small island stays small relative to
    a large one exactly as it was before packing. Scaling each island to fill
    its own shelf slot independently was the obvious alternative and is the
    one :mod:`.uv`'s own ``box_unwrap`` docstring already rejects for the same
    reason: texel density would then
    depend on how the packer happened to arrange the page, not on the
    geometry.

    ``rotate`` swaps a wider-than-tall island 90 degrees before packing,
    which is the one per-island decision cheap enough to make without a real
    bin packer -- it tends to shrink the target width for a page full of thin
    islands, not a guarantee of the minimum-waste layout.

    **This only guarantees no *new* overlap between islands.** An island
    that was already self-overlapping before packing -- :func:`islands`
    groups faces by uv-agreement, not by non-overlap, and :mod:`.uv`'s own
    ``box_unwrap`` docstring says plainly that its opposite-facing squares
    share one region on purpose -- is moved and scaled as one rigid piece,
    so whatever it overlapped with itself, it still does. Re-parameterising
    an island's own internal layout is not this function's job.
    """
    uv = _require_uv(mesh, "pack_islands")
    n_faces = face_count(mesh)
    if n_faces == 0:
        return mesh
    ids = islands(mesh)
    labels = np.unique(ids).tolist()
    if len(labels) > MAX_UV_ISLANDS:
        raise OpError(
            f"This mesh has {len(labels)} uv islands, past the {MAX_UV_ISLANDS} "
            f"pack_islands reads -- check a smaller selection instead of the "
            f"whole mesh."
        )
    foc = _face_of_corner(mesh)
    corner_island = ids[foc]
    new_uv = np.array(uv, dtype="f8", copy=True)

    items = []
    for label in labels:
        mask = corner_island == label
        pts = new_uv[mask]
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        w, h = float(hi[0] - lo[0]), float(hi[1] - lo[1])
        rotated = bool(rotate and w > h)
        if rotated:
            w, h = h, w
        items.append((label, mask, lo, w, h, rotated))

    # Tallest first; ties by label, so two same-height islands always land
    # in the same relative order regardless of dict/set iteration.
    items.sort(key=lambda it: (-it[4], it[0]))

    total_area = sum(w * h for *_rest, w, h, _r in items)
    widest = max((w for *_rest, w, _h, _r in items), default=0.0)
    target_width = max(math.sqrt(total_area), widest, 1e-9)

    x = y = shelf_h = 0.0
    max_x = 0.0
    placements = []
    for _label, mask, lo, w, h, rotated in items:
        if x > 0.0 and x + w > target_width:
            y += shelf_h + margin
            x = 0.0
            shelf_h = 0.0
        placements.append((mask, lo, rotated, x, y))
        x += w + margin
        max_x = max(max_x, x - margin)
        shelf_h = max(shelf_h, h)
    total_h = y + shelf_h
    scale = 1.0 / max(max_x, total_h, 1e-9)

    for mask, lo, rotated, ox, oy in placements:
        pts = new_uv[mask] - lo
        if rotated:
            # 90 degrees CCW about the island's own local origin, then
            # re-zeroed: the rotation can push the bbox negative, and the
            # shelf offset below assumes a bbox that starts at (0, 0).
            pts = np.stack([-pts[:, 1], pts[:, 0]], axis=1)
            pts -= pts.min(axis=0)
        new_uv[mask] = (pts + (ox, oy)) * scale

    return replace(mesh, uv=new_uv.astype("f4"))


# --- overlap -----------------------------------------------------------


def _tri_tri_overlap_2d(t1: np.ndarray, t2: np.ndarray) -> bool:
    """Separating-axis test for two 2D triangles: true unless some edge
    normal of either one separates their projections."""
    for tri in (t1, t2):
        for i in range(3):
            edge = tri[(i + 1) % 3] - tri[i]
            axis = np.array([-edge[1], edge[0]])
            n = float(np.linalg.norm(axis))
            if n < 1e-12:
                continue
            axis = axis / n
            p1, p2 = t1 @ axis, t2 @ axis
            # clay-07: touching, or a gap within the tolerance, is separated.
            if p1.max() <= p2.min() + 1e-9 or p2.max() <= p1.min() + 1e-9:
                return False
    return True


def overlap_faces(mesh: Mesh) -> np.ndarray:
    """``(F,)`` bool: does this face's uv overlap another (different) face's.

    Triangulated with :func:`~.earclip.corner_triangles`, not
    :func:`~.mesh.triangulate` -- the latter resolves to *vertex* indices,
    which throws away exactly the per-corner distinction a seam needs.
    Bucketed into a grid over the uv bounding box, sized to roughly one
    triangle per cell, so a pair is only tested when their cells overlap
    rather than every pair in the mesh.

    Refuses past :data:`MAX_OVERLAP_TRIANGLES`, and also if any one grid
    bucket collects more than :data:`MAX_OVERLAP_BUCKET` triangles: the grid
    keeps the *common* case cheap, but a mesh whose uv triangles nearly all
    cover the whole square degenerates to the full O(T^2) the grid exists to
    avoid regardless of the total count, and that is not a check to run
    silently on the frame thread. A third ceiling, :data:`MAX_OVERLAP_REGISTRATIONS`,
    catches the case neither of the first two does: many uniformly-narrow
    islands whose triangles each keep the same wide extent along one uv axis
    however many islands there are, so no single bucket ever fills past
    :data:`MAX_OVERLAP_BUCKET` but the aggregate registration cost still
    grows with island count (see that constant's own docstring). A fourth,
    :data:`MAX_OVERLAP_PAIRS`, catches the case none of the first three does:
    many separate buckets, each on its own comfortably under
    :data:`MAX_OVERLAP_BUCKET`, whose pairwise SAT-test counts still sum to an
    unbounded total (see that constant's own docstring).
    """
    uv = _require_uv(mesh, "overlap_faces")
    n_faces = face_count(mesh)
    out = np.zeros(n_faces, dtype=bool)
    if n_faces == 0:
        return out
    corners, tri_face = corner_triangles(
        mesh.positions, mesh.loops, mesh.starts, face_normals(mesh)
    )
    if len(corners) == 0:
        return out
    if len(corners) > MAX_OVERLAP_TRIANGLES:
        raise OpError(
            f"This mesh has {len(corners)} uv triangles, past the "
            f"{MAX_OVERLAP_TRIANGLES} an overlap check reads -- check a "
            f"selection instead of the whole mesh."
        )

    tris = uv[corners].astype("f8")  # (T, 3, 2)
    lo, hi = tris.min(axis=1), tris.max(axis=1)
    origin = lo.min(axis=0)
    span = np.maximum(hi.max(axis=0) - origin, 1e-9)
    res = max(1, int(math.sqrt(len(corners))))
    cell = np.maximum(span / res, 1e-9)
    lo_cell = np.floor((lo - origin) / cell).astype("i8")
    # clay-08: a triangle whose far edge lies exactly on a cell boundary does
    # not register in the next cell (touching is not overlap, see
    # _tri_tri_overlap_2d), or every cell-aligned layout doubled its pairs.
    hi_cell = np.maximum(
        np.ceil((hi - origin) / cell - 1e-9).astype("i8") - 1, lo_cell
    )

    # See MAX_OVERLAP_REGISTRATIONS: estimate what the bucket-building loop
    # below will cost -- one cell-span product per triangle, summed over the
    # whole mesh -- vectorised, before a single Python iteration runs. The
    # "refuse before the allocation" shape.
    span_cells = (hi_cell - lo_cell + 1).astype(np.int64)
    total_registrations = int((span_cells[:, 0] * span_cells[:, 1]).sum())
    if total_registrations > MAX_OVERLAP_REGISTRATIONS:
        raise OpError(
            f"This uv layout would need {total_registrations} grid "
            f"registrations, past the {MAX_OVERLAP_REGISTRATIONS} an overlap "
            f"check reads -- its islands keep a wide extent along one axis "
            f"regardless of how many there are. Check a smaller selection."
        )

    buckets: dict[tuple[int, int], list[int]] = {}
    for t in range(len(corners)):
        for cx in range(int(lo_cell[t, 0]), int(hi_cell[t, 0]) + 1):
            for cy in range(int(lo_cell[t, 1]), int(hi_cell[t, 1]) + 1):
                buckets.setdefault((cx, cy), []).append(t)

    # The 2026-09-26 audit's clay-mesh-uv-02: MAX_OVERLAP_BUCKET bounds one
    # over-full cell and MAX_OVERLAP_REGISTRATIONS bounds the total triangle-
    # into-cell registrations, but neither bounds the total number of
    # pairwise SAT tests the loop below performs -- many separate buckets,
    # each on its own comfortably under MAX_OVERLAP_BUCKET, still sum to an
    # unbounded total. Counted here, vectorised, from the bucket sizes the
    # loop below is about to walk -- the same "count before the walk" shape
    # ops_topo.collapse counts its own pairs with -- and refused before a
    # single SAT test runs. This over-counts slightly (the loop below also
    # skips same-face pairs and dedupes a pair tested from two shared cells),
    # so it only ever refuses early, never lets more through than the real
    # loop would do.
    # clay-08: gather exactly the pairs worth a SAT test -- cross-face,
    # bounding boxes that genuinely overlap (touching boxes cannot hold an
    # interior overlap), each pair once however many cells it shares -- and
    # count *those* against the ceiling, so an ordinary dense layout is not
    # refused by pairs the loop never needed to run. Vectorised per bucket;
    # a bucket past MAX_OVERLAP_BUCKET is refused before its matrix is built.
    keys: list[np.ndarray] = []
    total_pairs = 0
    n_tri = len(corners)
    for members in buckets.values():
        if len(members) < 2:
            continue
        if len(members) > MAX_OVERLAP_BUCKET:
            raise OpError(
                f"{len(members)} uv triangles share one grid cell, past the "
                f"{MAX_OVERLAP_BUCKET} an overlap check reads -- their uv "
                f"extents are too large for a grid to narrow the work down. "
                f"Check a smaller selection."
            )
        idx = np.asarray(members, dtype=np.int64)
        ii, jj = np.triu_indices(len(idx), k=1)
        ta, tb = idx[ii], idx[jj]
        keep = tri_face[ta] != tri_face[tb]
        keep &= (lo[ta] < hi[tb] - 1e-9).all(axis=1) & (lo[tb] < hi[ta] - 1e-9).all(axis=1)
        ta, tb = ta[keep], tb[keep]
        if len(ta):
            keys.append(np.minimum(ta, tb) * n_tri + np.maximum(ta, tb))
            total_pairs += len(ta)
            if total_pairs > 4 * MAX_OVERLAP_PAIRS:
                break  # far past the ceiling: refuse without finishing the count
    pairs = np.unique(np.concatenate(keys)) if keys else np.zeros(0, dtype=np.int64)
    if len(pairs) > MAX_OVERLAP_PAIRS:
        raise OpError(
            f"This uv layout would run {len(pairs):,} pairwise overlap "
            f"tests, past the {MAX_OVERLAP_PAIRS:,} an overlap check reads -- "
            f"too many triangles share too few grid cells. Check a smaller "
            f"selection."
        )
    for key in pairs.tolist():
        ta, tb = divmod(key, n_tri)
        if _tri_tri_overlap_2d(tris[ta], tris[tb]):
            out[tri_face[ta]] = True
            out[tri_face[tb]] = True
    return out
