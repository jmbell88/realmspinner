"""Islands, packing, density and distortion over an already-assigned uv.

Clay tranche 6 ("UV and materials", ``dev/CLAY-PLAN.md``). Everything a UV
editor pane needs once a mesh has ``uv`` set -- by :mod:`.uv`'s box/planar
projection, or by :mod:`.uvunwrap`'s LSCM solve -- lives here: which faces
form one connected patch, moving a patch around in uv space, packing several
into the unit square, and the three measurements (texel density, overlap,
stretch) a validator or a pane's overlay reads. :mod:`.uvunwrap` is the other
half, split out because the LSCM solve is a different kind of code (a sparse
linear system) from the array plumbing here, and because it is the one
function in the tranche heavy enough to earn its own ceiling.

**A seam is not stored.** ``Mesh.uv`` already carries it implicitly -- two
corners at one vertex with different uvs -- so a seam set here is always
*derived* (:func:`seams_from_uv`) or *supplied by the caller as a plan*
(:func:`islands_by_seams`, read by :func:`~.uvunwrap.unwrap_lscm` before any
uv exists to derive one from). Nothing here adds a field to :class:`~.mesh.Mesh`.

Everything is pure numpy over :mod:`.mesh` and :mod:`.adjacency`; the one
lazy exception is ``scipy.sparse``/``scipy.sparse.csgraph`` for connected
components, imported inside the two functions that need it -- the package's
own ``LAZY_ONLY`` rule (see ``tests/modes/clay/test_clay_imports.py``), which
already covers scipy for :mod:`.analyze`'s own ``cKDTree`` use.
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
    "edge_key",
    "edge_keys",
    "seams_from_uv",
    "islands",
    "islands_by_seams",
    "transform_islands",
    "pack_islands",
    "texel_density",
    "normalize_density",
    "overlap_faces",
    "stretch",
    "flipped_uv_faces",
]

#: Grid-bucketed :func:`overlap_faces` refuses a mesh with more uv triangles
#: than this rather than silently paying for it -- see the function's own
#: docstring for what "paying for it" means in the pathological case.
MAX_OVERLAP_TRIANGLES = 50_000

#: The other half of that guard: a bucket this full means the grid did not
#: help (every triangle's uv extent covers most of the square), and pairwise
#: testing everything in it is the O(n^2) the grid exists to avoid.
MAX_OVERLAP_BUCKET = 512


# --- seam/edge vocabulary -----------------------------------------------


def edge_key(a: int, b: int) -> tuple[int, int]:
    """Canonical ``(low, high)`` form of one vertex pair."""
    a, b = int(a), int(b)
    return (a, b) if a <= b else (b, a)


def edge_keys(pairs: np.ndarray | Sequence[Sequence[int]] | None) -> np.ndarray:
    """Vectorised :func:`edge_key`: ``(n, 2)`` -> sorted-unique, low first.

    The same canonical form :mod:`.elements`' ``ElementSel.edges`` uses, so a
    seam set and an edge selection are interchangeable without a conversion.
    """
    if pairs is None:
        return np.zeros((0, 2), dtype="i4")
    rows = np.asarray(pairs, dtype="i4").reshape(-1, 2)
    if len(rows) == 0:
        return np.zeros((0, 2), dtype="i4")
    lo = np.minimum(rows[:, 0], rows[:, 1])
    hi = np.maximum(rows[:, 0], rows[:, 1])
    return np.unique(np.stack([lo, hi], axis=1), axis=0).astype("i4")


def _isin_pairs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Boolean mask: which rows of ``a`` (low, high) also appear in ``b``.

    :func:`.elements._rows_minus`'s trick: pack each canonical pair into one
    int64 key and let ``np.isin`` do the set membership, rather than a dense
    ``(len(a), len(b))`` broadcast -- the same reasoning, for the same
    reason: a seam set can be most of a large import's edges.
    """
    if len(a) == 0 or len(b) == 0:
        return np.zeros(len(a), dtype=bool)
    scale = int(max(a.max(), b.max())) + 1
    a_keys = a[:, 0].astype(np.int64) * scale + a[:, 1].astype(np.int64)
    b_keys = b[:, 0].astype(np.int64) * scale + b[:, 1].astype(np.int64)
    return np.isin(a_keys, b_keys)


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


def seams_from_uv(mesh: Mesh, *, atol: float = 1e-6) -> np.ndarray:
    """Interior edges whose two sides disagree in uv, as sorted ``(K, 2)``
    vertex pairs.

    Only a two-sided edge can "disagree" at all -- a boundary, a non-manifold
    edge or a flipped pair has only one usable side and is not a seam by this
    definition, it is already a cut. :func:`islands` and
    :func:`islands_by_seams` both stop at those on their own, without needing
    them listed here too.

    The comparison reads each side's corner **at the same vertex**, not the
    two corners of one directed edge: for edge ``(u, v)``, corner ``c`` on one
    face runs ``u -> v`` and its twin ``d`` runs the *other* direction,
    ``v -> u`` (see :mod:`.adjacency`'s half-edge docstring), so the corner at
    vertex ``u`` on face ``d``'s side is ``next_corner[d]``, not ``d`` itself.
    """
    uv = _require_uv(mesh, "seams_from_uv")
    a = adjacency(mesh)
    c, d = _two_sided_corners(mesh)
    if len(c) == 0:
        return np.zeros((0, 2), dtype="i4")
    nc, nd = a.next_corner[c], a.next_corner[d]
    u_agree = np.all(np.abs(uv[c] - uv[nd]) <= atol, axis=1)
    v_agree = np.all(np.abs(uv[nc] - uv[d]) <= atol, axis=1)
    disagree = ~(u_agree & v_agree)
    if not disagree.any():
        return np.zeros((0, 2), dtype="i4")
    return edge_keys(a.edge_verts[a.corner_edge[c[disagree]]])


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
    asks. :func:`islands_by_seams` is the other question, "what *would* the
    patches be if I cut along these seams", asked before any uv exists.

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


def islands_by_seams(mesh: Mesh, seams: np.ndarray | None) -> np.ndarray:
    """``(F,)`` island id: face components that cross no *seams* edge and no
    boundary -- independent of whatever uv the mesh currently has, or lacks.

    This is what :func:`~.uvunwrap.unwrap_lscm` cuts along: a seam plan the
    caller supplies (typically :func:`seams_from_uv` read back from a
    previous unwrap, or a fresh selection converted to edges), evaluated
    against topology alone.
    """
    n_faces = face_count(mesh)
    if n_faces == 0:
        return np.zeros(0, dtype="i4")
    a = adjacency(mesh)
    c, d = _two_sided_corners(mesh)
    if len(c) == 0:
        return _renumber_by_lowest_face(np.arange(n_faces, dtype="i8"), n_faces)
    seam_set = edge_keys(seams)
    ev = a.edge_verts[a.corner_edge[c]]
    is_seam = _isin_pairs(ev, seam_set) if len(seam_set) else np.zeros(len(c), dtype=bool)
    keep = ~is_seam
    fc = a.corner_face[c[keep]].astype("i8")
    fd = a.corner_face[d[keep]].astype("i8")
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
    """
    uv = _require_uv(mesh, "transform_islands")
    ids = islands(mesh)
    wanted = np.unique(np.asarray(list(island_ids), dtype="i4"))
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
    a large one exactly as :func:`texel_density` would read it before
    packing. Scaling each island to fill its own shelf slot independently was
    the obvious alternative and is the one :mod:`.uv`'s own ``box_unwrap``
    docstring already rejects for the same reason: texel density would then
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
    foc = _face_of_corner(mesh)
    corner_island = ids[foc]
    new_uv = np.array(uv, dtype="f8", copy=True)

    items = []
    for label in np.unique(ids).tolist():
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


# --- density and distortion ------------------------------------------------


def _signed_polygon_areas(mesh: Mesh, values: np.ndarray) -> np.ndarray:
    """Per-face signed shoelace area over an ``(L, 2)`` array shaped like
    ``uv`` -- positive for the winding every unwrap in this package produces,
    negative for a mirrored (flipped) one. :func:`~.mesh._next_corner`'s own
    wrap-per-face trick, inlined: nothing here needs a :class:`Mesh` method,
    only ``starts``.
    """
    n_faces = face_count(mesh)
    if n_faces == 0 or len(mesh.loops) == 0:
        return np.zeros(n_faces, dtype="f8")
    starts = mesh.starts.astype("i8")
    nxt = np.arange(1, len(mesh.loops) + 1, dtype="i8")
    nxt[starts[1:] - 1] = starts[:-1]
    x, y = values[:, 0].astype("f8"), values[:, 1].astype("f8")
    cross = x * y[nxt] - x[nxt] * y
    face_of = _face_of_corner(mesh)
    return 0.5 * np.bincount(face_of, weights=cross, minlength=n_faces)


def _face_world_areas(mesh: Mesh) -> np.ndarray:
    """Per-face 3D area -- half the Newell normal's own magnitude, which
    :func:`~.mesh.face_normals` already leaves unnormalised for exactly this."""
    return 0.5 * np.linalg.norm(face_normals(mesh), axis=1)


def texel_density(
    mesh: Mesh, face_ids: np.ndarray | Sequence[int] | None = None, *, texture_px: int = 1024
) -> float:
    """Pixels per metre a *texture_px*-square texture achieves over
    *face_ids* (every face, if omitted).

    ``sqrt(uv_area * texture_px**2 / world_area)``: ``uv_area`` is in the
    ``0..1`` uv unit, ``texture_px`` squared turns that into actual pixels,
    ``world_area`` is in square metres, and the square root turns the area
    ratio back into a linear px/m rate -- the number a "2K on a 3 m wall"
    conversation is actually about. A 1 m square mapped onto the whole unit
    square at 1024 px is exactly 1024 px/m by this formula, which is the
    sanity check the test module runs against it.
    """
    _require_uv(mesh, "texel_density")
    n_faces = face_count(mesh)
    faces = np.arange(n_faces, dtype="i8") if face_ids is None else np.asarray(face_ids, dtype="i8")
    if len(faces) == 0:
        return 0.0
    uv_area = float(np.abs(_signed_polygon_areas(mesh, mesh.uv))[faces].sum())
    world_area = float(_face_world_areas(mesh)[faces].sum())
    if world_area <= 0.0:
        return 0.0
    return float(math.sqrt(uv_area * float(texture_px) ** 2 / world_area))


def normalize_density(mesh: Mesh, target: float, *, texture_px: int = 1024) -> Mesh:
    """Scale every island about its own centre so each one measures *target*
    px/m under :func:`texel_density`.

    One division per island, no iteration: :func:`texel_density` takes a
    square root of area, so a uniform uv scale of *k* multiplies the reading
    by exactly *k*, and the per-island factor is ``target / current``.
    """
    _require_uv(mesh, "normalize_density")
    ids = islands(mesh)
    result = mesh
    for label in np.unique(ids).tolist():
        faces = np.flatnonzero(ids == label)
        current = texel_density(result, faces, texture_px=texture_px)
        if current <= 0.0:
            continue
        result = transform_islands(result, [label], scale=float(target) / current)
    return result


def stretch(mesh: Mesh) -> np.ndarray:
    """``(F,)`` float: how far each face's *share* of the total uv area is
    from its share of the total 3D area, as ``log2`` of the ratio.

    Zero is a perfect (isometric up to one global scale) map: a mapping
    where every face uses the same fraction of the texture that it uses of
    the mesh's surface has ``uv_share == world_share`` for every face
    regardless of what that global scale is, which is what makes this a
    *share* comparison rather than a raw area comparison -- an isometric map
    at any zoom reads as flat zero. Positive means a face is stretched larger
    in uv than its geometry warrants (soft, blurry when textured); negative
    means it is squeezed smaller (aliased, noisy when textured).
    """
    _require_uv(mesh, "stretch")
    n_faces = face_count(mesh)
    if n_faces == 0:
        return np.zeros(0, dtype="f8")
    uv_area = np.abs(_signed_polygon_areas(mesh, mesh.uv))
    world_area = _face_world_areas(mesh)
    uv_total, world_total = float(uv_area.sum()), float(world_area.sum())
    if uv_total <= 0.0 or world_total <= 0.0:
        return np.zeros(n_faces, dtype="f8")
    uv_share = uv_area / uv_total
    world_share = world_area / world_total
    out = np.zeros(n_faces, dtype="f8")
    valid = (uv_share > 0.0) & (world_share > 0.0)
    out[valid] = np.log2(uv_share[valid] / world_share[valid])
    return out


def flipped_uv_faces(mesh: Mesh, *, eps: float = 1e-9) -> np.ndarray:
    """``(F,)`` bool: faces whose uv winding runs the opposite way from every
    unwrap in this package -- a mirrored texture, the telltale of a
    hand-edited or badly-imported uv.

    A face is flipped when its own uv polygon's signed area
    (:func:`_signed_polygon_areas`) is negative; a degenerate, near-zero-area
    face is reported clean rather than flipped, since there is no winding
    left to be wrong about.
    """
    _require_uv(mesh, "flipped_uv_faces")
    n_faces = face_count(mesh)
    if n_faces == 0:
        return np.zeros(0, dtype=bool)
    return _signed_polygon_areas(mesh, mesh.uv) < -abs(eps)


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
            if p1.max() < p2.min() - 1e-9 or p2.max() < p1.min() - 1e-9:
                return False
    return True


def overlap_faces(mesh: Mesh) -> np.ndarray:
    """``(F,)`` bool: does this face's uv overlap another (different) face's.

    Triangulated with :func:`~.earclip.corner_triangles`, not
    :func:`~.mesh.triangulate` -- the latter resolves to *vertex* indices,
    which throws away exactly the per-corner distinction a seam needs.
    Bucketed into a grid over the uv bounding box, sized to roughly one
    triangle per cell, so a pair is only tested when their cells overlap
    rather than every pair in the mesh -- the same "grid, not a full
    pairwise sweep" shape :func:`~.analyze._components` uses for a weld.

    Refuses past :data:`MAX_OVERLAP_TRIANGLES`, and also if any one grid
    bucket collects more than :data:`MAX_OVERLAP_BUCKET` triangles: the grid
    keeps the *common* case cheap, but a mesh whose uv triangles nearly all
    cover the whole square degenerates to the full O(T^2) the grid exists to
    avoid regardless of the total count, and that is not a check to run
    silently on the frame thread.
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
    hi_cell = np.floor((hi - origin) / cell).astype("i8")

    buckets: dict[tuple[int, int], list[int]] = {}
    for t in range(len(corners)):
        for cx in range(int(lo_cell[t, 0]), int(hi_cell[t, 0]) + 1):
            for cy in range(int(lo_cell[t, 1]), int(hi_cell[t, 1]) + 1):
                buckets.setdefault((cx, cy), []).append(t)

    tested: set[tuple[int, int]] = set()
    for members in buckets.values():
        if len(members) > MAX_OVERLAP_BUCKET:
            raise OpError(
                f"{len(members)} uv triangles share one grid cell, past the "
                f"{MAX_OVERLAP_BUCKET} an overlap check reads -- their uv "
                f"extents are too large for a grid to narrow the work down. "
                f"Check a smaller selection."
            )
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                ta, tb = members[i], members[j]
                if tri_face[ta] == tri_face[tb]:
                    continue
                key = (ta, tb) if ta < tb else (tb, ta)
                if key in tested:
                    continue
                tested.add(key)
                if _tri_tri_overlap_2d(tris[ta], tris[tb]):
                    out[tri_face[ta]] = True
                    out[tri_face[tb]] = True
    return out
