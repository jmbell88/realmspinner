"""LSCM: the one angle-preserving unwrap in this package, from a seam plan.

``uv.py``'s own docstring turned down a conformal solver on purpose, for
blockout geometry a box or planar projection answers instantly and without a
failure mode. Clay tranche 6 is the moment that stops being true everywhere:
a retopologised or imported organic surface wants a real unwrap, and the user
now has a way to say where to cut it (a seam, :func:`.uvtools.seams_from_uv`
reads back after a hand-marked one is applied to ``Mesh.uv`` itself, or a
selection converted to edges). This module is that solver, kept apart from
:mod:`.uvtools` because it is a different kind of code -- a sparse linear
least-squares system per island, with its own ceiling -- not more array
plumbing over an already-assigned uv.

**Least Squares Conformal Maps** (Lévy, Petitjean, Ray, Maillot 2002).  Per
triangle, in a local 2D isometric frame ``(x, y)`` built from the triangle's
own 3D edges, the Cauchy-Riemann equations ``dv/dx = du/dy`` and
``dv/dy = -du/dx`` -- read backwards from a complex-analysis text, but the
condition for an angle-preserving map from ``(x, y)`` to ``(u, v)`` -- give
two linear residuals in the unknown per-vertex ``(u, v)``. Two vertices per
island are pinned to fixed ``(u, v)`` (removed from the unknowns, folded into
the right-hand side) so the least-squares system has a unique minimum rather
than a whole family of solutions differing by a rigid motion, and the rest is
solved with ``scipy.sparse.linalg.lsqr`` -- a dependency this package already
has for :mod:`.analyze`'s ``connected_components``, imported inside the one
function that needs it for the same ``LAZY_ONLY`` reason.

Everything here is pure numpy/scipy over :mod:`.mesh` and :mod:`.adjacency`.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .adjacency import adjacency
from .earclip import corner_triangles
from .elements import OpError
from .mesh import Mesh, face_count, face_normals
from .uvtools import _isin_pairs, edge_keys, islands_by_seams, pack_islands

__all__ = ["MAX_LSCM_VERTICES", "unwrap_lscm"]

#: A per-island ceiling, not a whole-mesh one -- each island is its own
#: independent sparse solve. ``lsqr`` on a system this size is a fraction of
#: a second; past it, the honest answer is Blender's Smart UV Project
#: (tranche 4's Blender-backed heavy ops), not a solver that goes quiet for
#: a minute on the frame thread.
MAX_LSCM_VERTICES = 20_000


def _corner_mask(mesh: Mesh, faces: np.ndarray) -> np.ndarray:
    """*mesh*'s per-corner boolean mask selecting exactly the corners of
    *faces* -- vectorised over :func:`~.adjacency.adjacency`'s own
    ``corner_face`` table rather than a Python loop that set one slice per
    face (``for f in faces.tolist(): mask[starts[f]:starts[f+1]] = True``).
    That loop is why :func:`unwrap_lscm`'s own :data:`MAX_LSCM_VERTICES`
    refusal used to be unaffordable to check early: the 2026-09-19 audit's
    clay-13 found the refusal fired only after the whole mesh had already
    been triangulated *and* this per-face scan had already run once for the
    refusing island. Making the scan itself O(corners) numpy rather than
    O(faces) Python is what lets :func:`unwrap_lscm` call this to count an
    island's vertices *before* triangulating anything.
    """
    corner_face = adjacency(mesh).corner_face.astype("i8")
    n_faces = len(mesh.starts) - 1
    selected = np.zeros(n_faces, dtype=bool)
    selected[faces] = True
    return selected[corner_face]


def _island_is_closed(mesh: Mesh, faces: np.ndarray) -> bool:
    """True when no edge touched by *faces* has fewer than two of its uses
    inside the island -- a closed surface (a full sphere, an unseamed cube)
    with nothing to cut it open along.

    An edge counted twice *within the island* is a genuine interior edge:
    both its faces are inside. One counted once is a cut -- either the
    mesh's own boundary, or a seam separating this island from a neighbour.
    """
    a = adjacency(mesh)
    corner_idx = np.flatnonzero(_corner_mask(mesh, faces))
    if len(corner_idx) == 0:
        return False
    edge_ids = a.corner_edge[corner_idx]
    counts = np.bincount(edge_ids, minlength=a.n_edges)
    return bool((counts[edge_ids] == 2).all())


def _boundary_vertices(mesh: Mesh, faces: np.ndarray) -> np.ndarray:
    """Global vertex ids on the island's own cut boundary (mesh boundary or
    seam), in no particular order."""
    a = adjacency(mesh)
    corner_idx = np.flatnonzero(_corner_mask(mesh, faces))
    edge_ids = a.corner_edge[corner_idx]
    counts = np.bincount(edge_ids, minlength=a.n_edges)
    on_boundary = counts[edge_ids] == 1
    return np.unique(mesh.loops[corner_idx[on_boundary]]).astype("i8")


def _corner_groups(
    mesh: Mesh, seams: np.ndarray | None, corner_idx: np.ndarray
) -> tuple[np.ndarray, int]:
    """One LSCM unknown per (vertex, side-of-a-seam) -- a union-find over the
    island's own corners, merging two corners at the same vertex exactly
    when the edge joining their faces there is *not* a seam.

    Indexing the solve by raw vertex instead of this was the bug found
    proving this module out: a cylinder with one seam came back with ~30%
    per-face area distortion concentrated at the two faces touching the cut,
    because a single shared ``(u, v)`` unknown forced the two ends of the
    unrolled strip to agree on where the cut vertex sits, when they should
    be free to land anywhere. ``Mesh.uv`` is already per corner for exactly
    this reason (see :mod:`.mesh`'s own docstring on seams); this is that
    same reasoning applied to the solve's own unknowns. Two corners of the
    same face at the same vertex are never split -- only the edge *between
    two faces* is what a seam is drawn on.

    **Island membership is necessary but not sufficient** for "this edge
    should merge": :func:`~.uvtools.islands_by_seams` keeps two faces in one
    island whenever *any* unseamed path connects them, so a seam edge can
    sit directly between two faces that still end up in the same island by
    a longer route (a ring cut once, as in the cylinder above -- the two
    faces either side of the single seam edge are still connected the long
    way around). That edge must still not merge, which is why this checks
    the seam set per edge rather than trusting island membership alone.
    """
    parent = {int(c): int(c) for c in corner_idx.tolist()}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    a = adjacency(mesh)
    in_island = np.zeros(len(mesh.loops), dtype=bool)
    in_island[corner_idx] = True
    c = np.flatnonzero(a.twin >= 0)
    c = c[in_island[c] & in_island[a.twin[c]]]
    if len(c):
        d = a.twin[c]
        seam_set = edge_keys(seams)
        ev = a.edge_verts[a.corner_edge[c]]
        is_seam = _isin_pairs(ev, seam_set) if len(seam_set) else np.zeros(len(c), dtype=bool)
        nc, nd = a.next_corner[c], a.next_corner[d]
        for cc, dd, ncc, ndd, seam in zip(
            c.tolist(), d.tolist(), nc.tolist(), nd.tolist(), is_seam.tolist(), strict=True
        ):
            if seam:
                continue
            # Same vertex, two sides of one edge: c's "u" end matches d's
            # face's corner at that same vertex (next_corner[d]), and c's
            # "v" end (next_corner[c]) matches d itself -- the half-edge
            # reasoning :func:`.uvtools.seams_from_uv` already documents.
            for x, y in ((cc, ndd), (ncc, dd)):
                rx, ry = find(x), find(y)
                if rx != ry:
                    parent[ry] = rx

    group_of: dict[int, int] = {}
    ids = np.empty(len(corner_idx), dtype="i8")
    for i, c0 in enumerate(corner_idx.tolist()):
        root = find(c0)
        if root not in group_of:
            group_of[root] = len(group_of)
        ids[i] = group_of[root]
    return ids, len(group_of)


def _group_for_vertex(
    corner_idx: np.ndarray, group_of_corner: np.ndarray, loops: np.ndarray, vertex: int
) -> int:
    """The LSCM unknown for *vertex*'s first corner (in ``corner_idx`` order).

    A vertex split by a seam has more than one unknown; pinning "a vertex"
    pins whichever side its first corner belongs to. Auto-selected pins are
    boundary vertices chosen only for being far apart (:func:`_auto_pins`),
    never for which side of a cut they sit on, so this tie-break is not the
    deciding factor there in practice -- it only matters for an explicit
    caller-supplied pin that happens to name a seam vertex, where it is a
    documented, deterministic choice rather than an arbitrary one.
    """
    hit = np.flatnonzero(loops[corner_idx] == vertex)
    return int(group_of_corner[hit[0]])


def _auto_pins(mesh: Mesh, faces: np.ndarray, verts: np.ndarray) -> tuple[int, int]:
    """Two far-apart boundary vertices, by the double-sweep approximate
    diameter: farthest from an arbitrary start, then farthest from that --
    O(boundary length) rather than every pair of it, and the same "far
    apart" the plan calls for without needing the true diameter.
    """
    boundary = _boundary_vertices(mesh, faces)
    if len(boundary) < 2:
        boundary = verts  # Defensive: the closed-island refusal should have
        # already fired before this is ever reached with no boundary at all.
    pos = mesh.positions.astype("f8")
    v0 = int(boundary[0])
    d0 = np.linalg.norm(pos[boundary] - pos[v0], axis=1)
    v1 = int(boundary[int(np.argmax(d0))])
    d1 = np.linalg.norm(pos[boundary] - pos[v1], axis=1)
    v2 = int(boundary[int(np.argmax(d1))])
    if v2 == v1 and len(boundary) > 1:
        v2 = int(boundary[0]) if int(boundary[0]) != v1 else int(boundary[-1])
    return v1, v2


def _select_pins(
    mesh: Mesh, faces: np.ndarray, verts: np.ndarray, pins: list[tuple[int, int]] | None
) -> tuple[int, int]:
    vset = set(verts.tolist())
    if pins:
        for a, b in pins:
            a, b = int(a), int(b)
            if a != b and a in vset and b in vset:
                return a, b
    return _auto_pins(mesh, faces, verts)


def _local_frame(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray
) -> tuple[float, float, float] | None:
    """The triangle's own isometric 2D coordinates ``(x2, x3, y3)`` with
    ``x1 = y1 = y2 = 0`` -- ``None`` for a degenerate (collinear or
    zero-length-edge) triangle, which contributes no equations.

    ``ey`` is built as ``normal x ex`` from the triangle's *own* winding
    (``normal = e1 x (p3 - p1)``, not the face's stored Newell normal), which
    is what keeps every triangle's local frame the same handedness as its
    3D winding and is why the solve comes out with no flipped faces on a
    consistently-wound mesh: get this cross product backwards and every
    triangle's conformal equations would encode the mirror image instead.
    """
    e1 = p2 - p1
    e1len = float(np.linalg.norm(e1))
    if e1len < 1e-12:
        return None
    ex = e1 / e1len
    raw_n = np.cross(e1, p3 - p1)
    nlen = float(np.linalg.norm(raw_n))
    if nlen < 1e-12:
        return None
    ey = np.cross(raw_n / nlen, ex)
    x3 = float(np.dot(p3 - p1, ex))
    y3 = float(np.dot(p3 - p1, ey))
    return e1len, x3, y3


def _solve_island(
    mesh: Mesh,
    faces: np.ndarray,
    tri_corners: np.ndarray,
    seams: np.ndarray | None,
    out_uv: np.ndarray,
    pins: list[tuple[int, int]] | None,
) -> None:
    """Flatten one island in place into *out_uv*."""
    corner_idx = np.flatnonzero(_corner_mask(mesh, faces))
    verts = np.unique(mesh.loops[corner_idx]).astype("i8")
    if len(verts) > MAX_LSCM_VERTICES:
        raise OpError(
            f"This island has {len(verts)} vertices, past the "
            f"{MAX_LSCM_VERTICES} an unwrap by seams reads -- mark more seams "
            f"to split it, or use Smart Unwrap on a mesh this dense."
        )
    if _island_is_closed(mesh, faces):
        raise OpError(
            "A closed surface cannot be flattened with no seam -- mark a "
            "seam first."
        )
    if len(verts) < 3 or len(tri_corners) == 0:
        return  # A sliver or an unfaced vertex: nothing to solve, nothing to set.

    group_of_corner, n_groups = _corner_groups(mesh, seams, corner_idx)
    corner_to_group = dict(
        zip(corner_idx.tolist(), group_of_corner.tolist(), strict=True)
    )

    pin_a, pin_b = _select_pins(mesh, faces, verts, pins)
    group_a = _group_for_vertex(corner_idx, group_of_corner, mesh.loops, pin_a)
    group_b = _group_for_vertex(corner_idx, group_of_corner, mesh.loops, pin_b)
    pos_a = mesh.positions[pin_a].astype("f8")
    pos_b = mesh.positions[pin_b].astype("f8")
    pin_dist = float(np.linalg.norm(pos_a - pos_b))
    if group_a == group_b:
        # The two chosen vertices collapsed onto the same unknown (possible
        # only on a degenerate sliver of an island) -- one pin still fixes
        # translation and rotation, and the solve below is scale-free rather
        # than wrong.
        pinned = {group_a: (0.0, 0.0)}
    else:
        pinned = {group_a: (0.0, 0.0), group_b: (pin_dist, 0.0)}

    free = [g for g in range(n_groups) if g not in pinned]
    free_index = {g: k for k, g in enumerate(free)}
    nfree = len(free)

    positions = mesh.positions.astype("f8")
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    rhs: list[float] = []
    n_rows = 0
    for tri in tri_corners.tolist():
        gverts = mesh.loops[tri].astype("i8")
        frame = _local_frame(positions[gverts[0]], positions[gverts[1]], positions[gverts[2]])
        if frame is None:
            continue
        e1len, x3, y3 = frame
        # x1 = y1 = y2 = 0, x2 = e1len (see _local_frame).
        xs = (0.0, e1len, x3)
        ys = (0.0, 0.0, y3)
        gi = [corner_to_group[c] for c in tri]
        r1, r2 = n_rows, n_rows + 1
        n_rows += 2
        b1 = b2 = 0.0
        for k in range(3):
            nxt, prv = (k + 1) % 3, (k - 1) % 3
            p_k = ys[prv] - ys[nxt]  # coefficient in grad-x (of u, and of v)
            q_k = xs[nxt] - xs[prv]  # coefficient in grad-y (of u, and of v)
            g = gi[k]
            if g in pinned:
                u0, v0 = pinned[g]
                b1 -= q_k * u0 + p_k * v0
                b2 -= -p_k * u0 + q_k * v0
                continue
            col_u, col_v = free_index[g], nfree + free_index[g]
            rows += [r1, r1, r2, r2]
            cols += [col_u, col_v, col_u, col_v]
            vals += [q_k, p_k, -p_k, q_k]
        rhs += [b1, b2]

    uv_by_group = np.zeros((n_groups, 2), dtype="f8")
    for g, (u0, v0) in pinned.items():
        uv_by_group[g] = (u0, v0)

    if n_rows and nfree:
        from scipy.sparse import coo_matrix
        from scipy.sparse.linalg import lsqr

        a = coo_matrix((vals, (rows, cols)), shape=(n_rows, 2 * nfree)).tocsr()
        b = np.array(rhs, dtype="f8")
        x = lsqr(a, b, atol=1e-10, btol=1e-10)[0]
        u_free, v_free = x[:nfree], x[nfree:]
        for g in free:
            uv_by_group[g] = (u_free[free_index[g]], v_free[free_index[g]])
    # else: every triangle in the island was degenerate (collinear or a
    # zero-length edge); the free unknowns keep their zeroed default rather
    # than an unsolved system being invented an answer.

    # One vectorised scatter for the whole island: group_of_corner is
    # already parallel to corner_idx, so no per-vertex mask-and-write and no
    # lookup back through a vertex id is needed.
    out_uv[corner_idx] = uv_by_group[group_of_corner].astype("f4")


def unwrap_lscm(
    mesh: Mesh, seams: np.ndarray | None, *, pins: list[tuple[int, int]] | None = None
) -> Mesh:
    """Cut along *seams*, flatten each resulting island with LSCM, and pack.

    *pins*, if given, is a flat list of ``(vertex_a, vertex_b)`` pairs; for
    each island the first pair both of whose vertices belong to it is used
    to anchor that island's solve, and an island matched by none picks its
    own two boundary vertices automatically (:func:`_auto_pins`). A caller
    does not need to know island numbering up front -- islands are computed
    inside this call, from *seams*, which is the flat-list shape rather than
    a dict keyed by an id the caller has not seen yet.

    Refuses (:class:`~.elements.OpError`) an island with more vertices than
    :data:`MAX_LSCM_VERTICES`, or one that is a closed surface :func:`.uvtools
    .islands_by_seams` could not cut open -- both name what to do about it in
    the message.
    """
    n_faces = face_count(mesh)
    if n_faces == 0:
        return mesh

    island_ids = islands_by_seams(mesh, seams)

    # The 2026-09-19 audit's clay-13: this ceiling used to be checked only
    # inside _solve_island, which ran *after* the whole mesh had already
    # been triangulated below (1.56 s at 490,000 vertices, 3.74 s at
    # 1,000,000 on an ordinary dense unseamed import -- the ceiling's own
    # docstring promises "not a solver that goes quiet for a minute on the
    # frame thread", and that promise was paid for in full before it could
    # ever fire). _corner_mask is now vectorised (see its own docstring),
    # so counting each island's vertices is cheap enough to do here, before
    # corner_triangles ever touches the mesh.
    for label in np.unique(island_ids).tolist():
        faces = np.flatnonzero(island_ids == label)
        corner_idx = np.flatnonzero(_corner_mask(mesh, faces))
        n_verts = len(np.unique(mesh.loops[corner_idx]))
        if n_verts > MAX_LSCM_VERTICES:
            raise OpError(
                f"This island has {n_verts} vertices, past the "
                f"{MAX_LSCM_VERTICES} an unwrap by seams reads -- mark more "
                "seams to split it, or use Smart Unwrap on a mesh this dense."
            )

    normals = face_normals(mesh)
    tri_corners, tri_face = corner_triangles(mesh.positions, mesh.loops, mesh.starts, normals)

    new_uv = np.zeros((len(mesh.loops), 2), dtype="f4")
    for label in np.unique(island_ids).tolist():
        faces = np.flatnonzero(island_ids == label)
        tris = tri_corners[np.isin(tri_face, faces)]
        _solve_island(mesh, faces, tris, seams, new_uv, pins)

    return pack_islands(replace(mesh, uv=new_uv))
