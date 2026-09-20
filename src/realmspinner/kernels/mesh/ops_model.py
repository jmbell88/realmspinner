"""Tranche 5's general modelling ops: cut, slide, rip, poke, triangulate, join
and symmetrize. Lathe/screw live in :mod:`.ops_spin` -- a different family
(they build new topology from a profile rather than rewrite existing faces)
and this module was already long enough without them.

**Every op here is the one signature every op in this package is**:
``op(mesh, sel, **params) -> tuple[Mesh, ElementSel]``. A whole-mesh op
(:func:`symmetrize`) still takes ``sel`` and ignores it, the same shape
``ops_subdiv.catmull_clark`` already uses for the identical reason: a caller
wiring an op into the menu or the agent surface should never need to know
which ops read the selection and which do not.

**Bisect is the one Python per-face loop in this module**, and it is here on
purpose rather than vectorised: a plane cut produces a *different number* of
corners per face (an uncut face keeps its own count, a cut one gains exactly
two), which is exactly the shape ``ops_bevel``'s own per-corner rewrite loop
takes for the same reason -- see :data:`MAX_BISECT_CORNERS` for the ceiling
that keeps it off the frame thread on a large selection.
"""

from __future__ import annotations

import contextlib

import numpy as np

from . import ops_topo, topo
from .adjacency import adjacency
from .elements import ElementSel, OpError, empty
from .mesh import Mesh, face_count, face_normals

__all__ = [
    "bisect",
    "edge_slide",
    "grid_fill",
    "knife",
    "poke",
    "rip",
    "symmetrize",
    "triangulate_faces",
    "tris_to_quads",
    "vertex_slide",
]


def _new_face_starts(starts: np.ndarray, n_new: int, arity: int) -> np.ndarray:
    """``starts`` grown by ``n_new`` freshly appended faces of ``arity``
    corners each -- shared by every op below that appends a uniform-arity
    block of faces onto an existing CSR loop table.
    """
    grown = int(starts[-1]) + arity * np.arange(1, n_new + 1, dtype="i8")
    return np.concatenate([starts.astype("i8"), grown])


# Distances shorter than this count as "on the plane" -- a vertex here is
# neither cleared nor duplicated, it is shared by both sides exactly as a
# vertex already lying on the cut would be. Scale-free the way ``earclip``'s
# own ``TURN_EPS`` is stated to be would need a per-mesh unit; this instead
# tracks ``weld``'s own default distance (``ops_topo.weld``'s ``eps=1e-4``),
# which is the same "close enough to be the same point" judgement call, made
# at the same scale, for the reason two independently-invented numbers here
# and there would otherwise drift apart the first time either was tuned.
_PLANE_EPS = 1e-4

#: The largest selection :func:`bisect` will walk face by corner by corner in
#: Python. Every other size-varying op in this package (``ops_bevel``'s
#: rewrite loop, ``ops_topo.collapse``'s pair walk) states a measured rate and
#: extrapolates a ceiling from it; this one is new code with nothing measured
#: yet, so the bound is deliberately conservative -- a tenth of
#: ``ops_bevel.MAX_BEVELED_CORNERS`` -- rather than assumed comparable to a
#: vectorised op's own headroom. Re-measuring this against real hardware, the
#: way ``ops_topo.WELD_SEARCH_LIMIT``'s own comment says of *its* inherited
#: bound, is a reasonable future exercise.
MAX_BISECT_CORNERS = 200_000


def _refuse_bisect_size(n_corners: int) -> None:
    if n_corners > MAX_BISECT_CORNERS:
        raise OpError(
            f"Cutting this selection means walking {n_corners:,} corners one "
            f"at a time, past the {MAX_BISECT_CORNERS:,} bisect works with "
            "before the cut would stall the frame it runs on. Cut a smaller "
            "selection, or reduce the mesh's face count first."
        )


def bisect(
    mesh: Mesh,
    sel: ElementSel,
    *,
    point,
    normal,
    clear: int = 0,
    fill: bool = False,
) -> tuple[Mesh, ElementSel]:
    """Cut the selected faces with a plane, growing exactly two corners on
    every face the plane actually crosses and leaving every other face alone.

    ``clear`` is which side to remove, read as the sign of
    ``(vertex - point) . normal``: **0** keeps both pieces (the face is split
    in two, nothing deleted), **1** removes the negative side, **2** removes
    the positive side. ``fill`` caps the cut with one n-gon per closed ring it
    leaves behind, best-effort -- see below.

    **Object mode passes the whole mesh's faces as the selection.** There is
    no "cut everything" default here the way ``subdivide``'s empty selection
    means "every face": a bisect with nothing selected has no plane to draw a
    boundary against, so it refuses by name rather than guess. The door for
    "cut the whole object" is the caller filling ``sel.faces`` with
    ``np.arange(face_count(mesh))`` before calling in, exactly as
    :func:`symmetrize` (below) does internally.

    **One new vertex per crossing edge, never per corner.** Two selected
    faces sharing a crossed edge get the *same* new vertex -- keyed on the
    edge, the same rule ``ops_topo.loop_cut`` and ``ops_subdiv`` mint by, and
    for the identical reason: minting per corner would crack the seam between
    them. An **unselected** neighbour across a crossed edge gets the new
    vertex spliced into its own loop too (``topo.splice_corners``, the same
    T-junction fix ``ops_subdiv.subdivide_topology`` uses), so a partial
    selection never leaves a hairline gap at its own border.

    **A face that does not actually cross the plane is not special-cased.**
    Every corner of an all-positive face satisfies "front", so it is copied
    into the front list whole and dropped from the back one -- the same
    uniform per-corner walk handles "this face is entirely kept", "entirely
    removed" and "genuinely split" without three branches to keep in step.

    **``fill`` is best-effort, not a refusal.** It calls ``ops_topo.fill_hole``
    on the new cut boundary and, if that boundary is not a single clean closed
    ring -- open (a partial selection that never closes), pinched, or past
    ``ops_dissolve.MAX_DISSOLVED_RING`` -- silently leaves the cut open rather
    than raising: the cut itself already succeeded, and refusing the whole op
    over a cap that could not be built would throw away the part that worked.
    A selection whose cut boundary is *several* rings and only one is bad
    loses the cap on all of them this call, the one corner this shortcut cuts;
    rerunning ``fill`` after a cleanup pass caps the rest.

    UV **interpolated**: a new corner's uv is a lerp within the face that made
    it -- computed from that face's own two corner uvs at the crossed edge,
    never read from the edge's other face -- so a cut across a seam leaves the
    seam exactly where it was, the same promise ``loop_cut`` and
    ``ops_subdiv`` make.

    Selection out: the new cut edges, as vertex pairs -- the loop a follow-up
    bevel, slide or further cut would want next.
    """
    faces = sel.faces
    if len(faces) == 0:
        raise OpError(
            "Select the faces to cut; the object-mode caller passes the "
            "whole mesh's faces."
        )
    if int(clear) not in (0, 1, 2):
        raise OpError("Clear must be 0 (keep both sides), 1 (remove below) or 2 (remove above).")
    clear = int(clear)

    n = np.asarray(normal, dtype="f8").reshape(3)
    length = float(np.linalg.norm(n))
    if length <= 1e-12:
        raise OpError("The cut plane's normal has no length.")
    n = n / length
    p0 = np.asarray(point, dtype="f8").reshape(3)

    corners = topo.corner_spans(mesh.starts, faces)
    _refuse_bisect_size(len(corners))

    a = adjacency(mesh)
    d_all = (mesh.positions.astype("f8") - p0) @ n  # (V,) signed distance

    mine_edges = np.unique(a.corner_edge[corners]) if len(corners) else np.zeros(0, dtype="i8")
    ev = a.edge_verts[mine_edges].astype("i8")
    du = d_all[ev[:, 0]] if len(mine_edges) else np.zeros(0)
    dv = d_all[ev[:, 1]] if len(mine_edges) else np.zeros(0)
    crosses = ((du > _PLANE_EPS) & (dv < -_PLANE_EPS)) | ((du < -_PLANE_EPS) & (dv > _PLANE_EPS))
    cross_edges = mine_edges[crosses]

    n_verts = len(mesh.positions)
    edge_vertex = np.full(a.n_edges, -1, dtype="i8")
    edge_t = np.zeros(a.n_edges)
    new_pos = np.zeros((0, 3))
    if len(cross_edges):
        du_c, dv_c = du[crosses], dv[crosses]
        t = du_c / (du_c - dv_c)
        edge_vertex[cross_edges] = n_verts + np.arange(len(cross_edges), dtype="i8")
        edge_t[cross_edges] = t
        lo = mesh.positions[a.edge_verts[cross_edges, 0]].astype("f8")
        hi = mesh.positions[a.edge_verts[cross_edges, 1]].astype("f8")
        new_pos = lo + (hi - lo) * t[:, None]

    def edge_uv(corner: int, edge: int):
        if mesh.uv is None:
            return None
        nxt = int(a.next_corner[corner])
        from_lo = int(mesh.loops[corner]) == int(a.edge_verts[edge][0])
        frac = edge_t[edge] if from_lo else 1.0 - edge_t[edge]
        return mesh.uv[corner] + (mesh.uv[nxt] - mesh.uv[corner]) * frac

    starts = mesh.starts.astype("i8")
    front_loops: list[int] = []
    back_loops: list[int] = []
    front_uv: list = []
    back_uv: list = []
    front_counts: list[int] = []
    back_counts: list[int] = []
    front_src: list[int] = []
    back_src: list[int] = []
    new_edge_pairs: list[tuple[int, int]] = []

    for f in faces.tolist():
        lo_c, hi_c = int(starts[f]), int(starts[f + 1])
        f_front: list[int] = []
        f_back: list[int] = []
        f_front_uv: list = []
        f_back_uv: list = []
        new_here: list[int] = []
        for c in range(lo_c, hi_c):
            v = int(mesh.loops[c])
            s = float(d_all[v])
            uv_here = None if mesh.uv is None else mesh.uv[c]
            if s >= -_PLANE_EPS:
                f_front.append(v)
                f_front_uv.append(uv_here)
            if s <= _PLANE_EPS:
                f_back.append(v)
                f_back_uv.append(uv_here)
            e = int(a.corner_edge[c])
            nv = int(edge_vertex[e])
            if nv >= 0:
                new_here.append(nv)
                uv_new = edge_uv(c, e)
                f_front.append(nv)
                f_front_uv.append(uv_new)
                f_back.append(nv)
                f_back_uv.append(uv_new)
        if len(new_here) >= 2:
            new_edge_pairs.append((new_here[0], new_here[-1]))
        if clear != 2 and len(f_front) >= 3:
            front_loops.extend(f_front)
            front_uv.extend(f_front_uv)
            front_counts.append(len(f_front))
            front_src.append(f)
        if clear != 1 and len(f_back) >= 3:
            back_loops.extend(f_back)
            back_uv.extend(f_back_uv)
            back_counts.append(len(f_back))
            back_src.append(f)

    chosen = np.zeros(face_count(mesh), dtype=bool)
    chosen[faces] = True
    unselected = np.flatnonzero(~chosen)

    outside = np.flatnonzero(~chosen[a.corner_face] & (edge_vertex[a.corner_edge] >= 0))
    value_uv = None
    if mesh.uv is not None and len(outside):
        nxt = a.next_corner[outside]
        e_out = a.corner_edge[outside]
        frac = np.where(
            mesh.loops[outside] == a.edge_verts[e_out, 0], edge_t[e_out], 1.0 - edge_t[e_out]
        )
        value_uv = mesh.uv[outside] + (mesh.uv[nxt] - mesh.uv[outside]) * frac[:, None]
    loops2, starts2, uv2 = topo.splice_corners(
        mesh.loops,
        mesh.starts,
        mesh.uv,
        after=outside,
        values=edge_vertex[a.corner_edge[outside]],
        counts=np.ones(len(outside), dtype="i8"),
        value_uv=value_uv,
    )
    all_positions = np.concatenate([mesh.positions.astype("f8"), new_pos])
    base_full = topo.rebuild(all_positions, loops2, starts2, mesh.material, mesh.smooth, uv=uv2)
    base = topo.take_faces(base_full, unselected)

    new_loops = front_loops + back_loops
    counts = front_counts + back_counts
    front_material = [int(mesh.material[f]) for f in front_src]
    back_material = [int(mesh.material[f]) for f in back_src]
    material_rows = front_material + back_material
    front_smooth = [bool(mesh.smooth[f]) for f in front_src]
    back_smooth = [bool(mesh.smooth[f]) for f in back_src]
    smooth_rows = front_smooth + back_smooth
    new_uv_rows = front_uv + back_uv

    if counts:
        grown_starts = int(base.starts[-1]) + np.cumsum(np.array(counts, dtype="i8"))
        starts_out = np.concatenate([base.starts.astype("i8"), grown_starts])
    else:
        starts_out = base.starts.astype("i8")
    uv_out = None
    if mesh.uv is not None:
        rows = (
            np.array(new_uv_rows, dtype="f4").reshape(-1, 2)
            if new_uv_rows
            else np.zeros((0, 2), dtype="f4")
        )
        uv_out = np.concatenate([base.uv, rows])

    out = topo.rebuild(
        all_positions,
        np.concatenate([base.loops.astype("i8"), np.array(new_loops, dtype="i8")]),
        starts_out,
        np.concatenate([base.material, np.array(material_rows, dtype="i8")]),
        np.concatenate([base.smooth, np.array(smooth_rows, dtype=bool)]),
        uv=uv_out,
    )

    out2, old_to_new = topo.compact_vertices(out)

    if new_edge_pairs:
        pairs = np.array(new_edge_pairs, dtype="i8")
        remapped = old_to_new[pairs]
        valid = (remapped >= 0).all(axis=1)
        cut_pairs = remapped[valid].astype("i4")
    else:
        cut_pairs = np.zeros((0, 2), dtype="i4")

    if fill and clear in (1, 2) and len(cut_pairs):
        # Best-effort -- see the docstring above. The cut itself already
        # succeeded; a cap that cannot close is not a reason to lose it.
        with contextlib.suppress(OpError):
            out2, _ = ops_topo.fill_hole(out2, ElementSel(edges=cut_pairs))

    return out2, ElementSel(edges=cut_pairs)


def knife(mesh: Mesh, sel: ElementSel, *, point, normal) -> tuple[Mesh, ElementSel]:
    """:func:`bisect` restricted to the selected faces, with no clear and no
    fill -- the interactive knife's own shape, once a screen-space drag and
    the view direction have already been turned into a plane by the caller.

    A separate name rather than a default-args alias: "knife" is what a user
    reaches for and "bisect with clear=0" is an implementation detail neither
    the tools pane nor a docstring should make them read past.
    """
    return bisect(mesh, sel, point=point, normal=normal, clear=0, fill=False)


# --- sliding ------------------------------------------------------------


def _unit(v: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(v))
    return v / length if length > 1e-12 else v


def edge_slide(mesh: Mesh, sel: ElementSel, *, t: float = 0.0) -> tuple[Mesh, ElementSel]:
    """Slide every vertex of the selected edge loop along its own two rails.

    A "rail" is one of a vertex's incident edges that is **not** itself
    selected -- the edge continuing the strip sideways rather than along the
    loop. A vertex with exactly two takes them directly; a pole or a junction
    (more than two) takes the *most nearly opposite* pair, the direction a
    real quad-strip rail would be if the mesh were regular there, found by
    minimising the dot product between candidate directions rather than by
    walking a strip (``select.quad_strip`` answers "what is the strip",
    this answers "which two edges best continue it away from a single
    vertex", and the second question is all a slide needs).

    ``t`` runs **-1..1**: at ``t = -1`` a vertex sits exactly on its first
    rail's far neighbour, at ``t = +1`` on its second rail's, at ``t = 0`` it
    does not move at all. Which rail is "first" is picked by the far
    neighbour's own vertex index -- arbitrary, and stated as such, the same
    "consistent rather than meaningful" tie-break ``ops_topo.bridge_edges``'s
    own rotation search uses.

    Topology is untouched -- only positions move -- so UV is **preserved**
    trivially: every corner keeps the uv it had.
    """
    if len(sel.edges) == 0:
        raise OpError("Select an edge loop to slide.")
    a = adjacency(mesh)
    selected_edges = set(a.edge_ids(sel.edges).tolist())
    if -1 in selected_edges:
        raise OpError("That edge is not part of this mesh.")

    verts = np.unique(sel.edges.reshape(-1).astype("i8"))
    positions = mesh.positions.astype("f8").copy()
    t = float(t)

    for v in verts.tolist():
        corners = a.vertex_corners(int(v))
        e_out = a.corner_edge[corners].tolist()
        e_in = a.corner_edge[a.prev_corner[corners]].tolist()
        rails: dict[int, int] = {}
        for e in e_out + e_in:
            if e in selected_edges:
                continue
            ends = a.edge_verts[e]
            far = int(ends[1]) if int(ends[0]) == v else int(ends[0])
            rails[e] = far
        far_verts = sorted(set(rails.values()))
        if len(far_verts) < 2:
            raise OpError(
                f"Vertex {v} has no edge to slide along outside the "
                "selection; select a loop with a face on both sides."
            )
        if len(far_verts) == 2:
            a_far, b_far = far_verts
        else:
            here = positions[v]
            dirs = {fv: _unit(positions[fv] - here) for fv in far_verts}
            best = None
            for i, fi in enumerate(far_verts):
                for fj in far_verts[i + 1 :]:
                    score = float(dirs[fi] @ dirs[fj])
                    if best is None or score < best[0]:
                        best = (score, fi, fj)
            a_far, b_far = best[1], best[2]
        here = positions[v]
        if t < 0.0:
            positions[v] = here + (-t) * (positions[a_far] - here)
        elif t > 0.0:
            positions[v] = here + t * (positions[b_far] - here)

    out = topo.rebuild(positions, mesh.loops, mesh.starts, mesh.material, mesh.smooth, uv=mesh.uv)
    return out, sel


def vertex_slide(
    mesh: Mesh, sel: ElementSel, *, t: float = 0.0, direction_edge=None
) -> tuple[Mesh, ElementSel]:
    """Slide every selected vertex toward one neighbour along an edge.

    ``direction_edge`` is a direction **vector**, not an edge -- the name
    follows the task that asked for this op, which reads "the edge nearest a
    given direction". Each selected vertex picks, among its own incident
    edges, the one whose own direction has the largest dot product with
    ``direction_edge``; with no direction given, the incident edge to the
    **lowest-indexed** far vertex is used, a deterministic default rather
    than a geometric guess (the same "arbitrary but stable" convention
    :func:`edge_slide`'s own rail order uses).

    ``t`` is an unrestricted lerp fraction along the chosen edge: ``0``
    leaves the vertex where it is, ``1`` puts it exactly on the neighbour,
    and a value outside ``0..1`` overshoots past it -- the same "a plain
    magnitude, not a clamped dial" reading ``ops_topo.inset_faces``'s own
    ``depth`` is given.

    UV **preserved**: only positions move.
    """
    if len(sel.verts) == 0:
        raise OpError("Select the vertices to slide.")
    a = adjacency(mesh)
    positions = mesh.positions.astype("f8").copy()
    direction = None
    if direction_edge is not None:
        direction = _unit(np.asarray(direction_edge, dtype="f8").reshape(3))

    for v in sel.verts.astype("i8").tolist():
        if v >= len(mesh.positions):
            raise OpError(f"Vertex {v} is not part of this mesh.")
        corners = a.vertex_corners(int(v))
        out_edges = set(a.corner_edge[corners].tolist())
        in_edges = set(a.corner_edge[a.prev_corner[corners]].tolist())
        edges = out_edges | in_edges
        if not edges:
            raise OpError(f"Vertex {v} has no edge to slide along.")
        here = positions[v]
        candidates: list[tuple[int, np.ndarray]] = []
        for e in edges:
            ends = a.edge_verts[e]
            far = int(ends[1]) if int(ends[0]) == v else int(ends[0])
            candidates.append((far, positions[far]))
        if direction is None:
            far, far_pos = min(candidates, key=lambda pair: pair[0])
        else:

            def _aligned(pair: tuple[int, np.ndarray], _here: np.ndarray = here) -> float:
                return float(_unit(pair[1] - _here) @ direction)

            far, far_pos = max(candidates, key=_aligned)
        positions[v] = here + float(t) * (far_pos - here)

    out = topo.rebuild(positions, mesh.loops, mesh.starts, mesh.material, mesh.smooth, uv=mesh.uv)
    return out, sel


# --- rip ------------------------------------------------------------------

#: Total local valence :func:`rip` will union-find across the touched
#: vertices before refusing. Each vertex's own pass is O(its valence), so the
#: sum over every touched vertex is the true cost -- a conservative bound in
#: the same spirit as :data:`MAX_BISECT_CORNERS`, new code with nothing
#: measured yet, set well under a pole vertex's own worst case
#: (``primitives.MAX_SEGMENTS`` = 512 incident edges) times a selection large
#: enough to touch hundreds of such poles at once.
MAX_RIP_FAN_CORNERS = 200_000


def rip(mesh: Mesh, sel: ElementSel) -> tuple[Mesh, ElementSel]:
    """Split every vertex the selected edges touch, so the faces on each side
    of the cut stop sharing a vertex there. No vertex moves.

    **Local, not global.** Whether a vertex splits, and into how many pieces,
    is answered from that vertex's own fan of faces alone: two of its faces
    stay joined at it exactly when some edge *at that vertex*, not in the
    ripped set, still connects them -- the same "wedge" a real fan walk would
    find, computed as a small union-find over the (at most a handful of)
    faces touching that one vertex rather than a walk around it. A vertex
    whose fan does not actually split (the ripped edge there is a dead end,
    or another unripped edge already reconnects both sides) is left alone,
    which is what lets a rip line stop cleanly in the middle of a surface.

    The kept half of each split keeps the original vertex index -- the group
    containing the lowest local corner index, an arbitrary but stable choice
    -- and every other group gets one new, duplicate-position vertex, shared
    by every face in that group.

    Refuses a boundary edge (there is only one face there, nothing to
    separate) and a selection that ends up splitting nothing.

    UV **preserved**: a corner's own uv never changes, whichever vertex it
    ends up pointing at -- ripping moves no geometry and copies no corner.

    Selection out: every vertex the rip touched, kept and duplicated alike --
    what a drag apart acts on next.
    """
    if len(sel.edges) == 0:
        raise OpError("Select the edges to rip apart.")
    a = adjacency(mesh)
    ids = a.edge_ids(sel.edges)
    if (ids < 0).any():
        raise OpError("That edge is not part of this mesh.")
    bad = a.edge_uses[ids] != 2
    if bad.any():
        pair = sel.edges[bad][0]
        raise OpError(
            f"Edge {int(pair[0])}-{int(pair[1])} is on a boundary, so there "
            "is only one face there and nothing to separate."
        )
    ripped = set(ids.tolist())
    touched = np.unique(sel.edges.reshape(-1).astype("i8"))

    total_valence = int(sum(len(a.vertex_corners(int(v))) for v in touched.tolist()))
    if total_valence > MAX_RIP_FAN_CORNERS:
        raise OpError(
            f"Ripping this selection means walking {total_valence:,} face "
            f"corners across its touched vertices, past the "
            f"{MAX_RIP_FAN_CORNERS:,} rip works with before it would stall "
            "the frame it runs on. Rip a shorter run of edges."
        )

    loops2 = mesh.loops.astype("i8").copy()
    n_verts = len(mesh.positions)
    minted: dict[tuple[int, int], int] = {}
    new_positions: list[np.ndarray] = []
    split_originals: list[int] = []

    for v in touched.tolist():
        corners = a.vertex_corners(v)
        d = len(corners)
        if d <= 1:
            continue
        e_out = a.corner_edge[corners].tolist()
        e_in = a.corner_edge[a.prev_corner[corners]].tolist()
        parent = list(range(d))

        def find(x: int, parent: list[int] = parent) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        by_edge: dict[int, list[int]] = {}
        for i in range(d):
            if e_out[i] not in ripped:
                by_edge.setdefault(e_out[i], []).append(i)
            if e_in[i] not in ripped:
                by_edge.setdefault(e_in[i], []).append(i)
        for group in by_edge.values():
            for k in range(1, len(group)):
                ra, rb = find(group[0]), find(group[k])
                if ra != rb:
                    parent[ra] = rb
        roots = [find(i) for i in range(d)]
        if len(set(roots)) <= 1:
            continue
        keep_root = roots[0]
        split_originals.append(v)
        for i in range(d):
            if roots[i] == keep_root:
                continue
            key = (v, roots[i])
            if key not in minted:
                minted[key] = n_verts + len(new_positions)
                new_positions.append(mesh.positions[v].astype("f4"))
            loops2[int(corners[i])] = minted[key]

    if not minted:
        raise OpError(
            "That selection does not separate any surface; rip needs a run "
            "of edges with a face on each side that a non-selected edge "
            "does not already reconnect."
        )

    positions = (
        np.concatenate([mesh.positions, np.array(new_positions, dtype="f4")])
        if new_positions
        else mesh.positions
    )
    out = topo.rebuild(positions, loops2, mesh.starts, mesh.material, mesh.smooth, uv=mesh.uv)
    result_verts = np.array(sorted(set(split_originals) | set(minted.values())), dtype="i4")
    return out, ElementSel(verts=result_verts)


# --- poke -------------------------------------------------------------------


def poke(mesh: Mesh, sel: ElementSel, *, offset: float = 0.0) -> tuple[Mesh, ElementSel]:
    """Fan each selected face into triangles around a new centre vertex.

    The centre sits at the face's own centroid, pushed ``offset`` along the
    face normal -- zero by default, so a bare Poke changes topology only,
    exactly the "extrude at zero, then drag" shape :func:`~.ops_topo.
    extrude_faces` already uses for the same reason: a guessed offset is
    undone and redone at a different number every time.

    UV: the centre is **interpolated** (the mean of the face's own corner
    uvs, the same rule :func:`~.ops_topo.inset_faces`'s own centroid uses),
    every rim corner **preserved** -- it is the same corner it always was,
    only its face is now a triangle instead of an n-gon.
    """
    if len(sel.faces) == 0:
        raise OpError("Select at least one face to poke.")
    faces = sel.faces.astype("i8")
    starts = mesh.starts.astype("i8")
    corners = topo.corner_spans(mesh.starts, faces)
    counts = starts[faces + 1] - starts[faces]
    offsets, nxt, _ = topo.flat_next(counts)

    pts = mesh.positions[mesh.loops[corners]].astype("f8")
    centroid = np.add.reduceat(pts, offsets[:-1], axis=0) / counts[:, None]
    raw = face_normals(mesh)[faces]
    lengths = np.linalg.norm(raw, axis=1, keepdims=True)
    normals = np.divide(raw, lengths, out=np.zeros_like(raw), where=lengths > 1e-12)
    centre_pos = centroid + normals * float(offset)

    n_verts = len(mesh.positions)
    centre_idx = n_verts + np.arange(len(faces), dtype="i8")
    v_i = mesh.loops[corners].astype("i8")
    v_next = v_i[nxt]
    centre_per_corner = np.repeat(centre_idx, counts)
    tris = np.stack([v_i, v_next, centre_per_corner], axis=1)

    uv = None
    if mesh.uv is not None:
        uv_i = mesh.uv[corners].astype("f8")
        uv_next = uv_i[nxt]
        uv_centroid = np.repeat(
            np.add.reduceat(uv_i, offsets[:-1], axis=0) / counts[:, None], counts, axis=0
        )
        uv = np.stack([uv_i, uv_next, uv_centroid], axis=1).reshape(-1, 2)

    owner = np.repeat(faces, counts)
    chosen = np.zeros(face_count(mesh), dtype=bool)
    chosen[faces] = True
    unselected = np.flatnonzero(~chosen)
    base = topo.take_faces(mesh, unselected)

    out = topo.rebuild(
        np.concatenate([mesh.positions.astype("f8"), centre_pos]),
        np.concatenate([base.loops.astype("i8"), tris.reshape(-1)]),
        _new_face_starts(base.starts, len(tris), 3),
        np.concatenate([base.material, mesh.material[owner]]),
        np.concatenate([base.smooth, mesh.smooth[owner]]),
        uv=None if mesh.uv is None else np.concatenate([base.uv, uv]),
    )
    n_kept = len(unselected)
    return out, ElementSel(faces=np.arange(n_kept, n_kept + len(tris)))


# --- triangulate and rejoin ---------------------------------------------


def triangulate_faces(mesh: Mesh, sel: ElementSel) -> tuple[Mesh, ElementSel]:
    """Replace the selected faces (or all of them) with their own triangles.

    Goes through :mod:`.earclip`'s own **corner** triangulation, not
    ``mesh.triangulate``'s vertex one -- a triangle's three corners are
    literal existing corners of the source polygon, so UV is **preserved**
    outright rather than interpolated: no new point is ever minted, and using
    vertex indices instead would silently pick one of a seam's two uvs at
    random.

    An already-triangular face still costs one earclip pass and produces the
    same single triangle back, which is the idempotent case rather than a
    special one to detect.
    """
    faces = sel.faces if len(sel.faces) else np.arange(face_count(mesh), dtype="i4")
    if face_count(mesh) == 0:
        raise OpError("This object has no faces to triangulate.")
    chosen = np.zeros(face_count(mesh), dtype=bool)
    chosen[faces] = True

    from .earclip import corner_triangles

    normals = face_normals(mesh)
    tri_corners, tri_face = corner_triangles(mesh.positions, mesh.loops, mesh.starts, normals)
    keep = chosen[tri_face]
    new_tri_corners = tri_corners[keep]
    new_tri_face = tri_face[keep]

    unselected = np.flatnonzero(~chosen)
    base = topo.take_faces(mesh, unselected)
    new_loops = mesh.loops[new_tri_corners].reshape(-1)
    n_new = len(new_tri_face)

    new_uv = None if mesh.uv is None else mesh.uv[new_tri_corners].reshape(-1, 2)
    out = topo.rebuild(
        mesh.positions,
        np.concatenate([base.loops.astype("i8"), new_loops.astype("i8")]),
        _new_face_starts(base.starts, n_new, 3),
        np.concatenate([base.material, mesh.material[new_tri_face]]),
        np.concatenate([base.smooth, mesh.smooth[new_tri_face]]),
        uv=None if mesh.uv is None else np.concatenate([base.uv, new_uv]),
    )
    n_kept = len(unselected)
    return out, ElementSel(faces=np.arange(n_kept, n_kept + n_new))


#: The largest number of candidate triangle-pairs :func:`tris_to_quads` will
#: walk in its greedy consumption loop -- the same shape and the same reason
#: as ``ops_topo.collapse``'s own ``MAX_COLLAPSED_PAIRS``: one Python
#: iteration per candidate, run synchronously on the frame thread.
MAX_TRIS_TO_QUADS_CANDIDATES = 250_000


def _quad_convex(pts: np.ndarray) -> bool:
    """Whether the 4 points, taken in order, turn the same way at every
    corner -- the fixed "shape" threshold :func:`tris_to_quads` holds a
    merge to, alongside ``max_angle``. A direct four-point check rather than
    building a throwaway :class:`~.mesh.Mesh` and calling
    :func:`~.earclip.concave_faces` on it, which is the ``_refuse_concave_
    ring`` shape used elsewhere for exactly this question -- worth avoiding
    here because a candidate pair is evaluated once per edge in the whole
    mesh, not once per user action.
    """
    normal = np.cross(pts[1] - pts[0], pts[2] - pts[0]) + np.cross(pts[2] - pts[1], pts[3] - pts[1])
    for i in range(4):
        prev_p, cur, nxt = pts[i - 1], pts[i], pts[(i + 1) % 4]
        turn = np.cross(cur - prev_p, nxt - cur)
        if float(turn @ normal) < 0.0:
            return False
    return True


def tris_to_quads(
    mesh: Mesh, sel: ElementSel, *, max_angle: float = 40.0
) -> tuple[Mesh, ElementSel]:
    """Greedily join adjacent selected triangle pairs into quads.

    A pair qualifies when **both** thresholds pass: the dihedral angle
    between their two face normals is at most ``max_angle`` degrees (they
    read as "should have been one face"), and the merged quad is convex (the
    fixed shape check -- see :func:`_quad_convex`; a reflex corner there is
    exactly the kind of merge a fan-triangulated cap would produce and that
    :mod:`.earclip` exists to *not* need this op to also solve). Neither
    threshold is optional; ``max_angle`` is the one number this op takes.

    **Greedy, in a fixed order**: candidates are sorted by dihedral angle,
    flattest first, tied on the shared edge id, and consumed one at a time --
    a triangle already merged into a quad cannot join a second one. That
    order is what makes two runs over the same selection produce the same
    quads; a set's own iteration order would not.

    Faces this op does not touch -- non-triangles, unselected faces, a
    triangle whose only candidate partner was already taken -- are copied
    through unchanged. Nothing is selected past when it merges into nothing:
    finding zero mergeable pairs is a legitimate outcome, not a refusal.

    UV **preserved**: the merged quad's four corners are the four existing
    corners of the two source triangles (two apexes and the shared edge's
    pair), read directly rather than interpolated -- the same "an existing
    corner, not a new point" reasoning :func:`triangulate_faces` uses in the
    other direction. Material and smooth follow the *lower-indexed* source
    triangle, a documented and deterministic choice for the case the pair
    disagrees.
    """
    if face_count(mesh) == 0:
        raise OpError("This object has no faces to join.")
    faces = sel.faces if len(sel.faces) else np.arange(face_count(mesh), dtype="i4")
    chosen = np.zeros(face_count(mesh), dtype=bool)
    chosen[faces] = True
    arity = np.diff(mesh.starts.astype("i8"))
    tri_mask = chosen & (arity == 3)

    a = adjacency(mesh)
    interior = np.flatnonzero(a.twin >= 0)
    c1 = interior[mesh.loops[interior] < mesh.loops[a.next_corner[interior]]]
    f1 = a.corner_face[c1].astype("i8")
    c2 = a.twin[c1].astype("i8")
    f2 = a.corner_face[c2].astype("i8")
    both = tri_mask[f1] & tri_mask[f2] & (f1 != f2)
    c1, c2, f1, f2 = c1[both], c2[both], f1[both], f2[both]
    if len(c1) > MAX_TRIS_TO_QUADS_CANDIDATES:
        raise OpError(
            f"Joining this selection means weighing {len(c1):,} candidate "
            f"triangle pairs, past the {MAX_TRIS_TO_QUADS_CANDIDATES:,} "
            "tris-to-quads walks without stalling. Join a smaller selection."
        )

    normals = face_normals(mesh)
    unit = np.zeros_like(normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    np.divide(normals, lengths, out=unit, where=lengths > 1e-12)
    cos = np.clip(np.einsum("ij,ij->i", unit[f1], unit[f2]), -1.0, 1.0)
    angles = np.degrees(np.arccos(cos))
    within_angle = angles <= float(max_angle)

    order = np.lexsort((c1[within_angle], angles[within_angle]))
    idx = np.flatnonzero(within_angle)[order]

    consumed = np.zeros(face_count(mesh), dtype=bool)
    positions = mesh.positions.astype("f8")
    merged_loops: list[int] = []
    merged_uv: list = []
    merged_src: list[int] = []

    for k in idx.tolist():
        fa, fb = int(f1[k]), int(f2[k])
        if consumed[fa] or consumed[fb]:
            continue
        cc1, cc2 = int(c1[k]), int(c2[k])
        # f1's own three corners are [apex1, cc1 (va), next(cc1) (vb)]; f2
        # traverses the shared edge the other way, so *its* apex is the
        # corner after va within f2, i.e. two steps past cc2 -- prev_corner
        # on a triangle, since next^2 == prev at arity 3 -- not next_corner
        # (that lands back on va itself, not on f2's own third vertex).
        apex1 = int(a.prev_corner[cc1])
        apex2 = int(a.prev_corner[cc2])
        quad_corners = [apex1, cc1, apex2, int(a.next_corner[cc1])]
        quad_verts = mesh.loops[quad_corners].astype("i8")
        if not _quad_convex(positions[quad_verts]):
            continue
        consumed[fa] = True
        consumed[fb] = True
        merged_loops.extend(quad_verts.tolist())
        if mesh.uv is not None:
            merged_uv.extend(mesh.uv[quad_corners].tolist())
        merged_src.append(min(fa, fb))

    if not merged_src:
        return mesh, sel

    kept = np.flatnonzero(~consumed)
    base = topo.take_faces(mesh, kept)
    n_kept = len(kept)
    n_new = len(merged_src)
    src = np.array(merged_src, dtype="i8")

    merged_uv_arr = None if mesh.uv is None else np.array(merged_uv, dtype="f4").reshape(-1, 2)
    out = topo.rebuild(
        mesh.positions,
        np.concatenate([base.loops.astype("i8"), np.array(merged_loops, dtype="i8")]),
        _new_face_starts(base.starts, n_new, 4),
        np.concatenate([base.material, mesh.material[src]]),
        np.concatenate([base.smooth, mesh.smooth[src]]),
        uv=None if mesh.uv is None else np.concatenate([base.uv, merged_uv_arr]),
    )
    return out, ElementSel(faces=np.arange(n_kept, n_kept + n_new))


# --- symmetrize -------------------------------------------------------------

#: The distance :func:`symmetrize` welds its own seam at, after mirroring the
#: kept half back across the plane. Matches :data:`_PLANE_EPS` -- see its own
#: comment for why that tracks ``ops_topo.weld``'s default rather than
#: inventing a third number for "close enough to be the same point".
_SEAM_WELD_EPS = _PLANE_EPS


def symmetrize(
    mesh: Mesh, sel: ElementSel, *, axis: int, direction: int
) -> tuple[Mesh, ElementSel]:
    """Delete the half on ``direction``'s side of the local plane through the
    object's own origin, mirror what remains across it, and weld the seam.

    **Whole-mesh, and ``sel`` is ignored** -- the same shape
    ``ops_subdiv.catmull_clark`` already takes for a whole-mesh smoothing
    pass, kept here so a caller wiring ops into a menu never has to know
    which ones read the selection.

    Built from :func:`bisect` rather than a second cutting pass: the kept
    half is exactly what an object-mode bisect through the origin, clearing
    the other side, already produces, uncapped (``fill=False``) so its open
    edge lines up with its own mirror image rather than with a cap that would
    have to be welded away again. :func:`~.mesh.transformed` does the
    mirroring, which is also what reverses every face's winding -- see its
    own docstring for why that is not optional.

    The seam weld is scoped to the vertices *on the plane* after mirroring
    (within :data:`_SEAM_WELD_EPS`), not the whole mesh -- welding
    everything would merge any other pair of coincident vertices the source
    mesh happened to have, which is not what "close the seam" means.
    """
    del sel
    axis = int(axis)
    if axis not in (0, 1, 2):
        raise OpError("Axis must be 0 (X), 1 (Y) or 2 (Z).")
    if face_count(mesh) == 0:
        raise OpError("This object has no faces to symmetrize.")
    side = 1 if float(direction) >= 0 else -1

    normal = np.zeros(3)
    normal[axis] = 1.0
    all_faces = np.arange(face_count(mesh), dtype="i4")
    clear = 2 if side > 0 else 1
    kept, _ = bisect(
        mesh,
        ElementSel(faces=all_faces),
        point=(0.0, 0.0, 0.0),
        normal=normal,
        clear=clear,
        fill=False,
    )
    if face_count(kept) == 0:
        return kept, empty()

    mirror = np.eye(4)
    mirror[axis, axis] = -1.0
    from .mesh import transformed

    mirrored = transformed(kept, mirror)

    n = len(kept.positions)
    grown_starts = int(kept.starts[-1]) + mirrored.starts[1:].astype("i8")
    combined = topo.rebuild(
        np.concatenate([kept.positions, mirrored.positions]),
        np.concatenate([kept.loops.astype("i8"), mirrored.loops.astype("i8") + n]),
        np.concatenate([kept.starts.astype("i8"), grown_starts]),
        np.concatenate([kept.material, mirrored.material]),
        np.concatenate([kept.smooth, mirrored.smooth]),
        uv=None if kept.uv is None else np.concatenate([kept.uv, mirrored.uv]),
    )
    seam = np.flatnonzero(np.abs(combined.positions[:, axis].astype("f8")) <= _SEAM_WELD_EPS)
    if len(seam) >= 2:
        out, _ = ops_topo.weld(combined, ElementSel(verts=seam.astype("i4")), eps=_SEAM_WELD_EPS)
    else:
        out = combined
    return out, empty()


# --- grid fill ----------------------------------------------------------

#: The largest ``rows * span`` grid of quads :func:`grid_fill` will build.
#: Built the same way ``primitives.MAX_DIVISIONS`` bounds ``grid``'s own
#: ``divisions * divisions`` -- a conservative order of magnitude under
#: ``glbimport.MAX_TRIANGLES``, for a boundary loop an agent or a hand-drawn
#: selection could in principle make very large.
MAX_GRID_FILL_QUADS = 65_536


def grid_fill(mesh: Mesh, sel: ElementSel, *, span: int) -> tuple[Mesh, ElementSel]:
    """Fill the selected closed boundary loop with a grid of quads.

    The loop -- found the way :func:`~.ops_topo.fill_hole` finds one, via
    :func:`~.adjacency.boundary_ring_from` -- must have an **even** number of
    vertices: a grid fill treats it as the perimeter of a rectangle of quads,
    ``span`` columns by however many rows the remaining half of the perimeter
    implies, and an odd perimeter has no way to split into two matching
    halves at all. Refused by name rather than rounded, the same "an
    ambiguous count is not a number to guess at" reasoning
    :func:`~.primitives._clamp_profile` gives its own floors.

    The four "corners" of that rectangle are the loop's own vertex at index
    0, ``rows``, ``rows + span`` and ``2*rows + span`` -- a deterministic cut
    of the loop into two long sides of ``rows + 1`` vertices and two short
    sides of ``span + 1``, not a search for the "natural" corners a very
    non-rectangular loop might visually suggest. Interior grid points are
    placed by a Coons patch (bilinear blend of the four boundary polylines)
    so the fill matches a curved or uneven boundary rather than assuming it
    is flat; every boundary row and column reuses the loop's own vertices
    outright, and only genuinely interior points are new.

    UV **preserved** on the boundary (existing corners), **interpolated** at
    every interior point -- the same bilinear blend the positions use,
    applied to uv when the mesh has any, which keeps a textured fill
    continuous with the surface around it rather than at the origin the way
    ``fill_hole``'s own placeholder is.
    """
    if len(sel.edges) == 0:
        raise OpError("Select the boundary loop to fill.")
    span = int(span)
    if span < 1:
        raise OpError("Span must be at least 1 column.")

    from .adjacency import boundary_ring_from

    chosen, pinched = boundary_ring_from(mesh, sel.edges)
    if not chosen:
        raise OpError("No boundary ring runs through the selected edge.")
    if len(chosen) != 1:
        raise OpError("Select the edges of a single closed boundary loop.")
    ring = chosen[0]
    if pinched.size and set(pinched.tolist()) & set(ring.tolist()):
        raise OpError("That boundary pinches at a vertex, so it has no single ring to grid-fill.")
    if len(np.unique(ring)) != len(ring):
        raise OpError("That boundary crosses itself, so it cannot be grid-filled.")

    total = len(ring)
    if total % 2 != 0:
        raise OpError(
            f"The boundary has {total} vertices; grid fill needs an even "
            "number to pair opposite sides evenly."
        )
    rows = total // 2 - span
    if rows < 1:
        raise OpError(
            f"Span {span} leaves no rows; with {total} boundary vertices, "
            f"span must be between 1 and {total // 2 - 1}."
        )
    if rows * span > MAX_GRID_FILL_QUADS:
        raise OpError(
            f"That fill would build {rows * span:,} quads, past the "
            f"{MAX_GRID_FILL_QUADS:,} grid fill works with. Fill a smaller "
            "boundary, or choose a different span."
        )

    ring_l = ring.astype("i8").tolist()
    side_a = ring_l[0 : rows + 1]
    side_b = ring_l[rows : rows + span + 1]
    side_c = list(reversed(ring_l[rows + span : 2 * rows + span + 1]))
    side_d = list(reversed(ring_l[2 * rows + span :] + [ring_l[0]]))

    positions = mesh.positions.astype("f8")
    uv = None if mesh.uv is None else mesh.uv.astype("f8")

    def corner_uv_of(vertex: int):
        if uv is None:
            return None
        a = adjacency(mesh)
        return uv[int(a.vertex_corners(vertex)[0])]

    grid_vert = [[-1] * (span + 1) for _ in range(rows + 1)]
    grid_pos = [[None] * (span + 1) for _ in range(rows + 1)]
    grid_uv = [[None] * (span + 1) for _ in range(rows + 1)]
    n_verts = len(mesh.positions)
    new_positions: list[np.ndarray] = []

    for i in range(rows + 1):
        for j in range(span + 1):
            if i == 0:
                v = side_d[j]
            elif i == rows:
                v = side_b[j]
            elif j == 0:
                v = side_a[i]
            elif j == span:
                v = side_c[i]
            else:
                v = None
            if v is not None:
                grid_vert[i][j] = v
                grid_pos[i][j] = positions[v]
                if uv is not None:
                    grid_uv[i][j] = corner_uv_of(v)

    for i in range(rows + 1):
        for j in range(span + 1):
            if grid_vert[i][j] != -1:
                continue
            u = i / rows
            w = j / span
            sc = (1 - w) * grid_pos[i][0] + w * grid_pos[i][span]
            sd = (1 - u) * grid_pos[0][j] + u * grid_pos[rows][j]
            sb = (
                (1 - u) * (1 - w) * grid_pos[0][0]
                + u * (1 - w) * grid_pos[rows][0]
                + (1 - u) * w * grid_pos[0][span]
                + u * w * grid_pos[rows][span]
            )
            pos = sc + sd - sb
            idx = n_verts + len(new_positions)
            new_positions.append(pos.astype("f4"))
            grid_vert[i][j] = idx
            grid_pos[i][j] = pos
            if uv is not None:
                uc = (1 - w) * grid_uv[i][0] + w * grid_uv[i][span]
                ud = (1 - u) * grid_uv[0][j] + u * grid_uv[rows][j]
                ub = (
                    (1 - u) * (1 - w) * grid_uv[0][0]
                    + u * (1 - w) * grid_uv[rows][0]
                    + (1 - u) * w * grid_uv[0][span]
                    + u * w * grid_uv[rows][span]
                )
                grid_uv[i][j] = uc + ud - ub

    faces: list[list[int]] = []
    uv_faces: list[list] = []
    for i in range(rows):
        for j in range(span):
            quad = [
                grid_vert[i][j],
                grid_vert[i + 1][j],
                grid_vert[i + 1][j + 1],
                grid_vert[i][j + 1],
            ]
            faces.append(quad)
            if uv is not None:
                uv_quad = [
                    grid_uv[i][j],
                    grid_uv[i + 1][j],
                    grid_uv[i + 1][j + 1],
                    grid_uv[i][j + 1],
                ]
                uv_faces.append(uv_quad)

    n_faces = face_count(mesh)
    all_positions = (
        np.concatenate([mesh.positions, np.array(new_positions, dtype="f4")])
        if new_positions
        else mesh.positions
    )
    flat_loops = np.array([v for quad in faces for v in quad], dtype="i8")
    flat_uv = None
    if uv is not None:
        flat_uv = np.array([row for quad in uv_faces for row in quad], dtype="f4").reshape(-1, 2)

    out = topo.rebuild(
        all_positions,
        np.concatenate([mesh.loops.astype("i8"), flat_loops]),
        _new_face_starts(mesh.starts, len(faces), 4),
        np.concatenate([mesh.material, np.zeros(len(faces), dtype="i8")]),
        np.concatenate([mesh.smooth, np.zeros(len(faces), dtype=bool)]),
        uv=None if mesh.uv is None else np.concatenate([mesh.uv, flat_uv]),
    )
    return out, ElementSel(faces=np.arange(n_faces, n_faces + len(faces)))
