"""Bevel: the op that walks the surface before it touches it.

It starts from the seed edges and follows the mesh outward around each beveled
vertex's fan, which is what :mod:`.adjacency`'s conservative ``twin`` was built
for. A ``twin`` of -1 on a boundary, a non-manifold edge or a flipped pair stops
the walk, which is exactly what this wants: continuing across one would produce
a fan that never closes, and the failure is not visible until the geometry is
exported.

Clay no longer offers Bevel as an op; what remains here is the kernel the
``.rblk`` migration (:mod:`.legacy`) replays for a v3 file's bevel modifier.
"""

from __future__ import annotations

import numpy as np

from . import topo
from .adjacency import Adjacency, adjacency
from .elements import ElementSel, OpError
from .mesh import Mesh, face_count, face_normals

__all__ = ["bevel_edges"]


# --- bevel ------------------------------------------------------------------

# Below this, two edges at a corner count as collinear and the in-face bisector
# is undefined -- the miter distance formula divides by sin(theta/2), which is 1
# there, so only the *direction* needs the fallback.
_BISECTOR_EPS = 1e-9

#: What a bevel may run on, in mesh corners. Unlike ops_subdiv.subdivide,
#: ops_dissolve's merge and ops_boolean's kernel call, bevel_edges had no
#: ceiling at all -- the 2026-09-08 audit's clay-03 found that its rewrite
#: walks the *whole* mesh with two unconditional Python loops ("for face in
#: range(faces): ..." and "for corner in range(len(loops)): ...") no matter
#: how small the selection is, so a bevel's cost tracks the mesh clay_ops.
#: run_mesh_op calls it on -- the frame thread -- rather than the edge picked.
#:
#: The audit measured one edge beveled at 14ms on a 10k-face mesh and 254ms on
#: a 160k-face mesh: linear, about 1.6us per face (loops == 4 * faces on a
#: quad mesh; the 2026-09-11 audit's clay-09 found the original wording of
#: this line said "per corner", which reproducing the same measurement shows
#: is off by 4x -- the true per-corner rate is about 0.4us). Extrapolating
#: the per-face rate, a mesh past ~625,000 faces would stall
#: past a second on a single click; 2,000,000 corners (500,000 quad faces)
#: keeps a margin under that line, the same "well under a second" bar
#: ops_dissolve.MAX_DISSOLVED_RING uses.
MAX_BEVELED_CORNERS = 2_000_000

#: MAX_BEVELED_CORNERS was measured on a mesh with no UVs. The 2026-09-14
#: audit's clay-05 reproduced the same ceiling with UVs attached (probes
#: clay-mesh-04..07) and measured 1.62s against 0.81s without -- the extra
#: per-corner UV lerp in ``replace``/``one`` roughly doubles the cost, so the
#: plain ceiling leaves a UV-bearing mesh at almost double the "well under a
#: second" bar every other Clay op ceiling is held to. Halving the effective
#: ceiling when a mesh carries UVs restores that margin without a second
#: named constant to keep in sync with the first.
_UV_CEILING_FACTOR = 2

#: What a bevel may *touch*, in mesh corners: the corners of every vertex a
#: selected edge ends at. :data:`MAX_BEVELED_CORNERS` bounds the whole-mesh
#: copy and was measured with ONE edge selected, but every corner at a touched
#: vertex goes through Python ``replace``/miter/slide/dart work, and that costs
#: far more per corner than the copy does. The 2026-10-03 audit's clay-51 found
#: a select-all bevel at about 17 us per touched corner (0.67 s on a 40,000-
#: corner mesh; 9 ms for one edge), so the whole-mesh ceiling extrapolated to
#: roughly 30 s of frozen frame thread. Re-measured on a closed all-quad torus
#: with every edge selected (the worst ratio of edges to touched corners, so it
#: covers any smaller selection): 6,400 corners 0.23 s, 12,800 0.43 s, 20,000
#: 0.68 s, 28,800 0.99 s -- 34 us a corner here, linear. Ten thousand touched
#: corners is about 0.35 s, the "well under a second" bar the other ceilings
#: keep, and a UV-bearing mesh (which roughly doubles the per-corner cost) is
#: held to half of it for the same reason :data:`_UV_CEILING_FACTOR` halves the
#: size ceiling.
MAX_BEVELED_TOUCHED_CORNERS = 10_000


def _refuse_size(mesh: Mesh) -> None:
    """Refuse before the whole-mesh rewrite loops run, from the mesh's own size.

    Bevel's cost is in the mesh being edited, not in the selection -- see
    MAX_BEVELED_CORNERS -- so this reads ``len(mesh.loops)`` rather than
    anything selection-shaped, and is asked before any of the per-edge setup
    below, following the same "refuse before the allocation" pattern as
    ops_subdiv._refuse_growth, ops_dissolve._refuse_ring and
    ops_boolean._refuse_complexity.
    """
    grown = len(mesh.loops)
    # See _UV_CEILING_FACTOR: a UV-bearing mesh costs roughly double per
    # corner, so it is held to half the plain ceiling.
    ceiling = MAX_BEVELED_CORNERS if mesh.uv is None else MAX_BEVELED_CORNERS // _UV_CEILING_FACTOR
    if grown > ceiling:
        raise OpError(
            f"Beveling on this object would walk {grown:,} corners, past the "
            f"{ceiling:,} Clay works with. Bevel a simpler mesh, "
            "or reduce its face count first."
        )


def bevel_edges(
    mesh: Mesh, sel: ElementSel, *, width: float = 0.05
) -> tuple[Mesh, ElementSel]:
    """Replace each selected edge with a flat quad, rebuilding the corners.

    **Slide-vertex fan reconstruction.** Every vertex touched by a beveled edge
    is replaced, in each face around it, by one or two new vertices, and which
    ones is a three-row table over the corner's own two edges:

    ===================  ======================================================
    corner's two edges   what replaces the corner
    ===================  ======================================================
    neither beveled      two **slide** vertices, one along each edge
    exactly one          one **slide** vertex, along the *unbeveled* edge
    both beveled         one **miter** vertex, along the in-face bisector
    ===================  ======================================================

    A slide vertex is keyed on ``(vertex, edge)`` and therefore **shared by both
    faces flanking that edge**. That sharing is the whole crack-freeness
    argument: two faces that push their corners back along the same edge push
    them back to the same point, by construction rather than by both computing
    the same number and hoping the floats agree.

    A miter vertex is per corner, at ``width / sin(theta/2)`` along the bisector
    -- the distance at which two bevel quads meeting at that corner have their
    edges exactly ``width`` from the original ones. It is clamped to half the
    shorter incident edge so a wide bevel on a small feature collapses rather
    than turning inside out.

    Then a **bevel quad per edge**, wound from the traversal of the face that
    reads the edge forwards, and a **vertex polygon** wherever three or more
    distinct new vertices land on one original vertex -- which is exactly the
    textbook result: bevel one edge of a cube and the opposite face becomes a
    pentagon with no polygon at the corner (only two new vertices there); bevel
    all three edges at a corner and you get a miter triangle; bevel all twelve
    and the cube becomes 6 + 12 + 8 = 26 faces.

    **Refusals**: a boundary or non-manifold edge (there is no second face to
    wind the quad against), a boundary vertex carrying two or more beveled edges
    (its fan does not close, so the polygon has no ring), and a fan whose ring
    forks -- all named, all with a reason.

    **Stated limits**, none of them bugs: a single segment (no rounded profile);
    the width clamped per edge rather than solved globally, so a bevel wider
    than a feature flattens it instead of self-intersecting; the miter along the
    in-face bisector rather than the true intersection of the two offset planes,
    which drifts on a strongly non-planar corner; and no self-intersection guard
    at all.

    UV: slides **interpolated** along their edge within each face, miters keep
    their corner's uv, and the new quads and polygons copy the uv of the source
    corner each of their vertices came from -- flat in UV, and said out loud
    rather than dressed up as a projection.
    """
    if len(sel.edges) == 0:
        raise OpError("Select an edge to bevel.")
    # The 2026-10-03 audit, finding clay-93: the width Param's floor is 0, and a
    # zero-width bevel slid every corner by nothing -- coincident duplicate
    # vertices and zero-length edges as an undo step. ``not width > 0`` also
    # refuses NaN; a negative width used to be taken as its magnitude (``abs``
    # below), which no control offers.
    if not float(width) > 0.0:
        raise OpError("Bevel width must be greater than zero.")
    _refuse_size(mesh)
    a = adjacency(mesh)
    ids = a.edge_ids(sel.edges)
    if (ids < 0).any():
        raise OpError("That edge is not part of this mesh.")
    bad = a.edge_uses[ids] != 2
    if bad.any():
        pair = sel.edges[bad][0]
        uses = int(a.edge_uses[ids[bad][0]])
        why = "is on a boundary" if uses == 1 else f"has {uses} faces on it"
        raise OpError(
            f"Edge {int(pair[0])}-{int(pair[1])} {why}, so there is no second "
            "face to wind a bevel against."
        )

    if (a.twin[np.isin(a.corner_edge, ids)] < 0).any():
        raise OpError(
            "One of those edges has its two faces wound against each other, so "
            "a bevel there would inherit the flip. Fix the normals first."
        )

    touched_ceiling = (
        MAX_BEVELED_TOUCHED_CORNERS
        if mesh.uv is None
        else MAX_BEVELED_TOUCHED_CORNERS // _UV_CEILING_FACTOR
    )
    touched_corners = int(
        np.bincount(mesh.loops.astype("i8"), minlength=len(mesh.positions))[
            np.unique(a.edge_verts[ids])
        ].sum()
    )
    if touched_corners > touched_ceiling:
        raise OpError(
            f"Beveling that selection would touch {touched_corners:,} corners, "
            f"past the {touched_ceiling:,} Clay can bevel without stalling. "
            "Bevel fewer edges at a time."
        )

    beveled = set(ids.tolist())
    positions = mesh.positions.astype("f8")
    loops = mesh.loops.astype("i8")
    corner_uv = None if mesh.uv is None else mesh.uv.astype("f8")
    _check_bevel_vertices(mesh, a, beveled)

    minted: list[np.ndarray] = []
    n_verts = len(positions)

    def mint(point: np.ndarray) -> int:
        minted.append(point)
        return n_verts + len(minted) - 1

    slides: dict[tuple[int, int], tuple[int, float]] = {}

    def slide(vertex: int, edge: int) -> tuple[int, float]:
        key = (vertex, edge)
        if key not in slides:
            ends = a.edge_verts[edge]
            far = int(ends[1]) if int(ends[0]) == vertex else int(ends[0])
            delta = positions[far] - positions[vertex]
            length = float(np.linalg.norm(delta))
            s = 0.5 if length == 0.0 else min(abs(float(width)) / length, 0.5)
            slides[key] = (mint(positions[vertex] + delta * s), s)
        return slides[key]

    miters: dict[int, int] = {}
    normals_cache: list[np.ndarray] = []

    def _normals() -> np.ndarray:
        if not normals_cache:
            normals_cache.append(face_normals(mesh))
        return normals_cache[0]

    def miter(corner: int) -> int:
        if corner not in miters:
            here = int(loops[corner])
            back = positions[int(loops[a.prev_corner[corner]])] - positions[here]
            fore = positions[int(loops[a.next_corner[corner]])] - positions[here]
            n_back, n_fore = float(np.linalg.norm(back)), float(np.linalg.norm(fore))
            d1 = back / n_back if n_back else back
            d2 = fore / n_fore if n_fore else fore
            bisector = d1 + d2
            if float(np.linalg.norm(bisector)) < _BISECTOR_EPS:
                # A straight-through corner: the bisector is degenerate, so take
                # the in-face perpendicular instead of dividing by zero.
                normal = _normals()[int(a.corner_face[corner])]
                bisector = np.cross(normal.astype("f8"), d2)
            else:
                # clay-06: d1 + d2 bisects the *smaller* angle, which is the
                # outside of a reflex corner of a concave face. Flip it when
                # cross(fore, back) points against the face normal.
                normal = _normals()[int(a.corner_face[corner])].astype("f8")
                if float(np.cross(fore, back) @ normal) < 0.0:
                    bisector = -bisector
            bisector = bisector / max(float(np.linalg.norm(bisector)), _BISECTOR_EPS)
            half = np.arccos(np.clip(float(d1 @ d2), -1.0, 1.0)) * 0.5
            reach = abs(float(width)) / max(float(np.sin(half)), 1e-6)
            limit = 0.5 * min(n_back or reach, n_fore or reach)
            miters[corner] = mint(positions[here] + bisector * min(reach, limit))
        return miters[corner]

    touched = set(a.edge_verts[ids].reshape(-1).tolist())

    def replace(corner: int) -> list[tuple[int, np.ndarray | None]]:
        """The new corners replacing this one, each with its uv."""
        here = int(loops[corner])
        out_edge = int(a.corner_edge[corner])
        in_edge = int(a.corner_edge[a.prev_corner[corner]])
        out_bev, in_bev = out_edge in beveled, in_edge in beveled
        if out_bev and in_bev:
            return [(miter(corner), None if corner_uv is None else corner_uv[corner])]

        def one(edge: int, toward: int) -> tuple[int, np.ndarray | None]:
            index, s = slide(here, edge)
            if corner_uv is None:
                return index, None
            base = corner_uv[corner]
            return index, base + (corner_uv[toward] - base) * s

        back = int(a.prev_corner[corner])
        fore = int(a.next_corner[corner])
        if out_bev:
            return [one(in_edge, back)]
        if in_bev:
            return [one(out_edge, fore)]
        return [one(in_edge, back), one(out_edge, fore)]

    # --- rewrite every face -------------------------------------------------
    #
    # **A face the bevel does not touch is copied whole.** This walked every
    # corner of every face in Python -- so bevelling one edge of a 200k-face
    # sculpt did 200k iterations of the branch below for an answer that is
    # "unchanged" in all but a handful of them. The interesting faces still go
    # corner by corner, because which corners they grow depends on which of
    # their two edges are beveled and that is genuinely per corner.
    new_loops: list[int] = []
    new_uv: list[np.ndarray] = []
    counts: list[int] = []
    starts = mesh.starts.astype("i8")
    faces = face_count(mesh)
    hit = np.isin(loops, np.fromiter(touched, dtype="i8", count=len(touched)))
    per_face = (
        np.maximum.reduceat(hit.astype("i1"), starts[:-1]) > 0
        if faces
        else np.zeros(0, dtype=bool)
    )
    loop_list = loops.tolist()
    for face in range(faces):
        first, last = int(starts[face]), int(starts[face + 1])
        if not per_face[face]:
            new_loops.extend(loop_list[first:last])
            if corner_uv is not None:
                new_uv.extend(corner_uv[first:last])
            counts.append(last - first)
            continue
        size = 0
        for corner in range(first, last):
            if not hit[corner]:
                new_loops.append(loop_list[corner])
                if corner_uv is not None:
                    new_uv.append(corner_uv[corner])
                size += 1
                continue
            for index, uv_row in replace(corner):
                new_loops.append(index)
                if corner_uv is not None:
                    new_uv.append(uv_row)
                size += 1
        counts.append(size)

    # --- one quad per beveled edge, and the darts the fans chain on ---------
    # Each new face remembers a source face, so it inherits that face's
    # material slot and shading -- the rule every other face-growing op here
    # follows. Minting them at slot 0 / flat made a bevel on a painted, smooth
    # mesh sprout strips of the palette's first material.
    owners: list[int] = []
    darts: dict[int, list[tuple[int, int, int]]] = {}

    def dart(vertex: int, tail: int, head: int, source: int) -> None:
        darts.setdefault(vertex, []).append((tail, head, source))

    for corner in range(len(loops)):
        edge = int(a.corner_edge[corner])
        if edge not in beveled or int(loops[corner]) > int(loops[a.next_corner[corner]]):
            continue
        c1, c2 = corner, int(a.twin[corner])
        u, v = int(loops[c1]), int(loops[c2])
        a_u = replace(c1)[0]
        a_v = replace(int(a.next_corner[c1]))[0]
        b_v = replace(c2)[0]
        b_u = replace(int(a.next_corner[c2]))[0]
        for index, uv_row in (a_v, a_u, b_u, b_v):
            new_loops.append(index)
            if corner_uv is not None:
                new_uv.append(uv_row)
        counts.append(4)
        owners.append(int(a.corner_face[c1]))
        dart(v, a_v[0], b_v[0], int(a.next_corner[c1]))
        dart(u, b_u[0], a_u[0], int(a.next_corner[c2]))

    for vertex in sorted(touched):
        for corner in a.vertex_corners(vertex).tolist():
            pair = replace(int(corner))
            if len(pair) == 2:
                dart(vertex, pair[1][0], pair[0][0], int(corner))
        ring = _chain_darts(vertex, darts.get(vertex, []))
        if ring is None:
            continue
        for index, source in ring:
            new_loops.append(index)
            if corner_uv is not None:
                new_uv.append(corner_uv[source])
        counts.append(len(ring))
        owners.append(int(a.corner_face[ring[0][1]]))

    n_faces = face_count(mesh)
    n_new = len(counts) - n_faces
    sources = np.asarray(owners, dtype="i8")
    out, _ = topo.compact_vertices(
        topo.rebuild(
            np.concatenate([positions, np.array(minted).reshape(-1, 3)]),
            np.array(new_loops, dtype="i8"),
            topo.starts_from_counts(counts),
            np.concatenate([mesh.material, mesh.material[sources]]),
            np.concatenate([mesh.smooth, mesh.smooth[sources]]),
            uv=None if corner_uv is None else np.array(new_uv).reshape(-1, 2),
        )
    )
    return out, ElementSel(faces=np.arange(n_faces, n_faces + n_new))


def _check_bevel_vertices(mesh: Mesh, a: Adjacency, beveled: set[int]) -> None:
    """Refuse a boundary vertex carrying two or more beveled edges.

    One beveled edge ending on an open border is fine -- the two slide vertices
    it makes are all that vertex needs. Two of them want a polygon between them,
    and the fan they would have to close around does not close.
    """
    for vertex in sorted(set(a.edge_verts[sorted(beveled)].reshape(-1).tolist())):
        corners = a.vertex_corners(vertex)
        incident = set(a.corner_edge[corners].tolist()) | set(
            a.corner_edge[a.prev_corner[corners]].tolist()
        )
        if len(incident & beveled) < 2:
            continue
        open_edge = [e for e in sorted(incident) if a.edge_uses[e] != 2]
        if open_edge:
            pair = a.edge_verts[open_edge[0]]
            raise OpError(
                f"Vertex {vertex} is on the border (at edge {int(pair[0])}-"
                f"{int(pair[1])}) and carries more than one beveled edge, so the "
                "corner there cannot be closed. Bevel one of them, or fill the "
                "hole first."
            )


def _chain_darts(
    vertex: int, darts: list[tuple[int, int, int]]
) -> list[tuple[int, int]] | None:
    """Chain a vertex's darts into a ring, or ``None`` when there is no polygon.

    Fewer than three distinct new vertices means the corner closed itself --
    two slide vertices and a bevel quad between them need nothing more -- and
    that is the common case, not an error.
    """
    if len({tail for tail, _, _ in darts}) < 3:
        return None
    successor: dict[int, tuple[int, int]] = {}
    for tail, head, source in darts:
        if tail in successor:
            raise OpError(
                f"The corner at vertex {vertex} folds back on itself, so it "
                "cannot be closed with one face. Bevel fewer edges there."
            )
        successor[tail] = (head, source)

    start = darts[0][0]
    ring: list[tuple[int, int]] = []
    cursor = start
    while True:
        step = successor.get(cursor)
        if step is None:
            raise OpError(
                f"The faces around vertex {vertex} do not close into a ring, so "
                "the bevel there has no corner to fill. Fix the surface first."
            )
        ring.append((cursor, step[1]))
        cursor = step[0]
        if cursor == start:
            break
        if len(ring) > len(successor):  # pragma: no cover - the fork check covers it
            raise OpError(f"The corner at vertex {vertex} could not be closed.")
    if len(ring) != len(successor):
        raise OpError(
            f"The faces around vertex {vertex} form more than one ring, so the "
            "bevel there cannot be capped with a single face."
        )
    return ring
