"""Simplify: polygon reduction down to a face budget.

Two stages, and the order is the whole design.

**1. Flat merge -- always first, and free.** A subdivided box, a tessellated
floor or an imported triangulated wall carries hundreds of faces that say
nothing a single polygon does not. Faces that are edge-adjacent, share a
material and a smooth flag, lie in one plane and carry one continuous UV map
across that plane are grouped and each group becomes one n-gon through
:func:`~.ops_dissolve.merge_groups`; then every vertex that is *straight* in
every face that uses it is dropped. A 10x10x10-subdivided box ends as six quads
and eight vertices, and nothing about its shape or texture has moved -- which
is why this stage needs no tolerance on *how much* it removes, only on what
counts as flat. Triangles are not required: the result is n-gons, because
that is what the plane really is.

*Coplanar means coplanar with the group's seed face*, not with the face next
to it. Pairwise agreement within a degree chains around a gentle curve, so a
finely tessellated sphere would fold into one huge "flat" polygon; testing every
member against the one plane its group started from bounds the drift by the
tolerance however long the chain. The UV map gets the same treatment: every
corner of a group must lie on the *affine* map the seed's own corners define, so
merging a flat region whose UVs the user had deliberately warped is refused
rather than silently straightened. (Merging keeps only border-corner UVs and the
renderer interpolates linearly across the n-gon, which reproduces an affine map
exactly and nothing else.)

**A vertex is dropped only when it is straight in every face that uses it.**
Removing it from one face and not another leaves the neighbour with a corner the
first face's edge does not end at -- a T-junction, which renders as a hairline
crack that moves with the camera (the reason ``topo.splice_corners`` exists).
Candidates are removed as an independent set per round (no two neighbours at
once, chosen by a fixed random priority so a long straight run clears in a
handful of rounds). Each edge carries a running bound on how far it now lies from
the vertices dropped from it, and a candidate is accepted only while that bound
stays inside the tolerance, so a finely tessellated flat circle cannot be eaten
down to a triangle one in-tolerance step at a time -- the tolerance is a bound on
the whole stage, not on each round.

**2. Edge collapse -- only when the flat merge was not enough.** The remainder is
triangulated and collapsed cheapest-first by quadric error metric until the
budget is met or no legal collapse is left. Legal means all of:

* **Not across a border.** A vertex on an open boundary, a non-manifold or
  flipped edge, a material change, a smooth-flag change or a UV seam is *locked*,
  as is a vertex whose faces do not form one fan (a pinch) or whose corners carry
  different UVs. A locked vertex never moves; a collapse of two locked vertices
  is refused. A free vertex may be pulled onto a locked neighbour, which is how
  the interior of a region is thinned without ever touching its outline.
* **Manifold in, manifold out.** The link condition (the two endpoints share
  exactly the two vertices opposite the edge and no others), and never below
  four faces. Closed stays closed and a vertex never ends up shared by two
  separate fans.
* **No fold.** No surviving face's normal turns past a limit or shrinks to a
  sliver, and no surviving face's UV orientation reverses.

UV **interpolated**: a free vertex has one UV on every face around it (that is
what "not a seam" means), so the merged vertex gets one UV too -- the lerp of the
two endpoints' along the edge at the point the new position projects to -- and
every surviving face reads it. Giving each face its own extrapolation instead
would put a hairline seam down every collapsed edge. Onto a locked vertex the
new corners take that vertex's UV *as the collapsed edge's own faces see it*,
which is the island the surviving faces already belong to.

Stage two is a Python heap, which is what :data:`MAX_SIMPLIFY_CORNERS` bounds.
The result of stage two is triangles; if it ends with no fewer faces than the
flat merge left (nothing legal to collapse) the flat-merge result is returned
instead, so Simplify never trades n-gons for more triangles.

Returns the **same mesh object** when nothing changed, which the GPU cache and
undo both key on (see :mod:`.ops_clean`). Preserves winding, per-face material
and smooth, and per-corner UVs.
"""

from __future__ import annotations

import heapq
import itertools

import numpy as np

from . import earclip, ops_dissolve, ops_topo, topo
from .adjacency import adjacency
from .elements import OpError, empty
from .mesh import Mesh, face_count, face_normals

__all__ = [
    "MAX_SIMPLIFY_CORNERS",
    "MIN_SIMPLIFY_TARGET",
    "selection_cost",
    "simplify",
    "simplify_cost",
]

#: A closed solid needs four faces; a lower budget would be refused by the first
#: legal collapse anyway, so it is refused up front with a sentence. The floor is
#: **per shell** (see :func:`_collapse`): a mesh of several shells keeps four
#: faces in each, so the total it can reach is four times the shell count.
MIN_SIMPLIFY_TARGET = 4

#: The most corners one Simplify will take, halved when the mesh carries UVs.
#:
#: Stage two is a Python heap and is the cost: it is linear in the *triangles*
#: the mesh splits into, and a quad is two of them per four corners, the dearest
#: ordinary shape. ``Simplify`` runs on the frame thread from a button with no
#: cancel, and a stall there is the failure every ceiling in this package exists
#: to prevent (the same argument as ``MAX_DISSOLVED_RING``): the package bar is a
#: press that finishes in about a second.
#:
#: **20,000, lowered from 50,000 by the 2026-10-11 audit's clay-08.** The first
#: ceiling was set from a gently curved quad heightfield (59,536 corners 2.29 s,
#: 30,276 with UVs 1.41 s) and then read as "about 1.9 s", which is not the bar;
#: re-measured the same day, best of three, collapsed all the way to 50 faces:
#:
#: * closed quad sphere, no UVs: 48,600 corners 1.34 s (1.81 s stretched into an
#:   ellipsoid), 24,576 corners 0.64 s (0.79 s), 18,816 corners 0.45 s (0.52 s);
#: * gently curved quad heightfield, no UVs: 25,600 corners 0.74 s, 19,600 0.63 s;
#: * the same heightfield with UVs: 10,000 corners 0.48 s -- the dearest per
#:   corner, which is why a mesh with UVs is charged double.
#:
#: So 20,000 bare and 10,000 with UVs is at worst about two thirds of a second.
#: The triangle count is bounded alongside (see :func:`simplify_cost`) because an
#: n-gon mesh has more triangles per corner than a quad one does, and the op
#: charges a whole selection with the same function (:func:`selection_cost`), so
#: one press is bounded, not just one object.
MAX_SIMPLIFY_CORNERS = 20_000

#: Faces whose normals are within this angle may share a flat group.
_FLAT_COS = float(np.cos(np.radians(1.0)))
#: Coplanar / collinear distance, as a fraction of the mesh's bounding diagonal.
#: Well above float32 noise on a unit-scale mesh (~1e-7) and well below any
#: feature a user models, so a deliberate step is never flattened.
_FLAT_TOL = 1e-4
#: UV agreement, in UV units (a tenth of a texel on a 1024 map).
_UV_TOL = 1e-4

#: A collapse may not turn a surviving face's normal past this cosine, or shrink
#: it below this fraction of its old size -- both are the signature of a fold.
_FLIP_COS = 0.5
_MIN_AREA_RATIO = 0.02
_MAX_ROUNDS = 64


def simplify_cost(mesh: Mesh) -> int:
    """What one mesh costs against :data:`MAX_SIMPLIFY_CORNERS`, in corners.

    The corners themselves, or twice the triangles once every face is split
    (``corners - 2 * faces`` of them; a quad costs half a triangle per corner, so
    the ceiling in triangles is half the one in corners and a mesh of bigger
    polygons meets it sooner), whichever is more -- and doubled when the mesh
    carries UVs, which is the same as halving the ceiling for it. A mesh is
    refused exactly when this exceeds the ceiling, and the op sums it over a
    selection (:func:`selection_cost`), so the kernel and the press cannot
    disagree about what a mesh costs (the 2026-10-11 audit's clay-06 found the
    op summing bare corners against the kernel's halved ceiling).
    """
    triangles = len(mesh.loops) - 2 * face_count(mesh)
    return max(len(mesh.loops), 2 * triangles) * (2 if mesh.uv is not None else 1)


def selection_cost(meshes, target: int) -> int:
    """The summed :func:`simplify_cost` of the *meshes* a press would work on.

    Only a mesh with more faces than *target* is charged: :func:`simplify` hands
    the rest back untouched without deriving anything from them (the 2026-10-11
    audit's clay-24 found two 40,000-corner grids already under the target
    refused as a pair).
    """
    return sum(simplify_cost(m) for m in meshes if face_count(m) > int(target))


def _refuse_if_too_large(mesh: Mesh) -> None:
    """Refuse from the input's own counts, before anything is derived from it."""
    ceiling = MAX_SIMPLIFY_CORNERS // 2 if mesh.uv is not None else MAX_SIMPLIFY_CORNERS
    triangles = len(mesh.loops) - 2 * face_count(mesh)
    if simplify_cost(mesh) > MAX_SIMPLIFY_CORNERS:
        raise OpError(
            f"This mesh has {len(mesh.loops):,} corners ({triangles:,} triangles "
            f"once split), past what Simplify can work through without stalling "
            f"-- {ceiling:,} corners. Split the mesh first, or use Merge Faces "
            "on the flat parts."
        )


def simplify(mesh: Mesh, target: int) -> Mesh:
    """Reduce *mesh* toward *target* faces; fewer than asked is never an error.

    Flat regions merge first (see the module docstring); only if that leaves
    more than *target* faces are edges collapsed. When the target cannot be
    reached the result is as far as it legally got -- the caller reports the
    achieved count by comparing ``face_count``.

    Returns *mesh* itself when it already has *target* faces or fewer, or when
    nothing could be removed.
    """
    target = int(target)
    if target < MIN_SIMPLIFY_TARGET:
        raise OpError(
            f"Simplify needs a target of at least {MIN_SIMPLIFY_TARGET} faces; "
            "anything fewer cannot enclose a solid."
        )
    if face_count(mesh) <= target:
        return mesh
    _refuse_if_too_large(mesh)

    flat = _flat_merge(mesh)
    if face_count(flat) <= target:
        return flat
    collapsed = _collapse(flat, target)
    if collapsed is None or face_count(collapsed) >= face_count(flat):
        return flat
    return collapsed


# --- stage 1: flat merge ----------------------------------------------------


def _extent(mesh: Mesh) -> float:
    """The bounding diagonal of the vertices a face actually uses."""
    if len(mesh.loops) == 0:
        return 0.0
    used = mesh.positions[mesh.loops].astype("f8")
    return float(np.linalg.norm(used.max(axis=0) - used.min(axis=0)))


def _flat_groups(mesh: Mesh) -> list[np.ndarray]:
    """Groups of two or more faces that can become one polygon, ascending."""
    n_faces = face_count(mesh)
    diag = _extent(mesh)
    if n_faces < 2 or diag <= 0.0:
        return []
    tol = _FLAT_TOL * diag
    a = adjacency(mesh)
    pos = mesh.positions.astype("f8")
    loops = mesh.loops.astype("i8")
    starts = mesh.starts.astype("i8")
    counts = np.diff(starts)
    corner_face = a.corner_face.astype("i8")

    normals = face_normals(mesh)
    area2 = np.linalg.norm(normals, axis=1)
    live = area2 > 1e-12 * diag * diag
    unit = np.divide(normals, area2[:, None], out=np.zeros_like(normals), where=live[:, None])
    centre = np.add.reduceat(pos[loops], starts[:-1], axis=0) / counts[:, None]

    uv = None if mesh.uv is None else mesh.uv.astype("f8")
    nxt = a.next_corner.astype("i8")
    uv_mean = grad_u = grad_v = None
    if uv is not None:
        # The mean gradient of a piecewise-linear map over a planar polygon, by
        # the divergence theorem: the integral of grad(u) is the boundary
        # integral of u times the outward edge normal (e x n for a polygon wound
        # counter-clockwise about n). For an affine map it is exact, and
        # *whether* the map is affine is what the residual test below asks.
        outward = np.cross(pos[loops[nxt]] - pos[loops], unit[corner_face])
        inv_area = np.divide(2.0, area2, out=np.zeros_like(area2), where=live)[:, None]
        mid = (uv + uv[nxt]) * 0.5
        grad_u = np.add.reduceat(outward * mid[:, :1], starts[:-1], axis=0) * inv_area
        grad_v = np.add.reduceat(outward * mid[:, 1:], starts[:-1], axis=0) * inv_area
        uv_mean = np.add.reduceat(uv, starts[:-1], axis=0) / counts[:, None]

    # Candidate pairs: one per twinned edge. ``twin`` is -1 on a boundary, a
    # non-manifold edge and a flipped pair, none of which may carry a merge.
    corner = np.arange(len(loops), dtype="i8")
    twin = a.twin.astype("i8")
    first = np.flatnonzero(twin > corner)
    other = twin[first]
    f, g = corner_face[first], corner_face[other]
    keep = live[f] & live[g]
    keep &= mesh.material[f] == mesh.material[g]
    keep &= mesh.smooth[f] == mesh.smooth[g]
    keep &= np.einsum("ij,ij->i", unit[f], unit[g]) >= _FLAT_COS
    keep &= np.abs(np.einsum("ij,ij->i", unit[f], centre[g] - centre[f])) <= tol
    keep &= np.abs(np.einsum("ij,ij->i", unit[g], centre[f] - centre[g])) <= tol
    if uv is not None:
        keep &= (np.abs(uv[first] - uv[nxt[other]]) <= _UV_TOL).all(axis=1)
        keep &= (np.abs(uv[nxt[first]] - uv[other]) <= _UV_TOL).all(axis=1)
    f, g = f[keep], g[keep]
    if len(f) == 0:
        return []

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    graph = coo_matrix((np.ones(len(f), dtype="i1"), (f, g)), shape=(n_faces, n_faces))
    n_comp, label = connected_components(graph, directed=False)

    def conforms(seed: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """Per face: does it lie on *seed*'s plane and, with UVs, on its map."""
        ok = np.einsum("ij,ij->i", unit[faces], unit[seed]) >= _FLAT_COS
        owner = np.repeat(np.arange(len(faces), dtype="i8"), counts[faces])
        corners = topo.corner_spans(mesh.starts, faces)
        s = seed[owner]
        d = pos[loops[corners]] - centre[s]
        bad = np.abs(np.einsum("ij,ij->i", d, unit[s])) > tol
        if uv is not None:
            predicted = np.stack(
                [
                    uv_mean[s, 0] + np.einsum("ij,ij->i", d, grad_u[s]),
                    uv_mean[s, 1] + np.einsum("ij,ij->i", d, grad_v[s]),
                ],
                axis=1,
            )
            bad |= (np.abs(predicted - uv[corners]) > _UV_TOL).any(axis=1)
        return ok & (np.bincount(owner[bad], minlength=len(faces)) == 0)

    size = np.bincount(label, minlength=n_comp)
    members = np.flatnonzero(size[label] > 1)
    # The lowest-numbered face of each component seeds it (faces are scanned
    # ascending, so first appearance is the minimum).
    _, first_seen = np.unique(label[members], return_index=True)
    seed_of = np.zeros(n_comp, dtype="i8")
    seed_of[label[members][first_seen]] = members[first_seen]
    fits = conforms(seed_of[label[members]], members)

    broken = np.zeros(n_comp, dtype=bool)
    broken[label[members[~fits]]] = True
    whole = members[~broken[label[members]]]
    order = np.argsort(label[whole], kind="stable")
    whole = whole[order]
    splits = np.flatnonzero(np.diff(label[whole])) + 1
    groups = [grp for grp in np.split(whole, splits) if len(grp) > 1]

    if broken.any():
        # A component the one-seed test rejected is a drifting surface (a dense
        # gentle curve, a warped UV patch). Grow each group from its own seed,
        # a ring of faces at a time, so what is merged is exactly what lies on
        # the seed's plane and map. A face that fails one seed is left free for
        # the next group to claim.
        src = np.concatenate([f, g])
        dst = np.concatenate([g, f])
        inside = broken[label[src]]
        src, dst = src[inside], dst[inside]
        order = np.argsort(src, kind="stable")
        nbr = dst[order]
        nbr_starts = np.concatenate([[0], np.cumsum(np.bincount(src, minlength=n_faces))])
        taken = np.zeros(n_faces, dtype=bool)
        for start in members[broken[label[members]]].tolist():
            if taken[start]:
                continue
            root = np.array([start], dtype="i8")
            if not conforms(root, root)[0]:
                continue
            taken[start] = True
            grown = [root]
            ring = root
            while len(ring):
                reach = np.unique(nbr[topo.corner_spans(nbr_starts, ring)])
                reach = reach[~taken[reach]]
                if len(reach) == 0:
                    break
                ring = reach[conforms(np.full(len(reach), start, dtype="i8"), reach)]
                taken[ring] = True
                grown.append(ring)
            if len(grown) > 1:
                groups.append(np.sort(np.concatenate(grown)))
    return groups


def _mergeable(mesh: Mesh, groups: list[np.ndarray]) -> list[np.ndarray]:
    """The groups ``merge_groups`` would take, each judged on its own.

    ``merge_groups`` refuses a whole call when any one outline is an annulus, a
    bowtie or past a ring ceiling, which would leave an entire mesh unsimplified
    for the sake of one awkward patch. The same checks are run per group here
    (using that module's own helpers, so the two cannot drift) and the offender
    alone is left as it was.
    """
    borders = topo.region_boundary_corners_by_group(mesh, groups)
    kept: list[np.ndarray] = []
    rings: list[np.ndarray] = []
    for grp, border in zip(groups, borders, strict=True):
        try:
            ring = ops_dissolve._ring_corners(mesh, grp, border)
        except OpError:
            continue
        if len(ring) > ops_dissolve.MAX_DISSOLVED_RING:
            continue
        kept.append(grp)
        rings.append(ring)
    if not kept:
        return []

    # The concave ceilings are shared across the call (a per-ring cap and a
    # budget of summed squares), so they are charged here in the same order.
    verts = [mesh.loops[r].astype("i8") for r in rings]
    counts = np.array([len(r) for r in verts], dtype="i8")
    ids = np.concatenate(verts)
    starts = np.concatenate([[0], np.cumsum(counts)])
    nxt = np.arange(1, len(ids) + 1, dtype="i8")
    nxt[starts[1:] - 1] = starts[:-1]
    from .mesh import _newell

    normals = _newell(mesh.positions, ids, ids[nxt], starts[:-1])
    concave = earclip.concave_faces(mesh.positions, ids, starts, normals)
    budget = ops_dissolve.MAX_CONCAVE_DISSOLVE_RING**2
    drop = np.zeros(len(kept), dtype=bool)
    for k in np.flatnonzero(concave).tolist():
        n = int(counts[k])
        if n > ops_dissolve.MAX_CONCAVE_DISSOLVE_RING or budget - n * n < 0:
            drop[k] = True
            continue
        budget -= n * n
    return [grp for grp, gone in zip(kept, drop.tolist(), strict=True) if not gone]


def _merge(mesh: Mesh, groups: list[np.ndarray]) -> Mesh:
    try:
        return ops_dissolve.merge_groups(mesh, groups)[0]
    except OpError:
        pass
    # One awkward outline refused the call; retry with only the groups that
    # would each be accepted on their own.
    kept = _mergeable(mesh, groups)
    if not kept:
        return mesh
    try:
        return ops_dissolve.merge_groups(mesh, kept)[0]
    except OpError:
        # _mergeable charges the same ceilings merge_groups does, so this is the
        # arm nothing reaches; leaving the faces alone is the safe reading of it.
        return mesh


def _dissolve_straight(mesh: Mesh) -> Mesh:
    """Drop every vertex that is straight in each face using it.

    UV **preserved**: a surviving corner keeps its own UV, and a dropped vertex
    is only dropped where its UV lay on the line between its neighbours'.
    """
    diag = _extent(mesh)
    if diag <= 0.0:
        return mesh
    tol = _FLAT_TOL * diag
    out = mesh
    # Per corner, an upper bound on how far the edge leaving it (to the next
    # corner) lies from the original vertices it now stands in for. Zero for an
    # edge nothing was ever dropped from. The 2026-10-11 audit's clay-27: each
    # round was tested only against the polyline the previous round left, so the
    # error added up -- a fine flat disc fan ended as a 173-181-gon whose sagitta
    # was 2 to 2.5 times the tolerance. A dropped corner's edge pair is replaced
    # by the chord, which is within ``off`` of both old edges' far end and, by
    # convexity of distance to a segment, of every point between; so the chord's
    # bound is the larger of the two old bounds plus ``off``.
    dev = np.zeros(len(out.loops), dtype="f8")
    for _ in range(_MAX_ROUNDS):
        n_verts = len(out.positions)
        loops = out.loops.astype("i8")
        counts = np.diff(out.starts.astype("i8"))
        _, nxt, prv = topo.flat_next(counts)
        face_of = np.repeat(np.arange(len(counts), dtype="i8"), counts)
        pos = out.positions.astype("f8")
        chord = pos[loops[nxt]] - pos[loops[prv]]
        rel = pos[loops] - pos[loops[prv]]
        chord2 = np.einsum("ij,ij->i", chord, chord)
        usable = chord2 > 0.0
        along = np.divide(
            np.einsum("ij,ij->i", rel, chord), chord2, out=np.zeros_like(chord2), where=usable
        )
        off = rel - along[:, None] * chord
        straight = usable & (along > 1e-9) & (along < 1.0 - 1e-9)
        chord_dev = np.maximum(dev[prv], dev) + np.sqrt(np.einsum("ij,ij->i", off, off))
        straight &= chord_dev <= tol
        if out.uv is not None:
            uv = out.uv.astype("f8")
            predicted = uv[prv] + along[:, None] * (uv[nxt] - uv[prv])
            straight &= (np.abs(uv - predicted) <= _UV_TOL).all(axis=1)

        # Removing a corner from a triangle would leave a line, so a vertex any
        # triangle uses stays.
        blocked = ~straight | (counts[face_of] <= 3)
        used = np.bincount(loops, minlength=n_verts) > 0
        candidate = used & (np.bincount(loops, weights=blocked, minlength=n_verts) == 0)
        if not candidate.any():
            break

        priority = np.random.default_rng(0).permutation(n_verts)
        here, there = loops, loops[nxt]
        both = candidate[here] & candidate[there]
        beaten = np.full(n_verts, n_verts, dtype="i8")
        np.minimum.at(beaten, here[both], priority[there[both]])
        np.minimum.at(beaten, there[both], priority[here[both]])
        chosen = candidate & (priority < beaten)

        # Two opposite corners of a quad are not neighbours, yet dropping both
        # leaves two. Any face that would fall under three keeps all of them.
        gone = chosen[loops]
        left = counts - np.bincount(face_of[gone], minlength=len(counts))
        short = left < 3
        if short.any():
            chosen[np.unique(loops[gone & short[face_of]])] = False
            gone = chosen[loops]
        if not gone.any():
            break

        keep = ~gone
        # The corner before a dropped one now runs straight to the one after it.
        # It survives: a dropped corner never has a dropped neighbour.
        dev = dev.copy()
        dev[prv[gone]] = chord_dev[gone]
        dev = dev[keep]
        new_counts = np.bincount(face_of[keep], minlength=len(counts))
        out, _ = topo.compact_vertices(
            topo.rebuild(
                out.positions,
                loops[keep],
                topo.starts_from_counts(new_counts),
                out.material,
                out.smooth,
                uv=None if out.uv is None else out.uv[keep],
            )
        )
    return out


def _flat_merge(mesh: Mesh) -> Mesh:
    groups = _flat_groups(mesh)
    merged = _merge(mesh, groups) if groups else mesh
    return _dissolve_straight(merged)


# --- stage 2: edge collapse -------------------------------------------------


def _locked_vertices(mesh: Mesh, a) -> np.ndarray:
    """Vertices an edge collapse may never move or remove (see the docstring)."""
    n_verts = len(mesh.positions)
    loops = mesh.loops.astype("i8")
    nxt = a.next_corner.astype("i8")
    corner = np.arange(len(loops), dtype="i8")
    twin = a.twin.astype("i8")
    across = np.where(twin >= 0, twin, corner)
    face = a.corner_face.astype("i8")

    # Every edge that is not an ordinary two-face, consistently wound one, or
    # that separates two materials, two shading modes or two UV patches.
    bad = twin < 0
    bad |= mesh.material[face] != mesh.material[face[across]]
    bad |= mesh.smooth[face] != mesh.smooth[face[across]]
    if mesh.uv is not None:
        uv = mesh.uv.astype("f8")
        bad |= (np.abs(uv - uv[nxt[across]]) > _UV_TOL).any(axis=1)
        bad |= (np.abs(uv[nxt] - uv[across]) > _UV_TOL).any(axis=1)
    locked = np.zeros(n_verts, dtype=bool)
    locked[loops[bad]] = True
    locked[loops[nxt[bad]]] = True

    if mesh.uv is not None:
        # An edge test cannot see a vertex whose two fans disagree about its UV
        # without a seam edge between them; the corners themselves can.
        low = np.full((n_verts, 2), np.inf)
        high = np.full((n_verts, 2), -np.inf)
        np.minimum.at(low, loops, uv)
        np.maximum.at(high, loops, uv)
        locked |= ((high - low) > _UV_TOL).any(axis=1) & np.isfinite(low).all(axis=1)

    # A pinch: the corners at one vertex fall into more than one fan, which
    # crossing from a corner to the next face's corner at the same vertex
    # reveals. Collapsing an edge at a pinch would glue two fans into one.
    has_twin = np.flatnonzero(twin >= 0)
    if len(has_twin):
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        link = nxt[twin[has_twin]]
        graph = coo_matrix(
            (np.ones(len(has_twin), dtype="i1"), (has_twin, link)),
            shape=(len(loops), len(loops)),
        )
        n_fans, fan = connected_components(graph, directed=False)
        keys = np.unique(loops * n_fans + fan)
        locked |= np.bincount(keys // n_fans, minlength=n_verts) > 1
    return locked


def _vertex_quadrics(pos: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """``(V, 10)`` area-weighted plane quadrics, upper triangle row-major."""
    p0, p1, p2 = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    cross = np.cross(p1 - p0, p2 - p0)
    area2 = np.linalg.norm(cross, axis=1)
    unit = np.divide(cross, area2[:, None], out=np.zeros_like(cross), where=area2[:, None] > 0)
    plane = np.concatenate([unit, -np.einsum("ij,ij->i", unit, p0)[:, None]], axis=1)
    weight = area2 * 0.5
    cols = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]
    quad = np.zeros((len(pos), 10), dtype="f8")
    corners = tri.reshape(-1)
    for k, (i, j) in enumerate(cols):
        per_face = plane[:, i] * plane[:, j] * weight
        quad[:, k] = np.bincount(corners, weights=np.repeat(per_face, 3), minlength=len(pos))
    return quad


def _error(q: list[float], x: tuple[float, float, float] | list[float]) -> float:
    px, py, pz = x
    return (
        q[0] * px * px
        + 2.0 * q[1] * px * py
        + 2.0 * q[2] * px * pz
        + 2.0 * q[3] * px
        + q[4] * py * py
        + 2.0 * q[5] * py * pz
        + 2.0 * q[6] * py
        + q[7] * pz * pz
        + 2.0 * q[8] * pz
        + q[9]
    )


def _bilinear(q: list[float], u: list[float], v: list[float]) -> float:
    """``u^T Q v`` for homogeneous 4-vectors, Q symmetric in its packed form."""
    return (
        u[0] * (q[0] * v[0] + q[1] * v[1] + q[2] * v[2] + q[3] * v[3])
        + u[1] * (q[1] * v[0] + q[4] * v[1] + q[5] * v[2] + q[6] * v[3])
        + u[2] * (q[2] * v[0] + q[5] * v[1] + q[7] * v[2] + q[8] * v[3])
        + u[3] * (q[3] * v[0] + q[6] * v[1] + q[8] * v[2] + q[9] * v[3])
    )


def _best_on_edge(q: list[float], pu: list[float], pw: list[float]) -> list[float]:
    """The cheapest point on the segment ``pu -> pw`` under quadric *q*.

    **On the edge, not the quadric's free optimum.** The unconstrained minimiser
    slides a vertex sideways across the surface, and a UV that is only
    interpolated along the edge then disagrees with where the vertex went: on a
    gently curved textured heightfield the largest texture error was 0.13 of the
    whole map against 0.007 once the vertex was held to its edge. It also removes
    the one place a nearly flat quadric made the answer a wild guess.
    """
    start = [pu[0], pu[1], pu[2], 1.0]
    along = [pw[0] - pu[0], pw[1] - pu[1], pw[2] - pu[2], 0.0]
    curve = _bilinear(q, along, along)
    ts = [0.5, 0.0, 1.0]
    if curve > 1e-18:
        ts.insert(0, min(1.0, max(0.0, -_bilinear(q, start, along) / curve)))
    best_t, best_e = ts[0], None
    for t in ts:
        point = [pu[0] + t * along[0], pu[1] + t * along[1], pu[2] + t * along[2]]
        e = _error(q, point)
        if best_e is None or e < best_e - 1e-15:
            best_t, best_e = t, e
    return [pu[0] + best_t * along[0], pu[1] + best_t * along[1], pu[2] + best_t * along[2]]


def _cross(p: list[float], q: list[float], r: list[float]) -> tuple[float, float, float]:
    ax, ay, az = q[0] - p[0], q[1] - p[1], q[2] - p[2]
    bx, by, bz = r[0] - p[0], r[1] - p[1], r[2] - p[2]
    return (ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx)


def _shells(mesh: Mesh, a) -> tuple[list[int], list[int]]:
    """Per face, its shell's label, and each shell's face count.

    A shell is a set of faces joined through twinned edges. ``mesh`` is all
    triangles here, so a face is a third of its corners.
    """
    n_faces = face_count(mesh)
    twin = a.twin.astype("i8")
    face = a.corner_face.astype("i8")
    has_twin = np.flatnonzero(twin >= 0)
    if len(has_twin) == 0:
        return list(range(n_faces)), [1] * n_faces
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    graph = coo_matrix(
        (np.ones(len(has_twin), dtype="i1"), (face[has_twin], face[twin[has_twin]])),
        shape=(n_faces, n_faces),
    )
    n_shells, label = connected_components(graph, directed=False)
    return label.tolist(), np.bincount(label, minlength=n_shells).tolist()


def _collapse(mesh: Mesh, target: int) -> Mesh | None:
    """Triangulate *mesh* and collapse edges toward *target* faces.

    Returns ``None`` when the mesh is unusable for it (nothing to read), else the
    triangle mesh after as many legal collapses as reached the target.
    """
    work = mesh
    if bool((np.diff(mesh.starts) != 3).any()):
        work, _ = ops_topo.triangulate_faces(mesh, empty())
    if face_count(work) == 0:
        return None
    a = adjacency(work)
    has_uv = work.uv is not None
    locked_np = _locked_vertices(work, a)
    # The four-face floor is per shell, not for the mesh (the 2026-10-11 audit's
    # clay-03): it was ``faces_left - 2 < MIN_SIMPLIFY_TARGET`` on the total, so
    # an 80-face and a 32-face sphere at target 4 were both collapsed to two
    # faces -- a zero-volume pancake each, under a refusal sentence that promises
    # "anything fewer cannot enclose a solid". A shell is an edge-connected set of
    # faces; two shells that only touch at a vertex are separate here, and that
    # vertex is a pinch, which stays locked.
    shell_of, shell_faces = _shells(work, a)
    pos_np = work.positions.astype("f8")
    tri_np = work.loops.reshape(-1, 3).astype("i8")

    pos = pos_np.tolist()
    tv = tri_np.tolist()
    tuv = work.uv.astype("f8").reshape(-1, 3, 2).tolist() if has_uv else None
    quad = _vertex_quadrics(pos_np, tri_np).tolist()
    locked = locked_np.tolist()
    n_faces = len(tv)
    vf: list[set[int]] = [set() for _ in pos]
    for t, row in enumerate(tv):
        for v in row:
            vf[v].add(t)
    alive_t = [True] * n_faces
    alive_v = [True] * len(pos)
    stamp = [0] * len(pos)
    faces_left = n_faces
    scale2 = max(_extent(work) ** 2, 1e-30)
    counter = itertools.count()
    heap: list = []

    def push(u: int, w: int) -> None:
        if locked[u] and locked[w]:
            return
        q = [x + y for x, y in zip(quad[u], quad[w], strict=True)]
        if locked[u]:
            best = pos[u]
        elif locked[w]:
            best = pos[w]
        else:
            best = _best_on_edge(q, pos[u], pos[w])
        heapq.heappush(heap, (_error(q, best), next(counter), u, w, stamp[u], stamp[w], list(best)))

    def uv_of(face: int, vertex: int) -> list[float]:
        return tuv[face][tv[face].index(vertex)]

    def attempt(u: int, w: int, x: list[float]) -> bool:
        nonlocal faces_left
        fu, fw = vf[u], vf[w]
        shared = fu & fw
        if len(shared) != 2:
            return False
        shell = shell_of[next(iter(shared))]
        if shell_faces[shell] - 2 < MIN_SIMPLIFY_TARGET:
            return False
        nu = {v for f in fu for v in tv[f]}
        nw = {v for f in fw for v in tv[f]}
        nu.discard(u)
        nw.discard(w)
        opposite = {v for f in shared for v in tv[f] if v != u and v != w}
        if (nu & nw) != opposite:
            return False

        # The locked endpoint stays; with neither locked the first one stays and
        # takes the optimum. ``gone`` is what is removed.
        if locked[w] and not locked[u]:
            keeper, gone = w, u
        else:
            keeper, gone = u, w
        pin = locked[keeper]
        if pin:
            x = pos[keeper]
        changed = (vf[gone] if pin else fu | fw) - shared
        moved = (u, w) if not pin else (gone,)

        new_uv = None
        if has_uv:
            shared_face = next(iter(shared))
            if pin:
                new_uv = uv_of(shared_face, keeper)
            else:
                pu, pw = pos[u], pos[w]
                edge = [pw[i] - pu[i] for i in range(3)]
                length2 = sum(e * e for e in edge)
                t = 0.0
                if length2 > 0.0:
                    t = sum((x[i] - pu[i]) * edge[i] for i in range(3)) / length2
                    t = min(1.0, max(0.0, t))
                ua, ub = uv_of(shared_face, u), uv_of(shared_face, w)
                new_uv = [ua[0] + t * (ub[0] - ua[0]), ua[1] + t * (ub[1] - ua[1])]

        for f in changed:
            row = tv[f]
            old = _cross(pos[row[0]], pos[row[1]], pos[row[2]])
            moved_pts = [x if v in moved else pos[v] for v in row]
            new = _cross(moved_pts[0], moved_pts[1], moved_pts[2])
            old_len = (old[0] ** 2 + old[1] ** 2 + old[2] ** 2) ** 0.5
            new_len = (new[0] ** 2 + new[1] ** 2 + new[2] ** 2) ** 0.5
            if old_len > 1e-9 * scale2:  # a sliver already has no normal to flip
                dot = old[0] * new[0] + old[1] * new[1] + old[2] * new[2]
                if new_len < _MIN_AREA_RATIO * old_len or dot <= _FLIP_COS * old_len * new_len:
                    return False
            if has_uv:
                uvs = tuv[f]
                before = (uvs[1][0] - uvs[0][0]) * (uvs[2][1] - uvs[0][1]) - (
                    uvs[1][1] - uvs[0][1]
                ) * (uvs[2][0] - uvs[0][0])
                after_uv = [new_uv if v in moved else uvs[i] for i, v in enumerate(row)]
                after = (after_uv[1][0] - after_uv[0][0]) * (after_uv[2][1] - after_uv[0][1]) - (
                    after_uv[1][1] - after_uv[0][1]
                ) * (after_uv[2][0] - after_uv[0][0])
                if abs(before) > 1e-12 and before * after <= 0.0:
                    return False

        for f in shared:
            alive_t[f] = False
            for v in tv[f]:
                vf[v].discard(f)
        faces_left -= 2
        shell_faces[shell] -= 2
        for f in changed:
            row = tv[f]
            for i, v in enumerate(row):
                if v in moved:
                    row[i] = keeper
                    if has_uv:
                        tuv[f][i] = new_uv
            vf[keeper].add(f)
        alive_v[gone] = False
        vf[gone] = set()
        if not pin:
            pos[keeper] = list(x)
        quad[keeper] = [p + q for p, q in zip(quad[keeper], quad[gone], strict=True)]
        locked[keeper] = locked[u] or locked[w]
        stamp[keeper] += 1
        stamp[gone] += 1
        for other in {v for f in vf[keeper] for v in tv[f]} - {keeper}:
            push(keeper, other)
        return True

    progressed_any = False
    while faces_left > target:
        heap.clear()
        seen: set[tuple[int, int]] = set()
        for t, row in enumerate(tv):
            if not alive_t[t]:
                continue
            for i in range(3):
                lo, hi = sorted((row[i], row[(i + 1) % 3]))
                if (lo, hi) not in seen:
                    seen.add((lo, hi))
                    push(lo, hi)
        progressed = False
        while heap and faces_left > target:
            _, _, u, w, su, sw, x = heapq.heappop(heap)
            if not (alive_v[u] and alive_v[w]) or stamp[u] != su or stamp[w] != sw:
                continue
            if attempt(u, w, x):
                progressed = progressed_any = True
        if not progressed:
            break
    if not progressed_any:
        return None

    live = [t for t in range(n_faces) if alive_t[t]]
    out = topo.rebuild(
        np.array(pos, dtype="f8"),
        np.array([tv[t] for t in live], dtype="i8"),
        topo.starts_from_counts(np.full(len(live), 3, dtype="i8")),
        work.material[live],
        work.smooth[live],
        uv=np.array([tuv[t] for t in live], dtype="f8").reshape(-1, 2) if has_uv else None,
    )
    return topo.compact_vertices(out)[0]
