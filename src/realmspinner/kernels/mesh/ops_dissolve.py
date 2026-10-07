"""Dissolve: merging the faces a selection joins into one.

Delete and dissolve are different operations and the difference is the whole
module. Deleting faces leaves a hole; dissolving a block of them merges it into
one n-gon and leaves the surface intact (Clay's Merge Faces). The op is one
shape: **find the faces the selection joins, work out the outline of each
connected group, and replace the group with a single n-gon wound along that
outline.** The group-to-n-gon step lives in :func:`merge_groups`.

**This is where refusing matters.** The core walks a group's border
head-to-tail, and a walk has no defined next step when the border forks or
splits, so rather than guessing it names the element and stops:

* An **annulus** -- a group whose border is two separate rings, which happens
  when the selection encircles a face it does not include. A face with a hole in
  it is not representable in CSR at all, and the alternatives (leave the island,
  bridge it with a slit) are both worse than saying so.
* A **bowtie ring** -- a border that visits one vertex twice. The n-gon would
  self-intersect.
* A **non-manifold edge**, anywhere: three faces meet, so "the other side" is
  not a single face.

**Results are routinely concave**, and that is the first real consumer of
:mod:`.earclip` -- merging the two triangles of an L gives a polygon a fan
would triangulate outside itself.

**A stated limitation:** the two ends of a merged seam stay in the n-gon as
two-valence collinear corners. They are harmless (the Newell normal is stable
across them, and :mod:`.earclip` tolerates them).

UV **preserved**: a border corner keeps its own uv and an interior corner is
dropped along with the geometry it described.
"""

from __future__ import annotations

import numpy as np

from . import earclip, topo
from .adjacency import adjacency
from .elements import ElementSel, OpError
from .mesh import Mesh, _newell, face_count

__all__ = ["dissolve_faces", "merge_groups"]


def _group_by_label(labels: np.ndarray, subset: np.ndarray) -> list[np.ndarray]:
    """Group an ascending *subset* of face indices by ``labels[subset]``.

    This module used to group faces with a hand-rolled Python union-find
    (``_Union``, removed 2026-09-17). Its ``groups()`` scanned *subset*
    ascending and inserted each face under its root's list the first time that
    root was seen, so the groups came back **ordered by their smallest
    member**, each group's own members **ascending** (insertion order equalled
    scan order). ``_Union`` itself had already been narrowed once, on
    2026-09-08 (clay-08): it originally enumerated ``range(len(self.parent))``
    -- every face in the whole mesh, not the selection -- so dissolving one
    edge on a 408,321-face mesh cost 654 ms of Python ``find()`` calls the
    selection never asked for. Narrowing to *subset* fixed that, but the
    ``union()`` loop feeding it was still one Python call per interior corner,
    which the 2026-09-17 native-kernel review (batch 11) measured at 665 ms on
    a 200k-face select-all dissolve, 1.79M ``find()`` calls. Replaced with
    ``scipy.sparse.csgraph.connected_components`` (already used three lines
    away in ``ops_topo.py`` and in ``select.py``) for the
    union-find itself, and this function for the grouping step, which is
    exactly the ``_Union.groups`` contract above but vectorised: no
    ``.tolist()`` loop, no dict.
    """
    if len(subset) == 0:
        return []
    sub_labels = labels[subset]
    # Compact each label to the rank of its *first appearance* while scanning
    # subset ascending -- the same order ``_Union.groups``'s dict-insertion
    # produced, since every caller here passes an ascending, deduplicated
    # subset.
    _, first_index, inverse = np.unique(sub_labels, return_index=True, return_inverse=True)
    rank = np.empty(len(first_index), dtype="i8")
    rank[np.argsort(first_index, kind="stable")] = np.arange(len(first_index), dtype="i8")
    order_id = rank[inverse.reshape(-1)]
    order = np.argsort(order_id, kind="stable")
    sorted_subset = subset[order]
    sorted_id = order_id[order]
    splits = np.flatnonzero(np.diff(sorted_id)) + 1
    return list(np.split(sorted_subset, splits))


#: The largest outline a dissolve will produce. ``ops_subdiv`` refuses past
#: ``MAX_SUBDIVIDED_FACES`` and states why; this is the same argument for the
#: other unbounded growth an edit can ask for. The n-gon a dissolve makes is
#: handed to ``earclip``, whose ear search is worst-case quadratic in the
#: corner count, in Python, one triangle removed per scan of the remainder --
#: and ``clay_ops.run_mesh_op`` calls it synchronously from the key handler,
#: which is the frame thread. Past this the app stops responding rather than
#: becoming slower, which is the outcome every ceiling in this package exists
#: to prevent.
#:
#: Twenty thousand keeps that search well under a second. No hand-made
#: selection approaches it: a dissolve of a whole subdivided face loop on a
#: dense mesh is a few thousand corners.
MAX_DISSOLVED_RING = 20_000


def _refuse_ring(rings: list[np.ndarray]) -> None:
    """Refuse before the merge when the outline is too big to triangulate."""
    worst = max((len(r) for r in rings), default=0)
    if worst > MAX_DISSOLVED_RING:
        raise OpError(
            f"That merge would make a face with {worst:,} corners, past the "
            f"{MAX_DISSOLVED_RING:,} Clay can triangulate without stalling. "
            "Dissolve a smaller region."
        )


#: The largest *concave* outline a merge will attempt to triangulate, well
#: under MAX_DISSOLVED_RING itself.
#:
#: A first version of this fix put a size ceiling inside earclip's own
#: ``corner_triangles``, past which a concave face silently kept the plain
#: fan `fan_corners` already produced instead of running the ear search --
#: bounded, but wrong in a new way: a fan across a reflex corner puts a
#: triangle outside the polygon (earclip's own module docstring), and
#: ``adjacency.check_manifold`` reads only CSR topology, which a wrong
#: triangulation never changes, so nothing downstream could see that a
#: well-formed, resolvable concave face -- one earclip would have
#: triangulated correctly, just slowly -- had silently gotten a wedge that is
#: not there. earclip's own "rendering never raises" tolerance is real and
#: load-bearing (an exception from inside a draw takes down the frame loop),
#: so that fallback is correct for a search that is genuinely stuck on a
#: degenerate ring (self-intersecting, zero-area, all-collinear) -- but a
#: ring that is merely *large* is not degenerate, and silently mistriangulating
#: it is worse than refusing the edit that made it.
#:
#: So the guard moved up here instead, where refusing past a ceiling is
#: already this op's own behaviour (see MAX_DISSOLVED_RING/_refuse_ring just
#: above): a concave ring past this bound is refused by name before the merge
#: commits it to a face earclip would have to search. earclip itself is
#: unchanged and stays exactly as tolerant as it always was.
#:
#: The 2026-09-11 audit's clay-02 measured earclip's ear search directly on a
#: realistic concave (zigzag/comb) ring -- exactly the shape this module's own
#: docstring says a dissolve routinely produces -- at 897 ms at 1,600 corners
#: and 3.63 s at 3,200: a clean quadratic trend that extrapolates to roughly
#: 140 seconds at MAX_DISSOLVED_RING's own 20,000-corner ceiling, not the
#: "well under a second" that comment claims (it assumes a roughly linear
#: cost, which holds for a convex or lightly-concave ring but not for one
#: that is concave enough to force the full O(n^2) search). The measured
#: quadratic rate (897 ms / 1,600^2) puts one thousand corners at roughly
#: 350 ms -- well under a second, with margin under the ~1,700-corner point
#: where that stops being true.
MAX_CONCAVE_DISSOLVE_RING = 1_000


def _refuse_concave_ring(mesh: Mesh, vertex_rings: list[np.ndarray]) -> None:
    """Refuse a concave ring past MAX_CONCAVE_DISSOLVE_RING, before the merge
    commits it to a face earclip's O(n^2) ear search would have to walk --
    and refuse the *sum* of many rings that individually stay under it too.

    Each ring in *vertex_rings* is a face's worth of *vertex* ids in winding
    order -- the same shape ``mesh.loops`` stores, and what a caller with a
    corner-index ring (:func:`_ring_corners`) gets by indexing through
    ``mesh.loops`` before calling this, exactly as it indexes through
    ``mesh.loops`` to build the real merged face.

    Shared with :func:`~.ops_topo.fill_hole`, whose cap is the identical
    "one n-gon, triangulated by earclip on the frame thread" shape -- a
    hole's boundary has no more guarantee of convexity than a dissolved
    region's does, so the same concave ring can appear there too. Both
    callers can hand this a *list* of several rings from one call (several
    disjoint dissolve groups, or several pinched holes filled at once), which
    is exactly the shape the 2026-09-26 audit's clay-mesh-ops-05 found
    unbounded: a ring at or under the per-ring ceiling used to skip the
    concavity check entirely, so many separate concave rings, each
    individually just under it, passed one after another and paid their
    summed O(n^2) earclip cost with nothing to refuse it. Earclip's cost is
    quadratic in a ring's own corner count (see this constant's own
    measurement), so the fair total is a sum of corner-count *squares*,
    bounded to what one single ring at the ceiling would have cost --
    keeping one call's total earclip time under the same bar the per-ring
    ceiling was already set for, however the corners are distributed across
    rings.
    """
    budget = MAX_CONCAVE_DISSOLVE_RING * MAX_CONCAVE_DISSOLVE_RING
    for ring in vertex_rings:
        n = len(ring)
        if n < 3:
            continue
        # The 2026-10-07 audit's clay-26: this used to build a throwaway
        # single-face ``Mesh`` over ``mesh.positions`` to ask the question, and
        # ``Mesh.__post_init__`` copies every array it is given, so each ring
        # paid a copy of the whole position array -- rings x mesh vertices,
        # 5.3 s for 2,000 rings on an 832k-vertex mesh, and Fill Hole's call
        # here 6.4 s against the 0.83 s written beside its ceiling. The ring's
        # own points are all the question reads, so they are gathered once and
        # the same two kernels (``_newell``, ``concave_faces``) run over the
        # ring-local array: the answer is the same, the cost is the ring's.
        ring_ids = np.asarray(ring, dtype="i8")
        local = np.arange(n, dtype="i8")
        points = mesh.positions[ring_ids]
        normals = _newell(points, local, np.roll(local, -1), np.zeros(1, dtype="i8"))
        is_concave = earclip.concave_faces(
            points, local, np.array([0, n], dtype="i8"), normals
        )[0]
        if not is_concave:
            continue
        if n > MAX_CONCAVE_DISSOLVE_RING:
            raise OpError(
                f"That merge would make a concave face with {n:,} corners, "
                f"too complex for Clay to triangulate without stalling -- past the "
                f"{MAX_CONCAVE_DISSOLVE_RING:,} corners a concave merge can have. "
                "Dissolve a smaller region."
            )
        budget -= n * n
        if budget < 0:
            raise OpError(
                "This merge's concave faces would together cost as much to "
                f"triangulate as one {MAX_CONCAVE_DISSOLVE_RING:,}-corner "
                "concave face, past what Clay can do without stalling. "
                "Dissolve fewer regions in one go."
            )


def _ring_corners(mesh: Mesh, group: np.ndarray, border: np.ndarray | None = None) -> np.ndarray:
    """The group's outline as an ordered array of corner indices.

    Ordered by chaining the border's directed edges head to tail, which is what
    makes the resulting n-gon wound consistently with everything still around
    it: each border corner reads ``a -> b`` because its own face does, and the
    merged face inherits exactly those traversals.

    *border* is the group's border corners when the caller has already derived
    them for every group in one pass (:func:`merge_groups` does, see its
    comment); left out, they are derived here for this one group.
    """
    if border is None:
        border = topo.region_boundary_corners(mesh, group)
    if len(border) == 0:
        raise OpError(
            "Those faces make up a closed surface on their own, so there is "
            "nothing left to merge them into."
        )
    a = adjacency(mesh)
    heads = mesh.loops[border].astype("i8")
    succ: dict[int, int] = {}
    for corner, head in zip(border.tolist(), heads.tolist(), strict=True):
        if head in succ:
            raise OpError(
                f"Vertex {head} appears twice on the outline of that selection, "
                "so it cannot be merged into one face. Dissolve a smaller "
                "region."
            )
        succ[head] = corner

    start = int(border[0])
    ring = [start]
    cursor = int(mesh.loops[a.next_corner[start]])
    while cursor != int(mesh.loops[start]):
        nxt = succ.get(cursor)
        if nxt is None:  # pragma: no cover - a fork is caught above
            raise OpError(
                f"The outline of that selection breaks at vertex {cursor}, so it "
                "cannot be merged into one face. Dissolve a smaller region."
            )
        ring.append(nxt)
        cursor = int(mesh.loops[a.next_corner[nxt]])

    if len(ring) != len(border):
        raise OpError(
            "That selection surrounds a face it does not include, so merging it "
            "would need a face with a hole in it. Include the middle, or "
            "dissolve a smaller region."
        )
    return np.array(ring, dtype="i8")


def merge_groups(mesh: Mesh, groups: list[np.ndarray]) -> tuple[Mesh, ElementSel]:
    """Replace each group of faces with one n-gon along its outline.

    Groups of a single face are dropped rather than rebuilt: there is nothing to
    merge, and rebuilding would move the face to the end of the list for no
    reason.

    Refused past :data:`MAX_DISSOLVED_RING`, for the reason ``ops_subdiv``
    refuses past ``MAX_SUBDIVIDED_FACES`` -- and, separately, refused past
    :data:`MAX_CONCAVE_DISSOLVE_RING` when the outline is also concave, since
    that is what actually drives earclip's triangulation cost past this size,
    not corner count alone (see that constant's own comment).
    """
    real = [np.asarray(g, dtype="i8") for g in groups if len(g) > 1]
    if not real:
        raise OpError(
            "Nothing there to dissolve: a dissolve merges neighbours, so it "
            "needs at least two of them touching."
        )

    # The 2026-10-03 audit's clay-50: each group used to ask
    # `region_boundary_corners` for its own border, and every call does
    # whole-mesh work, so G groups on a mesh of C corners cost G x C -- 2.9 s
    # for 4,489 two-face groups on a 40k-vertex mesh, 49 s for 17,689 on 160k.
    # The borders of every group now come from one vectorised pass.
    borders = topo.region_boundary_corners_by_group(mesh, real)
    rings = [_ring_corners(mesh, g, b) for g, b in zip(real, borders, strict=True)]
    _refuse_ring(rings)
    _refuse_concave_ring(mesh, [mesh.loops[r] for r in rings])
    consumed = np.concatenate(real)
    keep = np.ones(face_count(mesh), dtype=bool)
    keep[consumed] = False
    kept = np.flatnonzero(keep)

    counts = np.array([len(r) for r in rings], dtype="i8")
    corners = np.concatenate(rings)
    base = topo.take_faces(mesh, kept)

    out, _ = topo.compact_vertices(
        topo.rebuild(
            mesh.positions,
            np.concatenate([base.loops.astype("i8"), mesh.loops[corners].astype("i8")]),
            np.concatenate([base.starts.astype("i8"), int(base.starts[-1]) + np.cumsum(counts)]),
            np.concatenate([base.material, [mesh.material[g[0]] for g in real]]),
            np.concatenate([base.smooth, [mesh.smooth[g[0]] for g in real]]),
            uv=None if mesh.uv is None else np.concatenate([base.uv, mesh.uv[corners]]),
        )
    )
    n_kept = len(kept)
    return out, ElementSel(faces=np.arange(n_kept, n_kept + len(rings)))


def dissolve_faces(mesh: Mesh, sel: ElementSel) -> tuple[Mesh, ElementSel]:
    """Merge each connected block of selected faces into one face."""
    if len(sel.faces) == 0:
        raise OpError("Select the faces to dissolve into one.")
    a = adjacency(mesh)
    n_faces = face_count(mesh)
    chosen = np.zeros(n_faces, dtype=bool)
    chosen[sel.faces] = True

    interior = np.flatnonzero(chosen[a.corner_face] & (a.twin >= 0))
    other = a.corner_face[a.twin[interior]].astype("i8")
    mask = chosen[other]
    fa = a.corner_face[interior[mask]].astype("i8")
    fb = other[mask]

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    graph = coo_matrix((np.ones(len(fa), dtype="i1"), (fa, fb)), shape=(n_faces, n_faces))
    labels = connected_components(graph, directed=False)[1]

    # Every edge above joins two ``chosen`` faces, so the selection itself is
    # the exact set worth grouping -- see ``_group_by_label``.
    groups = _group_by_label(labels, np.flatnonzero(chosen))
    return merge_groups(mesh, groups)
