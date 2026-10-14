"""Edge slide: moving a selected edge loop along the strip it sits in.

A slide changes positions and nothing else -- no topology, no uv, no material --
which is what makes it the cheap cousin of :mod:`.ops_bevel`'s loop cut: cut a
loop in, then slide it to where the form wants it. Clay's ``edge-slide`` op
(Edges, ``Ctrl+Shift+R``) wraps :func:`edge_slide`.

The kernel was cut with the Blender-shaped long tail in 112c69e2 (in
``ops_model.py``, which does not come back: it also held bisect, knife, rip, poke
and the rest) and restored here from ``112c69e2^`` on 2026-10-13, Tranche 1
landing 4 of the Clay rig plan. ``vertex_slide``, its sibling, stays cut.
"""

from __future__ import annotations

import numpy as np

from . import topo
from .adjacency import Adjacency, adjacency
from .elements import ElementSel, OpError
from .mesh import Mesh

__all__ = ["MAX_SLIDE_RAILS", "MAX_SLIDE_VERTICES", "edge_slide"]


def _unit(v: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(v))
    return v / length if length > 1e-12 else v


#: The largest number of touched vertices :func:`edge_slide` and
#: :func:`vertex_slide` will walk in their own per-vertex Python loop. The
#: 2026-09-19 audit's clay-15 found neither op had any ceiling at all --
#: unlike :func:`bisect`, :func:`rip`, :func:`tris_to_quads` and
#: :func:`grid_fill` in this same module, every one of which measures and
#: refuses first -- and a whole-mesh selection reaches either op with no seed
#: (Select All, then Edge Slide or Vertex Slide, straight off the menu or the
#: agent surface). Measured on this machine, a ``uv_sphere`` at increasing
#: resolution -- every vertex slid for :func:`vertex_slide`, every latitude
#: ring's own edges slid for :func:`edge_slide` (so every touched vertex keeps
#: its ordinary two up/down rails rather than tripping the O(k^2) pole
#: disambiguation below, the same way a real quad-strip selection would):
#:
#: | vertices | vertex_slide | edge_slide |
#: |---------:|-------------:|-----------:|
#: |    4,034 |        29 ms |       25 ms |
#: |    8,066 |        58 ms |       50 ms |
#: |   16,258 |       116 ms |       98 ms |
#: |   32,514 |       235 ms |      195 ms |
#: |   65,282 |       466 ms |      402 ms |
#: |  130,562 |       935 ms |      808 ms |
#:
#: ...linear throughout, about 7.2us/vertex for ``vertex_slide`` (the more
#: expensive of the two, so the one ceiling both ops share is read off its
#: rate). 130,562 vertices already measures within noise of a second, so the
#: ceiling sits at half that -- comfortably under the point (~140,000) where
#: "well under a second" stops being true, the same margin
#: ``ops_topo.MAX_BRIDGED_RING`` keeps under its own measured stall point.
MAX_SLIDE_VERTICES = 65_536


#: The most rails (non-selected incident edges) one touched vertex may have.
#:
#: The 2026-10-14 audit, finding clay-mesh-ops-03: :data:`MAX_SLIDE_VERTICES`
#: bounds how many vertices are touched, not what each costs, and a vertex with
#: more than two rails (a pole, a fan's hub) searches every *pair* of them for the
#: most nearly opposite -- O(valence squared). One spoke selected in a
#: 4,000-triangle fan touches two vertices and froze the frame for 5.4 s
#: (500 rails 0.09 s, 1,000 0.34 s, 2,000 1.35 s). The search is now one blocked
#: matrix product per 256 rails, and the valence that product may be asked about
#: is bounded here: measured 2026-10-14 on this machine, one spoke of a fan costs
#: 0.14 s at 4,000 rails and 0.61 s at 8,192 (the whole press), so a hub past that is
#: a degenerate fan, refused by name.
MAX_SLIDE_RAILS = 8_192

#: Rails per block of the pair search: a block is ``block x rails`` dot products, so
#: this bounds the scratch memory (256 x 8,192 doubles is 16 MB) and nothing else.
_PAIR_BLOCK = 256


def _refuse_slide_size(n_touched: int, what: str) -> None:
    if n_touched > MAX_SLIDE_VERTICES:
        raise OpError(
            f"{what} this selection means walking {n_touched:,} vertices one "
            f"at a time, past the {MAX_SLIDE_VERTICES:,} slide works with "
            "before it would stall the frame it runs on. Slide a smaller "
            "selection."
        )


def _most_opposite(original: np.ndarray, v: int, far: list[int]) -> tuple[int, int]:
    """The two of *far* (sorted vertex ids, three or more) whose directions from *v* are the
    most nearly opposite -- the smallest dot product, the first such pair in ``(i, j)`` order
    on a tie, which is what the one-pair-at-a-time search this replaced returned."""
    dirs = original[far] - original[v]
    lengths = np.linalg.norm(dirs, axis=1)
    live = lengths > 1e-12
    dirs[live] /= lengths[live][:, None]
    count = len(far)
    columns = np.arange(count)[None, :]
    best_score = np.inf
    best = (0, 1)
    for start in range(0, count - 1, _PAIR_BLOCK):
        block = dirs[start : start + _PAIR_BLOCK]
        scores = block @ dirs.T
        rows = np.arange(start, start + len(block))[:, None]
        scores[columns <= rows] = np.inf
        flat = int(np.argmin(scores))
        row, col = divmod(flat, count)
        if scores[row, col] < best_score:
            best_score, best = float(scores[row, col]), (start + row, col)
    return far[best[0]], far[best[1]]


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
    does not move at all. **Which side is "first" is the same side all along a
    loop**, and the side itself is arbitrary: it is seeded at the loop's lowest
    vertex (whose first rail is its lower-numbered one) and carried from each
    vertex to the next through the face they share, so a face either side of the
    loop is the "first" side everywhere or nowhere. The 2026-10-14 audit, finding
    clay-mesh-ops-01: it used to be picked per vertex by the far neighbour's own
    index, which agrees along a strip only on a mesh numbered in order along it --
    an imported or welded mesh tore a freshly cut loop into two heights (7 of 16
    vertices up, 9 down), and "try the other sign" cannot repair a split loop.

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
    _refuse_slide_size(len(verts), "Sliding")
    # The 2026-10-03 audit's clay-45: every neighbour position below used to be
    # read from the array this loop was already rewriting, so two selected
    # vertices that were each other's rail slid cumulatively and the answer
    # depended on vertex index order. Reads come from `original`, writes go to
    # `positions`.
    original = mesh.positions.astype("f8")
    positions = original.copy()
    t = float(t)

    # Pass one: each touched vertex's pair of rails, as (lower-numbered, higher) for two rails.
    pairs: dict[int, tuple[int, int]] = {}
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
        if len(far_verts) > MAX_SLIDE_RAILS:
            raise OpError(
                f"Vertex {v} has {len(far_verts):,} edges leaving it outside the selection, "
                f"past the {MAX_SLIDE_RAILS:,} slide can search for the two that best "
                "continue the strip. Slide a loop that does not run through that hub."
            )
        pairs[v] = (
            (far_verts[0], far_verts[1])
            if len(far_verts) == 2
            else _most_opposite(original, v, far_verts)
        )

    # The 2026-10-14 audit, finding clay-ops-tail-07: at ``t = 0`` nothing moves, but a
    # rebuilt mesh was handed back, so ``set_mesh`` (which judges by identity) pushed an
    # "Edge Slide" step for the dialog's own default and the drag's first frame. The
    # selection was still checked above, so a loop with nowhere to slide is refused at zero
    # as it is anywhere else; only the no-op answers with the mesh it was given.
    if t == 0.0:
        return mesh, sel

    # Pass two: which of each pair is the "first" side, carried along the loop.
    flipped = _orient(a, mesh, sel, pairs)
    for v, (lower, upper) in pairs.items():
        first, second = (upper, lower) if flipped[v] else (lower, upper)
        here = original[v]
        if t < 0.0:
            positions[v] = here + (-t) * (original[first] - here)
        elif t > 0.0:
            positions[v] = here + t * (original[second] - here)

    out = topo.rebuild(positions, mesh.loops, mesh.starts, mesh.material, mesh.smooth, uv=mesh.uv)
    return out, sel


def _orient(
    a: Adjacency, mesh: Mesh, sel: ElementSel, pairs: dict[int, tuple[int, int]]
) -> dict[int, bool]:
    """Per touched vertex, whether its pair is swapped so every vertex of a loop shares a side.

    Seeds each connected piece of the selection at its lowest vertex (unswapped) and walks
    the selected edges: across edge ``(u, w)`` the face of ``u``'s first rail that contains the
    edge is the same face ``w`` sees on its side, and the neighbour of ``w`` in that face is
    ``w``'s first rail. A vertex whose chosen pair does not reach that face (a pole, whose pair
    is the most nearly opposite rather than the adjoining one) keeps its own order: the slide
    still runs, the way a pole always did.
    """
    loops = mesh.loops
    around: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def faces_at(v: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        found = around.get(v)
        if found is None:
            corners = a.vertex_corners(v)
            found = (
                a.corner_face[corners],
                loops[a.next_corner[corners]],
                loops[a.prev_corner[corners]],
            )
            around[v] = found
        return found

    def other_in_face(v: int, w: int) -> dict[int, int]:
        """``{face: v's neighbour in it other than w}`` over the faces that hold edge (v, w)."""
        faces, nxt, prv = faces_at(v)
        found: dict[int, int] = {}
        after = nxt == w
        for face, other in zip(faces[after].tolist(), prv[after].tolist(), strict=True):
            found.setdefault(face, other)
        before = prv == w
        for face, other in zip(faces[before].tolist(), nxt[before].tolist(), strict=True):
            found.setdefault(face, other)
        return found

    neighbours: dict[int, list[int]] = {v: [] for v in pairs}
    for u, w in np.unique(np.sort(sel.edges.astype("i8"), axis=1), axis=0).tolist():
        neighbours[u].append(w)
        neighbours[w].append(u)

    flipped: dict[int, bool] = {}
    for seed in sorted(pairs):
        if seed in flipped:
            continue
        flipped[seed] = False
        todo = [seed]
        while todo:
            u = todo.pop()
            first_u = pairs[u][1 if flipped[u] else 0]
            for w in neighbours[u]:
                if w in flipped:
                    continue
                flip = False
                shared = [f for f, other in other_in_face(u, w).items() if other == first_u]
                if shared:
                    ow = other_in_face(w, u).get(shared[0])
                    flip = ow is not None and ow == pairs[w][1] and ow != pairs[w][0]
                flipped[w] = flip
                todo.append(w)
    return flipped
