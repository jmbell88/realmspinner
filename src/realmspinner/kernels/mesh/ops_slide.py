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
from .adjacency import adjacency
from .elements import ElementSel, OpError
from .mesh import Mesh

__all__ = ["MAX_SLIDE_VERTICES", "edge_slide"]


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


def _refuse_slide_size(n_touched: int, what: str) -> None:
    if n_touched > MAX_SLIDE_VERTICES:
        raise OpError(
            f"{what} this selection means walking {n_touched:,} vertices one "
            f"at a time, past the {MAX_SLIDE_VERTICES:,} slide works with "
            "before it would stall the frame it runs on. Slide a smaller "
            "selection."
        )


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
    _refuse_slide_size(len(verts), "Sliding")
    # The 2026-10-03 audit's clay-45: every neighbour position below used to be
    # read from the array this loop was already rewriting, so two selected
    # vertices that were each other's rail slid cumulatively and the answer
    # depended on vertex index order. Reads come from `original`, writes go to
    # `positions`.
    original = mesh.positions.astype("f8")
    positions = original.copy()
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
            here = original[v]
            dirs = {fv: _unit(original[fv] - here) for fv in far_verts}
            best = None
            for i, fi in enumerate(far_verts):
                for fj in far_verts[i + 1 :]:
                    score = float(dirs[fi] @ dirs[fj])
                    if best is None or score < best[0]:
                        best = (score, fi, fj)
            a_far, b_far = best[1], best[2]
        here = original[v]
        if t < 0.0:
            positions[v] = here + (-t) * (original[a_far] - here)
        elif t > 0.0:
            positions[v] = here + t * (original[b_far] - here)

    out = topo.rebuild(positions, mesh.loops, mesh.starts, mesh.material, mesh.smooth, uv=mesh.uv)
    return out, sel
