"""Regression test for the 2026-09-23 audit's clay-18: ``topo.
region_boundary_corners`` on a non-manifold edge."""

from __future__ import annotations

import numpy as np

from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import topo

# --- clay-18: region_boundary_corners on a non-manifold edge -----------------


def _hinge_fan(n_pages: int) -> bm.Mesh:
    """*n_pages* triangles hinged on one shared edge (v0, v1) -- the
    textbook non-manifold edge, ``edge_uses == n_pages`` for that one edge
    rather than the 1 or 2 an ordinary manifold mesh ever produces. Each
    page's apex sits at its own position so the faces are non-degenerate.
    """
    positions = [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    faces = []
    for i in range(n_pages):
        angle = i * (np.pi / max(n_pages, 1))
        positions.append((1.0 + 0.1 * i, 0.5, angle))  # apex, distinct per page
        faces.append([0, 1, 2 + i])
    return bm.from_faces(np.asarray(positions, dtype="f4"), faces)


def test_region_boundary_corners_includes_a_non_manifold_edge_with_an_unselected_face_on_it() -> (
    None
):
    """The 2026-09-23 audit, finding clay-18: an edge shared by three faces
    (two selected, one not) used to be missed. The old rule compared the
    *selected* corner count on an edge against exactly 1 -- right for an
    ordinary manifold edge, where at most two faces ever share one, but
    wrong here: both selected pages touch the hinge edge, so the old count
    read 2 and the edge was treated as interior even though the third,
    unselected page means it plainly borders the selection.
    """
    mesh = _hinge_fan(3)
    # Select pages 0 and 1; leave page 2 unselected. All three share the
    # hinge edge (vertices 0 and 1).
    border = topo.region_boundary_corners(mesh, [0, 1])
    # Every corner of both selected triangles should be on the border: the
    # hinge edge because an unselected third face also touches it, and the
    # two outer edges of each triangle because nothing else touches them at
    # all (true open boundary).
    assert len(border) == 6  # 2 faces * 3 corners each, all boundary

    # The closed-off case still reads as before: with every page selected,
    # the hinge edge is used by three *selected* corners and nothing
    # unselected touches it (per_edge == edge_uses == 3, not 1), so it is
    # correctly interior. Each triangle has exactly one corner leaving along
    # the hinge edge, so 3 of the 9 corners drop out, leaving the 6 that
    # leave along an outer, still-unshared edge.
    all_border = topo.region_boundary_corners(mesh, [0, 1, 2])
    assert len(all_border) == 6


