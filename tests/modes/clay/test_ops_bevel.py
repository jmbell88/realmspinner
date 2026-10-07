"""Bevel: the walking op."""

from __future__ import annotations

import numpy as np
import pytest

from realmspinner.kernels.mesh import adjacency as adj
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import ops_bevel as ob
from realmspinner.kernels.mesh import primitives as prim
from realmspinner.kernels.mesh import topo

from .topo_asserts import assert_closed, assert_consistently_oriented


def _grid(nx: int = 3, nz: int = 1) -> bm.Mesh:
    xs = np.arange(nx + 1, dtype="f4")
    zs = np.arange(nz + 1, dtype="f4")
    positions = np.stack(
        [
            np.repeat(xs, nz + 1),
            np.zeros((nx + 1) * (nz + 1), dtype="f4"),
            np.tile(zs, nx + 1),
        ],
        axis=1,
    )

    def v(i: int, j: int) -> int:
        return i * (nz + 1) + j

    faces = [
        [v(i, j), v(i, j + 1), v(i + 1, j + 1), v(i + 1, j)]
        for i in range(nx)
        for j in range(nz)
    ]
    return bm.Mesh(
        positions=positions,
        loops=np.array([c for f in faces for c in f], dtype="i4"),
        starts=topo.starts_from_counts([4] * len(faces)),
        material=np.zeros(len(faces), dtype="i4"),
        smooth=np.zeros(len(faces), dtype=bool),
    )


def _uvd(mesh: bm.Mesh) -> bm.Mesh:
    n = len(mesh.loops)
    uv = np.stack([np.arange(n, dtype="f4"), np.arange(n, dtype="f4") * 2], axis=1)
    return bm.Mesh(
        positions=mesh.positions,
        loops=mesh.loops,
        starts=mesh.starts,
        material=mesh.material,
        smooth=mesh.smooth,
        uv=uv,
    )


# --- bevel_edges ------------------------------------------------------------


def _cube_edge(m: bm.Mesh, i: int = 0) -> np.ndarray:
    return adj.adjacency(m).edge_verts[i]


def test_beveling_one_cube_edge_gives_a_quad_and_two_pentagons() -> None:
    m = prim.box()
    out, sel = ob.bevel_edges(m, el.ElementSel(edges=[_cube_edge(m)]), width=0.1)
    bm.validate(out)
    assert_closed(out)
    assert_consistently_oriented(out)
    assert adj.check_manifold(out).clean

    assert bm.face_count(out) == 7, "six faces plus one bevel quad, no corner caps"
    assert len(sel.faces) == 1
    sizes = sorted(np.diff(out.starts).tolist())
    assert sizes.count(5) == 2, "the two faces opposite the beveled edge grew a corner"
    assert sizes.count(4) == 5


def test_beveling_the_three_edges_at_a_cube_corner_mitres_it() -> None:
    m = prim.box()
    a = adj.adjacency(m)
    corner = 0
    at_corner = a.edge_verts[(a.edge_verts == corner).any(axis=1)]
    assert len(at_corner) == 3
    out, sel = ob.bevel_edges(m, el.ElementSel(edges=at_corner), width=0.1)
    bm.validate(out)
    assert_closed(out)
    assert_consistently_oriented(out)
    assert adj.check_manifold(out).clean
    # Six originals, three bevel quads, one miter triangle.
    assert bm.face_count(out) == 10
    assert len(sel.faces) == 4
    sizes = np.diff(out.starts)
    assert (sizes[-1] == 3).all() or int(sizes[-1]) == 3, "the corner cap is a triangle"


def test_beveling_all_twelve_cube_edges_gives_twenty_six_faces() -> None:
    m = prim.box()
    a = adj.adjacency(m)
    out, sel = ob.bevel_edges(m, el.ElementSel(edges=a.edge_verts), width=0.1)
    bm.validate(out)
    assert bm.face_count(out) == 26, "6 faces + 12 edge quads + 8 corner triangles"
    assert len(sel.faces) == 20
    assert_closed(out)
    assert_consistently_oriented(out)
    assert adj.check_manifold(out).clean
    assert len(out.positions) == 24, "three new vertices per original corner"


def test_a_bevel_is_crack_free_because_both_faces_share_the_slide_vertex() -> None:
    m = prim.box()
    out, _ = ob.bevel_edges(m, el.ElementSel(edges=[_cube_edge(m)]), width=0.1)
    report = adj.check_manifold(out)
    assert len(report.boundary_edges) == 0
    assert len(report.nonmanifold_edges) == 0
    assert len(report.duplicate_faces) == 0


def test_a_bevel_wider_than_the_feature_collapses_rather_than_inverting() -> None:
    m = prim.box()
    a = adj.adjacency(m)
    out, _ = ob.bevel_edges(m, el.ElementSel(edges=a.edge_verts), width=99.0)
    bm.validate(out)
    lo, hi = bm.bounds(out)
    assert (lo >= bm.bounds(m)[0] - 1e-5).all()
    assert (hi <= bm.bounds(m)[1] + 1e-5).all()


def test_a_bevels_slide_uvs_are_interpolated_and_its_miters_are_copied() -> None:
    m = _uvd(prim.box())
    out, _ = ob.bevel_edges(m, el.ElementSel(edges=[_cube_edge(m)]), width=0.1)
    assert out.uv is not None and len(out.uv) == len(out.loops)
    assert np.isfinite(out.uv).all()


def test_bevel_refuses_before_it_walks_the_whole_mesh_past_a_size_ceiling() -> None:
    """The 2026-09-08 audit's clay-03: bevel_edges had no growth/complexity
    ceiling at all, unlike ops_subdiv.subdivide, ops_dissolve's merge and
    ops_boolean's kernel call -- its rewrite walks the *whole* mesh with two
    unconditional Python loops ("for face in range(faces):" and "for corner in
    range(len(loops)):") regardless of how small the selection is, so a
    bevel's cost tracks mesh size, not selection size, with nothing to refuse
    it before clay_ops.run_mesh_op calls it on the frame thread. Build a mesh
    just past ob.MAX_BEVELED_CORNERS and check bevelling a single interior
    edge is refused up front, naming the corner count, rather than walking it.
    """
    n = 708  # 4 * n * n = 2,005,056 corners, just past MAX_BEVELED_CORNERS
    m = _grid(n, n)
    assert len(m.loops) > ob.MAX_BEVELED_CORNERS
    a = adj.adjacency(m)
    edge = a.edge_verts[a.edge_uses == 2][0]
    with pytest.raises(el.OpError, match="past the"):
        ob.bevel_edges(m, el.ElementSel(edges=[edge]), width=0.1)


def test_bevel_edges_refuses_a_uv_mesh_past_the_uv_corner_ceiling() -> None:
    """The 2026-09-14 audit's clay-05: MAX_BEVELED_CORNERS was tuned on a mesh
    with no UVs. bevel_edges's per-corner UV lerp (``replace``/``one`` above)
    roughly doubles the real per-corner cost, so a UV-bearing mesh at the
    plain ceiling measured 1.62s against 0.81s without UVs (probes
    clay-mesh-04..07) -- comfortably past the "well under a second" bar every
    other Clay op ceiling holds to. A mesh sized between half and all of
    MAX_BEVELED_CORNERS must refuse once it carries UVs and must not when it
    does not, which is the only way to prove the ceiling actually halves
    rather than just reusing the old message with new numbers.
    """
    n = 600  # 4 * n * n = 1,440,000 corners: over half, under all, of MAX_BEVELED_CORNERS
    m = _grid(n, n)
    assert ob.MAX_BEVELED_CORNERS // 2 < len(m.loops) < ob.MAX_BEVELED_CORNERS

    ob._refuse_size(m)  # no UVs: still comfortably inside the plain ceiling

    uv_mesh = _uvd(m)
    with pytest.raises(el.OpError, match="past the"):
        ob._refuse_size(uv_mesh)

    a = adj.adjacency(m)
    edge = a.edge_verts[a.edge_uses == 2][0]
    with pytest.raises(el.OpError, match="past the"):
        ob.bevel_edges(uv_mesh, el.ElementSel(edges=[edge]), width=0.1)


def test_bevel_refuses_a_boundary_edge_and_an_empty_selection() -> None:
    m = _grid(2, 1)
    boundary = adj.check_manifold(m).boundary_edges[0]
    with pytest.raises(el.OpError, match="on a boundary"):
        ob.bevel_edges(m, el.ElementSel(edges=[boundary]), width=0.1)
    with pytest.raises(el.OpError, match="Select an edge"):
        ob.bevel_edges(m, el.empty())
    with pytest.raises(el.OpError, match="not part of this mesh"):
        ob.bevel_edges(m, el.ElementSel(edges=[[0, 5]]))


def test_bevel_refuses_a_non_manifold_edge() -> None:
    positions = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, -1, -1]], dtype="f4"
    )
    m = bm.Mesh(
        positions=positions,
        loops=np.array([0, 1, 2, 0, 1, 3, 0, 1, 4], dtype="i4"),
        starts=np.array([0, 3, 6, 9], dtype="i4"),
        material=np.zeros(3, dtype="i4"),
        smooth=np.zeros(3, dtype=bool),
    )
    with pytest.raises(el.OpError, match="3 faces on it"):
        ob.bevel_edges(m, el.ElementSel(edges=[[0, 1]]), width=0.1)


def test_bevel_refuses_a_border_vertex_carrying_two_beveled_edges() -> None:
    # A 2x2 grid with one quad removed: the middle vertex now has two boundary
    # edges (where the missing quad was) and two interior ones. Beveling both
    # interior edges asks for a corner polygon at a vertex whose fan is open.
    xs = np.arange(3, dtype="f4")
    positions = np.stack(
        [np.repeat(xs, 3), np.zeros(9, dtype="f4"), np.tile(xs, 3)], axis=1
    )
    faces = [
        [i * 3 + j, i * 3 + j + 1, (i + 1) * 3 + j + 1, (i + 1) * 3 + j]
        for i in range(2)
        for j in range(2)
    ][1:]
    m = bm.Mesh(
        positions=positions,
        loops=np.array([c for f in faces for c in f], dtype="i4"),
        starts=topo.starts_from_counts([4] * 3),
        material=np.zeros(3, dtype="i4"),
        smooth=np.zeros(3, dtype=bool),
    )
    a = adj.adjacency(m)
    at_middle = (a.edge_verts == 4).any(axis=1)
    interior = a.edge_verts[at_middle & (a.edge_uses == 2)]
    assert len(interior) == 2
    assert (a.edge_uses[at_middle] == 1).any(), "the middle vertex is on the border"
    with pytest.raises(el.OpError, match="on the border"):
        ob.bevel_edges(m, el.ElementSel(edges=interior), width=0.1)


def test_beveling_one_interior_edge_of_an_open_sheet_caps_the_interior_end() -> None:
    # A 2x2 grid: one interior edge runs from the centre (four edges, so three
    # distinct slide vertices once one is beveled -- a corner polygon) out to a
    # border vertex (two, so none).
    xs = np.arange(3, dtype="f4")
    positions = np.stack(
        [np.repeat(xs, 3), np.zeros(9, dtype="f4"), np.tile(xs, 3)], axis=1
    )
    faces = [
        [i * 3 + j, i * 3 + j + 1, (i + 1) * 3 + j + 1, (i + 1) * 3 + j]
        for i in range(2)
        for j in range(2)
    ]
    m = bm.Mesh(
        positions=positions,
        loops=np.array([c for f in faces for c in f], dtype="i4"),
        starts=topo.starts_from_counts([4] * 4),
        material=np.zeros(4, dtype="i4"),
        smooth=np.zeros(4, dtype=bool),
    )
    a = adj.adjacency(m)
    one = a.edge_verts[a.edge_uses == 2][:1]
    out, sel = ob.bevel_edges(m, el.ElementSel(edges=one), width=0.1)
    bm.validate(out)
    assert bm.face_count(out) == 6, "four faces, one bevel quad, one corner cap"
    assert len(sel.faces) == 2
    assert_consistently_oriented(out)
    assert len(adj.check_manifold(out).nonmanifold_edges) == 0


def test_bevel_miter_at_a_reflex_corner_lands_inside_the_face() -> None:
    """clay-06: the miter went along d1+d2, the smaller-angle bisector, so at a
    reflex corner of a concave face it landed in the notch outside the face."""
    pts = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]  # L, CCW seen from +Z
    n = len(pts)
    pos = [[x, y, 1] for x, y in pts] + [[x, y, 0] for x, y in pts]
    faces = [list(range(n)), [n + i for i in reversed(range(n))]]
    for i in range(n):
        j = (i + 1) % n
        faces.append([n + i, n + j, j, i])  # outward side, CCW from outside
    m = bm.Mesh(
        positions=np.array(pos, dtype="f4"),
        loops=np.array([c for f in faces for c in f], dtype="i4"),
        starts=topo.starts_from_counts([len(f) for f in faces]),
        material=np.zeros(len(faces), dtype="i4"),
        smooth=np.zeros(len(faces), dtype=bool),
    )
    assert_consistently_oriented(m)
    out, _ = ob.bevel_edges(m, el.ElementSel(edges=[[2, 3], [3, 4]]), width=0.1)
    near = [p for p in out.positions if abs(p[0] - 1) < 0.4 and abs(p[1] - 1) < 0.4 and p[2] > 0.5]
    inside = [p for p in near if p[0] < 1 - 1e-4 and p[1] < 1 - 1e-4]
    assert inside, f"no miter inside the L's inner corner: {out.positions.tolist()}"
    assert not [p for p in near if p[0] > 1 + 1e-4 and p[1] > 1 + 1e-4], "a miter sits in the notch"
