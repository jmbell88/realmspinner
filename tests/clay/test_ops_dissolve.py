"""Dissolve: merging across an element, and refusing when the outline forks."""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.clay import adjacency as adj
from warlock.studio.clay import elements as el
from warlock.studio.clay import mesh as bm
from warlock.studio.clay import ops_dissolve as dis
from warlock.studio.clay import primitives as prim
from warlock.studio.clay import topo

from .topo_asserts import assert_closed, assert_consistently_oriented


def _grid(nx: int = 3, nz: int = 3) -> bm.Mesh:
    """An ``nx`` by ``nz`` grid of quads in the XZ plane."""
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
        [v(i, j), v(i, j + 1), v(i + 1, j + 1), v(i + 1, j)] for i in range(nx) for j in range(nz)
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


# --- dissolve_edges ---------------------------------------------------------


def test_dissolving_an_edge_merges_the_two_faces_across_it() -> None:
    m = _grid(2, 1)  # two quads side by side
    shared = np.array([[1, 3]])  # x = 1 column
    a = adj.adjacency(m)
    shared = a.edge_verts[a.edge_uses == 2]
    out, sel = dis.dissolve_edges(m, el.ElementSel(edges=shared))
    bm.validate(out)
    assert bm.face_count(out) == 1
    assert np.diff(out.starts)[0] == 6, "a hexagon: the two ends stay as corners"
    assert sel.faces.tolist() == [0]
    assert_consistently_oriented(out)


def test_a_dissolved_quad_pair_keeps_its_outline_and_drops_nothing_else() -> None:
    m = _grid(2, 1)
    a = adj.adjacency(m)
    out, _ = dis.dissolve_edges(m, el.ElementSel(edges=a.edge_verts[a.edge_uses == 2]))
    assert len(out.positions) == 6, "every vertex is still on the outline"
    lo, hi = bm.bounds(out)
    assert lo.tolist() == bm.bounds(m)[0].tolist()
    assert hi.tolist() == bm.bounds(m)[1].tolist()


def test_dissolving_two_box_edges_in_a_row_merges_three_faces() -> None:
    m = prim.box()
    a = adj.adjacency(m)
    # Two edges of the -Y face, each shared with a different side.
    quad = m.loops[m.starts[0] : m.starts[1]]
    edges = np.array([[quad[0], quad[1]], [quad[1], quad[2]]])
    out, sel = dis.dissolve_edges(m, el.ElementSel(edges=edges))
    bm.validate(out)
    assert bm.face_count(out) == 4, "three faces became one"
    assert len(sel.faces) == 1
    assert_closed(out)
    assert_consistently_oriented(out)
    assert len(a.edge_verts) == 12


def test_dissolve_edges_refuses_a_boundary_edge() -> None:
    m = prim.plane()
    with pytest.raises(el.OpError, match="on a boundary"):
        dis.dissolve_edges(m, el.ElementSel(edges=[[0, 1]]))


def test_dissolve_edges_refuses_a_non_manifold_edge() -> None:
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, -1, -1]], dtype="f4")
    m = bm.Mesh(
        positions=positions,
        loops=np.array([0, 1, 2, 0, 1, 3, 0, 1, 4], dtype="i4"),
        starts=np.array([0, 3, 6, 9], dtype="i4"),
        material=np.zeros(3, dtype="i4"),
        smooth=np.zeros(3, dtype=bool),
    )
    with pytest.raises(el.OpError, match="non-manifold"):
        dis.dissolve_edges(m, el.ElementSel(edges=[[0, 1]]))


def test_dissolve_edges_refuses_an_empty_or_foreign_selection() -> None:
    with pytest.raises(el.OpError, match="Select an edge"):
        dis.dissolve_edges(prim.box(), el.empty())
    with pytest.raises(el.OpError, match="not part of this mesh"):
        dis.dissolve_edges(prim.box(), el.ElementSel(edges=[[0, 6]]))


# --- dissolve_faces ---------------------------------------------------------


def test_dissolving_a_block_of_faces_gives_one_face_along_its_outline() -> None:
    m = _grid(3, 3)
    out, sel = dis.dissolve_faces(m, el.ElementSel(faces=[0, 1, 3, 4]))
    bm.validate(out)
    assert bm.face_count(out) == 6, "nine faces less four, plus one"
    assert np.diff(out.starts)[-1] == 8, "the 2x2 block's outline is eight corners"
    assert len(sel.faces) == 1
    assert_consistently_oriented(out)


def test_dissolving_two_separate_blocks_gives_two_faces() -> None:
    m = _grid(3, 3)
    out, sel = dis.dissolve_faces(m, el.ElementSel(faces=[0, 1, 7, 8]))
    bm.validate(out)
    assert len(sel.faces) == 2
    assert bm.face_count(out) == 7


def test_dissolve_refuses_a_selection_that_rings_a_face_it_leaves_out() -> None:
    m = _grid(3, 3)
    ring = [f for f in range(9) if f != 4]
    with pytest.raises(el.OpError, match="surrounds a face it does not include"):
        dis.dissolve_faces(m, el.ElementSel(faces=ring))


def test_dissolve_refuses_a_bowtie_outline() -> None:
    # A 4x4 grid with two diagonally-touching faces left out: the two holes
    # meet at one vertex, so the region's outline passes through it twice and
    # the merged n-gon would fold onto itself there.
    m = _grid(4, 4)
    region = [f for f in range(16) if f not in (5, 10)]
    with pytest.raises(el.OpError, match="appears twice on the outline"):
        dis.dissolve_faces(m, el.ElementSel(faces=region))


def test_dissolving_a_whole_closed_surface_is_refused() -> None:
    with pytest.raises(el.OpError, match="closed surface"):
        dis.dissolve_faces(prim.box(), el.ElementSel(faces=range(6)))


def test_dissolve_faces_refuses_an_empty_selection() -> None:
    with pytest.raises(el.OpError, match="Select the faces"):
        dis.dissolve_faces(prim.box(), el.empty())


def test_a_lone_selected_face_has_nothing_to_dissolve() -> None:
    with pytest.raises(el.OpError, match="Nothing there to dissolve"):
        dis.dissolve_faces(_grid(3, 3), el.ElementSel(faces=[0]))


# --- dissolve_verts ---------------------------------------------------------


def test_dissolving_an_interior_vertex_merges_its_fan() -> None:
    m = _grid(2, 2)
    centre = int(np.flatnonzero((m.positions == [1.0, 0.0, 1.0]).all(axis=1))[0])
    out, sel = dis.dissolve_verts(m, el.ElementSel(verts=[centre]))
    bm.validate(out)
    assert bm.face_count(out) == 1
    assert np.diff(out.starts)[0] == 8
    assert len(out.positions) == 8, "the dissolved vertex is gone"
    assert len(sel.faces) == 1
    assert_consistently_oriented(out)


def test_dissolving_a_box_corner_merges_its_three_faces() -> None:
    m = prim.box()
    out, sel = dis.dissolve_verts(m, el.ElementSel(verts=[0]))
    bm.validate(out)
    assert bm.face_count(out) == 4
    assert len(out.positions) == 7
    assert len(sel.faces) == 1
    assert_closed(out)
    assert_consistently_oriented(out)


def test_dissolve_verts_refuses_a_boundary_vertex() -> None:
    with pytest.raises(el.OpError, match="on a boundary"):
        dis.dissolve_verts(_grid(2, 2), el.ElementSel(verts=[0]))


def test_dissolve_verts_refuses_a_non_manifold_vertex() -> None:
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, -1, -1]], dtype="f4")
    m = bm.Mesh(
        positions=positions,
        loops=np.array([0, 1, 2, 0, 1, 3, 0, 1, 4], dtype="i4"),
        starts=np.array([0, 3, 6, 9], dtype="i4"),
        material=np.zeros(3, dtype="i4"),
        smooth=np.zeros(3, dtype=bool),
    )
    with pytest.raises(el.OpError, match="boundary|non-manifold"):
        dis.dissolve_verts(m, el.ElementSel(verts=[0]))


def test_dissolve_verts_refuses_an_empty_or_foreign_selection() -> None:
    with pytest.raises(el.OpError, match="Select a vertex"):
        dis.dissolve_verts(prim.box(), el.empty())
    with pytest.raises(el.OpError, match="not part of this mesh"):
        dis.dissolve_verts(prim.box(), el.ElementSel(verts=[99]))


# --- shared behaviour -------------------------------------------------------


def test_a_dissolve_keeps_each_border_corners_own_uv() -> None:
    m = _uvd(_grid(2, 1))
    a = adj.adjacency(m)
    out, _ = dis.dissolve_edges(m, el.ElementSel(edges=a.edge_verts[a.edge_uses == 2]))
    assert out.uv is not None and len(out.uv) == len(out.loops)
    assert all(row.tolist() in m.uv.tolist() for row in out.uv)


def test_a_dissolve_may_produce_a_concave_face_that_still_triangulates_inside() -> None:
    # An L of three quads: dissolving them leaves a genuinely concave octagon.
    m = _grid(2, 2)
    out, _ = dis.dissolve_faces(m, el.ElementSel(faces=[0, 1, 2]))
    bm.validate(out)
    normals = bm._face_normals(out)
    from warlock.studio.clay import earclip as ec

    assert ec.concave_faces(out.positions, out.loops, out.starts, normals).any()
    tris, tri_face = bm.triangulate(out)
    merged = tris[tri_face == bm.face_count(out) - 1]
    area = float(
        np.linalg.norm(
            np.cross(
                out.positions[merged[:, 1]] - out.positions[merged[:, 0]],
                out.positions[merged[:, 2]] - out.positions[merged[:, 0]],
            ),
            axis=1,
        ).sum()
        * 0.5
    )
    assert area == pytest.approx(3.0, rel=1e-5), "three unit quads, no overshoot"


def test_the_merged_face_inherits_material_and_smoothing() -> None:
    m = _grid(2, 1)
    marked = bm.Mesh(
        positions=m.positions,
        loops=m.loops,
        starts=m.starts,
        material=np.array([7, 7], dtype="i4"),
        smooth=np.array([True, True]),
    )
    a = adj.adjacency(marked)
    out, sel = dis.dissolve_edges(marked, el.ElementSel(edges=a.edge_verts[a.edge_uses == 2]))
    assert out.material[sel.faces[0]] == 7
    assert bool(out.smooth[sel.faces[0]])


# --- the growth ceiling ------------------------------------------------------


def test_a_dissolve_whose_outline_is_too_big_is_refused(monkeypatch) -> None:
    """``ops_subdiv`` refuses past ``MAX_SUBDIVIDED_FACES``; this is the other
    unbounded growth an edit can ask for.

    The n-gon goes to ``earclip``, whose ear search is quadratic in the corner
    count and runs in Python on the frame thread -- so the refusal has to come
    before the merge, not after it. Driven with the ceiling lowered rather than
    with a twenty-thousand-corner mesh, which would cost more to build than the
    thing it is testing.
    """
    m = _grid(3, 3)
    monkeypatch.setattr(dis, "MAX_DISSOLVED_RING", 4)
    with pytest.raises(el.OpError, match="corners, past the"):
        dis.dissolve_faces(m, el.ElementSel(faces=[0, 1, 3, 4]))


def _comb_row(teeth: int) -> bm.Mesh:
    """A row of *teeth* quads whose top edges alternate height.

    Dissolving the whole row merges them into one n-gon whose top boundary
    zigzags -- the concrete "routinely concave" shape this module's own
    docstring names, and the shape the 2026-09-11 audit's clay-02 measured
    earclip's ear search on.
    """
    n = teeth + 1
    xs = np.arange(n, dtype="f4")
    heights = np.where(np.arange(n) % 2 == 0, 1.0, 0.1).astype("f4")
    bottom = np.stack([xs, np.zeros(n, dtype="f4"), np.zeros(n, dtype="f4")], axis=1)
    top = np.stack([xs, np.zeros(n, dtype="f4"), heights], axis=1)
    positions = np.concatenate([bottom, top], axis=0)
    faces = [[i, i + 1, n + i + 1, n + i] for i in range(teeth)]
    return bm.Mesh(
        positions=positions,
        loops=np.array([c for f in faces for c in f], dtype="i4"),
        starts=topo.starts_from_counts([4] * teeth),
        material=np.zeros(teeth, dtype="i4"),
        smooth=np.zeros(teeth, dtype=bool),
    )


def test_a_zigzag_past_the_concave_bound_is_refused_not_mistriangulated() -> None:
    """The 2026-09-11 audit's clay-02: ``MAX_DISSOLVED_RING``'s own comment
    claims its 20,000-corner ceiling keeps earclip's ear search "well under a
    second", but the search is quadratic in a *concave* ring's corner count.
    Driving it directly on a realistic concave (zigzag) ring -- exactly what
    this module's own docstring says a dissolve routinely produces -- measured
    897 ms at 1,600 corners and 3.63 s at 3,200, a clean quadratic trend that
    extrapolates to roughly 140 seconds at the pinned 20,000-corner ceiling,
    not "well under a second".

    A first version of this fix put a size ceiling inside earclip itself, past
    which a concave face silently kept the plain fan instead of running the
    ear search: bounded, but wrong in a new way a fan across a reflex corner
    puts a triangle outside the polygon, and nothing downstream can see it,
    because ``check_manifold`` reads only CSR topology and a wrong
    triangulation changes no topology. The guard now lives in the op layer
    instead (``ops_dissolve.MAX_CONCAVE_DISSOLVE_RING`` /
    ``_refuse_concave_ring``), refusing the merge by name rather than quietly
    mistriangulating it -- earclip itself is unchanged and stays exactly as
    tolerant as it was (its own fan-on-stall fallback still exists, for a
    genuinely degenerate ring, not for a merely large one).

    Both sides of the real bound are checked, without referencing the
    constant itself: a zigzag whose merge would be 1,202 corners must be
    refused, naming the reason, before the merge runs at all -- and one whose
    merge would be 998 corners -- comfortably under the real 1,000-corner
    bound -- must still succeed and come back properly ear-clipped, not
    fanned. That second half is what stops a future change from quietly
    reintroducing the fan-fallback failure mode by other means.
    """
    over = _comb_row(teeth=600)  # would merge into a 1,202-corner zigzag
    with pytest.raises(el.OpError, match="too complex"):
        dis.dissolve_faces(over, el.ElementSel(faces=range(600)))

    from warlock.studio.clay import earclip as ec

    under = _comb_row(teeth=498)  # merges into a 998-corner zigzag
    out, sel = dis.dissolve_faces(under, el.ElementSel(faces=range(498)))
    bm.validate(out)
    face = int(sel.faces[0])
    normals = bm._face_normals(out)
    assert ec.concave_faces(out.positions, out.loops, out.starts, normals)[face]
    # ``_corner_triangles``/``_fan_corners`` return *corner* indices, unlike
    # ``triangulate`` which maps them through ``loops`` into vertex ids --
    # the same pair ``test_earclip.py`` compares for exactly this reason.
    tris, tri_face = bm._corner_triangles(out)
    got = tris[tri_face == face]
    want, want_face = bm._fan_corners(out)
    want = want[want_face == face]
    assert not np.array_equal(got, want), (
        "under the bound, the zigzag face must still be properly ear-clipped, not fanned"
    )


def test_the_ceiling_is_far_above_any_ordinary_dissolve() -> None:
    """It must not be felt: the 2x2 block above is an eight-corner outline."""
    assert dis.MAX_DISSOLVED_RING >= 20_000
    m = _grid(3, 3)
    out, sel = dis.dissolve_faces(m, el.ElementSel(faces=[0, 1, 3, 4]))
    bm.validate(out)
    assert len(sel.faces) == 1


# --- the 2026-09-08 audit, second run: cost must track the selection, not the mesh ------


def _padded(m: bm.Mesh, n_extra: int) -> bm.Mesh:
    """*m* plus ``n_extra`` disconnected triangles nothing else touches.

    Every extra triangle gets its own three fresh vertices, so it shares no
    edge with anything -- it exists only to inflate ``face_count`` the way a
    dense, unrelated part of a real mesh would, without needing to build one.
    """
    extra_positions = np.zeros((n_extra * 3, 3), dtype="f4")
    base_v = len(m.positions)
    extra_loops = np.arange(base_v, base_v + n_extra * 3, dtype="i4")
    counts = list(np.diff(m.starts)) + [3] * n_extra
    return bm.Mesh(
        positions=np.concatenate([m.positions, extra_positions]),
        loops=np.concatenate([m.loops, extra_loops]),
        starts=topo.starts_from_counts(counts),
        material=np.concatenate([m.material, np.zeros(n_extra, dtype="i4")]),
        smooth=np.concatenate([m.smooth, np.zeros(n_extra, dtype=bool)]),
    )


def test_dissolving_one_edge_does_not_walk_the_whole_meshs_face_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 2026-09-08 audit's second run (clay-08) found ``_Union.groups()`` built with
    ``for i in range(len(self.parent))`` in all three dissolve ops -- every
    face in the whole mesh, not the selection -- so one edge dissolved
    measured 3.5 ms at 2,401 faces and 654 ms at 408,321, linear in mesh size
    with no refusal, which broke ``dev/INVARIANTS.md``'s promise that every
    Clay op's cost tracks what it grows.

    Proven structurally rather than by wall clock (flaky under xdist): a mesh
    with 20,000 disconnected faces the selection never names should call
    ``find()`` a handful of times, not 20,000-plus.
    """
    m = _padded(_grid(2, 1), n_extra=20_000)
    a = adj.adjacency(m)
    shared = a.edge_verts[a.edge_uses == 2]
    assert len(shared) == 1, "only the grid's shared edge, none of the padding"

    calls = 0
    original = dis._Union.find

    def counting_find(self: dis._Union, x: int) -> int:
        nonlocal calls
        calls += 1
        return original(self, x)

    monkeypatch.setattr(dis._Union, "find", counting_find)
    out, sel = dis.dissolve_edges(m, el.ElementSel(edges=shared))
    bm.validate(out)
    assert bm.face_count(out) == 20_001, "the two grid faces merged, the padding untouched"
    assert calls < 50, f"dissolving one edge called find() {calls} times on 20,002 faces"
