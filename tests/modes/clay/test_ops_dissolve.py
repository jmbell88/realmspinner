"""Dissolve: merging across an element, and refusing when the outline forks."""

from __future__ import annotations

import time

import numpy as np
import pytest

from realmspinner.kernels.mesh import adjacency as adj
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import ops_dissolve as dis
from realmspinner.kernels.mesh import primitives as prim
from realmspinner.kernels.mesh import topo

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


def test_dissolve_verts_refuses_or_stays_fast_past_a_selection_size_ceiling_on_a_large_interior_selection(  # noqa: E501
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 2026-09-22 audit's clay-11: unlike every other walking op in this
    package, ``dissolve_verts`` had no ceiling at all on the size of the
    *selection* itself -- ``MAX_DISSOLVED_RING`` bounds only the merged
    n-gon's own outline, which stays small for a large *interior* selection
    (a k-vertex-square block's boundary has only ~4k corners, not k^2), so it
    cannot see this shape of cost. Reproduced (audit's own probe): 2.6s at
    249,000 vertices, linear; re-measured at merge on a contiguous interior
    block, the shape that keeps the output ring small: 0.65s at 40,000
    vertices, 3.9s at 250,000.

    Driven here with the ceiling lowered, on a selection that never reaches
    ``adjacency()`` at all (the refusal is the very first thing the function
    does past the empty check), rather than building a mesh anywhere near
    the real ceiling.
    """
    # The ceiling itself must not have crept down onto ordinary use.
    assert dis.MAX_DISSOLVED_VERTS >= 20_000
    m = prim.box()
    monkeypatch.setattr(dis, "MAX_DISSOLVED_VERTS", 4)
    with pytest.raises(el.OpError, match=r"5.*past the.*4"):
        dis.dissolve_verts(m, el.ElementSel(verts=np.arange(5, dtype="i8")))


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
    from realmspinner.kernels.mesh import earclip as ec

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

    from realmspinner.kernels.mesh import earclip as ec

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


def test_dissolving_one_edge_does_not_walk_the_whole_meshs_face_count() -> None:
    """The 2026-09-08 audit's second run (clay-08) found ``_Union.groups()`` built with
    ``for i in range(len(self.parent))`` in all three dissolve ops -- every
    face in the whole mesh, not the selection -- so one edge dissolved
    measured 3.5 ms at 2,401 faces and 654 ms at 408,321, linear in mesh size
    with no refusal, which broke ``dev/INVARIANTS.md``'s promise that every
    Clay op's cost tracks what it grows.

    ``_Union`` (and the per-face-corner Python ``union()``/``find()`` loop
    feeding it) was removed 2026-09-17 -- the 2026-09-17 native-kernel review
    (batch 11) found the *narrowed* ``groups(subset)`` this test originally
    guarded still cost 665 ms at 200k faces select-all, 1.79M Python
    ``find()`` calls -- so there is no ``find()`` left to count. The
    replacement, ``scipy.sparse.csgraph.connected_components``, does one pass
    over the *whole* face graph in C regardless of the selection's size --
    the same trade :func:`~.select.linked` already made for the identical
    reason (its own docstring). That makes cost roughly flat in mesh size,
    not merely sub-linear, so the property worth proving is no longer "few
    Python calls" but "large disconnected padding costs about what padding of
    a tenth the size does" -- checked here as a scaling ratio rather than an
    absolute bound, so it is not sensitive to the machine running it.
    """
    small = _padded(_grid(2, 1), n_extra=2_000)
    large = _padded(_grid(2, 1), n_extra=20_000)

    def _time_dissolve(m: bm.Mesh) -> float:
        a = adj.adjacency(m)
        shared = a.edge_verts[a.edge_uses == 2]
        assert len(shared) == 1, "only the grid's shared edge, none of the padding"
        # Warm the adjacency cache before timing: it is built once per mesh and
        # cached (adjacency.py's own ``_CACHE``), so a real drag never pays for
        # it twice, and this test should not either.
        start = time.perf_counter()
        out, _ = dis.dissolve_edges(m, el.ElementSel(edges=shared))
        elapsed = time.perf_counter() - start
        bm.validate(out)
        assert bm.face_count(out) == len(m.starts) - 2, "the two grid faces merged, padding intact"
        return elapsed

    small_time = _time_dissolve(small)
    large_time = _time_dissolve(large)
    # A Python-per-face loop (the removed antipattern) would be roughly 10x
    # slower at 10x the padding; one C pass over a sparse graph is not. 6x
    # gives real margin above 1x while catching a straight reintroduction of
    # a per-face Python loop.
    assert large_time < max(small_time * 6.0, 0.2), (
        f"20,000 padding faces took {large_time:.4f}s against {small_time:.4f}s "
        "at 2,000 -- scaling like mesh size again?"
    )


# --- parity with the removed ``_Union`` grouping, batch 11 ------------------
#
# The 2026-09-17 native-kernel review replaced the Python union-find (see the
# module's own history above) with ``scipy.sparse.csgraph.connected_components``
# plus a fully vectorised grouping step. These tests hold a byte-for-byte copy
# of the *old* ``_Union`` class and the three dissolve functions as they stood
# right before that change (git blame: the commit before this batch), and
# check the new functions against it on meshes the old code never crashed on.


class _OldUnion:
    """The removed ``ops_dissolve._Union``, verbatim, for parity only."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

    def groups(self, subset: np.ndarray | None = None) -> list[list[int]]:
        out: dict[int, list[int]] = {}
        indices = range(len(self.parent)) if subset is None else subset
        for i in indices:
            out.setdefault(self.find(int(i)), []).append(int(i))
        return list(out.values())


def _old_dissolve_edges(mesh: bm.Mesh, sel: el.ElementSel) -> tuple[bm.Mesh, el.ElementSel]:
    if len(sel.edges) == 0:
        raise el.OpError("Select an edge to dissolve.")
    ids = dis._check_edges(mesh, sel.edges)
    a = adj.adjacency(mesh)
    union = _OldUnion(bm.face_count(mesh))
    order = np.argsort(a.corner_edge, kind="stable")
    by_edge = a.corner_edge[order]
    faces_by_edge = a.corner_face[order]
    lo = np.searchsorted(by_edge, ids, side="left")
    hi = np.searchsorted(by_edge, ids, side="right")
    touched: list[int] = []
    for start, stop in zip(lo.tolist(), hi.tolist(), strict=True):
        pair = faces_by_edge[start:stop]
        union.union(int(pair[0]), int(pair[-1]))
        touched.extend(pair.tolist())
    subset = np.unique(np.asarray(touched, dtype="i8")) if touched else np.empty(0, dtype="i8")
    return dis.merge_groups(mesh, [np.array(g) for g in union.groups(subset)])


def _old_dissolve_faces(mesh: bm.Mesh, sel: el.ElementSel) -> tuple[bm.Mesh, el.ElementSel]:
    if len(sel.faces) == 0:
        raise el.OpError("Select the faces to dissolve into one.")
    a = adj.adjacency(mesh)
    chosen = np.zeros(bm.face_count(mesh), dtype=bool)
    chosen[sel.faces] = True
    union = _OldUnion(bm.face_count(mesh))
    interior = np.flatnonzero(chosen[a.corner_face] & (a.twin >= 0))
    for corner in interior.tolist():
        other = int(a.corner_face[a.twin[corner]])
        if chosen[other]:
            union.union(int(a.corner_face[corner]), other)
    groups = [np.array(g) for g in union.groups(np.flatnonzero(chosen))]
    return dis.merge_groups(mesh, groups)


def _old_dissolve_verts(mesh: bm.Mesh, sel: el.ElementSel) -> tuple[bm.Mesh, el.ElementSel]:
    if len(sel.verts) == 0:
        raise el.OpError("Select a vertex to dissolve.")
    a = adj.adjacency(mesh)
    union = _OldUnion(bm.face_count(mesh))
    for v in sel.verts.astype("i8").tolist():
        if v >= len(mesh.positions):
            raise el.OpError(f"Vertex {v} is not part of this mesh.")
        corners = a.vertex_corners(v)
        if len(corners) == 0:
            raise el.OpError(f"Vertex {v} belongs to no face.")
        uses = a.edge_uses[a.corner_edge[corners]]
        incoming = a.edge_uses[a.corner_edge[a.prev_corner[corners]]]
        touch = np.concatenate([uses, incoming])
        if (touch == 1).any():
            raise el.OpError(
                f"Vertex {v} is on a boundary, so the faces around it do not "
                "close into a ring. Fill the hole first, or delete the vertex."
            )
        if (touch >= 3).any():
            raise el.OpError(
                f"Vertex {v} sits on a non-manifold edge, so the faces around it "
                "have no single order. Fix that edge first."
            )
        faces = a.corner_face[corners].astype("i8")
        for f in faces[1:].tolist():
            union.union(int(faces[0]), f)
    touched = np.zeros(bm.face_count(mesh), dtype=bool)
    touched[a.corner_face[np.concatenate([a.vertex_corners(int(v)) for v in sel.verts])]] = True
    groups = [np.array(g) for g in union.groups(np.flatnonzero(touched))]
    return dis.merge_groups(mesh, groups)


def _assert_same_mesh_and_sel(
    got: tuple[bm.Mesh, el.ElementSel], want: tuple[bm.Mesh, el.ElementSel]
) -> None:
    got_mesh, got_sel = got
    want_mesh, want_sel = want
    assert np.array_equal(got_mesh.positions, want_mesh.positions)
    assert np.array_equal(got_mesh.loops, want_mesh.loops)
    assert np.array_equal(got_mesh.starts, want_mesh.starts)
    assert np.array_equal(got_mesh.material, want_mesh.material)
    assert np.array_equal(got_mesh.smooth, want_mesh.smooth)
    if want_mesh.uv is None:
        assert got_mesh.uv is None
    else:
        assert np.array_equal(got_mesh.uv, want_mesh.uv)
    assert np.array_equal(got_sel.faces, want_sel.faces)
    assert np.array_equal(got_sel.edges, want_sel.edges)
    assert np.array_equal(got_sel.verts, want_sel.verts)


def test_dissolve_faces_matches_the_old_union_find_on_several_disjoint_blocks() -> None:
    """A 6x6 grid, two disjoint 2x2 blocks selected: two separate merges, in
    the order the old ``_Union.groups`` dict-insertion produced them."""
    m = _grid(6, 6)
    sel = el.ElementSel(faces=np.array([0, 1, 6, 7, 20, 21, 26, 27]))
    _assert_same_mesh_and_sel(dis.dissolve_faces(m, sel), _old_dissolve_faces(m, sel))


def test_dissolve_faces_matches_the_old_union_find_on_boundary_faces() -> None:
    """The selection includes the grid's own outer boundary faces -- the
    dissolved outline touches the mesh's own open border, not just interior
    edges."""
    m = _grid(6, 6)
    sel = el.ElementSel(faces=np.array([0, 1, 2, 6, 7, 8, 12, 13, 14]))
    _assert_same_mesh_and_sel(dis.dissolve_faces(m, sel), _old_dissolve_faces(m, sel))


def test_dissolve_faces_matches_the_old_union_find_when_nothing_touches() -> None:
    """Every selected face is isolated from every other -- no ``union()`` call
    ever fires, so every group is a singleton and ``merge_groups`` refuses.
    Both implementations must refuse identically."""
    m = _grid(6, 6)
    sel = el.ElementSel(faces=np.array([0, 8, 16, 24]))  # no two adjacent
    with pytest.raises(el.OpError, match="Nothing there to dissolve"):
        dis.dissolve_faces(m, sel)
    with pytest.raises(el.OpError, match="Nothing there to dissolve"):
        _old_dissolve_faces(m, sel)


def test_dissolve_edges_matches_the_old_union_find_on_several_disjoint_pairs() -> None:
    m = _grid(6, 6)
    a = adj.adjacency(m)
    interior = a.edge_verts[a.edge_uses == 2]
    # A handful of scattered interior edges, none adjacent to another --
    # several independent one-edge dissolves folded into one call.
    sel = el.ElementSel(edges=interior[[0, 5, 11, 17]])
    _assert_same_mesh_and_sel(dis.dissolve_edges(m, sel), _old_dissolve_edges(m, sel))


def test_dissolve_edges_matches_the_old_union_find_on_a_chain() -> None:
    """A run of collinear edges that all merge into one long strip."""
    m = _grid(6, 1)
    a = adj.adjacency(m)
    interior = a.edge_verts[a.edge_uses == 2]
    sel = el.ElementSel(edges=interior)
    _assert_same_mesh_and_sel(dis.dissolve_edges(m, sel), _old_dissolve_edges(m, sel))


def test_dissolve_verts_matches_the_old_union_find_on_several_disjoint_vertices() -> None:
    m = _grid(6, 6)
    interior_verts = [
        i * 7 + j for i in range(1, 6) for j in range(1, 6)
    ]  # every interior vertex of a 7x7 vertex grid
    sel = el.ElementSel(verts=np.array([interior_verts[0], interior_verts[10], interior_verts[20]]))
    _assert_same_mesh_and_sel(dis.dissolve_verts(m, sel), _old_dissolve_verts(m, sel))


def test_group_by_label_matches_old_union_groups_on_random_partitions() -> None:
    """Direct comparison of the new grouping helper against the old
    ``_Union.groups`` contract, over random union-find partitions -- the part
    of this batch's change with no mesh semantics of its own to hide a bug
    behind."""
    rng = np.random.default_rng(0)
    for _trial in range(200):
        n = int(rng.integers(1, 60))
        union = _OldUnion(n)
        for _ in range(int(rng.integers(0, n * 2))):
            a_i, b_i = int(rng.integers(0, n)), int(rng.integers(0, n))
            union.union(a_i, b_i)
        subset_size = int(rng.integers(1, n + 1))
        subset = np.sort(rng.choice(n, size=subset_size, replace=False)).astype("i8")

        want = [np.array(g, dtype="i8") for g in union.groups(subset)]
        labels = np.array([union.find(i) for i in range(n)], dtype="i8")
        got = dis._group_by_label(labels, subset)

        assert len(got) == len(want)
        for g_got, g_want in zip(got, want, strict=True):
            assert np.array_equal(g_got, g_want)


def test_group_by_label_on_an_empty_subset_returns_no_groups() -> None:
    assert dis._group_by_label(np.array([0, 0, 1], dtype="i8"), np.empty(0, dtype="i8")) == []


# --- perf: the 2026-09-17 native-kernel review (batch 11) --------------------


def _big_grid(n_faces: int) -> bm.Mesh:
    import math

    side = max(1, int(round(math.sqrt(n_faces))))
    return _grid(side, side)


def test_dissolving_a_full_selection_on_a_large_mesh_finishes_well_under_the_old_time() -> None:
    """The measured case: select every face of a 200k-face mesh and dissolve
    it in one call -- what a "select all, then flatten to one n-gon" workflow
    does. The old ``_Union`` union-find measured 665 ms here (dev/measurements/
    2026-09-17-native-batch-11-candidates.md); replaced with
    ``scipy.sparse.csgraph.connected_components`` plus a vectorised grouping
    pass. 0.3 s fails the unfixed code by more than 2x and the fixed code
    passes it with more than 3x margin (measured well under 100 ms once scipy
    and the mesh's adjacency are warm)."""
    mesh = _big_grid(200_000)
    sel = el.ElementSel(faces=np.arange(bm.face_count(mesh), dtype="i8"))
    adj.adjacency(mesh)  # warm the cache, as a real drag would have already done
    # Warm scipy's own one-time import cost too -- it is a fixed process-wide
    # tax paid the first time *any* clay op reaches for it (select.linked,
    # weld, and now this), not a per-call cost this op owns.
    from scipy.sparse import coo_matrix as _warm_coo  # noqa: F401
    from scipy.sparse.csgraph import connected_components as _warm_cc  # noqa: F401

    start = time.perf_counter()
    out, out_sel = dis.dissolve_faces(mesh, sel)
    elapsed = time.perf_counter() - start

    bm.validate(out)
    assert len(out_sel.faces) == 1
    assert elapsed < 0.3, f"took {elapsed:.3f}s -- still the old union-find?"
