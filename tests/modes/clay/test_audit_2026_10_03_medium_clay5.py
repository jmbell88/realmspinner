"""The 2026-10-03 audit's Medium findings clay-45 .. clay-54 (the Clay mesh
kernels: slides, world-space measurement, weld clustering, clean, dissolve,
bevel and spin).

Each test's name is the claim, and each one fails against the code it was
written for -- the bugs are in the kernels, so every one of these drives the
kernel directly rather than going through a mode.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.spatial

from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.kernels.mesh import adjacency as adj
from realmspinner.kernels.mesh import analyze, readiness, topo
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import ops_bevel as ob
from realmspinner.kernels.mesh import ops_clean as oc
from realmspinner.kernels.mesh import ops_dissolve as od
from realmspinner.kernels.mesh import ops_model as om
from realmspinner.kernels.mesh import ops_spin as osp
from realmspinner.kernels.mesh import ops_topo as ot
from realmspinner.kernels.mesh import primitives as prim
from realmspinner.kernels.mesh.mesh import from_faces

# --- clay-45: slides read the *original* neighbour positions ---------------------


def _strip(n_quads: int = 3, rows: int = 2) -> bm.Mesh:
    """``rows`` rows of ``n_quads + 1`` vertices, one unit apart, row-major."""
    cols = n_quads + 1
    positions = [[c - 1.5, 0.0, float(r)] for r in range(rows) for c in range(cols)]
    faces = [
        [r * cols + c, r * cols + c + 1, (r + 1) * cols + c + 1, (r + 1) * cols + c]
        for r in range(rows - 1)
        for c in range(n_quads)
    ]
    return from_faces(positions, faces)


def test_vertex_slide_of_adjacent_selected_vertices_uses_the_original_neighbour_positions() -> None:
    m = _strip()
    sel = el.ElementSel(verts=np.array([1, 2, 3], dtype="i4"))

    full, _ = om.vertex_slide(m, sel, t=1.0)
    # Each vertex lands on its lowest-indexed neighbour *as it was*.
    assert full.positions[[1, 2, 3], 0].tolist() == pytest.approx([-1.5, -0.5, 0.5])

    half, _ = om.vertex_slide(m, sel, t=0.5)
    assert half.positions[[1, 2, 3], 0].tolist() == pytest.approx([-1.0, 0.0, 1.0])


def test_edge_slide_of_two_adjacent_loops_uses_the_original_rail_positions() -> None:
    m = _strip(n_quads=3, rows=4)
    cols = 4
    # The horizontal edges of rows 1 and 2: row 2's rail toward row 1 is a
    # selected vertex, which an in-place rewrite had already moved.
    edges = [[r * cols + c, r * cols + c + 1] for r in (1, 2) for c in range(cols - 1)]
    sel = el.ElementSel(edges=np.array(edges, dtype="i4"))
    out, _ = om.edge_slide(m, sel, t=-1.0)
    row2 = out.positions[2 * cols : 3 * cols, 2]
    assert row2.tolist() == pytest.approx([1.0] * cols), "row 2 slid onto row 1's *moved* position"


# --- clay-46 and clay-49: world-space measurement ----------------------------------


def _sheared_doc() -> tuple[bd.ClayDoc, bd.Obj]:
    doc = bd.ClayDoc()
    parent = doc.add_object(
        bd.Obj(
            uid=bd.new_uid(),
            name="parent",
            mesh=prim.box((0.1, 0.1, 0.1)),
            scale=m3.vec3(3.0, 1.0, 1.0),
        )
    )
    half = np.radians(45.0) / 2.0
    child = bd.Obj(
        uid=bd.new_uid(),
        name="child",
        mesh=prim.box((1.0, 1.0, 1.0)),
        rotation=np.array([0.0, 0.0, np.sin(half), np.cos(half)]),
        parent=parent.uid,
    )
    doc.add_object(child)
    return doc, child


def _world_corners(world: np.ndarray) -> np.ndarray:
    """The eight corners of a unit box, through *world*."""
    corners = np.array(
        [[x, y, z, 1.0] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)]
    )
    return (world @ corners.T).T[:, :3]


def test_analyze_bounds_of_a_child_under_a_non_uniformly_scaled_parent_match_the_world_matrix() -> (
    None
):
    doc, child = _sheared_doc()
    world = doc.world_matrix(child.uid)
    truth = _world_corners(world)

    result = analyze.analyze(list(doc.objects), doc=doc, pairs_among=[child.uid])
    row = next(r for r in result.objects if r.uid == child.uid)
    lo, hi = row.bounds
    assert lo.tolist() == pytest.approx(truth.min(axis=0).tolist(), abs=1e-6)
    assert hi.tolist() == pytest.approx(truth.max(axis=0).tolist(), abs=1e-6)
    assert row.volume == pytest.approx(abs(np.linalg.det(world[:3, :3])), rel=1e-5)


def test_readiness_scale_check_measures_the_sheared_world_box() -> None:
    doc, child = _sheared_doc()
    world = doc.world_matrix(child.uid)
    truth = _world_corners(world)
    # The parent box is 0.1 m, so the child decides the document's largest extent.
    bounds = readiness._world_bounds([readiness._evaluated_world(child, doc)])
    assert bounds is not None
    assert bounds[1].tolist() == pytest.approx(truth.max(axis=0).tolist(), abs=1e-6)


def _box_with_one_face_reversed() -> bm.Mesh:
    box = prim.box()
    loops = box.loops.copy()
    first, last = int(box.starts[0]), int(box.starts[1])
    loops[first:last] = loops[first:last][::-1]
    return bm.Mesh(
        positions=box.positions,
        loops=loops,
        starts=box.starts,
        material=box.material,
        smooth=box.smooth,
        uv=None,
    )


def test_analyze_reports_no_volume_for_a_closed_mesh_with_a_flipped_face() -> None:
    obj = bd.Obj(uid=bd.new_uid(), name="flipped", mesh=_box_with_one_face_reversed())
    row = analyze.analyze([obj]).objects[0]
    assert row.volume is None, f"a mis-wound box reported volume {row.volume}"
    assert not row.closed

    sound = analyze.analyze([bd.Obj(uid=bd.new_uid(), name="box", mesh=prim.box())]).objects[0]
    assert sound.closed and sound.volume == pytest.approx(1.0)


# --- clay-47 / clay-52 / clay-53: weld clustering ----------------------------------


class _PairSpy(scipy.spatial.cKDTree):
    """Records the largest pair array a KD-tree handed back."""

    largest = 0

    def query_pairs(self, r, *args, **kwargs):  # type: ignore[override]
        out = super().query_pairs(r, *args, **kwargs)
        type(self).largest = max(type(self).largest, len(out))
        return out


@pytest.fixture
def pair_spy(monkeypatch: pytest.MonkeyPatch) -> type[_PairSpy]:
    _PairSpy.largest = 0
    monkeypatch.setattr(scipy.spatial, "cKDTree", _PairSpy)
    return _PairSpy


def _soup(points: np.ndarray) -> bm.Mesh:
    n = len(points)
    faces = [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(n // 3)]
    return from_faces(points.tolist(), faces)


def test_survey_of_thousands_of_coincident_vertices_does_not_build_a_quadratic_pair_array(
    pair_spy: type[_PairSpy],
) -> None:
    n = 4_800
    m = _soup(np.zeros((n, 3)))
    survey = oc.survey(m)
    assert survey.coincident_vertices == n - 1
    # m(m-1)/2 = 11.5 million pairs before the fix.
    assert pair_spy.largest < 200_000


def test_weld_with_a_distance_larger_than_the_selection_does_not_build_a_quadratic_pair_array(
    pair_spy: type[_PairSpy],
) -> None:
    rng = np.random.default_rng(3)
    points = rng.uniform(0.0, 1.0, size=(4_800, 3))
    # A weld distance far larger than the selection's extent: every vertex is
    # within eps of every other, the n(n-1)/2 worst case.
    labels = ot._clusters(points, 1.0e3)
    assert labels.max() == 0 and len(labels) == len(points)
    assert pair_spy.largest < 200_000

    out, _ = ot.weld(_soup(points), el.empty(), eps=1.0e3)
    bm.validate(out)


def test_weld_exact_path_does_not_chain_points_farther_than_eps_apart() -> None:
    eps = 1.0
    spacing = 0.9 * eps
    points = np.zeros((50, 3))
    points[:, 0] = np.arange(50) * spacing
    labels = ot._clusters(points, eps)
    n_clusters = int(labels.max()) + 1
    assert n_clusters > 10, f"a {spacing * 49:.0f}-long chain collapsed to {n_clusters} cluster(s)"
    for label in range(n_clusters):
        member = points[labels == label, 0]
        assert member.max() - member.min() <= eps + 1e-9


def test_weld_exact_path_still_joins_a_near_pair_and_leaves_a_far_one() -> None:
    points = np.array([[0.0, 0, 0], [0.5, 0, 0], [5.0, 0, 0]])
    labels = ot._clusters(points, 1.0)
    assert labels[0] == labels[1] != labels[2]


# --- clay-48: clean keeps the back face of a two-sided card -------------------------


def test_clean_keeps_the_back_face_of_an_opposite_wound_face_pair() -> None:
    positions = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    card = from_faces(positions, [[0, 1, 2, 3], [3, 2, 1, 0]])
    assert oc.survey(card).duplicate_faces == 0
    out, report = oc.clean(card, recalc=False)
    assert report.duplicate_removed == 0
    assert bm.face_count(out) == 2


def test_clean_still_removes_a_same_wound_duplicate_whatever_corner_it_starts_on() -> None:
    positions = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    doubled = from_faces(positions, [[0, 1, 2, 3], [2, 3, 0, 1]])
    out, report = oc.clean(doubled, recalc=False)
    assert report.duplicate_removed == 1
    assert bm.face_count(out) == 1


# --- clay-50: dissolving many small groups ----------------------------------------


def _pair_groups(n: int) -> tuple[bm.Mesh, list[np.ndarray]]:
    m = prim.grid((1.0, 1.0), n)
    assert bm.face_count(m) == n * n
    groups = [
        np.array([r * n + 2 * j, r * n + 2 * j + 1], dtype="i8")
        for r in range(n)
        for j in range(n // 2)
    ]
    return m, groups


def test_dissolve_faces_cost_does_not_grow_with_group_count_times_mesh_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every group used to ask ``region_boundary_corners`` for its own border,
    and each of those calls does whole-mesh work, so a dissolve of G groups on
    a mesh of C corners cost G x C. Counting the whole-mesh calls is the
    stable form of that claim: it must not grow with the group count."""
    calls: list[int] = []
    real = topo.region_boundary_corners

    def spy(mesh, faces):
        calls.append(1)
        return real(mesh, faces)

    monkeypatch.setattr(topo, "region_boundary_corners", spy)
    counts = {}
    for n in (10, 20):
        m, groups = _pair_groups(n)
        calls.clear()
        out, sel = od.merge_groups(m, groups)
        bm.validate(out)
        assert len(sel.faces) == len(groups)
        counts[n] = len(calls)
    assert counts[20] == counts[10], f"whole-mesh border calls grew with groups: {counts}"


def test_the_batched_region_border_matches_the_per_group_one() -> None:
    m, groups = _pair_groups(8)
    groups += [np.array([0, 1, 8, 9, 2], dtype="i8")]  # overlaps, and is bigger
    batched = topo.region_boundary_corners_by_group(m, groups)
    for group, got in zip(groups, batched, strict=True):
        assert got.tolist() == topo.region_boundary_corners(m, group).tolist()


# --- clay-51: bevel bounds the corners the selection touches ------------------------


def test_bevel_refuses_a_select_all_selection_whose_touched_corners_would_stall_the_frame() -> None:
    m = prim.torus(segments=80, sides=40)  # 12,800 corners, far under the mesh-size ceiling
    assert len(m.loops) < ob.MAX_BEVELED_CORNERS
    a = adj.adjacency(m)
    every = el.ElementSel(edges=a.edge_verts[a.edge_uses == 2])
    with pytest.raises(el.OpError, match="touch"):
        ob.bevel_edges(m, every, width=0.001)

    # One edge on the same mesh is still the cheap case it always was.
    one = el.ElementSel(edges=a.edge_verts[a.edge_uses == 2][:1])
    out, _ = ob.bevel_edges(m, one, width=0.001)
    bm.validate(out)


def test_bevel_select_all_stays_reachable_on_an_ordinary_mesh() -> None:
    box = prim.box()
    a = adj.adjacency(box)
    out, _ = ob.bevel_edges(box, el.ElementSel(edges=a.edge_verts), width=0.05)
    assert bm.face_count(out) == 26


# --- clay-54: spin shares a profile vertex that sits on the axis -------------------


def test_spin_shares_one_vertex_for_a_profile_point_on_the_axis() -> None:
    quad = from_faces([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], [[0, 1, 2, 3]])
    top = el.ElementSel(edges=np.array([[2, 3]], dtype="i4"))  # (1,1,0) -> (0,1,0), the pole
    out, sel = osp.spin(quad, top, axis=1, angle=360.0, steps=8, center=(0.0, 0.0, 0.0))
    bm.validate(out)
    # 4 original + 7 copies of the rim vertex; the pole is one vertex, not 8.
    assert len(out.positions) == 4 + 7
    assert len(sel.faces) == 8
    for f in sel.faces.tolist():
        corners = bm.face(out, f).tolist()
        assert len(corners) == 3, "a band touching the pole is a triangle, not a degenerate quad"
        assert len(set(corners)) == 3


def test_screw_shares_an_on_axis_vertex_only_when_it_does_not_lift() -> None:
    quad = from_faces([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], [[0, 1, 2, 3]])
    top = el.ElementSel(edges=np.array([[2, 3]], dtype="i4"))
    flat, _ = osp.screw(quad, top, axis=1, angle=180.0, steps=4, height=0.0, center=(0, 0, 0))
    assert len(flat.positions) == 4 + 4  # the pole shared, the rim copied four times
    lifted, _ = osp.screw(quad, top, axis=1, angle=180.0, steps=4, height=2.0, center=(0, 0, 0))
    assert len(lifted.positions) == 4 + 8  # the pole climbs the axis: distinct vertices
