"""The 2026-10-03 audit's Medium findings clay-45 .. clay-54 (the Clay mesh
kernels: weld clustering, dissolve and bevel).

Each test's name is the claim, and each one fails against the code it was
written for -- the bugs are in the kernels, so every one of these drives the
kernel directly rather than going through a mode.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.spatial

from realmspinner.kernels.mesh import adjacency as adj
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import ops_bevel as ob
from realmspinner.kernels.mesh import ops_dissolve as od
from realmspinner.kernels.mesh import ops_topo as ot
from realmspinner.kernels.mesh import primitives as prim
from realmspinner.kernels.mesh import topo
from realmspinner.kernels.mesh.mesh import from_faces

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
