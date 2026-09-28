"""Regressions for the 2026-09-26 audit's clay-mesh-ops-01..05 and
clay-mesh-uv-02: five separate frame-thread ceilings/correctness gaps across
``ops_topo.py``, ``ops_spin.py``, ``ops_subdiv.py``, ``ops_dissolve.py`` and
``uvtools.py`` (see the 2026-09-26 audit's remaining-findings list, and the
fix-pass brief for wave 5, fixer 1). One file because the six findings are
small and share no state; each test names the finding it closes.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import ops_dissolve as dis
from realmspinner.kernels.mesh import ops_spin as osp
from realmspinner.kernels.mesh import ops_subdiv as sub
from realmspinner.kernels.mesh import ops_topo as ops
from realmspinner.kernels.mesh import primitives as prim
from realmspinner.kernels.mesh import topo
from realmspinner.kernels.mesh import uvtools as ut
from realmspinner.kernels.mesh.elements import OpError

# --- clay-mesh-ops-01: inset_faces(region=True, depth != 0) --------------


def test_region_inset_depth_moves_opposite_facing_regions_each_along_its_own_normal() -> (
    None
):
    """``_inset_region`` used to pull *every* touched vertex along one global
    direction, ``_unit(face_normals(out)[faces].sum(axis=0))`` -- a single
    mean normal over the *whole* selection, exactly the shape the 2026-09-08
    audit's clay-03 already found wrong for ``extrude_faces``'s own ``offset``
    (see that fix, ``_region_offsets``, and ``test_extrude_faces_offset_is_a_
    silent_noop_when_selected_regions_normals_cancel`` right above this file's
    sibling test in ``test_ops_topo.py``). A box's -Y and +Y caps (faces 0 and
    1) do not share an edge, so inset with ``region=True`` groups them as two
    separate regions; their normals point opposite ways and summed to zero,
    so a non-zero ``depth`` silently moved neither cap at all. Fixed by
    reusing ``_region_offsets`` (the same per-region grouping ``extrude_faces``
    already uses) for the depth displacement instead of one global sum.
    """
    m = prim.box()
    out, _ = ops.inset_faces(m, el.ElementSel(faces=[0, 1]), thickness=0.05, region=True, depth=0.2)
    bottom_before = m.positions[m.loops[m.starts[0] : m.starts[1]]]
    top_before = m.positions[m.loops[m.starts[1] : m.starts[2]]]
    bottom_after = out.positions[out.loops[out.starts[0] : out.starts[1]]]
    top_after = out.positions[out.loops[out.starts[1] : out.starts[2]]]
    # Depth is along Y for both -- toward -Y for the bottom cap, +Y for the
    # top -- so if it moved at all along Y, each ring's mean Y must have
    # shifted away from the box, in its own cap's own outward direction.
    assert bottom_after[:, 1].mean() < bottom_before[:, 1].mean() - 0.05, (
        "the -Y cap silently didn't move along its own normal"
    )
    assert top_after[:, 1].mean() > top_before[:, 1].mean() + 0.05, (
        "the +Y cap silently didn't move along its own normal"
    )


# --- clay-mesh-ops-02: spin(angle=360, steps<3) ---------------------------

_PROFILE = el.ElementSel(edges=np.array([[0, 4]], dtype="i4"))


def test_spin_a_full_turn_refuses_fewer_than_three_steps() -> None:
    """A whole-turn ``spin`` closes the ring by reusing ring zero's own
    vertex indices for the last band (see ``spin``'s own docstring), which
    needs at least three distinct rings to make a non-degenerate solid --
    with ``steps=1`` there is only ring zero, so ``ring_index[k2]`` for the
    single band wraps back onto the *same* ring it started from and every
    quad gets repeated corners; with ``steps=2`` the two bands fold directly
    back onto each other. Neither was refused before this fix -- reproduced
    on a box's single-edge profile, spun a full turn.
    """
    box = prim.box()
    with pytest.raises(el.OpError, match="at least three steps"):
        osp.spin(box, _PROFILE, axis=1, angle=360.0, steps=1, center=(0.0, 0.0, 0.0))
    with pytest.raises(el.OpError, match="at least three steps"):
        osp.spin(box, _PROFILE, axis=1, angle=360.0, steps=2, center=(0.0, 0.0, 0.0))
    # Three steps make a legitimate (if coarse) solid, and a partial turn is
    # never affected by this refusal, however few steps it asks for.
    out, _ = osp.spin(box, _PROFILE, axis=1, angle=360.0, steps=3, center=(0.0, 0.0, 0.0))
    bm.validate(out)
    out2, _ = osp.spin(box, _PROFILE, axis=1, angle=180.0, steps=1, center=(0.0, 0.0, 0.0))
    bm.validate(out2)


# --- clay-mesh-ops-03: collapse walks before it refuses -------------------


def test_collapse_refuses_an_oversized_face_selection_without_walking_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``collapse`` used to build one ``np.stack``/``np.roll`` pair *per
    selected face* (the per-face Python loop at the top of the function) and
    only check the resulting ``pairs`` array's length against
    ``MAX_COLLAPSED_PAIRS`` afterwards -- 0.65s at 90k faces just to discover
    the call would be refused anyway (the 2026-09-26 audit's clay-mesh-ops-03).
    Each face's own pair count is its corner span, already known from
    ``starts`` with no loop at all, so the count -- and the refusal -- must
    happen before a single face is walked. Proven here by spying on
    ``np.roll``, the one call the per-face loop makes: with the ceiling
    lowered under a box's own 24 pairs, ``collapse`` must refuse without
    ``np.roll`` ever being called.
    """
    box = prim.box()
    monkeypatch.setattr(ops, "MAX_COLLAPSED_PAIRS", 4)
    sel = el.ElementSel(faces=np.arange(bm.face_count(box), dtype="i4"))
    calls: list[int] = []
    original_roll = np.roll

    def spy_roll(*args: object, **kwargs: object) -> np.ndarray:
        calls.append(1)
        return original_roll(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(np, "roll", spy_roll)
    with pytest.raises(el.OpError, match="past the"):
        ops.collapse(box, sel)
    assert calls == [], "the per-face pair loop ran before the ceiling refused"


# --- clay-mesh-ops-04: MAX_SUBDIVIDED_FACES stalls at its own ceiling ------


def _independent_quads(n: int, uv: bool = False) -> bm.Mesh:
    """*n* quads sharing no vertex with each other -- the same
    per-face-independent shape ``MAX_INSET_CORNERS``'s and
    ``MAX_EXTRUDE_CORNERS``'s own measurement tables use, so a whole-mesh
    subdivide's ``grown`` count is exactly ``4 * n`` with nothing shared to
    complicate it.
    """
    xs = np.arange(n, dtype="f8")
    base = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    positions = (base[None, :, :] + xs[:, None, None] * np.array([2.0, 0.0, 0.0])).reshape(-1, 3)
    loops = np.arange(4 * n, dtype="i4")
    starts = np.arange(0, 4 * n + 1, 4, dtype="i4")
    material = np.zeros(n, dtype="i4")
    smooth = np.zeros(n, dtype=bool)
    uv_arr = None
    if uv:
        quad_uv = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype="f4")
        uv_arr = np.tile(quad_uv, (n, 1))
    return bm.Mesh(
        positions=positions.astype("f4"),
        loops=loops,
        starts=starts,
        material=material,
        smooth=smooth,
        uv=uv_arr,
    )


def test_max_subdivided_faces_keeps_the_ceiling_itself_well_under_a_second() -> None:
    """``MAX_SUBDIVIDED_FACES`` (1,000,000) was ``glbimport.MAX_TRIANGLES // 2``,
    never measured against ``subdivide_topology``'s own wall clock -- the
    2026-09-26 audit's clay-mesh-ops-04 measured 1.03-1.17s *at that ceiling
    itself* (reproduced on this machine: 1.18-1.21s at 250,000 independent
    quads, grown to the 1,000,000-face ceiling), past the "well under a
    second" bar every sibling growth-op ceiling in this package is held to.
    Separately, this machine's own measurement found subdividing a uv-bearing
    mesh costs no more than a uv-less one of the same size here (539.8ms vs
    546.1ms at 500,000 grown, 838.4ms vs 742.5ms at 600,000 -- noise, not a
    multiplier) -- unlike ``ops_bevel.MAX_BEVELED_CORNERS``, this op's own
    per-corner work does not double for carrying uv, so (matching
    ``ops_clean.py``'s identical reasoning for its own ``MAX_CLEAN_CORNERS``)
    one lowered constant serves both cases rather than a halving copied from
    a different op's different cost shape.
    """
    assert sub.MAX_SUBDIVIDED_FACES <= 500_000, (
        "the ceiling must be lowered to keep subdivide() under a second at "
        "its own worst case -- see test_subdividing_a_mesh_at_the_ceiling_"
        "finishes_comfortably_under_a_second for the wall-clock proof"
    )
    assert sub.MAX_SUBDIVIDED_FACES > 0


@pytest.mark.perf
def test_subdividing_a_mesh_at_the_ceiling_finishes_comfortably_under_a_second() -> None:
    """The wall-clock half of the test above: actually run ``subdivide`` at
    ``MAX_SUBDIVIDED_FACES``'s own worst case and time it. Kept out of the
    default lane (asserts a wall-clock budget) per this fix pass's own rule.
    """
    n = sub.MAX_SUBDIVIDED_FACES // 4
    mesh = _independent_quads(n)
    faces = np.arange(n, dtype="i8")
    start = time.perf_counter()
    sub.subdivide_topology(mesh, faces)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.9, f"subdivide at the ceiling took {elapsed:.2f}s, past the budget"


# --- clay-mesh-ops-05: many just-under-ceiling concave dissolve rings ------


def _comb_row(teeth: int, y_offset: float = 0.0) -> tuple[np.ndarray, list[list[int]]]:
    """A row of *teeth* quads whose top edges alternate height -- the same
    zigzag fixture ``test_ops_dissolve.py``'s own ``_comb_row`` builds,
    duplicated here (rather than imported across test modules) so this file
    stays self-contained. Dissolving the whole row merges it into one
    concave n-gon, ``2 * teeth + 2`` corners.
    """
    n = teeth + 1
    xs = np.arange(n, dtype="f4")
    heights = np.where(np.arange(n) % 2 == 0, 1.0, 0.1).astype("f4")
    bottom = np.stack([xs, np.full(n, y_offset, dtype="f4"), np.zeros(n, dtype="f4")], axis=1)
    top = np.stack([xs, np.full(n, y_offset, dtype="f4"), heights], axis=1)
    positions = np.concatenate([bottom, top], axis=0)
    faces = [[i, i + 1, n + i + 1, n + i] for i in range(teeth)]
    return positions, faces


def _two_comb_rows(teeth: int) -> tuple[bm.Mesh, list[np.ndarray]]:
    """Two of the zigzag strips above, far enough apart in 3D that dissolving
    both in one ``merge_groups`` call produces two independent concave
    rings -- the shape the 2026-09-26 audit's clay-mesh-ops-05 found
    unbounded in total: each ring on its own under
    ``MAX_CONCAVE_DISSOLVE_RING``, so neither is refused alone, but nothing
    bounded what the two of them cost *together*.
    """
    pos_a, faces_a = _comb_row(teeth, y_offset=0.0)
    pos_b, faces_b = _comb_row(teeth, y_offset=10.0)
    n_a = len(pos_a)
    faces_b_shifted = [[i + n_a for i in f] for f in faces_b]
    positions = np.concatenate([pos_a, pos_b], axis=0)
    faces = faces_a + faces_b_shifted
    mesh = bm.Mesh(
        positions=positions,
        loops=np.array([c for f in faces for c in f], dtype="i4"),
        starts=topo.starts_from_counts([4] * (2 * teeth)),
        material=np.zeros(2 * teeth, dtype="i4"),
        smooth=np.zeros(2 * teeth, dtype=bool),
    )
    group_a = np.arange(teeth, dtype="i8")
    group_b = np.arange(teeth, 2 * teeth, dtype="i8")
    return mesh, [group_a, group_b]


def test_dissolve_refuses_the_summed_cost_of_many_just_under_ceiling_concave_rings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_refuse_concave_ring`` bounded only each ring on its own -- a ring at
    or under ``MAX_CONCAVE_DISSOLVE_RING`` skipped the concavity check
    entirely (``continue`` before ``face_normals``/``concave_faces`` even
    ran). ``ops_topo.fill_hole`` and ``ops_dissolve.merge_groups`` both hand
    it a *list* of rings from one call, so many separate concave regions
    dissolved together -- each individually just under the ceiling -- paid
    their summed O(n^2) earclip cost with nothing to refuse it (the
    2026-09-26 audit's clay-mesh-ops-05). Reproduced at a size small enough
    to run in milliseconds: with the ceiling lowered to 10, two independent
    8-corner concave zigzags (each comfortably under it alone) sum to
    ``2 * 8**2 = 128``, past a total budget of ``10**2 = 100``.
    """
    monkeypatch.setattr(dis, "MAX_CONCAVE_DISSOLVE_RING", 10)
    mesh, groups = _two_comb_rows(teeth=3)
    with pytest.raises(el.OpError, match="together"):
        dis.merge_groups(mesh, groups)
    # Below the total budget, both dissolve cleanly in the one call: two
    # 6-corner rings sum to 2 * 6**2 == 72, under the same 100 budget.
    small_mesh, small_groups = _two_comb_rows(teeth=2)
    out, sel = dis.merge_groups(small_mesh, small_groups)
    bm.validate(out)
    assert len(sel.faces) == 2


# --- clay-mesh-uv-02: overlap_faces bounds cells but not total SAT tests ---


def _stacked_uv_clusters(n_clusters: int, per_cluster: int) -> bm.Mesh:
    """*n_clusters* groups of *per_cluster* triangles whose uvs all crowd
    into one tiny box per cluster, far enough apart from each other in uv
    that each cluster's own triangles land in one grid cell of their own --
    so no single bucket ever approaches ``MAX_OVERLAP_BUCKET`` however many
    clusters there are, and no single triangle's own registration count
    approaches ``MAX_OVERLAP_REGISTRATIONS`` either. The shape the
    2026-09-26 audit's clay-mesh-uv-02 found unbounded: many cells, each
    comfortably under the per-cell ceiling, whose *summed* pairwise SAT-test
    cost (measured on this machine: 65-77us per pair, so 24 clusters of a
    few hundred triangles each cost 61.6s) is not bounded by anything.
    """
    positions: list[list[float]] = []
    faces: list[list[int]] = []
    uvs: list[list[tuple[float, float]]] = []
    for c in range(n_clusters):
        for k in range(per_cluster):
            base = len(positions)
            x, cy = float(k), float(c)
            positions.extend([[x, cy, 0.0], [x + 1, cy, 0.0], [x, cy + 1, 0.0]])
            faces.append([base, base + 1, base + 2])
            jitter = (k % 3) * 1e-4
            u0 = c * 10.0 + jitter
            uvs.append([(u0, 0.0), (u0 + 0.01, 0.0), (u0, 0.01)])
    return bm.from_faces(positions, faces, uvs)


def test_overlap_faces_refuses_many_cells_each_holding_hundreds_of_stacked_triangles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``MAX_OVERLAP_TRIANGLES`` bounds the mesh's own triangle count,
    ``MAX_OVERLAP_REGISTRATIONS`` bounds the total triangle-into-cell
    registrations, and ``MAX_OVERLAP_BUCKET`` bounds one over-full cell --
    but nothing bounded the *total* pairwise SAT-test count the loop over
    every bucket performs. Reproduced at a size small enough to run in
    milliseconds by lowering the new ceiling: three clusters of five
    triangles each keep every bucket at 5 members (nowhere near
    ``MAX_OVERLAP_BUCKET``) and every triangle's own registration count at 1
    (nowhere near ``MAX_OVERLAP_REGISTRATIONS``), but their three buckets'
    summed pair count (``3 * C(5, 2) == 30``) is past ``MAX_OVERLAP_PAIRS``
    lowered here to 20.
    """
    monkeypatch.setattr(ut, "MAX_OVERLAP_PAIRS", 20)
    mesh = _stacked_uv_clusters(n_clusters=3, per_cluster=5)
    with pytest.raises(OpError, match="pairwise overlap"):
        ut.overlap_faces(mesh)
    # Comfortably under the (lowered) budget, the same shape still runs: one
    # cluster of two triangles is one pair, nowhere near it.
    monkeypatch.setattr(ut, "MAX_OVERLAP_PAIRS", 5)
    small = _stacked_uv_clusters(n_clusters=1, per_cluster=2)
    out = ut.overlap_faces(small)
    assert out.dtype == bool
