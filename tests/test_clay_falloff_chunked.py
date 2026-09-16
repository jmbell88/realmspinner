"""H10 / 2026-09-13: the falloff distance search is a ``cKDTree`` query.

Replaced the chunked brute-force broadcast this module used to pin
bit-identical (`dev/measurements/2026-09-13-native-batch-10-candidates.md`
§2): 1343 ms to 66 ms at the old cap's 40M pairs, on this machine. The tree's
distances agree with the broadcast to 1e-9, not exactly, so this module now
pins ``allclose`` against a kept brute-force reference rather than bit
identity -- except for the selected vertices themselves, which must still
land at distance exactly 0 and weight exactly 1, because those are set
explicitly rather than measured (``proportional_set`` does
``weights[selected] = 1.0``) and a tolerance there would be hiding a real
regression, not tolerating a numerics difference.

``MAX_FALLOFF_PAIRS`` (a time cap on the ``selected x vertices`` product) is
replaced by ``MAX_FALLOFF_VERTICES`` (a cap on the mesh's own vertex count):
the tree's cost is dominated by how many vertices the query touches, not by
how many are selected, so the pair product no longer describes what is
expensive. See ``drag.MAX_FALLOFF_VERTICES``'s docstring for the measurements
the new number is chosen from.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.clay import drag as bd


def _reference(
    positions: np.ndarray, selected: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    """The pre-tree ``proportional_set`` body, verbatim: the one-shot
    broadcast this module's output is now measured against with ``allclose``,
    not ``array_equal``."""
    delta = positions[None, :, :] - positions[selected][:, None, :]
    distance = np.sqrt((delta**2).sum(axis=2)).min(axis=0)
    weights = bd.falloff(distance, radius)
    weights[selected] = 1.0
    keep = np.flatnonzero(weights > 0.0)
    return keep.astype("i4"), weights[keep]


def _mesh(rng: np.random.Generator, verts: int, sel: int) -> tuple[np.ndarray, np.ndarray]:
    positions = rng.normal(size=(verts, 3)).astype("f8")
    selected = rng.choice(verts, size=sel, replace=False).astype("i8")
    return positions, selected


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize(
    ("verts", "sel"),
    [
        (50, 3),
        (50, 14),
        (50, 15),
        (50, 26),
        (200, 41),
    ],
)
def test_the_tree_matches_the_direct_broadcast_closely(seed, verts, sel) -> None:
    positions, selected = _mesh(np.random.default_rng(seed), verts, sel)
    for radius in (0.5, 1.5, 10.0):
        got_verts, got_weights = bd.proportional_set(positions, selected, radius)
        want_verts, want_weights = _reference(positions, selected, radius)
        assert np.array_equal(got_verts, want_verts), (seed, radius)
        # allclose, never array_equal: a KD-tree agrees with the broadcast to
        # 1e-9, not bit for bit (§2 of the 2026-09-13 measurement).
        assert np.allclose(got_weights, want_weights, atol=1e-9), (seed, radius)


def test_selected_vertices_land_at_distance_zero_and_weight_one_exactly() -> None:
    """Not a tolerance question: ``proportional_set`` sets these explicitly,
    so any drift here is a real regression, never a numerics difference."""
    rng = np.random.default_rng(4)
    positions, selected = _mesh(rng, 500, 30)
    verts, weights = bd.proportional_set(positions, selected, 2.0)
    index = {int(v): i for i, v in enumerate(verts)}
    for s in selected.tolist():
        assert index[s] is not None
        assert float(weights[index[s]]) == 1.0


def test_the_vertex_cap_still_declines_rather_than_searching_forever() -> None:
    """``MAX_FALLOFF_VERTICES`` survives as the cap on the search, now sized
    to the mesh's own vertex count rather than the ``selected x vertices``
    product: the tree's cost is dominated by how many vertices the query
    touches, not by how many of them are selected."""
    positions = np.zeros((10, 3))
    saved, bd.MAX_FALLOFF_VERTICES = bd.MAX_FALLOFF_VERTICES, 1
    try:
        verts, weights = bd.proportional_set(positions, np.array([0, 1]), 1.0)
    finally:
        bd.MAX_FALLOFF_VERTICES = saved
    assert verts.tolist() == [0, 1]
    assert weights.tolist() == [1.0, 1.0]


def test_the_set_comes_back_in_vertex_index_order_with_selected_at_weight_one() -> None:
    """What the docstring now states, pinned: the order is vertex-index order
    -- ``flatnonzero``'s -- and the selected vertices are *in* the set at
    weight 1, not sorted to its front."""
    positions = np.array(
        [[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]], dtype="f8"
    )
    verts, weights = bd.proportional_set(positions, np.array([3]), 1.5)
    assert verts.tolist() == [2, 3], "index order, selection last here"
    assert float(weights[1]) == 1.0
    assert 0.0 < float(weights[0]) < 1.0


def test_a_large_selection_on_a_large_mesh_still_gets_a_soft_falloff() -> None:
    """Regression for the 2026-09-13 rewrite: under the old ``selected x
    vertices`` product cap (40M), a 200k-vert mesh with just 201 selected
    already exceeded it (200_000 * 201 = 40,200,200 > 40,000,000) and fell
    back to a hard selection -- so a modest selection on an ordinary imported
    mesh silently lost proportional editing. The new cap is on vertex count
    alone (`MAX_FALLOFF_VERTICES`, default 300_000), which this mesh is well
    under, so the falloff must come back soft."""
    rng = np.random.default_rng(7)
    verts = 200_000
    positions = rng.normal(size=(verts, 3)).astype("f8")
    selected = rng.choice(verts, size=201, replace=False).astype("i8")

    kept, weights = bd.proportional_set(positions, selected, radius=5.0)

    assert len(kept) > len(selected), "must pick up neighbours beyond the selection itself"
    assert not np.allclose(weights, 1.0), "must not degrade to a hard selection"
    assert np.any((weights > 0.0) & (weights < 1.0)), "must actually be soft somewhere"
