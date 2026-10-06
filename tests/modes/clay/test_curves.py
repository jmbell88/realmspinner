"""Bézier handles on lathe, sweep and tube (Blender-Lite plan, Phase D, kernel half).

A curve is stored as the polyline plus per-anchor ``[in, out]`` handle offsets and
flattened back into that polyline at build time, so the claims are about the two
ends of that round trip:

* **no handles is the identity** -- a document from before handles existed
  flattens, clamps and builds to the very floats it always did;
* a **known Bézier** flattens to the expected stations, within the tolerance;
* the **clamps keep handles aligned** with the anchors they belong to through
  dedupe, the y-raise and the winding flip (which swaps in and out);
* a mismatched handle list **resets to nothing** rather than refusing.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from realmspinner.kernels.mesh import curves
from realmspinner.kernels.mesh import primitives as bp

KAPPA = 0.5522847498  # a quarter circle as one cubic


def _quarter() -> tuple[list[list[float]], list]:
    points = [[1.0, 0.0], [0.0, 1.0]]
    handles = [[[0.0, 0.0], [0.0, KAPPA]], [[KAPPA, 0.0], [0.0, 0.0]]]
    return points, handles


def _bezier(points, handles, t: float) -> np.ndarray:
    p0 = np.array(points[0])
    p3 = np.array(points[1])
    p1 = p0 + np.array(handles[0][1])
    p2 = p3 + np.array(handles[1][0])
    u = 1 - t
    return u**3 * p0 + 3 * u * u * t * p1 + 3 * u * t * t * p2 + t**3 * p3


def _distance_to_polyline(point: np.ndarray, poly: np.ndarray) -> float:
    best = math.inf
    for a, b in zip(poly[:-1], poly[1:], strict=True):
        ab = b - a
        t = float(np.clip(np.dot(point - a, ab) / max(float(np.dot(ab, ab)), 1e-18), 0.0, 1.0))
        best = min(best, float(np.linalg.norm(point - (a + t * ab))))
    return best


# --- flatten ------------------------------------------------------------------


def test_the_station_cap_matches_the_primitives_ceiling() -> None:
    assert curves.MAX_STATIONS == bp.MAX_PROFILE_STATIONS == bp.MAX_PATH_POINTS


@pytest.mark.parametrize("handles", [[], None, [[[0, 0], [0, 0]]] * 3])
def test_flatten_with_no_handles_is_the_identity_on_the_polyline(handles) -> None:
    points = [[0.1, -0.5], [0.3, 0.0], [0.2, 0.5]]
    assert curves.flatten(points, handles) == points
    assert curves.flatten(points, handles, closed=True) == points, "closed adds no repeat"


def test_a_known_bezier_has_the_expected_station_count_and_stays_within_tolerance() -> None:
    points, handles = _quarter()
    tol = 0.001
    out = np.array(curves.flatten(points, handles, tol=tol))
    # n from the documented bound: n = ceil(sqrt(0.75 * M / tol)), M the larger
    # second difference of the control points.
    p0, p3 = np.array(points[0]), np.array(points[1])
    p1, p2 = p0 + handles[0][1], p3 + handles[1][0]
    second = max(np.linalg.norm(p0 - 2 * p1 + p2), np.linalg.norm(p1 - 2 * p2 + p3))
    expected = math.ceil(math.sqrt(0.75 * second / tol))
    assert len(out) == expected + 1
    assert list(out[0]) == points[0] and list(out[-1]) == points[1]
    worst = max(
        _distance_to_polyline(_bezier(points, handles, t), out) for t in np.linspace(0, 1, 400)
    )
    assert worst <= tol


def test_a_tighter_tolerance_costs_more_stations() -> None:
    points, handles = _quarter()
    assert len(curves.flatten(points, handles, tol=0.0001)) > len(
        curves.flatten(points, handles, tol=0.01)
    )


def test_a_straight_segment_between_curved_ones_gets_no_interior_points() -> None:
    points = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
    handles = [[[0, 0], [0, 0]], [[0, 0], [0, 0.5]], [[0, 0.5], [0, 0]]]
    out = curves.flatten(points, handles, tol=0.01)
    first_segment = [p for p in out if p[0] < 1.0]
    assert first_segment == [[0.0, 0.0]], "the corner-to-corner segment stays straight"
    assert len(out) > 3


def test_a_closed_curve_wraps_and_does_not_repeat_its_first_point() -> None:
    points = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    handles = [[[0, 0], [0.2, 0]]] + [[[0, 0], [0, 0]]] * 3
    open_out = curves.flatten(points, handles, closed=False, tol=0.01)
    closed_out = curves.flatten(points, handles, closed=True, tol=0.01)
    assert closed_out[0] == [0.0, 0.0] and closed_out[-1] != closed_out[0]
    assert len(closed_out) >= len(open_out)


def test_the_cap_holds_even_for_a_pathological_curve() -> None:
    points = [[0.0, 0.0], [100.0, 0.0], [0.0, 100.0]]
    handles = [[[0, 0], [500, 0]], [[-500, 0], [0, 500]], [[0, -500], [0, 0]]]
    out = curves.flatten(points, handles, closed=True, tol=1e-9, cap=64)
    assert len(out) <= 64
    assert len(curves.flatten(points, handles, closed=True, tol=1e-9)) <= curves.MAX_STATIONS


@pytest.mark.parametrize(
    "bad",
    [[[[1, 2], [3]]], [[[1, 2]]], "nope", [[[float("nan"), 0], [0, 0]]], [[[1, 2], [3, 4]]] * 5],
)
def test_handles_that_do_not_fit_reset_to_nothing(bad) -> None:
    assert curves.normalise_handles(bad, 3, 2) == []


# --- the clamps keep handles aligned -------------------------------------------


def test_a_dropped_duplicate_anchor_takes_its_handle_row_with_it() -> None:
    profile = [[0.2, -0.5], [0.3, 0.0], [0.3, 0.0], [0.2, 0.5]]  # station 2 repeats 1
    handles = [
        [[0, 0], [0, 0.1]],
        [[0, -0.1], [0, 0.1]],
        [[9, 9], [9, 9]],  # the duplicate's row: must not survive
        [[0, -0.1], [0, 0]],
    ]
    out = bp.clamp_params("lathe", {"profile": profile, "profile_handles": handles})
    assert len(out["profile"]) == 3 and len(out["profile_handles"]) == 3
    assert [9, 9] not in [side for row in out["profile_handles"] for side in row]
    assert out["profile_handles"][1] == [[0.0, -0.1], [0.0, 0.1]]


def test_the_winding_flip_swaps_in_and_out_and_reverses_the_rows() -> None:
    clockwise = [[-1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [1.0, -1.0]]
    zero = [[0, 0], [0, 0]]
    handles = [[[0.1, 0.0], [0.2, 0.0]], zero, zero, [[0.3, 0.0], [0.4, 0.0]]]
    out = bp.clamp_params("sweep", {"outline": clockwise, "outline_handles": handles})
    reversed_anchors = np.array([[1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0]])
    # The plain clamp's flip, up to the re-centring the bulge's own extent causes.
    shift = np.array(out["outline"]) - reversed_anchors
    assert np.allclose(shift, shift[0]), "the same anchors in reversed order, one common offset"
    # Anchor 3 became anchor 0, and what arrived there now leaves.
    assert out["outline_handles"][0] == [[0.4, 0.0], [0.3, 0.0]]
    assert out["outline_handles"][3] == [[0.2, 0.0], [0.1, 0.0]]


def test_a_mismatched_handle_list_resets_to_nothing_and_the_anchors_still_clamp() -> None:
    out = bp.clamp_params(
        "tube",
        {
            "path": [[0, 0, 0], [1, 0, 0], [1, 1, 0]],
            "path_handles": [[[0, 0, 0], [0, 0.1, 0]]],  # one row for three anchors
        },
    )
    assert out["path_handles"] == []
    assert len(out["path"]) == 3


def test_changing_the_anchors_without_the_handles_resets_them() -> None:
    first = bp.clamp_params("lathe", {"profile": bp.LATHE_DEFAULT_PROFILE})["profile"]
    handles = [[[0.0, 0.0], [0.0, 0.05]] for _ in first]
    kept = bp.clamp_params("lathe", {"profile": first, "profile_handles": handles})
    assert len(kept["profile_handles"]) == len(first)
    fewer = bp.clamp_params("lathe", {"profile": first[:-1], "profile_handles": handles})
    assert fewer["profile_handles"] == []


def test_handles_alone_with_no_anchors_are_only_shape_checked() -> None:
    out = bp.clamp_params("lathe", {"profile_handles": [[[0, 0], [0, 1]]]})
    assert out["profile_handles"] == [[[0.0, 0.0], [0.0, 1.0]]]
    assert bp.clamp_params("lathe", {"profile_handles": "junk"})["profile_handles"] == []


def test_the_centring_offset_is_measured_on_the_curve_not_the_anchors() -> None:
    """A bulge past the anchors moves the box the mesh is centred on, and the
    stored anchors are centred on that same box -- so what the editor draws and
    what is built agree."""
    outline = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
    bulge = [[[0, 0], [0, -0.5]], [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]]
    out = bp.clamp_params("sweep", {"outline": outline, "outline_handles": bulge})
    flat = np.array(curves.flatten(out["outline"], out["outline_handles"], closed=True))
    assert float(flat[:, 1].min() + flat[:, 1].max()) == pytest.approx(0.0, abs=1e-9)


# --- the builders -----------------------------------------------------------------


def test_a_legacy_document_builds_identical_geometry() -> None:
    legacy = bp.lathe(bp.LATHE_DEFAULT_PROFILE, 16)
    for handles in ([], [[[0, 0], [0, 0]]] * len(bp.LATHE_DEFAULT_PROFILE)):
        same = bp.lathe(bp.LATHE_DEFAULT_PROFILE, 16, profile_handles=handles)
        assert np.array_equal(same.positions, legacy.positions)
        assert np.array_equal(same.loops, legacy.loops)
    outline = bp.sweep(bp.SWEEP_DEFAULT_OUTLINE)
    assert np.array_equal(
        bp.sweep(bp.SWEEP_DEFAULT_OUTLINE, outline_handles=[]).positions, outline.positions
    )
    path = bp.tube(bp.TUBE_DEFAULT_PATH)
    assert np.array_equal(bp.tube(bp.TUBE_DEFAULT_PATH, path_handles=[]).positions, path.positions)


def test_the_defaults_carry_empty_handle_lists_and_still_build() -> None:
    for generator, key in (
        ("lathe", "profile_handles"),
        ("sweep", "outline_handles"),
        ("tube", "path_handles"),
    ):
        defaults, build = bp.GENERATORS[generator]
        assert defaults[key] == []
        assert len(build(**defaults).positions) > 0


def test_a_curved_lathe_is_smoother_than_its_polyline() -> None:
    profile = [[0.0, -0.5], [0.3, 0.0], [0.0, 0.5]]
    handles = [
        [[0, 0], [0.0, 0.0]],
        [[0.0, -0.2], [0.0, 0.2]],
        [[0.0, 0.0], [0, 0]],
    ]
    flat = bp.lathe(profile, 12)
    curved = bp.lathe(profile, 12, profile_handles=handles)
    assert len(curved.positions) > len(flat.positions)
    rings = np.unique(np.round(curved.positions[:, 1], 6))
    assert len(rings) > 3, "the curve contributed stations of its own"


def test_a_curved_sweep_and_tube_build_closed_shells() -> None:
    from realmspinner.kernels.mesh import mesh as bm

    outline = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]
    corner_handles = [[[0, 0], [0.2, 0]], [[-0.2, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]]
    swept = bp.sweep(outline, outline_handles=corner_handles)
    assert bm.face_count(swept) > bm.face_count(bp.sweep(outline))

    path = [[-0.4, 0, 0], [0.0, 0, 0], [0.4, 0, 0]]
    bend = [
        [[0, 0, 0], [0.1, 0.2, 0]],
        [[-0.1, -0.2, 0], [0.1, 0.2, 0]],
        [[-0.1, -0.2, 0], [0, 0, 0]],
    ]
    assert len(bp.tube(path, 0.05, 8, path_handles=bend).positions) > len(
        bp.tube(path, 0.05, 8).positions
    )
