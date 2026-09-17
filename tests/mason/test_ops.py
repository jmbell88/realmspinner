"""``mason/ops.py``: placement arithmetic, pure over TRS values and world boxes.

Every test name is a claim, written to fail against the easy wrong version of
the op it covers: an origin-based ``align``/``distribute`` that ignores box
size, a ``drop_to_ground`` that drops a pivot instead of a box bottom, a
``scatter`` that leans on the global RNG, a radial array that forgets to spin
the copies it carries around the circle.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.kernels.geom3d import math3d as m3
from warlock.kernels.geom3d.gltf import Material
from warlock.studio.mason import ops
from warlock.studio.mason import terrain as T


def _base(translation=(0.0, 0.0, 0.0), rotation=None, scale=(1.0, 1.0, 1.0)) -> ops.TRS:
    return (
        np.array(translation, dtype="f8"),
        np.array(rotation if rotation is not None else m3.quat_identity(), dtype="f8"),
        np.array(scale, dtype="f8"),
    )


def _box(lo, hi) -> tuple[np.ndarray, np.ndarray]:
    return np.array(lo, dtype="f8"), np.array(hi, dtype="f8")


# --- array_linear -------------------------------------------------------------


def test_array_linear_of_six_is_six_evenly_spaced_translations_and_the_first_is_the_base():
    base = _base(translation=(1.0, 0.0, 0.0))
    out = ops.array_linear(6, (2.0, 0.0, 0.0), base_trs=base)
    assert len(out) == 6
    np.testing.assert_array_equal(out[0][0], base[0])
    for i, (t, _r, _s) in enumerate(out):
        np.testing.assert_allclose(t, [1.0 + 2.0 * i, 0.0, 0.0])


def test_array_linear_carries_rotation_and_scale_through_unchanged():
    base = _base(rotation=m3.quat_from_axis_angle((0.0, 0.0, 1.0), 0.7), scale=(2.0, 1.0, 1.0))
    out = ops.array_linear(3, (1.0, 0.0, 0.0), base_trs=base)
    for _t, r, s in out:
        np.testing.assert_allclose(r, base[1])
        np.testing.assert_allclose(s, base[2])


# --- array_radial ---------------------------------------------------------


def test_array_radial_of_four_at_360_puts_one_at_each_quadrant():
    # Base sits one metre out along +X from the centre, at the world origin.
    base = _base(translation=(1.0, 0.0, 0.0))
    out = ops.array_radial(
        4, centre=(0.0, 0.0, 0.0), axis=(0.0, 1.0, 0.0), degrees=360.0, base_trs=base
    )
    expected = [(1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
    for (t, _r, _s), want in zip(out, expected, strict=True):
        np.testing.assert_allclose(t, want, atol=1e-9)


def test_array_radial_orients_by_default_so_each_copy_turns_to_face_outward():
    # A base facing +X (rotation from +Y to +X) placed one metre out along
    # +X: at 90 degrees around +Y it should now sit on -Z and face -Z --
    # i.e. its rotation must have picked up the same 90-degree spin its
    # position did, not stayed fixed at "faces +X".
    base = _base(translation=(1.0, 0.0, 0.0))
    out = ops.array_radial(
        4, centre=(0.0, 0.0, 0.0), axis=(0.0, 1.0, 0.0), degrees=360.0, base_trs=base
    )
    spin_90 = m3.quat_from_axis_angle((0.0, 1.0, 0.0), np.pi / 2.0)
    _t, r1, _s = out[1]
    np.testing.assert_allclose(r1, spin_90, atol=1e-9)


def test_array_radial_with_orient_false_leaves_every_rotation_equal_to_the_base():
    base_rotation = m3.quat_from_axis_angle((0.0, 0.0, 1.0), 0.3)
    base = _base(translation=(1.0, 0.0, 0.0), rotation=base_rotation)
    out = ops.array_radial(
        6,
        centre=(0.0, 0.0, 0.0),
        axis=(0.0, 1.0, 0.0),
        degrees=360.0,
        base_trs=base,
        orient=False,
    )
    for _t, r, _s in out:
        np.testing.assert_allclose(r, base_rotation)


# --- scatter --------------------------------------------------------------


def test_scatter_with_one_seed_twice_is_identical():
    base = _base()
    a = ops.scatter(20, area=(-5.0, -5.0, 5.0, 5.0), base_trs=base, seed=7)
    b = ops.scatter(20, area=(-5.0, -5.0, 5.0, 5.0), base_trs=base, seed=7)
    for (ta, ra, sa), (tb, rb, sb) in zip(a, b, strict=True):
        np.testing.assert_array_equal(ta, tb)
        np.testing.assert_array_equal(ra, rb)
        np.testing.assert_array_equal(sa, sb)


def test_scatter_with_two_seeds_is_not_identical():
    base = _base()
    a = ops.scatter(20, area=(-5.0, -5.0, 5.0, 5.0), base_trs=base, seed=1)
    b = ops.scatter(20, area=(-5.0, -5.0, 5.0, 5.0), base_trs=base, seed=2)
    translations_a = np.array([t for t, _r, _s in a])
    translations_b = np.array([t for t, _r, _s in b])
    assert not np.array_equal(translations_a, translations_b)


def test_scatter_keeps_base_height_and_places_within_area():
    base = _base(translation=(0.0, 3.0, 0.0))
    out = ops.scatter(50, area=(-2.0, -2.0, 2.0, 2.0), base_trs=base, seed=5)
    for t, _r, _s in out:
        assert t[1] == pytest.approx(3.0)
        assert -2.0 <= t[0] <= 2.0
        assert -2.0 <= t[2] <= 2.0


def test_scatter_rotate_false_keeps_every_copy_at_the_base_rotation():
    base_rotation = m3.quat_from_axis_angle((1.0, 0.0, 0.0), 0.4)
    base = _base(rotation=base_rotation)
    out = ops.scatter(10, area=(-1.0, -1.0, 1.0, 1.0), base_trs=base, seed=3, rotate=False)
    for _t, r, _s in out:
        np.testing.assert_array_equal(r, base_rotation)


# --- align ------------------------------------------------------------------


def test_align_on_min_puts_every_boxs_low_edge_on_the_lowest():
    boxes = {
        1: _box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        2: _box((5.0, 0.0, 0.0), (6.0, 1.0, 1.0)),
        3: _box((-3.0, 0.0, 0.0), (-1.0, 1.0, 1.0)),
    }
    deltas = ops.align(boxes, axis=0, mode="min")
    lowest = min(lo[0] for lo, _hi in boxes.values())
    for owner, (lo, _hi) in boxes.items():
        assert lo[0] + deltas[owner][0] == pytest.approx(lowest)


def test_align_on_centre_is_not_the_same_as_aligning_origins_when_boxes_differ_in_size():
    # Both boxes already start at the same low edge (0), so an origin- or
    # lo-based "align to centre" would see them as already lined up and
    # hand back an all-zero delta. A box-*centre* alignment must not: the
    # narrow box's centre sits at 1, the wide one's at 2, so only one of the
    # two may end up with a zero delta.
    boxes = {
        "narrow": _box((0.0, 0.0, 0.0), (2.0, 1.0, 1.0)),
        "wide": _box((0.0, 0.0, 0.0), (4.0, 1.0, 1.0)),
    }
    deltas = ops.align(boxes, axis=0, mode="centre")
    assert not all(np.allclose(d, 0.0) for d in deltas.values())


def test_align_on_max_puts_every_boxs_high_edge_on_the_highest():
    boxes = {
        1: _box((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        2: _box((0.0, 0.0, 0.0), (3.0, 0.0, 0.0)),
    }
    deltas = ops.align(boxes, axis=0, mode="max")
    highest = max(hi[0] for _lo, hi in boxes.values())
    for owner, (_lo, hi) in boxes.items():
        assert hi[0] + deltas[owner][0] == pytest.approx(highest)


def test_align_rejects_an_unknown_mode():
    boxes = {1: _box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))}
    with pytest.raises(ValueError):
        ops.align(boxes, axis=0, mode="nope")  # type: ignore[arg-type]


# --- distribute ---------------------------------------------------------


def test_distribute_of_two_items_changes_nothing():
    boxes = {
        1: _box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        2: _box((10.0, 0.0, 0.0), (11.0, 1.0, 1.0)),
    }
    assert ops.distribute(boxes, axis=0) == {}


def test_distribute_of_five_leaves_equal_gaps_between_box_edges():
    # Five boxes of different widths, the two extremes far apart and the
    # middle three bunched near the start -- distribute must spread them so
    # the *gap between edges* (not between centres or origins) is uniform.
    # Centres are already increasing a..e, so the sorted-by-centre order
    # distribute uses internally matches this dict's own order.
    boxes = {
        "a": _box((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),  # fixed: width 1
        "b": _box((1.2, 0.0, 0.0), (1.7, 0.0, 0.0)),  # width 0.5
        "c": _box((1.8, 0.0, 0.0), (3.8, 0.0, 0.0)),  # width 2
        "d": _box((4.0, 0.0, 0.0), (4.2, 0.0, 0.0)),  # width 0.2
        "e": _box((20.0, 0.0, 0.0), (21.0, 0.0, 0.0)),  # fixed: width 1
    }
    deltas = ops.distribute(boxes, axis=0)
    # The two extreme items do not move.
    assert deltas["a"][0] == pytest.approx(0.0)
    assert deltas["e"][0] == pytest.approx(0.0)

    order = ["a", "b", "c", "d", "e"]
    new_lo = {k: boxes[k][0][0] + deltas[k][0] for k in order}
    new_hi = {k: boxes[k][1][0] + deltas[k][0] for k in order}
    gaps = [new_lo[order[i + 1]] - new_hi[order[i]] for i in range(len(order) - 1)]
    assert gaps == pytest.approx(gaps[0:1] * len(gaps))


def test_distribute_ignores_the_zero_axis_and_only_moves_along_the_chosen_one():
    boxes = {
        "a": _box((0.0, 0.0, 0.0), (1.0, 5.0, 9.0)),
        "b": _box((3.0, 1.0, 2.0), (4.0, 6.0, 10.0)),
        "c": _box((10.0, 2.0, 3.0), (11.0, 7.0, 11.0)),
    }
    deltas = ops.distribute(boxes, axis=0)
    for delta in deltas.values():
        assert delta[1] == 0.0
        assert delta[2] == 0.0


# --- drop_to_ground ---------------------------------------------------------


def test_drop_to_ground_rests_a_box_bottom_on_the_plane_rather_than_its_origin():
    # The box's own translation/origin is irrelevant here -- only lo[1]
    # (the bottom of the box) matters, and it is nowhere near y=0.
    boxes = {1: _box((-1.0, 5.0, -1.0), (1.0, 9.0, 1.0))}
    deltas = ops.drop_to_ground(boxes, ground=0.0)
    new_bottom = boxes[1][0][1] + deltas[1][1]
    assert new_bottom == pytest.approx(0.0)


def test_drop_to_ground_over_a_sculpted_terrain_lands_at_height_at():
    side = 4
    heights = np.zeros((side + 1, side + 1), dtype=np.float32)
    heights[2, 2] = 3.0  # a single raised vertex, an easy-to-hit landmark
    terrain = T.Terrain(heights=heights, size_x=8.0, size_z=8.0, material=Material())

    # Cell (2, 2) sits at local (0, 3, 0) for an 8x8 terrain of side 4 --
    # see terrain.height_at's own (x/size + 0.5) * n convention.
    cx, cz = 0.0, 0.0
    boxes = {1: _box((cx - 0.1, 10.0, cz - 0.1), (cx + 0.1, 11.0, cz + 0.1))}
    deltas = ops.drop_to_ground(boxes, terrain=terrain)
    expected = T.height_at(terrain, cx, cz)
    new_bottom = boxes[1][0][1] + deltas[1][1]
    assert new_bottom == pytest.approx(expected)
    assert expected == pytest.approx(3.0)


# --- snapping ----------------------------------------------------------------


def test_snap_translation_at_step_zero_returns_the_input_unchanged():
    vec = np.array([1.23, -4.56, 7.89])
    out = ops.snap_translation(vec, 0.0)
    np.testing.assert_array_equal(out, vec)


def test_snap_translation_at_a_real_step_lands_on_the_lattice():
    out = ops.snap_translation([1.3, -1.3, 0.7], 0.5)
    np.testing.assert_allclose(out, [1.5, -1.5, 0.5])


def test_snap_rotation_at_degrees_zero_returns_the_input_unchanged():
    q = m3.quat_from_axis_angle((0.0, 1.0, 0.0), 0.3)
    out = ops.snap_rotation(q, 0.0)
    np.testing.assert_array_equal(out, q)


def test_snap_rotation_at_a_real_step_lands_on_the_lattice():
    axis = np.array([0.0, 1.0, 0.0])
    q = m3.quat_from_axis_angle(axis, np.radians(46.0))
    out = ops.snap_rotation(q, 15.0)
    axis_out = out[:3] / np.linalg.norm(out[:3])
    np.testing.assert_allclose(axis_out, axis, atol=1e-9)
    angle_out = np.degrees(2.0 * np.arctan2(np.linalg.norm(out[:3]), out[3]))
    assert angle_out == pytest.approx(45.0, abs=1e-6)


def test_snap_rotation_keeps_the_identity_at_the_identity():
    out = ops.snap_rotation(m3.quat_identity(), 15.0)
    np.testing.assert_array_equal(out, m3.quat_identity())
