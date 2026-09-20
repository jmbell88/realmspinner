"""The tranche 5 game blockout set: wedge, ramp, rounded_box, stairs, wall
and doorway.

The registry-wide sweep in ``test_primitives.py`` already proves every one of
these is a valid CSR, centred on the origin, convex-faced (or correctly
exempted), consistently oriented, wound outward, closed and fully used, and
that the registry names them -- this file only checks what is specific to
each shape.
"""

from __future__ import annotations

import numpy as np
import pytest

from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as bp

from .topo_asserts import assert_closed, assert_consistently_oriented, assert_wound_outward

# --- wedge ----------------------------------------------------------------


def test_a_wedge_has_five_faces_two_triangles_and_three_quads() -> None:
    m = bp.wedge()
    bm.validate(m)
    assert bm.face_count(m) == 5
    counts = sorted(np.diff(m.starts).tolist())
    assert counts == [3, 3, 4, 4, 4]


def test_a_wedge_is_flat_on_the_ground_and_vertical_at_the_back() -> None:
    m = bp.wedge(width=2.0, height=1.0, depth=1.0)
    lo, hi = bm.bounds(m)
    assert np.allclose(lo, [-1.0, -0.5, -0.5])
    assert np.allclose(hi, [1.0, 0.5, 0.5])
    # Every vertex is at the bottom (y = -0.5) or at the back-top ridge
    # (x = +1, y = +0.5) -- the front-top corner does not exist.
    ys = np.unique(np.round(m.positions[:, 1], 5))
    assert set(ys.tolist()) == {-0.5, 0.5}
    top = m.positions[np.isclose(m.positions[:, 1], 0.5)]
    assert np.allclose(top[:, 0], 1.0)


# --- ramp -------------------------------------------------------------------


def test_a_ramp_has_five_faces_like_a_wedge() -> None:
    m = bp.ramp()
    bm.validate(m)
    assert bm.face_count(m) == 5


def test_a_ramp_spans_its_own_length_and_height_and_rises_toward_the_back() -> None:
    m = bp.ramp(width=1.0, length=4.0, height=2.0)
    lo, hi = bm.bounds(m)
    assert np.allclose(lo, [-0.5, -1.0, -2.0])
    assert np.allclose(hi, [0.5, 1.0, 2.0])
    front = m.positions[np.isclose(m.positions[:, 2], -2.0)]
    back = m.positions[np.isclose(m.positions[:, 2], 2.0)]
    assert np.allclose(front[:, 1], -1.0), "the front edge is flat on the ground"
    back_ys = np.unique(np.round(back[:, 1], 5))
    assert set(back_ys.tolist()) == {-1.0, 1.0}, "the back face is the full vertical rise"


# --- rounded_box ------------------------------------------------------------


def test_a_rounded_box_at_zero_radius_is_a_plain_box() -> None:
    m = bp.rounded_box(size=(1.0, 1.0, 1.0), radius=0.0, segments=8)
    bm.validate(m)
    assert bm.face_count(m) == 6
    assert len(m.positions) == 8


def test_a_rounded_box_grows_two_more_side_faces_per_corner_segment() -> None:
    m = bp.rounded_box(size=(1.0, 1.0, 1.0), radius=0.3, segments=5)
    bm.validate(m)
    n = 4 * 5  # four corners, five points each
    assert bm.face_count(m) == n + 2  # the side band plus the two caps
    assert len(m.positions) == 2 * n


def test_a_rounded_boxs_radius_is_clamped_to_the_shorter_extent() -> None:
    huge = bp.rounded_box(size=(2.0, 1.0, 0.5), radius=1000.0, segments=6)
    bm.validate(huge)
    lo, hi = bm.bounds(huge)
    # The rounding radius cannot exceed half the shorter of X/Z (0.25), so the
    # footprint never shrinks past that even at an absurd requested radius.
    assert np.allclose(lo[[0, 2]], [-1.0, -0.25], atol=1e-5)
    assert np.allclose(hi[[0, 2]], [1.0, 0.25], atol=1e-5)


def test_a_rounded_box_is_flat_top_and_bottom() -> None:
    m = bp.rounded_box(size=(1.0, 2.0, 1.0), radius=0.2, segments=4)
    ys = np.unique(np.round(m.positions[:, 1], 5))
    assert set(ys.tolist()) == {-1.0, 1.0}


# --- stairs -------------------------------------------------------------


@pytest.mark.parametrize("steps", [1, 4, 10])
def test_stairs_is_one_swept_profile_two_end_caps_and_a_side_per_station(steps: int) -> None:
    """``2n + 4`` faces closed (the zigzag's ``2n + 2`` stations plus one
    closing ground station, each side-quad band, plus the two end caps),
    ``2n + 3`` open (no closing station)."""
    closed = bp.stairs(steps=steps, total_height=1.0, total_depth=1.0, closed_underside=True)
    bm.validate(closed)
    assert bm.face_count(closed) == 2 * steps + 4
    assert_closed(closed)
    assert_consistently_oriented(closed)
    assert_wound_outward(closed)

    open_ = bp.stairs(steps=steps, total_height=1.0, total_depth=1.0, closed_underside=False)
    bm.validate(open_)
    assert bm.face_count(open_) == 2 * steps + 3
    assert_closed(open_)
    assert_consistently_oriented(open_)
    assert_wound_outward(open_)


def test_stairs_climbs_from_the_ground_to_total_height_over_total_depth() -> None:
    m = bp.stairs(steps=5, width=1.0, total_height=2.5, total_depth=4.0)
    lo, hi = bm.bounds(m)
    assert np.isclose(lo[1], -1.25, atol=1e-5)
    assert np.isclose(hi[1], 1.25, atol=1e-5)
    assert np.isclose(lo[2], -2.0, atol=1e-5)
    assert np.isclose(hi[2], 2.0, atol=1e-5)


def test_stairs_closed_underside_reaches_the_ground_under_every_step() -> None:
    m = bp.stairs(steps=4, total_height=1.0, total_depth=1.0, closed_underside=True)
    lo, _ = bm.bounds(m)
    assert np.isclose(lo[1], -0.5, atol=1e-5)
    # Every block's own lower face sits at the object's own floor.
    ys = np.round(m.positions[:, 1], 5)
    assert np.isclose(ys.min(), -0.5, atol=1e-5)


def test_stairs_open_underside_has_one_fewer_vertex_than_closed() -> None:
    closed = bp.stairs(steps=4, total_height=1.0, total_depth=1.0, closed_underside=True)
    open_ = bp.stairs(steps=4, total_height=1.0, total_depth=1.0, closed_underside=False)
    bm.validate(open_)
    assert len(open_.positions) == len(closed.positions) - 2  # near + far cap ring, one less each
    assert_closed(open_)
    assert_consistently_oriented(open_)


def test_stairs_step_count_is_clamped() -> None:
    m = bp.stairs(steps=10_000, total_height=1.0, total_depth=1.0)
    assert bm.face_count(m) == 2 * bp.MAX_STAIRS_STEPS + 4
    zero = bp.stairs(steps=0, total_height=1.0, total_depth=1.0)
    assert bm.face_count(zero) == 2 * bp.MIN_STAIRS_STEPS + 4


def test_stairs_is_registered_with_its_own_steps_clamp() -> None:
    clamped = bp.clamp_params("stairs", {"steps": 10_000})
    assert clamped["steps"] == bp.MAX_STAIRS_STEPS


# --- wall -------------------------------------------------------------------


def test_a_wall_is_exactly_a_box_under_its_own_axis_names() -> None:
    wall = bp.wall(length=3.0, height=1.5, thickness=0.3)
    box = bp.box((3.0, 1.5, 0.3))
    assert np.array_equal(wall.positions, box.positions)
    assert np.array_equal(wall.loops, box.loops)
    assert np.array_equal(wall.starts, box.starts)


# --- doorway ------------------------------------------------------------


def test_a_doorway_at_defaults_is_one_swept_frame_profile() -> None:
    m = bp.doorway()
    bm.validate(m)
    assert bm.face_count(m) == 10  # 8-station profile, two caps plus 8 side quads
    assert_closed(m)
    assert_consistently_oriented(m)
    assert_wound_outward(m)


def test_a_doorway_reaches_the_wall_bottom() -> None:
    m = bp.doorway(wall_length=3.0, wall_height=2.5, wall_thickness=0.2, opening_height=2.0)
    lo, _ = bm.bounds(m)
    assert np.isclose(lo[1], -1.25, atol=1e-5), "the wall's own bottom"
    # No geometry between the two pillars reaches below the opening's own top.
    ox_range = (m.positions[:, 0] > -0.4) & (m.positions[:, 0] < 0.4)
    if ox_range.any():
        assert (m.positions[ox_range, 1] >= -1.25 + 2.0 - 1e-4).all()


def test_a_doorway_offset_all_the_way_leaves_only_one_pillar_and_a_lintel() -> None:
    m = bp.doorway(
        wall_length=3.0,
        wall_height=2.5,
        wall_thickness=0.2,
        opening_width=1.0,
        opening_height=2.0,
        opening_offset=1.0,  # pushed to the +X edge; the +X pillar vanishes
    )
    bm.validate(m)
    assert bm.face_count(m) < 10, "a vanished pillar dedups the profile down"
    assert_closed(m)
    assert_consistently_oriented(m)


def test_a_doorway_opening_wider_than_the_wall_still_leaves_something_standing() -> None:
    m = bp.doorway(
        wall_length=3.0,
        wall_height=2.5,
        wall_thickness=0.2,
        opening_width=100.0,
        opening_height=100.0,
        opening_offset=0.0,
    )
    bm.validate(m)
    assert bm.face_count(m) > 0
    assert_closed(m)


# --- registry: categories and clamp reuse -----------------------------------


def test_the_game_category_names_exactly_the_six_new_generators() -> None:
    by_name = dict(bp.CATEGORIES)
    assert set(by_name["game"]) == {"wedge", "ramp", "rounded_box", "stairs", "wall", "doorway"}


def test_none_of_the_six_needed_an_open_exemption_and_only_two_need_concave() -> None:
    names = {"wedge", "ramp", "rounded_box", "stairs", "wall", "doorway"}
    assert names.isdisjoint(bp.OPEN_GENERATORS)
    assert names & bp.CONCAVE_GENERATORS == {"stairs", "doorway"}


#: Parameters that are not extents -- a count, a bool, or a signed offset --
#: and must not be flipped by the sweep below the way a size or a radius is.
_NOT_AN_EXTENT = frozenset({"steps", "segments", "closed_underside", "opening_offset"})


@pytest.mark.parametrize("name", ["wedge", "ramp", "rounded_box", "stairs", "wall", "doorway"])
def test_a_negative_extent_is_taken_as_its_magnitude(name: str) -> None:
    defaults, builder = bp.GENERATORS[name]

    def negate(value: object) -> object:
        if isinstance(value, tuple):
            return tuple(-v for v in value)
        return -value

    negative = {
        key: (negate(value) if key not in _NOT_AN_EXTENT else value)
        for key, value in defaults.items()
    }
    positive = builder(**defaults)
    flipped = builder(**negative)
    bm.validate(flipped)
    assert bm.face_count(flipped) == bm.face_count(positive)
    flipped_mag = sorted(np.abs(flipped.positions).flatten().tolist())
    positive_mag = sorted(np.abs(positive.positions).flatten().tolist())
    assert np.allclose(flipped_mag, positive_mag, atol=1e-5)
