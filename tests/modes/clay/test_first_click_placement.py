"""Where a hand-placed shape lands: the pure arithmetic in ``clay/placement.py``.

Each test's name is the claim it makes. The module is new, so against the code
before it these fail on the import; the claims that *changed behaviour* (the
human door grounding a shape, the camera framing the first one) are pinned in
``test_first_click_place_primitive.py`` and fail there on their assertions.
"""

from __future__ import annotations

import numpy as np
import pytest

from realmspinner.studio.modes.clay import placement as pl
from realmspinner.studio.viewer.camera import Camera


def box(lo, hi):
    return np.array(lo, dtype="f8"), np.array(hi, dtype="f8")


UNIT = box((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))


def at(shape, t):
    """*shape* moved by translation *t*: the world box it ends up occupying."""
    return shape[0] + t, shape[1] + t


# -- the ground -----------------------------------------------------------------


def test_a_shape_in_an_empty_document_lands_at_the_origin_with_its_lowest_point_on_the_ground():
    t = pl.spawn_translation(UNIT, [])
    assert t[0] == 0.0 and t[2] == 0.0
    assert t[1] == 0.5
    assert at(UNIT, t)[0][1] == 0.0, "the lowest point is exactly on y = 0, not near it"


def test_a_shape_whose_pivot_is_not_its_middle_still_has_its_lowest_point_on_the_ground():
    # A cone-like mesh built from y = -0.2 up to 0.8 (the pivot is not centred).
    lopsided = box((-0.3, -0.2, -0.3), (0.3, 0.8, 0.3))
    t = pl.spawn_translation(lopsided, [])
    assert at(lopsided, t)[0][1] == 0.0
    assert at(lopsided, t)[1][1] == pytest.approx(1.0)


# -- beside, never inside ---------------------------------------------------------


def test_the_second_shape_lands_on_the_plus_x_side_clear_of_the_first():
    first = at(UNIT, pl.spawn_translation(UNIT, []))
    t = pl.spawn_translation(UNIT, [first])
    assert t[0] > 0.0 and t[2] == 0.0, "the first free ring starts from +X"
    # 1 m wide boxes need a 1.15 m centre distance; the search steps by 0.5 m.
    assert t[0] == 1.5
    assert t[1] == 0.5


def test_a_new_shape_keeps_the_gap_from_every_existing_one_in_plan_view():
    others = [at(UNIT, np.array([x, 0.5, 0.0])) for x in (0.0, 1.5, -1.5)]
    t = pl.spawn_translation(UNIT, others)
    new = at(UNIT, t)
    for lo, hi in others:
        separated = (
            new[0][0] >= hi[0] + pl.GAP - 1e-9
            or new[1][0] <= lo[0] - pl.GAP + 1e-9
            or new[0][2] >= hi[2] + pl.GAP - 1e-9
            or new[1][2] <= lo[2] - pl.GAP + 1e-9
        )
        assert separated, (lo, hi, new)


def test_the_spot_is_on_the_eighth_of_a_metre_grid():
    # A selected object off the grid would otherwise carry its fractions across.
    near = box((2.03, 0.0, 1.07), (3.03, 1.0, 2.07))
    t = pl.spawn_translation(UNIT, [near], beside=near)
    for axis in (0, 2):
        assert t[axis] / pl.SNAP == pytest.approx(round(t[axis] / pl.SNAP)), t


def test_with_one_object_selected_the_search_starts_from_it_not_from_the_origin():
    far = at(UNIT, np.array([20.0, 0.5, 0.0]))
    origin = at(UNIT, np.array([0.0, 0.5, 0.0]))
    t = pl.spawn_translation(UNIT, [origin, far], beside=far)
    assert t[0] == pytest.approx(21.5), "next to the selected box, on its +X side"
    # And without a selection the same document puts it by the origin instead.
    assert pl.spawn_translation(UNIT, [origin, far])[0] == 1.5


def test_shapes_added_one_after_another_never_overlap_and_all_stand_on_the_ground():
    rng = np.random.default_rng(7)
    placed: list = []
    for _ in range(80):
        half = rng.uniform(0.1, 1.2, size=3)
        shape = box(-half, half)
        t = pl.spawn_translation(shape, placed)
        new = at(shape, t)
        assert new[0][1] == 0.0
        for lo, hi in placed:
            clear = (
                new[0][0] >= hi[0] + pl.GAP - 1e-9
                or new[1][0] <= lo[0] - pl.GAP + 1e-9
                or new[0][2] >= hi[2] + pl.GAP - 1e-9
                or new[1][2] <= lo[2] - pl.GAP + 1e-9
            )
            assert clear
        placed.append(new)


def test_a_box_floating_above_the_ground_still_keeps_a_new_shape_from_landing_under_it():
    roof = box((-1.0, 3.0, -1.0), (1.0, 3.2, 1.0))
    t = pl.spawn_translation(UNIT, [roof])
    assert abs(t[0]) > 0.5 or abs(t[2]) > 0.5, "landed under the floating box"


def test_non_finite_boxes_are_ignored_rather_than_poisoning_the_search():
    bad = box((np.nan, 0, 0), (1, 1, 1))
    t = pl.spawn_translation(UNIT, [bad])
    assert np.isfinite(t).all()
    assert t[0] == 0.0 and t[2] == 0.0


# -- resting on top ---------------------------------------------------------------


def test_with_no_free_spot_in_reach_the_shape_stands_on_top_of_what_is_under_it():
    slab = box((-60.0, 0.0, -60.0), (60.0, 2.0, 60.0))
    t = pl.spawn_translation(UNIT, [slab])
    assert t[0] == 0.0 and t[2] == 0.0
    assert at(UNIT, t)[0][1] == pytest.approx(2.0), "rests on the slab, not inside it"


def test_support_height_is_the_top_of_the_highest_box_under_the_footprint():
    low = box((-1, 0, -1), (1, 1, 1))
    high = box((0.5, 0, 0.5), (2, 4, 2))
    away = box((10, 0, 10), (12, 9, 12))
    got = pl.support_height(np.array([0.0, 0.0]), np.array([1.0, 1.0]), [low, high, away])
    assert got == 4.0


def test_support_height_is_the_ground_when_the_footprint_only_touches_a_side():
    wall = box((1.0, 0, -1), (2.0, 3, 1))
    assert pl.support_height(np.array([0.0, -0.5]), np.array([1.0, 0.5]), [wall]) == 0.0


# -- the frustum ------------------------------------------------------------------


def _view(distance=6.0, aspect=1.5):
    camera = Camera(aspect=aspect)
    camera.distance = distance
    return camera.projection() @ camera.view(), camera.position


def test_a_box_at_the_target_is_in_the_frustum():
    view_proj, eye = _view()
    assert pl.in_frustum(np.array([-0.5, 0, -0.5]), np.array([0.5, 1, 0.5]), view_proj, eye)


def test_a_box_far_off_to_the_side_is_not_in_the_frustum():
    view_proj, eye = _view()
    assert not pl.in_frustum(
        np.array([30.0, 0, -0.5]), np.array([31.0, 1, 0.5]), view_proj, eye
    )


def test_a_box_that_is_only_partly_in_the_picture_is_not_in_the_frustum():
    view_proj, eye = _view()
    # Straddles the right edge: some corners inside, some beyond.
    straddling = (np.array([0.0, 0, 0.0]), np.array([12.0, 1, 1.0]))
    assert not pl.in_frustum(*straddling, view_proj, eye)


def test_a_box_behind_the_eye_is_not_in_the_frustum_even_though_it_projects_onto_the_picture():
    camera = Camera(aspect=1.5)
    camera.distance = 6.0
    view_proj = camera.projection() @ camera.view()
    toward = camera.position - camera.target
    behind = camera.position + toward / np.linalg.norm(toward) * 4.0
    assert not pl.in_frustum(behind - 0.5, behind + 0.5, view_proj, camera.position)


def test_a_margin_pulls_the_edge_of_the_frustum_in():
    view_proj, eye = _view(distance=6.0)
    lo, hi = np.array([-0.5, 0, -0.5]), np.array([0.5, 1, 0.5])
    assert pl.in_frustum(lo, hi, view_proj, eye, margin=0.0)
    # A 49 percent margin leaves a sliver in the middle of the picture, which a
    # whole box cannot fit inside.
    assert not pl.in_frustum(lo, hi, view_proj, eye, margin=0.49)
