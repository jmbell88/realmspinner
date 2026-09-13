"""``charsheet.compass_name`` against the camera arithmetic it is derived
from, not against its own table -- see
``docs/measurements/2026-09-12-troupe-open-clip-vocabulary.md``
("Compass names follow the camera arithmetic, not the docstring")."""

from __future__ import annotations

import math

import pytest

from warlock.pipelines import blender_worker
from warlock.pipelines import charsheet as cs

# The eight compass points the eight principal yaws land on, walking the
# compass rose from "toward the camera" (0 degrees, in this test's own
# basis) clockwise: this is independent of cs.compass_name and cs.COMPASS_16,
# built straight from the vector arithmetic the measurement doc lays out.
_EIGHT_POINTS = ("S", "SE", "E", "NE", "N", "NW", "W", "SW")


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _expected_compass(yaw: float) -> str:
    """Derive the compass point straight from ``_view_forward``, with no
    reference to ``compass_name`` at all.

    The templates face -Y (fixed). "Toward the camera" is ``-forward`` (the
    camera sits at ``centre - forward * distance``); "screen-right" is
    ``forward`` cross ``(0, 0, 1)``, the measurement doc's own construction
    for the yaw-90 case. Projecting the fixed facing onto those two axes and
    reading off the angle gives the compass point with no appeal to
    ``compass_name``'s own bearing formula.
    """
    forward = blender_worker._view_forward(yaw, 0.0)
    toward_camera = tuple(-f for f in forward)
    screen_right = _cross(forward, (0.0, 0.0, 1.0))
    facing = (0.0, -1.0, 0.0)
    x = _dot(facing, screen_right)
    y = _dot(facing, toward_camera)
    angle = math.degrees(math.atan2(x, y)) % 360.0
    index = round(angle / 45.0) % 8
    return _EIGHT_POINTS[index]


@pytest.mark.parametrize("name, yaw", cs.DIRECTIONS)
def test_compass_names_follow_the_camera_arithmetic_not_the_docstring(name, yaw):
    assert cs.compass_name(yaw) == _expected_compass(yaw), name


#: The sixteen direction keys' compass points, written out by hand from
#: ``docs/measurements/2026-09-12-troupe-open-clip-vocabulary.md``'s rule
#: (``bearing = (180 + yaw) % 360``) rather than derived from
#: ``compass_name``/``COMPASS_16`` -- so this test can actually disagree with
#: the code it is checking, unlike the tautology it replaces (``COMPASS_16``
#: is built by calling ``compass_name`` on every preset key, so comparing the
#: two only proved a dict equals itself).
_LITERAL_COMPASS = {
    "front": "S",
    "front_front_left": "SSW",
    "front_left": "SW",
    "left_front_left": "WSW",
    "left": "W",
    "left_back_left": "WNW",
    "back_left": "NW",
    "back_back_left": "NNW",
    "back": "N",
    "back_back_right": "NNE",
    "back_right": "NE",
    "right_back_right": "ENE",
    "right": "E",
    "right_front_right": "ESE",
    "front_right": "SE",
    "front_front_right": "SSE",
}


def test_every_direction_preset_has_a_compass_name():
    for directions in cs.DIRECTION_PRESETS.values():
        for key, yaw in directions:
            assert key in _LITERAL_COMPASS, key
            assert cs.compass_name(yaw) == _LITERAL_COMPASS[key]


def test_sixteen_directions_use_the_three_letter_points():
    eight = {name for name, _yaw in cs.DIRECTIONS}
    for key, yaw in cs._DIRECTIONS_16:
        point = cs.compass_name(yaw)
        if key in eight:
            assert len(point) in (1, 2), (key, point)
        else:
            assert len(point) == 3, (key, point)
