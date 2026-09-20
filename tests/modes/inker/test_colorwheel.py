"""``colorwheel``'s one claim: a point on the disc and the colour it produces
are inverses of each other, and the baked texture agrees with both.

No imgui and no GL anywhere in this file -- the whole point of keeping the
maths in its own module (see its docstring) is that this is provable without
a window. ``modes/inker/ui/panes/picker.py``'s own tests cover the imgui half: reading
the mouse, drawing the marker, writing through ``write``.
"""

from __future__ import annotations

import colorsys
import math

from realmspinner.studio.modes.inker.ui import colorwheel


def test_a_point_on_the_disc_round_trips_through_a_colour_and_back():
    """``position_of(colour_at(p))`` lands within a pixel of ``p``, for a grid
    of points spanning the centre to the rim at every angle -- which is what
    makes the wheel invertible rather than merely plausible-looking.

    Byte quantisation (``colour_at`` rounds to 0-255 ints) means this cannot be
    exact; a bug that swapped the angle's sign or scaled saturation by the
    wrong factor would miss by many pixels rather than a fraction of one, so a
    tolerance of a single pixel still catches it.
    """
    radius = 96.0
    for steps in range(1, 13):
        angle = (steps / 12.0) * 2.0 * math.pi
        for fraction in (0.0, 0.15, 0.4, 0.65, 0.9, 1.0):
            dx = math.cos(angle) * fraction * radius
            dy = math.sin(angle) * fraction * radius
            red, green, blue = colorwheel.colour_at(dx, dy, radius, 1.0)
            back_x, back_y = colorwheel.position_of(red, green, blue, radius)
            error = math.hypot(dx - back_x, dy - back_y)
            assert error <= 1.0, (dx, dy, (red, green, blue), (back_x, back_y), error)


def test_the_rim_is_full_saturation_at_every_hue():
    """A point exactly on the rim never gets clamped down to less than full
    saturation -- the boundary a saturation bug (``<`` instead of ``<=``, or an
    off-by-one radius) would land on first."""
    radius = 64.0
    for degrees in range(0, 360, 15):
        angle = math.radians(degrees)
        dx, dy = math.cos(angle) * radius, math.sin(angle) * radius
        red, green, blue = colorwheel.colour_at(dx, dy, radius, 1.0)
        _hue, saturation, _value = colorsys.rgb_to_hsv(red / 255.0, green / 255.0, blue / 255.0)
        assert saturation >= 0.98, degrees


def test_a_drag_past_the_rim_clamps_instead_of_freezing():
    """Aseprite's own wheel keeps tracking a drag that has left the disc,
    reading the rim's colour at whatever angle the pointer is now at --
    exactly what the picker's ``_wheel_pick`` relies on for "a drag that
    leaves the disc clamps to the rim rather than stopping" (see its
    docstring). Ten times the radius, dead centre of a sextant, must answer
    the same colour as the rim itself at that angle."""
    radius = 40.0
    angle = math.radians(200.0)
    at_rim = colorwheel.colour_at(math.cos(angle) * radius, math.sin(angle) * radius, radius, 1.0)
    past_rim = colorwheel.colour_at(
        math.cos(angle) * radius * 10.0, math.sin(angle) * radius * 10.0, radius, 1.0
    )
    assert at_rim == past_rim


def test_the_centre_is_grey_at_every_value():
    """Zero saturation is zero saturation whatever angle a caller's rounding
    happens to compute it at -- ``colour_at`` must not let some stray angle at
    ``dx == dy == 0`` (``atan2(0, 0)`` is defined, but arbitrary) leak colour
    into what has to read as a neutral grey."""
    for value in (0.0, 0.25, 0.6, 1.0):
        red, green, blue = colorwheel.colour_at(0.0, 0.0, 50.0, value)
        assert red == green == blue == max(0, min(255, round(value * 255.0)))


def test_the_baked_wheel_agrees_with_colour_at_pixel_for_pixel():
    """The vectorised bake (``wheel_image``) is a from-scratch reimplementation
    of ``colour_at``'s formula, kept apart for speed (see its docstring) -- so
    nothing pins the two against drifting apart except this. A wrong sextant
    branch or a transposed x/y would show up as most, not all, pixels
    disagreeing, so an exact per-pixel match is the right bar."""
    size = 41
    image = colorwheel.wheel_image(size)
    centre = (size - 1) / 2.0
    radius = max(1.0, size / 2.0 - 1.0)
    checked = 0
    for y in range(size):
        for x in range(size):
            dx, dy = x - centre, y - centre
            if math.hypot(dx, dy) > radius - colorwheel.EDGE_FEATHER:
                continue  # the antialiased rim is not full alpha; skip it here
            red, green, blue = colorwheel.colour_at(dx, dy, radius, 1.0)
            pixel = image[y, x]
            assert (int(pixel[0]), int(pixel[1]), int(pixel[2])) == (red, green, blue), (x, y)
            assert int(pixel[3]) == 255
            checked += 1
    assert checked > (size * size) // 2  # most of a 41-square disc, not a sliver


def test_outside_the_disc_is_transparent_and_the_centre_is_opaque():
    size = 41
    image = colorwheel.wheel_image(size)
    assert int(image[0, 0, 3]) == 0
    assert int(image[size // 2, size // 2, 3]) == 255
