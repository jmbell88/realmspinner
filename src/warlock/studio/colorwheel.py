"""Hue/saturation-wheel maths, importable with no imgui.

Aseprite's colour wheel is hue x saturation at full value, with value pulled
onto its own bar underneath -- which is what makes a *disc* the right thing to
draw at all: value is the one dimension the wheel does not show, so nothing
about the wheel's own pixels ever changes when only Value moves. That is also
what makes a **pre-rendered texture** correct rather than merely convenient
(see ``panes/inker_picker.py``'s texture cache, which uploads this once and
re-uses it every frame until the pane's size or the UI scale changes).

Kept apart from the pane that draws it so the one claim this file makes --
that a point on the disc and the colour it produces are inverses of each
other -- can be proven without a window. ``tests/test_colorwheel.py`` is that
proof.

**Angle convention.** Hue is the angle of ``(dx, dy)`` around the centre,
measured the way screen coordinates already are (``atan2(dy, dx)``, y growing
downward) so a caller that already has screen-space mouse deltas needs no
sign flip before calling in. Saturation is distance from the centre divided by
the disc's radius, clamped to 1.0 -- which is what makes a drag that leaves
the disc read as "the rim colour at this angle" instead of nothing at all.
"""

from __future__ import annotations

import colorsys
import math

import numpy as np

#: The rim's antialiasing band, in the same px the disc is rasterised at. A
#: hard-edged disc showed a visible staircase once it was drawn much past
#: 100px across, which is well inside the sizes the picker draws it at; one
#: linear-alpha pixel is what Aseprite's own wheel uses and is invisible at
#: every size this pane asks for.
EDGE_FEATHER = 1.0


def colour_at(dx: float, dy: float, radius: float, value: float) -> tuple[int, int, int]:
    """The colour at ``(dx, dy)`` from the wheel's centre. -> 0-255 RGB ints.

    ``value`` is a parameter rather than something read off the wheel, because
    the wheel never encodes it (see the module docstring) -- the caller is
    ``panes/inker_picker.py``'s own Value bar. Saturation past the rim clamps
    to 1.0 instead of being rejected, which is what lets a drag that leaves the
    disc keep tracking the pointer's angle rather than freezing.
    """
    distance = math.hypot(dx, dy)
    saturation = 0.0 if radius <= 0.0 else min(distance / radius, 1.0)
    hue = (math.atan2(dy, dx) / (2.0 * math.pi)) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, max(0.0, min(1.0, value)))
    return (
        max(0, min(255, round(red * 255.0))),
        max(0, min(255, round(green * 255.0))),
        max(0, min(255, round(blue * 255.0))),
    )


def position_of(r: int, g: int, b: int, radius: float) -> tuple[float, float]:
    """The inverse of :func:`colour_at`: a colour's ``(dx, dy)`` on the wheel.

    ``value`` does not come back out -- an RGB triple's brightness is exactly
    what the wheel does not encode, so there is nothing here to return it from.
    That is also why ``panes/inker_picker.py`` does *not* call this for its own
    ring marker: reconstructing a colour from a held hue/saturation pair and
    then asking this function for its angle would round-trip a fully
    desaturated colour back to hue 0 every time (``colorsys.rgb_to_hsv`` picks
    an arbitrary hue at saturation 0), silently discarding the hue the picker
    is holding on to for exactly that case. This function is still the right
    tool for a colour with **no** held state to derive a starting marker from,
    and it is what proves :func:`colour_at` is invertible at all.
    """
    hue, saturation, _value = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    angle = hue * 2.0 * math.pi
    distance = saturation * radius
    return (distance * math.cos(angle), distance * math.sin(angle))


def _hsv_to_rgb_array(hue: np.ndarray, saturation: np.ndarray, value: np.ndarray) -> np.ndarray:
    """``colorsys.hsv_to_rgb``, vectorised over an array of hues.

    Reproduced rather than looped: baking a wheel by calling :func:`colour_at`
    once per pixel is a Python call per pixel, and a wheel baked at a 4K
    session's UI scale (a few hundred px across) is tens of thousands of them
    -- long enough to be a visible stall on the one frame that bakes it. The
    formula is copied from ``colorsys`` verbatim (not re-derived) so a pixel
    here and a single :func:`colour_at` call never disagree about the same
    input; ``tests/test_colorwheel.py`` pins that agreement directly.

    ``s == 0`` needs no branch: every one of ``p``, ``q`` and ``t`` reduces to
    ``v`` exactly when ``s`` is 0, so whichever sextant a hue of undefined
    saturation happens to land in, the three arrays already agree.
    """
    sextant = np.floor(hue * 6.0)
    frac = hue * 6.0 - sextant
    p = value * (1.0 - saturation)
    q = value * (1.0 - saturation * frac)
    t = value * (1.0 - saturation * (1.0 - frac))
    index = sextant.astype(np.int64) % 6
    red = np.select([index == 0, index == 1, index == 2, index == 3, index == 4, index == 5],
                     [value, q, p, p, t, value])
    green = np.select([index == 0, index == 1, index == 2, index == 3, index == 4, index == 5],
                       [t, value, value, q, p, p])
    blue = np.select([index == 0, index == 1, index == 2, index == 3, index == 4, index == 5],
                      [p, p, t, value, value, q])
    return np.stack([red, green, blue], axis=-1)


def wheel_image(size: int) -> np.ndarray:
    """A ``(size, size, 4)`` uint8 RGBA hue/saturation disc at full value.

    Baked once and uploaded once by ``panes/inker_picker.py``'s texture cache
    -- hundreds of per-frame triangles were the alternative, for a picture
    that never changes on its own (see the module docstring). Transparent
    outside the disc and antialiased at the rim (``EDGE_FEATHER``), so the
    pane can draw it straight over whatever is behind the panel without a
    background quad of its own.
    """
    size = max(1, int(size))
    centre = (size - 1) / 2.0
    radius = max(1.0, size / 2.0 - 1.0)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float64)
    dx = xs - centre
    dy = ys - centre
    distance = np.hypot(dx, dy)
    saturation = np.clip(distance / radius, 0.0, 1.0)
    hue = (np.arctan2(dy, dx) / (2.0 * math.pi)) % 1.0
    rgb = _hsv_to_rgb_array(hue, saturation, np.ones_like(hue))
    alpha = np.clip((radius + EDGE_FEATHER - distance) / EDGE_FEATHER, 0.0, 1.0)
    out = np.empty((size, size, 4), dtype=np.uint8)
    out[..., :3] = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
    out[..., 3] = np.clip(np.round(alpha * 255.0), 0, 255).astype(np.uint8)
    return out
