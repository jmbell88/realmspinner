"""Poses to pixels: one turned cut-out per part per frame.

The turn goes through ``selection.render_transform_about``, which is this
module's whole reason for being short. That function pads a plane so a chosen
point lands at its centre, runs the editor's one scale-shear-rotate kernel, crops
back to coverage and reports where the crop sits relative to the pivot -- which
is exactly "turn this limb about its shoulder and tell me where to put it". Its
docstring says it has two callers on purpose; this is the third, and the reason
to be the third rather than to re-spell the maths is that a walk that turned
pixels its own way would drift from what the free transform does to the same art.

``resample="rotsprite"`` throughout. Nearest-neighbour is the wrong filter for
*turning* pixel art -- a hard diagonal comes out as a staircase with a different
tread on every step -- and every smooth filter invents colours the palette does
not have. RotSprite is deterministic by construction (every step is an integer
copy) which is what lets a digest pin this module's output.

One sign conversion lives here and nowhere else: ``gait`` measures angles in
screen degrees, clockwise-positive because image y runs down, and Pillow rotates
counter-clockwise. :data:`_PIL_SIGN` is that minus one, written once.
"""

from __future__ import annotations

import numpy as np

from ..selection import render_transform_about
from ..transform import ROTSPRITE_MAX_PIXELS
from . import gait
from . import rig as R

#: Pillow turns counter-clockwise; ``gait.screen_angle`` grows clockwise because
#: image y runs down. One place, one sign.
_PIL_SIGN = -1.0

RESAMPLE = "rotsprite"

#: :func:`.transform.rotate`'s ``rotsprite_budget`` for this module's turns.
#:
#: The 2026-09-26 audit, finding inker-mode-14: a part turns about its own
#: *joint*, not its centre, through ``render_transform_about``'s pad-to-pivot
#: trick -- and that padding roughly doubles whichever side the joint sits
#: off-centre on, angle-independently (it has to cover the worst case before
#: the actual rotation is known). A 260x520 thigh with a hip joint near one
#: end pads past ``ROTSPRITE_MAX_PIXELS`` and silently fell back to
#: nearest-neighbour for a turn the module's own docstring says never happens
#: -- "RESAMPLE='rotsprite' throughout". ``ROTSPRITE_MAX_PIXELS`` is sized for
#: a live free-transform drag re-rendering on every mouse-move
#: (frame-thread cost, see its own docstring); the walk's preview and bake run
#: off the frame thread on a ``TaskRunner`` task (``inker.walk``; there is no
#: ``_q_*`` job), so they may spend more per turn.
#: Four times the interactive budget comfortably covers an ordinarily
#: proportioned limb's worst-case pivot padding without reopening the
#: interactive ceiling itself, which stays a `dev/measurements/`-keyed
#: constant this module does not touch.
ROTSPRITE_BUDGET = 4 * ROTSPRITE_MAX_PIXELS


def _mask_of(plane: np.ndarray) -> np.ndarray:
    """The coverage plane a transform needs: the art's own alpha.

    ``render_transform_about`` crops its result to the *mask*, so a mask that
    said "all of it" would keep the crop rectangular and put the offsets in the
    wrong place. The alpha is what the art actually covers.
    """
    return np.ascontiguousarray(plane[:, :, 3])


def part_frame(
    part: R.Part, spec: R.PartSpec, rest: R.Rig, pose: gait.Pose
) -> tuple[np.ndarray, tuple[int, int]] | None:
    """One part, turned and placed. ``(pixels, top_left)``, or None if unused.

    The pivot is the part's own joint **in the crop's coordinates** -- the crop's
    origin subtracted off -- and the destination is that same joint's posed
    position. Everything between is ``render_transform_about``'s.
    """
    if part.pixels is None:
        return None
    rest_pivot = rest.joints.get(spec.pivot)
    posed_pivot = pose.joints.get(spec.pivot)
    if rest_pivot is None or posed_pivot is None:
        return None
    degrees = pose.angles.get(spec.name, 0.0)
    if spec.follows:
        degrees = pose.angles.get(spec.follows, degrees)
    pivot = (rest_pivot[0] - part.origin[0], rest_pivot[1] - part.origin[1])
    pixels, _mask, offset = render_transform_about(
        part.pixels,
        _mask_of(part.pixels),
        degrees * _PIL_SIGN,
        (1.0, 1.0),
        (0.0, 0.0),
        RESAMPLE,
        pivot,
        rotsprite_budget=ROTSPRITE_BUDGET,
    )
    left = int(round(posed_pivot[0] + offset[0]))
    top = int(round(posed_pivot[1] + offset[1]))
    return pixels, (left, top)


#: part name -> (pixels, top_left) for the parts that drew this frame.
Frame = dict[str, tuple[np.ndarray, tuple[int, int]]]


def frames(
    rest: R.Rig, settings: gait.WalkSettings, count: int = gait.WALK_FRAMES
) -> list[Frame]:
    """Every part of every frame, turned and placed, in draw order.

    Deterministic: the poses are arithmetic and the turns are integer copies, so
    the same rig and the same settings give the same bytes on every machine.
    """
    out: list[Frame] = []
    for pose in gait.cycle(rest, settings, count):
        frame: Frame = {}
        for name in rest.order:
            drawn = part_frame(rest.parts[name], R.BY_NAME[name], rest, pose)
            if drawn is not None:
                frame[name] = drawn
        out.append(frame)
    return out


def bounds(rendered: list[Frame]) -> tuple[int, int, int, int] | None:
    """The box every frame of the cycle fits inside, or None for nothing drawn.

    What the pane's clipping warning reads. Taken over the *whole* cycle and not
    per frame, because the framing is fixed across the cycle -- a foot that only
    leaves the canvas on frame six still clips the animation.
    """
    boxes = [
        (left, top, left + pixels.shape[1], top + pixels.shape[0])
        for frame in rendered
        for pixels, (left, top) in frame.values()
        if pixels.size
    ]
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def clipping(rendered: list[Frame], size: tuple[int, int]) -> tuple[int, int, int, int]:
    """How far the cycle overflows the canvas on each side, in pixels.

    ``(left, top, right, bottom)``, all non-negative, all zero when it fits.
    Shown before the bake rather than discovered after it, because the bake
    crops silently and a foot lost to the canvas edge looks like a rig error.
    """
    box = bounds(rendered)
    if box is None:
        return (0, 0, 0, 0)
    width, height = size
    return (
        max(0, -box[0]),
        max(0, -box[1]),
        max(0, box[2] - width),
        max(0, box[3] - height),
    )


def place(drawn: tuple[np.ndarray, tuple[int, int]], size: tuple[int, int]) -> np.ndarray:
    """One part's pixels on a clear canvas-sized plane, cropped to fit.

    ``_doc_flourish._place``'s arithmetic; restated rather than imported because
    this package does not reach into the document layer, and it is six lines.
    """
    pixels, (ox, oy) = drawn
    width, height = size
    out = np.zeros((height, width, 4), dtype=np.uint8)
    part_h, part_w = pixels.shape[:2]
    x0, y0 = max(0, ox), max(0, oy)
    x1, y1 = min(width, ox + part_w), min(height, oy + part_h)
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = pixels[y0 - oy : y1 - oy, x0 - ox : x1 - ox]
    return out
