"""the 2026-09-26 audit, finding inker-paint-04: Alpha Lock restored only the
alpha channel on a dab, so a fully-transparent pixel's RGB still took the
dab's colour -- invisible immediately, but a real byte difference that (a)
reappeared the moment the pixel was ever unlocked or un-erased, and (b) made
``Document._commit_patch``'s ``np.array_equal(before, after)`` see a change
where nothing showed, pushing an undo step for a stroke a user could not tell
had happened. Covers the round/soft-nib path (``StrokeState._resolve``) and
the image-stamp path (``StrokeState._place``), which carried the same bug
independently.
"""

from __future__ import annotations

import numpy as np

from realmspinner.kernels.pixel import brush
from realmspinner.kernels.pixel.document import Document

RED = (255, 0, 0, 255)
SIZE = (32, 32)


def _doc(size=SIZE) -> Document:
    return Document.blank(*size)


def _tip(width=8, height=8, colour=(255, 0, 0)) -> np.ndarray:
    pixels = np.zeros((height, width, 4), dtype=np.uint8)
    pixels[..., :3] = colour
    pixels[..., 3] = 255
    return pixels


def test_a_round_nib_stroke_leaves_transparent_rgb_untouched_under_the_lock():
    doc = _doc()
    doc.stack.active.alpha_lock = True
    before_rgb = doc.stack.active.pixels[..., :3].copy()

    doc.begin_stroke((16.0, 16.0), RED, size=10)
    doc.stroke_to((20.0, 20.0))
    doc.end_stroke()

    layer = doc.stack.active
    assert int(layer.pixels[..., 3].max()) == 0, "alpha lock must still hold"
    assert np.array_equal(layer.pixels[..., :3], before_rgb), (
        "a fully transparent pixel's RGB must not take the dab's colour under alpha lock"
    )


def test_a_round_nib_stroke_confined_to_transparent_pixels_pushes_no_undo_step():
    doc = _doc()
    doc.stack.active.alpha_lock = True
    before = doc.history.head

    doc.begin_stroke((16.0, 16.0), RED, size=10)
    doc.stroke_to((20.0, 20.0))
    doc.end_stroke()

    assert doc.history.head == before, "nothing visible changed, so nothing should be undoable"


def test_a_locked_stroke_still_tints_a_pixel_that_already_had_colour():
    """The lock guards *transparent* pixels, not the whole canvas: a pixel
    that already carries colour and alpha still takes the dab's colour, since
    tinting existing content under the lock is the feature, not the bug."""
    doc = _doc()
    layer = doc.stack.active
    layer.pixels[10:22, 10:22] = (0, 0, 255, 255)  # opaque blue square first
    layer.alpha_lock = True

    doc.begin_stroke((16.0, 16.0), RED, size=10)
    doc.stroke_to((16.0, 16.0))
    doc.end_stroke()

    assert int(layer.pixels[16, 16, 3]) == 255, "alpha lock must still hold"
    assert tuple(int(v) for v in layer.pixels[16, 16, :3]) != (0, 0, 255), (
        "a pixel with existing content must still tint under alpha lock"
    )


def test_an_image_stamp_leaves_transparent_rgb_untouched_under_the_lock():
    stamp = brush.Stamp(_tip())
    doc = _doc()
    layer = doc.stack.active
    before_rgb = layer.pixels[..., :3].copy()
    layer.alpha_lock = True

    doc.begin_stroke((16.5, 16.5), RED, stamp=stamp)
    doc.end_stroke()

    assert int(layer.pixels[..., 3].max()) == 0, "alpha lock must still hold"
    assert np.array_equal(layer.pixels[..., :3], before_rgb), (
        "a fully transparent pixel's RGB must not take the stamp's colour under alpha lock"
    )
