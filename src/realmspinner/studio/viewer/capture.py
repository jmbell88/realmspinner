"""Turning a rendered viewport into a PNG.

One place, used by the thumbnail, the screenshot, the sheet preview and the
golden-image tests -- so a row-order or alpha bug is one bug rather than four.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np

from .glctx import Viewport


def image(viewport: Viewport) -> Any:
    """The viewport's current contents as an RGBA PIL image, top row first.

    The row flip is folded into Pillow's raw decoder (orientation -1 reads
    bottom-up) rather than done as a numpy slice-and-copy first (D41): one
    copy instead of two on a path the frame thread pays for thumbnails.

    Deliberately no un-premultiply pass here (see :func:`unpremultiply`):
    this is also the thumbnail path, which is opaque and paid for on the
    frame thread, and an opaque render's alpha is 1.0 everywhere so the pass
    would be a no-op bought at the cost of the very D41 copy this function
    exists to avoid. The one alpha-carrying consumer, :func:`png_bytes`'s
    ``opaque=False`` branch, applies it itself.
    """
    from PIL import Image

    return Image.frombuffer(
        "RGBA", viewport.size, viewport.read_raw(), "raw", "RGBA", 0, -1
    )


def unpremultiply(rgba: np.ndarray) -> np.ndarray:
    """Undo the premultiplication the MSAA resolve performs at a silhouette
    edge against a transparent clear.

    A partially covered edge texel out of ``glctx.Viewport.resolve``'s
    blit is several subsamples averaged together -- some the model's colour
    at full coverage, some the clear colour (``(0, 0, 0, 0)`` for a
    transparent background), so the average is ``colour * coverage``,
    already premultiplied. Every straight-alpha consumer here (a PNG viewer
    compositing the sheet preview over its own background) blends that
    colour by its alpha a *second* time, which is what turned every
    silhouette into a dark fringe -- the 2026-09-26 audit, finding
    create-viewer-04. A no-op for a fully opaque render, since dividing by
    alpha 255 leaves the colour unchanged.
    """
    array = rgba.astype(np.float64)
    rgb, alpha = array[..., :3], array[..., 3:4]
    covered = alpha > 0
    straight = np.divide(rgb * 255.0, alpha, out=rgb.copy(), where=covered)
    out = rgba.copy()
    out[..., :3] = np.clip(straight, 0, 255).astype(np.uint8)
    return out


def png_bytes(viewport: Viewport, *, opaque: bool = True) -> bytes:
    """PNG bytes for the current frame.

    Opaque by default: a thumbnail is shown against the library's own
    background and a transparent one would let the card show through the model.
    The sheet preview is the exception and asks for alpha -- and is exactly
    the render :func:`unpremultiply` exists for, since it renders against a
    transparent clear and keeps the alpha channel a straight-alpha PNG reader
    will composite again.
    """
    from PIL import Image

    img = image(viewport)
    img = (
        img.convert("RGB") if opaque else Image.fromarray(unpremultiply(np.asarray(img)), "RGBA")
    )
    buffer = io.BytesIO()
    img.save(buffer, "PNG")
    return buffer.getvalue()


# There was a ``save_png(viewport, path)`` here, and the only caller it had
# anywhere was the test that tested it. Every real capture path -- the
# thumbnail, the screenshot overlay, the sheet preview, the golden-image tests
# -- takes ``png_bytes`` and hands the bytes to whatever owns the destination,
# which is what lets those writes be staged. This one wrote in place, which is
# the opposite rule, so it was a trap sitting in a module four callers import.
