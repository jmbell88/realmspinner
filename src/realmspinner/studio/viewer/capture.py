"""Turning a rendered viewport into a PNG.

One place, used by the thumbnail, the screenshot and the golden-image tests --
so a row-order bug is one bug rather than three.
"""

from __future__ import annotations

import io
from typing import Any

from .glctx import Viewport


def image(viewport: Viewport) -> Any:
    """The viewport's current contents as an RGBA PIL image, top row first.

    The row flip is folded into Pillow's raw decoder (orientation -1 reads
    bottom-up) rather than done as a numpy slice-and-copy first (D41): one
    copy instead of two on a path the frame thread pays for thumbnails.

    Deliberately no un-premultiply pass here: this is also the thumbnail
    path, which is opaque and paid for on the frame thread, and an opaque
    render's alpha is 1.0 everywhere so the pass would be a no-op bought at
    the cost of the very D41 copy this function exists to avoid. (The one
    alpha-carrying consumer, the direction-strip preview, was removed on
    2026-10-14 with its un-premultiply pass: the 2026-10-14 audit, finding
    create-viewer-03.)
    """
    from PIL import Image

    return Image.frombuffer(
        "RGBA", viewport.size, viewport.read_raw(), "raw", "RGBA", 0, -1
    )


def png_bytes(viewport: Viewport) -> bytes:
    """PNG bytes for the current frame, always opaque.

    A thumbnail is shown against the library's own background and a
    transparent one would let the card show through the model.
    """
    img = image(viewport).convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, "PNG")
    return buffer.getvalue()


# There was a ``save_png(viewport, path)`` here, and the only caller it had
# anywhere was the test that tested it. Every real capture path -- the
# thumbnail, the screenshot overlay, the golden-image tests
# -- takes ``png_bytes`` and hands the bytes to whatever owns the destination,
# which is what lets those writes be staged. This one wrote in place, which is
# the opposite rule, so it was a trap sitting in a module four callers import.
