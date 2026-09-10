"""The headless half of reference-image handling for an MCP-driven Clay session.

A reference is the picture an agent is trying to match: handed in on a
session, kept small enough to be worth keeping, and put beside or blended
over a render so the agent can *look* at how close it got. Everything in this
module is Pillow and numpy over bytes -- no ``ctx``, no imgui, no moderngl, no
pygame, no GL -- so a session's reference handling is testable on a machine
with no display, the same reason ``studio/sirens/`` keeps its audio device out
of its engine. :mod:`warlock.studio.agent_clay` (the tool surface an MCP agent
actually calls) imports from here; nothing here imports from there, or from
anything else that would pull a window in behind it.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont

from ..service import files

REFERENCE_MAX_SIDE = 1024
"""A stored reference's longest side, in pixels.

A reference exists to be *looked at* next to a render, and ``clay_render``
(``agent_clay.py``) caps a compared render at 1024 too, so a stored copy
larger than that is bytes that can never reach a pixel -- there is no
comparison sheet this module will ever build that shows more of it.
"""

MAX_IMAGE_RESULT = 5 << 20
"""The biggest PNG this module will hand back, in bytes: 5 MiB.

``warlock.mcp.protocol.MAX_FRAME`` is 8 MiB for one JSON-RPC frame (see its
own docstring), and base64 costs a third on top of whatever this module
returns, so a 5 MiB PNG is the largest that comfortably fits inside one frame
with room left for the rest of the message.
"""

GUTTER = 4
"""Width, in pixels, of the white seam between the two halves of a `beside` sheet."""

LABEL_H = 22
"""Height, in pixels, of the label strip above each half of a `beside` sheet."""


@dataclass(frozen=True)
class Reference:
    """One image an agent has handed its session to match against.

    ``view`` is which way the picture is looking, so a comparison render can
    default to the same angle: one of Clay's six axis-view names
    (``Camera.AXIS_VIEWS``), ``"three_quarter"``, or ``"other"`` when it is
    unknown or not a Clay angle at all. ``source`` is where it came from, for
    a listing tool to show: ``f"job:{job_id}:{filename}"`` for a picture
    pulled off an existing job, or ``"inline"`` for one handed over in the
    call itself. This module only defines the shape; ``agent_clay.py`` is
    what constructs and stores these on a session.
    """

    name: str
    png: bytes
    width: int
    height: int
    view: str
    source: str


def normalise(data: bytes) -> tuple[bytes, int, int]:
    """Turn an arbitrary uploaded image into a reference-ready PNG.

    Goes through :func:`warlock.service.files.to_png` rather than
    ``Image.open`` directly: that function checks the pixel count from the
    header *before* decoding (the flat-20-MP-PNG bomb -- tiny on disk,
    enormous decoded) and keeps alpha only when the source already had it, and
    reimplementing either check here would be a second place for the two to
    drift apart. It raises ``files.ImageTooLarge`` (a ``ValueError``
    subclass); this function lets that propagate rather than catching it --
    ``agent_clay.py`` is what turns a refusal into a message naming the
    argument it came from.

    Once decoded, a picture already at or under ``REFERENCE_MAX_SIDE`` on its
    longest side is handed back exactly as ``to_png`` produced it --
    re-encoding a picture that is already small enough is bytes changed for
    nothing. Anything larger is thumbnailed to fit, aspect preserved, and
    re-encoded.
    """
    png = files.to_png(data)
    with Image.open(io.BytesIO(png)) as im:
        width, height = im.width, im.height
        if max(width, height) <= REFERENCE_MAX_SIDE:
            return png, width, height
        im.thumbnail((REFERENCE_MAX_SIDE, REFERENCE_MAX_SIDE), Image.LANCZOS)
        out = io.BytesIO()
        im.save(out, "PNG")
        return out.getvalue(), im.width, im.height


def _letterbox(png: bytes, size: int) -> Image.Image:
    """Fit ``png`` into a ``size`` x ``size`` white RGB cell, aspect preserved.

    Shared by :func:`beside` and :func:`overlay` because both exist to compare
    proportions rather than pixels-per-inch, and stretching to fill the cell
    is exactly the distortion that would defeat that: a squat barrel stretched
    to a square reads as a tall one. Centred, with white padding on whichever
    axis falls short -- never on a canvas that already matches the source's
    own aspect, since there is nothing to pad there.

    Any alpha is flattened onto white before the paste, not merely dropped:
    ``Image.convert("RGB")`` on an RGBA source keeps whatever RGB values sit
    under a transparent pixel, which is frequently garbage, and both
    :func:`beside` and :func:`overlay` need a clean white cell underneath a
    reference or render that was matted against nothing.
    """
    with Image.open(io.BytesIO(png)) as src:
        if src.mode in ("RGBA", "LA", "PA") or "transparency" in src.info:
            rgba = src.convert("RGBA")
            flat = Image.new("RGB", rgba.size, "white")
            flat.paste(rgba, mask=rgba.split()[-1])
        else:
            flat = src.convert("RGB")
        flat.thumbnail((size, size), Image.LANCZOS)
        cell = Image.new("RGB", (size, size), "white")
        x = (size - flat.width) // 2
        y = (size - flat.height) // 2
        cell.paste(flat, (x, y))
        return cell


def beside(
    ref_png: bytes, render_png: bytes, left_label: str, right_label: str, *, size: int
) -> bytes:
    """Build a side-by-side comparison sheet: the reference on the left, the render on the right.

    Each half is letterboxed into its own ``size`` x ``size`` cell beneath a
    label strip -- see :func:`_letterbox` for why letterboxing rather than
    stretching is the point of the whole tool. The canvas is
    ``2 * size + GUTTER`` wide by ``LABEL_H + size`` tall, white, with the two
    labels drawn in black at the top of their half using
    ``ImageFont.load_default()``. Returned as PNG bytes through
    :func:`bounded_png`.
    """
    width = 2 * size + GUTTER
    height = LABEL_H + size
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((0, 0), left_label, fill="black", font=font)
    draw.text((size + GUTTER, 0), right_label, fill="black", font=font)
    sheet.paste(_letterbox(ref_png, size), (0, LABEL_H))
    sheet.paste(_letterbox(render_png, size), (size + GUTTER, LABEL_H))
    out = io.BytesIO()
    sheet.save(out, "PNG")
    return bounded_png(out.getvalue())


def overlay(ref_png: bytes, render_png: bytes, alpha: float, *, size: int) -> bytes:
    """Blend a render over a reference in one ``size`` x ``size`` cell.

    Both images are letterboxed into the cell first (see :func:`_letterbox`),
    then combined with ``Image.blend(reference, render, alpha)`` -- so
    ``alpha`` is **the weight of the render over the reference**: 0.0 is the
    reference alone, 1.0 is the render alone, and 0.5, the default an agent
    should reach for first, is an even mix. Get this mapping backwards and
    every comparison reads inverted. ``alpha`` is clamped to ``0..1`` before
    the blend. Both cells are already flattened to plain RGB coming out of
    ``_letterbox`` -- ``Image.blend`` refuses a mode mismatch, and a reference
    or render with its own alpha channel would still carry one without that
    flattening. Returned as PNG bytes through :func:`bounded_png`.
    """
    alpha = max(0.0, min(1.0, alpha))
    reference = _letterbox(ref_png, size)
    render = _letterbox(render_png, size)
    blended = Image.blend(reference, render, alpha)
    out = io.BytesIO()
    blended.save(out, "PNG")
    return bounded_png(out.getvalue())


def bounded_png(data: bytes) -> bytes:
    """Keep a PNG under ``MAX_IMAGE_RESULT``, by halving its dimensions once.

    Under the ceiling, ``data`` is handed back untouched. Over it, both
    dimensions are halved and the image is re-encoded -- once, not in a loop:
    the ceiling is a frame budget, not a pixel budget, and a picture still too
    big after one halving is a picture whose *content* is the problem (noise,
    a format that does not compress this kind of image well), not its size.
    Looping would hand an agent an unreadable thumbnail instead of a refusal
    it can act on. This function does not refuse on the caller's behalf --
    whoever calls it is responsible for checking the length of what comes
    back and refusing if it still does not fit.
    """
    if len(data) <= MAX_IMAGE_RESULT:
        return data
    with Image.open(io.BytesIO(data)) as im:
        half = (max(1, im.width // 2), max(1, im.height // 2))
        im = im.convert("RGB") if im.mode not in ("RGB", "RGBA") else im
        resized = im.resize(half, Image.LANCZOS)
        out = io.BytesIO()
        resized.save(out, "PNG")
        return out.getvalue()
