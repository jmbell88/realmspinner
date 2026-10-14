"""One pixel ceiling for every image decoded out of a file we did not write.

**Why this exists, arithmetically.** ``plotter_io._within_ceiling``'s docstring
argued that Pillow's own ``MAX_IMAGE_PIXELS`` is the decompression-bomb guard
and that a second ceiling beside it would be a second answer to one question.
The first half was right about the *question* and wrong about the *guard*:
nothing in this repo ever set Pillow's limit, so its ~89.5 M default applies,
and that default only **warns** between one and two times itself. A file is
refused at 178.9 M pixels and decoded, with a warning nobody reads, at 178.8 M
-- which is a 715 MB allocation from a solid-colour PNG of about 200 KB, from
under every byte ceiling in the app, on a task thread.

The arithmetic picks the number. The largest image this app itself produces is
a packed atlas at ``kernels.sheet.MAX_ATLAS_PX`` a side -- 8192, so 67.1 M
pixels -- and the largest canvas Inker will *make* is ``inker_mode.NEW_MAX``
squared, which is the same 8192. So a ceiling at 8192 squared refuses nothing
this build can legitimately be handed and closes the whole of Pillow's warning
band, which is why this is a hard refusal here rather than the promotion of
``DecompressionBombWarning`` to an error: that promotion is a ``warnings``
filter, ``warnings`` filters are process-global state, and this app decodes on
task threads while the frame thread runs.

**Asked before ``convert``, never after**, which is the rule
``packwright/rpack.py`` already states verbatim at the one door that had it:
``Image.open`` reads a header and ``convert`` is the call that allocates, so a
ceiling consulted after it has been paid rather than applied.

A shared leaf for ``zipguard``'s reason. Thirteen doors across four engines and
the mode layer decode an image out of an untrusted container -- ``.ora``,
``.rmap``, ``.rblk``, ``.tmx``, ``.tsx``, a bare PNG the user picked -- and a
bound that eleven call sites have to remember is a bound that holds at ten.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import numpy as np

#: 8192 squared: ``kernels.sheet.MAX_ATLAS_PX`` and ``inker_mode.NEW_MAX`` are
#: both 8192, so this is the largest image the app itself will ever produce or
#: be asked to open. Module-level so a test lowers it rather than building a
#: 268 MB array, and read at call time for the same reason.
MAX_DECODE_PIXELS = 8192 * 8192

_armed = False


def _arm() -> None:
    """Lower Pillow's own ceiling to ours, once.

    Belt to the braces below: this leaf is wired to every door we know about,
    and Pillow's limit is what a door we have *not* thought of still gets. It
    lowers the point at which Pillow raises by itself from 178.9 M pixels to
    twice this ceiling, without touching a ``warnings`` filter -- an attribute
    assignment is safe from a task thread in a way that
    ``warnings.simplefilter`` is not.

    Raised limits are left alone: a caller that deliberately set a *lower* one
    keeps it, and this never argues a ceiling upwards.

    Pillow still emits its ``DecompressionBombWarning`` to stderr for a file
    between one and two times the limit, before :func:`check` refuses it by
    name; that line is noise, not a decode, and silencing it would take a
    ``warnings`` filter -- process-global state this leaf deliberately avoids
    (the 2026-10-14 audit, finding inker-mode-12, noted it and the refusal, not
    the stderr line, is what a caller sees).
    """
    global _armed
    if _armed:
        return
    from PIL import Image

    if Image.MAX_IMAGE_PIXELS is None or Image.MAX_IMAGE_PIXELS > MAX_DECODE_PIXELS:
        Image.MAX_IMAGE_PIXELS = MAX_DECODE_PIXELS
    _armed = True


def check(width: Any, height: Any, what: str = "an image") -> tuple[int, int]:
    """Refuse a picture too large to hold, from its declared size alone.

    Separate from :func:`opened` because three of the doors here never open a
    Pillow image at all: ``stack.xml`` states a canvas size and ``.aseprite``
    states one in its header, and both are consumed by ``np.zeros`` calls that
    no image decoder ever sees.
    """
    w, h = int(width), int(height)
    if w < 0 or h < 0:
        raise ValueError(f"{what} is {w}x{h}, which is not a size")
    if w * h > MAX_DECODE_PIXELS:
        raise ValueError(
            f"{what} is {w}x{h}, past the {MAX_DECODE_PIXELS} pixels this build"
            " will hold"
        )
    return w, h


def _default_formats() -> list[str]:
    """Every format Pillow can identify **except EPS**.

    The 2026-10-14 audit, finding packwright-15: ``Image.open`` identifies a
    file by its bytes, whatever its name says, and an EPS image's ``load()``
    runs Ghostscript when it is installed -- a child process started from inside
    Pillow, outside the ``winjob`` kill-on-close job, which the AST scan in
    ``tests/test_vram.py`` cannot see. Nothing here reads PostScript, so it is
    never offered. A door that knows it expects one container format (a ``.rpack``
    member is a PNG, an ``.ora`` layer is a PNG) says so with ``formats``.
    """
    from PIL import Image

    Image.init()
    return [name for name in Image.OPEN if name != "EPS"]


@contextlib.contextmanager
def opened(
    source: Any, what: str = "an image", formats: list[str] | None = None
) -> Iterator[Any]:
    """``Image.open`` with its size asked before anything allocates.

    ``formats`` is Pillow's own argument: the identifiers the file may be, or
    every format but EPS when ``None`` (see :func:`_default_formats`). An
    unidentifiable file raises ``UnidentifiedImageError`` (an ``OSError``) as it
    always has.

    A context manager rather than a function returning an image, because
    ``Image.open`` is lazy and holds its file object until it is closed -- the
    note ``clay/serialize._read_textures`` already carries about a document
    with twenty textures in it.
    """
    from PIL import Image

    _arm()
    try:
        image_cm = Image.open(source, formats=formats or _default_formats())
    except Image.DecompressionBombError as exc:
        # The 2026-10-14 audit, finding inker-mode-12: ``_arm`` lowers Pillow's
        # limit to this ceiling, and Pillow *raises* past twice its limit --
        # ``DecompressionBombError``, an ``Exception`` and not a
        # ``ValueError`` -- so a file past 2x the ceiling reached every caller
        # (each catches ``ValueError`` for the named refusal) as "Something went
        # wrong; see the log", while a file between 1x and 2x got the named
        # refusal below. One refusal for both bands.
        raise ValueError(
            f"{what} is past the {MAX_DECODE_PIXELS} pixels this build will hold"
        ) from exc
    with image_cm as image:
        check(image.width, image.height, what)
        yield image


def decode_rgba(
    source: Any, what: str = "an image", formats: list[str] | None = None
) -> np.ndarray:
    """One image as a ``(h, w, 4)`` uint8 array. Task thread only.

    ``.copy()`` deliberately: ``np.asarray`` over a PIL image can hand back a
    view onto buffers the image owns, and the image is closed on the way out of
    the ``with``.
    """
    with opened(source, what, formats) as image:
        return np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
