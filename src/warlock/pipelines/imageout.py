"""Re-encoding a reference or a tile's own picture into formats an engine
actually wants to import, mirroring ``pipelines/audioout``'s reasoning.

**No new dependency, again.** Pillow -- already a core dependency, since
``service/files.to_png`` and every 2D derivation in ``service/derive`` already
use it -- writes WebP and JPEG on its own. No second image library, no
binding to libwebp or libjpeg beyond what Pillow already vendors.

**Only ``input.png`` is a source, the way ``track.wav`` is audioout's only
one.** ``input.webp``/``input.jpg`` share its basename because they *are*
that picture, re-encoded -- not a new artifact. ``icon.png`` and
``sprite.png`` are cutouts derived from it by a completely different
operation (lifting a subject off a background) and stay PNG; converting an
already-PNG file to PNG would be a copy wearing a converter's name, which
``service.files.MEDIA``'s own "Source image" row already hands over.

**Not ``studio/atomic.py:save_image``.** That helper resolves a Pillow format
from a path's *real* suffix, which is exactly wrong here: every derivation in
``service/derive`` stages through a dotfile named ``.{name}.tmp`` (see
``derive._staged``), so the path this module is actually handed is
``.input.jpg.tmp`` and has no format in its suffix at all -- the same reason
``audioout.convert`` dispatches on the artifact *name* rather than reading
``out``. Calling ``atomic.save_image`` here would also stage a second time
underneath ``_staged``'s own staging, for no benefit: ``_staged`` already owns
the temp-file-and-rename half of "written atomically".

No torch: this module is imported in the app process, on a path that has no
reason to pay for it.
"""

from __future__ import annotations

from pathlib import Path


class AlphaUnsupported(ValueError):
    """The chosen format cannot carry the transparency this picture has.

    A ``ValueError`` defined *here* rather than a ``service.errors.Invalid``
    raised from here, because ``pipelines`` imports nothing from ``service`` --
    ``material.py`` and ``pixel.py`` both state that rule in their own
    docstrings, and this module was briefly the only file in the package to
    break it. The service layer catches this and re-raises it as the ``Invalid``
    the UI needs (``service/derive.py``, the ``DERIVED_IMAGE`` arm).

    ``field`` rides on the exception so the address survives that hop without
    the caller having to know which control this is about --
    ``errors.invalid_from`` reads exactly this attribute for the same reason
    ``guidance.GuidanceError`` carries one (S137).
    """

    field = "format"


#: Which Pillow format and save kwargs each derived name is written with.
#:
#: WebP is lossless -- the sibling of FLAC's choice in ``audioout``: a cutout
#: reference re-encoded here should not also lose fidelity to a re-quantised
#: photograph. JPEG's ``quality=95, subsampling=0`` is the closest that format
#: gets to "as good as JPEG allows", chosen for the one thing JPEG is actually
#: offered for -- a smaller file for an engine that does not read WebP -- not
#: for the smallest possible one; there is deliberately no quality knob, for
#: ``audioout.FORMATS``'s stated reason: no measurement says the default is
#: insufficient for what these are for.
FORMATS: dict[str, tuple[str, dict]] = {
    "input.webp": ("WEBP", {"lossless": True}),
    "input.jpg": ("JPEG", {"quality": 95, "subsampling": 0}),
}


def convert(source: Path, out: Path, name: str) -> None:
    """Re-encode ``source`` into ``name``'s format, writing to ``out``. Blocking.

    ``name`` dispatches the format for ``audioout.convert``'s reason: the path
    actually being written to is a staging dotfile with no format in its
    suffix.

    **JPEG cannot carry alpha, and a hand-edited or upload-sourced input.png
    can have it** -- ``to_png`` keeps alpha whenever the source already had
    it, the same way a cutout does. Flattening it onto black would be a silent
    lie about the pixels the user asked for, so this refuses instead: a
    :class:`AlphaUnsupported` naming the ``format`` field, which the service
    layer turns into the ``Invalid`` shape
    ``_derive_2d``'s other content-dependent refusals take (``NoSubject`` ->
    ``NotReady``) except that this one really is "the value you chose is not
    usable for this picture" rather than "not ready yet" -- which is exactly
    what ``Invalid`` means and ``NotReady`` does not. WebP is offered
    specifically because it *is* lossless-with-alpha, so a cutout always has a
    format this module can hand back.
    """
    from PIL import Image

    if name not in FORMATS:
        raise ValueError(f"{name} is not a format this build writes")
    fmt, opts = FORMATS[name]
    with Image.open(source) as image:
        image.load()
        # The same test ``to_png`` uses to decide whether to keep alpha: mode
        # alone misses a palette image whose transparency lives in ``info``
        # rather than in the mode.
        has_alpha = image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info
        if fmt == "JPEG":
            if has_alpha:
                raise AlphaUnsupported(
                    f"{source.name} has transparency, and JPEG cannot carry it. "
                    "Choose WebP to keep it."
                )
            if image.mode != "RGB":
                image = image.convert("RGB")
        else:
            image = image.convert("RGBA" if has_alpha else "RGB")
        image.save(out, fmt, **opts)


__all__ = ["FORMATS", "AlphaUnsupported", "convert"]
