"""The palette half of pixel-art sheets: one colour set across a whole atlas.

``quantize_shared`` and ``resolve_palette`` are what the queue's sprite stage,
the tile-sheet stage and Flourish's bake share, and ``BAND_BACKGROUND`` is the
neutral grey a restyle canvas is filled with. The band-restyle design that
once lived beside them (``Band``, ``bands``, ``check_restylable``,
``crop_back``, ``remask``, ``downscale``, ``pixel_sidecar``) was deleted with the
sheet pipeline that drove it: the 2026-10-14 audit, finding inker-80.

The idea that survives is the palette one: a colour means the same thing in
every direction because the palette is chosen once across the whole atlas, not
per cell.

Pure: Pillow and NumPy inside the functions, no torch, no service imports, no
decisions about where a file goes. The reduction and the palette work are
delegated to ``pipelines/pixel`` rather than reimplemented, for the reason that
module's docstring gives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .pixel import RGB

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL import Image as _ImageModule

    PILImage = _ImageModule.Image
else:  # pragma: no cover - runtime alias
    PILImage = Any

# Neutral mid-grey behind a restyle canvas, never a saturated key colour: SDXL bleeds the
# background into the subject's edges, and a magenta key would put magenta in
# the palette and a magenta rim on every sprite. The alpha is restored from the
# source render afterwards, so this colour never survives -- but its bleed
# would.
BAND_BACKGROUND = (128, 128, 128)


def quantize_shared(atlas: PILImage, colors: int) -> tuple[PILImage, list[str]]:
    """One palette across the whole atlas, and the palette it chose.

    Once, across every cell -- not per cell. Per-cell quantization is how the
    same shirt comes out two shades in two directions, which is the artefact
    that reads as flicker when a character turns.

    **The alpha is restored from the source, not quantized** -- ``opaque`` is
    computed before the median cut and stamped back after it, so no palette
    entry is spent representing a semi-transparent edge. That is as far as the
    handling goes, and the sentence here used to over-claim it as "opaque
    pixels only": the median cut is run over the *whole* RGB plane, transparent
    region included. Where that region is a uniform colour it costs one entry
    of the budget; a sprite sheet's transparent margin is black, so the cost is
    the black entry the sheet almost certainly wanted anyway.

    Worth stating because of the callers with a large transparent fraction --
    a sprite turnaround's cells are mostly margin -- where one entry out of
    eight is a different proposition from one out of sixty-four. (A tile sheet
    is the opposite case and costs nothing here: every cell is opaque edge to
    edge, so no entry of its budget is spent on a region nobody looks at.)
    Changing it means changing the bytes of every sheet already quantized, so
    it is written down rather than quietly fixed.
    """
    import numpy as np
    from PIL import Image

    from . import pixel as pixelmod

    # The 2026-09-18 audit (inker-05): an unchecked ``colors`` reached
    # ``Image.quantize`` verbatim, so a bad value (0, negative, or past
    # Pillow's own 256-entry palette) surfaced as Pillow's own exception
    # rather than a refusal this app names.
    if not (1 <= int(colors) <= 256):
        raise ValueError(f"a palette must have 1 to 256 colours, not {colors}")
    rgba = np.asarray(atlas.convert("RGBA"))
    opaque = rgba[:, :, 3] >= pixelmod.ALPHA_THRESHOLD
    if not opaque.any():
        raise ValueError("that sheet's cells are empty")
    flat = Image.fromarray(rgba[:, :, :3], "RGB").quantize(
        colors=int(colors), method=Image.Quantize.MEDIANCUT
    )
    mapped = np.asarray(flat.convert("RGB"))
    out = np.dstack([mapped, np.where(opaque, 255, 0).astype(np.uint8)])
    used = {tuple(int(c) for c in p) for p in mapped[opaque]}
    palette = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in sorted(used)]
    return (Image.fromarray(out, "RGBA"), palette)


def resolve_palette(
    atlas: PILImage,
    *,
    colors: int,
    entries: tuple[RGB, ...] | None = None,
) -> tuple[tuple[RGB, ...], str]:
    """The colour set one sheet is mapped to, and where it came from.

    Two sources, one answer: an authored palette is returned verbatim as
    ``"designed"``, and with none given the median cut above picks one off the
    atlas and it comes back as ``"derived"``. ``None`` is how a caller says "I
    have no palette"; an empty tuple is not -- it is a palette of no colours,
    and ``pixel.map_palette`` refuses it, which is the right answer to a
    palette file somebody emptied.

    **The insight this whole shape rests on: a palette file does not replace
    the shared-across-cells property.** It is tempting to read "designed"
    against "derived" as a choice between an authored palette and
    ``quantize_shared``, and so to think naming a palette gives up whatever
    the median cut was doing for the sheet. It gives up nothing.
    ``quantize_shared`` stops being a **mapper** and becomes a **palette
    source**: the thing that made it shared was never the median cut, it is
    that one entry set is applied over the whole atlas in one pass. That
    property survives verbatim here, because ``pixel.map_palette`` maps
    whole-atlas and ``pixelize.pixelize_atlas`` already maps before its
    per-cell loop -- so the same shirt is the same shade in every direction
    whichever branch produced the entries.

    Which is also why the derived branch parses the hex list back to triples
    rather than handing on the mapped image ``quantize_shared`` returns: the
    caller wants *the colours*, and taking the pixels too would run the
    mapping twice and make a derived sheet a second code path with its own
    bugs.
    """
    if entries is not None:
        return (tuple(entries), "designed")
    _mapped, hexes = quantize_shared(atlas, colors)
    return (
        tuple(
            (int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)) for h in hexes
        ),
        "derived",
    )

