"""The type ramp: three Inter faces with Lucide icons merged into each.

Loaded once at startup from the TTFs vendored under ``resources/fonts`` -- the
offline invariant applies to fonts as much as to model weights, so these ship
in the wheel and are never fetched. imgui 1.92 sizes fonts at ``push_font``
time, so each *face* is loaded once and the type scale is applied where text
is drawn, not in the atlas.

Headless tests never call :func:`load`; every helper here degrades to a no-op
on the default atlas font (icons render as the missing-glyph box, which the
smoke tests do not look at).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import tokens

FONT_DIR = Path(__file__).parent / "resources" / "fonts"

#: Every file the atlas is built from, in the order :func:`load` reads them.
FACES = (
    "Inter-Regular.ttf",
    "Inter-Medium.ttf",
    "Inter-SemiBold.ttf",
    "lucide.ttf",
    "familiar-sigil.ttf",
)

#: A one-glyph subset of Noto Sans Symbols 2 (SIL OFL 1.1) carrying only
#: U+2726 BLACK FOUR POINTED STAR -- the Familiar mark. Neither Inter nor
#: Lucide carries that codepoint (see menus.FAMILIAR_LABEL's prior history:
#: T0 stood ``icons.SPARKLES`` in for it because the vendored faces couldn't
#: draw it at all), so this is merged in the same way lucide is.
SIGIL_FACE = "familiar-sigil.ttf"

# Noto Sans Symbols 2's U+2726 sits upem 1000 / ascent 1069 / descent -630,
# an unusually tall span (a symbol font gives its glyphs more room than
# Latin text does) that imgui's merge bakes each source against -- so a
# merge at the *same* size_pixels as the base face scales the glyph down by
# its own ascent+descent (1699 units) rather than by its upem, and the star
# came out well under Inter's cap height (screenshot pass, 2026-09-13).
#
# SIGIL_SCALE corrects that: it inflates the size_pixels the sigil source is
# merged at so the glyph's ink height lands close to Inter Regular's cap
# height (``sCapHeight`` 1490 of a 2048 upem, 0.7275 of the em) rather than
# its own cramped fraction (ink y -42..726, 768 of 1699, 0.4520) --
# 0.7275 / 0.4520. Baked size, not glyph_offset, is what fixes this: the
# glyph is too *small*, not merely off the destination baseline, so scaling
# is the load-bearing correction and the offset stays zero.
SIGIL_SCALE = 1.61
SIGIL_OFFSET = (0.0, 0.0)


def _sigil_merge_size(base: float) -> float:
    """The pixel size the sigil source is merged at, given the base face's
    own size_pixels. A pure function so :data:`SIGIL_SCALE` is checkable
    without an imgui context (see ``test_fonts.py``)."""
    return base * SIGIL_SCALE


class FontsUnavailable(RuntimeError):
    """A vendored TTF is missing or unreadable.

    A named exception rather than whatever ``add_font_from_file_ttf`` does with
    a path that is not there, which is an ``IM_ASSERT`` surfacing as a bare
    ``RuntimeError`` with imgui's own wording in it. These files ship in the
    wheel, so the ways to reach this are a partial install, an antivirus
    quarantine and a half-copied directory -- all of which are worth naming,
    and none of which the user can act on from "assertion failed".
    """

    def __init__(self, missing: list[str]) -> None:
        self.missing = missing
        super().__init__(
            f"missing font files in {FONT_DIR}: {', '.join(missing)}"
        )


def _check_files() -> None:
    """Refuse before imgui is asked for a file that is not there.

    ``load`` had no existence check at all, and it is reachable **mid-session**:
    the UI-scale slider re-bakes the atlas through :func:`reload`, so a font
    file that went away while the app was running took the frame loop with it.
    """
    missing = [name for name in FACES if not (FONT_DIR / name).is_file()]
    if missing:
        raise FontsUnavailable(missing)

# Lucide is upem 1000 / ascent 1000 / descent 0, with its ink flush at x=0 and
# spanning y 0..~959; Inter's baseline sits at 0.801 of the line box (ascent
# 1984 of 1984+494 over a 2048 upem). imgui bakes each merged source against
# its *own* ascent-to-descent span and then draws it on the *destination*
# font's baseline, so an unnudged icon occupies -0.158..0.801 of a 0..1 line
# box -- 0.179 too high -- and sits 0.04 to the left inside its own advance.
# Every icon button in the app centres on the line box or the advance, so this
# one offset is what makes all of them land in the middle of their boxes.
ICON_OFFSET = (0.040, 0.179)

# ImFont handles, populated by load(). None means "run on imgui's default".
REGULAR: Any = None
MEDIUM: Any = None
SEMIBOLD: Any = None


def reload(imgui: Any) -> None:
    """Re-bake the atlas at the current ``tokens.SCALE`` (K99).

    **Between frames, never inside one.** ``clear_fonts`` invalidates every
    ``ImFont`` handle, and the ones this module holds are pushed and popped all
    over a frame -- rebuilding mid-frame would leave the rest of that frame
    drawing through freed pointers. ``main`` calls this before ``new_frame``.

    The reason it is needed at all is ``ICON_OFFSET``: the merged icon range's
    glyph offset is baked as an absolute pixel figure at *load* time, so at any
    scale but the one the atlas was built at, every icon sits off-centre in its
    button by a fraction of the difference. The glyph *shapes* would sharpen on
    their own -- imgui 1.92 rasterises per pushed size -- which is exactly why
    the old "text sharpens after a restart" note was only half the story.

    ``FontsUnavailable`` is raised *before* ``clear_fonts``, so a rebuild that
    cannot happen leaves the atlas it already had rather than an empty one. A
    failure after that point resets the three handles to ``None``, which is the
    documented headless state -- every helper here degrades onto imgui's own
    default font -- so the caller can report it and keep drawing.
    """
    _check_files()
    imgui.get_io().fonts.clear_fonts()
    global REGULAR, MEDIUM, SEMIBOLD
    try:
        load(imgui)
    except Exception:
        REGULAR = MEDIUM = SEMIBOLD = None
        raise


def load(imgui: Any) -> None:
    """Build the atlas fonts. Call between context creation and first frame."""
    global REGULAR, MEDIUM, SEMIBOLD
    _check_files()
    io = imgui.get_io()
    base = tokens.TEXT_BODY * tokens.SCALE

    def face(name: str) -> Any:
        font = io.fonts.add_font_from_file_ttf(str(FONT_DIR / name), base)
        merge = imgui.ImFontConfig()
        merge.merge_mode = True
        merge.glyph_offset = imgui.ImVec2(base * ICON_OFFSET[0], base * ICON_OFFSET[1])
        io.fonts.add_font_from_file_ttf(str(FONT_DIR / "lucide.ttf"), base, merge)

        sigil = imgui.ImFontConfig()
        sigil.merge_mode = True
        sigil.glyph_offset = imgui.ImVec2(base * SIGIL_OFFSET[0], base * SIGIL_OFFSET[1])
        io.fonts.add_font_from_file_ttf(
            str(FONT_DIR / SIGIL_FACE), _sigil_merge_size(base), sigil
        )
        return font

    # Regular first: the first atlas font is imgui's default, so every string
    # that never pushes a font still comes out in Inter.
    REGULAR = face("Inter-Regular.ttf")
    MEDIUM = face("Inter-Medium.ttf")
    SEMIBOLD = face("Inter-SemiBold.ttf")


@contextmanager
def push(imgui: Any, font: Any, size: float) -> Iterator[None]:
    """Push a face at a design-pixel size; no-op when fonts were never loaded."""
    if font is None:
        yield
        return
    imgui.push_font(font, size * tokens.SCALE)
    try:
        yield
    finally:
        imgui.pop_font()


def centred_glyph_pos(imgui: Any, glyph: str, cx: float, cy: float) -> tuple[float, float]:
    """Where to ``add_text`` one glyph so its *ink* is centred on ``(cx, cy)``.

    Centring on ``calc_text_size`` centres the glyph's advance and line box,
    and a Lucide icon's ink is neither: ``ICON_OFFSET`` is one average nudge
    for the whole face, so an individual glyph still sat about 3 px left of
    its rail button on a 44 dp column (screenshot, 2026-09-24), and the ✦
    sigil comes from a different face again. The baked glyph's own ``x0..x1``
    and ``y0..y1`` are the ink box relative to the draw position, so this
    centres what the eye actually sees. Falls back to the advance box when the
    glyph is not in the atlas (a headless test's default font).
    """
    size = imgui.calc_text_size(glyph)
    try:
        found = imgui.get_font_baked().find_glyph_no_fallback(ord(glyph[0]))
    except Exception:
        found = None
    if found is None or found.x1 <= found.x0 or found.y1 <= found.y0:
        return cx - size.x * 0.5, cy - size.y * 0.5
    return cx - (found.x0 + found.x1) * 0.5, cy - (found.y0 + found.y1) * 0.5


# There is deliberately no ``body()`` helper. Regular is loaded first, so it
# *is* imgui's default font, and it is loaded at ``tokens.TEXT_BODY`` -- a
# ``push(REGULAR, TEXT_BODY)`` therefore pushes what is already in force. The
# one that existed had no callers, and the reason it never gained any is that
# there is nothing for it to do.


def small(imgui: Any) -> Any:
    return push(imgui, REGULAR, tokens.TEXT_SMALL)


def label(imgui: Any) -> Any:
    """Medium weight at body size: buttons, field labels, card titles."""
    return push(imgui, MEDIUM, tokens.TEXT_BODY)


def title(imgui: Any) -> Any:
    return push(imgui, SEMIBOLD, tokens.TEXT_TITLE)


def heading(imgui: Any) -> Any:
    """One region's name: a pane header, a manual chapter."""
    return push(imgui, SEMIBOLD, tokens.TEXT_HEADING)


def display(imgui: Any) -> Any:
    """The one loud thing on a screen. There is deliberately only ever one.

    SemiBold rather than a fourth vendored face: at 28 px Inter SemiBold is
    already emphatic, and Bold at display size reads as a warning rather than
    as a title against near-black. The call was made against the screenshot
    pass, which is the only way it could be made.
    """
    return push(imgui, SEMIBOLD, tokens.TEXT_DISPLAY)
