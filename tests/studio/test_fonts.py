"""What the vendored atlas is allowed to contain.

The one thing here is a rule about *collisions*, and it is written down because
nothing else in the suite can see it. `fonts.load` builds each face as an Inter
TTF with `lucide.ttf` merged in, and imgui resolves a codepoint against a
merged font's sources **in order**: the base font wins, and a merged source
only fills gaps. So any codepoint Inter carries is a codepoint Lucide cannot
reach, silently -- a confident, wrongly-shaped glyph at the right size in the
right place.

That is not hypothetical. Inter ships 745 PUA cmap entries (stylistic-set
alternates), Lucide 0.525.0 assigns its icons across U+E038-U+E682, and the two
overlapped on 478 codepoints -- 41 of the 83 constants in `icons.py`. Settings
drew `divide.squared`, Clay's ruler drew `question.squared`, Quit's power
symbol drew `six.squared`. It shipped, because the GL smoke suite runs on
imgui's default atlas where every icon is a missing-glyph box by design.

`scripts/strip_font_pua.py` is the fix and this is the assertion that keeps it:
a font bump that re-introduces the PUA fails here rather than on somebody's
screen.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from realmspinner.studio import fonts, icons

pytest.importorskip("fontTools", reason="fonttools is a dev-group tool")

FACES = ("Inter-Regular.ttf", "Inter-Medium.ttf", "Inter-SemiBold.ttf")


def _cmap(name: str) -> dict[int, str]:
    from fontTools.ttLib import TTFont

    return TTFont(str(fonts.FONT_DIR / name)).getBestCmap()


def _icon_codepoints() -> dict[str, int]:
    out: dict[str, int] = {}
    for attr in dir(icons):
        if not attr.isupper():
            continue
        value = getattr(icons, attr)
        if isinstance(value, str) and len(value) == 1:
            out[attr] = ord(value)
    return out


def test_the_icon_constants_are_all_in_the_vendored_lucide():
    """The other direction, and the cheaper failure: a stale constant renders
    as a missing-glyph box, which is at least visible."""
    cmap = _cmap("lucide.ttf")
    missing = {n: cp for n, cp in _icon_codepoints().items() if cp not in cmap}
    assert not missing, f"codepoints absent from lucide.ttf: {missing}"


@pytest.mark.parametrize("face", FACES)
def test_no_inter_face_shadows_an_icon(face: str):
    cmap = _cmap(face)
    shadowed = {
        name: f"U+{cp:04X} would draw {cmap[cp]}"
        for name, cp in _icon_codepoints().items()
        if cp in cmap
    }
    assert not shadowed, (
        f"{face} carries codepoints the icon font needs; re-run "
        f"scripts/strip_font_pua.py: {shadowed}"
    )


@pytest.mark.parametrize("face", FACES)
def test_no_inter_face_carries_a_private_use_codepoint_at_all(face: str):
    """Wider than the test above on purpose.

    Shadowing only *one* of today's 83 constants is what the app is broken by,
    but the icon set grows, and a collision that arrives with a new icon would
    be attributed to the icon rather than to the font. The PUA entries are
    alternates reachable through OpenType features imgui does not run, so
    nothing loses anything by their absence.
    """
    live = [cp for cp in _cmap(face) if 0xE000 <= cp <= 0xF8FF]
    assert not live, f"{face} has {len(live)} PUA codepoints; run scripts/strip_font_pua.py"


def test_sigil_merge_size_matches_inter_cap_height():
    """2026-09-13: the sigil merged at the base face's own size_pixels came
    out well under Inter's cap height (screenshot pass) because imgui bakes
    a merged source against its own ascent+descent span (1699 units for this
    font), not its upem -- a plain 1:1 merge, the way lucide.ttf is merged,
    only happens to work for lucide because its ascent+descent equals its
    upem (1000/0).

    ``fonts.SIGIL_SCALE`` is the fix and ``_sigil_merge_size`` is what
    ``face()`` actually calls, so pin both: the ratio it applies, and that it
    lands close to Inter Regular's own cap-height fraction of the em.
    """
    from fontTools.ttLib import TTFont

    assert fonts._sigil_merge_size(100.0) == pytest.approx(100.0 * fonts.SIGIL_SCALE)

    inter = TTFont(str(fonts.FONT_DIR / "Inter-Regular.ttf"))
    cap_frac = inter["OS/2"].sCapHeight / inter["head"].unitsPerEm

    sigil = TTFont(str(fonts.FONT_DIR / fonts.SIGIL_FACE))
    glyph_name = sigil.getBestCmap()[0x2726]
    from fontTools.pens.boundsPen import BoundsPen

    pen = BoundsPen(sigil.getGlyphSet())
    sigil.getGlyphSet()[glyph_name].draw(pen)
    _, y_min, _, y_max = pen.bounds
    own_span = sigil["hhea"].ascent - sigil["hhea"].descent
    ink_frac = (y_max - y_min) / own_span

    merged_frac = ink_frac * fonts.SIGIL_SCALE
    assert merged_frac == pytest.approx(cap_frac, abs=0.05), (
        f"sigil merges to {merged_frac:.3f} of the em; Inter's cap height is "
        f"{cap_frac:.3f} -- adjust SIGIL_SCALE"
    )


def test_every_vendored_face_is_present():
    for name in (*FACES, "lucide.ttf", fonts.SIGIL_FACE):
        assert (Path(fonts.FONT_DIR) / name).is_file()


def test_familiar_sigil_is_covered_by_the_vendored_faces():
    """Familiar T0 (ef853790): ``menus.FAMILIAR_LABEL`` and the bottom pane's
    row are built with the literal ✦ (U+2726 BLACK FOUR POINTED STAR). Inter
    and Lucide don't carry that codepoint -- it fell through to the atlas's
    missing-glyph box, which at menu-bar size reads as "?" -- so a one-glyph
    subset of Noto Sans Symbols 2 (``fonts.SIGIL_FACE``) is merged in
    alongside Lucide specifically to cover it.

    Every character actually drawn for the sigil must resolve against one of
    the merged faces' cmaps, and U+2726 itself must be one of them: a
    regression that dropped the sigil face, or swapped the label back to
    ``icons.SPARKLES``, would otherwise slip past a check that only compared
    sets.
    """
    from realmspinner.studio import menus

    assert "✦" in menus.FAMILIAR_LABEL, (
        "FAMILIAR_LABEL must carry the literal ✦ (U+2726), not a stand-in icon"
    )

    covered: set[int] = set()
    for face in (*FACES, "lucide.ttf", fonts.SIGIL_FACE):
        covered |= set(_cmap(face))

    assert 0x2726 in covered, "no vendored face covers U+2726 BLACK FOUR POINTED STAR"

    missing = {ch: f"U+{ord(ch):04X}" for ch in menus.FAMILIAR_LABEL if ord(ch) not in covered}
    assert not missing, f"codepoints in FAMILIAR_LABEL absent from every vendored face: {missing}"


def test_no_icon_constant_is_an_empty_placeholder():
    """The 2026-09-08 audit's shell-06: ``_icon_codepoints`` above filters on
    ``len(value) == 1``, which silently drops an empty-string constant from
    both codepoint checks rather than flagging it -- a transcription gap left
    as `` "" `` renders no glyph at all (not even the wrong one) and nothing
    here would catch it. Checked directly rather than through
    ``_icon_codepoints``, since that helper is exactly what excludes it."""
    empty = [
        attr
        for attr in dir(icons)
        if attr.isupper()
        and isinstance(getattr(icons, attr), str)
        and getattr(icons, attr) == ""
    ]
    assert not empty, f"empty icon codepoint constant(s): {empty}"
