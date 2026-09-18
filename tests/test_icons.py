"""Icon catalogue: one concept, one glyph.

Regression tests for the 2026-09-08 icon inventory. It found: ``LINK`` and
``UNLINK`` shipped as empty strings (the private-use codepoints were never
filled in, so Inker's "keep width and height linked" toggle rendered as an
empty square with no glyph at all); a backwards ``EYE_OFF`` on Clay's "Show
all"; two literal ASCII ``"x"`` deletes standing in for ``icons.TRASH``; two
bare ``"+ "`` prefixes standing in for ``icons.PLUS``; one raw Unicode
``"✓"`` standing in for ``icons.CHECK``; and the export/import glyph split
three ways (``DOWNLOAD``/``UPLOAD``/``FOLDER_OPEN``) for what is only ever
two meanings. Each test name is the claim; each was verified to fail against
the pre-fix source by reverting the corresponding edit in a scratch copy
(not with git, since other fixers are editing this repository at the same
time).
"""

from __future__ import annotations

import re
from pathlib import Path

from _panes import pane_files

STUDIO_ROOT = Path(__file__).resolve().parents[1] / "src" / "warlock" / "studio"

# icons.py is the rulebook, not a pane: its own docstring is required to
# *name* the forbidden glyphs (that is where the rule is stated, per
# CLAUDE.md's "state each rule once" instruction), and a naive text scan
# would otherwise flag its own prose as a violation of the rule it states.
_RULEBOOK = STUDIO_ROOT / "icons.py"


def _studio_sources() -> dict[Path, str]:
    return {
        path: path.read_text(encoding="utf-8")
        for path in STUDIO_ROOT.rglob("*.py")
        if path != _RULEBOOK
    }


def test_the_sweep_found_the_studio_tree() -> None:
    """2026-09-17 (dev/RESTRUCTURE.md P3 sweep-coverage pass): stays scoped to
    ``studio/`` on purpose. Icon glyphs are drawn by imgui buttons and panes;
    none of P3's moves pull imgui with them, so a kernel or ``core/`` module
    cannot be an icon-constant offender. The floor is the guard against the
    failure this pass found elsewhere -- an empty (or merely smaller) file set
    still passes every ``== []`` assertion below.
    """
    files = list(_studio_sources())
    assert len(files) > 200, (
        f"only {len(files)} files under {STUDIO_ROOT} -- did the sweep root break?"
    )


def test_no_icon_constant_is_an_empty_string():
    """This is the check that would have caught LINK/UNLINK the day they
    landed empty: a Private-Use-Area constant with no codepoint renders as a
    glyph-less box, and nothing short of a rendered frame notices (the GL
    smoke suite runs on imgui's default atlas, where *every* icon is a
    missing-glyph box by design)."""
    import warlock.studio.icons as icons

    empty = [
        name
        for name, value in vars(icons).items()
        if name.isupper() and isinstance(value, str) and value == ""
    ]
    assert empty == []


def test_link_and_unlink_are_the_lucide_link_glyphs():
    """Pinned against the shipped font rather than against a guessed
    codepoint: ``lucide.ttf``'s own glyph names (not the PUA cmap, which
    renders as nothing in a terminal or an editor and is easy to mistake for
    empty) say U+E108 is "link" and U+E19C is "unlink"."""
    from fontTools.ttLib import TTFont

    import warlock.studio.icons as icons

    font = TTFont(str(STUDIO_ROOT / "resources" / "fonts" / "lucide.ttf"))
    cmap = font.getBestCmap()
    rev: dict[str, int] = {}
    for codepoint, glyph_name in cmap.items():
        rev.setdefault(glyph_name, codepoint)

    assert chr(rev["link"]) == icons.LINK
    assert chr(rev["unlink"]) == icons.UNLINK


# No explicit exemption list is needed for the two documented deliberate
# exceptions: library.py/library_full.py's ASCII "v"/"^" sort arrows
# (chevron-up is missing from the pinned 0.525.0 atlas -- library.py:555-559)
# don't match the lone-"x" pattern below, and plotter_bridge.py:127's
# "  • {detail}" bullet, being prose rather than a button label, doesn't
# match the checkmark pattern either. Both patterns are narrow by
# construction rather than by an allow-list that could go stale.


def _button_offenders(pattern: re.Pattern[str], *, exempt: set[Path] = frozenset()) -> list[str]:
    offenders = []
    for path, source in _studio_sources().items():
        if path in exempt:
            continue
        if pattern.search(source):
            offenders.append(str(path.relative_to(STUDIO_ROOT)))
    return offenders


# Matches a button-family call (``controls.button``, ``controls.small_button``,
# ``widgets.icon_button``, ``widgets.disabled_button``, ``widgets.primary_button``,
# ``widgets.ghost_button``, ...) whose *first* argument is the literal ASCII
# ``"x"``, optionally carrying an imgui ``##id`` suffix. Anchored at every
# call ending "button(" rather than at each helper by name, so a new button
# helper is covered without editing this file.
_LONE_X_BUTTON = re.compile(r'''button\(\s*f?(["'])x(##[^"']*)?\1''')


def test_no_button_uses_a_literal_x_as_a_delete_glyph():
    """icons.TRASH is the catalogue's row-delete glyph, used at every other
    row-delete site in the app (inker_timeline's frame delete, both
    plotter_layers row deletes, clay_outliner's row delete, ...).
    plotter_tileset_editor.py drew an animation-frame delete and a
    Wang-colour-slot delete as a literal ASCII "x" instead -- the same glyph
    ``icons.X`` uses for cancel/dismiss/clear, so the two meanings collided.

    ``inker_tools.py`` carried the same two -- a gradient-stop delete and a
    tool-preset delete -- and was fixed in the same pass, so there is no
    exemption here any more. The one this replaced could not have caught its
    own staleness: ``set(offenders) <= allowed`` stays true once ``offenders``
    is empty, so an allow-list that had become unnecessary would have gone on
    passing and nobody would ever have removed it. An exemption that cannot
    fail when it stops being needed is not a tripwire.
    """
    assert _button_offenders(_LONE_X_BUTTON) == []


# Matches a button-family call whose first argument is a bare ``"+ "``
# prefix -- the literal-plus spelling of an "add" control.
_PLUS_PREFIX_BUTTON = re.compile(r'''button\(\s*f?(["'])\+ [^"']*\1''')


def test_no_button_is_labelled_with_a_literal_plus_prefix():
    """icons.PLUS is drawn as an ``f"{icons.PLUS} {label}"`` prefix at every
    other "add" control in the app; inker_colors.py spelled two of them
    "+ swatch" and "+ from colour" with a literal plus instead, which reads
    as a keyboard hint (like the Ctrl+E chord in the same pane) rather than
    as an icon."""
    offenders = _button_offenders(_PLUS_PREFIX_BUTTON)
    assert offenders == [], offenders


# A raw checkmark inside *any* quoted string -- not anchored to "button(",
# because the real instance of this bug (muse_results.py) puts the glyph in
# the false branch of a ternary passed as the label, one hop away from the
# call itself: ``"Stems" if not stems else "Stems ✓"``.
_QUOTED_CHECK = re.compile(r'''(["'])[^"'\n]*✓[^"'\n]*\1''')


def test_no_pane_draws_a_raw_unicode_checkmark():
    """icons.CHECK is used for "already done" in eleven other places
    (widgets.py's stage-rail ticks, review's accept marks, ...), always
    drawn from the vendored font so it scales and themes with the rest of
    the atlas. muse_results.py spelled "this take has stems" as a literal
    U+2713 instead, which falls back to whatever the OS has for that
    codepoint rather than the pinned lucide glyph."""
    offenders = _button_offenders(_QUOTED_CHECK)
    assert offenders == [], offenders


# --- the export/import glyph split -------------------------------------
#
# icons.py states the rule once: DOWNLOAD is export, FOLDER_OPEN is import,
# UPLOAD is neither. These tests pin the files this pass converted so a
# revert reads as a failure instead of a silently absent assertion, the way
# tests/test_ux_consistency_pass3.py pins its own migrated call sites.

_CONVERTED_TO_DOWNLOAD = (
    (pane_files()["inker_tiles.py"], '{icons.DOWNLOAD} Export tileset...'),
    (STUDIO_ROOT / "component_gallery.py", 'icon=icons.DOWNLOAD'),
    (STUDIO_ROOT / "panes" / "packwright_bridge.py", '{icons.DOWNLOAD} {verbs.EXPORT_TO_LIBRARY}'),
    (STUDIO_ROOT / "panes" / "plotter_bridge.py", '{icons.DOWNLOAD} {verbs.EXPORT_TO_LIBRARY}'),
)

_CONVERTED_TO_FOLDER_OPEN = (
    (STUDIO_ROOT / "panes" / "landing.py", '{icons.FOLDER_OPEN} Import mesh...'),
    (STUDIO_ROOT / "panes" / "sirens_instruments.py", '{icons.FOLDER_OPEN} Import...'),
)


def test_export_sites_use_download_not_upload():
    for path, needle in _CONVERTED_TO_DOWNLOAD:
        source = path.read_text(encoding="utf-8")
        assert needle in source, (path, needle)


def test_import_sites_use_folder_open_not_upload():
    for path, needle in _CONVERTED_TO_FOLDER_OPEN:
        source = path.read_text(encoding="utf-8")
        assert needle in source, (path, needle)


def test_upload_is_used_nowhere_in_the_studio_tree():
    """UPLOAD is reserved for neither export nor import (icons.py's rule):
    after this pass every former UPLOAD site in the app became DOWNLOAD or
    FOLDER_OPEN, so the constant now has zero call sites -- which is fine
    (icons.py is a catalogue; an unreferenced constant is not dead code) but
    a regression back to UPLOAD anywhere is exactly the bug this pass fixed."""
    offenders = [
        str(path.relative_to(STUDIO_ROOT))
        for path, source in _studio_sources().items()
        if "icons.UPLOAD" in source
    ]
    assert offenders == [], offenders


def test_clay_outliners_show_all_uses_eye_not_eye_off():
    """"Show all" un-hides everything; drawing it with EYE_OFF said the
    opposite of what the button does. "Solo", right above it, correctly
    uses EYE for the same reason this one now does."""
    source = pane_files()["clay_outliner.py"].read_text(encoding="utf-8")
    assert '{icons.EYE} Show all##clayshowall' in source
    assert '{icons.EYE_OFF} Show all##clayshowall' not in source
