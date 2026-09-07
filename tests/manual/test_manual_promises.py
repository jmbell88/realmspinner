"""Chapter promises against the code that owns them.

Four findings from the 2026-09-06 docs audit: manual chapters that describe a
zoom ceiling, a purged ``examples/`` directory, a skeleton template count and
an "only place" claim, none of which agree any more with the modules that
decide them. ``tests/manual/test_docs.py`` gates the manual's *structure*
(links, anchors, headings) and ``test_shortcuts.py`` gates the shortcut popup
against chapter 38; neither reads a chapter's claims against the app code
that makes them true or false. This file does, one finding per test, each
reading its numbers from the module that owns them rather than repeating a
second hand-written copy.
"""

from __future__ import annotations

import re
from pathlib import Path

MANUAL = Path(__file__).resolve().parents[2] / "docs" / "manual"


def _chapter(name: str) -> str:
    return (MANUAL / name).read_text(encoding="utf-8")


def _section(text: str, heading: str) -> str:
    """The prose under one ``##`` heading, up to the next ``##`` (or EOF).

    Matches on a bare ``## <heading>`` line so a ``###`` subheading inside the
    section (three hashes, not two-then-space) does not end it early.
    """
    pattern = rf"^## {re.escape(heading)}\s*$"
    match = re.search(pattern, text, re.MULTILINE)
    assert match, f"no '## {heading}' heading found"
    rest = text[match.end():]
    nxt = re.search(r"^## ", rest, re.MULTILINE)
    return rest[: nxt.start()] if nxt else rest


def _has_percent(text: str, token: str) -> bool:
    """Whether ``token`` (e.g. ``"12.5"``) appears in ``text`` as its own
    percent figure, not as a substring of a longer number."""
    pattern = rf"(?<![\d.]){re.escape(token)}%(?!\d)"
    return re.search(pattern, text) is not None


# --- docs-04: Inker's zoom ceiling, ladder and status-bar presets ----------


def test_manual_ch28_zoom_ceiling_matches_inker_state_constant():
    """The chapter's stated ceiling (and floor) must be Inker's real ones.

    inker_state.py:37-44 records the deliberate 10x -> 64x reversal; the
    chapter's Zooming section was never revisited and still says the zoom
    "stops ... at 1000%" (six times below the real 6400% ceiling).
    """
    from warlock.studio import inker_state

    ceiling_pct = inker_state.zoom_key(inker_state.INKER_MAX_ZOOM)
    floor_pct = inker_state.zoom_key(inker_state.INKER_MIN_ZOOM)
    zooming = _section(_chapter("28-inker.md"), "Zooming")

    assert _has_percent(zooming, ceiling_pct), (
        f"docs/manual/28-inker.md's Zooming section does not state the real "
        f"ceiling of {ceiling_pct}% (inker_state.INKER_MAX_ZOOM="
        f"{inker_state.INKER_MAX_ZOOM})"
    )
    assert _has_percent(zooming, floor_pct), (
        f"docs/manual/28-inker.md's Zooming section does not state the real "
        f"floor of {floor_pct}% (inker_state.INKER_MIN_ZOOM="
        f"{inker_state.INKER_MIN_ZOOM})"
    )


def test_manual_ch28_zoom_ladder_matches_ZOOM_LADDER():
    """The +/- ladder list must enumerate every rung of ZOOM_LADDER.

    The chapter's list currently ends at 1000%; ZOOM_LADDER
    (inker_state.py:96-99) runs six rungs further, to 6400%.
    """
    from warlock.studio import inker_state

    zooming = _section(_chapter("28-inker.md"), "Zooming")
    missing = [
        inker_state.zoom_key(rung)
        for rung in inker_state.ZOOM_LADDER
        if not _has_percent(zooming, inker_state.zoom_key(rung))
    ]
    assert not missing, (
        "docs/manual/28-inker.md's +/- ladder list is missing rungs present "
        f"in inker_state.ZOOM_LADDER: {missing}% "
        f"(full ladder: {[inker_state.zoom_key(r) for r in inker_state.ZOOM_LADDER]}%)"
    )


def test_manual_ch28_status_bar_picker_matches_ZOOM_PRESETS():
    """The status-bar zoom picker list must enumerate every entry of
    ZOOM_PRESETS. The chapter's list currently ends at 800%; ZOOM_PRESETS
    (inker_state.py:105) runs to 6400%.
    """
    from warlock.studio import inker_state

    status_bar = _section(_chapter("28-inker.md"), "The status bar")
    missing = [
        inker_state.zoom_key(preset)
        for preset in inker_state.ZOOM_PRESETS
        if not _has_percent(status_bar, inker_state.zoom_key(preset))
    ]
    assert not missing, (
        "docs/manual/28-inker.md's status-bar picker list is missing "
        f"presets present in inker_state.ZOOM_PRESETS: {missing}% "
        f"(full presets: {[inker_state.zoom_key(p) for p in inker_state.ZOOM_PRESETS]}%)"
    )


# --- docs-06: the purged, gitignored Packwright examples/ directory --------


def test_packwright_manual_does_not_promise_a_gitignored_examples_directory():
    """``examples/`` was purged and gitignored on 2026-09-03 (.gitignore:84,
    because the underlying assets are third-party and unlicensed for
    redistribution), so a checkout has no such directory. The chapter must
    not send the reader looking for it.
    """
    gitignore = (MANUAL.parent.parent / ".gitignore").read_text(encoding="utf-8")
    assert "/examples/" in gitignore, "sanity: examples/ expected gitignored"

    text = _chapter("32-packwright.md")
    assert "examples/" not in text, (
        "docs/manual/32-packwright.md still points the reader at examples/, "
        "which .gitignore excludes from every checkout"
    )


# --- docs-08: the eighth shipped skeleton template, Blob -------------------


def test_templates_table_lists_all_eight_shipped_skeletons_including_blob():
    """rigging.catalog() offers every shipped template unfiltered to both
    skeleton pickers, including blob.json ("Blob (amorphous)"), the eighth
    template added alongside the other seven. The chapter's table and its
    "seven" still enumerate only the original seven.
    """
    from warlock import rigging

    catalog = rigging.catalog()
    assert len(catalog) == 8, f"sanity: expected eight shipped templates, found {len(catalog)}"

    text = _chapter("25-rigging-and-posing.md")
    templates_section = _section(text, "Templates")
    lowered = templates_section.lower()

    missing = [
        entry["label"]
        for entry in catalog
        if entry["key"].lower() not in lowered
        and entry["label"].split()[0].lower() not in lowered
    ]
    assert not missing, (
        "docs/manual/25-rigging-and-posing.md's Templates table does not "
        f"mention the shipped template(s) {missing}"
    )
    assert "eight" in lowered, (
        "docs/manual/25-rigging-and-posing.md's Templates section still says "
        "the app fits one of seven shipped templates; rigging.catalog() "
        "offers eight"
    )


# --- docs-09: the empty-viewport "only place" claim ------------------------


def test_create_viewport_placeholder_is_not_the_only_place_ctrl_n_appears():
    """overlay.PLACEHOLDERS gives the "Ctrl+N starts one, Ctrl+O opens a
    file" line to inker, plotter and sirens (overlay.py:466,474,488) -- not
    to create/mesh, the 3D viewport's own empty-state placeholder, which
    names neither shortcut. The chapter's "only place in the app" claim is
    false in both directions and must not survive.
    """
    from warlock.studio.panes import overlay

    mesh_hint = overlay.PLACEHOLDERS["create/mesh"][2]
    assert "Ctrl+N" not in mesh_hint and "Ctrl+O" not in mesh_hint, (
        "sanity: the 3D viewport's own placeholder was expected to carry "
        "neither shortcut"
    )

    elsewhere = [
        key
        for key in ("inker", "plotter", "sirens")
        if "Ctrl+N" in overlay.PLACEHOLDERS[key][2]
        and "Ctrl+O" in overlay.PLACEHOLDERS[key][2]
    ]
    assert elsewhere, (
        "sanity: expected at least one other placeholder to carry the "
        "Ctrl+N/Ctrl+O line the chapter claims is unique to the 3D viewport"
    )

    empty_view = _section(_chapter("24-the-3d-viewport.md"), "When the view is empty")
    assert "only place" not in empty_view.lower(), (
        "docs/manual/24-the-3d-viewport.md claims its placeholder is the "
        f"only place Ctrl+N/Ctrl+O appear on screen, but overlay.PLACEHOLDERS "
        f"gives that exact line to {elsewhere} and not to create/mesh itself"
    )
