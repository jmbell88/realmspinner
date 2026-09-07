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


# --- docs-03: chapter 08's "Measured joints" dead end ----------------------


def test_rig_stage_offers_no_measured_joints_control():
    """The 2026-09-07 audit, finding docs-03: chapter 08 sent a reader stuck
    in the A-pose trap to a "Measured joints" control in the Rig stage. No
    such control exists: create_rig takes no ``joints`` parameter,
    stage_rig.py draws no control by that name, and every
    ``joints="measured"`` site lives inside Troupe's character pipeline,
    which sets it automatically before a rig job is ever created -- two
    call sites there even explicitly *withhold* it. The chapter must not
    send a reader hunting for a button that was never there.
    """
    import inspect

    from warlock.service import rig as rig_service

    signature = inspect.signature(rig_service.create_rig)
    assert "joints" not in signature.parameters, (
        "sanity: create_rig grew a joints parameter -- chapter 08 may now "
        "have a real control to describe instead of denying one"
    )

    stage_rig_src = (
        Path(__file__).resolve().parents[2]
        / "src" / "warlock" / "studio" / "panes" / "stage_rig.py"
    ).read_text(encoding="utf-8")
    assert "Measured joints" not in stage_rig_src, (
        "sanity: stage_rig.py now draws a 'Measured joints' control -- "
        "chapter 08 should describe it rather than deny it"
    )

    trap = _section(_chapter("08-rigging-and-posing.md"), "The A-pose trap")
    assert "**Measured joints**" not in trap, (
        "docs/manual/08-rigging-and-posing.md's A-pose trap section still "
        "offers 'Measured joints' as a control to press, which does not "
        "exist in the Rig stage"
    )
    assert "Adjust joints" in trap, (
        "docs/manual/08-rigging-and-posing.md's A-pose trap section should "
        "still point the reader at Adjust joints, the control that does "
        "exist"
    )


# --- docs-07: chapter 02's asset-type roll call ----------------------------


def test_chapter_02_names_all_five_other_asset_types_including_character():
    """The 2026-09-07 audit, finding docs-07: chapter 02 said "the other
    four entries" and named four, while create_assets._ORDERED holds five
    others -- Character, the one entry needing no GPU, was the one left
    out, which is exactly what a reader without a capable card most needs
    to know.
    """
    from warlock.studio import create_assets

    others = [item for item in create_assets._ORDERED if item.key != "3d_model"]
    assert len(others) == 5, (
        f"sanity: expected five asset types besides 3D Model, found {len(others)}"
    )
    assert any(item.key == "character" for item in others), (
        "sanity: expected a 'character' asset type among the others"
    )

    text = _chapter("02-your-first-asset.md")
    assert "other five entries" in text, (
        "docs/manual/02-your-first-asset.md does not say 'the other five "
        "entries' even though create_assets._ORDERED offers five entries "
        "besides 3D Model"
    )
    assert "Character" in text, (
        "docs/manual/02-your-first-asset.md's asset-type roll call omits "
        "Character, the one entry that needs no GPU"
    )


# --- docs-10: "Teach the judge", matched to the pane -----------------------


def test_manual_ch04_and_ch37_name_the_labelling_section_teach_the_judge():
    """The 2026-09-07 audit, finding docs-10: chapters 04 and 37 called the
    labelling section "Teaching the judge"; the pane draws
    ``widgets.section("Teach the judge")`` (review_panes.py:329). A reader
    hunting for the heading the chapter names would not find it on screen.
    """
    review_panes_src = (
        Path(__file__).resolve().parents[2]
        / "src" / "warlock" / "studio" / "review_panes.py"
    ).read_text(encoding="utf-8")
    assert 'widgets.section("Teach the judge")' in review_panes_src, (
        "sanity: review_panes.py no longer draws a 'Teach the judge' "
        "section -- update this test and the manual together"
    )

    for name in ("04-judging-what-you-made.md", "37-review.md"):
        text = _chapter(name)
        assert "Teaching the judge" not in text, (
            f"docs/manual/{name} still calls the section 'Teaching the "
            "judge', which does not match the pane's 'Teach the judge'"
        )
        assert "Teach the judge" in text, (
            f"docs/manual/{name} does not name the section 'Teach the "
            "judge' to match the pane"
        )


# --- docs-11: the Packwright overflow-menu wording -------------------------


def test_chapter_03_and_32_quote_the_real_add_to_packwright_label():
    """The 2026-09-07 audit, finding docs-11: chapter 03 said the overflow
    entry reads "Add to a Packwright atlas"; ``verbs.add_to("packwright",
    "as an atlas source")`` builds "Add to Packwright as an atlas source"
    (library.py:1034), which chapter 10 already quoted correctly. The same
    stale wording was also sitting in chapter 32, fixed alongside it since
    both are owned by this pass.
    """
    from warlock.studio import verbs

    label = verbs.add_to("packwright", "as an atlas source")
    assert label == "Add to Packwright as an atlas source", (
        f"sanity: verbs.add_to changed shape, got {label!r}"
    )

    for name in ("03-finding-your-work.md", "32-packwright.md"):
        text = _chapter(name)
        assert "Add to a Packwright atlas" not in text, (
            f"docs/manual/{name} still quotes the stale 'Add to a "
            "Packwright atlas' wording"
        )
        # Hand-wrapped prose may break the label across a line; whitespace
        # (including the newline) is not part of the claim being checked.
        normalized = re.sub(r"\s+", " ", text)
        assert label in normalized, (
            f"docs/manual/{name} does not quote the real label {label!r}"
        )


# --- poser-03: root-offset clips are interpolated, not refused -------------


def test_manual_26_does_not_claim_clip_root_offsets_are_refused():
    """The 2026-09-07 audit, finding poser-03: chapter 26 said a clip with
    root offsets "is refused by name rather than rendered subtly wrong",
    but that refusal was deliberately removed and ``resample_clip`` now
    interpolates root translation, pinned by
    ``test_sheet.py::test_a_clip_end_with_a_root_offset_is_accepted_and_interpolated``.
    Stale since 2026-08-19, through a 2026-09-04 edit of the same page.
    """
    from warlock.pipelines import sheet as sheet_pipeline

    keys = [
        {"id": "a", "name": "A", "bones": {}, "root_translation": [0.0, 0.0, 0.0]},
        {"id": "b", "name": "B", "bones": {}, "root_translation": [0.0, 0.0, 0.25]},
    ]
    records = sheet_pipeline.resample_clip(keys, [1], 4)
    offsets = [r["root_translation"][2] for r in records]
    assert offsets[0] == 0.0 and offsets[-1] == 0.25 and offsets == sorted(offsets), (
        f"sanity: resample_clip no longer interpolates root translation "
        f"from one endpoint's offset to the other's, got {offsets}"
    )

    text = _chapter("26-poser.md")
    assert "refused by name" not in text, (
        "docs/manual/26-poser.md still claims a clip with root offsets is "
        "refused by name; resample_clip interpolates it instead"
    )


# --- packwright-04: the Troupe handoff belongs in Packwright's own chapter -


def test_the_packwright_manual_chapter_lists_every_source_door_the_code_has():
    """The 2026-09-07 audit, finding packwright-04: chapter 32 said "Four
    ways in" and never mentioned the Troupe handoff that chapter 10
    documents as a door -- ``troupe_mode.add_to_packwright`` calls
    ``packwright_mode.add_rendered_sheet``, the same bridge. A reader
    consulting Packwright's own reference chapter was told the door did
    not exist.
    """
    from warlock.studio import packwright_mode, troupe_mode

    assert hasattr(packwright_mode, "add_rendered_sheet"), (
        "sanity: packwright_mode lost its rendered-sheet door"
    )
    assert hasattr(troupe_mode, "add_to_packwright"), (
        "sanity: troupe_mode lost its Packwright bridge"
    )

    ch10_bridges = ("From Inker", "From the library", "From Troupe")
    ch10 = _chapter("10-packing-an-atlas.md")
    for bridge in ch10_bridges:
        assert f"**{bridge}**" in ch10, f"sanity: chapter 10 no longer lists {bridge!r}"

    sources = _section(_chapter("32-packwright.md"), "Sources")
    missing = [b for b in ch10_bridges if f"**{b}**" not in sources]
    assert not missing, (
        "docs/manual/32-packwright.md's Sources section is missing the "
        f"door(s) {missing}, which chapter 10 documents and the code has "
        "(troupe_mode.add_to_packwright -> packwright_mode.add_rendered_sheet)"
    )
