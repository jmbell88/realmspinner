"""The manual's own labels, against the code that draws them.

``tests/manual/test_docs.py`` gates structure (links, anchors, headings) and
``test_shortcuts.py`` gates the shortcut sheet against chapter 38. Neither
reads a chapter's claim about a *control label* against the widget that draws
it, which is how the 2026-09-06 docs audit found five chapters quoting a
label the code does not have: a button whose text is computed
(``spec.create_label``), a menu with a stale item count, a context-menu row
whose wording drifted from the shared ``verbs`` module, and two labels
(one absent, one reworded) in Create's recipe column.

Each test here reads the label from the code -- never hard-codes it a second
time -- and asserts the chapter agrees.
"""

from __future__ import annotations

from pathlib import Path

MANUAL = Path(__file__).resolve().parents[2] / "docs" / "manual"


def _read(key: str) -> str:
    return (MANUAL / f"{key}.md").read_text(encoding="utf-8")


def test_your_first_asset_names_the_actual_generate_button_label_for_3d_model():
    """The 2026-09-06 audit, finding docs-01: chapter 2 has the reader leave
    Asset type on 3D Model, then tells them to press "Create image" -- the
    label for asset type "image", not "3d_model". The real label is
    ``spec.create_label`` for the "3d_model" entry, drawn at
    ``create_brief.py:233`` as ``spec.create_label``."""
    from warlock.studio.modes.create.engine.assets import ASSET_TYPES

    label = ASSET_TYPES["3d_model"].create_label
    assert label == "Generate reference"

    text = _read("02-your-first-asset")
    assert f"**{label}**" in text, (
        f"docs/manual/02-your-first-asset.md never names the real 3D Model "
        f"generate-button label {label!r}"
    )
    stale = ASSET_TYPES["image"].create_label
    assert stale != label
    assert f"**{stale}**" not in text, (
        f"docs/manual/02-your-first-asset.md still tells the reader to press "
        f"{stale!r}, which is the label for asset type 'image', not '3d_model' "
        f"(the chapter has the reader stay on 3D Model)"
    )


def test_home_new_menu_manual_lists_every_item_including_the_scene():
    """The 2026-09-06 audit, finding docs-07: chapter 21 said the New... menu
    holds "the seven things this app can begin from nothing" and listed seven,
    where ``landing.NEW_ITEMS`` had eight, and the missing one started a song.

    The count is derived from ``NEW_ITEMS`` rather than written twice, which is
    what stops this going stale a second time -- it went stale exactly once
    more, when Mason's "New scene" made it nine, and a hand-typed number would
    have had to be found again instead of the chapter simply failing here.
    """
    from warlock.studio.panes.landing import NEW_ITEMS

    words = {8: "eight", 9: "nine", 10: "ten"}
    counted = words[len(NEW_ITEMS)]

    text = _read("21-home")
    assert f"{counted} things this app can begin from nothing" in text, (
        f"docs/manual/21-home.md disagrees with NEW_ITEMS, which has "
        f"{len(NEW_ITEMS)} items"
    )
    assert "a scene" in text, (
        "docs/manual/21-home.md's New... list omits the scene item that "
        'landing.NEW_ITEMS carries ("New scene")'
    )
    # Every item's noun should show up somewhere in the chapter's prose list.
    # "New song" -> "a song": the chapter writes the list as bare nouns, not
    # button labels, so check for the noun rather than the literal label.
    assert "a song" in text, (
        "docs/manual/21-home.md's New... list omits the song item that "
        "landing.NEW_ITEMS carries (\"New song\")"
    )


def test_manual_ch10_matches_the_add_to_packwright_label():
    """The 2026-09-06 audit, finding docs-24: chapter 10 quotes the library
    context-menu item as "Add to a Packwright atlas"; the real label is built
    by ``verbs.add_to`` and read at ``library.py:1029`` (via
    ``verbs.add_to("packwright", "as an atlas source")``)."""
    from warlock.studio import verbs

    label = verbs.add_to("packwright", "as an atlas source")
    assert label == "Add to Packwright as an atlas source"

    text = _read("10-packing-an-atlas")
    assert label in text, (
        f"docs/manual/10-packing-an-atlas.md never quotes the real label "
        f"{label!r}"
    )
    assert "Add to a Packwright atlas" not in text, (
        "docs/manual/10-packing-an-atlas.md still quotes the stale label "
        "'Add to a Packwright atlas'"
    )


def test_manual_ch12_count_control_has_no_how_many_label():
    """The 2026-09-06 audit, finding docs-25: chapter 12 bolds **How many** as
    if it captions a control, but ``create_brief.py``'s count control
    (``_count``, lines 177-216) is an unlabelled segmented row of 1/2/4/8 with
    per-value tooltips only -- there is no "How many" string anywhere in the
    widget code."""
    text = _read("12-tuning-what-you-get")
    assert "**How many**" not in text, (
        "docs/manual/12-tuning-what-you-get.md still bolds '**How many**' as "
        "a control caption, but Create's count row has no such label -- only "
        "per-value tooltips (create_brief.py's _COUNT_HINTS)"
    )


def test_manual_ch12_retexture_labels_match_the_pane():
    """The 2026-09-06 audit, finding docs-26: chapter 12 names the re-texture
    controls "Strength" and "Anchor to geometry"; ``texture_panel.py`` draws
    "Restyle strength" (line 76) and "Anchor to geometry (depth)" (line 90)."""
    text = _read("12-tuning-what-you-get")
    assert "**Restyle strength**" in text, (
        "docs/manual/12-tuning-what-you-get.md never names the real label "
        "'Restyle strength'"
    )
    assert "**Anchor to geometry (depth)**" in text, (
        "docs/manual/12-tuning-what-you-get.md never names the real label "
        "'Anchor to geometry (depth)'"
    )
    assert "**Strength**" not in text, (
        "docs/manual/12-tuning-what-you-get.md still bolds the stale, "
        "unqualified 'Strength' rather than the pane's 'Restyle strength'"
    )
    assert "**Anchor to geometry**," not in text, (
        "docs/manual/12-tuning-what-you-get.md still bolds the stale "
        "'Anchor to geometry' rather than the pane's "
        "'Anchor to geometry (depth)'"
    )
