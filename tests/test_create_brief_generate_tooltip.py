"""The Generate button's tooltip restates the count once its pills are gone.

The 2026-09-07 Create review, item 5.9: ``_row_widths`` drops the count pills
at the resize floor and used to leave the button's tooltip a bare
``"Ctrl+Enter"`` regardless -- and the pills' other echo, the settings
column's "N candidates" plan block, is itself a ``layout`` pane a person can
collapse. At a small window with that column hidden, the count was then
stated nowhere on screen at all while Generate went on drawing however many
candidates the hidden control still held.

A new file rather than an addition to ``test_create_brief.py``: this review
touches ``create_brief.py`` while a parallel pass is editing that test file's
neighbours, and a second writer in one file is exactly the collision this
suite's own file-ownership rule exists to avoid.
"""

from __future__ import annotations

import inspect

from warlock.studio import create_brief


def test_the_generate_tooltip_states_the_count_when_the_pills_are_hidden():
    assert create_brief._generate_tooltip(True, 4) == "Ctrl+Enter"
    assert create_brief._generate_tooltip(False, 1) == "Ctrl+Enter · 1 candidate"
    assert create_brief._generate_tooltip(False, 4) == "Ctrl+Enter · 4 candidates"
    assert create_brief._generate_tooltip(False, 8) == "Ctrl+Enter · 8 candidates"


def test_the_tooltip_says_nothing_extra_while_the_pills_are_visible():
    # A tooltip repeating a control drawn an inch to its left is noise.
    assert create_brief._generate_tooltip(True, 1) == "Ctrl+Enter"
    assert create_brief._generate_tooltip(True, 8) == "Ctrl+Enter"


def test_the_shortcut_hint_stays_first_when_the_count_is_appended():
    tooltip = create_brief._generate_tooltip(False, 2)
    assert tooltip.startswith("Ctrl+Enter")
    assert " · " in tooltip


def test_the_generate_button_is_wired_to_the_computed_tooltip_not_a_literal():
    """The regression: ``_generate`` used to hardcode ``tooltip="Ctrl+Enter"``,
    which was right while the pills were visible and silently wrong -- not
    merely incomplete -- the moment ``_row_widths`` dropped them, since
    nothing else on the bar or (at this width) in the settings column said how
    many candidates a press would draw."""
    body = inspect.getsource(create_brief._generate)
    assert 'tooltip="Ctrl+Enter"' not in body
    assert "_generate_tooltip(show_count" in body


def test_generate_is_called_with_the_row_widths_own_show_count_answer():
    """``show_count`` must be ``_row_widths``' answer, not re-derived: a second
    reading of "is the count visible" is exactly how the two could disagree.

    ``_row_widths`` grew two more return values when the rail and Reset
    joined the row (2026-09-07: ``rail_w`` and ``reset_compact``), so the
    call site's exact left-hand side changed shape -- what still has to hold
    is that ``show_count`` comes off that one call rather than being asked
    again."""
    body = inspect.getsource(create_brief.draw)
    assert "show_count, reset_compact = _row_widths(" in body
    assert "hide_count, rail_full_w, rail_floor_w" in body
    assert "show_count=show_count" in body
