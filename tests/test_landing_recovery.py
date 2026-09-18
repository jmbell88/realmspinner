"""Home's recovery rows: what a headless test can hold about their layout.

The defect was screenshot-class -- ``_recovery_row`` drew ``entry.title``
untrimmed and then right-aligned the Recover button over it, so a long title
ran straight under the button. The geometry cannot be asserted without a GL
context, but the call-site properties that produced it can be, which is the
same posture ``tests/inker/test_inker_ux.py`` takes for its pane scans.
"""

from __future__ import annotations

import inspect

from warlock.studio.modes.home.ui.panes import landing


def test_the_recovery_title_is_trimmed_not_drawn_raw() -> None:
    """Trimmed like the Resume grid's cells -- measured via ``fit_text``, not
    counted -- to the room left after the note beside it and the right-aligned
    button are both reserved."""
    source = inspect.getsource(landing._recovery_row)
    assert "widgets.fit_text(entry.title" in source
    assert "imgui.text(entry.title)" not in source
    # The reservation names both fixed occupants of the row.
    assert "RECOVER_BUTTON" in source
    assert "note_w" in source


def test_the_trim_happens_before_the_button_is_placed() -> None:
    """Order is the mechanism: a trim computed after ``same_line`` to the
    button's column would be measuring a region the title no longer owns."""
    source = inspect.getsource(landing._recovery_row)
    assert source.index("fit_text") < source.index("get_cursor_pos_x")


def test_an_empty_resume_region_offers_a_starter_not_a_sentence() -> None:
    """The empty Resume region used to read "Nothing yet. Start something
    above, or press Ctrl+K." -- a sentence pointing the reader somewhere
    else on the screen instead of giving them something to press right
    there. It must offer a starter of its own, routed through the same
    ``NEW_ITEMS`` callbacks ``_start()`` uses (reused, not redefined), with
    the Ctrl+K hint kept on as muted text alongside it."""
    resume_source = inspect.getsource(landing._resume)
    assert "Nothing yet" not in resume_source
    assert "_resume_empty" in resume_source

    empty_source = inspect.getsource(landing._resume_empty)
    assert "for key, label, icon, action in NEW_ITEMS:" in empty_source
    assert "action(ctx)" in empty_source
    assert "Ctrl+K" in empty_source
    assert "widgets.muted" in empty_source
