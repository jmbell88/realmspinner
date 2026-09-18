"""Settings search: whether a plain-language guess reaches the right row.

``tests/studio/test_startup_last_workspace_and_settings_search.py`` covers the
happy-path synonyms already ("bigger text" -> UI scale); this file is for the
2026-09-18 audit's own finding about the matcher's word order.
"""

from __future__ import annotations

from warlock.studio.modes.settings.ui.panes import app_settings


def test_settings_search_synonym_matches_partial_phrasing():
    """The 2026-09-18 audit, shell-09: ``_row_matches`` required ``needle`` to
    be a literal substring of the stored synonym phrase, so "bigger text"
    found UI scale but "make text bigger" -- just as plain a phrasing, with
    the same two words in a different order -- found nothing at all."""
    found = app_settings.search_rows("make text bigger")
    assert any(row.label == "UI scale" for row in found), (
        "'make text bigger' should reach the UI scale row the same way "
        "'bigger text' does"
    )
