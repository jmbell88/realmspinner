"""The Ctrl+/ sheet must not promise keys Create does not act on.

Create's Up/Down branch in ``studio/shell/events.py`` was made a consumed
``pass`` by the 2026-10-03 audit (Create draws no library list), but the sheet
kept saying "Previous / next asset in the library" -- the one place a user looks
to learn what a key does.
"""

from __future__ import annotations

from realmspinner.studio.shortcuts import shortcut_sections


def test_the_shortcut_sheet_lists_no_up_down_entry_for_create():
    create = dict(shortcut_sections())["Create"]
    offending = [
        (keys, what)
        for keys, what in create
        if {"Up", "Down"} & {part.strip() for part in keys.split("/")}
    ]
    assert not offending, (
        f"the sheet's Create group still lists {offending}, but Create's "
        f"Up/Down keys are consumed and move nothing"
    )
