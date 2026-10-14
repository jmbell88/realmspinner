"""The 2026-10-14 audit, finding shell-home-library-02: a Home status row that
leaves the trash view carried the trash's ticks into the live list."""

from __future__ import annotations

from types import SimpleNamespace

from realmspinner.studio.modes.home.ui.panes import landing
from realmspinner.studio.state import Filters


def test_a_status_row_that_leaves_the_trash_clears_the_tick_set():
    filters = Filters(trash=True)
    state = SimpleNamespace(
        mode="home",
        previous_mode="home",
        mode_observed="home",
        filters=filters,
        library_scroll_to=None,
        preview={},
        checked={"trashed-1", "trashed-2"},
    )
    ctx = SimpleNamespace(state=state)
    row = landing.Status(
        key="queue", icon="", text="1 running", colour=0, target="library", status_filter="running"
    )

    landing.open_status_row(ctx, row)

    assert filters.trash is False
    assert state.checked == set()
