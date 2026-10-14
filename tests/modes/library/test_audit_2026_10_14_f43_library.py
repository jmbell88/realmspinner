"""The 2026-10-14 audit's Library finding shell-25 (shell-home-library-03).

One dated file with a name unique across the tree.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from _ui_context import imgui_context

from realmspinner.studio import probe, widgets
from realmspinner.studio.modes.library.ui.panes import library
from realmspinner.studio.state import AppState, Filters


@pytest.fixture
def ui(monkeypatch):
    with imgui_context(monkeypatch) as imgui:
        yield imgui


def _draw_scope_row(ui, ctx):
    probe.begin_frame()
    ui.new_frame()
    ui.set_next_window_size((400.0, 300.0))
    ui.set_next_window_pos((0.0, 0.0))
    ui.begin("##host")
    drew = library.scope_row(ctx)
    ui.end()
    ui.end_frame()
    return drew, [c for c in probe.FRAME_CONTROLS if c.kind == "button"]


def test_the_library_offers_a_way_out_of_a_scoped_examples_view(ui):
    """Review's "Show examples" scopes ``Filters.job_ids`` and nothing in the
    Library said so or lifted it; only a restart did."""
    state = AppState()
    ctx = SimpleNamespace(state=state)
    state.filters.job_ids = frozenset({"a", "b", "c"})

    # It narrows the list, so it counts as filtering.
    assert library.filtering(state.filters)

    drew, buttons = _draw_scope_row(ui, ctx)
    assert drew
    show_all = [c for c in buttons if c.label.startswith("Show all")]
    assert show_all and show_all[0].enabled

    # What the button does.
    library.clear_scope(state.filters)
    assert state.filters.job_ids is None
    assert not library.filtering(state.filters)

    drew, buttons = _draw_scope_row(ui, ctx)
    assert not drew
    assert not [c for c in buttons if c.label.startswith("Show all")]


def test_a_scoped_list_that_is_empty_names_the_scope_not_the_filters(monkeypatch):
    said: list[tuple[str, str]] = []
    monkeypatch.setattr(
        widgets, "empty_state", lambda icon, title, body="": said.append((title, body))
    )
    ctx = SimpleNamespace(
        state=SimpleNamespace(filters=Filters(job_ids=frozenset({"gone"}))),
        cache=SimpleNamespace(jobs=[{"id": "other"}]),
    )

    library._empty(ctx)

    assert said and "Loosen the filters" not in said[0][1]
    assert "Show all" in said[0][1]
