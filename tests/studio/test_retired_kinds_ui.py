"""What the studio does with a row of a removed job kind (``retired.RETIRED_KINDS``).

The character-sheet kinds (``charsheet``, ``sheet``, ``pixel_sheet``) left
Realmspinner with Poser, but a user's database outlives the feature: the rows
are still there. The studio's half of the promise ``retired.py`` makes is that
such a row is *badged*, offered nothing, and -- when opened -- explains itself on
its own Library card instead of landing on a stage with no mesh. The backend
half (the queue never runs one, no table lists one, no file is deleted) is
``tests/test_retired_kinds.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from realmspinner import retired
from realmspinner.studio import asset_open, state, widgets
from realmspinner.studio.state import AppState

RETIRED = sorted(retired.RETIRED_KINDS)


def _row(kind: str, status: str = "done", **extra):
    return {
        "id": f"{kind}-row",
        "kind": kind,
        "stage": "model",
        "status": status,
        "files": [],
        "params": {"source_job": "MESH", "sheet_id": "abcdef123456"},
        **extra,
    }


@pytest.mark.parametrize("kind", RETIRED)
def test_a_retired_row_is_filed_under_the_retired_kind(kind):
    assert state.card_kind(_row(kind)) == "retired"


@pytest.mark.parametrize("kind", RETIRED)
def test_a_retired_row_wears_the_retired_badge(kind):
    icon, word = widgets.STAGE_BADGES["retired"]
    assert word == "retired"
    # The chip the card draws is keyed on ``card_kind``, so the badge is reached.
    assert widgets.STAGE_BADGES[state.card_kind(_row(kind))] == (icon, word)


@pytest.mark.parametrize("status", ["queued", "running", "done", "error", "cancelled"])
@pytest.mark.parametrize("kind", RETIRED)
def test_a_retired_row_has_no_primary_action_whatever_its_status(kind, status):
    """Not cancel, not retry, not open, not promote: its pipeline is gone."""
    assert state.primary_action(_row(kind, status)) is None
    assert state.primary_action(_row(kind, status), rigging_available=False) is None


@pytest.mark.parametrize("kind", RETIRED)
def test_a_retired_row_exposes_no_result_actions(kind):
    row = _row(kind, files=["input.png", "model.glb"])
    assert state.result_actions(row) == ()


@pytest.mark.parametrize("kind", RETIRED)
def test_a_retired_row_routes_to_the_library_on_itself(kind):
    assert asset_open.route(_row(kind)) == asset_open.Route(
        "library", "", f"{kind}-row", "", ""
    )


def test_a_retired_kind_is_a_library_filter_of_its_own():
    from realmspinner.studio.modes.library.ui.panes import library

    assert ("retired", "retired") in library.KIND_OPTIONS
    assert not [key for key, _label in library.KIND_OPTIONS if key == "sheet"]


@pytest.mark.parametrize("kind", RETIRED)
def test_a_retired_row_is_never_remeshable(kind):
    from realmspinner.studio.modes.library.ui.panes import library

    assert library._remeshable(_row(kind, files=["input.png"])) is False


@pytest.mark.parametrize("kind", RETIRED)
def test_opening_a_retired_row_selects_it_in_the_library_and_says_why(kind):
    toasts: list[tuple] = []
    app_state = AppState()
    app_state.mode = "create"
    row = _row(kind)
    ctx = SimpleNamespace(
        state=app_state,
        cache=SimpleNamespace(get=lambda job_id: row if job_id == row["id"] else None),
        toast=lambda *args, **kwargs: toasts.append((args, kwargs)),
    )

    asset_open.open_asset(ctx, row)

    assert app_state.mode == "library"
    assert app_state.selected == row["id"]
    assert app_state.library_scroll_to == row["id"]
    assert toasts and toasts[-1][0][0] == retired.NOTE


@pytest.mark.parametrize("kind", RETIRED)
def test_a_retired_row_is_not_mistaken_for_an_asset_by_the_open_ladder(kind):
    """``opens_elsewhere`` claims it (so the card, the menu and a double-click
    ask one place), but ``primary_action`` still answers ``None`` for it."""
    assert asset_open.opens_elsewhere(_row(kind))
    assert state.primary_action(_row(kind)) is None


def test_the_live_follow_up_kinds_are_untouched_by_retiring_the_others():
    """Retiring three kinds must not have taken the surviving ones' button."""
    assert state.primary_action(_row("rig")) == "open"
    assert state.card_kind(_row("rig")) == "rig"
    assert state.card_kind(_row("sprite_synthesis")) == "sprite"
