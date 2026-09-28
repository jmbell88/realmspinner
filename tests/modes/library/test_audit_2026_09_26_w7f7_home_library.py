"""Regression tests for the 2026-09-26 audit's Home/Library findings
(shell-home-library-02, -03, -06 and -08).

Kept out of ``test_library_browsing.py`` (which already covers most of this
pane) as its own file, per the repo's own convention of one dated file per
audit fix pass, with a name unique across the tree.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from realmspinner.studio import candidates as candidates_mod
from realmspinner.studio.modes.home.ui.panes import landing
from realmspinner.studio.modes.library.ui.panes import library
from realmspinner.studio.modes.settings.ui.panes import app_settings
from realmspinner.studio.panes import candidates_panel, inspector
from realmspinner.studio.state import AppState


def _asset(**over: Any) -> dict[str, Any]:
    row = {
        "id": "j1",
        "kind": "text",
        "status": "done",
        "stage": "model",
        "name": "",
        "prompt": "",
        "params": {},
        "created_at": 1000.0,
    }
    row.update(over)
    return row


# --- shell-home-library-02: sweep units and undecided candidates ------------


def test_resume_excludes_sweep_units_and_undecided_candidates():
    """A sweep's units and an undecided mesh candidate are both hidden from
    the library by ``Filters.matches`` (state.py) -- dozens of near-identical
    rows meant to be compared against each other, not offered as finished
    work. Home's Resume list read the job cache directly rather than through
    that filter, so a single 20-unit sweep could fill every one of its twelve
    slots. This pins that ``_asset_rows`` applies the same exclusion."""
    jobs = [
        _asset(id="ok", name="a chest", created_at=10.0),
        _asset(id="sweep-unit", name="sweep attempt", created_at=9.0, sweep_id="sweep-1"),
        _asset(id="candidate", name="undecided mesh", created_at=8.0, candidate_group="g1"),
    ]
    ctx = SimpleNamespace(cache=SimpleNamespace(jobs=jobs))

    rows = landing._asset_rows(ctx)

    assert [row.key for row in rows] == ["ok"]


# --- shell-home-library-03: "New character" has no gate ---------------------


def test_new_character_routes_to_settings_when_the_poser_pack_is_missing():
    """``start_poser`` used to call ``set_mode`` with no gate at all, unlike
    Create's own tiles (``_create_door``) -- so on a base install missing the
    Poser (rig) pack, ``poser_mode.ensure`` inside the target mode silently
    returned False and the tile did nothing that told the user why or where
    to go. This pins that a missing pack is caught before ``set_mode`` and
    routed to Settings > Packs instead, the same door the rail and the
    palette already use."""
    state = AppState()
    ctx = SimpleNamespace(
        state=state,
        pack_rows=[
            {
                "key": "rig",
                "label": "Rigging",
                "modes": ["poser"],
                "present": False,
                "download_gib": 0.3,
            }
        ],
        model_rows=[],
        cache=None,  # no finished work anywhere -- the pack gate must not be bypassed
        model_picks=set(),
    )

    landing.start_poser(ctx)

    assert state.mode != "poser"
    assert state.mode == "settings"
    assert state.preview.get(app_settings.CATEGORY_SLOT) == "packs"


def test_new_character_still_opens_poser_when_the_pack_is_present():
    """The gate must not lock out a machine that already has the pack -- the
    ordinary path (``poser_mode.ensure`` + the new-character section opening)
    still has to run when nothing is missing."""
    opened: list[str] = []
    state = AppState()
    ctx = SimpleNamespace(
        state=state,
        pack_rows=[
            {"key": "rig", "label": "Rigging", "modes": ["poser"], "present": True}
        ],
        model_rows=[],
        cache=None,
        model_picks=set(),
    )

    import realmspinner.studio.modes.poser.mode as poser_mode
    from realmspinner.studio import widgets

    real_ensure = poser_mode.ensure
    real_request_open = widgets.request_open
    try:
        poser_mode.ensure = lambda _ctx: SimpleNamespace(docs=[])  # type: ignore[assignment]
        widgets.request_open = lambda name: opened.append(name)  # type: ignore[assignment]
        landing.start_poser(ctx)
    finally:
        poser_mode.ensure = real_ensure  # type: ignore[assignment]
        widgets.request_open = real_request_open  # type: ignore[assignment]

    assert state.mode == "poser"
    assert opened, "expected the new-character section to be requested open"


# --- shell-home-library-06: delete_assets' toast ignores submit failures ----


def test_bulk_delete_toast_counts_only_the_submits_that_actually_succeeded():
    """``delete_assets`` used to ignore each ``submit`` call's return value and
    always toast the full requested count -- so if a delete for one job id was
    already in flight (``submit`` returns False for a duplicate key) the toast
    overstated what happened and the undo payload named a job this call never
    actually trashed."""
    state = AppState()
    toasts: list[tuple[Any, ...]] = []
    submitted: list[str] = []

    def fake_submit(_key: str, _fn: Any, *args: Any, **_kwargs: Any) -> bool:
        job_id = args[-1]
        submitted.append(job_id)
        return job_id != "busy"

    ctx = SimpleNamespace(
        state=state,
        submit=fake_submit,
        toast=lambda *a: toasts.append(a),
        svc=None,
    )

    library.delete_assets(ctx, ["a", "busy", "b"])

    assert submitted == ["a", "busy", "b"]  # every id is still attempted
    assert len(toasts) == 1
    message, _level, _action, payload = toasts[0]
    assert message == "Moved 2 to trash."
    assert payload.split() == ["a", "b"]


def test_bulk_delete_raises_no_toast_when_every_submit_is_dropped():
    """The degenerate case of the same bug: if every submit in the batch was
    dropped, nothing was actually moved, and a toast claiming "Moved 0 to
    trash." would be announcing an act that never happened."""
    state = AppState()
    toasts: list[tuple[Any, ...]] = []
    ctx = SimpleNamespace(
        state=state,
        submit=lambda *_a, **_k: False,
        toast=lambda *a: toasts.append(a),
        svc=None,
    )

    library.delete_assets(ctx, ["a", "b"])

    assert toasts == []


# --- shell-home-library-08: a failing verdict query retries every frame -----


def test_candidates_panels_grades_memoises_a_failing_query_once():
    """``_grades`` returned an unmemoised ``{}`` whenever ``verdicts_for``
    raised, so a failing query was retried -- and re-failed -- on every single
    frame this panel drew, instead of degrading once like the memoized
    success path a few lines below it."""
    candidates_panel._GRADES_CACHE = None  # isolate from whatever ran before this test
    calls: list[int] = []

    def failing_verdicts_for(*_a: Any, **_k: Any) -> Any:
        calls.append(1)
        raise RuntimeError("store is locked")

    ctx = SimpleNamespace(
        cache=SimpleNamespace(_generation=1),
        svc=SimpleNamespace(store=SimpleNamespace(verdicts_for=failing_verdicts_for)),
    )
    group = candidates_mod.Group(group="g1", members=[{"id": "a"}, {"id": "b"}])

    first = candidates_panel._grades(ctx, group)
    second = candidates_panel._grades(ctx, group)

    assert first == {}
    assert second == {}
    assert len(calls) == 1, "the failing query must not be retried every frame"


def test_inspectors_is_graded_memoises_a_failing_query_too():
    """``is_graded``'s own docstring already promises this is memoised, but
    the failing-query arm returned ``True`` without ever writing it into
    ``inspector_graded`` -- so it never actually degraded once; it re-ran and
    re-failed on every frame the verdict header was drawn."""
    calls: list[int] = []

    def failing_verdicts_for(*_a: Any, **_k: Any) -> Any:
        calls.append(1)
        raise RuntimeError("store is locked")

    ctx = SimpleNamespace(
        state=SimpleNamespace(inspector_graded={}),
        svc=SimpleNamespace(store=SimpleNamespace(verdicts_for=failing_verdicts_for)),
    )

    first = inspector.is_graded(ctx, "job-1")
    second = inspector.is_graded(ctx, "job-1")

    assert first is True
    assert second is True
    assert len(calls) == 1, "the failing query must not be retried every frame"
    assert ctx.state.inspector_graded.get("job-1") is True
