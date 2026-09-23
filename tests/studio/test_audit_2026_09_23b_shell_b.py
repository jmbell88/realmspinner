"""Regressions for the shell-b slice of the 2026-09-23 (second run) audit:
shell-06 (Home's Resume memo), shell-07 (the inspector's manifest cache),
tour-01 (the tour's off-the-end stale card) and agents-01 (the quit chain
missing an in-flight agent tool call).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

# --- shell-06 ----------------------------------------------------------------


def test_landing_rows_key_does_not_collide_on_a_freed_caches_reused_id():
    """shell-06 (2026-09-23, second run, audit): ``_rows_key`` keyed its memo
    on ``id(cache)`` alone -- the exact hazard create-05 (2026-09-20) fixed in
    ``candidates.pending_cached`` and ``candidates_panel._grades`` (19,992 of
    20,000 wrong hits in that audit's probe). CPython is free to hand a freed
    cache's address to a brand new, unrelated cache, so a bare id can name an
    object that no longer exists; only a strong reference to the cache itself
    can never be fooled that way.

    An actual GC-timed address collision is flaky to force in one test, so
    this plants the exact situation an address reuse produces: a memo entry
    keyed on the id of a cache that is very much alive right now, but which
    was never the cache that produced the memoized rows.
    """
    from realmspinner.studio.modes.home.ui.panes import landing

    class _FakeCache:
        def __init__(self, generation: int) -> None:
            self._generation = generation
            self.jobs: list[dict[str, Any]] = []

    live = _FakeCache(0)
    ctx = SimpleNamespace(cache=live)
    stale_row = landing.Row(kind="asset", key="ghost", icon="x", name="ghost")

    saved = landing._ROWS_CACHE
    try:
        # What a destroyed cache's memo looks like if its id is reused: a key
        # built from a bare id, planted under the id the *live* object now
        # happens to occupy.
        landing._ROWS_CACHE = ((id(live), 0, ()), [stale_row])
        found = landing.rows(ctx)
    finally:
        landing._ROWS_CACHE = saved

    assert found != [stale_row], (
        "rows() served a different cache's stale memo because _rows_key "
        "trusted a bare id(cache) instead of a strong reference to it"
    )


# --- shell-07 ------------------------------------------------------------


def test_manifest_does_not_serve_the_previous_jobs_manifest_after_the_selection_moves_to_a_different_job(  # noqa: E501
    tmp_path: Any,
):
    """shell-07 (2026-09-23, second run, audit): the in-flight fallback at the
    bottom of ``_manifest`` returned whatever ``ctx.state.manifest`` cached
    without checking it was an answer about *this* job -- so selecting a
    different job while a read for the previous one was still in flight (or
    had just been submitted this frame) served the previous job's provenance
    for a frame or two.
    """
    from realmspinner.studio.panes import inspector

    class _FakeFuture:
        def done(self) -> bool:
            return False

    class _FakeExecutor:
        def submit(self, fn: Any, path: Any) -> _FakeFuture:
            return _FakeFuture()

    ctx = SimpleNamespace(
        job_dir=lambda job_id: tmp_path / job_id,
        state=SimpleNamespace(manifest=(("jobA", 100), "jobA's manifest")),
    )

    saved_pool = inspector._manifest_pool
    saved_inflight = dict(inspector._manifest_inflight)
    inspector._manifest_pool = _FakeExecutor()
    inspector._manifest_inflight = {}
    try:
        # ``stamp_ns`` needs a real mtime for a job dir that does not exist on
        # disk to fake a "still in flight" read for jobB -- monkeypatch it
        # directly rather than touch the filesystem.
        orig_stamp_ns = inspector.stamps.stamp_ns
        inspector.stamps.stamp_ns = lambda path: 999
        try:
            result = inspector._manifest(ctx, "jobB")
        finally:
            inspector.stamps.stamp_ns = orig_stamp_ns
    finally:
        inspector._manifest_pool = saved_pool
        inspector._manifest_inflight = saved_inflight

    assert result is None, (
        "_manifest served jobA's cached manifest for a request about jobB "
        "-- the in-flight fallback must not answer for a job it was never "
        "asked about"
    )


# --- tour-01 ---------------------------------------------------------------


def test_draw_and_advance_clear_the_stale_card_rect_when_the_running_tours_key_no_longer_resolves():  # noqa: E501
    """tour-01 (2026-09-23, second run, audit): two off-the-end branches --
    ``advance`` when ``find_tour(state.key)`` returns ``None``, and ``draw``
    when the tour or its step no longer resolves -- called
    ``ctx.state.tour.stop()`` directly instead of the module's own
    ``stop(ctx)``, which also runs ``_clear_card()``. That is the exact
    stale-hole bug the 2026-09-15 audit's tour-01 fixed for the run-off-the-
    end-of-the-tour path; these two branches kept the old, unfixed shape.
    """
    from realmspinner.studio.panes import tour as tour_mod

    class _FakeTourState:
        def __init__(self, key: str) -> None:
            self.key = key
            self.running = True
            self.index = 0
            self.satisfied = False

        def stop(self) -> None:
            self.running = False

        def complete(self) -> None:
            self.running = False

    tour_mod._card_rect[0] = (1.0, 2.0, 3.0, 4.0)
    tour_mod._card_focused[0] = True
    state = _FakeTourState("no-such-tour")
    ctx = SimpleNamespace(state=SimpleNamespace(tour=state))

    tour_mod.advance(ctx)

    assert tour_mod._card_rect[0] is None, (
        "advance() left the stale card rect in place after a tour key that "
        "no longer resolves -- it must route through stop(ctx), not "
        "state.stop() directly"
    )


def test_draw_clears_the_stale_card_rect_when_the_running_tours_key_no_longer_resolves():
    """The ``draw`` half of tour-01's second branch."""
    from realmspinner.studio.panes import tour as tour_mod

    class _FakeTourState:
        def __init__(self, key: str) -> None:
            self.key = key
            self.running = True

        def stop(self) -> None:
            self.running = False

    tour_mod._card_rect[0] = (1.0, 2.0, 3.0, 4.0)
    tour_mod._card_focused[0] = True
    state = _FakeTourState("no-such-tour")
    ctx = SimpleNamespace(state=SimpleNamespace(tour=state))

    tour_mod.draw(ctx)

    assert tour_mod._card_rect[0] is None, (
        "draw() left the stale card rect in place after a tour key that no "
        "longer resolves -- it must route through stop(ctx), not "
        "state.stop() directly"
    )


# --- agents-01 ---------------------------------------------------------------


def test_quit_summary_warns_while_an_agent_character_export_is_busy():
    """agents-01 (2026-09-23, second run, audit): character-tool calls run on
    ``AgentHost``'s own service-lane ``TaskRunner``, invisible to
    ``ctx.tasks.busy_keys`` -- so quitting mid ``character_export`` warned
    nothing, unlike every other in-flight write ``_quit_summary`` already
    names.
    """
    from realmspinner.studio import main as main_mod

    app = main_mod.App.__new__(main_mod.App)
    app.runtime = SimpleNamespace(current_job_id=None)
    app.app_ctx = SimpleNamespace(
        cache=SimpleNamespace(active=None),
        tasks=SimpleNamespace(busy_keys=set()),
    )
    app.agent_host = SimpleNamespace(busy_tools=("character_export",))

    summary = app._quit_summary()

    assert summary, "an in-flight character_export must not pass through silently"
    assert "character_export" in summary


def test_quit_summary_says_nothing_extra_when_no_agent_tool_is_busy():
    """The companion case: an idle agent host adds no line (UX-21's own
    reasoning -- a warning about a thing that is not happening teaches people
    to click through warnings)."""
    from realmspinner.studio import main as main_mod

    app = main_mod.App.__new__(main_mod.App)
    app.runtime = SimpleNamespace(current_job_id=None)
    app.app_ctx = SimpleNamespace(
        cache=SimpleNamespace(active=None),
        tasks=SimpleNamespace(busy_keys=set()),
    )
    app.agent_host = SimpleNamespace(busy_tools=())

    assert app._quit_summary() == ""
