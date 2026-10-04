"""The 2026-10-03 audit's Medium shell findings, batch one: boot, chrome and
documents (shell-13 .. shell-20).

Each test's name is the claim, and each was run against the unfixed tree first.
shell-32 (the manual's boot-path wording) is a docs finding with no code half.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

from realmspinner.studio import jobs_cache, layouts, palette, toolbar
from realmspinner.studio import layout as layout_mod
from realmspinner.studio.state import Filters, parse_query

# --- shell-13: a failing job-list read must not be retried every frame --------


class _Runner:
    """Accepts every submit and remembers it; the read itself is never run."""

    def __init__(self) -> None:
        self.keys: list[str] = []

    def submit(self, key: str, fn: Any, *args: Any, **kwargs: Any) -> bool:
        self.keys.append(key)
        return True


def test_a_failed_job_list_read_backs_off_instead_of_resubmitting_every_frame():
    cache = jobs_cache.JobsCache(SimpleNamespace())
    runner = _Runner()

    assert cache.request(runner) is True
    cache.adopt({"error": "database is locked"})

    # The very next frame: the read just failed, so asking again now is the
    # per-frame retry (and per-frame traceback) the audit found.
    assert cache.request(runner) is False, "a failed read was resubmitted on the next frame"
    assert runner.keys == ["jobs-list"]
    assert cache.error == "database is locked"


# --- shell-14: an infinite stored number is junk, not a boot loop -------------


def test_an_infinite_stored_window_size_falls_back_to_the_default():
    from realmspinner.studio import main

    stored = json.loads("[1e999, 1e999]")
    assert math.isinf(stored[0])
    default = main._window_size(None, override=None, first_run_scale=1.0, desktop=None)
    got = main._window_size(stored, override=None, first_run_scale=1.0, desktop=None)
    assert got == default


def test_an_infinite_stored_pixel_preference_falls_back_to_the_default():
    from realmspinner.studio.app_ctx import pixel_prefs

    data = json.loads('{"pixel_size": 1e999, "pixel_colors": 1e999}')
    size, colors, _palette, _dither = pixel_prefs(SimpleNamespace(get=data.get))
    assert (size, colors) == (128, 0)


def test_an_infinite_stored_number_does_not_crash_form_restore():
    from realmspinner.studio import state

    assert state._restore_int(float("inf"), 7) == 7
    form = state.default_form_2d()
    key = next(k for k, v in form.items() if isinstance(v, int) and not isinstance(v, bool))
    restored = state.form_from_params({key: float("inf")})  # must not raise
    # (the default is not compared: the form's seed default is random)
    assert isinstance(restored[key], int) and restored[key] != float("inf")


# --- shell-15: a lowercase or unknown log level must not kill startup ---------


@pytest.mark.parametrize(
    ("raw", "level"),
    [("debug", logging.DEBUG), ("WARNING", logging.WARNING), ("nonsense", logging.INFO)],
)
def test_an_unknown_log_level_falls_back_to_info_instead_of_failing_startup(
    monkeypatch, raw, level
):
    import realmspinner.config as config_mod
    from realmspinner.studio import main

    root = logging.getLogger()
    before_handlers = list(root.handlers)
    before_level = root.level
    monkeypatch.setenv("REALMSPINNER_LOG_LEVEL", raw)

    def no_home() -> Any:
        raise OSError("no data dir in this test")

    # File logging is out of scope here (and would arm faulthandler); the level
    # parse is what ``basicConfig`` choked on.
    monkeypatch.setattr(config_mod, "get_config", no_home)
    try:
        main._setup_logging()
        assert root.level == level
    finally:
        for handler in list(root.handlers):
            if handler not in before_handlers:
                root.removeHandler(handler)
        root.handlers[:] = before_handlers
        root.setLevel(before_level)


# --- shell-16: apostrophes and backslashes are text, not shell quoting --------


def test_a_filter_phrase_with_apostrophes_matches_the_prompt_it_was_copied_from():
    prompt = "knight's sword and dragon's lair"
    job = {"id": "j1", "name": "", "prompt": prompt, "tags": "", "status": "done"}
    assert Filters(text=prompt).matches(job)

    windows = r"C:\art\knight"
    job = {"id": "j2", "name": windows, "prompt": "", "tags": "", "status": "done"}
    assert Filters(text=windows).matches(job)


def test_double_quotes_still_group_and_an_unclosed_one_is_closed_for_the_typist():
    assert parse_query("it's a 'test'") == (["it's", "a", "'test'"], [])
    assert parse_query('name:"a wooden chest" rusty') == (["rusty"], [("name", "a wooden chest")])
    assert parse_query('name:"a wooden') == ([], [("name", "a wooden")])


# --- shell-17: Generate is shut where Ctrl+Enter does nothing -----------------


def _palette_ctx(stage: str) -> Any:
    from studio.test_palette import _ctx

    return _ctx("create", stage=stage)


@pytest.mark.parametrize("stage", ["rig", "pose", "export"])
def test_generate_is_shut_at_the_stages_ctrl_enter_ignores(stage):
    command = next(c for c in palette.commands(_palette_ctx("mesh")) if c.key == "generate")
    assert command.enabled(_palette_ctx(stage)) is False
    assert command.why, "a greyed row must say why"
    # and the two stages the chord does act on stay open
    assert command.enabled(_palette_ctx("reference")) is True
    assert command.enabled(_palette_ctx("mesh")) is True


# --- shell-18: the Empty-trash gate must not query the store per evaluation ---


class _ForbiddenStore:
    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"the frame thread reached the store: {name}")


def test_empty_trash_gate_does_not_query_the_store_on_the_frame_thread():
    cache = SimpleNamespace(jobs=[{"id": "a", "deleted_at": None}], trash_present=True)
    ctx = SimpleNamespace(cache=cache, svc=SimpleNamespace(store=_ForbiddenStore()))
    assert palette._any_trashed(ctx) is True
    cache.trash_present = False
    assert palette._any_trashed(ctx) is False


def test_the_job_cache_learns_whether_anything_is_trashed_on_the_read_task(svc):
    old = svc.store.create("text", "an old lantern", {})
    svc.store.set_deleted_if_not_running(old, 5.0)
    svc.store.create("text", "a fresh crate", {})

    cache = jobs_cache.JobsCache(svc)
    assert cache.trash_present is None, "unknown until a read has landed"
    assert cache.tick() is True
    assert cache.trash_present is True

    svc.store.delete(old)
    cache.invalidate()
    cache.tick()
    assert cache.trash_present is False


# --- shell-33: an overflowed toggle keeps its on-state ------------------------


def test_an_overflowed_selected_item_is_drawn_checked_in_the_menu(monkeypatch):
    from _ui_context import imgui_context

    from realmspinner.studio import probe

    item = toolbar.Item("random", "Random", icon="R", selected=True, priority=1)
    with imgui_context(monkeypatch) as ui:
        io = ui.get_io()
        io.add_mouse_pos_event(-100.0, -100.0)
        io.add_mouse_button_event(0, False)
        census: list = []
        for _ in range(2):
            probe.begin_frame()
            ui.new_frame()
            ui.set_next_window_size((10.0, 400.0))
            ui.set_next_window_pos((0.0, 0.0))
            ui.begin("##host")
            ui.open_popup("bar/menu")
            toolbar.toolbar("bar", [item])
            ui.end()
            ui.end_frame()
            census = list(probe.FRAME_CONTROLS)
    (row,) = [c for c in census if c.kind == "menu_item"]
    assert row.selected is True


# --- shell-19: widening a search reads no job rows on the frame thread --------


def _widen_fixture(svc):
    old_id = svc.store.create("text", "a rusty iron lantern", {})
    svc.store._conn.execute("UPDATE jobs SET created_at = 1.0 WHERE id = ?", (old_id,))
    svc.store._conn.commit()
    svc.store.create("text", "an unrelated crate", {})
    cache = jobs_cache.JobsCache(svc, limit=1)
    cache.tick()
    assert old_id not in cache.by_id
    return cache, old_id


def _run_widen(cache: Any, filters: Any) -> None:
    from realmspinner.studio.tasks import TaskRunner

    runner = TaskRunner(workers=1)
    try:
        assert cache.request_widen(filters, runner) is True
        deadline = time.monotonic() + 5
        done: list[Any] = []
        while time.monotonic() < deadline and not done:
            done = runner.poll()
        assert done and done[0].key == jobs_cache.SEARCH_KEY, done
        cache.adopt_widen(done[0].result)
    finally:
        runner.shutdown(wait=False)


def test_widening_a_search_reads_no_job_rows_on_the_frame_thread(svc, monkeypatch):
    cache, old_id = _widen_fixture(svc)
    real = jobs_cache.svc_jobs.get_job
    threads: list[threading.Thread] = []

    def spy(*args: Any, **kwargs: Any) -> Any:
        threads.append(threading.current_thread())
        return real(*args, **kwargs)

    monkeypatch.setattr(jobs_cache.svc_jobs, "get_job", spy)
    filters = Filters(text="lantern")
    _run_widen(cache, filters)

    assert old_id in [j["id"] for j in cache.visible(filters)]
    assert threads, "the matched row was never fetched at all"
    assert threading.current_thread() not in threads, "get_job ran on the frame thread"


def test_a_widened_row_survives_the_next_list_refresh(svc):
    cache, old_id = _widen_fixture(svc)
    filters = Filters(text="lantern")
    _run_widen(cache, filters)
    assert old_id in cache.by_id

    cache.invalidate()
    assert cache.tick() is True  # an ordinary refresh replaces the window
    assert old_id in [j["id"] for j in cache.visible(filters)], (
        "the matched old row dropped out until the next search landed"
    )

    # ...and clearing the filter lets the next refresh drop it again.
    cache.request_widen(Filters(), _Runner())
    cache.invalidate()
    cache.tick()
    assert old_id not in cache.by_id


# --- shell-20: a NaN in the stored layout reads as absent ---------------------


def _nan_settings() -> Any:
    from studio.test_layouts import _Settings

    nan = float("nan")
    return _Settings(
        {
            "layout": {"settings_share": nan, "settings_shares": {"clay-tools": nan}},
            "workspace_layouts": {
                "default": {
                    "v": 2,
                    "workspaces": {
                        "clay": {"shares": {"clay-tools": nan}, "columns": {}, "hidden": []}
                    },
                }
            },
        }
    )


def test_a_nan_share_in_the_settings_file_reads_as_the_default():
    settings = _nan_settings()
    legacy = layout_mod.Layout(settings)
    assert math.isfinite(legacy.settings_share)
    assert math.isfinite(legacy.share("clay-tools"))
    assert legacy.share("clay-tools") == layout_mod.SHARE_DEFAULTS.get(
        "clay-tools", legacy.settings_share
    )

    library = layouts.Library(settings)
    assert math.isfinite(library.share("clay", "clay-tools"))
    assert "clay-tools" not in library.arrangement("clay").shares
    bound = layout_mod.Layout(settings)
    bound.bind_workspace(library, "clay")
    assert math.isfinite(bound.share("clay-tools"))
    saved = bound.saved_share("clay-tools")
    assert saved is None or math.isfinite(saved)
