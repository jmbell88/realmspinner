"""The 2026-10-14 audit, fixer F55: the shell's job cache, journal, layout editor,
boot path and teardown.

Each test name is the claim, and each was run against the unfixed modules first
(HEAD's source loaded as throwaway modules) and failed there.
"""

from __future__ import annotations

import inspect
import logging
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from realmspinner.service import jobs as svc_jobs
from realmspinner.studio import jobs_cache as cache_mod
from realmspinner.studio import journal, layout_edit, layouts, mode_manifest, skeletons
from realmspinner.studio.state import Filters
from realmspinner.studio.tasks import TaskRunner

# --- shell-documents-03: no COUNT(*) on the frame thread ----------------------


def test_the_count_is_taken_on_the_task_thread_and_published_by_adopt(svc, monkeypatch):
    """The count rides the reading: ``read`` (a task thread) takes it, ``adopt``
    (the frame thread) only publishes it."""
    for i in range(3):
        svc_jobs.create_job(svc, kind="text", prompt=f"job {i}")
    cache = cache_mod.JobsCache(svc, limit=2)  # a full page: the count is due
    frame = threading.get_ident()
    seen: list[int] = []
    real = svc.store.count

    def spy() -> int:
        seen.append(threading.get_ident())
        return real()

    monkeypatch.setattr(svc.store, "count", spy)
    cache._read_was_dirty = True
    reading: dict[str, Any] = {}
    worker = threading.Thread(target=lambda: reading.update(cache.read(dict(cache._files))))
    worker.start()
    worker.join(10.0)
    assert cache.adopt(reading) is True
    assert seen, "a full page re-counts"
    assert frame not in seen, "the COUNT(*) ran on the frame thread"
    assert cache.total == 3
    assert cache.count_error is None


def test_adopt_never_queries_the_store_on_the_frame_thread(svc, monkeypatch):
    """The record's name: handing ``adopt`` a reading never reaches the store."""
    for i in range(3):
        svc_jobs.create_job(svc, kind="text", prompt=f"job {i}")
    cache = cache_mod.JobsCache(svc, limit=2)
    cache._read_was_dirty = True
    reading = cache.read(dict(cache._files))

    def boom() -> int:
        raise AssertionError("adopt queried the store")

    monkeypatch.setattr(svc.store, "count", boom)
    assert cache.adopt(reading) is True
    assert cache.total == 3


# --- shell-home-library-04: Show examples reaches past the window ---------------


def _old_and_new(svc):
    old_id = svc.store.create("text", "a rusty iron lantern", {})
    svc.store._conn.execute("UPDATE jobs SET created_at = 1.0 WHERE id = ?", (old_id,))
    svc.store._conn.commit()
    svc.store.create("text", "an unrelated crate", {})
    return old_id


def _drain(runner: TaskRunner) -> Any:
    deadline = time.monotonic() + 10
    done: list[Any] = []
    while time.monotonic() < deadline and not done:
        done = runner.poll()
        time.sleep(0.01)
    assert done and done[0].key == cache_mod.SEARCH_KEY, done
    return done[0]


def test_show_examples_reaches_jobs_outside_the_loaded_window(svc):
    old_id = _old_and_new(svc)
    cache = cache_mod.JobsCache(svc, limit=1)
    cache.tick()
    assert old_id not in cache.by_id  # the one-row window missed it

    runner = TaskRunner(workers=1)
    try:
        filters = Filters(job_ids=frozenset({old_id}))
        assert cache.request_widen(filters, runner) is True
        cache.adopt_widen(_drain(runner).result)
        assert [j["id"] for j in cache.visible(filters)] == [old_id]
        # Asked once, not every frame draw() calls it on.
        assert cache.request_widen(filters, runner) is False
    finally:
        runner.shutdown(wait=False)


def test_show_examples_inside_the_window_asks_the_store_for_nothing(svc):
    new_id = svc.store.create("text", "inside", {})
    cache = cache_mod.JobsCache(svc)
    cache.tick()
    runner = TaskRunner(workers=1)
    try:
        assert cache.request_widen(Filters(job_ids=frozenset({new_id})), runner) is False
    finally:
        runner.shutdown(wait=False)


# --- shell-documents-09: a search for a cleared box adds no rows ------------------


def test_a_search_landing_after_the_filter_was_cleared_adds_no_rows(svc):
    old_id = _old_and_new(svc)
    cache = cache_mod.JobsCache(svc, limit=1)
    cache.tick()
    runner = TaskRunner(workers=1)
    try:
        assert cache.request_widen(Filters(text="lantern"), runner) is True
        landed = _drain(runner)
        # The box is cleared while the answer is still on its way to the frame.
        assert cache.request_widen(Filters(), runner) is False
        cache.adopt_widen(landed.result)
        assert old_id not in cache.by_id
        assert old_id not in [j["id"] for j in cache.jobs]
    finally:
        runner.shutdown(wait=False)


def test_a_search_for_the_box_as_it_stands_still_lands(svc):
    old_id = _old_and_new(svc)
    cache = cache_mod.JobsCache(svc, limit=1)
    cache.tick()
    runner = TaskRunner(workers=1)
    try:
        filters = Filters(text="lantern")
        assert cache.request_widen(filters, runner) is True
        cache.adopt_widen(_drain(runner).result)
        assert old_id in cache.by_id
    finally:
        runner.shutdown(wait=False)


# --- the journal ------------------------------------------------------------------


class _Ctx:
    def __init__(self, root: Path) -> None:
        self.svc = SimpleNamespace(config=SimpleNamespace(autosave_dir=root))
        self.state = SimpleNamespace(recovery=None)
        self.queued: list[Any] = []

    def submit(self, key: str, run: Any, *a: Any, **k: Any) -> bool:
        self.queued.append(run)
        return True

    def toast(self, *a: Any, **k: Any) -> None:
        pass


def _copy(root: Path, name: str, kind: str) -> Path:
    payload = root / name
    journal._write_pair(
        payload,
        b"{}",
        {"version": journal.VERSION, "kind": kind, "title": name, "uid": name, "at": 5.0},
    )
    return payload


def test_a_crash_copy_of_a_retired_kind_is_not_offered_and_does_not_block_the_list(tmp_path):
    retired = _copy(tmp_path, "left-by-poser.pose.json", "pose")
    future = _copy(tmp_path, "from-a-newer-build.zzz", "a-kind-only-a-newer-build-owns")
    ctx = _Ctx(tmp_path)

    kinds = [one.kind for one in journal.recoverable(ctx)]
    assert "pose" not in kinds
    # An unknown kind is still kept, for a build that may own it.
    assert "a-kind-only-a-newer-build-owns" in kinds

    found = journal.snapshot(ctx)
    assert [one.kind for one in found] == ["a-kind-only-a-newer-build-owns"]
    assert not retired.exists() and not journal.meta_path(retired).exists()
    assert future.exists() and journal.meta_path(future).exists()


def test_the_status_line_does_not_promise_a_retired_kind_back(tmp_path):
    _copy(tmp_path, "left-by-poser.pose.json", "pose")
    assert "No unsaved work" in journal.status_line(_Ctx(tmp_path))


class _Slot:
    def __init__(self) -> None:
        self.uid = "s1"
        self.title = "atlas"
        self.journal_name = ""
        self.journal_head = None
        self.journal_at = 0.0


def test_drop_does_not_wait_for_a_deferred_encode_in_flight(tmp_path):
    entered, gate = threading.Event(), threading.Event()

    def encode(slot: _Slot) -> Any:
        def later() -> bytes:
            entered.set()
            assert gate.wait(30.0)
            return b"png"

        return later

    provider = journal.Provider(
        kind="f55probe",
        ext=".f55probe",
        label="probe",
        slots=lambda ctx: [],
        uid_of=lambda s: s.uid,
        title_of=lambda s: s.title,
        head_of=lambda s: 1,
        encode=encode,
        adopt=lambda ctx, path, meta: True,
    )
    ctx = _Ctx(tmp_path)
    slot = _Slot()
    assert journal.write(ctx, provider, slot) is True
    writer = threading.Thread(target=ctx.queued[0], daemon=True)
    writer.start()
    try:
        assert entered.wait(10.0), "the deferred encode is running on the task"
        dropped = threading.Event()
        dropper = threading.Thread(
            target=lambda: (journal.drop(ctx, slot), dropped.set()), daemon=True
        )
        dropper.start()
        assert dropped.wait(3.0), "drop (frame thread) waited for the encode"
    finally:
        gate.set()
        writer.join(10.0)
    # And the write that finished after the drop did not resurrect the pair.
    assert list(tmp_path.glob("*.f55probe*")) == []


# --- shell-documents-04 / -12: the layout editor ----------------------------------


def test_the_layout_editor_closes_when_the_mode_has_no_skeleton(monkeypatch):
    monkeypatch.setitem(skeletons.BUILDERS, "inker", lambda ctx: {"left": object()})
    state = SimpleNamespace(mode="inker")
    ctx = SimpleNamespace(state=state)
    edit = layout_edit.ensure(state)
    edit.open = True
    edit.dragging = "something"

    layout_edit.settle(ctx)
    assert edit.open is True, "a mode with a skeleton keeps its editor"

    state.mode = "create"  # no builder: nothing to rearrange
    layout_edit.settle(ctx)
    assert edit.open is False
    assert edit.dragging == ""


def test_the_frame_settles_the_editor_before_the_splitters_read_it():
    from realmspinner.studio.shell import frame

    source = inspect.getsource(frame)
    assert source.index("layout_edit.settle(ctx)") < source.index(
        "layout_mod.begin_frame(layout_edit.ensure(ctx.state).open)"
    )


class _Settings:
    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value


def test_the_layout_editor_says_so_when_the_active_layout_is_a_newer_version(monkeypatch):
    monkeypatch.setitem(skeletons.BUILDERS, "inker", lambda ctx: {"left": object()})
    ctx = SimpleNamespace(state=SimpleNamespace(mode="inker"))

    ok = layouts.Library(_Settings({}))
    assert layout_edit.refusal(SimpleNamespace(layouts=ok), ctx) == ""

    future = layouts.Library(
        _Settings(
            {
                layouts.LAYOUTS_KEY: {"Future": {"v": 99, "workspaces": {}}},
                layouts.ACTIVE_KEY: "Future",
            }
        )
    )
    assert not future.current().readable
    assert "newer version" in layout_edit.refusal(SimpleNamespace(layouts=future), ctx)
    # A workspace with no skeleton keeps its own sentence.
    ctx.state.mode = "create"
    assert layout_edit.refusal(SimpleNamespace(layouts=future), ctx) == layout_edit.NO_SKELETON


# --- shell-boot-04 / -05 ----------------------------------------------------------


def test_a_bad_log_level_warning_is_written_to_the_log_file(monkeypatch, tmp_path):
    import faulthandler

    import realmspinner.config as config_mod
    from realmspinner.studio import main

    root = logging.getLogger()
    before_handlers = list(root.handlers)
    before_level = root.level
    monkeypatch.setenv("REALMSPINNER_LOG_LEVEL", "nonsense")
    monkeypatch.setattr(config_mod, "get_config", lambda: SimpleNamespace(data_dir=tmp_path))
    monkeypatch.setattr(faulthandler, "enable", lambda *a, **k: None)
    monkeypatch.setattr(main, "_crash_log", None)
    try:
        main._setup_logging()
        for handler in root.handlers:
            handler.flush()
        text = (tmp_path / "realmspinner.log").read_text(encoding="utf-8")
        assert "is not a log level name" in text
        assert root.level == logging.INFO
    finally:
        crash = main._crash_log
        for handler in list(root.handlers):
            if handler not in before_handlers:
                root.removeHandler(handler)
                handler.close()
        root.handlers[:] = before_handlers
        root.setLevel(before_level)
        if crash is not None:
            crash.close()


@pytest.mark.parametrize(
    "marker",
    ["[" * 200_000, '{"pid": ' + "9" * 40 + "}"],
    ids=["nested", "impossible-pid"],
)
def test_a_pathologically_nested_session_marker_does_not_stop_startup(
    monkeypatch, tmp_path, marker
):
    from realmspinner.studio import main

    path = tmp_path / "session.marker"
    path.write_text(marker, encoding="utf-8")
    monkeypatch.setattr(main, "_marker_path", lambda: path)
    main._note_previous_session()  # must not raise


# --- shell-documents-10 -----------------------------------------------------------


def test_persisting_modes_skips_a_module_that_will_not_import(monkeypatch):
    real = mode_manifest._import
    broken = mode_manifest.DOC_MODES[0]

    def fake(name: str) -> Any:
        if name == broken.module:
            raise ImportError("this host cannot import it")
        return real(name)

    monkeypatch.setattr(mode_manifest, "_import", fake)
    kept = mode_manifest.persisting_modes()
    assert broken not in kept
    assert kept, "the other modes still persist"
