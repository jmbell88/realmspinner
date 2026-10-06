"""When the Library's meaning index gets written: the scheduler, not the embedder.

A fake runner records what the frame loop would have submitted and the tests run
those bodies by hand, so the order, the back-off and the stop are all observable;
the two tests that must prove "not on the frame thread" use a real
``TaskRunner``. The embedder itself is ``tests/_embed_fakes``.
"""

from __future__ import annotations

import threading

import _embed_fakes as fakes
import pytest

from realmspinner.service import library_index as li
from realmspinner.studio import embed_indexer as ei
from realmspinner.studio import jobs_cache as cache_mod
from realmspinner.studio.tasks import TaskRunner


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Runner:
    """The submit half of ``TaskRunner``, with the work left undone until a test
    asks for it."""

    def __init__(self) -> None:
        self.queue: list[tuple[str, object, tuple]] = []
        self.busy: set[str] = set()

    def submit(self, key, fn, *args, **kwargs):
        if key in self.busy:
            return False
        self.queue.append((key, fn, args))
        return True

    def run_all(self) -> list[tuple[str, object]]:
        out = []
        while self.queue:
            key, fn, args = self.queue.pop(0)
            out.append((key, fn(*args)))
        return out

    @property
    def keys(self) -> list[str]:
        return [k for k, _fn, _args in self.queue]


def _done(svc, prompt, name="", **kw):
    job_id = svc.store.create("text", prompt, kw.pop("params", {}), status="done", **kw)
    if name:
        svc.store.set_meta(job_id, name=name)
    return job_id


@pytest.fixture
def indexer(svc, monkeypatch):
    fake = fakes.install(monkeypatch, svc)
    clock = Clock()
    ix = ei.LibraryIndexer(svc, clock=clock, batch=3)
    return ix, fake, clock


# --- off without the row ---------------------------------------------------------


def test_without_the_retrieval_row_nothing_is_scheduled_written_or_started(svc, monkeypatch):
    fake = fakes.install(monkeypatch, svc)
    monkeypatch.setattr(li, "installed", lambda _svc: False)
    job_id = _done(svc, "a rusty longsword")
    ix = ei.LibraryIndexer(svc, clock=Clock(), batch=3)
    runner = Runner()
    ix.note(job_id)
    for _ in range(50):
        ix.pump(runner)
    assert runner.queue == []
    assert ix.pending == ()
    assert fake.calls == []
    assert svc.store.embedding_stamps() == {}


def test_the_installed_check_is_a_memo_not_a_stat_per_frame(svc, monkeypatch):
    fakes.install(monkeypatch, svc)
    asked = []
    monkeypatch.setattr(li, "installed", lambda s: asked.append(1) or False)
    clock = Clock()
    ix = ei.LibraryIndexer(svc, clock=clock)
    for _ in range(100):
        ix.pump(Runner())
    assert len(asked) == 1
    clock.now += ei.AVAILABLE_SECONDS + 1
    ix.pump(Runner())
    assert len(asked) == 2


def test_a_job_cache_without_the_row_behaves_as_it_always_did(svc, monkeypatch):
    """The whole integration with the row absent: finishing a job queues
    nothing, ``request`` submits only the list read, and no index row exists."""
    monkeypatch.setattr(li, "installed", lambda _svc: False)
    job_id = svc.store.create("text", "a rusty longsword", {})
    cache = cache_mod.JobsCache(svc)
    cache.tick()
    svc.store.set_status(job_id, "done")
    cache.invalidate()
    cache.tick()
    runner = Runner()
    cache.invalidate()
    cache.request(runner)
    assert runner.keys == ["jobs-list"]
    assert cache.indexer.pending == ()
    assert svc.store.embedding_stamps() == {}


# --- a finished job -------------------------------------------------------------------


def test_a_finished_job_is_indexed_by_exactly_one_task_off_the_frame_thread(svc, monkeypatch):
    fake = fakes.install(monkeypatch, svc)
    job_id = svc.store.create("text", "a rusty iron longsword", {})
    cache = cache_mod.JobsCache(svc)
    cache.tick()  # the job is seen queued ...
    svc.store.set_status(job_id, "done")
    cache.invalidate()
    cache.tick()  # ... and then seen done: the edge the indexer is told about
    assert cache.indexer.pending == (job_id,)
    # A backfill is also due on a first pump; this test is about the one-job
    # path, so quiet the backfill the way a running job does.
    cache.indexer.set_live(True)

    runner = TaskRunner(workers=2)
    try:
        submitted = []
        real_submit = runner.submit

        def counting(key, fn, *a, **k):
            if key.startswith(ei.KEY_PREFIX):
                submitted.append(key)
            return real_submit(key, fn, *a, **k)

        runner.submit = counting  # type: ignore[method-assign]
        cache.request(runner)
        # Pumped again on the next frames: the id was taken, so nothing more.
        cache.request(runner)
        cache.request(runner)
        done = fakes.wait_done(runner, lambda got: any(d.key == ei.ONE_KEY for d in got))
        assert [d.key for d in done if d.key == ei.ONE_KEY] == [ei.ONE_KEY]
    finally:
        runner.shutdown(wait=True)

    assert submitted == [ei.ONE_KEY]
    assert len(fake.document_calls) == 1
    assert fake.document_calls[0]["thread"] != threading.main_thread().name
    assert fake.document_calls[0]["thread"].startswith("realmspinner-task")
    assert set(svc.store.embedding_stamps()) == {job_id}


def test_a_job_that_was_already_done_when_first_seen_is_left_to_the_backfill(svc, monkeypatch):
    fakes.install(monkeypatch, svc)
    _done(svc, "an old job")
    cache = cache_mod.JobsCache(svc)
    cache.tick()
    assert cache.indexer.pending == ()


def test_a_renamed_finished_job_is_queued_for_re_embedding(svc, monkeypatch):
    fakes.install(monkeypatch, svc)
    job_id = _done(svc, "a rusty longsword", name="Sword")
    cache = cache_mod.JobsCache(svc)
    cache.tick()
    assert cache.indexer.pending == ()
    svc.store.set_meta(job_id, name="Excalibur")
    cache.invalidate()
    cache.tick()
    assert cache.indexer.pending == (job_id,)


def test_a_sweep_unit_finishing_is_never_queued(svc, monkeypatch):
    fakes.install(monkeypatch, svc)
    sweep_id = svc.store.create_sweep("s", "p", {})
    unit = svc.store.create("text", "unit", {}, sweep_id=sweep_id)
    cache = cache_mod.JobsCache(svc)
    cache.tick()
    svc.store.set_status(unit, "done")
    cache.invalidate()
    cache.tick()
    assert cache.indexer.pending == ()


def test_an_unavailable_embedder_puts_the_finished_job_back_and_backs_off(indexer, svc):
    ix, fake, clock = indexer
    job_id = _done(svc, "a rusty longsword")
    ix.note(job_id)
    fake.fail = True
    runner = Runner()
    ix.pump(runner)
    # The one-job task, and the first backfill batch beside it.
    assert ei.ONE_KEY in runner.keys
    results = dict(runner.run_all())
    assert results[ei.ONE_KEY]["retry"] is True
    assert ix.pending == (job_id,)  # not lost: tried again later
    assert svc.store.embedding_stamps() == {}
    assert ix.failures >= 1


# --- the backfill --------------------------------------------------------------------------


def _drive(ix, clock, runner, *, frames=200, step=1.0):
    """Frame loop: pump, run what was submitted, advance the clock."""
    for _ in range(frames):
        ix.pump(runner)
        runner.run_all()
        clock.now += step
        if ix.phase == ei.DONE:
            return


def test_the_backfill_indexes_the_history_in_bounded_batches_and_then_stops(indexer, svc):
    ix, fake, clock = indexer
    ids = [_done(svc, f"thing {i}") for i in range(8)]
    runner = Runner()
    _drive(ix, clock, runner)
    assert ix.phase == ei.DONE
    assert set(svc.store.embedding_stamps()) == set(ids)
    # 3 + 3 + 2 in the missing phase; the verify pass found nothing to embed.
    sizes = [len(c["texts"]) for c in fake.document_calls]
    assert sizes == [3, 3, 2]
    # Finished means finished: no further task, however many frames follow.
    calls = len(fake.calls)
    for _ in range(50):
        ix.pump(runner)
    assert runner.queue == [] and len(fake.calls) == calls


def test_the_backfill_catches_a_rename_made_while_the_app_was_closed(indexer, svc):
    ix, fake, clock = indexer
    job_id = _done(svc, "a rusty longsword", name="Sword")
    li.index_pending(svc, 10)  # indexed in an earlier session
    svc.store.set_meta(job_id, name="Excalibur")  # renamed with the app closed
    fake.calls.clear()
    _drive(ix, clock, Runner())
    assert ix.phase == ei.DONE
    assert len(fake.document_calls) == 1
    assert fake.document_calls[0]["texts"][0][0] == "Excalibur"


def test_one_batch_is_submitted_at_a_time_with_a_gap_between(indexer, svc):
    ix, _fake, clock = indexer
    for i in range(10):
        _done(svc, f"thing {i}")
    runner = Runner()
    ix.pump(runner)
    assert runner.keys == [ei.BACKFILL_KEY]
    runner.busy.add(ei.BACKFILL_KEY)  # a slow batch is still running
    ix.pump(runner)
    assert runner.keys == [ei.BACKFILL_KEY]  # not queued behind itself
    runner.busy.clear()
    runner.run_all()
    ix.pump(runner)
    assert runner.queue == []  # the gap has not passed
    clock.now += ei.BATCH_GAP_SECONDS + 0.1
    ix.pump(runner)
    assert runner.keys == [ei.BACKFILL_KEY]


def test_the_backfill_waits_while_a_job_is_running_but_a_finished_job_still_indexes(
    indexer, svc
):
    ix, _fake, _clock = indexer
    old = _done(svc, "history")
    fresh = _done(svc, "just finished")
    ix.set_live(True)
    ix.note(fresh)
    runner = Runner()
    ix.pump(runner)
    assert runner.keys == [ei.ONE_KEY]
    runner.run_all()
    assert set(svc.store.embedding_stamps()) == {fresh}
    ix.set_live(False)
    ix.pump(runner)
    assert runner.keys == [ei.BACKFILL_KEY]
    runner.run_all()
    assert old in svc.store.embedding_stamps()


def test_a_failing_backfill_backs_off_exponentially_and_never_loops_tightly(indexer, svc):
    ix, fake, clock = indexer
    for i in range(5):
        _done(svc, f"thing {i}")
    fake.fail = True
    runner = Runner()
    # A hundred frames, a second apart: the embedder is asked a handful of
    # times, not a hundred.
    for _ in range(100):
        ix.pump(runner)
        runner.run_all()
        clock.now += 1.0
    attempts = len(fake.document_calls)
    assert 1 <= attempts <= 4, attempts
    assert ix.failures == attempts
    # The delays double: 30, 60, 120 ...
    before = ix.failures
    clock.now += ei.BACKOFF_BASE * 2 ** before + 1
    ix.pump(runner)
    assert runner.keys == [ei.BACKFILL_KEY]


def test_a_success_after_failures_resets_the_back_off(indexer, svc):
    ix, fake, clock = indexer
    _done(svc, "a thing")
    fake.fail = True
    runner = Runner()
    ix.pump(runner)
    runner.run_all()
    assert ix.failures == 1
    fake.fail = False
    clock.now += ei.BACKOFF_BASE + 1
    ix.pump(runner)
    runner.run_all()
    assert ix.failures == 0
    assert len(svc.store.embedding_stamps()) == 1


def test_the_back_off_is_capped(indexer):
    ix, _fake, _clock = indexer
    for _ in range(30):
        delay = ix._backoff()
    assert delay == ei.BACKOFF_MAX


def test_nothing_is_submitted_after_shutdown_and_a_running_batch_embeds_nothing(indexer, svc):
    ix, fake, clock = indexer
    for i in range(5):
        _done(svc, f"thing {i}")
    runner = Runner()
    ix.pump(runner)
    assert runner.keys == [ei.BACKFILL_KEY]
    ix.stop()
    # The batch was already submitted: it must notice and do nothing.
    results = runner.run_all()
    assert results == [(ei.BACKFILL_KEY, {"indexed": 0})]
    assert fake.calls == []
    clock.now += 100
    ix.pump(runner)
    ix.note("anything")
    assert runner.queue == [] and ix.pending == ()


def test_a_batch_that_finds_the_workers_loop_gone_embeds_nothing(svc, monkeypatch):
    """``Runtime.shutdown`` ends the loop before anything else; a batch that
    wakes up after it has nothing to embed with and must not try."""
    fake = fakes.install(monkeypatch, svc)
    _done(svc, "a thing")

    class DeadLoop:
        def is_running(self):
            return False

    svc.loop = DeadLoop()
    assert li.index_pending(svc, 10) == li.IndexPass(0, False)
    assert li.index_jobs(svc, ["x"]) == 0
    assert fake.calls == []


def test_installing_the_row_after_a_finished_backfill_starts_it_again(svc, monkeypatch):
    fake = fakes.install(monkeypatch, svc)
    state = {"on": False}
    monkeypatch.setattr(li, "installed", lambda _s: state["on"])
    clock = Clock()
    ix = ei.LibraryIndexer(svc, clock=clock, batch=3)
    ix.pump(Runner())
    ix._phase = ei.DONE  # as if a previous run had finished
    state["on"] = True
    clock.now += ei.AVAILABLE_SECONDS + 1
    _done(svc, "a thing")
    runner = Runner()
    ix.pump(runner)
    assert ix.phase == ei.MISSING
    assert runner.keys == [ei.BACKFILL_KEY]
    runner.run_all()
    assert len(fake.document_calls) == 1


def test_the_backfill_body_runs_on_the_task_pool_not_the_frame_thread(svc, monkeypatch):
    fake = fakes.install(monkeypatch, svc)
    for i in range(4):
        _done(svc, f"thing {i}")
    ix = ei.LibraryIndexer(svc, batch=10)
    runner = TaskRunner(workers=1)
    try:
        ix.pump(runner)
        done = fakes.wait_done(runner)
    finally:
        runner.shutdown(wait=True)
    assert done and done[0].key == ei.BACKFILL_KEY and done[0].ok
    assert fake.document_calls[0]["thread"].startswith("realmspinner-task")


def test_a_task_body_never_raises_so_a_failure_is_never_a_toast(indexer, svc, monkeypatch):
    ix, _fake, _clock = indexer
    _done(svc, "a thing")

    def boom(*_a, **_k):
        raise RuntimeError("the store exploded")

    monkeypatch.setattr(li, "index_pending", boom)
    monkeypatch.setattr(li, "index_jobs", boom)
    assert ix._run_backfill()["error"] is True
    assert ix._run_ids(("x",))["error"] is True
    assert ix.failures == 2
