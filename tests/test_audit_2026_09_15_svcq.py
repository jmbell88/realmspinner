"""Regression tests for the 2026-09-15 audit's service/queue findings.

Four rows:

- service-03: ``packs._run_worker``'s generic ``except BaseException`` force-
  killed the child regardless of which phase it had announced, so a callback
  error raised mid-commit (not a timeout -- ``TimeoutExpired`` alone was
  taught this in service-04 of the 2026-09-07 audit) still risked the
  half-written site-packages INVARIANTS.md forbids.
- service-04: ``doctor._t2i_checks`` iterated the live, mutable
  ``models.STYLE_LORAS`` instead of ``models.style_loras_snapshot()``.
- service-05: ``Worker._process`` ran its setup (``current_job_id``,
  ``_cancel``, ``progress.begin``) between ``claim()`` and its own
  ``try:``, so an exception raised during that setup left the row
  ``running`` and ``current_job_id`` stale -- the ``finally`` that clears
  both never ran because the job had not yet entered the try/finally.
- service-06: ``Worker._run``'s ``commit_refused`` flag survived an
  unrelated exception raised while dispatching the *next* job, so that
  job's unrelated failure bought the job after it a stale commit-refusal
  backoff meant only for the dispatch right after a real ``CommitRefused``.
"""

from __future__ import annotations

import asyncio

import pytest

from warlock import doctor as doctor_mod
from warlock import models as model_registry
from warlock.config import Config
from warlock.service import packs as svc_packs

# Not a module-level pytestmark: service-04's test below is plain sync (no
# child process, no event loop needed), and pytest-asyncio warns on a sync
# test carrying the asyncio mark. Each async test is marked individually.


# ---------------------------------------------------------------------------
# service-03
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_unexpected_error_during_the_commit_phase_does_not_force_kill_the_pack_install(
    monkeypatch, tmp_path
):
    """A callback (``on_progress``) that raises mid-commit used to hit the
    generic ``except BaseException`` and force-kill pip regardless of phase,
    risking a half-written install into the running app's own site-packages.
    Regression for the 2026-09-15 audit, finding service-03."""
    import json
    import sys
    import textwrap
    import time

    from warlock import packs

    manifest = tmp_path / packs.MANIFEST_NAME
    manifest.write_text(
        json.dumps(
            {
                "version": packs.MANIFEST_VERSION,
                "wheels": [
                    {
                        "filename": "bpy-5.2.0-cp313-cp313-win_amd64.whl",
                        "url": "https://example.invalid/bpy-5.2.0-cp313-cp313-win_amd64.whl",
                        "size_bytes": 10,
                        "sha256": "a" * 64,
                        "installed_bytes": 0,
                        "packs": ["rig"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(svc_packs, "manifest_path", lambda: manifest)
    monkeypatch.setattr(svc_packs, "installed_versions", dict)

    stub = tmp_path / "stub_worker.py"
    stub.write_text(
        textwrap.dedent(
            """
            import json, sys, time
            spec = json.loads(sys.stdin.read())
            payload = {"percent": 92.0, "label": "installing", "phase": "commit"}
            print(json.dumps(payload), flush=True)
            time.sleep(1.0)
            open(spec["result_path"], "w").write(json.dumps({"ok": True, "installed": ["bpy"]}))
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        svc_packs, "worker_argv", lambda: [sys.executable, str(stub)]
    )

    kill_calls: list[int] = []
    real_kill = svc_packs._kill_and_reap

    def spy_kill(proc):
        kill_calls.append(proc.pid)
        real_kill(proc)

    monkeypatch.setattr(svc_packs, "_kill_and_reap", spy_kill)

    def on_progress(percent, label, phase):
        if phase == "commit":
            raise RuntimeError("boom from a callback mid-commit")

    class FakeService:
        def __init__(self, home):
            self.config = type("C", (), {"home": home})()

    svc = FakeService(tmp_path)

    start = time.monotonic()
    with pytest.raises(RuntimeError, match="boom from a callback mid-commit"):
        svc_packs.install(svc, ["rig"], timeout=5.0, on_progress=on_progress)
    elapsed = time.monotonic() - start

    # Proof the child was not force-killed: it was allowed to run out its
    # sleep and write its result before the original error was re-raised.
    assert elapsed >= 0.9
    assert kill_calls == []


# ---------------------------------------------------------------------------
# service-04
# ---------------------------------------------------------------------------


def test_doctor_snapshots_style_loras_before_iterating_them_instead_of_the_live_mutable_table(
    monkeypatch, tmp_path
):
    """service-04: ``models.STYLE_LORAS`` is mutated from another thread as
    imports register style LoRAs, so iterating it live (rather than through
    ``style_loras_snapshot()``, which exists for exactly this) can raise
    ``RuntimeError: dictionary changed size during iteration`` inside doctor.
    Regression for the 2026-09-15 audit, finding service-04."""
    calls = {"n": 0}
    real_snapshot = model_registry.style_loras_snapshot

    def spy_snapshot():
        calls["n"] += 1
        return real_snapshot()

    monkeypatch.setattr(model_registry, "style_loras_snapshot", spy_snapshot)
    # And, symmetrically, prove doctor no longer needs to touch the live table
    # at all for this loop -- if it still iterated STYLE_LORAS directly this
    # would not fail, so the snapshot call count is the real assertion above.
    config = Config(t2i_model_root=tmp_path)

    doctor_mod._t2i_checks(config)

    assert calls["n"] >= 1, (
        "doctor._t2i_checks did not call models.style_loras_snapshot() -- it "
        "is iterating the live, mutable models.STYLE_LORAS again"
    )


# ---------------------------------------------------------------------------
# service-05 / service-06 (queue.py)
# ---------------------------------------------------------------------------


async def _wait_until(predicate, timeout: float = 10.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    pytest.fail("condition not met before timeout")


def _make_worker(tmp_path, **config_overrides):
    from warlock.db import JobStore
    from warlock.queue import Worker

    config = Config(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "models",
        **config_overrides,
    )
    store = JobStore(config.db_path)
    return Worker(config, store)


def _make_image_job(worker) -> str:
    job_id = worker.store.create("image", None, {"seed": 1, "resolution": 512})
    job_dir = worker.config.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input.png").write_bytes(b"fake-png")
    return job_id


@pytest.mark.asyncio
async def test_process_resets_current_job_id_when_setup_raises_after_claim(
    tmp_path, fake_pipelines, monkeypatch
):
    """service-05: ``current_job_id``/``_cancel`` and ``progress.begin`` used
    to run between ``claim()`` and ``_process``'s own ``try:``, so an
    exception raised during that setup (here, ``progress.begin``) left the
    row ``running`` and ``current_job_id`` stale until the next launch -- the
    ``finally`` that clears both never ran because the job had not yet
    entered the try/finally. Regression for the 2026-09-15 audit, finding
    service-05."""
    worker = _make_worker(tmp_path)
    try:
        real_begin = worker.progress.begin
        calls = {"n": 0}

        def begin_once_badly(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom during setup")
            return real_begin(*args, **kwargs)

        monkeypatch.setattr(worker.progress, "begin", begin_once_badly)

        doomed = _make_image_job(worker)
        worker.start()
        await _wait_until(lambda: calls["n"] >= 1)

        # Before the fix this never becomes true: the row is stuck 'running'
        # forever because _process's own finally (which would write 'error')
        # never ran -- the exception happened before _process entered its
        # try/finally at all.
        await _wait_until(
            lambda: worker.store.get(doomed)["status"] in ("done", "error"),
            timeout=5.0,
        )
        assert worker.store.get(doomed)["status"] == "error"
        assert worker.current_job_id != doomed
        assert worker._cancel is None

        # And the worker must not be wedged: a later job still completes.
        good = _make_image_job(worker)
        await _wait_until(lambda: worker.store.get(good)["status"] == "done")
        await worker.shutdown()
    finally:
        worker.store.close()


@pytest.mark.asyncio
async def test_worker_loop_does_not_apply_stale_commit_backoff_after_an_unrelated_dispatch_failure(
    tmp_path, fake_pipelines, monkeypatch
):
    """service-06: ``commit_refused`` survived an unrelated exception raised
    while dispatching the job right after a real ``CommitRefused``, so the
    *next* job again paid ``COMMIT_REFUSAL_BACKOFF`` even though nothing
    about it had anything to do with commit headroom. Regression for the
    2026-09-15 audit, finding service-06."""
    import warlock.queue as queue_mod

    backoff = 0.4
    monkeypatch.setattr(queue_mod, "COMMIT_REFUSAL_BACKOFF", backoff)
    worker = _make_worker(tmp_path)
    try:
        calls = {"n": 0}
        wait_calls: list[float] = []
        real_wait_for_work = worker._wait_for_work

        async def spy_wait_for_work(timeout=queue_mod.POLL_INTERVAL):
            wait_calls.append(timeout)
            return await real_wait_for_work(timeout)

        async def fake_process(job):
            calls["n"] += 1
            n = calls["n"]
            # _process is faked out entirely here -- the loop mechanics in
            # _run are what is under test, not a real job's pipeline -- but
            # claiming still has to happen so next_queued() advances past
            # this row instead of handing it back forever.
            await asyncio.to_thread(worker.store.claim, job["id"])
            if n == 1:
                # A real CommitRefused: the correctly-behaving case.
                return True
            if n == 2:
                # An unrelated bug in dispatch -- nothing to do with commit
                # headroom -- raised while commit_refused is still True from
                # job 1.
                raise RuntimeError("unrelated dispatch bug")
            return False

        monkeypatch.setattr(worker, "_wait_for_work", spy_wait_for_work)
        monkeypatch.setattr(worker, "_process", fake_process)

        # a, b, c: three queued rows for _run to dispatch in order. _process is
        # faked, so their ids are never referenced again -- only their count
        # and order (job_a refused, job_b unrelated bug, job_c clean) matter.
        worker.store.create("text", "a", {"seed": 1, "resolution": 512})
        worker.store.create("text", "b", {"seed": 2, "resolution": 512})
        worker.store.create("text", "c", {"seed": 3, "resolution": 512})

        worker.start()
        await _wait_until(lambda: calls["n"] >= 3, timeout=10.0)
        await worker.shutdown()

        # job_b, right after job_a's real CommitRefused, correctly backs off.
        # job_c, right after job_b's unrelated RuntimeError, must not.
        assert wait_calls.count(backoff) == 1, (
            f"expected exactly one backoff wait (for job_b only), got "
            f"{wait_calls.count(backoff)} in {wait_calls!r}"
        )
    finally:
        worker.store.close()
