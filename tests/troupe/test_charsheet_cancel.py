"""The reframe retry's cancel window.

The 2026-09-11 audit, finding troupe-03: ``_charsheet``'s reframe retry -- the
second, wider-margin Blender render fired when the first render comes back
clipped -- ran with no cancel check between the end of the first render and
the start of the retry. The only cancel check in the whole function sat after
*both* renders, right before the pixel-art pass, so a cancel requested during
or right after the first render was not honoured until a second full Blender
render had also completed -- "a minute a go" per ``_q_troupe.py``'s own
docstring.

This file is self-contained (a local ``worker`` fixture, its own render fake)
rather than importing ``test_troupe_chain.py``'s private helpers: that module
is not owned by this fix and its internals are free to change.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from warlock.config import Config
from warlock.db import JobStore
from warlock.kernels.rig import store as rig_store
from warlock.pipelines import blender_run
from warlock.queue import Worker

pytestmark = pytest.mark.asyncio


@pytest.fixture
def worker(tmp_path, fake_pipelines):
    config = Config(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "models",
    )
    store = JobStore(config.db_path)
    w = Worker(config, store)
    yield w
    store.close()


async def _wait_until(predicate, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    pytest.fail("condition not met before timeout")


def _always_clipped_render(monkeypatch, worker, *, cancel_after_first: bool):
    """A Blender fake that always renders edge-to-edge (measures as clipped),
    so ``_charsheet`` always wants the reframe retry -- and, when
    ``cancel_after_first`` is set, sets the worker's own cancel event the
    instant the first render finishes, the exact race the finding names."""
    from PIL import Image

    calls: list[dict] = []

    def fake(spec, **kwargs):
        calls.append({"spec": spec, **kwargs})
        frames_dir = Path(spec["frames_dir"])
        for cell in spec["cells"]:
            frame = Image.new("RGBA", (spec["frame_size"],) * 2, (0, 0, 0, 0))
            # Edge to edge, unconditionally: this fake never stops looking
            # clipped, so the only thing that can stop the retry from firing
            # is the cancel check under test.
            frame.paste(
                (128, 64, 200, 255), (0, 0, spec["frame_size"], spec["frame_size"])
            )
            frame.save(frames_dir / f"{cell['index']:04d}.png")
        if cancel_after_first and len(calls) == 1:
            worker._cancel.event.set()
        return {
            "ok": True,
            "pivot": [0.5, 0.9],
            "framing": {"extent": 2.24, "margin": spec.get("margin") or 1.12},
        }

    monkeypatch.setattr(blender_run, "run_worker", fake)
    return calls


def _rigged_source(worker):
    source = worker.store.create("image", "a ranger", {}, stage="model")
    source_dir = worker.config.job_dir(source)
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "model.glb").write_bytes(b"fake-glb")
    (source_dir / "rig.glb").write_bytes(b"fake-rig")
    (source_dir / "rig.json").write_text(json.dumps({"template": "humanoid"}), "utf-8")
    worker.store.set_status(source, "done")
    return source, source_dir


_TINY_LAYOUT = {
    "version": 2,
    "movements": [{"key": "idle", "frames": 3, "directions": 1}],
}


def _queue_charsheet(worker, source):
    return worker.store.create(
        "charsheet",
        "a ranger",
        {
            "source_job": source,
            "sheet_id": rig_store.new_id(),
            "logical_size": 16,
            "colors": 8,
            "layout": _TINY_LAYOUT,
        },
    )


async def test_a_cancel_between_the_first_render_and_the_reframe_retry_skips_the_second_render(
    worker, monkeypatch
):
    calls = _always_clipped_render(monkeypatch, worker, cancel_after_first=True)
    source, _source_dir = _rigged_source(worker)
    job_id = _queue_charsheet(worker, source)

    worker.start()
    try:
        await _wait_until(
            lambda: worker.store.get(job_id)["status"] in ("done", "error", "cancelled"),
            60.0,
        )
    finally:
        await worker.shutdown()

    # The claim under test: cancel means stopped, not "stopped after one more
    # full render". A fix that merely read the flag somewhere without acting
    # on it would still leave this at 2.
    assert len(calls) == 1, (
        f"the reframe retry fired ({len(calls)} Blender renders) after a cancel "
        "landed between it and the first render"
    )
    assert worker.store.get(job_id)["status"] == "cancelled"


async def test_without_a_cancel_the_reframe_retry_still_fires_as_before(worker, monkeypatch):
    """The control: an always-clipped render with no cancel still takes its
    one retry, so the fix above is a cancel check and not a second render that
    silently stopped firing altogether."""
    calls = _always_clipped_render(monkeypatch, worker, cancel_after_first=False)
    source, _source_dir = _rigged_source(worker)
    job_id = _queue_charsheet(worker, source)

    worker.start()
    try:
        await _wait_until(
            lambda: worker.store.get(job_id)["status"] in ("done", "error"), 60.0
        )
    finally:
        await worker.shutdown()

    assert len(calls) == 2
    assert worker.store.get(job_id)["status"] == "done"
