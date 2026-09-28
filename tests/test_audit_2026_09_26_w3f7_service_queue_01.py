"""service-queue-01 (2026-09-26 audit): a cancel during ``lowpoly`` didn't kill Blender.

``_q_mesh.py``'s in-job low-poly remesh (``params["lowpoly_triangles"]``, a
text/image job's own post-process, not a standalone mesh job) reports its
progress as ``phase="lowpoly"`` and runs Blender the same way ``remesh`` does
-- through ``blender_run.run_worker`` with ``on_start=self._note_blender``, so
``self._blender`` names the same live child. ``Worker.request_cancel`` killed
that child for ``rig``/``sheet``/``views``/``project``/``remesh``/``train``/
``separate`` but not for ``lowpoly``, so setting the cancel event during this
phase left the child running: the job sat through the whole Blender pass
before the cancel actually took effect.
"""

from __future__ import annotations

import pytest

from realmspinner.config import Config
from realmspinner.db import JobStore
from realmspinner.queue import Worker, _Cancel

pytestmark = pytest.mark.asyncio


def _make_worker(tmp_path) -> Worker:
    config = Config(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "models",
    )
    return Worker(config, JobStore(config.db_path))


class _FakeBlenderProc:
    """Stands in for the ``subprocess.Popen`` handle ``_note_blender`` records."""

    def __init__(self) -> None:
        self.kill_calls = 0

    def poll(self) -> int | None:
        return None if self.kill_calls == 0 else 0

    def kill(self) -> None:
        self.kill_calls += 1


async def test_request_cancel_kills_the_blender_child_during_the_lowpoly_phase(tmp_path):
    worker = _make_worker(tmp_path)
    job_id = "job-lowpoly"
    worker.current_job_id = job_id
    worker._cancel = _Cancel(job_id)
    worker.progress.begin(job_id, "text")
    worker.progress.update(job_id, phase="lowpoly", label="Remeshing", inner=0.3)
    proc = _FakeBlenderProc()
    worker._blender = proc

    await worker.request_cancel(job_id)

    assert worker._cancel.event.is_set()
    assert proc.kill_calls == 1, "the lowpoly phase's Blender child was never killed"

    worker.store.close()
