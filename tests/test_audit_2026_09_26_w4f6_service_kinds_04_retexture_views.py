"""``Worker._discard_artifacts``: a cancelled re-texture's own ``views/``.

The 2026-09-26 audit, finding service-kinds-04: the ``shutil.rmtree`` for a
job's own ``views/`` directory sat in the ``remesh`` arm of
``_discard_artifacts`` (``_q_jobs.py``), which never writes one at all --
``_q_mesh._remesh`` has no ``views_dir`` anywhere in it. ``_retexture``
(``_q_sprite.py``) is the one that renders ``op_views`` into its *own*
job_dir (the retexture job's id, not the source mesh's), so a cancelled
re-texture left up to ten renders and their bakes on disk forever. Moved to
the ``retexture`` arm.
"""

from __future__ import annotations

import pytest

from realmspinner.config import Config
from realmspinner.db import JobStore
from realmspinner.kernels.rig import store as rig_store
from realmspinner.queue import Worker


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


def _mesh_job(worker: Worker) -> str:
    job_id = worker.store.create("text", "a crate", {"seed": 1, "size_m": 1.0})
    job_dir = worker.config.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"old-model")
    worker.store.set_status(job_id, "done")
    return job_id


def test_a_cancelled_retexture_removes_its_own_views_directory(worker):
    source = _mesh_job(worker)
    retexture_id = "retextureA"
    # ``views/`` sits under the *retexture* job's own directory
    # (``self.config.job_dir(job_id)`` inside ``_q_sprite._retexture``), never
    # under the source mesh's -- that is where its published model.glb lands.
    views_dir = worker.config.job_dir(retexture_id) / "views"
    views_dir.mkdir(parents=True, exist_ok=True)
    (views_dir / "view_00.png").write_bytes(b"render")
    (views_dir / "restyled_00.png").write_bytes(b"restyle")

    source_dir = worker.config.job_dir(source)
    tmp = source_dir / rig_store.RETEXTURE_GLB_TMP
    tmp.write_bytes(b"half-written")

    job = {"id": retexture_id, "kind": "retexture", "params": {"source_job": source}}
    worker._discard_artifacts(job)

    assert not views_dir.exists(), "a cancelled re-texture must not strand its own renders"
    assert not tmp.exists()
    assert (source_dir / "model.glb").read_bytes() == b"old-model"


def test_a_cancelled_remesh_does_not_reach_for_a_views_directory_it_never_made(worker):
    """``_remesh`` writes no ``views/`` at all -- the rmtree call used to sit
    in this arm by copy-paste, doing nothing there but masking the real bug.
    A views directory that happens to exist beside a remesh job (left by an
    unrelated retexture of the same mesh) must survive a remesh cancel."""
    source = _mesh_job(worker)
    source_dir = worker.config.job_dir(source)
    stray_views = source_dir / "views"
    stray_views.mkdir(parents=True, exist_ok=True)
    (stray_views / "keep.png").write_bytes(b"not mine")
    tmp = source_dir / rig_store.REMESH_GLB_TMP
    tmp.write_bytes(b"half-written")

    job = {"id": "remeshA", "kind": "remesh", "params": {"source_job": source}}
    worker._discard_artifacts(job)

    assert not tmp.exists()
    assert stray_views.exists(), "a remesh cancel must not delete a directory it never wrote"
