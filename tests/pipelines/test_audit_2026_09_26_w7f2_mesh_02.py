"""Regression test for the 2026-09-26 audit, finding pipelines-mesh-02.

``modelhistory.stage()`` no-ops -- returns ``entries`` unchanged -- when there
is no ``model.glb`` on disk to snapshot (a job whose mesh failed to build this
time around). Before this fix, its return value gave no way to tell that
no-op apart from "a version really was pushed": ``discard_last`` after a
later failure could only look at the list and pop whatever was last in it --
which, on the no-op path, is the *previous, real* committed version -- and
delete its files, even though this ``stage`` call pushed nothing that needs
undoing.
"""

from __future__ import annotations

import time

from realmspinner.pipelines import modelhistory


def test_discard_after_a_no_op_stage_does_not_delete_the_real_previous_version(
    tmp_path,
):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "model.glb").write_bytes(b"model-v1")

    # A real, successful stage: version 1 is a genuine committed mesh.
    entries, staged_1 = modelhistory.stage(
        job_dir, [], {}, kind="remesh", geometry=True, detail="first", now=time.time()
    )
    assert staged_1 is True
    assert len(entries) == 1
    v1 = modelhistory.version_path(job_dir, entries[0]["n"])
    assert v1.exists()

    # This run's mesh never got as far as writing model.glb -- stage() has
    # nothing to snapshot, so it must no-op and say so.
    (job_dir / "model.glb").unlink()
    entries_after_noop, staged_2 = modelhistory.stage(
        job_dir, entries, {}, kind="retexture", geometry=False, detail="second",
        now=time.time(),
    )
    assert staged_2 is False
    assert entries_after_noop == entries

    # The caller's own write then fails and it backs out via discard_last().
    # Because nothing was staged, this must be a pure no-op: the real version
    # 1 -- staged before this call ever ran -- must survive on disk and in
    # the index.
    recovered = modelhistory.discard_last(job_dir, entries_after_noop, staged_2)

    assert recovered == entries, "a no-op stage must leave the real history untouched"
    assert v1.exists(), (
        "discard_last deleted the previous, real committed version's files "
        "after a stage() call that staged nothing"
    )


def test_discard_after_a_real_stage_does_undo_it(tmp_path):
    """The positive case: a genuine stage is still undone correctly."""
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "model.glb").write_bytes(b"model-v1")

    entries, staged = modelhistory.stage(
        job_dir, [], {}, kind="remesh", geometry=True, detail="first", now=time.time()
    )
    assert staged is True
    v1 = modelhistory.version_path(job_dir, entries[0]["n"])
    assert v1.exists()

    recovered = modelhistory.discard_last(job_dir, entries, staged)

    assert recovered == []
    assert not v1.exists()
