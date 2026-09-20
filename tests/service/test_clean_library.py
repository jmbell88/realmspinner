"""Clean library: the one bulk delete that keeps no jobs at all.

``prune_jobs`` and ``empty_trash`` both skip a job carrying evidence
(``retained_job_ids``), for the measured reason recorded there -- and the effect
was that the app had no way to say "remove everything". This is that way, and
the tests below pin the two halves that make it a *feature* rather than a
regression of the guard: it really does take the accepted meshes and the
labelled references, and it really does leave everything that is not a job.
"""

from __future__ import annotations

import pytest

from realmspinner.service import jobs as svc_jobs
from realmspinner.service import verdicts as svc_verdicts
from realmspinner.service.errors import Conflict


def _finished(svc, *, image: bool = False) -> str:
    """A done job with a directory on disk. -> its id."""
    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    svc.store.set_status(job_id, "done")
    svc.job_dir(job_id).mkdir(parents=True, exist_ok=True)
    (svc.job_dir(job_id) / "model.glb").write_bytes(b"mesh")
    if image:
        (svc.job_dir(job_id) / "input.png").write_bytes(b"not really a png")
    return job_id


class _Worker:
    def __init__(self, current: str | None = None) -> None:
        self.current_job_id = current


def test_it_takes_the_accepted_mesh_and_the_labelled_reference(svc):
    """The whole point, stated as a test so nobody restores the guard here by
    analogy with prune. Both of these survive a prune on purpose; neither
    survives this, and the confirmation dialog says so."""
    accepted = _finished(svc)
    svc_verdicts.record_verdict(svc, accepted, grade=3)
    labelled = _finished(svc, image=True)
    svc_verdicts.record_verdict(svc, labelled, verdict="reject", stage="reference")
    ordinary = _finished(svc)
    trashed = _finished(svc)
    svc_jobs.trash_job(svc, trashed)

    outcome = svc_jobs.clean_jobs(svc)

    assert outcome["deleted"] == 4
    for job_id in (accepted, labelled, ordinary, trashed):
        assert svc.store.get(job_id) is None
        assert not svc.job_dir(job_id).exists()


def test_the_verdict_rows_outlive_the_pixels(svc):
    """Deliberate, and the reason the database itself is never dropped: a
    verdict is denormalized so what a review taught survives its assets, and
    the corpus is the one thing here that cannot be regenerated."""
    job_id = _finished(svc)
    svc_verdicts.record_verdict(svc, job_id, grade=3)

    svc_jobs.clean_jobs(svc)

    assert [v["job_id"] for v in svc.store.latest_verdicts()] == [job_id]


def test_a_directory_no_row_names_is_collected_too(svc):
    """Orphans are invisible to every other path in the file -- which is why a
    library can measure larger than the sum of the jobs in it."""
    orphan = svc.config.data_dir / "0123456789ab"
    orphan.mkdir(parents=True)
    (orphan / "model.glb").write_bytes(b"stranded")

    outcome = svc_jobs.clean_jobs(svc)

    assert outcome["orphans"] == 1
    assert not orphan.exists()


def test_it_keeps_everything_that_is_not_a_job(svc):
    """The pose library, the style anchors, the autosaves and the settings are
    the user's own work rather than generated output, and a name that is not
    twelve hex characters can never be mistaken for a job directory."""
    data = svc.config.data_dir
    for name in ("poser", "profiles", "autosave"):
        (data / name).mkdir(parents=True, exist_ok=True)
        (data / name / "keep.json").write_text("{}", encoding="utf-8")
    (data / "studio_settings.json").write_text("{}", encoding="utf-8")
    _finished(svc)

    svc_jobs.clean_jobs(svc)

    for name in ("poser", "profiles", "autosave"):
        assert (data / name / "keep.json").exists()
    assert (data / "studio_settings.json").exists()
    assert svc.config.db_path.exists()


def test_a_queued_job_refuses_the_whole_call(svc):
    """Refused, not skipped. Prune skips a busy job because one rig in flight
    is no reason to keep two hundred other assets; "delete everything" that
    silently left three behind has failed at the only thing it claims to do."""
    _finished(svc)
    svc_jobs.create_job(svc, kind="text", prompt="queued one")

    with pytest.raises(Conflict) as caught:
        svc_jobs.clean_jobs(svc)

    assert "1" in str(caught.value)


def test_a_running_worker_refuses_even_with_no_active_row(svc):
    """``worker_is_inside``'s reason: a cancelled row is terminal in the DB
    while the reconstruction is still unwinding into its directory."""
    job_id = _finished(svc)
    svc.worker = _Worker(job_id)

    with pytest.raises(Conflict):
        svc_jobs.clean_jobs(svc)

    assert svc.store.get(job_id) is not None


def test_an_empty_library_is_not_an_error(svc):
    assert svc_jobs.clean_jobs(svc) == {"deleted": 0, "orphans": 0}


def test_clean_jobs_does_not_delete_a_directory_a_job_submitted_mid_walk_is_writing_into(
    svc, monkeypatch
):
    """service-01, the 2026-09-20 audit: ``clean_jobs`` read ``active_jobs()``
    once up front and then deleted across a multi-page walk guarded only by
    ``delete_if_not_running``, so a rig submitted after that snapshot -- while
    the walk collecting ``seen`` is still in progress -- could have its source
    mesh's directory removed while it was still writing into it.
    ``prune_jobs`` and ``empty_trash`` both re-check ``worker_is_inside``/
    ``dependent_jobs`` per row for exactly this reason; this pins that
    ``clean_jobs`` now does too.
    """
    source = _finished(svc)
    real_list = svc.store.list
    submitted = False

    def _list_and_submit(*args, **kwargs):
        # Simulates a rig being submitted for `source` after clean_jobs's
        # up-front active_jobs() snapshot, in the gap while the walk that
        # builds `seen` is still running -- the "mid-walk" race the finding
        # names. The rig's later created_at keeps it out of the page itself,
        # exactly as a genuinely concurrent submit would.
        nonlocal submitted
        page = real_list(*args, **kwargs)
        if not submitted:
            submitted = True
            svc.store.create(
                "rig", "", {"source_job": source}, stage="model", status="queued"
            )
        return page

    monkeypatch.setattr(svc.store, "list", _list_and_submit)

    svc_jobs.clean_jobs(svc)

    assert svc.store.get(source) is not None
    assert svc.job_dir(source).exists()


def test_trash_size_excludes_jobs_empty_trash_will_keep(svc):
    """shell-02, the 2026-09-13 audit: ``trash_size`` counted every trashed
    job and its bytes, but ``empty_trash`` keeps a job named by
    ``retained_job_ids`` (an accepted mesh or a labelled reference). The
    figure a user confirms the destructive "Empty trash" dialog against was
    therefore always an overstatement once anything had been graded or
    labelled -- and nothing afterwards said what emptying actually freed.
    """
    labelled = _finished(svc, image=True)
    svc_verdicts.record_verdict(svc, labelled, verdict="reject", stage="reference")
    svc_jobs.trash_job(svc, labelled)
    ordinary = _finished(svc)
    svc_jobs.trash_job(svc, ordinary)

    size = svc_jobs.trash_size(svc)
    outcome = svc_jobs.empty_trash(svc)

    # Both jobs are in the trash, but only the ordinary one will actually be
    # reclaimed -- the figure shown before the confirm must match that, not
    # the raw count of trashed rows.
    assert size["count"] == 1
    assert outcome == {"deleted": 1, "kept": 1}
    assert svc.store.get(labelled) is not None
    assert svc.store.get(ordinary) is None
