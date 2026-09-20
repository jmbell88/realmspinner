"""What a bulk delete keeps, and what it still takes.

The measured incident is in ``service/evidence.py``'s docstring and in
``dev/measurements/2026-09-07-mesh-probe-preregistration.md``: nine model-stage
human verdicts on the machine and **zero** of them still carrying a
``source.glb``, because filing the last verdict of a sweep is what fires
``cleanup_sweep``, and ``cleanup_sweep`` is what took them. Finishing the
judgement was the act of destroying the evidence for it.

Three claims, and they are separable on purpose:

* the archive happens **before** the delete, and the delete still happens --
  this is not a retention rule, and the reclaim the user asked for is unchanged;
* only what was judged or what a campaign tagged is kept, so a routine prune
  does not quietly turn into a rename;
* ``job.json`` is written last, so an interrupted archive cannot look complete.
"""

from __future__ import annotations

import json

from realmspinner.service import evidence
from realmspinner.service import jobs as svc_jobs
from realmspinner.service import sweeps as svc_sweeps
from realmspinner.service import verdicts as svc_verdicts
from realmspinner.service.sweeps import Axis, SweepPlan


def _plan(**kwargs) -> SweepPlan:
    fields = {"label": "lora", "prompt": "a wooden chest", "seeds": (1, 2)}
    fields.update(kwargs)
    fields["base"] = {"style_lora": "render3d", **dict(fields.get("base") or {})}
    return SweepPlan(**fields)


def _finished(svc, job_id: str) -> None:
    """A unit with the two artifacts a later question actually wants."""
    svc.store.set_status(job_id, "done")
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input.png").write_bytes(b"reference-pixels")
    (job_dir / "source.glb").write_bytes(b"glTF-reconstruction")
    (job_dir / "model.glb").write_bytes(b"glTF-derived")


def _archive_dir(svc, job_id: str):
    """Where this job's archive landed. The bucket name carries today's date."""
    for bucket in svc.config.evidence_dir.iterdir():
        if (bucket / job_id).is_dir():
            return bucket / job_id
    raise AssertionError(f"no archive for {job_id}")


def _archived(svc) -> dict[str, dict]:
    """Every complete archive on disk, by job id."""
    out: dict[str, dict] = {}
    root = svc.config.evidence_dir
    if not root.is_dir():
        return out
    for bucket in root.iterdir():
        for job_dir in bucket.iterdir():
            if evidence.is_complete(job_dir):
                out[job_dir.name] = json.loads(
                    (job_dir / evidence.MANIFEST).read_text("utf-8")
                )
    return out


# --- the grading loop stops eating its own corpus ---------------------------


def test_a_graded_sweeps_meshes_are_archived_before_cleanup_deletes_them(svc):
    """``cleanup_sweep`` fires with a toast and no dialog when a sweep's last
    verdict lands, and it used to be the end of those meshes."""
    plan = _plan(seeds=(1, 2), axes=(Axis("lora_weight", (0.6,)),))
    result = svc_sweeps.create_sweep(svc, plan)
    units = svc.store.sweep_jobs(result["id"])
    for unit in units:
        _finished(svc, unit["id"])
        svc_verdicts.record_verdict(svc, unit["id"], grade=-4)

    outcome = svc_sweeps.cleanup_sweep(svc, result["id"])

    # The reclaim is unchanged: the job rows and directories are gone.
    assert outcome["deleted"] == len(units)
    assert all(svc.store.get(unit["id"]) is None for unit in units)
    # And the meshes the grades were filed against are still readable.
    archives = _archived(svc)
    assert set(archives) == {unit["id"] for unit in units}
    for unit in units:
        kept = _archive_dir(svc, unit["id"])
        assert (kept / "source.glb").read_bytes() == b"glTF-reconstruction"
        assert (kept / "input.png").read_bytes() == b"reference-pixels"
    assert outcome["archived"] == len(units)


def test_a_rejected_mesh_is_archived_even_though_retention_does_not_keep_it(svc):
    """Retention's own docstring says a model-stage reject is carried by its row
    -- true of that finding, false of a regression probe over grades, which
    needs both ends of the scale. This is retention's other half."""
    plan = _plan(seeds=(1,), axes=(Axis("lora_weight", (0.6,)),))
    result = svc_sweeps.create_sweep(svc, plan)
    units = svc.store.sweep_jobs(result["id"])
    for unit in units:
        _finished(svc, unit["id"])
        svc_verdicts.record_verdict(svc, unit["id"], grade=-5)

    assert svc_jobs.retained_job_ids(svc) == set()

    svc_sweeps.delete_sweep(svc, result["id"])

    assert set(_archived(svc)) == {unit["id"] for unit in units}


def test_the_archive_carries_the_grade_and_the_settings_it_was_filed_against(svc):
    """A mesh with nothing saying what it was or how it scored is not evidence."""
    plan = _plan(seeds=(1,), axes=(Axis("lora_weight", (0.6,)),))
    result = svc_sweeps.create_sweep(svc, plan)
    unit = svc.store.sweep_jobs(result["id"])[0]
    _finished(svc, unit["id"])
    svc_verdicts.record_verdict(svc, unit["id"], grade=-2)

    svc_sweeps.cleanup_sweep(svc, result["id"])

    doc = _archived(svc)[unit["id"]]
    assert doc["job"]["prompt"] == "a wooden chest"
    assert doc["params"]["seed"] == 1
    assert [v["grade"] for v in doc["verdicts"]] == [-2]
    assert doc["files"]["source.glb"]["bytes"] == len(b"glTF-reconstruction")


# --- prune ------------------------------------------------------------------


def test_prune_archives_a_judged_job_before_removing_it(svc):
    job_id = svc_jobs.create_job(svc, kind="text", prompt="a barrel")["id"]
    _finished(svc, job_id)
    svc_verdicts.record_verdict(svc, job_id, grade=-3)

    outcome = svc_jobs.prune_jobs(svc, keep=0)

    assert outcome["deleted"] == 1
    assert outcome["archived"] == 1
    assert job_id in _archived(svc)


def test_prune_archives_a_tagged_campaign_job_nobody_has_graded_yet(svc):
    """A tag is a submitter saying in advance that this run is a measurement.
    ``campaign_props.py`` prints "do not clean the library until the writeup
    exists"; this is what happens when somebody does anyway."""
    job_id = svc_jobs.create_job(svc, kind="text", prompt="a barrel")["id"]
    _finished(svc, job_id)
    svc.store.merge_params(job_id, {"tags": ["props-v1"]})

    svc_jobs.prune_jobs(svc, keep=0)

    assert job_id in _archived(svc)


def test_prune_does_not_archive_ordinary_work(svc):
    """Otherwise the reclaim is a rename, which is the opposite of the ask."""
    job_id = svc_jobs.create_job(svc, kind="text", prompt="a barrel")["id"]
    _finished(svc, job_id)

    outcome = svc_jobs.prune_jobs(svc, keep=0)

    assert outcome["deleted"] == 1
    assert outcome["archived"] == 0
    assert _archived(svc) == {}


# --- the completion gate ----------------------------------------------------


def test_an_interrupted_archive_cannot_look_complete(svc, monkeypatch):
    """``job.json`` is written last, ``journal.py``'s rule. A directory of files
    nobody can say anything about must be distinguishable from a finished one
    without trusting a listing."""
    job_id = svc_jobs.create_job(svc, kind="text", prompt="a barrel")["id"]
    _finished(svc, job_id)
    svc_verdicts.record_verdict(svc, job_id, grade=-3)

    def boom(*_args, **_kwargs):
        raise OSError("the disk filled up between the mesh and the manifest")

    monkeypatch.setattr("pathlib.Path.write_text", boom)
    assert evidence.archive_job(svc, job_id, reason="prune") is None

    root = svc.config.evidence_dir
    written = [d for bucket in root.iterdir() for d in bucket.iterdir()]
    assert written, "the artifacts were copied before the failure"
    assert not any(evidence.is_complete(d) for d in written)
    assert _archived(svc) == {}


def test_an_archive_that_cannot_be_written_does_not_stop_the_delete(svc, monkeypatch):
    """The user asked for the disk back. A failed archive is a log line.

    Every other non-fatal step in this layer makes the same call, and the
    alternative here is the worse one in both directions: a prune that aborts
    mid-page reclaims almost nothing *and* leaves the rows it already deleted.
    """
    job_id = svc_jobs.create_job(svc, kind="text", prompt="a barrel")["id"]
    _finished(svc, job_id)
    svc_verdicts.record_verdict(svc, job_id, grade=-3)

    def boom(*_args, **_kwargs):
        raise OSError("the archive volume went away")

    monkeypatch.setattr(evidence, "archive_job", boom)

    outcome = svc_jobs.prune_jobs(svc, keep=0)

    assert outcome["deleted"] == 1
    assert outcome["archived"] == 0
    assert svc.store.get(job_id) is None


# --- what is deliberately not a caller --------------------------------------


def test_clean_jobs_still_keeps_nothing(svc):
    """``clean_jobs`` says in as many words that it keeps nothing -- "a user
    reclaiming a disk, handing a machine on, or starting a corpus over is not
    asking for a reclaim that quietly keeps the largest meshes on it". An
    archive behind their back would break the one promise it makes."""
    job_id = svc_jobs.create_job(svc, kind="text", prompt="a barrel")["id"]
    _finished(svc, job_id)
    svc_verdicts.record_verdict(svc, job_id, grade=5)

    svc_jobs.clean_jobs(svc)

    assert _archived(svc) == {}


# --- the size the user finds out about from the app -------------------------


def test_usage_reports_the_archive_and_tolerates_a_missing_tree(svc):
    assert evidence.usage(svc.config) == {"jobs": 0, "files": 0, "bytes": 0}

    job_id = svc_jobs.create_job(svc, kind="text", prompt="a barrel")["id"]
    _finished(svc, job_id)
    svc_verdicts.record_verdict(svc, job_id, grade=-3)
    evidence.archive_job(svc, job_id, reason="prune")

    used = evidence.usage(svc.config)
    assert used["jobs"] == 1
    assert used["bytes"] > 0
