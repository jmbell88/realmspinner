"""``service.export.plan_export``: what an export would write, before it
writes anything (W2.2).

The pane used to write immediately on "Export zip..." / "Save to project",
so what landed where -- and what got overwritten -- was only ever visible
after the fact. ``plan_export`` is the pure half that lets a popup say so
first; these tests never touch ``bulk_export``/``export_to_folder`` at all,
because the claim under test is that *planning* writes nothing.
"""

from __future__ import annotations

import pytest

from warlock.service import export as svc_export
from warlock.service import jobs as svc_jobs
from warlock.service.errors import Conflict


@pytest.fixture
def assets(svc):
    return svc.config.data_dir


def _done_job(svc, assets, name: str = "model.glb") -> str:
    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    job_dir = assets / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / name).write_bytes(b"glb")
    svc.store.set_status(job_id, "done")
    return job_id


def test_export_lists_every_file_it_will_write(svc, assets, tmp_path):
    a = _done_job(svc, assets)
    b = _done_job(svc, assets)
    dest = tmp_path / "project" / "assets"  # does not exist yet

    job = svc_export.ExportJob(svc=svc, ids=[a, b], names_wanted=None, as_zip=False)
    plan = svc_export.plan_export(job, dest)

    names = {f.name for f in plan.files}
    assert names == {f"{a}/model.glb", f"{b}/model.glb"}
    # Nothing was written -- planning is pure.
    assert not dest.exists()
    assert all(not f.exists for f in plan.files)


def test_plan_export_writes_nothing_to_disk_for_a_zip_destination(svc, assets, tmp_path):
    """The zip half of the same claim: a fresh destination stays untouched."""
    a = _done_job(svc, assets)
    dest = tmp_path / "out" / "warlock_export.zip"

    job = svc_export.ExportJob(svc=svc, ids=[a], names_wanted=None, as_zip=True)
    plan = svc_export.plan_export(job, dest)

    assert [f.name for f in plan.files] == ["warlock_export.zip"]
    assert not dest.exists()
    assert not dest.parent.exists()
    assert plan.files[0].exists is False


def test_an_existing_destination_file_is_named_before_overwrite(svc, assets, tmp_path):
    """A file already at the destination is marked, by name, in the plan --
    not discovered afterwards as a silently clobbered file."""
    a = _done_job(svc, assets)
    dest = tmp_path / "out" / "warlock_export.zip"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"an earlier export")

    job = svc_export.ExportJob(svc=svc, ids=[a], names_wanted=None, as_zip=True)
    plan = svc_export.plan_export(job, dest)

    assert len(plan.existing) == 1
    assert plan.existing[0].name == "warlock_export.zip"
    assert plan.existing[0].dest == dest
    # And the file plan_export found still holds exactly what it held before
    # -- named, not touched.
    assert dest.read_bytes() == b"an earlier export"


def test_an_existing_destination_file_is_named_before_overwrite_for_a_folder_export(
    svc, assets, tmp_path
):
    a = _done_job(svc, assets)
    b = _done_job(svc, assets)
    dest = tmp_path / "project" / "assets"
    (dest / a).mkdir(parents=True)
    (dest / a / "model.glb").write_bytes(b"already here")

    job = svc_export.ExportJob(svc=svc, ids=[a, b], names_wanted=None, as_zip=False)
    plan = svc_export.plan_export(job, dest)

    existing_names = {f.name for f in plan.existing}
    assert existing_names == {f"{a}/model.glb"}
    assert (dest / a / "model.glb").read_bytes() == b"already here"


def test_keep_both_suffixes_the_whole_set(svc, assets, tmp_path):
    """"Keep both" renames every planned file by the same number, even the
    ones with no collision of their own -- so the set stays matched, and a
    number already spoken for (an earlier "Keep both") is skipped for all of
    them rather than reused."""
    a = _done_job(svc, assets)
    b = _done_job(svc, assets)
    dest = tmp_path / "project" / "assets"
    # a/model.glb collides today; b/model.glb does not. And an earlier
    # "Keep both" already claimed a/model-2.glb, so plain -2 is not free for
    # the whole set even though b/model-2.glb itself is unclaimed.
    (dest / a).mkdir(parents=True)
    (dest / a / "model.glb").write_bytes(b"current")
    (dest / a / "model-2.glb").write_bytes(b"an earlier keep-both")

    job = svc_export.ExportJob(svc=svc, ids=[a, b], names_wanted=None, as_zip=False)
    plan = svc_export.plan_export(job, dest)
    kept = svc_export.keep_both(plan)

    # Every renamed file carries the same suffix -- 3, the smallest number
    # free for *all* of them -- never a mix of -2 and -3.
    for renamed in kept.files:
        assert renamed.name.endswith("-3.glb")
        assert renamed.exists is False
    assert {f.dest.name for f in kept.files} == {"model-3.glb"}


def test_keep_both_picks_the_smallest_free_number_when_nothing_collides(svc, assets, tmp_path):
    a = _done_job(svc, assets)
    dest = tmp_path / "project" / "assets"

    job = svc_export.ExportJob(svc=svc, ids=[a], names_wanted=None, as_zip=False)
    plan = svc_export.plan_export(job, dest)
    kept = svc_export.keep_both(plan)

    assert kept.files[0].name == f"{a}/model-2.glb"
    assert kept.files[0].exists is False


def test_export_planned_to_folder_refuses_cleanly_when_the_plan_goes_stale(svc, assets, tmp_path):
    """service-03 (2026-09-13 audit): the Keep both/Replace popup can outlive
    a readiness change -- a second job finishes, or one goes stale, while the
    popup is still on screen. ``export_planned_to_folder`` used to re-collect
    against the (by then wrong) ``plan.files`` with ``zip(..., strict=True)``,
    which raised a raw ``ValueError`` the pane had no ``field`` to toast. It
    must instead refuse with a ``service.errors`` exception."""
    a = _done_job(svc, assets)
    dest = tmp_path / "project" / "assets"
    svc.config.export_dir = dest

    job = svc_export.ExportJob(svc=svc, ids=[a], names_wanted=None, as_zip=False)
    plan = svc_export.plan_export(job, dest)

    # The plan was built for one ready job; a second job becomes ready before
    # "Keep both" is clicked, so ``collect`` now returns more members than the
    # stale plan describes.
    b = _done_job(svc, assets)

    with pytest.raises(Conflict) as excinfo:
        svc_export.export_planned_to_folder(svc, [a, b], None, plan)
    assert excinfo.value.field == "plan"
    # And nothing was written for the mismatch -- a refused plan writes
    # nothing, matching every other refusal in this module.
    assert not dest.exists()
