"""The rig surface. No Blender: every entry point here is reachable, and every
error it can return is reachable, without bpy installed."""

from __future__ import annotations

import json

import pytest

from warlock import doctor, rigging
from warlock.service import Invalid, NotFound, NotReady
from warlock.service import derive as svc_derive
from warlock.service import jobs as svc_jobs
from warlock.service import rig as svc_rig
from warlock.service import system as svc_system

IDENTITY = [0.0, 0.0, 0.0, 1.0]


@pytest.fixture
def assets(svc):
    return svc.config.data_dir


def _finished_mesh_job(svc, assets) -> str:
    """A job the service will agree is rigg-able: status done, model.glb on disk."""
    job_id = svc_jobs.create_job(svc, kind="text", prompt="a barrel")["id"]
    job_dir = assets / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"fake-glb")
    svc.store.set_status(job_id, "done")
    return job_id


# --- catalogue --------------------------------------------------------------


def test_the_template_catalogue_lists_both_skeletons(svc):
    body = svc_rig.rig_templates(svc)
    assert {t["key"] for t in body["templates"]} >= {"humanoid", "quadruped"}
    assert body["default"] == "humanoid"
    # available reflects bpy, which a given machine may or may not have -- but
    # it must be a bool and carry a reason either way, because the UI keys off it.
    assert isinstance(body["available"], bool)
    assert body["detail"]


def test_health_reports_rigging_without_failing_startup(svc):
    body = svc_system.health(svc)
    names = {c["name"]: c for c in body["checks"]}
    assert "Blender (rigging)" in names
    assert names["Blender (rigging)"]["fatal"] is False


# --- queueing a rig ---------------------------------------------------------


def test_rigging_an_unknown_job_is_not_found(svc):
    with pytest.raises(NotFound):
        svc_rig.create_rig(svc, "0123456789ab")


@pytest.mark.parametrize("bad", ["not-an-id", "ABCDEF012345", "0123456789abcd", "../.."])
def test_the_rig_entry_points_reject_malformed_job_ids(svc, bad):
    """config.job_dir() is a bare path join, so every id-bearing entry point
    needs the same guard get_file has."""
    with pytest.raises(NotFound):
        svc_rig.create_rig(svc, bad)
    with pytest.raises(NotFound):
        svc_rig.get_rig(svc, bad)


def test_rigging_requires_a_finished_mesh(svc):
    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    with pytest.raises(Invalid, match="finished mesh"):
        svc_rig.create_rig(svc, job_id)


def test_rigging_rejects_an_unknown_template(svc, assets):
    job_id = _finished_mesh_job(svc, assets)
    with pytest.raises(Invalid, match="dragon"):
        svc_rig.create_rig(svc, job_id, template="dragon")


def test_a_rig_job_points_back_at_its_source(svc, assets):
    job_id = _finished_mesh_job(svc, assets)
    out = svc_rig.create_rig(svc, job_id, template="quadruped")
    rig_job = svc.store.get(out["id"])
    assert out["id"] != job_id
    assert rig_job["kind"] == "rig"
    # source_job is what lets the UI attach this to the parent card instead of
    # listing it as an unrelated row.
    assert rig_job["params"]["source_job"] == job_id
    assert rig_job["params"]["template"] == "quadruped"


def test_a_rig_job_cannot_itself_be_rigged(svc, assets):
    job_id = _finished_mesh_job(svc, assets)
    rig_id = svc_rig.create_rig(svc, job_id)["id"]
    with pytest.raises(Invalid):
        svc_rig.create_rig(svc, rig_id)


def test_a_submit_can_opt_into_rigging(svc):
    out = svc_jobs.create_job(
        svc, kind="text", prompt="a knight", rig=True, rig_template="humanoid"
    )
    params = svc.store.get(out["id"])["params"]
    assert params["rig"] is True
    assert params["rig_template"] == "humanoid"


def test_a_submit_rejects_an_unknown_template_up_front(svc):
    """Better to lose the request than to lose the rig 90 seconds after the
    mesh finishes."""
    with pytest.raises(Invalid):
        svc_jobs.create_job(svc, kind="text", prompt="x", rig=True, rig_template="dragon")


def test_a_submit_without_the_flag_records_no_rig_intent(svc):
    out = svc_jobs.create_job(svc, kind="text", prompt="x")
    assert "rig" not in svc.store.get(out["id"])["params"]


# --- reading a rig back -----------------------------------------------------


def test_rig_metadata_is_absent_until_rigged(svc, assets):
    job_id = _finished_mesh_job(svc, assets)
    with pytest.raises(NotFound):
        svc_rig.get_rig(svc, job_id)


def test_rig_metadata_is_served_once_written(svc, assets):
    job_id = _finished_mesh_job(svc, assets)
    meta = {"version": 1, "template": "humanoid", "weighting": "envelope", "bones": []}
    (assets / job_id / "rig.json").write_text(json.dumps(meta))
    assert svc_rig.get_rig(svc, job_id) == meta


def test_a_rig_with_a_nameless_bone_is_refused_not_a_keyerror(svc, assets):
    """The 2026-09-08 audit (poser-02): rig.json passes read_record's three
    file-level guards (valid JSON, valid dict, under the byte ceiling) with a
    bone entry that has no "name" key, and get_rig used to let
    rigging.rig_bone_names' bare ``[b["name"] for b in ...]`` crash out as an
    uncaught KeyError instead of a field-addressed refusal -- one field
    deeper than the pose-record case docs/INVARIANTS.md already names as
    fixed. tests/test_poses_api.py covers the same fix for list_poses,
    save_pose and apply_library_pose."""
    job_id = _finished_mesh_job(svc, assets)
    (assets / job_id / "rig.json").write_text(
        json.dumps({"version": 1, "template": "humanoid", "bones": [{"name": "hips"}, {}]})
    )
    with pytest.raises(Invalid) as caught:
        svc_rig.get_rig(svc, job_id)
    assert caught.value.field == "bones"


def test_rig_glb_is_hidden_until_rig_json_lands(svc, assets):
    """rig.json is written last, so it -- not rig.glb's own existence -- is
    what says the rig finished. Otherwise a read can catch a half-exported GLB."""
    job_id = _finished_mesh_job(svc, assets)
    (assets / job_id / "rig.glb").write_bytes(b"half-written")
    assert "rig.glb" not in svc_jobs.get_job(svc, job_id)["files"]
    (assets / job_id / "rig.json").write_text('{"bones": []}')
    assert "rig.glb" in svc_jobs.get_job(svc, job_id)["files"]


def test_reading_rig_glb_is_gated_on_rig_json_too(svc, assets):
    """The same gate as the files listing, on the call that actually hands back
    a path: the Blender export is not atomic, so a direct read must not be able
    to walk in ahead of rig.json and collect a truncated GLB."""
    job_id = _finished_mesh_job(svc, assets)
    (assets / job_id / "rig.glb").write_bytes(b"half-written")
    with pytest.raises(NotReady):
        svc_derive.get_file(svc, job_id, "rig.glb")


def test_rig_glb_is_readable_once_complete(svc, assets):
    job_id = _finished_mesh_job(svc, assets)
    (assets / job_id / "rig.glb").write_bytes(b"glb-bytes")
    (assets / job_id / "rig.json").write_text('{"bones": []}')
    path = svc_derive.get_file(svc, job_id, "rig.glb")
    assert path.read_bytes() == b"glb-bytes"


def test_rig_glb_is_not_ready_when_absent(svc, assets):
    job_id = _finished_mesh_job(svc, assets)
    with pytest.raises(NotReady):
        svc_derive.get_file(svc, job_id, "rig.glb")


# --- corrected joints -------------------------------------------------------


def _rigged_job(svc, assets) -> tuple[str, list[dict]]:
    """A job with a rig on disk, and the fitted bones the editor would show."""
    from warlock import rigging

    job_id = _finished_mesh_job(svc, assets)
    job_dir = assets / job_id
    template = rigging.get_template("humanoid")
    fitted = rigging.fit_template(template, [-1, -1, 0], [1, 1, 2])
    (job_dir / "rig.json").write_text(
        json.dumps({"template": "humanoid", "bones": fitted}), encoding="utf-8"
    )
    (job_dir / "rig.glb").write_bytes(b"fake-rig")
    return job_id, fitted


def test_adjusting_joints_queues_a_rerig(svc, assets):
    job_id, fitted = _rigged_job(svc, assets)
    out = svc_rig.adjust_joints(
        svc,
        job_id,
        {"bones": [{"name": b["name"], "head": b["head"], "tail": b["tail"]} for b in fitted]},
    )
    rig_job = svc.store.get(out["id"])
    assert rig_job["kind"] == "rig"
    assert rig_job["params"]["source_job"] == job_id
    assert rig_job["params"]["bones"]
    assert rig_job["params"]["adjusted"] is True


def test_adjusting_joints_on_an_unrigged_job_is_refused(svc, assets):
    job_id = _finished_mesh_job(svc, assets)
    with pytest.raises(Invalid):
        svc_rig.adjust_joints(svc, job_id, {"bones": []})


def test_adjusting_joints_on_a_missing_job_is_not_found(svc):
    with pytest.raises(NotFound):
        svc_rig.adjust_joints(svc, "0123456789ab", {"bones": []})


def test_adjusting_joints_rejects_a_partial_skeleton(svc, assets):
    job_id, fitted = _rigged_job(svc, assets)
    with pytest.raises(Invalid, match="missing bone"):
        svc_rig.adjust_joints(
            svc,
            job_id,
            {"bones": [{"name": fitted[0]["name"], "head": [0, 0, 0], "tail": [0, 0, 1]}]},
        )


# --- the bpy door ------------------------------------------------------------
#
# Today the UI hides the Rig button when ``rig_templates``' own probe says bpy
# is absent, so the only paths that reach ``create_rig``/``adjust_joints`` on
# such a host are the MCP agent surface (``studio/agent_clay.py``, derived
# from a tool surface that has no notion of this greying) and a stale frame.
# Both used to queue a job that could only die in
# ``pipelines/blender_worker.py`` with exit code 3 minutes later, instead of
# being refused here, at the door, before anything is written.


def test_rigging_without_blender_is_refused_at_the_door(svc, assets, monkeypatch):
    job_id = _finished_mesh_job(svc, assets)
    monkeypatch.setattr(
        doctor, "blender_check", lambda **_kw: doctor.Check(
            name="Blender (rigging)", ok=False, detail="bpy is not installed", fatal=False,
        ),
    )
    with pytest.raises(Invalid, match="Blender"):
        svc_rig.create_rig(svc, job_id)
    # And nothing was queued: the refusal is at the door, not after it.
    assert len(svc.store.list()) == 1  # only the source mesh job


def test_rerigging_without_blender_is_refused_at_the_door(svc, assets, monkeypatch):
    job_id, fitted = _rigged_job(svc, assets)
    monkeypatch.setattr(
        doctor, "blender_check", lambda **_kw: doctor.Check(
            name="Blender (rigging)", ok=False, detail="bpy is not installed", fatal=False,
        ),
    )
    with pytest.raises(Invalid, match="Blender"):
        svc_rig.adjust_joints(
            svc,
            job_id,
            {"bones": [{"name": b["name"], "head": b["head"], "tail": b["tail"]} for b in fitted]},
        )


# --- a job's own pose file, validated at the read door -----------------------
#
# The library pose door (service.poses._record_or_not_found) re-validates a
# saved pose's bones on every read, because a pose is a file in a directory
# any other program can edit. A job-scoped pose (this module's list_poses/
# save_pose/posed_model) never got that second half: rigging.read_pose only
# gives read_record's three file-level guards (valid JSON, valid dict, under
# the byte ceiling), so a hand-edited pose file missing "bones" reached
# _pose_bake_spec's ``pose["bones"]`` as a bare KeyError, and one with a
# malformed quaternion was forwarded straight into the Blender worker spec
# with nothing to catch it (the 2026-09-11 audit, finding poser-01).


def _posable_job(svc, assets) -> str:
    job_id = svc_jobs.create_job(svc, kind="text", prompt="a knight")["id"]
    job_dir = assets / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"fake-glb")
    (job_dir / "rig.glb").write_bytes(b"fake-rig")
    (job_dir / "rig.json").write_text(
        json.dumps({"version": 1, "bones": [{"name": "hips"}, {"name": "spine"}]})
    )
    svc.store.set_status(job_id, "done")
    return job_id


def test_a_job_pose_file_missing_bones_is_refused_cleanly_not_a_key_error(svc, assets, monkeypatch):
    job_id = _posable_job(svc, assets)
    job_dir = assets / job_id

    # Case 1: "bones" stripped entirely, exactly like a library pose record
    # test_pose_library_service.py's own broken-pose case corrupts.
    record = svc_rig.save_pose(svc, job_id, {"name": "idle", "bones": {"hips": IDENTITY}})
    pose_path = rigging.pose_path(job_dir, record["id"])
    on_disk = json.loads(pose_path.read_text(encoding="utf-8"))
    del on_disk["bones"]
    pose_path.write_text(json.dumps(on_disk), encoding="utf-8")

    called = []
    monkeypatch.setattr(
        rigging, "run_worker", lambda spec, **kw: called.append(spec) or {}
    )
    with pytest.raises(Invalid) as caught:
        svc_rig.posed_model(svc, job_id, record["id"])
    assert caught.value.field == "bones"
    # Refused before a Blender subprocess would ever have been spent on it.
    assert not called

    # Case 2: "bones" present but a malformed quaternion (wrong length) --
    # must be refused before it is forwarded into the worker spec, not baked.
    record2 = svc_rig.save_pose(svc, job_id, {"name": "wave", "bones": {"hips": IDENTITY}})
    pose_path2 = rigging.pose_path(job_dir, record2["id"])
    on_disk2 = json.loads(pose_path2.read_text(encoding="utf-8"))
    on_disk2["bones"] = {"hips": [1.0, 2.0, 3.0]}
    pose_path2.write_text(json.dumps(on_disk2), encoding="utf-8")

    with pytest.raises(Invalid) as caught2:
        svc_rig.posed_model(svc, job_id, record2["id"])
    assert caught2.value.field == "bones"
    assert not called


# --- the skeleton editor (P3: service/rig.py) --------------------------------


def _rigged_job_full(svc, assets, *, skeleton=None, root=None, mirror_pairs=None, bones=None):
    """A rig.json with everything ``edit_skeleton``/``adjust_joints`` read:
    bounds (for ``validate_skeleton``'s far-outside check) and, optionally, a
    skeleton already recorded as custom."""
    job_id = _finished_mesh_job(svc, assets)
    job_dir = assets / job_id
    template = rigging.get_template("humanoid")
    fitted = bones if bones is not None else rigging.fit_template(template, [-1, -1, 0], [1, 1, 2])
    rig = {
        "version": 1,
        "template": "humanoid",
        "bones": fitted,
        "root": root or template.root,
        "mirror_pairs": [list(p) for p in template.mirror_pairs],
        "bounds": {"min": [-1, -1, 0], "max": [1, 1, 2]},
        "skeleton": skeleton or "template",
    }
    (job_dir / "rig.json").write_text(json.dumps(rig), encoding="utf-8")
    (job_dir / "rig.glb").write_bytes(b"fake-rig")
    return job_id, fitted


def test_editing_the_skeleton_queues_a_rerig(svc, assets):
    job_id, fitted = _rigged_job_full(svc, assets)
    edited = rigging.add_bone(fitted, "hips", "tail_01", [0, -0.1, 0.5], [0, -0.3, 0.5])
    out = svc_rig.edit_skeleton(svc, job_id, {"bones": edited})
    assert out["skeleton"] == "custom"
    rig_job = svc.store.get(out["id"])
    assert rig_job["kind"] == "rig"
    assert rig_job["params"]["source_job"] == job_id
    assert rig_job["params"]["skeleton"] == "custom"
    assert rig_job["params"]["root"] == "hips"
    assert any(b["name"] == "tail_01" for b in rig_job["params"]["bones"])
    assert rig_job["params"]["adjusted"] is True


def test_editing_the_skeleton_unchanged_reports_template(svc, assets):
    job_id, fitted = _rigged_job_full(svc, assets)
    out = svc_rig.edit_skeleton(svc, job_id, {"bones": fitted})
    assert out["skeleton"] == "template"


def test_editing_the_skeleton_on_an_unrigged_job_is_invalid(svc, assets):
    job_id = _finished_mesh_job(svc, assets)
    with pytest.raises(Invalid):
        svc_rig.edit_skeleton(svc, job_id, {"bones": []})


def test_editing_the_skeleton_on_a_missing_job_is_not_found(svc):
    with pytest.raises(NotFound):
        svc_rig.edit_skeleton(svc, "0123456789ab", {"bones": []})


def test_editing_the_skeleton_rejects_a_bad_payload_with_a_field(svc, assets):
    job_id, fitted = _rigged_job_full(svc, assets)
    dupe = [dict(fitted[0], name=fitted[1]["name"])] + fitted[1:]
    with pytest.raises(Invalid) as caught:
        svc_rig.edit_skeleton(svc, job_id, {"bones": dupe})
    assert caught.value.field == "bones"


def test_editing_the_skeleton_without_blender_is_refused_at_the_door(svc, assets, monkeypatch):
    job_id, fitted = _rigged_job_full(svc, assets)
    monkeypatch.setattr(
        doctor, "blender_check", lambda **_kw: doctor.Check(
            name="Blender (rigging)", ok=False, detail="bpy is not installed", fatal=False,
        ),
    )
    with pytest.raises(Invalid, match="Blender"):
        svc_rig.edit_skeleton(svc, job_id, {"bones": fitted})
    assert len(svc.store.list()) == 1  # only the source mesh job


def test_limb_presets_lists_the_shipped_presets(svc):
    entries = svc_rig.limb_presets()
    keys = {e["key"] for e in entries}
    assert keys == {"arm", "leg", "tail", "wing", "antenna"}
    assert all(set(e) == {"key", "label", "bone_count"} for e in entries)
    assert all(e["bone_count"] > 0 for e in entries)


def test_adjusting_joints_on_a_custom_rig_is_accepted(svc, assets):
    """A joint move on a custom skeleton must be checked against *its own*
    structure, not the base template's -- validating against the template
    here would refuse a rig with, say, an extra tail bone the template never
    had. The queued job must carry the custom shape forward rather than
    silently resetting the rig back to template shape."""
    template = rigging.get_template("humanoid")
    fitted = rigging.fit_template(template, [-1, -1, 0], [1, 1, 2])
    custom_bones = rigging.add_bone(fitted, "hips", "tail_01", [0, -0.1, 0.5], [0, -0.3, 0.5])
    job_id, bones = _rigged_job_full(
        svc, assets, skeleton="custom", root="hips", bones=custom_bones
    )
    payload = {"bones": [{"name": b["name"], "head": b["head"], "tail": b["tail"]} for b in bones]}
    out = svc_rig.adjust_joints(svc, job_id, payload)
    rig_job = svc.store.get(out["id"])
    assert rig_job["kind"] == "rig"
    assert rig_job["params"]["skeleton"] == "custom"
    assert any(b["name"] == "tail_01" for b in rig_job["params"]["bones"])
