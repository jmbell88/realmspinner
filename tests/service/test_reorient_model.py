"""``reorient_model``: change a finished asset's size, pivot or front in place.

The door is inline (no job kind, no queue), so these drive it against a real GLB
on disk. The claims are about what it never does -- write ``model.glb`` in place,
turn a rigged mesh, leave a derived export describing the old one -- and about
the way back: a restore across it must bring ``size_m`` and the front with it.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from realmspinner.pipelines import modelhistory, postprocess
from realmspinner.service import Conflict, Failed, Invalid
from realmspinner.service import jobs as svc_jobs
from realmspinner.service.files import DERIVED


def _glb(path, *, nub: bool = True) -> None:
    """A 1 m cube with a small nub sticking out of +X, well off the origin.

    The nub is what a person would call the front: after the front is baked it
    must point at +Z, and it is the only thing in the scene that does.
    """
    scene = trimesh.Scene()
    body = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    body.apply_translation([4.0, 3.0, 2.0])
    scene.add_geometry(body, geom_name="body")
    if nub:
        tip = trimesh.creation.box(extents=(0.3, 0.2, 0.2))
        tip.apply_translation([5.0, 3.0, 2.0])
        scene.add_geometry(tip, geom_name="nub")
    scene.export(path)


def _job(svc, **params) -> str:
    job_id = svc_jobs.create_job(svc, kind="text", prompt="a barrel", **params)["id"]
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    _glb(job_dir / "source.glb")
    _glb(job_dir / "model.glb")
    svc.store.merge_params(job_id, {"size_m": 1.3})
    svc.store.set_status(job_id, "done")
    return job_id


def _bounds(path):
    return trimesh.load(path).bounds


def _nub_and_body(path):
    """The two parts' centroids, told apart by volume -- scene order is not a contract."""
    meshes = sorted(trimesh.load(path).dump(), key=lambda m: m.volume)
    return meshes[0].centroid, meshes[1].centroid, meshes


def test_normalize_glb_with_yaw_faces_front_along_plus_z_and_stays_grounded(tmp_path):
    """The camera looks at (sin t, cos t); a front set at t = 90 is the +X nub, and
    baking it must send +X to +Z -- with the model still centred and on the floor."""
    glb = tmp_path / "m.glb"
    _glb(glb)
    postprocess.normalize_glb(glb, None, yaw_deg=-90.0)
    nub, body, _ = _nub_and_body(glb)
    assert nub[2] > body[2] + 0.4, "the nub must now point at +Z"
    assert abs(nub[0] - body[0]) < 1e-4
    lo, hi = _bounds(glb)
    assert lo[1] == pytest.approx(0.0, abs=1e-5)
    assert (lo[0] + hi[0]) / 2 == pytest.approx(0.0, abs=1e-5)
    assert (lo[2] + hi[2]) / 2 == pytest.approx(0.0, abs=1e-5)


def test_normalize_glb_without_yaw_is_what_it_always_was(tmp_path):
    glb = tmp_path / "m.glb"
    _glb(glb)
    before = _bounds(glb)
    postprocess.normalize_glb(glb, None)
    lo, hi = _bounds(glb)
    assert (hi - lo) == pytest.approx(before[1] - before[0], abs=1e-5)
    assert lo[1] == pytest.approx(0.0, abs=1e-5)


def test_normalize_glb_ground_false_keeps_the_pivot_where_it_was(tmp_path):
    glb = tmp_path / "m.glb"
    _glb(glb)
    before = _bounds(glb)
    postprocess.normalize_glb(glb, None, ground=False)
    assert _bounds(glb) == pytest.approx(before, abs=1e-5)


def test_reorient_resizes_grounds_keeps_the_old_mesh_and_remeasures(svc):
    job_id = _job(svc)
    out = svc_jobs.reorient_model(svc, job_id, size_m=2.0)
    job_dir = svc.job_dir(job_id)
    lo, hi = _bounds(job_dir / "model.glb")
    assert float(max(hi - lo)) == pytest.approx(2.0, rel=1e-4)
    assert lo[1] == pytest.approx(0.0, abs=1e-5)
    params = svc.store.get(job_id)["params"]
    assert params["size_m"] == 2.0
    assert params["mesh_report"]["achieved_size_m"] == pytest.approx(2.0, rel=1e-3)
    (entry,) = params["model_history"]
    assert entry["kind"] == "reorient" and entry["geometry"] is True
    assert modelhistory.version_path(job_dir, entry["n"]).exists()
    assert out["ok"] and out["stale"] == []
    assert not (job_dir / ".reorient.glb").exists()


def test_reorient_model_never_writes_model_glb_in_place(svc, monkeypatch):
    job_id = _job(svc)
    model = svc.job_dir(job_id) / "model.glb"
    original = model.read_bytes()
    seen: dict = {}
    real = postprocess.normalize_glb

    def spy(path, *args, **kwargs):
        seen["path"] = path
        seen["model_at_call"] = model.read_bytes()
        return real(path, *args, **kwargs)

    monkeypatch.setattr(postprocess, "normalize_glb", spy)
    svc_jobs.reorient_model(svc, job_id, size_m=2.0)
    assert seen["path"] != model, "the normalise must run on a copy, not on the served name"
    assert seen["model_at_call"] == original
    assert model.read_bytes() != original


def test_a_failed_reorient_leaves_the_mesh_the_history_and_no_scratch(svc, monkeypatch):
    job_id = _job(svc)
    job_dir = svc.job_dir(job_id)
    original = (job_dir / "model.glb").read_bytes()

    def boom(*_a, **_k):
        raise RuntimeError("trimesh could not parse the mesh")

    monkeypatch.setattr(postprocess, "normalize_glb", boom)
    with pytest.raises(Failed):
        svc_jobs.reorient_model(svc, job_id, size_m=2.0)
    assert (job_dir / "model.glb").read_bytes() == original
    assert not (job_dir / ".reorient.glb").exists()
    assert not svc.store.get(job_id)["params"].get("model_history")
    assert not list((job_dir / "versions").glob("*")) if (job_dir / "versions").exists() else True


def test_reorient_model_drops_every_derived_export(svc):
    job_id = _job(svc)
    job_dir = svc.job_dir(job_id)
    for name in DERIVED:
        (job_dir / name).write_bytes(b"describes the old mesh")
    svc_jobs.reorient_model(svc, job_id, size_m=2.0)
    assert [n for n in DERIVED if (job_dir / n).exists()] == []


def test_bake_front_turns_the_mesh_and_clears_the_front(svc):
    job_id = _job(svc)
    svc_jobs.set_front_yaw(svc, job_id, 90.0)
    svc_jobs.reorient_model(svc, job_id, bake_front=True)
    nub, body, _ = _nub_and_body(svc.job_dir(job_id) / "model.glb")
    assert nub[2] > body[2] + 0.4
    assert "front_yaw" not in svc.store.get(job_id)["params"]


def test_reorient_bake_front_is_refused_while_a_rig_exists(svc):
    job_id = _job(svc)
    job_dir = svc.job_dir(job_id)
    svc_jobs.set_front_yaw(svc, job_id, 90.0)
    (job_dir / "rig.glb").write_bytes(b"rig")
    (job_dir / "rig.json").write_text("{}")
    original = (job_dir / "model.glb").read_bytes()
    with pytest.raises(Invalid) as caught:
        svc_jobs.reorient_model(svc, job_id, bake_front=True)
    assert caught.value.field == "front_yaw"
    assert (job_dir / "model.glb").read_bytes() == original
    assert svc.store.get(job_id)["params"]["front_yaw"] == 90.0
    assert not svc.store.get(job_id)["params"].get("model_history")


def test_a_resize_on_a_rigged_asset_is_allowed_and_reports_the_rig_stale(svc):
    job_id = _job(svc)
    job_dir = svc.job_dir(job_id)
    (job_dir / "rig.glb").write_bytes(b"rig")
    (job_dir / "rig.json").write_text("{}")
    out = svc_jobs.reorient_model(svc, job_id, size_m=2.0)
    assert "rig.glb" in out["stale"]


def test_bake_front_without_a_front_is_refused_on_the_front_field(svc):
    job_id = _job(svc)
    with pytest.raises(Invalid) as caught:
        svc_jobs.reorient_model(svc, job_id, bake_front=True)
    assert caught.value.field == "front_yaw"


@pytest.mark.parametrize("bad", [0.0, 0.001, 500.0, "wide"])
def test_a_size_outside_the_guidance_range_is_refused_on_size_m(svc, bad):
    job_id = _job(svc)
    with pytest.raises(Invalid) as caught:
        svc_jobs.reorient_model(svc, job_id, size_m=bad)
    assert caught.value.field == "size_m"


def test_asking_for_nothing_is_refused(svc):
    job_id = _job(svc)
    with pytest.raises(Invalid):
        svc_jobs.reorient_model(svc, job_id, ground=False)


@pytest.mark.parametrize("status", ["queued", "running", "cancelled"])
def test_an_unfinished_or_cancelled_job_is_refused(svc, status):
    job_id = _job(svc)
    svc.store.set_status(job_id, status)
    with pytest.raises(Conflict):
        svc_jobs.reorient_model(svc, job_id, size_m=2.0)


def test_a_job_with_a_dependent_still_running_is_refused(svc):
    job_id = _job(svc)
    child = svc.store.create("retexture", "x", {"source_job": job_id}, "child0000001")
    svc.store.set_status(child, "running")
    with pytest.raises(Conflict):
        svc_jobs.reorient_model(svc, job_id, size_m=2.0)


def test_revert_across_a_reorient_restores_size_m_and_the_front(svc):
    """A restore puts the mesh's whole description back. The bake turned the mesh
    and cleared the front, so going back across it must bring the front with it
    -- and restoring forward again must clear it again, or the next sheet renders
    a mesh that already faces front from an angle that is no longer true."""
    job_id = _job(svc)
    job_dir = svc.job_dir(job_id)
    svc_jobs.set_front_yaw(svc, job_id, 90.0)
    original_nub = _nub_and_body(job_dir / "model.glb")
    svc_jobs.reorient_model(svc, job_id, size_m=2.0, bake_front=True)
    params = svc.store.get(job_id)["params"]
    assert params["size_m"] == 2.0 and "front_yaw" not in params

    svc_jobs.revert_model(svc, job_id, version=1)
    params = svc.store.get(job_id)["params"]
    assert params["size_m"] == 1.3
    assert params["front_yaw"] == 90.0
    nub, body, _ = _nub_and_body(job_dir / "model.glb")
    assert nub[0] > body[0] + 0.1, "the +X nub is back on +X"
    assert nub == pytest.approx(original_nub[0], abs=1e-5)

    forward = max(e["n"] for e in params["model_history"])
    svc_jobs.revert_model(svc, job_id, version=forward)
    params = svc.store.get(job_id)["params"]
    assert params["size_m"] == 2.0
    assert "front_yaw" not in params


def test_a_resize_only_reorient_leaves_front_yaw_alone_on_revert(svc):
    job_id = _job(svc)
    svc_jobs.set_front_yaw(svc, job_id, 90.0)
    svc_jobs.reorient_model(svc, job_id, size_m=2.0)
    assert svc.store.get(job_id)["params"]["front_yaw"] == 90.0
    svc_jobs.set_front_yaw(svc, job_id, 45.0)
    svc_jobs.revert_model(svc, job_id, version=1)
    params = svc.store.get(job_id)["params"]
    assert params["size_m"] == 1.3
    assert params["front_yaw"] == 45.0, "nothing turned, so the front the user set stays"


def test_history_entries_only_carry_a_front_when_the_geometry_was_turned(svc):
    job_id = _job(svc)
    svc_jobs.reorient_model(svc, job_id, size_m=2.0)
    svc_jobs.set_front_yaw(svc, job_id, 30.0)
    svc_jobs.reorient_model(svc, job_id, bake_front=True)
    entries = svc.store.get(job_id)["params"]["model_history"]
    assert "front_yaw" not in entries[0]
    assert entries[1]["front_yaw"] == 30.0
    assert modelhistory.front_yaw_restored(entries, 1) == 30.0
    assert modelhistory.front_yaw_restored(entries, 2) == 30.0
    assert np.isclose(modelhistory.front_yaw_restored(entries[:1], 1) or 0.0, 0.0)
