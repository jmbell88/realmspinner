"""The Game-ready budget: the engine finishes it, and a gate checks whatever did.

M0 (``dev/measurements/2026-10-15-lowpoly-silhouette-gate.md``): the Blender
preserve path asked for 5,000 triangles and shipped 13,813 with a fifth of the
silhouette gone. The engine's ``--faces`` simplifies *before* its own UV and PBR
bake, so the texture lands on the mesh that ships and Blender is not involved.
These tests drive the worker with the conftest fake server and a patched
``engine_caps``; nothing here starts a real engine.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from realmspinner import doctor
from realmspinner.config import Config
from realmspinner.db import JobStore
from realmspinner.pipelines import remesh, trellis
from realmspinner.queue import Worker
from realmspinner.service import jobs as svc_jobs
from realmspinner.service.errors import Invalid


async def _wait_until(predicate, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    pytest.fail("condition not met before timeout")


@pytest.fixture
def worker(tmp_path, fake_pipelines):
    config = Config(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "models",
        # Pinned absent: its default is the vendored binary, which this machine
        # has and CI does not, and the ladder's last rung would otherwise run it.
        gltfpack_exe=tmp_path / "no-gltfpack.exe",
    )
    store = JobStore(config.db_path)
    w = Worker(config, store)
    yield w
    store.close()


@pytest.fixture
def engine_with_faces(monkeypatch):
    monkeypatch.setattr(
        trellis, "engine_caps", lambda exe, **kw: {**trellis.NO_CAPS, "faces": True}
    )


@pytest.fixture
def blender_calls(monkeypatch):
    """Blender stands in as a recorder, and writes a tiny 'remeshed' file."""
    import realmspinner.pipelines.blender_run as blender_run_mod
    from realmspinner import tiercheck
    from realmspinner.pipelines import postprocess

    calls: list[dict] = []

    def fake_run_worker(spec, **kwargs):
        calls.append(spec)
        from pathlib import Path

        Path(spec["out_glb"]).write_bytes(b"remeshed")
        # What was asked for, to the triangle: the gate would (rightly) send a fake
        # that always said 4,996 on to the next rung when the request was 1,000.
        return {
            "ok": True, "method": "decimate", "faces": spec["target_faces"] * 2,
            "texture_size": 1024,
        }

    monkeypatch.setattr(blender_run_mod, "run_worker", fake_run_worker)
    monkeypatch.setattr(postprocess, "normalize_glb", lambda *a, **k: {"scale": 1.0})
    monkeypatch.setattr(tiercheck, "compare", lambda before, after: tiercheck.Verdict(True, (), ()))
    return calls


def _job(worker: Worker, **params) -> str:
    job_id = worker.store.create("image", None, {"seed": 1, "resolution": 512, **params})
    job_dir = worker.config.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input.png").write_bytes(b"fake-png")
    return job_id


async def _run(worker: Worker, job_id: str) -> dict:
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()
    return worker.store.get(job_id)


async def test_engine_finishing_skips_blender_when_budget_met(
    worker, engine_with_faces, blender_calls
):
    # The fake reconstruction is a 12-triangle cube, comfortably inside 1.25x of 1000.
    job_id = _job(worker, mesh_finishing="engine", lowpoly_triangles=1000, profile="raw")
    row = await _run(worker, job_id)

    assert blender_calls == []
    assert worker.trellis.generate_calls[0]["faces"] == 1000
    lowpoly = row["params"]["lowpoly"]
    assert lowpoly["status"] == "complete"
    assert lowpoly["method"] == "engine"
    assert lowpoly["requested"] == 1000
    assert lowpoly["achieved"] == 12
    # Nothing to compare against, and it says so rather than reporting a pass.
    assert lowpoly["geometry"]["measured"] is False
    assert row["params"]["engine_faces"] == 1000
    assert row["params"]["source_faces"] == 12


async def test_engine_finishing_falls_back_to_blender_when_engine_ignores_faces(
    worker, engine_with_faces, blender_calls
):
    # An exe that listed --faces and ignored the field returns its built-in mesh.
    # The fake's 12 triangles against a budget of 8 is 1.5x: past the slack.
    assert 8 * remesh.BUDGET_SLACK < 12
    job_id = _job(worker, mesh_finishing="engine", lowpoly_triangles=8, profile="raw")
    row = await _run(worker, job_id)

    assert len(blender_calls) == 1
    spec = blender_calls[0]
    # The first Blender rung, not the hole-closing one.
    assert spec["preserve_shape"] is True and spec["close_holes"] is False
    lowpoly = row["params"]["lowpoly"]
    assert lowpoly["finishing"] == "preserve_shape"
    assert lowpoly["method"] == "decimate"
    # What source.glb really holds is still recorded: it is the truth the retarget
    # door checks against.
    assert row["params"]["source_faces"] == 12


async def test_engine_finishing_without_faces_support_is_the_blender_path(worker, blender_calls):
    # No patched engine_caps: the configured exe does not exist, so nothing is
    # supported, and the request must not carry a field the exe cannot take.
    job_id = _job(worker, mesh_finishing="engine", lowpoly_triangles=1000, profile="raw")
    row = await _run(worker, job_id)

    assert worker.trellis.generate_calls[0]["faces"] is None
    assert row["params"]["engine_faces"] is None
    assert len(blender_calls) == 1
    assert row["params"]["lowpoly"]["finishing"] == "preserve_shape"


async def test_a_pinned_decim_keeps_the_exes_own_simplifier_and_sends_no_faces(
    worker, engine_with_faces, blender_calls
):
    """The exe refuses --faces together with a launch-time --decim (a 400)."""
    job_id = _job(
        worker, mesh_finishing="engine", lowpoly_triangles=1000, profile="raw", trellis_decim=200000
    )
    await _run(worker, job_id)
    assert worker.trellis.generate_calls[0]["faces"] is None


async def test_a_stored_row_with_no_finishing_key_still_means_repair(worker, blender_calls):
    job_id = _job(worker, lowpoly_triangles=1000, profile="raw")
    await _run(worker, job_id)
    assert blender_calls[0]["close_holes"] is True
    assert blender_calls[0]["preserve_shape"] is False


# --- the doors ---------------------------------------------------------------


@pytest.fixture
def blender_absent(monkeypatch):
    class _No:
        ok = False

    monkeypatch.setattr(doctor, "blender_check", lambda *a, **k: _No())


def _create(svc, **kwargs):
    out = svc_jobs.create_job(svc, kind="text", prompt="a barrel", output="model", **kwargs)
    return svc.store.get(out["id"])["params"]


def test_new_jobs_default_to_engine_finishing(svc, blender_absent, engine_with_faces):
    assert _create(svc)["mesh_finishing"] == "engine"


def test_default_budget_needs_no_blender_with_engine_faces(
    svc, blender_absent, engine_with_faces
):
    params = _create(svc)
    assert params["lowpoly_triangles"] == 5000
    assert params["profile"] == "raw"


def test_an_explicit_budget_needs_no_blender_with_engine_faces(
    svc, blender_absent, engine_with_faces
):
    assert _create(svc, lowpoly_triangles=10_000)["lowpoly_triangles"] == 10_000


def test_without_engine_faces_and_without_blender_the_budget_is_still_refused(svc, blender_absent):
    with pytest.raises(Invalid) as info:
        _create(svc, lowpoly_triangles=5000)
    assert info.value.field == "lowpoly_triangles"


def test_a_blender_finishing_still_needs_blender_even_when_the_engine_has_faces(
    svc, blender_absent, engine_with_faces
):
    with pytest.raises(Invalid) as info:
        _create(svc, lowpoly_triangles=5000, mesh_finishing="repair")
    assert info.value.field == "lowpoly_triangles"


def _done_job(svc, **params) -> str:
    job_id = svc.store.create("text", "a barrel", {"output": "model", **params})
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    import trimesh

    glb = trimesh.creation.box().export(file_type="glb")
    (job_dir / "source.glb").write_bytes(glb)
    (job_dir / "model.glb").write_bytes(glb)
    svc.store.set_stage(job_id, "model")
    svc.store.set_status(job_id, "done")
    return job_id


def test_retarget_above_source_faces_is_refused_with_field(svc, tmp_path):
    # The svc fixture pins gltfpack absent, and that refusal comes first; a stand-in
    # file lets the request reach the check under test.
    svc.config.gltfpack_exe = tmp_path / "gltfpack.exe"
    svc.config.gltfpack_exe.write_bytes(b"x")
    job_id = _done_job(svc, source_faces=5000, mesh_finishing="engine", lowpoly_triangles=5000)
    with pytest.raises(Invalid) as info:
        svc_jobs.optimize_job(svc, job_id, profile="custom", custom_triangles=20_000)
    assert info.value.field == "custom_triangles"
    assert "5,000" in str(info.value)


def test_retarget_at_or_below_source_faces_is_not_refused_by_the_source_check(svc):
    job_id = _done_job(svc, source_faces=5000, mesh_finishing="engine", lowpoly_triangles=5000)
    # The same number the job was made at is not "above" itself, and raw is no budget.
    from realmspinner.service import _jobs_rework

    job = svc.require_job(job_id)
    _jobs_rework._refuse_above_source(job, 5000, what="retarget", field="profile")
    _jobs_rework._refuse_above_source(job, None, what="retarget", field="profile")
    _jobs_rework._refuse_above_source(job, 4000, what="retarget", field="profile")


def test_a_row_with_no_source_faces_is_never_refused(svc):
    from realmspinner.service import _jobs_rework

    job_id = _done_job(svc)
    _jobs_rework._refuse_above_source(
        svc.require_job(job_id), 200_000, what="retarget", field="profile"
    )


# --- the shape gate and the ladder ---------------------------------------------------

#: The stored failure (job cc976fa923a1, 2026-10-14): asked for 5,000, got 13,821, and
#: lost 19.9% of the silhouette in its worst view.
DEFORMED = {
    "measured": True, "worst_lost_coverage": 0.1986,
    "views": [{"iou": 0.795}, {"iou": 0.80}],
}
FAITHFUL = {
    "measured": True, "worst_lost_coverage": 0.024,
    "views": [{"iou": 0.966}, {"iou": 0.97}],
}


@pytest.fixture
def ladder(monkeypatch, blender_calls):
    """Scripted finishers: ``outcomes[rung] = (triangles, geometry)``.

    Blender's two rungs are told apart by the spec they are handed; the staged
    file carries the rung's name so a test can see which one shipped.
    """
    import realmspinner.pipelines.blender_run as blender_run_mod

    outcomes: dict[str, tuple[int, dict]] = {}
    seen: dict[str, dict] = {}

    def rung_of(spec):
        return "preserve_shape" if spec["preserve_shape"] else "repair"

    def fake_run_worker(spec, **kwargs):
        from pathlib import Path

        rung = rung_of(spec)
        blender_calls.append(spec)
        triangles, _ = outcomes[rung]
        Path(spec["out_glb"]).write_bytes(f"mesh-from-{rung}".encode())
        return {"ok": True, "method": "decimate", "faces": triangles, "texture_size": 1024}

    def fake_compare(source, finished, resolution=256):
        rung = Path(finished).read_bytes().decode().removeprefix("mesh-from-")
        seen[rung] = {}
        return outcomes[rung][1]

    from pathlib import Path

    monkeypatch.setattr(blender_run_mod, "run_worker", fake_run_worker)
    monkeypatch.setattr(remesh, "compare_geometry", fake_compare)
    return outcomes


async def test_lowpoly_rejects_mesh_losing_more_than_the_gate(worker, ladder, blender_calls):
    """The stored 5,000-triangle failure: preserve_shape lands inside 1.25x of the
    budget here but loses a fifth of the silhouette, so it must not ship."""
    ladder["preserve_shape"] = (5000, DEFORMED)
    ladder["repair"] = (4995, FAITHFUL)
    job_id = _job(worker, mesh_finishing="preserve_shape", lowpoly_triangles=5000, profile="raw")
    row = await _run(worker, job_id)

    assert [spec["preserve_shape"] for spec in blender_calls] == [True, False]
    lowpoly = row["params"]["lowpoly"]
    assert lowpoly["finishing"] == "repair"
    assert lowpoly["achieved"] == 4995
    assert lowpoly["status"] == "complete"
    assert [a["rung"] for a in lowpoly["attempts"]] == ["preserve_shape", "repair"]
    assert "budget_miss" not in row["params"]
    assert (worker.config.job_dir(job_id) / "model.glb").read_bytes() != b"mesh-from-preserve_shape"


async def test_the_ladder_stops_at_the_first_finisher_that_passes(worker, ladder, blender_calls):
    ladder["preserve_shape"] = (4990, FAITHFUL)
    ladder["repair"] = (4995, FAITHFUL)
    job_id = _job(worker, mesh_finishing="preserve_shape", lowpoly_triangles=5000, profile="raw")
    row = await _run(worker, job_id)
    assert len(blender_calls) == 1
    assert row["params"]["lowpoly"]["finishing"] == "preserve_shape"


async def test_rejected_lowpoly_never_ships_raw_without_budget_miss(worker, ladder):
    """No finisher passes: the closest ships, and the row says it missed."""
    ladder["preserve_shape"] = (13821, DEFORMED)
    ladder["repair"] = (5000, {**DEFORMED, "worst_lost_coverage": 0.07, "views": [{"iou": 0.92}]})
    job_id = _job(worker, mesh_finishing="preserve_shape", lowpoly_triangles=5000, profile="raw")
    row = await _run(worker, job_id)

    params = row["params"]
    miss = params["budget_miss"]
    assert miss["requested"] == 5000
    # Repair missed on the silhouette alone; preserve on both, by far more.
    assert miss["achieved"] == 5000
    assert "7.0% of the silhouette" in miss["reason"]
    assert "lowpoly" in params["degraded"]
    assert params["lowpoly"]["finishing"] == "repair"
    assert (worker.config.job_dir(job_id) / "model.glb").read_bytes() != b"fake-glb"


async def test_over_budget_achieved_is_not_status_complete(worker, ladder):
    """Asked for 5,000, shipped 13,821 (the stored failure's own count): 2.8x is past
    1.5x, so the record says ``over_budget`` and a Complete badge cannot be earned."""
    ladder["repair"] = (13821, FAITHFUL)
    job_id = _job(worker, mesh_finishing="repair", lowpoly_triangles=5000, profile="raw")
    row = await _run(worker, job_id)
    lowpoly = row["params"]["lowpoly"]
    assert lowpoly["status"] == "over_budget"
    assert lowpoly["status"] != "complete"
    assert row["params"]["budget_miss"]["achieved"] == 13821


async def test_a_mesh_within_the_over_budget_line_but_off_its_gate_still_says_complete_with_a_miss(
    worker, ladder
):
    """1.3x is outside the 1.25x gate and inside 1.5x: shipped, flagged, not over_budget."""
    ladder["repair"] = (6500, FAITHFUL)
    job_id = _job(worker, mesh_finishing="repair", lowpoly_triangles=5000, profile="raw")
    row = await _run(worker, job_id)
    assert row["params"]["lowpoly"]["status"] == "complete"
    assert row["params"]["budget_miss"]["achieved"] == 6500


def test_closest_is_the_smallest_worst_miss_not_the_least_silhouette_loss():
    """From the M0 sweep (5k request, one reconstruction): the preserve floor is 13,813
    triangles and loses 19.9%; gltfpack is 53,894 and loses 2.5%. Neither passes. The
    gltfpack mesh has the smaller loss and is ten times the budget -- the floor mesh is
    the closer miss, and it is the one that must ship."""
    floor = remesh.gate(5000, 13813, DEFORMED)[0]
    gltfpack = remesh.gate(5000, 53894, {**FAITHFUL, "worst_lost_coverage": 0.025})[0]
    assert 1.0 < floor < gltfpack


def test_the_gate_passes_every_measured_good_row_and_fails_every_deformed_one():
    """The sweep's own rows (``dev/measurements/2026-10-15-lowpoly-silhouette-gate.md``)."""
    good = [(5000, 4995, 0.024, 0.966), (10000, 9995, 0.017, 0.976), (20000, 19957, 0.032, 0.96),
            (2000, 1995, 0.044, 0.943)]
    bad = [(5000, 13813, 0.199, 0.795), (5000, 7947, 0.268, 0.723),
           (10000, 9989, 0.073, 0.919), (5000, 46410, 0.025, 0.973)]
    for requested, achieved, lost, iou in good:
        geometry = {"measured": True, "worst_lost_coverage": lost, "views": [{"iou": iou}]}
        assert remesh.gate(requested, achieved, geometry)[0] <= 1.0, (requested, achieved)
    for requested, achieved, lost, iou in bad:
        geometry = {"measured": True, "worst_lost_coverage": lost, "views": [{"iou": iou}]}
        assert remesh.gate(requested, achieved, geometry)[0] > 1.0, (requested, achieved)


def test_unmeasured_geometry_does_not_fail_the_shape_half_of_the_gate():
    miss, reasons = remesh.gate(5000, 4990, {"measured": False, "error": "empty silhouette"})
    assert miss <= 1.0 and reasons == []
    assert remesh.gate(5000, 13821, {"measured": False})[0] > 1.0


def test_the_ladder_starts_where_the_finishing_says_and_ends_at_gltfpack():
    assert remesh.finishing_ladder("preserve_shape") == ("preserve_shape", "repair", "gltfpack")
    assert remesh.finishing_ladder("repair") == ("repair", "gltfpack")
    assert remesh.finishing_ladder("engine") == ("preserve_shape", "repair", "gltfpack")


async def test_a_gltfpack_pass_that_changed_nothing_is_not_a_finisher(worker, ladder, monkeypatch):
    """Blender is down and the reconstruction is already inside the budget, so
    gltfpack 'succeeds' by copying it. That is the failure it is, not a Complete."""
    import realmspinner.pipelines.blender_run as blender_run_mod
    from realmspinner.pipelines import optimize

    worker.config.gltfpack_exe.write_bytes(b"stand-in")

    def dead_blender(spec, **kwargs):
        raise blender_run_mod.BlenderError("boom")

    monkeypatch.setattr(blender_run_mod, "run_worker", dead_blender)
    monkeypatch.setattr(
        optimize, "run", lambda *a, **k: {"achieved": 12, "source_triangles": 12}
    )
    job_id = _job(worker, mesh_finishing="repair", lowpoly_triangles=5000, profile="raw")
    row = await _run(worker, job_id)
    assert row["params"]["lowpoly"]["status"] == "failed"
    assert "lowpoly" in row["params"]["degraded"]
