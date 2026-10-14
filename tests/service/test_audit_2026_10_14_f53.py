"""Regression tests for fixer F53's findings of the 2026-10-14 audit.

Each test's name is the claim, and each one fails against the code it was
written for. Source is located through ``realmspinner.__file__`` rather than the
repo layout so the same file can be pointed at a pre-fix copy of the package.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import realmspinner
from realmspinner import doctor, retired
from realmspinner.kernels.rig import store
from realmspinner.pipelines import blender_run
from realmspinner.service import Invalid, TooLarge
from realmspinner.service import jobs as svc_jobs
from realmspinner.service import rig as svc_rig

SRC = Path(realmspinner.__file__).resolve().parent


def _text(rel: str) -> str:
    return (SRC / rel).read_text("utf-8")


# --- poser-poses-03: a pose saved for another skeleton ---------------------------

BONES = ["hips", "spine", "head"]
OTHER_SKELETON = ["root", "tail", "foreleg_l"]
IDENTITY = [0.0, 0.0, 0.0, 1.0]


def _rigged_job_with_a_pose(svc) -> tuple[str, str]:
    job_id = svc_jobs.create_job(svc, kind="text", prompt="a knight")["id"]
    job_dir = svc.config.data_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"fake-glb")
    (job_dir / "rig.glb").write_bytes(b"fake-rig")
    (job_dir / "rig.json").write_text(
        json.dumps({"version": 1, "bones": [{"name": n} for n in BONES]})
    )
    svc.store.set_status(job_id, "done")
    record = svc_rig.save_pose(
        svc, job_id, {"name": "idle", "bones": {b: IDENTITY for b in BONES}}
    )
    return job_id, record["id"]


def _rerig_onto_another_skeleton(svc, job_id: str) -> None:
    (svc.job_dir(job_id) / "rig.json").write_text(
        json.dumps({"version": 1, "bones": [{"name": n} for n in OTHER_SKELETON]})
    )


def _bake_that_would_succeed(monkeypatch) -> list:
    """A Blender that is installed and happily bakes: the stale pose must be
    refused by the service, not by luck of the worker failing."""
    calls: list = []
    monkeypatch.setattr(
        doctor, "blender_check", lambda **_kw: doctor.Check("Blender", True, "ok", fatal=False)
    )

    def run_worker(spec, **kwargs):
        calls.append(spec)
        Path(spec["out_glb"]).write_bytes(b"posed-glb")
        return {"ok": True, "bones": len(spec["bones"]), "unknown": list(spec["bones"])}

    monkeypatch.setattr(blender_run, "run_worker", run_worker)
    return calls


def test_posed_model_refuses_a_pose_that_names_bones_the_current_rig_lacks(svc, monkeypatch):
    job_id, pose_id = _rigged_job_with_a_pose(svc)
    _rerig_onto_another_skeleton(svc, job_id)
    calls = _bake_that_would_succeed(monkeypatch)

    with pytest.raises(Invalid) as excinfo:
        svc_rig.posed_model(svc, job_id, pose_id)

    assert excinfo.value.field == "pose_id"
    assert "different skeleton" in excinfo.value.message
    assert calls == [], "Blender was run for a pose its rig cannot take"
    assert not store.pose_glb_path(svc.job_dir(job_id), pose_id).exists()


def test_list_poses_omits_a_pose_that_names_bones_the_current_rig_lacks(svc):
    job_id, pose_id = _rigged_job_with_a_pose(svc)
    assert [p["id"] for p in svc_rig.list_poses(svc, job_id)["poses"]] == [pose_id]
    _rerig_onto_another_skeleton(svc, job_id)
    listed = svc_rig.list_poses(svc, job_id)
    assert listed["bones"] == OTHER_SKELETON
    assert listed["poses"] == []


def test_a_pose_naming_a_subset_of_the_current_rig_is_still_applied(svc, monkeypatch):
    """A pose need not set every bone; only a name the rig lacks is the defect."""
    job_id, _ = _rigged_job_with_a_pose(svc)
    partial = svc_rig.save_pose(svc, job_id, {"name": "nod", "bones": {"head": IDENTITY}})
    calls = _bake_that_would_succeed(monkeypatch)
    assert svc_rig.posed_model(svc, job_id, partial["id"]).read_bytes() == b"posed-glb"
    assert len(calls) == 1


# --- docs-E-17: comments that name a removed agent surface -----------------------


def test_service_comments_name_no_removed_agent_surface():
    rig = _text("service/rig.py")
    assert "two service workers" not in rig
    assert "AgentHost" not in rig
    assert "agent host" not in rig.lower()
    assert "an agent adds rigs" not in rig


# --- poser-jobs-04 / poser-jobs-05 ------------------------------------------------

_DELETED_SHEET_NAMES = (
    "create_pixel_sheet",
    "create_sheet",
    "PIXEL_LOGICAL_SIZES",
    "_restyle_in_flight",
    "sheet_pixel_png",
    "troupe._check_options",
    "pixel-sheet restyle",
    "pixel-sheet bands",
    "pixel-sheet writer",
    "sibling pixel-sheet door",
)


def test_no_comment_in_the_sprite_modules_names_a_deleted_sheet_function():
    for rel in ("service/sprites.py", "_q_sprite.py"):
        text = _text(rel)
        for name in _DELETED_SHEET_NAMES:
            assert name not in text, f"{rel} still cites {name!r}"


def test_sprites_exports_no_constant_nothing_imports():
    from realmspinner.service import sprites

    assert not hasattr(sprites, "SPRITE_SHEET_TYPES")
    for path in SRC.rglob("*.py"):
        assert "SPRITE_SHEET_TYPES" not in path.read_text("utf-8", errors="replace"), path


# --- plotter-tiles-07 / plotter-tiles-08 ------------------------------------------


def test_create_tile_sheet_docstring_says_the_prompt_only_names_a_seamless_sheet():
    from realmspinner.service import tilesheets

    doc = " ".join((tilesheets.create_tile_sheet.__doc__ or "").split())
    assert "is the **style sentence**" not in doc
    assert "the words every cell shares" not in doc.split("(the 2026-10-14 audit")[0]
    assert "only names the sheet" in doc


def test_an_oversized_tile_sheet_reference_names_the_reference_field(svc):
    from realmspinner.service import tilesheets
    from realmspinner.service.validation import MAX_UPLOAD_BYTES

    # The grid mode is what the sibling tests drive: it is the path that reaches
    # the reference with no other weights or lists in the way.
    with pytest.raises(TooLarge) as excinfo:
        tilesheets.create_tile_sheet(
            svc,
            prompt="mossy dungeon",
            tile_size=32,
            view="top_down",
            mode=tilesheets.MODE_GRID,
            allow_grid=True,
            reference=b"x" * (MAX_UPLOAD_BYTES + 1),
        )
    assert excinfo.value.field == "reference"


# --- docs-E-16: the parameter ledger ------------------------------------------------


def _has_a_writer(key: str) -> bool:
    pattern = re.compile(r"[\"']" + re.escape(key) + r"[\"']")
    for path in SRC.rglob("*.py"):
        if path.name == "validation.py":
            continue
        if pattern.search(path.read_text("utf-8", errors="replace")):
            return True
    return False


def test_every_derived_param_has_a_writer_or_a_legacy_note():
    from realmspinner.service import validation

    legacy = validation.LEGACY_DERIVED_PARAMS
    # Nothing outside the tuple, and every legacy key is a key the tuple strips.
    assert legacy <= set(validation.DERIVED_PARAMS)
    for key in validation.DERIVED_PARAMS:
        if key in legacy:
            assert not _has_a_writer(key), f"{key} has a writer; it is not legacy"
        else:
            assert _has_a_writer(key), f"{key} is stripped but nothing writes it"
    # The comments must not name inputs that exist nowhere in the package.
    text = _text("service/validation.py")
    assert "base_sheet" not in text
    assert "automatic rig/sheet insert" not in text
    assert "collision, sheet --" not in text


# --- service-gates-03: a mesh grade on a row that is not a reconstruction ----------

_NOT_RECONSTRUCTIONS = (
    "rig",
    "remesh",
    "retexture",
    "sprite_synthesis",
    "separate",
    "lora_train",
    *sorted(retired.RETIRED_KINDS),
)


@pytest.mark.parametrize("kind", _NOT_RECONSTRUCTIONS)
def test_record_verdict_refuses_a_mesh_grade_for_a_rig_remesh_separate_or_retired_row(svc, kind):
    from realmspinner.service import verdicts

    job_id = svc.store.create(kind, "x", {"seed": 1}, stage="model", status="done")
    with pytest.raises(Invalid) as excinfo:
        verdicts.record_verdict(svc, job_id, grade=4)
    assert kind in excinfo.value.message
    assert [v for v in svc.store.latest_verdicts() if v["job_id"] == job_id] == []


@pytest.mark.parametrize("kind", ["text", "image"])
def test_record_verdict_still_grades_a_reconstruction(svc, kind):
    from realmspinner.service import verdicts

    job_id = svc.store.create(kind, "a barrel", {"seed": 1}, stage="model", status="done")
    assert verdicts.record_verdict(svc, job_id, grade=4)["grade"] == 4


def test_the_unverdicted_list_and_the_verdict_door_share_one_kind_list():
    from realmspinner import db
    from realmspinner.service import verdicts

    assert tuple(db.RECONSTRUCTION_KINDS) == ("text", "image")
    assert verdicts.RECONSTRUCTION_KINDS is db.RECONSTRUCTION_KINDS
