"""Regression tests for the 2026-10-11 audit's generation / service / sweep
findings (create-02, -09, -13, -17, -18, -19).

Each test name is the claim, and each failed against the code it was written
for.
"""

from __future__ import annotations

import asyncio
import pathlib

import pytest

from realmspinner import generation, models, provenance, sweep
from realmspinner.config import Config
from realmspinner.service import Invalid
from realmspinner.service import jobs as svc_jobs

# --- create-02: promoting a reference made on a retired model -------------------


def _stored_reference(svc, **params):
    """A finished reference-stage row whose params name a retired model.

    ``create_job`` would refuse these names, so the row is written the way an
    older build left it: straight into the store.
    """
    ref_id = svc.store.create(
        "text", "a barrel", {"seed": 1, **params}, stage="reference"
    )
    job_dir = svc.job_dir(ref_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input.png").write_bytes(b"png")
    svc.store.set_status(ref_id, "done")
    return ref_id


def test_promote_of_a_reference_made_on_a_retired_model_is_allowed(svc):
    """A promotion never runs the image model, so a retired ``base_model`` is
    history on it, exactly as it already is for a remesh. The 2026-10-11 audit,
    finding create-02: ``guidance.normalize`` re-validated the inherited name and
    refused Make 3D with "pick another image model", a control the Mesh column
    does not have, so every reference drawn on a removed model was stuck at the
    Reference stage."""
    assert models.retired_choice("base_model", "juggernaut") is not None
    ref_id = _stored_reference(svc, base_model="juggernaut")
    out = svc_jobs.promote_to_model(svc, ref_id)
    child = svc.store.get(out["id"])
    assert child["stage"] == "model" and child["parent_id"] == ref_id
    # Kept as history rather than replaced by the registry default: a mesh row
    # saying "sdxl_cfg" would claim a checkpoint the reference never used.
    assert child["params"]["base_model"] == "juggernaut"


def test_promote_of_a_reference_with_a_retired_style_is_allowed(svc):
    assert models.retired_choice("style_lora", "pixelklein") is not None
    ref_id = _stored_reference(svc, style_lora="pixelklein", lora_weight=0.8)
    out = svc_jobs.promote_to_model(svc, ref_id)
    params = svc.store.get(out["id"])["params"]
    assert params["style_lora"] == "pixelklein"
    assert params["lora_weight"] == 0.8


def test_promote_candidates_of_a_reference_on_a_retired_model_is_allowed(svc):
    ref_id = _stored_reference(svc, base_model="juggernaut")
    out = svc_jobs.promote_candidates(svc, ref_id, count=2)
    assert len(out["ids"]) == 2


def test_promote_still_refuses_an_unknown_model_name(svc):
    """Only a *retired* name is history; a typo is still a refusal."""
    ref_id = _stored_reference(svc, base_model="not-a-model-anyone-had")
    with pytest.raises(Invalid):
        svc_jobs.promote_to_model(svc, ref_id)


# --- create-09: a structure_control the structured door never forwards ----------


def _structure_request(**over):
    base = {
        "generation_type": "image",
        "prompt": "a wooden crate",
        "structure_control": "canny",
        "references": ("ref.png",),
        "reference_mode": "single",
    }
    base.update(over)
    return generation.GenerationRequest(**base)


def test_validate_request_refuses_a_structure_control_the_structured_door_does_not_forward():
    """The 2026-10-11 audit, finding create-09: ``request_to_legacy`` never emits
    the control, so a request naming one passed validation, was recorded in
    ``params["generation_request"]`` as ControlNet-conditioned and ran without
    one."""
    request = _structure_request()
    resolved = generation.resolve_recipe(request, None)
    issues = generation.validate_request(request, resolved, structure_forwarded=False)
    assert [i.field for i in issues] == ["structure_control"]


def test_validate_request_refuses_a_structure_control_with_no_reference():
    request = _structure_request(references=(), reference_mode="none")
    resolved = generation.resolve_recipe(request, None)
    issues = generation.validate_request(request, resolved)
    assert [i.field for i in issues] == ["structure_control"]


def test_the_pane_path_still_accepts_a_structure_control_with_a_reference():
    """The pane forwards ``control`` itself through ``create_job``'s guidance
    fields, so the default (forwarded) validation must keep passing it."""
    request = _structure_request()
    resolved = generation.resolve_recipe(request, None)
    assert generation.validate_request(request, resolved) == []


def test_create_generation_request_refuses_a_structure_control_on_the_structure_field(svc):
    request = _structure_request()
    with pytest.raises(Invalid) as info:
        svc_jobs.create_generation_request(svc, request)
    assert info.value.field == "structure_control"
    assert svc.store.list(limit=10) == []


# --- create-13: the sweep's matte --------------------------------------------------


class _FakeServer:
    """TrellisServer without the exe: records what ``generate`` was asked."""

    calls: list[dict] = []

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def generate(self, image, out, **kwargs):
        type(self).calls.append(kwargs)
        out.write_bytes(b"glb")

    def stop(self) -> None:
        pass


@pytest.fixture
def fake_sweep(monkeypatch, tmp_path):
    from realmspinner import meshaudit

    _FakeServer.calls = []
    monkeypatch.setattr(sweep, "TrellisServer", _FakeServer)
    monkeypatch.setattr(
        meshaudit,
        "hole_fraction",
        lambda *a, **k: {"worst": 0.1, "mean": 0.05, "faces": 1000},
    )
    models_dir = tmp_path / "trellis-models"
    models_dir.mkdir()
    config = Config(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_models_dir=models_dir,
    )
    return config, models_dir, tmp_path


def test_sweep_generates_with_the_matte_a_real_job_would_use(fake_sweep):
    """The 2026-10-11 audit, finding create-13: ``run_band`` sent no
    ``bg_removal``, so every band ran under the exe's own default while a real
    job gets birefnet wherever its weights are present -- and the closing line
    tells a human to hard-code a band from those numbers."""
    config, models_dir, tmp_path = fake_sweep
    (models_dir / "birefnet.gguf").write_bytes(b"weights")
    row = asyncio.run(
        sweep.run_band(
            config, 4, tmp_path / "ref.png", tmp_path,
            seed=1, resolution=512, audit_resolution=64,
        )
    )
    assert _FakeServer.calls[-1]["bg_removal"] == "birefnet"
    assert row["matte"] == "birefnet"


def test_sweep_falls_back_to_auto_when_the_matte_weights_are_absent(fake_sweep):
    config, _models_dir, tmp_path = fake_sweep
    row = asyncio.run(
        sweep.run_band(
            config, 4, tmp_path / "ref.png", tmp_path,
            seed=1, resolution=512, audit_resolution=64,
        )
    )
    assert _FakeServer.calls[-1]["bg_removal"] == "auto"
    assert row["matte"] == "auto"


def test_the_sweep_table_header_names_the_matte(capsys):
    rows = [
        {"band": "auto", "worst": 0.3, "mean": 0.1, "faces": 1000, "seconds": 1.0,
         "matte": "birefnet"},
    ]
    sweep.print_table(rows, 1024)
    assert "matte birefnet" in capsys.readouterr().out


# --- create-17: the quality refusal -------------------------------------------------


def test_a_quality_refusal_does_not_name_a_tier_the_registry_lacks():
    """The 2026-10-11 audit, finding create-17: "Quality must be Fast or
    Quality." named a Fast tier the shipped registry has not had since the
    distilled models were retired."""
    offered = {r.quality for r in generation.RECIPES}
    assert "fast" not in offered
    request = generation.GenerationRequest(
        generation_type="image", prompt="a crate", quality="bogus"
    )
    issues = [i for i in generation.validate_request(request, None) if i.field == "quality"]
    assert len(issues) == 1
    assert "fast" not in issues[0].message.lower()


def test_a_stored_fast_request_still_loads_as_quality():
    """The migration that keeps old rows loading must survive the rewording."""
    request = generation.GenerationRequest.from_dict(
        {"generation_type": "image", "prompt": "a crate", "quality": "fast"}
    )
    assert request.quality == "quality"


# --- create-18: file_fingerprint on a locked file --------------------------------------


def test_file_fingerprint_reports_an_unreadable_file_instead_of_raising(tmp_path, monkeypatch):
    """The 2026-10-11 audit, finding create-18: a sharing violation or an
    antivirus lock on a checkpoint raised out of a provenance-only read and
    failed a two-minute GPU job before it started."""
    target = tmp_path / "locked.gguf"
    target.write_bytes(b"x" * 64)
    real_open = pathlib.Path.open

    def locked(self, *args, **kwargs):
        if self == target:
            raise PermissionError(13, "The process cannot access the file")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "open", locked)
    assert provenance.file_fingerprint(target) == "unreadable"
    monkeypatch.setattr(pathlib.Path, "open", real_open)
    # Uncached: once the lock is gone the real fingerprint comes back.
    assert provenance.file_fingerprint(target).startswith("s64:")


# --- create-19: parse_bands upper bound --------------------------------------------------


def test_parse_bands_refuses_a_band_above_the_doors_maximum():
    """The 2026-10-11 audit, finding create-19: ``--bands 2,99999999`` was
    accepted and failed inside the exe minutes into the run, while the service
    door bounds the same flag at ``MAX_TRELLIS_BAND``."""
    from realmspinner.service import validation

    top = validation.MAX_TRELLIS_BAND
    assert sweep.parse_bands(str(top)) == [top]
    with pytest.raises(ValueError, match=str(top + 1)):
        sweep.parse_bands(f"auto,{top + 1}")
    with pytest.raises(ValueError):
        sweep.parse_bands("2,99999999")
