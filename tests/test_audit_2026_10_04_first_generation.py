"""The 2026-10-04 audit's ``generation`` door findings (create-03, create-04,
create-15): the structured door dropped img2img, ``validate_request`` never
compared the reference mode with the reference count, and a LoRA import
copied onto its served name in place."""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pytest

from realmspinner import generation
from realmspinner.service import jobs as svc_jobs
from realmspinner.service import loras as svc_loras
from realmspinner.service.errors import Failed


def _png(path: Path) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGBA", (32, 32), (200, 50, 50, 255)).save(buf, "PNG")
    path.write_bytes(buf.getvalue())
    return str(path)


def _img2img_request(ref: str, **over):
    return generation.GenerationRequest(
        generation_type="image",
        prompt="a barrel",
        model_mode="advanced",
        model_override="sdxl_cfg",
        references=(ref,),
        reference_mode="single",
        init_image=True,
        init_strength=0.55,
        **over,
    )


# --- create-03: the door forwards what request_to_legacy emits ---------------


def test_create_generation_request_forwards_init_image_and_strength_to_create_job(
    svc, installed_recipes, tmp_path, monkeypatch
):
    """``request_to_legacy`` writes ``init_image`` and ``init_strength`` on the
    legacy payload, and the door then called ``create_job`` without either, so
    a request that validated as img2img ran as a plain reference job while
    ``params["generation_request"]`` went on recording img2img."""
    seen: dict = {}
    real = svc_jobs.create_job

    def spy(svc_, **kwargs):
        seen.update(kwargs)
        return real(svc_, **kwargs)

    monkeypatch.setattr(svc_jobs, "create_job", spy)
    made = svc_jobs.create_generation_request(svc, _img2img_request(_png(tmp_path / "r.png")))
    assert seen["init_image"] is True
    assert seen["init_strength"] == pytest.approx(0.55)
    params = svc.store.get(made["id"])["params"]
    assert params["init_image"] is True
    assert params["init_strength"] == pytest.approx(0.55)


#: How ``create_generation_request`` carries each key ``request_to_legacy``
#: emits on its ``create_job`` arm. A key emitted and not named here fails
#: ``test_the_door_accounts_for_every_field_request_to_legacy_emits``, which is
#: the point: the next field someone adds to the adapter has to be wired to
#: ``create_job`` (or declared carried some other way) before it can pass.
_CARRIED = {
    "asset_type": ("kwarg", "asset_type"),
    "asset_intent": ("kwarg", "asset_intent"),
    "output": ("kwarg", "output"),
    "prompt": ("kwarg", "prompt"),
    "negative_prompt": ("kwarg", "negative_prompt"),
    "seed": ("kwarg", "seed"),
    "count": ("kwarg", "count"),
    "lora_weight": ("kwarg", "lora_weight"),
    "init_image": ("kwarg", "init_image"),
    "init_strength": ("kwarg", "init_strength"),
    "style_lora": ("guidance", "style_lora"),
    "references": ("files", "native_reference_files"),
    # The sprite arm builds ``sprite_sheet`` from the request itself.
    "sprite_settings": ("kwarg", "sprite_sheet"),
    # The tileset arm never reaches ``create_job``; ``create_tile_sheet`` has
    # its own door and its own tests (``tests/service/test_tileset_service.py``).
    "projection": ("tileset", None),
    "tile_settings": ("tileset", None),
}


def test_the_door_accounts_for_every_field_request_to_legacy_emits(
    svc, installed_recipes, tmp_path, monkeypatch
):
    seen: dict = {}

    def spy(svc_, **kwargs):
        seen.update(kwargs)
        return {"id": "stub"}

    monkeypatch.setattr(svc_jobs, "create_job", spy)
    request = _img2img_request(
        _png(tmp_path / "r.png"),
        style_lora="render3d",
        lora_weight=0.7,
        negative_prompt="blurry",
        seed=7,
        count=2,
    )
    resolved = generation.resolve_recipe(request, svc.config)
    legacy = generation.request_to_legacy(request, resolved)
    # Everything the adapter can emit on the non-tileset arm, in one request.
    assert {"init_image", "init_strength", "style_lora", "references"} <= set(legacy)
    unaccounted = set(legacy) - set(_CARRIED)
    assert not unaccounted, f"request_to_legacy emits {sorted(unaccounted)} with no carrier"

    svc_jobs.create_generation_request(svc, request)
    for key, value in legacy.items():
        how, name = _CARRIED[key]
        if how == "kwarg":
            assert name in seen, f"{key!r} was dropped: create_job never got {name!r}"
            if key not in ("asset_type", "negative_prompt"):
                assert seen[name] == value, (key, seen[name], value)
        elif how == "guidance":
            assert seen["guidance_fields"].get(name) == value, key
        elif how == "files":
            assert seen["extra_params"].get(name), key
    # ``asset_type`` comes from the request itself and the negative prompt is
    # ``effective_negative_prompt`` of it; both compared exactly above in spirit.
    assert seen["asset_type"] == legacy["asset_type"]
    assert (seen["negative_prompt"] or None) == legacy["negative_prompt"]


# --- create-04: the reference mode agrees with the reference count -----------


def _mode_request(mode: str, count: int, **over):
    return generation.GenerationRequest(
        generation_type="image",
        prompt="a barrel",
        references=tuple(f"ref{i}.png" for i in range(count)),
        reference_mode=mode,
        **over,
    )


def _reference_issues(request):
    # With no resolved recipe ``validate_request`` also reports the missing
    # recipe; only what it says about the references is under test here.
    return [i for i in generation.validate_request(request) if i.field == "references"]


@pytest.mark.parametrize(
    ("mode", "count"),
    [("single", 3), ("none", 3), ("none", 1), ("single", 0)],
    ids=["single-three", "none-three", "none-one", "single-none"],
)
def test_validate_request_refuses_references_that_disagree_with_reference_mode(mode, count):
    """On an SDXL recipe only the first reference is drawn from, yet all were
    decoded, written as ``native_reference_N.png`` and recorded, so the request
    document claimed images the pixels never saw."""
    issues = _reference_issues(_mode_request(mode, count))
    assert len(issues) == 1, issues


@pytest.mark.parametrize(
    ("mode", "count"),
    [("none", 0), ("single", 1), ("multi", 2), ("multi", 5)],
)
def test_validate_request_accepts_references_that_agree_with_reference_mode(mode, count):
    assert _reference_issues(_mode_request(mode, count)) == []


def test_from_dict_infers_the_reference_mode_a_caller_left_out():
    """A dict naming references and no mode used to read as ``"none"``, which is
    now a refusal; the mode a caller did not say is the one the images imply."""
    base = {"generation_type": "image", "prompt": "a barrel"}
    one = generation.GenerationRequest.from_dict({**base, "references": ["a.png"]})
    assert one.reference_mode == "single"
    two = generation.GenerationRequest.from_dict({**base, "references": ["a.png", "b.png"]})
    assert two.reference_mode == "multi"
    assert generation.GenerationRequest.from_dict(base).reference_mode == "none"
    said = generation.GenerationRequest.from_dict(
        {**base, "references": ["a.png"], "reference_mode": "none"}
    )
    assert said.reference_mode == "none"


# --- create-15: a LoRA import is staged, and a stale blob costs nothing ------


@pytest.fixture
def lora_home(svc, tmp_path, monkeypatch):
    """A private loras folder and style registry: the session's shared home
    already holds other tests' imports, and a registered row outlives the test."""
    from realmspinner import models

    monkeypatch.setattr(svc.config, "t2i_model_root", tmp_path / "models")
    monkeypatch.setattr(models, "STYLE_LORAS", dict(models.STYLE_LORAS))
    return tmp_path / "models" / "loras"


def _safetensors(path: Path) -> Path:
    import json
    import struct

    header = json.dumps({"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0\0\0")
    return path


def test_import_lora_survives_a_stale_blob_that_cannot_be_unlinked(
    svc, lora_home, tmp_path, monkeypatch, caplog
):
    """Re-importing a file under a new label rewrites the manifest row, then
    unlinks the old blob. A locked old blob raised after the import had worked,
    skipping ``register_imported_loras`` and surfacing a bare OSError."""
    src = _safetensors(tmp_path / "style.safetensors")
    first = svc_loras.import_lora(svc, src, label="first")
    root = lora_home
    old_blob = root / first["filename"]
    assert old_blob.is_file()

    real_unlink = Path.unlink

    def locked(self, *args, **kwargs):
        if self.name == old_blob.name:
            raise PermissionError(32, "in use by another process")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", locked)
    with caplog.at_level(logging.WARNING):
        second = svc_loras.import_lora(svc, src, label="second")
    assert second["key"] == first["key"]
    assert second["filename"] != first["filename"]
    assert (root / second["filename"]).is_file()
    assert any(old_blob.name in r.getMessage() for r in caplog.records)
    from realmspinner import models

    assert second["key"] in models.style_loras_snapshot()


def test_import_lora_stages_the_copy_and_leaves_no_truncated_blob(
    svc, lora_home, tmp_path, monkeypatch
):
    """The copy used to land on the final name in place, so a failure part-way
    left a truncated ``.safetensors`` no manifest row would ever clean up."""
    import shutil

    src = _safetensors(tmp_path / "style.safetensors")
    root = lora_home

    def dies_halfway(source, destination, **kwargs):
        Path(destination).write_bytes(b"trunc")
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(shutil, "copy2", dies_halfway)
    with pytest.raises(Failed) as excinfo:
        svc_loras.import_lora(svc, src, label="halfway")
    assert excinfo.value.field == "source"
    leftovers = [p.name for p in root.iterdir()] if root.exists() else []
    assert not [n for n in leftovers if n.endswith(".safetensors") or n.endswith(".tmp")], leftovers
