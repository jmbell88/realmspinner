"""The 2026-10-10 audit's structured-generation findings (create-13, create-22
to create-27): a tileset request recorded fields its door never used, a sprite
recipe advertised a reference mode only FLUX.2 draws from, ``from_dict`` did not
split a multi-line material list, a hand-edited LoRA manifest could shadow a
built-in style, the fingerprint skipped the tail of a 1-2 MiB file, the Fast
tier recorded the same checkpoint hash as Quality, and the curated recipe table
hand-typed facts the model registry already owns."""

from __future__ import annotations

import io
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from realmspinner import generation, models, provenance


def _png(path: Path) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGBA", (32, 32), (200, 50, 50, 255)).save(buf, "PNG")
    path.write_bytes(buf.getvalue())
    return str(path)


def _tileset(**over):
    base = dict(
        generation_type="tileset",
        prompt="mossy dungeon",
        tile=generation.TileSettings(mode="materials", prompt_items=("wood", "stone")),
    )
    base.update(over)
    return generation.GenerationRequest(**base)


def _fields(request, resolved=None):
    return {issue.field for issue in generation.validate_request(request, resolved)}


# --- create-13: a tileset request names only what the tile sheet door uses ----


def test_a_tileset_request_naming_a_style_lora_or_start_image_is_refused_not_recorded(
    tmp_path,
):
    """``create_tile_sheet`` pins its own base and pixel-art LoRA and has no
    parameter for a style LoRA, a start image, a model override or a count, so
    each validated clean, was recorded in ``generation_request`` and changed no
    pixel."""
    ref = _png(tmp_path / "r.png")
    resolved = generation.resolve_recipe(_tileset(), None)
    assert resolved is not None
    assert generation.validate_request(_tileset(), resolved) == []

    lora = next(iter(models.STYLE_LORAS))
    assert "style_lora" in _fields(_tileset(style_lora=lora, lora_weight=0.7), resolved)
    assert "init_image" in _fields(
        _tileset(references=(ref,), reference_mode="single", init_image=True), resolved
    )
    assert "base_model" in _fields(
        _tileset(model_mode="advanced", model_override="sdxl_cfg"), resolved
    )
    assert "count" in _fields(_tileset(count=2), resolved)
    # The one image a tile sheet does read (its style reference) is still fine.
    assert (
        generation.validate_request(
            _tileset(references=(ref,), reference_mode="single"), resolved
        )
        == []
    )


# --- create-22: the sprite recipe does not advertise a mode it cannot draw ----


def test_a_multi_reference_sprite_request_on_an_sdxl_recipe_is_refused(tmp_path):
    refs = tuple(_png(tmp_path / f"r{i}.png") for i in range(3))
    request = generation.GenerationRequest(
        generation_type="sprite_sheet",
        prompt="a knight",
        references=refs,
        reference_mode="multi",
    )
    resolved = generation.resolve_recipe(request, None)
    assert resolved is not None and resolved.base_model == "sdxl_cfg"
    assert "reference_mode" in _fields(request, resolved)
    assert not generation.capability_controls(request, resolved)["multi_reference"]


# --- create-23: a multi-line material list is split like the form splits it ---


def test_from_dict_splits_a_multi_line_tile_prompt_and_drops_blank_items():
    split = generation.GenerationRequest.from_dict(
        {
            "generation_type": "tileset",
            "prompt": "x",
            "tile": {"mode": "materials", "prompt_items": "wood\nstone\n"},
        }
    )
    assert split.tile.prompt_items == ("wood", "stone")
    blank = generation.GenerationRequest.from_dict(
        {
            "generation_type": "tileset",
            "prompt": "x",
            "tile": {"mode": "materials", "prompt_items": ["", "  "]},
        }
    )
    assert blank.tile.prompt_items == ()
    assert "tile.prompt_items" in _fields(blank, generation.resolve_recipe(blank, None))


# --- create-24: a manifest row cannot shadow a built-in or carry a bad weight -


def _row(key: str, **over) -> dict:
    row = {
        "key": key,
        "label": key,
        "family": models.FAMILY_SDXL,
        "trigger_text": "",
        "tuned_weight": 0.8,
        "license": "",
        "commercial": True,
        "source": "test",
        "checksum": "c",
        "filename": f"{key}.safetensors",
        "schema_version": 1,
    }
    row.update(over)
    return row


def test_a_manifest_row_cannot_shadow_a_builtin_style_or_carry_a_non_numeric_weight(
    tmp_path, monkeypatch
):
    root = tmp_path / "t2i"
    (root / "loras").mkdir(parents=True)
    builtin = next(iter(models.STYLE_LORAS))
    before = models.STYLE_LORAS[builtin]
    monkeypatch.setattr(models, "STYLE_LORAS", dict(models.STYLE_LORAS))
    rows = [
        _row(builtin, label="evil"),
        _row("imported_good"),
        _row("imported_str_weight", tuned_weight="abc"),
        _row("imported_nan", tuned_weight=float("nan")),
        _row("imported_inf", tuned_weight=float("inf")),
        _row("imported_bool", tuned_weight=True),
        _row("imported_family", family="nonsense"),
        _row("", label="blank key"),
    ]
    (root / "loras" / "manifests.json").write_text(
        json.dumps({"manifests": rows}), encoding="utf-8"
    )
    config = SimpleNamespace(t2i_model_root=root)
    generation._forget_manifests(generation.lora_manifest_path(config))
    assert [m.key for m in generation.load_lora_manifests(config)] == ["imported_good"]
    generation.register_imported_loras(config)
    assert models.STYLE_LORAS[builtin] == before
    assert "imported_str_weight" not in models.STYLE_LORAS


# --- create-25: the tail of a 1-2 MiB file is part of its fingerprint ---------


def test_a_file_between_one_and_two_mib_fingerprints_its_tail(tmp_path):
    f = tmp_path / "w.safetensors"
    size = int(1.5 * provenance.HASH_BYTES)
    f.write_bytes(b"\0" * size)
    first = provenance.file_fingerprint(f)
    f.write_bytes(b"\0" * (size - 10) + b"\1" * 10)
    provenance._cache.clear()
    assert provenance.file_fingerprint(f) != first


def test_the_fingerprint_of_a_file_over_two_mib_is_unchanged(tmp_path):
    """The format is a drift guard for stored recipes: only the 1-2 MiB band
    moves, and a file outside it hashes head||tail exactly as before."""
    import hashlib

    f = tmp_path / "big.bin"
    data = bytes(range(256)) * (3 * provenance.HASH_BYTES // 256)
    f.write_bytes(data)
    n = provenance.HASH_BYTES
    expect = hashlib.sha256(data[:n] + data[-n:]).hexdigest()[:32]
    assert provenance.file_fingerprint(f) == f"s{len(data)}:{expect}"


# --- create-26: Fast and Quality record different checkpoint provenance -------


def test_the_fast_tier_records_a_checksum_that_differs_when_the_hyper_sd_file_changes(
    svc, installed_recipes
):
    spec = models.BASE_MODELS["sdxl"]
    assert spec.base_lora
    lora = Path(svc.config.t2i_model_root) / "loras" / spec.base_lora
    lora.parent.mkdir(parents=True, exist_ok=True)

    def fast():
        provenance._cache.clear()
        resolved = generation.resolve_recipe(
            generation.GenerationRequest(
                generation_type="image", prompt="a barrel", quality="fast"
            ),
            svc.config,
        )
        assert resolved is not None and resolved.base_model == "sdxl"
        return resolved

    lora.write_bytes(b"hyper-sd v1")
    one = fast()
    lora.write_bytes(b"hyper-sd v2 -- different")
    two = fast()
    assert one.base_lora_checksum and two.base_lora_checksum
    assert one.base_lora_checksum != two.base_lora_checksum
    assert one.to_dict()["base_lora_checksum"] == one.base_lora_checksum

    quality = generation.resolve_recipe(
        generation.GenerationRequest(generation_type="image", prompt="a barrel"), svc.config
    )
    assert quality is not None and quality.base_model == "sdxl_cfg"
    assert quality.base_lora_checksum is None


# --- create-27: the curated recipes agree with the registry -------------------


def test_every_curated_recipe_agrees_with_its_base_model_registry_row():
    for recipe in generation.RECIPES:
        base = models.BASE_MODELS[recipe.base_model]
        assert recipe.vram_gib == base.vram_gib, recipe.key
        assert recipe.ram_gib == (base.host_peak_gib or base.vram_gib), recipe.key
        assert recipe.license == (base.license or ""), recipe.key
        assert recipe.commercial == base.commercial, recipe.key
    flux = next(r for r in generation.RECIPES if r.key == "image_flux2")
    assert flux.ram_gib == 24.0
    assert asdict(flux)["license"]
