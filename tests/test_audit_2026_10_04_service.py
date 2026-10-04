"""The 2026-10-04 audit's service-layer findings from the Create brief slice
(create-39 through create-42): the reroll's reference seed, the structured
door's negative prompt, the Automatic arm's LoRA checksum, and a number too
large to convert."""

from __future__ import annotations

import io

import pytest

from realmspinner import generation, guidance
from realmspinner.service import jobs as svc_jobs
from realmspinner.service.errors import Invalid


def _png_bytes() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGBA", (32, 32), (200, 50, 50, 255)).save(buf, "PNG")
    return buf.getvalue()


# --- create-39: a mesh reroll keeps the reference it reuses -------------------


def test_rerolling_a_mesh_keeps_the_reference_seed_of_the_reference_it_reuses(svc):
    """A model-stage row is an *image* job: ``input.png`` is copied from the
    source and SDXL never runs, so a reroll redraws only the 3D stage. The reroll
    arm wrote ``reference_seed = fresh`` regardless, and the Inspector then named
    a seed that did not draw the reference."""
    assets = svc.config.data_dir
    src = svc_jobs.create_job(
        svc, kind="text", prompt="a barrel", seed=42, reference_seed=99, mesh_seed=42
    )["id"]
    (assets / src).mkdir(parents=True, exist_ok=True)
    (assets / src / "input.png").write_bytes(_png_bytes())
    # The remesh is the model-stage image row a mesh card rerolls.
    mesh = svc_jobs.rerun_job(svc, src, mode="remesh")["id"]
    assert svc.store.get(mesh)["params"]["reference_seed"] == 99

    again = svc_jobs.rerun_job(svc, mesh, mode="reroll")["id"]
    params = svc.store.get(again)["params"]
    assert params["reference_seed"] == 99
    assert params["mesh_seed"] != svc.store.get(mesh)["params"]["mesh_seed"]


# --- create-40: the structured door records no inert Avoid text ---------------


def test_create_generation_request_does_not_record_an_inert_negative_prompt_on_a_distilled_recipe(
    svc, installed_recipes
):
    """Fast resolves to a distilled checkpoint (guidance 0) that never encodes a
    negative branch; every other path records ``effective_negative_prompt``, but
    this arm passed ``request.negative_prompt`` raw, so findings credited a
    no-op value."""
    made = svc_jobs.create_generation_request(
        svc, {
            "generation_type": "image",
            "prompt": "a barrel",
            "quality": "fast",
            "negative_prompt": "blurry",
        }
    )
    params = svc.store.get(made["id"])["params"]
    assert not params.get("negative_prompt"), params.get("negative_prompt")


# --- create-41: Automatic records the built-in LoRA checksum ------------------


def test_automatic_routing_records_the_builtin_lora_checksum_like_advanced(
    svc, installed_recipes
):
    from realmspinner import fetch

    entry = fetch.find("lora:render3d")
    assert entry is not None
    # A LoRA row's destination is the flat ``loras/`` directory, which Advanced
    # fingerprints whole; put one file in it so the answer is a real digest.
    folder = fetch.destination(svc.config, entry, entry.fetch[0])
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "3d_render_style_xl.safetensors").write_bytes(b"a stable stand-in")

    automatic = generation.resolve_recipe(
        generation.GenerationRequest(
            generation_type="image", prompt="a barrel", style_lora="render3d"
        ),
        svc.config,
    )
    assert automatic is not None
    advanced = generation.resolve_recipe(
        generation.GenerationRequest(
            generation_type="image",
            prompt="a barrel",
            style_lora="render3d",
            model_mode="advanced",
            model_override=automatic.base_model,
        ),
        svc.config,
    )
    assert advanced is not None and advanced.lora_checksum
    assert automatic.lora_checksum == advanced.lora_checksum


# --- create-42: a number too large to convert is a refusal, not a crash -------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("size_m", 10**400),
        ("lora_weight", 10**400),
        ("resolution", float("inf")),
        ("ip_scale", 10**400),
        ("control_scale", 10**400),
        ("control_end", 10**400),
        ("init_strength", 10**400),
    ],
    ids=lambda v: v if isinstance(v, str) else "huge",
)
def test_normalize_refuses_a_number_too_large_to_convert_with_its_field(field, value):
    """``float(10**400)`` and ``int(inf)`` raise ``OverflowError``, which is
    neither the ``TypeError`` nor the ``ValueError`` these conversions caught,
    so the exception left ``normalize`` bare and past the door's
    ``except ValueError``."""
    with pytest.raises(guidance.GuidanceError) as excinfo:
        guidance.normalize({field: value})
    # ``resolution`` reports against the platform control that supplies it.
    assert excinfo.value.field == ("platform" if field == "resolution" else field)


def test_the_door_refuses_a_number_too_large_to_convert_with_its_field(svc):
    with pytest.raises(Invalid) as excinfo:
        svc_jobs.create_job(svc, kind="text", prompt="a barrel", size_m=10**400)
    assert excinfo.value.field == "size_m"
