"""Stand-ins for registry shapes the shipped registry no longer contains.

The distilled and fine-tune checkpoints were retired on 2026-10-11
(``models.RETIRED_MODELS``), and with them every registry entry that carried a
step-distillation ``base_lora`` or a non-default ``scheduler``. The machinery
for both stays -- the next such entry will use it -- and the tests that cover it
need *some* entry shaped that way. They get one here, installed with
``monkeypatch.setitem(models.BASE_MODELS, spec.key, spec)`` so it is gone after
the test and never reaches a real registry walk.
"""

from __future__ import annotations

from realmspinner import models

#: The file the synthetic entry's step-distillation LoRA is called.
SYNTHETIC_BASE_LORA = "Test-4steps-lora.safetensors"


def distilled_base(key: str = "distilled_test", **overrides) -> models.BaseModel:
    """A 4-step SDXL base over the shared SDXL 1.0 weights plus a step LoRA.

    The shape of the retired ``sdxl`` (Hyper-SD) entry: guidance 0, trailing
    timestep spacing, a ``base_lora`` that ``load()`` fuses unconditionally, and
    the LoRA fetched into the flat ``loras/`` directory. ``overrides`` replaces
    any field.
    """
    fields = {
        "key": key,
        "label": "Distilled test model",
        "dir_name": "sdxl-base-1.0",
        "image_size": 1024,
        "steps": 4,
        "guidance_scale": 0.0,
        "scheduler": "ddim_trailing",
        "base_lora": SYNTHETIC_BASE_LORA,
        "fetch": (
            models._SDXL_BASE_1_0,
            models.Fetch(
                "acme/test-distill",
                "loras",
                revision="0" * 40,
                filenames=(SYNTHETIC_BASE_LORA,),
                size_gib=0.8,
            ),
        ),
        "description": "A synthetic distilled recipe for tests.",
        "license": "OpenRAIL++-M",
        "commercial": True,
    }
    fields.update(overrides)
    return models.BaseModel(**fields)


def install(monkeypatch, spec: models.BaseModel | None = None) -> models.BaseModel:
    """Register ``spec`` (default: :func:`distilled_base`) for one test."""
    spec = spec or distilled_base()
    monkeypatch.setitem(models.BASE_MODELS, spec.key, spec)
    return spec


def install_fast_tier(monkeypatch) -> models.BaseModel:
    """:func:`install` plus a ``quality="fast"`` recipe over it. -> the base spec.

    The retired ``image_fast`` tier's shape: guidance 0, so no negative prompt
    and no ControlNet, resolved by ``generation.resolve_recipe`` ahead of
    Quality only when a request says ``quality="fast"``.
    """
    from realmspinner import generation

    spec = install(monkeypatch)
    recipe = generation.Recipe(
        "image_fast",
        "Fast image",
        ("image",),
        "fast",
        spec.key,
        working_resolution=(1024, 1024),
        negative_prompt=False,
        vram_gib=spec.vram_gib,
        ram_gib=spec.vram_gib,
        license=spec.license,
        commercial=True,
        required_downloads=(f"base:{spec.key}",),
        note=(
            "Four steps instead of thirty. No structure control and no negative "
            "prompt: it runs at guidance 0."
        ),
        rank=20,
    )
    monkeypatch.setattr(generation, "RECIPES", (recipe, *generation.RECIPES))
    return spec
