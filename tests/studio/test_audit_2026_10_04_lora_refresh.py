"""The 2026-10-04 audit, finding create-21: Create's style LoRA picker was frozen
at startup.

``ctx.guidance`` (``loras_by_base``/``lora_bases``) was built once in
``App._load_static_answers`` and ``ctx.style_loras`` was rebuilt only after a
model download, while the ``lora:import`` / ``lora:remove:`` landings only
toasted. A style imported in-session was "added" and then absent from Create's
combo until restart; a removed one stayed offered and was refused at submit.
A finished ``lora_train`` registers its style through the same
``generation.import_lora`` and had the same hole.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from realmspinner import guidance, models
from realmspinner.studio import main as main_mod
from realmspinner.studio.modes.create.engine import recipe
from realmspinner.studio.state import AppState


class _Done:
    def __init__(self, key: str, result: Any = None, tag: Any = None) -> None:
        self.key = key
        self.result = result
        self.tag = tag
        self.ok = True


def _style_rows() -> list[tuple[str, str]]:
    """``App._refresh_model_answers``'s own shape for ``ctx.style_loras``."""
    return [("", "no style LoRA")] + [
        (k, spec.label) for k, spec in models.style_loras_snapshot().items()
    ]


class _Ctx:
    """The slice of the ctx the lora landings and Create's picker read."""

    def __init__(self) -> None:
        self.state = AppState()
        self.toasts: list[tuple] = []
        # As startup left them: the registry at that moment.
        self.guidance = guidance.catalog()
        self.style_loras = _style_rows()
        self.cache = SimpleNamespace(jobs={})

    def toast(self, *a: Any, **k: Any) -> None:
        self.toasts.append((a, k))


class _App(main_mod.App):
    """The real landing methods on a ctx that is just the slice above."""

    def __init__(self, ctx: _Ctx) -> None:  # noqa: D107 - skips App's window boot
        self.app_ctx = ctx

    def _request_storage(self, *_a: Any, **_k: Any) -> None:
        pass


@pytest.fixture
def registry(monkeypatch):
    """A private copy of the style table, so the test never edits the real one."""
    table = dict(models.STYLE_LORAS)
    monkeypatch.setattr(models, "STYLE_LORAS", table)
    return table


def _register(table: dict, key: str = "imported_abc", label: str = "Inkwash") -> None:
    with models.STYLE_LORAS_LOCK:
        table[key] = models.StyleLora(
            key=key, label=label, filename=f"{key}.safetensors", family=models.FAMILY_SDXL
        )


def _offered(ctx: _Ctx) -> list[str]:
    form = {"base_model": "sdxl_cfg"}
    return [key for key, _ in recipe.lora_options(ctx, form)]


def test_an_imported_style_lora_is_offered_by_create_without_a_restart(registry):
    ctx = _Ctx()
    app = _App(ctx)
    assert "imported_abc" not in _offered(ctx)

    # What ``service.loras.import_lora`` does on the task thread before the
    # landing runs: the in-memory registry already holds the style.
    _register(registry)
    app._on_task_done(_Done("lora:import", {"label": "Inkwash"}))

    assert "imported_abc" in _offered(ctx)
    assert recipe.lora_note(ctx, {"base_model": "sdxl_cfg"}) is None
    assert ("imported_abc", "Inkwash") in ctx.style_loras
    assert any(e["key"] == "imported_abc" for e in ctx.guidance["fields"]["style_lora"])


def test_a_removed_style_lora_stops_being_offered_by_create_without_a_restart(registry):
    _register(registry)
    ctx = _Ctx()
    app = _App(ctx)
    assert "imported_abc" in _offered(ctx)

    with models.STYLE_LORAS_LOCK:
        registry.pop("imported_abc")
    app._on_task_done(_Done("lora:remove:imported_abc", {"ok": True}, tag="Inkwash"))

    assert "imported_abc" not in _offered(ctx)
    assert all(key != "imported_abc" for key, _ in ctx.style_loras)


def test_a_finished_lora_train_job_offers_its_style_to_create_without_a_restart(registry):
    ctx = _Ctx()
    app = _App(ctx)
    _register(registry, "imported_trained", "Trained style")

    app._announce_job_transition(
        {"id": "job1", "kind": "lora_train", "status": "done", "stage": "reference"},
        "running",
    )

    assert "imported_trained" in _offered(ctx)


def test_refreshing_the_style_tables_leaves_the_rest_of_guidance_alone(registry):
    ctx = _Ctx()
    app = _App(ctx)
    style_slots = ("fields", "loras_by_base", "lora_bases")
    before = {k: v for k, v in ctx.guidance.items() if k not in style_slots}
    other_fields = {k: v for k, v in ctx.guidance["fields"].items() if k != "style_lora"}
    _register(registry)

    app._on_task_done(_Done("lora:import", {"label": "Inkwash"}))

    assert {k: v for k, v in ctx.guidance.items() if k not in style_slots} == before
    assert {k: v for k, v in ctx.guidance["fields"].items() if k != "style_lora"} == other_fields
