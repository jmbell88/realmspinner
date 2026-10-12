"""The 2026-10-10 audit's findings for Create's recipe column and result card.

create-09 (a removed style LoRA the form still holds), create-11 (the Tileset
arm's Conditioning section), create-19 (``resolved_recipe``'s memo key),
create-28 (Reproduce with no recorded seed) and create-31 (Vary and the seed
lock).
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import Any

import pytest

from realmspinner import generation, guidance
from realmspinner.studio.modes.create.engine import assets as create_assets
from realmspinner.studio.modes.create.engine import recipe as create_recipe
from realmspinner.studio.state import AppState, default_form_2d

# --- create-09 ------------------------------------------------------------------


def _lora_ctx(by_base: dict[str, list[str]], loras: list[tuple[str, str]]) -> SimpleNamespace:
    catalog = dict(guidance.catalog())
    catalog["loras_by_base"] = by_base
    catalog["lora_bases"] = [k for k, v in by_base.items() if v]
    return SimpleNamespace(
        guidance=catalog, base_models=[(k, k.upper()) for k in by_base], style_loras=loras
    )


def test_a_removed_style_lora_held_by_the_form_is_still_listed_or_cleared():
    """Removing an imported LoRA in Settings refreshes ``ctx.style_loras`` but
    not ``form_2d["style_lora"]``. ``lora_options`` only walked the registry,
    so the combo fell back to index 0 over a key ``validate`` was refusing
    ("Choose a recognised style LoRA.") -- the dead end its own docstring says
    it must not become."""
    ctx = _lora_ctx({"a": ["one"]}, [("", "no style LoRA"), ("one", "One")])
    form = {"base_model": "a", "style_lora": "imported-then-removed"}
    options = create_recipe.lora_options(ctx, form)
    keys = [key for key, _ in options]
    assert "imported-then-removed" in keys, options
    label = dict(options)["imported-then-removed"]
    assert "removed" in label or "no longer" in label, label


def test_the_style_lora_picker_always_offers_no_style():
    """Without the empty key the combo cannot set the form back to no style:
    the way out of a held key that no longer fits."""
    ctx = _lora_ctx({"a": ["one"]}, [("", "no style LoRA"), ("one", "One")])
    keys = [key for key, _ in create_recipe.lora_options(ctx, {"base_model": "a"})]
    assert keys[0] == ""
    assert "one" in keys


# --- create-11 ------------------------------------------------------------------


def _tile_form() -> dict[str, Any]:
    form = default_form_2d()
    form.update(
        output="sheet",
        sheet_type="tile",
        asset_type="tileset",
        prompt="a village",
        tile_mode="materials",
        materials="grass\nsand",
        ref_path="ref.png",
        ip_adapter="plus",
        control="canny",
        init_image=True,
    )
    return form


class _Recorder:
    """Stands in for ``widgets``/``controls``/``imgui``: every call is a no-op
    that notes its name, except the few that must hand a value back."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def __getattr__(self, name: str):
        def call(*args: Any, **kwargs: Any) -> Any:
            self.calls.append((name, args))
            if name == "combo":
                return args[1]
            if name in ("checkbox",):
                return (False, args[1])
            if name == "slider_float":
                return (False, args[1])
            if name in ("button", "disabled_button"):
                return False
            if name == "begin_popup":
                # The inspector's "More" menu: open, so its rows are drawn.
                return True
            return None

        return call

    def labels(self) -> list[str]:
        return [args[0] for name, args in self.calls if name == "field_label"]


def test_the_tile_arm_conditioning_section_offers_only_the_reference_and_counts_only_it(
    monkeypatch,
):
    """``tile_sheet_kwargs`` sends the reference bytes and nothing else, but the
    Conditioning section still drew the appearance picker, strength, "Start
    from this image" and Structure, and ``conditioning_tail`` counted them
    ("(4 on)")."""
    from realmspinner.studio.modes.create.ui.panes import settings_2d as mod

    form = _tile_form()
    kwargs = create_recipe.tile_sheet_kwargs(form)
    assert not {"ip_adapter", "control", "init_image"} & set(kwargs)
    assert create_recipe.conditioning_tail(form).strip() == "(1 on)"

    widgets, controls = _Recorder(), _Recorder()
    monkeypatch.setattr(mod, "widgets", widgets)
    monkeypatch.setattr(mod, "controls", controls)
    monkeypatch.setattr(mod, "imgui", _Recorder())
    ctx = SimpleNamespace(busy=lambda key: False, state=AppState(), submit=lambda *a, **k: True)
    mod._reference_body(ctx, form, None)
    assert widgets.labels() == [], widgets.labels()
    assert not [name for name, _ in controls.calls if name in ("checkbox", "slider_float")]
    # The picker itself is still there: the door takes a reference.
    assert any(
        name == "disabled_button" and args and "Replace" in str(args[0])
        for name, args in widgets.calls
    )


def test_the_non_tile_conditioning_section_is_unchanged(monkeypatch):
    """The counterpart: an image form still draws and counts all four."""
    from realmspinner.studio.modes.create.ui.panes import settings_2d as mod

    form = default_form_2d()
    form.update(ref_path="ref.png", ip_adapter="plus", control="canny", init_image=True)
    assert create_recipe.conditioning_tail(form).strip() == "(4 on)"
    widgets = _Recorder()
    monkeypatch.setattr(mod, "widgets", widgets)
    monkeypatch.setattr(mod, "controls", _Recorder())
    monkeypatch.setattr(mod, "imgui", _Recorder())
    # The Strength fields are ``create_brief.card_field`` blocks now (a label,
    # then the control, in the card's rhythm); this test is about *which* groups
    # draw, so the field helper is a no-op and the group names are sub-labels.
    monkeypatch.setattr(
        mod, "create_brief", SimpleNamespace(card_field=lambda *a, **k: contextlib.nullcontext())
    )
    monkeypatch.setattr(mod.create_recipe, "img2img_note", lambda ctx, form: None)
    monkeypatch.setattr(mod.create_recipe, "structure_picker_note", lambda ctx, form: None)
    ctx = SimpleNamespace(
        busy=lambda key: False,
        state=AppState(),
        guidance={},
        submit=lambda *a, **k: True,
    )
    mod._reference_body(ctx, form, None)
    sub = {args[0] for name, args in widgets.calls if name == "sub_label"}
    assert {"appearance", "start image", "structure"} <= sub


# --- create-19 ------------------------------------------------------------------


def _counting(monkeypatch) -> list[int]:
    calls: list[int] = []
    original = generation.resolve_recipe

    def counting(request, config=None, *, installed=None):
        calls.append(1)
        return original(request, config, installed=installed)

    monkeypatch.setattr(generation, "resolve_recipe", counting)
    return calls


def test_typing_in_the_prompt_does_not_re_resolve_the_recipe(monkeypatch):
    """The memo was keyed on the whole ``GenerationRequest``, so every typed
    character (and every Style-strength slider frame) re-ran ``resolve_recipe``
    and re-fingerprinted every installed checkpoint directory on the frame
    thread. Routing reads the type, quality, model mode/override and style
    only."""
    form = default_form_2d()
    create_assets.sync_legacy_fields(form)
    ctx = SimpleNamespace(state=AppState(), svc=SimpleNamespace(config=None))
    calls = _counting(monkeypatch)

    create_recipe.resolved_recipe(ctx, form)
    assert len(calls) == 1
    form["prompt"] = str(form["prompt"]) + "a"
    create_recipe.resolved_recipe(ctx, form)
    form["negative_prompt"] = "blurry"
    create_recipe.resolved_recipe(ctx, form)
    form["seed"] = 7
    create_recipe.resolved_recipe(ctx, form)
    form["lora_weight"] = 0.51
    create_recipe.resolved_recipe(ctx, form)
    assert len(calls) == 1, f"typing/slider frames re-resolved {len(calls) - 1} times"


@pytest.mark.parametrize(
    "key, value",
    [
        ("quality", "fast"),
        ("model_mode", "advanced"),
        ("style_lora", "ps1"),
    ],
)
def test_a_routing_field_still_re_resolves_the_recipe(monkeypatch, key, value):
    """The narrower key must still notice every field routing reads."""
    if key == "quality":
        # "fast" is only a distinct request while a Fast recipe is registered;
        # the shipped registry has none since 2026-10-11, so install one.
        from _synthetic_models import install_fast_tier

        install_fast_tier(monkeypatch)
    form = default_form_2d()
    create_assets.sync_legacy_fields(form)
    ctx = SimpleNamespace(state=AppState(), svc=SimpleNamespace(config=None))
    calls = _counting(monkeypatch)
    create_recipe.resolved_recipe(ctx, form)
    form[key] = value
    create_recipe.resolved_recipe(ctx, form)
    assert len(calls) == 2


def test_an_installed_set_change_still_re_resolves_the_recipe(monkeypatch):
    form = default_form_2d()
    create_assets.sync_legacy_fields(form)
    ctx = SimpleNamespace(
        state=AppState(),
        svc=SimpleNamespace(config=None),
        model_rows=[{"row_key": "a", "present": False}],
    )
    calls = _counting(monkeypatch)
    create_recipe.resolved_recipe(ctx, form)
    ctx.model_rows = [{"row_key": "a", "present": True}]
    create_recipe.resolved_recipe(ctx, form)
    assert len(calls) == 2


# --- create-28 ------------------------------------------------------------------


def _drive_selected_actions(monkeypatch, job: dict[str, Any]):
    from realmspinner.studio.modes.create.ui import workspace as mod

    widgets, controls = _Recorder(), _Recorder()
    controls.ButtonRole = mod.controls.ButtonRole  # an enum, not a call to record
    monkeypatch.setattr(mod, "widgets", widgets)
    monkeypatch.setattr(mod, "controls", controls)
    monkeypatch.setattr(mod, "imgui", _Recorder())
    submitted: list[tuple[Any, ...]] = []
    ctx = SimpleNamespace(
        state=AppState(),
        cache=SimpleNamespace(get=lambda key: None),
        svc=object(),
        submit=lambda *a, **k: submitted.append((a, k)) or True,
    )
    mod.selected_actions(ctx, job)
    return widgets, controls, submitted


def test_reproduce_recorded_attempt_is_greyed_when_no_seed_was_recorded(monkeypatch):
    """"Reproduce recorded attempt" was offered for every rerollable row, but a
    rig, retexture, sheet, charsheet or music row records no seed, so the press
    ran ``rerun_job(seed=None)`` -- which draws a random one."""
    from realmspinner.studio.modes.create.ui import workspace as mod

    job = {"id": "x", "kind": "rig", "stage": "model", "status": "done", "params": {}}
    assert mod.recorded_seed(job) is None
    widgets, controls, submitted = _drive_selected_actions(monkeypatch, job)
    reproduce = [
        (args, ) for name, args in widgets.calls
        if name == "disabled_button" and args and args[0] == "Reproduce recorded attempt"
    ]
    assert reproduce, "the button must be a greyed disabled_button, not a live one"
    live = [
        a for name, a in controls.calls
        if name == "button" and a[0] == "Reproduce recorded attempt"
    ]
    assert not live
    assert mod.reproduce_reason(job) == "This attempt recorded no seed."
    assert reproduce[0][0][1] is False
    assert submitted == []


def test_reproduce_recorded_attempt_is_live_when_a_seed_was_recorded(monkeypatch):
    from realmspinner.studio.modes.create.ui import workspace as mod

    job = {
        "id": "x", "kind": "text", "stage": "reference", "status": "done",
        "params": {"seed": 0},
    }
    assert mod.reproduce_reason(job) == ""
    widgets, _controls, _ = _drive_selected_actions(monkeypatch, job)
    call = next(
        args for name, args in widgets.calls
        if name == "disabled_button" and args and args[0] == "Reproduce recorded attempt"
    )
    assert call[1] is True


# --- create-31 ------------------------------------------------------------------


def _vary_ctx(monkeypatch):
    from realmspinner.studio.modes.create.ui import stages as create_stages

    monkeypatch.setattr(create_stages, "go", lambda *a, **k: None)
    toasts: list[str] = []
    ctx = SimpleNamespace(state=AppState(mode="create"), toast=lambda t, *a, **k: toasts.append(t))
    return ctx, toasts


def test_vary_on_a_reference_keeps_the_recorded_seed_for_the_next_generate(monkeypatch):
    """Vary restored the recorded seed but left ``seed_locked`` False, and
    Generate rolls a new seed before building the request -- so the very first
    press after "Change one thing, then generate a variation" threw it away."""
    from realmspinner.studio.modes.create.ui import workspace as mod

    ctx, toasts = _vary_ctx(monkeypatch)
    job = {
        "id": "r1", "stage": "reference", "kind": "text", "status": "done",
        "params": {"prompt": "a chest", "seed": 111, "reference_seed": 222},
    }
    mod.vary(ctx, job)
    form = ctx.state.form_2d
    assert form["seed_locked"] is True
    assert form["seed"] == 222  # the draw that shipped input.png
    assert "seed" in toasts[0] and toasts[0].startswith("Loaded this brief")


def test_vary_on_a_row_with_no_seed_does_not_lock_one(monkeypatch):
    from realmspinner.studio.modes.create.ui import workspace as mod

    ctx, toasts = _vary_ctx(monkeypatch)
    job = {"id": "r1", "stage": "reference", "kind": "text", "params": {"prompt": "a chest"}}
    mod.vary(ctx, job)
    assert ctx.state.form_2d["seed_locked"] is False
    assert "seed" not in toasts[0]


def test_vary_on_a_mesh_locks_the_recorded_mesh_seed_by_the_same_rule(monkeypatch):
    """The mesh card's Vary used the recorded seed once and then rolled; the two
    Vary verbs follow one rule."""
    from realmspinner.studio.modes.create.ui import workspace as mod

    ctx, toasts = _vary_ctx(monkeypatch)
    job = {
        "id": "m1", "stage": "model", "kind": "mesh", "status": "done",
        "params": {"mesh_seed": 333},
    }
    mod.vary(ctx, job)
    assert ctx.state.form_3d["mesh_seed"] == 333
    assert ctx.state.form_3d["mesh_seed_locked"] is True
    assert "seed" in toasts[0]
