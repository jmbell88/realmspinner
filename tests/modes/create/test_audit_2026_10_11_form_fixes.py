"""Findings closed from the 2026-10-11 Create audit: the persisted Advanced form,
the removed-model handling after commit 800001dc, the Sprite Sheet arm's stale
base and the per-frame cost of ``automatic_install``.

create-01: a stored Advanced form naming a removed ``model_override`` restored
intact while ``base_model`` fell back, so Generate was enabled and then refused
on the unringed ``recipe`` field.

create-04: Copy settings, Vary and reopen put a removed ``base_model`` into the
form silently; the refusal was the generic "Choose a recognised image model."

create-05: Automatic on the Sprite Sheet arm submitted the stale
``form["base_model"]`` an earlier Advanced pick left behind.

create-06: one ``0.0..1.5`` ceiling on three fields reverted a Structure
strength between 1.5 and 2.0 on every restart.

create-07: the findings "Use <value>" offer could name a removed model.

create-21: ``automatic_install`` re-resolved on every frame.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from realmspinner import generation, models
from realmspinner.bench import findings as findings_lib
from realmspinner.studio import settings as studio_settings
from realmspinner.studio.modes.create.engine import assets as create_assets
from realmspinner.studio.modes.create.engine import recipe as create_recipe
from realmspinner.studio.modes.create.ui.panes import settings_2d
from realmspinner.studio.state import AppState, default_form_2d, form_from_params


def _filled() -> dict:
    form = default_form_2d()
    form["prompt"] = "a wooden crate"
    create_assets.sync_legacy_fields(form)
    return form


# --- create-01 ---------------------------------------------------------------------


def test_restoring_a_form_with_a_removed_model_override_drops_it_rather_than_refusing_on_an_unringed_field():  # noqa: E501
    stored = {
        **default_form_2d(),
        "prompt": "a wooden crate",
        "model_mode": "advanced",
        "model_override": "juggernaut",
        "base_model": "sdxl_cfg",
    }
    restored = studio_settings.restore_form(default_form_2d(), stored)
    assert restored["model_override"] in ("", *models.BASE_MODELS), restored["model_override"]
    request = generation.request_from_legacy(restored)
    assert request.model_override in (None, *models.BASE_MODELS)
    assert generation.validate_request(request, generation.resolve_recipe(request, None)) == []


def test_a_restored_model_override_in_the_registry_still_restores():
    stored = {
        **default_form_2d(),
        "model_mode": "advanced",
        "model_override": "flux_klein",
        "base_model": "flux_klein",
    }
    assert "flux_klein" in models.BASE_MODELS
    restored = studio_settings.restore_form(default_form_2d(), stored)
    assert restored["model_override"] == "flux_klein"


def test_validate_names_a_removed_override_with_the_ledger_sentence():
    """The override is what ``request_from_legacy`` reads under Advanced, so a form
    built in memory (not restored) that holds a removed one is refused by name."""
    form = _filled()
    form["model_mode"] = "advanced"
    form["base_model"] = "sdxl_cfg"
    form["model_override"] = "juggernaut"
    problems = create_recipe.validate(form)
    entry = models.RETIRED_MODELS["juggernaut"]
    assert [(str(p), p.field) for p in problems] == [
        (models.retired_message(entry), "base_model")
    ]


# --- create-04 ---------------------------------------------------------------------


def test_copy_settings_from_a_job_on_a_removed_model_says_it_was_removed():
    form = form_from_params({"prompt": "x", "base_model": "juggernaut", "seed": 3})
    form["prompt"] = "a wooden crate"
    entry = models.RETIRED_MODELS["juggernaut"]
    problems = create_recipe.validate(form, SimpleNamespace(svc=SimpleNamespace(config=None)))
    assert [(str(p), p.field) for p in problems] == [
        (models.retired_message(entry), "base_model")
    ]
    ctx = SimpleNamespace(base_models=[("sdxl_cfg", "SDXL 1.0")])
    options = create_recipe.model_options(ctx, form)
    assert ("juggernaut", "Juggernaut XL v9 - removed") in options


def test_a_recorded_request_naming_a_removed_model_says_it_was_removed_too():
    request = generation.GenerationRequest(
        generation_type="3d_model",
        prompt="x",
        model_mode="advanced",
        model_override="dreamshaper",
    )
    form = form_from_params({"generation_request": request.to_dict()})
    form["prompt"] = "a wooden crate"
    entry = models.RETIRED_MODELS["dreamshaper"]
    problems = create_recipe.validate(form)
    assert [str(p) for p in problems] == [models.retired_message(entry)]


def test_a_removed_style_lora_in_copied_settings_says_it_was_removed():
    form = form_from_params({"style_lora": "pixelklein"})
    form["prompt"] = "a wooden crate"
    entry = models.RETIRED_MODELS["pixelklein"]
    messages = [str(p) for p in create_recipe.validate(form)]
    assert models.retired_message(entry) in messages


def test_model_options_lists_nothing_extra_for_a_live_or_automatic_form():
    ctx = SimpleNamespace(base_models=[("sdxl_cfg", "SDXL 1.0")])
    plain = [("", "Automatic"), ("sdxl_cfg", "SDXL 1.0")]
    assert create_recipe.model_options(ctx, _filled()) == plain
    assert create_recipe.model_options(ctx) == plain


# --- create-05 ---------------------------------------------------------------------


def test_sprite_sheet_under_automatic_submits_the_resolved_base_not_the_stale_form_base(
    monkeypatch,
):
    form = _filled()
    form["asset_type"] = "sprite_sheet"
    create_assets.sync_legacy_fields(form)
    form["model_mode"] = "auto"
    form["model_override"] = ""
    # What an earlier Advanced pick left behind and the switch back to Automatic
    # never cleared.
    stale = next(k for k in models.BASE_MODELS if k != "sdxl_cfg")
    form["base_model"] = stale
    resolved = generation.resolve_recipe(generation.request_from_legacy(form), None)
    assert resolved is not None and resolved.base_model != stale

    captured: dict = {}

    def fake_create_job(svc, **kwargs):
        captured.update(kwargs)
        return {"id": "job1"}

    monkeypatch.setattr(settings_2d.svc_jobs, "create_job", fake_create_job)
    toasts: list = []

    def _submit(key, run, *, tag=None):
        run()
        return True

    ctx = SimpleNamespace(
        state=AppState(),
        svc=SimpleNamespace(config=None),
        submit=_submit,
        toast=lambda *a, **k: toasts.append(a),
    )
    settings_2d.generate(ctx, form)
    assert not toasts, toasts
    assert captured, "generate() never reached the create_job door"
    assert captured["guidance_fields"]["base_model"] == resolved.base_model


# --- create-06 ---------------------------------------------------------------------


def test_a_structure_strength_above_one_and_a_half_survives_a_restart():
    stored = {**default_form_2d(), "control_scale": 1.8, "control_end": 0.9, "ip_scale": 1.2}
    restored = studio_settings.restore_form(default_form_2d(), stored)
    assert restored["control_scale"] == 1.8
    assert restored["control_end"] == 0.9
    assert restored["ip_scale"] == 1.2


@pytest.mark.parametrize(
    "key,value",
    [
        ("ip_scale", models.IP_SCALE_MAX + 0.1),
        ("control_scale", models.CONTROL_SCALE_MAX + 0.1),
        ("control_end", models.CONTROL_END_MAX + 0.1),
        ("control_end", -0.1),
    ],
)
def test_a_persisted_strength_past_its_own_ceiling_still_reverts(key, value):
    default = default_form_2d()
    restored = studio_settings.restore_form(default_form_2d(), {**default, key: value})
    assert restored[key] == default[key]


# --- create-07 ---------------------------------------------------------------------


def _bucket(n: int, wilson: float) -> dict:
    return {"n": n, "accepts": n, "wilson_low": wilson, "mean_grade": 1.0}


def test_best_value_offer_never_names_a_removed_model_or_style():
    doc = {
        "params": {
            "base_model": {"juggernaut": _bucket(9, 0.9), "sdxl_cfg": _bucket(9, 0.5)},
            "style_lora": {"pixelklein": _bucket(9, 0.9), "no-such-style": _bucket(9, 0.8)},
        }
    }
    found = findings_lib.best_value(doc, "base_model", "flux_klein", min_n=3)
    assert found is not None and found[0] == "sdxl_cfg"
    assert findings_lib.best_value(doc, "style_lora", "", min_n=3) is None


def test_best_value_offer_still_ranks_other_params_by_their_own_keys():
    doc = {"params": {"guidance": {"4.0": _bucket(9, 0.9), "7.0": _bucket(9, 0.5)}}}
    found = findings_lib.best_value(doc, "guidance", 7.0, min_n=3)
    assert found is not None and found[0] == "4.0"


# --- create-21 ---------------------------------------------------------------------


def _install_ctx(config, rows):
    return SimpleNamespace(
        state=AppState(),
        svc=SimpleNamespace(config=config),
        model_rows=rows,
    )


def test_automatic_install_does_not_re_resolve_while_nothing_it_reads_has_changed(monkeypatch):
    rows = [{"row_key": "base:sdxl_cfg", "present": False, "label": "SDXL 1.0", "size_gib": 7.0}]
    ctx = _install_ctx(None, rows)
    form = _filled()
    calls: list[int] = []
    original = generation.resolve_recipe

    def counting(request, config=None, *, installed=None):
        calls.append(1)
        return original(request, config, installed=installed)

    monkeypatch.setattr(generation, "resolve_recipe", counting)
    first = create_recipe.automatic_install(ctx, form)
    assert first is not None
    for _ in range(5):
        ctx.state.frame_index += 1
        assert create_recipe.automatic_install(ctx, form) == first
    assert len(calls) == 1, f"expected one resolution across many frames, got {len(calls)}"


def test_automatic_install_re_resolves_when_a_row_lands_or_the_form_routes_elsewhere(monkeypatch):
    rows = [{"row_key": "base:sdxl_cfg", "present": False, "label": "SDXL 1.0", "size_gib": 7.0}]
    ctx = _install_ctx(None, rows)
    form = _filled()
    assert create_recipe.automatic_install(ctx, form) is not None
    # The download lands: the answer must change on the next frame.
    ctx.model_rows = [{**rows[0], "present": True}]
    assert create_recipe.automatic_install(ctx, form) is None
    # And back (a delete): the memo must not pin the earlier answer.
    ctx.model_rows = rows
    assert create_recipe.automatic_install(ctx, form) is not None
