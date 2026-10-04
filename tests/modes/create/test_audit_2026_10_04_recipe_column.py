"""The 2026-10-04 audit, Create's recipe column (settings_2d.py, engine/recipe.py).

create-20  the findings "Use" button wrote ``form[param]`` alone, so the Model
           combo moved while ``model_override`` (which the request reads) kept
           the old model, and a style LoRA kept the previous adapter's weight.
create-22  the Sprite arm's Cell target was validated and sent, and the door
           (``_check_sprite_sheet``) throws it away.
create-24  Model -> Automatic left a style LoRA / start image the resolved base
           cannot take, marked "not fitted" with Generate refused.
create-25  Structure picker hidden by a stale ``base_model`` under Automatic.
create-27  a vanished reference kept its ip_adapter / control / init_image.
create-35  a refused Generate rerolled the unlocked seed.
create-52  the Conditioning section loaded findings per control.
create-53  "cleared" notes outlived an edit of the field they describe.
create-54  the inert-negative note said "guidance 0" for LCM and klein (1.0).
create-62  dead / untested helpers.
"""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

from realmspinner import generation, guidance, models
from realmspinner.bench import findings as findings_lib
from realmspinner.studio import problems as problem_types
from realmspinner.studio.modes.create.engine import assets as create_assets
from realmspinner.studio.modes.create.engine import recipe as create_recipe
from realmspinner.studio.modes.create.ui.panes import settings_2d
from realmspinner.studio.state import AppState, default_form_2d


def _catalog_ctx(**extra):
    catalog = guidance.catalog()
    ns = SimpleNamespace(
        guidance=catalog,
        base_models=[(m["key"], m["label"]) for m in catalog["fields"]["base_model"]],
        style_loras=[(m["key"], m["label"]) for m in catalog["fields"]["style_lora"]],
        svc=SimpleNamespace(config=None),
        state=AppState(),
        toast=lambda *a, **k: None,
    )
    for key, value in extra.items():
        setattr(ns, key, value)
    return ns


def _form(**over):
    form = default_form_2d()
    create_assets.sync_legacy_fields(form)
    form.update({"prompt": "a wooden crate", **over})
    return form


def _findings(tmp_path, params):
    (tmp_path / "findings.json").write_text(
        json.dumps({"version": 5, "generated": "x", "params": params}), encoding="utf-8"
    )
    findings_lib._CACHE.clear()


def _press_use(monkeypatch):
    monkeypatch.setattr(settings_2d.controls, "button", lambda *a, **k: True)
    monkeypatch.setattr(settings_2d.widgets, "muted", lambda *a, **k: None)
    monkeypatch.setattr(settings_2d.imgui, "same_line", lambda *a, **k: None)


# --- create-20 -----------------------------------------------------------------


def test_use_button_on_base_model_changes_model_override_and_clears_unusable_selections(
    tmp_path, monkeypatch
):
    """The combo shows ``base_model`` and the request reads ``model_override``:
    pressing "Use turbo" has to move both, and run the same clearing the combo's
    own pick runs (a structure control turbo cannot run)."""
    _findings(
        tmp_path,
        {
            "base_model": {
                "turbo": {"n": 8, "accepts": 7, "wilson_low": 0.55},
                "sdxl_cfg": {"n": 8, "accepts": 2, "wilson_low": 0.1},
            }
        },
    )
    _press_use(monkeypatch)
    form = _form(
        model_mode="advanced", base_model="sdxl_cfg", model_override="sdxl_cfg", control="canny"
    )
    ctx = _catalog_ctx()
    ctx.svc = SimpleNamespace(config=SimpleNamespace(bench_dir=tmp_path))
    ctx.state.form_2d = form

    settings_2d._best_value_offer(ctx, form, "base_model", form["base_model"])

    assert form["base_model"] == "turbo"
    assert form["model_override"] == "turbo", "the request would still run the old model"
    assert generation.request_from_legacy(form).model_override == "turbo"
    assert form["control"] == "", "turbo cannot run a ControlNet; the combo's own pick clears it"
    assert any(
        "structure control was cleared" in n for n in ctx.state.preview[settings_2d.CLEARED_KEY]
    )


def test_use_button_on_style_lora_reseeds_the_tuned_weight(tmp_path, monkeypatch):
    _findings(
        tmp_path,
        {
            "style_lora": {
                "pixelxl": {"n": 8, "accepts": 7, "wilson_low": 0.55},
                "render3d": {"n": 8, "accepts": 2, "wilson_low": 0.1},
            }
        },
    )
    _press_use(monkeypatch)
    form = _form(style_lora="render3d", lora_weight=models.STYLE_LORAS["render3d"].default_weight)
    ctx = _catalog_ctx()
    ctx.svc = SimpleNamespace(config=SimpleNamespace(bench_dir=tmp_path))
    ctx.state.form_2d = form

    settings_2d._best_value_offer(ctx, form, "style_lora", form["style_lora"])

    assert form["style_lora"] == "pixelxl"
    assert form["lora_weight"] == create_recipe.lora_default_weight("pixelxl")


# --- create-22 -----------------------------------------------------------------


def _sprite_form(**over):
    form = _form(
        output="sheet",
        sheet_type="sprite",
        asset_type="sprite_sheet",
        generation_type="sprite_sheet",
        target_cell_px="32",
    )
    form.update(over)
    return form


def test_every_key_the_sprite_block_sends_survives_the_door_or_is_not_sent(monkeypatch):
    """What ``_check_sprite_sheet`` returns is the follow-up's whole request:
    ``sheet_type``, ``candidates`` and ``check_pixel_options``' output
    (``logical_size``, ``colors``, ``outline``, ``dither``, ``palette``). A key
    the block sends and the door drops is a control that does nothing and says
    nothing -- the 2026-10-04 audit's create-22, ``target_cell_px``."""
    from realmspinner.service import _jobs_create as jobs_create
    from realmspinner.service import sprites as svc_sprites

    monkeypatch.setattr(svc_sprites, "_check_weights", lambda svc: None)
    monkeypatch.setattr(jobs_create, "check_vram", lambda *a, **k: None)

    block = create_recipe.sprite_sheet_kwargs(_sprite_form())
    kept = jobs_create._check_sprite_sheet(None, block)

    dropped = sorted(set(block) - set(kept))
    assert not dropped, f"sent to the door and thrown away there: {dropped}"


def test_the_sprite_arm_draws_no_cell_target_and_does_not_validate_one():
    """The control is hidden where the door drops its value, and a stale value
    left in the form from the tile arm cannot refuse a press on a control that
    is not on screen."""
    source = inspect.getsource(settings_2d.draw)
    sprite_branch = source.split('widgets.section("Sprite sheet")', 1)[1].split(
        'widgets.section("Recipe")', 1
    )[0]
    assert "_target_cell" not in sprite_branch
    form = _sprite_form(target_cell_px=str(generation.TARGET_CELL_MAX + 1))
    assert not [p for p in create_recipe.validate(form) if p.field == "target_cell_px"]


# --- create-24 -----------------------------------------------------------------


def test_switching_to_automatic_clears_a_style_lora_the_resolved_base_cannot_take():
    """Config None resolves to ``sdxl_cfg``, which a Klein adapter is not fitted
    to: the Automatic branch has to say so and clear it, as picking the model
    under Advanced does."""
    form = _form(
        style_lora="pixelklein",
        lora_weight=0.0625,
        model_mode="auto",
        base_model="flux_klein",
        model_override="",
    )
    ctx = _catalog_ctx()
    notes = create_recipe.clear_for_tier(ctx, form)
    assert form["style_lora"] == ""
    assert form["lora_weight"] == models.DEFAULT_LORA_WEIGHT
    assert any("style LoRA was cleared" in n for n in notes), notes
    assert not [p for p in create_recipe.validate(form, ctx) if p.field == "style_lora"]


def test_switching_to_automatic_clears_a_start_image_the_resolved_base_cannot_take(monkeypatch):
    flux = "flux_klein_distilled"
    assert models.BASE_MODELS[flux].family != models.FAMILY_SDXL
    recipe = generation.Recipe(
        key="x",
        label="X",
        generation_types=("image", "3d_model"),
        quality="fast",
        base_model=flux,
        negative_prompt=False,
    )
    resolved = generation.ResolvedRecipe(recipe=recipe, base_model=flux, style_lora=None)
    monkeypatch.setattr(create_recipe, "resolved_recipe", lambda ctx, form: resolved)
    form = _form(model_mode="auto", init_image=True, ref_path="x.png")
    notes = create_recipe.clear_for_tier(_catalog_ctx(), form)
    assert form["init_image"] is False
    assert any("start image was cleared" in n for n in notes), notes


# --- create-25 -----------------------------------------------------------------


def test_structure_picker_is_offered_under_automatic_even_when_base_model_is_stale():
    """Advanced (distilled) then Automatic leaves ``base_model`` at the distilled
    key; Automatic resolves to a ControlNet-capable base, so the picker stays."""
    form = _form(
        ref_path="x.png",
        control="canny",
        model_mode="auto",
        base_model="turbo",
        model_override="",
    )
    ctx = _catalog_ctx()
    assert create_recipe.recipe_structure_note(ctx, form) is None
    assert create_recipe.structure_picker_note(ctx, form) is None


def test_structure_picker_still_hides_under_advanced_on_a_distilled_base():
    form = _form(
        ref_path="x.png", model_mode="advanced", base_model="turbo", model_override="turbo"
    )
    assert "full-CFG" in create_recipe.structure_picker_note(_catalog_ctx(), form)


# --- create-27 and create-62 (conditioning_tail) -------------------------------


def test_a_vanished_reference_also_clears_its_conditioning_selections(tmp_path):
    form = _form(
        ref_path=str(tmp_path / "gone.png"), ip_adapter="plus", control="canny", init_image=True
    )
    ctx = _catalog_ctx()
    ctx.state.create.reference_path_checked = False
    toasts = []
    ctx.toast = lambda *a, **k: toasts.append(a)

    create_recipe.verify_reference_path(ctx, form)

    assert form["ref_path"] == ""
    assert form["ip_adapter"] == ""
    assert form["control"] == ""
    assert form["init_image"] is False
    assert create_recipe.conditioning_tail(form) == ""
    assert toasts


def test_conditioning_tail_counts_only_selections_with_a_reference():
    """The header says how many controls are live; with no reference the pickers
    are hidden and Generate refuses over them, so they are not "on"."""
    assert create_recipe.conditioning_tail(_form()) == ""
    stale = _form(ref_path="", ip_adapter="plus", control="canny", init_image=True)
    assert create_recipe.conditioning_tail(stale) == ""
    assert create_recipe.conditioning_tail(_form(ref_path="x.png")) == "  (1 on)"
    full = _form(ref_path="x.png", ip_adapter="plus", control="canny", init_image=True)
    assert create_recipe.conditioning_tail(full) == "  (4 on)"


# --- create-35 -----------------------------------------------------------------


def test_a_refused_generate_keeps_the_unlocked_seed(monkeypatch):
    form = _form(seed=1234, seed_locked=False)
    monkeypatch.setattr(
        generation,
        "validate_request",
        lambda request, resolved: [generation.CompatibilityIssue("base_model", "refused")],
    )
    toasts = []
    ctx = _catalog_ctx(submit=lambda *a, **k: True)
    ctx.toast = lambda *a, **k: toasts.append(a)

    settings_2d.generate(ctx, form)

    assert toasts, "the recipe refusal should have been shown"
    assert form["seed"] == 1234, "a refused press must not spend the seed"


# --- create-52 -----------------------------------------------------------------


def test_the_conditioning_section_loads_findings_once_per_frame(tmp_path, monkeypatch):
    from _ui_context import imgui_context

    _findings(tmp_path, {"ip_scale": {"0.6": {"n": 8, "accepts": 2, "wilson_low": 0.1}}})
    loads = []
    real = findings_lib.load
    monkeypatch.setattr(
        settings_2d.findings_lib, "load", lambda *a, **k: (loads.append(1), real(*a, **k))[1]
    )
    form = _form(
        ref_path="x.png",
        ip_adapter="plus",
        control="canny",
        init_image=True,
        model_mode="advanced",
        base_model="sdxl_cfg",
    )
    ctx = _catalog_ctx(busy=lambda key: False, submit=lambda *a, **k: True, model_rows=[])
    ctx.svc = SimpleNamespace(config=SimpleNamespace(bench_dir=tmp_path))
    ctx.state.form_2d = form
    doc = real(tmp_path / "findings.json")
    loads.clear()

    with imgui_context(monkeypatch) as imgui:
        imgui.new_frame()
        imgui.begin("test")
        settings_2d._references(ctx, form, doc)
        imgui.end()
        imgui.render()

    assert loads == [], (
        f"the section re-loaded findings {len(loads)} times instead of using the frame's"
    )


# --- create-53 -----------------------------------------------------------------


def test_the_cleared_note_goes_when_the_user_edits_the_field_it_names(monkeypatch):
    ctx = _catalog_ctx()
    form = _form(output="sheet", sheet_type="tile", tile_size="48")
    key = settings_2d.TILE_MODE_CLEARED_KEY

    class _Ui:
        def segmented_choice(self, *a, **k):
            return True, "16"

    ctx.state.preview[key] = ["The tile size moved to 32 px: ..."]
    settings_2d._tile_size(ctx, form, _Ui())
    assert key not in ctx.state.preview


def test_the_model_cleared_note_goes_when_the_user_picks_a_style(monkeypatch):
    ctx = _catalog_ctx()
    form = _form(model_mode="advanced", base_model="sdxl_cfg", model_override="sdxl_cfg")
    ctx.state.preview[settings_2d.CLEARED_KEY] = ["The style LoRA was cleared: ..."]
    for name in ("field_label", "field_error", "muted_wrapped"):
        monkeypatch.setattr(settings_2d.widgets, name, lambda *a, **k: None)
    monkeypatch.setattr(settings_2d, "_hint", lambda *a, **k: None)
    monkeypatch.setattr(settings_2d.widgets, "combo", lambda *a, **k: "pixelxl")

    settings_2d._lora(ctx, form, show_strength=False)

    assert settings_2d.CLEARED_KEY not in ctx.state.preview


# --- create-54 -----------------------------------------------------------------


def test_the_inert_negative_note_does_not_claim_guidance_zero_for_lcm(monkeypatch):
    """LCM and FLUX.2 klein distilled run at guidance 1.0, and the manual says
    "guidance 1.0 or lower"."""
    assert models.BASE_MODELS["pixel"].guidance_scale == 1.0
    ctx = _catalog_ctx()
    for key in ("turbo", "pixel", "flux_klein_distilled"):
        form = _form(model_mode="advanced", base_model=key, model_override=key)
        note = create_recipe.negative_prompt_note(ctx, form)
        assert note is not None
        assert "guidance 0" not in note, note
        assert "guidance 1.0 or lower" in note

    # The two sentences ``clear_for_tier`` files for a recipe at guidance 1.0 or
    # lower, with the resolved recipe pinned to one.
    recipe = generation.Recipe(
        key="x",
        label="X",
        generation_types=("image", "3d_model"),
        quality="fast",
        base_model="flux_klein_distilled",
        negative_prompt=False,
    )
    resolved = generation.ResolvedRecipe(
        recipe=recipe, base_model="flux_klein_distilled", style_lora=None
    )
    monkeypatch.setattr(create_recipe, "resolved_recipe", lambda ctx, form: resolved)
    form = _form(model_mode="auto", control="canny", negative_prompt="no hats")
    cleared = create_recipe.clear_for_tier(ctx, form)
    assert len(cleared) >= 2, cleared
    assert all("guidance 0" not in n for n in cleared), cleared
    assert "runs at guidance 0" not in inspect.getsource(create_recipe)


# --- create-62 -----------------------------------------------------------------


def test_seamless_subject_is_gone():
    assert not hasattr(create_recipe, "seamless_subject")


def test_advisory_fix_appends_the_closed_form_clause_once(monkeypatch):
    monkeypatch.setattr(settings_2d.controls, "button", lambda *a, **k: True)
    ctx = _catalog_ctx()
    ctx.state.note_field_error("prompt", "x")
    form = _form(prompt="a lace fan, ")

    settings_2d._advisory_fix(ctx, form, problem_types.Advisory("open form", "prompt"))
    assert form["prompt"] == f"a lace fan, {create_recipe.CLOSED_FORM_CLAUSE}"
    assert "prompt" not in ctx.state.field_errors

    # Already present: no second append, and no button is even drawn.
    pressed = []
    monkeypatch.setattr(settings_2d.controls, "button", lambda *a, **k: pressed.append(1) or True)
    before = form["prompt"]
    settings_2d._advisory_fix(ctx, form, problem_types.Advisory("open form", "prompt"))
    assert form["prompt"] == before and not pressed


def test_advisory_fix_offers_nothing_for_an_advisory_about_another_field(monkeypatch):
    pressed = []
    monkeypatch.setattr(settings_2d.controls, "button", lambda *a, **k: pressed.append(1) or True)
    form = _form(prompt="a lace fan")
    settings_2d._advisory_fix(_catalog_ctx(), form, problem_types.Advisory("x", "seed"))
    assert form["prompt"] == "a lace fan" and not pressed
