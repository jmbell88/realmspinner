"""The Fast/Quality tier, and what picking Fast actually costs.

**The shipped registry has no Fast tier any more.** ``image_fast`` (SDXL 1.0 +
Hyper-SD) was retired on 2026-10-11 with the other distilled checkpoints, so
every test below that is about what a *fast* recipe does runs against the
``fast_tier`` fixture: a synthetic distilled base (``_synthetic_models``) and a
``quality="fast"`` recipe over it, installed for the one test. The behaviours
-- tiers resolve to different runs, a guidance-0 recipe has no negative prompt
and no ControlNet, the door says so -- are properties of the machinery, which
the next distilled entry will use again.

Both tiers named ``sdxl_cfg`` until 2026-08-29 -- same checkpoint, same steps,
same resolution -- so the control changed the recipe's *name* and nothing a
user could see. These are the pins that would have caught that, plus the pins
on the consequences of fixing it: Fast is ``sdxl`` (four steps, guidance 0), so
it can run neither a ControlNet nor a negative prompt, and both of those have
to be said rather than silently dropped.

Every assertion here is about *plumbing* -- which recipe resolves, what it
declares, what the door refuses. Nothing here looks at a pixel, and nothing
here would be made true by a fake.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from realmspinner import generation, models
from realmspinner.studio.modes.create.engine import recipe as create_recipe


@pytest.fixture
def fast_tier(monkeypatch):
    """A synthetic Fast recipe over a synthetic distilled base. -> the base spec."""
    from _synthetic_models import install_fast_tier

    return install_fast_tier(monkeypatch)


def _request(**over):
    base = {"generation_type": "image", "prompt": "a wooden crate"}
    base.update(over)
    return generation.GenerationRequest(**base)


def _structured_request(**over):
    """A request naming a structure control. The 2026-10-11 audit, finding
    create-09: a control with no reference to derive its hint from is refused
    on its own field, so a request that is *about* the control's other
    refusals has to carry one, as the pane's always does."""
    return _request(
        structure_control="canny", references=("ref.png",), reference_mode="single", **over
    )


def _ctx():
    """A pane ctx whose config is ``None``.

    ``generation._present`` answers True for a ``None`` config -- "no config,
    no opinion about what is downloaded" -- which is exactly what these tests
    want: the resolver picks the recipe the *registry* would choose, rather
    than refusing because a throwaway ``REALMSPINNER_HOME`` holds no weights.
    """
    return SimpleNamespace(svc=SimpleNamespace(config=None))


# --- the two tiers are two pictures -------------------------------------------


def test_the_shipped_registry_has_no_fast_tier():
    """``image_fast`` named a retired model (``sdxl``); a recipe naming a key
    the registry lacks would KeyError ``_registry_facts`` at import."""
    assert [r.key for r in generation.RECIPES if r.quality == "fast"] == []
    assert generation.resolve_recipe(_request(quality="fast"), None) is None
    for recipe in generation.RECIPES:
        assert recipe.base_model in models.BASE_MODELS, recipe.key


def test_fast_and_quality_resolve_to_different_recipes(fast_tier):
    fast = generation.resolve_recipe(_request(quality="fast"), None)
    quality = generation.resolve_recipe(_request(quality="quality"), None)
    assert fast is not None and quality is not None
    assert fast.recipe.key != quality.recipe.key
    assert fast.base_model != quality.base_model


def test_no_two_tiers_of_one_generation_type_name_the_same_base():
    """The pin the original defect would have failed.

    A tier is a *choice*, so two tiers offered for the same asset that load the
    same checkpoint at the same size are a control with nothing behind it. This
    compares the resolved run, not the recipe key: two keys were exactly what
    the defect had.
    """
    for kind in generation.GENERATION_TYPES:
        # The *highest-ranked* recipe per tier: that is the one automatic
        # routing picks, and the lower-ranked siblings are a fallback order
        # rather than a choice offered to anybody.
        by_tier: dict[str, tuple[str, tuple[int, int]]] = {}
        for recipe in sorted(generation.RECIPES, key=lambda r: r.rank):
            if kind in recipe.generation_types:
                by_tier[recipe.quality] = (recipe.base_model, recipe.working_resolution)
        runs = list(by_tier.values())
        assert len(set(runs)) == len(runs), (
            f"{kind}: two quality tiers resolve to the same run {runs}"
        )


def test_fast_names_a_distilled_recipe_over_the_shared_sdxl_weights(fast_tier):
    fast = generation.resolve_recipe(_request(quality="fast"), None)
    assert fast.base_model == fast_tier.key
    # The same base weights as Quality, run differently -- which is why a Fast
    # tier costs no second 7 GB download.
    assert fast_tier.dir_name == models.BASE_MODELS["sdxl_cfg"].dir_name
    assert fast_tier.steps < models.BASE_MODELS["sdxl_cfg"].steps
    assert fast_tier.commercial


def test_advanced_selection_of_the_fast_checkpoint_keeps_its_curated_note(fast_tier):
    """The 2026-09-07 audit, finding create-03.

    ``create_recipe.model_options`` folded the retired Fast/Quality switch into
    the Model combo on the claim that "nothing is lost by folding them" --
    picking the fast checkpoint through Advanced is supposed to be the
    same act picking Fast used to be. But nothing sets ``quality="fast"``
    any more, so a request built the way the pane actually builds one
    (``model_mode="advanced"``, ``model_override=<fast key>``, ``quality`` left
    at its default) must still resolve the curated ``image_fast`` recipe,
    not the label-less synthetic one ``resolve_recipe`` falls back to when no
    curated recipe matches.
    """
    request = _request(model_mode="advanced", model_override=fast_tier.key)
    assert request.quality == "quality"  # nothing in the pane sets "fast" any more
    resolved = generation.resolve_recipe(request, None)
    assert resolved is not None
    assert resolved.recipe.key == "image_fast"
    assert resolved.recipe.label == "Fast image"
    assert resolved.recipe.note


def test_both_tiers_say_what_they_trade(fast_tier):
    keyed = {r.key: r for r in generation.RECIPES}
    assert keyed["image_fast"].note and keyed["image_quality"].note
    note = keyed["image_fast"].note.lower()
    assert "four steps" in note
    assert "structure control" in note
    assert "negative prompt" in note


# --- the negative prompt is inert on Fast, and declared inert ------------------


def test_the_fast_recipe_reports_no_negative_prompt_support(fast_tier):
    fast = generation.resolve_recipe(_request(quality="fast"), None)
    assert not fast.recipe.supports_negative_prompt
    # And the declaration agrees with the registry's own derivation rather
    # than being an independent opinion about the same checkpoint.
    assert fast.base_model not in models.cfg_bases()


def test_the_form_does_not_offer_avoid_under_fast(fast_tier):
    form = {"asset_type": "image", "generation_type": "image", "quality": "fast"}
    assert create_recipe.negative_supported(_ctx(), form) is False
    form["quality"] = "quality"
    assert create_recipe.negative_supported(_ctx(), form) is True


def test_a_distilled_klein_accepts_a_saved_negative_prompt(monkeypatch):
    """A prior full-CFG brief must not prevent a Klein run from starting. (A
    synthetic distilled FLUX.2 entry: the shipped one was retired.)"""
    from _synthetic_models import distilled_base, install

    spec = install(
        monkeypatch,
        distilled_base(
            "klein_distilled_test",
            dir_name="flux2-klein-4b-test",
            guidance_scale=1.0,
            scheduler=None,
            base_lora=None,
            variant=None,
            family=models.FAMILY_FLUX2_KLEIN,
            residency=models.OFFLOAD,
            fetch=(),
        ),
    )
    request = _request(
        model_mode="advanced",
        model_override=spec.key,
        negative_prompt="blurry, watermark",
    )
    resolved = generation.resolve_recipe(request, None)
    assert resolved is not None
    assert generation.validate_request(request, resolved) == []
    assert generation.effective_negative_prompt(request, resolved) == ""
    assert generation.request_to_legacy(request, resolved)["negative_prompt"] is None


def test_avoid_text_typed_under_quality_is_cleared_by_switching_to_fast(fast_tier):
    """The dead end the clear exists to prevent.

    The field is hidden under Fast, and ``validate_request`` refuses a request
    that carries text it cannot use -- so without this the Generate button
    refuses on a control that is off screen.
    """
    form = {
        "asset_type": "image",
        "generation_type": "image",
        "quality": "fast",
        "negative_prompt": "blurry, watermark",
    }
    cleared = create_recipe.clear_for_tier(_ctx(), form)
    assert form["negative_prompt"] == ""
    assert any("Avoid" in one for one in cleared)


# --- Fast and structure control are incompatible ------------------------------


def test_automatic_routing_controlnet_refusal_names_a_control_that_still_exists(fast_tier):
    """The 2026-09-18 audit, finding create-05: under automatic routing this
    refusal used to name ``quality`` and say "Switch the recipe to Quality"
    -- a control that has not existed since the Fast/Quality tier folded
    into the Model combo (the same retirement create-03/5.5.2 already fixed
    the repair button for). ``base_model`` is the field ``_model``
    (settings_2d.py) actually draws and shows a field error beside, under
    both routing modes, so that is what the refusal must name."""
    request = _structured_request(quality="fast")
    resolved = generation.resolve_recipe(request, None)
    issues = generation.validate_request(request, resolved)
    assert [i.field for i in issues] == ["base_model"]
    message = issues[0].message
    assert "ControlNet" in message
    assert "Quality" not in message
    assert "full-CFG" in message


def test_the_refusal_names_the_model_when_the_user_picked_it(fast_tier):
    request = _structured_request(
        quality="fast",
        model_mode="advanced",
        model_override=fast_tier.key,
    )
    resolved = generation.resolve_recipe(request, None)
    issues = generation.validate_request(request, resolved)
    assert [i.field for i in issues] == ["base_model"]


def test_quality_plus_structure_control_is_accepted():
    request = _structured_request(quality="quality")
    resolved = generation.resolve_recipe(request, None)
    assert generation.validate_request(request, resolved) == []


def test_the_pane_hides_the_structure_picker_under_fast(fast_tier):
    form = {"asset_type": "image", "generation_type": "image", "quality": "fast"}
    note = create_recipe.recipe_structure_note(_ctx(), form)
    assert note is not None and "ControlNet" in note
    form["quality"] = "quality"
    assert create_recipe.recipe_structure_note(_ctx(), form) is None


def test_switching_to_fast_clears_a_structure_control(fast_tier):
    form = {
        "asset_type": "image",
        "generation_type": "image",
        "quality": "fast",
        "ref_path": "ref.png",
        "control": "canny",
    }
    cleared = create_recipe.clear_for_tier(_ctx(), form)
    assert form["control"] == ""
    assert any("structure control" in one for one in cleared)


def test_advanced_mode_leaves_the_tier_clear_alone():
    """``clear_unusable`` owns that half: under Advanced the tier does not pick
    the checkpoint, so it cannot strand anything."""
    form = {
        "asset_type": "image",
        "generation_type": "image",
        "quality": "fast",
        "model_mode": "advanced",
        "base_model": "sdxl_cfg",
        "ref_path": "ref.png",
        "control": "canny",
    }
    assert create_recipe.clear_for_tier(_ctx(), form) == []
    assert form["control"] == "canny"


def test_a_form_carries_its_control_into_the_request_only_with_a_reference():
    form = {"asset_type": "image", "generation_type": "image", "control": "canny"}
    assert generation.request_from_legacy(form).structure_control == ""
    form["ref_path"] = "ref.png"
    assert generation.request_from_legacy(form).structure_control == "canny"


def test_request_from_legacy_does_not_raise_on_non_numeric_saved_scalars():
    """The 2026-09-26 audit, finding create-workspace-04: ``request_from_legacy``
    cast ``seed``, ``count`` and ``init_strength`` with bare ``int()``/``float()``
    calls, unlike ``from_dict``'s equivalent fields (already routed through
    ``_required_int``/``_required_float`` for exactly this reason -- create-01,
    2026-09-13). A non-numeric saved value in any of them raised a raw
    ``ValueError`` out of this adapter instead of surviving to be refused by
    ``validate_request``'s existing type checks on all three fields.
    """
    form = {
        "asset_type": "image",
        "generation_type": "image",
        "seed": "banana",
        "count": "banana",
        "ref_path": "ref.png",
        "init_image": True,
        "init_strength": "banana",
    }
    request = generation.request_from_legacy(form)  # must not raise
    issues = generation.validate_request(request)
    assert {"seed", "count", "init_strength"} <= {i.field for i in issues}


def test_request_from_legacy_does_not_raise_on_a_non_numeric_saved_variants():
    """The same finding (create-workspace-04), for ``tile.variants``:
    ``request_from_legacy`` cast it with a bare ``int()`` where ``from_dict``'s
    ``TileSettings`` construction already used ``_required_int`` (create2-07,
    2026-09-06)."""
    form = {
        "asset_type": "tileset",
        "generation_type": "tileset",
        "tile_mode": "collection",
        "prompt_items": ["grass"],
        "variants": "banana",
    }
    request = generation.request_from_legacy(form)  # must not raise
    issues = generation.validate_request(request)
    assert any(i.field == "tile.variants" for i in issues)


# --- img2img needs the SDXL family ---------------------------------------------


def _non_sdxl_base() -> str:
    return next(
        key for key, spec in models.BASE_MODELS.items()
        if spec.family != models.FAMILY_SDXL
    )


def test_validate_request_refuses_init_image_on_a_non_sdxl_base():
    """The 2026-09-05 audit, finding create-04: nothing before the queue door
    checked ``init_image`` against the base model's family. A request built
    under Advanced with a non-SDXL base and img2img ticked used to pass every
    pre-submit check here and fail only at ``guidance.normalize``, late, with
    the checkpoint already resident.
    """
    request = _request(
        model_mode="advanced",
        model_override=_non_sdxl_base(),
        references=("some/ref.png",),
        reference_mode="single",
        init_image=True,
        init_strength=0.5,
    )
    resolved = generation.resolve_recipe(request, None)
    assert resolved is not None
    issues = generation.validate_request(request, resolved)
    assert [i.field for i in issues] == ["init_image"]
    # The same two words the probe checks for in guidance.normalize's own
    # refusal, so the early and late refusals agree about what happened.
    assert "image" in issues[0].message and "start" in issues[0].message


def test_an_sdxl_base_with_init_image_is_accepted():
    request = _request(
        model_mode="advanced",
        model_override="sdxl_cfg",
        references=("some/ref.png",),
        reference_mode="single",
        init_image=True,
        init_strength=0.5,
    )
    resolved = generation.resolve_recipe(request, None)
    assert resolved is not None
    assert generation.validate_request(request, resolved) == []


def test_validate_request_refuses_init_image_with_no_reference_images():
    """The 2026-09-26 audit, finding create-workspace-01: ``request_to_legacy``
    only writes ``init_image``/``init_strength`` onto the legacy job payload
    inside its ``if request.references:`` branch, so a request built (or
    round-tripped through ``from_dict``) with ``init_image=True`` and no
    reference images used to clear every check here and then run as a plain
    text-to-image job while the stored request document still claimed
    img2img. There is no resolved recipe involved in the mistake, so this
    must be refused even before recipe resolution is attempted.
    """
    request = _request(init_image=True, init_strength=0.5)
    issues = generation.validate_request(request)
    assert any(i.field == "init_image" for i in issues)


def test_capability_controls_reports_img2img_only_for_the_sdxl_family():
    non_sdxl = generation.resolve_recipe(
        _request(model_mode="advanced", model_override=_non_sdxl_base()), None
    )
    sdxl = generation.resolve_recipe(
        _request(model_mode="advanced", model_override="sdxl_cfg"), None
    )
    assert non_sdxl is not None and sdxl is not None
    assert generation.capability_controls(_request(), non_sdxl)["img2img"] is False
    assert generation.capability_controls(_request(), sdxl)["img2img"] is True


# --- what the tier needs downloaded ---------------------------------------------


def test_fast_is_satisfied_by_the_base_row_alone(fast_tier):
    fast = next(r for r in generation.RECIPES if r.key == "image_fast")
    assert fast.required_downloads == (f"base:{fast_tier.key}",)
    resolved = generation.resolve_recipe(
        _request(quality="fast"), None, installed=set(fast.required_downloads)
    )
    assert resolved is not None and resolved.recipe.key == "image_fast"


def test_the_one_row_covers_the_step_lora_too(fast_tier):
    """Why there is no second key: a distilled entry's own fetch tuple carries
    both the shared SDXL 1.0 base and the step-distillation adapter, so naming
    the base row names the whole download."""
    from realmspinner import fetch

    entry = fetch.find(f"base:{fast_tier.key}")
    assert entry is not None
    repos = {one.repo_id for one in entry.spec.fetch}
    assert "stabilityai/stable-diffusion-xl-base-1.0" in repos
    assert "acme/test-distill" in repos


@pytest.mark.parametrize("installed", [set(), {"base:sdxl_cfg"}])
def test_a_host_without_the_fast_row_does_not_qualify_it(fast_tier, installed):
    assert generation.resolve_recipe(_request(quality="fast"), None, installed=installed) is None


def test_a_stored_fast_request_resolves_automatic_instead_of_nothing():
    """Settings and job params written before the Fast tier folded into the
    Model combo still carry ``quality="fast"``. With the Fast recipe retired
    on 2026-10-11 that tier has no recipe at all, so a rerun of such a job --
    or a Create form restored from such a settings file -- resolved to
    nothing and was refused as "no installed recipe" on a machine that has
    the default model. Both parse sites read the legacy word as Quality."""
    from_job = generation.GenerationRequest.from_dict({"quality": "fast", "prompt": "a crate"})
    from_form = generation.request_from_legacy({"quality": "fast", "prompt": "a crate"})
    assert from_job.quality == "quality"
    assert from_form.quality == "quality"
