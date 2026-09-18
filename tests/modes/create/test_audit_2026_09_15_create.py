"""Regressions for the 2026-09-15 audit's Create findings.

create-01/create-05 are one fix (the orchestrator's own framing): a
pre-registry job's ``asset_type`` can only be recovered from its ``stage``,
which ``asset_type_from_params`` always accepted but no caller ever passed --
and one of its two accepted spellings was never a real stage at all.
create-03 covers the Model combo's Automatic switch leaving ``validate``
comparing against a stale ``base_model``.
"""

from __future__ import annotations

from types import SimpleNamespace

from warlock.studio.modes.create.engine import assets as create_assets
from warlock.studio.modes.create.engine import recipe as create_recipe
from warlock.studio.modes.library.ui.panes import library
from warlock.studio.state import default_form_2d


def _copy_ctx():
    """The library's own fixture (``test_studio_plumbing.py``'s ``_copy_ctx``),
    duplicated here rather than imported since that is a sibling test module,
    not a library this one may depend on."""
    return SimpleNamespace(
        state=SimpleNamespace(
            form_2d=default_form_2d(),
            mode="create",
            previous_mode="create",
            mode_observed="create",
            create=SimpleNamespace(stage="mesh"),
            selected=None,
        ),
        toast=lambda _text: None,
    )


def test_asset_type_from_params_stage_fallback_is_reachable():
    """create-05: nothing passed ``stage`` into ``asset_type_from_params``, so
    its stage branches were dead, and one of the two spellings it matched --
    ``"tile_sheet"`` -- is the tile-sheet job's *kind*
    (``svc.store.create("tile_sheet", ...)``), never its ``stage``: the row's
    stage is written two lines above that call as ``stage="tilesheet"``
    (``service/tilesheets.py``). ``library.copy_settings`` now passes the
    job's real stage through ``state.form_from_params``, which is what makes
    this reachable at all -- and only the spelling a job row can actually
    carry may match.
    """
    assert create_assets.asset_type_from_params({}, stage="tile") == "seamless_material"
    assert create_assets.asset_type_from_params({}, stage="tilesheet") == "tileset"
    # The kind's own spelling must not match a stage question: matching it is
    # how create-01's tileset answer got quietly overwritten in the first
    # place (see the test below).
    assert create_assets.asset_type_from_params({}, stage="tile_sheet") == ""


def _legacy_tilesheet_job():
    return {
        "stage": "tilesheet",
        "kind": "tile_sheet",
        "params": {
            "seed": 7,
            "sheet": {
                "version": 3,
                "mode": "materials",
                "tile_w": 64,
                "projection": "top_down",
                "variants": 2,
                "materials": [
                    {"index": 0, "prompt": "grass", "variant": 1, "seed": 1},
                    {"index": 1, "prompt": "grass", "variant": 2, "seed": 2},
                    {"index": 2, "prompt": "dirt", "variant": 1, "seed": 3},
                    {"index": 3, "prompt": "dirt", "variant": 2, "seed": 4},
                ],
            },
        },
    }


def test_copying_a_legacy_tilesheet_job_keeps_its_tileset_type():
    """create-01: a pre-registry job carries no ``asset_type`` in ``params``,
    so ``copy_settings`` fell back to ``legacy_asset_type(form)`` -- which
    only recognises ``form["output"] == "tile"`` or ``"sheet"``, and the
    ``output`` restored two lines above it only ever becomes ``"tile"`` for
    ``job["stage"] == "tile"``. A ``tilesheet``-stage job's ``output``
    therefore came back ``"reference"``, ``legacy_asset_type`` fell through to
    its "3d_model" default, and every material/variant/layout field the copy
    exists to restore was silently dropped along with it.
    """
    ctx = _copy_ctx()
    job = _legacy_tilesheet_job()
    assert "asset_type" not in job["params"]

    library.copy_settings(ctx, job)

    form = ctx.state.form_2d
    assert form["asset_type"] == "tileset"
    assert form["generation_type"] == "tileset"
    # The rest of the sheet block that the wrong asset_type would have made
    # irrelevant -- ``sync_legacy_fields`` and ``_restore_sheet_block`` still
    # did their jobs, so proving this is proving the whole copy survived.
    assert form["tile_mode"] == "materials"
    assert form["materials"] == "grass\ndirt"
    assert form["variants"] == "2"
    assert form["tile_size"] == "64"


def _ctx_resolving():
    """What ``create_recipe.resolved_recipe`` needs: ``ctx.svc.config`` --
    ``None`` reads as "nothing downloaded, no opinion", matching
    ``test_settings_2d_notes.py``'s own ``_ctx_resolving``."""
    return SimpleNamespace(svc=SimpleNamespace(config=None))


def test_validate_agrees_with_the_resolved_recipe_after_switching_to_automatic(monkeypatch):
    """create-03: ``_model()``'s switch-to-Automatic branch (``picked ==
    ""``) sets ``model_mode = "auto"`` and moves the displayed notes onto
    ``_resolved_recipe(ctx, form)`` -- but leaves ``form["base_model"]`` at
    whatever was picked under Advanced. ``validate`` compared ControlNet,
    img2img and style-LoRA fit against that stale field, so a mismatch with
    what Automatic will actually load passed silently here and only
    surfaced as a toast refusal after the round trip through the queue door.

    ``_resolved_recipe`` is stubbed rather than steered through the real
    recipe table: the real table always resolves the default quality tier to
    ``sdxl_cfg`` regardless of installed state (``config=None`` reads
    everything as present), so there is no real request that leaves ``base``
    and ``resolved.base_model`` disagreeing without depending on the
    registry's current ranking. The claim under test is only that ``validate``
    asks ``_resolved_recipe`` at all once it has a ``ctx``.
    """
    monkeypatch.setattr(
        create_recipe,
        "resolved_recipe",
        lambda ctx, form: SimpleNamespace(base_model="flux_klein"),
    )

    form = default_form_2d()
    form["prompt"] = "a lantern"
    form["model_mode"] = "auto"
    # Stale from an Advanced pick that fit the SDXL base the combo no longer
    # shows -- "render3d" fits sdxl_cfg but not the non-SDXL flux_klein this
    # test's Automatic resolves to.
    form["base_model"] = "sdxl_cfg"
    form["style_lora"] = "render3d"

    ctx = _ctx_resolving()
    problems = create_recipe.validate(form, ctx)

    assert any(p.field == "style_lora" for p in problems), problems
    # And the pre-fix reading (no ctx, or a ctx nothing asks) still validates
    # against the raw field, so the same form passes -- proving the
    # disagreement really is the resolved-vs-stale base, not a broken form.
    assert not any(p.field == "style_lora" for p in create_recipe.validate(form))
