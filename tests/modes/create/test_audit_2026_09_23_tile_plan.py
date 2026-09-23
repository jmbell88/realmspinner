"""The 2026-09-23 audit, finding create-06: the tileset plan understated the
generation count for every mode but grid.

``plan_for``'s tileset arm hardcoded ``generations = 1`` regardless of
``tile_mode``/``materials``/``variants``, while the default mode (materials)
actually runs one SDXL generation per prompt line x variant and terrain runs
two -- see ``_q_tileset._tile_set``'s ``count = len(subjects)`` and
``service/tilesheets.py``'s ``create_tile_sheet`` materials/terrain arms.
"""

from __future__ import annotations

from realmspinner.service import tilesheets as svc_tilesheets
from realmspinner.studio.modes.create.engine import assets as create_assets
from realmspinner.studio.modes.create.ui import workspace as generation_workspace
from realmspinner.studio.state import default_form_2d


def _tileset_form(*, tile_mode: str, materials: str = "", variants: str = "1") -> dict:
    form = default_form_2d()
    form["asset_type"] = form["generation_type"] = "tileset"
    create_assets.sync_legacy_fields(form)
    form["tile_mode"] = tile_mode
    form["materials"] = materials
    form["variants"] = variants
    return form


def test_tileset_plan_counts_every_material_and_variant_generation():
    # Three material lines drawn three times each is nine real SDXL passes --
    # ``asset_workflows.collection_cells`` is what ``_q_tileset._tile_set``
    # actually expands into, and the plan must agree with it.
    form = _tileset_form(
        tile_mode=svc_tilesheets.MODE_MATERIALS,
        materials="mossy stone\ncracked earth\nwet gravel",
        variants="3",
    )

    plan = generation_workspace.plan_for(form)

    assert plan.generations == 9


def test_tileset_plan_counts_terrain_as_two_generations():
    form = _tileset_form(tile_mode=svc_tilesheets.MODE_TERRAIN)
    form["inner_terrain"] = "grass"
    form["outer_terrain"] = "sand"

    plan = generation_workspace.plan_for(form)

    assert plan.generations == 2


def test_tileset_plan_counts_grid_as_one_generation():
    form = _tileset_form(tile_mode=svc_tilesheets.MODE_GRID)

    plan = generation_workspace.plan_for(form)

    assert plan.generations == 1


def test_tileset_plan_duration_scales_with_the_real_count():
    # A nine-generation materials sheet takes visibly longer than "about a
    # minute" -- the duration line has to move with the count it now reports,
    # not stay pinned to the old single-generation phrase.
    from realmspinner.service import sprites as svc_sprites

    form = _tileset_form(
        tile_mode=svc_tilesheets.MODE_MATERIALS,
        materials="mossy stone\ncracked earth\nwet gravel",
        variants="3",
    )

    plan = generation_workspace.plan_for(form)

    assert plan.duration == svc_sprites.generation_time_phrase(9)
