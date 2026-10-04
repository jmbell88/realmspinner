"""The 2026-10-03 audit's Medium findings on Create's tile arm (plotter-19, plotter-23).

plotter-19: "Keep one style across the list" and "Erase the seam" are drawn only
under the Materials layout, so what is *sent* and what the Generate gate *asks
the host for* has to be what is drawn.

plotter-23: a style lock reaches the IP-Adapter only when the sheet has more
than one cell (the door, ``vram`` and the worker already said so), so the pane's
weight gate has to read the same effective lock.
"""

from __future__ import annotations

from realmspinner.service import tilesheets as svc_tilesheets
from realmspinner.studio.modes.create.engine import recipe as create_recipe
from realmspinner.studio.state import default_form_2d


def _sheet_form(mode: str, **overrides):
    form = default_form_2d()
    form["output"] = "sheet"
    form["sheet_type"] = "tile"
    form["tile_mode"] = mode
    form["prompt"] = "mossy stone"
    form["materials"] = "mossy stone\ncracked mud"
    form["inner_terrain"] = "grass"
    form["outer_terrain"] = "water"
    form["style_lock"] = True
    form["seam_erase"] = True
    form.update(overrides)
    return form


def _has_adapter(rows) -> bool:
    return any(row.startswith("adapter:") for row in rows)


def test_a_terrain_or_grid_sheet_does_not_carry_the_materials_checkboxes_or_demand_their_adapter():
    for mode in (svc_tilesheets.MODE_TERRAIN, svc_tilesheets.MODE_GRID):
        form = _sheet_form(mode)
        kwargs = create_recipe.tile_sheet_kwargs(form)
        assert kwargs["style_lock"] is False, mode
        assert kwargs["seam_erase"] is False, mode
        assert not _has_adapter(create_recipe.sheet_rows(form)), mode
    # And the Materials layout still carries both, so the fix is not "never".
    form = _sheet_form(svc_tilesheets.MODE_MATERIALS)
    kwargs = create_recipe.tile_sheet_kwargs(form)
    assert kwargs["style_lock"] is True and kwargs["seam_erase"] is True
    assert _has_adapter(create_recipe.sheet_rows(form))


def test_a_one_material_style_locked_sheet_does_not_ask_the_pane_for_the_adapter():
    form = _sheet_form(svc_tilesheets.MODE_MATERIALS, materials="mossy stone")
    assert not _has_adapter(create_recipe.sheet_rows(form))
    # Two draws of one line is two cells: the worker does lock those.
    form["variants"] = "2"
    assert _has_adapter(create_recipe.sheet_rows(form))
