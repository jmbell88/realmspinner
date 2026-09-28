"""A stale ``target_cell_px`` refusal has to ring its own control.

The 2026-09-26 audit, finding create-panes-05: ``generation.validate_target_cell``
(read into ``create_recipe.validate`` at ``engine/recipe.py:1388-1392``) refuses
an out-of-range tile/sprite cell size under ``field="target_cell_px"``, but
``settings_2d._target_cell`` never called ``widgets.field_error``/
``clear_field_error`` for that name -- the preset combo and the custom-size
number box were both drawn with no ring wiring at all, unlike every other
refusable control on this pane (``ip_scale``, ``control_scale``, ``control_end``,
``init_strength``, "base_model", "seed", ... -- ``test_settings_2d.py``'s
``test_a_stale_control_scale_refusal_rings_the_conditioning_slider`` proved the
same gap for the conditioning sliders under create-panes-01). A persisted form
carrying an out-of-range custom cell size reached this pane with nothing on
screen pointing at the control that held it.
"""

from __future__ import annotations

import inspect

from realmspinner import generation
from realmspinner.studio.modes.create.engine import recipe as create_recipe
from realmspinner.studio.modes.create.ui.panes import settings_2d
from realmspinner.studio.state import default_form_2d


def _source() -> str:
    return inspect.getsource(settings_2d)


def test_a_stale_target_cell_refusal_rings_the_cell_size_control():
    """Both halves of the wiring, the same contract create-panes-01 already
    proved for the conditioning sliders: the control rings when a refusal
    names it, and editing the control clears last time's ring."""
    source = _source()
    assert 'field_error(ctx.state, "target_cell_px")' in source, (
        "settings_2d never rings the target_cell_px control on a refusal"
    )
    assert 'clear_field_error("target_cell_px")' in source, (
        "settings_2d never clears the target_cell_px ring when its own "
        "control is edited"
    )


def test_validate_still_refuses_an_out_of_range_target_cell_by_that_name():
    """The refusal this ring answers to is real: an out-of-range custom cell
    size on a tileset request is refused under exactly this field name."""
    form = dict(default_form_2d())
    form["prompt"] = "a knight"
    form["output"] = "sheet"
    form["sheet_type"] = "sprite"
    form["target_cell_px"] = str(generation.TARGET_CELL_MAX + 1)
    problems = create_recipe.validate(form)
    assert any(p.field == "target_cell_px" for p in problems), problems
