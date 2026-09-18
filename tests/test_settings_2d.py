"""Stale conditioning-slider refusals ring their own control.

The 2026-09-16 audit, finding create-panes-01: ``settings_2d.py`` never wired
``widgets.field_error``/``clear_field_error`` for "ip_scale", "control_scale",
"control_end" or "init_strength", and ``validate()`` never range-checked any
of the four before Generate is enabled -- unlike "style_lora"'s weight, which
gets both a ``validate()`` range check and a ring. ``guidance.normalize``'s
``_number`` (``src/warlock/guidance.py``) refuses exactly these four names,
so a persisted form carrying an out-of-range value reached the queue door
with the Conditioning section still collapsed and the refusal had no control
on this pane to land on. Same shape of gap ``tests/test_settings_seed.py``
proved for "seed".

The second finding in the same slice (create-panes-02): the Style LoRA
strength slider hardcoded its bounds as the literal ``0.0, 1.5`` instead of
going through ``_range(ctx, "lora_weight_range", 0.0, 1.5)``, the pattern
every sibling numeric control in this file follows.
"""

from __future__ import annotations

import inspect

from warlock import models
from warlock.studio.modes.create.engine import recipe as create_recipe
from warlock.studio.modes.create.ui.panes import settings_2d
from warlock.studio.state import default_form_2d

#: The four fields ``guidance._number`` (src/warlock/guidance.py:299-313,
#: 457-488) refuses by exactly this name.
_CONDITIONING_FIELDS = ("ip_scale", "control_scale", "control_end", "init_strength")


def _source() -> str:
    return inspect.getsource(settings_2d)


def test_a_stale_control_scale_refusal_rings_the_conditioning_slider():
    """Both halves of the wiring, the same contract every other refusable
    control on this pane already keeps (``tests/test_field_error_wiring.py``,
    ``tests/test_settings_seed.py``): the control rings when a refusal names
    it, and editing the control clears last time's ring."""
    source = _source()
    for field in _CONDITIONING_FIELDS:
        assert f'field_error(ctx.state, "{field}")' in source, (
            f"settings_2d never rings the {field} control on a refusal"
        )
        assert f'clear_field_error("{field}")' in source, (
            f"settings_2d never clears the {field} ring when its own control "
            "is edited"
        )


def _unpinned_form() -> dict:
    form = dict(default_form_2d())
    form["prompt"] = "a knight"
    return form


def test_validate_range_checks_the_conditioning_sliders_before_generate():
    """The aggregate block above Generate (``create_recipe.validate``) has to
    catch an out-of-range slider before the round trip through
    ``guidance.normalize`` -- the same thing it already does for
    ``lora_weight`` (see the check right beside these in ``validate``)."""
    form = _unpinned_form()
    form["ip_adapter"] = "concept"
    form["ref_path"] = "ref.png"
    form["ip_scale"] = models.IP_SCALE_MAX + 1
    problems = create_recipe.validate(form)
    assert any(p.field == "ip_scale" for p in problems), problems

    form = _unpinned_form()
    form["control"] = "canny"
    form["ref_path"] = "ref.png"
    form["control_scale"] = models.CONTROL_SCALE_MAX + 1
    problems = create_recipe.validate(form)
    assert any(p.field == "control_scale" for p in problems), problems

    form = _unpinned_form()
    form["control"] = "canny"
    form["ref_path"] = "ref.png"
    form["control_end"] = models.CONTROL_END_MIN - 1
    problems = create_recipe.validate(form)
    assert any(p.field == "control_end" for p in problems), problems

    form = _unpinned_form()
    form["init_image"] = True
    form["ref_path"] = "ref.png"
    form["init_strength"] = models.IMG2IMG_STRENGTH_MAX + 1
    problems = create_recipe.validate(form)
    assert any(p.field == "init_strength" for p in problems), problems


def test_an_in_range_conditioning_form_is_not_flagged_by_the_new_checks():
    """The four checks above must not fire on the ordinary case -- a slider
    left at its default is never a refusal."""
    form = _unpinned_form()
    form["ip_adapter"] = "concept"
    form["control"] = "canny"
    form["init_image"] = True
    form["ref_path"] = "ref.png"
    problems = create_recipe.validate(form)
    assert {p.field for p in problems}.isdisjoint(_CONDITIONING_FIELDS), problems


def test_lora_strength_slider_bounds_come_from_the_guidance_catalog():
    """``_lora_strength`` used to hardcode ``0.0, 1.5`` instead of going
    through ``_range(ctx, "lora_weight_range", 0.0, 1.5)`` -- the bounds the
    service will actually enforce, so this slider can never produce a value
    the submit rejects."""
    source = inspect.getsource(settings_2d._lora_strength)
    assert '_range(ctx, "lora_weight_range", 0.0, 1.5)' in source, (
        "settings_2d._lora_strength still hardcodes its slider bounds as a "
        "literal instead of reading them from the guidance catalog"
    )
