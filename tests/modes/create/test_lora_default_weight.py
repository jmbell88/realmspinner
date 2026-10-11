"""A style LoRA's measured strength has to reach the form.

``StyleLora.default_weight`` is measured per adapter and
``guidance.normalize`` applies it only when the caller omits ``lora_weight``.
The 2D pane always sends a number, so without a seed at selection time every
adapter ran at the flat ``DEFAULT_LORA_WEIGHT`` -- which for pixel-art-klein
(retired 2026-10-11; an rslora adapter whose usable band was 0.02-0.08) was
~14x the top of its band, i.e. black frames. The rslora cases below run on a
synthetic adapter of that shape, because the mechanism outlived the entry.
"""

from __future__ import annotations

import pytest

from realmspinner import models
from realmspinner.studio.modes.create.engine import recipe as create_recipe
from realmspinner.studio.modes.create.ui.panes import settings_2d


@pytest.fixture
def rslora(monkeypatch):
    """A style LoRA whose measured strength is far below the flat default."""
    spec = models.StyleLora(
        "rslora_test", "rslora test", "rslora-test.safetensors", trigger="x", default_weight=0.0625
    )
    monkeypatch.setitem(models.STYLE_LORAS, spec.key, spec)
    return spec


def test_each_adapter_reports_its_own_tuned_weight(rslora):
    assert create_recipe.lora_default_weight("pixelxl") == 1.2
    assert create_recipe.lora_default_weight(rslora.key) == 0.0625


def test_no_selection_falls_back_to_the_flat_default():
    assert create_recipe.lora_default_weight("") == models.DEFAULT_LORA_WEIGHT
    assert create_recipe.lora_default_weight("not-an-adapter") == models.DEFAULT_LORA_WEIGHT


def test_every_registered_adapter_is_reachable_through_the_helper():
    for key, spec in models.STYLE_LORAS.items():
        assert create_recipe.lora_default_weight(key) == spec.default_weight


def test_an_rslora_default_sits_inside_its_usable_band(rslora):
    """The retired pixel-art-klein's band was measured at 0.02-0.08; the flat
    default does not sit in it, which is the whole defect."""
    assert 0.02 <= create_recipe.lora_default_weight(rslora.key) <= 0.08
    assert not 0.02 <= models.DEFAULT_LORA_WEIGHT <= 0.08


def test_the_reseed_writes_the_tuned_weight_only_when_the_selection_changed(rslora):
    """The rule itself, called directly now that both forms share it."""
    form = {"style_lora": rslora.key, "lora_weight": 0.9}
    create_recipe.reseed_lora_weight(form, "")
    assert form["lora_weight"] == 0.0625
    # A redraw with no change must not stomp a weight the user set by hand.
    form["lora_weight"] = 0.05
    create_recipe.reseed_lora_weight(form, rslora.key)
    assert form["lora_weight"] == 0.05


def test_every_form_that_edits_a_lora_seeds_its_weight():
    """The seeds live in draw paths, so this reads the source rather than a GL
    frame. One form now that the profile editor is gone -- but still written as
    a list, because the defect it guards was a *second* form shipping 0.9
    against an adapter whose band is 0.02-0.08, silently, with no error.
    """
    import inspect

    for func in (settings_2d._lora,):
        assert "reseed_lora_weight" in inspect.getsource(func), func.__name__
