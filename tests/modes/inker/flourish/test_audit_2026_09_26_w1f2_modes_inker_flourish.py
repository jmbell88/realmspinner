"""Regression for the 2026-09-26 audit's inker-flourish-01 finding."""

from __future__ import annotations

from realmspinner.kernels.pixel.flourish import keywords, presets


def _fireball():
    return presets.load("fireball")


def _layer(rec, kind, name=None):
    for layer in rec.layers:
        if layer.kind == kind and (name is None or layer.name == name):
            return layer
    raise KeyError(kind)


def test_green_fire_recolours_the_flame_green_and_no_fire_only_hides_it():
    """inker-flourish-01: "fire" is both a colour word (``COLOURS["fire"]``)
    and a layer-kind noun (``_KIND_WORDS["flame"]`` names it), so the colour
    loop in ``apply`` fired a *second* time on "fire" itself after already
    correctly colouring the flame layer through the word before it -- found
    no kind word after it (there is none left), read that as "no target",
    and repainted every coloured layer in the recipe orange. The same
    happened for "no fire" (which should only hide the flame layer) and
    "bigger fire" (which should not touch colour at all).
    """
    before = _fireball()

    rec, notes = keywords.apply(before, "green fire")
    assert _layer(rec, "flame").params["color_tip"] == keywords.COLOURS["green"][1]
    assert _layer(rec, "core", "Core").params == _layer(before, "core", "Core").params
    assert (
        _layer(rec, "particles", "Embers").params
        == _layer(before, "particles", "Embers").params
    )
    assert not any("fire:" in n for n in notes), notes

    rec2, _notes2 = keywords.apply(before, "no fire")
    assert _layer(rec2, "flame").visible is False
    assert _layer(rec2, "core", "Core").params == _layer(before, "core", "Core").params
    assert (
        _layer(rec2, "particles", "Embers").params
        == _layer(before, "particles", "Embers").params
    )

    # "bigger" legitimately scales every layer's size params (it is not
    # scoped to a kind at all) -- the claim here is only that "fire" tacked
    # onto it must not *also* repaint every coloured layer, so only the
    # colour-bearing params are checked.
    rec3, _notes3 = keywords.apply(before, "bigger fire")
    before_core = _layer(before, "core", "Core").params
    after_core = _layer(rec3, "core", "Core").params
    assert after_core["color_inner"] == before_core["color_inner"]
    assert after_core["color_outer"] == before_core["color_outer"]
    before_embers = _layer(before, "particles", "Embers").params
    after_embers = _layer(rec3, "particles", "Embers").params
    assert after_embers["color_start"] == before_embers["color_start"]
    assert after_embers["color_end"] == before_embers["color_end"]
