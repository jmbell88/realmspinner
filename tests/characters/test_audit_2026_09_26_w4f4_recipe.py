"""poser-characters-02/03/05 (2026-09-26 audit): ``recipe.py`` clamped instead
of refusing, and a species swap re-validated the old species' answers.

``_integer`` promised "it refuses; it never clamps" (``recipe.py``'s own
docstring) but ``int(raw)`` alone clamped a bool to 0/1, truncated a fraction,
and let ``int(float("inf"))``'s raw ``OverflowError`` escape this module's
``CharacterError`` vocabulary entirely. ``dither=bool(raw.get("dither", False))``
had the sibling bug ``service.troupe._check_options`` already named for its own
``pixel_art`` switch: ``bool("false")`` is ``True`` in Python. ``Recipe.replace``
re-validated a species swap's *old* appearance/theme against the *new*
species' registry row, which either raised a confusing "no such slider" (an
appearance key the new species does not define) or kept validating fine while
meaning something else (a theme name the new species happens to share).
"""

from __future__ import annotations

import pytest

from realmspinner.characters import CharacterError, Recipe
from realmspinner.characters.family import families


def test_an_infinite_or_overflowing_number_is_refused_with_its_field():
    with pytest.raises(CharacterError) as excinfo:
        Recipe.from_dict({"family": "ogre", "seed": float("inf")})
    assert excinfo.value.field == "seed"

    with pytest.raises(CharacterError) as excinfo:
        Recipe.from_dict({"family": "ogre", "seed": float("-inf")})
    assert excinfo.value.field == "seed"


def test_a_fractional_or_boolean_number_is_refused_not_truncated():
    with pytest.raises(CharacterError) as excinfo:
        Recipe.from_dict({"family": "ogre", "seed": 3.9})
    assert excinfo.value.field == "seed"

    with pytest.raises(CharacterError) as excinfo:
        Recipe.from_dict({"family": "ogre", "directions": True})
    assert excinfo.value.field == "directions"


def test_dither_string_false_is_refused_not_read_as_true():
    with pytest.raises(CharacterError) as excinfo:
        Recipe.from_dict({"family": "ogre", "dither": "false"})
    assert excinfo.value.field == "dither"
    # A real bool still answers, exactly as before.
    assert Recipe.from_dict({"family": "ogre", "dither": True}).dither is True
    assert Recipe.from_dict({"family": "ogre", "dither": False}).dither is False
    # Absence still defaults to off, byte-identical to a recipe that never
    # touched the control.
    assert Recipe.from_dict({"family": "ogre"}).dither is False


def test_integer_still_accepts_an_ordinary_whole_number_or_numeric_string():
    r = Recipe.from_dict({"family": "ogre", "seed": "7", "directions": 4})
    assert (r.seed, r.directions) == (7, 4)


def test_replace_swapping_family_drops_the_old_appearance_instead_of_revalidating_it():
    """"2:1 dimetric" isn't the subject here -- this is poser-characters-05.

    Ogre (humanoid) has a ``shoulder_width`` channel wolf (quadruped) does
    not. ``as_dict`` always carries a recipe's *whole* appearance dict (every
    channel, not only the ones a caller touched), so swapping family through
    ``replace`` used to always carry at least one channel key the new species
    does not define into ``from_dict``'s validation -- "wolf has no
    'shoulder_width' slider" instead of the swap the caller asked for.
    """
    reg = families()
    assert "shoulder_width" in {c.key for c in reg["ogre"].channels}
    assert "shoulder_width" not in {c.key for c in reg["wolf"].channels}

    base = Recipe.from_dict({"family": "ogre", "appearance": {"shoulder_width": 0.5}})
    # "natural" is a theme both species offer, so this isolates the
    # appearance question from the (already correctly refusing) theme one
    # below.
    swapped = base.replace(family="wolf", theme="natural")

    assert swapped.family == "wolf"
    assert "shoulder_width" not in swapped.appearance


def test_replace_swapping_family_still_refuses_a_theme_the_new_species_does_not_offer():
    """Unlike appearance, a species swap must **not** silently drop or
    default the old theme -- ``test_changing_species_never_silently_carries_a_
    palette_across`` (``tests/characters/test_recipe.py``) already pins that
    refusal as the deliberate contract: a theme the new species does not
    offer is a caller choice worth being asked about again, not free-form
    geometry no other species could ever share the way an appearance channel
    is.
    """
    reg = families()
    assert "fire" not in {t.key for t in reg["wolf"].themes}

    base = Recipe.from_dict({"family": "ogre", "theme": "fire"})
    with pytest.raises(CharacterError) as excinfo:
        base.replace(family="wolf")
    assert excinfo.value.field == "theme"


def test_replace_with_an_explicit_appearance_alongside_family_uses_it():
    """The drop in ``replace`` only discards the *carried-over* answer -- an
    explicit ``appearance=`` passed in the same call as ``family=`` must
    still land, not be silently overridden back to defaults."""
    reg = families()
    wolf_channel = next(iter({c.key for c in reg["wolf"].channels}))

    base = Recipe.from_dict({"family": "ogre"})
    swapped = base.replace(family="wolf", appearance={wolf_channel: 0.3}, theme="natural")
    assert swapped.appearance[wolf_channel] == pytest.approx(0.3)


def test_replace_without_a_family_change_still_keeps_the_old_appearance():
    """The drop above is conditional on an actual species swap -- an ordinary
    ``replace`` (no ``family`` in ``changes``, or the same family named again)
    must go on carrying the recipe's own appearance forward exactly as it
    always has."""
    base = Recipe.from_dict({"family": "ogre", "appearance": {"bulk": 0.4}})
    same = base.replace(name="Renamed")
    assert same.appearance["bulk"] == pytest.approx(0.4)

    same_family = base.replace(family="ogre")
    assert same_family.appearance["bulk"] == pytest.approx(0.4)
