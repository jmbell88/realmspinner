"""poser-11 (2026-10-03 audit): a hyphen, slash or dash glued a species to its
neighbour, so ``ogre-king`` resolved nothing while ``ogre king`` worked."""

from __future__ import annotations

import pytest

from realmspinner.characters.resolve import resolve


@pytest.mark.parametrize(
    "prompt",
    [
        "ogre-king",
        "orc-warrior",
        "wolf-like beast",
        "fire—ogre",
        "ogre/orc",
        "ogre – king",
    ],
)
def test_a_species_joined_to_another_word_by_a_hyphen_or_dash_is_still_found(prompt):
    assert resolve(prompt).family is not None


def test_a_hyphen_or_slash_splits_a_species_from_its_neighbour_the_first_one_wins():
    assert resolve("ogre/orc").family == "ogre"
    assert resolve("orc-warrior").family == "orc"


def test_side_view_with_a_hyphen_is_the_side_camera():
    assert resolve("side-view ogre").camera_preset == "side"


def test_pixel_art_with_a_hyphen_is_not_reported_unrecognised():
    result = resolve("pixel-art ogre")
    assert result.family == "ogre"
    assert result.unrecognised == ()


def test_a_hyphenated_alias_in_the_tables_is_still_matched_whole():
    # "top-down" and "2:1" are table keys; splitting them would lose the alias.
    assert resolve("top-down ogre").camera_preset == resolve("top down ogre").camera_preset
    assert resolve("2:1 dimetric ogre").camera_preset == "isometric"
