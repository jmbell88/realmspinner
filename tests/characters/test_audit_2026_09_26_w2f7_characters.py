"""Regression tests for the 2026-09-26 audit findings poser-characters-01 and
poser-characters-04, both in ``realmspinner.characters.resolve``.

Kept separate from ``tests/characters/test_resolve.py`` per the fixer brief
(``tests/<mirrored dir>/test_audit_2026_09_26_w2f7_<mirrored dir>.py``) so a
globally-unique basename never collides with another fixer's file for the
same audit pass.
"""

from __future__ import annotations

from realmspinner.characters.resolve import resolve


def test_a_curly_apostrophe_resolves_the_same_as_an_ascii_one() -> None:
    """poser-characters-01: U+2019 (curly apostrophe) was in ``_SEPARATORS``,
    so "bird's eye ogre" typed with a curly apostrophe split into the three
    tokens "bird", "s", "eye" instead of the two-token camera alias "bird's
    eye" + "ogre" the ASCII spelling produces -- and the stray "bird" token
    then matched the bird species instead of being consumed by the camera
    phrase, so a pasted curly apostrophe silently changed which species a
    prompt named.
    """
    ascii_result = resolve("bird's eye ogre")
    curly_result = resolve("bird’s eye ogre")

    assert ascii_result.family == "ogre"
    assert curly_result.family == ascii_result.family
    assert curly_result.camera_preset == ascii_result.camera_preset
    assert curly_result.unrecognised == ascii_result.unrecognised == ()

    # The other curly single quote (U+2018) is the same character used the
    # other way around by some input methods/fonts; it must not regress the
    # same way once U+2019 is fixed.
    other_curly = resolve("bird‘s eye ogre")
    assert other_curly.family == "ogre"


def test_a_creature_word_outvoted_by_a_species_is_reported_not_applied() -> None:
    """poser-characters-04: once a real species is named anywhere in the
    prompt, an unmade-creature word ("spider") is exactly as unaccounted for
    as a second species name already is -- it must not be left marked
    ``applied=True`` with nothing in ``unrecognised`` saying it did nothing.
    """
    result = resolve("giant spider knight")

    assert result.family == "knight"
    assert "spider" in result.unrecognised
    assert "giant" in result.unrecognised

    creature_spans = [s for s in result.spans if s.kind == "creature"]
    assert creature_spans, "expected at least one creature span for this prompt"
    assert all(not s.applied for s in creature_spans)


def test_a_creature_word_is_still_applied_when_no_species_is_named() -> None:
    """The outvote logic in the fix above must not fire when there genuinely
    is no species in the prompt -- the offer path still needs the creature
    span marked ``applied=True`` to point the UI at the word that produced
    the offer.
    """
    result = resolve("giant spider")

    assert result.family is None
    creature_spans = [s for s in result.spans if s.kind == "creature"]
    assert creature_spans
    assert any(s.applied for s in creature_spans)
