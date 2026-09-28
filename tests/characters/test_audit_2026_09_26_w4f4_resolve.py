"""poser-characters-06 (2026-09-26 audit): the "2:1 dimetric" camera alias
could never match.

``CAMERA_WORDS["isometric"]`` spells the ratio alias "2:1 dimetric", and
``_alias_key`` builds that table entry by splitting the alias on *whitespace*
first, so "2:1" reaches ``_normal`` as one whole token. ``_tokenise`` used to
treat ``:`` as a separator, splitting a prompt's own "2:1" into two tokens,
"2" and "1", before either ever reached ``_normal`` -- so the table's
two-word key could never match a prompt's own spelling, and "2:1 isometric
ogre" reported "2" and "1" as unrecognised instead of resolving the camera
and species it named.
"""

from __future__ import annotations

from realmspinner.characters.resolve import resolve


def test_the_two_to_one_dimetric_camera_alias_matches():
    result = resolve("2:1 dimetric ogre")
    assert result.camera_preset == "isometric"
    assert result.family == "ogre"
    assert result.unrecognised == ()


def test_two_to_one_isometric_ogre_does_not_report_the_ratio_digits_as_unrecognised():
    result = resolve("2:1 isometric ogre")
    assert result.camera_preset == "isometric"
    assert result.family == "ogre"
    assert "2" not in result.unrecognised
    assert "1" not in result.unrecognised
