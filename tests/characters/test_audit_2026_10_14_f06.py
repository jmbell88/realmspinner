"""The 2026-10-14 audit's characters-package findings: poser-characters-08 (prose
that names removed things) and poser-characters-03 (an alias that claims a look).

Both are the package saying something it no longer does. A docstring that cites
a file nobody can open sends a reader chasing it; an alias that resolves "earth
elemental" to the fire look makes the form look right while the Look combo says
otherwise.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from realmspinner.characters import CharacterError, Recipe, family, resolve

PACKAGE = Path(family.__file__).parent
REPO = PACKAGE.parents[2]

#: Strings that name a pipeline, mode or file that is gone. Not "Troupe" alone:
#: several comments say "the retired Troupe door" on purpose, to name what a
#: restated constant used to be pinned against. What may not stay is prose that
#: describes the *present* in the vocabulary of the removed.
_REMOVED = (
    "studio.modes.poser",
    "tests/modes/poser",
    "26-poser",
    "34-troupe",
    "a Troupe job",
    "a character sheet needs",
    "Troupe reads",
    "every sprite of every sheet",
    "one character sheet",
    "a camera and a frame table",
)


def _sources() -> list[Path]:
    return sorted(p for p in PACKAGE.rglob("*.py"))


def test_characters_package_text_names_no_removed_mode_or_test_file():
    """The 2026-10-14 audit, finding poser-characters-08."""
    found = [
        f"{path.relative_to(PACKAGE)}: {needle!r}"
        for path in _sources()
        for needle in _REMOVED
        if needle in path.read_text(encoding="utf-8")
    ]
    assert not found, found


def test_a_file_the_package_cites_exists():
    """A cited ``tests/...py`` or ``docs/manual/...md`` that is not on disk is a
    reader sent after a file that left (``26-poser.md`` became
    ``26-inker-animation.md``). ``dev/`` is never checked: it is not on a public
    checkout."""
    missing: list[str] = []
    for path in _sources():
        text = path.read_text(encoding="utf-8")
        for cited in set(re.findall(r"(?:tests/[\w/]+\.py|docs/manual/[\w-]+\.md)", text)):
            if not (REPO / cited).is_file():
                missing.append(f"{path.relative_to(PACKAGE)}: {cited}")
    assert not missing, missing


@pytest.mark.parametrize("rel", ["__init__.py", "family.py", "resolve.py", "recipe.py"])
def test_a_module_docstring_does_not_describe_the_retired_sheet(rel):
    """The package's front doors are what a reader opens first, and they described
    "a Troupe job", "a character sheet" and "a sprite sheet" long after the
    pipeline ended at a rigged mesh."""
    doc = ast.get_docstring(ast.parse((PACKAGE / rel).read_text(encoding="utf-8"))) or ""
    for word in ("Troupe", "character sheet", "sprite sheet"):
        assert word not in doc, f"{rel} docstring still says {word!r}"


def test_a_refusal_names_no_pipeline_that_is_gone():
    """The two user-visible strings in ``_check_animations``: a Recipe's refusal
    reaches a control or an agent's reply, and "a character sheet needs ..." /
    "Troupe reads ..." named a pipeline the user cannot reach."""
    with pytest.raises(CharacterError) as empty:
        Recipe.from_dict({"animations": {}})
    assert "sheet" not in str(empty.value)
    assert empty.value.field == "animations"

    with pytest.raises(CharacterError) as tagged:
        Recipe.from_dict({"animations": {"walk_left": 4}})
    assert "Troupe" not in str(tagged.value)
    assert tagged.value.field == "animations"
    assert "walk_left" in str(tagged.value)


# --- poser-characters-03 -------------------------------------------------------


def test_an_alias_that_names_a_look_either_selects_it_or_is_reported_as_not_interpreted():
    """"earth elemental" resolved to the elemental species with ``theme=None``, and
    an omitted theme defaults to the species' first look -- the lit *fire* one --
    so a user who typed "earth" got a glowing orange accent and nothing in "Not
    interpreted" (the 2026-10-14 audit, finding poser-characters-03)."""
    res = resolve.resolve("an earth elemental")
    assert res.family == "elemental"
    assert res.theme == "stone" or any("earth" in word.lower() for word in res.unrecognised), (
        res.theme,
        res.unrecognised,
    )


def test_a_species_alias_never_swallows_a_look_word_the_species_paints():
    """The general form: if an alias contains a word that means one of the looks
    its own species paints, the family span eats the word, ``theme`` stays None and
    the species' first look wins whatever the user said."""
    offenders: list[str] = []
    for key, fam in family.families().items():
        painted = {t.key for t in fam.themes}
        for alias in fam.aliases:
            tokens = alias.split()
            if len(tokens) < 2:
                continue
            for theme_key, words in resolve.THEME_WORDS.items():
                if theme_key in painted and any(t in words for t in tokens):
                    offenders.append(f"{key}: {alias!r} (look {theme_key!r})")
    assert not offenders, offenders


def test_a_look_word_beside_a_species_selects_that_look():
    for prompt, look in (("a fire elemental", "fire"), ("an earth elemental", "stone")):
        res = resolve.resolve(prompt)
        assert (res.family, res.theme) == ("elemental", look), prompt


# --- poser-characters-04 (the vocabulary half) ---------------------------------


def test_every_sheet_word_is_a_noise_word():
    """``SHEET_WORDS`` is the part of the noise vocabulary that asks for the other
    asset type (a sheet, a direction count); a word there that is not in
    ``NOISE_WORDS`` would be reported as unrecognised instead."""
    assert resolve.SHEET_WORDS
    noise = {resolve._alias_key(w) for w in resolve.NOISE_WORDS}
    for word in resolve.SHEET_WORDS:
        assert resolve._alias_key(word) in noise, word
