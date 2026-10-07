"""The manual's two shape lists, held against ``primitives.CLAY_GENERATORS``.

Clay's add panel is generated from the registry (``CLAY_GENERATORS``, the subset
of ``GENERATORS`` Clay offers -- Mason's saved scenes keep the whole registry), so
a sixteenth primitive is a function and one registry line -- and it appears in the app
the same day. The manual is the one place that does *not* follow along: two chapters write the
shapes out in prose, and until this test nothing held either list against the
registry. The failure mode is silent by construction. A shape missing from the
prose still has a button, still has properties and still exports; it is simply
undocumented, which is invisible to every other test in ``tests/manual/`` --
they check that chapters parse, that links resolve and that anchors exist, and
a complete sentence naming eight of twelve shapes passes all of them.

Named rather than merely counted: a chapter saying "twelve" while listing nine
would pass a count, and a count is exactly the drift a prose list invites.

The match is on a word boundary rather than a substring, so "architecture" does
not stand in for the arch and "column" has to be the shape rather than a column
of the window.
"""

from __future__ import annotations

import re

import pytest

from realmspinner.kernels.manual import loader
from realmspinner.kernels.mesh import primitives

# The two chapters that write the shapes out, and the heading of the section in
# each that does the writing. The search is scoped to that section rather than
# run over the whole chapter, because half these names are ordinary English in
# a manual about a window: "the middle column", "a grid of tiles", "the box
# asks first". Chapter-wide, ``column`` was already satisfied before the
# shape existed, which is a gate that cannot fail.
SECTIONS = {
    "07-modelling": "## Primitives",
    "30-clay": "## Adding a primitive",
}
CHAPTERS = tuple(SECTIONS)


def _shape_list(key: str) -> str:
    """The chapter's primitive section: its heading down to the next one."""
    text = loader.load(key)
    start = text.index(SECTIONS[key])
    end = text.find("\n## ", start + 1)
    return text[start : end if end != -1 else len(text)].lower()


def _display(name: str) -> str:
    """The registry key as a reader meets it -- ``uv_sphere`` is "UV sphere"."""
    return name.replace("_", " ")


@pytest.mark.parametrize("key", CHAPTERS)
@pytest.mark.parametrize("name", sorted(primitives.CLAY_GENERATOR_NAMES))
def test_every_clay_generator_is_named_in_the_manual(name: str, key: str) -> None:
    text = _shape_list(key)
    pattern = r"\b" + re.escape(_display(name)) + r"\b"
    assert re.search(pattern, text), (
        f"docs/manual/{key}.md's {SECTIONS[key]!r} section does not name "
        f"the {name!r} primitive; "
        "the add panel is generated from the registry, the prose is not"
    )


_WORDS = {15: "fifteen"}


@pytest.mark.parametrize("key", CHAPTERS)
def test_each_chapter_states_the_number_of_shapes_clay_really_offers(key: str) -> None:
    """A list that names every shape under a count that is wrong is still wrong:
    chapter 7 said "Twenty-one" for a palette that had stopped being that long."""
    count = len(primitives.CLAY_GENERATOR_NAMES)
    assert count in _WORDS, f"teach this test the word for {count}"
    assert _WORDS[count] in _shape_list(key), (
        f"docs/manual/{key}.md's {SECTIONS[key]!r} section does not say Clay has "
        f"{_WORDS[count]} shapes"
    )
