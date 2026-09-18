"""``scripts.py``'s module docstring against ``TOURS`` itself.

The docstring is prose that counts and explains the tours by hand; ``TOURS``
is the data. Nothing else compared them, so a fifth tour landed
(``muse-basics``, in the same file, on the same day this test was written)
with no docstring paragraph and a stale "Four of them" left standing.
"""

from __future__ import annotations

import ast
from pathlib import Path

from warlock.studio.tour import TOURS, scripts

SCRIPTS_PATH = Path(scripts.__file__)


def test_scripts_docstring_count_matches_TOURS():
    """The 2026-09-07 audit, finding tour-03: ``scripts.py``'s docstring said
    "Four of them" and explained four; ``TOURS`` holds five -- ``muse-basics``
    was unexplained, and it is the one tour that needs weights on disk.
    """
    doc = ast.get_docstring(ast.parse(SCRIPTS_PATH.read_text(encoding="utf-8"))) or ""
    assert "Four of them" not in doc, (
        f"scripts.py's docstring still says 'Four of them', but TOURS holds "
        f"{len(TOURS)}"
    )
    number_words = {4: "Four", 5: "Five", 6: "Six"}
    word = number_words.get(len(TOURS))
    assert word is not None, f"add a number word for {len(TOURS)} tours"
    assert f"{word} of them" in doc, (
        f"scripts.py's docstring does not say '{word} of them' even though "
        f"TOURS holds {len(TOURS)}"
    )


def test_scripts_docstring_explains_every_tour_by_key():
    """Every tour in ``TOURS`` gets a paragraph in the module docstring naming
    it by key (double-backtick, as every other tour already is) -- the shape
    ``muse-basics`` was missing from entirely before this fix.
    """
    doc = ast.get_docstring(ast.parse(SCRIPTS_PATH.read_text(encoding="utf-8"))) or ""
    missing = [tour.key for tour in TOURS if f"``{tour.key}``" not in doc]
    assert not missing, (
        f"scripts.py's docstring does not mention the tour(s) {missing} by key"
    )
