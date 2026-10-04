"""The 2026-10-03 audit's docs-39, docs-41, docs-49 and docs-66 Lows.

Only the claims a public checkout can see live here. The ones that read
``dev/INVARIANTS.md`` or ``dev/TODO.md`` are in
``dev/tests/test_audit_2026_10_03_lows_docs3_ledger.py``.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

from realmspinner.characters.recipe import Recipe
from realmspinner.service import characters as svc_characters
from realmspinner.service.errors import Invalid

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "realmspinner"

_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4}


def test_contributing_excluded_lane_count_matches_pytest_addopts():
    """CONTRIBUTING.md said "Three lanes are excluded" over a table whose third
    row was a speed tip for running one test, not a lane. ``addopts`` excludes
    exactly the markers it names, so the sentence's count and the table's rows
    are both derived from it."""
    opts = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))["tool"]["pytest"][
        "ini_options"
    ]["addopts"]
    expr = opts[opts.index("-m") + 1]
    excluded = re.findall(r"not (\w+)", expr)

    text = (ROOT / "CONTRIBUTING.md").read_text("utf-8")
    said = re.search(r"(\w+) lanes are excluded from the default run", text)
    assert said, "CONTRIBUTING.md no longer states how many lanes the default run excludes"
    assert _WORDS[said.group(1).lower()] == len(excluded)

    section = text[said.start() : text.index("**Never edit `src/`")]
    rows = [r for r in section.splitlines() if r.startswith("| ") and not r.startswith("| Lane")]
    rows = [r for r in rows if not r.startswith("|---")]
    assert len(rows) == len(excluded), "every table row must be an excluded lane"
    for marker in excluded:
        assert f"-m {marker}" in section


def test_targets_comment_count_matches_the_muse_entries():
    """The comment above ``muse-recipe`` and ``muse-player`` said "Muse's one
    target" after the player grew its own ``help_button``."""
    source = (SRC / "kernels" / "manual" / "targets.py").read_text("utf-8")
    entries = re.findall(r'^    "(muse-[\w-]+)":', source, re.M)
    assert entries, "no muse targets found"
    said = re.search(r"# Muse's (\w+) targets?\.", source)
    assert said, "the comment above Muse's targets does not state a count"
    word = said.group(1).lower()
    assert word != "one" or len(entries) == 1
    assert word in _WORDS and _WORDS[word] == len(entries)
    assert "player" in source[said.start() : source.index('"muse-recipe"')]


def test_characters_plan_files_a_missing_clip_under_animations_and_a_bad_layout_under_layout(
    monkeypatch,
):
    """``_plan``'s docstring (and the ledger) said a ``KeyError`` falls into the
    ``field="layout"`` branch. The code files it under ``animations``, the
    control ``create_character`` names for an unknown clip, and only a
    ``ValueError`` under ``layout``."""
    import realmspinner.clips as clips
    from realmspinner.kernels import charsheet

    recipe = Recipe.from_dict({})
    library = recipe.spec.clip_library

    def missing(*_a, **_k):
        raise KeyError("idle")

    monkeypatch.setattr(clips, "expand_clips", missing)
    with pytest.raises(Invalid) as err:
        svc_characters._plan(recipe, library, recipe.logical_size)
    assert err.value.field == "animations"

    def unfit(*_a, **_k):
        raise ValueError("does not fit")

    monkeypatch.undo()
    monkeypatch.setattr(charsheet, "plan", unfit)
    with pytest.raises(Invalid) as err:
        svc_characters._plan(recipe, library, recipe.logical_size)
    assert err.value.field == "layout"

    doc = svc_characters._plan.__doc__ or ""
    assert "``KeyError``" in doc and "``field=\"animations\"``" in doc
    assert 'falls\n    into the same ``field="layout"`` branch' not in doc


def test_src_and_tests_cite_no_p65_entry_number():
    """Five src comments and three test modules cited "P65 item 2/3/4" as the
    2026-09-23 audit's open findings, while ``dev/TODO.md``'s P65 is the closed
    record of the restored ``dev/measurements``. Name the audit and the finding
    instead; an entry number a plan file reuses sends the reader somewhere else."""
    this = Path(__file__).resolve()
    hits = []
    for base in (SRC, ROOT / "tests", ROOT / "scripts"):
        for path in base.rglob("*.py"):
            if path.resolve() == this:
                continue
            for n, line in enumerate(path.read_text("utf-8", errors="ignore").splitlines(), 1):
                if re.search(r"\bP65\b", line):
                    hits.append(f"{path.relative_to(ROOT)}:{n}")
    assert hits == []
