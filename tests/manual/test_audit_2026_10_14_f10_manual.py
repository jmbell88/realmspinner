"""The 2026-10-14 audit, fixer F10: the manual loader's and targets' comments."""

from __future__ import annotations

import inspect
import re
import tomllib
from pathlib import Path

from realmspinner.kernels.manual import loader, targets

_ROOT = Path(__file__).resolve().parents[2]


def test_loader_docstring_names_only_real_force_includes() -> None:
    doc = loader.__doc__ or ""
    assert "skeleton templates" not in doc
    wheel = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    includes = wheel["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    # Every "X.md makes" sibling the docstring points at is a real force-include.
    for named in re.findall(r"bargain (\S+) makes", doc):
        assert any(key.endswith(named) for key in includes), named


def test_parts_comment_part_names_match_the_index() -> None:
    source = inspect.getsource(loader)
    # The history the audit found: chapters and gotcha ids that no longer exist,
    # and a "Part I" that the index now gives to the tutorials.
    for stale in ("Poser", "Troupe", "G60", "belongs to Part I"):
        assert stale not in source, stale
    # The part labels the comment may use are the loader's own.
    labels = [label for label, _ in loader.PARTS]
    index = (_ROOT / "docs" / "manual" / "00-index.md").read_text(encoding="utf-8")
    for label in labels:
        assert f"## {label}" in index


def test_targets_sirens_comment_does_not_appeal_to_a_missing_entry() -> None:
    assert "Troupe" not in inspect.getsource(targets)
