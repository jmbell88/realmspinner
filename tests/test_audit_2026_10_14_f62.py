"""Regression tests for the 2026-10-14 audit, fixer F62: three test gates that
could not catch what they claimed to (docs-C2-16, shell-chrome-02, docs-C1-16).
"""

from __future__ import annotations

import re

import _pure_packages as pp
import test_accessibility as acc
import test_job_durability as durability

_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
}  # fmt: skip


def test_pure_packages_docstring_names_no_deleted_package():
    """The 2026-10-14 audit, finding docs-C2-16. The docstring's list of the
    packages that have since converted to ``siblings_of`` named Poser's
    ``engine/``, deleted with Poser by the character-sheet removal. Every name
    in that sentence must still be a directory under ``studio/modes/``."""
    doc = pp.__doc__ or ""
    match = re.search(r"\(([^()]*?)\s+have since converted", doc, re.DOTALL)
    assert match, "the 'have since converted' sentence moved: update this test with it"
    names = re.findall(r"[A-Z][a-z]+", match.group(1))
    assert names, match.group(1)
    modes_dir = pp.STUDIO / "modes"
    gone = [n for n in names if not (modes_dir / n.lower()).is_dir()]
    assert not gone, f"the docstring names packages that no longer exist: {gone}"


def test_the_binder_scan_finds_the_modes_that_bind_arrows_and_is_never_empty(tmp_path):
    """The 2026-10-14 audit, finding shell-chrome-02. The scan looked for
    ``studio/<mode>_mode.py`` -- a path no mode has -- so it found nothing and
    the guard passed whatever a mode bound. It now reads ``studio/modes/<mode>/``,
    in both key spellings, and refuses an empty answer."""
    real = acc._binding_modes(acc.SRC / "studio" / "modes")
    assert real, "scan found no mode at all"
    # K_* spelling in a file that is not mode.py (Inker, Sirens), and the
    # ``pygame.key.name`` spelling (Clay), both live in the tree today.
    assert {"inker", "sirens", "clay"} <= real, real

    # planted fixture: a new mode binding Space by name in a sub-package, a mode
    # binding by constant, a mode with neither, and the agent surface's
    # "space" argument, which is not a key.
    root = tmp_path / "modes"
    (root / "binds_name" / "ui").mkdir(parents=True)
    (root / "binds_name" / "ui" / "keys.py").write_text('if name == "space":\n    pass\n')
    (root / "binds_const").mkdir()
    (root / "binds_const" / "mode.py").write_text("if key == pygame.K_UP:\n    pass\n")
    (root / "quiet").mkdir()
    (root / "quiet" / "mode.py").write_text("x = 1\n")
    (root / "agent_only" / "agent").mkdir(parents=True)
    (root / "agent_only" / "agent" / "validate.py").write_text('if name == "space":\n    pass\n')
    assert acc._binding_modes(root) == {"binds_name", "binds_const"}


def test_the_publishers_docstring_count_matches_the_table():
    """The 2026-10-14 audit, finding docs-C1-16. The test's docstring said
    "(twelve, ..." beside a ``PUBLISHERS`` table of nine rows."""
    doc = durability.test_every_served_publish_commits_the_cancel_token.__doc__ or ""
    match = re.search(r"as\s+of\s+the\s+2026-10-03\s+audit\s+\((\w+)", doc)
    assert match, "the docstring no longer states a row count: update this test with it"
    stated = _WORDS.get(match.group(1).lower())
    assert stated == len(durability.PUBLISHERS), (
        f"docstring says {match.group(1)!r}, the table has {len(durability.PUBLISHERS)} rows"
    )
