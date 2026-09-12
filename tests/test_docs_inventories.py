"""Regressions for the 2026-09-12 audit, findings docs-04, docs-05, docs-07.

Three root documents each carried a hand-kept inventory that had drifted from
the tree: CLAUDE.md's Offline bullet and README.md's Setup section each named
only two of the three user-initiated subprocesses that reach the network
(``fetch_worker``, ``pack_worker`` and, missed by both, ``update_worker``),
and CONTRIBUTING.md's headless-package list named four of the eight packages
``tests/_pure_packages.py`` already derives (missing ``mason``, ``sirens``,
``troupe`` and ``muse``). Each set here is derived from the tree rather than
hard-coded, the way ``tests/_pure_packages.py``'s own docstring argues for --
a hand list "fails open", so the next worker or package this repo grows would
have slid past a test that just checked for today's names.
"""

from __future__ import annotations

import ast
from pathlib import Path

from _pure_packages import pure_packages

from warlock.studio.modes import MODES

ROOT = Path(__file__).resolve().parents[1]
PIPELINES = ROOT / "src" / "warlock" / "pipelines"


def editor_packages() -> tuple[str, ...]:
    """The "headless editor packages" CLAUDE.md's Architecture bullet names.

    ``pure_packages()`` is broader than that bullet on purpose: it also finds
    ``tilegrid`` (a shared leaf ``plotter``/``packwright``/``inker`` import,
    not a workspace of its own) and ``tour`` (pure data, explicitly *not* a
    mode per its own CLAUDE.md bullet). What CLAUDE.md's bullet and
    CONTRIBUTING.md's list both mean by "headless editor package" is narrower:
    a pure package that is also one of the workspaces in
    ``studio/modes.py``'s ``MODES`` -- so that set, not the raw derivation, is
    what a doc's prose list is held to.
    """
    mode_keys = {key for key, _, _, _ in MODES}
    return tuple(name for name in pure_packages() if name in mode_keys)


def _imports_download_sibling(path: Path) -> bool:
    """True if *path* has a top-level ``from . import download``.

    ``pipelines/download.py`` is the shared "urlopen with this app's
    User-Agent" primitive -- its own docstring explains that a bare urlopen
    gets a 403 from at least one host these workers talk to. Every pipelines
    module that reaches the network imports it for exactly that reason, and
    nothing else in pipelines/ does, which makes "imports download" the one
    thing in the tree that already answers "does this subprocess reach the
    network" instead of a hand list.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):  # pragma: no cover - nothing here fails to parse
        return False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.level == 1
            and node.module is None
            and any(alias.name == "download" for alias in node.names)
        ):
            return True
    return False


def network_workers() -> tuple[str, ...]:
    """Every ``pipelines/*.py`` module that reaches the network, sorted by name.

    ``download.py`` itself is the primitive, not a worker, and is excluded.
    """
    found = [
        path.stem
        for path in sorted(PIPELINES.glob("*.py"))
        if path.name not in ("download.py", "__init__.py") and _imports_download_sibling(path)
    ]
    return tuple(found)


def test_network_workers_are_the_three_this_test_was_written_for():
    """Sanity check on the derivation itself, not on any document.

    If this starts failing, a worker was added or removed and the two tests
    below are about to correctly ask the docs to catch up -- this one just
    makes the failure legible instead of a confusing diff inside a markdown
    assertion.
    """
    assert network_workers() == ("fetch_worker", "pack_worker", "update_worker")


def test_editor_packages_are_the_eight_this_test_was_written_for():
    """Sanity check on the derivation itself, mirroring the network-worker one
    above: a ninth headless workspace enrolling itself should make this fail
    first, legibly, rather than surface as a confusing diff inside the
    CONTRIBUTING.md assertion below.
    """
    assert editor_packages() == (
        "clay",
        "inker",
        "mason",
        "muse",
        "packwright",
        "plotter",
        "sirens",
        "troupe",
    )


def test_claude_md_offline_bullet_names_every_network_worker():
    """CLAUDE.md's Offline bullet said "the single exception is fetch_worker",
    undercounting pack_worker and update_worker -- the 2026-09-12 audit's
    docs-04. SECURITY.md already named all three; CLAUDE.md's one-paragraph
    summary of the app's core guarantee did not.
    """
    text = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    missing = [w for w in network_workers() if w not in text]
    assert not missing, f"CLAUDE.md never names {missing} as network-reaching workers (docs-04)"


def test_readme_setup_section_names_every_network_worker():
    """README.md's model-download passage said "those two are the only code
    in the project that reaches the network", omitting update_worker -- the
    2026-09-12 audit's docs-05.
    """
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    missing = [w for w in network_workers() if w not in text]
    assert not missing, f"README.md never names {missing} as network-reaching workers (docs-05)"
    assert "three" in text, (
        "README.md's network-worker count should read 'three' now that "
        "update_worker is named alongside fetch_worker and pack_worker (docs-05)"
    )


def test_contributing_headless_package_list_matches_pure_packages():
    """CONTRIBUTING.md's "before you write anything" list named only
    ``inker``, ``clay``, ``plotter`` and ``packwright`` as headless, missing
    ``mason``, ``sirens``, ``troupe`` and ``muse`` -- the 2026-09-12 audit's
    docs-07. ``tests/_pure_packages.py`` already derives the real set from the
    tree; this test holds CONTRIBUTING.md's prose to that same set instead of
    to a second hand list that can disagree with it again.
    """
    text = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    missing = [name for name in editor_packages() if name not in text]
    assert not missing, f"CONTRIBUTING.md never names headless package(s) {missing} (docs-07)"
