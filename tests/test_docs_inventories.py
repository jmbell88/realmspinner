"""Regressions for the 2026-09-12 audit, findings docs-04, docs-05, docs-07,
and for the fifth copy found the same evening.

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

The audit fixed the two documents it knew about. Reviewing CLAUDE.md later the
same day turned up a fifth carrying the same undercount -- CONTRIBUTING.md's
"surprises people" list -- which nothing here covered, because the *list of
documents* was itself hand-kept. Both lists are derived now.

docs-04's own document, CLAUDE.md's Offline bullet, is checked in
``dev/tests/test_docs_inventories.py`` since CLAUDE.md moved out of the
public checkout on 2026-09-16.
"""

from __future__ import annotations

import ast
from pathlib import Path

from _pure_packages import KERNELS, STUDIO, dotted_root, pure_packages

from warlock.studio.modes import MODES

ROOT = Path(__file__).resolve().parents[1]
PIPELINES = ROOT / "src" / "warlock" / "pipelines"


#: 2026-09-17, P3 of ``dev/RESTRUCTURE.md``: Clay's and Inker's own engines
#: moved out of ``studio/clay/`` and ``studio/inker/`` into
#: ``warlock/kernels/mesh/`` and ``warlock/kernels/pixel/`` -- named for the
#: domain they model, the way a shared kernel is, rather than for the one mode
#: that happens to be their only caller today. ``pure_packages()`` correctly
#: reports ``mesh`` and ``pixel`` now, not ``clay``/``inker``, which is right
#: for every sibling-ban pin that reads it (a pin bans *packages*, and the
#: package is really named ``mesh``). CLAUDE.md's and CONTRIBUTING.md's prose
#: is not about packages, though -- it is about *modes*, told to a contributor
#: by the name they already know the workspace by, and nobody browsing
#: CONTRIBUTING.md's list is looking for "mesh". This is the one, narrow
#: bridge between the two vocabularies, recorded here rather than left for
#: :func:`editor_packages` to get quietly wrong by matching on identity: every
#: other engine still directly under ``studio/`` (``mason``, ``muse``,
#: ``packwright``, ``plotter``, ``sirens``, ``troupe``) is still named for its
#: mode, so it needs no entry here at all -- adding one "just in case" would be
#: exactly the unread hand list this file's sibling derivation exists to
#: avoid.
_ENGINE_TO_MODE = {
    "mesh": "clay",
    "pixel": "inker",
}


def _package_dir(name: str) -> Path:
    """The directory :func:`pure_packages` found *name* in, by the same
    three searches that function makes.
    """
    if (KERNELS / name / "__init__.py").exists():
        return KERNELS / name
    engine = STUDIO / "modes" / name / "engine"
    if (engine / "__init__.py").exists():
        return engine
    return STUDIO / name


def _module_scope_targets(path: Path, package_dotted: str) -> set[str]:
    """Every absolute module a file's *module-scope* imports reach, resolving
    a relative ``from`` against ``package_dotted`` the way Python itself
    resolves it: ``level=1`` is the file's own containing package."""
    found: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module:
                    found.add(node.module)
            else:
                base = package_dotted.rsplit(".", node.level - 1)[0]
                if node.module:
                    found.add(f"{base}.{node.module}")
                else:
                    found.update(f"{base}.{alias.name}" for alias in node.names)
    return found


def _imports_service(name: str) -> bool:
    """Whether any module of pure package *name* imports ``warlock.service``
    at module scope.

    2026-09-18 restructure, P5: ``pure_packages()`` started finding ``create``
    the day its engine (``modes/create/engine/``) landed, because that
    function's whole test is "no window at module scope" -- and Create's
    engine is deliberately layer 5, built to import ``warlock.service``
    (``recipe.py``/``mesh.py``/``character.py`` all do). CLAUDE.md's and
    CONTRIBUTING.md's "headless editor package" claim is narrower than
    "no window": both documents say, in as many words, a package that
    imports "no imgui, moderngl, pygame **or service**". A package that
    fails that third clause is not one of them, and this derives the
    exclusion from the tree instead of hand-naming ``create`` -- the same
    argument ``tests/_pure_packages.py``'s own docstring makes against every
    hand list in this area.
    """
    package_dotted = dotted_root(name)
    for path in _package_dir(name).rglob("*.py"):
        targets = _module_scope_targets(path, package_dotted)
        if any(t == "warlock.service" or t.startswith("warlock.service.") for t in targets):
            return True
    return False


def editor_packages() -> tuple[str, ...]:
    """The "headless editor packages" CLAUDE.md's Architecture bullet names.

    ``pure_packages()`` is broader than that bullet on purpose: it also finds
    ``tilegrid``'s successor ``grid2d`` and the rest of the shared kernels
    (``geom3d``, ``audio``, ``manual`` -- none of them a workspace of its own),
    ``tour`` (pure data, explicitly *not* a mode per its own CLAUDE.md
    bullet), and, since P5, ``create`` (headless by the "no window" test, but
    not by the "no service either" one -- see :func:`_imports_service`). What
    CLAUDE.md's bullet and CONTRIBUTING.md's list both mean by "headless
    editor package" is narrower: a pure package that imports no service door
    either, and that is also one of the workspaces in
    ``studio/modes.py``'s ``MODES``, once :data:`_ENGINE_TO_MODE` translates
    Clay's and Inker's engines back to the mode name a contributor actually
    reads -- so that set, not the raw derivation, is what a doc's prose list
    is held to.
    """
    mode_keys = {key for key, _, _, _ in MODES}
    candidates = (name for name in pure_packages() if not _imports_service(name))
    named = (_ENGINE_TO_MODE.get(name, name) for name in candidates)
    return tuple(sorted(name for name in named if name in mode_keys))


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

    Still eight after 2026-09-17's P3 move, and deliberately the same eight
    names -- ``_ENGINE_TO_MODE`` exists so that Clay's and Inker's engines
    changing address (and name) inside ``warlock/kernels/`` does not also
    change what a contributor reads in a doc. If this count ever does move,
    say so in this test's name and docstring rather than just editing the
    tuple below -- that was the instruction this test itself was written to
    satisfy the last time the set changed.
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


def test_contributing_md_offline_bullet_names_every_network_worker():
    """CONTRIBUTING.md was the *fifth* copy of this fact, and the one nobody
    counted.

    ``dev/INVARIANTS.md`` records the offline exceptions as being stated in
    four documents -- itself, ``SECURITY.md``, ``README.md`` and ``CLAUDE.md``
    -- and docs-04/docs-05 fixed the two of those four that had gone stale.
    CONTRIBUTING.md holds the same fact in its "surprises people" list and was
    not in anybody's inventory, so it still said "Nothing downloads at runtime
    except the user-initiated fetch worker" on the evening of the day the other
    two were corrected. Found 2026-09-12 while reviewing CLAUDE.md.

    That is the fails-open shape twice over: a hand list of *documents* around
    a hand list of *workers*. This test closes the outer one, so the derived
    set now holds every document that makes the claim.
    """
    text = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    missing = [w for w in network_workers() if w not in text]
    assert not missing, (
        f"CONTRIBUTING.md never names {missing} as network-reaching workers -- "
        "the same undercount docs-04 and docs-05 fixed in CLAUDE.md and README"
    )
