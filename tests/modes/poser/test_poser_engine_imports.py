"""What ``modes/poser/engine/`` is allowed to reach for, pinned exactly.

Poser's engine arrived by relocation, not by a stage landing: P9 of
``dev/RESTRUCTURE.md`` folds Troupe into Poser as a stage, and this package --
the frame table (``spec``), the animation-quality scorer (``qa``) and the ULPC
reference reader (``ulpc``) -- moved here verbatim from ``modes/troupe/engine/``
as its first step. ``tests/modes/troupe/test_troupe_imports.py`` still pins the
same three modules by their old address (it is what this move made point at
the new one) and is left in place rather than deleted until the rest of Troupe
folds in too.

Written the same shape as ``tests/modes/create/test_create_engine_imports.py``
and ``tests/modes/mason/test_mason_imports.py``: an exact ``OUTWARD_IMPORTS``
tally plus a sibling ban *derived* from :mod:`_pure_packages` rather than a
seventh hand list. That derivation is why this file needed no entry added to
``_pure_packages.py`` itself: the moment ``modes/poser/engine/__init__.py``
existed on disk, :func:`_pure_packages.pure_packages` started finding
``poser`` on its own (the same walk that stopped finding ``troupe`` the moment
``modes/troupe/engine/__init__.py`` stopped existing) -- see that module's own
docstring for the six hand lists that disagreed before this shape existed, and
``test_the_engine_is_recognised_as_headless_by_pure_packages`` below for the
executable form of that claim.

Today's outward set is empty, the same claim Troupe's own pin made for the
same three files: the frame table is arithmetic over a JSON file, the QA
scorer is arithmetic over a numpy array (imported lazily, inside the
functions, never at module scope), and the ULPC reader is a crop table over a
literal layout -- none of the three needs ``warlock`` at all.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest
from _pure_packages import dotted_root, pure_packages, siblings_of

from warlock.studio.modes.poser import engine as poser_engine

ENGINE = Path(poser_engine.__file__).parent
PACKAGE = "warlock.studio.modes.poser.engine"

#: ``(module, imported name)`` for every import that leaves the package,
#: today. Empty -- see the module docstring.
OUTWARD_IMPORTS: set[tuple[str, str]] = set()

BANNED_ROOTS = {"imgui", "imgui_bundle", "moderngl", "pygame", "OpenGL", "glfw"}

#: numpy is imported inside ``qa.py``'s own functions, never at module scope --
#: a sheet is scored only when one is open and selected, and every other
#: module under ``tests/`` that imports this package pays for an eager import
#: it would never use.
LAZY_ONLY = {"numpy"}


def _outward(path: Path) -> set[str]:
    """Absolute module names this file imports from outside its own package."""
    found: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                found.add(node.module or "")
            elif node.level >= 2:
                # Level 1 is a sibling inside the package (``__init__.py``'s
                # ``from . import qa, spec, ulpc``), which never leaves it.
                base = PACKAGE.rsplit(".", node.level - 1)[0]
                if node.module:
                    found.add(f"{base}.{node.module}")
                else:
                    found.update(f"{base}.{alias.name}" for alias in node.names)
    return found


def _module_level(path: Path) -> set[str]:
    """Only the imports at the top of the file, not the ones inside functions."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            found.add((node.module or "").split(".")[0])
    return found


def _modules() -> list[Path]:
    return sorted(ENGINE.glob("*.py"))


def test_there_are_modules_to_check():
    """A glob that matched nothing would make every test below vacuously pass."""
    assert len(_modules()) >= 3


def test_the_engine_never_imports_a_window():
    for path in _modules():
        roots = {name.split(".")[0] for name in _outward(path)}
        assert not (roots & BANNED_ROOTS), f"{path.name} imports {roots & BANNED_ROOTS}"


def test_the_engine_never_imports_the_service_layer():
    for path in _modules():
        for name in _outward(path):
            assert "warlock.service" not in name, f"{path.name} imports {name}"


def test_the_engine_never_imports_the_queue_or_the_pipelines():
    for path in _modules():
        for name in _outward(path):
            assert not name.startswith("warlock.queue"), f"{path.name} imports {name}"
            assert not name.startswith("warlock.pipelines"), f"{path.name} imports {name}"
            assert not name.startswith("warlock._q"), f"{path.name} imports {name}"


def test_the_engine_never_imports_the_raster_editor():
    """Troupe *produces* a document for Inker to open; the handoff lives on the
    Inker side (``sheetin``), which is what keeps the frame table readable
    without dragging the editor in behind it -- unchanged by the move."""
    for path in _modules():
        for name in _outward(path):
            assert not name.startswith("warlock.kernels.pixel"), f"{path.name} imports {name}"


def test_the_only_outward_imports_are_the_ones_written_down():
    found = {
        (path.name, name)
        for path in _modules()
        for name in _outward(path)
        if name.split(".")[0] == "warlock"
    }
    assert found == OUTWARD_IMPORTS


def test_numpy_is_never_imported_at_module_scope():
    for path in _modules():
        assert not (_module_level(path) & LAZY_ONLY), f"{path.name} imports numpy eagerly"


def test_the_shipped_layout_table_is_part_of_the_package():
    assert (ENGINE / "data" / "layout.json").is_file()


def test_the_package_imports_with_no_optional_dependency_present():
    """The cheapest possible smoke test that every module here actually
    imports on its own, the way ``test_every_module_imports`` in the old
    location checked ``qa``/``spec``/``ulpc`` together."""
    for path in _modules():
        if path.stem == "__init__":
            continue
        importlib.import_module(f"{PACKAGE}.{path.stem}")


def test_the_engine_is_recognised_as_headless_by_pure_packages():
    """The property the rest of this suite reasons from: P9 of
    ``dev/RESTRUCTURE.md`` predicted ``_pure_packages.pure_packages()`` would
    start finding ``poser`` the day this package landed at
    ``modes/poser/engine/`` -- the same prediction P5 made for ``create`` --
    and this is the test that keeps that true rather than assumed.
    """
    assert "poser" in pure_packages()


@pytest.mark.parametrize("other", siblings_of("poser"))
def test_the_engine_never_imports_another_headless_package(other):
    """Packages that are pure for the same reason are not packages that may
    reach for each other.

    Parametrized over what :func:`_pure_packages.pure_packages` finds in the
    tree rather than a written list, so the next pure package enrols itself
    here on the day it lands -- see :mod:`_pure_packages` for the six hand
    lists that failed open and made that worth doing.
    """
    root = dotted_root(other)
    for path in _modules():
        for name in _outward(path):
            assert not (name == root or name.startswith(root + ".")), (
                f"{path.name} imports {name} ({root})"
            )


def test_the_sibling_ban_is_not_empty():
    """The parametrize above is derived, and a derivation that returned nothing
    would make it a test that runs zero cases and reports green."""
    siblings = siblings_of("poser")
    assert "mesh" in siblings
    assert "poser" not in siblings
    # The package this file itself is about used to answer to ``troupe`` --
    # the move is what makes this assertion, rather than the mirror one,
    # correct: the old name no longer resolves to anything on disk.
    assert "troupe" not in siblings
