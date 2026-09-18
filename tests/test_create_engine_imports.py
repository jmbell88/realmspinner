"""What ``modes/create/engine/`` is allowed to reach for, pinned exactly.

Create's engine is not like Mason's or Clay's: those ban ``warlock.service``
outright, and this one is *built* to import it -- ``recipe.py``,
``mesh.py`` and ``character.py`` are layer 5 by ``dev/RESTRUCTURE.md``'s own
table and exist precisely so Familiar, Review and Troupe can read "what a
recipe means" without reaching into a drawing pane. What this pin still
refuses, the same as every other headless package in this tree, is a window
(imgui/imgui_bundle/moderngl/pygame/OpenGL/glfw) and a sibling mode's own
module -- ``modes/create/ui/`` included, since an engine importing its own
UI sibling is the exact leak this split exists to close.

Written the same shape as ``tests/mason/test_mason_imports.py``: an exact
``OUTWARD_IMPORTS`` tally plus a derived sibling ban, rather than a seventh
hand list that disagrees with the other six the day a new mode arrives.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from _pure_packages import WINDOW_ROOTS, _module_roots, dotted_root, pure_packages, siblings_of

from warlock.studio.modes import create

ENGINE = Path(create.__file__).parent / "engine"
PACKAGE = "warlock.studio.modes.create.engine"

#: ``(module, imported name)`` for every module-scope import that leaves the
#: package, today. ``problems``/``problems as ...`` is ``studio/problems.py``,
#: the stdlib-only home ``Problem``/``Advisory`` moved to in this same
#: restructure step, for the identical reason ``widgets.py`` could not keep
#: them: a validation type an engine module builds may not live in the imgui
#: module. ``character`` is the sibling engine module within this same
#: package (``. import character``), not an outward edge -- recorded here
#: anyway isn't needed since :func:`_outward` only reports what leaves
#: ``PACKAGE``, and a same-package relative import never does.
OUTWARD_IMPORTS = {
    ("character.py", "warlock.characters"),
    ("character.py", "warlock.service"),
    ("character.py", "warlock.studio.problems"),
    ("mesh.py", "warlock.bench"),
    ("mesh.py", "warlock.service.validation"),
    ("mesh.py", "warlock.studio.problems"),
    ("mesh.py", "warlock.vectors"),
    ("recipe.py", "warlock.bench"),
    ("recipe.py", "warlock.generation"),
    ("recipe.py", "warlock.guidance"),
    ("recipe.py", "warlock.models"),
    ("recipe.py", "warlock.pipelines"),
    ("recipe.py", "warlock.service"),
    ("recipe.py", "warlock.service.validation"),
    ("recipe.py", "warlock.studio.problems"),
    ("recipe.py", "warlock.vectors"),
}

#: Reached only inside a function, never at module scope: ``character.py``'s
#: ``options()`` reads ``panes.stamps`` (a stdlib-only mtime cache, not a
#: pane -- see its own module docstring) and ``_fill()`` reads
#: ``state.default_form_2d`` for the same-shaped reason ``settings_2d.py``
#: always did: importing ``state`` at module scope from an engine module
#: that ``state.py`` does not itself depend on would be a needless coupling
#: for a dataclass factory read once per prompt change.
LAZY_ONLY = {"warlock.studio.panes", "warlock.studio.state"}


def _resolve(node: ast.stmt) -> set[str]:
    """Every fully-qualified module *this statement reaches into* -- not the
    names it binds. ``from ..... import models as modelslib`` and ``from
    .....service.validation import MAX_PROMPT`` both resolve to the module
    the name comes from (``warlock.models``, ``warlock.service.validation``);
    telling a submodule alias (``panes.stamps``) from a plain name
    (``MAX_PROMPT``) needs the filesystem, which :func:`_outward`'s caller
    does not have and does not need -- the module each statement reaches is
    enough to name the dependency and to drive the transitive window check.
    """
    found: set[str] = set()
    if isinstance(node, ast.Import):
        found.update(alias.name for alias in node.names)
    elif isinstance(node, ast.ImportFrom):
        if node.level == 0:
            if node.module:
                found.add(node.module)
        else:
            base = PACKAGE.rsplit(".", node.level - 1)[0]
            if node.module:
                found.add(f"{base}.{node.module}")
            else:
                found.update(f"{base}.{alias.name}" for alias in node.names)
    return found


#: Stdlib/``__future__`` names every module here carries and no pin cares
#: about -- filtered so :data:`OUTWARD_IMPORTS` reads as "what this package
#: depends on", not "every name in its import block".
_UNINTERESTING = {"__future__", "typing", "json", "pathlib", "re", "dataclasses"}


def _outward(path: Path) -> set[str]:
    """Absolute module names this file imports from outside its own package,
    at module scope. Mirrors ``tests/mason/test_mason_imports.py``'s helper.
    """
    found: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            found.update(_resolve(node))
    return {
        name
        for name in found
        if name.split(".")[0] not in _UNINTERESTING and not name.startswith(PACKAGE)
    }


def _lazy_imports(path: Path) -> set[str]:
    """Absolute module names reached from *inside* a function -- the same
    resolution as :func:`_outward`, but over every ``ast.ImportFrom``/
    ``ast.Import`` node the tree holds rather than only the top-level ones.
    """
    found: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    top_level_ids = {id(n) for n in tree.body}
    for node in ast.walk(tree):
        if id(node) in top_level_ids:
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            found.update(_resolve(node))
    return {
        name
        for name in found
        if name.split(".")[0] not in _UNINTERESTING and not name.startswith(PACKAGE)
    }


def _modules() -> list[Path]:
    return sorted(ENGINE.glob("*.py"))


def test_there_are_modules_to_check():
    """A glob that matched nothing would make every test below vacuously pass."""
    assert len(_modules()) >= 4


def test_the_outward_tally_is_exactly_this():
    """Every module-scope import that leaves the package, named -- so a new
    one is a diff, not a discovery three months later."""
    seen = {(path.name, name) for path in _modules() for name in _outward(path)}
    missing = seen - OUTWARD_IMPORTS
    stale = OUTWARD_IMPORTS - seen
    assert not missing, f"undeclared outward imports: {missing}"
    assert not stale, f"OUTWARD_IMPORTS lists an edge that no longer exists: {stale}"


def test_the_engine_never_imports_a_window_at_module_scope():
    """No imgui, no moderngl, no pygame, directly."""
    for path in _modules():
        roots = {name.split(".")[0] for name in _outward(path)}
        assert not (roots & WINDOW_ROOTS), f"{path.name} imports {roots & WINDOW_ROOTS}"


def test_the_engine_never_reaches_a_window_transitively():
    """Follows every relative import out of the package -- the way
    ``_pure_packages.pure_packages()`` proves any headless package is
    headless -- rather than trusting that today's direct imports stay pure.

    This is the test that must fail if pointed at the settings panes
    instead: see the module docstring of
    ``tests/test_create_engine_imports.py`` and this repo's restructure
    notes for how that was checked (``test_a_ui_module_does_reach_a_window``
    below runs the identical helper against
    ``modes/create/ui/panes/settings_2d.py`` and asserts it *does* find one,
    which is the proof this check is not vacuous).
    """
    for path in _modules():
        roots = _module_roots(path)
        bad = roots & WINDOW_ROOTS
        assert not bad, f"{path.name} transitively imports {bad}"


def test_a_ui_module_does_reach_a_window():
    """The negative control: run the identical transitive check against a
    pane, and it must find imgui. If it stopped finding one, the check above
    would be trivially passing for the wrong reason -- a pin that cannot fail
    is not a pin.
    """
    ui_settings_2d = ENGINE.parent / "ui" / "panes" / "settings_2d.py"
    assert ui_settings_2d.is_file()
    roots = _module_roots(ui_settings_2d)
    assert "imgui_bundle" in roots or "imgui" in roots, (
        "the settings_2d pane is expected to import imgui; if it no longer "
        "does, this negative control needs a different target"
    )


def test_the_engine_never_imports_its_own_ui_sibling():
    """The leak this whole split exists to close: an engine module reaching
    back into ``modes/create/ui/`` for the drawing half it was lifted out of.
    """
    for path in _modules():
        for name in _outward(path) | _lazy_imports(path):
            assert "modes.create.ui" not in name, f"{path.name} imports {name}"


def test_the_engine_never_imports_a_sibling_mode():
    """Nor any *other* mode's module -- ``troupe_mode``, ``clay``'s UI,
    Mason's panes. An engine may reach its own subtree and everything at L4
    and below; a sibling mode is neither.
    """
    other_mode_markers = (
        "warlock.studio.modes.troupe.mode",
        "warlock.studio.modes.review.mode",
        "warlock.studio.modes.review.ui.workspace",
        "warlock.studio.modes.clay",
        "warlock.studio.modes.mason",
        "warlock.studio.modes.mason.engine",
        "warlock.studio.inker",
        "warlock.studio.modes.plotter.engine",
        "warlock.studio.modes.packwright.engine",
        "warlock.studio.modes.sirens.engine",
        "warlock.studio.modes.sirens.mode",
        "warlock.studio.modes.muse.mode",
        "warlock.studio.modes.poser.mode",
    )
    for path in _modules():
        for name in _outward(path) | _lazy_imports(path):
            assert not name.startswith(other_mode_markers), f"{path.name} imports {name}"


def test_the_engine_is_recognised_as_headless_by_pure_packages():
    """The property the rest of the suite reasons from: ``dev/RESTRUCTURE.md``
    P5 predicted ``_pure_packages.pure_packages()`` would start finding
    ``create`` the day this package landed, and this is the test that keeps
    that true rather than assumed.
    """
    assert "create" in pure_packages()


def test_lazy_imports_are_declared():
    """Every function-scoped import out of the package is named above, so a
    new one is a decision recorded here rather than an invisible coupling."""
    seen: set[str] = set()
    for path in _modules():
        seen.update(_lazy_imports(path) - _outward(path))
    assert seen == LAZY_ONLY, seen


@pytest.mark.parametrize("other", siblings_of("create"))
def test_the_engine_never_imports_another_headless_package(other):
    """Packages that are pure for the same reason are not packages that may
    reach for each other -- Mason's own rule, applied here. Parametrized over
    what :func:`_pure_packages.pure_packages` finds in the tree rather than a
    written list, so the next mode's engine enrols itself the day it lands.
    """
    root = dotted_root(other)
    for path in _modules():
        for name in _outward(path) | _lazy_imports(path):
            assert not name.startswith(root), f"{path.name} imports {name} ({root})"
