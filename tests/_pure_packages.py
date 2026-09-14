"""Which packages under ``studio/`` are headless engines, worked out rather than listed.

Every import pin in this tree asks the same question twice. Once exactly --
"here is the complete set of names this package reaches for" -- and once
legibly, as "and in particular it does not reach for a sibling engine". The
exact form is the real gate; the sibling form is the one a reader believes.

**The sibling form was six hand lists and no two agreed.** Measured on
2026-09-11 while checking the Mason plan's claim that a new pure package
"joins the lists": ``tests/clay/test_clay_imports.py`` named ``inker``,
``plotter`` and ``packwright`` but not ``sirens``, ``muse`` or ``troupe``;
``tests/sirens`` named five; ``tests/muse`` named six; and ``inker``,
``plotter``, ``packwright`` and ``troupe`` had no sibling check at all. Each
of those lists fails **open** -- a package missing from one is a package that
one pin will never catch an import of -- which is the ``PUBLISHERS`` shape
this repo already writes down as a bad gate. Adding a seventh hand list for
Mason would have produced a seventh disagreement and left the six holes where
they were.

So the set is derived from what is actually in the tree: a directory under
``src/warlock/studio/`` is a headless package when it is a package and no
module in it imports a window at module scope. The property that buys is the
one a hand list cannot have -- **the package added after Mason enrols itself**,
in every pin that asks this helper, on the day it lands.

The derivation is deliberately *not* "everything under ``studio/``": the
viewport, the panes and the manual renderer all import imgui or moderngl, and
what makes them uninteresting here is exactly what the banned-roots test
already says about them. It is also deliberately not filtered down to "editor
engines" -- ``tilegrid`` is a shared leaf that ``plotter``, ``packwright`` and
``inker`` reach for on purpose, and it lands in this set. That is not a
mistake: a pin that bans a sibling it legitimately imports writes the import
down in its own ``OUTWARD_IMPORTS`` and passes it to :func:`siblings_of` as an
exception, which is a decision recorded in one place rather than an omission
from a list nobody reads.

**Not yet adopted by the other six pins, and that is a live loose end.**
``packwright`` imports ``plotter`` for real, so retrofitting this helper there
is a decision about that edge rather than a mechanical substitution, and it
belongs to whoever takes that decision. What is here is the derivation the next
pin can adopt without inventing it again.
"""

from __future__ import annotations

import ast
from pathlib import Path

import warlock.studio

STUDIO = Path(warlock.studio.__file__).parent

#: A module importing one of these at module scope is not headless. The same
#: set every pin's own banned-roots test uses, stated once here because this
#: file is what decides which packages those tests are even about.
WINDOW_ROOTS = frozenset({"imgui", "imgui_bundle", "moderngl", "pygame", "OpenGL", "glfw"})


#: Resolved file -> its own transitive root set, so a package whose modules
#: fan out through several relative imports (``agent_clay`` reaches ``clay_mode``,
#: ``clay_view``, ``panes.clay_tools`` and the whole ``clay``/``viewer`` trees)
#: is not re-parsed and re-walked once per importer. Keyed on resolved path
#: rather than cleared between calls: the tree does not change mid-process,
#: and a stale entry is never worse than the recomputation it replaces.
_ROOT_CACHE: dict[Path, set[str]] = {}


def _relative_targets(path: Path, node: ast.ImportFrom) -> list[Path]:
    """Every filesystem path one relative ``ImportFrom`` node may name.

    ``from .. import clay_mode`` inside ``studio/familiar/apply.py`` climbs
    one directory past ``familiar`` (``level - 1`` parents beyond the file's
    own package) to ``studio``, then resolves ``clay_mode`` there. ``from
    .panes import clay_tools`` additionally has to try the *module* itself
    (``studio/panes``, in case ``clay_tools`` is merely an attribute imported
    off its ``__init__.py``) alongside the submodule guess (``studio/panes/
    clay_tools.py``) -- this repo's real imports are always the latter, but
    guessing wrong costs nothing (:func:`_module_file` just returns ``None``
    for a path that is not a module), while guessing only the former would
    silently stop resolving a whole class of relative import.
    """
    base = path.parent
    for _ in range(node.level - 1):
        base = base.parent
    if node.module:
        base = base.joinpath(*node.module.split("."))
    candidates = [base] if node.module else []
    candidates.extend(base / alias.name for alias in node.names)
    return candidates


def _module_file(base: Path) -> Path | None:
    """The source file ``base`` names, as a plain module or a package."""
    as_module = base.with_suffix(".py")
    if as_module.is_file():
        return as_module
    as_package = base / "__init__.py"
    if as_package.is_file():
        return as_package
    return None


def _module_roots(path: Path, _stack: frozenset[Path] = frozenset()) -> set[str]:
    """The outward root of every import this file reaches, directly or
    through a chain of relative imports.

    Module scope only, and by AST rather than by text, for one reason each.
    Module scope because a lazy import inside a function is how three of these
    packages reach Pillow and it is not what makes a package a window. AST
    because ``inker/flourish/engines.py`` *emits* a pygame source file as a
    string literal, and a grep for ``import pygame`` reads that as an import
    and drops the whole raster editor out of this set -- silently, since a
    smaller ban list is still a passing test.

    **Relative imports are resolved, not skipped.** ``familiar/apply.py``
    does ``from .. import clay_mode`` and ``familiar/scratch_ctx.py`` does
    ``from .. import agent_clay`` -- both ``node.level > 0``, and an earlier
    version of this helper only ever looked at ``node.level == 0`` (an
    absolute import), so neither line contributed anything to this file's
    roots at all. That let ``familiar`` come back "pure" from
    :func:`pure_packages` even though ``agent_clay`` imports ``clay_view``
    (``moderngl``) and ``panes.clay_tools`` (``imgui_bundle``) at its own
    module scope -- a real window, two relative hops away. A relative import
    is resolved to the file it names (:func:`_relative_targets` plus
    :func:`_module_file`) and that file's own roots are folded in
    recursively, so the window import three sibling modules away is exactly
    as visible here as one written directly in this file. ``_stack`` is only
    a cycle guard against two modules that import each other -- it is never
    populated by a caller outside this function.
    """
    if path in _ROOT_CACHE:
        return _ROOT_CACHE[path]
    if path in _stack:  # a circular relative-import chain -- do not recurse forever
        return set()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):  # pragma: no cover - a file that will not parse
        return set()
    roots: set[str] = set()
    stack = _stack | {path}
    for node in tree.body:
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                roots.add((node.module or "").split(".")[0])
                continue
            for candidate in _relative_targets(path, node):
                resolved = _module_file(candidate)
                if resolved is not None:
                    roots.update(_module_roots(resolved, stack))
    _ROOT_CACHE[path] = roots
    return roots


def pure_packages() -> tuple[str, ...]:
    """Every headless package directly under ``studio/``, sorted.

    Recursive over each package's own modules, so a subpackage that imports a
    window disqualifies its parent -- ``inker/flourish/`` is part of what
    "inker is headless" claims, and a check that only read ``inker/*.py`` would
    let the claim be half true.
    """
    found = []
    for child in sorted(STUDIO.iterdir()):
        if not child.is_dir() or not (child / "__init__.py").exists():
            continue
        if any(_module_roots(path) & WINDOW_ROOTS for path in child.rglob("*.py")):
            continue
        found.append(child.name)
    return tuple(found)


def siblings_of(
    package: str, *, allowed: frozenset[str] | set[str] = frozenset()
) -> tuple[str, ...]:
    """The headless packages *package* may not import, sorted.

    ``allowed`` is how a package that genuinely depends on a sibling records
    it: the name comes out of the ban and is written down in that pin's
    ``OUTWARD_IMPORTS`` instead, where the exact-set test holds it.
    """
    return tuple(name for name in pure_packages() if name != package and name not in allowed)
