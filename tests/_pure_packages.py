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


def _module_roots(path: Path) -> set[str]:
    """The top-level package of every import at the top of one file.

    Module scope only, and by AST rather than by text, for one reason each.
    Module scope because a lazy import inside a function is how three of these
    packages reach Pillow and it is not what makes a package a window. AST
    because ``inker/flourish/engines.py`` *emits* a pygame source file as a
    string literal, and a grep for ``import pygame`` reads that as an import
    and drops the whole raster editor out of this set -- silently, since a
    smaller ban list is still a passing test.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):  # pragma: no cover - a file that will not parse
        return set()
    roots: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            roots.add((node.module or "").split(".")[0])
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
