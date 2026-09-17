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

**2026-09-17: a second shape, for the same reason as the first.** The
core-vs-subsystems restructure (``dev/RESTRUCTURE.md``) folds each mode's UI
into ``studio/modes/<name>/`` with the mode-private kernel, where one exists,
at ``studio/modes/<name>/engine/`` beside a UI sibling at
``studio/modes/<name>/ui/``. A derivation keyed on "a directory directly under
``studio/``" answers the headless question correctly today and stops
answering it the moment that fold lands -- not by breaking, which would at
least be loud, but by quietly returning a smaller set: ``clay`` living at
``studio/modes/clay/engine/`` instead of ``studio/clay/`` is a directory this
function would no longer iterate at all, so it drops out of
:func:`pure_packages` and, with it, out of every ``siblings_of(...)`` ban that
named it. That is the exact failure shape the rest of this file spends its
docstring arguing against -- a hand list (here, a hand-picked directory
depth) that is correct on the day it is written and silently wrong on the day
the tree changes shape under it. So :func:`pure_packages` looks in three
places rather than encoding the future layout as a rename: directly under
``studio/`` (today's shape, for every package that has not moved yet) *and*
at ``studio/modes/<name>/engine/`` for each ``<name>`` under ``studio/modes/``
(tomorrow's shape, for every one that has). Both are walked by the same
window-root check, so a mode's ``engine/`` that imports imgui is exactly as
disqualifying as a top-level package's would be.

**2026-09-17, the same day, a third place.** P3 of the restructure actually
landed before P5 did: ``studio/clay/``, ``studio/inker/`` (incl.
``flourish/``, ``walk/``), ``studio/tilegrid/``, the pure half of
``studio/viewer/`` (``math3d``/``gltf``/``glbwrite`` plus the top-level
``glbio``), ``studio/manual/{loader,parser,targets}`` and
``studio/sirens/wavout.py`` all moved out of ``studio/`` entirely, straight
into ``warlock/kernels/*`` -- not into a mode's future ``engine/``, because
they are shared domain kernels, not one mode's private engine (``tilegrid``
was already the shared-leaf case this file's own docstring names above; P3
just gave that shape a real package to live in and put four more engines
next to it). ``dev/RESTRUCTURE.md``'s own layer table calls ``warlock/kernels/``
pure by definition (``tests/test_layering.py`` is the pin that makes an
import out of it a build-time failure, not a maybe), so a directory under it
never needs the window-root walk to prove itself -- but this function runs
that walk anyway, uniformly, rather than special-casing "trust this root":
the day a kernel accidentally grows a GL import, this is the test that says
so, not a shrug that layering already covers it. Skipping the walk here would
be exactly the kind of "it can't happen" this whole file was written to stop
assuming.

The consequence for every sibling-ban pin that reads :func:`pure_packages`:
what used to be ``clay`` and ``inker`` are ``mesh`` and ``pixel`` now -- the
directory name, not a mode's name, because a kernel is named for the domain
it models rather than for the workspace that happens to be its only caller
today (``tests/mason/test_mason_imports.py`` still bans one sibling engine
from reaching Clay's primitives; it now bans ``warlock.kernels.mesh``, the
same rule wearing its new name). What remains directly under ``studio/`` is
the mode-owned set with no kernel of its own yet: ``mason``, ``muse``,
``packwright``, ``plotter``, ``sirens``, ``tour``, ``troupe``. ``familiar``
left both roots on the same day, straight to ``warlock/familiar/`` -- L3 in
the layer table, not L1 -- so it is not a headless *engine* in this
function's sense at all any more, and does not appear in
:func:`pure_packages`'s answer; its own purity is pinned directly in
``tests/familiar/test_familiar_imports.py`` instead, by AST, the way this
function proves purity for everything else.
"""

from __future__ import annotations

import ast
from pathlib import Path

import warlock.kernels
import warlock.studio

STUDIO = Path(warlock.studio.__file__).parent
KERNELS = Path(warlock.kernels.__file__).parent

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

    ``from .. import clay_mode`` inside the old ``studio/familiar/apply.py``
    (folded into ``studio/familiar_preview.py``, at ``studio`` level, since
    the 2026-09-14 T3 move) climbed one directory past ``familiar``
    (``level - 1`` parents beyond the file's own package) to ``studio``, then
    resolved ``clay_mode`` there -- still the worked example for what this
    function does with any relative import at ``level > 0``. ``from
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

    **Relative imports are resolved, not skipped.** The old ``familiar/apply.py``
    did ``from .. import clay_mode`` and the old ``familiar/scratch_ctx.py`` did
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


def _headless(package_dir: Path) -> bool:
    """Whether *package_dir* (a real package: has an ``__init__.py``) is
    headless -- recursive over its own modules, so a subpackage that imports
    a window disqualifies its parent. ``inker/flourish/`` is part of what
    "inker is headless" claims, and a check that only read ``inker/*.py``
    would let the claim be half true.
    """
    return not any(_module_roots(path) & WINDOW_ROOTS for path in package_dir.rglob("*.py"))


def pure_packages() -> tuple[str, ...]:
    """Every headless package this tree currently has, sorted.

    Looks in three places, because the restructure (``dev/RESTRUCTURE.md``) is
    mid-move: directly under ``warlock/kernels/`` -- the shared domain kernels
    P3 already moved out of ``studio/`` (``mesh``, ``pixel``, ``grid2d``,
    ``geom3d``, ``audio``, ``manual``), pure by construction
    (``tests/test_layering.py`` enforces it) but walked by the same
    window-root check as everywhere else rather than trusted on that account
    -- directly under ``studio/`` -- today's shape, for whichever mode-owned
    packages have not folded into a mode yet -- and at
    ``studio/modes/<name>/engine/`` for each ``<name>`` under
    ``studio/modes/`` -- tomorrow's shape, for whichever have. See this
    module's own docstring (the 2026-09-17 additions) for why a
    directory-depth or a directory-root assumption is exactly the kind of
    hand list the rest of this file argues against, and why the fix is
    deriving over every shape rather than picking one and editing this file
    again when the next one lands.
    """
    found = []
    for root in (KERNELS, STUDIO):
        for child in sorted(root.iterdir()):
            if not child.is_dir() or not (child / "__init__.py").exists():
                continue
            if _headless(child):
                found.append(child.name)
    modes_dir = STUDIO / "modes"
    if modes_dir.is_dir():
        for mode_dir in sorted(modes_dir.iterdir()):
            engine = mode_dir / "engine"
            if not mode_dir.is_dir() or not (engine / "__init__.py").exists():
                continue
            if _headless(engine):
                found.append(mode_dir.name)
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


def dotted_root(name: str) -> str:
    """The fully-dotted import prefix *name* (as :func:`pure_packages` names
    it) is actually reached through.

    A pin's sibling-ban test used to be able to assume every name in
    :func:`siblings_of`'s answer hung off ``warlock.studio.<name>`` -- true
    while every headless package lived directly under ``studio/``. It is
    silently false for half of them since P3 of ``dev/RESTRUCTURE.md``:
    checking a module's imports for ``"warlock.studio.mesh"`` bans nothing at
    all, because the mesh engine has always been imported as
    ``warlock.kernels.mesh`` -- Clay's own name for it never appeared in an
    import statement anywhere, it is only what :func:`pure_packages` calls the
    directory. This looks the prefix up against the tree itself, the same way
    :func:`pure_packages` found *name* there in the first place, rather than
    asking every pin to keep a second list of "which of these are kernels
    now" that can (and, before this function existed, silently did) go stale
    the moment a sibling-ban parametrize kept naming a package whose real
    import path had moved out from under it.
    """
    if (KERNELS / name / "__init__.py").exists():
        return f"warlock.kernels.{name}"
    if (STUDIO / "modes" / name / "engine" / "__init__.py").exists():
        return f"warlock.studio.modes.{name}.engine"
    return f"warlock.studio.{name}"
