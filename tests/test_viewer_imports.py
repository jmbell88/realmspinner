"""What ``studio/viewer/`` is allowed to reach for, pinned across the package.

The ``tests/modes/inker/test_sheetout.py`` pin, applied to the one package that did
not already have it. ``viewer/__init__.py``'s own docstring claims "Nothing
here imports imgui or pygame" for the whole package -- the split
``viewer_embed`` exists to draw a panel around -- but the 2026-09-07 audit,
finding create-10, found that claim enforced on only 2 of the package's 17
modules: ``test_poser_imports.py`` pins ``pose.py`` and ``bonelines.py`` by
name because those two are also *Poser's* pure half, and nothing else in the
tree ever globbed the rest of the directory the way Clay, Inker, Plotter and
Packwright's own package pins do. No live violation today -- every module here
already keeps the claim -- so this is the missing tripwire, not a fix to any
one module.

**Fourteen, not seventeen, since 2026-09-17.** P3 of ``dev/RESTRUCTURE.md``
moved this package's three genuinely pure modules -- ``math3d``, ``gltf`` and
``glbwrite`` -- out to ``realmspinner/kernels/geom3d/``, where they are pinned by
their own layering rather than by this package's "no window toolkit" claim
(they never needed the ModernGL half of it, only the pygame/imgui half, which
``kernels/geom3d`` gets for free by living outside ``studio/`` at all). What
is left here genuinely does build GL objects, the way the module docstring
already says half of it does -- so the count shrank, the claim did not.

Unlike those siblings, ``moderngl`` is not banned: this package is the
ModernGL viewport, and half its modules (``glctx``, ``render``, ``scene``,
``grid``, ``bonelines``) build GL objects directly. The claim under test is
narrower and is exactly what the docstring states -- no window toolkit, no
immediate-mode GUI.
"""

from __future__ import annotations

import ast
from pathlib import Path

from realmspinner.studio import viewer

ENGINE = Path(viewer.__file__).parent

#: Window toolkits and immediate-mode GUI libraries. Not ``moderngl`` -- see
#: the module docstring above.
BANNED_ROOTS = {"imgui", "imgui_bundle", "pygame", "OpenGL", "glfw"}


def _outward(path: Path) -> set[str]:
    """Absolute module names this file imports, at any import depth."""
    found: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            found.add(node.module or "")
    return found


def _modules() -> list[Path]:
    """Every real module in the package -- ``__init__.py`` excluded, since it
    is the docstring making the claim rather than a module the claim is about
    (the finding's own count, "2 of 17 modules", is of these)."""
    return sorted(p for p in ENGINE.glob("*.py") if p.name != "__init__.py")


def test_there_are_fourteen_modules_to_check():
    """A glob that quietly started matching fewer files would measure less
    than it claims to; a glob that started matching more (a stray script
    dropped in the package) is worth noticing too.

    Fourteen since 2026-09-17, not the seventeen this was first written for
    -- P3 of ``dev/RESTRUCTURE.md`` moved ``math3d``, ``gltf`` and
    ``glbwrite`` to ``realmspinner/kernels/geom3d/``. See the module docstring.
    """
    assert len(_modules()) == 14


def test_none_of_them_imports_a_window_or_an_immediate_mode_gui():
    """The claim ``viewer/__init__.py`` makes for the whole package, checked
    against every module in it rather than the two ``test_poser_imports.py``
    happens to also need for Poser's own reasons."""
    for path in _modules():
        roots = {name.split(".")[0] for name in _outward(path)}
        assert not (roots & BANNED_ROOTS), f"{path.name} imports {roots & BANNED_ROOTS}"


def test_none_of_them_imports_the_service_layer():
    """The renderer answers questions about rays and bones; it does not know
    about jobs, sqlite or VRAM admission."""
    for path in _modules():
        for name in _outward(path):
            assert not name.startswith("realmspinner.service"), f"{path.name} imports {name}"


def test_none_of_them_imports_the_queue_or_the_pipelines():
    for path in _modules():
        for name in _outward(path):
            assert not name.startswith("realmspinner.queue"), f"{path.name} imports {name}"
            # ``realmspinner._q_*`` too: the queue's worker halves are the same
            # dependency wearing a different name, and importing one of those
            # would drag torch behind a headless test as surely as importing
            # ``queue`` itself.
            assert not name.startswith("realmspinner._q"), f"{path.name} imports {name}"


def test_moderngl_is_allowed_everywhere_here_unlike_the_headless_packages():
    """The one deliberate departure from the sibling pins' ``BANNED_ROOTS``,
    stated as a passing assertion rather than left as an absence: this
    package is the GPU half, and at least one module genuinely needs it."""
    roots = {r for path in _modules() for r in {n.split(".")[0] for n in _outward(path)}}
    assert "moderngl" in roots
