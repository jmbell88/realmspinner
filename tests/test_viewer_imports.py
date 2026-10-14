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


#: Edges the 2026-10-11 audit's relative-import resolution (finding create-23)
#: made visible rather than introduced. ``viewer_embed.Viewer.handle_event``
#: does ``import pygame`` inside the method, because the panel is fed pygame
#: events by the shell (``pygame.MOUSEBUTTONDOWN`` and friends) and nothing
#: else in the file touches it; it was always there, the pin just never
#: globbed this file. Function-scope and event constants only -- the claim this
#: pin keeps is that the *package* stays headless.
ALLOWED_WINDOW_IMPORTS: dict[str, set[str]] = {"viewer_embed.py": {"pygame"}}


def _package_of(path: Path) -> str:
    """The dotted package a file sits in, read off its path from the last
    ``realmspinner`` directory down (so a probe file written under a scratch
    ``realmspinner/studio/viewer/`` resolves like a real one)."""
    parts = path.resolve().parent.parts
    start = max(i for i, part in enumerate(parts) if part == "realmspinner")
    return ".".join(parts[start:])


def _outward(path: Path) -> set[str]:
    """Absolute module names this file imports, at any import depth.

    The 2026-10-11 audit, finding create-23: this kept only ``level == 0``
    ``ImportFrom`` nodes, and every inward import in this package is relative
    (``from ..kernels import ...``), so ``from ...service import jobs`` passed
    every test below. Relative imports are resolved against the file's own
    package here, and each imported name is also listed as a submodule
    (``from .. import service`` names ``realmspinner.studio.service``)."""
    found: set[str] = set()
    package = _package_of(path)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")
                base = base[: len(base) - (node.level - 1)]
                module = ".".join(base + ([node.module] if node.module else []))
            else:
                module = node.module or ""
            found.add(module)
            found.update(f"{module}.{alias.name}" for alias in node.names)
    return found


#: Two modules outside the package glob that are still the viewer: the panel
#: around it and its pose half. Named here because the package glob cannot
#: reach them (the 2026-10-11 audit, finding create-23).
EXTRA = (ENGINE.parent / "viewer_embed.py", ENGINE.parent / "_viewer_pose.py")


def _modules() -> list[Path]:
    """Every real module in the package -- ``__init__.py`` excluded, since it
    is the docstring making the claim rather than a module the claim is about
    (the finding's own count, "2 of 17 modules", is of these)."""
    return sorted(p for p in ENGINE.glob("*.py") if p.name != "__init__.py")


def _pinned() -> list[Path]:
    """The package's modules plus :data:`EXTRA`."""
    return [*_modules(), *EXTRA]


def test_there_are_thirteen_modules_to_check():
    """A glob that quietly started matching fewer files would measure less
    than it claims to; a glob that started matching more (a stray script
    dropped in the package) is worth noticing too.

    Fourteen since 2026-09-17, not the seventeen this was first written for
    -- P3 of ``dev/RESTRUCTURE.md`` moved ``math3d``, ``gltf`` and
    ``glbwrite`` to ``realmspinner/kernels/geom3d/``. Thirteen since
    2026-10-14: ``sheet.py``, the direction-strip preview, had no caller once
    Poser left (the 2026-10-14 audit, finding create-viewer-03). See the
    module docstring.
    """
    assert len(_modules()) == 13


def test_none_of_them_imports_a_window_or_an_immediate_mode_gui():
    """The claim ``viewer/__init__.py`` makes for the whole package, checked
    against every module in it rather than the two ``test_poser_imports.py``
    happens to also need for Poser's own reasons."""
    for path in _pinned():
        roots = {name.split(".")[0] for name in _outward(path)}
        roots -= ALLOWED_WINDOW_IMPORTS.get(path.name, set())
        assert not (roots & BANNED_ROOTS), f"{path.name} imports {roots & BANNED_ROOTS}"


def test_none_of_them_imports_the_service_layer():
    """The renderer answers questions about rays and bones; it does not know
    about jobs, sqlite or VRAM admission."""
    for path in _pinned():
        for name in _outward(path):
            assert not name.startswith("realmspinner.service"), f"{path.name} imports {name}"


def test_none_of_them_imports_the_queue_or_the_pipelines():
    for path in _pinned():
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
    roots = {r for path in _pinned() for r in {n.split(".")[0] for n in _outward(path)}}
    assert "moderngl" in roots


def test_the_viewer_import_pin_sees_a_relative_import_of_the_service_layer(tmp_path):
    """The 2026-10-11 audit, finding create-23: ``_outward`` skipped every
    relative import, and every inward import here is relative, so a viewer
    module could say ``from ...service import jobs`` and pass the whole file.
    A probe module is written under a scratch ``realmspinner/studio/viewer``
    so it resolves exactly as a real one does."""
    probe_dir = tmp_path / "realmspinner" / "studio" / "viewer"
    probe_dir.mkdir(parents=True)
    probe = probe_dir / "probe.py"
    probe.write_text(
        "from ...service import jobs\n"
        "from ... import queue\n"
        "from .. import _q_thing\n"
        "from . import camera\n",
        encoding="utf-8",
    )
    found = _outward(probe)
    assert "realmspinner.service" in found
    assert "realmspinner.service.jobs" in found
    assert "realmspinner.queue" in found
    assert "realmspinner.studio._q_thing" in found
    assert "realmspinner.studio.viewer.camera" in found


def test_the_pin_covers_the_panel_and_its_pose_half():
    """``viewer_embed`` and ``_viewer_pose`` are the viewer too, and the
    package glob cannot see them (the same finding)."""
    names = {p.name for p in _pinned()}
    assert {"viewer_embed.py", "_viewer_pose.py"} <= names
    assert all(p.exists() for p in _pinned())
