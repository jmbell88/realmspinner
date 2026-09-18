"""What ``studio/troupe/`` is allowed to reach for, pinned exactly.

The ``tests/modes/inker/test_sheetout.py`` rule at its fourth instance, after
``packwright`` and ``plotter``: a headless engine imports no window, no
``service`` and no queue, so it can be tested without a GPU, driven from a
worker, and read by somebody who does not have to learn the app to follow it.

Troupe's outward set is *empty* today, deliberately. It owns a frame table and
a layout; the moment it needs an atlas ceiling or a trim rectangle it reaches
for ``kernels.sheet`` the way ``packwright.layout`` does -- and this file is
where that is written down rather than discovered.
"""

from __future__ import annotations

import ast
from pathlib import Path

from warlock.studio.modes.troupe import engine as troupe

ENGINE = Path(troupe.__file__).parent
PACKAGE = "warlock.studio.modes.troupe.engine"

# The 2026-09-15 audit, finding troupe-04: ``_modules()`` below globs only
# ``ENGINE`` (``studio/troupe/*.py``), so ``studio/troupe_state.py`` -- a
# sibling, not a member of the package -- had no pin at all. Named on its
# own rather than folded into ``_modules()``'s glob, per the orchestrator's
# call for this batch: widening what the existing pin's glob matches would
# also silently adopt whatever else later lands beside it in ``studio/``.
STATE_MODULE = ENGINE.parent / "state.py"

OUTWARD_IMPORTS: set[tuple[str, str]] = set()

# The 2026-09-18 audit, finding troupe-02: ``test_troupe_state_stays_headless``
# banned three root families (a window, ``service``, the queue) but never
# pinned the *set* the way ``OUTWARD_IMPORTS`` closes it for the engine
# modules above -- so a warlock import that tripped none of those three bans
# (``warlock.config``, say) would sail through undetected. Empty today,
# deliberately, the same claim ``OUTWARD_IMPORTS`` makes for the engine: the
# docstring's "these touch nothing but ``ctx.state.troupe``" is a promise
# this pin now keeps rather than merely states.
STATE_OUTWARD_IMPORTS: set[str] = set()

BANNED_ROOTS = {"imgui", "imgui_bundle", "moderngl", "pygame", "OpenGL", "glfw"}
LAZY_ONLY = {"PIL", "numpy"}


def _outward(path: Path) -> set[str]:
    found: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                found.add(node.module or "")
            elif node.level >= 2:
                base = PACKAGE.rsplit(".", node.level - 1)[0]
                if node.module:
                    found.add(f"{base}.{node.module}")
                else:
                    found.update(f"{base}.{alias.name}" for alias in node.names)
    return found


def _module_level(path: Path) -> set[str]:
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
    assert len(_modules()) >= 3


def test_the_engine_never_imports_a_window():
    for path in _modules():
        roots = {name.split(".")[0] for name in _outward(path)}
        assert not (roots & BANNED_ROOTS), f"{path.name} imports {roots & BANNED_ROOTS}"


def test_the_engine_never_imports_the_service_layer():
    for path in _modules():
        for name in _outward(path):
            assert "warlock.service" not in name, f"{path.name} imports {name}"


def test_the_engine_never_imports_the_queue():
    for path in _modules():
        for name in _outward(path):
            assert not name.startswith("warlock.queue"), f"{path.name} imports {name}"
            assert not name.startswith("warlock._q"), f"{path.name} imports {name}"


def test_the_engine_never_imports_the_raster_editor():
    """Troupe *produces* a document for Inker to open; the handoff lives on the
    Inker side (``sheetin``), which is what keeps the frame table readable
    without dragging the editor in behind it."""
    for path in _modules():
        for name in _outward(path):
            assert not name.startswith("warlock.kernels.pixel"), (
                f"{path.name} imports {name}"
            )


def test_the_only_outward_imports_are_the_ones_written_down():
    found = {
        (path.name, name)
        for path in _modules()
        for name in _outward(path)
        if name.split(".")[0] == "warlock"
    }
    assert found == OUTWARD_IMPORTS


def test_pillow_and_numpy_are_never_imported_at_module_scope():
    """The spec is arithmetic and the reader is ``crop``; neither should cost a
    numpy import to read a frame count."""
    for path in _modules():
        assert not (_module_level(path) & LAZY_ONLY), f"{path.name} imports eagerly"


def test_the_shipped_layout_table_is_part_of_the_package():
    assert (ENGINE / "data" / "layout.json").is_file()


def test_every_module_imports():
    from warlock.studio.modes.troupe.engine import qa, spec, ulpc  # noqa: F401


def test_troupe_state_stays_headless():
    """troupe-04: ``studio/troupe_state.py`` is Troupe's session state, split
    out of the controller precisely so a pane can read it without pulling in
    the half that talks to ``service`` and the task runner (see its own
    docstring) -- a promise this suite never actually checked."""
    assert STATE_MODULE.is_file()
    roots = {name.split(".")[0] for name in _outward(STATE_MODULE)}
    assert not (roots & BANNED_ROOTS), f"{STATE_MODULE.name} imports {roots & BANNED_ROOTS}"
    for name in _outward(STATE_MODULE):
        assert "warlock.service" not in name, f"{STATE_MODULE.name} imports {name}"
        assert not name.startswith("warlock.queue"), f"{STATE_MODULE.name} imports {name}"
        assert not name.startswith("warlock._q"), f"{STATE_MODULE.name} imports {name}"


def test_the_only_outward_imports_on_state_module_are_the_ones_written_down():
    """troupe-02 (2026-09-18 audit): the three bans above (a window,
    ``service``, the queue) do not add up to a closed set -- an import like
    ``warlock.config`` trips none of them and would sail through undetected,
    the same hole ``OUTWARD_IMPORTS`` closes for the engine modules. Pinned
    exactly, the ``test_the_only_outward_imports_are_the_ones_written_down``
    shape applied to ``state.py``."""
    found = {name for name in _outward(STATE_MODULE) if name.split(".")[0] == "warlock"}
    assert found == STATE_OUTWARD_IMPORTS
