"""What ``studio/plotter/`` is allowed to reach for, pinned exactly.

The package's whole claim is that every rule it has -- about where a tile lands,
what a gid means, what a ``.tmx`` may contain -- is assertable headlessly. That
claim is only worth anything while the imports stay honest, so a *new* outward
import is a failing test and a deliberate decision rather than something that
turns up in a review three months later.

This is the ``tests/modes/inker/test_sheetout.py`` pin applied to the second pure
package, and it is written the same way on purpose.

The tile vocabulary itself -- the gid word, the sliced atlas, the blob
collapse -- moved out to :mod:`warlock.kernels.grid2d` on 2026-08-18: the
second shared leaf after ``studio/undo.py``, reached for by every module here
that used to import ``.gid``, ``.tileset`` or ``.blob`` as a sibling.
"""

from __future__ import annotations

import ast
from pathlib import Path

from warlock.studio.modes.plotter import engine as plotter

ENGINE = Path(plotter.__file__).parent
PACKAGE = "warlock.studio.modes.plotter.engine"

#: ``(module, imported name)`` for every import that leaves the package.
#: :mod:`~warlock.core.undo` is the history engine the raster editor and Clay
#: already share -- as headless as this package is, and the reason a ``.wmap``
#: undo step and an ``.ora`` one obey the same byte budget. (2026-09-17: moved
#: from ``studio/undo.py`` to ``warlock/core/undo.py`` in P3 of
#: ``dev/RESTRUCTURE.md``, which resolves the reason this needed defending as
#: a sibling exception -- reaching down into ``core`` needs none.)
#: :mod:`~warlock.kernels.grid2d` (``studio/tilegrid/`` before the same move)
#: and its ``.tileset`` submodule are the shared tile vocabulary -- every
#: module that places, flips or slices a tile reaches for one or both.
OUTWARD_IMPORTS = {
    # The shared bounded zip reader. One rule for four container doors, and a
    # leaf for ``tilegrid``/``undo``'s reason exactly: the ``file_size`` sum
    # each of these carried is written by whoever wrote the archive, and a
    # fourth private copy of a security bound is a copy that stops agreeing.
    # The four guard leaves, and they are one finding rather than four: a bound
    # that eighteen call sites have to remember is a bound that holds at
    # seventeen. ``zipguard`` said so first and the rest are the same sentence
    # about a different declared number -- an image's pixel count, a ``.npy``
    # header's shape, an XML document's DTD and nesting depth. Shared leaves,
    # not sibling engines, so this package is free to reach for them. P3
    # folded the four modules into one ``core/safeio/`` package; ``wmap.py``
    # reaches three of them through a single ``from ... import`` line, which
    # is one outward edge, not three, and ``tsx.py`` reaches the fourth the
    # same way even alone -- the collapse is about the package the name lives
    # in, not about how many names one file happens to need from it.
    ("wmap.py", "warlock.core.safeio"),
    ("tsx.py", "warlock.core.safeio"),
    ("_map_geometry.py", "warlock.kernels.grid2d"),
    ("_map_layers.py", "warlock.kernels.grid2d"),
    ("_map_layers.py", "warlock.kernels.grid2d.tileset"),
    ("_map_model.py", "warlock.kernels.grid2d.tileset"),
    ("_map_paint.py", "warlock.kernels.grid2d"),
    # ``MapDoc.set_stamp`` writes a block of gids and needs their dtype. The
    # same leaf every painting module here already reaches for, for the same
    # reason: a gid is what a cell *is*, and this package is the editor of them.
    ("tilemap.py", "warlock.kernels.grid2d"),
    ("_map_tilesets.py", "warlock.kernels.grid2d"),
    ("_map_tilesets.py", "warlock.kernels.grid2d.tileset"),
    ("edits.py", "warlock.core.undo"),
    ("render.py", "warlock.kernels.grid2d"),
    ("scene.py", "warlock.kernels.grid2d.tileset"),
    ("terrain.py", "warlock.kernels.grid2d"),
    ("terrain.py", "warlock.kernels.grid2d.tileset"),
    ("tilemap.py", "warlock.kernels.grid2d.tileset"),
    ("tilemap.py", "warlock.core.undo"),
    # The four-connected flood kernel, with the frontier dilation beside it as
    # the reference and the fallback.
    ("tools.py", "warlock.native"),
    # The tile-blit kernel, likewise: ``_blit_cells_native`` declines for every
    # case it cannot answer and ``_blit_over`` is what runs then. ``native`` is
    # a leaf -- ctypes and a DLL path, nothing from the studio -- so the edge
    # costs this package none of the headlessness it is pinned for.
    ("render.py", "warlock.native"),
    ("tmx.py", "warlock.kernels.grid2d"),
    ("tsx.py", "warlock.kernels.grid2d.wang"),
    ("wmap.py", "warlock.kernels.grid2d.wang"),
    ("tmx.py", "warlock.kernels.grid2d.tileset"),
    ("tools.py", "warlock.kernels.grid2d"),
    ("tsx.py", "warlock.kernels.grid2d"),
    ("tsx.py", "warlock.kernels.grid2d.tileset"),
    ("wmap.py", "warlock.kernels.grid2d"),
    ("wmap.py", "warlock.kernels.grid2d.tileset"),
}

BANNED_ROOTS = {"imgui", "imgui_bundle", "moderngl", "pygame", "OpenGL", "glfw"}

#: Imported inside the functions that need it, never at module scope. Pillow
#: costs a tenth of a second to import and most of this package never decodes a
#: pixel; the ``.wblk`` writer follows the same rule for the same reason.
LAZY_ONLY = {"PIL"}


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
                # Level 1 is a sibling inside the package. Level 2+ climbs out
                # of it, which is exactly what this is measuring.
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
    """A glob that matched nothing would make every test below vacuously
    pass."""
    assert len(_modules()) >= 12


def test_the_engine_never_imports_a_window():
    """No imgui, no moderngl, no pygame -- which is what makes every rule this
    package has about tiles assertable in a test like this one."""
    for path in _modules():
        roots = {name.split(".")[0] for name in _outward(path)}
        assert not (roots & BANNED_ROOTS), f"{path.name} imports {roots & BANNED_ROOTS}"


def test_the_engine_never_imports_the_service_layer():
    for path in _modules():
        for name in _outward(path):
            assert "warlock.service" not in name, f"{path.name} imports {name}"


def test_the_engine_never_imports_the_queue_or_the_pipelines():
    """Plotter has no business in a worker process, and neither of those has
    any business in a headless test of tile arithmetic."""
    for path in _modules():
        for name in _outward(path):
            assert not name.startswith("warlock.queue"), f"{path.name} imports {name}"
            assert not name.startswith("warlock.pipelines"), f"{path.name} imports {name}"
            # ``warlock._q_*`` too: the queue's worker halves are the same
            # dependency wearing a different name, and importing one of those
            # would drag torch behind a headless test as surely as importing
            # ``queue`` itself.
            assert not name.startswith("warlock._q"), f"{path.name} imports {name}"


def test_the_only_outward_imports_are_the_ones_written_down():
    found = {
        (path.name, name)
        for path in _modules()
        for name in _outward(path)
        if name.split(".")[0] == "warlock"
    }
    assert found == OUTWARD_IMPORTS


def test_pillow_is_never_imported_at_module_scope():
    """Three modules decode or encode a PNG and the rest never touch one.
    A top-level Pillow import would put that cost on importing ``gid``."""
    for path in _modules():
        assert not (_module_level(path) & LAZY_ONLY), f"{path.name} imports PIL eagerly"


def test_the_package_imports_with_no_optional_dependency_present():
    """Importing every module is the cheapest possible smoke test that the
    lazy-import rule above is actually being followed."""
    from warlock.studio.modes.plotter.engine import (  # noqa: F401
        _map_geometry,
        _map_layers,
        _map_model,
        _map_objects,
        _map_paint,
        _map_project,
        _map_tilesets,
        edits,
        layer_rows,
        pngio,
        project,
        props,
        render,
        scene,
        terrain,
        tilemap,
        tmx,
        tools,
        tsx,
        wmap,
    )
