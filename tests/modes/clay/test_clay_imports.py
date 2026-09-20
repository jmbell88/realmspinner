"""What ``studio/clay/`` is allowed to reach for, pinned exactly.

Clay was the second of the four pure packages and the last to get this test,
which is the only reason it reads as a formality: the imports were already
clean, so nothing here is a fix. What it buys is the same thing the other three
pins buy -- the package's whole claim is that every rule it has about geometry
is assertable headlessly, and that claim is worth exactly as much as the imports
staying honest, so a *new* outward import is a failing test and a deliberate
decision rather than something found in a review three months later.

Written the same way as ``tests/modes/plotter/test_plotter_imports.py`` on purpose.
"""

from __future__ import annotations

import ast
from pathlib import Path

from _pure_packages import dotted_root, siblings_of

from warlock.kernels import mesh as clay

ENGINE = Path(clay.__file__).parent
PACKAGE = "warlock.kernels.mesh"

#: ``(module, imported name)`` for every import that leaves the package.
#: :mod:`~warlock.core.undo` is the history engine the raster editor already
#: shares -- it was extracted out of the raster editor *for* Clay, so a pure
#: Clay reaching back into the raster editor for its own history would defeat
#: the move. 2026-09-17: P3 of ``dev/RESTRUCTURE.md`` moved this module from
#: ``studio/undo.py`` to ``warlock/core/undo.py``, which resolves the reason
#: this entry needed defending in the first place -- a kernel reaching down
#: into ``core`` is just the ordinary shape of the layer table now, not a
#: reach back into the shell for a tool the raster editor happens to also
#: use. ``geom3d.gltf`` and ``geom3d.math3d`` are the ``sheetout.py``
#: argument: a Clay material **is** a ``gltf.Material`` rather than a parallel
#: type that would need a conversion function and a place for the two to drift,
#: and quaternions are XYZW in exactly one place in this project. Both lived
#: at ``studio/viewer/{gltf,math3d}.py`` until the same P3 move put them in
#: the kernel they always logically were, alongside ``glbio``.
OUTWARD_IMPORTS = {
    # The shared bounded zip reader. One rule for four container doors, and a
    # leaf for ``grid2d``/``undo``'s reason exactly: the ``file_size`` sum
    # each of these carried is written by whoever wrote the archive, and a
    # fourth private copy of a security bound is a copy that stops agreeing.
    # The four guard leaves, and they are one finding rather than four: a bound
    # that eighteen call sites have to remember is a bound that holds at
    # seventeen. ``zipguard`` said so first and the rest are the same sentence
    # about a different declared number -- an image's pixel count, a ``.npy``
    # header's shape, an XML document's DTD and nesting depth. Shared leaves,
    # not sibling engines, so this package is free to reach for them. P3
    # folded the three modules into one ``core/safeio/`` package and
    # ``serialize.py`` reaches all three through a single ``from ... import``
    # line, so there is exactly one outward edge to record now, not three.
    ("serialize.py", "warlock.core.safeio"),
    ("document.py", "warlock.core.undo"),
    ("drag.py", "warlock.kernels.geom3d"),
    ("document.py", "warlock.kernels.geom3d"),
    ("edits.py", "warlock.core.undo"),
    # Added deliberately on 2026-09-06 (the audit's clay-01): deleting or
    # duplicating a multi-object selection pushed one step per object, so one
    # ``Delete`` took three ``Ctrl+Z`` presses and the first landed on a state
    # the user had never made. Bundling the gesture needs ``CompoundEdit``, from
    # the same shared history engine ``document.py`` and ``edits.py`` already
    # reach for -- not a fourth private notion of what one step is.
    ("selection.py", "warlock.core.undo"),
    # H01: the declared-count preflight reads a GLB's JSON chunk before
    # ``gltf.load`` decodes anything, and ``glbio.split_glb`` is the one
    # container-level parser this project has -- the same one ``gltf``
    # itself is built on, both now siblings inside ``kernels/geom3d/``. One
    # entry, not two: ``glbimport.py`` reaches ``geom3d`` for ``glbio``,
    # ``gltf`` and ``math3d`` through relative imports of the same package.
    ("glbimport.py", "warlock.kernels.geom3d"),
    # 2026-09-19, Clay tranche 1: an OBJ's ``usemtl``/MTL colours become
    # ``gltf.Material`` palette slots, glbimport's reason exactly -- a Clay
    # material is a ``gltf.Material``, never a parallel type.
    ("objimport.py", "warlock.kernels.geom3d"),
    ("ops.py", "warlock.kernels.geom3d"),
    # Added deliberately on 2026-09-06 (the audit's clay-08): grounding a
    # figure preset has to know where its *built* geometry ends, not just
    # where its bone landmark sits, so ``presets.build`` places each part
    # through ``math3d.compose`` the same way ``drag.py``, ``ops.py``
    # and ``document.py`` already do -- one quaternion convention, not a
    # second one invented for this file.
    ("presets.py", "warlock.kernels.geom3d"),
    # analyze.py composes each object's world transform the same way
    # ops.py/document.py/presets.py already do, via ``math3d.compose`` --
    # not a second quaternion convention for a module that otherwise never
    # touches the viewport.
    ("analyze.py", "warlock.kernels.geom3d"),
    # serialize.py writes a material override straight out as a
    # ``gltf.Material`` -- the same "the export is the definition" reasoning
    # as the rest of this list, for the one file that also reaches
    # ``core.safeio`` above.
    ("serialize.py", "warlock.kernels.geom3d"),
    # Clay tranche 2, the modifier stack: radial-array spins each copy about
    # the object's own local origin, the same ``math3d.compose``/
    # ``quat_from_axis_angle`` pair ``ops.py``'s ``rotated_about_origin``
    # already reaches for -- one quaternion convention, not a second one for
    # a modifier that happens to rotate too.
    ("ops_modifiers.py", "warlock.kernels.geom3d"),
    # Clay tranche 3, scene structure: readiness's ``scale``/``pivot``/
    # ``transforms`` checks measure a document's *world* placement, not local
    # TRS (a parented object's own fields are relative to its parent, not
    # what an engine importing the document sees) -- so ``validate`` decomposes
    # ``doc.world_matrix`` onto a duck-typed copy of each object before any
    # check runs, the same ``math3d.compose``/``decompose`` pair ``ops.py``,
    # ``document.py`` and ``analyze.py`` already reach for.
    ("readiness.py", "warlock.kernels.geom3d"),
}

#: Which modules of ``kernels.geom3d``, since the entry above is recorded at
#: package granularity the way ``test_packwright_imports`` records
#: ``pipelines``. ``gltf``/``math3d`` lived under ``studio/viewer/`` until P3
#: moved them; ``glbio`` was already a standalone top-level module
#: (``warlock/glbio.py``) that the same move put in the same kernel package,
#: and ``glbimport.py``'s H01 preflight is what reaches for it (see
#: :data:`OUTWARD_IMPORTS`). The viewer package that remains still holds the
#: GL-side loader and the renderer's programs, and reaching for one of those
#: is the import this pin exists to catch.
VIEWER_MODULES = {"gltf", "math3d", "glbio"}

BANNED_ROOTS = {"imgui", "imgui_bundle", "moderngl", "pygame", "OpenGL", "glfw"}

#: Imported inside the functions that need it, never at module scope. Only the
#: serializer touches a pixel at all, and a top-level Pillow import would put a
#: tenth of a second onto importing ``mesh``. ``trimesh`` joins it for the same
#: reason wearing different clothes: it is reached for by exactly one function
#: in ``ops_boolean``, it drags a CSG kernel behind it, and this package is
#: imported to answer questions about what an extrude does to a UV.
#: ``scipy`` joins the other two on the same rule, for ``analyze.py``:
#: ``cKDTree`` and ``csgraph.connected_components`` are reached for by a
#: handful of functions in one module, and a top-level import would put a
#: whole second numerics stack behind every other Clay module that imports
#: this package for an unrelated question.
LAZY_ONLY = {"PIL", "trimesh", "manifold3d", "scipy"}


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
    package has about geometry assertable in a test like this one."""
    for path in _modules():
        roots = {name.split(".")[0] for name in _outward(path)}
        assert not (roots & BANNED_ROOTS), f"{path.name} imports {roots & BANNED_ROOTS}"


def test_the_engine_never_imports_the_service_layer():
    for path in _modules():
        for name in _outward(path):
            assert "warlock.service" not in name, f"{path.name} imports {name}"


def test_the_engine_never_imports_the_queue_or_the_pipelines():
    """Neither runs anywhere a mesh op does, and dragging either in would put
    torch and a job queue behind a test of what an extrude does to a UV."""
    for path in _modules():
        for name in _outward(path):
            assert not name.startswith("warlock.queue"), f"{path.name} imports {name}"
            assert not name.startswith("warlock.pipelines"), f"{path.name} imports {name}"
            # ``warlock._q_*`` too: the queue's worker halves are the same
            # dependency wearing a different name, and importing one of those
            # would drag torch behind a headless test as surely as importing
            # ``queue`` itself.
            assert not name.startswith("warlock._q"), f"{path.name} imports {name}"


#: ``geom3d`` is a shared kernel leaf this package legitimately imports (see
#: :data:`OUTWARD_IMPORTS`), not a peer engine -- the same exception
#: ``tests/modes/mason/test_mason_imports.py`` and ``tests/modes/inker/test_inker_imports.py``
#: record for the same import.
SHARED_LEAVES = frozenset({"geom3d"})


def test_the_engine_never_imports_the_other_pure_packages():
    """Packages that are pure for the same reason are not packages that may
    reach for each other: the raster editor's undo lives in ``core.undo``
    precisely so Clay does not have to import the raster editor.

    Derived over :func:`_pure_packages.siblings_of` rather than the
    ``("inker", "plotter", "packwright")`` this used to hard-code: that list
    predates P3 of ``dev/RESTRUCTURE.md`` and would have gone silently vacuous
    for ``inker`` the day it renamed to ``warlock.kernels.pixel`` -- a literal
    check for ``"warlock.studio.inker"`` bans an import string nothing in the
    tree has written since. :func:`_pure_packages.dotted_root` looks up each
    sibling's real import prefix instead of assuming it still hangs off
    ``warlock.studio``.
    """
    for other in siblings_of("mesh", allowed=SHARED_LEAVES):
        root = dotted_root(other)
        for path in _modules():
            for name in _outward(path):
                assert not (name == root or name.startswith(root + ".")), (
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


def test_only_three_modules_of_geom3d_are_reached_for():
    """The allowlist above is at package granularity; this says *which*
    modules, the way ``test_packwright_imports`` does for ``pipelines``.

    Named for three now, not two: checks ``geom3d``, not ``viewer`` --
    P3 of ``dev/RESTRUCTURE.md`` moved ``gltf``/``math3d`` there and put
    ``glbio`` (already its own module) in the same kernel package, and this
    package no longer imports ``studio.viewer`` at all.
    """
    reached = {
        alias.name
        for path in _modules()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("geom3d")
        for alias in node.names
    }
    assert reached == VIEWER_MODULES


def test_pillow_is_never_imported_at_module_scope():
    for path in _modules():
        assert not (_module_level(path) & LAZY_ONLY), f"{path.name} imports PIL eagerly"


def test_the_package_imports_with_no_optional_dependency_present():
    """Importing every module is the cheapest possible smoke test that the
    lazy-import rule above is actually being followed.

    Derived from ``_modules()`` rather than written out, which is the same
    argument every other derived list in this codebase rests on and which this
    test needed rather than merely deserved: it *was* a hand-kept import
    block, and it had already drifted -- ``select.py`` was missing from it,
    silently, so the one module holding the element-selection verbs was the
    one module this smoke test never smoked. A list naming twenty-four of
    twenty-five modules passes exactly as green as a correct one, which is
    what makes the hand-kept version worse than useless here.
    """
    import importlib

    for path in _modules():
        if path.stem == "__init__":
            continue
        importlib.import_module(f"{PACKAGE}.{path.stem}")
