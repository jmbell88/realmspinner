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

from realmspinner.kernels import mesh as clay

ENGINE = Path(clay.__file__).parent
PACKAGE = "realmspinner.kernels.mesh"

#: ``(module, imported name)`` for every import that leaves the package.
#: :mod:`~realmspinner.core.undo` is the history engine the raster editor already
#: shares -- it was extracted out of the raster editor *for* Clay, so a pure
#: Clay reaching back into the raster editor for its own history would defeat
#: the move. 2026-09-17: P3 of ``dev/RESTRUCTURE.md`` moved this module from
#: ``studio/undo.py`` to ``realmspinner/core/undo.py``, which resolves the reason
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
    ("serialize.py", "realmspinner.core.safeio"),
    ("document.py", "realmspinner.core.undo"),
    ("drag.py", "realmspinner.kernels.geom3d"),
    ("document.py", "realmspinner.kernels.geom3d"),
    ("edits.py", "realmspinner.core.undo"),
    # clay-mesh-core-01 (2026-09-26 audit): the object-mode branch of
    # ``delete_selected`` used to hand-build its own ``CompoundEdit`` around a
    # hand-popped ``doc.objects.pop`` -- which never re-parented a removed
    # object's children the way ``remove_object`` does. It now makes one
    # ``doc.remove_objects`` call for the whole selection (which pushes its own
    # single step, so no gesture is opened) instead, so
    # ``selection.py`` no longer imports ``core.undo`` directly at all.
    # Removed from this table rather than left stale.
    # H01: the declared-count preflight reads a GLB's JSON chunk before
    # ``gltf.load`` decodes anything, and ``glbio.split_glb`` is the one
    # container-level parser this project has -- the same one ``gltf``
    # itself is built on, both now siblings inside ``kernels/geom3d/``. One
    # entry, not two: ``glbimport.py`` reaches ``geom3d`` for ``glbio``,
    # ``gltf`` and ``math3d`` through relative imports of the same package.
    ("glbimport.py", "realmspinner.kernels.geom3d"),
    # clay-io-01 (2026-09-26 audit): a GLB import with a non-identity scale/up
    # conjugates each object's local TRS by the same matrix its mesh is
    # transformed by (``m3.compose``/``decompose``), so a child node's world
    # placement stays in lockstep with its rescaled mesh instead of detaching
    # from it -- the same ``math3d`` this table already grants every other
    # transform-composing module.
    ("meshimport.py", "realmspinner.kernels.geom3d"),
    # 2026-09-19, Clay tranche 1: an OBJ's ``usemtl``/MTL colours become
    # ``gltf.Material`` palette slots, glbimport's reason exactly -- a Clay
    # material is a ``gltf.Material``, never a parallel type.
    ("objimport.py", "realmspinner.kernels.geom3d"),
    ("ops.py", "realmspinner.kernels.geom3d"),
    # serialize.py writes a material override straight out as a
    # ``gltf.Material`` -- the same "the export is the definition" reasoning
    # as the rest of this list, for the one file that also reaches
    # ``core.safeio`` above.
    ("serialize.py", "realmspinner.kernels.geom3d"),
    # rigimport.py reads a skinned ``gltf.Model`` (nodes, skins, primitives) into bones
    # and weighted meshes: ``gltf`` and ``math3d`` again, the same two ``glbimport.py``
    # reaches, and nothing outside the kernel.
    ("rigimport.py", "realmspinner.kernels.geom3d"),
    # animimport.py lays a ``gltf.Model``'s animations onto a document's bones by name:
    # ``gltf`` for the model and ``math3d`` for the quaternions, as ``rigimport.py`` does.
    ("animimport.py", "realmspinner.kernels.geom3d"),
    # legacy.py is the migration-only home of the old modifier kernels
    # (radial-array and mirror spin each copy through ``math3d.compose``,
    # the one quaternion convention ``ops.py`` already uses); only
    # ``serialize`` may import it, so a version-3 file still opens.
    ("legacy.py", "realmspinner.kernels.geom3d"),
    # The 2026-10-07 audit's clay-25: the bake's vertex-growth guard asks the
    # same ``npyguard.MAX_ARRAY_BYTES`` the reader and the writer hold an array
    # to, so a mesh it hands back is one ``serialize`` can both save and reopen.
    ("legacy.py", "realmspinner.core.safeio"),
}

#: Which modules of ``kernels.geom3d``, since the entry above is recorded at
#: package granularity the way ``test_packwright_imports`` records
#: ``pipelines``. ``gltf``/``math3d`` lived under ``studio/viewer/`` until P3
#: moved them; ``glbio`` was already a standalone top-level module
#: (``realmspinner/glbio.py``) that the same move put in the same kernel package,
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
#: ``scipy`` joins the other two on the same rule: ``cKDTree`` and
#: ``csgraph.connected_components`` are reached for by a handful of functions,
#: and a top-level import would put a whole second numerics stack behind every
#: other Clay module that imports this package for an unrelated question.
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
            assert "realmspinner.service" not in name, f"{path.name} imports {name}"


def test_the_engine_never_imports_the_queue_or_the_pipelines():
    """Neither runs anywhere a mesh op does, and dragging either in would put
    torch and a job queue behind a test of what an extrude does to a UV."""
    for path in _modules():
        for name in _outward(path):
            assert not name.startswith("realmspinner.queue"), f"{path.name} imports {name}"
            assert not name.startswith("realmspinner.pipelines"), f"{path.name} imports {name}"
            # ``realmspinner._q_*`` too: the queue's worker halves are the same
            # dependency wearing a different name, and importing one of those
            # would drag torch behind a headless test as surely as importing
            # ``queue`` itself.
            assert not name.startswith("realmspinner._q"), f"{path.name} imports {name}"


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
    for ``inker`` the day it renamed to ``realmspinner.kernels.pixel`` -- a literal
    check for ``"realmspinner.studio.inker"`` bans an import string nothing in the
    tree has written since. :func:`_pure_packages.dotted_root` looks up each
    sibling's real import prefix instead of assuming it still hangs off
    ``realmspinner.studio``.
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
        if name.split(".")[0] == "realmspinner"
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


# --- the viewport-facing logic modules (the 2026-10-09 audit's clay-29) ---------
#
# ``studio/modes/clay/gestures.py`` and ``element_move.py`` both open with "no
# imgui, no GL" and are what the headless tests drive, but nothing enrolled them
# here: this file's ``_modules()`` only globs ``kernels/mesh``. A pygame import
# in the press classifier would have passed every test.

CLAY_MODE = (
    Path(__file__).resolve().parents[3] / "src" / "realmspinner" / "studio" / "modes" / "clay"
)
CLAY_MODE_PACKAGE = "realmspinner.studio.modes.clay"
HEADLESS_LOGIC = ("gestures.py", "element_move.py")
FORBIDDEN_FOR_LOGIC = ("realmspinner.service", "realmspinner.queue", "realmspinner.pipelines")


def _absolute_imports(source: str, package: str = CLAY_MODE_PACKAGE) -> set[str]:
    """Every module a source imports, relative ones resolved against *package*
    (the clay mode unless told otherwise), function-scope imports included."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                found.add(node.module or "")
            else:
                base = package.rsplit(".", node.level - 1)[0]
                found.add(f"{base}.{node.module}" if node.module else base)
    return found


def _headless_violations(source: str, package: str = CLAY_MODE_PACKAGE) -> list[str]:
    bad = []
    for name in sorted(_absolute_imports(source, package)):
        banned = name.split(".")[0] in BANNED_ROOTS
        service = any(name == f or name.startswith(f + ".") for f in FORBIDDEN_FOR_LOGIC)
        if banned or service:
            bad.append(name)
    return bad


def test_gestures_and_element_move_import_no_imgui_moderngl_or_pygame() -> None:
    for name in HEADLESS_LOGIC:
        path = CLAY_MODE / name
        assert path.is_file(), path
        assert _headless_violations(path.read_text(encoding="utf-8")) == [], name


def test_the_headless_logic_pin_catches_a_module_that_imports_a_window(tmp_path) -> None:
    scratch = tmp_path / "element_move.py"
    scratch.write_text(
        (CLAY_MODE / "element_move.py").read_text(encoding="utf-8")
        + "\nimport imgui\nfrom ....service import jobs\n",
        encoding="utf-8",
    )
    assert _headless_violations(scratch.read_text(encoding="utf-8")) == [
        "imgui",
        "realmspinner.service",
    ]


# --- studio/viewport_hints.py (the 2026-10-09 second audit's clay-126) --------------
#
# ``viewport_hints.py`` opens by claiming it imports nothing outward, and the
# pin above enrolled only the two modules under ``modes/clay``: a pygame or imgui
# import in the hint line or the axis ball would have passed every test. It
# lives one package up (``studio``), so its relative imports resolve against
# that package. ``studio/_view_frame.py`` is deliberately *not* enrolled: it
# imports pygame inside ``_mods`` on purpose (it is the one place the viewports
# read the modifier keys), so a pin that banned pygame there would be wrong.

STUDIO = CLAY_MODE.parents[1]
STUDIO_PACKAGE = "realmspinner.studio"


def test_viewport_hints_imports_no_imgui_moderngl_or_pygame() -> None:
    path = STUDIO / "viewport_hints.py"
    assert path.is_file(), path
    assert _headless_violations(path.read_text(encoding="utf-8"), STUDIO_PACKAGE) == []


def test_the_viewport_hints_pin_catches_a_module_that_imports_a_window() -> None:
    source = (STUDIO / "viewport_hints.py").read_text(encoding="utf-8")
    bad = source + "\nimport pygame\nfrom ..service import jobs\n"
    assert _headless_violations(bad, STUDIO_PACKAGE) == ["pygame", "realmspinner.service"]
