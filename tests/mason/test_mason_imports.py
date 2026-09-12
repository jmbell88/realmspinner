"""What ``studio/mason/`` is allowed to reach for, pinned exactly.

**Written before the package it pins.** The scene engine's whole claim is that
an arrangement of assets -- what inherits what, what a group's override reaches,
where an instance's copies land, what a sculpt brush costs -- is assertable with
no GL context, no imgui and no service layer. That claim is worth exactly as
much as the imports staying honest, and the cheapest moment to say so is before
there is anything to say it about: a pin written afterwards is a description,
and a pin written first is a contract.

Two sets, because they answer different questions and move at different rates.

:data:`CEILING` is the contract, and it does not move as the stages land: the
shared history engine, the viewer's pure modules, the container-level GLB
reader, and the four guard leaves. Anything outside it is a design change and
should read as one.

:data:`OUTWARD_IMPORTS` is the exact ``(module, imported name)`` set that is
there *today*, which is the gate a hand-kept ceiling cannot be -- it catches the
import that is inside the contract but that nobody decided to make. It grows a
row at a time as the serializer, the exporters and the asset resolver arrive,
and each row is a line in a diff someone had to write.

The sibling ban is :mod:`_pure_packages`' derivation rather than a seventh hand
list; see that module for the six disagreeing lists that is a reaction to.

Written the same way as ``tests/clay/test_clay_imports.py`` and
``tests/plotter/test_plotter_imports.py`` on purpose.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest
from _pure_packages import siblings_of

from warlock.studio import mason

ENGINE = Path(mason.__file__).parent
PACKAGE = "warlock.studio.mason"

#: Everything this package may *ever* reach for. The contract, not the tally.
#:
#: :mod:`~warlock.studio.undo` is the shared history engine, reached for the
#: same way Clay and Plotter reach for it -- a scene's undo step is a pair of
#: callbacks and a byte cost like every other, and a fourth private notion of
#: what one step is would be a fourth place for a gesture to fold wrongly.
#:
#: ``viewer.gltf`` is the ``clay/document.py`` argument in a second document:
#: a Mason material **is** a ``gltf.Material`` and a light's fields **are**
#: ``KHR_lights_punctual``'s, because the export is the definition and a
#: parallel type buys a conversion function and a place for the two to drift.
#: ``viewer.math3d`` is the one place quaternions are XYZW in this project.
#: ``viewer.picking`` is the ray arithmetic and the BVH -- picking a scene is
#: picking each of its items, and a second Moller-Trumbore here would be a
#: second set of answers about degenerate triangles and hits behind the origin.
#:
#: ``warlock.glbio`` is the container-level GLB parser, for the reason
#: ``clay/glbimport.py`` reaches for it: a declared-count preflight has to read
#: the JSON chunk before anything decodes it, and there is one such parser.
#:
#: The four guard leaves are one finding wearing four names -- a bound that
#: eighteen call sites have to remember is a bound that holds at seventeen --
#: and they are leaves rather than sibling engines, so this package is free to
#: reach for them.
CEILING = frozenset(
    {
        "warlock.studio.undo",
        "warlock.studio.viewer",
        "warlock.glbio",
        "warlock.studio.zipguard",
        "warlock.studio.npyguard",
        "warlock.studio.pixelguard",
    }
)

#: ``(module, imported name)`` for every import that leaves the package, today.
OUTWARD_IMPORTS = {
    ("nodes.py", "warlock.studio.viewer"),
    ("refs.py", "warlock.studio.viewer"),
    ("terrain.py", "warlock.studio.viewer"),
    ("edits.py", "warlock.studio.undo"),
    ("edits.py", "warlock.studio.viewer"),
    ("document.py", "warlock.studio.undo"),
    ("document.py", "warlock.studio.viewer"),
    ("scene.py", "warlock.studio.viewer"),
    ("ops.py", "warlock.studio.viewer"),
    ("pick.py", "warlock.studio.viewer"),
    ("serialize.py", "warlock.studio.viewer"),
    ("serialize.py", "warlock.studio.zipguard"),
    ("serialize.py", "warlock.studio.npyguard"),
    ("serialize.py", "warlock.studio.pixelguard"),
    # The three exporters. ``gltfout`` is the row that moved
    # :data:`VIEWER_MODULES` from three names to four -- see its comment
    # below -- and ``objout`` and ``manifest`` reach for the viewer only for
    # the types they are writing out (``gltf.Primitive``/``gltf.Material``,
    # and ``math3d.decompose`` for the manifest's world TRS).
    ("gltfout.py", "warlock.studio.viewer"),
    ("manifest.py", "warlock.studio.viewer"),
    ("objout.py", "warlock.studio.viewer"),
}

#: Which modules of the viewer, since the entry above is recorded at package
#: granularity the way ``test_clay_imports`` records it. The viewer package
#: also holds the GL-side model, the renderer's programs and the offscreen
#: context, and reaching for one of those is the import this pin exists to
#: catch: it would make the scene engine untestable in exactly the lane that
#: is supposed to test all of it.
#:
#: ``glbwrite`` is the fourth, and it arrived with Stage D's exporters rather
#: than being reserved up front -- which is the point of pinning the tally
#: separately from the ceiling. It is the *writer*, not the GL layer: there is
#: one GLB writer in this project and the loader beside it is the only thing
#: that can test it, so a Mason-owned writer would have been a second home for
#: the four container details ``glbwrite``'s own docstring names as easy to
#: get wrong (the POSITION accessor's min/max, four-byte view alignment, the
#: index component width, the two chunks' different padding) with no loader to
#: round-trip against. Adding it was a decision; this line is where it was
#: written down.
VIEWER_MODULES = {"gltf", "math3d", "picking", "glbwrite"}

BANNED_ROOTS = {"imgui", "imgui_bundle", "moderngl", "pygame", "OpenGL", "glfw"}

#: Imported inside the functions that need it, never at module scope. Two
#: modules touch a pixel -- ``serialize`` writes a material override's texture
#: into the document's zip and ``objout`` writes a base-colour map beside the
#: MTL -- and a top-level Pillow import would put a tenth of a second onto
#: importing ``nodes``, which every test in this directory does.
LAZY_ONLY = {"PIL", "trimesh"}


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
    """A glob that matched nothing would make every test below vacuously pass."""
    assert len(_modules()) >= 8


def test_the_engine_never_imports_a_window():
    """No imgui, no moderngl, no pygame -- which is what makes every rule this
    package has about a scene assertable in a test like this one."""
    for path in _modules():
        roots = {name.split(".")[0] for name in _outward(path)}
        assert not (roots & BANNED_ROOTS), f"{path.name} imports {roots & BANNED_ROOTS}"


def test_the_engine_never_imports_the_service_layer():
    """A ``LibraryRef`` is a job id and never a path, and this is the test that
    says so: resolving one means ``svc.config.job_dir``, which is the studio
    layer's business. The moment this package could resolve a reference itself,
    every test of what a scene *is* would need a service and a library on disk.
    """
    for path in _modules():
        for name in _outward(path):
            assert "warlock.service" not in name, f"{path.name} imports {name}"


def test_the_engine_never_imports_the_queue_or_the_pipelines():
    """Neither runs anywhere a scene walk does, and dragging either in would put
    torch and a job queue behind a test of what a group's override reaches."""
    for path in _modules():
        for name in _outward(path):
            assert not name.startswith("warlock.queue"), f"{path.name} imports {name}"
            assert not name.startswith("warlock.pipelines"), f"{path.name} imports {name}"
            # ``warlock._q_*`` too: the queue's worker halves are the same
            # dependency wearing a different name.
            assert not name.startswith("warlock._q"), f"{path.name} imports {name}"


@pytest.mark.parametrize("other", siblings_of("mason"))
def test_the_engine_never_imports_another_headless_package(other):
    """Packages that are pure for the same reason are not packages that may
    reach for each other.

    Parametrized over what is in the tree rather than over a written list, so
    the headless package added after Mason enrols itself here on the day it
    lands -- see :mod:`_pure_packages` for the six hand lists that failed open
    and made that worth doing.

    The one this test is really about is ``clay``. Mason places primitives and
    Clay owns the fifteen generators that build them, so the obvious move is to
    import ``clay.primitives`` and call it -- and that is exactly the import
    this refuses. A ``PrimitiveRef`` resolves through the same ``GeometrySource``
    callback a ``LibraryRef`` does: the library path has to be a callback
    anyway, one resolution mechanism beats two, and it keeps this package's
    reach a leaf rather than a chain.
    """
    for path in _modules():
        for name in _outward(path):
            assert f"warlock.studio.{other}" not in name, f"{path.name} imports {name}"


def test_the_sibling_ban_is_not_empty():
    """The parametrize above is derived, and a derivation that returned nothing
    would make it a test that runs zero cases and reports green."""
    siblings = siblings_of("mason")
    assert "clay" in siblings and "plotter" in siblings
    assert "mason" not in siblings


def test_nothing_is_reached_for_outside_the_pinned_ceiling():
    """The contract, which does not move as the stages land."""
    for path in _modules():
        for name in _outward(path):
            if name.split(".")[0] != "warlock":
                continue
            assert name in CEILING, f"{path.name} imports {name}, which is outside the ceiling"


def test_the_only_outward_imports_are_the_ones_written_down():
    found = {
        (path.name, name)
        for path in _modules()
        for name in _outward(path)
        if name.split(".")[0] == "warlock"
    }
    assert found == OUTWARD_IMPORTS


def test_only_the_four_pinned_modules_of_the_viewer_are_reached_for():
    """The allowlist above is at package granularity; this says *which*.

    Named for the set rather than for the count it happened to have when it
    was written: it said "three" until Stage D's exporters reached for
    ``glbwrite``, and a test whose name is a number is a test that has to be
    renamed every time the thing it guards legitimately grows -- which is
    exactly the moment nobody wants to be editing a gate's name.
    """
    reached = {
        alias.name
        for path in _modules()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("viewer")
        for alias in node.names
    }
    assert reached == VIEWER_MODULES


def test_pillow_is_never_imported_at_module_scope():
    for path in _modules():
        assert not (_module_level(path) & LAZY_ONLY), f"{path.name} imports Pillow eagerly"


def test_the_package_imports_with_no_optional_dependency_present():
    """The cheapest possible smoke test that the lazy-import rule above is
    actually being followed.

    Derived from ``_modules()`` rather than written out, for the reason Clay's
    equivalent needed rather than merely deserved: a hand-kept import block had
    already drifted there, and a list naming twenty-four of twenty-five modules
    passes exactly as green as a correct one.
    """
    for path in _modules():
        if path.stem == "__init__":
            continue
        importlib.import_module(f"{PACKAGE}.{path.stem}")
