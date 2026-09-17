"""What Poser's five pure modules are allowed to reach for, pinned exactly.

The ``tests/inker/test_sheetout.py`` pin, fifth instance -- with one structural
departure the others do not need. Clay, Inker, Plotter and Packwright each own a
*package*, so their pins glob a directory. Poser owns no package: its pure half
is three modules at the root of ``warlock`` (``poselib``, ``clipmaps``,
``cliptransfer``) and two inside the viewer (``pose``, ``bonelines``), and the
rest of it is panes. So the five are named, and a tripwire below fails if one
of them ever moves.

**``rigging.py`` is gone from this list, and deliberately not replaced
one-for-one.** P4 of ``dev/RESTRUCTURE.md`` (2026-09-17) split it into
``warlock.kernels.rig`` -- a shared Layer-1 kernel Troupe and the character
pipeline read exactly as much as Poser does, not a module Poser owns the way
it owned the single file. Pinning it here the way ``poselib``/``clipmaps``/
``cliptransfer`` are pinned would mean re-deciding, in a Poser-named file,
which of six submodules a Troupe caller may reach for -- so it is not named
here at all; each of the five modules below that used to import ``rigging``
now names the specific ``kernels.rig`` submodule(s) it actually needs in its
own ``OUTWARD_IMPORTS`` row, exactly like any other outward dependency.

They are pinned for the same reason the packages are, plus one of their own:
``poselib`` is what a service module reads a stored pose through. ``clipmaps``
is "Import clip"'s bone-name tables -- the mapping side of converting an
external animation (Mixamo, Rigify) onto a Warlock template rig -- and it
exists specifically so that conversion is decidable with no Blender, the same
argument ``kernels.rig`` already makes for its own half. ``cliptransfer``
restates its own bounds rather than reaching for ``warlock.pipelines.sheet``,
for the same reason. All three claim in their own docstrings to be usable
with no studio at all, and the two subprocess checks at the bottom make that
claim executable instead of merely stated.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import warlock

ROOT = Path(warlock.__file__).parent

#: ``relative path -> the package a relative import inside it resolves against``.
MODULES = {
    "poselib.py": "warlock",
    "clipmaps.py": "warlock",
    "cliptransfer.py": "warlock",
    "studio/viewer/pose.py": "warlock.studio.viewer",
    "studio/viewer/bonelines.py": "warlock.studio.viewer",
}

#: Every import that names something under ``warlock``, per module. Viewer
#: siblings are listed too -- unlike the package pins there is no "inside the
#: package" to be exempt, and the point of naming ``gltf`` and ``math3d`` here
#: is that the list is the whole dependency, not the part that left a directory.
OUTWARD_IMPORTS = {
    # The storage half of the library reaches into the rig kernel for ids/
    # staged-write helpers, the template registry and pose validation
    # (``store``/``templates``/``poses``) -- named as one package, not three
    # modules, because :func:`_outward` resolves ``from .kernels.rig import
    # a, b, c`` to the package itself (a dotted ``node.module`` always wins
    # over the imported names; see the function), the same coarseness
    # ``warlock.rigging`` had as a single file. Nothing else: a stored pose is
    # decidable with no service, no studio and no Blender, which is what
    # test_poselib.py stands on.
    "poselib.py": {"warlock.kernels.rig"},
    # The bone-name tables: which template a map targets, and validating a
    # map's bones against that template's own registry -- the rig kernel's
    # template registry (``templates``), reached the same package-granular
    # way :data:`poselib.py`'s row above is.
    "clipmaps.py": {"warlock.kernels.rig"},
    # The pure host math for "Import clip": which bone maps where
    # (``clipmaps``) and the target template's own rest pose, duration bounds
    # and clip-name rules (the rig kernel's ``templates``/``cliplib``).
    # Deliberately not ``warlock.pipelines.sheet`` --
    # ``test_none_of_them_imports_the_queue_or_the_pipelines`` refuses that
    # from every module pinned here, so ``sheet.slerp``,
    # ``sheet.MAX_CLIP_FRAMES`` and ``poselib.MAX_ROOT_TRANSLATION`` are
    # restated in ``cliptransfer.py`` instead, each pinned back to its
    # source of truth by a test in ``tests/test_cliptransfer.py``.
    "cliptransfer.py": {"warlock.kernels.rig", "warlock.clipmaps"},
    # The editor: rotations and mirroring, skeleton structure edits and
    # ``RigError``, all from the rig kernel now, plus matrices and the node
    # graph from the viewer's own.
    "studio/viewer/pose.py": {
        "warlock.poselib",
        "warlock.kernels.rig",
        # The shared undo engine, which is stdlib-only and has no opinion about
        # what an edit edits -- Clay borrows it for the same reason. Adding it
        # keeps the pose stack in the editor, where both entry points into pose
        # editing can reach one history, rather than in a pane that owns it
        # twice. It brings nothing imgui-shaped with it, which the headless
        # import assertion below is what actually guarantees.
        "warlock.core.undo",
        "warlock.kernels.geom3d.gltf",
        "warlock.kernels.geom3d",
    },
    # The GPU half, so it reaches only for viewer siblings.
    "studio/viewer/bonelines.py": {
        "warlock.studio.viewer.markers",
        "warlock.kernels.geom3d",
        "warlock.studio.viewer.render",
    },
}

BANNED_ROOTS = {"imgui", "imgui_bundle", "pygame", "OpenGL", "glfw"}

#: moderngl is banned everywhere except ``bonelines``, which *is* the GPU half:
#: it builds the line buffers, and a draw list is not expressible without the
#: context type. The other four are asserted headlessly and may never gain it.
MODERNGL_ALLOWED = {"studio/viewer/bonelines.py"}

# No LAZY_ONLY section here, deliberately: the package pins have one because
# their modules encode PNGs, and none of these five touches Pillow at all. A
# lazy-import test over five modules that never import it would pass forever
# without measuring anything.


def _outward(rel: str) -> set[str]:
    """Absolute module names this file imports, relative levels resolved."""
    package = MODULES[rel]
    found: set[str] = set()
    tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                found.add(node.module or "")
            else:
                base = package.rsplit(".", node.level - 1)[0]
                if node.module:
                    found.add(f"{base}.{node.module}")
                else:
                    found.update(f"{base}.{alias.name}" for alias in node.names)
    return found


def _run(stubs: tuple[str, ...], imports: str) -> subprocess.CompletedProcess:
    """Import something in a fresh interpreter with modules stubbed to None.

    ``sys.modules[name] = None`` makes any attempt to import it raise, so a
    hidden import fails loudly rather than succeeding on a developer machine
    that happens to have the whole studio installed. A subprocess rather than a
    reload, the ``test_rigging_stays_importable_with_no_bpy_anywhere`` rule:
    reloading mints new function objects and breaks the identity other tests
    assert about ``mirror_quaternion``.
    """
    script = (
        "import sys\n"
        f"for name in {stubs!r}: sys.modules[name] = None\n"
        f"import {imports}\n"
    )
    return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)


def test_the_pinned_modules_are_all_still_there():
    """A named-module pin goes vacuous by a rename rather than by a bad glob."""
    for rel in MODULES:
        assert (ROOT / rel).is_file(), f"{rel} moved; the pin below now measures nothing"


def test_none_of_them_imports_a_window():
    for rel in MODULES:
        roots = {name.split(".")[0] for name in _outward(rel)}
        assert not (roots & BANNED_ROOTS), f"{rel} imports {roots & BANNED_ROOTS}"


def test_only_the_gpu_half_imports_moderngl():
    for rel in MODULES:
        roots = {name.split(".")[0] for name in _outward(rel)}
        has = "moderngl" in roots
        assert has == (rel in MODERNGL_ALLOWED), f"{rel}: moderngl={has}"


def test_none_of_them_imports_the_service_layer():
    """The dependency runs the other way: ``service/poses.py`` reads a record
    through ``poselib``. An import back would be a cycle and, worse, would put
    a store-wide lock behind a function that is documented as pure."""
    for rel in MODULES:
        for name in _outward(rel):
            assert "warlock.service" not in name, f"{rel} imports {name}"


def test_none_of_them_imports_the_queue_or_the_pipelines():
    for rel in MODULES:
        for name in _outward(rel):
            assert not name.startswith("warlock.queue"), f"{rel} imports {name}"
            assert not name.startswith("warlock.pipelines"), f"{rel} imports {name}"
            # ``warlock._q_*`` too: the queue's worker halves are the same
            # dependency wearing a different name, and importing one of those
            # would drag torch behind a headless test as surely as importing
            # ``queue`` itself.
            assert not name.startswith("warlock._q"), f"{rel} imports {name}"


def test_the_only_warlock_imports_are_the_ones_written_down():
    for rel in MODULES:
        found = {name for name in _outward(rel) if name.split(".")[0] == "warlock"}
        assert found == OUTWARD_IMPORTS[rel], rel


def test_the_storage_half_imports_with_no_studio_at_all():
    """``poselib``'s docstring says a stored pose is decidable without the app;
    ``kernels.rig`` (which ``poselib`` now imports) makes the same claim for
    its own half; ``clipmaps``' whole point is that a bone-name mapping is
    decidable the same way, with no Blender either; ``cliptransfer`` restates
    its own bounds from ``kernels.rig`` and ``clipmaps`` rather than importing
    ``warlock.pipelines.sheet`` for them, for the same reason. All these
    claims, executed -- importing ``poselib``, ``clipmaps`` and
    ``cliptransfer`` already pulls in every ``kernels.rig`` submodule each of
    them needs, so there is no separate module to name here for it."""
    proc = _run(
        ("imgui", "imgui_bundle", "moderngl", "pygame", "warlock.studio"),
        "warlock.poselib, warlock.clipmaps, warlock.cliptransfer",
    )
    assert proc.returncode == 0, proc.stderr


def test_the_viewer_half_imports_with_no_imgui_and_no_service():
    """moderngl is left real -- ``bonelines`` genuinely needs it. What is being
    measured is that the renderer still knows nothing about panels or the
    business layer, which is the split ``viewer/__init__`` claims."""
    proc = _run(
        ("imgui", "imgui_bundle", "pygame", "warlock.service"),
        "warlock.studio.viewer.pose, warlock.studio.viewer.bonelines",
    )
    assert proc.returncode == 0, proc.stderr


def test_no_module_but_blender_worker_imports_bpy():
    """bpy is process-global and, per CLAUDE.md, documented as *crashing* --
    not raising -- the interpreter on geometry trellis produces. The whole
    safety argument for keeping it to one subprocess module rests on nothing
    else ever importing it, and until the 2026-09-08 audit (poser-05) that
    rested on convention: this file's own pin above checks three named
    modules, and ``test_rigging.py`` checks the ``kernels.rig`` package
    specifically, but nothing scanned the rest of ``src/warlock`` for a
    stray ``import bpy``.
    """
    exempt = ROOT / "pipelines" / "blender_worker.py"
    offenders: list[str] = []
    for path in ROOT.rglob("*.py"):
        if path == exempt:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            hit = isinstance(node, ast.Import) and any(
                alias.name == "bpy" or alias.name.startswith("bpy.") for alias in node.names
            ) or isinstance(node, ast.ImportFrom) and node.module and (
                node.module == "bpy" or node.module.startswith("bpy.")
            )
            if hit:
                offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, offenders
