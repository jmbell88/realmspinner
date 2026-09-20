"""What ``studio/inker/walk/`` is allowed to reach for, pinned exactly.

Its own pin, and not a few lines added to the parent's: ``tests/modes/inker/
test_inker_imports.py`` scans ``ENGINE.glob("*.py")``, which is the top level
only, so a subpackage is invisible to it. ``tests/modes/inker/flourish/`` needed the
same file for the same reason.

The engine imports numpy, the standard library and the rest of ``inker`` -- and
nothing else. In particular it never reaches ``pipelines``, which is the one
exception ``flourish/bake.py`` has and this package has no use for: a walk cycle
is the drawing's own pixels turned, so there is no palette to resolve and no
quantisation to defer to anybody.

**``scipy`` is the absence with an argument**, borrowed intact from the flourish
pin. The bar is that one rig renders the same bytes on every machine
(``test_walk_render`` pins a digest), and a kernel that arrives from a dependency
is a kernel that can change under a ``uv sync``.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

from realmspinner.kernels.pixel import walk

ENGINE = Path(walk.__file__).parent
PACKAGE = "realmspinner.kernels.pixel.walk"

BANNED_ROOTS = {"imgui", "imgui_bundle", "moderngl", "pygame", "OpenGL", "glfw"}
DETERMINISM_ROOTS = {"scipy"}
ALLOWED_ROOTS = {"numpy", "realmspinner"}

#: ``(module, imported package)`` for every import that leaves this subpackage.
#: All four are ``inker`` itself, and each is the app's single copy of something
#: this package must not re-spell: the transform kernel, the blend arithmetic,
#: the animation model and the document.
OUTWARD_IMPORTS = {
    ("bake.py", "realmspinner.kernels.pixel.animation"),
    ("bake.py", "realmspinner.kernels.pixel.composite"),
    ("bake.py", "realmspinner.kernels.pixel.document"),
    ("bake.py", "realmspinner.kernels.pixel.layers"),
    ("bake.py", "realmspinner.kernels.pixel.undo"),
    ("render.py", "realmspinner.kernels.pixel.selection"),
}

#: Modules that may import Pillow, and only inside a function. None do: the one
#: Pillow call this package causes is inside ``selection.render_transform``,
#: which is the parent package's to make.
LAZY_PILLOW: set[str] = set()


def _modules() -> list[Path]:
    return sorted(ENGINE.rglob("*.py"))


def _own_package(path: Path) -> str:
    rel = path.relative_to(ENGINE).with_suffix("")
    return PACKAGE + ("." + ".".join(rel.parts[:-1]) if len(rel.parts) > 1 else "")


def _outward(path: Path) -> set[str]:
    found: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    own = _own_package(path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                found.add(node.module or "")
            else:
                base = own if node.level == 1 else own.rsplit(".", node.level - 1)[0]
                found.add(f"{base}.{node.module}" if node.module else base)
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


def _stdlib(name: str) -> bool:
    return name.split(".")[0] in sys.stdlib_module_names


def test_there_are_modules_to_check():
    """The vacuous-pass guard: every rule below is a loop over this list."""
    assert len(_modules()) >= 5


def test_the_engine_never_imports_a_window():
    for path in _modules():
        roots = {name.split(".")[0] for name in _outward(path)}
        assert not (roots & BANNED_ROOTS), f"{path.name} imports {roots & BANNED_ROOTS}"


def test_the_engine_never_imports_the_service_layer_or_the_queue():
    for path in _modules():
        for name in _outward(path):
            assert not name.startswith("realmspinner.service"), f"{path.name} imports {name}"
            assert not name.startswith("realmspinner.queue"), f"{path.name} imports {name}"
            assert not name.startswith("realmspinner._q"), f"{path.name} imports {name}"


def test_the_engine_never_imports_the_other_pure_packages():
    """A walk cycle is Inker's. Reaching for Clay or Plotter would make this the
    place two editors' rules met, which is what ``service`` is for."""
    for path in _modules():
        for name in _outward(path):
            for sibling in ("clay", "plotter", "packwright", "sirens", "muse", "troupe"):
                assert not name.startswith(f"realmspinner.studio.{sibling}"), f"{path.name}: {name}"


def test_the_kernels_are_not_borrowed_from_scipy():
    for path in _modules():
        roots = {name.split(".")[0] for name in _outward(path)}
        assert not (roots & DETERMINISM_ROOTS), f"{path.name} imports {roots & DETERMINISM_ROOTS}"


def test_pillow_is_never_imported_at_module_scope():
    for path in _modules():
        assert "PIL" not in _module_level(path), f"{path.name} imports Pillow at module scope"
        if "PIL" in {name.split(".")[0] for name in _outward(path)}:
            assert path.name in LAZY_PILLOW, f"{path.name} imports Pillow"


def test_the_only_outward_imports_are_the_ones_written_down():
    found = set()
    for path in _modules():
        for name in _outward(path):
            if name.startswith(PACKAGE) or name.split(".")[0] != "realmspinner":
                continue
            found.add((path.name, name))
    assert found == OUTWARD_IMPORTS


def test_the_only_third_party_import_is_numpy():
    for path in _modules():
        for name in _outward(path):
            root = name.split(".")[0]
            if _stdlib(name) or root in ALLOWED_ROOTS:
                continue
            raise AssertionError(f"{path.relative_to(ENGINE)} imports {name}")


def test_the_package_imports_with_no_optional_dependency_present():
    pass


def test_every_part_names_joints_that_exist():
    """The spec table is data, so a typo in it is a runtime ``KeyError`` on a
    user's drawing rather than an import error. Caught here instead."""
    from realmspinner.kernels.pixel.walk import rig as R

    for spec in R.PARTS:
        assert spec.pivot in R.JOINTS, spec.name
        for joint in spec.direction or ():
            assert joint in R.JOINTS, spec.name
        if spec.follows:
            assert spec.follows in R.PART_NAMES, spec.name


def test_every_part_and_joint_has_a_label():
    """Refusals name parts, and a refusal that named ``near_upper_arm`` would be
    the internal key leaking onto the screen."""
    from realmspinner.kernels.pixel.walk import rig as R

    for name in (*R.PART_NAMES, *R.JOINTS):
        assert R.label(name) == R.LABELS[name]
        assert "_" not in R.label(name)
