"""The 2026-10-14 audit's dead rig and sheet chains: each is gone, and stays gone.

A symbol that only a test calls is kept alive by the test, and the test then reads
as proof the feature works. Five findings were that shape: the joint-measuring
route (poser-27), the meshless armature op and the sheet's socket and margin keys
(poser-29), the character sheet's cell arithmetic (poser-22) and the pixel-sheet
restyle helpers (inker-80). Each claim here is named after the thing that must not
come back; the two generic ones (a worker op has a production spec builder, a
sheet helper has a production caller) are what stops the next one arriving the same
way.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

import pytest

from realmspinner.kernels import charsheet
from realmspinner.kernels import sheet as sheetlib
from realmspinner.kernels.pixel import sheetin
from realmspinner.kernels.rig import blender_spec
from realmspinner.pipelines import blender_worker, pixelize, pixelsheet

SRC = Path(__file__).resolve().parents[1] / "src" / "realmspinner"


def _src_files() -> list[Path]:
    return [p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts]


def _referenced_names(files: list[Path]) -> dict[str, set[Path]]:
    """Every name a call, attribute or bare reference in ``files`` mentions."""
    seen: dict[str, set[Path]] = {}
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            name = None
            if isinstance(node, ast.Name):
                name = node.id
            elif isinstance(node, ast.Attribute):
                name = node.attr
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    seen.setdefault(alias.name, set()).add(path)
            if name is not None:
                seen.setdefault(name, set()).add(path)
    return seen


# --- poser-27: the joint-measuring chain -----------------------------------------


def test_the_jointfit_module_and_its_spec_key_are_gone():
    assert not (SRC / "pipelines" / "jointfit.py").exists()
    assert not hasattr(blender_spec, "JOINT_SOURCES")
    assert "joints" not in inspect.signature(blender_spec.rig_spec).parameters
    assert "jointfit" not in inspect.getsource(blender_worker)


# --- poser-29: the armature unit -------------------------------------------------


def test_the_meshless_armature_op_and_its_spec_builder_are_gone():
    assert not hasattr(blender_spec, "armature_spec")
    assert not hasattr(blender_worker, "op_armature")
    assert "armature" not in blender_worker.OPS


def test_sheet_spec_and_the_worker_carry_no_margin_or_socket_surface():
    params = inspect.signature(blender_spec.sheet_spec).parameters
    assert "margin" not in params and "sockets" not in params
    for name in ("_socket_world_point", "_sphere_corners", "_view_depth"):
        assert not hasattr(blender_worker, name), name
    assert "sockets" not in inspect.signature(blender_worker._pose_union).parameters
    source = inspect.getsource(blender_worker.op_sheet)
    assert 'spec.get("margin")' not in source and 'spec.get("sockets")' not in source


def _spec_builders_by_op() -> dict[str, tuple[str, Path]]:
    """``{op: (function name, file)}`` for every function that builds a worker spec."""
    found: dict[str, tuple[str, Path]] = {}
    for path in _src_files():
        if path.name == "blender_worker.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef):
                continue
            for node in ast.walk(fn):
                if not isinstance(node, ast.Dict):
                    continue
                for key, value in zip(node.keys, node.values, strict=True):
                    if (
                        isinstance(key, ast.Constant)
                        and key.value == "op"
                        and isinstance(value, ast.Constant)
                        and isinstance(value.value, str)
                    ):
                        found[value.value] = (fn.name, path)
    return found


def test_every_blender_worker_op_has_a_spec_builder_with_a_production_caller():
    """``armature`` had both halves in ``src/`` and a caller only in tests: the pose
    editor whose preview it served left, and ``test_animated_glb`` was the one thing
    still sending it (the 2026-10-14 audit, finding poser-29). An op nothing sends is
    dead weight in a subprocess whose every line costs a Blender start to verify."""
    builders = _spec_builders_by_op()
    references = _referenced_names(_src_files())
    missing = [op for op in blender_worker.OPS if op not in builders]
    assert not missing, f"worker ops with no spec builder in src/: {missing}"
    orphans = []
    for op in blender_worker.OPS:
        name, defined_in = builders[op]
        callers = references.get(name, set()) - {defined_in}
        # A builder called from its own module (a sibling helper) still has a
        # production caller; only a name nothing in src/ mentions elsewhere is dead.
        if not callers and not _called_within(defined_in, name):
            orphans.append(f"{op} -> {name} ({defined_in.name})")
    assert not orphans, f"worker ops whose spec builder only tests call: {orphans}"


def _called_within(path: Path, name: str) -> bool:
    """Whether the module itself calls ``name`` -- from a sibling function or at
    import time (``charsheet.COMPASS_16`` is built by calling ``compass_name``)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            callee = node.func
            called = callee.id if isinstance(callee, ast.Name) else getattr(callee, "attr", "")
            if called == name:
                return True
    return False


# --- poser-22: the character sheet's cell arithmetic -----------------------------


@pytest.mark.parametrize(
    "name",
    [
        "point_in_cell",
        "pivot_in_cell",
        "frames_per_direction",
        "RENDER_SIZE",
        "WARN_CELLS",
        "MOVEMENT_MIN_FRAMES",
        "subset_indices",
        "check_subset",
    ],
)
def test_the_character_sheet_cell_helpers_are_gone(name):
    assert not hasattr(charsheet, name), name
    assert name not in charsheet.__all__


# --- inker-80: the restyle and sheet helpers -------------------------------------


@pytest.mark.parametrize(
    ("module", "names"),
    [
        (
            pixelsheet,
            [
                "Band", "NotRestylable", "check_restylable", "bands", "band_init",
                "crop_back", "remask", "downscale", "pixel_sidecar", "PIXEL_SHEET_VERSION",
            ],
        ),
        (pixelize, ["reduce_frames"]),
        (sheetin, ["document_from_sheet"]),
        (sheetlib, ["interpolate", "compose_cells"]),
    ],
)
def test_no_kernel_sheet_helper_exists_only_for_a_retired_pipeline(module, names):
    """The retired helpers by name, and the two keyword arguments that only they
    used: ``pack(only=)`` and ``sidecar(front_yaw=)``."""
    for name in names:
        assert not hasattr(module, name), f"{module.__name__}.{name}"
    assert "only" not in inspect.signature(sheetlib.pack).parameters
    assert "front_yaw" not in inspect.signature(sheetlib.sidecar).parameters


_SHEET_MODULES = (
    "realmspinner.kernels.sheet",
    "realmspinner.kernels.charsheet",
    "realmspinner.pipelines.pixelsheet",
    "realmspinner.pipelines.pixelize",
    "realmspinner.kernels.pixel.sheetin",
)

#: Public helpers that are documented surface a caller outside ``src/`` (an agent's
#: client, a script) may use. Empty on purpose: a name added here needs a reason.
_PUBLIC_WITHOUT_A_SRC_CALLER: set[str] = set()


@pytest.mark.parametrize("module_name", _SHEET_MODULES)
def test_every_public_sheet_function_has_a_production_caller(module_name):
    """The generic half of the above: the next helper left behind by a retired
    pipeline fails here instead of waiting for an audit to find it."""
    module = importlib.import_module(module_name)
    defined_in = Path(inspect.getsourcefile(module) or "")
    references = _referenced_names(_src_files())
    dead = []
    for name, obj in vars(module).items():
        if name.startswith("_") or not inspect.isfunction(obj) or obj.__module__ != module_name:
            continue
        if name in _PUBLIC_WITHOUT_A_SRC_CALLER:
            continue
        callers = references.get(name, set()) - {defined_in}
        if not callers and not _called_within(defined_in, name):
            dead.append(name)
    assert not dead, f"{module_name}: public functions nothing in src/ calls: {dead}"
