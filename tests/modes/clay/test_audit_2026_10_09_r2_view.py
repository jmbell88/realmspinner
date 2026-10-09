"""The 2026-10-09 second audit's Clay viewport findings: clay-58, 70, 81, 123, 125, 127.

(clay-126, the import pin for ``viewport_hints.py``, lives in ``test_clay_imports.py``
beside the pin it extends.)

The drag and cache tests ride the session ``gl`` fixture and skip where there
is no GPU, like every sibling in this package.
"""

from __future__ import annotations

import ast
import dataclasses
import io
import re
import tokenize
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio import viewport_hints as clay_hints
from realmspinner.studio.modes.clay.state import ClayState
from realmspinner.studio.modes.clay.ui import view as clay_view

RECT = (0.0, 0.0, 128.0, 96.0)


class _Ctx:
    def __init__(self) -> None:
        self.state = type("S", (), {"clay": ClayState()})()
        self.clay_view = None
        self.errors: list[str] = []

    def toast(self, message: str, level: str = "info") -> None:
        if level == "error":
            self.errors.append(message)


@pytest.fixture(autouse=True)
def _headless_mods(monkeypatch):
    import pygame

    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0, raising=False)


@pytest.fixture
def ctx():
    return _Ctx()


@pytest.fixture
def view(gl, ctx):
    v = clay_view.ClayView(gl, ctx)
    ctx.clay_view = v
    yield v
    v.release()


def _ready(view: Any, doc: bd.ClayDoc) -> None:
    view.frame_selection(doc)
    view.draw(doc, RECT, 0.0)
    view._last_mouse = (100.0, 48.0)  # off the pivot, or a scale has no ratio


def _quat_z(degrees: float) -> np.ndarray:
    return m3.quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), np.radians(degrees))


def _rotated_box(degrees: float, scale=(1.0, 3.0, 1.0)) -> tuple[bd.ClayDoc, int]:
    doc = bd.ClayDoc()
    obj = doc.add_object(
        bd.Obj(
            uid=bd.new_uid(),
            name="B",
            mesh=bp.box(),
            scale=m3.vec3(*scale),
            rotation=_quat_z(degrees),
        )
    )
    doc.select([obj.uid])
    return doc, obj.uid


def _extent(view: Any, doc: bd.ClayDoc) -> np.ndarray:
    lo, hi = view.world_bounds(doc)
    return np.round(hi - lo, 3)


def _scale_drag(view: Any, doc: bd.ClayDoc, *keys: str) -> None:
    assert view.begin_keyboard_drag(doc, "scale")
    for key in keys:
        assert view.drag_key(doc, key)


# --- clay-58: a world-axis scale of a rotated object ----------------------------


def test_a_world_x_scale_of_an_object_turned_ninety_degrees_about_z_grows_world_x_not_world_y(
    view, ctx
) -> None:
    """The 2026-10-09 second audit's clay-58: the readout said ``X (global)`` but
    the factor was multiplied into the object's own local X, so a shape turned
    90 degrees about Z stretched along world Y (extent [3, 2, 1] where [6, 1, 1]
    was asked for)."""
    doc, uid = _rotated_box(90.0)
    _ready(view, doc)
    assert np.allclose(_extent(view, doc), [3, 1, 1])

    _scale_drag(view, doc, "x", "2")
    assert np.allclose(_extent(view, doc), [6, 1, 1]), _extent(view, doc)
    view.settle_drag(doc)

    assert np.allclose(_extent(view, doc), [6, 1, 1])
    # World X is the object's local Y once it is turned a quarter about Z.
    assert np.allclose(doc.by_uid(uid).scale, [1, 6, 1], atol=1e-5), doc.by_uid(uid).scale
    assert not ctx.errors


def test_a_world_y_scale_of_an_object_turned_ninety_degrees_about_z_grows_world_y(
    view, ctx
) -> None:
    doc, uid = _rotated_box(90.0)
    _ready(view, doc)

    _scale_drag(view, doc, "y", "2")
    view.settle_drag(doc)

    assert np.allclose(_extent(view, doc), [3, 2, 1]), _extent(view, doc)
    assert np.allclose(doc.by_uid(uid).scale, [2, 3, 1], atol=1e-5), doc.by_uid(uid).scale
    assert not ctx.errors


def test_a_world_axis_scale_of_an_object_turned_thirty_degrees_is_refused_and_changes_nothing(
    view, ctx
) -> None:
    """A scale along a world axis of a shape turned 30 degrees is a shear, which
    a translate/rotate/scale cannot hold: it is refused by name rather than
    applied along the wrong axis."""
    doc, uid = _rotated_box(30.0)
    _ready(view, doc)
    obj = doc.by_uid(uid)
    was = (np.array(obj.translation), np.array(obj.rotation), np.array(obj.scale))
    head = doc.history.head

    _scale_drag(view, doc, "x", "2")
    view.settle_drag(doc)

    obj = doc.by_uid(uid)
    for before, after in zip(was, (obj.translation, obj.rotation, obj.scale), strict=True):
        assert np.allclose(before, after)
    assert doc.history.head == head, "a refused scale pushed a history step"
    assert len(ctx.errors) == 1, ctx.errors
    assert "Size card" in ctx.errors[0], ctx.errors


def test_a_uniform_scale_of_a_rotated_object_still_works(view, ctx) -> None:
    doc, uid = _rotated_box(30.0, scale=(1.0, 1.0, 1.0))
    _ready(view, doc)

    _scale_drag(view, doc, "2")
    view.settle_drag(doc)

    assert np.allclose(doc.by_uid(uid).scale, [2, 2, 2], atol=1e-5), doc.by_uid(uid).scale
    assert not ctx.errors


def test_a_world_axis_scale_of_an_unrotated_object_is_unchanged(view, ctx) -> None:
    doc, uid = _rotated_box(0.0)
    _ready(view, doc)

    _scale_drag(view, doc, "x", "2")
    view.settle_drag(doc)

    assert np.allclose(doc.by_uid(uid).scale, [2, 3, 1], atol=1e-5)
    assert not ctx.errors


# --- clay-70: the palette half of the cache key --------------------------------


def _two_boxes_on_two_slots() -> tuple[bd.ClayDoc, int, int, int]:
    """Box A on slot 0, box B on slot 1, and a slot 2 nobody uses."""
    doc = bd.ClayDoc()
    uids = []
    for i in range(2):
        obj = doc.add_object(
            bd.Obj(
                uid=bd.new_uid(),
                name=f"Box{i}",
                mesh=bp.box(),
                translation=m3.vec3(float(i) * 3.0, 0.0, 0.0),
            )
        )
        uids.append(obj.uid)
    doc.add_material()
    spare = doc.add_material()
    assert doc.repaint_object(uids[1], 1)
    return doc, uids[0], uids[1], spare


def _edit_slot(doc: bd.ClayDoc, index: int) -> None:
    doc.set_material(
        index, dataclasses.replace(doc.materials[index], base_color_factor=(0.2, 0.4, 0.8, 1.0))
    )


def test_editing_a_palette_slot_no_object_uses_rebuilds_no_object(view) -> None:
    """The 2026-10-09 second audit's clay-70: the key carried the identity of the
    whole palette, so recolouring a slot nothing uses rebuilt the GPU model of
    every visible object (200 of 200 in 123 ms)."""
    doc, _a, _b, spare = _two_boxes_on_two_slots()
    view.sync(doc)
    before = view.rebuilds

    _edit_slot(doc, spare)
    view.sync(doc)

    assert view.rebuilds == before


def test_editing_a_palette_slot_an_object_uses_rebuilds_only_that_object(view) -> None:
    doc, a, b, _spare = _two_boxes_on_two_slots()
    view.sync(doc)
    gpu_a, gpu_b = view._cache[a].gpu, view._cache[b].gpu
    before = view.rebuilds

    _edit_slot(doc, 1)
    view.sync(doc)

    assert view.rebuilds == before + 1
    assert view._cache[a].gpu is gpu_a, "an object that does not use the slot was rebuilt"
    assert view._cache[b].gpu is not gpu_b, "the object that uses the slot was not rebuilt"


def test_an_object_whose_default_slot_is_edited_rebuilds(view) -> None:
    doc, a, _b, _spare = _two_boxes_on_two_slots()
    view.sync(doc)
    gpu_a = view._cache[a].gpu

    _edit_slot(doc, 0)
    view.sync(doc)

    assert view._cache[a].gpu is not gpu_a


# --- clay-81 and clay-123: a typed element scale -------------------------------


def _face_box(mode: str = "face") -> tuple[bd.ClayDoc, int]:
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="B", mesh=bp.box()))
    doc.set_element_mode(mode)
    doc.set_element_sel(obj.uid, el.select_all(obj.mesh, "face"))
    return doc, obj.uid


def test_a_typed_zero_scale_on_one_axis_flattens_elements_and_only_a_negative_one_is_refused(
    view, ctx
) -> None:
    """The 2026-10-09 second audit's clay-81: zero on one axis puts the picked
    elements on a plane, which inverts nothing; only a strictly negative factor
    turns faces inside out. The first pass refused both, with a false reason
    for zero."""
    doc, uid = _face_box()
    _ready(view, doc)
    before = np.array(doc.by_uid(uid).mesh.positions)
    assert float(np.ptp(before[:, 0])) > 0.0

    _scale_drag(view, doc, "x", "0")
    view.settle_drag(doc)

    flat = doc.by_uid(uid).mesh.positions
    assert not ctx.errors, ctx.errors
    assert float(np.ptp(flat[:, 0])) < 1e-6, "zero did not flatten onto the plane"
    assert float(np.ptp(flat[:, 1])) == pytest.approx(float(np.ptp(before[:, 1])))

    # And the negative one is still refused, with a reason that is true of it.
    doc, uid = _face_box()
    ctx.errors.clear()
    _ready(view, doc)
    before = np.array(doc.by_uid(uid).mesh.positions)
    _scale_drag(view, doc, "x", "-", "1")
    view.settle_drag(doc)

    assert np.array_equal(doc.by_uid(uid).mesh.positions, before)
    assert len(ctx.errors) == 1, ctx.errors
    assert "Mirror" in ctx.errors[0] and "zero" not in ctx.errors[0], ctx.errors


def test_switching_from_a_refused_typed_scale_to_a_move_commits_without_the_scale_toast(
    view, ctx
) -> None:
    """The 2026-10-09 second audit's clay-123: ``_element_scale_refused`` stayed
    set after S, a typed -1, then G, so the move that followed committed with a
    false "scaling refused" error."""
    doc, uid = _face_box()
    _ready(view, doc)
    before = np.array(doc.by_uid(uid).mesh.positions)

    _scale_drag(view, doc, "-", "1")
    assert view.drag_key(doc, "g")
    view._motion(doc, (110.0, 50.0))
    view.settle_drag(doc)

    assert not ctx.errors, ctx.errors
    after = doc.by_uid(uid).mesh.positions
    assert not np.array_equal(after, before), "the move did not commit"
    assert np.allclose(after - before, (after - before)[0]), "the move was not a pure move"


# --- clay-125: docstrings and comments name modules that exist -----------------

SRC = Path(clay_view.__file__).resolve().parents[5]  # src/
PACKAGE_ROOT = SRC / "realmspinner"
STUDIO = PACKAGE_ROOT / "studio"
CLAY = STUDIO / "modes" / "clay"
REPO = SRC.parent

OWNED = (
    CLAY / "ui" / "_view_drag.py",
    CLAY / "ui" / "_view_cache.py",
    CLAY / "ui" / "view.py",
    CLAY / "ui" / "viewport.py",
    CLAY / "ui" / "_view_overlay.py",
    CLAY / "ui" / "_view_pick.py",
    CLAY / "ui" / "_view_bounds.py",
    STUDIO / "viewport_hints.py",
    STUDIO / "_view_frame.py",
    CLAY / "gestures.py",
    CLAY / "element_move.py",
)

#: The stale names the first audit's relocations left in prose. Each was a real
#: citation of a module that moved (``ui/hud.py`` -> ``ui/panes/hud.py``) or was
#: deleted (``agent_clay`` -> ``agent/``, ``clay.drag`` -> ``kernels/mesh/drag``).
STALE = (
    r"ui/hud\.py",
    r"ui/header\.py",
    r"\bclay_hud\b",
    r"\bclay_viewport\b",
    r"\bagent_clay\b",
    r"(?<!modes\.)\bclay\.(drag|scratch|elements|ops|pick|ui)\b",
    r"\bclay/pick\.py",
    r"\bhud\.hint_line",
    r"~\.clay\.",
    r"\bclay_ops\.",
)

_PATH = re.compile(r"(?<![\w/.])((?:\w+/)+\w+\.py)\b")


def _prose(path: Path) -> list[tuple[int, str]]:
    """``(line, text)`` for every comment and every docstring in a module."""
    source = path.read_text(encoding="utf-8")
    found: list[tuple[int, str]] = []
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.COMMENT:
            found.append((tok.start[0], tok.string))
    for node in ast.walk(ast.parse(source)):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ) and ast.get_docstring(node, clean=False):
            first = node.body[0]
            found.append((first.lineno, ast.get_docstring(node, clean=False) or ""))
    return found


def _resolves(token: str) -> bool:
    roots = (PACKAGE_ROOT, STUDIO, CLAY, CLAY / "ui", REPO)
    return any((root / token).is_file() for root in roots)


def test_no_clay_view_docstring_names_a_module_path_that_does_not_exist() -> None:
    """The 2026-10-09 second audit's clay-125: the view's prose still cited
    ``ui/hud.py``, ``clay_hud``, ``clay_viewport``, ``agent_clay``, ``clay.drag``,
    ``clay/pick.py``, ``clay.scratch`` and ``clay.elements`` -- every one a module
    that moved or was deleted -- so following a comment led nowhere."""
    stale = [re.compile(p) for p in STALE]
    problems: list[str] = []
    for path in OWNED:
        assert path.is_file(), path
        for line, text in _prose(path):
            for pattern in stale:
                for match in pattern.finditer(text):
                    problems.append(f"{path.name}:{line}: stale {match.group(0)!r}")
            for match in _PATH.finditer(text):
                if not _resolves(match.group(1)):
                    problems.append(f"{path.name}:{line}: no file {match.group(1)!r}")
    assert not problems, "\n".join(problems)


def test_the_overlay_module_docstring_does_not_key_the_cache_on_hover() -> None:
    """``_SelOverlay`` keys on ``(id(mesh), id(sel), mode)`` and holds hover
    apart so a cursor move rebuilds nothing; the module docstring said the key
    included ``hover``."""
    path = CLAY / "ui" / "_view_overlay.py"
    module_doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))
    assert module_doc is not None
    assert "mode, hover)" not in module_doc
    assert "id(sel), mode)" in module_doc


# --- clay-127: the statistics edge count ----------------------------------------


def _calls_of(source: str, function: str) -> list[str]:
    tree = ast.parse(source)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function)
    return [ast.unparse(n.func) for n in ast.walk(fn) if isinstance(n, ast.Call)]


def _has_row_wise_unique(source: str) -> bool:
    return any(name.endswith("unique") for name in _calls_of(source, "_unique_edges"))


def test_unique_edges_counts_without_a_row_wise_unique() -> None:
    """The 2026-10-09 second audit's clay-127: the speed claim was a 0.2 s
    wall-clock assertion in the default ``-n 8`` lane, which a busy runner
    fails. What made the count fast is structural -- one int64 key per edge and
    a 1-D sort, never ``np.unique(pairs, axis=0)`` -- so the structure is
    pinned instead (the equivalence with the row-wise count is
    ``test_the_statistics_edge_count_matches_the_row_unique_it_replaced``)."""
    source = Path(clay_hints.__file__).read_text(encoding="utf-8")
    assert not _has_row_wise_unique(source)
    assert any(name.endswith("sort") for name in _calls_of(source, "_unique_edges"))


def test_the_row_wise_unique_pin_notices_the_slow_form() -> None:
    source = Path(clay_hints.__file__).read_text(encoding="utf-8")
    slow = source.replace(
        "keys = np.sort(low * span + high)", "keys = np.unique(np.stack([low, high], 1), axis=0)"
    )
    assert slow != source
    assert _has_row_wise_unique(slow)
