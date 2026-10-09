"""The 2026-10-09 audit's Clay viewport findings: clay-10, 12, 22, 28, 45.

(clay-29, the import pin for ``gestures.py`` and ``element_move.py``, lives in
``test_clay_imports.py`` beside the pin it extends.)

The viewport owns a renderer, so the drag tests ride the session ``gl`` fixture
and skip where there is no GPU, like every sibling in this package.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio import viewport_hints as clay_hints
from realmspinner.studio.modes.clay import ops as clay_ops
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


def _boxes(count: int = 1, mode: str = "face") -> tuple[bd.ClayDoc, list[int]]:
    doc = bd.ClayDoc()
    uids = []
    for i in range(count):
        obj = doc.add_object(
            bd.Obj(
                uid=bd.new_uid(),
                name=f"Box{i}",
                mesh=bp.box(),
                translation=m3.vec3(float(i) * 3.0, 0.0, 0.0),
            )
        )
        uids.append(obj.uid)
    doc.set_element_mode(mode)
    return doc, uids


def _ready(view: Any, doc: bd.ClayDoc) -> None:
    view.frame_selection(doc)
    view.draw(doc, RECT, 0.0)
    view._last_mouse = (64.0, 48.0)


def _outward(mesh: Any) -> tuple[int, int]:
    normals = bm.face_normals(mesh)
    centroid = mesh.positions.mean(axis=0)
    out = 0
    for i in range(bm.face_count(mesh)):
        centre = mesh.positions[bm.face(mesh, i)].mean(axis=0)
        out += int(float((normals[i] * (centre - centroid)).sum()) > 0.0)
    return out, bm.face_count(mesh)


# --- clay-10: a typed non-positive scale on elements ---------------------------


def test_a_typed_negative_scale_on_selected_elements_never_leaves_inside_out_faces(
    view, ctx
) -> None:
    doc, (uid,) = _boxes()
    doc.set_element_sel(uid, el.select_all(doc.by_uid(uid).mesh, "face"))
    _ready(view, doc)
    view._last_mouse = (100.0, 48.0)  # off the pivot, or a scale has no ratio

    assert view.begin_keyboard_drag(doc, "scale")
    for key in ("x", "-", "1"):
        assert view.drag_key(doc, key)
    view.settle_drag(doc)

    mesh = doc.by_uid(uid).mesh
    outward, total = _outward(mesh)
    assert outward == total, f"{total - outward} of {total} faces came out inside-out"
    assert ctx.errors and "Mirror" in ctx.errors[-1], ctx.errors


def test_a_typed_zero_scale_on_selected_elements_is_not_refused(view, ctx) -> None:
    """The first pass refused zero along with the negative factors, and its
    toast said zero turns faces inside out. The 2026-10-09 second audit's
    clay-81: zero puts the picked elements on a plane and inverts nothing, so
    only a strictly negative factor is refused (the per-axis case is
    ``test_audit_2026_10_09_r2_view.py``'s)."""
    doc, (uid,) = _boxes()
    doc.set_element_sel(uid, el.select_all(doc.by_uid(uid).mesh, "face"))
    _ready(view, doc)
    view._last_mouse = (100.0, 48.0)

    assert view.begin_keyboard_drag(doc, "scale")
    assert view.drag_key(doc, "0")
    view.settle_drag(doc)

    assert not ctx.errors, ctx.errors
    assert float(np.ptp(doc.by_uid(uid).mesh.positions, axis=0).max()) < 1e-6


def test_a_typed_positive_scale_on_selected_elements_still_scales(view, ctx) -> None:
    doc, (uid,) = _boxes()
    doc.set_element_sel(uid, el.select_all(doc.by_uid(uid).mesh, "face"))
    _ready(view, doc)
    view._last_mouse = (100.0, 48.0)
    before = np.array(doc.by_uid(uid).mesh.positions)

    assert view.begin_keyboard_drag(doc, "scale")
    assert view.drag_key(doc, "2")
    view.settle_drag(doc)

    after = doc.by_uid(uid).mesh.positions
    assert not ctx.errors
    assert np.allclose(np.abs(after).max(), np.abs(before).max() * 2.0, atol=1e-5)


# --- clay-12: an op-drag preview skips the object that refuses ------------------


def test_an_op_drag_preview_shows_the_objects_that_accept_the_value_when_another_refuses(
    view, ctx
) -> None:
    doc, (a, b) = _boxes(count=2)
    doc.set_element_sel(a, el.ElementSel(faces=[0]))
    doc.set_element_sel(b, el.ElementSel(faces=[0]))
    _ready(view, doc)
    assert view.begin_op_drag(doc, clay_ops.get("inset"))
    od = view._op_drag
    refusing_mesh = od.bases[a][0]
    real = od.kernel

    def kernel(mesh: Any, sel: Any, **params: Any) -> Any:
        if mesh is refusing_mesh:
            raise el.OpError("That inset is too wide for Box0.")
        return real(mesh, sel, **params)

    od.kernel = kernel
    gpu_b = view._cache[b].gpu
    view._motion(doc, (100.0, 48.0))

    assert view._cache[b].gpu is not gpu_b, "the accepting object was not drawn"
    assert "Box0" in view.drag_hud, view.drag_hud
    view.cancel_drag(doc)


# --- clay-22: element_centre's key does not scan per picked object --------------


def test_element_centre_key_never_scans_the_document_per_picked_object(
    view, monkeypatch
) -> None:
    doc, uids = _boxes(count=64)
    for uid in uids:
        doc.set_element_sel(uid, el.ElementSel(faces=[0]))
    scans = {"n": 0}
    real = bd.ClayDoc.index_of

    def counting(self: Any, uid: int) -> int:
        scans["n"] += 1
        return real(self, uid)

    monkeypatch.setattr(bd.ClayDoc, "index_of", counting)

    view.element_centre(doc)  # cold: builds the memo
    cold = scans["n"]
    scans["n"] = 0
    view.element_centre(doc)  # warm: only the key is computed

    assert cold == 0, f"{cold} document scans building the centre"
    assert scans["n"] == 0, f"{scans['n']} document scans computing the memo key"


# --- clay-28: the statistics edge count -----------------------------------------


def _old_unique_edges(mesh: Any) -> int:
    loops = np.asarray(mesh.loops)
    starts = np.asarray(mesh.starts)
    if len(loops) == 0:
        return 0
    nxt = np.arange(1, len(loops) + 1, dtype="i8")
    nxt[starts[1:] - 1] = starts[:-1]
    pairs = np.sort(np.stack([loops, loops[nxt]], axis=1), axis=1)
    return int(len(np.unique(pairs, axis=0)))


def _random_mesh(rng: np.random.Generator, faces: int, verts: int) -> bm.Mesh:
    sizes = rng.integers(3, 6, size=faces)
    loops = np.concatenate([rng.integers(0, verts, size=s) for s in sizes]).astype("i4")
    starts = np.concatenate([[0], np.cumsum(sizes)]).astype("i4")
    return bm.Mesh(
        positions=rng.random((verts, 3)).astype("f4"),
        loops=loops,
        starts=starts,
        material=np.zeros(faces, dtype="i4"),
        smooth=np.zeros(faces, dtype=bool),
    )


def test_the_statistics_edge_count_matches_the_row_unique_it_replaced() -> None:
    rng = np.random.default_rng(7)
    for faces, verts in ((1, 3), (40, 12), (500, 90), (3000, 4000)):
        mesh = _random_mesh(rng, faces, verts)
        assert clay_hints._unique_edges(mesh) == _old_unique_edges(mesh)


def test_the_statistics_edge_count_agrees_with_the_row_unique_on_a_large_mesh() -> None:
    """Equivalence at 200k faces. The *speed* claim that used to ride on this
    test as a 0.2 s wall-clock assertion (a flake in the default ``-n 8`` lane --
    the 2026-10-09 second audit's clay-127) is pinned structurally instead, by
    ``test_audit_2026_10_09_r2_view.py::test_unique_edges_counts_without_a_row_wise_unique``."""
    rng = np.random.default_rng(3)
    mesh = _random_mesh(rng, 200_000, 100_000)
    assert clay_hints._unique_edges(mesh) == _old_unique_edges(mesh)


# --- clay-45: a drawn frame syncs the live camera ---------------------------------

VIEWPORT = Path(clay_view.__file__).with_name("viewport.py")


def _viewport_calls(source: str) -> list[str]:
    """Dotted names of the calls ``_clay_viewport`` makes, in source order."""
    tree = ast.parse(source)
    fn = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_clay_viewport"
    )
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
    calls.sort(key=lambda n: (n.lineno, n.col_offset))
    return [ast.unparse(n.func) for n in calls]


def _syncs_before_the_handoff(source: str) -> bool:
    names = _viewport_calls(source)
    if "clay_mode.sync_active_camera" not in names:
        return False
    sync = names.index("clay_mode.sync_active_camera")
    return sync < names.index("clay_mode.apply_camera") and sync < names.index("view.draw")


def test_a_drawn_clay_frame_syncs_the_live_camera_onto_the_drawn_tab() -> None:
    """The 2026-10-09 audit's clay-45: ``tab.view`` is what a journal autosave
    writes, and only this per-frame call keeps it current between a save and a
    tab switch. Nothing else pinned the call, so deleting it would have lost
    minutes of free orbiting to a crash with every test green."""
    assert _syncs_before_the_handoff(VIEWPORT.read_text(encoding="utf-8"))


def test_the_camera_pin_notices_a_frame_with_the_sync_call_removed() -> None:
    source = VIEWPORT.read_text(encoding="utf-8")
    stripped = source.replace("clay_mode.sync_active_camera(ctx)", "pass")
    assert stripped != source
    assert not _syncs_before_the_handoff(stripped)
