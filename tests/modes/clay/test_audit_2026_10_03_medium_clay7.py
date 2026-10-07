"""Regression tests for the 2026-10-03 audit's Medium findings clay-61, 62, 66, 68 and 69
(Clay's ops tail and panes). Each test's name is the claim it makes about the
unfixed code.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from _ui_context import imgui_context

from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio.modes.clay import ops as clay_ops
from realmspinner.studio.modes.clay.ui.panes import outliner as clay_outliner
from realmspinner.studio.modes.clay.ui.panes import uv as clay_uv


class _Toasts:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.info: list[str] = []


class _Ctx:
    def __init__(self) -> None:
        self.toasts = _Toasts()
        self.inline = True

    def toast(self, message: str, level: str = "info") -> None:
        (self.toasts.errors if level == "error" else self.toasts.info).append(message)


@pytest.fixture
def ui(monkeypatch):
    with imgui_context(monkeypatch) as imgui:
        yield imgui


def _add(doc: bd.ClayDoc, name: str, mesh: bm.Mesh | None = None, **kw: Any) -> bd.Obj:
    return doc.add_object(
        bd.Obj(uid=bd.new_uid(), name=name, mesh=bp.box() if mesh is None else mesh, **kw)
    )


# --- clay-61: copies of a parent and its child hang off the copied parent ----


def _family() -> tuple[bd.ClayDoc, int, int]:
    doc = bd.ClayDoc()
    parent = _add(doc, "Parent", translation=m3.vec3(1.0, 0.0, 0.0))
    child = _add(doc, "Child", translation=m3.vec3(0.0, 1.0, 0.0))
    doc.set_parent(child.uid, parent.uid, keep_world=False)
    doc.select([parent.uid, child.uid])
    return doc, parent.uid, child.uid


def _copies(doc: bd.ClayDoc, originals: set[int]) -> tuple[list[bd.Obj], list[bd.Obj]]:
    new = [o for o in doc.objects if o.uid not in originals]
    return (
        [o for o in new if o.name.startswith("Parent")],
        [o for o in new if o.name.startswith("Child")],
    )


def test_mirror_copy_parents_a_copied_child_to_its_copied_parent() -> None:
    doc, parent, child = _family()
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("mirror-copy"), axis=0.0) is True
    parents, children = _copies(doc, {parent, child})
    assert len(parents) == len(children) >= 1
    parent_uids = {p.uid for p in parents}
    for kid in children:
        assert kid.parent in parent_uids, "a copied child must hang off a copied parent"
    # one child copy per parent copy, never two under the same parent
    assert sorted(k.parent for k in children) == sorted(parent_uids)
    # the original hierarchy is untouched
    assert doc.by_uid(child).parent == parent


def test_mirror_copy_child_lands_on_the_mirror_image_of_the_original_child() -> None:
    doc, parent, child = _family()
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("mirror-copy"), axis=0.0) is True
    parents, children = _copies(doc, {parent, child})
    assert len(parents) == len(children) == 1
    world = doc.world_matrix(children[0].uid)[:3, 3]
    assert world == pytest.approx([-1.0, 1.0, 0.0], abs=1e-5)
    assert children[0].parent == parents[0].uid


# --- clay-62: run() releases its history gesture when an op raises a bug ----


def test_run_closes_its_history_gesture_when_an_op_raises_something_that_is_not_an_operror() -> (
    None
):
    doc = bd.ClayDoc()
    obj = _add(doc, "Box")
    doc.select([obj.uid])

    def boom(ctx: Any, doc: Any, **_: Any) -> None:
        doc.set_props(obj.uid, name="half-done")
        raise RuntimeError("kernel bug")

    op = dataclasses.replace(clay_ops.get("mirror-x"), run=boom)
    with pytest.raises(RuntimeError):
        clay_ops.run(_Ctx(), doc, op)
    assert doc.history._open_gestures == 0, "a propagating bug must not wedge undo eviction"


# --- clay-66: an outliner click in an element mode keeps selection derived --


def test_an_outliner_click_in_an_element_mode_keeps_selection_derived_from_the_element_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc = bd.ClayDoc()
    a = _add(doc, "A")
    b = _add(doc, "B")
    doc.set_element_mode("face")
    doc.set_element_sel(a.uid, el.ElementSel(faces=[0, 1]))
    assert doc.selection == {a.uid}

    monkeypatch.setattr(
        clay_outliner.imgui, "get_io", lambda: SimpleNamespace(key_shift=False, key_ctrl=False)
    )
    state = SimpleNamespace(outliner_anchor=0)
    clay_outliner._click(state, doc, b)

    # The document invariant: in an element mode `selection` is exactly the
    # uids with a non-empty element selection. Object mode owns it otherwise.
    if doc.element_mode != "object":
        assert doc.selection == set(doc.element_sel)
    assert doc.selection == {b.uid}, "the clicked row is what is selected now"


# --- clay-68: an idle armed live UV gesture pushes nothing ------------------


def _two_islands() -> bm.Mesh:
    positions = [
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [2, 0, 0], [3, 0, 0], [3, 1, 0], [2, 1, 0],
    ]
    uv = [
        [[0.0, 0.0], [0.4, 0.0], [0.4, 0.4], [0.0, 0.4]],
        [[0.6, 0.6], [1.0, 0.6], [1.0, 1.0], [0.6, 1.0]],
    ]
    return bm.from_faces(positions, [[0, 1, 2, 3], [4, 5, 6, 7]], uv=uv)


@pytest.mark.parametrize("kind", ["rotate", "scale"])
def test_an_idle_live_uv_rotate_pushes_no_history_steps(kind: str) -> None:
    doc = bd.ClayDoc()
    obj = _add(doc, "A", _two_islands())
    ids = np.array([0, 1], dtype="i4")
    view_state = clay_uv.UvPaneState(selected_islands=frozenset({0, 1}))
    anchor = (0.05, 0.05)
    before = len(doc.history)
    assert clay_uv.begin_live_transform(doc, view_state, kind, obj.mesh, ids, anchor)

    # Pointer parked on the anchor: the gesture's own zero reading.
    for _ in range(60):
        assert clay_uv.update_live_transform(doc, obj.uid, view_state, anchor) is False
    assert len(doc.history) == before, "an idle gesture must not push a step per frame"

    # Pointer moved once, then parked: one step, not one per frame.
    for _ in range(60):
        clay_uv.update_live_transform(doc, obj.uid, view_state, (0.5, 0.3))
    assert len(doc.history) == before + 1

    # Returning to the zero reading after moving still restores the base.
    assert clay_uv.update_live_transform(doc, obj.uid, view_state, anchor) is True
    clay_uv.commit_live_transform(doc, view_state)
    assert np.allclose(doc.by_uid(obj.uid).mesh.uv, _two_islands().uv)


# --- clay-69: the UV canvas does not walk every face for an unchanged mesh --


class _Draw:
    def __init__(self) -> None:
        self.polys: list[tuple[Any, int]] = []
        self.lines: list[tuple[Any, int, float, int]] = []

    def add_convex_poly_filled(self, points: Any, colour: int) -> None:
        self.polys.append((list(points), colour))

    def add_line(self, p0: Any, p1: Any, colour: int, thickness: float) -> None:
        self.lines.append(([p0, p1], colour, thickness, 0))

    def add_polyline(self, points: Any, colour: int, thickness: float, flags: int) -> None:
        self.lines.append((list(points), colour, thickness, flags))


def test_the_uv_canvas_does_not_walk_every_face_each_frame_for_an_unchanged_mesh(
    ui, monkeypatch: pytest.MonkeyPatch
) -> None:
    from realmspinner.studio.shell import paintview

    mesh = bp.uv_sphere(segments=24, rings=12)
    assert mesh.uv is not None
    n_faces = len(mesh.starts) - 1
    view = paintview.PaintView()
    origin = (10.0, 20.0)
    geo: dict[str, Any] = {}
    state = clay_uv.UvPaneState()
    ids, overlap, _refusal = clay_uv._measurements(state, mesh)

    real_to_screen, real_fill = clay_uv._to_screen, clay_uv._face_fill
    calls = {"screen": 0, "fill": 0}

    def counting_screen(*a: Any, **k: Any) -> Any:
        calls["screen"] += 1
        return real_to_screen(*a, **k)

    def counting_fill(*a: Any, **k: Any) -> Any:
        calls["fill"] += 1
        return real_fill(*a, **k)

    monkeypatch.setattr(clay_uv, "_to_screen", counting_screen)
    monkeypatch.setattr(clay_uv, "_face_fill", counting_fill)

    def frame() -> _Draw:
        draw = _Draw()
        clay_uv._faces(draw, view, origin, mesh, overlap, geo)
        clay_uv._island_outlines(draw, view, origin, mesh, ids, {0}, geo)
        return draw

    first = frame()
    assert len(first.polys) == n_faces
    # The output is the same geometry the per-corner path draws.
    uv = np.asarray(mesh.uv)
    for corner in (0, 3, len(uv) - 1):
        expected = real_to_screen(view, origin, float(uv[corner][0]), float(uv[corner][1]))
        face = int(np.searchsorted(mesh.starts, corner, side="right")) - 1
        got = first.polys[face][0][corner - int(mesh.starts[face])]
        assert got == pytest.approx(expected, abs=1e-6)

    calls.update(screen=0, fill=0)
    for _ in range(3):
        again = frame()
    assert len(again.polys) == n_faces
    assert calls["fill"] == 0, "per-face fills are memoised on the overlap/stretch arrays"
    assert calls["screen"] <= 3 * 2 * 3, (  # 3 probes x 2 drawing calls x 3 frames
        f"{calls['screen']} per-corner conversions for an unchanged mesh and view "
        f"({n_faces} faces) -- only the three view probes per drawing call may remain"
    )

    # A pan changes the answer, so the memo must not serve stale points.
    view.pan = (view.pan[0] + 7.0, view.pan[1] - 3.0)
    panned = frame()
    assert panned.polys[0][0][0] != first.polys[0][0][0]
    expected = real_to_screen(view, origin, float(uv[0][0]), float(uv[0][1]))
    assert panned.polys[0][0][0] == pytest.approx(expected, abs=1e-6)
