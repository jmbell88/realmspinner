"""Regression tests for the 2026-10-03 audit's Medium findings clay-60..clay-69
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
from realmspinner.kernels.mesh import modifiers as mods
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio.modes.clay import ops as clay_ops
from realmspinner.studio.modes.clay.agent import dispatch
from realmspinner.studio.modes.clay.ui.panes import outliner as clay_outliner
from realmspinner.studio.modes.clay.ui.panes import props as clay_props
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


# --- clay-60: ops that renumber the base mesh drop the object's seams --------


def _seamed_doc() -> tuple[bd.ClayDoc, int]:
    doc = bd.ClayDoc()
    obj = _add(doc, "Box", generator="box")
    doc.set_seams(obj.uid, [(0, 1), (2, 3)])
    assert doc.by_uid(obj.uid).seams == ((0, 1), (2, 3))
    doc.select([obj.uid])
    return doc, obj.uid


def test_decimate_retopo_and_smart_unwrap_drop_the_objects_seams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _Ctx()
    replacement = bp.box((2.0, 1.0, 1.0))

    # Decimate: gltfpack's result has no vertex correspondence to the old mesh.
    doc, uid = _seamed_doc()
    monkeypatch.setattr(clay_ops, "_decimate_mesh_from_glb", lambda data, material: replacement)
    clay_ops._decimate_apply(
        ctx,
        doc,
        {
            "items": [
                {
                    "uid": uid,
                    "name": "Box",
                    "stamp": doc.mesh_stamp(uid),
                    "glb_out": b"",
                    "material": 0,
                    "before": 12,
                }
            ],
            "ratio": 0.5,
        },
    )
    assert doc.by_uid(uid).mesh is replacement
    assert doc.by_uid(uid).seams == (), "decimate renumbered the vertices; the seams must go"

    # Retopologize: same.
    doc, uid = _seamed_doc()
    meta = [{"uid": uid, "name": "Box", "stamp": doc.mesh_stamp(uid)}]
    monkeypatch.setattr(clay_ops, "_blender_objects_from_glb", lambda data, m: {uid: replacement})
    clay_ops._retopo_apply(ctx, doc, {"glb_out": b"x", "meta": meta})
    assert doc.by_uid(uid).mesh is replacement
    assert doc.by_uid(uid).seams == ()

    # Smart Unwrap, Blender's rebuilt mesh taken whole (no UV carry possible).
    doc, uid = _seamed_doc()
    meta = [{"uid": uid, "name": "Box", "stamp": doc.mesh_stamp(uid)}]
    monkeypatch.setattr(clay_ops, "_blender_objects_from_glb", lambda data, m: {uid: replacement})
    monkeypatch.setattr(clay_ops, "_carry_uvs", lambda original, unwrapped: None)
    clay_ops._unwrap_apply(ctx, doc, {"glb_out": b"x", "meta": meta})
    assert doc.by_uid(uid).mesh is replacement
    assert doc.by_uid(uid).seams == ()
    # ...and the drop is part of the one undoable step, not a second one.
    assert doc.undo()
    assert doc.by_uid(uid).seams == ((0, 1), (2, 3))


def test_smart_unwrap_keeps_the_seams_when_only_the_uvs_came_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The carried-UV path keeps the original's vertex numbering exactly, so
    the marked seams still name the edges the user marked."""
    doc, uid = _seamed_doc()
    mesh = doc.by_uid(uid).mesh
    meta = [{"uid": uid, "name": "Box", "stamp": doc.mesh_stamp(uid)}]
    carried = dataclasses.replace(mesh, uv=np.zeros_like(mesh.uv))
    monkeypatch.setattr(clay_ops, "_blender_objects_from_glb", lambda data, m: {uid: mesh})
    monkeypatch.setattr(clay_ops, "_carry_uvs", lambda original, unwrapped: carried)
    clay_ops._unwrap_apply(_Ctx(), doc, {"glb_out": b"x", "meta": meta})
    assert doc.by_uid(uid).mesh is carried
    assert doc.by_uid(uid).seams == ((0, 1), (2, 3))


def test_clean_up_drops_the_objects_seams() -> None:
    doc = bd.ClayDoc()
    a, b = bp.box(), bp.box()
    doubled = bm.Mesh(
        positions=np.concatenate([a.positions, b.positions]),
        loops=np.concatenate([a.loops, b.loops + len(a.positions)]),
        starts=np.concatenate([a.starts, a.starts[-1] + b.starts[1:]]),
        material=np.concatenate([a.material, b.material]),
        smooth=np.concatenate([a.smooth, b.smooth]),
    )
    obj = _add(doc, "Doubled", doubled)
    doc.set_seams(obj.uid, [(0, 1)])
    doc.select([obj.uid])
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("clean-mesh")) is True
    assert doc.by_uid(obj.uid).mesh is not doubled
    assert doc.by_uid(obj.uid).seams == ()


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


@pytest.mark.parametrize("op_name", ["array-linear", "array-radial", "mirror-copy"])
def test_array_and_mirror_copy_parent_a_copied_child_to_its_copied_parent(op_name: str) -> None:
    doc, parent, child = _family()
    params = {"array-linear": {"count": 3, "x": 5.0}, "array-radial": {"count": 3},
              "mirror-copy": {"axis": 0.0}}[op_name]
    assert clay_ops.run(_Ctx(), doc, clay_ops.get(op_name), **params) is True
    parents, children = _copies(doc, {parent, child})
    assert len(parents) == len(children) >= 1
    parent_uids = {p.uid for p in parents}
    for kid in children:
        assert kid.parent in parent_uids, "a copied child must hang off a copied parent"
    # one child copy per parent copy, never two under the same parent
    assert sorted(k.parent for k in children) == sorted(parent_uids)
    # the original hierarchy is untouched
    assert doc.by_uid(child).parent == parent


def test_array_linear_copies_keep_the_world_step_through_their_copied_parent() -> None:
    doc, parent, child = _family()
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("array-linear"), count=3, x=5.0) is True
    _, children = _copies(doc, {parent, child})
    worlds = sorted(float(doc.world_matrix(k.uid)[0, 3]) for k in children)
    assert worlds == pytest.approx([6.0, 11.0]), "the child rides its parent copy's step once"


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


# --- clay-63: an array refuses selection x count past its ceiling -----------


@pytest.mark.parametrize("op_name", ["array-linear", "array-radial"])
def test_array_refuses_a_selection_times_count_past_its_ceiling_before_copying(
    op_name: str,
) -> None:
    doc = bd.ClayDoc()
    uids = [_add(doc, f"Prop{i}").uid for i in range(30)]
    doc.select(uids)
    ctx = _Ctx()
    depth = len(doc.history)
    count = 200
    assert len(uids) * (count - 1) > clay_ops.MAX_ARRAY_COPIES

    assert clay_ops.run(ctx, doc, clay_ops.get(op_name), count=count) is False

    assert len(doc.objects) == 30, "nothing may be copied before the refusal"
    assert len(doc.history) == depth
    assert ctx.toasts.errors or ctx.toasts.info, "the refusal is said, not silent"
    # a selection x count inside the ceiling still runs
    doc.select(uids[:2])
    assert clay_ops.run(ctx, doc, clay_ops.get(op_name), count=5) is True
    assert len(doc.objects) == 30 + 2 * 4


# --- clay-64: Symmetrize's side choice says which half it deletes -----------


def test_symmetrize_keep_side_plus_keeps_the_positive_half() -> None:
    """The pre-fix label said "keep side" while the kernel deletes the chosen
    half: the choice and the words must agree. The label now names the half
    that goes, so "+" must remove the positive half and keep the negative."""
    param = next(p for p in clay_ops.get("symmetrize").params if p.name == "direction")
    assert "keep" not in param.label
    assert "delete" in param.label
    assert param.choices == ("-", "+")

    doc = bd.ClayDoc()
    box = bp.box()
    # x spans -0.5 .. 1.5: the positive half is the larger one.
    wide = dataclasses.replace(
        box,
        positions=np.asarray(box.positions, dtype="f4") * np.array([2.0, 1, 1], dtype="f4")
        + np.array([0.5, 0, 0], dtype="f4"),
    )
    obj = _add(doc, "Box", wide)
    lo, hi = float(wide.positions[:, 0].min()), float(wide.positions[:, 0].max())
    assert lo < 0 < hi and hi > -lo, "fixture: the positive half is the larger one"
    doc.select([obj.uid])

    assert clay_ops.run(_Ctx(), doc, clay_ops.get("symmetrize"), axis=0.0, direction=1.0)
    xs = doc.by_uid(obj.uid).mesh.positions[:, 0]
    assert float(xs.max()) == pytest.approx(-lo, abs=1e-5), "'+' deleted the positive half"
    assert float(xs.min()) == pytest.approx(lo, abs=1e-5)


# --- clay-65: Select Boundary seeds from every visible object ---------------


def test_select_boundary_selects_the_open_edges_with_nothing_selected() -> None:
    doc = bd.ClayDoc()
    plane = _add(doc, "Plane", bp.plane())
    hidden = _add(doc, "Hidden", bp.plane(), visible=False)
    doc.set_element_mode("edge")
    assert not doc.element_sel
    op = clay_ops.get("select-boundary")
    assert op.enabled(doc)

    assert clay_ops.run(_Ctx(), doc, op) is True

    assert plane.uid in doc.element_sel
    assert len(doc.element_sel[plane.uid].edges) == 4
    assert hidden.uid not in doc.element_sel, "a hidden object is never seeded"
    assert doc.selection == {plane.uid}


def test_the_agent_text_does_not_call_select_linked_more_and_less_seedless() -> None:
    text = dispatch.instructions()
    assert "select-linked, select-more, select-less, select-boundary" not in text
    assert "select-linked, select-more and select-less" in text
    described = next(t for t in dispatch.tools() if t.name == "clay_element_mode").description
    assert "select-all, select-boundary and the rest" not in described


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


# --- clay-67: transform and modifier fields grey on the lock that refuses ---


class _AnyStub:
    def __getattr__(self, name: str):
        return lambda *a, **k: None


def _recording_stubs(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, bool]]:
    """Replace props' imgui/controls/widgets with recorders. -> events, each
    ``(what, whether a begin_disabled(True) was open at that moment)``."""
    events: list[tuple[str, bool]] = []
    stack: list[bool] = []

    class _Imgui(_AnyStub):
        def begin_disabled(self, flag: bool = True) -> None:
            stack.append(bool(flag))

        def end_disabled(self) -> None:
            stack.pop()

    def at(name: str):
        def record(*a: Any, **k: Any) -> Any:
            events.append((name, any(stack)))
            return None

        return record

    class _Controls(_AnyStub):
        def input_vec(self, label: str, values: list[float], labels: Any) -> Any:
            events.append((label, any(stack)))
            return False, values

        def input_float(self, label: str, value: float, step: float = 0.0) -> Any:
            events.append((label, any(stack)))
            return False, value

        def input_int(self, label: str, value: int, step: int = 1) -> Any:
            events.append((label, any(stack)))
            return False, value

        def checkbox(self, label: str, value: bool) -> Any:
            return False, value

        def small_button(self, *a: Any, **k: Any) -> bool:
            return False

    class _Widgets(_AnyStub):
        def combo(self, label: str, current: str, options: Any) -> str:
            events.append((label, any(stack)))
            return current

    monkeypatch.setattr(clay_props, "imgui", _Imgui())
    monkeypatch.setattr(clay_props, "controls", _Controls())
    monkeypatch.setattr(clay_props, "widgets", _Widgets())
    monkeypatch.setattr(clay_props, "_dimensions", at("dimensions"))
    return events


def test_transform_and_modifier_fields_are_greyed_for_a_locked_ancestor_and_a_locked_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _recording_stubs(monkeypatch)

    # A child of a locked group: set_transform refuses it, so the fields grey.
    doc = bd.ClayDoc()
    group = _add(doc, "Group")
    child = _add(doc, "Child")
    doc.set_parent(child.uid, group.uid, keep_world=False)
    doc.set_props(group.uid, locked=True)
    assert doc.by_uid(child.uid).locked is False
    clay_props._transform(doc, doc.by_uid(child.uid))
    drawn = [greyed for label, greyed in events if label.endswith("##bt")]
    assert drawn == [True], "a locked ancestor must grey the position field"

    # A free, unlocked object stays editable (the predicate is not just "any").
    events.clear()
    free = _add(doc, "Free")
    clay_props._transform(doc, doc.by_uid(free.uid))
    assert [greyed for label, greyed in events if label.endswith("##bt")] == [False]

    # A locked object's modifier parameters grey too.
    events.clear()
    solo = _add(doc, "Solo")
    doc.set_modifiers(solo.uid, (mods.make("array", {"count": 3.0}, id=1),))
    doc.set_props(solo.uid, locked=True)
    obj = doc.by_uid(solo.uid)
    clay_props._modifier_row(_Ctx(), doc, obj, obj.modifiers[0], 0, 1, None)
    params = [greyed for label, greyed in events if label.startswith("##")]
    assert params and all(params), "every modifier parameter widget must be greyed when locked"


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
        self.lines: list[tuple[Any, Any, int, float]] = []

    def add_convex_poly_filled(self, points: Any, colour: int) -> None:
        self.polys.append((list(points), colour))

    def add_line(self, p0: Any, p1: Any, colour: int, thickness: float) -> None:
        self.lines.append((p0, p1, colour, thickness))


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
    ids, overlap, stretch, _refusal, seam_cuts = clay_uv._measurements(state, mesh)

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
        clay_uv._faces(draw, view, origin, mesh, ids, overlap, stretch, geo)
        clay_uv._edges(draw, view, origin, mesh, (), seam_cuts, geo)
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
    assert calls["screen"] <= 3 * 3 * 3, (  # 3 probes x 3 drawing calls x 3 frames
        f"{calls['screen']} per-corner conversions for an unchanged mesh and view "
        f"({n_faces} faces) -- only the three view probes per drawing call may remain"
    )

    # A pan changes the answer, so the memo must not serve stale points.
    view.pan = (view.pan[0] + 7.0, view.pan[1] - 3.0)
    panned = frame()
    assert panned.polys[0][0][0] != first.polys[0][0][0]
    expected = real_to_screen(view, origin, float(uv[0][0]), float(uv[0][1]))
    assert panned.polys[0][0][0] == pytest.approx(expected, abs=1e-6)
