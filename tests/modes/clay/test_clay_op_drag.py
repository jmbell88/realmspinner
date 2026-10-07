"""Interactive op drags and the E-then-drag extrude (Blender-Lite plan, Phase C).

The claims, all about the document rather than the picture:

* a drag **never touches the document** until it commits -- ``rev`` is flat
  during it and Esc leaves the mesh, the history and the selection as they were;
* the commit is **one** ``clay_ops.run`` -- equal to running the op directly at
  that value, one undo step, and the op becomes the recent op (the adjust card);
* a typed value is **exact** (``0.1`` then Enter is ``0.1``, not a pixel count);
* ``E`` is extrude **plus** a drag along the face normal, as one undo step, and
  Esc after it undoes the extrude too;
* a slow preview is **throttled**, and the commit is never throttled.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay import ops as clay_ops
from realmspinner.studio.modes.clay.state import ClayState
from realmspinner.studio.modes.clay.ui import _view_opdrag
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


@pytest.fixture
def ctx():
    return _Ctx()


@pytest.fixture
def view(gl, ctx):
    v = clay_view.ClayView(gl, ctx)
    ctx.clay_view = v
    yield v
    v.release()


def _doc(mode: str = "face", faces: tuple[int, ...] = (0,)) -> tuple[bd.ClayDoc, int]:
    doc = bd.ClayDoc()
    uid = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box())).uid
    doc.set_element_mode("face")
    doc.set_element_sel(uid, el.ElementSel(faces=list(faces)))
    if mode != "face":
        doc.set_element_mode(mode)
    return doc, uid


def _ready(view, doc: bd.ClayDoc) -> None:
    view.frame_selection(doc)
    view.draw(doc, RECT, 0.0)
    view._last_mouse = (64.0, 48.0)


def _positions(doc: bd.ClayDoc, uid: int) -> np.ndarray:
    return np.array(doc.by_uid(uid).mesh.positions)


def _type(view, doc: bd.ClayDoc, text: str) -> None:
    for ch in text:
        assert view.drag_key(doc, ch)


# --- the op metadata ----------------------------------------------------------


def test_the_inset_drag_op_names_a_real_param_and_the_kernel_its_run_wraps() -> None:
    names = {op.name for op in clay_ops.OPS if op.drag is not None}
    assert names == {"inset"}
    for op in clay_ops.OPS:
        if op.drag is None:
            continue
        assert op.drag.param in {p.name for p in op.params}
        assert op.drag.kernel == op.run.kernel, "preview and commit must be one function"
        assert op.key, "a drag op is started by its key"


# --- the drag never touches the document --------------------------------------


def test_an_inset_drag_leaves_the_document_alone_until_it_commits(view, ctx) -> None:
    doc, uid = _doc()
    _ready(view, doc)
    rev, mesh, steps = doc.rev, doc.by_uid(uid).mesh, len(doc.history)
    rebuilds = view.rebuilds
    entry_before = view._cache[uid].gpu

    assert view.begin_op_drag(doc, clay_ops.get("inset"))
    assert view.dragging and view._grab == "opdrag"
    view._motion(doc, (100.0, 48.0))
    view.draw(doc, RECT, 0.0)

    assert doc.rev == rev and doc.by_uid(uid).mesh is mesh and len(doc.history) == steps
    assert view.rebuilds == rebuilds, "the preview is a swap, not a document rebuild"
    assert view._cache[uid].gpu is not entry_before, "the picture did change"
    assert view._op_drag.value > 0.0


def test_escape_leaves_the_document_byte_identical(view, ctx) -> None:
    doc, uid = _doc()
    _ready(view, doc)
    rev, positions = doc.rev, _positions(doc, uid)
    sel = doc.element_sel[uid]
    steps = len(doc.history)

    view.begin_op_drag(doc, clay_ops.get("inset"))
    view._motion(doc, (110.0, 60.0))
    assert view.cancel_drag(doc)

    assert not view.dragging and view._op_drag is None
    assert doc.rev == rev
    assert np.array_equal(_positions(doc, uid), positions)
    assert doc.element_sel[uid] is sel and len(doc.history) == steps
    assert doc.recent_op is None
    assert uid not in view._cache, "the previewed entry is evicted, so sync rebuilds from truth"
    view.draw(doc, RECT, 0.0)
    assert view._cache[uid].mesh is doc.by_uid(uid).mesh


# --- the commit ---------------------------------------------------------------


def test_commit_equals_running_the_op_at_that_value_and_is_one_undo_step(view, ctx) -> None:
    doc, uid = _doc()
    _ready(view, doc)
    steps = len(doc.history)
    view.begin_op_drag(doc, clay_ops.get("inset"))
    view._motion(doc, (100.0, 48.0))
    value = view._op_drag.value
    assert view._release_drag(doc)

    reference, ref_uid = _doc()
    clay_ops.run(_Ctx(), reference, clay_ops.get("inset"), thickness=value)
    assert np.array_equal(_positions(doc, uid), _positions(reference, ref_uid))
    assert len(doc.history) == steps + 1
    assert doc.recent_op is not None and doc.recent_op.op_name == "inset"
    assert doc.recent_op.params["thickness"] == value
    assert doc.undo()
    assert np.array_equal(_positions(doc, uid), np.array(bp.box().positions))


def test_a_typed_value_then_enter_is_exactly_that_value(view, ctx) -> None:
    doc, uid = _doc()
    _ready(view, doc)
    view.begin_op_drag(doc, clay_ops.get("inset"))
    view._motion(doc, (120.0, 48.0))  # the pointer says something else entirely
    _type(view, doc, "0.1")
    assert "= 0.1" in view.drag_hud
    assert view._release_drag(doc)

    assert doc.recent_op.params["thickness"] == 0.1
    reference, ref_uid = _doc()
    clay_ops.run(_Ctx(), reference, clay_ops.get("inset"), thickness=0.1)
    assert np.array_equal(_positions(doc, uid), _positions(reference, ref_uid))


def test_the_hud_names_the_op_the_parameter_and_the_value(view, ctx) -> None:
    doc, _ = _doc()
    _ready(view, doc)
    view.begin_op_drag(doc, clay_ops.get("inset"))
    _type(view, doc, "0.05")
    assert view.drag_hud.startswith("Inset thickness 0.050 m")


def test_a_refusal_at_the_value_keeps_the_last_picture_and_says_why(view, ctx, monkeypatch) -> None:
    from realmspinner.kernels.mesh.elements import OpError

    doc, uid = _doc()
    _ready(view, doc)
    view.begin_op_drag(doc, clay_ops.get("inset"))

    def refuse(mesh, sel, **params):
        raise OpError("Too thick.")

    rev, steps = doc.rev, len(doc.history)
    view._op_drag.kernel = refuse
    view._motion(doc, (100.0, 48.0))
    assert "Too thick." in view.drag_hud
    assert doc.rev == rev and len(doc.history) == steps
    view.cancel_drag(doc)


def test_an_op_drag_will_not_start_without_a_selection_or_over_another_grab(view, ctx) -> None:
    doc, uid = _doc()
    _ready(view, doc)
    doc.clear_element_sel()
    assert not view.begin_op_drag(doc, clay_ops.get("inset"))

    doc.set_element_sel(uid, el.ElementSel(faces=[0]))
    assert view.begin_keyboard_drag(doc, "move")
    assert not view.begin_op_drag(doc, clay_ops.get("inset"))
    view.cancel_drag(doc)


# --- the keys -----------------------------------------------------------------


def test_the_ops_key_starts_the_drag_and_a_menu_click_keeps_the_dialog(view, ctx) -> None:
    doc, uid = _doc()
    _ready(view, doc)
    inset = clay_ops.get("inset")

    assert clay_mode._fire_op(ctx, doc, inset, interactive=True) is True
    assert view._grab == "opdrag"
    view.cancel_drag(doc)

    state = ctx.state.clay
    assert clay_mode.fire_op(ctx, doc, inset) is True, "the menu's door"
    assert view._grab is None and state.pending_op == "inset", "a dialog, not a drag"


# --- E, then drag --------------------------------------------------------------


def test_e_extrudes_and_starts_a_drag_locked_to_the_face_normal(view, ctx) -> None:
    doc, uid = _doc()
    _ready(view, doc)
    faces = bm.face_count(doc.by_uid(uid).mesh)

    assert clay_mode._fire_op(ctx, doc, clay_ops.get("extrude"), interactive=True)
    assert view._grab == "keydrag" and view.drag_input.axis == "normal"
    assert bm.face_count(doc.by_uid(uid).mesh) > faces
    view.cancel_drag(doc)


def test_e_then_a_drag_is_one_undo_step(view, ctx) -> None:
    doc, uid = _doc()
    _ready(view, doc)
    steps = len(doc.history)

    clay_mode._fire_op(ctx, doc, clay_ops.get("extrude"), interactive=True)
    view._motion(doc, (90.0, 20.0))
    view._release_drag(doc)

    assert len(doc.history) == steps + 1
    assert doc.history.top.label == "Extrude"
    assert doc.undo()
    assert np.array_equal(_positions(doc, uid), np.array(bp.box().positions))


def test_the_extrude_drag_moves_only_along_the_normal(view, ctx) -> None:
    doc, uid = _doc()
    _ready(view, doc)
    normal = np.asarray(bm.face_normals(doc.by_uid(uid).mesh)[0], dtype="f8")
    normal /= np.linalg.norm(normal)

    clay_mode._fire_op(ctx, doc, clay_ops.get("extrude"), interactive=True)
    extruded = _positions(doc, uid).copy()
    view._motion(doc, (100.0, 20.0))
    view._release_drag(doc)
    after = _positions(doc, uid)

    shift = after - extruded
    moving = np.linalg.norm(shift, axis=1) > 1e-9
    assert moving.any(), "the pointer did move the extrusion"
    along = shift[moving] @ normal
    assert np.allclose(shift[moving], np.outer(along, normal), atol=1e-6)


def test_escape_after_e_undoes_the_extrude_too(view, ctx) -> None:
    doc, uid = _doc()
    _ready(view, doc)
    steps, selection = len(doc.history), doc.element_sel[uid]

    clay_mode._fire_op(ctx, doc, clay_ops.get("extrude"), interactive=True)
    view._motion(doc, (100.0, 20.0))
    assert view.cancel_drag(doc)

    assert np.array_equal(_positions(doc, uid), np.array(bp.box().positions))
    assert bm.face_count(doc.by_uid(uid).mesh) == 6
    assert len(doc.history) == steps
    assert not doc.history.can_redo, "an Esc must not leave an extrude to redo"
    assert doc.element_sel[uid].same_as(selection), "and the selection is the one it began with"
    assert view._extrude_gesture is None


def test_an_extrude_with_no_viewport_is_the_plain_extrude(ctx) -> None:
    doc, uid = _doc()
    ctx.clay_view = None
    assert clay_mode._fire_op(ctx, doc, clay_ops.get("extrude"), interactive=True)
    assert bm.face_count(doc.by_uid(uid).mesh) > 6
    assert doc.history._open_gestures == 0, "no gesture left open"


# --- the throttle ---------------------------------------------------------------


def test_a_slow_preview_is_throttled_and_the_commit_never_is(view, ctx, monkeypatch) -> None:
    doc, uid = _doc()
    _ready(view, doc)
    clock = [0.0]
    monkeypatch.setattr(_view_opdrag, "_now", lambda: clock[0])
    view.begin_op_drag(doc, clay_ops.get("inset"))
    od = view._op_drag
    calls: list[float] = []
    real = od.kernel

    def slow(mesh, sel, **params):
        calls.append(params["thickness"])
        clock[0] += 0.2  # a 200 ms preview
        return real(mesh, sel, **params)

    od.kernel = slow

    view._motion(doc, (70.0, 48.0))  # first one always runs
    assert len(calls) == 1
    view._motion(doc, (80.0, 48.0))  # 0 ms after a 200 ms preview: skipped...
    # (the fake clock only moves inside the kernel, so "now" is the end of the first)
    assert len(calls) == 1
    clock[0] += _view_opdrag.SLOW_INTERVAL_S + 0.01
    view._motion(doc, (90.0, 48.0))  # ...and allowed again once the interval has passed
    assert len(calls) == 2

    view._motion(doc, (100.0, 48.0))
    assert len(calls) == 2
    od.kernel = real
    value = view._op_drag_value(doc)
    view._release_drag(doc)
    assert doc.recent_op.params["thickness"] == value, "the commit runs at the pointer's value"


def test_a_fast_preview_is_not_throttled(view, ctx, monkeypatch) -> None:
    doc, uid = _doc()
    _ready(view, doc)
    monkeypatch.setattr(_view_opdrag, "_now", lambda: 0.0)
    view.begin_op_drag(doc, clay_ops.get("inset"))
    od = view._op_drag
    calls: list[float] = []
    real = od.kernel
    od.kernel = lambda mesh, sel, **p: (calls.append(1), real(mesh, sel, **p))[1]
    for x in (70.0, 80.0, 90.0, 100.0):
        view._motion(doc, (x, 48.0))
    assert len(calls) == 4
    view.cancel_drag(doc)


@pytest.mark.perf
def test_a_big_mesh_op_drag_never_exceeds_the_throttled_cadence(view, ctx) -> None:
    """Wall-clock, so the perf lane: on a mesh whose preview is slow, the number of
    kernel runs over a drag is bounded by elapsed time / the slow interval, not by
    the number of mouse moves."""
    doc = bd.ClayDoc()
    uid = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="Ball", mesh=bp.uv_sphere(segments=256, rings=192))
    ).uid
    doc.set_element_mode("face")
    doc.set_element_sel(uid, el.ElementSel(faces=list(range(0, 2000))))
    _ready(view, doc)
    view.begin_op_drag(doc, clay_ops.get("inset"))
    od = view._op_drag
    calls = []
    real = od.kernel

    def counted(mesh, sel, **params):
        calls.append(time.perf_counter())
        return real(mesh, sel, **params)

    od.kernel = counted
    started = time.perf_counter()
    for i in range(60):
        view._motion(doc, (64.0 + i, 48.0))
    elapsed = time.perf_counter() - started
    view.cancel_drag(doc)
    if od.cost <= _view_opdrag.SLOW_PREVIEW_S:
        pytest.skip("this machine previews fast enough that no throttling applies")
    assert len(calls) <= 2 + elapsed / _view_opdrag.SLOW_INTERVAL_S
