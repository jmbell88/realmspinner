"""Clay's recent-op record: adjusted in place, repeated on demand.

**Reverses the 2026-09-07 audit's clay-10, deliberately.** ``ClayState.last_op``
was removed because it was written by every ``clay_ops.run`` and read by no
pane. The record is back (``ClayDoc.recent_op``, ``recent_op.RecentOp``) because
it now has two readers -- the viewport's adjust card, which re-runs the op at new
values, and the ``repeat-last`` op -- and the claims below are about those
readers rather than about the bookkeeping:

* adjusting is **one undo step** equal to running the op at the final value;
* the card is live **only while the model is as the op left it**;
* a refusal at the new value **keeps the previous result**;
* repeat runs the same op at the same values on **the current selection**.

``last_op`` itself stays gone: ``ClayState`` carries no such field.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh.elements import OpError
from realmspinner.studio.modes.clay import ops as clay_ops
from realmspinner.studio.modes.clay import recent_op
from realmspinner.studio.modes.clay import state as clay_state


class _Ctx:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def toast(self, message: str, level: str = "info") -> None:
        if level == "error":
            self.errors.append(message)


def _box_doc() -> tuple[bd.ClayDoc, int]:
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
    return doc, obj.uid


def _edges_of_face(doc: bd.ClayDoc, uid: int, face: int = 0) -> None:
    doc.set_element_mode("face")
    doc.set_element_sel(uid, el.ElementSel(faces=[face]))
    doc.set_element_mode("edge")


def _faces(doc: bd.ClayDoc, uid: int, *faces: int) -> None:
    doc.set_element_mode("face")
    doc.set_element_sel(uid, el.ElementSel(faces=list(faces)))


def _positions(doc: bd.ClayDoc, uid: int) -> np.ndarray:
    return np.array(doc.by_uid(uid).mesh.positions)


# --- the record -------------------------------------------------------------


def test_last_op_stays_gone_from_the_app_state() -> None:
    assert not hasattr(clay_state, "LastOp")
    assert "last_op" not in {f.name for f in dataclasses.fields(clay_state.ClayState)}


def test_a_parameterised_element_op_is_recorded_with_what_it_ran_with() -> None:
    doc, uid = _box_doc()
    _edges_of_face(doc, uid)
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("bevel"), width=0.1) is True
    recent = doc.recent_op
    assert recent is not None and recent.op_name == "bevel"
    assert recent.params == {"width": 0.1}
    assert recent.element_mode == "edge"
    assert recent.live(doc)


def test_a_bare_action_and_an_object_level_op_are_not_recorded() -> None:
    doc, uid = _box_doc()
    _faces(doc, uid, 0)
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("extrude")) is True
    assert doc.recent_op is None, "extrude has no numbers to adjust"
    doc.set_element_mode("object")
    doc.select([uid])
    clay_ops.run(_Ctx(), doc, clay_ops.get("array-linear"), count=2)
    assert doc.recent_op is None, "an object-level op has a different selection model"


def test_a_refused_op_leaves_no_record() -> None:
    doc, uid = _box_doc()
    _edges_of_face(doc, uid)  # four edges: loop-cut wants exactly one
    ctx = _Ctx()
    assert clay_ops.run(ctx, doc, clay_ops.get("loop-cut")) is False
    assert doc.recent_op is None


# --- adjust -----------------------------------------------------------------


def test_adjusting_bevel_width_twice_is_one_undo_step_equal_to_running_it_directly() -> None:
    doc, uid = _box_doc()
    _edges_of_face(doc, uid)
    base = len(doc.history)
    ctx = _Ctx()
    clay_ops.run(ctx, doc, clay_ops.get("bevel"), width=0.1)
    assert recent_op.adjust(ctx, doc, width=0.2).ok
    assert recent_op.adjust(ctx, doc, width=0.05).ok
    assert len(doc.history) == base + 1, "adjusting must not stack steps"

    direct, direct_uid = _box_doc()
    _edges_of_face(direct, direct_uid)
    clay_ops.run(_Ctx(), direct, clay_ops.get("bevel"), width=0.05)
    assert np.array_equal(_positions(doc, uid), _positions(direct, direct_uid))
    assert doc.recent_op.params == {"width": 0.05}

    assert doc.undo()
    assert np.array_equal(_positions(doc, uid), np.array(bp.box().positions)), (
        "one Ctrl+Z takes the whole adjusted operation back"
    )


def test_the_card_hides_after_any_other_edit_an_undo_or_a_selection_change() -> None:
    def fresh():
        doc, uid = _box_doc()
        _edges_of_face(doc, uid)
        clay_ops.run(_Ctx(), doc, clay_ops.get("bevel"), width=0.1)
        assert doc.recent_op.live(doc)
        return doc, uid

    doc, uid = fresh()
    doc.set_props(uid, name="Renamed")
    assert not doc.recent_op.live(doc), "a later edit"

    doc, uid = fresh()
    doc.undo()
    assert not doc.recent_op.live(doc), "an undo"

    doc, uid = fresh()
    doc.clear_element_sel()
    assert not doc.recent_op.live(doc), "a selection change"

    doc, uid = fresh()
    doc.set_element_mode("object")
    assert not doc.recent_op.live(doc), "a mode change"


def test_adjust_on_a_stale_record_refuses_and_touches_nothing() -> None:
    doc, uid = _box_doc()
    _edges_of_face(doc, uid)
    clay_ops.run(_Ctx(), doc, clay_ops.get("bevel"), width=0.1)
    doc.set_props(uid, name="Renamed")
    before = _positions(doc, uid)
    steps = len(doc.history)
    result = recent_op.adjust(_Ctx(), doc, width=0.3)
    assert not result.ok and result.message
    assert np.array_equal(_positions(doc, uid), before) and len(doc.history) == steps


def test_a_refusal_at_the_new_value_keeps_the_previous_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The op is swapped for a copy that refuses past 0.5, so the claim is about
    the adjust path and not about one kernel's limits."""
    index = next(i for i, op in enumerate(clay_ops.OPS) if op.name == "inset")
    real = clay_ops.OPS[index]

    def picky(ctx: Any, doc: Any, **params: Any) -> Any:
        if params["thickness"] > 0.5:
            raise OpError("Too thick for this face.")
        return real.run(ctx, doc, **params)

    patched = list(clay_ops.OPS)
    patched[index] = dataclasses.replace(real, run=picky)
    monkeypatch.setattr(clay_ops, "OPS", patched)

    doc, uid = _box_doc()
    _faces(doc, uid, 0)
    ctx = _Ctx()
    clay_ops.run(ctx, doc, clay_ops.get("inset"), thickness=0.1)
    kept = _positions(doc, uid)
    recent = doc.recent_op
    steps = len(doc.history)

    result = recent_op.adjust(ctx, doc, thickness=0.9)

    assert not result.ok and result.message == "Too thick for this face."
    assert np.array_equal(_positions(doc, uid), kept)
    assert len(doc.history) == steps
    assert doc.recent_op is recent and recent.live(doc), "the card stays up, still live"
    assert ctx.errors == [], "the reason goes to the card, not to a toast"
    # And the document still undoes cleanly, as one step.
    assert doc.undo()
    assert np.array_equal(_positions(doc, uid), np.array(bp.box().positions))


# --- repeat last ------------------------------------------------------------


def test_repeat_last_refuses_with_nothing_to_repeat() -> None:
    doc, uid = _box_doc()
    _faces(doc, uid, 0)
    repeat = clay_ops.get("repeat-last")
    assert not repeat.enabled(doc)
    assert clay_ops.reason_for(repeat, doc) == "Nothing to repeat."
    assert clay_ops.run(_Ctx(), doc, repeat) is False


def test_repeat_last_runs_the_same_op_at_the_same_values_on_the_new_selection() -> None:
    doc, uid = _box_doc()
    _faces(doc, uid, 0)
    clay_ops.run(_Ctx(), doc, clay_ops.get("inset"), thickness=0.2)
    _faces(doc, uid, 2)  # a face the first inset did not touch
    steps = len(doc.history)
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("repeat-last")) is True
    assert len(doc.history) == steps + 1

    reference, ref_uid = _box_doc()
    _faces(reference, ref_uid, 0)
    clay_ops.run(_Ctx(), reference, clay_ops.get("inset"), thickness=0.2)
    _faces(reference, ref_uid, 2)
    clay_ops.run(_Ctx(), reference, clay_ops.get("inset"), thickness=0.2)
    assert np.array_equal(_positions(doc, uid), _positions(reference, ref_uid))
    assert doc.recent_op.params["thickness"] == 0.2, "the repeat is itself repeatable"


def test_repeat_last_names_the_mode_it_does_not_work_in() -> None:
    doc, uid = _box_doc()
    _edges_of_face(doc, uid)
    clay_ops.run(_Ctx(), doc, clay_ops.get("bevel"), width=0.1)
    doc.set_element_mode("face")
    doc.set_element_sel(uid, el.ElementSel(faces=[1]))
    repeat = clay_ops.get("repeat-last")
    assert not repeat.enabled(doc)
    assert "edge mode" in clay_ops.reason_for(repeat, doc)


def test_each_document_repeats_only_its_own_last_op() -> None:
    mine, my_uid = _box_doc()
    theirs, their_uid = _box_doc()
    _faces(mine, my_uid, 0)
    clay_ops.run(_Ctx(), mine, clay_ops.get("inset"), thickness=0.3)
    _faces(theirs, their_uid, 0)
    assert theirs.recent_op is None
    assert not clay_ops.get("repeat-last").enabled(theirs)


def test_shift_r_is_the_repeat_bindings_and_clashes_with_nothing() -> None:
    repeat = clay_ops.get("repeat-last")
    assert repeat.key == "Shift+R"
    for mode in clay_ops.ELEMENT_MODES:
        bound = [op.name for op in clay_ops.menu(mode) if op.key == "Shift+R"]
        assert bound == ["repeat-last"]
        assert clay_ops.by_key(mode, "Shift+R") is repeat
    assert "object" not in repeat.modes


# --- the card ----------------------------------------------------------------


class _AppCtx(_Ctx):
    def __init__(self, state: clay_state.ClayState) -> None:
        super().__init__()
        self.state = type("S", (), {"clay": state})()
        self.settings = None


@pytest.fixture
def ui(monkeypatch):
    from _ui_context import imgui_context

    with imgui_context(monkeypatch) as imgui:
        yield imgui


def _draw_card(ui: Any, ctx: Any) -> bool:
    from realmspinner.studio.modes.clay.ui.panes import adjust

    ui.new_frame()
    ui.begin("##host")
    try:
        return adjust.draw(ctx, (0.0, 0.0, 800.0, 600.0))
    finally:
        ui.end()
        ui.end_frame()


def test_the_card_draws_only_while_live_and_a_field_change_adjusts_the_op(
    ui, monkeypatch: pytest.MonkeyPatch
) -> None:
    from realmspinner.studio.modes.clay import mode as clay_mode
    from realmspinner.studio.modes.clay.ui.panes import adjust

    doc, uid = _box_doc()
    state = clay_state.ClayState()
    ctx = _AppCtx(state)
    clay_mode.adopt(ctx, doc, title="Scene")
    drawn: list[str] = []

    def fake_widget(op_name: str, param: Any, value: float) -> tuple[bool, float]:
        drawn.append(param.name)
        return (True, 0.3) if param.name == "thickness" else (False, value)

    monkeypatch.setattr(adjust, "param_widget", fake_widget)

    _faces(doc, uid, 0)
    clay_ops.run(ctx, doc, clay_ops.get("inset"), thickness=0.1)
    steps = len(doc.history)

    _draw_card(ui, ctx)
    assert "thickness" in drawn, "the card shows the op's own parameters"
    assert doc.recent_op.params["thickness"] == 0.3
    assert len(doc.history) == steps, "an adjust replaces the step, it does not add one"

    drawn.clear()
    doc.set_props(uid, name="Renamed")
    _draw_card(ui, ctx)
    assert drawn == [], "a stale record draws nothing"
