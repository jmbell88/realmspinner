"""Mark as Bone, Unmark Bone and Unbind Skin: the rows, the doors and the cards that draw them.

Tranche 2 of the 2026-10-13 Clay rig plan. Every claim fails on the tree before the change:
``clay_ops.get("mark-bone")`` raises ``KeyError``, the Inspector has no Bone & rig card and
a weighted mesh can be dragged.
"""

from __future__ import annotations

import numpy as np
import pytest
from _ui_context import imgui_context

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import rig
from realmspinner.studio import icons, widgets
from realmspinner.studio.modes.clay import doc_io, menutree
from realmspinner.studio.modes.clay import ops as clay_ops
from realmspinner.studio.modes.clay.ui.panes import props

from ._rigkit import rigged_doc, skin_for, weighted
from .test_clay_palette import _clay
from .test_clay_palette import _ctx as palette_ctx
from .test_inspector import _document, _rig

NAMES = ("mark-bone", "unmark-bone", "unbind-skin")


@pytest.fixture
def imgui_ui(monkeypatch):
    with imgui_context(monkeypatch) as imgui:
        yield imgui


class _Ctx:
    def __init__(self) -> None:
        self.toasts: list[tuple[str, str]] = []

    def toast(self, message: str, level: str = "info") -> None:
        self.toasts.append((message, level))


def _boxes(n: int, **kw) -> tuple[bd.ClayDoc, list[int]]:
    doc = bd.ClayDoc()
    uids = [
        doc.add_object(bd.Obj(uid=bd.new_uid(), name=f"B{i}", mesh=bp.box(), **kw)).uid
        for i in range(n)
    ]
    doc.select(uids)
    return doc, uids


# -- the rows -------------------------------------------------------------------------------


def test_the_three_rows_are_object_mode_ops_in_a_rig_group_of_the_object_menu_only() -> None:
    assert menutree.GROUPS["rig"].ops == NAMES
    assert all(clay_ops.get(name).modes == ("object",) for name in NAMES)
    assert "rig" in [e.name for e in menutree.BARS["object"][1].entries]
    for mode in ("vertex", "edge", "face"):
        assert not set(NAMES) & {op.name for run in menutree.flat(mode) for op in run}
    assert [clay_ops.get(n).label for n in NAMES] == ["Mark as Bone", "Unmark Bone", "Unbind Skin"]


def test_each_row_is_enabled_only_when_it_could_change_something_and_says_why_otherwise() -> None:
    mark, unmark, unbind = (clay_ops.get(n) for n in NAMES)
    doc, uids = _boxes(2)
    assert mark.enabled(doc) and not unmark.enabled(doc) and not unbind.enabled(doc)
    assert "None of the selection is a bone" in clay_ops.reason_for(unmark, doc)
    assert "weighted" in clay_ops.reason_for(unbind, doc)

    doc.set_bone(uids, True)
    assert not mark.enabled(doc) and "already" in clay_ops.reason_for(mark, doc)
    assert unmark.enabled(doc)

    doc.select([])
    assert all(
        not op.enabled(doc) and clay_ops.reason_for(op, doc) for op in (mark, unmark, unbind)
    )

    rigged, bone, body = rigged_doc()
    rigged.select([body])
    assert unbind.enabled(rigged) and clay_ops.reason_for(unbind, rigged) == ""


# -- running them ---------------------------------------------------------------------------


def test_mark_bone_marks_every_selected_object_as_one_undo_step_and_unmark_reverses_it() -> None:
    doc, uids = _boxes(3)
    steps, ctx = len(doc.history), _Ctx()
    assert clay_ops.run(ctx, doc, clay_ops.get("mark-bone")) is True
    assert all(doc.by_uid(u).bone for u in uids) and ctx.toasts == []
    assert len(doc.history) == steps + 1
    assert clay_ops.run(ctx, doc, clay_ops.get("unmark-bone")) is True
    assert not any(doc.by_uid(u).bone for u in uids)
    assert len(doc.history) == steps + 2
    assert doc.undo() and all(doc.by_uid(u).bone for u in uids)


def test_marking_a_mixed_selection_changes_only_the_objects_that_are_not_bones_yet() -> None:
    doc, uids = _boxes(2)
    doc.set_bone(uids[:1], True)
    steps = len(doc.history)
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("mark-bone")) is True
    assert [doc.by_uid(u).bone for u in uids] == [True, True]
    assert len(doc.history) == steps + 1


def test_the_64_bone_cap_is_a_refusal_toast_that_changes_nothing() -> None:
    doc, uids = _boxes(rig.MAX_BONES + 1)
    steps, ctx = len(doc.history), _Ctx()
    assert clay_ops.run(ctx, doc, clay_ops.get("mark-bone")) is False
    assert any("past the 64" in message for message, _ in ctx.toasts)
    assert len(doc.history) == steps and not any(o.bone for o in doc.objects)


def test_unmarking_a_bone_a_weighted_mesh_uses_is_refused_naming_unbind_skin() -> None:
    doc, bone, body = rigged_doc()
    doc.select([bone])
    steps, ctx = len(doc.history), _Ctx()
    assert clay_ops.run(ctx, doc, clay_ops.get("unmark-bone")) is False
    assert any("Unbind Skin" in message for message, _ in ctx.toasts)
    assert doc.by_uid(bone).bone and len(doc.history) == steps


def test_unbind_skin_unbinds_every_selected_weighted_mesh_as_one_undo_step() -> None:
    doc = bd.ClayDoc()
    bone = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Hip", mesh=bp.box(), bone=True)).uid
    bodies = [
        doc.add_object(
            bd.Obj(uid=bd.new_uid(), name=f"Body{i}", mesh=weighted(bp.box()), skin=skin_for(bone))
        ).uid
        for i in range(2)
    ]
    doc.select(bodies)
    steps = len(doc.history)
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("unbind-skin")) is True
    assert all(doc.by_uid(u).skin is None and doc.by_uid(u).mesh.joints is None for u in bodies)
    assert len(doc.history) == steps + 1, "two objects, one gesture, one step"
    assert doc.undo() and all(doc.by_uid(u).skin is not None for u in bodies)


# -- a weighted mesh does not move on its own -------------------------------------------------


def test_the_document_refuses_to_move_a_weighted_mesh_and_names_the_way_out() -> None:
    doc, bone, body = rigged_doc()
    with pytest.raises(el.OpError, match="Move its bones"):
        doc.set_transform(body, translation=(1, 0, 0))
    assert np.allclose(doc.by_uid(body).translation, 0)
    assert doc.set_transform(bone, translation=(1, 0, 0)) is True, "a bone moves freely"
    assert doc.set_transform(body, translation=(0, 0, 0)) is False, "no change is not a refusal"


def test_a_keyboard_drag_skips_a_weighted_mesh_with_a_sentence_and_still_moves_the_rest(gl) -> None:
    from realmspinner.studio.modes.clay.state import ClayState
    from realmspinner.studio.modes.clay.ui import view as clay_view

    class Ctx(_Ctx):
        def __init__(self) -> None:
            super().__init__()
            self.state = type("S", (), {"clay": ClayState()})()
            self.clay_view = None

    ctx = Ctx()
    view = clay_view.ClayView(gl, ctx)
    ctx.clay_view = view
    doc, bone, body = rigged_doc()
    doc.select([body])
    view.frame_selection(doc)
    view.draw(doc, (0.0, 0.0, 128.0, 96.0), 0.0)
    view._last_mouse = (64.0, 48.0)
    assert view.begin_keyboard_drag(doc, "move") is False
    assert any("moves with its bones" in message for message, _ in ctx.toasts)

    doc.select([bone, body])
    ctx.toasts.clear()
    assert view.begin_keyboard_drag(doc, "move") is True
    assert view._drag_uids == [bone]
    view.cancel_drag(doc)
    view.release()


# -- the Rig card ---------------------------------------------------------------------------


def test_the_card_sentences_name_what_each_kind_of_object_is_bound_to() -> None:
    doc = bd.ClayDoc()
    root = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Spine", mesh=bp.box(), bone=True)).uid
    group = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Group", mesh=bp.box(), parent=root)).uid
    arm = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="Arm", mesh=bp.box(), parent=group, bone=True)
    ).uid
    hand = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Hand", mesh=bp.box(), parent=arm)).uid
    free = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Free", mesh=bp.box())).uid
    body = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="Body", mesh=weighted(bp.box()), skin=skin_for(root, arm))
    ).uid

    assert props.rig_lines(doc, doc.by_uid(root))[0] == "A bone (a root bone)."
    assert props.rig_lines(doc, doc.by_uid(arm))[0] == "A bone under 'Spine'."
    assert props.rig_lines(doc, doc.by_uid(hand))[0] == "Moves with the bone 'Arm'."
    assert props.rig_lines(doc, doc.by_uid(group))[0] == "Moves with the bone 'Spine'."
    assert props.rig_lines(doc, doc.by_uid(free))[0] == "Not bound to a bone."
    assert props.rig_lines(doc, doc.by_uid(body))[0] == "Weighted to 2 bones: Spine, Arm."


def test_the_model_card_counts_bones_only_when_there_are_some() -> None:
    doc, uids = _boxes(2)
    assert "bone" not in doc_io.model_counts_line(doc)
    doc.set_bone(uids[:1], True)
    assert doc_io.model_counts_line(doc).endswith(" · 1 bone")
    doc.set_bone(uids, True)
    assert doc_io.model_counts_line(doc).endswith(" · 2 bones")


def test_the_bone_checkbox_marks_the_selected_shape_as_one_undo_step(imgui_ui) -> None:
    ctx, tab, doc = _document(1)
    obj = doc.objects[0]
    widgets.request_open("clay/insp/rig")
    rig_ = _rig(imgui_ui, ctx)
    steps = len(doc.history)
    rig_.click(contains="bone##rigbone")
    assert obj.bone is True and len(doc.history) == steps + 1
    rig_.click(contains="bone##rigbone")
    assert obj.bone is False and len(doc.history) == steps + 2


def _add_rig(doc: bd.ClayDoc) -> tuple[int, int]:
    """A bone and a box weighted to it, added to *doc* through its own door."""
    bone = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Hip", mesh=bp.box(), bone=True)).uid
    body = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="Body", mesh=weighted(bp.box()), skin=skin_for(bone))
    ).uid
    return bone, body


def test_a_weighted_meshs_card_has_an_unbind_button_that_works(imgui_ui) -> None:
    ctx, tab, doc = _document(1)
    bone, body = _add_rig(doc)
    doc.select([body])
    widgets.request_open("clay/insp/rig")
    rig_ = _rig(imgui_ui, ctx)
    assert rig_.find(contains="Unbind Skin##rigunbind").enabled
    rig_.click(contains="Unbind Skin##rigunbind")
    assert doc.by_uid(body).skin is None
    with pytest.raises(LookupError):
        rig_.find(contains="Unbind Skin##rigunbind")


def test_the_outliner_draws_the_bone_glyph_on_a_bone_row_and_the_link_glyph_on_a_weighted_one(
    imgui_ui,
) -> None:
    from realmspinner.studio.modes.clay.ui.panes import inspector as clay_inspector

    ctx, tab, doc = _document(1)
    _add_rig(doc)
    widgets.request_open(clay_inspector.DOCK_KEY)
    rig_ = _rig(imgui_ui, ctx)
    labels = rig_.labels
    assert f"{icons.BONE} Hip##row" in labels
    assert f"{icons.LINK} Body##row" in labels


# -- the doors that are not the panes -------------------------------------------------------


def test_ctrl_k_lists_the_three_rows_and_mark_bone_runs() -> None:
    ctx = palette_ctx(objects=1)
    commands = _clay(ctx)
    for name in NAMES:
        assert commands[f"clay-op:{name}"].label == clay_ops.get(name).label
    row = commands["clay-op:mark-bone"]
    assert row.enabled(ctx)
    row.run(ctx)
    assert any(o.bone for o in ctx.state.clay.active.doc.objects)
    assert not commands["clay-op:unbind-skin"].enabled(ctx) and commands["clay-op:unbind-skin"].why


def test_the_agents_clay_op_can_mark_a_bone() -> None:
    from realmspinner.studio.modes.clay import mode as clay_mode
    from realmspinner.studio.modes.clay.agent import dispatch as agent_clay

    from .test_agent_clay import _Ctx as AgentCtx
    from .test_agent_clay import _new_agent_tab, _payload

    ctx = AgentCtx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    result = agent_clay.call(ctx, session, "clay_op", {"name": "mark-bone"})
    assert result["isError"] is False, result
    assert _payload(result)["ran"] is True
    assert tab.doc.by_uid(uid).bone is True
