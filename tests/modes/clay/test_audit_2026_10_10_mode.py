"""The 2026-10-10 audit's Clay mode findings: clay-03, 18, 19, 20, 21, 46, 47 and 49.
Each test's name is the claim."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.service.errors import Conflict
from realmspinner.studio import dialogs, journal
from realmspinner.studio.modes.clay import doc_io
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay import ops as clay_ops
from realmspinner.studio.modes.clay import starter_colours as sc

from .test_clay_mode import FakeCtx

_HEADER = b"# Written by Realmspinner's Clay\n"


class _Done:
    def __init__(self, key: str, result: Any = None) -> None:
        self.key = key
        self.result = result


def _box_tab(ctx: FakeCtx, title: str = "Scene") -> tuple[Any, bd.ClayDoc, Any]:
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box(), generator="box"))
    tab = clay_mode.adopt(ctx, doc, title=title)
    return tab, doc, obj


def _dragging(ctx: FakeCtx) -> None:
    ctx.clay_view = SimpleNamespace(dragging=True, grabbing=False)


# --- clay-03 -------------------------------------------------------------------


def test_export_obj_refuses_a_png_the_previous_clay_mtl_did_not_name(tmp_path: Path) -> None:
    # A previous Clay export with one texture (slot 1)...
    (tmp_path / "barrel.obj").write_bytes(_HEADER)
    (tmp_path / "barrel.mtl").write_bytes(_HEADER + b"newmtl a\nmap_Kd barrel_1.png\n")
    (tmp_path / "barrel_1.png").write_bytes(b"clay-png-1")
    # ...and the user's own picture that happens to share the naming scheme.
    (tmp_path / "barrel_2.png").write_bytes(b"USERS OWN IMAGE")
    targets = {
        tmp_path / "barrel_1.png": b"new1",
        tmp_path / "barrel_2.png": b"new2",
        tmp_path / "barrel.mtl": _HEADER + b"newmtl a\n",
        tmp_path / "barrel.obj": b"x",
    }

    with pytest.raises(Conflict) as caught:
        clay_mode._refuse_foreign_sidecars(targets, tmp_path / "barrel.obj")

    assert "barrel_2.png" in str(caught.value)
    assert (tmp_path / "barrel_2.png").read_bytes() == b"USERS OWN IMAGE"


def test_export_obj_still_replaces_a_png_the_previous_clay_mtl_named(tmp_path: Path) -> None:
    (tmp_path / "barrel.obj").write_bytes(_HEADER)
    (tmp_path / "barrel.mtl").write_bytes(_HEADER + b"newmtl a\nmap_Kd barrel_1.png\n")
    (tmp_path / "barrel_1.png").write_bytes(b"clay-png-1")
    targets = {
        tmp_path / "barrel_1.png": b"new1",
        tmp_path / "barrel.mtl": _HEADER,
        tmp_path / "barrel.obj": b"x",
    }

    clay_mode._refuse_foreign_sidecars(targets, tmp_path / "barrel.obj")  # does not raise


# --- clay-18 -------------------------------------------------------------------


def test_undo_redo_and_step_history_refuse_while_the_tab_is_saving() -> None:
    ctx = FakeCtx()
    tab, doc, obj = _box_tab(ctx)
    second = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Two", mesh=bp.box(), generator="box"))
    tab.saving = True  # an export is reading the document on a task thread

    clay_mode.undo(ctx, tab)
    assert [o.uid for o in doc.objects] == [obj.uid, second.uid], "undo ran under a save"

    tab.saving = False
    clay_mode.undo(ctx, tab)
    assert [o.uid for o in doc.objects] == [obj.uid]

    tab.saving = True
    clay_mode.redo(ctx, tab)
    assert [o.uid for o in doc.objects] == [obj.uid], "redo ran under a save"

    assert clay_mode.step_history(ctx, tab, 0) is False
    assert [o.uid for o in doc.objects] == [obj.uid], "a history jump ran under a save"


# --- clay-19 -------------------------------------------------------------------


def test_starter_colour_click_under_a_live_drag_adds_no_slot_and_pushes_no_step() -> None:
    # Faces mode: the slot used to be added, then ``ops.run`` refused.
    ctx = FakeCtx()
    tab, doc, obj = _box_tab(ctx)
    doc.select([obj.uid])
    doc.set_element_mode("face")
    doc.set_element_sel(obj.uid, el.ElementSel(faces=np.array([0], dtype=np.int64)))
    _dragging(ctx)
    slots, head = len(doc.materials), doc.history.head

    sc.paint_faces(ctx, doc, sc.by_name("Blue"))

    assert len(doc.materials) == slots, "a stray palette slot was left behind"
    assert doc.history.head == head, "an empty history step was pushed"

    # Object mode: ``repaint_object`` ran directly and pushed mid-gesture.
    ctx = FakeCtx()
    tab, doc, obj = _box_tab(ctx)
    doc.select([obj.uid])
    _dragging(ctx)
    slots, head = len(doc.materials), doc.history.head

    sc.paint(ctx, doc, [obj.uid], sc.by_name("Red"))

    assert len(doc.materials) == slots
    assert doc.history.head == head, "a step was pushed mid-gesture"
    assert tab.last_slot != len(doc.materials), "the refused slot was remembered"


def test_starter_colour_paint_faces_whose_op_refuses_keeps_no_added_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Not a drag: the op refuses for its own reason after the slot was added.
    ctx = FakeCtx()
    tab, doc, obj = _box_tab(ctx)
    doc.select([obj.uid])
    doc.set_element_mode("face")
    doc.set_element_sel(obj.uid, el.ElementSel(faces=np.array([0], dtype=np.int64)))
    slots, head = len(doc.materials), doc.history.head
    monkeypatch.setattr(clay_ops, "run", lambda *a, **k: False)

    result = sc.paint_faces(ctx, doc, sc.by_name("Blue"))

    assert result is None
    assert len(doc.materials) == slots
    assert doc.history.head == head


# --- clay-20 -------------------------------------------------------------------


def test_extrude_key_whose_op_raises_closes_its_history_gesture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = FakeCtx()
    ctx.clay_view = SimpleNamespace(dragging=False, grabbing=False)
    tab, doc, obj = _box_tab(ctx)
    doc.select([obj.uid])
    doc.set_element_mode("face")
    doc.set_element_sel(obj.uid, el.ElementSel(faces=np.array([0], dtype=np.int64)))
    op = clay_ops.get("extrude")

    def boom(*args: Any, **kwargs: Any) -> bool:
        raise MemoryError("simulated numpy failure")

    monkeypatch.setattr(clay_ops, "run", boom)
    assert doc.history._open_gestures == 0

    with pytest.raises(MemoryError):
        clay_mode._extrude_and_drag(ctx, doc, op)

    assert doc.history._open_gestures == 0, "the gesture mark was left open"


# --- clay-21 -------------------------------------------------------------------


def test_a_failed_clay_open_forgets_the_recent_path(tmp_path: Path) -> None:
    ctx = FakeCtx()
    gone = tmp_path / "moved.rblk"  # never written: the open will fail
    clay_mode.remember_path(ctx, gone)
    assert str(gone) in clay_mode.recent_paths(ctx) or any(
        Path(p) == gone for p in clay_mode.recent_paths(ctx)
    )
    keys: list[str] = []
    ctx.submit = lambda key, run, *a, **k: keys.append(key) or True  # type: ignore[method-assign]

    clay_mode.open_path(ctx, gone)
    clay_mode.on_task_failed(ctx, _Done(keys[-1]))

    assert not any(Path(p) == gone for p in clay_mode.recent_paths(ctx)), (
        "a dead path stays in every recent list"
    )


def test_a_failed_clay_open_whose_path_holds_a_colon_still_forgets_it(tmp_path: Path) -> None:
    # A Windows path carries a drive colon; the key is split on the first one only.
    ctx = FakeCtx()
    gone = tmp_path / "ab.rblk"
    clay_mode.remember_path(ctx, gone)
    clay_mode.on_task_failed(ctx, _Done(f"clay-open:{gone}"))
    assert not any(Path(p) == gone for p in clay_mode.recent_paths(ctx))


# --- clay-46 -------------------------------------------------------------------


def test_outputs_why_for_an_empty_document_does_not_claim_objects_are_hidden() -> None:
    why = doc_io.outputs_why(bd.ClayDoc(), False)
    assert why, "an empty document still refuses its output buttons"
    assert "hidden" not in why
    assert "shape" in why.lower()


def test_outputs_why_keeps_the_hidden_sentence_when_every_object_is_hidden() -> None:
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box(), generator="box"))
    obj.visible = False
    assert "hidden" in doc_io.outputs_why(doc, False)


# --- clay-47 -------------------------------------------------------------------


def test_journal_does_not_snapshot_a_tab_with_a_live_drag() -> None:
    ctx = FakeCtx()
    tab, doc, obj = _box_tab(ctx)
    doc.select([obj.uid])
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="Two", mesh=bp.box(), generator="box"))
    assert tab.dirty
    provider = next(p for p in journal.providers() if p.kind == "clay")

    assert provider.slots(ctx) == [tab], "a dirty, idle tab is the ordinary slot"
    _dragging(ctx)
    assert provider.slots(ctx) == [], "a half-dragged TRS would be encoded"


# --- clay-49 -------------------------------------------------------------------


def test_save_as_onto_another_open_tabs_path_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = FakeCtx()
    first, _doc, _obj = _box_tab(ctx, "First")
    clay_mode.save_to(ctx, first, tmp_path / "first.rblk")
    clay_mode.on_task_done(ctx, _Done(f"clay-save:{first.uid}", ctx.result))
    assert first.path is not None
    held = (tmp_path / "first.rblk").read_bytes()

    second, _doc2, _obj2 = _box_tab(ctx, "Second")
    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: tmp_path / "first.rblk")

    with pytest.raises(Conflict):
        clay_mode.save_as(ctx, second)
    # What the real task runner does with the raise.
    clay_mode.on_task_failed(ctx, _Done(f"clay-saveas:{second.uid}"))

    assert (tmp_path / "first.rblk").read_bytes() == held, "the other tab's file was overwritten"
    assert second.path is None
    assert second.saving is False
