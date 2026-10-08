"""Regression tests for the 2026-10-07 Clay audit's cross-file hand-offs: the
callers of the document's own rollback door (clay-05), the GLB skipped-texture
toast (clay-07), a live marquee surviving a tab switch or close (clay-85) and
the human Add-primitive path meeting the document's size ceilings (clay-13).

Each test's name is the claim, and each fails against the code as it stood
before these callers were switched.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import glbimport
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay.agent import dispatch as agent_clay
from realmspinner.studio.modes.clay.ui import view as clay_view
from realmspinner.studio.modes.clay.ui.panes import tools as pane_tools

from .test_agent_clay import _Ctx, _new_agent_tab, _payload
from .test_audit_2026_10_07_fix_io import _png, _textured_glb
from .test_audit_2026_10_07_fix_view import RECT, _doc, _event, _start_marquee
from .test_clay_mode import FakeCtx, _Done

# --- clay-05: a rolled-back run forgets the element selection it made ----------


def _agent_in_face_mode() -> tuple[_Ctx, agent_clay.Session, bd.ClayDoc, int]:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    doc = clay_mode.ensure(ctx).get(session.tab_uid).doc
    return ctx, session, doc, uid


def _subdivide_then_fail(uid: int) -> list[dict]:
    """Face mode, select all six faces, subdivide (24 faces), then a refusal."""
    return [
        {"name": "clay_element_mode", "arguments": {"mode": "face"}},
        {
            "name": "clay_select_elements",
            "arguments": {"uid": uid, "faces": list(range(6))},
        },
        {"name": "clay_op", "arguments": {"name": "subdivide"}},
        {"name": "clay_add_primitive", "arguments": {"generator": "nope"}},
    ]


def test_a_rolled_back_batch_leaves_no_element_selection_on_the_restored_mesh() -> None:
    ctx, session, doc, uid = _agent_in_face_mode()
    obj = next(o for o in doc.objects if o.uid == uid)

    result = agent_clay.call(
        ctx,
        session,
        "clay_batch",
        {"calls": _subdivide_then_fail(uid), "rollback_on_error": True},
    )

    assert result["isError"] is True
    assert _payload(result)["rolled_back"] is True
    assert bm.face_count(obj.mesh) == 6, "the subdivide was rolled back"
    sel = doc.element_sel_of(uid)
    stale = [] if sel.faces is None else [int(f) for f in sel.faces if f >= 6]
    assert el.is_empty(sel), f"face indices {stale[:3]} outlived the mesh they described"


def test_delete_faces_after_a_rolled_back_batch_refuses_cleanly_instead_of_raising_indexerror() -> (
    None
):
    from realmspinner.kernels.mesh import ops_topo

    ctx, session, doc, uid = _agent_in_face_mode()
    obj = next(o for o in doc.objects if o.uid == uid)
    agent_clay.call(
        ctx,
        session,
        "clay_batch",
        {"calls": _subdivide_then_fail(uid), "rollback_on_error": True},
    )
    # What an element-mode consumer does next with whatever selection is left:
    # an empty one is a named refusal, a stale one was an IndexError.
    with pytest.raises(el.OpError):
        ops_topo.delete_faces(obj.mesh, doc.element_sel_of(uid))


def test_a_rollback_after_an_exception_also_drops_the_element_selection() -> None:
    from realmspinner.studio.modes.clay.agent import tools_batch

    ctx, session, doc, uid = _agent_in_face_mode()

    def explodes(doc_: Any, session_: Any) -> dict:
        raise RuntimeError("boom")

    entries: list[Any] = [(c["name"], c["arguments"]) for c in _subdivide_then_fail(uid)[:3]]
    entries.append(explodes)
    with pytest.raises(RuntimeError):
        tools_batch._fold_run(ctx, session, doc, entries, rollback=True, label="x")

    assert el.is_empty(doc.element_sel_of(uid))


def test_a_dry_run_is_undone_through_the_documents_undo_unrecorded_door(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session, "box")
    doc = clay_mode.ensure(ctx).get(session.tab_uid).doc
    seen = {"n": 0}
    real = bd.ClayDoc.undo_unrecorded

    def spy(self: bd.ClayDoc) -> bool:
        seen["n"] += 1
        return real(self)

    monkeypatch.setattr(bd.ClayDoc, "undo_unrecorded", spy)
    objects_before = len(doc.objects)

    result = agent_clay.call(
        ctx,
        session,
        "clay_program",
        {"steps": [{"add": {"generator": "cylinder", "id": "c"}}], "dry_run": True},
    )

    assert result["isError"] is False
    assert len(doc.objects) == objects_before
    assert seen["n"] == 1, "the dry run reversed its step by reaching past the document"


def test_no_direct_unrecorded_stack_undo_is_left_in_the_agent_batch_module() -> None:
    from pathlib import Path

    from realmspinner.studio.modes.clay.agent import tools_batch

    # Prose in docstrings and comments may name the call; no executable line may make it.
    direct = [
        line
        for line in Path(tools_batch.__file__).read_text(encoding="utf-8").splitlines()
        if line.strip().startswith(("doc.history.undo(", "history.undo("))
        and "redoable=False" in line
    ]
    assert not direct, direct


# --- clay-07: a GLB texture Clay could not use is announced --------------------


_CORRUPT_PNG = b"\x89PNG\r\n\x1a\n" + b"junk" * 10


def test_a_task_half_import_of_a_glb_with_an_unreadable_texture_carries_the_count(
    tmp_path: Any,
) -> None:
    ctx = FakeCtx()
    path = tmp_path / "tex.glb"
    path.write_bytes(_textured_glb([_CORRUPT_PNG]))

    clay_mode.import_mesh_path(ctx, path)

    assert ctx.result["skipped_textures"] == 1
    # Task threads publish, they do not write: nothing is toasted on this half.
    assert ctx.toasts == []


def test_adopting_a_glb_whose_texture_was_unusable_toasts_a_warning(tmp_path: Any) -> None:
    ctx = FakeCtx()
    path = tmp_path / "tex.glb"
    path.write_bytes(_textured_glb([_CORRUPT_PNG]))
    clay_mode.import_mesh_path(ctx, path)

    clay_mode.on_task_done(ctx, _Done("clay-import:1", ctx.result))

    assert len(clay_mode.ensure(ctx).docs) == 1
    assert ctx.toasts == [(glbimport.skipped_textures_notice(1), "warn")]


def test_adopting_a_glb_whose_textures_all_loaded_toasts_nothing(tmp_path: Any) -> None:
    ctx = FakeCtx()
    path = tmp_path / "ok.glb"
    path.write_bytes(_textured_glb([_png(8)]))
    clay_mode.import_mesh_path(ctx, path)

    assert ctx.result["skipped_textures"] == 0
    clay_mode.on_task_done(ctx, _Done("clay-import:1", ctx.result))

    assert ctx.toasts == []


def test_the_picker_import_route_reports_skipped_textures_too(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "tex.glb"
    path.write_bytes(_textured_glb([_CORRUPT_PNG]))
    monkeypatch.setattr(clay_mode.dialogs, "open_file", lambda *a, **k: path)
    ctx = FakeCtx()

    clay_mode.ask_import_mesh(ctx)

    assert ctx.result["skipped_textures"] == 1


def test_editing_a_library_asset_reports_skipped_textures_too(tmp_path: Any) -> None:
    data = _textured_glb([_CORRUPT_PNG])

    result = clay_mode._parse_glb(data, "asset")

    assert result["skipped_textures"] == 1


def test_a_non_glb_import_reports_no_skipped_textures(tmp_path: Any) -> None:
    quad = tmp_path / "quad.obj"
    quad.write_bytes(b"v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n")
    ctx = FakeCtx()

    clay_mode.import_mesh_path(ctx, quad)

    assert ctx.result["skipped_textures"] == 0


def test_a_big_import_toasts_its_skipped_textures_once_the_user_says_go_on(
    tmp_path: Any,
) -> None:
    ctx = FakeCtx()
    path = tmp_path / "tex.glb"
    path.write_bytes(_textured_glb([_CORRUPT_PNG]))
    clay_mode.import_mesh_path(ctx, path)
    big = dict(ctx.result, triangles=clay_mode.SLOW_TRIANGLES + 1)

    clay_mode.on_task_done(ctx, _Done("clay-import:1", big))
    # Held back with the rest of the import until the question is answered...
    assert clay_mode.ensure(ctx).docs == []
    ctx.confirms.pending.on_confirm()

    assert len(clay_mode.ensure(ctx).docs) == 1
    assert ctx.toasts == [(glbimport.skipped_textures_notice(1), "warn")]


# --- clay-85: a live marquee does not survive a tab switch or a close ----------


def _two_tabs_with_a_view(gl: Any) -> tuple[Any, Any, Any, Any, Any]:
    old, new = _doc(count=1), _doc(count=1)
    state = clay_mode.ClayState()
    tab_a, tab_b = clay_mode.ClayTab(doc=old), clay_mode.ClayTab(doc=new)
    state.add(tab_a)
    state.add(tab_b)
    state.active_uid = tab_a.uid
    ctx = SimpleNamespace(state=SimpleNamespace(clay=state, mode="clay"), settings=None)
    view = clay_view.ClayView(gl, ctx)
    ctx.clay_view = view
    clay_mode.ensure(ctx)  # wires ClayState.settle_drag to this view
    return ctx, state, view, (tab_a, tab_b), (old, new)


def test_a_marquee_in_progress_is_dropped_by_a_tab_switch_and_the_release_selects_nothing(
    gl: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pygame

    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0, raising=False)
    ctx, state, view, (tab_a, tab_b), (old, new) = _two_tabs_with_a_view(gl)
    try:
        _start_marquee(view, old)
        assert not view.dragging, "a marquee is not a transform drag: the old gate skipped it"

        state.activate(tab_b.uid)  # what Ctrl+Tab and the tab bar do

        assert view._grab is None and view.marquee is None and view._marquee_from is None
        new.set_element_mode("face")
        view.draw(new, RECT, 0.0)
        view.handle_event(new, _event(pygame.MOUSEBUTTONUP, button=1, pos=(190, 140)), True)
        assert not new.element_sel, "the old document's rectangle selected in the new one"
    finally:
        view.release()


def test_closing_the_active_tab_mid_marquee_drops_the_marquee(
    gl: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pygame

    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0, raising=False)
    ctx = FakeCtx()
    state = clay_mode.ensure(ctx)
    old, new = _doc(count=1), _doc(count=1)
    tab_a, tab_b = clay_mode.ClayTab(doc=old), clay_mode.ClayTab(doc=new)
    tab_a.mark_saved()  # a clean document closes without the unsaved-work question
    state.add(tab_a)
    state.add(tab_b)
    state.active_uid = tab_a.uid
    view = clay_view.ClayView(gl, ctx)
    ctx.clay_view = view
    try:
        _start_marquee(view, old)

        clay_mode.close_tab(ctx, tab_a.uid)

        assert state.get(tab_a.uid) is None
        assert view._grab is None and view.marquee is None and view._marquee_from is None
    finally:
        view.release()


# --- clay-13: the human Add-primitive path meets the document ceilings ---------


class _ToastCtx:
    def __init__(self) -> None:
        self.toasts: list[tuple[str, str]] = []

    def toast(self, message: str, level: str = "info") -> None:
        self.toasts.append((message, level))


def _full_document(monkeypatch: pytest.MonkeyPatch, objects: int = 3) -> bd.ClayDoc:
    monkeypatch.setattr(glbimport, "MAX_OBJECTS", objects)
    doc = bd.ClayDoc()
    for i in range(objects):
        doc.add_object(bd.Obj(uid=bd.new_uid(), name=f"o{i}", mesh=bp.box()))
    return doc


def test_adding_a_primitive_to_a_full_document_toasts_the_refusal_and_pushes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc = _full_document(monkeypatch)
    ctx = _ToastCtx()
    steps, objects = len(doc.history), len(doc.objects)

    placed = pane_tools.place_primitive(ctx, doc, "box")

    assert placed is None
    assert len(ctx.toasts) == 1 and ctx.toasts[0][1] == "error"
    assert "3" in ctx.toasts[0][0] or "object" in ctx.toasts[0][0].lower()
    assert len(doc.history) == steps and len(doc.objects) == objects
    assert doc.history._open_gestures == 0, "a refused add left a gesture open"


def test_adding_a_primitive_to_a_document_with_room_still_places_and_selects_it() -> None:
    doc = bd.ClayDoc()
    ctx = _ToastCtx()

    placed = pane_tools.place_primitive(ctx, doc, "box")

    assert placed is not None and doc.objects == [placed]
    assert doc.selection == {placed.uid}
    assert ctx.toasts == []


def test_the_add_primitive_door_itself_still_raises_for_the_agent_to_translate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ``_h_add_primitive`` turns this OpError into its own tool refusal, so the
    # raising door stays; only the human-facing wrapper swallows and toasts.
    doc = _full_document(monkeypatch)
    with pytest.raises(el.OpError):
        pane_tools.add_primitive(_ToastCtx(), doc, "box")


def test_the_add_menu_places_through_the_refusing_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``draw_add_menu`` cannot be built headlessly, so its imgui calls are
    stubbed down to the one that matters: a click on a shape, on a full
    document, must come out as a toast and not as an exception out of the draw."""
    doc = _full_document(monkeypatch)
    ctx = _ToastCtx()
    state = SimpleNamespace(generator="box")
    tab = SimpleNamespace(doc=doc, saving=False)
    monkeypatch.setattr(pane_tools.widgets, "section", lambda *a, **k: None)
    monkeypatch.setattr(pane_tools.widgets, "field_label", lambda *a, **k: None)
    monkeypatch.setattr(pane_tools.manual_render, "help_button", lambda *a, **k: None)
    monkeypatch.setattr(pane_tools.controls, "menu_item", lambda *a, **k: (True, True))

    pane_tools.draw_add_menu(ctx, state, tab)

    assert ctx.toasts, "the refusal never reached the user"
    assert len(doc.objects) == 3
    assert doc.history._open_gestures == 0
