"""The 2026-10-07 audit's Clay mode findings: clay-02, 03, 07, 12, 15, 23, 24,
51, 60, 75 (the UV pane's half) and 76."""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pygame
import pytest
from _ui_context import imgui_context

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import serialize
from realmspinner.studio import dialogs
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay.ui.panes import uv as clay_uv
from realmspinner.studio.viewer.camera import Camera

from .test_clay_mode import FakeCtx


@pytest.fixture
def ui(monkeypatch):
    with imgui_context(monkeypatch) as imgui:
        yield imgui


def _box_doc() -> tuple[bd.ClayDoc, Any]:
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box(), generator="box"))
    return doc, obj


def _quad_mesh() -> bm.Mesh:
    return bm.from_faces(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        [[0, 1, 2, 3]],
        uv=[[[0.0, 0.0], [0.4, 0.0], [0.4, 0.4], [0.0, 0.4]]],
    )


def _key(key: int, mod: int = 0) -> Any:
    return pygame.event.Event(pygame.KEYDOWN, key=key, mod=mod)


# --- clay-02 ------------------------------------------------------------------


def test_the_first_clay_frame_keeps_an_opened_documents_stored_camera() -> None:
    ctx = FakeCtx()
    ctx.clay_view = SimpleNamespace(camera=Camera(), frame_selection=lambda doc: None)
    doc, _ = _box_doc()
    stored = {"yaw": 2.0, "pitch": 0.2, "distance": 30.0, "target": (5.0, 1.0, 5.0)}
    tab = clay_mode.adopt(ctx, doc, title="Big", view=stored)
    state = clay_mode.ensure(ctx)
    assert state.camera_tab is None, "nothing has been drawn yet"

    # The viewport calls this first on every frame, before its tab handoff.
    clay_mode.sync_active_camera(ctx)

    assert tab.view.distance == 30.0
    assert tab.view.yaw == 2.0
    clay_mode.apply_camera(ctx, tab)
    assert ctx.clay_view.camera.distance == 30.0


# --- clay-03 ------------------------------------------------------------------


def _hovered_uv_tab(ctx: FakeCtx, *, mode: str) -> tuple[Any, Any, Any]:
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box(), generator="box"))
    tab = clay_mode.adopt(ctx, doc, title="T")
    doc.select([obj.uid])
    if mode == "face":
        doc.set_element_mode("face")
        doc.set_element_sel(obj.uid, el.ElementSel(faces=[0]))
    tab.uv_view.selected_islands = frozenset({0})
    return tab, doc, obj


def test_e_over_the_uv_canvas_arms_the_uv_rotate_and_does_not_extrude_or_change_the_tool() -> None:
    ctx = FakeCtx()
    tab, doc, obj = _hovered_uv_tab(ctx, mode="face")
    state = clay_mode.ensure(ctx)
    state.frame_serial = 40
    tab.uv_view.key_hover_at = 40  # what the pane stamps on a frame it is hovered
    before = len(doc.by_uid(obj.uid).mesh.positions)

    assert clay_mode.handle_key(ctx, _key(pygame.K_e)) is True
    assert len(doc.by_uid(obj.uid).mesh.positions) == before, "E extruded the 3-D faces"
    assert state.tool == "select"

    doc.set_element_mode("object")
    assert clay_mode.handle_key(ctx, _key(pygame.K_e)) is True
    assert clay_mode.handle_key(ctx, _key(pygame.K_r)) is True
    assert state.tool == "select", "E/R over the UV canvas switched the 3-D tool"

    # The press lands before the next frame's panes draw, so last frame's stamp counts.
    state.frame_serial = 41
    assert clay_mode.handle_key(ctx, _key(pygame.K_e)) is True
    assert state.tool == "select"


def test_e_with_the_pointer_off_the_uv_canvas_still_reaches_clays_key_layer() -> None:
    ctx = FakeCtx()
    tab, doc, obj = _hovered_uv_tab(ctx, mode="face")
    state = clay_mode.ensure(ctx)
    # Never recorded, and a record gone stale (the pane stopped drawing). No
    # clock is read, so there is nothing to wait out: only the frame count moves.
    state.frame_serial = 40
    for stamp in (None, 37):
        tab.uv_view.key_hover_at = stamp
        doc.set_element_sel(obj.uid, el.ElementSel(faces=[0]))
        before = len(doc.by_uid(obj.uid).mesh.positions)
        clay_mode.handle_key(ctx, _key(pygame.K_e))
        assert len(doc.by_uid(obj.uid).mesh.positions) > before
        assert doc.undo()
    doc.set_element_mode("object")
    clay_mode.handle_key(ctx, _key(pygame.K_e))
    assert state.tool == "rotate"


def test_the_uv_canvas_records_that_the_pointer_is_over_it_with_islands_boxed(
    ui, monkeypatch
) -> None:
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Q", mesh=_quad_mesh()))
    view_state = clay_uv.UvPaneState(
        for_uid=obj.uid, selected_islands=frozenset({0}), frame_seen=7
    )
    tab = SimpleNamespace(doc=doc, uv_view=view_state, saving=False, uid="bd1")
    hovered = {"now": True}
    monkeypatch.setattr(ui, "is_item_hovered", lambda *a, **k: hovered["now"])

    def frame() -> None:
        ui.new_frame()
        ui.begin("##host")
        try:
            clay_uv._canvas(ctx=None, tab=tab, doc=doc, obj=obj, view_state=view_state)
        finally:
            ui.end()
            ui.end_frame()

    frame()
    assert view_state.key_hover_at == 7
    hovered["now"] = False
    frame()
    assert view_state.key_hover_at is None
    hovered["now"] = True
    view_state.selected_islands = frozenset()
    frame()
    assert view_state.key_hover_at is None, "nothing boxed: the keys are Clay's"


def test_every_drawn_clay_frame_advances_the_frame_serial_the_uv_stamp_is_read_against() -> None:
    import inspect

    from realmspinner.studio.modes.clay.ui import viewport

    source = inspect.getsource(viewport.ClayViewport._clay_viewport)
    assert source.index("frame_serial += 1") < source.index("self._clay_tabs(")
    assert "frame_serial" in inspect.getsource(clay_uv._body)


# --- clay-07 ------------------------------------------------------------------


def _textured_doc() -> bd.ClayDoc:
    doc = bd.ClayDoc(materials=[bd.default_material("A"), bd.default_material("B")])
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box(), generator="box"))
    doc.add_texture(0, 32)
    return doc


def _export_obj(ctx: FakeCtx, doc: bd.ClayDoc, target: Path, monkeypatch, title: str = "barrel"):
    tab = clay_mode.adopt(ctx, doc, title=title)
    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: target)
    clay_mode.export_mesh_file(ctx, tab, "obj")
    return tab


def test_a_failed_png_write_leaves_no_new_obj_or_mtl_and_never_replaces_a_foreign_sidecar(
    tmp_path, monkeypatch
) -> None:
    from realmspinner.service.errors import Conflict

    target = tmp_path / "barrel.obj"

    # Half one: a PNG that cannot be staged takes the whole set with it.
    real_write = Path.write_bytes

    def flaky(self: Path, data: Any) -> int:
        if ".png" in self.name:
            raise OSError("disk full")
        return real_write(self, data)

    monkeypatch.setattr(Path, "write_bytes", flaky)
    with pytest.raises(OSError, match="disk full"):
        _export_obj(FakeCtx(), _textured_doc(), target, monkeypatch)
    monkeypatch.setattr(Path, "write_bytes", real_write)
    assert sorted(p.name for p in tmp_path.iterdir()) == [], "a torn set was left behind"

    # Half two: a sidecar that is not a previous Clay export is not replaced.
    (tmp_path / "barrel_0.png").write_bytes(b"MY OWN ARTWORK")
    with pytest.raises(Conflict) as caught:
        _export_obj(FakeCtx(), _textured_doc(), target, monkeypatch)
    assert "barrel_0.png" in str(caught.value)
    assert (tmp_path / "barrel_0.png").read_bytes() == b"MY OWN ARTWORK"
    assert not target.exists() and not (tmp_path / "barrel.mtl").exists()

    (tmp_path / "barrel_0.png").unlink()
    (tmp_path / "barrel.mtl").write_text("# my hand-written material library\n")
    with pytest.raises(Conflict):
        _export_obj(FakeCtx(), _textured_doc(), target, monkeypatch)
    assert (tmp_path / "barrel.mtl").read_text() == "# my hand-written material library\n"

    # A previous export of Clay's own is still replaced without asking.
    (tmp_path / "barrel.mtl").unlink()
    _export_obj(FakeCtx(), _textured_doc(), target, monkeypatch)
    _export_obj(FakeCtx(), _textured_doc(), target, monkeypatch)
    assert (tmp_path / "barrel_0.png").read_bytes()[:4] == b"\x89PNG"


def test_an_obj_export_names_the_mtl_and_pngs_by_the_names_the_obj_text_carries(
    tmp_path, monkeypatch
) -> None:
    """A stem with ``#`` is neutralised in the OBJ's ``mtllib`` line, so the
    ``.mtl`` beside it must carry the same neutralised name."""
    from realmspinner.kernels.mesh import objexport

    target = tmp_path / "a#b.obj"
    _export_obj(FakeCtx(), _textured_doc(), target, monkeypatch, title="Scene")
    safe = objexport.safe_name("a#b")
    assert safe != "a#b"
    assert f"mtllib {safe}.mtl" in target.read_text(encoding="utf-8").splitlines()
    mtl = tmp_path / f"{safe}.mtl"
    assert mtl.is_file()
    assert f"map_Kd {objexport.texture_name('a#b', 0)}" in mtl.read_text().splitlines()
    assert (tmp_path / objexport.texture_name("a#b", 0)).is_file()


# --- clay-12 ------------------------------------------------------------------


def _armed_uv_tab(ctx: FakeCtx) -> tuple[Any, Any, Any, int]:
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Q", mesh=_quad_mesh()))
    tab = clay_mode.adopt(ctx, doc, title="A")
    doc.select([obj.uid])
    view_state = tab.uv_view
    view_state.selected_islands = frozenset({0})
    ids = np.array([0], dtype="i4")
    base = doc.by_uid(obj.uid).mesh
    before = len(doc.history)
    assert clay_uv.begin_live_transform(doc, view_state, "rotate", base, ids, (0.2, 0.2))
    clay_uv.update_live_transform(doc, obj.uid, view_state, (0.3, 0.25))
    clay_uv.update_live_transform(doc, obj.uid, view_state, (0.1, 0.35))
    assert doc.history._open_gestures == 1
    return tab, doc, obj, before


def test_switching_tabs_commits_an_armed_uv_live_transform_and_closes_its_gesture() -> None:
    ctx = FakeCtx()
    tab, doc, obj, before = _armed_uv_tab(ctx)
    other = bd.ClayDoc()
    other.add_object(bd.Obj(uid=bd.new_uid(), name="Other", mesh=bp.box()))
    clay_mode.adopt(ctx, other, title="B")  # the active tab is now B

    assert tab.uv_view.drag_mode == ""
    assert doc.history._open_gestures == 0, "the abandoned tab's undo gesture is still open"
    assert len(doc.history) == before + 1, "the frames fold into one undo step"
    assert tab.uv_view.drag_base is None


def test_emptying_the_selection_commits_an_armed_uv_live_transform(ui, monkeypatch) -> None:
    ctx = FakeCtx()
    tab, doc, obj, before = _armed_uv_tab(ctx)
    doc.select([])
    monkeypatch.setattr(clay_uv.clay_mode, "ensure", lambda c: SimpleNamespace(active=tab))
    ui.new_frame()
    ui.begin("##host")
    try:
        clay_uv._body(ctx=None)
    finally:
        ui.end()
        ui.end_frame()

    assert tab.uv_view.drag_mode == ""
    assert doc.history._open_gestures == 0
    assert len(doc.history) == before + 1


# --- clay-15 ------------------------------------------------------------------


def test_an_obj_and_mtl_with_a_utf8_bom_import_as_if_without_one(tmp_path) -> None:
    mtl_text = "newmtl Red\nKd 1 0 0\n"
    (tmp_path / "a.mtl").write_bytes(b"\xef\xbb\xbf" + mtl_text.encode())
    obj = tmp_path / "a.obj"
    obj.write_bytes(b"\xef\xbb\xbfmtllib a.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nusemtl Red\nf 1 2 3\n")

    found = clay_mode._sibling_mtl(obj, obj.read_bytes())

    assert found is not None
    assert not found.startswith("﻿"), "the BOM hid the first newmtl"
    assert found.splitlines()[0] == "newmtl Red"


# --- clay-23 ------------------------------------------------------------------

V3 = Path(__file__).parent / "data" / "v3_stack.rblk"


def test_opening_a_v3_file_raises_a_sticky_warning_with_bounded_text() -> None:
    from realmspinner.studio import state as app_state

    ctx = FakeCtx()
    doc = serialize.read_rblk(V3.read_bytes())
    assert len(doc.notices) > 3, "the fixture must have more sentences than the cap shows"
    clay_mode.adopt(ctx, doc, title="v3")

    ((text, level),) = ctx.toasts
    assert level == "warn" and level in app_state.TOAST_STICKY
    assert len(text) < 450, len(text)
    assert "more" in text.splitlines()[-1] or " more" in text
    assert any("dropped" in n for n in doc.notices)
    assert "dropped" in text, "the sentence that says something was lost must be shown"
    assert "older" not in text


def test_a_migration_that_dropped_nothing_stays_an_info_toast() -> None:
    ctx = FakeCtx()
    doc, _ = _box_doc()
    doc.notices = ["This model used modifiers, which Clay no longer has. They were applied."]
    clay_mode.announce_migration(ctx, doc)
    assert [level for _, level in ctx.toasts] == ["info"]

    # A current-format file can carry a notice too (a frozen unknown generator):
    # the lead sentence must not call it an older file.
    ctx = FakeCtx()
    doc.notices = ['"Gear" used a shape Clay no longer builds; the mesh is kept as a plain mesh.']
    clay_mode.announce_migration(ctx, doc)
    ((text, level),) = ctx.toasts
    assert level == "info" and "older" not in text and "Gear" in text


# --- clay-24 and clay-60 ------------------------------------------------------


def _journal_ctx(tmp_path: Path) -> tuple[FakeCtx, list[Any]]:
    ctx = FakeCtx()
    ctx.svc = SimpleNamespace(config=SimpleNamespace(autosave_dir=tmp_path))
    deferred: list[Any] = []
    ctx.submit = lambda key, run, *a, **k: deferred.append(run) or True
    return ctx, deferred


def test_the_clay_journal_encodes_its_archive_on_the_task_thread(tmp_path, monkeypatch) -> None:
    from realmspinner.studio import journal

    ctx, deferred = _journal_ctx(tmp_path)
    doc, obj = _box_doc()
    tab = clay_mode.adopt(ctx, doc, title="J")
    doc.set_props(obj.uid, name="Edited")  # dirty
    calls: list[str] = []
    real = serialize.snapshot_bytes

    def counting(snap: Any) -> bytes:
        calls.append(threading.current_thread().name)
        return real(snap)

    monkeypatch.setattr(serialize, "snapshot_bytes", counting)

    assert journal.write(ctx, clay_mode.JOURNAL, tab) is True

    assert calls == [], "journal.write built the whole archive on the frame thread"
    assert len(deferred) == 1
    deferred[0]()
    assert len(calls) == 1
    pair = [p for p in tmp_path.iterdir() if p.name.endswith(".rblk")]
    assert len(pair) == 1
    assert len(serialize.read_rblk(pair[0].read_bytes()).objects) == 1


def test_an_oversize_document_still_fails_the_journal_write_rather_than_being_skipped(
    tmp_path, monkeypatch
) -> None:
    from realmspinner.kernels.mesh import glbimport

    ctx = FakeCtx()
    doc, _ = _box_doc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="Two", mesh=bp.box()))
    tab = clay_mode.adopt(ctx, doc, title="Big")
    monkeypatch.setattr(glbimport, "MAX_OBJECTS", 1)

    deferred = clay_mode.JOURNAL.encode(tab)

    assert callable(deferred)
    with pytest.raises(ValueError, match="objects"):
        deferred()


def test_a_recovered_tab_that_was_never_drawn_is_framed_not_given_the_default_camera() -> None:
    ctx = FakeCtx()
    doc, _ = _box_doc()
    tab = clay_mode.adopt(ctx, doc, title="Never drawn")
    assert not tab.view.fitted

    data = clay_mode._journal_encode(tab)
    assert serialize.read_view(data) is None

    tab.view.yaw, tab.view.distance, tab.view.fitted = 1.25, 7.5, True
    stored = serialize.read_view(clay_mode.JOURNAL.encode(tab)())
    assert stored is not None and stored["distance"] == 7.5


# --- clay-51 ------------------------------------------------------------------


def test_quit_releases_every_tab_texture_family_including_clays_uv_texture(monkeypatch) -> None:
    from realmspinner.studio import main
    from realmspinner.studio.state import AppState

    stub = SimpleNamespace(quit=lambda: None, display=SimpleNamespace(set_caption=lambda _t: None))
    monkeypatch.setitem(sys.modules, "pygame", stub)

    class Texture:
        released = False

        def release(self) -> None:
            self.released = True

    held = Texture()
    state = AppState()
    state.preview["clay_uv_tex:bd1:img"] = held
    state.preview["clay_uv_tex:bd1:src"] = ("stamp", False)
    settings = SimpleNamespace(
        get=lambda key, default=None: default, set=lambda key, value: None, flush=lambda: None
    )
    ctx = SimpleNamespace(settings=settings, state=state, textures=None)
    app = main.App.__new__(main.App)
    app.runtime = SimpleNamespace(shutdown=lambda: None)
    app.window = app.imgui_renderer = app.viewer = app.eta = None
    app.clay_view = None
    app.app_ctx = ctx
    app._running = False
    app._last_frame = app._started_at = 0.0
    from realmspinner.studio.fps import FpsMeter

    app.fps = FpsMeter()

    app.teardown()

    assert held.released, "quit left Clay's UV pane texture alive in the one GL context"
    assert not [k for k in state.preview if k.startswith("clay_uv_tex:")]


# --- clay-75 (the UV pane's half) ---------------------------------------------


def test_apply_scale_and_apply_rotate_explain_why_they_are_greyed() -> None:
    for fn in (clay_uv.apply_rotate_reason, clay_uv.apply_scale_reason):
        assert fn(saving=False, live=False, selected=1, pending_scale=1.0) == ""
        assert fn(saving=True, live=False, selected=1, pending_scale=1.0)
        assert fn(saving=False, live=True, selected=1, pending_scale=1.0)
        assert fn(saving=False, live=False, selected=0, pending_scale=1.0)
    assert not clay_uv.apply_rotate_reason(saving=False, live=False, selected=1, pending_scale=0.0)
    reason = clay_uv.apply_scale_reason(saving=False, live=False, selected=1, pending_scale=0.0)
    assert reason and "0" in reason
    assert clay_uv.apply_scale_reason(saving=False, live=False, selected=1, pending_scale=-1.0)


def test_the_uv_toolbar_hands_the_reasons_to_the_buttons(ui, monkeypatch) -> None:
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Q", mesh=_quad_mesh()))
    view_state = clay_uv.UvPaneState(selected_islands=frozenset(), pending_scale=0.0)
    tab = SimpleNamespace(doc=doc, uv_view=view_state, saving=False, uid="bd1")
    seen: dict[str, str] = {}
    def spy(label: str, enabled: bool, *a: Any, reason: str = "", **k: Any) -> bool:
        seen[label] = reason
        return False

    monkeypatch.setattr(clay_uv.widgets, "disabled_button", spy)
    ui.new_frame()
    ui.begin("##host")
    try:
        clay_uv._toolbar(ctx=None, tab=tab, doc=doc, obj=obj, view_state=view_state)
    finally:
        ui.end()
        ui.end_frame()
    assert seen["Apply##uvrotate"] and seen["Apply##uvscale"]


# --- clay-76 ------------------------------------------------------------------


def test_the_uv_canvas_does_not_recompute_touched_islands_while_only_the_view_moves(
    ui, monkeypatch
) -> None:
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Q", mesh=_quad_mesh()))
    doc.select([obj.uid])
    doc.set_element_mode("face")
    doc.set_element_sel(obj.uid, el.ElementSel(faces=[0]))
    view_state = clay_uv.UvPaneState(for_uid=obj.uid)
    tab = SimpleNamespace(doc=doc, uv_view=view_state, saving=False, uid="bd1")
    calls: list[int] = []
    real = clay_uv.touched_islands

    def counting(*args: Any) -> set[int]:
        calls.append(1)
        return real(*args)

    monkeypatch.setattr(clay_uv, "touched_islands", counting)

    def frame() -> None:
        ui.new_frame()
        ui.begin("##host")
        try:
            current = doc.by_uid(obj.uid)
            clay_uv._canvas(ctx=None, tab=tab, doc=doc, obj=current, view_state=view_state)
        finally:
            ui.end()
            ui.end_frame()

    for _ in range(3):
        frame()
    assert len(calls) == 1, "touched_islands ran again with the mesh and selection unchanged"

    doc.set_element_sel(obj.uid, el.ElementSel(faces=[]))
    frame()
    assert len(calls) == 2, "a changed selection must be measured again"

