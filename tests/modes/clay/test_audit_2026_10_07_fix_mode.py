"""The 2026-10-07 audit's Clay mode / tab-state / Properties findings:
clay-09, 12 (the header-door half), 13 (the ``build_asset`` ordering half), 18
(the Properties memo half), 23 (the ``euler_cache`` half), 46, 47, 49, 50, 64
and 82."""

from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pygame
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio import dialogs
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay import state as clay_state
from realmspinner.studio.modes.clay import transform_edit
from realmspinner.studio.modes.clay.ui.panes import props

from .test_audit_2026_10_07_mode import _armed_uv_tab
from .test_clay_mode import FakeCtx


def _key(key: int, mod: int = 0) -> Any:
    return pygame.event.Event(pygame.KEYDOWN, key=key, mod=mod)


def _box_tab(ctx: FakeCtx, title: str = "Scene") -> tuple[Any, bd.ClayDoc, Any]:
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box(), generator="box"))
    tab = clay_mode.adopt(ctx, doc, title=title)
    return tab, doc, obj


class _Done:
    def __init__(self, key: str, result: Any = None) -> None:
        self.key = key
        self.result = result


# --- clay-09 -------------------------------------------------------------------


def test_the_document_tab_names_the_last_file_an_obj_or_glb_export_wrote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = FakeCtx()
    tab, _doc, _obj = _box_tab(ctx)
    assert props.export_lines(tab) == [], "nothing exported yet, nothing to name"

    out = tmp_path / "crate.glb"
    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: out)
    clay_mode.export_mesh_file(ctx, tab, "glb")
    clay_mode.on_task_done(ctx, _Done(ctx.submitted[-1], ctx.result))

    assert any(str(out) in line for line in props.export_lines(tab))

    # A library export keeps its own line; the two coexist.
    tab.job_id = "abc123"
    lines = props.export_lines(tab)
    assert any("abc123" in line for line in lines)
    assert any(str(out) in line for line in lines)


# --- clay-12 (the header Undo/Redo and history-jump half) ----------------------


def test_the_header_undo_is_refused_while_a_keyboard_drag_is_live() -> None:
    ctx = FakeCtx()
    tab, doc, obj = _box_tab(ctx)
    doc.set_transform(obj.uid, translation=[1.0, 0.0, 0.0])
    doc.set_transform(obj.uid, translation=[2.0, 0.0, 0.0])
    ctx.clay_view = SimpleNamespace(dragging=True)
    head = doc.history.head

    clay_mode.undo(ctx, tab)
    assert doc.history.head == head, "Undo ran under a live G/R/S drag"
    clay_mode.redo(ctx, tab)
    assert doc.history.head == head
    assert clay_mode.step_history(ctx, tab, 0) is False
    assert doc.history.head == head

    ctx.clay_view.dragging = False
    clay_mode.undo(ctx, tab)
    assert doc.history.head != head, "the button works again once the drag ends"


# --- clay-13 (the build_asset ordering half) -------------------------------------


def test_a_refused_snapshot_leaves_no_library_row(monkeypatch: pytest.MonkeyPatch) -> None:
    from realmspinner.kernels.mesh import serialize
    from realmspinner.service import files as svc_files
    from realmspinner.service import jobs as svc_jobs

    minted: list[str] = []
    monkeypatch.setattr(
        svc_jobs, "import_mesh", lambda *a, **k: minted.append("row") or {"id": "j1"}
    )
    monkeypatch.setattr(svc_files, "save_clay_source", lambda *a, **k: None)

    def refuse(_snap: Any) -> bytes:
        raise ValueError("this clay document places 4,097 objects")

    monkeypatch.setattr(serialize, "snapshot_bytes", refuse)
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))

    with pytest.raises(ValueError, match="4,097"):
        clay_mode.build_asset(object(), doc, title="T")

    assert minted == [], "import_mesh minted a Library row before the snapshot was refused"


# --- clay-18 (the Properties memo half) ------------------------------------------


def _face_doc() -> tuple[bd.ClayDoc, Any]:
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box(), generator="box"))
    doc.select([obj.uid])
    doc.set_element_mode("face")
    doc.set_element_sel(obj.uid, el.ElementSel(faces=[0, 1]))
    return doc, obj


def test_the_properties_element_measurement_does_not_retriangulate_the_mesh_each_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc, obj = _face_doc()
    areas = {"n": 0}
    real_area = props.measure.face_area

    def counting(*args: Any, **kwargs: Any) -> float:
        areas["n"] += 1
        return real_area(*args, **kwargs)

    monkeypatch.setattr(props.measure, "face_area", counting)
    medians = {"n": 0}
    real_median = props.element_move.element_median_world

    def counting_median(d: Any) -> Any:
        medians["n"] += 1
        return real_median(d)

    monkeypatch.setattr(props.element_move, "element_median_world", counting_median)

    first = props.element_measure_text(doc)
    props.element_measure_text(doc)
    props.element_measure_text(doc)
    assert first is not None
    assert areas["n"] == 1, "the face area was recomputed every frame"
    m1 = props.element_median_cached(doc)
    m2 = props.element_median_cached(doc)
    assert m1 is not None and m2 is m1
    assert medians["n"] == 1

    # A changed selection bumps ``doc.rev`` and so the key: the readout follows it.
    doc.set_element_sel(obj.uid, el.ElementSel(faces=[0]))
    second = props.element_measure_text(doc)
    assert areas["n"] == 2
    assert second != first
    props.element_median_cached(doc)
    assert medians["n"] == 2


def test_the_readout_memo_key_is_the_documents_revision_and_element_mode() -> None:
    doc, obj = _face_doc()
    key = props.readout_key(doc)
    assert key == props.readout_key(doc)
    doc.set_element_mode("edge")
    assert props.readout_key(doc) != key
    assert props.readout_key(doc, "cm") != props.readout_key(doc, "m")
    assert props.readout_key(SimpleNamespace(element_mode="face")) is None, "no rev, no memo"


# --- clay-23 (the euler_cache half) ----------------------------------------------


def test_the_euler_cache_of_one_tab_does_not_answer_for_the_same_uid_in_another() -> None:
    state = clay_state.ClayState()
    a = clay_state.ClayTab(doc=bd.ClayDoc(), title="A")
    b = clay_state.ClayTab(doc=bd.ClayDoc(), title="B")
    state.add(a)
    state.add(b)

    state.activate(a.uid)
    key_a = props.euler_key(state, 1)
    state.activate(b.uid)
    key_b = props.euler_key(state, 1)
    assert key_a != key_b, "two tabs share an object uid but not a cache entry"

    quat = (0.0, 0.0, 0.0, 1.0)
    transform_edit.remember_euler(state.euler_cache, key_a, quat, (190.0, 0.0, 0.0))
    shown = transform_edit.displayed_euler(state.euler_cache, key_b, quat)
    assert shown == (0.0, 0.0, 0.0), "tab A's typed 190 followed the uid into tab B"


# --- clay-46 ---------------------------------------------------------------------


def test_alt_plus_a_bare_binding_letter_does_nothing() -> None:
    ctx = FakeCtx()
    tab, doc, obj = _box_tab(ctx)
    state = clay_mode.ensure(ctx)
    doc.select([obj.uid])
    doc.set_element_mode("face")
    doc.set_element_sel(obj.uid, el.ElementSel(faces=[0]))
    steps = len(doc.history)

    assert clay_mode.handle_key(ctx, _key(pygame.K_e, pygame.KMOD_ALT)) is True
    assert len(doc.history) == steps, "Alt+E extruded"
    assert clay_mode.handle_key(ctx, _key(pygame.K_g, pygame.KMOD_ALT)) is True
    assert state.tool == "select", "Alt+G set the Move tool"
    assert clay_mode.handle_key(ctx, _key(pygame.K_1, pygame.KMOD_ALT)) is True
    assert doc.element_mode == "face", "Alt+1 set vertex mode"

    # The two documented Alt chords still work.
    assert state.xray is False
    clay_mode.handle_key(ctx, _key(pygame.K_z, pygame.KMOD_ALT))
    assert state.xray is True
    doc.set_element_mode("object")
    doc.select([obj.uid])
    clay_mode.handle_key(ctx, _key(pygame.K_h, pygame.KMOD_ALT))
    clay_mode.handle_key(ctx, _key(pygame.K_h))
    assert doc.by_uid(obj.uid).visible is False
    clay_mode.handle_key(ctx, _key(pygame.K_h, pygame.KMOD_ALT))
    assert doc.by_uid(obj.uid).visible is True
    del tab


# --- clay-47 ---------------------------------------------------------------------


def test_ctrl_z_waits_for_an_armed_uv_live_transform_to_commit() -> None:
    ctx = FakeCtx()
    tab, doc, _obj, _before = _armed_uv_tab(ctx)
    head = doc.history.head

    for chord in (pygame.K_z, pygame.K_y):
        assert clay_mode.handle_key(ctx, _key(chord, pygame.KMOD_CTRL)) is True
    assert doc.history.head == head, "Ctrl+Z/Y moved history under an armed UV gesture"
    assert tab.uv_view.drag_mode == "rotate"
    # The header's button is the same door.
    clay_mode.undo(ctx, tab)
    assert doc.history.head == head

    # Once the gesture commits, the chord is the history's again.
    clay_mode.handle_key(ctx, _key(pygame.K_RETURN))
    assert tab.uv_view.drag_mode == ""
    committed = doc.history.head
    clay_mode.handle_key(ctx, _key(pygame.K_z, pygame.KMOD_CTRL))
    assert doc.history.head != committed


# --- clay-49 ---------------------------------------------------------------------


def test_save_screenshot_encodes_the_png_on_the_task_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PIL import Image

    from realmspinner.studio.viewer import capture

    saved_on: list[str] = []
    real_save = Image.Image.save

    def tracing_save(self: Any, fp: Any, *args: Any, **kwargs: Any) -> Any:
        saved_on.append(threading.current_thread().name)
        return real_save(self, fp, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", tracing_save)
    monkeypatch.setattr(
        capture,
        "image",
        lambda viewport: Image.frombytes("RGBA", (4, 4), bytes([9, 8, 7, 255]) * 16),
    )

    class _View:
        def render_png(self, doc: Any, **kwargs: Any) -> Any:
            # What the real ``ClayView.render_png`` ends in.
            return capture.png_bytes(object())

    ctx = FakeCtx()
    ctx.clay_view = _View()
    held: list[Any] = []
    ctx.submit = lambda key, run, *a, **k: held.append(run) or True  # type: ignore[method-assign]
    tab, _doc, _obj = _box_tab(ctx)
    out = tmp_path / "shot"
    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: out)

    clay_mode.save_screenshot(ctx, tab)
    assert saved_on == [], "the PNG was encoded before the task was even submitted"

    worker = threading.Thread(target=lambda: held[0](), name="task-thread")
    worker.start()
    worker.join()
    assert saved_on == ["task-thread"]
    assert (tmp_path / "shot.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    # The seam is put back: nothing else that captures sees the raw image.
    assert capture.png_bytes.__module__ == capture.__name__


# --- clay-50 ---------------------------------------------------------------------


def test_the_slow_import_prompt_counts_what_the_manual_says_it_counts(tmp_path: Path) -> None:
    quad = tmp_path / "quad.obj"
    quad.write_bytes(b"v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n")
    ctx = FakeCtx()

    clay_mode.import_mesh_path(ctx, quad)

    assert ctx.result["triangles"] == 2, "one quad is two triangles, not one face"

    big = dict(ctx.result, triangles=clay_mode.SLOW_TRIANGLES + 1)
    clay_mode.on_task_done(ctx, _Done("clay-import:1", big))
    assert "triangles" in ctx.confirms.pending.message


# --- clay-64 ---------------------------------------------------------------------


def test_a_material_named_like_a_map_kd_line_does_not_defeat_the_texture_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = FakeCtx()
    doc = bd.ClayDoc(
        materials=[bd.default_material("Plain"), bd.default_material("map_Kd Scene_1.png")]
    )
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
    doc.paint_faces(doc.objects[0].uid, [0, 1], 1)
    doc.materials[1] = replace(
        doc.materials[1], base_color=(2, 2, bytes([10, 20, 30, 255] * 4)), nearest=True
    )
    tab = clay_mode.adopt(ctx, doc, title="Scene")
    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: tmp_path / "renamed.obj")

    clay_mode.export_mesh_file(ctx, tab, "obj")

    lines = (tmp_path / "renamed.mtl").read_text(encoding="utf-8").splitlines()
    assert "map_Kd renamed_1.png" in lines, "the real map_Kd line still names the old file"
    assert "# Clay material name: map_Kd Scene_1.png" in lines, "a comment was rewritten"
    assert (tmp_path / "renamed_1.png").is_file()


# --- clay-82 ---------------------------------------------------------------------


def test_the_material_tab_with_an_empty_palette_says_where_to_add_a_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    said: list[str] = []
    for name in ("muted", "muted_wrapped", "secondary"):
        monkeypatch.setattr(props.widgets, name, lambda text, *a, **k: said.append(text))
    monkeypatch.setattr(props.widgets, "field_label", lambda *a, **k: None)
    ctx = FakeCtx()
    tab, doc, obj = _box_tab(ctx)
    doc.materials.clear()

    props._material(ctx, tab, doc, obj)

    text = " ".join(said)
    assert "palette" in text and "+" in text, text
