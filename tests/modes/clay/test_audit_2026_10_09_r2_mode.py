"""The 2026-10-09 second audit's Clay mode-root findings: clay-60, 77, 115, 117,
118, 119, 120, 121 and 122 (the ``mode.py`` / ``doc_io.py`` halves)."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pygame
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import selection
from realmspinner.studio import dialogs
from realmspinner.studio.modes.clay import doc_io
from realmspinner.studio.modes.clay import mode as clay_mode

from .test_clay_mode import FakeCtx
from .test_clay_palette import _ctx

_CLAY = Path(clay_mode.__file__).parent


@pytest.fixture(autouse=True)
def _no_pygame_display(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0)


def _dragging_ctx() -> Any:
    ctx = _ctx(with_view=True)
    ctx.clay_view.camera = None  # the palette fixture's camera has no theta/phi
    ctx.clay_view.dragging = True
    ctx.clay_view.grabbing = False
    return ctx


# --- clay-60 -------------------------------------------------------------------


def test_save_waits_for_a_live_drag_and_never_writes_half_dragged_transforms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The menu, the palette row and the Model-file buttons call ``save`` /
    ``save_as`` directly; only Ctrl+S was drag-gated, so a save mid-G wrote the
    half-dragged transform and then marked the tab clean after Esc."""
    ctx = _dragging_ctx()
    tab = ctx.state.clay.active
    tab.path = tmp_path / "probe.rblk"
    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: tmp_path / "other.rblk")
    obj = tab.doc.objects[0]
    obj.translation[:] = (5.0, 0.0, 0.0)  # the drag writes TRS in place, no history step

    clay_mode.save(ctx, tab)
    clay_mode.save_to(ctx, tab, tab.path)
    clay_mode.save_as(ctx, tab)

    assert ctx.submitted == [], "a save was submitted under a live drag"
    assert not tab.saving
    assert not tab.path.exists() and not (tmp_path / "other.rblk").exists()
    assert ctx.toasts, "the refusal said nothing"

    obj.translation[:] = (0.0, 0.0, 0.0)  # Esc
    ctx.clay_view.dragging = False
    assert not tab.dirty, "a refused save must not have touched dirty"
    clay_mode.save(ctx, tab)
    assert ctx.submitted, "the save never came back once the drag ended"


# --- clay-77 -------------------------------------------------------------------


class _ThumbApp:
    def __init__(self, ctx: Any) -> None:
        self.app_ctx = ctx
        self.clay_view = "clay-viewport"
        self.captured: list[tuple[str, Any]] = []

    def _capture_clay_thumbnail(self, job_id: str) -> None:
        self.captured.append((job_id, self.clay_view))

    def _capture_thumbnail_from(self, job_id: str, view: Any) -> None:
        self.captured.append((job_id, view))


def test_export_thumbnail_is_not_captured_when_another_tab_is_drawn() -> None:
    """The thumbnail is the last-drawn framebuffer when the task lands, so a tab
    switch while the export ran made the card a picture of the wrong document."""
    from realmspinner.studio import main as main_mod

    ctx = _ctx(with_view=True)
    exported = ctx.state.clay.active
    done = SimpleNamespace(
        key=f"clay-export:{exported.uid}", result={"job_id": "j1", "exported": True}
    )

    app = _ThumbApp(ctx)
    ctx.state.clay.camera_tab = "bd-elsewhere"  # the viewport is drawing another tab
    main_mod.App._on_task_done(app, done)
    assert app.captured == [], f"captured another tab's picture: {app.captured}"

    ctx.state.clay.camera_tab = exported.uid
    main_mod.App._on_task_done(app, done)
    assert app.captured == [("j1", "clay-viewport")]


# --- clay-115 ------------------------------------------------------------------


def _textured_doc(textured: bool) -> bd.ClayDoc:
    doc = bd.ClayDoc(materials=[bd.default_material("A"), bd.default_material("B")])
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box(), generator="box"))
    if textured:
        doc.add_texture(0, 32)
    return doc


def _export_obj(doc: bd.ClayDoc, target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = FakeCtx()
    tab = clay_mode.adopt(ctx, doc, title="barrel")
    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: target)
    clay_mode.export_mesh_file(ctx, tab, "obj")


def test_re_exporting_an_obj_without_a_texture_leaves_no_stale_png_for_that_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "barrel.obj"
    _export_obj(_textured_doc(True), target, monkeypatch)
    assert (tmp_path / "barrel_0.png").is_file()
    (tmp_path / "barrel_9.png").write_bytes(b"MY OWN ARTWORK")  # not named by the .mtl

    _export_obj(_textured_doc(False), target, monkeypatch)

    assert "map_Kd" not in (tmp_path / "barrel.mtl").read_text(encoding="utf-8")
    assert not (tmp_path / "barrel_0.png").exists(), "the old texture was left beside the new set"
    assert (tmp_path / "barrel_9.png").read_bytes() == b"MY OWN ARTWORK"


def test_re_exporting_an_obj_deletes_nothing_when_there_was_no_clay_mtl_to_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "barrel_0.png").write_bytes(b"MY OWN ARTWORK")
    _export_obj(_textured_doc(False), tmp_path / "barrel.obj", monkeypatch)
    assert (tmp_path / "barrel_0.png").read_bytes() == b"MY OWN ARTWORK"


def test_re_exporting_an_obj_keeps_a_texture_the_new_set_still_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "barrel.obj"
    _export_obj(_textured_doc(True), target, monkeypatch)
    _export_obj(_textured_doc(True), target, monkeypatch)
    assert (tmp_path / "barrel_0.png").read_bytes()[:4] == b"\x89PNG"


# --- clay-117 ------------------------------------------------------------------


def _bevel_ctx() -> tuple[Any, list[str]]:
    ctx = _ctx(with_view=True)
    calls: list[str] = []
    ctx.clay_view = SimpleNamespace(
        camera=None,
        dragging=False,
        grabbing=False,
        begin_op_drag=lambda doc, op, params: calls.append(op.name) or True,
    )
    doc = ctx.state.clay.active.doc
    doc.select([doc.objects[0].uid])
    doc.set_element_mode("edge")
    selection.select_all(doc)
    return ctx, calls


def test_ctrl_b_is_refused_while_the_uv_live_transform_is_armed() -> None:
    ctx, calls = _bevel_ctx()
    event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_b, mod=pygame.KMOD_CTRL)

    clay_mode.handle_key(ctx, event)
    assert calls == ["bevel"], "Ctrl+B is the registry chord this test drives"

    calls.clear()
    ctx.state.clay.active.uv_view.drag_mode = "rotate"
    clay_mode.handle_key(ctx, event)
    assert calls == [], "Ctrl+B started a bevel drag under the UV pane's armed rotate"

    ctx.state.clay.active.uv_view.drag_mode = ""
    ctx.clay_view.dragging = True
    clay_mode.handle_key(ctx, event)
    assert calls == [], "Ctrl+B started a bevel drag under a 3-D drag"


# --- clay-118 ------------------------------------------------------------------


def test_sibling_mtl_finds_a_library_whose_name_contains_a_hash(tmp_path: Path) -> None:
    (tmp_path / "Level#2.mtl").write_text("newmtl red\nKd 1 0 0\n", encoding="utf-8")
    obj = tmp_path / "Level#2.obj"
    data = b"mtllib Level#2.mtl\nv 0 0 0\n"
    obj.write_bytes(data)
    assert (clay_mode._sibling_mtl(obj, data) or "").splitlines() == ["newmtl red", "Kd 1 0 0"]

    # The clay-42 comment cut still holds, glued or spaced.
    (tmp_path / "m.mtl").write_text("newmtl blue\n", encoding="utf-8")
    for line in (b"mtllib m.mtl # the colours\n", b"mtllib m.mtl\t# x\n"):
        assert (clay_mode._sibling_mtl(tmp_path / "m.obj", line) or "").startswith("newmtl blue")


# --- clay-119 ------------------------------------------------------------------


def test_export_asset_takes_its_camera_on_the_frame_thread_not_the_live_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``sync_active_camera`` rewrites ``tab.view`` every frame; the export task
    ran later on another thread and read whatever the camera had become."""
    ctx = _ctx(with_view=True)
    tab = ctx.state.clay.active
    tab.view.yaw = 1.0
    monkeypatch.setattr(clay_mode, "camera_of", lambda c, t: t.view)
    seen: list[Any] = []
    monkeypatch.setattr(
        clay_mode, "build_asset", lambda svc, doc, **kw: seen.append(kw["view"]) or "job"
    )
    pending: list[Any] = []
    ctx.submit = lambda key, run, *a, **k: pending.append(run) or True

    clay_mode.export_asset(ctx, tab)
    tab.view.yaw = 2.5  # the next frame's sync, before the task thread runs
    pending[0]()

    assert seen and seen[0] is not tab.view, "the task was handed the live camera object"
    assert seen[0].yaw == 1.0, "the exported camera is not the one at the press"


# --- clay-120 ------------------------------------------------------------------


def test_output_buttons_say_why_they_are_greyed_during_a_live_drag() -> None:
    doc = _textured_doc(False)
    assert doc_io.outputs_why(doc, False) == "", "the old call shape must keep working"
    why = doc_io.outputs_why(doc, False, gesture=True)
    assert why and "drag" in why.lower()
    assert doc_io.outputs_why(doc, True, gesture=True) == "Saving..."
    for obj in doc.objects:
        obj.visible = False
    assert doc_io.outputs_why(doc, False, gesture=True) == why, "the drag outranks 'hidden'"

    from realmspinner.studio import palette

    assert why == palette._CLAY_GESTURE_WHY, "the palette rows and the buttons say one sentence"


# --- clay-121 ------------------------------------------------------------------


def test_fire_op_has_no_stray_string_statement_after_it() -> None:
    tree = ast.parse((_CLAY / "mode.py").read_text(encoding="utf-8"))
    body = tree.body
    index = next(
        i for i, n in enumerate(body) if isinstance(n, ast.FunctionDef) and n.name == "fire_op"
    )
    after = body[index + 1]
    assert not (
        isinstance(after, ast.Expr)
        and isinstance(after.value, ast.Constant)
        and isinstance(after.value.value, str)
    ), "a bare string statement follows fire_op"
    assert "no window of their own" in (ast.get_docstring(body[index]) or "")


# --- clay-122 ------------------------------------------------------------------


def test_no_clay_mode_root_docstring_names_a_deleted_pane_or_tab() -> None:
    stale = (
        "Document tab",
        "Properties pane",
        "the bridge's",
        "tools pane",
        "tools-pane",
    )
    for name in ("mode.py", "doc_io.py"):
        text = (_CLAY / name).read_text(encoding="utf-8")
        for needle in stale:
            assert needle not in text, f"{name} still says {needle!r}"
