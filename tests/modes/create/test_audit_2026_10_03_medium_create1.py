"""The 2026-10-03 audit's Medium findings on Create's brief, panes and rail.

create-07 (a card dropped on the Mesh Source chip moved the creation),
create-08 (a deleted pose stayed in the sheet form), create-15 (two focus rings
in one column), create-16 (the rail lost the stage the user stood on) and
create-17 (the pose-save warning promised a variant it does not make).
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

import pytest
from imgui_bundle import imgui

from realmspinner.studio import layout
from realmspinner.studio.modes.create.engine import workspace as families
from realmspinner.studio.modes.create.ui import brief as create_brief
from realmspinner.studio.modes.create.ui import rail as create_rail
from realmspinner.studio.modes.create.ui import session as create_session
from realmspinner.studio.state import AppState


@pytest.fixture
def frames():
    """A bare imgui context (see ``test_create_brief.frames``: at most one may
    exist at a time, so the file builds and destroys its own)."""
    from realmspinner.studio import theme

    previous = imgui.get_current_context()
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (1600, 950)
    io.delta_time = 1 / 60
    io.fonts.add_font_default()
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures.value
    theme.apply(imgui)

    def draw(build: Any, size: tuple[float, float] = (1200.0, 900.0)) -> None:
        imgui.new_frame()
        imgui.set_next_window_size(size)
        imgui.begin("smoke")
        try:
            build()
        finally:
            imgui.end()
            imgui.end_frame()
            imgui.render()

    yield draw
    imgui.destroy_context(ctx)
    if previous is not None:
        imgui.set_current_context(previous)


def _ref(job_id: str, workspace: str, kind: str = "text") -> dict[str, Any]:
    return {
        "id": job_id,
        "kind": kind,
        "stage": "reference",
        "status": "done",
        "files": ["input.png"],
        "name": job_id,
        "created_at": 1,
        "params": {families.WORKSPACE_PARAM: workspace},
    }


def _ctx(jobs: list[dict[str, Any]], *, stage: str) -> Any:
    state = AppState(mode="create")
    state.create.stage = stage
    by_id = {j["id"]: j for j in jobs}
    toasts: list[str] = []
    return SimpleNamespace(
        state=state,
        svc=SimpleNamespace(config=None),
        busy=lambda _key: False,
        confirms=SimpleNamespace(ask=lambda _c: None),
        cache=SimpleNamespace(get=by_id.get, jobs=list(jobs), active=None),
        job=lambda: by_id.get(state.selected),
        textures=None,
        model_rows=[],
        submit=lambda *a, **k: True,
        toast=lambda text, *a, **k: toasts.append(text),
        toasts=toasts,
    )


# --- create-07 ---------------------------------------------------------------


def test_dropping_a_card_on_the_source_chip_keeps_the_current_creation_and_mesh_settings(
    frames, monkeypatch
):
    """The chip's drop went through ``library.select``, so the next
    ``session.sync`` saw a selection in another creation, resumed *that*
    creation and replaced the tuned Mesh form with its draft or the defaults."""
    ctx = _ctx([_ref("ref-a", "A"), _ref("ref-b", "B", kind="image")], stage="mesh")
    state = ctx.state
    state.create.workspace = "creation:A"
    state.create.workspace_selection = "ref-a"
    state.select("ref-a")
    state.source_job = "ref-a"
    state.form_3d["size_m"] = 2.5
    state.form_3d["mesh_seed"] = 77
    create_session.sync(ctx)

    # The drag is in flight and the pointer is released over the chip.
    state.dragging_job = "ref-b"
    monkeypatch.setattr(imgui, "begin_drag_drop_target", lambda: True)
    monkeypatch.setattr(imgui, "accept_drag_drop_payload_py_id", lambda *_a, **_k: object())
    monkeypatch.setattr(imgui, "end_drag_drop_target", lambda: None)

    frames(lambda: create_brief._source_chip(ctx, 300.0))
    create_session.sync(ctx)

    assert state.source_job == "ref-b", "the drop must still name the new source"
    assert state.create.workspace == "creation:A", "the drop moved the creation"
    assert state.selected == "ref-a"
    assert state.form_3d["size_m"] == 2.5
    assert state.form_3d["mesh_seed"] == 77


# --- create-15 ---------------------------------------------------------------


def test_one_tab_press_moves_one_focus_cursor_in_the_create_column(frames, monkeypatch):
    """The brief controls and the settings controls were two rings pumped in
    one frame, so one Tab moved both cursors and the single ``focus_moved``
    flag went to whichever focused control was drawn first."""
    from realmspinner.studio.modes.create.ui.panes import settings_3d

    monkeypatch.setattr(settings_3d, "_footer", lambda *_a, **_k: None)

    def settings_control(ctx: Any, *_a: Any, **_k: Any) -> None:
        from realmspinner.studio import focus

        with focus.item(ctx.state, settings_3d.FOCUS_PANE, "platform"):
            imgui.button("platform")

    monkeypatch.setattr(settings_3d, "_draw_form", settings_control)
    ctx = _ctx([_ref("ref-a", "A")], stage="mesh")
    ctx.state.source_job = "ref-a"

    pressed = {"tab": False}
    real_pressed = imgui.is_key_pressed
    monkeypatch.setattr(
        imgui,
        "is_key_pressed",
        lambda key, *a, **k: pressed["tab"] if key == imgui.Key.tab else real_pressed(key, *a, **k),
    )

    def build() -> None:
        layout.begin_frame()
        settings_3d.draw(ctx)

    frames(build)
    rings = {pane: order for pane, order in ctx.state.focus_order.items() if order}
    assert len(rings) == 1, f"the column recorded {sorted(rings)} as separate rings"
    (order,) = rings.values()
    assert "source" in order and "platform" in order and order[-1] == "generate"

    pressed["tab"] = True
    frames(build)
    assert len(ctx.state.focus_key) == 1, (
        f"one Tab moved {sorted(ctx.state.focus_key)}; it must move exactly one cursor"
    )


def test_both_settings_columns_share_the_brief_ring():
    from realmspinner.studio.modes.create.ui.panes import settings_2d, settings_3d

    assert settings_2d.FOCUS_PANE == create_brief.FOCUS_PANE
    assert settings_3d.FOCUS_PANE == create_brief.FOCUS_PANE


# --- create-16 ---------------------------------------------------------------


def test_the_rail_always_contains_the_stage_the_user_is_standing_on(monkeypatch):
    """Home's New 3D model, a file drop and Library's Make 3D ``go`` to Mesh
    without touching an Image form, whose journey has no Mesh segment -- the
    rail was drawn without the current stage and parked its pill under
    Reference."""
    from realmspinner.studio.shell import frame

    seen: dict[str, Any] = {}

    def fake_rail(_rail_id, items, current, **_k):
        seen["keys"] = [item[0] for item in items]
        seen["current"] = current
        return current

    monkeypatch.setattr(create_rail, "stage_rail", fake_rail)
    ctx = _ctx([], stage="mesh")
    ctx.state.form_2d["asset_type"] = "image"
    ctx.rigging_available = True

    frame.FrameMixin._stage_rail(SimpleNamespace(), ctx, max_width=300.0)

    assert seen["current"] == "mesh"
    assert "mesh" in seen["keys"], seen["keys"]
    # Still the canonical order, so the pill sits where the stage belongs.
    assert seen["keys"] == [k for k in families.STAGE_ORDER if k in seen["keys"]]


# --- create-08 ---------------------------------------------------------------


def test_a_deleted_pose_is_not_submitted_by_the_sheet_form(monkeypatch):
    """``form["poses"]`` outlived the pose it named: the Rows list is drawn from
    the surviving poses so the dead id could not be unticked, and every Render
    sheet press was refused with a fieldless "no such pose"."""
    from realmspinner.studio import widgets
    from realmspinner.studio.panes import sheet_panel

    submitted: dict[str, Any] = {}
    ctx = SimpleNamespace(
        state=SimpleNamespace(
            preview={"poses": [{"id": "p1", "name": "idle"}, {"id": "p2", "name": "wave"}]},
            clear_field_errors=lambda: None,
        ),
        sheet_options=None,
        busy=lambda _key: False,
        cache=SimpleNamespace(jobs=[]),
        viewer=None,
        submit=lambda key, fn, *a, **k: submitted.update(k),
        svc=None,
    )
    job = {"id": "m1", "status": "done", "files": ["model.glb", "rig.glb"]}

    form = sheet_panel._form(ctx, "m1")
    form["poses"].update({"p1", "p2"})
    form["clip"] = True
    form["clip_from"], form["clip_to"] = "p1", "p2"

    # p2 is deleted; the panel's own listing no longer carries it.
    ctx.state.preview["poses"] = [{"id": "p1", "name": "idle"}]
    monkeypatch.setattr(widgets, "disabled_button", lambda *_a, **_k: True)
    monkeypatch.setattr(widgets, "muted", lambda *_a, **_k: None)
    monkeypatch.setattr(widgets, "text_colored", lambda *_a, **_k: None)

    form = sheet_panel._form(ctx, "m1")
    sheet_panel._submit(ctx, job, form)

    assert submitted["poses"] == ["p1"], submitted
    assert submitted["clip_to"] != "p2"


# --- create-17 ---------------------------------------------------------------


def test_the_pose_save_warning_does_not_claim_a_different_name_makes_a_new_pose():
    """While a saved pose is loaded the save replaces it whatever name is typed
    (``payload["id"]`` is the loaded id), so the sentence above the button must
    not promise that a new name is safe."""
    from realmspinner.studio import dialogs
    from realmspinner.studio.panes import pose_panel

    assert "same name" not in inspect.getsource(pose_panel)

    asked: list[Any] = []
    saved: dict[str, Any] = {}
    ctx = SimpleNamespace(
        svc=None,
        prompts=SimpleNamespace(ask=asked.append),
        submit=lambda key, fn, svc, job_id, payload: saved.update(payload),
    )
    viewer = SimpleNamespace(
        editor=SimpleNamespace(current="idle", root_translation=lambda: [0, 0, 0]),
        get_pose=lambda: {},
    )
    pose_panel._save(ctx, {"id": "m1"}, viewer)
    (prompt,) = asked
    assert isinstance(prompt, dialogs.Prompt)
    prompt.on_accept("idle low")
    assert saved["id"] == "idle", "the loaded pose is replaced whatever the name"
    assert "whatever name" in pose_panel.SAVE_REPLACES_WARNING
