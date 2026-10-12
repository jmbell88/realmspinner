"""Slice B2: Create's History pane, labelling a reference from the contact sheet,
and the press id the contact sheet groups by.

Each test's name is the claim. The pane's rows are the service's
(``generation_log.history``), so most tests build a real store and read through
the pane's own task body (``history.load``); the drawing ones use a bare imgui
context.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from imgui_bundle import imgui

from realmspinner import models
from realmspinner.service import generation_log
from realmspinner.service import verdicts as svc_verdicts
from realmspinner.studio import theme
from realmspinner.studio.modes.create.engine import history as engine
from realmspinner.studio.modes.create.engine import workspace as families
from realmspinner.studio.modes.create.ui import canvas
from realmspinner.studio.modes.create.ui.panes import history
from realmspinner.studio.state import DEFAULT_FORM_3D, AppState

PARAMS = {
    "generation_type": "object",
    "base_model": "sdxl_cfg",
    "style_lora": "render3d",
    "lora_weight": 0.8,
    "negative_prompt": "blurry",
    "seed": 1204285292,
    "recipe": {
        "reference": {
            "base_model": "sdxl_cfg",
            "steps": 30,
            "guidance_scale": 7.0,
            "seed": 1204285292,
            "prompt": "3d style, a chest",
            "negative_prompt": "blurry",
            "style_lora": "render3d",
            "lora_weight": 0.8,
        },
        "trellis": {"seed": 7, "band": 4, "tex_res": 1024},
    },
}


@pytest.fixture
def frames():
    previous = imgui.get_current_context()
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (1600, 950)
    io.delta_time = 1 / 60
    io.fonts.add_font_default()
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures.value
    theme.apply(imgui)

    def draw(build: Any, size: tuple[float, float] = (400.0, 900.0)) -> None:
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
    imgui.destroy_context(context)
    if previous is not None:
        imgui.set_current_context(previous)


def _image_job(svc, params=None, prompt="a chest"):
    job_id = svc.store.create(
        "text", prompt, dict(PARAMS if params is None else params), stage="reference", status="done"
    )
    directory = svc.job_dir(job_id)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "input.png").write_bytes(b"x")
    return job_id


class _Ctx:
    """Just enough of AppCtx for History's Load settings and the canvas labels."""

    def __init__(self, svc: Any = None, cached: dict | None = None) -> None:
        self.svc = svc
        self.state = AppState(mode="create")
        self.toasts: list[tuple[str, str]] = []
        self.submitted: list[tuple[str, tuple]] = []
        self.cached = cached or {}
        self.cache = SimpleNamespace(
            get=self.cached.get, invalidate=lambda: None, jobs=[], _generation=1
        )
        self.textures = None

    def toast(self, message: str, kind: str = "info", *a: Any) -> None:
        self.toasts.append((message, kind))

    def submit(self, key: str, fn: Any, *args: Any, **kwargs: Any) -> bool:
        self.submitted.append((key, args))
        return True


@pytest.fixture
def no_stage_switch(monkeypatch):
    """Load settings ends by moving to the Reference stage; that is the stage
    module's own business and needs a shell this test does not have."""
    from realmspinner.studio.modes.create.ui import stages

    monkeypatch.setattr(stages, "go", lambda *a, **k: None)
    from realmspinner.studio.modes.create.ui.panes import settings_2d

    monkeypatch.setattr(settings_2d.create_recipe, "clear_unusable", lambda c, f: [])
    monkeypatch.setattr(settings_2d.create_recipe, "resolved_recipe", lambda c, f: None)


# --- the rows -----------------------------------------------------------------------


def test_history_lists_a_deleted_jobs_row_and_its_settings_line(svc):
    job_id = _image_job(svc)
    svc.store.set_stage(job_id, "model")
    svc_verdicts.record_verdict(svc, job_id, grade=4, note="nearly")
    svc.store.delete(job_id)

    loaded = history.load(svc, dict(engine.NO_FILTERS), 50, True)

    (row,) = loaded["rows"]
    assert row["job_id"] == job_id and row["exists"] is False
    assert engine.gone_text(row) == "deleted"
    assert "30 steps" in row["settings_line"] and "seed 1204285292" in row["settings_line"]
    assert engine.grade_text(row["mesh"]) == "+4"
    assert engine.row_notes(row) == ["nearly"]
    assert ("sdxl_cfg", generation_log.base_label("sdxl_cfg")) in loaded["options"]["base_model"]


def test_history_filter_options_come_from_the_rows_so_a_retired_model_still_filters(svc):
    gone = dict(PARAMS)
    gone["base_model"] = "turbo"
    gone["recipe"] = {"reference": {"base_model": "turbo", "steps": 4}}
    job_id = _image_job(svc, gone)
    svc.store.set_stage(job_id, "model")
    svc_verdicts.record_verdict(svc, job_id, grade=1)
    svc.store.delete(job_id)

    options = engine.options_from(generation_log.history(svc))

    assert ("turbo", "SDXL-Turbo") in options["base_model"]
    filters = dict(engine.NO_FILTERS, base_model="turbo")
    found = history.load(svc, filters, 50, False)["rows"]
    assert [r["job_id"] for r in found] == [job_id]


def test_a_filtered_page_never_narrows_the_options_that_filtered_it():
    old = {"base_model": [("a", "A"), ("b", "B")], "style_lora": [], "generation_type": []}
    new = {"base_model": [("a", "A")], "style_lora": [("s", "S")], "generation_type": []}
    merged = engine.merge_options(old, new)
    assert [k for k, _ in merged["base_model"]] == ["a", "b"]
    assert [k for k, _ in merged["style_lora"]] == ["s"]


def test_more_asks_for_a_page_beyond_and_one_row_extra_says_there_is_more(svc):
    for i in range(4):
        _image_job(svc, prompt=f"chest {i}")
    page = history.load(svc, dict(engine.NO_FILTERS), 3, False)["rows"]
    assert len(page) == 4, "limit + 1: the extra row is how the pane knows to offer More"

    ctx = _Ctx(svc)
    ctx.state.create.history_limit = 3
    ctx.state.create.history_filters = dict(engine.NO_FILTERS)
    history.land(
        ctx,
        SimpleNamespace(
            result={"rows": page, "options": None},
            tag={"filters": dict(engine.NO_FILTERS), "limit": 3},
        ),
    )
    assert len(ctx.state.create.history_rows) == 3 and ctx.state.create.history_more is True


def test_a_load_overtaken_by_a_filter_change_is_dropped_not_shown():
    ctx = _Ctx()
    ctx.state.create.history_filters = dict(engine.NO_FILTERS, style_lora="render3d")
    history.land(
        ctx,
        SimpleNamespace(
            result={"rows": [{"job_id": "stale"}], "options": None},
            tag={"filters": dict(engine.NO_FILTERS), "limit": 50},
        ),
    )
    assert ctx.state.create.history_rows is None


def test_a_verdict_a_finished_job_or_a_delete_asks_history_to_reload():
    from realmspinner.studio.modes.review import mode as review_mode

    ctx = _Ctx()
    ctx.state.create.history_dirty = False
    review_mode.refresh_findings(ctx)
    assert ctx.state.create.history_dirty is True

    ctx.state.create.history_dirty = False
    ctx.cache.jobs = [{"id": "a", "status": "running"}]
    ctx.cache._generation = 1
    ctx.state.create.history_seen = (0, (1, 0, 0))
    history.pump(ctx)  # same signature: a running job's progress ticks the generation
    assert ctx.submitted == [], "progress on a running job is not a reason to reload"

    ctx.state.create.history_seen = (1, (1, 0, 0))
    ctx.cache._generation = 2
    ctx.cache.jobs = [{"id": "a", "status": "done"}]
    ctx.submitted.clear()
    history.pump(ctx)
    assert [k for k, _ in ctx.submitted] == ["create-history"], "a finished job reloads"


def test_the_pane_draws_deleted_and_live_rows_for_real(frames, svc, monkeypatch):
    live = _image_job(svc)
    gone = _image_job(svc, prompt="a lamp")
    svc.store.set_stage(gone, "model")
    svc_verdicts.record_verdict(svc, gone, grade=-3, reasons=["holes"], note="rough")
    svc.store.delete(gone)
    ctx = _Ctx(svc)
    ctx.state.create.history_rows = history.load(svc, dict(engine.NO_FILTERS), 50, True)["rows"]
    ctx.state.create.history_options = engine.options_from(ctx.state.create.history_rows)
    ctx.state.create.history_dirty = False
    ctx.cached[live] = {"id": live}
    frames(lambda: history.draw(ctx))
    frames(lambda: history.draw(ctx))


# --- Load settings ---------------------------------------------------------------------


def _snapshot_row(**over: Any) -> dict[str, Any]:
    snapshot = {
        "v": 1,
        "stage": "reference",
        "prompt_raw": "a brass lantern",
        "generation_type": "object",
        "base_model": "sdxl_cfg",
        "style_lora": "render3d",
        "lora_weight": 0.7,
        "negative_prompt": "blurry, lowres",
        "seed": 4242,
        "trellis": {"seed": 9, "band": 4},
        "profile": "standard",
    }
    snapshot.update(over)
    return {"job_id": "gone1", "exists": False, "trashed": False, "snapshot": snapshot,
            "prompt": "a brass lantern"}


def test_load_settings_from_a_snapshot_fills_the_recipe_column(no_stage_switch):
    ctx = _Ctx()
    assert history.load_settings(ctx, _snapshot_row()) is True

    form = ctx.state.form_2d
    assert form["prompt"] == "a brass lantern"
    assert form["negative_prompt"] == "blurry, lowres"
    assert form["style_lora"] == "render3d" and form["lora_weight"] == 0.7
    assert (form["seed"], form["seed_locked"]) == (4242, True)
    # Not the Automatic pick (it resolved to nothing here), so it is an Advanced choice.
    assert (form["model_mode"], form["model_override"], form["base_model"]) == (
        "advanced",
        "sdxl_cfg",
        "sdxl_cfg",
    )
    mesh = ctx.state.form_3d
    assert (mesh["mesh_seed"], mesh["mesh_seed_locked"], mesh["trellis_band"]) == (9, True, 4)
    assert mesh["profile"] == "standard"
    assert ctx.toasts[-1][1] == "info"


def test_load_settings_keeps_automatic_when_automatic_would_pick_the_same_model(
    no_stage_switch, monkeypatch
):
    from realmspinner.studio.modes.create.ui.panes import settings_2d

    monkeypatch.setattr(
        settings_2d.create_recipe,
        "resolved_recipe",
        lambda c, f: SimpleNamespace(base_model="sdxl_cfg"),
    )
    ctx = _Ctx()
    history.load_settings(ctx, _snapshot_row())
    assert ctx.state.form_2d["model_mode"] == "auto"
    assert ctx.state.form_2d["model_override"] == ""


def test_load_settings_is_refused_for_a_retired_model(no_stage_switch):
    ctx = _Ctx()
    before = dict(ctx.state.form_2d)
    row = _snapshot_row(base_model="turbo")

    assert history.load_settings(ctx, row) is False

    message, kind = ctx.toasts[-1]
    assert kind == "error" and "SDXL-Turbo" in message and "removed" in message
    assert ctx.state.form_2d == before, "nothing was written into the form"
    assert models.retired_choice("base_model", "turbo") is not None


def test_load_settings_is_refused_for_a_retired_style_as_well(no_stage_switch):
    ctx = _Ctx()
    assert history.load_settings(ctx, _snapshot_row(style_lora="pixelklein")) is False
    assert "Pixel art (FLUX.2 klein)" in ctx.toasts[-1][0]


def test_load_settings_is_disabled_for_a_deleted_attempt_with_no_snapshot(frames, monkeypatch):
    row = {"job_id": "old1", "exists": False, "trashed": False, "snapshot": None,
           "prompt": "an old verdict", "settings_line": ""}
    reason = engine.no_settings_reason(row)
    assert "deleted before its settings were kept" in reason
    assert engine.no_settings_reason({**row, "exists": True}) == ""
    assert engine.no_settings_reason({**row, "snapshot": {"v": 1}}) == ""

    ctx = _Ctx(SimpleNamespace(config=SimpleNamespace(bench_dir="nowhere")))
    ctx.state.create.history_rows = [row]
    ctx.state.create.history_dirty = False
    seen: list[tuple[str, bool, str]] = []
    real = history.controls.small_button

    def spy(label, *a, enabled=True, reason="", **k):
        if label.startswith("Load settings"):
            seen.append((label, enabled, reason))
        return real(label, *a, enabled=enabled, reason=reason, **k)

    monkeypatch.setattr(history.controls, "small_button", spy)
    frames(lambda: history.draw(ctx))
    assert seen and seen[0][1] is False and seen[0][2] == reason


def test_load_settings_on_a_live_attempt_uses_the_trays_own_vary(no_stage_switch, monkeypatch):
    from realmspinner.studio.modes.create.ui import workspace

    job = {"id": "live1", "stage": "reference", "params": {}}
    ctx = _Ctx(cached={"live1": job})
    called: list[str] = []
    monkeypatch.setattr(workspace, "_vary", lambda c, j: called.append(j["id"]))
    row = {"job_id": "live1", "exists": True, "trashed": False, "snapshot": {"v": 1}}
    assert history.load_settings(ctx, row) is True
    assert called == ["live1"]


def test_a_mesh_snapshot_is_coerced_like_the_mesh_forms_defaults():
    got = engine.mesh_updates(
        {"trellis": {"seed": "12", "band": 2.0, "size_m": "1.5"}, "profile": "raw",
         "lowpoly_triangles": "junk"},
        DEFAULT_FORM_3D,
    )
    assert got == {"mesh_seed": 12, "trellis_band": 2, "size_m": 1.5, "profile": "raw"}


# --- labelling a reference ----------------------------------------------------------------


class _Keys:
    """An ``imgui`` double: one named key is pressed, no modifier held."""

    Key = imgui.Key

    def __init__(self, pressed: Any, **mods: bool) -> None:
        self.pressed = pressed
        self.mods = {"key_ctrl": False, "key_alt": False, "key_shift": False, "key_super": False}
        self.mods.update(mods)

    def is_key_pressed(self, key: Any, repeat: bool = True) -> bool:
        return key == self.pressed

    def get_io(self) -> Any:
        return SimpleNamespace(**self.mods)


def _label_ctx(svc):
    job_id = _image_job(svc)
    job = {"id": job_id, "stage": "reference", "status": "done", "files": ["input.png"]}
    ctx = _Ctx(svc, {job_id: job})
    return ctx, job


def test_g_on_the_sheet_files_a_good_reference_label_visible_in_latest_verdicts(svc, monkeypatch):
    ctx, job = _label_ctx(svc)
    monkeypatch.setattr(canvas, "imgui", _Keys(imgui.Key.g))

    assert canvas.handle_label_keys(ctx, job) is True

    (row,) = svc.store.latest_verdicts()
    assert (row["job_id"], row["stage"], row["verdict"], row["source"]) == (
        job["id"], "reference", "accept", "human",
    )
    assert row["grade"] is None, "an image label is binary"
    assert canvas.label_of(ctx, job["id"])["verdict"] == "accept"
    assert ctx.state.findings_dirty is True and ctx.state.create.history_dirty is True


def test_b_files_bad_and_pressing_it_again_writes_nothing_new(svc, monkeypatch):
    ctx, job = _label_ctx(svc)
    monkeypatch.setattr(canvas, "imgui", _Keys(imgui.Key.b))
    canvas.handle_label_keys(ctx, job)
    canvas.handle_label_keys(ctx, job)

    rows = svc.store.log_rows()[0]
    assert [r["verdict"] for r in rows] == ["reject"], "a held key must not fill the log"


def test_a_label_chord_with_ctrl_is_left_to_the_apps_shortcuts(svc, monkeypatch):
    ctx, job = _label_ctx(svc)
    monkeypatch.setattr(canvas, "imgui", _Keys(imgui.Key.g, key_ctrl=True))
    assert canvas.handle_label_keys(ctx, job) is False
    assert svc.store.latest_verdicts() == []


def test_the_note_popup_saves_tags_and_a_note_on_the_label_it_keeps(svc, monkeypatch):
    ctx, job = _label_ctx(svc)
    canvas.set_label(ctx, job["id"], "accept")

    monkeypatch.setattr(canvas, "imgui", _Keys(imgui.Key.n))
    assert canvas.handle_label_keys(ctx, job) is True
    state = ctx.state.create
    assert state.label_popup_job == job["id"] and state.label_verdict == "accept"

    assert canvas.set_label(
        ctx, job["id"], state.label_verdict, tags=["clean-shape"], note="crisp edges"
    )
    row = svc.store.latest_verdicts()[0]
    assert row["verdict"] == "accept"
    assert row["reasons"] == ["clean-shape"] and row["note"] == "crisp edges"
    # A later Bad keeps the tags and the note on file rather than dropping them.
    canvas.set_label(ctx, job["id"], "reject")
    row = svc.store.latest_verdicts()[0]
    assert (row["verdict"], row["reasons"], row["note"]) == (
        "reject",
        ["clean-shape"],
        "crisp edges",
    )


def test_a_note_with_no_label_yet_is_refused_with_a_reason(svc):
    ctx, job = _label_ctx(svc)
    assert canvas.set_label(ctx, job["id"], None, note="hmm") is False
    assert ctx.toasts[-1] == ("Choose Good or Bad first.", "error")
    assert svc.store.latest_verdicts() == []


def test_the_labelling_keys_collide_with_no_other_create_key():
    """G, B and N are free: F, W and S are the viewer's, Alt+1-5 the rail's, and
    the shell reads no bare G, B or N."""
    import inspect

    from realmspinner.studio import shortcuts
    from realmspinner.studio.shell import events

    source = inspect.getsource(events)
    assert "K_g" not in source and "K_b)" not in source and "K_n" not in source
    rows = {k: d for _title, table in shortcuts.shortcut_sections() for k, d in table}
    assert "G / B" in rows and "N" in rows


def test_a_labelled_sheet_and_the_single_view_draw_for_real(frames, svc, monkeypatch):
    ctx, job = _label_ctx(svc)
    canvas.set_label(ctx, job["id"], "reject", tags=["holes"], note="hole in the roof")
    monkeypatch.setattr(canvas.dialogs, "modal_open", lambda c: False)
    ctx.job = lambda: job
    ctx.state.create.stage = "reference"
    ctx.state.create.label_popup_job = job["id"]
    ctx.state.create.label_popup_ask = True
    frames(lambda: canvas.label_row(ctx, job))
    frames(lambda: canvas.label_popup(ctx))
    frames(lambda: canvas.grid_cell(ctx, job, (300.0, 300.0), selected=True, index=0))


# --- the contact sheet's grouping --------------------------------------------------------------


def _rows(specs):
    return [
        {"id": i, "stage": "reference", "kind": "text", "status": "done", "prompt": p,
         "created_at": t, "params": params, "files": ["input.png"]}
        for i, p, t, params in specs
    ]


def test_a_press_id_groups_a_batch_even_when_the_rows_were_created_far_apart():
    rows = _rows(
        [
            ("c", "a chest", 130.0, {"press_id": "p2"}),
            ("b", "a chest", 100.0, {"press_id": "p2"}),
            ("a", "a chest", 10.0, {"press_id": "p1"}),
        ]
    )
    assert [j["id"] for j in families.reference_batch(rows)] == ["b", "c"]


def test_rows_from_before_press_ids_still_group_by_the_clock():
    rows = _rows(
        [("b", "a chest", 100.5, {}), ("a", "a chest", 100.0, {}), ("z", "a chest", 5.0, {})]
    )
    assert [j["id"] for j in families.reference_batch(rows)] == ["a", "b"]


def test_a_rerun_of_one_image_is_not_folded_into_the_press_it_came_from():
    rows = _rows(
        [
            ("r", "a chest", 100.2, {"press_id": "p1", "rerun_of": "a"}),
            ("b", "a chest", 100.1, {"press_id": "p1"}),
            ("a", "a chest", 100.0, {"press_id": "p1"}),
        ]
    )
    assert [j["id"] for j in families.reference_batch(rows)] == ["r"]


def test_the_submit_door_stamps_one_press_id_on_every_image_of_a_count_press(svc):
    from realmspinner.service import jobs as svc_jobs

    made = svc_jobs.create_job(svc, kind="text", prompt="a chest", output="reference", count=3)
    presses = {svc.store.get(i)["params"].get("press_id") for i in made["ids"]}
    assert len(presses) == 1 and None not in presses

    single = svc_jobs.create_job(svc, kind="text", prompt="a lamp", output="reference")
    assert "press_id" not in svc.store.get(single["id"])["params"]

    again = svc_jobs.create_job(svc, kind="text", prompt="a chest", output="reference", count=2)
    other = svc.store.get(again["ids"][0])["params"]["press_id"]
    assert other not in presses, "a second press is a second group"
