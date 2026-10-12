"""The 2026-10-11 Create review, slice C: one vocabulary, the contact sheet, the
empty states and the smaller fixes.

Each test's name is the claim. The ones that need drawing use a bare imgui
context (the ``frames`` fixture, ``test_create_brief.py``'s); the rest are pure.
"""

from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace
from typing import Any

import pytest
from imgui_bundle import imgui

from realmspinner import models
from realmspinner.studio import settings as studio_settings
from realmspinner.studio import theme
from realmspinner.studio.modes.create.engine import recipe as create_recipe
from realmspinner.studio.modes.create.engine import workspace as families
from realmspinner.studio.modes.create.ui import brief as create_brief
from realmspinner.studio.modes.create.ui import canvas, workspace
from realmspinner.studio.modes.create.ui.panes import settings_2d, settings_3d
from realmspinner.studio.state import DEFAULT_FORM_3D, AppState, default_form_2d


@pytest.fixture
def frames():
    """A bare imgui context around one test, as ``test_create_brief.frames``."""
    previous = imgui.get_current_context()
    context = imgui.create_context()
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
    imgui.destroy_context(context)
    if previous is not None:
        imgui.set_current_context(previous)


class Recorder:
    """Stands in for ``widgets``/``controls``/``imgui``: records each call by name."""

    def __init__(self, answers: dict[str, Any] | None = None) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.answers = answers or {}

    def __getattr__(self, name: str):
        def call(*args: Any, **kwargs: Any) -> Any:
            self.calls.append((name, args, kwargs))
            if name in self.answers:
                answer = self.answers[name]
                return answer(*args, **kwargs) if callable(answer) else answer
            return None

        return call

    def named(self, name: str) -> list[tuple[Any, ...]]:
        return [args for n, args, _ in self.calls if n == name]

    def first_args(self, name: str) -> list[Any]:
        return [args[0] for args in self.named(name) if args]


def _gui(answers: dict[str, Any] | None = None) -> Recorder:
    """A recording ``imgui`` that still carries the enums the code reads."""
    gui = Recorder(answers)
    gui.Col_ = imgui.Col_
    gui.ImVec4 = imgui.ImVec4
    return gui


def _ctx(**over: Any) -> Any:
    state = AppState(mode="create")
    ns = SimpleNamespace(
        state=state,
        svc=SimpleNamespace(config=None),
        busy=lambda _k: False,
        submit=lambda *a, **k: True,
        toast=lambda *a, **k: None,
        job=lambda: None,
        cache=SimpleNamespace(get=lambda _i: None, jobs=[], active=None),
        confirms=SimpleNamespace(ask=lambda _d: None),
        textures=None,
        model_rows=[],
        model_picks=set(),
        rigging_available=True,
    )
    for key, value in over.items():
        setattr(ns, key, value)
    return ns


# --- item 1: no image model installed ----------------------------------------------

_BASE_ROW = {"row_key": "base:sdxl_cfg", "present": False, "label": "SDXL 1.0", "size_gib": 7.0}
_ENGINE_ROWS = [
    {"row_key": "engine:trellis_runtime", "present": False, "label": "3D runtime", "size_gib": 1.5},
    {"row_key": "engine:trellis_gguf", "present": False, "label": "3D weights", "size_gib": 4.0},
]


def _no_model_ctx(config: Any, rows: list[dict[str, Any]] | None = None) -> Any:
    return _ctx(svc=SimpleNamespace(config=config), model_rows=rows or [_BASE_ROW])


@pytest.fixture
def config(tmp_path):
    from realmspinner.config import Config

    return Config(data_dir=tmp_path) if "data_dir" in Config.__dataclass_fields__ else Config()


def _filled_form() -> dict[str, Any]:
    form = default_form_2d()
    form["prompt"] = "a wooden crate"
    return form


def test_automatic_with_no_installed_model_is_a_problem_before_the_press(config):
    ctx = _no_model_ctx(config)
    problems = create_recipe.problems_for(ctx, _filled_form())
    assert [str(p) for p in problems] == [create_recipe.NO_MODEL_PROBLEM]
    assert problems[0].field == "base_model"


def test_the_no_model_repair_names_the_download_and_its_size(config):
    ctx = _no_model_ctx(config)
    rows, label, gib = create_recipe.automatic_install(ctx, _filled_form())
    assert rows == ("base:sdxl_cfg",)
    assert label == "SDXL 1.0"
    assert gib == pytest.approx(7.0)


def test_no_model_says_nothing_before_the_install_answers_land(config):
    """``model_gate``'s doctrine: an empty snapshot is silence, not "all missing"."""
    ctx = _ctx(svc=SimpleNamespace(config=config), model_rows=[])
    assert create_recipe.problems_for(ctx, _filled_form()) == []


def test_the_no_model_repair_button_ticks_the_rows_and_labels_the_size(config, monkeypatch):
    ctx = _no_model_ctx(config)
    form = _filled_form()
    problem = create_recipe.problems_for(ctx, form)[0]
    controls = Recorder({"button": True})
    controls.ButtonRole = settings_2d.controls.ButtonRole
    monkeypatch.setattr(settings_2d, "controls", controls)
    ticked: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        settings_2d.model_gate, "request_install", lambda c, rows: ticked.append(rows)
    )
    settings_2d._preflight_fix(ctx, form, problem)
    assert controls.first_args("button")[0].startswith("Download SDXL 1.0 (7.0 GiB)")
    assert ticked == [("base:sdxl_cfg",)]


def test_the_plan_footer_says_none_installed_and_not_queue_ready():
    form = _filled_form()
    assert workspace.plan_for(form, None, installed=False).recipe == "none installed"
    assert workspace.plan_for(form, None).recipe == "Automatic recipe"

    widgets = Recorder()
    ctx = _ctx()
    from realmspinner.studio.problems import Problem

    problem = Problem(create_recipe.NO_MODEL_PROBLEM, "base_model")
    mod_widgets, mod_imgui = workspace.widgets, workspace.imgui
    workspace.widgets, workspace.imgui = widgets, _gui()
    try:
        workspace.plan_footer(
            ctx, workspace.plan_for(form, None, installed=False), [problem], lambda _p: None
        )
    finally:
        workspace.widgets, workspace.imgui = mod_widgets, mod_imgui
    said = [a[0] for n, a, _ in widgets.calls if n in ("muted", "muted_wrapped") and a]
    assert "Recipe: none installed" in said
    assert "Queue: ready" not in said


def test_the_reference_canvas_becomes_a_setup_card_with_no_image_model(monkeypatch):
    ctx = _ctx(model_rows=[_BASE_ROW, *_ENGINE_ROWS])
    controls, widgets = Recorder(), Recorder()
    monkeypatch.setattr(canvas, "controls", controls)
    monkeypatch.setattr(canvas, "widgets", widgets)
    monkeypatch.setattr(canvas, "imgui", Recorder())
    canvas.empty(ctx, "reference")
    labels = [str(a).split("##")[0] for a in controls.first_args("button")]
    assert labels == [
        "Get the image model (7.0 GiB)",
        "Get the 3D engine (5.5 GiB)",
        "Make a Character",
    ]
    said = " ".join(str(a[0]) for n, a, _ in widgets.calls if n.startswith("muted") and a)
    assert "or make a Character (no GPU needed)" in said


def test_the_setup_card_buttons_tick_exactly_their_rows(monkeypatch):
    ctx = _ctx(model_rows=[_BASE_ROW, *_ENGINE_ROWS])
    monkeypatch.setattr(canvas, "controls", Recorder({"button": True}))
    monkeypatch.setattr(canvas, "widgets", Recorder())
    monkeypatch.setattr(canvas, "imgui", Recorder())
    ticked: list[tuple[str, ...]] = []
    monkeypatch.setattr(canvas.model_gate, "request_install", lambda c, rows: ticked.append(rows))
    canvas.setup_card(ctx)
    assert ticked == [
        ("base:sdxl_cfg",),
        ("engine:trellis_runtime", "engine:trellis_gguf"),
    ]
    assert ctx.state.form_2d["asset_type"] == "character"


def test_the_setup_card_is_absent_once_an_image_model_is_present(monkeypatch):
    here = {**_BASE_ROW, "present": True}
    assert canvas.image_weights_missing(_ctx(model_rows=[here])) is False
    assert canvas.image_weights_missing(_ctx(model_rows=[])) is False
    assert canvas.image_weights_missing(_ctx(model_rows=[_BASE_ROW])) is True


def test_the_tour_no_longer_promises_a_refusal_after_the_press():
    from realmspinner.studio.tour import scripts

    step = next(s for s in scripts.FIRST_HOUR.steps if s.id == "generate")
    assert "press it anyway" not in step.body
    assert "download" in step.body.lower()


# --- item 2: one vocabulary ------------------------------------------------------------


def test_create_uses_one_name_per_action():
    source = inspect.getsource(workspace)
    for retired in (
        "Use as source for 3D",
        "Use these settings",
        "Generate another attempt",
        "Mark preferred",
        "Unmark preferred",
        "Vary##",
        "Rerun##",
    ):
        assert retired not in source, retired
    for name in ("Make 3D", "Try again", "Load settings"):
        assert name in source, name


def _reference_job(**over: Any) -> dict[str, Any]:
    job = {
        "id": "r1",
        "kind": "text",
        "stage": "reference",
        "status": "done",
        "files": ["input.png"],
        "params": {"seed": 5},
    }
    job.update(over)
    return job


def _drive_actions(monkeypatch, job, *, open_menu=True):
    # The Next step row is measured (primary = avail - More - gap), so the stand-ins
    # answer those three questions with plain numbers.
    widgets = Recorder({"primary_button": False, "disabled_button": False, "button_width": 56.0})
    controls = Recorder({"button": False})
    controls.ButtonRole = workspace.controls.ButtonRole
    gui = Recorder(
        {
            "begin_popup": open_menu,
            "get_style": SimpleNamespace(item_spacing=SimpleNamespace(x=8.0)),
            "get_content_region_avail": SimpleNamespace(x=260.0),
        }
    )
    monkeypatch.setattr(workspace, "widgets", widgets)
    monkeypatch.setattr(workspace, "controls", controls)
    monkeypatch.setattr(workspace, "imgui", gui)
    ctx = _ctx(cache=SimpleNamespace(get=lambda _i: None, jobs=[job], active=None))
    workspace.selected_actions(ctx, job)
    return widgets, controls, ctx


def test_the_inspectors_next_step_shows_one_primary_button(monkeypatch):
    widgets, controls, _ctx_ = _drive_actions(monkeypatch, _reference_job(), open_menu=False)
    assert widgets.first_args("primary_button") == ["Make 3D"]
    assert "More##next-step-more" in controls.first_args("button")
    # Nothing else is a full-width row until the menu is open.
    assert widgets.named("disabled_button") == []


def test_the_more_menu_holds_the_rest_under_the_shared_names(monkeypatch):
    widgets, controls, _ctx_ = _drive_actions(monkeypatch, _reference_job())
    rows = widgets.first_args("disabled_button") + controls.first_args("button")
    for expected in ("Load settings", "Try again", "Pin for comparison", "Discard"):
        assert expected in rows, rows
    assert not {"Favourite", "Mark preferred", "Unmark preferred"} & set(rows)


def test_the_inspectors_make_3d_goes_through_the_cutout_check(monkeypatch):
    """The old "Use as source for 3D" only navigated to Mesh; the canvas, the card
    and now the inspector all open the cutout check from the one function."""
    opened: list[tuple[Any, ...]] = []
    from realmspinner.studio.modes.create.ui import stages

    monkeypatch.setattr(stages, "go", lambda *a, **k: None)
    monkeypatch.setattr(settings_3d, "promote", lambda ctx, job, form: opened.append((job["id"],)))
    ctx = _ctx()
    job = _reference_job()
    label, enabled, _why, press = workspace.next_step(ctx, job)
    assert (label, enabled) == ("Make 3D", True)
    press()
    assert opened == [("r1",)]
    assert ctx.state.source_job == "r1"


def test_discard_trashes_through_the_librarys_door_and_offers_undo():
    submitted: list[str] = []
    toasts: list[tuple[Any, ...]] = []
    ctx = _ctx(
        submit=lambda key, fn, *a, **k: submitted.append(key) or True,
        toast=lambda *a, **k: toasts.append(a),
    )
    ctx.state.selected = "r1"
    workspace.discard(ctx, _reference_job())
    assert submitted == ["delete:r1"]
    assert toasts and toasts[0][2:] == ("undo", "r1")
    assert ctx.state.selected is None


def test_the_headers_favourite_is_the_only_favourite():
    from realmspinner.studio.panes import inspector

    assert '"Unfavourite" if favourite else "Favourite"' in inspect.getsource(inspector._header)
    assert "preferred" not in inspect.getsource(workspace.selected_actions).lower()
    assert "Preferred" not in inspect.getsource(workspace)


# --- item 3: the empty canvas -----------------------------------------------------------


def test_the_reference_type_cards_show_only_while_the_form_is_pristine(monkeypatch):
    cards: list[int] = []
    monkeypatch.setattr(workspace, "type_cards", lambda ctx: cards.append(1))
    monkeypatch.setattr(canvas, "widgets", Recorder())
    ctx = _ctx()
    canvas.empty(ctx, "reference")
    assert cards == [1]
    ctx.state.form_2d["prompt"] = "a lantern"
    canvas.empty(ctx, "reference")
    assert cards == [1], "a typed prompt retires the cards, which duplicate the type combo"


def test_the_mesh_canvas_is_a_drop_zone_not_the_reference_cards(frames, monkeypatch):
    cards: list[int] = []
    monkeypatch.setattr(workspace, "type_cards", lambda ctx: cards.append(1))
    ref = {**_reference_job(), "params": {"create_workspace": "w1"}}
    ctx = _ctx(
        cache=SimpleNamespace(get=lambda i: ref if i == "r1" else None, jobs=[ref], active=None)
    )
    ctx.state.create.workspace = "creation:w1"
    said: list[str] = []
    real = canvas.widgets.muted_wrapped
    monkeypatch.setattr(
        canvas.widgets, "muted_wrapped", lambda text: said.append(text) or real(text)
    )
    frames(lambda: canvas.empty(ctx, "mesh"))
    assert cards == []
    assert "Drop an image here, or pick a reference." in said


def test_a_finished_reference_thumbnail_on_the_mesh_canvas_becomes_the_source(frames, monkeypatch):
    ref = {**_reference_job(), "params": {"create_workspace": "w1"}}
    ctx = _ctx(
        cache=SimpleNamespace(get=lambda i: ref if i == "r1" else None, jobs=[ref], active=None)
    )
    ctx.state.create.workspace = "creation:w1"
    monkeypatch.setattr(canvas.thumbs, "job_thumb", lambda *a, **k: imgui.dummy((8, 8)))
    monkeypatch.setattr(imgui, "is_item_clicked", lambda *a, **k: True)
    frames(lambda: canvas.empty(ctx, "mesh"))
    assert ctx.state.source_job == "r1"


def test_the_source_chip_leaves_the_cursor_below_the_chip_not_beside_it(frames):
    """It ended at ``(high.x, low.y)``, so the next label drew over the chip."""
    source = {"id": "ref-1", "name": "well", "status": "done", "stage": "reference", "files": []}
    ctx = _ctx(cache=SimpleNamespace(get=lambda i: source, jobs=[source], active=None))
    ctx.state.source_job = "ref-1"
    ctx.state.create.stage = "mesh"
    seen: dict[str, float] = {}

    def build() -> None:
        start = imgui.get_cursor_screen_pos()
        create_brief._source_chip(ctx, 300.0)
        after = imgui.get_cursor_screen_pos()
        seen["bottom"] = start.y + create_brief.sp(create_brief.SOURCE_H)
        seen["after_y"], seen["after_x"], seen["start_x"] = after.y, after.x, start.x

    frames(build)
    assert seen["after_y"] >= seen["bottom"]
    assert seen["after_x"] == pytest.approx(seen["start_x"], abs=1.0)


# --- item 4: the contact sheet ------------------------------------------------------------


def _batch(n: int, *, status: str = "done", base: float = 100.0, wid: str = "w1") -> list[dict]:
    return [
        {
            "id": f"b{i}",
            "kind": "text",
            "stage": "reference",
            "status": status,
            "prompt": "a chest",
            "created_at": base + i * 0.01,
            "files": ["input.png"],
            "params": {"create_workspace": wid, "seed": i},
        }
        for i in range(n)
    ]


def _sheet_ctx(jobs: list[dict], selected: str | None = "b0") -> Any:
    by_id = {j["id"]: j for j in jobs}
    ctx = _ctx(cache=SimpleNamespace(get=by_id.get, jobs=jobs, active=None))
    ctx.state.create.workspace = "creation:w1"
    ctx.state.create.stage = "reference"
    ctx.state.selected = selected
    return ctx


def test_a_batch_is_the_newest_press_and_nothing_older():
    rows = families.results(
        families.build_index(_batch(3) + _batch(2, base=50.0)), "creation:w1", "reference"
    )
    batch = families.reference_batch(rows)
    assert [j["id"] for j in batch] == ["b0", "b1", "b2"]
    assert batch[0]["created_at"] < batch[-1]["created_at"], "in the order they were drawn"


@pytest.mark.parametrize(
    ("count", "shape"),
    [(2, (1, 2)), (3, (2, 2)), (4, (2, 2)), (5, (2, 4)), (8, (2, 4))],
)
def test_the_sheet_shape_follows_the_count(count, shape):
    assert families.grid_shape(count) == shape


def test_arrow_steps_clamp_and_move_by_a_row():
    assert families.grid_step(0, "right", 4) == 1
    assert families.grid_step(3, "right", 4) == 3
    assert families.grid_step(0, "down", 4) == 2
    assert families.grid_step(2, "up", 4) == 0
    assert families.grid_step(0, "left", 4) == 0
    assert families.grid_step(1, "down", 2) == 1, "a single row has no Down"


def test_a_finished_batch_of_two_or_more_shows_the_sheet_while_one_member_is_selected():
    ctx = _sheet_ctx(_batch(4))
    assert [j["id"] for j in canvas.grid_batch(ctx)] == ["b0", "b1", "b2", "b3"]


def test_the_sheet_waits_for_the_whole_batch_to_settle():
    jobs = _batch(3)
    jobs[2]["status"] = "running"
    assert canvas.grid_batch(_sheet_ctx(jobs)) is None


def test_one_image_is_not_a_contact_sheet():
    assert canvas.grid_batch(_sheet_ctx(_batch(1))) is None


def test_the_sheet_ends_when_the_selection_leaves_the_batch():
    jobs = _batch(3) + [{**_batch(1)[0], "id": "other", "created_at": 10.0}]
    assert canvas.grid_batch(_sheet_ctx(jobs, selected="other")) is None
    assert canvas.grid_batch(_sheet_ctx(jobs, selected=None)) is None


def test_the_sheet_is_a_reference_stage_surface():
    ctx = _sheet_ctx(_batch(3))
    ctx.state.create.stage = "mesh"
    assert canvas.grid_batch(ctx) is None


def test_esc_closes_the_sheet_and_the_toggle_brings_it_back(monkeypatch):
    ctx = _sheet_ctx(_batch(3))
    batch = canvas.grid_batch(ctx)
    monkeypatch.setattr(
        canvas, "imgui", Recorder({"is_key_pressed": lambda key: key == "ESC", "Key": None})
    )
    canvas.dismiss(ctx, batch)
    assert canvas.grid_batch(ctx) is None
    assert canvas.grid_batch(ctx, ignore_dismissed=True) is not None
    canvas.reopen(ctx, batch)
    assert canvas.grid_batch(ctx) is not None


class _Keys:
    """An ``imgui`` double whose ``is_key_pressed`` is true for one named key."""

    Key = imgui.Key

    def __init__(self, pressed: Any) -> None:
        self.pressed = pressed

    def is_key_pressed(self, key: Any, repeat: bool = True) -> bool:
        return key == self.pressed

    def get_io(self) -> Any:
        # No modifier held: the labelling keys refuse a chord (``handle_label_keys``).
        return SimpleNamespace(key_ctrl=False, key_alt=False, key_shift=False, key_super=False)


def test_arrow_keys_move_the_selection_and_enter_makes_3d_and_esc_returns(monkeypatch):
    jobs = _batch(4)
    ctx = _sheet_ctx(jobs, selected="b1")
    batch = canvas.grid_batch(ctx)
    chosen: list[str] = []
    made: list[str] = []
    monkeypatch.setattr(workspace, "inspect_result", lambda c, job: chosen.append(job["id"]))
    monkeypatch.setattr(workspace, "make_3d", lambda c, job: made.append(job["id"]))

    monkeypatch.setattr(canvas, "imgui", _Keys(imgui.Key.right_arrow))
    canvas.handle_keys(ctx, batch, 1)
    monkeypatch.setattr(canvas, "imgui", _Keys(imgui.Key.down_arrow))
    canvas.handle_keys(ctx, batch, 1)
    assert chosen == ["b2", "b3"]

    monkeypatch.setattr(canvas, "imgui", _Keys(imgui.Key.enter))
    canvas.handle_keys(ctx, batch, 1)
    assert made == ["b1"]

    monkeypatch.setattr(canvas, "imgui", _Keys(imgui.Key.escape))
    canvas.handle_keys(ctx, batch, 1)
    assert canvas.grid_batch(ctx) is None


def test_the_sheet_reads_keys_only_with_canvas_focus_and_no_text_field(monkeypatch):
    ctx = _sheet_ctx(_batch(2))

    def gate(focused: bool, typing: bool, palette: bool = False, modal: bool = False) -> bool:
        io = SimpleNamespace(want_text_input=typing)
        monkeypatch.setattr(
            canvas,
            "imgui",
            SimpleNamespace(is_window_focused=lambda *a: focused, get_io=lambda: io),
        )
        monkeypatch.setattr(canvas.dialogs, "modal_open", lambda c: modal)
        ctx.state.palette_open = palette
        return canvas.keys_enabled(ctx)

    assert gate(True, False) is True
    assert gate(False, False) is False, "focus is on the prompt or the inspector"
    assert gate(True, True) is False, "a text field is taking keystrokes"
    assert gate(True, False, palette=True) is False
    assert gate(True, False, modal=True) is False


def test_the_sheet_draws_every_member_through_the_one_cell_function(frames, monkeypatch):
    assert list(inspect.signature(canvas.grid_cell).parameters) == [
        "ctx",
        "job",
        "size",
        "selected",
        "index",
    ]
    jobs = _batch(4)
    ctx = _sheet_ctx(jobs, selected="b2")
    drawn: list[tuple[str, bool, int]] = []

    def cell(c, job, size, *, selected, index):
        drawn.append((job["id"], selected, index))
        return False

    monkeypatch.setattr(canvas, "grid_cell", cell)
    monkeypatch.setattr(canvas.dialogs, "modal_open", lambda c: False)
    result: list[bool] = []
    frames(lambda: result.append(canvas.draw_grid(ctx, 900.0, 600.0)))
    assert result == [True]
    assert drawn == [("b0", False, 0), ("b1", False, 1), ("b2", True, 2), ("b3", False, 3)]


def test_clicking_a_cell_selects_that_image(frames, monkeypatch):
    jobs = _batch(2)
    ctx = _sheet_ctx(jobs)
    chosen: list[str] = []
    monkeypatch.setattr(workspace, "inspect_result", lambda c, job: chosen.append(job["id"]))
    monkeypatch.setattr(canvas, "grid_cell", lambda c, job, size, **k: job["id"] == "b1")
    monkeypatch.setattr(canvas.dialogs, "modal_open", lambda c: False)
    frames(lambda: canvas.draw_grid(ctx, 900.0, 600.0))
    assert chosen == ["b1"]


def test_the_newest_finished_image_is_selected_unless_the_user_moved(monkeypatch):
    from realmspinner.studio.modes.create.ui.panes import settings_2d as _s  # noqa: F401

    jobs = _batch(3)
    ctx = _sheet_ctx(jobs, selected="b0")
    ctx.state.mode = "create"
    ctx.state.create.follow_selection = "b0"
    picked: list[str] = []
    from realmspinner.studio.modes.library.ui.panes import library

    monkeypatch.setattr(library, "select", lambda c, i: (picked.append(i), c.state.select(i)))
    assert workspace.follow_finished_reference(ctx, jobs[1]) is True
    assert ctx.state.selected == "b1"
    # The user clicks an earlier sibling mid-batch...
    ctx.state.select("b0")
    # ...and the next image to finish does not take it back.
    assert workspace.follow_finished_reference(ctx, jobs[2]) is False
    assert ctx.state.selected == "b0"
    assert picked == ["b1"]


def test_a_finished_image_does_not_pull_the_user_off_another_stage_or_creation(monkeypatch):
    jobs = _batch(2)
    ctx = _sheet_ctx(jobs, selected="b0")
    ctx.state.create.follow_selection = "b0"
    ctx.state.create.stage = "mesh"
    assert workspace.follow_finished_reference(ctx, jobs[1]) is False
    ctx.state.create.stage = "reference"
    ctx.state.create.workspace = "creation:elsewhere"
    assert workspace.follow_finished_reference(ctx, jobs[1]) is False


def test_the_canvas_make_3d_is_offered_over_a_finished_reference_only():
    job = _reference_job()
    assert canvas.make_3d_job(_ctx(job=lambda: job)) is job
    assert canvas.make_3d_job(_ctx(job=lambda: _reference_job(status="running"))) is None
    assert canvas.make_3d_job(_ctx(job=lambda: _reference_job(files=[]))) is None
    assert canvas.make_3d_job(_ctx(job=lambda: _reference_job(stage="model"))) is None


def test_the_canvas_make_3d_and_the_sheets_use_the_one_make_3d(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(workspace, "make_3d", lambda c, job: calls.append(job["id"]))
    job = _reference_job()
    ctx = _ctx(job=lambda: job)
    monkeypatch.setattr(canvas.widgets, "primary_button", lambda *a, **k: True)
    monkeypatch.setattr(canvas.controls, "button", lambda *a, **k: False)
    monkeypatch.setattr(
        canvas,
        "imgui",
        SimpleNamespace(
            get_window_pos=lambda: SimpleNamespace(x=0, y=0),
            get_window_size=lambda: SimpleNamespace(x=800, y=600),
            set_cursor_screen_pos=lambda *a: None,
                same_line=lambda *a, **k: None,
                is_window_focused=lambda *a: False,
        ),
    )
    canvas.draw_overlay(ctx)
    assert calls == ["r1"]
    assert "_make_3d = make_3d" in inspect.getsource(workspace)


# --- item 4 + 5: two count controls ---------------------------------------------------------


def test_mesh_count_hints_say_pick_the_best():
    assert "keep the better" not in " ".join(create_brief._MESH_COUNT_HINTS.values())
    assert all("pick the best" in v for k, v in create_brief._MESH_COUNT_HINTS.items() if k != "1")


def test_each_stage_writes_its_own_form_key(frames):
    ref = _ctx()
    mesh = _ctx()
    mesh.state.create.stage = "mesh"
    seen: list[str] = []
    for ctx, stage_mesh, key in ((ref, False, "count"), (mesh, True, "attempts")):
        form = ctx.state.form_3d if stage_mesh else ctx.state.form_2d
        assert key in form
        before = dict(form)
        counts = create_brief._MESH_COUNTS if stage_mesh else create_brief._COUNTS
        frames(
            lambda ctx=ctx, form=form, counts=counts, key=key: create_brief._count(
                ctx, form, counts, int(form[key]), key=key
            )
        )
        seen.append(key)
        assert form == before
    assert seen == ["count", "attempts"]
    assert "count" not in DEFAULT_FORM_3D and "attempts" in DEFAULT_FORM_3D


def test_a_mesh_form_saved_with_count_keeps_its_number_as_attempts():
    restored = studio_settings.restore_form(DEFAULT_FORM_3D, {"count": 3, "rig": True})
    assert restored["attempts"] == 3
    assert "count" not in restored
    # Out of the mesh range: dropped to the default rather than refused later.
    assert studio_settings.restore_form(DEFAULT_FORM_3D, {"count": 8})["attempts"] == 1
    # The new key wins when both are present.
    assert (
        studio_settings.restore_form(DEFAULT_FORM_3D, {"count": 2, "attempts": 3})["attempts"] == 3
    )
    # The image form is untouched by the alias.
    assert studio_settings.restore_form(default_form_2d(), {"count": 4})["count"] == 4


def test_the_reset_lives_in_the_new_popup(frames, monkeypatch):
    ctx = _ctx(confirms=SimpleNamespace(ask=lambda d: None))
    seen: list[str] = []
    real = create_brief.controls.button

    def spy(label, *a, **k):
        seen.append(str(label))
        real(label, *a, **k)
        return str(label) == "New..." and len(seen) < 3

    monkeypatch.setattr(create_brief.controls, "button", spy)
    rail = lambda *a, **k: None  # noqa: E731

    def build() -> None:
        create_brief.draw(ctx, rail)

    frames(build)
    frames(build)
    assert "Reset image settings..." in seen


# --- item 7: the Mesh column --------------------------------------------------------------------


def _cards(monkeypatch, *, open_collapsible: bool):
    """``widgets.group`` replaced by a recorder, so a test can say which cards a
    column opens, whether each is a collapsible one, and what its body drew.
    A collapsible card yields ``open_collapsible``; any other opens."""
    import contextlib

    opened: list[tuple[str, bool]] = []

    @contextlib.contextmanager
    def group(title=None, *, collapsible=False, **_kw):
        opened.append((title, collapsible))
        yield open_collapsible if collapsible else True

    monkeypatch.setattr(settings_3d.widgets, "group", group)
    return opened


def test_the_mesh_basic_view_holds_the_budget_size_and_rig_and_one_fold(monkeypatch):
    drawn: list[str] = []
    for name in ("_budget", "_size", "_rig", "_turnaround"):
        monkeypatch.setattr(settings_3d, name, lambda *a, _n=name, **k: drawn.append(_n))
    monkeypatch.setattr(settings_3d, "_advanced_basics", lambda *a, **k: drawn.append("advanced"))
    monkeypatch.setattr(settings_3d, "_hint", lambda *a, **k: None)
    monkeypatch.setattr(settings_3d.widgets, "help_marker", lambda *a: None)
    monkeypatch.setattr(settings_3d.manual_render, "help_button", lambda *a: None)
    cards = _cards(monkeypatch, open_collapsible=False)
    ctx = _ctx(svc=SimpleNamespace(config=SimpleNamespace(bench_dir="nowhere")))
    settings_3d._draw_form(ctx, SimpleNamespace(), "Mesh resolution", None)
    # The Mesh card, then the one collapsible card (Rig and Turnaround open
    # their own cards inside the functions replaced above).
    assert cards == [("Mesh", False), ("Advanced", True)]
    assert drawn == ["_budget", "_size", "_rig", "_turnaround"], "the fold was closed"


def test_the_fold_opens_onto_the_old_top_level_controls_and_the_engine_flags(monkeypatch):
    drawn: list[str] = []
    _cards(monkeypatch, open_collapsible=True)
    monkeypatch.setattr(settings_3d.widgets, "muted_wrapped", lambda *a: None)
    monkeypatch.setattr(settings_3d.widgets, "divider", lambda: drawn.append("divider"))
    monkeypatch.setattr(settings_3d.widgets, "secondary", lambda t: drawn.append(t))
    monkeypatch.setattr(settings_3d.widgets, "sub_label", lambda t: drawn.append(t))
    form_ui = SimpleNamespace(number=lambda key, *a, **k: (False, 0))
    monkeypatch.setattr(settings_3d, "_hint", lambda *a, **k: None)
    ctx = _ctx()
    settings_3d._engine(
        ctx, dict(DEFAULT_FORM_3D), form_ui, None, leading=lambda: drawn.append("leading")
    )
    # "Engine" names a run of fields inside the card: a sub-label, not a rule
    # and a second heading (which read as a card in a card).
    assert drawn[:2] == ["leading", "Engine"]
    assert "divider" not in drawn


def test_budget_entries_carry_their_unit(monkeypatch):
    monkeypatch.setattr(settings_3d.remesh_panel, "blender_available", lambda c: True)
    monkeypatch.setattr(settings_3d.retarget_panel, "gltfpack_available", lambda c: True)
    labels = [label for _key, label in settings_3d._budget_options(_ctx())]
    assert "Game-ready · 5,000 triangles" in labels
    assert not [label for label in labels if "(5k)" in label]
    assert settings_3d._stale_lowpoly_entry({"lowpoly_triangles": 7777})[1] == (
        "Game-ready · 7,777 triangles"
    )


def test_background_options_use_readable_labels():
    ctx = _ctx(guidance={"bg_removal": ["auto", "birefnet", "threshold"]})
    labels = [label for _key, label in settings_3d._bg_options(ctx)]
    assert labels[1:] == ["Automatic", "BiRefNet cutout", "Plain-background threshold"]


def test_no_user_facing_mesh_string_says_trellis():
    """Docstrings and comments name the engine for maintainers; the strings the
    user reads (tooltips, labels, help text) say "the reconstruction engine"."""
    tree = ast.parse(inspect.getsource(settings_3d))
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.Module))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    prose = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
        and " " in node.value
        and "trellis" in node.value.lower()
    ]
    assert not prose, prose


def test_the_engine_fold_is_called_advanced():
    source = inspect.getsource(settings_3d)
    assert '"Advanced",' in source and 'key="create/mesh-advanced"' in source
    assert "Engine (advanced)" not in source.replace('It was "Engine (advanced)"', "")


# --- item 9: the clean cutout is accepted by default ------------------------------------------


def test_a_clean_cutout_is_accepted_without_asking_for_a_user_who_never_chose(tmp_path):
    settings = studio_settings.Settings(tmp_path / "s.json")
    assert settings.get(settings_3d.SKIP_CLEAN_MATTE_SETTING) is True
    ctx = SimpleNamespace(settings=settings)
    state = SimpleNamespace(preview=None, _tried_and_failed=False)
    # Still being computed: wait rather than open a modal that may be unnecessary.
    assert settings_3d._wants_auto_accept(ctx, state) is True


def test_a_user_who_turned_it_off_still_gets_the_modal(tmp_path):
    settings = studio_settings.Settings(tmp_path / "s.json")
    settings.set(settings_3d.SKIP_CLEAN_MATTE_SETTING, False)
    assert settings.get(settings_3d.SKIP_CLEAN_MATTE_SETTING) is False
    ctx = SimpleNamespace(settings=settings)
    assert settings_3d._wants_auto_accept(ctx, SimpleNamespace(preview=None)) is False


def test_the_default_does_not_write_itself_into_the_file(tmp_path):
    settings = studio_settings.Settings.load(tmp_path)
    assert settings.get(settings_3d.SKIP_CLEAN_MATTE_SETTING) is True
    assert settings.data == {}
    assert settings.tick() is False


# --- item 10: the smaller fixes ---------------------------------------------------


def test_a_pristine_form_says_describe_the_asset_to_start_not_a_red_error():
    from realmspinner.studio.problems import Problem

    widgets, gui = Recorder(), _gui()
    ctx = _ctx()
    problem = Problem("A prompt is required.", "prompt")
    saved = workspace.widgets, workspace.imgui
    workspace.widgets, workspace.imgui = widgets, gui
    try:
        workspace.plan_footer(ctx, workspace.plan_for(default_form_2d()), [problem], lambda p: None)
        assert "Describe the asset to start" in widgets.first_args("muted")
        assert gui.named("text_wrapped") == [] or all(
            "Needs attention" not in a[0] for a in gui.named("text_wrapped")
        )
        # After an edit (or a refused press) the same problem is red.
        ctx.state.create.prompt_touched = True
        gui.calls.clear()
        workspace.plan_footer(ctx, workspace.plan_for(default_form_2d()), [problem], lambda p: None)
        assert any(
            "Needs attention: A prompt is required." in a[0] for a in gui.named("text_wrapped")
        )
    finally:
        workspace.widgets, workspace.imgui = saved


def test_editing_the_prompt_or_pressing_generate_arms_the_red(frames):
    ctx = _ctx()
    form = ctx.state.form_2d
    assert ctx.state.create.prompt_touched is False
    from realmspinner.studio.problems import Problem

    settings_2d.refuse(ctx, [Problem("A prompt is required.", "prompt")])
    assert ctx.state.create.prompt_touched is True
    settings_2d.reset(ctx)
    assert ctx.state.create.prompt_touched is False, "a reset form is pristine again"
    assert form is not ctx.state.form_2d


def test_the_plan_card_is_sized_from_its_content_not_a_fixed_reserve():
    for module in (settings_2d, settings_3d):
        source = inspect.getsource(module.draw)
        assert "(0, -sp(220))" not in source and "sp(52)" not in source
        # The form leaves the card's measured height and a gutter; the card notes
        # its own height when it closes.
        assert "plan_reserve(ctx," in source and "plan_card(ctx," in source
    assert "note_plan_height(ctx, key)" in inspect.getsource(workspace.plan_card)


def test_the_plan_height_follows_the_measured_card_within_bounds(frames):
    ctx = _ctx()
    seen: dict[str, float] = {}

    def build() -> None:
        seen["default"] = workspace.plan_height(ctx, "2d")
        with workspace.plan_card(ctx, "2d"):
            for _ in range(12):
                imgui.text("a line of the footer")

    frames(build)
    frames(build)
    frames(build)
    measured = ctx.state.create.plan_heights["2d"]
    # The whole card: its twelve lines and its padding, not a cursor position.
    assert measured > 12 * imgui.get_text_line_height() * 0.9 + 2 * workspace.sp(12) - 1
    frames(lambda: seen.__setitem__("after", workspace.plan_height(ctx, "2d")))
    floor = workspace.sp(workspace.PLAN_MIN_DP)
    assert seen["after"] >= floor
    assert seen["after"] <= 900 * workspace._PLAN_MAX_FRACTION + 1
    assert seen["after"] == max(floor, min(measured, 900 * workspace._PLAN_MAX_FRACTION)), (
        "the reserve is the measured card, clamped"
    )


def test_a_non_commercial_model_is_flagged_when_automatic_picked_it(monkeypatch):
    from _synthetic_models import distilled_base

    spec = distilled_base("nc_test", commercial=False, license="CC-BY-NC-4.0")
    monkeypatch.setitem(models.BASE_MODELS, spec.key, spec)
    resolved = SimpleNamespace(
        recipe=SimpleNamespace(label="NC recipe", note=""), base_model=spec.key, warning=""
    )
    monkeypatch.setattr(create_recipe, "resolved_recipe", lambda c, f: resolved)
    widgets = Recorder({"combo": ""})
    monkeypatch.setattr(settings_2d, "widgets", widgets)
    # The Image model field's label and rhythm are ``create_brief.card_field``'s.
    monkeypatch.setattr(settings_2d.create_brief, "widgets", widgets)
    form = default_form_2d()
    form["model_mode"] = "auto"
    ctx = _ctx()
    ctx.base_models = []
    settings_2d._model(ctx, form, None)
    warned = [a for a in widgets.named("wrapped") if a and "may NOT be used commercially" in a[1]]
    assert warned, widgets.calls


def test_the_progress_row_estimates_time_left_and_never_goes_negative():
    job = {"kind": "text", "stage": "reference", "started_at": 1000.0}
    assert workspace.eta_phrase(job, 1005.0) == "about 15 seconds left"
    assert workspace.eta_phrase(job, 1019.0) == "19 s so far", "inside the last five seconds"
    # Past the estimate: elapsed alone, never a negative or a made-up figure.
    late = workspace.eta_phrase(job, 1100.0)
    assert late == "1 min 40 s so far" and "-" not in late
    mesh = {"kind": "text", "stage": "model", "started_at": 1000.0}
    assert workspace.eta_phrase(mesh, 1040.0) == "about 80 seconds left"
    assert workspace.eta_phrase(mesh, 1000.0) == "about 2 minutes left"
    # No estimate for a kind we cannot size, no phrase before it starts, none on a bad clock.
    assert workspace.eta_phrase({"kind": "rig", "started_at": 1000.0}, 1010.0) == "10 s so far"
    assert workspace.eta_phrase({"kind": "text", "stage": "reference"}, 1010.0) == ""
    assert workspace.eta_phrase(job, 900.0) == ""


def test_the_lora_architecture_note_is_a_tooltip_not_a_line_of_text():
    source = inspect.getsource(settings_2d._lora)
    assert "tooltip=narrowed" in source
    assert "muted_wrapped(narrowed)" not in source


def test_alt_digits_travel_the_stage_rail_and_a_blocked_stage_says_why():
    from realmspinner.studio.modes.create.ui import stages

    toasts: list[str] = []
    went: list[str] = []
    ctx = _ctx(toast=lambda t, *a, **k: toasts.append(t))
    ctx.state.form_2d["asset_type"] = "image"
    from realmspinner.studio.modes.create.engine import assets as create_assets

    create_assets.sync_legacy_fields(ctx.state.form_2d)
    assert stages.rail_stages(ctx) == ("reference", "export")
    orig_go = stages.go
    stages.go = lambda c, s, **k: went.append(s)
    try:
        assert stages.go_nth(ctx, 1) is True
        assert stages.go_nth(ctx, 2) is False, "Export needs a finished asset"
        assert stages.go_nth(ctx, 9) is False
    finally:
        stages.go = orig_go
    assert went == ["reference"]
    assert len(toasts) == 1


def test_the_shortcut_sheet_lists_the_stage_keys():
    from realmspinner.studio.shortcuts import shortcut_sections

    create = dict(shortcut_sections())["Create"]
    keys = [k for k, _ in create]
    assert "Alt+1 - 5" in keys


def test_open_model_setup_ticks_the_rows_of_a_selected_model_that_is_not_downloaded(
    config, monkeypatch
):
    rows = [
        {"row_key": "base:sdxl_cfg_pag", "present": False, "label": "SDXL PAG", "size_gib": 7.0}
    ]
    ctx = _ctx(svc=SimpleNamespace(config=config), model_rows=rows)
    form = _filled_form()
    form["model_mode"] = "advanced"
    form["base_model"] = form["model_override"] = "sdxl_cfg_pag"
    (problem,) = create_recipe.problems_for(ctx, form)
    assert problem.rows == ("base:sdxl_cfg_pag",)
    controls = Recorder({"button": True})
    controls.ButtonRole = settings_2d.controls.ButtonRole
    monkeypatch.setattr(settings_2d, "controls", controls)
    ticked: list[tuple[str, ...]] = []
    monkeypatch.setattr(settings_2d.model_gate, "request_install", lambda c, r: ticked.append(r))
    settings_2d._preflight_fix(ctx, form, problem)
    assert ticked == [("base:sdxl_cfg_pag",)]


def test_the_shell_wires_the_selection_follow_to_a_landing_press_and_a_finishing_image():
    from realmspinner.studio.shell import tasks

    source = inspect.getsource(tasks)
    assert "create_workspace.note_submit_landed(ctx)" in source
    assert "create_workspace.follow_finished_reference(ctx, job)" in source


# --- the real draws, through imgui, so a bad call signature fails here ---------------------------


def test_the_contact_sheet_and_its_cells_draw_for_real(frames, monkeypatch):
    jobs = _batch(5)
    jobs[3]["status"] = "error"
    jobs[3]["files"] = []
    ctx = _sheet_ctx(jobs, selected="b1")
    monkeypatch.setattr(canvas.dialogs, "modal_open", lambda c: False)
    drawn: list[bool] = []
    frames(lambda: drawn.append(canvas.draw_grid(ctx, 900.0, 600.0)))
    assert drawn == [True]


def test_the_setup_card_and_the_canvas_overlay_draw_for_real(frames, monkeypatch):
    monkeypatch.setattr(canvas.dialogs, "modal_open", lambda c: False)
    ctx = _ctx(model_rows=[_BASE_ROW, *_ENGINE_ROWS], rigging_available=False)
    frames(lambda: canvas.empty(ctx, "reference"))
    frames(lambda: canvas.empty(ctx, "rig"))
    job = _reference_job()
    batch = _batch(3)
    ctx2 = _sheet_ctx(batch, selected="b0")
    ctx2.job = lambda: batch[0]
    ctx2.state.create.grid_dismissed.add("b0")
    frames(lambda: canvas.draw_overlay(ctx2))
    ctx3 = _ctx(job=lambda: job)
    frames(lambda: canvas.draw_overlay(ctx3))


def test_the_inspectors_next_step_and_more_menu_draw_for_real(frames, monkeypatch):
    job = _reference_job()
    ctx = _ctx(cache=SimpleNamespace(get=lambda _i: None, jobs=[job], active=None))
    spy = []
    real = workspace.controls.button

    def button(label, *a, **k):
        spy.append(str(label))
        real(label, *a, **k)
        return str(label).startswith("More") and len(spy) < 3

    monkeypatch.setattr(workspace.controls, "button", button)
    frames(lambda: workspace.selected_actions(ctx, job))
    frames(lambda: workspace.selected_actions(ctx, job))
    assert any(label.startswith("Discard") for label in spy), spy
    assert any(label.startswith("Try again") for label in spy), spy
