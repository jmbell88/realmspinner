"""The 2026-10-03 audit's Medium findings, batch shell-3: shell-25 .. shell-30,
shell-35 .. shell-37 (Review, Settings, shortcuts, tasks, events, frame), plus
the "four SDXL recipes" count in Settings' remove control.

Each test's name is the claim and was run red against the unfixed tree first.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pygame
import pytest
from modes.review.test_review_mode import FakeCtx, _Done, _scanned, _sweep

from realmspinner.service import verdicts as svc_verdicts
from realmspinner.studio.modes.review import mode as review_mode


@pytest.fixture
def ctx(svc):
    return FakeCtx(svc)


@pytest.fixture(autouse=True)
def _no_pygame_display(monkeypatch):
    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0)


# --- shell-25: a unit that can never be graded does not hold its sweep ---------


def test_a_pass_moves_on_when_the_only_unverdicted_unit_of_a_sweep_errored(ctx, svc):
    _, ids_a = _sweep(svc, "one", n=2)
    _, ids_b = _sweep(svc, "two", n=2)
    # The errored unit is the *first* of each sweep, which is where a pass used
    # to park: ``open_sweep`` lands on the first unit with no verdict.
    svc.store.set_status(ids_a[0], "error")
    svc.store.set_status(ids_b[0], "error")
    state = _scanned(ctx)

    assert review_mode.start_judging(ctx) is True
    first, second = state.judging.order
    assert state.sweep_id == first
    assert review_mode.current(state)["status"] == "done", "the cursor must skip the errored unit"

    review_mode.record(ctx, 3)

    assert state.sweep_id == second, "the pass parked on a unit that can never be graded"
    assert review_mode.CLEANUP_KEY in ctx.submitted, "the judged sweep was never cleaned up"


def test_the_start_judging_count_leaves_out_units_that_can_never_be_graded(ctx, svc):
    _sweep(svc, "one", n=3)
    sweep_id, ids = _sweep(svc, "two", n=2)
    svc.store.set_status(ids[0], "error")
    svc.store.set_status(_sweep_first(svc, "one"), "cancelled")
    state = _scanned(ctx)

    assert review_mode.todo_total(state) == 3
    assert next(s for s in state.sweeps if s["id"] == sweep_id)["todo"] == 1


def _sweep_first(svc, label):
    sweep = next(s for s in svc.store.list_sweeps() if s["label"] == label)
    return svc.store.sweep_jobs(sweep["id"])[0]["id"]


def test_a_queued_unit_still_owes_a_verdict_so_its_sweep_is_not_cleaned_up(ctx, svc):
    sweep_id, ids = _sweep(svc, n=2)
    svc.store.set_status(ids[1], "queued")
    svc_verdicts.record_verdict(svc, ids[0], grade=3)
    state = _scanned(ctx)

    assert next(s for s in state.sweeps if s["id"] == sweep_id)["todo"] == 1


# --- shell-26: a launch landing during a scan discards that scan ----------------


def test_a_scan_in_flight_when_a_launch_lands_is_discarded_and_reasked(ctx, svc):
    state = _scanned(ctx)
    # A scan that started before the launch: collected now, delivered later.
    state.scanning = True
    stale = (state.scan_generation, review_mode._collect(svc))
    launched, _ids = _sweep(svc, "just launched", n=2)

    review_mode.on_task_done(ctx, _Done(review_mode.LAUNCH_KEY, {"id": launched, "units": 2}))
    review_mode.on_task_done(ctx, _Done(review_mode.SCAN_KEY, stale))

    assert state.scanning is True, "the stale scan was applied instead of being re-asked"
    review_mode.on_task_done(ctx, _Done(review_mode.SCAN_KEY, ctx.result))
    assert any(s["id"] == launched for s in state.sweeps)
    assert state.sweep_id == launched


# --- shell-27: grading a unit that is not finished says why ---------------------


def test_grading_a_unit_that_is_not_finished_says_why(ctx, svc):
    sweep_id, ids = _sweep(svc, n=1)
    svc.store.set_status(ids[0], "queued")
    state = _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)

    review_mode.record(ctx, 3)

    message, kind = ctx.toasts[-1]
    assert kind == "error"
    assert "queued" in message and "finished" in message, message
    assert svc.store.latest_verdicts() == []
    # The controls are greyed with the same sentence, not left live.
    reason = review_mode.verdict_blocker(state, review_mode.current(state))
    assert "queued" in reason


def test_labelling_an_image_with_no_picture_says_why(ctx, svc):
    job_id = svc.store.create("image", "a", {}, stage="reference", status="done")
    ctx.state.review = review_mode.ReviewState()
    labels = review_mode.LabelPass(stage="blank")
    labels.rows = [{"job_id": job_id, "prompt": "a", "image": None, "verdict": None}]
    ctx.state.review.labels = labels

    assert review_mode.record_label(ctx, "accept") is False

    assert "reference image" in ctx.toasts[-1][0], ctx.toasts[-1]


# --- shell-28: an unparsable seed is named, not "fill in the prompt" ------------


def test_launch_sweep_names_an_unparsable_seed_instead_of_asking_for_the_prompt(ctx):
    from realmspinner.studio.modes.review.ui import workspace

    form = review_mode.ensure(ctx).form
    form.prompt = "a chest"
    form.seeds = "4x"
    form.axes = [{"param": "trellis_band", "values": "8"}]
    state = ctx.state.review

    planned = review_mode.preview_units(state)
    problem = review_mode.plan_problem(state)
    reason = workspace._launch_sweep_reason(
        planned, submitting=False, scanning=False, problem=problem[1] if problem else ""
    )

    assert planned < 0
    assert problem is not None and problem[0] == "seeds"
    assert "4x" in reason, reason
    assert "prompt" not in reason.lower()
    # And the launch path rings the control the message is about.
    assert review_mode.launch(ctx) is False
    assert ctx.state.field_errors.get("seeds")


def test_a_half_filled_axis_row_is_named_as_the_axes():
    state = review_mode.ReviewState()
    state.form.prompt = "a chest"
    state.form.seeds = "1"
    state.form.axes = [{"param": "trellis_band", "values": ""}]

    problem = review_mode.plan_problem(state)

    assert problem is not None and problem[0] == "axes"


# --- shell-35: a mesh that lands after the unit was first viewed loads ----------


class _Viewer:
    def __init__(self) -> None:
        self.path = None
        self.pending = None
        self.has_model = False
        self.cleared = 0

    def clear(self) -> None:
        self.cleared += 1
        self.has_model = False

    def parse_model(self, path):
        return path


def test_a_unit_whose_mesh_lands_after_it_was_first_viewed_is_loaded(ctx, svc):
    from realmspinner.studio.main import REVIEW_MESH_KEY
    from realmspinner.studio.modes.review.ui.workspace import ReviewPanes

    sweep_id, ids = _sweep(svc, n=1)
    svc.store.set_status(ids[0], "running")
    state = _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)
    unit = review_mode.current(state)
    model = review_mode.model_path(unit)
    model.unlink()
    app = SimpleNamespace(viewer=_Viewer(), app_ctx=ctx)
    ctx.submitted.clear()

    ReviewPanes._review_load(app, unit, review_mode)
    assert app.viewer.path == model and ctx.submitted == []

    model.write_bytes(b"glTF-not-really")
    state.mesh_wait_checked = float("-inf")  # the retry throttle has elapsed
    ReviewPanes._review_load(app, unit, review_mode)

    assert ctx.submitted == [REVIEW_MESH_KEY], "the mesh that landed was never loaded"
    assert app.viewer.pending == model


def test_a_unit_that_never_gets_a_mesh_is_not_retried_every_frame(ctx, svc):
    from realmspinner.studio.modes.review.ui.workspace import ReviewPanes

    sweep_id, ids = _sweep(svc, n=1)
    svc.store.set_status(ids[0], "error")
    state = _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)
    unit = review_mode.current(state)
    review_mode.model_path(unit).unlink()
    app = SimpleNamespace(viewer=_Viewer(), app_ctx=ctx)
    ctx.submitted.clear()

    for _ in range(5):
        ReviewPanes._review_load(app, unit, review_mode)

    assert ctx.submitted == []
    assert app.viewer.cleared == 1, "an absent mesh must be cleared once, not per frame"


def test_a_unit_that_finished_elsewhere_is_refreshed_from_the_jobs_cache(ctx, svc):
    sweep_id, ids = _sweep(svc, n=2)
    svc.store.set_status(ids[0], "running")
    state = _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)
    unit = next(u for u in state.units if u["job_id"] == ids[0])
    assert unit["status"] == "running"

    changed = review_mode.refresh_unit_statuses(
        state, [{"id": ids[0], "status": "done"}, {"id": ids[1], "status": "done"}]
    )

    assert changed is True
    assert unit["status"] == "done"
    assert review_mode.refresh_unit_statuses(state, [{"id": ids[0], "status": "done"}]) is False


# --- shell-36: a refused layout rename says so ----------------------------------


def _library(tmp_path):
    from realmspinner.studio import layouts

    class _Settings:
        def __init__(self) -> None:
            self.store: dict = {}

        def get(self, key, default=None):
            return self.store.get(key, default)

        def set(self, key, value) -> None:
            self.store[key] = value

    library = layouts.Library(_Settings())
    return library


def test_renaming_a_layout_to_a_taken_name_says_so(tmp_path):
    from realmspinner.studio.modes.settings.ui.panes import app_settings

    library = _library(tmp_path)
    library.duplicate(library.active, "mine")
    library.duplicate(library.active, "yours")
    library.set_active("mine")
    toasts: list[tuple[str, str]] = []
    ctx = SimpleNamespace(
        toast=lambda message, kind="info", *a, **k: toasts.append((message, kind))
    )

    app_settings._rename_layout(ctx, library, "mine", "yours")

    assert "mine" in library.layouts and "yours" in library.layouts
    assert toasts and "yours" in toasts[-1][0] and toasts[-1][1] != "info", toasts

    app_settings._rename_layout(ctx, library, "mine", "renamed")
    assert "renamed" in library.layouts and "mine" not in library.layouts
    assert library.active == "renamed"


# --- shell-29: a Muse refusal still rings its field after Create has a workspace


def _muse_submit_tag(ctx):
    """The tag Muse's own Generate passes to ``ctx.submit`` -- read off the real
    call rather than restated, so the test follows whatever Muse sends."""
    from realmspinner.studio.modes.muse import mode as muse_mode
    from realmspinner.studio.panes import model_gate

    seen: dict = {}

    class _Ctx:
        state = ctx.state

        def submit(self, key, run, *args, tag=None, **kw):
            seen["key"], seen["tag"] = key, tag
            return True

        def toast(self, *a, **k):
            pass

    original = model_gate.missing
    model_gate.missing = lambda *_a, **_k: []
    try:
        muse_mode.generate(_Ctx())
    finally:
        model_gate.missing = original
    assert seen["key"] == "submit"
    return seen["tag"]


def test_a_muse_submit_refusal_rings_its_field_even_after_create_has_a_workspace(ctx):
    from realmspinner.service.errors import Invalid
    from realmspinner.studio import main as main_mod
    from realmspinner.studio.state import AppState

    state = AppState()
    state.create.workspace = "creation:abc123"
    ctx.state = state
    tag = _muse_submit_tag(ctx)
    toasts: list = []
    done = SimpleNamespace(
        key="submit",
        ok=False,
        result=None,
        error=Invalid("duration is too long", field="duration"),
        message="duration is too long",
        action=None,
        tag=tag,
    )
    app = main_mod.App.__new__(main_mod.App)
    app.app_ctx = SimpleNamespace(
        toast=lambda *a, **k: toasts.append(a),
        tasks=SimpleNamespace(poll=lambda: [done]),
        state=state,
        svc=None,
    )

    app._collect_tasks()

    assert "duration" in state.field_errors, "Muse's refusal lost its ring"


def test_a_create_submit_from_another_workspace_still_does_not_ring():
    """The pin the fix must not weaken: the stale-workspace rule is Create's."""
    from realmspinner.service.errors import Invalid
    from realmspinner.studio import main as main_mod
    from realmspinner.studio.state import AppState

    state = AppState()
    state.create.workspace = "creation:now"
    done = SimpleNamespace(
        key="submit",
        ok=False,
        result=None,
        error=Invalid("bad seed", field="seed"),
        message="bad seed",
        action=None,
        tag="creation:before",
    )
    app = main_mod.App.__new__(main_mod.App)
    app.app_ctx = SimpleNamespace(
        toast=lambda *a, **k: None,
        tasks=SimpleNamespace(poll=lambda: [done]),
        state=state,
        svc=None,
    )

    app._collect_tasks()

    assert "seed" not in state.field_errors


# --- shell-30: Ctrl+S / Ctrl+W / Ctrl+F are not Create's viewer toggles ---------


def _press_in_create(key: int, mod: int) -> list[str]:
    """One KEYDOWN through ``App._shortcut`` in Create; -> which viewer toggles fired."""
    from types import MethodType

    from realmspinner.studio import main
    from realmspinner.studio.state import AppState

    fired: list[str] = []
    state = AppState()
    state.mode = state.mode_observed = state.previous_mode = "create"
    app = SimpleNamespace(
        app_ctx=SimpleNamespace(state=state, cache=SimpleNamespace(get=lambda _id: None)),
        viewer=SimpleNamespace(
            pose_mode=False,
            frame=lambda: fired.append("frame"),
            set_wireframe=lambda v: fired.append("wireframe"),
            set_turntable=lambda v: fired.append("turntable"),
            exit_compare=lambda: None,
        ),
    )
    for name in ("_note_mode", "_set_mode", "_escape_mode"):
        setattr(app, name, MethodType(getattr(main.App, name), app))
    main.App._shortcut(app, pygame.event.Event(pygame.KEYDOWN, key=key, mod=mod))
    return fired


@pytest.mark.parametrize("key", [pygame.K_s, pygame.K_w, pygame.K_f])
@pytest.mark.parametrize("mod", [pygame.KMOD_CTRL, pygame.KMOD_ALT, pygame.KMOD_META])
def test_a_ctrl_chord_on_s_w_or_f_does_not_fire_the_create_viewer_toggles(key, mod):
    fired = _press_in_create(key, mod)

    assert fired == [], f"{pygame.key.name(key)} with a chord fired {fired}"


@pytest.mark.parametrize("key", [pygame.K_s, pygame.K_w, pygame.K_f])
def test_plain_s_w_and_f_still_drive_the_create_viewer(key):
    assert _press_in_create(key, 0) != []


# --- shell-37: an unpaused Create animation preview keeps the full cadence ------


def _frame_probe(monkeypatch):
    from realmspinner.studio.shell import frame as frame_mod
    from realmspinner.studio.state import AppState

    monkeypatch.setattr(pygame.event, "peek", lambda: False)

    class _Cache:
        active = None

    class _Tasks:
        busy_keys = ()

    state = AppState()
    state.mode = "create"
    obj = frame_mod.FrameMixin.__new__(frame_mod.FrameMixin)
    obj.app_ctx = SimpleNamespace(state=state, cache=_Cache(), tasks=_Tasks())
    obj.runtime = SimpleNamespace(current_job_id=None)
    obj.viewer = None
    obj.clay_view = None
    obj.poser_viewer = None
    obj.mason_view = None
    return obj


def test_an_unpaused_create_animation_preview_keeps_the_frame_at_full_cadence(monkeypatch):
    from imgui_bundle import imgui

    from realmspinner.studio.modes.create.ui import preview

    imgui_ctx = imgui.create_context()
    try:
        probe = _frame_probe(monkeypatch)
        state = probe.app_ctx.state
        assert probe._frame_active() is False, "nothing is animating yet"

        # What ``preview.draw`` records when it draws a playing animation.
        preview.note_animation_frame(state.preview, imgui.get_frame_count(), paused=False)
        assert probe._frame_active() is True

        # Paused: a still image needs no cadence.
        preview.note_animation_frame(state.preview, imgui.get_frame_count(), paused=True)
        assert probe._frame_active() is False

        # Drawn on an earlier frame and not since (the user left the stage).
        preview.note_animation_frame(state.preview, imgui.get_frame_count() - 5, paused=False)
        assert probe._frame_active() is False

        # And only in Create.
        preview.note_animation_frame(state.preview, imgui.get_frame_count(), paused=False)
        state.mode = "library"
        assert probe._frame_active() is False
    finally:
        imgui.destroy_context(imgui_ctx)


# --- the "four SDXL recipes" count in Settings' remove control -----------------


def test_the_remove_control_counts_the_sdxl_recipes_from_the_registry():
    from realmspinner.models import BASE_MODELS
    from realmspinner.studio.modes.settings.ui.panes import app_settings

    sdxl_dir = BASE_MODELS["sdxl_cfg"].dir_name
    count = len([m for m in BASE_MODELS.values() if m.dir_name == sdxl_dir])
    word = {4: "four", 5: "five", 6: "six", 7: "seven"}[count]
    doc = re.sub(r"\s+", " ", app_settings._remove_control.__doc__ or "")

    assert f"one of the {word} SDXL 1.0 recipes" in doc
