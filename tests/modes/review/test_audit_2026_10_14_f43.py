"""The 2026-10-14 audit's Review findings shell-30, -31, -60, -62, -63, -33 and -61.

One dated file with a name unique across the tree. Each test's name is the
claim and each failed against the code as the audit found it.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
from _ui_context import imgui_context

from modes.review.test_review_mode import FakeCtx, _Done, _mesh, _scanned, _sweep
from realmspinner import models
from realmspinner.studio import probe
from realmspinner.studio.modes.review import mode as review_mode
from realmspinner.studio.modes.review.ui import workspace as review_panes


@pytest.fixture
def ctx(svc):
    return FakeCtx(svc)


@pytest.fixture
def ui(monkeypatch):
    with imgui_context(monkeypatch) as imgui:
        yield imgui


ENGINE_RUNTIME = "engine:trellis_runtime"
ENGINE_GGUF = "engine:trellis_gguf"


def _engine_rows(*, present: bool) -> list[dict]:
    return [
        {"row_key": ENGINE_RUNTIME, "label": "3D engine runtime", "present": present,
         "size_gib": 0.5},
        {"row_key": ENGINE_GGUF, "label": "3D engine weights", "present": present,
         "size_gib": 6.0},
    ]


def _ready_form(ctx):
    form = review_mode.ensure(ctx).form
    form.prompt = "a wooden chest"
    form.seeds = "1"
    form.axes = [{"param": "trellis_band", "values": "4, 8"}]
    return form


# --- shell-30 (shell-review-settings-03) -------------------------------------


def test_capture_base_carries_custom_triangles_with_the_custom_profile(ctx):
    """The Custom budget's count lives in ``custom_triangles``; capturing the
    profile name alone made planning refuse "custom triangles must be
    5000-250000" for a field the sweep form never draws."""
    ctx.state.form_3d["profile"] = "custom"
    ctx.state.form_3d["custom_triangles"] = 12000

    base = review_mode.capture_base(ctx)

    assert base["profile"] == "custom"
    assert base["custom_triangles"] == 12000


def test_capture_base_leaves_an_untouched_or_out_of_range_custom_count_out(ctx):
    ctx.state.form_3d["profile"] = "custom"
    ctx.state.form_3d["custom_triangles"] = 0
    assert "custom_triangles" not in review_mode.capture_base(ctx)

    # A stale count under a named profile is not the sweep's to carry.
    ctx.state.form_3d["profile"] = "raw"
    ctx.state.form_3d["custom_triangles"] = 12000
    assert "custom_triangles" not in review_mode.capture_base(ctx)


def test_a_sweep_from_the_custom_baseline_passes_the_profile_check(ctx, svc, monkeypatch):
    from realmspinner.service import jobs as jobs_mod
    from realmspinner.service import sweeps as sweeps_mod

    # The gltfpack-presence gate is not this finding's subject.
    seen: list[tuple] = []

    def fake_resolve(svc_, params, profile, custom):
        from realmspinner.pipelines import optimize

        seen.append((profile, custom))
        return optimize.resolve(profile, custom)

    monkeypatch.setattr(jobs_mod, "resolve_profile", fake_resolve)
    ctx.state.form_3d["profile"] = "custom"
    ctx.state.form_3d["custom_triangles"] = 12000
    form = _ready_form(ctx)
    form.base = review_mode.capture_base(ctx)

    sweeps_mod.validate_sweep(ctx.svc, review_mode.build_plan(ctx.state.review))

    assert seen and seen[0] == ("custom", 12000)


# --- shell-31 (shell-review-settings-04) -------------------------------------


def test_launch_refuses_a_model_sweep_when_the_engine_is_not_downloaded(ctx, svc):
    _ready_form(ctx)
    ctx.model_rows = _engine_rows(present=False)

    assert review_mode.launch(ctx) is False

    assert review_mode.LAUNCH_KEY not in ctx.submitted
    assert svc.store.list_sweeps() == []
    assert ctx.toasts and ctx.toasts[-1][1] == "error"
    assert "not downloaded" in ctx.toasts[-1][0]
    # The refusal carries the rows, so the Install button pre-ticks them.
    rows = next(iter(ctx.state.field_error_rows.values()))
    assert set(rows) == {ENGINE_RUNTIME, ENGINE_GGUF}


def test_launch_goes_through_when_the_engine_is_downloaded(ctx, svc):
    _ready_form(ctx)
    ctx.model_rows = _engine_rows(present=True)

    assert review_mode.launch(ctx) is True


def test_the_engine_gate_names_the_stale_check_weights_comment_no_more():
    """``check_weights`` said the engine rows were fatal doctor rows that a
    startup banner covers. They stopped being that on 2026-09-10, when the
    engine became a Settings download; the comment now says who checks."""
    from realmspinner.service import validation

    doc = inspect.getdoc(validation.check_weights) or ""
    assert "two fatal doctor rows" not in doc


# --- shell-60 (shell-review-settings-08) -------------------------------------


def test_regrading_a_unit_judged_before_the_pass_never_drives_the_pass_tally_negative(
    ctx, svc
):
    """``record`` took any previous verdict for one this pass had filed, so an
    older accept re-graded inside the pass took ``accepted`` to -1."""
    _mesh(svc, "a chest")
    _mesh(svc, "a sword")
    state = _scanned(ctx)
    # One unit was judged last week, outside any pass.
    old = state.units[0]
    review_mode.record(ctx, 3)
    assert old["verdict"] == "accept"
    review_mode.start_judging(ctx)
    judging = state.judging
    assert judging.filed == 0

    review_mode.step(state, state.units.index(old) - state.index)
    review_mode.record(ctx, -3)

    assert judging.accepted >= 0
    assert judging.rejected >= 0
    assert judging.filed == 0
    assert judging.accepted == 0 and judging.rejected == 0


# --- shell-62 (shell-review-settings-10) -------------------------------------


def _refused_for_a_missing_model(ctx, monkeypatch):
    from realmspinner import fetch

    monkeypatch.setattr(fetch, "base_model_state", lambda *a, **k: (False, None))
    missing = "sdxl_cfg_pag"
    assert missing in models.BASE_MODELS
    form = _ready_form(ctx)
    form.base = {"base_model": missing}
    assert review_mode.launch(ctx) is False
    return missing


def test_a_sweep_refused_for_a_missing_model_offers_the_install_button(ctx, svc, ui, monkeypatch):
    from realmspinner.studio import widgets

    monkeypatch.setattr(widgets, "FORCE_SECTIONS_OPEN", True)
    _refused_for_a_missing_model(ctx, monkeypatch)
    rows = ctx.state.field_error_rows.get("base_model")
    # The refusal's own rows reach the state, which is what the task-failure
    # path already did for Create.
    assert rows

    probe.begin_frame()
    ui.new_frame()
    ui.set_next_window_size((600.0, 900.0))
    ui.set_next_window_pos((0.0, 0.0))
    ui.begin("##host")
    review_panes.ReviewPanes()._review_form(ctx, ctx.state.review, review_mode)
    ui.end()
    ui.end_frame()

    labels = [c.label for c in probe.FRAME_CONTROLS if c.kind == "button"]
    assert any(label.startswith("Install ") for label in labels), labels


# --- shell-63 (shell-review-settings-11) -------------------------------------


def _two_open(ctx, svc, monkeypatch):
    from realmspinner.service import sweeps as sweeps_mod

    # The removal itself is the service's and is tested there; what is under
    # test is what Review forgets when the task lands.
    monkeypatch.setattr(sweeps_mod, "remove_reviewed_sweeps", lambda *a, **kw: {})
    alpha, _ = _sweep(svc, "alpha")
    beta, _ = _sweep(svc, "beta")
    state = _scanned(ctx)
    return state, alpha, beta


def test_bulk_removal_keeps_an_open_sweep_that_was_not_removed(ctx, svc, monkeypatch):
    state, alpha, beta = _two_open(ctx, svc, monkeypatch)
    state.sweep_id = alpha
    assert review_mode.remove_reviewed(ctx, [beta]) is True

    review_mode.on_task_done(
        ctx, _Done(review_mode.REMOVE_KEY, {"deleted": 1, "sweeps": 1}, tag="1 sweep(s)")
    )

    assert state.sweep_id == alpha


def test_bulk_removal_forgets_an_open_sweep_that_was_removed(ctx, svc, monkeypatch):
    state, alpha, beta = _two_open(ctx, svc, monkeypatch)
    state.sweep_id = alpha
    assert review_mode.remove_reviewed(ctx, [alpha, beta]) is True

    review_mode.on_task_done(
        ctx, _Done(review_mode.REMOVE_KEY, {"deleted": 2, "sweeps": 2}, tag="2 sweep(s)")
    )

    assert state.sweep_id != alpha


# --- shell-33 (shell-review-settings-06) -------------------------------------


def test_grade_help_does_not_tell_a_judging_pass_reviewer_to_press_r_first():
    """Inside a guided pass R is Reject, which files -3 at once. The Grade
    field's help said "press R first for a negative grade" there too."""
    outside = review_mode.grade_help(judging=False)
    inside = review_mode.grade_help(judging=True)

    assert "R first" in outside
    assert "R first" not in inside
    assert "reject" in inside.lower()


# --- shell-61 (shell-review-settings-09) -------------------------------------


def test_delete_sweep_icon_greys_with_the_scan_reason(ui):
    state = SimpleNamespace(scanning=True, drop_retained=False)
    sweep = {"id": "s1", "label": "alpha", "units": []}
    ctx = SimpleNamespace()

    probe.begin_frame()
    ui.new_frame()
    ui.set_next_window_size((400.0, 200.0))
    ui.set_next_window_pos((0.0, 0.0))
    ui.begin("##host")
    review_panes.ReviewPanes()._review_delete_button(ctx, state, review_mode, sweep)
    ui.end()
    ui.end_frame()

    trash = [c for c in probe.FRAME_CONTROLS if "delete-s1" in c.label]
    assert trash and not trash[0].enabled
    assert trash[0].reason == "A scan is already running."


# --- shell-62: the rows survive the per-unit re-raise -------------------------


def test_a_sweep_refusal_keeps_its_rows_through_the_unit_wrapper(svc, monkeypatch):
    """``sweeps._validate`` re-raised a unit's refusal with the message and the
    field only, so the rows ``check_weights`` attached never left the service."""
    from realmspinner import fetch
    from realmspinner.service import sweeps as sweeps_mod
    from realmspinner.service.errors import Invalid

    monkeypatch.setattr(fetch, "base_model_state", lambda *a, **k: (False, None))
    plan = sweeps_mod.SweepPlan(
        label="x", prompt="a chest", base={"base_model": "sdxl_cfg_pag"}, seeds=(1,),
        axes=(sweeps_mod.Axis("trellis_band", (4, 8)),),
    )

    with pytest.raises(Invalid) as caught:
        sweeps_mod.validate_sweep(svc, plan)

    assert caught.value.field == "base_model"
    assert caught.value.rows == ("base:sdxl_cfg_pag",)


def test_engine_problem_names_the_rows_it_found_missing():
    from realmspinner.studio.modes.create.engine import mesh as create_mesh

    problem = create_mesh.engine_problem(_engine_rows(present=False))
    assert problem is not None
    assert problem.rows == (ENGINE_RUNTIME, ENGINE_GGUF)
    assert create_mesh.engine_problem(_engine_rows(present=True)) is None
