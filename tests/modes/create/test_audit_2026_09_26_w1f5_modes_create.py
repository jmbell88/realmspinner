"""Findings closed from the 2026-09-26 audit, Create-mode slice (w1f5).

create-panes-01: an emptied Avoid box reached the queue door as the default
negative prompt instead of the explicit empty string the user asked for,
because ``settings_2d.generate``'s Automatic-routing branch wrapped
``generation.effective_negative_prompt`` in ``or None`` -- exactly the bug
the 2026-09-03 fix (``tests/modes/create/test_create_fixes_2026_09_03.py``)
closed for ``submit_kwargs``, reopened one call site over in the branch that
overrides it for a resolved recipe.

create-panes-02 (+create-workspace-05): ``create_recipe.resolved_recipe``'s
memo was keyed on the request's contents plus ``id(config)`` only, so a
download or delete landing between two otherwise-identical frames (the
form itself never changes) left a stale "No compatible installed recipe"
answer until some unrelated field edit finally changed the memo key.

create-brief-01: ``create_brief._row_widths`` measured the rail's "full"
width with ``create_rail.stage_rail_width(items, current)`` -- no ``done`` at
all -- so ``_rail_fit``'s ticks rung measured every stage as its plain label,
identical to the labels rung, and the returned width never accounted for the
check glyph a real done stage actually draws. The real rail is later handed
that undersized width as its own budget and, this time with the *real* done
set, always found the true (wider) ticks rung too wide for it and dropped
the checks -- even on a row with genuine room to spare.

create-panes-03: ``settings_3d._matte_body`` drew "Cutting the subject
out..." with Accept *and* Fix matte both disabled ("still being prepared")
whenever ``state.preview is None`` -- true both while the cutout is still in
flight and after ``on_task_failed`` has latched a definitive failure
(``failed_stamp``/``_tried_and_failed``), so a failed cutout looked
identical to a slow one and offered no way forward: Fix matte reads only
``state.job_id`` (``matte_preview.fix``), so there was never a reason to
disable it on a failure.

create-panes-04: ``_target_cell`` computed ``known`` (what the combo shows)
straight from ``form["target_cell_px"]`` every frame, with no memory of
having just picked "Custom" -- so picking it wrote nothing to the form until
the number field below was actually edited, and the very next frame recomputed
``known`` from the unchanged old value and snapped the combo back.

create-panes-06: the Model combo's own Automatic branch (``_model``'s
``else``) set ``model_mode``/``model_override`` but never called
``create_recipe.clear_for_tier`` -- the preflight banner's "Switch to
Automatic" repair button, a few hundred lines down, writes the identical two
fields and does call it. A ControlNet chosen under Advanced survived the
switch, the picker that shows it hides once ``model_mode`` is "auto", and
``validate`` refused on a field with no control left on screen.
"""

from __future__ import annotations

from types import SimpleNamespace

from _ui_context import imgui_context

from realmspinner import generation
from realmspinner.studio import icons, matte_preview, probe
from realmspinner.studio.modes.create.engine import assets as create_assets
from realmspinner.studio.modes.create.engine import recipe as create_recipe
from realmspinner.studio.modes.create.ui import brief as create_brief
from realmspinner.studio.modes.create.ui import rail as create_rail
from realmspinner.studio.modes.create.ui.panes import settings_2d, settings_3d
from realmspinner.studio.state import AppState, default_form_2d


def _ctx():
    return SimpleNamespace(state=AppState(), svc=SimpleNamespace(config=None))


def test_generate_sends_an_emptied_avoid_box_as_an_explicit_empty_string(monkeypatch):
    form = default_form_2d()
    create_assets.sync_legacy_fields(form)
    form["prompt"] = "a chest"
    # Deliberately emptied by the user -- not "unset".
    form["negative_prompt"] = ""

    captured: dict = {}

    def fake_create_job(svc, **kwargs):
        captured.update(kwargs)
        return {"id": "job1"}

    monkeypatch.setattr(settings_2d.svc_jobs, "create_job", fake_create_job)

    def _submit(key, run):
        run()
        return True

    toasts: list = []
    ctx = SimpleNamespace(
        state=AppState(),
        svc=SimpleNamespace(config=None),
        submit=_submit,
        toast=toasts.append,
    )

    settings_2d.generate(ctx, form)

    assert not toasts, f"unexpected refusal: {toasts}"
    assert "negative_prompt" in captured, "generate() never reached the create_job door"
    assert captured["negative_prompt"] == "", (
        "an emptied Avoid box must reach the door as an explicit empty "
        f"string, got {captured['negative_prompt']!r}"
    )


def _counting_resolve_recipe(calls: list[int]):
    original = generation.resolve_recipe

    def counting(request, config=None, *, installed=None):
        calls.append(1)
        return original(request, config, installed=installed)

    return counting, original


def test_resolved_recipe_re_resolves_after_a_download_lands_with_the_form_unchanged(
    monkeypatch,
):
    form = default_form_2d()
    create_assets.sync_legacy_fields(form)
    ctx = _ctx()
    # A row set the pane reads to know what is installed -- wholesale
    # replaced by the App whenever a fetch finishes (app_ctx.py's
    # ``model_rows`` field comment).
    ctx.model_rows = [{"row_key": "base:sdxl_cfg", "present": False}]

    calls: list[int] = []
    counting, original = _counting_resolve_recipe(calls)
    monkeypatch.setattr(generation, "resolve_recipe", counting)

    create_recipe.resolved_recipe(ctx, form)
    for _ in range(3):
        ctx.state.frame_index += 1
        create_recipe.resolved_recipe(ctx, form)
    assert len(calls) == 1, "expected one resolution while nothing changed"

    # A download lands between two frames; the form itself is untouched, and
    # so is ``ctx.svc.config`` (the same object for the app's whole life) --
    # only the installed set differs.
    ctx.model_rows = [{"row_key": "base:sdxl_cfg", "present": True}]
    ctx.state.frame_index += 1
    create_recipe.resolved_recipe(ctx, form)

    assert len(calls) == 2, (
        "resolved_recipe kept the pre-download memo answer after an "
        f"installed-set change with the form unchanged (calls={len(calls)})"
    )


def test_the_reference_stage_rail_keeps_its_ticks_when_the_row_has_room(monkeypatch):
    items = create_brief._rail_items_for_measurement()
    current = items[0][0]
    # A real done set: every other stage is actually finished. This is
    # exactly what ``App._stage_rail`` will pass as ``done`` when it actually
    # draws the rail -- the thing ``create_brief._rail_measurements``'s own
    # measurement must size for.
    real_done = frozenset(key for key, *_rest in items[1:])

    with imgui_context(monkeypatch) as imgui:
        imgui.new_frame()
        imgui.begin("test")

        # The production function under test, not a reimplementation of it.
        rail_full_w, _rail_floor_w = create_brief._rail_measurements(current)
        # The real draw, later, with the *real* done set and this measured
        # width as its own budget.
        shown, _widths, _titles, _done = create_rail._rail_fit(
            items, current, real_done, rail_full_w
        )

        imgui.end()
        imgui.render()

    labels_by_key = {key: label for key, label, _icon, _reason in items}
    for key in real_done:
        expected = f"{icons.CHECK} {labels_by_key[key]}"
        assert expected in shown, (
            f"{key!r} is done and the row was measured with room for its "
            f"check, but the real draw shows {shown!r}"
        )


def test_a_failed_cutout_modal_says_it_failed_and_keeps_fix_matte_enabled(monkeypatch):
    state = matte_preview.MatteState()
    state.job_id = "abc123456789"
    state.preview = None
    state.stamp = 5
    state.failed_stamp = 5  # on_task_failed's latch: this stamp will not be retried
    state._tried_and_failed = True

    ctx = SimpleNamespace(textures=None, svc=None)

    with imgui_context(monkeypatch) as imgui:
        imgui.new_frame()
        imgui.begin("test")
        probe.begin_frame()
        settings_3d._matte_body(ctx, state)
        imgui.end()
        imgui.render()

    fix = next(c for c in probe.FRAME_CONTROLS if c.label == "Fix matte")
    accept = next(c for c in probe.FRAME_CONTROLS if c.label in ("Accept", "Build anyway"))
    assert fix.enabled, (
        "Fix matte only reads state.job_id, never preview -- a failed cutout must not disable it"
    )
    assert not accept.enabled, "there is nothing to accept once the cutout has failed"
    assert accept.reason != "The cutout is still being prepared.", (
        "a definitively failed cutout must not say it is still being prepared"
    )


def test_choosing_custom_target_cell_keeps_the_custom_field_open(monkeypatch):
    from realmspinner.studio.modes.create.ui.panes import settings_2d

    form: dict = {"target_cell_px": ""}
    combo_shown: list[str] = []

    def fake_combo(name, before, values):
        combo_shown.append(before)
        return "custom"  # the user picks Custom every frame in this test

    class _FakeFormUI:
        def number(self, field, label, value, **kwargs):
            return False, value  # nothing typed into the number field yet

    monkeypatch.setattr(settings_2d.widgets, "combo", fake_combo)
    monkeypatch.setattr(settings_2d.widgets, "muted_wrapped", lambda *a, **k: None)

    ctx = SimpleNamespace()
    form_ui = _FakeFormUI()

    settings_2d._target_cell(ctx, form, form_ui)  # frame 1: the user picks Custom
    settings_2d._target_cell(ctx, form, form_ui)  # frame 2: still nothing typed

    assert combo_shown[-1] == "custom", (
        "the combo must keep showing Custom across frames until the user "
        f"picks something else, but it snapped back to {combo_shown[-1]!r}"
    )


def test_switching_the_model_combo_to_automatic_clears_a_structure_control_the_recipe_cannot_run(
    monkeypatch,
):
    from realmspinner.studio.modes.create.ui.panes import settings_2d

    # A recipe Automatic would resolve to that cannot run a ControlNet
    # (``turbo``'s own registry row has ``controlnet=False`` --
    # ``tests/modes/create/test_audit_2026_09_26_w1f5.py``'s own probe
    # confirms this against the real registry rather than assuming it).
    recipe = generation.Recipe(
        key="turbo_fast",
        label="Turbo",
        generation_types=("image", "3d_model"),
        quality="fast",
        base_model="turbo",
        negative_prompt=False,
    )
    resolved = generation.ResolvedRecipe(recipe=recipe, base_model="turbo", style_lora=None)
    monkeypatch.setattr(create_recipe, "resolved_recipe", lambda ctx, form: resolved)
    monkeypatch.setattr(settings_2d.widgets, "field_label", lambda *a, **k: None)
    monkeypatch.setattr(settings_2d.widgets, "combo", lambda *a, **k: "")  # picks Automatic
    monkeypatch.setattr(settings_2d.widgets, "field_error", lambda *a, **k: None)
    monkeypatch.setattr(settings_2d.widgets, "muted_wrapped", lambda *a, **k: None)
    monkeypatch.setattr(settings_2d.widgets, "wrapped", lambda *a, **k: None)

    form = default_form_2d()
    create_assets.sync_legacy_fields(form)
    form["model_mode"] = "advanced"
    form["base_model"] = "sdxl_cfg"
    form["control"] = "canny"  # a structure control chosen under Advanced
    ctx = SimpleNamespace(state=AppState(), svc=SimpleNamespace(config=None), base_models=[])

    settings_2d._model(ctx, form)

    assert form["model_mode"] == "auto"
    assert form["control"] == "", (
        "switching to Automatic must clear a structure control the resolved "
        f"recipe cannot run, but form['control'] is still {form['control']!r}"
    )

