"""The 2026-10-03 audit's Low findings for Create's panes, brief and workspace
(fixer create1): create-22/23/24/32/33/34/35/36/37/38/40/43/44/45/46/48.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

# --- create-22 / create-37: the Earlier meshes panel ---------------------------


class _FakeConfirms:
    def __init__(self) -> None:
        self.asked: list[Any] = []

    def ask(self, confirm: Any) -> None:
        self.asked.append(confirm)


def _history_ctx() -> SimpleNamespace:
    return SimpleNamespace(
        confirms=_FakeConfirms(),
        svc=object(),
        submit=lambda key, fn, *a, **k: True,
    )


def test_restore_on_an_entry_without_a_geometry_flag_still_asks():
    """A model_history row with an int ``n`` but no ``geometry`` key is listed
    by the panel, and pressing Restore raised KeyError('geometry') on the
    frame thread instead of opening the confirm."""
    from realmspinner.studio.panes import history_panel

    job = {
        "id": "j1",
        "params": {"model_history": [{"n": 1, "kind": "optimize"}]},
    }
    ctx = _history_ctx()
    entry = history_panel.entries_for(job)[0]
    history_panel._confirm_restore(ctx, job, entry)
    assert len(ctx.confirms.asked) == 1


def test_the_restore_confirm_names_the_mesh_as_it_was_before_the_step():
    """A row's label names the step that *replaced* that version, so the mesh
    restored is the one from before that step. "The one from triangle
    budget" read as the product of the step -- the opposite."""
    from realmspinner.studio.panes import history_panel

    job = {
        "id": "j1",
        "params": {
            "model_history": [{"n": 1, "kind": "optimize", "geometry": True}]
        },
    }
    ctx = _history_ctx()
    history_panel._confirm_restore(ctx, job, history_panel.entries_for(job)[0])
    message = ctx.confirms.asked[0].message
    assert "before" in message
    assert "the one from triangle budget" not in message.lower()


# --- create-23: Automatic says what it cleared ---------------------------------


def test_switching_to_automatic_shows_what_it_cleared(monkeypatch):
    """``clear_for_tier`` returns one sentence per selection it cleared and
    the Model combo's Automatic branch threw them away: typed Avoid text and
    a structure pick vanished with no notice."""
    from realmspinner import generation
    from realmspinner.studio.modes.create.engine import assets as create_assets
    from realmspinner.studio.modes.create.engine import recipe as create_recipe
    from realmspinner.studio.modes.create.ui.panes import settings_2d
    from realmspinner.studio.state import AppState, default_form_2d

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
    shown: list[str] = []
    monkeypatch.setattr(settings_2d.widgets, "field_label", lambda *a, **k: None)
    monkeypatch.setattr(settings_2d.widgets, "combo", lambda *a, **k: "")
    monkeypatch.setattr(settings_2d.widgets, "field_error", lambda *a, **k: None)
    monkeypatch.setattr(settings_2d.widgets, "muted_wrapped", lambda t, *a, **k: shown.append(t))
    monkeypatch.setattr(settings_2d.widgets, "wrapped", lambda *a, **k: None)

    form = default_form_2d()
    create_assets.sync_legacy_fields(form)
    form["model_mode"] = "advanced"
    form["base_model"] = "sdxl_cfg"
    form["control"] = "canny"
    form["negative_prompt"] = "no hats"
    ctx = SimpleNamespace(state=AppState(), svc=SimpleNamespace(config=None), base_models=[])

    settings_2d._model(ctx, form)

    assert form["negative_prompt"] == ""
    text = " ".join(shown)
    assert "Avoid text was cleared" in text
    assert "structure control was cleared" in text


# --- create-24: Reset clears the refusals recorded against the discarded form --


def test_reset_settings_clears_the_recorded_refusals():
    """Reset replaced ``form_2d`` / ``form_3d`` and left ``field_errors``
    ringing against the form it had just discarded, so "The prompt is over
    1000 characters." kept ringing an empty prompt."""
    from realmspinner.studio.modes.create.ui.panes import settings_2d, settings_3d
    from realmspinner.studio.state import AppState

    for reset in (settings_2d._reset, settings_3d._reset):
        state = AppState()
        state.note_field_error("prompt", "The prompt is over 1000 characters.")
        ctx = SimpleNamespace(state=state, toast=lambda *a, **k: None)
        reset(ctx)
        assert state.field_errors == {}, reset.__module__


# --- create-32: a busy submit gives the dead Generate a reason ------------------


def test_a_generate_disabled_by_a_running_submit_says_so(monkeypatch):
    """Generate was disabled with ``reason=""`` while ``ctx.busy("submit")``
    held (and no problem existed), so the greyed primary explained nothing on
    hover for the length of the submit task."""
    from _ui_context import imgui_context

    from realmspinner.studio.modes.create.ui import brief as create_brief
    from realmspinner.studio.state import AppState

    state = AppState(mode="create")
    state.create.stage = "reference"
    state.form_2d["prompt"] = "a mossy well"
    ctx = SimpleNamespace(
        state=state,
        svc=SimpleNamespace(config=None),
        busy=lambda key: key == "submit",
        confirms=SimpleNamespace(ask=lambda c: None),
        cache=SimpleNamespace(get=lambda _id: None, jobs=[], active=None),
        job=lambda: None,
        textures=None,
        model_rows=[],
        submit=lambda *a, **k: True,
        toast=lambda *a, **k: None,
    )
    seen: list[tuple[bool, str]] = []
    real = create_brief.widgets.primary_button

    def spy(label, *a, enabled=True, reason="", **k):
        seen.append((enabled, reason))
        return real(label, *a, enabled=enabled, reason=reason, **k)

    monkeypatch.setattr(create_brief.widgets, "primary_button", spy)
    with imgui_context(monkeypatch) as imgui:
        imgui.new_frame()
        imgui.begin("test")
        create_brief.submit_control(ctx)
        imgui.end()
        imgui.render()
    enabled, reason = seen[0]
    assert enabled is False
    assert reason.strip(), "a greyed Generate with no reason explains nothing on hover"


# --- create-33: no dead layout machinery in the brief ---------------------------


def test_every_function_in_brief_is_reached_from_a_draw_path():
    """The bar became a header and ``draw`` stopped calling ``shows``,
    ``_row_widths``, ``_count_width``, ``_reset``, ``_rail_measurements`` and
    friends, but they stayed -- and tests pinned the dead ladder and the dead
    second Reset instead of the live one in ``inputs``."""
    import ast
    import inspect

    from realmspinner.studio.modes.create.ui import brief as create_brief

    tree = ast.parse(inspect.getsource(create_brief))
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

    def names_in(node: ast.AST) -> set[str]:
        return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}

    # What the shell and the two settings columns call from outside.
    reached = {"draw", "inputs", "submit_control", "bar_height"}
    # Module-level statements (``_TYPE_HINTS`` calls ``_species_count``) run at
    # import, so what they name is reached too.
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            reached |= names_in(node) & funcs.keys()
    frontier = list(reached)
    while frontier:
        fn = funcs[frontier.pop()]
        for name in names_in(fn) & funcs.keys():
            if name not in reached:
                reached.add(name)
                frontier.append(name)
    assert set(funcs) - reached == set(), sorted(set(funcs) - reached)


# --- create-45: the refusal says where the picker is -----------------------------


def test_pending_candidates_refusal_points_at_the_stage_that_draws_the_picker():
    """Reference refuses Generate with "Decide the pending candidates first."
    while the picker is drawn only on Mesh, so the sentence named no
    destination."""
    from realmspinner.studio.modes.create.ui import brief as create_brief

    jobs = [
        {"id": "a1", "candidate_group": "g", "candidate_index": 0,
         "status": "done", "created_at": 1.0},
        {"id": "a2", "candidate_group": "g", "candidate_index": 1,
         "status": "done", "created_at": 1.0},
    ]
    ctx = SimpleNamespace(cache=SimpleNamespace(jobs=jobs))
    problems = create_brief._with_pending_candidates_problem(ctx, [])
    assert problems and "Mesh" in str(problems[0])


# --- create-35: one default for the mesh finishing ------------------------------


def test_a_form_without_mesh_finishing_sends_the_documented_default():
    """``engine_kwargs`` / ``upload_kwargs`` and the Finishing combo fell back
    to "repair" for a form lacking the key, while ``DEFAULT_FORM_3D``,
    ``generation.ModelSettings`` and the service doors default to
    "preserve_shape"."""
    from realmspinner import generation
    from realmspinner.studio.modes.create.engine import mesh as create_mesh
    from realmspinner.studio.state import DEFAULT_FORM_3D

    form = dict(DEFAULT_FORM_3D)
    del form["mesh_finishing"]
    documented = DEFAULT_FORM_3D["mesh_finishing"]
    assert documented == generation.ModelSettings().mesh_finishing
    assert create_mesh.engine_kwargs(form)["mesh_finishing"] == documented
    assert create_mesh.upload_kwargs(form)["mesh_finishing"] == documented


# --- create-36: guidance's docstring names the per-job engine axes ---------------


def test_guidance_docstring_names_every_per_job_engine_axis():
    """The module docstring said trellis-server "takes only an image, a seed
    and a geometry resolution" and that texture resolution is deliberately not
    per-request; Create now sends eight engine axes per job."""
    from realmspinner import guidance
    from realmspinner.studio.modes.create.engine import mesh as create_mesh
    from realmspinner.studio.state import DEFAULT_FORM_3D

    form = dict(DEFAULT_FORM_3D)
    form.update(
        trellis_band=4, trellis_tex_res=512, trellis_gss=1.0, trellis_gsh=1.0,
        trellis_max_tokens=100, trellis_decim=0, trellis_atlas=1024,
    )
    axes = set(create_mesh.engine_kwargs(form))
    assert {"trellis_tex_res", "mesh_finishing"} <= axes
    doc = guidance.__doc__ or ""
    for axis in sorted(axes):
        assert axis in doc, f"guidance's docstring never names {axis}"
    assert "takes only an image" not in doc


# --- create-34: manual 22 names controls the column draws ------------------------


def test_manual_22_names_only_controls_the_reference_column_draws():
    """Chapter 22 named "the large text box under **Prompt**", **Generation
    type**, "the **Seed** section" and **Recent prompts...** "under the seed
    row": none of those words is on screen in the redesigned brief."""
    import inspect
    from pathlib import Path

    from realmspinner.studio.modes.create.ui import brief as create_brief
    from realmspinner.studio.modes.create.ui.panes import settings_2d

    manual = Path(__file__).parents[3] / "docs" / "manual" / "22-generating-references.md"
    text = manual.read_text(encoding="utf-8")
    for stale in (
        "under **Prompt**",
        "**Generation type**",
        "**Seed** section",
        "under the seed row",
    ):
        assert stale not in text, stale
    drawn = inspect.getsource(create_brief) + inspect.getsource(settings_2d)
    for label in ("Your brief", "Describe the asset", "What are you making?", "Recent prompts"):
        assert label in drawn, label
        assert label in text, label
    # And the order the manual states: the history button is drawn before the
    # Advanced header that holds the seed row.
    draw = inspect.getsource(settings_2d.draw)
    assert draw.index("_history(ctx, form)") < draw.index("##create-generation")


# --- create-38: Re-texture names a missing ControlNet before it is pressed -------


def test_the_retexture_button_names_a_missing_depth_controlnet_before_it_is_pressed(
    monkeypatch,
):
    """The Re-texture button was enabled whatever was installed (anchor on by
    default) and the door's refusal carries a field no control here rings."""
    from _ui_context import imgui_context

    from realmspinner.studio import widgets
    from realmspinner.studio.panes import texture_panel

    seen: list[tuple[bool, str]] = []
    real = widgets.disabled_button

    def spy(label, enabled, size=(0, 0), *, reason="", **k):
        seen.append((enabled, reason))
        return real(label, enabled, size, reason=reason, **k)

    monkeypatch.setattr(texture_panel.widgets, "disabled_button", spy)
    ctx = SimpleNamespace(
        svc=SimpleNamespace(config=SimpleNamespace(t2i_model="sdxl_cfg")),
        state=SimpleNamespace(field_errors={}, clear_field_errors=lambda: None),
        busy=lambda key: False,
        cache=SimpleNamespace(jobs=[]),
        submit=lambda *a, **k: True,
        model_rows=[
            {"row_key": "base:sdxl_cfg", "present": True, "size_gib": 6.0},
            {"row_key": "control:depth", "present": False, "size_gib": 2.5},
        ],
        model_picks=set(),
    )
    form = {
        "prompt": "rusted iron", "strength": 0.5, "texture_size": "",
        "depth": True, "control_scale": 1.0,
    }
    assert "control:depth" in texture_panel.required_rows(ctx, form)
    with imgui_context(monkeypatch) as imgui:
        imgui.new_frame()
        imgui.begin("test")
        texture_panel._submit(ctx, "job1", form)
        imgui.end()
        imgui.render()
    enabled, reason = seen[-1]
    assert enabled is False
    assert reason.strip()


# --- create-40: no unused footer-height state in the Create columns --------------


def test_the_create_columns_do_not_keep_unused_footer_height_state():
    """``_submit_px`` / ``_footer_px`` were lists nothing read or wrote, and
    the docstrings promised a frame-late height feedback that does not run."""
    from realmspinner.studio.modes.create.ui.panes import settings_2d, settings_3d

    assert not hasattr(settings_2d, "_submit_px")
    assert not hasattr(settings_3d, "_footer_px")
    assert "No press is drawn here" not in (settings_3d.draw.__doc__ or "")


# --- create-44: the Creations list is memoised with the index --------------------


def test_creations_history_is_not_rebuilt_on_a_frame_that_changed_nothing(monkeypatch):
    """``history`` sorted every family by its newest created_at and ran
    ``families.results`` over each one on every frame the Creations list was on
    screen, although the index beneath it is memoised on the cache generation."""
    from _ui_context import imgui_context

    from realmspinner.studio.modes.create.engine import workspace as families
    from realmspinner.studio.modes.create.ui import workspace as create_workspace
    from realmspinner.studio.state import AppState

    jobs = [
        {"id": f"j{i}", "kind": "text", "stage": "reference", "status": "done",
         "created_at": float(i), "files": [], "params": {}}
        for i in range(5)
    ]
    cache = SimpleNamespace(
        jobs=jobs, _generation=1, can_load_more=lambda: False, get=lambda _id: None,
    )
    ctx = SimpleNamespace(state=AppState(mode="create"), cache=cache)
    calls: list[int] = []
    real = families.results
    monkeypatch.setattr(
        families, "results", lambda *a, **k: calls.append(1) or real(*a, **k)
    )
    with imgui_context(monkeypatch) as imgui:
        for _frame in range(2):
            imgui.new_frame()
            imgui.begin("test")
            create_workspace.history(ctx)
            imgui.end()
            imgui.render()
            if _frame == 0:
                after_first = len(calls)
    assert after_first >= 5, "the first frame has to build the list"
    assert len(calls) == after_first, "an unchanged second frame rebuilt it"


# --- create-46: the tray's stage filter is asserted on what draw calls ----------


def test_tray_stage_filter_is_asserted_on_the_function_draw_calls():
    """``_recent_results`` / ``_in_stage`` / ``_result_grid`` / ``_brief_caption``
    were called by nothing and tests guarded them instead of
    ``families.results``, which ``draw`` and ``should_draw`` actually read."""
    import inspect

    from realmspinner.studio.modes.create.engine import workspace as families
    from realmspinner.studio.modes.create.ui import workspace as create_workspace

    for gone in ("_recent_results", "_in_stage", "_result_grid", "_brief_caption",
                 "_RESULT_COLUMNS"):
        assert not hasattr(create_workspace, gone), gone
    assert "families.results(" in inspect.getsource(create_workspace.draw)
    assert "_recent_results" not in (create_workspace.should_draw.__doc__ or "")

    jobs = [
        {"id": "ref", "kind": "text", "stage": "reference", "created_at": 1.0, "params": {}},
        {"id": "mesh", "kind": "image", "stage": "model", "created_at": 2.0,
         "parent_id": "ref", "params": {}},
        {"id": "rig", "kind": "rig", "stage": "model", "created_at": 3.0,
         "parent_id": "mesh", "params": {"source_job": "mesh"}},
    ]
    index = families.build_index(jobs)
    (key,) = index.families
    assert [j["id"] for j in families.results(index, key, "reference")] == ["ref"]
    assert [j["id"] for j in families.results(index, key, "mesh")] == ["mesh"]
    assert [j["id"] for j in families.results(index, key)] == ["mesh", "ref"]


# --- create-48: Vary on a reference raises one toast ----------------------------


def test_vary_on_a_reference_raises_one_toast(monkeypatch):
    """``_vary`` called ``library.copy_settings`` (toast "Settings copied to the
    form." plus its own stage switch) and then toasted and switched again, so
    one press raised two toasts and the first switch could walk the selection
    that ``follow=False`` was written to prevent."""
    from realmspinner.studio.modes.create.ui import stages as create_stages
    from realmspinner.studio.modes.create.ui import workspace as create_workspace
    from realmspinner.studio.state import AppState

    toasts: list[str] = []
    switches: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        create_stages, "go",
        lambda ctx, stage, *, select=None, follow=True: switches.append((stage, follow)),
    )
    ctx = SimpleNamespace(state=AppState(mode="create"), toast=lambda t, *a, **k: toasts.append(t))
    job = {"id": "r1", "stage": "reference", "kind": "text", "params": {"prompt": "a chest"}}
    create_workspace._vary(ctx, job)
    assert len(toasts) == 1, toasts
    assert toasts[0].startswith("Loaded this brief")
    assert switches and all(follow is False for _stage, follow in switches), switches
    assert ctx.state.form_2d["prompt"] == "a chest"
