"""Regressions for the 2026-09-07 Create review's friction sweep (item 5.3,
"the app forgets what the user typed", and item 5.5, prose pointing at
controls that no longer exist).

Each test's name is the claim, and pins the behaviour rather than the
implementation: a form persisted, a session-scoped cache kept per job, a
history file surviving a restart, a repair button changing the field its own
note names.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

from warlock import guidance as guidancelib
from warlock.studio import problems
from warlock.studio import settings as settings_mod
from warlock.studio.modes.create.engine import recipe as create_recipe
from warlock.studio.modes.create.ui.panes import settings_2d
from warlock.studio.state import MAX_HISTORY, AppState, default_form_2d


def _note_ctx(**extra):
    """``test_settings_2d_notes.py``'s ``_ctx()``, repeated: both structure
    notes read ``ctx.base_models``/``ctx.guidance`` to label the checkpoints
    they name, so a bare stub raises before the sentence under test is even
    built."""
    catalog = guidancelib.catalog()
    base = SimpleNamespace(
        guidance=catalog,
        base_models=[(m["key"], m["label"]) for m in catalog["fields"]["base_model"]],
    )
    for key, value in extra.items():
        setattr(base, key, value)
    return base


class _Ctx:
    """The sliver of ``ctx`` the functions under test read."""

    def __init__(self) -> None:
        self.state = AppState()
        self.toasts: list[tuple[str, str]] = []

    def toast(self, message: str, level: str = "info") -> None:
        self.toasts.append((message, level))


# --- 5.3a: the reference path survives a restart -----------------------------


def test_the_reference_path_survives_a_restart_and_a_missing_file_does_not(tmp_path):
    # A restart round-trip: ``ref_path`` used to be the one conditioning field
    # ``settings.VOLATILE`` dropped while ``ip_adapter``/``control`` survived
    # it -- so a session that conditioned a job reopened with the pair split.
    real = tmp_path / "ref.png"
    real.write_bytes(b"\x89PNG\r\n")
    stored = settings_mod.sanitise_form(
        {**default_form_2d(), "ref_path": str(real), "control": "canny"}
    )
    assert stored["ref_path"] == str(real), "ref_path must not be stripped like the seed"
    restored = settings_mod.restore_form(default_form_2d(), stored)
    assert restored["ref_path"] == str(real)
    assert restored["control"] == "canny"

    # But a path that has since moved or been deleted must not come back as a
    # live value that only fails at the far end of a submit -- it is cleared,
    # on this pane's first frame each session, and said once.
    missing = str(tmp_path / "gone.png")
    ctx = _Ctx()
    form = {"ref_path": missing}
    create_recipe.verify_reference_path(ctx, form)
    assert form["ref_path"] == ""
    assert len(ctx.toasts) == 1
    assert "missing" in ctx.toasts[0][0] and missing in ctx.toasts[0][0]

    # Once per session, not once per frame: a second form still naming no
    # file must not toast again.
    create_recipe.verify_reference_path(ctx, {"ref_path": missing})
    assert len(ctx.toasts) == 1

    # And a genuinely live path is left alone.
    ctx2 = _Ctx()
    live_form = {"ref_path": str(real)}
    create_recipe.verify_reference_path(ctx2, live_form)
    assert live_form["ref_path"] == str(real)
    assert ctx2.toasts == []


# --- 5.3b: a per-job rework form survives a selection round-trip ------------


def test_a_remesh_form_survives_a_look_at_another_asset():
    from warlock.studio.panes import remesh_panel

    ctx = SimpleNamespace(state=AppState())
    first = remesh_panel._form(ctx, "aaaaaaaaaaaa")
    first["remesh_profile"] = "custom"
    first["custom_faces"] = 12345

    # Glance at another asset: a fresh job gets its own fresh form.
    other = remesh_panel._form(ctx, "bbbbbbbbbbbb")
    assert other["remesh_profile"] != "custom" or other is not first

    # Come back: the original form, with what was typed, not the defaults.
    again = remesh_panel._form(ctx, "aaaaaaaaaaaa")
    assert again is first
    assert again["remesh_profile"] == "custom"
    assert again["custom_faces"] == 12345


def test_a_sheet_form_survives_a_look_at_another_asset():
    from warlock.studio.panes import sheet_panel

    ctx = SimpleNamespace(state=AppState(), sheet_options={"defaults": {}})
    first = sheet_panel._form(ctx, "aaaaaaaaaaaa")
    first["poses"].add("deadbeefcafe")
    first["name"] = "walk cycle"

    sheet_panel._form(ctx, "bbbbbbbbbbbb")
    again = sheet_panel._form(ctx, "aaaaaaaaaaaa")
    assert again is first
    assert again["poses"] == {"deadbeefcafe"}
    assert again["name"] == "walk cycle"


def test_a_sprite_form_survives_a_look_at_another_asset():
    from warlock.studio.panes import sprite_panel

    ctx = SimpleNamespace(state=AppState())
    first = sprite_panel._form(ctx, "aaaaaaaaaaaa")
    first["colors"] = 8
    first_seed = first["seed_a"]

    sprite_panel._form(ctx, "bbbbbbbbbbbb")
    again = sprite_panel._form(ctx, "aaaaaaaaaaaa")
    assert again is first
    assert again["colors"] == 8
    assert again["seed_a"] == first_seed


def test_a_retexture_form_survives_a_look_at_another_asset():
    # The 2026-09-08 audit, finding create-02: commit 89cb6412 gave
    # remesh/sheet/sprite panels a per-job-id dict, but left this sibling on
    # one shared slot compared by ``form.get("job_id") != job_id`` -- so
    # glancing at another asset and coming back silently discarded a typed
    # surface prompt.
    from warlock.studio.panes import texture_panel

    ctx = SimpleNamespace(state=AppState())
    first = texture_panel._form(ctx, "aaaaaaaaaaaa")
    first["prompt"] = "rusted iron plating"
    first["strength"] = 0.9

    texture_panel._form(ctx, "bbbbbbbbbbbb")
    again = texture_panel._form(ctx, "aaaaaaaaaaaa")
    assert again is first
    assert again["prompt"] == "rusted iron plating"
    assert again["strength"] == 0.9


def test_a_retarget_form_survives_a_look_at_another_asset():
    # The 2026-09-08 audit, finding create-02: matches
    # ``test_a_retexture_form_survives_a_look_at_another_asset`` above -- the
    # same single-slot pattern discarded a chosen custom triangle budget.
    from warlock.studio.panes import retarget_panel

    ctx = SimpleNamespace(state=AppState())
    first = retarget_panel._form(ctx, "aaaaaaaaaaaa")
    first["profile"] = "custom"
    first["custom_triangles"] = 54321

    retarget_panel._form(ctx, "bbbbbbbbbbbb")
    again = retarget_panel._form(ctx, "aaaaaaaaaaaa")
    assert again is first
    assert again["profile"] == "custom"
    assert again["custom_triangles"] == 54321


# --- 5.3c: prompt history outlives the process ------------------------------


def test_prompt_history_outlives_the_process_and_a_corrupt_file_reads_as_empty(
    tmp_path,
):
    state = AppState()
    for i in range(MAX_HISTORY + 5):
        state.remember_prompt(f"prompt {i}")
    assert len(state.history) == MAX_HISTORY, "bounded, or a long session costs a frame to load"

    # Written the way this codebase writes any served file: staged to a temp
    # and ``os.replace``d, via ``Settings.flush`` -- the same call
    # ``main.py``'s ``_persist`` makes with ``ctx.settings.set("history", ...)``.
    settings = settings_mod.Settings.load(tmp_path)
    settings.set("history", state.history)
    assert settings.flush() is True

    # A fresh process reads it back: the same restore main.py's startup path
    # performs (``state.history = [str(e) for e in as_list(settings.get(...))]``).
    reloaded = settings_mod.Settings.load(tmp_path)
    restored = [str(e) for e in settings_mod.as_list(reloaded.get("history"))]
    assert restored == state.history

    # A corrupt file degrades to an empty history, never to a crash on start.
    (tmp_path / settings_mod.FILENAME).write_text("{not json at all", encoding="utf-8")
    corrupt = settings_mod.Settings.load(tmp_path)
    assert [str(e) for e in settings_mod.as_list(corrupt.get("history"))] == []


# --- 5.5.1: prose points at controls that exist -----------------------------


def test_structure_notes_do_not_name_a_recipe_switch_that_does_not_exist():
    """The Fast/Quality tier folded into the Model combo on 2026-08-17-ish
    (``model_options``); a "Recipe" control to "switch to Quality" has not
    existed since. Both notes used to say it anyway.

    Checked against the *returned* sentence, not the whole function source:
    a developer comment inside either is allowed to keep saying "Advanced"
    for the ``model_mode`` value, which is real -- only the words a reader
    sees have to name a control that exists.
    """
    note = create_recipe.structure_note(_note_ctx(), {"base_model": "turbo"})
    assert note is not None
    assert "Switch the Recipe to Quality" not in note
    assert "under Advanced" not in note

    # ``recipe_structure_note`` only speaks once automatic routing has
    # actually resolved to a guidance-0 recipe -- ``quality="fast"`` is the
    # one stored value that still reaches "image_fast", the same reachable
    # case ``test_the_pane_hides_the_structure_picker_under_fast`` exercises.
    resolved_ctx = _note_ctx(svc=SimpleNamespace(config=None))
    fast_form = {
        "asset_type": "image",
        "generation_type": "image",
        "model_mode": "auto",
        "quality": "fast",
    }
    recipe_note = create_recipe.recipe_structure_note(resolved_ctx, fast_form)
    assert recipe_note is not None
    assert "Switch the Recipe to Quality" not in recipe_note
    assert "under Advanced" not in recipe_note


# --- 5.5.2: the repair changes the control its own note names --------------


def test_the_structure_repair_changes_the_control_its_own_note_names(monkeypatch):
    """The create-03 shape again: a repair wrote ``form["quality"]``, a key no
    control on this pane sets any more, so pressing it left the resolver's
    answer unchanged. What the Model combo's own Automatic entry writes is
    ``model_mode``/``model_override`` (``_model``'s ``else`` branch), which is
    what actually decides whether the next resolved recipe can run a
    ControlNet."""
    monkeypatch.setattr(settings_2d.controls, "button", lambda *a, **k: True)
    monkeypatch.setattr(create_recipe, "clear_for_tier", lambda ctx, form: [])

    form = {
        "asset_type": "image",
        "generation_type": "image",
        "model_mode": "advanced",
        "base_model": "sdxl",  # a real, non-ControlNet checkpoint
        "model_override": "sdxl",
        "control": "canny",
        "quality": "fast",
    }
    ctx = SimpleNamespace(state=AppState())
    problem = problems.Problem("Structure control needs a full-CFG model.", "base_model")

    settings_2d._preflight_fix(ctx, form, problem)

    assert form["model_mode"] == "auto"
    assert form["model_override"] == ""


def test_preflight_fix_does_not_match_the_dead_guidance_0_text_on_an_unrelated_field(
    monkeypatch,
):
    """The 2026-09-18 audit, finding create-04: the "guidance 0" alternative
    in ``_preflight_fix``'s message match matches no reachable ``Problem`` --
    the only refusal that ever said "guidance 0"
    (``generation.validate_request``'s CompatibilityIssue) reaches the pane
    through ``refuse``'s field-error path, never through this function. A
    synthetic ``Problem`` naming a field that is not ``base_model`` but
    still saying "guidance 0" -- what the dead branch matched on before this
    fix added the field check -- must not trigger the ControlNet repair."""
    monkeypatch.setattr(settings_2d.controls, "button", lambda *a, **k: True)
    monkeypatch.setattr(create_recipe, "clear_for_tier", lambda ctx, form: [])

    form = {
        "asset_type": "image",
        "generation_type": "image",
        "model_mode": "advanced",
        "base_model": "sdxl",
        "model_override": "sdxl",
        "control": "canny",
        "quality": "fast",
    }
    ctx = SimpleNamespace(state=AppState())
    problem = problems.Problem("This model runs at guidance 0 and cannot run one.", "quality")

    settings_2d._preflight_fix(ctx, form, problem)

    assert form["model_mode"] == "advanced", "the dead guidance-0 text must not fire the repair"
    assert form["model_override"] == "sdxl"


# --- 5.5.3: the Model and Style LoRA combos are labelled --------------------


def test_the_model_and_style_lora_combos_are_labelled():
    """Drawn as bare ``##model``/``##style_lora`` widgets before this fix,
    unlike ``_locked_sheet_recipe``'s "Image model"/"Style LoRA" labels at the
    same spot in the tileset/sprite arms' pinned display."""
    assert 'field_label("Image model")' in inspect.getsource(settings_2d._model)
    assert 'field_label("Style LoRA")' in inspect.getsource(settings_2d._lora)
