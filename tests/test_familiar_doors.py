"""T8: ``studio/familiar_doors.py`` -- the acting half of Familiar's
navigate and create-draft doors.

Headless, the same shape ``tests/test_palette.py``'s own ``_ctx`` builds:
enough of ``app_ctx.Ctx`` for ``palette.commands``/``model_gate.mode_gate``
to answer without a window, extended with ``form_2d``/``preview`` (Create's
own form) and a ``submit`` recorder so a test can prove a draft never
submits.

``warlock.studio.familiar_doors`` does not exist on the pre-T8 tree, so
every test below fails with an ``ImportError``/``AttributeError`` before
its first assertion runs against the unmodified code.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from warlock.studio import familiar_doors, modes
from warlock.studio.panes import app_settings, model_gate
from warlock.studio.state import ManualState


def _ctx(mode: str = "home", *, model_rows: list[dict[str, Any]] | None = None) -> Any:
    ctx = SimpleNamespace(
        state=SimpleNamespace(
            mode=mode,
            previous_mode=mode,
            mode_observed=mode,
            create=SimpleNamespace(stage="reference"),
            selected=None,
            source_job=None,
            wireframe=False,
            turntable=False,
            show_fps=False,
            manual=ManualState(),
            preview={},
            form_2d={},
        ),
        cache=SimpleNamespace(jobs=[], get=lambda _id: None),
        viewer=None,
        submits=[],
    )
    if model_rows is not None:
        ctx.model_rows = model_rows

    def submit(key, fn, *args, tag=None, **kwargs) -> bool:
        ctx.submits.append(key)
        return True

    ctx.submit = submit
    return ctx


# --- destinations ------------------------------------------------------


def test_destinations_come_from_the_palettes_own_commands_and_settings_categories():
    """Every mode's ``go:<key>`` and every Settings category must be
    offered -- and nothing that is an action on the current document rather
    than a place (Save, Undo, reroll) may sneak in."""
    ctx = _ctx()

    keys = {d.key for d in familiar_doors.destinations(ctx)}

    assert {f"go:{key}" for key in modes.KEYS} <= keys
    assert {"manual", "shortcuts", "workspace-layout", "show-trash"} <= keys
    assert {f"settings:{key}" for key, _label in app_settings.CATEGORIES} <= keys
    assert "save" not in keys
    assert "undo" not in keys
    assert "reroll" not in keys


def test_a_settings_destination_names_its_own_category_title():
    ctx = _ctx()
    by_key = {d.key: d.label for d in familiar_doors.destinations(ctx)}
    assert by_key["settings:models"] == "Settings → Models"


# --- navigate ------------------------------------------------------------


def test_navigating_to_a_mode_switches_it():
    ctx = _ctx("home")

    message = familiar_doors.navigate(ctx, "go:clay")

    assert ctx.state.mode == "clay"
    assert "Clay" in message


def test_navigating_to_a_gated_mode_changes_nothing_and_says_why():
    """A mode ``model_gate`` would grey out in the rail must not move the
    app when the router names it -- the sentence returned is the same one
    the greyed rail item would show."""
    ctx = _ctx(
        "home",
        model_rows=[
            {"row_key": "engine:trellis_gguf", "present": False, "size_gib": 8.0},
            {"row_key": "engine:trellis_runtime", "present": False, "size_gib": 1.0},
            {"row_key": "base:sdxl_cfg", "present": False, "size_gib": 6.0},
        ],
    )
    expected_reason = model_gate.mode_reason(ctx, "create")
    assert expected_reason  # sanity: the fixture actually gates Create

    message = familiar_doors.navigate(ctx, "go:create")

    assert ctx.state.mode == "home"
    assert message == expected_reason


def test_navigating_to_a_settings_category_opens_that_category():
    ctx = _ctx("home")

    message = familiar_doors.navigate(ctx, "settings:packs")

    assert ctx.state.mode == "settings"
    assert ctx.state.preview[app_settings.CATEGORY_SLOT] == "packs"
    assert "Packs" in message


def test_navigating_to_an_unoffered_key_says_so_rather_than_crashing():
    ctx = _ctx("home")

    message = familiar_doors.navigate(ctx, "go:does-not-exist")

    assert ctx.state.mode == "home"
    assert message


# --- draft_in_create -------------------------------------------------------


def test_a_create_draft_fills_the_brief_and_never_submits():
    ctx = _ctx("home")

    message = familiar_doors.draft_in_create(ctx, "3d_model", "a lantern")

    assert ctx.state.mode == "create"
    assert ctx.state.create.stage == "reference"
    assert ctx.state.form_2d["asset_type"] == "3d_model"
    assert ctx.state.form_2d["generation_type"] == "3d_model"
    assert ctx.state.form_2d["prompt"] == "a lantern"
    assert "Generate" in message
    assert ctx.submits == [], "a draft must never submit"


def test_a_create_draft_while_create_is_gated_changes_nothing_and_says_why():
    ctx = _ctx(
        "home",
        model_rows=[
            {"row_key": "engine:trellis_gguf", "present": False, "size_gib": 8.0},
            {"row_key": "engine:trellis_runtime", "present": False, "size_gib": 1.0},
            {"row_key": "base:sdxl_cfg", "present": False, "size_gib": 6.0},
        ],
    )
    expected_reason = model_gate.mode_reason(ctx, "create")
    assert expected_reason

    message = familiar_doors.draft_in_create(ctx, "3d_model", "a lantern")

    assert ctx.state.mode == "home"
    assert ctx.state.form_2d == {}
    assert message == expected_reason
    assert ctx.submits == []
