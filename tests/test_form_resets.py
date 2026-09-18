"""Both Create forms can be put back to their defaults, and both ask first.

The 3D pane had no *Reset...* while the 2D pane did, which is the asymmetry
these tests are really about: both forms accumulate overrides across a session
and only one of them offered a way out of that. So the pair is asserted
together -- a reset that exists on one pane and not the other is exactly the
state this file exists to catch coming back.

The two now live in *different modules*: the 2D reset moved out of
``settings_2d`` and onto Create's command bar when the bar absorbed the stage
rail (2026-09-07), while the 3D one stayed in ``settings_3d``. That makes
asserting them together more valuable than it was when they were siblings --
there is no longer a shared file a reader would notice both in.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

from warlock.studio import dialogs
from warlock.studio.modes.create.ui import brief as create_brief
from warlock.studio.modes.create.ui.panes import settings_2d, settings_3d
from warlock.studio.state import DEFAULT_FORM_3D, default_form_2d


class _Ctx:
    def __init__(self) -> None:
        self.state = SimpleNamespace(
            form_2d=default_form_2d(),
            form_3d=dict(DEFAULT_FORM_3D),
            source_job="job-1",
            preview={"anything": 1},
            preview_dirty_at=0.0,
        )
        self.confirms = dialogs.ConfirmQueue()
        self.toasts: list[str] = []

    def toast(self, text: str, level: str = "info", *a, **kw) -> None:
        self.toasts.append(text)


# --- the 3D form ------------------------------------------------------------


def test_resetting_the_model_form_restores_every_default():
    ctx = _Ctx()
    ctx.state.form_3d.update(
        platform="high", size_m=2.5, mesh_seed=1234, candidates=3, rig=True
    )
    settings_3d._reset(ctx)
    assert ctx.state.form_3d == DEFAULT_FORM_3D


def test_resetting_the_model_form_does_not_alias_the_default_dict():
    """``DEFAULT_FORM_3D`` is module state; a form aliased onto it would edit
    the default for the rest of the process."""
    ctx = _Ctx()
    settings_3d._reset(ctx)
    ctx.state.form_3d["platform"] = "poisoned"
    assert DEFAULT_FORM_3D["platform"] == ""


def test_resetting_the_model_form_keeps_the_chosen_source():
    """The source is not part of this form -- it lives on ``state.source_job``
    -- and dropping it would be a larger action than the button offers."""
    ctx = _Ctx()
    settings_3d._reset(ctx)
    assert ctx.state.source_job == "job-1"


def test_resetting_the_model_form_says_so():
    ctx = _Ctx()
    settings_3d._reset(ctx)
    assert ctx.toasts and "default" in ctx.toasts[0]


# --- the 2D form, which is the precedent ------------------------------------


def test_resetting_the_image_form_restores_every_default():
    ctx = _Ctx()
    ctx.state.form_2d.update(prompt="a knight")
    settings_2d._reset(ctx)
    assert ctx.state.form_2d["prompt"] == default_form_2d()["prompt"]


def test_resetting_the_image_form_drops_the_stale_preview():
    """The preview describes the form that produced it."""
    ctx = _Ctx()
    settings_2d._reset(ctx)
    assert ctx.state.preview == {}


# --- both are behind a confirm ----------------------------------------------


def _row_source(fn) -> str:
    return inspect.getsource(fn)


def test_reset_confirm_text_names_every_field_it_discards():
    """The 2026-09-08 audit, finding create-03: the confirm dialog named only
    six things -- the prompt, the negative prompt, the model, the LoRA, the
    reference and the run controls -- while ``settings_2d._reset`` actually
    replaces the *whole* form with ``default_form_2d()``, silently discarding
    the asset type (the whole Image/3D Model/Seamless Material/Tileset/Sprite
    Sheet/Character selection) and every Tileset/Sprite/Character field the
    user had typed in (materials, variants, style_lock, seam_erase, palette,
    cell_size, ...), none of which the old text named.

    Built from a form where every field carries a value that differs from
    its own default, so a field Reset silently drops -- now, or one added
    later -- shows up as "changed" here the same way materials/variants/
    palette did for the audit, rather than this test only re-checking the
    fields the audit happened to name.
    """
    before = default_form_2d()
    for key, value in list(before.items()):
        if isinstance(value, bool):
            before[key] = not value
        elif isinstance(value, float):
            before[key] = value + 1.0
        elif isinstance(value, int):
            before[key] = value + 1
        elif isinstance(value, list):
            before[key] = ["a value the default form never has"]
        else:
            before[key] = "a value the default form never has"
    after = default_form_2d()

    # What the confirm text already named correctly, before this fix -- in
    # settings_2d's own vocabulary for "the run controls" (count, seed,
    # seed_locked) used elsewhere in that pane.
    already_named = {
        "prompt", "negative_prompt", "base_model", "style_lora",
        "lora_weight", "ref_path", "count", "seed", "seed_locked",
    }
    asset_type_fields = {"asset_type", "generation_type", "output"}
    arm_fields = {"materials", "variants", "style_lock", "seam_erase", "palette", "cell_size"}

    changed = {k for k in before if before[k] != after.get(k)}
    unnamed = changed - already_named

    # Sanity: this is the exact shape the audit reproduced. If either line
    # stops holding, the finding no longer describes the code and this test
    # should be re-read rather than loosened.
    assert asset_type_fields <= unnamed
    assert arm_fields <= unnamed

    message = create_brief._RESET_CONFIRM_MESSAGE.lower()
    assert "asset type" in message, (
        "the confirm text must name the asset type -- Reset silently "
        "discards asset_type/generation_type/output"
    )
    for word in ("tileset", "sprite", "character"):
        assert word in message, (
            f"the confirm text must name {word!r} -- Reset silently discards "
            "that arm's fields (materials, variants, style_lock, ...)"
        )


def test_both_resets_are_behind_a_confirm_dialog():
    """Read off the source, because drawing the row needs a live imgui context.

    What is being pinned is the *shape*: a form reset is unrecoverable (the 2D
    seed is rerolled, so even retyping the prompt does not get you back), so
    neither button may act on the click that draws it.
    """
    for source in (_row_source(settings_3d._reset_row), _row_source(create_brief._reset)):
        assert "dialogs.Confirm(" in source
        assert "on_confirm=" in source
        assert "_reset(ctx)" in source
