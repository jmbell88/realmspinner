"""``create_recipe.resolved_recipe`` must not re-resolve an unchanged form.

The 2026-09-08 audit's finding create-06 is two symptoms of one cause seen
from two ends: ``provenance._dir_fingerprint`` re-walks every installed
checkpoint directory on every call to ``file_fingerprint`` (unchanged --
that walk feeds provenance records and the orchestrator's ruling was not to
put a staleness window into it), and this pane called
``generation.resolve_recipe`` -- which triggers that same walk -- with no
memo key at all, from three note helpers plus the command bar. This module
tests the pane-side half: the memo is keyed on the *request's contents*, not
on wall time or frame count, per the same ruling.
"""

from __future__ import annotations

from types import SimpleNamespace

from warlock import generation
from warlock.studio.modes.create.engine import assets as create_assets
from warlock.studio.modes.create.engine import recipe as create_recipe
from warlock.studio.state import AppState, default_form_2d


def _ctx() -> SimpleNamespace:
    # config=None matches test_settings_2d_notes.py's _ctx_resolving: "no
    # config, no opinion about what is downloaded" -- resolve_recipe still
    # runs its full candidate search with config=None, which is what makes it
    # worth memoising.
    return SimpleNamespace(state=AppState(), svc=SimpleNamespace(config=None))


def _counting_resolve_recipe(calls: list[int]):
    original = generation.resolve_recipe

    def counting(request, config=None, *, installed=None):
        calls.append(1)
        return original(request, config, installed=installed)

    return counting, original


def test_resolved_recipe_is_not_recomputed_while_the_form_is_unchanged():
    """The 2026-09-08 audit, finding create-06, and the orchestrator's ruling
    on the fix: an unchanged form must resolve once, not once per frame.
    ``_resolved_recipe`` is memoised on the request built from the form, so
    the checkpoint-directory walk inside ``resolve_recipe`` runs once per form
    *edit*, never once per frame regardless of how many frames pass."""
    form = default_form_2d()
    create_assets.sync_legacy_fields(form)
    ctx = _ctx()

    calls: list[int] = []
    counting, original = _counting_resolve_recipe(calls)
    generation.resolve_recipe = counting
    try:
        first = create_recipe.resolved_recipe(ctx, form)
        for _ in range(5):
            ctx.state.frame_index += 1
            again = create_recipe.resolved_recipe(ctx, form)
            assert again == first
        assert len(calls) == 1, f"expected one resolution across many frames, got {len(calls)}"
    finally:
        generation.resolve_recipe = original


def test_resolved_recipe_reflects_a_form_edit_on_the_next_frame():
    """The memo is keyed on the request's contents, so an edited form
    resolves again -- an unrecognised Advanced override must be visible the
    moment it is chosen, not "eventually, once the frame index moves"."""
    form = default_form_2d()
    create_assets.sync_legacy_fields(form)
    ctx = _ctx()

    first = create_recipe.resolved_recipe(ctx, form)

    form["model_mode"] = "advanced"
    form["model_override"] = "does-not-exist-in-the-registry"
    changed = create_recipe.resolved_recipe(ctx, form)
    # An unknown override resolves to None (generation.resolve_recipe's
    # documented behaviour for an unrecognised base_model key).
    assert changed is None
    assert changed != first


def test_an_edit_and_undo_resolves_again_rather_than_reading_a_sticky_cache():
    """Editing away from the original values and then undoing back to them
    are two distinct requests, not "the same form seen twice": the cache
    compares the request's contents on every call, so it must not treat
    ``id(form)`` as identity and pin the edited answer.

    The memo holds one pair, not a history, so each of the three genuinely
    distinct *transitions* this test asks about -- the initial resolve, the
    edit, and the undo -- costs exactly one recompute; what it must not do is
    keep costing one on every one of the many unchanged frames that follow
    the undo, which is the "sticky on identity" failure this guards against
    (a cache keyed on ``id(form)`` alone would never invalidate on the edit at
    all, and a cache keyed on frame index would recompute on every one of
    those trailing frames too)."""
    form = default_form_2d()
    create_assets.sync_legacy_fields(form)
    ctx = _ctx()
    original_prompt = form["prompt"]

    calls: list[int] = []
    counting, original = _counting_resolve_recipe(calls)
    generation.resolve_recipe = counting
    try:
        first = create_recipe.resolved_recipe(ctx, form)

        form["prompt"] = "a wholly different prompt, chosen to differ"
        ctx.state.frame_index += 1
        # The prompt does not change which recipe automatic routing picks --
        # only the *request* differs, which is exactly the point: the cache
        # has to notice a changed request even when the changed field is not
        # one the resolved answer depends on, or it would only work by luck.
        create_recipe.resolved_recipe(ctx, form)

        form["prompt"] = original_prompt
        ctx.state.frame_index += 1
        restored = create_recipe.resolved_recipe(ctx, form)
        assert restored == first

        # Several more unchanged frames after the undo must not add a fourth
        # resolution -- the cache is content-keyed, not frame-keyed.
        for _ in range(3):
            ctx.state.frame_index += 1
            assert create_recipe.resolved_recipe(ctx, form) == restored

        assert len(calls) == 3, (
            "expected exactly one recompute per transition (initial, edit, "
            f"undo) and none for the idle frames after, got {len(calls)}"
        )
    finally:
        generation.resolve_recipe = original


def test_model_combo_resolves_the_recipe_through_the_memo_not_directly():
    """The 2026-09-13 audit, finding create-03.

    ``_model``'s Automatic branch called ``generation.resolve_recipe``
    directly instead of going through the ``_resolved_recipe`` memo,
    repeating ``provenance._dir_fingerprint``'s ``rglob`` over every
    installed checkpoint directory on every frame -- the five sibling call
    sites (the note helpers this module already covers) were moved onto the
    memo by the 2026-09-08 create-06 fix, but the Model combo itself was
    missed. Widgets calls are stubbed out (no imgui context in this suite)
    so the pane function can be driven headlessly, same as this module's
    other tests drive ``_resolved_recipe`` directly.
    """
    from warlock.studio.modes.create.ui.panes import settings_2d as mod

    form = default_form_2d()
    create_assets.sync_legacy_fields(form)
    form["model_mode"] = "auto"
    ctx = _ctx()
    ctx.base_models = []

    calls: list[int] = []
    counting, original = _counting_resolve_recipe(calls)
    generation.resolve_recipe = counting

    stub_widgets = SimpleNamespace(
        field_label=lambda *a, **k: None,
        combo=lambda name, before, options: before,
        field_error=lambda *a, **k: None,
        muted_wrapped=lambda *a, **k: None,
        wrapped=lambda *a, **k: None,
    )
    real_widgets = mod.widgets
    mod.widgets = stub_widgets
    try:
        mod._model(ctx, form)
        ctx.state.frame_index += 1
        mod._model(ctx, form)
        assert len(calls) == 1, (
            "expected one resolution across two unchanged frames "
            f"(the memo should have been reused), got {len(calls)}"
        )
    finally:
        mod.widgets = real_widgets
        generation.resolve_recipe = original
