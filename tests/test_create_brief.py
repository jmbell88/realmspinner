"""Create's command bar: what it draws, where, and what it refuses to draw.

The bar is drawn from ``main._build_ui`` rather than from ``panes/``, so the
pane smoke sweep does not reach it. These are its own gates.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

import pytest
from imgui_bundle import imgui

from warlock.studio import create_brief, create_rail, create_stages, layout, probe
from warlock.studio.state import AppState, default_form_2d


def _body(fn) -> str:
    """A function's source with its docstring taken off.

    These assertions are about the *code*: a prose mention of ``TYPE_W`` in the
    docstring explaining why it is absent would otherwise fail the test that
    checks it is absent.
    """
    source = inspect.getsource(fn)
    doc = inspect.getdoc(fn)
    if doc:
        for line in doc.splitlines():
            source = source.replace(line, "")
    return source


def _state(stage="reference", mode="create", **kw):
    return SimpleNamespace(
        mode=mode,
        create_stage=stage,
        form_2d=default_form_2d(),
        field_errors={},
        clear_field_error=lambda _f: None,
        **kw,
    )


def _real_ctx(*, stage: str = "reference") -> Any:
    """A real ``AppState`` rather than a ``SimpleNamespace`` stub, for the
    tests below that draw ``create_brief.draw`` for real: it goes through
    ``focus.pump``/``begin``/``item`` and ``settings_2d.problems_for``, both of
    which read fields (``focus_order``, ``focus_key``, ``focus_moved``,
    ``problems_cache``, ``frame_index``) a hand-rolled stub would have to grow
    one at a time. ``test_muse_panes_smoke.py`` sets the same precedent.
    """
    return SimpleNamespace(
        state=AppState(mode="create", create_stage=stage),
        svc=SimpleNamespace(config=None),
        busy=lambda _key: False,
        confirms=SimpleNamespace(ask=lambda _dialog: None),
    )


def _synthetic_rail_items() -> list[tuple[str, str, str, str | None]]:
    """The rail's five entries with no job behind them -- for the tests below
    that need *a* rail to draw, not the real ``App._stage_rail`` (which reads
    a job and, for Rig, a filesystem-cached read this test file has no
    business standing up)."""
    return [
        (stage, create_stages.LABELS[stage], create_stages.ICONS[stage], None)
        for stage in create_stages.STAGES
    ]


def _synthetic_rail(
    ctx: Any, *, max_width: float | None = None, row_height: float | None = None
) -> None:
    """A stand-in for ``App._stage_rail``, real enough to draw and measure --
    unblocked, nothing done, which is a legitimate (if uninteresting) rail
    state and costs no job lookup."""
    create_rail.stage_rail(
        "create-stages",
        _synthetic_rail_items(),
        ctx.state.create_stage,
        max_width=max_width,
        row_height=row_height,
    )


@pytest.fixture
def frames():
    """A bare imgui context, built and destroyed around this file --
    ``test_muse_panes_smoke.py``'s fixture of the same name and the same
    reason: at most one imgui context may exist at a time, and a file that
    wants one builds and destroys it rather than relying on collection order.
    No GL and no renderer: ``renderer_has_textures`` is what lets imgui finish
    a frame without a backend claiming its font atlas, and every widget call,
    draw-list call and layout pass still runs for real.
    """
    from warlock.studio import theme

    previous = imgui.get_current_context()
    ctx = imgui.create_context()
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
    imgui.destroy_context(ctx)
    if previous is not None:
        imgui.set_current_context(previous)


# --- where it draws ---------------------------------------------------------


def test_the_bar_draws_on_the_reference_stage_only():
    """Mesh, Rig, Pose and Export draw no bar and start their columns higher.

    ``create_stages``' own rule about the rail, applied to the bar under it: a
    row with one live control and three dead ones is not honest, and an inert
    bar is worse than an absent one.
    """
    for stage in create_stages.STAGES:
        ctx = SimpleNamespace(state=_state(stage))
        assert create_brief.shows(ctx) is (stage == "reference"), stage


def test_the_bar_draws_in_no_other_mode():
    """``create_stage`` is not cleared on a mode switch -- coming back from
    Inker lands where you left -- so asking the stage alone would put Create's
    brief across the top of another workspace."""
    for mode in ("inker", "clay", "plotter", "home", "library"):
        ctx = SimpleNamespace(state=_state("reference", mode=mode))
        assert create_brief.shows(ctx) is False, mode


# --- what it holds ----------------------------------------------------------


def test_the_bar_holds_exactly_the_four_controls_of_a_brief():
    source = inspect.getsource(create_brief.draw)
    for call in ("_type(ctx", "_prompt(ctx", "_count(ctx", "_generate(ctx"):
        assert call in source, call


def test_the_recipe_is_not_in_the_bar():
    """The split is the design: the bar is *what to make*, the column is *how*.

    A control belongs to exactly one of them, which is the one-owner rule the
    two generation panes already keep.
    """
    source = inspect.getsource(create_brief)
    for absent in ("base_model", "style_lora", "lora_weight", "negative_prompt", "ref_path"):
        assert absent not in source, absent


def test_the_count_is_the_service_capped_row():
    from warlock.service.validation import MAX_REFERENCE_COUNT

    assert create_brief._COUNTS == (1, 2, 4, 8)
    assert max(create_brief._COUNTS) == MAX_REFERENCE_COUNT


def test_every_generation_type_has_a_hint():
    from warlock.studio import create_assets

    offered = {key for key, _label in create_assets.ASSET_TYPE_OPTIONS}
    assert set(create_brief._TYPE_HINTS) == offered


def test_the_count_control_carries_a_tooltip_naming_what_it_counts():
    """Four bare pills sat between the prompt and Generate with no label and no
    tooltip -- unlike the Type combo beside them, which names each of its
    values on hover through ``_TYPE_HINTS``. ``segmented_choice`` already
    supports per-option hover text through its ``tooltips=`` mapping (Inker's
    tile-behaviour row and Clay's tool row both use it); Create's own count
    control is the one holdout. The 2026-09-05 audit, finding create-11."""
    body = _body(create_brief._count)
    assert "tooltips=" in body, "segmented_choice must be given per-option hover text"
    assert set(create_brief._COUNT_HINTS) == {str(n) for n in create_brief._COUNTS}
    for hint in create_brief._COUNT_HINTS.values():
        assert hint.strip(), "a blank tooltip names nothing"


# --- the count clears its own stale field-error ring -------------------------


def test_arrow_key_edit_of_count_clears_its_stale_field_error_ring():
    """The click path (``segmented_choice``) calls ``clear_field_error`` on a
    change; the hand-rolled left/right-arrow path edits ``form["count"]``
    directly and must clear the same error, or a user who fixes an invalid
    count with the keyboard instead of a click keeps a red ring around a value
    that is no longer wrong. The 2026-09-05 audit, finding create-07."""
    body = _body(create_brief._count)
    arrow_branch = body[body.index("is_key_pressed(imgui.Key.left_arrow)") :]
    assert 'clear_field_error("count")' in arrow_branch


# --- the count is hidden where the door refuses a batch ---------------------


@pytest.mark.parametrize("asset_type", ["tileset", "sprite_sheet"])
def test_a_sheet_hides_the_count_rather_than_offering_refusals(asset_type):
    """Both sheet doors refuse a batch and say why, so four radios of which
    three are refusals would be a control offering what the thing behind it
    will not do."""
    from warlock.studio import create_assets

    form = default_form_2d()
    form["asset_type"] = asset_type
    create_assets.sync_legacy_fields(form)
    assert form["output"] == "sheet"
    assert form["count"] == 1
    # The width calculation is the observable half of "the control is skipped".
    source = inspect.getsource(create_brief.draw)
    assert "if show_count:" in source
    assert "_count(ctx, form)" in source
    # And _row_widths gives the count no width at all for a sheet.
    assert "0.0 if sheet else" in inspect.getsource(create_brief._row_widths)


def test_the_row_gives_way_in_a_stated_order():
    """The prompt shrinks, then the count is dropped, then the rail shortens,
    then Reset goes icon-only -- and the type and Generate never give way,
    because Generate is the control the bar exists to keep visible and
    ``same_line`` past the pane edge draws a control nowhere.

    ``TYPE_W`` is now legitimately in this function's body (2026-09-07): the
    rail sits ahead of the type combo on the row, so ``_row_widths`` has to
    decide the rail's width *before* the combo has drawn, which means nothing
    has narrowed ``avail`` yet and ``TYPE_W`` has to be taken off explicitly,
    once. See :func:`create_brief._row_widths`'s own docstring for why that is
    not the double-count bug this test used to guard against -- the mechanism
    moved; the "count everything exactly once" rule it protects did not.
    """
    body = _body(create_brief._row_widths)
    assert "sp(TYPE_W)" in body, "the rail is ahead of the combo now, so this must reserve it"
    assert "GENERATE_W" in body
    assert "PROMPT_MIN_W" in body
    assert "rail_full_w" in body and "rail_floor_w" in body
    # The count is the one that goes first, and only after the prompt bottoms.
    assert "show_count = False" in body
    assert "reset_compact = True" in body


def test_the_four_rung_ladder_gives_way_at_decreasing_widths(frames):
    """``_row_widths`` walked at real, decreasing window widths -- a real
    frame feeding it real ``imgui.get_style()``/``get_content_region_avail()``
    numbers rather than a source scan. No GL: this only needs the numbers
    ``create_rail.stage_rail_width`` and ``imgui.calc_text_size`` already
    produce without a renderer.
    """
    items = _synthetic_rail_items()
    seen: dict[float, tuple] = {}

    def measure(width: float) -> None:
        def build() -> None:
            rail_full = create_rail.stage_rail_width(items, "reference")
            rail_floor = create_rail.stage_rail_width(items, "reference", max_width=0.0)
            seen[width] = create_brief._row_widths(False, rail_full, rail_floor)

        frames(build, (width, 700.0))

    for width in (1400.0, 700.0, 500.0, 320.0):
        measure(width)

    widths = sorted(seen)
    rail_w = [seen[w][0] for w in widths]
    prompt_w = [seen[w][1] for w in widths]
    show_count = [seen[w][2] for w in widths]
    reset_compact = [seen[w][3] for w in widths]

    # Nothing here gets *more* room as the window gets narrower.
    assert rail_w == sorted(rail_w)
    assert prompt_w == sorted(prompt_w)
    # The count is shown only once there is room for it (False before True,
    # narrow to wide) and never flips back as the window widens further.
    assert show_count == sorted(show_count)
    # Reset is icon-only only under pressure (True before False, narrow to
    # wide) and never flips back either.
    assert reset_compact == sorted(reset_compact, reverse=True)
    # At the widest, everything is at its natural size.
    assert show_count[-1] is True
    assert reset_compact[-1] is False
    # At the narrowest, the ladder has bottomed out on both ends.
    assert prompt_w[0] == pytest.approx(create_brief.sp(create_brief.PROMPT_MIN_W))
    assert reset_compact[0] is True


# --- the anchors the guided tour points at ----------------------------------


def test_the_tour_anchors_moved_with_the_controls():
    """``studio/tour/scripts.py`` binds steps to ``create/prompt`` and
    ``create/generate``. The tour reads positions and never computes them, so
    the anchors keep working precisely because they are marked at the controls'
    new home rather than left behind at the old one."""
    source = inspect.getsource(create_brief)
    assert 'anchors.mark("create/prompt")' in source
    assert 'anchors.mark("create/generate")' in source

    from warlock.studio.panes import settings_2d

    pane = inspect.getsource(settings_2d)
    assert 'anchors.mark("create/prompt")' not in pane
    assert 'anchors.mark("create/generate")' not in pane


def test_the_tours_still_name_anchors_something_marks():
    """Both directions, so a rename on either side fails here rather than as a
    tour step pointing at nothing."""
    from warlock.studio import anchors as anchors_mod  # noqa: F401
    from warlock.studio.tour import scripts

    wanted = {
        step.anchor
        for tour in scripts.TOURS
        for step in tour.steps
        if step.anchor and step.anchor.startswith("create/")
    }
    marked = set()
    for mod in (create_brief,):
        for line in inspect.getsource(mod).splitlines():
            if "anchors.mark(" in line:
                marked.add(line.split('"')[1])
    # Every create/ anchor the tours name is either marked here or elsewhere in
    # Create; the two this move touched must be here.
    assert {"create/prompt", "create/generate"} <= marked
    assert {"create/prompt", "create/generate"} <= wanted


# --- the pane registration --------------------------------------------------


def test_the_bar_is_a_registered_pane_not_a_bare_row():
    """``layout.pane`` is what gives the row the role fill, the divider,
    ``guard``'s isolation and a real child-window name for ``probe`` to
    attribute controls to -- drawn bare, the row's controls used to census
    against the empty-string pane, which reads as controls nobody owns.

    The pane opens unconditionally now (2026-09-07): the rail is a breadcrumb
    for every stage, so ``create_brief.shows`` no longer gates whether
    ``main.py`` opens the pane at all -- only ``create_brief.draw`` still asks
    it, to decide how much of the row to fill in. ``main.py`` sizes the pane
    from ``create_brief.bar_height`` instead.
    """
    from warlock.studio import main

    source = inspect.getsource(main.App._build_ui)
    assert 'layout_mod.pane(\n                        "brief",' in source
    assert "create_brief.bar_height(ctx)" in source
    assert "create_brief.draw(ctx, self._stage_rail)" in source
    # The gate moved *into* create_brief.draw, not away entirely.
    assert "create_brief.shows(ctx)" not in source
    assert "if not shows(ctx):" in inspect.getsource(create_brief.draw)


# --- the disabled button's reason -------------------------------------------


def test_the_disabled_generate_wears_the_first_problem_as_its_reason():
    """``widgets.Problem`` is a ``str`` subclass -- the message *is* the object,
    and there is no ``.message`` on it.

    This branch only runs when the form is *invalid*, which is why neither the
    suite nor a screenshot of a seeded (valid) form ever executed it: the bar
    raised ``AttributeError`` and the pane guard blanked it, on every frame with
    an empty prompt. ``/exercise-mode create`` is what found it.
    """
    from warlock.studio import widgets

    problem = widgets.Problem("Describe what to generate.", "prompt")
    assert not hasattr(problem, "message")
    assert str(problem) == "Describe what to generate."

    body = _body(create_brief._generate)
    assert ".message" not in body, "Problem is a str; there is no .message"
    assert "str(problems[0])" in body


def test_an_empty_prompt_leaves_the_bar_drawable():
    """The whole-object check behind the test above: every problem the column's
    validator can raise for an untouched form must render as a reason string."""
    from warlock.studio.panes import settings_2d

    form = default_form_2d()
    form["prompt"] = ""
    problems = settings_2d.validate(form)
    assert problems, "an empty prompt must refuse"
    for problem in problems:
        assert isinstance(problem, str)
        assert str(problem)


# --- the rail and the brief, drawn for real ----------------------------------


def test_the_rail_callable_is_called_at_every_stage(frames):
    """``draw`` calls whatever ``rail`` it is handed on all five stages, even
    the four that draw no brief -- the pane always opens now, and the rail is
    what it opens *for*."""
    calls: list[str] = []

    def rail_stub(ctx, *, max_width=None, row_height=None):
        calls.append(ctx.state.create_stage)

    for stage in create_stages.STAGES:
        ctx = _real_ctx(stage=stage)
        frames(lambda ctx=ctx: create_brief.draw(ctx, rail_stub))

    assert calls == list(create_stages.STAGES)


def test_the_brief_only_reaches_the_settings_door_on_reference(frames):
    """``ctx.busy`` is read only inside the brief half of ``draw`` (to decide
    whether Generate is enabled) -- a call on a non-Reference stage would mean
    the four dead controls' plumbing still ran even though nothing draws."""
    busy_calls: list[str] = []

    def rail_stub(ctx, *, max_width=None, row_height=None):
        return None

    for stage in create_stages.STAGES:
        ctx = _real_ctx(stage=stage)
        ctx.busy = lambda key, stage=stage: busy_calls.append(stage) or False
        frames(lambda ctx=ctx: create_brief.draw(ctx, rail_stub))

    assert busy_calls == ["reference"]


def test_the_bar_fits_the_height_it_declares(frames):
    """``BAR_H`` and ``RAIL_ONLY_H`` are measured, not derived -- this is the
    thing that measures them: a real frame draws the row (or, off Reference,
    just the rail) into a bare ``imgui.begin("smoke")`` window, and the
    content span plus the padding ``layout.pane`` would spend on top of it
    (added back on both edges, since this window has none of its own) must
    fit the pane height ``bar_height`` declares for that stage.

    Muse's own ``BAR_H`` was wrong exactly this way before its own version of
    this test existed -- 118 declared against ~142 dp actually drawn, because
    neither figure ever charged for that padding and nothing drew the bar for
    real to catch it.
    """
    for stage in create_stages.STAGES:
        ctx = _real_ctx(stage=stage)
        positions: dict[str, float] = {}

        def build(ctx=ctx, positions=positions) -> None:
            positions["before"] = imgui.get_cursor_pos_y()
            create_brief.draw(ctx, _synthetic_rail)
            positions["after"] = imgui.get_cursor_pos_y()

        frames(build)

        pad = imgui.get_style().window_padding.y
        content_h = (positions["after"] - positions["before"]) + 2 * pad
        declared = create_brief.bar_height(ctx)
        assert content_h <= declared, (
            f"stage={stage}: the row drew {content_h:.1f}px of content against "
            f"a declared {declared:.1f}px ({'BAR_H' if stage == 'reference' else 'RAIL_ONLY_H'} "
            f"= {create_brief.BAR_H if stage == 'reference' else create_brief.RAIL_ONLY_H})"
        )


def test_the_bar_fits_with_the_count_hidden_too(frames):
    """The sheet/character arm draws three brief controls instead of four --
    narrower, never taller -- but it is worth its own frame rather than an
    inference from the case above, since ``_row_widths`` treats it as its own
    branch."""
    ctx = _real_ctx(stage="reference")
    ctx.state.form_2d["asset_type"] = "tileset"
    positions: dict[str, float] = {}

    def build() -> None:
        positions["before"] = imgui.get_cursor_pos_y()
        create_brief.draw(ctx, _synthetic_rail)
        positions["after"] = imgui.get_cursor_pos_y()

    frames(build)

    pad = imgui.get_style().window_padding.y
    content_h = (positions["after"] - positions["before"]) + 2 * pad
    assert content_h <= create_brief.bar_height(ctx)


# --- Reset moved into the bar's own pane -------------------------------------


def test_reset_is_censused_against_the_bars_own_pane(frames, monkeypatch):
    """``probe``'s per-frame census -- the mechanism ``dev/scripts/exercise_mode.py``
    drives every control through -- must attribute Reset to the bar's own
    child window once it is drawn through ``layout.pane``, not to whatever
    happened to be current when it was a bare row.
    """
    monkeypatch.setattr(probe, "ENABLED", True)
    ctx = _real_ctx()

    def build() -> None:
        layout.begin_frame()
        probe.begin_frame()
        with layout.pane(
            "brief", (900.0, create_brief.bar_height(ctx)), layout.PaneRole.CONTENT
        ) as visible:
            if visible:
                create_brief.draw(ctx, _synthetic_rail)

    frames(build)

    census = probe.census()
    reset = next(c for c in census if c.text.startswith("Reset"))
    assert reset.where == "brief", census


def test_reset_no_longer_draws_from_the_settings_column():
    """The other half of the move: ``settings_2d`` must hold no copy of it --
    one owner per control, per ``CLAUDE.md``."""
    from warlock.studio.panes import settings_2d

    source = inspect.getsource(settings_2d)
    assert "_reset_row" not in source
    assert "Reset..." not in source
