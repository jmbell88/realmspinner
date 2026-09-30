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

from realmspinner.studio import layout, probe
from realmspinner.studio.modes.create.ui import brief as create_brief
from realmspinner.studio.modes.create.ui import rail as create_rail
from realmspinner.studio.modes.create.ui import stages as create_stages
from realmspinner.studio.state import AppState, default_form_2d


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
        create=SimpleNamespace(stage=stage),
        form_2d=default_form_2d(),
        field_errors={},
        clear_field_error=lambda _f: None,
        **kw,
    )


def _real_ctx(
    *,
    stage: str = "reference",
    source: dict[str, Any] | None = None,
    model_rows: list[dict[str, Any]] | None = None,
) -> Any:
    """A real ``AppState`` rather than a ``SimpleNamespace`` stub, for the
    tests below that draw ``create_brief.draw`` for real: it goes through
    ``focus.pump``/``begin``/``item`` and ``create_recipe.problems_for``, both of
    which read fields (``focus_order``, ``focus_key``, ``focus_moved``,
    ``create.problems_cache``, ``frame_index``) a hand-rolled stub would have to
    grow one at a time. ``test_muse_panes_smoke.py`` sets the same precedent.

    ``source`` is a finished reference the Mesh stage's Source chip names, and
    ``model_rows`` the download snapshot the engine's absence is read from.
    ``ctx.asked`` and ``ctx.toasts`` collect what the bar asks and says.
    """
    state = AppState(mode="create")
    state.create.stage = stage
    jobs = {source["id"]: source} if source else {}
    if source:
        state.source_job = source["id"]
    asked: list[Any] = []
    toasts: list[str] = []
    return SimpleNamespace(
        state=state,
        svc=SimpleNamespace(config=None),
        busy=lambda _key: False,
        confirms=SimpleNamespace(ask=asked.append),
        cache=SimpleNamespace(get=jobs.get, jobs=list(jobs.values()), active=None),
        job=lambda: None,
        textures=None,
        model_rows=model_rows or [],
        submit=lambda *a, **k: True,
        toast=lambda text, *a, **k: toasts.append(text),
        asked=asked,
        toasts=toasts,
    )


#: A finished reference, the least the Mesh stage needs to call it a source.
_SOURCE = {
    "id": "ref-1",
    "name": "mossy well",
    "status": "done",
    "stage": "reference",
    "files": ["input.png"],
}

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
        ctx.state.create.stage,
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
    from realmspinner.studio import theme

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


def test_the_bar_draws_on_reference_and_mesh_only():
    """Reference and Mesh each *generate*, so each carries the press; Rig, Pose
    and Export draw the rail alone and start their columns higher.

    ``create_stages``' own rule about the rail, applied to the bar under it: a
    bar with dead controls is not honest, and an inert bar is worse than an
    absent one -- so the stages that make nothing from this row get none of it.
    """
    for stage in create_stages.STAGES:
        ctx = SimpleNamespace(state=_state(stage))
        assert create_brief.shows(ctx) is (stage in ("reference", "mesh")), stage
    assert create_brief.GENERATING_STAGES == ("reference", "mesh")


def test_the_bar_draws_in_no_other_mode():
    """``create.stage`` is not cleared on a mode switch -- coming back from
    Inker lands where you left -- so asking the stage alone would put Create's
    brief across the top of another workspace."""
    for mode in ("inker", "clay", "plotter", "home", "library"):
        ctx = SimpleNamespace(state=_state("reference", mode=mode))
        assert create_brief.shows(ctx) is False, mode


# --- what it holds ----------------------------------------------------------


def test_the_bar_holds_the_same_controls_on_both_generating_stages(frames, monkeypatch):
    """Rail, what to generate, Count, Generate, Reset -- in that order, on both
    stages. Only *what to generate* differs: Reference draws the type and the
    prompt, Mesh a Source chip. Read off a real draw, not a source scan."""
    monkeypatch.setattr(probe, "ENABLED", True)
    seen: dict[str, list[str]] = {}
    for stage in ("reference", "mesh"):
        ctx = _real_ctx(stage=stage, source=_SOURCE)

        def build(ctx=ctx) -> None:
            layout.begin_frame()
            probe.begin_frame()
            with layout.pane(
                "brief", (1200.0, create_brief.bar_height(ctx)), layout.PaneRole.CONTENT
            ) as visible:
                if visible:
                    create_brief.draw(ctx, _synthetic_rail)

        frames(build)
        ordered = sorted(probe.census(), key=lambda c: c.rect[0])
        seen[stage] = [c.text for c in ordered if c.where == "brief"]

    from realmspinner.studio.modes.create.engine import assets as create_assets

    rail = [create_stages.LABELS[stage] for stage in create_stages.STAGES]
    label = create_assets.selected(_real_ctx().state.form_2d).create_label
    # The unnamed control after the rail is *what to generate*: the type combo
    # on Reference (the prompt is not a probed control), the Source chip on Mesh.
    assert seen["reference"] == [*rail, "", "1", "2", "4", "8", label, "Reset..."]
    assert seen["mesh"] == [*rail, "", "1", "2", "3", "Make 3D", "Reset..."]


def test_the_mesh_stage_draws_the_command_bar(frames, monkeypatch):
    """Step 3 of the Create redesign: the Mesh stage has the same bar as
    Reference -- a Source chip where the prompt was, Candidates 1/2/3, *Make 3D*
    and Reset -- and its column holds none of them."""
    labels: list[str] = []
    titles: list[str] = []
    real_button = create_brief.widgets.primary_button
    real_fit = create_brief.widgets.fit_text
    monkeypatch.setattr(
        create_brief.widgets,
        "primary_button",
        lambda label, *a, **k: labels.append(label) or real_button(label, *a, **k),
    )
    monkeypatch.setattr(
        create_brief.widgets,
        "fit_text",
        lambda text, width: titles.append(text) or real_fit(text, width),
    )

    ctx = _real_ctx(stage="mesh")
    frames(lambda: create_brief.draw(ctx, _synthetic_rail))
    assert labels == ["Make 3D"]
    assert titles[0] == "Choose an image..."

    titles.clear()
    ctx = _real_ctx(stage="mesh", source=_SOURCE)
    frames(lambda: create_brief.draw(ctx, _synthetic_rail))
    assert titles[0] == "mossy well"
    assert "reference - ref-1" in titles


def test_the_source_chip_names_the_reference_behind_a_selected_mesh(frames, monkeypatch):
    """With no explicit pick, a selected finished mesh's parent is what a press
    uses, so it is what the chip names (and says whose reference it is)."""
    titles: list[str] = []
    real_fit = create_brief.widgets.fit_text
    monkeypatch.setattr(
        create_brief.widgets,
        "fit_text",
        lambda text, width: titles.append(text) or real_fit(text, width),
    )
    mesh = {"id": "m1", "stage": "model", "status": "done", "parent_id": "ref-1", "files": []}
    ctx = _real_ctx(stage="mesh")
    ctx.cache = SimpleNamespace(
        get=lambda jid: {"ref-1": _SOURCE, "m1": mesh}.get(jid), jobs=[], active=None
    )
    ctx.job = lambda: mesh
    monkeypatch.setattr(create_stages, "parent", lambda _ctx, _job: _SOURCE)
    frames(lambda: create_brief.draw(ctx, _synthetic_rail))
    assert "mossy well" in titles
    assert "this mesh's reference" in titles


def test_the_mesh_bar_disables_make_3d_for_a_missing_engine(frames, monkeypatch):
    """The engine's download is a reason on the button, not a surprise after the
    cutout: ``engine_problem`` is in the same list the button reads."""
    seen: list[tuple[bool, str]] = []
    real_button = create_brief.widgets.primary_button

    def spy(label, *a, enabled=True, reason="", **k):
        seen.append((enabled, reason))
        return real_button(label, *a, enabled=enabled, reason=reason, **k)

    monkeypatch.setattr(create_brief.widgets, "primary_button", spy)
    rows = [{"row_key": "engine:trellis_runtime", "present": False, "label": "TRELLIS.2 runtime"}]
    frames(lambda: create_brief.draw(
        _real_ctx(stage="mesh", source=_SOURCE, model_rows=rows), _synthetic_rail
    ))
    frames(lambda: create_brief.draw(_real_ctx(stage="mesh", source=_SOURCE), _synthetic_rail))
    assert seen[0][0] is False and "not downloaded" in seen[0][1]
    assert seen[1] == (True, "")


def test_generate_has_the_same_width_and_placement_on_both_stages(frames):
    from realmspinner.studio import anchors

    seen: dict[str, tuple[float, float, float, float]] = {}
    for stage in ("reference", "mesh"):
        ctx = _real_ctx(stage=stage, source=_SOURCE)

        def build(ctx=ctx, stage=stage) -> None:
            anchors.begin_frame()
            create_brief.draw(ctx, _synthetic_rail)
            seen[stage] = anchors.rect("create/generate")

        frames(build, (1200.0, 300.0))
    assert seen["reference"] is not None and seen["mesh"] is not None
    assert seen["reference"][2] == pytest.approx(seen["mesh"][2])
    assert seen["reference"][0] == pytest.approx(seen["mesh"][0], abs=1.0)
    assert seen["reference"][1] == pytest.approx(seen["mesh"][1], abs=1.0)


def test_generate_is_labelled_by_the_stage(frames, monkeypatch):
    from realmspinner.studio.modes.create.engine import assets as create_assets

    labels: dict[str, str] = {}
    real_button = create_brief.widgets.primary_button
    for stage in ("reference", "mesh"):
        monkeypatch.setattr(
            create_brief.widgets,
            "primary_button",
            lambda label, *a, stage=stage, **k: labels.setdefault(stage, label)
            and real_button(label, *a, **k),
        )
        ctx = _real_ctx(stage=stage, source=_SOURCE)
        if stage == "reference":
            ctx.state.form_2d["asset_type"] = "tileset"
        frames(lambda ctx=ctx: create_brief.draw(ctx, _synthetic_rail))
    assert labels["reference"] == create_assets.ASSET_TYPES["tileset"].create_label
    assert labels["mesh"] == "Make 3D"


def test_the_count_pills_take_their_range_from_the_stage(frames, monkeypatch):
    from realmspinner.service.validation import MAX_MESH_CANDIDATES

    offered: dict[str, tuple[str, ...]] = {}
    real = create_brief.controls.segmented_choice
    for stage in ("reference", "mesh"):
        monkeypatch.setattr(
            create_brief.controls,
            "segmented_choice",
            lambda key, options, current, *a, stage=stage, **k: (
                offered.setdefault(stage, tuple(v for v, _ in options))
                and real(key, options, current, *a, **k)
            ),
        )
        ctx = _real_ctx(stage=stage, source=_SOURCE)
        frames(lambda ctx=ctx: create_brief.draw(ctx, _synthetic_rail))
    assert offered["reference"] == ("1", "2", "4", "8")
    assert offered["mesh"] == tuple(str(n) for n in range(1, MAX_MESH_CANDIDATES + 1))


def test_both_stages_show_a_visible_candidates_label(frames, monkeypatch):
    texts: dict[str, list[str]] = {}
    real = create_brief.widgets.muted
    for stage in ("reference", "mesh"):
        seen = texts.setdefault(stage, [])
        monkeypatch.setattr(
            create_brief.widgets, "muted", lambda text, seen=seen: seen.append(text) or real(text)
        )
        ctx = _real_ctx(stage=stage, source=_SOURCE)
        frames(lambda ctx=ctx: create_brief.draw(ctx, _synthetic_rail))
    assert "Candidates" in texts["reference"]
    assert "Candidates" in texts["mesh"]
    assert create_brief.COUNT_LABEL == "Candidates"


def test_the_recipe_is_not_in_the_bar():
    """The split is the design: the bar is *what to make*, the column is *how*.

    A control belongs to exactly one of them, which is the one-owner rule the
    two generation panes already keep.
    """
    source = inspect.getsource(create_brief)
    for absent in ("base_model", "style_lora", "lora_weight", "negative_prompt", "ref_path"):
        assert absent not in source, absent


def test_the_count_is_the_service_capped_row():
    from realmspinner.service.validation import MAX_MESH_CANDIDATES, MAX_REFERENCE_COUNT

    assert create_brief._COUNTS == (1, 2, 4, 8)
    assert max(create_brief._COUNTS) == MAX_REFERENCE_COUNT
    assert create_brief._MESH_COUNTS == (1, 2, 3)
    assert max(create_brief._MESH_COUNTS) == MAX_MESH_CANDIDATES


def test_every_generation_type_has_a_hint():
    from realmspinner.studio.modes.create.engine import assets as create_assets

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
    assert set(create_brief._MESH_COUNT_HINTS) == {str(n) for n in create_brief._MESH_COUNTS}
    for hint in (*create_brief._COUNT_HINTS.values(), *create_brief._MESH_COUNT_HINTS.values()):
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
    from realmspinner.studio.modes.create.engine import assets as create_assets

    form = default_form_2d()
    form["asset_type"] = asset_type
    create_assets.sync_legacy_fields(form)
    assert form["output"] == "sheet"
    assert form["count"] == 1
    # The width calculation is the observable half of "the control is skipped".
    source = inspect.getsource(create_brief.draw)
    assert "if show_count:" in source
    assert "_count(ctx, form, counts, current" in source
    # And _row_widths gives the count no width at all for a sheet.
    assert "show_count = not hide_count" in inspect.getsource(create_brief._row_widths)


def test_the_row_gives_way_in_a_stated_order():
    """The "Candidates" label drops, then the count, then the rail shortens,
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
    # The label is the first to go, then the count -- both only once the prompt
    # (the Source chip, on Mesh) has bottomed out.
    assert body.index("show_label = False") < body.index("show_count = False")
    assert "reset_compact = True" in body


def _ladder(frames, counts: tuple[int, ...]) -> list[str]:
    """The order in which things give way as the window narrows, read off a real
    frame at every width from 1600 down to 240 dp -- so a change to the ladder
    on *one* stage's count set is a failure rather than a surprise."""
    items = _synthetic_rail_items()
    steps: list[tuple[float, Any, float]] = []

    def build_at(width: float) -> None:
        def build() -> None:
            rail_full = create_rail.stage_rail_width(items, "reference")
            rail_floor = create_rail.stage_rail_width(items, "reference", max_width=0.0)
            steps.append(
                (width, create_brief._row_widths(False, rail_full, rail_floor, counts), rail_full)
            )

        frames(build, (width, 700.0))

    for width in range(1600, 230, -40):
        build_at(float(width))

    order: list[str] = []
    for _width, widths, rail_full in steps:
        gone = {
            "label": not widths.show_label,
            "count": not widths.show_count,
            "rail": widths.rail_w < rail_full,
            "reset": widths.reset_compact,
        }
        for name, is_gone in gone.items():
            if is_gone and name not in order:
                order.append(name)
    return order


def test_the_four_rung_ladder_gives_way_at_decreasing_widths(frames):
    """``_row_widths`` walked at real, decreasing window widths -- a real
    frame feeding it real ``imgui.get_style()``/``get_content_region_avail()``
    numbers rather than a source scan. No GL: this only needs the numbers
    ``create_rail.stage_rail_width`` and ``imgui.calc_text_size`` already
    produce without a renderer.
    """
    items = _synthetic_rail_items()
    seen: dict[float, Any] = {}

    def measure(width: float) -> None:
        def build() -> None:
            rail_full = create_rail.stage_rail_width(items, "reference")
            rail_floor = create_rail.stage_rail_width(items, "reference", max_width=0.0)
            seen[width] = create_brief._row_widths(False, rail_full, rail_floor)

        frames(build, (width, 700.0))

    for width in (1400.0, 700.0, 500.0, 320.0):
        measure(width)

    widths = sorted(seen)
    rail_w = [seen[w].rail_w for w in widths]
    prompt_w = [seen[w].prompt_w for w in widths]
    show_label = [seen[w].show_label for w in widths]
    show_count = [seen[w].show_count for w in widths]
    reset_compact = [seen[w].reset_compact for w in widths]

    # Nothing here gets *more* room as the window gets narrower.
    assert rail_w == sorted(rail_w)
    assert prompt_w == sorted(prompt_w)
    # The label and the count are shown only once there is room for them (False
    # before True, narrow to wide) and never flip back as the window widens.
    assert show_label == sorted(show_label)
    assert show_count == sorted(show_count)
    # Reset is icon-only only under pressure (True before False, narrow to
    # wide) and never flips back either.
    assert reset_compact == sorted(reset_compact, reverse=True)
    # At the widest, everything is at its natural size.
    assert show_label[-1] is True and show_count[-1] is True
    assert reset_compact[-1] is False
    # At the narrowest, the ladder has bottomed out on both ends.
    assert prompt_w[0] == pytest.approx(create_brief.sp(create_brief.PROMPT_MIN_W))
    assert reset_compact[0] is True


def test_the_ladder_gives_way_in_the_same_order_on_both_stages(frames):
    """The label first, then the count, then the rail, then Reset -- for
    Reference's 1/2/4/8 and for Mesh's 1/2/3 alike. One ladder, two count sets:
    a bar that lost its label at a different width order on each stage would be
    two bars."""
    expected = ["label", "count", "rail", "reset"]
    assert _ladder(frames, create_brief._COUNTS) == expected
    assert _ladder(frames, create_brief._MESH_COUNTS) == expected


# --- the anchors the guided tour points at ----------------------------------


def test_the_tour_anchors_moved_with_the_controls():
    """``studio/tour/scripts.py`` binds steps to ``create/prompt`` and
    ``create/generate``. The tour reads positions and never computes them, so
    the anchors keep working precisely because they are marked at the controls'
    new home rather than left behind at the old one."""
    source = inspect.getsource(create_brief)
    assert 'anchors.mark("create/prompt")' in source
    assert 'anchors.mark("create/generate")' in source

    from realmspinner.studio.modes.create.ui.panes import settings_2d

    pane = inspect.getsource(settings_2d)
    assert 'anchors.mark("create/prompt")' not in pane
    assert 'anchors.mark("create/generate")' not in pane


def test_the_tours_still_name_anchors_something_marks():
    """Both directions, so a rename on either side fails here rather than as a
    tour step pointing at nothing."""
    from realmspinner.studio import anchors as anchors_mod  # noqa: F401
    from realmspinner.studio.tour import scripts

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
    from realmspinner.studio import main

    source = inspect.getsource(main.App._build_ui)
    assert 'layout_mod.pane(\n                        "brief",' in source
    assert "create_brief.bar_height(ctx)" in source
    assert "create_brief.draw(ctx, self._stage_rail)" in source
    # The gate moved *into* create_brief.draw, not away entirely.
    assert "create_brief.shows(ctx)" not in source
    assert "if not shows(ctx):" in inspect.getsource(create_brief.draw)


# --- the disabled button's reason -------------------------------------------


def test_the_disabled_generate_wears_the_first_problem_as_its_reason():
    """``problems.Problem`` is a ``str`` subclass -- the message *is* the object,
    and there is no ``.message`` on it.

    This branch only runs when the form is *invalid*, which is why neither the
    suite nor a screenshot of a seeded (valid) form ever executed it: the bar
    raised ``AttributeError`` and the pane guard blanked it, on every frame with
    an empty prompt. ``/exercise-mode create`` is what found it.
    """
    from realmspinner.studio import problems

    problem = problems.Problem("Describe what to generate.", "prompt")
    assert not hasattr(problem, "message")
    assert str(problem) == "Describe what to generate."

    body = _body(create_brief._generate)
    assert ".message" not in body, "Problem is a str; there is no .message"
    assert "str(problems[0])" in body


def test_an_empty_prompt_leaves_the_bar_drawable():
    """The whole-object check behind the test above: every problem the column's
    validator can raise for an untouched form must render as a reason string."""
    from realmspinner.studio.modes.create.engine import recipe as create_recipe

    form = default_form_2d()
    form["prompt"] = ""
    problems = create_recipe.validate(form)
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
        calls.append(ctx.state.create.stage)

    for stage in create_stages.STAGES:
        ctx = _real_ctx(stage=stage)
        frames(lambda ctx=ctx: create_brief.draw(ctx, rail_stub))

    assert calls == list(create_stages.STAGES)


def test_the_bar_only_reaches_the_settings_door_on_the_generating_stages(frames):
    """``ctx.busy`` is read only inside the press half of ``draw`` (to decide
    whether Generate is enabled) -- a call on Rig, Pose or Export would mean
    dead controls' plumbing still ran even though nothing draws."""
    busy_calls: list[str] = []

    def rail_stub(ctx, *, max_width=None, row_height=None):
        return None

    for stage in create_stages.STAGES:
        ctx = _real_ctx(stage=stage, source=_SOURCE)
        ctx.busy = lambda key, stage=stage: busy_calls.append(stage) or False
        frames(lambda ctx=ctx: create_brief.draw(ctx, rail_stub))

    assert set(busy_calls) == {"reference", "mesh"}
    assert busy_calls.index("reference") < busy_calls.index("mesh")


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
        ctx = _real_ctx(stage=stage, source=_SOURCE)
        positions: dict[str, float] = {}

        def build(ctx=ctx, positions=positions) -> None:
            positions["before"] = imgui.get_cursor_pos_y()
            create_brief.draw(ctx, _synthetic_rail)
            positions["after"] = imgui.get_cursor_pos_y()

        frames(build)

        pad = imgui.get_style().window_padding.y
        content_h = (positions["after"] - positions["before"]) + 2 * pad
        declared = create_brief.bar_height(ctx)
        generating = stage in ("reference", "mesh")
        assert declared == create_brief.sp(
            create_brief.BAR_H if generating else create_brief.RAIL_ONLY_H
        ), f"{stage}: the height is the generating stages' pair, not one alone"
        assert content_h <= declared, (
            f"stage={stage}: the row drew {content_h:.1f}px of content against "
            f"a declared {declared:.1f}px ({'BAR_H' if generating else 'RAIL_ONLY_H'} "
            f"= {create_brief.BAR_H if generating else create_brief.RAIL_ONLY_H})"
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
    for stage in ("reference", "mesh"):
        ctx = _real_ctx(stage=stage, source=_SOURCE)

        def build(ctx=ctx) -> None:
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
        assert reset.where == "brief", (stage, census)


def test_reset_no_longer_draws_from_the_settings_column():
    """The other half of the move: neither column holds a copy of it or of the
    press -- one owner per control, per ``CLAUDE.md``. Extended to Mesh, whose
    column used to end in Candidates, Reset and Make 3D."""
    from realmspinner.studio.modes.create.ui.panes import settings_2d, settings_3d

    source = inspect.getsource(settings_2d)
    assert "_reset_row" not in source
    assert "Reset..." not in source

    mesh = inspect.getsource(settings_3d)
    assert "_reset_row" not in mesh
    assert "Reset..." not in mesh
    assert "primary_button" not in mesh, "the press is the bar's"
    assert "radio_button" not in mesh, "Candidates is the bar's"
    assert "def _candidates" not in mesh and "def _submit" not in mesh
    assert 'widgets.section("Source")' not in mesh and "def _source(" not in mesh


def test_both_stages_reset_in_one_pattern(frames, monkeypatch):
    """"Reset the {image|mesh} settings?" for the confirm's title and "The
    {image|mesh} settings are back to their defaults." for the toast -- the
    same sentence with the stage's own noun, read off a real press."""
    real = create_brief.controls.button

    def press_reset(label, *a, **k):
        real(label, *a, **k)
        return str(label).startswith("Reset")

    monkeypatch.setattr(create_brief.controls, "button", press_reset)
    for stage, noun in (("reference", "image"), ("mesh", "mesh")):
        ctx = _real_ctx(stage=stage, source=_SOURCE)
        frames(lambda ctx=ctx: create_brief.draw(ctx, _synthetic_rail))
        dialog = ctx.asked[0]
        assert dialog.title == f"Reset the {noun} settings?"
        dialog.on_confirm()
        assert ctx.toasts[-1] == f"The {noun} settings are back to their defaults."
    assert ctx.asked[0].message == create_brief._RESET_MESH_CONFIRM_MESSAGE


def test_mesh_reset_puts_the_seed_back_to_unset():
    from realmspinner.studio.modes.create.ui.panes import settings_3d

    ctx = _real_ctx(stage="mesh")
    ctx.state.form_3d.update(mesh_seed=7, mesh_seed_locked=True, count=3)
    settings_3d._reset(ctx)
    assert ctx.state.form_3d["mesh_seed"] is None
    assert ctx.state.form_3d["count"] == 1


def test_a_second_undecided_group_does_not_hide_the_older_one():
    """The 2026-09-18 audit, finding create-01, reproduced by
    ``probes/create-workspace-01.py``: group A settled (both members
    ``done``) but was never decided (kept or discarded), and group B was
    submitted after it. ``candidates.pending`` only ever offers the newest
    group, so nothing on screen ever pointed back at A -- and a third
    submission would have orphaned it for good, contradicting
    ``state.Filters.matches``'s own promise that nothing stays hidden without
    a picker able to reach it. Refusing a *further* submission while any
    group is still undecided means there is never a second group for the
    first to disappear behind.
    """
    jobs = [
        {"id": "a1", "candidate_group": "groupA", "candidate_index": 0,
         "status": "done", "created_at": 100.0},
        {"id": "a2", "candidate_group": "groupA", "candidate_index": 1,
         "status": "done", "created_at": 100.0},
        {"id": "b1", "candidate_group": "groupB", "candidate_index": 0,
         "status": "queued", "created_at": 200.0},
        {"id": "b2", "candidate_group": "groupB", "candidate_index": 1,
         "status": "queued", "created_at": 200.0},
    ]
    ctx = SimpleNamespace(cache=SimpleNamespace(jobs=jobs))

    problems = create_brief._with_pending_candidates_problem(ctx, [])

    assert problems, "a further submission must be refused while group A is still undecided"
    assert "pending" in str(problems[0]).lower()


def test_no_pending_group_adds_no_problem():
    ctx = SimpleNamespace(cache=SimpleNamespace(jobs=[]))

    assert create_brief._with_pending_candidates_problem(ctx, []) == []


def test_missing_cache_does_not_crash_the_bar():
    """Some construction paths in this file's own ``_real_ctx``/``_state``
    stubs carry no ``cache`` at all; the guard must degrade rather than
    raise."""
    ctx = SimpleNamespace()

    assert create_brief._with_pending_candidates_problem(ctx, []) == []
