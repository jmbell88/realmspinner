"""Create's command bar: the stage rail and, on the Reference stage, the
brief -- one row, drawn through one pane.

**What a press needs, on one row, never scrolled.** The four brief controls
here -- what to make, the words, how many, and the button -- were the top and
the bottom of a 316 dp column with six sections between them, so the prompt
sat fourth behind two dropdowns and Generate sat under a scroll. They are the
only four a common visit touches, and they are the four that never fit
together.

**The rail joined this row rather than sitting above it (2026-09-07).** It
used to draw bare, full width, into ``##content`` above a *second*, separate
pane holding the brief -- two vertical strips (~90 dp together) for what a
common visit reads as one control bar: where this asset is, and what to make
next. Sharing one line costs real width -- see :func:`_row_widths` for the
give-way order that pays for it -- but it is what turns "breadcrumb, then a
second bar" back into one row, which is the same trade the brief itself made
against the six-section column it replaced.

The recipe stays in the settings column (``panes/settings_2d``), which is the
other half of the split: this bar is *what to make*, and that column is *how*.
A control belongs to exactly one of them, the same one-owner rule the two
generation panes already keep.

**One bar on both generating stages, and the rail alone on the rest.** Reference
and Mesh each *generate*, so each draws the same row: the rail, then what to
generate (Reference: the type and the prompt; Mesh: a Source chip -- a
thumbnail and the name of the chosen reference, or "Choose an image..."), then
Count, Generate and Reset. The Count pills' range and Generate's label come
from the stage (1/2/4/8 and the asset type's label; 1/2/3 and "Make 3D"), and
Reset asks and toasts in one pattern ("Reset the {image|mesh} settings?").
Below the bar, the stage's left column holds settings only and never a submit
button. Rig, Pose and Export make nothing from this row, so they draw the rail
alone -- :func:`shows` says so -- and the pane shrinks to the rail's own height
for them (:func:`bar_height`); nothing reserves an empty strip under a bare
rail. That is ``create_stages``' own rule about the rail, now applied one level
down: a bar that is present but inert is worse than a bar that is absent. The
rail itself is unconditional -- it is the breadcrumb for every stage -- which is
why :func:`draw` runs at every stage while :func:`shows` gates only the rest.

Drawn through :func:`layout.pane` rather than bare, the way the brief always
was. That is what puts this row in ``layout.FRAME_PANES``, which is what gives
it the role fill, the divider, ``guard``'s error isolation, and a pane slot
for ``probe._pane_at`` -- the rail did not have any of that while it drew
straight into the shared content child, and without it ``/exercise-mode
create`` reported the bar's controls against the empty-string pane, which
reads downstream as controls nobody owns.

**The rail is drawn through a callable, never imported and called here.**
``App._stage_rail`` reads a job, an on-disk rig and the preview state to
build ``done``/``optional``, and its click writes ``state.create.stage``
through ``create_stages.go`` -- documented as "the one stage switch." Giving
this module that logic directly would make it a second place the switch could
fire from; instead ``main.py`` hands its own bound ``_stage_rail`` in as
``rail`` and this module only ever calls what it is given, at the width its
own :func:`_row_widths` computes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

from imgui_bundle import imgui

from .... import anchors, controls, dialogs, focus, icons, theme, tokens, widgets
from ....tokens import sp
from ..engine import assets as create_assets
from ..engine import mesh as create_mesh
from ..engine import recipe as create_recipe
from . import rail as create_rail

#: The pane's height in design pixels, on the Reference stage -- see
#: :func:`bar_height` for the other four. Measured rather than derived: the
#: naive arithmetic, ``PROMPT_H + 2 * PANE_PADDING`` (64), was tried first and
#: was wrong, the same way Muse's own ``BAR_H`` was wrong before it grew a
#: version of this same test (118 declared against ~142 dp of real content).
#: 64 does not charge for imgui's own trailing ``item_spacing`` after the
#: row's last item (Reset), which a bare arithmetic guess has no way to know
#: to add -- ``test_the_bar_fits_the_height_it_declares`` draws the row for
#: real and this is what it measured, not what the arithmetic guessed.
BAR_H = 96.0

#: The pane's height at the four stages that draw the rail alone -- see
#: :func:`bar_height`. Also measured by the same test, for the same reason:
#: the rail's own content is ~25 dp (see ``create_rail.stage_rail``'s
#: ``row_height`` note) but the pane it sits in also has to fit imgui's
#: trailing ``item_spacing`` past it.
RAIL_ONLY_H = 65.0

#: The prompt field's own height. Two lines rather than one, because
#: ``MAX_PROMPT`` is a thousand characters and a single-line input for a
#: paragraph shows the tail and hides the subject. It scrolls past two.
PROMPT_H = 40.0

#: Fixed widths for the controls that flank the prompt, which takes what is
#: left. Design pixels; ``sp`` scales them.
#:
#: ``PROMPT_MIN_W`` is where the prompt stops shrinking and the count is
#: dropped instead -- see :func:`_row_widths`. ``COUNT_W`` is what
#: ``imgui.push_item_width`` reserves for the pills' *fields*, kept because
#: ``_count`` still pushes it -- the pills themselves are buttons, which the
#: push does nothing to, so the row's own reservation for them is
#: :func:`_count_width`'s measurement instead (2026-09-07: the old constant,
#: 124, was smaller than the four pills' real drawn width, which is what
#: pushed Generate past the pane edge).
PROMPT_MIN_W = 150.0
TYPE_W = 138.0
COUNT_W = 124.0
GENERATE_W = 158.0

#: The Reference stage's count values. 8 is ``validation.MAX_REFERENCE_COUNT``.
_COUNTS: tuple[int, ...] = (1, 2, 4, 8)

#: The Mesh stage's, from the service's own ceiling (``MAX_MESH_CANDIDATES``,
#: 3): the pills are the *same control*, and only their range is the stage's.
_MESH_COUNTS: tuple[int, ...] = tuple(range(1, create_mesh.MAX_MESH_CANDIDATES + 1))

#: The visible label beside the pills, on both stages -- four bare digits said
#: nothing about what they choose, and this is the first thing to drop out as
#: the bar narrows (:func:`_row_widths`).
COUNT_LABEL = "Candidates"

#: Per-pill hover text for the count control, the same shape as
#: ``_TYPE_HINTS`` beside it. Four bare pills ("1 2 4 8") carried no label and
#: no tooltip -- unlike the Type combo, which names each value on hover -- so
#: a first-time user had nothing on screen or on hover saying they choose how
#: many candidates one press draws. The 2026-09-05 audit, finding create-11.
_COUNT_HINTS: dict[str, str] = {
    "1": "Draw one candidate.",
    "2": "Draw two candidates to compare.",
    "4": "Draw four candidates to compare.",
    "8": "Draw eight, the most one press can generate.",
}

#: The same shape for Mesh, whose count is attempts on one reference.
_MESH_COUNT_HINTS: dict[str, str] = {
    "1": "Reconstruct the reference once.",
    "2": "Reconstruct it twice, each from a new seed, and keep the better one.",
    "3": "Reconstruct it three times, each from a new seed -- the most one press buys.",
}

#: This bar's key in the focus ring. Its own rather than ``settings_2d``'s
#: "2d": the ring is walked per pane, and the two are two panes.
FOCUS_PANE = "brief"


#: The stages whose bar carries the press. Every other stage is the rail alone.
GENERATING_STAGES: tuple[str, ...] = ("reference", "mesh")


def shows(ctx: Any) -> bool:
    """Whether the bar has a press to draw beside the rail. -> on the
    Reference and Mesh stages, and nowhere else.

    Stopped gating the pane itself (2026-09-07): the rail is the row's
    breadcrumb for every stage, so the pane always opens, and this predicate
    decides only how much of the row :func:`draw` fills in and how tall
    :func:`bar_height` makes it.
    """
    from . import stages as create_stages

    return any(create_stages.at(ctx.state, stage) for stage in GENERATING_STAGES)


def _is_mesh(ctx: Any) -> bool:
    from . import stages as create_stages

    return create_stages.at(ctx.state, "mesh")


def bar_height(ctx: Any) -> float:
    """This frame's pane height. The full row on a generating stage; just the
    rail's own band everywhere else -- an inert strip under a bare rail is the
    same complaint the module docstring already makes about a bar with dead
    controls, so the other stages simply do not reserve one.
    """
    return sp(BAR_H) if shows(ctx) else sp(RAIL_ONLY_H)


def draw(ctx: Any, rail: Callable[..., None]) -> None:
    """The row. Called from ``main._build_ui`` for the one pane that holds the
    stage rail and, on Reference and Mesh, the rest of the bar.

    ``rail`` is ``App._stage_rail``, bound -- see the module docstring for why
    it is handed in rather than called through an import here.
    """

    state = ctx.state
    form_2d = state.form_2d
    # The same synchronisation the settings column runs, and for the same
    # reason: the five derived door fields are a function of the type, and the
    # type is edited *here* now, on the Reference stage. Running it before the
    # early return keeps every stage's rail agreeing with the column about
    # what the asset currently is, not only Reference's.
    if "asset_type" not in form_2d:
        form_2d["asset_type"] = create_assets.legacy_asset_type(form_2d)
    spec = create_assets.sync_legacy_fields(form_2d)

    if not shows(ctx):
        # Rig, Pose, Export: the rail alone, at whatever the pane has --
        # there is nothing else on the row competing for it, so none of
        # ``_row_widths``' give-way ladder applies.
        rail(ctx, max_width=imgui.get_content_region_avail().x)
        return

    from .panes import settings_3d

    mesh = _is_mesh(ctx)
    form = state.form_3d if mesh else form_2d
    counts = _MESH_COUNTS if mesh else _COUNTS
    focus.pump(state, FOCUS_PANE)
    focus.begin(state, FOCUS_PANE)

    # **Both sheet doors and the character door make exactly one thing per
    # press.** ``sync_legacy_fields`` has already written ``count = 1`` for all
    # three, so four radios of which three are refusals would be a control
    # offering what the thing behind it will not do. Mesh has no such door.
    hide_count = (not mesh) and form.get("output") in ("sheet", "character")
    if mesh:
        problems = settings_3d.problems(ctx, settings_3d.bar_source(ctx))
        current = create_mesh.candidate_count(form)
    else:
        problems = create_recipe.problems_for(ctx, form)
        problems = _with_pending_candidates_problem(ctx, problems)
        current = int(form["count"])
    busy = ctx.busy("submit")

    rail_full_w, rail_floor_w = _rail_measurements(state.create.stage)
    rail_w, prompt_w, show_label, show_count, reset_compact = _row_widths(
        hide_count, rail_full_w, rail_floor_w, counts
    )

    rail(ctx, max_width=rail_w, row_height=sp(PROMPT_H))
    imgui.same_line()
    if mesh:
        # The Source chip takes the type combo's slot *and* the prompt's, so
        # everything after it lands where it does on Reference.
        _source_chip(ctx, sp(TYPE_W) + imgui.get_style().item_spacing.x + prompt_w)
    else:
        _type(ctx, form)
        imgui.same_line()
        _prompt(ctx, form, prompt_w)
    imgui.same_line()
    if show_count:
        _count(ctx, form, counts, current, show_label=show_label)
        imgui.same_line()
    _generate(ctx, form, label="Make 3D" if mesh else spec.create_label,
              enabled=not problems and not busy, problems=problems,
              show_count=show_count, count=current,
              press=(lambda: settings_3d.promote(ctx, ctx.cache.get(ctx.state.source_job), form))
              if mesh else None)
    imgui.same_line()
    _reset(ctx, compact=reset_compact, mesh=mesh)


def _with_pending_candidates_problem(ctx: Any, problems: list[Any]) -> list[Any]:
    """Add a refusal while an undecided candidate group exists. -> ``problems``.

    The 2026-09-18 audit, finding create-01: a second candidate-producing
    brief used to hide the *older* undecided group with nothing on screen --
    ``candidates.pending`` only ever offers the newest group, and
    ``state.Filters.matches`` hides every ``candidate_group`` row from the
    library regardless of age, so a third submission before the second is
    decided orphaned the first forever. Refusing a new submission while any
    group -- decided-looking or not -- is still pending means there is never
    a second group to hide the first behind.
    """
    from .... import candidates as candidates_mod

    cache = getattr(ctx, "cache", None)
    jobs = getattr(cache, "jobs", None) if cache is not None else None
    if jobs is None or candidates_mod.pending_cached(cache) is None:
        return problems
    from ....problems import Problem

    return [*problems, Problem("Decide the pending candidates first.")]


def _rail_items_for_measurement() -> list[tuple[str, str, str, str | None]]:
    """The rail's five ``(key, label, icon, reason)`` entries, for
    :func:`create_rail.stage_rail_width` alone -- **not** what
    ``App._stage_rail`` hands ``create_rail.stage_rail`` to actually *draw*
    the rail.

    That real list carries each stage's live availability and its done-set,
    which need a job and (for Rig) a filesystem read this module has no
    business making just to size a row. ``stage_rail_width`` only reads the
    label text to measure it, and an unavailable stage's label is the same
    string as an available one's, so ``reason=None`` throughout costs the
    measurement nothing.
    """
    from . import stages as create_stages

    return [
        (stage, create_stages.LABELS[stage], create_stages.ICONS[stage], None)
        for stage in create_stages.STAGES
    ]


def _rail_measurements(current: str) -> tuple[float, float]:
    """-> ``(rail_full_w, rail_floor_w)``, what :func:`_row_widths` needs to
    know before it decides how much of the row the rail keeps.

    The 2026-09-26 audit, finding create-brief-01: ``rail_full_w`` used to
    come from ``create_rail.stage_rail_width(items, current)`` with no
    ``done`` at all, so ``_rail_fit``'s "ticks" rung measured every stage as
    its plain label -- identical to the "labels" rung -- and the number
    returned was too narrow for what the rail actually draws the moment any
    stage is really done. ``rail`` (``App._stage_rail``, bound in as
    :func:`draw`'s own ``rail`` argument) is then handed that undersized
    width as its ``max_width``, so its own ``_rail_fit`` call -- this time
    with the *real* done set -- never fit the ticks rung either, and dropped
    the checks even on a row with genuine room to spare.

    :func:`_rail_items_for_measurement` still has no business reading a job
    or the filesystem just to size a row, so every stage but ``current`` is
    measured here as if it were done -- the true done set can never make the
    ticks rung any wider than that, only narrower, so this never
    under-measures.
    """
    items = _rail_items_for_measurement()
    worst_case_done = frozenset(key for key, *_rest in items)
    rail_full_w = create_rail.stage_rail_width(items, current, done=worst_case_done)
    rail_floor_w = create_rail.stage_rail_width(items, current, max_width=0.0)
    return rail_full_w, rail_floor_w


class Widths(NamedTuple):
    """What :func:`_row_widths` decided, by name -- five answers were a tuple
    nobody could read at a call site."""

    rail_w: float
    prompt_w: float
    show_label: bool
    show_count: bool
    reset_compact: bool


def _row_widths(
    hide_count: bool,
    rail_full_w: float,
    rail_floor_w: float,
    counts: tuple[int, ...] = _COUNTS,
) -> Widths:
    """**The row's give-way order**, one ladder for both stages.

    Measured *before anything on the row has drawn*, which is why the type
    combo's width is subtracted explicitly, once, here: the rail is the row's
    first element, so nothing has narrowed ``avail`` by the time it has to
    decide how wide to be, and double-counting ``TYPE_W`` once turned a resize
    floor into a Generate drawn past the pane edge -- ``same_line`` past the
    content region "draws a control nowhere". The Mesh stage's Source chip
    takes the type combo's slot and the prompt's together (its width is
    ``TYPE_W`` + gap + ``prompt_w``), so the same arithmetic describes it and
    Generate lands in the same place on both stages.

    In the order things go as the bar narrows:

    1. The **"Candidates" label** drops (the pills stay, each with its own
       tooltip).
    2. The **count** is dropped -- ``_generate_tooltip`` restates its value
       once its pills are gone. Each rung runs only while the prompt (the chip,
       on Mesh) is still under ``PROMPT_MIN_W``.
    3. The **rail** is handed whatever is left as its own ``max_width`` and
       walks its own three rungs (``create_rail.stage_rail``: checks+labels,
       labels, icons). Every rung keeps every stage clickable and tooltipped,
       which is what makes it the right thing to give away next -- unlike the
       count or Reset, nothing about the rail actually disappears; it only
       gets terser. ``rail_full_w`` and ``rail_floor_w`` come from
       ``create_rail.stage_rail_width``, never guessed: 304 was a guess once,
       and wrong the moment a label changed.
    4. **Reset** drops to icon-only, its label moved to a tooltip, only if
       Reset at full width would leave the rail short of its own icon floor.

    The type (or chip) and Generate never give way: they are what the bar is for.
    """
    gap = imgui.get_style().item_spacing.x
    avail = imgui.get_content_region_avail().x
    fixed = sp(TYPE_W) + sp(GENERATE_W)
    reset_full = _reset_width(compact=False)
    reset_icon = _reset_width(compact=True)

    def gaps_for(show_count: bool) -> float:
        # rail, type, prompt, [count], generate, reset -- N items, N-1 gaps.
        elements = 5 + (1 if show_count else 0)
        return gap * (elements - 1)

    def prompt_for(show_count: bool, show_label: bool) -> float:
        count_w = _count_width(counts, label=show_label) if show_count else 0.0
        return avail - fixed - count_w - reset_full - rail_full_w - gaps_for(show_count)

    show_count = not hide_count
    show_label = show_count
    prompt = prompt_for(show_count, show_label)

    if prompt < sp(PROMPT_MIN_W) and show_label:
        # Rung 1: the label goes.
        show_label = False
        prompt = prompt_for(show_count, show_label)
    if prompt < sp(PROMPT_MIN_W) and show_count:
        # Rung 2: the count goes.
        show_count = False
        prompt = prompt_for(show_count, show_label)

    if prompt >= sp(PROMPT_MIN_W):
        return Widths(rail_full_w, prompt, show_label, show_count, False)

    # Rung 3: the prompt is pinned at its floor and the rail gives up the
    # width it would have kept -- handed only what's left, which is where
    # ``create_rail.stage_rail``'s own laddering takes over.
    prompt = sp(PROMPT_MIN_W)
    rail_w = avail - fixed - reset_full - prompt - gaps_for(show_count)
    reset_compact = False
    if rail_w < rail_floor_w:
        # Rung 4: even Reset at full width would leave the rail short of its
        # own icon floor -- Reset gives up its label, freeing the difference.
        reset_compact = True
        rail_w += reset_full - reset_icon
    return Widths(max(rail_w, rail_floor_w), prompt, show_label, show_count, reset_compact)


def _count_width(counts: tuple[int, ...] = _COUNTS, *, label: bool = False) -> float:
    """The pills' real drawn width -- ``COUNT_W`` was a reservation nothing
    enforced -- plus the "Candidates" label's when it is on screen.

    ``controls.segmented_choice`` chains ``controls.button`` calls with a
    plain ``same_line()`` and no explicit width, so each pill is exactly as
    wide as ``imgui.button`` draws its own label -- text plus the style's
    frame padding, twice -- which is what ``widgets.button_width`` measures.
    ``imgui.push_item_width`` (which ``_count`` still pushes, for the sibling
    widgets that do respect it) does nothing to a button. The old ``COUNT_W``
    was smaller than this real number, which is what pushed Generate past the
    pane edge -- this module's own words for exactly that: ``same_line`` past
    the content region "draws a control nowhere."
    """
    gap = imgui.get_style().item_spacing.x
    widths = [widgets.button_width(str(n)) for n in counts]
    total = sum(widths) + gap * (len(widths) - 1)
    if label:
        total += imgui.calc_text_size(COUNT_LABEL).x + gap
    return total


def _reset_width(*, compact: bool) -> float:
    """Reset's own two widths, measured the way :func:`_count_width` measures
    the pills: ``controls.button``'s auto width is text plus frame padding,
    twice, which is what actually lands on screen either way."""
    return widgets.button_width(icons.UNDO if compact else "Reset...")


def _type(ctx: Any, form: dict[str, Any]) -> None:
    """What to make. The one choice that decides what everything else means."""
    before = create_assets.selected(form).key
    with focus.item(ctx.state, FOCUS_PANE, "asset_type"):
        picked = widgets.combo(
            "##generation-type",
            before,
            list(create_assets.ASSET_TYPE_OPTIONS),
            width=sp(TYPE_W),
            tooltip=_TYPE_HINTS.get(before, ""),
        )
    form["asset_type"] = picked if picked in create_assets.ASSET_TYPES else before
    form["generation_type"] = form["asset_type"]
    if create_assets.sync_legacy_fields(form).key != before:
        ctx.state.clear_field_error("asset_type")
    _ring(ctx, "asset_type")


def _prompt(ctx: Any, form: dict[str, Any], width: float) -> None:
    """The words. The field this whole rearrangement is about.

    An explicit ``width``, which is the one thing that matters here:
    ``widgets.multiline`` defaults to -1, meaning *fill the row*. In a column
    that is what you want; on a row it takes the width the controls after it
    were going to use, and ``same_line`` past the pane edge draws a control
    nowhere -- so everything after it vanished off the right-hand side
    entirely.
    """
    before = form["prompt"]
    with focus.item(ctx.state, FOCUS_PANE, "prompt"):
        form["prompt"] = widgets.multiline(
            "##brief-prompt", before, sp(PROMPT_H), _max_prompt(), width=width
        )
        anchors.mark("create/prompt")
        widgets.char_count(form["prompt"], _max_prompt())
    if form["prompt"] != before:
        ctx.state.clear_field_error("prompt")
    if imgui.is_item_hovered() and not str(form["prompt"]).strip():
        imgui.set_tooltip("Describe one subject -- a prop, a character, a surface.")
    _ring(ctx, "prompt")


def _count(
    ctx: Any,
    form: dict[str, Any],
    counts: tuple[int, ...],
    current: int,
    *,
    show_label: bool,
) -> None:
    """How many alternatives one press should draw -- Candidates, on both stages.

    Not drawn at all for a sheet: both sheet doors refuse a batch and say why,
    so four radios of which three are refusals would be a control offering
    what the thing behind it will not do. ``sync_legacy_fields`` has already
    written ``count = 1`` for those. ``counts`` is the stage's range and
    ``current`` the value it reads back (Mesh clamps a stale stored one).

    The pills are ``tokens.CONTROL_HEIGHT_COMPACT`` (26 dp) tall against a
    ``PROMPT_H`` (40 dp) row (B3, 2026-09-07): a bare ``same_line`` left them
    flush with the row's *top* rather than its middle, beside a two-line
    prompt field and a full-height Generate. The spacer dummies before and
    after reserve the full row height inside this group -- the same amount --
    so the row's shared baseline is exactly where it was without them, and
    only the pills themselves float centred inside it.
    """
    hints = _MESH_COUNT_HINTS if counts is _MESH_COUNTS else _COUNT_HINTS
    pill_h = sp(tokens.CONTROL_HEIGHT_COMPACT)
    pad = max(0.0, (sp(PROMPT_H) - pill_h) * 0.5)
    imgui.begin_group()
    if pad:
        imgui.dummy((1.0, pad))
    if show_label:
        # Centred on the pills by a spacer inside a group of its own:
        # ``align_text_to_frame_padding`` would make the line as tall as a full
        # *frame*, which is taller than a 26 dp pill and grows the whole bar.
        text_h = imgui.get_text_line_height()
        off = max(0.0, (pill_h - text_h) * 0.5 - imgui.get_style().item_spacing.y)
        imgui.begin_group()
        if off:
            imgui.dummy((1.0, off))
        widgets.muted(COUNT_LABEL)
        imgui.end_group()
        imgui.same_line()
    imgui.push_item_width(sp(COUNT_W))
    with focus.item(ctx.state, FOCUS_PANE, "count") as focused:
        changed, picked = controls.segmented_choice(
            "brief-count",
            tuple((str(n), str(n)) for n in counts),
            str(current),
            compact=True,
            tooltips=hints,
        )
        if changed:
            form["count"] = int(picked)
            ctx.state.clear_field_error("count")
        # Hand-answered, as it was in the column: a row of radios is one
        # control to the keyboard even though it is four items to imgui.
        if focused:
            here = counts.index(current) if current in counts else 0
            before_arrow = form["count"]
            if imgui.is_key_pressed(imgui.Key.left_arrow):
                form["count"] = counts[(here - 1) % len(counts)]
            if imgui.is_key_pressed(imgui.Key.right_arrow):
                form["count"] = counts[(here + 1) % len(counts)]
            # The click branch above clears the ring on a change; this
            # hand-rolled branch edits ``form["count"]`` the same way and must
            # clear the same error, or a user who fixes an invalid count with
            # the keyboard instead of a click keeps a ring pointing at a value
            # that is no longer wrong. The 2026-09-05 audit, finding create-07.
            if form["count"] != before_arrow:
                ctx.state.clear_field_error("count")
    imgui.pop_item_width()
    _ring(ctx, "count")
    if pad:
        imgui.dummy((1.0, pad))
    imgui.end_group()


def _generate(
    ctx: Any,
    form: dict[str, Any],
    *,
    label: str,
    enabled: bool,
    problems: list[Any],
    show_count: bool,
    count: int,
    press: Callable[[], None] | None = None,
) -> None:
    """The press. Always visible, which is the point of the bar.

    ``label`` is the stage's own (the asset type's on Reference, "Make 3D" on
    Mesh) and ``press`` the stage's door -- ``None`` means Reference's
    ``settings_2d.generate``. Width and placement are the stage's business
    nowhere: :data:`GENERATE_W` on both.

    The *reason* it is disabled is the button's own tooltip, and the pinned
    plan footer under the stage's column lists every problem and offers the
    one-click repairs: a bar has no room for a list, and a button that says
    nothing about why it is dead is the complaint this redesign started from.

    ``show_count`` is ``_row_widths``' own answer, not re-derived: the count
    pills carry their own value the moment they are on screen, so restating it
    here as well would be a second control saying the same number an inch to
    its left. It is only appended once ``_row_widths`` has dropped them.
    """
    from .panes import settings_2d

    with focus.item(ctx.state, FOCUS_PANE, "generate") as focused:
        pressed = widgets.primary_button(
            label,
            (sp(GENERATE_W), sp(PROMPT_H)),
            enabled=enabled,
            # ``Problem`` is a str subclass -- the message *is* the object.
            reason=str(problems[0]) if problems else "",
            tooltip=_generate_tooltip(show_count, count),
        )
        anchors.mark("create/generate")
        if focused and enabled and _enter_pressed():
            pressed = True
    if pressed:
        if press is not None:
            press()
        else:
            settings_2d.generate(ctx, form)


def _generate_tooltip(show_count: bool, count: int) -> str:
    """Ctrl+Enter, plus the count once its own pills are off screen.

    The shortcut stays first because that is what the tooltip is mainly for;
    the count is appended rather than replacing it, separated the same way a
    plan block reads a list of facts. Silent while the pills are visible --
    a tooltip repeating a control drawn an inch to its left is noise, not help.
    """
    if show_count:
        return "Ctrl+Enter"
    noun = "candidate" if count == 1 else "candidates"
    return f"Ctrl+Enter · {count} {noun}"


# The Reset confirm's own words, module-level so a regression test can read
# them without an imgui context (``_reset`` below draws a button first, which
# needs one). The 2026-09-08 audit, finding create-03: this text used to name
# only six things -- the prompt, the negative prompt, the model, the LoRA,
# the reference and the run controls -- while ``settings_2d._reset`` actually
# replaces the *whole* form with ``default_form_2d()``, silently discarding
# the asset type (the whole Image/3D Model/Seamless Material/Tileset/Sprite
# Sheet/Character selection) and every Tileset/Sprite/Character field the
# user had typed in (materials, variants, style_lock, seam_erase, palette,
# cell_size, and the rest) -- none of which the old text named. The fix
# broadens the *text* to match what Reset has always discarded, per the
# 2026-09-08 audit's own call: narrowing ``_reset`` instead would make "another
# like this" lose the asset type on every rerun, which is a bigger behaviour
# change than a confirm dialog earns on its own.
_RESET_CONFIRM_MESSAGE = (
    "The prompt, the negative prompt, the model, the LoRA, the reference and "
    "the run controls go back to their defaults, with a freshly rolled seed "
    "-- and so does everything else on this form, including the asset type "
    "(Image, 3D Model, Seamless Material, Tileset, Sprite Sheet or Character) "
    "and any Tileset, Sprite Sheet or Character fields you have filled in. "
    "The 3D form is untouched."
)


#: The Mesh stage's confirm, the same pattern as the image one above: it says
#: what *its* Reset replaces (the whole of ``DEFAULT_FORM_3D``) and what it
#: keeps. The seed goes back to *unset*, not to a rolled number.
_RESET_MESH_CONFIRM_MESSAGE = (
    "Every mesh setting -- the resolution, budget, size, background, seed, "
    "candidate count, rig and engine controls -- goes back to its default, "
    "with the seed unset. The chosen source is kept, and the image settings "
    "are untouched."
)


def _reset_title(noun: str) -> str:
    """One pattern for both stages: "Reset the image settings?" / "... mesh ...?"."""
    return f"Reset the {noun} settings?"


def _reset(ctx: Any, *, compact: bool, mesh: bool = False) -> None:
    """*Reset...*, beside Generate now rather than pinned above the settings
    column it used to sit atop -- on both generating stages.

    ``_reset_row``'s old home in ``settings_2d`` kept a rule its own docstring
    stated: *"Above the submit rather than below it: a destructive control
    under the primary action is one the hand reaches by accident."* This
    placement -- immediately after Generate on the same row -- is that same
    adjacency turned sideways, at the user's own request. Two things keep it
    honest rather than silently dropping the old argument: Reset stays a
    **GHOST** button, never a filled slab standing next to a primary action,
    so it does not compete with Generate for the eye the way a second solid
    button would; and it keeps the ``dialogs.Confirm`` it already had, which
    is the *only* guard against the accidental press now -- placement no
    longer is one.

    ``mesh`` picks the stage's own clearing (``settings_3d._reset`` owns
    ``form_3d``, ``settings_2d._reset`` owns ``form_2d``) and its noun; the
    confirm's title and both toasts follow one pattern.

    ``compact`` is ``_row_widths``' rung 4: past that width Reset draws as a
    bare glyph with its label moved to the tooltip, the same rule
    ``widgets.icon_button`` states for any control with no visible label --
    applied here through ``controls.button`` directly so the GHOST role
    survives the swap, which ``icon_button``'s own paint does not offer.
    """
    from .panes import settings_2d, settings_3d

    noun = "mesh" if mesh else "image"
    label = icons.UNDO if compact else "Reset..."
    if controls.button(
        label,
        (0, sp(PROMPT_H)),
        role=controls.ButtonRole.GHOST,
        tooltip=f"Reset the {noun} settings to their defaults." if compact else "",
    ):
        ctx.confirms.ask(
            dialogs.Confirm(
                title=_reset_title(noun),
                message=_RESET_MESH_CONFIRM_MESSAGE if mesh else _RESET_CONFIRM_MESSAGE,
                confirm_label="Reset",
                cancel_label="Cancel",
                on_confirm=(lambda: settings_3d._reset(ctx)) if mesh else (
                    lambda: settings_2d._reset(ctx)
                ),
            )
        )


def _source_chip(ctx: Any, width: float) -> None:
    """The Mesh stage's "what": the reference this press reconstructs.

    A thumbnail and the reference's name, or "Choose an image..." with a drop
    hint when there is none. **It is a real button drawn at ``width`` with its
    picture laid over it**, so it has a hit rect, a keyboard stop and a tooltip
    like any control, and it is a drop target for a library card dragged out of
    the Library and for a file dropped on the window (``events`` flashes it).
    Clicking it opens the file picker -- whose result is still *uploaded* as a
    new reference, today's behaviour, not a pick of an existing one.

    The name shown is the *effective* source (``settings_3d.bar_source``): the
    explicit pick, else the reference behind a selected finished mesh -- what
    the press will actually use, which is the whole job of the line it replaced.
    """
    from ....panes import thumbs
    from ...library.ui.panes import library
    from .panes import settings_3d

    state = ctx.state
    height = sp(PROMPT_H)
    source = settings_3d.bar_source(ctx)
    derived = source is not None and ctx.cache.get(state.source_job) is None
    dragging = library.dragged_job(ctx)
    picker_open = ctx.busy("upload")
    if source is None:
        title, sub = "Choose an image...", "or drop a card or a file here"
        tip = "Click to choose an image, or drop a library card or a file here."
    else:
        title = str(source.get("name") or source.get("prompt") or source["id"])
        sub = "this mesh's reference" if derived else f"reference - {source['id']}"
        tip = f"{title}\nClick to upload a different image, or drop a library card here."
    if dragging is not None:
        sub = "drop it here to use it"

    with focus.item(state, FOCUS_PANE, "source") as focused:
        clicked = controls.button(
            "##mesh-source",
            (width, height),
            role=controls.ButtonRole.GHOST,
            enabled=not picker_open,
            reason="A file picker is already open.",
            tooltip=tip,
        )
        if focused and not picker_open and _enter_pressed():
            clicked = True
    low = imgui.get_item_rect_min()
    high = imgui.get_item_rect_max()
    if clicked:
        ctx.submit("upload", dialogs.open_file, "Choose a reference image", dialogs.IMAGE_FILTER)
    if imgui.begin_drag_drop_target():
        payload = imgui.accept_drag_drop_payload_py_id(library.DRAG_JOB)
        if payload is not None and state.dragging_job:
            # Through ``library.select`` rather than by assigning ``source_job``
            # here: that function is what also moves the selection, so a
            # dropped card is the selected card and the inspector on the right
            # is showing the thing the bar now names.
            library.select(ctx, state.dragging_job)
            state.source_job = state.dragging_job
            state.dragging_job = None
        imgui.end_drag_drop_target()
    if dragging is not None:
        hovered = imgui.is_item_hovered(
            imgui.HoveredFlags_.allow_when_blocked_by_active_item.value
        )
        widgets.ring(
            low,
            high,
            theme.ACCENT if hovered else theme.MUTED,
            0.9 if hovered else 0.4,
            2.0 if hovered else 1.0,
        )
    else:
        # The same ring, fading, for a file dropped from Explorer (H70): the
        # two arrivals look the same because they are the same event.
        widgets.ring(low, high, theme.ACCENT, widgets.drop_flash(state, "3d-source"))

    # The picture laid over the button. Drawn after the drop target above,
    # which must attach to the *button* item, and never interactive itself.
    pad = sp(4.0)
    side = max(1.0, height - 2 * pad)
    imgui.set_cursor_screen_pos((low.x + pad, low.y + pad))
    if source is not None:
        thumbs.job_thumb(ctx, source, side)
    else:
        widgets.thumb_placeholder(side, icons.IMAGE)
    text_w = max(1.0, width - side - 3 * pad)
    text_h = imgui.get_text_line_height() * 2 + imgui.get_style().item_spacing.y
    imgui.set_cursor_screen_pos(
        (low.x + 2 * pad + side, low.y + max(0.0, (height - text_h) * 0.5))
    )
    imgui.begin_group()
    imgui.text(widgets.fit_text(title, text_w))
    widgets.muted(widgets.fit_text(sub, text_w))
    imgui.end_group()
    # Back to the row: the next control continues from the chip's right edge,
    # not from wherever the overlay left the cursor.
    imgui.set_cursor_screen_pos((high.x, low.y))
    imgui.dummy((0.0, 0.0))


def _ring(ctx: Any, field: str) -> bool:
    """Mark the control just drawn if a refusal named it. -> whether it did.

    ``widgets.field_error`` is the column's version and draws the message and
    any install offer *below* the control, which in a one-row bar would push
    the row apart. The ring alone here; the words are in the plan block.
    """
    if not (getattr(ctx.state, "field_errors", None) or {}).get(field):
        return False
    widgets.ring(
        imgui.get_item_rect_min(), imgui.get_item_rect_max(), theme.ERR, 0.9, thick=1.5
    )
    return True


def _max_prompt() -> int:
    """``MAX_PROMPT``, imported lazily so this module stays cheap to import."""
    from .....service.validation import MAX_PROMPT

    return MAX_PROMPT


def _enter_pressed() -> bool:
    return imgui.is_key_pressed(imgui.Key.enter) or imgui.is_key_pressed(
        imgui.Key.keypad_enter
    )


def _species_count() -> int:
    """How many species the character registry ships. Counted, never typed.

    ``characters.family`` imports nothing but the standard library, so reading
    it at import time here costs a dict copy and drags nothing in behind it.
    """
    from .....characters.family import families

    return len(families())


#: One line per type, on the combo's tooltip. The five-row descriptive block
#: that used to sit under the selector is gone with the column it sat in; this
#: is what survives of it, which is the orientation and not the prose.
_TYPE_HINTS: dict[str, str] = {
    "image": "A standalone 2D image.",
    "3d_model": "A reference image you can turn into a 3D model.",
    "seamless_material": "A seamless surface texture.",
    "tileset": "A coherent pixel-art tile sheet.",
    "sprite_sheet": "A character in several frames and directions.",
    # The real scope, in one line, because every other hint here describes an
    # SDXL request and this one describes the opposite: a body built from the
    # species registry, rigged and rendered on the CPU. "No GPU needed" is the
    # half a user with a small card most needs to read.
    #
    # The species count is *counted*, never typed: a sibling adding a row to
    # ``characters.family._FAMILIES`` must not leave a tooltip promising the
    # number the registry held last month.
    "character": (
        f"{_species_count()} species across four body plans, built and animated "
        f"into a sprite sheet. No GPU needed."
    ),
}

__all__ = [
    "BAR_H",
    "FOCUS_PANE",
    "GENERATING_STAGES",
    "RAIL_ONLY_H",
    "bar_height",
    "draw",
    "shows",
]
