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

**The four brief controls, Reference stage only.** Mesh, Rig, Pose and Export
draw the rail alone -- :func:`shows` says so -- and the pane shrinks to the
rail's own height for them (:func:`bar_height`); nothing reserves an empty
strip under a bare rail. That is ``create_stages``' own rule about the rail,
now applied one level down: shipping a row with one live control and three
dead ones is not honest, and a bar that is present but inert is worse than a
bar that is absent. The rail itself is unconditional -- it is the breadcrumb
for every stage, not only Reference's -- which is why :func:`draw` runs at
every stage while :func:`shows` gates only the brief.

Drawn through :func:`layout.pane` rather than bare, the way the brief always
was. That is what puts this row in ``layout.FRAME_PANES``, which is what gives
it the role fill, the divider, ``guard``'s error isolation, and a pane slot
for ``probe._pane_at`` -- the rail did not have any of that while it drew
straight into the shared content child, and without it ``/exercise-mode
create`` reported the bar's controls against the empty-string pane, which
reads downstream as controls nobody owns.

**The rail is drawn through a callable, never imported and called here.**
``App._stage_rail`` reads a job, an on-disk rig and the preview state to
build ``done``/``optional``, and its click writes ``state.create_stage``
through ``create_stages.go`` -- documented as "the one stage switch." Giving
this module that logic directly would make it a second place the switch could
fire from; instead ``main.py`` hands its own bound ``_stage_rail`` in as
``rail`` and this module only ever calls what it is given, at the width its
own :func:`_row_widths` computes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from imgui_bundle import imgui

from . import anchors, controls, create_assets, dialogs, focus, icons, theme, tokens, widgets
from .tokens import sp

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
#: the rail's own content is ~25 dp (see ``widgets.stage_rail``'s
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

#: This bar's key in the focus ring. Its own rather than ``settings_2d``'s
#: "2d": the ring is walked per pane, and the two are two panes.
FOCUS_PANE = "brief"


def shows(ctx: Any) -> bool:
    """Whether the *brief* -- the four controls beside the rail -- has
    anything true to say. -> only on the Reference stage.

    Stopped gating the pane itself (2026-09-07): the rail is the row's
    breadcrumb for every stage, so the pane always opens, and this predicate
    now decides only how much of the row :func:`draw` fills in and how tall
    :func:`bar_height` makes it.
    """
    from . import create_stages

    return create_stages.at(ctx.state, "reference")


def bar_height(ctx: Any) -> float:
    """This frame's pane height. The full row on Reference; just the rail's
    own band everywhere else -- an inert strip under a bare rail is the same
    complaint the module docstring already makes about a bar with three dead
    controls, so the four other stages simply do not reserve one.
    """
    return sp(BAR_H) if shows(ctx) else sp(RAIL_ONLY_H)


def draw(ctx: Any, rail: Callable[..., None]) -> None:
    """The row. Called from ``main._build_ui`` for the one pane that now
    holds the stage rail and, on the Reference stage, the rest of the brief.

    ``rail`` is ``App._stage_rail``, bound -- see the module docstring for why
    it is handed in rather than called through an import here.
    """
    from .panes import settings_2d

    state = ctx.state
    form = state.form_2d
    # The same synchronisation the settings column runs, and for the same
    # reason: the five derived door fields are a function of the type, and the
    # type is edited *here* now, on the Reference stage. Running it before the
    # early return keeps every stage's rail agreeing with the column about
    # what the asset currently is, not only Reference's.
    if "asset_type" not in form:
        form["asset_type"] = create_assets.legacy_asset_type(form)
    spec = create_assets.sync_legacy_fields(form)

    if not shows(ctx):
        # Mesh, Rig, Pose, Export: the rail alone, at whatever the pane has --
        # there is nothing else on the row competing for it, so none of
        # ``_row_widths``' give-way ladder applies.
        rail(ctx, max_width=imgui.get_content_region_avail().x)
        return

    focus.pump(state, FOCUS_PANE)
    focus.begin(state, FOCUS_PANE)

    # **Both sheet doors and the character door make exactly one thing per
    # press.** ``sync_legacy_fields`` has already written ``count = 1`` for all
    # three, so four radios of which three are refusals would be a control
    # offering what the thing behind it will not do.
    hide_count = form.get("output") in ("sheet", "character")
    problems = settings_2d.problems_for(ctx, form)
    busy = ctx.busy("submit")

    items = _rail_items_for_measurement()
    rail_full_w = widgets.stage_rail_width(items, state.create_stage)
    rail_floor_w = widgets.stage_rail_width(items, state.create_stage, max_width=0.0)
    rail_w, prompt_w, show_count, reset_compact = _row_widths(
        hide_count, rail_full_w, rail_floor_w
    )

    rail(ctx, max_width=rail_w, row_height=sp(PROMPT_H))
    imgui.same_line()
    _type(ctx, form)
    imgui.same_line()
    _prompt(ctx, form, prompt_w)
    imgui.same_line()
    if show_count:
        _count(ctx, form)
        imgui.same_line()
    _generate(ctx, form, spec, enabled=not problems and not busy, problems=problems,
              show_count=show_count)
    imgui.same_line()
    _reset(ctx, compact=reset_compact)


def _rail_items_for_measurement() -> list[tuple[str, str, str, str | None]]:
    """The rail's five ``(key, label, icon, reason)`` entries, for
    :func:`widgets.stage_rail_width` alone -- **not** what ``App._stage_rail``
    hands ``widgets.stage_rail`` to actually *draw* the rail.

    That real list carries each stage's live availability and its done-set,
    which need a job and (for Rig) a filesystem read this module has no
    business making just to size a row. ``stage_rail_width`` only reads the
    label text to measure it, and an unavailable stage's label is the same
    string as an available one's, so ``reason=None`` throughout costs the
    measurement nothing.
    """
    from . import create_stages

    return [
        (stage, create_stages.LABELS[stage], create_stages.ICONS[stage], None)
        for stage in create_stages.STAGES
    ]


def _row_widths(
    sheet: bool, rail_full_w: float, rail_floor_w: float
) -> tuple[float, float, bool, bool]:
    """-> (rail's max width, prompt width, whether to draw the count, whether
    Reset is icon-only). **The row's give-way order** -- four rungs now
    rather than two, because the rail and Reset both joined this line
    (2026-09-07) and two more things now compete for the width the type combo
    and Generate never gave up.

    Measured *before anything on the row has drawn*, which is the one thing
    that changed about how this is measured rather than only what it now
    covers. The two-rung version this replaces measured ``avail`` *after* the
    type combo's own ``same_line`` had already taken ``TYPE_W`` off it, and
    the shipped bug it warns about was subtracting ``TYPE_W`` a second time on
    top of that -- double-counting turned a resize floor into a Generate
    drawn past the pane edge, ``same_line`` past the content region's own
    "draws a control nowhere." The rail is the row's first element now, ahead
    of the combo, so nothing has been drawn yet by the time this has to decide
    how wide to make the rail -- there is no already-narrowed ``avail`` left
    to read. ``TYPE_W`` is therefore subtracted explicitly, once, here. That
    is the same "single-count everything" discipline the old bug violated,
    carried to the new, earlier point where it has to happen.

    1. The **prompt** shrinks to ``PROMPT_MIN_W``.
    2. The **count** is dropped -- ``_generate_tooltip`` restates its value
       once its pills are gone.
    3. The **rail** is handed whatever is left as its own ``max_width`` and
       walks its own three rungs (``widgets.stage_rail``: checks+labels,
       labels, icons). Every rung keeps every stage clickable and tooltipped,
       which is what makes it the right thing to give away next -- unlike the
       count or Reset, nothing about the rail actually disappears; it only
       gets terser. ``rail_full_w`` and ``rail_floor_w`` come from
       ``widgets.stage_rail_width`` (see :func:`_rail_items_for_measurement`),
       never guessed: 304 was a guess once, and wrong the moment a label
       changed.
    4. **Reset** drops to icon-only, its label moved to a tooltip, only if
       Reset at full width would leave the rail short of its own icon floor.

    The type combo and Generate never give way: they are what the bar is for.
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

    show_count = not sheet
    count_w = 0.0 if sheet else _count_width()
    prompt = avail - fixed - count_w - reset_full - rail_full_w - gaps_for(show_count)

    if prompt < sp(PROMPT_MIN_W) and show_count:
        # Rung 2: the count goes.
        show_count = False
        count_w = 0.0
        prompt = avail - fixed - count_w - reset_full - rail_full_w - gaps_for(show_count)

    if prompt >= sp(PROMPT_MIN_W):
        return rail_full_w, prompt, show_count, False

    # Rung 3: the prompt is pinned at its floor and the rail gives up the
    # width it would have kept -- handed only what's left, which is where
    # ``widgets.stage_rail``'s own laddering takes over.
    prompt = sp(PROMPT_MIN_W)
    rail_w = avail - fixed - count_w - reset_full - prompt - gaps_for(show_count)
    reset_compact = False
    if rail_w < rail_floor_w:
        # Rung 4: even Reset at full width would leave the rail short of its
        # own icon floor -- Reset gives up its label, freeing the difference.
        reset_compact = True
        rail_w += reset_full - reset_icon
    return max(rail_w, rail_floor_w), prompt, show_count, reset_compact


def _count_width() -> float:
    """The pills' real drawn width -- ``COUNT_W`` was a reservation nothing
    enforced.

    ``controls.segmented_choice`` chains four ``controls.button`` calls with a
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
    widths = [widgets.button_width(str(n)) for n in _COUNTS]
    return sum(widths) + gap * (len(widths) - 1)


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


def _count(ctx: Any, form: dict[str, Any]) -> None:
    """How many alternatives one press should draw.

    Not drawn at all for a sheet: both sheet doors refuse a batch and say why,
    so four radios of which three are refusals would be a control offering
    what the thing behind it will not do. ``sync_legacy_fields`` has already
    written ``count = 1`` for those.

    The pills are ``tokens.CONTROL_HEIGHT_COMPACT`` (26 dp) tall against a
    ``PROMPT_H`` (40 dp) row (B3, 2026-09-07): a bare ``same_line`` left them
    flush with the row's *top* rather than its middle, beside a two-line
    prompt field and a full-height Generate. The spacer dummies before and
    after reserve the full row height inside this group -- the same amount --
    so the row's shared baseline is exactly where it was without them, and
    only the pills themselves float centred inside it.
    """
    pill_h = sp(tokens.CONTROL_HEIGHT_COMPACT)
    pad = max(0.0, (sp(PROMPT_H) - pill_h) * 0.5)
    imgui.begin_group()
    if pad:
        imgui.dummy((1.0, pad))
    imgui.push_item_width(sp(COUNT_W))
    with focus.item(ctx.state, FOCUS_PANE, "count") as focused:
        changed, picked = controls.segmented_choice(
            "brief-count",
            tuple((str(n), str(n)) for n in _COUNTS),
            str(form["count"]),
            compact=True,
            tooltips=_COUNT_HINTS,
        )
        if changed:
            form["count"] = int(picked)
            ctx.state.clear_field_error("count")
        # Hand-answered, as it was in the column: a row of radios is one
        # control to the keyboard even though it is four items to imgui.
        if focused:
            here = _COUNTS.index(form["count"]) if form["count"] in _COUNTS else 0
            before_arrow = form["count"]
            if imgui.is_key_pressed(imgui.Key.left_arrow):
                form["count"] = _COUNTS[(here - 1) % len(_COUNTS)]
            if imgui.is_key_pressed(imgui.Key.right_arrow):
                form["count"] = _COUNTS[(here + 1) % len(_COUNTS)]
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
    spec: Any,
    *,
    enabled: bool,
    problems: list[Any],
    show_count: bool,
) -> None:
    """The press. Always visible, which is the point of the bar.

    The *reason* it is disabled stays in the settings column's plan block,
    which lists every problem and offers the one-click repairs. Here it is a
    tooltip: a bar has no room for a list, and a button that says nothing about
    why it is dead is the complaint this redesign started from.

    ``show_count`` is ``_row_widths``' own answer, not re-derived: the count
    pills carry their own value the moment they are on screen, so restating it
    here as well would be a second control saying the same number an inch to
    its left. It is only appended once ``_row_widths`` has dropped them --
    which the settings column's plan block, the count's other echo, cannot be
    relied on to catch either, since that column is itself a ``layout`` pane a
    person can collapse (the 2026-09-07 Create review, item 5.9).
    """
    from .panes import settings_2d

    with focus.item(ctx.state, FOCUS_PANE, "generate") as focused:
        pressed = widgets.primary_button(
            spec.create_label,
            (sp(GENERATE_W), sp(PROMPT_H)),
            enabled=enabled,
            # ``Problem`` is a str subclass -- the message *is* the object.
            reason=str(problems[0]) if problems else "",
            tooltip=_generate_tooltip(show_count, int(form["count"])),
        )
        anchors.mark("create/generate")
        if focused and enabled and _enter_pressed():
            pressed = True
    if pressed:
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


def _reset(ctx: Any, *, compact: bool) -> None:
    """*Reset...*, beside Generate now rather than pinned above the settings
    column it used to sit atop.

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

    ``compact`` is ``_row_widths``' rung 4: past that width Reset draws as a
    bare glyph with its label moved to the tooltip, the same rule
    ``widgets.icon_button`` states for any control with no visible label --
    applied here through ``controls.button`` directly so the GHOST role
    survives the swap, which ``icon_button``'s own paint does not offer.
    """
    from .panes import settings_2d

    label = icons.UNDO if compact else "Reset..."
    if controls.button(
        label,
        (0, sp(PROMPT_H)),
        role=controls.ButtonRole.GHOST,
        tooltip="Reset the image settings to their defaults." if compact else "",
    ):
        ctx.confirms.ask(
            dialogs.Confirm(
                title="Reset the image settings?",
                message=_RESET_CONFIRM_MESSAGE,
                confirm_label="Reset",
                cancel_label="Cancel",
                on_confirm=lambda: settings_2d._reset(ctx),
            )
        )


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
    from ..service.validation import MAX_PROMPT

    return MAX_PROMPT


def _enter_pressed() -> bool:
    return imgui.is_key_pressed(imgui.Key.enter) or imgui.is_key_pressed(
        imgui.Key.keypad_enter
    )


#: The count control's four values. 8 is ``validation.MAX_REFERENCE_COUNT``.
_COUNTS: tuple[int, ...] = (1, 2, 4, 8)

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

def _species_count() -> int:
    """How many species the character registry ships. Counted, never typed.

    ``characters.family`` imports nothing but the standard library, so reading
    it at import time here costs a dict copy and drags nothing in behind it.
    """
    from ..characters.family import families

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

__all__ = ["BAR_H", "FOCUS_PANE", "RAIL_ONLY_H", "bar_height", "draw", "shows"]
