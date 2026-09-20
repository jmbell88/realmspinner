"""Muse's command bar: the whole brief, across the top of the mode.

``create_brief``'s shape, and deliberately so -- the two are the same claim
about the same kind of screen: *what a press needs, on one row, never scrolled.*
The bar is **what to make** (the style tags, the lyrics, how long, how many, and
the button) and the column beside it (``modes/muse/ui/panes/recipe``) is **how**. A
control belongs to exactly one of them, which is the one-owner rule Create's
two panes already keep.

**Unconditional, unlike Create's.** ``create_brief.shows`` gates on the
Reference stage because the other four stages have no brief to press; Muse has
no stages, so the bar is simply always there. That is the only structural
difference between the two files, and it is why this one has no ``shows``.

Drawn through :func:`layout.pane` rather than bare, for ``create_brief``'s
reason: that is what puts it in ``layout.FRAME_PANES``, which is what gives it
the role fill, the divider, ``guard``'s error isolation and a pane slot for
``probe._pane_at`` -- without it ``/exercise-mode muse`` reports the bar's
controls against the empty-string pane, which reads downstream as controls
nobody owns.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .... import anchors, controls, focus, fonts, theme, widgets
from ....tokens import sp
from .. import mode as muse_mode

#: The pane's height in design pixels. Taller than Create's 62: this bar
#: carries a *second* multi-line field (the lyrics), because a lyric block is
#: the model's other real input and burying it in the recipe column would say
#: it was a setting.
#:
#: **270, not the naively-added 201 (2026-09-07).** 201 was 118 (the original
#: figure) + 62 (the lyrics field's growth, 62 -> 124) + one label row's ~21:
#: naming every control (Task 1 below) put a small-caps line above three of
#: the four top-row controls, and the fourth (Generate) is pushed down to
#: match rather than left sitting a row high, so the bar grows a label row's
#: worth on top of the doubled lyric field. That arithmetic undercounts for the
#: same reason 118 already undercounted *before* this pass -- against roughly
#: 142 dp of actual content, a gap this file never had a test to catch -- and
#: the reason both times is the same: neither figure ever charged for the
#: pane's own top and bottom padding, which ``layout.pane`` spends on top of
#: every widget drawn inside it and a bare-window frame test cannot see unless
#: it adds ``imgui.get_style().window_padding.y`` back in twice. 270 is that
#: sum, measured rather than guessed: confirmed by
#: ``test_the_bar_fits_the_height_it_declares``, which draws this bar for real
#: and asserts its content -- widget span plus that padding on both edges --
#: fits ``sp(BAR_H)``, not by eye.
BAR_H = 270.0

#: The two text fields' heights. The tags field is two lines for
#: ``create_brief``'s reason -- a single line for a paragraph shows the tail and
#: hides the subject -- and the lyrics field is four lines, doubled to eight
#: (2026-09-07): four was "a verse is four lines", but the common case is a
#: verse *and* its chorus, and eight shows both without a press of Expand.
#: Expand still exists, for anything longer.
TAGS_H = 40.0
LYRICS_H = 124.0

#: Fixed widths for the controls that flank the text fields, which take what is
#: left. ``TEXT_MIN_W`` is where they stop shrinking and the count is dropped
#: instead -- see :func:`_row_widths`.
TEXT_MIN_W = 180.0
#: 268, not 150 (2026-09-07): the preset group grew a fifth preset and a
#: Custom pill -- see :data:`_DURATIONS` -- so six pills need the room four
#: used to share.
DURATION_W = 268.0
COUNT_W = 124.0
GENERATE_W = 158.0

#: The custom-seconds field beside the pills, drawn only while
#: ``MuseState.duration_custom`` is set. Sized for four digits and a unit
#: letter, not for the row -- it sits beside the pills rather than in their
#: place.
CUSTOM_W = 76.0

#: This bar's key in the focus ring. Its own, as ``create_brief``'s is: the ring
#: is walked per pane and the bar and the recipe column are two panes.
FOCUS_PANE = "muse-brief"

#: The count control's values, and the top one is ``_jobs_music.MAX_COUNT``.
_COUNTS: tuple[int, ...] = (1, 2, 4)

#: The key a picked ``_CUSTOM`` pill writes, alongside the five presets below.
_CUSTOM = "custom"

#: The duration presets, in seconds. A segmented choice rather than a free
#: number because these are lengths game music is actually asked for, and a
#: drag that can land on 143 seconds offers precision nobody wants for the
#: common case -- which is why **Custom** (2026-09-07) exists beside them
#: rather than instead of them: the four-minute ceiling this list used to top
#: out at was ``_jobs_music.MAX_DURATION``, and once that door was raised to
#: ten minutes for a real use (a dungeon's whole ambient loop, not a fifteen-
#: second stinger) a fixed list could not reach it without either guessing at
#: a sixth preset nobody asked for or leaving the door's own range unreachable
#: from this control. Every preset is inside ``_jobs_music``'s bounds, which is
#: the only thing about them the door cares about.
_DURATIONS: tuple[int, ...] = (30, 60, 120, 240, 600)


def _duration_label(seconds: int) -> str:
    """A preset's pill text. -> ``"30s"`` below five minutes, else ``"10m"``.

    Module-level and imgui-free, like :func:`generate_label` and
    :func:`_lyrics_height` below it -- this file's discipline is that a claim
    drawn on a pill can be checked without a frame. The cutover at five
    minutes rather than at sixty seconds is so 240 still reads ``"240s"``, the
    figure every existing screenshot and manual mention already shows.
    """
    return f"{seconds}s" if seconds < 300 else f"{seconds // 60}m"


#: The segmented choice's options: the five presets, plus Custom. Built once
#: rather than inline in :func:`_duration` so :func:`_step` can walk the same
#: keys the pills are drawn from.
_DURATION_OPTIONS: tuple[tuple[str, str], ...] = tuple(
    (str(n), _duration_label(n)) for n in _DURATIONS
) + ((_CUSTOM, "Custom"),)

#: The instrumental choice's two options, key first. Not a checkbox: a
#: checkbox reads as "add lyrics", which is not the choice being made -- the
#: choice is which of two request shapes this is, and a segmented pair says so
#: the same way ``_duration`` and ``_count`` already do.
_LYRIC_MODES: tuple[tuple[str, str], ...] = (
    ("instrumental", "Instrumental"),
    ("lyrics", "With lyrics"),
)


def draw(ctx: Any) -> None:
    """The bar. Called from ``main._muse_workspace`` above the columns."""
    state = muse_mode.ensure(ctx)
    form = state.form

    focus.pump(ctx.state, FOCUS_PANE)
    focus.begin(ctx.state, FOCUS_PANE)

    busy = ctx.busy("submit")
    text_w, show_count, generate_w = _row_widths(state.duration_custom)

    _tags(ctx, form, text_w)
    imgui.same_line()
    _duration(ctx, form)
    if show_count:
        imgui.same_line()
        _count(ctx, form)
    imgui.same_line()
    imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() + _label_row_h())
    # The count travels on the button when its own control was dropped (W3).
    _generate(ctx, enabled=not busy, width=generate_w,
              takes=None if show_count else int(form["count"]))
    _lyrics(ctx, form, text_w)


def _label_row_h() -> float:
    """The vertical space one ``field_label`` line plus its gap takes.

    Every control but Generate is drawn as a group with ``field_label`` at its
    head, so its own label pushes its control down to line up with the fields
    beside it. Generate carries no label -- the button *is* its own name --
    so left where ``same_line`` puts it, it renders level with "TAGS" /
    "LENGTH" / "TAKES" instead of with the controls under them. This is the
    one row's worth of small-font text plus a style gap that closes that gap,
    read at draw time rather than hard-coded because the small font's line
    height is a font-metrics fact, not a design constant.
    """
    with fonts.small(imgui):
        line = imgui.get_text_line_height()
    return line + imgui.get_style().item_spacing.y


def _row_widths(duration_custom: bool) -> tuple[float, bool, float]:
    """-> (text width, whether to draw the count, Generate's width).

    **The row's give-way order.**

    Stated, the way ``create_brief._row_widths`` states it, and for the reason
    that file learned the hard way: ``same_line`` past the pane edge draws a
    control *nowhere*, so an unstated order does not produce a cramped row, it
    produces a missing Generate.

    The text fields shrink first, to ``TEXT_MIN_W``. Then the **count** is
    dropped -- it is the one control with a sane default whose value is also
    visible in the tray, as the number of cards a press produced. Duration and
    Generate never give way: the first is the parameter that decides what the
    press costs, and the second is what the bar is for.

    **The number does not leave with the control (W3, 2026-09-05.)** Dropping
    the count outright left the user pressing a button whose cost they could not
    see -- four takes is four times the wait and four rows in the tray. So the
    space the count gave up goes to Generate, which relabels itself *Generate 4
    takes*: the figure stays on screen at every width, in the one place it
    cannot be missed.

    ``duration_custom`` (2026-09-07) is not another give-way tier -- Custom's
    seconds field is fixed like Duration and Generate, never dropped, because
    a control the user just opened to type an exact number is the last one
    that should vanish out from under them. It only changes how much of
    ``fixed`` is spoken for before the text fields and the count divide what
    is left.
    """
    gap = imgui.get_style().item_spacing.x
    avail = imgui.get_content_region_avail().x
    fixed = sp(DURATION_W) + gap + sp(GENERATE_W) + gap
    if duration_custom:
        fixed += sp(CUSTOM_W) + gap
    count = sp(COUNT_W) + gap
    text = avail - fixed - count
    if text < sp(TEXT_MIN_W):
        return max(sp(TEXT_MIN_W), avail - fixed), False, sp(GENERATE_W) + count
    return max(sp(TEXT_MIN_W), text), True, sp(GENERATE_W)


def _tags(ctx: Any, form: dict[str, Any], width: float) -> None:
    """The style tags. The model's first input, under the model's own name.

    Comma-separated, because that is literally what ACE-Step's text encoder was
    trained on -- not a sentence. That advice lives in the empty-state tooltip
    below, which a label cannot carry -- it is one word, not a sentence about
    punctuation.

    **Named, where it used to argue against a name (2026-09-07).** This
    docstring used to say a label "would take a row this bar does not have."
    That was true of the bar Task 2 replaced: the bar has a label row now,
    deliberately, so the old argument no longer holds, and holding onto it
    would mean this is the one control on the row the manual cannot refer to
    by name. ``field_label`` costs the row that already exists; leaving the
    field unnamed no longer buys anything.
    """
    imgui.begin_group()
    widgets.field_label("Tags")
    before = form["prompt"]
    with focus.item(ctx.state, FOCUS_PANE, "prompt"):
        form["prompt"] = widgets.multiline(
            "##muse-tags", before, sp(TAGS_H), _max_prompt(), width=width
        )
        anchors.mark("muse/tags")
        widgets.char_count(form["prompt"], _max_prompt())
    if form["prompt"] != before:
        ctx.state.clear_field_error("prompt")
    if imgui.is_item_hovered() and not str(form["prompt"]).strip():
        imgui.set_tooltip(
            "Style tags, comma separated -- 'dark ambient, dungeon, low strings, "
            "slow'. Not a sentence."
        )
    _ring(ctx, "prompt")
    imgui.end_group()


def _lyrics(ctx: Any, form: dict[str, Any], width: float) -> None:
    """The lyric block. The model's second input, and optional.

    On its own row under the tags rather than beside them: two multi-line
    fields sharing a row would each be half a field. It is still *the bar* and
    not the recipe column, because it is part of what to make -- an instrumental
    and a song with a chorus are different requests, not the same request at a
    different setting.

    **Instrumental is a choice, not an empty field (2026-09-07).** Leaving the
    field blank always meant "no lyrics" -- that was never in question -- but
    nothing on screen said so, so an empty field read as unfinished rather
    than as decided. The segmented choice above it is the same claim made
    explicit, and picking *Instrumental* clears the field and greys it: this
    is still exactly today's implicit semantics, not a new one.
    """
    state = muse_mode.ensure(ctx)
    # ``align_text_to_frame_padding`` first: a bare ``Text`` sits a few pixels
    # above a frame-height control's baseline, which is fine on its own line
    # but reads as misaligned joined to one by ``same_line`` -- the same
    # correction ``plotter_layers._row_label`` applies before its own text.
    # This spends no extra row: the label rides the choice/Expand row rather
    # than opening one of its own, unlike Tags/Length/Takes above it.
    imgui.align_text_to_frame_padding()
    widgets.field_label("Lyrics")
    imgui.same_line()
    _lyric_mode(ctx, form)
    imgui.same_line()
    _expand(ctx, state)

    instrumental = bool(form.get("instrumental"))
    height = _lyrics_height(state, imgui.get_content_region_avail().y)
    before = form["lyrics"]
    if instrumental:
        imgui.begin_disabled()
    with focus.item(ctx.state, FOCUS_PANE, "lyrics"):
        form["lyrics"] = widgets.multiline(
            "##muse-lyrics", before, height, _max_lyrics(), width=width
        )
        anchors.mark("muse/lyrics")
    if instrumental:
        imgui.end_disabled()
    if form["lyrics"] != before:
        ctx.state.clear_field_error("lyrics")
    if imgui.is_item_hovered() and not instrumental and not str(form["lyrics"]).strip():
        imgui.set_tooltip(
            "Lyrics, with [verse] and [chorus] markers. Leave it empty for an "
            "instrumental."
        )
    _ring(ctx, "lyrics")


def _lyric_mode(ctx: Any, form: dict[str, Any]) -> None:
    """Instrumental, or with lyrics. See :func:`_lyrics`."""
    current = "instrumental" if form.get("instrumental", True) else "lyrics"
    with focus.item(ctx.state, FOCUS_PANE, "instrumental"):
        changed, picked = controls.segmented_choice(
            "muse-lyric-mode", _LYRIC_MODES, current, compact=True
        )
        if changed:
            _set_instrumental(form, picked == "instrumental")
            ctx.state.clear_field_error("lyrics")


def _set_instrumental(form: dict[str, Any], instrumental: bool) -> None:
    """Flip the choice. -> nothing; mutates ``form`` in place.

    Choosing *Instrumental* clears the field along with greying it: a field
    that still shows a verse while greyed out would say two different things
    about what is about to be submitted.
    """
    form["instrumental"] = instrumental
    if instrumental:
        form["lyrics"] = ""


def _expand(ctx: Any, state: Any) -> None:
    """Grow the lyric field to fill the bar's remaining height, or don't."""
    label = "Collapse" if state.lyrics_expanded else "Expand"
    with focus.item(ctx.state, FOCUS_PANE, "expand"):
        if controls.small_button(f"{label}##muse-lyrics-expand"):
            state.lyrics_expanded = not state.lyrics_expanded


def _lyrics_height(state: Any, avail_y: float) -> float:
    """The lyric field's height: fixed, or whatever is left in the bar.

    A pure function of ``state.lyrics_expanded`` and the space actually left,
    so the claim -- expanding grows the field rather than merely relabelling
    the toggle -- can be checked without an imgui frame.
    """
    return avail_y if state.lyrics_expanded else sp(LYRICS_H)


def _current_duration_key(state: Any, form: dict[str, Any]) -> str:
    """Which pill is lit. -> ``_CUSTOM``, or the preset matching the number.

    Not membership in ``_DURATIONS``: that was the old rule, and it is the bug
    this control fixes. ``str(current if current in _DURATIONS else
    _DURATIONS[1])`` always drew "60s" as selected the instant ``duration``
    held anything else -- there was no way to type a number and see the pills
    agree it had been typed. ``state.duration_custom`` is the one bit that
    says which of the two requests, "pick a preset" or "trust the field", is
    actually live, so it -- not the number -- decides which pill lights.

    The number still gets a say in one direction only: a ``duration`` that is
    not a preset reads as Custom whatever the flag holds. Nothing in this file
    can produce that pair -- ``_pick_duration`` sets the flag with the number,
    and ``reset_form`` restores a preset -- but "no pill lit at all" is the
    one state worse than the lie this replaced, because a bar showing no
    selection does not say what a press is about to submit either. So one
    function decides, and :func:`_duration` draws the seconds field on its
    answer rather than on the flag, which is what stops the pill and the
    field disagreeing about which of them owns the number.
    """
    if state.duration_custom or int(form["duration"]) not in _DURATIONS:
        return _CUSTOM
    return str(int(form["duration"]))


def _pick_duration(state: Any, form: dict[str, Any], key: str) -> None:
    """Apply a duration pick, preset or Custom. -> nothing; mutates in place.

    A preset writes the number and drops the flag. Custom sets the flag and
    leaves the number exactly where it was, so the field it opens shows what
    the user already had rather than snapping to some other starting value --
    the same courtesy ``_set_instrumental`` does not owe, because that choice
    has no number underneath it to preserve.
    """
    if key == _CUSTOM:
        state.duration_custom = True
    else:
        state.duration_custom = False
        form["duration"] = float(key)


def _duration(ctx: Any, form: dict[str, Any]) -> None:
    """How long. The one parameter that decides what the press costs."""
    state = muse_mode.ensure(ctx)
    imgui.begin_group()
    widgets.field_label("Length")
    imgui.push_item_width(sp(DURATION_W))
    with focus.item(ctx.state, FOCUS_PANE, "duration") as focused:
        current = _current_duration_key(state, form)
        changed, picked = controls.segmented_choice(
            "muse-duration", _DURATION_OPTIONS, current, compact=True
        )
        if changed:
            _pick_duration(state, form, picked)
            ctx.state.clear_field_error("duration")
        # Hand-answered, as Create's count is: a row of radios is one control
        # to the keyboard even though it is six items to imgui, Custom
        # included now that _step walks keys rather than _DURATIONS itself.
        if focused:
            stepped = _step(current, tuple(key for key, _ in _DURATION_OPTIONS))
            if stepped is not None:
                _pick_duration(state, form, stepped)
                ctx.state.clear_field_error("duration")
        # Rung on the pills, before the optional custom field is drawn beside
        # them: ``_ring`` reads the *last* item's rect, and a refusal on
        # "duration" belongs on the control the number is chosen through even
        # on a press where Custom supplied it -- not on the seconds field's
        # unit letter, which is what the last item would be otherwise.
        _ring(ctx, "duration")
        if current == _CUSTOM:
            imgui.same_line()
            _duration_custom(ctx, form)
    imgui.pop_item_width()
    imgui.end_group()


def _duration_custom(ctx: Any, form: dict[str, Any]) -> None:
    """The typed-seconds field, drawn only while the Custom pill is lit.

    ``##``-hidden and not ``"Length"`` again: :func:`_duration` has already
    named this control once, and a second visible label beside the pills
    would be exactly the "second field face" ``test_ux_consistency_pass3``'s
    ``_LABELLED_RAW`` regex exists to catch on every other ``controls.*``
    call site in this app -- see that test before touching this call.
    """
    imgui.set_next_item_width(sp(CUSTOM_W))
    changed, value = controls.input_int("##muse-duration-custom", int(form["duration"]), 0)
    if changed:
        form["duration"] = float(_clamp_duration(value))
        ctx.state.clear_field_error("duration")
    imgui.same_line()
    widgets.muted("s")


def _clamp_duration(value: int) -> int:
    """Hold a typed seconds value inside the door's range. -> the clamped int.

    **Not** the client-side ``validate(form)`` ``muse_mode.py`` (78-84)
    argues against -- that argument is about duplicating a *refusal*, and this
    duplicates nothing: it only stops a spinner from displaying a number
    ``create_music_job`` was always going to refuse, which is a worse moment
    to learn the bound than while still typing it. The bound itself is
    imported lazily, exactly as ``_max_prompt``/``_max_lyrics``/``_max_seed``
    below do, so this file never holds a second copy of it.
    """
    return int(min(max(value, _min_duration()), _max_duration()))


def _count(ctx: Any, form: dict[str, Any]) -> None:
    """How many takes one press should draw."""
    imgui.begin_group()
    widgets.field_label("Takes")
    imgui.push_item_width(sp(COUNT_W))
    with focus.item(ctx.state, FOCUS_PANE, "count") as focused:
        current = str(form["count"])
        options = tuple(str(n) for n in _COUNTS)
        changed, picked = controls.segmented_choice(
            "muse-count",
            tuple((key, key) for key in options),
            current,
            compact=True,
        )
        if changed:
            form["count"] = int(picked)
            ctx.state.clear_field_error("count")
        if focused:
            stepped = _step(current, options)
            if stepped is not None:
                form["count"] = int(stepped)
                ctx.state.clear_field_error("count")
    imgui.pop_item_width()
    _ring(ctx, "count")
    imgui.end_group()


def _step(current: str, keys: tuple[str, ...]) -> str | None:
    """Left/Right through a segmented choice's keys. -> the newly picked key.

    Retargeted at the option *keys* rather than a numeric values tuple
    (2026-09-07): duration grew a Custom pill that is not a number, and the
    old ``values.index(int(...))`` shape had no way to land on it, which made
    Custom the one pill Left/Right could never reach. ``_count`` walks the
    same function over its own keys, so "a row of radios is one control to
    the keyboard" keeps meaning the same thing for both.
    """
    here = keys.index(current) if current in keys else 0
    if imgui.is_key_pressed(imgui.Key.left_arrow):
        return keys[(here - 1) % len(keys)]
    if imgui.is_key_pressed(imgui.Key.right_arrow):
        return keys[(here + 1) % len(keys)]
    return None


def generate_label(takes: int | None) -> str:
    """The button's text. -> ``"Generate"``, or the count-carrying form (W3).

    A function rather than three lines inside the draw so the claim can be
    asserted without an imgui frame: the number a press costs stays on screen
    at every pane width, and one take says "Generate" because "Generate 1 take"
    is a control apologising for itself.
    """
    return "Generate" if takes is None or int(takes) <= 1 else f"Generate {int(takes)} takes"


def _generate(
    ctx: Any, *, enabled: bool, width: float | None = None, takes: int | None = None
) -> None:
    """The press. Always visible, which is the point of the bar.

    ``takes`` is set only when the count control was dropped for width, and the
    label carries the number instead (W3): a Generate whose cost is off screen
    is a button pressed without knowing what it will do.

    Disabled while a submit is in flight, and **also when the music weights are
    not on this host**. An empty prompt is still left to the service, and the
    reasoning above still holds for it: a refusal naming the control is more
    use than a dead button.

    Missing weights are the case where that stopped being true. There is no
    fallback -- Muse refuses outright rather than generating something worse --
    so the answer never changes until an 8 GB download happens, and finding
    that out by pressing the button was the whole complaint. The Recipe pane
    now carries the notice and the Install button; this is the same fact on the
    control, so the two agree and the reason is on the hover.
    """
    from .....service import jobs as svc_jobs
    from ....panes import model_gate

    blocked = bool(model_gate.missing(ctx, svc_jobs.MUSIC_ROWS))
    with focus.item(ctx.state, FOCUS_PANE, "generate") as focused:
        pressed = widgets.primary_button(
            generate_label(takes),
            (sp(GENERATE_W) if width is None else float(width), sp(TAGS_H)),
            enabled=enabled and not blocked,
            reason=_generate_reason(blocked),
            tooltip="Ctrl+Enter",
        )
        anchors.mark("muse/generate")
        if focused and enabled and not blocked and _enter_pressed():
            pressed = True
    if pressed:
        muse_mode.generate(ctx)


def _generate_reason(blocked: bool) -> str:
    """Why Generate is greyed for missing weights. -> "" once they are on disk.

    **muse-07** (2026-09-05 audit): this sentence used to be a string literal
    chosen inline in the ternary passed to ``widgets.primary_button``, so
    nothing could assert it stayed right without an imgui frame -- the
    2026-09-02 review's T4 defect, in the one mode whose reasons had not been
    pulled out that way (``plotter_menu._layer_reason``,
    ``inker_mode._no_document_reason`` and ``overlay.cancel_reason`` all were).
    """
    if blocked:
        return "The music model is not downloaded. See the Recipe panel."
    return ""


def _ring(ctx: Any, field: str) -> bool:
    """Mark the control just drawn if a refusal named it. -> whether it did.

    The ring alone, no message: ``widgets.field_error`` draws its text *below*
    the control, which in a bar would push the row apart. The words arrive as
    the toast ``ctx.toast`` already raised.
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


def _max_lyrics() -> int:
    from .....service._jobs_music import MAX_LYRICS

    return MAX_LYRICS


def _min_duration() -> float:
    from .....service._jobs_music import MIN_DURATION

    return MIN_DURATION


def _max_duration() -> float:
    from .....service._jobs_music import MAX_DURATION

    return MAX_DURATION


def _enter_pressed() -> bool:
    return imgui.is_key_pressed(imgui.Key.enter) or imgui.is_key_pressed(
        imgui.Key.keypad_enter
    )


__all__ = ["BAR_H", "FOCUS_PANE", "draw"]
