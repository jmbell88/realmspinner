"""Sirens' third right-column pane: the song's sound effects.

**A list over the document, and nothing more.** A one-shot has been a
first-class part of a ``.wsng`` since Phase 1 -- ``SongDoc.oneshots`` holds
``OneShot(uid, name, pattern, tempo, speed)``, ``add_oneshot`` mints an effect
*and a pattern of its own* as one collapsed undo step, and ``synth.render_oneshot``
renders it at its own tempo -- so what was missing was never a model, it was a
way in. This is that: add, remove, rename, the two timing fields, and Audition.

**The grid is the effect editor.** Selecting an effect points the caret at that
effect's pattern (``sirens_mode.set_caret`` takes a pattern uid), so writing a
coin pickup uses the same five columns, the same piano row and the same undo
stack as writing a bassline. A second, smaller grid in this sidebar would be a
second set of keyboard bindings for the same job, and the one it would be
smaller than is the one people already know.

**Why an effect keeps its own tempo and speed.** ``document.OneShot`` states it:
a coin pickup is forty milliseconds whatever the music is doing, and tying it to
the song's tempo would mean every effect in the document changed length the
moment somebody slowed the track down. So the two fields are here, per row, and
the transport's pair does not reach them.

**Auditioning does not touch the song's buffer.** It goes through
``sirens_mode.audition``, which renders under its own task key and hands the
samples straight to the mixer -- see ``AUDITION_PREFIX`` for why sharing
``sirens-render:`` would leave a coin pickup where the song used to be.
"""

from __future__ import annotations

from typing import Any

from ..... import anchors, controls, icons, tokens, widgets
from .....manual import render as manual_render
from .....tokens import sp
from ... import audio as sirens_audio
from ... import mode as sirens_mode
from ...engine import document as D
from ...engine import instruments as inst

#: How many rows a new effect's pattern gets. ``document.add_oneshot``'s own
#: default, named here because this pane draws the sentence that explains it: at
#: the default speed eight rows is about a third of a second, which is longer
#: than nearly every effect and short enough that the whole thing is on screen
#: without scrolling.
NEW_ROWS = 8

#: What this pane refuses to shrink past, in design pixels: the Add/Delete row,
#: one effect row, and the selected effect's Name/Tempo/Speed block.
#:
#: It had none while this was Sirens' third of three ordinary SHARE columns --
#: ``layout.column`` handed every declared share key ``Layout.share``'s
#: always-answer 0.55, so ``layout_skeleton.heights`` saw three slots each
#: wanting 0.55 of the room and gave every one of them exactly that, with
#: nothing left for this pane or the Song file FILL pane under it. Fixed at the
#: root by ``Layout.saved_share`` (only a *saved* proportion reaches
#: ``heights`` now, so an untouched key gets its even-division default instead
#: of a borrowed 0.55) -- but a floor still has to say what "even division"
#: must never shrink below, the same way ``sirens_envelopes.ENVELOPES_FLOOR``
#: and ``BRIDGE_FLOOR`` do for their own panes. This is the pane the shipped
#: 0.0.39 ``dev/screenshots/dark-sirens.png`` caught missing outright -- that image
#: has since been refreshed, so the evidence is the release rather than the
#: file as it stands now.
EFFECTS_FLOOR = 210.0

_BUSY_WHY = "This song is being written; the buttons come back when it lands."


def delete_reason(editable: bool, selected: bool) -> str:
    """Why the Delete button is disabled, or "" while it is live.

    Pulled out as a pure function so the 2026-09-07 audit's finding sirens-04
    has something a test can call with no imgui frame: the inline ternary this
    replaced tested ``editable`` inverted, so a busy song showed "No sound
    effect is selected" and an idle one with nothing picked showed the busy
    sentence -- each state naming the other's reason.
    """
    if not editable:
        return _BUSY_WHY
    if not selected:
        return "No sound effect is selected."
    return ""


def draw(ctx: Any) -> None:
    from imgui_bundle import imgui

    anchors.mark_window("sirens/effects")
    state = sirens_mode.ensure(ctx)
    tab = state.active
    widgets.section("Sound effects")
    manual_render.help_button(ctx, "sirens-effects")

    if tab is None:
        return

    doc = tab.doc
    editable = not tab.busy

    width = widgets.grid_width(2)
    if widgets.disabled_button(f"{icons.PLUS} Add", editable, (width, 0), reason=_BUSY_WHY):
        _add(ctx, state, tab)
    imgui.same_line()
    if widgets.disabled_button(
        f"{icons.TRASH} Delete",
        editable and state.oneshot is not None,
        (width, 0),
        reason=delete_reason(editable, state.oneshot is not None),
    ):
        _remove(ctx, state, tab)

    if not doc.oneshots:
        widgets.muted_wrapped(
            "No sound effects yet. Each one is a little pattern of its own,"
            " played at its own tempo and exported to sfx/ beside the song."
        )
        return

    imgui.dummy((0, sp(tokens.SP_1)))
    _rows(ctx, state, tab, editable)

    selected = None if state.oneshot is None else doc.oneshot(state.oneshot)
    if selected is None:
        return
    imgui.dummy((0, sp(tokens.SP_2)))
    _fields(state, tab, selected, editable)


def _add(ctx: Any, state: Any, tab: Any) -> None:
    """One new effect, selected, with the caret already in its pattern.

    Selecting it here rather than leaving the user to click the row they just
    made: ``add_oneshot`` creates a pattern nobody is looking at, and an Add
    button whose only visible result is one more row in a list is a button that
    appears not to have worked.
    """
    try:
        one = tab.doc.add_oneshot(rows=NEW_ROWS)
    except ValueError as exc:
        # sirens-03 (2026-09-18 audit, second run): this used to say
        # ``MAX_ONESHOTS`` is "the only way this refuses" -- but
        # ``add_oneshot`` also makes a pattern of its own for the effect
        # (``document.py``'s ``add_oneshot``), and that inner ``add_pattern``
        # call can itself raise at ``MAX_PATTERNS`` first. Framed rather than
        # swallowed either way: both ceilings are reachable by working, and a
        # button that silently stops adding is worse than one that says why.
        ctx.toast(f"That sound effect was not added: {exc}", "error")
        return
    _select(ctx, state, one)


def _remove(ctx: Any, state: Any, tab: Any) -> None:
    """Drop the selected effect. Its pattern is **left in the document.**

    ``remove_oneshot``'s own behaviour, and the one worth saying out loud here
    because this pane is where somebody would notice: the pattern the effect
    named survives in the pattern list, exactly as an instrument survives a
    sample being removed. It is undoable either way, and a removal that also
    deleted a pattern the user had put in the song's order would be a removal
    that changed the music.
    """
    uid = state.oneshot
    try:
        removed = tab.doc.remove_oneshot(uid)
    except ValueError as exc:
        ctx.toast(f"That sound effect was not removed: {exc}", "error")
        return
    if removed:
        state.oneshot = None
        sirens_mode.clamp_caret(ctx, tab)


def _select(ctx: Any, state: Any, one: Any) -> None:
    """Point the pane's selection and the grid's caret at the same effect."""
    state.oneshot = one.uid
    sirens_mode.set_caret(ctx, pattern=one.pattern)


def _label(doc: Any, one: Any, index: int) -> str:
    rows = doc.pattern(one.pattern)
    length = f"{rows.rows} rows" if rows is not None else "no pattern"
    return f"{one.name or f'Effect {index + 1}'}  ({length})"


def audition_reason(editable: bool) -> str:
    """The Audition button's disabled reason, or "" while it is live.

    Pulled out pure for the 2026-09-13 audit's finding sirens-05: this was an
    inline ternary, the shape that produced findings sirens-03/04/05 of the
    2026-09-07 audit (see :func:`.sirens_orders.add_to_order_reason`) and, in
    this same file, :func:`delete_reason` -- an untested priority among
    competing disabled causes with nothing to catch a wrong order before it
    shipped. ``editable`` wins over the device: a busy song should read "the
    song is being written," not the device's own sentence, once both are true.
    """
    if not editable:
        return _BUSY_WHY
    return sirens_audio.unavailable_reason()


def _rows(ctx: Any, state: Any, tab: Any, editable: bool) -> None:
    """One row per effect: the name, and the button that plays it.

    Audition is per row rather than one button under the selection, because the
    thing a user does with a folder of sound effects is listen down the list --
    and a design where that costs two clicks per effect is one where nobody
    checks the last five.
    """
    from imgui_bundle import imgui

    doc = tab.doc
    device = sirens_audio.available()
    # One sentence per way the button can be dead, and the device's is
    # ``sirens_audio``'s own so this pane and the transport cannot say two
    # different things about the same missing card.
    why = audition_reason(editable)
    for index, one in enumerate(list(doc.oneshots)):
        if controls.selectable(
            f"{_label(doc, one, index)}###sirens-oneshot-{one.uid}",
            state.oneshot == one.uid,
        )[0]:
            _select(ctx, state, one)
        imgui.same_line()
        if widgets.disabled_button(
            f"{icons.PLAY}###sirens-oneshot-play-{one.uid}",
            editable and device,
            (0, 0),
            reason=why,
            tooltip="Render this effect and play it once.",
        ):
            sirens_mode.audition(ctx, tab, one.uid)


def _fields(state: Any, tab: Any, selected: Any, editable: bool) -> None:
    """The selected effect's name, tempo and speed, and where it is edited.

    Every change goes through ``update_oneshot``, which is what makes it one
    reversible step and what refuses a no-op -- a slider held still is a stream
    of frames, and every one of them would otherwise be an undo step.

    None of the three re-arms the renderer, and that is not an omission: an
    effect is not in the song's order list, so nothing about it can change what
    ``song.wav`` sounds like. What it changes is the next audition and the next
    export, both of which read the document when they run.
    """
    from imgui_bundle import imgui

    doc = tab.doc
    # ``##``-hidden with the name drawn above, the same rule the instrument
    # list's own Name field follows (``sirens_instruments.py``): imgui draws a
    # field's label to its *right*, and this field is set to width -1, so a
    # visible "Name" here would land past the content region and simply not be
    # drawn.
    widgets.field_label("Name")
    imgui.set_next_item_width(-1)
    # ``controls.input_text``, not ``widgets.input_text``: the latter has no
    # ``enabled``, so this field stayed live and kept pushing undo steps while
    # ``tab.busy`` -- a rename typed mid-save landed on the document a save
    # was in the middle of reading (the 2026-09-15 audit, finding sirens-04).
    changed, name = controls.input_text(
        "##sirens-fx-name", selected.name, enabled=editable, commit=True
    )
    if changed:
        doc.update_oneshot(selected.uid, name=str(name)[: inst.MAX_NAME_LEN])

    # Through ``widgets.labeled_slider_int`` rather than a bare
    # ``controls.slider_int`` (the 2026-09-07 audit): the same -1-width rule
    # the Name field above states left Tempo and Speed's own names undrawn --
    # two bare numbers with nothing on screen saying which was which.
    changed, value = widgets.labeled_slider_int(
        "Tempo", selected.tempo, D.MIN_TEMPO, D.MAX_TEMPO, enabled=editable
    )
    controls.fold_undo(doc.history)
    if changed:
        doc.update_oneshot(selected.uid, tempo=int(value))
    changed, value = widgets.labeled_slider_int(
        "Speed", selected.speed, D.MIN_SPEED, D.MAX_SPEED, enabled=editable
    )
    controls.fold_undo(doc.history)
    if changed:
        doc.update_oneshot(selected.uid, speed=int(value))

    pattern = doc.pattern(selected.pattern)
    if pattern is None:
        # Unreachable through the app -- ``add_oneshot`` mints the pattern and
        # the pair is one undo step -- but a hand-edited ``.wsng`` can carry it,
        # and a row that renders as an exception is worse than one that says
        # what is wrong.
        widgets.muted_wrapped("This effect names a pattern this song no longer holds.")
        return
    # Only while the caret is actually in it: clicking a song pattern in the
    # Order panel leaves this row selected -- the selection is what the two
    # sliders edit -- and a panel that then insisted the grid was showing the
    # effect would be telling the user something they can see is untrue.
    if state.pattern == selected.pattern:
        widgets.muted_wrapped(
            "The grid is editing this effect. Click a pattern in the Order panel"
            " to go back to the song."
        )
    else:
        widgets.muted_wrapped("Select this effect again to edit it in the grid.")
