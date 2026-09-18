"""The guided tour, drawn over whatever is on screen.

Point-and-wait: the tour rings a real control, says one thing about it, and
waits for the reader to use it. **It never clicks anything for them**, which is
what keeps a tour from mutating a document, a setting or the job queue -- and it
is also why nothing here reaches ``App._modal_open``. The app underneath has to
stay live.

Two consequences shape the whole file.

The scrim cannot be a mask. ``ImDrawData`` arrives as a flat list of command
lists with nothing on them saying which window each came from -- the same fact
``vibrancy`` is built around -- so there is no "everything except the
highlight" to dim. The dimming fills the viewport *around* holes instead, which
is exact where a mask would be a guess, and leaves the control genuinely
visible rather than approximately so.

There are two holes, and the second is the one nobody predicts: the scrim is on
the foreground draw list, which imgui paints above every window -- so without a
hole for it, the tour dims its own card.

And the scrim takes no input, for a better reason than a flag. It is draw-list
geometry rather than a window, so there is nothing there to hit: a click aimed
at the ringed control reaches the control, and the point-and-wait promise holds
without anything having to be careful about it. The card *is* a window and does
take input, because its buttons are the only part the reader is meant to press
here.

Positions are read from ``anchors``, never computed. The rail alone recomputes
every item's box each frame across a three-rung compression ladder, so a ring
that did its own arithmetic would drift the moment the window got short.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import anchors, controls, icons, theme, tokens, widgets
from ..manual import render as manual_render
from ..settings import as_list
from ..tokens import sp
from ..tour import find as find_tour

#: The card's width, in design px. Wide enough for two sentences at a readable
#: measure and narrow enough to sit beside a ringed control rather than over it.
CARD_W = 380

#: How far the scrim goes. Not opaque: the reader is meant to keep their
#: bearings, and a tour that blacked out the app would be a modal wearing a
#: different name.
VEIL_ALPHA = 0.55

#: Breathing room between the ringed control and the hole's edge.
HOLE_PAD = 6

_was_open = [False]

#: Last frame's card rect, so this frame's scrim can leave a hole for it.
#:
#: One frame stale, and the same staleness ``widgets.window_shadow`` documents
#: and accepts: the card is auto-height, so its rectangle does not exist until
#: it has drawn. It only moves when a step changes side, and on that one frame
#: the scrim's hole is where the card was rather than where it is -- which
#: costs a frame of dimmed text and never a frame of hidden text, because the
#: card is drawn after the scrim either way.
_card_rect: list[tuple[float, float, float, float] | None] = [None]

#: Whether ``##tour-card`` had keyboard focus as of the last frame it drew.
#: One frame stale, for ``_viewport_hovered``'s reason (``main.py``): the
#: card's own Enter/Left/Right read gates on ``imgui.is_window_focused()``
#: fresh, in the same frame, because it runs *after* the window is drawn --
#: but ``App._shortcut`` runs on the raw pygame event, *before* this frame's
#: imgui pass, so last frame's answer is the only one available to it, and a
#: one-frame lag on "did the reader just click away from the card" is not a
#: window anyone can feel. See :func:`has_focus`.
_card_focused: list[bool] = [False]


def has_focus() -> bool:
    """Whether the tour card currently owns the keyboard. -> ``main.py``.

    The 2026-09-07 audit, finding tour-01: the card reads Enter/Left/Right
    unconditionally, and so did ``App._shortcut`` underneath it -- an arrow
    that stepped the tour also moved the Library grid's cursor, and an Enter
    that confirmed a rename also advanced the tour. The card's own read is
    gated on this same question (see ``_card``); this is the other half, so
    the mode below can refuse the keys the card already claimed.
    """
    return _card_focused[0]


# -- what a step is waiting for -------------------------------------------
#
# One snapshot per frame, built here and read by name, so ``studio/tour`` stays
# free of imgui and ``service`` and its rules stay assertable headlessly.


def _inker_doc(ctx: Any) -> Any:
    inker = getattr(ctx.state, "inker", None)
    return getattr(inker, "active", None) if inker is not None else None


def satisfied(ctx: Any, name: str, arg: str | None) -> bool:
    """Whether the condition ``name`` holds right now.

    Every name in ``tour.steps.CONDITIONS`` must be answered here, and nothing
    else may be; ``tests/studio/tour/test_tour_conditions.py`` asserts both directions.
    An unknown name reads as "never satisfied", which on a point-and-wait step
    is indistinguishable from the app being broken -- so it is a test failure
    rather than something to discover at runtime.
    """
    if name == "manual":
        return False
    if name == "mode_is":
        return ctx.state.mode == arg
    if name == "doc_open":
        if arg == "inker":
            return _inker_doc(ctx) is not None
        holder = getattr(ctx.state, str(arg), None)
        return bool(getattr(holder, "docs", None))
    if name == "tool_is":
        inker = getattr(ctx.state, "inker", None)
        return getattr(inker, "tool", None) == arg
    if name == "layers_at_least":
        doc = _inker_doc(ctx)
        if doc is None:
            return False
        try:
            return len(doc.doc.stack) >= int(arg or 0)
        except (AttributeError, TypeError, ValueError):
            return False
    if name == "animated":
        doc = _inker_doc(ctx)
        return doc is not None and getattr(doc.doc, "anim", None) is not None
    if name == "sfx_at_least":
        sirens = getattr(ctx.state, "sirens", None)
        tab = getattr(sirens, "active", None) if sirens is not None else None
        if tab is None:
            return False
        try:
            return len(tab.doc.oneshots) >= int(arg or 0)
        except (AttributeError, TypeError, ValueError):
            return False
    if name == "notes_at_least":
        threshold = _count(arg)
        return threshold is not None and _notes(ctx) >= threshold
    return False


def _count(arg: str | None) -> int | None:
    """A condition's numeric argument, or ``None`` when it is not a number.

    ``None`` rather than zero, and the difference is the whole point. This
    returned zero and claimed to match ``sfx_at_least``'s rule -- but that
    condition (and ``layers_at_least``) catch the same ``ValueError`` and
    answer *not satisfied*, so the three never agreed. Zero is the worse half
    of the disagreement: every one of these is a ``>=``, so a mistyped
    threshold did not fail the step, it satisfied it immediately, and a tour
    step that completes before the reader does anything is indistinguishable
    from one they finished.

    A missing arg is still zero -- an absent threshold is a step with nothing
    to wait for, which is different from a threshold nobody can read.
    ``tests/studio/tour`` gates the authored ones statically, so this path is the
    runtime backstop rather than the guard.
    """
    if arg is None or arg == "":
        return 0
    try:
        return int(arg)
    except (TypeError, ValueError):
        return None


def _notes(ctx: Any) -> int:
    """How many real pitches the active song holds, over every pattern.

    ``notes.EMPTY`` is -1 and the two sentinels are *above* the pitch range, so
    "is a note" is ``0 <= value <= MAX_NOTE`` rather than ``!= EMPTY``: a
    note-off is something the reader typed, but it is not a note they wrote.

    Swallows the three shapes of missing attribute rather than raising --
    ``sfx_at_least``'s rule, and its reason: a traceback in the frame loop is
    worse than a tour that will not advance.
    """
    sirens = getattr(ctx.state, "sirens", None)
    tab = getattr(sirens, "active", None) if sirens is not None else None
    if tab is None:
        return 0
    try:
        import numpy as np

        from ..modes.sirens.engine import document as D
        from ..modes.sirens.engine import notes as N

        total = 0
        for pattern in tab.doc.patterns:
            column = np.asarray(pattern.cells)[:, :, D.NOTE]
            total += int(np.count_nonzero((column >= 0) & (column <= N.MAX_NOTE)))
        return total
    except (AttributeError, TypeError, ValueError, IndexError):
        return 0


#: The names :func:`satisfied` answers. Written out rather than derived from the
#: function, so the agreement test compares two independently authored lists
#: instead of one list against itself.
HANDLED: frozenset[str] = frozenset(
    {
        "manual",
        "mode_is",
        "doc_open",
        "tool_is",
        "layers_at_least",
        "animated",
        "sfx_at_least",
        "notes_at_least",
    }
)


# -- running a tour --------------------------------------------------------


def start(ctx: Any, key: str) -> None:
    """Begin a tour. Unknown keys are ignored rather than raising.

    The other half of ``landing._tour_offer``'s guard: that stops the card
    from being *offered* for a tour whose mode is gated, and this stops it
    being *started* by any other route -- the palette's asset actions, a
    manual chapter's link, a direct call from a test. Same question
    (``Tour.mode`` and ``model_gate.mode_gate``), asked again here rather than
    trusted from the caller, because the first step would otherwise wait on a
    ``mode_is`` condition ``state.set_mode`` refuses to ever satisfy -- the
    tour hangs on step 1 with no way forward. A toast names the reason rather
    than the card opening and then never advancing.
    """

    tour = find_tour(key)
    if tour is None:
        return
    if tour.mode:
        from . import model_gate

        where, _keys = model_gate.mode_gate(ctx, tour.mode)
        if where:
            reason = model_gate.mode_reason(ctx, tour.mode)
            fallback = f"{tour.title} needs a mode this machine hasn't unlocked yet."
            ctx.toast(reason or fallback, "warn")
            return
    ctx.state.tour.start(key)


def _clear_card() -> None:
    """The hole/focus bookkeeping a running tour leaves behind -- shared by
    every way a tour can stop, so a fresh tour's first frame never veils
    around a hole left by the last one."""
    _card_rect[0] = None
    _card_focused[0] = False


def stop(ctx: Any) -> None:
    ctx.state.tour.stop()
    _clear_card()


def advance(ctx: Any, delta: int = 1) -> None:
    """Step forward or back, completing the tour when it runs off the end."""

    state = ctx.state.tour
    tour = find_tour(state.key)
    if tour is None:
        state.stop()
        return
    index = state.index + delta
    if index >= len(tour):
        # The 2026-09-15 audit (tour-01): finishing by running off the end
        # called `state.complete()` directly rather than going through
        # `stop`'s cleanup, so `_card_rect`/`_card_focused` survived a
        # completed tour -- the next tour started, its first frame veiled
        # around a hole left by the one that just finished.
        state.complete()
        _clear_card()
        _remember(ctx)
        return
    state.index = max(0, index)
    state.satisfied = False


def is_open(ctx: Any) -> bool:
    return bool(getattr(ctx.state, "tour", None) and ctx.state.tour.running)


def _remember(ctx: Any) -> None:
    """Persist which tours have been finished, so Home can stop offering them."""

    settings = getattr(ctx, "settings", None)
    if settings is None:
        return
    try:
        settings.set("tours_finished", list(ctx.state.tour.finished))
    except Exception as exc:  # pragma: no cover - a settings write is best effort
        ctx.toast(f"Could not remember the finished tour: {exc}", "warn")


def restore(ctx: Any) -> None:
    """Read the finished-tour list back at startup."""

    settings = getattr(ctx, "settings", None)
    if settings is None:
        return
    try:
        saved = as_list(settings.get("tours_finished"))
    except Exception:  # pragma: no cover - a missing key is not an error
        return
    ctx.state.tour.finished = tuple(str(key) for key in saved)


# -- drawing ---------------------------------------------------------------


def _hole(ctx: Any, step: Any) -> tuple[float, float, float, float] | None:
    """This frame's rect for the step's anchor, padded, or ``None``.

    ``None`` is an ordinary state rather than a failure: a control inside a
    collapsed section or behind another tab simply did not draw, and the card
    still has something to say about it.
    """
    if not step.anchor:
        return None
    found = anchors.rect(step.anchor)
    if found is None:
        return None
    pad = sp(HOLE_PAD)
    x, y, w, h = found
    return (x - pad, y - pad, w + pad * 2, h + pad * 2)


def _veil_spans(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    holes: list[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    """The scrim's paint rectangles for one viewport and any number of holes.

    A horizontal-band decomposition rather than four rectangles around one
    hole, because ``_veil`` has two: the control being pointed at, and the card
    doing the pointing. Bands are cut at every hole edge and each band is
    filled only where no hole covers it, so no pixel is painted twice -- two
    overlapping fills at 0.55 would leave a visibly darker patch wherever they
    crossed, and "darker where two rectangles happen to meet" is the kind of
    artefact that reads as a rendering bug rather than as a design.

    Pure geometry, out of ``_veil`` on purpose (the 2026-09-11 audit's
    tour-02): the overlap/merge arithmetic here is the part most likely to hide
    an off-by-one that double-paints a pixel or leaves a hole's edge dim, and
    it is exactly the "decidable half" the 2026-09-08 audit split ``_hole`` and
    ``_card_pos`` out for -- this is the third piece of ``_veil`` that was
    still missing a test of its own.
    """
    edges = sorted({y0, y1} | {v for _x, y, _w, h in holes for v in (y, y + h) if y0 < v < y1})
    rects: list[tuple[float, float, float, float]] = []
    for top, bottom in zip(edges, edges[1:], strict=False):
        spans = sorted(
            (max(x0, hx), min(x1, hx + hw))
            for hx, hy, hw, hh in holes
            if hy < bottom and hy + hh > top and hx + hw > x0 and hx < x1
        )
        cursor = x0
        for left, right in spans:
            if left > cursor:
                rects.append((cursor, top, left, bottom))
            cursor = max(cursor, right)
        if cursor < x1:
            rects.append((cursor, top, x1, bottom))
    return rects


def _veil(viewport: Any, holes: list[tuple[float, float, float, float]]) -> None:
    """Dim the viewport except where the holes are.

    See :func:`_veil_spans` for the band/overlap arithmetic; this is just the
    draw call over what it returns. The card needs a hole for a reason that is
    not obvious until you see it -- the scrim is on the *foreground* draw list,
    which imgui paints above every window including the card, so without a
    hole the tour dims its own text.
    """
    draw = imgui.get_foreground_draw_list()
    colour = imgui.get_color_u32(theme.rgba(theme.TOUR_VEIL, VEIL_ALPHA))
    # The whole viewport, not the work area, which is what
    # ``App._transition_overlay`` covers for the same reason: the work area
    # excludes the setup banner, and a scrim that dims the app apart from one
    # bright strip along the top reads as the scrim having failed.
    x0 = viewport.pos.x
    y0 = viewport.pos.y
    x1 = x0 + viewport.size.x
    y1 = y0 + viewport.size.y
    for left, top, right, bottom in _veil_spans(x0, y0, x1, y1, holes):
        draw.add_rect_filled((left, top), (right, bottom), colour)


def _ring(hole: tuple[float, float, float, float]) -> None:
    """The outline around the hole, on the same list as the scrim."""

    hx, hy, hw, hh = hole
    imgui.get_foreground_draw_list().add_rect(
        (hx, hy),
        (hx + hw, hy + hh),
        imgui.get_color_u32(theme.rgba(theme.TOUR_RING, 0.95)),
        sp(6),
        thickness=sp(2.0),
    )


def _card_pos(
    viewport: Any,
    hole: tuple[float, float, float, float] | None,
    bottom_offset: float = 0.0,
) -> tuple[float, float]:
    """Bottom-right by default, and out of the hole's way when there is one.

    Only the horizontal side is swapped. A card that also chased the hole
    vertically would jump the length of the window between two steps pointing
    at the top and bottom of the same pane, and a reader tracking a moving card
    is not reading it.

    ``bottom_offset`` -- already-scaled design pixels -- lifts the card clear
    of ``panes.bottom_pane``, which anchors to the same edge.
    """
    margin = sp(tokens.SP_4) + bottom_offset
    y = viewport.work_pos.y + viewport.work_size.y - margin
    right = viewport.work_pos.x + viewport.work_size.x - margin
    if hole is not None:
        hx, _hy, hw, _hh = hole
        centre = viewport.work_pos.x + viewport.work_size.x * 0.5
        if hx + hw * 0.5 > centre:
            return (viewport.work_pos.x + margin + sp(CARD_W), y)
    return (right, y)


def draw(ctx: Any) -> None:
    """The whole tour: scrim, ring and card. Called once from ``App._overlays``."""

    state = getattr(ctx.state, "tour", None)
    if state is None or not state.running:
        _was_open[0] = False
        return
    tour = find_tour(state.key)
    step = tour.step(state.index) if tour is not None else None
    if tour is None or step is None:
        state.stop()
        _was_open[0] = False
        return

    appearing = not _was_open[0]
    _was_open[0] = True
    state.satisfied = satisfied(ctx, step.done.name, step.done.arg)

    # Suspended -- not stopped -- while a real modal is up. The scrim is on the
    # *foreground* draw list, which composites above every window
    # unconditionally; that is why the card needs a hole of its own, and it is
    # equally why a confirm, a prompt, the matte preview or the first-run sheet
    # opening mid-tour gets dimmed by a tour that has no idea it is there. The
    # ring is worse than the veil: it would circle a control the modal is now
    # covering, pointing the reader at something they cannot reach.
    #
    # Suspending rather than holing out each modal's rect, because a hole needs
    # a rectangle and imgui does not hand those out at foreground-draw time --
    # and because the step is not actionable anyway while something else owns
    # the keyboard. ``_was_open`` is left set, so the card does not replay its
    # appear animation when the modal closes and the tour comes back.
    from ..dialogs import modal_open

    if modal_open(ctx):
        return

    viewport = imgui.get_main_viewport()
    hole = _hole(ctx, step)
    holes = [one for one in (hole, _card_rect[0]) if one is not None]
    _veil(viewport, holes)
    if hole is not None:
        _ring(hole)
    _card(ctx, viewport, tour, step, hole, appearing)


def _card(
    ctx: Any,
    viewport: Any,
    tour: Any,
    step: Any,
    hole: tuple[float, float, float, float] | None,
    appearing: bool,
) -> None:
    from . import bottom_pane

    state = ctx.state.tour
    alpha, rise = widgets.popover_enter("tour", appearing)
    x, y = _card_pos(viewport, hole, sp(bottom_pane.reserve(ctx)))
    imgui.set_next_window_pos((x, y + rise), imgui.Cond_.always.value, (1.0, 1.0))
    imgui.set_next_window_size((sp(CARD_W), 0))
    frosted = widgets.frosted()
    if frosted:
        imgui.set_next_window_bg_alpha(0.0)
    imgui.push_style_var(imgui.StyleVar_.alpha.value, alpha)
    radius = widgets.push_surface_rounding()
    opened = imgui.begin(
        "##tour-card",
        None,
        imgui.WindowFlags_.no_title_bar.value
        | imgui.WindowFlags_.no_move.value
        | imgui.WindowFlags_.no_resize.value
        | imgui.WindowFlags_.no_collapse.value
        | imgui.WindowFlags_.always_auto_resize.value
        | imgui.WindowFlags_.no_saved_settings.value,
    )[0]
    widgets.pop_surface_rounding()
    if opened:
        widgets.window_shadow("overlay", radius=radius)
        if frosted:
            widgets.window_backdrop(radius=radius)
        # tour-01 (2026-09-07 audit): computed while ``##tour-card`` is still
        # the current window, which is the only place ``is_window_focused()``
        # can answer for *this* window rather than whichever one imgui last
        # gave focus to. Stashed for ``has_focus()`` before anything below can
        # change it.
        focused = imgui.is_window_focused()
        _card_focused[0] = focused
        _card_body(ctx, tour, step, state, focused)
        pos = imgui.get_window_pos()
        size = imgui.get_window_size()
        # Padded, so the shadow under the card is not the one thing the scrim
        # still darkens -- a bright card with a dimmed halo reads as a seam.
        pad = sp(HOLE_PAD)
        _card_rect[0] = (pos.x - pad, pos.y - pad, size.x + pad * 2, size.y + pad * 2)
    else:
        _card_focused[0] = False
    imgui.end()
    imgui.pop_style_var()


def _card_body(ctx: Any, tour: Any, step: Any, state: Any, focused: bool) -> None:
    widgets.secondary(f"{tour.title} - {state.index + 1} of {len(tour)}")
    imgui.same_line()
    close_w = imgui.get_frame_height()
    imgui.set_cursor_pos_x(
        max(imgui.get_cursor_pos_x() + imgui.get_content_region_avail().x - close_w, 0.0)
    )
    if widgets.icon_button(f"{icons.CIRCLE_X}##tour-close", "End the tour (Esc)", borderless=True):
        stop(ctx)
        return

    widgets.pane_title(step.title)
    for paragraph in step.body.split("\n\n"):
        imgui.text_wrapped(paragraph)
        imgui.dummy((0, sp(tokens.SP_1)))

    if step.done.name != "manual":
        widgets.secondary("Waiting for you." if not state.satisfied else "Done.")

    imgui.dummy((0, sp(tokens.SP_1)))
    # The card takes the keyboard for its own two verbs. Esc already ends the
    # tour from ``App._shortcut``; without these the card could only be
    # clicked, so a tour meant to teach the app could not itself be read from
    # the keyboard. Read through imgui rather than as a pygame binding, for
    # ``panes/palette.py``'s reason: this is a floating surface and the keys
    # belong to it rather than to the mode behind it.
    #
    # Gated on ``focused``: the 2026-09-07 audit (tour-01) found these reads
    # unconditional, so an Enter confirming an unrelated rename, or an arrow
    # moving a text caret somewhere else on screen, also advanced -- or
    # completed -- the tour. imgui's key state is global, not scoped to the
    # window that is current when it is read, so the card has to ask
    # ``is_window_focused()`` itself rather than assume nothing else wants
    # the same press.
    back = focused and imgui.is_key_pressed(imgui.Key.left_arrow)
    # Both Enter keys, the 2026-09-13 audit's tour-02: ``App._shortcut``
    # swallows ``K_KP_ENTER`` whenever the tour has focus (main.py), and
    # every other Enter-confirms site in the app reads both keys -- reading
    # only ``Key.enter`` here left the numpad key dead on this one surface.
    forward = focused and (
        imgui.is_key_pressed(imgui.Key.right_arrow)
        or imgui.is_key_pressed(imgui.Key.enter)
        or imgui.is_key_pressed(imgui.Key.keypad_enter)
    )
    if state.index > 0 and back:
        advance(ctx, -1)
        return
    if state.index > 0 and controls.button("Back##tour", role=controls.ButtonRole.GHOST):
        advance(ctx, -1)
        return
    if state.index > 0:
        imgui.same_line()
    label = "Next" if state.index + 1 < len(tour) else "Finish"
    # A satisfied condition offers the step's exit rather than taking it: the
    # reader has just done the thing and the card would otherwise vanish
    # mid-sentence, which reads as the app having lost their place.
    #
    # Deliberately not gated on state.satisfied: for a non-manual `done`,
    # satisfied() only changes the "Waiting for you." / "Done." sentence
    # above -- Next/Finish stays enabled either way, so a reader can click
    # past a step without doing the named thing and land on a later card
    # whose copy assumes it happened. That is the accepted cost of tours
    # being conventionally skippable, not an oversight; making Next actually
    # wait on satisfied() is a product decision, not a bug fix.
    if widgets.primary_button(f"{label}##tour") or forward:
        advance(ctx)
        return
    if step.chapter is not None:
        imgui.same_line()
        if controls.button("Read more##tour", role=controls.ButtonRole.GHOST):
            manual_render.open_at(ctx, step.chapter)
