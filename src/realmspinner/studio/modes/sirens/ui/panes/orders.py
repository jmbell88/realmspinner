"""Sirens' left-bottom pane: the order list, and the patterns it points at.

**The order holds pattern uids, and this pane never shows one.** It draws each
entry's *position* and the pattern's name, because a uid is an implementation
detail the user has no way to act on -- and because the document's rule (uids,
never indices) exists precisely so that deleting a pattern cannot silently
repoint an order entry. A pane that displayed the number would be teaching the
user to rely on a number that is deliberately meaningless to them.

**Adding a pattern does not add it to the order**, and removing one from the
order does not delete it. They are two lists and the whole point of an order
list is that a pattern can appear in it more than once, or not at all.

**Every verb here is one ``SongDoc`` already had.** ``set_order`` is insert,
remove, move and retarget over a list of integers; ``loop_order`` is any entry
and not only the first; ``remove_pattern``, ``duplicate_pattern`` and
``rename_pattern`` are one call each. Until 2026-09-03 the pane offered two of
them, so an intro-then-loop song -- the ordinary shape of a game track -- could
not be expressed from the UI at all (the 2026-09-02 review, section 8).
"""

from __future__ import annotations

from typing import Any

from ..... import anchors, controls, icons, tokens, widgets
from .....manual import render as manual_render
from .....tokens import sp
from ... import mode as sirens_mode
from ...engine import document as D

_BUSY_WHY = "This song is being written; the buttons come back when it lands."
_ROW_WHY = "This entry is already at the end it would move to."
_FULL_WHY = f"A song holds {D.MAX_PATTERNS} patterns."
_ORDER_FULL_WHY = f"A song's order holds {D.MAX_ORDER} steps."


def pattern_room(doc: Any, editable: bool) -> tuple[bool, str]:
    """Whether "Add a pattern" or "Duplicate" can fire, and why not if not.

    Pulled out as a pure function so the 2026-09-07 audit's finding sirens-03
    has something a test can call with no imgui frame: both buttons used to
    call ``add_pattern``/``duplicate_pattern`` with no cap check and no
    try/except, so filling a song to its own documented ``MAX_PATTERNS``
    ceiling raised a bare ``ValueError`` out of ``draw()`` -- and ``guard.py``
    replaces the whole pane after three of those in a row.
    """
    if not editable:
        return False, _BUSY_WHY
    if len(doc.patterns) >= D.MAX_PATTERNS:
        return False, _FULL_WHY
    return True, ""


def add_to_order_reason(effect: str, doc: Any, editable: bool) -> str:
    """"Add to the order"'s disabled reason, in priority order. -> "" when
    the button should be enabled.

    Pulled out pure for the 2026-09-08 audit's finding sirens-05: this was an
    inline three-way ternary, unlike every sibling disabled-reason in this
    file and its neighbours -- :func:`pattern_room`,
    ``sirens_instruments.instrument_room``, ``sirens_effects.delete_reason``,
    ``sirens_instruments.sample_delete_reason`` -- each pulled out and unit
    tested after the 2026-09-07 audit found the same shape of bug in them
    (findings sirens-03/04/05 of that round). An inline ternary with untested
    priority among competing disabled causes is precisely what produced those,
    and this one had nothing to catch a wrong priority before it shipped.

    **``editable`` goes first (the 2026-09-14 audit, finding sirens-04).**
    The order used to put the effect-column case before it, so a song that
    was busy saving while the caret happened to sit on an effect cell named
    "pick a song pattern first" -- a fix that does nothing, since the button
    stays disabled either way until the save lands. "Busy" is the state that
    is actually blocking the button, and it is also the one about to change
    on its own; the other two reasons describe what the user typed and stay
    true until they change it.
    """
    if not editable:
        return _BUSY_WHY
    if effect:
        return (
            f"The grid is editing the sound effect {effect}, and an effect's "
            "pattern is not part of the song. Pick a song pattern first."
        )
    if not doc.patterns:
        return "There is no pattern to add yet."
    # sirens-02 (2026-09-18 audit, second run): ``set_order`` used to clip
    # silently to ``MAX_ORDER`` rather than raise -- every sibling ceiling in
    # ``document.py`` (patterns, channels, oneshots, samples) already raises
    # by name, and this pane's own ``_FULL_WHY`` above already greys "Add a
    # pattern" the same way. At 256 entries this button stayed live and
    # calling it silently did nothing; it now raises a ``ValueError`` its
    # caller does not catch, so it must be greyed here first.
    if len(doc.order) >= D.MAX_ORDER:
        return _ORDER_FULL_WHY
    return ""


def draw(ctx: Any) -> None:
    from imgui_bundle import imgui

    anchors.mark_window("sirens/orders")
    state = sirens_mode.ensure(ctx)
    tab = state.active
    widgets.section("Order")
    manual_render.help_button(ctx, "sirens-orders")

    if tab is None:
        return

    doc = tab.doc
    editable = not tab.busy

    width = widgets.grid_width(2)
    addable_pattern, pattern_why = pattern_room(doc, editable)
    if widgets.disabled_button(
        f"{icons.PLUS} Add a pattern", addable_pattern, (width, 0), reason=pattern_why
    ):
        pattern = doc.add_pattern()
        sirens_mode.request_rerender(ctx, tab)
        sirens_mode.set_caret(ctx, pattern=pattern.uid)
    imgui.same_line()
    # **A sound effect's pattern is not a song pattern.** Adding an effect mints
    # a pattern of its own and points the grid at it, so with the caret there
    # this button used to append a coin pickup into the middle of the song and
    # say nothing.
    effect = sirens_mode.oneshot_name_for_caret(ctx, tab)
    addable = editable and bool(doc.patterns) and not effect
    add_why = add_to_order_reason(effect, doc, editable)
    if widgets.disabled_button(
        f"{icons.PLUS} Add to the order", addable, (width, 0), reason=add_why
    ):
        current = state.pattern or doc.patterns[0].uid
        if doc.set_order(list(doc.order) + [current]):
            sirens_mode.request_rerender(ctx, tab)

    imgui.dummy((0, sp(tokens.SP_1)))
    _order(ctx, state, tab, editable)
    imgui.dummy((0, sp(tokens.SP_2)))
    widgets.section("Patterns")
    _patterns(ctx, state, tab, editable)


def _name_of(doc: Any, uid: int) -> str:
    pattern = doc.pattern(uid)
    if pattern is None:
        # An order entry whose pattern is gone. ``set_order`` refuses one, so
        # this is unreachable through the app -- but a hand-edited ``.rsng``
        # can carry it and a row that renders as an exception is worse than a
        # row that says what is wrong.
        return "(missing)"
    return pattern.name or f"Pattern {doc.patterns.index(pattern) + 1}"


def reuse_counts(order: list[int]) -> dict[int, int]:
    """Every pattern uid in ``order`` -> how many entries name it.

    Pulled out pure so a test can call it with no imgui frame, the
    ``pattern_room``/``moved_loop`` idiom this file already uses. A song where
    the chorus is entries 02 and 05 used to draw two identical rows with
    nothing on screen saying they were the same pattern -- reuse is the whole
    point of an order list being a list of references rather than a list of
    patterns, and it was invisible.
    """
    counts: dict[int, int] = {}
    for uid in order:
        counts[uid] = counts.get(uid, 0) + 1
    return counts


def moved_loop(loop: int, index: int, to: int) -> int:
    """Where a loop point ends up when the entry at ``index`` moves to ``to``.

    The loop is an *index* into the order list, so moving entries under it
    silently repoints it at whatever landed there -- a song that loops from
    somewhere the user never chose. Pure, so the arithmetic is assertable
    without a frame: the moved entry carries its own loop with it, and every
    entry the move stepped over shifts by one the other way.
    """
    if loop < 0 or index == to:
        return loop
    if loop == index:
        return to
    if index < loop <= to:
        return loop - 1
    if to <= loop < index:
        return loop + 1
    return loop


def _reorder(ctx: Any, tab: Any, index: int, to: int) -> None:
    """Move one order entry, and carry the loop point with it.

    Two ``SongDoc`` calls and one intention, which is why the loop follows
    here rather than being left for the user to notice and fix.
    """
    doc = tab.doc
    order = list(doc.order)
    if not (0 <= index < len(order) and 0 <= to < len(order)) or index == to:
        return
    after = moved_loop(doc.loop_order, index, to)
    order.insert(to, order.pop(index))
    if not doc.set_order(order):
        return
    if after != doc.loop_order:
        doc.set_song(loop_order=after)
    sirens_mode.request_rerender(ctx, tab)


def _order(ctx: Any, state: Any, tab: Any, editable: bool) -> None:
    from imgui_bundle import imgui

    doc = tab.doc
    if not doc.order:
        widgets.muted("Nothing in the order yet -- the song plays nothing.")
        return
    order = list(doc.order)
    looping = doc.loop_order >= 0
    counts = reuse_counts(order)
    for index, uid in enumerate(order):
        # The *entry*, not the pattern (S3): a chorus at 00 and 03 used to draw
        # both rows highlighted at once, because a uid cannot tell them apart.
        selected = state.pattern == uid and state.order_index in (None, index)
        mark = f"{icons.UNDO} " if looping and doc.loop_order == index else ""
        if controls.selectable(
            f"{index:02d}  {mark}{_name_of(doc, uid)}###sirens-order-{index}", selected
        )[0]:
            sirens_mode.set_caret(ctx, pattern=uid, order_index=index)
        # A pattern used more than once in the order is not a coincidence
        # worth reading two identical rows to spot -- "x2" says it once, on
        # every row it is true of.
        if counts[uid] > 1:
            imgui.same_line()
            widgets.muted(f"×{counts[uid]}")
        # The row's own verbs, in the order a person reaches for them: move it,
        # point it somewhere else, take it out. The loop below breaks after any
        # of them, because each rewrites the list being walked.
        imgui.same_line()
        if widgets.disabled_button(
            f"{icons.ARROW_UP}###sirens-order-up-{index}",
            editable and index > 0,
            (0, 0),
            reason=_BUSY_WHY if not editable else _ROW_WHY,
        ):
            _reorder(ctx, tab, index, index - 1)
            break
        imgui.same_line()
        if widgets.disabled_button(
            f"{icons.ARROW_DOWN}###sirens-order-down-{index}",
            editable and index < len(order) - 1,
            (0, 0),
            reason=_BUSY_WHY if not editable else _ROW_WHY,
        ):
            _reorder(ctx, tab, index, index + 1)
            break
        imgui.same_line()
        if _retarget(ctx, tab, index, uid, editable):
            break
        imgui.same_line()
        if widgets.disabled_button(
            f"{icons.TRASH}###sirens-order-drop-{index}", editable, (0, 0),
            reason=_BUSY_WHY,
        ):
            del order[index]
            if doc.set_order(order):
                sirens_mode.request_rerender(ctx, tab)
            break

    changed, value = controls.checkbox("Loop the song", looping)
    if changed and doc.set_song(loop_order=0 if value else -1):
        sirens_mode.request_rerender(ctx, tab)
    if not looping:
        widgets.muted_wrapped(
            "Off, the song plays once and its instruments ring out. On, it "
            "loops from whichever entry you pick -- an intro that plays once "
            "is the point of the choice."
        )
        return
    # Through ``widgets.labeled_slider_int`` rather than a bare
    # ``controls.slider_int`` (the 2026-09-07 audit): the -1-width rule leaves
    # a bare imgui label undrawn, and "Loop from" was one of the sliders it
    # was happening to.
    changed, value = widgets.labeled_slider_int(
        "Loop from", doc.loop_order, 0, max(0, len(order) - 1), enabled=editable
    )
    controls.fold_undo(doc.history)
    if changed and doc.set_song(loop_order=int(value)):
        sirens_mode.request_rerender(ctx, tab)
    if 0 <= doc.loop_order < len(order):
        widgets.muted(
            f"{icons.UNDO} back to {doc.loop_order:02d}  "
            f"{_name_of(doc, order[doc.loop_order])}"
        )


def _retarget(ctx: Any, tab: Any, index: int, uid: int, editable: bool) -> bool:
    """The "point this entry at another pattern" menu. -> whether it changed.

    A menu rather than a second list to drag between: an order entry *is* a
    pattern reference, and the shortest way to say "bar 3 is the chorus after
    all" is to pick the chorus where bar 3 is drawn.
    """
    from imgui_bundle import imgui

    doc = tab.doc
    name = f"sirens-order-point-popup-{index}"
    if widgets.disabled_button(
        f"{icons.MOVE}###sirens-order-point-{index}", editable, (0, 0),
        reason=_BUSY_WHY,
    ):
        imgui.open_popup(name)
    changed = False
    if imgui.begin_popup(name):
        # **Gated on ``editable``, not merely drawn (the 2026-09-13 audit,
        # finding sirens-04).** The button above that opens this popup already
        # checks ``editable``, but a popup already open stays open across a
        # save starting -- imgui does not close it for you -- so with these
        # rows undisabled, a click landed mid-save still called ``set_order``
        # on a tab the rest of the pane was refusing to touch.
        if not editable:
            widgets.muted(_BUSY_WHY)
        for pattern in list(doc.patterns):
            clicked = controls.selectable(
                f"{_name_of(doc, pattern.uid)}###sirens-point-{index}-{pattern.uid}",
                pattern.uid == uid,
                enabled=editable,
            )[0]
            # The visual disable above is imgui's job and is not this
            # function's to trust: a popup left open across a save starting is
            # exactly the frame where "disabled" and "still clickable" can
            # disagree, so the actual mutation below is refused in software
            # too, whatever the widget answered.
            if clicked and editable:
                order = list(doc.order)
                order[index] = pattern.uid
                if doc.set_order(order):
                    sirens_mode.request_rerender(ctx, tab)
                    changed = True
        imgui.end_popup()
    return changed


def _patterns(ctx: Any, state: Any, tab: Any, editable: bool) -> None:
    from imgui_bundle import imgui

    doc = tab.doc
    if not doc.patterns:
        widgets.muted("No patterns yet.")
        return
    for pattern in list(doc.patterns):
        selected = state.pattern == pattern.uid
        if controls.selectable(
            f"{_name_of(doc, pattern.uid)}  ({pattern.rows} rows)"
            f"###sirens-pattern-{pattern.uid}",
            selected,
        )[0]:
            sirens_mode.set_caret(ctx, pattern=pattern.uid)
        if not selected:
            continue
        # ``##``-hidden with the name drawn above (the 2026-09-07 audit):
        # ``"Name###sirens-pattern-name-{uid}"`` at width -1 kept "Name" as
        # visible text and only overrode the id, so imgui drew that visible
        # text to the right of a field with no room for it -- the same
        # not-drawn-at-all defect a plain ``##`` prefix avoids, and every other
        # field in this pane already uses.
        widgets.field_label("Name")
        imgui.set_next_item_width(-1)
        changed, name = controls.input_text(
            f"##sirens-pattern-name-{pattern.uid}",
            pattern.name,
            enabled=editable,
            commit=True,
        )
        if changed:
            doc.rename_pattern(pattern.uid, name)
        changed, value = widgets.labeled_slider_int(
            "Rows", pattern.rows, 1, 256, enabled=editable
        )
        controls.fold_undo(doc.history)
        if changed and doc.resize_pattern(pattern.uid, int(value)):
            sirens_mode.request_rerender(ctx, tab)
            sirens_mode.clamp_caret(ctx, tab)
        width = widgets.grid_width(2)
        duplicatable, duplicate_why = pattern_room(doc, editable)
        if widgets.disabled_button(
            f"{icons.COPY} Duplicate###sirens-pattern-copy-{pattern.uid}",
            duplicatable,
            (width, 0),
            reason=duplicate_why,
        ):
            copy = doc.duplicate_pattern(pattern.uid)
            sirens_mode.set_caret(ctx, pattern=copy.uid)
            # Not added to the order: the two lists are separate by design, and
            # a copy that started playing in the middle of the song the moment
            # it was made would be a surprise, not a shortcut.
            sirens_mode.request_rerender(ctx, tab)
            break
        imgui.same_line()
        # **Delete says what it takes with it.** ``remove_pattern`` drops every
        # order entry naming this pattern in the same undo step, which is right
        # and is exactly the kind of thing a person should be told before the
        # press rather than after it.
        used = sum(1 for one in doc.order if one == pattern.uid)
        if widgets.disabled_button(
            f"{icons.TRASH} Delete###sirens-pattern-drop-{pattern.uid}",
            editable and len(doc.patterns) > 1,
            (width, 0),
            reason=_BUSY_WHY
            if not editable
            else "A song keeps at least one pattern.",
        ):
            sirens_mode.confirm_remove_pattern(ctx, tab, pattern.uid, used)
            break
        if used:
            widgets.muted(
                f"In the order {used} time(s); deleting takes those entries too."
            )
