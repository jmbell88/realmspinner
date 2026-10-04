"""Rearranging the workspace, drawn as one overlay over the whole viewport.

**Not a ``panes/`` file**, deliberately: it trips neither the help-button gates
nor the no-imgui-controls guard, and both would be false about it -- it is not
a pane, it is a picture of where the panes are.

It draws **nothing inside any pane**. ``layout.column`` records each pane's
rect into ``layout.FRAME_PANES`` as it draws, and this renders every handle and
drop bar into a single full-viewport window afterwards. One window, one draw
list: ``widgets.section_blocks``' double-split corruption -- which surfaces in
a *different* pane from the one that caused it -- is out of the picture by
construction rather than by care.

Splitters are suppressed while editing, because a handle and a drag target on
the same two pixels is a gesture nobody can aim; sizes are dragged in normal
mode, where the handles are the only thing there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Design px. How close to a pane's top or bottom edge a drop lands "before" or
#: "after" it rather than "onto" it.
EDGE_BAND = 24.0


@dataclass
class EditState:
    """What the editor is in the middle of. Lives on ``AppState``."""

    open: bool = False
    #: The slot being dragged, or "".
    dragging: str = ""
    #: The column ``dragging`` was picked up from, or "". The 2026-09-26
    #: audit, finding shell-documents-03: ``_commit`` used to accept a drop
    #: into any column under the mouse, including one the slot did not come
    #: from, and persist it there -- but ``reconcile`` filters a stored id
    #: against *that* column's own built-in set, so the moved id was dropped
    #: from the column it landed in and reinserted at its original built-in
    #: position in the column it left, on the very next load. The drop looked
    #: accepted and silently undid itself, while still overwriting the source
    #: column's saved order. Recorded here so ``_commit`` can refuse a
    #: cross-column drop instead of accepting one ``reconcile`` will undo.
    dragging_column: str = ""
    #: Where it would land: ``(column_id, index)``, or None.
    target: tuple[str, int] | None = None
    #: The active layout's hidden slots for the open workspace, mirrored from
    #: ``layouts.Library.hidden`` every frame and written straight back by
    #: :func:`draw`'s hide badge -- there is no separate "done" gesture here,
    #: so a toggle is as immediate as a drag's own commit.
    hidden: set[str] = field(default_factory=set)
    #: Where the last frame drew each hidden slot's "Show" chip, as
    #: ``(x, y, w, h)`` by slot id. The twin of ``layout.FRAME_PANES`` for the
    #: one kind of slot that has no pane to record: a hidden slot is dropped by
    #: ``skeletons.ordered`` before ``layout.column`` draws it, so it never gets
    #: a rect and the editor's per-pane badge could never be drawn for it
    #: (shell-11). Written by :func:`draw`; read by whatever drives it headlessly.
    chips: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)


def drop_index(rects: list[tuple[str, tuple[float, float, float, float]]], y: float) -> int:
    """Where a pointer at *y* would insert into a column. Pure.

    The whole of the placement rule, as arithmetic: above a pane's midpoint is
    before it, below is after it, and past the last pane is the end. Pure so
    every case -- an empty column, one pane, a pointer above the first -- is a
    plain assertion rather than a drag nobody can repeat.
    """

    for index, (_slot, rect) in enumerate(rects):
        middle = rect[1] + rect[3] * 0.5
        if y < middle:
            return index
    return len(rects)


def moved(order: list[str], slot: str, index: int) -> list[str]:
    """*order* with *slot* moved to *index*. Pure, and clamped.

    Removing first and then inserting is what makes a drag onto a pane's own
    place a no-op rather than a duplicate -- the index the pointer produced was
    computed against the list that still contained it.
    """

    out = [entry for entry in order if entry != slot]
    if slot in order:
        at = min(max(0, index if index <= order.index(slot) else index - 1), len(out))
        out.insert(at, slot)
    else:
        out.insert(min(max(0, index), len(out)), slot)
    return out


def toggle(state: Any, ctx: Any = None) -> None:
    """Shift+W. Enter and leave the editor.

    Given *ctx*, it refuses to *enter* in a mode with no skeleton: the editor
    there only says "cannot be rearranged yet" while ``layout.begin_frame``
    switches the workspace's own splitters off (shell-02).
    """

    edit = ensure(state)
    if ctx is not None and not edit.open:
        from . import skeletons

        if not skeletons.for_mode(ctx, state.mode):
            return
    edit.open = not edit.open
    edit.dragging = ""
    edit.dragging_column = ""
    edit.target = None


def ensure(state: Any) -> EditState:
    """The editor's state, made on first use."""

    edit = getattr(state, "layout_edit", None)
    if edit is None:
        edit = EditState()
        state.layout_edit = edit
    return edit


def draw(app: Any, ctx: Any, viewport: Any) -> None:
    """The overlay: a handle on every movable pane, and a drop bar.

    Called from ``App._overlays``, after the workspace has drawn and recorded
    its rects.
    """
    from imgui_bundle import imgui

    from . import icons, skeletons, theme
    from . import layout as layout_mod
    from .tokens import sp

    edit = ensure(ctx.state)
    if not edit.open:
        return
    columns = skeletons.for_mode(ctx, ctx.state.mode)
    if not columns:
        # A workspace whose columns are not data yet cannot be rearranged, and
        # saying so is better than an editor that appears and does nothing.
        _banner(ctx, "This workspace cannot be rearranged yet.")
        return
    library = getattr(app, "layouts", None)
    if library is not None:
        # Read fresh every frame rather than seeded once: a toggle below
        # writes straight through ``_persist``, so this and the library agree
        # by the next frame, and reopening the editor never shows a stale
        # in-session set left over from a workspace switch.
        edit.hidden = set(library.hidden(ctx.state.mode))
    rects = layout_mod.FRAME_PANES
    draw_list = imgui.get_foreground_draw_list()
    mouse = imgui.get_mouse_pos()
    hovered = ""
    hovered_column = ""
    toggled = False
    for column in columns.values():
        for slot in column.live(ctx):
            rect = rects.get(slot.id)
            if rect is None:
                continue
            x, y, w, h = rect
            inside = x <= mouse.x < x + w and y <= mouse.y < y + h
            colour = theme.ACCENT if inside else theme.DIVIDER
            # ``thickness`` scaled, like every other highlight rect in the app.
            # ``add_rect``'s fifth positional is ``thickness``, not ``flags``
            # -- the previous call had them swapped and passed a float
            # (``sp(1.0)``) where ``flags`` wants an ``int``, which
            # ``imgui_bundle`` 1.92 raises a ``TypeError`` on rather than
            # coercing. Found while building shell-07's hide badge below:
            # every frame the editor was open threw here, before this fixer's
            # own change ever ran, on the first movable slot in the workspace.
            draw_list.add_rect(
                (x, y),
                (x + w, y + h),
                imgui.get_color_u32(theme.rgba(colour, 0.9)),
                sp(4),
                sp(1.0),
            )
            # Only a *shown* slot reaches here: a hidden one has no rect (see
            # ``EditState.chips``), so its way back is the chip strip below.
            label = slot.label if slot.movable else f"{slot.label} (fixed)"
            draw_list.add_text(
                (x + sp(8), y + sp(6)),
                imgui.get_color_u32(theme.rgba(theme.TEXT)),
                label,
            )
            # The 2026-09-08 audit's shell-07: ``EditState.hidden``'s
            # docstring promised a slot could be hidden from here, but no
            # gesture ever wrote to it -- the hiding machinery underneath
            # (``Slot.hideable``, ``Arrangement.hidden``,
            # ``skeletons.ordered``'s filtering) was built and tested but
            # unreachable from the editor. A fixed-size square badge, mirroring
            # the "(fixed)" label already drawn for a non-movable slot, rather
            # than a hit target sized to its icon glyph -- so the click target
            # does not move if the glyph's metrics ever do.
            badge_hit = False
            if slot.hideable:
                side = imgui.get_frame_height()
                bx = x + w - sp(6) - side
                by = y + sp(6)
                badge_hit = bx <= mouse.x < bx + side and by <= mouse.y < by + side
                badge_colour = theme.ACCENT if badge_hit else theme.ELEV_2
                draw_list.add_rect_filled(
                    (bx, by),
                    (bx + side, by + side),
                    imgui.get_color_u32(theme.rgba(badge_colour, 0.9)),
                    sp(4),
                )
                draw_list.add_text(
                    (bx + side * 0.2, by + side * 0.15),
                    imgui.get_color_u32(theme.rgba(theme.TEXT)),
                    icons.EYE,
                )
                if imgui.is_mouse_clicked(0) and badge_hit:
                    edit.hidden.add(slot.id)
                    toggled = True
            if inside and slot.movable and not badge_hit:
                hovered = slot.id
                hovered_column = column.id
    # shell-11: the way back for a pane the eye above has hidden. Settings has
    # always listed it, but the editor is where the user just hid it and where
    # they will look; the strip is drawn here because a hidden slot has no pane
    # (hence no badge) to carry the control.
    bottom = _banner(
        ctx,
        "Drag a pane onto another to reorder it, or press the eye to hide "
        "one. Shift+W leaves; Settings > Advanced resets.",
    )
    shown = _chips(columns, ctx, edit, mouse, draw_list, bottom)
    if shown is not None:
        edit.hidden.discard(shown)
        toggled = True
    chip_hit = shown is not None or any(
        x <= mouse.x < x + w and y <= mouse.y < y + h for x, y, w, h in edit.chips.values()
    )
    if imgui.is_mouse_clicked(0) and hovered and not chip_hit:
        edit.dragging = hovered
        # shell-documents-03: recorded so ``_commit`` can tell a reorder
        # within this column from a drop onto another one.
        edit.dragging_column = hovered_column
    if edit.dragging and not imgui.is_mouse_down(0):
        _commit(app, ctx, columns, edit, mouse)
        edit.dragging = ""
    if toggled:
        # Immediate, like a drag's own commit: there is no separate "done"
        # gesture in this editor, so a hide toggle with no drag afterward
        # must not be lost when the panel closes. Written from the *stored*
        # order (``Library.order`` over the built-in ids), not from
        # ``skeletons.ordered`` and not from ``column.live(ctx)`` (the 2026-09-15
        # audit's shell-01: the built-in order would reset every column's
        # saved arrangement on a plain badge press). ``ordered`` also drops a
        # hidden slot, so the press that *un*-hid one wrote an arrangement with
        # its id missing and ``reconcile`` put it back at its built-in place,
        # not where the user had left it.
        _persist(
            app,
            ctx,
            edit,
            {
                c.id: _stored_order(ctx, library, c)
                for c in columns.values()
            },
        )


def _stored_order(ctx: Any, library: Any, column: Any) -> list[str]:
    """One column's saved order with its hidden slots still in it.

    **Reconciled against every slot the column declares, not only the live
    ones** (shell-43, the 2026-10-03 audit): a slot absent through
    ``Slot.when`` (``inker-tiles`` with no tileset, ``inker-preview`` with no
    frames) is not drawn, but its saved place is still the user's. Reconciling
    against ``column.live`` dropped it, so an unrelated drag or hide press made
    while it was away rewrote the column without it and it returned wherever
    ``reconcile`` re-inserted it.
    """

    builtin = [slot.id for slot in column.slots]
    if library is None:
        return builtin
    return library.order(ctx.state.mode, column.id, builtin)


def _seat_absent(visible: list[str], stored: list[str]) -> list[str]:
    """A reordered visible column with every slot it did not draw put back.

    A slot is absent when the layout hides it or when its ``Slot.when`` is
    false right now. Each such id goes after the nearest slot that preceded it
    in the stored order and is still shown, or first if nothing did -- so a drag
    moves the panes the user can see and leaves the invisible ones where they
    were left.
    """

    out = list(visible)
    for i, slot_id in enumerate(stored):
        if slot_id in out:
            continue
        before = next((p for p in reversed(stored[:i]) if p in out), None)
        out.insert(out.index(before) + 1 if before is not None else 0, slot_id)
    return out


def _chips(
    columns: Any, ctx: Any, edit: EditState, mouse: Any, draw_list: Any, top: float
) -> str | None:
    """Draw a "Show <pane>" chip for every hidden slot. -> the id pressed, if any.

    Recorded into ``edit.chips`` so the geometry has one owner (this function)
    and a headless test reads it rather than restating the layout.
    """
    from imgui_bundle import imgui

    from . import icons, theme
    from .tokens import sp

    edit.chips = {}
    pressed: str | None = None
    pad = sp(10)
    gap = sp(6)
    height = imgui.get_frame_height()
    room = imgui.get_io().display_size.x - pad
    x, y = pad, top + sp(6)
    for column in columns.values():
        for slot in column.live(ctx):
            if not slot.hideable or slot.id not in edit.hidden:
                continue
            text = f"{icons.EYE} Show {slot.label}"
            width = imgui.calc_text_size(text).x + sp(16)
            if x + width > room and x > pad:
                x, y = pad, y + height + gap
            edit.chips[slot.id] = (x, y, width, height)
            inside = x <= mouse.x < x + width and y <= mouse.y < y + height
            draw_list.add_rect_filled(
                (x, y),
                (x + width, y + height),
                imgui.get_color_u32(theme.rgba(theme.ACCENT if inside else theme.ELEV_2, 0.95)),
                sp(4),
            )
            draw_list.add_text(
                (x + sp(8), y + (height - imgui.get_text_line_height()) * 0.5),
                imgui.get_color_u32(theme.rgba(theme.TEXT)),
                text,
            )
            if inside and imgui.is_mouse_clicked(0):
                pressed = slot.id
            x += width + gap
    return pressed


def _banner(ctx: Any, text: str) -> float:
    """Draw the one-line hint. -> its bottom edge, so a strip can sit below it."""
    from imgui_bundle import imgui

    from . import theme
    from .tokens import sp

    draw_list = imgui.get_foreground_draw_list()
    size = imgui.calc_text_size(text)
    pad = sp(10)
    origin = (pad, pad)
    draw_list.add_rect_filled(
        (origin[0] - pad * 0.5, origin[1] - pad * 0.5),
        (origin[0] + size.x + pad * 0.5, origin[1] + size.y + pad * 0.5),
        imgui.get_color_u32(theme.rgba(theme.ELEV_2, 0.95)),
        sp(6),
    )
    draw_list.add_text(origin, imgui.get_color_u32(theme.rgba(theme.TEXT)), text)
    return origin[1] + size.y + pad * 0.5


def _commit(app: Any, ctx: Any, columns: Any, edit: EditState, mouse: Any) -> None:
    """Land a drag: work out the column and index, and record the arrangement."""

    from . import layout as layout_mod
    from . import skeletons

    library = getattr(app, "layouts", None)
    rects = layout_mod.FRAME_PANES
    for column in columns.values():
        # The 2026-09-15 audit's shell-01: this used to build ``live`` and
        # ``arrangement`` from ``column.live(ctx)``, the skeleton's built-in
        # declaration order -- not ``skeletons.ordered``, the order the
        # workspace actually drew this frame (reconciled against the saved
        # layout, hidden slots dropped). ``rects`` was recorded in the drawn
        # order, so a mismatch between ``live``'s order and the rects made
        # the drop index land wrong, and writing every *other* column back
        # from ``live(ctx)`` too silently reset each one's saved arrangement
        # to the built-in order on every drag or hide-badge press.
        live = [
            (slot.id, rects[slot.id])
            for slot in skeletons.ordered(ctx, library, ctx.state.mode, column)
            if slot.id in rects
        ]
        if not live:
            continue
        x = live[0][1][0]
        width = live[0][1][2]
        if not (x <= mouse.x < x + width):
            continue
        if edit.dragging_column and column.id != edit.dragging_column:
            # The 2026-09-26 audit, finding shell-documents-03: a drop onto a
            # column the slot did not come from used to be accepted and
            # persisted here, but ``layouts.Library.order`` reconciles a
            # stored id against *that* column's own built-in set
            # (``skeletons.ordered`` -> ``reconcile``), which drops an id
            # that column never natively owned -- so the pane vanished from
            # where it landed and reappeared at its built-in position in the
            # column it left, on the very next load, while the source
            # column's saved order had already been overwritten to remove
            # it. Refusing the drop outright is the safer, less invasive
            # fix: the pane stays exactly where it started and nothing is
            # rewritten.
            return
        index = drop_index(live, mouse.y)
        order = moved([slot for slot, _rect in live], edit.dragging, index)
        # Written from the *stored* order, as the hide toggle's persist is:
        # ``skeletons.ordered`` drops a hidden slot, so building either the
        # dragged column or any other from it forgot where a hidden pane had
        # been left, and showing it again put it at its built-in place.
        arrangement = {
            other.id: _stored_order(ctx, library, other) for other in columns.values()
        }
        arrangement[column.id] = _seat_absent(order, arrangement[column.id])
        # Anything the drag took *out* of another column leaves it.
        for key, ids in arrangement.items():
            if key != column.id:
                arrangement[key] = [entry for entry in ids if entry != edit.dragging]
        _persist(app, ctx, edit, arrangement)
        return


def _persist(app: Any, ctx: Any, edit: EditState, arrangement: dict[str, list[str]]) -> None:
    """Write one arrangement plus the current hidden set.

    Shared by a drag's own commit and a hide toggle -- both are immediate,
    since this editor has no separate "done" gesture: a drag lands the moment
    the mouse releases, and a hide toggle with no drag after it must not be
    lost when Shift+W simply closes the overlay.
    """

    library = getattr(app, "layouts", None)
    if library is not None:
        library.record(ctx.state.mode, arrangement, edit.hidden)
