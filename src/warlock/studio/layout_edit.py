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
    #: Where it would land: ``(column_id, index)``, or None.
    target: tuple[str, int] | None = None
    #: The active layout's hidden slots for the open workspace, mirrored from
    #: ``layouts.Library.hidden`` every frame and written straight back by
    #: :func:`draw`'s hide badge -- there is no separate "done" gesture here,
    #: so a toggle is as immediate as a drag's own commit.
    hidden: set[str] = field(default_factory=set)


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


def toggle(state: Any) -> None:
    """Shift+W. Enter and leave the editor."""

    edit = ensure(state)
    edit.open = not edit.open
    edit.dragging = ""
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
            hidden_now = slot.id in edit.hidden
            label = slot.label if slot.movable else f"{slot.label} (fixed)"
            if hidden_now:
                label = f"{label} (hidden)"
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
                    icons.EYE_OFF if hidden_now else icons.EYE,
                )
                if imgui.is_mouse_clicked(0) and badge_hit:
                    if hidden_now:
                        edit.hidden.discard(slot.id)
                    else:
                        edit.hidden.add(slot.id)
                    toggled = True
            if inside and slot.movable and not badge_hit:
                hovered = slot.id
    if imgui.is_mouse_clicked(0) and hovered:
        edit.dragging = hovered
    if edit.dragging and not imgui.is_mouse_down(0):
        _commit(app, ctx, columns, edit, mouse)
        edit.dragging = ""
    if toggled:
        # Immediate, like a drag's own commit: there is no separate "done"
        # gesture in this editor, so a hide toggle with no drag afterward
        # must not be lost when the panel closes.
        _persist(app, ctx, edit, {c.id: [s.id for s in c.live(ctx)] for c in columns.values()})
    _banner(
        ctx,
        "Drag a pane onto another to reorder it, or press the eye to hide "
        "one. Shift+W leaves; Settings > Advanced resets.",
    )


def _banner(ctx: Any, text: str) -> None:
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


def _commit(app: Any, ctx: Any, columns: Any, edit: EditState, mouse: Any) -> None:
    """Land a drag: work out the column and index, and record the arrangement."""

    from . import layout as layout_mod

    rects = layout_mod.FRAME_PANES
    for column in columns.values():
        live = [
            (slot.id, rects[slot.id])
            for slot in column.live(ctx)
            if slot.id in rects
        ]
        if not live:
            continue
        x = live[0][1][0]
        width = live[0][1][2]
        if not (x <= mouse.x < x + width):
            continue
        index = drop_index(live, mouse.y)
        order = moved([slot for slot, _rect in live], edit.dragging, index)
        arrangement = {
            other.id: [
                slot.id
                for slot in other.live(ctx)
            ]
            for other in columns.values()
        }
        arrangement[column.id] = order
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
