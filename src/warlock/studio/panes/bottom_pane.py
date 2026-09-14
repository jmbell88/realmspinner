"""The pane that replaced the flat 24 dp status bar at the foot of the window.

T0 of the Familiar programme: the per-item status readouts that used to live
in ``status_bar.draw`` moved to a right-aligned group in the top menu bar
(``menus.py``, built off ``status_bar.items``), and this pane is what is left
at the bottom -- one collapsed row announcing that Familiar is not
installed. Nothing here drives expansion yet; the growth arithmetic
(:func:`max_height`) exists so a later tranche that gives Familiar a reply
to show can grow this pane into it without inventing a second geometry system
under time pressure.

No drag handle and no keyboard shortcut: T0 has nothing to resize into, so
sizing is content-driven only, same as every other pane that has not yet
earned an affordance for a state it cannot reach.
"""

from __future__ import annotations

from typing import Any

#: The one collapsed row's height, in design pixels -- the same figure
#: ``status_bar.STATUS_H`` used to reserve. Kept as its own name because this
#: pane, not the status bar, owns the geometry now.
COLLAPSED_H = 24.0

#: The top menu bar's own height, in the same design-pixel units as
#: :data:`COLLAPSED_H` and :data:`MIN_CENTER_HEIGHT` -- see ``menus.draw``'s
#: docstring ("the 26 dp global menu bar").
MENU_BAR_H = 26.0

#: The floor a *grown* bottom pane must leave the centre content area.
#:
#: Chosen against Muse at ``main.MIN_SIZE`` (1100x700), the narrowest case:
#: Muse's own chrome (``muse_brief.BAR_H`` 270 + ``muse_player.STRIP_H`` 148 =
#: 418 dp) already takes most of a 700 dp window before the bottom pane is
#: even drawn. Growing the pane to a naive quarter of the window (175 dp)
#: would leave::
#:
#:     700 - MENU_BAR_H(26) - 418 - 175 = 81
#:
#: eighty-one design pixels for Muse's results list -- unusably thin, and the
#: discrepancy the spec for this tranche flagged (its own arithmetic landed
#: on "~80px" without naming the menu bar; adding it back in is what makes the
#: two figures agree). Capping the grown height so a 200 dp floor survives
#: instead::
#:
#:     700 - 26 - 418 - 200 = 56
#:
#: caps the pane at 56 dp when Muse is at its minimum size -- well under the
#: 175 dp quarter-window ceiling -- and leaves Muse's body the 200 dp floor
#: this constant names. 200 was chosen as comfortably more than one row of
#: Muse's results list (each result card runs two to three text lines plus
#: padding, well under 100 dp) while still being visibly a floor rather than
#: the whole budget.
MIN_CENTER_HEIGHT = 200.0


def _mode_chrome(ctx: Any) -> float:
    """Fixed vertical chrome the current mode reserves above its own body,
    beyond the menu bar -- only Muse has one large enough to matter yet."""

    mode = str(getattr(getattr(ctx, "state", None), "mode", ""))
    if mode == "muse":
        from .. import muse_brief
        from . import muse_player

        return muse_brief.BAR_H + muse_player.STRIP_H
    return 0.0


def max_height(window_h: float, mode_chrome: float = 0.0) -> float:
    """The tallest the pane may grow to in a ``window_h`` dp tall window with
    ``mode_chrome`` dp of other fixed chrome (menu bar aside) above the centre
    content.

    Never more than a quarter of the window (:data:`COLLAPSED_H` is the
    floor, for a window too short for even a quarter to reach it), and never
    so much that the centre content area would drop under
    :data:`MIN_CENTER_HEIGHT`.
    """

    quarter = 0.25 * window_h
    floor_cap = window_h - MENU_BAR_H - mode_chrome - MIN_CENTER_HEIGHT
    return max(COLLAPSED_H, min(quarter, floor_cap))


def height(ctx: Any) -> float:
    """The pane's current height, in design pixels.

    T0 has no model to grow it: the pane is always collapsed, one row tall.
    Kept as a function -- mirroring how call sites used to read
    ``status_bar.STATUS_H`` -- because every caller this replaces (``main``'s
    content-area reservation, and the overlay/toast/tour bottom anchors) calls
    it rather than a bare constant, so a later tranche that makes this return
    a grown height (bounded by :func:`max_height`) touches no call site.
    """

    return COLLAPSED_H


def draw(ctx: Any) -> None:
    """Draw the collapsed row: the one line T0 shows in place of the status
    items that moved to the menu bar."""

    from imgui_bundle import imgui

    from .. import fonts, theme, tokens

    pad_x = tokens.sp(tokens.SP_2)
    # Same vertical-centring ordering as the status bar this replaces: the
    # small face is pushed before its line height is measured, which is what
    # centres one line of ``TEXT_SMALL`` text in the reserved height rather
    # than half of ``TEXT_BODY``'s larger one.
    with fonts.small(imgui):
        line = imgui.get_text_line_height()
        row_h = tokens.sp(height(ctx))
        imgui.push_style_var(
            imgui.StyleVar_.window_padding.value,
            (pad_x, max((row_h - line) * 0.5, 0.0)),
        )
        imgui.push_style_color(
            imgui.Col_.child_bg.value, imgui.ImVec4(*theme.rgba(theme.PANEL))
        )
        visible = imgui.begin_child("##bottom-pane", (0, row_h))
        imgui.pop_style_color()
        imgui.pop_style_var()
        if visible:
            # The literal ✦ -- see menus.FAMILIAR_LABEL's docstring; the
            # familiar-sigil face merged into every font (fonts.py) is what
            # draws U+2726.
            imgui.text_colored(
                imgui.ImVec4(*theme.rgba(theme.MUTED)),
                "✦ Familiar isn't installed — Install…",
            )
        imgui.end_child()
