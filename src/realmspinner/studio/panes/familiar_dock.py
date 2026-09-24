"""Familiar's full-height right dock -- replaced the bottom pane 2026-09-23,
then given a fixed proportional width the same day (the shell's proportional
rail/left/canvas/right/dock split).

The bottom pane (``bottom_pane.py``, now deleted) sat below every mode's
columns, capped at a quarter of the window with a Muse-specific floor. Once
the per-item status readouts moved to the menu bar it held only Familiar, so
it was one document-wide strip spent on one control. This module gives
Familiar a shell-level dock on the far right instead, from the menu bar to
the window's bottom edge, outside every mode's columns -- mirroring
``rail.py`` on the left, down to the tick/width/draw split and the
closed-strip idiom.

**The dock's width is no longer its own fact.** It used to be fitted (``fit``)
independently of the side columns' own fit (``layout.fit_widths``), against
the same room -- and nothing kept the two sums from exceeding the window, so
the dock could overflow underneath the columns. Now ``layout.proportions`` is
the one function that divides the room among the rail, the two sidebars, the
centre and the dock together, so this module's only job is to ease *how open*
the dock is (0..1) and publish that fraction for ``layout.measure`` to read
before the columns are sized -- the same ordering ``rail.tick`` already
followed for the rail's own column.
"""

from __future__ import annotations

from typing import Any

from .. import rail

#: Same width as the collapsed rail (``rail.RAIL_W``) -- one 44 dp hit target,
#: mirrored onto the opposite edge of the window. Also
#: ``layout.DOCK_FLOOR_CLOSED``'s own figure: the dock's closed width and the
#: rail's collapsed width are the same design decision on opposite edges.
STRIP_W = rail.RAIL_W

_T_KEY = "layout/familiar-open"

#: The eased open-fraction, 0 (closed) to 1 (open) -- module state for the
#: same reason ``rail._WIDTH`` is: drawn in one place (:func:`draw`) but
#: measured in another (``layout.measure``, before the dock itself exists
#: this frame).
_T = [0.0]

#: This frame's dock width, physical px -- kept alongside ``_T`` so
#: :func:`width`/:func:`reserve` do not have to re-run ``layout.proportions``
#: (and re-read the viewport) every time something asks how wide the dock is.
_WIDTH = [0.0]

#: Module-level ``(state, timestamp)`` cache for :func:`familiar_state`,
#: carried over unchanged from the bottom pane -- see its own comment there
#: (now here): the app polls it every frame and re-stat'ing three
#: ``FAMILIAR_MODELS`` rows every frame is wasted work for a value that only
#: changes when a Settings -> Models download finishes.
_familiar_state_cache: tuple[str, float] | None = None


def width() -> float:
    """This frame's dock width in physical px."""
    return _WIDTH[0]


def reserve() -> float:
    """Physical px the dock takes this frame -- what the shell and every
    bottom/right-anchored overlay must leave clear.

    No grip is added any more: the dock no longer drags (its own width is a
    fixed share of the room, from ``layout.proportions``), so nothing beside
    it needs the extra hit-zone width ``layout.GRIP`` used to reserve.
    """
    return width()


def imgui_ready() -> bool:
    """Whether an imgui context exists -- pure-arithmetic tests build a fake
    ``ctx`` with no window at all, the same guard ``rail.expanded_fits`` uses."""
    from imgui_bundle import imgui

    return imgui.get_current_context() is not None


def tick(ctx: Any) -> float:
    """Settle the dock's open-fraction and width for this frame. -> physical px.

    Called once per frame right after ``rail.tick``, before
    ``layout.tick``/``layout.measure`` -- the same ordering reason
    ``rail.tick``'s own docstring gives: how wide the columns can be is a
    fact about how open the dock is, and a dock measured afterwards would
    leave the columns disagreeing with the window by exactly its own width
    for one frame every time it was toggled.
    """
    from .. import layout as layout_mod
    from .. import motion, tokens

    ui = getattr(getattr(ctx, "state", None), "familiar", None)
    open_ = bool(getattr(ui, "expanded", False))
    t = motion.value(_T_KEY, 1.0 if open_ else 0.0, duration=tokens.DUR_BASE)
    _T[0] = t
    layout_mod.FAMILIAR_OPEN_T = t

    if imgui_ready():
        from imgui_bundle import imgui

        style = imgui.get_style()
        room = imgui.get_main_viewport().work_size[0] - style.window_padding.x * 2
        spacing = style.item_spacing.x
    else:
        room = spacing = 0.0
    dock = (
        layout_mod.proportions(room, t, spacing, rail=layout_mod.RAIL_RESERVED)[4]
        if room
        else STRIP_W * tokens.SCALE
    )
    _WIDTH[0] = dock
    layout_mod.DOCK_RESERVED = dock
    return dock


def familiar_state(config: Any) -> str:
    """"missing" if a row from ``models.FAMILIAR_MODELS`` is absent from
    disk, "idle" once every row is present. Carried over unchanged from the
    bottom pane -- see :mod:`realmspinner.models` for the row list."""
    import time

    global _familiar_state_cache
    now = time.monotonic()
    if _familiar_state_cache is not None and now - _familiar_state_cache[1] <= 2.0:
        return _familiar_state_cache[0]

    from ... import fetch, models

    state = (
        "idle"
        if all(
            fetch.present(config, "familiar", spec)
            for spec in models.FAMILIAR_MODELS.values()
        )
        else "missing"
    )
    _familiar_state_cache = (state, now)
    return state


def _sigil_button(
    *, muted: bool, active: bool, tooltip: str, box: float, height: float
) -> bool:
    """The ✦ toggle, drawn the way a rail item is. -> whether it was clicked.

    A rail item rather than ``controls.small_button``: the button sized
    itself to the glyph and sat flush left in the 44 dp strip, so the star
    was visibly off-centre in the column it is the only thing in -- and its
    hover fill did not match the rail's on the opposite edge. The hit box is
    the strip's full content width, the hover fill is the rail's (same inset,
    radius and colour), and the glyph is centred on its ink
    (``fonts.centred_glyph_pos``).
    """
    from imgui_bundle import imgui

    from .. import fonts, motion, probe, theme, tokens

    origin = imgui.get_cursor_screen_pos()
    clicked = imgui.invisible_button("##familiar-dock/toggle", (box, height))
    probe.record(label="✦##familiar-dock/toggle", kind="button", tooltip=tooltip)
    hovered = imgui.is_item_hovered()
    focused = imgui.is_item_focused() and imgui.get_io().nav_visible
    lit = motion.value(
        "familiar-dock/toggle/hover", 1.0 if (hovered or focused) else 0.0,
        duration=tokens.DUR_FAST,
    )
    draw = imgui.get_window_draw_list()
    inset = tokens.sp(rail.PILL_INSET)
    if lit > 0.0:
        draw.add_rect_filled(
            (origin.x + inset, origin.y + inset),
            (origin.x + box - inset, origin.y + height - inset),
            imgui.get_color_u32(theme.rgba(theme.ELEV_1, lit)),
            tokens.sp(tokens.RADIUS_M),
        )
    if muted:
        colour = theme.rgba(theme.MUTED, 0.65)
    else:
        colour = theme.rgba(theme.TEXT, 1.0 if active or lit > 0.0 else 0.85)
    draw.add_text(
        fonts.centred_glyph_pos(imgui, "✦", origin.x + box * 0.5, origin.y + height * 0.5),
        imgui.get_color_u32(colour),
        "✦",
    )
    if hovered:
        imgui.set_tooltip(tooltip)
        imgui.set_mouse_cursor(imgui.MouseCursor_.hand.value)
    return clicked


def draw(ctx: Any) -> None:
    """Draw the dock: the closed 44 dp strip, or the open dock with its
    header and T5's conversation body."""
    from imgui_bundle import imgui

    from .. import fonts, icons, theme, tokens, widgets
    from .. import state as state_mod
    from ..assistant import ui as familiar_ui
    from ..modes.settings.ui.panes import app_settings

    w = width()
    # The eased open-fraction, not the drawn width: the dock slides from the
    # ``STRIP_W`` strip to 15% of the room, and a width comparison mid-slide
    # would flip at a threshold that moves with the window. ``ui.expanded``
    # is the fact this eases toward; reading the
    # eased fraction rather than the flag itself keeps the switch in step with
    # the same animation the width follows, instead of snapping the instant
    # the toggle is pressed while the column is still sliding.
    open_ = _T[0] > 0.5
    pad = (tokens.sp(tokens.SP_1), tokens.sp(tokens.SP_2))
    imgui.push_style_var(imgui.StyleVar_.window_padding.value, pad)
    imgui.push_style_color(imgui.Col_.child_bg.value, imgui.ImVec4(*theme.rgba(theme.PANEL)))
    # The dock itself never scrolls: open, its body pins the input to the
    # bottom edge and scrolls only the transcript (``draw_expanded``), so a
    # dock-level scrollbar would only ever be a layout bug made scrollable.
    no_scroll = (
        imgui.WindowFlags_.no_scrollbar.value | imgui.WindowFlags_.no_scroll_with_mouse.value
    )
    visible = imgui.begin_child(
        "##familiar-dock", (w, 0), imgui.ChildFlags_.borders.value, no_scroll
    )
    imgui.pop_style_color()
    imgui.pop_style_var()
    if not visible:
        imgui.end_child()
        return

    # The literal ✦ -- see menus.FAMILIAR_LABEL's docstring; the
    # familiar-sigil face merged into every font (fonts.py) draws U+2726.
    # One button in both states, at the same spot, and it *toggles*: it
    # used to only open the dock, with the open header's ✦ drawn as plain
    # text, so the icon that opened the pane could not close it again.
    state = familiar_state(ctx.svc.config)
    muted = state == "missing"
    if muted:
        tip = "Familiar isn't installed — Install… in Settings → Models"
    else:
        tip = "Hide Familiar" if open_ else "Familiar"
    box_h = tokens.sp(rail.ITEM_H)
    top_y = imgui.get_cursor_pos_y()
    # The ✦'s box is symmetric about the 44 dp strip's centre, measured from
    # the dock's window edge -- whatever inset the child actually gives the
    # cursor, the box keeps the same margin on both sides.
    margin = max(imgui.get_cursor_screen_pos().x - imgui.get_window_pos().x, 0.0)
    box = tokens.sp(STRIP_W) - 2.0 * margin
    if open_:
        # Open, "✦ Familiar" is one centred heading: the ✦ toggle and the
        # title as a group, centred on the dock's content width. The ✦'s own
        # box carries blank space either side of its glyph, so the group is
        # nudged left by half of it to centre what is actually visible.
        with fonts.heading(imgui):
            title_w = imgui.calc_text_size("Familiar").x
        side = max((box - imgui.calc_text_size("✦").x) * 0.5, 0.0)
        content_w = w - 2.0 * margin
        imgui.set_cursor_pos_x(margin + max((content_w - box - title_w - side) * 0.5, 0.0))
    if _sigil_button(muted=muted, active=open_, tooltip=tip, box=box, height=box_h):
        if muted:
            state_mod.set_mode(ctx.state, "settings")
            ctx.state.preview[app_settings.CATEGORY_SLOT] = "models"
        else:
            ui = familiar_ui.ensure(ctx)
            ui.expanded = not ui.expanded
    if open_:
        # Title and ✕ vertically centred on the 44 dp sigil row beside them;
        # the title in the heading face, since it names the whole pane.
        imgui.same_line(0.0, 0.0)
        with fonts.heading(imgui):
            imgui.set_cursor_pos_y(top_y + (box_h - imgui.get_text_line_height()) * 0.5)
            imgui.text("Familiar")
        # Lucide's X, not a literal ✕: U+2715 is in none of the atlas faces,
        # so the close button drew imgui's fallback "?" (screenshot,
        # 2026-09-24). Square and flush with the content's right edge.
        side = imgui.get_frame_height()
        imgui.same_line(w - margin - side)
        imgui.set_cursor_pos_y(top_y + (box_h - side) * 0.5)
        if widgets.icon_button(icons.X, "Hide Familiar", borderless=True):
            familiar_ui.ensure(ctx).expanded = False
        widgets.divider()
        familiar_ui.draw_expanded(ctx)
    imgui.end_child()
