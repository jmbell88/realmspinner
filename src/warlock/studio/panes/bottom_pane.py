"""The pane that replaced the flat 24 dp status bar at the foot of the window.

T0 of the Familiar programme: the per-item status readouts that used to live
in ``status_bar.draw`` moved to a right-aligned group in the top menu bar
(``menus.py``, built off ``status_bar.items``), and this pane is what is left
at the bottom -- one collapsed row announcing that Familiar is not
installed. Nothing here drives expansion yet; the growth arithmetic
(:func:`max_height`) exists so a later tranche that gives Familiar a reply
to show can grow this pane into it without inventing a second geometry system
under time pressure.

No drag handle and no keyboard shortcut in T0: it had nothing to resize into,
so sizing was content-driven only, same as every other pane with no
affordance for a state it could not reach. 2026-09-16 gave it one: once
expanded, a handle above the pane lets a user drag its height, persisted
through :func:`familiar_ui.pane_height`/``set_pane_height`` -- see
:func:`draw` and :func:`reserve`.
"""

from __future__ import annotations

import time
from typing import Any

#: Module-level ``(state, timestamp)`` cache for :func:`familiar_state` --
#: the app polls it every frame from the frame thread, and re-stat'ing three
#: ``FAMILIAR_MODELS`` rows every frame is wasted work for a value that only
#: changes when a Settings -> Models download finishes. 2s mirrors the
#: "cheap poll, not a push" allowance the Familiar programme's T0 plan gave
#: this row; nothing invalidates it early, so a fresh download can take up to
#: 2s to be reflected here.
_familiar_state_cache: tuple[str, float] | None = None

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
        from ..modes.muse.ui import brief as muse_brief
        from ..modes.muse.ui.panes import player as muse_player

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

    Collapsed (:data:`COLLAPSED_H`) unless T5's Familiar UI state
    (``ctx.state.familiar``) exists and is expanded, in which case this
    returns :func:`max_height`'s clamp of :data:`familiar_ui.EXPANDED_H`
    against the current viewport -- the growth every bottom-anchored overlay
    (``overlay.fps_meter``/``progress_card``, ``widgets.toasts``, ``tour``'s
    card) already subtracts through this same function, per T0's own plan.

    ``getattr``-guarded rather than a plain ``ctx.state.familiar`` read: a
    fake ``ctx`` built for a pure-arithmetic test (this module's own test
    file builds ``SimpleNamespace(state=SimpleNamespace(mode=mode))``, with
    no ``familiar`` attribute at all) must still read as collapsed rather
    than raise.
    """

    ui = getattr(getattr(ctx, "state", None), "familiar", None)
    if ui is None or not getattr(ui, "expanded", False):
        return COLLAPSED_H

    from imgui_bundle import imgui

    from ..assistant import ui as familiar_ui

    viewport = imgui.get_main_viewport()
    ceiling = max_height(viewport.work_size.y, _mode_chrome(ctx))
    return min(familiar_ui.pane_height(ctx), ceiling)


def reserve(ctx: Any) -> float:
    """Vertical design px the shell must leave above the pane -- :func:`height`
    plus the drag handle's own hit-zone once expanded (see :func:`draw`).

    ``height`` alone is what :func:`draw` sizes the pane's own child to; a
    caller reserving space *above* this pane (``main.py``'s own content
    child, the toast overlay's ``bottom_offset``) needs the handle counted
    too, or it is drawn into space nobody left for it and pushes the pane's
    bottom edge past the window by exactly the handle's grip width -- the
    same "a floor that does not cover its own content clips something" defect
    ``inker_picker.PICKER_FLOOR``'s own docstring already names once.
    """
    from .. import layout as layout_mod

    h = height(ctx)
    if h > COLLAPSED_H + 0.5:
        return h + layout_mod.GRIP
    return h


def familiar_state(config: Any) -> str:
    """"missing" if a row from ``models.FAMILIAR_MODELS`` is absent from
    disk, "idle" once every row is present.

    Same idiom ``doctor._familiar_checks`` uses (``fetch.present`` over each
    row) rather than a second presence test -- before this function existed
    the bottom pane and the ✦ menu both hardcoded "isn't installed" even once
    a real download had completed, because neither one asked.
    """
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


def draw(ctx: Any) -> None:
    """Draw the bottom pane: the one collapsed row when idle, or that row
    plus T5's conversation body once the user has expanded it."""

    from imgui_bundle import imgui

    from .. import controls, fonts, theme, tokens
    from .. import layout as layout_mod
    from .. import state as state_mod
    from ..assistant import ui as familiar_ui
    from ..modes.settings.ui.panes import app_settings

    pad_x = tokens.sp(tokens.SP_2)
    row_h = tokens.sp(height(ctx))
    expanded = row_h > tokens.sp(COLLAPSED_H) + 0.5
    if expanded:
        # Drawn *above* the pane it resizes, since this pane is anchored to
        # the window's bottom edge -- reserve()'s own docstring is what makes
        # sure the shell left room for this handle in the first place.
        width = imgui.get_content_region_avail().x
        drag = layout_mod.splitter("familiar-height", vertical=False, length=width)
        if drag:
            # Dragging up is a negative delta and must *grow* the pane, the
            # opposite sign from Inker's own canvas/timeline handle (main.py)
            # -- that one shrinks the strip below it as it drags down because
            # the strip is what the share names, while this pane's own
            # height is what a drag here names directly.
            familiar_ui.set_pane_height(ctx, familiar_ui.pane_height(ctx) - drag)
    # Only the collapsed row centres its text vertically in the reserved
    # height -- an expanded pane has its own rows (transcript, input) to lay
    # out top-down, and centring *those* against the whole grown height would
    # shove the transcript into empty space at the top instead.
    with fonts.small(imgui):
        line = imgui.get_text_line_height()
        top_pad = tokens.sp(tokens.SP_2) if expanded else max((row_h - line) * 0.5, 0.0)
        imgui.push_style_var(imgui.StyleVar_.window_padding.value, (pad_x, top_pad))
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
            if familiar_state(ctx.svc.config) == "missing":
                imgui.text_colored(
                    imgui.ImVec4(*theme.rgba(theme.MUTED)),
                    "✦ Familiar isn't installed —",
                )
                imgui.same_line()
                if controls.small_button("Install…##bottom-pane/familiar-install"):
                    state_mod.set_mode(ctx.state, "settings")
                    ctx.state.preview[app_settings.CATEGORY_SLOT] = "models"
            else:
                ui = familiar_ui.ensure(ctx)
                label = "▾ ✦ Familiar" if ui.expanded else "▸ ✦ Familiar"
                if controls.small_button(f"{label}##bottom-pane/familiar-toggle"):
                    ui.expanded = not ui.expanded
                if ui.expanded:
                    familiar_ui.draw_expanded(ctx)
        imgui.end_child()
