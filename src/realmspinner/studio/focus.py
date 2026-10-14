"""A keyboard focus ring, scoped to the panes that declare one (UX.md Phase 3).

**Not imgui's nav.** ``setup_window`` leaves ``nav_enable_keyboard`` set so a
focused control *draws* as focused, but nothing here relies on nav to move or
activate anything: with it as the only mechanism a focused button drew as
focused and did nothing, and ``dialogs`` reads Esc and Enter itself for the
same reason. What this is instead is the Home-tiles pattern generalised -- a
pane states an order over its own controls, one of them is the cursor, Tab and
Shift-Tab move it, and :func:`widgets.ring` is the one focus visual the whole
app uses.

The order is **the order the controls are drawn in**, recorded as the frame
builds them rather than written out beside the pane: a hand-kept list over a
hand-written column of widgets is two orderings, and they drift the first time
a control is inserted -- which is exactly the argument ``landing.rows`` and
``palette.py`` already make about the two other indices of the same thing.

Two rules keep this from fighting imgui rather than riding on it.

**Focus is *asked for* only on the frame the cursor moved.** ``imgui``'s
``set_keyboard_focus_here`` is a command, not a state: issuing it every frame
would re-focus the control on every one of them, which resets a text cursor
mid-word and makes a slider undraggable. So the request is one-shot and the
ring is what persists.

**And the ring is drawn from the pane's own order rather than from imgui's
idea of focus**, because half these controls are hand-drawn (the segmented
output switch, the count radios) and imgui has no idea about them at all.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from imgui_bundle import imgui

from . import theme, widgets


def _frame_number() -> int | None:
    """imgui's frame count, or ``None`` without a live context -- asked of
    ``get_current_context`` first because a frame call with none can crash
    rather than raise."""
    try:
        if imgui.get_current_context() is None:
            return None
        return int(imgui.get_frame_count())
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


def _first_this_frame(state: Any, what: str, pane: str) -> bool:
    """Whether ``what`` (``"pump"`` or ``"begin"``) is the first call for
    ``pane`` in this frame.

    The 2026-10-14 audit (finding shell-boot-01): Muse draws a brief bar and a
    recipe column in one frame, and each pumped and began its own ring from the
    same Tab press and the one global ``focus_moved`` flag, so one press moved
    two rings and issued two ``set_keyboard_focus_here`` requests, the later
    winning. Two panes that name the *same* ring key (as Create's do) now share
    one cursor: only the first to pump reads the press, and only the first to
    begin clears the order, so the second pane's items append to it. Keyed by
    imgui's own frame count; with no live context, or a state that keeps no
    stamps (a unit-test stub), every call counts as the first -- the old
    behaviour."""
    stamps = getattr(state, "focus_stamps", None)
    frame = _frame_number()
    if stamps is None or frame is None:
        return True
    key = f"{what}:{pane}"
    if stamps.get(key) == frame:
        return False
    stamps[key] = frame
    return True


def pump(state: Any, pane: str) -> None:
    """Answer Tab / Shift-Tab for ``pane``. Called *before* :func:`begin`.

    Before, because :func:`begin` clears the order and :func:`move` needs the
    one the last frame recorded -- which is also the only order that describes
    what is actually on screen right now.

    Read from imgui rather than from the pygame event loop, and that is not a
    style choice: ``App._shortcut`` is skipped entirely while a text field has
    the keyboard (``io.want_text_input``), and tabbing *out of the prompt box*
    is the single most important thing this has to do.

    Suppressed while any popup is up, for the reason ``App._modal_open`` exists:
    a modal owns the keyboard, and a Tab that also moved the ring behind it
    would leave the cursor somewhere the user never saw it go.
    """
    if not _first_this_frame(state, "pump", pane):
        return
    if imgui.is_popup_open("", imgui.PopupFlags_.any_popup.value):
        return
    if not imgui.is_key_pressed(imgui.Key.tab):
        return
    back = imgui.is_key_down(imgui.Key.left_shift) or imgui.is_key_down(imgui.Key.right_shift)
    move(state, pane, -1 if back else 1)


def begin(state: Any, pane: str) -> None:
    """Start recording ``pane``'s tab order for this frame.

    Called once, before the first :func:`item`. The list is rebuilt every frame
    because what is on screen changes -- a fold closes, a slider appears with
    the LoRA that justifies it -- and a cursor over controls that are not there
    is a Tab key that appears to do nothing.

    A second pane that shares ``pane``'s key in the same frame keeps what the
    first recorded and appends to it -- see :func:`_first_this_frame`.
    """
    if not _first_this_frame(state, "begin", pane):
        return
    state.focus_order[pane] = []


@contextmanager
def item(state: Any, pane: str, key: str) -> Iterator[bool]:
    """Wrap one control. -> whether it is the one the cursor is on.

    ``key`` is a stable name, never an index: the index of a control is a
    function of what else happens to be visible, so a remembered one would
    silently become a different control when a section is collapsed.
    """
    state.focus_order.setdefault(pane, []).append(key)
    focused = state.focus_key.get(pane) == key
    if focused and state.focus_moved:
        # Consumed here rather than at the end of the frame: exactly one
        # control is the target of a move, and leaving the flag set would have
        # every later control in the order ask for focus too -- the last one
        # drawn winning, which is not the one the user tabbed to.
        state.focus_moved = False
        imgui.set_keyboard_focus_here()
    try:
        yield focused
    finally:
        if focused:
            widgets.ring(
                imgui.get_item_rect_min(),
                imgui.get_item_rect_max(),
                theme.ACCENT,
                0.8,
                thick=1.5,
            )


def move(state: Any, pane: str, delta: int) -> bool:
    """Tab (or Shift-Tab). -> whether there was anywhere to go.

    Wraps, as Home's tiles do and unlike the library's arrow keys: a form is a
    fixed ring of controls, where a two-hundred-row list is not.

    A cursor whose control is no longer on screen restarts at the top rather
    than being clamped: it was somewhere that no longer exists, and the nearest
    surviving position is a guess, where the first control is an answer.
    """
    order = state.focus_order.get(pane) or []
    if not order:
        return False
    current = state.focus_key.get(pane)
    index = order.index(current) + delta if current in order else 0
    state.focus_key[pane] = order[index % len(order)]
    state.focus_moved = True
    return True


def activate(state: Any, pane: str) -> str | None:
    """Which control Enter would act on, or ``None``.

    A getter rather than a dispatcher: what "activate" *means* is the pane's
    business -- a button is pressed, a combo is opened, a text field is already
    typing into -- and a table of callbacks here would be a second place that
    knows what a pane's controls are.
    """
    return state.focus_key.get(pane)
