"""Clay's tool rail: the four tools and a few shapes, beside the viewport.

picoCAD's tools sit in a slim column against the model, and so does every
modeller's. Clay's used to be a 300 px sidebar of unlabelled shape icons on the
far side of the window plus a tool pill in the header; the sidebar is gone
(``skeletons.clay``'s left column is empty and the centre widened by its width)
and its two jobs live here, drawn **inside the centre pane** to the left of the
render, the way Plotter's toolbar is drawn inside its own -- a strip the
viewport reserves rather than a docked pane, so a saved layout has nothing to
permute and nothing to lose.

**Tools first, then shapes.** Select, Move, Rotate and Scale write
``state.tool`` (the same field the keys Q G R S write -- the tooltips read their
letter off ``clay_state.TOOLS``, so a rebinding is one edit and cannot leave the
rail quoting a key that no longer works). A rule, then :data:`tools.QUICK_SHAPES`
as one-click adds, then ``+``, which opens a flyout of **every** shape with its
name beside its glyph (``tools.draw_add_menu``). The Inker flyout is not reused:
it belongs to a sibling mode, and ``tests/test_layering.py`` forbids the reach.

No (?) of its own: the flyout's heading carries the manual link, and a help
button inside a column of 32 px buttons would be a tenth thing to click by
accident.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from ..... import controls, icons, tool_palette, widgets
from .....tokens import sp
from ... import mode as clay_mode
from ... import state as clay_state
from . import tools as clay_tools

#: The rail's width in design pixels: a 32 px button and the child's padding.
WIDTH = 44.0

#: A rail button's side in design pixels.
BUTTON = 32.0

#: The flyout's popup id. Opened from, and drawn inside, the rail's own child so
#: the id stack matches (a popup only renders in the window that opened it).
FLYOUT_POPUP = "clay-rail-add"

_NO_DOC = "Open or start a document to build in."
_SAVING = "This document is being written; the shapes come back when it lands."


def width() -> float:
    """:data:`WIDTH` in physical pixels, for the viewport's reservation."""
    return float(sp(WIDTH))


def tool_tooltip(key: str, label: str, shortcut: str) -> str:
    """``"Move  (G)"`` -- a tool's name and the key that picks it."""
    del key
    return f"{label}  ({shortcut})"


def draw(ctx: Any, height: float = 0.0) -> None:
    """The rail. Called by the viewport, to the left of the render.

    ``height`` 0 takes whatever is left of the window, which is what a pane walk
    wants; the viewport passes the render's own height so the rail ends where the
    render does and the palette strip runs under both.
    """
    state = clay_mode.ensure(ctx)
    tab = state.active
    imgui.push_style_var(imgui.StyleVar_.window_padding.value, (sp(5), sp(5)))
    shown = imgui.begin_child(
        "##clay-rail", (width(), height), 0, imgui.WindowFlags_.no_scrollbar.value
    )
    try:
        if shown:
            _body(ctx, state, tab)
    finally:
        imgui.end_child()
        imgui.pop_style_var()


def _body(ctx: Any, state: Any, tab: Any) -> None:
    side = sp(BUTTON)
    for key, label, shortcut in clay_state.TOOLS:
        glyph = clay_tools.TOOL_ICONS.get(key) or label[:1]
        if controls.button(
            f"{glyph}##clay-rail/tool/{key}",
            (side, side),
            selected=state.tool == key,
            tooltip=tool_tooltip(key, label, shortcut),
        ):
            state.tool = key
    widgets.divider()
    blocked = tab is None or bool(tab.saving)
    reason = _NO_DOC if tab is None else _SAVING
    for name in clay_tools.QUICK_SHAPES:
        glyph = tool_palette.PRIMITIVE_ICONS.get(name, icons.BOX)
        if controls.button(
            f"{glyph}##clay-rail/add/{name}",
            (side, side),
            enabled=not blocked,
            reason=reason,
            tooltip=f"Add a {clay_tools.display_name(name)}",
        ):
            clay_tools.add_primitive(ctx, tab.doc, name)
            state.generator = name
    if controls.button(
        f"{icons.PLUS}##clay-rail/more",
        (side, side),
        tooltip="Every shape, by name",
    ):
        imgui.open_popup(FLYOUT_POPUP)
    # Beside the button, not under the pointer: a flyout opened from a column
    # belongs to the column's right edge. Asked for on every frame the popup is
    # open and never otherwise -- imgui re-places a popup that has just come out
    # of its hidden sizing frame at the pointer, which would undo a position given
    # only on the click's frame, while ``set_next_window_pos`` left pending by a
    # popup that is *not* open would land on whichever window is begun next.
    low, high = imgui.get_item_rect_min(), imgui.get_item_rect_max()
    if imgui.is_popup_open(FLYOUT_POPUP):
        imgui.set_next_window_pos((high.x + sp(4), low.y))
    with controls.menu_popup(FLYOUT_POPUP) as opened:
        if opened:
            clay_tools.draw_add_menu(ctx, state, tab)
