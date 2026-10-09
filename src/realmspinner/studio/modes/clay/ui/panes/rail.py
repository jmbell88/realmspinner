"""Clay's tool rail: the four tools and a few shapes, beside the viewport.

picoCAD's tools sit in a slim column against the model, and so does every
modeller's. Clay's used to be a 300 px sidebar of unlabelled shape icons on the
far side of the window plus a tool pill in the header; the sidebar is gone
(``skeletons.clay``'s left column is empty and the centre widened by its width)
and its two jobs live here, drawn **inside the centre pane** to the left of the
render, the way Plotter's toolbar is drawn inside its own -- a strip the
viewport reserves rather than a docked pane, so a saved layout has nothing to
permute and nothing to lose.

**Every button is named.** Each cell is the glyph over a 10 px word -- Select,
Move, Turn, Resize -- because an icon column was a column of guesses, and the
tooltip is where the key lives ("Turn  (R)"). The words are plain: the stored
tool names (``rotate``, ``scale``) and the keys are unchanged, and
:data:`TOOL_WORDS` is the one table that says what each is called on the rail.
The vendored Lucide subset has no cursor and no mode glyphs, so the four glyphs
are the ones ``tools.TOOL_ICONS`` already chose.

**Tools first, then shapes.** Select, Move, Turn and Resize write
``state.tool`` (the same field the keys Q G R S write -- the tooltips read their
letter off ``clay_state.TOOLS``, so a rebinding is one edit and cannot leave the
rail quoting a key that no longer works). A rule, then :data:`tools.QUICK_SHAPES`
as one-click adds, then ``+``, which opens a flyout of **every** shape with its
name beside its glyph (``tools.draw_add_menu``). The Inker flyout is not reused:
it belongs to a sibling mode, and ``tests/test_layering.py`` forbids the reach.

**The quick shapes give way to the ``+`` when the window is short.** Named cells
are 46 px tall, so the whole stack is about 490 and the viewport under the
resize floor leaves fewer than that; a column that clips its last button is a
button nobody can find. :func:`quick_shapes_fit` asks the height the child was
actually given, and when the four do not fit they are not drawn -- the flyout
lists them with every other shape, and the ``+`` is labelled "Add" instead of
"More" to say so. All or none, never two of four: a rail whose contents change
by one button as the window is dragged is a rail nobody can build a habit on.

No (?) of its own: the flyout's heading carries the manual link, and a help
button inside a column of buttons would be one more thing to click by accident.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from ..... import anchors, controls, fonts, icons, theme, tool_palette, widgets
from .....tokens import sp
from ... import mode as clay_mode
from ... import state as clay_state
from . import tools as clay_tools

#: The rail's width in design pixels: a 46 px cell and the child's padding.
WIDTH = 56.0

#: The child's own padding on every side, in design pixels.
PAD = 5.0

#: A rail cell's width and height in design pixels: a 32 px glyph square and the
#: word under it.
CELL_W = WIDTH - 2.0 * PAD
CELL_H = 46.0

#: The word's size under its glyph, in design pixels.
LABEL_PX = 10.0

#: The flyout's popup id. Opened from, and drawn inside, the rail's own child so
#: the id stack matches (a popup only renders in the window that opened it).
FLYOUT_POPUP = "clay-rail-add"

#: What each tool is called on the rail. The stored key and the letter are the
#: tool's own (``clay_state.TOOLS``); only the word differs, because "Rotate" and
#: "Scale" are the verbs a package uses and "Turn" and "Resize" are what a person
#: who has never opened one calls it.
TOOL_WORDS = {"select": "Select", "move": "Move", "rotate": "Turn", "scale": "Resize"}

#: What each quick shape is called under its glyph: the flyout's names, except
#: "UV Sphere", which does not fit a 46 px cell and is a sphere to anyone who
#: has not used a modeller.
SHAPE_WORDS = {"box": "Box", "cylinder": "Cylinder", "uv_sphere": "Sphere", "cone": "Cone"}

_NO_DOC = "Open or start a document to build in."
_SAVING = "This document is being written; the shapes come back when it lands."


def width() -> float:
    """:data:`WIDTH` in physical pixels, for the viewport's reservation."""
    return float(sp(WIDTH))


def tool_tooltip(key: str, label: str, shortcut: str) -> str:
    """``"Move  (G)"`` -- a tool's name and the key that picks it."""
    del key
    return f"{label}  ({shortcut})"


def stack_height(shapes: int, cell: float, spacing: float) -> float:
    """How tall the rail's contents are with *shapes* quick shapes, in whatever
    unit *cell* and *spacing* are in. Pure.

    Every tool, every quick shape and the ``+`` are one cell each with the style's
    spacing between them, and the rule between the two groups costs its 1 px and
    a spacing on each side.
    """
    rows = len(clay_state.TOOLS) + shapes + 1
    return rows * cell + (rows - 1) * spacing + 1.0 + 2.0 * spacing


def quick_shapes_fit(avail: float, cell: float, spacing: float) -> bool:
    """Whether the four quick shapes fit under the tools in *avail*. Pure."""
    return stack_height(len(clay_tools.QUICK_SHAPES), cell, spacing) <= avail


def draw(ctx: Any, height: float = 0.0) -> None:
    """The rail. Called by the viewport, to the left of the render.

    ``height`` 0 takes whatever is left of the window, which is what a pane walk
    wants; the viewport passes the render's own height so the rail ends where the
    render does and the hint line runs under both.
    """
    state = clay_mode.ensure(ctx)
    tab = state.active
    imgui.push_style_var(imgui.StyleVar_.window_padding.value, (sp(PAD), sp(PAD)))
    # ``always_use_window_padding``: a child with no border ignores
    # ``window_padding`` without it, so the cells sat against the rail's left edge
    # with the whole 10 px of padding on the right (the 2026-10-09 centring report).
    # The padding is what centres a 46 px cell in the 56 px column.
    shown = imgui.begin_child(
        "##clay-rail",
        (width(), height),
        imgui.ChildFlags_.always_use_window_padding.value,
        imgui.WindowFlags_.no_scrollbar.value,
    )
    try:
        if shown:
            _body(ctx, state, tab)
    finally:
        imgui.end_child()
        imgui.pop_style_var()


def _cell(
    glyph: str,
    ident: str,
    word: str,
    *,
    selected: bool = False,
    enabled: bool = True,
    reason: str = "",
    tooltip: str = "",
) -> bool:
    """One rail button: the glyph at the top of the cell, its word at the bottom.

    **One button, not a button and a caption**, so the whole cell is the hit area,
    hovers as one and carries one tooltip. The glyph is the button's own label,
    pinned to the top by ``button_text_align``; the word is drawn over the bottom
    of the same rectangle with the draw list, because a button's label is one run
    of one font and the word is another size.
    """
    imgui.push_style_var(imgui.StyleVar_.button_text_align.value, (0.5, 0.0))
    clicked = controls.button(
        f"{glyph}##{ident}",
        (sp(CELL_W), sp(CELL_H)),
        selected=selected,
        enabled=enabled,
        reason=reason,
        tooltip=tooltip,
    )
    imgui.pop_style_var()
    low, high = imgui.get_item_rect_min(), imgui.get_item_rect_max()
    colour = theme.TEXT if selected else theme.MUTED
    with fonts.push(imgui, fonts.MEDIUM, LABEL_PX):
        size = imgui.calc_text_size(word)
        pos = (
            low.x + (high.x - low.x - size.x) * 0.5,
            high.y - size.y - sp(3.0),
        )
        imgui.get_window_draw_list().add_text(
            pos, imgui.get_color_u32(theme.rgba(colour, 1.0 if enabled else 0.5)), word
        )
    return bool(clicked)


def _body(ctx: Any, state: Any, tab: Any) -> None:
    # One ring for the column: the add buttons are drawn in a loop, and the
    # tour's anchor test only sees literal keys.
    anchors.mark_window("clay/rail")
    spacing = imgui.get_style().item_spacing.y
    room = imgui.get_content_region_avail().y
    quick = clay_tools.QUICK_SHAPES if quick_shapes_fit(room, sp(CELL_H), spacing) else ()
    for key, label, shortcut in clay_state.TOOLS:
        glyph = clay_tools.TOOL_ICONS.get(key) or label[:1]
        word = TOOL_WORDS.get(key, label)
        if _cell(
            glyph,
            f"clay-rail/tool/{key}",
            word,
            selected=state.tool == key,
            tooltip=tool_tooltip(key, word, shortcut),
        ):
            state.tool = key
    widgets.divider()
    blocked = tab is None or bool(tab.saving)
    reason = _NO_DOC if tab is None else _SAVING
    for name in quick:
        glyph = tool_palette.PRIMITIVE_ICONS.get(name, icons.BOX)
        clicked = _cell(
            glyph,
            f"clay-rail/add/{name}",
            SHAPE_WORDS.get(name, clay_tools.display_name(name)),
            enabled=not blocked,
            reason=reason,
            tooltip=f"Add a {clay_tools.display_name(name)}",
        )
        # The refusing door, not ``add_primitive``: a full document would raise
        # out of this draw (the 2026-10-07 audit's clay-13 missed this button).
        # The remembered shape moves only for a shape that was placed.
        if clicked and clay_tools.place_primitive(ctx, tab.doc, name) is not None:
            state.generator = name
    if _cell(
        icons.PLUS,
        "clay-rail/more",
        "More" if quick else "Add",
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
