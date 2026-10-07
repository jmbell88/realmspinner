"""A palette swatch: how one slot is drawn, and what a click on it does.

Lifted out of ``props.py`` when the swatch row left the Material tab for the
palette strip under the viewport (``palette_strip.py``). Two callers want the
same three things -- the colour a slot is drawn in, the button that draws it
and the rule for what pressing it means -- and a second copy of the rule is
where "click paints, Ctrl+click only picks" would have come apart between them.
``props`` re-exports the names it used to define, because the test files that
drive them reach for ``clay_props._swatch`` and friends.

No pane here: this module draws a button and decides a click, it does not lay
anything out.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from ..... import theme
from .....tokens import sp
from ... import ops as clay_ops

#: A palette swatch's side, in design px -- the size Inker's own palette uses.
SWATCH = 22.0


def _swatch_colour(material: Any) -> tuple[float, float, float, float]:
    """What a palette swatch is filled with.

    A textured slot's colour factor is white (``add_texture`` moves the colour
    into the picture), so the factor would draw every textured slot the same;
    its first texel, already sRGB bytes, tells them apart.
    """
    image = material.base_color
    if image is not None and len(image[2]) >= 4:
        r, g, b = image[2][0], image[2][1], image[2][2]
        return (r / 255.0, g / 255.0, b / 255.0, 1.0)
    return tuple(float(c) for c in material.base_color_factor)  # type: ignore[return-value]


def _swatch(
    label: str,
    colour: tuple[float, float, float, float],
    side: float,
    *,
    selected: bool,
    textured: bool,
    tooltip: str,
) -> bool:
    """One palette swatch button. -> whether it was clicked.

    The one place the palette touches imgui's colour button, so a test can stand
    in a press without a mouse. The slot the object defaults to is outlined in
    the accent; a textured slot carries a corner notch so it reads as a picture
    and not as a flat colour.
    """
    if selected:
        imgui.push_style_color(imgui.Col_.border.value, imgui.ImVec4(*theme.rgba(theme.ACCENT)))
        imgui.push_style_var(imgui.StyleVar_.frame_border_size.value, sp(2.0))
    clicked = imgui.color_button(label, imgui.ImVec4(*colour), 0, (side, side))
    if selected:
        imgui.pop_style_var()
        imgui.pop_style_color()
    if textured:
        low = imgui.get_item_rect_max()
        notch = max(sp(5.0), side * 0.3)
        draw = imgui.get_window_draw_list()
        # Dark under light, so the notch reads on a pale texel and a dark one.
        for size, tint in ((notch, theme.BG), (notch * 0.6, theme.TEXT)):
            draw.add_triangle_filled(
                imgui.ImVec2(low.x, low.y),
                imgui.ImVec2(low.x - size, low.y),
                imgui.ImVec2(low.x, low.y - size),
                imgui.get_color_u32(imgui.ImVec4(*theme.rgba(tint))),
            )
    if imgui.is_item_hovered():
        imgui.set_tooltip(tooltip)
    return bool(clicked)


def single_selected(doc: Any) -> Any:
    """The one selected object, or None.

    One rather than the first of many: a panel that silently edited whichever
    object happened to sort first under a multi-selection is worse than one that
    says it cannot. ``props._selected`` is this function under its old name.
    """
    if len(doc.selection) != 1:
        return None
    try:
        return doc.by_uid(next(iter(doc.selection)))
    except KeyError:
        return None


def _pick_slot(ctx: Any, doc: Any, obj: Any, index: int) -> None:
    """What a click on palette swatch *index* does.

    Object mode keeps the old combo's behaviour exactly: the object's default
    slot **and every face** go to the slot (the 2026-10-03 audit's clay-17).
    In face mode with faces selected the click paints those faces through the
    ``assign-material`` op and leaves the object's default slot alone; Ctrl
    (or no faces selected) only makes it the slot the Material tab edits, the
    one way to reach another slot's colour and texture from a selection.

    Ctrl does the same in object mode now. It used to be ignored there, which
    left the strip with a swatch that could only ever repaint: reaching a slot's
    colour to edit it meant repainting the object with it first, then undoing.

    Vertex and edge mode only pick the slot, like a face click with nothing
    selected (the 2026-10-07 audit's clay-73): they used to take the object-mode
    branch and repaint every face of the object, though the selection in front of
    the user was a few points or edges and no face was named. Painting faces is
    face mode's job.
    """
    mode = doc.element_mode
    if mode in ("vertex", "edge"):
        # Ctrl is moot: there is no face named to paint, so the press only picks.
        if index != obj.material:
            doc.set_props(obj.uid, material=index)
        return
    ctrl = _ctrl_held()
    if mode == "object" and not ctrl:
        doc.repaint_object(obj.uid, index)
        return
    op = clay_ops.get("assign-material")
    if mode != "face" or ctrl or not op.enabled(doc):
        if index != obj.material:
            doc.set_props(obj.uid, material=index)
        return
    clay_ops.run(ctx, doc, op, index=index)


def _ctrl_held() -> bool:
    """Whether Ctrl is down this frame. One door, so a headless test can stand in."""
    return bool(imgui.get_io().key_ctrl)


def click_slot(ctx: Any, doc: Any, index: int) -> None:
    """A press on palette slot *index* from the strip, for whatever is selected.

    The strip is the document's, not one object's, so it cannot assume the one
    selected object :func:`_pick_slot` wants. With exactly one, that rule
    applies unchanged. Without one:

    * **face mode** paints every selected face, across however many objects --
      the op is what the Edit menu's Assign Material runs, so one click is one
      undo step by the op's own contract;
    * **object mode** repaints each selected object, folded into one step so a
      Ctrl+Z puts the whole selection back rather than one object at a time;
    * **Ctrl, or vertex and edge mode** have no single slot to retarget, so the
      press does nothing -- the tooltip says to select one object.
    """
    obj = single_selected(doc)
    if obj is not None:
        _pick_slot(ctx, doc, obj, index)
        return
    if _ctrl_held():
        return
    if doc.element_mode == "face":
        op = clay_ops.get("assign-material")
        if op.enabled(doc):
            clay_ops.run(ctx, doc, op, index=index)
        return
    if doc.element_mode == "object" and doc.selection:
        mark = doc.history.mark()
        try:
            for uid in sorted(doc.selection):
                doc.repaint_object(uid, index)
        finally:
            doc.history.collapse_since(mark)


def current_slot(doc: Any) -> int | None:
    """The slot the Material tab is editing, or ``None`` with no single object.

    What the strip outlines. It is the one selected object's default slot -- the
    same index ``props._material`` reads -- so the outline and the fields below
    it name one slot.
    """
    obj = single_selected(doc)
    if obj is None or not doc.materials:
        return None
    return min(max(int(obj.material), 0), len(doc.materials) - 1)
