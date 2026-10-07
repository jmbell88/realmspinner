"""The palette strip: the document's materials in one row under the viewport.

picoCAD keeps its palette on screen beside the model, because painting is what
you do between nearly every pair of edits: pick a face, pick a colour. The row
of swatches used to live in the Properties pane's Material tab, one tab away
from the viewport and only drawn while exactly one object was selected -- so the
most frequent material action needed a tab switch first and a selection the
action did not actually need.

**The slots are the document's, and only the document's.** A new document keeps
the palette it already has; nothing here seeds sixteen colours. The strip draws
``doc.materials`` as they are, with a ``+`` that appends one (a slot is an index
every mesh's per-face ``material`` array names, so it is appended and never
inserted -- ``ClayDoc.add_material``).

**Fixed height, never wrapped.** The viewport reserves :data:`STRIP_H` for it
(the way Inker's timeline strip is reserved), so the strip cannot push the model
around as slots are added. A row that wrapped would grow with the palette and
take the height back from the render; instead the swatches scroll sideways in
their own child, with the active slot's name on the line beneath.

What a click does is :func:`.swatches.click_slot`'s rule, shared with the code
the Material tab used to hold: a plain click paints, Ctrl+click only makes the
slot the one Properties edits.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from ..... import controls, icons, widgets
from .....tokens import sp
from ... import mode as clay_mode
from ... import ops as clay_ops
from . import swatches

#: The strip's height in design pixels, reserved by the viewport below the
#: render. Fixed (see the module docstring); 52 is two lines -- a row of 22 px
#: swatches with a thin scrollbar, and the name of the slot under the pointer.
STRIP_H = 52.0

_SAVING = "This document is being written; the palette comes back when it lands."


def height() -> float:
    """:data:`STRIP_H` in physical pixels, for the viewport's reservation."""
    return float(sp(STRIP_H))


def click_hint(doc: Any) -> str:
    """What a plain click on a swatch would do right now, in a sentence.

    A pure function of the document, so the sentence the tooltip carries is
    checkable without a frame. It names the *current* behaviour rather than all
    of them: "paint the selected faces" while faces are selected, and the reason
    the swatch would do nothing while they are not.
    """
    mode = doc.element_mode
    one = swatches.single_selected(doc) is not None
    if mode == "face":
        if clay_ops.get("assign-material").enabled(doc):
            return "Click paints the selected faces. Ctrl+click edits this slot in Properties."
        return "Select faces to paint them. Ctrl+click edits this slot in Properties."
    if mode == "object":
        if not doc.selection:
            return "Select an object to repaint it with a swatch."
        return (
            "Click repaints the selected object. Ctrl+click edits this slot in Properties."
            if one
            else "Click repaints every selected object."
        )
    return "Click picks the slot Properties edits. Switch to face mode (3) to paint faces."


def slot_label(doc: Any, index: int) -> str:
    """``"2  Red"``, or ``"2"`` for an unnamed slot -- the line under the row."""
    name = doc.materials[index].name if 0 <= index < len(doc.materials) else ""
    return f"{index}  {name}" if name else f"{index}"


def draw(ctx: Any, width: float = 0.0) -> None:
    """The strip. Called by the viewport, below the render and above the hint.

    ``width`` 0 takes whatever the window has left, which is what a pane walk
    wants; the viewport passes the centre's full width so the strip runs under
    the rail as well as under the model.
    """
    state = clay_mode.ensure(ctx)
    tab = state.active
    if tab is None:
        return
    avail = imgui.get_content_region_avail().x
    full = width if width > 0 else avail
    imgui.push_style_var(imgui.StyleVar_.window_padding.value, (sp(6), sp(3)))
    flags = (
        imgui.WindowFlags_.no_scrollbar.value | imgui.WindowFlags_.no_scroll_with_mouse.value
    )
    shown = imgui.begin_child(
        "##clay-palette-strip", (full, height()), imgui.ChildFlags_.borders.value, flags
    )
    try:
        if shown:
            _body(ctx, tab)
    finally:
        imgui.end_child()
        imgui.pop_style_var()


def _body(ctx: Any, tab: Any) -> None:
    doc = tab.doc
    saving = bool(tab.saving)
    side = sp(swatches.SWATCH)
    gap = imgui.get_style().item_spacing.x
    plus_w = widgets.button_width(icons.PLUS)
    # The row's own child holds the scroll, so the label line below stays put
    # while a long palette slides under the pointer.
    row_w = max(imgui.get_content_region_avail().x - plus_w - gap, side)
    thin = sp(8)
    imgui.push_style_var(imgui.StyleVar_.scrollbar_size.value, thin)
    row_flags = imgui.WindowFlags_.horizontal_scrollbar.value
    hovered_slot: int | None = None
    clicked: int | None = None
    if imgui.begin_child("##clay-palette-row", (row_w, side + thin + sp(2)), 0, row_flags):
        current = swatches.current_slot(doc)
        hint = click_hint(doc)
        for i, entry in enumerate(doc.materials):
            if i:
                imgui.same_line()
            textured = entry.base_color is not None
            name = entry.name or f"slot {i}"
            tip = f"{i}: {name}" + (" (textured)" if textured else "") + "\n" + hint
            if saving:
                tip = _SAVING
            if swatches._swatch(
                f"##palsw{i}",
                swatches._swatch_colour(entry),
                side,
                selected=i == current,
                textured=textured,
                tooltip=tip,
            ):
                clicked = i
            if imgui.is_item_hovered():
                hovered_slot = i
        wheel = imgui.get_io().mouse_wheel
        if wheel and imgui.is_window_hovered():
            # A mouse wheel has no horizontal axis to offer; sliding the row with
            # it is the only way to reach a far slot without aiming at a 8 px bar.
            imgui.set_scroll_x(imgui.get_scroll_x() - wheel * side * 2.0)
    imgui.end_child()
    imgui.pop_style_var()
    imgui.same_line()
    if controls.small_button(
        f"{icons.PLUS}##clay-palette/add",
        enabled=not saving,
        reason=_SAVING,
        tooltip="Add a palette slot. It starts as a grey; Ctrl+click it to edit its colour.",
    ):
        doc.add_material()
    if clicked is not None and not saving:
        swatches.click_slot(ctx, doc, clicked)
    shown = hovered_slot if hovered_slot is not None else swatches.current_slot(doc)
    if shown is None:
        widgets.muted("Palette")
    else:
        widgets.muted(slot_label(doc, shown))
