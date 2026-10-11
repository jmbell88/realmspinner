"""A palette swatch: how one slot is drawn, and what a click on it does.

Lifted out of ``props.py`` when the swatch row left the Material tab for the
palette strip under the viewport, and kept when the strip itself went: the
Inspector's Colour card (``inspector.py``) draws the document's slots with
:func:`draw_slots` under its eight starter colours. The callers want the same three
things -- the colour a slot is drawn in, the button that draws it and the rule for
what pressing it means -- and a second copy of the rule is where "click paints,
Ctrl+click only picks" would have come apart between them. ``props`` re-exports the
names it used to define, because the test files that drive them reach for
``clay_props._swatch`` and friends.

:func:`draw_slots` is the one place a row of the document's own slots is laid out:
wrapped to the column's width (never sideways), capped at :data:`SLOT_ROWS_MAX` rows
before it scrolls in a child of its own, so sixty-four slots cannot push the cards
below them out of the pane.
"""

from __future__ import annotations

import weakref
from typing import Any

from imgui_bundle import imgui

from ..... import controls, icons, probe, theme, widgets
from .....tokens import sp
from ... import doc_io
from ... import mode as clay_mode
from ... import ops as clay_ops
from ...state import tab_over

#: A palette swatch's side, in design px -- the size Inker's own palette uses.
SWATCH = 22.0

#: Rows of slots the Colour card shows before the grid scrolls vertically. Three rows
#: of nine fit the 300 px column's twenty-seven slots without a scrollbar, which is
#: more than a model made by hand grows; a bigger palette (an imported one) scrolls
#: in place instead of taking the pane. Three, not four: with the "All shapes" dock
#: open at 1600x950 a fourth row put Delete's foot under the dock's header.
SLOT_ROWS_MAX = 3

_SAVING = "This document is being written; the palette comes back when it lands."


def refused_by_drag(ctx: Any) -> bool:
    """Whether a live G/R/S, gizmo or UV drag shuts a human palette or Add door, said
    in a toast.

    The 2026-10-11 audit's clay-16 and clay-17: a swatch, the ``+`` and every Add
    button called the document directly, so a press under a live drag added a slot or a
    shape and pushed a step the drag's commit then reverted against the history head it
    started from -- the shape of the 2026-10-10 audit's clay-19, which closed it for the
    starter colours only (``starter_colours._refused_by_drag``, the same toast). Asked at
    the top of each door, so nothing is added or pushed.
    """
    if not clay_ops.gesture_live(ctx):
        return False
    ctx.toast(doc_io.GESTURE_WHY, "error")
    return True


def add_reason(*, saving: bool, gesture: bool) -> str:
    """Why the grid's ``+`` is greyed, or ``""``. Pure, so the sentence is checkable."""
    if saving:
        return _SAVING
    return doc_io.GESTURE_WHY if gesture else ""


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
    # Into the census (the first-try test presses a swatch the way a person does,
    # and a colour button is not one of the controls ``_finish_item`` sees).
    probe.record(label=label, kind="swatch", selected=selected, tooltip=tooltip)
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
    says it cannot.
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
    (or no faces selected) only makes it the slot the Colours & textures page edits
    (``ClayTab.edit_slots`` -- not the object's own slot, the 2026-10-09 audit's
    clay-16), the one way to reach another slot's colour and texture from a selection.

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
        _edit_slot_of(ctx, doc, obj, index)
        return
    ctrl = _ctrl_held()
    if not ctrl:
        clay_mode.remember_slot(ctx, doc, index)
    if mode == "object" and not ctrl:
        doc.repaint_object(obj.uid, index)
        tab = _tab_for(ctx, doc)
        if tab is not None:
            # The paint made it the object's slot, so the page follows it.
            tab.forget_edit_slot(obj.uid)
        return
    op = clay_ops.get("assign-material")
    if mode != "face" or ctrl or not op.enabled(doc):
        _edit_slot_of(ctx, doc, obj, index)
        return
    clay_ops.run(ctx, doc, op, index=index)


def _tab_for(ctx: Any, doc: Any) -> Any:
    """The tab over *doc*, or ``None`` (a headless ctx): ``state.tab_over``."""
    return tab_over(ctx, doc)


def _edit_slot_of(ctx: Any, doc: Any, obj: Any, index: int) -> None:
    """Make *index* the slot the Colours & textures page edits for *obj*.

    Pane state on the tab, **not** ``set_props(material=...)``: the 2026-10-09
    audit's clay-16. ``Obj.material`` is what a rebuilt shape's faces are stamped
    with, so a Ctrl+click "only to edit" recoloured the whole shape the next time
    a segment count changed. ``choose_edit_slot`` records what the shape wore when it was
    chosen, so the choice lapses once a repaint, an Add or an undo changes that (the
    2026-10-09 second audit's clay-66).
    """
    tab = _tab_for(ctx, doc)
    if tab is not None:
        tab.choose_edit_slot(doc, obj, index)


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
    if refused_by_drag(ctx):
        return
    obj = single_selected(doc)
    if obj is not None:
        _pick_slot(ctx, doc, obj, index)
        return
    if _ctrl_held():
        return
    # A plain press only (above and in ``_pick_slot``): Ctrl+click means "let me
    # edit this slot", not "I want this colour", and remembering it would paint
    # the next shape in a slot the user only opened. Recorded even when the
    # press paints nothing (nothing selected) -- picking a colour and *then*
    # adding the shape is as natural an order as the reverse.
    clay_mode.remember_slot(ctx, doc, index)
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


#: ``doc -> (rev, sentence)``. The sentence asks the registry whether faces can be
#: painted, and the grid wants it every frame; ``rev`` covers every edit, selection
#: and undo, so nothing can change the answer without moving it first. Weak, so a
#: closed document takes its entry with it.
_HINTS: weakref.WeakKeyDictionary[Any, tuple[Any, str]] = weakref.WeakKeyDictionary()


def click_hint(doc: Any) -> str:
    """What a plain click on a slot swatch would do right now, in a sentence.

    A pure function of the document (memoised on ``doc.rev``, so an unchanged
    document asks the registry nothing), so the sentence the tooltip carries is
    checkable without a frame. It names the *current* behaviour rather than all
    of them: "paint the selected faces" while faces are selected, and the reason
    the swatch would do nothing while they are not.
    """
    rev = getattr(doc, "rev", None)
    cached = _HINTS.get(doc) if rev is not None else None
    if cached is not None and cached[0] == rev:
        return cached[1]
    text = _click_hint(doc)
    if rev is not None:
        _HINTS[doc] = (rev, text)
    return text


def _click_hint(doc: Any) -> str:
    mode = doc.element_mode
    one = single_selected(doc) is not None
    # The 2026-10-07 audit's clay-38: the Ctrl+click clause and the vertex/edge
    # "picks the slot" sentence were shown for any number of objects, but
    # ``click_slot`` retargets the Inspector only with exactly one object
    # selected -- with none or several the press does nothing of the sort.
    edit = (
        "Ctrl+click edits this slot in the Inspector."
        if one
        else "Select one object to edit its slot."
    )
    if mode == "face":
        if clay_ops.get("assign-material").enabled(doc):
            return f"Click paints the selected faces. {edit}"
        return f"Select faces to paint them. {edit}"
    if mode == "object":
        if not doc.selection:
            return "Click picks the colour the next shape you add will wear."
        if one:
            return f"Click repaints the selected object. {edit}"
        return f"Click repaints every selected object. {edit}"
    if one:
        return "Click picks the slot the Inspector edits. Switch to face mode (3) to paint faces."
    return f"{edit} Switch to face mode (3) to paint faces."


def slot_label(doc: Any, index: int) -> str:
    """``"2  Red"``, or ``"2"`` for an unnamed slot -- the line under the grid."""
    name = doc.materials[index].name if 0 <= index < len(doc.materials) else ""
    return f"{index}  {name}" if name else f"{index}"


def slot_rows(count: int, avail: float, side: float, gap: float) -> int:
    """How many rows *count* swatches of width *side* wrap into over *avail* px.

    Pure, so the cap on the grid's height is a number a test can check without a
    frame. At least one swatch to a row, however narrow the column.
    """
    per_row = max(1, int((avail + gap) // (side + gap)))
    return max(1, -(-count // per_row))


def add_slot_and_paint(ctx: Any, doc: Any) -> int | None:
    """The grid's ``+``: append a grey slot and wear it, as **one** undo step. -> its
    index, or ``None`` under a live drag (nothing added; :func:`refused_by_drag`).

    "Wear" is :func:`click_slot`'s rule for whatever is selected -- the shapes
    repainted, the picked faces painted, or (nothing selected) the colour the next
    shape arrives in -- so a new colour is a click and a trip to Edit colour, not a
    click, a click on the swatch it made and the trip. The slot and the paint are
    folded so a Ctrl+Z takes both back, the way a starter colour's click does.
    """
    if refused_by_drag(ctx):
        return None
    mark = doc.history.mark()
    try:
        index = doc.add_material()
        click_slot(ctx, doc, index)
    finally:
        doc.history.collapse_since(mark)
    return index


def draw_slots(
    ctx: Any, doc: Any, *, current: int | None, saving: bool, mixed: bool = False
) -> None:
    """The document's slots as one wrapped grid of swatches, then a ``+``.

    The slots are the document's, and only the document's: a new model keeps the
    one grey slot it has and nothing here seeds more. The grid wraps to the column;
    past :data:`SLOT_ROWS_MAX` rows it scrolls in a child of its own. Labels carry
    their own ``##slotsw/`` prefix, apart from the Colour card's ``##inspsw/``
    starters, so a control found by label is never the wrong row's. A press goes to
    :func:`click_slot`, which paints, retargets (Ctrl) or remembers by what is
    selected; the line under the grid names the slot under the pointer, else the
    one in use -- or says "Mixed" when *mixed* (the shape's faces wear more than one
    slot, so none is "the" one: the 2026-10-09 second audit's clay-67).
    """
    side = sp(SWATCH)
    gap = imgui.get_style().item_spacing.x
    plus_w = widgets.button_width(icons.PLUS)
    avail = imgui.get_content_region_avail().x
    rows = slot_rows(len(doc.materials) + 1, avail, side, gap)
    capped = rows > SLOT_ROWS_MAX
    if capped:
        height = SLOT_ROWS_MAX * side + (SLOT_ROWS_MAX - 1) * imgui.get_style().item_spacing.y
        imgui.begin_child("##slot-grid", (0.0, height), 0, 0)
    hint = click_hint(doc)
    hovered: int | None = None
    clicked: int | None = None
    for i, entry in enumerate(doc.materials):
        if i:
            widgets.same_line_or_wrap(side)
        textured = entry.base_color is not None
        name = entry.name or f"slot {i}"
        tip = _SAVING if saving else f"{i}: {name}" + (" (textured)" if textured else "")
        if not saving:
            tip += "\n" + hint
        if _swatch(
            f"##slotsw/{i}",
            _swatch_colour(entry),
            side,
            selected=i == current,
            textured=textured,
            tooltip=tip,
        ):
            clicked = i
        if imgui.is_item_hovered():
            hovered = i
    widgets.same_line_or_wrap(plus_w)
    gesture = clay_ops.gesture_live(ctx)
    why_add = add_reason(saving=saving, gesture=gesture)
    add = controls.small_button(
        f"{icons.PLUS}##slot-add",
        enabled=not why_add,
        reason=why_add,
        tooltip="Add a palette slot and use it. It starts as a grey; Edit colour changes it.",
    )
    if capped:
        imgui.end_child()
    if saving:
        return
    if add:
        add_slot_and_paint(ctx, doc)
    elif clicked is not None:
        click_slot(ctx, doc, clicked)
    shown = hovered if hovered is not None else current
    if shown is None:
        widgets.muted("Mixed: faces differ" if mixed else "Palette")
    else:
        widgets.muted(slot_label(doc, shown))
