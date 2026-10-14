"""The timeline: a strip under the viewport, in Pose mode, for the animation being authored.

Two rows of controls over a dope sheet. The controls are the transport (first, previous, play, next,
last), the animation list (new, copy, rename, delete) and the clip's own settings (frame rate, loop,
interpolation, auto-key). The dope sheet is a ruler over one row per bone with a diamond at every
time the bone is keyed; the playhead is the line through it.

**What is drawn is the document's, and what is held is the strip's.** Every number comes off
``ClayDoc`` (``rig_time``, ``rig_state``, the selected animation) through
:mod:`~realmspinner.studio.modes.clay.timeline`, whose arithmetic is tested without a frame. The
strip owns only the gesture in hand (``ClayTab.key_drag``), the diamonds picked (``key_sel``) and
how far the rows are scrolled. A diamond drag is previewed as ghosts and written once, on release,
as one step; scrubbing writes nothing but the playhead.

Drawn by the viewport (``_clay_viewport``) in the height it reserves, only in Pose mode. Its (?) is
on the settings row, since the strip has no heading to carry one, and opens *The timeline*.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from ..... import controls, dialogs, icons, theme, toolbar, widgets
from .....manual import render as manual_render
from .....tokens import sp
from ... import ops as clay_ops
from ... import timeline as tl

#: The strip's height in design pixels: two control rows and room for four bone rows.
HEIGHT = 196.0
#: A bone row's height, and the ruler's.
ROW_H = 20.0
RULER_H = 22.0
#: The bone-name column's width.
GUTTER = 132.0
PAD = 6.0
#: How far (pixels) a diamond may be from the pointer and still be the one pressed.
HIT = 8.0
#: A diamond's half-diagonal.
DIAMOND = 5.0


def height(doc: Any) -> float:
    """The height the viewport reserves: the strip in Pose mode, nothing otherwise."""
    return sp(HEIGHT) if getattr(doc, "pose_mode", False) else 0.0


def draw(ctx: Any, tab: Any, width: float, h: float) -> None:
    """The strip, *width* by *h* pixels, at the cursor."""
    imgui.push_style_var(imgui.StyleVar_.window_padding.value, (sp(PAD), sp(PAD)))
    shown = imgui.begin_child(
        "##clay-timeline",
        (width, h),
        imgui.ChildFlags_.always_use_window_padding.value,
        imgui.WindowFlags_.no_scrollbar.value | imgui.WindowFlags_.no_scroll_with_mouse.value,
    )
    try:
        if shown:
            _body(ctx, tab)
    finally:
        imgui.end_child()
        imgui.pop_style_var()


def _body(ctx: Any, tab: Any) -> None:
    doc = tab.doc
    _transport(ctx, tab, doc)
    _settings(ctx, tab, doc)
    _sheet(ctx, tab, doc)


def _refuse(ctx: Any, action: Any) -> None:
    """Run *action*; an ``OpError`` is a sentence on screen, not a traceback."""
    from ......kernels.mesh.elements import OpError

    try:
        action()
    except OpError as error:
        clay_ops.toast(ctx, str(error))


# -- row 1: the transport and the animation list ---------------------------------------------------


def _readout(doc: Any) -> str:
    return f"{doc.rig_time:.2f} s  ·  frame {tl.frame_of(doc, doc.rig_time)}"


def _transport(ctx: Any, tab: Any, doc: Any) -> None:
    animation = doc.animation()
    playable = tl.length(doc) > 0.0
    why_play = "Key a bone first: there is nothing to play yet."
    label, glyph, tip = widgets.transport_label(doc.playing)
    saving = bool(tab.saving)
    items = [
        toolbar.Item("first", "|<", tooltip="Go to the start (Home)", pinned=True),
        toolbar.Item("prev", "<", tooltip="Back one frame (Left)", pinned=True),
        toolbar.Item(
            "play", label, glyph, tooltip=tip, enabled=playable or doc.playing,
            reason=why_play, pinned=True,
        ),
        toolbar.Item("next", ">", tooltip="Forward one frame (Right)", pinned=True),
        toolbar.Item("last", ">|", tooltip="Go to the last key (End)", pinned=True),
        toolbar.Item(
            "new", "New", icons.PLUS, tooltip="Start an empty animation",
            enabled=not saving, reason=widgets.DOCUMENT_SAVING_WHY, priority=1,
        ),
        toolbar.Item(
            "copy", "Copy", icons.COPY, tooltip="Duplicate this animation",
            enabled=animation is not None and not saving,
            reason=widgets.DOCUMENT_SAVING_WHY if saving else "There is no animation to copy.",
            priority=1,
        ),
        toolbar.Item(
            "rename", "Rename", icons.PENCIL, tooltip="Rename this animation",
            enabled=animation is not None and not saving,
            reason=widgets.DOCUMENT_SAVING_WHY if saving else "There is no animation to rename.",
            priority=2,
        ),
        toolbar.Item(
            "delete", "Delete", icons.TRASH, tooltip="Delete this animation",
            enabled=animation is not None and not saving,
            reason=widgets.DOCUMENT_SAVING_WHY if saving else "There is no animation to delete.",
            role=controls.ButtonRole.DESTRUCTIVE, priority=1,
        ),
    ]
    text = _readout(doc)
    toolbar.toolbar(
        "clay-timeline-bar",
        items,
        lambda key: _transport_action(ctx, tab, doc, key),
        trailing=(imgui.calc_text_size(text).x, lambda: imgui.text_unformatted(text)),
    )


def _transport_action(ctx: Any, tab: Any, doc: Any, key: str) -> None:
    if key == "first":
        tl.go_start(doc)
    elif key == "prev":
        tl.step(doc, -1)
    elif key == "play":
        tl.toggle_play(doc)
    elif key == "next":
        tl.step(doc, 1)
    elif key == "last":
        tl.go_end(doc)
    elif key == "new":
        _refuse(ctx, lambda: doc.add_animation("Animation"))
        tab.key_sel = []
    elif key == "copy":
        source = doc.active_animation
        if source is not None:
            _refuse(ctx, lambda: doc.add_animation(doc.animation().name, copy_of=source))
            tab.key_sel = []
    elif key == "rename":
        _ask_rename(ctx, doc)
    elif key == "delete":
        index = doc.active_animation
        if index is not None:
            _refuse(ctx, lambda: doc.delete_animation(index))
            tab.key_sel = []


def _ask_rename(ctx: Any, doc: Any) -> None:
    index = doc.active_animation
    animation = doc.animation()
    if index is None or animation is None:
        return

    def accept(text: str) -> None:
        _refuse(ctx, lambda: doc.rename_animation(index, text))

    ctx.prompts.ask(
        dialogs.Prompt(
            title="Rename animation", label="Name", value=animation.name, on_accept=accept
        )
    )


# -- row 2: the clip's settings ------------------------------------------------------------

_INTERPOLATION_WORDS = {"linear": "Linear", "step": "Step"}


def _settings(ctx: Any, tab: Any, doc: Any) -> None:
    animations = doc.rig_state.animations
    index = doc.active_animation
    current = "-1" if index is None else str(index)
    options = [("-1", "(no animation)")] + [(str(i), a.name) for i, a in enumerate(animations)]
    imgui.set_next_item_width(sp(170.0))
    changed, picked = controls.combo(
        "##clay-timeline-anim", current, options, tooltip="Which animation the strip edits"
    )
    if changed:
        doc.set_active_animation(None if picked == "-1" else int(picked))
        doc.set_rig_time(doc.rig_time)
        doc.pose_rev += 1
        tab.key_sel = []

    animation = doc.animation()
    have = animation is not None and index is not None
    imgui.same_line()
    imgui.set_next_item_width(sp(84.0))
    changed, picked = controls.combo(
        "##clay-timeline-fps",
        str(doc.rig_state.fps),
        [(str(f), f"{f} fps") for f in tl.FPS_CHOICES],
        tooltip="The frame rate the strip counts in and the file is written at",
    )
    if changed:
        _refuse(ctx, lambda: doc.set_fps(int(picked)))
    imgui.same_line()
    changed, value = controls.checkbox(
        "Loop##clay-timeline-loop",
        bool(animation.loop) if have else False,
        enabled=have,
        reason="There is no animation yet.",
        tooltip="Play it round and round. A one-shot stops on its last key and is exported as such",
    )
    if changed and index is not None:
        _refuse(ctx, lambda: doc.set_animation_loop(index, value))
    imgui.same_line()
    imgui.set_next_item_width(sp(84.0))
    changed, picked = controls.combo(
        "##clay-timeline-interp",
        animation.interpolation if have else "linear",
        list(_INTERPOLATION_WORDS.items()),
        enabled=have,
        reason="There is no animation yet.",
        tooltip="Linear blends between keys; Step holds each key until the next",
    )
    if changed and index is not None:
        _refuse(ctx, lambda: doc.set_animation_interpolation(index, picked))
    imgui.same_line()
    changed, value = controls.checkbox(
        "Auto-key##clay-timeline-autokey",
        bool(doc.auto_key),
        tooltip=(
            "On: posing a bone keys it at the playhead. Off: the pose is held until you "
            "press I, or move the playhead"
        ),
    )
    if changed:
        doc.auto_key = bool(value)
        doc.pose_rev += 1
    imgui.same_line()
    manual_render.help_button_inline(ctx, "clay-timeline")


# -- the dope sheet ------------------------------------------------------------


def _diamond(draw_list: Any, x: float, y: float, colour: int, half: float) -> None:
    draw_list.add_quad_filled((x, y - half), (x + half, y), (x, y + half), (x - half, y), colour)


def _sheet(ctx: Any, tab: Any, doc: Any) -> None:
    rows = tl.rows(doc)
    avail = imgui.get_content_region_avail()
    if avail.x < sp(60.0) or avail.y < sp(RULER_H):
        return
    origin = imgui.get_cursor_screen_pos()
    gutter, ruler_h, row_h = sp(GUTTER), sp(RULER_H), sp(ROW_H)
    sheet = tl.Sheet(
        x0=origin.x + gutter,
        x1=origin.x + avail.x - sp(4.0),
        y0=origin.y,
        ruler_h=ruler_h,
        row_h=row_h,
        visible=tl.span(doc),
        gutter=gutter,
        scroll=tab.timeline_scroll,
        rows_fit=max(int((avail.y - ruler_h) // row_h), 1),
        rows=len(rows),
    )
    tab.timeline_scroll = sheet.scroll
    # Diamonds that no longer exist (an undo, a deleted bone) are no longer picked.
    live = {(r.uid, t) for r in rows for t in r.times}
    tab.key_sel = [k for k in tab.key_sel if k in live]

    draw_list = imgui.get_window_draw_list()
    _paint(draw_list, doc, tab, rows, sheet)

    # The gutter's names are buttons: a bone is selected from its row.
    for slot, row in enumerate(rows[sheet.scroll : sheet.scroll + sheet.rows_fit]):
        imgui.set_cursor_screen_pos((origin.x, sheet.row_top(slot)))
        if controls.selectable_row(
            f"clay-timeline-bone/{row.uid}",
            "",
            selected=row.selected,
            size=(gutter - sp(2.0), row_h),
        ):
            doc.select([row.uid])
    # One button over the whole track area takes the mouse; what a press means is decided
    # from where it landed (``_interact``).
    imgui.set_cursor_screen_pos((sheet.x0, origin.y))
    imgui.invisible_button(
        "##clay-timeline-sheet", (max(sheet.x1 - sheet.x0, 1.0), ruler_h + sheet.rows_fit * row_h)
    )
    _interact(ctx, tab, doc, rows, sheet)


def _paint(draw_list: Any, doc: Any, tab: Any, rows: list[Any], sheet: Any) -> None:
    def col(name: str, alpha: float = 1.0) -> int:
        return imgui.get_color_u32(theme.rgba(getattr(theme, name), alpha))

    body_bottom = sheet.y0 + sheet.ruler_h + sheet.rows_fit * sheet.row_h
    draw_list.add_rect_filled(
        (sheet.x0, sheet.y0), (sheet.x1, body_bottom), col("ELEV_1"), sp(3.0)
    )
    # The ruler.
    for time, label, major in tl.ruler(sheet.visible, sheet.x1 - sheet.x0):
        x = sheet.to_x(time)
        top = sheet.y0 + sheet.ruler_h * (0.4 if major else 0.7)
        draw_list.add_line((x, top), (x, sheet.y0 + sheet.ruler_h), col("MUTED", 0.7))
        if label:
            draw_list.add_text((x + sp(3.0), sheet.y0 + sp(2.0)), col("MUTED"), label)
    draw_list.add_line(
        (sheet.x0, sheet.y0 + sheet.ruler_h), (sheet.x1, sheet.y0 + sheet.ruler_h), col("EDGE")
    )
    # Rows: a stripe under every other one, the name in the gutter, a diamond per key time.
    drag = tab.key_drag if tab.key_drag and tab.key_drag.get("mode") == "keys" else None
    shown = rows[sheet.scroll : sheet.scroll + sheet.rows_fit]
    picked = set(tab.key_sel)
    delta = 0.0 if drag is None else drag["delta"]
    for slot, row in enumerate(shown):
        top = sheet.row_top(slot)
        mid = top + sheet.row_h * 0.5
        if (slot + sheet.scroll) % 2:
            draw_list.add_rect_filled(
                (sheet.x0, top), (sheet.x1, top + sheet.row_h), col("PANEL", 0.5)
            )
        draw_list.push_clip_rect(
            (sheet.x0 - sheet.gutter, top), (sheet.x0 - sp(4.0), top + sheet.row_h), True
        )
        draw_list.add_text(
            (sheet.x0 - sheet.gutter + sp(4.0), mid - imgui.get_text_line_height() * 0.5),
            col("TEXT" if row.selected else "MUTED"),
            row.name,
        )
        draw_list.pop_clip_rect()
        for time in row.times:
            chosen = (row.uid, time) in picked
            at_head = abs(time - doc.rig_time) < 1e-6
            colour = col("ACCENT") if chosen else col("TEXT", 1.0 if at_head else 0.75)
            if chosen and drag is not None and delta:
                _diamond(draw_list, sheet.to_x(time), mid, col("MUTED", 0.4), sp(DIAMOND))
                _diamond(draw_list, sheet.to_x(max(time + delta, 0.0)), mid, colour, sp(DIAMOND))
            else:
                _diamond(draw_list, sheet.to_x(time), mid, colour, sp(DIAMOND))
    # The playhead, over everything.
    head = sheet.to_x(min(doc.rig_time, sheet.visible))
    draw_list.add_line((head, sheet.y0), (head, body_bottom), col("ACCENT"), sp(1.5))
    draw_list.add_triangle_filled(
        (head - sp(5.0), sheet.y0),
        (head + sp(5.0), sheet.y0),
        (head, sheet.y0 + sp(8.0)),
        col("ACCENT"),
    )


def _interact(ctx: Any, tab: Any, doc: Any, rows: list[Any], sheet: Any) -> None:
    io = imgui.get_io()
    mouse = (io.mouse_pos.x, io.mouse_pos.y)
    hovered = imgui.is_item_hovered()
    if hovered and io.mouse_wheel and sheet.rows > sheet.rows_fit:
        tab.timeline_scroll = sheet.clamp_scroll(sheet.scroll - int(io.mouse_wheel))
    drag = tab.key_drag
    if imgui.is_item_activated():
        tab.key_drag = _begin(tab, doc, rows, sheet, mouse, io.key_shift)
        drag = tab.key_drag
    if drag is None:
        return
    if drag["mode"] == "scrub":
        if imgui.is_mouse_down(0):
            tl.go_to(doc, min(max(sheet.to_t(mouse[0]), 0.0), sheet.visible))
        else:
            tab.key_drag = None
    elif drag["mode"] == "keys":
        if imgui.is_mouse_down(0):
            delta = tl.snap(doc, sheet.to_t(mouse[0])) - drag["anchor"]
            drag["delta"] = delta if abs(delta) > 1e-9 else 0.0
        else:
            tab.key_drag = None
            if drag["delta"] and not tab.saving:
                _refuse(ctx, lambda: doc.move_bone_keys(tab.key_sel, drag["delta"]))
                tab.key_sel = [
                    (uid, round(max(time + drag["delta"], 0.0), 6)) for uid, time in tab.key_sel
                ]


def _begin(
    tab: Any, doc: Any, rows: list[Any], sheet: Any, mouse: tuple[float, float], additive: bool
) -> dict[str, Any] | None:
    """What a press on the sheet starts: a scrub (ruler or empty track) or a diamond drag."""
    row = sheet.row_at(mouse[1])
    if row is None:
        tl.go_to(doc, min(max(sheet.to_t(mouse[0]), 0.0), sheet.visible))
        return {"mode": "scrub"}
    key = tl.nearest_key(rows, mouse[0], row, sheet.to_x, sp(HIT))
    if key is None:
        if not additive:
            tab.key_sel = []
        tl.go_to(doc, min(max(sheet.to_t(mouse[0]), 0.0), sheet.visible))
        return {"mode": "scrub"}
    if additive:
        if key in tab.key_sel:
            tab.key_sel = [k for k in tab.key_sel if k != key]
        else:
            tab.key_sel = [*tab.key_sel, key]
    elif key not in tab.key_sel:
        tab.key_sel = [key]
    tl.go_to(doc, key[1])
    return {"mode": "keys", "anchor": key[1], "delta": 0.0}
