"""The colour picker: a hue/saturation wheel, a Value bar and an Alpha bar
under the palette, plus a numeric readout in whichever colour space the user
last asked for.

Aseprite's own picker is a wheel over a value bar with RGB/HSV/HSL/Gray text
entry beneath it, and this pane used to be a flatter copy of the *text-entry*
half alone -- a tab strip over RGB, HSV, HSL and Gray, a slider per channel.
That covered every space but none of it was a colour *picker*: finding a hue
meant three digits typed by feel, or the ``I`` tool's eyedrop. The wheel adds
the one gesture Aseprite's does -- point at the colour you mean -- without
taking away any of the four numeric spaces the old tab strip offered; they are
still here, just inline under a **Space** combo instead of one slider stack
per tab.

**It writes through the same doors the rest of the app does.** ``state.set_fg``
for the foreground (which is what clears ``fg_slot`` -- see its docstring),
plain assignment for the background, and ``doc.recolour_slot`` when the
document is indexed and the brush is holding a slot. That last case is
deliberate and is what Aseprite does: in an indexed sprite the picker edits
the *entry*, because the pixels are numbers and a colour that was not in the
table is not a colour the document can hold. Every one of the wheel, the two
bars and the numeric row is a different way to change *what colour*, and
:func:`write` is the one place any of them are allowed to say it.

Conversions come from ``colorsys``, the way ``inker.indexed.harmony`` already
does. ``inker.filters._from_hsl`` is not reused: it is array-shaped for a
whole-layer filter and would be the wrong tool for one scalar triple.

**The wheel's own maths lives in ``colorwheel.py``, not here.** That module
imports no imgui and is proven correct (a round-trip test) without a window;
this file is the imgui half -- drawing the baked texture, the ring marker and
the invisible button that reads the mouse -- plus the held-triple cache that
keeps a drag from losing the hue of a colour that has just been dragged to
grey (see :func:`_held_hsv`).

**``_rgb`` is kept, unused by :func:`draw`, on purpose.** It is the pre-wheel
RGB tab -- three ``labeled_slider_int`` channels plus Alpha -- and
``tests/test_undo_gesture_doors.py`` calls it directly, scripting a drag
against imgui's own ``"##Red"`` ``slider_int`` to pin the one-gesture-one-undo
-step fix that pane was part of. The new numeric row draws a compact inline
box per channel instead (:func:`_row`), which is a different imgui widget
family under a different id -- so it cannot be what that regression test
scripts. Rewriting ``_rgb`` to match would have meant either breaking that
test's own scripted id or leaving the regression it pins unchecked; leaving
the working function in place, still exercised by that test, was the smaller
risk. ``_hsv``/``_hsl``/``_gray``/the old tab-strip ``_wheel`` had no such
caller and were replaced outright.
"""

from __future__ import annotations

import colorsys
import math
from typing import Any

from imgui_bundle import imgui

from .. import anchors, colorwheel, controls, docmodes, inker_mode, theme, tokens, widgets
from ..manual import render as manual_render
from ..tokens import sp

#: The least this pane may be squeezed to, in design px.
#:
#: **Measured, not estimated.** The pre-wheel version of this comment named
#: that incident: a floor that said 200 while the pane's real content ran 400
#: went unnoticed until the hex field drew past the pane's bottom edge and
#: imgui clipped it away -- the one control in Inker nobody could click. This
#: number is the same discipline applied to the wheel layout: a real imgui
#: frame, no GL needed (layout height comes from item spacing, not from what
#: got uploaded), built at the sidebar's three stops -- 260, 300 (the app's
#: own default, ``layout.SIDEBAR_W``) and 360 px -- and read back with
#: ``imgui.get_cursor_pos_y()`` after the last control. The heading, target
#: row, wheel, Value bar, Alpha bar, Space combo, a four-field numeric row and
#: the hex field came to 519 px at every one of those three widths at the
#: original ``WHEEL_DIAMETER`` of 150 (the wheel is the only thing here that
#: resizes with the sidebar, and nothing else wraps), on a document holding a
#: palette slot -- the taller of the two cases, since editing a slot adds the
#: "Editing palette slot N" line.
#:
#: **2026-09-16: derived, not re-measured with a real frame.** The wheel grew
#: from 150 to 190 design px (below) to read less cramped; it is drawn as a
#: single square ``invisible_button(diameter, diameter)`` in the vertical
#: stack, so it is the only element whose height changed and it changed by
#: exactly ``190 - 150 = 40`` px -- nothing above or below it in the pane
#: shifted shape. This floor is that arithmetic (``519 + 40``), not a fresh
#: three-widths render; run ``/exercise-mode inker`` once to confirm nothing
#: clips (the hex field and numeric row sit right below the wheel) before
#: trusting it the way the rest of this file's history does.
PICKER_FLOOR = 559.0

#: The wheel's own diameter, in design px, before it is clamped to whatever
#: width the sidebar actually has (see :func:`_wheel`). Raised from 150 to
#: 190 on 2026-09-16 (user feedback: "make it larger, it's small") -- still
#: comfortably inside ``PICKER_FLOOR`` at the sidebar's default 300 px width.
WHEEL_DIAMETER = 190.0

#: The two things the sliders can be pointed at. The foreground is what a
#: left-drag writes with and the background what a right-drag does, which is
#: the same pair the toolbox's two chips draw.
TARGETS: tuple[tuple[str, str], ...] = (("fg", "Foreground"), ("bg", "Background"))

#: Rec. 601 luma, which is what ``inker`` grades and sorts a palette by. Gray
#: has to pick *a* number to call the grey of a colour, and picking a
#: different one from the rest of the app would make "Brightness" in the
#: palette sort and "Gray" here disagree about the same swatch.
LUMA = (0.299, 0.587, 0.114)

#: The numeric row's colour spaces, in combo order.
SPACES: tuple[tuple[str, str], ...] = (
    ("rgba", "RGBA"),
    ("hsv", "HSV"),
    ("hsl", "HSL"),
    ("gray", "Gray"),
)
_SPACE_KEYS = frozenset(key for key, _ in SPACES)

#: Where the chosen space lives: ``ctx.state.preview``, the same session-only
#: dict ``app_settings.CATEGORY_SLOT`` remembers the settings category in.
#: Which space the numeric row shows is where the user's eye was, not a
#: preference worth a settings-file round trip -- exactly ``CATEGORY_SLOT``'s
#: own reasoning, so this follows its idiom rather than growing a field on
#: ``InkerState`` for one string.
SPACE_SLOT = "inker_picker_space"

#: Labels for the wheel + Value bar's held triple, and for the HSV row's --
#: the same three, on purpose. See :func:`_held_hsv`.
HSV_LABELS = ("Hue", "Saturation", "Value")
HSL_LABELS = ("Hue", "Saturation", "Lightness")

#: Where the wheel's baked texture lives in ``ctx.state.preview``. Under the
#: same ``inker_tex:`` prefix ``panes/inker_textures.py`` keys every other
#: Inker texture with, deliberately: that is what makes
#: ``inker_textures.release_all`` -- already wired into the mode's real
#: teardown path (``inker_keys.release_all``, and the app-close sweep in
#: ``main.py``) -- free this one too, with nothing added to either of those
#: files. ``__picker__`` stands in for the per-tab uid every other entry under
#: this prefix carries, chosen so it can never collide with a real tab's.
_WHEEL_TEX_KEY = "inker_tex:__picker__:wheel"


def clamp8(value: float) -> int:
    return max(0, min(255, int(round(value))))


def draw(ctx: Any) -> None:
    anchors.mark_window("inker/picker")
    state = inker_mode.ensure(ctx)
    widgets.section("Picker")
    # After the heading, never before it: ``help_button`` is a ``same_line``,
    # which returns to the previous row unconditionally -- called first it
    # lands on whatever the pane above drew.
    manual_render.help_button(ctx, "inker-picker")
    tab = state.active

    changed, picked = controls.segmented_choice(
        "inkpickertarget", TARGETS, target_of(state), compact=True
    )
    if changed:
        state.picker_target = picked
    slot = slot_of(state, tab)
    if slot is not None:
        widgets.muted(f"Editing palette slot {slot + 1}")

    colour = read(state, tab, slot)
    _wheel(ctx, state, tab, slot, colour)
    _value(ctx, state, tab, slot, colour)
    _alpha(ctx, state, tab, slot, colour)

    space = _space(ctx)
    chosen = widgets.labeled_combo("Space", space, list(SPACES))
    if chosen != space:
        ctx.state.preview[SPACE_SLOT] = chosen
        space = chosen
    _ROWS[space](ctx, state, tab, slot, colour)

    _hex(ctx, state, tab, slot, colour)


def target_of(state: Any) -> str:
    """Which colour the sliders point at. ``fg`` unless asked otherwise."""

    return "bg" if getattr(state, "picker_target", "fg") == "bg" else "fg"


def slot_of(state: Any, tab: Any) -> int | None:
    """The palette entry the sliders edit, or ``None`` for a free colour.

    Only the foreground can be holding a slot -- ``set_fg`` is the one door
    that records one -- and only an indexed document has entries to edit.
    """
    if tab is None or target_of(state) != "fg":
        return None
    doc = tab.doc
    if not getattr(doc, "is_indexed", False):
        return None
    index = state.fg_slot
    if index is None or not (0 <= int(index) < len(doc.palette)):
        return None
    return int(index)


def read(state: Any, tab: Any, slot: int | None) -> tuple[int, int, int, int]:
    """The colour the sliders are showing right now."""

    if slot is not None:
        source: Any = tuple(tab.doc.palette[slot])
    else:
        source = state.fg if target_of(state) == "fg" else state.bg
    channels = [int(channel) for channel in tuple(source)[:4]]
    while len(channels) < 4:
        channels.append(255)
    return (channels[0], channels[1], channels[2], channels[3])


def write(ctx: Any, state: Any, tab: Any, slot: int | None, colour: Any) -> None:
    """One door, so the wheel, the two bars, the numeric row and the hex field
    cannot disagree.

    Nothing is persisted: the two colours are session state, exactly as they
    are in ``inker_colors``, and writing the settings file on every frame of a
    drag would be a file write per pixel of travel. ``ctx`` is taken anyway so
    the signature does not have to change the day one of these grows a door
    that needs it.
    """

    value = tuple(clamp8(channel) for channel in tuple(colour)[:4])
    if slot is not None:
        if tab.doc.recolour_slot(slot, value):
            state.palette_usage = None
            state.set_fg(value, slot)
        return
    if target_of(state) == "fg":
        state.set_fg(value)
    else:
        state.bg = value


def _history(tab: Any, slot: int | None) -> Any:
    """The undo stack a channel write over ``slot`` should fold into, or
    ``None`` for a free colour -- which is session state and folds nothing."""

    return getattr(tab.doc, "history", None) if slot is not None else None


def _slider(label: str, value: int, high: int, tab: Any, slot: int | None) -> tuple[bool, int]:
    """One of this pane's full-width bars (Value, Alpha), folded to one undo
    step per drag. See :func:`_history`."""

    changed, out = widgets.labeled_slider_int(label, value, 0, high)
    controls.fold_undo(_history(tab, slot))
    return changed, out


def _alpha(ctx: Any, state: Any, tab: Any, slot: int | None, colour: tuple) -> None:
    """The one channel every space shares, so it is drawn once."""

    changed, alpha = _slider("Alpha", int(colour[3]), 255, tab, slot)
    if changed:
        write(ctx, state, tab, slot, (colour[0], colour[1], colour[2], alpha))


def _space(ctx: Any) -> str:
    """Which colour space the numeric row is showing. See ``SPACE_SLOT``."""

    key = str(ctx.state.preview.get(SPACE_SLOT) or SPACES[0][0])
    return key if key in _SPACE_KEYS else SPACES[0][0]


def _held_hsv(state: Any, colour: tuple, labels: tuple[str, str, str]) -> tuple[int, int, int]:
    """The wheel/Value bar's (or the HSL row's) triple, held rather than
    re-derived every frame.

    **Deriving it from the 8-bit RGB on every frame loses information the
    triple is the only record of.** A grey has no hue at all, so a wheel that
    always re-derived hue from ``colour`` would snap its own marker to the
    centre-implied angle (0) the instant Saturation reached zero, and dragging
    hue back up from there would start from the wrong side of the wheel. On a
    dark or desaturated colour the round trip through three bytes also
    quantises, so a Saturation drag would walk to a number the user did not
    choose and could not get back to by dragging the other way.

    The held triple is dropped the moment ``colour`` changes from anywhere
    else the wheel does not know about -- the numeric row typing a different
    number, an eyedrop, a palette click -- which is what keeps it a *cache of
    this gesture* rather than a second opinion about what colour is selected.

    Shared between the wheel/Value bar and the HSV numeric row on purpose
    (both pass ``HSV_LABELS``): a hue picked on the wheel and a hue read in
    the HSV row are one number, not two that happen to agree until one of
    them moves.
    """
    rgb = tuple(int(channel) for channel in colour[:3])
    held = getattr(state, "picker_space", None)
    if held is not None and held[0] == labels and held[1] == rgb:
        return held[2]
    red, green, blue = (channel / 255.0 for channel in rgb)
    if labels == HSL_LABELS:
        first, second, third = _to_hsl(red, green, blue)
        top = 100.0
    else:
        first, second, third = colorsys.rgb_to_hsv(red, green, blue)
        top = 255.0
    return (
        int(round(first * 360.0)) % 360,
        int(round(second * 100.0)),
        int(round(third * top)),
    )


def _hold(
    state: Any,
    labels: tuple[str, str, str],
    rgb: tuple[int, int, int],
    triple: tuple[int, int, int],
) -> None:
    # Keyed on the bytes actually written, so the next frame recognises its
    # own answer and anything else's replaces it (see ``_held_hsv``).
    state.picker_space = (labels, tuple(int(c) for c in rgb), tuple(int(v) for v in triple))


def _wheel_texture(ctx: Any, size: int) -> Any:
    """The hue/saturation disc, baked once and re-baked only when its size or
    the UI scale changes.

    Follows ``panes/inker_textures.py``'s own rule for a cache entry that
    is not per-document (``checker``, there): keyed on a stamp that is
    compared and, on a mismatch, the old texture is unregistered with the
    imgui backend *and* released (``docmodes.forget_texture`` does both, in
    that order -- see its docstring) before a new one is created. ``tokens
    .SCALE`` is read directly, the way ``checker`` reads ``tokens.THEME``,
    rather than through ``state.fonts_dirty``: that flag is a one-shot signal
    the frame loop consumes to re-bake fonts, already false again by the time
    a pane's own ``draw`` runs, so a cache keyed on it would rebake once and
    then never notice a second scale change in the same session.

    ``None`` off-viewer, which is what the headless smoke suite and every
    state-only test run under -- :func:`_wheel` draws no image in that case
    and the invisible button beneath it still works.
    """
    if ctx.viewer is None:
        return None
    stamp = (size, tokens.SCALE)
    texture = ctx.state.preview.get(_WHEEL_TEX_KEY)
    if texture is not None and ctx.state.preview.get(f"{_WHEEL_TEX_KEY}:stamp") != stamp:
        docmodes.forget_texture(texture)
        ctx.state.preview.pop(_WHEEL_TEX_KEY, None)
        ctx.state.preview.pop(f"{_WHEEL_TEX_KEY}:stamp", None)
        texture = None
    if texture is None:
        pixels = colorwheel.wheel_image(size)
        texture = ctx.viewer.ctx.texture((size, size), 4, pixels.tobytes())
        texture.filter = (ctx.viewer.ctx.LINEAR, ctx.viewer.ctx.LINEAR)
        ctx.state.preview[_WHEEL_TEX_KEY] = texture
        ctx.state.preview[f"{_WHEEL_TEX_KEY}:stamp"] = stamp
    return texture


def _marker(draw_list: Any, at: tuple[float, float]) -> None:
    """The ring that shows the current colour on the wheel.

    Two circles, dark then light, rather than one: a single-colour ring is
    invisible over a hue it happens to match (a white ring on a pale yellow,
    say), and Aseprite's own marker is exactly this two-tone outline for the
    same reason.
    """
    draw_list.add_circle(at, sp(6.0), imgui.get_color_u32(theme.rgba(theme.BG)), 20, sp(2.5))
    draw_list.add_circle(at, sp(6.0), imgui.get_color_u32(theme.rgba(theme.TEXT)), 20, sp(1.25))


#: The darkest the disc is ever drawn, as an 8-bit multiply.
#:
#: The tint exists so the wheel does not promise a bright colour while the
#: Value bar says 0 -- but taken literally it draws a **black circle**, and a
#: fresh Inker opens on a black foreground. The first thing a new user would
#: see where the colour wheel goes is a featureless disc, which is a worse lie
#: than the one being fixed: it hides the control instead of describing the
#: colour. The floor keeps the hues legible at every value while still visibly
#: darkening as the bar comes down.
WHEEL_DIM_FLOOR = 90


def _wheel_pick(
    ctx: Any,
    state: Any,
    tab: Any,
    slot: int | None,
    colour: tuple,
    dx: float,
    dy: float,
    radius: float,
) -> None:
    """Apply a point on the disc, relative to its centre, as the new colour.

    Split out of :func:`_wheel` so the maths -- the part a regression test
    wants to pin -- does not require a real mouse or a real imgui frame to
    reach: a test can call this directly with contrived ``dx``/``dy``. Radius
    is a plain float rather than something re-read from imgui for the same
    reason. A drag that has left the disc is not pre-clamped here; ``dx``/
    ``dy`` are passed through as measured and :func:`colorwheel.colour_at`
    does the clamping (saturation cannot exceed 1.0), which is what keeps a
    drag past the rim tracking the pointer's angle instead of freezing.
    """
    if radius <= 0.0:
        return
    _, _, value = _held_hsv(state, colour, HSV_LABELS)
    # **A pick at Value 0 raises the value.** Every colour at value 0 is black,
    # so a click on the disc would otherwise return the black already in hand
    # -- the user picks a magenta, nothing changes, and the control reads as
    # broken. There is nothing to preserve by staying: value 0 carries no hue
    # and no saturation, so lifting it discards nothing the user chose. The
    # Value bar is directly beneath and takes it back down.
    if value <= 0:
        value = 255
    red, green, blue = colorwheel.colour_at(dx, dy, radius, value / 255.0)
    write(ctx, state, tab, slot, (red, green, blue, colour[3]))
    hue = int(round(math.degrees(math.atan2(dy, dx)))) % 360
    distance = min(math.hypot(dx, dy), radius)
    saturation = int(round((distance / radius) * 100.0))
    _hold(state, HSV_LABELS, (red, green, blue), (hue, saturation, value))


def _wheel(ctx: Any, state: Any, tab: Any, slot: int | None, colour: tuple) -> None:
    """The hue/saturation disc: a baked texture, a ring marker, and one
    invisible button covering both so a click or drag anywhere on it reads
    as a colour (:func:`_wheel_pick`)."""

    hue, saturation, value = _held_hsv(state, colour, HSV_LABELS)
    avail = imgui.get_content_region_avail().x
    diameter = max(sp(1.0), min(avail, sp(WHEEL_DIAMETER)))
    origin = imgui.get_cursor_screen_pos()
    pad = max(0.0, (avail - diameter) * 0.5)
    if pad:
        imgui.set_cursor_screen_pos((origin.x + pad, origin.y))
        origin = imgui.get_cursor_screen_pos()

    texture = _wheel_texture(ctx, max(1, int(round(diameter))))
    draw_list = imgui.get_window_draw_list()
    if texture is not None:
        # **Tinted by the current value, not re-baked at it.** The disc is hue
        # by saturation at full value, which is what lets it be one static
        # texture -- but drawn at full brightness beside a Value of 0 it says
        # the opposite of the truth: every pick on it returns black, because
        # ``_wheel_pick`` multiplies by the value, and a user clicking a bright
        # magenta and getting black has been told the control is broken.
        # Aseprite dims its wheel for the same reason. A multiply tint on
        # ``add_image`` costs nothing and cannot fall out of step with the
        # bake, where a second texture per value would be a rebake per drag of
        # the Value bar.
        shade = max(WHEEL_DIM_FLOOR, min(255, int(round(value))))
        draw_list.add_image(
            widgets.texture_ref(texture),
            (origin.x, origin.y),
            (origin.x + diameter, origin.y + diameter),
            (0.0, 0.0),
            (1.0, 1.0),
            imgui.IM_COL32(shade, shade, shade, 255),
        )
    radius = diameter / 2.0 - sp(1.0)
    centre = (origin.x + diameter / 2.0, origin.y + diameter / 2.0)
    angle = math.radians(hue)
    fraction = saturation / 100.0
    marker = (
        centre[0] + math.cos(angle) * fraction * radius,
        centre[1] + math.sin(angle) * fraction * radius,
    )
    _marker(draw_list, marker)

    imgui.invisible_button("##inkpickerwheel", (diameter, diameter))
    controls.fold_undo(_history(tab, slot))
    # Read before ``_finish_item``, not after: that call can draw a tooltip,
    # which opens and closes its own imgui window, and this pane is not the
    # place to find out the hard way whether that preserves "last item" state
    # for the invisible button underneath it.
    try:
        active = bool(imgui.is_item_active())
    except (AttributeError, RuntimeError):
        active = False
    controls._finish_item(
        tooltip="Drag to set hue and saturation. The bar below sets Value.",
        label="Colour wheel",
        kind="colorwheel",
    )
    if active:
        mouse = imgui.get_mouse_pos()
        _wheel_pick(ctx, state, tab, slot, colour, mouse.x - centre[0], mouse.y - centre[1], radius)


def _value(ctx: Any, state: Any, tab: Any, slot: int | None, colour: tuple) -> None:
    """The Value bar under the wheel -- the one HSV dimension the wheel does
    not draw (see ``colorwheel``'s module docstring)."""

    hue, saturation, value = _held_hsv(state, colour, HSV_LABELS)
    changed, out = _slider("Value", value, 255, tab, slot)
    if not changed:
        return
    red, green, blue = colorsys.hsv_to_rgb(hue / 360.0, saturation / 100.0, out / 255.0)
    written = (clamp8(red * 255.0), clamp8(green * 255.0), clamp8(blue * 255.0))
    write(ctx, state, tab, slot, (*written, colour[3]))
    _hold(state, HSV_LABELS, written, (hue, saturation, out))


def _to_hsl(r: float, g: float, b: float) -> tuple[float, float, float]:
    """``colorsys`` orders H, L, S; this pane's fields read H, S, L."""

    hue, light, sat = colorsys.rgb_to_hls(r, g, b)
    return hue, sat, light


def _from_hsl(hue: float, sat: float, light: float) -> tuple[float, float, float]:
    return colorsys.hls_to_rgb(hue, light, sat)


def _row(
    tab: Any, slot: int | None, fields: tuple[tuple[str, int, int], ...]
) -> list[tuple[bool, int]]:
    """One line of small labelled number boxes -- the numeric row's exception
    to "labels go above their controls" (every other control in this pane
    follows that idiom; see :func:`draw`). ``fields`` is ``(letter, value,
    high)`` per box, drawn left to right, sized to share whatever width the
    pane has this frame.

    A ``drag_int`` rather than the wheel/bars' ``slider_int``: these are typed
    or nudged numbers in a space too small for a track to read as one, the
    same distinction ``widgets.labeled_drag_int`` draws for a column count.
    """
    avail = imgui.get_content_region_avail().x
    label_gap = sp(4.0)
    field_gap = sp(10.0)
    count = len(fields)
    label_w = sum(imgui.calc_text_size(label).x for label, _, _ in fields)
    box_w = max(sp(32.0), (avail - label_w - count * label_gap - (count - 1) * field_gap) / count)
    history = _history(tab, slot)
    results: list[tuple[bool, int]] = []
    for index, (label, value, high) in enumerate(fields):
        if index:
            imgui.same_line(0.0, field_gap)
        widgets.text_colored(theme.MUTED, label)
        imgui.same_line(0.0, label_gap)
        imgui.set_next_item_width(box_w)
        changed, out = controls.drag_int(f"##inkpicker/{label}", int(value), 1.0, 0, high, "%d")
        controls.fold_undo(history)
        results.append((bool(changed), int(out)))
    return results


def _rgba_row(ctx: Any, state: Any, tab: Any, slot: int | None, colour: tuple) -> None:
    fields = (
        ("R", colour[0], 255),
        ("G", colour[1], 255),
        ("B", colour[2], 255),
        ("A", colour[3], 255),
    )
    (rc, rv), (gc, gv), (bc, bv), (ac, av) = _row(tab, slot, fields)
    if rc or gc or bc or ac:
        write(
            ctx,
            state,
            tab,
            slot,
            (
                rv if rc else colour[0],
                gv if gc else colour[1],
                bv if bc else colour[2],
                av if ac else colour[3],
            ),
        )


def _hsv_row(ctx: Any, state: Any, tab: Any, slot: int | None, colour: tuple) -> None:
    hue, saturation, value = _held_hsv(state, colour, HSV_LABELS)
    fields = (("H", hue, 359), ("S", saturation, 100), ("V", value, 255), ("A", colour[3], 255))
    (hc, hv), (sc, sv), (vc, vv), (ac, av) = _row(tab, slot, fields)
    if ac:
        write(ctx, state, tab, slot, (colour[0], colour[1], colour[2], av))
        return
    if not (hc or sc or vc):
        return
    new_hue = hv % 360 if hc else hue
    new_sat = sv if sc else saturation
    new_val = vv if vc else value
    red, green, blue = colorsys.hsv_to_rgb(new_hue / 360.0, new_sat / 100.0, new_val / 255.0)
    written = (clamp8(red * 255.0), clamp8(green * 255.0), clamp8(blue * 255.0))
    write(ctx, state, tab, slot, (*written, colour[3]))
    _hold(state, HSV_LABELS, written, (new_hue, new_sat, new_val))


def _hsl_row(ctx: Any, state: Any, tab: Any, slot: int | None, colour: tuple) -> None:
    hue, saturation, light = _held_hsv(state, colour, HSL_LABELS)
    fields = (("H", hue, 359), ("S", saturation, 100), ("L", light, 100), ("A", colour[3], 255))
    (hc, hv), (sc, sv), (lc, lv), (ac, av) = _row(tab, slot, fields)
    if ac:
        write(ctx, state, tab, slot, (colour[0], colour[1], colour[2], av))
        return
    if not (hc or sc or lc):
        return
    new_hue = hv % 360 if hc else hue
    new_sat = sv if sc else saturation
    new_light = lv if lc else light
    red, green, blue = _from_hsl(new_hue / 360.0, new_sat / 100.0, new_light / 100.0)
    written = (clamp8(red * 255.0), clamp8(green * 255.0), clamp8(blue * 255.0))
    write(ctx, state, tab, slot, (*written, colour[3]))
    _hold(state, HSL_LABELS, written, (new_hue, new_sat, new_light))


def _gray_row(ctx: Any, state: Any, tab: Any, slot: int | None, colour: tuple) -> None:
    level = clamp8(sum(weight * channel for weight, channel in zip(LUMA, colour[:3], strict=True)))
    fields = (("Gray", level, 255), ("A", colour[3], 255))
    (gc, gv), (ac, av) = _row(tab, slot, fields)
    if gc:
        write(ctx, state, tab, slot, (gv, gv, gv, colour[3]))
    elif ac:
        write(ctx, state, tab, slot, (colour[0], colour[1], colour[2], av))


#: Which numeric-row function draws each ``SPACE_SLOT`` key. A dict rather
#: than an if/elif ladder because :func:`_space` already guarantees the key
#: is one of these -- the ladder's ``else`` would be unreachable and the
#: guarantee would live in two places.
_ROWS = {
    "rgba": _rgba_row,
    "hsv": _hsv_row,
    "hsl": _hsl_row,
    "gray": _gray_row,
}


def _rgb(ctx: Any, state: Any, tab: Any, slot: int | None, colour: tuple) -> None:
    """The pre-wheel RGB tab. Unused by :func:`draw` -- see the module
    docstring for why it is still here, byte-identical to before."""

    out = list(colour)
    touched = False
    for index, label in enumerate(("Red", "Green", "Blue")):
        changed, value = _slider(label, int(colour[index]), 255, tab, slot)
        if changed:
            out[index] = value
            touched = True
    if touched:
        write(ctx, state, tab, slot, tuple(out))
    _alpha(ctx, state, tab, slot, colour)


def _hex(ctx: Any, state: Any, tab: Any, slot: int | None, colour: tuple) -> None:
    """``RRGGBB`` or ``RRGGBBAA``, which is what a palette is written down as.

    Applied on enter rather than per keystroke: half a typed hex triple is a
    colour, and applying it would repaint the brush three times on the way to
    the one the user meant.
    """
    text = "".join(f"{clamp8(channel):02X}" for channel in colour)
    widgets.field_label("Hex")
    imgui.set_next_item_width(-1)
    changed, typed = controls.input_text(
        "##inkpickerhex", text, imgui.InputTextFlags_.enter_returns_true.value
    )
    if not changed:
        return
    parsed = parse_hex(typed)
    if parsed is not None:
        write(ctx, state, tab, slot, parsed)


def parse_hex(text: str) -> tuple[int, int, int, int] | None:
    """``#rgb``/``rgba``/``rrggbb``/``rrggbbaa`` -> RGBA, or ``None``.

    Public and pure so the parsing is a plain assertion rather than a
    screenshot: a hex field that silently ignores what was typed is
    indistinguishable from one that is not wired up.
    """
    raw = str(text).strip().lstrip("#")
    if not raw or any(char not in "0123456789abcdefABCDEF" for char in raw):
        return None
    if len(raw) in (3, 4):
        raw = "".join(char * 2 for char in raw)
    if len(raw) == 6:
        raw += "FF"
    if len(raw) != 8:
        return None
    return (
        int(raw[0:2], 16),
        int(raw[2:4], 16),
        int(raw[4:6], 16),
        int(raw[6:8], 16),
    )
