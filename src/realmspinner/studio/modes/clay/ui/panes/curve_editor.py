"""The curve editor: a canvas for a lathe's profile, a sweep's outline and a tube's path.

Those three parameters are polylines of points, and the generic parameter loop
used to draw each as one read-only line of numbers. This is the widget that
replaces it, in the properties panel under the generator it belongs to:

* **lathe** -- the ``(radius, y)`` half-silhouette on the right of a vertical
  axis, with its mirror drawn as a ghost on the left so the shape reads as the
  solid it revolves into;
* **sweep** -- the closed outline, with any segments that cross another drawn in
  the warning colour (the manual's *uncaught figure-eight*: it survives every
  clamp and builds a self-intersecting solid);
* **tube** -- the 3D path in a chosen plane (XY, XZ or ZY), the third coordinate
  one numeric field away.

Interaction: drag a point; **Alt**-drag a point to pull a smooth handle pair out
of it, or drag a handle dot (**Shift** mirrors the opposite one); click a segment
to add a point on it (the curve's shape does not change); ``Delete`` or a
right-click removes one; numeric fields below edit the selected point and its
handles exactly.

**What each action does to the data is decided in** :mod:`~...curve_edit`
(``step`` is the whole gesture state machine, headless-tested); this module draws
and turns imgui's mouse state into one ``Pointer`` per frame. An edit goes through
``props.apply_generator_params`` -- the same clamp, rebuild and
``set_generator_params`` step the generic loop uses -- so the mesh previews live
**every frame of a drag** and the whole drag folds into one undo step
(``history.mark`` on press, ``collapse_since`` on release). While a drag is open the
canvas draws the *working* curve, not the stored one: the clamp may reorder,
dedupe or re-centre what the document keeps, and drawing that copy would slide the
point out from under the cursor. After release the stored, clamped params are what
is read.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from imgui_bundle import imgui

from ......kernels.mesh import primitives as bp
from ..... import controls, theme, widgets
from .....tokens import sp
from ... import curve_edit
from ...curve_edit import Curve, CurveUi, Pointer

#: Which parameter of which generator is a curve, and which key carries its handles.
KINDS: dict[str, str] = {"lathe": "profile", "sweep": "outline", "tube": "path"}
CURVE_KEYS: dict[str, str] = {
    "profile": "profile_handles",
    "outline": "outline_handles",
    "path": "path_handles",
}
HANDLE_KEYS = frozenset(CURVE_KEYS.values())

_AXIS_LABELS = {"profile": ("R", "Y"), "outline": ("X", "Y"), "path": ("X", "Y", "Z")}
_TITLES = {"profile": "profile", "outline": "outline", "path": "path"}
_HINTS = {
    "profile": "Drag a point. Alt-drag pulls a smooth handle out of it; click the line to add one; "
    "Delete removes. The left half is the mirror the lathe revolves.",
    "outline": "Drag a point. Alt-drag pulls a smooth handle out of it; click the line to add one; "
    "Delete removes. Red segments cross another.",
    "path": "Drag a point in this plane. Alt-drag pulls a smooth handle out of it; click the line "
    "to add one; Delete removes.",
}
CANVAS_H = 220.0
PAD = 0.12  # the fraction of the canvas left empty around a fitted curve


def draw(
    doc: Any,
    obj: Any,
    key: str,
    state: Any,
    apply: Any,
    *,
    ctx: Any = None,
) -> None:
    """The editor for *obj*'s curve parameter *key*, in the properties panel."""
    handle_key = CURVE_KEYS[key]
    defaults, _build = bp.GENERATORS[obj.generator]
    params = dict(defaults)
    params.update({k: v for k, v in obj.params.items() if k in defaults})
    ui = state.curve_ui.setdefault(obj.uid, CurveUi())
    closed = key == "outline"
    stored = Curve.from_params(params[key], params.get(handle_key), closed=closed)
    curve = ui.working if (ui.working is not None and ui.drag is not None) else stored
    locked = doc.lock_refusal(obj.uid) is not None

    widgets.field_label(_TITLES[key])
    if key == "path":
        plane = widgets.combo(
            "##curve-plane", ui.plane, [(name, name) for name in curve_edit.PLANES], sp(80)
        )
        ui.plane = plane
    axes = curve_edit.PLANES[ui.plane] if key == "path" else (0, 1)

    width = max(float(imgui.get_content_region_avail().x), 60.0)
    height = sp(CANVAS_H)
    origin = imgui.get_cursor_screen_pos()
    imgui.invisible_button("##curve-canvas", (width, height))
    hovered = bool(imgui.is_item_hovered())
    ox, oy = float(origin.x), float(origin.y)

    if ui.scale <= 0.0 or ui.centre is None:
        _fit(ui, curve, key, axes, width, height)
    to_screen, to_model = _transforms(ui, ox, oy, width, height)

    # --- input --------------------------------------------------------------
    io = imgui.get_io()
    if hovered:
        wheel = float(io.mouse_wheel)
        if wheel:
            before = to_model(imgui.get_mouse_pos())
            ui.scale = max(ui.scale * (1.1**wheel), 1e-3)
            to_screen, to_model = _transforms(ui, ox, oy, width, height)
            after = to_model(imgui.get_mouse_pos())
            ui.centre = (
                ui.centre[0] + before[0] - after[0],
                ui.centre[1] + before[1] - after[1],
            )
            to_screen, to_model = _transforms(ui, ox, oy, width, height)
    if (hovered or ui.drag is None) and imgui.is_mouse_dragging(2):
        delta = imgui.get_mouse_drag_delta(2)
        imgui.reset_mouse_drag_delta(2)
        ui.centre = (ui.centre[0] - delta.x / ui.scale, ui.centre[1] + delta.y / ui.scale)
        to_screen, to_model = _transforms(ui, ox, oy, width, height)

    if not locked:
        pointer = Pointer(
            pos=to_model(imgui.get_mouse_pos()),
            per_pixel=1.0 / ui.scale,
            pressed=hovered and imgui.is_mouse_clicked(0),
            down=imgui.is_mouse_down(0),
            released=imgui.is_mouse_released(0),
            alt=bool(io.key_alt),
            shift=bool(io.key_shift),
            delete=hovered and imgui.is_key_pressed(imgui.Key.delete),
            right_pressed=hovered and imgui.is_mouse_clicked(1),
        )
        updated, events = curve_edit.step(ui, curve, key, pointer, axes)
        curve = apply_events(doc, obj, key, ui, curve, updated, events, apply, ctx)

    ui.selected = min(ui.selected, len(curve.points) - 1)
    _paint(ui, curve, key, axes, to_screen, ox, oy, width, height, locked)

    widgets.muted_wrapped(_HINTS[key])
    if controls.small_button("Fit view##curve-fit", tooltip="Frame the whole curve"):
        _fit(ui, curve, key, axes, width, height)
    _fields(doc, obj, key, ui, curve, apply, ctx, locked)


# --- editing -----------------------------------------------------------------------


def apply_events(
    doc: Any,
    obj: Any,
    key: str,
    ui: CurveUi,
    curve: Curve,
    updated: Curve,
    events: list[str],
    apply: Any,
    ctx: Any = None,
) -> Curve:
    """Do what one ``curve_edit.step`` owes: open, apply, fold. -> the curve now current.

    Separate from :func:`draw` so the glue between the state machine and the
    document -- *a drag is one undo step* -- is provable without imgui: the tests
    drive ``step`` and this, exactly as the pane does once a frame.
    """
    if "begin" in events:
        ui.mark = doc.history.mark()
        ui.working = curve.copy()
    if "change" in events:
        ui.working = updated if ui.drag is not None else None
        _commit(doc, obj, key, updated, apply, ctx)
        curve = updated
    if "end" in events:
        _end_gesture(doc, ui, key)
    return curve


def _commit(doc: Any, obj: Any, key: str, curve: Curve, apply: Any, ctx: Any) -> bool:
    """Send *curve* through the generic generator door: clamp, rebuild, one step."""
    defaults = bp.GENERATORS[obj.generator][0]
    edited = dict(defaults)
    edited.update({k: v for k, v in obj.params.items() if k in defaults})
    edited[key] = [list(p) for p in curve.points]
    edited[CURVE_KEYS[key]] = curve.stored_handles()
    return bool(apply(doc, obj, edited, ctx=ctx))


def _end_gesture(doc: Any, ui: CurveUi, key: str) -> None:
    """Fold everything the gesture pushed into one undo step, and read the clamp back."""
    if ui.mark >= 0:
        doc.history.collapse_since(ui.mark)
        top = doc.history.top
        if top is not None:
            top.label = f"Edit {_TITLES[key]}"
    ui.mark = -1
    ui.working = None


def _fields(
    doc: Any, obj: Any, key: str, ui: CurveUi, curve: Curve, apply: Any, ctx: Any, locked: bool
) -> None:
    """The selected point and its two handles, as exact numbers."""
    index = ui.selected
    if not 0 <= index < len(curve.points):
        widgets.muted("Select a point to type its position.")
        return
    labels = _AXIS_LABELS[key]
    imgui.begin_disabled(locked)
    rows = (
        (f"point {index + 1}", "##curve-pt", curve.points[index], None),
        ("in handle", "##curve-in", curve.handles[index][0], 0),
        ("out handle", "##curve-out", curve.handles[index][1], 1),
    )
    for title, label, values, side in rows:
        widgets.field_label(title)
        changed, typed = controls.input_vec(label, list(values), labels)
        controls.fold_undo(doc.history)
        if not changed:
            continue
        new = curve.copy()
        if side is None:
            new.points[index] = [float(v) for v in typed]
        else:
            new.handles[index][side] = [float(v) for v in typed]
        _commit(doc, obj, key, new, apply, ctx)
    imgui.end_disabled()


# --- view --------------------------------------------------------------------------------


def _fit(ui: CurveUi, curve: Curve, key: str, axes: tuple[int, int], w: float, h: float) -> None:
    flat = np.asarray(curve.flat(), dtype="f8")[:, list(axes)]
    lo, hi = flat.min(axis=0), flat.max(axis=0)
    if key == "profile":  # the mirror is drawn too, so frame both halves
        lo[0], hi[0] = -float(max(abs(lo[0]), abs(hi[0]))), float(max(abs(lo[0]), abs(hi[0])))
    span = np.maximum(hi - lo, 1e-6)
    ui.centre = (float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2))
    ui.scale = float(min(w / span[0], h / span[1]) * (1.0 - 2 * PAD))


def _transforms(ui: CurveUi, ox: float, oy: float, w: float, h: float) -> tuple[Any, Any]:
    cx, cy = ui.centre
    scale = ui.scale

    def to_screen(p: tuple[float, float]) -> tuple[float, float]:
        return ox + w / 2 + (p[0] - cx) * scale, oy + h / 2 - (p[1] - cy) * scale

    def to_model(p: Any) -> tuple[float, float]:
        x, y = (p.x, p.y) if hasattr(p, "x") else p
        return cx + (x - ox - w / 2) / scale, cy - (y - oy - h / 2) / scale

    return to_screen, to_model


def _colour(token: int, alpha: float = 1.0) -> int:
    return imgui.get_color_u32(theme.rgba(token, alpha))


def _paint(
    ui: CurveUi,
    curve: Curve,
    key: str,
    axes: tuple[int, int],
    to_screen: Any,
    ox: float,
    oy: float,
    w: float,
    h: float,
    locked: bool,
) -> None:
    draw_list = imgui.get_window_draw_list()
    draw_list.push_clip_rect((ox, oy), (ox + w, oy + h), True)
    draw_list.add_rect_filled((ox, oy), (ox + w, oy + h), _colour(theme.ELEV_1))
    draw_list.add_rect((ox, oy), (ox + w, oy + h), _colour(theme.EDGE))

    muted = _colour(theme.MUTED, 0.5)
    if key == "profile":
        # The revolve axis.
        top, bottom = to_screen((0.0, 1e4)), to_screen((0.0, -1e4))
        draw_list.add_line(top, bottom, _colour(theme.MUTED, 0.6), 1.0)
    flat = [[p[axes[0]], p[axes[1]]] for p in curve.flat()]
    closed = key == "outline"
    bad = _crossing_segments(ui, flat, closed) if closed else frozenset()
    count = len(flat) if closed else len(flat) - 1
    line = _colour(theme.ACCENT)
    warn = _colour(theme.WARN)
    for i in range(max(count, 0)):
        a, b = flat[i], flat[(i + 1) % len(flat)]
        colour = warn if i in bad else line
        draw_list.add_line(to_screen(tuple(a)), to_screen(tuple(b)), colour, 2.0)
        if key == "profile":
            ghost = (to_screen((-a[0], a[1])), to_screen((-b[0], b[1])))
            draw_list.add_line(ghost[0], ghost[1], muted, 1.5)

    for i, point in enumerate(curve.points):
        anchor = (point[axes[0]], point[axes[1]])
        sx, sy = to_screen(anchor)
        selected = i == ui.selected
        for side in (0, 1):
            offset = curve.handles[i][side]
            if not any(offset):
                continue
            tip = to_screen((anchor[0] + offset[axes[0]], anchor[1] + offset[axes[1]]))
            draw_list.add_line((sx, sy), tip, _colour(theme.MUTED, 0.8), 1.0)
            draw_list.add_circle_filled(tip, sp(4.0), _colour(theme.TEXT), 12)
        radius = sp(5.5 if selected else 4.0)
        draw_list.add_circle_filled(
            (sx, sy), radius, _colour(theme.ACCENT if selected else theme.TEXT), 14
        )
        draw_list.add_circle((sx, sy), radius, _colour(theme.EDGE), 14, 1.0)
    if locked:
        draw_list.add_rect_filled((ox, oy), (ox + w, oy + h), _colour(theme.ELEV_1, 0.35))
    draw_list.pop_clip_rect()


def _crossing_segments(ui: CurveUi, flat: list[list[float]], closed: bool) -> frozenset[int]:
    """The flattened outline's self-crossing segments, recomputed only when it moved."""
    key = (closed, tuple(map(tuple, flat)))
    if ui.cross_key != key:
        ui.cross_key = key
        ui.cross_value = frozenset(curve_edit.crossings(flat, closed))
    return ui.cross_value
