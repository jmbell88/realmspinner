"""The inspector's Size card: the one place a shape's size is typed.

Width, height and depth in the display unit (with the unit chooser on the card's
heading row), a Lock aspect switch and where the shape's box sits in the world.
The arithmetic is :mod:`~realmspinner.studio.modes.clay.transform_edit`'s -- a
size is the mesh's local extent times ``|scale|``, so an edit lands on the scale
that reproduces it -- and the apply door is Properties' own ``_apply_transform``,
so a refusal (a flat axis, an all-zero scale) is the toast it always was.

**What is new is where the box stays.** Properties scaled about the shape's
pivot, which for every primitive is its middle: type a taller Height into a box
standing on the ground and it grew in both directions, its lower half through the
floor. Typing a size is the first thing anyone does to a new shape, and a shape
that sinks for it reads as broken. A height change here therefore keeps the
shape's **lowest world point where it was** -- the translation is corrected in
the same undo step -- while width and depth keep the middle, which a scale about
the pivot already does. (An edit to width or depth under Lock aspect also moves
the height; the base stays put then too, for the same reason.)

**The field commits when it is left** (``commit=True``), so typing "2.5" is one
resize to 2.5 and not three to 2, 2. and 2.5 -- one undo step per typed value --
and a click on the next control after typing both commits it and lands on its own
target (``test_first_try_path`` pins the lost-click case).

After any resize, a shape whose box has grown out of the picture asks the camera
to frame the model (:func:`mode.frame_after_add`'s rule, the one a new shape
follows), so a size typed past the edge of the view is not a shape that vanished.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from imgui_bundle import imgui

from .....kernels.geom3d import units
from .... import anchors, controls, widgets
from ....tokens import sp
from .. import mode as clay_mode
from .. import ops as clay_ops
from .. import transform_edit
from .panes import props

#: The unit chooser's width on the heading row, in design pixels.
UNIT_W = 72.0


def resize(ctx: Any, doc: Any, obj: Any, new_scale: Any, was: Any = None) -> bool:
    """Apply *new_scale* to *obj* as **one** undo step, keeping its lowest world
    point where it was whenever the height changed. -> whether it landed.

    The shift is :func:`~.ops._apply_deltas` with a world-space Y delta -- the
    door Drop to Ground and Snap use, which already converts a parented shape's
    delta back into its own frame -- so a child of a scaled parent is anchored on
    the world floor and not on its local one. Everything it pushes sits under one
    ``mark``/``collapse_since``, so a Ctrl+Z puts back the size *and* the height.
    """
    from .....kernels.mesh import ops as bops

    uid = obj.uid
    keeps_base = abs(float(new_scale[1]) - float(obj.scale[1])) > 0.0
    before = bops.world_box(obj, world=doc.world_matrix(uid)) if keeps_base else None
    mark = doc.history.mark()
    try:
        landed = props._apply_transform(doc, obj, ctx, scale=new_scale, was=was)
        if landed and before is not None:
            after = bops.world_box(doc.by_uid(uid), world=doc.world_matrix(uid))
            if after is not None:
                drop = float(before[0][1]) - float(after[0][1])
                if abs(drop) > 1e-12:
                    clay_ops._apply_deltas(ctx, doc, {uid: np.array([0.0, drop, 0.0])})
    finally:
        doc.history.collapse_since(mark)
    if landed:
        clay_mode.frame_after_add(ctx, doc, doc.by_uid(uid), first=False)
    return landed


def draw(ctx: Any, state: Any, doc: Any, obj: Any) -> None:
    """The card: heading with the unit chooser, W/H/D, Lock aspect, the box."""
    from .....kernels.mesh import ops as bops

    ui = props._display_state(state)
    unit = props._unit_of(state)
    widgets.section("Size")
    imgui.same_line(
        max(imgui.get_cursor_pos_x() + imgui.get_content_region_avail().x - sp(UNIT_W), 0.0)
    )
    picked = widgets.combo(
        "##clay-length-unit",
        unit,
        props._UNIT_OPTIONS,
        sp(UNIT_W),
        tooltip="The unit sizes and positions are shown in. Storage stays in metres.",
    )
    if picked != unit:
        ui.length_unit = unit = picked

    mesh = obj.mesh
    box = bops.world_box(obj, mesh, world=doc.world_matrix(obj.uid))
    if box is None:
        widgets.muted("This shape has no mesh to size.")
        return
    extent = transform_edit.local_extent(mesh)
    shown = units.vec_to_display(transform_edit.size_of(extent, obj.scale), unit)
    was = tuple(v.copy() for v in obj.trs())

    edited, typed = controls.input_vec(
        "##size##insp",
        list(shown),
        ("W", "H", "D"),
        commit=True,
        tooltip="Width, height and depth. Type a number and press Enter; a taller "
        "height keeps the shape standing where it is.",
    )
    anchors.mark("clay/props/size")
    controls.fold_undo(doc.history)
    _, ui.size_lock_aspect = controls.checkbox(
        "Lock aspect##insp-lock",
        ui.size_lock_aspect,
        tooltip="Keep the proportions when you change one size.",
    )
    if edited:
        axis = next((i for i in range(3) if typed[i] != shown[i]), None)
        if axis is not None:
            new_scale = transform_edit.resized_scale(
                extent,
                obj.scale,
                axis,
                units.from_display(typed[axis], unit),
                lock_aspect=ui.size_lock_aspect,
            )
            if new_scale is not None:
                resize(ctx, doc, obj, new_scale, was)
    for axis in range(3):
        reason = transform_edit.axis_refusal(extent, axis)
        if reason is not None:
            widgets.muted(reason)
    widgets.muted(bounds_line(box, unit))


def _g(value: float) -> str:
    """*value* to millimetres' worth of digits, with no "-0" and no float dust."""
    return f"{round(value, 3) + 0.0:.6g}"


def bounds_line(box: Any, unit: str) -> str:
    """``"Spans x -0.5 to 0.5 · y 0 to 3 · z -0.5 to 0.5 m"``: where the shape's
    world box sits, in the display unit. Pure, so the sentence is checkable."""
    lo, hi = box

    def part(axis: int) -> str:
        a, b = (units.to_display(float(v[axis]), unit) for v in (lo, hi))
        return f"{'xyz'[axis]} {_g(a)} to {_g(b)}"

    return f"Spans {part(0)} · {part(1)} · {part(2)} {unit}"
