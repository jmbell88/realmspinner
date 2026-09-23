"""Mason's Tools column: which transform tool is in hand, its snap, and the
placement ops that move or multiply a selection.

Clay's Tools panel split its old grab-bag of switches between the viewport
header (what changes *between* clicks) and the sidebar (what wants the
height, read down rather than flicked between) -- see that pane's own module
docstring for the argument. Only Mason's pivot actually made that move, onto
``mason_header`` (``header.py``'s ``_pivot_field``); what earns a *sidebar*
row here is what Clay's sidebar kept for the identical reason -- a list of
operations invoked once per press, not a switch flicked every few seconds.
Align, distribute and the two array ops are exactly that list, one level over
from Clay's Duplicate, Bake and Mirror: they read the selection's world boxes
and write a batch of new transforms or a batch of new nodes, in one press.
The transform tool grid and snap are between-clicks settings by the same
argument and belong on the header too, but that move has not happened --
both are still drawn here, below.

The 2026-09-23 audit's mason-04 found this docstring's previous wording
naming the tool grid together with pivot and snap as all belonging on
``mason_header``, describing a three-item move only one item of which had
actually happened -- and the pivot control drawn *here* besides its new home
in ``header.py``, so it appeared twice in the same frame. Corrected to
describe what this module actually draws, and the duplicate pivot control
removed below.

**Every op here is a pure function of ``mason.ops`` plus the document's own
mutators.** This file decides which axis, which mode, which count; the
arithmetic that decides where a box's edge lands is ``mason/ops.py``'s, not
this pane's -- the same boundary that module's own docstring draws between
itself and ``document.py``.

Every control that changes the document is disabled while a save is in
flight, ``clay_tools``'s reason verbatim: serialising reads the live document
on a task thread, and a control that restructured it mid-encode would write a
file describing a document that never existed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from imgui_bundle import imgui

from ..... import controls, icons, widgets
from .....manual import render as manual_render
from .....tokens import sp
from ... import assets as mason_assets
from ... import mode as mason_mode
from ... import state as mason_state
from ...engine import document as md
from ...engine import nodes as nd
from ...engine import ops as mops
from ...engine import scene

TOOL_ICONS = {
    "select": icons.SQUARE_DASHED,
    "move": icons.MOVE,
    "rotate": icons.ROTATE_CW,
    "scale": icons.SCALING,
}

AXES = (("x", "X"), ("y", "Y"), ("z", "Z"))
ALIGN_MODES = (("min", "Min"), ("centre", "Centre"), ("max", "Max"))


def draw(ctx: Any) -> None:
    """This pane's headings, on tinted blocks -- ``clay_tools.draw``'s
    reasoning, verbatim."""
    with widgets.section_blocks():
        _body(ctx)


def _body(ctx: Any) -> None:
    state = mason_mode.ensure(ctx)
    tab = mason_mode.active(ctx)
    widgets.section("Tools")
    manual_render.help_button(ctx, "mason-tools")
    if tab is None:
        widgets.muted("Open or start a scene to build in.")
        return

    imgui.begin_disabled(tab.saving)
    _tool_grid(ctx, state)
    imgui.dummy((0, sp(8)))
    # Pivot itself is not drawn here -- it moved to ``mason_header``
    # (``header.py``'s ``_pivot_field``); see this module's own docstring for
    # why drawing it here too (the 2026-09-23 audit's mason-04) was wrong.
    _snap(ctx, state)
    imgui.dummy((0, sp(8)))
    _placement(ctx, state, tab)
    imgui.end_disabled()


def _tool_grid(ctx: Any, state: Any) -> None:
    widgets.field_label("transform")
    width = widgets.grid_width(4)
    for index, (key, label, shortcut) in enumerate(mason_state.TOOLS):
        if controls.button(
            f"{TOOL_ICONS.get(key, icons.SQUARE_DASHED)}##masontool{key}",
            (width, sp(28)),
            selected=state.tool == key,
            tooltip=f"{label}  ({shortcut})",
        ):
            state.tool = key
        if index < len(mason_state.TOOLS) - 1:
            imgui.same_line()
    del ctx


def _snap(ctx: Any, state: Any) -> None:
    widgets.field_label("snap")
    changed, value = widgets.toggle(f"{icons.MAGNET} Snap", state.snap)
    if changed:
        state.snap = value
    imgui.begin_disabled(not state.snap)
    widgets.field_label("grid (m)")
    _, state.snap_translate = controls.input_float(
        "##masongrid", state.snap_translate, 0.25, 0.0, "%.3f"
    )
    widgets.field_label("angle (deg)")
    _, state.snap_rotate = controls.input_float(
        "##masonangle", state.snap_rotate, 5.0, 0.0
    )
    imgui.end_disabled()
    state.snap_translate = max(0.0, float(state.snap_translate))
    state.snap_rotate = max(0.0, float(state.snap_rotate))
    # A separate switch from the grid, ``mason_state``'s own reason: it
    # answers "put it on the surface below" rather than "put it on round
    # numbers", and a row of props wants both together as often as either
    # alone.
    changed, value = widgets.toggle(f"{icons.ARROW_DOWN} Drop to ground", state.snap_ground)
    if changed:
        state.snap_ground = value
    del ctx


def _world_boxes(ctx: Any, doc: md.MasonDoc, uids: list[int]) -> dict[int, tuple[Any, Any]]:
    source = mason_assets.ensure(ctx)
    out: dict[int, tuple[Any, Any]] = {}
    for uid in uids:
        box = scene.world_bounds(doc, source, uids=[uid])
        if box is not None:
            out[uid] = box
    return out


def _apply_deltas(doc: md.MasonDoc, deltas: dict[int, Any]) -> None:
    """Move every ``(uid, delta)`` pair, as **one** undo step.

    The 2026-09-14 audit's mason-02: this used to push one ``TransformEdit``
    per node with no ``mark``/``collapse_since`` around the loop, so Align,
    Distribute and Drop selection to ground each cost as many Ctrl+Z presses
    as nodes moved -- docs/manual/31-mason.md promises "Each of these lands as
    a single undo step", and every other multi-node mutator in
    ``mason_mode.py`` (``group_selected``, ``duplicate_selected``...) already
    folds the same way. This is the one call site all three buttons share.
    """
    mark = doc.mark()
    for uid, delta in deltas.items():
        node = doc.node(uid)
        if node is None:
            continue
        was = node.trs()
        translation = np.asarray(node.translation, dtype="f8") + np.asarray(delta, dtype="f8")
        doc.set_transform(uid, translation=translation, was=was)
    doc.collapse_since(mark)


#: Transient widget state for the align/array rows -- which axis, which mode,
#: what count and offset the next press uses. Not on ``MasonState``: that
#: dataclass's ``overlays`` is typed and tested as ``dict[str, bool]`` (what
#: the viewport draws over the scene), and stuffing an array count into it
#: would be exactly the "two lists of one thing" drift ``clay_tools``'s own
#: docstring warns against. Module-level and keyed by nothing -- like
#: ``clay_menu``'s param popup values, these are "what the next press starts
#: from", not part of the document and not worth a save/restore path.
_PENDING: dict[str, Any] = {
    "axis": "y",
    "mode": "min",
    "count": 4,
    "offset": [1.0, 0.0, 0.0],
    "degrees": 360.0,
}


def _placement(ctx: Any, state: Any, tab: Any) -> None:
    doc = tab.doc
    widgets.field_label("align / distribute")
    changed, axis_key = controls.segmented_choice("mason-align-axis", list(AXES), _PENDING["axis"])
    if changed:
        _PENDING["axis"] = axis_key
    changed, mode_key = controls.segmented_choice(
        "mason-align-mode", ALIGN_MODES, _PENDING["mode"]
    )
    if changed:
        _PENDING["mode"] = mode_key
    axis = {"x": 0, "y": 1, "z": 2}[_PENDING["axis"]]
    mode_key = _PENDING["mode"]

    uids = list(doc.selection)
    width = widgets.grid_width(2)
    align_ok = len(uids) >= 2
    if widgets.disabled_button(
        "Align##masonalign",
        align_ok,
        (width, 0),
        reason="Select at least 2 nodes to align.",
    ):
        boxes = _world_boxes(ctx, doc, uids)
        _apply_deltas(doc, mops.align(boxes, axis, mode_key))
    imgui.same_line()
    distribute_ok = len(uids) >= 3
    if widgets.disabled_button(
        "Distribute##masondistribute",
        distribute_ok,
        (width, 0),
        reason="Select at least 3 nodes to distribute.",
    ):
        boxes = _world_boxes(ctx, doc, uids)
        _apply_deltas(doc, mops.distribute(boxes, axis))
    if widgets.disabled_button(
        f"{icons.ARROW_DOWN} Drop selection to ground##masondrop",
        bool(uids),
        reason="Select at least 1 node to drop.",
    ):
        boxes = _world_boxes(ctx, doc, uids)
        _apply_deltas(doc, mops.drop_to_ground(boxes, terrain=doc.terrain))

    imgui.dummy((0, sp(8)))
    widgets.field_label("array")
    _array(ctx, state, doc)


def _over_max_placed(doc: md.MasonDoc, node: nd.Node, copies: int) -> int | None:
    """The document's node count after adding ``copies`` more duplicates of
    ``node``'s whole subtree, or ``None`` when that stays within
    :data:`scene.MAX_PLACED`.

    The 2026-09-14 audit's mason-01: an array count someone typed an extra
    zero into used to run straight through -- ``array_linear``/``array_radial``
    building every copy and ``_spawn_array`` calling ``copy_subtree`` on each
    one (150,000 copies measured at 1.18 s) -- before ``MasonDoc.add_nodes``
    ever got a chance to refuse, by which point the copies already existed and
    the refusal there just meant the work was wasted rather than avoided.
    Checked here first, with the same cheap structural count ``add_nodes``
    itself refuses on, so the button can toast and build nothing at all.

    The 2026-09-16 audit's mason-engine-01: this used to add ``copies`` --
    the number of *top-level* duplicates -- straight onto the current count,
    which undercounts whenever the array's source ``node`` is a GroupNode (or
    any node with children): each copy is a whole ``copy_subtree()`` of
    ``node``, not one node, so the real growth is ``copies`` times the
    source's own subtree size. Counted here the same way
    ``MasonDoc.add_nodes`` now counts it, so this pre-flight toast and the
    door it is guarding agree.
    """
    per_copy = len(list(nd.walk([node])))
    total = len(doc.all_nodes()) + copies * per_copy
    return total if total > scene.MAX_PLACED else None


def _array(ctx: Any, state: Any, doc: md.MasonDoc) -> None:
    """Duplicate the one selected node along a line or around a circle,
    through ``mason.ops.array_linear``/``array_radial`` -- the arithmetic that
    decides where each copy lands, applied here to fresh copies of the node
    ``copy_subtree`` makes, added as one undo step through ``add_nodes``."""
    uids = list(doc.selection)
    one = len(uids) == 1
    node = doc.node(uids[0]) if one else None

    widgets.field_label("count")
    _, count = controls.input_int("##masonarraycount", int(_PENDING["count"]), 1)
    _PENDING["count"] = max(1, count)

    widgets.field_label("offset (m)")
    _, offset = controls.input_float3("##masonarrayoffset", list(_PENDING["offset"]))
    _PENDING["offset"] = list(offset)

    width = widgets.grid_width(1)
    array_reason = "Select exactly 1 node to array."
    if (
        widgets.disabled_button(
            "Array (linear)##masonarraylinear", one, (width, 0), reason=array_reason
        )
        and node
    ):
        over = _over_max_placed(doc, node, _PENDING["count"] - 1)
        if over is not None:
            ctx.toast(
                f"That array would bring this scene to {over} nodes, past "
                f"the {scene.MAX_PLACED} limit -- refusing rather than "
                "building it.",
                "error",
            )
        else:
            trs_list = mops.array_linear(
                _PENDING["count"], _PENDING["offset"], base_trs=node.trs()
            )
            _spawn_array(doc, node, trs_list[1:])

    widgets.field_label("degrees")
    _, degrees = controls.input_float("##masonarraydegrees", float(_PENDING["degrees"]), 5.0)
    _PENDING["degrees"] = degrees

    if (
        widgets.disabled_button(
            "Array (radial)##masonarrayradial", one, (width, 0), reason=array_reason
        )
        and node
    ):
        over = _over_max_placed(doc, node, _PENDING["count"] - 1)
        if over is not None:
            ctx.toast(
                f"That array would bring this scene to {over} nodes, past "
                f"the {scene.MAX_PLACED} limit -- refusing rather than "
                "building it.",
                "error",
            )
        else:
            centre = node.translation
            trs_list = mops.array_radial(
                _PENDING["count"],
                centre=centre,
                axis=(0.0, 1.0, 0.0),
                degrees=_PENDING["degrees"],
                base_trs=node.trs(),
            )
            _spawn_array(doc, node, trs_list[1:])
    del state


def _spawn_array(doc: md.MasonDoc, node: nd.Node, trs_list: list[Any]) -> None:
    if not trs_list:
        return
    parent_uid = doc.parent_uid_of(node.uid)
    copies = []
    for translation, rotation, scale in trs_list:
        copy = nd.copy_subtree(node, fresh_uids=True)
        copy.translation = np.asarray(translation, dtype="f8")
        copy.rotation = np.asarray(rotation, dtype="f8")
        copy.scale = np.asarray(scale, dtype="f8")
        copies.append(copy)
    doc.add_nodes(copies, parent_uid=parent_uid, label="Array")
