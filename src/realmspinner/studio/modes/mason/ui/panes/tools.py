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
    # One resolve for every selected node (the 2026-10-03 audit's mason-17: a
    # ``world_bounds`` call per uid ran a whole-scene resolve each, so Align /
    # Distribute / Drop over a big selection froze the frame for seconds).
    return scene.world_bounds_by_owner(doc, mason_assets.ensure(ctx), uids=uids)


#: ``MasonView._parent_basis``'s own guard against a parent scaled flat on
#: some axis, restated here (view.py's own comment names why: a determinant
#: too close to zero has an inverse that blows up into non-finite numbers
#: before ``np.linalg.inv`` even gets a chance to raise on it).
_SINGULAR_DET_EPS = 1e-12


def _parent_inverse_basis(doc: md.MasonDoc, uid: int) -> Any:
    """The inverse of ``uid``'s parent's world rotation/scale, or ``None``
    when it has none (a parent scaled flat on some axis).

    ``MasonView._parent_basis``'s own shape, restated here because
    ``_apply_deltas`` below builds a *world*-space delta the same way a gizmo
    drag does, and both need the identical conversion into the node's own
    parent-local space before it can be added to a local translation.
    Identity at the root, which is the common case and costs nothing.
    """
    parent_uid = doc.parent_uid_of(uid)
    if parent_uid is None:
        return np.eye(3)
    try:
        found = scene.resolved_for(doc, parent_uid)
    except ValueError:
        # mason-03, the 2026-09-13 audit: a document past MAX_PLACED must not
        # crash this lookup -- ``MasonView._parent_basis``'s own reason.
        found = None
    if found is None:
        return np.eye(3)
    basis = np.asarray(found.world, dtype="f8")[:3, :3]
    det = float(np.linalg.det(basis))
    if not np.isfinite(det) or abs(det) < _SINGULAR_DET_EPS:
        return None
    try:
        inverse = np.linalg.inv(basis)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(inverse).all():
        return None
    return inverse


def _apply_deltas(doc: md.MasonDoc, deltas: dict[int, Any]) -> None:
    """Move every ``(uid, delta)`` pair, as **one** undo step.

    The 2026-09-14 audit's mason-02: this used to push one ``TransformEdit``
    per node with no ``mark``/``collapse_since`` around the loop, so Align,
    Distribute and Drop selection to ground each cost as many Ctrl+Z presses
    as nodes moved -- docs/manual/31-mason.md promises "Each of these lands as
    a single undo step", and every other multi-node mutator in
    ``mason_mode.py`` (``group_selected``, ``duplicate_selected``...) already
    folds the same way. This is the one call site all three buttons share.

    The 2026-09-26 audit's mason-mode-08: ``delta`` comes from
    ``mops.align``/``distribute``/``drop_to_ground``, all three of which work
    from *world* boxes (``_world_boxes``, above) -- but this used to add that
    world delta straight onto ``node.translation``, which is expressed in the
    node's own *parent* space. Identical to the child of a scaled group for a
    gizmo drag before ``MasonView._apply_drag``'s own fix: a child of a group
    scaled 2x moved by half the world distance a click asked for, or the
    wrong direction entirely under a rotated one. Converted through the
    parent's inverse basis first, the same call the gizmo drag makes.
    """
    #
    # The 2026-10-03 audit's mason-16: a node whose ancestor is *also* being
    # moved was moved twice -- once by its own delta, and again because the
    # ancestor's move already carries every child -- so a child 5 m up under a
    # prop 5 m up dropped to world y = -4.0 instead of 0.5. Dropped from
    # ``deltas`` the way ``group_selected`` and the gizmo drag drop a node with
    # a selected ancestor. (An ancestor with no box -- a group -- is not in
    # ``deltas`` and does not move, so its children keep their own deltas.)
    movers = set(mason_mode._selection_without_selected_ancestor(doc, list(deltas)))
    mark = doc.mark()
    for uid, delta in deltas.items():
        if uid not in movers:
            continue
        node = doc.node(uid)
        if node is None:
            continue
        inverse = _parent_inverse_basis(doc, uid)
        if inverse is None:
            # A parent whose own basis has no inverse (scaled flat on some
            # axis) is skipped rather than written with a non-finite
            # transform -- ``objout._normal_matrix``'s own guard, one door
            # over.
            continue
        was = node.trs()
        local_delta = inverse @ np.asarray(delta, dtype="f8")
        translation = np.asarray(node.translation, dtype="f8") + local_delta
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
        # The 2026-10-03 audit's mason-22: through the ground node's own world
        # matrix, as the context menu's drop already does.
        _apply_deltas(
            doc,
            mops.drop_to_ground(
                boxes, terrain=doc.terrain, terrain_world=doc.terrain_world()
            ),
        )

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


def _over_resolved_placed(doc: md.MasonDoc, node: nd.Node, copies: int) -> int | None:
    """The document's *resolved* size (what :func:`scene.resolve` actually
    expands to) after adding ``copies`` more duplicates of ``node``, or
    ``None`` when that stays within :data:`scene.MAX_PLACED`.

    The 2026-09-26 audit's mason-mode-02 (was High): :func:`_over_max_placed`
    above is exactly ``MasonDoc._check_max_placed``'s own tree-side count --
    it counts arraying a ``PrefabNode`` instance as ``copies`` plain nodes,
    never as what each one expands to every time the scene resolves, draws or
    exports (``document.py``'s ``resolved_growth`` docstring). An array of a
    prefab instance could sail past that check and still hit
    ``MasonDoc.add_nodes``'s own ``_check_resolved_placed`` backstop, which
    raises past ``_spawn_array`` uncaught. Charged the same way
    ``mode.place_prefab`` already charges one placement: every copy of
    ``node`` resolves to the same size ``node`` itself does (a fresh copy
    changes only uid and transform, never structure or refs), so one call to
    :meth:`~.document.MasonDoc.resolved_growth` times ``copies`` is exact
    without having to build any of them first.
    """
    try:
        per_copy = doc.resolved_growth([node])
    except ValueError:
        # scene.resolved_count refuses rather than counting an oversized walk
        # to completion once either half alone would already exceed
        # MAX_PLACED -- treat that refusal the same as "over", since it is.
        return scene.MAX_PLACED + 1
    total = doc.resolved_total() + copies * per_copy
    return total if total > scene.MAX_PLACED else None


def _over_either_ceiling(doc: md.MasonDoc, node: nd.Node, copies: int) -> int | None:
    """Both ceilings an array press must respect, tree-side
    (:func:`_over_max_placed`) first since it is the cheaper of the two, then
    resolved (:func:`_over_resolved_placed`, mason-mode-02) -- whichever fires
    first is the number the toast reports."""
    over = _over_max_placed(doc, node, copies)
    if over is not None:
        return over
    return _over_resolved_placed(doc, node, copies)


def _array(ctx: Any, state: Any, doc: md.MasonDoc) -> None:
    """Duplicate the one selected node along a line or around a circle,
    through ``mason.ops.array_linear``/``array_radial`` -- the arithmetic that
    decides where each copy lands, applied here to fresh copies of the node
    ``copy_subtree`` makes, added as one undo step through ``add_nodes``."""
    uids = list(doc.selection)
    node = doc.node(uids[0]) if len(uids) == 1 else None
    # The 2026-09-26 audit's mason-mode-10: a ``TerrainNode`` carries no
    # heightfield of its own -- it only refers to the document-singleton
    # ``doc.terrain`` (``nodes.TerrainNode``'s own docstring) -- so arraying
    # it would build several outliner rows all resolving and drawing the same
    # one ground a second, third and fourth time. Refused at the button
    # rather than built.
    is_terrain = isinstance(node, nd.TerrainNode)
    one = len(uids) == 1 and not is_terrain

    widgets.field_label("count")
    _, count = controls.input_int("##masonarraycount", int(_PENDING["count"]), 1)
    _PENDING["count"] = max(1, count)

    widgets.field_label("offset (m)")
    _, offset = controls.input_float3("##masonarrayoffset", list(_PENDING["offset"]))
    _PENDING["offset"] = list(offset)

    width = widgets.grid_width(1)
    array_reason = (
        "The ground can't be arrayed -- there is only one terrain."
        if is_terrain
        else "Select exactly 1 node to array."
    )
    if (
        widgets.disabled_button(
            "Array (linear)##masonarraylinear", one, (width, 0), reason=array_reason
        )
        and node
    ):
        over = _over_either_ceiling(doc, node, _PENDING["count"] - 1)
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
            _spawn_array(ctx, doc, node, trs_list[1:])

    widgets.field_label("degrees")
    _, degrees = controls.input_float("##masonarraydegrees", float(_PENDING["degrees"]), 5.0)
    _PENDING["degrees"] = degrees

    if (
        widgets.disabled_button(
            "Array (radial)##masonarrayradial", one, (width, 0), reason=array_reason
        )
        and node
    ):
        over = _over_either_ceiling(doc, node, _PENDING["count"] - 1)
        if over is not None:
            ctx.toast(
                f"That array would bring this scene to {over} nodes, past "
                f"the {scene.MAX_PLACED} limit -- refusing rather than "
                "building it.",
                "error",
            )
        else:
            # The 2026-09-26 audit's mason-mode-04: this used to read
            # ``node.translation`` -- exactly ``node.trs()``'s own
            # translation, the ``base_trs`` passed below -- so
            # ``array_radial``'s ``t0 - centre`` was zero for every copy and
            # all of them landed on top of the original. ``base_trs`` is in
            # the node's *parent* space (``_spawn_array`` adds every copy
            # under that same parent), so the origin of that space -- the
            # parent's own pivot -- is the centre that actually spins the
            # node's own offset from it around a circle, the way a fence post
            # a few metres from a hub spins around the hub rather than in place.
            centre = (0.0, 0.0, 0.0)
            trs_list = mops.array_radial(
                _PENDING["count"],
                centre=centre,
                axis=(0.0, 1.0, 0.0),
                degrees=_PENDING["degrees"],
                base_trs=node.trs(),
            )
            _spawn_array(ctx, doc, node, trs_list[1:])
    del state


def _spawn_array(ctx: Any, doc: md.MasonDoc, node: nd.Node, trs_list: list[Any]) -> None:
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
    try:
        doc.add_nodes(copies, parent_uid=parent_uid, label="Array")
    except ValueError as exc:
        # The 2026-09-26 audit's mason-mode-02 (was High): ``add_nodes``
        # checks both ceilings *before* attaching anything (see its own
        # docstring), so this backstop never leaves a half-built array
        # attached -- but before this existed, its ``ValueError`` propagated
        # out of a button press uncaught rather than the friendly toast
        # ``_over_either_ceiling``'s precheck gives the common case.
        # ``group_selected``'s mason-mode-01 fix, one door over.
        ctx.toast(f"Could not build that array: {exc}", "error")
