"""Properties' bodies: what the Inspector's cards, folds and pages are built from.

**Not a pane any more.** The right column is one slot, the Inspector (``inspector.py``,
slot ``clay-inspector``), and this module is the box of widgets it draws from. The
four-tab pane this file used to be -- Object, Material, Document and UV, with an
Adjust card over the viewport and a median row in an element mode -- is gone: the
Object tab's body is the Inspector's Shape states and the element state
(``ui/inspector_elements.py``), and the other three tabs are pushed pages. Every
function below stays, by name, because the Inspector calls it and a long tail of
tests drives it directly: Position & turn is ``_transform(show_unit=False,
show_scale=False, show_size=False)``, Advanced is ``scale_fields``, Shape settings is
``_generator``, Group & parent is ``_relations``, the Colours & textures page is
``_material``, the Model & history page is ``_document``, and the element state's
median row is ``_element_median``. The Size card is its own
(``ui/inspector_size.py``).

**The parameter widgets are generated from the registry, not written by hand.**
``primitives.GENERATORS`` maps a name to ``(defaults, builder)`` and every
default dictionary is a complete call, so the panel enumerates it and binds one
widget per key. A sixteenth primitive therefore needs no edit here at all -- which
is the entire reason that registry is data rather than a chain of ``if``s, and
is asserted by a test that registers a fake generator and looks for its
parameters.

A change to a parameter regenerates the mesh as **one** ``MeshEdit`` plus the
props edit that recorded the new parameters, so a Ctrl+Z takes the object back
to the shape it had. That only works while ``generator`` is not None; the first
topology edit clears it and this panel switches to a vertex and face count with
a "frozen" note -- which is the state an *imported* object arrives in too.

Every control here is disabled while a save is in flight, for the reason the
tool panel states.
"""

from __future__ import annotations

import logging
import weakref
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from imgui_bundle import imgui

from ......kernels.geom3d import math3d as m3
from ......kernels.geom3d import units
from ......kernels.mesh import document as clay_document
from ......kernels.mesh import elements as el
from ......kernels.mesh import measure, regen
from ......kernels.mesh import primitives as bp
from ......kernels.mesh import uv as uv_projection
from ..... import anchors, controls, icons, tokens, verbs, widgets
from .....tokens import sp
from ... import doc_io, element_move, transform_edit
from ... import mode as clay_mode
from ... import ops as clay_ops
from ...state import ClayState, worn_slot
from . import header as clay_header
from . import outliner as clay_outliner

# The swatch code left for ``swatches.py`` with the swatch row (it is the Inspector's
# Colour card's now). Re-exported under the names this module always had,
# because the tests that drive a press reach for ``clay_props._pick_slot`` and
# ``clay_props._swatch_colour`` by those names.
from .swatches import SWATCH, _pick_slot, _swatch, _swatch_colour  # noqa: F401

log = logging.getLogger(__name__)

# How a parameter's type decides its widget. Read off the *default value*,
# because a registry entry carries no schema and does not need one: a float
# default means a float field, and a tuple means one field per component.
_STEP = {"segments": 1, "rings": 1, "sides": 1}


def _document(ctx: Any, tab: Any, doc: Any) -> None:
    """The document as a whole: its file, what is in it, the history, how the next
    import reads a file, and the ways out.

    This is the Inspector's Model & history page (``inspector.py`` pushes it): the
    file row, the counts, Undo/Redo and the step popover, the exports and the last
    export's id.

    The counts are of what leaves the document -- visible objects only, the same
    set the exporters write -- so the triangle line is a promise about the
    exported file.
    """
    # "Model file", the shape of every other bridge's heading ("Drawing file",
    # "Map file", "Song file", "Atlas file").
    widgets.section("Model file")
    widgets.document_header(
        tab,
        new=lambda: clay_mode.new_document(ctx),
        open_=lambda: clay_mode.ask_open(ctx),
        save=lambda: clay_mode.save(ctx, tab),
        save_as=lambda: clay_mode.save_as(ctx, tab),
    )
    # The header's last item is its path/status line, not the Save button, so
    # the ring is the file block as a whole.
    anchors.mark("clay/props/save")
    imgui.dummy((0, sp(tokens.SP_2)))
    widgets.field_label("counts")
    widgets.muted(doc_io.objects_line(doc))
    widgets.muted(doc_io.geometry_line(doc))
    imgui.dummy((0, sp(tokens.SP_2)))
    _history(ctx, tab)
    imgui.dummy((0, sp(tokens.SP_2)))
    _import_settings(ctx)
    imgui.dummy((0, sp(tokens.SP_2)))
    _outputs(ctx, tab)
    widgets.recent_files(
        clay_mode.recent_paths(ctx), lambda path: clay_mode.open_path(ctx, Path(path))
    )


def _history(ctx: Any, tab: Any) -> None:
    """Undo and Redo, on screen -- and the step count as the history popover.

    ``clay_mode.undo``/``redo`` rather than ``tab.doc.undo()`` here, so the
    button and the chord carry the same side effects (see the history block in
    that module).

    **Under a live G/R/S or UV rotate/scale the block is drawn greyed, with the
    header's own sentence** (the 2026-10-10 audit's clay-58): ``clay_mode.undo``,
    ``redo`` and ``step_history`` return early then, and ``widgets.history_block``
    only knows ``tab.busy``, so its buttons stayed lit and did nothing. The ids are
    the block's own, so a drag that ends hands the controls back with no flicker.
    """
    if clay_mode._gesture_live(ctx, tab) and not getattr(tab, "saving", False):
        _history_while_dragging(tab)
        return
    widgets.history_block(
        ctx,
        tab,
        key="clay",
        undo=lambda: clay_mode.undo(ctx, tab),
        redo=lambda: clay_mode.redo(ctx, tab),
        step=lambda index: clay_mode.step_history(ctx, tab, index),
    )


def _history_while_dragging(tab: Any) -> None:
    """``widgets.history_block``'s three controls, all greyed for a live gesture."""
    why = clay_header.history_why(busy=False, gesture=True, idle="")
    width = widgets.grid_width(2)
    widgets.disabled_button(f"{icons.UNDO} Undo##clay-undo", False, (width, 0), reason=why)
    imgui.same_line()
    widgets.disabled_button(f"{icons.REDO} Redo##clay-redo", False, (width, 0), reason=why)
    widgets.disabled_button(
        f"{tab.doc.history.total} step(s)##clay-history", False, (-1, 0), reason=why
    )


def _import_settings(ctx: Any) -> None:
    """Import Mesh... and the units and up-axis the import uses.

    **The row that runs the import lives here** (it was the Add menu's, in the menu
    strip that is gone): the same picker as File > Import Mesh... and the Ctrl+K row,
    ``clay_mode.ask_import_mesh``. It opens the file as a *new* document tab, as a file
    dropped on the viewport does. It greys while this document is being written, as
    every other control on the page does, and says so.

    ``ClayState.import_scale``/``import_up`` are what the picker and a file dropped on
    the viewport both read (``clay_mode.import_mesh_path``'s own defaults) -- set here
    and remembered for the next import in either form.
    """
    state = clay_mode.ensure(ctx)
    tab = state.active
    saving = bool(tab is not None and tab.saving)
    widgets.field_label("import a mesh")
    if widgets.disabled_button(
        f"{icons.FOLDER_OPEN} Import Mesh...",
        not saving,
        reason=widgets.DOCUMENT_SAVING_WHY,
        tooltip="Open a .glb, .obj, .stl or .ply as a new document. Units and up axis "
        "are set just below.",
    ):
        clay_mode.ask_import_mesh(ctx)
    widgets.field_label("import units")
    scale_key = f"{state.import_scale:g}"
    picked = widgets.combo(
        "##clay-import-scale", scale_key, doc_io.IMPORT_SCALE_OPTIONS, sp(90)
    )
    if picked != scale_key:
        state.import_scale = float(picked)
    widgets.field_label("import up axis")
    state.import_up = widgets.combo(
        "##clay-import-up", state.import_up, doc_io.IMPORT_UP_OPTIONS, sp(90)
    )


def _outputs(ctx: Any, tab: Any) -> None:
    """The ways out of the document: the library, a mesh file, a picture.

    Two genuinely different destinations: the exact geometry into the library as
    an asset (everything downstream -- rigging, posing, sprite sheets -- is a
    function of it), or a plain file on disk the library never sees.
    """
    # The one heading every mode's exits are under. See ``inker_bridge``'s
    # ``_pipeline`` for why the five of them agree on a name.
    widgets.section("Take it somewhere")
    doc = tab.doc
    why = doc_io.outputs_why(doc, tab.saving, clay_mode._gesture_live(ctx, tab))
    ready = not why

    if widgets.primary_button(
        f"{icons.DOWNLOAD} {verbs.EXPORT_TO_LIBRARY}",
        enabled=ready,
        reason=why,
        tooltip=doc_io.EXPORT_LIBRARY_TOOLTIP,
    ):
        clay_mode.export_asset(ctx, tab)
    anchors.mark("clay/props/export")

    # Two labelled buttons rather than "Export File..." plus a bare "OBJ": the
    # first spelling wrote GLB without saying so, and a format is exactly the
    # thing a user reading the row needs to see before pressing.
    if widgets.disabled_button(
        f"{icons.DOWNLOAD} Export GLB...", ready, reason=why, tooltip=doc_io.EXPORT_GLB_TOOLTIP
    ):
        clay_mode.export_mesh_file(ctx, tab, "glb")
    # Wrapped, not run on: three labelled buttons are wider than the 300 px column
    # the inspector hosts this row in, and the third ("Save screenshot...") was
    # clipped to "Save scre".
    widgets.same_line_or_wrap(widgets.button_width("Export OBJ..."))
    if widgets.disabled_button(
        "Export OBJ...", ready, reason=why, tooltip=doc_io.EXPORT_OBJ_TOOLTIP
    ):
        clay_mode.export_mesh_file(ctx, tab, "obj")
    shot = f"{icons.CAMERA} Save screenshot..."
    widgets.same_line_or_wrap(widgets.button_width(shot))
    if widgets.disabled_button(
        shot,
        ready,
        reason=why,
        tooltip=doc_io.SCREENSHOT_TOOLTIP,
    ):
        clay_mode.save_screenshot(ctx, tab)

    for line in export_lines(tab):
        widgets.muted(line)


def export_lines(tab: Any) -> list[str]:
    """The muted lines under the export buttons: the library row last minted and
    the last file written to disk.

    The 2026-10-07 audit's clay-09: the manual promises "a line names the last
    file written", but only ``Last exported as <job id>`` (the library export)
    was ever drawn -- an Export GLB/OBJ or a screenshot left its path in a toast
    and nowhere else. Both lines show when both exports happened.
    """
    lines: list[str] = []
    if tab.job_id:
        lines.append(f"Last exported as {tab.job_id}")
    if getattr(tab, "exported_path", ""):
        lines.append(f"Last file written: {tab.exported_path}")
    return lines


def _unit_of(state: ClayState | None) -> str:
    """The unit position and size are shown in (``ClayState.length_unit``)."""
    ui = _display_state(state)
    return ui.length_unit if ui.length_unit in dict(units.LENGTH_UNITS) else units.DEFAULT_UNIT


#: Why the median row is greyed with nothing picked.
NO_MEDIAN_WHY = "Pick something in the viewport first, then type where its middle should be."


def _element_median(doc: Any, state: ClayState | None, ctx: Any = None) -> None:
    """The selection's world median as three numbers you can type into.

    ``commit=True``: the field reports on Enter or on leaving it, so typing
    "120" is one move to 120 and not three steps to 1, 12 and 120 -- the rule
    every undoable field in this pane follows (the Inspector's name field, and
    the transform fields via ``fold_undo``). The move itself is
    :func:`element_move.move_elements`, the door the G drag also ends in: one
    undo step however many objects hold a selection, parented objects handled,
    hidden ones skipped.
    """
    median = element_median_cached(doc)
    unit = _unit_of(state)
    # Always drawn: with nothing picked the row is greyed and says why, rather than
    # vanishing and pulling every card under it up the moment the first pick lands.
    shown = units.vec_to_display(median if median is not None else (0.0, 0.0, 0.0), unit)
    widgets.field_label("median")
    edited, typed = controls.input_vec(
        "##median##bmedian",
        list(shown),
        ("X", "Y", "Z"),
        commit=True,
        enabled=median is not None,
        reason=NO_MEDIAN_WHY,
        tooltip="The middle of the selected elements, in world space. Type a "
        "coordinate to move them there.",
    )
    controls.fold_undo(doc.history)
    if edited and median is not None:
        refused: list[Exception] = []
        commit_median(doc, median, shown, typed, unit, refused)
        if ctx is not None:
            for error in refused:
                clay_ops.toast(ctx, str(error))


#: ``doc -> {readout name: (key, value)}``, one slot per readout. A weak key, so
#: a closed document takes its entry with it (``viewport_hints._MEASURE_CACHE``'s
#: own shape).
_READOUT_CACHE: weakref.WeakKeyDictionary[Any, dict[str, tuple[Any, Any]]] = (
    weakref.WeakKeyDictionary()
)


def readout_key(doc: Any, extra: Any = None) -> tuple[Any, ...] | None:
    """What the element readouts below are memoised on: the document's change
    counter and its element mode (plus *extra*, the display unit for the one
    that prints it). ``None`` when the document has no ``rev`` -- no memo, measure
    fresh, as ``viewport_hints.measure_line`` does.

    ``rev`` covers a changed selection as well as a changed mesh
    (``select``/``set_element_sel``/``set_element_mode`` each ``touch``), and a
    live drag touches it every frame, so a drag still recomputes -- once a frame,
    not several.
    """
    rev = getattr(doc, "rev", None)
    if rev is None:
        return None
    return (rev, doc.element_mode, extra)


def _memoised(doc: Any, name: str, extra: Any, compute: Any) -> Any:
    key = readout_key(doc, extra)
    if key is None:
        return compute()
    slots = _READOUT_CACHE.setdefault(doc, {})
    cached = slots.get(name)
    if cached is not None and cached[0] == key:
        return cached[1]
    value = compute()
    slots[name] = (key, value)
    return value


def element_median_cached(doc: Any) -> Any:
    """:func:`element_move.element_median_world`, memoised on :func:`readout_key`.

    The 2026-10-07 audit's clay-18: the Properties readouts re-ran every frame
    (``affected_verts`` 31 ms and an uncached ``triangulate`` for the area 194-204 ms
    at 200k triangles), where the HUD's measure line was already memoised.
    """
    return _memoised(doc, "median", None, lambda: element_move.element_median_world(doc))


def commit_median(
    doc: Any,
    median: Any,
    shown: Any,
    typed: Any,
    unit: str,
    refused: list[Exception] | None = None,
) -> bool:
    """Move the selection so its median is where the user typed. -> whether it moved.

    Only the axes that were typed into go back through the unit: converting an
    untouched axis out and back would let a unit's rounding nudge it, and a
    centimetre display of 0.0125 m must not turn a Y-only edit into an X move.
    Each object the document refused leaves its error in *refused*, for the
    caller to toast (the 2026-10-07 audit's clay-14).
    """
    delta = [
        0.0
        if typed[i] == shown[i]
        else units.from_display(typed[i], unit) - float(median[i])
        for i in range(3)
    ]
    return element_move.move_elements(doc, delta, refused=refused)


def element_measure_text(doc: Any, unit: str = units.DEFAULT_UNIT) -> str | None:
    """:func:`_element_measure_text`, memoised on :func:`readout_key` -- see
    :func:`element_median_cached` for the incident."""
    return _memoised(doc, "measure", unit, lambda: _element_measure_text(doc, unit))


def _element_measure_text(doc: Any, unit: str = units.DEFAULT_UNIT) -> str | None:
    """A read-only measurement of the element selection, or ``None``.

    Edge mode: the summed length of the selected edges. Face mode: the summed
    area of the selected faces -- :func:`~.kernels.mesh.measure.face_area`, the
    function the HUD's ``area`` line and the agent's ``clay_measure`` both call,
    so the three cannot disagree about one face. Vertex mode has the HUD's
    distance and angle already, and object mode has no element selection.
    Hidden objects are skipped for ``element_summary_text``'s reason.
    """
    from ......kernels.mesh.selection import _element_pickable

    if doc.element_mode not in ("edge", "face"):
        return None
    total = 0.0
    found = False
    for uid, sel in doc.element_sel.items():
        try:
            obj = doc.by_uid(uid)
        except KeyError:
            continue
        if not _element_pickable(obj):
            continue
        world = doc.world_matrix(uid)
        if doc.element_mode == "face":
            if not len(sel.faces):
                continue
            found = True
            total += measure.face_area(obj.mesh, sel.faces, world)
        else:
            if not len(sel.edges):
                continue
            found = True
            points = element_move.apply_affine(world, obj.mesh.positions.astype("f8"))
            pairs = np.asarray(sel.edges, dtype="i8")
            total += float(
                np.linalg.norm(points[pairs[:, 1]] - points[pairs[:, 0]], axis=1).sum()
            )
    if not found:
        return None
    if doc.element_mode == "face":
        return f"area  {total:.4f} m\u00b2"
    return f"length  {units.to_display(total, unit):.4f} {unit}"


def _clear_parent_reason(obj: Any) -> str:
    """Why "Clear parent" is greyed, or ``""`` (the 2026-10-07 audit's clay-75)."""
    return "" if obj.parent is not None else "This object has no parent."


def _set_parent(ctx: Any, doc: Any, uid: int, parent: int | None) -> None:
    """``doc.set_parent(..., keep_world=True)``, refused as a toast.

    ``_relations``'s combo already excludes every uid that would make
    ``set_parent`` refuse a cycle, so this is defensive rather than the
    expected path -- a refusal goes through :func:`~.clay.ops.toast`, the door
    every other refusal in this mode uses.
    """
    from ......kernels.mesh.elements import OpError

    try:
        doc.set_parent(uid, parent, keep_world=True)
    except OpError as error:
        clay_ops.toast(ctx, str(error))


def _relations(ctx: Any, doc: Any, obj: Any) -> None:
    """Parent, and a way out of it (tranche 3: scene structure).

    The combo excludes every one of *obj*'s own descendants
    (``ClayDoc.descendants``) -- offering one would let a click ask
    ``set_parent`` for a cycle it refuses anyway, and a control that visibly
    offers a choice it is about to refuse is worse than one that never shows
    it. ``keep_world=True`` throughout: reparenting through this panel never
    visibly moves the object, only which frame the "Local" transform fields
    below are read in.
    """
    widgets.field_label("relations")
    # The 2026-09-26 audit's clay-document-05: ``ClayDoc.descendants`` calls
    # ``children_of`` once per node in the subtree, and that method scans
    # every object in the document each time -- 0.40 s/frame reproduced
    # against a 4,095-child parent, on a combo this pane rebuilds on every
    # selection. ``clay_outliner.fast_descendants`` answers the identical
    # question from a parent->children map built once per ``doc.rev`` (see
    # its own docstring) instead of a linear scan per node.
    descendants = set(clay_outliner.fast_descendants(doc, obj.uid))
    options = [("0", "(none)")] + [
        (str(other.uid), other.name or f"object {other.uid}")
        for other in doc.objects
        if other.uid != obj.uid and other.uid not in descendants
    ]
    current = "0" if obj.parent is None else str(obj.parent)
    picked = widgets.labeled_combo(
        "parent",
        current,
        options,
        help_text="Reparenting keeps this object's world position -- only "
        "the local numbers below, and which frame they are read in, change.",
    )
    if widgets.disabled_button(
        "Clear parent##clearparent",
        obj.parent is not None,
        reason=_clear_parent_reason(obj),
        tooltip="Detach this object from its parent. It stays where it is in the world.",
    ):
        picked = "0"
    if picked != current:
        _set_parent(ctx, doc, obj.uid, None if picked == "0" else int(picked))


def rig_lines(doc: Any, obj: Any) -> tuple[str, str]:
    """``(binding, hint)``: what the Rig card says *obj* is bound to, in plain words.

    Pure, so the sentences are checkable without imgui. A bone says so; a weighted mesh
    names its bones; anything else says which bone it moves with, or that none does.
    """
    from ......kernels.mesh import rig

    by_uid = doc._uid_map()
    if obj.skin is not None:
        names = [by_uid[j].name for j in obj.skin.joints if j in by_uid]
        shown = ", ".join(names[:4]) + (f" and {len(names) - 4} more" if len(names) > 4 else "")
        count = len(obj.skin.joints)
        return (
            f"Weighted to {count} bone{'s' if count != 1 else ''}: {shown}.",
            "Its points move with those bones by weight. Its own position is ignored when "
            "it is exported, so move the bones, not the mesh.",
        )
    if obj.bone:
        above = rig.bone_parent(by_uid, obj.uid)
        parent = f" under {by_uid[above].name!r}" if above is not None else " (a root bone)"
        return (
            f"A bone{parent}.",
            "Its origin is the joint's pivot. Shapes parented under it move with it.",
        )
    home = rig.nearest_bone(by_uid, obj.uid)
    if home is not None:
        return (
            f"Moves with the bone {by_uid[home].name!r}.",
            "Bound rigidly: every point follows that one bone at full weight.",
        )
    return ("Not bound to a bone.", "Parent it under a bone to make it move with one.")


def _rig(ctx: Any, doc: Any, obj: Any) -> None:
    """The Rig card: the bone mark, what the shape is bound to, and the way out of a skin.

    The mark and the unbind go through the op registry (``mark-bone``, ``unmark-bone``,
    ``unbind-skin``), so this card, the outliner's row menu, Ctrl+K and the agent run the
    same code and are refused for the same reasons.
    """
    changed, wanted = controls.checkbox(
        f"{icons.BONE} bone##rigbone",
        bool(obj.bone),
        tooltip="Mark this object as a bone: its origin is a joint's pivot and its own "
        "transform is the joint's rest pose.",
    )
    if changed:
        clay_ops.run(ctx, doc, clay_ops.get("mark-bone" if wanted else "unmark-bone"))
    binding, hint = rig_lines(doc, obj)
    widgets.muted_wrapped(binding)
    widgets.muted_wrapped(hint)
    if obj.skin is not None:
        unbind = clay_ops.get("unbind-skin")
        why = clay_ops.reason_for(unbind, doc)
        if widgets.disabled_button(
            f"{icons.LINK} Unbind Skin##rigunbind",
            not why,
            reason=why,
            tooltip=unbind.hint,
        ):
            clay_ops.run(ctx, doc, unbind)
    if any(o.bone for o in doc.objects):
        _weight_button(ctx, doc)


def _weight_button(ctx: Any, doc: Any) -> None:
    """The document-level door to Blender's automatic weights, under the Rig card."""
    tab = clay_mode.active(ctx)
    why = clay_mode.rig_blocked_why(ctx, tab) if tab is not None and tab.doc is doc else (
        "Open or start a document first."
    )
    if widgets.disabled_button(
        f"{icons.LINK} Weight with Blender...##rigweights",
        not why,
        reason=why,
        tooltip="Skin the visible shapes that are not bones to these bones with Blender's "
        "automatic weights, so the shapes bend with them. It merges the shapes into one mesh and "
        "takes a few minutes; Undo puts them back.",
    ):
        clay_mode.rig_with_blender(ctx, tab)


#: What a bare ``_transform(doc, obj)`` -- no pane, no ``ClayState`` -- shows its
#: display state in. Several tests drive the door with nothing but a document.
_BARE_STATE: ClayState | None = None

_UNIT_OPTIONS = [(key, key) for key, _ in units.LENGTH_UNITS]


def euler_key(state: ClayState | None, obj_uid: int) -> tuple[str, int]:
    """The ``euler_cache`` key for an object: ``(active tab uid, object uid)``.

    The 2026-10-07 audit's clay-23: the cache is one dict on the app-level
    ``ClayState`` and was keyed on the bare object uid, but uids are only unique
    inside one document -- two separately authored files both carry 1 and 2 -- so
    a rotation typed into one tab's object was shown for the other tab's.
    """
    tab = getattr(state, "active", None)
    return (getattr(tab, "uid", ""), obj_uid)


def _display_state(state: ClayState | None) -> ClayState:
    global _BARE_STATE
    if state is not None:
        return state
    if _BARE_STATE is None:
        _BARE_STATE = ClayState()
    return _BARE_STATE


#: Why a bone's Position, Rotation and Scale fields are greyed in Pose mode.
POSE_FIELDS_WHY = (
    "Pose mode shows the pose, and these fields are the bone's rest transform, which posing "
    "never writes. Pose it with the gizmo or G, R, S, or leave Pose mode to edit the rest."
)


def transform_lock(doc: Any, obj: Any) -> str:
    """Why *obj*'s Position, Rotation and Scale fields cannot be edited, or ``""``.

    Two reasons, asked up front so the fields can be greyed once with the sentence rather than
    refused afterwards: a bone in Pose mode (the 2026-10-14 audit, finding clay-panes-04), and a
    shape weighted to a skeleton (finding clay-panes-05, where every keystroke that reached
    ``set_transform`` toasted the document's refusal and the field snapped back).
    """
    if doc.pose_mode and obj.bone:
        return POSE_FIELDS_WHY
    return doc.transform_refusal(obj)


def _shown_trs(doc: Any, obj: Any) -> tuple[Any, Any, Any]:
    """The transform the fields display: the *posed* one for a bone in Pose mode, where the
    viewport shows it, else the object's own."""
    if doc.pose_mode and obj.bone:
        return doc.posed_trs(obj, doc.pose_overrides())
    return obj.trs()


def _apply_transform(doc: Any, obj: Any, ctx: Any, **fields: Any) -> bool:
    """``doc.set_transform`` with the panel's refusal handling. -> whether it landed.

    ``set_transform`` refuses a transform the document cannot hold (an all-zero
    scale) with an ``OpError``; uncaught, the pane's guard would replace
    Properties with "stopped drawing" for the rest of the frame. Catch it, toast
    it, leave the field showing the value the user typed. ``ctx`` is
    optional -- several tests in ``tests/modes/clay/`` drive this door straight
    with a bare ``(doc, obj)``, no pane and no ``ctx`` to toast through, so
    with none given the refusal is re-raised rather than swallowed.
    """
    from ......kernels.mesh.elements import OpError

    # The 2026-10-11 audit's clay-18: Position, Rotation, Scale and the Size card all
    # land here, and a transform typed during a live G/R/S or gizmo drag pushed a step
    # the drag's commit then reverted against the head it started from. The fields are
    # greyed by the Inspector (``inspector._one``); this is the door under them, for a
    # caller that reaches it another way. Silent: the greyed card says why, and a toast
    # per keystroke is the failure this door's docstring already avoids.
    if ctx is not None and clay_ops.gesture_live(ctx):
        return False
    if doc.pose_mode and obj.bone:
        # The 2026-10-14 audit, finding clay-panes-04: in Pose mode a bone's fields showed its
        # rest transform and wrote it, while the viewport showed the pose -- typing a value for
        # a keyed property changed the document and nothing on screen. The fields are greyed
        # (:func:`transform_lock`); this is the door under them, and it does not write the rest
        # pose from Pose mode, which posing never does.
        if ctx is None:
            raise OpError(POSE_FIELDS_WHY)
        return False
    try:
        doc.set_transform(obj.uid, **fields)
    except OpError as error:
        if ctx is None:
            raise
        clay_ops.toast(ctx, str(error))
        return False
    return True


def _transform(
    doc: Any,
    obj: Any,
    *,
    ctx: Any = None,
    state: ClayState | None = None,
    show_unit: bool = True,
    show_scale: bool = True,
    show_size: bool = True,
) -> None:
    # ``show_*`` are the inspector's: its Size card owns the unit combo and the
    # one sizing control, and Scale sits behind an "Advanced" disclosure of its
    # own (:func:`scale_fields`), so "Position & turn" draws this block with all
    # three off. Every default is today's block, which is what the element-mode
    # legacy body and the tests that drive this door by name still get.
    #
    # Tranche 3: scene structure. TRS is local to the parent now, and a root's
    # local TRS *is* its world TRS (``document.py``'s module docstring) --
    # which is what keeps an unparented object's fields, and their labels,
    # exactly as they always were. "Local" on the section heading and on
    # every field beneath it: the numbers say where the object sits relative
    # to its parent, not where it sits in the viewport.
    #
    # Each field below is two literal input_vec calls (one per parent state)
    # rather than one f-string-built label: ``test_ux_consistency_pass4.py``'s
    # allow-list is a static source scan for a literal string argument, and
    # an f-string is invisible to it -- it would not flag anything, but it
    # would also silently drop this row off the label-above-control
    # inventory instead of keeping it an accounted-for, visible exception
    # (see that file's own clay_props.py entries).
    # Same id suffix (``##bt``/``##bs``/``##br``) either way, so reparenting
    # mid-edit does not reset the field's own imgui state.
    ui = _display_state(state)
    unit = ui.length_unit if ui.length_unit in dict(units.LENGTH_UNITS) else units.DEFAULT_UNIT
    parented = obj.parent is not None
    widgets.field_label("local transform" if parented else "transform")
    # Position and size are *shown* in this unit and stored in metres; nothing
    # about the document changes with it. Rotation and scale are unitless.
    if show_unit:
        picked = widgets.combo(
            "##clay-length-unit",
            unit,
            _UNIT_OPTIONS,
            sp(72),
            tooltip="The unit position and size are shown in. Storage stays in metres.",
        )
        if picked != unit:
            ui.length_unit = unit = picked
    was = tuple(v.copy() for v in obj.trs())
    changed = False
    lock = transform_lock(doc, obj)
    gate: dict[str, Any] = {"enabled": False, "reason": lock} if lock else {}
    if lock:
        widgets.muted_wrapped(lock)
    src_t, src_r, src_s = _shown_trs(doc, obj)
    shown = units.vec_to_display(src_t, unit)
    if parented:
        edited, typed = controls.input_vec(
            "local position##bt", list(shown), ("X", "Y", "Z"), **gate
        )
    else:
        edited, typed = controls.input_vec("position##bt", list(shown), ("X", "Y", "Z"), **gate)
    # The 2026-09-07 audit's clay-01: these three fields fired ``set_transform``
    # -- an unconditional ``history.push`` -- on every keystroke, same as the
    # material sliders below already fold. ``InputFloat3``/``InputFloat4`` fire
    # per keystroke like any imgui text field, so typing a multi-digit number
    # into Position pushed one undo step per digit and a lone Ctrl+Z only took
    # the last character back rather than the whole edit.
    controls.fold_undo(doc.history)
    # Only the axes that were typed into go back through the unit: converting
    # an untouched axis out and back would let a unit's rounding drift it.
    translation = [
        float(obj.translation[i]) if typed[i] == shown[i] else units.from_display(typed[i], unit)
        for i in range(3)
    ]
    changed |= edited
    if show_scale:
        edited, scale = _scale_row(doc, obj, lock=lock)
        changed |= edited
    else:
        scale = [float(v) for v in obj.scale]
    # Rotation is Euler degrees in the app's one rotation order (XYZ, the MCP
    # surface's), which is what lets the panel and an agent's ``clay_transform``
    # read the same three numbers. The object stores a quaternion and
    # decomposing one is not stable frame to frame, so the angles shown are
    # cached per object until something else moves the quaternion
    # (``transform_edit``'s docstring).
    euler_at = euler_key(ui, obj.uid)
    if doc.pose_mode and obj.bone:
        # A pose is not typed, so its angles stay out of the cache that remembers what was.
        euler = m3.euler_xyz_from_quat(src_r)
    else:
        euler = transform_edit.displayed_euler(ui.euler_cache, euler_at, obj.rotation)
    if parented:
        edited, typed_euler = controls.input_vec(
            "local rotation (deg)##br", list(euler), ("X", "Y", "Z"), **gate
        )
    else:
        edited, typed_euler = controls.input_vec(
            "rotation (deg)##br", list(euler), ("X", "Y", "Z"), **gate
        )
    controls.fold_undo(doc.history)
    widgets.help_marker(
        "Euler angles in degrees, turned X then Y then Z -- the same numbers "
        "an agent's clay_transform takes. The object itself stores an XYZW "
        "quaternion, which the viewer gizmo and the pose files share."
    )
    rotation = m3.quat_from_euler_xyz(typed_euler) if edited else [float(v) for v in obj.rotation]
    changed |= edited
    if changed:
        # ``was`` is the values the fields started from. imgui writes the new
        # ones into the widget's own state as they are typed, so reading
        # "before" off the object here would compare a value against itself and
        # record an empty step -- which is the trap ``set_transform``'s ``was``
        # argument exists for.
        landed = _apply_transform(
            doc, obj, ctx, translation=translation, rotation=rotation, scale=scale, was=was
        )
        if landed and edited:
            transform_edit.remember_euler(
                ui.euler_cache, euler_at, doc.by_uid(obj.uid).rotation, typed_euler
            )
    if show_size:
        _dimensions(doc, obj, ctx=ctx, state=ui)


def _scale_row(doc: Any, obj: Any, *, lock: str = "") -> tuple[bool, list[float]]:
    """The three scale boxes, folded -> ``(edited, the numbers shown)``.

    Two literal calls, one per parent state, for the reason ``_transform``'s
    opening comment gives (the label-above-control scan reads literals). *lock* is
    :func:`transform_lock`'s sentence: the boxes are greyed with it, and show the posed
    scale for a bone in Pose mode.
    """
    gate: dict[str, Any] = {"enabled": False, "reason": lock} if lock else {}
    shown = list(_shown_trs(doc, obj)[2])
    if obj.parent is not None:
        edited, scale = controls.input_vec("local scale##bs", shown, ("X", "Y", "Z"), **gate)
    else:
        edited, scale = controls.input_vec("scale##bs", shown, ("X", "Y", "Z"), **gate)
    controls.fold_undo(doc.history)
    return edited, list(scale)


def scale_fields(doc: Any, obj: Any, *, ctx: Any = None) -> bool:
    """:func:`_transform`'s scale boxes on their own, applied. -> whether it landed.

    The inspector's "Advanced" disclosure: scale compares the shape with the mesh
    as first made, which is not the same number as its size, and almost nobody
    wants it -- so it is one fold away rather than a third row of three boxes
    under Position.
    """
    was = tuple(v.copy() for v in obj.trs())
    edited, scale = _scale_row(doc, obj, lock=transform_lock(doc, obj))
    if not edited:
        return False
    return _apply_transform(doc, obj, ctx, scale=scale, was=was)


#: *mesh* -> its local ``(lo, hi)``, weak on the mesh so a replaced or closed mesh takes
#: its entry with it. ``Mesh`` is frozen and its arrays are read-only (``mesh.py``), so
#: identity is the whole of the key: an edit builds a new ``Mesh`` and misses.
_LOCAL_BOUNDS: weakref.WeakKeyDictionary[Any, tuple[np.ndarray, np.ndarray]] = (
    weakref.WeakKeyDictionary()
)


def local_bounds(mesh: Any) -> tuple[np.ndarray, np.ndarray]:
    """``mesh.bounds`` memoised on the mesh's identity.

    The 2026-10-09 second audit's clay-68: the Size card (and the Model page's size
    row) ran two full ``positions.min``/``max`` passes per frame -- 7 ms on a 130k
    vertex shape -- against ``world_box``'s own docstring, which promises O(1) "after
    the local bounds". The bounds only change when the mesh does.
    """
    from ......kernels.mesh import mesh as bm

    cached = _LOCAL_BOUNDS.get(mesh)
    if cached is None:
        cached = _LOCAL_BOUNDS[mesh] = bm.bounds(mesh)
    return cached


def measured(
    obj: Any, mesh: Any, world: Any
) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray] | None:
    """``(world_box, local_extent)`` for the size rows, or ``None`` for an empty mesh.

    The world box is :func:`~.ops.world_box`'s own answer (the camera's framing reads
    it, and the two must not disagree), given a two-point stand-in for *mesh* whose
    box is the cached local one, so it does not walk the positions again.
    """
    from ......kernels.mesh import ops as bops

    if len(mesh.positions) == 0:
        return None
    lo, hi = local_bounds(mesh)
    stand_in = SimpleNamespace(positions=np.array([lo, hi], dtype="f4"))
    box = bops.world_box(obj, stand_in, world=world)  # type: ignore[arg-type]
    return (box, hi - lo) if box is not None else None


def _dimensions(doc: Any, obj: Any, *, ctx: Any = None, state: ClayState | None = None) -> None:
    """How big the thing is -- editable -- and where its world box sits.

    **Width / height / depth are the mesh's local extent times
    ``|scale|``**, because that is the one number that maps back to a scale
    without ambiguity: editing an axis sets ``scale[i] = new / extent[i]``
    (``transform_edit.resized_scale``). A rotated object's world box is a
    different quantity -- no scale reproduces it -- so it stays below as a
    read-only line, ``world bounds``.

    The world line is :func:`~.ops.world_box`'s answer rather than a second
    measurement here, so it and the camera's framing cannot disagree about one
    object, with
    ``world=doc.world_matrix(obj.uid)`` (tranche 3: scene structure) so a
    parented object reports where it sits rather than a root at its parent's
    place.

    An axis the mesh has no extent on (a plane's height) is read-only in effect:
    an edit to it is ignored and the reason is printed under the row. One
    greyed-out box inside a three-box field is not something imgui offers, and a
    toast per keystroke is the failure a refusal would otherwise cause.
    """
    ui = _display_state(state)
    unit = ui.length_unit if ui.length_unit in dict(units.LENGTH_UNITS) else units.DEFAULT_UNIT
    mesh = obj.mesh
    measure = measured(obj, mesh, doc.world_matrix(obj.uid))
    if measure is None:
        return
    box, extent = measure
    size = transform_edit.size_of(extent, obj.scale)
    shown = units.vec_to_display(size, unit)
    was = tuple(v.copy() for v in obj.trs())

    widgets.field_label("size")
    edited, typed = controls.input_vec("size##bz", list(shown), ("W", "H", "D"))
    controls.fold_undo(doc.history)
    _, ui.size_lock_aspect = controls.checkbox("lock aspect##bzlock", ui.size_lock_aspect)
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
                _apply_transform(doc, obj, ctx, scale=new_scale, was=was)
    for axis in range(3):
        reason = transform_edit.axis_refusal(extent, axis)
        if reason is not None:
            widgets.muted(reason)

    w, h, d = (units.to_display(float(v), unit) for v in (box[1] - box[0]))
    widgets.muted(f"world bounds  {w:.3f} x {h:.3f} x {d:.3f} {unit}  (W x H x D)")
    widgets.help_marker(
        "The object's world-space bounding box, after its transform. A rotated "
        "object reports the box around its rotated box, which is the same "
        "measurement the camera frames against."
    )


def _generator(doc: Any, obj: Any, *, ctx: Any = None, state: ClayState | None = None) -> None:
    if obj.generator is None:
        # A frozen object: edited topology, or imported. The panel says what
        # the object is rather than pretending it still has parameters that
        # would silently discard the edit if changed.
        widgets.field_label("mesh")
        line = (
            f"frozen -- {len(obj.mesh.positions)} vertices, "
            f"{len(obj.mesh.starts) - 1} faces"
        )
        widgets.muted(line)
        return
    entry = bp.GENERATORS.get(obj.generator)
    if entry is None:
        widgets.muted(f"unknown generator '{obj.generator}'")
        return

    defaults, build = entry
    widgets.field_label(obj.generator.replace("_", " "))
    params = dict(defaults)
    params.update({k: v for k, v in obj.params.items() if k in defaults})
    edited = dict(params)
    changed = False
    for key, default in defaults.items():
        # A name line per param (2026-09-08 consistency pass): the block
        # label above names the generator, not its individual fields, and
        # the old beside-the-box text was the only place a param's name
        # appeared.
        widgets.field_label(key.replace("_", " "))
        was, changed_here = _widget(key, params.get(key, default), default)
        # The 2026-09-07 audit's clay-01: a generator field fired
        # ``set_generator_params`` -- also an unconditional ``history.push`` --
        # per keystroke, for the same reason the transform fields above needed
        # ``fold_undo``. Folded here, before ``set_generator_params`` runs
        # below, so the invariant every other door in this file follows
        # (draw, fold, act) holds for every field in the loop, not only the
        # one the user happened to stop typing in.
        controls.fold_undo(doc.history)
        if changed_here:
            edited[key] = was
            changed = True
    if not changed:
        return
    apply_generator_params(doc, obj, edited, ctx=ctx)


def apply_generator_params(
    doc: Any, obj: Any, edited: dict[str, Any], *, ctx: Any = None
) -> bool:
    """Clamp, rebuild and record one edit of a generator's parameters. -> whether it landed.

    The generic parameter loop's tail, public so a second caller sends its edits
    through the **same door**: one clamp, one rebuild, one ``regen.carry_over``,
    one ``set_generator_params`` step, one refusal handling. A second copy would
    be a second place for "store what the mesh was built from" to rot.
    """
    entry = bp.GENERATORS.get(obj.generator)
    if entry is None:
        return False
    defaults, build = entry
    params = dict(defaults)
    params.update({k: v for k, v in obj.params.items() if k in defaults})
    # Match what the generator will actually build *before* building it: the
    # 2026-09-06 audit's clay-05 finding was that a segment count of zero (or
    # a torus tube wider than its radius, clay-04) gets clamped inside the
    # generator without being reported back, so this panel used to save the
    # number the user typed rather than the one the mesh was built from.
    #
    # Inside the ``try``: ``clamp_params`` refuses a non-finite number anywhere
    # in a parameter (``ValueError`` naming the generator and key; an integer
    # key past ``int()`` raises ``OverflowError`` from the same call), and a
    # pasted ``inf`` or ``nan`` reached the frame thread uncaught when this
    # call sat above it (the 2026-10-03 audit's clay-28 follow-up).
    try:
        edited = bp.clamp_params(obj.generator, edited)
        mesh = build(**edited)
    except Exception:  # noqa: BLE001
        # A generator raises on a value it cannot build at all -- not the
        # zero segment count or oversized torus tube this comment used to
        # name (both are clamped, by clamp_params above and by the generator
        # itself; see the 2026-09-06 audit's clay-04 and clay-05 findings),
        # but a non-finite number, which ``clamp_params`` refuses and which
        # would otherwise survive to ``int()``, such as a pasted value large
        # enough to parse as infinity. The edit is refused: the old mesh and
        # parameters stay, no history step is pushed, and the field keeps the
        # number the user typed while it is focused, so they can correct it.
        #
        # Logged, not merely swallowed. A refusal about a number and a
        # ``TypeError`` from a renamed keyword are the same silence here, and
        # the second is a defect that would look exactly like a slider that
        # stopped working. Every comparable site in the tree logs; this one is
        # on the frame thread and per-keystroke, so it must not toast.
        log.debug("generator %r refused %r", getattr(build, "__name__", build), edited,
                  exc_info=True)
        return False
    # ``regen.carry_over`` is what keeps a rebuild from silently discarding a
    # hand-picked Shade Smooth/Flat or a hand-painted per-face material the
    # moment any generator field is touched -- see that module's own
    # docstring for the two-case rule and why it now lives there rather than
    # here (``agent_clay._h_set_params`` is the other door that rebuilds a
    # generator's mesh, and it used to carry neither attribute at all, so the
    # rule had to move somewhere both could reach rather than staying a
    # method on this pane).
    mesh = regen.carry_over(obj.mesh, mesh, material=obj.material)
    # One step, not two. The numbers and the mesh they build are one act, and
    # a lone Ctrl+Z restoring half the pair showed the old mesh in the viewport
    # while this panel still read the new radius -- and ``InputFloat`` fires
    # per keystroke, so typing a multi-digit number made several of them.
    # ``set_generator_params`` also implies ``keep_generator``: this mesh is
    # precisely what the generator builds from the edited parameters, which is
    # the one case where the object's generator claim is still true.
    # ``ctx`` is keyword-only and optional for the same reason
    # ``_apply_transform``'s own catch gives: several tests drive this door
    # with a bare ``(doc, obj)`` and no ``ctx``.
    from ......kernels.mesh.elements import OpError

    try:
        doc.set_generator_params(obj.uid, edited, mesh, was={"params": params})
    except OpError as error:
        if ctx is None:
            raise
        clay_ops.toast(ctx, str(error))
        return False
    return True


def _widget(key: str, value: Any, default: Any) -> tuple[Any, bool]:
    """One widget for one parameter, chosen from the default's type.

    The visible name moved to the ``field_label`` line the loop above draws
    (2026-09-08 consistency pass); hidden here the same way a previously
    visible control is hidden elsewhere in this pass -- "Foo" becomes
    "##Foo", not a new id -- so the id stays recognisably the old one.
    """
    label = f"##{key.replace('_', ' ')}##gen{key}"
    if isinstance(default, bool):
        return controls.checkbox(label, bool(value))[::-1]
    if isinstance(default, int):
        changed, out = controls.input_int(label, int(value), _STEP.get(key, 1))
        return out, changed
    if isinstance(default, float):
        changed, out = controls.input_float(label, float(value), 0.05)
        return out, changed
    # A *flat* sequence of 2 or 3 real numbers -- box's ``size``, plane's and
    # grid's ``size`` -- is the whole shape this branch knows how to draw.
    # Nothing past that may take it: a length-4-or-more default used to be
    # silently cut to 3 and written back that short (a five-number lathe
    # profile's flat form, say, would lose two numbers with nothing on
    # screen to say so), and a *nested* default such as
    # ``[[0.5, -0.5], [0.5, 0.5]]`` -- what a lathe profile actually is --
    # matched ``len(default) == 2`` and handed ``input_float2`` two Python
    # lists instead of two floats, which raises on the frame thread with no
    # try/except above it. Both, and a saved value that disagrees in shape
    # with the default (a bare scalar, where ``list(value)`` used to raise
    # outright), now fall through to the read-only fallback below, same as
    # any other type nobody has built a widget for yet.
    if (
        isinstance(default, (tuple, list))
        and len(default) in (2, 3)
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in default)
        and isinstance(value, (tuple, list))
    ):
        try:
            values = [float(v) for v in value]
        except (TypeError, ValueError):
            values = None
        if values is not None:
            pad = max(0, len(default) - len(values))
            values = values[: len(default)] + [0.0] * pad
            axes = ("X", "Y") if len(default) == 2 else ("X", "Y", "Z")
            changed, out = controls.input_vec(label, values, axes)
            return tuple(float(v) for v in out), changed
    # A parameter type nobody has added yet: shown, not editable, rather than
    # silently dropped from the panel.
    widgets.secondary(f"{key}: {value!r}")
    return value, False


#: The 2026-10-07 audit's clay-82: this tab said only "the palette is empty" and
#: returned before any control, so nothing on it pointed at the one place a slot
#: can be added from an empty palette -- the + in the Inspector's Colour card.
EMPTY_PALETTE_TEXT = (
    "The palette is empty. Add a slot with the + in the Inspector's Colour card."
)


def _material(ctx: Any, tab: Any, doc: Any, obj: Any) -> None:
    widgets.field_label("material")
    if not doc.materials:
        widgets.muted_wrapped(EMPTY_PALETTE_TEXT)
        return
    # The swatches themselves are the Inspector's Colour card: painting is a click
    # on a shape, and this page is where a slot's *options* live. Said here because
    # a page that quietly lost its picker would read as a bug.
    widgets.muted_wrapped(
        "Pick or paint a slot in the Colour card "
        "(Ctrl+click a swatch to edit it here)."
    )
    # The slot this page edits: the Ctrl+clicked one if any (tab state, not
    # ``obj.material`` -- the 2026-10-09 audit's clay-16), else the object's own.
    index = _page_slot(tab, doc, obj)
    _palette_row(doc, obj, index)

    material = doc.materials[index]
    widgets.muted_wrapped(_edit_slot_note(doc, obj, index, tab))
    # Each a sub-field of the "slot" combo above, named on its own line
    # (2026-09-08 consistency pass); ids kept stable, "Foo##bm" -> "##Foo##bm".
    widgets.field_label("base colour")
    changed, colour = controls.color_edit4(
        "##base colour##bm", list(material.base_color_factor)
    )
    # One gesture, one step: a drag reports on every frame the pointer moves,
    # and ``set_material`` pushes a step per report without this.
    controls.fold_undo(doc.history)
    # Clay keeps a palette entry to a colour, a base-colour texture,
    # double-sided and cutout (``document.reduce_material``): metallic,
    # roughness, emissive and the other texture maps are fixed at defaults, and
    # a file that carried them is stripped on read, so offering them here would
    # show a look the next reopen discards.
    cutout_changed, cutout = controls.checkbox(
        f"{icons.LAYERS} cutout##bm", material.alpha_mode == "MASK"
    )
    ds_changed, double_sided = controls.checkbox(
        f"{icons.LAYERS} double-sided##bm", bool(material.double_sided)
    )
    if changed or cutout_changed or ds_changed:
        # A *replacement*, never an in-place edit. Identity is what the GPU
        # cache, ``to_model`` and the writer all de-duplicate on, so editing
        # the object in place would leave every one of them showing the old
        # values with nothing in the data to say why.
        fresh = replace(
            material,
            base_color_factor=tuple(float(c) for c in colour),
            alpha_mode="MASK" if cutout else "OPAQUE",
            double_sided=bool(double_sided),
        )
        doc.set_material(index, fresh)
        material = fresh

    _texture_slots(ctx, tab, doc, index, material)


def _page_slot(tab: Any, doc: Any, obj: Any) -> int:
    """The palette slot the Colours & textures page edits for *obj*."""
    pick = getattr(tab, "edit_slot", None)
    if pick is not None:
        return int(pick(doc, obj))
    worn = worn_slot(doc, obj)
    return worn if worn is not None else min(max(int(obj.material), 0), len(doc.materials) - 1)


def _uv_view_slot(doc: Any, obj: Any, tab: Any = None) -> int:
    """The palette slot the UV pane draws the picture of for *obj*.

    Mirrors ``_uv_texture.material_for_pane``'s choice: the slot of the first
    selected face, else -- the 2026-10-10 audit's clay-28 -- the slot this page
    edits (:func:`_page_slot`: the Ctrl+clicked one, else the one every face
    wears), and not ``obj.material``, which is only the slot a *new* face is
    stamped with. A document whose faces all wear slot 2 drew nothing behind the
    islands while the page showed slot 2's picture. The rule is restated here
    rather than imported across the two panes.
    """
    selection = doc.element_sel.get(obj.uid)
    faces = getattr(selection, "faces", None)
    per_face = obj.mesh.material
    if faces is not None and len(faces) and 0 <= int(faces[0]) < len(per_face):
        return int(per_face[int(faces[0])])
    return _page_slot(tab, doc, obj)


def _slot_label(doc: Any, index: int) -> str:
    name = doc.materials[index].name if 0 <= index < len(doc.materials) else ""
    return f"slot {index} ({name})" if name else f"slot {index}"


def _edit_slot_note(doc: Any, obj: Any, index: int, tab: Any = None) -> str:
    """One sentence naming the slot the fields below edit, and -- under a face
    selection -- the other slot the UV view is showing.

    The 2026-10-07 audit's clay-22: the UV pane draws the first selected face's
    slot's picture while these fields, Edit texture in Inker, Take back and clear
    all act on the object's default slot, and neither side said which it was. The
    two are kept separate on purpose (Ctrl+click on a swatch is the documented way
    to edit another slot from a face selection), so the fix is to name them.
    """
    note = f"Editing {_slot_label(doc, index)}"
    # The 2026-10-11 audit's clay-42: with several shapes selected the page edits the
    # first of them (``inspector._page_object``) and its Add paints only that one, and
    # nothing said so -- ``swatches.single_selected`` calls a panel that silently edits
    # whichever shape sorts first worse than one that says it cannot.
    count = len(doc.selection)
    if count > 1:
        note += f" of {obj.name or f'object {obj.uid}'}, the first of {count} selected shapes"
    note += "."
    shown = _uv_view_slot(doc, obj, tab)
    if shown != index and 0 <= shown < len(doc.materials):
        note += (
            f" The UV view shows {_slot_label(doc, shown)}, the first selected face's;"
            " Ctrl+click its swatch to edit it instead."
        )
    return note


def _palette_remove_reason(material_count: int, users: int) -> str:
    """Why "Remove" is refused for the palette's current slot, or ``""``.

    A pure function beside the draw call -- the shape ``clay_ops.reason_for``
    and ``plotter_menu._layer_reason`` already use for the same rule -- so the
    decision is testable without a live imgui frame. Split out for the
    2026-09-12 audit's finding clay-05: the single-material case fell through
    both the enabling check and the explaining one, because both were gated
    on the same ``len(doc.materials) > 1``, so a document with exactly one
    material showed Remove greyed with nothing on screen saying why.
    """
    if material_count <= 1:
        return "A document keeps at least one material."
    if users:
        # The count spans objects the undo stack still holds, not only the
        # ones in the document -- a slot removed while an undone deletion was
        # the last thing using it came back magenta on redo.
        return f"{users} face(s) use this slot (including undone deletions)"
    return ""


#: *doc* -> ``(rev, {index: users})``, weak so a closed tab's own document
#: takes its entry with it -- ``outliner._HIERARCHY_CACHE``'s own shape.
_MATERIAL_USERS_CACHE: weakref.WeakKeyDictionary[Any, tuple[int, dict[int, int]]] = (
    weakref.WeakKeyDictionary()
)


def _material_users(doc: Any, index: int) -> int:
    """``doc.material_users(index)``, memoised on ``doc.rev``.

    The 2026-09-26 audit's clay-panes-06: ``_palette_row`` called
    ``doc.material_users`` every single frame the properties panel was open
    with an object selected, and that method's own docstring says it sums a
    face count over every object *and* every object an undo step is still
    holding out of the document -- expensive, and unmemoised, for a number
    that is on screen every frame regardless of whether anything changed.
    ``rev`` already covers both halves: every edit that could add, remove or
    reassign a face's material bumps it (``ClayDoc.touch``), and so does
    every undo/redo that moves an object into or out of the stack's own
    holding (each ``Edit.undo``/``redo`` in ``edits.py`` calls ``touch()``
    too), so there is nothing ``material_users`` can see change without
    ``rev`` moving first.
    """
    cached = _MATERIAL_USERS_CACHE.get(doc)
    if cached is not None and cached[0] == doc.rev and index in cached[1]:
        return cached[1][index]
    users = doc.material_users(index)
    by_index = dict(cached[1]) if cached is not None and cached[0] == doc.rev else {}
    by_index[index] = users
    _MATERIAL_USERS_CACHE[doc] = (doc.rev, by_index)
    return users


def _palette_row(doc: Any, obj: Any, index: int | None = None) -> None:
    """Add, rename and remove palette entries.

    Add appends -- never inserts -- because a slot *is* an index that every
    mesh's per-face ``material`` array names, and inserting one in the middle
    would renumber those arrays in every object in the document.

    Remove is offered only for an entry no face uses, and says so rather than
    reassigning those faces somewhere. Reassigning is a silent change to how
    part of the model looks, which is exactly the kind of thing a user
    discovers three edits later with no idea what did it.
    """
    if index is None:
        index = min(max(int(obj.material), 0), len(doc.materials) - 1)
    users = _material_users(doc, index)
    if controls.small_button(
        f"{icons.PLUS} Add##matadd",
        tooltip="Add a palette slot and paint this object with it -- one undo step.",
    ):
        # One step, not two -- the 2026-09-08 audit's clay-02: pushed as
        # ``add_material()`` then ``set_props(...)`` separately, one Ctrl+Z
        # after this click left a stray, unreferenced palette entry behind.
        # repaint=True: the faces carry the slot that renders (clay-17).
        doc.add_material_and_assign(obj.uid, repaint=True)
    imgui.same_line()
    reason = _palette_remove_reason(len(doc.materials), users)
    if widgets.disabled_button(
        "Remove##matdel",
        not reason,
        reason=reason,
        tooltip="Remove this slot from the palette. Only a slot no face uses can go.",
    ):
        # Same fold as Add, for the same reason.
        doc.remove_material_and_reassign(obj.uid, index)
    if reason:
        widgets.muted_wrapped(reason)

    # commit=True for the reason ``_identity``'s name field needs it: the
    # 2026-09-06 audit's clay-02 found this one reporting per keystroke too,
    # pushing a ``set_material`` history step for every letter of a slot name.
    name = widgets.input_text(
        "slot name##matname", doc.materials[index].name or "", max_length=60, commit=True
    )
    if name != (doc.materials[index].name or ""):
        # A replacement, never an in-place edit, for the reason the colour
        # fields below state: identity is what every cache de-duplicates on.
        doc.set_material(index, replace(doc.materials[index], name=name))


#: The one texture slot a Clay material has: the base colour.
TEXTURE_SLOTS = ("base_color",)

#: This pane's own task-key prefix for "assign a texture from a file" --
#: **not** a bare ``clay-`` key (``clay-open``, ``clay-save``, ``clay-export``...),
#: because landing the result is not a document task in
#: ``clay_mode.on_task_done``'s sense (see
#: ``shell/tasks.py``'s own ``clay-mattex:`` branch, checked before its
#: ``clay-`` one for exactly this reason).
TEXTURE_TASK_PREFIX = "clay-mattex"


def _texture_task_key(tab_uid: str, index: int, slot: str) -> str:
    return f"{TEXTURE_TASK_PREFIX}:{tab_uid}:{index}:{slot}"


def _pick_texture(slot: str) -> dict[str, Any] | None:
    """Blocking; task thread only. -> ``{"width", "height", "rgba"}``, or
    ``None`` for a cancelled picker.

    The picker, then the decode, both off the frame thread -- Inker's
    ``ui/panes/opening.py::ask_import_sheet`` is the precedent this follows:
    a file dialog is the one place an *arbitrary* image reaches this app, so
    it is read through the same ``pixelguard`` pixel ceiling every other
    hand-picked image is, rather than trusted because it came from a picker.
    """
    from ......core.safeio import pixelguard
    from ......service.errors import Invalid
    from ..... import dialogs
    from ... import texture_link

    path = dialogs.open_file(f"Assign a {slot.replace('_', ' ')} texture", dialogs.PNG_FILTER)
    if path is None:
        return None
    with pixelguard.opened(path, path.name) as image:
        # The 2026-10-10 audit's clay-60: ``pixelguard`` allows 8192 x 8192, but a
        # texture is held to ``MAX_TEXTURE_SIDE``, so a 4096-8192 px PNG was decoded
        # whole (~268 MB of RGBA plus a ``tobytes()`` copy) before ``on_task_done``
        # refused it. The size is a header read; ask it before ``convert`` allocates.
        # An ``Invalid`` rather than a ``ValueError``, so the task runner toasts the
        # sentence instead of "Something went wrong".
        width, height = image.size
        if max(width, height) > texture_link.MAX_TEXTURE_SIDE:
            raise Invalid(
                f"That picture is {width} x {height}; a texture can be at most "
                f"{texture_link.MAX_TEXTURE_SIDE} px on a side."
            )
        pixels = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    return {"width": int(width), "height": int(height), "rgba": pixels.tobytes()}


def _assign_texture(ctx: Any, tab: Any, index: int, slot: str, material: Any) -> None:
    """*material* is carried as the task's ``tag`` -- the object identity of
    the palette slot at *index* the moment the picker opened -- so
    :func:`on_task_done` can tell whether that slot is still the same
    material by the time the picker returns.

    clay-16 (the 2026-09-23 audit): a lower slot removed while this dialog
    is open shifts every index above it down, so ``index`` alone can name a
    different material by the time the file comes back; the identity in
    ``tag`` is what lets the landing refuse rather than paint the wrong slot.
    """
    key = _texture_task_key(tab.uid, index, slot)
    if not ctx.submit(key, _pick_texture, slot, tag=material):
        ctx.toast("A file dialog is already open.", "info")


def on_task_done(ctx: Any, done: Any) -> None:
    """Land a texture picked for a material slot -- ``clay-mattex:<tab uid>:
    <material index>:<slot>``, dispatched here by ``shell/tasks.py`` rather
    than by ``clay_mode.on_task_done`` (see :data:`TEXTURE_TASK_PREFIX`).

    Every way this can be stale is an ordinary no-op, not a refusal: the
    document can have been closed, the palette slot removed or the picker
    simply cancelled (``done.result`` is then ``None``) while the dialog was
    open, and none of those is a failure worth a toast over -- the user asked
    for a file, or didn't pick one, and either way nothing here was promised
    to still exist by the time the answer comes back.

    """
    parts = done.key.split(":", 3)
    if len(parts) != 4:
        return
    _prefix, tab_uid, index_text, slot = parts
    if slot not in TEXTURE_SLOTS:
        return
    result = done.result
    if not isinstance(result, dict):
        return
    state = clay_mode.ensure(ctx)
    tab = state.get(tab_uid)
    if tab is None:
        return
    try:
        index = int(index_text)
    except ValueError:
        return
    doc = tab.doc
    if not 0 <= index < len(doc.materials):
        return
    material = doc.materials[index]
    if done.tag is not None and material is not done.tag:
        # clay-16 (the 2026-09-23 audit): a slot below this one removed
        # while the file dialog was open shifts every index above it down,
        # so ``index`` alone can now name a different material than the one
        # the picker was opened for. ``done.tag`` is the material object
        # identity ``_assign_texture`` captured at pick time -- if the slot
        # at this index is no longer that same object, the pick is stale,
        # the same silent no-op every other staleness case in this function
        # already gets.
        return
    if getattr(tab, "saving", False) or clay_mode._gesture_live(ctx, tab):
        # The 2026-10-10 audit's clay-26: the landing is ``set_material`` plus a
        # box-unwrap ``set_mesh``, and the OS picker does not block the app, so a
        # Ctrl+S or Export started while it was open had a worker thread reading
        # the live document as this rewrote its material and UVs -- a file mixing
        # old and new. The Inker pull refuses the same way (``texture_link.pull_all``,
        # the 2026-10-07 audit's clay-48) but keeps its link and lands next frame;
        # this pick has no link to wait on, so it is dropped and said so.
        ctx.toast(
            "A save or a drag was in progress, so that texture was not applied. "
            "Pick the file again once it finishes.",
            "info",
        )
        return
    from ... import texture_link

    width, height = int(result["width"]), int(result["height"])
    if max(width, height) > texture_link.MAX_TEXTURE_SIDE:
        # The 2026-10-07 audit's clay-79 (and clay-20): the picker's decode only
        # bounds a *file*, and an 8192 px picture here made a ``.rblk`` that saved
        # but passed the reader's decoded-bytes budget on reopen -- and a slot
        # Inker could not take back. The same ceiling the pull applies, by name.
        ctx.toast(
            f"That picture is {width} x {height}; a texture can be at most "
            f"{texture_link.MAX_TEXTURE_SIDE} px on a side.",
            "error",
        )
        return
    _land_assigned_texture(doc, index, material, slot, (width, height, result["rgba"]))


def _land_assigned_texture(
    doc: Any, index: int, material: Any, slot: str, image: tuple[int, int, bytes]
) -> None:
    """*image* as palette slot *index*'s texture, set up the way Add texture sets
    one up, as **one** undo step.

    The 2026-10-07 audit's clay-21: a PNG from a file landed only the picture, so
    the slot kept its 0.8 grey colour factor (the shader multiplies factor by
    texel, tinting the picture), a UV-less object showed one texel everywhere, and
    ``nearest`` kept whatever the slot had -- True over a crisp texture, so a big
    smooth PNG sampled NEAREST. Colour factor white (alpha kept), every UV-less
    object with a face on the slot box-unwrapped, and ``nearest`` False: a picture
    from disk is smooth until it has been through Inker, which is what the manual
    says. Done here through ``ClayDoc``'s public doors because ``add_texture``
    refuses a slot that already has a picture.
    """
    mark = doc.history.mark()
    try:
        factor = material.base_color_factor
        doc.set_material(
            index,
            replace(
                material,
                **{slot: image},
                base_color_factor=(1.0, 1.0, 1.0, float(factor[3])),
                nearest=False,
            ),
        )
        for obj in list(doc.objects):
            if obj.mesh.uv is None and np.any(obj.mesh.material == index):
                doc.set_mesh(obj.uid, uv_projection.box_unwrap(obj.mesh), keep_generator=True)
    finally:
        doc.history.collapse_since(mark)


#: The size "Add texture" makes, in texels a side. Pane state rather than
#: document state on purpose: it is a default for the next click, not a fact
#: about any palette entry, and it must not travel into a saved document or
#: back through undo.
_TEXTURE_SIZE = 64

_TEXTURE_SIZE_OPTIONS = tuple((str(n), f"{n} px") for n in clay_document.TEXTURE_SIZES)

#: The popup listing open Inker documents for "Take texture back from Inker".
_TAKE_BACK_POPUP = "##claytexback"


def _texture_slots(ctx: Any, tab: Any, doc: Any, index: int, material: Any) -> None:
    """The base-colour texture of palette entry *index*: make one, edit it in
    Inker, bring it back, clear it, or assign one from a file.

    Never mutated in place, always a fresh ``replace(material, ...)``, the same
    rule the colour fields above follow -- and every button is one undo step.
    """
    global _TEXTURE_SIZE
    widgets.field_label("texture")
    for slot in TEXTURE_SLOTS:
        image = getattr(material, slot, None)
        if image is None:
            widgets.muted("none")
            if controls.small_button(
                f"{icons.PLUS} Add texture##texadd",
                tooltip="A blank picture in this slot's colour, painted in Inker. "
                "Faces without UVs get a box unwrap.",
            ):
                try:
                    doc.add_texture(index, _TEXTURE_SIZE)
                except el.OpError as error:
                    ctx.toast(f"No texture was added: {error}", "error")
            imgui.same_line()
            imgui.set_next_item_width(sp(76))
            changed, picked = controls.combo(
                "##texsize", str(_TEXTURE_SIZE), _TEXTURE_SIZE_OPTIONS
            )
            if changed:
                _TEXTURE_SIZE = int(picked)
        else:
            width, height, _data = image
            widgets.muted(f"{width} x {height}")
            if controls.small_button(
                f"{icons.PENCIL} Edit texture in Inker##texinker",
                tooltip="Opens the picture in Inker with the 16 PICO-8 colours; "
                "what you paint lands back here.",
            ):
                from ... import texture_link

                texture_link.edit_in_inker(ctx, tab, index)
            _take_back_row(ctx, tab, index)
        if ctx.busy(_texture_task_key(tab.uid, index, slot)):
            widgets.muted("...")
        else:
            assign_tip = "Assign from a file..."
            # FOLDER_OPEN, not UPLOAD (icons.py's rule: UPLOAD reads backwards
            # for both import and export in an app with no remote server) --
            # this button is "in from a file", the same reason the Import Mesh...
            # button above already uses it.
            if controls.small_button(f"{icons.FOLDER_OPEN}##texassign{slot}", tooltip=assign_tip):
                _assign_texture(ctx, tab, index, slot, material)
            imgui.same_line()
            # Glyph-only, so it carries a tooltip of its own, and a reason while
            # the slot has no picture (the 2026-10-07 audit's clay-75).
            if widgets.disabled_button(
                f"{icons.X}##texclear{slot}",
                image is not None,
                reason=_clear_texture_reason(image),
                tooltip="Clear the texture -- one undo step.",
            ):
                _clear_texture(tab, doc, index, material, slot)


def _clear_texture_reason(image: Any) -> str:
    """Why the clear-texture button is greyed, or ``""``."""
    return "" if image is not None else "This slot has no texture to clear."


def _clear_texture(tab: Any, doc: Any, index: int, material: Any, slot: str) -> None:
    """Drop the picture as **one** step, and the Inker link that was feeding it.

    ``nearest`` goes with the picture: it is a sampler choice for *that*
    texture, and a flat colour left flagged would read as a stale setting. The
    link goes because a later return from Inker would otherwise land a picture
    on a slot the user just emptied.
    """
    from ... import texture_link

    doc.set_material(index, replace(material, **{slot: None, "nearest": False}))
    texture_link.unlink(tab, index)


def _take_back_row(ctx: Any, tab: Any, index: int) -> None:
    """The manual fallback to the automatic pull: pick an open Inker document
    and take its picture, Plotter's "Back onto" row for a material."""
    from ... import texture_link

    if controls.small_button(
        f"{icons.IMAGE} Take texture back from Inker##texback",
        tooltip="Pick an open Inker document to copy into this slot.",
    ):
        imgui.open_popup(_TAKE_BACK_POPUP)
    with controls.menu_popup(_TAKE_BACK_POPUP) as opened:
        if not opened:
            return
        entries = list(texture_link.open_inker_docs(ctx))
        if not entries:
            widgets.muted("No Inker documents are open.")
        for entry in entries:
            hit = controls.menu_item(f"{icons.IMAGE} {entry.title}##texback-{entry.uid}")
            if bool(hit[0] if isinstance(hit, tuple) else hit):
                texture_link.take_back(ctx, tab, index, entry)
