"""The selected object -- its transform, its generator's parameters, its material -- and
the document's own counts and import settings.

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
from typing import Any

import numpy as np
from imgui_bundle import imgui

from ......kernels.geom3d import math3d as m3
from ......kernels.geom3d import units
from ......kernels.mesh import document as clay_document
from ......kernels.mesh import elements as el
from ......kernels.mesh import primitives as bp
from ......kernels.mesh import regen
from ......kernels.mesh import uv as uv_projection
from ..... import controls, icons, theme, tokens, widgets
from .....manual import render as manual_render
from .....tokens import sp
from ... import mode as clay_mode
from ... import ops as clay_ops
from ... import transform_edit
from ...state import ClayState
from . import outliner as clay_outliner

log = logging.getLogger(__name__)

# How a parameter's type decides its widget. Read off the *default value*,
# because a registry entry carries no schema and does not need one: a float
# default means a float field, and a tuple means one field per component.
_STEP = {"segments": 1, "rings": 1, "sides": 1}


def draw(ctx: Any) -> None:
    """This pane's headings, on tinted blocks.

    The blocks are opened *here* rather than in :func:`layout.pane`, which is
    flat: a pane on a wide canvas wants no tint, and this is one of the four
    narrow sidebars the grouping was written for (see
    ``tests/test_section_blocks.py`` for the report it came from). Wrapping
    ``_body`` rather than inlining the ``with`` keeps every early return inside
    the scope, and the scope closes its last block on the way out.
    """
    with widgets.section_blocks():
        _body(ctx)


#: The tabs of this pane, Blender's Properties editor reduced to what Clay has:
#: ``(key, label, glyph, what it holds)``. Glyph-only on the strip (a 300 px
#: sidebar has no room for three words) with the label and contents in the
#: tooltip, as the header's tool pill does.
TABS: tuple[tuple[str, str, str, str], ...] = (
    ("object", "Object", icons.BOX, "Name, parent, transform and the shape's own numbers"),
    ("material", "Material", icons.PALETTE, "The palette and textures"),
    ("document", "Document", icons.SETTINGS, "Counts and the import settings, document-wide"),
)

#: The tab a pane falls back to when ``ClayState.props_tab`` names one that no
#: longer exists (a saved setting from a build with different tabs).
DEFAULT_TAB = "object"


def tab_key(state: Any) -> str:
    """The tab to draw: ``state.props_tab`` if it is a real one, else Object."""
    key = getattr(state, "props_tab", DEFAULT_TAB)
    return key if any(key == known for known, *_ in TABS) else DEFAULT_TAB


def _body(ctx: Any) -> None:
    state = clay_mode.ensure(ctx)
    tab = state.active
    widgets.section("Properties")
    manual_render.help_button(ctx, "clay-props")
    if tab is None:
        # The heading and nothing else; see ``clay_outliner``.
        return
    doc = tab.doc
    current = tab_key(state)
    changed, picked = controls.segmented_choice(
        "clay-props-tab",
        [(key, glyph) for key, _label, glyph, _what in TABS],
        current,
        tooltips={key: f"{label} -- {what}" for key, label, _glyph, what in TABS},
        compact=True,
    )
    if changed:
        state.props_tab = current = picked
    imgui.dummy((0, sp(tokens.SP_2)))
    if current == "document":
        # The only tab about the document rather than the selection, so it is
        # the one that needs no object.
        imgui.begin_disabled(tab.saving)
        _document(ctx, doc)
        imgui.end_disabled()
        return
    _element_summary(doc)
    obj = _selected(doc)
    if obj is None:
        # Two sentences, because ``_selected`` returns None for two different
        # reasons and one of them used to lie: with sixteen objects lit up the
        # pane said "Nothing selected", which the viewport plainly contradicts
        # -- and a multi-object selection is an ordinary case rather than the
        # odd one. The *refusal* is unchanged
        # (see ``_selected``); only the sentence the user reads is.
        count = len(doc.selection)
        if count > 1:
            widgets.empty_state(
                icons.BOX,
                f"{count} objects selected",
                "Select one to edit it.",
            )
        else:
            title, hint = _nothing_selected_text(doc, current)
            widgets.empty_state(icons.BOX, title, hint)
        return

    imgui.begin_disabled(tab.saving)
    if current == "object":
        _identity(doc, obj)
        imgui.dummy((0, sp(tokens.SP_2)))
        _relations(ctx, doc, obj)
        imgui.dummy((0, sp(tokens.SP_2)))
        _transform(doc, obj, ctx=ctx, state=state)
        imgui.dummy((0, sp(tokens.SP_2)))
        _generator(doc, obj, ctx=ctx, state=state)
    else:
        _material(ctx, tab, doc, obj)
    imgui.end_disabled()


def _nothing_selected_text(doc: Any, tab: str = "object") -> tuple[str, str]:
    """The empty state's ``(title, hint)`` for a pane with no object to show.

    The 2026-10-07 audit's clay-74: in an element mode an object is selected
    *through* its elements (``ClayDoc.selection`` is exactly the uids with a
    non-empty element selection), so "Click an object" asked for the one gesture
    that mode does not make -- and the swatch row the manual describes for face
    mode was unreachable behind it. The hint names what the mode picks.
    """
    noun = {"vertex": "a vertex", "edge": "an edge", "face": "a face"}.get(doc.element_mode)
    if noun is None:
        return "Nothing selected", "Click an object in the viewport."
    hint = f"Click {noun} in the viewport."
    if doc.element_mode == "face" and tab == "material":
        hint += " Then pick a swatch to paint it."
    return "Nothing selected", hint


def _document(ctx: Any, doc: Any) -> None:
    """The document as a whole: what is in it, and how the next import reads a file.

    The counts are of what leaves the document -- visible objects only, the same
    set the exporters write -- so the triangle line is a promise about the
    exported file.
    """
    from . import bridge as clay_bridge

    visible = [obj for obj in doc.objects if obj.visible]
    widgets.field_label("counts")
    widgets.muted(
        f"{len(visible)} of {len(doc.objects)} objects visible  -  "
        f"{len(doc.materials)} materials"
    )
    widgets.muted(
        f"{sum(len(o.mesh.positions) for o in visible):,} vertices  -  "
        f"{sum(max(0, len(o.mesh.starts) - 1) for o in visible):,} faces  -  "
        f"{sum(clay_bridge._triangles(o.mesh) for o in visible):,} triangles"
    )
    imgui.dummy((0, sp(tokens.SP_2)))
    clay_bridge.import_settings(ctx)


def _element_summary(doc: Any) -> None:
    """One line saying what is selected inside the objects, in element modes.

    The object panel below stays exactly as it was -- an element selection is
    still an object selection, by the document's own invariant -- so this adds
    a line rather than replacing the pane. It is also where the *frozen* branch
    of the generator section finally becomes reachable: an op that edits
    topology clears ``generator``, and this is usually the first thing the user
    sees afterwards.
    """
    text = element_summary_text(doc)
    if text is None:
        return
    widgets.muted(text)
    imgui.dummy((0, sp(tokens.SP_1)))


def element_summary_text(doc: Any) -> str | None:
    """:func:`_element_summary`'s line, or ``None`` in object mode.

    The count summed every entry of
    ``doc.element_sel``, but a drag, the gizmo centre and every element door
    skip a hidden object (``selection._element_pickable``), so
    hiding an object that still held a selection made "N selected" promise
    elements the next drag would not move. Only eligible objects are counted,
    and "across N objects" counts only those too.
    """
    from ......kernels.mesh.selection import _element_pickable

    if doc.element_mode == "object":
        return None
    total = 0
    objects = 0
    for uid, sel in doc.element_sel.items():
        try:
            if not _element_pickable(doc.by_uid(uid)):
                continue
        except KeyError:
            continue
        count = sel.count(doc.element_mode)
        if count:
            total += count
            objects += 1
    noun = {"vertex": "vertices", "edge": "edges", "face": "faces"}[doc.element_mode]
    if total == 0:
        return f"{doc.element_mode} mode -- nothing selected"
    across = "1 object" if objects == 1 else f"{objects} objects"
    return f"{doc.element_mode} mode -- {total} {noun} across {across}"


def _selected(doc: Any) -> Any:
    """The one selected object, or None.

    One rather than the first of many: a properties panel that silently edited
    whichever object happened to sort first under a multi-selection is worse
    than one that says it cannot.
    """
    if len(doc.selection) != 1:
        return None
    try:
        return doc.by_uid(next(iter(doc.selection)))
    except KeyError:
        return None


def _identity(doc: Any, obj: Any) -> None:
    # commit=True: the 2026-09-06 audit's clay-02 found this field reporting a
    # change on every keystroke, so ``set_props`` -- an unconditional
    # ``history.push`` -- fired once per letter typed and a lone Ctrl+Z after
    # a rename undid one character instead of the whole name.
    name = widgets.input_text("name##buildname", obj.name, max_length=120, commit=True)
    if name != obj.name:
        doc.set_props(obj.uid, name=name)
    changed, value = widgets.toggle(f"{icons.EYE} Visible", obj.visible, tag=str(obj.uid))
    if changed:
        doc.set_props(obj.uid, visible=value)


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
    from ... import ops as clay_ops

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
        "Clear parent##clearparent", obj.parent is not None, reason=_clear_parent_reason(obj)
    ):
        picked = "0"
    if picked != current:
        _set_parent(ctx, doc, obj.uid, None if picked == "0" else int(picked))


#: What a bare ``_transform(doc, obj)`` -- no pane, no ``ClayState`` -- shows its
#: display state in. Several tests drive the door with nothing but a document.
_BARE_STATE: ClayState | None = None

_UNIT_OPTIONS = [(key, key) for key, _ in units.LENGTH_UNITS]


def _display_state(state: ClayState | None) -> ClayState:
    global _BARE_STATE
    if state is not None:
        return state
    if _BARE_STATE is None:
        _BARE_STATE = ClayState()
    return _BARE_STATE


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
    from ... import ops as clay_ops

    try:
        doc.set_transform(obj.uid, **fields)
    except OpError as error:
        if ctx is None:
            raise
        clay_ops.toast(ctx, str(error))
        return False
    return True


def _transform(doc: Any, obj: Any, *, ctx: Any = None, state: ClayState | None = None) -> None:
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
    shown = units.vec_to_display(obj.translation, unit)
    if parented:
        edited, typed = controls.input_vec("local position##bt", list(shown), ("X", "Y", "Z"))
    else:
        edited, typed = controls.input_vec("position##bt", list(shown), ("X", "Y", "Z"))
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
    if parented:
        edited, scale = controls.input_vec(
            "local scale##bs", list(obj.scale), ("X", "Y", "Z")
        )
    else:
        edited, scale = controls.input_vec("scale##bs", list(obj.scale), ("X", "Y", "Z"))
    controls.fold_undo(doc.history)
    changed |= edited
    # Rotation is Euler degrees in the app's one rotation order (XYZ, the MCP
    # surface's), which is what lets the panel and an agent's ``clay_transform``
    # read the same three numbers. The object stores a quaternion and
    # decomposing one is not stable frame to frame, so the angles shown are
    # cached per object until something else moves the quaternion
    # (``transform_edit``'s docstring).
    euler = transform_edit.displayed_euler(ui.euler_cache, obj.uid, obj.rotation)
    if parented:
        edited, typed_euler = controls.input_vec(
            "local rotation (deg)##br", list(euler), ("X", "Y", "Z")
        )
    else:
        edited, typed_euler = controls.input_vec(
            "rotation (deg)##br", list(euler), ("X", "Y", "Z")
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
                ui.euler_cache, obj.uid, doc.by_uid(obj.uid).rotation, typed_euler
            )
    _dimensions(doc, obj, ctx=ctx, state=ui)


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
    from ......kernels.mesh import ops as bops

    ui = _display_state(state)
    unit = ui.length_unit if ui.length_unit in dict(units.LENGTH_UNITS) else units.DEFAULT_UNIT
    mesh = obj.mesh
    box = bops.world_box(obj, mesh, world=doc.world_matrix(obj.uid))
    if box is None:
        return
    extent = transform_edit.local_extent(mesh)
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
    from ... import ops as clay_ops

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

    The one place the row touches imgui's colour button, so a test can stand in
    a press without a mouse. The slot the object defaults to is outlined in the
    accent; a textured slot carries a corner notch so it reads as a picture and
    not as a flat colour.
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


def _pick_slot(ctx: Any, doc: Any, obj: Any, index: int) -> None:
    """What a click on palette swatch *index* does.

    Object mode keeps the old combo's behaviour exactly: the object's default
    slot **and every face** go to the slot (the 2026-10-03 audit's clay-17).
    In face mode with faces selected the click paints those faces through the
    ``assign-material`` op and leaves the object's default slot alone; Ctrl
    (or no faces selected) only makes it the slot the fields below edit, the
    one way to reach another slot's colour and texture from a face selection.

    Vertex and edge mode only pick the slot, like a face click with nothing
    selected (the 2026-10-07 audit's clay-73): they used to take the object-mode
    branch and repaint every face of the object, though the selection in front of
    the user was a few points or edges and no face was named. Painting faces is
    face mode's job.
    """
    if doc.element_mode == "object":
        doc.repaint_object(obj.uid, index)
        return
    op = clay_ops.get("assign-material")
    if doc.element_mode != "face" or imgui.get_io().key_ctrl or not op.enabled(doc):
        if index != obj.material:
            doc.set_props(obj.uid, material=index)
        return
    clay_ops.run(ctx, doc, op, index=index)


def _swatch_row(ctx: Any, doc: Any, obj: Any) -> None:
    """One swatch per palette slot, wrapped to the pane."""
    current = min(max(int(obj.material), 0), len(doc.materials) - 1)
    side = sp(SWATCH)
    gap = imgui.get_style().item_spacing.x
    per_row = max(1, int((imgui.get_content_region_avail().x + gap) // (side + gap)))
    clicked: int | None = None
    for i, entry in enumerate(doc.materials):
        if i % per_row:
            imgui.same_line()
        name = entry.name or f"slot {i}"
        textured = entry.base_color is not None
        tip = f"{i}: {name}" + (" (textured)" if textured else "")
        if _swatch(
            f"##matsw{i}",
            _swatch_colour(entry),
            side,
            selected=i == current,
            textured=textured,
            tooltip=tip,
        ):
            clicked = i
    if doc.element_mode == "face":
        widgets.muted_wrapped(
            "Click a swatch to paint the selected faces; Ctrl+click to edit that slot instead."
            if clay_ops.get("assign-material").enabled(doc)
            else "Select faces to paint them with a swatch."
        )
    elif doc.element_mode != "object":
        widgets.muted_wrapped(
            "A click picks the slot the fields below edit. Switch to face mode to paint faces."
        )
    if clicked is not None:
        _pick_slot(ctx, doc, obj, clicked)


def _material(ctx: Any, tab: Any, doc: Any, obj: Any) -> None:
    widgets.field_label("material")
    if not doc.materials:
        widgets.muted("the palette is empty")
        return
    _swatch_row(ctx, doc, obj)
    _palette_row(doc, obj)

    index = min(max(int(obj.material), 0), len(doc.materials) - 1)
    material = doc.materials[index]
    widgets.muted_wrapped(_edit_slot_note(doc, obj, index))
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


def _uv_view_slot(doc: Any, obj: Any) -> int | None:
    """The palette slot the UV pane draws the picture of for *obj* under a face
    selection, or ``None`` when it draws the object's own slot.

    Mirrors ``_uv_texture.material_for_pane``'s choice (the slot of the first
    selected face); that module belongs to the UV pane, so the rule is restated
    here rather than imported across the two panes.
    """
    selection = doc.element_sel.get(obj.uid)
    faces = getattr(selection, "faces", None)
    per_face = obj.mesh.material
    if faces is not None and len(faces) and 0 <= int(faces[0]) < len(per_face):
        return int(per_face[int(faces[0])])
    return None


def _slot_label(doc: Any, index: int) -> str:
    name = doc.materials[index].name if 0 <= index < len(doc.materials) else ""
    return f"slot {index} ({name})" if name else f"slot {index}"


def _edit_slot_note(doc: Any, obj: Any, index: int) -> str:
    """One sentence naming the slot the fields below edit, and -- under a face
    selection -- the other slot the UV view is showing.

    The 2026-10-07 audit's clay-22: the UV pane draws the first selected face's
    slot's picture while these fields, Edit texture in Inker, Take back and clear
    all act on the object's default slot, and neither side said which it was. The
    two are kept separate on purpose (Ctrl+click on a swatch is the documented way
    to edit another slot from a face selection), so the fix is to name them.
    """
    note = f"Editing {_slot_label(doc, index)}."
    shown = _uv_view_slot(doc, obj)
    if shown is not None and shown != index and 0 <= shown < len(doc.materials):
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


def _palette_row(doc: Any, obj: Any) -> None:
    """Add, rename and remove palette entries.

    Add appends -- never inserts -- because a slot *is* an index that every
    mesh's per-face ``material`` array names, and inserting one in the middle
    would renumber those arrays in every object in the document.

    Remove is offered only for an entry no face uses, and says so rather than
    reassigning those faces somewhere. Reassigning is a silent change to how
    part of the model looks, which is exactly the kind of thing a user
    discovers three edits later with no idea what did it.
    """
    index = min(max(int(obj.material), 0), len(doc.materials) - 1)
    users = _material_users(doc, index)
    if controls.small_button(f"{icons.PLUS} Add##matadd"):
        # One step, not two -- the 2026-09-08 audit's clay-02: pushed as
        # ``add_material()`` then ``set_props(...)`` separately, one Ctrl+Z
        # after this click left a stray, unreferenced palette entry behind.
        # repaint=True: the faces carry the slot that renders (clay-17).
        doc.add_material_and_assign(obj.uid, repaint=True)
    imgui.same_line()
    reason = _palette_remove_reason(len(doc.materials), users)
    if widgets.disabled_button("Remove##matdel", not reason):
        # Same fold as Add, for the same reason.
        doc.remove_material_and_reassign(obj.uid, index)
    if reason:
        widgets.muted(reason)

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
    from ..... import dialogs

    path = dialogs.open_file(f"Assign a {slot.replace('_', ' ')} texture", dialogs.PNG_FILTER)
    if path is None:
        return None
    pixels = pixelguard.decode_rgba(path, path.name)
    height, width = pixels.shape[:2]
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
            # this button is "in from a file", the same reason bridge.py's
            # "Import Mesh..." already uses it.
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
