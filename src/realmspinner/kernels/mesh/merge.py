"""Landing a second document's objects inside a first, as one undo step.

The kernel half of "Generate into the current Clay tab" (the controller lives
in ``studio/modes/clay/generate.py`` above this): a mesh built elsewhere --
a promoted reference, an uploaded image -- arrives as a whole
:class:`~.document.ClayDoc` of its own (:func:`~.glbimport.glb_to_claydoc`
already builds exactly that from a ``model.glb``), and this is what folds it
into the document the user was already looking at, rather than opening it as
a second tab the way ``clay_mode.edit_asset_in_clay`` still does for the
Library door.

Two questions, two functions. :func:`placement_offset` decides *where* --
beside whatever is selected, or grounded at the origin -- entirely in world
space and with no side effect on either document, so a caller can show a
preview or a confirm before ever calling :func:`merge_into`. :func:`merge_into`
does the actual adoption: new palette slots, renumbered face materials, unique
names, the offset applied to *incoming*'s own roots, one history step and one
group if more than one object arrived.

Headless like the rest of this package: numpy and its own siblings only (see
``tests/modes/clay/test_clay_imports.py``'s pin, which discovers this module
by globbing the package directory and needs no edit for it to be covered).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import numpy as np

from . import ops as mesh_ops
from .document import ClayDoc, Obj

if TYPE_CHECKING:
    from . import mesh as bm

#: The floor under the gap :func:`placement_offset` leaves beside a selection.
#: Ten per cent of the larger of the two boxes' own extents can be a hair's
#: width for two tiny primitives sitting flush against each other -- a gap a
#: user cannot see is a gap that reads as a seam, not as "these are two
#: things".
MIN_GAP = 0.1


def _world_box(doc: ClayDoc, uids: list[int]) -> tuple[np.ndarray, np.ndarray] | None:
    """The combined world-space box of *uids* in *doc*, or ``None`` if none of
    them has any geometry.

    Shared by both halves of :func:`placement_offset` -- the selection's own
    box and *incoming*'s -- and it is exactly :meth:`ClayDoc.group`'s own
    bounds arithmetic, over an arbitrary uid list rather than always "the
    uids about to be grouped". World, not local: :func:`~.ops.world_box`
    already composes the full ancestor chain through ``doc.world_matrix``, so
    a child deep in a hierarchy contributes its real position, not its
    TRS relative to its own parent.
    """
    lo = hi = None
    for uid in uids:
        try:
            obj = doc.by_uid(uid)
        except KeyError:
            continue
        box = mesh_ops.world_box(obj, mesh=doc.evaluated(uid), world=doc.world_matrix(uid))
        if box is None:
            continue
        b_lo, b_hi = box
        lo = b_lo if lo is None else np.minimum(lo, b_lo)
        hi = b_hi if hi is None else np.maximum(hi, b_hi)
    return None if lo is None else (lo, hi)


def placement_offset(
    doc: ClayDoc, incoming: ClayDoc, beside_uids: Any = ()
) -> np.ndarray:
    """Where :func:`merge_into` should shift *incoming*'s roots, in world space.

    **With a selection** (*beside_uids* names at least one object with
    geometry): *incoming*'s combined box is placed with its minimum X at the
    selection's own box's maximum X, plus a gap -- ten per cent of whichever
    box's largest extent is bigger, floored at :data:`MIN_GAP` so two small
    objects never end up touching. Z is centred on the selection's own box;
    Y's minimum lands on 0, so the new object sits on the ground beside the
    old one rather than floating at whatever height its own origin happened
    to be authored at.

    **With nothing selected** (or a selection with no geometry -- a lone empty,
    say): X and Z are centred on the world origin and Y's minimum is 0, which
    is "dropped at the spawn point, standing on the grid" -- the same reading
    an import onto an *empty* document already gets today.

    An *incoming* with no geometry at all (every object empty, or none)
    returns the zero vector: there is nothing to place, and :func:`merge_into`
    still runs -- it always did, an empty group is not an error -- just with
    no translation applied.
    """
    incoming_box = _world_box(incoming, [obj.uid for obj in incoming.objects])
    if incoming_box is None:
        return np.zeros(3, dtype="f8")
    in_lo, in_hi = incoming_box
    in_center = (in_lo + in_hi) * 0.5
    in_extent = float(np.max(in_hi - in_lo))

    selection_box = _world_box(doc, [int(uid) for uid in beside_uids])
    if selection_box is None:
        return np.array([-in_center[0], -in_lo[1], -in_center[2]], dtype="f8")

    sel_lo, sel_hi = selection_box
    sel_extent = float(np.max(sel_hi - sel_lo))
    gap = max(MIN_GAP, 0.1 * max(in_extent, sel_extent))
    offset_x = (sel_hi[0] + gap) - in_lo[0]
    offset_z = ((sel_lo[2] + sel_hi[2]) * 0.5) - in_center[2]
    offset_y = -in_lo[1]
    return np.array([offset_x, offset_y, offset_z], dtype="f8")


def merge_into(
    doc: ClayDoc,
    incoming: ClayDoc,
    *,
    offset: Any,
    label: str = "Generate",
    group_name: str | None = None,
) -> list[Obj]:
    """Adopt every object of *incoming* into *doc*, as **one** undo step.

    **Materials are appended, never renumbered.** Every one of *incoming*'s
    palette entries becomes a new slot at the end of *doc*'s own palette
    (:meth:`ClayDoc.add_material`, which is itself append-only for exactly
    this reason -- an insertion would renumber every existing mesh's own
    per-face indices), and every incoming face's material index -- and every
    incoming object's own default slot -- is shifted by that same offset. An
    existing object's palette reference never moves.

    **Names are made unique against the document being merged into**
    (:func:`~.ops.next_name`), checked against *doc*'s names plus whatever
    this call has already placed -- so two same-named objects arriving
    together do not collide with each other either.

    **Only root translations move.** *offset* is added to the local
    translation of every incoming object whose own ``parent`` is ``None`` --
    a child's placement is relative to its parent, which is moving with it,
    so shifting it a second time would double the offset for every object
    not at the top of *incoming*'s own hierarchy. Uids are not reminted:
    :func:`~.document.new_uid` is a process-wide counter, so an object built
    in a *different* :class:`ClayDoc` in this same process already cannot
    collide with one in this one.

    **One step, one group.** ``doc.history.mark()``/``collapse_since`` fold
    the material appends, the object insert and (when more than one root
    arrived) the grouping into a single entry, the same shape
    :meth:`ClayDoc.group` and :meth:`ClayDoc.add_material_and_assign` already
    use for their own multi-call gestures -- and its label is set to *label*
    afterwards, because a bare fold reads as "compound" in the history panel,
    which says nothing about what is about to be undone. A single-root
    import is not wrapped in a group at all: an already-one-thing hierarchy
    gains nothing from a second empty around it.

    Returns the objects that were added (**not** including a synthesizing
    group empty, if one was made) in *doc*'s own object list, so a caller can
    select them or read back their uids.
    """
    if not incoming.objects:
        return []

    mark = doc.history.mark()
    base = len(doc.materials)
    for material in incoming.materials:
        doc.add_material(material)

    taken = {o.name for o in doc.objects}
    offset_arr = np.asarray(offset, dtype="f8")
    built: list[Obj] = []
    for obj in incoming.objects:
        mesh = obj.mesh
        shifted_mesh: bm.Mesh = dataclasses.replace(
            mesh, material=np.asarray(mesh.material, dtype="i4") + base
        )
        name = obj.name
        if name in taken:
            name = mesh_ops.next_name(name, taken)
        taken.add(name)
        translation = obj.translation
        if obj.parent is None:
            translation = np.asarray(obj.translation, dtype="f8") + offset_arr
        built.append(
            dataclasses.replace(
                obj,
                mesh=shifted_mesh,
                name=name,
                material=int(obj.material) + base,
                translation=translation,
            )
        )

    added = doc.add_objects(built, label=label)
    roots = [obj.uid for obj in added if obj.parent is None]
    # clay-13 (the 2026-09-23 audit): this used to branch on ``len(added) > 1``
    # -- the *total* objects added, not the number of roots -- so a
    # one-root, multi-object hierarchy (a root plus its children) was wrapped
    # in a synthesizing group exactly like this docstring says it would not
    # be. Only more than one *root* actually needs a group to hold them.
    if len(roots) > 1:
        doc.group(roots, name=group_name)
    doc.history.collapse_since(mark)
    top = doc.history.top
    if top is not None:
        top.label = label
    return added
