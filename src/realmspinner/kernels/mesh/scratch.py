"""A disposable clone of a :class:`~.document.ClayDoc`, for previewing an
agent's edits before they touch the document the user is looking at.

**Why a clone rather than a dry-run flag threaded through every op.** Clay's
tool surface (``studio/modes/clay/agent/dispatch.py``) already runs a person's whole vocabulary --
joins, element ops, primitives -- against a real ``ClayDoc``, and none
of it was written to ask "what would this do" without doing it. Cloning the
document and running the same tools against the clone costs one shallow copy
and answers the question exactly, with no second code path for every op to
keep honest.

**Sharing is the point, not an optimisation.** :func:`clone` shares every
``Mesh`` the base document holds (meshes are immutable, see ``mesh.py``'s own
docstring) and shares every material object in the palette, because neither
of those bought anything by being copied -- an op that touches one always
replaces it with a *new* object (``Mesh -> Mesh``, ``set_material`` always
writes a replacement). What can never be shared is anything an op mutates in
place: the TRS arrays (``Obj.__post_init__`` already copies them once; the
clone copies them again, off the base's copies) and each object's ``params``
dict, which several op handlers read with ``dict(obj.params)`` and then
mutate before handing back -- sharing the base's dict would let a scratch
edit silently rewrite the object the user is looking at.

**Uids need no special handling.** :func:`~.document.new_uid` is a
process-global counter (``document.py``'s own module comment: "per process,
not per document"), so an object minted while a scratch run is in effect
comes from the *same* sequence the base document's objects did and can never
repeat one of them -- there is no separate "scratch counter" to keep below the
base's high-water mark, because there is only ever one counter for the whole
process. (This is the one spec claim this module's design review turned out
not to need: a first draft reserved a counter offset for the clone, and it
was dead code -- the global counter already guarantees it.)

**Diffing by identity, not by value.** :func:`diff` compares meshes with
``is not`` and materials the same way, for the reason ``document.py``'s
module docstring gives ``to_model`` and the GPU cache: every op is
``X -> X`` over a replacement, so identity *is* "did this change". Transforms
are the one place value comparison is right instead -- ``set_transform``
already treats "wrote the same numbers back" as no change via
``np.array_equal``, and a preview should agree with the document about what a
no-op edit looks like.

**Transplant replays no tool call.** It cannot: an added object's uid is
already picked, and any generator or op the scratch ran already built
the mesh it built. Replaying the tool call would mint a *different* uid for
the same "add a box" and run the heavy op (a subdivide) a second
time for nothing. Transplant instead carries the scratch's result onto the
real document through the document's ordinary doors --
``add_objects``, ``remove_object``, ``set_mesh``, ``set_props``,
``set_visibility``, ``set_transform`` -- so every rule those doors already
enforce (the generator freeze, the undo bookkeeping, the material-user refusal,
hiding deselects) applies to a transplanted edit exactly as it would to a person's.

What it carries is a *copy*, not the scratch's own objects: the new objects are
rebuilt and the changed props deep-copied, so an in-place edit on either side
never moves the other (the 2026-10-09 second audit's clay-93 found this
paragraph saying transplant "moves" them, and the code handing the scratch's
``params`` dict to the document). Meshes and materials are shared by identity,
as :func:`clone` shares them, because both are immutable.

**Objects transplant before materials, not after.** A first draft of
:func:`transplant` copied ``document.py``'s own rule for a *live* edit --
"materials first, because ``remove_material`` renumbers faces" -- and it is
backwards here. A live ``remove_material`` renumbers faces because it is
mutating the one document those faces already belong to; a scratch run's own
``remove_material`` call already renumbered the *scratch's* faces, so a
transplanted mesh already carries indices relative to the scratch's final
palette. Shrinking the real document's palette before those meshes have
landed asks ``remove_material`` to refuse a slot the document's own faces
still point at -- exactly the refusal it exists to raise, firing on a state
this transplant is one step from making true. See :func:`transplant`'s own
docstring for the fix, and ``tests/modes/clay/test_scratch.py::
test_transplant_with_material_removal_keeps_face_indices_right`` for the
regression.
"""

from __future__ import annotations

import contextlib
import copy
import weakref
from dataclasses import dataclass, field

import numpy as np

from . import document as bd
from . import elements as el


def clone(doc: bd.ClayDoc) -> bd.ClayDoc:
    """A new, independent :class:`~.document.ClayDoc` a scratch run can edit
    freely without the base document ever seeing it.

    Objects keep their uid and name, share the base's ``Mesh`` (meshes are
    immutable; nothing here needs to copy one), copy the TRS arrays and
    ``params`` dict (both are mutated in place by real callers) and keep the
    same material index and generator name. The palette is a fresh list
    holding the *same* material objects -- an edit to one replaces it with a
    new object (see ``set_material``), so nothing here needs to deep-copy a
    ``gltf.Material``. Selection, element mode and the per-object element
    selection are copied so a preview run inherits exactly what the user had
    selected; history is empty, because a scratch run's own undo stack is
    private to it and never shown to anyone.
    """
    objects = [
        bd.Obj(
            uid=obj.uid,
            name=obj.name,
            mesh=obj.mesh,  # shared: Mesh is immutable
            translation=np.array(obj.translation, dtype="f8", copy=True),
            rotation=np.array(obj.rotation, dtype="f8", copy=True),
            scale=np.array(obj.scale, dtype="f8", copy=True),
            generator=obj.generator,
            params=dict(obj.params),
            visible=obj.visible,
            material=obj.material,
            # ``parent`` is an int or ``None`` -- carried as-is rather than
            # defensively copied, since every write to it replaces the value
            # wholesale, through ``set_parent``. Left out until the
            # 2026-09-19 field-sweep test in ``tests/modes/clay/test_scratch.py``
            # caught it: a scratch run previewing anything that reads a
            # parented object's world matrix (``world_bounds``) saw every
            # object as a root.
            parent=obj.parent,
        )
        for obj in doc.objects
    ]
    scratch = bd.ClayDoc(objects=objects, materials=list(doc.materials))
    scratch.selection = set(doc.selection)
    scratch.element_mode = doc.element_mode
    scratch.element_sel = dict(doc.element_sel)
    # ``ClayDoc.__init__`` already builds a fresh, empty ``UndoStack`` -- there
    # is nothing above for this function to have pushed to it, so the clone's
    # history starts exactly where a brand-new document's does.
    return scratch


@dataclass
class PreviewDiff:
    """Everything :func:`transplant` needs, and nothing it has to recompute.

    ``base_head``/``base_selection``/``base_element_mode`` are the document's
    state *at preview time* -- read once here rather than by whoever applies
    the preview later, because by then the user may have kept editing, and
    ``apply`` needs to compare against what was true when the preview was
    shown, not against whatever is true now.

    ``base_doc_id`` is ``id(base)``, not a value snapshot: a revert, reload or
    journal-recovery path can replace a tab's document with a *fresh*
    ``ClayDoc`` whose head/selection/element_mode all happen to coincide with
    the base's own (a freshly-opened document is a common starting point on
    both sides), which the three value checks alone would miss entirely.
    Identity is the only thing that actually distinguishes "the same document,
    edited" from "a different document that happens to match".

    **A bare ``id()`` outlives the object it names.** The 2026-10-10 audit's
    clay-40: once the base was freed CPython could hand its address to a new
    ``ClayDoc``, so a stale preview passed the "same document" check onto a
    different document with the same head, selection and mode. The diff now also
    holds a ``weakref`` to the base, and the moment the base is collected the
    ref's callback zeroes ``base_doc_id`` (no live object has address 0), so the
    outside consumers that compare ``id(doc) != diff.base_doc_id`` keep working
    unchanged and a recycled address can no longer match. :meth:`is_base` is the
    check to prefer: it compares identity through the ref.
    """

    added: set[int] = field(default_factory=set)
    removed: set[int] = field(default_factory=set)
    mesh_changed: set[int] = field(default_factory=set)
    transform_changed: set[int] = field(default_factory=set)
    props_changed: dict[int, set[str]] = field(default_factory=dict)
    # The 2026-09-23 audit's clay-02: kept apart from ``props_changed`` rather
    # than folded into ``_PROP_FIELDS`` because ``set_props`` refuses
    # ``parent`` by name -- see that tuple's own comment -- so this needs its
    # own leg through ``set_parent`` in :func:`transplant`. Maps a reparented
    # uid to its *new* parent (``None`` for "became a root").
    parent_changed: dict[int, int | None] = field(default_factory=dict)
    order_changed: bool = False
    materials_changed: bool = False

    base_head: int = 0
    base_selection: set[int] = field(default_factory=set)
    base_element_mode: str = "object"
    base_doc_id: int = 0
    _base_ref: weakref.ref | None = field(default=None, repr=False, compare=False)

    def bind_base(self, base: bd.ClayDoc) -> None:
        """Remember *base* weakly, and zero ``base_doc_id`` when it is freed."""
        self.base_doc_id = id(base)
        me = weakref.ref(self)

        def _gone(_ref: weakref.ref, me: weakref.ref = me) -> None:
            diff_ = me()
            if diff_ is not None:
                diff_.base_doc_id = 0

        self._base_ref = weakref.ref(base, _gone)

    def is_base(self, doc: bd.ClayDoc) -> bool:
        """Whether *doc* is the very document this preview was built from."""
        return self._base_ref is not None and self._base_ref() is doc

    @property
    def changed(self) -> set[int]:
        """Every uid the preview touches at all, added ones included."""
        return (
            self.added
            | self.mesh_changed
            | self.transform_changed
            | set(self.props_changed)
            | set(self.parent_changed)
        )

    @property
    def empty(self) -> bool:
        return not (
            self.added
            or self.removed
            or self.mesh_changed
            or self.transform_changed
            or self.props_changed
            or self.parent_changed
            or self.order_changed
            or self.materials_changed
        )


# ``parent`` is deliberately absent from this tuple: :meth:`~.document.ClayDoc.set_props`
# refuses it by name ("use set_parent"), so adding it here would make
# :func:`transplant` raise on the very first scratch run that reparented
# anything. :func:`diff` tracks a reparent separately, in its own
# ``parent_changed`` field, and :func:`transplant` carries it through
# :meth:`~.document.ClayDoc.set_parent` -- see :func:`diff`'s and
# :func:`transplant`'s own docstrings. Until the 2026-09-23 audit's clay-02,
# this was a known, undocumented gap instead: a scratch run that reparented
# an object under ``keep_world=True`` recomputes the object's local TRS
# relative to its *new* parent's frame (see ``document.set_parent``'s own
# docstring), and that recomputed TRS *did* flow through
# ``transform_changed``/``set_transform`` below, landing the new-parent-
# relative numbers on an object Apply left under its *old* parent -- the
# object jumped the instant Apply ran, even though the preview picture the
# user approved showed it standing still.
_PROP_FIELDS = ("name", "visible", "generator", "params", "material")


def diff(base: bd.ClayDoc, scratch: bd.ClayDoc) -> PreviewDiff:
    """What a scratch run changed, relative to the document it was cloned from.

    Meshes and materials compare by identity (see the module docstring);
    transforms and props compare by value, because a scratch run that read a
    number and wrote the same one back should preview as "nothing changed"
    -- exactly what ``set_transform``/``set_props`` already decide for a
    person's edit.
    """
    # The 2026-10-10 audit's clay-39: ``by_uid`` per common uid was a linear
    # scan each, quadratic at the object ceiling (0.49 s at 4,000, on the frame
    # thread). One map per document per call, the shape clay-71 gave
    # ``ClayDoc._uid_map`` for the same reason.
    base_map, scratch_map = base._uid_map(), scratch._uid_map()
    base_uids = [obj.uid for obj in base.objects]
    scratch_uids = [obj.uid for obj in scratch.objects]
    base_set, scratch_set = set(base_uids), set(scratch_uids)
    added = scratch_set - base_set
    removed = base_set - scratch_set
    common = base_set & scratch_set

    mesh_changed: set[int] = set()
    transform_changed: set[int] = set()
    props_changed: dict[int, set[str]] = {}
    parent_changed: dict[int, int | None] = {}
    for uid in common:
        b, s = base_map[uid], scratch_map[uid]
        if b.mesh is not s.mesh:
            mesh_changed.add(uid)
        if not (
            np.array_equal(b.translation, s.translation)
            and np.array_equal(b.rotation, s.rotation)
            and np.array_equal(b.scale, s.scale)
        ):
            transform_changed.add(uid)
        changed_fields = {f for f in _PROP_FIELDS if getattr(b, f) != getattr(s, f)}
        if changed_fields:
            props_changed[uid] = changed_fields
        # The 2026-09-23 audit's clay-02: tracked apart from ``props_changed``
        # -- see ``_PROP_FIELDS``'s own comment for why ``parent`` cannot sit
        # in that tuple.
        if b.parent != s.parent:
            parent_changed[uid] = s.parent

    # The 2026-10-07 audit's clay-24: this compared the common uids only, so an
    # object the scratch inserted mid-list (``separate`` puts its pieces where
    # the source stood) never set it and ``transplant`` appended them. What
    # transplant produces without a reorder is the common objects in base
    # order, then the added ones; the preview's order is what the user approved.
    produced = [u for u in base_uids if u in common] + [u for u in scratch_uids if u in added]
    order_changed = produced != scratch_uids
    materials_changed = len(base.materials) != len(scratch.materials) or any(
        a is not b for a, b in zip(base.materials, scratch.materials, strict=False)
    )

    result = PreviewDiff(
        added=added,
        removed=removed,
        mesh_changed=mesh_changed,
        transform_changed=transform_changed,
        props_changed=props_changed,
        parent_changed=parent_changed,
        order_changed=order_changed,
        materials_changed=materials_changed,
        base_head=base.history.head,
        base_selection=set(base.selection),
        base_element_mode=base.element_mode,
    )
    result.bind_base(base)
    return result


def _transplant_materials(doc: bd.ClayDoc, scratch: bd.ClayDoc) -> list[int]:
    """Materials go last -- see the module docstring's "Objects transplant
    before materials, not after" section for why. -> the indices
    ``remove_material`` refused to drop.

    A refusal here is not a bug in the transplant: ``remove_material``'s
    ``material_users`` also counts faces on objects the *undo stack* still
    holds (for redo), and the scratch clone's own stack starts empty (see
    :func:`clone`'s own comment), so a removal that succeeded on the scratch
    -- nothing there used the slot, undo history included -- can still be
    refused when replayed against the real document, whose undo stack may
    hold an object that named it. The caller surfaces these rather than
    letting the palette silently fail to shrink.
    """
    shared = min(len(doc.materials), len(scratch.materials))
    # The 2026-10-10 audit's clay-12: this was read *after* the adds below, so
    # the trailing-removal loop started at ``shared`` and tried to drop the
    # slots the preview had just added -- an unused one vanished (the document
    # ended a slot short of what the preview showed) and a used one was
    # refused and reported as a slot the document would not drop. Only
    # document slots beyond the scratch's palette, counted before any add,
    # are the scratch's own removals.
    doc_count = len(doc.materials)
    for index in range(shared):
        if doc.materials[index] is not scratch.materials[index]:
            doc.set_material(index, scratch.materials[index])
    for index in range(shared, len(scratch.materials)):
        doc.add_material(scratch.materials[index])
    # Extra trailing entries on the *document* side that the scratch run
    # dropped -- removed highest index first, so an earlier removal cannot
    # renumber an index this loop has not visited yet.
    kept: list[int] = []
    for index in range(doc_count - 1, shared - 1, -1):
        if not doc.remove_material(index):
            kept.append(index)
    return kept


@dataclass
class TransplantResult:
    """What :func:`transplant` did, beyond the plain "did anything happen"
    every existing caller already asserts as a truthy/``is False`` check.

    ``__bool__`` is always ``True`` for an instance of this class -- the
    empty-diff fast path returns bare ``False`` instead of one of these, so
    every existing ``assert changed`` / ``assert clay_scratch.transplant(...)``
    keeps passing unmodified, and only a caller that wants more reads
    ``.kept_materials``.
    """

    kept_materials: list[int] = field(default_factory=list)

    def __bool__(self) -> bool:
        return True


def transplant(doc: bd.ClayDoc, scratch: bd.ClayDoc, diff_: PreviewDiff) -> bool | TransplantResult:
    """Apply a preview's changes to the real document, as **one** undo step.

    Never replays a tool call -- see the module docstring for why. Applies
    through the document's own doors, and **objects before materials** --
    the opposite of the order a real-time single-document edit uses, for a
    reason specific to a cross-document transplant. ``document.py``'s own
    ``remove_material`` renumbers faces *because* it is mutating the one
    document a face array already belongs to; here the scratch's own
    ``remove_material`` call already renumbered the *scratch's* face arrays
    (``document.py``'s ``_shift_materials`` walks every object whose mesh
    names the removed slot, which is exactly what makes such an object part
    of :attr:`PreviewDiff.mesh_changed`), so a transplanted mesh already
    carries indices relative to the scratch's *final* palette. Removing the
    real document's doomed slot before those meshes have landed would ask
    ``remove_material`` to refuse a slot the document's own faces still
    point at -- the refusal that door exists for, firing on a state this
    transplant is one step away from making true. Applying every object edit
    first, so every face in ``doc`` already carries final numbering, is what
    lets the material-list resize run last and land clean.
    """
    if diff_.empty:
        return False
    mark = doc.history.mark()
    # The 2026-10-10 audit's clay-39: a uid set rebuilt per removed object and a
    # ``by_uid`` scan per touched one made Apply quadratic on the frame thread.
    scratch_map = scratch._uid_map()
    try:
        live = {o.uid for o in doc.objects}
        for uid in diff_.removed:
            if uid in live:
                # The 2026-09-22 audit's clay-17: this loop runs first, ahead
                # of the mesh/transform loops below that tolerate a refusal
                # the same way -- the base object may have moved since the
                # preview was built -- and one refusal here must not abort
                # the whole Apply before its siblings get a chance.
                with contextlib.suppress(el.OpError):
                    doc.remove_object(uid)
                    live.discard(uid)

        if diff_.added:
            # The 2026-10-07 audit's clay-53: ``diff_.added`` is a ``set``, so
            # iterating it put the new objects in hash order -- uids 125..129
            # straddle a multiple of the table size and landed 128, 129, 125,
            # 126, 127 -- while the preview showed them in ``scratch.objects``
            # order. The document's object order is what the outliner, the
            # exports and ``order_changed`` below all read.
            # The 2026-10-09 audit's clay-37: these were the scratch's own
            # ``Obj`` instances, so the document and the scratch shared TRS
            # arrays and the params dict -- an in-place edit on one moved the
            # other. ``clone`` copies the same fields for the same reason.
            added_objs = [
                bd.Obj(
                    uid=o.uid,
                    name=o.name,
                    mesh=o.mesh,
                    translation=np.array(o.translation, dtype="f8", copy=True),
                    rotation=np.array(o.rotation, dtype="f8", copy=True),
                    scale=np.array(o.scale, dtype="f8", copy=True),
                    generator=o.generator,
                    params=dict(o.params),
                    visible=o.visible,
                    material=o.material,
                    parent=o.parent,
                )
                for o in scratch.objects
                if o.uid in diff_.added
            ]
            try:
                doc.add_objects(added_objs)
            except el.OpError:
                # ``add_objects`` refuses a batch that would take the document
                # past the object or triangle ceiling the writer enforces, and
                # the base may have grown since the preview was built (or the
                # removals above may have failed). The refusal arrives after
                # the removals above were already pushed, so an Apply that let
                # it propagate left half of itself on the undo stack. Apply
                # lands whole or not at all: whatever this call pushed is
                # reversed and forgotten (no redo of a half-apply), and the
                # caller sees the same ``False`` as for a preview with nothing
                # to apply.
                while doc.history.head != mark and doc.undo_unrecorded():
                    pass
                return False

        for uid, new_parent in diff_.parent_changed.items():
            if uid in diff_.added:
                # An added object already carries its final ``parent`` --
                # ``add_objects`` above placed it there directly, so there is
                # no existing document link for ``set_parent`` to move.
                continue
            # The 2026-09-23 audit's clay-02: reparenting here with
            # ``keep_world=False`` moves only the ``parent`` link, leaving
            # the object's local TRS exactly as it sits on the real document
            # right now -- the transform_changed loop just below then
            # overwrites that TRS with the scratch's own local numbers, which
            # were computed relative to this *same* new parent (the scratch
            # run reparented under ``keep_world=True``, see ``_PROP_FIELDS``'s
            # comment above). Doing both in this order reproduces exactly
            # what the preview showed; doing only the transform half, as
            # before this fix, landed the new parent's local numbers on an
            # object that was still hanging under its old parent and the
            # object visibly jumped. Tolerated the same way a removed base
            # object is elsewhere in this function: the base object, or the
            # intended new parent, may have been removed since the preview
            # was shown.
            # ``KeyError`` too (the 2026-10-07 audit's clay-55): the document's
            # doors raise it, not an OpError, for a uid that vanished.
            with contextlib.suppress(el.OpError, KeyError):
                doc.set_parent(uid, new_parent, keep_world=False)

        touched = (diff_.mesh_changed | set(diff_.props_changed)) - diff_.added
        for uid in touched:
            s = scratch_map[uid]
            mesh_landed = True
            if uid in diff_.mesh_changed:
                # The 2026-09-20 audit's clay-13: state can move between
                # preview and apply, and an uncaught refusal here aborted
                # the whole transplant against this docstring's own "the
                # rest of the transplant still lands" -- so it is
                # tolerated, and the loop moves on.
                try:
                    doc.set_mesh(uid, s.mesh, keep_generator=True)
                except (el.OpError, KeyError):
                    mesh_landed = False
            fields = diff_.props_changed.get(uid)
            if fields and not mesh_landed:
                # The 2026-10-10 audit's clay-38: the refusal was swallowed but
                # the scratch's ``generator``/``params`` were still applied, so
                # the object claimed parameters its stored mesh was not built
                # from -- what ``set_mesh``'s freeze rule exists to prevent.
                # Those two describe the mesh; the rest (name, visibility,
                # material) stand on their own.
                fields = fields - {"generator", "params"}
            if fields:
                # The 2026-10-09 second audit's clay-86 and clay-93.  ``visible``
                # goes through ``set_visibility`` -- the door every hide in Clay
                # uses, and the only one that deselects what it hides -- because
                # ``set_props`` left a selected hidden object that
                # ``selection_problems`` flags.  The calls sit inside this
                # transplant's own gesture, so Apply is still one undo step.
                # And every other prop is a *copy*: the scratch's ``params`` dict
                # (and any list inside it) used to become the document object's,
                # the aliasing clay-37 fixed for added objects and missed here.
                with contextlib.suppress(KeyError):
                    if "visible" in fields:
                        doc.set_visibility({uid: s.visible})
                    others = {
                        f: copy.deepcopy(getattr(s, f)) for f in fields if f != "visible"
                    }
                    if others:
                        doc.set_props(uid, **others)

        for uid in diff_.transform_changed - diff_.added:
            s = scratch_map[uid]
            # Same tolerance as the mesh branch above, and the same
            # clay-13 incident: an uncaught refusal here aborted every
            # transform still queued behind it in this loop, not just this
            # object's own.
            with contextlib.suppress(el.OpError, KeyError):
                doc.set_transform(
                    uid, translation=s.translation, rotation=s.rotation, scale=s.scale
                )

        if diff_.order_changed:
            live = {o.uid for o in doc.objects}
            wanted = [o.uid for o in scratch.objects if o.uid in live]
            for index, uid in enumerate(wanted):
                doc.move_object(uid, index)

        kept_materials: list[int] = []
        if diff_.materials_changed:
            kept_materials = _transplant_materials(doc, scratch)
    finally:
        doc.history.collapse_since(mark)

    # The 2026-10-07 audit's clay-55: when every door was a no-op nothing was
    # pushed, ``top`` was the user's own previous step, and it was relabelled
    # as the agent's. Only a head that actually moved past the mark is ours.
    if doc.history.head == mark:
        return TransplantResult(kept_materials=kept_materials) if kept_materials else False
    top = doc.history.top
    if top is not None:
        top.label = "Familiar: agent preview applied"
    return TransplantResult(kept_materials=kept_materials)
