"""A disposable clone of a :class:`~.document.ClayDoc`, for previewing an
agent's edits before they touch the document the user is looking at.

**Why a clone rather than a dry-run flag threaded through every op.** Clay's
tool surface (``agent_clay.py``) already runs a person's whole vocabulary --
booleans, element ops, figure presets -- against a real ``ClayDoc``, and none
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
already picked, and any generator or boolean the scratch ran already built
the mesh it built. Replaying the tool call would mint a *different* uid for
the same "add a box" and run the heavy op (a boolean, a subdivide) a second
time for nothing. Transplant instead moves the *scratch's own objects and
edits* onto the real document through the document's ordinary doors --
``add_objects``, ``remove_object``, ``set_mesh``, ``set_props``,
``set_transform`` -- so every rule those doors already enforce (the generator
freeze, the undo bookkeeping, the material-user refusal) applies to a
transplanted edit exactly as it would to a person's.

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
docstring for the fix, and ``tests/clay/test_scratch.py::
test_transplant_with_material_removal_keeps_face_indices_right`` for the
regression.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import document as bd


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
    """

    added: set[int] = field(default_factory=set)
    removed: set[int] = field(default_factory=set)
    mesh_changed: set[int] = field(default_factory=set)
    transform_changed: set[int] = field(default_factory=set)
    props_changed: dict[int, set[str]] = field(default_factory=dict)
    order_changed: bool = False
    materials_changed: bool = False

    base_head: int = 0
    base_selection: set[int] = field(default_factory=set)
    base_element_mode: str = "object"

    @property
    def changed(self) -> set[int]:
        """Every uid the preview touches at all, added ones included."""
        return self.added | self.mesh_changed | self.transform_changed | set(self.props_changed)

    @property
    def empty(self) -> bool:
        return not (
            self.added
            or self.removed
            or self.mesh_changed
            or self.transform_changed
            or self.props_changed
            or self.order_changed
            or self.materials_changed
        )


_PROP_FIELDS = ("name", "visible", "generator", "params", "material")


def diff(base: bd.ClayDoc, scratch: bd.ClayDoc) -> PreviewDiff:
    """What a scratch run changed, relative to the document it was cloned from.

    Meshes and materials compare by identity (see the module docstring);
    transforms and props compare by value, because a scratch run that read a
    number and wrote the same one back should preview as "nothing changed"
    -- exactly what ``set_transform``/``set_props`` already decide for a
    person's edit.
    """
    base_uids = [obj.uid for obj in base.objects]
    scratch_uids = [obj.uid for obj in scratch.objects]
    base_set, scratch_set = set(base_uids), set(scratch_uids)
    added = scratch_set - base_set
    removed = base_set - scratch_set
    common = base_set & scratch_set

    mesh_changed: set[int] = set()
    transform_changed: set[int] = set()
    props_changed: dict[int, set[str]] = {}
    for uid in common:
        b, s = base.by_uid(uid), scratch.by_uid(uid)
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

    order_changed = [u for u in base_uids if u in common] != [
        u for u in scratch_uids if u in common
    ]
    materials_changed = len(base.materials) != len(scratch.materials) or any(
        a is not b for a, b in zip(base.materials, scratch.materials, strict=False)
    )

    return PreviewDiff(
        added=added,
        removed=removed,
        mesh_changed=mesh_changed,
        transform_changed=transform_changed,
        props_changed=props_changed,
        order_changed=order_changed,
        materials_changed=materials_changed,
        base_head=base.history.head,
        base_selection=set(base.selection),
        base_element_mode=base.element_mode,
    )


def _transplant_materials(doc: bd.ClayDoc, scratch: bd.ClayDoc) -> None:
    """Materials before objects -- ``remove_material`` renumbers every face's
    ``material`` index, so it has to run before a transplanted mesh's own
    face indices are trusted to mean the same slot they meant in the scratch.
    """
    shared = min(len(doc.materials), len(scratch.materials))
    for index in range(shared):
        if doc.materials[index] is not scratch.materials[index]:
            doc.set_material(index, scratch.materials[index])
    for index in range(shared, len(scratch.materials)):
        doc.add_material(scratch.materials[index])
    # Extra trailing entries on the *document* side that the scratch run
    # dropped -- removed highest index first, so an earlier removal cannot
    # renumber an index this loop has not visited yet.
    for index in range(len(doc.materials) - 1, shared - 1, -1):
        doc.remove_material(index)


def transplant(doc: bd.ClayDoc, scratch: bd.ClayDoc, diff_: PreviewDiff) -> bool:
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
    try:
        for uid in diff_.removed:
            if uid in {o.uid for o in doc.objects}:
                doc.remove_object(uid)

        if diff_.added:
            added_objs = [scratch.by_uid(uid) for uid in diff_.added]
            doc.add_objects(added_objs)

        touched = (diff_.mesh_changed | set(diff_.props_changed)) - diff_.added
        for uid in touched:
            s = scratch.by_uid(uid)
            if uid in diff_.mesh_changed:
                doc.set_mesh(uid, s.mesh, keep_generator=True)
            fields = diff_.props_changed.get(uid)
            if fields:
                doc.set_props(uid, **{f: getattr(s, f) for f in fields})

        for uid in diff_.transform_changed - diff_.added:
            s = scratch.by_uid(uid)
            doc.set_transform(uid, translation=s.translation, rotation=s.rotation, scale=s.scale)

        if diff_.order_changed:
            wanted = [uid for uid in (o.uid for o in scratch.objects) if uid in {
                o.uid for o in doc.objects
            }]
            for index, uid in enumerate(wanted):
                doc.move_object(uid, index)

        if diff_.materials_changed:
            _transplant_materials(doc, scratch)
    finally:
        doc.history.collapse_since(mark)

    top = doc.history.top
    if top is not None:
        top.label = "Familiar: agent preview applied"
    return True
