"""What "everything", "the opposite", "delete this" and "copy this" mean.

Four operations that are entirely about the *document* -- which objects or
elements are selected, and what happens to them -- and that had come to live in
``studio/modes/clay/mode.py``, the mode layer, because that is where the keyboard
shortcuts firing them are. The ops registry then had to import the mode layer to
reach them, and the mode layer imports the registry: a cycle between a
UI-shaped module and a headless one, held together by both sides importing the
other lazily inside function bodies.

They are here instead, in the headless package, where the rest of this
vocabulary already is. The mode layer keeps same-signature wrappers (its
privates are what the panes and the tests call), and ``studio/modes/clay/ops.py``
now calls straight through to this module -- which is what makes that module free
of any reference to ``studio/modes/clay/mode.py`` at all.

**Nothing here toasts.** A refusal is a string handed back to the caller, and
the caller decides what a refusal looks like -- the mode layer shows a toast,
and a test reads the list. That is the same boundary the rest of ``clay/``
keeps, and it is why this module can be tested without a ``ctx``.
"""

from __future__ import annotations

from typing import Any

from . import elements as el

__all__ = ["delete_selected", "duplicate_selected", "invert", "select_all", "with_descendants"]


def with_descendants(doc: Any, uids: Any) -> set[int]:
    """*uids* and every object under any of them.

    What "copy this" means for a group: the 2026-10-09 second audit's clay-63 found
    Duplicate and Mirror Copy on a selected group copied only the empty, because
    a group *is* an empty and its children are separate objects the selection
    does not name. Copying a parent without its children is never what pressing
    Duplicate on a group means.

    One parent -> children map for the whole call, every selected uid's subtree
    walked through it: the 2026-10-11 audit's clay-21 found a
    ``doc.descendants`` call per selected uid, each rebuilding the map, which
    was 0.8 s for a 4,096-object selection (the same shape the 2026-10-10
    audit's clay-32 and clay-71 closed for Delete and ``descendants``). The
    visited set also stops a hand-built cycle, as ``descendants`` does.
    """
    out: set[int] = set(uids)
    kids: dict[int | None, list[int]] = {}
    for obj in doc.objects:
        kids.setdefault(obj.parent, []).append(obj.uid)
    stack = list(out)
    while stack:
        for child in kids.get(stack.pop(), ()):
            if child not in out:
                out.add(child)
                stack.append(child)
    return out


def select_all(doc: Any) -> None:
    """Everything *visible*, in the current mode's sense of everything.

    The object branch used to take every object and the element branch below it
    has always skipped the invisible ones, which is the asymmetry rather than a
    choice: ``visible=False`` is documented to mean an object does not render,
    does not export and **cannot be picked**, and Ctrl+A was picking them. It
    only showed once merge existed -- Ctrl+A then Ctrl+M pulled geometry the
    user could not see into the survivor -- but Delete and Duplicate were reading
    the same selection all along.

    In an element mode it is every element of the shapes being edited
    (:func:`_edited_objects`), or of every visible shape when none is.
    """
    if doc.element_mode == "object":
        doc.select([obj.uid for obj in doc.objects if obj.visible])
        return
    for obj in _edited_objects(doc):
        doc.set_element_sel(obj.uid, el.select_all(obj.mesh, doc.element_mode))


def _element_pickable(obj: Any) -> bool:
    """The eligibility every element door shares: a visible object."""
    return bool(obj.visible)


def _edited_objects(doc: Any) -> list[Any]:
    """The objects an element-mode "all" or "invert" works on.

    The shapes being edited (``doc.selection``) when there are any, else every
    visible object. The selection rule keeps the edited shape selected, so
    Ctrl+Shift+I with one shape open used to pull every *other* object in the
    scene into the edit -- inverting a shape you were not looking at. With no
    shape being edited there is nothing to scope to, and "all" keeps its old
    meaning, which is also how a first pick elsewhere would have chosen one.
    """
    pickable = [obj for obj in doc.objects if _element_pickable(obj)]
    if not doc.selection:
        return pickable
    return [obj for obj in pickable if obj.uid in doc.selection]


def invert(doc: Any) -> None:
    """Ctrl+Shift+I. In object mode it inverts which objects are selected; in an
    element mode, which elements are picked on the shapes being edited."""
    if doc.element_mode == "object":
        # Visible only, for ``select_all``'s reason: inverting into a hidden
        # object selects something the user cannot see or click back off.
        doc.select([o.uid for o in doc.objects if o.visible and o.uid not in doc.selection])
        return
    for obj in _edited_objects(doc):
        doc.set_element_sel(
            obj.uid,
            el.invert(obj.mesh, doc.element_sel_of(obj.uid), doc.element_mode),
        )


# What a Points/Edges pick is called, and what makes a face of it, in the
# refusal Delete gives a pick that covers no whole face.
_PICK_TERMS = {"vertex": ("vertices", "corners"), "edge": ("edges", "edges")}


def delete_selected(doc: Any) -> list[str]:
    """Delete what is selected, in whatever sense the current mode means it.

    -> the refusals, one sentence each, for the caller to show. A refusal on one
    object does not abandon the others, which is the ``run_mesh_op`` rule.

    In an element mode this **never** falls through to removing objects. A user
    in face mode pressing Delete means "get rid of these faces", and quietly
    deleting the whole object instead is the most destructive possible
    misreading of one keystroke -- the more so because the object selection in
    an element mode is the shape being edited, so the whole shape would go
    whether or not anything was picked inside it.

    The object-mode branch makes one :meth:`~.document.ClayDoc.remove_objects`
    call for the whole selection and opens no gesture of its own: that door
    pushes one step itself, so the whole selection is one undo step -- the
    2026-09-06 audit (finding clay-01) found that selecting three objects and
    pressing Delete once took three presses of Ctrl+Z to undo, landing on a
    two-deleted/one-restored state the user never produced. ``remove_objects``
    records the removals highest list index first, for the same reason
    :meth:`~.document.ClayDoc.join_objects` records its own doomed list that
    way: a ``CompoundEdit`` undoes in reverse, which re-inserts them ascending,
    which is the only order in which every recorded index is still correct.
    (The 2026-10-11 audit's clay-22: this paragraph described the older shape,
    an undo gesture folded around one ``remove_object`` call per uid, which
    the 2026-10-10 audit's clay-32 replaced.) The 2026-09-26
    audit (finding clay-mesh-core-01) found this branch had drifted from
    ``remove_object`` by reimplementing its bookkeeping inline instead of
    calling it -- see the comment at the call site for what that cost.

    The element-mode branch below folds the same way, through
    ``UndoStack.mark``/``collapse_since``: each object's ``doc.set_mesh`` call
    already pushes its own step (a ``MeshEdit``, plus a generator-freeze
    ``ObjectPropsEdit`` when that object had one), and there is no "build the
    edit but do not push it" form of ``set_mesh`` to collect from instead. The
    2026-09-08 audit (second run, finding clay-10) found that with three boxes
    and every face selected, one Delete pushed three of those steps and one
    Ctrl+Z restored one box while leaving two empty -- the direct-call twin of
    clay-01, reachable from ``tests/modes/clay/test_select.py`` and any other
    caller that reaches this function without going through
    ``studio/modes/clay/ops.py``'s ``run`` (which already folds
    everything an op pushes, but only for callers that go
    through it -- see the 2026-09-07 audit's clay-02 in
    ``studio/modes/clay/mode.py``). ``mark``/``collapse_since`` is the
    primitive built for exactly this composed-op shape (its own docstring in
    ``core/undo.py`` names "delete these eight rows"), and it already folds
    nothing into nothing: a single touched object still pushes the one plain
    step ``set_mesh`` (or, in the object-mode branch, ``remove_object``)
    always pushed, so the existing single-object undo tests are unaffected.
    """
    from . import ops_topo

    if doc.element_mode == "object":
        if not doc.selection:
            return []
        # One uid -> object map for the whole press (the 2026-10-10 audit's
        # clay-32: ``sorted(key=doc.index_of)`` and a ``by_uid`` per uid made
        # this quadratic before ``remove_object`` ran at all).
        by_uid = {obj.uid: obj for obj in reversed(doc.objects)}
        removable = [
            uid
            for uid in doc.selection
            # The 2026-10-03 audit's clay-41: hiding an object leaves it
            # selected, and Delete took it along with the visible ones --
            # against the manual's "a hidden object stays hidden and untouched
            # even when it is selected". Skipped silently, as Merge and Ctrl+A
            # already do: nothing the user can see was refused.
            if by_uid[int(uid)].visible
        ]
        if not removable:
            return []
        # The 2026-09-26 audit, finding clay-mesh-core-01: this branch used to
        # reimplement remove_object's bookkeeping inline and never re-parented
        # the removed object's children, so deleting a parent left each child
        # naming a uid the document no longer carries, and read_rblk refuses
        # to reopen that file. It then called ``remove_object`` per uid inside
        # a mark()/collapse_since() gesture, which got every guarantee back and
        # was quadratic (the 2026-10-10 audit's clay-32: 1.5 s at 4,000
        # objects, every call rebuilding the uid map and popping from the
        # middle of the list). ``remove_objects`` does the same work in one
        # pass and pushes one step itself, so no gesture is needed here -- and
        # none can be left open by a refusal (the 2026-10-07 audit's clay-11).
        # A refusal on one object is collected and the rest still go.
        return doc.remove_objects(removable)
    refusals = []
    mark = doc.history.mark()
    try:
        for uid in list(doc.element_sel):
            obj = doc.by_uid(uid)
            # clay-41 (2026-10-03 audit): a hidden object can still hold an
            # element selection, and its faces are not on screen to be deleted.
            if not obj.visible:
                continue
            faces = el.convert(obj.mesh, doc.element_sel_of(uid), "face")
            # The 2026-09-19 audit, finding clay-03: this used to ``continue``
            # past an object whose selection converted to zero faces -- a
            # partial vertex or edge selection, the ordinary case rather than
            # an edge case -- which swallowed ``delete_faces``'s own refusal
            # ("Select at least one face to delete.") before it could reach the
            # caller's toast. Calling ``delete_faces`` unconditionally lets it
            # raise that refusal itself, and collecting it here keeps the same
            # "a refusal on one object does not abandon the others" contract
            # this function's docstring already promises for every other
            # ``OpError``: an object whose selection does convert to a face is
            # still deleted.
            #
            # The 2026-10-10 audit's clay-33: that refusal is written in face
            # terms, and a lone vertex or edge in Points/Edges mode got "Select
            # at least one face to delete." beside an Inspector that says Delete
            # "deletes the picked elements". When something *is* picked and
            # covers no whole face, say what Delete actually does with it. An
            # empty pick keeps the face sentence -- there is nothing to
            # explain. Worded here, not in ``ops_topo``, which stays a pure
            # face-count refusal for every other caller.
            picked = doc.element_sel_of(uid).count(doc.element_mode)
            if len(faces.faces) == 0 and picked and doc.element_mode in _PICK_TERMS:
                noun, part = _PICK_TERMS[doc.element_mode]
                refusals.append(
                    f"Only faces fully covered by the picked {noun} are deleted. "
                    f"Pick all of a face's {part}, or switch to Faces mode."
                )
                continue
            # The 2026-10-14 audit, finding clay-mesh-core-01: ``set_mesh`` sat outside this
            # ``try``, so a weighted object whose edit changes its point count (it raises
            # ``OpError`` from ``_with_skin_of``) escaped the loop: a plain prop picked beside a
            # rigged body had already lost its faces as a pushed step while Delete reported only
            # the body's refusal -- and which objects changed depended on dict order. The door's
            # refusal is collected like ``delete_faces``'s own and the loop carries on.
            try:
                mesh, sel = ops_topo.delete_faces(obj.mesh, faces)
                doc.set_mesh(uid, mesh, select=sel)
            except el.OpError as error:
                refusals.append(str(error))
                continue
    finally:
        doc.history.collapse_since(mark)
    return refusals


def duplicate_selected(doc: Any) -> list[int]:
    """A copy of every selected object -- and of everything under a selected group --
    with the copies of the selected ones selected in their place. -> the new uids.

    ``taken`` grows as it goes so two copies of one name in a single press do
    not both land on ``Box.001``.

    Built and inserted through :meth:`~.document.ClayDoc.add_objects` rather
    than one ``add_object`` call per copy, for ``add_objects``'s own reason:
    the 2026-09-06 audit (finding clay-01) found that duplicating three
    selected objects pushed three ``ObjectAddEdit`` steps, so one Ctrl+Z
    undid only one of the three copies the single keypress had just made.
    ``add_objects`` already selects everything it inserts and is a no-op on
    an empty list, so an empty selection here pushes nothing.
    """
    from dataclasses import replace

    from . import document as bd
    from . import ops

    taken = ops.UsedNames(obj.name for obj in doc.objects)
    copies = []
    # The 2026-09-14 audit's clay-07: this used to iterate ``doc.selection``
    # directly, a plain ``set``, so a Ctrl+D on several objects produced
    # copies in whatever order the set's hash buckets happened to land in
    # rather than the order the objects actually sit in the outliner --
    # sorting by ``doc.index_of`` restores the document's own order.
    old_to_new: dict[int, int] = {}
    # The 2026-10-09 second audit's clay-63: a selected group is an empty, and its
    # children are other objects -- Duplicate copied the empty alone. A selected
    # visible object now takes its descendants with it (the re-pointing below
    # then hangs each child's copy off its parent's copy). Document order is
    # kept rather than parents-first: the copies keep their local transforms, so
    # nothing here needs a parent's copy to exist before its child's, and the
    # outliner order (the 2026-09-14 audit's clay-07) is the one a user sees.
    #
    # The 2026-10-03 audit's clay-41: a *selected* hidden object stays untouched,
    # since a copy of it would appear (selected) in a scene the user cannot see
    # the original in -- hence ``chosen`` filters on ``visible``. A hidden child
    # of a visible group is part of the group and its copy stays hidden, as it was.
    chosen = [uid for uid in doc.selection if doc.by_uid(uid).visible]
    for uid in sorted(with_descendants(doc, chosen), key=doc.index_of):
        copy = ops.duplicate(doc.by_uid(uid), bd.new_uid(), taken=taken)
        taken.add(copy.name)
        old_to_new[uid] = copy.uid
        copies.append(copy)
    # The 2026-09-26 audit (clay-mesh-core-06): ``ops.duplicate`` copies
    # ``parent`` verbatim, so a Ctrl+D on a selected parent *and* child left
    # the child copy parented to the *original* parent rather than its new
    # sibling copy -- the two hierarchies tangled together instead of the
    # keypress producing one independent duplicate of the whole selection.
    # Both parent and child copy share the parent's own local transform
    # unchanged (``ops.duplicate`` never touches translation/rotation/scale),
    # so re-pointing ``parent`` at the new copy -- without recomputing local
    # TRS the way ``set_parent(keep_world=True)`` would -- keeps the copy
    # exactly where it already sits. A copy whose parent was not itself part
    # of this selection is left naming the original, unchanged: it was never
    # duplicated, so there is no sibling copy to move to.
    #
    # The 2026-10-14 audit, finding clay-mesh-core-02: the same was never done for a *skin*, so
    # duplicating a bone together with the mesh weighted to it gave a copy still driven by the
    # ORIGINAL skeleton while the copied bones drove nothing, and ``rig_problems`` said nothing.
    # ``ops.rebind_skin`` moves a skin onto the copied bones when every bone it names was copied
    # in this press, and leaves it on the original rig otherwise (as ``parent`` is left).
    copies = [
        ops.rebind_skin(
            replace(copy, parent=old_to_new[copy.parent]) if copy.parent in old_to_new else copy,
            old_to_new,
        )
        for copy in copies
    ]
    doc.add_objects(copies)
    # Only the copies of what the user picked stay selected: ``add_objects``
    # selected every copy, and a group's copied children riding inside the
    # selection would be moved twice by the first gizmo drag (once with their
    # parent, once on their own).
    if chosen and doc.element_mode == "object":
        doc.select([old_to_new[uid] for uid in chosen])
    return [copy.uid for copy in copies]
