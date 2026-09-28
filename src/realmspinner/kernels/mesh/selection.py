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
privates are what the panes and the tests call), and ``clay_ops`` now calls
straight through to this module -- which is what makes ``clay_ops`` free of any
reference to ``clay_mode`` at all.

**Nothing here toasts.** A refusal is a string handed back to the caller, and
the caller decides what a refusal looks like -- the mode layer shows a toast,
and a test reads the list. That is the same boundary the rest of ``clay/``
keeps, and it is why this module can be tested without a ``ctx``.
"""

from __future__ import annotations

from typing import Any

from . import elements as el

__all__ = ["delete_selected", "duplicate_selected", "invert", "select_all"]


def select_all(doc: Any) -> None:
    """Everything *visible*, in the current mode's sense of everything.

    The object branch used to take every object and the element branch below it
    has always skipped the invisible ones, which is the asymmetry rather than a
    choice: ``visible=False`` is documented to mean an object does not render,
    does not export and **cannot be picked**, and Ctrl+A was picking them. It
    only showed once merge existed -- Ctrl+A then Ctrl+M pulled geometry the
    user could not see into the survivor -- but Delete and Duplicate were reading
    the same selection all along.
    """
    if doc.element_mode == "object":
        doc.select([obj.uid for obj in doc.objects if obj.visible])
        return
    for obj in doc.objects:
        if obj.visible:
            doc.set_element_sel(obj.uid, el.select_all(obj.mesh, doc.element_mode))


def invert(doc: Any) -> None:
    """Ctrl+Shift+I. In object mode it inverts which objects are selected."""
    if doc.element_mode == "object":
        # Visible only, for ``select_all``'s reason: inverting into a hidden
        # object selects something the user cannot see or click back off.
        doc.select([o.uid for o in doc.objects if o.visible and o.uid not in doc.selection])
        return
    for obj in doc.objects:
        if not obj.visible:
            continue
        doc.set_element_sel(
            obj.uid,
            el.invert(obj.mesh, doc.element_sel_of(obj.uid), doc.element_mode),
        )


def delete_selected(doc: Any) -> list[str]:
    """Delete what is selected, in whatever sense the current mode means it.

    -> the refusals, one sentence each, for the caller to show. A refusal on one
    object does not abandon the others, which is the ``run_mesh_op`` rule.

    In an element mode this **never** falls through to removing objects. A user
    in face mode pressing Delete means "get rid of these faces", and quietly
    deleting the whole object instead is the most destructive possible
    misreading of one keystroke -- the more so because the object selection in
    an element mode is derived from the element selection, so every object with
    anything selected inside it would go.

    The object-mode branch folds one ``mark()``/``collapse_since()`` gesture
    (see below) around one ``remove_object`` call per selected uid, so the
    whole selection is one undo step -- the 2026-09-06 audit (finding clay-01)
    found that selecting three objects and pressing Delete once took three
    presses of Ctrl+Z to undo, landing on a two-deleted/one-restored state the
    user never produced. The removals are processed in *descending* index
    order for the same reason :meth:`~.document.ClayDoc.join_objects` records
    its own doomed list that way: a ``CompoundEdit`` undoes in reverse, which
    re-inserts them ascending, which is the only order in which every
    recorded index is still correct. The 2026-09-26 audit (finding
    clay-mesh-core-01) found this branch had drifted from ``remove_object`` by
    reimplementing its bookkeeping inline instead of calling it -- see the
    comment at the call site for what that cost.

    The element-mode branch below folds the same way, through
    ``UndoStack.mark``/``collapse_since``: each object's ``doc.set_mesh`` call
    already pushes its own step (a ``MeshEdit``, plus a generator-freeze
    ``ObjectPropsEdit`` when that object had one), and there is no "build the
    edit but do not push it" form of ``set_mesh`` to collect from instead. The
    2026-09-08 audit (second run, finding clay-10) found that with three boxes
    and every face selected, one Delete pushed three of those steps and one
    Ctrl+Z restored one box while leaving two empty -- the direct-call twin of
    clay-01, reachable from ``tests/modes/clay/test_select.py`` and any other
    caller that reaches this function without going through ``clay_ops.run``
    (which already folds everything an op pushes, but only for callers that go
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
        doomed = sorted({int(u) for u in doc.selection}, key=doc.index_of, reverse=True)
        if not doomed:
            return []
        # The 2026-09-22 audit, finding clay-01: this branch pops straight out
        # of ``doc.objects`` rather than going through ``remove_object``, which
        # is the only place that had ever checked ``obj.locked`` -- so a
        # locked object selected in the outliner (locking is deliberately not
        # a picking door, only a drag/transform one) was removed with no
        # refusal at all. Collecting the refusal here, before anything is
        # popped, keeps this function's own "a refusal on one object does not
        # abandon the others" promise for the unlocked rest of the selection.
        refusals: list[str] = []
        removable = []
        for uid in doomed:
            obj = doc.by_uid(uid)
            if obj.locked:
                refusals.append(f"{obj.name!r} is locked.")
                continue
            removable.append(uid)
        if not removable:
            return refusals
        # The 2026-09-26 audit, finding clay-mesh-core-01: this branch used to
        # reimplement remove_object's bookkeeping inline (a hand-built
        # CompoundEdit around a hand-popped ``doc.objects.pop``) and it never
        # re-parented the removed object's children the way remove_object
        # does, nor popped ``_evaluated`` -- deleting a parent left each
        # child's ``parent`` naming a uid the document no longer carries, and
        # read_rblk refuses to reopen that file (a saved model or journal
        # copy). Calling remove_object itself, per uid, inside a
        # mark()/collapse_since() gesture -- the same primitive the
        # element-mode branch below already uses to fold its own per-object
        # set_mesh steps -- gets every one of remove_object's guarantees
        # (reparenting, the _mesh_stamps and _evaluated pops) for the whole
        # selection at once, still as one undo step, with no logic to
        # duplicate or drift out of sync.
        mark = doc.history.mark()
        for uid in removable:
            doc.remove_object(uid)
        doc.history.collapse_since(mark)
        return refusals
    refusals = []
    mark = doc.history.mark()
    for uid in list(doc.element_sel):
        obj = doc.by_uid(uid)
        # The 2026-09-22 audit, finding clay-02: this used to call
        # ``delete_faces``/``set_mesh`` unconditionally for every object with
        # something in ``doc.element_sel`` -- a locked object could be in
        # there at all because neither ``pick_element`` nor
        # ``_commit_marquee`` skipped one the way object-mode ``pick_face``
        # already does, and once it was, ``set_mesh`` raised past this loop
        # (only the ``delete_faces`` call was wrapped in ``try/except
        # OpError``), aborting after earlier objects had already been
        # rewritten. Checking the lock before either call keeps this
        # function's own "a refusal on one object does not abandon the
        # others" promise instead of a partial commit disguised as a total
        # refusal.
        if obj.locked:
            refusals.append(f"{obj.name!r} is locked.")
            continue
        faces = el.convert(obj.mesh, doc.element_sel_of(uid), "face")
        # The 2026-09-19 audit, finding clay-03: this used to ``continue`` past
        # an object whose selection converted to zero faces -- a partial
        # vertex or edge selection, the ordinary case rather than an edge
        # case -- which swallowed ``delete_faces``'s own refusal ("Select at
        # least one face to delete.") before it could reach the caller's
        # toast. Calling ``delete_faces`` unconditionally lets it raise that
        # refusal itself, and collecting it here keeps the same "a refusal on
        # one object does not abandon the others" contract this function's
        # docstring already promises for every other ``OpError``: an object
        # whose selection does convert to a face is still deleted.
        try:
            mesh, sel = ops_topo.delete_faces(obj.mesh, faces)
        except el.OpError as error:
            refusals.append(str(error))
            continue
        doc.set_mesh(uid, mesh, select=sel)
    doc.history.collapse_since(mark)
    return refusals


def duplicate_selected(doc: Any) -> list[int]:
    """A copy of every selected object, selected in its place. -> the new uids.

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

    taken = [obj.name for obj in doc.objects]
    copies = []
    # The 2026-09-14 audit's clay-07: this used to iterate ``doc.selection``
    # directly, a plain ``set``, so a Ctrl+D on several objects produced
    # copies in whatever order the set's hash buckets happened to land in
    # rather than the order the objects actually sit in the outliner --
    # sorting by ``doc.index_of`` restores the document's own order.
    old_to_new: dict[int, int] = {}
    for uid in sorted(doc.selection, key=doc.index_of):
        copy = ops.duplicate(doc.by_uid(uid), bd.new_uid(), taken=taken)
        taken.append(copy.name)
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
    copies = [
        replace(copy, parent=old_to_new[copy.parent]) if copy.parent in old_to_new else copy
        for copy in copies
    ]
    doc.add_objects(copies)
    return [copy.uid for copy in copies]
