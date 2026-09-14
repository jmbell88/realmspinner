"""What "everything", "the opposite", "delete this" and "copy this" mean.

Four operations that are entirely about the *document* -- which objects or
elements are selected, and what happens to them -- and that had come to live in
``studio/clay_mode.py``, the mode layer, because that is where the keyboard
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

    The object-mode branch pushes one ``CompoundEdit`` for the whole selection
    rather than one ``ObjectRemoveEdit`` per ``remove_object`` call, the same
    shape :meth:`~.document.ClayDoc.join_objects` already uses -- the
    2026-09-06 audit (finding clay-01) found that selecting three objects and
    pressing Delete once took three presses of Ctrl+Z to undo, landing on a
    two-deleted/one-restored state the user never produced. The removals are
    recorded in *descending* index order for ``join_objects``'s own reason: a
    ``CompoundEdit`` undoes in reverse, which re-inserts them ascending, which
    is the only order in which every recorded index is still correct.

    The element-mode branch below folds the same way, but through
    ``UndoStack.mark``/``collapse_since`` rather than a hand-built
    ``CompoundEdit``: each object's ``doc.set_mesh`` call already pushes its
    own step (a ``MeshEdit``, plus a generator-freeze ``ObjectPropsEdit`` when
    that object had one), and there is no "build the edit but do not push it"
    form of ``set_mesh`` to collect from instead. The 2026-09-08 audit
    (second run, finding clay-10) found that with three boxes and every face selected, one
    Delete pushed three of those steps and one Ctrl+Z restored one box while
    leaving two empty -- the direct-call twin of clay-01, reachable from
    ``tests/clay/test_select.py`` and any other caller that reaches this
    function without going through ``clay_ops.run`` (which already folds
    everything an op pushes, but only for callers that go through it -- see
    the 2026-09-07 audit's clay-02 in ``clay_mode.py``). ``mark``/
    ``collapse_since`` is the primitive built for exactly this composed-op
    shape (its own docstring in ``studio/undo.py`` names "delete these eight
    rows"), and it already folds nothing into nothing: a single touched
    object still pushes the one plain step ``set_mesh`` always pushed, so the
    existing single-object undo tests are unaffected.
    """
    from . import ops_topo
    from .edits import ObjectRemoveEdit

    if doc.element_mode == "object":
        from ..undo import CompoundEdit

        doomed = sorted({int(u) for u in doc.selection}, key=doc.index_of, reverse=True)
        if not doomed:
            return []
        edits: list[Any] = []
        for uid in doomed:
            index = doc.index_of(uid)
            obj = doc.objects.pop(index)
            doc.selection.discard(uid)
            doc.element_sel.pop(uid, None)
            # The 2026-09-11 audit, finding clay-06: this branch reimplements
            # ClayDoc.remove_object's bookkeeping inline (to build one
            # CompoundEdit for the whole selection, the clay-01 fix) and used
            # to stop one line short of it -- a deleted uid's _mesh_stamps
            # entry, which pins its Mesh's full CSR arrays in memory, was
            # never popped. Same shape of bug already named and fixed twice
            # for ClayState.manifold: the rule is every way an object leaves
            # doc.objects, not only the one call site an earlier audit
            # happened to reach. Mirrors remove_object's own comment.
            doc._mesh_stamps.pop(uid, None)
            edits.append(ObjectRemoveEdit(index, obj))
        doc.history.push(edits[0] if len(edits) == 1 else CompoundEdit(edits))
        doc.touch()
        return []
    refusals: list[str] = []
    mark = doc.history.mark()
    for uid in list(doc.element_sel):
        obj = doc.by_uid(uid)
        faces = el.convert(obj.mesh, doc.element_sel_of(uid), "face")
        if el.is_empty(faces):
            continue
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
    from . import document as bd
    from . import ops

    taken = [obj.name for obj in doc.objects]
    copies = []
    # The 2026-09-14 audit's clay-07: this used to iterate ``doc.selection``
    # directly, a plain ``set``, so a Ctrl+D on several objects produced
    # copies in whatever order the set's hash buckets happened to land in
    # rather than the order the objects actually sit in the outliner --
    # sorting by ``doc.index_of`` restores the document's own order.
    for uid in sorted(doc.selection, key=doc.index_of):
        copy = ops.duplicate(doc.by_uid(uid), bd.new_uid(), taken=taken)
        taken.append(copy.name)
        copies.append(copy)
    doc.add_objects(copies)
    return [copy.uid for copy in copies]
