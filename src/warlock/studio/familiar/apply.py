"""Turning a Familiar ghost preview into a real edit, on the frame thread.

**Why this refuses rather than trusting the diff.** A preview is computed
once, against the document as it stood at that instant -- but the frame
thread is free-running, and a person can keep editing, undo, redo or start a
drag in the seconds between "here is what I would do" and "Apply". Applying
the diff anyway would land it on a document it was never actually computed
against: an undo the user made in between would come back the moment the
transplant's own edits landed on top of it, a live drag would have its own
in-progress edit clobbered mid-gesture, and a selection or element-mode
change would make ``transplant``'s object-mode assumptions (``add_objects``
selecting what it just added, for one) act on the wrong thing. So
:func:`apply` reads exactly the facts :func:`~.clay.scratch.diff`
snapshot at preview time -- ``base_doc_id``, ``base_head``, ``base_selection``,
``base_element_mode``, and (new here) whether the tab itself is even still
open -- and refuses the moment any of them has moved, naming the one
sentence every one of those refusals shares: the document changed, and the
preview no longer describes it. The one deliberate exception is whether the
tab is the *active* one: nothing about the document moved, the user is just
looking at something else, so it gets its own sentence (see :func:`_refusal`'s
own note).

**No tab open is not a refusal.** An agent's scratch run can start from a
document that was never opened in the interactive UI at all (a session that
built its preview against ``familiar.build``'s own throwaway clone of an
empty document, say), and there is nothing to have "changed" under a preview
nobody was looking at. :func:`apply` mints exactly one new document in that
case -- through ``clay_mode.new_document``, the same door every other empty
session uses -- and transplants onto it, rather than refusing a preview that
was never shown against anything a person could edit out from under it.
"""

from __future__ import annotations

from typing import Any

from .. import clay_mode
from ..clay import scratch as clay_scratch


def apply(ctx: Any, tab_uid: str, diff: Any, scratch: Any) -> dict:
    """Transplant ``diff`` (computed against ``scratch``) onto the real
    document named by ``tab_uid``. -> a result dict, ``ok`` or a refusal.

    Refuses, naming the reason, when: the tab is gone and was not empty to
    begin with (see the module docstring's second paragraph for the one
    case that is not a refusal); the tab is mid-save; its view is mid-drag,
    -orbit or -pan (``view.grabbing``); its undo head has moved; or its
    selection or element mode differs from what the preview was shown
    against. Every refusal is the same sentence, because the fix is the same
    regardless of which fact moved: preview again, against what is there now.
    """
    state = clay_mode.ensure(ctx)

    if not tab_uid:
        # The 2026-09-14 audit (agents-07): this branch is the module
        # docstring's second paragraph -- a preview that was never shown
        # against a real tab at all (an agent's scratch run built from
        # ``familiar.build``'s own throwaway clone) -- and it must stay
        # reachable only through a *falsy* ``tab_uid``, the one signal a
        # caller has for "there was nothing to preview against". A real,
        # previously-open tab always arrives with its own non-empty uid, so
        # collapsing this case with "the tab is gone" below (as a bare
        # ``state.get(tab_uid) if tab_uid else None`` used to, treating both
        # as `tab is None`) would silently mint a fresh document for a
        # preview whose tab the user had since closed -- exactly the
        # refusal this module's first paragraph promises.
        tab = clay_mode.new_document(ctx)
    else:
        tab = state.get(tab_uid)
        if tab is None:
            # The previewed tab existed and is simply gone now (closed since
            # the preview was computed) -- not the "never opened" case
            # above. Refused with the same sentence every other refusal in
            # this module uses, per the module docstring: the fix is always
            # "preview again", regardless of which fact moved.
            return {"ok": False, "message": "the document changed -- preview again"}
        # Not the "document moved" family below -- the document snapshotted
        # by the diff may be entirely untouched, but if it is not the one on
        # screen the user was never shown this preview against what they are
        # now looking at. Its own sentence, per the module docstring's one
        # named exception, and only reachable here (the real, non-empty
        # ``tab_uid`` branch) -- the falsy-``tab_uid`` mint-a-document path
        # above has no ``state.active_uid`` of its own to compare against.
        if tab.uid != state.active_uid:
            return {
                "ok": False,
                "message": (
                    "that document is not the one in front -- switch to it, "
                    "then preview again"
                ),
            }
        refusal = _refusal(ctx, tab, diff)
        if refusal is not None:
            return {"ok": False, "message": refusal}

    changed = clay_scratch.transplant(tab.doc, scratch, diff)
    view = getattr(ctx, "clay_view", None)
    if view is not None:
        view.clear_preview()
    result: dict[str, Any] = {"ok": True, "changed": changed, "tab_uid": tab.uid}
    result["kept_materials"] = list(getattr(changed, "kept_materials", []))
    return result


def _refusal(ctx: Any, tab: Any, diff: Any) -> str | None:
    if tab.saving:
        return "the document changed -- preview again"
    view = getattr(ctx, "clay_view", None)
    if view is not None and view.grabbing:
        return "the document changed -- preview again"
    doc = tab.doc
    # Identity, not just the three values below: a revert/reload/journal-
    # recovery path can swap in a *fresh* ``ClayDoc`` whose head/selection/
    # element_mode all happen to coincide with the base's own (both sides are
    # commonly a just-opened document), which the value checks alone would
    # miss -- see ``PreviewDiff.base_doc_id``'s own docstring.
    if id(doc) != diff.base_doc_id:
        return "the document changed -- preview again"
    if doc.history.head != diff.base_head:
        return "the document changed -- preview again"
    if set(doc.selection) != diff.base_selection:
        return "the document changed -- preview again"
    if doc.element_mode != diff.base_element_mode:
        return "the document changed -- preview again"
    return None


def discard(ctx: Any, tab_uid: str) -> None:
    """Drop the ghost and leave the document byte-identical.

    The document was never touched -- a preview is shown on a clone, and
    :func:`apply` is the only door that ever writes onto the real one -- so
    this is nothing but releasing the view's preview slot.
    """
    view = getattr(ctx, "clay_view", None)
    if view is not None:
        view.clear_preview()
