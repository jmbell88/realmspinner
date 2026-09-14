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
:func:`apply` reads exactly the four facts :func:`~.clay.scratch.diff`
snapshot at preview time -- ``base_head``, ``base_selection``,
``base_element_mode``, and (new here) whether the tab itself is even still
open -- and refuses the moment any of them has moved, naming the one
sentence every one of those refusals shares: the document changed, and the
preview no longer describes it.

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
    tab = state.get(tab_uid) if tab_uid else None

    if tab is None:
        tab = clay_mode.new_document(ctx)
    else:
        refusal = _refusal(ctx, tab, diff)
        if refusal is not None:
            return {"ok": False, "message": refusal}

    changed = clay_scratch.transplant(tab.doc, scratch, diff)
    view = getattr(ctx, "clay_view", None)
    if view is not None:
        view.clear_preview()
    return {"ok": True, "changed": changed, "tab_uid": tab.uid}


def _refusal(ctx: Any, tab: Any, diff: Any) -> str | None:
    if tab.saving:
        return "the document changed -- preview again"
    view = getattr(ctx, "clay_view", None)
    if view is not None and view.grabbing:
        return "the document changed -- preview again"
    doc = tab.doc
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
