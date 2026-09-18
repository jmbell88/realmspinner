"""The Familiar programme's GL-side half: previewing an agent's scratch edit
and turning it into a real one on the frame thread.

**Why this lives at studio level and not inside ``studio/familiar/``.**
``studio/familiar/`` is meant to be the pure half of Familiar -- T3 adds
``contract.py``, ``retrieval.py``, ``router.py``, ``threads.py`` and
``cards/`` there, none of which may import imgui, moderngl, pygame or the
service layer (see ``tests/familiar/test_familiar_imports.py``). This module
imports ``agent_clay``, which reaches ``clay_view`` (``moderngl``) and
``panes.clay_tools`` (``imgui_bundle``) -- exactly the kind of window import
that package must never carry, even two relative hops away
(``tests/_pure_packages.py::_module_roots`` now resolves relative imports
precisely so a chain like that cannot hide). ``agent_clay.py`` and the other
studio-level modules (``clay_mode.py``, ``clay_view.py``) already sit
alongside imgui/moderngl imports without being under a "pure" package, so
this is that same shelf, not a new one.

See :mod:`~warlock.kernels.mesh.scratch` for the clone/diff/transplant
mechanics this module drives, and :func:`build`/:func:`run_scratch`'s own
docstrings for the sandboxed ``ctx`` a scratch run executes an agent tool
call against.

**Why apply refuses rather than trusting the diff.** A preview is computed
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
built its preview against :func:`build`'s own throwaway clone of an empty
document, say), and there is nothing to have "changed" under a preview
nobody was looking at. :func:`apply` mints exactly one new document in that
case -- through ``clay_mode.new_document``, the same door every other empty
session uses -- and transplants onto it, rather than refusing a preview that
was never shown against anything a person could edit out from under it.

**Why a whole fake ``ctx`` rather than reusing the real one with the tab
swapped in.** ``agent_clay.Session.tab_uid`` names one tab inside *the app's*
``ClayState`` (``ctx.state.clay``), which is the interactive user's state --
every other open document lives there too. Handing a scratch run the real
``ctx`` would mean either mutating that shared state (inserting a scratch tab
alongside the user's real ones, for the length of one tool call, on whatever
thread the preview runs on) or trusting every one of ``agent_clay``'s ~30
handlers to notice a special case that does not otherwise exist anywhere in
that module. :class:`ScratchCtx` instead carries its *own*, private
``ClayState`` holding exactly one tab -- the scratch clone -- so
``clay_mode.ensure(ctx)`` and ``_tab(ctx, session)`` do exactly what they
always do and land on a document nothing else can see.

**No ``svc``, no ``cache``, no ``viewer``.** Read ``agent_clay.py``'s own
module docstring: every handler that would need one of those three is a tool
this module refuses before the door (see :data:`PREVIEW_EXCLUDED`) --
``clay_export`` mints a Library job through ``ctx.svc`` and invalidates
``ctx.cache``, and the four reference tools are session-scoped bookkeeping
that never touches a document at all and previewing them is meaningless (an
agent comparing a render against a reference is not an edit to transplant).
Every other handler in ``agent_clay._HANDLERS`` reaches ``ctx`` only through
``ctx.state`` (via ``clay_mode.ensure``) and, for a render, ``ctx.viewer`` --
and ``clay_render`` is in the excluded set too, since a preview run has
nothing to render *to*: it has no viewport of its own and must not touch the
app's. Omitting the three attributes entirely, rather than stubbing them out
to no-ops, is deliberate: an excluded tool never reaches them (refused before
``agent_clay.call`` is ever invoked), and a *newly added* handler that reached
for one of them would raise ``AttributeError`` instead of silently touching
GL or the service layer from a scratch run -- a loud failure in exactly the
place a quiet one would be worst.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from ...kernels.mesh import document as bd
from ...kernels.mesh import scratch as clay_scratch
from .. import agent_clay, clay_mode
from ..clay_state import ClayState, ClayTab

#: Tools a scratch run refuses before ``agent_clay.call`` is ever reached.
#:
#: ``clay_export`` mints a Library asset (a real, permanent row) and touches
#: ``ctx.svc``/``ctx.cache``, neither of which a preview run holds. ``clay_
#: render`` wants a viewport (``ctx.viewer.ctx``) a scratch run has no
#: business building -- rendering the agent's own hypothetical document
#: would be a second, throwaway ``ClayView`` per preview, for a picture
#: nothing asks to see. The four reference tools
#: (``clay_reference_add``/``_get``/``_list``/``_remove``) are session-scoped
#: bookkeeping about pictures to match against, not edits to the document at
#: all -- there is nothing in them for :func:`~.clay.scratch.diff` to see,
#: and a name added to one session's private reference list is not a change
#: a Familiar preview should offer to "apply".
PREVIEW_EXCLUDED = frozenset(
    {
        "clay_export",
        "clay_render",
        "clay_reference_add",
        "clay_reference_get",
        "clay_reference_list",
        "clay_reference_remove",
    }
)


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
        # against a real tab at all (a scratch run built from :func:`build`'s
        # own throwaway clone) -- and it must stay reachable only through a
        # *falsy* ``tab_uid``, the one signal a caller has for "there was
        # nothing to preview against". A real, previously-open tab always
        # arrives with its own non-empty uid, so collapsing this case with
        # "the tab is gone" below (as a bare ``state.get(tab_uid) if tab_uid
        # else None`` used to, treating both as `tab is None`) would silently
        # mint a fresh document for a preview whose tab the user had since
        # closed -- exactly the refusal this module's first paragraph
        # promises.
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
                    "that document is not the one in front -- switch to it, then preview again"
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


@dataclass
class ScratchCtx:
    """A private, disposable stand-in for the app ``ctx``, holding exactly
    one Clay tab: the scratch clone a preview edits.

    Deliberately carries no ``svc``, ``cache`` or ``viewer`` attribute at
    all -- see the module docstring for why an ``AttributeError`` here is the
    right failure mode rather than a stub.
    """

    state: Any
    tab_uid: str
    settings: Any = field(default_factory=SimpleNamespace)
    toasts: list[tuple[str, str]] = field(default_factory=list)

    def toast(self, message: str, level: str = "info") -> None:
        """Collected, never shown -- a scratch run is invisible to the user
        at the keyboard, so nothing it does may reach the real toast queue."""
        self.toasts.append((message, level))


def build(doc: bd.ClayDoc) -> ScratchCtx:
    """Clone ``doc`` and wrap it in a fresh :class:`ScratchCtx`.

    The returned ctx's one tab is the *only* thing its private ``ClayState``
    holds, and its ``tab_uid`` already names it -- so the caller never has to
    reach past this function to mint one, and no ``agent_clay`` handler
    reachable through :func:`run_scratch` can ever see ``create=True`` land
    on the "mint a new document" branch (``agent_clay._tab``): the tab it
    would otherwise create already exists.
    """
    scratch_doc = clay_scratch.clone(doc)
    tab = ClayTab(doc=scratch_doc, title="Familiar scratch")
    state = ClayState(docs=[tab])
    return ScratchCtx(state=SimpleNamespace(clay=state), tab_uid=tab.uid)


def run_scratch(ctx: ScratchCtx, name: str, args: dict) -> dict:
    """Run one agent tool call against a scratch clone, never the real
    document.

    ``name`` is refused, before ``agent_clay.call`` is ever reached, when it
    is in :data:`PREVIEW_EXCLUDED`. The session's ``tab_uid`` is pinned to
    the scratch ctx's own tab *before* the call, which is what keeps a
    ``create=True`` tool (``clay_add_primitive``, ``clay_add_figure``,
    ``clay_add_mesh``) from ever reaching ``clay_mode.new_document`` --
    ``agent_clay._tab`` only takes that branch when ``session.tab_uid`` is
    falsy, and here it never is.
    """
    if name in PREVIEW_EXCLUDED:
        return agent_clay.fail(
            f"{name} cannot be previewed -- it is excluded from Familiar scratch runs.",
        )
    session = agent_clay.Session(tab_uid=ctx.tab_uid)
    return agent_clay.call(ctx, session, name, args)
