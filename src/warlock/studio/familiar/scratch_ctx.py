"""The sandboxed ``ctx`` a Familiar preview runs an agent tool call against.

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

from .. import agent_clay
from ..clay import document as bd
from ..clay import scratch as clay_scratch
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
