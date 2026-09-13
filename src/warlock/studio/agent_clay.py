"""The tool surface an MCP agent drives Clay through.

**The tool list is derived, never hand-written.** Every schema in :func:`tools`
is built from a registry that already exists for a human surface --
``primitives.GENERATORS`` for what a primitive is and what it defaults to,
``presets.ASSEMBLIES`` for which figures exist, ``clay_ops.OPS`` for the whole
of what an object or an element can be told to do, and ``select.QUERIES`` for
the handful of selection verbs that answer from a seed or a parameter rather
than acting on what is already selected. A thirteenth primitive, a ninth
figure, a new op or a seventh query in its registry needs no edit here: it
shows up in the next ``tools()`` call because the source it is drawn from
changed, which is the same property ``clay_ops.menu`` already gives the
context menu, the tools pane and the key handler -- one list, so nothing here
can drift out of step with what Clay can actually do. Writing a query enum out
by hand would have been a *fifth* place to remember one exists, beside those
same three surfaces and ``OPS`` itself. ``clay_batch``'s own name enum is
derived the same way, from ``_HANDLERS`` minus ``BATCH_EXCLUDED`` -- see that
constant for which tools are left out and why.

**An agent may take a mesh apart the way a person can, once it has a
selection to work from.** ``clay_element_mode`` switches vertex/edge/face
mode (and back to object mode); ``clay_select_elements`` and ``clay_select_by``
write what is selected inside one object, either by explicit index or by a
seed/parameter through :data:`select.QUERIES`; and every element-gated
``clay_op`` row -- ``inset``, ``bevel``, ``extrude`` and the rest -- reads
that selection exactly as the keyboard and the context menu do. Before this,
nothing in this module ever called ``ClayDoc.set_element_mode`` or
``set_element_sel``, so those rows refused unconditionally, forever, and an
agent could place and boolean shapes but never touch a single face. **The
derived-selection invariant is what makes ``clay_select``/``clay_boolean``
refuse in an element mode instead of silently breaking it**: ``document.py``'s
own module docstring says ``selection`` is *derived* from ``element_sel`` once
the document leaves object mode, and those two tools write object uids
straight into ``selection`` -- so refusing by name, and pointing at
``clay_element_mode``, is what keeps a still-truthy ``selection`` from ever
naming an object with nothing selected inside it. An element selection is
indices into one mesh, and the document is what keeps it from going stale --
dropped on an undo that changes geometry, restricted to what survives a
params rebuild, replaced outright by whatever an op produces -- and
``mesh_stamp`` is how an agent detects the one case that leaves both of those
untouched: the human at the keyboard editing the agent's own tab in between
two of its calls.

**An agent reaches exactly one document, and never by falling back to
whichever tab the user has open.** ``Session.tab_uid`` names the one
:class:`~.clay_state.ClayTab` this session owns; every tool resolves it fresh,
every call, through ``clay_mode.ensure(ctx).get(session.tab_uid)`` -- read
*through* the same ``ClayState`` the interactive UI uses rather than a
snapshot taken once, so a document closed from the keyboard mid-session is
seen as gone on the very next call. A missing tab is never a *substitution*:
an agent with no document of its own must never be handed the user's, because
that is the one way a scripted client could edit, export or close something
the person at the keyboard never offered it. The empty default
(``tab_uid == ""``) means "this session owns nothing yet," and only the three
tools that can start a document from nothing (:func:`clay_add_primitive`,
:func:`clay_add_figure`, :func:`clay_add_mesh`) are allowed to mint one and
adopt it into the session -- ``clay_batch`` is a documented fourth way in,
but only because its first call is one of those three; see its own
docstring.

**A closed document is a refusal for every tool that needs an existing one,
and a fresh start for the two that do not.** Those same two creators release
the dead pin and mint a new document rather than refusing, which is not a
softening of the rule above: minting is the opposite of substituting -- what
arrives is an empty document this session has just been given, never one that
was already open and belongs to somebody else, so the blast radius is
unchanged at exactly one tab. The refusal every *other* tool gives names
those two as the way out, and until this branch existed that sentence was
impossible to follow: ``clay_add_primitive`` arrives with ``create=True``,
but a still-truthy ``tab_uid`` sent it back out with the identical refusal,
so a session whose document the user closed was bricked for the rest of the
connection -- every tool refusing, and the one the refusal named refusing
the same way.

**Rendering owns a private viewport.** ``ClayView`` is a real GL object --
buffers, gizmos, a camera -- and the one already on screen
(``ctx.clay_view``) belongs to whatever tab the user is looking at. Reusing
it for ``clay_render`` would mean every render this module produces first
yanks the user's camera onto the agent's document and back, which is a visible
stutter in the middle of whatever the person is doing. This module builds and
keeps its own :class:`~.clay_view.ClayView`, off the moderngl context the app
already has (``ctx.viewer.ctx``) but with no ``app_ctx`` of its own, so it
never reads or writes ``ctx.clay_view`` at all. :func:`release` is the
matching teardown, called by ``agent_host`` at **teardown**, not when the
bridge disconnects -- a reconnect is routine, and releasing on disconnect
would either run GL work off the frame thread or race teardown's own
ordered stop-then-release; a viewport torn down and rebuilt per connection
would be churn bought for nothing.

**``clay_op`` is handed a sandboxed proxy, never the real app ``ctx``.**
Handing over the real one used to mean an op an agent ran through this
escape hatch could, like the same op fired from the keyboard, move the
viewport the user is looking through mid-gesture (Frame Selection chief
among them) -- the argument for doing so was that every op in the registry
is already written against the real shape (``ctx.toast``, ``ctx.clay_view``,
``ctx.state``), and reimplementing the registry against a second ``ctx``
shape it was never written for looked like the wrong trade. That argument
does not survive contact with a running agent: ``clay_ops`` reaches ``ctx``
in exactly three places (``toast``, ``getattr(ctx, "clay_view", None)`` in
``_frame``, ``getattr(ctx, "state", None)`` in ``_forget_manifold``), which
is few enough to sandbox properly rather than hand over wholesale. See
:class:`_OpCtx`: the absent ``clay_view`` makes Frame Selection the no-op it
should always have been for an agent with no viewport of its own, ``state``
passes through for real because the manifold-cache pop is real work that
still has to happen, and every ``toast`` lands in the result instead of the
running app -- so a per-object refusal inside ``run`` that used to become a
toast the user saw and the agent never did now comes back as a message the
agent can actually read.

**``clay_batch`` folds several tool calls into one undo step.** An agent
that wants to block out a scene one primitive at a time pays one round trip
per primitive and, worse, one Ctrl+Z per primitive for a user who wants to
back the whole attempt out; ``clay_batch`` runs up to ``BATCH_MAX`` calls
through :func:`call` under one ``history.mark()``/``collapse_since`` pair,
stopping at the first refusal and keeping the successful prefix. A batch
entry never sees an earlier entry's own result -- that only exists once the
whole batch has returned -- so it has no uid to hand a later entry an object
an earlier one just created; a persistent recipe object with named parts was
argued down in design review as more machinery than that ask needed in
favour of four cheap pieces already mostly built (object names, ``clay_scene``
reporting them, a plural ``clay_set_params``) plus this last one:
``{"$ref": "<name>"}``, found anywhere inside an entry's ``arguments``,
resolves to that name's uid the moment its entry runs (:func:`_resolve_batch_ref`).
Resolved per entry, not against the whole ``calls`` list up front, for the
same reason a bad ``uid`` already refuses at its own entry rather than in a
pre-flight pass: a name only exists once whatever created it has actually
run. And ``$ref`` is batch-only on purpose -- outside a batch an agent
already holds the creating call's own uid, so :func:`call` never learns
``$ref`` exists; a ``$ref`` handed to an ordinary call is refused as the
malformed ``uid`` it is.

**``rollback_on_error`` is an opt-in beside the stop-and-keep contract, not a
replacement for it.** Because the whole run already folds into one step,
backing a kept prefix out is already a single ``clay_undo`` -- but an agent
that would rather the partial work never have existed can pass
``rollback_on_error=True``, and when the batch stops at a refusal
:func:`_h_batch` reverses the folded step with ``history.undo(doc,
redoable=False)`` before it returns (see that method's own docstring in
``undo.py`` for the cancelled-lift incident ``redoable=False`` exists for --
``redoable=True`` here would leave the abandoned batch on the redo stack for
a later ``clay_redo`` to bring back, exactly the outcome the agent asked to
avoid). This is **not** a third exception to "one tool call is one undo
step": a rolled-back batch pushes no step at all, the same shape as a
refusal that never mutated the document to begin with. And it reverses only
the document's own undo stack -- a tab this same batch minted still exists,
because minting one pushes no undo step either (see :func:`_h_add_primitive`'s
own comment), and the two families that push nothing (references, the
selection tools) are exactly as untouched by a rollback as by an ordinary
``clay_undo``.

**The undo enumeration, in full.** Together with ``clay_undo``/``clay_redo``
(which move the history head rather than pushing one of their own),
``clay_batch`` is one of two exceptions that fold or move a step -- what
makes "one tool call is one undo step" true rather than approximately true.
Two families push none at all instead: references (``clay_reference_add``
and friends), because nothing in the document changes when a picture is
merely held on the session, and the selection tools (``clay_element_mode``,
``clay_select_elements``, ``clay_select_by``, ``clay_select``), because
selection is not undoable by design (``document.py``'s own module docstring)
-- an undoable selection would push a step, the step would move
``history.head``, and a document would ask to be saved again because
somebody looked at a different object. The two families differ from each
other in one way worth stating rather than blurring: a reference never
touches the ``ClayDoc`` at all, while a selection tool genuinely changes the
document and still pushes nothing.

**A call that outruns ``agent_host.CALL_TIMEOUT`` is dropped if the frame
thread has not started it, and finishes if it has.** The two outcomes tell
an agent different things and want different recoveries: a dropped call
changed nothing, so sending it again is safe. A started call is already
running and will finish on its own, and its answer is not lost along with
the refusal -- ``warlock_status`` says what became of it even while the
window is still busy, and sending the identical call again once it has
finished replays that answer rather than running the call a second time.
Only a call whose answer never reached the client is replayed that way: two
identical calls that both got answered are two calls, on purpose. See
:mod:`.agent_host`'s own module docstring for the compare-and-set that makes
the two outcomes mutually exclusive rather than a race, and for the intent
fingerprint that recognises the retry.

**Every tool answers the same JSON twice, on purpose.** :func:`_json` -- what
most of this module's tools return through -- puts *payload* in the result
as text (``json.dumps``, what an agent's model actually reads) and again as
``structuredContent`` (the same data, for a client that wants to branch on a
field instead of re-parsing prose out of the text block). Duplication, not
an oversight: a model and a client are two different readers of one answer,
and neither can stand in for the other.

**A refusal is machine-readable, not only readable.** :func:`fail` -- the thin
wrapper over ``protocol.fail`` a few lines below -- gives every refusal in
this file a ``changed`` key, defaulted in that one wrapper rather than at each
of its ~100 call sites: whether *this session's own document* -- the
``ClayDoc`` itself -- was modified before the refusal fired. Not whether a
Library row was minted (``clay_export``'s own refusal is ``changed: false``
by this definition even where it has already written one) and not whether a
session-scoped reference was added (``clay_reference_add`` touches no
``ClayDoc`` at all) -- the document, exactly as the rest of this file already
uses the word: ``_h_transform`` and ``_h_set_params`` answer ``changed`` on
the *success* side today, and a refusal now answers the same question the
same way, rather than leaving a client to infer it from prose. ``field`` is
untouched by any of this -- still ``service.errors``' own convention, shared
with the panes, which is why it is never folded into a differently-shaped
key. Where a refusal already names the objects or the op it is about in its
message, ``uids``/``op`` ride along too, so a client need not parse them back
out of the text block. ``recovery`` is a closed, bounded vocabulary
(:data:`RECOVERY`, below) naming what a client should try next --
``"fix_arguments"``, ``"read_scene"``, ``"switch_mode"``, ``"start_document"``,
``"retry"``, ``"wait"`` -- and a refusal whose recovery is genuinely unknown
(the blanket backstops in :func:`call` and in ``agent_host._call`` itself)
carries no ``recovery`` key at all: an absent key is a real, distinct answer,
never a seventh member invented to avoid omitting the field.

**A result that carries a picture does not duplicate its header into
``structuredContent``** -- the rule, stated once, rather than a list of tool
names it happens to apply to today. ``clay_render`` builds its result
directly with ``ok(header, *pngs)`` and ``clay_reference_get`` with
``ok(text(json.dumps(meta)), image_png(...))``, both bypassing :func:`_json`
for the same reason: an image block has no JSON to duplicate. It matters
most for ``clay_render``, whose header is already checked twice against
``protocol.MAX_FRAME`` before it leaves (see the render paragraph below) --
giving that header a second life in ``structuredContent`` would spend frame
budget on bytes nothing reads. :func:`.agent_host._carries_an_image` asks
this exact question -- does this result's ``content`` include an image
block -- to decide whether a remembered reply is worth replaying rather than
re-run; two decisions, in two modules, arriving at the same structural test,
neither one a hand-kept list of picture-shaped tools.

**An unknown argument is refused now, rather than silently dropped.**
Measured at HEAD before this paragraph was true: ``clay_add_primitive`` given
``{"generator": "box", "translaton": [0, 9, 0]}`` placed a box at the origin
and reported success, because nothing compared the arguments a call actually
carried against the schema :func:`tools` had just published for it -- an
agent that mistyped one argument spent its next several calls wondering why
the number it had set had no effect. :func:`call` -- the one door every tool
call passes through, ``clay_batch``'s own nested calls included, which is why
this check lives here rather than in ``agent_host`` or ``mcp/protocol.py`` --
now refuses before a handler ever runs if ``arguments`` names a key the
tool's own schema does not declare in ``properties``, naming every offending
key, suggesting what each was probably meant to be
(``difflib.get_close_matches`` against that tool's real names), and refusing
with ``field=`` set so :func:`fail` derives ``recovery="fix_arguments"`` the
same way every other named-field refusal already does. The allowed names are
derived from the schema :func:`tools` already builds -- once, memoised --
never a second hand-kept table beside it, which is exactly the drift class
this module's own derivation paragraphs above exist to rule out. Everything
about an argument's *value* -- a two-element vector, a NaN, an enum member
the registry does not have -- is still, as it always was, the handler's own
job to check before it mutates; only the *name* moved to this one door.
``tests/test_agent_schemas.py`` is what stands in for the validator
``mcp/protocol.py`` deliberately does not carry: it walks every constraint
every schema in this file actually declares and proves the handler enforces
it, because a validator bolted onto that leaf would have had no way to check
the ``anyOf``, ``minimum``, ``maximum`` and ``exclusiveMinimum`` shapes these
schemas really use without ``mcp/protocol.py`` learning what Clay is -- and
that leaf staying ignorant of Clay is a decision this file does not get to
revisit. A test that checks real behaviour is worth more than a validator
that checks only some of it.

Five tools -- ``clay_scene``, ``clay_add_primitive``, ``clay_add_mesh``,
``clay_diagnose`` and ``clay_analyze`` -- go one step further and declare an
``outputSchema`` describing that structured shape; the rest deliberately do
not, because a schema for a uid and a count is authorship with no reader.
``clay_add_mesh`` composes its schema from :func:`_object_row_output_schema`
rather than repeating it -- the same row ``clay_add_primitive`` declares,
plus the two keys only this tool answers with -- because a hand-copied
second row schema is exactly the drift the derivation paragraphs above rule
out for a query enum or a generator list, and a row's own shape is no
different. None of the five declares ``required``: a refusal shares this
same result envelope (``protocol.fail``'s own ``structuredContent`` is
whatever ``field`` it was given, nothing more), so a ``required`` list on
the success shape would make every refusal of these tools non-conforming
for a client validating strictly against its schema.

**``clay_render``'s payload is bounded before the GPU work, not after.**
``RENDER_PIXEL_BUDGET`` refuses a request for too many total pixels across
its views before a single frame is drawn, and the base64-encoded result is
checked again against ``protocol.MAX_FRAME`` (less ``RENDER_FRAME_RESERVE``)
before it is returned -- both refusals name the ceiling and suggest asking
for fewer or smaller views, because the alternative is minutes of rendering
spent on a reply nothing on the other end of the pipe could ever receive.

**References live on the session, never on the document.** ``clay_reference_add``
keeps a picture in memory on :class:`Session`, in memory only -- never in the
``ClayDoc`` and never in the ``.wblk``, because a :class:`~.clay_state.ClayTab`
outlives the session that opened it and putting pictures in it would drag in
journal and serialisation questions the format's VERSION 2 has no answer for.
Adding one pushes no undo step either -- one of the two families that push
none at all; see the ``clay_batch`` paragraph above for the whole
enumeration.

**Two intentional departures from calling ``clay_mode.save_to`` and
``clay_mode.export_asset`` by name**, both because those functions hand their
result to whichever tab is on screen when a task finishes, and an MCP
``tools/call`` has no such moment: it returns once, from this frame's
``call()``, and the job id the contract asks for has to be in hand by then.

* ``save_to`` submits its encode to a task thread and, on completion,
  retitles the tab and repoints ``tab.path`` at wherever it was told to
  write -- exactly right for a user's own Ctrl+S, and exactly wrong here: an
  agent-chosen path must never become what the user's *next* Ctrl+S silently
  overwrites. :func:`clay_export` never calls it at all, so ``tab.path``,
  ``tab.title`` and ``tab.saving`` are left exactly as they were. The
  authored document survives anyway: :func:`clay_mode.build_asset` already
  writes it as the job's own ``build.wblk`` sidecar (via ``save_clay_source``),
  which is what ``clay_mode.edit_asset_in_clay`` reopens from the Library --
  there is no second copy for this module to keep of its own.
* ``export_asset`` submits its own encode the same way and only knows the new
  job's id once that task finishes and calls back into ``clay_mode.
  on_task_done`` -- there is no id to return from this call if this module
  goes through it as written. :func:`clay_export` instead calls
  ``clay_mode.build_asset`` -- the same document -> model-row chain
  ``export_asset`` itself now calls from its own task thread -- directly and
  synchronously, so the id is in hand before ``call()`` returns.

Both run once, off the interactive 60 fps loop, for a deliberate one-shot
action rather than every frame -- the stall a synchronous encode would be if
it ran on every draw is not what is happening here.
"""

from __future__ import annotations

import difflib
import functools
import json
import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ..service import files as svc_files
from ..service import validation as svc_validation
from ..service.errors import NotFound, ServiceError
from . import clay_mode, clay_ops
from .clay import analyze as clay_analyze
from .clay import diagnose as clay_diagnose
from .clay import document as bd
from .clay import elements as el
from .clay import mesh as bm
from .clay import ops as clay_geom_ops
from .clay import ops_boolean, presets, regen, shading
from .clay import primitives as bp
from .clay import select as bsel
from .clay.adjacency import adjacency
from .clay.elements import OpError
from .clay_view import ClayView
from .panes import clay_tools as pane_clay_tools
from .viewer import gltf
from .viewer import math3d as m3
from .viewer.camera import Camera

log = logging.getLogger(__name__)


# --- module constants ---------------------------------------------------------

BATCH_MAX = 32
"""The most tool calls one ``clay_batch`` request may fold into a single undo
step. The byte budget (``UndoStack``) already bounds what a single step can
*cost*, but nothing bounded how many calls a batch could ask for before this
-- an unbounded batch would let one MCP round trip queue an arbitrarily long
run with no natural place to hit a ceiling first."""

MINTS_A_DOCUMENT: tuple[str, ...] = (
    "clay_add_primitive",
    "clay_add_figure",
    "clay_add_mesh",
)
"""The three tools that may start a document from nothing -- and so the three
a ``clay_batch`` may open with when the session owns no tab yet.

A constant rather than a tuple literal inside ``_h_batch`` because the list
was already written out in two places that then disagreed: the handler's own
membership check grew ``clay_add_mesh`` when that tool landed, and
``clay_batch``'s published description -- the sentence an agent reads before
it ever calls anything -- did not, so the catalogue told a model that a batch
starting with ``clay_add_mesh`` would be refused while the code was happily
running it. That is the hand-kept-copy-of-another-table drift this file
already refuses to write for its generator, op and query enums; the
description now interpolates this tuple instead of restating it, and
``tests/test_agent_clay.py`` pins it against the same ``_MINTS_A_TAB`` list
that keeps every tool classified.

The three prose refusals that also name these three (``_tab``'s two and
``_h_batch``'s own) are left as English rather than interpolated: "call
clay_add_primitive, clay_add_figure or clay_add_mesh" is a sentence, not a
list, and the test below is what catches one of them going stale.
"""

BATCH_EXCLUDED = frozenset(
    {
        "clay_batch",  # nesting buys nothing and bounds nothing
        # An image block is a result a client must see as one; a batch can
        # only hand back JSON text, so neither of these has a shape a batch
        # result could carry.
        "clay_render",
        "clay_reference_get",
        "clay_export",  # a deliberate one-shot that mints a Library row
        # A batch is itself one undo step; moving the history head from
        # inside the very run that is about to fold into one is incoherent.
        "clay_undo",
        "clay_redo",
    }
)
"""Tools ``clay_batch`` refuses to run -- see :func:`_h_batch`'s docstring
for the derivation, and the comments above for why each one is excluded."""

MAX_REFERENCES = 8
"""How many pictures one session may hold at once. A session's references
live in memory for the session's whole life (:attr:`Session.references`) and
nothing ever evicts one on its own, so a ceiling is what keeps a client that
forgets ``clay_reference_remove`` from growing an unbounded set of decoded
PNGs behind a session nobody is watching."""

RENDER_PIXEL_BUDGET = 6 * 1024 * 1024
"""The most total pixels one ``clay_render`` call may ask for, summed across
every view it requests. A caller asking for the maximum -- a dozen views at
2048x2048 -- would otherwise spend real GPU minutes producing a reply that
was refused for size the moment it tried to leave; this stops that before
the first frame is drawn rather than after."""

RENDER_FRAME_RESERVE = 64 * 1024
"""Headroom subtracted from ``protocol.MAX_FRAME`` when checking whether a
render's base64-encoded payload will fit in one reply frame. The JSON
envelope around the image blocks costs bytes of its own; without this an
encode that is over by a few hundred bytes would reach ``send_bytes`` and
fail there, past the point a refusal could explain itself."""

ELEMENT_PAGE_MAX = 4096
"""The most element indices one ``clay_elements`` call may hand back for one
kind, per object. This is **not** a wire limit the way ``RENDER_PIXEL_BUDGET``
and ``protocol.MAX_FRAME`` are -- a mesh's entire face list is real image-sized
data with nowhere smaller to fit, and a whole one would clear the frame budget
with room to spare. The bound here is the *reader*: four thousand bare
integers is on the order of twenty thousand tokens dropped into the context of
whatever is about to act on them, and an agent that genuinely needs that many
at once is reasoning about the document a different way than one call's reply
can serve -- ``offset``/``limit`` paging is the answer, not a bigger ceiling."""

ELEMENT_PAGE_DEFAULT = 256
"""``clay_elements``'s own default ``limit`` when none is given -- small
enough that the common case ("what did that extrude just make") comes back as
a paragraph rather than a printout, and still a small fraction of
``ELEMENT_PAGE_MAX`` for the rarer caller that has to page through more."""

MAX_MESH_VERTICES = 50_000
MAX_MESH_FACES = 50_000
"""The most vertices and faces one ``clay_add_mesh`` call may hand over,
checked -- and refused, naming the field -- before either array in ``mesh.
from_faces`` is built. Both stay well under ``glbimport.MAX_TRIANGLES``
(2,000,000 triangles), the document-wide ceiling ``serialize.py`` enforces on
import, because one call is one object among however many a document already
holds and this tool has no business spending most of that budget in one
shot. In practice this is the ceiling that actually fires: even a maximally
compact encoding of 50,000 vertices or faces (bare digits, no whitespace)
fits inside ``protocol.MAX_FRAME``'s 8 MiB request budget with room to
spare, so the wire frame limit is a backstop for a verbose encoding, not the
check doing the real work here."""

RECOVERY = frozenset(
    {
        "fix_arguments",
        "read_scene",
        "switch_mode",
        "start_document",
        "retry",
        "wait",
    }
)
"""The closed vocabulary a refusal's ``recovery`` key may name -- lives here,
not in ``mcp/protocol.py``, because several members (``"switch_mode"``) are
Clay vocabulary and that module must stay ignorant of Clay (its own module
docstring's rule). ``agent_host`` imports this module already, so it reaches
for the same six words for its own transport-level refusals rather than
inventing a second vocabulary next to this one. A refusal whose recovery is
not one of these six is a refusal with no ``recovery`` key at all -- see
:func:`fail`'s own docstring -- so this set is a ceiling, never a default to
fall back to.

* ``"fix_arguments"`` -- the request itself was malformed, out of range, or
  named something the registry this tool checks against has no entry for.
  Resending the identical call refuses the identical way; the arguments have
  to change first. Never written out at a call site: :func:`fail` derives it
  from ``field=``, because a refusal that names the argument it is unhappy
  with is already saying which one to change. That covers the shared
  validators (:func:`_validate_vec3`, :func:`_validate_number`,
  :func:`_validate_unit`, :func:`_validate_number_or_vec`,
  :func:`_validate_query_arg`) and equally the eighty-odd refusals that name
  a field directly -- a bad ``generator``, an unknown ``kind``, a ``name``
  another object already wears -- without a second keyword on any of them.
* ``"read_scene"`` -- the request named a uid or an object this document does
  not have right now, so the client's picture of it is stale. Attached in
  :func:`_resolve_uid`/:func:`_resolve_uids`. ``agent_host`` reuses it for its
  own started-call timeout refusal, where what is stale is not the document
  but the client's knowledge of whether the call it sent actually happened --
  re-reading (``clay_scene``, or ``warlock_status`` for that transport case)
  is the same recovery either way: go look before acting on a guess.
* ``"switch_mode"`` -- the document is in the wrong element mode for this
  call. ``clay_element_mode`` (or ``clay_select_elements``/``clay_select_by``'s
  own ``mode``) first, then repeat the call. Attached at the two sites using
  :data:`_OBJECT_SELECTION_DERIVED_REFUSAL`. **Not** attached to ``clay_op``'s
  own disabled-op refusal (:func:`_h_op`, via ``clay_ops.reason_for``) even
  though its wording sometimes names a mode gate -- ``op.reason`` covers
  several unrelated predicates (``_has_objects_reason``, ``_selection_reason``,
  ``_has_two_visible_reason`` among them, see ``clay_ops.py``'s own "reasons"
  section) and only some of them are about element mode; :func:`_h_op` has no
  way to tell which fired from the string alone, so it names the op
  (``op=``) instead of guessing a recovery that would be wrong for "Select an
  object first."
* ``"start_document"`` -- this session owns no document yet.
  ``clay_add_primitive``, ``clay_add_figure`` or ``clay_add_mesh`` starts
  one. Attached in :func:`_tab`'s own refusal.
* ``"retry"`` -- nothing ran; the identical call is safe to send again.
  ``agent_host``'s dropped-call timeout refusal.
* ``"wait"`` -- the same work is already running or queued; sending the same
  call again would run it twice. ``agent_host._replay``'s in-flight refusal.
"""

_OBJECT_SELECTION_DERIVED_REFUSAL = (
    "The object selection is derived from the element selection in "
    "vertex/edge/face mode. Call clay_element_mode with mode='object' first."
)
"""What ``clay_select`` and ``clay_boolean`` say in an element mode, verbatim
in both -- see each handler's own comment for why they refuse rather than
guess, and :func:`_h_delete`'s docstring for why it is deliberately not a
third."""


@dataclass
class Session:
    """One MCP connection's claim on Clay. See the module docstring."""

    tab_uid: str = ""
    references: dict[str, Any] = field(default_factory=dict)
    """Pictures this session has been handed to match against, by name --
    values are ``agent_refs.Reference``. In memory on the session only; see
    the module docstring's references paragraph for why never on the
    document."""

    last_render_png: bytes | None = field(default=None, repr=False)
    """The most recent PNG this session's ``clay_render`` produced -- read
    by ``agent_resources.read_dynamic`` for the ``warlock://clay/render/last``
    resource, ``None`` until the first render. Holds at most one picture,
    overwritten by the next render, never a history -- bounded the same way
    ``references`` is bounded by being session-scoped rather than kept
    forever."""


# The private viewport :func:`_view_for` builds -- module-level rather than
# per-session because a second concurrent session cannot exist yet (see
# ``mcp/pipe.py``'s v1 decision, one connection at a time), so there is
# nothing for a per-session instance to isolate that a shared one does not
# already give for free.
_view: ClayView | None = None


def _view_for(ctx: Any) -> ClayView:
    """This module's own viewport, built the first time a render is asked for.

    ``ctx.viewer`` is the app's main 3D pane and always exists once the app has
    a window, which is what makes it the one place to borrow a moderngl
    context from without reaching for the interactive Clay viewport itself.
    ``app_ctx=None`` is what keeps :class:`ClayView` from reading
    ``ctx.clay_view`` or the shared tool setting through ``self.state`` --
    see the module docstring's rendering claim.
    """
    global _view
    if _view is None:
        _view = ClayView(ctx.viewer.ctx)
    return _view


def release() -> None:
    """Free the GL objects this module opened. Called by ``agent_host`` at
    teardown -- a pipe that is never reopened this session should not hold a
    viewport's worth of buffers and gizmos alive for the rest of the app's
    life."""
    global _view
    if _view is not None:
        _view.release()
        _view = None


# --- resolving the session's document ---------------------------------------


def _tab(ctx: Any, session: Session, *, create: bool = False) -> tuple[Any, dict | None]:
    """The session's own tab, or a failure result to return unchanged.

    ``create`` is only ever passed by the three tools that can act on an
    empty session -- adding the first primitive, figure or hand-built mesh --
    so every other tool refuses outright rather than silently starting a
    document nobody asked for. Resolved through ``ClayState`` on every call,
    never cached on the session, so a tab the user closed from the keyboard
    is seen as gone on the very next tool call rather than on whichever call
    happens to notice.
    """
    state = clay_mode.ensure(ctx)
    if session.tab_uid:
        tab = state.get(session.tab_uid)
        if tab is not None:
            return tab, None
        if not create:
            return None, fail(
                "This session's document was closed. Call clay_add_primitive, "
                "clay_add_figure or clay_add_mesh to start a new one.",
                recovery="start_document",
            )
        # The pin is released here rather than left standing, because leaving
        # it made the refusal above impossible to follow. It named
        # ``clay_add_primitive`` as the way out, but that tool is exactly the
        # one that arrives with ``create=True`` -- and a truthy ``tab_uid``
        # sent it straight back into this branch and out with the same
        # sentence, forever. A session whose document the user closed was
        # therefore bricked for the rest of the connection: every tool
        # refused, and the one the refusal told it to call refused
        # identically. Clearing the pin first is what makes the mint below
        # reachable, and it keeps the blast-radius rule intact rather than
        # widening it -- the session still owns exactly one tab and still
        # cannot name anybody else's, it is simply allowed to be handed a new
        # one after the old one is provably gone.
        session.tab_uid = ""
    if not create:
        return None, fail(
            "This session has no document yet. Call clay_add_primitive, "
            "clay_add_figure or clay_add_mesh first.",
            recovery="start_document",
        )
    tab = clay_mode.new_document(ctx)
    session.tab_uid = tab.uid
    return tab, None


# --- protocol glue ------------------------------------------------------------
#
# Imported lazily inside functions rather than at module scope. The original
# reason was that ``warlock.mcp`` was still being written alongside this
# module and a module-scope import would have failed at collection time on
# however far that sibling had got; both modules exist now, so that reason is
# spent and is not what keeps this here. What keeps it is the direction of the
# dependency: ``studio.agent_clay`` is imported by panes and by their tests
# -- ``panes.clay_tools`` among them -- none of which want the protocol leaf
# loaded to ask this module a question about Clay, and ``mcp/`` is a leaf that
# must never learn about ``studio`` in return (``tests/mcp/test_mcp_imports.py``
# pins that). One accessor, below, is the whole of the coupling.


def _protocol() -> Any:
    """Named `_protocol` for history, not for where the names now live:
    `Tool`/`ok`/`fail`/`text`/`image_png`/`MAX_FRAME` are `mcp/rpc.py`'s own
    vocabulary (`mcp/protocol.py` only re-exports them for the bridge's MCP
    path), and importing `rpc` directly here -- rather than `protocol` --
    is what keeps `warlock.studio` from ever importing `warlock.mcp.protocol`
    (`tests/mcp/test_mcp_imports.py` pins that)."""
    from ..mcp import rpc

    return rpc


def ok(*content: dict, structured: dict | None = None) -> dict:
    return _protocol().ok(*content, structured=structured)


def fail(message: str, *, changed: bool = False, **extra: Any) -> dict:
    """Thin wrapper over ``protocol.fail`` -- see that function's own
    docstring for the wire shape ``extra`` (``field=``, now also ``changed=``,
    ``recovery=``, ``uids=``, ``op=``) lands in.

    ``changed`` says whether *this session's own document* -- the ``ClayDoc``
    itself, walked by uid through ``tab.doc`` -- was modified before this
    refusal fired. Not whether a Library row was minted (``clay_export``
    refusing after ``clay_mode.build_asset`` has already run would still be
    ``changed: false``, because a model row is not this document) and not
    whether a session-scoped reference was added or removed (``clay_reference_add``
    never touches a ``ClayDoc`` at all) -- the document, exactly as
    :func:`_h_transform` and :func:`_h_set_params` already use the word on
    the success side.

    Defaulted here, once, rather than passed at each of this file's ~100
    ``fail(...)`` call sites, so every refusal answers the question by
    construction and a handler that refuses *after* it has already mutated
    the document is the only kind that has to say so explicitly. Exactly one
    does: ``clay_batch``, whose documented contract is that it stops at the
    first refusal and *keeps what already ran*, so it computes the answer
    from its own history mark rather than defaulting. Every other refusal in
    this file validates before it mutates -- the rule
    ``docs/manual/46-extending.md`` states for a new tool -- and
    ``tests/test_agent_clay.py`` proves it against the document itself, by
    walking every handler and checking a ``changed: false`` refusal really
    did leave the history, the dirty flag and the object count alone.

    ``recovery`` is defaulted the same way, and from the field itself: a
    refusal that names a ``field`` is by definition telling the client which
    argument was wrong, which is ``"fix_arguments"`` -- so naming one is
    enough and the 80-odd refusals that already do get their recovery
    without a second keyword each. An explicit ``recovery=`` always wins,
    which is what the exceptions rely on: ``_resolve_uid``'s missing uid is
    ``"read_scene"`` even though it names ``field="uid"``, because the
    argument may be perfectly well-formed and the document simply no longer
    holds it, and the stale-``expect_stamp`` refusal is ``"read_scene"`` for
    the same reason -- its own sentence already says to go and read the
    stamp again. A refusal that names no field and passes no recovery keeps
    none, which is the honest answer for the blanket ``except`` in
    :func:`call`: nothing there knows what a client should do differently.
    """
    if "recovery" not in extra and extra.get("field"):
        extra["recovery"] = "fix_arguments"
    return _protocol().fail(message, changed=changed, **extra)


def text(s: str) -> dict:
    return _protocol().text(s)


def image_png(data: bytes) -> dict:
    return _protocol().image_png(data)


def _json(payload: Any) -> dict:
    """The shape every tool whose reply carries no picture answers in:
    *payload* as text (``json.dumps``, what a model actually reads) and,
    duplicated, as ``structuredContent`` (what a client branches on instead
    of re-parsing that text) -- see the module docstring's structured-results
    paragraph for the reasoning in full, and for the rule (a result that
    carries a picture does not duplicate its header) that excludes the two
    tools which do. The duplication is deliberate, not an oversight to
    dedupe away later.

    ``structured=payload`` only when *payload* is a ``dict`` -- MCP requires
    an object there, never a list or a scalar. Every one of this file's own
    call sites already passes a dict, so this guard is a floor for whatever
    calls ``_json`` next, not a case any of them actually hits today.
    Routed through the already-serialized text (``json.loads`` of the same
    ``json.dumps`` the text block uses) rather than *payload* itself, so a
    tuple or a numpy scalar buried in ``params`` reaches ``structuredContent``
    as the plain list or number the wire format would have turned it into
    anyway -- the two blocks are meant to be the same JSON, not merely
    ``==``-comparable Python objects that happen to serialize the same way.

    ``clay_render`` and ``clay_reference_get`` never call this -- both answer
    with an image block, which is not JSON to duplicate, so each builds its
    own result directly with ``ok(...)`` instead. It matters most for
    ``clay_render``, whose header is already checked twice against
    ``protocol.MAX_FRAME`` (``RENDER_PIXEL_BUDGET``, ``RENDER_FRAME_RESERVE``)
    before it is returned -- giving that header a second life in
    ``structuredContent`` would spend frame budget on bytes with no reader.
    See :func:`_h_render`'s and :func:`_h_reference_get`'s own returns for
    where that exclusion is made.
    """
    encoded = json.dumps(payload)
    structured = json.loads(encoded) if isinstance(payload, dict) else None
    return ok(text(encoded), structured=structured)


# --- Euler XYZ, for clay_transform and clay_scene ----------------------------


def _quat_from_euler_xyz(degrees: Any) -> Any:
    """Three degrees -- rotate-X, then Y, then Z -- as this document's XYZW quaternion.

    ``viewer.math3d`` carries ``quat_from_axis_angle`` and ``quat_mul`` but no
    Euler helper at all, and that is not an oversight to fix upstream: nothing
    else in Clay needs one. A gizmo drag accumulates a single axis-angle
    increment directly into the object's quaternion and never decomposes it
    back into three numbers, so there has never been a second caller to share
    this with. An agent describing an orientation has no such luxury -- "face
    this way" arrives as three degrees -- so the composition lives here, once,
    for the one tool that takes them.

    Intrinsic X, then Y, then Z, which is the order a person reaching for
    "rotation" with no further qualification expects (it is Blender's default
    Euler order). ``quat_mul(a, b)`` applies ``b`` first, so building the
    result as ``qz * qy * qx`` puts X innermost -- applied first -- exactly
    matching that order.
    """
    rx, ry, rz = (math.radians(float(v)) for v in degrees)
    qx = m3.quat_from_axis_angle(m3.vec3(1.0, 0.0, 0.0), rx)
    qy = m3.quat_from_axis_angle(m3.vec3(0.0, 1.0, 0.0), ry)
    qz = m3.quat_from_axis_angle(m3.vec3(0.0, 0.0, 1.0), rz)
    return m3.quat_mul(m3.quat_mul(qz, qy), qx)


def _euler_xyz_from_quat(q: Any) -> tuple[float, float, float]:
    """The exact inverse of :func:`_quat_from_euler_xyz`, in degrees.

    A gizmo drag accumulates axis-angle increments straight into an object's
    quaternion and never decomposes them back -- so until an agent needed to
    *read back* what it placed, nothing in Clay ever needed this inverse.
    ``clay_scene`` hands an agent three degrees rather than four quaternion
    components precisely so the readout is something ``clay_transform`` can
    be fed straight back into; a scene description an agent cannot act on is
    not a description worth giving it.

    Derived from ``m3.quat_to_mat4``, which is column-vector convention, so
    with ``R = Rz.Ry.Rx`` (the same composition order ``_quat_from_euler_xyz``
    builds): ``ry = asin(clamp(-R[2, 0], -1, 1))``; away from gimbal lock,
    ``rx = atan2(R[2, 1], R[2, 2])`` and ``rz = atan2(R[1, 0], R[0, 0])``; at
    gimbal lock (``|cos(ry)|`` tiny) ``rz`` is pinned to 0 and ``rx`` is read
    off row 0 instead -- ``atan2(R[0, 1], R[0, 2])`` at ``ry`` ~= +90 deg,
    ``atan2(-R[0, 1], -R[0, 2])`` at ``ry`` ~= -90 deg. The round-trip claim
    this exists to satisfy is about the *rotation* the three angles describe,
    not the three numbers themselves -- at gimbal lock a whole family of
    ``(rx, rz)`` pairs describes the same orientation, and picking ``rz = 0``
    is simply one member of it.
    """
    r = m3.quat_to_mat4(q)[:3, :3]
    sin_ry = -float(r[2, 0])
    ry = math.asin(max(-1.0, min(1.0, sin_ry)))
    if abs(math.cos(ry)) > 1e-6:
        rx = math.atan2(r[2, 1], r[2, 2])
        rz = math.atan2(r[1, 0], r[0, 0])
    else:
        rz = 0.0
        rx = (
            math.atan2(r[0, 1], r[0, 2])
            if sin_ry > 0
            else math.atan2(-r[0, 1], -r[0, 2])
        )
    return (math.degrees(rx), math.degrees(ry), math.degrees(rz))


# --- shared validation and mutation helpers -----------------------------------


def _resolve_uid(doc: Any, args: dict, key: str = "uid") -> tuple[Any, dict | None]:
    """*doc*'s object named by ``args[key]``, or a refusal naming ``field=key``.

    The ``int(args[key])`` / ``doc.by_uid`` / refusal dance that
    ``clay_transform``, ``clay_set_params`` and ``clay_diagnose`` each spelled
    out separately -- three copies of one lookup, free to drift on the wording
    or the field name the moment one of them was edited and the others were
    not.
    """
    try:
        uid = int(args[key])
        obj = doc.by_uid(uid)
    except (KeyError, ValueError, TypeError):
        return None, fail(
            f"no object with uid {args.get(key)!r}.", field=key, recovery="read_scene"
        )
    return obj, None


def _resolve_uids(
    doc: Any, values: Any, field: str = "uids"
) -> tuple[list[int] | None, dict | None]:
    """*values* cast to ints, every one of them present in *doc*, or a refusal.

    The list version of :func:`_resolve_uid`, shared by ``clay_material`` and
    ``clay_select`` -- each of which cast to int, refused on a bad type, and
    refused again on an unknown uid, by hand. Deliberately silent about
    emptiness: ``clay_select`` means "clear the selection" by an empty list,
    while ``clay_material`` and ``clay_delete`` refuse one themselves, because
    only they have an opinion about it.
    """
    try:
        uids = [int(u) for u in values or []]
    except (TypeError, ValueError):
        return None, fail(
            f"{field} must be a list of integers.", field=field, recovery="fix_arguments"
        )
    known = {obj.uid for obj in doc.objects}
    missing = [u for u in uids if u not in known]
    if missing:
        return None, fail(
            f"no object with uid(s) {missing}.",
            field=field,
            recovery="read_scene",
            uids=missing,
        )
    return uids, None


def _validate_vec3(value: Any, field: str) -> tuple[list[float] | None, dict | None]:
    """Three finite numbers, or a refusal naming *field*.

    Shared by every optional TRS vector ``clay_add_primitive`` and
    ``clay_add_figure`` take, so a malformed one is caught before anything is
    placed -- see those tools' "validate everything before the first
    mutation" rule.
    """
    if not isinstance(value, list) or len(value) != 3:
        return None, fail(
            f"{field} must be an array of 3 numbers.", field=field, recovery="fix_arguments"
        )
    try:
        out = [float(v) for v in value]
    except (TypeError, ValueError):
        return None, fail(
            f"{field} must be an array of 3 numbers.", field=field, recovery="fix_arguments"
        )
    if not all(math.isfinite(v) for v in out):
        return None, fail(
            f"{field} must be finite numbers.", field=field, recovery="fix_arguments"
        )
    return out, None


def _validate_number(value: Any, field: str) -> tuple[float | None, dict | None]:
    """One finite number, unbounded -- the plain scalar case
    :func:`_validate_unit` (0..1) and :func:`_validate_number_or_vec` (number
    *or* array) both specialise. Added for ``clay_select_by``'s ``max_angle``,
    which is neither: a query argument this module owns the schema for (see
    ``_QUERY_ARG_SCHEMAS``), not a colour component or a params value.
    """
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None, fail(f"{field} must be a number.", field=field, recovery="fix_arguments")
    if not math.isfinite(out):
        return None, fail(f"{field} must be finite.", field=field, recovery="fix_arguments")
    return out, None


def _validate_unit(value: Any, field: str) -> tuple[float | None, dict | None]:
    """One finite number in 0..1, or a refusal naming *field*.

    Added beside :func:`_validate_vec3` for the same reason: ``clay_material``
    used to check a colour component with a bare ``isinstance(c, int | float)``,
    which ``float("nan")`` passes as readily as a real number is a float, and
    checked ``metallic``/``roughness`` with nothing at all
    (``float(args.get("metallic", 0.0))``) -- so a NaN in any of the three
    landed straight in the palette and rode along into every export from
    then on.
    """
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None, fail(
            f"{field} must be a number, 0..1.", field=field, recovery="fix_arguments"
        )
    if not math.isfinite(out) or not (0.0 <= out <= 1.0):
        return None, fail(
            f"{field} must be a number, 0..1.", field=field, recovery="fix_arguments"
        )
    return out, None


def _validate_range(
    value: Any, field: str, lo: float, hi: float
) -> tuple[float | None, dict | None]:
    """One finite number in ``lo..hi``, or a refusal naming *field*.

    :func:`_validate_unit` fixed at 0..1 for a colour component; this is the
    same check with the bound as an argument, for ``clay_analyze``'s three
    tolerances, each declared with its own ``minimum``/``maximum`` in the
    schema and none of them 0..1.
    """
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None, fail(
            f"{field} must be a number, {lo}..{hi}.", field=field, recovery="fix_arguments"
        )
    if not math.isfinite(out) or not (lo <= out <= hi):
        return None, fail(
            f"{field} must be a number, {lo}..{hi}.", field=field, recovery="fix_arguments"
        )
    return out, None


def _validate_number_or_vec(
    value: Any, field: str
) -> tuple[float | list[float] | list[list[float]] | None, dict | None]:
    """A number, an array of numbers, or an array of arrays of numbers, every
    one of them finite -- the ``number | array-of-numbers | array-of-arrays``
    shape ``clay_set_params``'s own schema declares for a param value (a
    cylinder's ``radius`` is one number, a box's ``size`` is three, a
    lathe's ``profile`` is an array of ``[radius, y]`` pairs) -- or a refusal
    naming *field*.

    A schema declaring a shape does not enforce it on the wire:
    ``mcp/protocol.py``'s ``tools/call`` handling checks only that
    ``arguments`` as a whole is a dict before handing it to the handler, so a
    NaN or an infinity reaches here exactly as an agent typed it. Added
    alongside :func:`_validate_unit` when an unvalidated ``clay_transform``
    committed a two-element translation that bricked ``clay_scene`` for the
    whole document (see ``document.set_transform``'s own backstop) --
    ``clay_set_params`` had the identical hole: a non-finite value in
    ``size`` sailed past ``bp.clamp_params`` (which only clamps the keys it
    knows a floor for) and baked straight into the generator's vertex
    positions.

    The array-of-arrays branch was added for ``lathe``'s ``profile``, the
    first generator parameter whose own elements are arrays rather than
    numbers: before it, this function's flat-array branch tried
    ``float(v)`` on each *row* of a profile and raised ``TypeError``, which
    came back as "params must be a number or an array of numbers" -- true of
    the old schema and wrong about the new one, since an array of arrays is
    exactly what a profile is and exactly what :func:`_params_value_schema`
    now declares. Every row must itself be a non-empty array of finite
    numbers, and the outer array must not be empty either -- the same two
    rules the flat case already holds a bare array to, one level up.
    """
    if isinstance(value, list) and value and all(isinstance(row, list) for row in value):
        try:
            rows = [[float(v) for v in row] for row in value]
        except (TypeError, ValueError):
            return None, fail(
                f"{field} must be a number, an array of numbers, or an "
                "array of arrays of numbers.",
                field=field,
                recovery="fix_arguments",
            )
        if not all(row and all(math.isfinite(v) for v in row) for row in rows):
            return None, fail(
                f"{field} must be finite numbers, with no empty row.",
                field=field,
                recovery="fix_arguments",
            )
        return rows, None
    if isinstance(value, list):
        try:
            out = [float(v) for v in value]
        except (TypeError, ValueError):
            return None, fail(
                f"{field} must be a number, an array of numbers, or an "
                "array of arrays of numbers.",
                field=field,
                recovery="fix_arguments",
            )
        if not out or not all(math.isfinite(v) for v in out):
            return None, fail(
                f"{field} must be finite numbers.", field=field, recovery="fix_arguments"
            )
        return out, None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None, fail(
            f"{field} must be a number, an array of numbers, or an array "
            "of arrays of numbers.",
            field=field,
            recovery="fix_arguments",
        )
    if not math.isfinite(out):
        return None, fail(f"{field} must be finite.", field=field, recovery="fix_arguments")
    return out, None


def _validate_params_values(params: dict, field: str) -> dict | None:
    """Every value of a generator's ``params`` dict through
    :func:`_validate_number_or_vec`, naming *every* offending key in one
    refusal rather than only the first -- the shared body behind the
    identical loops ``_h_add_primitive`` and ``_h_set_params`` used to run,
    each passing the literal string ``"params"`` in as *field* for every
    value, so a lathe's bad ``profile`` beside a good ``segments`` came back
    as "params must be finite numbers." with nothing to say which of the
    two was wrong -- the same class of defect ``_unknown_argument_refusal``
    closed for a misspelled argument name.

    ``field`` on the returned refusal is always exactly *field* itself
    (``"params"`` for both callers): ``tests/test_agent_schemas.py``'s
    ``_run_case`` walks only a refusal's top-level property, never a
    sub-field, so widening it to ``"params.profile"`` would break that
    walk. Only the *message* may name a key, which is why each value is
    checked under a per-key display name (``"params.profile"``) that never
    leaves this function -- :func:`_validate_number_or_vec` builds its
    message from whatever field string it is handed, so handing it a
    compound one is enough to get the key into the text without teaching it
    anything about ``params`` itself. Every failing key's message is kept,
    sorted the same deterministic way :func:`_unknown_argument_refusal`
    sorts its unknown names, so a caller that got two params wrong learns
    about both without a second round trip.

    Returns ``None`` when every value already validates.
    """
    messages = []
    for key in sorted(params):
        _, failure = _validate_number_or_vec(params[key], f"{field}.{key}")
        if failure:
            messages.append(failure["content"][0]["text"])
    if not messages:
        return None
    return fail(" ".join(messages), field=field)


def _params_shape_refusal(params: dict, defaults: dict, field: str, subject: str) -> dict | None:
    """Every value of *params* held to the **shape of the generator's own
    default** for that key, or a refusal naming every key that disagrees.

    :func:`_validate_params_values` checks a value is made of finite numbers
    and stops there, because that is all the wire schema declares: a param
    value is ``number | array-of-numbers | array-of-arrays``, one shape for
    every key of every generator. Which of the three a *particular* key
    wants is not in that schema, and nothing downstream asked either -- so
    ``clay_add_primitive("pyramid", params={"base": [1, 1, 1]})`` walked
    straight into ``primitives.pyramid``'s ``float(base)`` and came back as
    "failed unexpectedly; see the log" with a ``TypeError`` traceback in it,
    the generic backstop catching what should have been a field-named
    refusal (found by the furniture author, 2026-09-12). The same hole ran
    the other way: ``box`` with ``size=1.0`` raised ``TypeError`` on the
    unpack, and ``size=[1, 1]`` a ``ValueError`` about three values, both
    with the same unhelpful face.

    The rule is **derived from ``GENERATORS``' defaults, never listed**, for
    the reason :func:`tools` is: every default dictionary is a complete call
    (``primitives.GENERATORS``' own docstring), so the default *is* the
    shape, and a sixteenth generator enrols itself. A scalar default wants a
    scalar; a flat sequence default wants a flat array of exactly that many
    numbers; a sequence-of-rows default (``lathe``'s ``profile``,
    ``tube``'s ``path``, ``sweep``'s ``outline``) wants an array of rows of
    exactly that row's width. The outer length of a row array is free --
    that is how many points the profile has, which is the caller's to
    choose -- but the row width is not, and a three-number row handed to
    ``lathe`` silently dropped its third column rather than saying so.

    *subject* is what the message calls the generator (``'pyramid'`` for
    ``clay_add_primitive``, ``"'pyramid' (uid 4)"`` for ``clay_set_params``,
    which addresses many objects and must say which one). ``field`` on the
    refusal stays exactly *field*, for the reason
    :func:`_validate_params_values` gives: only the message may name a key.
    """
    messages = []
    for key in sorted(params):
        if key not in defaults:
            continue
        want, value = defaults[key], params[key]
        if not isinstance(want, list | tuple):
            if isinstance(value, list):
                messages.append(f"{field}.{key} must be a single number for {subject}.")
            continue
        rows = [r for r in want if isinstance(r, list | tuple)]
        if rows:
            width = len(rows[0])
            ok = (
                isinstance(value, list)
                and bool(value)
                and all(isinstance(row, list) and len(row) == width for row in value)
            )
            if not ok:
                messages.append(
                    f"{field}.{key} must be a non-empty array of "
                    f"{width}-number arrays for {subject}."
                )
        elif not (isinstance(value, list) and len(value) == len(want)):
            messages.append(
                f"{field}.{key} must be an array of {len(want)} numbers for {subject}."
            )
    if not messages:
        return None
    return fail(" ".join(messages), field=field, recovery="fix_arguments")


def _repaint(doc: Any, uids: Iterable[int], index: int) -> None:
    """Rewrite every face of each object in *uids* to material *index*.

    The trap: ``Obj.material`` is only the default slot *new* faces are
    stamped with. What actually renders and exports is the per-face
    ``mesh.material`` array -- ``to_primitives`` groups a mesh's faces by
    ``np.unique(mesh.material)`` and looks each index up in the palette.
    Pointing only ``obj.material`` at the new slot (a lone ``set_props``)
    would leave an existing box's faces still naming their old slot, so it
    would export in the wrong colour with nothing here to say why. Repainting
    means rewriting that array and rebuilding the mesh.
    """
    for uid in uids:
        obj = doc.by_uid(uid)
        mesh = replace(obj.mesh, material=np.full(len(obj.mesh.material), index, dtype="i4"))
        doc.set_mesh(uid, mesh, keep_generator=True)
        doc.set_props(uid, material=index)


def _label_top(doc: Any, mark: int, label: str) -> None:
    """Name the step this call just pushed -- but only when it actually pushed
    one. ``doc.history.head != mark`` is the guard: relabelling
    ``doc.history.top`` when nothing was pushed since ``mark`` would rename
    whatever step was already on top -- the user's *previous* action, not
    this call's."""
    if doc.history.head == mark:
        return
    top = doc.history.top
    if top is not None:
        top.label = label


def _round(value: Any, dp: int = 4) -> Any:
    """*value* rounded to *dp* decimal places -- a float, or a list/array of
    them. ``clay_scene``'s readout is meant to be read, and a world-space
    translation computed through several matrix multiplies comes back with
    sixteen digits of float noise that carries no information an agent could
    act on."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_round(v, dp) for v in value]
    if isinstance(value, (int, float, np.floating, np.integer)):
        return round(float(value), dp)
    return value


def _sel_counts(sel: el.ElementSel) -> dict:
    """One ``ElementSel``'s size, per kind -- the shape every element-mode
    result in this module reports instead of the indices themselves (see
    ``clay_elements`` for the one tool that hands those back, paged)."""
    return {"verts": len(sel.verts), "edges": len(sel.edges), "faces": len(sel.faces)}


def _scene_row(doc: Any, obj: Any) -> dict:
    """Everything ``clay_scene`` says about one object -- and everything
    ``clay_add_primitive``/``clay_add_figure`` hand back too, so an agent that
    just placed something never needs a second call to learn where it landed.
    """
    box = clay_geom_ops.world_box(obj)
    rx, ry, rz = _euler_xyz_from_quat(obj.rotation)
    size = center = None
    if box is not None:
        lo, hi = box
        size = _round(hi - lo)
        center = _round((lo + hi) * 0.5)
    return {
        "uid": obj.uid,
        "name": obj.name,
        "visible": obj.visible,
        "generator": obj.generator,
        "params": obj.params,
        "faces": bm.face_count(obj.mesh),
        "material": obj.material,
        "bbox": None if box is None else {"min": box[0].tolist(), "max": box[1].tolist()},
        "translation": _round(obj.translation),
        "rotation": _round([rx, ry, rz]),
        "scale": _round(obj.scale),
        "size": size,
        "center": center,
        "verts": len(obj.mesh.positions),
        # Purely additive -- see the module's element-mode paragraph. An
        # agent that has just switched mode or selected something does not
        # need a second call to learn what came across on this object.
        "stamp": doc.mesh_stamp(obj.uid),
        "selected": _sel_counts(doc.element_sel_of(obj.uid)),
    }


# --- the tools -----------------------------------------------------------------


def instructions() -> str:
    """The prose ``agent_host`` puts in the RPC v1 ``catalogue`` reply.

    Read once, by whatever model is driving the bridge, before its first tool
    call -- so this is where the conventions no single schema field can carry
    live: which units, which axis is up, that a rotation is always degrees and
    never a quaternion, that undo is call-scoped with two named exceptions
    (plus two families that push none at all), the working loop an agent that
    skips straight to numbers tends to get wrong, element mode, selection
    staleness and what a timeout does and does not mean. A function rather
    than a module constant, so ``BATCH_MAX``, the live generator catalogue and
    ``agent_host.CALL_TIMEOUT`` are embedded fresh rather than duplicated --
    the same reason ``tools()`` itself is rebuilt every time it is asked for.

    ``agent_host`` is imported lazily, inside the function body, rather than
    at module scope: it imports this module at *its* module scope (to reach
    :func:`call`), so a module-scope import back would be a cycle.
    :func:`_protocol` is this file's own established precedent for the same
    move, one section up.
    """
    from . import agent_host

    return (
        "Warlock's Clay, over MCP. Units are metres; the axes are glTF's -- "
        "Y is up, Z is toward the viewer, and the ground is y=0. Every "
        "generator is centred on its own origin, so a box of height h "
        "stands on the ground at translation=[0, h/2, 0]; a figure placed "
        "with clay_add_figure arrives already grounded. Rotations are "
        "always Euler XYZ in degrees, never a quaternion.\n\n"
        "uids are the only addresses -- clay_scene reports one for every "
        "object, and every other tool that names an object takes one. "
        "Names are for humans and may be renamed (clay_rename); a uid never "
        "changes.\n\n"
        "One tool call is one undo step, with two exceptions that fold or "
        "move steps -- clay_batch folds its whole run into one, and "
        "clay_undo/clay_redo move the history head rather than pushing one "
        "of their own -- and two families that push none at all: adding a "
        "reference (clay_reference_add) touches nothing in the document, and "
        "the selection tools (clay_element_mode, clay_select_elements, "
        "clay_select_by, clay_select) change the document without pushing a "
        "step, because selection is not undoable by design.\n\n"
        "Element mode is document state, not a per-call flag. clay_op's "
        "inset/bevel/extrude and the rest of the element-only rows refuse by "
        "name (\"Switch to face mode first.\") until it is set; "
        "clay_element_mode sets it on its own, and clay_select_elements and "
        "clay_select_by can set it in the same call as the selection they "
        "make. Entering vertex/edge/face mode from object mode selects "
        "nothing, which is what makes clay_op's seedless rows -- select-all, "
        "select-none, select-invert, select-linked, select-more, "
        "select-less, select-boundary -- reachable with no seed at all; "
        "clay_select_by is the door for the rest of clay_op's element menu, "
        "the ones that need one (a loop, a face, a material slot, a "
        "direction, a box). clay_select, clay_boolean and every tool that "
        "addresses a whole object want object mode, and refuse by name "
        "rather than switch back for you if the document is not in it.\n\n"
        "A selection is indices into one mesh, and indices go stale the "
        "moment the mesh they describe is replaced: an undo drops it, a "
        "params rebuild that changes the face count restricts it to what "
        "still exists, and an op replaces it outright with whatever the op "
        "produced -- extrude hands back its own new caps, so inset is "
        "usually the very next call with nothing to re-select. Every "
        "element-selection result carries a stamp; pass one back as "
        "expect_stamp on clay_select_elements or clay_select_by if anything "
        "might have touched the mesh since it was read, and a stale one is "
        "refused rather than acted on.\n\n"
        "The working loop that avoids building something plausible in "
        "numbers and wrong on screen: block out with primitives, "
        "clay_render from three_quarter and front, adjust, boolean, "
        "clay_diagnose, then clay_export. A boolean needs closed solids, "
        "and its survivor is whichever object comes first in the "
        "document's own order, never first in the uids list handed to "
        "it.\n\n"
        "clay_diagnose and clay_analyze both read without selecting anything "
        "you did not ask them to: diagnose finds what is wrong with a mesh "
        "(a hole, a non-manifold edge) and can select the offending elements; "
        "analyze measures facts about one or more objects that are not "
        "defects -- exact bounds, area, volume, ground contact, symmetry, and "
        "for a pair, distance, contact and overlap -- and never selects "
        "anything. Reach for analyze to check placement (is this resting on "
        "the ground, do these two touch or overlap, by how much) and "
        "diagnose to check mesh health before a boolean.\n\n"
        "Materials are linear RGB, 0..1. clay_scene's 'materials' lists the "
        "palette already in use -- reuse an index from it rather than "
        "appending a near-duplicate.\n\n"
        "References: clay_reference_add takes a Library job id or inline "
        "base64; pass 'compare' to clay_render to see the current render "
        "beside the reference or blended over it.\n\n"
        f"Up to {BATCH_MAX} tool calls can be folded into one clay_batch "
        "call; it stops at the first refusal and keeps everything that "
        "already ran. Inside a batch entry's arguments, "
        "{\"$ref\": \"<name>\"} resolves to the uid of the object holding "
        "that name at the moment that entry runs -- give an earlier entry a "
        "name (clay_add_primitive/clay_add_figure/clay_add_mesh's own "
        "argument) and a later entry in the same batch can address it "
        "without a clay_scene read in between; $ref only works inside "
        "clay_batch. Pass rollback_on_error=true to undo that folded step "
        "outright when the batch stops at a refusal, instead of keeping the "
        "successful prefix -- it only unwinds the document's own undo "
        "stack, so a tab this same batch minted, an element mode or "
        "selection change, or a reference added along the way all survive "
        "it untouched.\n\n"
        f"A call that outruns this bridge's {int(agent_host.CALL_TIMEOUT)}-"
        "second timeout is handled one of two ways, and the reply says "
        "which. If Warlock had not started the call yet, it is dropped and "
        "nothing changed -- it is safe to send the same call again. If "
        "Warlock had already started it, the call keeps running and will "
        "finish on its own; send the identical call again and, if its "
        "answer never reached you, it is replayed rather than run a second "
        f"time, or ask {agent_host.STATUS_TOOL} with the operation id the "
        "refusal named, or re-read clay_scene if you would rather see the "
        "document than the call's own answer. This only works for a call "
        "whose answer never arrived -- two identical calls that both got "
        "answered stay two calls, deliberately.\n\n"
        "Restarting the bridge, or losing its connection to the app, starts "
        "a new session with a new document -- never assume continuity across "
        "either; re-read clay_scene before trusting anything about the "
        "document again.\n\n"
        "Known generators: " + _generator_catalog()
    )


def tools() -> list[Any]:
    """Every tool Clay's agent surface offers, built fresh from the registries
    named in the module docstring. Called once per ``tools/list`` request, so
    rebuilding it from ``GENERATORS``/``ASSEMBLIES``/``OPS``/``QUERIES``/
    ``_HANDLERS`` each time costs nothing and can never go stale against an
    edit to any of them -- a query enum written out by hand would have been a
    fifth place to remember one exists, beside the menu, the tools pane, the
    key handler and ``OPS`` itself."""

    protocol = _protocol()
    primitive_names = sorted(bp.GENERATORS)
    figure_keys = sorted(presets.ASSEMBLIES)
    op_names = [op.name for op in clay_ops.OPS]
    axis_views = sorted(Camera.AXIS_VIEWS) + ["three_quarter"]
    batch_names = sorted(set(_HANDLERS) - BATCH_EXCLUDED)
    element_modes = list(el.MODES)
    query_names = sorted(bsel.QUERIES)

    return [
        protocol.Tool(
            name="clay_scene",
            title="Describe the scene",
            description=(
                "Every object in this session's document -- its generator and "
                "parameters (or its shape once an edit has frozen them, see "
                "'params'), its world-space translation/rotation/scale, its "
                "bounding box (and the box's own size and center), its face "
                "and vertex counts and its material slot -- plus the "
                "document's own bounds over its visible objects, its "
                "material palette, its current selection, element mode and "
                "whether it has unsaved changes."
            ),
            schema={"type": "object", "properties": {}, "additionalProperties": False},
            output_schema=_clay_scene_output_schema(),
        ),
        protocol.Tool(
            name="clay_add_primitive",
            title="Add a primitive",
            description=(
                "Place one primitive, selected, as one undo step. Starts "
                "this session's document if it has none yet. 'generator' "
                "picks the shape; 'params' overrides its own numbers "
                "(radius, segments and so on) -- one generator, lathe, "
                "takes a 'profile' instead: an array of [radius, y] "
                "stations, bottom to top, revolved about Y; "
                "'translation'/'rotation'/'scale' place it directly rather "
                "than at the origin; "
                "'name' sets what it is called, refused if another object "
                "already wears it; 'material' paints it with an existing "
                "palette index (see clay_scene's 'materials') rather than "
                "the default grey -- appending a new material is "
                "clay_material's job, not this one's. Everything is "
                "validated before anything is placed, so a refused call "
                "places nothing. Returns the same row clay_scene would show "
                "for it. Known generators: " + _generator_catalog()
            ),
            schema={
                "type": "object",
                "properties": {
                    "generator": {"type": "string", "enum": primitive_names},
                    "params": {
                        "type": "object",
                        "additionalProperties": _params_value_schema(),
                    },
                    "translation": _vec3_schema("metres"),
                    "rotation": _vec3_schema("degrees, Euler XYZ"),
                    "scale": _vec3_schema("a multiplier per axis"),
                    "name": {"type": "string"},
                    "material": {"type": "integer", "minimum": 0},
                },
                "required": ["generator"],
                "additionalProperties": False,
            },
            output_schema=_object_row_output_schema(),
        ),
        protocol.Tool(
            name="clay_add_figure",
            title="Add a figure",
            description=(
                "Place every part of a rigged figure preset -- a humanoid, a "
                "quadruped, a bird and so on -- as one grounded group, as "
                "one undo step. Starts this session's document if it has "
                "none yet. 'translation' offsets the whole group; 'yaw' "
                "turns it about Y, in degrees, around the group's own "
                "origin -- it rotates where the parts sit, not each part in "
                "place; 'scale' is one uniform number, since a figure is a "
                "proportioned thing and a per-axis scale is how you get a "
                "squashed head; 'name_prefix' is prepended to every part's "
                "name, refused if it would collide with an object already "
                "in the document."
            ),
            schema={
                "type": "object",
                "properties": {
                    "key": {"type": "string", "enum": figure_keys},
                    "translation": _vec3_schema("metres"),
                    "yaw": {"type": "number", "description": "degrees, about Y"},
                    "scale": {"type": "number", "exclusiveMinimum": 0.0},
                    "name_prefix": {"type": "string"},
                },
                "required": ["key"],
                "additionalProperties": False,
            },
        ),
        protocol.Tool(
            name="clay_add_mesh",
            title="Add a hand-built mesh",
            description=(
                "Place geometry an agent computed itself -- a shape none of "
                "the fifteen generators expresses -- selected, as one undo "
                "step. Starts this session's document if it has none yet. "
                "'positions' is an array of [x, y, z]; 'faces' is an array "
                "of vertex-index loops, three or more per face, wound "
                "counter-clockwise seen from outside; 'uv', if given, is "
                "one (u, v) per face *corner* rather than per vertex, "
                "nested exactly like 'faces', because a texture seam is "
                "one vertex carrying two different coordinates, which a "
                "per-vertex array cannot express. 'translation'/'rotation'/"
                "'scale'/'name'/'material' are exactly clay_add_primitive's "
                "own. The placed object has no generator -- there is no "
                "recipe to hand clay_set_params for geometry that arrived "
                "as raw coordinates -- so clay_set_params refuses it the "
                "same way it refuses any object whose topology has already "
                "been edited. Everything is validated, naming the "
                "offending face or corner, before anything is placed. "
                f"Accepts up to {MAX_MESH_VERTICES:,} vertices and "
                f"{MAX_MESH_FACES:,} faces per call. Returns the same row "
                "clay_scene would show for it, plus 'closed' (true if the "
                "mesh has no hole and no non-manifold edge -- what "
                "clay_boolean needs) and 'findings' (the same rows "
                "clay_diagnose reports)."
            ),
            schema={
                "type": "object",
                "properties": {
                    "positions": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 3,
                            "maxItems": 3,
                        },
                    },
                    "faces": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 3,
                        },
                    },
                    "uv": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 2,
                                "maxItems": 2,
                            },
                        },
                    },
                    "translation": _vec3_schema("metres"),
                    "rotation": _vec3_schema("degrees, Euler XYZ"),
                    "scale": _vec3_schema("a multiplier per axis"),
                    "name": {"type": "string"},
                    "material": {"type": "integer", "minimum": 0},
                },
                "required": ["positions", "faces"],
                "additionalProperties": False,
            },
            output_schema=_mesh_row_output_schema(),
        ),
        protocol.Tool(
            name="clay_transform",
            title="Move, rotate or scale an object",
            description=(
                "Set one object's translation, rotation and/or scale, as one "
                "undo step. Rotation is Euler X, then Y, then Z, in degrees -- "
                "not a quaternion."
            ),
            schema={
                "type": "object",
                "properties": {
                    "uid": {"type": "integer"},
                    "translation": _vec3_schema("metres"),
                    "rotation": _vec3_schema("degrees, Euler XYZ"),
                    "scale": _vec3_schema("a multiplier per axis"),
                },
                "required": ["uid"],
                "additionalProperties": False,
            },
        ),
        protocol.Tool(
            name="clay_set_params",
            title="Set a primitive's own parameters",
            description=(
                "Change how tall a cylinder is, how many segments it has, how "
                "thick a torus's tube is, or a column's base and capital -- "
                "the numbers a generator was built from, which scale alone "
                "cannot reach. Merges over the object's current params, then "
                "rebuilds the mesh, as one undo step. Only for an object whose "
                "generator is still set (clay_scene's 'generator' is not "
                "null) -- once an edit has frozen its topology there are no "
                "generator params left to set. Values may be a number, an "
                "array -- box's size is (x, y, z), plane's is (w, h) -- or "
                "an array of arrays -- a lathe's profile is a list of "
                "[radius, y] stations, bottom to top. "
                "Give exactly one of uid (one object) or uids (several): "
                "'make the wheels larger' is one call naming every wheel's "
                "uid, not one call per wheel, and it stays one undo step. "
                "The same params go to every object named, and the call is "
                "all-or-nothing -- each must exist, still have a generator, "
                "and accept every key in params for *its own* generator, all "
                "checked before any is rebuilt, so a radius handed to a box "
                "among five cylinders refuses the whole call, names that uid "
                "and its generator, and rebuilds nothing. "
                "Known generators: " + _generator_catalog()
            ),
            schema={
                "type": "object",
                "properties": {
                    "uid": {"type": "integer"},
                    "uids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 1,
                    },
                    "params": {
                        "type": "object",
                        "additionalProperties": _params_value_schema(),
                        "description": "Only the keys to change; every other one keeps its value.",
                    },
                },
                "required": ["params"],
                "additionalProperties": False,
            },
        ),
        protocol.Tool(
            name="clay_material",
            title="Give objects a material",
            description=(
                "Append one palette entry and paint every named object's "
                "whole surface with it, as one undo step -- 'a wooden barrel "
                "with iron bands' is two calls, one material each. Repaints "
                "every existing face, not just the object's default slot for "
                "future ones, so an already-built object comes out the "
                "colour asked for. Color is linear 0..1 RGB or RGBA; a "
                "3-element color exports fully opaque. Metallic and "
                "roughness default to a plain painted dielectric."
            ),
            schema={
                "type": "object",
                "properties": {
                    "uids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 1,
                    },
                    "name": {"type": "string"},
                    "color": {
                        "type": "array",
                        "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "minItems": 3,
                        "maxItems": 4,
                    },
                    "metallic": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "roughness": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
                "required": ["uids", "color"],
                "additionalProperties": False,
            },
        ),
        protocol.Tool(
            name="clay_boolean",
            title="Boolean two or more objects",
            description=(
                "Union, subtract or intersect the given objects into the one "
                "that comes first in the document's own object order -- not "
                "the order given here, which only says which objects take "
                "part. A closed-solid requirement applies to all three; the "
                "refusal names which object is not one. Refused in "
                "vertex/edge/face mode for the same reason clay_select is -- "
                "call clay_element_mode with mode='object' first."
            ),
            schema={
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": list(ops_boolean.KINDS)},
                    "uids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 2,
                    },
                },
                "required": ["kind", "uids"],
                "additionalProperties": False,
            },
        ),
        protocol.Tool(
            name="clay_select",
            title="Set the object selection",
            description=(
                "Replace the document's object selection. An empty list "
                "clears it. Refused in vertex/edge/face mode: the object "
                "selection is derived from the element selection there, so "
                "this tool would either be overwritten by it or manufacture "
                "an object 'selected' with nothing selected inside it. Call "
                "clay_element_mode with mode='object' first."
            ),
            schema={
                "type": "object",
                "properties": {"uids": {"type": "array", "items": {"type": "integer"}}},
                "required": ["uids"],
                "additionalProperties": False,
            },
        ),
        protocol.Tool(
            name="clay_element_mode",
            title="Switch object/vertex/edge/face mode",
            description=(
                "Switch the document's element mode, converting whatever is "
                "already selected into the new mode's own currency -- going "
                "down (face to edge to vertex) is everything touched; going "
                "up, only an element every one of whose lower parts is "
                "selected. Entering vertex/edge/face mode from object mode "
                "selects nothing, which is what makes clay_op's seedless "
                "rows (select-all, select-boundary and the rest) reachable "
                "with no prior selection. Every element-gated clay_op row -- "
                "inset, bevel, extrude and the rest -- refuses by name until "
                "this has been called at least once; clay_select_elements "
                "and clay_select_by can switch mode in the same call "
                "instead of a separate one. Not undoable -- element mode is "
                "document state, never an edit. Returns the new mode plus, "
                "per object that still has something selected, its uid, "
                "stamp and per-kind counts."
            ),
            schema={
                "type": "object",
                "properties": {"mode": {"type": "string", "enum": element_modes}},
                "required": ["mode"],
                "additionalProperties": False,
            },
        ),
        protocol.Tool(
            name="clay_select_elements",
            title="Select vertices, edges or faces by index",
            description=(
                "Set what is selected inside one object, by explicit "
                "vertex/edge/face index. 'mode' switches the document's "
                "element mode first, converting whatever was already "
                "selected -- the order matters for how='add', since the "
                "prior selection is converted into the new mode's currency "
                "before the union runs. 'how' is replace/add/subtract, the "
                "same three click modifiers the viewport's Shift and Ctrl "
                "give. Every index is validated, never clamped: a vertex or "
                "face past the mesh's own count is refused by name, and "
                "every edge is a [vertex, vertex] pair that must actually be "
                "an edge of this mesh -- an unchecked pair would draw an "
                "overlay line between two vertices with nothing between "
                "them, and an out-of-range face would take the overlay "
                "build down. 'expect_stamp' refuses the whole call, "
                "unchanged, if the mesh has moved on since that stamp was "
                "read (clay_scene, clay_element_mode, an op's own result) -- "
                "see clay_elements for reading indices back."
            ),
            schema={
                "type": "object",
                "properties": {
                    "uid": {"type": "integer"},
                    "mode": {"type": "string", "enum": element_modes},
                    "verts": {"type": "array", "items": {"type": "integer"}},
                    "edges": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 2,
                            "maxItems": 2,
                        },
                    },
                    "faces": {"type": "array", "items": {"type": "integer"}},
                    "how": {"type": "string", "enum": ["replace", "add", "subtract"]},
                    "expect_stamp": {"type": "integer"},
                },
                "required": ["uid"],
                "additionalProperties": False,
            },
        ),
        protocol.Tool(
            name="clay_select_by",
            title="Select elements by a question, not an index",
            description=(
                "Answer a selection from a seed or a parameter instead of "
                "listing indices by hand -- the loop or ring through an "
                "edge, the strip of faces through one, everything painted "
                "with a material slot, the faces facing a direction, or "
                "everything inside a box. Refused when the document's "
                "current element mode cannot answer the query named -- "
                "switch with clay_element_mode, or clay_select_elements's "
                "own 'mode', first. 'how' and 'expect_stamp' work exactly as "
                "they do on clay_select_elements. The seven seedless verbs "
                "(select-all, select-none, select-invert, select-linked, "
                "select-more, select-less, select-boundary) are clay_op "
                "rows, not here -- this tool is only for a query that needs "
                "a seed or a parameter to answer. Known queries: "
                + _query_catalog()
            ),
            schema={
                "type": "object",
                "properties": {
                    "uid": {"type": "integer"},
                    "query": {"type": "string", "enum": query_names},
                    "how": {"type": "string", "enum": ["replace", "add", "subtract"]},
                    "expect_stamp": {"type": "integer"},
                    **_QUERY_ARG_SCHEMAS,
                },
                "required": ["uid", "query"],
                "additionalProperties": False,
            },
        ),
        protocol.Tool(
            name="clay_elements",
            title="Read one object's element indices, paged",
            description=(
                "Per object -- the named uid, or every object that "
                "currently has something selected -- the element mode's own "
                "per-kind counts, and, when 'kind' ('vertex', 'edge' or "
                "'face') is given, a bounded page of that kind's raw indices "
                "with its own total and offset. Omit 'kind' for counts "
                "alone: most of what an agent needs is already in "
                "clay_scene, clay_element_mode or an op's own 'selected' "
                "field, and this exists for the rarer moment it has to "
                f"reason about which ones. At most {ELEMENT_PAGE_MAX:,} "
                f"indices a call; default limit {ELEMENT_PAGE_DEFAULT}."
            ),
            schema={
                "type": "object",
                "properties": {
                    "uid": {"type": "integer"},
                    "kind": {"type": "string", "enum": ["vertex", "edge", "face"]},
                    "offset": {"type": "integer", "minimum": 0},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": ELEMENT_PAGE_MAX,
                    },
                },
                "additionalProperties": False,
            },
        ),
        protocol.Tool(
            name="clay_op",
            title="Run a Clay op",
            description=(
                "The escape hatch over every op Clay can run, by name -- the "
                "same registry the context menu, the tools pane and the "
                "keyboard read. Runs sandboxed: it never moves the user's own "
                "viewport (Frame Selection is a no-op here) and any refusal "
                "an op raises per object comes back as a 'messages' list in "
                "the result rather than a toast only the person at the "
                "keyboard would see. Known ops: " + _op_catalog(op_names)
            ),
            schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": op_names},
                    "params": {
                        "type": "object",
                        "additionalProperties": {"type": "number"},
                        "description": (
                            "Only for a parameterised op; missing fields fall "
                            "back to that op's own defaults."
                        ),
                    },
                },
                "required": ["name"],
                "additionalProperties": False,
            },
        ),
        protocol.Tool(
            name="clay_render",
            title="Render the document",
            description=(
                "One or more flat-shaded, white-background square renders "
                "of the document -- no gizmos -- from the standard "
                "three-quarter framing, a named axis view, or a free "
                "yaw/pitch pair. 'view' and 'views' are exclusive; giving "
                "both is refused. 'grid' draws the ground plane at y=0, the "
                "one scale cue available with no viewport to walk around in: "
                "16 cells across a span rounded up to a power of ten "
                "containing 2.5x the framed footprint, so one cell reads as "
                "span/16 metres. 'focus' points the camera at the union of "
                "the named objects' boxes -- everything else is still "
                "drawn, since the renderer has no per-object alpha. Refused, "
                "before any GPU work, when the requested views would exceed "
                f"the {RENDER_PIXEL_BUDGET:,}-pixel render budget or would "
                "not fit in one reply frame once encoded -- ask for fewer or "
                "smaller views instead. Pass 'compare' (a stored reference's "
                "name) to render exactly one view beside it or blended over "
                "it instead of the normal multi-view result -- see "
                "clay_reference_add."
            ),
            schema={
                "type": "object",
                "properties": {
                    "size": {"type": "integer", "minimum": 64, "maximum": 2048},
                    "view": {"type": "string", "enum": axis_views},
                    "views": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "anyOf": [
                                {"type": "string", "enum": axis_views},
                                {
                                    "type": "object",
                                    "properties": {
                                        "yaw": {"type": "number"},
                                        "pitch": {
                                            "type": "number",
                                            "minimum": -89.0,
                                            "maximum": 89.0,
                                        },
                                    },
                                    "required": ["yaw", "pitch"],
                                    "additionalProperties": False,
                                },
                            ]
                        },
                    },
                    "grid": {"type": "boolean"},
                    "focus": {"type": "array", "items": {"type": "integer"}},
                    "compare": {"type": "string"},
                    "compare_mode": {"type": "string", "enum": ["beside", "overlay"]},
                    "alpha": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
                "additionalProperties": False,
            },
        ),
        protocol.Tool(
            name="clay_diagnose",
            title="Check a mesh for defects",
            description=(
                "Holes, non-manifold or flipped edges, duplicate faces and "
                "unused vertices -- for one object, or every visible object "
                "when none is named. Pass 'select' with the uid and the "
                "'kind' of one finding this call reported (or a prior one) "
                "to switch to that finding's own element mode and select "
                "exactly the elements it names -- refused if that object has "
                "no finding of that kind right now. A whole-document call "
                "also answers with 'scene': findings about how objects "
                "relate rather than about one mesh, such as a family of "
                "copies (Box, Box.001...) that no longer agree about their "
                "material -- what naming only the original in clay_material "
                "after arraying or mirroring it leaves behind."
            ),
            schema={
                "type": "object",
                "properties": {
                    "uid": {"type": "integer"},
                    "select": {
                        "type": "object",
                        "properties": {
                            "uid": {"type": "integer"},
                            "kind": {"type": "string"},
                        },
                        "required": ["uid", "kind"],
                        "additionalProperties": False,
                    },
                },
                "additionalProperties": False,
            },
            output_schema=_clay_diagnose_output_schema(),
        ),
        protocol.Tool(
            name="clay_analyze",
            title="Measure bounds, mass and contact -- never selects",
            description=(
                "Facts, not defects: exact world-space bounds, area, volume "
                "(null unless closed), connected components, ground contact "
                "and symmetry for one or more objects, plus pairwise "
                "distance/contact/overlap and -- for a whole-document call, "
                "no uids given -- which objects are floating (touching "
                "nothing that reaches the ground). Use clay_diagnose to find "
                "what is wrong with a mesh and select it; use this to learn "
                "how big something is, whether it is touching the ground or "
                "another object, or how deep two objects overlap. Bounds "
                "here are the object's own exact extent under its current "
                "rotation, which is tighter than clay_scene's 'bbox' -- that "
                "one transforms the local bounding box's own corners, "
                "conservative for anything that is not itself box-shaped. "
                "Refused past 64 objects or 200,000 triangles combined; "
                "past 500,000 candidate triangle pairs for one object pair, "
                "that pair's distance is a cheaper vertex estimate marked "
                "exact:false instead."
            ),
            schema={
                "type": "object",
                "properties": {
                    "uids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 1,
                        "description": "Only these objects, and pairs among "
                        "them -- no floating check. Omitted means every "
                        "visible object, with floating computed.",
                    },
                    "contact_tol": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Metres apart still counted as touching. Default 0.001.",
                    },
                    "near": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 10.0,
                        "description": "Metres of margin a pair's boxes must "
                        "overlap by to be looked at closely at all. Default 0.05.",
                    },
                    "symmetry_tol": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Mirror-partner tolerance, as a "
                        "fraction of the object's own bounds diagonal. Default 0.002.",
                    },
                },
                "additionalProperties": False,
            },
            output_schema=_clay_analyze_output_schema(),
        ),
        protocol.Tool(
            name="clay_export",
            title="Export the document as an asset",
            description=(
                "Mint a finished model row from the document, the way Clay's "
                "own Export does, and also keep the authored document as a "
                ".wblk this session can be resumed from. Returns the new "
                "job's id."
            ),
            schema={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        protocol.Tool(
            name="clay_undo",
            title="Undo",
            description=(
                "Reverse up to 'steps' tool calls (default 1), moving the "
                "history head backwards rather than pushing a step of its "
                "own -- the documented exception to 'one call is one undo "
                "step'. Over-asking moves as far as it can rather than "
                "refusing; 'moved' in the result is a count of steps "
                "actually reversed, not a flag."
            ),
            schema={
                "type": "object",
                "properties": {"steps": {"type": "integer", "minimum": 1, "maximum": 64}},
                "additionalProperties": False,
            },
        ),
        protocol.Tool(
            name="clay_redo",
            title="Redo",
            description=(
                "Replay up to 'steps' undone tool calls (default 1), moving "
                "the history head forwards -- clay_undo's own counterpart "
                "and the same documented exception to 'one call is one undo "
                "step'. Over-asking moves as far as it can rather than "
                "refusing; 'moved' is a count of steps actually replayed."
            ),
            schema={
                "type": "object",
                "properties": {"steps": {"type": "integer", "minimum": 1, "maximum": 64}},
                "additionalProperties": False,
            },
        ),
        protocol.Tool(
            name="clay_delete",
            title="Delete objects",
            description=(
                "Remove the named objects, in any element mode, as one undo "
                "step -- never a wrapper over clay_op's own Delete row, "
                "which acts on the selection and in a face or edge mode can "
                "leave every object standing."
            ),
            schema={
                "type": "object",
                "properties": {
                    "uids": {"type": "array", "items": {"type": "integer"}, "minItems": 1}
                },
                "required": ["uids"],
                "additionalProperties": False,
            },
        ),
        protocol.Tool(
            name="clay_rename",
            title="Rename an object",
            description=(
                "Give one object a new name. Refused if the name is empty "
                "or already worn by another object."
            ),
            schema={
                "type": "object",
                "properties": {"uid": {"type": "integer"}, "name": {"type": "string"}},
                "required": ["uid", "name"],
                "additionalProperties": False,
            },
        ),
        protocol.Tool(
            name="clay_batch",
            title="Run several tools as one undo step",
            description=(
                "Run up to "
                f"{BATCH_MAX} tool calls in order, folded into a single undo "
                "step, stopping at the first refusal and keeping the "
                "successful prefix. If this session owns no document yet, "
                "the first call must be one of "
                f"{', '.join(MINTS_A_DOCUMENT)}. "
                "clay_batch, clay_render, clay_export, "
                "clay_undo, clay_redo and clay_reference_get cannot be "
                "batched -- see their own tools for why. Anywhere inside a "
                "later entry's arguments, {\"$ref\": \"<name>\"} resolves to "
                "the uid of the object of that name as the document stands "
                "when that entry runs -- so an earlier entry can name an "
                "object (clay_add_primitive/clay_add_figure/clay_add_mesh's "
                "own name argument) and a later one can address it by that "
                "name, with no clay_scene read in between. rollback_on_error "
                "(default false): when true and the batch stops at a "
                "refusal, the folded step is undone -- not left for a later "
                "clay_undo, and not redoable -- before this call returns, so "
                "the successful prefix never stays on the document. This "
                "only unwinds the document's own undo stack: a document "
                "this very batch minted still exists, empty rather than "
                "gone, because minting one pushes no undo step to begin "
                "with; an element mode or selection change an earlier entry "
                "made, and a reference clay_reference_add took, both "
                "survive it exactly as they would survive an ordinary "
                "clay_undo, because neither one ever pushed a step either."
            ),
            schema={
                "type": "object",
                "properties": {
                    "calls": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": BATCH_MAX,
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "enum": batch_names},
                                "arguments": {"type": "object"},
                            },
                            "required": ["name"],
                            "additionalProperties": False,
                        },
                    },
                    "rollback_on_error": {"type": "boolean"},
                },
                "required": ["calls"],
                "additionalProperties": False,
            },
        ),
        protocol.Tool(
            name="clay_reference_add",
            title="Add a reference image",
            description=(
                "Hand this session a picture to match against -- from a "
                "Library job's own image, or inline as base64 -- kept in "
                "memory on the session only, never in the document and "
                f"never saved to disk. Up to {MAX_REFERENCES} at a time; "
                "re-adding an existing name replaces it without spending a "
                "slot. Pushes no undo step: nothing in the document "
                "changed."
            ),
            schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "job_id": {"type": "string"},
                    "file": {
                        "type": "string",
                        "description": (
                            "One of input.png/ref.png/reference.png/thumb.png, "
                            "with job_id. Omit to use the first one ready."
                        ),
                    },
                    "png_base64": {"type": "string"},
                    "view": {
                        "type": "string",
                        "enum": sorted(set(Camera.AXIS_VIEWS) | {"three_quarter", "other"}),
                    },
                },
                "required": ["name"],
                "additionalProperties": False,
            },
        ),
        protocol.Tool(
            name="clay_reference_list",
            title="List this session's references",
            description="Every reference this session holds, by name.",
            schema={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        protocol.Tool(
            name="clay_reference_get",
            title="Fetch one reference image",
            description="One stored reference's metadata and its picture, by name.",
            schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
                "additionalProperties": False,
            },
        ),
        protocol.Tool(
            name="clay_reference_remove",
            title="Forget a reference image",
            description="Drop one stored reference, freeing its slot.",
            schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
                "additionalProperties": False,
            },
        ),
    ]


def _vec3_schema(unit: str) -> dict:
    """The one shape every optional TRS vector in this module shares."""
    return {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 3,
        "maxItems": 3,
        "description": unit,
    }


def _params_value_schema() -> dict:
    """The one shape every ``params`` value in this module shares: a number, a
    flat array of numbers, or an array of arrays of numbers.

    Written out twice before this -- at ``clay_add_primitive`` and
    ``clay_set_params`` -- as an ``anyOf`` of only the first two branches,
    which is exactly the drift the module docstring's derivation paragraphs
    exist to rule out for everything else in this file. The third branch is
    for ``lathe``'s ``profile``: a cylinder's ``radius`` is one number, a
    box's ``size`` is three, and a lathe's ``profile`` is an array of
    ``[radius, y]`` pairs -- the first generator parameter this registry has
    whose own elements are arrays rather than numbers. A schema declaring
    this shape is not enforcement of it; :func:`_validate_number_or_vec` is
    the handler-side half both tools already call.
    """
    return {
        "anyOf": [
            {"type": "number"},
            {"type": "array", "items": {"type": "number"}},
            {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
        ]
    }


# --- output schemas -----------------------------------------------------------
#
# Five tools below declare an ``outputSchema`` at all -- ``clay_scene``,
# ``clay_add_primitive``, ``clay_add_mesh``, ``clay_diagnose`` and
# ``clay_analyze``. Every other tool's result is small and self-explanatory
# (a uid, a count, a list of names); writing a schema for each would be
# schema authoring with no reader, so this file deliberately does not.
# These five are the ones whose shape is worth writing down once rather
# than making a client work it back out of a sample reply.
#
# None of the five declares ``required``, and none sets
# ``additionalProperties: false``. That is not an oversight -- a refusal
# from any of these tools answers through the *same* result envelope
# (``protocol.fail``), and a refusal's own ``structuredContent`` is whatever
# ``fail``'s ``**extra`` was given -- ``{"field": "uids"}`` and nothing else.
# A ``required`` list on the success shape would make every refusal of
# these tools non-conforming for a client that validates strictly against
# ``outputSchema``, and MCP's own wording on whether an ``isError`` result
# must still conform to it is not explicit enough to bet a client's error
# handling on that reading. Declaring ``properties`` still documents the
# shape for a reader -- it just never claims a key the envelope cannot
# promise to keep filled.


def _sel_counts_schema() -> dict:
    """The shape :func:`_sel_counts` returns -- three counts, shared by the
    object row schema below and ``clay_diagnose``'s own ``selected`` field,
    exactly as the one ``_sel_counts`` function is shared by both callers."""
    return {
        "type": "object",
        "properties": {
            "verts": {"type": "integer"},
            "edges": {"type": "integer"},
            "faces": {"type": "integer"},
        },
    }


def _object_row_output_schema() -> dict:
    """The JSON Schema for one row of :func:`_scene_row` -- read that
    function, not this one, when deciding what belongs here: every key it
    returns must appear below with the right type, or this schema has
    drifted from the function that actually builds the row. That is the
    drift class this module's own docstring warns about for a hand-written
    second copy of a shape a real function already owns, so this helper is
    the one place it is written, used by both :func:`_clay_scene_output_schema`
    (inside ``objects``) and ``clay_add_primitive``'s own declared schema,
    which *is* this schema -- its result is one row, unwrapped.

    ``bbox``, ``size`` and ``center`` admit ``null``: :func:`_scene_row`
    reports all three as ``None`` for an object with no box. ``params`` is
    an open object -- what is in it depends on ``generator``, which this
    schema has no way to branch on. See the module comment above for why
    nothing here is ``required``.
    """
    vec3 = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
    nullable_vec3 = {"anyOf": [{"type": "null"}, vec3]}
    return {
        "type": "object",
        "properties": {
            "uid": {"type": "integer"},
            "name": {"type": "string"},
            "visible": {"type": "boolean"},
            "generator": {"type": "string"},
            "params": {"type": "object"},
            "faces": {"type": "integer"},
            "material": {"type": "integer"},
            "bbox": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "object",
                        "properties": {
                            "min": {"type": "array", "items": {"type": "number"}},
                            "max": {"type": "array", "items": {"type": "number"}},
                        },
                    },
                ]
            },
            "translation": vec3,
            "rotation": vec3,
            "scale": vec3,
            "size": nullable_vec3,
            "center": nullable_vec3,
            "verts": {"type": "integer"},
            "stamp": {"type": "integer"},
            "selected": _sel_counts_schema(),
        },
    }


def _clay_scene_output_schema() -> dict:
    """``clay_scene``'s declared ``outputSchema`` -- built from what
    :func:`_h_scene` actually returns (``objects``, ``selection``,
    ``element_mode``, ``dirty``, ``object_count``, ``bounds``, ``materials``),
    with ``objects`` built from the one shared :func:`_object_row_output_schema`
    rather than a second, hand-written copy of the row shape."""
    vec3 = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
    return {
        "type": "object",
        "properties": {
            "objects": {"type": "array", "items": _object_row_output_schema()},
            "selection": {"type": "array", "items": {"type": "integer"}},
            "element_mode": {"type": "string"},
            "dirty": {"type": "boolean"},
            "object_count": {"type": "integer"},
            "bounds": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "object",
                        "properties": {
                            "min": vec3,
                            "max": vec3,
                            "size": vec3,
                            "center": vec3,
                        },
                    },
                ]
            },
            "materials": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "name": {"type": "string"},
                        "color": {"type": "array", "items": {"type": "number"}},
                        "metallic": {"type": "number"},
                        "roughness": {"type": "number"},
                    },
                },
            },
        },
    }


def _finding_row_output_schema() -> dict:
    """The shape of one ``clay_diagnose.Finding`` row, as JSON -- shared by
    ``clay_diagnose``'s own output schema and ``clay_add_mesh``'s (see
    :func:`_mesh_row_output_schema`), so a finding's shape is written once
    rather than copied the second time a tool needed to describe it."""
    return {
        "type": "object",
        "properties": {
            "kind": {"type": "string"},
            "label": {"type": "string"},
            "count": {"type": "integer"},
            "mode": {"type": "string"},
        },
    }


def _mesh_row_output_schema() -> dict:
    """``clay_add_mesh``'s declared ``outputSchema`` -- the same object row
    :func:`_object_row_output_schema` already describes, composed rather than
    copied, plus the two keys only this tool answers with: ``closed`` (no
    hole, no non-manifold edge -- see ``_h_add_mesh``'s own docstring for why
    those two findings are what "closed" means here) and ``findings`` (the
    same rows ``clay_diagnose`` reports, via the shared
    :func:`_finding_row_output_schema`), so an agent that just handed over
    geometry learns what -- if anything -- is wrong with it in the same call
    that placed it."""
    row = _object_row_output_schema()
    return {
        **row,
        "properties": {
            **row["properties"],
            "closed": {"type": "boolean"},
            "findings": {"type": "array", "items": _finding_row_output_schema()},
        },
    }


def _clay_diagnose_output_schema() -> dict:
    """``clay_diagnose``'s declared ``outputSchema`` -- built from what
    :func:`_h_diagnose` actually returns: ``objects`` always, ``selected``
    only when ``select`` was given and matched a finding."""
    return {
        "type": "object",
        "properties": {
            "objects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "uid": {"type": "integer"},
                        "name": {"type": "string"},
                        "clean": {"type": "boolean"},
                        "findings": {
                            "type": "array",
                            "items": _finding_row_output_schema(),
                        },
                    },
                },
            },
            # Document-level rows, present only on a whole-document call and
            # only when there is something to say. Not ``findings``' shape:
            # they point at objects (``uids``) rather than at elements, so
            # they carry no ``mode`` and are not reachable through ``select``.
            "scene": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string"},
                        "label": {"type": "string"},
                        "uids": {"type": "array", "items": {"type": "integer"}},
                    },
                },
            },
            "selected": {
                "type": "object",
                "properties": {
                    "uid": {"type": "integer"},
                    "kind": {"type": "string"},
                    "mode": {"type": "string"},
                    "stamp": {"type": "integer"},
                    "selected": _sel_counts_schema(),
                },
            },
        },
    }


def _clay_analyze_output_schema() -> dict:
    """``clay_analyze``'s declared ``outputSchema`` -- built from what
    :func:`_h_analyze` actually returns. ``bounds`` admits ``null`` for an
    object with no vertices, exactly as ``clay_scene``'s own ``bbox`` does,
    and for the same reason: :func:`~.analyze.analyze` cannot measure a box
    around nothing."""
    vec3 = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
    ground_schema = {
        "type": "object",
        "properties": {
            "min_y": {"type": "number"},
            "contact": {"type": "boolean"},
            "penetration": {"type": "number"},
        },
    }
    overlap_schema = {
        "type": "object",
        "properties": {
            "volume": {"type": "number"},
            "depth": {"type": "number"},
        },
    }
    return {
        "type": "object",
        "properties": {
            "objects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "uid": {"type": "integer"},
                        "name": {"type": "string"},
                        "bounds": {
                            "anyOf": [
                                {"type": "null"},
                                {"type": "object", "properties": {"min": vec3, "max": vec3}},
                            ]
                        },
                        "area": {"type": "number"},
                        "volume": {"anyOf": [{"type": "null"}, {"type": "number"}]},
                        "closed": {"type": "boolean"},
                        "components": {"type": "integer"},
                        "ground": {"anyOf": [{"type": "null"}, ground_schema]},
                        "symmetry": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 3,
                            "maxItems": 3,
                        },
                    },
                },
            },
            "pairs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "uids": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 2,
                            "maxItems": 2,
                        },
                        "distance": {"anyOf": [{"type": "null"}, {"type": "number"}]},
                        "intersects": {"type": "boolean"},
                        "contact": {"type": "boolean"},
                        "overlap": {"anyOf": [{"type": "null"}, overlap_schema]},
                        "exact": {"type": "boolean"},
                    },
                },
            },
            "floating": {"type": "array", "items": {"type": "integer"}},
            "truncated": {"type": "boolean"},
            "tolerances": {
                "type": "object",
                "properties": {
                    "contact_tol": {"type": "number"},
                    "near": {"type": "number"},
                    "symmetry_tol": {"type": "number"},
                },
            },
        },
    }


def _param_prose(p: clay_ops.Param) -> str:
    """One param, worded for what it actually *is*.

    A bare "name (low-high, default x)" describes a number field, and that is
    all a model is ever told about an op's argument -- so it was a poor
    description of the boolean and three-way choice ``clay_ops.Param`` grew
    on 2026-09-10 (``place-between``'s ``fit``, ``array-radial`` and
    ``mirror-copy``'s ``axis``): "fit (0.0-1.0, default 1.0)" does not say
    "this is on unless you turn it off", and "axis (0.0-2.0, default 1.0)"
    does not say which number is which axis. The call itself is unchanged --
    ``params`` in the schema below is still ``{"type": "number"}`` for every
    field, a checkbox is still 0/1 and a choice is still its index -- only
    the sentence describing it gets to say what kind it is.
    """
    if p.boolean:
        return f"{p.name} (boolean 0/1, default {int(p.default)})"
    if p.choices:
        named = ", ".join(f"{i}={choice}" for i, choice in enumerate(p.choices))
        return f"{p.name} (choice: {named}, default {int(p.default)})"
    return f"{p.name} ({p.low}-{p.high}, default {p.default})"


def _op_catalog(names: list[str]) -> str:
    """Every op's params and bounds, folded into one sentence.

    Built from :data:`clay_ops.OPS` rather than written out, so a thirteenth
    op -- with or without parameters -- appears here the next time ``tools()``
    is called and nowhere needs editing for it to.
    """
    parts = []
    for op in clay_ops.OPS:
        modes = "/".join(op.modes)
        if op.params:
            fields = ", ".join(_param_prose(p) for p in op.params)
            parts.append(f"{op.name} [{modes}: {fields}]")
        else:
            parts.append(f"{op.name} [{modes}]")
    del names
    return "; ".join(parts)


def _generator_catalog() -> str:
    """Every generator's own keys and their defaults, folded into one sentence.

    Built from :data:`primitives.GENERATORS` rather than written out, so a
    thirteenth primitive's parameters appear here the next time ``tools()`` is
    called and nowhere needs editing for it to -- the same rule
    :func:`_op_catalog` already follows for ``clay_op``.
    """
    parts = []
    for name in sorted(bp.GENERATORS):
        defaults = bp.GENERATORS[name][0]
        fields = ", ".join(f"{key}={value!r}" for key, value in defaults.items())
        parts.append(f"{name} [{fields}]")
    return "; ".join(parts)


def _query_catalog() -> str:
    """Every ``clay_select_by`` query's own argument names and its hint,
    folded into one sentence.

    Built from :data:`select.QUERIES` rather than written out, so a seventh
    query appears here the next time ``tools()`` is called and nowhere needs
    editing for it to -- the same rule :func:`_op_catalog` and
    :func:`_generator_catalog` already follow for their own registries.
    """
    parts = []
    for name in sorted(bsel.QUERIES):
        query = bsel.QUERIES[name]
        args = ", ".join(query.args)
        parts.append(f"{name} [{args}]: {query.hint}")
    return " ".join(parts)


# The one mapping from a query argument's *name* to the JSON-schema fragment
# that describes it -- ``select.py`` holds no JSON-schema knowledge at all
# (its own module docstring's rule), so this is where that vocabulary is
# spelled out, once, for every query that shares an argument name rather than
# once per query. Keys are the union of every ``Query.args`` tuple in
# :data:`select.QUERIES`; ``test_every_query_argument_name_has_a_schema_
# fragment_and_vice_versa`` gates both directions, so a query that grows an
# argument nobody here can express -- or an entry here nothing asks for any
# more -- fails the suite instead of drifting quietly.
_QUERY_ARG_SCHEMAS: dict[str, dict] = {
    "edge": {
        "type": "array",
        "items": {"type": "integer"},
        "minItems": 2,
        "maxItems": 2,
        "description": "A [vertex, vertex] pair naming one edge -- the seed "
        "to walk the loop or ring from.",
    },
    "face": {
        "type": "integer",
        "description": "A face index -- the seed to walk the face loop from.",
    },
    "slot": {
        "type": "integer",
        "minimum": 0,
        "description": "A palette material index -- see clay_scene's 'materials'.",
    },
    "direction": _vec3_schema(
        "world-space; need not be unit length. Converted into the object's "
        "own local frame before the query runs -- see clay_scene's "
        "'rotation'/'scale' for the transform an agent read this out of."
    ),
    "max_angle": {
        "type": "number",
        "minimum": 0.0,
        "maximum": 180.0,
        "description": "Degrees off 'direction' still counted as facing it. Default 45.",
    },
    "min": _vec3_schema("metres, the box's lower corner"),
    "max": _vec3_schema("metres, the box's upper corner"),
    "space": {
        "type": "string",
        "enum": ["world", "local"],
        "description": "Which frame 'min'/'max' are given in. Default 'world'.",
    },
}

# The two query arguments that are optional -- ``max_angle`` (``_q_normal``'s
# own default) and ``space`` (this file's own default of "world", see
# ``_h_select_by``). Every other name in ``_QUERY_ARG_SCHEMAS`` is required
# whenever a query declares it.
_QUERY_OPTIONAL_ARGS = frozenset({"max_angle", "space"})


def _validate_query_arg(name: str, value: Any) -> tuple[Any, dict | None]:
    """One ``clay_select_by`` argument, validated against the fixed
    vocabulary :data:`_QUERY_ARG_SCHEMAS` describes -- the one place a query
    argument's shape is checked before it reaches a pure ``clay.select``
    function that has no JSON-schema knowledge of its own to check it with.
    """
    if name == "edge":
        if not isinstance(value, list) or len(value) != 2:
            return None, fail(
                f"{name} must be a [vertex, vertex] pair.", field=name, recovery="fix_arguments"
            )
        try:
            return [int(v) for v in value], None
        except (TypeError, ValueError):
            return None, fail(
                f"{name} must be a [vertex, vertex] pair.", field=name, recovery="fix_arguments"
            )
    if name == "face":
        try:
            return int(value), None
        except (TypeError, ValueError):
            return None, fail(
                f"{name} must be an integer.", field=name, recovery="fix_arguments"
            )
    if name == "slot":
        try:
            slot = int(value)
        except (TypeError, ValueError):
            return None, fail(
                f"{name} must be an integer.", field=name, recovery="fix_arguments"
            )
        # ``_QUERY_ARG_SCHEMAS["slot"]`` declares ``minimum: 0`` -- a palette
        # has no negative indices -- and until this line nothing here checked
        # it, so a negative slot sailed through to ``_q_material`` and matched
        # no face, a silent no-op rather than the refusal the schema promised.
        if slot < 0:
            return None, fail(
                f"{name} must be a non-negative integer.", field=name, recovery="fix_arguments"
            )
        return slot, None
    if name in ("direction", "min", "max"):
        return _validate_vec3(value, name)
    if name == "max_angle":
        # ``_QUERY_ARG_SCHEMAS["max_angle"]`` declares ``minimum: 0.0,
        # maximum: 180.0`` -- past 180 degrees off a direction nothing is
        # excluded any more -- but ``_validate_number`` alone only checks
        # finiteness, not this query's own bound.
        out, failure = _validate_number(value, name)
        if failure:
            return None, failure
        if not (0.0 <= out <= 180.0):
            return None, fail(
                f"{name} must be between 0 and 180 degrees.",
                field=name,
                recovery="fix_arguments",
            )
        return out, None
    if name == "space":
        if value not in ("world", "local"):
            return None, fail(
                "space must be 'world' or 'local'.", field="space", recovery="fix_arguments"
            )
        return value, None
    return None, fail(
        f"unknown query argument {name!r}.", field=name, recovery="fix_arguments"
    )  # pragma: no cover


# --- dispatch -----------------------------------------------------------------


@functools.cache
def _allowed_argument_names() -> dict[str, frozenset[str]]:
    """Tool name -> the frozenset of its schema's own top-level ``properties``
    keys -- what :func:`call` checks a real call's ``arguments`` against
    before any handler runs. See the module docstring's own paragraph on why
    an unknown argument is refused rather than dropped; this is the lookup
    that makes the refusal *derived* rather than a second hand-kept table.

    **Memoised, and safe to memoise for a reason worth being precise about.**
    :func:`tools` is rebuilt from scratch on every call -- 25 ``Tool`` objects
    with full description strings, cheap enough for one ``tools/list`` request
    but not for a cost paid again on *every* :func:`call`, which runs on the
    frame thread and, through ``clay_batch``, potentially several dozen times
    in one call. What makes caching *this* projection of it safe, where
    caching the whole catalogue would not be, is that a property *name* is a
    literal written directly into :func:`tools`'s own source -- ``"generator"``,
    ``"translation"``, ``"uid"`` -- and never derived from a live registry,
    while only an *enum's values* are (``bp.GENERATORS``, ``presets.ASSEMBLIES``,
    ``clay_ops.OPS``, ``bsel.QUERIES`` -- see the module docstring's opening
    paragraph). A thirteenth generator changes what
    ``tools()["clay_add_primitive"].schema["properties"]["generator"]["enum"]``
    contains; it cannot add or remove the key ``"generator"`` itself, which is
    all this cache answers questions about. CLAUDE.md's own reason for
    ``--dist loadfile`` -- preserving module-level cache couplings across a
    worker's tests rather than treating them as a hazard to avoid -- is why a
    module-level cache is the ordinary shape for something like this in this
    codebase, not a novel one. ``test_the_allowed_argument_names_are_the_
    schemas_own`` checks this reasoning against a fresh :func:`tools` call
    rather than trusting it.
    """
    return {t.name: frozenset(t.schema.get("properties", {})) for t in tools()}


def _unknown_argument_refusal(name: str, unknown: list[str], allowed: frozenset[str]) -> dict:
    """The refusal :func:`call` gives for one or more argument names a tool's
    own schema does not declare. Every offending key is named, each with a
    did-you-mean suggestion when :func:`difflib.get_close_matches` finds one
    against the tool's real property names -- there is no precedent for this
    in ``src/`` before this refusal, so the wording is kept plain rather than
    inventing a house style for it: ``"... did you mean 'translation'?"``, and
    a name with no plausible match just lists what the tool does take instead
    of guessing one. ``field`` is the first unknown key (sorted, so which one
    is deterministic) -- naming exactly one is what lets :func:`fail` derive
    ``recovery="fix_arguments"`` for free; the message beside it still lists
    every bad key, so a caller that misspelled two arguments does not need two
    round trips to learn about the second.
    """
    parts = []
    for key in sorted(unknown):
        match = difflib.get_close_matches(key, allowed, n=1, cutoff=0.6)
        if match:
            parts.append(f"{key!r} (did you mean {match[0]!r}?)")
        else:
            parts.append(f"{key!r}")
    noun = "an argument" if len(parts) == 1 else "arguments"
    legal = ", ".join(sorted(allowed)) if allowed else "none -- it takes no arguments at all"
    message = f"{name} does not take {noun} named {', '.join(parts)}. Legal arguments: {legal}."
    return fail(message, field=sorted(unknown)[0])


def call(ctx: Any, session: Session, name: str, arguments: dict) -> dict:
    """Run one tool. Never raises -- see the module docstring's safety claim
    and the class of error each of these three turns into a refusal for."""
    args = arguments or {}
    handler = _HANDLERS.get(name)
    if handler is None:
        return fail(f"no such tool: {name!r}")
    allowed = _allowed_argument_names()[name]
    unknown = [k for k in args if k not in allowed]
    if unknown:
        # A refused call runs nothing -- the house rule every handler already
        # follows for its own arguments, kept true here too: this fires
        # before the handler is ever reached, so an unknown key never even
        # gets the chance to be silently ignored the way it used to be.
        return _unknown_argument_refusal(name, unknown, allowed)
    try:
        return handler(ctx, session, args)
    except OpError as error:
        return fail(str(error))
    except ServiceError as error:
        return fail(error.message, field=error.field)
    except ValueError as error:
        return fail(str(error))
    except Exception:
        # Broad and logged, never silent -- ``tests/test_failure_paths.py``
        # scans for exactly this shape. An agent tool failing is not a crash
        # the app should show the user a traceback for; it is a refusal the
        # agent should be told about, with the detail kept in the log for
        # whoever has to work out what a library call did.
        log.exception("agent tool %r failed", name)
        return fail(f"{name} failed unexpectedly; see the log.")


def _h_scene(ctx: Any, session: Session, args: dict) -> dict:
    del args
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    objects = [_scene_row(doc, obj) for obj in doc.objects]

    boxes = [
        clay_geom_ops.world_box(obj)
        for obj in doc.objects
        if obj.visible and clay_geom_ops.world_box(obj) is not None
    ]
    bounds = None
    if boxes:
        lo = np.min([b[0] for b in boxes], axis=0)
        hi = np.max([b[1] for b in boxes], axis=0)
        bounds = {
            "min": _round(lo),
            "max": _round(hi),
            "size": _round(hi - lo),
            "center": _round((lo + hi) * 0.5),
        }

    materials = [
        {
            "index": i,
            "name": m.name,
            "color": _round(list(m.base_color_factor)),
            "metallic": _round(m.metallic_factor),
            "roughness": _round(m.roughness_factor),
        }
        for i, m in enumerate(doc.materials)
    ]

    return _json(
        {
            "objects": objects,
            "selection": sorted(doc.selection),
            "element_mode": doc.element_mode,
            "dirty": doc.dirty,
            "object_count": len(doc.objects),
            "bounds": bounds,
            "materials": materials,
        }
    )


def _h_add_primitive(ctx: Any, session: Session, args: dict) -> dict:
    """Place one primitive, with everything validated before the first
    mutation so a refused call places nothing -- see the tool's own
    description in :func:`tools` for the full argument list. Order:
    ``generator`` in the registry; ``params`` keys legal for it; the three
    TRS vectors well-formed; *then* the tab is resolved (minting one if the
    session owns none); *then* the object name (non-empty, not already
    taken) and the material index (in range) -- both of which need the
    document to answer.
    """
    generator = args.get("generator")
    if generator not in bp.GENERATORS:
        return fail(
            f"generator must be one of {', '.join(sorted(bp.GENERATORS))}.",
            field="generator",
        )

    params = args.get("params")
    if params is not None:
        if not isinstance(params, dict):
            return fail("params must be an object.", field="params")
        defaults = bp.GENERATORS[generator][0]
        unknown = sorted(set(params) - set(defaults))
        if unknown:
            return fail(
                f"unknown params {unknown} for {generator!r}; legal keys are "
                f"{sorted(defaults)}.",
                field="params",
            )
        # The schema declares each value ``number | array-of-numbers`` --
        # ``clay_set_params`` already checks every value against that shape
        # (``_validate_number_or_vec``) before it touches anything; this
        # tool never did, so a NaN or a string reached the generator
        # function directly and either poisoned a mesh's positions or, for
        # a non-numeric string, raised a bare ``TypeError`` that only
        # ``call()``'s generic backstop caught -- a logged "failed
        # unexpectedly" instead of a clean, field-named refusal. Run through
        # ``_validate_params_values`` rather than a bare loop over
        # ``_validate_number_or_vec`` so a bad value's refusal names *which*
        # key it was, not just "params" -- see that function's own docstring.
        failure = _validate_params_values(params, "params")
        if failure:
            return failure
        # ...and then each value against the *shape* that generator's own
        # default declares, which the wire schema cannot say -- see
        # ``_params_shape_refusal`` for the pyramid that crashed on a list.
        failure = _params_shape_refusal(params, defaults, "params", repr(generator))
        if failure:
            return failure

    translation = rotation_deg = scale = None
    if args.get("translation") is not None:
        translation, failure = _validate_vec3(args["translation"], "translation")
        if failure:
            return failure
    if args.get("rotation") is not None:
        rotation_deg, failure = _validate_vec3(args["rotation"], "rotation")
        if failure:
            return failure
    if args.get("scale") is not None:
        scale, failure = _validate_vec3(args["scale"], "scale")
        if failure:
            return failure

    # A boundary case surfaced by the ``changed`` audit, not missed: when this
    # session owns no document yet, ``_tab(..., create=True)`` below mints an
    # empty one and adopts it -- and the two refusals right after this can
    # still fire on that brand-new, empty document (an out-of-range
    # ``material`` needs no other object in the document to trigger). That
    # mint is deliberately not treated as ``changed=True`` here: it pushes no
    # undo step, adds no object and paints no face, so the three witnesses
    # this file's own tests use to mean "the document moved" (history length,
    # ``dirty``, object count) read identically to a document that was never
    # minted at all -- the same reasoning ``fail()``'s own docstring gives
    # for why minting a Library row is not "the document" either.
    tab, failure = _tab(ctx, session, create=True)
    if failure:
        return failure
    doc = tab.doc

    obj_name = args.get("name")
    if obj_name is not None:
        # The schema declares ``name`` a string; a bare ``str(obj_name)``
        # coercion used to accept anything stringifiable with no refusal at
        # all, the same unchecked-type hole ``clay_rename`` never had (it
        # already checks ``isinstance(name, str)`` for the identical field).
        if not isinstance(obj_name, str) or not obj_name.strip():
            return fail("name must not be empty.", field="name")
        if any(o.name == obj_name for o in doc.objects):
            return fail(f"an object is already named {obj_name!r}.", field="name")

    material_index = args.get("material")
    if material_index is not None:
        try:
            material_index = int(material_index)
        except (TypeError, ValueError):
            return fail("material must be a palette index.", field="material")
        if not (0 <= material_index < len(doc.materials)):
            return fail(
                f"material must be an index into the palette (0..{len(doc.materials) - 1}).",
                field="material",
            )

    mark = doc.history.mark()
    obj = pane_clay_tools.add_primitive(ctx, doc, generator)
    if params:
        merged = bp.clamp_params(generator, {**obj.params, **params})
        mesh = shading.auto_smooth(bp.GENERATORS[generator][1](**merged))
        was = {"params": dict(obj.params)}
        doc.set_generator_params(obj.uid, merged, mesh, was=was)
    if translation is not None or rotation_deg is not None or scale is not None:
        doc.set_transform(
            obj.uid,
            translation=translation,
            rotation=None if rotation_deg is None else _quat_from_euler_xyz(rotation_deg),
            scale=scale,
        )
    if obj_name is not None:
        doc.set_props(obj.uid, name=obj_name)
    if material_index is not None:
        _repaint(doc, [obj.uid], material_index)
    doc.history.collapse_since(mark)
    _label_top(doc, mark, f"Add {obj.name}")
    return _json(_scene_row(doc, obj))


def _h_add_figure(ctx: Any, session: Session, args: dict) -> dict:
    """Place a figure preset as one group, one undo step. See
    :func:`tools`'s description for ``translation``/``yaw``/``scale``/
    ``name_prefix``.

    Per part, with ``T`` the translation, ``s`` the uniform scale and
    ``q_y`` the yaw quaternion: ``t' = R_y(yaw) . (s . t) + T``,
    ``q' = q_y (x) q`` and ``s' = s . s_part``. Yaw and scale are applied
    to every part's *offset from the group origin*, not to each part in
    its own local frame -- a yawed figure turns where its limbs sit, it
    does not spin each limb about its own centre.
    """
    key = args.get("key")
    if key not in presets.ASSEMBLIES:
        return fail(f"key must be one of {', '.join(sorted(presets.ASSEMBLIES))}.", field="key")

    translation = None
    if args.get("translation") is not None:
        translation, failure = _validate_vec3(args["translation"], "translation")
        if failure:
            return failure

    yaw_deg = args.get("yaw")
    if yaw_deg is not None:
        try:
            yaw_deg = float(yaw_deg)
        except (TypeError, ValueError):
            return fail("yaw must be a number.", field="yaw")
        if not math.isfinite(yaw_deg):
            return fail("yaw must be finite.", field="yaw")

    scale = args.get("scale")
    if scale is not None:
        try:
            scale = float(scale)
        except (TypeError, ValueError):
            return fail("scale must be a number.", field="scale")
        if not (math.isfinite(scale) and scale > 0):
            return fail("scale must be a positive, finite number.", field="scale")

    name_prefix = args.get("name_prefix")
    # Same unchecked-type hole as ``clay_add_primitive``'s own ``name``, fixed
    # the same way: the schema declares a string, so a non-string is refused
    # rather than silently coerced.
    if name_prefix is not None and not isinstance(name_prefix, str):
        return fail("name_prefix must be a string.", field="name_prefix")

    tab, failure = _tab(ctx, session, create=True)
    if failure:
        return failure
    doc = tab.doc

    mark = doc.history.mark()
    objs = pane_clay_tools.add_assembly(ctx, doc, key)

    if name_prefix:
        # Checked *after* placement, against the names ``add_assembly`` chose
        # (already run through ``pane_clay_tools._unique_name`` for whatever
        # this document already held) rather than predicted beforehand
        # against ``presets.build``'s raw part names -- re-deriving that
        # de-duplication here to guess its answer would be a second copy of
        # it, free to drift the day it changes. A collision is undone rather
        # than left half-renamed, so a refused prefix still places nothing.
        placed = {o.uid for o in objs}
        existing = {o.name for o in doc.objects if o.uid not in placed}
        prefixed = [f"{name_prefix}{o.name}" for o in objs]
        if len(set(prefixed)) != len(prefixed) or existing & set(prefixed):
            # A mutate-then-refuse path, audited rather than missed: the
            # figure's parts were already placed by ``add_assembly`` above,
            # so this refusal fires *after* a real mutation. ``doc.undo()``
            # on the line below is what keeps ``changed`` honestly ``False``
            # here (the wrapper's default, left unoverridden) rather than a
            # gap in the audit -- it reverses the very compound step
            # ``collapse_since`` just folded, so the object count, the undo
            # history's own length and ``doc.dirty`` all read exactly as they
            # did before this call started. See ``document.py``'s ``undo()``
            # and ``UndoStack.undo()`` for why that revert is exact rather
            # than approximate: the compound edit's own ``undo`` puts back
            # the very objects it added, by uid.
            doc.history.collapse_since(mark)
            doc.undo()
            return fail(
                f"{name_prefix!r} would collide with an existing object name.",
                field="name_prefix",
            )
        for obj, new_name in zip(objs, prefixed, strict=True):
            doc.set_props(obj.uid, name=new_name)

    if translation is not None or yaw_deg is not None or scale is not None:
        yaw_quat = m3.quat_from_axis_angle(m3.vec3(0.0, 1.0, 0.0), math.radians(yaw_deg or 0.0))
        s = 1.0 if scale is None else scale
        t = m3.vec3(*translation) if translation is not None else m3.vec3()
        for obj in objs:
            new_t = m3.quat_rotate(yaw_quat, obj.translation * s) + t
            new_q = m3.quat_mul(yaw_quat, obj.rotation)
            new_s = obj.scale * s
            doc.set_transform(obj.uid, translation=new_t, rotation=new_q, scale=new_s)

    doc.history.collapse_since(mark)
    label, _builder = presets.ASSEMBLIES[key]
    _label_top(doc, mark, f"Add {label}")
    return _json({"uids": [o.uid for o in objs], "objects": [_scene_row(doc, o) for o in objs]})


def _h_add_mesh(ctx: Any, session: Session, args: dict) -> dict:
    """Place one hand-built mesh, selected, as one undo step -- the door for
    geometry an agent computed itself rather than named by recipe. See
    :func:`tools`'s description for the exact shape of ``positions``/
    ``faces``/``uv``.

    Order, the same template :func:`_h_add_primitive` sets: ``positions`` and
    ``faces`` well-formed and within :data:`MAX_MESH_VERTICES`/
    :data:`MAX_MESH_FACES`; ``uv`` (if given) nested exactly like ``faces``;
    the three TRS vectors well-formed; the mesh actually built and run
    through ``mesh.validate`` as a backstop -- *then* the tab is resolved
    (minting one if the session owns none), *then* the object name and
    material index, both of which need the document to answer. Nothing above
    that line needs a document, so nothing above it should wait for one, and
    a call that was always going to be refused should never have minted an
    empty tab just to be refused against -- see the module docstring's mint
    paragraph and :func:`_h_add_primitive`'s own comment on the identical
    boundary.

    **This object has no generator, from birth.** Every primitive keeps its
    generator's name and params until an element edit freezes them
    (``document.set_mesh``'s own docstring); a mesh handed over as raw
    coordinates has no recipe for ``clay_set_params`` to re-run, so it starts
    in exactly the state that freeze leaves an edited primitive in, rather
    than passing through it.
    """
    positions_arg = args.get("positions")
    if not isinstance(positions_arg, list) or not positions_arg:
        return fail("positions must be a non-empty array of [x, y, z].", field="positions")
    if len(positions_arg) > MAX_MESH_VERTICES:
        return fail(
            f"positions has {len(positions_arg)} entries, past the "
            f"{MAX_MESH_VERTICES:,} this tool accepts in one call.",
            field="positions",
        )
    positions: list[list[float]] = []
    for row in positions_arg:
        vec, failure = _validate_vec3(row, "positions")
        if failure:
            return failure
        positions.append(vec)
    n_positions = len(positions)

    faces_arg = args.get("faces")
    if not isinstance(faces_arg, list) or not faces_arg:
        return fail("faces must be a non-empty array of vertex-index loops.", field="faces")
    if len(faces_arg) > MAX_MESH_FACES:
        return fail(
            f"faces has {len(faces_arg)} entries, past the {MAX_MESH_FACES:,} "
            "this tool accepts in one call.",
            field="faces",
        )
    faces: list[list[int]] = []
    for fi, loop in enumerate(faces_arg):
        if not isinstance(loop, list):
            return fail(f"face {fi} must be an array of vertex indices.", field="faces")
        if len(loop) < 3:
            return fail(
                f"face {fi} has {len(loop)} corners; a face needs at least 3.", field="faces"
            )
        corners: list[int] = []
        for ci, idx in enumerate(loop):
            try:
                vi = int(idx)
            except (TypeError, ValueError):
                return fail(
                    f"face {fi} corner {ci} is {idx!r}, not a vertex index.", field="faces"
                )
            # Named down to the corner, not just the face: an agent that
            # miscounted one index in a thousand-face mesh cannot fix what
            # "a loop index is out of range" (mesh.validate's own wording)
            # does not say which of them it was.
            if not (0 <= vi < n_positions):
                return fail(
                    f"face {fi} corner {ci} indexes vertex {vi}, but positions "
                    f"has {n_positions} entries.",
                    field="faces",
                )
            corners.append(vi)
        faces.append(corners)

    uv_arg = args.get("uv")
    uv: list[list[list[float]]] | None = None
    if uv_arg is not None:
        if not isinstance(uv_arg, list):
            return fail("uv must be an array, nested exactly like faces.", field="uv")
        if len(uv_arg) != len(faces):
            return fail(
                f"uv has {len(uv_arg)} faces, but faces has {len(faces)}.", field="uv"
            )
        uv = []
        for fi, (loop, uv_loop) in enumerate(zip(faces, uv_arg, strict=True)):
            if not isinstance(uv_loop, list):
                return fail(f"uv[{fi}] must be an array of (u, v) pairs.", field="uv")
            # The single easiest mistake to make with this argument, so the
            # refusal says which face disagrees and by how much rather than
            # a bare "uv is the wrong shape".
            if len(uv_loop) != len(loop):
                return fail(
                    f"uv[{fi}] has {len(uv_loop)} corners, but face {fi} has "
                    f"{len(loop)}.",
                    field="uv",
                )
            corners_uv: list[list[float]] = []
            for ci, corner in enumerate(uv_loop):
                if not isinstance(corner, list) or len(corner) != 2:
                    return fail(f"uv[{fi}][{ci}] must be an array of 2 numbers.", field="uv")
                try:
                    u, v = float(corner[0]), float(corner[1])
                except (TypeError, ValueError):
                    return fail(
                        f"uv[{fi}][{ci}] must be an array of 2 numbers.", field="uv"
                    )
                if not (math.isfinite(u) and math.isfinite(v)):
                    return fail(f"uv[{fi}][{ci}] must be finite numbers.", field="uv")
                corners_uv.append([u, v])
            uv.append(corners_uv)

    translation = rotation_deg = scale = None
    if args.get("translation") is not None:
        translation, failure = _validate_vec3(args["translation"], "translation")
        if failure:
            return failure
    if args.get("rotation") is not None:
        rotation_deg, failure = _validate_vec3(args["rotation"], "rotation")
        if failure:
            return failure
    if args.get("scale") is not None:
        scale, failure = _validate_vec3(args["scale"], "scale")
        if failure:
            return failure

    mesh = bm.from_faces(positions, faces, uv)
    try:
        bm.validate(mesh)
    except ValueError as error:
        # Defence in depth, not the primary refusal path -- every rule
        # ``validate`` checks beyond what this handler already validated
        # above is about the CSR structure ``faces`` describes (``starts``
        # bracketing ``loops``), which is why this backstop names ``faces``
        # rather than leaving the ``ValueError`` to escape into ``call``'s
        # generic, field-blind "failed unexpectedly".
        return fail(f"not a valid mesh: {error}", field="faces")

    tab, failure = _tab(ctx, session, create=True)
    if failure:
        return failure
    doc = tab.doc

    obj_name = args.get("name")
    if obj_name is not None:
        if not isinstance(obj_name, str) or not obj_name.strip():
            return fail("name must not be empty.", field="name")
        if any(o.name == obj_name for o in doc.objects):
            return fail(f"an object is already named {obj_name!r}.", field="name")

    material_index = args.get("material")
    if material_index is not None:
        try:
            material_index = int(material_index)
        except (TypeError, ValueError):
            return fail("material must be a palette index.", field="material")
        if not (0 <= material_index < len(doc.materials)):
            return fail(
                f"material must be an index into the palette (0..{len(doc.materials) - 1}).",
                field="material",
            )

    if obj_name is None:
        # No generator name to base a default on, unlike ``add_primitive``'s
        # own ``pane_clay_tools.add_primitive`` -- "Mesh" is this door's own
        # base, counted up the same way ``ops.next_name`` already counts up
        # a duplicate.
        taken = {o.name for o in doc.objects}
        obj_name = "Mesh" if "Mesh" not in taken else clay_geom_ops.next_name("Mesh", taken)

    mark = doc.history.mark()
    obj = bd.Obj(uid=bd.new_uid(), name=obj_name, mesh=mesh, generator=None, params={})
    doc.add_object(obj)
    doc.select([obj.uid])
    if translation is not None or rotation_deg is not None or scale is not None:
        doc.set_transform(
            obj.uid,
            translation=translation,
            rotation=None if rotation_deg is None else _quat_from_euler_xyz(rotation_deg),
            scale=scale,
        )
    if material_index is not None:
        _repaint(doc, [obj.uid], material_index)
    doc.history.collapse_since(mark)
    _label_top(doc, mark, f"Add {obj.name}")

    # ``clay_diagnose.findings`` measures a mesh, not a live object, so it is
    # reused directly rather than routed back through ``_h_diagnose`` (which
    # resolves a uid, a tab and an optional ``select`` this call has no use
    # for). "Closed" is narrower than "clean": a flipped edge, a duplicate
    # face or an unused vertex is a real defect ``findings`` still reports,
    # but none of them is what stops ``clay_boolean`` -- only an open
    # boundary or a non-manifold edge does (``ops_boolean``'s own "needs
    # every selected object to be a closed solid" refusal), so those are the
    # two kinds this boolean is read from.
    rows = clay_diagnose.findings(obj.mesh)
    row = _scene_row(doc, obj)
    row["closed"] = not any(r.kind in ("hole", "nonmanifold") for r in rows)
    row["findings"] = [
        {"kind": r.kind, "label": r.label, "count": r.count, "mode": r.mode} for r in rows
    ]
    return _json(row)


def _h_transform(ctx: Any, session: Session, args: dict) -> dict:
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    obj, failure = _resolve_uid(doc, args)
    if failure:
        return failure
    translation = args.get("translation")
    rotation_deg = args.get("rotation")
    scale = args.get("scale")
    if translation is None and rotation_deg is None and scale is None:
        return fail("give at least one of translation, rotation or scale.")
    # Every vector given validated before the single mutation below -- the
    # incident this closes: an unvalidated two-element ``translation`` once
    # reached ``set_transform`` and committed, and every later ``clay_scene``
    # raised trying to broadcast it into a 3x3 matrix, bricking introspection
    # for the whole document with no recovery but a blind undo. ``rotation``
    # only looked safe by accident -- ``_quat_from_euler_xyz``'s unpack
    # raises on the wrong length -- but let a NaN straight through
    # ``math.radians`` and out the other side as a poisoned quaternion; this
    # is the same "validate everything before the first mutation" rule
    # ``_h_add_primitive`` and ``_h_add_figure`` already follow.
    if translation is not None:
        translation, failure = _validate_vec3(translation, "translation")
        if failure:
            return failure
    if rotation_deg is not None:
        rotation_deg, failure = _validate_vec3(rotation_deg, "rotation")
        if failure:
            return failure
    if scale is not None:
        scale, failure = _validate_vec3(scale, "scale")
        if failure:
            return failure
    changed = doc.set_transform(
        obj.uid,
        translation=translation,
        rotation=None if rotation_deg is None else _quat_from_euler_xyz(rotation_deg),
        scale=scale,
    )
    return _json({"uid": obj.uid, "changed": changed})


def _h_set_params(ctx: Any, session: Session, args: dict) -> dict:
    """Set one object's generator params, or -- tranche 5's plural form --
    the same params on several at once: "make the wheels larger" is one call
    against every wheel's uid, not one call per wheel and one undo step per
    wheel. A persistent "these six objects are a wheel set" object was
    argued down in design review as more machinery than the ask needed; this
    is the cheap alternative -- a caller (or a script inside one) already
    holds the uids from ``clay_scene``, so paying params once per call is
    enough.

    ``uid`` and ``uids`` are both declared as plain optional properties in
    the schema (mirroring ``clay_reference_add``'s ``job_id``/``png_base64``
    pair) with only ``params`` required -- the exactly-one rule lives here,
    not in the schema, so it can run *after* ``uids`` has already been
    checked shape-and-membership sound. That ordering is deliberate, not
    incidental: ``tests/test_agent_schemas.py`` synthesises a call for every
    declared constraint on ``uids`` (wrong type, a non-integer element, an
    empty list) by taking a valid plural baseline and violating exactly one
    of those -- which leaves ``uid`` absent in every one of those cases -- so
    checking ``uids``' own shape before asking whether both were given is
    what makes those cases refuse naming ``field="uids"`` rather than the
    exactly-one rule's ``"uid"`` swallowing a more specific refusal.

    All-or-nothing: every named object is checked -- exists, still has a
    generator (not frozen by a topology edit), and accepts every key in
    ``params`` for *its own* generator -- before any of them is rebuilt. A
    per-object try/refuse loop would have let four cylinders retune and left
    a fifth, illegal box call half-applied; a caller with no way to inspect
    the document mid-call has no use for "some of these changed".
    """
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc

    params = args.get("params")
    if not isinstance(params, dict) or not params:
        return fail("give at least one param to change.", field="params")

    uid_arg = args.get("uid")
    uids_arg = args.get("uids")
    uids: list[int] | None = None
    if uids_arg is not None:
        # Resolved (and, on ``[]``, refused) before the exactly-one check
        # below even looks at ``uid`` -- see this function's own docstring
        # on why that order is what gets the schema-walk's generated cases
        # naming ``field="uids"`` for free.
        uids, failure = _resolve_uids(doc, uids_arg, field="uids")
        if failure:
            return failure
        if not uids:
            return fail("give at least one uid.", field="uids")
    if (uid_arg is None) == (uids is None):
        return fail("give exactly one of uid or uids.", field="uid")

    if uids is None:
        obj, failure = _resolve_uid(doc, args)
        if failure:
            return failure
        uids = [obj.uid]

    objects = [doc.by_uid(u) for u in uids]
    # Whichever of the two the caller actually used, so a refusal below
    # points at an argument that is really in the call -- ``_resolve_uid``'s
    # own docstring holds itself to the same rule, and a plural call told to
    # fix its ``uid`` would be told to fix an argument it never sent.
    uid_field = "uids" if uids_arg is not None else "uid"

    # Pass 1: every object's own legality, checked in full before pass 2
    # rebuilds anything -- see the docstring's all-or-nothing paragraph.
    for obj in objects:
        if obj.generator is None:
            return fail(
                f"uid {obj.uid}: this object's topology has been edited, so "
                "it is no longer a generated shape -- there are no "
                "generator params left to set (see document.set_mesh's "
                "freeze).",
                field=uid_field,
                uids=[obj.uid],
            )
        defaults = bp.GENERATORS[obj.generator][0]
        unknown = sorted(set(params) - set(defaults))
        if unknown:
            return fail(
                f"unknown params {unknown} for {obj.generator!r} (uid "
                f"{obj.uid}); legal keys are {sorted(defaults)}.",
                field="params",
                uids=[obj.uid],
            )
        # Shape is per-generator, so unlike the finiteness sweep below this
        # cannot be hoisted out of the loop: one params dict may be aimed at
        # two objects whose generators want different shapes for the same
        # key name.
        failure = _params_shape_refusal(
            params, defaults, "params", f"{obj.generator!r} (uid {obj.uid})"
        )
        if failure:
            return failure
    # Every value validated before pass 2 touches anything -- ``bp.clamp_params``
    # only clamps the handful of keys it knows a floor or a relational limit
    # for, so a NaN or an infinity in a key it does not (or does, past the
    # clamp -- inf clamped against a finite ceiling is still inf) used to
    # sail straight through into the generator function and out the other
    # side as vertex positions, with nothing downstream ever checking a mesh
    # is made of finite numbers. Run through ``_validate_params_values``
    # rather than a bare loop over ``_validate_number_or_vec`` so the
    # refusal names *which* key was bad -- see that function's own
    # docstring. One check for every object: the params dict is the same
    # for all of them, and a value's own finiteness does not depend on which
    # generator reads it.
    failure = _validate_params_values(params, "params")
    if failure:
        return failure

    # Pass 2: nothing above can refuse anymore, so every object is rebuilt.
    # Folded into one undo step only when more than one object is
    # addressed. A single object's own ``set_generator_params`` call already
    # pushes exactly one step (it folds its own params-edit/mesh-edit pair
    # into one ``push`` -- see that method's docstring), so there is nothing
    # to fold, and wrapping it in ``mark()``/``collapse_since()`` unconditionally
    # the way ``_h_material`` does would relabel that already-one step "Set
    # Params", changing what the human's undo panel says for a call whose
    # behaviour never changed. ``_h_material`` can get away with an
    # unconditional label because it always pushes the same
    # ``add_material``/``_repaint`` pair regardless of how many uids it
    # paints; a single-uid ``clay_set_params`` has no such pair to fold.
    mark = doc.history.mark() if len(objects) > 1 else None
    rows = []
    for obj in objects:
        # Captured before anything below mutates ``obj.params`` -- the merge
        # two lines down edits it in place via a fresh dict, but
        # ``set_generator_params`` itself reassigns ``obj.params`` to the
        # very dict it is handed, so reading "before" off the object once
        # this call has run would compare a value against itself. See that
        # method's own docstring on why ``was`` is mandatory for this caller.
        was = {"params": dict(obj.params)}
        merged = bp.clamp_params(obj.generator, {**obj.params, **params})
        # ``regen.carry_over`` rather than a bare rebuild-and-``auto_smooth``:
        # this handler used to call ``shading.auto_smooth`` directly on
        # every rebuild, which re-derives shading from scratch and never
        # touched ``material`` at all -- so an object painted through
        # ``clay_material`` or given a hand-picked Shade Smooth by a prior
        # tool call came back grey and flat the moment its numbers changed
        # here. ``panes/clay_props.py``'s own rebuild carried shading (never
        # material) through the same two-case rule this module now shares
        # rather than reimplements, which is exactly how the two doors
        # built two different meshes for the same edit before this.
        mesh = regen.carry_over(
            obj.mesh, bp.GENERATORS[obj.generator][1](**merged), material=obj.material
        )
        changed = doc.set_generator_params(obj.uid, merged, mesh, was=was)
        # Reported back rather than echoed: a caller that asked for
        # segments=2 learns here that clamp_params raised it to the
        # generator's own floor.
        rows.append(
            {"uid": obj.uid, "generator": obj.generator, "params": merged, "changed": changed}
        )
    if mark is not None:
        doc.history.collapse_since(mark)
        _label_top(doc, mark, "Set Params")

    payload = {"objects": rows, "changed": any(r["changed"] for r in rows)}
    # The single-uid shape (``uid``/``generator``/``params`` at the top
    # level, no ``objects`` list) predates the plural form, and
    # ``tests/test_agent_clay.py`` -- among them
    # ``test_set_params_clamps_and_reports_the_clamped_value_back`` and
    # ``test_a_profile_param_survives_the_whole_agent_door`` -- reads
    # ``payload["params"]``/``payload["uid"]`` directly, as does whatever
    # client is already out there driving today's tool. Rather than break
    # that shape, the one-object case mirrors its row at the top level too,
    # alongside the new ``objects`` list every caller can grow into.
    if len(rows) == 1:
        payload.update(rows[0])
    return _json(payload)


def _h_material(ctx: Any, session: Session, args: dict) -> dict:
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    uids, failure = _resolve_uids(doc, args.get("uids"), field="uids")
    if failure:
        return failure
    if not uids:
        return fail("give at least one uid.", field="uids")
    color = args.get("color")
    if not isinstance(color, list) or len(color) not in (3, 4):
        return fail("color must be an array of 3 or 4 numbers, 0..1.", field="color")
    # Per component through ``_validate_unit`` rather than the old bare
    # ``isinstance(c, int | float)`` -- that check let ``float("nan")``
    # through (NaN *is* a float) straight into the palette, and
    # ``metallic``/``roughness`` had no check at all beyond the bare
    # ``float()`` conversion below. The same unvalidated-number hole
    # ``clay_transform`` had for its translation, one tool over.
    rgba = []
    for c in color:
        value, failure = _validate_unit(c, "color")
        if failure:
            return failure
        rgba.append(value)
    rgba = tuple(rgba)
    if len(rgba) == 3:
        rgba = (*rgba, 1.0)
    metallic, failure = _validate_unit(args.get("metallic", 0.0), "metallic")
    if failure:
        return failure
    roughness, failure = _validate_unit(args.get("roughness", 0.6), "roughness")
    if failure:
        return failure
    name_arg = args.get("name")
    # The schema declares ``name`` a string; a bare ``str(name_arg or "")``
    # coercion used to accept anything stringifiable with no refusal at all
    # -- the same hole ``clay_add_primitive``'s own ``name`` had, fixed the
    # same way ``clay_rename`` already checks its identical field.
    if name_arg is not None and not isinstance(name_arg, str):
        return fail("name must be a string.", field="name")
    material = gltf.Material(
        name=name_arg or "",
        base_color_factor=rgba,
        metallic_factor=metallic,
        roughness_factor=roughness,
    )

    # One material for the whole call -- never one per object -- folded into
    # one undo step the way ``add_material_and_assign`` folds its own pair,
    # so one tool call is one Ctrl+Z.
    mark = doc.history.mark()
    index = doc.add_material(material)
    _repaint(doc, uids, index)
    doc.history.collapse_since(mark)
    _label_top(doc, mark, "Set Material")
    return _json(
        {
            "index": index,
            "uids": uids,
            "color": list(rgba),
            "metallic": material.metallic_factor,
            "roughness": material.roughness_factor,
        }
    )


def _h_boolean(ctx: Any, session: Session, args: dict) -> dict:
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    # This op writes object uids straight into ``doc.selection`` a few lines
    # down -- harmless before this change, because an agent could never leave
    # object mode at all. The moment element mode is reachable that write can
    # manufacture "selected with nothing selected inside it", the state
    # ``document.py``'s module docstring says the derived-selection invariant
    # forbids in an element mode. Refused rather than auto-switched: silently
    # changing the document's mode under a call that did not ask for it is
    # the hidden state change this codebase refuses instead of guessing at.
    if doc.element_mode != "object":
        return fail(_OBJECT_SELECTION_DERIVED_REFUSAL, recovery="switch_mode")
    kind = args.get("kind")
    if kind not in ops_boolean.KINDS:
        return fail(f"kind must be one of {', '.join(ops_boolean.KINDS)}.", field="kind")
    try:
        wanted = [int(u) for u in args.get("uids") or []]
    except (TypeError, ValueError):
        return fail("uids must be a list of integers.", field="uids")
    # ``_union``'s own shape, generalised over the three kinds: the targets
    # are read in the document's own object order, so "first" means the
    # target's place in that order -- never the order this list happened to
    # name them in. See ``ops_boolean.KINDS``' own docstring for why that is
    # the rule for a difference, where the order changes the answer.
    #
    # Derived by walking ``doc.objects`` against *wanted* rather than by
    # writing ``doc.select(wanted)`` first and re-reading it: the write was
    # only ever a way to get that ordering, and doing it here put a mutation
    # ahead of the count check below -- so a boolean refused for naming too
    # few visible objects left the person's own selection overwritten by a
    # call that changed nothing else. Walking the list gives the identical
    # answer (a uid naming no object simply never matches) with nothing
    # written, which is what lets the refusal below be honest that the
    # document did not move. The selection this op does mean to leave behind
    # is set once, at the end, to the survivor.
    keep = {int(u) for u in wanted}
    targets = [obj.uid for obj in doc.objects if obj.uid in keep and obj.visible]
    if len(targets) < 2:
        return fail(
            "Select at least two visible objects.",
            field="uids",
            uids=targets,
        )
    mesh = ops_boolean.boolean([doc.by_uid(u) for u in targets], kind)
    doc.join_objects(targets[0], mesh, targets[1:])
    # clay-08 (2026-09-08 audit), the same pop ``clay_ops._join``/``_union``
    # make: the objects a boolean absorbs must not leave their manifold-check
    # cache entries pinned alive under a uid nothing owns any more.
    clay_ops._forget_manifold(ctx, targets[1:])
    doc.select([targets[0]])
    return _json({"uid": targets[0], "kind": kind})


def _h_select(ctx: Any, session: Session, args: dict) -> dict:
    """Replace the *object* selection. Refused in an element mode -- see
    :data:`_OBJECT_SELECTION_DERIVED_REFUSAL` and :func:`_h_boolean`'s own
    comment, which this shares the exact reason and the exact wording with.
    """
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    if doc.element_mode != "object":
        return fail(_OBJECT_SELECTION_DERIVED_REFUSAL, recovery="switch_mode")
    # The schema declares ``uids`` required, and ``_resolve_uids`` alone does
    # not enforce that: it treats a missing value the same as an explicit
    # empty list (``values or []``), because an empty list is this tool's own
    # "clear the selection" -- see that function's own docstring. Checked
    # for here, once, ahead of it, so *omitting* the argument entirely is
    # refused rather than silently read as the identical clearing call.
    if "uids" not in args:
        return fail(
            "give uids -- an empty list clears the selection.", field="uids"
        )
    uids, failure = _resolve_uids(doc, args.get("uids"), field="uids")
    if failure:
        return failure
    doc.select(uids)
    return _json({"selection": sorted(doc.selection)})


def _h_element_mode(ctx: Any, session: Session, args: dict) -> dict:
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    mode = args.get("mode")
    if mode not in el.MODES:
        return fail(f"mode must be one of {', '.join(el.MODES)}.", field="mode")
    doc.set_element_mode(mode)
    objects = [
        {"uid": uid, "stamp": doc.mesh_stamp(uid), "selected": _sel_counts(sel)}
        for uid, sel in doc.element_sel.items()
    ]
    return _json({"mode": doc.element_mode, "objects": objects})


def _h_select_elements(ctx: Any, session: Session, args: dict) -> dict:
    """Select vertices, edges or faces of one object by explicit index. See
    the tool's own description in :func:`tools` for the full contract --
    everything is validated against *this object's own mesh* before anything
    is switched or written, the same "validate everything before the first
    mutation" rule every other handler in this module follows.
    """
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    obj, failure = _resolve_uid(doc, args)
    if failure:
        return failure

    mode = args.get("mode")
    if mode is not None and mode not in el.MODES:
        return fail(f"mode must be one of {', '.join(el.MODES)}.", field="mode")

    how = args.get("how", "replace")
    if how not in ("replace", "add", "subtract"):
        return fail("how must be 'replace', 'add' or 'subtract'.", field="how")

    expect_stamp = args.get("expect_stamp")
    if expect_stamp is not None:
        _, failure = _check_expect_stamp(doc, obj.uid, expect_stamp)
        if failure:
            return failure

    n_verts = len(obj.mesh.positions)
    n_faces = bm.face_count(obj.mesh)

    verts_arg = args.get("verts")
    vert_arr: list[int] | None = None
    if verts_arg is not None:
        try:
            vert_arr = [int(v) for v in verts_arg]
        except (TypeError, ValueError):
            return fail("verts must be a list of integers.", field="verts")
        bad = [v for v in vert_arr if not (0 <= v < n_verts)]
        if bad:
            return fail(
                f"vertex index {bad[0]} is out of range for this mesh "
                f"(0..{n_verts - 1}).",
                field="verts",
            )

    faces_arg = args.get("faces")
    face_arr: list[int] | None = None
    if faces_arg is not None:
        try:
            face_arr = [int(f) for f in faces_arg]
        except (TypeError, ValueError):
            return fail("faces must be a list of integers.", field="faces")
        bad = [f for f in face_arr if not (0 <= f < n_faces)]
        if bad:
            return fail(
                f"face index {bad[0]} is out of range for this mesh (0..{n_faces - 1}).",
                field="faces",
            )

    edges_arg = args.get("edges")
    edge_arr: list[list[int]] | None = None
    if edges_arg is not None:
        try:
            pairs = [[int(a), int(b)] for a, b in edges_arg]
        except (TypeError, ValueError):
            return fail("edges must be a list of [vertex, vertex] pairs.", field="edges")
        if pairs:
            # ``ElementSel`` accepts any vertex pair with no complaint -- it
            # is only an overlay index buffer once it reaches the viewport --
            # so an unchecked pair would draw a line between two vertices
            # with nothing between them, and a face index past the mesh's
            # count would take the overlay build down rather than refuse
            # cleanly. Mapped through the mesh's own adjacency and refused by
            # naming the first pair that is not really an edge here instead.
            ids = adjacency(obj.mesh).edge_ids(np.asarray(pairs, dtype="i4"))
            bad_at = next((i for i, e in enumerate(ids) if e < 0), None)
            if bad_at is not None:
                return fail(f"{pairs[bad_at]} is not an edge of this mesh.", field="edges")
        edge_arr = pairs

    # Mode switches *first*, converting whatever was already selected -- and
    # only after every index above has been checked against the unchanged
    # mesh, so a refused call has touched neither the mode nor the selection.
    # The order matters for how="add": switching first is what puts the
    # prior selection into the new mode's own currency before the union
    # below runs, rather than unioning arrays that describe two different
    # element kinds.
    if mode is not None:
        doc.set_element_mode(mode)

    requested = el.ElementSel(verts=vert_arr, edges=edge_arr, faces=face_arr)
    current = doc.element_sel_of(obj.uid)
    doc.set_element_sel(obj.uid, el.combine(current, requested, how))

    return _json(
        {
            "uid": obj.uid,
            "mode": doc.element_mode,
            "stamp": doc.mesh_stamp(obj.uid),
            "selected": _sel_counts(doc.element_sel_of(obj.uid)),
        }
    )


def _check_expect_stamp(
    doc: Any, uid: int, expect_stamp: Any
) -> tuple[int | None, dict | None]:
    """*expect_stamp* as an int, or a refusal naming ``field="expect_stamp"``
    if it does not match ``doc.mesh_stamp(uid)`` right now. Shared by
    :func:`_h_select_elements` and :func:`_h_select_by`, both of which refuse
    a stale stamp before touching the mode or the selection."""
    try:
        expect_stamp = int(expect_stamp)
    except (TypeError, ValueError):
        return None, fail("expect_stamp must be an integer.", field="expect_stamp")
    current = doc.mesh_stamp(uid)
    if expect_stamp != current:
        return None, fail(
            f"expect_stamp {expect_stamp} does not match this object's "
            f"current stamp {current} -- the mesh changed since that stamp "
            "was read; call clay_elements or clay_scene to see what it is "
            "now.",
            field="expect_stamp",
            # Not the ``"fix_arguments"`` a named field defaults to: the
            # stamp the client sent was the right shape and was true when it
            # read it. What is stale is its picture of the mesh, which is
            # exactly what this message already tells it to go and re-read.
            recovery="read_scene",
        )
    return expect_stamp, None


def _h_select_by(ctx: Any, session: Session, args: dict) -> dict:
    """Select elements by a seed or a parameter, through :data:`select.
    QUERIES`. See the tool's own description in :func:`tools`, and the
    module-level ``_QUERY_ARG_SCHEMAS``/``_validate_query_arg`` for the one
    mapping from a query argument's name to what it is checked against --
    ``select.py`` itself holds no JSON-schema knowledge, by that module's own
    design.
    """
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    obj, failure = _resolve_uid(doc, args)
    if failure:
        return failure

    name = args.get("query")
    query = bsel.QUERIES.get(name)
    if query is None:
        return fail(
            f"query must be one of {', '.join(sorted(bsel.QUERIES))}.", field="query"
        )
    if doc.element_mode not in query.modes:
        # ``_in_mode_reason``'s own wording, not a paraphrase of it -- an
        # agent reading this refusal and a person reading the same query's
        # greyed-out menu row must never be told two different sentences for
        # the same gate.
        return fail(clay_ops._in_mode_reason(*query.modes)(doc), field="query")

    how = args.get("how", "replace")
    if how not in ("replace", "add", "subtract"):
        return fail("how must be 'replace', 'add' or 'subtract'.", field="how")

    expect_stamp = args.get("expect_stamp")
    if expect_stamp is not None:
        _, failure = _check_expect_stamp(doc, obj.uid, expect_stamp)
        if failure:
            return failure

    values: dict[str, Any] = {}
    for arg_name in query.args:
        if arg_name == "space":
            continue  # resolved below, never passed to a pure query function
        if arg_name not in args:
            if arg_name in _QUERY_OPTIONAL_ARGS:
                continue
            return fail(f"give a value for {arg_name!r}.", field=arg_name)
        value, failure = _validate_query_arg(arg_name, args[arg_name])
        if failure:
            return failure
        values[arg_name] = value

    if "direction" in values:
        # An agent reads translation/rotation/scale off clay_scene in world
        # space, so a direction it names ("up", "the way this object is
        # facing") is in that same frame -- and a face *normal* transforms by
        # the inverse-transpose of the object's matrix, not by its rotation
        # alone the moment the object carries a non-uniform scale. See
        # ``clay_geom_ops.local_direction``'s own docstring for the one that
        # is easy to get wrong.
        values["direction"] = clay_geom_ops.local_direction(obj, values["direction"])

    if name == "bounds":
        space = args.get("space", "world")
        if space not in ("world", "local"):
            return fail("space must be 'world' or 'local'.", field="space")
        lo = np.asarray(values["min"], dtype="f8")
        hi = np.asarray(values["max"], dtype="f8")
        # ``_q_bounds`` (this query's own ``run``) always measures against
        # the mesh's own local positions -- it has no ``positions=`` hook to
        # ask it for anything else -- so "world" is resolved here instead,
        # the way the module docstring's ``clay_select_by`` paragraph says:
        # passing ``clay_geom_ops.world_positions(obj)`` in for the mesh's
        # own local ``positions`` before the same box test ``_q_bounds`` and
        # ``select.faces_in_bounds`` already run.
        positions = clay_geom_ops.world_positions(obj) if space == "world" else None
        pts = np.asarray(obj.mesh.positions if positions is None else positions, dtype="f8")
        if len(pts) == 0:
            sel = el.empty()
        else:
            inside = np.all((pts >= lo) & (pts <= hi), axis=1)
            sel = el.ElementSel(
                verts=np.flatnonzero(inside).astype("i4"),
                faces=bsel.faces_in_bounds(obj.mesh, lo, hi, positions=positions),
            )
    else:
        sel = query.run(obj.mesh, **values)

    current = doc.element_sel_of(obj.uid)
    doc.set_element_sel(obj.uid, el.combine(current, sel, how))

    return _json(
        {
            "uid": obj.uid,
            "query": name,
            "mode": doc.element_mode,
            "stamp": doc.mesh_stamp(obj.uid),
            "selected": _sel_counts(doc.element_sel_of(obj.uid)),
        }
    )


def _h_elements(ctx: Any, session: Session, args: dict) -> dict:
    """The read side of the element-selection tools -- paged raw indices, for
    the rarer moment an agent has to reason about which ones rather than how
    many. See :data:`ELEMENT_PAGE_MAX`."""
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc

    kind = args.get("kind")
    if kind is not None and kind not in ("vertex", "edge", "face"):
        return fail("kind must be 'vertex', 'edge' or 'face'.", field="kind")

    offset = args.get("offset", 0)
    try:
        offset = int(offset)
    except (TypeError, ValueError):
        return fail("offset must be an integer.", field="offset")
    if offset < 0:
        return fail("offset must not be negative.", field="offset")

    limit = args.get("limit", ELEMENT_PAGE_DEFAULT)
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return fail("limit must be an integer.", field="limit")
    if not (1 <= limit <= ELEMENT_PAGE_MAX):
        return fail(f"limit must be between 1 and {ELEMENT_PAGE_MAX}.", field="limit")

    if args.get("uid") is not None:
        obj, failure = _resolve_uid(doc, args)
        if failure:
            return failure
        targets = [obj]
    else:
        targets = [doc.by_uid(uid) for uid in doc.element_sel]

    field_name = {"vertex": "verts", "edge": "edges", "face": "faces"}.get(kind)
    rows = []
    for obj in targets:
        sel = doc.element_sel_of(obj.uid)
        row: dict[str, Any] = {
            "uid": obj.uid,
            "stamp": doc.mesh_stamp(obj.uid),
            "counts": _sel_counts(sel),
        }
        if field_name is not None:
            arr = getattr(sel, field_name)
            row["kind"] = kind
            row["total"] = len(arr)
            row["offset"] = offset
            row["indices"] = arr[offset : offset + limit].tolist()
        rows.append(row)

    return _json({"mode": doc.element_mode, "objects": rows})


@dataclass
class _OpCtx:
    """A sandboxed stand-in for the real app ``ctx``, handed to ``clay_ops.run``.

    ``clay_ops`` reaches ``ctx`` in exactly three places (verified by reading
    the module before writing this): ``toast`` -- a module-level helper every
    refusal goes through -- and ``getattr(ctx, "clay_view", None)`` in
    ``_frame`` (Frame Selection), and ``getattr(ctx, "state", None)`` in
    ``_forget_manifold`` (the clay-08 manifold-cache pop). The absent
    ``clay_view`` is what makes Frame Selection the no-op it should always
    have been for an agent with no viewport of its own; ``state`` is passed
    through for real because the manifold-cache pop is real work that still
    has to happen. See the module docstring's ``clay_op`` paragraph for why
    the real ``ctx`` used to be handed over unsandboxed.
    """

    state: Any = None
    messages: list[str] = field(default_factory=list)

    def toast(self, message: str, level: str = "info") -> None:
        del level
        self.messages.append(message)


def _h_op(ctx: Any, session: Session, args: dict) -> dict:
    """Run one op by name. See the module docstring's ``clay_op`` paragraph
    for the sandboxed proxy, and the "counts, never raw indices" rule this
    result follows: an agent does not need a 200k-element array back from a
    ``select-all``, it needs to know that something changed and by how much.
    """
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    name = args.get("name")
    try:
        op = clay_ops.get(name)
    except KeyError:
        return fail(f"no op named {name!r}.", field="name", recovery="fix_arguments", op=name)
    # ``run`` itself returns False, silently, for a disabled op -- exactly the
    # answer that is useless to an agent with no menu to look at and read the
    # greyed row's tooltip from. Checked here, once, so the refusal names the
    # gate (``reason_for`` is only ever consulted once ``enabled`` has already
    # said no, matching ``clay_ops``'s own rule for the two never disagreeing).
    if not op.enabled(doc):
        # Not always ``recovery="switch_mode"``: ``op.reason`` covers several
        # unrelated gates (``_has_objects_reason``, ``_selection_reason``,
        # ``_has_two_visible_reason`` and the rest, see ``clay_ops.py``'s own
        # "reasons" section), and only some of them -- the ones built from
        # ``_in_mode_reason`` -- are about element mode at all. This handler
        # has no way to tell which gate fired from the string alone, so it
        # names the op rather than guessing a recovery that would be wrong
        # for "Select an object first."
        return fail(clay_ops.reason_for(op, doc), op=op.name)
    params = args.get("params")
    if params is None:
        params = {}
    elif not isinstance(params, dict):
        # The schema declares ``params`` an object; before this a non-dict
        # (a bare number, a list) reached ``clay_ops.run(proxy, doc, op,
        # **params)`` and failed there on ``**`` unpacking a non-mapping --
        # a real refusal, but the generic backstop in ``call()``'s own
        # ``except Exception``, logged as an unhandled failure rather than
        # named cleanly the way every other bad-shaped argument in this file
        # already is.
        return fail("params must be an object.", field="params")
    proxy = _OpCtx(state=getattr(ctx, "state", None))
    # Snapshotted by identity, before the op runs -- ``Mesh`` is ``eq=False``
    # and every op is ``Mesh -> Mesh`` (``document.py``'s own rule, the same
    # one ``set_mesh`` and ``mesh_stamp`` both rely on identity for), so
    # ``obj.mesh is before.get(obj.uid)`` after the call is a read of what the
    # op actually touched, not a second bookkeeping mechanism running beside
    # it. An object absent from ``before`` (an op like Duplicate makes one) is
    # "changed" too: there is no prior mesh for it to equal.
    before = {obj.uid: obj.mesh for obj in doc.objects}
    head = doc.history.head
    ran = clay_ops.run(proxy, doc, op, **params)
    changed = [
        {
            "uid": obj.uid,
            "stamp": doc.mesh_stamp(obj.uid),
            "faces": bm.face_count(obj.mesh),
            "verts": len(obj.mesh.positions),
            "selected": _sel_counts(doc.element_sel_of(obj.uid)),
        }
        for obj in doc.objects
        if before.get(obj.uid) is not obj.mesh
    ]
    return _json(
        {
            "op": op.name,
            "ran": ran,
            "pushed": doc.history.head != head,
            "element_mode": doc.element_mode,
            "messages": proxy.messages,
            "changed": changed,
        }
    )


def _parse_view_entry(entry: Any, valid_views: set[str]) -> tuple[str, dict, dict | None]:
    """One ``views[]`` entry -> ``(label, render_png kwargs, failure)``.

    A named axis/``three_quarter`` view maps straight to ``render_png``'s own
    ``view`` keyword. A free ``{yaw, pitch}`` pair maps to ``angles=(theta,
    phi)`` in radians: ``theta = radians(yaw)``, ``phi = radians(90 - pitch)``.
    That makes yaw 0 the front (``Camera.AXIS_VIEWS["front"]`` is theta=0) and
    +pitch look *down* from above (``AXIS_VIEWS["top"]`` is phi ~= 0, i.e.
    pitch +90) -- the sign that is wrong half the time and invisible until a
    picture is looked at, so it is written down here rather than trusted to
    memory.
    """
    if isinstance(entry, str):
        if entry not in valid_views:
            return (
                "",
                {},
                fail(
                    f"each view must be one of {', '.join(sorted(valid_views))} or "
                    "an object with yaw and pitch.",
                    field="views",
                ),
            )
        return entry, {"view": entry}, None
    if isinstance(entry, dict) and set(entry) == {"yaw", "pitch"}:
        try:
            yaw = float(entry["yaw"])
            pitch = float(entry["pitch"])
        except (TypeError, ValueError):
            return "", {}, fail("yaw and pitch must be numbers.", field="views")
        if not math.isfinite(yaw) or not (-89.0 <= pitch <= 89.0):
            return "", {}, fail("pitch must be between -89 and 89 degrees.", field="views")
        angles = (math.radians(yaw), math.radians(90.0 - pitch))
        return f"yaw={yaw:g},pitch={pitch:g}", {"angles": angles}, None
    return "", {}, fail("each view must be a name or an object with yaw and pitch.", field="views")


def _h_render(ctx: Any, session: Session, args: dict) -> dict:
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc

    size_arg = args.get("size")
    if size_arg is None:
        size = 1024
    else:
        try:
            size = int(size_arg)
        except (TypeError, ValueError):
            return fail("size must be an integer.", field="size")
        # Refused, not clamped: the schema declares ``minimum: 64, maximum:
        # 2048``, and silently rounding a caller's own number into range
        # answered a request that was never made with no way to tell the
        # schema had lied about the ceiling it claimed to enforce.
        if not (64 <= size <= 2048):
            return fail("size must be between 64 and 2048.", field="size")

    view = args.get("view")
    views_arg = args.get("views")
    if view is not None and views_arg is not None:
        return fail("give either view or views, not both.", field="views")

    valid_views = set(Camera.AXIS_VIEWS) | {"three_quarter"}
    if views_arg is not None:
        if not isinstance(views_arg, list) or not views_arg:
            return fail("views must be a non-empty list.", field="views")
        entries = views_arg
    elif view is not None:
        if view not in valid_views:
            return fail(f"view must be one of {', '.join(sorted(valid_views))}.", field="view")
        entries = [view]
    else:
        entries = ["three_quarter"]

    parsed: list[tuple[str, dict]] = []
    for entry in entries:
        label, kwargs, failure = _parse_view_entry(entry, valid_views)
        if failure:
            return failure
        parsed.append((label, kwargs))

    grid = args.get("grid", False)
    # The schema declares this a boolean; a bare ``bool(grid)`` coercion
    # used to accept anything (a non-empty string, say) with no refusal at
    # all -- ``bool("off")`` is ``True``, which drew the grid an agent's own
    # value looked like it was asking not to see.
    if not isinstance(grid, bool):
        return fail("grid must be a boolean.", field="grid")

    focus = args.get("focus")
    bounds = None
    if focus is not None:
        uids, failure = _resolve_uids(doc, focus, field="focus")
        if failure:
            return failure
        boxes = [
            box for box in (clay_geom_ops.world_box(doc.by_uid(u)) for u in uids) if box is not None
        ]
        if boxes:
            lo = np.min([b[0] for b in boxes], axis=0)
            hi = np.max([b[1] for b in boxes], axis=0)
            bounds = (lo, hi)

    compare = args.get("compare")
    reference = None
    compare_mode = args.get("compare_mode", "beside")
    alpha = args.get("alpha", 0.5)
    if compare is not None:
        reference = session.references.get(compare)
        if reference is None:
            return fail(f"no reference named {compare!r}.", field="compare")
        if compare_mode not in ("beside", "overlay"):
            return fail("compare_mode must be 'beside' or 'overlay'.", field="compare_mode")
        if compare_mode == "overlay":
            # The schema declares ``alpha`` a number, 0..1 -- a bare
            # ``float(alpha)`` only ever checked it converted, so a NaN, an
            # infinity, or a value past either end of the schema's own
            # declared range reached the blend with nothing having refused
            # it, the same unvalidated-number hole every other 0..1 knob in
            # this file (``clay_material``'s colour and metallic/roughness)
            # already closed with this same helper.
            alpha, failure = _validate_unit(alpha, "alpha")
            if failure:
                return failure
        if len(parsed) > 1:
            return fail("compare renders exactly one view.", field="views")
        if views_arg is None and view is None:
            # No view was asked for: default to the reference's own angle, so
            # the comparison is framed the way the picture being matched was.
            label = reference.view if reference.view in valid_views else "three_quarter"
            parsed = [(label, {"view": label})]
        size = min(size, 1024)
    else:
        total_pixels = len(parsed) * size * size
        if total_pixels > RENDER_PIXEL_BUDGET:
            return fail(
                f"{len(parsed)} view(s) at {size}x{size} would be "
                f"{total_pixels:,} pixels, over the {RENDER_PIXEL_BUDGET:,}-pixel "
                "render budget for one call -- ask for fewer or smaller views."
            )

    try:
        view_obj = _view_for(ctx)
    except Exception:
        log.exception("agent render of a Clay document failed")
        return fail("That document could not be rendered; see the log.")

    try:
        pngs = [
            view_obj.render_png(doc, size=size, grid=grid, bounds=bounds, **kwargs)
            for _label, kwargs in parsed
        ]
    except Exception:
        log.exception("agent render of a Clay document failed")
        return fail("That document could not be rendered; see the log.")

    if compare is not None:
        from . import agent_refs

        if compare_mode == "beside":
            sheet = agent_refs.beside(reference.png, pngs[0], compare, "render", size=size)
        else:
            sheet = agent_refs.overlay(reference.png, pngs[0], alpha, size=size)
        import io

        from PIL import Image

        with Image.open(io.BytesIO(sheet)) as im:
            width, height = im.width, im.height
        session.last_render_png = sheet
        return ok(
            text(
                json.dumps(
                    {
                        "view": parsed[0][0],
                        "reference": compare,
                        "mode": compare_mode,
                        "width": width,
                        "height": height,
                    }
                )
            ),
            image_png(sheet),
        )

    # base64 costs 4 bytes for every 3 of input, rounded up: the frame budget
    # is checked against what actually crosses the wire, not the raw PNG size.
    b64_total = sum(((len(png) + 2) // 3) * 4 for png in pngs)
    if b64_total > _protocol().MAX_FRAME - RENDER_FRAME_RESERVE:
        return fail(
            "This render is too large to send back in one reply frame; ask "
            "for fewer or smaller views."
        )

    session.last_render_png = pngs[0]
    header = text(json.dumps({"views": [label for label, _ in parsed], "size": size, "grid": grid}))
    # Deliberately not `_json` -- an image block has no JSON to duplicate,
    # and this header is already checked twice against `protocol.MAX_FRAME`
    # above (`RENDER_PIXEL_BUDGET`, `RENDER_FRAME_RESERVE`) before it leaves,
    # so a second copy in `structuredContent` would spend frame budget on
    # bytes nothing reads. See `_json`'s own docstring for the same claim.
    return ok(header, *(image_png(png) for png in pngs))


def _h_diagnose(ctx: Any, session: Session, args: dict) -> dict:
    """Report what is wrong with one or every visible mesh, and -- given
    ``select`` -- act on one finding the way the properties pane's own click
    handler does.

    ``clay_diagnose.Finding`` already carries the ``ElementSel`` that fixes
    each defect; before this, that was thrown away the moment it was turned
    into a JSON row, and an agent could describe a hole but never point at
    one. ``select`` closes that loop with the same three-call template
    ``panes/clay_props.py``'s ``_select_finding`` uses, for the same reason
    named there: the object selection must not be set by hand, because in an
    element mode it is *derived*, and the clear is what stops this finding's
    selection landing beside a stale one on another object.
    """
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    uid = args.get("uid")
    if uid is None:
        targets = [obj for obj in doc.objects if obj.visible]
    else:
        obj, failure = _resolve_uid(doc, args, "uid")
        if failure:
            return failure
        targets = [obj]

    reports: dict[int, list] = {}
    report = []
    for obj in targets:
        rows = clay_diagnose.findings(obj.mesh)
        reports[obj.uid] = rows
        report.append(
            {
                "uid": obj.uid,
                "name": obj.name,
                "clean": not rows,
                "findings": [
                    {"kind": row.kind, "label": row.label, "count": row.count, "mode": row.mode}
                    for row in rows
                ],
            }
        )

    select_arg = args.get("select")
    selected = None
    if select_arg is not None:
        # The schema declares this sub-object ``additionalProperties: False``
        # -- only ``uid`` and ``kind`` -- which nothing here checked before:
        # an extra key rode along unnoticed rather than being refused the way
        # the schema promises a client it will be.
        if not isinstance(select_arg, dict) or set(select_arg) - {"uid", "kind"}:
            return fail("select must be an object with only uid and kind.", field="select")
        sel_obj, failure = _resolve_uid(doc, select_arg, "uid")
        if failure:
            return failure
        kind = select_arg.get("kind")
        rows = reports.get(sel_obj.uid)
        if rows is None:
            # The object this call was asked to select in was not among this
            # call's own targets (a narrower ``uid`` was given, or it is
            # hidden) -- measured fresh rather than refused for a technicality
            # this call could answer on its own.
            rows = clay_diagnose.findings(sel_obj.mesh)
        row = next((r for r in rows if r.kind == kind), None)
        if row is None:
            available = sorted({r.kind for r in rows})
            return fail(
                f"{sel_obj.name!r} has no {kind!r} finding right now"
                + (f" -- it has {available}." if available else " -- it is clean."),
                field="select",
            )
        doc.set_element_mode(row.mode)
        doc.clear_element_sel()
        doc.set_element_sel(sel_obj.uid, row.sel)
        selected = {
            "uid": sel_obj.uid,
            "kind": row.kind,
            "mode": doc.element_mode,
            "stamp": doc.mesh_stamp(sel_obj.uid),
            "selected": _sel_counts(row.sel),
        }

    payload: dict[str, Any] = {"objects": report}
    # Document-level findings only on a whole-document call: they are about
    # how objects relate to each other, so asking them of a single named uid
    # would answer about objects the caller did not ask about.
    if uid is None:
        scene = clay_diagnose.scene_findings(list(doc.objects))
        if scene:
            payload["scene"] = [
                {"kind": row.kind, "label": row.label, "uids": list(row.uids)} for row in scene
            ]
    if selected is not None:
        payload["selected"] = selected
    return _json(payload)


def _h_analyze(ctx: Any, session: Session, args: dict) -> dict:
    """Bounds, mass properties, ground contact, symmetry and pairwise
    distance/contact/overlap -- read-only, and selects nothing.

    ``uids`` given restricts both which objects are reported on and which
    pairs are computed among them, and switches ``floating`` off entirely --
    see :func:`~.analyze.analyze`'s own docstring for why a scoped call
    cannot answer that question. Omitted, every visible object takes part
    and ``floating`` is always present in the reply, even when empty.
    """
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc

    uids_arg = args.get("uids")
    if uids_arg is None:
        targets = [obj for obj in doc.objects if obj.visible]
        pairs_among = None
    else:
        uids, failure = _resolve_uids(doc, uids_arg, field="uids")
        if failure:
            return failure
        if not uids:
            return fail("uids must name at least one object.", field="uids")
        by_uid = {obj.uid: obj for obj in doc.objects}
        targets = [by_uid[uid] for uid in uids]
        pairs_among = uids

    contact_tol, failure = _validate_range(
        args.get("contact_tol", 0.001), "contact_tol", 0.0, 1.0
    )
    if failure:
        return failure
    near, failure = _validate_range(args.get("near", 0.05), "near", 0.0, 10.0)
    if failure:
        return failure
    symmetry_tol, failure = _validate_range(
        args.get("symmetry_tol", 0.002), "symmetry_tol", 0.0, 1.0
    )
    if failure:
        return failure

    result = clay_analyze.analyze(
        targets,
        pairs_among=pairs_among,
        contact_tol=contact_tol,
        near=near,
        symmetry_tol=symmetry_tol,
    )

    objects_out = [
        {
            "uid": row.uid,
            "name": row.name,
            "bounds": None
            if row.bounds is None
            else {"min": _round(row.bounds[0]), "max": _round(row.bounds[1])},
            "area": _round(row.area),
            "volume": None if row.volume is None else _round(row.volume),
            "closed": row.closed,
            "components": row.components,
            "ground": None
            if row.ground is None
            else {
                "min_y": _round(row.ground.min_y),
                "contact": row.ground.contact,
                "penetration": _round(row.ground.penetration),
            },
            "symmetry": _round(list(row.symmetry)),
        }
        for row in result.objects
    ]

    pairs_out = [
        {
            "uids": list(pair.uids),
            "distance": None if pair.distance is None else _round(pair.distance),
            "intersects": pair.intersects,
            "contact": pair.contact,
            "overlap": None
            if pair.overlap is None
            else {"volume": _round(pair.overlap.volume), "depth": _round(pair.overlap.depth)},
            "exact": pair.exact,
        }
        for pair in result.pairs
    ]

    payload: dict[str, Any] = {
        "objects": objects_out,
        "pairs": pairs_out,
        "tolerances": {"contact_tol": contact_tol, "near": near, "symmetry_tol": symmetry_tol},
    }
    if result.floating is not None:
        payload["floating"] = list(result.floating)
    if result.truncated:
        payload["truncated"] = True
    return _json(payload)


def _h_export(ctx: Any, session: Session, args: dict) -> dict:
    del args
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    if tab.saving:
        return fail("A save for this document is already in progress.")
    if not any(obj.visible for obj in doc.objects):
        return fail("There is nothing visible to export.")

    # ``clay_mode.camera_of`` reads *whatever tab the interactive viewport is
    # currently showing*, which is never this one -- an agent's document is
    # never on screen by definition. Passing ``tab.view`` straight through
    # avoids stamping the agent's document with the user's current camera.
    #
    # The chain itself -- ``to_model``, the GLB write, ``import_mesh``, and the
    # ``.wblk`` source sidecar (``save_clay_source``) -- is
    # ``clay_mode.build_asset`` now; see its docstring for why both this call
    # and ``export_asset``'s go through it, and the module docstring's first
    # departure from ``save_to`` for why that sidecar is the only copy this
    # module ever needs -- there is no second one to keep of its own. There is
    # no frame boundary an MCP call can hand the encode across the way
    # ``export_asset`` hands it to a task thread, so it all runs right here,
    # synchronously, before this handler returns -- a deliberate one-shot
    # cost, not the per-frame stall the task-thread split exists to prevent.
    job_id = clay_mode.build_asset(ctx.svc, doc, title=tab.title, view=tab.view)

    tab.job_id = job_id
    ctx.cache.invalidate()
    return _json({"job_id": job_id})


def _move_history(ctx: Any, session: Session, args: dict, *, redo: bool) -> dict:
    """The shared body of ``clay_undo``/``clay_redo``. See ``tools()`` for the
    documented "moves the head, pushes nothing" exception both belong to."""
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    steps = args.get("steps", 1)
    try:
        steps = int(steps)
    except (TypeError, ValueError):
        return fail("steps must be an integer.", field="steps")
    if not (1 <= steps <= 64):
        return fail("steps must be between 1 and 64.", field="steps")

    step_fn = doc.redo if redo else doc.undo
    moved = 0
    for _ in range(steps):
        if not step_fn():
            break
        moved += 1

    # ``ObjectAddEdit.undo``/``ObjectRemoveEdit.redo`` already discard their
    # own uid from ``doc.selection`` when they run (read them) -- but that
    # self-pruning lives on those two edit types alone, and a compound step
    # can bundle them with edit types that carry no such rule (``join_objects``'s
    # own ``MeshEdit``/``ObjectPropsEdit`` siblings, say). Pruned here, once,
    # after every move, rather than trusted to every current and future
    # ``Edit.undo``/``redo`` to have covered it.
    known = {obj.uid for obj in doc.objects}
    doc.selection = {uid for uid in doc.selection if uid in known}

    return _json(
        {
            "moved": moved,
            "done_steps": len(doc.history),
            "can_undo": doc.history.can_undo,
            "can_redo": doc.history.can_redo,
        }
    )


def _h_undo(ctx: Any, session: Session, args: dict) -> dict:
    return _move_history(ctx, session, args, redo=False)


def _h_redo(ctx: Any, session: Session, args: dict) -> dict:
    return _move_history(ctx, session, args, redo=True)


def _h_delete(ctx: Any, session: Session, args: dict) -> dict:
    """Remove objects by uid, directly -- never through ``clay_op``'s own
    Delete row.

    ``clay_op`` ``delete`` acts on the document's *selection*, and in a face
    or edge mode ``selection.delete_selected`` never removes an object at
    all -- so an agent that had switched element mode (perhaps from an
    earlier ``clay_op`` call) would see a silent no-op where it asked for a
    deletion. Working from the uids given, in whatever element mode the
    document happens to be in, is what keeps "delete these objects" meaning
    that regardless.

    **Deliberately not given the same element-mode refusal as ``clay_select``
    and ``clay_boolean``.** Those two *write* object uids straight into
    ``doc.selection``, which in an element mode can manufacture "selected
    with nothing selected inside it" -- the state the derived-selection
    invariant forbids. This handler never does: :meth:`~.document.ClayDoc.
    remove_object` only ever *removes* a uid from ``selection`` (and from
    ``element_sel``, on the same line), and removing an entry from a set
    cannot put it into the forbidden state that only a write can create.
    Refusing here would be refusing a call that was never capable of the
    defect the refusal exists to prevent.
    """
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    uids, failure = _resolve_uids(doc, args.get("uids"), field="uids")
    if failure:
        return failure
    if not uids:
        return fail("give at least one uid.", field="uids")

    mark = doc.history.mark()
    for uid in uids:
        doc.remove_object(uid)
    doc.history.collapse_since(mark)
    _label_top(doc, mark, "Delete")
    # clay-08 (2026-09-08 audit): an object that leaves ``doc.objects`` must
    # not leave its manifold-check cache entry pinning a whole ``Mesh`` alive
    # under a uid nobody owns -- the same pop ``clay_ops._join``/``_union``
    # make when they absorb objects, here for the direct-delete path
    # ``clay_op``'s own Delete row does not take.
    clay_ops._forget_manifold(ctx, uids)
    return _json({"deleted": uids})


def _h_rename(ctx: Any, session: Session, args: dict) -> dict:
    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc
    obj, failure = _resolve_uid(doc, args)
    if failure:
        return failure
    name = args.get("name")
    if not isinstance(name, str) or not name.strip():
        return fail("name must not be empty.", field="name")
    if any(other.uid != obj.uid and other.name == name for other in doc.objects):
        return fail(f"an object is already named {name!r}.", field="name")
    doc.set_props(obj.uid, name=name)
    return _json({"uid": obj.uid, "name": name})


def _resolve_batch_ref(doc: Any, value: Any, field: str) -> tuple[Any, dict | None]:
    """Walk *value* (one batch-entry argument, in full -- a plain scalar, or
    a dict/list nested arbitrarily deep) and replace every ``{"$ref": name}``
    found anywhere inside it with the uid of *doc*'s object named *name*, as
    *doc* stands right now. Returns a fresh copy; *value* itself is never
    mutated, so a refusal partway through a list leaves the caller's own
    ``entry["arguments"]`` exactly as it sent it.

    ``field`` is always the *top-level* argument key this value hangs off of
    in the entry's ``arguments`` -- passed down unchanged through every
    recursive call, so ``{"uids": [1, {"$ref": "b"}]}``'s ambiguous ``b``
    still refuses naming ``field="uids"`` rather than some deeper path
    nothing else in this file has a name for. That is also why this is
    ``_h_batch``'s own helper and not a general tree-walker: "the top-level
    argument" is a batch-entry concept, meaningless for any other caller.

    Only a dict of the *exact* shape ``{"$ref": <name>}`` is treated as a
    reference -- one that also carries any other key is refused rather than
    guessed at (which key wins?), and ``tests/test_agent_clay.py``'s
    ``test_a_dict_carrying_ref_beside_another_key_is_refused`` pins that. A
    dict with no ``$ref`` key at all -- an ordinary object argument, or one
    that merely nests a real ``$ref`` somewhere inside it -- is walked key by
    key instead.
    """
    if isinstance(value, dict):
        if "$ref" in value:
            if len(value) != 1:
                return None, fail(
                    f"a $ref object may carry no other key; got {sorted(value)}.",
                    field=field,
                )
            name = value["$ref"]
            if not isinstance(name, str) or not name:
                return None, fail(
                    "$ref must be a non-empty string naming an object by name.",
                    field=field,
                )
            matches = [obj.uid for obj in doc.objects if obj.name == name]
            if not matches:
                return None, fail(f"no object named {name!r}.", field=field, recovery="read_scene")
            if len(matches) > 1:
                # Names are unique at this door's own creation tools
                # (clay_add_primitive/clay_add_figure/clay_add_mesh each
                # refuse a collision) but not globally -- clay_rename's own
                # lower-level door, document.set_props, carries no such
                # check, so a document reached by other means (the human
                # panel, clay_duplicate) can genuinely hold two objects
                # wearing one name. Picking the first would silently act on
                # the wrong one; naming both is the only honest answer.
                return None, fail(
                    f"{len(matches)} objects are named {name!r}; give a uid "
                    f"instead of $ref (uids {matches}).",
                    field=field,
                    uids=matches,
                )
            return matches[0], None
        out: dict[str, Any] = {}
        for key, sub_value in value.items():
            resolved, failure = _resolve_batch_ref(doc, sub_value, field)
            if failure:
                return None, failure
            out[key] = resolved
        return out, None
    if isinstance(value, list):
        out_list: list[Any] = []
        for item in value:
            resolved, failure = _resolve_batch_ref(doc, item, field)
            if failure:
                return None, failure
            out_list.append(resolved)
        return out_list, None
    return value, None


def _h_batch(ctx: Any, session: Session, args: dict) -> dict:
    """Run several tools as one undo step. See the module docstring's own
    paragraph on the fold and :data:`BATCH_EXCLUDED` for what this refuses to
    run and why.

    The whole list's shape is validated before anything runs, so a malformed
    batch runs nothing. If the session owns no tab yet, this refuses unless
    the *first* call is ``clay_add_primitive``, ``clay_add_figure`` or
    ``clay_add_mesh``, in which case it mints one through
    ``_tab(..., create=True)`` itself -- ``_h_batch`` needs a document in
    hand before the loop starts (to open the ``history.mark()`` the whole
    run folds into), so the mint has to happen here rather than be left to
    the first sub-call, but it is still one of the three creator tools that
    is about to run, which is what keeps "only those three mint a document"
    true.

    ``$ref``: a value of the exact form ``{"$ref": "<object name>"}``
    appearing anywhere inside an entry's ``arguments`` is replaced, the
    moment that entry runs, with the uid of the object of that name in this
    document *as it then stands* -- see :func:`_resolve_batch_ref`. This is
    what lets a later entry act on an object an earlier entry in the same
    batch just created: a batch's own results are invisible to the batch
    itself until the whole thing returns, so without this the only way to
    build a hub and then act on it was two batches with a ``clay_scene``
    read in between. Deliberately resolved here, per entry, rather than
    up front against the whole ``calls`` list: the up-front validation above
    only checks shape (an entry is an object, its name is batchable, its
    arguments are a dict-or-absent) precisely because none of it can know
    what a name resolves to before earlier entries have actually run, and an
    unresolvable ``$ref`` is refused *at the entry that carries it* --
    exactly like a bad ``uid`` in that same entry already is -- rather than
    given a second, pre-flight contract of its own. And deliberately *not*
    wired into :func:`call`: outside a batch an agent already holds the
    creating call's own result, uid included, so a ``$ref`` there would
    solve nothing that a uid does not already solve, and the only thing
    resolving it there would buy is a second place this file has to explain
    what ``$ref`` means. A ``$ref`` handed to an ordinary, non-batched call
    is refused as the malformed ``uid`` it is -- ``test_a_ref_in_an_
    ordinary_non_batched_call_is_not_resolved`` pins that boundary so a
    later reader does not "finish the job" by moving resolution down into
    ``call()``.

    ``rollback_on_error``: the stop-at-first-refusal-and-keep-the-prefix
    contract above is unchanged and this argument does not touch it -- what
    changes is what happens to that kept prefix once the batch has already
    stopped. Default false leaves today's behaviour exactly alone. True
    reverses the folded step with ``doc.history.undo(doc, redoable=False)``
    -- see the module docstring's own paragraph on this argument for why
    ``redoable=False`` and for what a rollback does and does not reach.
    """
    calls = args.get("calls")
    if not isinstance(calls, list) or not (1 <= len(calls) <= BATCH_MAX):
        return fail(f"calls must be a list of 1 to {BATCH_MAX} tool calls.", field="calls")
    rollback_on_error = args.get("rollback_on_error", False)
    # The schema declares this a boolean; checked the same way ``clay_render``'s
    # own ``grid`` already is (see that handler) rather than coerced with a
    # bare ``bool(...)``, which would have accepted any truthy value with no
    # refusal at all and silently decided an agent's typo meant "yes, roll
    # back my work".
    if not isinstance(rollback_on_error, bool):
        return fail("rollback_on_error must be a boolean.", field="rollback_on_error")
    allowed = set(_HANDLERS) - BATCH_EXCLUDED
    for entry in calls:
        if not isinstance(entry, dict):
            return fail("every call must be an object with a name.", field="calls")
        # The schema declares each entry ``additionalProperties: False`` --
        # only ``name`` and ``arguments`` -- which nothing here checked before:
        # a typo'd sibling key (``argumets``, say) rode along silently instead
        # of being refused, leaving the intended ``arguments`` unset and the
        # call it was meant to carry run with none at all.
        extra = set(entry) - {"name", "arguments"}
        if extra:
            return fail(
                f"unknown keys in a batch call entry: {sorted(extra)}.", field="calls"
            )
        name = entry.get("name")
        if name not in allowed:
            return fail(f"{name!r} is not a batchable tool.", field="calls")
        arguments = entry.get("arguments")
        if arguments is not None and not isinstance(arguments, dict):
            return fail("each call's arguments must be an object.", field="calls")

    if not session.tab_uid:
        first_name = calls[0].get("name")
        if first_name not in MINTS_A_DOCUMENT:
            return fail(
                "This session has no document yet. The first call in a "
                "batch that starts one must be clay_add_primitive, "
                "clay_add_figure or clay_add_mesh.",
                recovery="start_document",
            )
        _, failure = _tab(ctx, session, create=True)
        if failure:
            return failure

    tab, failure = _tab(ctx, session)
    if failure:
        return failure
    doc = tab.doc

    mark = doc.history.mark()
    results: list[dict] = []
    stopped_at: int | None = None
    for i, entry in enumerate(calls):
        arguments = entry.get("arguments") or {}
        # Resolved fresh against *doc* on every entry, not once up front --
        # see this function's own docstring's ``$ref`` paragraph for why an
        # unresolvable name refuses here rather than before the loop starts.
        resolved: dict[str, Any] = {}
        ref_failure: dict | None = None
        for key, value in arguments.items():
            resolved[key], ref_failure = _resolve_batch_ref(doc, value, key)
            if ref_failure:
                break
        result = ref_failure if ref_failure else call(ctx, session, entry["name"], resolved)
        results.append(result)
        if result.get("isError"):
            stopped_at = i
            break
    doc.history.collapse_since(mark)

    rolled_back = False
    # Keyed off ``doc.history.head != mark``, never off ``collapse_since``'s
    # own return value: that return is a folding decision -- ``False`` for a
    # run that pushed exactly one step, because wrapping a lone edit in a
    # ``CompoundEdit`` would read as "compound" in the history panel where
    # the edit already reads as what it did -- not a "did anything happen"
    # signal. A single pushed step is still the right thing to undo, and
    # ``head != mark`` answers "did the document move" the same way whether
    # collapsing found one step or several to fold.
    if rollback_on_error and stopped_at is not None and doc.history.head != mark:
        # ``redoable=False``: this batch's whole point is that the agent
        # wants the partial work to never have existed. ``redoable=True``
        # (the default ``undo()`` a human's Ctrl+Z takes) would leave the
        # abandoned attempt sitting on the redo stack, where a later
        # ``clay_redo`` would bring back exactly the work this call was
        # asked to erase -- the same cancelled-lift shape ``UndoStack.undo``'s
        # own docstring describes: the buffer needs putting back, but the
        # user asked for the lift to not have happened, so redoable=True
        # would let Ctrl+Y replay it. Only the document's own undo stack is
        # unwound here -- a tab this batch minted still exists (that mint
        # pushed no undo step to begin with, so it sits before ``mark`` and
        # is untouched), and neither does an element-mode/selection change
        # or a reference add along the way, because neither ever pushed a
        # step either.
        doc.history.undo(doc, redoable=False)
        rolled_back = True

    # Skipped once rolled back: undoing the folded step already put
    # ``doc.history.head`` back at ``mark``, so there is no step left on top
    # to (mis)label -- ``_label_top`` would no-op on its own guard here too,
    # but this says so rather than relying on that guard to be read.
    if not rolled_back:
        _label_top(doc, mark, "Agent batch")

    # "completed" is diagnostic and unaffected by rollback: how many calls
    # succeeded before the refusal fired stays true regardless of whether
    # that work was then reversed, so "completed: 2, rolled_back: true" is
    # not a contradiction -- one reports what ran, the other what remains.
    completed = len(results) - (1 if stopped_at is not None else 0)
    # Truthfully computed, not hard-coded: ``mark`` is the head serial before
    # the loop above ran anything, so a head that has moved past it means at
    # least one sub-call genuinely pushed a step -- exactly what "did the
    # document move" asks, whether the batch ran to completion, stopped at
    # its first refusal with a successful prefix already folded in, or (once
    # a rollback above has run) landed back at ``mark`` by construction. The
    # same expression answers all three rather than a rollback branch hand-
    # setting ``changed`` to ``False``.
    changed = doc.history.head != mark
    payload = {
        "completed": completed,
        "stopped_at": stopped_at,
        "changed": changed,
        # Always present, the same reasoning ``changed`` is always present
        # for: a client should be able to branch on this key without first
        # checking whether it exists.
        "rolled_back": rolled_back,
        "results": results,
    }
    # Routed through the same encode-then-decode ``_json`` uses, rather than
    # handing *payload* to ``structured=`` as-is: ``results`` is a list of
    # whole tool results, each already built by ``ok()``/``fail()``/``_json``
    # -- so its own ``structuredContent`` (or ``fail``'s flat extras) is
    # already plain-JSON, and today nothing this handler adds on top
    # (``completed``, ``stopped_at``) is anything but a plain int or ``None``
    # either. Doing the round trip anyway is what keeps that true by
    # construction rather than by audit: a future field on *this* payload
    # that was not itself JSON-native would otherwise reach
    # ``structuredContent`` unrounded while the text block beside it had
    # already been normalised by ``json.dumps`` -- the same "same JSON, not
    # merely comparable" guarantee ``_json``'s own docstring keeps, applied
    # by hand here because this is the one JSON payload in the file built
    # without going through ``_json`` itself.
    encoded = json.dumps(payload)
    result = ok(text(encoded), structured=json.loads(encoded))
    # Set by hand rather than through ``fail()``: a batch that stopped early
    # is a failure the agent must notice, but the payload it needs in order
    # to recover -- the successful prefix, and the failing call's own message
    # -- is a JSON result block, and ``fail()`` can only carry a message plus
    # flat ``structuredContent``, not both of those. Because this bypasses
    # ``_json``, its ``structuredContent`` duplication is not inherited for
    # free the way every other tool's is -- it is passed explicitly above,
    # which is also why this is the one JSON-answering tool that would have
    # been left without a structured twin had this call not been written out.
    result["isError"] = stopped_at is not None
    return result


def _h_reference_add(ctx: Any, session: Session, args: dict) -> dict:
    """Hand this session a picture from a Library job or inline base64. See
    the module docstring's references paragraph and :func:`tools`'s
    description for the full contract."""
    from . import agent_refs

    name = args.get("name")
    if not isinstance(name, str) or not name.strip():
        return fail("name must not be empty.", field="name")

    job_id = args.get("job_id")
    png_b64 = args.get("png_base64")
    if (job_id is None) == (png_b64 is None):
        return fail("give exactly one of job_id or png_base64.", field="job_id")

    view = args.get("view", "other")
    valid_views = set(Camera.AXIS_VIEWS) | {"three_quarter", "other"}
    if view not in valid_views:
        return fail(f"view must be one of {', '.join(sorted(valid_views))}.", field="view")

    if job_id is not None:
        # The schema declares this a string; unchecked, a non-string reached
        # ``service.validation.check_job_id``'s own regex match and raised a
        # bare ``TypeError`` there, caught only by ``call()``'s generic
        # backstop rather than refused by name the way a job id this
        # document simply does not have already is.
        if not isinstance(job_id, str):
            return fail("job_id must be a string.", field="job_id")
        try:
            job = ctx.svc.require_job(job_id)
        except NotFound as error:
            return fail(error.message, field="job_id")
        job_dir = ctx.svc.job_dir(job_id)
        file_arg = args.get("file")
        # The schema declares this a string; unchecked, a non-string reached
        # ``job_dir / name`` inside ``svc_files.ready`` and raised a bare
        # ``TypeError`` there, caught only by ``call()``'s generic backstop.
        if file_arg is not None and not isinstance(file_arg, str):
            return fail("file must be a string.", field="file")
        candidates = (
            [file_arg] if file_arg else ["input.png", "ref.png", "reference.png", "thumb.png"]
        )
        chosen = next((c for c in candidates if svc_files.ready(job, job_dir, c)), None)
        if chosen is None:
            return fail(
                "no reference image is ready for this job; looked for "
                + ", ".join(candidates)
                + ".",
                field="job_id",
            )
        path = svc_files.job_dir_file(ctx.svc, job_id, chosen)
        data = path.read_bytes()
        source = f"job:{job_id}:{chosen}"
    else:
        import base64
        import binascii

        try:
            data = base64.b64decode(png_b64, validate=True)
        except (binascii.Error, ValueError, TypeError):
            # TypeError joins the two decoding errors here rather than a
            # separate isinstance check up front: b64decode raises it for
            # anything that is not str/bytes-like (an int, say), and the
            # schema's declared "type": "string" is exactly the same
            # "this was never valid base64" refusal from the caller's side.
            return fail("png_base64 must be valid base64.", field="png_base64")
        if len(data) > svc_validation.MAX_UPLOAD_BYTES:
            # Belt and braces: the 8 MiB protocol frame this call arrived over
            # already bounds an inline upload's size, so this should never
            # actually be reachable -- but the ceiling is worth naming in
            # case that frame limit ever moves.
            return fail(
                f"png_base64 decodes to more than {svc_validation.MAX_UPLOAD_BYTES:,} bytes.",
                field="png_base64",
            )
        source = "inline"

    try:
        png, width, height = agent_refs.normalise(data)
    except svc_files.ImageTooLarge as error:
        return fail(str(error), field="job_id" if job_id is not None else "png_base64")

    replaced = name in session.references
    if not replaced and len(session.references) >= MAX_REFERENCES:
        return fail(
            f"this session already holds {MAX_REFERENCES} references, its "
            "cap; remove one with clay_reference_remove first.",
            field="name",
        )

    session.references[name] = agent_refs.Reference(
        name=name, png=png, width=width, height=height, view=view, source=source
    )
    # The whole of what the human at the keyboard is shown this round -- see
    # the module docstring for why that is a deliberate scope line, not an
    # oversight.
    ctx.toast(f"Agent reference {name!r} added.")
    return _json(
        {
            "name": name,
            "width": width,
            "height": height,
            "view": view,
            "source": source,
            "replaced": replaced,
        }
    )


def _h_reference_list(ctx: Any, session: Session, args: dict) -> dict:
    del ctx, args
    refs = sorted(session.references.values(), key=lambda r: r.name)
    return _json(
        {
            "references": [
                {
                    "name": r.name,
                    "width": r.width,
                    "height": r.height,
                    "view": r.view,
                    "source": r.source,
                }
                for r in refs
            ],
            "max": MAX_REFERENCES,
        }
    )


def _h_reference_get(ctx: Any, session: Session, args: dict) -> dict:
    from . import agent_refs

    del ctx
    name = args.get("name")
    ref = session.references.get(name)
    if ref is None:
        return fail(f"no reference named {name!r}.", field="name")
    meta = {
        "name": ref.name,
        "width": ref.width,
        "height": ref.height,
        "view": ref.view,
        "source": ref.source,
    }
    # Deliberately not `_json` -- this reply carries a picture, the same
    # image-carrying exclusion `clay_render` is answered with (see `_json`'s
    # own docstring): an image block has no JSON to duplicate, so `meta`
    # exists only as text here, never a second time as `structuredContent`.
    # This is the second, not the only, place that rule applies.
    return ok(text(json.dumps(meta)), image_png(agent_refs.bounded_png(ref.png)))


def _h_reference_remove(ctx: Any, session: Session, args: dict) -> dict:
    del ctx
    name = args.get("name")
    if name not in session.references:
        return fail(f"no reference named {name!r}.", field="name")
    del session.references[name]
    return _json({"removed": name})


_HANDLERS = {
    "clay_scene": _h_scene,
    "clay_add_primitive": _h_add_primitive,
    "clay_add_figure": _h_add_figure,
    "clay_add_mesh": _h_add_mesh,
    "clay_transform": _h_transform,
    "clay_set_params": _h_set_params,
    "clay_material": _h_material,
    "clay_boolean": _h_boolean,
    "clay_select": _h_select,
    "clay_element_mode": _h_element_mode,
    "clay_select_elements": _h_select_elements,
    "clay_select_by": _h_select_by,
    "clay_elements": _h_elements,
    "clay_op": _h_op,
    "clay_render": _h_render,
    "clay_diagnose": _h_diagnose,
    "clay_analyze": _h_analyze,
    "clay_export": _h_export,
    "clay_undo": _h_undo,
    "clay_redo": _h_redo,
    "clay_delete": _h_delete,
    "clay_rename": _h_rename,
    "clay_batch": _h_batch,
    "clay_reference_add": _h_reference_add,
    "clay_reference_list": _h_reference_list,
    "clay_reference_get": _h_reference_get,
    "clay_reference_remove": _h_reference_remove,
}
