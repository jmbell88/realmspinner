"""The tool surface an MCP agent drives Clay through -- the dispatch core.

**Split across five files in the P4 restructure** (``dev/RESTRUCTURE.md``:
"``studio/modes/clay/agent/dispatch.py`` (5,482) -> ``modes/clay/agent/{schema,validate,handlers,
program}.py``"). ``studio/modes/`` does not exist on disk yet -- P5 creates
it and folds Clay into it, taking this split with it -- so for now the five
land as siblings right here in ``studio/``, keeping the name every external
caller and the 15-odd test files that already say ``warlock.studio.modes.clay.agent.dispatch``
import:

* **This file** keeps the module name, the module docstring, and Clay's own
  dispatch machinery: :class:`Session`'s resolution into a live document
  (re-exported from ``agent_clay_validate``, see below), the ``_HANDLERS``
  table, :func:`call` (the one door every tool call passes through), the
  unknown-argument check that guards it, and the two functions that publish
  the tool surface itself, :func:`tools` and :func:`instructions` -- kept
  here rather than in the schema module because both read ``_HANDLERS``
  straight off this module's own dispatch table (``tools()``'s ``batch_names``
  is ``set(_HANDLERS) - BATCH_EXCLUDED``), and a schema-vocabulary module
  that had to import this one's dispatch state back would be the exact
  cycle the split exists to avoid.
* **``studio/modes/clay/agent/schema.py``** is every JSON-schema fragment and prose
  catalogue :func:`tools`/:func:`instructions` compose from, plus the
  numeric ceilings a tool's schema and its handler both read (``BATCH_MAX``,
  ``MAX_MESH_VERTICES`` and the rest) -- see that module's own docstring for
  why those particular constants count as "schema vocabulary" rather than
  dispatch state.
* **``studio/modes/clay/agent/validate.py``** is the true leaf of the split: the MCP
  result envelope (``ok``/``fail``/``text``/``image_png``/``_json``),
  :class:`Session` and :func:`_tab` (the session/document resolver), the
  Euler helper a rotation argument needs, and every shared "check this
  argument" validator. It imports no sibling of this fold, which is what
  lets every handler file *and* this one import it with no risk of a cycle.
* **``studio/modes/clay/agent/tools.py``**, **``studio/modes/clay/agent/tools_ops.py``** and
  **``studio/modes/clay/agent/tools_batch.py``** are the handler families: the ten
  object-level tools (create/move/reshape/paint/delete a whole object), the
  ten selection/element-op/render/inspection tools, and the eight
  batch/program/history/reference tools, respectively -- see
  ``studio/modes/clay/agent/tools.py``'s own docstring for why the split lands there
  rather than along the brief's original scene/selection/ops guess (a
  banner-name mismatch in the file this split started from), and
  ``studio/modes/clay/agent/tools_ops.py``'s for the one place a handler reaches back
  into *this* module through a lazy, function-scope accessor rather than a
  plain import.

**The tool list is derived, never hand-written**, and that discipline
crosses every one of those five files rather than living in just one of
them. Every schema in :func:`tools` is built from a registry that already
exists for a human surface -- ``primitives.GENERATORS`` for what a
primitive is and what it defaults to, ``presets.ASSEMBLIES`` for which
figures exist, ``clay_ops.OPS`` for the whole of what an object or an
element can be told to do, and ``select.QUERIES`` for the handful of
selection verbs that answer from a seed or a parameter rather than acting on
what is already selected. A thirteenth primitive, a ninth figure, a new op
or a seventh query in its registry needs no edit here: it shows up in the
next ``tools()`` call because the source it is drawn from changed, which is
the same property ``clay_ops.menu`` already gives the context menu, the
tools pane and the key handler -- one list, so nothing here can drift out of
step with what Clay can actually do. Writing a query enum out by hand would
have been a *fifth* place to remember one exists, beside those same three
surfaces and ``OPS`` itself. ``clay_batch``'s own name enum is derived the
same way, from ``_HANDLERS`` minus ``BATCH_EXCLUDED`` -- see that constant's
own docstring (``studio/modes/clay/agent/schema.py``) for which tools are left out and why.

**An agent may take a mesh apart the way a person can, once it has a
selection to work from.** ``clay_element_mode`` switches vertex/edge/face
mode (and back to object mode); ``clay_select_elements`` and ``clay_select_by``
write what is selected inside one object, either by explicit index or by a
seed/parameter through :data:`select.QUERIES`; and every element-gated
``clay_op`` row -- ``inset``, ``bevel``, ``extrude`` and the rest -- reads
that selection exactly as the keyboard and the context menu do. Before this,
nothing in this fold ever called ``ClayDoc.set_element_mode`` or
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
tools that can start a document from nothing (``clay_add_primitive``,
``clay_add_figure``, ``clay_add_mesh`` -- ``agent_clay_tools._h_add_primitive``,
``_h_add_figure`` and ``_h_add_mesh``) are allowed to mint one and adopt it
into the session -- ``clay_batch`` is a documented fourth way in, but only
because its first call is one of those three; see its own docstring
(``agent_clay_tools_batch._h_batch``).

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
the same way. See :func:`_tab` (``studio/modes/clay/agent/validate.py``) for the mechanics.

**Rendering owns a private viewport.** ``ClayView`` is a real GL object --
buffers, gizmos, a camera -- and the one already on screen
(``ctx.clay_view``) belongs to whatever tab the user is looking at. Reusing
it for ``clay_render`` would mean every render this fold produces first
yanks the user's camera onto the agent's document and back, which is a visible
stutter in the middle of whatever the person is doing. This module builds and
keeps its own :class:`~.clay_view.ClayView`, off the moderngl context the app
already has (``ctx.viewer.ctx``) but with no ``app_ctx`` of its own, so it
never reads or writes ``ctx.clay_view`` at all. :func:`release` is the
matching teardown, called by ``agent_host`` at **teardown**, not when the
bridge disconnects -- a reconnect is routine, and releasing on disconnect
would either run GL work off the frame thread or race teardown's own
ordered stop-then-release; a viewport torn down and rebuilt per connection
would be churn bought for nothing. :func:`_view_for` and :func:`release` stay
in *this* file rather than moving to ``studio/modes/clay/agent/tools_ops.py`` beside their
one caller (``_h_render``) because two tests monkeypatch ``_view_for`` on
``agent_clay`` by name (``tests/test_agent_clay.py``,
``tests/mcp/test_rpc_studio.py``) -- a handler that imported it as a plain
name from wherever it "really" lived would keep its own, unpatched copy, so
the fix that keeps the patch working is the same one that keeps the GL
object's one real owner in one place.

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
``agent_clay_tools_ops._OpCtx``: the absent ``clay_view`` makes Frame
Selection the no-op it should always have been for an agent with no
viewport of its own, ``state`` passes through for real because the
manifold-cache pop is real work that still has to happen, and every
``toast`` lands in the result instead of the running app -- so a per-object
refusal inside ``run`` that used to become a toast the user saw and the
agent never did now comes back as a message the agent can actually read.

**``clay_batch`` folds several tool calls into one undo step.** An agent
that wants to block out a scene one primitive at a time pays one round trip
per primitive and, worse, one Ctrl+Z per primitive for a user who wants to
back the whole attempt out; ``clay_batch`` (``agent_clay_tools_batch._h_batch``)
runs up to ``BATCH_MAX`` calls through :func:`call` under one
``history.mark()``/``collapse_since`` pair, stopping at the first refusal
and keeping the successful prefix. A batch entry never sees an earlier
entry's own result -- that only exists once the whole batch has returned --
so it has no uid to hand a later entry an object an earlier one just
created; a persistent recipe object with named parts was argued down in
design review as more machinery than that ask needed in favour of four
cheap pieces already mostly built (object names, ``clay_scene`` reporting
them, a plural ``clay_set_params``) plus this last one:
``{"$ref": "<name>"}``, found anywhere inside an entry's ``arguments``,
resolves to that name's uid the moment its entry runs
(``agent_clay_tools_batch._resolve_batch_ref``). Resolved per entry, not
against the whole ``calls`` list up front, for the same reason a bad
``uid`` already refuses at its own entry rather than in a pre-flight pass: a
name only exists once whatever created it has actually run. And ``$ref`` is
batch-only on purpose -- outside a batch an agent already holds the creating
call's own uid, so :func:`call` never learns ``$ref`` exists; a ``$ref``
handed to an ordinary call is refused as the malformed ``uid`` it is.

**``rollback_on_error`` is an opt-in beside the stop-and-keep contract, not a
replacement for it.** Because the whole run already folds into one step,
backing a kept prefix out is already a single ``clay_undo`` -- but an agent
that would rather the partial work never have existed can pass
``rollback_on_error=True``, and when the batch stops at a refusal
``_h_batch`` reverses the folded step with ``history.undo(doc,
redoable=False)`` before it returns (see that method's own docstring in
``undo.py`` for the cancelled-lift incident ``redoable=False`` exists for --
``redoable=True`` here would leave the abandoned batch on the redo stack for
a later ``clay_redo`` to bring back, exactly the outcome the agent asked to
avoid). This is **not** a third exception to "one tool call is one undo
step": a rolled-back batch pushes no step at all, the same shape as a
refusal that never mutated the document to begin with. And it reverses only
the document's own undo stack -- a tab this same batch minted still exists,
because minting one pushes no undo step either (see
``agent_clay_tools._h_add_primitive``'s own comment), and the two families
that push nothing (references, the selection tools) are exactly as
untouched by a rollback as by an ordinary ``clay_undo``.

**``clay_program`` compiles a small declarative program to a list of tool
calls (:mod:`.agent_program`) and folds the whole run into one undo step,
the same shape ``clay_batch`` already gives a run built by hand.** Where
``clay_batch`` takes calls an agent already assembled, one at a time,
``clay_program`` (``agent_clay_tools_batch._h_program``) takes a program --
``variables``, a nested ``steps`` list with ``repeat``/``array``/``mirror``/
``group``/``let``/``if`` sugar over expressions, and its own ``id``/``$ref``
namespace reusing ``clay_batch``'s own convention -- and
:func:`agent_program.compile_program` expands it, *before* any of it runs,
into the identical ``(tool_name, arguments)`` shape ``_fold_run`` already
knows how to run for ``clay_batch``. It is always atomic, unlike
``clay_batch``'s opt-in ``rollback_on_error``: a program that stops partway
rolls the whole attempt back rather than keeping a prefix, because a
compiled program is one request an agent reasons about as a whole, not a
sequence it is watching call by call. ``dry_run`` runs the compiled program
for real -- the only way to answer "would this refuse partway through"
honestly -- and then always undoes it with ``redoable=False`` before
returning, the same reversal ``rollback_on_error`` uses, so a dry run costs
exactly what the run it previews would have cost and leaves nothing behind.
A compiled ``("live", kind, arguments, path)`` placeholder -- a relative
move, a runtime assertion, none of them answerable without the live
document a batch alone cannot see -- is refused the moment its turn comes,
which folds and rolls back precisely like any other mid-run refusal;
:data:`PROGRAM_DEADLINE_S` bounds the whole run the same way, so an
agent-visible refusal always beats a frame stalled past what one MCP round
trip should cost.

**The undo enumeration, in full.** Together with ``clay_undo``/``clay_redo``
(which move the history head rather than pushing one of their own),
``clay_batch`` and ``clay_program`` are three exceptions that fold or move a
step -- what makes "one tool call is one undo step" true rather than
approximately true. Two families push none at all instead: references
(``clay_reference_add``
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

**Every tool answers the same JSON twice, on purpose.** ``_json``
(``studio/modes/clay/agent/validate.py``) -- what most of this fold's tools return
through -- puts *payload* in the result as text (``json.dumps``, what an
agent's model actually reads) and again as ``structuredContent`` (the same
data, for a client that wants to branch on a field instead of re-parsing
prose out of the text block). Duplication, not an oversight: a model and a
client are two different readers of one answer, and neither can stand in
for the other.

**A refusal is machine-readable, not only readable.** ``fail``
(``studio/modes/clay/agent/validate.py``) -- the thin wrapper over ``protocol.fail`` --
gives every refusal in this fold a ``changed`` key, defaulted in that one
wrapper rather than at each of its ~100 call sites: whether *this session's
own document* -- the ``ClayDoc`` itself -- was modified before the refusal
fired. Not whether a Library row was minted (``clay_export``'s own refusal
is ``changed: false`` by this definition even where it has already written
one) and not whether a session-scoped reference was added
(``clay_reference_add`` touches no ``ClayDoc`` at all) -- the document,
exactly as the rest of this fold already uses the word:
``agent_clay_tools._h_transform`` and ``_h_set_params`` answer ``changed``
on the *success* side today, and a refusal now answers the same question
the same way, rather than leaving a client to infer it from prose. ``field``
is untouched by any of this -- still ``service.errors``' own convention,
shared with the panes, which is why it is never folded into a differently-
shaped key. Where a refusal already names the objects or the op it is about
in its message, ``uids``/``op`` ride along too, so a client need not parse
them back out of the text block. ``recovery`` is a closed, bounded
vocabulary (:data:`RECOVERY`, below) naming what a client should try next --
``"fix_arguments"``, ``"read_scene"``, ``"switch_mode"``, ``"start_document"``,
``"retry"``, ``"wait"`` -- and a refusal whose recovery is genuinely unknown
(the blanket backstops in :func:`call` and in ``agent_host._call`` itself)
carries no ``recovery`` key at all: an absent key is a real, distinct answer,
never a seventh member invented to avoid omitting the field.

**A result that carries a picture does not duplicate its header into
``structuredContent``** -- the rule, stated once, rather than a list of tool
names it happens to apply to today. ``clay_render`` builds its result
directly with ``ok(header, *pngs)`` and ``clay_reference_get`` with
``ok(text(json.dumps(meta)), image_png(...))``, both bypassing ``_json`` for
the same reason: an image block has no JSON to duplicate. It matters most
for ``clay_render``, whose header is already checked twice against
``protocol.MAX_FRAME`` (see the render paragraph below) before it leaves --
giving that header a second life in ``structuredContent`` would spend frame
budget on bytes nothing reads. ``agent_host._carries_an_image`` asks this
exact question -- does this result's ``content`` include an image block --
to decide whether a remembered reply is worth replaying rather than re-run;
two decisions, in two modules, arriving at the same structural test, neither
one a hand-kept list of picture-shaped tools.

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
with ``field=`` set so ``fail`` derives ``recovery="fix_arguments"`` the
same way every other named-field refusal already does. The allowed names are
derived from the schema :func:`tools` already builds -- once, memoised --
never a second hand-kept table beside it, which is exactly the drift class
this fold's own derivation paragraphs above exist to rule out. Everything
about an argument's *value* -- a two-element vector, a NaN, an enum member
the registry does not have -- is still, as it always was, the handler's own
job to check before it mutates; only the *name* moved to this one door.
``tests/test_agent_schemas.py`` is what stands in for the validator
``mcp/protocol.py`` deliberately does not carry: it walks every constraint
every schema in this fold actually declares and proves the handler enforces
it, because a validator bolted onto that leaf would have had no way to check
the ``anyOf``, ``minimum``, ``maximum`` and ``exclusiveMinimum`` shapes these
schemas really use without ``mcp/protocol.py`` learning what Clay is -- and
that leaf staying ignorant of Clay is a decision this fold does not get to
revisit. A test that checks real behaviour is worth more than a validator
that checks only some of it.

Five tools -- ``clay_scene``, ``clay_add_primitive``, ``clay_add_mesh``,
``clay_diagnose`` and ``clay_analyze`` -- go one step further and declare an
``outputSchema`` describing that structured shape; the rest deliberately do
not, because a schema for a uid and a count is authorship with no reader.
``clay_add_mesh`` composes its schema from
``agent_clay_schema._object_row_output_schema`` rather than repeating it --
the same row ``clay_add_primitive`` declares, plus the two keys only this
tool answers with -- because a hand-copied second row schema is exactly the
drift the derivation paragraphs above rule out for a query enum or a
generator list, and a row's own shape is no different. None of the five
declares ``required``: a refusal shares this same result envelope
(``protocol.fail``'s own ``structuredContent`` is whatever ``field`` it was
given, nothing more), so a ``required`` list on the success shape would make
every refusal of these tools non-conforming for a client validating
strictly against its schema.

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
  overwrites. ``clay_export`` (``agent_clay_tools_ops._h_export``) never
  calls it at all, so ``tab.path``, ``tab.title`` and ``tab.saving`` are left
  exactly as they were. The authored document survives anyway:
  ``clay_mode.build_asset`` already writes it as the job's own ``build.wblk``
  sidecar (via ``save_clay_source``), which is what ``clay_mode.
  edit_asset_in_clay`` reopens from the Library -- there is no second copy
  for this fold to keep of its own.
* ``export_asset`` submits its own encode the same way and only knows the new
  job's id once that task finishes and calls back into ``clay_mode.
  on_task_done`` -- there is no id to return from this call if this fold
  goes through it as written. ``clay_export`` instead calls
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
import logging
from typing import Any

from .....kernels.mesh import elements as el
from .....kernels.mesh import ops as clay_geom_ops  # noqa: F401 -- re-exported, see below
from .....kernels.mesh import ops_boolean, presets
from .....kernels.mesh import primitives as bp
from .....kernels.mesh import select as bsel
from .....kernels.mesh.elements import OpError
from .....service.errors import ServiceError
from ....viewer.camera import Camera
from .. import ops as clay_ops
from ..ui.view import ClayView
from . import program as agent_program
from .schema import (
    _QUERY_ARG_SCHEMAS,
    BATCH_EXCLUDED,
    BATCH_MAX,
    ELEMENT_PAGE_DEFAULT,
    ELEMENT_PAGE_MAX,
    MAX_MESH_FACES,
    MAX_MESH_VERTICES,
    MAX_REFERENCES,
    MINTS_A_DOCUMENT,
    RENDER_PIXEL_BUDGET,
    RENDER_SHADINGS,
    _clay_analyze_output_schema,
    _clay_diagnose_output_schema,
    _clay_scene_output_schema,
    _figure_part_catalog,
    _generator_catalog,
    _mesh_row_output_schema,
    _object_row_output_schema,
    _op_catalog,
    _params_value_schema,
    _query_catalog,
    _vec3_schema,
)
from .tools import (
    _h_add_figure,
    _h_add_mesh,
    _h_add_primitive,
    _h_boolean,
    _h_delete,
    _h_material,
    _h_rename,
    _h_scene,
    _h_set_params,
    _h_transform,
)
from .tools_batch import (
    _h_batch,
    _h_program,
    _h_redo,
    _h_reference_add,
    _h_reference_get,
    _h_reference_list,
    _h_reference_remove,
    _h_undo,
)
from .tools_ops import (
    _h_analyze,
    _h_diagnose,
    _h_element_mode,
    _h_elements,
    _h_export,
    _h_op,
    _h_render,
    _h_select,
    _h_select_by,
    _h_select_elements,
)
from .validate import Session, _protocol, fail
from .validate import _euler_xyz_from_quat as _euler_xyz_from_quat
from .validate import _quat_from_euler_xyz as _quat_from_euler_xyz
from .validate import _tab as _tab

log = logging.getLogger(__name__)

# The three ``as``-aliased imports just above are re-exports, not uses: no
# function in this file calls ``_tab``, ``_quat_from_euler_xyz`` or
# ``_euler_xyz_from_quat`` itself (every caller lives in a handler file and
# reaches them through ``agent_clay_validate`` directly), but ``agent_host``
# calls ``agent_clay._tab`` by name and ``tests/test_agent_clay.py`` calls
# ``agent_clay._quat_from_euler_xyz``/``_euler_xyz_from_quat`` by name --
# both reaching through *this* module because it is the one every external
# caller and every test already imports. The self-aliasing (``import x as
# x``) is what tells ruff the "unused" import is deliberate.

# --- module constants, the dispatch half -------------------------------------
#
# See ``studio/modes/clay/agent/schema.py``'s own module docstring for the schema-vocabulary
# constants this file re-exports above (``BATCH_MAX``, ``MAX_MESH_VERTICES``
# and the rest) rather than owning. The two below stay here instead, and for
# the identical reason each other: both are monkeypatched directly on
# ``agent_clay`` by name in the test suite, which only works if the handler
# that reads them looks them up through this module, at call time, rather
# than importing a plain name from wherever the constant is textually
# defined.

PROGRAM_DEADLINE_S = 4.0
"""The wall-clock budget one ``clay_program`` call gets, measured from the
moment its compiled calls start running and checked between them (never
mid-call, so one already-running call is never cut off) -- past it, the run
rolls back and refuses rather than keep going into a second, third frame.
``clay_program`` is deliberately not chunked across frames the way ``pump``'s
own queued-job budget chunks ordinary calls (see ``dev/INVARIANTS.md``'s
agent paragraph): a program's whole point is that it is one MCP round trip,
and a caller that needs more than this buys should split the program into
several smaller ``clay_program`` calls rather than have this tool silently
spread one across an unbounded number of frames.

``tests/test_agent_clay.py`` shrinks this to ``0.0`` with ``monkeypatch.
setattr(agent_clay, "PROGRAM_DEADLINE_S", 0.0)`` to exercise the deadline
refusal without a real 4-second wait -- ``agent_clay_tools_batch._h_program``
reads this value back through a lazy ``from . import agent_clay`` rather
than importing the name directly, which is what lets that patch actually
change the handler's behaviour rather than being shadowed by an
already-bound copy."""

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
``fail``'s own docstring (``studio/modes/clay/agent/validate.py``) -- so this set is a
ceiling, never a default to fall back to.

* ``"fix_arguments"`` -- the request itself was malformed, out of range, or
  named something the registry this tool checks against has no entry for.
  Resending the identical call refuses the identical way; the arguments have
  to change first. Never written out at a call site: ``fail`` derives it
  from ``field=``, because a refusal that names the argument it is unhappy
  with is already saying which one to change. That covers the shared
  validators (``_validate_vec3``, ``_validate_number``, ``_validate_unit``,
  ``_validate_number_or_vec``, ``_validate_query_arg`` -- all
  ``studio/modes/clay/agent/validate.py``) and equally the eighty-odd refusals that name
  a field directly -- a bad ``generator``, an unknown ``kind``, a ``name``
  another object already wears -- without a second keyword on any of them.
* ``"read_scene"`` -- the request named a uid or an object this document does
  not have right now, so the client's picture of it is stale. Attached in
  ``_resolve_uid``/``_resolve_uids``. ``agent_host`` reuses it for its own
  started-call timeout refusal, where what is stale is not the document but
  the client's knowledge of whether the call it sent actually happened --
  re-reading (``clay_scene``, or ``warlock_status`` for that transport case)
  is the same recovery either way: go look before acting on a guess.
* ``"switch_mode"`` -- the document is in the wrong element mode for this
  call. ``clay_element_mode`` (or ``clay_select_elements``/``clay_select_by``'s
  own ``mode``) first, then repeat the call. Attached at the two sites using
  ``agent_clay_validate._OBJECT_SELECTION_DERIVED_REFUSAL``. **Not** attached
  to ``clay_op``'s own disabled-op refusal (``agent_clay_tools_ops._h_op``,
  via ``clay_ops.reason_for``) even though its wording sometimes names a mode
  gate -- ``op.reason`` covers several unrelated predicates
  (``_has_objects_reason``, ``_selection_reason``, ``_has_two_visible_reason``
  among them, see ``studio/modes/clay/ops.py``'s own "reasons" section) and only some of
  them are about element mode; ``_h_op`` has no way to tell which fired from
  the string alone, so it names the op (``op=``) instead of guessing a
  recovery that would be wrong for "Select an object first."
* ``"start_document"`` -- this session owns no document yet.
  ``clay_add_primitive``, ``clay_add_figure`` or ``clay_add_mesh`` starts
  one. Attached in ``_tab``'s own refusal.
* ``"retry"`` -- nothing ran; the identical call is safe to send again.
  ``agent_host``'s dropped-call timeout refusal.
* ``"wait"`` -- the same work is already running or queued; sending the same
  call again would run it twice. ``agent_host._replay``'s in-flight refusal.
"""


_view: ClayView | None = None
"""This module's own private viewport -- module-level rather than per-session
because a second concurrent session cannot exist yet (see ``mcp/pipe.py``'s
v1 decision, one connection at a time), so there is nothing for a per-session
instance to isolate that a shared one does not already give for free."""


def _view_for(ctx: Any) -> ClayView:
    """This module's own viewport, built the first time a render is asked for.

    ``ctx.viewer`` is the app's main 3D pane and always exists once the app has
    a window, which is what makes it the one place to borrow a moderngl
    context from without reaching for the interactive Clay viewport itself.
    ``app_ctx=None`` is what keeps :class:`ClayView` from reading
    ``ctx.clay_view`` or the shared tool setting through ``self.state`` --
    see this module's own docstring's rendering claim.
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


# --- the tool surface itself --------------------------------------------------


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
    :func:`call`), so a module-scope import back would be a cycle. This is
    this file's own established precedent for the same move -- ``agent_clay_
    tools_ops._core``/``agent_clay_tools_batch._core`` now carry the identical
    reasoning for the opposite direction, a handler reaching back into this
    module.
    """
    from .... import agent_host

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
        "One tool call is one undo step, with three exceptions that fold or "
        "move steps -- clay_batch and clay_program each fold their whole "
        "run into one, and clay_undo/clay_redo move the history head "
        "rather than pushing one of their own -- and two families that "
        "push none at all: adding a "
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
        "clay_program compiles a declarative program -- variables, "
        "expressions, repeat/array/mirror/group/let/if -- to the same kind "
        "of call list and runs it the same way, always atomic: unlike "
        "clay_batch's opt-in rollback_on_error, any failure rolls the "
        "whole attempt back. Prefer clay_program over clay_batch once a "
        "build has real structure -- a repeated part, a computed "
        "placement, a name reused across several steps -- rather than "
        "assembling and resolving that structure call by call; reach for "
        "clay_batch instead when the calls are already known and few, or "
        "when a partial, kept prefix is useful on a refusal. clay_program "
        "cannot be a clay_batch entry, and dry_run lets a program be "
        "previewed -- built for real and then undone -- before it is run "
        "for keeps. Its move/turn/scale_by steps read the live document to "
        "compose a relative delta, and assert checks a condition against it "
        "(lo/hi/size/center/count/exists/touches/grounded/floating/volume) "
        "-- a false or unevaluable assert rolls the whole program back, the "
        "same as any other failed step.\n\n"
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
    named in this module's own docstring. Called once per ``tools/list``
    request, so rebuilding it from ``GENERATORS``/``ASSEMBLIES``/``OPS``/
    ``QUERIES``/``_HANDLERS`` each time costs nothing and can never go stale
    against an edit to any of them -- a query enum written out by hand would
    have been a fifth place to remember one exists, beside the menu, the
    tools pane, the key handler and ``OPS`` itself."""

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
                "in the document. Parts, each prefixed by name_prefix: "
                + _figure_part_catalog(tuple(sorted(presets.ASSEMBLIES)))
                + "."
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
                "One or more white-background square renders of the "
                "document -- no gizmos -- from the standard three-quarter "
                "framing, a named axis view, or a free yaw/pitch pair. "
                "'view' and 'views' are exclusive; giving both is refused. "
                "'shading' picks how the surface is drawn: 'unlit' "
                "(default) is the material's own colour with no lighting -- "
                "not 'flat-shaded' in the lit-and-shaded sense that phrase "
                "usually means, just the albedo, which is what makes two "
                "renders comparable regardless of where the light sits. "
                "'lit' adds the same lighting the viewport itself uses. "
                "'wireframe' draws edges only, 'wire_overlay' draws the "
                "shaded surface with edges over it, and 'xray' draws the "
                "surface translucent. 'object_id' draws every visible "
                "object as a flat, unique colour instead -- see 'ids' below "
                "-- and refuses combined with 'grid'. 'grid' draws the "
                "ground plane at y=0, the one scale cue available with no "
                "viewport to walk around in: 16 cells across a span rounded "
                "up to a power of ten containing 2.5x the framed footprint, "
                "so one cell reads as span/16 metres. 'focus' points the "
                "camera at the union of the named objects' boxes -- "
                "everything else is still drawn, since the renderer has no "
                "per-object alpha. Refused, before any GPU work, when the "
                f"requested views would exceed the {RENDER_PIXEL_BUDGET:,}"
                "-pixel render budget or would not fit in one reply frame "
                "once encoded -- ask for fewer or smaller views instead. "
                "Pass 'compare' (a stored reference's name) to render "
                "exactly one view beside it or blended over it instead of "
                "the normal multi-view result -- see clay_reference_add; "
                "'object_id' cannot be combined with 'compare'. A compare "
                "reply's header also carries 'silhouette': shape IoU and "
                "bounding-box aspect between the reference and the render, "
                "or null with a 'reason' (no subject, a flood-fill leak, or "
                "a mask covering almost the whole frame) -- never a "
                "refusal, the picture returns either way. An 'object_id' "
                "render's text reply carries 'ids': one row per visible "
                "object, [uid, '#rrggbb' colour, pixel count], the same "
                "colour for a uid in every view of one call -- a pixel "
                "count of 0 means that object is hidden from that view, not "
                "that it does not exist."
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
                    "shading": {"type": "string", "enum": list(RENDER_SHADINGS)},
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
                "clay_batch, clay_program, clay_render, clay_export, "
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
            name="clay_program",
            title="Compile and run a build program",
            description=(
                "Compile a small declarative program to a list of tool "
                "calls and run it as one atomic undo step, labelled 'Agent "
                "program' -- like clay_batch but built from a program "
                "rather than assembled call by call, and always atomic: "
                "any failure rolls the whole attempt back rather than "
                "keeping a prefix. 'variables' seeds named numbers; "
                "'steps' is a list, each entry exactly one kind: add "
                "(generator/params/translation/rotation/scale/id/material, "
                "like clay_add_primitive), figure (key/translation/yaw/"
                "scale/id, like clay_add_figure), mesh (positions/faces/uv/"
                "translation/rotation/scale/id/material, like "
                "clay_add_mesh), transform (uid/translation/rotation/"
                "scale), params (uid or uids, plus params), material "
                "(uids/color/name/metallic/roughness), delete (uids), op "
                "(name/params/uids -- an object-mode clay_op row only), "
                "boolean (kind/uids), select (uids), repeat (ranges/steps, "
                "expanding every named range's cartesian product), array "
                "(id/count/var/add, sugar for a numbered row), mirror "
                "(axis/add, places the original and a reflected copy -- "
                "placement only, never the mesh itself), group (id/"
                "members, names a set for a later uids field), let (vars, "
                "binds more variables for the rest of this steps list), if "
                "(cond/then/else), move (uid/by -- by added to the "
                "target's current translation), turn (uid/by -- degrees "
                "composed in world space onto the current rotation), "
                "scale_by (uid/factor -- a number or [x,y,z] multiplied "
                "onto the current scale) and assert (condition, optionally "
                "uid -- a false or unevaluable condition refuses the whole "
                "program). move/turn/scale_by read the live document, so "
                "their uid may also name a group (one call per member, "
                "still one undo step). assert's condition is the numeric "
                "expression language below plus facts, each taking bare "
                "ids/groups (never $name or a string): lo/hi/size/center"
                "(id, axis 0|1|2) read a world-space box; count(group); "
                "exists(name); touches(id, id); grounded(id); floating(id) "
                "(whole-document); volume(id) (0 unless closed). A numeric field (translation, a "
                "params value, a range bound, ...) takes a plain number or "
                "an expression string: + - * / % and ^ for power, "
                "comparisons and and/or/not, parentheses, degree trig "
                "(sin/cos/tan/asin/acos/atan2), sqrt/abs/min/max/floor/"
                "ceil/clamp/lerp/round, the constant pi, and $name for a "
                "variable -- never Python, nothing is eval'd. An id field "
                "takes a plain string or one templated with {name} (e.g. "
                "\"leg_{i}\" inside a repeat over i) -- never $name, which "
                "is for numeric fields only. A reference to an object this "
                "program placed is its id (a bare string, or {\"id\": "
                "...}/{\"$ref\": ...}, the same convention clay_batch "
                "uses); a reference to one already in the document is its "
                "uid (an integer, or {\"uid\": ...}); a uids field also "
                "takes a group name or a list mixing any of those. "
                "Limits: up to "
                f"{agent_program.PROGRAM_MAX_STEPS} steps in one list "
                "(every nested repeat/if body counts its own, up to "
                f"{agent_program.PROGRAM_MAX_NESTING} lists deep), "
                f"{agent_program.PROGRAM_MAX_CALLS} expanded tool calls "
                f"total, {agent_program.PROGRAM_MAX_REPEAT} iterations per "
                f"repeat/array, {agent_program.PROGRAM_MAX_BOOLEANS} "
                "boolean steps, and "
                f"{agent_program.PROGRAM_MAX_VARIABLES} variables in scope "
                "at once. dry_run runs the program for real and then "
                "undoes it before returning, reporting what would have "
                "been built (uids omitted) with no lasting change; with no "
                "document open yet, a dry run only compiles and never "
                "starts one. clay_program cannot itself be a clay_batch "
                "entry. A program expensive enough to near these limits -- "
                "many repeat iterations, several booleans -- is exactly "
                "what a client that has declared the MCP Tasks extension "
                "should let run as a task rather than wait on "
                "synchronously."
            ),
            schema={
                "type": "object",
                "properties": {
                    "variables": {"type": "object"},
                    "steps": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": agent_program.PROGRAM_MAX_STEPS,
                        "items": {"type": "object"},
                    },
                    "dry_run": {"type": "boolean"},
                },
                "required": ["steps"],
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


# --- dispatch -----------------------------------------------------------------


@functools.cache
def _allowed_argument_names() -> dict[str, frozenset[str]]:
    """Tool name -> the frozenset of its schema's own top-level ``properties``
    keys -- what :func:`call` checks a real call's ``arguments`` against
    before any handler runs. See this module's own docstring's paragraph on
    why an unknown argument is refused rather than dropped; this is the
    lookup that makes the refusal *derived* rather than a second hand-kept
    table.

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
    ``clay_ops.OPS``, ``bsel.QUERIES`` -- see this module's own docstring's
    opening paragraph). A thirteenth generator changes what
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
    """Run one tool. Never raises -- see this module's own docstring's
    safety claim and the class of error each of these three turns into a
    refusal for."""
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
    "clay_program": _h_program,
    "clay_reference_add": _h_reference_add,
    "clay_reference_list": _h_reference_list,
    "clay_reference_get": _h_reference_get,
    "clay_reference_remove": _h_reference_remove,
}
