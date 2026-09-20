"""Clay's agent tool surface, the batch/program/history/reference handler
family: ``clay_undo``, ``clay_redo``, ``clay_batch``, ``clay_program``,
``clay_reference_add``, ``clay_reference_list``, ``clay_reference_get`` and
``clay_reference_remove`` -- everything that runs several tool calls as one
unit, moves the undo head instead of pushing a step, or manages a session's
own reference pictures.

Split out of ``studio/modes/clay/agent/dispatch.py`` in the P4 restructure (``dev/RESTRUCTURE.md``);
see ``studio/modes/clay/agent/tools.py``'s own module docstring for the general shape of
this three-way handler split and ``studio/modes/clay/agent/tools_ops.py``'s for why a
handler here reaches ``studio/modes/clay/agent/dispatch.py`` itself only through a lazy,
function-scope accessor (:func:`_core` below, the identical pattern) rather
than a plain import: ``_h_batch`` needs the live ``_HANDLERS`` table to know
what is batchable, and ``_h_program``/``_resolve_and_call``/
``_run_live_transform`` all need to run a compiled step back through
``agent_clay.call`` itself -- both of which are ``studio/modes/clay/agent/dispatch.py``'s own
dispatch machinery, reached back into rather than duplicated.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Any

from .....kernels.geom3d import math3d as m3
from .....kernels.mesh import analyze as clay_analyze
from .....kernels.mesh import ops as clay_geom_ops
from .....kernels.mesh.elements import OpError
from .....service import files as svc_files
from .....service import validation as svc_validation
from .....service.errors import NotFound
from ....viewer.camera import Camera
from .. import mode as clay_mode
from . import program as agent_program
from .schema import BATCH_MAX, MAX_REFERENCES, MINTS_A_DOCUMENT, RENDER_FRAME_RESERVE
from .validate import (
    Session,
    _euler_xyz_from_quat,
    _json,
    _label_top,
    _protocol,
    _quat_from_euler_xyz,
    _tab,
    fail,
    image_png,
    ok,
    text,
)


def _core() -> Any:
    """``studio/modes/clay/agent/dispatch.py`` itself, imported lazily -- the same reach and the
    same two reasons ``agent_clay_tools_ops._core`` already carries this
    docstring for: a module-scope import back would cycle (``studio/modes/clay/agent/dispatch.py``
    imports this module to build ``_HANDLERS``) and would be a fresh,
    undocumented ``tests/test_layering.py`` violation (that walk is
    module-scope only, by that test's own design)."""
    from . import dispatch as agent_clay

    return agent_clay


def _move_history(ctx: Any, session: Session, args: dict, *, redo: bool) -> dict:
    """The shared body of ``clay_undo``/``clay_redo``. See ``agent_clay.tools``
    for the documented "moves the head, pushes nothing" exception both
    belong to."""
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
    guessed at (which key wins?), and ``tests/modes/clay/test_agent_clay.py``'s
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


def _resolve_and_call(
    ctx: Any, session: Session, doc: Any, name: str, arguments: dict
) -> dict:
    """Resolve every ``{"$ref": "<name>"}`` in *arguments* against *doc* --
    see :func:`_resolve_batch_ref` -- and run *name* through
    ``agent_clay.call``. The single-entry step :func:`_fold_run` takes for a
    plain ``(name, arguments)`` entry, factored out so a caller that has to
    interpose something of its own around the call (``clay_program``'s
    deadline check) still reaches the identical ``$ref`` handling rather than
    a second copy of it."""
    resolved: dict[str, Any] = {}
    for key, value in (arguments or {}).items():
        resolved[key], ref_failure = _resolve_batch_ref(doc, value, key)
        if ref_failure:
            return ref_failure
    return _core().call(ctx, session, name, resolved)


def _run_entry(ctx: Any, session: Session, doc: Any, entry: Any) -> dict:
    """One :func:`_fold_run` entry: a plain ``(name, arguments)`` pair, run
    through :func:`_resolve_and_call`, or a live thunk -- anything callable,
    taking ``(doc, session)`` and returning a tool result already built --
    invoked directly. The thunk shape is what lets a caller other than
    ``clay_batch`` (``clay_program``'s own live-kind placeholders, and its
    deadline check ahead of an ordinary call) plug into the identical fold
    with nothing in :func:`_fold_run` itself needing to know about either."""
    if callable(entry):
        return entry(doc, session)
    name, arguments = entry
    return _resolve_and_call(ctx, session, doc, name, arguments)


def _fold_run(
    ctx: Any,
    session: Session,
    doc: Any,
    entries: list[Any],
    *,
    rollback: bool,
    label: str,
) -> tuple[list[dict], int | None, bool, bool]:
    """Run *entries* -- see :func:`_run_entry` for the two shapes one may
    take -- under one ``history.mark()``/``collapse_since`` fold, stopping at
    the first refusal and keeping the successful prefix. Shared by
    ``clay_batch`` and ``clay_program``, which differ only in what they hand
    it: a batch's entries are the calls an agent already assembled, one at a
    time; a program's are :func:`agent_program.compile_program`'s own
    expanded call list, wrapped in a deadline-checking thunk apiece.

    -> ``(results, stopped_at, rolled_back, changed)``: *results* is one
    tool result per entry actually run (stopping at the first refusal, so
    shorter than *entries* on a stop); *stopped_at* is that entry's index, or
    ``None`` if every entry succeeded; *rolled_back* is whether *rollback*
    fired; *changed* is ``doc.history.head != mark`` after everything above,
    true whether the run completed, stopped with a successful prefix kept,
    or (once a rollback has run) landed back at ``mark`` by construction --
    the same single expression answers all three rather than a rollback
    branch hand-setting it.

    ``rollback``: when true and the run stops at a refusal, the folded step
    is undone with ``history.undo(doc, redoable=False)`` before this
    returns -- not left for a later ``clay_undo``, and not redoable, because
    the whole point of asking for this is that the partial work should never
    have existed. ``redoable=True`` (the default ``undo()`` a human's Ctrl+Z
    takes) would leave the abandoned attempt on the redo stack, where a
    later ``clay_redo`` could bring back exactly the work the caller asked
    to erase -- the cancelled-lift shape ``UndoStack.undo``'s own docstring
    describes that keyword for. Only the document's own undo stack is
    unwound: a tab this same run minted still exists, because minting one
    pushes no undo step to begin with, and neither does an element-mode or
    selection change, or a reference add, along the way -- both of those
    families survive this exactly as they survive an ordinary ``clay_undo``.
    Gated on ``doc.history.head != mark`` first, never merely on ``stopped_at
    is not None``: a run that stopped at its very first entry, before that
    entry ever mutated anything, has nothing to undo, and calling
    ``history.undo`` there would unwind whatever step was already on top
    before this run started -- the caller's *previous* action, not this
    one's.

    ``label``: the step this run just pushed is renamed to *label* -- but
    only when it actually pushed one, and skipped entirely once rolled back
    (undoing the folded step already put ``doc.history.head`` back at
    ``mark``, so there is no step left on top to rename) -- see
    :func:`_label_top`.
    """
    mark = doc.history.mark()
    results: list[dict] = []
    stopped_at: int | None = None
    for i, entry in enumerate(entries):
        result = _run_entry(ctx, session, doc, entry)
        results.append(result)
        if result.get("isError"):
            stopped_at = i
            break
    doc.history.collapse_since(mark)

    rolled_back = False
    if rollback and stopped_at is not None and doc.history.head != mark:
        doc.history.undo(doc, redoable=False)
        rolled_back = True

    if not rolled_back:
        _label_top(doc, mark, label)

    changed = doc.history.head != mark
    return results, stopped_at, rolled_back, changed


def _h_batch(ctx: Any, session: Session, args: dict) -> dict:
    """Run several tools as one undo step, through :func:`_fold_run`. See
    ``studio/modes/clay/agent/dispatch.py``'s own module docstring's paragraph on the fold and
    ``agent_clay_schema.BATCH_EXCLUDED`` for what this refuses to run and why.

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
    wired into ``agent_clay.call``: outside a batch an agent already holds
    the creating call's own result, uid included, so a ``$ref`` there would
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
    -- see ``studio/modes/clay/agent/dispatch.py``'s own module docstring's paragraph on this
    argument for why ``redoable=False`` and for what a rollback does and
    does not reach.
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
    core = _core()
    allowed = set(core._HANDLERS) - core.BATCH_EXCLUDED
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

    # ``$ref`` resolution, the ``history.head != mark`` fold key,
    # ``redoable=False`` and the top-step label are all in :func:`_fold_run`
    # now -- entries here are plain ``(name, arguments)`` pairs, exactly the
    # shape it already knows how to run.
    entries = [(entry["name"], entry.get("arguments") or {}) for entry in calls]
    results, stopped_at, rolled_back, changed = _fold_run(
        ctx, session, doc, entries, rollback=rollback_on_error, label="Agent batch",
    )

    # "completed" is diagnostic and unaffected by rollback: how many calls
    # succeeded before the refusal fired stays true regardless of whether
    # that work was then reversed, so "completed: 2, rolled_back: true" is
    # not a contradiction -- one reports what ran, the other what remains.
    completed = len(results) - (1 if stopped_at is not None else 0)
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


def _step_locator(compiled_call: tuple[Any, ...]) -> dict[str, Any]:
    """Where one :class:`agent_program.Compiled` call came from, and what it
    is -- the ``{"step", "call"}`` shape :func:`_h_program` reports in
    ``stopped_at`` and ``failure``. ``step`` is the compiler's own path
    (``"steps[2].add"``, ...), the same string a compile refusal's message
    already carries; ``call`` is the tool name, or ``"live:<kind>"`` for one
    of :data:`agent_program.LIVE_KINDS`, which has no tool name of its own."""
    if compiled_call[0] == "live":
        return {"step": compiled_call[-1], "call": f"live:{compiled_call[1]}"}
    return {"step": compiled_call[-1], "call": compiled_call[0]}


def _describe_compiled_call(compiled_call: tuple[Any, ...]) -> dict[str, Any]:
    """One compiled call, as ``dry_run``'s own ``calls`` preview reports it --
    what would run, without running it."""
    if compiled_call[0] == "live":
        _, kind, arguments, _path = compiled_call
        return {"live": kind, "arguments": arguments}
    name, arguments, _path = compiled_call
    return {"name": name, "arguments": arguments}


@dataclass
class _ConditionAccess:
    """The ``access`` adapter :func:`agent_program.evaluate_condition` calls
    into -- the one seam that hands a *live* fact something to measure,
    built fresh per ``assert`` step rather than once per program, since the
    document it wraps must be exactly the one the step is running against
    right now (not the one at program-compile time, which for a live step
    reached mid-program has already been mutated by every step before it).

    Every method here either resolves a name against *doc* the same way
    :func:`_resolve_batch_ref` already does for a real tool call's own
    ``$ref``, or reads a fact off :mod:`.clay.analyze`/``clay_geom_ops`` the
    same way ``clay_scene`` and ``clay_diagnose`` already do -- nothing here
    is a new way to look at the document, only a new door into the old one.
    """

    doc: Any
    groups: dict[str, tuple[str, ...]]

    def resolve(self, name: str) -> int:
        matches = [obj.uid for obj in self.doc.objects if obj.name == name]
        if not matches:
            raise agent_program.ConditionError(f"no object named {name!r}.")
        if len(matches) > 1:
            raise agent_program.ConditionError(
                f"{len(matches)} objects are named {name!r}; this program's "
                "own ids are no longer unique in the document."
            )
        return matches[0]

    def resolve_group(self, name: str) -> list[int]:
        return [self.resolve(member) for member in self.groups.get(name, ())]

    def exists(self, name: str) -> bool:
        return any(obj.name == name for obj in self.doc.objects)

    def bounds(self, uid: int) -> tuple[Any, Any]:
        obj = self._by_uid(uid)
        # Evaluated, not the base -- a program's own assert reads the same
        # box clay_scene/clay_render would show a person, not half of it
        # from an object that carries a mirror or an array modifier.
        # world=: tranche 3 -- a parented object's own TRS is local to its
        # parent, not its world placement, so an assert against a parented
        # object needs the ancestor-composed matrix the same way clay_scene's
        # own bbox does.
        box = clay_geom_ops.world_box(
            obj, self.doc.evaluated(uid), world=self.doc.world_matrix(uid)
        )
        if box is None:
            raise agent_program.ConditionError(f"{obj.name!r} has no geometry to measure.")
        return box

    def touches(self, uid_a: int, uid_b: int) -> bool:
        obj_a, obj_b = self._by_uid(uid_a), self._by_uid(uid_b)
        analysis = self._analyze([obj_a, obj_b], pairs_among=[uid_a, uid_b])
        return bool(analysis.pairs) and analysis.pairs[0].contact

    def grounded(self, uid: int) -> bool:
        obj = self._by_uid(uid)
        analysis = self._analyze([obj], pairs_among=[uid])
        row = analysis.objects[0]
        return bool(row.ground and row.ground.contact)

    def floating(self, uid: int) -> bool:
        # The one fact that genuinely needs the whole document, not just the
        # object(s) named in the condition -- see clay_analyze.analyze's own
        # docstring on why ``pairs_among=None`` is what turns "floating" on
        # at all.
        analysis = self._analyze(list(self.doc.objects), pairs_among=None)
        return uid in (analysis.floating or ())

    def volume(self, uid: int) -> float:
        obj = self._by_uid(uid)
        analysis = self._analyze([obj], pairs_among=[uid])
        vol = analysis.objects[0].volume
        return float(vol) if vol is not None else 0.0

    def _by_uid(self, uid: int) -> Any:
        try:
            return self.doc.by_uid(uid)
        except KeyError:
            raise agent_program.ConditionError(f"no object with uid {uid}.") from None

    def _analyze(self, objects: list[Any], *, pairs_among: list[int] | None) -> Any:
        # doc=self.doc: touches/grounded/floating/volume are all questions
        # about what an object actually occupies, which a modifier stack (a
        # solidify's own thickness, a mirror's own second half) changes as
        # much as a transform does -- analyze.analyze's own ``doc`` kwarg
        # swaps every object's mesh for its evaluated one before measuring.
        try:
            return clay_analyze.analyze(objects, doc=self.doc, pairs_among=pairs_among)
        except OpError as error:
            raise agent_program.ConditionError(str(error)) from None


def _run_live_transform(ctx: Any, session: Session, doc: Any, kind: str, arguments: dict) -> dict:
    """Run one compiled ``move``/``turn``/``scale_by`` entry: resolve its
    already-validated target, read that object's *current* transform,
    compose this step's own delta onto it, and issue the resulting absolute
    values through the real ``clay_transform`` tool -- never ``doc.
    set_transform`` directly, so this step gets exactly the same argument
    validation, history behaviour and refusal shape any other caller of that
    tool already gets."""
    uid, failure = _resolve_batch_ref(doc, arguments["uid"], "uid")
    if failure:
        return failure
    try:
        obj = doc.by_uid(int(uid))
    except (KeyError, TypeError, ValueError):
        return fail(f"no object with uid {uid!r}.", field="uid", recovery="read_scene")

    call = _core().call

    if kind == "move":
        delta = arguments["by"]
        translation = [float(obj.translation[i]) + delta[i] for i in range(3)]
        return call(ctx, session, "clay_transform", {"uid": uid, "translation": translation})

    if kind == "turn":
        # Composed in world space, the same frame ``by`` already implies for
        # move (a plain vector add, not one rotated into the object's own
        # axes first) and factor already implies for scale_by (a plain
        # multiply) -- so all three read the same way to an agent: "add this
        # delta to what is already there," never "in this object's own,
        # possibly already-turned, frame." ``quat_mul(a, b)`` applies ``b``
        # first, so the delta quaternion goes on the *left* to apply after
        # the object's current orientation.
        delta_deg = arguments["by"]
        new_quat = m3.quat_mul(_quat_from_euler_xyz(delta_deg), obj.rotation)
        rotation = list(_euler_xyz_from_quat(new_quat))
        return call(ctx, session, "clay_transform", {"uid": uid, "rotation": rotation})

    # scale_by
    factor = arguments["factor"]
    if isinstance(factor, list):
        scale = [float(obj.scale[i]) * factor[i] for i in range(3)]
    else:
        scale = [float(obj.scale[i]) * factor for i in range(3)]
    return call(ctx, session, "clay_transform", {"uid": uid, "scale": scale})


def _run_live_assert(doc: Any, arguments: dict, groups: dict[str, tuple[str, ...]]) -> dict:
    """Run one compiled ``assert`` entry: evaluate its condition against
    *doc*, right now, through :func:`agent_program.evaluate_condition`. A
    false result and an unevaluable one (:class:`agent_program.
    ConditionError` -- an id that stopped resolving, a division by zero, a
    non-finite result) both refuse the same way, because either means this
    program's own assumption did not hold; the message tells the two apart."""
    access = _ConditionAccess(doc=doc, groups=groups)
    try:
        value = agent_program.evaluate_condition(arguments["ast"], arguments["scope"], access)
    except agent_program.ConditionError as error:
        return fail(
            f"assert {arguments['condition']!r} could not be evaluated: {error}",
            field="condition",
        )
    if not math.isfinite(value):
        return fail(
            f"assert {arguments['condition']!r} evaluated to a non-finite value.",
            field="condition",
        )
    if value != 0.0:
        return _json({"assert": arguments["condition"], "result": True})
    return fail(f"assert failed: {arguments['condition']}.", field="condition")


def _run_live_step(
    ctx: Any, session: Session, doc: Any, kind: str, arguments: dict,
    groups: dict[str, tuple[str, ...]],
) -> dict:
    if kind == "assert":
        return _run_live_assert(doc, arguments, groups)
    if kind in ("move", "turn", "scale_by"):
        return _run_live_transform(ctx, session, doc, kind, arguments)
    return fail(f"the {kind!r} step kind cannot run yet.", field="steps")  # pragma: no cover


def _h_program(ctx: Any, session: Session, args: dict) -> dict:
    """Compile a declarative program (:mod:`.agent_program`) and run it as
    one atomic undo step, through the same :func:`_fold_run` ``clay_batch``
    runs on. See ``studio/modes/clay/agent/dispatch.py``'s own module docstring's paragraph on the
    fold for the contract in full; this docstring covers only what is
    specific to this handler.

    Always atomic, unlike ``clay_batch``'s opt-in ``rollback_on_error``: a
    program that stops at a refusal always rolls the whole attempt back
    (``_fold_run(..., rollback=True, ...)``), because a compiled program is
    one request an agent reasons about as a whole rather than a sequence it
    watches call by call and might want a kept prefix from.

    ``dry_run``: with no document open yet, this only compiles -- reported as
    ``validated: "compile"`` -- and never mints a tab, because there would be
    nothing to run the program against without minting one first, and a dry
    run's whole point is to cost nothing lasting. With a document already
    open (or once one exists), a dry run behaves exactly like a real run and
    then, on success, undoes it the same ``redoable=False`` way a rollback
    does -- so a dry run costs what the run it previews would have cost and
    leaves nothing behind; ``doc.dirty`` is a comparison against
    ``saved_head`` (``document.py``), not a latch, so undoing back to the
    same head restores it exactly as it stood before this call. ``objects``
    is read off the document immediately after the run, before that undo, so
    a dry run still reports what *would* exist -- with ``uid`` omitted,
    since by the time a caller reads the reply those uids are gone.

    Object mode is required at the start: every compiled step is object-mode
    (:func:`agent_program._Compiler._compile_op` already refuses to compile
    an element-mode op), so a document left in vertex/edge/face mode could
    only ever have every compiled call refuse in turn -- refusing once,
    up front, is the honest version of that rather than a confusing
    per-step echo of it.

    A compiled ``("live", kind, arguments, path)`` placeholder -- one of
    :data:`agent_program.LIVE_KINDS`, each needing the live document a batch
    alone cannot see -- runs through :func:`_run_live_step` at its own turn
    in the fold: ``move``/``turn``/``scale_by`` read the target's current
    transform and issue their own ``clay_transform`` call, and ``assert``
    evaluates its condition through :func:`agent_program.evaluate_condition`
    against a fresh :class:`_ConditionAccess`. A refusal there (a failed
    ``clay_transform``, a false or unevaluable assert) folds into the run
    exactly like any other, so it rolls the whole attempt back the same way
    a real tool's refusal would.
    """
    dry_run = args.get("dry_run", False)
    if not isinstance(dry_run, bool):
        return fail("dry_run must be a boolean.", field="dry_run")

    state = clay_mode.ensure(ctx)
    existing_tab = state.get(session.tab_uid) if session.tab_uid else None
    live_names = (
        frozenset(o.name for o in existing_tab.doc.objects) if existing_tab is not None
        else frozenset()
    )

    try:
        compiled = agent_program.compile_program(args, live_names=live_names)
    except agent_program.ProgramError as error:
        return fail(str(error), field=error.field)

    if dry_run and existing_tab is None:
        payload = {
            "dry_run": True,
            "validated": "compile",
            "expanded": compiled.expanded,
            "completed": 0,
            "stopped_at": None,
            "changed": False,
            "rolled_back": False,
            "objects": [],
            "groups": {name: list(members) for name, members in compiled.groups.items()},
            "calls": [_describe_compiled_call(c) for c in compiled.calls],
        }
        return _json(payload)

    tab, failure = _tab(ctx, session, create=True)
    if failure:
        return failure
    doc = tab.doc

    if doc.element_mode != "object":
        return fail(
            "clay_program requires object mode -- every compiled step is "
            "object-mode. Call clay_element_mode with mode='object' first.",
            recovery="switch_mode",
        )

    # Captured before anything runs, so a rollback below can put it back:
    # ``doc.select`` pushes no undo step (selection is not undoable by
    # design), so ``history.undo`` alone would leave whatever a compiled
    # ``add`` step's own selecting-on-placement left behind -- the newly
    # created object's own uid, already gone -- rather than what was
    # selected before this call ever touched the document.
    prior_selection = set(doc.selection)

    deadline_s = _core().PROGRAM_DEADLINE_S
    deadline = time.monotonic() + deadline_s

    def _make_entry(index: int, compiled_call: tuple[Any, ...]) -> Any:
        def _run(doc: Any, session: Session) -> dict:
            # Checked between calls, never mid-call -- a call already running
            # is never cut off, and the very first call always gets to run
            # regardless of how close the budget already is, so a run that
            # does nothing at all can never be blamed on this.
            if index > 0 and time.monotonic() > deadline:
                return fail(
                    f"clay_program exceeded its {deadline_s:g}s "
                    "deadline before this step ran; split the program into "
                    "smaller clay_program calls.",
                )
            if compiled_call[0] == "live":
                kind, arguments = compiled_call[1], compiled_call[2]
                return _run_live_step(ctx, session, doc, kind, arguments, compiled.groups)
            name, arguments, _path = compiled_call
            return _resolve_and_call(ctx, session, doc, name, arguments)

        return _run

    entries = [_make_entry(i, c) for i, c in enumerate(compiled.calls)]
    results, stopped_at, rolled_back, changed = _fold_run(
        ctx, session, doc, entries, rollback=True, label="Agent program",
    )

    # Read off *doc* right now, before any dry-run undo below -- a program
    # id that was deleted or consumed by a boolean along the way genuinely
    # has no object to report, exactly as the document itself would say.
    by_name = {obj.name: obj for obj in doc.objects}
    objects_out: list[dict[str, Any]] = []
    for pid, wire_name in compiled.objects.items():
        obj = by_name.get(wire_name)
        if obj is not None:
            objects_out.append({"id": pid, "name": wire_name, "uid": obj.uid})

    if dry_run and not rolled_back and changed:
        doc.history.undo(doc, redoable=False)
        rolled_back = True
        changed = False

    # A rollback (a run that failed, or a dry run undoing its own success)
    # leaves the document exactly as it stood before this call in every way
    # ``history.undo`` reaches -- except the selection, which never pushed a
    # step for it to reach; restored by hand here for the identical reason.
    if rolled_back:
        doc.select(prior_selection)

    if dry_run:
        objects_out = [{"id": r["id"], "name": r["name"]} for r in objects_out]

    completed = len(results) - (1 if stopped_at is not None else 0)
    payload = {
        "dry_run": dry_run,
        "validated": "execute",
        "expanded": compiled.expanded,
        "completed": completed,
        "stopped_at": None if stopped_at is None else _step_locator(compiled.calls[stopped_at]),
        "changed": changed,
        "rolled_back": rolled_back,
        "objects": objects_out,
        "groups": {name: list(members) for name, members in compiled.groups.items()},
    }
    if dry_run:
        payload["calls"] = [_describe_compiled_call(c) for c in compiled.calls]
    if stopped_at is not None:
        payload["failure"] = results[stopped_at]

    encoded = json.dumps(payload)
    result = ok(text(encoded), structured=json.loads(encoded))
    result["isError"] = stopped_at is not None
    return result


def _h_reference_add(ctx: Any, session: Session, args: dict) -> dict:
    """Hand this session a picture from a Library job or inline base64. See
    ``studio/modes/clay/agent/dispatch.py``'s own module docstring's references paragraph and
    ``agent_clay.tools``'s description for the full contract."""
    from . import refs as agent_refs

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
    # ``studio/modes/clay/agent/dispatch.py``'s own module docstring for why that is a deliberate
    # scope line, not an oversight.
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
    from . import refs as agent_refs

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
    png = agent_refs.bounded_png(ref.png)
    # ``bounded_png``'s own docstring: "This function does not refuse on the
    # caller's behalf -- whoever calls it is responsible for checking the
    # length of what comes back". `_h_render`'s `_over_frame_budget` already
    # does this after every `bounded_png` call it makes; this handler did
    # not (2026-09-14 audit, agents-09), latent at today's constants
    # (`MAX_IMAGE_RESULT` base64-encodes to well under `MAX_FRAME` minus
    # `RENDER_FRAME_RESERVE`) but not proof against a stored reference dense
    # enough to still be over budget after halving once. Same formula as
    # `_over_frame_budget`: base64 costs 4 bytes for every 3 of input,
    # rounded up.
    b64_len = ((len(png) + 2) // 3) * 4
    if b64_len > _protocol().MAX_FRAME - RENDER_FRAME_RESERVE:
        return fail(
            "This reference is too large to send back in one reply frame; see the log."
        )
    # Deliberately not `_json` -- this reply carries a picture, the same
    # image-carrying exclusion `clay_render` is answered with (see `_json`'s
    # own docstring): an image block has no JSON to duplicate, so `meta`
    # exists only as text here, never a second time as `structuredContent`.
    # This is the second, not the only, place that rule applies.
    return ok(text(json.dumps(meta)), image_png(png))


def _h_reference_remove(ctx: Any, session: Session, args: dict) -> dict:
    del ctx
    name = args.get("name")
    if name not in session.references:
        return fail(f"no reference named {name!r}.", field="name")
    del session.references[name]
    return _json({"removed": name})
