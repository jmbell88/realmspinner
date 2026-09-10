"""The MCP listener that lets an external agent drive Clay through Warlock.

**One thread reads the pipe; only the frame thread ever touches a document,
GL or imgui.** ``AgentHost`` owns a :class:`~warlock.mcp.pipe.Server` and a
daemon thread that loops ``accept`` -> a per-connection ``recv_bytes` ->
``protocol.decode`` -> ``protocol.dispatch`` -> ``protocol.encode`` ->
``send_bytes``. ``protocol.dispatch`` needs a ``call(name, arguments)``
callback to actually run a tool, and running a tool means touching a
:class:`~.clay.document.Document` and, for ``clay_render``, a moderngl
context -- exactly the two things this thread must never reach for itself
(see ``CLAUDE.md``'s "one GL context" rule and ``docs/INVARIANTS.md``'s
three-thread model). So the callback this module hands ``dispatch`` does not
run the tool at all: it drops a job on a queue and blocks *this* thread on a
:class:`threading.Event` until :meth:`AgentHost.pump`, called once a frame
from ``main.py:App.frame``, dequeues it and runs ``agent_clay.call`` for
real. The listener thread waits; it never works.

**A notification gets a zero-length reply, never no reply.** ``protocol.
dispatch`` returns ``None`` for a JSON-RPC notification (``notifications/
initialized`` today), which correctly means "nothing to say" at the protocol
level -- but ``warlock.mcp.bridge`` relays exactly one frame back to its
stdout per frame it reads from stdin, and an empty write there is how it
tells the calling agent's MCP client "no reply, keep going" without the two
ends of the relay losing count of whose turn it is. Sending literally
nothing for a notification would leave the bridge blocked in its own
``recv_bytes``, waiting for a reply that the *next* real request's answer
would then be misread as. ``b""`` is a legitimate, empty frame on this
transport (length-prefixed, per ``pipe.py``'s use of ``multiprocessing.
connection``) and is exactly what keeps the two sides in step.

**`initialize`'s `instructions` text is supplied by `agent_clay`, not written
here or in `protocol.py`.** `protocol.dispatch` takes `instructions` as a
plain keyword and has no opinion about its content -- it is a stdlib leaf
that knows nothing about Clay (see its own module docstring). The text this
module passes is entirely about Clay's units and conventions ("a metre is a
metre", elements are indices into a mesh that already exists, that kind of
thing), so the module that owns the tools -- `agent_clay` -- is the one that
owns the sentence that introduces them, the same division that already puts
the tool catalogue itself in `agent_clay.tools` rather than here.

**The tab a connecting agent gets is opened before it can ask for one.**
`docs/manual/45-extending.md` promises "It opens one when it connects", so
the tab has to exist before the first ``tools/call`` a bridge sends, not
lazily on the first ``clay_add_primitive``. That still has to happen on the
frame thread (:func:`agent_clay._tab` walks ``ClayState``), so it is routed
through the exact same job queue as a tool call, calling ``agent_clay._tab``
directly rather than inventing a fake tool name for it -- ``_tab`` with
``create=True`` is already the one function that knows how to mint a
document for a session that owns nothing yet, because the two creator tools
call it themselves.

**A call the frame thread has not started is dropped when the listener stops
waiting for it; one it has started is not, and cannot be.** This overturns
the rule this module used to follow: a call that outran ``CALL_TIMEOUT``
still ran exactly once, late, because the job was already queued and running
it anyway was judged the honest thing to do. That was deliberate, and its
cost was real -- an agent that saw a timeout had no way to tell "nothing
happened" from "it happened after I stopped listening," so the only safe
recovery was to re-read ``clay_scene`` before doing anything else, and a
retry could silently repeat the same edit. The mechanism is five states on
``_Job`` (``QUEUED``, ``RUNNING``, ``DONE``, ``RAISED``, ``DROPPED``), one
lock on the host (``self._job_lock``), and a compare-and-set on each side:
:meth:`AgentHost.pump` claims ``QUEUED -> RUNNING`` before it calls
``job.run()``, and a waiter that has given up abandons ``QUEUED -> DROPPED``
instead; whichever thread gets the lock first wins, and the other sees the
state that thread left behind. A bare boolean flag cannot do this --
``queue.Queue.get_nowait`` takes a job off the queue but not out of the
listener's reach, so "check the flag, then run" is a check-then-act race
that a plain flag cannot close. The lock is held only long enough to compare
and set one field, never across ``run()`` itself, so a slow call cannot
block the listener thread waiting on it. The agent-visible consequence: the
two refusals a timeout can now produce say different things, and only one of
them is safe to retry -- see :meth:`AgentHost._call`. One more thing changes
along with it: a dropped job stays on the queue as a tombstone until a
``pump`` call pops and discards it, so ``self._queue.qsize()`` is no longer a
count of live work.

**A retry of a call that ran but whose answer never reached the peer is
recognised, not repeated.** :func:`_fingerprint` keys a per-connection store
(:class:`_Calls`) on the tool and its canonicalised arguments, never on the
JSON-RPC id the client happened to attach -- a retry is a new message with a
new id, and a driving model has no reliable way to reuse the one it made up,
so the only handle a repeated intent can be recognised by is the intent
itself. A retry of that same intent is answered from memory
(:meth:`AgentHost._replay`) exactly once, and the operation is closed the
moment it is, so a third identical call runs for real rather than matching a
replay of a replay; two calls whose answers both genuinely reached the peer
are never folded together -- placing two identical boxes on purpose places
two boxes, not one box and a memory of it. An operation whose result has
been delivered keeps no payload at all (see :class:`_Op`'s own docstring for
why that bound matters -- a remembered ``clay_render`` would otherwise pin a
reply up to ``protocol.MAX_FRAME`` for the rest of the connection), and the
store remembers at most ``MAX_REMEMBERED_CALLS`` operations, discarded whole
with the connection -- a reconnecting agent gets a fresh document, so
remembering a previous connection's calls would only be remembering answers
about a document that is gone. One tool answers about this store from
outside it: ``warlock_status`` is published by this module rather than
joining ``agent_clay.tools()``'s derived catalogue, and it is answered on
the listener thread without ever being queued -- it exists for the case
where the frame thread is busy, and queueing it would make it unanswerable
in exactly that situation.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import agent_clay

log = logging.getLogger(__name__)

#: How long ``_call`` (a tool invocation) waits for :meth:`pump` before
#: giving up and answering with a refusal. Matches the contract's own number:
#: a frame budget of 8 ms means the window is never more than a handful of
#: frames from servicing a call, so 30 s is "the app is doing something else
#: entirely" territory, not "the queue is merely busy".
CALL_TIMEOUT = 30.0

#: Bounded, on purpose -- see :meth:`AgentHost.stop`'s docstring for why an
#: unbounded join is the one thing this must never do.
STOP_JOIN_TIMEOUT = 2.0

#: The settings key this host is switched on and off by. Matches the
#: contract (``"agent_server"`` in ``ctx.settings``, bool, default False).
SETTING = "agent_server"

STATUS_TOOL = "warlock_status"
"""The one tool this module publishes itself, rather than by way of
``agent_clay.tools()``'s derived catalogue -- see :func:`_transport_tools`
for why it cannot be a ``_HANDLERS`` entry, and :meth:`AgentHost._call` for
where it is answered."""

#: A ``_Job``'s five states, read and written only under ``AgentHost.
#: _job_lock``. Plain strings, not an enum private to this module, because
#: the strings themselves are the vocabulary a caller reads -- a planned
#: status tool answers in exactly these words. ``RAISED`` is "ran and
#: raised", kept distinct from ``DONE`` because a waiter reads ``error`` in
#: one case and ``result`` in the other. ``DROPPED`` is "abandoned before it
#: ever ran, and never will".
QUEUED = "queued"
RUNNING = "running"
DONE = "done"
RAISED = "raised"
DROPPED = "dropped"

MAX_REMEMBERED_CALLS = 16
"""How many operations one connection's dedup store (:class:`_Calls`) holds
onto at once, oldest evicted first. What is actually retained is small: an
operation whose result reached the peer keeps no payload at all, only the
fact that it was delivered (see :meth:`AgentHost._call`), so the store's real
cost is only the handful of operations whose results were never delivered
because the caller had already timed out. This ceiling is what stops a
client that keeps timing out, over and over, from accumulating those without
limit."""


def _fingerprint(tool: str, args: dict) -> str:
    """A key for what was *asked*, not for who asked or which JSON-RPC id
    they happened to send it with -- a retry is a new message with a new id,
    and a driving model does not reliably repeat one it made up, so the
    handle a repeat is recognised by has to be the intent itself.

    ``sort_keys=True`` makes argument order irrelevant: a client rebuilding
    the same request has no reason to serialise its keys in the same order
    it used the first time, and two calls that mean the same thing must
    fingerprint the same regardless. ``default=str`` is a backstop for a
    value ``json.dumps`` would otherwise refuse outright, not a case this
    transport can actually hand us -- *args* already arrived as JSON, so it
    is already JSON-representable.

    The errors this can make all fall on the safe side. A fingerprint that
    differs for what was in fact the same intent (``1`` written as ``1.0``)
    only costs a missed dedup -- indistinguishable from two calls that
    really were different, which is always safe to treat as two calls. The
    unsafe direction would be two *different* intents colliding on one
    fingerprint, and an 8-byte digest over the full canonical argument text
    does not do that by accident.
    """
    canonical = json.dumps(
        {"tool": tool, "args": args}, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.blake2b(canonical.encode("utf-8"), digest_size=8).hexdigest()


@dataclass
class _Job:
    """One piece of frame-thread work, queued by the listener and drained by
    :meth:`AgentHost.pump`. ``event`` is always created, even for the
    fire-and-forget toasts :meth:`AgentHost._toast` queues, because a single
    shape here is simpler than an optional one and the cost of an unwaited
    ``Event`` is nothing.

    ``state`` is only ever read or written under ``AgentHost._job_lock`` --
    it is the single fact both the listener thread and the frame thread have
    to agree on to make a drop and a run mutually exclusive."""

    run: Any
    event: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: BaseException | None = None
    state: str = QUEUED


@dataclass
class _Op:
    """One operation remembered by a connection's :class:`_Calls` store, so
    that a retry of the same intent (:func:`_fingerprint`) can be recognised
    instead of run again.

    ``job`` is held only while this operation's result has not yet reached
    the peer. The moment it has -- a genuine reply, or the existing error
    backstop, either way something the caller actually received -- there is
    nothing left worth remembering except the fact that it happened, so
    :meth:`AgentHost._call` lets ``job`` (and ``result``) go right then. That
    is what keeps the bound in ``MAX_REMEMBERED_CALLS`` meaningful rather
    than nominal: without it, a remembered ``clay_render`` would keep
    pinning a payload up to ``protocol.MAX_FRAME`` in memory for the rest of
    the connection, for a reply the peer already has.
    """

    operation_id: str
    fingerprint: str
    tool: str
    job: Any = None  # the _Job, held only while its result is still undelivered
    state: str = QUEUED  # the last observed state; authoritative once ``job`` is None
    result: Any = None  # the reply worth replaying; None once delivered, or never kept
    delivered: bool = False

    def status(self, lock: threading.Lock) -> str:
        """This operation's state now. While ``job`` is held the frame thread
        may still be moving it, so it is read under *lock*; once the job has
        been let go the snapshot in ``state`` is the whole truth."""
        job = self.job
        if job is None:
            return self.state
        with lock:
            return job.state


class _Calls:
    """One connection's memory of the operations it has already run, keyed
    by :func:`_fingerprint` so a retried intent can be recognised without a
    client-supplied id (see :func:`_fingerprint`'s own docstring for why a
    request id cannot serve that role). Built fresh in
    :meth:`AgentHost._serve` and discarded with the connection: a
    reconnecting agent gets a fresh ``agent_clay.Session`` and therefore a
    fresh document anyway, so remembering a previous connection's calls
    would only be remembering answers about a document that is gone.

    Operation ids are minted locally (``op-1``, ``op-2``, ...) rather than as
    uuids -- the store is per connection, so global uniqueness buys nothing,
    and ``op-4`` is legible in a refusal or a log line where a uuid is not.
    """

    def __init__(self) -> None:
        # A plain dict, not an OrderedDict: Python's dicts are already
        # insertion-ordered, which is what lets mint() evict the oldest
        # entry with next(iter(self._ops)) instead of keeping a second
        # structure just to track FIFO order.
        self._ops: dict[str, _Op] = {}
        self._minted = 0

    def mint(self, tool: str, args: dict) -> _Op:
        """A fresh operation id and its intent fingerprint, remembered as
        undelivered. Evicts the oldest when the store is already at
        ``MAX_REMEMBERED_CALLS``."""
        if len(self._ops) >= MAX_REMEMBERED_CALLS:
            oldest = next(iter(self._ops))
            del self._ops[oldest]
        # Incremented before use, so the first minted id is op-1, not op-0.
        self._minted += 1
        operation_id = f"op-{self._minted}"
        op = _Op(operation_id=operation_id, fingerprint=_fingerprint(tool, args), tool=tool)
        self._ops[operation_id] = op
        return op

    def pending(self, tool: str, args: dict) -> _Op | None:
        """The newest remembered operation with the same intent whose result
        never reached the peer, or ``None``. Newest first, because an agent
        retrying means the most recent attempt, not some earlier one that
        has since been superseded."""
        fingerprint = _fingerprint(tool, args)
        for op in reversed(self._ops.values()):
            if op.fingerprint == fingerprint and not op.delivered:
                return op
        return None

    def get(self, operation_id: str) -> _Op | None:
        """The remembered operation with this id, or ``None``."""
        return self._ops.get(operation_id)

    def recent(self, limit: int) -> list[_Op]:
        """Newest first, at most *limit*."""
        return list(reversed(self._ops.values()))[:limit]


def _carries_an_image(result: dict) -> bool:
    """Whether *result*'s content includes an image block.

    Two reasons together, not either alone. An image is the largest thing
    this store could end up pinning in memory -- a ``clay_render`` reply's
    base64 payload runs up to ``protocol.MAX_FRAME`` less
    ``agent_clay.RENDER_FRAME_RESERVE`` -- and a render is a pure read of the
    document, so re-running one is strictly better for the agent than being
    handed a picture of the document as it stood whenever the original call
    ran, which by the time a retry happens may be long out of date.

    Detecting it structurally, from the result's own content blocks, is
    deliberate: a hand-kept list of "tools whose results are pictures" would
    be a second copy of a fact ``agent_clay`` already owns, and keeping two
    copies of the same fact in sync is the drift class this codebase has
    paid for repeatedly -- see ``agent_clay.tools``'s own docstring on
    derived registries.
    """
    return any(
        block.get("type") == "image"
        for block in result.get("content", [])
        if isinstance(block, dict)
    )


def _transport_tools() -> list[Any]:
    """The tools this module publishes on its own behalf, alongside
    ``agent_clay.tools()`` -- today, just :data:`STATUS_TOOL`.

    This cannot live in ``agent_clay._HANDLERS`` the way every Clay verb
    does. ``agent_clay.tools()`` is pinned to exactly ``_HANDLERS``'s keys
    (``tests/test_agent_clay.py::
    test_every_handler_has_a_tool_and_every_tool_has_a_handler``), and every
    ``_HANDLERS`` entry is a frame-thread job queued and run through
    :meth:`AgentHost._run_on_frame_job` -- the one thing ``warlock_status``
    must never be, since it exists to answer *while* the frame thread is
    busy. Publishing it here, as its own short list ``_serve`` appends to
    Clay's catalogue, keeps ``agent_clay``'s derived-catalogue claim true
    rather than quietly widened to cover a tool that holds no document.
    """
    from ..mcp import protocol

    return [
        protocol.Tool(
            name=STATUS_TOOL,
            title="Check on a call",
            description=(
                "What became of a tool call this connection made -- "
                "including one still running -- answered right away, even "
                "while Warlock is busy with something else, because this "
                "tool never waits on the frame thread the way every other "
                "one does. Pass 'operation_id' (the id a timeout refusal "
                "named) to ask about that call specifically: whether it is "
                "still running, finished, or still waiting to be picked "
                "up, and whether its result has ever reached you. Omit "
                "'operation_id' to see this connection's most recent "
                "operations instead, newest first."
            ),
            schema={
                "type": "object",
                "properties": {
                    "operation_id": {
                        "type": "string",
                        "description": (
                            "An id named by a timeout refusal, e.g. "
                            "'op-4'. Omit to list recent operations rather "
                            "than ask about one."
                        ),
                    }
                },
                "additionalProperties": False,
            },
        )
    ]


class AgentHost:
    """Owns the pipe server, the listener thread and the frame-thread queue.

    Constructed once ``ctx`` (the app's :class:`~.app_ctx.Ctx`) and the GL
    context both exist, and kept for the app's whole life whether or not an
    agent session is ever switched on -- the Settings toggle calls
    :meth:`start`/:meth:`stop` on the same instance at runtime, so there has
    to be one to call them on before the setting is ever true.
    """

    def __init__(self, ctx: Any, home: Path) -> None:
        self.ctx = ctx
        self.home = home
        self._server: Any = None
        self._thread: threading.Thread | None = None
        self._queue: queue.Queue[_Job] | None = None
        # Set *before* anything else in ``stop()`` -- every other teardown
        # step and every in-flight ``_run_on_frame`` reads this first, which
        # is what stops a call queued in the gap between "we decided to stop"
        # and "the queue is actually drained" from waiting the full
        # ``CALL_TIMEOUT`` for an answer that will never come.
        self._stopped = threading.Event()
        self._connected = False
        # The live per-connection ``Connection``, so :meth:`stop` can close it
        # out from under a listener thread blocked in that connection's own
        # ``recv_bytes`` -- closing the *Listener* (``pipe.Server.close``)
        # does not touch a ``Connection`` already accepted from it.
        self._active_conn: Any = None
        # One lock for the host's whole life -- created here, once, and never
        # reassigned by start()/stop(), unlike self._queue. A job abandoned
        # across a stop() must not be racing a lock that was replaced out
        # from under it. One host-level lock rather than one per job: the
        # contended window is two comparisons long, only one connection is
        # ever served at a time (pipe.py's v1 decision), so no more than two
        # threads can ever contend for it, and a per-job lock would be an
        # allocation on every call to protect a window nothing else shares.
        self._job_lock = threading.Lock()

    # -- lifecycle -----------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def connected(self) -> bool:
        """Whether a bridge is attached right now, for the status bar chip."""
        return self._connected

    def start(self) -> None:
        """Open the pipe and spawn the listener. Idempotent: the Settings
        switch calls this on every frame it is drawn true on a form that
        re-reads the stored value, not only on the transition, so a second
        call while one is already listening must be a no-op rather than a
        second server racing the first for the same pipe name."""
        if self.running:
            return
        from .. import __version__
        from ..mcp import pipe, protocol

        # Set once, here, per ``protocol.SERVER_VERSION``'s own docstring:
        # that module cannot import ``warlock`` itself without breaking the
        # "pure stdlib, no warlock imports" rule that keeps it testable on a
        # headless box.
        protocol.SERVER_VERSION = __version__
        server = pipe.Server(self.home)
        server.start()
        self._server = server
        self._stopped.clear()
        self._connected = False
        self._queue = queue.Queue()
        self._thread = threading.Thread(
            target=self._listen,
            name="warlock-agent-host",
            # A daemon, because a ``Listener.accept()`` blocked in this
            # thread is not guaranteed to unblock the moment ``close()`` runs
            # on every platform -- see :meth:`stop`'s own comment. A thread
            # that can outlive an orderly ``stop()`` must not be able to hold
            # the process open past it.
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Idempotent: safe on a host never started, one already stopped, and
        from ``App.teardown`` after a setup that failed before this ever ran.

        Never an unbounded join. ``pipe.Server.accept``'s own docstring
        states that a platform can leave ``Listener.accept()`` blocked past
        ``close()`` -- closing the underlying handle from under it usually
        raises, but "usually" is not a guarantee this method can lean on.
        The thread is a daemon precisely so a stuck one cannot hold the
        process open; this still bounds the wait so ``teardown`` itself
        returns promptly rather than hanging on the same platform quirk.
        """
        if self._thread is None:
            return
        self._stopped.set()
        if self._server is not None:
            self._server.close()
        conn = self._active_conn
        if conn is not None:
            # Closing the *Listener* does not touch an already-accepted
            # ``Connection``; a listener thread blocked in that connection's
            # own ``recv_bytes`` needs this instead.
            with contextlib.suppress(OSError):
                conn.close()
        self._fail_pending()
        self._thread.join(timeout=STOP_JOIN_TIMEOUT)
        self._thread = None
        self._server = None
        self._queue = None
        self._connected = False

    def _fail_pending(self) -> None:
        """Wake every call still sitting in the queue with a failure, so a
        bridge blocked in :meth:`_call` is not left waiting out the full
        ``CALL_TIMEOUT`` once nothing will ever service the queue again.

        Each job is marked ``DROPPED`` under ``_job_lock`` before its event
        is set -- ``DROPPED`` is the truthful state here (the job never ran
        and never will), and it is why the waiter still gets a failure
        result stamped onto it rather than the generic timeout refusal
        ``_run_on_frame`` would otherwise have to invent for a job with no
        result at all. A job already claimed by ``pump`` (state no longer
        ``QUEUED``) is left alone -- it is mid-``run()`` or finished, and
        this method has no business overwriting either outcome."""
        from ..mcp import protocol

        q = self._queue
        if q is None:
            return
        while True:
            try:
                job = q.get_nowait()
            except queue.Empty:
                return
            with self._job_lock:
                if job.state != QUEUED:
                    continue
                job.state = DROPPED
                job.result = protocol.fail("Warlock's agent server was switched off.")
            job.event.set()

    # -- the listener thread ---------------------------------------------------

    def _listen(self) -> None:
        """``accept()`` in a loop, one connection served at a time (the v1
        decision ``pipe.py`` documents). Never touches ``ClayDoc``, GL or
        imgui -- everything that must is a job on ``self._queue`` instead,
        drained by :meth:`pump` on the frame thread."""
        server = self._server
        while not self._stopped.is_set():
            conn = server.accept()
            if conn is None:
                # Either ``close()`` ran (the normal shutdown path) or the
                # listener never started -- both mean "stop looping", per
                # ``pipe.Server.accept``'s own contract.
                return
            if self._stopped.is_set():
                with contextlib.suppress(OSError):
                    conn.close()
                return
            self._serve(conn)

    def _serve(self, conn: Any) -> None:
        """One bridge's whole lifetime: open a session and a tab for it,
        answer every request until it disconnects, then say so."""
        from ..mcp import protocol

        session = agent_clay.Session()
        calls = _Calls()
        self._active_conn = conn
        self._connected = True
        self._toast("An agent connected.")
        # See the module docstring's tab-on-connect claim. The result is not
        # inspected: ``create=True`` cannot fail, and every tool call after
        # this resolves the tab fresh through ``agent_clay._tab`` regardless.
        self._run_on_frame(lambda: agent_clay._tab(self.ctx, session, create=True))
        try:
            while True:
                try:
                    frame_bytes = conn.recv_bytes()
                except (EOFError, OSError):
                    # The bridge went away -- not this host's problem to
                    # report, just to notice.
                    return
                try:
                    message = protocol.decode(frame_bytes)
                except ValueError as exc:
                    reply_bytes = _parse_error_frame(exc)
                    if reply_bytes is None:
                        return
                else:
                    reply = protocol.dispatch(
                        message,
                        # Clay's derived catalogue, plus the transport-level
                        # tools this module publishes itself -- see
                        # _transport_tools for why STATUS_TOOL cannot join
                        # agent_clay._HANDLERS instead.
                        tools=lambda: [*agent_clay.tools(), *_transport_tools()],
                        call=lambda name, arguments: self._call(session, calls, name, arguments),
                        instructions=agent_clay.instructions(),
                    )
                    # A notification: see the module docstring for why this
                    # is a zero-length frame and never a skipped write.
                    reply_bytes = b"" if reply is None else protocol.encode(reply)
                try:
                    conn.send_bytes(reply_bytes)
                except OSError:
                    return
        finally:
            with contextlib.suppress(OSError):
                conn.close()
            self._active_conn = None
            self._connected = False
            self._toast("The agent disconnected.")

    def _call(
        self, session: agent_clay.Session, calls: _Calls, name: str, arguments: dict
    ) -> dict:
        """The ``call`` callback handed to ``protocol.dispatch``. Runs on the
        listener thread but never runs the tool itself -- it queues the real
        work for :meth:`pump` and blocks here (never the frame thread) until
        an answer lands or ``CALL_TIMEOUT`` passes.

        Before any of that, *calls* may already hold an undelivered answer
        for this exact intent: a prior call that outran the timeout while it
        was already running (or was still queued), whose reply never reached
        the peer. :meth:`_replay` decides what to do with that -- stand in
        for running the tool again, refuse because it is still in flight, or
        say nothing and let this call run for real. Only once that check has
        cleared does a fresh operation get minted and actually run, and its
        outcome recorded: delivered and let go if the peer got an answer (a
        genuine result, or the error backstop below), kept replayable if a
        timeout refusal is about to be returned instead.
        """
        from ..mcp import protocol

        # STATUS_TOOL is answered here, before anything else, and never
        # queued: it exists precisely for the case where the frame thread
        # is busy (a call already running, or simply a backlog behind it),
        # so routing it through the same queue it is meant to answer about
        # would make it unanswerable in exactly the situation it was built
        # for. It also mints no operation of its own and never touches
        # *calls* below -- asking twice must always give a fresh answer,
        # never a dedup hit. Compare agent_clay's four reference tools,
        # which already answer without touching a document -- those still
        # run on the frame thread through the ordinary queue; this one
        # never reaches it at all.
        if name == STATUS_TOOL:
            return self._status(calls, arguments)

        prior = calls.pending(name, arguments)
        if prior is not None:
            replay = self._replay(prior)
            if replay is not None:
                return replay

        op = calls.mint(name, arguments)
        job, result, error, state = self._run_on_frame_job(
            lambda: agent_clay.call(self.ctx, session, name, arguments)
        )
        if error is not None:
            # ``agent_clay.call`` promises never to raise; this is the same
            # backstop ``protocol.dispatch`` keeps around its own call site,
            # for the day that promise is broken anyway. Delivered: the
            # refusal below is the peer's answer.
            op.state, op.job, op.result, op.delivered = state, None, None, True
            return protocol.fail(f"{type(error).__name__}: {error}")
        # Checked before ``state``: ``agent_clay.call`` is contracted never to
        # return ``None`` (it always answers with a result dict, even a
        # refusal), so a result in hand means the job genuinely answered --
        # including the "switched off" refusal ``_fail_pending`` stamps onto
        # a job that never ran at all. That answer outranks any state.
        if result is not None:
            # Delivered, and this is the "two identical boxes stay two
            # boxes" case: the intent was satisfied and the peer has the
            # answer, so nothing remains here for a later identical call to
            # match against -- it mints its own operation and runs for real.
            op.state, op.job, op.result, op.delivered = state, None, None, True
            return result
        # The result never reached the peer -- a timeout refusal is about to
        # be returned instead -- so keep this operation replayable rather
        # than delivered. *job* still refers to the live ``_Job``; a retry
        # that lands after it finishes reads the outcome straight off it
        # (see :meth:`_replay`).
        op.job, op.state, op.delivered = job, state, False
        if state == DROPPED:
            return protocol.fail(
                f"Warlock did not answer within {int(CALL_TIMEOUT)} seconds; the window is busy. "
                f"The call was dropped before it ran, as operation {op.operation_id}, so nothing "
                "changed -- send it again."
            )
        return protocol.fail(
            f"Warlock did not answer within {int(CALL_TIMEOUT)} seconds; the call had already "
            f"started and will finish on its own, as operation {op.operation_id}. Ask "
            f"{STATUS_TOOL} about it -- and once it has finished, sending that same call "
            "again hands back the result it produced rather than running it twice. Do not "
            "assume it did not happen."
        )

    def _op_row(self, op: _Op) -> dict[str, Any]:
        """The four-field summary :meth:`_status` reports for one operation.
        ``state`` always goes through :meth:`_Op.status`, never ``op.state``
        directly -- while ``op.job`` is still held the frame thread may be
        moving it, and ``op.state`` is only that field's last snapshot, per
        :class:`_Op`'s own docstring."""
        return {
            "operation_id": op.operation_id,
            "tool": op.tool,
            "state": op.status(self._job_lock),
            "delivered": op.delivered,
        }

    def _status(self, calls: _Calls, args: dict) -> dict:
        """Answers :data:`STATUS_TOOL`. Never touches the queue or the frame
        thread -- see the comment at :meth:`_call`'s own early return for
        why -- and never mints or consults an operation of its own; *calls*
        is read here, never written.

        Two shapes, chosen by whether ``operation_id`` was given:

        * Given and known -- a four-field row (see :meth:`_op_row`). If the
          job has finished (``DONE``/``RAISED``) but its result was never
          handed to the peer, a ``note`` says so and points at the recovery
          :meth:`_call` itself now offers: send the identical call again and
          it comes back as a replay rather than running twice.
        * Given but unknown to this connection's store -- ``protocol.fail``,
          naming the id and ``field="operation_id"`` (the convention every
          other refusal in this bridge follows), and saying plainly that the
          store only remembers the most recent ``MAX_REMEMBERED_CALLS``.
        * Omitted -- the recent operations, newest first, each the same
          four-field row, under a bounded list this connection's store can
          hold (:data:`MAX_REMEMBERED_CALLS`).

        The reply is always ``ok(text(json.dumps(payload)))`` -- the same
        shape ``agent_clay._json`` builds for every Clay tool, so a client
        sees one consistent result shape across the whole bridge rather than
        a special case for this one transport-level tool.
        """
        from ..mcp import protocol

        operation_id = args.get("operation_id")
        if operation_id is not None and not isinstance(operation_id, str):
            return protocol.fail("'operation_id' must be a string.", field="operation_id")

        if operation_id:
            op = calls.get(operation_id)
            if op is None:
                return protocol.fail(
                    f"No operation named {operation_id!r} on this connection -- its store "
                    f"only remembers the most recent {MAX_REMEMBERED_CALLS} calls.",
                    field="operation_id",
                )
            payload = self._op_row(op)
            if payload["state"] in (DONE, RAISED) and not payload["delivered"]:
                payload["note"] = (
                    "Its result is still waiting -- sending the same call again will hand "
                    "it back rather than run it a second time."
                )
            return protocol.ok(protocol.text(json.dumps(payload)))

        payload = {"operations": [self._op_row(op) for op in calls.recent(MAX_REMEMBERED_CALLS)]}
        q = self._queue
        if q is not None:
            # Deliberately not called "queued": a dropped job is a
            # tombstone that stays on the queue until pump() pops it (see
            # pump's own docstring), so qsize() here can overcount live
            # work by however many tombstones have not yet been reaped.
            # "queue_depth" says only what it actually is -- the number of
            # entries on the queue right now -- and never claims to be a
            # count of work still to run.
            payload["queue_depth"] = q.qsize()
        return protocol.ok(protocol.text(json.dumps(payload)))

    def _replay(self, prior: _Op) -> dict | None:
        """Whether a retry of *prior*'s intent should be answered from memory
        instead of run again. Returns the reply to send in its place, or
        ``None`` meaning "nothing to replay -- run it for real."

        *prior* is only ever handed in here undelivered (:meth:`_Calls.
        pending` filters on that), but "undelivered" was a snapshot from
        whenever it was minted or last checked -- the job behind it may have
        moved since, so the live state is read fresh via
        :meth:`_Op.status`:

        * ``RUNNING`` or ``QUEUED`` -- still in flight, and running it again
          would race the original. Refused by name, naming the operation, so
          the agent knows to wait rather than resend.
        * ``DROPPED`` -- nothing ran and nothing ever will, so a retry is
          exactly the right thing to do. The old operation is closed
          (``delivered = True``) so it cannot also match a third call, and
          ``None`` sends this one to run for real.
        * ``DONE`` or ``RAISED`` -- the job finished, but its answer was
          never delivered (or nothing worth replaying was ever kept, or it
          is a render -- see :func:`_carries_an_image`). Either replay it,
          or fall through to ``None`` and let a fresh run happen; either way
          the old operation is closed first, so a third identical call runs
          for real rather than matching a replay of a replay.
        """
        from ..mcp import protocol

        state = prior.status(self._job_lock)
        if state in (RUNNING, QUEUED):
            return protocol.fail(
                f"That same call is already {state} as operation {prior.operation_id} and its "
                "result was never delivered; sending it again would run it twice. Wait for it "
                "rather than repeating it."
            )
        if state == DROPPED:
            # Nothing ran, so a retry is exactly right -- close the old
            # operation first so it cannot match a third identical call too.
            prior.delivered = True
            return None
        # state in (DONE, RAISED): the job finished, but its own outcome was
        # never handed back to the peer. ``prior.result`` is only ever
        # populated once a job has been let go (see ``_Op``'s own
        # docstring); while ``job`` is still held, the outcome to replay is
        # whatever ended up on the job itself.
        payload = prior.result if prior.job is None else (prior.job.result if prior.job else None)
        prior.delivered = True
        prior.job = None
        prior.result = None
        if payload is None or _carries_an_image(payload):
            # Nothing worth replaying (the job raised rather than
            # answering), or a picture that is better re-taken than handed
            # back stale -- either way, run it for real.
            return None
        # Never mutate the remembered payload: it is a dict this store built
        # once and must not accumulate flags across replays, so the reply is
        # a fresh copy all the way down to structuredContent.
        content = list(payload.get("content", [])) + [
            protocol.text(
                f"(This is the remembered result of operation {prior.operation_id}, replayed "
                "because that call's answer never reached you. It was not run again.)"
            )
        ]
        structured = dict(payload.get("structuredContent") or {})
        structured["replayed"] = True
        structured["operation_id"] = prior.operation_id
        replay = dict(payload)
        replay["content"] = content
        replay["structuredContent"] = structured
        return replay

    def _run_on_frame(self, run: Any, timeout: float = CALL_TIMEOUT) -> tuple[Any, Any, str]:
        """Queue *run* for :meth:`pump` and block until it executes, until
        *timeout* passes, or until :meth:`stop` gives up on this thread's
        behalf. Returns ``(result, error, state)`` -- a thin wrapper over
        :meth:`_run_on_frame_job` that drops the ``_Job`` itself, for every
        caller (and every existing test) that only ever wanted the outcome
        and never needed to keep tracking the job afterwards.

        The caller can see three states here. ``DONE``/``RAISED`` means the
        job ran to completion (or raised) before *timeout* elapsed, or in the
        gap after it elapsed but before this thread could claim the lock --
        either way, a result is already sitting on the job and is worth more
        than a timeout refusal. ``DROPPED`` means this thread gave up first
        and the job will never run. A result in hand always outranks the
        state; see :meth:`_call` for how a caller is expected to use that.
        """
        _job, result, error, state = self._run_on_frame_job(run, timeout)
        return result, error, state

    def _run_on_frame_job(
        self, run: Any, timeout: float = CALL_TIMEOUT
    ) -> tuple[_Job | None, Any, Any, str]:
        """As :meth:`_run_on_frame`, and also hands back the ``_Job`` itself
        as its first element. :meth:`_call` needs the job, not just its
        outcome, to keep tracking an operation that is still ``RUNNING`` (or
        was ``DROPPED``) once its own wait gives up -- see :class:`_Op` and
        :meth:`_replay`. Every other caller (``_serve``'s tab-open included)
        only ever wanted the outcome and goes through :meth:`_run_on_frame`
        instead, which is why that 3-tuple wrapper still exists rather than
        every caller being made to unpack and discard a job it does not want.

        ``None`` for the job only on the switched-off early-out: nothing was
        ever queued, so there is no job to hand back.
        """
        q = self._queue
        if q is None or self._stopped.is_set():
            # Never queued, so nothing ran -- and this is the same situation
            # ``_fail_pending`` answers, so it gets the same sentence rather
            # than a second wording for one condition.
            from ..mcp import protocol

            return None, protocol.fail("Warlock's agent server was switched off."), None, DROPPED
        job = _Job(run)
        q.put(job)
        if job.event.wait(timeout):
            with self._job_lock:
                return job, job.result, job.error, job.state
        with self._job_lock:
            # Two comparisons, one lock hold, and never across ``run()``:
            # still queued means this thread wins and the job is dropped
            # before the frame thread can claim it; anything else means the
            # frame thread got there first -- and if it has already
            # finished, the result is right here and is worth more than a
            # timeout refusal.
            if job.state == QUEUED:
                job.state = DROPPED
            return job, job.result, job.error, job.state

    def _toast(self, text: str) -> None:
        """Queue a toast for :meth:`pump` to raise. ``ctx.toast`` reaches
        into imgui/state that only the frame thread may touch, so the
        listener never calls it directly; fire-and-forget, because nothing
        here is waiting on a toast landing."""
        q = self._queue
        if q is None:
            return
        q.put(_Job(lambda: self.ctx.toast(text)))

    # -- the frame thread ------------------------------------------------------

    def pump(self, budget: float = 0.008) -> None:
        """Drain queued work under a wall-clock budget. **Frame thread only**
        -- this is what actually calls ``agent_clay.call`` (and, at connect,
        ``agent_clay._tab``), the one place a document or GL may be touched.

        At least one job always runs, budget or not: a frame that is
        permanently over budget for other reasons must not starve an agent's
        calls forever, and the ceiling on any single call's own cost is
        whatever the op it runs already costs, not this number.

        A job the listener already dropped is skipped, not run: ``get_nowait``
        takes it off the queue but not out of the listener's reach, so this
        claims each job (``QUEUED -> RUNNING``) under ``_job_lock`` before
        touching it, and a job that is no longer ``QUEUED`` is a tombstone --
        the listener abandoned it first. A tombstone is woken and skipped
        without spending the one-job floor above, so it cannot starve the
        next real job behind it. One consequence follows from this:
        ``self._queue.qsize()`` is no longer a count of live work, because a
        dropped job stays on the queue as a tombstone until a ``pump`` call
        pops it.
        """
        q = self._queue
        if q is None:
            return
        deadline = time.monotonic() + budget
        ran_one = False
        while not ran_one or time.monotonic() < deadline:
            try:
                job = q.get_nowait()
            except queue.Empty:
                return
            with self._job_lock:
                if job.state != QUEUED:
                    # A tombstone: the listener gave up waiting and dropped
                    # it. Not work, so it does not spend the one-job floor,
                    # and the drain keeps going.
                    job.event.set()
                    continue
                job.state = RUNNING
            ran_one = True
            try:
                job.result = job.run()
            except Exception as exc:  # noqa: BLE001 -- one bad job must not
                # stop the drain or take the frame down with it; whoever is
                # waiting on ``job.event`` reads ``job.error`` and turns it
                # into a refusal (``_call``) or simply ignores it (``_toast``,
                # the tab-open at connect).
                job.error = exc
                log.exception("agent host: a queued call raised")
                with self._job_lock:
                    job.state = RAISED
            else:
                with self._job_lock:
                    job.state = DONE
            finally:
                job.event.set()


def _parse_error_frame(exc: ValueError) -> bytes | None:
    """A JSON-RPC parse-error reply for a frame ``protocol.decode`` refused.

    There is no request id to answer with -- decoding is what would have
    told us one -- so this builds the one JSON-RPC error shape that is
    allowed to omit it (``id: null``, code ``-32700``) by hand rather than
    through ``protocol``, which only ever builds a reply *from* a parsed
    message. ``None`` means even this could not be built, which the caller
    reads as "drop the connection" -- the last resort the module docstring
    promises for a frame malformed enough to fail here too.
    """
    from ..mcp import protocol

    try:
        return protocol.encode(
            {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"parse error: {exc}"},
            }
        )
    except Exception:
        # Silent on purpose, and this is the one place in this module that is.
        # Everything encoded here is a literal but the interpolated ``exc``
        # message, so reaching this means ``json.dumps`` refused a string --
        # which should not happen and, if it somehow does, has nowhere useful
        # to be reported: the connection is about to be dropped, which is the
        # consequence the peer actually observes, and a log line about a
        # failure to describe a malformed frame is noise a hostile peer could
        # produce on demand by sending more of them.
        return None
