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
"""

from __future__ import annotations

import contextlib
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
                        tools=agent_clay.tools,
                        call=lambda name, arguments: self._call(session, name, arguments),
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

    def _call(self, session: agent_clay.Session, name: str, arguments: dict) -> dict:
        """The ``call`` callback handed to ``protocol.dispatch``. Runs on the
        listener thread but never runs the tool itself -- it queues the real
        work for :meth:`pump` and blocks here (never the frame thread) until
        an answer lands or ``CALL_TIMEOUT`` passes."""
        from ..mcp import protocol

        result, error, state = self._run_on_frame(
            lambda: agent_clay.call(self.ctx, session, name, arguments)
        )
        if error is not None:
            # ``agent_clay.call`` promises never to raise; this is the same
            # backstop ``protocol.dispatch`` keeps around its own call site,
            # for the day that promise is broken anyway.
            return protocol.fail(f"{type(error).__name__}: {error}")
        # Checked before ``state``: ``agent_clay.call`` is contracted never to
        # return ``None`` (it always answers with a result dict, even a
        # refusal), so a result in hand means the job genuinely answered --
        # including the "switched off" refusal ``_fail_pending`` stamps onto
        # a job that never ran at all. That answer outranks any state.
        if result is not None:
            return result
        if state == DROPPED:
            return protocol.fail(
                f"Warlock did not answer within {int(CALL_TIMEOUT)} seconds; the window is busy. "
                "The call was dropped before it ran, so nothing changed -- send it again."
            )
        return protocol.fail(
            f"Warlock did not answer within {int(CALL_TIMEOUT)} seconds; the call had already "
            "started and will finish on its own. Re-read clay_scene to see what it did rather "
            "than sending it again."
        )

    def _run_on_frame(self, run: Any, timeout: float = CALL_TIMEOUT) -> tuple[Any, Any, str]:
        """Queue *run* for :meth:`pump` and block until it executes, until
        *timeout* passes, or until :meth:`stop` gives up on this thread's
        behalf. Returns ``(result, error, state)``.

        The caller can see three states here. ``DONE``/``RAISED`` means the
        job ran to completion (or raised) before *timeout* elapsed, or in the
        gap after it elapsed but before this thread could claim the lock --
        either way, a result is already sitting on the job and is worth more
        than a timeout refusal. ``DROPPED`` means this thread gave up first
        and the job will never run. A result in hand always outranks the
        state; see :meth:`_call` for how a caller is expected to use that.
        """
        q = self._queue
        if q is None or self._stopped.is_set():
            # Never queued, so nothing ran -- and this is the same situation
            # ``_fail_pending`` answers, so it gets the same sentence rather
            # than a second wording for one condition.
            from ..mcp import protocol

            return protocol.fail("Warlock's agent server was switched off."), None, DROPPED
        job = _Job(run)
        q.put(job)
        if job.event.wait(timeout):
            with self._job_lock:
                return job.result, job.error, job.state
        with self._job_lock:
            # Two comparisons, one lock hold, and never across ``run()``:
            # still queued means this thread wins and the job is dropped
            # before the frame thread can claim it; anything else means the
            # frame thread got there first -- and if it has already
            # finished, the result is right here and is worth more than a
            # timeout refusal.
            if job.state == QUEUED:
                job.state = DROPPED
            return job.result, job.error, job.state

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
