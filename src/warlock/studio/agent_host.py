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


@dataclass
class _Job:
    """One piece of frame-thread work, queued by the listener and drained by
    :meth:`AgentHost.pump`. ``event`` is always created, even for the
    fire-and-forget toasts :meth:`AgentHost._toast` queues, because a single
    shape here is simpler than an optional one and the cost of an unwaited
    ``Event`` is nothing."""

    run: Any
    event: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: BaseException | None = None


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
        ``CALL_TIMEOUT`` once nothing will ever service the queue again."""
        from ..mcp import protocol

        q = self._queue
        if q is None:
            return
        while True:
            try:
                job = q.get_nowait()
            except queue.Empty:
                return
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

        result, error, timed_out = self._run_on_frame(
            lambda: agent_clay.call(self.ctx, session, name, arguments)
        )
        if timed_out:
            return protocol.fail("Warlock did not answer in time; the window may be busy.")
        if error is not None:
            # ``agent_clay.call`` promises never to raise; this is the same
            # backstop ``protocol.dispatch`` keeps around its own call site,
            # for the day that promise is broken anyway.
            return protocol.fail(f"{type(error).__name__}: {error}")
        return result

    def _run_on_frame(self, run: Any, timeout: float = CALL_TIMEOUT) -> tuple[Any, Any, bool]:
        """Queue *run* for :meth:`pump` and block until it executes, until
        *timeout* passes, or until :meth:`stop` gives up on this thread's
        behalf. Returns ``(result, error, timed_out)``."""
        q = self._queue
        if q is None or self._stopped.is_set():
            return None, None, True
        job = _Job(run)
        q.put(job)
        if not job.event.wait(timeout):
            return None, None, True
        return job.result, job.error, False

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
