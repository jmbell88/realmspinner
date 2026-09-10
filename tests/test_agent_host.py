"""What ``studio/agent_host.py`` promises about *threads*, never about Clay.

``AgentHost`` is the seam between the listener thread an MCP bridge talks to
and the frame thread that is the only place a ``Document`` or a GL context may
be touched (see ``docs/INVARIANTS.md``'s three-thread model). Everything below
is about that seam holding, with no real GL and no real app:

* :meth:`AgentHost.pump` runs queued work **on the thread that calls it** --
  proven by comparing ``threading.get_ident()`` from inside the queued job
  against the id captured on the test thread that called ``pump()``.
* ``pump`` always runs at least one job, budget or not, so a frame that is
  permanently over budget cannot starve an agent's calls forever.
* One job raising does not stop the drain -- the two good jobs either side of
  a bad one both still run, and the bad one's own waiter gets an ``error``
  rather than nothing at all.
* :meth:`AgentHost.stop` wakes every pending waiter itself, rather than
  leaving it to time out against ``CALL_TIMEOUT`` (30 s) -- and both ``stop``
  and ``start`` are idempotent, which the Settings toggle relies on by calling
  them every frame the checkbox is drawn, not only on the transition.
* A real round trip over a real pipe: ``pipe.connect`` against a started
  host, an ``initialize`` and a ``tools/list`` request answered while a
  background thread drives ``pump()`` the way ``main.py:App.frame`` would,
  and the one case the module docstring calls out by name -- a JSON-RPC
  *notification* gets a zero-length reply frame, never a skipped write,
  because that is what keeps the bridge relaying one frame in for one frame
  out.
* ``_serve`` hands ``agent_clay.instructions()`` to ``protocol.dispatch`` --
  a thread-boundary claim like every other bullet here, proven the same way,
  by reading it back out of a real ``initialize`` reply rather than off the
  source. This is the one place this file's title bends: it does not care
  *what* Clay's conventions say, only that whatever ``agent_clay.
  instructions()`` returns is what a bridge actually receives. One of the two
  tests exercising it is red with ``AttributeError`` until agent B lands that
  function in ``studio/agent_clay.py`` -- expected, and not a claim about
  threads breaking.

Every wait below is bounded (``conn.poll(timeout=...)`` before every
``recv_bytes``, and explicit ``join`` timeouts), so a regression that makes
the host go silent fails this file with a clear assertion rather than hanging
it past pytest's own timeout.
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from warlock.mcp import pipe, protocol
from warlock.studio import agent_clay, agent_host

#: A generous but bounded ceiling for anything that talks over the real pipe
#: in this file -- comfortably under pytest's 120 s default and comfortably
#: over what a healthy host ever takes.
WAIT = 5.0


class _Ctx:
    """Just enough of the app's ``Ctx`` for ``agent_clay._tab`` to mint a tab.

    ``settings`` is read by ``clay_mode.adopt``'s ``remember_path`` on every
    new document (a no-op for a pathless one, but the attribute access to get
    there still has to succeed); ``state.clay`` is Clay's own state, built
    lazily the first time anything asks for it.
    """

    def __init__(self) -> None:
        self.state = SimpleNamespace(clay=None)
        self.settings = SimpleNamespace()
        self.toasts: list[tuple[str, str]] = []

    def toast(self, message: str, level: str = "info") -> None:
        self.toasts.append((message, level))


def _bare_host() -> agent_host.AgentHost:
    """A host with a queue but no listener thread -- for :meth:`pump` itself.

    ``home`` is never touched unless :meth:`start` runs, so a path that does
    not exist is fine for the tests that only exercise the queue.
    """
    host = agent_host.AgentHost(_Ctx(), Path("unused-for-these-tests"))
    host._queue = queue.Queue()
    return host


def _recv(conn, timeout: float = WAIT) -> bytes:
    """One frame, or a clear assertion failure rather than an unbounded block."""
    assert conn.poll(timeout), f"no reply within {timeout}s"
    return conn.recv_bytes()


# --- pump() is the frame-thread door ------------------------------------------


def test_pump_runs_queued_work_on_the_thread_that_calls_it() -> None:
    host = _bare_host()
    seen: dict[str, int] = {}
    job = agent_host._Job(lambda: seen.__setitem__("thread", threading.get_ident()))
    host._queue.put(job)

    host.pump()

    assert job.event.is_set()
    assert seen["thread"] == threading.get_ident()


def test_pump_runs_at_least_one_job_even_when_the_budget_is_already_spent() -> None:
    """A permanently over-budget frame must not starve an agent's calls."""
    host = _bare_host()
    ran: list[str] = []
    host._queue.put(agent_host._Job(lambda: ran.append("a")))
    host._queue.put(agent_host._Job(lambda: ran.append("b")))

    host.pump(budget=0.0)

    assert ran == ["a"]  # the first ran despite a budget of zero
    assert host._queue.qsize() == 1  # the second waits for the next frame


def test_one_job_raising_does_not_stop_the_drain() -> None:
    host = _bare_host()
    ran: list[int] = []
    good1 = agent_host._Job(lambda: ran.append(1))

    def boom() -> None:
        raise RuntimeError("a queued call raised")

    bad = agent_host._Job(boom)
    good2 = agent_host._Job(lambda: ran.append(2))
    for job in (good1, bad, good2):
        host._queue.put(job)

    host.pump(budget=1.0)

    assert ran == [1, 2]  # both good jobs ran either side of the bad one
    assert good1.event.is_set() and good1.error is None
    assert good2.event.is_set() and good2.error is None
    assert bad.event.is_set()
    assert isinstance(bad.error, RuntimeError)  # the waiter gets an error, not a hang


# --- stop() wakes waiters, and both stop() and start() are idempotent --------


def test_stop_unblocks_a_pending_call_rather_than_waiting_out_call_timeout(tmp_path) -> None:
    host = agent_host.AgentHost(_Ctx(), tmp_path)
    host.start()
    try:
        outcome: dict[str, object] = {}

        def waiter() -> None:
            # Nothing ever drains this: no pump() runs while this thread
            # blocks, so the only way it returns is stop()'s own wake-up.
            outcome["result"] = host._run_on_frame(lambda: None, timeout=agent_host.CALL_TIMEOUT)

        thread = threading.Thread(target=waiter, daemon=True)
        thread.start()
        # Give the waiter a moment to actually queue its job before stopping.
        deadline = time.monotonic() + WAIT
        while host._queue.empty() and time.monotonic() < deadline:
            time.sleep(0.005)
        assert not host._queue.empty(), "the waiter's job never reached the queue"

        started = time.monotonic()
        host.stop()
        elapsed = time.monotonic() - started

        thread.join(timeout=WAIT)
        assert not thread.is_alive()
        # Bounded by STOP_JOIN_TIMEOUT, and nowhere near the 30s CALL_TIMEOUT
        # the waiter asked for -- that gap is exactly the claim under test.
        assert elapsed < agent_host.CALL_TIMEOUT / 2
        _result, _error, timed_out = outcome["result"]
        assert timed_out is False  # answered by _fail_pending, not abandoned
    finally:
        host.stop()


def test_stop_is_idempotent_on_a_host_never_started(tmp_path) -> None:
    host = agent_host.AgentHost(_Ctx(), tmp_path)
    host.stop()
    host.stop()  # must not raise


def test_stop_is_idempotent_on_a_host_already_stopped(tmp_path) -> None:
    host = agent_host.AgentHost(_Ctx(), tmp_path)
    host.start()
    host.stop()
    host.stop()  # a second stop is a no-op, not an error


def test_start_is_idempotent_while_already_running(tmp_path) -> None:
    host = agent_host.AgentHost(_Ctx(), tmp_path)
    host.start()
    try:
        first_thread = host._thread
        host.start()  # the Settings toggle calls this every frame it is drawn true
        assert host._thread is first_thread
    finally:
        host.stop()


# --- a real round trip over a real pipe --------------------------------------


def test_a_real_round_trip_answers_requests_and_gives_a_notification_zero_bytes(
    tmp_path,
) -> None:
    """Drives ``host.pump()`` from this thread the way ``App.frame`` would,
    while a real bridge connection (``pipe.connect``) talks JSON-RPC at it."""
    host = agent_host.AgentHost(_Ctx(), tmp_path)
    host.start()
    stop_pumping = threading.Event()

    def pump_loop() -> None:
        while not stop_pumping.is_set():
            host.pump(budget=0.01)
            time.sleep(0.005)

    pumper = threading.Thread(target=pump_loop, daemon=True)
    pumper.start()
    try:
        conn = pipe.connect(tmp_path)
        try:
            conn.send_bytes(protocol.encode({"jsonrpc": "2.0", "id": 1, "method": "initialize"}))
            reply = protocol.decode(_recv(conn))
            assert reply["id"] == 1
            assert reply["result"]["protocolVersion"] == protocol.PROTOCOL_VERSION
            assert reply["result"]["serverInfo"]["name"] == protocol.SERVER_NAME
            # ``_serve`` passes ``agent_clay.instructions()`` to
            # ``protocol.dispatch`` unconditionally -- see the module
            # docstring's note on why the text lives in ``agent_clay`` rather
            # than here or in ``protocol.py`` -- so a real ``initialize``
            # reply over a real bridge carries Clay's own conventions, named
            # by the word every one of them is stated in: a metre.
            #
            # **Expected to fail with ``AttributeError`` until agent B lands
            # ``agent_clay.instructions()``** in ``src/warlock/studio/
            # agent_clay.py`` -- every other assertion in this test is
            # unrelated to that landing and must keep passing regardless.
            assert "metre" in reply["result"]["instructions"]

            conn.send_bytes(protocol.encode({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}))
            reply = protocol.decode(_recv(conn))
            names = {tool["name"] for tool in reply["result"]["tools"]}
            assert "clay_scene" in names
            assert "clay_export" in names

            # The claim the module docstring names by hand: a notification (no
            # "id") gets a reply frame of zero length, never a skipped write --
            # anything else desynchronises the bridge's one-frame-in-one-out
            # relay with its own MCP client on the other side.
            conn.send_bytes(
                protocol.encode({"jsonrpc": "2.0", "method": "notifications/initialized"})
            )
            frame = _recv(conn)
            assert frame == b""
        finally:
            conn.close()
    finally:
        stop_pumping.set()
        pumper.join(timeout=WAIT)
        host.stop()


# --- initialize carries agent_clay's instructions -----------------------------


def test_the_initialize_reply_carries_agent_clay_supplied_instructions(
    tmp_path, monkeypatch
) -> None:
    """``_serve`` passes ``agent_clay.instructions()`` to ``protocol.dispatch``
    -- proven over a real pipe, the way the round trip above proves everything
    else ``_serve`` promises, rather than by reading the source. Monkeypatched
    so this passes today regardless of whether ``agent_clay.instructions`` is
    a real sentence about Clay's conventions yet -- the module docstring's
    reasoning for *why* the text lives there is a separate claim from *that*
    ``_serve`` wires it through, and this test is only the second one."""
    monkeypatch.setattr(agent_clay, "instructions", lambda: "known string", raising=False)

    host = agent_host.AgentHost(_Ctx(), tmp_path)
    host.start()
    stop_pumping = threading.Event()

    def pump_loop() -> None:
        while not stop_pumping.is_set():
            host.pump(budget=0.01)
            time.sleep(0.005)

    pumper = threading.Thread(target=pump_loop, daemon=True)
    pumper.start()
    try:
        conn = pipe.connect(tmp_path)
        try:
            conn.send_bytes(protocol.encode({"jsonrpc": "2.0", "id": 1, "method": "initialize"}))
            reply = protocol.decode(_recv(conn))
            assert reply["result"]["instructions"] == "known string"
        finally:
            conn.close()
    finally:
        stop_pumping.set()
        pumper.join(timeout=WAIT)
        host.stop()


