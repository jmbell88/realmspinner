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
  instructions()`` returns is what a bridge actually receives.
* A call the frame thread has not started yet is **dropped**, not run late,
  once the listener stops waiting for it -- proven with no pump running at
  all, so the job is provably still ``QUEUED`` when the wait gives up. A call
  already inside ``run()`` is not dropped and cannot be: the waiter observes
  ``RUNNING`` and the job still finishes. A dropped job is a tombstone, not a
  removal -- it stays on the queue and costs ``pump`` nothing but a skip, and
  does not spend the one-job floor a real job would. And the second
  compare-and-set (a job that finishes in the gap after the wait gives up but
  before the waiter's own lock hold) answers with the result rather than a
  timeout refusal.

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
        result, _error, state = outcome["result"]
        # Nothing ran, and never will -- DROPPED is the truthful state, not
        # a generic timeout. But the waiter still gets an answer: stop()'s
        # own _fail_pending stamps a result onto the job before waking it,
        # rather than leaving _run_on_frame to invent a refusal for a job
        # with none.
        assert state == agent_host.DROPPED
        assert result is not None
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


# --- cancel-on-timeout: a call not yet started is dropped, one already ------
# --- running is not, and the two refusals say which -------------------------


def test_a_call_that_times_out_before_it_runs_is_dropped_rather_than_run_late() -> None:
    host = _bare_host()
    ran: list[str] = []

    # Nothing pumps this: Event.wait(0.0) returns False immediately, so the
    # job is still QUEUED when _run_on_frame gives up. Deterministic, no
    # sleep needed.
    result, error, state = host._run_on_frame(lambda: ran.append("x"), timeout=0.0)

    assert state == agent_host.DROPPED
    assert result is None
    assert error is None

    host.pump(budget=1.0)
    assert ran == []  # the tombstone was skipped, never run


def test_a_dropped_job_is_a_tombstone_and_does_not_spend_pumps_one_job_floor() -> None:
    host = _bare_host()
    host._run_on_frame(lambda: None, timeout=0.0)  # drop one, per the test above
    ran: list[str] = []
    host._queue.put(agent_host._Job(lambda: ran.append("good")))

    # Today the tombstone would run and consume the one-job floor, so the
    # good job behind it would not run at budget 0.
    host.pump(budget=0.0)

    assert ran == ["good"]


def test_a_call_the_frame_thread_already_started_is_not_dropped_and_still_completes() -> None:
    host = _bare_host()
    started = threading.Event()
    release = threading.Event()
    finished: list[str] = []

    def job() -> None:
        started.set()
        assert release.wait(WAIT), "release never came"
        finished.append("done")

    outcome: dict[str, object] = {}

    def waiter() -> None:
        outcome["result"] = host._run_on_frame(job, timeout=0.2)

    stop_pumping = threading.Event()

    def pump_loop() -> None:
        while not stop_pumping.is_set():
            host.pump(budget=0.01)
            time.sleep(0.005)

    pumper = threading.Thread(target=pump_loop, daemon=True)
    pumper.start()
    waiter_thread = threading.Thread(target=waiter, daemon=True)
    waiter_thread.start()
    try:
        # Bounded by WAIT, not a sleep: proceeds the moment the job is
        # genuinely inside run(), which is what makes the 0.2s wait below
        # expire while the job is RUNNING rather than still QUEUED.
        assert started.wait(WAIT), "job never started"
        waiter_thread.join(timeout=WAIT)
        assert not waiter_thread.is_alive(), "the 0.2s wait never returned"

        _result, _error, state = outcome["result"]
        assert state == agent_host.RUNNING  # not dropped -- it was already running

        release.set()
        deadline = time.monotonic() + WAIT
        while not finished and time.monotonic() < deadline:
            time.sleep(0.005)
        assert finished == ["done"]  # it still finished on its own
    finally:
        stop_pumping.set()
        pumper.join(timeout=WAIT)


def test_a_call_that_finished_in_the_gap_after_the_wait_gave_up_answers_with_its_result(
    monkeypatch,
) -> None:
    """The second compare-and-set: the frame thread claims and finishes the
    job in the gap between the waiter's ``Event.wait`` giving up and the
    waiter taking ``_job_lock`` for itself. Staged deterministically instead
    of relying on real scheduling to land in that gap."""
    host = _bare_host()
    real_job_cls = agent_host._Job

    class _NeverWaits(threading.Event):
        """Stands in for a real ``Event`` whose ``wait()`` would return
        ``True`` only once ``CALL_TIMEOUT`` had genuinely elapsed -- this one
        always reports "not yet", so ``_run_on_frame`` takes its timeout
        branch immediately rather than after a real wait."""

        def wait(self, timeout=None):  # noqa: ARG002 -- match Event.wait's shape
            return False

    def _job_factory(run):
        return real_job_cls(run, event=_NeverWaits())

    monkeypatch.setattr(agent_host, "_Job", _job_factory)

    class _RunsPumpOnPut(queue.Queue):
        """Stands in for the frame thread reaching ``pump()`` in the real gap
        between this thread's ``put()`` and its own timeout check -- a race
        that only shows up under real scheduling, forced to happen every
        time instead."""

        def put(self, item, *args, **kwargs):
            super().put(item, *args, **kwargs)
            host.pump(budget=1.0)

    host._queue = _RunsPumpOnPut()

    result, error, state = host._run_on_frame(lambda: "the real answer", timeout=0.0)

    assert (result, error, state) == ("the real answer", None, agent_host.DONE)


def test_a_dropped_call_tells_the_agent_nothing_changed(tmp_path, monkeypatch) -> None:
    host = agent_host.AgentHost(_Ctx(), tmp_path)
    monkeypatch.setattr(
        host, "_run_on_frame", lambda run, timeout=None: (None, None, agent_host.DROPPED)
    )

    result = host._call(agent_clay.Session(), "clay_scene", {})
    text = result["content"][0]["text"]

    assert "nothing changed" in text
    assert "send it again" in text


def test_a_call_that_started_tells_the_agent_to_re_read_rather_than_retry(
    tmp_path, monkeypatch
) -> None:
    host = agent_host.AgentHost(_Ctx(), tmp_path)

    monkeypatch.setattr(
        host, "_run_on_frame", lambda run, timeout=None: (None, None, agent_host.DROPPED)
    )
    dropped_text = host._call(agent_clay.Session(), "clay_scene", {})["content"][0]["text"]

    monkeypatch.setattr(
        host, "_run_on_frame", lambda run, timeout=None: (None, None, agent_host.RUNNING)
    )
    started_text = host._call(agent_clay.Session(), "clay_scene", {})["content"][0]["text"]

    assert "clay_scene" in started_text
    assert "re-read" in started_text.lower()
    assert "send it again" not in started_text
    # One sentence answering both conditions was the defect this change fixes.
    assert started_text != dropped_text


