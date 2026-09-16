"""What ``studio/agent_host.py`` promises about *threads*, never about Clay.

``AgentHost`` is the seam between the listener thread an MCP bridge talks to
and the frame thread that is the only place a ``Document`` or a GL context may
be touched (see ``dev/INVARIANTS.md``'s three-thread model). Everything below
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
  host, a ``hello``/``catalogue``/``call`` sequence answered while a
  background thread drives ``pump()`` the way ``main.py:App.frame`` would --
  RPC v1 is the only wire format Studio's pipe answers now (see ``tests/mcp/
  test_rpc_studio.py`` for the fuller surface of that).
* ``_catalogue_payload`` hands ``agent_clay.instructions()`` straight
  through -- a thread-boundary claim like every other bullet here, proven
  the same way, by reading it back out of a real ``catalogue`` reply rather
  than off the source. This is the one place this file's title bends: it
  does not care *what* Clay's conventions say, only that whatever
  ``agent_clay.instructions()`` returns is what a bridge actually receives.
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
* A retry of a call that *did* run, but whose answer never reached the peer
  because the peer had already stopped waiting for it, is answered from
  memory instead of run a second time -- proven by driving ``_call`` twice
  with a stubbed ``agent_clay.call`` and counting how many times it actually
  ran. What makes two calls "the same" is never the JSON-RPC id a client
  attaches (a retry has no reason to reuse one it made up); it is a
  fingerprint of the tool name and its arguments, which is also why two
  calls that both genuinely got answered -- two identical boxes placed on
  purpose -- are never folded into one. A replay says so in the reply's own
  words, not only in ``structuredContent`` a client's model might never
  render, and merges its two flags into whatever ``structuredContent`` the
  tool's own answer already carried rather than starting from ``{}`` --
  proven by remembering a payload shaped the way ``agent_clay._json`` builds
  one -- and never mutates the payload it was built from. A call still
  running (or still queued) when the retry arrives is refused by name rather
  than replayed or run again; one dropped before it ever started is simply
  run for real; and a remembered render is re-run rather than handed back
  stale, because a picture costs nothing to retake and everything to keep.
* ``warlock_status`` answers what became of a call -- including one still
  running -- on the listener thread, without ever being queued, which is
  the one claim above that a busy frame thread cannot get in the way of.
  Both timeout refusals now name the operation to ask about, it is never
  itself remembered as an operation (asking twice never dedups), and it is
  published alongside ``agent_clay.tools()`` rather than living inside it.
  Its own reply carries the same ``structuredContent`` duplication every
  Clay tool's does (``ok(text(...), structured=payload)``), so it is not the
  one inconsistent result shape on the bridge.
* ``WARLOCK_AGENT_TRANSCRIPT`` gates tier two's recorder (see
  ``agent_host._record_completed_call``): unset, a completed call writes
  nothing anywhere; set, it is appended in ``agent_transcript``'s own format;
  and a path that cannot be written is logged and never reaches the caller of
  ``_call`` -- the diagnostic must not be able to fail a real tool call.
* The three transport-level refusals this module raises directly each name a
  ``recovery`` from ``agent_clay.RECOVERY`` -- the dropped-call timeout is
  ``"retry"`` (nothing ran), the in-flight refusal ``_replay`` raises when a
  retry arrives while the original is still ``RUNNING``/``QUEUED`` is
  ``"wait"`` (do not resend), and the started-call timeout is ``"read_scene"``:
  not because anything named a uid, but because what is stale is the client's
  own knowledge of whether the call happened at all, and re-reading is the
  same recovery either way. This module holds no vocabulary of its own for
  any of that -- it already imports ``agent_clay`` (for ``call``), so it
  reuses ``agent_clay.RECOVERY`` rather than inventing a second one next to
  it.

Every wait below is bounded (``conn.poll(timeout=...)`` before every
``recv_bytes``, and explicit ``join`` timeouts), so a regression that makes
the host go silent fails this file with a clear assertion rather than hanging
it past pytest's own timeout.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from warlock.mcp import pipe, rpc
from warlock.studio import agent_character, agent_clay, agent_host, agent_transcript
from warlock.studio import tasks as tasks_mod

#: A generous but bounded ceiling for anything that talks over the real pipe
#: in this file -- comfortably under pytest's 120 s default and comfortably
#: over what a healthy host ever takes.
WAIT = 5.0

#: The shortened ``_call`` timeout the tests below use when they need a job
#: to be genuinely *inside* ``run()`` at the moment the waiter gives up. It
#: is wall-clock, and the thing it races is ``pump`` on a background thread
#: claiming the job at all -- so it is deliberately several times longer
#: than the ~10 ms that loop actually needs, because this file runs in one
#: xdist worker while seven others compete for the same cores and a margin
#: that is merely sufficient on an idle box is what makes a suite flaky
#: under load. It never costs the wall-clock it names: every test using it
#: holds the job open on an ``Event`` until well past the timeout anyway.
RUNNING_WAIT = 0.5


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


def test_a_pipe_that_will_not_open_switches_the_feature_off_rather_than_raising(
    tmp_path, monkeypatch
) -> None:
    """An optional feature must not be able to stop the app from launching.

    ``start()`` used to let a failed ``pipe.Server.start`` raise, and both
    callers were bare. The Settings switch stores the setting *before* calling
    it, so one failed toggle persisted ``agent_server=True``; the next launch
    then hit the same failure inside ``main.setup_context``, which sits in the
    try whose message is "Warlock Studio could not start". A pipe another
    program was holding therefore cost the whole app, every run, with no way
    back that did not involve hand-editing settings.

    The reason is kept rather than swallowed: the Settings pane reads
    ``failure`` to say why the switch will not stay on.
    """
    from warlock.mcp import pipe as pipe_mod

    def refuse(self) -> None:
        raise PermissionError("[WinError 5] Access is denied")

    monkeypatch.setattr(pipe_mod.Server, "start", refuse)
    host = agent_host.AgentHost(_Ctx(), tmp_path)
    try:
        assert host.start() is False, "start() must report the failure, not raise it"
        assert not host.running
        assert host.failure and "Access is denied" in host.failure
    finally:
        host.stop()


def test_a_start_that_succeeds_clears_an_earlier_failure(tmp_path, monkeypatch) -> None:
    """``failure`` is what the Settings pane prints, so a stale one would
    accuse a listener that is in fact running."""
    from warlock.mcp import pipe as pipe_mod

    def refuse(self) -> None:
        raise PermissionError("nope")

    monkeypatch.setattr(pipe_mod.Server, "start", refuse)
    host = agent_host.AgentHost(_Ctx(), tmp_path)
    assert host.start() is False
    assert host.failure
    monkeypatch.undo()
    try:
        assert host.start() is True
        assert host.failure is None
    finally:
        host.stop()


# --- a real round trip over a real pipe --------------------------------------


def test_a_real_round_trip_answers_hello_catalogue_and_call(
    tmp_path,
) -> None:
    """Drives ``host.pump()`` from this thread the way ``App.frame`` would,
    while a real bridge connection (``pipe.connect``) talks RPC v1 at it --
    the wire format Studio's pipe answers exclusively now (see ``tests/mcp/
    test_rpc_studio.py`` for the fuller RPC v1 surface; this file's own
    round trip stays here so ``AgentHost``'s thread claims are proven
    against the same pipe/pump harness every other test in this file
    uses)."""
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
            conn.send_bytes(rpc.encode_request("hello", versions=[1], bridge_version="test"))
            header, body = rpc.split_reply(_recv(conn))
            assert body == b""
            assert header["rpc"] == 1
            assert isinstance(header["studio_version"], str)

            conn.send_bytes(rpc.encode_request("catalogue"))
            cat_header, body = rpc.split_reply(_recv(conn))
            assert body == b""
            names = {tool["name"] for tool in cat_header["tools"]}
            assert "clay_scene" in names
            assert "clay_export" in names
            # ``_serve``/`` _catalogue_payload`` pass ``agent_clay.
            # instructions()`` through unconditionally -- see the module
            # docstring's note on why the text lives in ``agent_clay`` --
            # so a real ``catalogue`` reply over a real pipe carries Clay's
            # own conventions, named by the word every one of them is
            # stated in: a metre.
            assert "metre" in cat_header["instructions"]

            conn.send_bytes(
                rpc.encode_request("call", tool=agent_host.STATUS_TOOL, args={})
            )
            call_header, body = rpc.split_reply(_recv(conn))
            result = json.loads(body.decode("utf-8"))
            assert result["isError"] is False
            assert call_header["hash"] == cat_header["hash"]
        finally:
            conn.close()
    finally:
        stop_pumping.set()
        pumper.join(timeout=WAIT)
        host.stop()


# --- recv_bytes is bounded, not just the frame it decodes --------------------


def test_a_pipe_peer_cannot_force_an_unbounded_recv_bytes_allocation(monkeypatch) -> None:
    """The 2026-09-16 audit (agents-04): ``_serve``'s listener-thread
    ``conn.recv_bytes()`` call omitted stdlib's own ``maxlength`` argument,
    so a peer past the pipe handshake could force this thread to fully
    buffer whatever it sent before ``rpc.decode_request``'s own length
    check (against ``rpc.MAX_FRAME``) ever got a chance to run -- the check
    both functions perform used to happen only *after* the oversized bytes
    object already existed in memory. Proven with a stub connection that
    records every ``maxlength`` it was called with, rather than a real 8 MiB
    transfer: this file's own real-pipe round trip
    (``test_a_real_round_trip_answers_hello_catalogue_and_call``) already
    proves ``_serve`` works end to end; this test only needs to prove the
    one argument is passed."""
    host = _bare_host()
    # ``_serve`` blocks in ``_run_on_frame`` for the tab-open job until
    # ``pump()`` claims it (up to ``CALL_TIMEOUT``) -- this test has no pump
    # loop running, so that call is stubbed to return immediately rather
    # than waiting out a real 30 s timeout for something this test does not
    # care about.
    monkeypatch.setattr(
        host, "_run_on_frame", lambda run, timeout=None, owner=None: (None, None, "done")
    )

    seen_maxlength: list[object] = []

    class _StubConn:
        def __init__(self) -> None:
            self._frame = rpc.encode_request("hello", versions=[1], bridge_version="test")
            self._delivered = False
            self.sent: list[bytes] = []

        def recv_bytes(self, maxlength=None):
            seen_maxlength.append(maxlength)
            if not self._delivered:
                self._delivered = True
                return self._frame
            # A second read: nothing more to send, the same "peer went
            # away" EOFError a real Connection raises once the other end
            # closes -- `_serve`'s own `except (EOFError, OSError): return`
            # already treats this as an ordinary disconnect.
            raise EOFError()

        def send_bytes(self, data: bytes) -> None:
            self.sent.append(data)

        def close(self) -> None:
            pass

    host._serve(_StubConn())

    assert seen_maxlength, "recv_bytes() was never called"
    assert all(m == rpc.MAX_FRAME for m in seen_maxlength), (
        f"recv_bytes() was called without maxlength=rpc.MAX_FRAME: {seen_maxlength}"
    )


# --- catalogue carries agent_clay's instructions ------------------------------


def test_the_catalogue_reply_carries_clay_then_character_instructions(
    tmp_path, monkeypatch
) -> None:
    """``_catalogue_payload`` passes ``agent_clay.instructions()`` and
    ``agent_character.instructions()`` through, concatenated in that order
    with a blank line between them -- proven over a real pipe, the way the
    round trip above proves everything else ``_serve`` promises, rather than
    by reading the source."""
    monkeypatch.setattr(agent_clay, "instructions", lambda: "clay says hello", raising=False)
    monkeypatch.setattr(
        agent_character, "instructions", lambda: "character says hello", raising=False
    )

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
            conn.send_bytes(rpc.encode_request("catalogue"))
            header, body = rpc.split_reply(_recv(conn))
            assert body == b""
            assert header["instructions"] == "clay says hello\n\ncharacter says hello"
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
        outcome["result"] = host._run_on_frame(job, timeout=RUNNING_WAIT)

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
        # genuinely inside run(), which is what makes the wait below
        # expire while the job is RUNNING rather than still QUEUED.
        assert started.wait(WAIT), "job never started"
        waiter_thread.join(timeout=WAIT)
        assert not waiter_thread.is_alive(), "the shortened wait never returned"

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

    def _job_factory(run, **kwargs):
        kwargs.pop("event", None)
        return real_job_cls(run, event=_NeverWaits(), **kwargs)

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
    # ``_call`` now calls the four-tuple ``_run_on_frame_job`` (it needs the
    # ``_Job`` itself to track a still-live operation for the dedup store),
    # so that is what gets monkeypatched -- ``_run_on_frame`` is now a thin
    # wrapper over it and no longer the seam ``_call`` reads through.
    monkeypatch.setattr(
        host,
        "_run_on_frame_job",
        lambda run, timeout=None, **kw: (None, None, None, agent_host.DROPPED),
    )

    result = host._call(agent_clay.Session(), agent_host._Calls(), "clay_scene", {})
    text = result["content"][0]["text"]

    assert "nothing changed" in text
    assert "send it again" in text


def test_a_call_that_started_tells_the_agent_to_ask_or_retry_rather_than_assume_nothing_happened(
    tmp_path, monkeypatch
) -> None:
    host = agent_host.AgentHost(_Ctx(), tmp_path)

    # A fresh ``_Calls()`` per call: this test is about the wording for two
    # different states, not about the dedup store, so each call is its own
    # first-ever attempt at its intent rather than a retry of the other.
    monkeypatch.setattr(
        host,
        "_run_on_frame_job",
        lambda run, timeout=None, **kw: (None, None, None, agent_host.DROPPED),
    )
    dropped_text = host._call(agent_clay.Session(), agent_host._Calls(), "clay_scene", {})[
        "content"
    ][0]["text"]

    monkeypatch.setattr(
        host,
        "_run_on_frame_job",
        lambda run, timeout=None, **kw: (None, None, None, agent_host.RUNNING),
    )
    started_text = host._call(agent_clay.Session(), agent_host._Calls(), "clay_scene", {})[
        "content"
    ][0]["text"]

    # The recovery this refusal now points at: ask warlock_status about the
    # named operation, or send the same call again to be handed its result.
    assert agent_host.STATUS_TOOL in started_text
    assert "op-1" in started_text
    assert "send it again" not in started_text
    # One sentence answering both conditions was the defect this change fixes.
    assert started_text != dropped_text


# --- dedup: a retry of a call that ran but never answered replays instead ---
# --- of running twice, and two genuinely-answered calls stay two calls -----

# The genuine, unpatched ``_run_on_frame_job``, cached the first time any
# test below asks for it -- not at import/collection time, so a codebase
# that does not have it yet fails each test individually (a clear
# AttributeError from the test that needed it) rather than failing every
# test in this file at collection. Cached rather than re-fetched every call
# so that a test calling ``_shorten_call_timeout`` twice (a shorter timeout,
# then a longer one) always rewraps the true original, never a
# already-wrapped version from its own earlier call.
_real_run_on_frame_job_cache: list = []


def _real_run_on_frame_job():
    if not _real_run_on_frame_job_cache:
        _real_run_on_frame_job_cache.append(agent_host.AgentHost._run_on_frame_job)
    return _real_run_on_frame_job_cache[0]


def _shorten_call_timeout(monkeypatch, host: agent_host.AgentHost, timeout: float) -> None:
    """Make ``host._call`` give up waiting on a running job after *timeout*
    seconds instead of the real ``CALL_TIMEOUT`` (30 s). ``_call`` has no
    timeout parameter of its own for a test to pass a smaller one through,
    so this substitutes one by wrapping the genuine ``_run_on_frame_job``
    (preserving its real job-tracking behaviour) and overriding only how
    long it waits before giving up.
    """
    real = _real_run_on_frame_job()

    def _short(self, run, timeout_arg=agent_host.CALL_TIMEOUT, **kwargs):  # noqa: ARG001
        return real(self, run, timeout=timeout, **kwargs)

    monkeypatch.setattr(agent_host.AgentHost, "_run_on_frame_job", _short)


def _pump_loop(host: agent_host.AgentHost, stop: threading.Event) -> None:
    """Drains *host*'s queue on a background thread the way ``App.frame``
    would, until *stop* is set. The same small pattern the round-trip tests
    above already use, factored out because every test below needs it."""
    while not stop.is_set():
        host.pump(budget=0.01)
        time.sleep(0.005)


# The genuine, unpatched ``_run_on_service_job`` -- the service lane's own
# counterpart to ``_real_run_on_frame_job`` above, same caching reason.
_real_run_on_service_job_cache: list = []


def _real_run_on_service_job():
    if not _real_run_on_service_job_cache:
        _real_run_on_service_job_cache.append(agent_host.AgentHost._run_on_service_job)
    return _real_run_on_service_job_cache[0]


def _shorten_service_call_timeout(monkeypatch, host: agent_host.AgentHost, timeout: float) -> None:
    """As :func:`_shorten_call_timeout`, but for the service lane's own
    ``_run_on_service_job`` -- used by the character-call tests below that
    need a job to be genuinely still running when a waiter gives up on it."""
    real = _real_run_on_service_job()

    def _short(self, run, timeout_arg=agent_host.CALL_TIMEOUT, **kwargs):  # noqa: ARG001
        return real(self, run, timeout=timeout, **kwargs)

    monkeypatch.setattr(agent_host.AgentHost, "_run_on_service_job", _short)


def _bare_host_with_service() -> agent_host.AgentHost:
    """As :func:`_bare_host`, plus a real service-lane ``TaskRunner`` -- for
    tests that exercise ``_call``/``_call_task`` against a character tool
    without a real pipe or a real ``start()``. Callers are responsible for
    ``host._service.shutdown(wait=True)`` once done, the same teardown a
    started host's own ``stop()`` performs."""
    host = _bare_host()
    host._service = tasks_mod.TaskRunner(workers=agent_host.SERVICE_WORKERS)
    return host


def test_a_retry_of_a_call_that_ran_but_never_answered_replays_instead_of_running_again(
    monkeypatch,
) -> None:
    """The central claim: a call that outran the timeout while it was
    already running finishes, its answer never reaches the peer, and a
    retry of the same intent gets that answer handed back -- without
    ``agent_clay.call`` running a second time."""
    host = _bare_host()
    calls = agent_host._Calls()
    _shorten_call_timeout(monkeypatch, host, timeout=RUNNING_WAIT)

    call_count = {"n": 0}
    started = threading.Event()
    release = threading.Event()

    def fake_call(ctx, session, name, arguments):  # noqa: ARG001
        call_count["n"] += 1
        started.set()
        assert release.wait(WAIT), "release never came"
        return {"content": [{"type": "text", "text": "the answer"}], "isError": False}

    monkeypatch.setattr(agent_clay, "call", fake_call)

    stop_pumping = threading.Event()
    pumper = threading.Thread(target=_pump_loop, args=(host, stop_pumping), daemon=True)
    pumper.start()
    session = agent_clay.Session()
    try:
        outcome: dict[str, object] = {}

        def first_call() -> None:
            outcome["first"] = host._call(session, calls, "clay_scene", {})

        first_thread = threading.Thread(target=first_call, daemon=True)
        first_thread.start()
        assert started.wait(WAIT), "the job never started running"
        first_thread.join(timeout=WAIT)
        assert not first_thread.is_alive(), "the shortened wait never returned"
        first_text = outcome["first"]["content"][0]["text"]
        # The "already started" refusal, not "dropped" -- the job was
        # genuinely running when the wait gave up.
        assert "already started" in first_text.lower()
        assert "op-1" in first_text

        # Let the real call actually finish, and wait for pump() to have
        # observed that (rather than sleeping a guessed amount).
        release.set()
        op = calls.get("op-1")
        deadline = time.monotonic() + WAIT
        while (
            op.status(host._job_lock) not in (agent_host.DONE, agent_host.RAISED)
            and time.monotonic() < deadline
        ):
            time.sleep(0.005)

        second = host._call(session, calls, "clay_scene", {})
    finally:
        stop_pumping.set()
        pumper.join(timeout=WAIT)

    assert call_count["n"] == 1  # agent_clay.call never ran a second time
    assert second["content"][0]["text"] == "the answer"


def test_a_replayed_result_says_in_words_that_it_was_not_run_again(monkeypatch) -> None:
    host = _bare_host()
    calls = agent_host._Calls()
    _shorten_call_timeout(monkeypatch, host, timeout=RUNNING_WAIT)

    started = threading.Event()
    release = threading.Event()
    remembered_payload = {"content": [{"type": "text", "text": "the answer"}], "isError": False}

    def fake_call(ctx, session, name, arguments):  # noqa: ARG001
        started.set()
        assert release.wait(WAIT), "release never came"
        return remembered_payload

    monkeypatch.setattr(agent_clay, "call", fake_call)

    stop_pumping = threading.Event()
    pumper = threading.Thread(target=_pump_loop, args=(host, stop_pumping), daemon=True)
    pumper.start()
    session = agent_clay.Session()
    try:
        first_thread = threading.Thread(
            target=lambda: host._call(session, calls, "clay_scene", {}), daemon=True
        )
        first_thread.start()
        assert started.wait(WAIT), "the job never started running"
        first_thread.join(timeout=WAIT)
        assert not first_thread.is_alive()

        release.set()
        op = calls.get("op-1")
        deadline = time.monotonic() + WAIT
        while (
            op.status(host._job_lock) not in (agent_host.DONE, agent_host.RAISED)
            and time.monotonic() < deadline
        ):
            time.sleep(0.005)

        replay = host._call(session, calls, "clay_scene", {})
    finally:
        stop_pumping.set()
        pumper.join(timeout=WAIT)

    assert replay["structuredContent"]["replayed"] is True
    assert replay["structuredContent"]["operation_id"] == "op-1"
    text_blocks = [b.get("text", "") for b in replay["content"] if b.get("type") == "text"]
    assert any("op-1" in t and "not run again" in t for t in text_blocks)
    # The remembered object itself was never mutated -- a later replay of
    # the same op must not find flags already baked into it.
    assert "replayed" not in remembered_payload


def test_a_replayed_result_keeps_the_tools_own_structured_payload_alongside_the_replay_flag(
    monkeypatch,
) -> None:
    """A replayed reply's ``structuredContent`` is a merge, not a
    replacement: before a tool's own result ever carried ``structuredContent``
    of its own, the merge in ``_replay`` started from ``{}`` and this was
    trivially true. Now it starts from the tool's real payload -- proven
    here by remembering a payload that already carries one (the shape
    ``agent_clay._json`` builds) and checking the replay still has both the
    original keys and the two replay flags, with the remembered object
    itself still untouched."""
    host = _bare_host()
    calls = agent_host._Calls()
    _shorten_call_timeout(monkeypatch, host, timeout=RUNNING_WAIT)

    started = threading.Event()
    release = threading.Event()
    remembered_payload = {
        "content": [{"type": "text", "text": '{"uid": 7, "name": "Box"}'}],
        "isError": False,
        "structuredContent": {"uid": 7, "name": "Box"},
    }
    original_structured = dict(remembered_payload["structuredContent"])

    def fake_call(ctx, session, name, arguments):  # noqa: ARG001
        started.set()
        assert release.wait(WAIT), "release never came"
        return remembered_payload

    monkeypatch.setattr(agent_clay, "call", fake_call)

    stop_pumping = threading.Event()
    pumper = threading.Thread(target=_pump_loop, args=(host, stop_pumping), daemon=True)
    pumper.start()
    session = agent_clay.Session()
    try:
        first_thread = threading.Thread(
            target=lambda: host._call(session, calls, "clay_scene", {}), daemon=True
        )
        first_thread.start()
        assert started.wait(WAIT), "the job never started running"
        first_thread.join(timeout=WAIT)
        assert not first_thread.is_alive()

        release.set()
        op = calls.get("op-1")
        deadline = time.monotonic() + WAIT
        while (
            op.status(host._job_lock) not in (agent_host.DONE, agent_host.RAISED)
            and time.monotonic() < deadline
        ):
            time.sleep(0.005)

        replay = host._call(session, calls, "clay_scene", {})
    finally:
        stop_pumping.set()
        pumper.join(timeout=WAIT)

    structured = replay["structuredContent"]
    assert structured["uid"] == 7
    assert structured["name"] == "Box"
    assert structured["replayed"] is True
    assert structured["operation_id"] == "op-1"
    # Never mutated: the remembered object's own structuredContent gained
    # no "replayed"/"operation_id" keys, and still holds its original ones.
    assert remembered_payload["structuredContent"] == original_structured
    assert "replayed" not in remembered_payload["structuredContent"]


def test_a_third_identical_call_runs_for_real_because_the_replay_was_delivered(
    monkeypatch,
) -> None:
    host = _bare_host()
    calls = agent_host._Calls()
    _shorten_call_timeout(monkeypatch, host, timeout=RUNNING_WAIT)

    call_count = {"n": 0}
    started = threading.Event()
    release = threading.Event()

    def fake_call(ctx, session, name, arguments):  # noqa: ARG001
        call_count["n"] += 1
        started.set()
        assert release.wait(WAIT), "release never came"
        return {"content": [{"type": "text", "text": "the answer"}], "isError": False}

    monkeypatch.setattr(agent_clay, "call", fake_call)

    stop_pumping = threading.Event()
    pumper = threading.Thread(target=_pump_loop, args=(host, stop_pumping), daemon=True)
    pumper.start()
    session = agent_clay.Session()
    try:
        first_thread = threading.Thread(
            target=lambda: host._call(session, calls, "clay_scene", {}), daemon=True
        )
        first_thread.start()
        assert started.wait(WAIT), "the job never started running"
        first_thread.join(timeout=WAIT)
        assert not first_thread.is_alive()

        release.set()
        op = calls.get("op-1")
        deadline = time.monotonic() + WAIT
        while (
            op.status(host._job_lock) not in (agent_host.DONE, agent_host.RAISED)
            and time.monotonic() < deadline
        ):
            time.sleep(0.005)

        host._call(session, calls, "clay_scene", {})  # the replay
        assert call_count["n"] == 1

        host._call(session, calls, "clay_scene", {})  # the third, identical call
    finally:
        stop_pumping.set()
        pumper.join(timeout=WAIT)

    assert call_count["n"] == 2  # ran for real -- the replay had already delivered


def test_two_identical_calls_that_both_answered_are_two_calls_not_one(monkeypatch) -> None:
    """Dedup must not fire on two calls that both genuinely got answered --
    an agent placing two identical boxes in a row must place two boxes, not
    one box and a memory of it."""
    host = _bare_host()
    calls = agent_host._Calls()
    _shorten_call_timeout(monkeypatch, host, timeout=2.0)

    call_count = {"n": 0}

    def fake_call(ctx, session, name, arguments):  # noqa: ARG001
        call_count["n"] += 1
        return {"content": [{"type": "text", "text": "placed"}], "isError": False}

    monkeypatch.setattr(agent_clay, "call", fake_call)

    stop_pumping = threading.Event()
    pumper = threading.Thread(target=_pump_loop, args=(host, stop_pumping), daemon=True)
    pumper.start()
    session = agent_clay.Session()
    try:
        args = {"generator": "box"}
        host._call(session, calls, "clay_add_primitive", args)
        host._call(session, calls, "clay_add_primitive", args)
    finally:
        stop_pumping.set()
        pumper.join(timeout=WAIT)

    assert call_count["n"] == 2


def test_a_retry_of_a_dropped_call_runs_rather_than_replaying_nothing(monkeypatch) -> None:
    host = _bare_host()
    calls = agent_host._Calls()
    session = agent_clay.Session()

    call_count = {"n": 0}

    def fake_call(ctx, session, name, arguments):  # noqa: ARG001
        call_count["n"] += 1
        return {"content": [{"type": "text", "text": "placed"}], "isError": False}

    monkeypatch.setattr(agent_clay, "call", fake_call)

    # Nothing pumps this call at all: Event.wait(0.0) returns False
    # immediately, so the job is still QUEUED when _call's wait gives up --
    # DROPPED, per the cancel-on-timeout behaviour this store must not
    # paper over by pretending a dropped call is something to replay.
    _shorten_call_timeout(monkeypatch, host, timeout=0.0)
    first = host._call(session, calls, "clay_scene", {})
    assert "send it again" in first["content"][0]["text"]
    assert call_count["n"] == 0  # nothing ran

    stop_pumping = threading.Event()
    pumper = threading.Thread(target=_pump_loop, args=(host, stop_pumping), daemon=True)
    pumper.start()
    try:
        _shorten_call_timeout(monkeypatch, host, timeout=2.0)  # give the retry room to run
        second = host._call(session, calls, "clay_scene", {})
    finally:
        stop_pumping.set()
        pumper.join(timeout=WAIT)

    assert call_count["n"] == 1  # the retry actually ran
    assert second["content"][0]["text"] == "placed"


def test_a_retry_while_the_original_is_still_running_is_refused_by_name(monkeypatch) -> None:
    host = _bare_host()
    calls = agent_host._Calls()
    _shorten_call_timeout(monkeypatch, host, timeout=RUNNING_WAIT)

    started = threading.Event()
    release = threading.Event()

    def fake_call(ctx, session, name, arguments):  # noqa: ARG001
        started.set()
        assert release.wait(WAIT), "release never came"
        return {"content": [{"type": "text", "text": "done"}], "isError": False}

    monkeypatch.setattr(agent_clay, "call", fake_call)

    stop_pumping = threading.Event()
    pumper = threading.Thread(target=_pump_loop, args=(host, stop_pumping), daemon=True)
    pumper.start()
    session = agent_clay.Session()
    try:
        first_thread = threading.Thread(
            target=lambda: host._call(session, calls, "clay_scene", {}), daemon=True
        )
        first_thread.start()
        assert started.wait(WAIT), "the job never started running"
        first_thread.join(timeout=WAIT)
        assert not first_thread.is_alive()

        # The job is still RUNNING (blocked on release) when this retry
        # arrives -- it must be refused, never replayed and never run again.
        retry = host._call(session, calls, "clay_scene", {})
    finally:
        release.set()
        stop_pumping.set()
        pumper.join(timeout=WAIT)

    assert retry["isError"] is True
    text = retry["content"][0]["text"]
    assert "op-1" in text
    assert "twice" in text


def test_a_remembered_render_is_re_run_rather_than_replayed(monkeypatch) -> None:
    host = _bare_host()
    calls = agent_host._Calls()
    _shorten_call_timeout(monkeypatch, host, timeout=RUNNING_WAIT)

    call_count = {"n": 0}
    started = threading.Event()
    release = threading.Event()
    render_reply = {
        "content": [{"type": "image", "data": "iVBORw0KGgo=", "mimeType": "image/png"}],
        "isError": False,
    }

    def fake_call(ctx, session, name, arguments):  # noqa: ARG001
        call_count["n"] += 1
        started.set()
        assert release.wait(WAIT), "release never came"
        return render_reply

    monkeypatch.setattr(agent_clay, "call", fake_call)

    stop_pumping = threading.Event()
    pumper = threading.Thread(target=_pump_loop, args=(host, stop_pumping), daemon=True)
    pumper.start()
    session = agent_clay.Session()
    try:
        first_thread = threading.Thread(
            target=lambda: host._call(session, calls, "clay_render", {"view": "front"}),
            daemon=True,
        )
        first_thread.start()
        assert started.wait(WAIT), "the job never started running"
        first_thread.join(timeout=WAIT)
        assert not first_thread.is_alive()

        release.set()
        op = calls.get("op-1")
        deadline = time.monotonic() + WAIT
        while (
            op.status(host._job_lock) not in (agent_host.DONE, agent_host.RAISED)
            and time.monotonic() < deadline
        ):
            time.sleep(0.005)

        host._call(session, calls, "clay_render", {"view": "front"})
    finally:
        stop_pumping.set()
        pumper.join(timeout=WAIT)

    assert call_count["n"] == 2  # re-run rather than handed back a stale picture


def test_the_same_arguments_in_a_different_key_order_are_one_intent() -> None:
    same_a = agent_host._fingerprint("clay_x", {"a": 1, "b": 2})
    same_b = agent_host._fingerprint("clay_x", {"b": 2, "a": 1})
    assert same_a == same_b

    different = agent_host._fingerprint("clay_x", {"a": 1, "b": 3})
    assert different != same_a


def test_the_remembered_calls_are_bounded_and_evict_oldest_first() -> None:
    calls = agent_host._Calls()
    total = agent_host.MAX_REMEMBERED_CALLS + 4
    for i in range(total):
        calls.mint("clay_scene", {"i": i})

    assert len(calls.recent(total)) == agent_host.MAX_REMEMBERED_CALLS
    assert calls.get("op-1") is None  # the earliest, evicted
    newest_id = f"op-{total}"
    assert calls.get(newest_id) is not None


# --- warlock_status: the transport tool answered without a frame thread -----


def test_warlock_status_answers_while_the_frame_thread_is_busy() -> None:
    """The load-bearing claim: warlock_status is answered on the listener
    thread directly, never queued for pump() -- proven by staging a job
    genuinely inside run() and blocked, with a second job queued behind it
    that nothing drains, and confirming the status call answers promptly
    anyway, without that trailing job ever being touched."""
    host = _bare_host()
    started = threading.Event()
    release = threading.Event()

    def blocking_job():
        started.set()
        assert release.wait(WAIT), "release never came"
        return "the blocked job's own result"

    host._queue.put(agent_host._Job(blocking_job))
    # One pump() call, on its own thread, claims and blocks inside this job.
    pumper = threading.Thread(target=host.pump, kwargs={"budget": 1.0}, daemon=True)
    pumper.start()
    try:
        assert started.wait(WAIT), "the blocking job never started running"

        # Queued behind it, and left untouched below -- proof that the
        # queue was never drained to answer warlock_status.
        trailing = agent_host._Job(lambda: "never reached")
        host._queue.put(trailing)

        calls = agent_host._Calls()
        began = time.monotonic()
        result = host._call(agent_clay.Session(), calls, agent_host.STATUS_TOOL, {})
        elapsed = time.monotonic() - began

        assert result["isError"] is False
        assert elapsed < 2.0, "warlock_status waited on the busy frame thread"
        assert host._queue.qsize() == 1  # the trailing job is still sitting there
        assert not trailing.event.is_set()  # ... and was never run
    finally:
        release.set()
        pumper.join(timeout=WAIT)


def test_warlock_status_reports_an_operation_that_ran_but_never_delivered(monkeypatch) -> None:
    host = _bare_host()
    calls = agent_host._Calls()
    _shorten_call_timeout(monkeypatch, host, timeout=RUNNING_WAIT)

    started = threading.Event()
    release = threading.Event()

    def fake_call(ctx, session, name, arguments):  # noqa: ARG001
        started.set()
        assert release.wait(WAIT), "release never came"
        return {"content": [{"type": "text", "text": "the answer"}], "isError": False}

    monkeypatch.setattr(agent_clay, "call", fake_call)

    stop_pumping = threading.Event()
    pumper = threading.Thread(target=_pump_loop, args=(host, stop_pumping), daemon=True)
    pumper.start()
    session = agent_clay.Session()
    try:
        first_thread = threading.Thread(
            target=lambda: host._call(session, calls, "clay_scene", {}), daemon=True
        )
        first_thread.start()
        assert started.wait(WAIT), "the job never started running"
        first_thread.join(timeout=WAIT)
        assert not first_thread.is_alive(), "the shortened wait never returned"

        release.set()
        op = calls.get("op-1")
        deadline = time.monotonic() + WAIT
        while (
            op.status(host._job_lock) not in (agent_host.DONE, agent_host.RAISED)
            and time.monotonic() < deadline
        ):
            time.sleep(0.005)

        status = host._call(session, calls, agent_host.STATUS_TOOL, {"operation_id": "op-1"})
    finally:
        stop_pumping.set()
        pumper.join(timeout=WAIT)

    assert status["isError"] is False
    payload = json.loads(status["content"][0]["text"])
    assert payload["operation_id"] == "op-1"
    assert payload["tool"] == "clay_scene"
    assert payload["delivered"] is False
    assert payload["state"] in (agent_host.DONE, agent_host.RAISED)
    assert "waiting" in payload["note"].lower()
    assert "again" in payload["note"].lower()


def test_warlock_status_refuses_an_operation_id_it_never_minted() -> None:
    host = _bare_host()
    calls = agent_host._Calls()

    result = host._call(
        agent_clay.Session(), calls, agent_host.STATUS_TOOL, {"operation_id": "op-999"}
    )

    assert result["isError"] is True
    assert "op-999" in result["content"][0]["text"]
    assert result["structuredContent"]["field"] == "operation_id"


def test_warlock_status_is_offered_to_a_bridge_but_is_not_one_of_clays_tools() -> None:
    transport_names = {t.name for t in agent_host._transport_tools()}
    clay_names = {t.name for t in agent_clay.tools()}

    assert agent_host.STATUS_TOOL in transport_names
    assert agent_host.STATUS_TOOL not in clay_names
    assert agent_host.STATUS_TOOL not in agent_clay._HANDLERS
    # Every name a bridge publishes, across the two catalogues, is unique.
    assert not (transport_names & clay_names)


def test_warlock_status_is_never_remembered_as_an_operation() -> None:
    host = _bare_host()
    calls = agent_host._Calls()

    host._call(agent_clay.Session(), calls, agent_host.STATUS_TOOL, {})
    host._call(agent_clay.Session(), calls, agent_host.STATUS_TOOL, {})

    assert calls.recent(agent_host.MAX_REMEMBERED_CALLS) == []


def test_a_timeout_refusal_names_the_operation_to_ask_about(tmp_path, monkeypatch) -> None:
    host = agent_host.AgentHost(_Ctx(), tmp_path)

    monkeypatch.setattr(
        host,
        "_run_on_frame_job",
        lambda run, timeout=None, **kw: (None, None, None, agent_host.DROPPED),
    )
    dropped_text = host._call(agent_clay.Session(), agent_host._Calls(), "clay_scene", {})[
        "content"
    ][0]["text"]
    assert "op-1" in dropped_text

    monkeypatch.setattr(
        host,
        "_run_on_frame_job",
        lambda run, timeout=None, **kw: (None, None, None, agent_host.RUNNING),
    )
    started_text = host._call(agent_clay.Session(), agent_host._Calls(), "clay_scene", {})[
        "content"
    ][0]["text"]
    assert "op-1" in started_text
    assert agent_host.STATUS_TOOL in started_text


# --- recovery: the three transport-level refusals each name one -------------


def test_the_transport_refusals_name_their_recovery(tmp_path, monkeypatch) -> None:
    """The dropped-call and started-call timeout refusals, staged the same
    way ``test_a_timeout_refusal_names_the_operation_to_ask_about`` above
    already does; the in-flight refusal needs a job genuinely ``RUNNING``,
    staged the same real-thread way the dedup tests earlier in this file
    do."""
    host = agent_host.AgentHost(_Ctx(), tmp_path)

    monkeypatch.setattr(
        host,
        "_run_on_frame_job",
        lambda run, timeout=None, **kw: (None, None, None, agent_host.DROPPED),
    )
    dropped = host._call(agent_clay.Session(), agent_host._Calls(), "clay_scene", {})
    assert (dropped.get("structuredContent") or {}).get("recovery") == "retry"

    monkeypatch.setattr(
        host,
        "_run_on_frame_job",
        lambda run, timeout=None, **kw: (None, None, None, agent_host.RUNNING),
    )
    started = host._call(agent_clay.Session(), agent_host._Calls(), "clay_scene", {})
    assert (started.get("structuredContent") or {}).get("recovery") == "read_scene"

    host2 = _bare_host()
    calls = agent_host._Calls()
    _shorten_call_timeout(monkeypatch, host2, timeout=RUNNING_WAIT)

    started_evt = threading.Event()
    release = threading.Event()

    def fake_call(ctx, session, name, arguments):  # noqa: ARG001
        started_evt.set()
        assert release.wait(WAIT), "release never came"
        return {"content": [{"type": "text", "text": "done"}], "isError": False}

    monkeypatch.setattr(agent_clay, "call", fake_call)

    stop_pumping = threading.Event()
    pumper = threading.Thread(target=_pump_loop, args=(host2, stop_pumping), daemon=True)
    pumper.start()
    session = agent_clay.Session()
    try:
        first_thread = threading.Thread(
            target=lambda: host2._call(session, calls, "clay_scene", {}), daemon=True
        )
        first_thread.start()
        assert started_evt.wait(WAIT), "the job never started running"
        first_thread.join(timeout=WAIT)
        assert not first_thread.is_alive()

        # Still RUNNING (blocked on release) when this retry arrives -- the
        # in-flight refusal, not the started-call timeout above.
        in_flight = host2._call(session, calls, "clay_scene", {})
    finally:
        release.set()
        stop_pumping.set()
        pumper.join(timeout=WAIT)

    assert in_flight["isError"] is True
    assert (in_flight.get("structuredContent") or {}).get("recovery") == "wait"


# --- WARLOCK_AGENT_TRANSCRIPT: tier two's recorder ---------------------------


def test_no_transcript_is_recorded_when_the_env_var_is_unset(monkeypatch) -> None:
    host = _bare_host()
    monkeypatch.delenv(agent_host.TRANSCRIPT_ENV, raising=False)
    recorded: list = []
    monkeypatch.setattr(agent_transcript, "record", lambda *a, **k: recorded.append((a, k)))
    result = {"content": [], "isError": False}
    monkeypatch.setattr(
        host,
        "_run_on_frame_job",
        lambda run, timeout=None, **kw: (None, result, None, agent_host.DONE),
    )

    host._call(agent_clay.Session(), agent_host._Calls(), "clay_scene", {})

    assert recorded == []  # off unless the variable names a path


def test_a_completed_call_is_recorded_when_the_env_var_is_set(tmp_path, monkeypatch) -> None:
    host = _bare_host()
    transcript = tmp_path / "subject.jsonl"
    monkeypatch.setenv(agent_host.TRANSCRIPT_ENV, str(transcript))
    result = {
        "content": [{"type": "text", "text": '{"uid": 7}'}],
        "isError": False,
        "structuredContent": {"uid": 7},
    }
    monkeypatch.setattr(
        host,
        "_run_on_frame_job",
        lambda run, timeout=None, **kw: (None, result, None, agent_host.DONE),
    )

    host._call(
        agent_clay.Session(), agent_host._Calls(), "clay_add_primitive", {"generator": "box"}
    )

    lines = [
        json.loads(line)
        for line in transcript.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # Exactly tier one's own format -- see agent_transcript.record's docstring
    # on why that agreement is the point.
    assert lines == [
        {"tool": "clay_add_primitive", "arguments": {"generator": "box"}, "ok": True, "made": [7]}
    ]


def test_warlock_status_is_never_recorded(tmp_path, monkeypatch) -> None:
    """``warlock_status`` answers about the dedup store, not a Clay document
    -- see :func:`agent_host._transport_tools`'s own docstring for why it is
    not one of ``agent_clay.tools()`` at all -- and tier one's replay only
    knows how to run a tool through ``agent_clay.call``, so a recorded
    ``warlock_status`` line would be a transcript entry tier one could never
    meaningfully replay against a corpus subject. It is answered and
    returned before :meth:`AgentHost._call` ever reaches the recording
    point, which this proves by checking that nothing was written at all."""
    host = _bare_host()
    transcript = tmp_path / "subject.jsonl"
    monkeypatch.setenv(agent_host.TRANSCRIPT_ENV, str(transcript))

    host._call(agent_clay.Session(), agent_host._Calls(), agent_host.STATUS_TOOL, {})

    assert not transcript.exists()


def test_a_transcript_that_cannot_be_written_does_not_break_the_call(
    tmp_path, monkeypatch
) -> None:
    host = _bare_host()
    # A file, not a directory, so agent_transcript.record's own
    # ``path.parent.mkdir(parents=True, exist_ok=True)`` raises FileExistsError
    # -- an OSError subtype -- when it tries to create a "directory" that
    # already exists as something else.
    blocked = tmp_path / "not_a_directory"
    blocked.write_text("x", encoding="utf-8")
    monkeypatch.setenv(agent_host.TRANSCRIPT_ENV, str(blocked / "subject.jsonl"))
    result = {"content": [], "isError": False}
    monkeypatch.setattr(
        host,
        "_run_on_frame_job",
        lambda run, timeout=None, **kw: (None, result, None, agent_host.DONE),
    )

    returned = host._call(agent_clay.Session(), agent_host._Calls(), "clay_scene", {})

    # The call's own answer, unaffected by the transcript write failing --
    # an agent session must not die because a diagnostic path was unwritable.
    assert returned is result




# =============================================================================
# Task mode: `call` with `wait: false`, RPC v1 `status`/`cancel` --
# the MCP Tasks extension's own vocabulary, mapped onto `_Job`'s five states.
# =============================================================================


def test_task_status_maps_each_job_state() -> None:
    assert agent_host.TASK_STATUS[agent_host.QUEUED] == "working"
    assert agent_host.TASK_STATUS[agent_host.RUNNING] == "working"
    assert agent_host.TASK_STATUS[agent_host.DONE] == "completed"
    assert agent_host.TASK_STATUS[agent_host.RAISED] == "failed"
    assert agent_host.TASK_STATUS[agent_host.DROPPED] == "cancelled"


def test_call_task_mints_an_operation_and_returns_immediately_without_a_result() -> None:
    """The listener thread must never block for a task-mode call -- proven
    here by never running pump() at all and still getting a header back
    with no body, status "working"."""
    host = _bare_host()
    calls = agent_host._Calls()
    session = agent_clay.Session()

    header = host._call_task(session, calls, "clay_scene", {})

    assert header["status"] == "working"
    op = calls.get(header["operation_id"])
    assert op is not None
    assert op.task_mode is True
    assert host._queue.qsize() == 1  # queued, not run


def test_a_task_mode_warlock_status_call_answers_immediately_instead_of_queuing_behind_the_frame_thread() -> (  # noqa: E501
    None
):
    """The 2026-09-14 audit (agents-04): a client that negotiated the MCP
    Tasks extension has *every* tools/call routed through `_call_task`
    (`protocol.bridge_dispatch`'s modern-era branch does not special-case
    any one tool name), `warlock_status` included -- so before this fix,
    `_call_task` queued it as an ordinary `agent_clay` frame job, which does
    not know that name and answers "no such tool", even though `_call`
    already answers it synchronously without ever touching the frame thread
    (`test_warlock_status_answers_while_the_frame_thread_is_busy` above).
    Proven here the same way
    `test_call_task_mints_an_operation_and_returns_immediately_without_a_result`
    proves the ordinary queuing case: never pump() at all, and the answer
    must already be there."""
    host = _bare_host()
    calls = agent_host._Calls()
    session = agent_clay.Session()

    header = host._call_task(session, calls, agent_host.STATUS_TOOL, {})

    assert header["status"] == "completed"
    assert host._queue.qsize() == 0  # never queued onto the frame thread

    reply = host._task_status(calls, header["operation_id"])
    status_header, body = rpc.split_reply(reply)
    assert status_header["status"] == "completed"
    result = json.loads(body)
    assert result.get("isError") is not True
    assert "no such tool" not in json.dumps(result)


def test_a_task_mode_warlock_status_calls_own_arguments_are_dropped_once_answered() -> None:
    """The 2026-09-16 audit (agents-05): `_Op.args`'s own docstring promises
    task-mode arguments are "Dropped (set back to None) the moment they are
    used" -- the only code that ever clears it is `_task_status`'s
    `if job is not None:` branch, but a `STATUS_TOOL` task-mode operation's
    `job` is always `None` from the moment `_call_task` mints it (answered
    synchronously, never queued -- see the test just above), so that branch
    never ran for it and `op.args` stayed alive for as long as the operation
    survived eviction, contradicting the class's own contract."""
    host = _bare_host()
    calls = agent_host._Calls()
    session = agent_clay.Session()

    header = host._call_task(session, calls, agent_host.STATUS_TOOL, {"operation_id": "x"})

    op = calls.get(header["operation_id"])
    assert op is not None
    assert op.args is None


def test_task_mode_call_is_exempt_from_call_timeout(monkeypatch) -> None:
    """Patch CALL_TIMEOUT tiny, wait well past it with nothing pumping,
    then pump late -- a task-mode call must still complete, because
    _call_task never waited on it in the first place."""
    monkeypatch.setattr(agent_host, "CALL_TIMEOUT", 0.01)
    host = _bare_host()
    calls = agent_host._Calls()
    session = agent_clay.Session()

    def fake_call(ctx, session, name, arguments):  # noqa: ARG001
        return {"content": [{"type": "text", "text": "late but fine"}], "isError": False}

    monkeypatch.setattr(agent_clay, "call", fake_call)

    header = host._call_task(session, calls, "clay_scene", {})
    time.sleep(0.2)  # well past the shrunken CALL_TIMEOUT; nothing is pumping

    host.pump(budget=1.0)  # only now does the job actually run

    body_reply = host._task_status(calls, header["operation_id"])
    status_header, body = rpc.split_reply(body_reply)
    assert status_header["status"] == "completed"
    assert json.loads(body)["content"][0]["text"] == "late but fine"


def test_task_status_reports_completed_and_splices_the_result_body(monkeypatch) -> None:
    host = _bare_host()
    calls = agent_host._Calls()
    session = agent_clay.Session()
    payload = {"content": [{"type": "text", "text": "the answer"}], "isError": False}
    monkeypatch.setattr(agent_clay, "call", lambda ctx, session, name, arguments: payload)

    header = host._call_task(session, calls, "clay_scene", {})
    host.pump(budget=1.0)

    reply = host._task_status(calls, header["operation_id"])
    status_header, body = rpc.split_reply(reply)
    assert status_header["status"] == "completed"
    assert json.loads(body) == payload


def test_task_status_reports_working_before_pump_runs_it() -> None:
    host = _bare_host()
    calls = agent_host._Calls()
    session = agent_clay.Session()
    header = host._call_task(session, calls, "clay_scene", {})

    reply = host._task_status(calls, header["operation_id"])
    status_header, body = rpc.split_reply(reply)
    assert status_header["status"] == "working"
    assert body == b""


def test_task_status_reports_failed_for_a_job_that_raised(monkeypatch) -> None:
    host = _bare_host()
    calls = agent_host._Calls()
    session = agent_clay.Session()

    def boom(ctx, session, name, arguments):  # noqa: ARG001
        raise RuntimeError("kaboom")

    monkeypatch.setattr(agent_clay, "call", boom)

    header = host._call_task(session, calls, "clay_scene", {})
    host.pump(budget=1.0)

    reply = host._task_status(calls, header["operation_id"])
    status_header, body = rpc.split_reply(reply)
    assert status_header["status"] == "failed"
    assert b"kaboom" in body


def test_task_status_unknown_operation_id_is_not_found() -> None:
    host = _bare_host()
    calls = agent_host._Calls()
    reply = host._task_status(calls, "op-does-not-exist")
    header, _body = rpc.split_reply(reply)
    assert header["error"]["code"] == "not_found"


def test_cancel_a_queued_task_drops_it_before_it_runs() -> None:
    host = _bare_host()
    calls = agent_host._Calls()
    session = agent_clay.Session()
    header = host._call_task(session, calls, "clay_scene", {})

    reply = host._cancel_task(calls, header["operation_id"])
    status_header, _body = rpc.split_reply(reply)
    assert status_header["status"] == "cancelled"

    host.pump(budget=1.0)  # the tombstone is drained, never runs the tool
    status_reply = host._task_status(calls, header["operation_id"])
    status_header, _body = rpc.split_reply(status_reply)
    assert status_header["status"] == "cancelled"


def test_cancel_a_running_task_is_not_honored(monkeypatch) -> None:
    """A started job cannot be cancelled -- the compare-and-set only ever
    succeeds queued -> dropped. Cancellation is cooperative, per the MCP
    Tasks extension's own semantics."""
    host = _bare_host()
    calls = agent_host._Calls()
    session = agent_clay.Session()
    started = threading.Event()
    release = threading.Event()

    def fake_call(ctx, session, name, arguments):  # noqa: ARG001
        started.set()
        assert release.wait(WAIT), "release never came"
        return {"content": [], "isError": False}

    monkeypatch.setattr(agent_clay, "call", fake_call)

    header = host._call_task(session, calls, "clay_scene", {})
    stop_pumping = threading.Event()
    pumper = threading.Thread(target=_pump_loop, args=(host, stop_pumping), daemon=True)
    pumper.start()
    try:
        assert started.wait(WAIT), "the job never started running"
        reply = host._cancel_task(calls, header["operation_id"])
        status_header, _body = rpc.split_reply(reply)
        assert status_header["status"] == "working"  # still running, not cancelled
    finally:
        release.set()
        stop_pumping.set()
        pumper.join(timeout=WAIT)


def test_a_working_task_survives_max_remembered_calls_eviction_pressure() -> None:
    """A task-mode operation still queued or running must never be evicted
    by _Calls.mint's bound, even under sustained pressure -- unlike an
    ordinary (non-task) operation, which this store has always been willing
    to evict regardless of state."""
    host = _bare_host()
    calls = agent_host._Calls()
    session = agent_clay.Session()

    # Never pumped: this job stays QUEUED (working) for the whole test.
    working = host._call_task(session, calls, "clay_scene", {"tag": "protect-me"})
    protected_id = working["operation_id"]

    for i in range(agent_host.MAX_REMEMBERED_CALLS + 8):
        calls.mint("clay_scene", {"i": i})

    assert calls.get(protected_id) is not None
    assert len(calls.recent(agent_host.MAX_REMEMBERED_CALLS + 8)) == agent_host.MAX_REMEMBERED_CALLS


def test_a_completed_tasks_result_is_retained_until_first_fetch_then_evictable(
    monkeypatch,
) -> None:
    host = _bare_host()
    calls = agent_host._Calls()
    session = agent_clay.Session()
    monkeypatch.setattr(
        agent_clay, "call", lambda ctx, session, name, arguments: {"content": [], "isError": False}
    )

    header = host._call_task(session, calls, "clay_scene", {})
    host.pump(budget=1.0)
    op = calls.get(header["operation_id"])
    assert op.fetched is False

    # Not yet fetched: protected from eviction even though it is terminal.
    for i in range(agent_host.MAX_REMEMBERED_CALLS + 4):
        calls.mint("clay_scene", {"i": i})
    assert calls.get(header["operation_id"]) is not None

    host._task_status(calls, header["operation_id"])
    op = calls.get(header["operation_id"])
    assert op.fetched is True

    # Fetched once: now an ordinary evictable operation.
    for i in range(agent_host.MAX_REMEMBERED_CALLS + 4):
        calls.mint("clay_scene", {"j": i})
    assert calls.get(header["operation_id"]) is None


def test_task_mode_call_does_not_participate_in_fingerprint_dedup(monkeypatch) -> None:
    """A task-mode operation is never matched by _Calls.pending -- two
    identical wait:false calls each get their own operation id, and a
    matching blocking call does not treat a pending task as its own retry
    either."""
    host = _bare_host()
    calls = agent_host._Calls()
    session = agent_clay.Session()

    first = host._call_task(session, calls, "clay_scene", {"a": 1})
    second = host._call_task(session, calls, "clay_scene", {"a": 1})
    assert first["operation_id"] != second["operation_id"]
    assert calls.pending("clay_scene", {"a": 1}) is None


def test_a_task_mode_call_is_recorded_in_the_transcript_exactly_once(tmp_path, monkeypatch) -> None:
    transcript = tmp_path / "transcript.jsonl"
    monkeypatch.setenv(agent_host.TRANSCRIPT_ENV, str(transcript))
    host = _bare_host()
    calls = agent_host._Calls()
    session = agent_clay.Session()
    monkeypatch.setattr(
        agent_clay, "call", lambda ctx, session, name, arguments: {"content": [], "isError": False}
    )

    header = host._call_task(session, calls, "clay_scene", {"a": 1})
    host.pump(budget=1.0)

    host._task_status(calls, header["operation_id"])
    host._task_status(calls, header["operation_id"])  # a second fetch: no second recording

    lines = transcript.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_a_dropped_task_is_never_recorded(tmp_path, monkeypatch) -> None:
    transcript = tmp_path / "transcript.jsonl"
    monkeypatch.setenv(agent_host.TRANSCRIPT_ENV, str(transcript))
    host = _bare_host()
    calls = agent_host._Calls()
    session = agent_clay.Session()

    header = host._call_task(session, calls, "clay_scene", {})
    host._cancel_task(calls, header["operation_id"])
    host.pump(budget=1.0)
    host._task_status(calls, header["operation_id"])

    assert not transcript.exists()


# --- the service lane: character tool calls never touch the frame thread ----
# --- or pump(), and share the same job/dedup/stop machinery Clay already ----
# --- proved above -------------------------------------------------------


def test_a_character_call_runs_on_a_service_thread_never_the_frame_thread(monkeypatch) -> None:
    """Routing in ``_call``: a tool named in ``agent_character.HANDLERS``
    runs on the service lane's own ``TaskRunner`` pool, never inline on
    whatever thread called ``_call`` and never on a thread that also drains
    ``self._queue`` -- proven by comparing thread identities, the same way
    this file's very first test proves ``pump`` runs a job on the thread
    that calls it."""
    host = _bare_host_with_service()
    calls = agent_host._Calls()
    session = agent_clay.Session()
    seen: dict[str, int] = {}

    def fake_call(svc, char_session, name, arguments):  # noqa: ARG001
        seen["thread"] = threading.get_ident()
        return {"content": [], "isError": False}

    monkeypatch.setattr(agent_character, "HANDLERS", {"character_probe": fake_call})
    monkeypatch.setattr(agent_character, "call", fake_call)

    try:
        result = host._call(session, calls, "character_probe", {})
    finally:
        host._service.shutdown(wait=True)

    assert result["isError"] is False
    assert seen["thread"] != threading.get_ident()  # not the caller's own thread
    assert host._queue.qsize() == 0  # and it never touched the frame queue either


def test_a_character_call_completes_while_pump_is_never_called(monkeypatch) -> None:
    """A character call needs no frame-thread drain at all: ``_call`` blocks
    the caller until the service lane answers, and ``self._queue`` (the
    frame lane ``pump`` drains) is never touched."""
    host = _bare_host_with_service()
    calls = agent_host._Calls()
    session = agent_clay.Session()

    monkeypatch.setattr(agent_character, "HANDLERS", {"character_probe": None})
    monkeypatch.setattr(
        agent_character,
        "call",
        lambda svc, char_session, name, arguments: {  # noqa: ARG005
            "content": [{"type": "text", "text": "ok"}],
            "isError": False,
        },
    )

    try:
        result = host._call(session, calls, "character_probe", {})
    finally:
        host._service.shutdown(wait=True)

    assert result["content"][0]["text"] == "ok"
    assert host._queue.qsize() == 0  # pump() was never needed to answer this


def test_a_blocked_character_call_never_holds_pump(monkeypatch) -> None:
    """A character call stuck mid-run must not be able to hold up the frame
    lane -- an ordinary Clay-style frame job queued behind it still runs,
    and ``pump`` still returns promptly, because the two lanes share no
    lock across ``run()`` and no queue."""
    host = _bare_host_with_service()
    calls = agent_host._Calls()
    session = agent_clay.Session()
    started = threading.Event()
    release = threading.Event()

    def slow_call(svc, char_session, name, arguments):  # noqa: ARG001
        started.set()
        assert release.wait(WAIT), "release never came"
        return {"content": [], "isError": False}

    monkeypatch.setattr(agent_character, "HANDLERS", {"character_probe": slow_call})
    monkeypatch.setattr(agent_character, "call", slow_call)

    outcome: dict[str, object] = {}

    def runner() -> None:
        outcome["result"] = host._call(session, calls, "character_probe", {})

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    try:
        assert started.wait(WAIT), "the character call never started"

        ran: list[str] = []
        host._queue.put(agent_host._Job(lambda: ran.append("frame-job")))
        frame_start = time.monotonic()
        host.pump(budget=0.01)
        frame_elapsed = time.monotonic() - frame_start

        assert ran == ["frame-job"]
        assert frame_elapsed < 1.0, "pump() waited on the stuck service call"
    finally:
        release.set()
        thread.join(timeout=WAIT)
        host._service.shutdown(wait=True)

    assert outcome["result"]["isError"] is False


def test_warlock_status_reports_a_running_character_call(monkeypatch) -> None:
    """``warlock_status`` (answered by ``_status``, never queued) reports a
    character call's own progress exactly as it already does for a Clay
    call -- both are the same ``_Job``/``_Op`` shapes, whichever lane ran
    them."""
    host = _bare_host_with_service()
    calls = agent_host._Calls()
    session = agent_clay.Session()
    started = threading.Event()
    release = threading.Event()

    def slow_call(svc, char_session, name, arguments):  # noqa: ARG001
        started.set()
        assert release.wait(WAIT), "release never came"
        return {"content": [], "isError": False}

    monkeypatch.setattr(agent_character, "HANDLERS", {"character_probe": slow_call})
    monkeypatch.setattr(agent_character, "call", slow_call)
    _shorten_service_call_timeout(monkeypatch, host, timeout=RUNNING_WAIT)

    outcome: dict[str, object] = {}

    def runner() -> None:
        outcome["result"] = host._call(session, calls, "character_probe", {})

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    try:
        assert started.wait(WAIT), "the character call never started"
        thread.join(timeout=WAIT)
        assert not thread.is_alive(), "the shortened wait never returned"

        status = host._status(calls, {"operation_id": "op-1"})
        assert status["isError"] is False
        assert status["structuredContent"]["state"] == agent_host.RUNNING
    finally:
        release.set()
        op = calls.get("op-1")
        deadline = time.monotonic() + WAIT
        while (
            op.status(host._job_lock) not in (agent_host.DONE, agent_host.RAISED)
            and time.monotonic() < deadline
        ):
            time.sleep(0.005)
        host._service.shutdown(wait=True)


def test_a_character_call_in_task_mode_completes_and_is_fetched_by_status(monkeypatch) -> None:
    """The ``call`` op's ``wait: false`` path (``_call_task``) and the
    ``status`` op (``_task_status``) both route a character tool through
    the service lane exactly as they already route Clay tools through the
    frame lane -- proven by a full task-mode round trip."""
    host = _bare_host_with_service()
    calls = agent_host._Calls()
    session = agent_clay.Session()

    monkeypatch.setattr(agent_character, "HANDLERS", {"character_export": None})
    monkeypatch.setattr(
        agent_character,
        "call",
        lambda svc, char_session, name, arguments: {  # noqa: ARG005
            "content": [{"type": "text", "text": "exported"}],
            "isError": False,
        },
    )

    try:
        header = host._call_task(session, calls, "character_export", {})
        assert header["status"] == "working"

        deadline = time.monotonic() + WAIT
        status_word = "working"
        body = b""
        while time.monotonic() < deadline:
            reply = host._task_status(calls, header["operation_id"])
            status_header, body = rpc.split_reply(reply)
            status_word = status_header["status"]
            if status_word != "working":
                break
            time.sleep(0.005)
    finally:
        host._service.shutdown(wait=True)

    assert status_word == "completed"
    result = json.loads(body.decode("utf-8"))
    assert result["isError"] is False
    assert result["content"][0]["text"] == "exported"


def test_a_timed_out_character_export_is_replayed_not_run_again(monkeypatch) -> None:
    """As ``test_a_retry_of_a_call_that_ran_but_never_answered_replays_instead_
    of_running_again`` above, but for a character tool on the service lane --
    proving the dedup/replay store (``_Calls``, ``_replay``) is genuinely
    shared between lanes, not a frame-lane-only guarantee the service lane
    quietly lacks."""
    host = _bare_host_with_service()
    calls = agent_host._Calls()
    session = agent_clay.Session()
    _shorten_service_call_timeout(monkeypatch, host, timeout=RUNNING_WAIT)

    call_count = {"n": 0}
    started = threading.Event()
    release = threading.Event()

    def fake_call(svc, char_session, name, arguments):  # noqa: ARG001
        call_count["n"] += 1
        started.set()
        assert release.wait(WAIT), "release never came"
        return {"content": [{"type": "text", "text": "exported"}], "isError": False}

    monkeypatch.setattr(agent_character, "HANDLERS", {"character_export": fake_call})
    monkeypatch.setattr(agent_character, "call", fake_call)

    try:
        outcome: dict[str, object] = {}

        def first_call() -> None:
            outcome["first"] = host._call(session, calls, "character_export", {})

        first_thread = threading.Thread(target=first_call, daemon=True)
        first_thread.start()
        assert started.wait(WAIT), "the job never started running"
        first_thread.join(timeout=WAIT)
        assert not first_thread.is_alive(), "the shortened wait never returned"
        first_text = outcome["first"]["content"][0]["text"]
        assert "already started" in first_text.lower()
        assert "op-1" in first_text

        release.set()
        op = calls.get("op-1")
        deadline = time.monotonic() + WAIT
        while (
            op.status(host._job_lock) not in (agent_host.DONE, agent_host.RAISED)
            and time.monotonic() < deadline
        ):
            time.sleep(0.005)

        second = host._call(session, calls, "character_export", {})
    finally:
        host._service.shutdown(wait=True)

    assert call_count["n"] == 1  # agent_character.call never ran a second time
    assert second["content"][0]["text"] == "exported"


def test_switching_off_fails_a_queued_service_lane_call(tmp_path, monkeypatch) -> None:
    """As ``test_stop_unblocks_a_pending_call_rather_than_waiting_out_call_
    timeout`` above, but for the service lane: a job still sitting behind
    ``SERVICE_WORKERS`` busy workers is never claimed by ``_execute`` at
    all, yet ``stop()`` still wakes its waiter with a ``DROPPED`` refusal
    rather than leaving it to time out against the full ``CALL_TIMEOUT``."""
    host = agent_host.AgentHost(_Ctx(), tmp_path)
    host.start()
    release = threading.Event()

    def occupy() -> None:
        host._run_on_service_job(lambda: release.wait(WAIT), timeout=agent_host.CALL_TIMEOUT)

    occupants = [
        threading.Thread(target=occupy, daemon=True) for _ in range(agent_host.SERVICE_WORKERS)
    ]
    for t in occupants:
        t.start()
    deadline = time.monotonic() + WAIT
    while len(host._service_jobs) < agent_host.SERVICE_WORKERS and time.monotonic() < deadline:
        time.sleep(0.005)
    assert len(host._service_jobs) == agent_host.SERVICE_WORKERS, "the pool never saturated"

    outcome: dict[str, object] = {}

    def waiter() -> None:
        outcome["result"] = host._run_on_service_job(lambda: None, timeout=agent_host.CALL_TIMEOUT)

    thread = threading.Thread(target=waiter, daemon=True)
    thread.start()
    deadline = time.monotonic() + WAIT
    while (
        len(host._service_jobs) < agent_host.SERVICE_WORKERS + 1
        and time.monotonic() < deadline
    ):
        time.sleep(0.005)
    assert len(host._service_jobs) == agent_host.SERVICE_WORKERS + 1, "the third job never queued"

    try:
        started = time.monotonic()
        host.stop()
        elapsed = time.monotonic() - started

        thread.join(timeout=WAIT)
        assert not thread.is_alive()
        assert elapsed < agent_host.CALL_TIMEOUT / 2
        _job, result, error, state = outcome["result"]
        assert state == agent_host.DROPPED
        assert result is not None and result["isError"] is True
    finally:
        release.set()
        for t in occupants:
            t.join(timeout=WAIT)


def test_a_call_submitted_while_stopping_is_answered_not_orphaned(tmp_path, monkeypatch) -> None:
    """Regression for the 2026-09-13 race: ``_submit_service`` used to read
    ``self._service``/``self._stopped`` and only *then* register the job in
    ``self._service_jobs``, as two separate, unguarded steps -- so a call
    that read the flag clear just before ``stop()`` set it could still
    register *after* ``_fail_pending``'s own snapshot of that registry had
    already run and missed it. ``stop()``'s very next step,
    ``TaskRunner.shutdown(wait=False)``, cancels any future the pool has not
    yet started (``cancel_futures=True``, see ``tasks.py``), so a job
    accepted into the pool in that gap never has its callable invoked at
    all: ``job.event`` is never set, the caller waits out the full
    ``CALL_TIMEOUT`` for an answer that never comes (here, nothing at all --
    not even the switched-off refusal, since the thing that would have
    stamped one onto ``job.result`` is exactly what never ran), and the
    registry entry survives into the next ``start()``.

    The fix makes the check and the registration one atomic step under
    ``_job_lock`` -- the same lock ``stop()`` now sets ``_stopped`` and swaps
    ``self._service`` to ``None`` under, before it ever calls
    ``_fail_pending`` -- so a job can no longer register after that snapshot
    has already run and missed it.

    Forced deterministically, since hoping two unsynchronised threads
    happen to interleave badly is not a regression test: the service lane's
    pool is saturated first (so a further job genuinely sits ``QUEUED``
    behind busy workers rather than starting at once -- the exact
    precondition ``cancel_futures=True`` needs), then two seams both the old
    and new code share -- ``TaskRunner.poll`` (called once between the
    check and the actual hand-off to the pool) and ``TaskRunner.submit``
    (the hand-off itself) -- are wrapped to pause the racing call right
    there, while ``AgentHost._fail_pending`` is wrapped to pause ``stop()``
    right after its own sweep and before ``TaskRunner.shutdown(wait=False)``
    runs. That lands the racing call's registration and hand-off to the
    pool inside the one gap the old code left open between
    ``_fail_pending``'s snapshot and the shutdown that follows it -- and
    against the fix, the same wrapping instead lands the registration
    *before* ``_fail_pending`` even runs (registration and the stopped
    check are one step now), so it is caught there instead, before the
    racing call ever reaches ``TaskRunner.submit`` at all."""
    host = agent_host.AgentHost(_Ctx(), tmp_path)
    host.start()
    release = threading.Event()

    def occupy() -> None:
        host._run_on_service_job(lambda: release.wait(WAIT), timeout=agent_host.CALL_TIMEOUT)

    occupants = [
        threading.Thread(target=occupy, daemon=True) for _ in range(agent_host.SERVICE_WORKERS)
    ]
    for t in occupants:
        t.start()
    deadline = time.monotonic() + WAIT
    while len(host._service_jobs) < agent_host.SERVICE_WORKERS and time.monotonic() < deadline:
        time.sleep(0.005)
    assert len(host._service_jobs) == agent_host.SERVICE_WORKERS, "the pool never saturated"

    at_poll = threading.Event()
    let_poll_return = threading.Event()
    real_poll = tasks_mod.TaskRunner.poll

    def paced_poll(self):
        at_poll.set()
        assert let_poll_return.wait(WAIT), "stop() never reached _fail_pending"
        return real_poll(self)

    submit_done = threading.Event()
    real_submit = tasks_mod.TaskRunner.submit

    def paced_submit(self, key, fn, *a, **kw):
        try:
            return real_submit(self, key, fn, *a, **kw)
        finally:
            submit_done.set()

    monkeypatch.setattr(tasks_mod.TaskRunner, "poll", paced_poll)
    monkeypatch.setattr(tasks_mod.TaskRunner, "submit", paced_submit)

    fail_pending_done = threading.Event()
    let_shutdown_proceed = threading.Event()
    real_fail_pending = agent_host.AgentHost._fail_pending

    def paced_fail_pending(self, *args, **kwargs):
        real_fail_pending(self, *args, **kwargs)
        fail_pending_done.set()
        assert let_shutdown_proceed.wait(WAIT), "the racing call never finished submitting"

    monkeypatch.setattr(agent_host.AgentHost, "_fail_pending", paced_fail_pending)

    outcome: dict[str, object] = {}

    def racing_call() -> None:
        outcome["result"] = host._run_on_service_job(lambda: None, timeout=RUNNING_WAIT)

    racer = threading.Thread(target=racing_call, daemon=True)
    racer.start()
    try:
        assert at_poll.wait(WAIT), "the racing call never reached the pool"

        stopper = threading.Thread(target=host.stop, daemon=True)
        stopper.start()
        try:
            assert fail_pending_done.wait(WAIT), "stop() never reached _fail_pending"

            let_poll_return.set()
            assert submit_done.wait(WAIT), "the racing call's hand-off to the pool never ran"

            let_shutdown_proceed.set()
        finally:
            stopper.join(timeout=WAIT)
        assert not stopper.is_alive(), "stop() never returned"
    finally:
        racer.join(timeout=WAIT)
        release.set()
        for t in occupants:
            t.join(timeout=WAIT)
    assert not racer.is_alive(), "the racing call never returned"

    result = outcome["result"]
    assert result is not None, "the racing call returned nothing to unpack"
    _job, call_result, error, state = result
    assert state == agent_host.DROPPED
    assert error is None
    assert call_result is not None, "the caller was never answered at all"
    assert call_result["isError"] is True
    assert host._service_jobs == {}, "a job survived stop() into the next start()"


def test_a_call_racing_stop_is_either_refused_or_caught_by_the_first_sweep(
    tmp_path, monkeypatch
) -> None:
    """Tightens the test above, which proves only the *outcome* of the
    2026-09-13 race (the caller is told "switched off" and the registry
    ends up empty), never the fix it names. Reverting the fix alone --
    :meth:`AgentHost._submit_service` back to checking
    ``self._stopped``/``self._service`` unguarded and only registering the
    job in ``self._service_jobs`` afterwards, once ``TaskRunner.poll`` and
    ``TaskRunner.submit`` have already returned -- still passes the test
    above, because :meth:`AgentHost.stop`'s *second*
    ``_drop_queued_service_jobs()`` sweep (run right after
    ``TaskRunner.shutdown(wait=False)``, kept as a second line of defence
    even with the fix in place) answers the caller anyway.

    This test names which sweep is allowed to catch the race. With the fix,
    the lane-open check and the registration happen as one step under
    ``_job_lock``, before :meth:`_submit_service` ever calls
    ``TaskRunner.poll`` -- so by the time this test's paced ``poll`` has the
    racing call paused (the same seam the test above uses), the job is
    already sitting in ``self._service_jobs``, and
    :meth:`AgentHost._fail_pending`'s own (first) sweep -- run while the
    racing call is still paused there -- is guaranteed to already see it.
    Revert the fix and registration cannot happen until *after* ``poll``
    returns, which this test's pause points force to be strictly after the
    first sweep has already run and returned (the racing call is only
    released from ``poll`` once ``_fail_pending`` is confirmed done) -- so
    the first sweep's snapshot can never contain it, whatever the second,
    post-shutdown sweep goes on to do with it later.

    Same deterministic interleaving as the test above (the pool saturated
    first, then ``TaskRunner.poll``, ``TaskRunner.submit`` and
    ``AgentHost._fail_pending`` each paced at the same three seams), plus one
    more seam: ``AgentHost._drop_queued_service_jobs`` is wrapped to record a
    snapshot of ``self._service_jobs`` on each of the two calls ``stop()``
    makes to it, so the assertion below can name the first one specifically
    rather than only the end state both tests already share."""
    host = agent_host.AgentHost(_Ctx(), tmp_path)
    host.start()
    release = threading.Event()

    def occupy() -> None:
        host._run_on_service_job(lambda: release.wait(WAIT), timeout=agent_host.CALL_TIMEOUT)

    occupants = [
        threading.Thread(target=occupy, daemon=True) for _ in range(agent_host.SERVICE_WORKERS)
    ]
    for t in occupants:
        t.start()
    deadline = time.monotonic() + WAIT
    while len(host._service_jobs) < agent_host.SERVICE_WORKERS and time.monotonic() < deadline:
        time.sleep(0.005)
    assert len(host._service_jobs) == agent_host.SERVICE_WORKERS, "the pool never saturated"

    at_poll = threading.Event()
    let_poll_return = threading.Event()
    real_poll = tasks_mod.TaskRunner.poll

    def paced_poll(self):
        at_poll.set()
        assert let_poll_return.wait(WAIT), "stop() never reached _fail_pending"
        return real_poll(self)

    submit_done = threading.Event()
    real_submit = tasks_mod.TaskRunner.submit

    def paced_submit(self, key, fn, *a, **kw):
        try:
            return real_submit(self, key, fn, *a, **kw)
        finally:
            submit_done.set()

    monkeypatch.setattr(tasks_mod.TaskRunner, "poll", paced_poll)
    monkeypatch.setattr(tasks_mod.TaskRunner, "submit", paced_submit)

    fail_pending_done = threading.Event()
    let_shutdown_proceed = threading.Event()
    real_fail_pending = agent_host.AgentHost._fail_pending

    def paced_fail_pending(self, *args, **kwargs):
        real_fail_pending(self, *args, **kwargs)
        fail_pending_done.set()
        assert let_shutdown_proceed.wait(WAIT), "the racing call never finished submitting"

    monkeypatch.setattr(agent_host.AgentHost, "_fail_pending", paced_fail_pending)

    # The one seam the test above does not need: a snapshot of
    # ``self._service_jobs`` taken on each of the two calls ``stop()`` makes
    # to ``_drop_queued_service_jobs`` -- so the assertion below can tell
    # "found by the first sweep" apart from "found by the second" (or never
    # found at all), rather than only checking the registry is empty by the
    # very end, which either sweep satisfies equally.
    drop_snapshots: list[set[int]] = []
    real_drop_queued_service_jobs = agent_host.AgentHost._drop_queued_service_jobs

    def recording_drop(self, *args, **kwargs) -> None:
        with self._job_lock:
            drop_snapshots.append(set(self._service_jobs))
        real_drop_queued_service_jobs(self, *args, **kwargs)

    monkeypatch.setattr(agent_host.AgentHost, "_drop_queued_service_jobs", recording_drop)

    outcome: dict[str, object] = {}

    def racing_call() -> None:
        outcome["result"] = host._run_on_service_job(lambda: None, timeout=RUNNING_WAIT)

    racer = threading.Thread(target=racing_call, daemon=True)
    racer.start()
    try:
        assert at_poll.wait(WAIT), "the racing call never reached the pool"

        stopper = threading.Thread(target=host.stop, daemon=True)
        stopper.start()
        try:
            assert fail_pending_done.wait(WAIT), "stop() never reached _fail_pending"

            let_poll_return.set()
            assert submit_done.wait(WAIT), "the racing call's hand-off to the pool never ran"

            let_shutdown_proceed.set()
        finally:
            stopper.join(timeout=WAIT)
        assert not stopper.is_alive(), "stop() never returned"
    finally:
        racer.join(timeout=WAIT)
        release.set()
        for t in occupants:
            t.join(timeout=WAIT)
    assert not racer.is_alive(), "the racing call never returned"

    result = outcome["result"]
    assert result is not None, "the racing call returned nothing to unpack"
    job, call_result, error, state = result
    assert job is not None, "the racing call was refused before it ever minted a job"
    assert state == agent_host.DROPPED
    assert error is None
    assert call_result is not None, "the caller was never answered at all"
    assert call_result["isError"] is True
    assert host._service_jobs == {}, "a job survived stop() into the next start()"

    assert len(drop_snapshots) == 2, (
        "stop() must sweep _drop_queued_service_jobs exactly twice -- once "
        "inside _fail_pending, once more after TaskRunner.shutdown(wait=False)"
    )
    assert id(job) in drop_snapshots[0], (
        "the racing job was absent from _fail_pending's own (first) sweep -- "
        "it was only caught later, by the post-shutdown safeguard, or not at "
        "all. The fix requires the lane-open check and the registration into "
        "_service_jobs to be one atomic step under _job_lock, completed "
        "before _submit_service ever calls TaskRunner.poll, so the very "
        "first sweep -- run while this test's paced poll still has the "
        "racing call paused there -- is guaranteed to already see it."
    )


def test_stop_never_terminates_tracked_child_processes(tmp_path, monkeypatch) -> None:
    """The critical constraint this tranche rests on: ``stop()`` must call
    ``TaskRunner.shutdown(wait=False)`` and never pass ``timeout`` --
    ``tasks.py``'s own ``shutdown`` docstring names the ``timeout`` branch as
    the one that calls ``winjob.terminate_tracked()`` on whatever is still
    running, which kills every tracked child process in the whole app (a
    Blender bake, a fetch download, a matting worker), not just this host's
    own service-lane workers. Proven by monkeypatching
    ``winjob.terminate_tracked`` itself and checking it is never reached
    while a character call is still stuck mid-run when ``stop()`` runs."""
    from warlock import winjob

    terminate_calls: list[str] = []
    monkeypatch.setattr(
        winjob, "terminate_tracked", lambda *a, **kw: terminate_calls.append("called") or []
    )

    host = agent_host.AgentHost(_Ctx(), tmp_path)
    host.start()
    release = threading.Event()

    def stuck(svc, char_session, name, arguments):  # noqa: ARG001
        release.wait(WAIT)
        return {"content": [], "isError": False}

    monkeypatch.setattr(agent_character, "HANDLERS", {"character_probe": stuck})
    monkeypatch.setattr(agent_character, "call", stuck)

    session = agent_clay.Session()
    calls = agent_host._Calls()
    thread = threading.Thread(
        target=lambda: host._call(session, calls, "character_probe", {}), daemon=True
    )
    thread.start()
    try:
        deadline = time.monotonic() + WAIT
        while not host._service_jobs and time.monotonic() < deadline:
            time.sleep(0.005)
        assert host._service_jobs, "the service job never registered"

        host.stop()
    finally:
        release.set()
        thread.join(timeout=WAIT)

    assert terminate_calls == [], "stop() must never reach winjob.terminate_tracked"


def test_pump_and_the_service_lane_skip_the_same_tombstone() -> None:
    """``_execute`` is the one function both lanes claim a job through --
    proven by dropping a job by hand for each lane and driving it through
    each lane's own *real* entry point: ``pump()`` itself for the frame
    queue, and ``_run_service_job`` (not ``_execute`` directly) for the
    service lane, the way :meth:`AgentHost._submit_service` actually hands
    work to the pool. Both report a tombstone the same way (``event`` set,
    nothing run) rather than two copies of the claim/run/finish sequence
    quietly drifting apart."""
    host = _bare_host_with_service()
    try:
        frame_ran: list[str] = []
        frame_job = agent_host._Job(lambda: frame_ran.append("frame"))
        frame_job.state = agent_host.DROPPED
        host._queue.put(frame_job)

        service_ran: list[str] = []
        service_job = agent_host._Job(lambda: service_ran.append("service"))
        service_job.state = agent_host.DROPPED
        host._service_jobs[id(service_job)] = service_job

        host.pump(budget=1.0)
        host._run_service_job(service_job)

        assert frame_ran == []
        assert service_ran == []
        assert frame_job.event.is_set()
        assert service_job.event.is_set()
        # ``_run_service_job``'s own ``finally`` pops the registry entry no
        # matter what ``_execute`` found -- a tombstone must not linger.
        assert id(service_job) not in host._service_jobs
    finally:
        host._service.shutdown(wait=True)


def test_a_character_call_toasts_the_human_on_the_frame_thread(tmp_path, monkeypatch) -> None:
    """A character call runs on the service lane and must never touch
    ``ctx.toast`` directly from there -- proven through the real ``_serve``
    wiring (``calls.character.toast = self._toast``, set on connect) rather
    than by wiring ``calls.character.toast`` up by hand in the test: a real
    pipe connection is made, a real ``call`` RPC op is answered by the
    service lane while ``pump()`` is deliberately not running, and only
    once ``pump()`` is driven from this thread does the toast land."""
    host = agent_host.AgentHost(_Ctx(), tmp_path)
    host.start()
    stop_pumping = threading.Event()

    def pump_loop() -> None:
        while not stop_pumping.is_set():
            host.pump(budget=0.01)
            time.sleep(0.005)

    pumper = threading.Thread(target=pump_loop, daemon=True)
    pumper.start()
    conn = pipe.connect(tmp_path)
    try:
        # Settle the connection (the tab-open job needs pump() running)
        # before turning the pump loop off -- a ``hello`` round trip only
        # replies once ``_serve`` has moved past that blocking open and is
        # reading requests.
        conn.send_bytes(rpc.encode_request("hello", versions=[1], bridge_version="test"))
        _recv(conn)
        # Pump is now off for good: the toast this test cares about must
        # not land until this test drives pump() itself, below.
        stop_pumping.set()
        pumper.join(timeout=WAIT)

        def minting_call(svc, char_session, name, arguments):  # noqa: ARG001
            char_session.toast("a character was minted")
            return {"content": [], "isError": False}

        monkeypatch.setattr(agent_character, "HANDLERS", {"character_create": minting_call})
        monkeypatch.setattr(agent_character, "call", minting_call)

        conn.send_bytes(rpc.encode_request("call", tool="character_create", args={}))
        _header, body = rpc.split_reply(_recv(conn))
        result = json.loads(body.decode("utf-8"))
        assert result["isError"] is False

        # The service lane answered the call already (no pump needed for
        # that), but the toast it raised only reaches ``ctx`` through the
        # frame queue -- and pump() has not run since before the call.
        assert host.ctx.toasts == [("An agent connected.", "info")]
        host.pump(budget=1.0)
        assert host.ctx.toasts == [
            ("An agent connected.", "info"),
            ("a character was minted", "info"),
        ]
    finally:
        stop_pumping.set()
        pumper.join(timeout=WAIT)
        conn.close()
        host.stop()


def test_every_name_the_host_publishes_is_unique() -> None:
    """``_rpc_tools`` concatenates Clay's tools, the character surface's
    (the real ``agent_character.tools()``, not a stand-in -- a collision or
    a vanished character surface must be able to fail this), and this
    module's own transport tools -- no two entries may share a name, or a
    client's tool list would silently shadow one of them."""
    host = _bare_host()
    names = [tool.name for tool in host._rpc_tools()]
    assert len(names) == len(set(names)), names
    # A guard on the guard: if ``agent_character.tools()`` ever returned
    # nothing (the surface removed, or every tool renamed out from under
    # this scan), the uniqueness check above would still pass vacuously.
    assert any(name.startswith("character_") for name in names), names


# --- Familiar: an in-app session, independent of the pipe (T1) --------------


def test_the_familiar_session_works_while_the_agent_server_is_off(tmp_path) -> None:
    """The whole point of :meth:`AgentHost.open_session`: an in-app caller
    gets a working Clay tool surface even though ``start()`` (the pipe) is
    never called at all -- Familiar has no pipe and no bridge, and must not
    need either."""
    host = agent_host.AgentHost(_Ctx(), tmp_path)
    assert not host.running

    session = host.open_session()
    stop_pumping = threading.Event()
    pumper = threading.Thread(target=_pump_loop, args=(host, stop_pumping), daemon=True)
    pumper.start()
    try:
        outcome: dict[str, object] = {}

        def call() -> None:
            outcome["result"] = session.call("clay_add_primitive", {"generator": "box"})

        thread = threading.Thread(target=call, daemon=True)
        thread.start()
        thread.join(timeout=WAIT)
        assert not thread.is_alive()
        result = outcome["result"]
        assert result is not None and result.get("isError") is False, result
    finally:
        stop_pumping.set()
        pumper.join(timeout=WAIT)
        session.close()


def test_switching_the_agent_server_off_never_drops_familiar_jobs(tmp_path) -> None:
    """A job an in-app session queued must survive ``stop()`` -- turning the
    pipe server off is only ever supposed to fail *its own* jobs
    (:meth:`AgentHost._fail_pending`'s owner scoping), never a Familiar
    session's, and the frame lane itself must stay open for it because the
    session is still holding its own lane-ownership token."""
    host = agent_host.AgentHost(_Ctx(), tmp_path)
    host.start()
    session = host.open_session()
    stop_pumping = threading.Event()
    pumper = threading.Thread(target=_pump_loop, args=(host, stop_pumping), daemon=True)
    pumper.start()
    try:
        release = threading.Event()
        started = threading.Event()

        def slow(ctx, sess, name, arguments):  # noqa: ARG001
            started.set()
            assert release.wait(WAIT), "release never came"
            return {"content": [], "isError": False}

        import warlock.studio.agent_clay as agent_clay_mod

        original_call = agent_clay_mod.call
        agent_clay_mod.call = slow
        try:
            outcome: dict[str, object] = {}

            def call() -> None:
                outcome["result"] = session.call("clay_scene", {})

            thread = threading.Thread(target=call, daemon=True)
            thread.start()
            assert started.wait(WAIT), "the familiar job never started running"

            # Switching the pipe server off must not touch this job.
            host.stop()
            assert not host.running

            release.set()
            thread.join(timeout=WAIT)
            assert not thread.is_alive()
            result = outcome["result"]
            assert result is not None and result.get("isError") is False, result
        finally:
            agent_clay_mod.call = original_call
    finally:
        stop_pumping.set()
        pumper.join(timeout=WAIT)
        session.close()


def test_stopping_the_last_lane_owner_never_terminates_tracked_children(
    tmp_path, monkeypatch
) -> None:
    """As ``test_stop_never_terminates_tracked_child_processes``, but for the
    lane-ownership rewrite: closing an :class:`InAppSession` that turns out
    to be the last owner of the service lane must still shut the runner
    down with ``wait=False`` and never a ``timeout`` -- never reaching
    ``winjob.terminate_tracked()`` -- exactly the constraint ``AgentHost.
    stop()`` already keeps."""
    from warlock import winjob

    terminate_calls: list[str] = []
    monkeypatch.setattr(
        winjob, "terminate_tracked", lambda *a, **kw: terminate_calls.append("called") or []
    )

    host = agent_host.AgentHost(_Ctx(), tmp_path)
    session = host.open_session()
    release = threading.Event()

    def stuck(svc, char_session, name, arguments):  # noqa: ARG001
        release.wait(WAIT)
        return {"content": [], "isError": False}

    monkeypatch.setattr(agent_character, "HANDLERS", {"character_probe": stuck})
    monkeypatch.setattr(agent_character, "call", stuck)

    thread = threading.Thread(
        target=lambda: session.call("character_probe", {}), daemon=True
    )
    thread.start()
    try:
        deadline = time.monotonic() + WAIT
        while not host._service_jobs and time.monotonic() < deadline:
            time.sleep(0.005)
        assert host._service_jobs, "the service job never registered"

        session.close()
    finally:
        release.set()
        thread.join(timeout=WAIT)

    assert terminate_calls == [], "close() must never reach winjob.terminate_tracked"


def test_an_in_app_call_on_the_frame_thread_raises_instead_of_deadlocking(tmp_path) -> None:
    """``InAppSession.call`` blocks on an ``Event`` only ``AgentHost.pump``
    ever sets, and nothing calls ``pump`` concurrently with a synchronous,
    same-thread call -- so a caller already on the frame thread (the main
    thread, per this module's own convention) must be refused outright
    rather than left to hang for the full ``CALL_TIMEOUT``."""
    host = agent_host.AgentHost(_Ctx(), tmp_path)
    session = host.open_session()
    try:
        assert threading.current_thread() is threading.main_thread()
        with pytest.raises(RuntimeError):
            session.call("clay_scene", {})
    finally:
        session.close()


def test_opening_an_in_app_session_mints_no_tab(tmp_path) -> None:
    """Unlike a pipe connection (:meth:`AgentHost._serve`, which mints a tab
    before a bridge can ask for one), :meth:`AgentHost.open_session` must
    not queue any frame-thread work at all -- a fresh session's own
    ``agent_clay.Session`` starts with no tab pinned, and nothing about
    opening it should touch ``ClayState``."""
    host = agent_host.AgentHost(_Ctx(), tmp_path)
    session = host.open_session()
    try:
        assert not session._session.tab_uid
    finally:
        session.close()


def test_a_pipe_call_after_stop_is_refused_while_familiar_holds_the_lanes(tmp_path) -> None:
    """The hole a plain ``self._queue is None`` check reopened: once a
    Familiar session has its own hold on the lanes, ``stop()`` releasing
    only the pipe's own ownership leaves ``self._queue``/``self._service``
    non-``None``. A pipe-owned call arriving after ``stop()`` must still be
    refused outright -- accepted onto lanes that are still open with nothing
    left to ever fail it (the pipe is gone) would otherwise hang the caller
    out to the full ``CALL_TIMEOUT`` for an answer that never comes."""
    host = agent_host.AgentHost(_Ctx(), tmp_path)
    session = host.open_session()
    try:
        host.start()
        host.stop()
        assert not host.running
        # The lanes themselves are still open: Familiar's own session still
        # owns them, so neither is torn down by a pipe-only stop().
        assert host._queue is not None
        assert host._service is not None

        _job, frame_result, frame_error, frame_state = host._run_on_frame_job(
            lambda: "ran", timeout=0.2, owner=agent_host.PIPE_OWNER
        )
        assert frame_state == agent_host.DROPPED, "a pipe job was accepted onto live lanes"
        assert frame_error is None
        assert frame_result is not None and frame_result["isError"] is True

        _job2, service_result, service_error, service_state = host._run_on_service_job(
            lambda: "ran", timeout=0.2, owner=agent_host.PIPE_OWNER
        )
        assert service_state == agent_host.DROPPED, "a pipe job was accepted onto live lanes"
        assert service_error is None
        assert service_result is not None and service_result["isError"] is True
    finally:
        session.close()


def test_a_closed_in_app_session_refuses_calls(tmp_path) -> None:
    """A closed :class:`InAppSession` must refuse rather than queue -- it
    has released its own lane ownership, and calling into a torn-down (or
    someone-else's still-open) lane after that would be answering for a
    session that no longer exists. Run off the main thread so the
    frame-thread guard cannot be what raises here -- this test is about the
    closed check specifically."""
    host = agent_host.AgentHost(_Ctx(), tmp_path)
    session = host.open_session()
    session.close()

    outcome: dict[str, object] = {}

    def call() -> None:
        try:
            session.call("clay_scene", {})
        except RuntimeError as exc:
            outcome["error"] = exc

    thread = threading.Thread(target=call, daemon=True)
    thread.start()
    thread.join(timeout=WAIT)
    assert not thread.is_alive()
    assert isinstance(outcome.get("error"), RuntimeError)
