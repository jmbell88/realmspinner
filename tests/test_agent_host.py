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

from warlock.mcp import pipe, rpc
from warlock.studio import agent_clay, agent_host, agent_transcript

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


# --- catalogue carries agent_clay's instructions ------------------------------


def test_the_catalogue_reply_carries_agent_clay_supplied_instructions(
    tmp_path, monkeypatch
) -> None:
    """``_catalogue_payload`` passes ``agent_clay.instructions()`` through --
    proven over a real pipe, the way the round trip above proves everything
    else ``_serve`` promises, rather than by reading the source."""
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
            conn.send_bytes(rpc.encode_request("catalogue"))
            header, body = rpc.split_reply(_recv(conn))
            assert body == b""
            assert header["instructions"] == "known string"
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
    # ``_call`` now calls the four-tuple ``_run_on_frame_job`` (it needs the
    # ``_Job`` itself to track a still-live operation for the dedup store),
    # so that is what gets monkeypatched -- ``_run_on_frame`` is now a thin
    # wrapper over it and no longer the seam ``_call`` reads through.
    monkeypatch.setattr(
        host, "_run_on_frame_job", lambda run, timeout=None: (None, None, None, agent_host.DROPPED)
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
        host, "_run_on_frame_job", lambda run, timeout=None: (None, None, None, agent_host.DROPPED)
    )
    dropped_text = host._call(agent_clay.Session(), agent_host._Calls(), "clay_scene", {})[
        "content"
    ][0]["text"]

    monkeypatch.setattr(
        host, "_run_on_frame_job", lambda run, timeout=None: (None, None, None, agent_host.RUNNING)
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

    def _short(self, run, timeout_arg=agent_host.CALL_TIMEOUT):  # noqa: ARG001
        return real(self, run, timeout=timeout)

    monkeypatch.setattr(agent_host.AgentHost, "_run_on_frame_job", _short)


def _pump_loop(host: agent_host.AgentHost, stop: threading.Event) -> None:
    """Drains *host*'s queue on a background thread the way ``App.frame``
    would, until *stop* is set. The same small pattern the round-trip tests
    above already use, factored out because every test below needs it."""
    while not stop.is_set():
        host.pump(budget=0.01)
        time.sleep(0.005)


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
        host, "_run_on_frame_job", lambda run, timeout=None: (None, None, None, agent_host.DROPPED)
    )
    dropped_text = host._call(agent_clay.Session(), agent_host._Calls(), "clay_scene", {})[
        "content"
    ][0]["text"]
    assert "op-1" in dropped_text

    monkeypatch.setattr(
        host, "_run_on_frame_job", lambda run, timeout=None: (None, None, None, agent_host.RUNNING)
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
        host, "_run_on_frame_job", lambda run, timeout=None: (None, None, None, agent_host.DROPPED)
    )
    dropped = host._call(agent_clay.Session(), agent_host._Calls(), "clay_scene", {})
    assert (dropped.get("structuredContent") or {}).get("recovery") == "retry"

    monkeypatch.setattr(
        host, "_run_on_frame_job", lambda run, timeout=None: (None, None, None, agent_host.RUNNING)
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
        host, "_run_on_frame_job", lambda run, timeout=None: (None, result, None, agent_host.DONE)
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
        host, "_run_on_frame_job", lambda run, timeout=None: (None, result, None, agent_host.DONE)
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
        host, "_run_on_frame_job", lambda run, timeout=None: (None, result, None, agent_host.DONE)
    )

    returned = host._call(agent_clay.Session(), agent_host._Calls(), "clay_scene", {})

    # The call's own answer, unaffected by the transcript write failing --
    # an agent session must not die because a diagnostic path was unwritable.
    assert returned is result


