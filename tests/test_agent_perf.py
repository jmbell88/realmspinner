"""The agent bridge's own frame-thread and per-call budget. ``uv run pytest
-m perf -n 0``.

Five commits (``git log --oneline 9fd36833..HEAD``) landed on ``AgentHost``
and ``agent_clay`` and each added per-call work that can run **on the frame
thread**, where the drain budget (``AgentHost.pump``'s default) is 8 ms: a
``blake2b`` fingerprint over every call's canonicalised arguments
(``agent_host._fingerprint``), a linear scan of the per-connection dedup
store (``_Calls.pending``), a ``json.loads`` of every reply payload that was
just ``json.dumps``-ed (``agent_clay._json``), and an unknown-argument check
(a set difference on every call, plus a ``difflib.get_close_matches`` pass
only when a name is actually unknown -- not exercised here, since every call
below is well-formed). This measures the **whole round trip an agent
actually experiences**: a real ``pipe.connect`` to a started ``AgentHost``, a
real ``tools/call`` frame in, a real reply frame out, with a background
thread driving ``pump()`` the way ``main.py:App.frame`` does -- the same
harness shape as ``tests/test_agent_host.py``'s
``test_a_real_round_trip_answers_requests_and_gives_a_notification_zero_bytes``,
reused rather than reinvented.

Measured on the development machine (Windows 11, Python 3.13.13, ``-n 0``,
nothing else running), median of 101 round trips for the small
``clay_scene`` and 51 for ``tools/list`` and the 50-object ``clay_scene``
(those two cost more to set up and a wall-clock claim about them does not
need as many samples to stop moving):

* ``clay_scene`` on a 1-object document: ~1.7 ms.
* ``clay_scene`` on a 50-object document: ~5.6 ms.
* ``tools/list`` (25 Clay tools plus ``warlock_status``, all rebuilt from the
  live registries every time): ~0.5 ms.

See ``docs/measurements/2026-09-10-agent-bridge-round-trip.md`` for the full
before/after against ``9fd36833`` (the commit just before this tranche) and
the conclusion: the round trip is dominated by pipe I/O and JSON framing,
not by the five commits' additions -- the ``clay_scene`` numbers above are
within measurement noise of ``9fd36833``'s, and only ``tools/list`` shows a
real (if tiny, ~0.1 ms) difference, from one extra tool in the catalogue.
The budgets below carry roughly 5-8x headroom on the two ``clay_scene``
shapes and ~20x on ``tools/list`` (its absolute cost is small enough that a
tighter multiple would just be measuring pipe scheduling jitter) -- generous
enough that a modest CI box (this suite already runs under 7-8 concurrent
xdist workers on other lanes, and this one is serial only because a
wall-clock *budget* is meaningless under contention, not because the call
itself is slow) passes comfortably, tight enough that a regression that
makes one of these additions scale badly -- an unbounded dedup store, an
accidentally-quadratic fingerprint -- fails here rather than first being
noticed as a sluggish agent session.

Excluded from the parallel run for the reason every ``perf`` case is (see
``pyproject.toml``'s own comment on the marker).
"""

from __future__ import annotations

import statistics
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from warlock.mcp import pipe, protocol
from warlock.studio import agent_host

#: A generous but bounded ceiling for anything that talks over the real pipe
#: in this file -- comfortably under pytest's 120 s default and comfortably
#: over what a healthy host ever takes. Matches ``tests/test_agent_host.py``'s
#: own ``WAIT``.
WAIT = 5.0

#: Run counts. Smaller for the two shapes that cost more to set up (building
#: a 50-object document, or listing 25 tools is comparatively self-contained
#: but timed alongside the others for consistency) -- both are still well
#: above the "at least a couple dozen" floor a median needs to stop being a
#: coin flip between two samples.
SMALL_SCENE_RUNS = 101
LARGE_SCENE_RUNS = 51
TOOLS_LIST_RUNS = 51

#: Budgets, generously above the measured medians in the module docstring --
#: see there for the reasoning (5-8x on the two ``clay_scene`` shapes, ~20x
#: on ``tools/list``). Milliseconds.
MAX_MEDIAN_MS_SMALL_SCENE = 15.0
MAX_MEDIAN_MS_LARGE_SCENE = 30.0
MAX_MEDIAN_MS_TOOLS_LIST = 10.0


class _Ctx:
    """As ``tests/test_agent_host.py``'s own ``_Ctx`` -- just enough of the
    app's ``Ctx`` for ``agent_clay._tab`` to mint a tab."""

    def __init__(self) -> None:
        self.state = SimpleNamespace(clay=None)
        self.settings = SimpleNamespace()
        self.toasts: list[tuple[str, str]] = []

    def toast(self, message: str, level: str = "info") -> None:
        self.toasts.append((message, level))


def _recv(conn, timeout: float = WAIT) -> bytes:
    """One frame, or a clear assertion failure rather than an unbounded block."""
    assert conn.poll(timeout), f"no reply within {timeout}s"
    return conn.recv_bytes()


def _call(conn, msg_id: int, method: str, params: dict | None = None) -> dict:
    """One JSON-RPC request, sent and answered, over the real pipe."""
    message: dict = {"jsonrpc": "2.0", "id": msg_id, "method": method}
    if params is not None:
        message["params"] = params
    conn.send_bytes(protocol.encode(message))
    reply = protocol.decode(_recv(conn))
    assert reply["id"] == msg_id, reply
    return reply


class _Bridge:
    """A started ``AgentHost``, a background thread draining ``pump()`` the
    way ``App.frame`` would, and a connected, initialized pipe -- the setup
    every measurement below needs, built once so the timed loops measure
    only the call itself.
    """

    def __init__(self, tmp_path: Path) -> None:
        self.host = agent_host.AgentHost(_Ctx(), tmp_path)
        self.host.start()
        self._stop = threading.Event()
        self._pumper = threading.Thread(target=self._pump_loop, daemon=True)
        self._pumper.start()
        self.conn = pipe.connect(tmp_path)
        self._next_id = 1
        reply = self.call("initialize")
        assert reply["result"]["protocolVersion"] == protocol.PROTOCOL_VERSION

    def _pump_loop(self) -> None:
        while not self._stop.is_set():
            self.host.pump(budget=0.01)
            time.sleep(0.001)

    def call(self, method: str, params: dict | None = None) -> dict:
        self._next_id += 1
        return _call(self.conn, self._next_id, method, params)

    def add_box(self) -> dict:
        reply = self.call(
            "tools/call", {"name": "clay_add_primitive", "arguments": {"generator": "box"}}
        )
        assert reply["result"]["isError"] is False, reply
        return reply

    def close(self) -> None:
        self.conn.close()
        self._stop.set()
        self._pumper.join(timeout=WAIT)
        self.host.stop()


def _median_round_trip_ms(fn, runs: int, label: str, warmup: int = 3) -> float:
    """The median wall-clock cost of *runs* calls to *fn* (each a full
    request/reply round trip), in milliseconds -- a median, not a minimum:
    see ``docs/measurements/2026-09-07-library-frame-times.md`` for why a
    single best run says nothing about what an agent actually experiences
    call to call. *warmup* calls run first and are discarded, so the first
    real import/allocation inside the call path is not what gets measured.

    The figure is printed as well as returned, so ``-m perf -n 0 -s`` re-runs
    the measurement in ``docs/measurements/2026-09-10-agent-bridge-round-trip.md``
    rather than only checking it still fits the budget. That document's own
    method note keeps its raw numbers in a probe's ``-s`` output for exactly
    this reason: a budget that passes tells a future reader nothing about
    where the real cost has drifted to, and a committed benchmark is the
    natural place to keep the probe rather than rebuilding one.
    """
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    median = statistics.median(samples)
    print(
        f"\n{label}: median {median:.3f} ms "
        f"(min {min(samples):.3f}, max {max(samples):.3f}, n={runs})"
    )
    return median


@pytest.fixture
def bridge(tmp_path):
    b = _Bridge(tmp_path)
    try:
        yield b
    finally:
        b.close()


@pytest.mark.perf
def test_clay_scene_round_trip_on_a_one_object_document(bridge: _Bridge) -> None:
    bridge.add_box()

    def one_call() -> None:
        reply = bridge.call("tools/call", {"name": "clay_scene", "arguments": {}})
        assert reply["result"]["isError"] is False

    median_ms = _median_round_trip_ms(one_call, SMALL_SCENE_RUNS, "clay_scene, 1 object")
    assert median_ms < MAX_MEDIAN_MS_SMALL_SCENE, f"{median_ms:.3f} ms median"


@pytest.mark.perf
def test_clay_scene_round_trip_on_a_fifty_object_document(bridge: _Bridge) -> None:
    for _ in range(50):
        bridge.add_box()

    def one_call() -> None:
        reply = bridge.call("tools/call", {"name": "clay_scene", "arguments": {}})
        result = reply["result"]
        assert result["isError"] is False
        assert len(result["structuredContent"]["objects"]) == 50

    median_ms = _median_round_trip_ms(one_call, LARGE_SCENE_RUNS, "clay_scene, 50 objects")
    assert median_ms < MAX_MEDIAN_MS_LARGE_SCENE, f"{median_ms:.3f} ms median"


@pytest.mark.perf
def test_tools_list_round_trip(bridge: _Bridge) -> None:
    """``tools/list`` rebuilds all 25 ``Tool`` schemas from the live
    registries on every call (``agent_clay.tools``'s own docstring) -- the
    one call in this file whose cost is pure catalogue construction, nothing
    to do with a document at all.
    """

    def one_call() -> None:
        reply = bridge.call("tools/list")
        assert len(reply["result"]["tools"]) >= 25

    median_ms = _median_round_trip_ms(one_call, TOOLS_LIST_RUNS, "tools/list")
    assert median_ms < MAX_MEDIAN_MS_TOOLS_LIST, f"{median_ms:.3f} ms median"
