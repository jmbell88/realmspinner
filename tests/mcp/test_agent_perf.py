"""The agent bridge's own frame-thread and per-call budget. ``uv run pytest
-m perf -n 0``.

Realmspinner's pipe answers RPC v1 exclusively now (no bare-MCP path -- see
``dev/INVARIANTS.md``'s agent paragraph), so every round trip measured
here goes through both layers a real agent session actually pays for:
``_RpcBridge`` speaks RPC v1 (``hello``/``catalogue``/``call``) to a real,
started ``AgentHost`` over a real pipe, exactly the way ``realmspinner mcp``
(``bridge.py``) does, and then runs each MCP request through
``protocol.bridge_dispatch`` in this process -- the same two steps
``bridge.py`` itself performs, without a child process or its own pipe I/O
in the way of the measurement. A background thread drives ``AgentHost.
pump()`` the way ``main.py:App.frame`` would, the same harness shape as
``tests/studio/test_agent_host.py``'s own real-pipe round trip.

Per-call work that can run **on the frame thread**, where the drain budget
(``AgentHost.pump``'s default) is 8 ms, includes: a ``blake2b`` fingerprint
over every call's canonicalised arguments (``agent_host._fingerprint``), a
linear scan of the per-connection dedup store (``_Calls.pending``), a
``json.loads`` of every reply payload that was just ``json.dumps``-ed
(``agent_clay._json``), and an unknown-argument check (a set difference on
every call, plus a ``difflib.get_close_matches`` pass only when a name is
actually unknown -- not exercised here, since every call below is
well-formed).

Measured on the development machine (Windows 11, Python 3.13.13, ``-n 0``,
nothing else running), median of 101 round trips for the small
``clay_scene`` and 51 for the 50-object one (which costs more to set up and
a wall-clock claim about it does not need as many samples to stop moving):

* ``clay_scene`` on a 1-object document: ~1.7 ms.
* ``clay_scene`` on a 50-object document: ~5.6 ms.

See ``dev/measurements/2026-09-10-agent-bridge-round-trip.md`` for the
fuller history of these numbers. The budgets below (the measured medians
above, plus 0.5 ms of headroom for the RPC v1 + ``bridge_dispatch`` layers
on top of raw pipe I/O and JSON framing) are generous enough that a modest
CI box (this suite already runs under 7-8 concurrent xdist workers on other
lanes, and this one is serial only because a wall-clock *budget* is
meaningless under contention, not because the call itself is slow) passes
comfortably, tight enough that a regression that makes one of these
additions scale badly -- an unbounded dedup store, an accidentally-
quadratic fingerprint or catalogue refetch -- fails here rather than first
being noticed as a sluggish agent session.

Excluded from the parallel run for the reason every ``perf`` case is (see
``pyproject.toml``'s own comment on the marker).
"""

from __future__ import annotations

import json
import statistics
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from realmspinner.mcp import pipe, protocol, rpc
from realmspinner.studio import agent_host

#: A generous but bounded ceiling for anything that talks over the real pipe
#: in this file -- comfortably under pytest's 120 s default and comfortably
#: over what a healthy host ever takes. Matches ``tests/studio/test_agent_host.py``'s
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
ANALYZE_KITBASH_RUNS = 51

#: Budgets, generously above the measured medians in the module docstring --
#: see there for the reasoning (5-8x on the two ``clay_scene`` shapes, ~20x
#: on ``tools/list``). Milliseconds.
MAX_MEDIAN_MS_SMALL_SCENE = 15.0
MAX_MEDIAN_MS_LARGE_SCENE = 30.0
MAX_MEDIAN_MS_TOOLS_LIST = 10.0
#: ``clay.analyze``'s own module docstring states the target this pins:
#: "about 100 ms for a typical kitbash." Measured on the development machine
#: (Windows 11, Python 3.13.13, ``-n 0``), median of 51 round trips over a
#: six-object kitbash (three boxes, three cylinders, two pairs close enough
#: to intersect and trigger an overlap boolean): ~9.8 ms. This budget is
#: about 5x that, the same headroom ``MAX_MEDIAN_MS_LARGE_SCENE`` gives its
#: own measurement.
MAX_MEDIAN_MS_ANALYZE_KITBASH = 50.0


class _Ctx:
    """As ``tests/studio/test_agent_host.py``'s own ``_Ctx`` -- just enough of the
    app's ``Ctx`` for ``agent_clay._tab`` to mint a tab."""

    def __init__(self) -> None:
        self.state = SimpleNamespace(clay=None)
        self.settings = SimpleNamespace()
        self.toasts: list[tuple[str, str]] = []

    def toast(self, message: str, level: str = "info") -> None:
        self.toasts.append((message, level))

    def submit(self, key: str, fn, *args, tag=None, **kwargs) -> bool:
        # A stand-in for app_ctx.Ctx.submit/TaskRunner, the 2026-09-18 audit
        # (agents-06): AgentHost.start() now calls ctx.submit to write its
        # catalogue snapshot off whichever thread calls start(), matching
        # tests/studio/test_agent_host.py's own _Ctx.
        threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=True).start()
        return True


def _median_round_trip_ms(fn, runs: int, label: str, warmup: int = 3) -> float:
    """The median wall-clock cost of *runs* calls to *fn* (each a full
    request/reply round trip), in milliseconds -- a median, not a minimum:
    see ``dev/measurements/2026-09-07-library-frame-times.md`` for why a
    single best run says nothing about what an agent actually experiences
    call to call. *warmup* calls run first and are discarded, so the first
    real import/allocation inside the call path is not what gets measured.

    The figure is printed as well as returned, so ``-m perf -n 0 -s`` re-runs
    the measurement in ``dev/measurements/2026-09-10-agent-bridge-round-trip.md``
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


MAX_MEDIAN_MS_BRIDGE_SMALL_SCENE = MAX_MEDIAN_MS_SMALL_SCENE + 0.5
MAX_MEDIAN_MS_BRIDGE_LARGE_SCENE = MAX_MEDIAN_MS_LARGE_SCENE + 0.5


class _RpcBridge:
    """A started ``AgentHost``, a background thread draining ``pump()`` the
    way ``App.frame`` would, and a real RPC v1 connection to it -- plus one
    in-process ``protocol.bridge_dispatch`` per call, the same two steps
    ``bridge.py`` itself performs, without a child process or its own pipe
    I/O in the way of the measurement."""

    def __init__(self, tmp_path: Path) -> None:
        self.host = agent_host.AgentHost(_Ctx(), tmp_path)
        self.host.start()
        self._stop = threading.Event()
        self._pumper = threading.Thread(target=self._pump_loop, daemon=True)
        self._pumper.start()
        self.conn = pipe.connect(tmp_path)
        self.conn.send_bytes(rpc.encode_request("hello", versions=[1], bridge_version="test"))
        header, _ = rpc.split_reply(self.conn.recv_bytes())
        assert "error" not in header, header
        self.conn.send_bytes(rpc.encode_request("catalogue"))
        cat_header, _ = rpc.split_reply(self.conn.recv_bytes())
        self.catalogue = cat_header
        self.era = protocol.BridgeEra()
        self.mcp_call("initialize", {})

    def _pump_loop(self) -> None:
        while not self._stop.is_set():
            self.host.pump(budget=0.01)
            time.sleep(0.001)

    def call_tool(self, name: str, args: dict) -> bytes:
        self.conn.send_bytes(rpc.encode_request("call", tool=name, args=args))
        _header, body = rpc.split_reply(self.conn.recv_bytes())
        return body

    def mcp_call(self, method: str, params: dict) -> dict:
        line = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode(
            "utf-8"
        )
        reply = protocol.bridge_dispatch(
            line, self.era, catalogue=self.catalogue, call_tool=self.call_tool
        )
        return json.loads(reply)

    def add_box(self) -> None:
        reply = self.mcp_call(
            "tools/call", {"name": "clay_add_primitive", "arguments": {"generator": "box"}}
        )
        assert reply["result"]["isError"] is False, reply

    def close(self) -> None:
        self.conn.close()
        self._stop.set()
        self._pumper.join(timeout=WAIT)
        self.host.stop()


@pytest.fixture
def rpc_bridge(tmp_path):
    b = _RpcBridge(tmp_path)
    try:
        yield b
    finally:
        b.close()


@pytest.mark.perf
def test_bridge_clay_scene_round_trip_on_a_one_object_document(rpc_bridge: _RpcBridge) -> None:
    rpc_bridge.add_box()

    def one_call() -> None:
        reply = rpc_bridge.mcp_call("tools/call", {"name": "clay_scene", "arguments": {}})
        assert reply["result"]["isError"] is False

    median_ms = _median_round_trip_ms(one_call, SMALL_SCENE_RUNS, "bridge clay_scene, 1 object")
    assert median_ms < MAX_MEDIAN_MS_BRIDGE_SMALL_SCENE, f"{median_ms:.3f} ms median"


@pytest.mark.perf
def test_bridge_clay_scene_round_trip_on_a_fifty_object_document(rpc_bridge: _RpcBridge) -> None:
    for _ in range(50):
        rpc_bridge.add_box()

    def one_call() -> None:
        reply = rpc_bridge.mcp_call("tools/call", {"name": "clay_scene", "arguments": {}})
        result = reply["result"]
        assert result["isError"] is False
        assert len(result["structuredContent"]["objects"]) == 50

    median_ms = _median_round_trip_ms(one_call, LARGE_SCENE_RUNS, "bridge clay_scene, 50 objects")
    assert median_ms < MAX_MEDIAN_MS_BRIDGE_LARGE_SCENE, f"{median_ms:.3f} ms median"


@pytest.mark.perf
def test_bridge_tools_list_round_trip(rpc_bridge: _RpcBridge) -> None:
    """``tools/list`` on the legacy MCP path rebuilds every ``Tool`` schema
    from the live registries on every call (``agent_clay.tools``'s own
    docstring) -- the one call in this file whose cost is pure catalogue
    construction, nothing to do with a document at all. Answered from
    ``rpc_bridge.catalogue`` -- the RPC v1 ``catalogue`` reply already
    fetched once at fixture setup, exactly as ``bridge.py``'s own
    ``_Session`` answers it -- so this also measures ``bridge_dispatch``'s
    own per-call cost rather than a fresh catalogue fetch.
    """

    def one_call() -> None:
        reply = rpc_bridge.mcp_call("tools/list", {})
        assert len(reply["result"]["tools"]) >= 26

    median_ms = _median_round_trip_ms(one_call, TOOLS_LIST_RUNS, "bridge tools/list")
    assert median_ms < MAX_MEDIAN_MS_TOOLS_LIST, f"{median_ms:.3f} ms median"


@pytest.mark.perf
def test_bridge_clay_analyze_round_trip_on_a_typical_kitbash(rpc_bridge: _RpcBridge) -> None:
    """``clay.analyze``'s own module docstring states a budget in these
    terms -- "a typical kitbash (a few primitives/figure parts)" -- so this
    builds exactly that rather than a document sized to make some other
    property (triangle count, object count) round: three boxes and three
    cylinders, two of the pairs close enough to actually intersect, so the
    measured cost includes at least one overlap boolean rather than only the
    cheap broad-phase-rejects-everything path.
    """
    positions = [
        (0.0, 0.5, 0.0),
        (1.2, 0.5, 0.0),
        (2.4, 0.5, 0.0),
        (0.0, 1.5, 0.0),
        (1.2, 0.25, 1.5),
        (0.0, 0.5, 3.0),
    ]
    for i, translation in enumerate(positions):
        generator = "box" if i % 2 == 0 else "cylinder"
        reply = rpc_bridge.mcp_call(
            "tools/call",
            {
                "name": "clay_add_primitive",
                "arguments": {"generator": generator, "translation": list(translation)},
            },
        )
        assert reply["result"]["isError"] is False, reply

    def one_call() -> None:
        reply = rpc_bridge.mcp_call("tools/call", {"name": "clay_analyze", "arguments": {}})
        assert reply["result"]["isError"] is False, reply

    median_ms = _median_round_trip_ms(
        one_call, ANALYZE_KITBASH_RUNS, "bridge clay_analyze, kitbash"
    )
    assert median_ms < MAX_MEDIAN_MS_ANALYZE_KITBASH, f"{median_ms:.3f} ms median"
