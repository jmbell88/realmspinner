"""`python -m warlock mcp` as a real subprocess, against a real `AgentHost`
pumped in-test -- the one place this tranche proves the whole chain (a real
MCP client's stdio, through `bridge.py`'s RPC v1 client, to Studio's RPC v1
server, to `agent_clay.call`) rather than each half on its own.

Modelled on `tests/test_agent_host.py`'s and `tests/test_agent_perf.py`'s own
harness shape: an `AgentHost` started against a throwaway home, a background
thread draining `pump()` the way `main.py:App.frame` would, and a real pipe
connection -- except the peer here is a real `warlock mcp` child process
rather than this test driving the pipe directly, so `WARLOCK_HOME` has to
reach the child through its environment, not just this process's own.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

from warlock.mcp import pipe, protocol, rpc
from warlock.studio import agent_host

WAIT = 20.0


class _Ctx:
    def __init__(self) -> None:
        self.state = SimpleNamespace(clay=None)
        self.settings = SimpleNamespace()
        self.toasts: list[tuple[str, str]] = []

    def toast(self, message: str, level: str = "info") -> None:
        self.toasts.append((message, level))


@pytest.fixture
def host(tmp_path):
    h = agent_host.AgentHost(_Ctx(), tmp_path)
    assert h.start()
    stop = threading.Event()

    def pump_loop() -> None:
        while not stop.is_set():
            h.pump(budget=0.01)
            time.sleep(0.001)

    pumper = threading.Thread(target=pump_loop, daemon=True)
    pumper.start()
    try:
        yield h, tmp_path
    finally:
        stop.set()
        pumper.join(timeout=WAIT)
        h.stop()


def _spawn(home) -> subprocess.Popen:
    env = dict(os.environ)
    env["WARLOCK_HOME"] = str(home)
    return subprocess.Popen(
        [sys.executable, "-m", "warlock", "mcp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )


def _send(proc: subprocess.Popen, message: dict) -> None:
    line = json.dumps(message).encode("utf-8") + b"\n"
    proc.stdin.write(line)
    proc.stdin.flush()


def _readline(proc: subprocess.Popen, timeout: float = WAIT) -> dict:
    """One line of the child's stdout, parsed -- bounded by killing the
    child and failing loudly rather than hanging pytest if it never answers."""
    deadline = time.monotonic() + timeout
    line = b""
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if line:
            break
    if not line:
        proc.kill()
        raise AssertionError(f"no reply from the bridge within {timeout}s")
    return json.loads(line)


def test_legacy_initialize_tools_list_and_tools_call_over_a_real_subprocess(host) -> None:
    _host, home = host
    proc = _spawn(home)
    try:
        _send(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        reply = _readline(proc)
        assert reply["result"]["protocolVersion"]

        _send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        reply = _readline(proc)
        names = {t["name"] for t in reply["result"]["tools"]}
        assert "clay_add_primitive" in names
        assert "warlock_status" in names

        _send(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "clay_add_primitive", "arguments": {"generator": "box"}},
            },
        )
        reply = _readline(proc)
        assert reply["result"]["isError"] is False
    finally:
        proc.stdin.close()
        proc.wait(timeout=WAIT)


def test_two_ordinary_calls_over_a_real_subprocess_emit_no_spurious_list_changed(host) -> None:
    """A regression test for a real bug this tranche's own catalogue-hash
    comparison in ``bridge.py`` exposed: ``AgentHost._serve_rpc_frame``'s
    ``call`` branch used to reply with ``{"hash": rpc.canonical_hash(result)}``
    -- a hash of the *result* -- rather than the catalogue's own hash, so
    the bridge's ``_maybe_refresh_catalogue`` believed the catalogue moved
    after every single call and pushed a ``notifications/tools/list_changed``
    the legacy client never asked for. Two ordinary calls here must produce
    exactly two replies, nothing between them, and each call's own header
    hash must equal ``AgentHost._catalogue_hash()``."""
    real_host, home = host
    proc = _spawn(home)
    try:
        _send(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        _readline(proc)

        for msg_id in (2, 3):
            _send(
                proc,
                {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "method": "tools/call",
                    "params": {"name": "warlock_status", "arguments": {}},
                },
            )
            reply = _readline(proc)
            assert reply["id"] == msg_id
            assert "method" not in reply  # never a notification in this slot
            assert reply["result"]["isError"] is False

        # Nothing else was queued to write -- a spurious notification would
        # have arrived as an extra line before either of the two replies
        # above, or be sitting here now.
        proc.stdin.close()
        remaining, _err = proc.communicate(timeout=WAIT)
        leftover_methods = [
            json.loads(line).get("method") for line in remaining.splitlines() if line
        ]
        assert "notifications/tools/list_changed" not in leftover_methods
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=WAIT)


def test_modern_discover_and_tools_call_over_a_real_subprocess(host) -> None:
    _host, home = host
    proc = _spawn(home)
    try:
        _send(proc, {"jsonrpc": "2.0", "id": 1, "method": "server/discover"})
        reply = _readline(proc)
        assert reply["result"]["resultType"] == "complete"
        version = reply["result"]["supportedVersions"][-1]

        meta = {"_meta": {"io.modelcontextprotocol/protocolVersion": version}}
        _send(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "clay_add_primitive", "arguments": {"generator": "box"}, **meta},
            },
        )
        reply = _readline(proc)
        assert reply["result"]["resultType"] == "complete"
        assert reply["result"]["isError"] is False
    finally:
        proc.stdin.close()
        proc.wait(timeout=WAIT)


def test_a_stdin_line_over_max_frame_is_refused_without_killing_the_connection(host) -> None:
    from warlock.mcp import protocol

    _host, home = host
    proc = _spawn(home)
    try:
        huge = (
            b'{"jsonrpc":"2.0","id":1,"method":"ping","params":{"x":"'
            + b"x" * protocol.MAX_FRAME
            + b'"}}\n'
        )
        proc.stdin.write(huge)
        proc.stdin.flush()
        reply = _readline(proc)
        assert reply["error"]["code"] == -32600

        _send(proc, {"jsonrpc": "2.0", "id": 2, "method": "initialize"})
        reply = _readline(proc)
        assert reply["result"]["protocolVersion"]
    finally:
        proc.stdin.close()
        proc.wait(timeout=WAIT)


# ---------------------------------------------------------------------------
# Resilience: snapshot-served discovery, lazy reconnect, mid-call and
# timeout disconnects. `_FakeStudio` here speaks only `rpc.py`'s wire
# format -- no `AgentHost`, no Clay, no GL -- and is deliberately started
# and stopped at whatever point each test wants, unlike the `host` fixture
# above which is up for the whole test.
# ---------------------------------------------------------------------------


def _snapshot(home, *, instructions: str = "Clay measures in metres.") -> dict:
    """Write `<home>/mcp.catalogue.json`, the same shape
    `agent_host._write_catalogue_snapshot` writes, and return it."""
    payload = rpc.catalogue_payload(
        [rpc.Tool(name="warlock_status", title="Status", description="d", schema={})],
        instructions=instructions,
        server_name="warlock-studio",
        server_version="9.9.9",
    )
    home.joinpath("mcp.catalogue.json").write_text(json.dumps(payload), encoding="utf-8")
    return payload


class _FakeStudio:
    """The app's side of the pipe, speaking only `rpc.py` -- configurable
    per test for the shapes this tranche has to survive: a call that never
    gets a reply (the backstop timeout) and a call whose connection is
    closed mid-flight (an `EOFError` on the bridge's read)."""

    def __init__(
        self,
        home,
        *,
        tool_result: dict | None = None,
        call_timeout: float = 30.0,
        instructions: str = "Clay measures in metres.",
        die_after_call_request: bool = False,
        never_reply_to_call: bool = False,
    ) -> None:
        self.home = home
        self.server = pipe.Server(home)
        self.server.start()
        self._tool_result = tool_result or protocol.ok(protocol.text("done"))
        self._call_timeout = call_timeout
        self._instructions = instructions
        self._die_after_call_request = die_after_call_request
        self._never_reply_to_call = never_reply_to_call
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.calls: list[tuple[str, dict]] = []
        self._conn: Any = None

    def _catalogue(self) -> dict:
        return rpc.catalogue_payload(
            [rpc.Tool(name="warlock_status", title="Status", description="d", schema={})],
            instructions=self._instructions,
            server_name="warlock-studio",
            server_version="9.9.9",
        )

    def _run(self) -> None:
        conn = self.server.accept()
        if conn is None:
            return
        self._conn = conn
        try:
            while True:
                frame = conn.recv_bytes()
                message = rpc.decode_request(frame)
                op = message.get("op")
                if op == "hello":
                    header = rpc.hello_header(
                        message.get("versions"),
                        studio_version="9.9.9",
                        catalogue_hash=self._catalogue()["hash"],
                        call_timeout=self._call_timeout,
                    )
                    conn.send_bytes(rpc.encode_reply(header))
                elif op == "catalogue":
                    conn.send_bytes(rpc.encode_reply(self._catalogue()))
                elif op == "call":
                    self.calls.append((message.get("tool"), message.get("args")))
                    if self._die_after_call_request:
                        conn.close()
                        return
                    if self._never_reply_to_call:
                        # Say nothing, ever, for this connection -- the
                        # bridge's own `conn.poll` backstop is what has to
                        # give up here, not this side closing anything.
                        time.sleep(WAIT * 2)
                        return
                    body = json.dumps(self._tool_result, separators=(",", ":")).encode("utf-8")
                    conn.send_bytes(
                        rpc.encode_reply({"hash": self._catalogue()["hash"]}, body)
                    )
                else:
                    conn.send_bytes(rpc.encode_reply(rpc.unknown_op_header()))
        except (EOFError, OSError):
            pass

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self.server.close()
        # A `never_reply_to_call` run is parked in `time.sleep`, holding the
        # accepted `Connection` (and, on Windows, the pipe instance handle
        # it owns) -- closing the *Listener* above never touches an already
        # accepted connection (see `pipe.py`'s own docstring on exactly this
        # hazard), so this is what actually frees it for the next
        # `pipe.Server` in the same test to bind.
        if self._conn is not None:
            with contextlib.suppress(OSError):
                self._conn.close()

    def join(self) -> None:
        self._thread.join(timeout=WAIT)


def test_bridge_subprocess_exits_1_with_no_snapshot_and_no_app_listening(tmp_path) -> None:
    """Unchanged from before this tranche: no saved catalogue and nothing
    answering the pipe leaves the bridge nothing to serve at all."""
    proc = _spawn(tmp_path)
    _, err = proc.communicate(timeout=WAIT)
    assert proc.returncode == 1
    text = err.decode("utf-8")
    assert "not accepting agent connections" in text
    assert "Settings" in text
    assert "Advanced" in text
    assert "Allow AI agents to drive the Studio" in text


def test_bridge_subprocess_serves_from_snapshot_when_the_app_is_not_running(tmp_path) -> None:
    """A saved catalogue with no app listening: `initialize`/`tools/list`
    answer from it, a `tools/call` gets a readable `isError` naming how to
    switch the feature on, and the process is still alive afterwards --
    none of that used to be true; this bridge used to exit(1) outright."""
    _snapshot(tmp_path)
    proc = _spawn(tmp_path)
    try:
        _send(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        reply = _readline(proc)
        assert reply["result"]["protocolVersion"]

        _send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        reply = _readline(proc)
        assert reply["result"]["tools"][0]["name"] == "warlock_status"

        _send(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "warlock_status", "arguments": {}},
            },
        )
        reply = _readline(proc)
        assert reply["result"]["isError"] is True
        text = reply["result"]["content"][0]["text"]
        assert "not accepting agent connections" in text
        assert "Settings" in text and "Advanced" in text

        # The bridge is still alive and still answers -- it did not exit
        # over a refused call.
        assert proc.poll() is None
        _send(proc, {"jsonrpc": "2.0", "id": 4, "method": "ping"})
        reply = _readline(proc)
        assert reply["result"] == {}
    finally:
        proc.stdin.close()
        proc.wait(timeout=WAIT)


def test_bridge_subprocess_reconnects_once_the_app_starts_after_snapshot_serving(
    tmp_path,
) -> None:
    """Snapshot-served at start-up, then Studio comes up: the next
    `tools/call` after that succeeds for real, against the real fake
    Studio, rather than the stale refusal."""
    _snapshot(tmp_path)
    proc = _spawn(tmp_path)
    studio = None
    try:
        _send(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        _readline(proc)

        _send(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "warlock_status", "arguments": {}},
            },
        )
        reply = _readline(proc)
        assert reply["result"]["isError"] is True  # nothing was listening yet

        studio = _FakeStudio(tmp_path)
        studio.start()

        _send(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "warlock_status", "arguments": {}},
            },
        )
        reply = _readline(proc)
        assert reply["result"]["isError"] is False
        assert reply["result"]["content"] == [{"type": "text", "text": "done"}]
        assert studio.calls == [("warlock_status", {})]
    finally:
        proc.stdin.close()
        proc.wait(timeout=WAIT)
        if studio is not None:
            studio.close()
            studio.join()


def test_bridge_subprocess_notifies_legacy_clients_when_catalogue_hash_changes_on_reconnect(
    tmp_path,
) -> None:
    """The snapshot's instructions differ from the live Studio's -- a
    different `catalogue_hash` at reconnect -- so the legacy-era client
    gets `notifications/tools/list_changed` once the bridge notices."""
    _snapshot(tmp_path, instructions="Clay measures in metres.")
    proc = _spawn(tmp_path)
    studio = None
    try:
        _send(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        _readline(proc)

        studio = _FakeStudio(tmp_path, instructions="Clay measures in furlongs now.")
        studio.start()

        _send(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "warlock_status", "arguments": {}},
            },
        )
        reply = _readline(proc)
        assert reply["result"]["isError"] is False

        notice = _readline(proc)
        assert notice.get("method") == "notifications/tools/list_changed"
    finally:
        proc.stdin.close()
        proc.wait(timeout=WAIT)
        if studio is not None:
            studio.close()
            studio.join()


def test_bridge_subprocess_survives_studio_dying_mid_call_and_reconnects_next_call(
    tmp_path,
) -> None:
    """Studio accepts the `call` request and then the pipe dies before any
    reply arrives -- an `EOFError` on the bridge's read. The bridge answers
    with an `isError` saying the call may or may not have happened, stays
    alive, and the *next* `tools/call` opens a fresh connection and
    succeeds."""
    dying = _FakeStudio(tmp_path, die_after_call_request=True)
    dying.start()
    proc = _spawn(tmp_path)
    healthy = None
    try:
        _send(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        _readline(proc)

        _send(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "warlock_status", "arguments": {}},
            },
        )
        reply = _readline(proc)
        assert reply["result"]["isError"] is True
        text = reply["result"]["content"][0]["text"]
        assert "may or may not have" in text
        assert "new" in text.lower() and "session" in text.lower()

        dying.close()
        dying.join()
        healthy = _FakeStudio(tmp_path)
        healthy.start()

        _send(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "warlock_status", "arguments": {}},
            },
        )
        reply = _readline(proc)
        assert reply["result"]["isError"] is False
        assert healthy.calls == [("warlock_status", {})]
        assert proc.poll() is None
    finally:
        proc.stdin.close()
        proc.wait(timeout=WAIT)
        if healthy is not None:
            healthy.close()
            healthy.join()


def test_bridge_subprocess_survives_a_call_timeout_backstop_and_reconnects_next_call(
    tmp_path,
) -> None:
    """Studio accepts the `call` request and then never answers at all --
    the `conn.poll(call_timeout + 5s)` backstop is what has to give up.
    Same `isError`/"may or may not have happened" shape, the bridge process
    stays alive, and the following `tools/call` reconnects and succeeds."""
    silent = _FakeStudio(tmp_path, call_timeout=0.2, never_reply_to_call=True)
    silent.start()
    proc = _spawn(tmp_path)
    healthy = None
    try:
        _send(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        _readline(proc)

        _send(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "warlock_status", "arguments": {}},
            },
        )
        # The backstop is call_timeout (0.2s) + 5s, so give this a real
        # window rather than the tight WAIT most other reads here use.
        reply = _readline(proc, timeout=WAIT + 10.0)
        assert reply["result"]["isError"] is True
        text = reply["result"]["content"][0]["text"]
        assert "may or may not have" in text

        silent.close()
        silent.join()
        healthy = _FakeStudio(tmp_path)
        healthy.start()

        _send(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "warlock_status", "arguments": {}},
            },
        )
        reply = _readline(proc)
        assert reply["result"]["isError"] is False
        assert healthy.calls == [("warlock_status", {})]
        assert proc.poll() is None
    finally:
        proc.stdin.close()
        proc.wait(timeout=WAIT)
        if healthy is not None:
            healthy.close()
            healthy.join()


def test_legacy_client_lists_and_reads_resources_and_prompts_over_a_real_subprocess(host) -> None:
    """The whole chain this tranche adds: a real MCP client's stdio, through
    `warlock mcp`'s RPC v1 client, to a real `AgentHost`'s RPC v1 server, and
    back -- for `resources/list`, `resources/read` and `prompts/list`/`get`,
    not just `tools/call` (already proven above)."""
    _host, home = host
    proc = _spawn(home)
    try:
        _send(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        _readline(proc)

        _send(proc, {"jsonrpc": "2.0", "id": 2, "method": "resources/list"})
        reply = _readline(proc)
        uris = {r["uri"] for r in reply["result"]["resources"]}
        assert "warlock://clay/conventions" in uris
        assert "warlock://clay/scene" in uris

        _send(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "resources/read",
                "params": {"uri": "warlock://clay/conventions"},
            },
        )
        reply = _readline(proc)
        assert "metres" in reply["result"]["contents"][0]["text"]

        _send(proc, {"jsonrpc": "2.0", "id": 4, "method": "prompts/list"})
        reply = _readline(proc)
        prompt_names = {p["name"] for p in reply["result"]["prompts"]}
        assert "model_from_description" in prompt_names

        _send(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "prompts/get",
                "params": {
                    "name": "model_from_description",
                    "arguments": {"description": "a small barrel"},
                },
            },
        )
        reply = _readline(proc)
        assert "a small barrel" in reply["result"]["messages"][0]["content"]["text"]
    finally:
        proc.stdin.close()
        proc.wait(timeout=WAIT)
