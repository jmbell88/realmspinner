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

import json
import os
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

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
