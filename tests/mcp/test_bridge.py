"""`warlock mcp` -- the real MCP server.

`bridge.main` negotiates Studio's private RPC v1 (`hello`/`catalogue`/
`call`) and then answers whatever MCP era its stdio peer negotiates, via
`protocol.bridge_dispatch` -- see `test_protocol.py` for that dispatcher's
own behaviour, pinned without a real pipe. What this file pins is the
bridge process itself: the readable remedy on stderr when nothing is
listening, a real RPC v1 handshake and `tools/call` round trip against a
fake Studio that only speaks `rpc.py`'s wire format, and the `MAX_FRAME`
bound on one stdin line. There is no relay-hatch escape any more -- Studio's
pipe answers RPC v1 only now, so a byte-for-byte relay would have nothing to
talk to.
"""

from __future__ import annotations

import io
import json
import sys
import threading
from types import SimpleNamespace

import pytest

from warlock.mcp import bridge, pipe, protocol, rpc


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A throwaway `WARLOCK_HOME` this file's `get_config().home` resolves to."""
    import warlock.config as config_mod

    monkeypatch.setenv("WARLOCK_HOME", str(tmp_path))
    monkeypatch.setattr(config_mod, "_config", None)
    return tmp_path


def _patch_stdio(monkeypatch, stdin_bytes: bytes) -> io.BytesIO:
    stdin = io.BytesIO(stdin_bytes)
    stdout = io.BytesIO()
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=stdin))
    monkeypatch.setattr(sys, "stdout", SimpleNamespace(buffer=stdout))
    return stdout


# --- nothing listening -----------------------------------------------------------


def test_with_nothing_listening_main_returns_1_and_names_the_settings_toggle(
    home, capsys
) -> None:
    assert bridge.main([]) == 1
    err = capsys.readouterr().err
    assert "Settings" in err
    assert "Advanced" in err
    assert "Allow AI agents to drive the Studio" in err


def test_with_a_token_but_no_listener_main_still_returns_1(home, capsys) -> None:
    pipe.write_token(home)
    assert bridge.main([]) == 1
    assert "not accepting agent connections" in capsys.readouterr().err


# --- a fake Studio speaking RPC v1 -----------------------------------------------


class _FakeStudio:
    """The app's side of the pipe, speaking only `rpc.py` -- enough of
    `AgentHost._serve_rpc_frame` to drive `bridge.main`'s real handshake and
    one `tools/call`, with no Clay, no GL, nothing studio-shaped at all."""

    def __init__(self, home, *, tool_result: dict | None = None) -> None:
        self.server = pipe.Server(home)
        self.server.start()
        self._tool_result = tool_result or protocol.ok(protocol.text("done"))
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.calls: list[tuple[str, dict]] = []

    def _run(self) -> None:
        conn = self.server.accept()
        if conn is None:
            return
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
                        call_timeout=5.0,
                    )
                    conn.send_bytes(rpc.encode_reply(header))
                elif op == "catalogue":
                    conn.send_bytes(rpc.encode_reply(self._catalogue()))
                elif op == "call":
                    self.calls.append((message.get("tool"), message.get("args")))
                    body = json.dumps(self._tool_result, separators=(",", ":")).encode("utf-8")
                    conn.send_bytes(rpc.encode_reply({"hash": self._catalogue()["hash"]}, body))
                else:
                    conn.send_bytes(rpc.encode_reply(rpc.unknown_op_header()))
        except (EOFError, OSError):
            pass

    def _catalogue(self) -> dict:
        return rpc.catalogue_payload(
            [rpc.Tool(name="warlock_status", title="Status", description="d", schema={})],
            instructions="Clay measures in metres.",
            server_name="warlock-studio",
            server_version="9.9.9",
        )

    def start(self) -> None:
        self._thread.start()

    def join(self) -> None:
        self._thread.join(timeout=10)

    def close(self) -> None:
        self.server.close()


@pytest.fixture
def studio(home):
    fake = _FakeStudio(home)
    fake.start()
    try:
        yield fake
    finally:
        fake.close()
        fake.join()


def test_legacy_initialize_tools_list_and_tools_call_round_trip(studio, home, monkeypatch) -> None:
    lines = [
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "warlock_status", "arguments": {}},
            }
        ),
    ]
    stdin_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    stdout = _patch_stdio(monkeypatch, stdin_bytes)

    assert bridge.main([]) == 0

    replies = [json.loads(line) for line in stdout.getvalue().splitlines() if line]
    assert len(replies) == 3
    assert replies[0]["result"]["protocolVersion"] in protocol.LEGACY
    assert replies[1]["result"]["tools"][0]["name"] == "warlock_status"
    assert replies[2]["result"]["content"] == [{"type": "text", "text": "done"}]
    assert studio.calls == [("warlock_status", {})]


def test_modern_discover_and_tools_call_round_trip(studio, monkeypatch) -> None:
    meta = {"_meta": {"io.modelcontextprotocol/protocolVersion": protocol.MODERN[0]}}
    lines = [
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "server/discover"}),
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "warlock_status", "arguments": {}, **meta},
            }
        ),
    ]
    stdin_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    stdout = _patch_stdio(monkeypatch, stdin_bytes)

    assert bridge.main([]) == 0

    replies = [json.loads(line) for line in stdout.getvalue().splitlines() if line]
    assert replies[0]["result"]["resultType"] == "complete"
    assert replies[1]["result"]["resultType"] == "complete"
    assert replies[1]["result"]["_meta"]["serverInfo"]["name"] == "warlock-studio"
    assert replies[1]["result"]["content"] == [{"type": "text", "text": "done"}]


def test_an_oversize_stdin_line_is_refused_and_the_connection_keeps_going(
    studio, monkeypatch
) -> None:
    huge_line = (
        b'{"jsonrpc":"2.0","id":1,"method":"ping","params":{"x":"'
        + b"x" * protocol.MAX_FRAME
        + b'"}}\n'
    )
    good_line = (
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "initialize"}).encode("utf-8") + b"\n"
    )
    stdout = _patch_stdio(monkeypatch, huge_line + good_line)

    assert bridge.main([]) == 0

    replies = [json.loads(line) for line in stdout.getvalue().splitlines() if line]
    assert replies[0]["error"]["code"] == -32600
    assert replies[1]["result"]["protocolVersion"] in protocol.LEGACY


