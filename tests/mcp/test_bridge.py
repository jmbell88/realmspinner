"""`realmspinner mcp` -- the real MCP server.

`bridge.main` negotiates Realmspinner's private RPC v1 (`hello`/`catalogue`/
`call`) and then answers whatever MCP era its stdio peer negotiates, via
`protocol.bridge_dispatch` -- see `test_protocol.py` for that dispatcher's
own behaviour, pinned without a real pipe. What this file pins is the
bridge process itself: the readable remedy on stderr when nothing is
listening, a real RPC v1 handshake and `tools/call` round trip against a
fake Realmspinner that only speaks `rpc.py`'s wire format, and the `MAX_FRAME`
bound on one stdin line. There is no relay-hatch escape any more -- Realmspinner's
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

from realmspinner.mcp import bridge, pipe, protocol, rpc


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A throwaway `REALMSPINNER_HOME` this file's `get_config().home` resolves to."""
    import realmspinner.config as config_mod

    monkeypatch.setenv("REALMSPINNER_HOME", str(tmp_path))
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
    assert "Allow AI agents to drive Realmspinner" in err


def test_with_a_token_but_no_listener_main_still_returns_1(home, capsys) -> None:
    pipe.write_token(home)
    assert bridge.main([]) == 1
    assert "not accepting agent connections" in capsys.readouterr().err


def test_bridge_docstring_names_every_fatal_startup_case_the_code_actually_has() -> None:
    """The 2026-09-14 audit (agents-08): `main` has two distinct paths that
    return 1 at start-up rather than serving anything -- no snapshot and no
    reachable Realmspinner (`test_with_nothing_listening_main_returns_1_...` and
    `test_with_a_token_but_no_listener_main_still_returns_1` above), and a
    reachable Realmspinner whose `hello` reply names an RPC version mismatch
    (`_hello` returns `None`, `main` closes the connection and returns 1
    without falling back to a snapshot -- see `main`'s own comment on that
    branch). The module docstring used to claim "exactly one" such case,
    naming only the first; both must be named."""
    doc = bridge.__doc__ or ""
    assert "one case that is fatal" not in doc, "docstring still claims only one fatal case"
    assert "rpc version this bridge does not understand" in doc.lower()
    assert "two cases fatal at start-up" in doc.lower()


# --- a fake Realmspinner speaking RPC v1 -----------------------------------------------


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
                elif op == "resources":
                    conn.send_bytes(
                        rpc.encode_reply(
                            {"resources": self._catalogue()["resources"], "templates": []}
                        )
                    )
                elif op == "read":
                    uri = message.get("uri")
                    if uri == "realmspinner://clay/conventions":
                        conn.send_bytes(
                            rpc.encode_reply(
                                {"uri": uri, "mimeType": "text/markdown"}, b"units are metres"
                            )
                        )
                    else:
                        conn.send_bytes(rpc.encode_reply({"error": {"code": "not_found"}}))
                elif op == "prompts":
                    conn.send_bytes(rpc.encode_reply({"prompts": self._catalogue()["prompts"]}))
                elif op == "prompt":
                    name = message.get("name")
                    arguments = message.get("arguments") or {}
                    if name != "model_from_description":
                        conn.send_bytes(rpc.encode_reply({"error": {"code": "not_found"}}))
                    elif "description" not in arguments:
                        conn.send_bytes(
                            rpc.encode_reply(
                                {"error": {"code": "bad_arguments", "missing": ["description"]}}
                            )
                        )
                    else:
                        conn.send_bytes(
                            rpc.encode_reply(
                                {
                                    "description": "d",
                                    "messages": [
                                        {
                                            "role": "user",
                                            "content": {
                                                "type": "text",
                                                "text": f"build {arguments['description']}",
                                            },
                                        }
                                    ],
                                }
                            )
                        )
                else:
                    conn.send_bytes(rpc.encode_reply(rpc.unknown_op_header()))
        except (EOFError, OSError):
            pass

    def _catalogue(self) -> dict:
        return rpc.catalogue_payload(
            [rpc.Tool(name="realmspinner_status", title="Status", description="d", schema={})],
            instructions="Clay measures in metres.",
            server_name="realmspinner",
            server_version="9.9.9",
            resources=[
                {
                    "uri": "realmspinner://clay/conventions",
                    "name": "clay-conventions",
                    "mimeType": "text/markdown",
                    "text": "units are metres",
                },
                {
                    "uri": "realmspinner://clay/scene",
                    "name": "clay-scene",
                    "mimeType": "application/json",
                },
            ],
            prompts=[
                {
                    "name": "model_from_description",
                    "title": "Model from a description",
                    "description": "d",
                    "arguments": [
                        {"name": "description", "description": "d", "required": True}
                    ],
                }
            ],
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
                "params": {"name": "realmspinner_status", "arguments": {}},
            }
        ),
    ]
    stdin_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    stdout = _patch_stdio(monkeypatch, stdin_bytes)

    assert bridge.main([]) == 0

    replies = [json.loads(line) for line in stdout.getvalue().splitlines() if line]
    assert len(replies) == 3
    assert replies[0]["result"]["protocolVersion"] in protocol.LEGACY
    assert replies[1]["result"]["tools"][0]["name"] == "realmspinner_status"
    assert replies[2]["result"]["content"] == [{"type": "text", "text": "done"}]
    assert studio.calls == [("realmspinner_status", {})]


def test_modern_discover_and_tools_call_round_trip(studio, monkeypatch) -> None:
    meta = {"_meta": {"io.modelcontextprotocol/protocolVersion": protocol.MODERN[0]}}
    lines = [
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "server/discover"}),
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "realmspinner_status", "arguments": {}, **meta},
            }
        ),
    ]
    stdin_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    stdout = _patch_stdio(monkeypatch, stdin_bytes)

    assert bridge.main([]) == 0

    replies = [json.loads(line) for line in stdout.getvalue().splitlines() if line]
    assert replies[0]["result"]["resultType"] == "complete"
    assert replies[1]["result"]["resultType"] == "complete"
    assert replies[1]["result"]["_meta"]["serverInfo"]["name"] == "realmspinner"
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


# --- recv_bytes is bounded, not just the frame it decodes --------------------


class _StubConn:
    """A duck-typed stand-in for a real `Connection`, holding one canned
    reply and recording every `maxlength` `recv_bytes` was called with.
    Real `Connection.recv_bytes(maxlength=...)` enforces the bound at the
    stdlib level (checking the frame's own length prefix before reading its
    body), which this file's real-pipe fixtures already exercise for the
    ordinary request/reply path; this stub only needs to prove *this
    module's* call sites actually pass the argument."""

    def __init__(self, reply: bytes) -> None:
        self._reply = reply
        self.maxlengths: list[object] = []
        self.sent: list[bytes] = []

    def send_bytes(self, data: bytes) -> None:
        self.sent.append(data)

    def poll(self, timeout: float | None = None) -> bool:
        return True

    def recv_bytes(self, maxlength=None) -> bytes:
        self.maxlengths.append(maxlength)
        return self._reply

    def close(self) -> None:
        pass


def test_a_pipe_peer_cannot_force_an_unbounded_recv_bytes_allocation() -> None:
    """The 2026-09-16 audit (agents-04): every `conn.recv_bytes()` call in
    this module (`_hello`, `_fetch_catalogue`, and each of `_Session`'s six
    Realmspinner-reply reads) omitted stdlib's own `maxlength` argument, so a
    confused or hostile peer past the handshake could force this process to
    fully buffer whatever it sent before `rpc.split_reply`'s own length
    check ever got a chance to run -- and `split_reply` (unlike
    `rpc.decode_request`) checks no length at all, so a reply frame had no
    backstop whatsoever without `maxlength` bounding the read itself."""
    seen: list[object] = []

    hello_conn = _StubConn(
        rpc.encode_reply(
            {"rpc": 1, "studio_version": "9.9.9", "catalogue_hash": "h", "call_timeout": 5.0}
        )
    )
    bridge._hello(hello_conn)
    seen += hello_conn.maxlengths

    catalogue_conn = _StubConn(
        rpc.encode_reply({"hash": "h", "tools": [], "instructions": None, "server": {}})
    )
    bridge._fetch_catalogue(catalogue_conn)
    seen += catalogue_conn.maxlengths

    def _session(reply: bytes) -> bridge._Session:
        conn = _StubConn(reply)
        session = bridge._Session("home", conn, {"call_timeout": 5.0}, {"hash": "h"})
        return session, conn

    session, conn = _session(rpc.encode_reply({"hash": "h"}, b"{}"))
    session.call_tool("t", {})
    seen += conn.maxlengths

    session, conn = _session(rpc.encode_reply({"operation_id": "op-1", "status": "working"}))
    session.call_tool_task("t", {})
    seen += conn.maxlengths

    session, conn = _session(
        rpc.encode_reply({"operation_id": "op-1", "status": "completed"}, b"{}")
    )
    session.get_task("op-1")
    seen += conn.maxlengths

    session, conn = _session(rpc.encode_reply({"status": "cancelled"}))
    session.cancel_task("op-1")
    seen += conn.maxlengths

    session, conn = _session(
        rpc.encode_reply({"uri": "u", "mimeType": "text/plain"}, b"hi")
    )
    session.read_resource("u")
    seen += conn.maxlengths

    session, conn = _session(rpc.encode_reply({"description": "d", "messages": []}))
    session.get_prompt("p", {})
    seen += conn.maxlengths

    assert seen, "recv_bytes() was never called"
    assert all(m == rpc.MAX_FRAME for m in seen), (
        f"recv_bytes() was called without maxlength=rpc.MAX_FRAME: {seen}"
    )


# --- resources and prompts, through a real bridge process against a fake Realmspinner --


def test_legacy_client_can_list_and_read_resources_and_prompts(studio, monkeypatch) -> None:
    """The e2e-shaped claim this tranche's plan calls for: a legacy MCP
    client listing resources and prompts, and reading one of each, against
    `bridge.main` -- driven here with a fake Realmspinner speaking RPC v1 directly
    (see `_FakeStudio`) rather than a real `AgentHost` subprocess, which
    `test_bridge_e2e.py` covers separately for `tools/call`."""
    lines = [
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "resources/list"}),
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "resources/read",
                "params": {"uri": "realmspinner://clay/conventions"},
            }
        ),
        json.dumps({"jsonrpc": "2.0", "id": 4, "method": "prompts/list"}),
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "prompts/get",
                "params": {
                    "name": "model_from_description",
                    "arguments": {"description": "a red barrel"},
                },
            }
        ),
    ]
    stdin_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    stdout = _patch_stdio(monkeypatch, stdin_bytes)

    assert bridge.main([]) == 0

    replies = [json.loads(line) for line in stdout.getvalue().splitlines() if line]
    assert {r["uri"] for r in replies[1]["result"]["resources"]} == {
        "realmspinner://clay/conventions",
        "realmspinner://clay/scene",
    }
    assert replies[2]["result"]["contents"][0]["text"] == "units are metres"
    assert {p["name"] for p in replies[3]["result"]["prompts"]} == {"model_from_description"}
    assert replies[4]["result"]["messages"][0]["content"]["text"] == "build a red barrel"


def test_resources_read_not_found_is_minus_32002_on_legacy(studio, monkeypatch) -> None:
    lines = [
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "resources/read",
                "params": {"uri": "realmspinner://nonsense"},
            }
        ),
    ]
    stdout = _patch_stdio(monkeypatch, ("\n".join(lines) + "\n").encode("utf-8"))
    assert bridge.main([]) == 0
    replies = [json.loads(line) for line in stdout.getvalue().splitlines() if line]
    assert replies[1]["error"]["code"] == -32002


def test_prompts_get_missing_argument_is_minus_32602_on_legacy(studio, monkeypatch) -> None:
    lines = [
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "prompts/get",
                "params": {"name": "model_from_description", "arguments": {}},
            }
        ),
    ]
    stdout = _patch_stdio(monkeypatch, ("\n".join(lines) + "\n").encode("utf-8"))
    assert bridge.main([]) == 0
    replies = [json.loads(line) for line in stdout.getvalue().splitlines() if line]
    assert replies[1]["error"]["code"] == -32602


