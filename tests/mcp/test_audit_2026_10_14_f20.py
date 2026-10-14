"""The 2026-10-14 audit's agent-bridge transport / protocol / rpc findings (F20).

One file for the whole slice: the pipe's handshake and close races, the bridge's
"why am I not connected" reason, the era latch, the task-id reuse and the two
hostile-frame guards. Every test name is the claim it pins.
"""

from __future__ import annotations

import contextlib
import json
import multiprocessing.connection as mpconn
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from realmspinner.mcp import bridge, pipe, protocol, rpc

# --- the pipe -------------------------------------------------------------------


def _raw_peer(address: str, *, read_only: bool) -> Any:
    """Open the pipe without ever answering the challenge.

    `read_only` is what the default pipe DACL grants Everyone and Anonymous on
    Windows (FILE_GENERIC_READ): such a peer can open the pipe and can never
    answer, which is the audit's attacker. Elsewhere it is an ordinary silent
    socket.
    """
    if sys.platform == "win32":
        import _winapi

        access = _winapi.GENERIC_READ if read_only else _winapi.GENERIC_READ | _winapi.GENERIC_WRITE
        # The acceptor opens the next instance only after it has taken the
        # last, so a burst of raw dials can find every instance busy for a
        # moment; a real client waits on the pipe, and so does this.
        deadline = time.monotonic() + 3
        while True:
            try:
                return _winapi.CreateFile(
                    address, access, 0, _winapi.NULL, _winapi.OPEN_EXISTING, 0, _winapi.NULL
                )
            except OSError as exc:
                if exc.winerror != _winapi.ERROR_PIPE_BUSY or time.monotonic() > deadline:
                    raise
                time.sleep(0.01)
    import socket

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(address)
    return sock


def _drop_raw(handle: Any) -> None:
    if sys.platform == "win32":
        import _winapi

        with contextlib.suppress(OSError):
            _winapi.CloseHandle(handle)
    else:
        handle.close()


def _accept_in_thread(server: pipe.Server) -> tuple[threading.Thread, list[Any]]:
    got: list[Any] = []
    thread = threading.Thread(target=lambda: got.append(server.accept()), daemon=True)
    thread.start()
    return thread, got


def test_a_peer_holding_only_read_access_cannot_delay_a_legitimate_bridge_handshake(
    tmp_path, monkeypatch
) -> None:
    """The 2026-10-14 audit, finding agents-protocol-01: handshakes ran one at a
    time, so two peers that could open the pipe and never answer cost every
    legitimate bridge the sum of their timeouts. The timeout is stretched so
    the serial cost cannot hide inside the scheduler's noise."""
    monkeypatch.setattr(pipe, "HANDSHAKE_TIMEOUT", 2.0)
    server = pipe.Server(tmp_path)
    server.start()
    thread, got = _accept_in_thread(server)
    silent = [_raw_peer(server.address, read_only=True) for _ in range(2)]
    try:
        started = time.monotonic()
        conn = pipe.connect(tmp_path)
        elapsed = time.monotonic() - started
        thread.join(5)
        assert got and got[0] is not None, "the legitimate bridge was never admitted"
        assert elapsed < 1.0, f"a legitimate bridge waited {elapsed:.1f}s behind silent peers"
        conn.close()
        got[0].close()
    finally:
        for handle in silent:
            _drop_raw(handle)
        server.close()


def test_a_flood_of_silent_peers_past_the_cap_still_serves_a_legitimate_bridge(
    tmp_path, monkeypatch
) -> None:
    """The in-flight handshake set is bounded; the oldest silent peer is the one
    dropped, so a flood cannot hold every slot against the bridge that follows."""
    monkeypatch.setattr(pipe, "HANDSHAKE_TIMEOUT", 3.0)
    monkeypatch.setattr(pipe, "MAX_PENDING_HANDSHAKES", 2)
    server = pipe.Server(tmp_path)
    server.start()
    thread, got = _accept_in_thread(server)
    silent = []
    try:
        for _ in range(5):
            silent.append(_raw_peer(server.address, read_only=True))
        started = time.monotonic()
        conn = pipe.connect(tmp_path)
        assert time.monotonic() - started < 1.5
        thread.join(5)
        assert got and got[0] is not None
        conn.close()
        got[0].close()
    finally:
        for handle in silent:
            _drop_raw(handle)
        server.close()


def test_server_close_returns_promptly_while_a_peer_is_mid_handshake(tmp_path, monkeypatch) -> None:
    """The 2026-10-14 audit, finding agents-protocol-02: `close()` self-dialled
    with no deadline and, with a peer mid-handshake, no pipe instance was free,
    so the frame thread waited out the whole handshake (4.5 s measured)."""
    monkeypatch.setattr(pipe, "HANDSHAKE_TIMEOUT", 4.0)
    server = pipe.Server(tmp_path)
    server.start()
    thread, _ = _accept_in_thread(server)
    time.sleep(0.3)
    peer = _raw_peer(server.address, read_only=True)
    time.sleep(0.5)  # the handshake is now in flight and will not be answered
    try:
        started = time.monotonic()
        server.close()
        elapsed = time.monotonic() - started
        assert elapsed < 2.0, f"close() blocked {elapsed:.1f}s behind a half-finished handshake"
        thread.join(5)
        assert not thread.is_alive(), "accept() was left blocked after close()"
    finally:
        _drop_raw(peer)


@pytest.mark.skipif(sys.platform != "win32", reason="the leaked instance is a Windows pipe handle")
def test_start_close_start_in_quick_succession_never_leaves_a_pipe_instance_behind(
    tmp_path, monkeypatch
) -> None:
    """The 2026-10-14 audit, finding agents-protocol-05: a `close()` landing
    between the stdlib listener's guard and its inner `accept()` left the thread
    creating a fresh pipe instance on a closed listener, and CPython opens the
    pipe with FILE_FLAG_FIRST_PIPE_INSTANCE, so the next `start()` raised
    `PermissionError`. The gap is widened here with a sleep at the inner call;
    the 1-in-300 race the audit measured becomes every time."""
    real_accept = mpconn.PipeListener.accept

    def slow_accept(self: Any) -> Any:
        time.sleep(0.3)
        return real_accept(self)

    monkeypatch.setattr(mpconn.PipeListener, "accept", slow_accept)
    first = pipe.Server(tmp_path)
    first.start()
    thread, _ = _accept_in_thread(first)
    time.sleep(0.05)
    first.close()
    thread.join(5)
    time.sleep(0.8)  # long enough for a leaked instance to have been created

    again = pipe.Server(tmp_path)
    try:
        again.start()  # PermissionError while the leaked instance exists
    finally:
        again.close()


def test_accept_survives_a_transient_oserror_from_a_live_listener(tmp_path, monkeypatch) -> None:
    """The 2026-10-14 audit, finding agents-protocol-06: any `OSError` from a
    live listener (ECONNABORTED, EMFILE, a failed CreateNamedPipe) was read as
    'closed' and ended the host's listener thread for good."""
    real_accept = mpconn.Listener.accept
    calls = {"n": 0}

    def flaky(self: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionAbortedError("a peer gave up between SYN and accept")
        return real_accept(self)

    monkeypatch.setattr(mpconn.Listener, "accept", flaky)
    server = pipe.Server(tmp_path)
    server.start()
    thread, got = _accept_in_thread(server)
    try:
        time.sleep(0.3)
        assert not got, "a transient accept failure was reported as 'the server is closed'"
        conn = pipe.connect(tmp_path)
        thread.join(5)
        assert got and got[0] is not None
        conn.close()
        got[0].close()
    finally:
        server.close()


# --- the bridge -------------------------------------------------------------------


class _OneReplyConn:
    """A conn that answers whatever it is sent with one canned reply frame."""

    def __init__(self, reply: bytes | None) -> None:
        self._reply = reply
        self.sent: list[bytes] = []

    def send_bytes(self, data: bytes) -> None:
        self.sent.append(data)

    def poll(self, timeout: float | None = None) -> bool:
        return self._reply is not None

    def recv_bytes(self, maxlength: Any = None) -> bytes:
        assert self._reply is not None
        return self._reply

    def close(self) -> None:
        pass


def _snapshot_session(tmp_path: Any) -> bridge._Session:
    return bridge._Session.from_snapshot(
        tmp_path, {"hash": "h", "tools": [], "instructions": "", "server": {}, "resources": []}
    )


def _tool_text(body: bytes) -> str:
    result = json.loads(body)
    assert result["isError"] is True
    return result["content"][0]["text"]


@pytest.mark.parametrize(
    ("reply", "needle"),
    [
        (rpc.encode_reply({"error": {"code": "busy"}}), "another agent client"),
        (
            rpc.encode_reply({"error": {"code": "rpc_version", "supported": [2]}}),
            "different agent RPC version",
        ),
        (None, "did not answer"),
    ],
    ids=["busy", "rpc_version", "silent"],
)
def test_a_busy_studio_is_reported_to_the_agent_as_busy_not_as_switched_off(
    tmp_path, monkeypatch, reply, needle
) -> None:
    """The 2026-10-14 audit, finding agents-host-01: a reachable Realmspinner that
    answered `hello` with `busy` or `rpc_version`, or said nothing, was reported
    to the agent as "not accepting agent connections ... switch on Settings",
    though the switch was already on. The real reason went to stderr only."""
    monkeypatch.setattr(bridge, "_connect", lambda home: _OneReplyConn(reply))
    monkeypatch.setattr(bridge, "SETUP_TIMEOUT", 0.01)
    session = _snapshot_session(tmp_path)

    for text in (
        _tool_text(session.call_tool("clay_scene", {})),
        _tool_text(session.call_tool_task("clay_scene", {})),  # type: ignore[arg-type]
    ):
        assert needle in text
        assert "Allow AI agents" not in text, "the agent was sent to toggle a switch that is on"


def test_a_pipe_nobody_answers_still_says_to_switch_the_agent_server_on(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(bridge, "_connect", lambda home: None)
    session = _snapshot_session(tmp_path)
    assert bridge.NOT_ACCEPTING_MESSAGE in _tool_text(session.call_tool("clay_scene", {}))


def test_bridge_marks_every_dynamic_resource_private(tmp_path, monkeypatch) -> None:
    """The 2026-10-14 audit, finding agents-host-08: the bridge's hand-spelled
    `endswith("/scene")` could not follow a third dynamic resource. Privacy is
    now read from the catalogue, so every `DYNAMIC_URIS` entry is private and
    every static one public, and a dynamic resource nobody has heard of is too."""
    from realmspinner.studio import agent_resources

    catalogue = {
        "hash": "h",
        "tools": [],
        "instructions": "",
        "server": {},
        "resources": [
            *agent_resources.catalogue_resources(),
            {"uri": "clay://a-third/dynamic", "name": "x", "mimeType": "application/json"},
        ],
    }
    session = bridge._Session.from_snapshot(tmp_path, catalogue)
    session.conn = _OneReplyConn(
        rpc.encode_reply({"uri": "u", "mimeType": "application/json"}, b"{}")
    )

    dynamic = set(agent_resources.DYNAMIC_URIS) | {"clay://a-third/dynamic"}
    static = {e["uri"] for e in catalogue["resources"] if "text" in e}
    assert static, "the catalogue lists no static resource to compare against"
    for uri in sorted(dynamic | static):
        read = session.read_resource(uri)
        assert read is not None
        expected = "public" if uri in static else "private"
        assert read["cacheScope"] == expected, uri
        assert (read["ttlMs"] == 0) == (expected == "private"), uri


def test_a_reused_operation_id_after_reconnect_does_not_inherit_the_old_tasks_created_at_or_result(
    tmp_path, monkeypatch
) -> None:
    """The 2026-10-14 audit, finding agents-protocol-03. Realmspinner numbers its
    operations `op-1`... afresh on every connection, but the bridge's era state
    and session outlive each reconnect. Two halves: a reused id is a new task
    and gets its own `createdAt`; and a handle from the previous connection is
    "unknown task" rather than being answered by whatever now carries its id."""
    # Protocol half: the second mint of "op-1" is not "already known".
    stamps = iter(["2026-10-14T10:00:00Z", "2026-10-14T10:00:01Z", "never-asked-for"])
    monkeypatch.setattr(protocol, "_iso_now", lambda: next(stamps))
    era = protocol.BridgeEra()
    assert era.note_task_created("op-1") == "2026-10-14T10:00:00Z"
    assert era.note_task_created("op-1") == "2026-10-14T10:00:01Z"
    assert era.task_created_at("op-1") == "2026-10-14T10:00:01Z"

    # Bridge half: the old connection's task, polled after a reconnect.
    first = _OneReplyConn(rpc.encode_reply({"operation_id": "op-1", "status": "working"}))
    session = bridge._Session(tmp_path, first, {"call_timeout": 5.0}, {"hash": "h"})
    assert session.call_tool_task("clay_scene", {}) == ("op-1", "working")

    session._disconnect()  # Studio restarted, or the agent switch was toggled
    second = _OneReplyConn(rpc.encode_reply({"operation_id": "op-1", "status": "completed"}, b"{}"))
    monkeypatch.setattr(bridge, "_connect", lambda home: second)
    hello = rpc.encode_reply(
        {"rpc": 1, "studio_version": "x", "catalogue_hash": "h", "call_timeout": 5}
    )
    replies = iter([hello])
    second.recv_bytes = lambda maxlength=None: next(replies)  # type: ignore[method-assign]
    assert session.get_task("op-1") is None, (
        "an old connection's task id was answered by the new one"
    )
    assert session.cancel_task("op-1") is None
    assert not second.sent, "the bridge asked Realmspinner about a task its connection never minted"


def test_a_task_minted_on_the_live_connection_is_still_polled() -> None:
    """The other side of the same change: tracking must not break the ordinary
    mint-then-poll path."""
    conn = _OneReplyConn(rpc.encode_reply({"operation_id": "op-1", "status": "working"}))
    session = bridge._Session("home", conn, {"call_timeout": 5.0}, {"hash": "h"})
    assert session.call_tool_task("clay_scene", {}) == ("op-1", "working")
    conn._reply = rpc.encode_reply({"operation_id": "op-1", "status": "completed"}, b'{"ok":1}')
    assert session.get_task("op-1") == ("completed", b'{"ok":1}')


def _discover(state: protocol.BridgeEra, **params: Any) -> dict[str, Any]:
    message: dict[str, Any] = {"jsonrpc": "2.0", "id": 1, "method": "server/discover"}
    if params:
        message["params"] = params
    reply = protocol.bridge_dispatch(
        json.dumps(message).encode(),
        state,
        catalogue={"hash": "h", "tools": [], "instructions": "", "server": {}},
        call_tool=lambda n, a: b"{}",
    )
    assert reply is not None
    return json.loads(reply)


def test_initialize_after_server_discover_is_served_for_every_legacy_version_it_advertised() -> (
    None
):
    """The 2026-10-14 audit, finding agents-protocol-04: `server/discover`
    advertised every legacy revision and latched the connection to the modern
    era, so the `initialize` a client then sent from that list was refused."""
    advertised = _discover(protocol.BridgeEra())["result"]["supportedVersions"]
    for version in protocol.LEGACY:
        assert version in advertised
        state = protocol.BridgeEra()
        _discover(state)
        reply = protocol.bridge_dispatch(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "initialize",
                    "params": {"protocolVersion": version},
                }
            ).encode(),
            state,
            catalogue={"hash": "h", "tools": [], "instructions": "", "server": {}},
            call_tool=lambda n, a: b"{}",
        )
        assert reply is not None
        assert "error" not in json.loads(reply), (version, reply)
        assert state.era == "legacy"


def test_a_discover_that_names_a_modern_version_still_locks_the_modern_era() -> None:
    """The other side of the same change: a client that already spoke the
    modern version on its `server/discover` has chosen, and keeps that choice."""
    state = protocol.BridgeEra()
    _discover(state, _meta={protocol.MODERN_META_KEY: protocol.MODERN[0]})
    assert state.era == "modern"


def test_bridge_dispatch_and_decode_refuse_the_same_malformed_frames() -> None:
    """The 2026-10-14 audit, finding agents-protocol-08: `protocol.decode` had
    no caller and its comment said `bridge_dispatch` inherited its hardening,
    though that function parses for itself. There is now one parse site, and
    this pins that it refuses every shape `decode` was written for."""
    assert not hasattr(protocol, "decode"), "a second, caller-less parse site came back"
    catalogue = {"hash": "h", "tools": [], "instructions": "", "server": {}}
    state = protocol.BridgeEra()
    state.era = "modern"
    frames = {
        "nested": (b"[" * 20000) + (b"]" * 20000),
        "huge integer": b'{"id": ' + b"9" * 5000 + b"}",
        "junk": b"{not json",
        "not utf-8": b"\xff\xfe\x00\x01",
    }
    for name, frame in frames.items():
        reply = protocol.bridge_dispatch(
            frame, state, catalogue=catalogue, call_tool=lambda n, a: b"{}"
        )
        assert reply is not None and json.loads(reply)["error"]["code"] == -32700, name
    for name, frame in {"scalar": b"42", "string": b'"hi"'}.items():
        reply = protocol.bridge_dispatch(
            frame, state, catalogue=catalogue, call_tool=lambda n, a: b"{}"
        )
        assert reply is not None and json.loads(reply)["error"]["code"] == -32600, name


# --- the rpc wire ------------------------------------------------------------------


def test_a_deeply_nested_rpc_frame_is_bad_request_not_a_dead_session() -> None:
    """The 2026-10-14 audit, finding agents-host-03: `rpc.decode_request` and
    `looks_like_rpc` let `RecursionError` out of `json.loads`, which ended the
    host's whole session thread (and a busy-refusal thread the same way)."""
    from realmspinner.studio import agent_host

    nested = (b"[" * 20000) + (b"]" * 20000)
    assert len(nested) < rpc.MAX_FRAME
    with pytest.raises(ValueError):
        rpc.decode_request(nested)
    assert rpc.looks_like_rpc(nested) is False

    host = agent_host.AgentHost.__new__(agent_host.AgentHost)
    reply = host._serve_rpc_frame(None, None, nested)  # type: ignore[arg-type]
    assert rpc.split_reply(reply)[0] == rpc.bad_request_header()


def test_rpc_module_doc_names_every_error_code_the_host_emits() -> None:
    """The 2026-10-14 audit, finding agents-host-09: `saturated` was a reply a
    peer implementing RPC v1 from `rpc.py`'s own wire spec could not know."""
    import re
    from pathlib import Path

    from realmspinner.studio import agent_host

    emitted: set[str] = set()
    for module in (agent_host, rpc):
        source = Path(module.__file__).read_text(encoding="utf-8")
        emitted |= set(re.findall(r'"code":\s*"(\w+)"', source))
    assert {"busy", "saturated", "not_found", "bad_arguments", "rpc_version"} <= emitted
    emitted |= {"unknown_op", "bad_request"}  # built by `rpc`'s own helper functions
    missing = sorted(code for code in emitted if code not in (rpc.__doc__ or ""))
    assert not missing, f"rpc.py's wire spec never mentions: {missing}"


# --- the pipe, after the redesign ----------------------------------------------------


def test_the_acceptor_thread_ends_quietly_when_the_listener_is_cleared_mid_accept() -> None:
    """Replaces the 2026-09-26 audit's double-read pin, which was written
    against an `accept()` that drove the listener on the caller's thread.
    stdlib's own `Listener.accept` reads its inner listener twice, so a
    `close()` between the reads raises `AttributeError` there; the acceptor
    must treat any failure after close as the end, not as a crash."""

    class _Racy:
        address = "racy"

        def accept(self) -> Any:
            raise AttributeError("'NoneType' object has no attribute 'accept'")

        def close(self) -> None:
            pass

    server = pipe.Server(Path("."))  # never started: only the acceptor loop is driven
    racy = _Racy()
    server._listener = racy  # type: ignore[assignment]
    outcome: list[BaseException] = []

    def run() -> None:
        try:
            server._accept_loop(racy)  # type: ignore[arg-type]
        except BaseException as exc:  # noqa: BLE001
            outcome.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    time.sleep(0.15)
    server._listener = None  # close() landed
    server._closed.set()
    thread.join(5)
    assert not thread.is_alive(), "the acceptor never noticed the server had closed"
    assert not outcome, f"the acceptor thread died with {outcome!r}"
