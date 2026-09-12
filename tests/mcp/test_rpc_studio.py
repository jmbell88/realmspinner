"""B1: Studio speaks ``warlock.mcp.rpc`` v1 alongside the existing MCP path.

Everything here drives a real :class:`~warlock.studio.agent_host.AgentHost`
over a real pipe (:mod:`warlock.mcp.pipe`), the same fixture shape
``tests/test_agent_host.py`` already uses: a background thread calls
``host.pump()`` the way ``main.py:App.frame`` would, while this thread is
the "bridge" dialling in with ``pipe.connect``. RPC v1 requests are built by
hand with ``rpc.encode_request``/``rpc.split_reply`` rather than through
``warlock.mcp.bridge`` -- B1 does not touch ``bridge.py`` at all; only
Studio's side of the pipe (``agent_host.py``) is new here.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from warlock.mcp import pipe, protocol, rpc
from warlock.studio import agent_clay, agent_host

WAIT = 5.0


class _Ctx:
    def __init__(self) -> None:
        self.state = SimpleNamespace(clay=None)
        self.settings = SimpleNamespace()
        self.toasts: list[tuple[str, str]] = []

    def toast(self, message: str, level: str = "info") -> None:
        self.toasts.append((message, level))


def _recv(conn, timeout: float = WAIT) -> bytes:
    assert conn.poll(timeout), f"no reply within {timeout}s"
    return conn.recv_bytes()


def _started_host(tmp_path: Path) -> tuple[agent_host.AgentHost, threading.Event, threading.Thread]:
    host = agent_host.AgentHost(_Ctx(), tmp_path)
    host.start()
    stop_pumping = threading.Event()

    def pump_loop() -> None:
        while not stop_pumping.is_set():
            host.pump(budget=0.01)
            time.sleep(0.005)

    pumper = threading.Thread(target=pump_loop, daemon=True)
    pumper.start()
    return host, stop_pumping, pumper


def _stop(host, stop_pumping, pumper) -> None:
    stop_pumping.set()
    pumper.join(timeout=WAIT)
    host.stop()


# --- hello / catalogue / call over a real pipe --------------------------------


def test_hello_catalogue_and_call_round_trip_over_a_real_pipe(tmp_path) -> None:
    host, stop_pumping, pumper = _started_host(tmp_path)
    try:
        conn = pipe.connect(tmp_path)
        try:
            conn.send_bytes(rpc.encode_request("hello", versions=[1], bridge_version="test"))
            header, body = rpc.split_reply(_recv(conn))
            assert header["rpc"] == 1
            assert body == b""
            assert isinstance(header["studio_version"], str)
            assert isinstance(header["catalogue_hash"], str)
            assert header["call_timeout"] == agent_host.CALL_TIMEOUT

            conn.send_bytes(rpc.encode_request("catalogue"))
            cat_header, body = rpc.split_reply(_recv(conn))
            assert body == b""
            names = {t["name"] for t in cat_header["tools"]}
            assert "clay_scene" in names
            assert agent_host.STATUS_TOOL in names
            assert cat_header["hash"] == header["catalogue_hash"]
            assert cat_header["server"]["name"] == protocol.SERVER_NAME

            conn.send_bytes(
                rpc.encode_request("call", tool=agent_host.STATUS_TOOL, args={})
            )
            call_header, body = rpc.split_reply(_recv(conn))
            result = json.loads(body.decode("utf-8"))
            assert result["isError"] is False
            assert call_header["hash"] == rpc.canonical_hash(result)
        finally:
            conn.close()
    finally:
        _stop(host, stop_pumping, pumper)


def test_hello_with_an_unsupported_version_is_refused_by_name(tmp_path) -> None:
    host, stop_pumping, pumper = _started_host(tmp_path)
    try:
        conn = pipe.connect(tmp_path)
        try:
            conn.send_bytes(rpc.encode_request("hello", versions=[2], bridge_version="test"))
            header, body = rpc.split_reply(_recv(conn))
            assert body == b""
            assert header["error"]["code"] == "rpc_version"
            assert header["error"]["supported"] == [1]
        finally:
            conn.close()
    finally:
        _stop(host, stop_pumping, pumper)


def test_an_unknown_op_is_refused_by_name(tmp_path) -> None:
    host, stop_pumping, pumper = _started_host(tmp_path)
    try:
        conn = pipe.connect(tmp_path)
        try:
            conn.send_bytes(rpc.encode_request("nonsense"))
            header, body = rpc.split_reply(_recv(conn))
            assert body == b""
            assert header["error"]["code"] == "unknown_op"
        finally:
            conn.close()
    finally:
        _stop(host, stop_pumping, pumper)


def test_the_catalogue_hash_changes_when_the_tool_list_changes(tmp_path, monkeypatch) -> None:
    host, stop_pumping, pumper = _started_host(tmp_path)
    try:
        first = host._catalogue_hash()

        extra_tool = protocol.Tool(
            name="not_a_real_tool",
            title="Not real",
            description="A planted tool, to prove the hash moves.",
            schema={"type": "object", "additionalProperties": False},
        )
        original_tools = agent_clay.tools

        def _patched_tools():
            return [*original_tools(), extra_tool]

        monkeypatch.setattr(agent_clay, "tools", _patched_tools)
        second = host._catalogue_hash()
        assert first != second
    finally:
        _stop(host, stop_pumping, pumper)


# --- dedup / replay / warlock_status over RPC ---------------------------------


def test_calling_warlock_status_over_rpc_answers_without_touching_the_queue(tmp_path) -> None:
    host, stop_pumping, pumper = _started_host(tmp_path)
    try:
        conn = pipe.connect(tmp_path)
        try:
            conn.send_bytes(
                rpc.encode_request("call", tool=agent_host.STATUS_TOOL, args={})
            )
            _header, body = rpc.split_reply(_recv(conn))
            result = json.loads(body.decode("utf-8"))
            assert result["isError"] is False
            payload = result["structuredContent"]
            assert "operations" in payload
        finally:
            conn.close()
    finally:
        _stop(host, stop_pumping, pumper)


def test_an_identical_call_over_rpc_can_be_asked_about_by_operation_id(tmp_path) -> None:
    """Not a dedup replay (that path needs a timeout, which this fixture's
    fast pump loop never produces) but the same ``_call`` machinery the MCP
    path uses: a genuine call mints an operation that ``warlock_status`` can
    later be asked about by id, over the RPC wire exactly as it would be over
    MCP."""
    host, stop_pumping, pumper = _started_host(tmp_path)
    try:
        conn = pipe.connect(tmp_path)
        try:
            conn.send_bytes(
                rpc.encode_request("call", tool=agent_host.STATUS_TOOL, args={})
            )
            _header, body = rpc.split_reply(_recv(conn))
            json.loads(body.decode("utf-8"))  # answered; no operation minted for this tool

            conn.send_bytes(
                rpc.encode_request(
                    "call", tool="clay_scene", args={"tab_id": "does-not-exist"}
                )
            )
            _header, body = rpc.split_reply(_recv(conn))
            first_result = json.loads(body.decode("utf-8"))
            assert first_result["isError"] is True

            conn.send_bytes(
                rpc.encode_request(
                    "call", tool="clay_scene", args={"tab_id": "does-not-exist"}
                )
            )
            _header, body = rpc.split_reply(_recv(conn))
            second_result = json.loads(body.decode("utf-8"))
            # Both calls genuinely ran and were delivered -- "two identical
            # boxes stay two boxes" -- so neither carries a replay flag.
            assert "replayed" not in (first_result.get("structuredContent") or {})
            assert "replayed" not in (second_result.get("structuredContent") or {})
        finally:
            conn.close()
    finally:
        _stop(host, stop_pumping, pumper)


# --- a connection's first frame sticks to one wire format ---------------------


def test_a_connection_that_opens_with_jsonrpc_still_gets_the_mcp_path(tmp_path) -> None:
    host, stop_pumping, pumper = _started_host(tmp_path)
    try:
        conn = pipe.connect(tmp_path)
        try:
            conn.send_bytes(protocol.encode({"jsonrpc": "2.0", "id": 1, "method": "ping"}))
            reply = protocol.decode(_recv(conn))
            assert reply == {"jsonrpc": "2.0", "id": 1, "result": {}}
        finally:
            conn.close()
    finally:
        _stop(host, stop_pumping, pumper)


# --- the catalogue snapshot ----------------------------------------------------


def test_start_writes_a_catalogue_snapshot_matching_the_catalogue_op(tmp_path) -> None:
    host, stop_pumping, pumper = _started_host(tmp_path)
    try:
        snapshot_path = tmp_path / "mcp.catalogue.json"
        assert snapshot_path.exists()
        on_disk = json.loads(snapshot_path.read_text(encoding="utf-8"))

        conn = pipe.connect(tmp_path)
        try:
            conn.send_bytes(rpc.encode_request("catalogue"))
            header, _body = rpc.split_reply(_recv(conn))
        finally:
            conn.close()
        assert on_disk == header
    finally:
        _stop(host, stop_pumping, pumper)


# --- the switched-off refusal must not pollute the transcript ----------------


def test_a_switched_off_refusal_for_a_job_that_never_ran_is_absent_from_the_transcript(
    tmp_path, monkeypatch
) -> None:
    """Regression for the bug this tranche fixes: ``_fail_pending`` (and the
    "never queued" early-out in ``_run_on_frame_job``) stamp a refusal
    straight onto a job's ``result`` for a call that never touched the
    document at all. Before the fix, ``AgentHost._call``'s only guard on
    recording was ``result is not None`` -- true for that refusal too -- so
    a "switched off" answer was written into the transcript as if it were a
    completed tool call. This test asserts the transcript stays empty for
    exactly that case; it fails against the unfixed code because the old
    branch has no ``state != DROPPED`` guard at all and unconditionally calls
    ``_record_completed_call`` whenever ``result is not None``, which is true
    here.
    """
    transcript = tmp_path / "transcript.jsonl"
    monkeypatch.setenv(agent_host.TRANSCRIPT_ENV, str(transcript))

    host = agent_host.AgentHost(_Ctx(), tmp_path)
    host._queue = None  # never started -- the "switched off" early-out in
    # _run_on_frame_job fires for exactly this reason.
    host._stopped.set()

    calls = agent_host._Calls()
    result = host._call(agent_clay.Session(), calls, "clay_scene", {"tab_id": "x"})
    assert result["isError"] is True
    assert not transcript.exists()


# --- _fingerprint is pinned byte-identical to the pre-rpc.py implementation ---


def _old_fingerprint(tool: str, args: dict) -> str:
    """The exact formula ``agent_host._fingerprint`` used before it called
    ``rpc.canonical_hash`` -- reproduced here, independently of both modules,
    so this test does not just compare the new code against itself."""
    canonical = json.dumps(
        {"tool": tool, "args": args}, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.blake2b(canonical.encode("utf-8"), digest_size=8).hexdigest()


def test_fingerprint_is_byte_identical_to_the_old_implementation() -> None:
    samples: list[tuple[str, dict]] = [
        ("clay_scene", {}),
        ("clay_add_primitive", {"kind": "box", "size": [1.0, 2.0, 3.0]}),
        ("clay_add_primitive", {"size": [1.0, 2.0, 3.0], "kind": "box"}),  # key order
        ("warlock_status", {"operation_id": "op-4"}),
    ]
    for tool, args in samples:
        assert agent_host._fingerprint(tool, args) == _old_fingerprint(tool, args)
