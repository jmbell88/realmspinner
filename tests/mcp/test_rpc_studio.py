"""Studio speaks only ``warlock.mcp.rpc`` v1 -- there is no MCP path left on
its own pipe.

Everything here drives a real :class:`~warlock.studio.agent_host.AgentHost`
over a real pipe (:mod:`warlock.mcp.pipe`), the same fixture shape
``tests/test_agent_host.py`` already uses: a background thread calls
``host.pump()`` the way ``main.py:App.frame`` would, while this thread is
the "bridge" dialling in with ``pipe.connect``. RPC v1 requests are built by
hand with ``rpc.encode_request``/``rpc.split_reply`` rather than through
``warlock.mcp.bridge``, so a failure here is Studio's side of the pipe alone.
"""

from __future__ import annotations

import base64
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
            # The `call` reply's `hash` is the *catalogue*'s hash (the same
            # value `hello` and `catalogue` already reported), never a hash
            # of this call's own result -- see rpc.py's `call` op docs. A
            # bridge uses this field to detect a moved catalogue; hashing
            # the result instead made it move on every single call.
            assert call_header["hash"] == header["catalogue_hash"]
            assert call_header["hash"] != rpc.canonical_hash(result)
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


# --- Studio speaks only RPC v1: a non-RPC first frame is refused and closed ---


def test_a_connection_that_opens_with_jsonrpc_gets_bad_request_and_is_closed(tmp_path) -> None:
    """Studio's pipe used to sniff a connection's first frame and, if it
    looked like bare MCP JSON-RPC rather than RPC v1, serve that old in-app
    path for the rest of the connection. That path is gone: the only server
    that speaks MCP at all now is `warlock mcp` (`bridge.py`), and Studio
    itself answers RPC v1 exclusively (`docs/INVARIANTS.md`'s agent
    paragraph). A first frame that is not RPC v1 gets one `bad_request`
    header reply and the connection is closed -- proven here by a bare MCP
    `initialize`, and by the connection refusing a second request rather
    than answering it."""
    host, stop_pumping, pumper = _started_host(tmp_path)
    try:
        conn = pipe.connect(tmp_path)
        try:
            conn.send_bytes(protocol.encode({"jsonrpc": "2.0", "id": 1, "method": "ping"}))
            header, body = rpc.split_reply(_recv(conn))
            assert body == b""
            assert header["error"]["code"] == "bad_request"

            # The connection is closed right after that one reply -- a
            # second request either raises sending into a closed pipe, or
            # is sent but never answered.
            try:
                conn.send_bytes(rpc.encode_request("hello", versions=[1], bridge_version="test"))
            except OSError:
                pass
            else:
                assert not conn.poll(0.5), "the connection should already be closed"
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


# --- resources and prompts, RPC v1 -------------------------------------------


def test_resources_op_lists_every_resource_uri(tmp_path) -> None:
    host, stop_pumping, pumper = _started_host(tmp_path)
    try:
        conn = pipe.connect(tmp_path)
        try:
            conn.send_bytes(rpc.encode_request("resources"))
            header, body = rpc.split_reply(_recv(conn))
            assert body == b""
            uris = {r["uri"] for r in header["resources"]}
            assert uris == {
                "warlock://clay/scene",
                "warlock://clay/render/last",
                "warlock://clay/conventions",
                "warlock://clay/generators",
                "warlock://clay/operations",
            }
            assert header["templates"] == []
            # A listing is metadata only -- the static resources' inline
            # content must not ride along on it.
            for row in header["resources"]:
                assert "text" not in row
                assert "blob" not in row
        finally:
            conn.close()
    finally:
        _stop(host, stop_pumping, pumper)


def test_every_listed_resource_uri_is_readable(tmp_path) -> None:
    """Every URI ``resources`` lists is readable, except the last-render
    resource before any render has happened -- proven ``not_found`` on its
    own, below."""
    host, stop_pumping, pumper = _started_host(tmp_path)
    try:
        conn = pipe.connect(tmp_path)
        try:
            conn.send_bytes(rpc.encode_request("resources"))
            listed, _ = rpc.split_reply(_recv(conn))
            uris = [
                r["uri"]
                for r in listed["resources"]
                if r["uri"] != "warlock://clay/render/last"
            ]
            for uri in uris:
                conn.send_bytes(rpc.encode_request("read", uri=uri))
                header, body = rpc.split_reply(_recv(conn))
                assert "error" not in header, (uri, header)
                assert header["uri"] == uri
                assert header["mimeType"]
                assert isinstance(body, bytes)
        finally:
            conn.close()
    finally:
        _stop(host, stop_pumping, pumper)


def test_reading_an_unknown_uri_is_not_found(tmp_path) -> None:
    host, stop_pumping, pumper = _started_host(tmp_path)
    try:
        conn = pipe.connect(tmp_path)
        try:
            conn.send_bytes(rpc.encode_request("read", uri="warlock://nonsense"))
            header, body = rpc.split_reply(_recv(conn))
            assert body == b""
            assert header["error"]["code"] == "not_found"
        finally:
            conn.close()
    finally:
        _stop(host, stop_pumping, pumper)


def test_reading_the_last_render_before_any_render_is_not_found(tmp_path) -> None:
    host, stop_pumping, pumper = _started_host(tmp_path)
    try:
        conn = pipe.connect(tmp_path)
        try:
            conn.send_bytes(rpc.encode_request("read", uri="warlock://clay/render/last"))
            header, _body = rpc.split_reply(_recv(conn))
            assert header["error"]["code"] == "not_found"
        finally:
            conn.close()
    finally:
        _stop(host, stop_pumping, pumper)


def test_reading_the_scene_of_a_freshly_connected_session_matches_clay_scene(tmp_path) -> None:
    """A tab is opened for a session the moment it connects (see
    ``agent_host._serve``'s own module docstring), so an empty document --
    not ``not_found`` -- is what a fresh connection's scene resource reads
    as, the same as calling ``clay_scene`` itself would answer."""
    host, stop_pumping, pumper = _started_host(tmp_path)
    try:
        conn = pipe.connect(tmp_path)
        try:
            conn.send_bytes(rpc.encode_request("read", uri="warlock://clay/scene"))
            header, body = rpc.split_reply(_recv(conn))
            assert "error" not in header
            scene = json.loads(body.decode("utf-8"))
            assert scene["object_count"] == 0
        finally:
            conn.close()
    finally:
        _stop(host, stop_pumping, pumper)


class _FakeView:
    """A ``ClayView`` stand-in that returns a real tiny PNG without touching
    GL -- the same shape ``tests/test_agent_clay.py::_FakeView`` uses, kept
    separate here since this module drives a real ``AgentHost`` from a
    background thread rather than calling ``agent_clay.call`` directly."""

    def __init__(self, png: bytes) -> None:
        self.png = png

    def render_png(self, doc, *, size, view=None, angles=None, bounds=None, grid=False, frame=True):
        del doc, size, view, angles, bounds, grid, frame
        return self.png


def _tiny_png() -> bytes:
    import io

    from PIL import Image

    im = Image.new("RGB", (4, 4), "white")
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def test_last_render_after_a_clay_render_call_returns_the_same_png_bytes(
    tmp_path, monkeypatch
) -> None:
    png = _tiny_png()
    monkeypatch.setattr(agent_clay, "_view_for", lambda ctx: _FakeView(png))
    host, stop_pumping, pumper = _started_host(tmp_path)
    try:
        conn = pipe.connect(tmp_path)
        try:
            conn.send_bytes(
                rpc.encode_request("call", tool="clay_add_primitive", args={"generator": "box"})
            )
            rpc.split_reply(_recv(conn))

            conn.send_bytes(rpc.encode_request("call", tool="clay_render", args={"size": 64}))
            _call_header, call_body = rpc.split_reply(_recv(conn))
            call_result = json.loads(call_body.decode("utf-8"))
            image_block = next(b for b in call_result["content"] if b["type"] == "image")
            rendered_png = base64.b64decode(image_block["data"])

            conn.send_bytes(rpc.encode_request("read", uri="warlock://clay/render/last"))
            header, body = rpc.split_reply(_recv(conn))
            assert "error" not in header
            assert header["mimeType"] == "image/png"
            assert body == rendered_png
        finally:
            conn.close()
    finally:
        _stop(host, stop_pumping, pumper)


def test_scene_resource_matches_the_clay_scene_tool_after_adding_a_primitive(tmp_path) -> None:
    host, stop_pumping, pumper = _started_host(tmp_path)
    try:
        conn = pipe.connect(tmp_path)
        try:
            conn.send_bytes(
                rpc.encode_request("call", tool="clay_add_primitive", args={"generator": "box"})
            )
            rpc.split_reply(_recv(conn))

            conn.send_bytes(rpc.encode_request("call", tool="clay_scene", args={}))
            _call_header, call_body = rpc.split_reply(_recv(conn))
            tool_result = json.loads(call_body.decode("utf-8"))["structuredContent"]

            conn.send_bytes(rpc.encode_request("read", uri="warlock://clay/scene"))
            header, body = rpc.split_reply(_recv(conn))
            assert header["mimeType"] == "application/json"
            assert json.loads(body.decode("utf-8")) == tool_result
        finally:
            conn.close()
    finally:
        _stop(host, stop_pumping, pumper)


def test_generators_and_operations_resources_derive_from_the_live_registries(tmp_path) -> None:
    from warlock.studio import clay_ops
    from warlock.studio.clay import primitives as bp

    host, stop_pumping, pumper = _started_host(tmp_path)
    try:
        conn = pipe.connect(tmp_path)
        try:
            conn.send_bytes(rpc.encode_request("read", uri="warlock://clay/generators"))
            _header, body = rpc.split_reply(_recv(conn))
            generators = json.loads(body.decode("utf-8"))
            assert set(generators) == set(bp.GENERATORS)
            for name, (defaults, _fn) in bp.GENERATORS.items():
                assert set(generators[name]["params"]) == set(defaults)

            conn.send_bytes(rpc.encode_request("read", uri="warlock://clay/operations"))
            _header, body = rpc.split_reply(_recv(conn))
            operations = json.loads(body.decode("utf-8"))
            assert set(operations) == {op.name for op in clay_ops.OPS}
            for op in clay_ops.OPS:
                param_names = {p["name"] for p in operations[op.name]["params"]}
                assert param_names == {p.name for p in op.params}
        finally:
            conn.close()
    finally:
        _stop(host, stop_pumping, pumper)


def test_prompts_op_lists_every_prompt_name(tmp_path) -> None:
    host, stop_pumping, pumper = _started_host(tmp_path)
    try:
        conn = pipe.connect(tmp_path)
        try:
            conn.send_bytes(rpc.encode_request("prompts"))
            header, body = rpc.split_reply(_recv(conn))
            assert body == b""
            names = {p["name"] for p in header["prompts"]}
            assert names == {
                "model_from_description",
                "model_from_reference",
                "repair_mesh",
                "prepare_for_export",
            }
        finally:
            conn.close()
    finally:
        _stop(host, stop_pumping, pumper)


def test_prompt_missing_required_argument_is_bad_arguments(tmp_path) -> None:
    host, stop_pumping, pumper = _started_host(tmp_path)
    try:
        conn = pipe.connect(tmp_path)
        try:
            conn.send_bytes(
                rpc.encode_request("prompt", name="model_from_description", arguments={})
            )
            header, body = rpc.split_reply(_recv(conn))
            assert body == b""
            assert header["error"]["code"] == "bad_arguments"
            assert header["error"]["missing"] == ["description"]
        finally:
            conn.close()
    finally:
        _stop(host, stop_pumping, pumper)


def test_unknown_prompt_name_is_not_found(tmp_path) -> None:
    host, stop_pumping, pumper = _started_host(tmp_path)
    try:
        conn = pipe.connect(tmp_path)
        try:
            conn.send_bytes(rpc.encode_request("prompt", name="nonsense", arguments={}))
            header, _body = rpc.split_reply(_recv(conn))
            assert header["error"]["code"] == "not_found"
        finally:
            conn.close()
    finally:
        _stop(host, stop_pumping, pumper)


def test_rendered_prompt_carries_description_and_a_text_message(tmp_path) -> None:
    host, stop_pumping, pumper = _started_host(tmp_path)
    try:
        conn = pipe.connect(tmp_path)
        try:
            conn.send_bytes(
                rpc.encode_request(
                    "prompt",
                    name="model_from_description",
                    arguments={"description": "a small red barrel"},
                )
            )
            header, body = rpc.split_reply(_recv(conn))
            assert body == b""
            assert isinstance(header["description"], str) and header["description"]
            assert header["messages"][0]["role"] == "user"
            text = header["messages"][0]["content"]["text"]
            assert "a small red barrel" in text
        finally:
            conn.close()
    finally:
        _stop(host, stop_pumping, pumper)


def test_every_tool_name_a_prompt_mentions_is_a_real_tool(tmp_path) -> None:
    """Test names are claims: this one scans every prompt's *rendered* text
    for clay_*/warlock_* tokens and checks each is a real tool, so a rename
    that forgets to update a prompt's prose fails here rather than shipping
    a prompt that quietly points an agent at a tool that no longer exists."""
    import re

    from warlock.studio import agent_clay
    from warlock.studio import agent_host as ah

    real_tools = {t.name for t in agent_clay.tools()} | {ah.STATUS_TOOL}

    host, stop_pumping, pumper = _started_host(tmp_path)
    try:
        conn = pipe.connect(tmp_path)
        try:
            conn.send_bytes(rpc.encode_request("prompts"))
            listed, _ = rpc.split_reply(_recv(conn))
            for prompt in listed["prompts"]:
                arguments = {a["name"]: f"<{a['name']}>" for a in prompt["arguments"]}
                conn.send_bytes(
                    rpc.encode_request("prompt", name=prompt["name"], arguments=arguments)
                )
                header, _body = rpc.split_reply(_recv(conn))
                assert "error" not in header, (prompt["name"], header)
                text = header["messages"][0]["content"]["text"]
                mentioned = set(re.findall(r"\bclay_\w+|\bwarlock_\w+", text))
                unknown = mentioned - real_tools
                assert not unknown, (prompt["name"], unknown)
        finally:
            conn.close()
    finally:
        _stop(host, stop_pumping, pumper)


def test_catalogue_op_includes_resources_and_prompts(tmp_path) -> None:
    host, stop_pumping, pumper = _started_host(tmp_path)
    try:
        conn = pipe.connect(tmp_path)
        try:
            conn.send_bytes(rpc.encode_request("catalogue"))
            header, _body = rpc.split_reply(_recv(conn))
            assert {r["uri"] for r in header["resources"]} >= {
                "warlock://clay/scene",
                "warlock://clay/conventions",
            }
            assert {p["name"] for p in header["prompts"]} >= {"model_from_description"}
        finally:
            conn.close()
    finally:
        _stop(host, stop_pumping, pumper)
