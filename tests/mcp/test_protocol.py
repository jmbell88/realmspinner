"""The wire format `dispatch` speaks, pinned with no notion of what a tool does.

The module's own docstring names the one distinction the whole file exists to
get right: a tool that runs and fails its job is a **successful** JSON-RPC
response whose `result.isError` is `True`, never a JSON-RPC `error`. Malformed
*requests* -- an unknown method, a non-object `params`, `tools/call` with no
name -- are the opposite: those really are JSON-RPC errors, because the
request itself was unusable before any tool ever ran. The tests below pin
both halves of that line, plus the framing (`encode`/`decode`) and content
helpers (`text`, `image_png`, `ok`, `fail`) `dispatch` is built from.
"""

from __future__ import annotations

import base64
import json

import pytest

from warlock.mcp import protocol as p

# --- initialize / notifications / ping ---------------------------------------


def test_initialize_returns_protocol_version_capabilities_and_server_info() -> None:
    reply = p.dispatch(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize"},
        tools=lambda: [],
        call=lambda name, args: p.ok(),
    )
    assert reply["result"]["protocolVersion"] == p.PROTOCOL_VERSION
    assert reply["result"]["capabilities"] == {"tools": {}}
    assert reply["result"]["serverInfo"] == {
        "name": p.SERVER_NAME,
        "version": p.SERVER_VERSION,
    }


def test_initialize_carries_the_instructions_it_was_given() -> None:
    """`dispatch` has no text of its own -- see the module docstring's "knows
    nothing about Clay" -- so a caller that passes `instructions` gets it back
    verbatim, under MCP's own top-level key for it."""
    reply = p.dispatch(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize"},
        tools=lambda: [],
        call=lambda name, args: p.ok(),
        instructions="Clay measures in metres.",
    )
    assert reply["result"]["instructions"] == "Clay measures in metres."


def test_initialize_omits_the_instructions_key_when_it_has_none() -> None:
    """No `instructions` argument, and an empty string, both mean "nothing to
    say" -- the key must not appear at all, which is what a server with
    nothing to say does per the MCP spec, rather than sending `""`."""
    reply = p.dispatch(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize"},
        tools=lambda: [],
        call=lambda name, args: p.ok(),
    )
    assert "instructions" not in reply["result"]

    reply = p.dispatch(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize"},
        tools=lambda: [],
        call=lambda name, args: p.ok(),
        instructions="",
    )
    assert "instructions" not in reply["result"]


def test_initialize_echoes_a_supported_older_protocol_version() -> None:
    """A client that only speaks an older revision this server can honestly
    serve (see `SUPPORTED_PROTOCOL_VERSIONS`) must be told that revision back,
    not our preferred one -- otherwise it has no correct way to know whether
    to proceed."""
    reply = p.dispatch(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05"},
        },
        tools=lambda: [],
        call=lambda name, args: p.ok(),
    )
    assert reply["result"]["protocolVersion"] == "2024-11-05"


def test_initialize_with_an_unknown_protocol_version_still_succeeds() -> None:
    """An unrecognised `protocolVersion` is not a JSON-RPC error -- the spec's
    answer to "I don't speak that" is a successful reply naming the version we
    do support, and leaving the client to decide whether to continue."""
    reply = p.dispatch(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "1999-01-01"},
        },
        tools=lambda: [],
        call=lambda name, args: p.ok(),
    )
    assert "error" not in reply
    assert reply["result"]["protocolVersion"] == p.PROTOCOL_VERSION


def test_initialize_with_no_protocol_version_falls_back_to_our_preferred_one() -> None:
    """No `protocolVersion` in `params` at all must not regress the existing
    behaviour: the reply still carries `PROTOCOL_VERSION`, not some error or a
    missing key."""
    reply = p.dispatch(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        tools=lambda: [],
        call=lambda name, args: p.ok(),
    )
    assert reply["result"]["protocolVersion"] == p.PROTOCOL_VERSION


def test_notifications_initialized_gets_no_reply() -> None:
    """A notification -- no `id` -- gets no reply, success or error, ever."""
    reply = p.dispatch(
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        tools=lambda: [],
        call=lambda name, args: p.ok(),
    )
    assert reply is None


def test_any_request_without_an_id_returns_none_even_for_an_unknown_method() -> None:
    """The id-less path is unconditional: it does not first check whether the
    method is one dispatch recognises. An unknown method with no `id` is still
    a notification as far as JSON-RPC is concerned, so it still gets nothing
    back -- not the -32601 a *request* for the same bad method would get."""
    reply = p.dispatch(
        {"jsonrpc": "2.0", "method": "not/a/real/method"},
        tools=lambda: [],
        call=lambda name, args: p.ok(),
    )
    assert reply is None


def test_ping_with_an_id_returns_an_empty_result() -> None:
    reply = p.dispatch(
        {"jsonrpc": "2.0", "id": "abc", "method": "ping"},
        tools=lambda: [],
        call=lambda n, a: p.ok(),
    )
    assert reply == {"jsonrpc": "2.0", "id": "abc", "result": {}}


# --- tools/list ----------------------------------------------------------------


def test_tools_list_renders_every_tool_into_the_mcp_json_shape() -> None:
    catalogue = [
        p.Tool(
            name="clay.extrude",
            title="Extrude",
            description="Extrude the selected faces.",
            schema={"type": "object", "properties": {"distance": {"type": "number"}}},
        ),
        p.Tool(name="clay.bevel", title="Bevel", description="Bevel the selected edge.", schema={}),
    ]
    reply = p.dispatch(
        {"jsonrpc": "2.0", "id": 7, "method": "tools/list"},
        tools=lambda: catalogue,
        call=lambda n, a: p.ok(),
    )
    assert reply["result"]["tools"] == [
        {
            "name": "clay.extrude",
            "title": "Extrude",
            "description": "Extrude the selected faces.",
            "inputSchema": {"type": "object", "properties": {"distance": {"type": "number"}}},
        },
        {
            "name": "clay.bevel",
            "title": "Bevel",
            "description": "Bevel the selected edge.",
            "inputSchema": {},
        },
    ]


def test_a_tool_that_declares_no_output_schema_does_not_emit_the_key() -> None:
    """Through a real `dispatch({"method": "tools/list", ...})`, not by
    calling `_tool_json` directly -- `output_schema=None`, the field's own
    default, spelled out explicitly rather than left off the call, so this
    exercises the same declared-but-empty path `Tool`'s default takes rather
    than a construction a `Tool` with no such field at all would answer
    identically anyway."""
    catalogue = [
        p.Tool(
            name="clay.bevel",
            title="Bevel",
            description="Bevel the selected edge.",
            schema={},
            output_schema=None,
        )
    ]
    reply = p.dispatch(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        tools=lambda: catalogue,
        call=lambda n, a: p.ok(),
    )
    assert "outputSchema" not in reply["result"]["tools"][0]


def test_a_declared_output_schema_reaches_the_tools_list() -> None:
    schema = {"type": "object", "properties": {"uid": {"type": "integer"}}}
    catalogue = [
        p.Tool(
            name="clay.scene",
            title="Scene",
            description="Describe the scene.",
            schema={},
            output_schema=schema,
        )
    ]
    reply = p.dispatch(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        tools=lambda: catalogue,
        call=lambda n, a: p.ok(),
    )
    assert reply["result"]["tools"][0]["outputSchema"] == schema


def test_a_tools_callback_that_raises_is_minus_32603_and_never_propagates() -> None:
    def exploding_tools():
        raise RuntimeError("catalogue is not built yet")

    reply = p.dispatch(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        tools=exploding_tools,
        call=lambda n, a: p.ok(),
    )
    assert reply["error"]["code"] == -32603
    assert "catalogue is not built yet" in reply["error"]["message"]


# --- the headline claim: a tool's own failure is not a JSON-RPC error --------


def test_a_tool_that_fails_is_a_successful_response_carrying_is_error() -> None:
    reply = p.dispatch(
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "clay.extrude"}},
        tools=lambda: [],
        call=lambda name, args: p.fail("that boolean has no valid manifold"),
    )
    assert "error" not in reply
    assert reply["result"]["isError"] is True
    assert reply["result"]["content"] == [
        {"type": "text", "text": "that boolean has no valid manifold"}
    ]


def test_a_call_that_raises_becomes_is_error_content_never_a_json_rpc_error() -> None:
    """`call`'s contract is "must not raise"; this is the backstop for the day
    it breaks anyway, and the backstop's job is to preserve the same shape a
    well-behaved `fail()` would have produced -- not to escalate into a
    transport-level error an agent's tool runner would throw away."""

    def exploding_call(name, args):
        raise KeyError("thickness")

    reply = p.dispatch(
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "clay.extrude"}},
        tools=lambda: [],
        call=exploding_call,
    )
    assert "error" not in reply
    assert reply["result"]["isError"] is True
    assert "KeyError" in reply["result"]["content"][0]["text"]
    assert "thickness" in reply["result"]["content"][0]["text"]


# --- malformed requests really are JSON-RPC errors ---------------------------


def test_unknown_method_with_an_id_is_minus_32601() -> None:
    reply = p.dispatch(
        {"jsonrpc": "2.0", "id": 1, "method": "not/a/real/method"},
        tools=lambda: [],
        call=lambda n, a: p.ok(),
    )
    assert reply["error"]["code"] == -32601


def test_non_object_params_is_minus_32602() -> None:
    reply = p.dispatch(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": [1, 2, 3]},
        tools=lambda: [],
        call=lambda n, a: p.ok(),
    )
    assert reply["error"]["code"] == -32602


def test_tools_call_with_a_missing_name_is_minus_32602() -> None:
    reply = p.dispatch(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {}},
        tools=lambda: [],
        call=lambda n, a: p.ok(),
    )
    assert reply["error"]["code"] == -32602


def test_tools_call_with_a_non_string_name_is_minus_32602() -> None:
    reply = p.dispatch(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": 42}},
        tools=lambda: [],
        call=lambda n, a: p.ok(),
    )
    assert reply["error"]["code"] == -32602


def test_tools_call_with_non_object_arguments_is_minus_32602() -> None:
    reply = p.dispatch(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "clay.extrude", "arguments": "not an object"},
        },
        tools=lambda: [],
        call=lambda n, a: p.ok(),
    )
    assert reply["error"]["code"] == -32602


def test_tools_call_with_missing_arguments_defaults_to_an_empty_dict() -> None:
    seen = {}

    def call(name, args):
        seen["args"] = args
        return p.ok()

    p.dispatch(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "clay.extrude"}},
        tools=lambda: [],
        call=call,
    )
    assert seen["args"] == {}


# --- encode / decode -----------------------------------------------------------


def test_encode_decode_round_trip() -> None:
    message = {"jsonrpc": "2.0", "id": 1, "method": "ping"}
    assert p.decode(p.encode(message)) == message


def test_encode_is_one_newline_terminated_utf8_line() -> None:
    frame = p.encode({"a": 1})
    assert frame.endswith(b"\n")
    assert frame.count(b"\n") == 1
    json.loads(frame.decode("utf-8"))  # does not raise


def test_decode_refuses_a_frame_over_max_frame() -> None:
    huge = b"{" + b" " * (p.MAX_FRAME + 1) + b"}"
    with pytest.raises(ValueError):
        p.decode(huge)


def test_an_oversized_frame_is_refused_without_the_message_being_parsed() -> None:
    """The size check runs before `json.loads` -- proven here by handing decode
    an oversized frame that is not even valid JSON. If parsing ran first, this
    would fail with a "malformed JSON" message instead of the frame-size one."""
    huge_and_invalid = b"not json at all, and also " + b"x" * p.MAX_FRAME
    with pytest.raises(ValueError, match="MAX_FRAME"):
        p.decode(huge_and_invalid)


def test_decode_refuses_non_object_json() -> None:
    with pytest.raises(ValueError):
        p.decode(b"[1, 2, 3]\n")
    with pytest.raises(ValueError):
        p.decode(b'"just a string"\n')


def test_decode_refuses_junk_bytes() -> None:
    with pytest.raises(ValueError):
        p.decode(b"{not json")
    with pytest.raises(ValueError):
        p.decode(b"\xff\xfe\x00\x01")  # not valid UTF-8


# --- content helpers -----------------------------------------------------------


def test_image_png_round_trips_through_base64_with_the_png_mime_type() -> None:
    original = b"\x89PNG\r\n\x1a\n" + bytes(range(256))
    block = p.image_png(original)
    assert block["type"] == "image"
    assert block["mimeType"] == "image/png"
    assert base64.b64decode(block["data"]) == original


def test_ok_carries_content_and_is_not_an_error() -> None:
    result = p.ok(p.text("done"))
    assert result["isError"] is False
    assert result["content"] == [{"type": "text", "text": "done"}]


def test_ok_carries_structured_content_only_when_it_is_given() -> None:
    """Both directions: a bare `ok(text(...))` has no `structuredContent` key
    at all -- not `None` -- and one given a dict carries it verbatim, the
    same "omitted, never null" convention `fail`'s own `extra` follows."""
    bare = p.ok(p.text("hi"))
    assert "structuredContent" not in bare

    given = p.ok(p.text("hi"), structured={"uid": 3})
    assert given["structuredContent"] == {"uid": 3}


def test_fail_carries_extra_as_structured_content() -> None:
    result = p.fail("no valid manifold", field="thickness")
    assert result["isError"] is True
    assert result["structuredContent"] == {"field": "thickness"}


def test_fail_with_no_extra_omits_structured_content() -> None:
    result = p.fail("no valid manifold")
    assert "structuredContent" not in result
