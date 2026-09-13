"""`protocol.py`'s own leaf pieces, plus `bridge_dispatch` -- the dual-era
dispatcher `bridge.py` calls.

The in-app `dispatch` this file used to pin was deleted the day Studio's
listener stopped speaking bare MCP JSON-RPC over the pipe (`studio/
agent_host.py` answers RPC v1 only now, per `docs/INVARIANTS.md`'s agent
paragraph) -- there is no longer a caller inside the app for it to be. What
remains here: the framing (`encode`/`decode`) and content helpers (`text`,
`image_png`, `ok`, `fail`) both wire formats share, and `bridge_dispatch`
itself, the real MCP server logic `warlock mcp` speaks to a third-party
client. The distinction `bridge_dispatch`'s own tests still pin: a tool that
runs and fails its job is a **successful** JSON-RPC response whose
`result.isError` is `True`, never a JSON-RPC `error`; a malformed request
(an unknown method, a non-object `params`, `tools/call` with no name) is the
opposite, a real JSON-RPC error, because the request itself was unusable
before any tool ever ran.
"""

from __future__ import annotations

import base64
import json

import pytest

from warlock.mcp import protocol as p

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


# =============================================================================
# bridge_dispatch -- the dual-era dispatcher `bridge.py` calls, not the app.
# =============================================================================

def _catalogue(**overrides):
    base = {
        "hash": "h1",
        "tools": [
            {
                "name": "clay_scene",
                "title": "Scene",
                "description": "Describe the scene.",
                "inputSchema": {},
            }
        ],
        "instructions": "Clay measures in metres.",
        "server": {"name": "warlock-studio", "version": "1.2.3"},
    }
    base.update(overrides)
    return base


def _ok_call_tool(name, args):
    return json.dumps(p.ok(p.text(f"ran {name}"))).encode("utf-8")


def _dispatch(
    payload, state, catalogue=None, call_tool=_ok_call_tool, read_resource=None, get_prompt=None
):
    raw = json.dumps(payload).encode("utf-8")
    reply = p.bridge_dispatch(
        raw,
        state,
        catalogue=catalogue or _catalogue(),
        call_tool=call_tool,
        read_resource=read_resource,
        get_prompt=get_prompt,
    )
    if reply is None:
        return None
    assert reply.endswith(b"\n")
    return json.loads(reply)


# --- legacy era ----------------------------------------------------------------


def test_legacy_initialize_for_each_supported_version() -> None:
    for version in p.LEGACY:
        state = p.BridgeEra()
        reply = _dispatch(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": version},
            },
            state,
        )
        assert reply["result"]["protocolVersion"] == version
        assert state.era == "legacy"
        assert state.legacy_version == version


def test_legacy_initialize_with_an_unknown_version_falls_back_to_the_newest_legacy() -> None:
    state = p.BridgeEra()
    reply = _dispatch(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "1999-01-01"},
        },
        state,
    )
    assert reply["result"]["protocolVersion"] == p.LEGACY[0]


def test_legacy_notification_gets_no_reply() -> None:
    state = p.BridgeEra()
    _dispatch({"jsonrpc": "2.0", "id": 1, "method": "initialize"}, state)
    reply = _dispatch({"jsonrpc": "2.0", "method": "notifications/initialized"}, state)
    assert reply is None


def test_legacy_ping_ok() -> None:
    state = p.BridgeEra()
    _dispatch({"jsonrpc": "2.0", "id": 1, "method": "initialize"}, state)
    reply = _dispatch({"jsonrpc": "2.0", "id": 2, "method": "ping"}, state)
    assert reply == {"jsonrpc": "2.0", "id": 2, "result": {}}


def test_legacy_tools_list_uses_the_catalogues_own_json() -> None:
    state = p.BridgeEra()
    _dispatch({"jsonrpc": "2.0", "id": 1, "method": "initialize"}, state)
    reply = _dispatch({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, state)
    assert reply["result"]["tools"] == _catalogue()["tools"]
    assert "ttlMs" not in reply["result"]
    assert "cacheScope" not in reply["result"]


def test_legacy_2025_03_26_batch_is_accepted() -> None:
    state = p.BridgeEra()
    _dispatch(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-03-26"},
        },
        state,
    )
    raw = json.dumps(
        [
            {"jsonrpc": "2.0", "id": 2, "method": "ping"},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 3, "method": "ping"},
        ]
    ).encode("utf-8")
    reply_bytes = p.bridge_dispatch(raw, state, catalogue=_catalogue(), call_tool=_ok_call_tool)
    replies = json.loads(reply_bytes)
    assert [r["id"] for r in replies] == [2, 3]


def test_legacy_2025_03_26_all_notification_batch_gets_no_reply() -> None:
    state = p.BridgeEra()
    _dispatch(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-03-26"},
        },
        state,
    )
    raw = json.dumps([{"jsonrpc": "2.0", "method": "notifications/initialized"}]).encode("utf-8")
    reply_bytes = p.bridge_dispatch(raw, state, catalogue=_catalogue(), call_tool=_ok_call_tool)
    assert reply_bytes is None


def test_legacy_2025_06_18_batch_is_refused() -> None:
    """A bug this tranche fixes: today a `2025-03-26` batch is refused
    `-32700` even though that revision still had batching; this pins the
    fix (accepted above) and the real refusal case (a *newer* revision that
    dropped batching) at once."""
    state = p.BridgeEra()
    _dispatch(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18"},
        },
        state,
    )
    raw = json.dumps([{"jsonrpc": "2.0", "id": 2, "method": "ping"}]).encode("utf-8")
    reply_bytes = p.bridge_dispatch(raw, state, catalogue=_catalogue(), call_tool=_ok_call_tool)
    reply = json.loads(reply_bytes)
    assert reply["error"]["code"] == -32600


def test_a_batch_before_any_era_is_decided_is_refused() -> None:
    state = p.BridgeEra()
    raw = json.dumps([{"jsonrpc": "2.0", "id": 1, "method": "ping"}]).encode("utf-8")
    reply_bytes = p.bridge_dispatch(raw, state, catalogue=_catalogue(), call_tool=_ok_call_tool)
    reply = json.loads(reply_bytes)
    assert reply["error"]["code"] == -32600


# --- modern era ------------------------------------------------------------------


def _modern_meta(version=None):
    return {"_meta": {p.MODERN_META_KEY: version or p.MODERN[0]}}


def test_server_discover_shape() -> None:
    state = p.BridgeEra()
    reply = _dispatch({"jsonrpc": "2.0", "id": 1, "method": "server/discover"}, state)
    result = reply["result"]
    assert set(p.LEGACY) | set(p.MODERN) <= set(result["supportedVersions"])
    assert result["capabilities"] == {"tools": {}, "resources": {}, "prompts": {}}
    assert result["ttlMs"] == 60000
    assert result["cacheScope"] == "public"
    assert result["resultType"] == "complete"
    assert result["_meta"]["serverInfo"] == {"name": "warlock-studio", "version": "1.2.3"}
    assert result["instructions"] == "Clay measures in metres."
    assert state.era == "modern"


def test_modern_era_is_entered_by_a_meta_version_with_no_discover_call() -> None:
    state = p.BridgeEra()
    reply = _dispatch(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": _modern_meta()}, state
    )
    assert reply["result"]["resultType"] == "complete"
    assert state.era == "modern"


def test_modern_request_before_discover_or_meta_is_refused() -> None:
    state = p.BridgeEra()
    reply = _dispatch({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, state)
    assert reply["error"]["code"] == -32600


def test_modern_unsupported_version_is_minus_32022_with_data() -> None:
    state = p.BridgeEra()
    _dispatch({"jsonrpc": "2.0", "id": 1, "method": "server/discover"}, state)
    reply = _dispatch(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": _modern_meta("1999-01-01"),
        },
        state,
    )
    assert reply["error"]["code"] == -32022
    assert reply["error"]["data"] == {"supported": list(p.MODERN), "requested": "1999-01-01"}


def test_modern_ping_and_logging_set_level_are_minus_32601() -> None:
    state = p.BridgeEra()
    _dispatch({"jsonrpc": "2.0", "id": 1, "method": "server/discover"}, state)
    for method in ("ping", "logging/setLevel"):
        reply = _dispatch(
            {"jsonrpc": "2.0", "id": 2, "method": method, "params": _modern_meta()}, state
        )
        assert reply["error"]["code"] == -32601, method


def test_modern_tools_list_carries_cache_hints_and_result_type() -> None:
    state = p.BridgeEra()
    _dispatch({"jsonrpc": "2.0", "id": 1, "method": "server/discover"}, state)
    reply = _dispatch(
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": _modern_meta()}, state
    )
    result = reply["result"]
    assert result["tools"] == _catalogue()["tools"]
    assert result["ttlMs"] == 60000
    assert result["cacheScope"] == "public"
    assert result["resultType"] == "complete"
    assert result["_meta"]["serverInfo"] == {"name": "warlock-studio", "version": "1.2.3"}


# --- splicing tools/call results: never json.loads the body --------------------


def test_legacy_tools_call_splices_the_raw_body_verbatim() -> None:
    state = p.BridgeEra()
    _dispatch({"jsonrpc": "2.0", "id": 1, "method": "initialize"}, state)
    body = json.dumps(
        {"content": [{"type": "text", "text": 'a "quoted" } brace'}], "isError": False}
    ).encode("utf-8")
    reply = _dispatch(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "t", "arguments": {}},
        },
        state,
        call_tool=lambda n, a: body,
    )
    expected = {
        "jsonrpc": "2.0",
        "id": 2,
        "result": {"content": [{"type": "text", "text": 'a "quoted" } brace'}], "isError": False},
    }
    assert reply == expected


def test_legacy_tools_call_with_an_empty_object_body() -> None:
    state = p.BridgeEra()
    _dispatch({"jsonrpc": "2.0", "id": 1, "method": "initialize"}, state)
    reply = _dispatch(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "t", "arguments": {}},
        },
        state,
        call_tool=lambda n, a: b"{}",
    )
    assert reply == {"jsonrpc": "2.0", "id": 2, "result": {}}


def test_modern_tools_call_splices_meta_in_front_of_the_body() -> None:
    state = p.BridgeEra()
    _dispatch({"jsonrpc": "2.0", "id": 1, "method": "server/discover"}, state)
    body = json.dumps(
        {"content": [], "isError": False, "structuredContent": {"n": 1}}
    ).encode("utf-8")
    reply = _dispatch(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "t", "arguments": {}, **_modern_meta()},
        },
        state,
        call_tool=lambda n, a: body,
    )
    result = reply["result"]
    assert result["resultType"] == "complete"
    assert result["_meta"]["serverInfo"] == {"name": "warlock-studio", "version": "1.2.3"}
    assert result["content"] == []
    assert result["isError"] is False
    assert result["structuredContent"] == {"n": 1}


def test_modern_tools_call_with_an_empty_object_body_has_no_dangling_comma() -> None:
    state = p.BridgeEra()
    _dispatch({"jsonrpc": "2.0", "id": 1, "method": "server/discover"}, state)
    reply_bytes_holder = {}

    def call_tool(n, a):
        return b"{}"

    raw = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "t", "arguments": {}, **_modern_meta()},
        }
    ).encode("utf-8")
    reply_bytes = p.bridge_dispatch(raw, state, catalogue=_catalogue(), call_tool=call_tool)
    reply_bytes_holder["r"] = reply_bytes
    # Must be valid JSON with no dangling comma.
    reply = json.loads(reply_bytes)
    assert reply["result"]["resultType"] == "complete"
    assert "content" not in reply["result"]


def test_merge_body_matches_a_fully_built_reference_dict() -> None:
    body = json.dumps(
        {
            "content": [{"type": "text", "text": "hi"}],
            "isError": False,
            "structuredContent": {"a": {"b": 1}},
        }
    ).encode("utf-8")
    meta = {"resultType": "complete", "_meta": {"serverInfo": {"name": "s", "version": "1"}}}
    spliced = p.splice_tool_result(7, body, meta=meta)
    reference = {
        "jsonrpc": "2.0",
        "id": 7,
        "result": {
            "resultType": "complete",
            "_meta": {"serverInfo": {"name": "s", "version": "1"}},
            "content": [{"type": "text", "text": "hi"}],
            "isError": False,
            "structuredContent": {"a": {"b": 1}},
        },
    }
    assert json.loads(spliced) == reference


def test_a_call_tool_that_raises_becomes_is_error_never_a_transport_error() -> None:
    state = p.BridgeEra()
    _dispatch({"jsonrpc": "2.0", "id": 1, "method": "initialize"}, state)

    def exploding(name, args):
        raise KeyError("thickness")

    reply = _dispatch(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "t", "arguments": {}},
        },
        state,
        call_tool=exploding,
    )
    assert "error" not in reply
    assert reply["result"]["isError"] is True
    assert "thickness" in reply["result"]["content"][0]["text"]


def test_unknown_tool_call_args_still_get_the_usual_minus_32602s() -> None:
    state = p.BridgeEra()
    _dispatch({"jsonrpc": "2.0", "id": 1, "method": "initialize"}, state)
    reply = _dispatch(
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {}}, state
    )
    assert reply["error"]["code"] == -32602


# =============================================================================
# resources/prompts -- both eras.
# =============================================================================


def _catalogue_with_resources_and_prompts():
    return _catalogue(
        resources=[
            {
                "uri": "warlock://clay/conventions",
                "name": "clay-conventions",
                "mimeType": "text/markdown",
                "text": "units are metres",
            },
            {"uri": "warlock://clay/scene", "name": "clay-scene", "mimeType": "application/json"},
        ],
        prompts=[
            {
                "name": "model_from_description",
                "title": "Model from a description",
                "description": "d",
                "arguments": [{"name": "description", "description": "d", "required": True}],
            }
        ],
    )


def _ok_read_resource(uri):
    if uri == "warlock://clay/scene":
        return {"contents": [{"uri": uri, "mimeType": "application/json", "text": "{}"}]}
    return None


def _ok_get_prompt(name, arguments):
    if name != "model_from_description":
        return None
    if "description" not in arguments:
        return ["description"]
    return {
        "description": "d",
        "messages": [{"role": "user", "content": {"type": "text", "text": "hi"}}],
    }


def test_legacy_resources_list_strips_inline_content() -> None:
    state = p.BridgeEra()
    _dispatch({"jsonrpc": "2.0", "id": 1, "method": "initialize"}, state)
    reply = _dispatch(
        {"jsonrpc": "2.0", "id": 2, "method": "resources/list"},
        state,
        catalogue=_catalogue_with_resources_and_prompts(),
    )
    resources = reply["result"]["resources"]
    assert {r["uri"] for r in resources} == {"warlock://clay/conventions", "warlock://clay/scene"}
    for r in resources:
        assert "text" not in r
    assert "ttlMs" not in reply["result"]


def test_legacy_resources_templates_list_is_empty() -> None:
    state = p.BridgeEra()
    _dispatch({"jsonrpc": "2.0", "id": 1, "method": "initialize"}, state)
    reply = _dispatch({"jsonrpc": "2.0", "id": 2, "method": "resources/templates/list"}, state)
    assert reply["result"]["templates"] == []


def test_legacy_resources_read_ok() -> None:
    state = p.BridgeEra()
    _dispatch({"jsonrpc": "2.0", "id": 1, "method": "initialize"}, state)
    reply = _dispatch(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "resources/read",
            "params": {"uri": "warlock://clay/scene"},
        },
        state,
        read_resource=_ok_read_resource,
    )
    assert reply["result"]["contents"][0]["uri"] == "warlock://clay/scene"
    assert "ttlMs" not in reply["result"]


def test_legacy_resources_read_not_found_is_minus_32002_with_uri() -> None:
    state = p.BridgeEra()
    _dispatch({"jsonrpc": "2.0", "id": 1, "method": "initialize"}, state)
    reply = _dispatch(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "resources/read",
            "params": {"uri": "warlock://nonsense"},
        },
        state,
        read_resource=_ok_read_resource,
    )
    assert reply["error"]["code"] == -32002
    assert reply["error"]["data"] == {"uri": "warlock://nonsense"}


def test_legacy_prompts_list() -> None:
    state = p.BridgeEra()
    _dispatch({"jsonrpc": "2.0", "id": 1, "method": "initialize"}, state)
    reply = _dispatch(
        {"jsonrpc": "2.0", "id": 2, "method": "prompts/list"},
        state,
        catalogue=_catalogue_with_resources_and_prompts(),
    )
    assert {pr["name"] for pr in reply["result"]["prompts"]} == {"model_from_description"}


def test_legacy_prompts_get_missing_required_argument_is_minus_32602() -> None:
    state = p.BridgeEra()
    _dispatch({"jsonrpc": "2.0", "id": 1, "method": "initialize"}, state)
    reply = _dispatch(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "prompts/get",
            "params": {"name": "model_from_description", "arguments": {}},
        },
        state,
        get_prompt=_ok_get_prompt,
    )
    assert reply["error"]["code"] == -32602
    assert reply["error"]["data"] == {"missing": ["description"]}


def test_legacy_prompts_get_unknown_name_is_minus_32002() -> None:
    state = p.BridgeEra()
    _dispatch({"jsonrpc": "2.0", "id": 1, "method": "initialize"}, state)
    reply = _dispatch(
        {"jsonrpc": "2.0", "id": 2, "method": "prompts/get", "params": {"name": "nope"}},
        state,
        get_prompt=_ok_get_prompt,
    )
    assert reply["error"]["code"] == -32002


def test_legacy_prompts_get_ok() -> None:
    state = p.BridgeEra()
    _dispatch({"jsonrpc": "2.0", "id": 1, "method": "initialize"}, state)
    reply = _dispatch(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "prompts/get",
            "params": {
                "name": "model_from_description",
                "arguments": {"description": "a barrel"},
            },
        },
        state,
        get_prompt=_ok_get_prompt,
    )
    assert reply["result"]["description"] == "d"
    assert reply["result"]["messages"][0]["content"]["text"] == "hi"


def test_modern_resources_list_carries_cache_hints() -> None:
    state = p.BridgeEra()
    _dispatch({"jsonrpc": "2.0", "id": 1, "method": "server/discover"}, state)
    reply = _dispatch(
        {"jsonrpc": "2.0", "id": 2, "method": "resources/list", "params": _modern_meta()},
        state,
        catalogue=_catalogue_with_resources_and_prompts(),
    )
    result = reply["result"]
    assert result["ttlMs"] == 60000
    assert result["cacheScope"] == "public"
    assert result["resultType"] == "complete"


def test_modern_resources_read_not_found_is_minus_32602() -> None:
    state = p.BridgeEra()
    _dispatch({"jsonrpc": "2.0", "id": 1, "method": "server/discover"}, state)
    reply = _dispatch(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "resources/read",
            "params": {"uri": "warlock://nonsense", **_modern_meta()},
        },
        state,
        read_resource=_ok_read_resource,
    )
    assert reply["error"]["code"] == -32602
    assert reply["error"]["data"] == {"uri": "warlock://nonsense"}


def test_modern_prompts_get_missing_required_argument_is_minus_32602() -> None:
    state = p.BridgeEra()
    _dispatch({"jsonrpc": "2.0", "id": 1, "method": "server/discover"}, state)
    reply = _dispatch(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "prompts/get",
            "params": {"name": "model_from_description", "arguments": {}, **_modern_meta()},
        },
        state,
        get_prompt=_ok_get_prompt,
    )
    assert reply["error"]["code"] == -32602


def test_modern_prompts_get_ok_carries_meta() -> None:
    state = p.BridgeEra()
    _dispatch({"jsonrpc": "2.0", "id": 1, "method": "server/discover"}, state)
    reply = _dispatch(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "prompts/get",
            "params": {
                "name": "model_from_description",
                "arguments": {"description": "a barrel"},
                **_modern_meta(),
            },
        },
        state,
        get_prompt=_ok_get_prompt,
    )
    assert reply["result"]["_meta"]["serverInfo"] == {"name": "warlock-studio", "version": "1.2.3"}


def test_no_read_resource_or_get_prompt_configured_is_unknown_method() -> None:
    """A caller that never wired resources/prompts up at all (neither
    callback given) gets the ordinary unknown-method refusal, not a crash --
    this is what every pre-existing `_dispatch(...)` call in this file above
    was already relying on."""
    state = p.BridgeEra()
    _dispatch({"jsonrpc": "2.0", "id": 1, "method": "initialize"}, state)
    reply = _dispatch(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "resources/read",
            "params": {"uri": "warlock://clay/scene"},
        },
        state,
    )
    assert reply["error"]["code"] == -32601
