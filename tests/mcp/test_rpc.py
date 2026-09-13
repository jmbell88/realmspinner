"""`warlock.mcp.rpc` -- Warlock's own private RPC v1, pinned with no pipe at
all: everything here is `encode_request`/`decode_request`/`hello_header`/
`catalogue_payload` called directly, the same way `test_protocol.py` pins
`dispatch` without a real connection.
"""

from __future__ import annotations

from warlock.mcp import rpc

# --- hello / version negotiation --------------------------------------------------


def test_hello_header_refuses_when_no_version_overlaps() -> None:
    header = rpc.hello_header([2, 3], studio_version="1.0", catalogue_hash="abc", call_timeout=30.0)
    assert header == {"error": {"code": "rpc_version", "supported": [1]}}


def test_hello_header_refuses_a_non_list_versions_field() -> None:
    header = rpc.hello_header(
        "not-a-list", studio_version="1.0", catalogue_hash="abc", call_timeout=30.0
    )
    assert header["error"]["code"] == "rpc_version"


def test_hello_header_succeeds_when_a_version_overlaps() -> None:
    header = rpc.hello_header(
        [0, 1, 2], studio_version="1.2.3", catalogue_hash="abc", call_timeout=30.0
    )
    assert header == {
        "rpc": 1,
        "studio_version": "1.2.3",
        "catalogue_hash": "abc",
        "call_timeout": 30.0,
    }


# --- unknown op / bad request ------------------------------------------------------


def test_unknown_op_header() -> None:
    assert rpc.unknown_op_header() == {"error": {"code": "unknown_op"}}


def test_bad_request_header() -> None:
    assert rpc.bad_request_header() == {"error": {"code": "bad_request"}}


def test_decode_request_refuses_an_oversize_frame() -> None:
    huge = b'{"rpc":1,"op":"hello","junk":"' + b"x" * rpc.MAX_FRAME + b'"}'
    try:
        rpc.decode_request(huge)
    except ValueError as exc:
        assert "MAX_FRAME" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_decode_request_refuses_non_object_json() -> None:
    for bad in (b"[1,2,3]", b'"just a string"', b"not json"):
        try:
            rpc.decode_request(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad!r}")


def test_looks_like_rpc_true_only_for_a_dict_with_an_rpc_key() -> None:
    assert rpc.looks_like_rpc(b'{"rpc":1,"op":"hello"}') is True
    assert rpc.looks_like_rpc(b'{"jsonrpc":"2.0","id":1,"method":"ping"}') is False
    assert rpc.looks_like_rpc(b"not json") is False
    assert rpc.looks_like_rpc(b"[1,2,3]") is False


# --- encode/decode round trip -------------------------------------------------------


def test_encode_request_decode_request_round_trip() -> None:
    frame = rpc.encode_request("call", tool="warlock_status", args={"operation_id": "op-1"})
    message = rpc.decode_request(frame)
    assert message == {
        "rpc": 1,
        "op": "call",
        "tool": "warlock_status",
        "args": {"operation_id": "op-1"},
    }


def test_encode_reply_split_reply_round_trip_with_a_body() -> None:
    frame = rpc.encode_reply({"hash": "abc123"}, body=b'{"content":[],"isError":false}')
    header, body = rpc.split_reply(frame)
    assert header == {"hash": "abc123"}
    assert body == b'{"content":[],"isError":false}'


def test_encode_reply_split_reply_round_trip_with_no_body() -> None:
    frame = rpc.encode_reply({"rpc": 1})
    header, body = rpc.split_reply(frame)
    assert header == {"rpc": 1}
    assert body == b""


def test_split_reply_refuses_a_frame_with_no_separator() -> None:
    try:
        rpc.split_reply(b'{"rpc":1}')
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


# --- catalogue_hash stability -------------------------------------------------------


def test_canonical_hash_is_stable_across_key_order() -> None:
    a = rpc.canonical_hash({"tools": [{"name": "x"}], "instructions": "hi"})
    b = rpc.canonical_hash({"instructions": "hi", "tools": [{"name": "x"}]})
    assert a == b


def test_catalogue_payload_hash_is_stable_for_the_same_inputs() -> None:
    tools = [rpc.Tool(name="a", title="A", description="d", schema={})]
    first = rpc.catalogue_payload(tools, instructions="hi", server_name="s", server_version="1")
    second = rpc.catalogue_payload(tools, instructions="hi", server_name="s", server_version="1")
    assert first["hash"] == second["hash"]
    # server name/version do not enter the hash -- see catalogue_payload's own docstring.
    third = rpc.catalogue_payload(tools, instructions="hi", server_name="s", server_version="2")
    assert first["hash"] == third["hash"]


def test_catalogue_payload_hash_changes_when_tools_change() -> None:
    tools_a = [rpc.Tool(name="a", title="A", description="d", schema={})]
    tools_b = [rpc.Tool(name="b", title="B", description="d", schema={})]
    a = rpc.catalogue_payload(tools_a, instructions="hi", server_name="s", server_version="1")
    b = rpc.catalogue_payload(tools_b, instructions="hi", server_name="s", server_version="1")
    assert a["hash"] != b["hash"]


def test_catalogue_payload_key_order() -> None:
    payload = rpc.catalogue_payload([], instructions=None, server_name="s", server_version="1")
    assert list(payload.keys()) == [
        "hash",
        "tools",
        "instructions",
        "server",
        "resources",
        "prompts",
    ]


def test_catalogue_payload_resources_and_prompts_default_to_empty_lists() -> None:
    payload = rpc.catalogue_payload([], instructions=None, server_name="s", server_version="1")
    assert payload["resources"] == []
    assert payload["prompts"] == []


def test_catalogue_payload_hash_changes_when_resources_or_prompts_change() -> None:
    base = rpc.catalogue_payload([], instructions="hi", server_name="s", server_version="1")
    with_resources = rpc.catalogue_payload(
        [], instructions="hi", server_name="s", server_version="1",
        resources=[{"uri": "warlock://x"}],
    )
    with_prompts = rpc.catalogue_payload(
        [], instructions="hi", server_name="s", server_version="1",
        prompts=[{"name": "p"}],
    )
    assert base["hash"] != with_resources["hash"]
    assert base["hash"] != with_prompts["hash"]
