"""MCP over JSON-RPC 2.0 -- the wire format, with no idea what a tool does.

Two dispatchers live here now. `dispatch` (below) is the original, in-app
path `studio/agent_host.py` still calls directly -- one revision family, one
JSON-RPC shape, no era negotiation, because Studio only ever needs to answer
what it has always answered. `bridge_dispatch` (at the bottom of this file)
is what `warlock mcp` (`bridge.py`) calls instead: it is the module's
dual-era MCP *server* logic -- legacy `initialize`-first negotiation and a
newer, no-`initialize` "modern" era, JSON-RPC batching, and splicing a tool
result's raw bytes (fetched from Studio over `rpc.py`'s private RPC v1)
into whichever envelope a connection's negotiated era calls for, all
without ever `json.loads`-ing that result. The two share `Tool`/`ok`/
`fail`/`text`/`image_png`/`encode`/`decode` because both wire formats need
the same vocabulary, not because they are the same dispatcher.

`dispatch` takes a decoded JSON-RPC message plus two callbacks -- `tools`
(the catalogue) and `call` (run one) -- and returns the reply, or `None` for
a notification. Nothing here knows Clay exists; `studio/agent_clay.py` and
`studio/agent_host.py` own that, which is what makes this module testable
without a GL context and reusable the day a second mode grows tools.

**The one distinction that matters more than any other in this file: a tool
failing its job is not a JSON-RPC error.** `tools/call` on an *unknown* tool,
or with malformed params, is a JSON-RPC error (`-32602`) -- the request
itself was bad. But a *known* tool that ran and hit a real problem --
"that boolean has no valid manifold", "the tab was closed", "the file does
not exist" -- is a **successful** JSON-RPC response whose result carries
`isError: true` and the reason as readable text content (`fail()` below).
Get this backwards and an agent's tool runner throws the failure away as a
transport error instead of handing the model the sentence it needs to
recover; this is reportedly the single most common MCP protocol bug, and the
whole reason `call()`'s contract is "must not raise, and if it does anyway
this module still turns it into `isError` rather than propagating it."
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

from .rpc import MAX_FRAME, Tool, fail, image_png, ok, text  # noqa: F401 -- re-exported

PROTOCOL_VERSION = "2025-06-18"
SERVER_NAME = "warlock-studio"

SUPPORTED_PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
"""Every revision `dispatch`'s `initialize` may honestly echo back, newest first.

The MCP spec's own rule: a server SHOULD reply with the client's requested
`protocolVersion` if it supports that version, and otherwise with the version
it does support -- the client then decides whether to continue or disconnect.
`PROTOCOL_VERSION` above stays the first element and the one used when nothing
else applies; it must keep meaning "the one we prefer" because `test_agent_host.
py` and `test_agent_perf.py` already depend on that name.

The two older revisions earn their place here, not a free pass: this module's
JSON-RPC message shapes for `initialize`/`notifications/initialized`/
`tools/list`/`tools/call`/`ping` are otherwise unchanged across all three --
the only 2025-06-18-specific wire additions are `structuredContent`/
`outputSchema` (`ok`, `_tool_json`) and a tool's `title`. All three are
additive keys on objects an older client already parses, so it ignores them;
none is a shape it has to recognise in order to proceed.
The other deltas between these revisions -- JSON-RPC batching (added
2025-03-26, removed 2025-06-18), the `MCP-Protocol-Version` HTTP header,
OAuth/resource-server changes -- are transport- and auth-layer, and this
server was never on the spec's stdio or Streamable HTTP transport to begin
with: `mcp/pipe.py` is a bespoke authenticated named pipe/socket, so those
deltas never applied here under any version this module has ever claimed,
2025-06-18 included. That is what makes 2024-11-05 and 2025-03-26 honest
claims rather than merely convenient ones."""

SERVER_VERSION = "0.0.0"
"""Overwritten by whoever knows the real version, before the first `initialize`.

This module cannot import `warlock.__version__` -- that would break the "pure
stdlib, no warlock imports" rule that lets it be unit-tested and imported on a
headless box -- and `dispatch`'s signature is fixed by the interface contract
this package was built against, so it cannot take a `version` keyword either.
A module constant is the remaining option: `studio/agent_host.py` sets this
once, at startup, from the real `warlock.__version__`, the same way logging
configuration is a module-level knob rather than a parameter threaded through
every call.
"""

# Tool, ok, fail, text, image_png and MAX_FRAME now live in `rpc.py` -- the
# private RPC v1 wire format shares them with this MCP one -- and are
# imported above so every existing `protocol.Tool` / `protocol.ok` / etc.
# call site keeps working unchanged.


def encode(message: dict[str, Any]) -> bytes:
    """One JSON-RPC message, newline-delimited, UTF-8."""
    return (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")


def decode(line: bytes) -> dict[str, Any]:
    """The inverse of `encode`. Raises `ValueError` on anything unusable.

    Two refusals live here rather than in the caller: an oversize frame (see
    `MAX_FRAME`) and a frame that parses but is not a JSON *object* -- a bare
    `42` or `"hi"` is valid JSON and invalid JSON-RPC, and letting it through
    would hand `dispatch` a `message.get` call on a list or a string.
    """
    if len(line) > MAX_FRAME:
        raise ValueError(f"frame of {len(line)} bytes exceeds MAX_FRAME ({MAX_FRAME})")
    try:
        message = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed JSON: {exc}") from exc
    if not isinstance(message, dict):
        raise ValueError(f"expected a JSON object, got {type(message).__name__}")
    return message


def _tool_json(tool: Tool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": tool.name,
        "title": tool.title,
        "description": tool.description,
        "inputSchema": tool.schema,
    }
    if tool.output_schema is not None:
        result["outputSchema"] = tool.output_schema
    return result


def _result(msg_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _error(msg_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def dispatch(
    message: dict[str, Any],
    *,
    tools: Callable[[], Sequence[Tool]],
    call: Callable[[str, dict[str, Any]], dict[str, Any]],
    instructions: str | None = None,
) -> dict[str, Any] | None:
    """One JSON-RPC request in, one response out. `None` for a notification.

    Handles `initialize`, `notifications/initialized`, `tools/list`,
    `tools/call` and `ping`; anything else is `-32601`. See the module
    docstring for why a tool's own failure never reaches this function as an
    exception worth turning into a JSON-RPC error -- `call`'s contract already
    promises `isError` content instead, and the `try` around it below is a
    backstop for that promise being broken, not the advertised path.

    `instructions` is MCP's optional top-level `initialize` field -- prose a
    client may show its model before the first tool call. This module has no
    text to put there: it knows nothing about Clay, so the caller supplies
    the sentence, exactly the way `tools` and `call` are already supplied
    rather than imported. Left `None` (the default), the key is omitted
    entirely rather than sent as `""` -- an empty string is a thing a server
    said, and a server with nothing to say omits the field, per the spec.
    """
    method = message.get("method")
    has_id = "id" in message
    msg_id = message.get("id")

    if not isinstance(method, str) or not method:
        # A request this malformed cannot have been a well-formed notification
        # either -- both require a real method name -- so it always gets an
        # answer, with whatever id (possibly none) we could find. That is the
        # one place this function replies to something lacking an "id".
        return _error(msg_id, -32600, "invalid request: 'method' must be a non-empty string")

    params = message.get("params", {})
    if params is None:
        params = {}
    if not isinstance(params, dict):
        return _error(msg_id, -32602, "'params' must be an object") if has_id else None

    if not has_id:
        # JSON-RPC notifications never get a reply, success or error -- the
        # sender already told us it is not listening for one by omitting
        # "id". `notifications/initialized` is the only one MCP defines today;
        # anything else unrecognised is silently ignored, per spec, rather
        # than reported.
        return None

    try:
        if method == "initialize":
            # Echo the client's requested version back if it's one we can
            # honestly serve (see SUPPORTED_PROTOCOL_VERSIONS); otherwise fall
            # back to the one we prefer. An unrecognised or absent version is
            # not a JSON-RPC error -- the spec's answer to "I don't speak
            # that" is a successful reply naming what we *do* speak, leaving
            # the client to decide whether to continue or disconnect.
            requested = params.get("protocolVersion")
            version = requested if requested in SUPPORTED_PROTOCOL_VERSIONS else PROTOCOL_VERSION
            result: dict[str, Any] = {
                "protocolVersion": version,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            }
            if instructions:
                result["instructions"] = instructions
            return _result(msg_id, result)
        if method == "ping":
            return _result(msg_id, {})
        if method == "tools/list":
            return _result(msg_id, {"tools": [_tool_json(t) for t in tools()]})
        if method == "tools/call":
            name = params.get("name")
            if not isinstance(name, str) or not name:
                return _error(msg_id, -32602, "tools/call needs a string 'name'")
            arguments = params.get("arguments", {})
            if arguments is None:
                arguments = {}
            if not isinstance(arguments, dict):
                return _error(msg_id, -32602, "'arguments' must be an object")
            try:
                result = call(name, arguments)
            except Exception as exc:  # noqa: BLE001 -- call() promises not to raise; this
                # is what happens the day that promise is broken anyway, so an
                # agent still gets isError content instead of a dropped
                # connection.
                result = fail(f"{type(exc).__name__}: {exc}")
            return _result(msg_id, result)
        return _error(msg_id, -32601, f"unknown method: {method}")
    except Exception as exc:  # noqa: BLE001 -- e.g. tools() raising; dispatch itself
        # must never propagate, since it runs on the listener thread in
        # studio/agent_host.py and an uncaught exception there would take the
        # whole agent session down instead of reporting one bad call.
        return _error(msg_id, -32603, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Bridge-side, dual-era dispatch.
#
# Everything above this point still runs *inside the app*
# (``studio/agent_host.py``) and answers exactly one revision family -- the
# legacy MCP revisions ``SUPPORTED_PROTOCOL_VERSIONS`` names, all sharing one
# JSON-RPC shape. ``bridge.py`` (``warlock mcp``) is a different peer
# entirely: it is the real MCP server a third-party client's tool runner
# dials, so it has to speak whatever revision that client actually
# negotiates -- both the legacy family and a newer "modern" era that drops
# `initialize` for `server/discover` and carries its version per-request.
# Studio itself never has to know about any of this: it answers a private
# RPC (`rpc.py`), not MCP, and the bridge is the only thing that speaks MCP
# to the outside world. The functions below build that: era negotiation,
# per-request modern versioning, and splicing a tool result's raw JSON bytes
# (fetched from Studio over RPC v1, never re-parsed here) into whichever
# envelope this connection's era calls for.
# ---------------------------------------------------------------------------

LEGACY = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
"""Every MCP revision the bridge may honestly negotiate under classic,
`initialize`-first JSON-RPC semantics, newest first. Distinct from
``SUPPORTED_PROTOCOL_VERSIONS`` above (the in-app path's own, narrower list)
because the bridge is a real MCP server facing real clients and has to keep
up with what they actually request; the in-app dispatcher answers a fixed,
already-shipped set instead."""

MODERN = ("2026-07-28",)
"""Every revision the bridge may serve under the newer, no-`initialize`
era: `server/discover` replaces it, and every other request carries its
version in `params._meta` instead of negotiating one up front."""

MODERN_META_KEY = "io.modelcontextprotocol/protocolVersion"
"""The `_meta` key a modern-era request carries its protocol version under.
`"protocolVersion"` (no namespace) is also accepted, for a client that has
not adopted the namespaced key yet."""


def _meta_version(params: Any) -> Any:
    """The modern-era protocol version named in *params*'s `_meta`, or
    `None` if there is no `_meta`, it is not an object, or it names neither
    key this module recognises."""
    if not isinstance(params, dict):
        return None
    meta = params.get("_meta")
    if not isinstance(meta, dict):
        return None
    if MODERN_META_KEY in meta:
        return meta[MODERN_META_KEY]
    return meta.get("protocolVersion")


class BridgeEra:
    """One connection's negotiated era, decided by its first request and
    locked for the connection's life. `era` is `None` (undecided),
    `"legacy"` or `"modern"`. `legacy_version` is only meaningful once
    `era == "legacy"` -- the version `initialize` negotiated, which is what
    decides whether a JSON-RPC batch array is ever accepted (only
    `"2025-03-26"`, the one legacy revision that still had batching).

    A request arriving before either `initialize` or a modern `_meta`
    version has decided anything is refused (`-32600`) rather than guessed
    at -- see `bridge_dispatch`'s own docstring."""

    def __init__(self) -> None:
        self.era: str | None = None
        self.legacy_version: str | None = None


def _error_bytes(msg_id: Any, code: int, message: str, data: Any = None) -> bytes:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return json.dumps(
        {"jsonrpc": "2.0", "id": msg_id, "error": err}, separators=(",", ":")
    ).encode("utf-8")


def _result_bytes(msg_id: Any, result: dict[str, Any]) -> bytes:
    return json.dumps(
        {"jsonrpc": "2.0", "id": msg_id, "result": result}, separators=(",", ":")
    ).encode("utf-8")


def _merge_body(prefix: dict[str, Any], body: bytes) -> bytes:
    """*body* (a complete JSON object, verbatim bytes -- never parsed) with
    *prefix*'s keys spliced in front of its own. The one case that needs
    care is an empty body (`b"{}"`): dropping its two bytes and closing on
    `prefix`'s own brace is what avoids a dangling comma, since there is
    nothing of the body's to append after one."""
    if not body.startswith(b"{") or not body.endswith(b"}"):
        raise ValueError("tool result body must be a JSON object")
    prefix_json = json.dumps(prefix, separators=(",", ":")).encode("utf-8")
    head = prefix_json[:-1]  # drop the trailing '}'
    if body == b"{}":
        return head + b"}"
    rest = body[1:]  # drop the leading '{'; the object's own trailing '}' stays
    if head.endswith(b"{"):
        return head + rest
    return head + b"," + rest


def splice_tool_result(msg_id: Any, body: bytes, *, meta: dict[str, Any] | None = None) -> bytes:
    """One JSON-RPC `tools/call` reply, built by splicing *body* -- a tool
    result already serialised elsewhere (Studio, over RPC v1) -- into the
    envelope, never by `json.loads`-ing it. `meta` is `None` for the legacy
    era (the body becomes `result` verbatim) or `{"resultType": "complete",
    "_meta": {...}}`-shaped for modern (see `_merge_body`)."""
    id_json = json.dumps(msg_id).encode("utf-8")
    result = body if meta is None else _merge_body(meta, body)
    return b'{"jsonrpc":"2.0","id":' + id_json + b',"result":' + result + b"}"


def discover_result(
    *, instructions: str | None, server_name: str, server_version: str
) -> dict[str, Any]:
    """The `server/discover` result object, modern era."""
    result: dict[str, Any] = {
        "supportedVersions": list(LEGACY) + list(MODERN),
        "capabilities": {"tools": {}},
        "ttlMs": 60000,
        "cacheScope": "public",
        "resultType": "complete",
        "_meta": {"serverInfo": {"name": server_name, "version": server_version}},
    }
    if instructions:
        result["instructions"] = instructions
    return result


def _legacy_initialize(
    msg_id: Any,
    params: dict[str, Any],
    *,
    instructions: str | None,
    server_name: str,
    server_version: str,
) -> tuple[str, bytes]:
    requested = params.get("protocolVersion")
    version = requested if requested in LEGACY else LEGACY[0]
    result: dict[str, Any] = {
        "protocolVersion": version,
        "capabilities": {"tools": {"listChanged": True}},
        "serverInfo": {"name": server_name, "version": server_version},
    }
    if instructions:
        result["instructions"] = instructions
    return version, _result_bytes(msg_id, result)


def _dispatch_one(
    item: Any,
    state: BridgeEra,
    *,
    catalogue: dict[str, Any],
    call_tool: Callable[[str, dict[str, Any]], bytes],
) -> bytes | None:
    """One JSON-RPC request or notification, either era. `None` means "this
    was a notification -- say nothing", the same convention `dispatch`
    above uses. Never raises: every branch that could fail the request
    itself becomes a JSON-RPC error instead."""
    if not isinstance(item, dict):
        return _error_bytes(None, -32600, "invalid request: expected a JSON object")

    method = item.get("method")
    has_id = "id" in item
    msg_id = item.get("id")
    if not isinstance(method, str) or not method:
        return _error_bytes(msg_id, -32600, "invalid request: 'method' must be a non-empty string")

    params = item.get("params", {})
    if params is None:
        params = {}
    if not isinstance(params, dict):
        return _error_bytes(msg_id, -32602, "'params' must be an object") if has_id else None

    server = catalogue.get("server") or {}
    server_name = server.get("name", SERVER_NAME)
    server_version = server.get("version", SERVER_VERSION)
    instructions = catalogue.get("instructions")

    if method == "initialize":
        state.era = "legacy"
        version, reply = _legacy_initialize(
            msg_id,
            params,
            instructions=instructions,
            server_name=server_name,
            server_version=server_version,
        )
        state.legacy_version = version
        return reply if has_id else None

    if method == "server/discover":
        if state.era is None:
            state.era = "modern"
        result = discover_result(
            instructions=instructions, server_name=server_name, server_version=server_version
        )
        return _result_bytes(msg_id, result) if has_id else None

    meta_version = _meta_version(params)

    if state.era is None:
        if meta_version is not None:
            state.era = "modern"
        else:
            return (
                _error_bytes(msg_id, -32600, "initialize or server/discover first")
                if has_id
                else None
            )

    if state.era == "legacy":
        if not has_id:
            return None
        if method == "notifications/initialized":
            return None
        if method == "ping":
            return _result_bytes(msg_id, {})
        if method == "tools/list":
            return _result_bytes(msg_id, {"tools": catalogue.get("tools", [])})
        if method == "tools/call":
            return _dispatch_tools_call(msg_id, params, call_tool=call_tool, meta=None)
        return _error_bytes(msg_id, -32601, f"unknown method: {method}")

    # modern
    if meta_version not in MODERN:
        return _error_bytes(
            msg_id,
            -32022,
            "unsupported protocol version",
            data={"supported": list(MODERN), "requested": meta_version},
        )
    if not has_id:
        return None
    server_info_meta = {"serverInfo": {"name": server_name, "version": server_version}}
    if method == "tools/list":
        result = {
            "tools": catalogue.get("tools", []),
            "ttlMs": 60000,
            "cacheScope": "public",
            "resultType": "complete",
            "_meta": server_info_meta,
        }
        return _result_bytes(msg_id, result)
    if method == "tools/call":
        splice_prefix = {"resultType": "complete", "_meta": server_info_meta}
        return _dispatch_tools_call(msg_id, params, call_tool=call_tool, meta=splice_prefix)
    return _error_bytes(msg_id, -32601, f"unknown method: {method}")


def _dispatch_tools_call(
    msg_id: Any,
    params: dict[str, Any],
    *,
    call_tool: Callable[[str, dict[str, Any]], bytes],
    meta: dict[str, Any] | None,
) -> bytes:
    name = params.get("name")
    if not isinstance(name, str) or not name:
        return _error_bytes(msg_id, -32602, "tools/call needs a string 'name'")
    arguments = params.get("arguments", {})
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        return _error_bytes(msg_id, -32602, "'arguments' must be an object")
    try:
        body = call_tool(name, arguments)
    except Exception as exc:  # noqa: BLE001 -- call_tool promises not to raise;
        # this is the same backstop dispatch() keeps above, so a broken
        # promise still becomes isError content, not a dropped connection.
        body = json.dumps(
            fail(f"{type(exc).__name__}: {exc}"), separators=(",", ":")
        ).encode("utf-8")
    return splice_tool_result(msg_id, body, meta=meta)


def bridge_dispatch(
    raw: bytes,
    state: BridgeEra,
    *,
    catalogue: dict[str, Any],
    call_tool: Callable[[str, dict[str, Any]], bytes],
) -> bytes | None:
    """One line of stdin, from an MCP client, answered as one line of
    stdout (or `None` for "nothing to send": an all-notification batch, or a
    lone notification). *catalogue* is the RPC v1 `catalogue` reply (`hash`,
    `tools`, `instructions`, `server`) `bridge.py` already holds -- its
    `tools` are already the JSON dicts this function puts straight on the
    wire, unparsed and unrebuilt. *call_tool* runs one tool via Studio's RPC
    v1 `call` op and returns the raw result body bytes; never raises (see
    `_dispatch_tools_call`'s backstop for the day that promise breaks).

    A JSON-RPC batch (a top-level array) is only ever answered when this
    connection is legacy-era and negotiated exactly `"2025-03-26"` -- the
    one legacy revision that still had batching. Every other era/version
    combination refuses a batch outright (`-32600`), per the MCP spec's own
    removal of it in `"2025-06-18"`.
    """
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return _error_bytes(None, -32700, f"parse error: {exc}") + b"\n"

    if isinstance(parsed, list):
        if not (state.era == "legacy" and state.legacy_version == "2025-03-26"):
            return (
                _error_bytes(None, -32600, "batch requests are not supported on this connection")
                + b"\n"
            )
        if not parsed:
            return _error_bytes(None, -32600, "empty batch") + b"\n"
        fragments = [
            _dispatch_one(item, state, catalogue=catalogue, call_tool=call_tool) for item in parsed
        ]
        fragments = [f for f in fragments if f is not None]
        if not fragments:
            return None
        return b"[" + b",".join(fragments) + b"]\n"

    if not isinstance(parsed, dict):
        return _error_bytes(None, -32600, "invalid request: expected a JSON object") + b"\n"

    reply = _dispatch_one(parsed, state, catalogue=catalogue, call_tool=call_tool)
    return None if reply is None else reply + b"\n"
