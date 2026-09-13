"""MCP over JSON-RPC 2.0 -- the wire format, with no idea what a tool does.

`bridge_dispatch` (at the bottom of this file) is `warlock mcp` (`bridge.
py`)'s dual-era MCP *server* logic -- legacy `initialize`-first negotiation
and a newer, no-`initialize` "modern" era, JSON-RPC batching, and splicing a
tool result's raw bytes (fetched from Studio over `rpc.py`'s private RPC v1)
into whichever envelope a connection's negotiated era calls for, all
without ever `json.loads`-ing that result.

There used to be a second dispatcher here, `dispatch`, that answered bare
MCP JSON-RPC directly on Studio's own pipe -- `studio/agent_host.py` called
it before Studio spoke only RPC v1. It is gone: `docs/INVARIANTS.md`'s agent
paragraph is now "Studio speaks only RPC v1; the bridge is the only MCP
server", and `warlock.studio` importing this module at all is a pinned
regression (`tests/mcp/test_mcp_imports.py`). Nothing here knows Clay
exists; `studio/agent_clay.py` and `studio/agent_host.py` own that, and now
reach `Tool`/`ok`/`fail`/`text`/`image_png`/`MAX_FRAME` through `rpc.py`
directly rather than through this module.

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
whole reason a tool's own `call` contract is "must not raise, and if it does
anyway `bridge_dispatch`'s own backstop still turns it into `isError` rather
than propagating it."
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from .rpc import (  # noqa: F401 -- re-exported
    MAX_FRAME,
    SERVER_NAME,
    Tool,
    fail,
    image_png,
    ok,
    text,
)

SERVER_VERSION = "0.0.0"
"""Overwritten by whoever knows the real version, before this process's first
MCP request. Only `bridge.py` sets this now -- once, from the RPC v1 `hello`
reply's `studio_version` -- since `warlock.studio` no longer imports this
module at all (see the module docstring). This module still cannot import
`warlock.__version__` itself: that would break the "pure stdlib, no warlock
imports" rule that lets it be unit-tested and imported on a headless box, so
a module constant, set by whoever knows the real version, remains the way
`bridge_dispatch` and `discover_result` learn it without either taking a
`version` parameter that would have to be threaded through every call.
"""

# Tool, ok, fail, text, image_png, MAX_FRAME and SERVER_NAME now live in
# `rpc.py` -- the private RPC v1 wire format shares them with this MCP one --
# and are imported above so every existing `protocol.Tool` / `protocol.ok` /
# `protocol.SERVER_NAME` / etc. call site keeps working unchanged.


def encode(message: dict[str, Any]) -> bytes:
    """One JSON-RPC message, newline-delimited, UTF-8."""
    return (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")


def decode(line: bytes) -> dict[str, Any]:
    """The inverse of `encode`. Raises `ValueError` on anything unusable.

    Two refusals live here rather than in the caller: an oversize frame (see
    `MAX_FRAME`) and a frame that parses but is not a JSON *object* -- a bare
    `42` or `"hi"` is valid JSON and invalid JSON-RPC, and letting it through
    would hand a caller a `.get` call on a list or a string.
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


# ---------------------------------------------------------------------------
# Bridge-side, dual-era dispatch.
#
# `bridge.py` (``warlock mcp``) is the real MCP server a third-party
# client's tool runner dials, so it has to speak whatever revision that
# client actually negotiates -- both a legacy family and a newer "modern"
# era that drops `initialize` for `server/discover` and carries its version
# per-request. Studio itself never has to know about any of this: it
# answers a private RPC (`rpc.py`), not MCP, and the bridge is the only
# thing that speaks MCP to the outside world. The functions below build
# that: era negotiation, per-request modern versioning, and splicing a tool
# result's raw JSON bytes (fetched from Studio over RPC v1, never re-parsed
# here) into whichever envelope this connection's era calls for.
# ---------------------------------------------------------------------------

LEGACY = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
"""Every MCP revision the bridge may honestly negotiate under classic,
`initialize`-first JSON-RPC semantics, newest first. The bridge is a real
MCP server facing real clients and has to keep up with what they actually
request."""

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
    """The `server/discover` result object, modern era. `capabilities`
    advertises `resources` and `prompts` alongside `tools` -- both empty
    objects, the same "this exists, nothing further to negotiate" shape
    `tools` already used."""
    result: dict[str, Any] = {
        "supportedVersions": list(LEGACY) + list(MODERN),
        "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
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
        "capabilities": {
            "tools": {"listChanged": True},
            "resources": {"listChanged": False},
            "prompts": {"listChanged": False},
        },
        "serverInfo": {"name": server_name, "version": server_version},
    }
    if instructions:
        result["instructions"] = instructions
    return version, _result_bytes(msg_id, result)


_NOT_HANDLED = object()
"""Sentinel :func:`_resource_prompt_method` returns for any method that is
not one of the five resources/prompts methods -- distinct from `None`,
which that function also returns legitimately (a notification with no id
gets no reply)."""


def _strip_inline_content(resources: list[Any]) -> list[dict[str, Any]]:
    """*resources*, each with its `text`/`blob` key (if any) removed.

    A `resources/list` result is metadata only (`uri`, `name`, `title`,
    `description`, `mimeType`) -- the catalogue's own resource entries carry
    an inline `text` for the three static Clay resources too, so a bridge
    dialled while Studio is unreachable can still answer `resources/read`
    for them from the snapshot (see `studio/agent_resources.py`), but that
    inline copy has no business riding along on a mere listing."""
    return [{k: v for k, v in r.items() if k not in ("text", "blob")} for r in resources]


def _resource_prompt_method(
    method: str,
    msg_id: Any,
    has_id: bool,
    params: dict[str, Any],
    *,
    modern: bool,
    catalogue: dict[str, Any],
    read_resource: Callable[[str], dict[str, Any] | None] | None,
    get_prompt: Callable[[str, dict[str, Any]], Any] | None,
    server_info_meta: dict[str, Any] | None,
) -> bytes | None:
    """The five MCP resources/prompts methods, shared between the legacy and
    modern branches of :func:`_dispatch_one` -- the two eras differ only in
    whether a success result carries `ttlMs`/`cacheScope`/`resultType`/
    `_meta` (*modern*) or not, and in the not-found error code (`-32002`
    legacy, `-32602` modern, per the MCP spec's own per-era numbering).

    *read_resource* answers one `resources/read`: `None` for "no such
    resource", or `{"contents": [...], "ttlMs": ..., "cacheScope": ...}`.
    *get_prompt* answers one `prompts/get`: `None` for "no such prompt", a
    `list[str]` of missing required argument names, or `{"description":
    ..., "messages": [...]}`. Both are `None` (rather than a callable) when
    the caller has nothing to reach Studio with -- see `bridge.py`'s own
    `_Session` for what it passes when the app is down and only the
    catalogue snapshot is available.

    Returns :data:`_NOT_HANDLED` for any other method, so a caller can fall
    through to its own "unknown method" refusal."""
    if method == "resources/list":
        if not has_id:
            return None
        result: dict[str, Any] = {
            "resources": _strip_inline_content(catalogue.get("resources", []))
        }
        if modern:
            result.update(
                ttlMs=60000, cacheScope="public", resultType="complete", _meta=server_info_meta
            )
        return _result_bytes(msg_id, result)

    if method == "resources/templates/list":
        if not has_id:
            return None
        result = {"templates": []}
        if modern:
            result.update(
                ttlMs=60000, cacheScope="public", resultType="complete", _meta=server_info_meta
            )
        return _result_bytes(msg_id, result)

    if method == "resources/read":
        uri = params.get("uri")
        if not isinstance(uri, str) or not uri:
            return (
                _error_bytes(msg_id, -32602, "resources/read needs a string 'uri'")
                if has_id
                else None
            )
        if read_resource is None:
            return _error_bytes(msg_id, -32601, f"unknown method: {method}") if has_id else None
        found = read_resource(uri)
        if found is None:
            code = -32602 if modern else -32002
            return (
                _error_bytes(msg_id, code, "resource not found", data={"uri": uri})
                if has_id
                else None
            )
        if not has_id:
            return None
        result = {"contents": found.get("contents", [])}
        if modern:
            result.update(
                ttlMs=found.get("ttlMs", 0),
                cacheScope=found.get("cacheScope", "private"),
                resultType="complete",
                _meta=server_info_meta,
            )
        return _result_bytes(msg_id, result)

    if method == "prompts/list":
        if not has_id:
            return None
        result = {"prompts": catalogue.get("prompts", [])}
        if modern:
            result.update(
                ttlMs=60000, cacheScope="public", resultType="complete", _meta=server_info_meta
            )
        return _result_bytes(msg_id, result)

    if method == "prompts/get":
        name = params.get("name")
        if not isinstance(name, str) or not name:
            return (
                _error_bytes(msg_id, -32602, "prompts/get needs a string 'name'")
                if has_id
                else None
            )
        arguments = params.get("arguments", {})
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return (
                _error_bytes(msg_id, -32602, "'arguments' must be an object") if has_id else None
            )
        if get_prompt is None:
            return _error_bytes(msg_id, -32601, f"unknown method: {method}") if has_id else None
        found = get_prompt(name, arguments)
        if found is None:
            code = -32602 if modern else -32002
            return (
                _error_bytes(msg_id, code, "prompt not found", data={"name": name})
                if has_id
                else None
            )
        if isinstance(found, list):
            return (
                _error_bytes(msg_id, -32602, "missing required arguments", data={"missing": found})
                if has_id
                else None
            )
        if not has_id:
            return None
        result = {"description": found.get("description"), "messages": found.get("messages", [])}
        if modern:
            result["_meta"] = server_info_meta
        return _result_bytes(msg_id, result)

    return _NOT_HANDLED  # type: ignore[return-value]


def _dispatch_one(
    item: Any,
    state: BridgeEra,
    *,
    catalogue: dict[str, Any],
    call_tool: Callable[[str, dict[str, Any]], bytes],
    read_resource: Callable[[str], dict[str, Any] | None] | None = None,
    get_prompt: Callable[[str, dict[str, Any]], Any] | None = None,
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
        handled = _resource_prompt_method(
            method,
            msg_id,
            has_id,
            params,
            modern=False,
            catalogue=catalogue,
            read_resource=read_resource,
            get_prompt=get_prompt,
            server_info_meta=None,
        )
        if handled is not _NOT_HANDLED:
            return handled
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
    handled = _resource_prompt_method(
        method,
        msg_id,
        has_id,
        params,
        modern=True,
        catalogue=catalogue,
        read_resource=read_resource,
        get_prompt=get_prompt,
        server_info_meta=server_info_meta,
    )
    if handled is not _NOT_HANDLED:
        return handled
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
    read_resource: Callable[[str], dict[str, Any] | None] | None = None,
    get_prompt: Callable[[str, dict[str, Any]], Any] | None = None,
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
            _dispatch_one(
                item,
                state,
                catalogue=catalogue,
                call_tool=call_tool,
                read_resource=read_resource,
                get_prompt=get_prompt,
            )
            for item in parsed
        ]
        fragments = [f for f in fragments if f is not None]
        if not fragments:
            return None
        return b"[" + b",".join(fragments) + b"]\n"

    if not isinstance(parsed, dict):
        return _error_bytes(None, -32600, "invalid request: expected a JSON object") + b"\n"

    reply = _dispatch_one(
        parsed,
        state,
        catalogue=catalogue,
        call_tool=call_tool,
        read_resource=read_resource,
        get_prompt=get_prompt,
    )
    return None if reply is None else reply + b"\n"
