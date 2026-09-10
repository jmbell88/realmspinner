"""MCP over JSON-RPC 2.0 -- the wire format, with no idea what a tool does.

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

import base64
import json
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

PROTOCOL_VERSION = "2025-06-18"
SERVER_NAME = "warlock-studio"

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

MAX_FRAME = 8 << 20
"""Ceiling on one newline-delimited JSON frame, in `encode`/`decode`.

MCP messages here are small requests and modest tool results (an image is
carried as base64 content, not a raw blob); 8 MiB is generous for that and
tight enough that a confused or hostile peer sending an unbounded line gets a
clean refusal instead of an unbounded `bytes` accumulation on the read side."""


class Tool(NamedTuple):
    name: str
    title: str
    description: str
    schema: dict[str, Any]  # JSON Schema for the tool's arguments


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


def text(s: str) -> dict[str, str]:
    return {"type": "text", "text": s}


def image_png(data: bytes) -> dict[str, str]:
    return {
        "type": "image",
        "data": base64.b64encode(data).decode("ascii"),
        "mimeType": "image/png",
    }


def ok(*content: dict[str, Any]) -> dict[str, Any]:
    return {"content": list(content), "isError": False}


def fail(message: str, **extra: Any) -> dict[str, Any]:
    """A tool result that reports failure without ever becoming a JSON-RPC error.

    `message` always lands as readable text content -- an agent's model reads
    that, not `structuredContent` -- and `extra` (e.g. `field="thickness"`,
    echoing `service.errors`' own convention) rides along as structured data
    for a caller that wants to branch on it instead of parsing prose.
    """
    result: dict[str, Any] = {"content": [text(message)], "isError": True}
    if extra:
        result["structuredContent"] = extra
    return result


def _tool_json(tool: Tool) -> dict[str, Any]:
    return {
        "name": tool.name,
        "title": tool.title,
        "description": tool.description,
        "inputSchema": tool.schema,
    }


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
            result: dict[str, Any] = {
                "protocolVersion": PROTOCOL_VERSION,
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
