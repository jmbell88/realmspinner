"""Warlock's own private RPC v1 -- stdlib-only, and a leaf like its siblings.

**Why this exists beside `protocol.py`.** MCP is spoken only between an
agent's client and `bridge.py` (`protocol.py`). This module is the one
private channel between `bridge.py` and Studio over the pipe (`pipe.py`) --
compact JSON requests, a JSON header plus an optional raw body for replies --
so Studio never has to know which MCP revision a client speaks.

**Wire shape.** A request is one frame: compact JSON, `{"rpc": 1, "op": ...}`
plus whatever fields that op needs. A reply is one frame too, but two parts
concatenated: a JSON header object, then a single `b"\\n"`, then an optional
raw body -- `split_reply` is the inverse of that concatenation, not a second
JSON parse of the whole frame. The header is always small and always valid
JSON on its own; the body, when present, is arbitrary bytes the caller
already has cheaply serialised (today, always a tool result already built
as JSON text) and this module never touches its contents.

**Ops, v1.**

* `hello` -- request carries `versions`, the list of RPC integers the bridge
  understands, and `bridge_version`, its release string (informational). The
  envelope's own `"rpc"` key is not the list: it names the version *this
  frame* is written in. If none overlaps :data:`SUPPORTED_RPC_VERSIONS`, the
  reply is `{"error": {"code": "rpc_version", "supported": [...]}}` and
  nothing else; otherwise it is `{"rpc": 1, "studio_version": ...,
  "catalogue_hash": ..., "call_timeout": ...}`. No body.
* `catalogue` -- reply header is `{"hash": ..., "tools": [...], "instructions":
  ..., "server": {"name": ..., "version": ...}, "resources": [...],
  "prompts": [...]}`, in that key order, built by :func:`catalogue_payload`.
  `resources` and `prompts` are the same lists :func:`bridge_dispatch`
  serves for `resources/list`/`prompts/list` -- see that module's own docs.
  No body.
* `resources` -- reply header is `{"resources": [...], "templates": []}`,
  the exact shape `resources/list`/`resources/templates/list` need on the
  wire. No body.
* `read` -- request carries `uri`. Reply header is `{"uri": ..., "mimeType":
  ...}` and the body is the resource's own bytes (UTF-8 text for a text/JSON
  resource, raw bytes -- e.g. a PNG -- otherwise; the bridge base64-encodes
  a non-text body into a `blob`, never a `text`), or, when nothing answers to
  that uri, `{"error": {"code": "not_found"}}` with no body.
* `prompts` -- reply header is `{"prompts": [...]}`. No body.
* `prompt` -- request carries `name` and `arguments`. Reply header is
  `{"description": ..., "messages": [...]}`, or `{"error": {"code":
  "not_found"}}` for an unknown name, or `{"error": {"code":
  "bad_arguments", "missing": [...]}}` for a known prompt missing a required
  argument. No body.
* `call` -- request carries `tool` and `args`. Reply header is `{"hash": ...}`
  and the body is the raw result JSON, byte-for-byte what `tools/call` would
  have put in its JSON-RPC `result` field on the MCP path -- the same object,
  a different envelope. **`hash` here is the *catalogue*'s hash (the same
  value `hello` and `catalogue` report), never a hash of this call's own
  result** -- a bridge compares it against the catalogue it already holds to
  decide whether to re-fetch, and hashing the result instead would change
  this field on every single call (two different tool results hash
  differently) and make the bridge believe the catalogue moved after every
  ordinary call.
* Anything else -- `{"error": {"code": "unknown_op"}}`.
* A request this module cannot decode at all (oversize, not JSON, not an
  object) -- `{"error": {"code": "bad_request"}}`.

**Versioning rule, and it is the whole contract:** adding a key to any op's
request or reply, or adding a new op, is allowed within v1 -- an older peer
that does not recognise a key ignores it, and one that does not recognise an
op gets `unknown_op` rather than a confusing partial answer. Removing a key,
renaming one, or changing what an existing key *means* is not a v1 change --
that bumps the leading integer, and :data:`SUPPORTED_RPC_VERSIONS` is where a
peer negotiates which integers it can honestly speak.
"""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Any, NamedTuple

SUPPORTED_RPC_VERSIONS = (1,)
"""Every RPC integer this module can honestly answer `hello` with. See the
module docstring's versioning rule for what does, and does not, bump this."""

SERVER_NAME = "warlock-studio"
"""The name Studio answers as, on both wire formats -- moved here from
`protocol.py` (which re-exports it) since it is `rpc.py`'s own
`catalogue_payload`/`hello_header` that build the envelope carrying it, and
`studio/agent_host.py` reads it from here directly rather than from
`protocol`, which `warlock.studio` must never import."""

MAX_FRAME = 8 << 20
"""Ceiling on one frame -- request or reply -- moved here from `protocol.py`
(re-exported there) since both wire formats share the same transport and the
same reasoning: generous for a small request or a modest tool result, tight
enough that a confused or hostile peer cannot force an unbounded read."""


class Tool(NamedTuple):
    name: str
    title: str
    description: str
    schema: dict[str, Any]  # JSON Schema for the tool's arguments
    output_schema: dict[str, Any] | None = None
    """JSON Schema for `structuredContent`, when a tool declares one. `None`
    (the default) for every tool that does not. Moved here from `protocol.py`
    (re-exported there) since it is shared vocabulary between both wire
    formats -- a `Tool` is a `Tool` whether it is being listed for `tools/list`
    or for the `catalogue` op."""


def encode_request(op: str, **fields: Any) -> bytes:
    """One RPC request, compact JSON, UTF-8. No trailing newline -- unlike
    `protocol.encode`, a request here is the *whole* frame handed to
    `send_bytes`, never one line among several sharing it."""
    message: dict[str, Any] = {"rpc": 1, "op": op, **fields}
    return json.dumps(message, separators=(",", ":")).encode("utf-8")


def decode_request(frame: bytes) -> dict[str, Any]:
    """The inverse of `encode_request`. Raises `ValueError` on anything
    unusable -- oversize, undecodable, or not a JSON object -- mirroring
    `protocol.decode`'s own refusals for the same reasons."""
    if len(frame) > MAX_FRAME:
        raise ValueError(f"frame of {len(frame)} bytes exceeds MAX_FRAME ({MAX_FRAME})")
    try:
        message = json.loads(frame.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed JSON: {exc}") from exc
    if not isinstance(message, dict):
        raise ValueError(f"expected a JSON object, got {type(message).__name__}")
    return message


def looks_like_rpc(frame: bytes) -> bool:
    """Whether *frame* is plausibly a v1 RPC request, cheaply -- used only to
    check the *first* frame of a connection, which `studio/agent_host.py`
    answers with `bad_request` and a closed connection when it is not.
    Never raises: an undecodable frame is declined, not diagnosed."""
    try:
        message = json.loads(frame.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(message, dict) and "rpc" in message


def encode_reply(header: dict[str, Any], body: bytes = b"") -> bytes:
    """A reply frame: *header* as compact JSON, a single newline, then
    *body* verbatim. *body* is never itself JSON-encoded by this function --
    the caller already has it as bytes (typically a tool result already
    serialised once) and re-encoding it here would be a second, pointless
    copy."""
    return json.dumps(header, separators=(",", ":")).encode("utf-8") + b"\n" + body


def split_reply(frame: bytes) -> tuple[dict[str, Any], bytes]:
    """The inverse of `encode_reply`: the header object and the raw body
    that followed its newline. Raises `ValueError` if there is no newline to
    split on, or the header half is not a JSON object -- the same shape of
    refusal `decode_request` uses, for the same reason."""
    try:
        header_bytes, body = frame.split(b"\n", 1)
    except ValueError as exc:
        raise ValueError("reply frame has no header/body separator") from exc
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed header JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"expected a JSON object header, got {type(header).__name__}")
    return header, body


def canonical_hash(obj: Any) -> str:
    """A short, stable digest of *obj*'s canonical JSON form.

    `sort_keys=True` makes key order irrelevant and `separators=(",", ":")`
    makes whitespace irrelevant, so two callers building "the same" object
    by different paths hash identically; `default=str` is a backstop for a
    value `json.dumps` would otherwise refuse, not a case any caller here is
    expected to actually hit. This is the same recipe
    `studio/agent_host.py`'s `_fingerprint` used before this module existed
    -- it now calls this function instead, and a pinned test in
    `tests/mcp/test_rpc_studio.py` holds the two byte-identical.

    Used for two unrelated things today: a tool call's dedup fingerprint
    (tool + arguments), and `catalogue_hash` (the tool list + instructions)
    -- both want "same input, same short key", nothing more.
    """
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.blake2b(canonical.encode("utf-8"), digest_size=8).hexdigest()


def text(s: str) -> dict[str, str]:
    return {"type": "text", "text": s}


def image_png(data: bytes) -> dict[str, str]:
    return {
        "type": "image",
        "data": base64.b64encode(data).decode("ascii"),
        "mimeType": "image/png",
    }


def ok(*content: dict[str, Any], structured: dict[str, Any] | None = None) -> dict[str, Any]:
    """A tool result that succeeded. Moved here from `protocol.py` (which
    re-exports it) -- see that module's history for the full rationale;
    unchanged."""
    result: dict[str, Any] = {"content": list(content), "isError": False}
    if structured is not None:
        result["structuredContent"] = structured
    return result


def fail(message: str, **extra: Any) -> dict[str, Any]:
    """A tool result that reports failure without ever becoming a transport
    error. Moved here from `protocol.py` (which re-exports it); unchanged."""
    result: dict[str, Any] = {"content": [text(message)], "isError": True}
    if extra:
        result["structuredContent"] = extra
    return result


def tool_dict(tool: Tool) -> dict[str, Any]:
    """One `Tool` as the JSON object the `catalogue` op puts on the wire --
    the same shape MCP's `tools/list` (`bridge_dispatch`, via the
    `catalogue` payload this module builds) forwards verbatim, never
    re-encoding it."""
    result: dict[str, Any] = {
        "name": tool.name,
        "title": tool.title,
        "description": tool.description,
        "inputSchema": tool.schema,
    }
    if tool.output_schema is not None:
        result["outputSchema"] = tool.output_schema
    return result


def hello_header(
    bridge_versions: Any,
    *,
    studio_version: str,
    catalogue_hash: str,
    call_timeout: float,
) -> dict[str, Any]:
    """The `hello` reply header. *bridge_versions* is whatever the request's
    `versions` field held -- validated here, not assumed to already be
    a list of ints, since it arrived over the wire. No overlap with
    :data:`SUPPORTED_RPC_VERSIONS` is the one refusal this op has."""
    versions = bridge_versions if isinstance(bridge_versions, list) else []
    overlap = [v for v in versions if v in SUPPORTED_RPC_VERSIONS]
    if not overlap:
        return {"error": {"code": "rpc_version", "supported": list(SUPPORTED_RPC_VERSIONS)}}
    return {
        "rpc": 1,
        "studio_version": studio_version,
        "catalogue_hash": catalogue_hash,
        "call_timeout": call_timeout,
    }


def catalogue_payload(
    tools: Any,
    *,
    instructions: str | None,
    server_name: str,
    server_version: str,
    resources: Any = None,
    prompts: Any = None,
) -> dict[str, Any]:
    """The `catalogue` reply header, in deterministic key order: `hash`,
    `tools`, `instructions`, `server`, `resources`, `prompts`. The hash
    covers the tool list (as the JSON dicts `tool_dict` builds), plus
    `instructions`, `resources` and `prompts` -- everything a client
    actually needs to know changed -- not `server`, which changes every
    release regardless of whether a single tool moved.

    *resources* and *prompts* default to an empty list each so every
    caller written before resources and prompts existed (a `catalogue_payload(tools,
    instructions=..., server_name=..., server_version=...)` call with no
    opinion about either) keeps working unchanged."""
    tools_json = [tool_dict(t) for t in tools]
    resources_json = list(resources or [])
    prompts_json = list(prompts or [])
    server = {"name": server_name, "version": server_version}
    digest = canonical_hash(
        {
            "tools": tools_json,
            "instructions": instructions,
            "resources": resources_json,
            "prompts": prompts_json,
        }
    )
    return {
        "hash": digest,
        "tools": tools_json,
        "instructions": instructions,
        "server": server,
        "resources": resources_json,
        "prompts": prompts_json,
    }


def unknown_op_header() -> dict[str, Any]:
    return {"error": {"code": "unknown_op"}}


def bad_request_header() -> dict[str, Any]:
    return {"error": {"code": "bad_request"}}
