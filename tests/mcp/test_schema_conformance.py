"""`bridge_dispatch`'s modern-era replies, checked against the real upstream
JSON Schemas rather than this codebase's own reading of the spec's prose.

Every audit note in `protocol.py` before 2026-09-26 about "no worked JSON
example was available" was this module's gap: nothing here ever fetched the
actual `ext-tasks` or core MCP schema and checked a reply against it, so a
plausible-looking but wrong shape (a `task` object nested where the schema
wants flat fields, a bare `serverInfo` key where the schema reserves a
namespaced one) could sit unnoticed indefinitely. This file is the
regression gate for both: it validates `discover_result`, a `tools/call`
reply, `CreateTaskResult`, `tasks/get` and `tasks/cancel` against the vendored
fixtures in `tests/mcp/fixtures/` (trimmed copies of the real schemas,
fetched 2026-09-26 -- see that directory's own README for source URLs and
what was kept).

Offline by design, like the rest of this suite: the fixtures are checked-in
JSON, never fetched at test time.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from realmspinner.mcp import protocol as p

FIXTURES = Path(__file__).parent / "fixtures"

_EXT_TASKS = json.loads((FIXTURES / "ext_tasks_2026-07-28.json").read_text(encoding="utf-8"))
_MCP = json.loads((FIXTURES / "mcp_2026-07-28.json").read_text(encoding="utf-8"))


def _validator(schema_doc: dict, def_name: str) -> jsonschema.protocols.Validator:
    """A validator for one `$defs` entry of *schema_doc*, resolved against
    that same document (every fixture here was checked, when it was built,
    to have no `$ref` escaping its own trimmed `$defs` -- see the fixtures'
    own README)."""
    schema = {**schema_doc["$defs"][def_name], "$defs": schema_doc["$defs"]}
    cls = jsonschema.validators.validator_for(schema)
    cls.check_schema(schema)
    return cls(schema)


def _assert_valid(schema_doc: dict, def_name: str, instance: dict) -> None:
    validator = _validator(schema_doc, def_name)
    errors = list(validator.iter_errors(instance))
    assert not errors, "\n".join(f"{def_name}: {e.message} at {list(e.path)}" for e in errors)


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
        "server": {"name": "realmspinner", "version": "1.2.3"},
    }
    base.update(overrides)
    return base


def _ok_call_tool(name, args):
    return json.dumps(p.ok(p.text(f"ran {name}"))).encode("utf-8")


def _dispatch(payload, state, **kwargs):
    raw = json.dumps(payload).encode("utf-8")
    reply = p.bridge_dispatch(
        raw,
        state,
        catalogue=kwargs.pop("catalogue", None) or _catalogue(),
        call_tool=kwargs.pop("call_tool", _ok_call_tool),
        **kwargs,
    )
    if reply is None:
        return None
    return json.loads(reply)


def _modern_meta(version=None):
    return {"_meta": {p.MODERN_META_KEY: version or p.MODERN[0]}}


def _modern_meta_with_tasks(version=None):
    return {
        "_meta": {
            p.MODERN_META_KEY: version or p.MODERN[0],
            "io.modelcontextprotocol/clientCapabilities": {
                "extensions": {p.TASKS_EXTENSION: {}}
            },
        }
    }


# --- server/discover ---------------------------------------------------------


def test_discover_result_matches_the_real_schema_with_tasks_declared() -> None:
    state = p.BridgeEra()
    reply = _dispatch(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "server/discover",
            "params": _modern_meta_with_tasks(),
        },
        state,
    )
    _assert_valid(_MCP, "DiscoverResult", reply["result"])


def test_discover_result_matches_the_real_schema_without_tasks() -> None:
    state = p.BridgeEra()
    reply = _dispatch({"jsonrpc": "2.0", "id": 1, "method": "server/discover"}, state)
    _assert_valid(_MCP, "DiscoverResult", reply["result"])


# --- tools/call (ordinary, non-task) -----------------------------------------


def test_modern_tools_call_reply_meta_matches_result_meta_object() -> None:
    state = p.BridgeEra()
    _dispatch({"jsonrpc": "2.0", "id": 1, "method": "server/discover"}, state)
    reply = _dispatch(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "clay_scene", "arguments": {}, **_modern_meta()},
        },
        state,
    )
    result = reply["result"]
    assert result["resultType"] == "complete"
    _assert_valid(_MCP, "ResultMetaObject", result["_meta"])


# --- MCP Tasks extension ------------------------------------------------------


def test_create_task_result_matches_the_real_ext_tasks_schema() -> None:
    state = p.BridgeEra()
    _dispatch(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "server/discover",
            "params": _modern_meta_with_tasks(),
        },
        state,
    )
    reply = _dispatch(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "clay_scene", "arguments": {}, **_modern_meta_with_tasks()},
        },
        state,
        call_tool_task=lambda name, args: ("op-1", "working"),
    )
    _assert_valid(_EXT_TASKS, "CreateTaskResult", reply["result"])


@pytest.mark.parametrize(
    "status,body",
    [
        ("working", None),
        ("completed", json.dumps(p.ok(p.text("done"))).encode("utf-8")),
        ("cancelled", None),
    ],
)
def test_tasks_get_reply_matches_the_real_ext_tasks_schema(status, body) -> None:
    state = p.BridgeEra()
    _dispatch(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "server/discover",
            "params": _modern_meta_with_tasks(),
        },
        state,
    )
    reply = _dispatch(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tasks/get",
            "params": {"taskId": "op-1", **_modern_meta_with_tasks()},
        },
        state,
        get_task=lambda task_id: (status, body),
    )
    _assert_valid(_EXT_TASKS, "GetTaskResult", reply["result"])


def test_tasks_get_failed_matches_the_schema_outside_its_documented_error_deviation() -> None:
    """`FailedTask.error` is schema-typed as the JSON-RPC `Error` shape
    (`{code, message}`) -- but `_tasks_get_bytes`'s own docstring documents a
    deliberate deviation: it splices in whatever `fail()` shape (`{content,
    isError}`) Realmspinner already produced for the failed call, because "a tool
    failing is not a JSON-RPC error" is this codebase's own load-bearing
    rule (protocol.py's opening docstring). So this checks everything
    `GetTaskResult` requires *except* validating `error`'s own inner shape
    against `Error` -- the one place this bridge knowingly does not conform,
    by design, not by oversight."""
    state = p.BridgeEra()
    _dispatch(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "server/discover",
            "params": _modern_meta_with_tasks(),
        },
        state,
    )
    reply = _dispatch(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tasks/get",
            "params": {"taskId": "op-1", **_modern_meta_with_tasks()},
        },
        state,
        get_task=lambda task_id: ("failed", json.dumps(p.fail("nope")).encode("utf-8")),
    )
    result = reply["result"]
    assert result["resultType"] == "complete"
    assert result["taskId"] == "op-1"
    assert result["status"] == "failed"
    for field in ("createdAt", "lastUpdatedAt"):
        assert isinstance(result[field], str) and result[field]
    assert result["ttlMs"] == p.TASK_TTL_MS
    assert "error" in result
    _assert_valid(_MCP, "ResultMetaObject", result["_meta"])


def test_tasks_cancel_result_matches_the_real_ext_tasks_schema() -> None:
    state = p.BridgeEra()
    _dispatch(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "server/discover",
            "params": _modern_meta_with_tasks(),
        },
        state,
    )
    reply = _dispatch(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tasks/cancel",
            "params": {"taskId": "op-1", **_modern_meta_with_tasks()},
        },
        state,
        cancel_task=lambda task_id: "cancelled",
    )
    _assert_valid(_EXT_TASKS, "CancelTaskResult", reply["result"])


def test_call_tool_task_returning_bytes_is_a_plain_complete_reply_never_a_task() -> None:
    """The 2026-09-26 `call_tool_task` contract: it may return `bytes` (a
    complete tool-result body, same shape `call_tool` returns) instead of
    `(operation_id, status)` when Realmspinner refused the call before any task
    could exist for it. That reply must validate as an ordinary spliced
    `tools/call` result (`resultType: "complete"`) -- never a
    `CreateTaskResult`, and never carrying a fabricated task id."""
    state = p.BridgeEra()
    _dispatch(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "server/discover",
            "params": _modern_meta_with_tasks(),
        },
        state,
    )
    refusal_body = json.dumps(p.fail("vram exhausted")).encode("utf-8")
    reply = _dispatch(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "clay_scene", "arguments": {}, **_modern_meta_with_tasks()},
        },
        state,
        call_tool_task=lambda name, args: refusal_body,
    )
    result = reply["result"]
    assert result["resultType"] == "complete"
    assert "taskId" not in result
    assert "task" not in result
    assert result["isError"] is True
    assert result["content"] == [{"type": "text", "text": "vram exhausted"}]
    _assert_valid(_MCP, "ResultMetaObject", result["_meta"])


# --- capability sequence: task call, then plain call -------------------------


def test_task_call_then_plain_call_on_the_same_connection() -> None:
    """A task-augmented `tools/call` followed by an ordinary one (no Tasks
    declaration) on the *same* `BridgeEra` -- proves task mode is decided
    per request, never latched across a connection's later calls (see
    `BridgeEra`'s own docstring on why the old `self.tasks` flag was
    removed)."""
    state = p.BridgeEra()
    _dispatch(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "server/discover",
            "params": _modern_meta_with_tasks(),
        },
        state,
    )
    task_reply = _dispatch(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "clay_scene", "arguments": {}, **_modern_meta_with_tasks()},
        },
        state,
        call_tool_task=lambda name, args: ("op-1", "working"),
    )
    _assert_valid(_EXT_TASKS, "CreateTaskResult", task_reply["result"])

    plain_reply = _dispatch(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "clay_scene", "arguments": {}, **_modern_meta()},
        },
        state,
        call_tool_task=lambda name, args: ("op-2", "working"),
    )
    result = plain_reply["result"]
    assert result["resultType"] == "complete"
    assert "taskId" not in result
    assert result["content"] == [{"type": "text", "text": "ran clay_scene"}]
