"""Regression tests for the 2026-09-23 audit's agents findings.

Each test here targets exactly one finding and is written to fail against
the tree as it stood before this fixer's changes:

* agents-01 -- ``realmspinner_status`` after a task-mode result has already
  been fetched via the ``status`` RPC op used to still claim it "is still
  waiting" (``op.fetched`` was set but ``op.delivered`` never was).
* agents-03 -- task-mode ``tools/call`` coerced a malformed, falsy
  ``arguments`` (``[]``/``0``/``""``) into ``{}`` instead of refusing with
  -32602, unlike the synchronous path and ``prompts/get``.

See "the 2026-09-23 audit, finding <id>" in each test's docstring rather
than this file's own name, per this repo's rule against citing audit/plan
file names from ``src/``.
"""

from __future__ import annotations

import json
import queue
from pathlib import Path
from types import SimpleNamespace

from realmspinner.mcp import protocol as p
from realmspinner.mcp import rpc
from realmspinner.studio import agent_host
from realmspinner.studio.modes.clay.agent import dispatch as agent_clay

# --- agents-01 ---------------------------------------------------------------


class _Ctx:
    def __init__(self) -> None:
        self.state = SimpleNamespace(clay=None)
        self.settings = SimpleNamespace()

    def toast(self, message: str, level: str = "info") -> None:
        pass

    def submit(self, key, fn, *args, tag=None, **kwargs) -> bool:
        import threading

        threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=True).start()
        return True


def _bare_host() -> agent_host.AgentHost:
    host = agent_host.AgentHost(_Ctx(), Path("unused-for-these-tests"))
    host._queue = queue.Queue()
    return host


def test_realmspinner_status_does_not_claim_a_fetched_task_mode_result_is_still_waiting() -> None:
    """The 2026-09-23 audit, finding agents-01.

    Once ``_task_status`` has handed a task-mode operation's terminal result
    out over the ``status`` RPC op, ``realmspinner_status`` must not still say
    "Its result is still waiting -- sending the same call again will hand it
    back" -- that call has already happened, and (per ``_call_task``'s own
    docstring) a task-mode resend mints a fresh operation and reruns the
    tool rather than replaying anything.
    """
    host = _bare_host()
    calls = agent_host._Calls()
    session = agent_clay.Session()

    header = host._call_task(session, calls, "clay_scene", {})
    op_id = header["operation_id"]
    host.pump(budget=1.0)

    # Fetch once via the RPC v1 `status` op -- what a real task-mode client
    # does for `tasks/get`. (An empty session refusing `clay_scene` with
    # isError=True is expected here and beside the point -- what matters is
    # that the fetch happened at all.)
    reply = host._task_status(calls, op_id)
    _status_header, body = rpc.split_reply(reply)
    assert json.loads(body).get("content")

    op = calls.get(op_id)
    assert op.fetched is True

    status_result = host._call(session, calls, agent_host.STATUS_TOOL, {"operation_id": op_id})
    payload = json.loads(status_result["content"][0]["text"])

    assert payload["delivered"] is True, (
        "a task-mode operation must be marked delivered once its result has "
        "gone out over the status RPC op, or the note below keeps firing"
    )
    assert "note" not in payload, (
        f"realmspinner_status still claims a fetched result 'is still waiting': {payload}"
    )


# --- agents-03 ---------------------------------------------------------------


def test_task_mode_tools_call_refuses_non_dict_arguments() -> None:
    """The 2026-09-23 audit, finding agents-03.

    ``arguments: []`` (a list, not an object) must be refused with -32602
    "'arguments' must be an object", the same as the synchronous
    ``tools/call`` path and ``prompts/get`` -- not silently coerced to
    ``{}`` by an ``or {}`` default and run anyway.
    """
    captured: dict = {}

    def call_tool_task(name, arguments):
        captured["arguments"] = arguments
        return ("op-1", "working")

    state = p.BridgeEra()
    item = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "clay_scene",
            "arguments": [],
            "_meta": {
                "io.modelcontextprotocol/protocolVersion": "2026-07-28",
                "io.modelcontextprotocol/clientCapabilities": {
                    "extensions": {"io.modelcontextprotocol/tasks": {}}
                },
            },
        },
    }

    reply = p._dispatch_one(
        item,
        state,
        catalogue={"tools": [], "server": {"name": "realmspinner", "version": "0.0.0"}},
        call_tool=lambda name, args: (_ for _ in ()).throw(AssertionError("sync path called")),
        call_tool_task=call_tool_task,
    )

    assert "arguments" not in captured, (
        f"call_tool_task was invoked with coerced arguments {captured.get('arguments')!r} "
        "instead of the request being refused"
    )
    decoded = json.loads(reply)
    assert decoded["error"]["code"] == -32602
    assert "arguments" in decoded["error"]["message"]
