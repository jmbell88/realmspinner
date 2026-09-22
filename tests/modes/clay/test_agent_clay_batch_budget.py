"""``clay_batch``'s wall-clock budget: the 2026-09-22 audit, finding clay-14.

Before this fix, ``_h_batch`` folded up to ``BATCH_MAX`` (32) calls -- any of
them a synchronous, subprocess-backed ``clay_op`` (decimate, retopo,
smart-unwrap, bake-detail; see ``_OpCtx.inline``'s own docstring) -- with no
deadline at all, unlike ``clay_program``'s own ``PROGRAM_DEADLINE_S``. This
module drives the fix through the real ``agent_clay.call`` dispatch path,
with ``tools_batch.BATCH_DEADLINE_S`` monkeypatched down to 0 so the second
entry in a batch is always past the deadline -- the same shape
``tests/mcp/test_rpc_studio.py`` uses to shrink ``PROGRAM_DEADLINE_S`` for a
program test, without needing a real slow subprocess in this suite.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from realmspinner.studio.modes.clay.agent import dispatch as agent_clay
from realmspinner.studio.modes.clay.agent import tools_batch


class _Ctx:
    def __init__(self) -> None:
        self.state = SimpleNamespace(clay=None)
        self.svc = None
        self.settings = SimpleNamespace()
        self.toasts: list[tuple[str, str]] = []

    def toast(self, message: str, level: str = "info") -> None:
        self.toasts.append((message, level))


def _payload(result: dict) -> Any:
    return json.loads(result["content"][0]["text"])


def test_clay_batch_refuses_or_bounds_several_synchronous_subprocess_ops_folded_into_one_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the budget forced to 0, a batch of several calls stops partway
    through rather than running unboundedly -- the deadline is checked
    between entries, so the first always runs (this is not a "batches of one
    are useless" refusal) and a later one is refused by name."""
    monkeypatch.setattr(tools_batch, "BATCH_DEADLINE_S", 0.0)

    ctx = _Ctx()
    session = agent_clay.Session()
    calls = [
        {"name": "clay_add_primitive", "arguments": {"generator": "cylinder"}},
        {"name": "clay_add_primitive", "arguments": {"generator": "cone"}},
        {"name": "clay_add_primitive", "arguments": {"generator": "box"}},
    ]
    result = agent_clay.call(ctx, session, "clay_batch", {"calls": calls})

    assert result["isError"] is True
    payload = _payload(result)
    # The first entry always runs regardless of the budget -- see
    # tools_batch._make_entry's own "index > 0" guard.
    assert payload["completed"] == 1
    assert payload["stopped_at"] == 1
    assert "deadline" in payload["results"][1]["content"][0]["text"]
