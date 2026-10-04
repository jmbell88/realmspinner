"""Regressions for the 2026-10-03 audit's Low findings, fixer ``agents2``.

agents-16: a list or object where ``clay_batch``'s ``calls[].name`` belongs raised an
unhashable ``TypeError`` at the ``in allowed`` set test.
agents-17: evidence for the reference doors (the fix itself landed with clay-81).
agents-18: a repeat range's list was evaluated element by element before its length
was compared with ``PROGRAM_MAX_REPEAT``.
agents-19: a group with a consumed member compiled and failed only at run time.
agents-34: the scene resource read every ``clay_scene`` refusal as "not found".
agents-36: a reconnect whose handshake or catalogue re-fetch failed kept the connection.

The agents-24 flake lives beside the test it fixes, in ``tests/studio/test_agent_host.py``.
"""

from __future__ import annotations

import base64

import pytest

from realmspinner.mcp import bridge, rpc
from realmspinner.studio import agent_resources
from realmspinner.studio.modes.clay.agent import dispatch as agent_clay
from realmspinner.studio.modes.clay.agent import program as ap

from .test_agent_clay import _Ctx, _new_agent_tab


def _backstop(result: dict) -> bool:
    return "failed unexpectedly" in result["content"][0]["text"]


# --- agents-16 / agents-17 ----------------------------------------------------


def test_a_non_string_name_is_refused_by_field_at_batch_and_reference_get_and_remove() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session, "box")

    for bad in (["clay_scene"], {"a": 1}):
        result = agent_clay.call(ctx, session, "clay_batch", {"calls": [{"name": bad}]})
        assert not _backstop(result), result
        assert result["isError"] is True, result
        assert result["structuredContent"].get("field") == "calls", result
        for tool in ("clay_reference_get", "clay_reference_remove"):
            result = agent_clay.call(ctx, session, tool, {"name": bad})
            assert not _backstop(result), (tool, result)
            assert result["structuredContent"].get("field") == "name", (tool, result)


def test_reference_add_refuses_bytes_that_are_not_an_image_by_field() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session, "box")
    for payload in ("", base64.b64encode(b"definitely not an image").decode("ascii")):
        result = agent_clay.call(
            ctx, session, "clay_reference_add", {"name": "r", "png_base64": payload}
        )
        assert not _backstop(result), result
        assert result["isError"] is True, result
        assert result["structuredContent"].get("field") == "png_base64", result


# --- agents-18 / agents-19 ----------------------------------------------------


def test_a_range_list_over_the_repeat_cap_is_refused_before_any_element_is_evaluated(
    monkeypatch,
) -> None:
    evaluated: list[object] = []
    real_num = ap._num

    def counting(value, *args, **kwargs):
        evaluated.append(value)
        return real_num(value, *args, **kwargs)

    monkeypatch.setattr(ap, "_num", counting)
    program = {
        "steps": [
            {
                "repeat": {
                    "ranges": {"i": {"list": ["1+1"] * (ap.PROGRAM_MAX_REPEAT + 1)}},
                    "steps": [{"add": {"generator": "box"}}],
                }
            }
        ]
    }
    with pytest.raises(ap.ProgramError) as exc_info:
        ap.compile_program(program)
    assert exc_info.value.field == "steps"
    assert "PROGRAM_MAX_REPEAT" in exc_info.value.reason
    assert evaluated == [], "an element was evaluated before the length was checked"


def test_a_group_with_a_consumed_member_is_refused_at_compile_time() -> None:
    program = {
        "steps": [
            {"add": {"generator": "box", "id": "a"}},
            {"add": {"generator": "box", "id": "b"}},
            {"group": {"id": "g", "members": ["a", "b"]}},
            {"delete": {"uids": ["b"]}},
            {"material": {"uids": "g", "color": [1, 0, 0]}},
        ]
    }
    with pytest.raises(ap.ProgramError) as exc_info:
        ap.compile_program(program)
    assert "'b'" in exc_info.value.reason
    assert "consumed by a delete" in exc_info.value.reason
    assert exc_info.value.field == "steps"


def test_a_group_whose_members_are_all_alive_still_resolves() -> None:
    program = {
        "steps": [
            {"add": {"generator": "box", "id": "a"}},
            {"add": {"generator": "box", "id": "b"}},
            {"group": {"id": "g", "members": ["a", "b"]}},
            {"material": {"uids": "g", "color": [1, 0, 0]}},
        ]
    }
    assert ap.compile_program(program).calls[-1][1]["uids"] == [{"$ref": "a"}, {"$ref": "b"}]


# --- agents-34 ----------------------------------------------------------------


def test_scene_resource_names_an_over_budget_refusal_instead_of_not_found(monkeypatch) -> None:
    refusal = {
        "content": [{"type": "text", "text": "This scene is too large to send in one frame."}],
        "isError": True,
        "structuredContent": {"changed": False},
    }
    monkeypatch.setattr(agent_clay, "call", lambda *a, **kw: refusal)
    answered = agent_resources.read_dynamic(object(), object(), agent_resources.SCENE_URI)
    assert answered is not None, "an over-budget scene read came back as not_found"
    mime, body = answered
    assert mime == "application/json"
    assert b"too large" in body


def test_scene_resource_still_reads_no_document_as_not_found() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    assert agent_resources.read_dynamic(ctx, session, agent_resources.SCENE_URI) is None


# --- agents-36 ----------------------------------------------------------------


class _ScriptedConn:
    """A duck-typed connection answering ``hello`` (and then, optionally,
    failing the ``catalogue`` send), recording whether it was closed."""

    def __init__(self, *, hello_raises: bool = False, catalogue_send_raises: bool = False):
        self.hello_raises = hello_raises
        self.catalogue_send_raises = catalogue_send_raises
        self.closed = False
        self.sent: list[bytes] = []

    def send_bytes(self, data: bytes) -> None:
        self.sent.append(data)
        if self.catalogue_send_raises and len(self.sent) > 1:
            raise OSError("the pipe broke")

    def poll(self, timeout: float | None = None) -> bool:
        return True

    def recv_bytes(self, maxlength=None) -> bytes:
        if self.hello_raises:
            raise EOFError("Studio dropped the handshake")
        return rpc.encode_reply({"rpc": 1, "call_timeout": 5.0, "catalogue_hash": "moved"})

    def close(self) -> None:
        self.closed = True


def _session(monkeypatch, conn: _ScriptedConn) -> bridge._Session:
    session = bridge._Session(None, None, {"call_timeout": 5.0}, {"hash": "old"})
    monkeypatch.setattr(bridge, "_connect", lambda home: conn)
    return session


def test_ensure_connected_closes_the_connection_when_hello_raises(monkeypatch) -> None:
    conn = _ScriptedConn(hello_raises=True)
    session = _session(monkeypatch, conn)
    assert session._ensure_connected() is False
    assert session.conn is None
    assert conn.closed, "the half-open connection leaked a pipe instance"


def test_ensure_connected_drops_the_connection_when_the_catalogue_refresh_raises(
    monkeypatch,
) -> None:
    conn = _ScriptedConn(catalogue_send_raises=True)
    session = _session(monkeypatch, conn)
    assert session._ensure_connected() is False
    assert session.conn is None, "the half-dead connection was kept for the next call"
    assert conn.closed
