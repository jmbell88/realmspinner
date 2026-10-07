"""Regression tests for the 2026-10-03 audit's Low findings agents-15..38 (the
agents/ segment: Clay's tool door, the MCP host, the protocol, the character
surface and the prompts), fixer ``agents1``.

One file for all of them although they span ``tests/modes/clay``,
``tests/mcp`` and ``tests/studio``: the fixer rules ask for a single new test
file per agent.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from realmspinner.mcp import pipe, protocol, rpc
from realmspinner.studio import agent_character as ac
from realmspinner.studio import agent_host, agent_prompts
from realmspinner.studio.modes.clay.agent import dispatch as agent_clay

pytestmark = pytest.mark.filterwarnings("ignore")

WAIT = 5.0


class _Ctx:
    """The minimal ``ctx`` double ``tests/modes/clay/test_agent_clay.py`` uses."""

    def __init__(self) -> None:
        self.state = SimpleNamespace(clay=None)
        self.settings = SimpleNamespace()
        self.toasts: list[tuple[str, str]] = []

    def toast(self, message: str, level: str = "info") -> None:
        self.toasts.append((message, level))

    def submit(self, key: str, fn: Any, *args: Any, tag: Any = None, **kwargs: Any) -> bool:
        threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=True).start()
        return True


def _box(ctx: _Ctx, session: agent_clay.Session) -> int:
    result = agent_clay.call(ctx, session, "clay_add_primitive", {"generator": "box"})
    assert result["isError"] is False, result
    return json.loads(result["content"][0]["text"])["uid"]


# --- agents-15: an unrepresentable integer is refused by field ------------------


@pytest.mark.parametrize("bad", [2**70, 1e30, float("inf")])
def test_an_infinite_number_for_an_integer_argument_is_refused_by_field_not_by_the_backstop(
    bad: Any,
) -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _box(ctx, session)
    cases = [
        ("clay_select_elements", {"uid": uid, "mode": "edge", "edges": [[bad, 1]]}, "edges"),
        ("clay_select_elements", {"uid": uid, "mode": "vertex", "verts": [bad]}, "verts"),
        ("clay_select_elements", {"uid": uid, "mode": "face", "faces": [bad]}, "faces"),
        ("clay_select_elements", {"uid": uid, "mode": "vertex", "expect_stamp": bad},
         "expect_stamp"),
        ("clay_uv", {"uid": uid, "action": "mark_seam", "edges": [[bad, 1]]}, "edges"),
    ]
    for tool, args, field in cases:
        result = agent_clay.call(ctx, session, tool, args)
        assert result["isError"] is True, (tool, args)
        text = result["content"][0]["text"]
        assert "failed unexpectedly" not in text, (tool, args, text)
        assert result["structuredContent"]["field"] == field, (tool, args, result)


# --- agents-20 / agents-31: prompts ---------------------------------------------


@pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
def test_a_required_prompt_argument_that_is_blank_is_reported_missing(blank: str) -> None:
    assert agent_prompts.render("model_from_description", {"description": blank}) == [
        "description"
    ]
    assert agent_prompts.render("model_from_reference", {"job_id": blank}) == ["job_id"]
    # An optional blank falls back to its own default instead of rendering "".
    _desc, messages = agent_prompts.render(
        "character_sheets_from_description", {"description": "a knight", "movements": " "}
    )
    assert "movements omitted" in messages[0]["content"]["text"]


def test_character_sheets_prompt_renders_movements_in_the_shape_character_create_accepts() -> None:
    _desc, messages = agent_prompts.render(
        "character_sheets_from_description",
        {"description": "a knight", "movements": "walk, idle"},
    )
    text = messages[0]["content"]["text"]
    assert '{"name": "walk"}' in text and '{"name": "idle"}' in text
    assert "[walk, idle]" not in text


# --- agents-21: no stale generator count in anything published -------------------


def test_no_published_tool_description_states_a_stale_generator_count() -> None:
    import re

    from realmspinner.kernels.mesh import primitives

    words = {
        "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
        "twenty": 20, "twenty-one": 21, "twenty-two": 22,
    }  # fmt: skip
    pattern = re.compile(r"\b(" + "|".join(words) + r"|\d+)\s+(?:primitive\s+)?generators\b")
    real = len(primitives.GENERATORS)
    published = [agent_clay.instructions()]
    for tool in agent_clay.tools():
        published.append(tool.description)
        published.append(json.dumps(tool.schema))
    for text in published:
        for match in pattern.finditer(text):
            token = match.group(1)
            claimed = words.get(token) or int(token)
            assert claimed == real, f"{match.group(0)!r} but the registry holds {real}"


# --- agents-22: busy_tools -------------------------------------------------------


def test_busy_tools_does_not_name_a_dropped_job() -> None:
    host = agent_host.AgentHost(_Ctx(), Path("unused-for-these-tests"))
    live = agent_host._Job(lambda: None, tool="character_export")
    live.state = agent_host.RUNNING
    queued = agent_host._Job(lambda: None, tool="character_rig")
    dropped = agent_host._Job(lambda: None, tool="character_cancel")
    dropped.state = agent_host.DROPPED
    host._service_jobs = {id(j): j for j in (live, queued, dropped)}
    assert host.busy_tools == ("character_export", "character_rig")


# --- agents-23: realmspinner_status refuses what it does not take ----------------


def test_realmspinner_status_refuses_an_unknown_argument_name() -> None:
    host = agent_host.AgentHost(_Ctx(), Path("unused-for-these-tests"))
    calls = agent_host._Calls()
    result = host._status(calls, {"operationid": "abc"})
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "operationid"
    blank = host._status(calls, {"operation_id": ""})
    assert blank["isError"] is True
    assert blank["structuredContent"]["field"] == "operation_id"
    # The two legal shapes still answer.
    assert host._status(calls, {}).get("isError") is not True


# --- agents-25 / agents-26: protocol ---------------------------------------------


def _bridge(item: dict, state: protocol.BridgeEra | None = None) -> tuple[Any, Any]:
    state = state or protocol.BridgeEra()
    raw = protocol.bridge_dispatch(
        json.dumps(item).encode("utf-8"),
        state,
        catalogue={"tools": [], "instructions": ""},
        call_tool=lambda name, args: b"{}",
    )
    return raw, state


def _strict_loads(raw: bytes) -> Any:
    def refuse(token: str) -> Any:
        raise AssertionError(f"bare {token} on the wire")

    return json.loads(raw, parse_constant=refuse)


def test_an_error_that_echoes_a_nan_protocol_version_or_task_id_is_still_valid_json() -> None:
    raw, _ = _bridge(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {"_meta": {protocol.MODERN_META_KEY: float("nan")}},
        }
    )
    reply = _strict_loads(raw)
    assert reply["error"]["code"] == -32022

    state = protocol.BridgeEra()
    state.era = "modern"
    raw, _ = _bridge(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tasks/update",
            "params": {
                "taskId": float("inf"),
                "_meta": {protocol.MODERN_META_KEY: protocol.MODERN[0]},
            },
        },
        state,
    )
    assert _strict_loads(raw)["error"]["code"] == -32602


def test_an_unsupported_version_probe_does_not_lock_the_era_before_initialize() -> None:
    state = protocol.BridgeEra()
    raw, state = _bridge(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {"_meta": {protocol.MODERN_META_KEY: "2099-01-01"}},
        },
        state,
    )
    reply = json.loads(raw)
    assert reply["error"]["code"] == -32022
    # The reply tells the client the legacy handshake exists.
    assert list(protocol.LEGACY) == reply["error"]["data"]["legacy"]
    assert state.era is None
    raw, state = _bridge(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "initialize",
            "params": {
                "protocolVersion": protocol.LEGACY[0],
                "capabilities": {},
                "clientInfo": {"name": "t", "version": "0"},
            },
        },
        state,
    )
    reply = json.loads(raw)
    assert "error" not in reply, reply
    assert state.era == "legacy"


def test_a_notification_naming_a_modern_version_does_not_lock_the_era() -> None:
    state = protocol.BridgeEra()
    raw, state = _bridge(
        {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {"_meta": {protocol.MODERN_META_KEY: protocol.MODERN[0]}},
        },
        state,
    )
    assert raw is None
    assert state.era is None


# --- agents-27 / agents-29: character prose ---------------------------------------


def test_the_polling_instructions_name_every_terminal_status() -> None:
    for text in (
        ac.instructions(),
        agent_prompts.render(
            "character_sheets_from_description", {"description": "a knight"}
        )[1][0]["content"]["text"],
    ):
        assert "cancelled" in text, text
        assert "error" in text, text


def test_the_character_instructions_do_not_say_the_call_blocks_on_the_rig() -> None:
    text = ac.instructions()
    assert "subprocess the call blocks on" not in text
    assert "returns as soon as" in text or "return as soon as" in text


def test_character_options_reports_the_size_range_the_schema_accepts(svc: Any) -> None:
    result = ac.call(svc, ac.Session(), "character_options", {})
    assert result["isError"] is False
    payload = result["structuredContent"]
    assert payload["size_range"] == list(ac._enums().size_range)


# --- agents-30: a movement's frames refusal names the entry and the key ------------


def test_a_bad_movement_frames_refusal_names_frames_and_the_entry(svc: Any) -> None:
    names = list(ac._enums().movements)
    result = ac.call(
        svc,
        ac.Session(),
        "character_create",
        {
            "prompt": "a knight",
            "movements": [{"name": names[0]}, {"name": names[1], "frames": 100000}],
        },
    )
    assert result["isError"]
    assert result["structuredContent"]["field"] == "movements"
    assert "movements[1].frames" in result["content"][0]["text"]
    result = ac.call(
        svc,
        ac.Session(),
        "character_create",
        {"prompt": "a knight", "movements": [{"name": names[0], "frames": "x"}]},
    )
    assert "movements[0].frames must be a number" in result["content"][0]["text"]


# --- agents-28: follow_up_sheet_job -------------------------------------------------


def _mesh(svc: Any) -> str:
    """A finished character mesh row, minted directly -- no Blender, no queue."""
    params = {"asset_type": "character", "family": "ogre", "built": True}
    return svc.store.create("image", "an ogre", params, stage="model", status="done")


def _rig_row(svc: Any, mesh_id: str, *, troupe_sheet: dict, status: str) -> str:
    params = {
        "source_job": mesh_id,
        "template": "humanoid",
        "auto": True,
        "troupe_sheet": troupe_sheet,
    }
    return svc.store.create("rig", "a hooded ranger", params, stage="model", status=status)


def _charsheet_row(svc: Any, mesh_id: str, *, sheet_id: str, extra: dict) -> str:
    params = {"source_job": mesh_id, "sheet_id": sheet_id, **extra}
    return svc.store.create("charsheet", "a hooded ranger", params, stage="model", status="queued")


def _touch(svc: Any, job_id: str, column: str, value: float) -> None:
    svc.store._conn.execute(f"UPDATE jobs SET {column} = ? WHERE id = ?", (value, job_id))
    svc.store._conn.commit()


def _touch_created_at(svc: Any, job_id: str, value: float) -> None:
    _touch(svc, job_id, "created_at", value)


def _touch_finished_at(svc: Any, job_id: str, value: float) -> None:
    _touch(svc, job_id, "finished_at", value)



def test_a_humans_identical_sheet_is_not_the_follow_up_when_the_real_one_failed_to_queue(
    svc: Any,
) -> None:
    from realmspinner import followups
    from realmspinner.kernels.rig import store
    from realmspinner.service import troupe as svc_troupe

    block = {"template": "humanoid", "logical_size": 32}

    # (a) A human's sheet made *before* the rig finished is never its follow-up.
    mesh_id = _mesh(svc)
    rig_id = _rig_row(svc, mesh_id, troupe_sheet=block, status="done")
    _touch_created_at(svc, rig_id, 900.0)
    _touch_finished_at(svc, rig_id, 1000.0)
    early = _charsheet_row(svc, mesh_id, sheet_id=store.new_id(), extra=dict(block))
    _touch_created_at(svc, early, 990.0)
    assert svc_troupe.follow_up_sheet_job(svc, rig_id) is None

    # (b) The real follow-up failed to queue (a failure is on the mesh), and a
    # human pressed the same settings seconds later: it is theirs, not the rig's.
    mesh_id = _mesh(svc)
    rig_id = _rig_row(svc, mesh_id, troupe_sheet=block, status="done")
    _touch_created_at(svc, rig_id, 1900.0)
    _touch_finished_at(svc, rig_id, 2000.0)
    svc.store.merge_param_entry(
        mesh_id,
        followups.PARAM_KEY,
        "charsheet",
        followups.failure_record("charsheet", "x", recorded_at=2000.5),
    )
    human = _charsheet_row(svc, mesh_id, sheet_id=store.new_id(), extra=dict(block))
    _touch_created_at(svc, human, 2010.0)
    assert svc_troupe.follow_up_sheet_job(svc, rig_id) is None

    # (c) Without a recorded failure the same row IS the follow-up.
    mesh_id = _mesh(svc)
    rig_id = _rig_row(svc, mesh_id, troupe_sheet=block, status="done")
    _touch_created_at(svc, rig_id, 2900.0)
    _touch_finished_at(svc, rig_id, 3000.0)
    real = _charsheet_row(svc, mesh_id, sheet_id=store.new_id(), extra=dict(block))
    _touch_created_at(svc, real, 3000.2)
    assert svc_troupe.follow_up_sheet_job(svc, rig_id) == real


# --- agents-35: a silent second connection ------------------------------------------


def _closed(conn: Any) -> bool:
    try:
        if conn.poll(0.05):
            conn.recv_bytes()
        return False
    except (EOFError, OSError):
        return True


def test_a_silent_second_connection_is_closed_after_a_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(agent_host, "BUSY_FIRST_FRAME_TIMEOUT", 0.3)
    host = agent_host.AgentHost(_Ctx(), tmp_path)
    assert host.start()
    stop_pumping = threading.Event()

    def pump_loop() -> None:
        while not stop_pumping.is_set():
            host.pump(budget=0.01)
            time.sleep(0.005)

    pumper = threading.Thread(target=pump_loop, daemon=True)
    pumper.start()
    try:
        first = pipe.connect(tmp_path)
        try:
            first.send_bytes(rpc.encode_request("hello", versions=[1], bridge_version="t"))
            assert first.poll(WAIT)
            first.recv_bytes()
            silent = pipe.connect(tmp_path)
            try:
                deadline = time.monotonic() + WAIT
                while time.monotonic() < deadline and not _closed(silent):
                    time.sleep(0.05)
                assert _closed(silent), "a peer that never sends a frame must be dropped"
            finally:
                with contextlib.suppress(OSError):
                    silent.close()
        finally:
            first.close()
    finally:
        stop_pumping.set()
        pumper.join(timeout=WAIT)
        host.stop()


def test_concurrent_busy_refusals_are_capped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = agent_host.AgentHost(_Ctx(), tmp_path)
    host._connected = True
    host._busy_refusals = agent_host.MAX_BUSY_REFUSALS
    closed: list[bool] = []

    class _Conn:
        def recv_bytes(self, maxlength: int | None = None) -> bytes:  # pragma: no cover
            raise AssertionError("an over-cap connection must not be waited on")

        def poll(self, timeout: float = 0.0) -> bool:  # pragma: no cover
            raise AssertionError("an over-cap connection must not be waited on")

        def close(self) -> None:
            closed.append(True)

    host._admit(_Conn())
    assert closed == [True]


# --- agents-37: pipe docstring --------------------------------------------------------


def test_pipe_docstring_describes_the_busy_refusal_not_a_wait() -> None:
    doc = pipe.__doc__ or ""
    assert "simply waits" not in doc
    assert "busy" in doc
    assert "one connection at a time" not in (pipe.Server.__doc__ or "").lower()
    assert "dispatch()" not in (protocol.__doc__ or "") or "gone" in (protocol.__doc__ or "")


# --- agents-38: a non-JSON value in a result does not drop the connection ----------


def test_a_result_with_a_non_json_value_is_refused_not_a_dropped_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    host = agent_host.AgentHost(_Ctx(), Path("unused-for-these-tests"))
    session = agent_clay.Session()
    calls = agent_host._Calls()
    hostile = {
        "content": [{"type": "text", "text": "ok"}],
        "structuredContent": {"n": np.int64(3), "path": Path("x"), "s": {1, 2}},
    }
    monkeypatch.setattr(host, "_call", lambda *a, **k: hostile)
    frame = rpc.encode_request("call", tool="clay_scene", args={})
    reply = host._serve_rpc_frame(session, calls, frame)
    header, body = rpc.split_reply(reply)
    assert "error" not in header
    parsed = json.loads(body)
    assert parsed["content"][0]["text"] == "ok"
    assert parsed["structuredContent"]["n"] == "3"


def test_a_representative_run_of_clay_tools_returns_strictly_serialisable_results() -> None:
    """The evidence gap behind agents-38: nothing asserted that a tool's whole
    result (``structuredContent`` included) survives a strict ``json.dumps``,
    so a payload slip would only ever show up as a dropped connection. A
    sweep over a working run of the common tools -- not every tool, which
    need arguments only a real scene can supply -- with no ``default=``."""
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _box(ctx, session)
    steps = [
        ("clay_scene", {}),
        ("clay_transform", {"uid": uid, "translation": [0, 1, 0]}),
        ("clay_set_params", {"uid": uid, "params": {"size": [1, 2, 3]}}),
        ("clay_material", {"uids": [uid], "color": [0.2, 0.4, 0.6]}),
        ("clay_element_mode", {"mode": "face"}),
        ("clay_select_elements", {"uid": uid, "faces": [0, 1]}),
        ("clay_elements", {"uid": uid}),
        ("clay_catalog", {"topic": "ops"}),
        ("clay_undo", {}),
    ]
    for tool, args in steps:
        result = agent_clay.call(ctx, session, tool, args)
        json.dumps(result, allow_nan=False)  # TypeError/ValueError is the failure
        failed = result.get("isError") and tool != "clay_undo"
        assert not failed, (tool, result["content"][0].get("text"))
