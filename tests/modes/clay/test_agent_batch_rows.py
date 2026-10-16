"""The shape of one row in ``clay_batch``'s ``results`` and in ``clay_program``'s
``failure`` (finding car-04).

Before, a row was the nested call's WHOLE MCP result -- ``{"content": [{"type":
"text", "text": "<the same json again>"}], "isError": ..., "structuredContent":
{...}}`` -- and the batch envelope was then sent as text and again as
structuredContent, so every field of every row travelled four times. A 26-box
batch measured 28,981 bytes of structuredContent and 61,004 bytes on the wire
before this change, most of it the duplicate and noise fields (bbox, verts, uv,
stamp, selected, default rotation/scale) of a row the agent had just made.

A row is now the structured payload once; a mutation tool's row also drops its
noise; a refusal row is ``{"error": <sentence>, ...extras}``. The uid keys stay,
because ``transcript.produced_uids`` and the recorded fixtures count them.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from realmspinner.studio.modes.clay.agent import dispatch as agent_clay
from realmspinner.studio.modes.clay.agent import transcript

# Measured against the unchanged code, 26 boxes at translation [i, 0.5, 0]:
OLD_STRUCTURED_BYTES = 28_981
OLD_WIRE_BYTES = 61_004


class _Ctx:
    def __init__(self) -> None:
        self.state = SimpleNamespace(clay=None)
        self.svc = None
        self.settings = SimpleNamespace()
        self.toasts: list[tuple[str, str]] = []

    def toast(self, message: str, level: str = "info") -> None:
        self.toasts.append((message, level))


def _batch(calls: list[dict]) -> tuple[dict, dict]:
    result = agent_clay.call(_Ctx(), agent_clay.Session(), "clay_batch", {"calls": calls})
    return result, result["structuredContent"]


def _add(generator: str = "box", **arguments: Any) -> dict:
    return {"name": "clay_add_primitive", "arguments": {"generator": generator, **arguments}}


def test_a_batch_row_is_one_copy_with_no_content_wrapper() -> None:
    _result, payload = _batch([_add(), _add()])
    for row in payload["results"]:
        assert "content" not in row
        assert "structuredContent" not in row
        assert "isError" not in row
        assert isinstance(row["uid"], int)
    # The text twin of the envelope parses to the same rows.
    assert payload["completed"] == 2


def test_a_26_add_batch_reply_is_under_half_its_old_size() -> None:
    calls = [_add(translation=[i, 0.5, 0]) for i in range(26)]
    result, payload = _batch(calls)
    assert len(payload["results"]) == 26
    structured = len(json.dumps(payload))
    wire = len(json.dumps(result, separators=(",", ":")))
    assert structured < OLD_STRUCTURED_BYTES / 2, structured
    assert wire < OLD_WIRE_BYTES / 2, wire


def test_a_mutation_row_drops_bbox_verts_uv_stamp_and_selected() -> None:
    _result, payload = _batch([_add()])
    row = payload["results"][0]
    for noisy in ("bbox", "verts", "uv", "stamp", "selected", "bone", "visible", "parent"):
        assert noisy not in row, noisy
    # Defaults are implied.
    assert "rotation" not in row
    assert "scale" not in row
    for kept in ("uid", "name", "generator", "params", "translation", "size", "faces", "material"):
        assert kept in row, kept


def test_a_mutation_row_keeps_a_non_default_rotation_and_scale() -> None:
    _result, payload = _batch([_add(rotation=[0, 45, 0], scale=[2, 2, 2])])
    row = payload["results"][0]
    assert row["rotation"] == [0.0, 45.0, 0.0]
    assert row["scale"] == [2.0, 2.0, 2.0]


def test_a_read_row_keeps_every_field() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    result = agent_clay.call(
        ctx, session, "clay_batch", {"calls": [_add(), {"name": "clay_scene", "arguments": {}}]}
    )
    row = result["structuredContent"]["results"][1]
    direct = agent_clay.call(ctx, session, "clay_scene", {})["structuredContent"]
    assert "content" not in row
    assert row == direct
    assert "bbox" in row["objects"][0]
    assert "stamp" in row["objects"][0]


def test_a_refused_batch_row_carries_error_field_and_recovery() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = agent_clay.call(ctx, session, "clay_add_primitive", {"generator": "box"})[
        "structuredContent"
    ]["uid"]
    result = agent_clay.call(
        ctx,
        session,
        "clay_batch",
        {
            "calls": [
                _add(),
                {"name": "clay_set_params", "arguments": {"uid": uid, "params": {"nope": 1}}},
            ]
        },
    )
    payload = result["structuredContent"]
    assert result["isError"] is True
    assert payload["stopped_at"] == 1
    row = payload["results"][1]
    assert isinstance(row["error"], str) and row["error"]
    assert row["field"] == "params"
    assert row["recovery"] == "fix_arguments"
    assert row["changed"] is False
    assert "content" not in row
    assert "isError" not in row


def test_uids_in_a_slim_batch_equal_what_produced_uids_counted_before() -> None:
    result, payload = _batch([_add(), _add("cylinder"), _add("cone")])
    made = [row["uid"] for row in payload["results"]]
    assert len(made) == 3
    assert transcript.produced_uids(result) == made


def test_a_refused_row_keeps_its_uids_for_produced_uids() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    made = agent_clay.call(ctx, session, "clay_add_primitive", {"generator": "box"})
    uid = made["structuredContent"]["uid"]
    result = agent_clay.call(
        ctx,
        session,
        "clay_batch",
        {"calls": [{"name": "clay_set_params", "arguments": {"uid": uid, "params": {"nope": 1}}}]},
    )
    row = result["structuredContent"]["results"][0]
    assert row.get("uids") == [uid]
    assert transcript.produced_uids(result) == [uid]


def test_familiars_refusal_sentence_reads_the_failing_rows_error() -> None:
    from realmspinner.studio.assistant import ui as assistant_ui

    result, payload = _batch(
        [_add(), {"name": "clay_transform", "arguments": {"uid": 999999, "translation": [1, 0, 0]}}]
    )
    assert assistant_ui._refusal_sentence(result) == payload["results"][1]["error"]
    assert "999999" in assistant_ui._refusal_sentence(result)


def test_clay_program_failure_row_is_slim() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    program = {
        "steps": [
            {"add": {"generator": "box", "id": "c"}},
            {"transform": {"uid": 999999, "translation": [1.0, 0.0, 0.0]}},
        ]
    }
    result = agent_clay.call(ctx, session, "clay_program", program)
    assert result["isError"] is True
    failure = result["structuredContent"]["failure"]
    assert "content" not in failure
    assert "isError" not in failure
    assert "999999" in failure["error"]
    assert failure["field"] == "uid"
