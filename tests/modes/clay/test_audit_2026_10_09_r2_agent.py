"""Regression tests for the 2026-10-09 second audit's Clay agent-door findings:
clay-71 (an idempotent paint is a no-op, not an error), clay-72 (an op that
deletes nothing is a refusal), clay-73 (every mode-gated op carries
``switch_mode``), clay-74 (``clay_delete`` closes its gesture and is
all-or-nothing), clay-130 (a hidden object cannot be selected), clay-131 (the
instructions name the tools that really want object mode), clay-132 (an
integer past float range), clay-133 (``clay_uv`` translate bound) and
clay-134 (arguments the chosen query/action does not take are refused).

Each test's name is the claim.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from realmspinner.kernels.mesh import select as bsel
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay import ops as clay_ops
from realmspinner.studio.modes.clay.agent import dispatch as agent_clay
from realmspinner.studio.modes.clay.agent import schema as agent_schema
from realmspinner.studio.modes.clay.agent.validate import UV_MAX

from .test_agent_clay import _Ctx, _new_agent_tab, _payload


def _doc(ctx: _Ctx, session: agent_clay.Session):
    return clay_mode.ensure(ctx).get(session.tab_uid).doc


def _text(result: dict) -> str:
    return result["content"][0]["text"]


def _call(ctx: _Ctx, session: agent_clay.Session, name: str, args: dict) -> dict:
    return agent_clay.call(ctx, session, name, args)


def _face_mode_with_faces(ctx: _Ctx, session: agent_clay.Session, uid: int, faces: list[int]):
    assert _call(ctx, session, "clay_select", {"uids": [uid]})["isError"] is False
    assert _call(ctx, session, "clay_element_mode", {"mode": "face"})["isError"] is False
    r = _call(ctx, session, "clay_select_elements", {"uid": uid, "faces": faces})
    assert r["isError"] is False, r


# --- clay-71 -----------------------------------------------------------------


def test_clay_op_assign_material_onto_the_same_slot_is_a_noop_and_does_not_stop_a_batch() -> None:
    ctx, session = _Ctx(), agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    _face_mode_with_faces(ctx, session, uid, [0, 1])

    result = _call(ctx, session, "clay_op", {"name": "assign-material", "params": {"index": 0}})
    assert result["isError"] is False, result
    payload = _payload(result)
    assert payload["ran"] is False
    assert payload["pushed"] is False
    assert any("already use" in m for m in payload["messages"])

    batch = _call(
        ctx,
        session,
        "clay_batch",
        {
            "rollback_on_error": True,
            "calls": [
                {
                    "name": "clay_op",
                    "arguments": {"name": "assign-material", "params": {"index": 0}},
                },
                {"name": "clay_element_mode", "arguments": {"mode": "object"}},
                {"name": "clay_rename", "arguments": {"uid": uid, "name": "after"}},
            ],
        },
    )
    body = json.loads(_text(batch))
    assert batch["isError"] is False, body
    assert body["stopped_at"] is None
    assert _doc(ctx, session).by_uid(uid).name == "after"


# --- clay-72 -----------------------------------------------------------------


def test_clay_op_delete_that_deletes_nothing_is_a_refusal_and_stops_a_batch() -> None:
    ctx, session = _Ctx(), agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    r = _call(ctx, session, "clay_select_elements", {"uid": uid, "mode": "vertex", "verts": [0]})
    assert r["isError"] is False, r

    refused = _call(ctx, session, "clay_op", {"name": "delete"})
    assert refused["isError"] is True, refused
    assert "face" in _text(refused).lower()
    assert refused["structuredContent"]["messages"], refused["structuredContent"]

    batch = _call(
        ctx,
        session,
        "clay_batch",
        {
            "rollback_on_error": True,
            "calls": [
                {"name": "clay_op", "arguments": {"name": "delete"}},
                {"name": "clay_element_mode", "arguments": {"mode": "object"}},
            ],
        },
    )
    body = json.loads(_text(batch))
    assert batch["isError"] is True
    assert body["stopped_at"] == 0
    assert _doc(ctx, session).element_mode == "vertex"


# --- clay-73 -----------------------------------------------------------------


@pytest.mark.parametrize("mode", ["object", "vertex", "edge", "face"])
def test_clay_op_mode_gated_ops_in_the_wrong_mode_all_carry_recovery_switch_mode(mode: str) -> None:
    ctx, session = _Ctx(), agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    assert _call(ctx, session, "clay_select", {"uids": [uid]})["isError"] is False
    if mode != "object":
        assert _call(ctx, session, "clay_element_mode", {"mode": mode})["isError"] is False
    doc = _doc(ctx, session)
    gated = [op for op in clay_ops.OPS if doc.element_mode not in op.modes]
    assert gated, mode
    for op in gated:
        result = _call(ctx, session, "clay_op", {"name": op.name})
        assert result["isError"] is True, (mode, op.name, result)
        assert result["structuredContent"].get("recovery") == "switch_mode", (
            mode,
            op.name,
            result["structuredContent"],
        )
        assert "mode first" in _text(result), (mode, op.name, _text(result))


# --- clay-74 -----------------------------------------------------------------


def test_clay_delete_refusal_on_a_later_uid_closes_its_gesture_and_reports_changed_truthfully() -> (
    None
):
    ctx, session = _Ctx(), agent_clay.Session()
    solo = _new_agent_tab(ctx, session, "box")
    doc = _doc(ctx, session)
    # A 54-deep chain of boxes at scale 1e-6, each under the last with the
    # world placement kept: deleting a link deep enough re-places its child
    # at a non-finite local scale, which ``remove_object`` refuses.
    uids: list[int] = []
    for _ in range(54):
        r = _call(ctx, session, "clay_add_primitive", {"generator": "box", "scale": [1e-6] * 3})
        uids.append(_payload(r)["uid"])
    for parent, child in zip(uids, uids[1:], strict=False):
        r = _call(
            ctx, session, "clay_parent", {"uid": child, "parent": parent, "keep_world": False}
        )
        assert r["isError"] is False, r

    refused = None
    for k in range(1, 53):
        before = len(doc.objects)
        head = doc.history.head
        r = _call(ctx, session, "clay_delete", {"uids": [solo, uids[k]]})
        if r["isError"]:
            refused = r
            break
        _call(ctx, session, "clay_undo", {})
    assert refused is not None, "no deletion in the chain was refused"

    assert refused["structuredContent"]["field"] == "uids"
    assert refused["structuredContent"]["changed"] is False
    # The gesture is closed: an open one switches undo eviction off for good.
    assert doc.history._open_gestures == 0
    # All-or-nothing: the uid before the refusing one is still there.
    assert len(doc.objects) == before
    assert any(o.uid == solo for o in doc.objects)
    assert doc.history.head == head


# --- clay-130 ----------------------------------------------------------------


def test_clay_select_and_select_elements_refuse_a_hidden_object_by_name() -> None:
    ctx, session = _Ctx(), agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    doc = _doc(ctx, session)
    doc.set_visibility({uid: False})

    r = _call(ctx, session, "clay_select", {"uids": [uid]})
    assert r["isError"] is True, r
    assert r["structuredContent"]["field"] == "uids"
    assert "hidden" in _text(r)
    assert str(uid) in _text(r)

    r = _call(ctx, session, "clay_select_elements", {"uid": uid, "mode": "face", "faces": [0, 1]})
    assert r["isError"] is True, r
    assert r["structuredContent"]["field"] == "uid"
    assert "hidden" in _text(r)
    # Refused before anything switched: the document is still in object mode.
    assert doc.element_mode == "object"


def test_clay_select_by_refuses_a_hidden_object_by_name() -> None:
    ctx, session = _Ctx(), agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    doc = _doc(ctx, session)
    other = _payload(
        _call(ctx, session, "clay_add_primitive", {"generator": "box", "translation": [3, 0, 0]})
    )["uid"]
    assert _call(ctx, session, "clay_select", {"uids": [other]})["isError"] is False
    assert _call(ctx, session, "clay_element_mode", {"mode": "face"})["isError"] is False
    doc.set_visibility({uid: False})
    r = _call(ctx, session, "clay_select_by", {"uid": uid, "query": "material", "slot": 0})
    assert r["isError"] is True, r
    assert r["structuredContent"]["field"] == "uid"
    assert "hidden" in _text(r)


# --- clay-131 ----------------------------------------------------------------


def test_instructions_claim_object_mode_only_for_the_tools_that_actually_refuse() -> None:
    text = agent_clay.instructions()
    assert "every tool that addresses a whole object" not in text
    assert "clay_select, and clay_program, want object mode" in text

    ctx, session = _Ctx(), agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    assert _call(ctx, session, "clay_element_mode", {"mode": "face"})["isError"] is False
    # The sentence is true of the two it names...
    assert _call(ctx, session, "clay_select", {"uids": [uid]})["isError"] is True
    program = _call(
        ctx, session, "clay_program", {"steps": [{"add": {"generator": "box", "id": "a"}}]}
    )
    assert program["isError"] is True, program
    # ...and it must not be widened to the tools that work in any element mode.
    for name, args in [
        ("clay_transform", {"uid": uid, "translation": [1, 0, 0]}),
        ("clay_rename", {"uid": uid, "name": "x"}),
        ("clay_delete", {"uids": [uid]}),
    ]:
        assert _call(ctx, session, name, args)["isError"] is False, name


# --- clay-132 ----------------------------------------------------------------


def test_clay_program_refuses_an_integer_past_float_range_by_field() -> None:
    ctx, session = _Ctx(), agent_clay.Session()
    big = 10**400
    cases = [
        (
            {"steps": [{"add": {"generator": "box", "id": "a", "translation": [big, 0, 0]}}]},
            "steps",
        ),
        (
            {"variables": {"x": big}, "steps": [{"add": {"generator": "box", "id": "a"}}]},
            "variables",
        ),
    ]
    for program, field in cases:
        r = _call(ctx, session, "clay_program", program)
        assert r["isError"] is True, r
        assert "unexpectedly" not in _text(r), _text(r)
        assert r["structuredContent"]["field"] == field, r["structuredContent"]
        assert "finite" in _text(r)


# --- clay-133 ----------------------------------------------------------------


def _unwrapped(ctx: _Ctx, session: agent_clay.Session) -> int:
    uid = _new_agent_tab(ctx, session, "box")
    assert _call(ctx, session, "clay_uv", {"uid": uid, "action": "unwrap"})["isError"] is False
    return uid


def test_clay_uv_transform_translate_is_held_to_uv_max() -> None:
    ctx, session = _Ctx(), agent_clay.Session()
    uid = _unwrapped(ctx, session)
    for bad in ([2 * UV_MAX, 0], [0, -2 * UV_MAX], [1e30, 0]):
        r = _call(ctx, session, "clay_uv", {"uid": uid, "action": "transform", "translate": bad})
        assert r["isError"] is True, (bad, r)
        assert r["structuredContent"]["field"] == "translate", r["structuredContent"]
    ok = _call(
        ctx, session, "clay_uv", {"uid": uid, "action": "transform", "translate": [UV_MAX, 0]}
    )
    assert ok["isError"] is False, ok


def test_clay_uv_transform_float32_overflow_is_reported_against_the_argument_that_caused_it() -> (
    None
):
    ctx, session = _Ctx(), agent_clay.Session()
    uid = _unwrapped(ctx, session)
    refused = None
    # Each pass spreads the island by 1e3 about its centre; float32 runs out
    # after a dozen of them, and the refusal must not blame the uid.
    for _ in range(20):
        r = _call(ctx, session, "clay_uv", {"uid": uid, "action": "transform", "scale": 1e3})
        if r["isError"]:
            refused = r
            break
    assert refused is not None
    assert refused["structuredContent"]["field"] in ("scale", "translate"), refused[
        "structuredContent"
    ]


# --- clay-134 ----------------------------------------------------------------

_QUERY_SAMPLES: dict[str, Any] = {
    "slot": 0,
    "direction": [0, 1, 0],
    "max_angle": 10,
    "min": [-1, -1, -1],
    "max": [1, 1, 1],
    "space": "world",
}

_UV_SAMPLES: dict[str, Any] = {
    "margin": 0.01,
    "rotate": False,
    "islands": [0],
    "translate": [0.1, 0.1],
    "rotate_deg": 10,
    "scale": 1.0,
}


def test_the_sweep_samples_cover_every_query_and_uv_argument() -> None:
    assert set(_QUERY_SAMPLES) == set(agent_schema._QUERY_ARG_SCHEMAS)
    assert set(_UV_SAMPLES) == {n for names in agent_schema.UV_ACTION_ARGS.values() for n in names}


def test_the_uv_action_argument_table_agrees_with_the_action_descriptions() -> None:
    assert set(agent_schema.UV_ACTION_ARGS) == set(agent_schema.UV_ACTIONS)
    for action, names in agent_schema.UV_ACTION_ARGS.items():
        described = agent_schema.UV_ACTIONS[action]
        bracket = described[described.index("[") + 1 : described.rindex("]")]
        for name in names:
            assert name in bracket, (action, name)
        for other_action, other_names in agent_schema.UV_ACTION_ARGS.items():
            if other_action == action:
                continue
            for name in set(other_names) - set(names):
                # Word-boundary check: ``rotate`` is a prefix of ``rotate_deg``.
                assert f"{name} (" not in bracket and f"[{name} " not in described, (action, name)


def test_clay_select_by_and_clay_uv_refuse_arguments_the_chosen_query_or_action_does_not_take() -> (
    None
):
    ctx, session = _Ctx(), agent_clay.Session()
    uid = _unwrapped(ctx, session)
    assert _call(ctx, session, "clay_select", {"uids": [uid]})["isError"] is False
    assert _call(ctx, session, "clay_element_mode", {"mode": "face"})["isError"] is False

    for qname, query in bsel.QUERIES.items():
        own = {k: _QUERY_SAMPLES[k] for k in query.args}
        base = {"uid": uid, "query": qname, **own}
        assert _call(ctx, session, "clay_select_by", base)["isError"] is False, (qname, base)
        for stray in sorted(set(_QUERY_SAMPLES) - set(query.args)):
            r = _call(ctx, session, "clay_select_by", {**base, stray: _QUERY_SAMPLES[stray]})
            assert r["isError"] is True, (qname, stray, r)
            assert r["structuredContent"]["field"] == stray, (qname, stray, r["structuredContent"])

    assert _call(ctx, session, "clay_element_mode", {"mode": "object"})["isError"] is False
    for action, names in agent_schema.UV_ACTION_ARGS.items():
        own = {k: _UV_SAMPLES[k] for k in names}
        base = {"uid": uid, "action": action, **own}
        assert _call(ctx, session, "clay_uv", base)["isError"] is False, (action, base)
        for stray in sorted(set(_UV_SAMPLES) - set(names)):
            r = _call(ctx, session, "clay_uv", {**base, stray: _UV_SAMPLES[stray]})
            assert r["isError"] is True, (action, stray, r)
            assert r["structuredContent"]["field"] == stray, (action, stray, r["structuredContent"])
