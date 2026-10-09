"""Regression tests for the 2026-10-09 audit's Clay agent-door findings:
clay-09 (``clay_op`` refusals are errors), clay-49 and clay-51 (tool
descriptions), clay-52 (float arguments refuse bool and str) and the
``set_generator_params`` follow-up (an OpError must close the gesture).

Each test's name is the claim.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from realmspinner.kernels.mesh import glbimport
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay.agent import dispatch as agent_clay
from realmspinner.studio.modes.clay.agent import schema as agent_schema
from realmspinner.studio.modes.clay.agent import tools_ops

from .test_agent_clay import _Ctx, _new_agent_tab, _payload


def _doc(ctx: _Ctx, session: agent_clay.Session):
    return clay_mode.ensure(ctx).get(session.tab_uid).doc


def _text(result: dict) -> str:
    return result["content"][0]["text"]


def _two_boxes_selected(ctx: _Ctx, session: agent_clay.Session) -> tuple[int, int]:
    a = _new_agent_tab(ctx, session, "box")
    b = _payload(
        agent_clay.call(
            ctx, session, "clay_add_primitive", {"generator": "box", "translation": [3, 0, 0]}
        )
    )["uid"]
    assert agent_clay.call(ctx, session, "clay_select", {"uids": [a, b]})["isError"] is False
    return a, b


def test_clay_op_join_in_face_mode_is_a_refusal_and_stops_a_batch() -> None:
    ctx, session = _Ctx(), agent_clay.Session()
    a, _b = _two_boxes_selected(ctx, session)
    assert agent_clay.call(ctx, session, "clay_element_mode", {"mode": "face"})["isError"] is False

    refused = agent_clay.call(ctx, session, "clay_op", {"name": "join"})
    assert refused["isError"] is True, refused
    assert refused["structuredContent"]["recovery"] == "switch_mode"
    assert "mode" in _text(refused).lower()

    batch = agent_clay.call(
        ctx,
        session,
        "clay_batch",
        {
            "calls": [
                {"name": "clay_op", "arguments": {"name": "join"}},
                {"name": "clay_rename", "arguments": {"uid": a, "name": "after"}},
            ]
        },
    )
    payload = json.loads(_text(batch))
    assert batch["isError"] is True
    assert payload["stopped_at"] == 0
    assert _doc(ctx, session).by_uid(a).name != "after"


def test_clay_op_that_the_kernel_refuses_is_an_error_with_its_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx, session = _Ctx(), agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    agent_clay.call(ctx, session, "clay_select", {"uids": [uid]})

    def fake_run(proxy: Any, doc: Any, op: Any, **params: Any) -> bool:
        proxy.toast("could not do the thing", "error")
        return False

    monkeypatch.setattr(tools_ops.clay_ops, "run", fake_run)
    result = agent_clay.call(ctx, session, "clay_op", {"name": "duplicate"})
    assert result["isError"] is True, result
    assert "could not do the thing" in _text(result)
    assert result["structuredContent"]["messages"] == ["could not do the thing"]
    assert ctx.toasts == []


def _tool(name: str):
    return next(t for t in agent_clay.tools() if t.name == name)


def test_add_tool_descriptions_state_that_selection_happens_in_object_mode_only() -> None:
    for name in ("clay_add_primitive", "clay_add_mesh"):
        assert "selected only in object mode" in _tool(name).description, name


def test_clay_render_description_names_the_view_cap() -> None:
    tool = _tool("clay_render")
    assert str(agent_schema.MAX_RENDER_VIEWS) in tool.description



@pytest.mark.parametrize("bad", [True, "0.2"])
def test_float_arguments_refuse_a_bool_or_a_numeric_string(bad: Any) -> None:
    ctx, session = _Ctx(), agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    cases = [
        ("clay_transform", {"uid": uid, "translation": [bad, 0, 0]}),
        ("clay_add_primitive", {"generator": "box", "scale": [bad, 1, 1]}),
        ("clay_set_params", {"uid": uid, "params": {"size": bad}}),
        ("clay_uv", {"uid": uid, "action": "transform", "translate": [bad, 0]}),
        ("clay_uv", {"uid": uid, "action": "transform", "rotate_deg": bad}),
        ("clay_op", {"name": "bevel", "params": {"width": bad}}),
        ("clay_material", {"uids": [uid], "color": [bad, 0.5, 0.5]}),
    ]
    for tool, args in cases:
        result = agent_clay.call(ctx, session, tool, args)
        assert result["isError"] is True, (tool, args, result)


def _ceiling_just_above_the_document(monkeypatch: pytest.MonkeyPatch, ctx, session) -> None:
    from realmspinner.kernels.mesh.document import _triangles_of

    doc = _doc(ctx, session)
    monkeypatch.setattr(
        glbimport, "MAX_TRIANGLES", sum(_triangles_of(o.mesh) for o in doc.objects) + 3000
    )


def test_add_primitive_params_past_the_triangle_ceiling_closes_its_gesture_and_names_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx, session = _Ctx(), agent_clay.Session()
    _new_agent_tab(ctx, session, "box")
    doc = _doc(ctx, session)
    _ceiling_just_above_the_document(monkeypatch, ctx, session)
    objects, head = len(doc.objects), doc.history.head
    result = agent_clay.call(
        ctx,
        session,
        "clay_add_primitive",
        {"generator": "torus", "params": {"segments": 200, "sides": 200}},
    )
    assert result["isError"] is True, result
    assert result["structuredContent"].get("field") == "params"
    assert len(doc.objects) == objects
    assert doc.history.head == head
    assert doc.history._open_gestures == 0


def test_set_params_past_the_triangle_ceiling_closes_its_gesture_and_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx, session = _Ctx(), agent_clay.Session()
    a = _new_agent_tab(ctx, session, "torus")
    b = _payload(
        agent_clay.call(
            ctx, session, "clay_add_primitive", {"generator": "torus", "translation": [3, 0, 0]}
        )
    )["uid"]
    doc = _doc(ctx, session)
    before = {u: doc.by_uid(u).mesh for u in (a, b)}
    _ceiling_just_above_the_document(monkeypatch, ctx, session)
    head = doc.history.head
    result = agent_clay.call(
        ctx, session, "clay_set_params", {"uids": [a, b], "params": {"segments": 200, "sides": 200}}
    )
    assert result["isError"] is True, result
    assert result["structuredContent"].get("field") == "params"
    assert doc.history.head == head
    assert doc.history._open_gestures == 0
    assert all(doc.by_uid(u).mesh is before[u] for u in (a, b))


def test_dispatch_call_turns_a_stray_op_error_into_an_error_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from realmspinner.kernels.mesh.elements import OpError

    ctx, session = _Ctx(), agent_clay.Session()

    def boom(*_a: Any, **_k: Any) -> dict:
        raise OpError("kernel said no")

    monkeypatch.setitem(agent_clay._HANDLERS, "clay_scene", boom)
    result = agent_clay.call(ctx, session, "clay_scene", {})
    assert result["isError"] is True
    assert "kernel said no" in _text(result)
