"""The 2026-10-14 audit, cleanup C7: the Pose-mode ops an agent can never run (finding clay-12)."""

from __future__ import annotations

import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio.modes.clay import ops as clay_ops
from realmspinner.studio.modes.clay.agent import dispatch as agent_clay

from .test_agent_clay import _Ctx, _new_agent_tab


def _gated_on_pose_mode() -> set[str]:
    """Every op whose refusal, with a bone selected and Pose mode off, says "Pose mode"."""
    doc = bd.ClayDoc()
    uid = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Hip", mesh=bp.box(), bone=True)).uid
    doc.select([uid])
    doc.set_pose_mode(False)
    return {
        op.name
        for op in clay_ops.OPS
        if "object" in op.modes
        and not op.enabled(doc)
        and "Pose mode" in clay_ops.reason_for(op, doc)
    }


def _agent_op_enum() -> list[str]:
    tools = {t.name: t for t in agent_clay.tools()}
    return tools["clay_op"].schema["properties"]["name"]["enum"]


def test_every_op_in_clay_ops_enum_can_run_for_an_agent_or_is_excluded_from_it() -> None:
    # The agent has no tool that switches Pose mode on (and the playhead is session state), so
    # an op whose gate says "Switch to Pose mode first." can never be enabled for it.
    pose_gated = _gated_on_pose_mode()
    assert pose_gated, "the probe found no Pose-mode op: it no longer proves anything"
    enum = set(_agent_op_enum())
    assert not (pose_gated & enum), pose_gated & enum
    # and nothing else was dropped with them
    assert enum == {op.name for op in clay_ops.OPS} - pose_gated


def test_pose_ops_names_exactly_the_ops_gated_on_pose_mode() -> None:
    assert set(clay_ops.POSE_OPS) == _gated_on_pose_mode()
    for name in clay_ops.POSE_OPS:
        clay_ops.get(name)  # each names a real op


@pytest.mark.parametrize("name", ["insert-key", "delete-key", "reset-pose", "apply-pose-as-rest"])
def test_clay_op_refuses_a_pose_op_by_name_if_called_anyway(name: str) -> None:
    ctx, session = _Ctx(), agent_clay.Session()
    _new_agent_tab(ctx, session)
    result = agent_clay.call(ctx, session, "clay_op", {"name": name})
    assert result["isError"] is True
    text = result["content"][0]["text"]
    assert name in text and "Pose mode" in text
    assert result["structuredContent"]["field"] == "name"


def test_the_agent_can_still_run_a_non_pose_op() -> None:
    ctx, session = _Ctx(), agent_clay.Session()
    _new_agent_tab(ctx, session)
    result = agent_clay.call(ctx, session, "clay_op", {"name": "select-all"})
    assert result["isError"] is False, result
