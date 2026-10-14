"""Regression tests for the 2026-10-14 audit's clay-53 and clay-55."""

from __future__ import annotations

from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay.agent import dispatch as agent_clay

from .test_agent_clay import _Ctx, _new_agent_tab


def test_clay_select_refuses_a_null_uids() -> None:
    ctx, session = _Ctx(), agent_clay.Session()
    uid = _new_agent_tab(ctx, session)
    doc = clay_mode.ensure(ctx).get(session.tab_uid).doc
    doc.select({uid})
    assert doc.selection == {uid}

    result = agent_clay.call(ctx, session, "clay_select", {"uids": None})

    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "uids"
    assert doc.selection == {uid}  # a null is a type error, never "clear"

    # The documented clearing call is untouched.
    cleared = agent_clay.call(ctx, session, "clay_select", {"uids": []})
    assert cleared["isError"] is False
    assert doc.selection == set()


def test_clay_export_during_a_save_refuses_with_recovery_retry() -> None:
    ctx, session = _Ctx(), agent_clay.Session()
    _new_agent_tab(ctx, session)
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    tab.saving = True
    try:
        result = agent_clay.call(ctx, session, "clay_export", {})
    finally:
        tab.saving = False

    assert result["isError"] is True
    assert "already in progress" in result["content"][0]["text"]
    assert result["structuredContent"]["recovery"] == "retry"
    assert result["structuredContent"]["changed"] is False
