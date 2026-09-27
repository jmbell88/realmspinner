"""Regressions for the 2026-09-26 audit's Clay-agent findings (fixer w1f1).

clay-agent-tools-01: ``_over_frame_budget`` measured one ``json.dumps`` of a
reply against ``protocol.MAX_FRAME``, but every caller hands the same
payload to ``_json``, which puts it on the wire **twice** (the text block
and, duplicated, ``structuredContent``) -- a payload that fit once could
still build a wire frame past ``MAX_FRAME``.

clay-agent-tools-02: ``clay_batch``/``clay_program`` assemble a reply out of
whole nested tool results with no ceiling of their own, unlike
``clay_scene``/``clay_diagnose``.

clay-agent-tools-03: ``clay_add_figure``'s prefix-collision refusal reverted
its own mutate-then-refuse placement with ``doc.undo()`` (redoable by
default), leaving the refused figure on the redo stack for a later
``clay_redo`` to bring back.
"""

from __future__ import annotations

import json

import pytest

from realmspinner.mcp import rpc
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay.agent import dispatch as agent_clay
from realmspinner.studio.modes.clay.agent import validate as agent_validate

from .test_agent_clay import _Ctx, _new_agent_tab

# --- clay-agent-tools-01 -----------------------------------------------------


def test_over_frame_budget_refuses_a_payload_whose_doubled_wire_frame_exceeds_max_frame_even_if_one_copy_fits(  # noqa: E501
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A synthetic payload sized so *one* ``json.dumps`` comfortably fits
    under the patched-down ``MAX_FRAME``, but two copies -- the shape
    ``_json`` actually puts on the wire (text block plus ``structuredContent``)
    -- would not."""
    monkeypatch.setattr(rpc, "MAX_FRAME", 1000)
    payload = {"blob": "x" * 700}
    one_copy = len(json.dumps(payload))
    assert one_copy < 1000, "sanity: one copy alone must still fit the budget"
    assert one_copy * 2 > 1000, "sanity: two copies must not"

    result = agent_validate._over_frame_budget(payload)

    assert result is not None, "the unfixed code let this through (reproduced)"
    assert result["isError"] is True
    assert "too large" in result["content"][0]["text"]


def test_clay_scene_reply_is_refused_when_its_wire_frame_would_exceed_max_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same claim, through the real dispatch path: ``MAX_FRAME`` is patched
    to exactly one ``clay_scene`` reply's own measured size, so two copies of
    it (what actually reaches the wire) can never fit -- deterministic
    without needing anywhere near the audit's own 11,000-object reproduction.
    """

    def _scene(ctx: _Ctx, session: agent_clay.Session) -> dict:
        _new_agent_tab(ctx, session, "box")
        for _ in range(4):
            agent_clay.call(ctx, session, "clay_add_primitive", {"generator": "box"})
        return agent_clay.call(ctx, session, "clay_scene", {})

    measured = _scene(_Ctx(), agent_clay.Session())
    assert measured["isError"] is False, measured
    real_size = len(measured["content"][0]["text"])

    monkeypatch.setattr(rpc, "MAX_FRAME", real_size)
    result = _scene(_Ctx(), agent_clay.Session())

    assert result["isError"] is True, "the unfixed code let this through (reproduced)"
    assert "too large" in result["content"][0]["text"]


# --- clay-agent-tools-02 -----------------------------------------------------


def test_a_batch_of_scene_reads_is_refused_before_its_reply_exceeds_max_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``MAX_FRAME`` patched to exactly one *unbudgeted* batch reply's own
    measured size -- a second, identical batch (the wire frame doubling
    ``_json`` always does) can then never fit."""

    def _calls(n: int) -> list[dict]:
        return [{"name": "clay_scene", "arguments": {}} for _ in range(n)]

    def _batch(ctx: _Ctx, session: agent_clay.Session) -> dict:
        _new_agent_tab(ctx, session, "box")
        for _ in range(9):
            agent_clay.call(ctx, session, "clay_add_primitive", {"generator": "box"})
        return agent_clay.call(ctx, session, "clay_batch", {"calls": _calls(5)})

    measured = _batch(_Ctx(), agent_clay.Session())
    assert measured["isError"] is False, measured
    real_size = len(measured["content"][0]["text"])

    monkeypatch.setattr(rpc, "MAX_FRAME", real_size)
    result = _batch(_Ctx(), agent_clay.Session())

    assert result["isError"] is True, "the unfixed code let this through (reproduced)"
    assert "too large" in result["content"][0]["text"]


def test_clay_programs_reply_is_also_refused_before_its_reply_exceeds_max_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The twin site (``_h_program``, tools_batch.py:887-890 in the finding)
    gets the identical check -- same measure-then-halve trick.

    A tab must already exist before this call: with none, ``dry_run`` takes
    ``_h_program``'s *other*, compile-only early return (``validated:
    "compile"``, no document touched at all), which is a different reply
    built at a different call site than the one this finding names."""

    def _program(ctx: _Ctx, session: agent_clay.Session) -> dict:
        _new_agent_tab(ctx, session, "box")
        steps = [{"add": {"generator": "box"}} for _ in range(5)]
        return agent_clay.call(ctx, session, "clay_program", {"steps": steps, "dry_run": True})

    measured = _program(_Ctx(), agent_clay.Session())
    assert measured["isError"] is False, measured
    real_size = len(measured["content"][0]["text"])

    monkeypatch.setattr(rpc, "MAX_FRAME", real_size)
    result = _program(_Ctx(), agent_clay.Session())

    assert result["isError"] is True, "the unfixed code let this through (reproduced)"
    assert "too large" in result["content"][0]["text"]


# --- clay-agent-tools-03 -----------------------------------------------------


def test_a_refused_add_figure_prefix_collision_leaves_nothing_on_the_redo_stack() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session, "box")
    # Pre-create an object named exactly what add_figure's own name_prefix
    # would produce for the humanoid's "Head" part, so the collision refusal
    # fires *after* add_assembly has already placed all nineteen parts.
    added = agent_clay.call(
        ctx, session, "clay_add_primitive", {"generator": "box", "name": "Rig_Head"}
    )
    assert added["isError"] is False, added

    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    count_before = len(tab.doc.objects)
    depth_before = len(tab.doc.history.history())

    result = agent_clay.call(
        ctx, session, "clay_add_figure", {"key": "humanoid", "name_prefix": "Rig_"}
    )

    assert result["isError"] is True
    assert "collide" in result["content"][0]["text"]
    assert len(tab.doc.objects) == count_before, "the refused figure's parts must not remain"
    assert len(tab.doc.history.history()) == depth_before, "a total refusal records no undo step"
    # The 2026-09-26 audit, finding clay-agent-tools-03: the unfixed code's
    # ``doc.undo()`` reversed the placement redoably, so it was still sitting
    # on the redo stack right here.
    assert tab.doc.history.can_redo is False

    redo_result = agent_clay.call(ctx, session, "clay_redo", {})
    assert len(tab.doc.objects) == count_before, (
        "clay_redo must not resurrect a figure that was refused (19 objects, reproduced)"
    )
    del redo_result
