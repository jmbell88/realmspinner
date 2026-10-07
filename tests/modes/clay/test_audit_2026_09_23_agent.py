"""Regressions for the 2026-09-23 audit's Clay-agent findings.

agents-02: ``PROGRAM_DEADLINE_S`` is a 4s budget between ``clay_program``
entries, and ``clay_program`` always rolls back on any failure -- so a
step that alone ran past the budget (a synchronous subprocess, say) caused
the very next entry to find the deadline already gone and discard the
finished work along with everything else, even though nothing was idle. ``_h_program``'s
``_make_entry`` now pushes the deadline out by exactly what each entry took
to run, so only the gap *between* calls counts against the budget.

agents-04: the ``clay_batch``/``clay_program`` tool descriptions did not
state their wall-clock deadlines at all.
"""

from __future__ import annotations

import pytest

from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay.agent import dispatch as agent_clay
from realmspinner.studio.modes.clay.agent import tools_batch as agent_clay_tools_batch

from .test_agent_clay import _Ctx, _payload

# --- agents-02 -------------------------------------------------------------


class _FakeClock:
    """Hands back scripted ``time.monotonic()`` readings in order, so a
    program's own budget arithmetic can be exercised without a real
    multi-second sleep. Installed by replacing ``tools_batch``'s own ``time``
    name (see that module's six ``time.monotonic()`` call sites) rather than
    the stdlib module, so nothing else timing anything during this test is
    touched."""

    def __init__(self, values: list[float]) -> None:
        self._values = list(values)

    def monotonic(self) -> float:
        assert self._values, "clock ran out of scripted readings"
        return self._values.pop(0)


def test_clay_program_does_not_roll_back_a_completed_slow_step_when_a_later_step_misses_the_deadline(  # noqa: E501
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _Ctx()
    session = agent_clay.Session()

    # Six readings, one per ``time.monotonic()`` call ``_h_program``'s
    # ``_make_entry`` makes for a two-step, no-refusal run:
    #   A: the initial deadline = 0.0 + PROGRAM_DEADLINE_S (4.0) = 4.0
    #   B: entry 0's ``started``                              = 0.0
    #   C: entry 0's completion (a 100s "slow" step)           = 100.0
    #      -> deadline pushed to 4.0 + (100.0 - 0.0) = 104.0
    #   D: entry 1's deadline check: 100.0 > 104.0? No.
    #      (the unfixed code checked 100.0 > 4.0 -- True -- and refused here)
    #   E: entry 1's ``started``                               = 100.0
    #   F: entry 1's completion                                = 100.1
    monkeypatch.setattr(
        agent_clay_tools_batch, "time", _FakeClock([0.0, 0.0, 100.0, 100.0, 100.0, 100.1])
    )

    result = agent_clay.call(
        ctx, session, "clay_program",
        {"steps": [{"add": {"generator": "box", "id": "a"}}, {"add": {"generator": "cylinder"}}]},
    )

    assert result["isError"] is False, result
    payload = _payload(result)
    assert payload["rolled_back"] is False
    assert payload["completed"] == 2
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    assert len(tab.doc.objects) == 2


def test_clay_programs_deadline_still_rolls_back_when_the_gap_between_calls_is_too_long(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fix only excuses a *running* call's own duration -- idle time
    between two calls still counts, exactly as ``test_clay_programs_deadline_
    rolls_back`` (``tests/modes/clay/test_agent_clay.py``) already pins with
    a real ``PROGRAM_DEADLINE_S`` of ``0.0``."""
    ctx = _Ctx()
    session = agent_clay.Session()
    monkeypatch.setattr(agent_clay, "PROGRAM_DEADLINE_S", 0.0)

    result = agent_clay.call(
        ctx, session, "clay_program",
        {"steps": [{"add": {"generator": "box", "id": "a"}}, {"add": {"generator": "cylinder"}}]},
    )

    assert result["isError"] is True
    payload = _payload(result)
    assert payload["rolled_back"] is True
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    assert len(tab.doc.objects) == 0


# --- agents-04 -------------------------------------------------------------


def test_clay_batch_description_states_its_deadline() -> None:
    tools = {t.name: t for t in agent_clay.tools()}
    desc = tools["clay_batch"].description
    assert f"{agent_clay_tools_batch.BATCH_DEADLINE_S:g}s" in desc
    assert "deadline" in desc


def test_clay_program_description_states_its_deadline() -> None:
    tools = {t.name: t for t in agent_clay.tools()}
    desc = tools["clay_program"].description
    assert f"{agent_clay.PROGRAM_DEADLINE_S:g}s" in desc
    assert "deadline" in desc
