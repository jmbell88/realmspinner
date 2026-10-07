"""Regressions for the 2026-09-23 audit's second run, Clay-agent findings.

clay-15: ``_view_drag.DragOps.handle_event`` marked ``_render_dirty`` on
every event unconditionally, so a bare hover (nothing pressed) defeated the
redraw skip ``viewer_embed`` was fixed for on 2026-09-02 -- exactly the same
bug, in Clay's own viewport rather than the shared one.

clay-16: ``clay_batch``'s ``BATCH_DEADLINE_S`` was never pushed out by an
entry's own run time, the gap agents-02 (the first run) closed for
``clay_program``'s ``PROGRAM_DEADLINE_S`` the same day. With
``rollback_on_error``, a slow entry (a synchronous subprocess, say) that
alone ran past the budget caused the very next entry to find the deadline
already gone and discard the completed work along with the rest of the run.

clay-20: ``schema.py`` carried a second, dead ``PROGRAM_DEADLINE_S = 4.0``
that nothing read (every real reader goes through ``agent_clay.
PROGRAM_DEADLINE_S``, i.e. ``dispatch.py``'s own copy) -- contradicting the
module's own "there is only ever one copy of it, here" claim, which was true
of every other constant in the file except this one.

clay-19 (docstring only, no regression test -- see the return): corrected in
``tools_structure.py``.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay.agent import dispatch as agent_clay
from realmspinner.studio.modes.clay.agent import schema as agent_clay_schema
from realmspinner.studio.modes.clay.agent import tools_batch as agent_clay_tools_batch
from realmspinner.studio.modes.clay.ui import _view_drag

from .test_agent_clay import _Ctx as _AgentCtx
from .test_agent_clay import _new_agent_tab, _payload

# --- clay-15: a bare hover must not mark the Clay viewport dirty --------------


def test_a_bare_hover_does_not_redraw_the_clay_scene():
    """The same technique ``test_findings_viewports.py``'s own
    ``test_a_bare_hover_does_not_redraw_the_scene`` used to pin the identical
    fix in the shared ``viewer_embed.Viewer`` on 2026-09-02: before this fix,
    ``DragOps.handle_event`` set ``self._render_dirty = True`` unconditionally,
    once per event, so a bare hover (``MOUSEMOTION`` with nothing pressed)
    marked the picture dirty every frame regardless of whether anything it
    depends on had actually changed.
    """
    handler = inspect.getsource(_view_drag.DragOps.handle_event)
    # Press, release and wheel each still mark dirty unconditionally --
    # motion's own dirtiness is decided inside ``_motion``, not here.
    assert handler.count("self._render_dirty = True") == 3

    motion = inspect.getsource(_view_drag.DragOps._motion)
    # A live drag (grab in progress) always redraws; a bare hover only when
    # what it is over actually changed.
    assert "gizmo.hover != prev_gizmo_hover" in motion
    assert "self.hover_element != prev_hover_element" in motion


# --- clay-16: clay_batch's deadline must not count an entry's own run time ----


def test_batch_deadline_does_not_count_an_entrys_own_running_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real wall-clock time, not a scripted clock: a fake ``time.monotonic``
    that hands back the identical number of readings old and fixed code
    consume cannot tell the two apart, since the unfixed ``_h_batch`` reads
    ``time.monotonic()`` at only two sites (the initial deadline and each
    later entry's check) while the fixed one reads it at four (also
    ``started``/``completion`` per entry) -- a script tuned to one's call
    count silently misaligns with the other's. A real sleep standing in for
    "a subprocess-backed entry's own run time" (a synchronous
    subprocess in production) sidesteps that entirely: it
    elapses the same real time regardless of which code reads the clock how
    many times.
    """
    ctx = _AgentCtx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session, "box")
    monkeypatch.setattr(agent_clay_tools_batch, "BATCH_DEADLINE_S", 0.05)

    import time as real_time

    original = agent_clay_tools_batch._resolve_and_call
    slept = False

    def _slow_first_entry(ctx: Any, session: Any, doc: Any, name: str, arguments: dict) -> Any:
        nonlocal slept
        result = original(ctx, session, doc, name, arguments)
        if not slept:
            slept = True
            real_time.sleep(0.3)  # stands in for a synchronous subprocess
        return result

    monkeypatch.setattr(agent_clay_tools_batch, "_resolve_and_call", _slow_first_entry)

    calls = [
        {"name": "clay_add_primitive", "arguments": {"generator": "cylinder"}},
        {"name": "clay_add_primitive", "arguments": {"generator": "cone"}},
    ]
    result = agent_clay.call(
        ctx, session, "clay_batch", {"calls": calls, "rollback_on_error": True}
    )

    assert result["isError"] is False, result
    payload = _payload(result)
    assert payload["rolled_back"] is False
    assert payload["completed"] == 2
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    assert len(tab.doc.objects) == 3  # the seed box plus both batched primitives


def test_batch_deadline_still_refuses_when_the_gap_between_entries_is_too_long(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fix only excuses an entry's own duration -- idle time between two
    entries still counts against the budget."""
    ctx = _AgentCtx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session, "box")
    monkeypatch.setattr(agent_clay_tools_batch, "BATCH_DEADLINE_S", 0.0)

    calls = [
        {"name": "clay_add_primitive", "arguments": {"generator": "cylinder"}},
        {"name": "clay_add_primitive", "arguments": {"generator": "cone"}},
    ]
    result = agent_clay.call(
        ctx, session, "clay_batch", {"calls": calls, "rollback_on_error": True}
    )

    assert result["isError"] is True, result
    payload = _payload(result)
    assert payload["stopped_at"] == 1
    assert payload["rolled_back"] is True


# --- clay-20: schema.py must not shadow dispatch.py's PROGRAM_DEADLINE_S -----


def test_schema_module_does_not_shadow_program_deadline_s() -> None:
    """``schema.py``'s own module docstring: every constant it declares has
    "only ever one copy of it, here" -- but ``PROGRAM_DEADLINE_S`` was never
    one of those constants (the same docstring says it "stay[s] in
    dispatch.py itself"), so a second, dead definition of it in this module
    contradicted that claim rather than fulfilling it. Nothing ever read the
    schema-module copy: every real caller reaches ``agent_clay.
    PROGRAM_DEADLINE_S``, i.e. ``dispatch.py``'s own."""
    assert not hasattr(agent_clay_schema, "PROGRAM_DEADLINE_S")
    assert agent_clay.PROGRAM_DEADLINE_S == 4.0
