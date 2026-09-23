"""Regressions for the 2026-09-23 audit's second run, Clay-Generate and
Clay-agent findings.

clay-05: ``generate.py``'s ``_queued``/``on_task_failed`` did not check that
a landing task's own tab uid (from ``done.key``) still matched
``generate_pending["tab_uid"]`` -- the check the first run's clay-04 added,
but only to ``_landed``. A cancelled tab's reference/mesh task, still queued
when Cancel was pressed, could land late and overwrite a newer tab's own
pending job id.

clay-15: ``_view_drag.DragOps.handle_event`` marked ``_render_dirty`` on
every event unconditionally, so a bare hover (nothing pressed) defeated the
redraw skip ``viewer_embed`` was fixed for on 2026-09-02 -- exactly the same
bug, in Clay's own viewport rather than the shared one.

clay-16: ``clay_batch``'s ``BATCH_DEADLINE_S`` was never pushed out by an
entry's own run time, the gap agents-02 (the first run) closed for
``clay_program``'s ``PROGRAM_DEADLINE_S`` the same day. With
``rollback_on_error``, a subprocess-backed entry (decimate/retopo/
smart-unwrap/bake-detail) that alone ran past the budget caused the very
next entry to find the deadline already gone and discard the completed work
along with the rest of the run.

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
from pathlib import Path
from typing import Any

import pytest

from realmspinner.studio.modes.clay import generate as clay_generate
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay.agent import dispatch as agent_clay
from realmspinner.studio.modes.clay.agent import schema as agent_clay_schema
from realmspinner.studio.modes.clay.agent import tools_batch as agent_clay_tools_batch
from realmspinner.studio.modes.clay.ui import _view_drag
from realmspinner.studio.state import DEFAULT_FORM_3D, default_form_2d
from realmspinner.studio.tasks import Done

from .test_agent_clay import _Ctx as _AgentCtx
from .test_agent_clay import _new_agent_tab, _payload

# --- clay-05: a stale queued task must not overwrite a newer tab's pending ----

# The same harness shape ``test_audit_2026_09_23_generate.py`` (the first
# run's own Clay-Generate fixer) built -- read, not imported, for the
# identical reason that file's own docstring gives: this fixer owns only a
# new test file.


class _Store:
    def __init__(self) -> None:
        self.jobs: dict[str, dict] = {}

    def get(self, job_id: str) -> dict | None:
        return self.jobs.get(job_id)


class _Svc:
    def __init__(self, root: Path) -> None:
        self.store = _Store()
        self.root = root
        self.config = None

    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id


class _AppState:
    def __init__(self) -> None:
        self.clay = None
        self.mode = "home"
        self.form_2d = default_form_2d()
        self.form_3d = dict(DEFAULT_FORM_3D)


class _Settings:
    def __init__(self) -> None:
        self.store: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self.store.get(key)

    def set(self, key: str, value: Any) -> None:
        self.store[key] = value


class _Ctx:
    """A queue that lands *nothing* automatically -- unlike
    ``test_audit_2026_09_23_generate.py``'s own ``_Ctx.submit``, which runs
    the job inline, this one only records a scripted result so the test can
    choose the order two tabs' own ``Done``s are handed to
    ``clay_mode.on_task_done``, the one thing this finding is about.
    """

    def __init__(self, tmp_path: Path) -> None:
        self.svc = _Svc(tmp_path)
        self.state = _AppState()
        self.settings = _Settings()
        self.toasts: list[tuple[str, str]] = []
        self._busy: set[str] = set()
        self._results: dict[str, Any] = {}
        self.clay_view = None

    def toast(self, message: str, kind: str = "info", action: Any = None) -> None:
        self.toasts.append((message, kind))

    def busy(self, key: str) -> bool:
        return key in self._busy

    def submit(self, key: str, fn: Any, *args: Any, tag: Any = None, **kwargs: Any) -> bool:
        if key in self._busy:
            return False
        self._busy.add(key)
        self._results[key] = fn(*args, **kwargs)
        return True

    def land(self, key: str) -> None:
        """Deliver exactly one already-submitted key's result now."""
        self._busy.discard(key)
        clay_mode.on_task_done(self, Done(key=key, result=self._results.pop(key)))


def _tab(ctx: Any) -> Any:
    from realmspinner.kernels.mesh import document as bd
    from realmspinner.kernels.mesh import primitives as bp

    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
    return clay_mode.adopt(ctx, doc, title="Scene")


def test_a_stale_queued_task_from_a_cancelled_tab_does_not_overwrite_a_newer_tabs_pending_job_id(
    tmp_path, monkeypatch
):
    from realmspinner.service import jobs as svc_jobs

    ctx = _Ctx(tmp_path)
    tab1 = _tab(ctx)

    ids = iter(["stale-ref-job", "live-ref-job"])
    monkeypatch.setattr(svc_jobs, "create_job", lambda svc, **kw: {"id": next(ids)})

    # tab1's own reference job is queued but never landed -- exactly "the
    # decode task this landing comes from was already submitted by the time
    # Cancel was pressed" (``_landed``'s own docstring), one stage earlier.
    assert clay_generate.submit_text(ctx, tab1, "a wooden barrel")
    stale_key = f"{clay_generate.GEN_REF_KEY}:{tab1.uid}"
    assert stale_key in ctx._results

    clay_generate.cancel(ctx, tab1)
    assert ctx.state.clay.generate_pending is None

    tab2 = _tab(ctx)
    assert clay_generate.submit_text(ctx, tab2, "a clay pot")
    live_key = f"{clay_generate.GEN_REF_KEY}:{tab2.uid}"
    assert live_key in ctx._results

    # The stale task from the cancelled tab lands *after* tab2's own pending
    # request already exists -- before this fix, ``_queued`` wrote whatever
    # job id it carried straight into ``generate_pending`` with no check at
    # all, so tab1's stale "stale-ref-job" clobbered tab2's live request.
    ctx.land(stale_key)
    pending = ctx.state.clay.generate_pending
    assert pending is not None and pending["tab_uid"] == tab2.uid, (
        "a stale task from a cancelled tab must not touch a newer tab's pending request"
    )
    assert pending["reference_job_id"] == "", (
        "the stale job id must not have been written into the live tab's pending request"
    )

    ctx.land(live_key)
    pending = ctx.state.clay.generate_pending
    assert pending is not None and pending["reference_job_id"] == "live-ref-job", (
        "tab2's own job must still land normally once its own task arrives"
    )


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
    "a subprocess-backed entry's own run time" (decimate/retopo/
    smart-unwrap/bake-detail in production) sidesteps that entirely: it
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
            real_time.sleep(0.3)  # stands in for a synchronous Blender spawn
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
