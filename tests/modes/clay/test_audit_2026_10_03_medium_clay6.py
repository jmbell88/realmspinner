"""The 2026-10-03 audit's Medium Clay findings, batch 6 (clay-55..59, 77, 78):
the unwrap's whole-call bound, the Generate request's lifecycle (closed tab,
double confirm, cancel), a non-finite stored grid size, the mesh-file export's
encode thread and a background op's landing against a live drag.
"""

from __future__ import annotations

# ruff: noqa: E501 - a regression test's name is the claim, and these two are long
import threading
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import uvunwrap as lscm
from realmspinner.kernels.mesh.elements import OpError
from realmspinner.service import jobs as svc_jobs
from realmspinner.studio import dialogs
from realmspinner.studio.modes.clay import generate as clay_generate
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay import ops as clay_ops
from realmspinner.studio.modes.clay.state import ClayState
from realmspinner.studio.tasks import Done

from .test_clay_generate import (
    _Ctx,
    _patch_landing,
    _tab,
    _ThreadedCtx,
    _through_mesh_queued,
    _through_preview,
)


def _mk(tmp_path: Any, cls: Any = _Ctx) -> Any:
    """The shared test ctx plus the ``cache`` a non-``clay-`` task key (the
    ``cancel:<job>`` one) reaches if a test drains it through Clay's handler;
    the real shell never routes those here."""
    ctx = cls(tmp_path)
    ctx.cache = SimpleNamespace(invalidate=lambda: None)
    return ctx


# --- clay-55: the unwrap's bound ----------------------------------------------


def _grid(n: int) -> bm.Mesh:
    xs, ys = np.meshgrid(np.arange(n), np.arange(n))
    pos = np.stack([xs.ravel(), np.zeros(n * n), ys.ravel()], 1).astype("f4")
    faces = [
        [j * n + i, j * n + i + 1, (j + 1) * n + i + 1, (j + 1) * n + i]
        for j in range(n - 1)
        for i in range(n - 1)
    ]
    return bm.from_faces(pos, faces)


def _separate_quads(k: int) -> bm.Mesh:
    pos: list[list[float]] = []
    faces: list[list[int]] = []
    for i in range(k):
        o = len(pos)
        pos += [[i * 2, 0, 0], [i * 2 + 1, 0, 0], [i * 2 + 1, 0, 1], [i * 2, 0, 1]]
        faces.append([o, o + 1, o + 2, o + 3])
    return bm.from_faces(np.array(pos, dtype="f4"), faces)


NO_SEAMS = np.zeros((0, 2), dtype="i4")


def test_unwrap_lscm_admits_nothing_that_takes_over_a_second():
    """A 100x100 grid patch (10,000 vertices, under the old 20,000 ceiling)
    solved for 2.8 s on the frame thread, and 2,000 separate quads for over a
    second: both are refused now, before anything is triangulated or solved."""
    started = time.perf_counter()
    with pytest.raises(OpError, match="vertices"):
        lscm.unwrap_lscm(_grid(100), NO_SEAMS)
    with pytest.raises(OpError, match="too big"):
        lscm.unwrap_lscm(_separate_quads(2000), NO_SEAMS)
    # Refusing is cheap: nothing was solved.
    assert time.perf_counter() - started < 1.5


def test_unwrap_lscm_bounds_the_whole_call_not_only_each_island(monkeypatch):
    """Two islands each under the per-island ceiling that together are over the
    whole-call budget are refused -- the ceiling alone was per island."""
    monkeypatch.setattr(lscm, "MAX_LSCM_WORK", lscm._island_work(16) * 1.5)
    with pytest.raises(OpError, match="too big"):
        lscm.unwrap_lscm(_separate_quads(2), NO_SEAMS)
    # And one island of the same size alone is still admitted.
    out = lscm.unwrap_lscm(_separate_quads(1), NO_SEAMS)
    assert out.uv is not None


def test_unwrap_lscm_still_admits_an_ordinary_patch():
    out = lscm.unwrap_lscm(_grid(20), NO_SEAMS)
    assert out.uv is not None and len(out.uv) == len(out.loops)


# --- clay-56: closing the tab a Generate was started from ----------------------


def test_closing_the_tab_a_generate_started_from_clears_the_pending_request_before_it_reaches_preview(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(svc_jobs, "create_job", lambda svc, **kw: {"id": "ref1"})
    monkeypatch.setattr(svc_jobs, "cancel_job", lambda svc, job_id: {"ok": True})
    ctx = _mk(tmp_path)
    tab = _tab(ctx)
    clay_generate.submit_text(ctx, tab, "a wooden barrel")
    ctx.land_all()  # the reference job id is known, the job still rendering

    clay_mode.close_tab(ctx, tab.uid)
    ctx.svc.store.jobs["ref1"] = {"status": "done"}
    clay_generate.poll(ctx)

    state = ctx.state.clay
    assert state.generate_pending is None, "a preview with no tab to show it would wedge Generate"
    # ...so the next press, on any tab, is not refused.
    other = _tab(ctx)
    assert clay_generate.submit_text(ctx, other, "a stone well") is True


def test_a_tab_closed_behind_the_pending_requests_back_is_cleared_by_the_poll(
    tmp_path, monkeypatch
):
    """The backstop for a close that did not go through ``close_tab``."""
    monkeypatch.setattr(svc_jobs, "create_job", lambda svc, **kw: {"id": "ref1"})
    monkeypatch.setattr(svc_jobs, "cancel_job", lambda svc, job_id: {"ok": True})
    ctx = _mk(tmp_path)
    tab = _tab(ctx)
    clay_generate.submit_text(ctx, tab, "a wooden barrel")
    ctx.land_all()
    ctx.state.clay.close(tab.uid)
    ctx.svc.store.jobs["ref1"] = {"status": "done"}

    clay_generate.poll(ctx)

    assert ctx.state.clay.generate_pending is None


# --- clay-57: the deferred landing's confirm -------------------------------------


class _CountingConfirms:
    def __init__(self) -> None:
        self.asked: list[Any] = []

    def ask(self, confirm: Any) -> None:
        self.asked.append(confirm)


def test_a_deferred_landing_that_asks_to_confirm_asks_exactly_once(tmp_path, monkeypatch):
    _patch_landing(monkeypatch)
    monkeypatch.setattr(clay_mode, "SLOW_TRIANGLES", 1)
    ctx = _mk(tmp_path)
    ctx.confirms = _CountingConfirms()
    ctx.clay_view = SimpleNamespace(dragging=True)
    tab = _tab(ctx)
    _through_mesh_queued(ctx, tab, monkeypatch, lambda svc, job_id, **kw: {"id": "mesh1"})
    ctx.svc.store.jobs["mesh1"] = {"status": "done"}
    clay_generate.poll(ctx)
    ctx.land_all()  # mid-drag: the landing is deferred
    assert ctx.state.clay.generate_pending["deferred"] is not None

    ctx.clay_view.dragging = False
    for _ in range(4):  # four poll ticks while the first question is on screen
        ctx.state.clay.generate_pending["next_poll"] = 0.0
        clay_generate.poll(ctx)

    assert len(ctx.confirms.asked) == 1


# --- clay-58: a non-finite stored grid size ---------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), 1e999])
def test_a_non_finite_stored_grid_size_is_ignored_not_raised(bad):
    state = ClayState()
    default = state.grid_size
    clay_mode._restore_view(state, {"grid_size": bad, "grid": False})
    assert state.grid_size == default
    assert state.grid is False  # the rest of the block still restores


def test_a_finite_stored_grid_size_is_still_clamped():
    state = ClayState()
    clay_mode._restore_view(state, {"grid_size": 5000})
    assert state.grid_size == 1000.0


# --- clay-59: the mesh-file export's encode ---------------------------------------


def _spy(monkeypatch: pytest.MonkeyPatch, owner: Any, name: str) -> list[str]:
    threads: list[str] = []
    real = getattr(owner, name)

    def spy(*args: Any, **kwargs: Any) -> Any:
        threads.append(threading.current_thread().name)
        return real(*args, **kwargs)

    monkeypatch.setattr(owner, name, spy)
    return threads


def test_clay_export_mesh_file_encodes_off_the_frame_thread(tmp_path, monkeypatch):
    from realmspinner.kernels.geom3d import glbwrite
    from realmspinner.kernels.mesh import objexport

    ctx = _mk(tmp_path, _ThreadedCtx)
    tab = _tab(ctx)
    glb_threads = _spy(monkeypatch, glbwrite, "write_glb")
    obj_threads = _spy(monkeypatch, objexport, "claydoc_to_obj")
    model_threads = _spy(monkeypatch, bd, "to_model")
    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: tmp_path / "out.glb")

    clay_mode.export_mesh_file(ctx, tab, "glb")
    assert glb_threads == ["realmspinner-task-test"]
    assert model_threads == ["realmspinner-task-test"]
    assert (tmp_path / "out.glb").is_file()

    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: tmp_path / "out.obj")
    ctx._busy.clear()
    tab.saving = False
    clay_mode.export_mesh_file(ctx, tab, "obj")
    assert obj_threads == ["realmspinner-task-test"]
    assert (tmp_path / "out.obj").is_file()


# --- clay-77: a background op landing against a live drag --------------------------


def _bg_ctx(tmp_path, monkeypatch) -> tuple[Any, Any, list[int]]:
    ctx = _mk(tmp_path)
    ctx.clay_view = SimpleNamespace(dragging=False)
    tab = _tab(ctx)
    applied: list[int] = []
    monkeypatch.setattr(clay_ops, "decimate_apply", lambda c, doc, result: applied.append(1))
    return ctx, tab, applied


def test_a_background_op_result_waits_for_a_live_element_drag_instead_of_being_overwritten_by_its_commit(
    tmp_path, monkeypatch
):
    ctx, tab, applied = _bg_ctx(tmp_path, monkeypatch)
    tab.bg_busy = "Decimating..."
    ctx.clay_view.dragging = True

    clay_mode.on_task_done(ctx, Done(key=f"clay-bg:{tab.uid}", result={"items": []}))
    assert applied == [], "the drag's commit would write the pre-op mesh back over it"

    clay_generate.poll(ctx)
    assert applied == []  # still dragging

    ctx.clay_view.dragging = False
    clay_generate.poll(ctx)
    assert applied == [1]
    assert tab.bg_busy == ""
    clay_generate.poll(ctx)
    assert applied == [1], "applied exactly once"


def test_a_background_op_result_waits_for_an_export_reading_the_document(tmp_path, monkeypatch):
    ctx, tab, applied = _bg_ctx(tmp_path, monkeypatch)
    tab.saving = True

    clay_mode.on_task_done(ctx, Done(key=f"clay-bg:{tab.uid}", result={"items": []}))
    assert applied == []

    # The save's own completion is the moment the held result lands.
    tab.saving = False
    clay_generate.poll(ctx)
    assert applied == [1]


def test_a_background_op_result_for_a_closed_tab_is_dropped(tmp_path, monkeypatch):
    ctx, tab, applied = _bg_ctx(tmp_path, monkeypatch)
    ctx.clay_view.dragging = True
    clay_mode.on_task_done(ctx, Done(key=f"clay-bg:{tab.uid}", result={"items": []}))
    ctx.state.clay.close(tab.uid)
    ctx.clay_view.dragging = False

    clay_generate.poll(ctx)

    assert applied == []


# --- clay-78: Cancel cancels the job ---------------------------------------------


def _cancel_spy(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    cancelled: list[str] = []
    monkeypatch.setattr(
        svc_jobs, "cancel_job", lambda svc, job_id: cancelled.append(job_id) or {"ok": True}
    )
    return cancelled


def test_cancelling_a_generate_cancels_its_running_job(tmp_path, monkeypatch):
    cancelled = _cancel_spy(monkeypatch)
    ctx = _mk(tmp_path)
    tab = _tab(ctx)
    monkeypatch.setattr(svc_jobs, "create_job", lambda svc, **kw: {"id": "ref1"})
    clay_generate.submit_text(ctx, tab, "a wooden barrel")
    ctx.land_all()
    ctx.svc.store.jobs["ref1"] = {"status": "running"}

    clay_generate.cancel(ctx, tab)

    assert ctx.state.clay.generate_pending is None
    assert cancelled == ["ref1"]


def test_cancelling_a_generate_cancels_its_mesh_job(tmp_path, monkeypatch):
    cancelled = _cancel_spy(monkeypatch)
    ctx = _mk(tmp_path)
    tab = _tab(ctx)
    _through_mesh_queued(ctx, tab, monkeypatch, lambda svc, job_id, **kw: {"id": "mesh1"})
    ctx.svc.store.jobs["mesh1"] = {"status": "running"}

    clay_generate.cancel(ctx, tab)

    assert cancelled == ["mesh1"]


def test_cancelling_a_generate_at_the_preview_cancels_no_job(tmp_path, monkeypatch):
    """The reference is done by then: there is nothing running to stop."""
    cancelled = _cancel_spy(monkeypatch)
    ctx = _mk(tmp_path)
    tab = _tab(ctx)
    _through_preview(ctx, tab, monkeypatch)

    clay_generate.cancel(ctx, tab)

    assert ctx.state.clay.generate_pending is None
    assert cancelled == []


def test_a_job_queued_after_its_generate_was_cancelled_is_cancelled_when_it_lands(
    tmp_path, monkeypatch
):
    """Cancel pressed while the create task was still in flight: the job id
    was not known, so it is cancelled when the late result arrives."""
    cancelled = _cancel_spy(monkeypatch)
    ctx = _mk(tmp_path)
    tab = _tab(ctx)
    monkeypatch.setattr(svc_jobs, "create_job", lambda svc, **kw: {"id": "ref1"})
    clay_generate.submit_text(ctx, tab, "a wooden barrel")

    clay_generate.cancel(ctx, tab)  # the queue task has run but not landed
    assert cancelled == []
    ctx.land_all()

    assert cancelled == ["ref1"]
    assert ctx.state.clay.generate_pending is None
