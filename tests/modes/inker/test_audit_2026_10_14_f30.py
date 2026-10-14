"""Regressions from the 2026-10-14 audit, fixer F30: Flourish's texture and
restyle doors, the text-model memo, and Inker's quick-tool hold."""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path
from types import SimpleNamespace

import pygame
from PIL import ImageGrab

from realmspinner.kernels import pixel as inker
from realmspinner.kernels.pixel.flourish import bake as B
from realmspinner.kernels.pixel.flourish import presets
from realmspinner.studio.modes.inker import flourish as inker_flourish
from realmspinner.studio.modes.inker import mode as inker_mode
from realmspinner.studio.modes.inker import ops as inker_ops
from realmspinner.studio.modes.inker import state as inker_state
from realmspinner.studio.tasks import Done


class _Store:
    def __init__(self) -> None:
        self.jobs: dict[str, dict] = {}
        self.reads: list[str] = []

    def get(self, job_id):
        self.reads.append(job_id)
        return self.jobs.get(job_id)


class _Ctx:
    """``submit`` either runs the task inline (``run=True``) or only records it."""

    def __init__(self, root: Path, *, run: bool = True) -> None:
        self.state = SimpleNamespace(inker=inker_state.InkerState())
        self.svc = SimpleNamespace(store=_Store(), job_dir=lambda j: root / j, config=None)
        self.toasts: list = []
        self.tasks = SimpleNamespace(set_progress=lambda *a, **k: None)
        self.pending: list[Done] = []
        self.submitted: list = []
        self._run = run
        self._busy: set[str] = set()

    def toast(self, text, level="info", **_):
        self.toasts.append((text, level))

    def busy(self, key):
        return key in self._busy

    def progress(self, key):
        return None

    def submit(self, key, fn, *args, **kwargs):
        if key in self._busy:
            return False
        self._busy.add(key)
        self.submitted.append((key, fn, args, kwargs))
        if self._run:
            try:
                done = Done(key=key, result=fn(*args, **kwargs))
            except Exception as exc:  # noqa: BLE001
                done = Done(key=key, error=exc)
            self.pending.append(done)
        return True

    def land_all(self):
        while self.pending:
            done = self.pending.pop(0)
            self._busy.discard(done.key)
            inker_mode.on_task_done(self, done)


def _scene(tmp_path, *, run: bool = True):
    ctx = _Ctx(tmp_path, run=run)
    tab = inker_state.InkerDoc(doc=inker.Document.blank(32, 32))
    ctx.state.inker.docs.append(tab)
    ctx.state.inker.active_uid = tab.uid
    rec = dataclasses.replace(presets.load("sword_impact"), width=32, height=32, supersample=2)
    group = tab.doc.insert_flourish(B.bake(rec))
    return ctx, tab, group


# -- inker-flourish-01 -------------------------------------------------------------------------


def test_flourish_texture_and_restyle_call_create_job_with_keywords_it_accepts(
    tmp_path, monkeypatch
):
    from realmspinner.service import jobs as svc_jobs

    real_signature = inspect.signature(svc_jobs.create_job)
    bound: list[str] = []

    def strict_create_job(svc, **kwargs):
        # Raises TypeError for a keyword the real door does not have -- the
        # whole of the defect.
        real_signature.bind(svc, **kwargs)
        bound.append(kwargs["prompt"])
        return {"id": f"job{len(bound)}"}

    monkeypatch.setattr(svc_jobs, "create_job", strict_create_job)
    ctx, tab, group = _scene(tmp_path)
    assert inker_mode.flourish_texture_generate(ctx, tab, subject="skull ember")
    ctx.land_all()
    assert ctx.state.inker.flourish_texture_pending["job_id"] == "job1"
    assert not any("not queued" in text for text, _level in ctx.toasts)
    assert inker_mode.flourish_restyle(ctx, tab, phase="sparks", anchors=2)
    ctx.land_all()
    assert len(ctx.state.inker.flourish_restyle_pending["jobs"]) == 2
    assert not any("not queued" in text for text, _level in ctx.toasts)


# -- inker-flourish-15 -------------------------------------------------------------------------


def test_text_model_available_notices_a_model_placed_after_first_use(tmp_path, monkeypatch):
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: object())
    now = [100.0]
    monkeypatch.setattr(inker_flourish, "clock", lambda: now[0])
    inker_flourish.reset_text_model_available_cache()
    config = SimpleNamespace(t2i_model_root=tmp_path)
    try:
        assert inker_flourish.text_model_available(config) is False
        base = inker_flourish.text_model_dir(config)
        base.mkdir(parents=True)
        (base / "config.json").write_text("{}")
        (base / "model.safetensors").write_bytes(b"x")
        # Within the window the memo still answers (the per-frame walk it exists to stop).
        assert inker_flourish.text_model_available(config) is False
        now[0] += 3600.0
        assert inker_flourish.text_model_available(config) is True
    finally:
        inker_flourish.reset_text_model_available_cache()


# -- inker-flourish-16 -------------------------------------------------------------------------


def test_poll_texture_reads_the_job_store_off_the_frame_thread(tmp_path):
    ctx, tab, group = _scene(tmp_path, run=False)
    state = ctx.state.inker
    state.flourish_texture_pending = {
        "tab_uid": tab.uid, "group": group, "layer": None, "job_id": "j", "next_poll": 0.0
    }
    ctx.svc.store.jobs["j"] = {"status": "failed", "error": "out of memory"}
    inker_flourish.poll_texture(ctx, state, now=1.0)
    assert ctx.svc.store.reads == []  # nothing queried on the calling (frame) thread
    assert len(ctx.submitted) == 1
    _key, fn, args, kwargs = ctx.submitted[0]
    result = fn(*args, **kwargs)  # the task thread's half
    assert ctx.svc.store.reads == ["j"]
    inker_flourish.on_texture_polled(ctx, state, Done(key=_key, result=result))
    assert state.flourish_texture_pending is None
    assert ctx.toasts[-1][1] == "warn"


def test_poll_restyle_reads_the_job_store_off_the_frame_thread(tmp_path):
    ctx, tab, group = _scene(tmp_path, run=False)
    state = ctx.state.inker
    state.flourish_restyle_pending = {
        "tab_uid": tab.uid, "group": group, "phase": "hit", "span": [0, 2],
        "jobs": {0: "a", 2: "b"}, "frames": [0, 2], "next_poll": 0.0, "subject": "",
    }
    ctx.svc.store.jobs["a"] = {"status": "done"}
    ctx.svc.store.jobs["b"] = {"status": "running"}
    inker_flourish.poll_restyle(ctx, state, now=1.0)
    assert ctx.svc.store.reads == []
    _key, fn, args, kwargs = ctx.submitted[0]
    result = fn(*args, **kwargs)
    assert sorted(ctx.svc.store.reads) == ["a", "b"]
    inker_flourish.on_restyle_polled(ctx, state, Done(key=_key, result=result))
    assert state.flourish_restyle_pending is not None  # b is still running


# -- inker-flourish-19 -------------------------------------------------------------------------


def test_a_restyle_that_fails_midway_cancels_the_keyframe_jobs_it_already_queued(
    tmp_path, monkeypatch
):
    from realmspinner.service import jobs as svc_jobs

    made: list[str] = []
    cancelled: list[str] = []

    def create_job(svc, **kwargs):
        if len(made) == 2:
            raise RuntimeError("the queue refused the third keyframe")
        made.append(f"job{len(made) + 1}")
        return {"id": made[-1]}

    monkeypatch.setattr(svc_jobs, "create_job", create_job)
    monkeypatch.setattr(svc_jobs, "cancel_job", lambda svc, job_id: cancelled.append(job_id))
    ctx, tab, _group = _scene(tmp_path)
    assert inker_mode.flourish_restyle(ctx, tab, phase="sparks", anchors=3)
    ctx.land_all()
    assert ctx.state.inker.flourish_restyle_pending is None
    assert ctx.toasts[-1][1] == "warn"
    assert made == ["job1", "job2"]
    assert cancelled == made


# -- inker-flourish-20 -------------------------------------------------------------------------


def test_keep_painted_cells_is_refused_while_the_document_is_busy(tmp_path, monkeypatch):
    ctx, tab, group = _scene(tmp_path)
    state = ctx.state.inker
    resolved: list = []
    monkeypatch.setattr(tab.doc, "flourish_conflicts", lambda g: [1])
    monkeypatch.setattr(tab.doc, "resolve_flourish", lambda g, flagged: resolved.append(flagged))
    op = inker_ops.get("flourish_keep_edits")
    assert op.enabled(state, tab)
    tab.saving = True
    assert tab.busy
    assert not op.enabled(state, tab)
    assert inker_ops.reason_for(op, state, tab) == inker_flourish.BUSY
    assert inker_mode.flourish_keep_edits(ctx, tab) is False
    assert resolved == []
    tab.saving = False
    inker_mode.flourish_keep_edits(ctx, tab)
    assert resolved == [[1]]


# -- inker-mode-08 / inker-mode-09 -------------------------------------------------------------


def _ev(kind, key, mod):
    return pygame.event.Event(kind, key=key, mod=mod)


def _quick_scene(uid="p"):
    state = inker_state.InkerState(tool="brush")
    tab = inker_state.InkerDoc(doc=inker.Document.blank(8, 8), uid=uid, title="P")
    state.add(tab)
    toasts: list = []
    ctx = SimpleNamespace(
        state=SimpleNamespace(inker=state), toast=lambda *a, **k: toasts.append(a)
    )
    return ctx, state, tab


def test_a_tool_chosen_during_a_ctrl_hold_survives_the_release(monkeypatch):
    monkeypatch.setattr(ImageGrab, "grabclipboard", lambda *a, **k: None)
    ctx, state, tab = _quick_scene()
    doc = tab.doc
    doc.select_all()
    doc.copy()
    doc.deselect()

    inker_mode.handle_key(ctx, _ev(pygame.KEYDOWN, pygame.K_LCTRL, pygame.KMOD_CTRL))
    assert state.tool == "move"
    inker_mode.handle_key(ctx, _ev(pygame.KEYDOWN, pygame.K_v, pygame.KMOD_CTRL))
    assert doc.floating is not None
    inker_mode.handle_key(ctx, _ev(pygame.KEYUP, pygame.K_LCTRL, 0))
    assert state.tool == "move"  # the paste's tool, not the pre-hold brush
    assert doc.floating is not None
    assert state.quick_key == ""
    assert "set_tool" not in vars(state)  # the watch is gone once the key is handled


def test_a_ctrl_hold_with_nothing_chosen_still_restores_the_brush():
    ctx, state, _tab = _quick_scene()
    inker_mode.handle_key(ctx, _ev(pygame.KEYDOWN, pygame.K_LCTRL, pygame.KMOD_CTRL))
    assert state.tool == "move"
    inker_mode.handle_key(ctx, _ev(pygame.KEYDOWN, pygame.K_z, pygame.KMOD_CTRL))  # undo: no tool
    inker_mode.handle_key(ctx, _ev(pygame.KEYUP, pygame.K_LCTRL, 0))
    assert state.tool == "brush"


def test_closing_the_last_tab_under_a_held_ctrl_restores_the_tool():
    ctx, state, tab = _quick_scene("q")
    inker_mode.handle_key(ctx, _ev(pygame.KEYDOWN, pygame.K_LCTRL, pygame.KMOD_CTRL))
    assert state.tool == "move"
    state.close(tab.uid)  # what Ctrl+W ends in
    inker_mode.handle_key(ctx, _ev(pygame.KEYUP, pygame.K_LCTRL, 0))
    assert (state.quick_key, state.quick_tool) == ("", "")
    state.add(inker_state.InkerDoc(doc=inker.Document.blank(8, 8), uid="r", title="R"))
    assert state.tool == "brush"
