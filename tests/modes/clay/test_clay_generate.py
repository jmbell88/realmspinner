"""``clay.generate``: the request/poll/land machinery for "Generate into the
current tab" -- Inker's own "Regenerate selection" shape (``inker/ui/panes/
bridge.py``), reused for a mesh instead of a patch of pixels.

A small, self-contained ctx rather than the real service: every door this
module opens (``create_job``, ``promote_to_model``, ``rerun_job``) is
monkeypatched, the same way ``tests/modes/inker/test_flourish_texture.py``
tests its own submit/poll/land trio, so nothing here needs real weights, a
real trellis exe or a real GLB on disk.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.service import errors as svc_errors
from realmspinner.service import jobs as svc_jobs
from realmspinner.service.files import MAX_CLAY_SOURCE_BYTES
from realmspinner.studio.modes.clay import generate as clay_generate
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.state import DEFAULT_FORM_3D, default_form_2d
from realmspinner.studio.tasks import Done

WORKER = "realmspinner-task-test"


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


class _Confirms:
    def __init__(self) -> None:
        self.pending: Any = None

    def ask(self, confirm: Any) -> None:
        self.pending = confirm


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
    """``submit`` runs the task inline and queues its ``Done`` for
    :meth:`land_all` -- ``test_flourish_texture.py``'s own ``_Ctx`` shape."""

    def __init__(self, tmp_path: Path) -> None:
        self.svc = _Svc(tmp_path)
        self.state = _AppState()
        self.settings = _Settings()
        self.toasts: list[tuple[str, str]] = []
        self.confirms = _Confirms()
        self.submitted: list[str] = []
        self._busy: set[str] = set()
        self._queue: list[Done] = []
        self.clay_view = None

    def toast(self, message: str, kind: str = "info", action: Any = None) -> None:
        self.toasts.append((message, kind))

    def submit(self, key: str, fn: Any, *args: Any, tag: Any = None, **kwargs: Any) -> bool:
        if key in self._busy:
            return False
        self.submitted.append(key)
        try:
            done = Done(key=key, result=fn(*args, **kwargs), tag=tag)
        except Exception as exc:  # noqa: BLE001 - the same failure a real pool reports
            done = Done(key=key, error=exc, message=str(exc), tag=tag)
        self._busy.add(key)
        self._queue.append(done)
        return True

    def land_all(self) -> None:
        while self._queue:
            done = self._queue.pop(0)
            self._busy.discard(done.key)
            if done.ok:
                clay_mode.on_task_done(self, done)
            else:
                clay_mode.on_task_failed(self, done)


class _ThreadedCtx(_Ctx):
    """:meth:`submit` runs on a real, joined worker thread -- so a spy on the
    decode can name the thread it ran on, ``test_frame_thread_doors.py``'s own
    ``_Threaded``."""

    def submit(self, key: str, fn: Any, *args: Any, tag: Any = None, **kwargs: Any) -> bool:
        if key in self._busy:
            return False
        self.submitted.append(key)
        box: dict[str, Any] = {}

        def go() -> None:
            try:
                box["result"] = fn(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                box["error"] = exc

        worker = threading.Thread(target=go, name=WORKER)
        worker.start()
        worker.join()
        done = (
            Done(key=key, error=box["error"], message=str(box["error"]), tag=tag)
            if "error" in box
            else Done(key=key, result=box.get("result"), tag=tag)
        )
        self._busy.add(key)
        self._queue.append(done)
        return True


def _tab(ctx: Any) -> Any:
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
    return clay_mode.adopt(ctx, doc, title="Scene")


def _small_incoming_doc() -> bd.ClayDoc:
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
    return doc


def _patch_landing(monkeypatch: pytest.MonkeyPatch, make_doc: Any = _small_incoming_doc) -> None:
    from realmspinner.kernels.mesh import glbimport

    monkeypatch.setattr(clay_mode, "_within_mesh_ceiling", lambda path: b"")
    monkeypatch.setattr(glbimport, "glb_to_claydoc", lambda data, name="Imported": make_doc())


def _through_preview(ctx: Any, tab: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Everything up to and including a ready preview: queue, land the queue
    result, mark the job done, poll. Shared setup for the tests below that
    start from "the reference is ready"."""
    monkeypatch.setattr(svc_jobs, "create_job", lambda svc, **kw: {"id": "ref1"})
    clay_generate.submit_text(ctx, tab, "a wooden barrel")
    ctx.land_all()
    ctx.svc.store.jobs["ref1"] = {"status": "done"}
    clay_generate.poll(ctx)


def _through_mesh_queued(
    ctx: Any, tab: Any, monkeypatch: pytest.MonkeyPatch, promote: Any
) -> None:
    _through_preview(ctx, tab, monkeypatch)
    monkeypatch.setattr(svc_jobs, "promote_to_model", promote)
    clay_generate.accept_reference(ctx, tab)
    ctx.land_all()


def test_text_generate_queues_a_reference_not_a_mesh(tmp_path, monkeypatch):
    calls: list[dict] = []

    def fake_create_job(svc, **kwargs):
        calls.append(kwargs)
        return {"id": "ref1"}

    monkeypatch.setattr(svc_jobs, "create_job", fake_create_job)
    ctx = _Ctx(tmp_path)
    tab = _tab(ctx)

    assert clay_generate.submit_text(ctx, tab, "a wooden barrel") is True
    ctx.land_all()

    assert len(calls) == 1
    assert calls[0]["kind"] == "text"
    assert calls[0]["output"] == "reference"
    assert calls[0]["prompt"] == "a wooden barrel"
    pending = ctx.state.clay.generate_pending
    assert pending is not None
    assert pending["stage"] == "reference"
    assert pending["reference_job_id"] == "ref1"
    assert tab.bg_busy  # the hint line says something


def test_accepting_the_reference_promotes_it_to_an_ordinary_model_job(tmp_path, monkeypatch):
    promote_calls: list[tuple[str, dict]] = []

    def fake_promote(svc, job_id, **kw):
        promote_calls.append((job_id, kw))
        return {"id": "mesh1"}

    ctx = _Ctx(tmp_path)
    tab = _tab(ctx)
    _through_mesh_queued(ctx, tab, monkeypatch, fake_promote)

    assert promote_calls[0][0] == "ref1"
    assert promote_calls[0][1]["rig"] is False
    assert promote_calls[0][1]["force"] is False
    pending = ctx.state.clay.generate_pending
    assert pending["stage"] == "mesh"
    assert pending["mesh_job_id"] == "mesh1"


def test_a_composition_refusal_offers_build_anyway(tmp_path, monkeypatch):
    def fake_promote(svc, job_id, **kw):
        if not kw.get("force"):
            raise svc_errors.Invalid("this reference cannot reconstruct")
        return {"id": "mesh1"}

    ctx = _Ctx(tmp_path)
    tab = _tab(ctx)
    _through_preview(ctx, tab, monkeypatch)
    # The reference stage's own measurement is what makes the refusal a
    # composition one -- ``promote_to_model`` refuses on exactly this.
    ctx.svc.store.jobs["ref1"]["params"] = {"reference_report": {"ok": False}}
    monkeypatch.setattr(svc_jobs, "promote_to_model", fake_promote)
    clay_generate.accept_reference(ctx, tab)
    ctx.land_all()

    pending = ctx.state.clay.generate_pending
    assert pending is not None
    assert pending["stage"] == "preview", "dropped back to the preview, not lost entirely"
    assert pending["force_offer"] is True
    # The real toast for *why* comes from the shell (``shell/tasks.py``
    # toasts every failed ``clay-`` task before routing it here, which this
    # harness does not reproduce); this module's own job is only to keep the
    # retry offer alive so the popup can show "Build anyway".

    assert clay_generate.accept_reference(ctx, tab, force=True) is True
    ctx.land_all()
    pending = ctx.state.clay.generate_pending
    assert pending["stage"] == "mesh"
    assert pending["mesh_job_id"] == "mesh1"


def test_a_refusal_force_cannot_bypass_does_not_offer_build_anyway(tmp_path, monkeypatch):
    """Only the composition check is bypassable by ``force``; a VRAM or
    weights refusal from the same door would fail again on the retry, so
    offering "Build anyway" for one is a button that can only fail."""

    def fake_promote(svc, job_id, **kw):
        raise svc_errors.Invalid("not enough free VRAM for this mesh")

    ctx = _Ctx(tmp_path)
    tab = _tab(ctx)
    _through_preview(ctx, tab, monkeypatch)
    ctx.svc.store.jobs["ref1"]["params"] = {"reference_report": {"ok": True}}
    monkeypatch.setattr(svc_jobs, "promote_to_model", fake_promote)
    clay_generate.accept_reference(ctx, tab)
    ctx.land_all()

    assert ctx.state.clay.generate_pending is None


def test_generate_lands_in_the_tab_it_was_asked_from_after_a_tab_switch(tmp_path, monkeypatch):
    _patch_landing(monkeypatch)
    ctx = _Ctx(tmp_path)
    tab_a = _tab(ctx)
    tab_b = _tab(ctx)
    ctx.state.clay.activate(tab_b.uid)
    assert ctx.state.clay.active_uid == tab_b.uid

    _through_mesh_queued(ctx, tab_a, monkeypatch, lambda svc, job_id, **kw: {"id": "mesh1"})
    ctx.svc.store.jobs["mesh1"] = {"status": "done"}
    clay_generate.poll(ctx)  # submits the decode
    ctx.land_all()  # runs it and lands

    assert ctx.state.clay.generate_pending is None
    assert len(tab_a.doc.objects) == 2, "landed in the tab it was asked from"
    assert len(tab_b.doc.objects) == 1, "the tab the user switched to is untouched"
    assert ctx.state.clay.active_uid == tab_b.uid, "active_uid never moved"


def test_a_generate_landing_in_a_closed_tab_is_refused_by_name(tmp_path, monkeypatch):
    _patch_landing(monkeypatch)
    ctx = _Ctx(tmp_path)
    tab = _tab(ctx)
    _through_mesh_queued(ctx, tab, monkeypatch, lambda svc, job_id, **kw: {"id": "mesh1"})
    ctx.svc.store.jobs["mesh1"] = {"status": "done"}
    clay_generate.poll(ctx)

    ctx.state.clay.close(tab.uid)
    ctx.land_all()

    assert ctx.state.clay.generate_pending is None
    assert any("closed" in m for m, _ in ctx.toasts)


def test_a_generate_landing_during_a_save_waits_instead_of_mutating(tmp_path, monkeypatch):
    _patch_landing(monkeypatch)
    ctx = _Ctx(tmp_path)
    tab = _tab(ctx)
    _through_mesh_queued(ctx, tab, monkeypatch, lambda svc, job_id, **kw: {"id": "mesh1"})
    ctx.svc.store.jobs["mesh1"] = {"status": "done"}

    tab.saving = True
    clay_generate.poll(ctx)  # submits and runs the decode
    ctx.land_all()  # land() sees the tab busy and defers

    before = len(tab.doc.objects)
    pending = ctx.state.clay.generate_pending
    assert pending is not None
    assert pending["stage"] == "landing"
    assert pending.get("deferred") is not None
    assert len(tab.doc.objects) == before

    tab.saving = False
    clay_generate.poll(ctx)  # retries, now that the tab is free

    assert ctx.state.clay.generate_pending is None
    assert len(tab.doc.objects) == before + 1


def test_a_refused_generate_does_not_unlock_a_tab_mid_save(tmp_path, monkeypatch):
    def fake_promote(svc, job_id, **kw):
        raise svc_errors.Invalid("no")

    ctx = _Ctx(tmp_path)
    tab = _tab(ctx)
    _through_preview(ctx, tab, monkeypatch)
    monkeypatch.setattr(svc_jobs, "promote_to_model", fake_promote)
    clay_generate.accept_reference(ctx, tab)

    # A real save happens to be in flight on this same tab when the
    # composition refusal comes back.
    tab.saving = True
    ctx.land_all()

    assert tab.saving is True


def test_a_generated_mesh_past_slow_triangles_asks_first(tmp_path, monkeypatch):
    incoming = _small_incoming_doc()

    def fake_decode(svc, job_id, tab_uid, group_name):
        return {
            "tab_uid": tab_uid,
            "doc": incoming,
            "triangles": 999_999,
            "incoming_bytes": 10,
            "group_name": group_name,
        }

    monkeypatch.setattr(clay_generate, "_decode_landing", fake_decode)
    ctx = _Ctx(tmp_path)
    tab = _tab(ctx)
    _through_mesh_queued(ctx, tab, monkeypatch, lambda svc, job_id, **kw: {"id": "mesh1"})
    ctx.svc.store.jobs["mesh1"] = {"status": "done"}
    clay_generate.poll(ctx)
    ctx.land_all()

    assert ctx.confirms.pending is not None
    assert "999,999" in ctx.confirms.pending.message
    assert len(tab.doc.objects) == 1, "not landed yet"
    assert ctx.state.clay.generate_pending is None, "only the confirm is left to answer"

    ctx.confirms.pending.on_confirm()
    assert len(tab.doc.objects) == 2, "confirming lands it"


def test_confirming_after_the_tab_closed_refuses(tmp_path, monkeypatch):
    incoming = _small_incoming_doc()

    def fake_decode(svc, job_id, tab_uid, group_name):
        return {
            "tab_uid": tab_uid,
            "doc": incoming,
            "triangles": 999_999,
            "incoming_bytes": 10,
            "group_name": group_name,
        }

    monkeypatch.setattr(clay_generate, "_decode_landing", fake_decode)
    ctx = _Ctx(tmp_path)
    tab = _tab(ctx)
    _through_mesh_queued(ctx, tab, monkeypatch, lambda svc, job_id, **kw: {"id": "mesh1"})
    ctx.svc.store.jobs["mesh1"] = {"status": "done"}
    clay_generate.poll(ctx)
    ctx.land_all()
    confirm = ctx.confirms.pending
    assert confirm is not None

    ctx.state.clay.close(tab.uid)
    confirm.on_confirm()

    assert any("closed" in m for m, _ in ctx.toasts)


def test_a_generate_that_would_make_the_document_too_big_to_reopen_is_refused(
    tmp_path, monkeypatch
):
    incoming = _small_incoming_doc()

    def fake_decode(svc, job_id, tab_uid, group_name):
        return {
            "tab_uid": tab_uid,
            "doc": incoming,
            "triangles": 10,
            "incoming_bytes": MAX_CLAY_SOURCE_BYTES,
            "group_name": group_name,
        }

    monkeypatch.setattr(clay_generate, "_decode_landing", fake_decode)
    ctx = _Ctx(tmp_path)
    tab = _tab(ctx)
    tab.rblk_bytes = MAX_CLAY_SOURCE_BYTES  # already measured, at the ceiling
    _through_mesh_queued(ctx, tab, monkeypatch, lambda svc, job_id, **kw: {"id": "mesh1"})
    ctx.svc.store.jobs["mesh1"] = {"status": "done"}
    clay_generate.poll(ctx)
    ctx.land_all()

    assert ctx.state.clay.generate_pending is None
    assert len(tab.doc.objects) == 1, "nothing landed"
    assert any("too large to reopen" in m for m, _ in ctx.toasts)


def test_the_generate_poll_reads_model_glb_only_on_a_task_thread(tmp_path, monkeypatch):
    from realmspinner.kernels.mesh import glbimport

    real = glbimport.glb_to_claydoc
    threads: list[str] = []

    def spy(data, name="Imported"):
        threads.append(threading.current_thread().name)
        return _small_incoming_doc()

    monkeypatch.setattr(clay_mode, "_within_mesh_ceiling", lambda path: b"")
    monkeypatch.setattr(glbimport, "glb_to_claydoc", spy)
    del real  # unused past documenting what was patched over

    ctx = _ThreadedCtx(tmp_path)
    tab = _tab(ctx)
    _through_mesh_queued(ctx, tab, monkeypatch, lambda svc, job_id, **kw: {"id": "mesh1"})
    ctx.svc.store.jobs["mesh1"] = {"status": "done"}

    clay_generate.poll(ctx)  # submits the decode -- ``_ThreadedCtx`` runs and
    # joins it on a real worker thread before ``submit`` returns, so the read
    # has already happened, on that thread, by this point.
    assert threads == [WORKER]
    assert len(tab.doc.objects) == 1, "not yet: nothing has landed on the frame thread"

    ctx.land_all()  # the frame-thread half: on_task_done applies the result

    assert threads == [WORKER], "still only the one read"
    assert len(tab.doc.objects) == 2
