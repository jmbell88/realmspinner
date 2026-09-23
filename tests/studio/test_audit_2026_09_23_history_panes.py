"""Regression tests for the 2026-09-23 audit's history-panes fixer.

Three findings, three panes:

- shell-02: Create's live Rig stage (``inspector._STAGE_SECTIONS["rig"]``)
  never drew ``history_panel``, though Manual 23 promises "Earlier meshes"
  there, below Surface texture, with no caveat -- and the Library's Rig &
  Pose tab (``_rig_tab``) already drew it. The mirror image of create-02
  (2026-09-07 audit), which fixed the gap in the other direction.
- create-05: ``retarget_panel._warn_stale`` only reassured that the replaced
  mesh was kept when undoing a prior remesh; ``optimize_job`` keeps it on
  every retarget, so the line must be unconditional, matching
  ``remesh_panel`` and ``texture_panel``.
- shell-03: ``QuitMixin._quit_summary`` named a busy download, export, pack
  or review sweep but not a busy ``retarget:`` or ``model-revert:`` task,
  either of which rewrites ``model.glb`` under the viewer.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from realmspinner.studio.panes import inspector
from realmspinner.studio.state import AppState


class FakeCtx:
    def __init__(self, svc: Any) -> None:
        self.svc = svc
        self.state = AppState()

    def submit(self, key: str, run: Any, *args: Any) -> bool:  # pragma: no cover
        raise AssertionError("no task should be submitted by a draw-routing test")


def _job(svc, job_id):
    return svc.store.get(job_id)


# --- shell-02: Create's Rig stage must offer Earlier meshes -----------------


def test_the_create_rig_stage_offers_earlier_meshes(svc, monkeypatch):
    """shell-02 (2026-09-23 audit): ``history_panel.draw`` was wired into the
    Library's Rig & Pose tab (``_rig_tab``) but never into
    ``inspector._STAGE_SECTIONS["rig"]``, so Create's own live Rig stage
    never showed it -- though Manual 23 promises it "in the inspector at the
    Rig stage, below Surface texture" with no such caveat.
    """
    from realmspinner.studio.panes import (
        history_panel,
        pose_panel,
        remesh_panel,
        retarget_panel,
        sheet_panel,
        texture_panel,
    )

    calls: list[str] = []
    ctx = FakeCtx(svc)
    ctx.state.create.stage = "rig"
    job_id = svc.store.create("image", "a chest", {}, stage="model", status="done")
    svc.job_dir(job_id).mkdir(parents=True, exist_ok=True)
    job = _job(svc, job_id)

    monkeypatch.setattr(inspector, "_weighting", lambda *_a: calls.append("weighting"))
    monkeypatch.setattr(inspector, "_bones", lambda *_a: calls.append("bones"))
    monkeypatch.setattr(inspector, "_deform_qa", lambda *_a: calls.append("deform_qa"))
    monkeypatch.setattr(retarget_panel, "draw", lambda *_a: calls.append("retarget"))
    monkeypatch.setattr(remesh_panel, "draw", lambda *_a: calls.append("remesh"))
    monkeypatch.setattr(texture_panel, "draw", lambda *_a: calls.append("retexture"))
    monkeypatch.setattr(history_panel, "draw", lambda *_a: calls.append("history"))
    monkeypatch.setattr(pose_panel, "draw", lambda *_a: calls.append("pose"))
    monkeypatch.setattr(sheet_panel, "draw", lambda *_a: calls.append("sheet"))

    inspector._stage_body(ctx, job)

    assert "history" in calls, "Create's Rig stage must offer Earlier meshes"
    # Manual 23: "below Surface texture".
    assert calls.index("retexture") < calls.index("history")


# --- create-05: the retarget panel's kept-mesh reassurance is unconditional -


def test_retarget_warns_the_current_mesh_is_kept_even_with_no_prior_remesh(monkeypatch):
    """create-05 (2026-09-23 audit): ``_warn_stale`` only said the replaced
    mesh was "kept" when a prior remesh was about to be discarded
    (``params["remesh"]``) -- but ``optimize_job`` snapshots the mesh a
    retarget is about to replace under Earlier meshes unconditionally
    (``pipelines.modelhistory``, 2026-09-22), the same guarantee
    ``remesh_panel`` and ``texture_panel`` already state with no caveat.
    """
    from realmspinner.studio.panes import retarget_panel

    calls: list[tuple] = []
    monkeypatch.setattr(
        retarget_panel.widgets, "muted_wrapped", lambda *a, **k: calls.append(a)
    )
    # No rig.glb, no params["remesh"]: the case that used to say nothing.
    job = {"id": "job-1", "files": ["source.glb", "model.glb"], "params": {}}

    retarget_panel._warn_stale(SimpleNamespace(), job)

    assert any("kept" in call[0].lower() for call in calls), (
        "a retarget with no prior remesh must still say the replaced mesh is kept"
    )


# --- shell-03: the quit summary must warn about a busy mesh rework ----------


def test_quit_summary_warns_while_a_mesh_rework_or_revert_is_busy():
    """shell-03 (2026-09-23 audit): ``_quit_summary`` named a busy download,
    export, pack install or review sweep, but a busy ``retarget:<uid>`` task
    or a ``model-revert:<uid>`` restore (``history_panel``'s own key) --
    either of which rewrites ``model.glb`` under the viewer -- raised no
    warning at all.
    """
    from realmspinner.studio import main as main_mod

    for key in ("retarget:aaaaaaaaaaaa", "model-revert:aaaaaaaaaaaa"):
        app = main_mod.App.__new__(main_mod.App)
        app.runtime = SimpleNamespace(current_job_id=None)
        app.app_ctx = SimpleNamespace(
            cache=SimpleNamespace(active=None),
            tasks=SimpleNamespace(busy_keys={key}),
        )

        summary = app._quit_summary()

        assert summary, f"a busy {key!r} task must not pass through silently"
