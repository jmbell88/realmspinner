"""Regression tests for the 2026-09-23 audit's history-panes fixer.

Three findings, three panes:

- shell-02: Create's live inspector never drew ``history_panel`` at the stage
  Manual 23 promises "Earlier meshes" at, below Surface texture -- and the
  Library's Rig & Pose tab (``_rig_tab``) already drew it. That stage is now
  Mesh (2026-10-10), where the rest of the reworks live too.
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


def test_the_create_mesh_stage_offers_every_rework_and_earlier_meshes(svc, monkeypatch):
    """shell-02 (2026-09-23 audit) put Earlier meshes on the Rig stage; the
    2026-10-10 Create parity work moved the whole run of mesh reworks to the Mesh
    stage, because a prop never visits Rig ("props skip Rig") and so never met a
    remesh, a re-texture or a way back. Rig keeps only what is about the rig.
    """
    from realmspinner.studio.panes import (
        history_panel,
        remesh_panel,
        reorient_panel,
        retarget_panel,
        texture_panel,
    )

    def run(stage):
        calls: list[str] = []
        ctx = FakeCtx(svc)
        ctx.state.create.stage = stage
        job_id = svc.store.create("image", "a chest", {}, stage="model", status="done")
        svc.job_dir(job_id).mkdir(parents=True, exist_ok=True)
        job = _job(svc, job_id)
        for name in ("_lineage", "_settings", "_reference", "_pixel", "_seam", "_quality",
                     "_verdict", "_weighting", "_bones", "_deform_qa"):
            monkeypatch.setattr(inspector, name, lambda *_a, n=name: calls.append(n))
        for label, module in (
            ("reorient", reorient_panel), ("retarget", retarget_panel),
            ("remesh", remesh_panel), ("retexture", texture_panel), ("history", history_panel),
        ):
            monkeypatch.setattr(module, "draw", lambda *_a, n=label: calls.append(n))
        inspector._stage_body(ctx, job)
        return calls

    mesh = run("mesh")
    assert mesh[-5:] == ["reorient", "retarget", "remesh", "retexture", "history"], (
        "the Mesh stage must end with the reworks, Earlier meshes last (below Surface texture)"
    )
    rig = run("rig")
    assert rig == ["_weighting", "_bones", "_deform_qa"], (
        "the Rig stage keeps what is about the rig and none of the mesh reworks"
    )


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
