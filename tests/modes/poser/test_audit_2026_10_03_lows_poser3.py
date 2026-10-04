"""The 2026-10-03 audit's Low findings poser-48 and poser-52."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from realmspinner.service import Invalid
from realmspinner.service import jobs as svc_jobs
from realmspinner.service import rig as svc_rig

# --- poser-48: a rig.json naming a template the registry no longer has -------


def _rig_naming(svc, template: str) -> str:
    job_id = svc_jobs.create_job(svc, kind="text", prompt="a barrel")["id"]
    job_dir = svc.config.data_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"fake-glb")
    svc.store.set_status(job_id, "done")
    rig = {"template": template, "bones": [], "bounds": {"min": [-1] * 3, "max": [1] * 3}}
    (job_dir / "rig.json").write_text(json.dumps(rig), encoding="utf-8")
    (job_dir / "rig.glb").write_bytes(b"fake-rig")
    return job_id


def test_adjust_joints_refuses_a_rig_whose_template_is_not_in_the_registry(svc):
    job_id = _rig_naming(svc, "a_template_that_was_renamed")
    with pytest.raises(Invalid) as caught:
        svc_rig.adjust_joints(svc, job_id, {"bones": []})
    assert caught.value.field == "rig_template"


def test_edit_skeleton_refuses_a_rig_whose_template_is_not_in_the_registry(svc):
    job_id = _rig_naming(svc, "a_template_that_was_renamed")
    with pytest.raises(Invalid) as caught:
        svc_rig.edit_skeleton(svc, job_id, {"bones": []})
    assert caught.value.field == "rig_template"


# --- poser-52: the pixel report is write-once, so stop asking for it ----------


class _CountingStore:
    def __init__(self, rows):
        self._rows = rows
        self.calls = 0

    def list(self, limit=100, kind=None):
        self.calls += 1
        return self._rows[:limit]


def test_pixel_report_stops_querying_the_store_once_the_sheets_report_is_known():
    from realmspinner.studio.modes.poser import mode as poser_mode
    from realmspinner.studio.modes.poser.ui.panes import sheet as poser_sheet

    store = _CountingStore([{"params": {"sheet_id": "abc", "pixel_report": {"colors": 16}}}])
    ctx = SimpleNamespace(svc=SimpleNamespace(store=store))
    state = poser_mode.PoserState()
    state.sheet_id = "abc"
    assert poser_sheet._pixel_report(ctx, state) == {"colors": 16}
    assert store.calls == 1

    # Every later SHEETS_REFRESH window elapses; the report cannot have changed.
    for _ in range(3):
        state.pixel_report_next = 0.0
        assert poser_sheet._pixel_report(ctx, state) == {"colors": 16}
    assert store.calls == 1


def test_pixel_report_keeps_polling_while_the_sheets_report_is_not_yet_written():
    from realmspinner.studio.modes.poser import mode as poser_mode
    from realmspinner.studio.modes.poser.ui.panes import sheet as poser_sheet

    store = _CountingStore([])
    ctx = SimpleNamespace(svc=SimpleNamespace(store=store))
    state = poser_mode.PoserState()
    state.sheet_id = "abc"
    assert poser_sheet._pixel_report(ctx, state) == {}
    state.pixel_report_next = 0.0
    store._rows = [{"params": {"sheet_id": "abc", "pixel_report": {"colors": 8}}}]
    assert poser_sheet._pixel_report(ctx, state) == {"colors": 8}
    assert store.calls == 2
