"""The 2026-10-14 audit's residue round (fixer R): the rig panes without Blender.

shell-29 (finding F33): "Rig by hand in Clay" does not need Blender -- only the
template fit does -- but the Pose panel and the Rig stage both returned before drawing
it on a machine without Blender, so such a machine had no way to a rig at all.
create-panes-06: "Rig this mesh again" asks before replacing a rig Clay wrote, and
that needs the rig's source to be known for a job the Pose editor was never opened on.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from realmspinner.studio.panes import pose_panel, stage_rig


class _Recorder:
    """Stands in for ``widgets`` / ``imgui`` / ``manual_render``: records what is drawn."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def __getattr__(self, name: str):
        def draw(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return False

        return draw

    def labels(self, name: str) -> list[str]:
        return [str(args[0]) for called, args, _kw in self.calls if called == name and args]


def _ctx(*, rigging_available: bool) -> SimpleNamespace:
    return SimpleNamespace(
        rigging_available=rigging_available,
        rig_templates=[],
        rig_default="humanoid",
        viewer=None,
        busy=lambda key: False,
        job=lambda: {"id": "j1", "files": ["model.glb"]},
        state=SimpleNamespace(form_3d={}, clear_field_errors=lambda: None),
    )


@pytest.fixture
def rec(monkeypatch):
    recorder = _Recorder()
    for module in (stage_rig, pose_panel):
        monkeypatch.setattr(module, "widgets", recorder)
        monkeypatch.setattr(module, "imgui", recorder)
        monkeypatch.setattr(module, "manual_render", recorder)
    return recorder


def test_the_rig_stage_offers_rig_by_hand_in_clay_without_blender(rec):
    stage_rig.draw(_ctx(rigging_available=False))

    assert rec.labels("ghost_button") == ["Rig by hand in Clay"]
    # The template fit is Blender's, and is neither offered nor priced.
    assert rec.labels("disabled_button") == []
    assert rec.labels("cost_note") == []


def test_the_rig_stage_still_offers_both_doors_with_blender(rec):
    stage_rig.draw(_ctx(rigging_available=True))

    assert rec.labels("disabled_button") == ["Rig this mesh"]
    assert rec.labels("ghost_button") == ["Rig by hand in Clay"]
    assert len(rec.labels("cost_note")) == 1


def test_the_pose_panel_offers_rig_by_hand_in_clay_without_blender(rec):
    pose_panel.draw(
        _ctx(rigging_available=False), {"id": "j1", "files": ["model.glb"]}, hosted=True
    )

    assert rec.labels("ghost_button") == ["Rig by hand in Clay"]
    assert rec.labels("disabled_button") == []
    assert rec.labels("cost_note") == []
    # A sentence in a 300 dp sidebar wraps, so it is drawn with ``muted_wrapped``.
    assert any("Blender" in text for text in rec.labels("muted_wrapped"))


def test_the_pose_panel_still_offers_both_doors_with_blender(rec):
    pose_panel.draw(_ctx(rigging_available=True), {"id": "j1", "files": ["model.glb"]}, hosted=True)

    assert rec.labels("disabled_button") == ["Rig this mesh"]
    assert rec.labels("ghost_button") == ["Rig by hand in Clay"]


# --- create-panes-06: the source is recorded by the poses landing -------------------


def test_the_poses_landing_records_a_clay_rig_for_the_rig_again_confirm():
    from realmspinner.studio import main as main_mod

    pose_panel._RIG_SOURCE.pop("j-clay", None)
    preview: dict = {}
    app = SimpleNamespace(
        app_ctx=SimpleNamespace(state=SimpleNamespace(preview=preview, selected="j-clay"))
    )
    done = SimpleNamespace(
        key="poses:j-clay", ok=True, result={"poses": [], "bones": ["hips"], "source": "clay"}
    )
    try:
        assert not pose_panel.clay_authored("j-clay")

        main_mod.App._on_task_done(app, done)

        assert pose_panel.clay_authored("j-clay"), "the Pose editor was never opened for this job"
        assert preview["bones"] == ["hips"]
    finally:
        pose_panel._RIG_SOURCE.pop("j-clay", None)


def test_the_poses_landing_for_another_selection_records_nothing():
    from realmspinner.studio import main as main_mod

    pose_panel._RIG_SOURCE.pop("j-old", None)
    app = SimpleNamespace(
        app_ctx=SimpleNamespace(state=SimpleNamespace(preview={}, selected="j-new"))
    )
    done = SimpleNamespace(key="poses:j-old", ok=True, result={"poses": [], "source": "clay"})

    main_mod.App._on_task_done(app, done)

    assert not pose_panel.clay_authored("j-old")


def test_the_side_data_task_carries_the_rigs_source_with_the_poses(monkeypatch):
    from realmspinner.service import rig as svc_rig
    from realmspinner.studio.shell import tasks

    monkeypatch.setattr(svc_rig, "list_poses", lambda svc, job_id: {"bones": ["hips"], "poses": []})
    monkeypatch.setattr(svc_rig, "get_rig", lambda svc, job_id: {"source": "clay", "bones": []})
    assert tasks._read_rig_side_data(object(), "j1") == {
        "bones": ["hips"],
        "poses": [],
        "source": "clay",
    }

    def unreadable(svc, job_id):
        raise ValueError("no")

    monkeypatch.setattr(svc_rig, "get_rig", unreadable)
    # An unreadable rig.json costs the confirm, never the pose list.
    assert tasks._read_rig_side_data(object(), "j1")["source"] == ""
