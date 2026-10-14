"""The 2026-10-14 audit, finding create-panes-06: "Apply joint positions" on a
rig Clay wrote re-rigs the static mesh with Blender's automatic weights and
replaces the hand-built binding, with no warning. A Clay rig (``rig.json``
``source`` "clay") is asked about first; any other rig is not."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from realmspinner.studio.panes import pose_panel


class _Viewer:
    pose_mode = True

    def corrected_bones(self) -> dict[str, Any]:
        return {"hips": {"head": [0, 0, 1], "tail": [0, 0, 2]}}


class _Ctx:
    def __init__(self) -> None:
        self.submitted: list[tuple] = []
        self.asked: list[Any] = []
        self.svc = object()
        self.confirms = SimpleNamespace(ask=self.asked.append)

    def submit(self, key: str, fn: Any, *args: Any) -> bool:
        self.submitted.append((key, fn, args))
        return True


@pytest.fixture(autouse=True)
def _clean():
    pose_panel._RIG_SOURCE.clear()
    pose_panel._JOINTS_LANDED.clear()
    yield
    pose_panel._RIG_SOURCE.clear()
    pose_panel._JOINTS_LANDED.clear()


def test_adjust_joints_warns_before_replacing_a_clay_authored_rig():
    ctx = _Ctx()
    pose_panel._RIG_SOURCE["abc123abc123"] = "clay"
    pose_panel._apply_joints(ctx, "abc123abc123", _Viewer())
    assert ctx.submitted == [], "a Clay rig's binding must not be replaced before the user answers"
    assert len(ctx.asked) == 1
    confirm = ctx.asked[0]
    assert "binding" in confirm.message and "Clay" in confirm.message
    confirm.on_confirm()
    assert [k for k, *_ in ctx.submitted] == ["joints:abc123abc123"]


def test_adjust_joints_on_a_template_rig_still_queues_without_asking():
    ctx = _Ctx()
    pose_panel._RIG_SOURCE["abc123abc123"] = ""
    pose_panel._apply_joints(ctx, "abc123abc123", _Viewer())
    assert ctx.asked == []
    assert [k for k, *_ in ctx.submitted] == ["joints:abc123abc123"]


def test_clay_authored_reads_the_rig_source_of_that_job_only():
    pose_panel._RIG_SOURCE["a"] = "clay"
    assert pose_panel.clay_authored("a")
    assert not pose_panel.clay_authored("b")


def test_entering_pose_mode_records_the_rigs_source_for_the_apply_press():
    entry = pose_panel._Entering("abc123abc123", "rig.glb", None)
    entry.parsed = object()
    entry.rig = {"source": "clay", "template": "blank", "bones": []}
    entry.done.set()

    class V:
        pose_loading = entry
        pending = "rig.glb"

        def adopt_model(self, parsed, path):
            pass

        def enter_pose_mode(self, rig, job_id):
            return True

    ctx = SimpleNamespace(
        viewer=V(),
        load_presets=lambda t: None,
        refresh_rig_data=lambda: None,
        toast=lambda *a: None,
    )
    pose_panel.land_enter(ctx, {"id": "abc123abc123"})
    assert pose_panel.clay_authored("abc123abc123")
