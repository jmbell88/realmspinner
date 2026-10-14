"""The 2026-10-14 audit, finding create-panes-06: the Rig stage's "Rig this mesh
again" replaced a rig Clay wrote with a template fit and no question."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from realmspinner.studio.panes import pose_panel, stage_rig


class _Ctx:
    def __init__(self) -> None:
        self.submitted: list[tuple] = []
        self.asked: list[Any] = []
        self.svc = object()
        self.confirms = SimpleNamespace(ask=self.asked.append)
        self.state = SimpleNamespace(form_3d={"rig_template": "biped"})

    def submit(self, key: str, fn: Any, *args: Any, **kwargs: Any) -> bool:
        self.submitted.append((key, fn, args, kwargs))
        return True


@pytest.fixture(autouse=True)
def _clean():
    pose_panel._RIG_SOURCE.clear()
    yield
    pose_panel._RIG_SOURCE.clear()


def test_rig_this_mesh_again_asks_before_replacing_a_clay_authored_rig():
    ctx = _Ctx()
    pose_panel._RIG_SOURCE["job1"] = "clay"
    stage_rig._rig(ctx, {"id": "job1"}, "rig:job1")
    assert ctx.submitted == [], "a hand-built rig must not be replaced before the user answers"
    assert len(ctx.asked) == 1
    assert "Clay" in ctx.asked[0].message
    ctx.asked[0].on_confirm()
    assert [k for k, *_ in ctx.submitted] == ["rig:job1"]
    assert ctx.submitted[0][3] == {"template": "biped"}


def test_rig_this_mesh_on_a_template_rig_or_a_new_mesh_queues_without_asking():
    ctx = _Ctx()
    pose_panel._RIG_SOURCE["job1"] = ""
    stage_rig._rig(ctx, {"id": "job1"}, "rig:job1")
    stage_rig._rig(ctx, {"id": "unseen"}, "rig:unseen")
    assert ctx.asked == []
    assert [k for k, *_ in ctx.submitted] == ["rig:job1", "rig:unseen"]
