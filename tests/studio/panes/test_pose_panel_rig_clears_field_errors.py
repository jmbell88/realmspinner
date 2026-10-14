"""Fix-pass-3: ``pose_panel``'s Rig button submitted without
``ctx.state.clear_field_errors()``, which ``stage_rig``'s identical button calls
first -- a stale red field from the last refusal stayed up against a submit that
never saw it. (The second, "Rig manually", became the Clay hand-off
"Rig by hand in Clay" and submits nothing.) The draw needs a live imgui frame, so
this pins the source, the way ``tests/test_field_error_wiring.py`` does."""

from __future__ import annotations

import re
from pathlib import Path

import realmspinner.studio.panes.pose_panel as pose_panel


def test_the_pose_panel_rig_button_clears_field_errors_before_submitting():
    source = Path(pose_panel.__file__).read_text(encoding="utf-8")
    door = r"ctx\.submit\(\s*rig_key,\s*svc_rig\.create_rig"
    submits = [m.start() for m in re.finditer(door, source)]
    assert len(submits) == 1, "the Pose card has exactly one Rig door that submits"
    for at in submits:
        before = source[:at].rsplit("disabled_button(", 1)[-1]
        assert "ctx.state.clear_field_errors()" in before, (
            "a Rig button must clear stale field errors before it submits"
        )
