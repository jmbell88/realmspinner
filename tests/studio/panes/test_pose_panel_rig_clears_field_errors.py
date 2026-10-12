"""Fix-pass-3: ``pose_panel``'s two Rig buttons (the automatic one and "Rig
manually") submitted without ``ctx.state.clear_field_errors()``, which
``stage_rig``'s identical buttons call first -- a stale red field from the last
refusal stayed up against a submit that never saw it. The draw needs a live imgui
frame, so this pins the source, the way ``tests/test_field_error_wiring.py`` does."""

from __future__ import annotations

import re
from pathlib import Path

import realmspinner.studio.panes.pose_panel as pose_panel


def test_both_pose_panel_rig_buttons_clear_field_errors_before_submitting():
    source = Path(pose_panel.__file__).read_text(encoding="utf-8")
    door = r"ctx\.submit\(\s*rig_key,\s*svc_rig\.create_rig"
    submits = [m.start() for m in re.finditer(door, source)]
    assert len(submits) == 2, "the Pose card has exactly the two Rig doors"
    for at in submits:
        before = source[:at].rsplit("disabled_button(", 1)[-1]
        assert "ctx.state.clear_field_errors()" in before, (
            "a Rig button must clear stale field errors before it submits"
        )
