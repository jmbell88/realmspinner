"""Poser's manual-rig bootstrap (2026-09-16): "Rig manually" sits beside the
automatic "Rig this mesh" at each of the three places that already offer it
(the Pose panel, Create's Rig stage, a library card's context menu), and
submits the same queued rig job with the hidden ``blank`` template instead of
the config default -- see ``rigging.py``'s ``blank.json`` and
``test_rig_api.py::test_the_blank_template_is_a_real_template_hidden_from_the_catalogue``
for why a real (if minimal) Blender pass is what backs it, rather than
host-side coordinate math.

Asserted on source, the same way ``tests/panes/test_bottom_pane.py``'s own
toast-offset test pins a one-line wiring claim: these are template-string and
gating edits, not behaviour a fake imgui context usefully exercises further
than the smoke tests those three panes already have.
"""

from __future__ import annotations

import inspect

from warlock.studio.panes import library, pose_panel, stage_rig


def test_pose_panel_offers_a_manual_rig_button_gated_on_being_unrigged():
    source = inspect.getsource(pose_panel.draw)
    assert 'template="blank"' in source
    # Both buttons live inside the same "not rigged" branch -- pose_panel.draw
    # returns out of it before the rigged half of the function even starts,
    # so finding the string anywhere in this function's source already
    # proves the gating; this also pins the literal button label.
    assert "Rig manually" in source


def test_stage_rig_offers_a_manual_rig_button_only_when_unrigged():
    source = inspect.getsource(stage_rig.draw)
    assert 'template="blank"' in source
    assert "Rig manually" in source
    assert "if not rigged:" in source


def test_library_context_menu_offers_rig_manually_only_when_unrigged():
    menu_source = inspect.getsource(library._overflow)
    assert '"rig.glb" not in files' in menu_source
    assert "Rig manually" in menu_source

    action_source = inspect.getsource(library.run_action)
    assert 'elif action == "rig_manual":' in action_source
    assert 'template="blank"' in action_source
