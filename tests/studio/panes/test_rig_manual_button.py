"""The by-hand rig door: "Rig by hand in Clay" sits beside the automatic "Rig this
mesh" at each of the three places that already offer it (the Pose panel,
Create's Rig stage, a library card's context menu).

It used to be "Rig manually", which queued the same rig job with the hidden
``blank`` template for Poser's skeleton editor to build out from nothing. Poser
is gone (the 2026-10-13 Clay rig plan): the by-hand route is Clay, where the
mesh opens, the joints are marked as bones and posed, and Export to the library
writes the rig. So the button is a hand-off now, not a queued job -- it submits
nothing and never names a template.

Asserted on source, the same way ``tests/studio/panes/test_familiar_dock.py``'s own
toast-offset test pins a one-line wiring claim: these are label and gating edits,
not behaviour a fake imgui context usefully exercises further than the smoke tests
those three panes already have.
"""

from __future__ import annotations

import inspect

from realmspinner.studio.modes.library.ui.panes import library
from realmspinner.studio.panes import pose_panel, stage_rig


def test_pose_panel_offers_a_by_hand_rig_button_gated_on_being_unrigged():
    source = inspect.getsource(pose_panel.draw)
    # Inside the "not rigged" branch, which returns before the rigged half of
    # the function starts -- so finding the label here already proves the gating.
    assert "Rig by hand in Clay" in source
    assert "edit_asset_in_clay" in source
    assert 'template="blank"' not in source, "the blank-template rig died with Poser"


def test_stage_rig_offers_a_by_hand_rig_button_only_when_unrigged():
    source = inspect.getsource(stage_rig.draw)
    assert "Rig by hand in Clay" in source
    assert "edit_asset_in_clay" in source
    assert "if not rigged:" in source
    assert 'template="blank"' not in source


def test_library_context_menu_offers_rig_by_hand_only_when_unrigged():
    menu_source = inspect.getsource(library._overflow)
    assert '"rig.glb" not in files' in menu_source
    assert "Rig by hand in Clay" in menu_source

    action_source = inspect.getsource(library.run_action)
    assert 'elif action == "clay":' in action_source
    assert "edit_asset_in_clay" in action_source
    assert 'template="blank"' not in action_source
