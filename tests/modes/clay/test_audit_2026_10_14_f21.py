"""Regression test for the 2026-10-14 audit's shell-chrome-01.

Clay's Pose mode binds Space, Home, End, Left and Right itself (``_pose_key``)
but Clay was never in ``NAV_KEY_MODES``, so imgui's keyboard navigation also
received each press: Space stopping a clip could press the focused Play button
and Left/Right stepped the focus ring as well as the playhead.
"""

from __future__ import annotations

from types import SimpleNamespace

from realmspinner.kernels.mesh import document as bd
from realmspinner.studio import modes
from realmspinner.studio.modes.clay import mode as clay_mode


def _ctx_state(*, pose: bool, tabs: int = 1) -> SimpleNamespace:
    state = clay_mode.ClayState()
    for _ in range(tabs):
        doc = bd.ClayDoc()
        doc.pose_mode = pose
        state.add(clay_mode.ClayTab(doc=doc))
    return SimpleNamespace(mode="clay", clay=state)


def test_clay_reserves_the_arrows_and_space_while_a_document_is_in_pose_mode() -> None:
    """Fails against the unfixed code with:
    ``assert modes.reserves_nav_keys('clay', app_state)`` -> False (``"clay"`` was
    not in ``NAV_KEY_MODES``, so imgui nav took Space and the arrows too).
    """
    posing = _ctx_state(pose=True)
    assert modes.reserves_nav_keys("clay", posing)

    # Only while posing: in Object / element modes Clay binds none of those keys
    # and the arrows keep walking the focus ring like anywhere else.
    editing = _ctx_state(pose=False)
    assert not modes.reserves_nav_keys("clay", editing)

    # No document, and no Clay state at all (the app before Clay is first opened).
    empty = SimpleNamespace(mode="clay", clay=clay_mode.ClayState())
    assert not modes.reserves_nav_keys("clay", empty)
    assert not modes.reserves_nav_keys("clay", SimpleNamespace(mode="clay", clay=None))


def test_the_reservation_follows_the_active_tab_not_any_tab() -> None:
    state = _ctx_state(pose=False, tabs=2)
    first, second = state.clay.docs
    second.doc.pose_mode = True
    state.clay.active_uid = first.uid
    assert not modes.reserves_nav_keys("clay", state)
    state.clay.active_uid = second.uid
    assert modes.reserves_nav_keys("clay", state)
