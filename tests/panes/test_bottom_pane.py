"""T0 of the Familiar programme: the bottom pane that replaced the flat
24 dp status bar, and the overlays anchored above it.

``bottom_pane.max_height`` is pure arithmetic (no imgui, no ``ctx``), which is
what lets these be asserted without a window -- the same shape as
``status_bar.items``' own tests.
"""

from __future__ import annotations

import inspect

from warlock.studio import main as main_mod
from warlock.studio import muse_brief, widgets
from warlock.studio.panes import bottom_pane, muse_player, overlay, tour


def test_the_bottom_pane_never_exceeds_a_quarter_of_the_window():
    """Whatever chrome a mode reserves above it, the grown pane is capped at
    a quarter of the window -- the ceiling half of :func:`bottom_pane.max_height`'s
    two-way clamp."""
    for window_h in (700.0, 900.0, 1400.0, 2000.0):
        for mode_chrome in (0.0, 100.0, 418.0):
            grown = bottom_pane.max_height(window_h, mode_chrome)
            assert grown <= 0.25 * window_h + 1e-9, (window_h, mode_chrome, grown)


def test_the_grown_pane_leaves_muse_its_body_at_min_size():
    """At Muse's own ``main.MIN_SIZE`` (1100x700), a naive quarter-of-window
    grown height (175) would leave Muse's results list under the app's menu
    bar and Muse's own brief bar + player strip only ~80 design pixels of
    body -- far under a usable floor. ``max_height`` must cap the grown
    height well short of that quarter so :data:`bottom_pane.MIN_CENTER_HEIGHT`
    of body survives instead.
    """
    window_w, window_h = main_mod.MIN_SIZE
    assert (window_w, window_h) == (1100, 700)

    mode_chrome = muse_brief.BAR_H + muse_player.STRIP_H
    quarter = 0.25 * window_h

    grown = bottom_pane.max_height(float(window_h), mode_chrome)

    # The naive figure the T0 spec itself flagged as too thin: menu bar +
    # Muse's chrome + a full quarter-window pane leaves ~80 dp of body.
    naive_body = window_h - bottom_pane.MENU_BAR_H - mode_chrome - quarter
    assert 70.0 <= naive_body <= 90.0, naive_body

    # The capped grown height must be well under that naive quarter...
    assert grown < quarter
    # ...and must leave the floor this module names, not a hand-picked number
    # re-derived in the test.
    body_left = window_h - bottom_pane.MENU_BAR_H - mode_chrome - grown
    assert body_left >= bottom_pane.MIN_CENTER_HEIGHT - 1e-6


def test_collapsed_height_never_changes_with_the_mode():
    """T0 wires no model, so the pane is always one row tall regardless of
    which mode or how tall the window is."""
    from types import SimpleNamespace

    for mode in ("home", "muse", "inker", "clay"):
        ctx = SimpleNamespace(state=SimpleNamespace(mode=mode))
        assert bottom_pane.height(ctx) == bottom_pane.COLLAPSED_H


def test_toasts_progress_and_tour_cards_sit_above_the_bottom_pane():
    """Each of the four bottom-anchored overlays must offset its anchor by
    the bottom pane's current height, so none of them sit behind or overlap
    it. Asserted on source, the same way the manual-chapter prose-drift tests
    pin a claim against the module that makes it true -- these are one-line
    arithmetic edits, not behaviour a fake imgui context usefully exercises.
    """
    fps_source = inspect.getsource(overlay.fps_meter)
    assert "bottom_pane.height(ctx)" in fps_source

    progress_source = inspect.getsource(overlay.progress_card)
    assert "bottom_pane.height(ctx)" in progress_source

    toasts_source = inspect.getsource(widgets.toasts)
    assert "bottom_offset" in toasts_source

    card_source = inspect.getsource(tour._card)
    assert "bottom_pane.height(ctx)" in card_source
    card_pos_source = inspect.getsource(tour._card_pos)
    assert "bottom_offset" in card_pos_source

    # And main.py actually threads the toast offset through, rather than the
    # keyword existing on ``toasts`` with nothing supplying it.
    main_source = inspect.getsource(main_mod.App._overlays)
    assert "bottom_offset=" in main_source
