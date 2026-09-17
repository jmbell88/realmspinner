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
    the bottom pane's current *reserve* (its own height plus 2026-09-16's
    drag handle, once expanded -- see ``bottom_pane.reserve``'s docstring),
    so none of them sit behind, overlap, or land on top of that handle.
    Asserted on source, the same way the manual-chapter prose-drift tests
    pin a claim against the module that makes it true -- these are one-line
    arithmetic edits, not behaviour a fake imgui context usefully exercises.
    """
    fps_source = inspect.getsource(overlay.fps_meter)
    assert "bottom_pane.reserve(ctx)" in fps_source

    progress_source = inspect.getsource(overlay.progress_card)
    assert "bottom_pane.reserve(ctx)" in progress_source

    toasts_source = inspect.getsource(widgets.toasts)
    assert "bottom_offset" in toasts_source

    card_source = inspect.getsource(tour._card)
    assert "bottom_pane.reserve(ctx)" in card_source
    card_pos_source = inspect.getsource(tour._card_pos)
    assert "bottom_offset" in card_pos_source

    # And main.py actually threads the toast offset through, rather than the
    # keyword existing on ``toasts`` with nothing supplying it.
    main_source = inspect.getsource(main_mod.App._overlays)
    assert "bottom_offset=" in main_source


def _familiar_config(tmp_path):
    from warlock.config import Config

    return Config(
        data_dir=tmp_path / "assets", db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe", trellis_models_dir=tmp_path / "models",
        t2i_model_root=tmp_path / "t2i-models",
        familiar_runtime_dir=tmp_path / "engine" / "llama",
        familiar_models_dir=tmp_path / "models" / "familiar",
    )


def test_reserve_adds_the_grip_only_once_expanded():
    """Collapsed, there is no handle above the pane, so reserve() must equal
    height() exactly -- the whole point of the 2026-09-16 drag handle is that
    it only exists once there is something to drag."""
    from types import SimpleNamespace

    ctx = SimpleNamespace(state=SimpleNamespace(mode="home"))
    assert bottom_pane.height(ctx) == bottom_pane.COLLAPSED_H
    assert bottom_pane.reserve(ctx) == bottom_pane.height(ctx)


def test_reserve_adds_the_splitter_grip_once_expanded(monkeypatch):
    """Expanded, ``reserve()`` must be taller than ``height()`` by exactly
    the splitter's own grip width -- ``layout.GRIP`` -- or the shell leaves
    too little room and the handle main.py now draws above the pane pushes
    its bottom edge past the window (the ``PICKER_FLOOR`` incident, in the
    other direction). ``height()`` reaches ``imgui.get_main_viewport()``
    once expanded, so this needs the shared real-but-unrendered context
    every other pane test in this repo builds one of (``tests/_ui_context``)
    rather than the bare ``SimpleNamespace`` the collapsed-only tests above
    get away with.
    """
    from types import SimpleNamespace

    from _ui_context import imgui_context

    from warlock.studio import familiar_ui, layout

    ctx = SimpleNamespace(
        state=SimpleNamespace(mode="home", familiar=familiar_ui.FamiliarUIState(expanded=True)),
        settings=SimpleNamespace(get=lambda key, default: default),
    )
    with imgui_context(monkeypatch) as imgui:
        # ``get_main_viewport().work_size`` only reflects ``io.display_size``
        # once a frame has actually started -- read before any ``new_frame()``
        # it is (0, 0), which clamps ``max_height`` down to ``COLLAPSED_H`` and
        # would make this assertion pass for the wrong reason.
        imgui.new_frame()
        try:
            assert bottom_pane.reserve(ctx) == bottom_pane.height(ctx) + layout.GRIP
        finally:
            imgui.end_frame()
            imgui.render()


def test_pane_height_defaults_to_expanded_h_and_round_trips_a_drag(tmp_path):
    """No prior drag reads back :data:`familiar_ui.EXPANDED_H`; a drag is
    persisted through ``ctx.settings`` and clamped on both write and read, so
    a stray large or tiny stored value cannot hand the pane an unusable size."""
    from types import SimpleNamespace

    from warlock.studio import familiar_ui
    from warlock.studio.settings import Settings

    settings = Settings.load(tmp_path)
    ctx = SimpleNamespace(settings=settings)

    assert familiar_ui.pane_height(ctx) == familiar_ui.EXPANDED_H

    familiar_ui.set_pane_height(ctx, 250.0)
    assert familiar_ui.pane_height(ctx) == 250.0

    familiar_ui.set_pane_height(ctx, 10_000.0)
    assert familiar_ui.pane_height(ctx) == familiar_ui.MAX_EXPANDED_H

    familiar_ui.set_pane_height(ctx, -50.0)
    assert familiar_ui.pane_height(ctx) == familiar_ui.MIN_EXPANDED_H


def test_familiar_state_is_missing_on_an_empty_home(tmp_path):
    """A fresh WARLOCK_HOME has none of Familiar's three rows -- the pane and
    the ✦ menu must both read this as "not installed", not silently pass an
    AttributeError up from a bottom pane that used to hardcode the sentence."""
    config = _familiar_config(tmp_path)
    bottom_pane._familiar_state_cache = None
    assert bottom_pane.familiar_state(config) == "missing"


def test_familiar_state_is_idle_once_every_row_is_present(tmp_path):
    """Once every ``models.FAMILIAR_MODELS`` row's files are on disk,
    ``familiar_state`` must flip to "idle" -- the pane's install sentence and
    the menu's disabled item both key off this, and before this test the pane
    said "isn't installed" even after a real download completed."""
    from warlock import models

    config = _familiar_config(tmp_path)
    for spec in models.FAMILIAR_MODELS.values():
        base = config.familiar_runtime_dir if spec.runtime else config.familiar_models_dir
        base.mkdir(parents=True, exist_ok=True)
        for name in spec.probe:
            (base / name).write_bytes(b"x")
    # Bypass the module's 2s cache: an earlier "missing" read (this test's own
    # sibling, or an app frame) can still be within the window here.
    bottom_pane._familiar_state_cache = None
    assert bottom_pane.familiar_state(config) == "idle"


def test_the_expanded_pane_renders_bubbles_the_handle_and_autoscrolls(monkeypatch, tmp_path):
    """A full render of the expanded pane: the drag handle, the padded and
    bubble-coloured transcript, and the auto-scroll check -- with no GL
    needed, the same "the assertion is the frame completing" idiom
    ``test_studio_smoke.py`` already uses for this pane's collapsed row. An
    unbalanced ``push_style_var``/``push_style_color`` or a bad draw-list
    call does not raise where it happens, it corrupts the draw stack and
    shows up later, so ``imgui.end()``/``render()`` completing clean is the
    real check that 2026-09-16's bubbles and padding paired every push.
    """
    from types import SimpleNamespace

    from _ui_context import imgui_context

    from warlock import models
    from warlock.familiar import threads as threads_mod
    from warlock.studio import familiar_ui
    from warlock.studio.settings import Settings

    config = _familiar_config(tmp_path)
    for spec in models.FAMILIAR_MODELS.values():
        base = config.familiar_runtime_dir if spec.runtime else config.familiar_models_dir
        base.mkdir(parents=True, exist_ok=True)
        for name in spec.probe:
            (base / name).write_bytes(b"x")
    bottom_pane._familiar_state_cache = None

    familiar_threads = threads_mod.Threads()
    familiar_threads.append(
        threads_mod.STUDIO, threads_mod.Turn(role="user", text="Make me a small stone hut.")
    )
    familiar_threads.append(
        threads_mod.STUDIO,
        threads_mod.Turn(role="familiar", text="Here is a plan for a small stone hut."),
    )

    ctx = SimpleNamespace(
        state=SimpleNamespace(mode="home", familiar=familiar_ui.FamiliarUIState(expanded=True)),
        settings=Settings.load(tmp_path),
        svc=SimpleNamespace(config=config),
        familiar_threads=familiar_threads,
    )

    with imgui_context(monkeypatch) as imgui:
        imgui.new_frame()
        imgui.set_next_window_size((1200, 900))
        imgui.begin("##shell-host")
        try:
            bottom_pane.draw(ctx)
        finally:
            imgui.end()
            imgui.end_frame()
            imgui.render()
        # Asserted inside the ``with``: the context manager destroys the
        # context on exit, so reading it after would test the teardown
        # instead of the render.
        assert imgui.get_current_context() is not None
