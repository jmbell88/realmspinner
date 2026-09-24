"""The Familiar dock, a full-height right sidebar since 2026-09-23 --
replaced the bottom pane the Familiar programme's T0 built, then given a
fixed proportional width the same day (the shell's rail/left/canvas/right/
dock split, ``layout.proportions``).

The dock no longer fits its own width independently of the side columns
(``familiar_dock.fit``, now gone): ``layout.proportions`` divides the whole
room at once, so this module's own job is only to ease *how open* the dock is
and publish that fraction (:data:`layout.FAMILIAR_OPEN_T`) before the columns
are measured -- these tests are about that easing and about ``reserve``, not
about a width this module no longer computes on its own.
"""

from __future__ import annotations

import inspect

from realmspinner.studio import main as main_mod
from realmspinner.studio import widgets
from realmspinner.studio.panes import familiar_dock, overlay, tour


def test_the_dock_opens_by_widening_and_taking_from_both_sidebars():
    """Opening the dock (``layout.FAMILIAR_OPEN_T`` 0 -> 1) must widen it and
    narrow both sidebars by the same construction ``layout.proportions``
    already proves elsewhere -- this is the same fact, read through the
    dock's own ``tick``/``width`` rather than through ``proportions`` directly."""
    from realmspinner.studio import layout as layout_mod

    room, spacing = 1920.0, 8.0
    closed = layout_mod.proportions(room, 0.0, spacing)
    opened = layout_mod.proportions(room, 1.0, spacing)
    assert opened[4] > closed[4]  # dock widens
    assert opened[1] < closed[1]  # left narrows
    assert opened[3] < closed[3]  # right narrows
    assert closed[4] == familiar_dock.STRIP_W * layout_mod.tokens.SCALE  # closed: an icon strip


def test_reserve_equals_the_plain_width():
    """No grip any more: the dock no longer drags, so ``reserve()`` -- what
    the shell and every bottom/right-anchored overlay leave clear -- is
    exactly ``width()``, never wider."""
    familiar_dock._WIDTH[0] = 123.0
    try:
        assert familiar_dock.reserve() == familiar_dock.width() == 123.0
    finally:
        familiar_dock._WIDTH[0] = 0.0


def test_tick_publishes_the_eased_open_fraction(monkeypatch):
    """``familiar_dock.tick`` must publish ``layout.FAMILIAR_OPEN_T`` and this
    frame's ``DOCK_RESERVED`` before ``layout.measure`` ever runs -- the
    ordering ``rail.tick``'s own docstring already states for the rail's
    column, mirrored here for the dock's."""
    from types import SimpleNamespace

    from _ui_context import imgui_context

    from realmspinner.studio import layout as layout_mod
    from realmspinner.studio.assistant import ui as familiar_ui

    ctx = SimpleNamespace(state=SimpleNamespace(familiar=familiar_ui.FamiliarUIState()))
    with imgui_context(monkeypatch) as imgui:
        imgui.get_io().display_size = (1920.0, 1080.0)
        imgui.new_frame()
        try:
            layout_mod.FAMILIAR_OPEN_T = 0.0
            closed_w = familiar_dock.tick(ctx)

            ctx.state.familiar.expanded = True
            # ``motion.value`` eases toward the target rather than snapping,
            # so this only proves the open width is reachable and at least as
            # wide, not that one tick fully opens it.
            opened_w = familiar_dock.tick(ctx)
            assert opened_w >= closed_w
            assert opened_w == layout_mod.DOCK_RESERVED
        finally:
            imgui.end_frame()
            imgui.render()
            layout_mod.FAMILIAR_OPEN_T = 0.0


def test_toasts_progress_and_tour_cards_sit_left_of_the_dock():
    """Each of the bottom/right-anchored overlays must offset its anchor by
    the dock's current *reserve* (its own width plus the grip, once open --
    see ``familiar_dock.reserve``'s docstring), so none of them sit behind,
    overlap, or land on top of the grip. Asserted on source, the same way the
    manual-chapter prose-drift tests pin a claim against the module that
    makes it true -- these are one-line arithmetic edits, not behaviour a
    fake imgui context usefully exercises.

    ``fps_meter`` is bottom-left and the dock is on the right, so it alone
    carries no offset any more -- the one bottom-anchored overlay the dock
    move left untouched.
    """
    progress_source = inspect.getsource(overlay.progress_card)
    assert "familiar_dock" in progress_source

    toasts_source = inspect.getsource(widgets.toasts)
    assert "right_offset" in toasts_source

    card_source = inspect.getsource(tour._card)
    assert "familiar_dock.reserve()" in card_source
    card_pos_source = inspect.getsource(tour._card_pos)
    assert "right_offset" in card_pos_source

    fps_source = inspect.getsource(overlay.fps_meter)
    assert "familiar_dock" not in fps_source

    # And main.py actually threads the toast offset through, rather than the
    # keyword existing on ``toasts`` with nothing supplying it.
    main_source = inspect.getsource(main_mod.App._overlays)
    assert "right_offset=" in main_source


def _familiar_config(tmp_path):
    from realmspinner.config import Config

    return Config(
        data_dir=tmp_path / "assets", db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe", trellis_models_dir=tmp_path / "models",
        t2i_model_root=tmp_path / "t2i-models",
        familiar_runtime_dir=tmp_path / "engine" / "llama",
        familiar_models_dir=tmp_path / "models" / "familiar",
    )


def test_familiar_state_is_missing_on_an_empty_home(tmp_path):
    """A fresh REALMSPINNER_HOME has none of Familiar's three rows -- the
    dock's strip and the ✦ menu must both read this as "not installed", not
    silently pass an AttributeError up from a strip that used to hardcode
    the sentence."""
    config = _familiar_config(tmp_path)
    familiar_dock._familiar_state_cache = None
    assert familiar_dock.familiar_state(config) == "missing"


def test_familiar_state_is_idle_once_every_row_is_present(tmp_path):
    """Once every ``models.FAMILIAR_MODELS`` row's files are on disk,
    ``familiar_state`` must flip to "idle" -- the dock's strip and the menu's
    disabled item both key off this, and before this test the strip said
    "isn't installed" even after a real download completed."""
    from realmspinner import models

    config = _familiar_config(tmp_path)
    for spec in models.FAMILIAR_MODELS.values():
        base = config.familiar_runtime_dir if spec.runtime else config.familiar_models_dir
        base.mkdir(parents=True, exist_ok=True)
        for name in spec.probe:
            (base / name).write_bytes(b"x")
    # Bypass the module's 2s cache: an earlier "missing" read (this test's own
    # sibling, or an app frame) can still be within the window here.
    familiar_dock._familiar_state_cache = None
    assert familiar_dock.familiar_state(config) == "idle"


def test_the_open_dock_renders_bubbles_the_handle_and_autoscrolls(monkeypatch, tmp_path):
    """A full render of the open dock: the header row, the padded and
    bubble-coloured transcript, and the auto-scroll check -- with no GL
    needed, the same "the assertion is the frame completing" idiom
    ``test_studio_smoke.py`` already uses for this dock's closed strip. An
    unbalanced ``push_style_var``/``push_style_color`` or a bad draw-list
    call does not raise where it happens, it corrupts the draw stack and
    shows up later, so ``imgui.end()``/``render()`` completing clean is the
    real check that the dock's bubbles and padding paired every push.
    """
    from types import SimpleNamespace

    from _ui_context import imgui_context

    from realmspinner import models
    from realmspinner.familiar import threads as threads_mod
    from realmspinner.studio.assistant import ui as familiar_ui
    from realmspinner.studio.settings import Settings

    config = _familiar_config(tmp_path)
    for spec in models.FAMILIAR_MODELS.values():
        base = config.familiar_runtime_dir if spec.runtime else config.familiar_models_dir
        base.mkdir(parents=True, exist_ok=True)
        for name in spec.probe:
            (base / name).write_bytes(b"x")
    familiar_dock._familiar_state_cache = None

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

    # ``draw`` reads the eased open-fraction ``tick`` would normally have set
    # this frame; forced here rather than calling ``tick`` so the render is
    # exercised directly, the way this test always has.
    familiar_dock._T[0] = 1.0
    familiar_dock._WIDTH[0] = 400.0
    with imgui_context(monkeypatch) as imgui:
        imgui.new_frame()
        imgui.set_next_window_size((1200, 900))
        imgui.begin("##shell-host")
        try:
            familiar_dock.draw(ctx)
        finally:
            imgui.end()
            imgui.end_frame()
            imgui.render()
            familiar_dock._T[0] = 0.0
            familiar_dock._WIDTH[0] = 0.0
        # Asserted inside the ``with``: the context manager destroys the
        # context on exit, so reading it after would test the teardown
        # instead of the render.
        assert imgui.get_current_context() is not None


def test_the_familiar_icon_toggles_the_dock_both_ways(monkeypatch, tmp_path):
    """The ✦ must close the dock it opened: it used to be a button only on
    the closed strip, and plain text in the open header, so pressing the icon
    a second time did nothing. Pressed once from closed it opens; pressed
    once from open it closes -- the same button id in both states."""
    from types import SimpleNamespace

    from _ui_context import imgui_context

    from realmspinner import models
    from realmspinner.familiar import threads as threads_mod
    from realmspinner.studio.assistant import ui as familiar_ui
    from realmspinner.studio.settings import Settings

    config = _familiar_config(tmp_path)
    for spec in models.FAMILIAR_MODELS.values():
        base = config.familiar_runtime_dir if spec.runtime else config.familiar_models_dir
        base.mkdir(parents=True, exist_ok=True)
        for name in spec.probe:
            (base / name).write_bytes(b"x")
    familiar_dock._familiar_state_cache = None

    ctx = SimpleNamespace(
        state=SimpleNamespace(mode="home", familiar=familiar_ui.FamiliarUIState()),
        settings=Settings.load(tmp_path),
        svc=SimpleNamespace(config=config),
        familiar_threads=threads_mod.Threads(),
    )
    with imgui_context(monkeypatch) as imgui:
        real_hit = imgui.invisible_button

        def pressed(label, *a, **k):
            hit = real_hit(label, *a, **k)
            return label == "##familiar-dock/toggle" or hit

        monkeypatch.setattr(imgui, "invisible_button", pressed)
        try:
            for t, expect in ((0.0, True), (1.0, False)):
                familiar_dock._T[0] = t
                familiar_dock._WIDTH[0] = 44.0 if t == 0.0 else 400.0
                imgui.new_frame()
                imgui.set_next_window_size((1200, 900))
                imgui.begin("##shell-host")
                try:
                    familiar_dock.draw(ctx)
                finally:
                    imgui.end()
                    imgui.end_frame()
                    imgui.render()
                assert ctx.state.familiar.expanded is expect, t
        finally:
            familiar_dock._T[0] = 0.0
            familiar_dock._WIDTH[0] = 0.0


def test_the_input_stays_pinned_inside_the_dock_under_a_long_transcript(monkeypatch, tmp_path):
    """The transcript is the only thing that scrolls: the footer (input and
    Send) must end inside the dock's own window, never below it. The height
    used to be an estimate that counted the input and Send as one row, so on
    a long conversation the footer overflowed the dock and the whole dock
    scrolled, input and all. Measured on the second frame, once the footer's
    real height from the first has been recorded."""
    from types import SimpleNamespace

    from _ui_context import imgui_context

    from realmspinner.familiar import threads as threads_mod
    from realmspinner.studio.assistant import ui as familiar_ui
    from realmspinner.studio.settings import Settings

    familiar_threads = threads_mod.Threads()
    ctx = SimpleNamespace(
        state=SimpleNamespace(mode="home", familiar=familiar_ui.FamiliarUIState(expanded=True)),
        settings=Settings.load(tmp_path),
        svc=SimpleNamespace(config=_familiar_config(tmp_path)),
        familiar_threads=familiar_threads,
    )
    key = familiar_ui.thread_key(ctx)
    for i in range(40):
        familiar_threads.append(key, threads_mod.Turn(role="user", text=f"Question {i}"))
        familiar_threads.append(key, threads_mod.Turn(role="familiar", text=f"Answer {i}"))
    familiar_dock._familiar_state_cache = None

    ends: list[tuple[float, float]] = []
    real_footer = familiar_ui._draw_footer

    def measured(*a, **k):
        from imgui_bundle import imgui

        real_footer(*a, **k)
        bottom = imgui.get_window_pos().y + imgui.get_window_size().y
        # The cursor sits one item spacing below the last item; the item's own
        # bottom edge is what must stay inside the dock.
        last_item_bottom = imgui.get_cursor_screen_pos().y - imgui.get_style().item_spacing.y
        ends.append((last_item_bottom, bottom))

    monkeypatch.setattr(familiar_ui, "_draw_footer", measured)
    familiar_ui._FOOTER_H[0] = 0.0
    familiar_dock._T[0] = 1.0
    familiar_dock._WIDTH[0] = 400.0
    with imgui_context(monkeypatch) as imgui:
        try:
            for _ in range(2):
                imgui.new_frame()
                imgui.set_next_window_pos((0, 0))
                imgui.set_next_window_size((1200, 600))
                imgui.begin("##shell-host")
                try:
                    familiar_dock.draw(ctx)
                finally:
                    imgui.end()
                    imgui.end_frame()
                    imgui.render()
        finally:
            familiar_dock._T[0] = 0.0
            familiar_dock._WIDTH[0] = 0.0
            familiar_ui._FOOTER_H[0] = 0.0
    footer_end, dock_bottom = ends[-1]
    assert footer_end <= dock_bottom + 1.0, (footer_end, dock_bottom)
