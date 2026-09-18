"""``EventsMixin._events``: the per-frame pygame event pump and its dispatch
to a mode's shortcut/viewport handlers.

shell-04 (the 2026-09-18 audit): a mode's ``handle_key`` or a shortcut arm
raising inside ``_events`` used to propagate straight out of it, into
``App.frame`` and then into ``App.run()``'s whole-loop ``except`` -- ending
the session over one failed gesture (mason-01, mason-02). ``_events`` runs
every frame *before* ``imgui.new_frame()`` (see ``shell/frame.py``'s own
``frame()``), so there is no imgui window open at the point these handlers
run -- which is also why the fix here is a plain try/except around each
event's dispatch rather than ``guard.surface``: that guard reaches into dear
imgui's current-window/error-recovery state to unwind a half-drawn pane, and
calling it with no frame open does not merely skip protecting anything, it
segfaults the process (verified against this exact calling point with
``tests/_ui_context``'s own real-but-unrendered context, before this fix:
a native access violation in ``imgui.internal.get_current_window_read``/
``error_recovery_store_state``, not a catchable Python exception).
"""

from __future__ import annotations

from types import SimpleNamespace


def _fake_app(monkeypatch, imgui):
    """A minimal object carrying just what ``EventsMixin._events`` reads."""
    from warlock.studio.shell.events import EventsMixin

    class FakeApp(EventsMixin):
        pass

    toasts: list[tuple] = []
    errors: list[str] = []

    app = FakeApp()
    app.app_ctx = SimpleNamespace(
        state=SimpleNamespace(
            mode="inker",
            note_error=lambda text: errors.append(text),
        ),
        toast=lambda *a, **k: toasts.append(a),
        toast_once=lambda *a, **k: (toasts.append(a), True)[1],
        settings=SimpleNamespace(set=lambda *a, **k: None),
    )
    app.viewer = None
    app._viewport_hovered = False
    app._min_size = (100, 100)
    # ``_events`` reads ``imgui.get_io()`` unconditionally, off-frame, exactly
    # as ``App.frame`` calls it -- no ``imgui.new_frame()`` here either.
    io = imgui.get_io()
    io.want_text_input = False
    return app, toasts, errors


def test_a_key_handler_that_raises_is_contained_not_session_ending(monkeypatch):
    """A shortcut arm that raises must not escape ``_events`` -- it is caught,
    logged and toasted, and the frame's event pump carries on."""
    import pygame
    from _ui_context import imgui_context

    with imgui_context(monkeypatch) as imgui:
        app, toasts, errors = _fake_app(monkeypatch, imgui)
        monkeypatch.setattr(app, "_modal_open", lambda: False)

        def _boom(_event):
            raise RuntimeError("a broken shortcut arm (mason-01/mason-02's shape)")

        monkeypatch.setattr(app, "_shortcut", _boom)

        event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_w, mod=0, unicode="w")
        monkeypatch.setattr(pygame.event, "get", lambda: [event])

        # The claim: this must not raise.
        app._events()

        assert toasts, "a contained failure must still be announced to the user"
        assert errors, "a contained failure must still reach the doctor banner"


def test_a_later_event_in_the_same_pump_still_runs_after_one_fails(monkeypatch):
    """One bad gesture must not swallow the rest of the frame's input --
    only the event whose dispatch raised is skipped."""
    import pygame
    from _ui_context import imgui_context

    with imgui_context(monkeypatch) as imgui:
        app, _toasts, _errors = _fake_app(monkeypatch, imgui)
        monkeypatch.setattr(app, "_modal_open", lambda: False)

        seen: list[str] = []

        def _shortcut(event):
            if event.unicode == "w":
                raise RuntimeError("boom")
            seen.append(event.unicode)

        monkeypatch.setattr(app, "_shortcut", _shortcut)

        bad = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_w, mod=0, unicode="w")
        good = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_s, mod=0, unicode="s")
        monkeypatch.setattr(pygame.event, "get", lambda: [bad, good])

        app._events()

        assert seen == ["s"]
