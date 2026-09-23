"""Regressions for the 2026-09-23 (second run) audit's shell findings.

Four records, closed together because they share owners (``shell/tasks.py``,
``shortcuts.py``, ``jobs_cache.py``, ``widgets.py``) rather than a theme.
shell-02 and shell-03 (the Ctrl+/ sheet's missing Home/Library and labelling
groups) are gated by ``tests/manual/test_shortcuts.py`` instead, whose
``SECTIONS`` table this pair had to join -- see the named regression tests
there. See "the 2026-09-23 audit, finding <id>" in each fix's comments for
the paragraph this test proves.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from realmspinner.studio import main, widgets
from realmspinner.studio.jobs_cache import JobsCache
from realmspinner.studio.state import AppState, Filters

# --- shell-01: a queued re-texture invalidates the job cache ---------------


class _FakeCache:
    def __init__(self) -> None:
        self.jobs: list[dict[str, Any]] = []
        self.storage: dict[str, Any] = {}
        self.invalidated = 0

    def invalidate(self) -> None:
        self.invalidated += 1


class _FakeApp:
    """Enough of ``App`` for ``_on_task_done`` to run against.

    Mirrors ``tests/studio/test_studio_plumbing.py``'s ``FakeApp`` (which
    already proves ``"sheet:"`` reaches ``cache.invalidate()`` the same way);
    duplicated in miniature here rather than imported, since this file owns
    no changes to that one.
    """

    def __init__(self) -> None:
        self.toasts: list[tuple[str, str]] = []
        self.calls: list[str] = []
        self.app_ctx = SimpleNamespace(
            state=AppState(),
            cache=_FakeCache(),
            svc=None,
            toast=lambda message, level="info": self.toasts.append((message, level)),
            submit=lambda key, run, *args: self.calls.append(f"submit:{key}") or True,
        )
        self.runtime = SimpleNamespace(checks=["startup-snapshot"])
        self._last_health_poll = 0.0
        self.viewer = SimpleNamespace(
            pose_mode=False, path="model.glb", editor=SimpleNamespace(dirty=True)
        )

    def _refresh_rig_side_data(self) -> None:
        self.calls.append("_refresh_rig_side_data")

    def dispatch(self, key: str, result: Any = None) -> None:
        main.App._on_task_done(self, SimpleNamespace(key=key, result=result, ok=True))


def test_a_queued_retexture_invalidates_the_job_cache_like_a_queued_remesh_does():
    """The 2026-09-23 audit, finding shell-01: ``texture_panel.py``'s
    ``_submit`` keys a queued re-texture job ``f"retexture:{job_id}"``
    (texture_panel.py:294), but ``"retexture:"`` was missing from the prefix
    tuple ``"remesh:"`` is in, so the row landing on screen relied on the 3 s
    idle backstop rather than on the same immediate cache drop a remesh gets.
    """
    app = _FakeApp()
    app.dispatch("retexture:aaaaaaaaaaaa")

    assert app.app_ctx.cache.invalidated == 1, (
        "queueing a re-texture did not call ctx.cache.invalidate() -- the "
        '"retexture:" key is still missing from the prefix tuple "remesh:" is in'
    )


def test_a_queued_remesh_still_invalidates_the_job_cache():
    """The guard above must not be the only key answering this -- "remesh:"
    already worked and must go on working."""
    app = _FakeApp()
    app.dispatch("remesh:aaaaaaaaaaaa")
    assert app.app_ctx.cache.invalidated == 1


# --- shell-04: widening a search past a full window still reaches matches --


class _StubStore:
    """Enough of ``JobStore`` for ``request_widen`` to run against.

    ``request_widen`` never reaches the store directly -- it submits
    ``self._search`` through ``runner.submit`` -- so a store that would
    raise if actually queried still proves the point: the bug is whether the
    submit happens at all, not what it would return.
    """

    def search_ids(self, *a, **k):  # pragma: no cover - never reached
        raise AssertionError("request_widen queried the store directly")


class _StubSvc:
    def __init__(self) -> None:
        self.store = _StubStore()


class _RecordingRunner:
    def __init__(self) -> None:
        self.submitted: list[str] = []

    def submit(self, key, fn, *args, **kwargs):
        self.submitted.append(key)
        return True


def test_widening_a_search_still_reaches_matches_beyond_a_full_window(monkeypatch):
    """The 2026-09-23 audit, finding shell-04: once the window reached
    ``MAX_LIST_LIMIT``, ``request_widen`` skipped the id-only store search
    entirely (it tested ``not can_load_more()``, which is also true once the
    window is at the ceiling) -- so on a store with more than
    ``MAX_LIST_LIMIT`` rows, a real match outside the window returned
    nothing, silently, forever. The fix asks the real question instead:
    whether the loaded window (``len(self.jobs)``) already covers the whole
    store (``self.total``).
    """
    from realmspinner.service import jobs as svc_jobs

    monkeypatch.setattr(svc_jobs, "MAX_LIST_LIMIT", 3)

    cache = JobsCache(_StubSvc())
    cache.limit = 3  # already at the (patched) ceiling
    cache.jobs = [{"id": f"job-{i}"} for i in range(3)]  # the loaded window
    cache.total = 10  # the store holds more than the window covers

    runner = _RecordingRunner()
    filters = Filters()
    filters.text = "barrel"

    submitted = cache.request_widen(filters, runner)

    assert submitted is True, (
        "request_widen refused to widen a full-but-not-whole window -- a "
        "real match past MAX_LIST_LIMIT rows is now unreachable"
    )
    from realmspinner.studio.jobs_cache import SEARCH_KEY

    assert runner.submitted == [SEARCH_KEY]


def test_widening_a_search_is_still_skipped_once_the_window_is_the_whole_store():
    """The guard above must not turn this into an unconditional submit: once
    the window really does hold everything the store has, there is nothing
    left outside it to widen with."""
    from realmspinner.service import jobs as svc_jobs

    cache = JobsCache(_StubSvc())
    cache.limit = svc_jobs.MAX_LIST_LIMIT
    cache.jobs = [{"id": "only-job"}]
    cache.total = 1  # the window covers the whole store

    runner = _RecordingRunner()
    filters = Filters()
    filters.text = "barrel"

    submitted = cache.request_widen(filters, runner)

    assert submitted is False
    assert runner.submitted == []


# --- shell-05: a disabled icon-tier button shows its reason ----------------


class _GlyphImgui:
    """Just enough imgui for ``widgets._glyph_button`` to run headless.

    Mirrors ``tests/studio/test_probe.py``'s ``_ButtonImgui`` (built for
    ``_button_with_note``) one tier up: ``_glyph_button`` also pushes its own
    style colours/vars around a raw ``imgui.button`` and reads ``get_io()``
    for the keyboard-focus tooltip path.
    """

    class HoveredFlags_:  # noqa: N801 -- imgui's own spelling
        allow_when_disabled = type("V", (), {"value": 1 << 10})()

    class Col_:  # noqa: N801
        button = type("V", (), {"value": 0})()
        button_hovered = type("V", (), {"value": 1})()
        text = type("V", (), {"value": 2})()

    class StyleVar_:  # noqa: N801
        frame_padding = type("V", (), {"value": 0})()
        button_text_align = type("V", (), {"value": 1})()

    @staticmethod
    def ImVec2(*a):
        return a

    @staticmethod
    def ImVec4(*a):
        return a

    def __init__(self, *, hovered: bool = True) -> None:
        self.tooltips: list[str] = []
        self._hovered = hovered
        self._next_id = 0

    def get_id(self, _text):
        self._next_id += 1
        return self._next_id

    def push_style_color(self, *_a):
        pass

    def pop_style_color(self, *_a):
        pass

    def push_style_var(self, *_a):
        pass

    def pop_style_var(self, *_a):
        pass

    def begin_disabled(self):
        pass

    def end_disabled(self):
        pass

    def button(self, _label, _size=(0, 0)):
        return False

    def is_item_hovered(self, _flags=0):
        return self._hovered

    def is_item_focused(self):
        return False

    def get_io(self):
        return SimpleNamespace(nav_visible=False)

    def set_tooltip(self, text):
        self.tooltips.append(text)


def test_a_disabled_icon_button_shows_its_reason_not_its_plain_tooltip(monkeypatch):
    """The 2026-09-23 audit, finding shell-05: ``icon_button``'s ``reason``
    (added for the first run's shell-05) reached ``probe.record`` only --
    ``_glyph_button`` always showed the plain ``tooltip``, so a disabled icon
    button never surfaced why. ``_button_with_note`` already picks the
    disabled reason over the live tooltip (``note = reason if not enabled
    else tooltip``); this is the same rule for the icon tier.
    """
    stub = _GlyphImgui()
    monkeypatch.setattr(widgets, "imgui", stub)

    widgets._glyph_button(
        "X",
        24.0,
        "Delete",
        enabled=False,
        reason="3 jobs are still queued or running",
    )

    assert stub.tooltips == ["3 jobs are still queued or running"], (
        "a disabled _glyph_button still showed its plain tooltip instead of "
        "the reason it was disabled"
    )


def test_an_enabled_icon_button_still_shows_its_plain_tooltip(monkeypatch):
    """The guard above must not swallow the ordinary case: a live icon
    button's tooltip is its only name (UX-02), so it must still show."""
    stub = _GlyphImgui()
    monkeypatch.setattr(widgets, "imgui", stub)

    widgets._glyph_button("X", 24.0, "Delete", enabled=True)

    assert stub.tooltips == ["Delete"]
