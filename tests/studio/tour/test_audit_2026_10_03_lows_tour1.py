"""Regressions for the tour/tour-1 slice of the 2026-10-03 audit's Lows:
tour-05 (a card-driven stop leaves a stale hole), tour-06 (the import pin's
relative-import blind spot), tour-07 (docstrings naming dead paths), tour-08
(Esc while a text field has focus) and tour-09 (the character-type step
naming a control the screen never captions).
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from realmspinner.studio.panes import tour as tour_mod
from realmspinner.studio.state import TourState

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src" / "realmspinner"


# --- tour-05 -----------------------------------------------------------------


class _Enum:
    """``imgui.WindowFlags_.x.value`` for any ``x``."""

    def __getattr__(self, _name: str) -> Any:
        return SimpleNamespace(value=1)


class _FakeImgui:
    """Enough imgui to run ``_card`` once, with the card window "open"."""

    Cond_ = _Enum()
    WindowFlags_ = _Enum()
    StyleVar_ = _Enum()

    def set_next_window_pos(self, *_a: Any) -> None: ...
    def set_next_window_size(self, *_a: Any) -> None: ...
    def set_next_window_bg_alpha(self, *_a: Any) -> None: ...
    def push_style_var(self, *_a: Any) -> None: ...
    def pop_style_var(self, *_a: Any) -> None: ...
    def begin(self, *_a: Any) -> tuple[bool, bool]:
        return (True, True)

    def end(self) -> None: ...
    def is_window_focused(self) -> bool:
        return False

    def get_window_pos(self) -> Any:
        return SimpleNamespace(x=10.0, y=20.0)

    def get_window_size(self) -> Any:
        return SimpleNamespace(x=300.0, y=120.0)


def _drive_card(monkeypatch: Any, body: Any) -> Any:
    monkeypatch.setattr(tour_mod, "imgui", _FakeImgui())
    w = tour_mod.widgets
    monkeypatch.setattr(w, "popover_enter", lambda *a, **k: (1.0, 0.0))
    monkeypatch.setattr(w, "frosted", lambda *a, **k: False)
    monkeypatch.setattr(w, "push_surface_rounding", lambda *a, **k: 0.0)
    monkeypatch.setattr(w, "pop_surface_rounding", lambda *a, **k: None)
    monkeypatch.setattr(w, "window_shadow", lambda *a, **k: None)
    monkeypatch.setattr(w, "window_backdrop", lambda *a, **k: None)
    monkeypatch.setattr(tour_mod, "_card_body", body)
    state = TourState()
    state.start("first-hour")
    ctx = SimpleNamespace(state=SimpleNamespace(tour=state), settings=None, toast=lambda *a: None)
    viewport = SimpleNamespace(
        work_pos=SimpleNamespace(x=0.0, y=0.0), work_size=SimpleNamespace(x=1000.0, y=800.0)
    )
    tour_mod._card_rect[0] = None
    try:
        tour_mod._card(ctx, viewport, SimpleNamespace(), SimpleNamespace(), None, False)
        return tour_mod._card_rect[0]
    finally:
        tour_mod._card_rect[0] = None
        tour_mod._card_focused[0] = False


def test_ending_the_tour_from_the_card_leaves_no_card_rect_behind(monkeypatch: Any) -> None:
    """tour-05: ``stop`` clears the rect from inside ``_card_body``, and
    ``_card`` then wrote it again from the window it was still inside -- the
    next tour's first frame was veiled around the previous card's hole."""

    def end_button(ctx: Any, *_rest: Any) -> None:
        tour_mod.stop(ctx)

    assert _drive_card(monkeypatch, end_button) is None


def test_a_card_that_keeps_the_tour_running_still_records_its_rect(monkeypatch: Any) -> None:
    """The other direction: the veil's hole for the card must survive the
    ordinary frame, or the tour dims its own text."""

    assert _drive_card(monkeypatch, lambda *a: None) is not None


# --- tour-06 -----------------------------------------------------------------


def test_a_relative_import_that_leaves_the_tour_package_is_an_outward_import() -> None:
    """tour-06: ``if node.level: continue`` waved every relative import
    through, including the three shapes a drawing helper would add."""
    import test_tour_imports as pin

    found = pin.imports_of(
        ast.parse(
            "from ..state import X\nfrom .. import tokens\nfrom ...service import jobs\n"
            "from .steps import Step\nfrom . import steps\n"
        ),
        "realmspinner.studio.tour",
    )
    assert "realmspinner.studio.state" in found
    assert "realmspinner.studio.tokens" in found
    assert "realmspinner.service" in found
    # Inside the package they resolve to it and are not outward.
    assert "realmspinner.studio.tour.steps" in found


def test_the_tour_package_import_leaves_the_renderers_out_of_sys_modules() -> None:
    """tour-06's second half: the subprocess test only asked that the import
    succeed, never that imgui/pygame stayed out."""
    code = (
        "import sys, realmspinner.studio.tour\n"
        "bad = sorted(m for m in ('imgui_bundle', 'pygame', 'moderngl') if m in sys.modules)\n"
        "print(','.join(bad))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"importing studio.tour pulled in {result.stdout.strip()}"


# --- tour-07 -----------------------------------------------------------------

_DOCSTRING_FILES = (
    SRC / "studio" / "tour" / "__init__.py",
    SRC / "studio" / "panes" / "tour.py",
    ROOT / "tests" / "studio" / "tour" / "test_tour_imports.py",
)


def test_tour_docstrings_name_only_paths_that_exist() -> None:
    """tour-07: the comments said ``studio/inker/``, ``clay/``, ``plotter/``,
    ``packwright/`` (now ``kernels/`` and ``studio/modes/``) and ``main.py``
    for names that moved to ``shell/app.py`` and ``shell/events.py``."""
    text = "\n".join(p.read_text(encoding="utf-8") for p in _DOCSTRING_FILES)
    assert "studio/inker/" not in text
    assert "``clay/``" not in text
    assert "``plotter/``" not in text
    assert "``packwright/``" not in text
    for line in text.splitlines():
        if "main.py" in line:
            pytest.fail(f"a tour file still points at the old main.py: {line.strip()}")
    for path in re.findall(r"``((?:studio|kernels)/[\w/]+\.py)``", text):
        assert (SRC / path).exists(), f"{path} does not exist"


# --- tour-08 -----------------------------------------------------------------


def test_escape_ends_a_running_tour_even_while_a_text_field_has_focus(monkeypatch: Any) -> None:
    """tour-08: a plain Esc is not delivered to ``_shortcut`` while imgui has a
    text field focused, so on the ``prompt`` step the card's "Esc ends it at any
    point" was false for the one step that asks the reader to type."""
    import pygame
    from _ui_context import imgui_context

    from realmspinner.studio.shell.events import EventsMixin

    class FakeApp(EventsMixin):
        pass

    with imgui_context(monkeypatch) as imgui:
        app = FakeApp()
        tour = TourState()
        tour.start("first-hour")
        app.app_ctx = SimpleNamespace(
            state=SimpleNamespace(mode="create", tour=tour, note_error=lambda t: None),
            toast=lambda *a, **k: None,
            settings=SimpleNamespace(set=lambda *a, **k: None),
        )
        app.viewer = None
        app._viewport_hovered = False
        app._min_size = (100, 100)
        imgui.get_io().want_text_input = True
        monkeypatch.setattr(app, "_modal_open", lambda: False)
        seen: list[int] = []
        monkeypatch.setattr(app, "_shortcut", lambda event: seen.append(event.key))

        esc = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0, unicode="")
        letter = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a, mod=0, unicode="a")
        monkeypatch.setattr(pygame.event, "get", lambda: [letter, esc])
        app._events()

        assert seen == [pygame.K_ESCAPE], "Esc must reach the shortcuts; typing must not"

        # And with no tour running, Esc in a text field stays the field's own.
        tour.stop()
        seen.clear()
        monkeypatch.setattr(pygame.event, "get", lambda: [esc])
        app._events()
        assert seen == []


# --- tour-09 -----------------------------------------------------------------


def test_the_character_type_step_names_a_caption_the_create_screen_draws() -> None:
    """tour-09: the step said "Generation type", a caption nothing draws (the
    combo is ``##generation-type`` under "What are you making?"), and carried
    no anchor, so nothing rang the control to change."""
    from realmspinner.studio.tour import scripts

    step = next(s for t in scripts.TOURS for s in t.steps if s.id == "character-type")
    brief = (SRC / "studio" / "modes" / "create" / "ui" / "brief.py").read_text(encoding="utf-8")
    assert "Generation type" not in step.body
    assert "What are you making?" in step.body
    assert 'widgets.secondary("What are you making?")' in brief
    assert step.anchor == "create/type"
    assert 'anchors.mark("create/type")' in brief
