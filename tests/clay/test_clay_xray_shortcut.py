"""Clay: the X-ray toggle's tooltip claims a chord that did nothing.

The 2026-09-07 audit's clay-08. ``panes/clay_header.py``'s X-ray button has
advertised "(Alt+Z)" in its tooltip since the button was added, but nothing in
``clay_mode.handle_key`` bound it -- Alt+Z reached the handler, matched no
branch, and was consumed anyway (the mode's own rule: a key is swallowed
whenever a document is open, so nothing under it can react either). Wiring the
chord to the same ``state.xray`` flag the button flips is what makes the
tooltip's claim true rather than dropping it.
"""

from __future__ import annotations

from typing import Any

import pygame
import pytest

from warlock.kernels.mesh import document as bd
from warlock.kernels.mesh import primitives as bp
from warlock.studio import clay_mode
from warlock.studio.panes import clay_header


class FakeCtx:
    def __init__(self) -> None:
        self.state = _AppState()
        self.settings = _Settings()
        self.toasts: list[tuple[str, str]] = []

    def toast(self, message: str, kind: str = "info") -> None:
        self.toasts.append((message, kind))


class _AppState:
    def __init__(self) -> None:
        self.clay = None


class _Settings:
    def __init__(self) -> None:
        self.store: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self.store.get(key)

    def set(self, key: str, value: Any) -> None:
        self.store[key] = value


@pytest.fixture(autouse=True)
def _no_pygame_display(monkeypatch):
    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0)


def _tab() -> tuple[FakeCtx, Any]:
    ctx = FakeCtx()
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
    clay_mode.adopt(ctx, doc, title="Scene")
    return ctx, doc


def test_the_x_ray_tooltip_still_names_alt_z() -> None:
    """A guard against the other half of the row's fix -- if the chord is ever
    dropped instead of wired, this must be updated in the same change rather
    than left to drift."""
    import inspect

    source = inspect.getsource(clay_header._trailing)
    assert "Alt+Z" in source


def test_alt_z_toggles_x_ray_and_consumes_the_key() -> None:
    ctx, _doc = _tab()
    state = clay_mode.ensure(ctx)
    assert state.xray is False

    event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_z, mod=pygame.KMOD_ALT)
    assert clay_mode.handle_key(ctx, event) is True
    assert state.xray is True

    assert clay_mode.handle_key(ctx, event) is True
    assert state.xray is False


def test_plain_z_and_ctrl_z_are_unaffected() -> None:
    """Alt+Z must not steal the bare Z tool key or Ctrl+Z's undo."""
    ctx, doc = _tab()
    state = clay_mode.ensure(ctx)

    bare_z = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_z, mod=0)
    clay_mode.handle_key(ctx, bare_z)
    assert state.xray is False, "the bare Z key must not toggle X-ray"

    depth = len(doc.history)
    ctrl_z = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_z, mod=pygame.KMOD_CTRL)
    clay_mode.handle_key(ctx, ctrl_z)
    assert state.xray is False, "Ctrl+Z must still be undo, not the X-ray toggle"
    assert len(doc.history) <= depth
