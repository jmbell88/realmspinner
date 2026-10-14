"""The 2026-10-14 audit's Packwright cleanup round (fixer C5)."""

from __future__ import annotations

import pytest

from realmspinner.studio.modes.packwright import mode as packwright_mode

from .test_packwright_mode import FakeCtx, _tab


@pytest.fixture(autouse=True)
def _no_pygame_display(monkeypatch):
    import pygame

    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0)


@pytest.mark.parametrize("export", ["export_files", "export_library"])
def test_an_export_after_a_failed_first_pack_toasts_the_pack_error_not_nothing_packed(export):
    """packwright-07: a pack that failed outright leaves ``layout`` and ``atlas``
    None and the reason in ``pack_error``; the disabled button already said it
    and Ctrl+E / Ctrl+Shift+E said "Nothing packed yet." over it."""
    ctx = FakeCtx()
    tab = _tab(ctx, sources=1)
    tab.layout = None
    tab.atlas = None
    tab.pack_error = "The sprites do not fit in 256 x 256."

    getattr(packwright_mode, export)(ctx, tab)

    errors = [text for text, level in ctx.toasts if level == "error"]
    assert errors == ["The sprites do not fit in 256 x 256."], ctx.toasts


@pytest.mark.parametrize("export", ["export_files", "export_library"])
def test_an_export_with_nothing_packed_and_no_error_still_says_nothing_packed_yet(export):
    ctx = FakeCtx()
    tab = _tab(ctx, sources=1)
    tab.layout = None
    tab.atlas = None
    tab.pack_error = ""

    getattr(packwright_mode, export)(ctx, tab)

    assert [t for t, level in ctx.toasts if level == "error"] == ["Nothing packed yet."]
