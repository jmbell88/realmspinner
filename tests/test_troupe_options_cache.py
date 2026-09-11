"""The palette directory, as seen by Troupe's own forms.

Modelled directly on ``tests/test_panes_mtime_guard.py``'s palette pair for
``panes.inspector.palette_names`` -- the same directory, the same hazard, and
until the 2026-09-11 audit (finding troupe-04) neither of these two caches
applied the rule that file already had to learn.

``troupe_mode.options`` backs three surfaces (itself, ``panes.troupe_send``
and ``panes.troupe_settings``, which all call it directly) and
``panes.settings_character.options`` backs Create's Character arm and its own
New Character form. Before the fix both cached ``ctx.state.preview`` forever
on ``OPTIONS_SLOT`` with no key at all, so a palette file dropped in while the
app was running never appeared in any of the four until restart, even though
``service.palettes``' module docstring states the directory's whole design
intent is drop-in-while-running use.
"""

from __future__ import annotations

from typing import Any

import pytest

from warlock.studio import troupe_mode
from warlock.studio.panes import settings_character, stamps


class FakeState:
    def __init__(self) -> None:
        self.preview: dict[str, Any] = {}


class FakeCtx:
    def __init__(self, svc: Any) -> None:
        self.svc = svc
        self.state = FakeState()


def _frozen(monkeypatch, path):
    """Freeze the clock *at* ``path``'s mtime -- the worst case the racily-
    clean rule exists for, and the one where an unguarded cache hides a drop
    forever (``panes.stamps``'s own module doc)."""
    monkeypatch.setattr(stamps.time, "time_ns", lambda: path.stat().st_mtime_ns)


def _settled(monkeypatch, path):
    from warlock.service.files import MTIME_RACE_NS

    settled = path.stat().st_mtime_ns + MTIME_RACE_NS * 2
    monkeypatch.setattr(stamps.time, "time_ns", lambda: settled)


@pytest.fixture
def palette_dir(svc, monkeypatch, tmp_path):
    directory = tmp_path / "palettes"
    directory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(svc.config, "palette_dir", directory)
    return directory


# --- troupe_mode.options: the door troupe_send and troupe_settings share ------


def test_a_palette_dropped_in_mid_session_appears_in_the_troupe_and_character_forms(
    svc, monkeypatch, palette_dir
):
    """The regression, for ``troupe_mode.options`` -- read by ``troupe_mode``
    itself, ``panes.troupe_send`` and ``panes.troupe_settings`` alike."""
    ctx = FakeCtx(svc)
    (palette_dir / "nes.hex").write_text("000000\nffffff\n", encoding="utf-8")
    _frozen(monkeypatch, palette_dir)

    assert troupe_mode.options(ctx)["palettes"] == ["nes"]

    (palette_dir / "gameboy.hex").write_text("081820\ne0f8d0\n", encoding="utf-8")

    assert troupe_mode.options(ctx)["palettes"] == ["gameboy", "nes"]


def test_a_settled_troupe_options_directory_is_walked_once(svc, monkeypatch, palette_dir):
    """The other half: once the stamp is safely in the past the door is asked
    once, which is what keeps a directory walk off the frame thread."""
    ctx = FakeCtx(svc)
    (palette_dir / "nes.hex").write_text("000000\n", encoding="utf-8")
    _settled(monkeypatch, palette_dir)

    from warlock.service import troupe as svc_troupe

    calls: list[Any] = []
    real = svc_troupe.troupe_options
    monkeypatch.setattr(
        svc_troupe, "troupe_options", lambda s: (calls.append(s), real(s))[1]
    )

    for _ in range(5):
        assert troupe_mode.options(ctx)["palettes"] == ["nes"]

    assert len(calls) == 1


# --- settings_character.options: Create's Character arm -----------------------


def test_a_palette_dropped_in_mid_session_appears_in_settings_character(
    svc, monkeypatch, palette_dir
):
    ctx = FakeCtx(svc)
    (palette_dir / "nes.hex").write_text("000000\nffffff\n", encoding="utf-8")
    _frozen(monkeypatch, palette_dir)

    assert settings_character.options(ctx)["troupe"]["palettes"] == ["nes"]

    (palette_dir / "gameboy.hex").write_text("081820\ne0f8d0\n", encoding="utf-8")

    assert settings_character.options(ctx)["troupe"]["palettes"] == ["gameboy", "nes"]


def test_a_settled_settings_character_options_directory_is_read_once(
    svc, monkeypatch, palette_dir
):
    ctx = FakeCtx(svc)
    (palette_dir / "nes.hex").write_text("000000\n", encoding="utf-8")
    _settled(monkeypatch, palette_dir)

    from warlock.service import characters as svc_characters

    calls: list[Any] = []
    real = svc_characters.character_options
    monkeypatch.setattr(
        svc_characters, "character_options", lambda s: (calls.append(s), real(s))[1]
    )

    for _ in range(5):
        assert settings_character.options(ctx)["troupe"]["palettes"] == ["nes"]

    assert len(calls) == 1
