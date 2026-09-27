"""Closes the 2026-09-26 audit's shell-home-library-05: Chapter 21 (Home) said
"There is no 'last mode' setting" -- but Settings ▸ Startup has exactly one,
"Last workspace", read by ``shell.events.initial_mode`` and documented in
Chapter 42 (``app_settings.py:432-438``, ``events.py:44-65``). This test reads
the prose directly, so it fails on the original chapter (which denies the
setting outright) and passes once the chapter names it instead.
"""

from pathlib import Path

MANUAL = Path(__file__).resolve().parents[2] / "docs" / "manual"


def _text(name: str) -> str:
    return (MANUAL / name).read_text(encoding="utf-8")


def test_manual_21_does_not_deny_the_startup_mode_setting():
    text = _text("21-home.md")
    assert "no \"last mode\" setting" not in text, (
        "chapter 21 still denies the Startup ▸ Last workspace setting outright"
    )
    assert "Last workspace" in text, (
        "chapter 21 should name the Settings ▸ Startup ▸ Last workspace option "
        "it now no longer denies"
    )


def test_manual_21_and_42_agree_last_workspace_falls_back_to_home():
    """Both chapters describe the same fallback (a gated mode reopens Home
    instead) -- the two must not drift into disagreeing about it."""
    home_text = _text("21-home.md")
    settings_text = _text("42-app-settings.md")
    assert "Home" in home_text.split("Last workspace", 1)[1][:400]
    assert "Home" in settings_text.split("Last workspace", 1)[1][:400]
