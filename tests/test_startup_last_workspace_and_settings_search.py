"""W3.1 (reopen the last workspace) and W3.2 (Settings search).

W3.1 persists the mode the user was last in and offers a Settings choice to
reopen it on the next launch; a fresh install (no stored preference at all,
or "Home" chosen) must still land on Home exactly as it always has, and a
remembered mode that is gated (weights or a pack not installed) falls back to
Home through the door every mode switch already goes through -- ``mode_gate``
-- rather than a second refusal invented for this.

W3.2 adds a search field above Settings' category rail that filters a small,
static index of rows by label, tooltip and a synonym table, so "bigger text"
finds the UI-scale row without the word "bigger" appearing anywhere near it.
"""

from __future__ import annotations

from types import SimpleNamespace

from warlock.studio import main as main_mod
from warlock.studio.panes import app_settings


class _Settings:
    """Only what ``initial_mode`` and ``App._note_last_workspace`` touch."""

    def __init__(self, data=None):
        self.data: dict = dict(data or {})

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value


# --- W3.1: startup mode -------------------------------------------------------


def test_startup_can_reopen_the_last_workspace_and_defaults_to_home():
    # A fresh install: no "startup_mode" key exists at all. Must read as Home,
    # not crash on a missing key and not require the key to be seeded.
    fresh = _Settings()
    assert main_mod.initial_mode(fresh, lambda _key: True) == "home"

    # "Home" chosen explicitly is the same answer as no preference at all.
    home_chosen = _Settings({"startup_mode": "home", "last_workspace": "clay"})
    assert main_mod.initial_mode(home_chosen, lambda _key: True) == "home"

    # "Last workspace" chosen, and the door is open: reopens exactly that mode.
    remembers = _Settings({"startup_mode": "last", "last_workspace": "clay"})
    assert main_mod.initial_mode(remembers, lambda _key: True) == "clay"

    # "Last workspace" chosen, but that workspace is gated on this machine
    # (missing weights or a pack) -- falls back to Home via the existing
    # refusal (``available`` returning False), not a new one.
    gated = _Settings({"startup_mode": "last", "last_workspace": "create"})
    assert main_mod.initial_mode(gated, lambda key: key != "create") == "home"

    # "Last workspace" chosen, but a settings file written before this
    # existed has nothing under "last_workspace" yet.
    never_remembered = _Settings({"startup_mode": "last"})
    assert main_mod.initial_mode(never_remembered, lambda _key: True) == "home"


def test_initial_mode_falls_back_to_home_for_a_mode_that_no_longer_exists():
    """shell-04 (the 2026-09-08 audit): a stale or hand-edited
    ``last_workspace`` naming a retired mode used to reach ``available()``
    unchecked and, with the door reported open, land on it -- opening
    ``_build_ui``'s else branch (Create) while ``state.mode`` held a value
    nothing else in the app recognises as a member of ``modes.KEYS``.
    ``_escape_mode`` already refuses this way for its own history; this is
    the other reader of a persisted mode name doing the same.
    """
    retired = _Settings({"startup_mode": "last", "last_workspace": "retired_mode"})
    # ``available`` says yes to everything -- the retired name must still be
    # refused on membership alone, before the gate is ever asked.
    assert main_mod.initial_mode(retired, lambda _key: True) == "home"


def test_the_last_workspace_is_written_down_as_the_mode_changes():
    """The other half of W3.1: something has to persist ``last_workspace``
    for ``initial_mode`` to have anything to read, or "Last workspace" opens
    on Home forever regardless of what the user chose."""
    app = main_mod.App.__new__(main_mod.App)
    app._last_mode = "home"
    settings = _Settings()
    app.app_ctx = SimpleNamespace(state=SimpleNamespace(mode="home"), settings=settings)

    app.app_ctx.state.mode = "clay"
    app._note_last_workspace(app.app_ctx)
    assert settings.get("last_workspace") == "clay"


# --- W3.2: Settings search ----------------------------------------------------


def test_settings_search_finds_a_row_by_plain_language():
    """"bigger text" names no word on the UI-scale row itself -- only the
    small synonym table connects the two, which is what makes this "plain
    language" rather than a literal label match."""
    found = app_settings.search_rows("bigger text")
    assert any(row.label == "UI scale" for row in found)

    disk = app_settings.search_rows("disk space")
    assert any(row.category == "models" for row in disk)

    music = app_settings.search_rows("music")
    assert any(row.category == "packs" for row in music)


def test_settings_search_also_matches_a_tooltip_word_directly():
    found = app_settings.search_rows("palette")
    assert any(row.label == "Theme" for row in found)


def test_settings_search_with_no_match_returns_nothing():
    assert app_settings.search_rows("xyzzy-not-a-real-setting") == []


def test_settings_search_with_a_blank_query_returns_nothing():
    """A blank query means the filter is off; the rail draws its ordinary
    category list rather than an always-on "nothing matches"."""
    assert app_settings.search_rows("") == []
    assert app_settings.search_rows("   ") == []
