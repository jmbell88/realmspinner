"""Thread memory, and its one integration point with the real tab lifecycle.

Memory is one thread per document tab, for the session only; non-document
modes share the "Studio" thread. ``test_a_closed_tab_drops_its_thread`` drives
the real ``docmodes.close_tab`` (T3's hook), because the claim being tested is
about the *wiring* -- a closed tab actually ending its thread -- not just
about ``Threads.drop`` in isolation.
"""

from __future__ import annotations

from typing import Any

from realmspinner.familiar import threads as familiar_threads
from realmspinner.studio import docmodes


def test_a_turn_round_trips_through_a_thread():
    t = familiar_threads.Threads()
    key = t.key_for("clay", "bd1")
    t.append(key, familiar_threads.Turn(role="user", text="make a barrel"))
    t.append(key, familiar_threads.Turn(role="familiar", text="here you go"))
    assert t.get(key) == (
        familiar_threads.Turn(role="user", text="make a barrel"),
        familiar_threads.Turn(role="familiar", text="here you go"),
    )


def test_an_unknown_thread_is_empty_not_a_key_error():
    t = familiar_threads.Threads()
    assert t.get(("clay", "nope")) == ()


def test_a_tabless_mode_shares_the_studio_thread():
    t = familiar_threads.Threads()
    assert familiar_threads.Threads.key_for("home", "") == familiar_threads.STUDIO
    assert familiar_threads.Threads.key_for("settings", "") == familiar_threads.STUDIO

    home_key = familiar_threads.Threads.key_for("home", "")
    settings_key = familiar_threads.Threads.key_for("settings", "")
    t.append(home_key, familiar_threads.Turn(role="user", text="hi"))
    assert t.get(settings_key) == t.get(home_key)


def test_clear_forgets_every_thread():
    t = familiar_threads.Threads()
    t.append(("clay", "bd1"), familiar_threads.Turn(role="user", text="x"))
    t.append(familiar_threads.STUDIO, familiar_threads.Turn(role="user", text="y"))
    t.clear()
    assert t.get(("clay", "bd1")) == ()
    assert t.get(familiar_threads.STUDIO) == ()


# --- wired through the real docmodes.close_tab -----------------------------


class _Tab:
    def __init__(self, uid: str, *, dirty: bool = False, saving: bool = False) -> None:
        self.uid = uid
        self.dirty = dirty
        self.saving = saving
        self.title = uid


class ClayState:
    """Named for a real ``*_state`` module: ``docmodes._mode_for`` reads
    "clay" straight off this class's name, the same way it would off the
    real ``clay_state.ClayState``."""

    def __init__(self, *tabs: _Tab) -> None:
        self._tabs = {tab.uid: tab for tab in tabs}

    def get(self, uid: str) -> Any:
        return self._tabs.get(uid)

    def close(self, uid: str) -> None:
        self._tabs.pop(uid, None)


class _FakeCtx:
    def __init__(self) -> None:
        self.confirms = _Confirms()

    def toast(self, *_args: Any, **_kwargs: Any) -> None:
        pass


class _Confirm:
    def __init__(self, on_confirm: Any) -> None:
        self.on_confirm = on_confirm


class _Confirms:
    def __init__(self) -> None:
        self.pending: _Confirm | None = None

    def ask(self, confirm: Any) -> None:
        self.pending = confirm


def _register(t: familiar_threads.Threads):
    docmodes.TAB_CLOSED.append(t.drop)
    return t


def test_a_closed_tab_drops_its_thread():
    t = familiar_threads.Threads()
    _register(t)
    try:
        other_key = ("clay", "bd2")
        closing_key = ("clay", "bd1")
        t.append(closing_key, familiar_threads.Turn(role="user", text="doomed"))
        t.append(other_key, familiar_threads.Turn(role="user", text="survives"))
        t.append(familiar_threads.STUDIO, familiar_threads.Turn(role="user", text="studio"))

        ctx = _FakeCtx()
        state = ClayState(_Tab("bd1"), _Tab("bd2"))
        docmodes.close_tab(ctx, state, "bd1", lambda tab: None)

        assert t.get(closing_key) == ()
        assert t.get(other_key) != ()
        assert t.get(familiar_threads.STUDIO) != ()
    finally:
        docmodes.TAB_CLOSED.remove(t.drop)


def test_a_refused_close_keeps_its_thread():
    """A tab still saving is refused outright (``CLOSE_WHILE_SAVING``); its
    thread must survive because the tab itself does."""
    t = familiar_threads.Threads()
    _register(t)
    try:
        key = ("clay", "bd1")
        t.append(key, familiar_threads.Turn(role="user", text="mid-save"))

        ctx = _FakeCtx()
        state = ClayState(_Tab("bd1", saving=True))
        docmodes.close_tab(ctx, state, "bd1", lambda tab: None)

        assert t.get(key) != ()
        assert state.get("bd1") is not None
    finally:
        docmodes.TAB_CLOSED.remove(t.drop)


def test_a_cancelled_close_keeps_its_thread():
    """Dirty and asked-about, but never confirmed: the tab is still open, so
    the listener must not have fired yet."""
    t = familiar_threads.Threads()
    _register(t)
    try:
        key = ("clay", "bd1")
        t.append(key, familiar_threads.Turn(role="user", text="unsaved"))

        ctx = _FakeCtx()
        state = ClayState(_Tab("bd1", dirty=True))
        docmodes.close_tab(ctx, state, "bd1", lambda tab: None)

        assert t.get(key) != ()
        assert ctx.confirms.pending is not None
        assert state.get("bd1") is not None
    finally:
        docmodes.TAB_CLOSED.remove(t.drop)
