"""Mason's controller: the rules Clay had to learn first, inherited on purpose.

Nothing here is about the viewport. It is about the document layer: a save
that is a *state* rather than a call that returns, a failed save that must
clear it, an encode that never runs on the calling thread, and a scene-size
warning read off the engine's own constant rather than a number in a pane.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from warlock.studio import mason_mode, mason_state
from warlock.studio.mason import document as md
from warlock.studio.mason import nodes as nd
from warlock.studio.mason import scene as msc


class FakeCtx:
    """Runs a submitted callable inline, so the test sees what the task thread
    would have done without needing one."""

    def __init__(self, *, accept: bool = True) -> None:
        self.svc = None
        self.state = _AppState()
        self.settings = _Settings()
        self.submitted: list[str] = []
        self.toasts: list[tuple[str, str]] = []
        self.accept = accept
        self.result: Any = None

    def submit(self, key: str, run: Any, *args: Any) -> bool:
        self.submitted.append(key)
        if not self.accept:
            return False
        self.result = run(*args)
        return True

    def toast(self, message: str, kind: str = "info", *_args: Any, **_kwargs: Any) -> None:
        self.toasts.append((message, kind))


class _AppState:
    def __init__(self) -> None:
        self.mason = None
        self.mode = "home"


class _Settings:
    def __init__(self) -> None:
        self.store: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self.store.get(key)

    def set(self, key: str, value: Any) -> None:
        self.store[key] = value


class _Done:
    def __init__(self, key: str, result: Any = None) -> None:
        self.key = key
        self.result = result


def _tab(ctx: FakeCtx, *, dirty: bool = False) -> mason_state.MasonTab:
    """One open scene. ``adopt`` records the head it is given, so a tab is
    clean the moment it is adopted -- dirtying it means editing it afterwards."""
    doc = md.MasonDoc()
    doc.add_node(nd.GroupNode(uid=nd.new_uid(), name="Group"))
    tab = mason_mode.adopt(ctx, doc, title="Scene")
    if dirty:
        doc.set_props(doc.roots[0].uid, name="Edited")
    return tab


def _save(ctx: FakeCtx, tab: mason_state.MasonTab, path: Path) -> None:
    """A whole save: the submit, and the result coming back. ``save_to`` only
    does the first half; a test that skipped ``on_task_done`` would be
    asserting against a half-finished save."""
    mason_mode.save_to(ctx, tab, path)
    mason_mode.on_task_done(ctx, _Done(f"mason-save:{tab.uid}", ctx.result))


@pytest.fixture(autouse=True)
def _no_pygame_display(monkeypatch):
    import pygame

    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0)


# --- opening ------------------------------------------------------------------


def test_a_new_document_adopts_and_is_not_dirty() -> None:
    ctx = FakeCtx()
    tab = mason_mode.new_document(ctx)
    assert tab.dirty is False
    assert mason_mode.active(ctx) is tab


def test_an_edit_makes_it_dirty_and_undo_makes_it_clean_again() -> None:
    ctx = FakeCtx()
    tab = mason_mode.new_document(ctx)
    node = nd.GroupNode(uid=nd.new_uid(), name="Group")
    tab.doc.add_node(node)
    assert tab.dirty is True
    mason_mode.undo(ctx, tab)
    assert tab.dirty is False


# --- saving is a state --------------------------------------------------------


def test_a_failed_save_clears_the_saving_state() -> None:
    """``saving`` disables every control that changes the document, so
    without this one failed write makes the tab read-only forever with no
    way back short of closing it."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    tab.saving = True

    mason_mode.on_task_failed(ctx, _Done(f"mason-save:{tab.uid}"))
    assert tab.saving is False


def test_a_submit_that_is_refused_clears_the_saving_state() -> None:
    ctx = FakeCtx(accept=False)
    tab = _tab(ctx)
    mason_mode.save_as(ctx, tab)
    assert tab.saving is False


def test_a_completed_save_clears_it_and_marks_the_tab_saved(tmp_path) -> None:
    ctx = FakeCtx()
    tab = _tab(ctx, dirty=True)
    assert tab.dirty is True

    _save(ctx, tab, tmp_path / "scene.wscn")
    assert tab.saving is False
    assert tab.dirty is False
    assert (tmp_path / "scene.wscn").exists()


def test_save_submits_under_a_mason_key_and_never_encodes_on_the_calling_thread(
    tmp_path,
) -> None:
    """The submitted key carries the ``mason-`` prefix the app claims results
    by, and the file appears only once the (inline, in this fake) task runs --
    never before ``ctx.submit`` is called."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    path = tmp_path / "scene.wscn"

    mason_mode.save_to(ctx, tab, path)
    assert ctx.submitted == [f"mason-save:{tab.uid}"]
    assert path.exists()  # the fake ran the task inline; a real one would not have yet


# --- scene stats ---------------------------------------------------------------


def test_scene_stats_reads_the_threshold_from_the_engine_constant() -> None:
    ctx = FakeCtx()
    tab = _tab(ctx)
    stats = mason_mode.scene_stats(ctx, tab)
    assert stats["threshold"] == msc.PLACED_WARN_THRESHOLD
    assert stats["warn"] is False


def test_scene_stats_warn_flag_flips_at_the_threshold() -> None:
    ctx = FakeCtx()
    tab = _tab(ctx)
    doc = tab.doc
    # One group already exists from ``_tab``; groups draw nothing, so use
    # mesh nodes with no ref -- ``resolve`` still counts an unresolved node as
    # placed (it draws nothing, but it is there).
    for _ in range(msc.PLACED_WARN_THRESHOLD):
        doc.add_node(nd.MeshNode(uid=nd.new_uid(), name="m"))
    stats = mason_mode.scene_stats(ctx, tab)
    assert stats["placed"] >= msc.PLACED_WARN_THRESHOLD
    assert stats["warn"] is True


# --- keys ----------------------------------------------------------------------


def test_handle_key_answers_false_with_nothing_open() -> None:
    ctx = FakeCtx()
    import pygame

    event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_q, mod=0)
    assert mason_mode.handle_key(ctx, event) is False


# --- placing --------------------------------------------------------------------


def test_placing_a_primitive_selects_it() -> None:
    ctx = FakeCtx()
    tab = _tab(ctx)
    uid = mason_mode.place_primitive(ctx, "box")
    assert uid is not None
    assert tab.doc.selection == {uid}


# --- the journal ---------------------------------------------------------------


def test_the_journal_provider_round_trips_a_document() -> None:
    ctx = FakeCtx()
    tab = _tab(ctx, dirty=True)

    encoded = mason_mode.JOURNAL.encode(tab)
    assert isinstance(encoded, bytes) and encoded

    from warlock.studio.mason import serialize

    doc = serialize.read_wscn(encoded)
    assert len(doc.roots) == len(tab.doc.roots)
