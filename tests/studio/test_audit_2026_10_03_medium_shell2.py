"""Regression tests for the 2026-10-03 audit's second batch of Medium shell
findings (shell-21, -22, -23, -24 and -34): Home, the Library, the journal's
Discard all, the bulk-export toast and Create's arrow keys."""

from __future__ import annotations

import json
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import pygame

from realmspinner.studio import journal, main
from realmspinner.studio.modes.home.ui.panes import landing
from realmspinner.studio.state import AppState, Filters, _valid_filter_options

from .test_studio_plumbing import FakeApp

# --- shell-21 ----------------------------------------------------------------


def test_a_library_bulk_export_toast_names_the_destination_not_the_result_dict():
    """``bulk_export`` and ``export_planned_to_folder`` return dicts, and the
    task-done branch formatted them with an f-string: the toast read
    ``Exported to {'path': 'D:\\out\\realmspinner_export.zip', 'files': 3}``."""
    app = FakeApp()
    app.dispatch("export-zip", {"path": "D:/out/realmspinner_export.zip", "files": 3})
    message, _level = app.toasts[-1]
    assert "D:/out/realmspinner_export.zip" in message
    assert "{" not in message and "'path'" not in message

    app = FakeApp()
    app.dispatch("export-folder", {"copied": 2, "dir": "D:/game/assets", "degraded": []})
    message, level = app.toasts[-1]
    assert "D:/game/assets" in message
    assert "{" not in message and "'dir'" not in message
    assert level == "info"


def test_a_bulk_export_toast_words_the_degraded_meshes():
    app = FakeApp()
    app.dispatch(
        "export-folder",
        {"copied": 2, "dir": "D:/game/assets", "degraded": ["aaaaaaaaaaaa", "bbbbbbbbbbbb"]},
    )
    message, level = app.toasts[-1]
    assert "D:/game/assets" in message
    assert "2 meshes" in message
    assert "aaaaaaaaaaaa" not in message and "[" not in message
    assert level == "warn"


def test_a_plain_path_export_result_still_toasts_its_path():
    """export-convert and export-godot return a Path or a string."""
    app = FakeApp()
    app.dispatch("export-convert", "D:/out/converted")
    assert any("D:/out/converted" in message for message, _ in app.toasts)


# --- shell-22 ----------------------------------------------------------------


def _queue_ctx(status: str) -> SimpleNamespace:
    job = {"id": "q1", "name": "a chest", "status": status}
    return SimpleNamespace(cache=SimpleNamespace(active=job), progress=None)


def test_the_home_queue_row_for_a_queued_job_opens_a_library_that_shows_it():
    """The row said "queued: ..." but set the Library's status filter to
    "running", which ``Filters.matches`` compares by equality: the click landed
    on "Nothing matches." and a stale "running" filter was saved."""
    job = {"id": "q1", "name": "a chest", "status": "queued"}
    row = landing._queue_status(_queue_ctx("queued"))
    filters = Filters(status=row.status_filter)
    assert filters.matches(job), f"filter {row.status_filter!r} hides the queued job"
    assert row.status_filter in _valid_filter_options()["status"]

    running = {"id": "r1", "name": "a chest", "status": "running"}
    row = landing._queue_status(_queue_ctx("running"))
    assert Filters(status=row.status_filter).matches(running)


# --- shell-23 ----------------------------------------------------------------


def test_home_resume_lists_a_create_workspace_candidate_the_library_shows():
    """``Filters.matches`` lets a ``candidate_group`` row carrying
    ``params["create_workspace"]`` into the library; Home's Resume list still
    dropped every candidate row, so Create-workspace work was missing from
    "what was I working on"."""
    job = {
        "id": "ws1",
        "status": "done",
        "stage": "model",
        "name": "workspace mesh",
        "created_at": 5.0,
        "candidate_group": "g1",
        "params": {"create_workspace": "w1"},
    }
    undecided = dict(job, id="c1", name="undecided", params={})
    assert Filters().matches(job)
    assert not Filters().matches(undecided)
    rows = landing._asset_rows(SimpleNamespace(cache=SimpleNamespace(jobs=[job, undecided])))
    assert [row.key for row in rows] == ["ws1"]


# --- shell-24 ----------------------------------------------------------------


def _crash_copy(root, title: str, kind: str):
    payload = root / f"{title}.probe"
    payload.write_bytes(b"unsaved work")
    journal.meta_path(payload).write_text(
        json.dumps(
            {"version": journal.VERSION, "kind": kind, "title": title, "uid": title, "at": 1.0}
        ),
        encoding="utf-8",
    )
    return journal.Recovered(path=payload, kind=kind, title=title, at=1.0, meta={"kind": kind})


def test_discard_all_leaves_a_crash_copy_no_provider_can_adopt(tmp_path, monkeypatch):
    """ "Discard all" ran ``journal.discard`` over every snapshot row, including
    the ones ``_recovery_row`` drew as "unavailable" -- a copy the manual says
    is left alone for a build that can open it."""
    adoptable = _crash_copy(tmp_path, "adoptable", "probe-kind")
    orphan = _crash_copy(tmp_path, "orphan", "no-such-kind")
    provider = journal.Provider(
        kind="probe-kind",
        ext=".probe",
        label="probe",
        slots=lambda ctx: [],
        uid_of=lambda s: "x",
        title_of=lambda s: "x",
        head_of=lambda s: 1,
        encode=lambda s: b"",
        adopt=lambda ctx, path, meta: True,
    )
    before = dict(journal._PROVIDERS)
    journal.register(provider)
    try:
        assert adoptable.adoptable and not orphan.adoptable
        toasts: list[tuple[str, str]] = []
        ctx = SimpleNamespace(
            state=SimpleNamespace(recovery=[adoptable, orphan]),
            toast=lambda text, level="info", **_kw: toasts.append((text, level)),
            svc=SimpleNamespace(config=SimpleNamespace(autosave_dir=tmp_path)),
        )
        # The real ``_recovery`` with the drawing stubbed out: the only real
        # interaction is the Discard all press.
        for name in ("widgets", "imgui", "fonts"):
            monkeypatch.setattr(landing, name, MagicMock())
        monkeypatch.setattr(landing, "_recovery_row", lambda *a, **k: None)
        monkeypatch.setattr(
            landing.controls, "button", lambda label, *a, **k: "Discard all" in str(label)
        )

        landing._recovery(ctx)

        assert not adoptable.path.exists()
        assert orphan.path.exists(), "a copy nothing can adopt was unlinked"
        assert journal.meta_path(orphan.path).exists()
        assert [row.path for row in ctx.state.recovery] == [orphan.path]
    finally:
        journal._PROVIDERS.clear()
        journal._PROVIDERS.update(before)


# --- shell-34 ----------------------------------------------------------------


def test_create_arrow_keys_do_not_move_a_selection_through_an_undrawn_library(monkeypatch):
    """Create draws no library list any more, but Up and Down still fell
    through to ``library.select_relative`` and walked ``state.selected`` through
    the Library's workshop while the viewer and inspector changed underfoot."""
    moved: list[int] = []
    monkeypatch.setattr(
        "realmspinner.studio.modes.library.ui.panes.library.select_relative",
        lambda ctx, delta: moved.append(delta),
    )
    state = AppState()
    state.mode = state.mode_observed = state.previous_mode = "create"
    state.selected = "job-1"
    app = SimpleNamespace(
        app_ctx=SimpleNamespace(state=state, cache=SimpleNamespace(get=lambda _id: None)),
        viewer=SimpleNamespace(pose_mode=False, exit_compare=lambda: None),
    )
    for name in ("_note_mode", "_set_mode", "_escape_mode"):
        setattr(app, name, MethodType(getattr(main.App, name), app))
    for key in (pygame.K_UP, pygame.K_DOWN):
        main.App._shortcut(app, pygame.event.Event(pygame.KEYDOWN, key=key, mod=0))
    assert moved == []
    assert state.selected == "job-1"
