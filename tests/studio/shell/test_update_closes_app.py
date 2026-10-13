"""Updating: the app closes itself for the installer, and the packs come back at launch.

Two complaints from the same sitting. Pressing Run Installer left Realmspinner
open, so the installer stalled on "still running" and told the user to close it
themselves; and the first launch after an update opened with Create, Muse and the
rig modes grey, because the installer wipes ``site-packages`` and the only remedy
was a Settings button nobody was pointed at.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from realmspinner.studio import dialogs, mode_manifest
from realmspinner.studio import main as main_mod


def _app(*, busy=(), commit_busy=False):
    app = main_mod.App.__new__(main_mod.App)
    app._running = True
    app._quit_deferred = False
    app._quit_deferred_then = None
    app.runtime = SimpleNamespace(current_job_id=None)
    app.toasts = []
    app.queue = dialogs.ConfirmQueue()
    app.app_ctx = SimpleNamespace(
        cache=SimpleNamespace(active=None),
        tasks=SimpleNamespace(busy_keys=set(busy), commit_busy=lambda prefix: commit_busy),
        confirms=app.queue,
        toast=lambda *a, **k: app.toasts.append(a),
    )
    return app


@pytest.fixture
def opened(monkeypatch):
    """Stands in for ``os.startfile`` -- no test may launch a real installer."""
    seen: list[str] = []
    monkeypatch.setattr(os, "startfile", seen.append, raising=False)
    # No document guards: this is about the chain's last step, and the guards'
    # own coverage is test_every_doc_mode_has_a_guard_in_the_quit_chain.
    monkeypatch.setattr(mode_manifest, "DOC_MODES", ())
    return seen


def test_an_idle_app_opens_the_installer_then_exits(opened):
    app = _app()
    app._quit_to_install("C:/updates/RealmspinnerSetup-v9.9.9.exe")
    assert opened == ["C:/updates/RealmspinnerSetup-v9.9.9.exe"]
    assert app._running is False, "the installer cannot replace files a running app holds"


def test_an_installer_that_will_not_open_leaves_the_app_running(monkeypatch):
    """Closing on the strength of an installer that never appeared would leave
    the user with neither the app nor the update."""
    monkeypatch.setattr(mode_manifest, "DOC_MODES", ())

    def refuse(_path):
        raise OSError("no association")

    monkeypatch.setattr(os, "startfile", refuse, raising=False)
    app = _app()
    app._quit_to_install("C:/updates/setup.exe")
    assert app._running is True
    assert app.toasts, "the failure has to be said"


def test_staying_on_the_quit_question_opens_no_installer(opened):
    app = _app(busy={"clay-exportfile:tab-1"})
    app._quit_to_install("C:/updates/setup.exe")
    assert app.queue.pending is not None
    assert opened == [] and app._running is True
    app.queue.dismiss()  # "Stay"
    assert opened == [] and app._running is True


def test_confirming_the_quit_question_opens_the_installer_and_exits(opened):
    app = _app(busy={"clay-exportfile:tab-1"})
    app._quit_to_install("C:/updates/setup.exe")
    app.queue.pending.on_confirm()
    assert opened == ["C:/updates/setup.exe"]
    assert app._running is False


def test_a_pack_commit_defers_the_install_not_just_the_exit(opened):
    """A quit withheld for pip's commit phase must still end in the installer,
    not in a plain exit that forgets why the user asked."""
    app = _app(commit_busy=True)
    app._quit_to_install("C:/updates/setup.exe")
    assert opened == [] and app._running is True
    app.app_ctx.tasks.commit_busy = lambda prefix: False
    app._resume_deferred_quit()
    assert opened == ["C:/updates/setup.exe"]
    assert app._running is False


def test_a_plain_quit_is_unchanged_by_the_install_hand_off(opened):
    app = _app()
    app._ask_quit()
    assert opened == [] and app._running is False


def test_the_settings_button_goes_through_the_app_not_straight_to_the_shell():
    import inspect

    from realmspinner.studio.modes.settings.ui.panes import app_settings

    source = inspect.getsource(app_settings._update_result)
    assert "quit_to_install(str(ready))" in source
    assert "Close and Install##update-run" in source


# --- the packs come back before the app finishes loading ---------------------


def test_the_splash_restores_packs_after_the_service_exists_and_before_the_context(monkeypatch):
    import inspect

    assert "_restore_packs_behind_splash" in inspect.getsource(main_mod.App.setup_runtime)
    # ``setup_context`` snapshots ``pack_rows``; it must not be what restores.
    assert "restore_at_startup" not in inspect.getsource(main_mod.App.setup_context)


def test_a_splash_quit_stops_a_restore_that_is_still_downloading(monkeypatch):
    from realmspinner import winjob

    killed: list[str] = []
    monkeypatch.setattr(winjob, "terminate_tracked", lambda prefix=None: killed.append(prefix))
    app = main_mod.App.__new__(main_mod.App)
    app._restore_cancelled = False

    app._restoring_phase = ""
    app._cancel_startup_restore()
    assert killed == [] and app._restore_cancelled is False

    app._restoring_phase = "commit"
    app._cancel_startup_restore()
    assert killed == [], "pip is writing into site-packages; that is waited out"

    app._restoring_phase = "download"
    app._cancel_startup_restore()
    assert killed == ["pack install"] and app._restore_cancelled is True
