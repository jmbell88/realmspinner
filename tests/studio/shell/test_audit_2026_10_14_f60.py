"""The 2026-10-14 audit, fixer F60: the shell's frame-thread landings and quit paths.

Every test is named for the claim of the finding it closes, and each one fails
against the code as it was before the fix.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from realmspinner.studio import dialogs  # noqa: E402
from realmspinner.studio import main as main_mod  # noqa: E402
from realmspinner.studio.shell.tasks import TasksMixin  # noqa: E402
from realmspinner.studio.state import AppState  # noqa: E402
from realmspinner.studio.tasks import Done  # noqa: E402


class _Ctx:
    """The slice of ``Ctx`` the drop router and the landings touch, with
    ``TaskRunner.submit``'s one-key-at-a-time rule: a key already in flight is
    refused, silently, and the caller is told only by the return value."""

    def __init__(self, mode: str = "library") -> None:
        self.state = AppState()
        self.state.mode = mode
        self.submitted: list[tuple[str, Any, tuple]] = []
        self._inflight: set[str] = set()
        self.toasts: list[tuple[str, str]] = []
        self.svc = None

    def submit(self, key: str, fn: Any, *args: Any, tag: Any = None, **kwargs: Any) -> bool:
        if key in self._inflight:
            return False
        self._inflight.add(key)
        self.submitted.append((key, fn, args))
        return True

    def toast(self, text: str, level: str = "info", *a: Any, **k: Any) -> None:
        self.toasts.append((text, level))

    def toast_once(self, text: str, level: str = "info", *a: Any, **k: Any) -> bool:
        self.toast(text, level)
        return True


def _app(ctx: Any) -> Any:
    app = main_mod.App.__new__(main_mod.App)
    app.app_ctx = ctx
    return app


# --- shell-09: the splash loop ----------------------------------------------------


def test_a_splash_loop_failure_waits_for_the_load_before_teardown(monkeypatch):
    """The frame-thread splash loop raising (imgui, GL, ``set_mode``, Ctrl+C) went
    straight to teardown while ``Startup``'s thread was still in
    ``Runtime.start()``."""
    import pygame

    from realmspinner.studio import splash

    release = threading.Event()
    loaded = threading.Event()
    state = {"load_finished_when_the_loop_unwound": None}

    app = _app(None)
    app._min_size = (100, 100)
    app._restoring_phase = ""
    app.ctx = None
    app.imgui_renderer = None

    def slow_runtime(note: Any) -> None:
        release.wait(5)
        loaded.set()

    app.setup_runtime = slow_runtime
    monkeypatch.setattr(splash, "load_logo", lambda gl: None)

    def exploding_pump() -> list:
        # The loop has started; free the load a moment after the loop dies.
        threading.Timer(0.2, release.set).start()
        raise RuntimeError("imgui blew up")

    monkeypatch.setattr(pygame.event, "get", exploding_pump)

    with pytest.raises(RuntimeError, match="imgui blew up"):
        app._startup_with_splash()
    state["load_finished_when_the_loop_unwound"] = loaded.is_set()
    release.set()
    assert state["load_finished_when_the_loop_unwound"], (
        "teardown would have run Runtime.shutdown() concurrently with Runtime.start()"
    )


# --- service-gates-13: the pack scan is off the frame thread ----------------------


def _refresh_app(monkeypatch, calls: list[str]):
    from realmspinner.service import downloads as svc_downloads
    from realmspinner.service import packs as svc_packs

    monkeypatch.setattr(svc_downloads, "rows", lambda svc: [])

    def rows(svc: Any) -> list:
        calls.append(f"rows@{threading.current_thread().name}")
        return [{"key": "rig"}]

    monkeypatch.setattr(svc_packs, "rows", rows)
    monkeypatch.setattr(svc_packs, "packs_to_restore", lambda svc: ["rig"])
    ctx = _Ctx()
    ctx.base_models = []
    ctx.pack_rows = []
    ctx.packs_to_restore = []
    app = _app(ctx)
    app.runtime = SimpleNamespace(checks=[])
    app.svc = SimpleNamespace(vram_plan=None)
    app._refresh_style_lora_answers = lambda: None
    return app, ctx


def test_model_answers_refresh_reads_pack_metadata_off_the_frame_thread(monkeypatch):
    calls: list[str] = []
    app, ctx = _refresh_app(monkeypatch, calls)

    app._refresh_model_answers()

    assert calls == [], "svc_packs.rows scanned every distribution on the frame thread"
    from realmspinner.studio.shell.tasks import PACK_ANSWERS_KEY

    assert [key for key, *_ in ctx.submitted] == [PACK_ANSWERS_KEY]

    # The task is what reads them; its result lands on the frame thread.
    _, fn, args = ctx.submitted[0]
    result = fn(*args)
    assert calls == [f"rows@{threading.current_thread().name}"]
    TasksMixin._on_task_done(app, Done(PACK_ANSWERS_KEY, result=result))
    assert ctx.pack_rows == [{"key": "rig"}]
    assert ctx.packs_to_restore == ["rig"]


def test_a_pack_refresh_asked_while_one_is_in_flight_runs_again_when_it_lands(monkeypatch):
    calls: list[str] = []
    app, ctx = _refresh_app(monkeypatch, calls)
    from realmspinner.studio.shell.tasks import PACK_ANSWERS_KEY

    app._refresh_model_answers()
    app._refresh_model_answers()  # refused: the first is still running
    assert len(ctx.submitted) == 1

    ctx._inflight.discard(PACK_ANSWERS_KEY)  # the runner frees the key before polling
    _, fn, args = ctx.submitted[0]
    TasksMixin._on_task_done(app, Done(PACK_ANSWERS_KEY, result=fn(*args)))
    assert len(ctx.submitted) == 2, "the second ask was dropped"


# --- shell-shell-pkg-03: several files in one drop ---------------------------------


def test_two_meshes_dropped_together_are_both_imported_or_the_refused_one_is_reported(
    monkeypatch,
):
    from realmspinner.studio.modes.library.ui.panes import library

    imported: list[str] = []
    monkeypatch.setattr(
        library,
        "_read_and_import",
        lambda ctx, path: imported.append(path.name) or {"id": path.stem},
    )
    ctx = _Ctx("library")
    app = _app(ctx)
    app._on_drop = lambda path: main_mod.App._on_drop(app, path)

    app._on_drops([Path("a.glb"), Path("b.glb")])
    for _key, fn, args in ctx.submitted:  # every task the runner accepted, run
        fn(*args)

    errors = [text for text, level in ctx.toasts if level == "error"]
    assert set(imported) == {"a.glb", "b.glb"} or errors, (
        f"only {imported} were imported and nothing said so: {ctx.toasts}"
    )
    claimed = [text for text, _ in ctx.toasts if text.startswith("Importing")]
    for name in ("a.glb", "b.glb"):
        if name not in imported:
            assert not any(name in text for text in claimed), claimed


def test_a_mesh_drop_that_the_door_refused_does_not_toast_that_it_is_importing(monkeypatch):
    from realmspinner.studio.modes.library.ui.panes import library

    monkeypatch.setattr(library, "_read_and_import", lambda ctx, path: {"id": "x"})
    ctx = _Ctx("home")
    ctx._inflight.add(library.IMPORT_MESH_KEY)  # a picker or an import is already up
    app = _app(ctx)

    app._on_drop(Path("b.glb"))

    assert ctx.submitted == []
    assert not any(text.startswith("Importing") for text, _ in ctx.toasts), ctx.toasts


def test_a_batch_import_keeps_the_good_files_and_names_the_refused_one(monkeypatch):
    from realmspinner.studio.modes.library.ui.panes import library

    def fake(ctx: Any, path: Path) -> dict:
        if path.name == "bad.glb":
            raise ValueError("bad.glb is not a mesh")
        return {"id": path.stem}

    monkeypatch.setattr(library, "_read_and_import", fake)
    result = library._read_and_import_many(None, [Path("a.glb"), Path("bad.glb"), Path("c.glb")])
    assert result["id"] == "c" and result["imported"] == 2
    assert result["refused"] == [("bad.glb", "bad.glb is not a mesh")]

    ctx = _Ctx("library")
    ctx.cache = SimpleNamespace(invalidate=lambda: None)
    app = _app(ctx)
    app._request_storage = lambda *a, **k: None
    TasksMixin._on_task_done(app, Done(library.IMPORT_MESH_KEY, result=result))
    texts = [text for text, _ in ctx.toasts]
    assert any("bad.glb" in text for text in texts), texts
    assert any("2 meshes imported" in text for text in texts), texts


def test_two_samples_dropped_on_sirens_are_both_taken_or_the_refused_one_is_reported(
    monkeypatch,
):
    from realmspinner.studio.modes.sirens import mode as sirens_mode

    ctx = _Ctx("sirens")
    tab = SimpleNamespace(uid="tab1", busy=False)
    monkeypatch.setattr(sirens_mode, "active", lambda c: tab)
    app = _app(ctx)
    app._on_drop = lambda path: main_mod.App._on_drop(app, path)

    app._on_drops([Path("kick.wav"), Path("snare.wav")])

    errors = [text for text, level in ctx.toasts if level == "error"]
    assert len(ctx.submitted) == 2 or errors, (
        f"{len(ctx.submitted)} sample decode started and nothing said the other was refused"
    )


# --- shell-home-library-09: Delete in the trash ------------------------------------


def test_delete_in_the_trash_view_does_not_say_moved_to_trash():
    from realmspinner.studio.modes.library.ui.panes import library

    ctx = _Ctx("library")
    ctx.state.filters.trash = True
    ctx.state.selected = "job1"
    ctx.confirms = dialogs.ConfirmQueue()

    library.delete_asset(ctx, "job1")

    assert not any("Moved to trash" in text for text, _ in ctx.toasts), ctx.toasts
    assert ctx.submitted == [], "an already-trashed row was trashed again"
    # The question the row's own menu asks.
    assert ctx.confirms.pending is not None
    assert ctx.confirms.pending.title == "Delete permanently?"


def test_delete_outside_the_trash_still_moves_to_trash_with_an_undo():
    from realmspinner.studio.modes.library.ui.panes import library

    ctx = _Ctx("library")
    library.delete_asset(ctx, "job1")
    assert [key for key, *_ in ctx.submitted] == ["delete:job1"]
    assert ("Moved to trash.", "info") in ctx.toasts


# --- muse-mode-05: leaving Muse with the derive popup open ---------------------------


def test_leaving_muse_closes_the_derive_popup_so_modal_open_is_false_elsewhere():
    from realmspinner.studio import state as state_mod
    from realmspinner.studio.modes.muse import mode as muse_mode
    from realmspinner.studio.shell import events

    ctx = SimpleNamespace(
        state=AppState(),
        confirms=SimpleNamespace(pending=None),
        prompts=SimpleNamespace(pending=None),
    )
    ctx.state.mode = "muse"
    muse_mode.open_derive(ctx, "jobA", "retake")
    assert dialogs.modal_open(ctx)

    state_mod.set_mode_leave(lambda was: events._leave_mode_if_needed(ctx, was))
    try:
        assert state_mod.set_mode(ctx.state, "inker")
    finally:
        state_mod.set_mode_leave(None)

    assert muse_mode.ensure(ctx).derive_job == ""
    assert not dialogs.modal_open(ctx), "every KEYDOWN would be dropped with no popup on screen"


# --- shell-shell-pkg-04: an Inker export keeps the full cadence ---------------------


def test_frame_active_while_an_inker_export_is_being_stepped(monkeypatch):
    import pygame
    from imgui_bundle import imgui

    from realmspinner.studio.shell import frame as frame_mod

    monkeypatch.setattr(pygame.event, "peek", lambda: False)
    imgui_ctx = imgui.create_context()
    try:
        state = AppState()
        state.mode = "inker"
        state.inker = SimpleNamespace(export=None, active=None)
        probe = frame_mod.FrameMixin.__new__(frame_mod.FrameMixin)
        probe.app_ctx = SimpleNamespace(
            state=state,
            cache=SimpleNamespace(active=None),
            tasks=SimpleNamespace(busy_keys=()),
        )
        probe.runtime = SimpleNamespace(current_job_id=None)
        probe.viewer = None
        probe.clay_view = None
        probe.mason_view = None
        assert probe._frame_active() is False, "nothing is armed yet"

        state.inker.export = object()
        assert probe._frame_active() is True, "an armed export is stepped one frame per frame"

        state.mode = "library"  # armed state, not the mode: the file must still arrive
        assert probe._frame_active() is True

        state.inker.export = None
        assert probe._frame_active() is False
    finally:
        imgui.destroy_context(imgui_ctx)


# --- shell-shell-pkg-05 / -06: the quit chain ---------------------------------------


def _quit_app(*, commit_busy: bool = False):
    app = main_mod.App.__new__(main_mod.App)
    app._running = True
    app._quit_deferred = False
    app._quit_deferred_then = None
    app.runtime = SimpleNamespace(current_job_id=None)
    app.toasts = []
    app.queue = dialogs.ConfirmQueue()
    app._commit_busy = commit_busy
    app.app_ctx = SimpleNamespace(
        cache=SimpleNamespace(active=None),
        tasks=SimpleNamespace(busy_keys=set(), commit_busy=lambda prefix: app._commit_busy),
        confirms=app.queue,
        toast=lambda *a, **k: app.toasts.append(a),
    )
    app._quit_summary = lambda: ""
    return app


def test_a_second_window_close_press_does_not_stack_another_discard_confirm(monkeypatch):
    from realmspinner.studio import docmodes, mode_manifest
    from realmspinner.studio.panes import pose_panel

    app = _quit_app()

    def dirty_guard(ctx: Any, action: str, proceed: Any) -> bool:
        # What ``docmodes.guard`` does for a dirty document, through the real call.
        ctx.state = SimpleNamespace(
            doc=SimpleNamespace(any_dirty=True, docs=[SimpleNamespace(dirty=True)])
        )
        return docmodes.guard(ctx, "doc", "document", "documents", action, proceed)

    monkeypatch.setattr(pose_panel, "guard", dirty_guard)
    monkeypatch.setattr(mode_manifest, "DOC_MODES", ())

    app._ask_quit()
    assert app.queue.pending is not None and app.queue.waiting == 0
    app._ask_quit()  # the X pressed again while the question is up
    app._ask_quit()  # and again

    assert app.queue.waiting == 0, f"{app.queue.waiting + 1} identical modals queued"


def test_the_quit_chain_titles_are_the_ones_docmodes_asks_with():
    """``quit.QUIT_CHAIN_TITLES`` is the list the no-stacking rule matches by; the
    words belong to ``docmodes``."""
    from realmspinner.studio import docmodes
    from realmspinner.studio.shell.quit import QUIT_CHAIN_TITLES

    asked: list[str] = []
    ctx = SimpleNamespace(
        confirms=SimpleNamespace(ask=lambda confirm: asked.append(confirm.title)),
        state=SimpleNamespace(
            doc=SimpleNamespace(any_dirty=True, docs=[SimpleNamespace(dirty=True)])
        ),
    )
    docmodes.guard(ctx, "doc", "document", "documents", "quit", lambda: None)
    viewer = SimpleNamespace(pose_mode=True, editor=SimpleNamespace(has_unsaved_edits=lambda: True))
    docmodes.viewer_guard(ctx, viewer, "pose", "quit", lambda: None)

    assert len(asked) == 2
    assert set(asked) <= set(QUIT_CHAIN_TITLES)


def test_a_second_quit_press_during_a_pack_commit_keeps_the_install_continuation():
    app = _quit_app(commit_busy=True)
    launched: list[str] = []
    asked: list[Any] = []
    app._request_quit = lambda then=None: asked.append(then)
    launch = lambda: launched.append("installer")  # noqa: E731

    app._ask_quit(launch)  # "Update and quit", withheld for the commit phase
    app._ask_quit()  # then the window's X
    app._commit_busy = False
    app._resume_deferred_quit()

    assert asked == [launch], "the installer the user asked for was dropped"


def test_a_plain_quit_deferred_first_is_upgraded_by_a_later_install_quit():
    app = _quit_app(commit_busy=True)
    asked: list[Any] = []
    app._request_quit = lambda then=None: asked.append(then)

    def launch() -> None:
        return None

    app._ask_quit()
    app._ask_quit(launch)
    app._commit_busy = False
    app._resume_deferred_quit()

    assert asked == [launch]


# --- shell-shell-pkg-01: a failed Familiar task --------------------------------------


@pytest.mark.parametrize("key", ["familiar/chat", "familiar/build", "familiar/character"])
def test_a_failed_familiar_task_lands_on_the_dock_and_clears_thinking(key):
    from realmspinner.studio.assistant import ui as familiar_ui

    toasts: list[tuple] = []

    class Tasks:
        def poll(self) -> list:
            return [
                Done(
                    key,
                    error=RuntimeError("the card is busy"),
                    message="Familiar cannot start while a job holds the card.",
                    tag={"thread_key": None},
                )
            ]

    ctx = SimpleNamespace(
        state=AppState(),
        tasks=Tasks(),
        toast=lambda *a, **k: toasts.append(a),
        familiar_threads=None,
    )
    ui = familiar_ui.ensure(ctx)
    # The character door never sets ``thinking``; chat and build do.
    ui.thinking = {"familiar/chat": "chat", "familiar/build": "build"}.get(key, "")
    app = _app(ctx)

    app._collect_tasks()

    assert ui.thinking == "", "Send and Build stay disabled for the rest of the session"
    assert ui.message == "Familiar cannot start while a job holds the card."
    said = [t[0] for t in toasts]
    assert said.count("Familiar cannot start while a job holds the card.") == 1, said


# --- shell-shell-pkg-02: a failed compare parse -------------------------------------


def test_a_failed_compare_parse_clears_the_comparison_state():
    from realmspinner.studio.modes.library.ui.panes import library

    path = Path("model.glb")
    exits: list[bool] = []
    state = AppState()
    state.comparing = "job-b"
    state.compare_pending = path

    class Tasks:
        def poll(self) -> list:
            return [
                Done(
                    library.COMPARE_KEY,
                    error=RuntimeError("corrupt glb"),
                    message="Could not read that.",
                    tag=path,
                )
            ]

    ctx = SimpleNamespace(state=state, tasks=Tasks(), toast=lambda *a, **k: None)
    app = _app(ctx)
    app.viewer = SimpleNamespace(exit_compare=lambda: exits.append(True))

    app._collect_tasks()

    assert state.comparing is None, "the viewport stays split with an empty half"
    assert state.compare_pending is None
    assert exits == [True]


def test_a_failed_compare_parse_for_an_older_path_leaves_the_newer_compare_alone():
    from realmspinner.studio.modes.library.ui.panes import library

    newer = Path("newer.glb")
    state = AppState()
    state.comparing = "job-c"
    state.compare_pending = newer

    class Tasks:
        def poll(self) -> list:
            return [Done(library.COMPARE_KEY, error=RuntimeError("x"), tag=Path("older.glb"))]

    ctx = SimpleNamespace(state=state, tasks=Tasks(), toast=lambda *a, **k: None)
    app = _app(ctx)
    app.viewer = SimpleNamespace(exit_compare=lambda: None)
    app._collect_tasks()

    assert state.comparing == "job-c" and state.compare_pending == newer


# --- shell-shell-pkg-07: a late pose save --------------------------------------------


def _pose_app(*, job: str, bones: dict, root=None):
    ctx = SimpleNamespace(
        state=AppState(),
        cache=SimpleNamespace(invalidate=lambda: None),
        toast=lambda *a, **k: None,
    )
    app = _app(ctx)
    editor = SimpleNamespace(dirty=True, root_translation=lambda: list(root or [0.0, 0.0, 0.0]))
    app.viewer = SimpleNamespace(
        pose_mode=True, pose_job_id=job, editor=editor, get_pose=lambda: bones
    )
    app._refresh_rig_side_data = lambda: None
    return app, editor


def test_a_late_pose_save_does_not_mark_a_newer_edit_or_another_job_clean():
    from realmspinner.studio.panes import pose_panel

    def land(app: Any) -> None:
        # What ``pose_panel._save`` recorded when the save was submitted.
        pose_panel.remember_save("job1", {"arm": [0.0, 0.0, 0.0, 1.0]}, [0.0, 0.0, 0.0])
        TasksMixin._on_task_done(app, Done("pose-save:job1", result=None))

    # The editor still holds exactly what was written: now it is clean.
    app, editor = _pose_app(job="job1", bones={"arm": [0.0, 0.0, 0.0, 1.0]})
    land(app)
    assert editor.dirty is False

    # The user kept posing while the save waited on a bake.
    app, editor = _pose_app(job="job1", bones={"arm": [0.0, 0.7, 0.0, 0.7]})
    land(app)
    assert editor.dirty is True, "an edit made after the snapshot was marked saved"

    # ...or entered another asset's pose mode.
    app, editor = _pose_app(job="job2", bones={"arm": [0.0, 0.0, 0.0, 1.0]})
    land(app)
    assert editor.dirty is True, "another job's pose was marked saved"

    # The root offset is part of the pose.
    app, editor = _pose_app(job="job1", bones={"arm": [0.0, 0.0, 0.0, 1.0]}, root=[0.0, 0.2, 0.0])
    land(app)
    assert editor.dirty is True

    assert pose_panel.take_save("job1") is None, "the record must not outlive its landing"


def test_the_pose_save_records_what_it_wrote_for_the_landing():
    from realmspinner.studio.panes import pose_panel

    submitted: list[str] = []
    viewer = SimpleNamespace(
        editor=SimpleNamespace(current=None, root_translation=lambda: [0.0, 0.1, 0.0]),
        get_pose=lambda: {"arm": [0.0, 0.0, 0.0, 1.0]},
    )
    ctx = SimpleNamespace(
        svc=None,
        prompts=SimpleNamespace(ask=lambda prompt: prompt.on_accept("idle")),
        submit=lambda key, fn, *a, **k: submitted.append(key) or True,
    )

    pose_panel._save(ctx, {"id": "job1"}, viewer)

    assert submitted == ["pose-save:job1"]
    assert pose_panel.take_save("job1") == {
        "bones": {"arm": [0.0, 0.0, 0.0, 1.0]},
        "root_translation": [0.0, 0.1, 0.0],
    }


def test_a_pose_save_the_runner_refused_records_nothing():
    """A refused submit (that job's save already in flight) must not overwrite the
    snapshot of the one that is."""
    from realmspinner.studio.panes import pose_panel

    pose_panel.remember_save("job1", {"arm": [0.0, 0.0, 0.0, 1.0]}, [0.0, 0.0, 0.0])
    viewer = SimpleNamespace(
        editor=SimpleNamespace(current=None, root_translation=lambda: [9.0, 9.0, 9.0]),
        get_pose=lambda: {"arm": [1.0, 0.0, 0.0, 0.0]},
    )
    ctx = SimpleNamespace(
        svc=None,
        prompts=SimpleNamespace(ask=lambda prompt: prompt.on_accept("idle")),
        submit=lambda *a, **k: False,
    )
    pose_panel._save(ctx, {"id": "job1"}, viewer)
    assert pose_panel.take_save("job1")["root_translation"] == [0.0, 0.0, 0.0]


# --- mason-mode-01: a failed library-asset parse ---------------------------------------


def test_a_failed_library_asset_parse_landing_through_the_shell_marks_the_ref_missing_and_clears_pending(  # noqa: E501
    tmp_path,
):
    from realmspinner.studio.modes.mason import assets as mason_assets
    from realmspinner.studio.modes.mason.engine import refs

    submitted: dict[str, Any] = {}

    def submit(key: str, fn: Any, *args: Any, tag: Any = None, **kwargs: Any) -> bool:
        if key in submitted:
            return False
        submitted[key] = tag
        return True

    state = SimpleNamespace(mason=None, create=SimpleNamespace(workspace=None, submit_refusal=""))
    toasts: list[Any] = []
    svc = SimpleNamespace(config=SimpleNamespace(job_dir=lambda job_id: tmp_path / job_id))
    ctx = SimpleNamespace(
        svc=svc, submit=submit, state=state, toast=lambda *a, **k: toasts.append(a)
    )
    source = mason_assets.ensure(ctx)
    ref = refs.LibraryRef(job_id="0000000000de", artifact="model.glb")
    source.primitives(ref)
    key = f"{mason_assets.TASK_PREFIX}0000000000de:model.glb"
    tag = submitted[key]
    refkey = refs.ref_key(ref)
    assert refkey in source._pending

    class Tasks:
        def poll(self) -> list:
            return [Done(key, error=ValueError("not a glb"), message="That did not work.", tag=tag)]

    app_ctx = SimpleNamespace(
        tasks=Tasks(),
        state=state,
        toast=lambda *a, **k: toasts.append(a),
        svc=svc,
        mason_assets=source,
        submit=submit,
    )
    app = SimpleNamespace(app_ctx=app_ctx)

    TasksMixin._collect_tasks(app)

    assert refkey in source.missing, "the 'N missing' readout would stay empty"
    assert refkey not in source._pending, "every export would refuse it as still loading"
