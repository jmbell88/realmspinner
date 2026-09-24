"""Editor-first host shell: shared commands, status, and compact navigation."""

from __future__ import annotations

from types import SimpleNamespace


def _ctx():
    return SimpleNamespace(
        state=SimpleNamespace(
            mode="home",
            selected=None,
            wireframe=False,
            turntable=False,
            show_fps=False,
            filters=SimpleNamespace(trash=False),
            errors=[],
        ),
        cache=SimpleNamespace(get=lambda _key: None, jobs=[]),
        runtime=SimpleNamespace(checks=[]),
        viewer=None,
    )


def test_workspace_menu_lists_every_named_mode():
    from realmspinner.studio import menus, modes

    app_ctx = _ctx()
    rows = menus.specs(app_ctx)
    workspace = {
        row.identity.removeprefix("command:go:")
        for row in rows
        if row.identity.startswith("command:go:")
    }
    assert workspace == set(modes.KEYS)


def test_shared_command_specs_keep_palette_state_and_reason():
    from realmspinner.studio import menus, palette

    app_ctx = _ctx()
    commands = {command.key: command for command in palette.commands(app_ctx)}
    rows = {
        row.identity.removeprefix("command:"): row
        for row in menus.specs(app_ctx)
        if row.identity.startswith("command:")
    }
    for key in ("save", "undo", "wireframe", "manual", "quit"):
        assert rows[key].enabled == bool(commands[key].enabled(app_ctx))
        assert rows[key].shortcut == commands[key].hint
        assert rows[key].disabled_reason == commands[key].why


def test_status_reports_queue_and_health_without_permanent_ok_noise():
    from realmspinner.doctor import Check
    from realmspinner.studio import status_bar

    app_ctx = _ctx()
    app_ctx.cache.jobs = [{"status": "queued"}, {"status": "running"}]
    app_ctx.runtime.checks = [Check("CUDA", True, "ready", fatal=False)]
    keys = {item.key for item in status_bar.items(app_ctx)}
    assert "queue" in keys
    assert "health" not in keys
    app_ctx.runtime.checks = [Check("CUDA", False, "missing", fatal=False)]
    assert any(item.key == "health" and item.warning for item in status_bar.items(app_ctx))


def test_the_status_bar_names_what_the_running_job_is_doing():
    """W3.7: "Queue 1 active" alone never said what that job was *doing*.

    The name is the running job's own progress label, not a stage. A job row's
    ``stage`` is the record vocabulary (``model``) while
    ``create_stages.LABELS`` is the UI one (``mesh``), and ``create_stages``
    keeps the two apart on purpose -- translating between them here would be
    exactly the second representation of "how far along is this" that module
    exists to prevent. Progress already publishes a sentence for a human.
    """
    from realmspinner.studio import status_bar

    app_ctx = _ctx()
    app_ctx.cache.jobs = [
        {"status": "running", "stage": "model", "progress": {"label": "Generating"}},
        {"status": "queued"},
        {"status": "queued"},
    ]
    text = next(item.text for item in status_bar.items(app_ctx) if item.key == "queue")
    assert text == "Queue 1 active (Generating) / 2 waiting"

    # Its own case is kept: lowercasing mangles the labels that carry "3D".
    app_ctx.cache.jobs = [
        {"status": "running", "progress": {"label": "Starting 3D generation"}},
        {"status": "queued"},
    ]
    text = next(item.text for item in status_bar.items(app_ctx) if item.key == "queue")
    assert text == "Queue 1 active (Starting 3D generation) / 1 waiting"

    # No progress on the running row -> the old wording, unchanged. A row
    # carrying a stage but no live progress is not the job the worker is on.
    app_ctx.cache.jobs = [{"status": "running", "stage": "model"}, {"status": "queued"}]
    text = next(item.text for item in status_bar.items(app_ctx) if item.key == "queue")
    assert text == "Queue 1 active / 1 waiting"


def test_a_fresh_layout_prefers_the_44dp_icon_rail():
    from realmspinner.studio import layout, rail

    class Settings:
        def get(self, key, default=None):
            return default

        def set(self, key, value):
            pass

    assert layout.Layout(Settings()).rail == "icons"
    assert rail.RAIL_W == 44.0


def test_an_inker_row_that_is_a_document_state_reports_its_tick():
    """``MenuSpec.checked`` was hardcoded False for every Inker op, so the one
    row that is a *setting* rather than an action drew no tick and the user had
    no way to see which way it was set."""
    from realmspinner.kernels import pixel as inker
    from realmspinner.studio import menus
    from realmspinner.studio.modes.inker import state as inker_state

    app_ctx = _ctx()
    app_ctx.state.mode = "inker"
    state = inker_state.InkerState()
    tab = inker_state.InkerDoc(doc=inker.Document.blank(8, 8), uid="t1", title="Untitled")
    state.add(tab)
    app_ctx.state.inker = state

    def _row():
        return next(
            row for row in menus.specs(app_ctx) if row.identity == "inker:toggle_matte"
        )

    assert _row().checked is False
    assert tab.doc.toggle_matte() is True
    assert _row().checked is True


def test_the_resource_meter_is_not_one_of_the_elided_status_items():
    """It is right-anchored, and that is the whole point.

    ``items`` is drawn left to right with the tail dropped as the window
    narrows, so a meter in that list would be the *first* thing to go --
    backwards for the one figure a user consults while deciding whether to
    start a generation. ``overlay.doctor_banner``'s rule instead: reserve the
    trailing item, then trim the leading detail.
    """
    from realmspinner.studio import resources, status_bar

    app_ctx = _ctx()
    app_ctx.state.show_resources = True
    app_ctx.resources = resources.Sampler()
    app_ctx.resources.reading = resources.Reading(
        vram_used_gib=9.2, vram_total_gib=32.0, ram_used_gib=23.4, ram_total_gib=64.0, cpu=0.07
    )

    assert "resources" not in {item.key for item in status_bar.items(app_ctx)}
    item = status_bar.resource_item(app_ctx)
    assert item is not None and item.key == "resources"
    assert item.text == "VRAM 9.2/32   RAM 23.4/64   CPU 7%"
    assert not item.warning, "a machine being busy is not a fault"


def test_the_meter_is_an_opt_out_and_omits_what_it_cannot_read():
    """Off costs nothing, and a figure absent is a figure left out."""
    from realmspinner.studio import resources, status_bar

    app_ctx = _ctx()
    app_ctx.resources = resources.Sampler()
    app_ctx.resources.reading = resources.Reading(ram_used_gib=1.0, ram_total_gib=8.0)

    app_ctx.state.show_resources = False
    assert status_bar.resource_item(app_ctx) is None
    app_ctx.state.show_resources = True
    # No NVIDIA card and no interval yet: RAM alone, rather than a row of
    # dashes pretending to three readings.
    assert status_bar.resource_item(app_ctx).text == "RAM 1.0/8"
    # Nothing readable at all is no item, not an empty one.
    app_ctx.resources.reading = resources.Reading()
    assert status_bar.resource_item(app_ctx) is None


def test_the_sampler_holds_its_cadence_and_its_own_cpu_baseline():
    """One sampler per app: the CPU figure is a delta between calls."""
    from realmspinner.studio import resources

    sampler = resources.Sampler()
    first = sampler.tick(1000.0)
    # Inside the window: the previous reading, not a fresh syscall.
    assert sampler.tick(1000.0 + resources.TICK_SECONDS / 2) is first
    assert sampler.tick(1000.0 + resources.TICK_SECONDS * 2) is sampler.reading
    assert resources.TICK_SECONDS == 1.0


def test_a_document_name_never_carries_its_imgui_id():
    """Every mode, which is the half that was missing.

    A tab's label is keyed for the widget that draws it -- ``Untitled##pd1`` in
    Inker, ``Untitled###pl1`` in Plotter -- and the status bar printed the
    Plotter form verbatim, because only the Inker branch split it. Both
    branches go through one function now, and this is what compares them.
    """
    from types import SimpleNamespace

    from realmspinner.studio import status_bar

    assert status_bar._document_name(SimpleNamespace(label="Untitled##pd1")) == "Untitled"
    assert status_bar._document_name(SimpleNamespace(label="Untitled###pl1")) == "Untitled"
    assert status_bar._document_name(SimpleNamespace(label="level.tmx")) == "level.tmx"
    # A label that is *only* an id still has to say something.
    assert status_bar._document_name(SimpleNamespace(label="###pl1")) == "Untitled"
    assert status_bar._document_name(SimpleNamespace()) == "Untitled"


def test_the_frame_rate_rides_the_meter_and_is_omitted_when_unknown():
    """The fourth field, and the one the sampler is *given* rather than reads.

    Omitted rather than zeroed when it is unknown: the screenshot harness calls
    ``frame`` without ever calling ``_tick``, so the meter has no recorded
    frame there, and a confident "0 fps" in every shipped picture would be a
    claim about the app that is not true. ``Reading.text`` already drops any
    figure it could not read, and this is that rule extended by one.
    """
    from realmspinner.studio import resources

    assert resources.Reading().fps is None
    assert "fps" not in resources.Reading(ram_used_gib=1.0, ram_total_gib=8.0).text()

    line = resources.Reading(ram_used_gib=1.0, ram_total_gib=8.0, fps=59.6).text()
    assert line.endswith("60 fps"), line
    # Last on the line, because the three above answer "can I start a 7 GB
    # generation right now" and a frame rate is ambient beside that.
    assert line.index("RAM") < line.index("fps")


def test_the_closed_dock_strips_button_reads_the_missing_or_idle_branch():
    """The 2026-09-23 dock move replaced the bottom pane's collapsed row with
    a 44 dp strip holding one ✦ button, and the branch it draws from has to
    stay the same one the row used to read: ``familiar_state`` -- "missing"
    muted with an Install... tooltip pointing at Settings -> Models, "idle"
    live and opening the dock. Asserted on the source, the same reason
    ``test_the_bottom_pane_centres_on_the_face_it_actually_draws_with`` (this
    test's predecessor) gave: the branch needs a real font atlas and window to
    prove interactively, which is more than this claim is about.
    """
    import inspect

    from realmspinner.studio.panes import familiar_dock

    source = inspect.getsource(familiar_dock.draw)
    assert "familiar_state(ctx.svc.config)" in source
    assert "muted" in source and "Install" in source


def test_resources_imports_nothing_from_the_ui():
    """``status_bar.items``' rule: the sampling and the formatting are data."""
    import inspect

    from realmspinner.studio import resources

    source = inspect.getsource(resources)
    for banned in ("imgui", "moderngl", "pygame"):
        assert banned not in source, banned


def test_the_document_name_is_not_imguis_widget_id():
    """An Inker tab's label carries imgui's ``##`` id suffix, which is markup
    for the control that draws the tab -- and it was printed verbatim in the
    status bar, so the bottom of the window read "Untitled##pd1 *"."""
    from types import SimpleNamespace

    from realmspinner.studio import status_bar

    app_ctx = _ctx()
    app_ctx.state.mode = "inker"
    tab = SimpleNamespace(
        label="Untitled##pd1", dirty=True, view=SimpleNamespace(zoom=0.5)
    )
    app_ctx.state.inker = SimpleNamespace(active=tab, tool="brush", tabs=[tab])
    items = {item.key: item.text for item in status_bar.items(app_ctx)}
    assert items["document"] == "Untitled *"


def test_the_meter_is_ticked_where_every_frame_goes_through():
    """``_tick`` belongs to the run loop; ``frame`` is what every path calls.

    The screenshot harness calls ``frame`` directly, so a tick in ``_tick``
    left the meter blank in every shipped picture -- which is how this was
    found, and why the assertion is about the method rather than the number.
    """
    import inspect

    from realmspinner.studio import main

    # ``tick(`` rather than ``tick()``: the call carries the frame rate now,
    # and this assertion is about *where* the meter is ticked -- which is the
    # thing that went wrong -- not about its arguments.
    assert "self.resources.tick(" in inspect.getsource(main.App.frame)
    assert "resources.tick" not in inspect.getsource(main.App._tick)


# --- create-02/03: a character preview's landing (2026-09-11 audit) ---------
#
# ``tests/test_frame_thread_doors.py`` pins the parse/adopt split for every
# other model load in this tree (the selection-driven sync, the Review mesh
# load, the Troupe atlas...) but does not reach "character-preview" -- these
# two are that door's own proof, in the file that owns ``App``.


class _PreviewViewer:
    """Records which thread touched each half of a character preview's load,
    the same way ``tests/test_frame_thread_doors.py``'s ``_GL``/spy pair does
    for the other doors: a regression that moved the decode back onto the
    frame thread shows up here as the test thread's own name.
    """

    def __init__(self) -> None:
        self.load_model_threads: list[str] = []
        self.parse_model_threads: list[str] = []
        self.adopted: list[object] = []

    def load_model(self, path):
        import threading

        # The pre-fix path: both halves at once, whichever thread calls this.
        self.load_model_threads.append(threading.current_thread().name)

    def parse_model(self, path):
        import threading

        self.parse_model_threads.append(threading.current_thread().name)
        return {"parsed": str(path)}

    def adopt_model(self, model, path):
        self.adopted.append(path)


class _PreviewCtx:
    """A ``ctx.submit`` that runs the task on a real worker thread and joins
    -- ``tests/test_frame_thread_doors.py``'s ``_Threaded`` idiom, repeated
    here rather than imported so this file does not reach into one it does
    not own.
    """

    def __init__(self, selected="job1", mode="create", create_stage="reference"):
        self.state = SimpleNamespace(
            mode=mode,
            create=SimpleNamespace(stage=create_stage),
            selected=selected,
            preview={},
        )
        self.submitted: list[str] = []
        self.tags: list[object] = []
        self.result: object = None
        self.toasts: list[tuple] = []

    def submit(self, key, fn, *args, tag=None, **kwargs):
        import threading

        self.submitted.append(key)
        self.tags.append(tag)
        box: dict = {}

        def go() -> None:
            box["result"] = fn(*args, **kwargs)

        worker = threading.Thread(target=go, name="realmspinner-task-test")
        worker.start()
        worker.join()
        self.result = box["result"]
        return True

    def toast(self, *a, **k):
        self.toasts.append((a, k))


def _preview_app(ctx=None):
    from realmspinner.studio import main as main_mod

    app = main_mod.App.__new__(main_mod.App)
    app.viewer = _PreviewViewer()
    app.app_ctx = ctx or _PreviewCtx()
    return app


def test_the_character_preview_landing_does_not_decode_on_the_frame_thread():
    """create-02 (2026-09-11 audit): a finished "character-preview" build was
    shown with ``viewer.load_model`` -- a full glTF parse plus a PNG texture
    decode per slot -- called directly inside ``_on_task_done``, on the frame
    thread. That is exactly the T2-class stall the 2026-09-02 review already
    fixed for the reference-PNG and mesh-load paths, and this proves the same
    split now applies here: the parse runs on a task thread under its own
    key, and only the landing of *that* task uploads to the viewer.
    """
    from pathlib import Path

    from realmspinner.studio import main as main_mod

    app = _preview_app()

    app._on_task_done(SimpleNamespace(key="character-preview", result="/tmp/preview.glb"))

    assert app.viewer.load_model_threads == [], (
        "the combined blocking load must never run from this path"
    )
    assert app.app_ctx.submitted == [main_mod.CHARACTER_PREVIEW_LOAD_KEY]
    assert app.viewer.parse_model_threads == ["realmspinner-task-test"], (
        "the parse must run off the frame thread"
    )
    assert app.viewer.adopted == [], "nothing is uploaded until that task lands"

    # Landing the parse is the other half of the split -- the frame-thread
    # GPU upload -- and it is what actually puts the preview on screen.
    app._on_task_done(
        SimpleNamespace(
            key=main_mod.CHARACTER_PREVIEW_LOAD_KEY,
            result=app.app_ctx.result,
            tag=app.app_ctx.tags[-1],
        )
    )
    assert app.viewer.adopted == [Path("/tmp/preview.glb")]


def test_a_character_preview_that_lands_after_the_user_navigates_away_does_not_hijack_the_viewport():  # noqa: E501
    """create-03 (2026-09-11 audit): landing a character-preview build used
    to adopt it unconditionally, with no token or selection/stage check --
    unlike ``_adopt_model``, which checks ``done.tag`` against
    ``viewer.pending`` first. A build that landed after the user picked a
    different asset, or left Create's Reference stage, silently replaced
    whatever the viewport was already showing.
    """
    from realmspinner.studio import main as main_mod

    ctx = _PreviewCtx(selected="job1")
    app = _preview_app(ctx)

    app._on_task_done(SimpleNamespace(key="character-preview", result="/tmp/preview.glb"))
    landed_tag = ctx.tags[-1]
    landed_result = ctx.result

    # The user picks a different asset before the parse lands.
    ctx.state.selected = "job2"
    app._on_task_done(
        SimpleNamespace(
            key=main_mod.CHARACTER_PREVIEW_LOAD_KEY, result=landed_result, tag=landed_tag
        )
    )

    assert app.viewer.adopted == [], "a stale build must not replace the newly selected asset"
    assert main_mod.CHARACTER_PIN not in ctx.state.preview

    # And the control: nothing about the selection moved, so the same landing
    # does adopt -- this is not a check that refuses everything.
    from pathlib import Path

    ctx2 = _PreviewCtx(selected="job1")
    app2 = _preview_app(ctx2)
    app2._on_task_done(SimpleNamespace(key="character-preview", result="/tmp/preview.glb"))
    app2._on_task_done(
        SimpleNamespace(
            key=main_mod.CHARACTER_PREVIEW_LOAD_KEY, result=ctx2.result, tag=ctx2.tags[-1]
        )
    )
    assert app2.viewer.adopted == [Path("/tmp/preview.glb")]
    assert ctx2.state.preview[main_mod.CHARACTER_PIN] == (str(Path("/tmp/preview.glb")), "job1")


# --- shell-09: F10 is "Everywhere", including while the Manual is open ------


def test_f10_toggles_the_frame_rate_readout_even_while_the_manual_is_open():
    """shell-09 (2026-09-11 audit): the Ctrl+/ sheet documents F10 in its
    "Everywhere" section beside Ctrl+K, Ctrl+/, F1 and Esc -- but
    ``App._shortcut``'s "the Manual owns it too" guard sat above the F10
    check, so every press while the Manual overlay was open did nothing, with
    no indication why.
    """
    import pygame

    from realmspinner.studio import main as main_mod

    app = main_mod.App.__new__(main_mod.App)
    state = SimpleNamespace(
        mode="create",
        mode_observed="create",
        previous_mode="",
        palette_open=False,
        manual=SimpleNamespace(open=True),
        show_fps=False,
    )
    app.app_ctx = SimpleNamespace(state=state)
    event = SimpleNamespace(type=pygame.KEYDOWN, key=pygame.K_F10, mod=0)

    app._shortcut(event)

    assert state.show_fps is True


# --- shell-14: a quit confirm should name an in-flight review sweep --------


def test_quit_summary_names_a_sweep_launch_or_delete_in_flight():
    """shell-14 (2026-09-11 audit): ``App._quit_summary`` named an in-flight
    model download, export or pack install so a quit confirm could warn about
    it, but checked no ``review-``-prefixed key at all -- so quitting
    mid-sweep-launch (twenty to forty job creations) or mid-sweep-deletion
    gave no warning at all, unlike the three lines beside it.
    """
    from realmspinner.studio import main as main_mod
    from realmspinner.studio.modes.review import mode as review_mode

    app = main_mod.App.__new__(main_mod.App)
    app.runtime = SimpleNamespace(current_job_id=None)
    app.app_ctx = SimpleNamespace(
        cache=SimpleNamespace(active=None),
        tasks=SimpleNamespace(busy_keys={review_mode.LAUNCH_KEY}),
    )

    summary = app._quit_summary()

    assert summary, "a sweep launch in flight must not pass through silently"
    assert "sweep" in summary.lower()

    # And a sweep deletion in flight is named too, not just a launch.
    app.app_ctx.tasks = SimpleNamespace(busy_keys={review_mode.DELETE_KEY})
    assert app._quit_summary()


# --- shell-03: a quit confirm should name a library export or update download


def test_quit_summary_warns_while_a_packwright_library_export_is_busy():
    """shell-03 (2026-09-13 audit): ``_quit_summary`` matched "-export:" (the
    per-mode in-editor export queue key, muse-05) but Packwright, Mason and
    Plotter each queue their *library* export as "<mode>-library:<name>" --
    a shape neither "-export:" nor any prefix checked here matches. Quitting
    mid-write could leave a library export whose sidecar never landed,
    unopenable in its own editor afterwards.
    """
    from realmspinner.studio import main as main_mod

    app = main_mod.App.__new__(main_mod.App)
    app.runtime = SimpleNamespace(current_job_id=None)
    app.app_ctx = SimpleNamespace(
        cache=SimpleNamespace(active=None),
        tasks=SimpleNamespace(busy_keys={"packwright-library:atlas-1"}),
    )

    summary = app._quit_summary()

    assert summary, "a library export in flight must not pass through silently"
    assert "library" in summary.lower()

    # Mason and Plotter follow the same convention and must be caught too.
    for key in ("mason-library:scene-1", "plotter-library:map-1"):
        app.app_ctx.tasks = SimpleNamespace(busy_keys={key})
        assert app._quit_summary(), key


def test_quit_summary_warns_while_an_update_download_is_in_flight():
    """shell-03 (2026-09-13 audit): an update-installer download
    (``app_ctx.UPDATE_DOWNLOAD_KEY``) has no resume marker, so a quit mid-
    download loses the whole thing -- exactly like a model download, which
    *was* checked here (it starts with "download:"). This key does not.
    """
    from realmspinner.studio import app_ctx as app_ctx_mod
    from realmspinner.studio import main as main_mod

    app = main_mod.App.__new__(main_mod.App)
    app.runtime = SimpleNamespace(current_job_id=None)
    app.app_ctx = SimpleNamespace(
        cache=SimpleNamespace(active=None),
        tasks=SimpleNamespace(busy_keys={app_ctx_mod.UPDATE_DOWNLOAD_KEY}),
    )

    summary = app._quit_summary()

    assert summary, "an update download in flight must not pass through silently"
    assert "update" in summary.lower()


# --- shell-01 (2026-09-20 audit): the export/save predicate missed every key
# that does not spell "export" or "save" immediately before the colon --------


def test_quit_summary_warns_while_a_clay_mesh_export_or_a_mason_scene_export_is_busy():
    """shell-01 (2026-09-20 audit): ``_quit_summary`` matched "-export:" as a
    literal substring, which only ever matched keys spelled exactly
    "<mode>-export:<uid>" -- Clay's own file export queues as
    "clay-exportfile:<uid>" and Mason's two export buttons queue as
    "mason-exportglb:<uid>"/"mason-exportobj:<uid>", none of which have a
    colon immediately after "export", so quitting mid-write through any of
    the three gave no warning at all.
    """
    from realmspinner.studio import main as main_mod

    app = main_mod.App.__new__(main_mod.App)
    app.runtime = SimpleNamespace(current_job_id=None)

    for key in ("clay-exportfile:tab-1", "mason-exportglb:tab-1", "mason-exportobj:tab-1"):
        app.app_ctx = SimpleNamespace(
            cache=SimpleNamespace(active=None),
            tasks=SimpleNamespace(busy_keys={key}),
        )
        assert app._quit_summary(), key


def test_quit_summary_warns_while_a_packwright_save_task_is_busy():
    """shell-01 (2026-09-20 audit): the same predicate checked a bare
    "save:" prefix, which no real key has -- every mode queues its save as
    "<mode>-save:<uid>" (and Packwright's "Save As" as
    "packwright-saveas:<uid>"), so quitting mid-save gave no warning in any
    mode, not just Packwright.
    """
    from realmspinner.studio import main as main_mod

    app = main_mod.App.__new__(main_mod.App)
    app.runtime = SimpleNamespace(current_job_id=None)

    for key in ("packwright-save:tab-1", "packwright-saveas:tab-1", "clay-save:tab-1"):
        app.app_ctx = SimpleNamespace(
            cache=SimpleNamespace(active=None),
            tasks=SimpleNamespace(busy_keys={key}),
        )
        assert app._quit_summary(), key
