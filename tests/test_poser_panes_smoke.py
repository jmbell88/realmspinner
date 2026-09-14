"""Poser's two panes, built for real, once -- the test_studio_smoke idiom.

Not a screenshot test: it asserts that a frame containing each pane can be
*built* in its empty, populated and mid-session states -- no missing begin/end
pair, no attribute that moved. The harness is test_studio_smoke's, module-scoped
here and restoring the previous imgui context on the way out, because two live
harnesses in one merged run would otherwise fight over the current-context
pointer.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from warlock.studio.app_ctx import Ctx
from warlock.studio.jobs_cache import JobsCache
from warlock.studio.settings import Settings
from warlock.studio.state import AppState
from warlock.studio.viewer import math3d as m3
from warlock.studio.viewer.camera import Camera
from warlock.studio.viewer.gltf import Model, Node
from warlock.studio.viewer.pose import PoseEditor


@pytest.fixture(scope="module")
def imgui_ctx(gl):
    """An imgui context with a real renderer, over the standalone GL context.

    The renderer is needed even though nothing is presented: imgui 1.92 hands
    its font atlas to the backend, and a context whose backend never claims it
    cannot finish a frame.
    """
    from imgui_bundle import imgui

    from warlock.studio import imgui_backend, theme, widgets

    prev_ctx = imgui.get_current_context()
    prev_screen = type(gl).__dict__.get("screen")
    # A standalone context has no default framebuffer; the renderer targets
    # ctx.screen, so give it one that exists.
    fbo = gl.simple_framebuffer((1600, 950))
    fbo.use()
    type(gl).screen = property(lambda _self: fbo)

    # Every collapsing section forced open: a frame of collapsed headings would
    # build without ever touching the code this test exists to exercise.
    prev_force = widgets.FORCE_SECTIONS_OPEN
    widgets.FORCE_SECTIONS_OPEN = True

    ctx = imgui.create_context()
    io = imgui.get_io()
    # See ``test_studio_smoke``'s fixture: a persisted collapsed flag in
    # ``imgui.ini`` survives the process and silently empties every window.
    io.set_ini_filename(None)
    io.display_size = (1600, 950)
    io.delta_time = 1 / 60
    io.fonts.add_font_default()
    theme.apply(imgui)
    renderer = imgui_backend.ImguiRenderer(gl)
    yield imgui, renderer
    renderer.shutdown()
    imgui.destroy_context(ctx)
    widgets.FORCE_SECTIONS_OPEN = prev_force
    if prev_screen is not None:
        type(gl).screen = prev_screen
    if prev_ctx is not None:
        imgui.set_current_context(prev_ctx)


@pytest.fixture
def app_ctx(gl, svc, tmp_path, imgui_ctx):
    from warlock.studio import textures
    from warlock.studio.runtime import Runtime
    from warlock.studio.tasks import TaskRunner
    from warlock.studio.viewer_embed import Viewer

    runtime = Runtime(svc.config)
    runtime.store = svc.store
    runtime.tasks = TaskRunner(workers=1)
    viewer = Viewer(gl)
    ctx = Ctx(
        svc=svc,
        runtime=runtime,
        state=AppState(),
        cache=JobsCache(svc),
        tasks=runtime.tasks,
        settings=Settings.load(tmp_path),
        viewer=viewer,
        textures=textures.ThumbnailCache(gl),
    )
    ctx.rig_default = "humanoid"
    yield ctx
    viewer.release()
    ctx.textures.release()
    runtime.tasks.shutdown(wait=False)


def _frame(imgui_ctx, build):
    """Run one complete imgui frame around ``build``."""
    imgui, renderer = imgui_ctx
    imgui.new_frame()
    imgui.set_next_window_size((1200, 900))
    imgui.begin("##host")
    build()
    imgui.end()
    imgui.render()
    renderer.render(imgui.get_draw_data())


class _FakeGpu:
    """Stands in for ``GpuScene``: only the one call poser_controls makes.

    Before the 2026-09-08 audit's poser-02, ``_PoserViewer`` had no ``.gpu``
    at all, so ``test_a_selected_joint_can_be_rotated_by_number`` could not
    see that ``_rotate_selected_to_euler`` (and ``_root``'s typed offset)
    wrote the pose without ever calling ``refresh_palettes`` -- a double that
    cannot see the defect is why that test passed on the unfixed code.
    """

    def __init__(self) -> None:
        self.refreshed = 0

    def refresh_palettes(self) -> None:
        self.refreshed += 1


class _PoserViewer:
    """The surface the panes touch, over a real editor on a meshless armature
    -- the shape the preview loads as, minus the file and the GL."""

    def __init__(self) -> None:
        nodes = [
            Node(name="rig", children=[1]),
            Node(name="hips", translation=m3.vec3(0.0, 0.53, 0.0), children=[2, 3, 4]),
            Node(name="spine", translation=m3.vec3(0.0, 0.07, 0.0)),
            Node(name="arm.L", translation=m3.vec3(-0.2, 0.1, 0.0)),
            Node(name="arm.R", translation=m3.vec3(0.2, 0.1, 0.0)),
        ]
        self.editor = PoseEditor()
        self.editor.bind(Model(nodes, roots=[0], meshes=[], skins=[]),
                         ["hips", "spine", "arm.L", "arm.R"])
        self.editor.mirror_pairs = [["arm.L", "arm.R"]]
        self.editor.root = "hips"
        self.pose_mode = True
        self.selected_bone = None
        # The front control reads and turns this one -- a real Camera rather
        # than a stub, so the damping the press has to see through is the
        # damping the app actually runs.
        self.camera = Camera()
        # poser-02: the bound-mesh skin palette, recomputed on every pose
        # change (viewer/scene.py) -- not on every draw call, so nothing else
        # refreshes it on a frame.
        self.gpu = _FakeGpu()

    # -- skeleton mode (P6, 2026-09-13) --------------------------------------
    #
    # ``_viewer_pose.PoseOps``'s own pass-throughs, minus the GPU refresh they
    # also do -- nothing here reads a skin palette.

    def enter_skeleton_mode(self, rig) -> None:
        self.editor.enter_skeleton_mode(rig)

    def exit_skeleton_mode(self) -> None:
        self.editor.exit_skeleton_mode()

    def skeleton_payload(self):
        return self.editor.skeleton_payload()

    def skel_add_child(self, parent):
        return self.editor.skel_add_child(parent)

    def skel_split(self, name):
        return self.editor.skel_split(name)

    def skel_remove_pivot(self, name) -> None:
        self.editor.skel_remove_pivot(name)

    def skel_remove_subtree(self, name):
        return self.editor.skel_remove_subtree(name)

    def skel_rename(self, old, new) -> None:
        self.editor.skel_rename(old, new)

    def skel_attach_limb(self, preset_key, parent, side, mirror):
        return self.editor.skel_attach_limb(preset_key, parent, side, mirror)

    def subtree_size(self, name) -> int:
        return self.editor.subtree_size(name)


def _skeleton_rig():
    return {
        "bones": [
            {"name": "hips", "parent": None, "head": [0.0, 0.0, 0.0], "tail": [0.0, 0.1, 0.0]},
            {"name": "spine", "parent": "hips", "head": [0.0, 0.1, 0.0], "tail": [0.0, 0.2, 0.0]},
            {"name": "arm.L", "parent": "hips", "head": [-0.1, 0.1, 0.0], "tail": [-0.2, 0.1, 0.0]},
            {"name": "arm.R", "parent": "hips", "head": [0.1, 0.1, 0.0], "tail": [0.2, 0.1, 0.0]},
        ],
        "root": "hips",
        "mirror_pairs": [["arm.L", "arm.R"]],
    }


def test_the_skeleton_pane_builds_in_both_states(app_ctx, imgui_ctx):
    """P6 (2026-09-13): the entry button, the editor with nothing and
    something selected, and a field-addressed refusal shown under a control
    -- each its own branch of ``panes/poser_skeleton.py``."""
    from warlock.studio import poser_mode
    from warlock.studio.panes import poser_controls

    app_ctx.rigging_available = True
    state = poser_mode.ensure(app_ctx)
    state.job_id = "0123456789ab"
    state.asset_label = "a ranger"
    state.asset_rig = _skeleton_rig()
    app_ctx.poser_viewer = _PoserViewer()

    # Not editing: the "Edit skeleton" entry point.
    _frame(imgui_ctx, lambda: poser_controls.draw(app_ctx))

    # Editing, nothing selected.
    poser_mode.enter_skeleton_edit(app_ctx)
    assert state.skeleton_editing is True
    _frame(imgui_ctx, lambda: poser_controls.draw(app_ctx))

    # Editing, a pivot selected -- the rename box and the structure buttons.
    app_ctx.poser_viewer.editor.selected = "hips"
    _frame(imgui_ctx, lambda: poser_controls.draw(app_ctx))

    # A refused Apply, addressed to a field the pane shows under Apply itself.
    state.skeleton_error = {"field": "bones", "message": "a skeleton may hold at most 64 bones"}
    _frame(imgui_ctx, lambda: poser_controls.draw(app_ctx))


def test_the_poser_panes_build_without_rigging(app_ctx, imgui_ctx):
    """The Blender-missing branch, which is the state a bare install opens in."""
    from warlock.studio.panes import poser_controls, poser_library

    app_ctx.rigging_available = False
    _frame(imgui_ctx, lambda: poser_library.draw(app_ctx))
    _frame(imgui_ctx, lambda: poser_controls.draw(app_ctx))


def test_the_controls_pane_builds_while_the_preview_loads(app_ctx, imgui_ctx):
    from warlock.studio.panes import poser_controls

    app_ctx.rigging_available = True
    assert app_ctx.poser_viewer is None
    _frame(imgui_ctx, lambda: poser_controls.draw(app_ctx))


def test_the_poser_panes_build_with_a_session_and_a_library(app_ctx, imgui_ctx):
    from warlock.studio import poser_mode
    from warlock.studio.panes import poser_controls, poser_library

    app_ctx.rigging_available = True
    state = poser_mode.ensure(app_ctx)
    state.poses = [
        {"id": "0123456789ab", "name": "Crouch", "bones": {}},
        {"id": "0123456789ac", "name": "Leap", "bones": {}},
    ]
    state.presets = [{"name": "idle", "bones": {}}]
    app_ctx.poser_viewer = _PoserViewer()
    _frame(imgui_ctx, lambda: poser_library.draw(app_ctx))
    _frame(imgui_ctx, lambda: poser_controls.draw(app_ctx))

    # And mid-edit: the unsaved banner, the highlighted row, the root-selected
    # checkbox and the nonzero offset line are each their own branch.
    viewer = app_ctx.poser_viewer
    viewer.editor.current = "0123456789ab"
    viewer.editor.dirty = True
    viewer.editor.selected = "hips"
    viewer.selected_bone = "hips"
    viewer.editor.root_translate = True
    viewer.editor.set_root_translation([0.1, 0.0, 0.2])
    _frame(imgui_ctx, lambda: poser_library.draw(app_ctx))
    _frame(imgui_ctx, lambda: poser_controls.draw(app_ctx))

    # And mid-skeleton-edit (the 2026-09-14 audit, poser-03): New pose and
    # every Apply button must still draw -- greyed, not hidden -- rather than
    # crash the pane now that they read ``state.skeleton_editing``.
    state.job_id = "0123456789ab"
    state.asset_rig = {
        "bones": [{"name": "hips", "parent": None, "head": [0, 0, 0], "tail": [0, 1, 0]}],
        "root": "hips",
        "mirror_pairs": [],
    }
    state.asset_poses = [{"id": "0123456789ad", "name": "Sit", "bones": {}}]
    viewer.editor.dirty = False  # enter_skeleton_edit refuses over an unsaved pose
    poser_mode.enter_skeleton_edit(app_ctx)
    assert state.skeleton_editing is True
    _frame(imgui_ctx, lambda: poser_library.draw(app_ctx))


def test_the_front_section_draws_bound_and_unbound(app_ctx, imgui_ctx):
    """The section only exists in an asset session, and both of its states
    have to build: no front chosen (the readout says so and two of the three
    buttons are greyed with a reason) and a front chosen."""
    from warlock.studio import poser_mode
    from warlock.studio.panes import poser_controls

    app_ctx.rigging_available = True
    state = poser_mode.ensure(app_ctx)
    app_ctx.poser_viewer = _PoserViewer()

    # Template session: no asset, so no front to set.
    _frame(imgui_ctx, lambda: poser_controls.draw(app_ctx))

    state.job_id = "0123456789ab"
    state.asset_label = "a ranger"
    _frame(imgui_ctx, lambda: poser_controls.draw(app_ctx))
    state.asset_front_yaw = 137.5
    _frame(imgui_ctx, lambda: poser_controls.draw(app_ctx))


def test_setting_the_front_records_the_goal_not_the_damped_angle(app_ctx, monkeypatch):
    """The camera chases a goal, so ``theta`` is wherever the glide had got to
    when the button was pressed. Recording that instead of the goal would
    store an angle the user never chose and never saw settle -- and at
    ``DAMPING`` 0.05 it can be tens of degrees short. ``clay_state.read_from``
    made this call already; this is the test that it was copied."""
    from warlock.studio import poser_mode

    app_ctx.rigging_available = True
    state = poser_mode.ensure(app_ctx)
    state.job_id = "0123456789ab"
    app_ctx.poser_viewer = _PoserViewer()
    camera = app_ctx.poser_viewer.camera
    camera.theta = math.radians(10.0)
    camera._goal_theta = math.radians(137.0)

    sent: list[float] = []
    monkeypatch.setattr(
        app_ctx, "submit", lambda key, fn, *a, **k: sent.append(a[-1]) or True
    )
    poser_mode.set_front(app_ctx)
    assert sent and abs(sent[0] - 137.0) < 1e-6


def test_looking_at_the_front_turns_the_camera_and_keeps_the_framing(app_ctx):
    """An angle change that also reframed would throw away the part of the
    model the user had lined up -- ``Camera.look_along``'s own rule, which is
    why this is not a call to it."""
    from warlock.studio import poser_mode

    app_ctx.rigging_available = True
    state = poser_mode.ensure(app_ctx)
    state.job_id = "0123456789ab"
    state.asset_front_yaw = 137.0
    app_ctx.poser_viewer = _PoserViewer()
    camera = app_ctx.poser_viewer.camera
    before = (camera.phi, camera.distance, tuple(camera.target))

    poser_mode.look_at_front(app_ctx)
    assert abs(camera._goal_theta - math.radians(137.0)) < 1e-6
    assert (camera.phi, camera.distance, tuple(camera.target)) == before


def test_a_landed_front_write_dirties_the_jobs_cache(app_ctx, monkeypatch):
    """Poser keeps its own copy of the front, but it is not the only reader.

    ``panes/overlay.py``'s copy of this control labels itself from the *row*
    (``job["params"]["front_yaw"]``) out of ``ctx.cache``, and so does the
    Send to Troupe helper line. Updating only ``PoserState`` left a press from
    the viewport toolbar writing the front and then going on drawing "Set
    front" until something unrelated dirtied the cache -- which reads as the
    button having done nothing at all. Invalidated for *any* job, not just the
    bound one, because the toolbar's asset is routinely not Poser's.
    """
    from types import SimpleNamespace

    from warlock.studio import poser_mode

    state = poser_mode.ensure(app_ctx)
    state.job_id = ""  # no asset bound here: the toolbar's press, not Poser's.
    calls: list[int] = []
    monkeypatch.setattr(app_ctx.cache, "invalidate", lambda: calls.append(1))

    done = SimpleNamespace(
        key=f"{poser_mode.FRONT_KEY_PREFIX}0123456789ab",
        result={"id": "0123456789ab", "front_yaw": 137.0},
    )
    poser_mode.on_task_done(app_ctx, done)
    assert calls, "a landed front write left every row-reading pane stale"


def test_drawing_the_library_pane_pumps_the_refresh_flag(app_ctx, imgui_ctx):
    """The per-frame half of the refresh idiom is wired through this pane's
    draw, so a drawn frame with the flag up must submit the list."""
    from warlock.studio import poser_mode
    from warlock.studio.panes import poser_library

    app_ctx.rigging_available = True
    state = poser_mode.ensure(app_ctx)
    state.refresh_dirty = True
    _frame(imgui_ctx, lambda: poser_library.draw(app_ctx))
    assert state.refresh_dirty is False, "cleared because the submit was accepted"
    assert state.loading is True


# --- the clip editor ----------------------------------------------------------


def _library() -> dict:
    """A clip library shaped exactly as ``service.clips.library`` returns one."""
    return {
        "template": "humanoid",
        "space": "delta",
        "edited": False,
        "poses": [
            {"name": "contact A", "bones": {"hips": [0.0, 0.0, 0.0, 1.0]}},
            {"name": "passing", "bones": {"spine": [0.0, 0.0, 0.0, 1.0]}},
            {"name": "contact B", "bones": {"arm.L": [0.0, 0.0, 0.0, 1.0]}},
        ],
        "clips": [
            {
                "name": "walk",
                "keys": ["contact A", "passing", "contact B"],
                "segments": [2, 2, 2],
                "closed": True,
                "easing": "linear",
                "space": "delta",
                "duration_ms": 100,
            }
        ],
    }


def test_update_key_reason_checks_posing_before_the_stale_frame():
    """poser-07 (the 2026-09-07 audit): while the preview is still loading,
    ``state.frame`` may still hold whatever in-between value it had before the
    switch -- checking it first named that stale frame as the reason instead
    of the real one, that posing had not started yet."""
    from warlock.studio.panes.poser_clips import _update_key_reason

    assert _update_key_reason(False, 3) == "The skeleton preview is still loading."
    assert (
        _update_key_reason(True, 3)
        == "That is an in-between frame, not a key. Pick a key first."
    )


def test_update_key_reason_names_a_build_failure_not_still_loading():
    """poser-07 (the 2026-09-11 audit): both clip-editor buttons that grey on
    "not posing" always said "The skeleton preview is still loading.", even
    when the true cause was that the preview build failed (state.error) or
    the bound asset's rig failed to load (state.asset_error) rather than
    being in progress. state.clips refreshes independently of either, so a
    Blender build failure with a perfectly good clip library on screen used
    to leave both buttons claiming the preview was "still loading"
    indefinitely."""
    from warlock.studio.panes.poser_clips import _new_key_reason, _update_key_reason

    assert (
        _update_key_reason(False, 3, error="Could not build the pose preview.")
        == "Could not build the pose preview."
    )
    assert (
        _update_key_reason(False, 3, asset_error="Could not open the rig.")
        == "Could not open the rig."
    )
    # A build failure takes priority over an asset-load failure when somehow
    # both are set -- the preview build is what state.error names, and it is
    # what actually gates posing here.
    assert (
        _update_key_reason(False, 3, error="build broke", asset_error="rig broke")
        == "build broke"
    )
    # Still loading, absent either failure -- the existing, still-correct case.
    assert _update_key_reason(False, 3) == "The skeleton preview is still loading."

    # "New key from pose..." gets the identical treatment.
    assert (
        _new_key_reason(False, error="Could not build the pose preview.")
        == "Could not build the pose preview."
    )
    assert (
        _new_key_reason(False, asset_error="That GLB carries no skeleton.")
        == "That GLB carries no skeleton."
    )
    assert _new_key_reason(False) == "The skeleton preview is still loading."


def test_a_provisional_clip_shows_its_badge(app_ctx, imgui_ctx):
    """The picker's badge text, pinned as a pure lookup so it is testable with
    no imgui frame (``_update_key_reason``'s own reason for being a function),
    then proven not to break the pane it is actually drawn into."""
    from warlock.studio import poser_mode
    from warlock.studio.panes import poser_clips
    from warlock.studio.panes.poser_clips import _provisional_note

    assert _provisional_note({"provisional": True}) == "provisional"
    assert _provisional_note({"provisional": False}) == ""
    assert _provisional_note({}) == ""
    assert _provisional_note(None) == ""

    app_ctx.rigging_available = True
    app_ctx.poser_viewer = _PoserViewer()
    state = poser_mode.ensure(app_ctx)
    state.clips_dirty_flag = False
    library = _library()
    library["clips"][0]["provisional"] = True
    poser_mode.adopt_clips(app_ctx, library)
    assert state.open_clip()["provisional"] is True
    _frame(imgui_ctx, lambda: poser_clips.draw(app_ctx))


def test_the_import_report_is_hidden_while_a_skeleton_edit_is_open(app_ctx, imgui_ctx, monkeypatch):
    """P6 (2026-09-13): master hides the whole Clips section during a skeleton
    edit; this branch keeps "Import clip..." drawn through it (see
    ``poser_clips.draw``'s comment) but the report of a *finished* import is
    exactly the section master hides, so it must not be the one piece of it
    left on screen. ``poser_clips._import_report`` only ever draws through
    imgui, so the assertion is on whether ``_import_button`` calls it at all
    rather than on anything the renderer produced."""
    from warlock.studio import poser_mode
    from warlock.studio.panes import poser_clips

    app_ctx.rigging_available = True
    app_ctx.poser_viewer = _PoserViewer()
    state = poser_mode.ensure(app_ctx)
    state.clips_dirty_flag = False
    poser_mode.adopt_clips(app_ctx, _library())
    state.clip_import_reports = [
        {"map": "mixamo", "loop": {}, "ignored": [], "left_at_rest": [], "frames": 10, "keys": 3}
    ]
    # ``enter_skeleton_edit``'s own requirements (see its docstring): a real
    # asset open with a readable rig and the viewer already in pose mode.
    state.job_id = "0123456789ab"
    state.asset_rig = _skeleton_rig()

    calls: list[Any] = []
    monkeypatch.setattr(
        poser_clips, "_import_report", lambda ctx, state: calls.append(state)
    )

    _frame(imgui_ctx, lambda: poser_clips.draw(app_ctx))
    assert calls, "not editing the skeleton: the report is drawn"

    calls.clear()
    poser_mode.enter_skeleton_edit(app_ctx)
    assert state.skeleton_editing is True
    _frame(imgui_ctx, lambda: poser_clips.draw(app_ctx))
    assert calls == [], "editing the skeleton: the report must not be drawn"


def test_the_clip_pane_builds_without_rigging(app_ctx, imgui_ctx):
    from warlock.studio.panes import poser_clips

    app_ctx.rigging_available = False
    _frame(imgui_ctx, lambda: poser_clips.draw(app_ctx))


def test_the_clip_pane_builds_for_a_skeleton_with_no_clips(app_ctx, imgui_ctx):
    """The state every template but the humanoid opens in, and the one a
    collapsed heading would hide."""
    from warlock.studio import poser_mode
    from warlock.studio.panes import poser_clips

    app_ctx.rigging_available = True
    state = poser_mode.ensure(app_ctx)
    state.clips = {"template": state.template, "clips": [], "poses": [], "edited": False}
    state.clips_dirty_flag = False
    _frame(imgui_ctx, lambda: poser_clips.draw(app_ctx))


def test_the_clip_pane_builds_over_a_clip_and_a_session(app_ctx, imgui_ctx):
    from warlock.studio import poser_mode
    from warlock.studio.panes import poser_clips

    app_ctx.rigging_available = True
    app_ctx.poser_viewer = _PoserViewer()
    state = poser_mode.ensure(app_ctx)
    state.clips_dirty_flag = False
    poser_mode.adopt_clips(app_ctx, _library())
    assert state.clip == "walk"
    _frame(imgui_ctx, lambda: poser_clips.draw(app_ctx))


def test_the_clip_pane_builds_while_scrubbing_and_while_unsaved(app_ctx, imgui_ctx):
    """Three branches that only exist in the middle of a session: the
    in-between banner, the Back-to-key button and the unsaved marker."""
    from warlock.studio import poser_mode
    from warlock.studio.panes import poser_clips

    app_ctx.rigging_available = True
    app_ctx.poser_viewer = _PoserViewer()
    state = poser_mode.ensure(app_ctx)
    state.clips_dirty_flag = False
    poser_mode.adopt_clips(app_ctx, _library())
    poser_mode.scrub(app_ctx, 3)
    assert state.frame == 3
    _frame(imgui_ctx, lambda: poser_clips.draw(app_ctx))

    state.clips_unsaved = True
    state.clips["edited"] = True
    _frame(imgui_ctx, lambda: poser_clips.draw(app_ctx))


def test_the_clip_pane_builds_when_the_clip_will_not_expand(app_ctx, imgui_ctx):
    """Mid-edit inconsistency is ordinary -- the segments briefly do not match
    the keys -- and it must draw a reason rather than raise into a frame."""
    from warlock.studio import poser_mode
    from warlock.studio.panes import poser_clips

    app_ctx.rigging_available = True
    app_ctx.poser_viewer = _PoserViewer()
    state = poser_mode.ensure(app_ctx)
    state.clips_dirty_flag = False
    poser_mode.adopt_clips(app_ctx, _library())
    state.open_clip()["segments"] = [2]
    poser_mode.rebuild_frames(app_ctx)
    assert state.clips_error and not state.frames
    _frame(imgui_ctx, lambda: poser_clips.draw(app_ctx))


def test_drawing_the_clip_pane_pumps_its_own_refresh_flag(app_ctx, imgui_ctx):
    from warlock.studio import poser_mode
    from warlock.studio.panes import poser_clips

    app_ctx.rigging_available = True
    state = poser_mode.ensure(app_ctx)
    state.clips_dirty_flag = True
    _frame(imgui_ctx, lambda: poser_clips.draw(app_ctx))
    assert state.clips_dirty_flag is False
    assert state.clips_loading is True


def test_an_unsaved_editor_is_never_reloaded_underneath_the_user(app_ctx, imgui_ctx):
    """A background refresh landing on unsaved keys would discard them without
    anyone asking, so the pump refuses while there is work in the editor."""
    from warlock.studio import poser_mode
    from warlock.studio.panes import poser_clips

    app_ctx.rigging_available = True
    state = poser_mode.ensure(app_ctx)
    poser_mode.adopt_clips(app_ctx, _library())
    state.clips_unsaved = True
    state.clips_dirty_flag = True
    _frame(imgui_ctx, lambda: poser_clips.draw(app_ctx))
    assert state.clips_loading is False
    assert state.clips_dirty_flag is True, "still wanted, just not now"


def test_a_failed_skeleton_build_offers_retry(app_ctx, imgui_ctx, gl, monkeypatch):
    """A build that has already failed once must not strand the user on a
    dead-end overlay -- the "Try again" button has to exist and it has to
    reuse ``request_preview`` rather than a second copy of its logic."""
    from warlock.studio import poser_mode
    from warlock.studio.panes import overlay
    from warlock.studio.poser_viewport import PoserViewport

    app_ctx.rigging_available = True
    state = poser_mode.ensure(app_ctx)
    state.template = "humanoid"
    state.building = False
    state.error = "Could not build the pose preview."

    captured: dict[str, Any] = {}
    real_centred_empty = overlay.centred_empty

    def spy(icon, title, hint, *, action=None):
        captured["action"] = action
        return real_centred_empty(icon, title, hint, action=action)

    monkeypatch.setattr(overlay, "centred_empty", spy)

    class _App(PoserViewport):
        def __init__(self, gl_ctx, ctx):
            self.ctx = gl_ctx
            self.app_ctx = ctx
            self.poser_viewer = None
            self._poser_hovered = False

    app = _App(gl, app_ctx)
    _frame(imgui_ctx, lambda: app._poser_viewport(app_ctx))

    assert "action" in captured, "the failure overlay never drew"
    action = captured["action"]
    assert action is not None, "a failed build must offer a retry action"
    label, on_click = action
    assert label == "Try again"

    # The button must call the same entry point every other preview request
    # goes through, not a duplicate of its guts: reuse shows up as
    # ``request_preview``'s own side effects -- the error cleared and the
    # building flag set -- immediately after the button fires.
    on_click()
    assert state.error == ""
    assert state.building is True


def test_the_joint_menu_switches_to_skeleton_items_in_skeleton_mode(app_ctx, imgui_ctx, gl):
    """P6 (2026-09-13): the right-click menu over a skeleton draft offers Add
    child/Split/Delete rather than the pose menu's rotate/reset items, which
    have nothing to act on while a draft has no pose at all."""
    from warlock.studio.poser_viewport import PoserViewport

    class _App(PoserViewport):
        def __init__(self, gl_ctx, ctx):
            self.ctx = gl_ctx
            self.app_ctx = ctx
            self.poser_viewer = None
            self._poser_hovered = False

    app_ctx.rigging_available = True
    viewer = _PoserViewer()
    viewer.editor.enter_skeleton_mode(_skeleton_rig())
    viewer.menu_request = (0.0, 0.0)
    app = _App(gl, app_ctx)

    # No pivot selected.
    _frame(imgui_ctx, lambda: app._poser_menu(app_ctx, viewer))

    # A pivot selected: the same real frame, opened again.
    viewer.editor.selected = "hips"
    viewer.menu_request = (0.0, 0.0)
    _frame(imgui_ctx, lambda: app._poser_menu(app_ctx, viewer))


# --- W1.7: numeric joint editing, the rest marker, the pending key ----------


def test_a_selected_joint_can_be_rotated_by_number(app_ctx, imgui_ctx):
    """The typed-degrees path must reach the same node the gizmo drags --
    proven by going through the same undo step, not by a matching quaternion,
    since a duplicate write path could match the number and still bypass
    ``rotate_selected``."""
    from warlock.studio.panes import poser_controls

    app_ctx.rigging_available = True
    app_ctx.poser_viewer = _PoserViewer()
    viewer = app_ctx.poser_viewer
    viewer.editor.selected = "hips"
    viewer.selected_bone = "hips"

    before = viewer.editor.model.get_rotation("hips").copy()
    poser_controls._rotate_selected_to_euler(viewer, [90.0, 0.0, 0.0])
    after = viewer.editor.model.get_rotation("hips")
    assert not np.allclose(before, after)
    degrees = poser_controls._quat_to_euler_degrees(after)
    assert degrees[0] == pytest.approx(90.0, abs=1e-3)
    assert degrees[1] == pytest.approx(0.0, abs=1e-3)
    assert degrees[2] == pytest.approx(0.0, abs=1e-3)
    assert viewer.editor.dirty is True

    # Same write path as the gizmo drag: the edit is one undo step, and
    # undoing it puts the bone back exactly where it started.
    assert viewer.editor.undo() is True
    assert np.allclose(viewer.editor.model.get_rotation("hips"), before, atol=1e-9)

    # And the pane builds over the edited joint without raising.
    _frame(imgui_ctx, lambda: poser_controls.draw(app_ctx))


def test_typing_a_joint_rotation_or_root_offset_refreshes_the_bound_meshs_skin_palette(
    app_ctx, imgui_ctx
):
    """poser-02 (the 2026-09-08 audit): every other mutating door in Poser --
    Reset joint, Reset all, Mirror, the gizmo drag -- refreshes
    ``GpuScene.refresh_palettes`` after it writes the pose; the typed Rotate
    X/Y/Z and Offset X/Y/Z fields wrote the correct data but skipped it,
    leaving a bound mesh's skin visibly frozen at the old pose."""
    from warlock.studio.panes import poser_controls

    app_ctx.rigging_available = True
    app_ctx.poser_viewer = _PoserViewer()
    viewer = app_ctx.poser_viewer
    viewer.editor.selected = "hips"
    viewer.selected_bone = "hips"

    assert viewer.gpu.refreshed == 0
    poser_controls._rotate_selected_to_euler(viewer, [90.0, 0.0, 0.0])
    assert viewer.gpu.refreshed == 1, "a typed joint rotation must refresh the skin palette"

    poser_controls._set_root_offset(viewer, [0.1, 0.0, 0.2])
    assert viewer.gpu.refreshed == 2, "a typed root offset must refresh the skin palette too"


def test_joints_changed_from_rest_are_marked(app_ctx, imgui_ctx):
    from warlock.studio.panes import poser_controls
    from warlock.studio.viewer import math3d as m3

    app_ctx.rigging_available = True
    app_ctx.poser_viewer = _PoserViewer()
    viewer = app_ctx.poser_viewer
    viewer.editor.selected = "hips"
    viewer.selected_bone = "hips"

    assert poser_controls._changed_from_rest(viewer, "hips") is False
    viewer.editor.rotate_selected(m3.quat_from_axis_angle((0.0, 1.0, 0.0), 0.6))
    assert poser_controls._changed_from_rest(viewer, "hips") is True

    # Reset puts it back to rest, and the marker must follow.
    viewer.editor.reset_bone("hips")
    assert poser_controls._changed_from_rest(viewer, "hips") is False

    _frame(imgui_ctx, lambda: poser_controls.draw(app_ctx))


def test_poser_reset_all_tooltip_does_not_claim_no_undo():
    """The 2026-09-08 audit (docs-06): poser_controls.py's "Reset all" button
    still carried the tooltip "There is no undo, so this asks first." even
    though ``Viewer.reset_all`` has been ``@_undoable`` since Ctrl+Z was wired
    into the pose editor -- the sibling asset-pose panel (pose_panel.py) had
    the same false claim removed already, and this copy was left behind."""
    import inspect

    from warlock.studio.panes import poser_controls

    source = inspect.getsource(poser_controls._joint)
    assert 'tooltip="Put every joint back to rest."' in source
    assert (
        'tooltip="Put every joint back to rest. There is no undo, so this asks first."'
        not in source
    )


def test_update_key_shows_pending_when_the_pose_drifted(app_ctx, imgui_ctx):
    """``_key_pending`` is what draws the accent dot beside "Update key from
    pose" -- true only once a key is loaded and the live pose has moved off
    it, and false again while scrubbing an in-between frame."""
    from warlock.studio import poser_mode
    from warlock.studio.panes import poser_clips
    from warlock.studio.viewer import math3d as m3

    app_ctx.rigging_available = True
    app_ctx.poser_viewer = _PoserViewer()
    state = poser_mode.ensure(app_ctx)
    state.clips_dirty_flag = False
    poser_mode.adopt_clips(app_ctx, _library())
    viewer = app_ctx.poser_viewer

    assert poser_clips._key_pending(viewer, state.frame) is False, (
        "a freshly adopted clip has nothing drifted yet"
    )

    viewer.editor.selected = "hips"
    viewer.editor.rotate_selected(m3.quat_from_axis_angle((0.0, 1.0, 0.0), 0.4))
    assert poser_clips._key_pending(viewer, state.frame) is True

    # Scrubbing shows an in-between frame, which has nowhere to store an
    # edit, so pending must not claim one even though ``dirty`` is still set.
    assert poser_clips._key_pending(viewer, 2) is False

    _frame(imgui_ctx, lambda: poser_clips.draw(app_ctx))


def test_revert_clips_reason_names_still_saving_when_a_save_is_in_flight():
    """poser-06 (the 2026-09-08 audit): "Revert to shipped clips" always said
    "These are already the clips the build ships." when disabled, even while
    the real reason was a save mid-flight -- unlike "Save clips" two lines
    above it, which already branches on ``busy``. Asserted with no imgui
    frame, the ``inker_mode._no_document_reason``/``clay_ops.reason_for``
    pattern this repository already uses for a greyed control's reason."""
    from types import SimpleNamespace as NS

    from warlock.studio.panes import poser_clips

    unsaved = NS(clips_unsaved=True, clips={})
    edited = NS(clips_unsaved=False, clips={"edited": True})
    clean = NS(clips_unsaved=False, clips={})

    # Busy wins: there *is* an unsaved change, and it is mid-write, not sitting
    # there unwritten the way the fixed string implied.
    assert poser_clips._revert_clips_reason(unsaved, busy=True) == "Still saving."
    assert poser_clips._revert_clips_reason(edited, busy=True) == "Still saving."
    # Nothing to revert, and no save running: the original reason still holds.
    assert (
        poser_clips._revert_clips_reason(clean, busy=False)
        == "These are already the clips the build ships."
    )
    # Something to revert and nothing running: the button is live, no reason
    # needed.
    assert poser_clips._revert_clips_reason(unsaved, busy=False) == ""
