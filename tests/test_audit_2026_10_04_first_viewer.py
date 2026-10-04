"""The 2026-10-04 audit's first-run viewer findings create-01, -02, -13 and -14.

No GL context: ``Viewer`` is driven unbound over stubs, the idiom
``tests/studio/test_viewer_poser.py`` already uses, with a real ``PoseEditor``
where the claim is about its undo depth and a real ``Camera`` where it is about
the redraw skip.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pygame

from realmspinner.studio import viewer_embed
from realmspinner.studio.viewer.camera import Camera
from realmspinner.studio.viewer.pose import PoseEditor
from realmspinner.studio.viewer_embed import Viewer


class _BoundEditor(PoseEditor):
    """A real editor's undo bookkeeping, bound without a model."""

    bound = True  # type: ignore[assignment]


class _Gizmo:
    """Hit on any ray, so a left press at once starts a drag."""

    hover = None

    def __init__(self) -> None:
        self.dragging = False

    def hit(self, origin, direction):
        return "x"

    def begin(self, axis, origin, direction):
        self.dragging = True
        return True

    def end_drag(self):
        self.dragging = False


def _posing_viewer() -> Viewer:
    viewer = Viewer.__new__(Viewer)
    viewer.pose_mode = True
    viewer.editor = _BoundEditor()
    viewer.editor.selected = "hips"
    viewer.gizmo = _Gizmo()
    viewer._active_gizmo = lambda: viewer.gizmo
    viewer._mods = lambda: (False, False, False)
    viewer._ray = lambda local: (np.zeros(3), np.array([0.0, 0.0, -1.0]))
    viewer.on_pose_dirty = None
    viewer._grab = None
    viewer._pose_step = None
    viewer._deselect_on_click = False
    viewer._rmb_at = None
    viewer._render_dirty = False
    viewer._last_mouse = (0.0, 0.0)
    viewer._rect = (0.0, 0.0, 100.0, 100.0)
    viewer.camera = Camera()
    return viewer


def _event(kind: int, button: int) -> pygame.event.Event:
    return pygame.event.Event(kind, button=button, pos=(10, 10))


# --- create-01: a second button mid gizmo drag --------------------------------


def test_a_middle_press_mid_gizmo_drag_does_not_leave_the_pose_step_open():
    viewer = _posing_viewer()
    assert viewer.handle_event(_event(pygame.MOUSEBUTTONDOWN, 1), True)
    assert viewer._grab == "gizmo" and viewer.editor._depth == 1

    # The middle button goes down while the left is still held, then both come
    # up. The middle press used to replace "gizmo" with "pan", so the left
    # release found nothing to close and the open ``record()`` never exited.
    viewer.handle_event(_event(pygame.MOUSEBUTTONDOWN, 2), True)
    viewer.handle_event(_event(pygame.MOUSEBUTTONUP, 2), True)
    viewer.handle_event(_event(pygame.MOUSEBUTTONUP, 1), True)

    assert viewer._grab is None
    assert viewer.editor._depth == 0, "pose undo is dead while a step is left open"
    assert viewer._pose_step is None


# --- create-02: a wheel notch ends the grab -----------------------------------


def test_a_wheel_notch_release_does_not_end_a_live_orbit():
    viewer = _posing_viewer()
    viewer.pose_mode = False
    viewer.handle_event(_event(pygame.MOUSEBUTTONDOWN, 1), True)
    assert viewer._grab == "orbit"

    # One notch of the wheel delivers a button 4 (or 5) press and release.
    viewer.handle_event(_event(pygame.MOUSEBUTTONDOWN, 4), True)
    viewer.handle_event(_event(pygame.MOUSEBUTTONUP, 4), True)
    viewer.handle_event(_event(pygame.MOUSEBUTTONUP, 5), True)

    assert viewer._grab == "orbit", "scrolling mid-orbit dropped the grab"
    viewer.handle_event(_event(pygame.MOUSEBUTTONUP, 1), True)
    assert viewer._grab is None


def test_a_wheel_notch_release_does_not_commit_a_live_gizmo_drag_early():
    viewer = _posing_viewer()
    viewer.handle_event(_event(pygame.MOUSEBUTTONDOWN, 1), True)
    assert viewer.editor._depth == 1

    viewer.handle_event(_event(pygame.MOUSEBUTTONUP, 5), True)

    assert viewer._grab == "gizmo" and viewer.editor._depth == 1
    viewer.handle_event(_event(pygame.MOUSEBUTTONUP, 1), True)
    assert viewer.editor._depth == 0


def test_a_middle_pan_still_ends_on_the_middle_release_alone():
    viewer = _posing_viewer()
    viewer.pose_mode = False
    viewer.handle_event(_event(pygame.MOUSEBUTTONDOWN, 2), True)
    assert viewer._grab == "pan"

    viewer.handle_event(_event(pygame.MOUSEBUTTONUP, 1), True)
    assert viewer._grab == "pan"
    viewer.handle_event(_event(pygame.MOUSEBUTTONUP, 2), True)
    assert viewer._grab is None


# --- create-13: a refused reference request ------------------------------------


def _refusing_ctx(inflight: Path, **extra):
    viewer = SimpleNamespace(
        pending=inflight,
        path=None,
        parse_model=lambda path: path,
        parse_reference=lambda path: path,
        has_model=False,
        clear=lambda: None,
    )
    ctx = SimpleNamespace(
        viewer=viewer, poser_viewer=viewer, submit=lambda *a, **k: False, **extra
    )
    return ctx, viewer


def test_a_refused_reference_request_leaves_the_inflight_loads_pending_alone(tmp_path):
    inflight = tmp_path / "first.png"
    ctx, viewer = _refusing_ctx(inflight)

    assert viewer_embed.request_reference(ctx, tmp_path / "second.png") is False

    assert viewer.pending == inflight, "the load already running lost its freshness check"


def test_a_refused_preview_load_leaves_the_inflight_loads_pending_alone(tmp_path):
    from realmspinner.studio.modes.poser import mode as poser_mode

    inflight = tmp_path / "earlier.glb"
    ctx, viewer = _refusing_ctx(inflight, state=SimpleNamespace(poser=None, preview={}))
    state = poser_mode.ensure(ctx)
    state.preview_path = tmp_path / "preview.glb"
    state.preview_template = state.template

    assert poser_mode.sync_preview(ctx, viewer) is False

    assert viewer.pending == inflight


def test_a_refused_asset_load_leaves_the_inflight_loads_pending_alone(tmp_path):
    from realmspinner.studio.modes.poser import mode as poser_mode

    inflight = tmp_path / "earlier.glb"
    ctx, viewer = _refusing_ctx(
        inflight,
        state=SimpleNamespace(poser=None, preview={}),
        job_dir=lambda job_id: tmp_path / job_id,
    )
    viewer.pose_mode = False
    viewer.pose_job_id = None
    state = poser_mode.ensure(ctx)
    state.job_id = "abcdef012345"

    assert poser_mode.sync_asset(ctx, viewer) is False

    assert viewer.pending == inflight


def test_a_refused_review_mesh_load_leaves_the_inflight_loads_pending_alone(tmp_path):
    from realmspinner.studio.modes.review.ui.workspace import ReviewPanes

    inflight = tmp_path / "earlier.glb"
    wanted = tmp_path / "model.glb"
    wanted.write_bytes(b"glTF")
    ctx, viewer = _refusing_ctx(inflight)
    review_mode = SimpleNamespace(
        model_path=lambda unit: wanted,
        ensure=lambda c: SimpleNamespace(
            mesh_wait=None, mesh_wait_checked=0.0
        ),
        mesh_retry_due=lambda state, path, now: False,
    )
    app = SimpleNamespace(viewer=viewer, app_ctx=ctx)

    ReviewPanes._review_load(app, {"status": "done"}, review_mode)

    assert viewer.pending == inflight


# --- create-14: an idle pose session ---------------------------------------------


def _rendering_viewer():
    viewer = _posing_viewer()
    viewer.draws = 0
    viewer.model = None
    viewer.wireframe = False
    viewer.compare_gpu = None
    viewer.placement = np.eye(4)
    viewer.gpu = None
    viewer.onion = []
    viewer._last_render_key = None
    viewer._render_dirty = True
    viewer.viewport = SimpleNamespace(texture=object(), size=(100, 100))

    def draw(*args, **kwargs):
        viewer.draws += 1

    viewer.renderer = SimpleNamespace(draw=draw)
    viewer._overlays = lambda height: []
    return viewer


def test_an_idle_pose_session_does_not_rerender_every_frame():
    viewer = _rendering_viewer()
    rect = (0.0, 0.0, 100.0, 100.0)

    for _ in range(6):
        viewer.render(rect, 0.016)

    assert viewer.draws == 1, "an idle pose session re-rendered the MSAA target every frame"


def test_a_pose_session_still_redraws_when_the_selection_the_gizmo_follows_changes():
    viewer = _rendering_viewer()
    rect = (0.0, 0.0, 100.0, 100.0)
    viewer.render(rect, 0.016)
    viewer.render(rect, 0.016)
    assert viewer.draws == 1

    # A pane (the hierarchy list, a click on a bone name) writes the selection
    # straight onto the editor; nothing else marks the viewer dirty.
    viewer.editor.selected = "spine"
    viewer.render(rect, 0.016)
    assert viewer.draws == 2

    viewer.editor.mode = "joints"
    viewer.render(rect, 0.016)
    assert viewer.draws == 3


def test_a_pose_session_redraws_when_poser_sets_the_onion_ghosts():
    viewer = _rendering_viewer()
    rect = (0.0, 0.0, 100.0, 100.0)
    viewer.render(rect, 0.016)
    viewer.render(rect, 0.016)
    assert viewer.draws == 1

    viewer.onion = [{"hips": [0.0, 0.0, 0.0, 1.0]}]
    viewer.render(rect, 0.016)

    assert viewer.draws == 2, "the ghosts are drawn from viewer.onion and nothing marked it dirty"
