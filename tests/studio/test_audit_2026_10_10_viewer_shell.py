"""Regression tests for the 2026-10-10 audit, findings create-06, create-32,
create-33, create-34, create-35 and create-39 (the viewer's shell: toolbar,
framing, the replaced-model memo, Poser's asset open, Inker's reference
nudge and the pose editor's undo).
"""

from __future__ import annotations

import importlib.util
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.studio import viewer_embed
from realmspinner.studio.viewer.camera import Camera
from realmspinner.studio.viewer.pose import PoseEditor

_HERE = Path(__file__).resolve().parent


def _ndc(cam: Camera, point) -> np.ndarray:
    clip = cam.projection() @ cam.view() @ np.append(np.asarray(point, dtype="f8"), 1.0)
    return clip[:3] / clip[3]


# --- create-06: the toolbar's zoom buttons -----------------------------------


def _toolbar_with_pressed(monkeypatch, pressed_tooltip_prefix: str, camera: Camera) -> None:
    from realmspinner.studio.panes import overlay

    def icon_button(icon, tooltip, **kwargs):
        return tooltip.startswith(pressed_tooltip_prefix)

    widgets = overlay.widgets
    monkeypatch.setattr(widgets, "icon_button", icon_button)
    monkeypatch.setattr(widgets, "same_line_or_wrap", lambda *_a, **_k: None)
    monkeypatch.setattr(widgets, "button_width", lambda *_a, **_k: 0.0)
    monkeypatch.setattr(overlay.manual_render, "help_button_inline", lambda *_a, **_k: None)
    monkeypatch.setattr(overlay, "offers_inker", lambda *_a: False)
    monkeypatch.setattr(overlay, "shows_tiled", lambda *_a: False)
    monkeypatch.setattr(overlay, "shows_front", lambda *_a: False)
    monkeypatch.setattr(overlay, "_has_content", lambda *_a: False)
    monkeypatch.setattr(overlay, "_texture_losses", lambda *_a: None)
    monkeypatch.setattr(overlay.controls, "button", lambda *_a, **_k: False)
    monkeypatch.setattr(overlay.widgets, "toggle", lambda _label, value, **_k: (False, value))
    viewer = SimpleNamespace(
        has_model=True,
        reference=None,
        model=object(),
        camera=camera,
        frame=lambda: None,
        set_wireframe=lambda _on: None,
        set_turntable=lambda _on: None,
    )
    ctx = SimpleNamespace(
        state=SimpleNamespace(comparing=None, wireframe=False, turntable=False),
        viewer=viewer,
        job=lambda: None,
        clear_viewport=None,
    )
    overlay.toolbar(ctx)


def test_the_zoom_in_button_brings_the_camera_closer(monkeypatch):
    # The 2026-10-10 audit: Camera.dolly(+n) moves the eye away, and the button
    # labelled "Zoom in" called dolly(1).
    camera = Camera(aspect=1.0)
    before = camera._goal_distance
    _toolbar_with_pressed(monkeypatch, "Zoom in", camera)
    assert camera._goal_distance < before


def test_the_zoom_out_button_takes_the_camera_further_away(monkeypatch):
    camera = Camera(aspect=1.0)
    before = camera._goal_distance
    _toolbar_with_pressed(monkeypatch, "Zoom out", camera)
    assert camera._goal_distance > before


# --- create-32: frame_bounds targets the box's own centre --------------------


def test_frame_bounds_puts_the_boxs_own_centre_in_view():
    # The 2026-10-10 audit: Camera.frame aimed at (0, height/2, 0), which is the
    # centre of a box standing on the ground at the origin and nowhere else; a
    # caller-supplied box (Poser's skeleton preview) is not placed.
    lo = np.array([2.0, 5.0, -3.0])
    hi = np.array([3.0, 7.0, -1.0])
    viewer = object.__new__(viewer_embed.Viewer)
    viewer.camera = Camera(aspect=1.0)
    viewer.renderer = SimpleNamespace(fit_grid=lambda *_a: None)
    viewer._render_dirty = False
    viewer.frame_bounds(lo, hi)
    centre = _ndc(viewer.camera, (lo + hi) / 2)
    assert abs(centre[0]) < 1e-6 and abs(centre[1]) < 1e-6


def test_a_grounded_box_is_still_framed_where_it_was():
    # The case the old target was right for must not move.
    lo = np.array([-0.5, 0.0, -0.5])
    hi = np.array([0.5, 2.0, 0.5])
    cam = Camera(aspect=1.0)
    cam.frame(lo, hi)
    assert cam.target == pytest.approx([0.0, 1.0, 0.0])


# --- create-33: the replaced model's bounds memo -----------------------------


def test_adopting_a_model_drops_the_previous_models_bounds_memo(monkeypatch):
    from realmspinner.studio.viewer import scene as scenelib

    class _Gpu:
        def __init__(self, ctx, model):
            pass

        def release(self):
            pass

    monkeypatch.setattr(scenelib, "GpuModel", _Gpu)
    monkeypatch.setattr(scenelib, "placement", lambda _m: np.eye(4))
    monkeypatch.setattr(viewer_embed.Viewer, "exit_pose_mode", lambda self: None)
    monkeypatch.setattr(viewer_embed.Viewer, "clear_reference", lambda self: None)
    monkeypatch.setattr(viewer_embed.Viewer, "frame", lambda self: 1.0)
    viewer = object.__new__(viewer_embed.Viewer)
    viewer.ctx = None
    viewer._strip = None
    viewer.gpu = None
    viewer.pending = None
    viewer._render_dirty = False
    old = object()
    viewer._bounds_memo = (old, 1, np.zeros(3), np.ones(3))
    viewer.adopt_model(object(), Path("next.glb"))
    memo = getattr(viewer, "_bounds_memo", None)
    assert memo is None or memo[0] is not old, "the replaced model must not stay referenced"


# --- create-34: Poser's asset open parses off the frame thread ---------------


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_poser_opens_an_asset_rig_without_parsing_on_the_frame_thread(tmp_path):
    # The 2026-10-10 audit: _bind_asset_now called viewer.load_model, a full glTF
    # parse and texture decode, on the frame thread; the precedent it cited
    # (pose_panel._enter) was moved onto a task by the 2026-10-04 audit.
    from realmspinner.studio.modes.poser import mode as poser_mode

    pm = _load(_HERE.parent / "modes" / "poser" / "test_poser_mode.py", "_pm_2026_10_10_shell")
    ctx = pm._ThreadedCtx(tmp_path)
    viewer = ctx.poser_viewer = pm.FakeViewer()
    state = poser_mode.ensure(ctx)
    job_id = "abcdef012345"
    state.job_id = job_id
    state.asset_rig = {}

    poser_mode._bind_asset_now(ctx, state, viewer, job_id)

    threads = getattr(ctx, "threads", [])
    assert threads == [pm.WORKER], "the parse must run on a task thread"
    assert threading.current_thread().name not in threads
    assert viewer.loaded == [], "load_model is the blocking spelling and is not the door"
    assert viewer.pose_mode is False, "nothing is adopted until the parse lands"

    key = poser_mode.ASSET_LOAD_KEY
    poser_mode.on_task_done(
        ctx, SimpleNamespace(key=key, result=ctx.results[key], tag=ctx.tags[key])
    )
    assert viewer.pose_mode is True and viewer.pose_job_id == job_id
    assert viewer.loaded == [ctx.job_dir(job_id) / "rig.glb"]


# --- create-35: a refused reference request is retried -----------------------


def test_a_refused_reference_request_is_retried_not_dropped(tmp_path):
    # The 2026-10-10 audit: _nudge_viewer cleared the viewer and asked for the
    # reference again, but a load already in flight made the runner refuse the
    # request, the in-flight result then failed the freshness check, and the
    # canvas stayed blank.
    from realmspinner.studio.modes.inker import mode as inker_mode

    path = tmp_path / "input.png"
    shown: list[Path] = []

    class _Viewer:
        def __init__(self) -> None:
            self.path: Path | None = path
            self.pending: Path | None = None

        def clear(self) -> None:
            self.path = None
            self.pending = None

        def parse_reference(self, p):
            return ("parsed", p)

        def load_reference(self, p) -> None:
            shown.append(Path(p))

    viewer = _Viewer()
    ctx = SimpleNamespace(viewer=viewer, submit=lambda *_a, **_k: False)
    tab = SimpleNamespace(path=path)

    inker_mode._nudge_viewer(ctx, tab)

    assert shown == [path], "a refused request must be retried, not left blank"
    assert viewer.path == path, "the reference is on screen, so sync has nothing to redo"


# --- create-39: undo while a gizmo drag is open ------------------------------


class _Node:
    def __init__(self, world):
        self.world = world


class _Model:
    def __init__(self, names):
        self.by_name = {n: i for i, n in enumerate(names)}
        self.rotations = {n: m3.quat_identity() for n in names}
        self.nodes = [_Node(m3.translation(m3.vec3(i, 0, 0))) for i, _ in enumerate(names)]
        self.skins: list[Any] = []

    def get_rotation(self, bone):
        return self.rotations.get(bone)

    def set_rotation(self, bone, quat):
        self.rotations[bone] = m3.quat_normalize(np.asarray(quat, dtype="f8"))
        return True

    def update_world(self):
        pass


def _quarter_turn():
    return m3.quat_from_axis_angle(m3.vec3(0, 0, 1), 0.5)


def test_undo_is_refused_while_a_gizmo_drag_is_open():
    # The 2026-10-10 audit: Ctrl+Z under an open record() restored the history
    # head while the drag's own "before" snapshot was already taken, so the
    # pushed edit's "before" contained the step just undone.
    editor = PoseEditor()
    editor.bind(_Model(["hip", "arm"]), ["hip", "arm"])
    editor.selected = "arm"
    with editor.record():
        editor.rotate_selected(_quarter_turn())

    with editor.record():
        editor.rotate_selected(_quarter_turn())
        mid_drag = editor.snapshot()
        assert editor.undo() is False, "undo must not run under an open step"
        assert editor.redo() is False
        assert _same_pose(editor.snapshot(), mid_drag)

    # The drag closed: the two quarter turns are two steps and undo walks them.
    assert editor.undo() is True
    one_turn = editor.model.get_rotation("arm")
    assert editor.undo() is True
    assert editor.model.get_rotation("arm") == pytest.approx(m3.quat_identity())
    assert not np.allclose(one_turn, m3.quat_identity())


def _same_pose(a: dict, b: dict) -> bool:
    return a["rotations"] == b["rotations"]
