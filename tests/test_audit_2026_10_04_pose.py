"""The 2026-10-04 audit's Pose-column findings, create-31 and create-50 -- one
regression each, named as the audit row names it.

create-31: a saved pose applied from the Pose column (``pose_panel._apply_saved_pose``)
or the Poser's asset list (``poser_mode.apply_asset_pose``) ran ``reset_all`` as one
undo step and the folded ``set_pose`` + ``set_root_translation`` as a second, so the
first Ctrl+Z landed on the rest pose instead of the pose the user had before pressing
Apply.

create-50: "Edit pose" and "Apply" called ``viewer.load_model`` -- the glTF parse and
the texture decode -- synchronously on the frame thread.
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
from realmspinner.studio._viewer_pose import PoseOps
from realmspinner.studio.modes.poser import mode as poser_mode
from realmspinner.studio.panes import pose_panel
from realmspinner.studio.viewer.pose import PoseEditor

WORKER = "realmspinner-task-test"
_HERE = Path(__file__).resolve().parent


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- create-31 ---------------------------------------------------------------


class _Node:
    def __init__(self, i: int) -> None:
        self.world = m3.translation(m3.vec3(i, 0, 0))
        self.translation = m3.vec3(0, 0, 0)


class _Model:
    def __init__(self, names: list[str]) -> None:
        self.by_name = {n: i for i, n in enumerate(names)}
        self.rotations = {n: m3.quat_identity() for n in names}
        self.nodes = [_Node(i) for i, _ in enumerate(names)]
        self.rest_translations = [m3.vec3(0, 0, 0) for _ in names]
        self.skins: list[Any] = []

    def get_rotation(self, bone):
        return self.rotations.get(bone)

    def set_rotation(self, bone, quat):
        if bone not in self.rotations:
            return False
        self.rotations[bone] = m3.quat_normalize(np.asarray(quat, dtype="f8"))
        return True

    def update_world(self) -> None:
        pass


class _RealOpsViewer(PoseOps):
    """The smallest object carrying what ``PoseOps`` reads, over a real editor."""

    def __init__(self) -> None:
        self.editor = PoseEditor()
        self.gpu = None
        self.on_pose_dirty = None
        self._render_dirty = False
        self.pose_mode = True
        self.pose_job_id = "job"
        self.editor.bind(_Model(["hip"]), ["hip"])
        self.editor.root = "hip"


def _pose_of(viewer: Any) -> list[float]:
    return [round(v, 4) for v in viewer.editor.pose()["hip"]]


def _posed_viewer() -> tuple[_RealOpsViewer, list[float]]:
    """A viewer holding a hand-made, saved-clean pose: -> (viewer, that pose)."""
    viewer = _RealOpsViewer()
    with viewer.editor.record():
        viewer.editor.apply({"hip": list(m3.quat_from_axis_angle(m3.vec3(1, 0, 0), 0.4))})
    # What the guard would already have settled: nothing unsaved stands in the way.
    viewer.editor.dirty = False
    return viewer, _pose_of(viewer)


_SAVED = {
    "id": "p1",
    "name": "Crouch",
    "bones": {"hip": list(m3.quat_from_axis_angle(m3.vec3(0, 1, 0), 0.9))},
    "root_translation": [0.1, 0.0, 0.25],
}


def test_applying_a_saved_pose_undoes_back_to_the_previous_pose_in_one_step():
    viewer, before = _posed_viewer()
    ctx = SimpleNamespace(viewer=viewer)

    pose_panel._apply_saved_pose(ctx, {"id": "job"}, _SAVED, "p1")
    assert _pose_of(viewer) != before

    assert viewer.editor.undo()
    assert _pose_of(viewer) == before, (
        "one Ctrl+Z after Apply must land on the pose the user had, not the rest pose "
        "the reset step left behind"
    )
    assert viewer.editor.redo()
    assert _pose_of(viewer) == [round(v, 4) for v in _SAVED["bones"]["hip"]]
    assert viewer.editor.root_translation() == pytest.approx([0.1, 0.0, 0.25])


def test_applying_a_saved_asset_pose_in_the_poser_undoes_back_in_one_step(svc):
    """The same defect in ``poser_mode.apply_asset_pose``, which the audit names as
    the second site. The Poser's test viewer routes ``set_pose``/``reset_all``/
    ``set_root_translation`` straight to the editor, which would never push the
    second step, so this one binds the real ``PoseOps`` methods."""
    pm = _load(_HERE / "modes" / "poser" / "test_poser_mode.py", "_poser_mode_helpers_pose")

    class PoserViewer(pm.FakeViewer):
        set_pose = PoseOps.set_pose
        reset_all = PoseOps.reset_all
        set_root_translation = PoseOps.set_root_translation
        _after_pose_change = PoseOps._after_pose_change
        _notify_pose_dirty = PoseOps._notify_pose_dirty
        gpu = None
        on_pose_dirty = None
        _render_dirty = False

    ctx = pm.FakeCtx(svc)
    state = poser_mode.ensure(ctx)
    state.job_id = "job"
    names = ["hips"]
    viewer = ctx.poser_viewer = PoserViewer(_Model(names), names)
    viewer.pose_job_id = "job"
    viewer.editor.root = "hips"
    with viewer.editor.record():
        viewer.editor.apply({"hips": list(m3.quat_from_axis_angle(m3.vec3(1, 0, 0), 0.4))})
    viewer.editor.dirty = False
    before = [round(v, 4) for v in viewer.editor.pose()["hips"]]
    state.asset_poses = [
        {"id": "p1", "name": "Crouch", "bones": {"hips": _SAVED["bones"]["hip"]}}
    ]

    poser_mode.apply_asset_pose(ctx, "p1")
    assert [round(v, 4) for v in viewer.editor.pose()["hips"]] != before

    assert viewer.editor.undo()
    assert [round(v, 4) for v in viewer.editor.pose()["hips"]] == before


# --- create-50 ---------------------------------------------------------------


class _EnterViewer:
    """What ``pose_panel`` touches on the shared viewer, recording the thread of
    every call so a regression shows up as the test thread's name."""

    def __init__(self) -> None:
        self.pose_mode = False
        self.pose_job_id: str | None = None
        self.pending: Path | None = None
        self.pose_loading: Any = None
        self.parsed_on: list[str] = []
        self.loaded_on: list[str] = []
        self.adopted: list[tuple[Any, Path, str]] = []
        self.entered: list[tuple[Any, str]] = []
        self.skinned = True
        self.editor = SimpleNamespace(mode="pose", has_unsaved_edits=lambda: False)

    def load_model(self, path: Path) -> None:
        self.loaded_on.append(threading.current_thread().name)

    def parse_model(self, path: Path) -> Any:
        self.parsed_on.append(threading.current_thread().name)
        return ("parsed", Path(path))

    def adopt_model(self, parsed: Any, path: Path) -> None:
        self.adopted.append((parsed, Path(path), threading.current_thread().name))
        self.pending = None
        self.pose_mode = False

    def enter_pose_mode(self, rig: Any, job_id: str) -> bool:
        if not self.skinned:
            return False
        self.entered.append((rig, job_id))
        self.pose_mode = True
        self.pose_job_id = job_id
        return True


class _Ctx:
    """``submit`` on a real worker thread, joined -- the test_frame_thread_doors shape."""

    def __init__(self, root: Path, viewer: _EnterViewer, selected: str = "job1") -> None:
        self.root = root
        self.viewer = viewer
        self.svc = object()
        self.state = SimpleNamespace(selected=selected, preview={})
        self.submitted: list[str] = []
        self.toasts: list[tuple[str, str]] = []
        self.presets: list[Any] = []
        self.refreshed = 0

    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def toast(self, text: str, level: str = "info", *a: Any, **k: Any) -> None:
        self.toasts.append((text, level))

    def load_presets(self, template: Any) -> None:
        self.presets.append(template)

    def refresh_rig_data(self) -> None:
        self.refreshed += 1

    def busy(self, key: str) -> bool:
        return False

    def submit(self, key: str, fn: Any, *args: Any, tag: Any = None, **kwargs: Any) -> bool:
        self.submitted.append(key)
        box: dict[str, Any] = {}

        def go() -> None:
            try:
                box["result"] = fn(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001 - re-raised on the caller
                box["error"] = exc

        worker = threading.Thread(target=go, name=WORKER)
        worker.start()
        worker.join()
        if "error" in box:
            raise box["error"]
        return True


def _rigged(tmp_path: Path, job_id: str = "job1") -> dict[str, Any]:
    folder = tmp_path / job_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "rig.glb").write_bytes(b"glTF")
    return {"id": job_id, "files": ["model.glb", "rig.glb"]}


@pytest.fixture
def no_rig_json(monkeypatch):
    monkeypatch.setattr(pose_panel.svc_rig, "get_rig", lambda svc, job_id: {"template": "humanoid"})


def test_entering_pose_mode_parses_the_rig_off_the_frame_thread(tmp_path, no_rig_json):
    viewer = _EnterViewer()
    ctx = _Ctx(tmp_path, viewer)
    job = _rigged(tmp_path)

    pose_panel._enter(ctx, job)

    assert viewer.parsed_on == [WORKER], "the glTF parse must run on the task thread"
    assert viewer.loaded_on == [], "load_model is the blocking spelling and is not the door"
    assert viewer.adopted == [] and viewer.pose_mode is False, (
        "nothing is adopted on the press: that is the landing's half"
    )

    pose_panel.land_enter(ctx, job)

    here = threading.current_thread().name
    assert [(a[1], a[2]) for a in viewer.adopted] == [(tmp_path / "job1" / "rig.glb", here)]
    assert viewer.entered == [({"template": "humanoid"}, "job1")]
    assert viewer.pose_mode is True
    assert ctx.presets == ["humanoid"] and ctx.refreshed == 1


def test_a_rig_parse_that_lands_for_another_selection_is_dropped(tmp_path, no_rig_json):
    viewer = _EnterViewer()
    ctx = _Ctx(tmp_path, viewer)
    job = _rigged(tmp_path)
    pose_panel._enter(ctx, job)

    other = _rigged(tmp_path, "job2")
    ctx.state.selected = "job2"
    pose_panel.land_enter(ctx, other)

    assert viewer.adopted == [] and viewer.entered == []
    assert viewer.pending is None and viewer.pose_loading is None


def test_a_rig_parse_that_the_viewport_has_since_replaced_is_dropped(tmp_path, no_rig_json):
    """``adopt_model`` and ``clear`` null ``pending``; that is the freshness check."""
    viewer = _EnterViewer()
    ctx = _Ctx(tmp_path, viewer)
    job = _rigged(tmp_path)
    pose_panel._enter(ctx, job)

    viewer.pending = None  # something else loaded or cleared the viewport meanwhile
    pose_panel.land_enter(ctx, job)

    assert viewer.adopted == [] and viewer.entered == []


def test_edit_pose_is_a_loading_state_while_the_parse_is_pending(tmp_path, no_rig_json):
    viewer = _EnterViewer()
    ctx = _Ctx(tmp_path, viewer)
    job = _rigged(tmp_path)
    assert pose_panel.entering(viewer, job) is False

    pose_panel._enter(ctx, job)
    assert pose_panel.entering(viewer, job) is True
    assert pose_panel.entering(viewer, {"id": "job2"}) is False

    pose_panel._enter(ctx, job)  # a second press while the first is in flight
    assert ctx.submitted == [f"{pose_panel.ENTER_KEY_PREFIX}job1"], "no second parse is queued"

    pose_panel.land_enter(ctx, job)
    assert pose_panel.entering(viewer, job) is False


def test_apply_from_outside_pose_mode_waits_for_the_rig_then_applies_in_one_undo_step(
    tmp_path, no_rig_json
):
    """Apply with the editor closed enters first. The pose it was pressed for is
    carried across the landing and applied then, as one undo step."""
    viewer = _RealOpsViewer()
    viewer.pose_mode = False
    viewer.pose_job_id = None
    viewer.pending = None
    viewer.pose_loading = None
    viewer.parsed = []

    def parse_model(path):
        viewer.parsed.append(threading.current_thread().name)
        return ("parsed", path)

    def adopt_model(parsed, path):
        viewer.pending = None

    def enter_pose_mode(rig, job_id):
        viewer.pose_mode = True
        viewer.pose_job_id = job_id
        return True

    viewer.parse_model = parse_model
    viewer.adopt_model = adopt_model
    viewer.enter_pose_mode = enter_pose_mode
    ctx = _Ctx(tmp_path, viewer)
    job = _rigged(tmp_path)
    steps = len(viewer.editor.history)

    pose_panel._apply_saved_pose(ctx, job, _SAVED, "p1")
    assert viewer.parsed == [WORKER]
    assert _pose_of(viewer) == [0.0, 0.0, 0.0, 1.0], "nothing is applied until the rig lands"

    pose_panel.land_enter(ctx, job)
    assert _pose_of(viewer) == [round(v, 4) for v in _SAVED["bones"]["hip"]]
    assert len(viewer.editor.history) == steps + 1
