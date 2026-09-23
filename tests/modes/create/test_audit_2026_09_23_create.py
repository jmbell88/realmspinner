"""Regression tests for the 2026-09-23 audit's Create findings.

create-01: ``settings_2d._reset`` used to do ``ctx.state.preview = {}``, which
throws away every key the *3D* side of Create keeps in that shared dict --
poses, sheets, bones, ``library_poses`` and the Mesh pane's
``_LAST_AUTO_MATTE_SLOT`` -- even though the Reset confirm the command bar
shows says only "The 3D form is untouched."

create-02: applying a saved pose (``pose_panel._apply_saved_pose`` and
``poser.mode.apply_asset_pose`` both run the same
``reset_all`` / ``set_pose`` / ``set_root_translation`` sequence) used to push
only the ``reset_all`` step, because neither ``Viewer.set_pose`` nor
``Viewer.set_root_translation`` wrapped its editor call in ``record()`` at
all -- so redoing that one step replayed the bare reset, not the pose that
was loaded after it.

create-04: ``generation.request_from_legacy`` raised ``KeyError`` on an
advanced-mode form that carried ``base_model`` but no ``model_override`` key
at all, because its guard only checks that *one* of the two is truthy before
indexing both with ``form[...]``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from realmspinner import generation
from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.studio import dialogs
from realmspinner.studio._viewer_pose import PoseOps
from realmspinner.studio.modes.create.ui.panes import settings_2d, settings_3d
from realmspinner.studio.state import default_form_2d
from realmspinner.studio.viewer.pose import PoseEditor

# --- create-01 ---------------------------------------------------------------


class _Ctx:
    def __init__(self) -> None:
        self.state = SimpleNamespace(
            form_2d=default_form_2d(),
            source_job="job-1",
            preview={
                # 2D-owned: what settings_2d._reset itself may write.
                settings_2d.TILE_MODE_CLEARED_KEY: ["a tile-mode note"],
                settings_2d.CLEARED_KEY: ["a base-model note"],
                # Shared with the 3D side -- a rig's pose stage, the Mesh
                # pane's auto-matte memory, Poser's pose library.
                "poses": [{"id": "p1"}],
                "sheets": [{"id": "s1"}],
                "bones": ["hip", "spine"],
                "library_poses": [{"id": "lp1"}],
                settings_3d._LAST_AUTO_MATTE_SLOT: {"slot": 3},
            },
            preview_dirty_at=0.0,
        )
        self.confirms = dialogs.ConfirmQueue()
        self.toasts: list[str] = []

    def toast(self, text: str, level: str = "info", *a, **kw) -> None:
        self.toasts.append(text)


def test_reset_leaves_the_rig_side_data_of_the_selected_asset_alone():
    """create-01: Reset clears only what settings_2d itself put in ``preview``."""
    ctx = _Ctx()
    settings_2d._reset(ctx)
    assert ctx.state.preview["poses"] == [{"id": "p1"}]
    assert ctx.state.preview["sheets"] == [{"id": "s1"}]
    assert ctx.state.preview["bones"] == ["hip", "spine"]
    assert ctx.state.preview["library_poses"] == [{"id": "lp1"}]
    assert ctx.state.preview[settings_3d._LAST_AUTO_MATTE_SLOT] == {"slot": 3}


def test_reset_still_drops_its_own_cleared_notes():
    """The keys settings_2d itself writes are still gone -- Reset is not a
    no-op, only narrower than wiping the whole shared dict."""
    ctx = _Ctx()
    settings_2d._reset(ctx)
    assert settings_2d.TILE_MODE_CLEARED_KEY not in ctx.state.preview
    assert settings_2d.CLEARED_KEY not in ctx.state.preview


# --- create-02 ----------------------------------------------------------------


class _FakeNode:
    def __init__(self, world: np.ndarray, translation: np.ndarray) -> None:
        self.world = world
        self.translation = translation


class _FakeModel:
    """Just enough of ``gltf.Model`` for ``PoseEditor``: named nodes, rotations
    and a rest translation per node, so root-translation moves have something
    to add a delta onto."""

    def __init__(self, names: list[str]) -> None:
        self.by_name = {n: i for i, n in enumerate(names)}
        self.rotations = {n: m3.quat_identity() for n in names}
        self.nodes = [
            _FakeNode(m3.translation(m3.vec3(i, 0, 0)), m3.vec3(0, 0, 0))
            for i, _ in enumerate(names)
        ]
        self.rest_translations = [m3.vec3(0, 0, 0) for _ in names]
        self.skins = []

    def get_rotation(self, bone):
        return self.rotations.get(bone)

    def set_rotation(self, bone, quat):
        if bone not in self.rotations:
            return False
        self.rotations[bone] = m3.quat_normalize(np.asarray(quat, dtype="f8"))
        return True

    def update_world(self) -> None:
        pass


class _FakeViewer(PoseOps):
    """``PoseOps`` is a mixin over ``Viewer`` (methods only, per its module
    docstring) -- this is the smallest object that carries what it reads:
    ``editor``, ``gpu`` and ``on_pose_dirty``."""

    def __init__(self) -> None:
        self.editor = PoseEditor()
        self.gpu = None
        self.on_pose_dirty = None
        self._render_dirty = False
        self.pose_mode = True


def _bound_viewer() -> _FakeViewer:
    viewer = _FakeViewer()
    viewer.editor.bind(_FakeModel(["hip"]), ["hip"])
    viewer.editor.root = "hip"
    return viewer


def test_applying_a_saved_pose_pushes_exactly_one_undo_step():
    """create-02: ``set_pose`` + ``set_root_translation`` -- the sequence
    both ``pose_panel._apply_saved_pose`` and ``poser.mode.apply_asset_pose``
    run after their own ``reset_all`` -- fold into one step, not zero (the
    bug: neither call pushed anything) and not two (each pushing its own
    would still be closer, but the audit asked for one)."""
    viewer = _bound_viewer()
    saved_bones = {"hip": list(m3.quat_from_axis_angle(m3.vec3(0, 1, 0), 0.9))}

    viewer.reset_all(dirty=False)
    steps_after_reset = len(viewer.editor.history)

    viewer.set_pose(saved_bones, pose_id="P1", dirty=False)
    viewer.set_root_translation([0.1, 0.0, 0.0], dirty=False)

    assert len(viewer.editor.history) == steps_after_reset + 1


def test_redo_after_applying_a_saved_pose_restores_it_not_the_reset_state():
    """create-02, reproduced through the real call path (``Viewer.set_pose``/
    ``Viewer.set_root_translation``, not the editor directly): load a saved
    pose over a manually-posed rig, undo once, redo once, and land back on
    the loaded pose's bones *and* root offset -- not the bare reset the bug
    left behind in ``after``."""
    viewer = _bound_viewer()

    manual_pose = list(m3.quat_from_axis_angle(m3.vec3(1, 0, 0), 0.4))
    with viewer.editor.record():
        viewer.editor.apply({"hip": manual_pose})

    saved_bones = {"hip": list(m3.quat_from_axis_angle(m3.vec3(0, 1, 0), 0.9))}
    viewer.reset_all(dirty=False)
    viewer.set_pose(saved_bones, pose_id="P1", dirty=False)
    viewer.set_root_translation([0.1, 0.0, 0.0], dirty=False)

    loaded_pose = [round(v, 6) for v in viewer.editor.pose()["hip"]]
    loaded_root = [round(v, 6) for v in viewer.editor.root_translation()]
    assert loaded_pose == [round(v, 6) for v in saved_bones["hip"]]

    assert viewer.editor.undo()
    assert viewer.editor.redo()

    redo_pose = [round(v, 6) for v in viewer.editor.pose()["hip"]]
    redo_root = [round(v, 6) for v in viewer.editor.root_translation()]
    assert redo_pose == loaded_pose
    assert redo_root == loaded_root


# --- create-04 -----------------------------------------------------------------


def test_request_from_legacy_does_not_crash_on_an_advanced_form_missing_model_override():
    """create-04: an advanced form naming only ``base_model`` (no
    ``model_override`` key at all) used to raise ``KeyError`` -- unreachable
    from the shipped UI, which always writes both, but reachable from any
    other caller of this adapter (an MCP tool, a saved recipe replayed by
    hand)."""
    form = {
        "asset_type": "image",
        "generation_type": "image",
        "model_mode": "advanced",
        "base_model": "sdxl-turbo",
    }
    request = generation.request_from_legacy(form)
    assert request.model_override == "sdxl-turbo"
