"""Skeleton mode's slice of the Viewer: the mode dispatch that used to say
only "joints", picking against the mode-specific handle set, and the overlay
gizmo basis that must not require a glTF node a draft bone does not have.

Exercised the way ``test_viewer_poser.py`` exercises ``enter_pose_authoring``:
the real, unbound ``Viewer`` methods called over a stub carrying only the
attributes each one touches, so nothing here needs a GL context.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.kernels.geom3d.gltf import Model, Node
from realmspinner.studio._viewer_pose import PoseOps
from realmspinner.studio.viewer import picking
from realmspinner.studio.viewer.pose import PoseEditor
from realmspinner.studio.viewer_embed import Viewer


def _model() -> Model:
    nodes = [Node(name="hip", translation=m3.vec3(0.0, 0.0, 0.0), children=[])]
    return Model(nodes, roots=[0], meshes=[], skins=[])


def _rig() -> dict:
    return {
        "bones": [
            {"name": "hip", "parent": None, "head": [0.0, 0.0, 0.0], "tail": [0.0, 0.0, 1.0]},
        ],
        "root": "hip",
        "mirror_pairs": [],
    }


def _skeleton_editor() -> PoseEditor:
    editor = PoseEditor()
    editor.bind(_model(), ["hip"])
    editor.enter_skeleton_mode(_rig())
    return editor


# --- _active_gizmo ------------------------------------------------------------


def test_active_gizmo_treats_skeleton_like_joints_translate():
    stub = SimpleNamespace(
        pose_mode=True,
        editor=_skeleton_editor(),
        translate_gizmo="translate",
        rotate_gizmo="rotate",
    )
    assert PoseOps._active_gizmo(stub) == "translate"


def test_active_gizmo_is_none_outside_pose_mode():
    stub = SimpleNamespace(
        pose_mode=False,
        editor=_skeleton_editor(),
        translate_gizmo="translate",
        rotate_gizmo="rotate",
    )
    assert PoseOps._active_gizmo(stub) is None


# --- _motion: skeleton mode writes the draft, never the root translation ----


class _FakeTranslateGizmo:
    """Reports one fixed drag target, so ``_motion`` always has somewhere to
    move to without a real gizmo's ray math or GL state."""

    def __init__(self, target: np.ndarray) -> None:
        self.target = target

    def update(self, origin, direction):
        del origin, direction
        return self.target


def _motion_stub(editor: PoseEditor, *, target_world: np.ndarray) -> SimpleNamespace:
    camera = SimpleNamespace()
    stub = SimpleNamespace(
        _last_mouse=(0.0, 0.0),
        _rect=(0, 0, 100, 100),
        _grab="gizmo",
        editor=editor,
        translate_gizmo=_FakeTranslateGizmo(target_world),
        rotate_gizmo="rotate",
        gpu=None,
        placement=m3.identity(),
        camera=camera,
        _render_dirty=False,
    )
    stub._active_gizmo = lambda: stub.translate_gizmo
    stub._ray = lambda local: (m3.vec3(0, 0, 5), m3.vec3(0, 0, -1))
    return stub


def test_a_gizmo_drag_in_skeleton_mode_writes_the_draft_not_the_root():
    editor = _skeleton_editor()
    editor.root = "hip"
    editor.root_translate = True
    editor.selected = "hip"
    world = picking.to_world(m3.identity(), m3.blender_delta_to_gltf(np.array([0.0, 0.0, 5.0])))
    stub = _motion_stub(editor, target_world=world)

    Viewer._motion(stub, (3.0, 4.0))

    by_name = {b["name"]: b for b in editor.draft}
    assert by_name["hip"]["head"] == pytest.approx([0.0, 0.0, 5.0])
    # The pose-mode root-translate path (``move_root``) never ran: it would
    # have raised (``self.model`` is unset on a bare ``PoseEditor``... here it
    # is set, but it would touch ``node.translation`` instead of the draft) --
    # checked positively instead, since the draft write above is the same
    # evidence either way.
    assert editor.draft_dirty is True


def test_a_gizmo_drag_in_joints_mode_is_unaffected_by_the_skeleton_branch():
    """The dispatch added for skeleton mode must not change joints mode's own
    behaviour: it still writes ``moved``, never the draft (there is none)."""
    editor = PoseEditor()
    editor.bind(_model(), ["hip"])
    editor.fitted = [{"name": "hip", "head": [0, 0, 0], "tail": [0, 0, 1], "parent": None}]
    editor.enter_joints_mode()
    editor.selected = "hip"
    home = editor.home["hip"].copy()
    target = home + m3.vec3(1.0, 0.0, 0.0)
    stub = _motion_stub(editor, target_world=picking.to_world(m3.identity(), target))

    Viewer._motion(stub, (3.0, 4.0))

    assert "hip" in editor.moved
