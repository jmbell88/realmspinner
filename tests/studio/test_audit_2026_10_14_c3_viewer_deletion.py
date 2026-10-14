"""The 2026-10-14 audit's cleanup round C3: the dead Viewer / skeleton-editor surface.

Finding create-viewer-03 (F56) named the Viewer's strip, onion, skeleton-mode and
right-click-menu code as having no production caller once Poser left; F53's
poser-rig-03 named the skeleton-editing helpers in ``kernels/rig`` and the five
limb presets the same way. These are the deletion's tests: each symbol is gone, and
what is kept is kept because ``src/`` still calls it (the notes say which).
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pygame
import pytest

import realmspinner
from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.kernels.geom3d.gltf import Model, Node
from realmspinner.kernels.rig import skeleton, templates
from realmspinner.studio import _viewer_pose, viewer_embed
from realmspinner.studio.viewer import bonelines, capture, render
from realmspinner.studio.viewer.pose import PoseEditor

SRC = Path(realmspinner.__file__).resolve().parent


# --- the Viewer ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "load_model",
        "frame_bounds",
        "begin_sheet_strip",
        "step_sheet_strip",
        "cancel_sheet_strip",
        "strip_progress",
        "onion",
        "ghostlines",
        "enter_pose_authoring",
        "enter_skeleton_mode",
        "exit_skeleton_mode",
        "skeleton_payload",
        "subtree_size",
        "skel_add_child",
        "skel_split",
        "skel_remove_pivot",
        "skel_remove_subtree",
        "skel_rename",
        "skel_attach_limb",
    ],
)
def test_the_viewer_has_no_method_nothing_calls(name):
    assert not hasattr(viewer_embed.Viewer, name), name


def test_the_pose_mixin_binds_only_an_asset_rig():
    assert hasattr(_viewer_pose.PoseOps, "enter_pose_mode")
    assert not hasattr(_viewer_pose.PoseOps, "enter_pose_authoring")


def test_no_viewer_source_names_the_deleted_strip_or_ghost_machinery():
    sources = {
        path.name: path.read_text(encoding="utf-8")
        for path in [
            SRC / "studio" / "viewer_embed.py",
            SRC / "studio" / "_viewer_pose.py",
            *sorted((SRC / "studio" / "viewer").glob("*.py")),
        ]
    }
    for needle in (
        "sheet_strip",
        "StripRender",
        "ghost_handles",
        "draft_segments",
        "unpremultiply",
        "keep_alpha",
        "skel_",
        "menu_request",
    ):
        for name, text in sources.items():
            assert needle not in text, f"{name} still mentions {needle!r}"


def test_the_direction_strip_module_is_gone():
    assert not (SRC / "studio" / "viewer" / "sheet.py").exists()


def test_the_shell_does_not_cancel_a_strip_that_no_longer_exists():
    tasks = (SRC / "studio" / "shell" / "tasks.py").read_text(encoding="utf-8")
    assert "cancel_sheet_strip" not in tasks
    assert "viewer.load_model" not in tasks


def test_a_right_button_release_is_not_routed_to_a_menu_the_viewer_no_longer_has():
    """``handle_event`` used to send button 3's release to ``FrameOps._rmb_release``,
    which reads ``self._rmb_at``/``self.menu_request`` -- attributes the Viewer no
    longer builds. Over a stub without them the old route raised AttributeError."""
    stub = SimpleNamespace(
        _local=lambda event: (5.0, 5.0),
        _render_dirty=False,
        _grab=None,
        _deselect_on_click=False,
    )
    stub._release = lambda button: viewer_embed.Viewer._release(stub, button)
    event = pygame.event.Event(pygame.MOUSEBUTTONUP, button=3, pos=(5, 5))

    assert viewer_embed.Viewer.handle_event(stub, event, hovered=True) is False


def test_the_viewer_builds_no_right_button_state():
    source = inspect.getsource(viewer_embed.Viewer.__init__)
    assert "_rmb_at" not in source
    assert "menu_request" not in source


# --- the pose editor ----------------------------------------------------------------


def _model() -> Model:
    nodes = [
        Node(name="rig", children=[1]),
        Node(name="hips", translation=m3.vec3(0.0, 0.5, 0.0), children=[2]),
        Node(name="spine", translation=m3.vec3(0.0, 0.1, 0.0)),
    ]
    return Model(nodes, roots=[0], meshes=[], skins=[])


def test_the_pose_editor_carries_no_skeleton_draft():
    editor = PoseEditor()
    editor.bind(_model(), ["hips", "spine"])
    for attribute in (
        "draft",
        "draft_pairs",
        "draft_root",
        "draft_dirty",
        "mirror_edit",
        "root_translate",
    ):
        assert not hasattr(editor, attribute), attribute
    assert not any(key.startswith("draft") for key in editor.snapshot())


def test_the_pose_editor_has_no_skeleton_mode_or_ghost_walk():
    from realmspinner.studio.viewer import pose

    for name in (
        "enter_skeleton_mode",
        "exit_skeleton_mode",
        "skeleton_payload",
        "subtree_size",
        "selected_bone",
        "move_root",
        "skel_add_child",
        "_recompute_skeleton_handles",
    ):
        assert not hasattr(PoseEditor, name), name
    assert not hasattr(pose, "ghost_handles")


def test_a_joints_drag_still_moves_the_marker_and_records_the_blender_delta():
    """``move_handle`` lost its skeleton branch and kept the joints one."""
    editor = PoseEditor()
    editor.bind(_model(), ["hips", "spine"])
    editor.enter_joints_mode()
    start = editor.handles["spine"].copy()

    editor.move_handle("spine", start + np.array([0.0, 0.2, 0.0]))

    assert editor.moved["spine"] == pytest.approx(m3.gltf_delta_to_blender(np.array([0, 0.2, 0])))


def test_the_active_gizmo_is_translate_in_joints_mode_and_rotate_otherwise():
    editor = PoseEditor()
    editor.bind(_model(), ["hips", "spine"])
    stub = SimpleNamespace(
        pose_mode=True,
        editor=editor,
        translate_gizmo="translate",
        rotate_gizmo="rotate",
    )
    assert _viewer_pose.PoseOps._active_gizmo(stub) == "rotate"
    editor.mode = "joints"
    assert _viewer_pose.PoseOps._active_gizmo(stub) == "translate"
    stub.pose_mode = False
    assert _viewer_pose.PoseOps._active_gizmo(stub) is None


# --- bonelines, capture, render -----------------------------------------------------


def test_bonelines_has_no_draft_or_ghost_paint():
    assert not hasattr(bonelines, "draft_segments")
    assert not hasattr(bonelines, "GHOST")
    assert not hasattr(bonelines, "GHOST_ALPHA")
    # ``colour`` and ``alpha`` stay: Clay's pose view passes them (see the r3 test).


def test_capture_has_no_straight_alpha_path():
    assert not hasattr(capture, "unpremultiply")
    assert "opaque" not in inspect.signature(capture.png_bytes).parameters


def test_the_renderer_lost_keep_alpha_but_kept_background():
    """``background`` has a production caller -- ``ClayView.render_png`` draws on a
    plain white clear -- so it stays; ``keep_alpha`` served only the strip."""
    parameters = inspect.signature(render.Renderer.draw).parameters
    assert "keep_alpha" not in parameters
    assert "background" in parameters
    clay = (SRC / "studio" / "modes" / "clay" / "ui" / "view.py").read_text(encoding="utf-8")
    assert "background=(1.0, 1.0, 1.0, 1.0)" in clay


# --- the rig kernel -----------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "add_bone",
        "split_bone",
        "remove_pivot",
        "remove_subtree",
        "rename_bone",
        "prune_pairs",
        "mirror_partner_name",
        "unique_name",
        "_limb_frame",
        "_limb_point",
        "attach_limb",
        "clip_coverage",
    ],
)
def test_the_skeleton_module_has_no_editor_nothing_calls(name):
    assert not hasattr(skeleton, name), name


def test_the_skeleton_module_keeps_the_validators_service_rig_still_calls():
    kept = ("fit_template", "validate_joints", "validate_skeleton", "check_skeleton_structure")
    for name in kept:
        assert callable(getattr(skeleton, name)), name
    service_rig = (SRC / "service" / "rig.py").read_text(encoding="utf-8")
    assert "skeleton.validate_skeleton" in service_rig


def test_limb_presets_and_their_files_are_gone():
    for name in ("limb_presets", "get_limb_preset", "_load_limb_presets", "_parse_limb_preset"):
        assert not hasattr(templates, name), name
    assert not hasattr(templates, "LIMB_DIR")
    assert not (templates.TEMPLATE_DIR / "limbs").exists()
    assert templates.templates(), "the template registry itself is untouched"


def test_the_template_acyclic_check_survives_for_the_skeleton_validators():
    assert callable(templates._check_acyclic)
    assert "_check_acyclic" in inspect.getsource(skeleton)
