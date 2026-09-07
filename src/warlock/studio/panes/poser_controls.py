"""Poser's right sidebar: the selected joint, the root, saving.

The mirror of ``pose_panel``'s pose-mode half, over the Poser viewer instead
of the shared one -- which is the whole reason it can exist beside an open
inspector pose session without either discarding the other.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from imgui_bundle import imgui
from scipy.spatial.transform import Rotation

from .. import controls, forms, poser_mode, theme, tokens, widgets
from ..manual import render as manual_render
from ..tokens import sp
from ..viewer import math3d as m3

# Blender's own default pose-bone Euler order -- ``blender_worker.py`` never
# sets a per-bone ``rotation_mode`` other than QUATERNION for a rig bone, so
# there is no per-rig order to read back; XYZ is what a fresh bone's would be,
# and is the one order every rig in this app implicitly agrees on.
_EULER_ORDER = "xyz"


def _quat_to_euler_degrees(quat: Any) -> list[float]:
    return [float(v) for v in Rotation.from_quat(quat).as_euler(_EULER_ORDER, degrees=True)]


def _euler_degrees_to_quat(degrees: Any) -> list[float]:
    return [float(v) for v in Rotation.from_euler(_EULER_ORDER, degrees, degrees=True).as_quat()]


def _rotate_selected_to_euler(viewer: Any, degrees: Any) -> None:
    """Turn the selected joint to an absolute Euler orientation, in degrees.

    Goes through ``PoseEditor.rotate_selected`` -- the same post-multiplied
    delta the gizmo drag in ``viewer_embed`` calls -- rather than a second
    path straight onto the node: the delta between the current quaternion and
    the one the typed degrees describe, wrapped in the editor's own
    ``record()`` so a numeric edit is one undo step exactly like a drag is.
    """
    editor = viewer.editor
    if editor.model is None or editor.selected is None:
        return
    current = editor.model.get_rotation(editor.selected)
    if current is None:
        current = m3.quat_identity()
    target = _euler_degrees_to_quat(degrees)
    delta = m3.quat_mul(m3.quat_conjugate(current), target)
    with editor.record():
        editor.rotate_selected(delta)


def _changed_from_rest(viewer: Any, bone: str | None) -> bool:
    """Whether ``bone``'s live rotation differs from its rest pose."""
    if bone is None:
        return False
    editor = viewer.editor
    if editor.model is None:
        return False
    current = editor.model.get_rotation(bone)
    rest = editor.rest.get(bone)
    if current is None or rest is None:
        return False
    # A quaternion and its negation describe the same rotation, so the
    # comparison is on the dot product rather than component equality.
    return not np.isclose(abs(float(np.dot(current, rest))), 1.0, atol=1e-6)


def draw(ctx: Any) -> None:
    state = poser_mode.ensure(ctx)
    widgets.section("Pose")
    manual_render.help_button(ctx, "poser-controls")
    if not ctx.rigging_available:
        widgets.muted("Posing needs Blender, which is not installed.")
        return
    viewer = poser_mode.viewer_of(ctx)
    if viewer is None or not viewer.pose_mode:
        # Silently: the viewport beside this pane is already a centred empty
        # state saying the armature is being built, and this was the same fact
        # in different words a column away.
        return

    with forms.Form("poser-controls"):
        if state.job_id:
            _asset_banner(ctx, state)
            widgets.divider()
        _banner(state, viewer)
        _joint(ctx, viewer)
        _root(viewer)
        _save(ctx, state, viewer)


def _asset_banner(ctx: Any, state: Any) -> None:
    """Who this session is bound to, and the way back to browsing templates."""
    widgets.text_colored(theme.ACCENT, f"Editing pose for {state.asset_label or 'this asset'}")
    if controls.button("Close", tooltip="Return to browsing the shared skeleton library."):
        poser_mode.close_asset(ctx)


def _banner(state: Any, viewer: Any) -> None:
    record = state.find_asset_pose(viewer.editor.current) if state.job_id else state.find(
        viewer.editor.current
    )
    label = str(record.get("name")) if record else "New pose"
    if viewer.editor.has_unsaved_edits():
        label += " - unsaved changes"
    widgets.text_colored(theme.ACCENT, label)


def _joint(ctx: Any, viewer: Any) -> None:
    selected = viewer.selected_bone
    changed_from_rest = _changed_from_rest(viewer, selected)
    label = (selected + " *") if changed_from_rest and selected else selected
    widgets.muted(label or "Click a joint to rotate it.")
    if changed_from_rest and imgui.is_item_hovered():
        imgui.set_tooltip("Changed from rest")
    if selected is not None and viewer.editor.model is not None:
        current = viewer.editor.model.get_rotation(selected)
        if current is None:
            current = m3.quat_identity()
        degrees = _quat_to_euler_degrees(current)
        new_degrees = list(degrees)
        edited = False
        for axis, axis_label in enumerate(("Rotate X", "Rotate Y", "Rotate Z")):
            # ``commit=True``: undoable, the gizmo-drag rule -- per-keystroke
            # would push one undo step per digit typed.
            settled, value = controls.input_float(
                f"{axis_label}##poser-joint-rot-{axis}",
                float(degrees[axis]),
                commit=True,
                tooltip=f"{axis_label.split()[-1]} rotation, in degrees, Euler XYZ.",
            )
            if settled:
                new_degrees[axis] = value
                edited = True
        if edited:
            _rotate_selected_to_euler(viewer, new_degrees)
    if selected is None:
        # The whole of what this pane said about itself was the line above.
        # Two more sentences, inline rather than in a tooltip: there is room,
        # and a hover is a poor place to put the one rule of the mode that is
        # not guessable -- that there is no undo.
        widgets.muted_wrapped(
            "Drag the ring gizmo on a joint to rotate it. Every joint you "
            "touch is recorded in the pose; the ones you do not are left at "
            "rest, so a wave is two joints rather than a whole skeleton."
        )
        # It used to say "There is no undo here", which stopped being true
        # when Ctrl+Z / Ctrl+Shift+Z / Ctrl+Y were bound over
        # ``viewer.pose.undo()``: the pane was telling the user a working key
        # did not exist.
        widgets.muted_wrapped(
            "Ctrl+Z undoes a rotation. Reset joint puts one joint back and "
            "Reset all puts every joint back; both ask first if there is "
            "anything unsaved."
        )
    if widgets.disabled_button(
        "Reset joint",
        selected is not None,
        reason="Click a joint first.",
        tooltip="Put this one joint back to rest, leaving the rest of the pose.",
    ):
        viewer.reset_bone()
    imgui.same_line()
    # Behind the guard, exactly as ``poser_mode.new_pose`` is -- and it is the
    # same act: both put every joint back to rest, so both throw away whatever
    # was being authored. Bare, this was the one control in the mode that could
    # discard an unsaved pose with no way back, in the mode whose whole output
    # is the shared library every asset poses from.
    if controls.button(
        "Reset all",
        tooltip="Put every joint back to rest. There is no undo, so this asks first.",
    ):
        poser_mode.guard(ctx, "reset every joint", viewer.reset_all)
    if viewer.editor.mirror_pairs:
        # Hidden for a serpent or a fish, the pose_panel rule: a skeleton with
        # no mirror pairs has nothing to mirror.
        imgui.same_line()
        if controls.button(
            "Mirror",
            tooltip="Swap every left joint's rotation with its right one.",
        ):
            # Behind the guard for Reset all's reason: mirroring rewrites every
            # rotation, this mode has no undo at all, and it was the last bare
            # route to losing an unsaved pose. It does prompt mid-authoring --
            # accepted, because the alternative is an undo stack.
            poser_mode.guard(ctx, "mirror the pose", viewer.mirror)


def _root(viewer: Any) -> None:
    editor = viewer.editor
    if editor.root is None:
        return
    offset = editor.root_translation()
    if editor.selected == editor.root:
        # Shown only while the root is selected: the toggle changes what the
        # gizmo on that joint does, and drawing it against any other joint
        # would claim a capability the selection does not have.
        changed, value = controls.checkbox(
            "Move root",
            editor.root_translate,
            tooltip=(
                "Turns the root joint's gizmo from a rotator into arrows, so "
                "the whole pose can be offset rather than turned."
            ),
        )
        if changed:
            editor.root_translate = bool(value)
        if editor.root_translate:
            widgets.muted_wrapped(
                "Drag the arrows to offset the whole pose. Units are character "
                "heights; the bake scales them onto each asset's own rig."
            )
    new_offset = list(offset)
    edited = False
    for axis, axis_label in enumerate(("Offset X", "Offset Y", "Offset Z")):
        # ``commit=True`` for the joint fields' reason: undoable, so only the
        # settled value should push a step.
        settled, value = controls.input_float(
            f"{axis_label}##poser-root-offset-{axis}",
            float(offset[axis]),
            commit=True,
            tooltip="Character heights, Blender axes -- what the bake reads.",
        )
        if settled:
            new_offset[axis] = value
            edited = True
    if edited:
        # ``set_root_translation``: the entry point ``apply_key`` and
        # ``mirror`` already use to place the root from a *value* rather
        # than a live drag point -- ``move_root`` takes a model-space point
        # off the gizmo's own drag math, which a typed number has no way to
        # supply without recomputing that math a second time.
        with editor.record():
            editor.set_root_translation(new_offset)
    if any(offset):
        widgets.muted(
            f"root offset  x {offset[0]:+.2f}  y {offset[1]:+.2f}  z {offset[2]:+.2f}"
        )


def _save(ctx: Any, state: Any, viewer: Any) -> None:
    imgui.dummy((0, sp(tokens.SP_2)))
    if state.job_id:
        _save_asset(ctx, state)
        widgets.section("Reusable pose")
        widgets.muted_wrapped(
            "Also contribute this to the shared library, for every asset on "
            "this skeleton -- not just this one."
        )
    _save_library(ctx, viewer)


def _save_asset(ctx: Any, state: Any) -> None:
    busy = ctx.busy(f"{poser_mode.ASSET_SAVE_KEY_PREFIX}{state.job_id}")
    if widgets.disabled_button(
        "Save pose to this asset",
        not busy,
        (-1, 0),
        tooltip="Write this pose into the asset's own poses, the way the "
        "inspector's Pose tab would.",
    ):
        poser_mode.save_pose_to_asset(ctx)


def _save_library(ctx: Any, viewer: Any) -> None:
    busy = ctx.busy(poser_mode.SAVE_KEY)
    editing = viewer.editor.current is not None
    if editing and widgets.disabled_button(
        "Save",
        not busy,
        (-1, 0),
        tooltip="Overwrite the pose named above, in the shared library.",
    ):
        poser_mode.save(ctx)
    if widgets.disabled_button(
        "Save as reusable pose...",
        not busy,
        (-1, 0),
        tooltip="Add a new pose to the library every asset on this skeleton can use.",
    ):
        poser_mode.save_as(ctx)
