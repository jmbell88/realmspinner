"""Poser's skeleton editor: the pane half of P6 (2026-09-13).

Drawn from ``poser_controls.draw``, inside the same column and the same
``forms.Form`` as the pose controls it takes over from while a
skeleton-editing session is open -- the right sidebar shows one editing
surface at a time, never a pose-rotate panel over an armature that has no
pose (``PoseEditor.enter_skeleton_mode`` resets it to rest on the way in).

Every mutating control goes through ``poser_mode``'s own ``skeleton_*``
doors, never the viewer or the editor directly, the ``poser_controls`` rule:
state and logic live in the controller, this file only draws it.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import controls, poser_mode, theme, widgets
from ..manual import render as manual_render
from ..tokens import sp


def draw(ctx: Any) -> None:
    state = poser_mode.ensure(ctx)
    viewer = poser_mode.viewer_of(ctx)
    widgets.section("Skeleton")
    manual_render.help_button(ctx, "poser-skeleton")
    if not state.skeleton_editing:
        _entry(ctx, state, viewer)
        return
    _editor(ctx, state, viewer)


def _entry_reason(state: Any, viewer: Any) -> str:
    """Why "Edit skeleton" is disabled right now, or "" if it is not.

    A pure function for ``poser_clips``'s own reason: a greyed control naming
    the wrong cause is a defect this codebase already treats as one
    (``_not_posing_reason``, ``inker_mode._no_document_reason``), and that is
    only assertable without a GL context if the words live outside the draw
    call.
    """
    if not state.job_id or viewer is None or not viewer.pose_mode:
        return "Open a rigged asset first."
    if viewer.editor.mode == "skeleton":
        return ""
    if viewer.editor.has_unsaved_edits():
        return "Save or reset the pose before editing the skeleton."
    if not state.asset_rig or not state.asset_rig.get("bones"):
        return "This asset has no readable rig to edit."
    return ""


def _entry(ctx: Any, state: Any, viewer: Any) -> None:
    reason = _entry_reason(state, viewer)
    widgets.muted_wrapped(
        "Add or remove pivots, rename bones and graft limb presets onto this "
        "asset's own skeleton."
    )
    if widgets.disabled_button(
        "Edit skeleton",
        not reason,
        (-1, 0),
        reason=reason,
        tooltip="Enter skeleton editing. Applying re-rigs and reweights the mesh.",
    ):
        poser_mode.enter_skeleton_edit(ctx)


def _error_for(state: Any, field: str) -> str:
    """The skeleton pane's own field-addressed refusal, or "" if none names
    this control. ``state.skeleton_error`` rather than the app-wide
    ``ctx.state.field_errors`` ring: this pane's controls (Add child, the
    rename box, Apply) are not wired into that mechanism."""
    error = state.skeleton_error
    if error is None or error.get("field") != field:
        return ""
    return str(error.get("message") or "")


def _editor(ctx: Any, state: Any, viewer: Any) -> None:
    from ... import rigging

    editor = viewer.editor
    count = len(editor.draft)
    widgets.text_colored(
        theme.ACCENT if count <= rigging.MAX_SKELETON_BONES else theme.ERR,
        f"{count} / {rigging.MAX_SKELETON_BONES} bones",
    )
    selected = editor.selected_bone()
    _rename(ctx, state, editor, selected)
    imgui.dummy((0, sp(4)))
    _structure_buttons(ctx, state, viewer, selected)
    imgui.dummy((0, sp(4)))
    _limb_row(ctx, state, viewer, selected)
    imgui.dummy((0, sp(4)))
    changed, mirror = controls.checkbox(
        "Mirror edits",
        bool(editor.mirror_edit),
        tooltip="Add child/Split/Delete also apply to the bone's .L/.R partner.",
    )
    if changed:
        editor.mirror_edit = bool(mirror)
    general = _error_for(state, "")
    if general:
        widgets.wrapped(theme.ERR, general)
    imgui.dummy((0, sp(8)))
    _actions(ctx, state, viewer)


def _rename(ctx: Any, state: Any, editor: Any, selected: str | None) -> None:
    if selected is None:
        widgets.muted("Click a pivot to select it.")
        return
    if state.skeleton_rename_for != selected:
        state.skeleton_rename = selected
        state.skeleton_rename_for = selected
    widgets.field_label("Name")
    entered, value = controls.input_text(
        "##poser-skeleton-rename",
        state.skeleton_rename,
        imgui.InputTextFlags_.enter_returns_true.value,
        error=_error_for(state, "name") or _error_for(state, "new"),
    )
    state.skeleton_rename = value
    if (entered or imgui.is_item_deactivated()) and value != selected:
        poser_mode.skeleton_rename(ctx, selected, value)
        # A refused rename (a duplicate, a bad name) must leave the box
        # showing what was typed, not silently revert to the old name --
        # the same "a failed write leaves the guard standing" rule every
        # other save in this mode follows.
        if state.skeleton_error is None:
            state.skeleton_rename_for = value
    field_error = _error_for(state, "name") or _error_for(state, "new")
    if field_error:
        widgets.wrapped(theme.ERR, field_error)


def _structure_buttons(ctx: Any, state: Any, viewer: Any, selected: str | None) -> None:
    editor = viewer.editor
    has_selection = selected is not None
    if widgets.disabled_button(
        "Add child",
        has_selection,
        reason="Select a pivot first.",
        tooltip="A new pivot, continuing this one's own direction.",
    ):
        poser_mode.skeleton_add_child(ctx, selected)
    widgets.same_line_or_wrap(widgets.button_width("Split"))
    if widgets.disabled_button(
        "Split",
        has_selection,
        reason="Select a pivot first.",
        tooltip="Cut this bone at its midpoint, inserting a new pivot there.",
    ):
        poser_mode.skeleton_split(ctx, selected)
    widgets.same_line_or_wrap(widgets.button_width("Delete pivot"))
    is_root = has_selection and selected == editor.draft_root
    if widgets.disabled_button(
        "Delete pivot",
        has_selection,
        reason="Select a pivot first.",
        tooltip="Remove just this bone; its children reparent to its own parent.",
    ):
        poser_mode.skeleton_remove_pivot(ctx, selected)
    widgets.same_line_or_wrap(widgets.button_width("Delete limb"))
    if widgets.disabled_button(
        "Delete limb",
        has_selection and not is_root,
        reason="The root cannot be removed as a subtree." if is_root else "Select a pivot first.",
        tooltip="Remove this bone and everything beneath it. Asks first.",
    ):
        poser_mode.skeleton_remove_subtree(ctx, selected)
    field_error = _error_for(state, "parent") or _error_for(state, "root")
    if field_error:
        widgets.wrapped(theme.ERR, field_error)


def _limb_row(ctx: Any, state: Any, viewer: Any, selected: str | None) -> None:
    presets = poser_mode.limb_preset_rows(ctx)
    widgets.section("Add limb")
    if not presets:
        widgets.muted("No limb presets are installed.")
        return
    keys = {row["key"] for row in presets}
    if state.limb_preset not in keys:
        state.limb_preset = next(iter(keys))
    state.limb_preset = widgets.labeled_combo(
        "Preset",
        state.limb_preset,
        [(row["key"], f"{row['label']} ({row['bone_count']} bones)") for row in presets],
    )
    state.limb_side = widgets.labeled_combo(
        "Side",
        state.limb_side,
        [("", "Centre"), ("L", "Left"), ("R", "Right")],
    )
    changed, mirror = controls.checkbox(
        "Mirror",
        state.limb_mirror,
        tooltip="Also graft the preset onto the opposite side.",
    )
    if changed:
        state.limb_mirror = bool(mirror)
    if widgets.disabled_button(
        "Add limb",
        selected is not None,
        (-1, 0),
        reason="Select the bone to attach it to first.",
        tooltip="Graft the preset onto the selected bone.",
    ):
        poser_mode.skeleton_attach_limb(
            ctx, state.limb_preset, selected, state.limb_side, state.limb_mirror
        )
    field_error = _error_for(state, "preset_key") or _error_for(state, "side")
    if field_error:
        widgets.wrapped(theme.ERR, field_error)


def _actions(ctx: Any, state: Any, viewer: Any) -> None:
    busy = ctx.busy(f"{poser_mode.ASSET_RERIG_KEY_PREFIX}{state.job_id}")
    if widgets.disabled_button(
        "Apply skeleton",
        not busy,
        (-1, 0),
        reason="Still re-rigging this asset." if busy else "",
        tooltip="Re-rig and reweight the mesh in Blender on this skeleton. Takes a moment.",
    ):
        poser_mode.apply_skeleton(ctx)
    field_error = _error_for(state, "bones")
    if field_error:
        widgets.wrapped(theme.ERR, field_error)
    if controls.button("Cancel", (-1, 0), tooltip="Leave skeleton editing. Asks first if unsaved."):
        poser_mode.cancel_skeleton_edit(ctx)
