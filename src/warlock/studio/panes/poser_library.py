"""Poser's left sidebar: the skeleton, the pose library, the shipped presets.

Everything here is a call into :mod:`..poser_mode` -- the pane draws, the
controller decides, which is what keeps the guard logic (a library pose
overwrites the editor) in one importable place.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from ... import rigging
from .. import controls, icons, poser_mode, theme, tokens, widgets
from ..manual import render as manual_render
from ..tokens import sp


def draw(ctx: Any) -> None:
    state = poser_mode.ensure(ctx)
    # The pane is Poser's per-frame heartbeat, so the refresh flag is pumped
    # here -- the findings_dirty idiom; ``refresh`` only raises the flag.
    poser_mode.pump(ctx)
    widgets.section("Pose library")
    manual_render.help_button(ctx, "poser-library")
    if not ctx.rigging_available:
        # The pose_panel wording, verbatim: one sentence for one fact.
        widgets.muted("Posing needs Blender, which is not installed.")
        return

    if state.job_id:
        # The skeleton follows the bound asset's own rig; changing it makes no
        # sense until the asset session is closed, so the combo is replaced
        # with a fact rather than shown disabled with no way to act on it.
        label = next(
            (e["label"] for e in rigging.catalog() if e["key"] == state.template),
            state.template,
        )
        widgets.field_label("Skeleton")
        widgets.muted(f"{label} (from this asset's rig)")
    else:
        widgets.field_label("Skeleton")
        chosen = widgets.combo(
            "##poser-template",
            state.template,
            [(entry["key"], entry["label"]) for entry in rigging.catalog()],
        )
        if chosen and chosen != state.template:
            poser_mode.set_template(ctx, chosen)

    imgui.dummy((0, sp(tokens.SP_1)))
    if controls.button("New pose", (-1, 0)):
        poser_mode.new_pose(ctx)

    if state.job_id:
        _asset_poses(ctx, state)
    _library(ctx, state)
    _presets(ctx, state)


def _asset_poses(ctx: Any, state: Any) -> None:
    """Poses saved onto the bound asset itself, distinct from the shared
    library below -- ``service.rig.list_poses``, not ``service.poses``."""
    widgets.section("This asset's poses")
    if not state.asset_poses:
        widgets.empty_state(
            icons.PERSON_STANDING,
            "No saved poses",
            "Rotate a joint, then Save pose to this asset.",
        )
        return
    viewer = poser_mode.viewer_of(ctx)
    editing = None if viewer is None else viewer.editor.current
    needle = widgets.list_filter(ctx, "poser-asset-poses", len(state.asset_poses))
    shown = 0
    for pose in state.asset_poses:
        pose_id = str(pose.get("id") or "")
        name = str(pose.get("name") or pose_id)
        if needle and needle not in name.lower():
            continue
        shown += 1
        imgui.push_id(f"asset-pose-{pose_id}")
        if pose_id == editing:
            widgets.text_colored(theme.ACCENT, name)
        else:
            imgui.text(name)
        widgets.same_line_or_wrap(widgets.button_width("Apply"))
        if controls.small_button("Apply"):
            poser_mode.apply_asset_pose(ctx, pose_id)
        widgets.same_line_or_wrap(widgets.button_width("Delete"))
        if controls.small_button("Delete"):
            poser_mode.delete_asset_pose(ctx, pose_id, name)
        imgui.pop_id()
    widgets.no_matches(needle, shown)


def _library(ctx: Any, state: Any) -> None:
    widgets.section("Library")
    if not state.poses:
        widgets.empty_state(
            icons.PERSON_STANDING,
            "No poses yet",
            "Rotate a joint, then Save as. Poses here apply to every asset "
            "on this skeleton.",
        )
        return
    viewer = poser_mode.viewer_of(ctx)
    editing = None if viewer is None else viewer.editor.current
    needle = widgets.list_filter(ctx, "poser-library", len(state.poses))
    shown = 0
    for pose in state.poses:
        pose_id = str(pose.get("id") or "")
        name = str(pose.get("name") or pose_id)
        if needle and needle not in name.lower():
            continue
        shown += 1
        imgui.push_id(pose_id)
        if pose_id == editing:
            widgets.text_colored(theme.ACCENT, name)
        else:
            imgui.text(name)
        # Wrapped, not chained: the row starts with a user-typed pose name and
        # then asks for four buttons after it, in a sidebar. A long name pushed
        # Duplicate and Delete past the content edge, where imgui clips them --
        # so the only way to remove a pose from the shared library was to
        # rename it shorter first.
        widgets.same_line_or_wrap(widgets.button_width("Apply"))
        if controls.small_button("Apply"):
            poser_mode.apply_pose(ctx, pose_id)
        widgets.same_line_or_wrap(widgets.button_width("Rename"))
        if controls.small_button("Rename"):
            poser_mode.rename(ctx, pose_id)
        widgets.same_line_or_wrap(widgets.button_width("Duplicate"))
        if controls.small_button("Duplicate"):
            poser_mode.duplicate(ctx, pose_id)
        widgets.same_line_or_wrap(widgets.button_width("Delete"))
        if controls.small_button("Delete"):
            poser_mode.delete(ctx, pose_id)
        imgui.pop_id()
    widgets.no_matches(needle, shown)


def _presets(ctx: Any, state: Any) -> None:
    if not state.presets:
        return
    widgets.section("Shipped presets")
    # Read-only by design: a preset is a starting point, and apply-then-Save-as
    # is the promotion path into the library.
    # Wrapped: at the sidebar's width this clipped mid-word at the panel edge.
    widgets.muted_wrapped("Apply one, adjust it, then Save as to keep your version.")
    selected = str(ctx.state.preview.get("poser_preset") or "")
    if imgui.begin_table("poser-presets", 2):
        for preset in state.presets:
            name = str(preset.get("name") or "")
            imgui.push_id(f"preset-{name}")
            imgui.table_next_column()
            if controls.selectable_row(
                f"preset-{name}", name, selected=name == selected
            ):
                selected = name
                ctx.state.preview["poser_preset"] = name
            imgui.table_next_column()
            if controls.small_button("Apply", role=controls.ButtonRole.GHOST):
                poser_mode.apply_preset(ctx, preset)
                selected = name
                ctx.state.preview["poser_preset"] = name
            imgui.pop_id()
        imgui.end_table()
