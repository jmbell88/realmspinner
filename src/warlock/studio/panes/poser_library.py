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

# The 2026-09-14 audit's poser-03: a skeleton draft holds the armature at
# rest for the whole of a skeleton-editing session (enter_skeleton_edit resets
# it on the way in), so New pose and every Apply button here would repose the
# mesh out from under the editor still assuming that rest -- greyed with this
# reason, and poser_mode's own doors refuse by name for whatever still
# reaches them past a disabled button (a keyboard shortcut, an agent's call).
_SKELETON_EDIT_REASON = "Apply or cancel the skeleton edit before changing the pose."


def draw(ctx: Any) -> None:
    state = poser_mode.ensure(ctx)
    # The pane is Poser's per-frame heartbeat, so the refresh flag is pumped
    # here -- the findings_dirty idiom; ``refresh`` only raises the flag.
    poser_mode.pump(ctx)
    # Same idiom, for the re-rig this pane may have queued: the write lands
    # minutes later, out of process, and nothing else here is told when.
    poser_mode.pump_rerig(ctx)
    widgets.section("Pose library")
    manual_render.help_button(ctx, "poser-library")
    if not ctx.rigging_available:
        # The pose_panel wording, verbatim: one sentence for one fact.
        widgets.muted("Posing needs Blender, which is not installed.")
        return

    _rigged_assets(ctx, state)

    if state.job_id:
        # The skeleton follows the bound asset's own rig -- shown as a fact,
        # not a combo, because changing it here does not edit anything in
        # place. It queues a *new* rig job, the same as the Library's own Rig
        # action, which is what ``_rerig`` below offers directly from the
        # session that already has the asset open (the 2026-09-07 finding:
        # this used to be a dead end, closing the session and finding the
        # source job in the Library the only way back to a different
        # skeleton).
        label = next(
            (e["label"] for e in rigging.catalog() if e["key"] == state.template),
            state.template,
        )
        widgets.field_label("Skeleton")
        widgets.muted(f"{label} (from this asset's rig)")
        _rerig(ctx, state)
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
    if controls.button(
        "New pose",
        (-1, 0),
        enabled=not state.skeleton_editing,
        reason=_SKELETON_EDIT_REASON,
    ):
        poser_mode.new_pose(ctx)

    if state.job_id:
        _asset_poses(ctx, state)
    _library(ctx, state)
    _presets(ctx, state)


def _rigged_assets(ctx: Any, state: Any) -> None:
    """The mode's own door in: every rigged mesh, newest first, with the
    currently bound one marked. Drawn above the skeleton block, the same
    order the picker gates in -- which asset is open decides which skeleton's
    library the rest of the pane shows.

    The pane draws, the controller decides (this module's own docstring): a
    click calls :func:`poser_mode.open_asset` and nothing here re-asks the
    dirty-editor guard or the template-switch discard confirm that function
    already carries.

    **A section, not a collapsing header**, like the three lists below it.
    This shipped as a ``widgets.header`` and
    ``tests/test_ux_consistency_pass2.py`` caught it by name: the 2026-09-05
    consistency pass unfolded Poser's panes out from under collapsing headers
    and pins that none of its three pane modules may reintroduce one. The
    reason survives the pin -- this sidebar is a column of peer lists (this
    asset's poses, the library, the shipped presets), and a fourth that alone
    could be folded away would be the odd one out in the one place a user
    looks for "what can I open".
    """
    widgets.section("Rigged assets")
    assets = poser_mode.riggable_assets(ctx)
    if not assets:
        # No button here on purpose (docs/INVARIANTS.md's "one empty-state
        # vocabulary" exempts a hint that points at a control worked
        # elsewhere): the control that fixes this is Create's Rig stage, not
        # anything this pane owns.
        widgets.empty_state(
            icons.PERSON_STANDING,
            "Nothing rigged yet",
            "Rig a mesh from Create's Rig stage, then it appears here.",
        )
        imgui.dummy((0, sp(tokens.SP_1)))
        return
    needle = widgets.list_filter(ctx, "poser-rigged-assets", len(assets))
    shown = 0
    for asset in assets:
        asset_id = str(asset.get("id") or "")
        # ``open_asset``'s own preference order (``poser_mode.riggable_assets``'
        # docstring): ``name`` and ``prompt`` are carried raw on this row, not
        # pre-merged, so the label shown here must apply the same order it
        # applies rather than silently reading ``prompt`` alone.
        name = str(asset.get("name") or asset.get("prompt") or asset_id)
        if needle and needle not in name.lower():
            continue
        shown += 1
        imgui.push_id(f"rigged-asset-{asset_id}")
        if controls.selectable_row(
            f"rigged-asset-{asset_id}", name, selected=asset_id == state.job_id
        ):
            _pick(ctx, asset)
        imgui.pop_id()
    widgets.no_matches(needle, shown)
    imgui.dummy((0, sp(tokens.SP_1)))


def _pick(ctx: Any, asset: dict[str, Any]) -> None:
    """One row's click. Named rather than inlined so it is callable with no
    imgui frame at all -- ``tests/test_poser_mode.py``'s own idiom for a
    pane's click, proving the row a click reaches, not merely that the
    button exists.

    Hands off whole: ``open_asset`` already carries the dirty-editor guard
    and the template-switch discard confirm, and this module's own docstring
    is "the pane draws, the controller decides" -- reimplementing either
    guard here would be the second copy that rule exists to prevent.
    """
    poser_mode.open_asset(ctx, asset)


def _rerig(ctx: Any, state: Any) -> None:
    """The "Re-rig..." control under the asset-bound skeleton fact.

    Drawn only from :func:`draw`'s ``state.job_id`` branch, which itself only
    runs once the pane's own top-of-draw refusal ("Posing needs Blender,
    which is not installed.") has already returned -- so this never has to
    ask the question again or gate itself a second, silent way. Collapsed by
    default: a picker sitting open under a fact nobody asked to change would
    read as the combo the fact just replaced.

    The open flag and the chosen key live on ``PoserState`` rather than in
    ``ctx.state.preview``, which is where the rest of the app keeps this kind
    of pane scratch -- see those fields' own note: that dict outlives the
    session, and this picker must not.
    """
    if not state.rerig_open:
        if controls.button("Re-rig...", (-1, 0)):
            # Defaults to the asset's own template, not the last thing picked
            # in the unbound browser -- reopening the picker after a change of
            # mind should not silently default to switching skeletons.
            state.rerig_open = True
            state.rerig_choice = state.template
        return
    state.rerig_choice = widgets.combo(
        "##poser-rerig-template",
        state.rerig_choice or state.template,
        [(entry["key"], entry["label"]) for entry in rigging.catalog()],
    )
    # Both buttons full width, stacked: a bare ``disabled_button`` sizes itself
    # to its label, so Confirm and Cancel came out different widths under each
    # other and read as unrelated controls.
    key = f"{poser_mode.ASSET_RERIG_KEY_PREFIX}{state.job_id}"
    if widgets.disabled_button(
        "Confirm re-rig",
        not ctx.busy(key),
        (-1, 0),
        reason="Already re-rigging this asset.",
    ):
        poser_mode.rerig(ctx, state.rerig_choice or state.template)
        state.rerig_open = False
    if controls.button("Cancel", (-1, 0)):
        state.rerig_open = False


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
        if controls.small_button(
            "Apply", enabled=not state.skeleton_editing, reason=_SKELETON_EDIT_REASON
        ):
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
        if controls.small_button(
            "Apply", enabled=not state.skeleton_editing, reason=_SKELETON_EDIT_REASON
        ):
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
            if controls.small_button(
                "Apply",
                role=controls.ButtonRole.GHOST,
                enabled=not state.skeleton_editing,
                reason=_SKELETON_EDIT_REASON,
            ):
                poser_mode.apply_preset(ctx, preset)
                selected = name
                ctx.state.preview["poser_preset"] = name
            imgui.pop_id()
        imgui.end_table()
