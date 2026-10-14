"""The 2026-10-14 audit, fixer F56: Clay's palette rows and viewport hint line.

shell-chrome-06 (the rig rows' reasons) and clay-view-03 (a selected bone and the
measure line).
"""

from __future__ import annotations

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio import viewport_hints
from realmspinner.studio.modes.clay.ui.panes import header as clay_header

from .test_clay_palette import _clay, _ctx


def _with_bone(ctx):
    ctx.rigging_available = True
    tab = ctx.state.clay.active
    tab.doc.add_bone()
    return tab


def test_the_weight_with_blender_row_names_the_real_reason_it_is_shut():
    ctx = _ctx()
    tab = _with_bone(ctx)
    assert _clay(ctx)["clay-rig:weights"].enabled(ctx)

    tab.pending_rig = {"reading": False}
    row = _clay(ctx)["clay-rig:weights"]
    assert not row.enabled(ctx)
    assert row.why == "Blender is already weighting this document."

    tab.pending_rig = None
    tab.saving = True
    row = _clay(ctx)["clay-rig:weights"]
    assert not row.enabled(ctx)
    assert row.why == "Saving..."

    tab.saving = False
    ctx.rigging_available = False
    row = _clay(ctx)["clay-rig:weights"]
    assert row.why == "Weighting needs Blender, which is not installed."


def test_the_import_animation_row_names_saving_and_not_the_missing_bones():
    ctx = _ctx()
    tab = _with_bone(ctx)
    assert _clay(ctx)["clay-file:import-animation"].enabled(ctx)

    tab.saving = True
    row = _clay(ctx)["clay-file:import-animation"]
    assert not row.enabled(ctx)
    assert row.why == clay_header.SAVING_WHY

    tab.saving = False
    # The bones are what it is waiting on only when there are none.
    for obj in [o for o in tab.doc.objects if o.bone]:
        tab.doc.remove_object(obj.uid)
    row = _clay(ctx)["clay-file:import-animation"]
    assert not row.enabled(ctx)
    assert row.why == "Mark some shapes as bones first."


def test_the_measure_line_ignores_a_selected_bone_so_the_pose_hint_shows():
    doc = bd.ClayDoc()
    box = bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box())
    doc.add_object(box)
    bone = doc.add_bone()

    doc.select([bone.uid])
    assert viewport_hints.measure_line(doc) == ""
    # So the legend wins: ``resolve_hint`` prefers a measure only when there is one.
    assert viewport_hints.resolve_hint(measure="", default="G/R/S pose it") == "G/R/S pose it"

    # A closed solid beside the bone still measures, and the bone does not spoil it.
    doc.select([box.uid, bone.uid])
    line = viewport_hints.measure_line(doc)
    assert line.startswith("volume") and "open mesh" not in line
