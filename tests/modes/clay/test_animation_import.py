"""Import animation: a file's motion laid onto a document's bones, by name.

Tranche 3 (U4) of the 2026-10-13 Clay rig plan. The source here is always a real GLB written by
Clay's own exporter and read back by its own loader, so what is retargeted is what a file holds.
Each claim fails on the tree before the change.
"""

from __future__ import annotations

import numpy as np
import pytest

from realmspinner.kernels.geom3d import glbwrite, gltf
from realmspinner.kernels.mesh import anim, animimport
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh.elements import OpError
from realmspinner.studio.modes.clay import mode as clay_mode

from .test_clay_mode import FakeCtx

Q0 = np.array([0.0, 0.0, 0.0, 1.0])
Q90 = np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
Q180 = np.array([0.0, 0.0, 1.0, 0.0])
Q270 = np.array([0.0, 0.0, -np.sin(np.pi / 4), np.cos(np.pi / 4)])


def _bone(doc: bd.ClayDoc, name: str, **kw) -> int:
    return doc.add_object(bd.Obj(uid=bd.new_uid(), name=name, mesh=bp.box(), bone=True, **kw)).uid


def _source(names=("Hips", "LeftArm"), **rest) -> tuple[bd.ClayDoc, dict[str, int]]:
    """A rigged document with a clip: the first bone turns 90 degrees over a second, the second
    slides up a metre."""
    doc = bd.ClayDoc()
    first = _bone(doc, names[0], **rest.get("first", {}))
    second = _bone(doc, names[1], parent=first, **rest.get("second", {}))
    doc.add_animation("Wave")
    for time, rotation, lift in ((0.0, Q0, 0.0), (1.0, Q90, 1.0)):
        doc.set_rig_time(time)
        base = np.array(doc.by_uid(second).translation, dtype="f8")
        doc.key_pose(
            {first: {"rotation": rotation}, second: {"translation": base + [0.0, lift, 0.0]}}
        )
    return doc, {names[0]: first, names[1]: second}


def _model(doc: bd.ClayDoc) -> gltf.Model:
    return gltf.load(glbwrite.write_glb(bd.to_model(doc)))


def _target(names=("hips", "arm.L"), **rest) -> tuple[bd.ClayDoc, int, int]:
    doc = bd.ClayDoc()
    first = _bone(doc, names[0], **rest.get("first", {}))
    second = _bone(doc, names[1], parent=first, **rest.get("second", {}))
    return doc, first, second


# -- names ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "group",
    [
        ("LeftArm", "arm.L", "mixamorig:LeftArm", "Arm_L", "left_arm", "ARM-l", "arm L"),
        ("RightUpLeg", "upleg.R", "mixamorig1:RightUpLeg", "UpLeg_r"),
        ("Hips", "hips", "mixamorig:Hips", "Armature:Hips"),
        ("Spine_01", "spine01", "mixamorig:Spine01"),
    ],
)
def test_names_that_differ_only_in_case_separators_prefix_or_side_spelling_are_one_bone(group):
    assert len({animimport.canonical(name) for name in group}) == 1, group


def test_the_left_and_right_of_a_bone_and_different_bones_stay_apart():
    assert animimport.canonical("arm.L") != animimport.canonical("arm.R")
    assert animimport.canonical("Arm") != animimport.canonical("Forearm")
    assert animimport.canonical("Spine1") != animimport.canonical("Spine2")
    assert animimport.canonical("Left") == "left", "a lone word is not a side"


def test_two_bones_answering_to_one_name_are_left_alone_and_named():
    bones = {1: "arm.L", 2: "LeftArm", 3: "hips"}
    pairs, ambiguous = animimport.match_bones(bones, {10: "mixamorig:LeftArm", 11: "Hips"})
    assert pairs == {11: 3}
    assert ambiguous == ["mixamorig:LeftArm"]


def test_two_source_names_for_one_bone_are_ambiguous_too():
    pairs, ambiguous = animimport.match_bones({1: "hips"}, {10: "Hips", 11: "mixamorig:Hips"})
    assert pairs == {} and ambiguous == ["Hips", "mixamorig:Hips"]


# -- the retarget --------------------------------------------------------------------------------


def test_a_rig_built_the_same_way_gets_the_source_numbers_unchanged():
    source, _ = _source()
    target, first, second = _target()
    imported = animimport.import_by_name(target, _model(source))
    [clip] = imported.animations
    assert clip.name == "Wave"
    assert np.allclose(clip.channels[first]["rotation"].values, [Q0, Q90], atol=1e-6)
    lift = clip.channels[second]["translation"].values
    assert np.allclose(lift, [[0, 0, 0], [0, 1, 0]], atol=1e-6)
    assert imported.matched == {"Hips": "hips", "LeftArm": "arm.L"}


def test_a_rest_pose_that_differs_shifts_the_motion_instead_of_replacing_it():
    source, _ = _source()
    target, first, second = _target(
        first={"rotation": Q90}, second={"translation": (3.0, 0.0, 0.0)}
    )
    [clip] = animimport.import_by_name(target, _model(source)).animations
    # The source turned from rest by 0 then 90 degrees; this bone starts turned 90 already.
    assert np.allclose(clip.channels[first]["rotation"].values, [Q90, Q180], atol=1e-6)
    # And the source arm rose a metre from where it was: so does this one, from 3 m out.
    assert np.allclose(
        clip.channels[second]["translation"].values, [[3, 0, 0], [3, 1, 0]], atol=1e-6
    )


def test_a_source_rest_pose_is_taken_out_before_the_bones_own_goes_in():
    source, _ = _source(first={"rotation": Q90})
    target, first, _second = _target()
    [clip] = animimport.import_by_name(target, _model(source)).animations
    # The source's second key *is* its rest (turned 90), which relative to rest is no motion:
    # this bone stays at its own rest. Its first key is 90 degrees back from rest.
    assert np.allclose(clip.channels[first]["rotation"].values[1], Q0, atol=1e-6)
    assert np.allclose(clip.channels[first]["rotation"].values[0], Q270, atol=1e-6)


def test_size_keys_are_a_ratio_to_the_sources_rest_applied_to_the_bones():
    source = bd.ClayDoc()
    hips = _bone(source, "Hips", scale=(2.0, 2.0, 2.0))
    source.add_animation("Pulse")
    source.set_rig_time(1.0)
    source.key_pose({hips: {"scale": (4.0, 4.0, 4.0)}})
    target, first, _ = _target(first={"scale": (3.0, 3.0, 3.0)})
    [clip] = animimport.import_by_name(target, _model(source)).animations
    assert np.allclose(clip.channels[first]["scale"].values, [[6.0, 6.0, 6.0]])


def test_the_source_loop_flag_and_interpolation_come_across():
    source, _ = _source()
    source.set_animation_loop(0, False)
    source.set_animation_interpolation(0, "step")
    target, *_ = _target()
    [clip] = animimport.import_by_name(target, _model(source)).animations
    assert clip.loop is False and clip.interpolation == "step"


def test_every_animation_in_the_file_comes_in_and_names_are_made_unique():
    source, _ = _source()
    source.add_animation("Wave", copy_of=0)  # "Wave.001"
    target, *_ = _target()
    imported = animimport.import_by_name(target, _model(source))
    assert [a.name for a in imported.animations] == ["Wave", "Wave.001"]


# -- what it refuses and what it reports ---------------------------------------------------------


def test_a_file_with_no_animation_is_refused_by_that_sentence():
    plain = bd.ClayDoc()
    _bone(plain, "Hips")
    target, *_ = _target()
    with pytest.raises(OpError, match="no animation"):
        animimport.import_by_name(target, _model(plain))


def test_a_document_with_no_bones_is_told_to_mark_some():
    source, _ = _source()
    bare = bd.ClayDoc()
    bare.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
    with pytest.raises(OpError, match="Mark the joints as bones"):
        animimport.import_by_name(bare, _model(source))


def test_a_file_whose_nodes_match_no_bone_is_refused_naming_the_bones():
    source, _ = _source(names=("Torso", "Wing"))
    target, *_ = _target()
    with pytest.raises(OpError, match="hips"):
        animimport.import_by_name(target, _model(source))


def test_what_matched_and_what_did_not_are_both_reported():
    source, _ = _source(names=("Hips", "Tail"))
    target, *_ = _target()
    imported = animimport.import_by_name(target, _model(source))
    assert imported.matched == {"Hips": "hips"}
    assert imported.unmatched == ("Tail",)


# -- the document's door -------------------------------------------------------------------------


def test_importing_adds_after_what_is_there_selects_the_first_new_and_is_one_step():
    source, _ = _source()
    target, first, _ = _target()
    target.add_animation("Mine")
    target.key_pose({first: {"rotation": Q90}})
    steps = len(target.history)
    imported = animimport.import_by_name(target, _model(source))
    assert target.import_animations(imported.animations, imported.fps) == 1
    assert len(target.history) == steps + 1
    assert [a.name for a in target.rig_state.animations] == ["Mine", "Wave"]
    assert target.active_animation == 1
    target.undo()
    assert [a.name for a in target.rig_state.animations] == ["Mine"]


def test_a_clash_of_names_is_resolved_and_the_frame_rate_is_kept_when_clips_exist():
    source, _ = _source()
    target, first, _ = _target()
    target.add_animation("Wave")
    target.set_fps(60)
    imported = animimport.import_by_name(target, _model(source))
    target.import_animations(imported.animations, 12)
    assert [a.name for a in target.rig_state.animations] == ["Wave", "Wave.001"]
    assert target.rig_state.fps == 60, "a file's rate does not retime what is already authored"


def test_the_first_import_into_a_document_with_no_clips_adopts_the_files_rate():
    source, _ = _source()
    target, *_ = _target()
    imported = animimport.import_by_name(target, _model(source))
    target.import_animations(imported.animations, 12)
    assert target.rig_state.fps == 12


def test_the_cap_on_animations_is_a_refusal_that_adds_nothing():
    source, _ = _source()
    target, *_ = _target()
    imported = animimport.import_by_name(target, _model(source))
    for i in range(anim.MAX_ANIMATIONS):
        target.add_animation(f"A{i}")
    steps = len(target.history)
    with pytest.raises(OpError, match="at most"):
        target.import_animations(imported.animations)
    assert len(target.history) == steps and len(target.rig_state.animations) == anim.MAX_ANIMATIONS


def test_adding_nothing_pushes_nothing():
    target, *_ = _target()
    steps = len(target.history)
    assert target.import_animations([]) == 0 and len(target.history) == steps


# -- the flow ------------------------------------------------------------------------------------


def test_import_animation_reads_the_file_off_the_frame_thread_and_adds_the_clips_when_it_lands(
    tmp_path,
):
    source, _ = _source()
    path = tmp_path / "wave.glb"
    path.write_bytes(glbwrite.write_glb(bd.to_model(source)))
    ctx = FakeCtx()
    target, first, _ = _target()
    tab = clay_mode.adopt(ctx, target, title="Hero")

    clay_mode.import_animation(ctx, tab, path)
    assert tab.saving, "the bones' rest is read on the task: nothing may edit it meanwhile"
    assert target.rig_state.animations == (), "nothing is added until the task lands"
    key = ctx.submitted[-1]
    assert key == f"clay-animport:{tab.uid}"
    clay_mode.on_task_done(ctx, type("D", (), {"key": key, "result": ctx.result})())

    assert not tab.saving
    assert [a.name for a in target.rig_state.animations] == ["Wave"]
    assert any("Imported 1 animation from wave.glb" in m for m, _level in ctx.toasts)
    assert any("2 bones" in m for m, _level in ctx.toasts)


def test_a_document_with_no_bones_is_refused_before_anything_is_read(tmp_path):
    ctx = FakeCtx()
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
    tab = clay_mode.adopt(ctx, doc, title="Box")
    clay_mode.import_animation(ctx, tab, tmp_path / "nope.glb")
    assert not tab.saving and not ctx.submitted
    assert any("no bones" in m for m, _level in ctx.toasts)


def test_a_failed_import_unlocks_the_tab(tmp_path):
    ctx = FakeCtx()
    target, *_ = _target()
    tab = clay_mode.adopt(ctx, target, title="Hero")
    with pytest.raises(FileNotFoundError):  # the fake runs the task inline
        clay_mode.import_animation(ctx, tab, tmp_path / "missing.glb")
    key = ctx.submitted[-1]
    clay_mode.on_task_failed(ctx, type("D", (), {"key": key})())
    assert not tab.saving


# -- the doors -----------------------------------------------------------------------------------


def test_the_file_menu_offers_import_animation_greyed_until_there_are_bones(monkeypatch):
    from realmspinner.studio import menus

    from .test_clay_compact_layout import _clay_menu_ctx, _doc, _tab

    monkeypatch.setattr(clay_mode, "recent_paths", lambda ctx: [])
    plain, _uid = _doc()
    rows = {r.identity: r for r in menus._clay_file_specs(_clay_menu_ctx(_tab(plain)), [])}
    row = rows["clay:import-animation"]
    assert row.path == ("File",) and row.label == "Import Animation..."
    assert not row.enabled and "bones" in row.disabled_reason

    target, *_ = _target()
    rows = {r.identity: r for r in menus._clay_file_specs(_clay_menu_ctx(_tab(target)), [])}
    assert rows["clay:import-animation"].enabled

    none = {r.identity: r for r in menus._clay_file_specs(_clay_menu_ctx(None), [])}
    assert not none["clay:import-animation"].enabled


def test_the_palette_offers_import_animation_only_when_there_are_bones():
    from .test_clay_palette import _clay
    from .test_clay_palette import _ctx as palette_ctx

    ctx = palette_ctx()
    command = _clay(ctx)["clay-file:import-animation"]
    assert command.label == "Import Animation..."
    assert not command.enabled(ctx)
    ctx.state.clay.active.doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="Hips", mesh=bp.box(), bone=True)
    )
    assert command.enabled(ctx)
