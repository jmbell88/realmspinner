"""``op_clip_sample`` -- the Blender half of "Import clip".

**Never called "retarget"** here -- see ``clipmaps``'s module docstring for why
that word is reserved for triangle-budget re-optimisation in this codebase.
"Import clip" brings a Mixamo/Rigify animation onto a Warlock template rig;
this file is Blender SAMPLES only -- it proves the worker reads an external
file's world bone transforms correctly, picks the right armature, and reports
the Warlock template's own rest frames the same way ``op_armature`` builds
them. The pure math that converts one onto the other is a later step
(``cliptransfer.py``) and is not exercised here.

The fixture below is a synthetic Mixamo-named armature -- built, keyframed and
exported with Blender's own operators, so this file never needs to ship a
binary FBX/GLB test asset. Its object carries a non-trivial transform (a
uniform 0.01 scale plus a +90 degree X rotation) precisely because a real
Mixamo download's armature node is never at identity, and ``op_clip_sample``
must read *world* space, not the file's own local bone space, to agree with
itself across formats.

Run with: uv run pytest tests/test_clip_import_blender.py -n 0
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

#: Three tests here (the ascii-FBX, no-skeleton and mirrored-armature cases)
#: build their own scene with no shared fixture, so nothing upstream of them
#: already skips on a machine with no bpy -- without this they ERROR instead
#: of SKIP. A module-level skip rather than one per test, consistent with
#: ``tests/test_rig_supplied_mesh.py``'s ``pytest.importorskip("bpy")``.
bpy = pytest.importorskip("bpy")

pytestmark = pytest.mark.timeout(600)

#: One side's worth of the mapped Mixamo chain, plus one bone no shipped clip
#: map claims -- ``LeftHandIndex1`` -- so "listed but not sampled" has
#: something to point at. ``(name, parent, head, tail)``, all in the
#: armature's own local space (roughly human-proportioned centimetres; the
#: object's own 0.01 scale turns this back into metres on import).
_BONES: list[tuple[str, str | None, tuple[float, float, float], tuple[float, float, float]]] = [
    ("Hips", None, (0, 0, 100), (0, 0, 110)),
    ("Spine", "Hips", (0, 0, 110), (0, 0, 120)),
    ("Spine1", "Spine", (0, 0, 120), (0, 0, 130)),
    ("Spine2", "Spine1", (0, 0, 130), (0, 0, 140)),
    ("Neck", "Spine2", (0, 0, 140), (0, 0, 148)),
    ("Head", "Neck", (0, 0, 148), (0, 0, 165)),
    ("LeftShoulder", "Spine2", (5, 0, 138), (15, 0, 138)),
    ("LeftArm", "LeftShoulder", (15, 0, 138), (35, 0, 138)),
    ("LeftForeArm", "LeftArm", (35, 0, 138), (55, 0, 138)),
    ("LeftHand", "LeftForeArm", (55, 0, 138), (65, 0, 138)),
    ("LeftHandIndex1", "LeftHand", (65, 0, 138), (70, 0, 138)),  # unmapped
    ("RightShoulder", "Spine2", (-5, 0, 138), (-15, 0, 138)),
    ("RightArm", "RightShoulder", (-15, 0, 138), (-35, 0, 138)),
    ("RightForeArm", "RightArm", (-35, 0, 138), (-55, 0, 138)),
    ("RightHand", "RightForeArm", (-55, 0, 138), (-65, 0, 138)),
    ("LeftUpLeg", "Hips", (10, 0, 100), (10, 0, 55)),
    ("LeftLeg", "LeftUpLeg", (10, 0, 55), (10, 0, 10)),
    ("LeftFoot", "LeftLeg", (10, 0, 10), (10, 15, 2)),
    ("RightUpLeg", "Hips", (-10, 0, 100), (-10, 0, 55)),
    ("RightLeg", "RightUpLeg", (-10, 0, 55), (-10, 0, 10)),
    ("RightFoot", "RightLeg", (-10, 0, 10), (-10, 15, 2)),
]

#: A distinctive non-rest roll per bone family, so the rest quaternion this
#: file checks is not vacuously identity for every bone.
_ROLLS = {"LeftArm": 15.0, "RightArm": -15.0, "LeftLeg": -10.0, "RightLeg": 10.0, "Hips": 20.0}

#: (bone, axis index 0/1/2, degrees) at frame 2 and frame 3 -- distinct poses,
#: applied as a local pose-bone rotation about that axis. Knee bend, arm
#: raise, and (via ``_HIPS_LIFT``) a hips translation, as CLAUDE.md's plan asks.
_POSE_FRAME2 = {"LeftLeg": (0, -45.0), "RightArm": (2, 60.0)}
_POSE_FRAME3 = {"LeftLeg": (0, -20.0), "RightArm": (2, 30.0)}
_HIPS_LIFT_FRAME2 = (0.0, 0.0, 10.0)
_HIPS_LIFT_FRAME3 = (0.0, 0.0, 5.0)

MAPPED_NAMES = {name for name, *_ in _BONES if name != "LeftHandIndex1"}
ALL_NAMES = {name for name, *_ in _BONES}


def _axis_quat(axis: int, degrees: float):
    from mathutils import Quaternion

    direction = [0.0, 0.0, 0.0]
    direction[axis] = 1.0
    return Quaternion(direction, math.radians(degrees))


def _build_mixamo_rig(bpy):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    arm_data = bpy.data.armatures.new("MixamoRig")
    obj = bpy.data.objects.new("MixamoRig", arm_data)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    # A real Mixamo download's armature node is never at identity: this is
    # what op_clip_sample's `arm.matrix_world @ ...` reads through.
    obj.scale = (0.01, 0.01, 0.01)
    obj.rotation_euler = (math.radians(90.0), 0.0, 0.0)

    bpy.ops.object.mode_set(mode="EDIT")
    created = {}
    for name, _parent, head, tail in _BONES:
        eb = arm_data.edit_bones.new(f"mixamorig:{name}")
        eb.head = head
        eb.tail = tail
        eb.roll = math.radians(_ROLLS.get(name, 0.0))
        created[name] = eb
    for name, parent, *_ in _BONES:
        if parent is not None:
            created[name].parent = created[parent]
    bpy.ops.object.mode_set(mode="OBJECT")

    obj.animation_data_create()
    action = bpy.data.actions.new("MixamoTake")
    obj.animation_data.action = action
    for pbone in obj.pose.bones:
        pbone.rotation_mode = "QUATERNION"

    def _key(frame: int, poses: dict, hips_lift: tuple[float, float, float] | None) -> None:
        for pbone in obj.pose.bones:
            pbone.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        for bone_name, (axis, degrees) in poses.items():
            q = _axis_quat(axis, degrees)
            pbone = obj.pose.bones[f"mixamorig:{bone_name}"]
            pbone.rotation_quaternion = (q.w, q.x, q.y, q.z)
        hips = obj.pose.bones["mixamorig:Hips"]
        hips.location = hips_lift or (0.0, 0.0, 0.0)
        for pbone in obj.pose.bones:
            pbone.keyframe_insert(data_path="rotation_quaternion", frame=frame)
        hips.keyframe_insert(data_path="location", frame=frame)

    _key(1, {}, None)
    _key(2, _POSE_FRAME2, _HIPS_LIFT_FRAME2)
    _key(3, _POSE_FRAME3, _HIPS_LIFT_FRAME3)
    if action.slots:
        obj.animation_data.action_slot = action.slots[0]
    action.use_fake_user = True

    # FBX bakes whatever the *current* pose is as the file's bind pose (unlike
    # the glTF exporter's rest-position flag) -- return to true rest first, or
    # the exported "rest" frame would be whatever frame the timeline was left
    # on, not the T-pose these tests assume.
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.select_all(action="SELECT")
    bpy.ops.pose.transforms_clear()
    bpy.ops.object.mode_set(mode="OBJECT")
    return obj


@pytest.fixture(scope="module")
def clip_paths(tmp_path_factory):
    """The same synthetic Mixamo rig, exported once as both FBX and GLB."""
    pytest.importorskip("bpy")
    import bpy

    tmp = tmp_path_factory.mktemp("clip_import")
    _build_mixamo_rig(bpy)
    out_fbx = tmp / "mixamo.fbx"
    out_glb = tmp / "mixamo.glb"
    bpy.ops.export_scene.fbx(
        filepath=str(out_fbx),
        use_selection=False,
        bake_anim=True,
        add_leaf_bones=False,
        axis_forward="-Z",
        axis_up="Y",
    )
    bpy.ops.export_scene.gltf(
        filepath=str(out_glb), export_format="GLB", use_selection=False, export_animations=True
    )
    assert out_fbx.is_file() and out_glb.is_file()
    return {"fbx": out_fbx, "glb": out_glb}


def _spec(source: Path, tmp_path: Path, **overrides) -> dict:
    from warlock.kernels.rig import blender_spec

    spec = blender_spec.clip_sample_spec(source, "humanoid", tmp_path / ".clip_result.json")
    spec.pop("result_path", None)  # op_clip_sample never reads it; only run_worker/main do
    spec.update(overrides)
    return spec


def _run(spec: dict) -> dict:
    import bpy

    from warlock.pipelines import blender_worker as bw

    return bw.op_clip_sample(bpy, spec)


def _quat_angle_deg(a, b) -> float:
    """Angle between two XYZW quaternions, degrees, treating q and -q as equal."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    dot = max(-1.0, min(1.0, abs(dot)))
    return math.degrees(2.0 * math.acos(dot))


def test_clip_sample_is_a_registered_op():
    from warlock.pipelines import blender_worker as bw

    assert bw.OPS["clip_sample"] is bw.op_clip_sample


def test_clip_sample_reads_fbx_and_glb_to_the_same_world_rotations(clip_paths, tmp_path):
    fbx_result = _run(_spec(clip_paths["fbx"], tmp_path))
    glb_result = _run(_spec(clip_paths["glb"], tmp_path))
    assert fbx_result["ok"] is True, fbx_result
    assert glb_result["ok"] is True, glb_result

    assert len(fbx_result["actions"]) == 1
    assert len(glb_result["actions"]) == 1
    fbx_action = fbx_result["actions"][0]
    glb_action = glb_result["actions"][0]
    assert fbx_action["frame_start"] == glb_action["frame_start"] == 1
    assert fbx_action["frame_end"] == glb_action["frame_end"] == 3

    mapped = {f"mixamorig:{n}" for n in MAPPED_NAMES}
    fbx_frames = {f["frame"]: f["bones"] for f in fbx_action["frames"]}
    glb_frames = {f["frame"]: f["bones"] for f in glb_action["frames"]}
    checked = 0
    for frame in (1, 2, 3):
        fbx_bones = fbx_frames[frame]
        glb_bones = glb_frames[frame]
        assert set(fbx_bones) == mapped == set(glb_bones)
        for name in mapped:
            angle = _quat_angle_deg(fbx_bones[name]["rotation"], glb_bones[name]["rotation"])
            assert angle < 0.5, f"frame {frame} bone {name}: {angle:.3f} deg apart"
            fbx_head = fbx_bones[name]["head"]
            glb_head = glb_bones[name]["head"]
            assert fbx_head == pytest.approx(glb_head, abs=1e-3), f"frame {frame} bone {name}"
            checked += 1
    assert checked == len(mapped) * 3


def test_clip_sample_reports_the_warlock_target_rest_frames_from_the_armature_builder(
    clip_paths, tmp_path
):
    import bpy

    from warlock import poselib
    from warlock.kernels.rig import skeleton, templates
    from warlock.pipelines import blender_worker as bw

    result = _run(_spec(clip_paths["glb"], tmp_path))
    assert result["ok"] is True, result
    assert result["target"]["template"] == "humanoid"

    template = templates.get_template("humanoid")
    expected_names = {b["name"] for b in template.bones}
    assert set(result["target"]["bones"]) == expected_names

    # Build the same reference the worker's op_armature preview builds, in a
    # scene of our own, and check the two constructions agree exactly -- the
    # whole point of sharing _build_armature is that there is one definition
    # of a bone's roll, not two that can drift apart.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    fitted = skeleton.fit_template(template, poselib.UNIT_LO, poselib.UNIT_HI)
    reference_arm = bw._build_armature(bpy, fitted)
    for bone in reference_arm.data.bones:
        got = result["target"]["bones"][bone.name]
        assert got["parent"] == (bone.parent.name if bone.parent is not None else None)
        quat = bone.matrix_local.to_quaternion()
        assert got["rest_rotation"] == pytest.approx([quat.x, quat.y, quat.z, quat.w], abs=1e-9)
        assert got["head"] == pytest.approx(list(bone.head_local), abs=1e-9)
        assert got["tail"] == pytest.approx(list(bone.tail_local), abs=1e-9)


def test_unmapped_source_bones_are_listed_but_not_sampled(clip_paths, tmp_path):
    result = _run(_spec(clip_paths["glb"], tmp_path))
    assert result["ok"] is True, result

    unmapped = "mixamorig:LeftHandIndex1"
    assert unmapped in result["all_bone_names"]
    assert unmapped not in result["source_bones"]
    for action in result["actions"]:
        for frame in action["frames"]:
            assert unmapped not in frame["bones"]
    # Every mapped bone, meanwhile, is both listed and sampled.
    mapped = {f"mixamorig:{n}" for n in MAPPED_NAMES}
    assert mapped <= set(result["all_bone_names"])
    assert set(result["source_bones"]) == mapped


def test_an_ascii_fbx_is_a_sentence_not_a_crash(tmp_path):
    ascii_fbx = tmp_path / "ascii.fbx"
    ascii_fbx.write_text(
        "; FBX 7.4.0 project file\n"
        "; ----------------------------------------------------\n\n"
        "FBXHeaderExtension:  {\n"
        "\tFBXHeaderVersion: 1003\n"
        "\tFBXVersion: 7400\n"
        "}\n",
        encoding="ascii",
    )
    result = _run(_spec(ascii_fbx, tmp_path))
    assert result["ok"] is False
    assert isinstance(result["error"], str) and result["error"]
    assert "ascii" in result["error"].lower() or "import" in result["error"].lower()


def test_a_file_with_no_known_skeleton_is_a_sentence(tmp_path):
    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)
    arm_data = bpy.data.armatures.new("GenericRig")
    obj = bpy.data.objects.new("GenericRig", arm_data)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    a = arm_data.edit_bones.new("Bone")
    a.head, a.tail = (0, 0, 0), (0, 0, 1)
    b = arm_data.edit_bones.new("Bone.001")
    b.head, b.tail = (0, 0, 1), (0, 0, 2)
    b.parent = a
    bpy.ops.object.mode_set(mode="OBJECT")

    out_glb = tmp_path / "generic.glb"
    bpy.ops.export_scene.gltf(filepath=str(out_glb), export_format="GLB", use_selection=False)

    result = _run(_spec(out_glb, tmp_path))
    assert result["ok"] is False
    assert "skeleton" in result["error"].lower() or "recognise" in result["error"].lower()


def test_a_mirrored_armature_is_refused(tmp_path):
    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)
    arm_data = bpy.data.armatures.new("MirroredRig")
    obj = bpy.data.objects.new("MirroredRig", arm_data)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.scale = (-0.01, 0.01, 0.01)  # negative determinant -> mirrored
    bpy.ops.object.mode_set(mode="EDIT")
    hips = arm_data.edit_bones.new("mixamorig:Hips")
    hips.head, hips.tail = (0, 0, 100), (0, 0, 110)
    spine = arm_data.edit_bones.new("mixamorig:Spine")
    spine.head, spine.tail = (0, 0, 110), (0, 0, 120)
    spine.parent = hips
    bpy.ops.object.mode_set(mode="OBJECT")
    assert obj.matrix_world.determinant() < 0

    out_glb = tmp_path / "mirrored.glb"
    bpy.ops.export_scene.gltf(filepath=str(out_glb), export_format="GLB", use_selection=False)

    result = _run(_spec(out_glb, tmp_path))
    assert result["ok"] is False
    assert "mirror" in result["error"].lower()
