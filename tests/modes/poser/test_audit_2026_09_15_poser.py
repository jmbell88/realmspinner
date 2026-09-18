"""Regression tests for the 2026-09-15 audit's Poser findings.

Four rows, four modules:

- poser-01 (``studio/poser_mode.py``): ``capture_key`` wrote the armature's
  pose into the selected key but never cleared ``editor.dirty``/``moved``,
  unlike every sibling mutator that folds a live edit into stored data
  (``apply_key`` is the clearest example) -- so the pending dot and "unsaved
  changes" guard kept firing after a capture that had, in fact, already
  saved the edit.
- poser-02 (``service/rig.py``): ``adjust_joints`` and ``edit_skeleton``
  never called ``rig_in_flight`` before minting a fresh rig job, even though
  service-03 (the 2026-09-14 audit) moved that check into ``create_rig`` on
  the claim that "every caller is covered" -- these two doors mint their own
  rig rows and were never on that list, so the Joints pane's "Apply joint
  positions" (submit key ``joints:<id>``, unshared with ``create_rig``'s
  callers) could queue a second rig for a mesh that already had one running.
- poser-03 (``cliptransfer.py``): an explicit ``clip_name`` was handed to
  every sampled action unchanged, so a multi-action source file produced
  several clips sharing one name -- ``import_into_library``'s own collision
  check then either misreported a name clash the caller never asked for, or,
  with ``replace=True``, silently kept only the last action.
- poser-04 (``clipmaps.py``): ``_resolve``'s ``by_normal.setdefault`` kept
  only the first source bone for a normalized name and dropped every other
  one with nothing recorded in ``MatchResult``.
"""

from __future__ import annotations

import json

import pytest

from warlock import clipmaps, cliptransfer
from warlock.kernels.geom3d import math3d as m3
from warlock.kernels.geom3d.gltf import Model, Node
from warlock.kernels.rig import skeleton, templates
from warlock.service import Conflict
from warlock.service import jobs as svc_jobs
from warlock.service import rig as svc_rig
from warlock.studio.modes.poser import mode as poser_mode
from warlock.studio.viewer.pose import PoseEditor

# --- poser-01: capture_key must clear dirty/moved like every sibling -------


class _FakeViewer:
    """Just the surface ``capture_key``/``select_key`` touch, over a real
    ``PoseEditor`` -- ``tests/modes/poser/test_poser_mode.py``'s own ``FakeViewer``,
    trimmed to the pose-mode-only path this test needs (no camera, no GLB
    loading)."""

    def __init__(self, model: Model, bones: list[str]) -> None:
        self.editor = PoseEditor()
        self.editor.bind(model, bones)
        self.editor.root = "hips"
        self.pose_mode = True
        self.onion: list = []


class _FakeCtx:
    def __init__(self) -> None:
        from types import SimpleNamespace

        self.state = SimpleNamespace(poser=None)
        self.poser_viewer: _FakeViewer | None = None
        self.toasts: list = []

    def toast(self, message, level="info", *_a) -> None:
        self.toasts.append((message, level))


def _armature_model() -> Model:
    names = [b["name"] for b in templates.get_template("humanoid").bones]
    nodes = [Node(name="rig", children=list(range(1, len(names) + 1)))]
    for i, name in enumerate(names):
        nodes.append(Node(name=name, translation=m3.vec3(0.0, 0.1 * i, 0.0)))
    return Model(nodes, roots=[0], meshes=[], skins=[])


def _clip_library() -> dict:
    return {
        "template": "humanoid",
        "space": "delta",
        "edited": False,
        "poses": [
            {"name": "A", "bones": {"hips": [0.0, 0.0, 0.0, 1.0]}},
            {"name": "B", "bones": {"spine": [0.0, 0.0, 0.0, 1.0]}},
            {"name": "C", "bones": {"head": [0.0, 0.0, 0.0, 1.0]}},
        ],
        "clips": [
            {
                "name": "walk",
                "keys": ["A", "B", "C"],
                "segments": [2, 2, 2],
                "closed": True,
                "easing": "linear",
                "space": "delta",
                "duration_ms": 100,
            }
        ],
    }


def _clip_ctx() -> tuple[_FakeCtx, poser_mode.PoserState]:
    ctx = _FakeCtx()
    bones = [b["name"] for b in templates.get_template("humanoid").bones]
    ctx.poser_viewer = _FakeViewer(_armature_model(), bones)
    state = poser_mode.ensure(ctx)
    poser_mode.adopt_clips(ctx, _clip_library())
    return ctx, state


def test_capture_key_clears_the_editors_unsaved_pose_flag():
    """poser-01: before the fix, ``capture_key`` wrote the pose into the
    key but left ``editor.dirty``/``moved`` set, so ``has_unsaved_edits()``
    kept reporting the just-saved pose as unsaved -- the next guarded action
    (selecting another key, closing the asset) asked to discard a change
    that had already landed."""
    ctx, state = _clip_ctx()
    editor = ctx.poser_viewer.editor
    poser_mode.select_key(ctx, 1)
    editor.apply({"spine": [0.0, 0.3894183, 0.0, 0.9210610]}, dirty=True)
    # ``moved`` is ordinarily populated by a joints-mode drag, not a pose
    # edit -- set by hand here so the assertion covers both halves of the
    # flag the fix clears, exactly as ``apply_key``'s own precedent does.
    editor.moved["hips"] = [0.01, 0.0, 0.0]
    assert editor.has_unsaved_edits()

    poser_mode.capture_key(ctx)

    assert state.key_pose("B")["bones"]["spine"] == pytest.approx(
        [0.0, 0.3894183, 0.0, 0.9210610]
    )
    assert editor.dirty is False
    assert editor.moved == {}
    assert not editor.has_unsaved_edits()


# --- poser-02: adjust_joints/edit_skeleton must also refuse an in-flight rig


def _finished_mesh_job(svc, assets) -> str:
    job_id = svc_jobs.create_job(svc, kind="text", prompt="a barrel")["id"]
    job_dir = assets / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"fake-glb")
    svc.store.set_status(job_id, "done")
    return job_id


def _rigged_job(svc, assets) -> tuple[str, list[dict]]:
    job_id = _finished_mesh_job(svc, assets)
    job_dir = assets / job_id
    template = templates.get_template("humanoid")
    fitted = skeleton.fit_template(template, [-1, -1, 0], [1, 1, 2])
    (job_dir / "rig.json").write_text(
        json.dumps(
            {
                "template": "humanoid",
                "bones": fitted,
                "bounds": {"min": [-1, -1, 0], "max": [1, 1, 2]},
            }
        ),
        encoding="utf-8",
    )
    (job_dir / "rig.glb").write_bytes(b"fake-rig")
    return job_id, fitted


@pytest.fixture
def assets(svc):
    return svc.config.data_dir


def test_adjust_joints_refuses_while_another_rig_job_is_already_in_flight(svc, assets):
    job_id, fitted = _rigged_job(svc, assets)
    bones = [{"name": b["name"], "head": b["head"], "tail": b["tail"]} for b in fitted]
    first = svc_rig.adjust_joints(svc, job_id, {"bones": bones})
    assert svc_rig.rig_in_flight(svc, job_id) == first["id"]

    with pytest.raises(Conflict) as caught:
        svc_rig.adjust_joints(svc, job_id, {"bones": bones})
    assert caught.value.field == "job_id"
    # Refused at the door: no second rig row was written.
    assert sorted(row["kind"] for row in svc.store.list()) == ["rig", "text"]


def test_edit_skeleton_refuses_while_another_rig_job_is_already_in_flight(svc, assets):
    """The same door, the same missing check, on ``edit_skeleton`` -- the
    finding names both."""
    job_id, fitted = _rigged_job(svc, assets)
    edited = skeleton.add_bone(fitted, "hips", "tail_01", [0, -0.1, 0.5], [0, -0.3, 0.5])
    first = svc_rig.edit_skeleton(svc, job_id, {"bones": edited})
    assert svc_rig.rig_in_flight(svc, job_id) == first["id"]

    with pytest.raises(Conflict) as caught:
        svc_rig.edit_skeleton(svc, job_id, {"bones": edited})
    assert caught.value.field == "job_id"
    assert sorted(row["kind"] for row in svc.store.list()) == ["rig", "text"]


# --- poser-03: an explicit clip_name across several sampled actions --------


def _target_sample() -> dict:
    template = templates.get_template("humanoid")
    bones = {}
    for b in template.bones:
        bones[b["name"]] = {
            "parent": b["parent"],
            "rest_rotation": [0.0, 0.0, 0.0, 1.0],
            "head": list(b["head"]),
            "tail": list(b["tail"]),
        }
    return {"template": "humanoid", "bones": bones}


_MIXAMO_SINGLE = {
    "hips": "Hips",
    "spine": "Spine",
    "chest": "Spine1",
    "neck": "Neck",
    "head": "Head",
    "shoulder.L": "LeftShoulder",
    "upper_arm.L": "LeftArm",
    "forearm.L": "LeftForeArm",
    "hand.L": "LeftHand",
    "shoulder.R": "RightShoulder",
    "upper_arm.R": "RightArm",
    "forearm.R": "RightForeArm",
    "hand.R": "RightHand",
    "thigh.L": "LeftUpLeg",
    "shin.L": "LeftLeg",
    "foot.L": "LeftFoot",
    "thigh.R": "RightUpLeg",
    "shin.R": "RightLeg",
    "foot.R": "RightFoot",
}


def _baseline_source_bones() -> dict:
    template = templates.get_template("humanoid")
    target = {b["name"]: b for b in template.bones}
    bones = {}
    for tname, sname in _MIXAMO_SINGLE.items():
        t = target[tname]
        bones[sname] = {
            "rest_rotation": [0.0, 0.0, 0.0, 1.0],
            "head": list(t["head"]),
            "tail": list(t["tail"]),
        }
    return bones


def _rest_action(name: str, source_bones: dict) -> dict:
    frame_bones = {
        sname: {"rotation": list(b["rest_rotation"]), "head": list(b["head"])}
        for sname, b in source_bones.items()
    }
    return {
        "name": name,
        "fps": 30.0,
        "frame_start": 0,
        "frame_end": 0,
        "frames": [{"frame": 0, "bones": frame_bones}],
    }


def test_an_explicit_clip_name_collides_silently_across_multiple_sampled_actions():
    """poser-03: before the fix, ``clip_name="walk"`` handed to a two-action
    sample produced two clips both named "walk" -- ``import_into_library``
    would then raise "a clip named \"walk\" already exists" against the
    caller's own explicit request, or silently drop the first action if
    ``replace=True``. Refused here instead, before either action is even
    converted."""
    source_bones = _baseline_source_bones()
    sample = {
        "source_bones": source_bones,
        "all_bone_names": list(source_bones),
        "actions": [
            _rest_action("Idle", source_bones),
            _rest_action("Walk", source_bones),
        ],
        "target": _target_sample(),
    }
    with pytest.raises(cliptransfer.ClipTransferError) as caught:
        cliptransfer.transfer(
            sample, template="humanoid", clip_name="walk", frames=2, root_motion="none"
        )
    assert caught.value.field == "clip_name"

    # A single-action sample is exactly what ``clip_name`` is for, and must
    # still work.
    single = {
        "source_bones": source_bones,
        "all_bone_names": list(source_bones),
        "actions": [_rest_action("Idle", source_bones)],
        "target": _target_sample(),
    }
    [result] = cliptransfer.transfer(
        single, template="humanoid", clip_name="walk", frames=2, root_motion="none"
    )
    assert result["clip"]["name"] == "walk"


# --- poser-04: a duplicate normalized source bone name must be reported ----


def test_clip_map_match_does_not_silently_drop_a_duplicate_normalized_source_bone_name():
    """poser-04: before the fix, ``_resolve``'s bare ``setdefault`` kept only
    the first of two source bones that normalize to the same name and threw
    the second away with no trace in ``MatchResult`` -- a hand-renamed
    duplicate (or a second Mixamo export merged into one skeleton) vanished
    from the match silently."""
    clip_map = clipmaps.load_clip_maps()["mixamo"]
    source = list(_MIXAMO_SINGLE.values()) + ["Spine2"]
    # A duplicate of "Hips" under a different raw spelling that normalizes
    # (via the mixamo map's own strip pattern) to the same name as "Hips"
    # itself.
    assert clipmaps.normalise("mixamorig:Hips", clip_map) == clipmaps.normalise(
        "Hips", clip_map
    )
    source_with_dupe = source + ["mixamorig:Hips"]

    result = clipmaps.match(source_with_dupe, template="humanoid")

    normalized = clipmaps.normalise("Hips", clip_map)
    assert normalized in result.duplicate_source_names
    assert result.duplicate_source_names[normalized] == ("Hips", "mixamorig:Hips")
