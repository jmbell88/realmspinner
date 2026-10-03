"""The 2026-10-03 audit's poser-01..10, one regression each (named as the row names it)."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from realmspinner.characters.resolve import resolve
from realmspinner.kernels.rig import cliplib, templates
from realmspinner.kernels.rig import store as rig_store
from realmspinner.studio.modes.poser import mode as poser_mode
from realmspinner.studio.viewer.pose import PoseEditor

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_pm = _load(_HERE / "test_poser_mode.py", "_poser_mode_helpers_2026_10_03")
FakeCtx, FakeViewer = _pm.FakeCtx, _pm.FakeViewer
_armature_model, _full_bones = _pm._armature_model, _pm._full_bones


# --- poser-01 ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("prompt", "family", "not_family"),
    [
        ("a hero bearing a torch", None, "bear"),
        ("a guarded dwarf", "dwarf", "knight"),
        ("a cloaked figure guarding a gate", None, "knight"),
        ("shaded elf", "elf", "ghost"),
        ("a being of fire", None, "bee"),
    ],
)
def test_an_inflected_verb_never_resolves_to_a_species_it_merely_contains(
    prompt, family, not_family
):
    result = resolve(prompt)
    assert result.family != not_family
    if family is not None:
        assert result.family == family
    if family is None:
        assert not result.creature_words, "no species was named in this prompt"


def test_the_named_wolf_wins_over_ravening():
    assert resolve("a ravening wolf").family == "wolf"


def test_walking_and_flaming_still_lemmatise_for_actions_and_themes():
    result = resolve("a walking ogre")
    assert result.family == "ogre"
    assert "walk" in result.actions
    assert resolve("a flaming ogre").theme is not None


# --- poser-02 ----------------------------------------------------------------


def _skel_editor() -> PoseEditor:
    from realmspinner.kernels.geom3d import math3d as m3
    from realmspinner.kernels.geom3d.gltf import Model, Node

    nodes = [
        Node(name="hip", translation=m3.vec3(0.0, 0.0, 0.0)),
        Node(name="upper_arm.L", translation=m3.vec3(0.0, 1.0, 0.0)),
    ]
    nodes[0].children = [1]
    editor = PoseEditor()
    editor.bind(Model(nodes, roots=[0], meshes=[], skins=[]), ["hip", "upper_arm.L"])
    return editor


_RIG = {
    "bones": [
        {"name": "hip", "parent": None, "head": [0.0, 0.0, 0.0], "tail": [0.0, 0.0, 1.0]},
        {"name": "upper_arm.L", "parent": "hip", "head": [0.0, 0.0, 1.0], "tail": [0.5, 0.0, 1.0]},
    ],
    "root": "hip",
    "mirror_pairs": [],
}


def test_undo_never_crosses_the_skeleton_mode_boundary():
    editor = _skel_editor()
    editor.enter_skeleton_mode(_RIG)
    assert editor.mode == "skeleton"
    editor.undo()
    assert editor.mode == "skeleton", "Ctrl+Z must not undo the entry into skeleton mode"
    assert editor.draft, "the draft survives a stray Ctrl+Z"

    editor.exit_skeleton_mode()
    assert editor.mode == "pose"
    editor.undo()
    assert editor.mode == "pose", "Ctrl+Z after Cancel must not re-enter skeleton mode"


# --- poser-03 ----------------------------------------------------------------


def _asset_ctx(svc, *, job_id):
    ctx = FakeCtx(svc)
    state = poser_mode.ensure(ctx)
    state.job_id = job_id
    names = [b["name"] for b in templates.get_template("humanoid").bones]
    state.asset_rig = {
        "root": "hips",
        "bones": [{"name": n, "parent": None if n == "hips" else "hips"} for n in names],
    }
    viewer = ctx.poser_viewer = FakeViewer(_armature_model(), names)
    viewer.pose_job_id = job_id
    assert viewer.editor.root is None, "the fake binds like enter_pose_mode: no root"
    return ctx, state, viewer


def test_resaving_an_asset_pose_keeps_its_root_offset_in_a_real_asset_session(svc):
    job_id = _pm._rigged_job(
        svc, bones=[{"name": b["name"]} for b in templates.get_template("humanoid").bones]
    )
    ctx, state, viewer = _asset_ctx(svc, job_id=job_id)
    poser_mode._bind_asset_now(ctx, state, viewer, job_id)
    assert viewer.editor.root == "hips", "an asset session must bind editor.root"

    viewer.editor.set_root_translation([0.1, 0.0, 0.25], dirty=True)
    poser_mode.save_pose_to_asset(ctx)
    ctx.prompts.asked[-1].on_accept("Crouch")
    saved = ctx.results[f"{poser_mode.ASSET_SAVE_KEY_PREFIX}{job_id}"]
    assert saved["root_translation"] == pytest.approx([0.1, 0.0, 0.25])

    # Re-save it from a fresh bind of the same asset: apply, then save over it.
    state.asset_poses = [saved]
    viewer.editor.set_root_translation([0, 0, 0], dirty=False)
    poser_mode.apply_asset_pose(ctx, saved["id"])
    assert viewer.editor.root_translation() == pytest.approx([0.1, 0.0, 0.25])
    poser_mode.save_pose_to_asset(ctx)
    ctx.prompts.asked[-1].on_accept("Crouch")
    again = ctx.results[f"{poser_mode.ASSET_SAVE_KEY_PREFIX}{job_id}"]
    assert again["root_translation"] == pytest.approx([0.1, 0.0, 0.25])


# --- poser-04 ----------------------------------------------------------------


def test_save_to_asset_after_applying_a_library_pose_creates_a_new_asset_pose(svc):
    job_id = _pm._rigged_job(
        svc, bones=[{"name": b["name"]} for b in templates.get_template("humanoid").bones]
    )
    ctx = FakeCtx(svc)
    state = poser_mode.ensure(ctx)
    state.job_id = job_id
    viewer = ctx.poser_viewer = _pm._bound_viewer()
    viewer.pose_job_id = job_id
    library_id = "0123456789ab"
    state.poses = [{"id": library_id, "name": "Leap", "bones": _full_bones()}]
    poser_mode.apply_pose(ctx, library_id)
    assert viewer.editor.current == library_id
    state.asset_poses = []

    poser_mode.save_pose_to_asset(ctx)
    ctx.prompts.asked[-1].on_accept("Leap on this asset")
    saved = ctx.results[f"{poser_mode.ASSET_SAVE_KEY_PREFIX}{job_id}"]
    assert saved["id"] != library_id
    assert saved["name"] == "Leap on this asset"


# --- poser-05 ----------------------------------------------------------------


def _library(segments, *, closed=False):
    keys = ["a", "b"]
    return {
        "poses": [
            {"name": "a", "bones": {"hips": [0.0, 0.0, 0.0, 1.0]}},
            {"name": "b", "bones": {"hips": [0.0, 0.0, 0.0, 1.0]}},
        ],
        "clips": [
            {
                "name": "long",
                "keys": keys,
                "segments": segments,
                "closed": closed,
                "easing": "linear",
                "duration_ms": 100,
            }
        ],
    }


def test_save_refuses_a_clip_whose_segments_expand_past_the_frame_ceiling(svc):
    from realmspinner.service import clips as svc_clips
    from realmspinner.service.errors import Invalid

    with pytest.raises(Invalid) as err:
        svc_clips._check_shape(_library([64]))
    assert err.value.field == "segments"

    # The read door agrees.
    raw = {
        "version": 3,
        "space": "node",
        "poses": [
            {"name": "a", "bones": {"hips": [0, 0, 0, 1]}},
            {"name": "b", "bones": {"hips": [0, 0, 0, 1]}},
        ],
        "clips": [
            {
                "name": "long", "keys": ["a", "b"], "segments": [64], "closed": False,
                "easing": "linear", "duration_ms": 100,
            }
        ],
    }
    with pytest.raises(ValueError, match="frames"):
        cliplib.parse_clip_library(raw)


# --- poser-06 ----------------------------------------------------------------


def test_an_action_whose_name_ends_in_a_facing_is_renamed_not_refused_when_auto_named():
    from realmspinner import cliptransfer

    w3 = _load(_ROOT / "tests" / "test_audit_2026_09_26_w3f5_cliptransfer.py", "_w3f5_helpers")
    source_bones = w3._baseline_source_bones()
    rest = w3._rest_frame_bones(source_bones)
    sample = w3._make_sample(
        source_bones,
        [w3._one_frame_action("strafe_left", rest), w3._one_frame_action("Idle", rest)],
    )
    results = cliptransfer.transfer(
        sample, template="humanoid", frames=2, loop="off", root_motion="none"
    )
    names = [r["clip"]["name"] for r in results]
    assert names[0] == "strafe_left_2"
    assert names[1] == "idle"

    # An explicit name stays refused.
    one = w3._make_sample(source_bones, [w3._one_frame_action("x", rest)])
    with pytest.raises(cliptransfer.ClipTransferError):
        cliptransfer.transfer(
            one, template="humanoid", clip_name="walk_back", frames=2, loop="off",
            root_motion="none",
        )


# --- poser-07 ----------------------------------------------------------------


def test_quitting_with_unsaved_clip_edits_asks_before_discarding_them():
    ctx = FakeCtx()
    state = poser_mode.ensure(ctx)
    state.clips_unsaved = True
    went = []
    assert poser_mode.guard(ctx, "quit", lambda: went.append(1)) is False
    assert not went
    assert ctx.confirms.asked, "quit must ask about the clip working copy"
    ctx.confirms.asked[-1].on_confirm()
    assert went == [1]

    # A non-quit guard (Apply, New pose) does not ask about clips.
    ctx2 = FakeCtx()
    poser_mode.ensure(ctx2).clips_unsaved = True
    ran = []
    assert poser_mode.guard(ctx2, "apply a preset", lambda: ran.append(1)) is True
    assert ran == [1]


def test_unsaved_clip_edits_mark_the_window_title(monkeypatch):
    import pygame

    from realmspinner.studio.shell.quit import QuitMixin

    captions = []
    monkeypatch.setattr(pygame.display, "set_caption", captions.append)
    app = object.__new__(QuitMixin)
    app._title_marked = None
    state = SimpleNamespace(pose_dirty=False, poser=SimpleNamespace(clips_unsaved=True))
    app.app_ctx = SimpleNamespace(state=state)
    app._sync_title()
    assert captions and captions[-1].endswith(" *")


# poser-07's other half: the clip working copy has a crash journal provider.


def _clips_ctx(tmp_path):
    ctx = FakeCtx(svc=SimpleNamespace(config=SimpleNamespace(autosave_dir=tmp_path)))
    state = poser_mode.ensure(ctx)
    state.template = "humanoid"
    state.clips = {
        "template": "humanoid",
        "space": "node",
        "poses": [{"name": "a", "bones": {}}],
        "clips": [{"name": "walk", "keys": ["a", "a"], "segments": [4, 4]}],
        "edited": False,
    }
    state.clip = "walk"
    return ctx, state


def test_unsaved_clip_edits_are_a_journal_slot_and_clean_clips_are_not(tmp_path):
    ctx, state = _clips_ctx(tmp_path)
    assert [s for s in poser_mode._journal_slots(ctx) if s.key == "clips"] == []
    state.clips_unsaved = True
    slots = [s for s in poser_mode._journal_slots(ctx) if s.key == "clips"]
    assert len(slots) == 1
    head = poser_mode.JOURNAL.head_of(slots[0])
    state.clips_touch_serial += 1
    assert poser_mode.JOURNAL.head_of(slots[0]) != head


def test_clip_edits_survive_a_crash_through_the_journal_pair(tmp_path):
    from realmspinner.studio import journal

    ctx, state = _clips_ctx(tmp_path)
    state.clips_unsaved = True
    state.clips_touch_serial = 3
    slot = next(s for s in poser_mode._journal_slots(ctx) if s.key == "clips")
    ctx.submit = lambda key, fn, *a, **k: (fn(*a, **k), True)[1]
    assert journal.write(ctx, poser_mode.JOURNAL, slot, 1.0)
    found = journal.recoverable(ctx)
    assert [f.kind for f in found] == ["pose"]  # sidecar written last, so it is offered

    # A fresh session: nothing open, the Poser has no template yet.
    fresh = FakeCtx(svc=ctx.svc)
    assert poser_mode._journal_adopt(fresh, found[0].path, found[0].meta) is True
    new = poser_mode.ensure(fresh)
    assert new.template == "humanoid"
    assert new.clips_unsaved is True
    assert [c["name"] for c in new.clips["clips"]] == ["walk"]
    assert new.clip == "walk"
    # Saving (or reverting) the recovered edits retires the copy.
    new.clips_unsaved = False
    poser_mode._journal_slots(fresh)
    assert journal.recoverable(fresh) == []


def test_a_recovered_clip_copy_never_overwrites_clip_edits_already_open(tmp_path):
    from realmspinner.studio import journal

    ctx, state = _clips_ctx(tmp_path)
    state.clips_unsaved = True
    slot = next(s for s in poser_mode._journal_slots(ctx) if s.key == "clips")
    ctx.submit = lambda key, fn, *a, **k: (fn(*a, **k), True)[1]
    journal.write(ctx, poser_mode.JOURNAL, slot, 1.0)
    found = journal.recoverable(ctx)[0]

    state.clips["clips"] = [{"name": "mine", "keys": ["a", "a"], "segments": [4, 4]}]
    assert poser_mode._journal_adopt(ctx, found.path, found.meta) is False
    assert state.clips["clips"][0]["name"] == "mine"
    assert found.path.exists(), "a declined copy stays on disk"


def test_a_clip_read_landing_over_unsaved_edits_does_not_replace_them(tmp_path):
    ctx, state = _clips_ctx(tmp_path)
    state.clips_unsaved = True
    done = SimpleNamespace(
        key=poser_mode.CLIPS_KEY,
        result={"template": "humanoid", "space": "node", "poses": [], "clips": []},
        error=None,
    )
    poser_mode.on_task_done(ctx, done)
    assert state.clips["clips"], "the working copy survived the late read"


# --- poser-08 / poser-10 (manual truth) --------------------------------------


def _flat(path: str) -> str:
    return re.sub(r"\s+", " ", (_ROOT / path).read_text("utf-8"))


def test_chapter_08_root_offset_paragraph_agrees_with_chapter_26():
    ch08 = _flat("docs/manual/08-rigging-and-posing.md")
    assert "previews the offset along with the rotations" not in ch08
    assert "rotations only" in ch08
    assert "rotations only" in _flat("docs/manual/26-poser.md")


def test_chapter_25_does_not_call_the_bbox_fit_right_for_a_t_pose():
    ch25 = _flat("docs/manual/25-rigging-and-posing.md")
    assert "That is right for a subject standing in a T-pose" not in ch25
    assert "A-pose" in ch25
    spec = (_ROOT / "src/realmspinner/kernels/rig/blender_spec.py").read_text("utf-8")
    assert "still right for a reference that really is standing in a T-pose" not in re.sub(
        r"\s+", " ", spec
    )


# --- poser-09 ----------------------------------------------------------------


def test_a_re_rig_whose_deformation_qa_is_skipped_leaves_no_stale_rig_qa(tmp_path):
    job_dir = tmp_path
    (job_dir / rig_store.RIG_GLB_TMP).write_bytes(b"new-rig")
    (job_dir / rig_store.RIG_JSON_TMP).write_text(json.dumps({"version": 1}), "utf-8")
    rig_store.rig_qa_png_path(job_dir).write_bytes(b"old-png")
    rig_store.rig_qa_path(job_dir).write_text("{}", "utf-8")

    rig_store.finalize_rig(job_dir)

    assert (job_dir / "rig.glb").read_bytes() == b"new-rig"
    assert not rig_store.rig_qa_png_path(job_dir).exists()
    assert not rig_store.rig_qa_path(job_dir).exists()
