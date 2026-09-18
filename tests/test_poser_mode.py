"""Poser mode's controller, with tasks run inline and no GL anywhere.

The FakeCtx inline-submit pattern: a submitted callable runs immediately, so
the test sees what the task thread would have done, and the on_task_done half
is driven by hand with the captured result -- which is exactly the seam the
dirty-clears-only-on-landing rule lives on.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from warlock import doctor
from warlock.doctor import Check
from warlock.kernels.geom3d import math3d as m3
from warlock.kernels.geom3d.gltf import Model, Node
from warlock.kernels.rig import cliplib, templates
from warlock.pipelines import blender_run
from warlock.service import poses as svc_poses
from warlock.studio.modes.poser import mode as poser_mode
from warlock.studio.viewer.pose import PoseEditor


class _Asks:
    def __init__(self) -> None:
        self.asked: list = []

    def ask(self, item) -> None:
        self.asked.append(item)


class FakeCtx:
    def __init__(self, svc=None, accept=True) -> None:
        self.svc = svc
        self.state = SimpleNamespace(poser=None)
        self.submitted: list[str] = []
        self.results: dict = {}
        # ``TaskRunner.submit``'s own ``tag``, by key -- real ``submit`` pulls
        # it off the kwargs before calling ``fn``; this must too; see
        # ``sync_asset``/``sync_preview`` (create-04, the 2026-09-11 audit).
        self.tags: dict = {}
        self.accept = accept
        self.busy_keys: set[str] = set()
        self.confirms = _Asks()
        self.prompts = _Asks()
        self.toasts: list = []
        self.rig_default = "humanoid"
        self.rigging_available = True
        self.viewer = None  # the shared viewer, for pose_panel.guard
        self.poser_viewer = None
        # ``ctx.job``'s live window -- what :func:`poser_mode.pump_rerig`
        # polls to notice a queued rig job reach "done". Keyed by job id, set
        # by hand in a test the way the real ``JobsCache`` would land it after
        # a poll -- no cadence to fake, only the state it produces.
        self.jobs: dict[str, dict] = {}

    def submit(self, key, fn, *args, tag=None, **kwargs) -> bool:
        self.submitted.append(key)
        self.tags[key] = tag
        if not self.accept:
            return False
        self.results[key] = fn(*args, **kwargs)
        return True

    def busy(self, key) -> bool:
        return key in self.busy_keys

    def toast(self, message, level="info", *args) -> None:
        self.toasts.append((message, level))

    def job_dir(self, job_id):
        return self.svc.job_dir(job_id)

    def job(self, job_id):
        return self.jobs.get(job_id)


#: ``test_frame_thread_doors.py``'s own worker thread name -- reused here so a
#: failure reads the same way theirs does.
WORKER = "warlock-task-test"


class _ThreadedCtx(FakeCtx):
    """``FakeCtx.submit``, but on a real worker thread, joined.

    ``FakeCtx.submit`` runs a task inline, on the calling thread -- fine for
    most of this file, but useless for proving create-04 (the 2026-09-11
    audit): a test whose submit already runs on the caller could not tell
    "the decode ran here" from "the decode ran where it should have". This is
    ``test_frame_thread_doors.py``'s own ``_Threaded``, reused by shape rather
    than by import since that file is owned by a different fixer this pass.
    """

    def __init__(self, root: Path) -> None:
        super().__init__(svc=None)
        self._root = root

    def job_dir(self, job_id):
        return self._root / job_id

    def submit(self, key, fn, *args, tag=None, **kwargs) -> bool:
        self.submitted.append(key)
        self.tags[key] = tag
        if not self.accept:
            return False
        box: dict = {}

        def go() -> None:
            box["result"] = fn(*args, **kwargs)
            box["thread"] = threading.current_thread().name

        worker = threading.Thread(target=go, name=WORKER)
        worker.start()
        worker.join()
        self.results[key] = box["result"]
        self.threads = getattr(self, "threads", [])
        self.threads.append(box["thread"])
        return True


class FakeViewer:
    """Just the surface poser_mode touches, over a real PoseEditor."""

    def __init__(self, model=None, bones=None) -> None:
        self.editor = PoseEditor()
        self.pose_mode = False
        self.path = None
        self.loaded: list = []
        self.cleared = 0
        self.framed = None
        # The onion-skin ghosts the clip editor pushes; empty everywhere else.
        self.onion: list = []
        # The real camera, because the view keys are about what it does and a
        # stub would pass whatever they did.
        from warlock.studio.viewer.camera import Camera

        self.camera = Camera()
        # ``enter_pose_mode``'s stand-in for "the loaded GLB has a skeleton" --
        # real ``Viewer.enter_pose_mode`` checks ``self.model.skins``, which
        # this fake never parses a real GLB into, so it is a plain flag
        # instead. Defaults True: most callers are testing the orchestration
        # around it, not the skin check itself.
        self.skinned = True
        self.pose_job_id: str | None = None
        # The real Viewer's parse/adopt tracking (create-04, the 2026-09-11
        # audit): the path a dispatched-but-not-yet-landed parse is for, or
        # None. See ``parse_model``/``adopt_model`` below.
        self.pending: Path | None = None
        if model is not None:
            self.editor.bind(model, bones)
            self.pose_mode = True

    def load_model(self, path) -> None:
        self.loaded.append(Path(path))
        self.path = Path(path)

    def parse_model(self, path) -> Any:
        """The task-thread half, real ``Viewer.parse_model``'s contract: pure,
        touches nothing on ``self``. Returns the path so ``adopt_model`` below
        has something to tell apart from a stray ``None``."""
        return ("parsed", Path(path))

    def adopt_model(self, parsed, path) -> None:
        """The frame-thread half: the ``load_model`` side effects, plus the
        pose-mode reset the real ``Viewer.adopt_model`` always does -- whatever
        was bound before this landed is not this."""
        self.pose_mode = False
        self.pose_job_id = None
        self.loaded.append(Path(path))
        self.path = Path(path)
        self.pending = None

    def enter_pose_authoring(self, bones, mirror_pairs, token) -> bool:
        self.editor.mirror_pairs = [list(p) for p in mirror_pairs]
        self.pose_mode = True
        self.token = token
        self.pose_job_id = token
        return True

    def enter_pose_mode(self, rig, job_id) -> bool:
        if not self.skinned:
            return False
        self.rig = rig
        self.pose_job_id = job_id
        self.pose_mode = True
        return True

    def exit_pose_mode(self) -> None:
        self.pose_mode = False
        self.pose_job_id = None

    def frame(self) -> float:
        self.framed = "whole-model"
        return 1.0

    def frame_bounds(self, lo, hi) -> float:
        self.framed = (lo, hi)
        return 1.0

    def clear(self) -> None:
        self.cleared += 1
        self.path = None
        self.editor.clear()
        self.pose_mode = False
        # Real ``Viewer.clear`` resets this too (see ``adopt_model``'s own
        # docstring): a parse dispatched for whatever this viewer was showing
        # is not wanted once it has been cleared -- create-04's freshness
        # check depends on this.
        self.pending = None

    def set_pose(self, bones, *, pose_id=None, dirty=True) -> None:
        self.editor.apply(bones, pose_id=pose_id, dirty=dirty)

    def get_pose(self):
        return self.editor.pose()

    def set_root_translation(self, v, *, dirty=True) -> None:
        self.editor.set_root_translation(v, dirty=dirty)

    def apply_preset(self, preset) -> None:
        self.editor.apply_preset(preset)

    def reset_all(self, *, dirty=True) -> None:
        self.editor.reset_all(dirty=dirty)

    def mirror(self) -> None:
        self.editor.mirror()

    # -- skeleton mode (P6, 2026-09-13) --------------------------------------
    #
    # ``_viewer_pose.PoseOps``'s own pass-throughs, minus the GPU refresh
    # they also do -- this fake never binds a mesh a palette could be
    # recomputed for, so there is nothing that call would touch.

    def enter_skeleton_mode(self, rig) -> None:
        self.editor.enter_skeleton_mode(rig)

    def exit_skeleton_mode(self) -> None:
        self.editor.exit_skeleton_mode()

    def skeleton_payload(self):
        return self.editor.skeleton_payload()

    def skel_add_child(self, parent):
        return self.editor.skel_add_child(parent)

    def skel_split(self, name):
        return self.editor.skel_split(name)

    def skel_remove_pivot(self, name) -> None:
        self.editor.skel_remove_pivot(name)

    def skel_remove_subtree(self, name):
        return self.editor.skel_remove_subtree(name)

    def skel_rename(self, old, new) -> None:
        self.editor.skel_rename(old, new)

    def skel_attach_limb(self, preset_key, parent, side, mirror):
        return self.editor.skel_attach_limb(preset_key, parent, side, mirror)

    def subtree_size(self, name) -> int:
        return self.editor.subtree_size(name)


def _full_bones(template="humanoid"):
    """Every bone at identity -- validate_record requires the whole skeleton,
    which is what get_pose(), the library's only real writer, produces."""
    return {b["name"]: [0.0, 0.0, 0.0, 1.0] for b in templates.get_template(template).bones}


def _armature_model() -> Model:
    """One node per humanoid bone under a 'rig' object node -- flat, because
    the editor needs only names and positions here, and a save built off this
    must carry the template's *whole* skeleton (validate_record's bar)."""
    names = [b["name"] for b in templates.get_template("humanoid").bones]
    nodes = [Node(name="rig", children=list(range(1, len(names) + 1)))]
    for i, name in enumerate(names):
        nodes.append(Node(name=name, translation=m3.vec3(0.0, 0.1 * i, 0.0)))
    return Model(nodes, roots=[0], meshes=[], skins=[])


def _bound_viewer() -> FakeViewer:
    viewer = FakeViewer(
        _armature_model(), [b["name"] for b in templates.get_template("humanoid").bones]
    )
    viewer.editor.root = "hips"
    return viewer


def _fake_blender(monkeypatch) -> None:
    monkeypatch.setattr(
        doctor, "blender_check", lambda **kw: Check("Blender (rigging)", True, "bpy 5.2.0", False)
    )

    def run_worker(spec, **kwargs):
        Path(spec["out_glb"]).write_bytes(b"armature-glb")
        return {"ok": True}

    monkeypatch.setattr(blender_run, "run_worker", run_worker)


# --- entering ----------------------------------------------------------------


def test_enter_refreshes_the_library_and_requests_the_preview(svc, monkeypatch):
    _fake_blender(monkeypatch)
    stored = svc_poses.create_library_pose(
        svc,
        {"name": "Crouch", "template": "humanoid", "bones": _full_bones()},
    )
    ctx = FakeCtx(svc)
    poser_mode.enter(ctx)
    state = poser_mode.ensure(ctx)
    assert state.template == "humanoid"
    assert poser_mode.LIST_KEY in ctx.submitted
    assert f"{poser_mode.PREVIEW_KEY_PREFIX}humanoid" in ctx.submitted

    # The results land through on_task_done, exactly as the app delivers them.
    poser_mode.on_task_done(
        ctx, SimpleNamespace(key=poser_mode.LIST_KEY, result=ctx.results[poser_mode.LIST_KEY])
    )
    assert [p["id"] for p in state.poses] == [stored["id"]]
    assert any(p["name"] == "idle" for p in state.presets), "shipped presets ride along"
    assert state.loading is False

    key = f"{poser_mode.PREVIEW_KEY_PREFIX}humanoid"
    poser_mode.on_task_done(ctx, SimpleNamespace(key=key, result=ctx.results[key]))
    assert state.building is False
    assert state.preview_template == "humanoid"
    assert Path(state.preview_path).exists()


def test_a_refused_submit_clears_loading(svc):
    ctx = FakeCtx(svc, accept=False)
    poser_mode.refresh(ctx)
    assert poser_mode.ensure(ctx).loading is False


def test_a_failed_task_clears_its_flags(svc):
    ctx = FakeCtx(svc)
    state = poser_mode.ensure(ctx)
    state.loading = True
    poser_mode.on_task_failed(ctx, SimpleNamespace(key=poser_mode.LIST_KEY, message=""))
    assert state.loading is False
    state.building = True
    poser_mode.on_task_failed(
        ctx,
        SimpleNamespace(key=f"{poser_mode.PREVIEW_KEY_PREFIX}humanoid", message="no bpy"),
    )
    assert state.building is False
    assert state.error == "no bpy"


def test_a_stale_preview_failure_does_not_clobber_the_template_the_user_switched_to(svc):
    """The 2026-09-08 audit (poser-03): state.building/state.error are single,
    un-scoped fields, so a PREVIEW_KEY_PREFIX landing always overwrote them
    even when it was submitted for a template the user has since switched
    away from -- unlike CLIPS_KEY's own landing, which already checks
    ``done.result.get("template") == state.template`` before adopting
    (test_a_landing_for_another_template_is_ignored). This is the same guard
    for its sibling, on both the success and the failure landing.
    """
    ctx = FakeCtx(svc)
    state = poser_mode.ensure(ctx)
    assert state.template == "humanoid"
    state.building = True
    state.error = ""

    # The user switches to quadruped while humanoid's build is still in
    # flight, and humanoid's build then fails.
    state.template = "quadruped"
    poser_mode.on_task_failed(
        ctx,
        SimpleNamespace(key=f"{poser_mode.PREVIEW_KEY_PREFIX}humanoid", message="no bpy"),
    )
    assert state.building is True, "quadruped's own build is still in flight"
    assert state.error == "", "humanoid's stale failure must not show over quadruped"

    # A stale *success* landing is dropped the same way.
    poser_mode.on_task_done(
        ctx,
        SimpleNamespace(
            key=f"{poser_mode.PREVIEW_KEY_PREFIX}humanoid", result="/tmp/humanoid.glb"
        ),
    )
    assert state.building is True
    assert state.preview_template != "humanoid"

    # quadruped's own landing still works normally.
    poser_mode.on_task_failed(
        ctx,
        SimpleNamespace(key=f"{poser_mode.PREVIEW_KEY_PREFIX}quadruped", message="no bpy"),
    )
    assert state.building is False
    assert state.error == "no bpy"


# --- the template switch -----------------------------------------------------


def test_a_dirty_editor_guards_the_template_switch(svc):
    ctx = FakeCtx(svc)
    state = poser_mode.ensure(ctx)
    ctx.poser_viewer = _bound_viewer()
    ctx.poser_viewer.editor.dirty = True

    poser_mode.set_template(ctx, "fish")
    assert state.template == "humanoid", "nothing moved before the answer"
    assert len(ctx.confirms.asked) == 1

    ctx.confirms.asked[0].on_confirm()
    assert state.template == "fish"
    assert ctx.poser_viewer.cleared == 1, "the old armature must not stay poseable"
    assert poser_mode.LIST_KEY in ctx.submitted


def test_a_clean_editor_switches_immediately(svc):
    ctx = FakeCtx(svc)
    ctx.poser_viewer = _bound_viewer()
    poser_mode.set_template(ctx, "fish")
    assert poser_mode.ensure(ctx).template == "fish"
    assert ctx.confirms.asked == []


def test_switching_template_refreshes_the_clip_library_and_guards_unsaved_clip_edits(svc):
    """poser-01 (the 2026-09-07 audit): ``set_template``'s ``proceed()`` reset
    the pose library but never touched ``state.clips``/``clips_unsaved`` and
    never called ``clips_refresh`` -- so "Save clips" after a switch wrote the
    *old* template's working copy under the *new* template's name, with no
    prompt at all, since the pose-gizmo guard never reads ``clips_unsaved``.
    """
    ctx = FakeCtx(svc)
    state = poser_mode.ensure(ctx)
    state.clips = {
        "space": "node",
        "poses": [{"name": "a", "bones": {"hips": [0.0, 0.0, 0.0, 1.0]}}],
        "clips": [{"name": "walk", "keys": ["a", "a"], "segments": [4]}],
    }
    state.clip = "walk"
    state.clips_unsaved = True

    poser_mode.set_template(ctx, "fish")
    assert state.template == "humanoid", "nothing moved before the answer"
    assert len(ctx.confirms.asked) == 1, "an unsaved clip edit must be asked about too"

    ctx.confirms.asked[0].on_confirm()
    assert state.template == "fish"
    assert state.clips == {}, "the old template's clip library must not survive the switch"
    assert state.clip == ""
    assert state.clips_unsaved is False
    assert poser_mode.CLIPS_KEY in ctx.submitted, "the new template's clips must be re-read"


# --- the asset session --------------------------------------------------------


def _rigged_job(svc, **meta):
    """A finished mesh with a rig beside it, the way the worker leaves one --
    ``test_inspector_rig.py``'s helper, needed here because ``open_asset``
    goes through the real ``service.rig`` doors (``get_rig``, ``list_poses``),
    which refuse a job with no rig on disk."""
    job_id = svc.store.create("image", "a prop", {}, stage="model", status="done")
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"glTF-not-really")
    (job_dir / "rig.glb").write_bytes(b"glTF-not-really")
    (job_dir / "rig.json").write_text(
        json.dumps({"version": 1, "template": "humanoid", "bones": [], **meta}), "utf-8"
    )
    return job_id


def test_poser_does_not_decode_the_rig_on_the_frame_thread(tmp_path):
    """create-04 (the 2026-09-11 audit). ``sync_asset``/``sync_preview`` run
    from ``_poser_viewport``'s draw on *every* frame Poser is open, and used
    to call ``viewer.load_model`` -- a full glTF parse, texture decode and GPU
    upload -- directly, on whatever thread called them. That is fine for the
    genuinely click-driven case (``open_asset``'s own proceed, which still
    binds synchronously -- see the test above and below this one) but not for
    the *automatic* one: a queued re-rig landing (``_land_rerig``) while the
    user is doing nothing in particular clears the viewer and leaves the next
    frame's ``sync_asset`` to rebind it, with no click behind that frame at
    all.

    Driven the way ``tests/test_frame_thread_doors.py`` drives the other ten
    doors of this same class: a ``ctx`` whose ``submit`` runs the task on a
    real, joined worker thread, so a regression that moves the decode back
    onto the caller shows up as the calling thread's own name rather than
    ``WORKER``.
    """
    ctx = _ThreadedCtx(tmp_path)
    viewer = ctx.poser_viewer = FakeViewer()
    state = poser_mode.ensure(ctx)
    # No open_asset door here on purpose: this asset is already "open" --
    # the state a re-rig landing while idle finds -- with no click in this
    # test to hide the wait behind. Only the automatic path is under test.
    state.job_id = "abcdef012345"
    state.asset_rig = {}

    assert poser_mode.sync_asset(ctx, viewer) is False, "must not block this call"
    assert ctx.threads == [WORKER], "the parse must run off the frame thread"
    assert viewer.loaded == [], "no synchronous load_model call from this thread"
    assert viewer.pose_mode is False, "not adopted yet -- on_task_done's job"

    key = poser_mode.ASSET_LOAD_KEY
    poser_mode.on_task_done(
        ctx, SimpleNamespace(key=key, result=ctx.results[key], tag=ctx.tags[key])
    )
    assert viewer.pose_mode is True and viewer.pose_job_id == "abcdef012345"
    assert viewer.loaded == [ctx.job_dir(state.job_id) / "rig.glb"]


def test_open_asset_binds_via_enter_pose_mode_not_authoring(svc):
    """Opening a rigged asset must show its real mesh -- ``enter_pose_mode``,
    the skin-checked entry point -- never fall back to the meshless armature's
    ``enter_pose_authoring``."""
    job_id = _rigged_job(svc)
    ctx = FakeCtx(svc)
    ctx.poser_viewer = viewer = FakeViewer()
    job = {"id": job_id, "name": "Test Prop"}

    poser_mode.open_asset(ctx, job)
    state = poser_mode.ensure(ctx)
    assert state.job_id == job_id
    assert state.asset_label == "Test Prop"

    assert poser_mode.sync_asset(ctx, viewer) is True
    assert viewer.loaded == [ctx.job_dir(job_id) / "rig.glb"]
    assert viewer.pose_mode is True
    assert viewer.pose_job_id == job_id, "a save from here must address the job"
    assert viewer.framed == "whole-model", "the real mesh is framed, not a bone box"

    # Idempotent: already showing it, so no second load.
    assert poser_mode.sync_asset(ctx, viewer) is True
    assert len(viewer.loaded) == 1


def test_sync_asset_reports_a_skeletonless_glb_without_retrying_every_frame(svc):
    job_id = _rigged_job(svc)
    ctx = FakeCtx(svc)
    ctx.poser_viewer = viewer = FakeViewer()
    viewer.skinned = False

    poser_mode.open_asset(ctx, {"id": job_id})
    state = poser_mode.ensure(ctx)

    assert poser_mode.sync_asset(ctx, viewer) is False
    assert state.asset_error
    loaded_once = list(viewer.loaded)

    # A load failure is not retried every frame until asked to be.
    assert poser_mode.sync_asset(ctx, viewer) is False
    assert viewer.loaded == loaded_once

    poser_mode.retry_asset(ctx)
    viewer.skinned = True
    # retry_asset only clears the flag: the actual retry runs through
    # sync_asset like every other automatic bind (create-04, the 2026-09-11
    # audit) -- dispatched here, adopted once ``on_task_done`` lands it.
    assert poser_mode.sync_asset(ctx, viewer) is False
    key = poser_mode.ASSET_LOAD_KEY
    poser_mode.on_task_done(
        ctx, SimpleNamespace(key=key, result=ctx.results[key], tag=ctx.tags[key])
    )
    assert viewer.pose_mode is True


def test_a_dirty_editor_guards_opening_an_asset(svc):
    job_id = _rigged_job(svc)
    ctx = FakeCtx(svc)
    ctx.poser_viewer = viewer = _bound_viewer()
    viewer.editor.dirty = True

    poser_mode.open_asset(ctx, {"id": job_id})
    state = poser_mode.ensure(ctx)
    assert state.job_id == "", "nothing moved before the answer"
    assert len(ctx.confirms.asked) == 1

    ctx.confirms.asked[0].on_confirm()
    assert state.job_id == job_id
    assert viewer.cleared == 1, "the old session must not stay poseable"


def test_opening_a_different_asset_resets_the_stale_rerig_picker_and_limb_form(svc):
    """The 2026-09-13 audit, finding poser-01: ``open_asset``'s ``proceed()``
    reset the asset session's fields but not ``rerig_open``, ``rerig_choice``,
    ``limb_preset``, ``limb_side`` or ``limb_mirror``, although their own
    docstrings say the session clears them. Picking a different asset while
    A's Re-rig picker was open showed B with A's skeleton pre-chosen, and
    Confirm would re-rig B under it -- discarding B's own skeleton, poses and
    baked animation."""
    job_a = _rigged_job(svc)
    job_b = _rigged_job(svc)
    ctx = FakeCtx(svc)
    ctx.poser_viewer = FakeViewer()

    poser_mode.open_asset(ctx, {"id": job_a, "name": "A"})
    state = poser_mode.ensure(ctx)
    state.rerig_open = True
    state.rerig_choice = "quadruped"
    state.limb_preset = "arm"
    state.limb_side = "L"
    state.limb_mirror = True

    poser_mode.open_asset(ctx, {"id": job_b, "name": "B"})
    assert state.job_id == job_b
    assert state.rerig_open is False, "B must not inherit A's open picker"
    assert state.rerig_choice == "", "B must not inherit A's chosen skeleton"
    assert state.limb_preset == ""
    assert state.limb_side == ""
    assert state.limb_mirror is False


def test_close_asset_resets_every_asset_field(svc, monkeypatch):
    """The poser-01 lesson, restated for the asset session: every field
    ``open_asset`` can leave set has to be cleared by its own exit door."""
    _fake_blender(monkeypatch)
    job_id = _rigged_job(svc)
    ctx = FakeCtx(svc)
    ctx.poser_viewer = viewer = FakeViewer()
    poser_mode.open_asset(ctx, {"id": job_id, "name": "Prop"})
    state = poser_mode.ensure(ctx)
    poser_mode.sync_asset(ctx, viewer)
    state.asset_poses = [{"id": "p1", "name": "idle"}]

    poser_mode.close_asset(ctx)
    assert state.job_id == ""
    assert state.asset_label == ""
    assert state.asset_rig is None
    assert state.asset_poses == []
    assert state.asset_error == ""
    assert viewer.pose_mode is False, "exit_pose_mode must run before clear"
    assert viewer.cleared == 2, "once when open_asset bound it, once on close"
    assert poser_mode.PREVIEW_KEY_PREFIX + state.template in ctx.submitted, (
        "closing falls back to the template preview"
    )


def test_opening_a_second_asset_while_the_first_ones_pose_list_is_still_loading_does_not_strand_it(
    svc,
):
    """The 2026-09-08 audit (poser-04): ``asset_poses_loading`` was one flag
    shared by every asset even though its landing key
    (``ASSET_POSES_KEY_PREFIX`` + job id) is already per-job -- so opening a
    second rigged asset while the first one's saved-pose fetch was still in
    flight silently refused to submit a read for the second asset, and
    nothing re-armed it when the first asset's stale result landed.
    """
    job1 = _rigged_job(svc)
    job2 = _rigged_job(svc)
    ctx = FakeCtx(svc)
    state = poser_mode.ensure(ctx)

    # job1's fetch is still in flight -- set the way a real submit leaves it
    # standing until its own landing.
    state.job_id = job1
    state.asset_poses_loading.add(job1)

    # The user opens a second rigged asset before job1's fetch lands.
    state.job_id = job2
    poser_mode.refresh_asset_poses(ctx)
    key2 = f"{poser_mode.ASSET_POSES_KEY_PREFIX}{job2}"
    assert key2 in ctx.submitted, "job2's own read must still be submitted"
    assert job2 in state.asset_poses_loading

    # job1's stale result lands: it must clear only its own flag, and it must
    # not strand job1 loading forever or touch job2's session.
    poser_mode.on_task_done(
        ctx,
        SimpleNamespace(
            key=f"{poser_mode.ASSET_POSES_KEY_PREFIX}{job1}",
            result={"poses": [{"id": "stale"}]},
        ),
    )
    assert job1 not in state.asset_poses_loading
    assert job2 in state.asset_poses_loading, "job2's own fetch is unaffected"
    assert state.asset_poses == [], "job1's stale poses must not land on job2's session"

    # job2's own landing still completes normally.
    poser_mode.on_task_done(
        ctx, SimpleNamespace(key=key2, result=ctx.results[key2])
    )
    assert job2 not in state.asset_poses_loading
    assert state.asset_poses == ctx.results[key2]["poses"]


def test_reframe_frames_the_real_mesh_when_an_asset_is_bound(svc):
    job_id = _rigged_job(svc)
    ctx = FakeCtx(svc)
    ctx.poser_viewer = viewer = FakeViewer()
    poser_mode.open_asset(ctx, {"id": job_id})
    poser_mode.reframe(ctx)
    assert viewer.framed == "whole-model"


# --- re-rigging an already-open asset -----------------------------------------
#
# The 2026-09-07 finding: with an asset bound, the Skeleton combo becomes a
# bare fact ("... from this asset's rig") and there was no way back to a
# different skeleton short of closing the session, finding the source job in
# the Library, and choosing Rig from there. ``rerig`` queues the same job the
# Library's own Rig action does, reachable from the session that is already
# open.


def _opened_asset(svc, monkeypatch, **rig_meta):
    """A Poser session bound to a real rigged asset, with the viewer already
    showing it -- the state every test below starts from."""
    _fake_blender(monkeypatch)
    job_id = _rigged_job(svc, **rig_meta)
    ctx = FakeCtx(svc)
    ctx.poser_viewer = viewer = FakeViewer()
    poser_mode.open_asset(ctx, {"id": job_id, "name": "Prop"})
    poser_mode.sync_asset(ctx, viewer)
    return ctx, viewer, job_id


def test_rerig_submits_under_a_poser_key_not_the_shared_pose_key(svc, monkeypatch):
    ctx, viewer, job_id = _opened_asset(svc, monkeypatch)

    poser_mode.rerig(ctx, "humanoid")
    key = f"{poser_mode.ASSET_RERIG_KEY_PREFIX}{job_id}"
    assert key in ctx.submitted
    # "poser-", not "pose-": main.py's generic "pose-" dispatch clears the
    # *shared* viewer's dirty flag, and this has to land on Poser's own.
    assert key.startswith("poser-")
    assert not any(k.startswith("pose-") and not k.startswith("poser-") for k in ctx.submitted)


def test_a_refused_rerig_submit_is_toasted_not_silent(svc, monkeypatch):
    """``TaskRunner.submit`` returning False (a live key) is the whole
    concurrency guard the plan asks for -- this pins that a refusal is still
    answered with a sentence, the same rule ``_mutate`` follows for the
    library, rather than a press that does nothing."""
    _fake_blender(monkeypatch)
    job_id = _rigged_job(svc)
    ctx = FakeCtx(svc, accept=False)
    ctx.poser_viewer = viewer = FakeViewer()
    poser_mode.open_asset(ctx, {"id": job_id})
    poser_mode.sync_asset(ctx, viewer)

    poser_mode.rerig(ctx, "humanoid")
    assert any("re-rig" in msg.lower() for msg, _level in ctx.toasts)
    # Nothing to watch for: the submit never landed.
    assert poser_mode.ensure(ctx).rerig_jobs == {}


def test_rerig_is_guarded_by_unsaved_pose_edits(svc, monkeypatch):
    ctx, viewer, job_id = _opened_asset(svc, monkeypatch)
    viewer.editor.dirty = True

    poser_mode.rerig(ctx, "humanoid")
    key = f"{poser_mode.ASSET_RERIG_KEY_PREFIX}{job_id}"
    assert len(ctx.confirms.asked) == 1
    assert key not in ctx.submitted, "nothing queued before the answer"

    ctx.confirms.asked[0].on_confirm()
    assert key in ctx.submitted


def test_pump_rerig_rebinds_once_the_queued_job_lands(svc, monkeypatch):
    """The whole point: ``sync_asset`` short-circuits on ``viewer.pose_mode
    and viewer.pose_job_id == job_id``, both still true after a same-job
    re-rig, so unfixed this is a silent no-op -- the button appears to work
    and the viewport never changes, because nothing here is told when the
    queue actually finishes writing the new rig.glb."""
    ctx, viewer, job_id = _opened_asset(svc, monkeypatch)
    assert viewer.pose_mode is True and viewer.pose_job_id == job_id

    poser_mode.rerig(ctx, "humanoid")
    key = f"{poser_mode.ASSET_RERIG_KEY_PREFIX}{job_id}"
    result = ctx.results[key]
    poser_mode.on_task_done(ctx, SimpleNamespace(key=key, result=result))
    state = poser_mode.ensure(ctx)
    assert state.rerig_jobs[job_id] == result["id"]

    # The queue has not gotten to it yet: nothing rebinds.
    ctx.jobs[result["id"]] = {"id": result["id"], "status": "queued"}
    poser_mode.pump_rerig(ctx)
    assert viewer.pose_mode is True, "still the old session until the job lands"
    assert state.rerig_jobs[job_id] == result["id"]

    ctx.jobs[result["id"]] = {"id": result["id"], "status": "done"}
    poser_mode.pump_rerig(ctx)
    assert viewer.pose_mode is False, "sync_asset's same-job short-circuit is defeated"
    assert job_id not in state.rerig_jobs

    # And genuinely rebindable, not just knocked out of pose mode -- through
    # the parse/adopt split now, since this landing has no click behind it
    # (create-04, the 2026-09-11 audit): dispatched here, adopted once
    # ``on_task_done`` lands it, never blocking this call itself.
    assert poser_mode.sync_asset(ctx, viewer) is False
    load_key = poser_mode.ASSET_LOAD_KEY
    poser_mode.on_task_done(
        ctx,
        SimpleNamespace(key=load_key, result=ctx.results[load_key], tag=ctx.tags[load_key]),
    )
    assert viewer.pose_mode is True
    assert len(viewer.loaded) == 2, "sync_asset reloads rig.glb a second time"


def test_land_rerig_asks_before_discarding_a_pose_edited_while_the_job_was_queued(
    svc, monkeypatch
):
    """poser-01 (the 2026-09-08 audit): ``rerig``'s own guard protects only the
    moment the re-rig is *submitted* -- an ordinary thing to do while a
    Blender rig job serialises on the queue is to keep posing the old rig, and
    unfixed ``_land_rerig`` called ``viewer.exit_pose_mode()``/``clear()``
    unconditionally the instant the job landed, discarding that edit with no
    confirm and no toast."""
    ctx, viewer, job_id = _opened_asset(svc, monkeypatch)
    poser_mode.rerig(ctx, "humanoid")
    key = f"{poser_mode.ASSET_RERIG_KEY_PREFIX}{job_id}"
    result = ctx.results[key]
    poser_mode.on_task_done(ctx, SimpleNamespace(key=key, result=result))
    state = poser_mode.ensure(ctx)

    # The user keeps posing the *old* rig while the re-rig is still queued.
    viewer.editor.dirty = True

    ctx.jobs[result["id"]] = {"id": result["id"], "status": "done"}
    poser_mode.pump_rerig(ctx)

    assert viewer.pose_mode is True, "must not discard the unsaved edit with no confirm"
    assert len(ctx.confirms.asked) == 1
    assert job_id not in state.rerig_jobs, "the landed job is not re-polled while the confirm waits"

    ctx.confirms.asked[0].on_confirm()
    assert viewer.pose_mode is False, "confirming goes ahead and lands the re-rig"


def test_pump_rerig_ignores_a_failed_job(svc, monkeypatch):
    ctx, viewer, job_id = _opened_asset(svc, monkeypatch)
    poser_mode.rerig(ctx, "humanoid")
    key = f"{poser_mode.ASSET_RERIG_KEY_PREFIX}{job_id}"
    result = ctx.results[key]
    poser_mode.on_task_done(ctx, SimpleNamespace(key=key, result=result))

    ctx.jobs[result["id"]] = {"id": result["id"], "status": "error"}
    poser_mode.pump_rerig(ctx)
    assert poser_mode.ensure(ctx).rerig_jobs == {}
    assert viewer.pose_mode is True, "the old session survives a failed re-rig"


def test_rerigging_a_second_asset_does_not_drop_tracking_of_the_first(svc, monkeypatch):
    """poser-05, the 2026-09-11 audit: rerig_job_id/rerig_source_job were a
    single pair of fields, so re-rigging asset A and then, before that job
    landed, opening a different rigged asset B and re-rigging it too silently
    overwrote A's tracking with B's, dropping pump_rerig's ability to ever
    notice A's job finish -- exactly the ``asset_poses_loading`` hole the
    2026-09-08 audit's poser-04 fixed by scoping per job id, never applied
    here."""
    ctx, viewer, job_a = _opened_asset(svc, monkeypatch)
    poser_mode.rerig(ctx, "humanoid")
    key_a = f"{poser_mode.ASSET_RERIG_KEY_PREFIX}{job_a}"
    result_a = ctx.results[key_a]
    poser_mode.on_task_done(ctx, SimpleNamespace(key=key_a, result=result_a))
    state = poser_mode.ensure(ctx)
    assert state.rerig_jobs[job_a] == result_a["id"]

    # Before A's job lands, the user opens a second rigged asset and re-rigs
    # it too, in the same session.
    job_b = _rigged_job(svc)
    poser_mode.open_asset(ctx, {"id": job_b, "name": "Prop 2"})
    poser_mode.sync_asset(ctx, viewer)
    poser_mode.rerig(ctx, "humanoid")
    key_b = f"{poser_mode.ASSET_RERIG_KEY_PREFIX}{job_b}"
    result_b = ctx.results[key_b]
    poser_mode.on_task_done(ctx, SimpleNamespace(key=key_b, result=result_b))

    # Both re-rigs are still tracked -- landing B's must not have clobbered A's.
    assert state.rerig_jobs[job_a] == result_a["id"], "A's tracking must survive B's own re-rig"
    assert state.rerig_jobs[job_b] == result_b["id"]

    # A's queue job lands while B is the asset actually open: it must be
    # noticed and retired, but never rebind the live, unrelated session.
    ctx.jobs[result_a["id"]] = {"id": result_a["id"], "status": "done"}
    cleared_before = viewer.cleared
    poser_mode.pump_rerig(ctx)
    assert job_a not in state.rerig_jobs
    assert job_b in state.rerig_jobs, "B's own tracking must survive A landing"
    assert viewer.cleared == cleared_before, "A's landing must not touch B's live session"

    # And B's own job landing still rebinds normally.
    ctx.jobs[result_b["id"]] = {"id": result_b["id"], "status": "done"}
    poser_mode.pump_rerig(ctx)
    assert job_b not in state.rerig_jobs
    assert viewer.pose_mode is False, "B's own landing still defeats the same-job short-circuit"


def test_pump_rerig_does_nothing_once_the_session_has_moved_on(svc, monkeypatch):
    """The user can close this asset (or open a different one) while the
    queue is still working; landing the stale re-rig onto whatever is bound
    now would rebind the wrong session."""
    ctx, viewer, job_id = _opened_asset(svc, monkeypatch)
    poser_mode.rerig(ctx, "humanoid")
    key = f"{poser_mode.ASSET_RERIG_KEY_PREFIX}{job_id}"
    result = ctx.results[key]
    poser_mode.on_task_done(ctx, SimpleNamespace(key=key, result=result))

    poser_mode.close_asset(ctx)
    cleared = viewer.cleared
    ctx.jobs[result["id"]] = {"id": result["id"], "status": "done"}
    poser_mode.pump_rerig(ctx)
    assert viewer.cleared == cleared, "no further rebind for a session that already closed"


def test_rerig_landing_with_a_new_template_runs_the_switching_template_reset(svc, monkeypatch):
    """poser-01's reset (:func:`poser_mode._reset_for_template`), reused for a
    third door: a re-rig that lands under a different skeleton than the one
    being browsed must not leave the clip editor pointed at the old
    template's working copy."""
    ctx, viewer, job_id = _opened_asset(svc, monkeypatch)
    state = poser_mode.ensure(ctx)
    state.clips = {"clips": [{"name": "walk"}]}
    state.clip = "walk"
    state.key_index = 2
    state.frame = 5
    state.frames = [{"x": 1}]
    state.clips_error = "boom"
    state.clips_unsaved = True

    poser_mode.rerig(ctx, "quadruped")
    key = f"{poser_mode.ASSET_RERIG_KEY_PREFIX}{job_id}"
    result = ctx.results[key]
    poser_mode.on_task_done(ctx, SimpleNamespace(key=key, result=result))

    # The queue "finishes": the mesh's rig.json now names the new template.
    (svc.job_dir(job_id) / "rig.json").write_text(
        json.dumps({"version": 1, "template": "quadruped", "bones": []}), "utf-8"
    )
    ctx.jobs[result["id"]] = {"id": result["id"], "status": "done"}
    poser_mode.pump_rerig(ctx)

    assert state.template == "quadruped"
    assert state.clips == {}
    assert state.clip == ""
    assert state.key_index == 0
    assert state.frame == -1
    assert state.frames == []
    assert state.clips_error == ""
    assert state.clips_unsaved is False


def test_rerig_control_is_gated_by_the_pane_s_own_blender_check():
    """Requirement: refuse with a reason when Blender is unavailable, never a
    greyed button with none. ``draw`` already returns before the ``job_id``
    branch (and so before ``_rerig``) once ``ctx.rigging_available`` is
    false; this pins that the control never grew a second, silent gate."""
    import inspect

    from warlock.studio.modes.poser.ui.panes import library as poser_library

    source = inspect.getsource(poser_library.draw)
    assert "_rerig(ctx, state)" in source, "the control must actually be wired in"
    reason = source.index("Posing needs Blender")
    branch = source.index("if state.job_id:")
    assert reason < branch, "the availability refusal must guard the whole branch"


# --- the skeleton editor (P6, 2026-09-13) -------------------------------------


def _humanoid_bones():
    return [dict(b) for b in templates.get_template("humanoid").bones]


def _custom_rig_meta():
    bones = _humanoid_bones()
    root = next(b["name"] for b in bones if b["parent"] is None)
    return {
        "skeleton": "custom",
        "bones": bones,
        "root": root,
        "mirror_pairs": [],
        # ``service.rig.edit_skeleton`` refuses outright with no usable
        # bounds on the rig -- the canonical unit box every template is fit
        # inside (``poselib.UNIT_LO``/``UNIT_HI``) is generous enough that no
        # humanoid bone in this file's tests falls outside it.
        "bounds": {"min": [-0.5, -0.5, 0.0], "max": [0.5, 0.5, 1.0]},
    }


def _opened_asset_for_skeleton(svc, monkeypatch, **rig_meta):
    """``_opened_asset``'s own setup, with the editor actually bound to a
    model.

    ``PoseEditor.enter_skeleton_mode`` no-ops on an unbound editor
    (``PoseEditor.bound``), and the fake asset session ``_opened_asset``
    builds never binds one -- ``FakeViewer.enter_pose_mode`` (unlike the real
    ``Viewer``'s) only flips ``pose_mode``, since most of this file's asset-
    session tests never look at the editor's own bones. The skeleton editor
    is the first thing here that does.
    """
    ctx, viewer, job_id = _opened_asset(svc, monkeypatch, **rig_meta)
    names = [b["name"] for b in templates.get_template("humanoid").bones]
    viewer.editor.bind(_armature_model(), names)
    viewer.editor.root = next(
        b["name"] for b in templates.get_template("humanoid").bones if b["parent"] is None
    )
    return ctx, viewer, job_id


def test_enter_skeleton_edit_requires_an_open_asset(svc):
    ctx = FakeCtx(svc)
    poser_mode.enter_skeleton_edit(ctx)
    assert any("rigged asset" in msg for msg, _level in ctx.toasts)
    assert poser_mode.ensure(ctx).skeleton_editing is False


def test_enter_skeleton_edit_refuses_over_an_unsaved_pose(svc, monkeypatch):
    ctx, viewer, _job_id = _opened_asset(svc, monkeypatch, bones=_humanoid_bones())
    viewer.editor.dirty = True

    poser_mode.enter_skeleton_edit(ctx)
    assert any("before editing the skeleton" in msg for msg, _level in ctx.toasts)
    assert viewer.editor.mode != "skeleton"
    assert poser_mode.ensure(ctx).skeleton_editing is False


def test_enter_skeleton_edit_refuses_a_rig_with_no_bones(svc, monkeypatch):
    ctx, viewer, _job_id = _opened_asset(svc, monkeypatch)  # default meta: bones=[]

    poser_mode.enter_skeleton_edit(ctx)
    assert any("no readable rig" in msg for msg, _level in ctx.toasts)
    assert poser_mode.ensure(ctx).skeleton_editing is False


def test_enter_skeleton_edit_switches_the_editor_and_the_pane_state(svc, monkeypatch):
    ctx, viewer, _job_id = _opened_asset_for_skeleton(svc, monkeypatch, **_custom_rig_meta())

    poser_mode.enter_skeleton_edit(ctx)
    state = poser_mode.ensure(ctx)
    assert state.skeleton_editing is True
    assert viewer.editor.mode == "skeleton"
    assert len(viewer.editor.draft) == len(_humanoid_bones())


def test_apply_skeleton_submits_edit_skeleton_under_the_rerig_key(svc, monkeypatch):
    ctx, viewer, job_id = _opened_asset_for_skeleton(svc, monkeypatch, **_custom_rig_meta())
    poser_mode.enter_skeleton_edit(ctx)
    before = len(viewer.editor.draft)
    viewer.editor.skel_add_child(viewer.editor.draft_root)

    poser_mode.apply_skeleton(ctx)
    key = f"{poser_mode.ASSET_RERIG_KEY_PREFIX}{job_id}"
    assert key in ctx.submitted
    assert key.startswith("poser-"), "must land on this module's own on_task_done"
    result = ctx.results[key]
    assert result["source_job"] == job_id
    assert result["skeleton"] == "custom"
    # The queued job's own params carry the drafted skeleton -- one more bone
    # than the session started with, from the added child.
    queued = svc.store.get(result["id"])
    assert len(queued["params"]["bones"]) == before + 1


def test_apply_skeleton_records_a_field_addressed_refusal(svc, monkeypatch):
    """``FakeCtx.submit`` runs the call inline, so a raised refusal surfaces
    straight out of :func:`poser_mode.apply_skeleton` here -- the real
    ``TaskRunner`` instead catches it into a ``Done`` and hands it to
    :func:`poser_mode.on_task_failed`, which is exercised directly below."""
    from warlock.service.errors import Invalid

    ctx, viewer, job_id = _opened_asset_for_skeleton(svc, monkeypatch, **_custom_rig_meta())
    poser_mode.enter_skeleton_edit(ctx)
    # ``skeleton.validate_skeleton`` derives the root from the bones' own
    # ``parent`` fields, not from the payload's own ``root`` -- so a second
    # parentless bone, poked straight into the draft rather than through a
    # ``skel_*`` door (none of which can produce this on their own), is what
    # makes it refuse with ``field="root"``.
    viewer.editor.draft.append(
        {"name": "stray-root", "parent": None, "head": [0.0, 0.0, 0.0], "tail": [0.0, 0.0, 1.0]}
    )

    key = f"{poser_mode.ASSET_RERIG_KEY_PREFIX}{job_id}"
    with pytest.raises(Invalid) as excinfo:
        poser_mode.apply_skeleton(ctx)

    poser_mode.on_task_failed(
        ctx, SimpleNamespace(key=key, error=excinfo.value, message=str(excinfo.value))
    )
    state = poser_mode.ensure(ctx)
    assert state.skeleton_error is not None
    assert state.skeleton_error["field"] == "root"


def test_cancel_skeleton_edit_asks_only_when_the_draft_is_dirty(svc, monkeypatch):
    ctx, viewer, _job_id = _opened_asset_for_skeleton(svc, monkeypatch, **_custom_rig_meta())
    poser_mode.enter_skeleton_edit(ctx)

    poser_mode.cancel_skeleton_edit(ctx)
    assert ctx.confirms.asked == [], "a clean draft needs no confirm"
    assert poser_mode.ensure(ctx).skeleton_editing is False
    assert viewer.editor.mode != "skeleton"

    poser_mode.enter_skeleton_edit(ctx)
    viewer.editor.skel_add_child(viewer.editor.draft_root)
    assert viewer.editor.draft_dirty is True

    poser_mode.cancel_skeleton_edit(ctx)
    assert len(ctx.confirms.asked) == 1
    assert poser_mode.ensure(ctx).skeleton_editing is True, "not discarded until confirmed"
    ctx.confirms.asked[0].on_confirm()
    assert poser_mode.ensure(ctx).skeleton_editing is False
    assert viewer.editor.mode != "skeleton"


def test_scrub_and_capture_key_refuse_by_name_while_editing_the_skeleton(svc, monkeypatch):
    ctx, viewer, _job_id = _opened_asset_for_skeleton(svc, monkeypatch, **_custom_rig_meta())
    poser_mode.enter_skeleton_edit(ctx)
    state = poser_mode.ensure(ctx)
    state.frames = [{"bones": {}}]

    poser_mode.scrub(ctx, 0)
    assert any("scrubbing" in msg for msg, _level in ctx.toasts)
    assert state.frame == -1

    poser_mode.capture_key(ctx)
    assert any("capturing a key" in msg for msg, _level in ctx.toasts)


def test_pose_saves_refuse_by_name_while_editing_the_skeleton(svc, monkeypatch):
    ctx, viewer, _job_id = _opened_asset_for_skeleton(svc, monkeypatch, **_custom_rig_meta())
    poser_mode.enter_skeleton_edit(ctx)

    poser_mode.save(ctx)
    poser_mode.save_as(ctx)
    poser_mode.save_pose_to_asset(ctx)
    assert ctx.prompts.asked == [], "no save reached the naming prompt"
    assert poser_mode.SAVE_KEY not in ctx.submitted
    assert sum(
        "before saving a pose" in msg for msg, _level in ctx.toasts
    ) == 3


def test_apply_pose_refuses_while_editing_the_skeleton(svc, monkeypatch):
    """The 2026-09-14 audit's poser-03: New pose and the three Apply buttons
    (the library's own, the asset's own, and a shipped preset's) stayed live
    during a skeleton edit and reposed the mesh the skeleton editor still
    assumes is at rest -- ``enter_skeleton_edit`` resets the armature to rest
    on the way in, and nothing brings it back until the drafted skeleton
    lands or the edit is cancelled."""
    ctx, viewer, _job_id = _opened_asset_for_skeleton(svc, monkeypatch, **_custom_rig_meta())
    poser_mode.enter_skeleton_edit(ctx)
    state = poser_mode.ensure(ctx)
    state.poses = [
        {"id": "lib1", "name": "Lib", "bones": {}, "root_translation": [0.0, 0.0, 0.0]}
    ]
    state.asset_poses = [
        {"id": "asset1", "name": "Asset", "bones": {}, "root_translation": [0.0, 0.0, 0.0]}
    ]
    state.presets = [{"name": "Preset"}]

    poser_mode.new_pose(ctx)
    poser_mode.apply_pose(ctx, "lib1")
    poser_mode.apply_asset_pose(ctx, "asset1")
    poser_mode.apply_preset(ctx, state.presets[0])

    assert viewer.editor.mode == "skeleton", "none of the four doors left skeleton editing"
    assert viewer.editor.current is None, "nothing was applied onto the editor"
    assert ctx.confirms.asked == [], "refused before ever reaching the guard's confirm"
    assert (
        sum("before changing the pose" in msg for msg, _level in ctx.toasts) == 4
    ), "all four doors must say why, not just the first"


def test_rerig_of_a_custom_skeleton_always_confirms_even_with_a_clean_editor(svc, monkeypatch):
    """The generic ``guard`` only asks about an unsaved *pose*; a custom
    skeleton's own shape needs its own warning even over a clean editor."""
    ctx, viewer, job_id = _opened_asset(svc, monkeypatch, **_custom_rig_meta())
    assert not viewer.editor.has_unsaved_edits()

    poser_mode.rerig(ctx, "humanoid")
    assert len(ctx.confirms.asked) == 1
    assert "custom skeleton" in ctx.confirms.asked[0].title.lower()
    key = f"{poser_mode.ASSET_RERIG_KEY_PREFIX}{job_id}"
    assert key not in ctx.submitted, "not submitted until the confirm is answered"
    ctx.confirms.asked[0].on_confirm()
    assert key in ctx.submitted


def test_rerig_of_a_template_skeleton_is_unaffected(svc, monkeypatch):
    """A plain template rig keeps the ordinary guard's behaviour: nothing
    unsaved means no confirm at all."""
    ctx, viewer, job_id = _opened_asset(svc, monkeypatch, bones=_humanoid_bones())
    poser_mode.rerig(ctx, "quadruped")
    assert ctx.confirms.asked == []
    assert f"{poser_mode.ASSET_RERIG_KEY_PREFIX}{job_id}" in ctx.submitted


def test_skeleton_state_resets_on_open_and_close_but_not_on_template_switch(svc, monkeypatch):
    ctx, viewer, job_id = _opened_asset_for_skeleton(svc, monkeypatch, **_custom_rig_meta())
    poser_mode.enter_skeleton_edit(ctx)
    state = poser_mode.ensure(ctx)
    assert state.skeleton_editing is True

    # A template switch (poser-01's own reset) must not silently end an
    # editing session it says nothing about -- ``_reset_for_template`` is not
    # in the call chain for a mid-session skeleton edit at all, but this pins
    # that adding a field there was a deliberate choice, not an oversight.
    import warlock.studio.modes.poser.mode as poser_mode_module

    fields_reset = poser_mode_module._reset_for_template.__code__.co_names
    assert "skeleton_editing" not in fields_reset

    poser_mode.close_asset(ctx)
    assert state.skeleton_editing is False
    assert state.skeleton_error is None


def test_close_asset_confirm_names_the_skeleton_draft_not_the_pose(svc, monkeypatch):
    """The 2026-09-14 audit's poser-04: ``PoseEditor.has_unsaved_edits()``
    folds a posed armature (``dirty``/``moved``) and an open skeleton draft
    (``draft_dirty``) into one flag, and :func:`poser_mode.guard` named the
    confirm "pose changes" either way -- so closing (or opening a different)
    asset while only the skeleton draft was dirty warned about discarding a
    pose that had not, in fact, been touched."""
    ctx, viewer, _job_id = _opened_asset_for_skeleton(svc, monkeypatch, **_custom_rig_meta())
    poser_mode.enter_skeleton_edit(ctx)
    viewer.editor.skel_add_child(viewer.editor.draft_root)
    assert viewer.editor.draft_dirty is True
    assert viewer.editor.dirty is False, "only the skeleton draft is dirty here, not the pose"

    poser_mode.close_asset(ctx)

    assert len(ctx.confirms.asked) == 1
    message = ctx.confirms.asked[0].message
    assert "skeleton changes" in message
    assert "pose changes" not in message

    # open_asset's own confirm goes through the same guard() -- the finding
    # names both doors, and the fix has to live where both of them read it.
    poser_mode.open_asset(ctx, {"id": "some-other-job"})
    assert len(ctx.confirms.asked) == 2
    message = ctx.confirms.asked[1].message
    assert "skeleton changes" in message
    assert "pose changes" not in message


def test_land_rerig_ends_the_skeleton_editing_session_and_announces_a_custom_skeleton(
    svc, monkeypatch
):
    ctx, viewer, job_id = _opened_asset_for_skeleton(svc, monkeypatch, **_custom_rig_meta())
    poser_mode.enter_skeleton_edit(ctx)
    assert poser_mode.ensure(ctx).skeleton_editing is True

    # A second rig job lands on the same source, now recorded as custom with
    # an envelope fallback -- the banner names both facts.
    new_rig = _custom_rig_meta()
    new_rig["weighting"] = "envelope"
    new_rig["weighting_reason"] = "bone-heat weighting failed: non-manifold mesh"
    (ctx.job_dir(job_id) / "rig.json").write_text(
        json.dumps({"version": 1, "template": "humanoid", **new_rig}), "utf-8"
    )

    poser_mode._land_rerig(ctx)
    state = poser_mode.ensure(ctx)
    assert state.skeleton_editing is False
    assert state.skeleton_error is None
    assert any(
        "Custom skeleton from humanoid" in msg and "non-manifold" in msg
        for msg, _level in ctx.toasts
    )


def test_the_skeleton_rename_buffer_does_not_leak_an_uncommitted_edit_across_asset_sessions(
    svc, monkeypatch
):
    """The 2026-09-16 audit, finding poser-03: ``skeleton_rename`` (the rename box's live typing
    buffer) and ``skeleton_rename_for`` (which bone it was seeded for) are the
    only fields in the skeleton-editor's session-state block that ``open_asset``,
    ``close_asset`` and ``_land_rerig`` left untouched -- ``poser_skeleton._rename``
    re-seeds the box only when the *selected bone's name* changes underneath it,
    and every humanoid-template rig shares bone names ("hips", "spine", "head",
    ...). Closing an asset with typed-but-uncommitted rename text left in the
    box, then opening a different asset built from the same template and
    selecting a same-named bone, showed the first asset's abandoned text as
    though it were the new asset's own bone name -- and committing it (Enter,
    or tabbing away) renamed a pivot on the wrong rig for real.
    """
    ctx, _viewer_a, _job_a = _opened_asset_for_skeleton(svc, monkeypatch, **_custom_rig_meta())
    state = poser_mode.ensure(ctx)
    # What poser_skeleton._rename leaves standing on the state after the user
    # typed into the box for "hips" but never pressed Enter or tabbed away.
    state.skeleton_rename = "an uncommitted edit belonging to the first asset"
    state.skeleton_rename_for = "hips"

    job_b = _rigged_job(svc, bones=_humanoid_bones())
    poser_mode.open_asset(ctx, {"id": job_b, "name": "B"})
    assert state.skeleton_rename == "", "open_asset must not carry A's stale buffer onto B"
    assert state.skeleton_rename_for is None

    # Re-arm, and prove close_asset clears it too.
    state.skeleton_rename = "another uncommitted edit"
    state.skeleton_rename_for = "spine"
    poser_mode.close_asset(ctx)
    assert state.skeleton_rename == ""
    assert state.skeleton_rename_for is None

    # And _land_rerig -- the automatic path, with no click behind it to hide
    # a session boundary the way open_asset's and close_asset's guards do.
    ctx2, _viewer_c, _job_c = _opened_asset_for_skeleton(
        svc, monkeypatch, **_custom_rig_meta()
    )
    state2 = poser_mode.ensure(ctx2)
    state2.skeleton_rename = "a third uncommitted edit"
    state2.skeleton_rename_for = "head"
    poser_mode._land_rerig(ctx2)
    assert state2.skeleton_rename == ""
    assert state2.skeleton_rename_for is None


# --- applying ----------------------------------------------------------------


def test_apply_pose_loads_the_record_clean(svc):
    ctx = FakeCtx(svc)
    state = poser_mode.ensure(ctx)
    viewer = ctx.poser_viewer = _bound_viewer()
    record = {
        "id": "0123456789ab",
        "name": "Leap",
        "bones": {"spine": [0.0, 0.0, 0.7071068, 0.7071068]},
        "root_translation": [0.1, 0.0, 0.2],
    }
    state.poses = [record]
    poser_mode.apply_pose(ctx, "0123456789ab")
    assert viewer.editor.current == "0123456789ab"
    assert viewer.editor.dirty is False, "an applied pose is not an unsaved edit"
    assert viewer.editor.root_translation() == pytest.approx([0.1, 0.0, 0.2])


def test_applying_a_pose_after_another_leaves_no_residue(svc):
    """apply_pose resets first, the apply_preset order: set_pose writes only
    the bones the record lists, so a partial record (pre-completeness saves
    exist on disk) applied over a posed editor would keep stale rotations --
    which get_pose() would then save as authored."""
    ctx = FakeCtx(svc)
    state = poser_mode.ensure(ctx)
    viewer = ctx.poser_viewer = _bound_viewer()
    state.poses = [
        {"id": "0123456789ab", "name": "Twist",
         "bones": {"spine": [0.0, 0.0, 0.7071068, 0.7071068]}},
        {"id": "0123456789ac", "name": "Idle", "bones": {"hips": [0.0, 0.0, 0.0, 1.0]}},
    ]
    poser_mode.apply_pose(ctx, "0123456789ab")
    poser_mode.apply_pose(ctx, "0123456789ac")
    assert viewer.editor.pose()["spine"] == pytest.approx([0.0, 0.0, 0.0, 1.0])
    assert viewer.editor.current == "0123456789ac"


def test_apply_preset_resets_the_root(svc):
    """A preset lists rotations only; the reset-first rule is what makes it
    zero-translation rather than inheriting the last pose's offset."""
    ctx = FakeCtx(svc)
    viewer = ctx.poser_viewer = _bound_viewer()
    viewer.editor.set_root_translation([0.0, 0.0, 0.5], dirty=False)
    poser_mode.apply_preset(ctx, {"name": "idle", "bones": {"spine": [0, 0, 0, 1]}})
    assert viewer.editor.root_translation() == [0.0, 0.0, 0.0]


def test_save_pose_to_asset_round_trips_the_root_offset(svc):
    """poser-02, the 2026-09-11 audit: save_pose_to_asset never put
    root_translation in the payload it submitted, and apply_asset_pose never
    restored one -- unlike the shared library's own save/apply pair
    (_payload/apply_pose), which both do. A root offset authored with Move
    root and saved directly onto an asset (rather than into the shared
    library) silently vanished, with no error and nothing on screen to say
    so. Round-tripped through the real service door (svc_rig.save_pose),
    not just asserted present in the payload dict -- that door used to drop
    the field on the floor even when it was sent."""
    job_id = _rigged_job(
        svc, bones=[{"name": b["name"]} for b in templates.get_template("humanoid").bones]
    )
    ctx = FakeCtx(svc)
    state = poser_mode.ensure(ctx)
    state.job_id = job_id
    viewer = ctx.poser_viewer = _bound_viewer()
    viewer.pose_mode = True
    viewer.pose_job_id = job_id
    viewer.editor.set_root_translation([0.1, 0.0, 0.25], dirty=True)

    poser_mode.save_pose_to_asset(ctx)
    assert ctx.prompts.asked, "the name prompt must have been raised"
    ctx.prompts.asked[-1].on_accept("Crouch")

    save_key = f"{poser_mode.ASSET_SAVE_KEY_PREFIX}{job_id}"
    saved = ctx.results[save_key]
    assert saved["root_translation"] == pytest.approx([0.1, 0.0, 0.25]), (
        "the offset must actually be persisted on disk, not merely present "
        "in the payload sent to the service door"
    )
    poser_mode.on_task_done(ctx, SimpleNamespace(key=save_key, result=saved))

    poses_key = f"{poser_mode.ASSET_POSES_KEY_PREFIX}{job_id}"
    poser_mode.on_task_done(
        ctx, SimpleNamespace(key=poses_key, result=ctx.results[poses_key])
    )
    assert state.asset_poses and state.asset_poses[0]["root_translation"] == pytest.approx(
        [0.1, 0.0, 0.25]
    )

    # Move the root away, then apply the saved pose back -- the actual round
    # trip, not just a key present in a dict.
    viewer.editor.set_root_translation([0.0, 0.0, 0.0], dirty=False)
    poser_mode.apply_asset_pose(ctx, saved["id"])
    assert viewer.editor.root_translation() == pytest.approx([0.1, 0.0, 0.25])


# --- saving ------------------------------------------------------------------


def test_dirty_clears_only_when_the_save_lands(svc):
    ctx = FakeCtx(svc)
    viewer = ctx.poser_viewer = _bound_viewer()
    viewer.editor.dirty = True

    poser_mode.save(ctx)  # nothing being edited -> falls through to Save as
    assert len(ctx.prompts.asked) == 1
    ctx.prompts.asked[0].on_accept("Crouch")
    assert poser_mode.SAVE_KEY in ctx.submitted
    assert viewer.editor.dirty is True, "submit is not landing"

    result = ctx.results[poser_mode.SAVE_KEY]
    poser_mode.on_task_done(ctx, SimpleNamespace(key=poser_mode.SAVE_KEY, result=result))
    assert viewer.editor.dirty is False
    assert viewer.editor.current == result["id"]
    # And the library re-reads itself so the new row appears.
    assert ctx.submitted.count(poser_mode.LIST_KEY) == 1


def test_save_over_the_edited_pose_updates_in_place(svc):
    stored = svc_poses.create_library_pose(
        svc,
        {"name": "Crouch", "template": "humanoid", "bones": _full_bones()},
    )
    ctx = FakeCtx(svc)
    state = poser_mode.ensure(ctx)
    state.poses = [stored]
    viewer = ctx.poser_viewer = _bound_viewer()
    viewer.editor.current = stored["id"]
    viewer.editor.dirty = True

    poser_mode.save(ctx)
    assert ctx.prompts.asked == [], "an in-place save asks no name"
    result = ctx.results[poser_mode.SAVE_KEY]
    assert result["id"] == stored["id"]
    on_disk = {p["id"]: p for p in svc_poses.list_library(svc)["poses"]}
    assert on_disk[stored["id"]]["updated"] == result["updated"]


def test_saving_to_the_shared_library_filters_bones_the_template_does_not_have(svc):
    """P4 (2026-09-13): a custom skeleton's own bone is not on the shared
    template and would make ``poselib.validate_record`` refuse the whole save
    outright ("unknown bone"); the library save drops it instead and says so,
    rather than losing the rest of a pose over one bone the template cannot
    place."""
    ctx = FakeCtx(svc)
    poser_mode.ensure(ctx)
    viewer = ctx.poser_viewer = _bound_viewer()
    viewer.editor.dirty = True
    extra = dict(_full_bones())
    extra["tail.001"] = [0.0, 0.0, 0.0, 1.0]
    viewer.get_pose = lambda: extra

    poser_mode.save_as(ctx)
    ctx.prompts.asked[0].on_accept("Crouch")
    result = ctx.results[poser_mode.SAVE_KEY]
    assert "tail.001" not in result["bones"]
    assert any("1 custom bone" in message for message, _level in ctx.toasts)


def test_saving_to_the_shared_library_with_no_extra_bones_toasts_nothing(svc):
    ctx = FakeCtx(svc)
    poser_mode.ensure(ctx)
    viewer = ctx.poser_viewer = _bound_viewer()
    viewer.editor.dirty = True

    poser_mode.save_as(ctx)
    ctx.prompts.asked[0].on_accept("Crouch")
    assert ctx.toasts == []


def test_deleting_the_edited_pose_clears_current(svc):
    stored = svc_poses.create_library_pose(
        svc,
        {"name": "Crouch", "template": "humanoid", "bones": _full_bones()},
    )
    ctx = FakeCtx(svc)
    viewer = ctx.poser_viewer = _bound_viewer()
    viewer.editor.current = stored["id"]
    poser_mode.on_task_done(
        ctx,
        SimpleNamespace(key=f"{poser_mode.DELETE_KEY}:{stored['id']}", result={"ok": True}),
    )
    assert viewer.editor.current is None, "Save must not write to a record that is gone"


# --- the guard ---------------------------------------------------------------


def test_guard_asks_only_for_the_poser_session(svc):
    from warlock.studio.panes import pose_panel

    ctx = FakeCtx(svc)
    ctx.poser_viewer = _bound_viewer()
    ctx.poser_viewer.editor.dirty = True

    ran: list[str] = []
    # The inspector's guard reads the *shared* viewer, which holds nothing --
    # so with a dirty Poser session there is still exactly one question.
    assert pose_panel.guard(ctx, "quit", lambda: ran.append("pose")) is True
    assert poser_mode.guard(ctx, "quit", lambda: ran.append("poser")) is False
    assert ran == ["pose"]
    assert len(ctx.confirms.asked) == 1


def test_guard_proceeds_with_no_session(svc):
    ctx = FakeCtx(svc)
    ran: list[str] = []
    assert poser_mode.guard(ctx, "quit", lambda: ran.append("go")) is True
    assert ran == ["go"]


# --- Mirror, behind the same guard -------------------------------------------
#
# Mirror rewrites every rotation and this mode has no undo at all: it was the
# last control that could discard an unsaved pose with no way back.


def _mirrored_viewer():
    viewer = _bound_viewer()
    viewer.editor.mirror_pairs = [["upper_arm.L", "upper_arm.R"]]
    return viewer


def _left_arm(viewer):
    return list(viewer.editor.pose()["upper_arm.L"])


def test_mirror_over_unsaved_edits_asks_first_and_moves_nothing(svc):
    ctx = FakeCtx(svc)
    viewer = _mirrored_viewer()
    ctx.poser_viewer = viewer
    viewer.editor.apply({"upper_arm.R": [0.0, 0.0, 0.7071068, 0.7071068]})
    before = _left_arm(viewer)

    assert poser_mode.guard(ctx, "mirror the pose", viewer.mirror) is False
    assert len(ctx.confirms.asked) == 1
    assert "mirror the pose" in ctx.confirms.asked[0].message
    assert _left_arm(viewer) == before, "nothing moved before the answer"

    ctx.confirms.asked[0].on_confirm()
    assert _left_arm(viewer) != before


def test_mirror_on_a_clean_editor_just_runs(svc):
    ctx = FakeCtx(svc)
    viewer = _mirrored_viewer()
    ctx.poser_viewer = viewer
    viewer.editor.apply({"upper_arm.R": [0.0, 0.0, 0.7071068, 0.7071068]}, dirty=False)
    before = _left_arm(viewer)

    assert poser_mode.guard(ctx, "mirror the pose", viewer.mirror) is True
    assert ctx.confirms.asked == []
    assert _left_arm(viewer) != before


@pytest.mark.parametrize(
    ("module", "guard_name"),
    [("modes.poser.ui.panes.controls", "poser_mode.guard"), ("panes.pose_panel", "guard")],
)
def test_both_mirror_buttons_go_through_their_pane_s_guard(module, guard_name):
    """Source-scanned rather than clicked: the button lives inside an imgui
    frame, and what is being pinned is that no bare ``viewer.mirror()`` call
    comes back -- which is a fact about the text."""
    import importlib
    import inspect

    source = inspect.getsource(importlib.import_module(f"warlock.studio.{module}"))
    assert f'{guard_name}(ctx, "mirror the pose", viewer.mirror)' in source
    assert "viewer.mirror()" not in source


# --- task-key routing --------------------------------------------------------


def test_every_poser_task_key_is_prefixed_poser():
    """The prefix is the routing, so it is not a naming convention: a Poser key
    that did not start with ``poser-`` would be handed to another mode's
    handler by ``main._on_task_done``."""
    keys = [
        poser_mode.LIST_KEY,
        poser_mode.SAVE_KEY,
        poser_mode.DELETE_KEY,
        poser_mode.DUPLICATE_KEY,
        poser_mode.RENAME_KEY,
        poser_mode.PREVIEW_KEY_PREFIX,
        poser_mode.ASSET_RERIG_KEY_PREFIX,
    ]
    assert all(k.startswith("poser-") for k in keys), keys


def test_poser_results_are_claimed_before_the_asset_pose_branches():
    """``"poser-save"`` also starts with ``"pose-"``. Nothing but the order of
    two ``if`` statements keeps a Poser result out of the inspector's pose
    handler, and the fix if they were ever swapped is not obvious from either
    site -- so the order is the pin.
    """
    import inspect

    from warlock.studio import main

    source = inspect.getsource(main.App._on_task_done)
    poser = source.index('key.startswith("poser-")')
    assert poser < source.index('key.startswith("pose-library:")')
    assert poser < source.index('key.startswith("pose-")')


# --- the preview binding -----------------------------------------------------


def test_sync_preview_binds_once_and_not_again(svc, tmp_path):
    """The parse is dispatched, never run inline (create-04, the 2026-09-11
    audit): a preview build lands on an arbitrary frame with no click behind
    it, so ``sync_preview`` only ever submits the parse here -- landing it is
    ``on_task_done``'s job, exercised the same way ``test_frame_thread_doors``
    proves the other nine doors."""
    ctx = FakeCtx(svc)
    state = poser_mode.ensure(ctx)
    glb = tmp_path / "humanoid.glb"
    glb.write_bytes(b"glb")
    state.preview_path = glb
    state.preview_template = "humanoid"

    ctx.poser_viewer = viewer = FakeViewer()
    viewer.editor.model = None
    assert poser_mode.sync_preview(ctx, viewer) is False
    key = poser_mode.PREVIEW_LOAD_KEY
    assert key in ctx.submitted
    assert viewer.loaded == [], "the parse must not touch the viewer directly"

    poser_mode.on_task_done(
        ctx, SimpleNamespace(key=key, result=ctx.results[key], tag=ctx.tags[key])
    )
    assert viewer.loaded == [glb]
    assert viewer.token == "poser:humanoid"
    assert viewer.editor.root == templates.get_template("humanoid").root
    assert viewer.framed is not None

    assert poser_mode.sync_preview(ctx, viewer) is True
    assert viewer.loaded == [glb], "already showing it: no reload"


def test_a_preview_for_a_switched_away_template_never_binds(svc, tmp_path):
    ctx = FakeCtx(svc)
    state = poser_mode.ensure(ctx)
    glb = tmp_path / "fish.glb"
    glb.write_bytes(b"glb")
    state.preview_path = glb
    state.preview_template = "fish"  # but state.template is humanoid

    viewer = FakeViewer()
    assert poser_mode.sync_preview(ctx, viewer) is False
    assert viewer.loaded == []


def test_preview_bounds_cover_the_whole_skeleton():
    lo, hi = poser_mode.preview_bounds("humanoid")
    assert all(b > a for a, b in zip(lo, hi, strict=True))
    # The armature is one character-height tall; the glTF vertical axis is Y.
    assert hi[1] - lo[1] >= 1.0
    # And tails are included: the head bone's tail reaches z=1.0 exactly, so a
    # joint-only box would stop short of it.
    assert hi[1] > 1.0 - 1e-6


# --- keys --------------------------------------------------------------------


def test_escape_deselects_the_joint(svc):
    import pygame

    ctx = FakeCtx(svc)
    viewer = ctx.poser_viewer = _bound_viewer()
    viewer.editor.selected = "spine"
    event = SimpleNamespace(type=pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0)
    assert poser_mode.handle_key(ctx, event) is True
    assert viewer.editor.selected is None
    assert poser_mode.handle_key(ctx, event) is False
    # An unbound letter still falls through.
    other = SimpleNamespace(type=pygame.KEYDOWN, key=pygame.K_g, mod=0)
    assert poser_mode.handle_key(ctx, other) is False


def test_poser_has_clays_view_keys_under_the_same_ctrl(svc):
    """It had none: the only way to look at a pose from the side was to orbit
    there by hand, and no way back once the model was off screen. And once it
    had them it fired them on the *bare* digit, where Clay wants Ctrl -- the
    2026-09-05 consistency pass: one chord, both viewports."""
    import pygame

    from warlock.studio.modes.clay import mode as clay_mode

    ctx = FakeCtx(svc)
    viewer = ctx.poser_viewer = _bound_viewer()

    def press(key, mod=0):
        return poser_mode.handle_key(
            ctx, SimpleNamespace(type=pygame.KEYDOWN, key=key, mod=mod)
        )

    at_rest = (viewer.camera.theta, viewer.camera.phi, viewer.camera.orthographic)
    assert press(pygame.K_1) is False
    assert press(pygame.K_5) is False
    assert (viewer.camera.theta, viewer.camera.phi, viewer.camera.orthographic) == at_rest

    assert press(pygame.K_1, pygame.KMOD_CTRL) is True
    front = (viewer.camera.theta, viewer.camera.phi)
    assert press(pygame.K_1, pygame.KMOD_CTRL | pygame.KMOD_SHIFT) is True
    assert (viewer.camera.theta, viewer.camera.phi) != front

    ortho = viewer.camera.orthographic
    assert press(pygame.K_5, pygame.KMOD_CTRL) is True
    assert viewer.camera.orthographic is not ortho

    assert press(pygame.K_f) is True

    # The table is Clay's, read rather than restated, so the two viewports
    # cannot come to disagree about which number is the front.
    assert set(clay_mode.AXIS_VIEW_KEYS) == {"1", "3", "7"}


def test_the_mode_is_wired_into_the_switch():
    from warlock.studio import modes

    assert "poser" in modes.KEYS
    assert "poser" in modes.WORK_MODES
    assert "poser" in modes.WORKSPACE_MODES
    # And in the rail's workspace group, which is the navigation the app
    # actually draws (the UI redesign, wave 3). This used to assert that the derived
    # ``GROUP_BREAKS`` still collapsed to two; the grouping is written out now,
    # so what a new mode has to do is *be in a group*.
    assert "poser" in modes.RAIL_GROUPS[1]


# --- crash recovery ------------------------------------------------------------


def test_a_recovered_pose_restores_joint_corrections(tmp_path):
    """``_pose_payload`` writes ``moved`` and ``mode``; an adopt that applied
    only bones and root translation recovered a crash mid-joint-placement as a
    rest pose under a success toast, the corrections silently dropped."""
    viewer = _bound_viewer()
    editor = viewer.editor
    editor.enter_joints_mode()
    editor.move_handle("hips", editor.home["hips"] + m3.vec3(0.1, 0.0, 0.2))
    assert editor.moved, "the session holds a correction"
    moved = {name: list(delta) for name, delta in editor.moved.items()}

    slot = poser_mode._PoseSlot(viewer, editor, "poser")
    path = tmp_path / "poser.pose.json"
    path.write_bytes(poser_mode._pose_payload(slot))

    ctx = FakeCtx()
    ctx.poser_viewer = _bound_viewer()
    assert poser_mode._journal_adopt(ctx, path, {}) is True
    recovered = ctx.poser_viewer.editor
    assert recovered.mode == "joints"
    assert recovered.moved == moved
    assert recovered.has_unsaved_edits()


def _skeleton_rig_for_journal():
    bones = [dict(b) for b in templates.get_template("humanoid").bones]
    root = next(b["name"] for b in bones if b["parent"] is None)
    return {"bones": bones, "root": root, "mirror_pairs": []}


def test_a_recovered_skeleton_draft_reenters_skeleton_mode(tmp_path):
    """P7 (2026-09-13): a skeleton draft is not a pose -- ``_pose_payload``
    carries ``draft``/``draft_pairs``/``draft_root`` for it, and a recovered
    one has to land back in skeleton mode, not as a rest pose with the edit
    silently dropped."""
    viewer = _bound_viewer()
    viewer.enter_skeleton_mode(_skeleton_rig_for_journal())
    new_name = viewer.editor.skel_add_child(viewer.editor.draft_root)
    assert viewer.editor.draft_dirty is True

    slot = poser_mode._PoseSlot(viewer, viewer.editor, "poser")
    path = tmp_path / "poser.pose.json"
    path.write_bytes(poser_mode._pose_payload(slot))

    ctx = FakeCtx()
    ctx.poser_viewer = target = _bound_viewer()
    assert poser_mode._journal_adopt(ctx, path, {}) is True
    recovered = target.editor
    assert recovered.mode == "skeleton"
    assert recovered.draft_dirty is True
    assert any(b["name"] == new_name for b in recovered.draft)
    assert poser_mode.ensure(ctx).skeleton_editing is True


def test_an_invalid_recovered_skeleton_draft_keeps_the_journal_file_and_warns(tmp_path):
    """A draft re-validated on the way back in and found unusable (here: two
    parentless bones, which ``skel_*`` can never itself produce but a
    hand-edited recovery file or a stale rig might) is declined -- kept, not
    silently discarded, the rule every other adopt refusal here follows."""
    viewer = _bound_viewer()
    viewer.enter_skeleton_mode(_skeleton_rig_for_journal())
    viewer.editor.draft.append(
        {"name": "stray-root", "parent": None, "head": [0.0, 0.0, 0.0], "tail": [0.0, 0.0, 1.0]}
    )
    viewer.editor.draft_dirty = True

    slot = poser_mode._PoseSlot(viewer, viewer.editor, "poser")
    path = tmp_path / "poser.pose.json"
    path.write_bytes(poser_mode._pose_payload(slot))

    ctx = FakeCtx()
    ctx.poser_viewer = _bound_viewer()
    assert poser_mode._journal_adopt(ctx, path, {}) is False
    assert path.exists(), "a declined adopt must not delete the recovery file itself"
    assert any("no longer usable" in msg for msg, _level in ctx.toasts)
    assert ctx.poser_viewer.editor.mode != "skeleton"
    assert poser_mode.ensure(ctx).skeleton_editing is False


# --- the clip editor ----------------------------------------------------------
#
# The Troupe programme's clip-authoring half. The armature *is* the editor
# here -- a key
# is authored by posing the preview and putting it back -- so what these pin is
# the part that is not the pose editor: the timing model, and the places a clip
# can be edited into something the renderer would refuse.


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


def _clip_ctx():
    ctx = FakeCtx()
    ctx.poser_viewer = _bound_viewer()
    state = poser_mode.ensure(ctx)
    state.clips_dirty_flag = False
    poser_mode.adopt_clips(ctx, _clip_library())
    return ctx, state


def _turned():
    """A rotation that is not the identity, spelled once."""
    return [0.0, 0.3894183, 0.0, 0.9210610]


def test_adopting_a_library_opens_its_first_clip_and_expands_it():
    ctx, state = _clip_ctx()
    assert state.clip == "walk"
    assert state.clips_unsaved is False
    # Three closed segments of two frames each.
    assert len(state.frames) == 6


def test_selecting_a_key_puts_its_pose_on_the_armature():
    """And it goes through ``apply_preset``, so the bones the key does *not*
    name go back to rest -- otherwise loading a walk key after an attack key
    leaves the sword arm up."""
    ctx, state = _clip_ctx()
    editor = ctx.poser_viewer.editor
    editor.apply({"head": _turned()}, dirty=True)
    poser_mode.select_key(ctx, 0)
    # The armature holds an unsaved edit, so the door asks first (the
    # 2026-09-02 review's finding 6); the click lands once it is answered.
    assert len(ctx.confirms.asked) == 1
    ctx.confirms.asked[0].on_confirm()
    assert state.key_index == 0
    assert state.frame == -1
    assert editor.pose()["head"] == pytest.approx([0.0, 0.0, 0.0, 1.0])


def test_capturing_writes_the_armature_back_into_the_selected_key():
    ctx, state = _clip_ctx()
    editor = ctx.poser_viewer.editor
    poser_mode.select_key(ctx, 1)
    editor.apply({"spine": _turned()}, dirty=True)
    poser_mode.capture_key(ctx)

    assert state.clips_unsaved is True
    assert state.key_pose("B")["bones"]["spine"] == pytest.approx(_turned())


def test_capturing_an_in_between_frame_is_refused_by_name():
    """The armature shows an interpolated pose while scrubbing, and storing
    that into a key would silently replace an authored pose with a computed
    one. Refused rather than snapped to the nearest key, which is the same
    mistake with the evidence removed."""
    ctx, state = _clip_ctx()
    poser_mode.scrub(ctx, 3)
    before = dict(state.key_pose("A")["bones"])
    poser_mode.capture_key(ctx)
    assert state.key_pose("A")["bones"] == before
    assert state.clips_unsaved is False
    assert any("in-between" in msg for msg, _kind in ctx.toasts)


def test_scrubbing_pushes_no_undo_steps():
    """It runs once per frame of a slider drag. A step per frame is a stack
    full of one gesture -- the rule ``rotate_selected`` already follows."""
    ctx, state = _clip_ctx()
    editor = ctx.poser_viewer.editor
    poser_mode.select_key(ctx, 0)
    head = editor.history.head
    for frame in range(len(state.frames)):
        poser_mode.scrub(ctx, frame)
    assert editor.history.head == head


def test_closing_or_opening_a_clip_resizes_its_timing():
    """An open clip of N keys has N-1 steps and a closed one has N. Resized
    here rather than left for the save to refuse, because the count is
    *derived*: there is no other value it could take."""
    ctx, state = _clip_ctx()
    poser_mode.set_closed(ctx, False)
    assert state.open_clip()["segments"] == [2, 2]
    poser_mode.set_closed(ctx, True)
    assert state.open_clip()["segments"] == [2, 2, 2]


# --- the frame-time control (schema v3's duration_ms) ------------------------


def test_frame_time_snaps_to_the_library_step():
    """Snapped rather than refused -- ``set_segment``'s own precedent: a typed
    83 lands on 80, not an error toast over one keystroke."""
    ctx, state = _clip_ctx()
    poser_mode.set_duration(ctx, 83)
    assert state.open_clip()["duration_ms"] == 80
    poser_mode.set_duration(ctx, 3)
    assert state.open_clip()["duration_ms"] == cliplib.MIN_CLIP_DURATION_MS
    poser_mode.set_duration(ctx, 5000)
    assert state.open_clip()["duration_ms"] == cliplib.MAX_CLIP_DURATION_MS


def test_the_frame_time_control_writes_duration_ms_and_a_save_round_trips(monkeypatch):
    from warlock.service import clips as svc_clips

    ctx, state = _clip_ctx()
    poser_mode.set_duration(ctx, 250)
    assert state.open_clip()["duration_ms"] == 250
    assert state.clips_unsaved is True

    def fake_save(svc, template, payload):
        return {**payload, "template": template, "edited": True}

    monkeypatch.setattr(svc_clips, "save", fake_save)
    poser_mode.save_clips(ctx)
    sent = ctx.results[poser_mode.CLIPS_SAVE_KEY]
    saved = next(c for c in sent["clips"] if c["name"] == "walk")
    assert saved["duration_ms"] == 250

    poser_mode.on_task_done(ctx, SimpleNamespace(key=poser_mode.CLIPS_SAVE_KEY, result=sent))
    assert state.clips_unsaved is False
    assert state.open_clip()["duration_ms"] == 250


def test_a_frame_time_edit_is_discarded_when_the_library_is_re_adopted():
    """This used to be named as an undo test, and Poser's clip editor has none
    -- ``revert_clips``'s own docstring says so plainly ("this mode has no
    undo"), the same fact that keeps ``set_easing``/``set_segment`` from
    pushing a step of their own. What this actually proves is narrower and
    still true: re-adopting a library (what a landed revert, or a landed
    CLIPS_KEY refresh, both do) discards whatever the working copy held,
    which is not the same claim as "the edit can be undone" -- there is no
    door here that takes you from the re-adopted state back to the edit."""
    ctx, state = _clip_ctx()
    original = state.open_clip()["duration_ms"]
    poser_mode.set_duration(ctx, original + 50)
    assert state.open_clip()["duration_ms"] == original + 50
    assert state.clips_unsaved is True

    poser_mode.adopt_clips(ctx, _clip_library())
    assert state.open_clip()["duration_ms"] == original
    assert state.clips_unsaved is False


def test_play_speed_follows_the_clips_frame_time():
    """The Timing section's own "fps" hint, and the arithmetic
    ``clips.animation_tracks``' bake ``step`` inverts."""
    assert poser_mode.clip_fps(80) == pytest.approx(12.5)
    assert poser_mode.clip_fps(100) == pytest.approx(10.0)
    assert poser_mode.clip_fps(0) == 0.0
    assert poser_mode.clip_fps(None) == 0.0

    ctx, state = _clip_ctx()
    poser_mode.set_duration(ctx, 200)
    assert poser_mode.clip_fps(state.open_clip()["duration_ms"]) == pytest.approx(5.0)


def test_saving_keeps_the_provisional_flag(monkeypatch):
    """``provisional``/``source`` ride in the working copy untouched, so a
    save of a shipped provisional clip (attack_02, cast, fall, hit, death)
    cannot silently drop the flag an animator's pass still owes."""
    from warlock.service import clips as svc_clips

    ctx, state = _clip_ctx()
    record = state.open_clip()
    record["provisional"] = True
    record["source"] = {"imported": True}
    poser_mode.set_duration(ctx, 90)

    def fake_save(svc, template, payload):
        return {**payload, "template": template, "edited": True}

    monkeypatch.setattr(svc_clips, "save", fake_save)
    poser_mode.save_clips(ctx)
    sent = ctx.results[poser_mode.CLIPS_SAVE_KEY]
    saved = next(c for c in sent["clips"] if c["name"] == "walk")
    assert saved["provisional"] is True
    assert saved["source"] == {"imported": True}
    assert saved["duration_ms"] == 90


def test_moving_a_key_carries_its_own_segment():
    """A segment is the step *out of* a key, so reordering without it would
    move the poses and leave the timing behind -- which reads as the reorder
    having corrupted the clip."""
    ctx, state = _clip_ctx()
    state.open_clip()["segments"] = [1, 5, 9]
    poser_mode.move_key(ctx, 0, 1)
    assert state.open_clip()["keys"] == ["B", "A", "C"]
    assert state.open_clip()["segments"] == [5, 1, 9]
    assert state.key_index == 1


def test_removing_a_key_leaves_the_pose_in_the_library():
    """Two different things: another clip may use that pose, and a delete that
    silently reached into the shared list would have a blast radius the button
    does not describe."""
    ctx, state = _clip_ctx()
    poser_mode.remove_key(ctx, 1)
    assert state.open_clip()["keys"] == ["A", "C"]
    assert state.key_pose("B") is not None


def test_a_clip_cannot_be_cut_below_two_keys():
    ctx, state = _clip_ctx()
    poser_mode.remove_key(ctx, 0)
    poser_mode.remove_key(ctx, 0)
    assert len(state.open_clip()["keys"]) == 2
    assert any("at least" in msg for msg, _kind in ctx.toasts)


def test_a_new_key_is_authored_from_the_armature_and_inserted():
    ctx, state = _clip_ctx()
    editor = ctx.poser_viewer.editor
    poser_mode.select_key(ctx, 0)
    editor.apply({"head": _turned()}, dirty=True)
    poser_mode.new_key(ctx, "A crouch")

    assert state.key_pose("A crouch") is not None
    assert state.open_clip()["keys"] == ["A", "A crouch", "B", "C"]
    assert len(state.open_clip()["segments"]) == 4, "a closed clip gains a step"


def test_a_duplicate_key_name_is_refused():
    """Names are what a clip references by, so two of them is an ambiguity
    rather than a duplicate."""
    ctx, state = _clip_ctx()
    poser_mode.new_key(ctx, "B")
    assert any("already exists" in msg for msg, _kind in ctx.toasts)
    assert state.open_clip()["keys"] == ["A", "B", "C"]


def test_a_clip_that_will_not_expand_empties_the_scrubber_instead_of_raising():
    """Mid-edit the segments briefly do not match the keys, which is ordinary.
    The reason is shown; Save is what actually refuses."""
    ctx, state = _clip_ctx()
    state.open_clip()["segments"] = [2]
    poser_mode.rebuild_frames(ctx)
    assert state.frames == []
    assert state.clips_error


def test_a_pump_never_reloads_over_unsaved_edits():
    ctx, state = _clip_ctx()
    state.clips_unsaved = True
    state.clips_dirty_flag = True
    poser_mode.clips_pump(ctx)
    assert poser_mode.CLIPS_KEY not in ctx.submitted
    assert state.clips_dirty_flag is True


def test_unsaved_clears_on_the_landing_and_never_at_submit():
    """A failed write has to leave the guard standing -- the same rule the pose
    save follows."""
    ctx, state = _clip_ctx()
    state.clips_unsaved = True
    poser_mode.on_task_failed(
        ctx, SimpleNamespace(key=poser_mode.CLIPS_SAVE_KEY, message="nope")
    )
    assert state.clips_unsaved is True

    poser_mode.on_task_done(
        ctx,
        SimpleNamespace(
            key=poser_mode.CLIPS_SAVE_KEY, result={**_clip_library(), "edited": True}
        ),
    )
    assert state.clips_unsaved is False
    assert state.clips["edited"] is True


def test_a_landing_for_another_template_is_ignored():
    """The ``viewer.path`` idiom: an answer arriving after the user switched
    skeletons simply never binds."""
    ctx, state = _clip_ctx()
    state.clips_unsaved = True
    poser_mode.on_task_done(
        ctx,
        SimpleNamespace(
            key=poser_mode.CLIPS_KEY, result={**_clip_library(), "template": "quadruped"}
        ),
    )
    assert state.clips_unsaved is True
    assert state.clips["template"] == "humanoid"


def test_a_revert_asked_with_unsaved_edits_is_adopted_when_it_lands(monkeypatch):
    """The pre-existing defect: ``on_task_done``'s shared CLIPS_KEY/
    CLIPS_SAVE_KEY branch refuses to adopt a landed result whenever
    ``state.clips_unsaved`` and ``clips_touch_serial != clips_save_serial`` --
    a guard written for :func:`poser_mode.save_clips`, which records the
    serial before submitting so its own answer is never mistaken for a stale
    one. :func:`poser_mode.revert_clips` submits under the identical
    ``CLIPS_SAVE_KEY`` but never recorded that serial, so a Revert asked while
    there *were* unsaved edits -- the one case Revert exists for -- landed
    with the touch serial still ahead of whatever a previous save (or the
    default 0) had left in ``clips_save_serial``, was silently refused, and
    left ``clips_unsaved`` stuck True forever after a Revert the user had
    just confirmed. ``set_easing`` (rather than the newer ``set_duration``)
    is what dirties the working copy here, so this reaches ``revert_clips``
    and ``on_task_done`` exactly as they existed when the defect was live."""
    from warlock.service import clips as svc_clips

    ctx, state = _clip_ctx()
    poser_mode.set_easing(ctx, "ease_in")
    assert state.clips_unsaved is True

    monkeypatch.setattr(svc_clips, "revert", lambda svc, template: _clip_library())

    poser_mode.revert_clips(ctx)
    assert len(ctx.confirms.asked) == 1, "revert asks before discarding unsaved edits"
    ctx.confirms.asked[0].on_confirm()

    sent = ctx.results[poser_mode.CLIPS_SAVE_KEY]
    poser_mode.on_task_done(ctx, SimpleNamespace(key=poser_mode.CLIPS_SAVE_KEY, result=sent))

    assert state.clips_unsaved is False
    assert state.open_clip()["easing"] == _clip_library()["clips"][0]["easing"]


def test_onion_ghosts_are_the_neighbouring_keys_not_the_neighbouring_frames():
    """A keyframe is judged against the keys it steps *between*; the frames
    either side of the playhead are what the scrubber is for."""
    ctx, state = _clip_ctx()
    ctx.poser_viewer.onion = []
    poser_mode.select_key(ctx, 1)
    poser_mode.set_onion(ctx, True)
    assert ctx.poser_viewer.onion == [
        {"hips": [0.0, 0.0, 0.0, 1.0]},
        {"head": [0.0, 0.0, 0.0, 1.0]},
    ]


def test_a_looping_clip_wraps_its_first_and_last_keys():
    """The comparison a looping walk actually needs, and the one that is
    easiest to get wrong by hand."""
    ctx, state = _clip_ctx()
    ctx.poser_viewer.onion = []
    poser_mode.set_onion(ctx, True)
    poser_mode.select_key(ctx, 0)
    before, after = ctx.poser_viewer.onion
    assert before == {"head": [0.0, 0.0, 0.0, 1.0]}, "wraps to the last key"
    assert after == {"spine": [0.0, 0.0, 0.0, 1.0]}


def test_an_open_clip_has_no_ghost_before_its_first_key():
    ctx, state = _clip_ctx()
    ctx.poser_viewer.onion = []
    poser_mode.set_closed(ctx, False)
    poser_mode.set_onion(ctx, True)
    poser_mode.select_key(ctx, 0)
    assert ctx.poser_viewer.onion[0] == {}


def test_scrubbing_clears_the_ghosts_and_coming_back_restores_them():
    """The live skeleton is an in-between pose while scrubbing, so ghosts of
    the neighbouring *keys* around it would be three poses on screen with no
    stated relationship between them."""
    ctx, state = _clip_ctx()
    ctx.poser_viewer.onion = []
    poser_mode.select_key(ctx, 1)
    poser_mode.set_onion(ctx, True)
    assert ctx.poser_viewer.onion

    poser_mode.scrub(ctx, 3)
    assert ctx.poser_viewer.onion == []

    poser_mode.apply_key(ctx)
    assert ctx.poser_viewer.onion


def test_turning_onion_skin_off_clears_the_viewer():
    ctx, state = _clip_ctx()
    poser_mode.set_onion(ctx, True)
    assert ctx.poser_viewer.onion
    poser_mode.set_onion(ctx, False)
    assert ctx.poser_viewer.onion == []


# --- importing a clip ---------------------------------------------------------
#
# ``service.clip_import.analyse`` (a Blender subprocess plus a pure convert) is
# faked wholesale here: every one of these tests is about what
# ``poser_mode.import_clip``/``adopt_imported_clips`` do with its answer, never
# about the sampling itself, which ``tests/test_clip_import_service.py`` and
# ``tests/test_clip_import_blender.py`` already own.


def _import_result(
    template: str = "humanoid",
    name: str = "run",
    *,
    poses: dict[str, Any] | None = None,
    keys: list[str] | None = None,
    report_extra: dict[str, Any] | None = None,
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One ``service.clip_import.analyse``-shaped answer, one action.

    Shaped exactly like ``cliptransfer.transfer``'s own return -- ``{"clip",
    "poses", "report"}`` -- because that is what a fake ``analyse`` hands back
    verbatim, one entry per source action.
    """
    poses = poses if poses is not None else {
        f"{name} k00": {"bones": {"hips": [0.0, 0.0, 0.0, 1.0]}},
        f"{name} k01": {"bones": {"hips": [0.0, 0.3894183, 0.0, 0.9210610]}},
    }
    keys = keys if keys is not None else list(poses.keys())
    report = {
        "map": "mixamo",
        "left_at_rest": [],
        "ignored": [],
        "loop": {"closed": False, "residual_deg": 0.0},
        "frames": 24,
        "keys": len(keys),
        "root_motion": "in_place",
    }
    if report_extra:
        report.update(report_extra)
    clip: dict[str, Any] = {
        "name": name,
        "keys": keys,
        "segments": [1] * max(len(keys) - 1, 1),
        "closed": False,
        "easing": "linear",
        "duration_ms": 100,
    }
    if source is not None:
        clip["source"] = source
    return {"template": template, "clips": [{"clip": clip, "poses": poses, "report": report}]}


def test_import_clip_asks_for_a_file_on_the_task_thread(tmp_path, monkeypatch):
    """``troupe_mode.export_package``'s arrangement: a blocking OS picker on
    the frame thread freezes the window behind it, so it has to be asked from
    inside the submitted task, not before ``ctx.submit`` is even called."""
    from warlock.service import clip_import as svc_clip_import
    from warlock.studio import dialogs

    ctx = _ThreadedCtx(tmp_path)
    state = poser_mode.ensure(ctx)
    state.clips = {"template": "humanoid", "clips": [{"name": "walk"}], "poses": []}

    threads: dict[str, str] = {}
    picked = tmp_path / "mixamo_run.fbx"

    def fake_open_file(title, filters):
        assert title == "Import a clip"
        assert any("fbx" in p.lower() for p in filters)
        threads["dialog"] = threading.current_thread().name
        return picked

    def fake_analyse(svc, template, path):
        threads["analyse"] = threading.current_thread().name
        assert Path(path) == picked
        return {"template": template, "clips": []}

    monkeypatch.setattr(dialogs, "open_file", fake_open_file)
    monkeypatch.setattr(svc_clip_import, "analyse", fake_analyse)

    assert poser_mode.import_clip(ctx) is True
    assert ctx.submitted == [poser_mode.CLIP_IMPORT_KEY]
    assert threads["dialog"] == WORKER
    assert threads["analyse"] == WORKER
    assert ctx.results[poser_mode.CLIP_IMPORT_KEY]["source_name"] == "mixamo_run.fbx"


def test_a_cancelled_import_changes_nothing(monkeypatch):
    from warlock.studio import dialogs

    ctx, state = _clip_ctx()
    before = json.loads(json.dumps(state.clips))
    monkeypatch.setattr(dialogs, "open_file", lambda *a, **k: None)

    assert poser_mode.import_clip(ctx) is True
    assert ctx.results[poser_mode.CLIP_IMPORT_KEY] is None

    poser_mode.on_task_done(
        ctx, SimpleNamespace(key=poser_mode.CLIP_IMPORT_KEY, result=None)
    )
    assert state.clips == before
    assert state.clips_unsaved is False
    assert ctx.toasts == []


def test_an_imported_clip_joins_the_working_copy_unsaved_and_selected():
    ctx, state = _clip_ctx()
    result = _import_result(name="run")
    result["source_name"] = "mixamo_run.fbx"

    poser_mode.on_task_done(
        ctx, SimpleNamespace(key=poser_mode.CLIP_IMPORT_KEY, result=result)
    )

    assert state.clips_unsaved is True
    assert state.clip == "run"
    names = [c["name"] for c in state.clips["clips"]]
    assert names == ["walk", "run"]
    assert state.key_pose("run k00") is not None
    assert any(
        "Imported 1 clip(s) from mixamo_run.fbx" in msg for msg, _kind in ctx.toasts
    )


def test_an_imported_clip_name_clash_is_renamed_not_overwritten():
    ctx, state = _clip_ctx()
    original_keys = list(state.open_clip()["keys"])
    result = _import_result(name="walk")  # clashes with the shipped "walk"
    result["source_name"] = "mixamo_walk.fbx"

    poser_mode.adopt_imported_clips(ctx, result)

    names = [c["name"] for c in state.clips["clips"]]
    assert names.count("walk") == 1, "the existing clip must not be replaced"
    assert "walk_2" in names
    original = next(c for c in state.clips["clips"] if c["name"] == "walk")
    assert original["keys"] == original_keys, "the original clip's own keys are untouched"
    assert state.clip == "walk_2", "the newly imported clip is the one selected"


def test_imported_pose_names_never_overwrite_working_copy_poses():
    ctx, state = _clip_ctx()
    original_a = dict(state.key_pose("A")["bones"])
    result = _import_result(
        name="run",
        poses={"A": {"bones": {"spine": [0.0, 0.3894183, 0.0, 0.9210610]}}},
        keys=["A"],
    )
    result["source_name"] = "mixamo_run.fbx"

    poser_mode.adopt_imported_clips(ctx, result)

    assert state.key_pose("A")["bones"] == original_a, "the working copy's own pose is untouched"
    assert state.key_pose("A 2") is not None, "the imported pose is renamed instead"
    assert state.key_pose("A 2")["bones"] == {"spine": [0.0, 0.3894183, 0.0, 0.9210610]}
    imported = next(c for c in state.clips["clips"] if c["name"] == "run")
    assert imported["keys"] == ["A 2"], "the clip's own key list follows the rename"


def test_import_clip_is_disabled_without_blender_with_a_reason():
    from warlock.studio.modes.poser.ui.panes.clips import _import_clip_reason

    assert (
        _import_clip_reason(False, True, False)
        == "Importing an animation needs Blender, which is not installed."
    )
    assert (
        _import_clip_reason(True, False, False)
        == "This skeleton has no clip library to import into."
    )
    assert _import_clip_reason(True, True, True) == "Still importing."
    assert _import_clip_reason(True, True, False) == ""


def test_import_clip_is_disabled_with_a_reason_while_a_skeleton_edit_is_open():
    """P6 (2026-09-13): master hides the whole Clips section while a skeleton
    edit is open because every control in it reads or writes the armature's
    pose; this branch keeps "Import clip..." drawn through that state instead
    (``poser_clips.draw``'s comment), so it must say why it is greyed rather
    than pretend Blender or the library is the reason. Checked first: even
    with Blender missing and no library at all, this is still the one true
    reason while a skeleton edit is open."""
    from warlock.studio.modes.poser.ui.panes.clips import _import_clip_reason

    assert (
        _import_clip_reason(True, True, False, True)
        == "Apply or cancel the skeleton edit first."
    )
    assert (
        _import_clip_reason(False, False, True, True)
        == "Apply or cancel the skeleton edit first."
    )
    assert _import_clip_reason(True, True, False, False) == ""


def test_import_clip_submits_nothing_while_a_skeleton_edit_is_open():
    """The button is disabled, but a disabled button only stops a mouse -- a
    keyboard shortcut or an agent's own call still has to go through
    ``poser_mode.import_clip`` itself, so the refusal has to live here too,
    not only in the pane's reason string."""
    ctx, state = _clip_ctx()
    state.skeleton_editing = True

    assert poser_mode.import_clip(ctx) is False
    assert ctx.submitted == []


def test_saving_after_an_import_keeps_where_the_clip_came_from(monkeypatch):
    """``source`` rides in the working copy untouched, the same
    ``test_saving_keeps_the_provisional_flag`` argument -- so a later Save
    keeps the file it came from, the map that was used, and the date."""
    from warlock.service import clips as svc_clips

    ctx, state = _clip_ctx()
    result = _import_result(name="run")
    result["source_name"] = "mixamo_run.fbx"
    poser_mode.adopt_imported_clips(ctx, result)

    def fake_save(svc, template, payload):
        return {**payload, "template": template, "edited": True}

    monkeypatch.setattr(svc_clips, "save", fake_save)
    poser_mode.save_clips(ctx)
    sent = ctx.results[poser_mode.CLIPS_SAVE_KEY]
    saved = next(c for c in sent["clips"] if c["name"] == "run")
    assert saved["source"]["file"] == "mixamo_run.fbx"
    assert saved["source"]["map"] == "mixamo"


# --- rotation space ---------------------------------------------------------
#
# ``PoseEditor`` is node-local; the shipped clip library is ``space="delta"``.
# Nothing studio-side used to convert, so loading a humanoid key contorted the
# preview and capturing one back wrote node-local values into a delta file.


class _RestEditor:
    """Just enough editor for the two converters: a rest map, XYZW."""

    def __init__(self, rest) -> None:
        self.rest = {name: np.asarray(q, dtype="f8") for name, q in rest.items()}


def _swing(angle):
    """A rotation about local X, the swing axis both converters have to respect."""
    import math

    return [math.sin(angle / 2), 0.0, 0.0, math.cos(angle / 2)]


def test_delta_rotations_round_trip_through_the_node_local_editor():
    editor = _RestEditor({"upper_arm.L": _swing(1.2), "thigh.L": _swing(-0.3)})
    stored = {"upper_arm.L": _swing(-1.0), "thigh.L": _swing(0.4)}

    node = poser_mode._to_node(editor, stored, "delta")
    back = poser_mode._from_node(editor, node, "delta")

    for name, quat in stored.items():
        assert np.allclose(back[name], quat, atol=1e-12)
        # And the trip through the editor genuinely moved the values -- a
        # converter that was quietly a no-op would pass the line above.
        assert not np.allclose(node[name], quat, atol=1e-6)


def test_node_space_libraries_are_passed_through_untouched():
    editor = _RestEditor({"upper_arm.L": _swing(1.2)})
    stored = {"upper_arm.L": _swing(-1.0)}
    want = stored["upper_arm.L"]
    assert np.allclose(poser_mode._to_node(editor, stored, "node")["upper_arm.L"], want)
    assert np.allclose(poser_mode._from_node(editor, stored, "node")["upper_arm.L"], want)


def test_the_shipped_humanoid_library_is_delta_so_the_conversion_is_load_bearing():
    from warlock.kernels.rig import cliplib

    library = cliplib.clip_library("humanoid")
    assert library["space"] == "delta"


# --- the "Rigged assets" picker (B2) -----------------------------------------
#
# Poser had no way to open an asset from inside the mode itself -- the only
# doors in were the inspector's Pose panel link and, once B1 closed it, the
# library/inspector exits list, both of which mean leaving whatever the user
# was looking at. ``can_open_in_poser``/``riggable_assets`` are
# ``troupe_mode.can_send_to_troupe``/``sendable_meshes``'s pattern, reused
# line for line.


def _mesh_row(*, rigged: bool = True, status: str = "done", deleted: bool = False) -> dict:
    job = {"id": "abcdef012345", "stage": "model", "status": status}
    job["files"] = ["model.glb", "rig.glb"] if rigged else ["model.glb"]
    if deleted:
        job["deleted_at"] = 12345.0
    return job


def test_can_open_in_poser_table(svc):
    ctx = FakeCtx(svc)
    assert poser_mode.can_open_in_poser(ctx, _mesh_row(rigged=True)) is True, "rigged mesh: yes"
    assert poser_mode.can_open_in_poser(ctx, _mesh_row(rigged=False)) is False, (
        "unrigged mesh: no"
    )
    rig_followup = {
        "id": "112233445566",
        "stage": "model",
        "status": "done",
        "files": [],
        "params": {"source_job": "abcdef012345"},
    }
    assert poser_mode.can_open_in_poser(ctx, rig_followup) is False, (
        "a rig row's own files are always empty -- asset_open's own docstring"
    )
    assert poser_mode.can_open_in_poser(ctx, _mesh_row(rigged=True, deleted=True)) is False, (
        "deleted: no"
    )
    assert poser_mode.can_open_in_poser(ctx, _mesh_row(rigged=True, status="running")) is False, (
        "unfinished: no"
    )


def test_riggable_assets_is_throttled_page_capped_and_passes_a_files_cache(svc, monkeypatch):
    """``sendable_meshes``'s two costs, paid here too: a second call inside the
    throttle window must not re-list, the page cap is ``troupe_mode``'s own
    constant reused rather than restated, and ``files_cache`` is handed to
    ``list_jobs`` so the picker is not a stat per listed name per row every
    frame its header is open."""
    from warlock.service import jobs as svc_jobs
    from warlock.studio import troupe_mode

    job_id = _rigged_job(svc)
    ctx = FakeCtx(svc)

    calls: list[tuple] = []
    original = svc_jobs.list_jobs

    def counted(service, limit=100, before=None, *, files_cache=None):
        calls.append((limit, files_cache))
        return original(service, limit=limit, before=before, files_cache=files_cache)

    monkeypatch.setattr(svc_jobs, "list_jobs", counted)

    first = poser_mode.riggable_assets(ctx)
    for _ in range(10):
        poser_mode.riggable_assets(ctx)

    assert len(calls) == 1, "a second call inside the throttle window must not re-list"
    assert poser_mode.riggable_assets(ctx) == first
    assert first and first[0]["id"] == job_id

    limit, files_cache = calls[0]
    assert limit == troupe_mode.SCAN_LIMIT, "the same page cap, reused rather than restated"
    state = poser_mode.ensure(ctx)
    assert files_cache is state.riggable_files, "the caller must own the files_cache dict"


def test_invalidate_riggable_makes_the_next_call_relist(svc):
    _rigged_job(svc)
    ctx = FakeCtx(svc)
    poser_mode.riggable_assets(ctx)
    state = poser_mode.ensure(ctx)
    assert state.riggable_cache is not None

    poser_mode.invalidate_riggable(ctx)
    assert state.riggable_cache is None


def test_the_picker_row_carries_what_open_asset_reads_back(svc):
    """The 2026-09-09 review defect: a first cut of ``riggable_assets``
    trimmed its rows to ``sendable_meshes``' shape (``id``/``prompt``/
    ``created_at``) without checking that its consumer is different --
    ``send_to_troupe`` re-reads a picked row through the service before
    acting, ``open_asset`` reads the dict it is handed and never again. Every
    session opened from the picker silently lost the asset's recorded front
    (``params["front_yaw"]``) and opened facing yaw 0 regardless of what
    ``poser_mode.set_front`` had recorded. Also pins the label: ``open_asset``
    prefers ``name`` over ``prompt``, so a job with a name set must not lose
    it to the picker's row shape either.
    """
    job_id = svc.store.create(
        "image", "a prop", {"front_yaw": 137.5}, stage="model", status="done"
    )
    svc.store.set_meta(job_id, name="Test Prop")
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"glTF-not-really")
    (job_dir / "rig.glb").write_bytes(b"glTF-not-really")
    (job_dir / "rig.json").write_text(
        json.dumps({"version": 1, "template": "humanoid", "bones": []}), "utf-8"
    )

    ctx = FakeCtx(svc)
    row = poser_mode.riggable_assets(ctx)[0]
    assert row["id"] == job_id

    poser_mode.open_asset(ctx, row)
    state = poser_mode.ensure(ctx)
    assert state.job_id == job_id
    assert state.asset_front_yaw == pytest.approx(137.5), (
        "the picker's row must carry front_yaw through to open_asset, not 0.0"
    )
    assert state.asset_label == "Test Prop", (
        "and the asset's name, which open_asset prefers over its prompt"
    )


def test_the_rigged_assets_picker_click_reaches_open_asset_with_the_row(svc, monkeypatch):
    """``poser_library._pick`` is named on purpose so a click is callable with
    no imgui frame at all -- what is pinned is that it hands the picker's own
    row straight to ``open_asset``, unmodified, and does not re-implement the
    dirty-editor guard or the template-switch discard confirm that function
    already carries."""
    from warlock.studio.modes.poser.ui.panes import library as poser_library

    job_id = _rigged_job(svc)
    ctx = FakeCtx(svc)
    row = poser_mode.riggable_assets(ctx)[0]
    assert row["id"] == job_id

    poser_library._pick(ctx, row)
    state = poser_mode.ensure(ctx)
    assert state.job_id == job_id, "the click must open the row it was drawn from"
