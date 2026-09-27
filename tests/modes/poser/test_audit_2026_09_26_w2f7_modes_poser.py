"""Regression tests for two 2026-09-26 audit findings in Poser's clip editor
and skeleton editor, both in ``src/realmspinner/studio/modes/poser/mode.py``:

* **poser-mode-01** -- ``apply_key``/``scrub`` write the pose straight onto
  the bound ``PoseEditor`` rather than through the ``Viewer`` wrapper methods
  of the same names, so ``Viewer._after_pose_change`` (``studio/
  _viewer_pose.py``) -- which refreshes a bound skinned mesh's GPU skin
  palettes -- never ran, and a mesh bound to the armature did not follow a
  key click or a scrub tick.
* **poser-mode-02** -- ``apply_skeleton`` never cleared
  ``editor.draft_dirty`` after submitting the drafted skeleton as a re-rig,
  so the generic landing guard (:func:`poser_mode.guard`, read through
  :func:`poser_mode._pump_rerig`) asked "Unsaved skeleton changes will be
  lost" about the user's own already-submitted Apply once the queued job
  finished, and declining that prompt dropped the job's tracking (the pop
  runs before the guard) with the new rig never bound.

Named per the fixer brief's convention; deliberately self-contained rather
than importing from ``tests/modes/poser/test_poser_mode.py`` (not owned by
this fixer and not a package tests can import from -- ``pytest`` runs this
tree with no ``__init__.py``).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.kernels.geom3d.gltf import Model, Node
from realmspinner.kernels.rig import templates
from realmspinner.studio import _viewer_pose
from realmspinner.studio.modes.poser import mode as poser_mode
from realmspinner.studio.viewer.pose import PoseEditor

# -- shared fake context (trimmed from test_poser_mode.py's own FakeCtx,
# whose surface this module also uses; kept minimal and self-contained) -----


class FakeCtx:
    def __init__(self, svc=None, accept=True) -> None:
        self.svc = svc
        self.state = SimpleNamespace(poser=None, preview={})
        self.submitted: list[str] = []
        self.results: dict = {}
        self.tags: dict = {}
        self.accept = accept
        self.busy_keys: set[str] = set()
        self.confirms = _Asks()
        self.toasts: list = []
        self.rig_default = "humanoid"
        self.rigging_available = True
        self.poser_viewer = None
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


class _Asks:
    def __init__(self) -> None:
        self.asked: list = []

    def ask(self, item) -> None:
        self.asked.append(item)


# -- poser-mode-01 -------------------------------------------------------------


class _FakeGpu:
    """Just enough of the real ``GpuTexturePool``-shaped surface for
    ``Viewer._after_pose_change`` to touch."""

    def __init__(self) -> None:
        self.refresh_calls = 0

    def refresh_palettes(self) -> None:
        self.refresh_calls += 1


class _PosingViewer(_viewer_pose.PoseOps):
    """A stand-in bound to the *real* ``PoseOps`` mixin -- unlike
    ``test_poser_mode.py``'s own ``FakeViewer``, which deliberately omits any
    GPU refresh because none of its tests bind a mesh a palette could be
    recomputed for (see that file's comment on its own skeleton pass-
    throughs). This one exists specifically to prove ``apply_key``/``scrub``
    reach ``_after_pose_change``.
    """

    def __init__(self, model: Model, bones: list[str]) -> None:
        self.editor = PoseEditor()
        self.editor.bind(model, bones)
        self.pose_mode = True
        self.pose_job_id = "job"
        self.onion: list = []
        self.gpu = _FakeGpu()
        self.on_pose_dirty = None
        self._render_dirty = False


def _tiny_model() -> tuple[Model, list[str]]:
    names = ["hips", "spine", "head"]
    nodes = [Node(name="rig", children=[1, 2, 3])]
    for i, name in enumerate(names):
        nodes.append(Node(name=name, translation=m3.vec3(0.0, 0.1 * i, 0.0)))
    return Model(nodes, roots=[0], meshes=[], skins=[]), names


def _clip_library() -> dict[str, Any]:
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


def _posing_ctx() -> tuple[FakeCtx, _PosingViewer]:
    model, bones = _tiny_model()
    ctx = FakeCtx()
    ctx.poser_viewer = viewer = _PosingViewer(model, bones)
    poser_mode.adopt_clips(ctx, _clip_library())
    return ctx, viewer


def test_apply_key_and_scrub_refresh_the_skin_palettes_of_a_bound_mesh() -> None:
    """poser-mode-01: both must reach ``_after_pose_change`` -- clicking a
    key or dragging the scrubber must actually move a bound skinned mesh, not
    only the wireframe armature."""
    ctx, viewer = _posing_ctx()

    before = viewer.gpu.refresh_calls
    poser_mode.select_key(ctx, 1)  # -> apply_key(ctx)
    assert viewer.gpu.refresh_calls > before, "apply_key did not refresh skin palettes"

    before = viewer.gpu.refresh_calls
    poser_mode.scrub(ctx, 3)
    assert viewer.gpu.refresh_calls > before, "scrub did not refresh skin palettes"


def test_apply_key_still_moves_the_armature_itself() -> None:
    """The fix must not replace the pose write -- only add the missing
    refresh call beside it."""
    ctx, viewer = _posing_ctx()
    poser_mode.select_key(ctx, 1)
    # Key "B" moves "spine"; the armature should show it once selected.
    assert viewer.editor.pose().get("spine") is not None


# -- poser-mode-02 ---------------------------------------------------------------


class FakeViewer:
    """A trimmed copy of ``test_poser_mode.py``'s own fake, for the one path
    this pass's fix touches (``apply_skeleton`` / ``pump_rerig``)."""

    def __init__(self, model=None, bones=None) -> None:
        self.editor = PoseEditor()
        self.pose_mode = False
        self.path = None
        self.loaded: list = []
        self.cleared = 0
        self.onion: list = []
        self.skinned = True
        self.pose_job_id: str | None = None
        self.pending = None
        if model is not None:
            self.editor.bind(model, bones)
            self.pose_mode = True

    def load_model(self, path) -> None:
        self.loaded.append(path)
        self.path = path

    def parse_model(self, path):
        return ("parsed", path)

    def adopt_model(self, parsed, path) -> None:
        self.pose_mode = False
        self.pose_job_id = None
        self.loaded.append(path)
        self.path = path
        self.pending = None

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
        return 1.0

    def frame_bounds(self, lo, hi) -> float:
        return 1.0

    def clear(self) -> None:
        self.cleared += 1
        self.path = None
        self.editor.clear()
        self.pose_mode = False
        self.pending = None

    def enter_skeleton_mode(self, rig) -> None:
        self.editor.enter_skeleton_mode(rig)

    def exit_skeleton_mode(self) -> None:
        self.editor.exit_skeleton_mode()

    def skeleton_payload(self):
        return self.editor.skeleton_payload()

    def skel_add_child(self, parent):
        return self.editor.skel_add_child(parent)


def _fake_blender(monkeypatch) -> None:
    from realmspinner import doctor
    from realmspinner.doctor import Check
    from realmspinner.pipelines import blender_run

    monkeypatch.setattr(
        doctor, "blender_check", lambda **kw: Check("Blender (rigging)", True, "bpy 5.2.0", False)
    )

    def run_worker(spec, **kwargs):
        from pathlib import Path

        Path(spec["out_glb"]).write_bytes(b"armature-glb")
        return {"ok": True}

    monkeypatch.setattr(blender_run, "run_worker", run_worker)


def _rigged_job(svc, **meta):
    job_id = svc.store.create("image", "a prop", {}, stage="model", status="done")
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"glTF-not-really")
    (job_dir / "rig.glb").write_bytes(b"glTF-not-really")
    (job_dir / "rig.json").write_text(
        json.dumps({"version": 1, "template": "humanoid", "bones": [], **meta}), "utf-8"
    )
    return job_id


def _armature_model() -> Model:
    names = [b["name"] for b in templates.get_template("humanoid").bones]
    nodes = [Node(name="rig", children=list(range(1, len(names) + 1)))]
    for i, name in enumerate(names):
        nodes.append(Node(name=name, translation=m3.vec3(0.0, 0.1 * i, 0.0)))
    return Model(nodes, roots=[0], meshes=[], skins=[])


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
        "bounds": {"min": [-0.5, -0.5, 0.0], "max": [0.5, 0.5, 1.0]},
    }


def _opened_asset_for_skeleton(svc, monkeypatch, **rig_meta):
    _fake_blender(monkeypatch)
    job_id = _rigged_job(svc, **rig_meta)
    ctx = FakeCtx(svc)
    ctx.poser_viewer = viewer = FakeViewer()
    poser_mode.open_asset(ctx, {"id": job_id, "name": "Prop"})
    poser_mode.sync_asset(ctx, viewer)
    names = [b["name"] for b in templates.get_template("humanoid").bones]
    viewer.editor.bind(_armature_model(), names)
    viewer.editor.root = next(
        b["name"] for b in templates.get_template("humanoid").bones if b["parent"] is None
    )
    return ctx, viewer, job_id


def test_landing_an_applied_skeleton_rerig_does_not_ask_to_discard_the_draft_it_applied(
    svc, monkeypatch
):
    """poser-mode-02: once ``apply_skeleton`` has submitted the draft as a
    re-rig, the draft is no longer "unsaved" -- the queued job's landing must
    not ask to discard the very changes it is about to apply, and must not
    drop the job's tracking on a decline it never had to offer."""
    ctx, viewer, job_id = _opened_asset_for_skeleton(svc, monkeypatch, **_custom_rig_meta())
    poser_mode.enter_skeleton_edit(ctx)
    viewer.editor.skel_add_child(viewer.editor.draft_root)
    assert viewer.editor.draft_dirty is True, "the draft edit itself must still be tracked"

    poser_mode.apply_skeleton(ctx)
    assert viewer.editor.draft_dirty is False, "the applied draft must read as saved"

    key = f"{poser_mode.ASSET_RERIG_KEY_PREFIX}{job_id}"
    result = ctx.results[key]
    poser_mode.on_task_done(ctx, SimpleNamespace(key=key, result=result))
    state = poser_mode.ensure(ctx)
    assert state.rerig_jobs[job_id] == result["id"]

    ctx.jobs[result["id"]] = {"id": result["id"], "status": "done"}
    poser_mode.pump_rerig(ctx)

    assert ctx.confirms.asked == [], "landing the user's own Apply must not confirm"
    assert job_id not in state.rerig_jobs, "the landed job must still be retired"
    assert viewer.pose_mode is False, "the new rig must actually land, not be left unbound"
