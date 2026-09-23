"""Regression tests for the 2026-09-23 audit's poser-rig-clips findings.

Four independent defects, four independent tests:

* **poser-01** -- ``characters.instantiate`` axis-swapped every non-human
  species' joint displacement, because it summed a glTF-axis ``jdisp/*``
  field onto a Blender-axis base and only then ran the sum through
  ``_to_gltf``. Pinned here for ``ogre`` (see the finding's note: the gpu
  lane should confirm the corrected joints against a real render).
* **poser-02** -- ``service.clips._check_renders`` only held the humanoid
  template to the render-time frame-table check, so a quadruped/bird/blob
  clip library could be saved missing a clip its own default sheet layout
  needs.
* **poser-05** -- ``cliplib.parse_clip_library`` (the read door) accepted a
  ``segments`` list whose length did not match its keys, unlike the write
  door (``service.clips._check_shape``).
* **poser-06** -- ``clip_import.analyse`` bounded a many-action Blender
  sample with ``pose_timeout``, sized for one inline pose bake.
"""

from __future__ import annotations

import pytest
from test_cliptransfer import (
    _axis_angle,
    _baseline_source_bones,
    _entry,
    _make_sample,
    _rest_frame_bones,
)

from realmspinner import doctor
from realmspinner.characters import DEFAULT_RECIPE
from realmspinner.characters.instantiate import instantiate
from realmspinner.doctor import Check
from realmspinner.kernels.rig import cliplib
from realmspinner.pipelines import blender_run
from realmspinner.service import Invalid, clip_import
from realmspinner.service import clips as svc_clips


@pytest.fixture(autouse=True)
def _fresh_clip_cache():
    # The library caches are module globals filled once -- the same isolation
    # tests/service/test_clip_editing.py and tests/test_clip_library_v3.py use.
    cliplib.invalidate_clips()
    yield
    cliplib.invalidate_clips()


# --- poser-01 ----------------------------------------------------------------


def test_displaced_joints_add_the_gltf_axis_field_onto_a_gltf_axis_base_not_a_blender_one(
    tmp_path,
):
    """``arrays["joints"]`` is stored in Blender axes and ``arrays["jdisp/*"]``
    is baked in glTF axes -- the same frame ``arrays["disp/*"]`` already
    displaces the mesh vertices in, with no further transform
    (``instantiate._displaced``). Before the fix, ``instantiate.instantiate``
    added the glTF-axis field onto the Blender-axis base and ran ``_to_gltf``
    over that mismatched sum, which axis-permutes the already-correct
    displacement a second time. Pinned against the default (non-zero
    appearance) ogre recipe's ``hips`` joint head, read back from the written
    ``character.json`` sidecar so the test exercises the same path a job
    directory does, not the bare helper functions.

    Against the unfixed tree this asserted ``[0.0, 0.11712002411484239,
    1.3604462765017142]`` (confirmed by running the pre-fix
    ``instantiate.instantiate`` directly against the ogre bake) -- the y and z
    components the axis-swap bug moved. The gpu lane should confirm the
    corrected joints against a real render, per the audit note: this test only
    pins the number, not that it looks right in a viewer.
    """
    inst = instantiate(DEFAULT_RECIPE, tmp_path)
    assert inst.family == "ogre"
    hips = next(b for b in inst.joints if b["name"] == "hips")
    assert hips["head"] == pytest.approx(
        [0.0, 0.060999755876971776, 1.4165665447395848], abs=1e-9
    )


# --- poser-02 ----------------------------------------------------------------


@pytest.mark.parametrize("template", ["humanoid", "quadruped", "bird", "blob"])
def test_saving_a_clip_library_missing_a_needed_clip_is_refused(svc, template):
    """Every shipped template's default sheet layout needs "idle" -- dropping
    every clip named "idle" from a template's own shipped library and saving
    it used to be refused only for humanoid (``_check_renders`` held only
    ``TROUPE_TEMPLATE`` to the frame-table check), so a quadruped, bird or
    blob library saved clean with a hole a sheet needs, contradicting this
    module's own docstring ("the editor refuses the edit"). Against the
    unfixed tree, the quadruped/bird/blob legs of this parametrize passed
    (wrongly) with no exception raised at all.
    """
    view = svc_clips.library(svc, template)
    payload = {
        "space": view["space"],
        "poses": view["poses"],
        "clips": [c for c in view["clips"] if c["name"] != "idle"],
    }
    with pytest.raises(Invalid) as caught:
        svc_clips.save(svc, template, payload)
    assert caught.value.field == "clips"
    assert "idle" in str(caught.value)


# --- poser-05 ----------------------------------------------------------------


def _raw(clips: list[dict]) -> dict:
    return {
        "version": 3,
        "poses": [{"name": n, "bones": {}} for n in ("a", "b", "c")],
        "clips": clips,
    }


def test_parse_clip_library_rejects_a_segments_list_that_does_not_match_its_keys():
    """An open clip over 3 keys takes 2 segment lengths -- exactly what
    ``service.clips._check_shape`` (the write door) has always refused a
    mismatch against. Before the fix, ``cliplib.parse_clip_library`` (the
    read door both the shipped files and any hand-edited or externally
    produced file pass through) accepted this one-segment version with no
    complaint at all.
    """
    raw = _raw(
        [{"name": "bad", "keys": ["a", "b", "c"], "segments": [1], "duration_ms": 100}]
    )
    with pytest.raises(ValueError, match="needs 2 segment lengths, not 1"):
        cliplib.parse_clip_library(raw)


def test_parse_clip_library_rejects_a_segment_outside_the_frame_range():
    """The write door's own range, 1-64 frames per segment, mirrored at the
    read door. Before the fix, a segment of 0 (or an absurdly large count)
    parsed clean here and would only ever be caught if that exact clip
    happened to be resampled."""
    raw = _raw(
        [
            {
                "name": "bad",
                "keys": ["a", "b"],
                "segments": [0],
                "duration_ms": 100,
            }
        ]
    )
    with pytest.raises(ValueError, match="1-64"):
        cliplib.parse_clip_library(raw)


def test_parse_clip_library_still_accepts_the_pre_existing_short_filler_clip_shape():
    """Several committed tests (tests/test_clip_library_v3.py and others) use a
    single-key, one-segment clip as minimal filler for unrelated assertions
    (duration_ms migration, the direction-name guard, pose-count). The
    poser-05 fix must not refuse that shape: a sub-2-key clip is degenerate
    either way and ``sheet.interpolate_clip`` already refuses it by name
    ("a clip needs at least two keyframes") the moment it is actually
    expanded, so the read door leaves it alone rather than duplicating
    ``service.clips.MIN_KEYS``."""
    raw = _raw([{"name": "filler", "keys": ["a"], "segments": [1], "duration_ms": 100}])
    parsed = cliplib.parse_clip_library(raw)
    assert parsed["clips"][0]["name"] == "filler"


# --- poser-06 ----------------------------------------------------------------


def _canned_payload() -> dict:
    """The same minimal, synthetic worker result
    ``tests/service/test_clip_import_service.py``'s own ``_canned_payload``
    builds: two frames, one bone swinging 30 degrees, so ``cliptransfer.transfer``
    has real motion to convert rather than raising on an all-static sample."""
    source_bones = _baseline_source_bones()
    rest = _rest_frame_bones(source_bones)
    moved = dict(rest)
    moved["LeftArm"] = _entry(_axis_angle((0, 0, 1), 30.0), source_bones["LeftArm"]["head"])
    action = {
        "name": "Motion",
        "fps": 30.0,
        "frame_start": 0,
        "frame_end": 1,
        "frames": [{"frame": 0, "bones": rest}, {"frame": 1, "bones": moved}],
    }
    sample = _make_sample(source_bones, [action])
    return {"ok": True, "armature": "Armature", **sample}


def test_import_clip_is_not_bounded_by_the_single_pose_bake_timeout(svc, monkeypatch, tmp_path):
    """``pose_timeout`` (300s by default) is sized, by its own docstring in
    config.py, for one inline Blender pose bake -- seconds, not minutes -- but
    ``clip_import.analyse`` can sample up to ``MAX_ACTIONS`` (64) actions of
    up to 900 frames each in the same subprocess call. Before the fix, the
    worker call was timed at exactly ``svc.config.pose_timeout`` with no
    scaling for the action ceiling -- this test's own assertion
    ``captured["timeout"] > svc.config.pose_timeout`` is exactly what failed
    against the unfixed code (the two were equal).
    """
    monkeypatch.setattr(
        doctor, "blender_check", lambda **kw: Check("Blender (rigging)", True, "bpy 5.2.0", False)
    )

    captured: dict = {}
    payload = _canned_payload()

    def fake_run_worker(spec, **kwargs):
        captured["timeout"] = kwargs.get("timeout")
        result_path = __import__("pathlib").Path(spec["result_path"])
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(__import__("json").dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr(blender_run, "run_worker", fake_run_worker)

    source = tmp_path / "clip.fbx"
    source.write_bytes(b"stub")

    clip_import.analyse(svc, "humanoid", source)

    assert captured["timeout"] is not None
    assert captured["timeout"] == pytest.approx(
        svc.config.pose_timeout * clip_import.MAX_ACTIONS
    )
    # Not merely equal to the single-bake ceiling any more.
    assert captured["timeout"] > svc.config.pose_timeout
