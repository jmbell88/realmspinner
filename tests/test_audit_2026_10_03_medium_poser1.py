"""The 2026-10-03 audit's poser-13, -14, -15, -16 and -19 (Medium)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from PIL import Image
from test_cliptransfer import (
    _baseline_source_bones,
    _make_sample,
    _rest_frame_bones,
)
from test_rerender_door import _published, _rigged_mesh, _runs

from realmspinner import cliptransfer
from realmspinner.godotscene import scene_text
from realmspinner.kernels.rig import cliplib, poses
from realmspinner.pipelines import pixelize
from realmspinner.service import Invalid
from realmspinner.service import clips as svc_clips
from realmspinner.service import troupe as svc_troupe

# -- poser-13: a clip name written bare into a .tscn property-path key ---------


@pytest.mark.parametrize(
    "bad", ["a=b", "a;b", "a#b", "a[0]", "a,b", "Heavy Attack", "tab\there"]
)
def test_a_clip_name_with_an_equals_a_semicolon_or_a_space_is_refused_rather_than_written_into_a_states_key(  # noqa: E501
    bad,
):
    with pytest.raises(ValueError, match="clip name"):
        scene_text(
            root_name="Hero", glb_file="hero.glb", clips=[("idle", True), (bad, False)]
        )


def test_the_godot_refusal_names_the_clip_and_an_ordinary_name_is_still_written():
    with pytest.raises(ValueError, match=r"Heavy Attack"):
        scene_text(
            root_name="Hero",
            glb_file="hero.glb",
            clips=[("idle", True), ("Heavy Attack", False)],
        )
    text = scene_text(
        root_name="Hero",
        glb_file="hero.glb",
        clips=[("idle", True), ("attack_02", False), ("Dodge-Roll", False)],
    )
    assert "states/attack_02/node" in text
    assert "states/Dodge-Roll/node" in text


# -- poser-14: a re-render reads the mesh's current rig ------------------------


def test_a_re_render_is_refused_when_the_mesh_has_been_re_rigged_on_another_template(svc):
    job_id = _rigged_mesh(svc)
    _row, sheet_id = _published(svc, job_id)
    (svc.job_dir(job_id) / "rig.json").write_text(
        json.dumps({"template": "quadruped"}), "utf-8"
    )
    with pytest.raises(Invalid) as caught:
        svc_troupe.rerender_charsheet(svc, job_id, sheet_id=sheet_id, subset=_runs(1))
    assert caught.value.field == "sheet_id"
    assert "quadruped" in str(caught.value)


def test_a_re_render_is_refused_at_the_door_when_the_rig_is_gone(svc):
    job_id = _rigged_mesh(svc)
    _row, sheet_id = _published(svc, job_id)
    (svc.job_dir(job_id) / "rig.glb").unlink()
    with pytest.raises(Invalid) as caught:
        svc_troupe.rerender_charsheet(svc, job_id, sheet_id=sheet_id, subset=_runs(1))
    assert caught.value.field == "sheet_id"
    assert "no longer rigged" in str(caught.value)


def test_a_re_render_on_the_same_rig_is_still_accepted(svc):
    job_id = _rigged_mesh(svc)
    _row, sheet_id = _published(svc, job_id)
    made = svc_troupe.rerender_charsheet(svc, job_id, sheet_id=sheet_id, subset=_runs(1))
    assert made["sheet_id"] != sheet_id


# -- poser-15: a static action is a rest clip, not a refusal at Save -----------


def _static_sample():
    source_bones = _baseline_source_bones()
    rest = _rest_frame_bones(source_bones)
    action = {
        "name": "TPose",
        "fps": 30.0,
        "frame_start": 0,
        "frame_end": 1,
        "frames": [{"frame": 0, "bones": rest}, {"frame": 1, "bones": rest}],
    }
    return _make_sample(source_bones, [action])


def test_a_static_action_converts_to_a_clip_the_save_door_accepts(svc):
    cliplib.invalidate_clips()
    try:
        (entry,) = cliptransfer.transfer(_static_sample(), template="humanoid")
        assert all(not pose["bones"] for pose in entry["poses"].values()), (
            "premise: a wholly static action leaves every key pose's bones empty"
        )
        library = svc_clips.library(svc, "humanoid")
        payload = {
            "space": library["space"],
            "poses": [
                *library["poses"],
                *({"name": name, **body} for name, body in entry["poses"].items()),
            ],
            "clips": [*library["clips"], dict(entry["clip"])],
            "template": "humanoid",
        }
        document = svc_clips._check_shape(payload)
        cliplib.parse_clip_library(document)
    finally:
        cliplib.invalidate_clips()


def test_a_key_pose_with_no_bones_map_at_all_is_still_refused(svc):
    library = svc_clips.library(svc, "humanoid")
    poses_in = [dict(p) for p in library["poses"]]
    del poses_in[0]["bones"]
    with pytest.raises(Invalid) as caught:
        svc_clips._check_shape(
            {
                "space": library["space"],
                "poses": poses_in,
                "clips": library["clips"],
                "template": "humanoid",
            }
        )
    assert caught.value.field == "poses"


# -- poser-16: mirroring reflects a centre bone's twist and lean ---------------


def _about(axis, degrees):
    half = math.radians(degrees) / 2
    s = math.sin(half)
    return [axis[0] * s, axis[1] * s, axis[2] * s, math.cos(half)]


def test_mirror_pose_reflects_a_centre_bones_twist_and_lean():
    pairs = [["upper_arm.L", "upper_arm.R"]]
    yaw = _about((0, 0, 1), 30)
    lean = _about((0, 1, 0), 20)
    out = poses.mirror_pose({"head": yaw, "spine": lean}, pairs)
    assert out["head"] == pytest.approx(poses.mirror_quaternion(yaw))
    assert out["head"] != pytest.approx(yaw)
    assert out["spine"] == pytest.approx(poses.mirror_quaternion(lean))


def test_mirror_pose_twice_returns_the_original_pose():
    pairs = [["upper_arm.L", "upper_arm.R"]]
    posed = {
        "upper_arm.L": _about((0, 1, 0), 40),
        "head": _about((0, 0, 1), 30),
        "spine": _about((1, 0, 0), 10),
    }
    back = poses.mirror_pose(poses.mirror_pose(posed, pairs), pairs)
    # The mirrored arm lands on the partner bone; the originals come back.
    assert back["head"] == pytest.approx(posed["head"])
    assert back["spine"] == pytest.approx(posed["spine"])
    assert back["upper_arm.L"] == pytest.approx(posed["upper_arm.L"])


# -- poser-19: the atlas dither is anchored to each cell -----------------------


def test_pixelize_atlas_dithers_identical_cells_identically_at_a_size_that_is_not_a_multiple_of_four():  # noqa: E501
    cell, columns, rows, stride = 30, 3, 2, 2
    src_cell = cell * stride
    ramp = np.linspace(40, 215, src_cell).astype(np.uint8)
    one = np.zeros((src_cell, src_cell, 4), dtype=np.uint8)
    one[..., 0] = ramp[None, :]
    one[..., 1] = ramp[:, None]
    one[..., 2] = 120
    one[..., 3] = 255
    atlas = Image.fromarray(np.tile(one, (rows, columns, 1)), "RGBA")
    palette = [(30, 30, 40), (100, 90, 120), (200, 180, 160), (240, 240, 230)]
    out, _report = pixelize.pixelize_atlas(
        atlas,
        columns=columns,
        rows=rows,
        cell=cell,
        palette=palette,
        dither=True,
        clean=False,
    )
    arr = np.asarray(out)
    first = arr[:cell, :cell]
    for row in range(rows):
        for column in range(columns):
            piece = arr[row * cell : (row + 1) * cell, column * cell : (column + 1) * cell]
            assert np.array_equal(piece, first), (row, column)


def test_a_multiple_of_four_cell_dithers_exactly_as_the_atlas_anchored_tile_did():
    from realmspinner.pipelines.pixel import map_palette

    rng = np.random.default_rng(3)
    arr = rng.integers(0, 255, (64, 96, 4), dtype=np.uint8)
    arr[..., 3] = 255
    image = Image.fromarray(arr, "RGBA")
    palette = ((10, 10, 20), (120, 100, 90), (230, 220, 200))
    anchored = np.asarray(map_palette(image, palette, dither=True, cell=32))
    legacy = np.asarray(map_palette(image, palette, dither=True))
    assert np.array_equal(anchored, legacy)
