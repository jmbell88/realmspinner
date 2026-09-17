"""Clip expansion, the character-sheet plan, and the sidecar's timing block."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from warlock import clips
from warlock.kernels.rig import cliplib
from warlock.pipelines import charsheet as cs
from warlock.pipelines import sheet as sheetlib

IDENT = [0.0, 0.0, 0.0, 1.0]


def _pose(name, z=0.0, root=None):
    row = {"id": name, "name": name, "bones": {"hips": [0.0, 0.0, z, (1 - z * z) ** 0.5]}}
    if root:
        row["root_translation"] = root
    return row


# --- multi-key clips ---------------------------------------------------------


def test_a_cyclic_clip_never_lands_on_a_key_twice():
    """The seam rule, generalised: each segment stops short of its far key
    because that key is the next segment's frame 0."""
    keys = [_pose("a"), _pose("b", 0.2), _pose("c", 0.4), _pose("d", 0.2)]
    out = sheetlib.interpolate_clip(keys, [2, 2, 2, 2], closed=True)
    assert len(out) == 8
    assert [r["frame"] for r in out] == list(range(8))
    assert out[0]["bones"]["hips"] == keys[0]["bones"]["hips"]
    assert out[2]["bones"]["hips"] == keys[1]["bones"]["hips"]


def test_normalized_resampling_uses_cycle_and_one_shot_endpoints() -> None:
    keys = [_pose("a"), _pose("b", 0.2), _pose("c", 0.4), _pose("d", 0.2)]
    cycle = sheetlib.resample_clip(keys, [2, 2, 2, 2], 4, closed=True)
    assert [frame["bones"] for frame in cycle] == [key["bones"] for key in keys]
    one_shot = sheetlib.resample_clip(keys, [2, 2, 2], 2, closed=False)
    assert one_shot[0]["bones"] == keys[0]["bones"]
    assert one_shot[-1]["bones"] == keys[-1]["bones"]


def test_a_one_shot_lands_exactly_on_its_last_key():
    keys = [_pose("a"), _pose("b", 0.3), _pose("c", 0.5)]
    out = sheetlib.interpolate_clip(keys, [2, 2], closed=False)
    assert len(out) == 5
    assert out[-1]["bones"]["hips"] == pytest.approx(keys[-1]["bones"]["hips"])


def test_the_old_two_key_call_is_the_new_one_with_a_single_segment():
    a, b = _pose("a"), _pose("b", 0.4)
    assert sheetlib.interpolate(a, b, 6) == sheetlib.interpolate_clip(
        [a, b], [6, 6], closed=True
    )[:6]


def test_root_translation_is_interpolated_rather_than_refused():
    """Gap 1's other half: a bob is root translation, and it used to be a
    refusal by name."""
    keys = [_pose("down", root=[0.0, 0.0, -0.02]), _pose("up", root=[0.0, 0.0, 0.02])]
    out = sheetlib.interpolate_clip(keys, [2, 2], closed=True)
    assert [round(r["root_translation"][2], 4) for r in out] == [-0.02, 0.0, 0.02, 0.0]


def test_a_clip_of_offsetless_poses_records_no_offset_at_all():
    """Byte-identical to what this produced before offsets were interpolated,
    which is what keeps ``_sheet_root_offsets`` short-circuiting on them."""
    out = sheetlib.interpolate_clip([_pose("a"), _pose("b", 0.2)], [2, 2], closed=True)
    assert all("root_translation" not in r for r in out)


def test_easing_reshapes_the_spacing_without_moving_the_keys():
    keys = [_pose("a"), _pose("b", 0.5)]
    linear = sheetlib.interpolate_clip(keys, [4, 4], closed=True, easing="linear")
    eased = sheetlib.interpolate_clip(keys, [4, 4], closed=True, easing="ease")
    assert linear[0]["bones"] == eased[0]["bones"]
    assert linear[1]["bones"] != eased[1]["bones"]


@pytest.mark.parametrize(
    "args, kwargs, message",
    [
        (([_pose("a")], [2]), {}, "at least two keyframes"),
        (([_pose("a"), _pose("b"), _pose("c")], [2]), {}, "takes 2 segment"),
        (([_pose("a"), _pose("b")], [2, 2, 2]), {"closed": True}, "takes 2 segment"),
        (([_pose("a"), _pose("b")], [0, 1]), {"closed": True}, "at least one frame"),
        (([_pose("a"), _pose("b")], [30, 30]), {"closed": True}, "must be 1-"),
    ],
)
def test_a_clip_that_does_not_add_up_is_refused_by_name(args, kwargs, message):
    with pytest.raises(ValueError, match=message):
        sheetlib.interpolate_clip(*args, **kwargs)


def test_an_unknown_easing_is_refused():
    with pytest.raises(ValueError, match="easing must be one of"):
        sheetlib.interpolate_clip([_pose("a"), _pose("b")], [2], easing="bounce")


# --- the shipped clip library ------------------------------------------------


def test_the_legacy_five_clips_are_present_with_their_original_frames_loop_and_timing():
    """The vocabulary is open now (``attack_02``, ``cast``, ``fall``, ``hit``,
    ``death`` and whatever else an author adds), but the five original clips
    must still mean exactly what they meant before -- this holds whether or
    not the new ones have landed alongside them in the shipped library."""
    library = cliplib.clip_library("humanoid")
    by_name = {c["name"]: c for c in library["clips"]}
    legacy_keys: set[str] = set()
    for name, frames, loop, duration_ms in cs.ANIMATIONS:
        clip = by_name[name]
        assert clip["closed"] == loop, name
        assert clip["duration_ms"] == duration_ms, name
        keys = cliplib.clip_keys("humanoid", name)
        out = sheetlib.interpolate_clip(
            keys, clip["segments"], closed=clip["closed"], easing=clip["easing"]
        )
        assert len(out) == frames, name
        legacy_keys.update(clip["keys"])
    assert len(legacy_keys) == 22


def test_the_walk_and_run_carry_a_vertical_bob():
    """The thing the root-translation refusal was costing."""
    for name in ("walk", "run"):
        keys = cliplib.clip_keys("humanoid", name)
        heights = [k.get("root_translation", [0, 0, 0])[2] for k in keys]
        assert max(heights) > 0 > min(heights), name


def test_a_clip_naming_a_pose_the_file_lacks_costs_that_library(tmp_path, caplog):
    (tmp_path / "broken.json").write_text(
        '{"poses": [{"name": "a", "bones": {}}], '
        '"clips": [{"name": "c", "keys": ["a", "ghost"], "segments": [1, 1]}]}',
        encoding="utf-8",
    )
    assert cliplib._load_clip_library(tmp_path) == {}


def test_a_template_with_no_clips_gets_an_empty_library():
    assert cliplib.clip_library("fish") == {"poses": {}, "clips": [], "space": "node"}


# --- the plan ----------------------------------------------------------------


def _records():
    return {
        name: sheetlib.interpolate_clip(
            cliplib.clip_keys("humanoid", name),
            next(c for c in cliplib.clip_library("humanoid")["clips"] if c["name"] == name)[
                "segments"
            ],
            closed=loop,
            easing="linear",
        )
        for name, _frames, loop, _ms in cs.ANIMATIONS
    }


def test_the_plan_is_the_atlas_the_program_committed_to():
    layout = cs.plan(_records(), frame_size=128)
    assert (layout.columns, layout.rows) == (8, 32)
    assert (layout.width, layout.height) == (1024, 4096)
    assert len(layout.cells) == 256


def test_every_cell_lands_on_its_own_rectangle():
    layout = cs.plan(_records(), frame_size=64)
    placed = {(c.x, c.y) for c in layout.cells}
    assert len(placed) == len(layout.cells)
    for cell in layout.cells:
        assert cell.x == cell.column * 64 and cell.y == cell.row * 64


def test_a_cell_carries_the_pose_of_its_own_frame():
    layout = cs.plan(_records(), frame_size=64)
    records = _records()
    for cell, table in zip(layout.cells, cs.frame_table(), strict=True):
        assert cell.pose_name == table.animation
        assert cell.yaw == table.yaw
        assert cell.frame == table.frame
        assert cell.pose == records[table.animation][table.frame]["id"]


def test_a_clip_that_does_not_fill_the_table_is_refused_by_name():
    records = _records()
    records["walk"] = records["walk"][:7]
    with pytest.raises(ValueError, match="walk clip expands to 7 frames"):
        cs.plan(records)


def test_an_animation_the_table_does_not_have_is_refused():
    records = _records()
    records["cartwheel"] = []
    with pytest.raises(ValueError, match="not Troupe animations"):
        cs.plan(records)


def test_an_atlas_over_the_texture_limit_is_refused_before_anything_renders():
    with pytest.raises(ValueError, match="the limit is 8192"):
        cs.plan(_records(), frame_size=512)


def test_a_custom_frame_size_off_both_ladders_still_builds():
    """Task G: 40px names neither ``SIZES`` (Troupe's presets) nor
    ``sheet.FRAME_SIZES`` (the plain-sheet ladder), but it is a whole number
    inside ``MIN_FRAME_SIZE``..``MAX_FRAME_SIZE`` and the renderer lays it out
    exactly as it would any preset -- refusing it was the ladder mistaking
    itself for the actual limit."""
    assert 40 not in cs.SIZES
    layout = cs.plan(_records(), frame_size=40)
    assert (layout.width, layout.height) == (8 * 40, 32 * 40)
    assert len(layout.cells) == 256


@pytest.mark.parametrize("frame_size", [7, 257])
def test_a_frame_size_outside_the_custom_range_is_still_refused(frame_size):
    with pytest.raises(ValueError, match="frame_size must be"):
        cs.plan(_records(), frame_size=frame_size)


def test_v2_layout_gives_each_movement_its_own_direction_count_and_frames():
    layout = cs.resolve_layout(
        {
            "version": 2,
            "columns": 8,
            "movements": [
                {"key": "idle", "frames": 3, "directions": 1},
                {"key": "walk", "frames": 6, "directions": 4},
            ],
        }
    )
    records = clips.expand_clips("humanoid", layout)
    planned = cs.plan(records, frame_size=32, layout=layout)
    assert layout.cell_count == 27
    assert (planned.columns, planned.rows) == (8, 4)
    assert len(cs.spans(layout)) == 5
    assert [len(records[name]) for name in ("idle", "walk")] == [3, 6]


def test_one_frame_movements_sample_the_first_pose() -> None:
    layout = cs.resolve_layout(
        {"version": 2, "movements": [{"key": "attack", "frames": 1, "directions": 1}]}
    )
    records = clips.expand_clips("humanoid", layout)
    source = cliplib.clip_keys("humanoid", "attack")[0]
    assert len(records["attack"]) == 1
    for bone, rotation in source["bones"].items():
        assert records["attack"][0]["bones"][bone] == pytest.approx(rotation, abs=1e-4)


def test_troupe_columns_are_fixed_at_eight() -> None:
    with pytest.raises(ValueError, match="exactly 8 columns"):
        cs.resolve_layout(
            {"version": 2, "columns": 4, "movements": [{"key": "idle", "frames": 1}]}
        )


def test_a_v2_direction_object_missing_yaw_is_refused_by_name_not_by_typeerror():
    """The 2026-09-13 audit (troupe-01): a direction object with no ``yaw``
    used to surface as a raw ``float(None)`` ``TypeError`` instead of the
    named refusal its sibling branch (an out-of-range direction preset)
    already gives."""
    with pytest.raises(ValueError, match="yaw"):
        cs.resolve_layout(
            {
                "version": 2,
                "movements": [
                    {
                        "key": "idle",
                        "frames": 1,
                        "directions": [{"key": "front"}],
                    }
                ],
            }
        )


def test_a_resolved_v2_snapshot_round_trips_without_changing_cell_identity():
    first = cs.resolve_layout(
        {
            "version": 2,
            "movements": [{"key": "run", "frames": 12, "directions": 16}],
        }
    )
    second = cs.resolve_layout(first.as_dict())
    assert second == first
    assert cs.frame_table(second) == cs.frame_table(first)
    assert len(cs.frame_table(second)) == 192


def test_v2_warns_above_legacy_size_but_refuses_only_above_the_512_cap():
    large = cs.resolve_layout(
        {
            "version": 2,
            "movements": [{"key": "walk", "frames": 32, "directions": 16}],
        }
    )
    assert large.cell_count == cs.MAX_CELLS == 512
    assert large.cell_count > cs.WARN_CELLS
    with pytest.raises(ValueError, match="at most 512 cells"):
        cs.resolve_layout(
            {
                "version": 2,
                "movements": [
                    {"key": "walk", "frames": 32, "directions": 16},
                    {"key": "idle", "frames": 3, "directions": 1},
                ],
            }
        )


# --- the open clip vocabulary (v3) -------------------------------------------


#: ``cs.resolve_layout().as_dict()``, captured verbatim at 67a54124 -- before
#: the open vocabulary existed -- so a build from before this change and a
#: build from after it agree byte-for-byte on the one layout every default
#: Troupe sheet has ever carried.
_FROZEN_V2_LAYOUT = json.loads(
    '{"version": 2, "columns": 8, "movements": [{"key": "idle", "label": "Idle", '
    '"frames": 4, "loop": true, "duration_ms": 150, "directions": [{"key": "front", '
    '"label": "Front", "yaw": 0.0}, {"key": "front_left", "label": "Front Left", '
    '"yaw": 45.0}, {"key": "left", "label": "Left", "yaw": 90.0}, {"key": "back_left", '
    '"label": "Back Left", "yaw": 135.0}, {"key": "back", "label": "Back", "yaw": 180.0}, '
    '{"key": "back_right", "label": "Back Right", "yaw": 225.0}, {"key": "right", '
    '"label": "Right", "yaw": 270.0}, {"key": "front_right", "label": "Front Right", '
    '"yaw": 315.0}]}, {"key": "walk", "label": "Walk", "frames": 8, "loop": true, '
    '"duration_ms": 100, "directions": [{"key": "front", "label": "Front", "yaw": 0.0}, '
    '{"key": "front_left", "label": "Front Left", "yaw": 45.0}, {"key": "left", '
    '"label": "Left", "yaw": 90.0}, {"key": "back_left", "label": "Back Left", '
    '"yaw": 135.0}, {"key": "back", "label": "Back", "yaw": 180.0}, {"key": "back_right", '
    '"label": "Back Right", "yaw": 225.0}, {"key": "right", "label": "Right", '
    '"yaw": 270.0}, {"key": "front_right", "label": "Front Right", "yaw": 315.0}]}, '
    '{"key": "run", "label": "Run", "frames": 8, "loop": true, "duration_ms": 60, '
    '"directions": [{"key": "front", "label": "Front", "yaw": 0.0}, {"key": "front_left", '
    '"label": "Front Left", "yaw": 45.0}, {"key": "left", "label": "Left", "yaw": 90.0}, '
    '{"key": "back_left", "label": "Back Left", "yaw": 135.0}, {"key": "back", '
    '"label": "Back", "yaw": 180.0}, {"key": "back_right", "label": "Back Right", '
    '"yaw": 225.0}, {"key": "right", "label": "Right", "yaw": 270.0}, '
    '{"key": "front_right", "label": "Front Right", "yaw": 315.0}]}, {"key": "attack", '
    '"label": "Attack", "frames": 6, "loop": false, "duration_ms": 80, "directions": '
    '[{"key": "front", "label": "Front", "yaw": 0.0}, {"key": "front_left", '
    '"label": "Front Left", "yaw": 45.0}, {"key": "left", "label": "Left", "yaw": 90.0}, '
    '{"key": "back_left", "label": "Back Left", "yaw": 135.0}, {"key": "back", '
    '"label": "Back", "yaw": 180.0}, {"key": "back_right", "label": "Back Right", '
    '"yaw": 225.0}, {"key": "right", "label": "Right", "yaw": 270.0}, '
    '{"key": "front_right", "label": "Front Right", "yaw": 315.0}]}, {"key": "jump", '
    '"label": "Jump", "frames": 6, "loop": false, "duration_ms": 100, "directions": '
    '[{"key": "front", "label": "Front", "yaw": 0.0}, {"key": "front_left", '
    '"label": "Front Left", "yaw": 45.0}, {"key": "left", "label": "Left", "yaw": 90.0}, '
    '{"key": "back_left", "label": "Back Left", "yaw": 135.0}, {"key": "back", '
    '"label": "Back", "yaw": 180.0}, {"key": "back_right", "label": "Back Right", '
    '"yaw": 225.0}, {"key": "right", "label": "Right", "yaw": 270.0}, '
    '{"key": "front_right", "label": "Front Right", "yaw": 315.0}]}], "runs": '
    '[{"movement": "idle", "direction": "front", "yaw": 0.0, "start": 0, "end": 3}, '
    '{"movement": "idle", "direction": "front_left", "yaw": 45.0, "start": 4, "end": 7}, '
    '{"movement": "idle", "direction": "left", "yaw": 90.0, "start": 8, "end": 11}, '
    '{"movement": "idle", "direction": "back_left", "yaw": 135.0, "start": 12, "end": 15}, '
    '{"movement": "idle", "direction": "back", "yaw": 180.0, "start": 16, "end": 19}, '
    '{"movement": "idle", "direction": "back_right", "yaw": 225.0, "start": 20, "end": 23}, '
    '{"movement": "idle", "direction": "right", "yaw": 270.0, "start": 24, "end": 27}, '
    '{"movement": "idle", "direction": "front_right", "yaw": 315.0, "start": 28, "end": 31}, '
    '{"movement": "walk", "direction": "front", "yaw": 0.0, "start": 32, "end": 39}, '
    '{"movement": "walk", "direction": "front_left", "yaw": 45.0, "start": 40, "end": 47}, '
    '{"movement": "walk", "direction": "left", "yaw": 90.0, "start": 48, "end": 55}, '
    '{"movement": "walk", "direction": "back_left", "yaw": 135.0, "start": 56, "end": 63}, '
    '{"movement": "walk", "direction": "back", "yaw": 180.0, "start": 64, "end": 71}, '
    '{"movement": "walk", "direction": "back_right", "yaw": 225.0, "start": 72, "end": 79}, '
    '{"movement": "walk", "direction": "right", "yaw": 270.0, "start": 80, "end": 87}, '
    '{"movement": "walk", "direction": "front_right", "yaw": 315.0, "start": 88, "end": 95}, '
    '{"movement": "run", "direction": "front", "yaw": 0.0, "start": 96, "end": 103}, '
    '{"movement": "run", "direction": "front_left", "yaw": 45.0, "start": 104, "end": 111}, '
    '{"movement": "run", "direction": "left", "yaw": 90.0, "start": 112, "end": 119}, '
    '{"movement": "run", "direction": "back_left", "yaw": 135.0, "start": 120, "end": 127}, '
    '{"movement": "run", "direction": "back", "yaw": 180.0, "start": 128, "end": 135}, '
    '{"movement": "run", "direction": "back_right", "yaw": 225.0, "start": 136, "end": 143}, '
    '{"movement": "run", "direction": "right", "yaw": 270.0, "start": 144, "end": 151}, '
    '{"movement": "run", "direction": "front_right", "yaw": 315.0, "start": 152, "end": 159}, '
    '{"movement": "attack", "direction": "front", "yaw": 0.0, "start": 160, "end": 165}, '
    '{"movement": "attack", "direction": "front_left", "yaw": 45.0, "start": 166, "end": 171}, '
    '{"movement": "attack", "direction": "left", "yaw": 90.0, "start": 172, "end": 177}, '
    '{"movement": "attack", "direction": "back_left", "yaw": 135.0, "start": 178, "end": 183}, '
    '{"movement": "attack", "direction": "back", "yaw": 180.0, "start": 184, "end": 189}, '
    '{"movement": "attack", "direction": "back_right", "yaw": 225.0, "start": 190, "end": 195}, '
    '{"movement": "attack", "direction": "right", "yaw": 270.0, "start": 196, "end": 201}, '
    '{"movement": "attack", "direction": "front_right", "yaw": 315.0, "start": 202, "end": 207}, '
    '{"movement": "jump", "direction": "front", "yaw": 0.0, "start": 208, "end": 213}, '
    '{"movement": "jump", "direction": "front_left", "yaw": 45.0, "start": 214, "end": 219}, '
    '{"movement": "jump", "direction": "left", "yaw": 90.0, "start": 220, "end": 225}, '
    '{"movement": "jump", "direction": "back_left", "yaw": 135.0, "start": 226, "end": 231}, '
    '{"movement": "jump", "direction": "back", "yaw": 180.0, "start": 232, "end": 237}, '
    '{"movement": "jump", "direction": "back_right", "yaw": 225.0, "start": 238, "end": 243}, '
    '{"movement": "jump", "direction": "right", "yaw": 270.0, "start": 244, "end": 249}, '
    '{"movement": "jump", "direction": "front_right", "yaw": 315.0, "start": 250, "end": 255}], '
    '"cell_count": 256}'
)


def test_a_version_2_snapshot_resolves_exactly_as_it_always_did():
    """A build from before the open vocabulary and a build from after it must
    write the identical wire form for the one layout every default Troupe
    sheet has ever carried."""
    assert cs.resolve_layout().as_dict() == _FROZEN_V2_LAYOUT


def test_a_legacy_only_layout_still_writes_version_2():
    layout = cs.resolve_layout(
        {
            "version": 3,
            "movements": [
                {"key": "idle", "loop": True, "duration_ms": 150, "directions": 1},
                {"key": "walk", "loop": True, "duration_ms": 100, "directions": 1},
            ],
        }
    )
    assert layout.as_dict()["version"] == 2


def test_a_version_3_snapshot_carries_its_own_timing_without_a_library():
    layout = cs.resolve_layout(
        {
            "version": 3,
            "movements": [
                {
                    "key": "wave",
                    "loop": False,
                    "duration_ms": 90,
                    "frames": 5,
                    "directions": 1,
                }
            ],
        }
    )
    movement = layout.movements[0]
    assert (movement.loop, movement.duration_ms, movement.frames) == (False, 90, 5)
    as_dict = layout.as_dict()
    assert as_dict["version"] == 3
    assert cs.resolve_layout(as_dict) == layout


def test_a_layout_may_name_any_clip_the_skeleton_defines():
    timing = {"wave": cs.ClipTiming(frames=5, loop=False, duration_ms=90)}
    layout = cs.resolve_layout(
        {"version": 3, "movements": [{"key": "wave", "directions": 1}]},
        timing=timing,
    )
    movement = layout.movements[0]
    assert (movement.name, movement.frames, movement.loop, movement.duration_ms) == (
        "wave", 5, False, 90,
    )


def test_a_name_outside_the_skeletons_timing_is_refused_by_name():
    timing = {"wave": cs.ClipTiming(frames=5, loop=False, duration_ms=90)}
    with pytest.raises(ValueError, match="not a clip of this skeleton"):
        cs.resolve_layout(
            {"version": 3, "movements": [{"key": "cartwheel", "directions": 1}]},
            timing=timing,
        )


def test_a_global_fps_sets_every_frame_time():
    layout = cs.resolve_layout(
        {
            "version": 3,
            "fps": 12,
            "movements": [
                {"key": "idle", "frames": 4, "directions": 1},
                {"key": "walk", "frames": 8, "directions": 1},
            ],
        }
    )
    assert layout.fps == 12
    assert {m.duration_ms for m in layout.movements} == {round(1000 / 12)}


def test_a_global_fps_derives_default_frames_from_the_clip_length():
    layout = cs.resolve_layout(
        {"version": 3, "fps": 24, "movements": [{"key": "walk", "directions": 1}]}
    )
    movement = layout.movements[0]
    # walk is 8 frames at 100ms = 800ms of motion; at 24fps that keeps its
    # real length: round(8 * 100 * 24 / 1000) = 19 frames.
    assert movement.frames == 19
    assert movement.duration_ms == round(1000 / 24)


def test_a_stored_fps_snapshot_resolves_again_for_every_frame_rate():
    """The multiple-of-10 rule is for AUTHORED durations, not fps-derived ones.

    Every ``FPS_CHOICES`` value but 10 gives ``round(1000 / fps)`` a duration
    that is not a multiple of 10 (167, 125, 83, 67, 42, 33ms for 6/8/12/15/24/
    30 fps), and a stored sheet's ``params["layout"]`` -- ``as_dict()``'s v3
    output -- is exactly a snapshot re-resolved with no ``timing`` in hand,
    the way the worker (``_q_troupe.py``), a subset re-render
    (``service.troupe.rerender_charsheet``) and ``export_frames`` all resolve
    it. Before this fix, only ``fps=10`` round-tripped; every other rate's
    stored snapshot refused with "duration_ms must be a multiple of 10" on a
    row that had already rendered once.
    """
    for fps in cs.FPS_CHOICES:
        first = cs.resolve_layout(
            {
                "version": 3,
                "fps": fps,
                "movements": [{"key": "idle", "directions": 1}],
            }
        )
        second = cs.resolve_layout(first.as_dict())
        assert second == first
        assert cs.frame_table(second) == cs.frame_table(first)


def test_an_fps_off_the_ladder_is_refused():
    with pytest.raises(ValueError, match="fps must be one of"):
        cs.resolve_layout(
            {"version": 3, "fps": 20, "movements": [{"key": "idle", "directions": 1}]}
        )


def test_the_animation_block_states_the_fps_only_when_one_was_chosen():
    assert "fps" not in cs.animation_block()
    layout = cs.resolve_layout(
        {"version": 3, "fps": 15, "movements": [{"key": "idle", "directions": 1}]}
    )
    assert cs.animation_block(layout)["fps"] == 15


def test_a_movement_named_after_a_direction_is_refused():
    with pytest.raises(ValueError, match="ends in"):
        cs.resolve_layout(
            {"version": 2, "movements": [{"key": "walk_front", "frames": 4}]}
        )


def test_256_is_a_sheet_size():
    assert 256 in cs.SIZES


# --- the sidecar's animation block -------------------------------------------


def test_every_cell_gets_a_duration():
    block = cs.animation_block()
    assert len(block["frames"]) == 256
    assert [f["cell_index"] for f in block["frames"]] == list(range(256))
    assert all(f["duration_ms"] > 0 for f in block["frames"])


def test_a_frame_carries_its_own_animation_speed():
    block = cs.animation_block()
    by_cell = {f["cell_index"]: f["duration_ms"] for f in block["frames"]}
    for animation, _direction, start, _end, _loop in cs.spans():
        wanted = next(a[3] for a in cs.ANIMATIONS if a[0] == animation)
        assert by_cell[start] == wanted


def test_there_is_one_tag_per_animation_and_direction():
    tags = cs.animation_block()["tags"]
    assert len(tags) == len(cs.ANIMATIONS) * len(cs.DIRECTIONS)
    assert tags[0]["name"] == "idle_front"
    assert {t["name"] for t in tags} == {
        f"{a[0]}_{d[0]}" for a in cs.ANIMATIONS for d in cs.DIRECTIONS
    }


def test_a_one_shot_tag_plays_once_and_a_cycle_does_not_say_so():
    tags = {t["name"]: t for t in cs.animation_block()["tags"]}
    assert tags["attack_front"]["loop"] is False
    assert tags["attack_front"]["repeat"] == 1
    assert tags["walk_front"]["loop"] is True
    assert "repeat" not in tags["walk_front"]


def test_the_block_goes_into_a_sidecar_unchanged():
    layout = cs.plan(_records(), frame_size=64)
    meta = sheetlib.sidecar(
        layout,
        sheet_id="a" * 12,
        source_job="b" * 12,
        image="sheet.png",
        created=0.0,
        animation=cs.animation_block(),
    )
    assert meta["animation"]["tags"][0]["name"] == "idle_front"
    assert len(meta["animation"]["frames"]) == len(meta["cells"])


# --- the handoff into Inker --------------------------------------------------


def test_a_rendered_sheet_opens_in_inker_with_its_tags_and_timing():
    """Gap 4's other end: the sidecar's ``animation`` block goes straight into
    a document, so a character arrives in the editor already tagged per
    direction and already playing at the speed it was rendered for."""
    import numpy as np

    from warlock.kernels.pixel import sheetin

    layout = cs.plan(_records(), frame_size=16)
    block = cs.animation_block()
    atlas = np.zeros((layout.height, layout.width, 4), dtype=np.uint8)
    doc = sheetin.document_from_sheet(
        atlas, [c.as_dict(16) for c in layout.cells], block
    )
    assert len(doc.anim.frames) == 256
    # No DirectionalLayout: these cells are not the four named directions of a
    # fixed grid, and the tags carry the directions instead.
    assert doc.anim.layout is None
    tags = {t.name: t for t in doc.anim.tags}
    assert tags["walk_left"].loop is True
    assert (tags["walk_left"].start, tags["walk_left"].end) == (
        cs.spans()[10][2],
        cs.spans()[10][3],
    )
    assert tags["jump_back"].loop is False and tags["jump_back"].repeat == 1
    idle_ms = next(a[3] for a in cs.ANIMATIONS if a[0] == "idle")
    run_start = next(s[2] for s in cs.spans() if s[0] == "run")
    run_ms = next(a[3] for a in cs.ANIMATIONS if a[0] == "run")
    assert doc.anim.frames[0].duration_ms == idle_ms
    assert doc.anim.frames[run_start].duration_ms == run_ms


def test_the_general_tag_builder_still_produces_the_walk_sheet_tags():
    from warlock.kernels.pixel import sheetin

    tags = sheetin.walk_tags()
    assert [t.name for t in tags] == [
        "walk_front", "walk_left", "walk_right", "walk_back"
    ]
    assert [(t.start, t.end) for t in tags] == [(0, 3), (4, 7), (8, 11), (12, 15)]


def test_a_tag_that_runs_backwards_is_refused():
    from warlock.kernels.pixel import sheetin

    with pytest.raises(ValueError, match="covers frames 5-2"):
        sheetin.span_tags([{"name": "bad", "start": 5, "end": 2}])


# --- pose space --------------------------------------------------------------


def test_a_clip_records_the_frame_its_rotations_are_in():
    keys = [_pose("a"), _pose("b", 0.3)]
    node = sheetlib.interpolate_clip(keys, [2, 2], closed=True)
    delta = sheetlib.interpolate_clip(keys, [2, 2], closed=True, space="delta")
    # Absent on the default, so a record from the two-key door is byte-identical
    # to the one it always was.
    assert all("space" not in r for r in node)
    assert all(r["space"] == "delta" for r in delta)
    assert [r["bones"] for r in node] == [r["bones"] for r in delta]


def test_an_unknown_pose_space_is_refused():
    with pytest.raises(ValueError, match="space must be one of"):
        sheetlib.interpolate_clip([_pose("a"), _pose("b")], [2], space="world")


def test_the_two_modules_name_the_same_pose_spaces():
    """``sheet`` decides what a clip record says and ``blender_worker`` decides
    what it means; a spelling in one and not the other is a clip that silently
    applies in the wrong frame."""
    from warlock.pipelines import blender_worker

    assert sheetlib.POSE_SPACES == blender_worker.POSE_SPACES


def test_the_shipped_clips_are_authored_as_deltas_from_rest():
    """The finding that made them work at all. A node-local value bakes in the
    rest orientation of the skeleton it was authored against, so the same
    numbers on a rig whose joints were measured rather than fitted produce a
    different -- and in practice broken -- pose."""
    library = cliplib.clip_library("humanoid")
    assert library["space"] == "delta"
    assert {c["space"] for c in library["clips"]} == {"delta"}


def test_a_clip_library_that_says_nothing_is_read_as_node_local(tmp_path):
    (tmp_path / "old.json").write_text(
        '{"poses": [{"name": "a", "bones": {}}, {"name": "b", "bones": {}}], '
        '"clips": [{"name": "c", "keys": ["a", "b"], "segments": [1, 1], '
        '"closed": true}]}',
        encoding="utf-8",
    )
    library = cliplib._load_clip_library(tmp_path)["old"]
    assert library["space"] == "node"
    assert library["clips"][0]["space"] == "node"


def test_a_delta_clip_reaches_the_worker_as_a_cell_that_says_so():
    """The end of the thread: ``_q_rig`` puts ``pose_space`` on the cell only
    where the record carries one, so every pose row is the cell it always was."""
    import warlock._q_rig as q_rig

    source = Path(q_rig.__file__).read_text(encoding="utf-8")
    assert '"pose_space"] = space' in source
    assert 'r["space"] for r in records if r.get("space")' in source


# --- the pivot's unit --------------------------------------------------------


def test_a_pivot_projected_at_the_render_size_lands_inside_the_cell():
    """The sidecar's pivot is cell-relative, and Troupe renders at 512 while
    packing at the logical size -- so the projected number has to be converted
    or it names a point far outside the cell it belongs to."""
    # Centre-bottom of the render, which is where a grounded subject's origin
    # projects: it must come back as centre-bottom of the *cell*.
    half = cs.RENDER_SIZE / 2.0
    assert cs.pivot_in_cell((half, float(cs.RENDER_SIZE)), 32) == (16.0, 32.0)
    assert cs.pivot_in_cell((half, float(cs.RENDER_SIZE)), 64) == (32.0, 64.0)


def test_every_supported_size_keeps_the_pivot_within_its_cell():
    """The failure in its own terms: unconverted, a 32px cell recorded a pivot
    near (256, 470) -- sixteen cells below the sprite."""
    for size in cs.SIZES:
        x, y = cs.pivot_in_cell((256.0, 470.0), size)
        assert 0.0 <= x <= size, size
        assert 0.0 <= y <= size, size


def test_no_pivot_stays_no_pivot():
    """``sidecar`` falls back to centre-bottom when there is none, so None has
    to survive the conversion rather than becoming (0, 0)."""
    assert cs.pivot_in_cell(None, 32) is None
