"""Keyframe animation for a Clay rig: the pure model, the document's doors and the file.

Tranche 3 of the 2026-10-13 Clay rig plan. Every claim fails on the tree before the change
(``kernels.mesh.anim`` does not exist and ``ClayDoc`` has no animation doors).
"""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest

from realmspinner.kernels.mesh import anim
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import serialize as ser

from ._rigkit import rigged_doc

Q0 = [0.0, 0.0, 0.0, 1.0]
Q90 = [0.0, 0.0, float(np.sin(np.pi / 4)), float(np.cos(np.pi / 4))]


def _doc() -> tuple[bd.ClayDoc, int]:
    doc = bd.ClayDoc()
    bone = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Hip", mesh=bp.box(), bone=True)).uid
    return doc, bone


# -- channels and animations ------------------------------------------------------------------


def test_a_channel_is_frozen_and_owns_its_arrays() -> None:
    times = np.array([0.0, 1.0])
    channel = anim.Channel(times, [Q0, Q90])
    times[0] = 5.0
    assert channel.times[0] == 0.0 and not channel.times.flags.writeable
    assert not channel.values.flags.writeable


@pytest.mark.parametrize(
    ("times", "values", "message"),
    [
        ([], np.zeros((0, 3)), "at least one key"),
        ([0.0, 0.0], [Q0, Q0], "strictly increase"),
        ([1.0, 0.5], [Q0, Q0], "strictly increase"),
        ([0.0], [[1.0, 2.0]], "3 or 4 numbers"),
        ([0.0], [[0.0, 0.0, 0.0, 0.0]], "zero quaternion"),
        ([-1.0], [Q0], "outside 0.."),
        ([anim.MAX_ANIMATION_SECONDS + 1], [Q0], "outside 0.."),
        ([float("nan")], [Q0], "finite"),
        ([0.0], [[0.0, float("inf"), 0.0, 1.0]], "finite"),
    ],
)
def test_a_channel_refuses_what_cannot_be_played(times, values, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        anim.Channel(times, values)


def test_a_channel_refuses_more_keys_than_the_cap() -> None:
    n = anim.MAX_KEYS_PER_CHANNEL + 1
    with pytest.raises(ValueError, match="keys, past"):
        anim.Channel(np.arange(n) * 0.001, np.tile(Q0, (n, 1)))


def test_keys_are_set_inserted_replaced_and_deleted_without_mutating_the_original() -> None:
    base = anim.Animation("Walk")
    one = base.with_key(7, "rotation", 1.0, Q90)
    two = one.with_key(7, "rotation", 0.0, Q0)
    assert base.channels == {} and len(one.channels[7]["rotation"]) == 1
    assert two.channels[7]["rotation"].times.tolist() == [0.0, 1.0], "inserted in time order"
    again = two.with_key(7, "rotation", 1.0 + 1e-8, Q0)
    assert len(again.channels[7]["rotation"]) == 2, "a time within epsilon is the same key"
    assert np.allclose(again.channels[7]["rotation"].values[1], Q0)
    gone = two.without_key(7, "rotation", 0.0)
    assert gone.channels[7]["rotation"].times.tolist() == [1.0]
    empty = gone.without_key(7, "rotation", 1.0)
    assert empty.channels == {}, "a channel with no keys is dropped"
    assert two.without_key(7, "rotation", 0.5) is two, "no such key changes nothing"


def test_a_rotation_key_is_normalised_and_a_zero_one_refused() -> None:
    animation = anim.Animation("A").with_key(1, "rotation", 0.0, [0, 0, 0, 2.0])
    assert np.allclose(animation.channels[1]["rotation"].values[0], Q0)
    with pytest.raises(ValueError, match="non-zero"):
        animation.with_key(1, "rotation", 1.0, [0, 0, 0, 0])
    with pytest.raises(ValueError, match="not an animatable"):
        animation.with_key(1, "colour", 0.0, [0, 0, 0])
    with pytest.raises(ValueError, match="3 numbers"):
        animation.with_key(1, "translation", 0.0, [0, 0, 0, 1])


def test_moving_a_key_takes_the_place_of_one_already_there() -> None:
    a = (
        anim.Animation("A")
        .with_key(1, "translation", 0.0, [0, 0, 0])
        .with_key(1, "translation", 1.0, [1, 0, 0])
    )
    moved = a.moved(1, "translation", 0.0, 1.0)
    assert moved.channels[1]["translation"].times.tolist() == [1.0]
    assert np.allclose(moved.channels[1]["translation"].values[0], [0, 0, 0])


def test_duration_key_times_and_bones() -> None:
    a = (
        anim.Animation("A")
        .with_key(1, "translation", 0.5, [0, 0, 0])
        .with_key(2, "scale", 2.0, [1, 1, 1])
        .with_key(2, "scale", 0.5, [2, 2, 2])
    )
    assert a.duration == 2.0 and a.key_times() == [0.5, 2.0] and a.key_times(1) == [0.5]
    assert a.bones() == [1, 2] and anim.Animation("empty").duration == 0.0


def test_without_bones_drops_their_channels_and_is_identity_when_nothing_matches() -> None:
    a = (
        anim.Animation("A")
        .with_key(1, "scale", 0.0, [1, 1, 1])
        .with_key(2, "scale", 0.0, [1, 1, 1])
    )
    assert a.without_bones([1]).bones() == [2]
    assert a.without_bones([99]) is a


def test_animation_validates_its_name_interpolation_and_property_widths() -> None:
    with pytest.raises(ValueError, match="needs a name"):
        anim.Animation("  ")
    with pytest.raises(ValueError, match="interpolation"):
        anim.Animation("A", interpolation="cubic")
    channel = anim.Channel([0.0], [[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="rotation channel holds 4"):
        anim.Animation("A", channels={1: {"rotation": channel}})


# -- sampling ---------------------------------------------------------------------------------


def test_linear_translation_and_scale_interpolate_and_hold_outside_the_keys() -> None:
    channel = anim.Channel([1.0, 3.0], [[0, 0, 0], [2, 4, 6]])
    assert np.allclose(anim.sample(channel, 0.0), [0, 0, 0]), "held before the first key"
    assert np.allclose(anim.sample(channel, 2.0), [1, 2, 3])
    assert np.allclose(anim.sample(channel, 9.0), [2, 4, 6]), "held after the last key"


def test_rotation_takes_the_shorter_arc_and_stays_unit_length() -> None:
    q_a = [0.0, 0.0, 0.0, 1.0]
    q_b = [0.0, 0.0, -float(np.sin(np.pi / 4)), -float(np.cos(np.pi / 4))]  # -Q90: same turn
    channel = anim.Channel([0.0, 1.0], [q_a, q_b])
    mid = anim.sample(channel, 0.5)
    assert np.linalg.norm(mid) == pytest.approx(1.0)
    angle = 2 * np.arctan2(abs(mid[2]), abs(mid[3]))
    assert angle == pytest.approx(np.pi / 4, abs=1e-6), "halfway to a quarter turn, not 270 degrees"


def test_step_holds_the_last_key_until_the_next() -> None:
    channel = anim.Channel([0.0, 1.0], [[0, 0, 0], [10, 0, 0]])
    assert np.allclose(anim.sample(channel, 0.99, "step"), [0, 0, 0])
    assert np.allclose(anim.sample(channel, 1.0, "step"), [10, 0, 0])


def test_pose_wraps_a_looping_animation_and_clamps_a_one_shot() -> None:
    a = (
        anim.Animation("A")
        .with_key(1, "translation", 0.0, [0, 0, 0])
        .with_key(1, "translation", 2.0, [2, 0, 0])
    )
    assert np.allclose(anim.pose(a, 3.0)[1]["translation"], [1, 0, 0]), "3 s into a 2 s loop"
    one_shot = a.with_loop(False)
    assert np.allclose(anim.pose(one_shot, 3.0)[1]["translation"], [2, 0, 0])
    assert np.allclose(anim.pose(one_shot, -1.0)[1]["translation"], [0, 0, 0])
    assert anim.pose(None, 1.0) == {}, "no animation is the rest pose"
    assert set(anim.pose(a, 1.0)[1]) == {"translation"}, "only animated properties appear"


def test_cubic_spline_is_resampled_to_linear_keys_through_its_values() -> None:
    times = np.array([0.0, 1.0])
    zero = np.zeros(3)
    # in-tangent, value, out-tangent per key
    values = np.array([zero, zero, zero, zero, [1.0, 0, 0], zero])
    grid, out = anim.resample_cubic(times, values, fps=4.0, width=3)
    assert grid[0] == 0.0 and grid[-1] == 1.0 and len(grid) >= 5
    assert np.allclose(out[0], 0) and np.allclose(out[-1], [1, 0, 0])
    assert (np.diff(out[:, 0]) >= 0).all(), "a zero-tangent Hermite is monotone"


# -- the rig state ---------------------------------------------------------------------------


def test_rig_state_validates_fps_count_and_name_uniqueness_and_is_falsy_at_its_defaults() -> None:
    assert not anim.RigState()
    assert anim.RigState(fps=30)
    with pytest.raises(ValueError, match="fps"):
        anim.RigState(fps=25)
    with pytest.raises(ValueError, match="share a name"):
        anim.RigState((anim.Animation("A"), anim.Animation("A")))
    with pytest.raises(ValueError, match="past the 64"):
        anim.RigState(tuple(anim.Animation(f"A{i}") for i in range(65)))


# -- the document's doors --------------------------------------------------------------------


def test_adding_animations_names_them_uniquely_selects_the_new_one_and_is_one_step_each() -> None:
    doc, bone = _doc()
    steps = len(doc.history)
    assert doc.add_animation("Walk") == 0 and doc.add_animation("Walk") == 1
    assert [a.name for a in doc.rig_state.animations] == ["Walk", "Walk.001"]
    assert doc.active_animation == 1 and len(doc.history) == steps + 2
    doc.undo()
    assert [a.name for a in doc.rig_state.animations] == ["Walk"]
    assert doc.active_animation == 0, "an undo cannot leave the selection past the end"
    doc.undo()
    assert doc.rig_state.animations == () and doc.active_animation is None


def test_the_animation_cap_is_a_refusal() -> None:
    doc, _ = _doc()
    for i in range(anim.MAX_ANIMATIONS):
        doc.add_animation(f"A{i}")
    with pytest.raises(el.OpError, match="at most 64"):
        doc.add_animation("one too many")


def test_keys_go_through_the_active_animation_and_each_is_one_undo_step() -> None:
    doc, bone = _doc()
    doc.add_animation("Walk")
    steps = len(doc.history)
    assert doc.set_key(bone, "rotation", 0.0, Q0) is True
    assert doc.set_key(bone, "rotation", 1.0, Q90) is True
    assert len(doc.history) == steps + 2
    assert doc.animation().duration == 1.0
    assert doc.set_key(bone, "rotation", 1.0, Q90) is False, "the same key again changes nothing"
    assert len(doc.history) == steps + 2
    assert doc.move_key(bone, "rotation", 1.0, 2.0) and doc.animation().duration == 2.0
    assert doc.delete_key(bone, "rotation", 2.0) and doc.animation().duration == 0.0
    assert doc.undo() and doc.animation().duration == 2.0


def test_a_key_needs_a_bone_an_animation_and_a_legal_time() -> None:
    doc, bone = _doc()
    plain = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box())).uid
    with pytest.raises(el.OpError, match="Select or create an animation"):
        doc.set_key(bone, "rotation", 0.0, Q0)
    doc.add_animation("Walk")
    with pytest.raises(el.OpError, match="not a bone"):
        doc.set_key(plain, "rotation", 0.0, Q0)
    with pytest.raises(ValueError, match="key time"):
        doc.set_key(bone, "rotation", -1.0, Q0)


def test_renaming_looping_interpolation_and_fps_are_edits_that_refuse_what_they_must() -> None:
    doc, bone = _doc()
    doc.add_animation("A")
    doc.add_animation("B")
    assert doc.rename_animation(0, "Run") is True
    with pytest.raises(el.OpError, match="already an animation"):
        doc.rename_animation(1, "Run")
    with pytest.raises(el.OpError, match="needs a name"):
        doc.rename_animation(1, " ")
    assert doc.rename_animation(0, "Run") is False
    assert doc.set_animation_loop(0, False) and not doc.animation(0).loop
    assert doc.set_animation_interpolation(0, "step") and doc.animation(0).interpolation == "step"
    assert doc.set_fps(30) is True and doc.rig_state.fps == 30 and doc.set_fps(30) is False
    with pytest.raises(el.OpError, match="frame rate"):
        doc.set_fps(25)
    assert doc.delete_animation(0) is True and [a.name for a in doc.rig_state.animations] == ["B"]
    assert doc.delete_animation(5) is False


def test_duplicating_an_animation_copies_its_keys_under_a_new_name() -> None:
    doc, bone = _doc()
    doc.add_animation("Walk")
    doc.set_key(bone, "translation", 0.0, [1, 2, 3])
    copy = doc.add_animation("Walk", copy_of=0)
    assert doc.animation(copy).name == "Walk.001"
    assert np.allclose(doc.animation(copy).channels[bone]["translation"].values[0], [1, 2, 3])


def test_selecting_an_animation_is_not_an_edit() -> None:
    doc, bone = _doc()
    doc.add_animation("A")
    doc.add_animation("B")
    steps, rev = len(doc.history), doc.rev
    doc.set_active_animation(0)
    assert doc.active_animation == 0 and len(doc.history) == steps and doc.rev == rev
    doc.set_active_animation(9)
    assert doc.active_animation is None


def test_unmarking_or_deleting_a_bone_drops_its_keys_in_the_same_undo_step() -> None:
    doc, bone = _doc()
    doc.add_animation("Walk")
    doc.set_key(bone, "rotation", 0.0, Q0)
    steps = len(doc.history)
    assert doc.set_bone([bone], False) is True
    assert doc.animation().channels == {} and len(doc.history) == steps + 1
    assert doc.undo()
    assert bone in doc.animation().channels and doc.by_uid(bone).bone is True
    assert doc.remove_objects([bone]) == []
    assert doc.animation().channels == {}
    assert doc.undo()
    assert bone in doc.animation().channels and doc.by_uid(bone).bone is True


def test_a_rig_state_that_animates_a_non_bone_is_refused() -> None:
    doc, bone = _doc()
    plain = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box())).uid
    bad = anim.Animation("A", channels={plain: {"scale": anim.Channel([0.0], [[1, 1, 1]])}})
    with pytest.raises(el.OpError, match="not a bone"):
        doc.set_rig_state(anim.RigState((bad,)))


def test_the_undo_budget_counts_the_channels_once_however_many_steps_share_them() -> None:
    from realmspinner.kernels.mesh.edits import RigEdit

    channel = anim.Channel(np.arange(100) * 0.01, np.tile(Q0, (100, 1)))
    one = anim.Animation("A", channels={1: {"rotation": channel}})
    before = anim.RigState((one,))
    after = anim.RigState((one, anim.Animation("B")))
    assert RigEdit(before, after).cost == channel.times.nbytes + channel.values.nbytes


# -- the file -----------------------------------------------------------------------------------


def _animated() -> tuple[bd.ClayDoc, int, bytes]:
    doc, bone, _body = rigged_doc()
    doc.add_animation("Walk")
    doc.add_animation("Idle")
    doc.set_active_animation(0)
    doc.set_key(bone, "rotation", 0.0, Q0, animation=0)
    doc.set_key(bone, "rotation", 1.0, Q90, animation=0)
    doc.set_key(bone, "translation", 0.5, [0, 1, 0], animation=0)
    doc.set_key(bone, "scale", 0.25, [2, 2, 2], animation=1)
    doc.set_fps(30)
    doc.set_animation_loop(1, False)
    doc.set_animation_interpolation(1, "step")
    return doc, bone, ser.rblk_bytes(doc)


def test_animations_round_trip_exactly_and_the_selection_comes_back() -> None:
    doc, bone, data = _animated()
    out = ser.read_rblk(data)
    assert out.rig_state.fps == 30 and out.active_animation == 0
    assert [a.name for a in out.rig_state.animations] == ["Walk", "Idle"]
    walk, idle = out.animation(0), out.animation(1)
    assert walk.loop and walk.interpolation == "linear"
    assert not idle.loop and idle.interpolation == "step"
    for prop in ("rotation", "translation"):
        src = doc.animation(0).channels[bone][prop]
        assert np.array_equal(walk.channels[bone][prop].times, src.times)
        assert np.array_equal(walk.channels[bone][prop].values, src.values)
    assert out.undo() is False, "a reopened file has no history to undo"


def test_a_saved_animated_document_is_byte_identical_twice_and_after_a_reopen() -> None:
    doc, _bone, data = _animated()
    assert ser.rblk_bytes(doc) == data
    assert ser.rblk_bytes(ser.read_rblk(data)) == data


def test_a_document_without_motion_writes_no_rig_key_and_no_anims_member() -> None:
    doc, _bone, _body = rigged_doc()
    data = ser.rblk_bytes(doc)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        scene = json.loads(zf.read(ser.SCENE))
        assert "rig" not in scene
        assert not [n for n in zf.namelist() if n.startswith("anims/")]
    assert ser.read_rblk(data).rig_state.animations == ()


def _rewrite(data: bytes, edit, *, drop: str = "") -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as src, zipfile.ZipFile(out, "w") as dst:
        for info in src.infolist():
            if info.filename == drop:
                continue
            body = src.read(info.filename)
            if info.filename == ser.SCENE:
                scene = json.loads(body)
                edit(scene)
                body = json.dumps(scene).encode()
            dst.writestr(info, body)
    return out.getvalue()


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda s: s["rig"].__setitem__("fps", 25), "fps"),
        (lambda s: s["rig"].__setitem__("fps", "fast"), "frame rate"),
        (lambda s: s.__setitem__("rig", [1]), "not an object"),
        (lambda s: s["rig"].__setitem__("animations", {}), "not a list"),
        (lambda s: s["rig"]["animations"].__setitem__(0, 5), "not readable"),
        (
            lambda s: s["rig"]["animations"][0].__setitem__("interpolation", "cubic"),
            "interpolation",
        ),
        (
            lambda s: s["rig"]["animations"][0]["channels"].__setitem__("99999", ["scale"]),
            "not a bone",
        ),
        (lambda s: s["rig"]["animations"][0].__setitem__("name", "Idle"), "share a name"),
        (
            lambda s: s["rig"]["animations"].__iadd__([s["rig"]["animations"][0]] * 70),
            "past the 64",
        ),
    ],
)
def test_the_reader_refuses_motion_that_does_not_hold_together_by_name(edit, message: str) -> None:
    _doc_, _bone, data = _animated()
    with pytest.raises(ValueError, match=message):
        ser.read_rblk(_rewrite(data, edit))


def test_the_reader_refuses_a_missing_member_a_bad_property_and_a_corrupt_channel() -> None:
    _d, bone, data = _animated()
    with pytest.raises(ValueError, match="is missing"):
        ser.read_rblk(_rewrite(data, lambda s: None, drop="anims/0.npz"))
    with pytest.raises(ValueError, match="not an animatable"):
        ser.read_rblk(
            _rewrite(
                data, lambda s: s["rig"]["animations"][0]["channels"][str(bone)].append("colour")
            )
        )
    with pytest.raises(ValueError, match="not usable"):
        # Claims a scale channel the archive has no arrays for.
        ser.read_rblk(
            _rewrite(
                data, lambda s: s["rig"]["animations"][0]["channels"][str(bone)].append("scale")
            )
        )


def test_the_writer_refuses_motion_the_reader_would_refuse() -> None:
    doc, bone, _data = _animated()
    plain = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box())).uid
    doc.rig_state = doc.rig_state.with_animations(
        [anim.Animation("Bad", channels={plain: {"scale": anim.Channel([0.0], [[1, 1, 1]])}})]
    )
    with pytest.raises(ValueError, match="not a bone"):
        ser.rblk_bytes(doc)


def test_a_version_4_file_with_a_stray_rig_key_ignores_it() -> None:
    _d, _bone, data = _animated()
    old = _rewrite(data, lambda s: s.__setitem__("version", 4))
    out = ser.read_rblk(old)
    assert out.rig_state.animations == () and out.active_animation is None
