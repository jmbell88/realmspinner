"""Keying the pose: the doors, the viewport drag and the Rig-group ops.

Tranche 3 (U2) of the 2026-10-13 Clay rig plan. Posing a bone in Pose mode never writes the
bone's rest transform: it keys the playhead (auto-key) or holds an unkeyed pose, as one step.
Each claim fails on the tree before the change.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import rig
from realmspinner.studio.modes.clay import ops as clay_ops
from realmspinner.studio.modes.clay.state import ClayState
from realmspinner.studio.modes.clay.ui import view as clay_view

from ._rigkit import skin_for, weighted

Q0 = np.array([0.0, 0.0, 0.0, 1.0])
Q90 = np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
RECT = (0.0, 0.0, 200.0, 150.0)


def _add(doc: bd.ClayDoc, name: str, **kw) -> int:
    kw.setdefault("mesh", bp.box())
    return doc.add_object(bd.Obj(uid=bd.new_uid(), name=name, **kw)).uid


def _rig() -> tuple[bd.ClayDoc, int, int]:
    """Hip (a bone at y=1) with an Arm bone a metre along x."""
    doc = bd.ClayDoc()
    hip = _add(doc, "Hip", bone=True, translation=(0, 1, 0))
    arm = _add(doc, "Arm", bone=True, parent=hip, translation=(1, 0, 0))
    doc.set_pose_mode(True)
    return doc, hip, arm


def _steps(doc: bd.ClayDoc) -> int:
    return len(doc.history)


# -- the doors ----------------------------------------------------------------------------------


def test_the_first_key_from_nothing_creates_the_animation_and_is_one_undo_step() -> None:
    doc, hip, _ = _rig()
    assert doc.animation() is None
    steps = _steps(doc)
    assert doc.key_pose({hip: {"rotation": Q90}})
    assert _steps(doc) == steps + 1, "creating the clip and keying it is one step"
    assert doc.animation() is not None
    assert np.allclose(doc.animation().channels[hip]["rotation"].values[0], Q90)
    doc.undo()
    assert doc.animation() is None and doc.rig_state.animations == ()


def test_keying_a_value_already_keyed_there_pushes_nothing() -> None:
    doc, hip, _ = _rig()
    doc.key_pose({hip: {"rotation": Q90}})
    steps = _steps(doc)
    assert not doc.key_pose({hip: {"rotation": Q90}})
    assert _steps(doc) == steps


def test_a_key_lands_at_the_playhead_and_replaces_the_one_already_there() -> None:
    doc, hip, _ = _rig()
    doc.set_rig_time(0.5)
    doc.key_pose({hip: {"translation": [0, 2, 0]}})
    doc.key_pose({hip: {"translation": [0, 3, 0]}})
    channel = doc.animation().channels[hip]["translation"]
    assert list(channel.times) == [0.5] and np.allclose(channel.values[0], [0, 3, 0])


def test_keying_a_property_stops_it_being_held_as_a_live_pose() -> None:
    doc, hip, _ = _rig()
    doc.set_live_pose(hip, "rotation", Q90)
    doc.set_live_pose(hip, "scale", [2, 2, 2])
    doc.key_pose({hip: {"rotation": Q90}})
    assert set(doc.live_pose[hip]) == {"scale"}


def test_keying_something_that_is_not_a_bone_is_refused() -> None:
    doc, hip, _ = _rig()
    box = _add(doc, "Box")
    with pytest.raises(Exception, match="bone"):
        doc.key_pose({box: {"rotation": Q90}})


def test_with_auto_key_off_a_pose_is_held_not_keyed_and_the_playhead_drops_it() -> None:
    doc, hip, _ = _rig()
    doc.auto_key = False
    steps = _steps(doc)
    doc.pose_bones({hip: {"rotation": Q90}})
    assert _steps(doc) == steps and doc.animation() is None
    assert np.allclose(doc.pose_overrides()[hip]["rotation"], Q90)
    doc.set_rig_time(1.0)
    assert doc.pose_overrides() == {}


def test_insert_key_keys_all_three_properties_as_posed() -> None:
    doc, hip, _ = _rig()
    doc.auto_key = False
    doc.pose_bones({hip: {"rotation": Q90}})
    assert doc.insert_key([hip])
    props = doc.animation().channels[hip]
    assert set(props) == {"translation", "rotation", "scale"}
    assert np.allclose(props["rotation"].values[0], Q90)
    assert np.allclose(props["translation"].values[0], [0, 1, 0]), "unposed ones key their rest"
    assert hip not in doc.live_pose or "rotation" not in doc.live_pose[hip]


def test_delete_keys_removes_only_the_keys_at_the_playhead() -> None:
    doc, hip, arm = _rig()
    for time in (0.0, 1.0):
        doc.set_rig_time(time)
        doc.insert_key([hip, arm])
    doc.set_rig_time(1.0)
    assert doc.delete_keys_at([hip])
    assert list(doc.animation().channels[hip]["rotation"].times) == [0.0]
    assert list(doc.animation().channels[arm]["rotation"].times) == [0.0, 1.0]
    assert not doc.delete_keys_at([hip]), "nothing left there"


def test_reset_pose_poses_the_bone_at_its_rest_transform() -> None:
    doc, hip, _ = _rig()
    doc.key_pose({hip: {"rotation": Q90}})
    assert doc.reset_pose([hip])
    assert np.allclose(doc.pose_overrides()[hip]["rotation"], Q0)
    assert np.allclose(doc.animation().channels[hip]["rotation"].values[0], Q0)


def test_apply_pose_as_rest_moves_the_rest_to_the_pose_and_keeps_every_point_where_it_is() -> None:
    doc, hip, arm = _rig()
    body = _add(doc, "Body", mesh=weighted(bp.box()), skin=skin_for(hip))
    doc.key_pose({hip: {"rotation": Q90}})
    over = doc.pose_overrides()
    before_body = rig.skin_positions(
        doc.by_uid(body).mesh.positions,
        doc.by_uid(body).mesh.joints,
        doc.by_uid(body).mesh.weights,
        doc.skin_palette(doc.by_uid(body), over),
        doc._world_in(body, doc._uid_map()),
    )
    arm_world = doc.posed_world(arm, over)
    steps = _steps(doc)

    assert doc.apply_pose_as_rest()
    assert _steps(doc) == steps + 1, "bones and meshes change as one step"
    assert np.allclose(doc.by_uid(hip).rotation, Q90, atol=1e-6)
    assert np.allclose(doc.world_matrix(arm), arm_world, atol=1e-6)
    assert np.allclose(doc.by_uid(body).mesh.positions, before_body, atol=1e-5)
    # The keys are absolute, so the same pose is still shown -- and it is now the rest.
    assert np.allclose(doc.skin_palette(doc.by_uid(body), doc.pose_overrides()), np.eye(4)[None])

    doc.undo()
    assert np.allclose(doc.by_uid(hip).rotation, Q0)
    assert not np.allclose(doc.by_uid(body).mesh.positions, before_body)


def test_apply_pose_as_rest_with_nothing_posed_does_nothing() -> None:
    doc, *_ = _rig()
    steps = _steps(doc)
    assert not doc.apply_pose_as_rest() and _steps(doc) == steps


# -- the viewport drag ----------------------------------------------------------------------------


class _Ctx:
    def __init__(self) -> None:
        self.state = type("S", (), {"clay": ClayState()})()
        self.clay_view = None
        self.toasts: list[tuple[str, str]] = []

    def toast(self, message: str, level: str = "info") -> None:
        self.toasts.append((message, level))


@pytest.fixture
def view(gl):
    ctx = _Ctx()
    v = clay_view.ClayView(gl, ctx)
    ctx.clay_view = v
    v.app_ctx = ctx
    yield v
    v.release()


def _grab(view, doc: bd.ClayDoc, kind: str, to: tuple[float, float]) -> bool:
    view.draw(doc, RECT, 0.016)
    view._last_mouse = (100.0, 75.0)
    if not view.begin_keyboard_drag(doc, kind):
        return False
    view._drag_keyboard(doc, to)
    return True


def test_moving_a_bone_keys_translation_only_and_leaves_its_rest_alone(view) -> None:
    doc, hip, _ = _rig()
    doc.select([hip])
    rest = np.array(doc.by_uid(hip).translation, copy=True)
    steps = _steps(doc)
    assert _grab(view, doc, "move", (160.0, 75.0))
    assert hip in doc.live_pose, "the drag is held live until release"
    assert _steps(doc) == steps, "nothing is recorded mid-drag"
    view.settle_drag(doc)

    assert _steps(doc) == steps + 1 and doc.history.top.label == "Pose"
    assert set(doc.animation().channels[hip]) == {"translation"}
    assert not np.allclose(doc.animation().channels[hip]["translation"].values[0], rest)
    assert np.array_equal(doc.by_uid(hip).translation, rest), "the rest pose is never written"
    assert doc.live_pose == {}


def test_turning_a_bone_keys_rotation_about_its_own_pivot(view) -> None:
    doc, hip, arm = _rig()
    doc.select([hip])
    pivot = np.array(doc.world_matrix(hip)[:3, 3], copy=True)
    assert _grab(view, doc, "rotate", (160.0, 75.0))
    view.settle_drag(doc)
    assert set(doc.animation().channels[hip]) == {"rotation"}
    posed = doc.posed_world(hip, doc.pose_overrides())
    assert np.allclose(posed[:3, 3], pivot), "a joint turns about its own origin"
    assert not np.allclose(doc.animation().channels[hip]["rotation"].values[0], Q0)


def test_the_gizmo_sits_on_the_bones_pivot_not_the_centre_of_its_box(view) -> None:
    doc = bd.ClayDoc()
    box = bp.box()
    high = replace(box, positions=box.positions + np.array([0.0, 2.0, 0.0], dtype="f4"))
    off = _add(doc, "Wing", bone=True, translation=(4, 0, 0), mesh=high)  # box centre is 2 m up
    doc.set_pose_mode(True)
    doc.select([off])
    view.draw(doc, RECT, 0.016)
    assert np.allclose(view.selection_centre(doc), [4, 0, 0])


def test_with_auto_key_off_a_drag_is_held_and_records_no_step(view) -> None:
    doc, hip, _ = _rig()
    doc.auto_key = False
    doc.select([hip])
    steps = _steps(doc)
    assert _grab(view, doc, "move", (160.0, 75.0))
    view.settle_drag(doc)
    assert _steps(doc) == steps and doc.animation() is None
    assert set(doc.live_pose[hip]) == {"translation"}


def test_a_held_pose_survives_a_later_drag_of_another_property(view) -> None:
    doc, hip, _ = _rig()
    doc.auto_key = False
    doc.select([hip])
    doc.set_live_pose(hip, "rotation", Q90)
    assert _grab(view, doc, "move", (160.0, 75.0))
    view.settle_drag(doc)
    assert set(doc.live_pose[hip]) == {"translation", "rotation"}
    assert np.allclose(doc.live_pose[hip]["rotation"], Q90)


def test_esc_puts_the_held_pose_back_and_records_nothing(view) -> None:
    doc, hip, _ = _rig()
    doc.auto_key = False
    doc.select([hip])
    doc.set_live_pose(hip, "rotation", Q90)
    steps = _steps(doc)
    assert _grab(view, doc, "move", (160.0, 75.0))
    assert "translation" in doc.live_pose[hip]
    assert view.cancel_drag(doc)
    assert _steps(doc) == steps
    assert set(doc.live_pose[hip]) == {"rotation"}
    assert np.allclose(doc.live_pose[hip]["rotation"], Q90)


def test_a_drag_that_moved_nothing_keys_nothing(view) -> None:
    doc, hip, _ = _rig()
    doc.select([hip])
    steps = _steps(doc)
    assert _grab(view, doc, "move", (100.0, 75.0))
    view.settle_drag(doc)
    assert _steps(doc) == steps and doc.animation() is None


def test_a_shape_that_is_not_a_bone_cannot_be_posed(view) -> None:
    doc, hip, _ = _rig()
    box = _add(doc, "Box")
    doc.select([box])
    assert not _grab(view, doc, "move", (160.0, 75.0))
    assert any("bone" in message for message, _level in view.app_ctx.toasts)
    assert not view.dragging


def test_a_bone_posed_under_a_posed_parent_keys_its_local_transform(view) -> None:
    doc, hip, arm = _rig()
    doc.key_pose({hip: {"rotation": Q90}})
    doc.select([arm])
    assert _grab(view, doc, "move", (160.0, 75.0))
    view.settle_drag(doc)
    local = doc.animation().channels[arm]["translation"].values[0]
    world = doc.posed_world(arm, doc.pose_overrides())
    parent = doc.posed_world(hip, doc.pose_overrides())
    assert np.allclose(parent @ np.append(local, 1.0), world @ [0, 0, 0, 1.0], atol=1e-6)


# -- the ops --------------------------------------------------------------------------------------


def _run(doc: bd.ClayDoc, name: str) -> bool:
    ctx = _Ctx()
    return clay_ops.run(ctx, doc, clay_ops.get(name))


@pytest.mark.parametrize("name", ["insert-key", "delete-key", "reset-pose"])
def test_the_pose_ops_need_pose_mode_and_a_selected_bone_and_say_so(name: str) -> None:
    doc, hip, _ = _rig()
    doc.key_pose({hip: {"rotation": Q90}})
    op = clay_ops.get(name)
    assert op.modes == ("object",)
    doc.select([hip])
    assert op.enabled(doc)
    doc.select([])
    assert not op.enabled(doc) and "bone" in clay_ops.reason_for(op, doc)
    doc.select([hip])
    doc.set_pose_mode(False)
    assert not op.enabled(doc) and "Pose mode" in clay_ops.reason_for(op, doc)


def test_the_rows_run_through_the_op_layer_as_one_labelled_step_each() -> None:
    doc, hip, _ = _rig()
    doc.select([hip])
    doc.set_rig_time(0.5)
    steps = _steps(doc)
    assert _run(doc, "insert-key")
    assert _steps(doc) == steps + 1 and doc.history.top.label == "Insert Key"
    assert _run(doc, "delete-key")
    assert doc.animation().channels == {}
    assert clay_ops.by_key("object", "I").name == "insert-key"


def test_apply_pose_as_rest_is_enabled_only_when_something_is_posed() -> None:
    doc, hip, _ = _rig()
    op = clay_ops.get("apply-pose-as-rest")
    assert not op.enabled(doc) and "rest" in clay_ops.reason_for(op, doc)
    doc.key_pose({hip: {"rotation": Q90}})
    assert op.enabled(doc)
    assert _run(doc, "apply-pose-as-rest")


# -- Export Posed GLB ---------------------------------------------------------------------------


def test_export_posed_glb_writes_the_pose_at_the_playhead_with_no_animation(tmp_path, monkeypatch):
    from realmspinner.kernels.geom3d import gltf
    from realmspinner.studio import dialogs
    from realmspinner.studio.modes.clay import mode as clay_mode

    from .test_clay_mode import FakeCtx

    doc, hip, arm = _rig()
    _add(doc, "Body", mesh=weighted(bp.box()), skin=skin_for(hip))
    doc.set_rig_time(1.0)
    doc.key_pose({hip: {"rotation": Q0}})
    doc.set_rig_time(0.0)
    doc.key_pose({hip: {"rotation": Q0}})
    doc.set_rig_time(1.0)
    doc.key_pose({hip: {"rotation": Q90}})
    doc.set_animation_loop(0, False)
    ctx = FakeCtx()
    tab = clay_mode.adopt(ctx, doc, title="Rig")
    out = tmp_path / "posed.glb"
    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: out)

    clay_mode.export_mesh_file(ctx, tab, "posed")
    assert out.exists() or ctx.result is not None
    model = gltf.load(out.read_bytes())
    assert not model.animations, "a posed export carries the pose, not the clip"
    joint = next(n for n in model.nodes if n.name == "Hip")
    assert np.allclose(np.asarray(joint.rotation, dtype="f8"), Q90, atol=1e-5)


def test_the_palette_offers_the_posed_export_only_for_a_rig_with_an_animation() -> None:
    from .test_clay_palette import _clay
    from .test_clay_palette import _ctx as palette_ctx

    ctx = palette_ctx()
    doc = ctx.state.clay.active.doc
    assert "clay-export:posed" not in _clay(ctx)
    bone = doc.add_object(bd.Obj(uid=bd.new_uid(), name="B", mesh=bp.box(), bone=True)).uid
    assert "clay-export:posed" not in _clay(ctx), "a rig with no animation has nothing to pose"
    doc.set_pose_mode(True)
    doc.key_pose({bone: {"rotation": Q90}})
    assert "clay-export:posed" in _clay(ctx)
