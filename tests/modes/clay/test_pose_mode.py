"""Pose mode: the session state, the evaluated pose in the viewport, skinned meshes and bone glyphs.

Tranche 3 of the 2026-10-13 Clay rig plan. Pose is a layer over Shape mode (``ClayDoc.pose_mode``),
not an element mode: a bone is selected the way any object is. In it the viewport composes the
*evaluated* pose, every other mode shows rest, a shape under a bone follows it, a weighted mesh is
skinned, and a click picks a bone. Each claim fails on the tree before the change.
"""

from __future__ import annotations

import numpy as np
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import rig
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay.state import ClayState
from realmspinner.studio.modes.clay.ui import view as clay_view

from ._rigkit import skin_for, weighted

Q0 = [0.0, 0.0, 0.0, 1.0]
Q90 = [0.0, 0.0, float(np.sin(np.pi / 4)), float(np.cos(np.pi / 4))]
RECT = (0.0, 0.0, 320.0, 240.0)


def _add(doc: bd.ClayDoc, name: str, **kw) -> int:
    kw.setdefault("mesh", bp.box())
    return doc.add_object(bd.Obj(uid=bd.new_uid(), name=name, **kw)).uid


def _arm() -> tuple[bd.ClayDoc, int, int, int]:
    """Hip (bone, y=1) -> Arm (bone, +1 x) -> Hand (a plain shape); Hip turns 90 degrees by t=1."""
    doc = bd.ClayDoc()
    hip = _add(doc, "Hip", bone=True, translation=(0, 1, 0))
    arm = _add(doc, "Arm", bone=True, parent=hip, translation=(1, 0, 0))
    hand = _add(doc, "Hand", parent=arm, translation=(1, 0, 0))
    index = doc.add_animation("Turn")
    doc.set_animation_loop(index, False)  # a looping clip wraps its last key back to 0
    doc.set_key(hip, "rotation", 0.0, Q0)
    doc.set_key(hip, "rotation", 1.0, Q90)
    return doc, hip, arm, hand


# -- the session state ------------------------------------------------------------------------


def test_pose_mode_is_a_layer_over_shape_mode_and_any_element_mode_leaves_it() -> None:
    doc, *_ = _arm()
    rev = doc.rev
    doc.set_pose_mode(True)
    assert doc.pose_mode and doc.element_mode == "object" and doc.rev == rev + 1
    doc.set_element_mode("face")
    assert not doc.pose_mode and doc.element_mode == "face"
    doc.set_pose_mode(True)
    assert doc.element_mode == "object", "entering goes through Shape mode"
    doc.set_element_mode("object")
    assert not doc.pose_mode, "even choosing Shape again leaves Pose"


def test_leaving_pose_mode_stops_playback_and_drops_an_unkeyed_pose() -> None:
    doc, hip, *_ = _arm()
    doc.set_pose_mode(True)
    doc.playing = True
    doc.set_live_pose(hip, "rotation", Q90)
    doc.set_pose_mode(False)
    assert not doc.playing and doc.live_pose == {}


def test_moving_the_playhead_is_not_an_edit_and_drops_the_unkeyed_pose() -> None:
    doc, hip, *_ = _arm()
    steps, rev, pose_rev = len(doc.history), doc.rev, doc.pose_rev
    doc.set_live_pose(hip, "rotation", Q90)
    doc.set_rig_time(0.5)
    assert doc.rig_time == 0.5 and doc.live_pose == {}
    assert len(doc.history) == steps and doc.rev == rev and doc.pose_rev > pose_rev
    doc.set_rig_time(-3.0)
    assert doc.rig_time == 0.0, "time cannot go negative"


def test_the_pose_is_the_animation_at_the_playhead_under_the_unkeyed_pose() -> None:
    doc, hip, arm, _hand = _arm()
    doc.set_rig_time(0.5)
    over = doc.pose_overrides()
    assert set(over) == {hip} and set(over[hip]) == {"rotation"}
    assert np.linalg.norm(over[hip]["rotation"]) == pytest.approx(1.0)
    doc.set_live_pose(arm, "translation", [2, 0, 0])
    doc.set_live_pose(hip, "rotation", Q0)
    over = doc.pose_overrides()
    assert np.allclose(over[hip]["rotation"], Q0), "the live pose wins over the animation"
    assert np.allclose(over[arm]["translation"], [2, 0, 0])


def test_posed_world_composes_the_driven_properties_up_the_chain() -> None:
    doc, hip, arm, hand = _arm()
    doc.set_rig_time(1.0)
    over = doc.pose_overrides()
    world = doc.posed_world(hand, over)
    # Hip (y=1) turned 90 degrees about Z: the hand's 2 m along x becomes 2 m along y.
    assert np.allclose(world[:3, 3], [0, 3, 0], atol=1e-9)
    assert np.allclose(doc.world_matrix(hand)[:3, 3], [2, 1, 0]), "the rest pose is untouched"


def test_the_skin_palette_is_posed_over_rest_and_is_the_identity_at_rest() -> None:
    doc, hip, *_ = _arm()
    body = _add(doc, "Body", mesh=weighted(bp.box()), skin=skin_for(hip))
    obj = doc.by_uid(body)
    assert np.allclose(doc.skin_palette(obj, {}), np.eye(4)[None])
    palette = doc.skin_palette(obj, {hip: {"rotation": np.array(Q90)}})[0]
    # A point at the hip's own pivot does not move; one a metre out swings round it.
    assert np.allclose(palette @ [0, 1, 0, 1], [0, 1, 0, 1], atol=1e-9)
    assert np.allclose(palette @ [1, 1, 0, 1], [0, 2, 0, 1], atol=1e-9)


# -- skin_positions -----------------------------------------------------------------------------


def test_skinning_with_the_identity_palette_returns_the_positions_unchanged() -> None:
    mesh = weighted(bp.box())
    out = rig.skin_positions(mesh.positions, mesh.joints, mesh.weights, np.eye(4)[None])
    assert np.allclose(out, mesh.positions) and out.dtype == np.float32


def test_a_blend_of_two_joints_lands_at_the_weighted_average_of_their_destinations() -> None:
    positions = np.array([[1.0, 0.0, 0.0]])
    joints = np.array([[0, 1, 0, 0]])
    weights = np.array([[0.25, 0.75, 0, 0]])
    shift = np.eye(4)
    shift[:3, 3] = [0, 4, 0]
    palette = np.stack([np.eye(4), shift])
    out = rig.skin_positions(positions, joints, weights, palette)
    assert np.allclose(out, [[1.0, 3.0, 0.0]])


def test_the_objects_own_world_matrix_is_applied_before_and_removed_after() -> None:
    positions = np.array([[0.0, 0.0, 0.0]])
    world = np.eye(4)
    world[:3, 3] = [5, 0, 0]
    shift = np.eye(4)
    shift[:3, 3] = [0, 1, 0]
    out = rig.skin_positions(
        positions, np.array([[0, 0, 0, 0]]), np.array([[1.0, 0, 0, 0]]), shift[None], world
    )
    assert np.allclose(out, [[0.0, 1.0, 0.0]]), "a world shift is a local shift when translated"


def test_skinning_a_large_mesh_is_chunked_but_exact() -> None:
    n = rig._SKIN_CHUNK * 2 + 17
    positions = np.random.default_rng(1).random((n, 3))
    joints = np.zeros((n, 4), dtype="i4")
    weights = np.zeros((n, 4))
    weights[:, 0] = 1.0
    shift = np.eye(4)
    shift[:3, 3] = [1, 2, 3]
    out = rig.skin_positions(positions, joints, weights, shift[None])
    assert np.allclose(out, positions + [1, 2, 3], atol=1e-5)


# -- the viewport ---------------------------------------------------------------------------------


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
    yield v
    v.release()


def _drawn(view, doc: bd.ClayDoc) -> None:
    view.frame_selection(doc)
    view.draw(doc, RECT, 0.0)


def test_pose_mode_draws_the_evaluated_pose_and_shape_mode_draws_rest(view) -> None:
    doc, hip, arm, hand = _arm()
    doc.set_rig_time(1.0)
    _drawn(view, doc)
    hand_obj = doc.by_uid(hand)
    rest = view._world(doc, hand_obj)[:3, 3]
    assert np.allclose(rest, [2, 1, 0]), "Shape mode shows the rest pose"

    doc.set_pose_mode(True)
    _drawn(view, doc)
    posed = view._world(doc, hand_obj)[:3, 3]
    assert np.allclose(posed, [0, 3, 0], atol=1e-9), "a shape under a bone follows the posed parent"

    doc.set_pose_mode(False)
    _drawn(view, doc)
    assert np.allclose(view._world(doc, hand_obj)[:3, 3], [2, 1, 0])


def test_a_held_pose_hands_out_the_same_arrays_so_the_world_memo_still_hits(view) -> None:
    doc, hip, arm, hand = _arm()
    doc.set_rig_time(1.0)
    doc.set_pose_mode(True)
    _drawn(view, doc)
    first = view._posed_arrays(doc)[hip][1]
    assert view._posed_arrays(doc)[hip][1] is first
    doc.set_rig_time(0.5)
    assert view._posed_arrays(doc)[hip][1] is not first, "a new time is a new pose"


def test_the_redraw_key_moves_with_the_playhead_without_the_document_changing(view) -> None:
    doc, hip, *_ = _arm()
    doc.set_pose_mode(True)
    view.draw(doc, RECT, 0.0)
    key = view._last_render_key
    rev = doc.rev
    doc.set_rig_time(0.25)
    view.draw(doc, RECT, 0.0)
    assert view._last_render_key != key and doc.rev == rev


def test_a_weighted_mesh_is_skinned_on_entering_pose_mode_and_restored_on_leaving(view) -> None:
    doc, hip, arm, hand = _arm()
    body = _add(doc, "Body", mesh=weighted(bp.box()), skin=skin_for(hip), translation=(0, 0, 0))
    doc.set_rig_time(1.0)
    _drawn(view, doc)
    calls: list[tuple[int, np.ndarray]] = []
    real = view._preview_positions

    def spy(d, uid, positions):
        calls.append((uid, positions))
        return real(d, uid, positions)

    view._preview_positions = spy  # type: ignore[method-assign]

    doc.set_pose_mode(True)
    view.draw(doc, RECT, 0.0)
    [(uid, positions)] = [c for c in calls if c[0] == body]
    expected = rig.skin_positions(
        doc.by_uid(body).mesh.positions,
        doc.by_uid(body).mesh.joints,
        doc.by_uid(body).mesh.weights,
        doc.skin_palette(doc.by_uid(body), doc.pose_overrides()),
    )
    assert np.allclose(positions, expected)
    assert not np.allclose(positions, doc.by_uid(body).mesh.positions)

    view.draw(doc, RECT, 0.0)
    assert len([c for c in calls if c[0] == body]) == 1, "a held pose is not re-skinned"
    doc.set_rig_time(0.5)
    view.draw(doc, RECT, 0.0)
    assert len([c for c in calls if c[0] == body]) == 2

    cached = view._cache[body]
    doc.set_pose_mode(False)
    view.draw(doc, RECT, 0.0)
    assert view._cache[body] is not cached, "the skinned buffers were evicted and rebuilt at rest"
    assert body not in view._skin_stamps


def test_bone_glyphs_are_drawn_for_a_rig_and_not_for_a_document_without_one(view) -> None:
    doc, hip, arm, hand = _arm()
    _drawn(view, doc)
    items = view._bone_draws(doc)
    assert items, "a cross at each pivot, a line to the parent, a tick on the leaf"
    plain = bd.ClayDoc()
    _add(plain, "Box")
    assert view._bone_draws(plain) == []
    assert view._bone_lines is None, "the buffers are released when the last bone goes"


def test_a_selected_bone_is_drawn_in_the_accent_over_the_idle_ones(view) -> None:
    doc, hip, arm, hand = _arm()
    _drawn(view, doc)
    idle = view._bone_draws(doc)
    doc.select([arm])
    both = view._bone_draws(doc)
    assert len(both) == len(idle) + 1, "one more draw: the selected bones' own lines"
    colours = {tuple(round(c, 3) for c in item.color[:3]) for item in both}
    assert len(colours) == 2


def test_hidden_bones_draw_no_glyph(view) -> None:
    doc, hip, arm, hand = _arm()
    _drawn(view, doc)
    n = len(view._bone_draws(doc))
    doc.set_visibility({arm: False})
    assert len(view._bone_draws(doc)) <= n


def _screen_of(view, doc: bd.ClayDoc, uid: int) -> tuple[float, float]:
    from realmspinner.kernels.mesh import pick as bp_

    point = view._world(doc, doc.by_uid(uid))[:3, 3][None]
    shot = bp_.project(
        point,
        np.eye(4),
        view.camera.projection() @ view.camera.view(),
        view.camera.position,
        int(RECT[2]),
        int(RECT[3]),
    )
    return float(shot.xy[0, 0]), float(shot.xy[0, 1])


def test_a_click_near_a_bone_glyph_picks_that_bone(view) -> None:
    doc, hip, arm, hand = _arm()
    _drawn(view, doc)
    x, y = _screen_of(view, doc, arm)
    assert view.pick_bone(doc, (x + 3.0, y - 2.0)) == arm
    x, y = _screen_of(view, doc, hip)
    assert view.pick_bone(doc, (x, y)) == hip


def test_a_click_on_a_shape_picks_its_bone_and_empty_space_picks_nothing(view) -> None:
    doc, hip, arm, hand = _arm()
    free = _add(doc, "Free", translation=(-6, 0, 0))
    _drawn(view, doc)
    # Hand has no glyph of its own: aim at its centre, which lies away from both pivots.
    x, y = _screen_of(view, doc, hand)
    got = view.pick_bone(doc, (x, y))
    assert got == arm, "the hand moves with the arm"
    x, y = _screen_of(view, doc, free)
    assert view.pick_bone(doc, (x, y)) is None, "a shape no bone moves picks nothing"
    assert view.pick_bone(doc, (2.0, 2.0)) is None


# -- the header and keys ------------------------------------------------------------------------


def test_the_header_offers_pose_and_5_enters_it() -> None:
    from realmspinner.studio.modes.clay.ui.panes import tools as clay_tools

    assert ("pose", "Pose", "5") in clay_tools.MODE_BUTTONS
    assert clay_mode.MODE_KEYS["5"] == "pose" and "5" not in clay_mode.ELEMENT_KEYS


def test_a_bare_5_enters_pose_mode(svc, gl) -> None:
    import pygame

    from .test_clay_mode import FakeCtx, _tab

    ctx = FakeCtx(svc)
    tab = _tab(ctx)
    event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_5, mod=0)
    assert clay_mode.handle_key(ctx, event) is True
    assert tab.doc.pose_mode
