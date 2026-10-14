"""Regression tests for the 2026-10-14 audit, fixer F26.

Clay's Pose-mode viewport, outliner, inspector, timeline and UV panes.
"""

from __future__ import annotations

import numpy as np
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio.modes.clay.state import ClayState
from realmspinner.studio.modes.clay.ui import view as clay_view

from ._rigkit import skin_for, weighted

RECT = (0.0, 0.0, 320.0, 240.0)


def _add(doc: bd.ClayDoc, name: str, **kw) -> int:
    kw.setdefault("mesh", bp.box())
    return doc.add_object(bd.Obj(uid=bd.new_uid(), name=name, **kw)).uid


class _Ctx:
    def __init__(self) -> None:
        self.state = type("S", (), {"clay": ClayState()})()
        self.clay_view = None
        self.toasts: list[tuple[str, str]] = []

    def toast(self, message: str, level: str = "info", **_kw) -> None:
        self.toasts.append((message, level))


@pytest.fixture
def view(gl):
    ctx = _Ctx()
    v = clay_view.ClayView(gl, ctx)
    ctx.clay_view = v
    v.app_ctx = ctx
    yield v
    v.release()


# -- clay-view-05 (clay-50) -----------------------------------------------------------------------


def test_a_move_of_a_bone_with_a_negative_w_rotation_key_keys_no_rotation(view) -> None:
    doc = bd.ClayDoc()
    hip = _add(doc, "Hip", bone=True)
    q = np.array([0.3, 0.1, 0.5, -0.8])
    q /= np.linalg.norm(q)  # w < 0: the same turn as -q, which decompose hands back
    doc.add_animation("Clip")
    doc.set_key(hip, "rotation", 0.0, q)
    doc.set_key(hip, "rotation", 1.0, q)
    doc.set_rig_time(0.5)
    doc.set_pose_mode(True)
    doc.select([hip])
    view.frame_whole(doc)
    view.draw(doc, RECT, 0.0)
    view._last_mouse = (160.0, 120.0)
    assert view.begin_keyboard_drag(doc, "move")
    view._drag_keyboard(doc, (200.0, 100.0))
    view._release_drag(doc, 1)
    channels = doc.animation().channels[hip]
    assert "translation" in channels
    assert list(channels["rotation"].times) == [0.0, 1.0], "a Move keys position only"


# -- clay-view-02 (clay-11) -----------------------------------------------------------------------


def _slid_body(view) -> tuple[bd.ClayDoc, int, np.ndarray, np.ndarray]:
    """A box weighted to a hip keyed 3 m along X: ``(doc, body, rest_centre, drawn_centre)``."""
    doc = bd.ClayDoc()
    hip = doc.add_bone().uid
    body = _add(doc, "Body", mesh=weighted(bp.box()), skin=skin_for(hip))
    index = doc.add_animation("Slide")
    doc.set_animation_loop(index, False)
    doc.set_key(hip, "translation", 0.0, [0, 0, 0])
    doc.set_key(hip, "translation", 1.0, [3, 0, 0])
    doc.set_rig_time(1.0)
    view.frame_whole(doc)
    view.draw(doc, RECT, 0.0)
    doc.set_pose_mode(True)
    view.draw(doc, RECT, 0.0)
    return doc, body, np.zeros(3), np.array([3.0, 0.0, 0.0])


def _screen(view, point) -> tuple[float, float]:
    from realmspinner.kernels.mesh import pick as bp_

    shot = bp_.project(
        np.array([point], dtype="f8"),
        np.eye(4),
        view.camera.projection() @ view.camera.view(),
        view.camera.position,
        int(RECT[2]),
        int(RECT[3]),
    )
    return float(shot.xy[0, 0]), float(shot.xy[0, 1])


def test_picking_a_posed_weighted_mesh_hits_where_it_is_drawn(view) -> None:
    doc, body, rest, drawn = _slid_body(view)
    assert view.pick(doc, _screen(view, drawn)) == body, "the click lands on the box that is drawn"
    assert view.pick(doc, _screen(view, rest)) is None, "the empty rest position is empty"


def test_the_framing_box_and_the_hover_outline_follow_a_posed_weighted_mesh(view) -> None:
    doc, body, rest, drawn = _slid_body(view)
    lo, hi = view.world_bounds(doc)
    assert lo[0] > 2.0 and hi[0] < 4.0, "F frames the box where it is drawn"
    view.hover_object = body
    items = view._object_overlays(doc)
    assert items, "the hovered weighted mesh is outlined"
    overlay = view._obj_overlays[body]
    xs = np.frombuffer(overlay.pos_vbo.read(), dtype="f4").reshape(-1, 3)[:, 0]
    assert 2.0 < float(xs.mean()) < 4.0, "the yellow outline is laid on the skinned positions"
    doc.set_pose_mode(False)
    view._object_overlays(doc)
    xs = np.frombuffer(overlay.pos_vbo.read(), dtype="f4").reshape(-1, 3)[:, 0]
    assert abs(float(xs.mean())) < 0.1, "leaving Pose mode puts the outline back on the rest mesh"


# -- clay-panes-02 (clay-36) ----------------------------------------------------------------------


def test_more_actions_status_follows_a_held_pose_made_with_auto_key_off() -> None:
    from realmspinner.studio.modes.clay import ops as clay_ops
    from realmspinner.studio.modes.clay.ui import inspector_actions

    doc = bd.ClayDoc()
    hip = _add(doc, "Hip", bone=True)
    doc.set_pose_mode(True)
    doc.auto_key = False
    op = clay_ops.get("apply-pose-as-rest")
    assert "Nothing is posed" in inspector_actions.status(doc, (op,), False)[op.name]
    rev = doc.rev
    doc.set_live_pose(hip, "rotation", [0.0, 0.0, 0.7071067811865476, 0.7071067811865476])
    assert doc.rev == rev, "a held pose is not an edit: the memo key's rev did not move"
    assert inspector_actions.status(doc, (op,), False)[op.name] == "", "the held pose lights it"
    doc.set_rig_time(0.5)  # the playhead moving clears the held pose
    assert "Nothing is posed" in inspector_actions.status(doc, (op,), False)[op.name]


# -- the Inspector, driven through a real (GL-less) imgui context ---------------------------------


@pytest.fixture
def imgui_ui(monkeypatch):
    from _ui_context import imgui_context

    from realmspinner.studio import widgets

    widgets._OPEN_REQUESTS.clear()
    with imgui_context(monkeypatch) as imgui:
        yield imgui
    widgets._OPEN_REQUESTS.clear()


def _inspector(imgui_ui, doc_setup):
    """``(rig, doc, uid)``: the Inspector over a document *doc_setup* builds, its Position & turn
    fold open. The tab's own document is the one edited, as the pane sees it."""
    from realmspinner.studio import widgets

    from .test_inspector import _document, _rig

    ctx, tab, doc = _document(0)
    uid = doc_setup(doc)
    doc.select([uid])
    widgets.request_open("clay/insp/position")
    widgets.request_open("clay/insp/advanced")
    return _rig(imgui_ui, ctx), doc, uid


def test_pose_mode_inspector_transform_fields_do_not_silently_edit_the_rest_pose(imgui_ui) -> None:
    from realmspinner.studio.modes.clay.ui.panes import props

    def setup(doc):
        hip = _add(doc, "Hip", bone=True, translation=(0, 1, 0))
        doc.add_animation("Clip")
        doc.set_key(hip, "translation", 0.0, [0, 1, 0])
        doc.set_key(hip, "translation", 1.0, [0, 5, 0])
        doc.set_rig_time(1.0)
        doc.set_pose_mode(True)
        return hip

    rig, doc, hip = _inspector(imgui_ui, setup)
    for label in ("position##bt", "rotation (deg)##br", "scale##bs"):
        field = rig.find(label=label)
        assert not field.enabled and field.reason == props.POSE_FIELDS_WHY, label
    # The door under the fields refuses too, and writes nothing, for a caller that reaches it
    # another way.
    steps = len(doc.history)
    assert not props._apply_transform(doc, doc.by_uid(hip), object(), translation=[0, 9, 0])
    assert len(doc.history) == steps and tuple(doc.by_uid(hip).translation) == (0, 1, 0)
    # Out of Pose mode the same fields are the rest pose, and live.
    doc.set_pose_mode(False)
    rig.settle()
    assert rig.find(label="position##bt").enabled


def test_weighted_mesh_transform_fields_are_greyed_with_a_reason_not_refused_per_keystroke(
    imgui_ui,
) -> None:
    def setup(doc):
        hip = _add(doc, "Hip", bone=True)
        return _add(doc, "Body", mesh=weighted(bp.box()), skin=skin_for(hip))

    rig, doc, body = _inspector(imgui_ui, setup)
    sentence = doc.transform_refusal(doc.by_uid(body))
    assert "weighted to a skeleton" in sentence
    for label in ("position##bt", "rotation (deg)##br", "scale##bs"):
        field = rig.find(label=label)
        assert not field.enabled and field.reason == sentence, label
    # A plain shape keeps its fields.
    doc2_rig, doc2, plain = _inspector(imgui_ui, lambda d: _add(d, "Plain"))
    assert doc2_rig.find(label="position##bt").enabled


# -- clay-panes-06 (clay-80) ----------------------------------------------------------------------


def test_context_menu_heading_says_pose_in_pose_mode() -> None:
    from realmspinner.studio.modes.clay.ui.panes import menu

    doc = bd.ClayDoc()
    _add(doc, "Hip", bone=True)
    assert menu.heading(doc) == "Shape mode"
    doc.set_pose_mode(True)
    assert menu.heading(doc) == "Pose mode"
    doc.set_element_mode("face")
    assert menu.heading(doc) == "Faces mode"


# -- clay-panes-01 (clay-35) ----------------------------------------------------------------------


def test_outliner_row_delete_removes_a_bone_together_with_the_mesh_weighted_to_it() -> None:
    from realmspinner.studio.modes.clay.ui.panes import outliner as ol

    doc = bd.ClayDoc()
    hip = bd.new_uid()
    body = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="Body", mesh=weighted(bp.box()), skin=skin_for(hip))
    ).uid
    # The bone sits later in the list than the mesh that skins to it.
    doc.add_object(bd.Obj(uid=hip, name="Hip", mesh=bp.box(), bone=True))
    keep = _add(doc, "Keep")
    doc.select([body, hip])
    ctx = _Ctx()
    steps = len(doc.history)
    refusals = ol._remove_objects(ctx, doc, ol._menu_targets(doc, doc.by_uid(body)))
    assert refusals == [] and ctx.toasts == []
    assert [o.uid for o in doc.objects] == [keep], "the bone went with the mesh that skins to it"
    assert len(doc.history) == steps + 1, "one undo step"
    assert doc.undo() and len(doc.objects) == 3


# -- clay-panes-03 (clay-37) ----------------------------------------------------------------------


def test_hidden_row_menu_rig_rows_do_not_act_on_another_selection() -> None:
    from realmspinner.studio.modes.clay.ui.panes import outliner as ol

    doc = bd.ClayDoc()
    shown = _add(doc, "A")
    hidden = _add(doc, "H")
    doc.set_props(hidden, visible=False)
    doc.select([shown])
    why = ol._rig_rows_reason(doc, doc.by_uid(hidden))
    assert why and "hidden" in why, "greyed, with the reason, because the op would act on A"
    assert ol._rig_rows_reason(doc, doc.by_uid(shown)) == ""
    # A hidden row that is in the selection (a Shift-range over it) is the selection's own.
    doc.selection = {shown, hidden}
    assert ol._rig_rows_reason(doc, doc.by_uid(hidden)) == ""


# -- clay-rig-11 (clay-86) ------------------------------------------------------------------------


def test_a_clamped_slide_reselects_the_keys_where_they_landed() -> None:
    from realmspinner.studio.modes.clay.ui.panes import timeline as pane

    doc = bd.ClayDoc()
    hip = _add(doc, "Hip", bone=True)
    doc.set_pose_mode(True)
    for time in (0.5, 2.0):
        doc.set_rig_time(time)
        doc.key_pose({hip: {"rotation": [0.0, 0.0, np.sin(time), np.cos(time)]}})
    selection = [(hip, 0.5), (hip, 2.0)]
    assert doc.slide_delta(selection, -1.0) == -0.5, "the earliest key stops at 0, the rest follow"
    assert doc.move_bone_keys(selection, -1.0)
    times = [float(t) for t in doc.animation().channels[hip]["rotation"].times]
    assert times == [0.0, 1.5]
    assert pane._landed_selection(doc, selection, -1.0) == [(hip, 0.0), (hip, 1.5)]
    assert all(key[1] in times for key in pane._landed_selection(doc, selection, -1.0))


# -- clay-rig-13 and clay-rig-14 (clay-88, clay-89) ------------------------------------------------


def test_the_fps_tooltip_does_not_claim_the_file_carries_the_rate() -> None:
    from realmspinner.studio.modes.clay.ui.panes import timeline as pane

    assert "written at" not in pane.FPS_TOOLTIP
    assert "seconds" in pane.FPS_TOOLTIP, "it says what the file does hold"


def test_play_reason_says_to_key_a_second_time_when_every_key_is_at_zero() -> None:
    from realmspinner.studio.modes.clay import timeline as tl

    doc = bd.ClayDoc()
    hip = _add(doc, "Hip", bone=True)
    doc.set_pose_mode(True)
    assert "Key a bone first" in tl.play_reason(doc), "no animation at all"
    doc.key_pose({hip: {"rotation": [0.0, 0.0, 0.0, 1.0]}})  # a key at 0 s only
    assert tl.length(doc) == 0.0
    why = tl.play_reason(doc)
    assert "Key a bone first" not in why and "0 s" in why, why
    doc.set_rig_time(1.0)
    doc.key_pose({hip: {"rotation": [0.0, 0.0, 0.7071067811865476, 0.7071067811865476]}})
    assert tl.play_reason(doc) == "", "a key after 0 s gives Play something to play"


# -- clay-texturing-03 (clay-91) ------------------------------------------------------------------


def test_concave_triangles_are_not_recomputed_for_a_uv_only_edit(monkeypatch) -> None:
    from dataclasses import replace

    from realmspinner.kernels.mesh import earclip
    from realmspinner.kernels.mesh import uv as bm_uv
    from realmspinner.studio.modes.clay.ui.panes import uv as clay_uv

    mesh = bm_uv.box_unwrap(bp.grid((1, 1), 6))
    calls = []
    real = earclip.concave_faces
    monkeypatch.setattr(
        earclip, "concave_faces", lambda *a, **k: calls.append(1) or real(*a, **k)
    )
    cache: dict = {}
    clay_uv._concave_triangles(cache, mesh)
    assert len(calls) == 1
    moved = replace(mesh, uv=mesh.uv + np.float32(0.01))  # what one frame of a UV drag does
    assert moved.positions is not mesh.positions, "Mesh copies its arrays: not identity"
    clay_uv._concave_triangles(cache, moved)
    assert len(calls) == 1, "a uv-only edit is not clipped again"
    clay_uv._concave_triangles(cache, moved)
    assert len(calls) == 1
    # A real change to the triangulation's inputs still is.
    grown = replace(mesh, positions=mesh.positions * np.float32(2.0))
    clay_uv._concave_triangles(cache, grown)
    assert len(calls) == 2
