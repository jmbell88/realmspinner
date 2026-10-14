"""The 2026-10-14 audit, fixer F25: Clay's op tail, texture link, timeline and bone gizmo.

Each test is named for the claim of one finding and was run against the unfixed tree first.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from realmspinner.kernels.mesh import anim, ops_slide
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.pixel.document import Document
from realmspinner.studio.modes.clay import ops as clay_ops
from realmspinner.studio.modes.clay import texture_link as tl_link
from realmspinner.studio.modes.clay import timeline as tl
from realmspinner.studio.modes.clay.state import ClayState
from realmspinner.studio.modes.clay.ui import view as clay_view
from realmspinner.studio.modes.clay.ui.panes import timeline as pane

from ._rigkit import rigged_doc, skin_for, weighted

RECT = (0.0, 0.0, 320.0, 240.0)


class _Ctx:
    def __init__(self) -> None:
        self.toasts: list[tuple[str, str]] = []

    def toast(self, message: str, level: str = "info") -> None:
        self.toasts.append((message, level))


def _add(doc: bd.ClayDoc, name: str, **kw) -> int:
    kw.setdefault("mesh", bp.box())
    return doc.add_object(bd.Obj(uid=bd.new_uid(), name=name, **kw)).uid


# -- clay-32: a plain and a weighted object together ----------------------------------------------


def _plain_and_weighted() -> tuple[bd.ClayDoc, int, int]:
    doc, _bone, body = rigged_doc()
    plain = _add(doc, "Armour")
    doc.select([plain, body])
    doc.set_element_mode("face")
    doc.set_element_sel(plain, el.ElementSel(faces=[0]))
    doc.set_element_sel(body, el.ElementSel(faces=[0]))
    return doc, plain, body


def test_extrude_over_a_plain_and_a_weighted_object_changes_neither_or_skips_only_the_weighted_one(
):
    doc, plain, body = _plain_and_weighted()
    plain_before, body_before = doc.by_uid(plain).mesh, doc.by_uid(body).mesh
    steps, ctx = len(doc.history), _Ctx()

    ran = clay_ops.run(ctx, doc, clay_ops.get("extrude"))

    assert doc.by_uid(body).mesh is body_before, "the weighted object is never rewritten"
    assert any("Unbind Skin" in message for message, _ in ctx.toasts), ctx.toasts
    changed = doc.by_uid(plain).mesh is not plain_before
    # The invariant: ``run`` answers True iff a step was pushed.
    assert ran is (len(doc.history) > steps)
    assert ran is changed, "the plain object's edit and the answer must agree"
    assert ran, "the weighted object's refusal does not abandon the plain one"


def test_smooth_over_a_plain_and_a_weighted_object_skips_only_the_weighted_one():
    doc, plain, body = _plain_and_weighted()
    doc.set_element_mode("object")
    doc.select([plain, body])
    plain_before, body_before = doc.by_uid(plain).mesh, doc.by_uid(body).mesh
    steps, ctx = len(doc.history), _Ctx()

    ran = clay_ops.run(ctx, doc, clay_ops.get("smooth"))

    assert doc.by_uid(body).mesh is body_before
    assert any("Unbind Skin" in message for message, _ in ctx.toasts), ctx.toasts
    assert ran is (len(doc.history) > steps)
    assert ran and doc.by_uid(plain).mesh is not plain_before


# -- clay-33: mirror a weighted mesh --------------------------------------------------------------


def _two_sided_body() -> tuple[bd.ClayDoc, int]:
    """A body weighted to two bones: +X half to the first, -X half to the second."""
    doc = bd.ClayDoc()
    left = _add(doc, "BoneL", bone=True)
    right = _add(doc, "BoneR", bone=True)
    mesh = weighted(bp.box(), 2)
    x = np.asarray(mesh.positions)[:, 0]
    joints = np.asarray(mesh.joints).copy()
    weights = np.asarray(mesh.weights).copy()
    joints[:, 0] = np.where(x > 0, 0, 1)
    joints[:, 1] = 0
    weights[:] = 0.0
    weights[:, 0] = 1.0
    mesh = replace(mesh, joints=joints, weights=weights)
    body = _add(doc, "Body", mesh=mesh, skin=skin_for(left, right))
    doc.select([body])
    return doc, body


def test_mirror_x_on_a_weighted_mesh_is_refused_or_keeps_each_point_on_its_own_side_of_the_rig():
    doc, body = _two_sided_body()
    before = doc.by_uid(body).mesh
    steps, ctx = len(doc.history), _Ctx()

    ran = clay_ops.run(ctx, doc, clay_ops.get("mirror-x"))

    after = doc.by_uid(body).mesh
    if ran:
        # Not refused: then a point driven by the +X bone must still sit on the +X side.
        x, j = np.asarray(after.positions)[:, 0], np.asarray(after.joints)[:, 0]
        assert (x[j == 0] > 0).all() and (x[j == 1] < 0).all()
    else:
        assert after is before and len(doc.history) == steps
        assert any("Unbind Skin" in message for message, _ in ctx.toasts), ctx.toasts


# -- clay-34: an Add Bone chain's joint is not a group --------------------------------------------


def test_ungroup_is_not_offered_for_an_empty_bone_with_children():
    doc = bd.ClayDoc()
    root = doc.add_bone()
    child = doc.add_bone(parent=root.uid)
    doc.select([root.uid])
    op = clay_ops.get("ungroup")

    assert not op.enabled(doc), "Ungroup must not offer to delete a joint"
    steps = len(doc.history)
    assert clay_ops.run(_Ctx(), doc, op) is False
    assert len(doc.history) == steps
    assert {o.uid for o in doc.objects} >= {root.uid, child.uid}


def test_a_real_group_is_still_ungrouped():
    doc = bd.ClayDoc()
    a, b = _add(doc, "A"), _add(doc, "B")
    doc.select([a, b])
    clay_ops.run(_Ctx(), doc, clay_ops.get("group"))
    group = doc.selection and next(iter(doc.selection))
    assert clay_ops.get("ungroup").enabled(doc)
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("ungroup"))
    assert doc.by_uid(a).parent is None and group not in {o.uid for o in doc.objects}


def test_the_no_geometry_refusal_names_a_bone_as_a_bone():
    doc = bd.ClayDoc()
    bone = doc.add_bone()
    doc.select([bone.uid])
    ctx = _Ctx()
    assert clay_ops.run(ctx, doc, clay_ops.get("mirror-x")) is False
    assert ctx.toasts and "bone" in ctx.toasts[-1][0].lower()
    assert "group" not in ctx.toasts[-1][0].lower()
    assert "no geometry" in ctx.toasts[-1][0]


# -- clay-77: Simplify on a weighted mesh ---------------------------------------------------------


def test_simplify_on_a_weighted_mesh_toasts_only_the_skin_refusal():
    doc, _bone, body = rigged_doc()
    sphere = weighted(bp.icosphere())
    # Give the body a mesh Simplify can reduce, still weighted to the one bone.
    doc.by_uid(body).mesh = sphere
    doc.select([body])
    ctx = _Ctx()

    ran = clay_ops.run(ctx, doc, clay_ops.get("simplify"), target=10)

    messages = [message for message, _ in ctx.toasts]
    assert ran is False
    assert any("Unbind Skin" in m for m in messages), messages
    assert not any(
        "Nothing more can be merged" in m or "nothing to simplify" in m for m in messages
    )


# -- clay-78: Edge Slide at zero ------------------------------------------------------------------


def _slide_doc() -> tuple[bd.ClayDoc, int]:
    positions = [[x, 0, z] for x in range(4) for z in (0, 1)]
    mesh = bm.from_faces(positions, [[0, 2, 3, 1], [2, 4, 5, 3], [4, 6, 7, 5]])
    doc = bd.ClayDoc()
    uid = _add(doc, "Strip", mesh=mesh)
    doc.select([uid])
    doc.set_element_mode("edge")
    doc.set_element_sel(uid, el.ElementSel(edges=np.array([[2, 3]], dtype="i4")))
    return doc, uid


def test_edge_slide_at_zero_pushes_no_step():
    doc, uid = _slide_doc()
    steps, mesh = len(doc.history), doc.by_uid(uid).mesh

    ran = clay_ops.run(_Ctx(), doc, clay_ops.get("edge-slide"), t=0.0)

    assert ran is False
    assert len(doc.history) == steps and doc.by_uid(uid).mesh is mesh


def test_edge_slide_kernel_at_zero_hands_the_mesh_back():
    doc, uid = _slide_doc()
    mesh = doc.by_uid(uid).mesh
    out, _sel = ops_slide.edge_slide(mesh, doc.element_sel_of(uid), t=0.0)
    assert out is mesh


def test_edge_slide_still_slides_off_zero():
    doc, uid = _slide_doc()
    before = np.array(doc.by_uid(uid).mesh.positions, copy=True)
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("edge-slide"), t=0.5) is True
    assert not np.array_equal(before, doc.by_uid(uid).mesh.positions)


# -- clay-92 / clay-93: the texture link ----------------------------------------------------------


def _link_ctx() -> SimpleNamespace:
    ctx = SimpleNamespace(toasts=[])
    ctx.state = SimpleNamespace(clay=ClayState(), inker=SimpleNamespace(docs=[], activate=None))
    ctx.toast = lambda text, level="info", **_kw: ctx.toasts.append((text, level))
    return ctx


def _clay_tab(ctx):
    from realmspinner.studio.modes.clay.state import ClayTab

    doc = bd.ClayDoc()
    assert doc.add_texture(0, 32)
    tab = ClayTab(doc=doc, title="Crate")
    ctx.state.clay.add(tab)
    return tab


def _inker_tab(side: int):
    px = np.zeros((side, side, 4), np.uint8)
    px[..., 3] = 255
    doc = Document.from_pixels(px, name="T")
    return SimpleNamespace(uid="id1", title="Crate (texture)", doc=doc)


def test_an_oversized_inker_document_is_refused_without_being_flattened(monkeypatch):
    ctx = _link_ctx()
    inker = _inker_tab(tl_link.MAX_TEXTURE_SIDE + 1)
    calls: list[int] = []
    real = Document.flatten
    monkeypatch.setattr(Document, "flatten", lambda self, **kw: calls.append(1) or real(self, **kw))

    assert tl_link._flatten(ctx, inker.doc) is None

    assert calls == [], "the refusal comes before the allocation it exists to avoid"
    assert ctx.toasts and ctx.toasts[-1][1] == "error"


def test_take_back_refuses_while_the_tab_is_saving():
    ctx = _link_ctx()
    tab = _clay_tab(ctx)
    inker = _inker_tab(4)
    material, steps = tab.doc.materials[0], len(tab.doc.history)
    tab.saving = True

    tl_link.take_back(ctx, tab, 0, inker)

    assert tab.doc.materials[0] is material and len(tab.doc.history) == steps
    assert tab.inker_links == []
    assert ctx.toasts, "a refusal with no link to retry from must say so"


def test_take_back_refuses_while_a_gesture_is_live():
    ctx = _link_ctx()
    ctx.clay_view = SimpleNamespace(dragging=True)
    tab = _clay_tab(ctx)
    inker = _inker_tab(4)
    material = tab.doc.materials[0]

    tl_link.take_back(ctx, tab, 0, inker)

    assert tab.doc.materials[0] is material and tab.inker_links == []
    assert ctx.toasts


def test_take_back_still_lands_when_nothing_is_in_flight():
    ctx = _link_ctx()
    tab = _clay_tab(ctx)
    inker = _inker_tab(4)
    tl_link.take_back(ctx, tab, 0, inker)
    assert len(tab.inker_links) == 1


# -- clay-41: the dope sheet's rows ---------------------------------------------------------------


def _clip_doc(bones: int = 3, keys: int = 40) -> bd.ClayDoc:
    doc = bd.ClayDoc()
    uids = [_add(doc, f"B{i}", bone=True) for i in range(bones)]
    times = np.arange(keys) / 30.0
    channels = {
        u: {
            "translation": anim.Channel(times, np.zeros((keys, 3))),
            "rotation": anim.Channel(times, np.tile([0, 0, 0, 1.0], (keys, 1))),
        }
        for u in uids
    }
    clip = anim.Animation("Clip", True, "linear", channels)
    doc.set_rig_state(doc.rig_state.with_animations([clip]))
    doc.set_active_animation(0)
    doc.set_pose_mode(True)
    return doc


def test_the_dope_sheet_rows_are_not_rebuilt_for_an_unchanged_animation(monkeypatch):
    doc = _clip_doc()
    calls: list[int] = []
    real = anim.Animation.key_times

    def counting(self, uid=None):
        calls.append(1)
        return real(self, uid)

    monkeypatch.setattr(anim.Animation, "key_times", counting)
    first = tl.rows(doc)
    built = len(calls)
    assert built, "the first call has to build the rows"
    for _ in range(5):
        again = tl.rows(doc)
    assert len(calls) == built, "a frame with the animation unchanged rebuilds nothing"
    assert [(r.uid, r.times) for r in again] == [(r.uid, r.times) for r in first]


def test_the_dope_sheet_rows_follow_an_edit_and_a_selection():
    doc = _clip_doc()
    hip = doc.objects[0].uid
    before = tl.rows(doc)[0].times
    doc.set_key(hip, "translation", 5.0, [0, 1, 0])
    assert 5.0 in tl.rows(doc)[0].times and 5.0 not in before
    doc.select([doc.objects[1].uid])
    assert [r.selected for r in tl.rows(doc)] == [False, True, False]


# -- clay-42: key-targeting jumps land on the key -------------------------------------------------


def _off_grid_doc() -> tuple[bd.ClayDoc, int]:
    """A 24 fps document holding a 30 fps clip's last key (1.0333 s, frame 24.8)."""
    doc = bd.ClayDoc()
    hip = _add(doc, "Hip", bone=True)
    doc.add_animation("A")
    doc.set_key(hip, "translation", 0.0, [0, 0, 0])
    doc.set_key(hip, "translation", 1.0 + 1 / 30, [0, 1, 0])
    doc.set_rig_time(0.0)
    return doc, hip


def test_go_to_end_parks_the_playhead_exactly_on_the_last_key():
    doc, _ = _off_grid_doc()
    tl.go_end(doc)
    assert doc.rig_time == pytest.approx(tl.length(doc), abs=1e-9)


def test_clicking_an_off_grid_diamond_parks_the_playhead_exactly_on_its_key():
    doc, hip = _off_grid_doc()
    tab = SimpleNamespace(key_sel=[], key_drag=None)
    rows = tl.rows(doc)
    last = rows[0].times[-1]
    sheet = tl.Sheet(
        x0=100.0, x1=500.0, y0=10.0, ruler_h=20.0, row_h=20.0, visible=2.0, gutter=60.0,
        scroll=0, rows_fit=4, rows=len(rows),
    )

    started = pane._begin(tab, doc, rows, sheet, (sheet.to_x(last), 40.0), False)

    assert started["mode"] == "keys" and tab.key_sel == [(hip, last)]
    assert doc.rig_time == pytest.approx(last, abs=1e-9), "not the nearest frame to it"


def test_scrubbing_the_ruler_still_snaps_to_a_frame():
    doc, _ = _off_grid_doc()
    tl.go_to(doc, 0.51)
    assert doc.rig_time == pytest.approx(tl.time_of(doc, tl.frame_of(doc, 0.51)))


# -- clay-43: keying past the ruler or the channel cap --------------------------------------------


def _bone_doc() -> tuple[bd.ClayDoc, int]:
    doc = bd.ClayDoc()
    hip = _add(doc, "Hip", bone=True)
    doc.set_pose_mode(True)
    doc.select([hip])
    return doc, hip


def test_the_playhead_stops_at_the_end_of_the_ruler():
    doc, _ = _bone_doc()
    doc.set_rig_time(anim.MAX_ANIMATION_SECONDS - 0.01)
    tl.step(doc, 1)
    tl.step(doc, 1)
    assert doc.rig_time <= anim.MAX_ANIMATION_SECONDS


def test_keying_past_the_ruler_or_the_channel_cap_is_an_operror_not_a_valueerror():
    doc, hip = _bone_doc()
    doc.set_rig_state(
        doc.rig_state.with_animations(
            [
                anim.Animation(
                    "Full",
                    True,
                    "linear",
                    {
                        hip: {
                            "translation": anim.Channel(
                                np.arange(anim.MAX_KEYS_PER_CHANNEL) / 60.0,
                                np.zeros((anim.MAX_KEYS_PER_CHANNEL, 3)),
                            )
                        }
                    },
                )
            ]
        )
    )
    doc.set_active_animation(0)
    doc.set_rig_time(100.0)
    steps = len(doc.history)

    doc.auto_key = True
    with pytest.raises(el.OpError, match="keys"):
        doc.pose_bones({hip: {"translation": [0, 1, 0]}})

    assert len(doc.history) == steps

    ctx = _Ctx()
    assert clay_ops.run(ctx, doc, clay_ops.get("insert-key")) is False
    assert ctx.toasts, "the Insert Key op toasts the refusal instead of raising"


# -- clay-10: a meshless bone has a gizmo in Shape mode -------------------------------------------


class _ViewCtx:
    def __init__(self) -> None:
        self.state = type("S", (), {"clay": ClayState()})()
        self.clay_view = None
        self.toasts: list[tuple[str, str]] = []

    def toast(self, message: str, level: str = "info") -> None:
        self.toasts.append((message, level))


@pytest.fixture
def view(gl):
    ctx = _ViewCtx()
    v = clay_view.ClayView(gl, ctx)
    ctx.clay_view = v
    v.app_ctx = ctx
    yield v
    v.release()


def test_a_selected_meshless_bone_has_a_gizmo_and_g_moves_it_in_shape_mode(view) -> None:
    doc = bd.ClayDoc()
    bone = doc.add_bone()
    doc.set_transform(bone.uid, translation=(2.0, 1.0, 0.0))
    doc.select([bone.uid])
    view.draw(doc, RECT, 0.016)

    centre = view.selection_centre(doc)
    assert centre is not None and np.allclose(centre, [2.0, 1.0, 0.0])
    assert view.active_gizmo(doc) is not None

    before = np.array(doc.by_uid(bone.uid).translation, copy=True)
    view._last_mouse = (160.0, 120.0)
    assert view.begin_keyboard_drag(doc, "move"), "G must start on a bone with no mesh"
    view._drag_keyboard(doc, (220.0, 120.0))
    view.settle_drag(doc)
    assert not np.allclose(doc.by_uid(bone.uid).translation, before)


def test_f_frames_a_selected_meshless_bone(view) -> None:
    doc = bd.ClayDoc()
    _add(doc, "Far", translation=(50.0, 0.0, 0.0))
    bone = doc.add_bone()
    doc.set_transform(bone.uid, translation=(-30.0, 0.0, 0.0))
    doc.select([bone.uid])
    view.draw(doc, RECT, 0.016)
    view.frame_selection(doc)
    assert np.allclose(view.camera.target, [-30.0, 0.0, 0.0], atol=0.5)


def test_a_gizmo_the_selection_cannot_place_is_not_hit_testable(view) -> None:
    doc = bd.ClayDoc()
    group_member = _add(doc, "A")
    doc.select([group_member])
    group = doc.group([group_member])
    doc.select([group.uid])
    view.draw(doc, RECT, 0.016)
    if view.selection_centre(doc) is None:
        assert view.active_gizmo(doc) is None
