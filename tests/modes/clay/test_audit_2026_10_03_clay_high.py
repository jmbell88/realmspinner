"""Regression tests for the 2026-10-03 audit's High Clay findings
clay-01, 02, 03, 04, 12, 17, 18 and 19."""

from __future__ import annotations

import contextlib
import json
from types import SimpleNamespace
from typing import Any

from realmspinner.mcp import protocol
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay.agent import dispatch as agent_clay


class _Cache:
    def invalidate(self) -> None:
        pass


class _Ctx:
    def __init__(self) -> None:
        self.state = SimpleNamespace(clay=None)
        self.svc = None
        self.cache = _Cache()
        self.toasts: list[tuple[str, str]] = []
        self.settings = SimpleNamespace()

    def toast(self, message: str, level: str = "info") -> None:
        self.toasts.append((message, level))


def _payload(result: dict) -> Any:
    return json.loads(result["content"][0]["text"])


def _boxes(n: int = 4) -> tuple[_Ctx, agent_clay.Session, list[int]]:
    ctx = _Ctx()
    session = agent_clay.Session()
    uids = []
    for i in range(n):
        r = agent_clay.call(
            ctx, session, "clay_add_primitive", {"generator": "box", "translation": [3.0 * i, 0, 0]}
        )
        assert r["isError"] is False, r
        uids.append(_payload(r)["uid"])
    return ctx, session, uids


def _doc(ctx: _Ctx, session: agent_clay.Session) -> Any:
    return clay_mode.ensure(ctx).get(session.tab_uid).doc


# --- clay-01 ---------------------------------------------------------------


def test_clay_separate_reply_stays_under_the_frame_budget_for_the_most_pieces_the_kernel_allows() -> None:  # noqa: E501
    ctx = _Ctx()
    session = agent_clay.Session()
    n = 9000
    positions = []
    faces = []
    for i in range(n):
        x = float(i)
        positions += [[x, 0.0, 0.0], [x + 0.5, 0.0, 0.0], [x, 0.5, 0.0]]
        faces.append([3 * i, 3 * i + 1, 3 * i + 2])
    added = agent_clay.call(
        ctx, session, "clay_add_mesh", {"positions": positions, "faces": faces}
    )
    assert added["isError"] is False, added
    uid = _payload(added)["uid"]
    result = agent_clay.call(ctx, session, "clay_separate", {"uid": uid, "by": "loose_parts"})
    wire = len(json.dumps(result))
    assert wire < protocol.MAX_FRAME, wire
    assert result["isError"] is False
    assert len(_payload(result)["uids"]) == n


# --- clay-02 ---------------------------------------------------------------


def test_a_digit_string_in_an_array_argument_is_refused_not_read_as_digits() -> None:
    cases = [
        ("clay_delete", {}),
        ("clay_select", {}),
        ("clay_lock", {"locked": True}),
        ("clay_boolean", {"kind": "union"}),
    ]
    for tool, extra in cases:
        ctx, session, uids = _boxes(3)
        doc = _doc(ctx, session)
        before = [o.uid for o in doc.objects]
        digits = "".join(str(u) for u in uids)
        r = agent_clay.call(ctx, session, tool, {"uids": digits, **extra})
        assert r["isError"] is True, (tool, r)
        assert [o.uid for o in doc.objects] == before, tool


# --- clay-03 / clay-04 -----------------------------------------------------


def _kobj(name: str, **kwargs: Any) -> Any:
    from realmspinner.kernels.mesh import document as bd
    from realmspinner.kernels.mesh import primitives as bp

    return bd.Obj(uid=bd.new_uid(), name=name, mesh=bp.box(), **kwargs)


def test_a_document_the_app_can_author_always_reopens_after_a_save_with_zero_scale() -> None:
    import pytest

    from realmspinner.kernels.mesh import document as bd
    from realmspinner.kernels.mesh import elements as el
    from realmspinner.kernels.mesh import serialize as ser

    a = _kobj("A")
    doc = bd.ClayDoc([a])
    for kwargs in ({"scale": [0.0, 0.0, 0.0]}, {"rotation": [0.0, 0.0, 0.0, 0.0]}):
        with contextlib.suppress(el.OpError):
            doc.set_transform(a.uid, **kwargs)
    # Whatever the door let through must be something the reader reopens.
    ser.read_rblk(ser.rblk_bytes(doc))
    with pytest.raises(el.OpError):
        doc.set_transform(a.uid, scale=[0.0, 0.0, 0.0])


def test_join_objects_that_refuses_a_zero_scale_ancestor_leaves_the_target_untouched() -> None:
    import pytest

    from realmspinner.kernels.mesh import document as bd
    from realmspinner.kernels.mesh import elements as el
    from realmspinner.kernels.mesh import ops as geom

    gp = _kobj("GP")
    a = _kobj("A")
    b = _kobj("B", translation=(5.0, 0.0, 0.0))
    c = _kobj("C")
    doc = bd.ClayDoc([gp, a, b, c])
    doc.set_parent(b.uid, gp.uid)
    doc.set_parent(c.uid, b.uid)
    doc.set_transform(gp.uid, scale=[0.0, 1.0, 1.0])
    mesh_before = doc.by_uid(a.uid).mesh
    steps = len(doc.history.history())
    merged = geom.join([doc.by_uid(a.uid), doc.by_uid(b.uid)], eps=0.0)
    with pytest.raises(el.OpError):
        doc.join_objects(a.uid, merged, [b.uid])
    assert doc.by_uid(a.uid).mesh is mesh_before
    assert doc.by_uid(a.uid).generator == a.generator
    assert {o.uid for o in doc.objects} == {gp.uid, a.uid, b.uid, c.uid}
    assert len(doc.history.history()) == steps


# --- clay-12 ---------------------------------------------------------------


def test_a_keyboard_tab_switch_does_not_overwrite_the_incoming_tabs_stored_camera() -> None:
    import pytest

    from realmspinner.kernels.mesh import document as bd
    from realmspinner.kernels.mesh import primitives as bp
    from realmspinner.studio.modes.clay import generate as clay_generate

    ctx = _Ctx()

    def make(title: str) -> Any:
        doc = bd.ClayDoc()
        doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
        return clay_mode.adopt(ctx, doc, title=title)

    first = make("One")
    second = make("Two")
    state = clay_mode.ensure(ctx)
    state.activate(second.uid)
    second.view.yaw = 0.5
    # The viewport last drew *first*; Ctrl+Tab then moved active_uid to second.
    state.camera_tab = first.uid
    state.activate(second.uid)
    camera = SimpleNamespace(theta=2.5, phi=0.7, distance=11.0, target=(4.0, 5.0, 6.0))
    ctx.clay_view = SimpleNamespace(camera=camera)
    clay_generate.poll(ctx)
    assert second.view.yaw == pytest.approx(0.5)
    assert first.view.yaw == pytest.approx(2.5)


# --- clay-17 ---------------------------------------------------------------


def test_picking_a_palette_slot_in_properties_repaints_the_objects_faces_so_it_exports_with_that_slot() -> None:  # noqa: E501
    import numpy as np

    from realmspinner.kernels.mesh import document as bd
    from realmspinner.kernels.mesh import primitives as bp
    from realmspinner.studio.modes.clay.ui.panes import props as clay_props

    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box()))
    steps = len(doc.history)
    index = doc.add_material_and_assign(obj.uid, repaint=True)
    assert np.all(obj.mesh.material == index)
    assert len(doc.history) == steps + 1
    assert doc.undo()
    assert not np.any(obj.mesh.material == index)

    doc.add_material()
    steps = len(doc.history)
    doc.repaint_object(obj.uid, 1)
    assert obj.material == 1 and np.all(obj.mesh.material == 1)
    assert len(doc.history) == steps + 1

    assert clay_props._apply_library_material(doc, [obj.uid], doc.materials[0])
    assert np.all(obj.mesh.material == len(doc.materials) - 1)


# --- clay-18 ---------------------------------------------------------------


def test_escape_during_a_live_uv_rotate_cancels_it_and_leaves_the_selection_alone() -> None:
    import numpy as np
    import pygame

    from realmspinner.kernels.mesh import document as bd
    from realmspinner.kernels.mesh import mesh as bm
    from realmspinner.studio.modes.clay.ui.panes import uv as clay_uv

    mesh = bm.from_faces(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        [[0, 1, 2, 3]],
        uv=[[[0.0, 0.0], [0.4, 0.0], [0.4, 0.4], [0.0, 0.4]]],
    )
    ctx = _Ctx()
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=mesh))
    tab = clay_mode.adopt(ctx, doc, title="T")
    doc.select([obj.uid])
    view_state = tab.uv_view
    view_state.selected_islands = frozenset({0})
    before = len(doc.history)
    base = doc.by_uid(obj.uid).mesh
    ids = np.array([0], dtype="i4")
    assert clay_uv.begin_live_transform(doc, view_state, "rotate", base, ids, (0.2, 0.2))
    clay_uv.update_live_transform(doc, obj.uid, view_state, (0.3, 0.3))
    assert doc.by_uid(obj.uid).mesh is not base

    event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0)
    assert clay_mode.handle_key(ctx, event) is True

    assert view_state.drag_mode == ""
    assert doc.selection == {obj.uid}
    assert doc.by_uid(obj.uid).mesh is base
    assert len(doc.history) == before
