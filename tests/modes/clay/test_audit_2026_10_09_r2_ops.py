"""Regression tests for the 2026-10-09 second Clay audit's op-registry findings
(clay-55, 59, 61, 62, 63, 64, 69, 96, 108, 109, 110, 124). Each test name is the
claim, and each was run against the unfixed tree first."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import glbimport, ops_clean
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio.modes.clay import ops as clay_ops
from realmspinner.studio.modes.clay import recent_op

ROOT = Path(__file__).resolve().parents[3]
OPS_SRC = ROOT / "src" / "realmspinner" / "studio" / "modes" / "clay" / "ops.py"
MENUTREE_SRC = ROOT / "src" / "realmspinner" / "studio" / "modes" / "clay" / "menutree.py"


class _Ctx:
    """Only what the real ``Ctx`` offers: ``toast(text, level)``."""

    def __init__(self) -> None:
        self.msgs: list[str] = []
        self.clay_view: Any = None
        self.state: Any = None

    def toast(self, message: str, level: str = "info") -> None:
        self.msgs.append(message)


def _box(doc: bd.ClayDoc, name: str = "Box", **kw: Any) -> int:
    obj = doc.add_object(
        bd.Obj(
            uid=bd.new_uid(),
            name=name,
            mesh=bp.box(),
            generator="box",
            params={"size": (1.0, 1.0, 1.0)},
            **kw,
        )
    )
    return obj.uid


def _scaled_box(scale: Any) -> tuple[bd.ClayDoc, int]:
    doc = bd.ClayDoc()
    uid = _box(doc)
    doc.set_transform(uid, scale=np.array(scale, dtype="f8"))
    return doc, uid


def _top_face(doc: bd.ClayDoc, uid: int) -> int:
    return int(np.argmax(bm.face_normals(doc.by_uid(uid).mesh)[:, 1]))


def _group_of_two() -> tuple[bd.ClayDoc, int, int, int]:
    doc = bd.ClayDoc()
    a = _box(doc, "A", translation=np.array([1.0, 0.5, 0.0]))
    b = _box(doc, "B", translation=np.array([3.0, 0.5, 0.0]))
    group = doc.group([a, b])
    doc.select([group.uid])
    return doc, group.uid, a, b


# --- clay-55: a distance labelled metres means world metres ----------------------


def test_an_element_op_distance_means_world_metres_on_a_scaled_object() -> None:
    doc, uid = _scaled_box((4.0, 4.0, 4.0))
    doc.set_element_mode("face")
    doc.set_element_sel(uid, el.ElementSel(faces=[_top_face(doc, uid)]))
    ctx = _Ctx()

    assert clay_ops.run(ctx, doc, clay_ops.get("inset"), thickness=0.1), ctx.msgs

    pts = doc.by_uid(uid).mesh.positions
    top = pts[np.isclose(pts[:, 1], pts[:, 1].max())]
    gaps = np.linalg.norm(top - np.array([0.5, pts[:, 1].max(), 0.5]), axis=1)
    slid_world = float(gaps[gaps > 1e-6].min()) * 4.0
    assert slid_world == pytest.approx(0.1, abs=1e-4), (
        f"Inset '0.100 m' on a 4x object slid the rim {slid_world:.3f} m in the world"
    )


def _seam_strip() -> bm.Mesh:
    """Two quads whose facing edges are 0.05 apart: a seam a weld can close."""
    positions = [
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (1.05, 0, 0), (2, 0, 0), (2, 1, 0), (1.05, 1, 0),
    ]  # fmt: skip
    return bm.from_faces(positions, [[0, 1, 2, 3], [4, 5, 6, 7]])


def _weld_seam(scale: float, eps: float) -> tuple[bool, int, list[str]]:
    doc = bd.ClayDoc()
    uid = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Strip", mesh=_seam_strip())).uid
    doc.set_transform(uid, scale=np.array([scale] * 3))
    doc.set_element_mode("vertex")
    doc.set_element_sel(uid, el.ElementSel(verts=np.arange(8)))
    ctx = _Ctx()
    ran = clay_ops.run(ctx, doc, clay_ops.get("weld"), eps=eps)
    return ran, len(doc.by_uid(uid).mesh.positions), ctx.msgs


def test_weld_distance_in_metres_closes_a_seam_on_a_small_scaled_object() -> None:
    # Seam 0.05 local = 0.0005 m at scale 0.01; a 1 mm weld must close it.
    ran, verts, msgs = _weld_seam(0.01, 0.001)
    assert ran, msgs
    assert verts == 6, f"the 1 mm weld left {verts} vertices on a 0.5 mm seam"


def test_weld_distance_in_metres_leaves_a_wider_seam_alone_on_a_large_scaled_object() -> None:
    # Seam 0.05 local = 0.2 m at scale 4; a 0.1 m weld must not reach across it.
    _, verts, _ = _weld_seam(4.0, 0.1)
    assert verts == 8, f"a 0.1 m weld merged a 0.2 m seam ({verts} vertices left)"


def test_bevel_width_is_converted_like_inset_and_weld(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from realmspinner.kernels.mesh import ops_bevel

    seen: dict[str, float] = {}

    def bevel(mesh: Any, sel: Any, *, width: float = 0.05) -> Any:
        seen["width"] = width
        return mesh, sel

    monkeypatch.setattr(ops_bevel, "bevel_edges", bevel)
    doc, uid = _scaled_box((4.0, 4.0, 4.0))
    doc.set_element_mode("edge")
    doc.set_element_sel(uid, el.ElementSel(edges=[(0, 1)]))
    clay_ops.run(_Ctx(), doc, clay_ops.get("bevel"), width=0.4)

    assert seen["width"] == pytest.approx(0.1)


def test_a_non_uniform_scale_is_read_as_the_geometric_mean_of_its_axes() -> None:
    doc, uid = _scaled_box((8.0, 1.0, 1.0))
    assert clay_ops.world_scale(doc, uid) == pytest.approx(2.0)
    child_doc, parent = _scaled_box((2.0, 2.0, 2.0))
    child = _box(child_doc, "Child")
    child_doc.set_transform(child, scale=np.array([3.0, 3.0, 3.0]))
    child_doc.set_parent(child, parent, keep_world=False)
    assert clay_ops.world_scale(child_doc, child) == pytest.approx(6.0)


def test_a_distance_on_a_zero_scale_object_is_refused_naming_the_cause() -> None:
    doc, uid = _scaled_box((1.0, 1.0, 0.0))
    doc.set_element_mode("face")
    doc.set_element_sel(uid, el.ElementSel(faces=[0]))
    ctx = _Ctx()
    steps = len(doc.history)

    assert not clay_ops.run(ctx, doc, clay_ops.get("inset"), thickness=0.1)
    assert len(doc.history) == steps
    assert ctx.msgs and "scale" in ctx.msgs[-1], ctx.msgs


# --- clay-59: a refused multi-object run changes no object -----------------------


def test_a_multi_object_element_op_refused_at_the_triangle_ceiling_changes_no_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc = bd.ClayDoc()
    uids = [_box(doc, f"b{i}") for i in range(3)]
    doc.select(uids)
    doc.set_element_mode("face")
    for uid in uids:
        doc.set_element_sel(uid, el.select_all(doc.by_uid(uid).mesh, "face"))
    # 3 boxes = 36 triangles; subdividing one takes it to 144/3. The first two
    # fit under this, the third would cross it.
    monkeypatch.setattr(glbimport, "MAX_TRIANGLES", 36 + 40 + 40 + 10)
    before = [doc.by_uid(u).mesh for u in uids]
    steps = len(doc.history)
    ctx = _Ctx()

    assert not clay_ops.run(ctx, doc, clay_ops.get("subdivide"))

    assert [doc.by_uid(u).mesh for u in uids] == before, "an earlier object was rewritten"
    assert len(doc.history) == steps, "a refused run left an undo step"
    assert ctx.msgs and "triangles" in ctx.msgs[-1], ctx.msgs


# --- clay-61: the live-gesture refusal is at ops.run and recent_op.adjust --------


def _gesture_ctx(*, dragging: bool = False, uv_mode: str = "") -> _Ctx:
    ctx = _Ctx()
    ctx.clay_view = SimpleNamespace(dragging=dragging)
    uv_view = SimpleNamespace(drag_mode=uv_mode)
    ctx.state = SimpleNamespace(clay=SimpleNamespace(active=SimpleNamespace(uv_view=uv_view)))
    return ctx


def test_run_and_adjust_refuse_under_a_live_gesture_from_every_door() -> None:
    doc, uid = _scaled_box((1.0, 1.0, 1.0))
    doc.select([uid])
    ctx = _gesture_ctx(dragging=True)
    head = doc.history.head

    assert not clay_ops.run(ctx, doc, clay_ops.get("duplicate")), "run ignored a live 3-D drag"
    assert doc.history.head == head and len(doc.objects) == 1

    uv_ctx = _gesture_ctx(uv_mode="rotate")
    assert not clay_ops.run(uv_ctx, doc, clay_ops.get("duplicate")), "run ignored a live UV rotate"
    assert doc.history.head == head and len(doc.objects) == 1

    # The Last action card: a live op, then a gesture, then an adjust.
    doc.set_element_mode("face")
    doc.set_element_sel(uid, el.select_all(doc.by_uid(uid).mesh, "face"))
    quiet = _gesture_ctx()
    assert clay_ops.run(quiet, doc, clay_ops.get("inset"), thickness=0.1)
    pushed = doc.history.head
    mesh = doc.by_uid(uid).mesh
    quiet.clay_view.dragging = True
    result = recent_op.adjust(quiet, doc, thickness=0.2)

    assert not result.ok and result.message
    assert doc.history.head == pushed and doc.by_uid(uid).mesh is mesh, (
        "adjust undid and re-ran the op under a live drag"
    )

    quiet.clay_view.dragging = False
    assert recent_op.adjust(quiet, doc, thickness=0.2).ok


def test_run_goes_through_when_the_key_layer_says_the_chord_is_allowed_mid_gesture() -> None:
    doc, uid = _scaled_box((1.0, 1.0, 1.0))
    doc.select([uid])
    ctx = _gesture_ctx(dragging=True)
    assert clay_ops.run(ctx, doc, clay_ops.get("duplicate"), during_gesture=True)
    assert len(doc.objects) == 2


# --- clay-62: a group has no geometry, and an op that needs some says so ---------


@pytest.mark.parametrize(
    "name",
    [
        "drop-to-ground",
        "origin-to-bounds",
        "origin-to-base",
        "mirror-x",
        "mirror-y",
        "mirror-z",
        "unwrap",
        "recalc-normals",
        "shade-smooth",
        "shade-flat",
    ],
)
def test_an_op_that_needs_geometry_on_a_selected_group_says_why_it_did_nothing(
    name: str,
) -> None:
    doc, group, a, b = _group_of_two()
    ctx = _Ctx()
    steps = len(doc.history)

    assert not clay_ops.run(ctx, doc, clay_ops.get(name))

    assert len(doc.history) == steps
    assert ctx.msgs and "no geometry" in ctx.msgs[-1], f"{name} said nothing: {ctx.msgs}"


def test_a_group_toast_does_not_hide_the_shapes_of_a_mixed_selection() -> None:
    doc, group, a, b = _group_of_two()
    doc.select([group, a])
    ctx = _Ctx()
    before = doc.by_uid(a).mesh

    assert clay_ops.run(ctx, doc, clay_ops.get("mirror-x"))
    assert doc.by_uid(a).mesh is not before
    assert not ctx.msgs


# --- clay-63: duplicating a group takes its children ------------------------------


def test_duplicating_a_selected_group_copies_its_children_with_it() -> None:
    doc, group, a, b = _group_of_two()
    before = {o.uid for o in doc.objects}
    ctx = _Ctx()

    assert clay_ops.run(ctx, doc, clay_ops.get("duplicate")), ctx.msgs

    new = [o for o in doc.objects if o.uid not in before]
    assert len(new) == 3, [o.name for o in new]
    copy_group = next(o for o in new if o.parent is None)
    kids = [o for o in new if o.parent == copy_group.uid]
    assert {k.name for k in kids} == {"A.001", "B.001"}
    assert all(k.uid not in before for k in kids)
    assert doc.by_uid(a).parent == group and doc.by_uid(b).parent == group
    assert doc.selection == {copy_group.uid}
    doc.undo()
    assert {o.uid for o in doc.objects} == before, "one press must be one undo step"


def test_mirror_copy_of_a_selected_group_copies_its_children_reflected() -> None:
    doc, group, a, b = _group_of_two()
    before = {o.uid for o in doc.objects}
    ctx = _Ctx()

    assert clay_ops.run(ctx, doc, clay_ops.get("mirror-copy"), axis=0, offset=0.0), ctx.msgs

    new = [o for o in doc.objects if o.uid not in before]
    assert len(new) == 3
    copy_group = next(o for o in new if o.parent is None)
    kids = {o.name: o for o in new if o.parent == copy_group.uid}
    assert set(kids) == {"A.001", "B.001"}
    assert doc.world_matrix(kids["A.001"].uid)[0, 3] == pytest.approx(-1.0, abs=1e-5)
    assert doc.world_matrix(kids["B.001"].uid)[0, 3] == pytest.approx(-3.0, abs=1e-5)
    assert doc.by_uid(a).parent == group


def test_duplicating_a_group_still_refuses_past_the_object_ceiling_and_creates_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc, group, a, b = _group_of_two()
    monkeypatch.setattr(glbimport, "MAX_OBJECTS", 5)  # 3 now, +3 would be 6
    ctx = _Ctx()
    steps = len(doc.history)

    for name, kw in (("duplicate", {}), ("mirror-copy", {"axis": 0, "offset": 0.0})):
        assert not clay_ops.run(ctx, doc, clay_ops.get(name), **kw)
    assert len(doc.objects) == 3 and len(doc.history) == steps
    assert ctx.msgs and "objects" in ctx.msgs[-1]


# --- clay-64: an inset of nothing is a refusal ------------------------------------


def _inset_doc() -> tuple[bd.ClayDoc, int]:
    doc = bd.ClayDoc()
    uid = _box(doc)
    doc.select([uid])
    doc.set_element_mode("face")
    doc.set_element_sel(uid, el.ElementSel(faces=[0]))
    return doc, uid


def test_inset_with_zero_thickness_and_zero_depth_is_refused_rather_than_leaving_zero_area_faces(
) -> None:
    for region in (0.0, 1.0):
        doc, uid = _inset_doc()
        ctx = _Ctx()
        mesh = doc.by_uid(uid).mesh
        steps = len(doc.history)

        inset = clay_ops.get("inset")
        assert not clay_ops.run(ctx, doc, inset, thickness=0.0, depth=0.0, region=region)

        assert doc.by_uid(uid).mesh is mesh and len(doc.history) == steps
        assert ctx.msgs and "thickness" in ctx.msgs[-1] and "depth" in ctx.msgs[-1], ctx.msgs


def test_a_depth_only_inset_is_still_legal() -> None:
    doc, uid = _inset_doc()
    ctx = _Ctx()
    assert clay_ops.run(ctx, doc, clay_ops.get("inset"), thickness=0.0, depth=-0.2), ctx.msgs


# --- clay-96: the clean ceiling is charged across the whole selection -------------


def test_recalculate_normals_refuses_a_selection_whose_total_corners_pass_the_clean_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ops_clean, "MAX_CLEAN_CORNERS", 40)  # a box has 24 corners
    doc = bd.ClayDoc()
    one = _box(doc, "one")
    two = _box(doc, "two")
    doc.select([one])
    ctx = _Ctx()
    clay_ops.run(ctx, doc, clay_ops.get("recalc-normals"))
    assert not ctx.msgs, "a single object under the ceiling was refused"

    doc.select([one, two])
    steps = len(doc.history)
    assert not clay_ops.run(ctx, doc, clay_ops.get("recalc-normals"))

    assert len(doc.history) == steps
    assert ctx.msgs and "48" in ctx.msgs[-1] and "40" in ctx.msgs[-1], ctx.msgs


# --- clay-108: Snap to Grid's step has a floor ------------------------------------


def test_snap_to_grid_with_a_denormal_step_is_a_refusal_not_an_overflow_error() -> None:
    snap = clay_ops.get("snap-to-grid")
    (param,) = snap.params
    assert param.low > 0.0, "the step range still admits 0, which the op then refuses"

    doc, uid = _scaled_box((1.0, 1.0, 1.0))
    doc.set_transform(uid, translation=np.array([0.37, 1.2, -3.3]))
    doc.select([uid])
    ctx = _Ctx()
    assert clay_ops.run(ctx, doc, snap, step=5e-324) is False  # a refusal, not a raise
    assert ctx.msgs and "step" in ctx.msgs[-1], ctx.msgs

    # The op's own guard, past the clamp: a step whose quotient is not finite is
    # a refusal (a toast, nothing moved), never a bare OverflowError.
    guard_ctx = _Ctx()
    assert clay_ops._snap_to_grid(guard_ctx, doc, step=5e-324) is False
    assert guard_ctx.msgs and "too small" in guard_ctx.msgs[-1], guard_ctx.msgs


# --- clay-109: the ops import pin resolves relative imports ------------------------


def test_the_ops_registry_pin_resolves_relative_imports_before_banning_them() -> None:
    from .test_clay_ops import ops_import_violations

    clean = OPS_SRC.read_text(encoding="utf-8")
    assert ops_import_violations(clean) == []

    for smuggled, expect in (
        ("from .ui.panes import menu", "realmspinner.studio.modes.clay.ui.panes"),
        ("from .ui import view", "realmspinner.studio.modes.clay.ui"),
        ("from . import ui", "realmspinner.studio.modes.clay.ui"),
        ("from ...panes import familiar_dock", "realmspinner.studio.panes"),
        ("from ..poser import mode", "realmspinner.studio.modes.poser"),
        ("from ...modes.poser.engine import spec", "realmspinner.studio.modes.poser.engine"),
        ("import imgui", "imgui"),
        ("from ....service import jobs", "realmspinner.service"),
    ):
        found = ops_import_violations(clean + f"\n\ndef _smuggled():\n    {smuggled}\n")
        assert any(v == expect or v.startswith(expect + ".") for v in found), (smuggled, found)


# --- clay-110: no deleted surface is named as live ----------------------------------


def test_the_ops_registry_docstring_names_no_deleted_tools_pane_surface() -> None:
    for path in (OPS_SRC, MENUTREE_SRC):
        text = path.read_text(encoding="utf-8").lower()
        assert "tools pane" not in text and "tools-pane" not in text, path.name
    assert "the only thing that knows a popup exists" not in OPS_SRC.read_text(encoding="utf-8")


# --- clay-124 / clay-69 / clay-55 (preview): the op drag --------------------------


def test_the_op_drag_distance_scale_in_an_orthographic_view_ignores_the_element_depth() -> None:
    from realmspinner.studio.modes.clay.ui._view_opdrag import OpDragOps
    from realmspinner.studio.viewer.camera import Camera

    camera = Camera()
    camera.distance = 8.0
    camera.orthographic = True
    fake = SimpleNamespace(camera=camera, _rect=(0.0, 0.0, 128.0, 96.0))
    near = OpDragOps._metres_per_pixel(fake, camera.position + np.array([0.0, 0.0, 0.5]))
    far = OpDragOps._metres_per_pixel(fake, camera.position + np.array([0.0, 0.0, 40.0]))
    expected = 2.0 * 8.0 * math.tan(math.radians(camera.fov * 0.5)) / 96.0

    assert near == pytest.approx(expected) and far == pytest.approx(expected)

    camera.orthographic = False  # perspective still scales with the depth
    assert OpDragOps._metres_per_pixel(fake, camera.position + np.array([0.0, 0.0, 0.5])) < near


RECT = (0.0, 0.0, 128.0, 96.0)


class _ViewCtx:
    def __init__(self) -> None:
        from realmspinner.studio.modes.clay.state import ClayState

        self.state = type("S", (), {"clay": ClayState()})()
        self.clay_view = None
        self.errors: list[str] = []

    def toast(self, message: str, level: str = "info") -> None:
        if level == "error":
            self.errors.append(message)


@pytest.fixture
def view(gl, monkeypatch):
    import pygame

    from realmspinner.studio.modes.clay.ui import view as clay_view

    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0, raising=False)
    ctx = _ViewCtx()
    v = clay_view.ClayView(gl, ctx)
    ctx.clay_view = v
    yield v
    v.release()


def _two_boxes(scale: float = 1.0) -> tuple[bd.ClayDoc, int, int]:
    doc = bd.ClayDoc()
    uids = []
    for i in range(2):
        uid = _box(doc, f"Box{i}", translation=np.array([i * 3.0, 0.0, 0.0]))
        doc.set_transform(uid, scale=np.array([scale] * 3))
        uids.append(uid)
    doc.set_element_mode("face")
    for uid in uids:
        doc.set_element_sel(uid, el.ElementSel(faces=[0]))
    return doc, uids[0], uids[1]


def _ready(view: Any, doc: bd.ClayDoc) -> None:
    view.frame_selection(doc)
    view.draw(doc, RECT, 0.0)
    view._last_mouse = (64.0, 48.0)


def test_an_op_drag_preview_puts_back_an_object_that_accepted_a_smaller_value_and_now_refuses(
    view,
) -> None:
    doc, a, b = _two_boxes()
    _ready(view, doc)
    assert view.begin_op_drag(doc, clay_ops.get("inset"))
    od = view._op_drag
    refusing, real = od.bases[a][0], od.kernel

    def kernel(mesh: Any, sel: Any, **params: Any) -> Any:
        if mesh is refusing and params["thickness"] > 1.0:
            raise el.OpError("too wide for Box0")
        return real(mesh, sel, **params)

    od.kernel = kernel
    view._motion(doc, (70.0, 48.0))
    assert not od.refused
    previewed = view._cache[a].gpu

    view._motion(doc, (100.0, 48.0))
    assert "Box0" in od.refused
    assert view._cache[a].gpu is not previewed, (
        "Box0 kept the earlier preview although Enter would leave it untouched"
    )
    view.cancel_drag(doc)


def test_the_op_drag_preview_converts_metres_to_local_units_like_the_commit(view) -> None:
    doc, a, b = _two_boxes(scale=4.0)
    _ready(view, doc)
    assert view.begin_op_drag(doc, clay_ops.get("inset"))
    od = view._op_drag
    real = od.kernel
    seen: list[float] = []
    od.kernel = lambda mesh, sel, **p: (seen.append(p["thickness"]), real(mesh, sel, **p))[1]

    view._motion(doc, (90.0, 48.0))

    assert od.value > 0.0
    assert seen and all(v == pytest.approx(od.value / 4.0) for v in seen), (seen, od.value)
    view.cancel_drag(doc)


def test_an_inset_drag_preview_refuses_zero_thickness_and_zero_depth_like_the_commit(
    view,
) -> None:
    doc, a, b = _two_boxes()
    _ready(view, doc)
    assert view.begin_op_drag(doc, clay_ops.get("inset"))
    od = view._op_drag

    od.value = 0.0
    view._op_drag_preview(doc, 0.0)

    assert "thickness" in od.refused and "depth" in od.refused, od.refused
    view.cancel_drag(doc)
