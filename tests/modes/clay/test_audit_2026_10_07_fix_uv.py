"""The 2026-10-07 ``clay`` audit, UVs / texturing / the UV pane: clay-01 (the
``overlap_faces`` half), clay-03, clay-04 (the ``clamp_params`` half), clay-15,
clay-16, clay-48, clay-57, clay-77, clay-78 and clay-79.

Each test's name is the claim. clay-03 has a rendered test (a real standalone GL
context, skipped where there is none) and a GL-free one of the same claim.
"""

from __future__ import annotations

import inspect
import io
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from _ui_context import imgui_context

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import uv as bmuv
from realmspinner.kernels.mesh import uvtools
from realmspinner.kernels.mesh.elements import OpError
from realmspinner.kernels.pixel import Document
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay import texture_link as tl
from realmspinner.studio.modes.clay.state import ClayState, ClayTab
from realmspinner.studio.modes.clay.ui import view as clay_view
from realmspinner.studio.modes.clay.ui.panes import uv as clay_uv
from realmspinner.studio.state import AppState


@pytest.fixture
def ui(monkeypatch):
    with imgui_context(monkeypatch) as imgui:
        yield imgui


class _Ctx:
    """The one thing a refusal path needs from the app: ``toast``."""

    def __init__(self) -> None:
        self.toasts: list[tuple[str, str]] = []

    def toast(self, text: str, level: str = "info", **_kw: Any) -> None:
        self.toasts.append((text, level))


def _strip_mesh(count: int) -> bm.Mesh:
    """*count* disjoint unit quads in a row, each with its own uv island."""
    positions, faces = [], []
    for i in range(count):
        x = float(i) * 2.0
        base = len(positions)
        positions += [(x, 0, 0), (x + 1, 0, 0), (x + 1, 1, 0), (x, 1, 0)]
        faces.append([base, base + 1, base + 2, base + 3])
    return bmuv.box_unwrap(bp._mesh(np.array(positions, "f4"), faces))


def _doc_with(mesh: bm.Mesh) -> tuple[bd.ClayDoc, bd.Obj]:
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="O", mesh=mesh))
    return doc, doc.by_uid(obj.uid)


def _two_island_mesh() -> bm.Mesh:
    positions = [
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [2, 0, 0], [3, 0, 0], [3, 1, 0], [2, 1, 0],
    ]
    faces = [[0, 1, 2, 3], [4, 5, 6, 7]]
    uv = [
        [[0.0, 0.0], [0.4, 0.0], [0.4, 0.4], [0.0, 0.4]],
        [[0.6, 0.6], [1.0, 0.6], [1.0, 1.0], [0.6, 1.0]],
    ]
    return bm.from_faces(positions, faces, uv=uv)


# --- clay-03: v runs down the image --------------------------------------------


def _textured_box_png(gl, image: np.ndarray, view: str) -> np.ndarray:
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="a", mesh=bp.box((1.0, 1.0, 1.0))))
    doc.add_texture(0, 32)
    material = doc.materials[0]
    doc.set_material(
        0,
        replace(
            material,
            base_color=(32, 32, image.tobytes()),
            base_color_factor=(1, 1, 1, 1),
            nearest=True,
        ),
    )
    state = clay_mode.ClayState()
    tab = clay_mode.ClayTab(doc=doc)
    state.add(tab)
    state.active_uid = tab.uid
    ctx = SimpleNamespace(state=SimpleNamespace(clay=state, mode="clay"), settings=None)
    clay_viewer = clay_view.ClayView(gl, ctx)
    ctx.clay_view = clay_viewer
    try:
        png = clay_viewer.render_png(doc, size=256, view=view, shading="unlit")
    finally:
        clay_viewer.release()
    from PIL import Image

    return np.array(Image.open(io.BytesIO(png)).convert("RGB"))


def test_a_picture_painted_top_red_renders_top_red_on_a_box_wall(gl) -> None:
    """The audit's reproduction (clay-texturing-02): row 0 of an Inker picture
    is red, and the box's front wall rendered it at the *bottom*."""
    image = np.zeros((32, 32, 4), np.uint8)
    image[:16] = (255, 0, 0, 255)
    image[16:] = (0, 0, 255, 255)
    a = _textured_box_png(gl, image, "front")
    cy = a.shape[0] // 2
    top, bottom = a[cy - 40, 128], a[cy + 40, 128]
    assert top[0] > top[2], f"screen-top pixel {top} is not red: the picture is upside down"
    assert bottom[2] > bottom[0], f"screen-bottom pixel {bottom} is not blue"


def test_a_picture_painted_left_red_still_renders_left_red_on_a_box_wall(gl) -> None:
    """The horizontal half was always right; the v fix must not disturb it."""
    image = np.zeros((32, 32, 4), np.uint8)
    image[:, :16] = (255, 0, 0, 255)
    image[:, 16:] = (0, 0, 255, 255)
    a = _textured_box_png(gl, image, "front")
    cy = a.shape[0] // 2
    left, right = a[cy, 128 - 40], a[cy, 128 + 40]
    assert left[0] > left[2] and right[2] > right[0]


def test_a_box_wall_has_a_smaller_v_at_its_top_edge_than_at_its_bottom_edge() -> None:
    """The GL-free form of the same claim: row 0 of the image is v = 0, so the
    top of a wall is the small v."""
    for mesh in (bp.box((1.0, 2.0, 3.0)), bmuv.box_unwrap(bp.box((3.0, 1.0, 0.5)))):
        normals = bm.face_normals(mesh)
        # The two walls facing +/-Z, where the cube projection reads u from X and
        # v from Y. (The X-facing pair reads u from Y and v from Z, a sideways
        # picture, which is the table's layout and not the v convention.)
        walls = np.flatnonzero(np.abs(normals[:, 2]) > 0.9)
        assert len(walls) == 2
        for face in walls:
            corners = np.arange(int(mesh.starts[face]), int(mesh.starts[face + 1]))
            y = mesh.positions[mesh.loops[corners], 1]
            v = mesh.uv[corners, 1]
            assert v[y > 0].max() < v[y < 0].min(), f"wall {face}: top v is not above bottom v"


@pytest.mark.parametrize(
    "name",
    ["cylinder", "cone", "uv_sphere", "capsule", "icosphere", "column",
     "pyramid", "wedge", "ramp", "rounded_box", "wall", "doorway", "stairs", "lathe"],
)
def test_every_upright_generator_writes_v_down_the_image(name: str) -> None:
    """Heights and v run opposite ways over a generator's side faces (the
    generators are written v-up and ``_mesh`` turns them over once). The torus is
    left out: its v loops round the tube, so it has no top."""
    defaults, build = bp.GENERATORS[name]
    mesh = build(**defaults)
    normals = bm.face_normals(mesh)
    side = np.flatnonzero(np.abs(normals[:, 1]) < 0.6)
    assert len(side)
    ys, vs = [], []
    for face in side:
        corners = np.arange(int(mesh.starts[face]), int(mesh.starts[face + 1]))
        ys.extend(mesh.positions[mesh.loops[corners], 1].tolist())
        vs.extend(mesh.uv[corners, 1].tolist())
    ys, vs = np.asarray(ys), np.asarray(vs)
    assert float(np.corrcoef(ys, vs)[0, 1]) < -0.3, f"{name}: v grows with height (v-up)"
    assert vs.min() >= -1e-6 and vs.max() <= 1.0 + 1e-6, "islands stay in the unit square"


def test_every_generators_uv_stays_inside_the_unit_square_after_the_turn() -> None:
    for name, (defaults, build) in bp.GENERATORS.items():
        uv = build(**defaults).uv
        assert uv is not None
        assert uv.min() >= -1e-6 and uv.max() <= 1.0 + 1e-6, name


class _RecordingDrawList:
    def __init__(self) -> None:
        self.images: list[tuple] = []
        self.rects: list[tuple] = []

    def add_rect_filled(self, p0, p1, *_a) -> None:
        self.rects.append((p0, p1))

    def add_rect(self, *_a, **_k) -> None:
        pass

    def add_image(self, *args, **kwargs) -> None:
        self.images.append((args, kwargs))


def test_the_uv_pane_backdrop_shows_the_texture_the_way_up_the_viewport_samples_it(
    ui, monkeypatch
) -> None:
    """Row 0 of the image is v = 0 in the viewport, so the pane must put it at
    the top of the square with the image's own uv range (no flip passed)."""
    monkeypatch.setattr(clay_uv.widgets, "texture_ref", lambda t: "ref")
    draw = _RecordingDrawList()
    view = clay_uv.UvPaneState().view
    ui.new_frame()
    try:
        clay_uv._backdrop(draw, view, (0.0, 0.0), SimpleNamespace(glo=1))
    finally:
        ui.end_frame()
    (args, kwargs), = draw.images
    top_left, bottom_right = args[1], args[2]
    assert top_left[1] < bottom_right[1], "v = 0 is the top edge of the square on screen"
    assert len(args) == 3 and not kwargs, "the image is drawn with its default (unflipped) uv"
    # And an island corner at v = 0 maps to that same top edge.
    assert clay_uv._to_screen(view, (0.0, 0.0), 0.0, 0.0)[1] == top_left[1]


# --- clay-04: clamp_params and float32 -------------------------------------------


def test_clamp_params_refuses_a_length_that_overflows_float32_positions() -> None:
    with pytest.raises(ValueError, match="radius"):
        bp.clamp_params("cylinder", {**bp.GENERATORS["cylinder"][0], "radius": 1e39})
    with pytest.raises(ValueError, match="size"):
        bp.clamp_params("box", {"size": (1e39, 1.0, 1.0)})


def _everything_huge(value: Any, big: float) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return big
    if isinstance(value, (list, tuple)):
        return type(value)(_everything_huge(v, big) for v in value)
    return value


def test_every_generator_refuses_a_parameter_at_1e39_by_name() -> None:
    assert set(bp.GENERATORS) >= bp.CLAY_GENERATOR_NAMES
    for name, (defaults, _build) in bp.GENERATORS.items():
        for key, default in defaults.items():
            huge = _everything_huge(default, 1e39)
            if huge == default:
                continue  # a bool or a string: nothing numeric to overflow
            with pytest.raises(ValueError, match=key):
                bp.clamp_params(name, {**defaults, key: huge})


def test_a_length_just_inside_float32_still_builds_finite_positions() -> None:
    for name in ("box", "cylinder", "wall"):
        defaults, build = bp.GENERATORS[name]
        edited = {
            k: _everything_huge(v, 3e38) if k in ("size", "radius", "length") else v
            for k, v in defaults.items()
        }
        mesh = build(**bp.clamp_params(name, edited))
        assert np.isfinite(mesh.positions).all(), name


# --- clay-01: overlap_faces and a non-finite uv ------------------------------------


@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
def test_overlap_faces_refuses_a_non_finite_uv_before_any_cell_arithmetic(bad: float) -> None:
    mesh = bp.box()
    uv = np.array(mesh.uv, copy=True)
    uv[3, 0] = bad
    mesh = replace(mesh, uv=uv)
    # ``invalid="raise"`` turns the NaN->int64 cast the old arithmetic made into a
    # loud FloatingPointError instead of the unbounded loop that followed it.
    with np.errstate(invalid="raise"), pytest.raises(OpError, match="finite"):
        uvtools.overlap_faces(mesh)


# --- clay-15: transform_islands / pack_islands refuse non-finite -------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scale": float("nan")},
        {"scale": float("inf")},
        {"scale": 1e300},
        {"rotate_deg": float("nan")},
        {"translate": (1e39, 0.0)},
        {"translate": (0.0, float("nan"))},
    ],
)
def test_transform_islands_refuses_a_factor_that_overflows_float32(kwargs: dict) -> None:
    mesh = _two_island_mesh()
    with pytest.raises(OpError):
        uvtools.transform_islands(mesh, [0, 1], **kwargs)


def test_pack_islands_refuses_a_non_finite_margin_and_a_non_finite_uv() -> None:
    with pytest.raises(OpError):
        uvtools.pack_islands(_two_island_mesh(), margin=float("nan"))
    uv = np.array(_two_island_mesh().uv, copy=True)
    uv[0, 0] = np.inf
    with pytest.raises(OpError):
        uvtools.pack_islands(replace(_two_island_mesh(), uv=uv))


def test_the_pane_doors_toast_a_non_finite_factor_and_push_no_step() -> None:
    doc, obj = _doc_with(_two_island_mesh())
    ctx = _Ctx()
    before = len(doc.history)
    assert clay_uv.apply_scale(doc, obj.uid, [0, 1], 1e300, ctx=ctx) is False
    assert clay_uv.apply_rotate(doc, obj.uid, [0, 1], float("nan"), ctx=ctx) is False
    assert clay_uv.apply_translate(doc, obj.uid, [0, 1], (1e39, 0.0), ctx=ctx) is False
    assert len(ctx.toasts) == 3 and len(doc.history) == before
    assert np.isfinite(doc.by_uid(obj.uid).mesh.uv).all()


def test_apply_scale_reason_rejects_non_finite_input() -> None:
    for bad in (float("nan"), float("inf"), float("-inf")):
        why = clay_uv.apply_scale_reason(saving=False, live=False, selected=1, pending_scale=bad)
        assert why and "finite" in why
    assert clay_uv.apply_rotate_reason(
        saving=False, live=False, selected=1, pending_rotate=float("nan")
    )


# --- clay-16: the pane doors past the island ceiling -------------------------------


def test_apply_rotate_with_more_islands_than_the_ceiling_toasts_instead_of_raising() -> None:
    mesh = _strip_mesh(uvtools.MAX_UV_ISLANDS + 100)
    doc, obj = _doc_with(mesh)
    ids = uvtools.islands(mesh)
    every = np.unique(ids).tolist()
    ctx = _Ctx()
    before = len(doc.history)
    assert clay_uv.apply_rotate(doc, obj.uid, every, 10.0, ctx=ctx) is False
    assert clay_uv.apply_scale(doc, obj.uid, every, 0.5, ctx=ctx) is False
    assert clay_uv.apply_translate(doc, obj.uid, every, (0.1, 0.0), ctx=ctx) is False
    assert clay_uv.apply_pack(doc, obj.uid, ctx=ctx) is False
    assert len(ctx.toasts) == 4 and all(level == "error" for _t, level in ctx.toasts)
    assert len(doc.history) == before
    with pytest.raises(OpError):  # a headless caller with no ctx still learns of it
        clay_uv.apply_rotate(doc, obj.uid, every, 10.0)


def test_a_live_rotate_past_the_island_ceiling_cancels_the_gesture_and_toasts_once() -> None:
    mesh = _strip_mesh(uvtools.MAX_UV_ISLANDS + 100)
    doc, obj = _doc_with(mesh)
    ids = uvtools.islands(mesh)
    every = frozenset(int(i) for i in np.unique(ids))
    view_state = clay_uv.UvPaneState(selected_islands=every)
    assert clay_uv.begin_live_transform(doc, view_state, "rotate", mesh, ids, (0.1, 0.1))
    ctx = _Ctx()
    assert clay_uv.update_live_transform(doc, obj.uid, view_state, (0.3, 0.4), ctx=ctx) is False
    assert view_state.drag_mode == "", "the refused gesture is over"
    assert len(ctx.toasts) == 1
    # The next frame has no live gesture to refuse again.
    assert clay_uv.update_live_transform(doc, obj.uid, view_state, (0.5, 0.5), ctx=ctx) is False
    assert len(ctx.toasts) == 1
    assert doc.by_uid(obj.uid).mesh is mesh


# --- clay-48: a pull waits for a save -------------------------------------------------


class _Inker:
    def __init__(self) -> None:
        self.docs: list[Any] = []

    def activate(self, uid: str) -> None:
        pass


class _AppCtx:
    def __init__(self) -> None:
        self.state = AppState()
        self.state.clay = ClayState()
        self.state.inker = _Inker()
        self.toasts: list[Any] = []

    def toast(self, text: str, level: str = "info", **_kw: Any) -> None:
        self.toasts.append((text, level))


def test_pull_all_waits_while_the_tab_is_saving() -> None:
    ctx = _AppCtx()
    doc = bd.ClayDoc()
    assert doc.add_texture(0, 32)
    tab = ClayTab(doc=doc, title="Crate")
    ctx.state.clay.add(tab)
    px = np.zeros((4, 4, 4), np.uint8)
    px[..., 3] = 255
    inker = SimpleNamespace(uid="id1", title="T", doc=Document.from_pixels(px, name="T"))
    ctx.state.inker.docs.append(inker)
    link = tl.InkerLink(0, doc.materials[0], inker.uid, inker.doc.history.head)
    tab.inker_links.append(link)
    assert inker.doc.fill((0, 0), (255, 0, 77, 255), thresh=0)
    before = len(doc.history)

    tab.saving = True
    tl.pull_all(ctx)
    assert len(doc.history) == before, "no material is written under a running save"
    assert link.pulled_rev != inker.doc.history.head, "the link still owes the picture"

    tab.saving = False
    tl.pull_all(ctx)
    assert len(doc.history) == before + 1, "it lands on the first frame after the save"
    assert link.pulled_rev == inker.doc.history.head


# --- clay-57: leftover text ----------------------------------------------------------


def test_pack_islands_refusal_does_not_suggest_a_selection() -> None:
    mesh = _strip_mesh(uvtools.MAX_UV_ISLANDS + 1)
    with pytest.raises(OpError) as caught:
        uvtools.pack_islands(mesh)
    message = str(caught.value)
    assert "selection" not in message, "Pack always takes the whole object"
    assert "Separate" in message, "and the sentence names something the user can do"
    assert "LSCM" not in (uvtools.transform_islands.__doc__ or "")


# --- clay-77: Shift+R over the UV canvas ---------------------------------------------


class _IoWith:
    """The real imgui io, with the modifier reads replaced."""

    def __init__(self, real: Any, **mods: bool) -> None:
        self._real = real
        self._mods = {"key_ctrl": False, "key_alt": False, "key_shift": False, **mods}

    def __getattr__(self, name: str) -> Any:
        if name in self._mods:
            return self._mods[name]
        return getattr(self._real, name)


def _canvas_frame(ui, tab, doc, obj, view_state) -> None:
    ui.new_frame()
    ui.begin("##host")
    try:
        clay_uv._canvas(ctx=None, tab=tab, doc=doc, obj=obj, view_state=view_state)
    finally:
        ui.end()
        ui.end_frame()


def _press_r(ui, monkeypatch, **mods: bool) -> str:
    doc, obj = _doc_with(_two_island_mesh())
    view_state = clay_uv.UvPaneState(for_uid=obj.uid, selected_islands=frozenset({0}))
    tab = SimpleNamespace(doc=doc, uv_view=view_state, saving=False, uid="bd1")
    real_get_io = ui.get_io
    monkeypatch.setattr(ui, "is_item_hovered", lambda *a, **k: True)
    monkeypatch.setattr(ui, "is_key_pressed", lambda key, *a, **k: key == ui.Key.r)
    monkeypatch.setattr(ui, "get_io", lambda: _IoWith(real_get_io(), **mods))
    _canvas_frame(ui, tab, doc, obj, view_state)
    return view_state.drag_mode


def test_shift_r_over_the_uv_canvas_does_not_arm_a_live_rotate(ui, monkeypatch) -> None:
    assert _press_r(ui, monkeypatch) == "rotate", "a bare R still arms (the control)"
    assert _press_r(ui, monkeypatch, key_shift=True) == "", "Shift+R is Repeat Last, not a rotate"


def test_shift_ctrl_and_alt_r_over_the_uv_canvas_arm_nothing(ui, monkeypatch) -> None:
    for mod in ("key_shift", "key_ctrl", "key_alt"):
        assert _press_r(ui, monkeypatch, **{mod: True}) == "", mod


# --- clay-78: the click hit test -------------------------------------------------------


def _grid_uv_mesh(n: int) -> bm.Mesh:
    """n x n quads, each with its own uv cell inside the unit square."""
    positions, faces, uv = [], [], []
    for j in range(n):
        for i in range(n):
            base = len(positions)
            x, y = i * 2.0, j * 2.0
            positions += [(x, y, 0), (x + 1, y, 0), (x + 1, y + 1, 0), (x, y + 1, 0)]
            faces.append([base, base + 1, base + 2, base + 3])
            u0, v0, u1, v1 = i / n, j / n, (i + 0.8) / n, (j + 0.8) / n
            uv.append([[u0, v0], [u1, v0], [u1, v1], [u0, v1]])
    return bm.from_faces(positions, faces, uv=uv)


def test_a_uv_canvas_click_on_empty_space_is_not_linear_python_in_the_face_count(
    monkeypatch,
) -> None:
    mesh = _grid_uv_mesh(64)  # 4,096 faces
    ids = uvtools.islands(mesh)
    calls: list[int] = []
    real = clay_uv._point_in_face
    monkeypatch.setattr(
        clay_uv, "_point_in_face", lambda *a, **k: calls.append(1) or real(*a, **k)
    )
    # Past the square, and in the 0.2-cell gap between two faces.
    assert clay_uv.islands_in_rect(mesh, ids, (5.0, 5.0, 5.0, 5.0)) == set()
    gap = (0.9 / 64, 0.5 / 64)
    assert clay_uv.islands_in_rect(mesh, ids, (*gap, *gap)) == set()
    assert len(calls) <= 1, "a miss may not run the per-face polygon loop over every face"
    # A hit in a face's middle still finds exactly it, testing only its neighbours' boxes.
    calls.clear()
    middle = (10.4 / 64, 20.4 / 64)
    hit = clay_uv.islands_in_rect(mesh, ids, (*middle, *middle))
    assert hit == {int(ids[20 * 64 + 10])}
    assert len(calls) <= 2


def test_the_release_of_a_click_reuses_the_press_result(ui, monkeypatch) -> None:
    doc, obj = _doc_with(_two_island_mesh())
    view_state = clay_uv.UvPaneState(for_uid=obj.uid)
    tab = SimpleNamespace(doc=doc, uv_view=view_state, saving=False, uid="bd1")
    degenerate: list[tuple] = []
    real = clay_uv.islands_in_rect

    def counting(mesh, ids, rect):
        if rect[0] == rect[2] and rect[1] == rect[3]:
            degenerate.append(rect)
        return real(mesh, ids, rect)

    monkeypatch.setattr(clay_uv, "islands_in_rect", counting)
    state = {"activated": True, "deactivated": False}
    monkeypatch.setattr(ui, "is_item_hovered", lambda *a, **k: True)
    monkeypatch.setattr(ui, "is_item_activated", lambda: state["activated"])
    monkeypatch.setattr(ui, "is_item_deactivated", lambda: state["deactivated"])
    _canvas_frame(ui, tab, doc, obj, view_state)  # the press
    assert len(degenerate) == 1 and view_state.press_hit is not None
    state.update(activated=False, deactivated=True)
    _canvas_frame(ui, tab, doc, obj, view_state)  # the release
    assert len(degenerate) == 1, "the release did not ask the same question again"
    assert view_state.press_hit is None


# --- clay-79: the identity value ---------------------------------------------------------


def test_uv_apply_is_greyed_with_a_reason_at_the_identity_value(ui, monkeypatch) -> None:
    assert clay_uv.apply_rotate_reason(saving=False, live=False, selected=1, pending_rotate=0.0)
    assert not clay_uv.apply_rotate_reason(
        saving=False, live=False, selected=1, pending_rotate=45.0
    )
    assert clay_uv.apply_scale_reason(saving=False, live=False, selected=1, pending_scale=1.0)
    assert not clay_uv.apply_scale_reason(
        saving=False, live=False, selected=1, pending_scale=2.0
    )

    doc, obj = _doc_with(_two_island_mesh())
    view_state = clay_uv.UvPaneState(selected_islands=frozenset({0, 1}))
    seen: dict[str, tuple[bool, str]] = {}

    def fake(label, enabled, *_a, reason="", **_k):
        seen[label] = (bool(enabled), reason)
        return False

    monkeypatch.setattr(clay_uv.widgets, "disabled_button", fake)
    tab = SimpleNamespace(doc=doc, uv_view=view_state, saving=False, uid="bd1")
    ui.new_frame()
    ui.begin("##host")
    try:
        clay_uv._toolbar(ctx=None, tab=tab, doc=doc, obj=obj, view_state=view_state)
    finally:
        ui.end()
        ui.end_frame()
    for label in ("Apply##uvrotate", "Apply##uvscale"):
        enabled, reason = seen[label]
        assert not enabled and reason, f"{label} is greyed at the identity value with a reason"


def test_the_uv_pane_source_still_cites_no_plan_file() -> None:
    assert "TODO.md" not in inspect.getsource(clay_uv)
