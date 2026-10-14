"""Regressions for the 2026-10-14 audit's Mason mode findings (fixer F35).

Each test's name is the claim, and each failed against the unfixed code before
the fix in this same change. Finding ids are the audit's (``mason-NN``, with the
explorer record's ``mason-mode-NN`` in the docstring).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from _ui_context import imgui_context

from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.studio.modes.mason import assets as mason_assets
from realmspinner.studio.modes.mason import mode as mm
from realmspinner.studio.modes.mason.engine import document as md
from realmspinner.studio.modes.mason.engine import nodes as nd
from realmspinner.studio.modes.mason.engine import objout
from realmspinner.studio.modes.mason.engine import refs as mrefs
from realmspinner.studio.modes.mason.engine import scene as sc
from realmspinner.studio.modes.mason.ui import view as mason_view
from realmspinner.studio.modes.mason.ui.panes import outliner as mason_outliner
from realmspinner.studio.modes.mason.ui.panes import palette as mason_palette
from realmspinner.studio.modes.mason.ui.panes import props as mason_props
from realmspinner.studio.modes.mason.ui.panes import tools as mason_tools

INF = float("inf")


def _box_mesh(name: str = "Box") -> nd.MeshNode:
    return nd.MeshNode(uid=nd.new_uid(), name=name, ref=mrefs.primitive_ref("box", {}))


def _ctx(doc: md.MasonDoc, **extra: Any) -> Any:
    """A ctx with one open scene, recording its toasts and submitted tasks."""
    tab = SimpleNamespace(doc=doc, saving=False, title="t", uid="ms1")
    ctx = SimpleNamespace(
        state=SimpleNamespace(
            mason=SimpleNamespace(
                active=tab, place_kind="", place_prefab="", snap_ground=False, renaming=0
            )
        ),
        toasts=[],
        submitted=[],
        **extra,
    )
    ctx.toast = lambda *a, **k: ctx.toasts.append(a)
    ctx.submit = lambda key, fn, *a, **k: ctx.submitted.append((key, fn, k)) or True
    return ctx


# --- mason-11 (mason-mode-05) -----------------------------------------------------


def test_export_obj_over_the_vertex_ceiling_toasts_instead_of_raising(monkeypatch) -> None:
    doc = md.MasonDoc()
    doc.add_node(_box_mesh())
    ctx = _ctx(doc)
    monkeypatch.setattr(objout, "MAX_OBJ_VERTS", 1)  # stand-in for a scene past 1,000,000

    mm.export_obj(ctx, ctx.state.mason.active)  # must not raise

    assert len(ctx.toasts) == 1 and "OBJ vertices" in ctx.toasts[0][0]
    assert ctx.submitted == [], "a refused export must not start a task"


# --- mason-13 (mason-mode-08) -----------------------------------------------------


def test_place_refuses_with_a_toast_when_scene_plus_templates_would_pass_max_placed(
    monkeypatch,
) -> None:
    monkeypatch.setattr(sc, "MAX_PLACED", 10)
    doc = md.MasonDoc()
    template = nd.GroupNode(uid=nd.new_uid(), name="T")
    for i in range(4):
        template.children.append(_box_mesh(f"p{i}"))
    doc.define_prefab("Tpl", template)  # five template nodes count toward the ceiling
    for i in range(5):
        doc.add_node(_box_mesh(f"m{i}"))
    ctx = _ctx(doc)

    uid = mm.place_primitive(ctx, "box")  # must not raise ValueError

    assert uid is None
    assert len(ctx.toasts) == 1 and "limit" in ctx.toasts[0][0]
    assert len(doc.all_nodes()) == 5


def test_the_arrays_pre_check_counts_prefab_templates_too(monkeypatch) -> None:
    monkeypatch.setattr(sc, "MAX_PLACED", 10)
    doc = md.MasonDoc()
    template = nd.GroupNode(uid=nd.new_uid(), name="T")
    for i in range(4):
        template.children.append(_box_mesh(f"p{i}"))
    doc.define_prefab("Tpl", template)
    node = doc.add_node(_box_mesh())
    assert mason_tools._over_max_placed(doc, node, 5) == 11  # 1 + 5 template + 5 copies


# --- mason-16 (mason-mode-11) -----------------------------------------------------


def _library_ctx(tmp_path: Path, doc: md.MasonDoc) -> Any:
    return _ctx(
        doc,
        svc=SimpleNamespace(config=SimpleNamespace(job_dir=lambda j: tmp_path / j)),
        cache=SimpleNamespace(jobs=[]),
    )


def _box_prims() -> list[Any]:
    """A unit cube's primitives, from the one door that builds them."""
    probe = mason_assets.AssetSource(SimpleNamespace())
    ref = mrefs.primitive_ref("box", {})
    return probe.primitives(ref)


def _land(ctx: Any, key: str, tag: Any, prims: list[Any]) -> None:
    mm.on_task_done(ctx, SimpleNamespace(key=key, tag=tag, result=prims, error=None))


def test_drop_to_ground_applies_to_a_library_assets_first_placement_once_it_has_parsed(
    tmp_path,
) -> None:
    doc = md.MasonDoc()
    ctx = _library_ctx(tmp_path, doc)
    ctx.state.mason.snap_ground = True
    ctx.state.mason.place_kind = "job:0000000000ab"

    uid = mm.place_armed(ctx, np.array([2.0, 0.0, 3.0]))
    node = doc.node(uid)
    assert node.translation[1] == 0.0, "the parse has not landed: nothing to measure yet"
    assert [k for k, _f, _kw in ctx.submitted] == ["mason-asset:0000000000ab:model.glb"]

    key, _fn, kw = ctx.submitted[0]
    _land(ctx, key, kw["tag"], _box_prims())

    source = mason_assets.ensure(ctx)
    box = sc.world_bounds(doc, source, uids=[uid])
    assert box is not None
    assert float(box[0][1]) == pytest.approx(0.0), "the box must rest on the ground"
    assert node.translation[0] == pytest.approx(2.0)


def test_a_late_drop_leaves_a_node_the_user_has_moved_since_alone(tmp_path) -> None:
    doc = md.MasonDoc()
    ctx = _library_ctx(tmp_path, doc)
    ctx.state.mason.snap_ground = True
    ctx.state.mason.place_kind = "job:0000000000ab"
    uid = mm.place_armed(ctx, np.array([2.0, 0.0, 3.0]))
    node = doc.node(uid)
    doc.set_transform(uid, translation=np.array([9.0, 4.0, 9.0]), was=node.trs())

    key, _fn, kw = ctx.submitted[0]
    _land(ctx, key, kw["tag"], _box_prims())

    assert list(node.translation) == [9.0, 4.0, 9.0]


def test_place_dropped_mesh_reads_snap_ground(tmp_path) -> None:
    doc = md.MasonDoc()
    ctx = _library_ctx(tmp_path, doc)
    ctx.state.mason.snap_ground = True
    source = mason_assets.ensure(ctx)
    source._store(mrefs.ref_key(mrefs.LibraryRef(job_id="0000000000ab")), _box_prims())

    uid = mm.place_dropped_mesh(ctx, {"id": "0000000000ab", "name": "Crate"}, [1.0, 0.0, 1.0])

    box = sc.world_bounds(doc, source, uids=[uid])
    assert float(box[0][1]) == pytest.approx(0.0)


# --- mason-23 (mason-mode-13) -----------------------------------------------------


def test_the_mason_journal_encoder_defers_the_archive_to_the_write_task(monkeypatch) -> None:
    from realmspinner.studio.modes.mason.engine import serialize

    doc = md.MasonDoc()
    doc.add_node(_box_mesh())
    view = SimpleNamespace(
        **{name: 0.0 for name in serialize.VIEW_FIELDS}, target=(0.0, 0.0, 0.0)
    )
    tab = SimpleNamespace(doc=doc, view=view)
    calls: list[str] = []
    real = serialize.snapshot_bytes
    monkeypatch.setattr(
        serialize, "snapshot_bytes", lambda snap: calls.append("zip") or real(snap)
    )

    encoded = mm.JOURNAL.encode(tab)

    assert callable(encoded), "the frame thread must get a closure, not the archive"
    assert calls == [], "no zip/npy/PNG encode may run on the frame thread"
    data = encoded()
    assert calls == ["zip"]
    assert len(serialize.read_rscn(data).roots) == 1


# --- mason-24 (mason-mode-14) -----------------------------------------------------


def test_ungroup_of_nested_selected_groups_leaves_no_stale_uid_in_the_selection() -> None:
    doc = md.MasonDoc()
    outer = doc.add_node(nd.GroupNode(uid=nd.new_uid(), name="outer"))
    inner = doc.add_node(nd.GroupNode(uid=nd.new_uid(), name="inner"), parent_uid=outer.uid)
    leaf = doc.add_node(_box_mesh("leaf"), parent_uid=inner.uid)
    doc.select([outer.uid, inner.uid])
    ctx = _ctx(doc)

    mm.ungroup_selected(ctx)

    live = {n.uid for n in doc.all_nodes()}
    assert live == {leaf.uid}
    assert set(doc.selection) == {leaf.uid}


# --- mason-15 (mason-mode-10) -----------------------------------------------------


def test_the_rename_field_takes_keyboard_focus_on_the_frame_it_appears(monkeypatch) -> None:
    doc = md.MasonDoc()
    node = doc.add_node(nd.GroupNode(uid=nd.new_uid(), name="West wall"))
    state = SimpleNamespace(renaming=node.uid, outliner_anchor=0)
    ctx = SimpleNamespace()
    active: list[bool] = []
    real = mason_outliner.widgets.input_text

    with imgui_context(monkeypatch) as imgui:

        def spy(*a: Any, **k: Any) -> Any:
            out = real(*a, **k)
            active.append(bool(imgui.is_item_active()))
            return out

        monkeypatch.setattr(mason_outliner.widgets, "input_text", spy)
        for _ in range(4):
            imgui.new_frame()
            imgui.set_next_window_size((400.0, 300.0))
            imgui.begin("##host")
            mason_outliner._row(
                ctx, state, doc, node, None, 0, 0, filtered=False, saving=False
            )
            imgui.end()
            imgui.end_frame()

    assert any(active), f"the field was never focused: {active}"


def test_escape_ends_the_rename_without_committing(monkeypatch) -> None:
    doc = md.MasonDoc()
    node = doc.add_node(nd.GroupNode(uid=nd.new_uid(), name="West wall"))
    state = SimpleNamespace(renaming=node.uid, outliner_anchor=0)
    ctx = SimpleNamespace()

    with imgui_context(monkeypatch) as imgui:

        def frame() -> None:
            imgui.new_frame()
            imgui.set_next_window_size((400.0, 300.0))
            imgui.begin("##host")
            mason_outliner._row(
                ctx, state, doc, node, None, 0, 0, filtered=False, saving=False
            )
            imgui.end()
            imgui.end_frame()

        for _ in range(3):
            frame()
        assert state.renaming == node.uid
        io = imgui.get_io()
        io.add_key_event(imgui.Key.escape, True)
        frame()
        io.add_key_event(imgui.Key.escape, False)
        frame()

    assert state.renaming == 0
    assert node.name == "West wall"


# --- mason-17 (mason-mode-12) -----------------------------------------------------


class _JobsCache:
    def __init__(self, jobs: list[dict[str, Any]]) -> None:
        self.jobs = jobs
        self._generation = 1


def _library_jobs(count: int) -> list[dict[str, Any]]:
    return [
        {"id": f"job{i:05d}", "name": f"Model {i}", "stage": "model", "status": "done"}
        for i in range(count)
    ]


def test_assets_pane_draws_only_the_visible_library_rows(monkeypatch) -> None:
    ctx = SimpleNamespace(cache=_JobsCache(_library_jobs(500)), textures=None)
    state = SimpleNamespace(place_kind="", place_prefab="")
    drawn: list[str] = []

    with imgui_context(monkeypatch) as imgui:

        def thumb(_ctx: Any, job: Any, side: float) -> None:
            drawn.append(job["id"])
            imgui.dummy((side, side))  # the real thumbnail is an image of this size

        monkeypatch.setattr(mason_palette.thumbs, "job_thumb", thumb)
        imgui.new_frame()
        imgui.set_next_window_size((320.0, 500.0))
        imgui.begin("##host")
        mason_palette._library(ctx, state)
        imgui.end()
        imgui.end_frame()

    assert 0 < len(drawn) < 60, f"{len(drawn)} of 500 rows were submitted in one frame"


def test_mesh_jobs_is_memoised_on_the_jobs_cache_generation(monkeypatch) -> None:
    cache = _JobsCache(_library_jobs(50))
    ctx = SimpleNamespace(cache=cache)
    calls: list[str] = []
    real = mason_palette.library.can_drag_mesh
    monkeypatch.setattr(
        mason_palette.library, "can_drag_mesh", lambda job: calls.append(job["id"]) or real(job)
    )

    first = mason_palette._mesh_jobs(ctx)
    second = mason_palette._mesh_jobs(ctx)
    assert len(first) == 50 and second == first
    assert len(calls) == 50, "the second frame must not re-filter the library"

    cache._generation += 1
    mason_palette._mesh_jobs(ctx)
    assert len(calls) == 100


# --- mason-14 (mason-mode-09) -----------------------------------------------------


class _InfControls:
    """Every numeric field reports an edit of +inf, what imgui's InputFloat hands back
    for a typed ``1e999``."""

    @staticmethod
    def input_float(*_a: Any, **_k: Any) -> Any:
        return True, INF

    @staticmethod
    def input_float3(_label: str, v: Any, **_k: Any) -> Any:
        return True, [INF, v[1], v[2]]

    @staticmethod
    def input_float4(_label: str, v: Any, **_k: Any) -> Any:
        return True, list(v)

    @staticmethod
    def color_edit3(_label: str, v: Any, **_k: Any) -> Any:
        return False, v

    @staticmethod
    def color_edit4(_label: str, v: Any, **_k: Any) -> Any:
        return False, v

    @staticmethod
    def fold_undo(*_a: Any, **_k: Any) -> None:
        return None


class _NullWidgets:
    @staticmethod
    def toggle(*_a: Any, **_k: Any) -> Any:
        return False, False

    def __getattr__(self, _name: str) -> Any:
        return lambda *a, **k: None


def test_properties_pane_survives_a_non_finite_typed_value_without_raising(monkeypatch) -> None:
    monkeypatch.setattr(mason_props, "controls", _InfControls)
    monkeypatch.setattr(mason_props, "widgets", _NullWidgets())
    doc = md.MasonDoc()
    light = doc.add_node(nd.LightNode(uid=nd.new_uid(), name="L", kind="spot"))
    cam = doc.add_node(nd.CameraNode(uid=nd.new_uid(), name="C"))
    group = doc.add_node(nd.GroupNode(uid=nd.new_uid(), name="G"))
    ctx = _ctx(doc)

    mason_props._light_block(doc, light, ctx=ctx)
    mason_props._camera_block(doc, cam, ctx=ctx)
    mason_props._transform(doc, group, ctx=ctx)

    for node in (light, cam, group):
        assert np.isfinite(node.translation).all()
    assert np.isfinite(light.intensity) and np.isfinite(light.range)
    assert np.isfinite(cam.yfov) and np.isfinite(cam.zfar)
    assert ctx.toasts, "a refused field must say so"


def test_properties_terrain_size_survives_a_non_finite_typed_value(monkeypatch) -> None:
    from realmspinner.studio.modes.mason.engine.terrain import Terrain

    monkeypatch.setattr(mason_props, "controls", _InfControls)
    monkeypatch.setattr(mason_props, "widgets", _NullWidgets())
    doc = md.MasonDoc()
    doc.set_terrain(Terrain(heights=np.zeros((9, 9), dtype="f4"), size_x=8.0, size_z=8.0,
                            material=_terrain_material()))
    ctx = _ctx(doc)

    mason_props._terrain_block(doc, ctx=ctx)

    assert doc.terrain.size_x == 8.0 and doc.terrain.size_z == 8.0


def _terrain_material() -> Any:
    from realmspinner.kernels.geom3d import gltf

    return gltf.Material(name="t")


# --- mason-03 (mason-mode-02) -----------------------------------------------------


@pytest.mark.parametrize("radial", [False, True])
def test_array_with_a_non_finite_offset_or_degrees_is_refused_and_attaches_nothing(
    radial, monkeypatch
) -> None:
    doc = md.MasonDoc()
    node = doc.add_node(_box_mesh())
    ctx = _ctx(doc)
    monkeypatch.setitem(mason_tools._PENDING, "count", 3)
    monkeypatch.setitem(mason_tools._PENDING, "offset", [INF, 0.0, 0.0])
    monkeypatch.setitem(mason_tools._PENDING, "degrees", INF)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mason_tools._build_array(ctx, doc, node, radial=radial)  # must not raise

    assert len(doc.roots) == 1, "nothing may attach"
    assert len(ctx.toasts) == 1
    assert "offset" in ctx.toasts[0][0] or "degrees" in ctx.toasts[0][0]


def test_add_nodes_refuses_a_non_finite_transform_it_could_not_reopen() -> None:
    doc = md.MasonDoc()
    node = _box_mesh()
    node.translation = np.array([INF, 0.0, 0.0])
    with pytest.raises(ValueError, match="finite"):
        doc.add_nodes([node])
    assert doc.roots == []


# --- mason-04 (mason-mode-03) -----------------------------------------------------


def test_a_non_finite_snap_grid_never_leaves_a_nan_translation_on_a_dragged_node() -> None:
    doc = md.MasonDoc()
    node = doc.add_node(_box_mesh())
    state = SimpleNamespace(
        snap=True, snap_translate=INF, snap_rotate=INF, snap_ground=False, tool="move",
        pivot="median",
    )
    view = mason_view.MasonView.__new__(mason_view.MasonView)
    view.app_ctx = SimpleNamespace(state=SimpleNamespace(mason=state))
    view._drag_start = {node.uid: tuple(np.array(v, copy=True) for v in node.trs())}
    view._drag_pivot = np.zeros(3)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        view._apply_drag(doc, None, delta=np.array([1.0, 0.0, 0.0]))
        view._apply_drag(doc, None, quat=m3.quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), 0.3))

    assert np.isfinite(node.translation).all() and np.isfinite(node.rotation).all()


def test_the_snap_fields_refuse_a_non_finite_typed_value(monkeypatch) -> None:
    monkeypatch.setattr(mason_tools, "controls", _InfControls)
    monkeypatch.setattr(mason_tools, "widgets", _NullWidgets())
    monkeypatch.setattr(mason_tools, "imgui", SimpleNamespace(
        begin_disabled=lambda *_a: None, end_disabled=lambda: None))
    state = SimpleNamespace(snap=True, snap_translate=0.25, snap_rotate=15.0, snap_ground=False)

    mason_tools._snap(None, state)

    assert state.snap_translate == 0.25 and state.snap_rotate == 15.0


# --- mason-05 (mason-mode-04) -----------------------------------------------------


def test_rotate_drag_of_a_child_of_a_rotated_group_turns_it_about_the_world_axis_the_gizmo_shows(
) -> None:
    doc = md.MasonDoc()
    group = doc.add_node(nd.GroupNode(uid=nd.new_uid(), name="G"))
    doc.set_transform(
        group.uid,
        rotation=m3.quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), np.radians(90)),
    )
    child = doc.add_node(nd.GroupNode(uid=nd.new_uid(), name="C"), parent_uid=group.uid)
    view = mason_view.MasonView.__new__(mason_view.MasonView)
    view.app_ctx = None
    view._drag_start = {child.uid: tuple(np.array(v, copy=True) for v in child.trs())}
    view._drag_pivot = sc.resolved_for(doc, child.uid).world[:3, 3].copy()
    turn = m3.quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), np.radians(30))
    before = sc.resolved_for(doc, child.uid).world[:3, :3].copy()

    view._apply_drag(doc, None, quat=turn)

    after = sc.resolved_for(doc, child.uid).world[:3, :3]
    applied = after @ np.linalg.inv(before)
    expected = m3.quat_to_mat4(turn)[:3, :3]
    assert np.allclose(applied, expected, atol=1e-6)
