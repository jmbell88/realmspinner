"""The 2026-10-03 audit's Medium findings clay-39 .. clay-44 (Clay's mesh model,
selection, drag and picking). Each test's name is the claim."""

from __future__ import annotations

import gc

import numpy as np

from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import selection


def _obj(name: str, **kw) -> bd.Obj:
    return bd.Obj(uid=bd.new_uid(), name=name, mesh=bp.box(), **kw)


# --- clay-39 ----------------------------------------------------------------


def _moved(mesh: bm.Mesh, which: np.ndarray, delta: float) -> np.ndarray:
    positions = np.array(mesh.positions, dtype="f4")
    positions[which] = positions[which] + np.float32(delta)
    return positions


def test_a_second_drag_on_an_unchanged_mesh_does_not_reuse_the_first_drags_normals() -> None:
    mesh = bp.uv_sphere(segments=16, rings=12)
    first = np.array([3, 4, 5, 20, 21], dtype="i8")
    second = np.array([90, 91, 92], dtype="i8")
    materials: list = []

    # Drag one: two frames, so the second is the incremental one and the stash
    # ends up holding normals for a state with ``first`` displaced.
    bd.preview_primitives(mesh, _moved(mesh, first, 0.2), materials, moved=first)
    bd.preview_primitives(mesh, _moved(mesh, first, 0.3), materials, moved=first)

    # Esc, or a release where it started: the mesh object is unchanged. Drag two
    # grabs a different vertex set on the very same object.
    positions = _moved(mesh, second, 0.25)
    hinted = bd.preview_primitives(mesh, positions, materials, moved=second)
    plain = bd.preview_primitives(mesh, positions, materials)

    for a, b in zip(hinted, plain, strict=True):
        assert np.array_equal(a.normals, b.normals), "stale normals from the first drag"


def test_a_repeated_drag_of_the_same_vertex_set_still_matches_the_full_pass() -> None:
    mesh = bp.uv_sphere(segments=16, rings=12)
    verts = np.array([3, 4, 5], dtype="i8")
    materials: list = []
    for delta in (0.1, 0.2, 0.3):
        positions = _moved(mesh, verts, delta)
        hinted = bd.preview_primitives(mesh, positions, materials, moved=verts.copy())
        plain = bd.preview_primitives(mesh, positions, materials)
        for a, b in zip(hinted, plain, strict=True):
            assert np.array_equal(a.normals, b.normals)


# --- clay-40 ----------------------------------------------------------------


def test_typing_zero_scale_on_one_axis_keeps_the_objects_rotation_unit_and_unchanged() -> None:
    quat = m3.quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), np.radians(45.0))
    for scale in ((0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0)):
        world = m3.compose(np.array([1.0, 2.0, 3.0]), quat, np.array(scale))
        _t, rot, s = m3.decompose(world)
        assert abs(float(np.linalg.norm(rot)) - 1.0) < 1e-9, scale
        assert abs(abs(float(np.dot(rot, quat))) - 1.0) < 1e-9, scale
        assert np.allclose(s, scale)


def test_a_flattened_object_round_trips_through_local_from_world_with_its_rotation() -> None:
    doc = bd.ClayDoc()
    quat = m3.quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), np.radians(45.0))
    obj = doc.add_object(_obj("A"))
    world = m3.compose(np.array([0.0, 0.0, 0.0]), quat, np.array([0.0, 1.0, 1.0]))
    _t, rotation, scale = doc.local_from_world(obj.uid, world)
    assert abs(float(np.linalg.norm(rotation)) - 1.0) < 1e-9
    assert abs(abs(float(np.dot(rotation, quat))) - 1.0) < 1e-9
    assert np.allclose(scale, (0.0, 1.0, 1.0))


# --- clay-41 ----------------------------------------------------------------


def test_delete_selected_leaves_a_hidden_selected_object_untouched() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    b = doc.add_object(_obj("B"))
    doc.select([a.uid, b.uid])
    doc.set_props(b.uid, visible=False)
    doc.select([a.uid, b.uid])  # hiding may or may not drop it; select both either way

    selection.delete_selected(doc)

    assert [o.uid for o in doc.objects] == [b.uid], "only the visible one was deleted"


def test_delete_selected_in_face_mode_leaves_a_hidden_selected_object_untouched() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    b = doc.add_object(_obj("B"))
    doc.set_props(b.uid, visible=False)
    doc.set_element_mode("face")
    doc.set_element_sel(a.uid, el.ElementSel(faces=[0]))
    doc.set_element_sel(b.uid, el.ElementSel(faces=[0]))
    b_mesh = b.mesh

    selection.delete_selected(doc)

    assert doc.by_uid(b.uid).mesh is b_mesh, "the hidden object's faces were deleted"
    assert len(doc.by_uid(a.uid).mesh.starts) - 1 == 5


def test_duplicate_selected_skips_a_hidden_selected_object() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    b = doc.add_object(_obj("B"))
    doc.set_props(b.uid, visible=False)
    doc.select([a.uid, b.uid])

    made = selection.duplicate_selected(doc)

    assert len(made) == 1
    assert len(doc.objects) == 3


# --- clay-42 ----------------------------------------------------------------


def test_select_all_in_an_element_mode_skips_collider_objects() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    c = doc.add_object(_obj("C", role="collider", collider_kind="box"))
    doc.set_element_mode("face")

    selection.select_all(doc)
    assert not el.is_empty(doc.element_sel_of(a.uid))
    assert el.is_empty(doc.element_sel_of(c.uid)), "Ctrl+A reached a collider's faces"

    selection.invert(doc)
    assert el.is_empty(doc.element_sel_of(c.uid)), "invert reached a collider's faces"


# --- clay-43 ----------------------------------------------------------------


def test_render_arrays_does_not_pin_its_throwaway_layout_in_the_raw_normal_cache() -> None:
    mesh = bp.uv_sphere(segments=16, rings=12)
    gc.collect()
    before = len(bm._RAW_CACHE)
    for _ in range(4):
        bm.render_arrays(mesh)
    gc.collect()
    assert len(bm._RAW_CACHE) == before


# --- clay-44 ----------------------------------------------------------------


def _strip() -> bm.Mesh:
    """Two quads side by side: verts 0-2 along the bottom, 3-5 along the top."""
    positions = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0), (1, 1, 0), (2, 1, 0)]
    return bm.from_faces(positions, [[0, 1, 4, 3], [1, 2, 5, 4]])


def test_converting_two_of_a_triangles_edges_selects_no_face() -> None:
    tri = bm.from_faces([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [[0, 1, 2]])
    two = el.ElementSel(edges=[[0, 1], [1, 2]])
    assert len(el.convert(tri, two, "face").faces) == 0
    three = el.ElementSel(edges=[[0, 1], [1, 2], [2, 0]])
    assert list(el.convert(tri, three, "face").faces) == [0]


def test_converting_two_opposite_edges_of_a_quad_selects_no_face() -> None:
    quad = _strip()
    opposite = el.ElementSel(edges=[[0, 1], [3, 4]])
    assert len(el.convert(quad, opposite, "face").faces) == 0
    all_four = el.ElementSel(edges=[[0, 1], [1, 4], [4, 3], [3, 0]])
    assert list(el.convert(quad, all_four, "face").faces) == [0]


def test_converting_a_full_vertex_set_still_selects_the_face() -> None:
    quad = _strip()
    assert list(el.convert(quad, el.ElementSel(verts=[0, 1, 4, 3]), "face").faces) == [0]
