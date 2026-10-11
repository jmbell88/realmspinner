"""The 2026-10-10 Clay audit's preview-scratch and import findings: clay-12,
16, 17, 38, 39, 40 and 41."""

from __future__ import annotations

import gc

import numpy as np
import pytest

from realmspinner.kernels.geom3d import glbio
from realmspinner.kernels.mesh import curves, glbimport, meshimport, scratch
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as bp

# --- clay-12 -----------------------------------------------------------------


def test_transplant_keeps_a_material_the_preview_added_and_does_not_report_it_as_kept():
    doc = bd.ClayDoc(objects=[bd.Obj(uid=bd.new_uid(), name="a", mesh=bp.box())])
    uid = doc.objects[0].uid
    # An added slot nothing uses yet: the old trailing-removal loop dropped it.
    sc = scratch.clone(doc)
    sc.add_material()
    result = scratch.transplant(doc, sc, scratch.diff(doc, sc))
    assert len(doc.materials) == len(sc.materials)
    assert not getattr(result, "kept_materials", [])

    # An added slot a face uses: the loop tried it, ``remove_material`` refused
    # it, and ``kept_materials`` named it as one the document would not drop.
    doc = bd.ClayDoc(objects=[bd.Obj(uid=bd.new_uid(), name="a", mesh=bp.box())])
    uid = doc.objects[0].uid
    sc = scratch.clone(doc)
    index = sc.add_material()
    sc.repaint_object(uid, index)
    result = scratch.transplant(doc, sc, scratch.diff(doc, sc))
    assert len(doc.materials) == len(sc.materials)
    assert int((doc.by_uid(uid).mesh.material == index).sum()) > 0
    assert result.kept_materials == []


# --- clay-38 -----------------------------------------------------------------


def test_transplant_does_not_apply_scratch_params_when_the_mesh_was_refused(monkeypatch):
    obj = bd.Obj(
        uid=bd.new_uid(), name="a", mesh=bp.box(), generator="box", params={"size": [1.0, 1.0, 1.0]}
    )
    doc = bd.ClayDoc(objects=[obj])
    sc = scratch.clone(doc)
    sc.set_mesh(obj.uid, bp.box((2.0, 2.0, 2.0)), keep_generator=True)
    sc.set_props(obj.uid, params={"size": [2.0, 2.0, 2.0]})
    d = scratch.diff(doc, sc)
    assert obj.uid in d.mesh_changed and "params" in d.props_changed[obj.uid]

    def refuse(*_a, **_k):
        raise el.OpError("past the triangle ceiling")

    monkeypatch.setattr(doc, "set_mesh", refuse)
    scratch.transplant(doc, sc, d)
    assert doc.by_uid(obj.uid).params == {"size": [1.0, 1.0, 1.0]}
    assert doc.by_uid(obj.uid).generator == "box"


# --- clay-39 -----------------------------------------------------------------


class _CountingDoc(bd.ClayDoc):
    by_uid_calls = 0

    def by_uid(self, uid):
        type(self).by_uid_calls += 1
        return super().by_uid(uid)


def test_scratch_diff_and_transplant_do_not_scan_the_object_list_per_object():
    box = bp.box()
    base = bd.ClayDoc(
        objects=[bd.Obj(uid=bd.new_uid(), name=f"o{i}", mesh=box) for i in range(60)]
    )
    sc = scratch.clone(base)
    sc.__class__ = _CountingDoc
    for o in sc.objects:
        o.translation = np.array([1.0, 0.0, 0.0])
    _CountingDoc.by_uid_calls = 0
    d = scratch.diff(base, sc)
    assert len(d.transform_changed) == 60
    scratch.transplant(base, sc, d)
    assert _CountingDoc.by_uid_calls == 0, (
        "diff/transplant looked a scratch object up by a linear scan, once per object"
    )
    assert all(float(o.translation[0]) == 1.0 for o in base.objects)


# --- clay-40 -----------------------------------------------------------------


def test_a_preview_diff_does_not_match_a_new_document_allocated_at_the_old_ones_address():
    base = bd.ClayDoc(objects=[bd.Obj(uid=bd.new_uid(), name="a", mesh=bp.box())])
    sc = scratch.clone(base)
    sc.set_transform(sc.objects[0].uid, translation=np.array([1.0, 0.0, 0.0]))
    d = scratch.diff(base, sc)
    old = id(base)
    assert d.base_doc_id == old
    assert d.is_base(base)

    del base
    gc.collect()
    # Whatever lives at that address now, the preview no longer vouches for it.
    reused = []
    for _ in range(2000):
        fresh = bd.ClayDoc()
        reused.append(fresh)
        if id(fresh) == old:
            break
    for fresh in reused:
        assert id(fresh) != d.base_doc_id
        assert not d.is_base(fresh)
    assert isinstance(d.base_doc_id, int)


# --- clay-41 -----------------------------------------------------------------


def test_normalise_handles_returns_empty_for_a_row_that_is_a_dict():
    row = {"in": [0, 0], "out": [1, 1]}
    assert curves.normalise_handles([row, row], 2, 2) == []
    assert curves.normalise_handles([[[0, 0], [1, 1]], {"x": 1}], 2, 2) == []
    assert curves.normalise_handles([[[0, 0], [1, 1]], [[0, 0], [1, 1]]], 2, 2) != []


# --- clay-17 -----------------------------------------------------------------


class _CountingList(list):
    iterations = 0

    def __iter__(self):
        type(self).iterations += 1
        return super().__iter__()


def _shared_mesh_glb(n_nodes: int, primitives: list) -> dict:
    return {
        "asset": {"version": "2.0"},
        "scenes": [{"nodes": list(range(n_nodes))}],
        "nodes": [{"mesh": 0} for _ in range(n_nodes)],
        "meshes": [{"primitives": primitives}],
    }


def test_declared_counts_does_not_walk_a_shared_mesh_of_null_primitives_once_per_node(
    monkeypatch,
):
    n_nodes = 40
    doc = _shared_mesh_glb(n_nodes, _CountingList([None] * 200))
    monkeypatch.setattr(glbio, "split_glb", lambda _data: (b"", doc, b""))
    _CountingList.iterations = 0
    assert glbimport._declared_counts(b"") == (0, 0, 0)
    assert _CountingList.iterations <= 1, (
        f"the primitive list was walked {_CountingList.iterations} times for {n_nodes} nodes"
    )


def test_declared_counts_scales_a_shared_meshs_primitives_by_the_nodes_naming_it(monkeypatch):
    prim = {"attributes": {"POSITION": 0}}
    doc = _shared_mesh_glb(3, [prim, prim])
    doc["accessors"] = [{"count": 6}]
    monkeypatch.setattr(glbio, "split_glb", lambda _data: (b"", doc, b""))
    tris, objects, vertices = glbimport._declared_counts(b"")
    assert (tris, objects, vertices) == (3 * 2 * 2, 6, 3 * 2 * 6)


def test_declared_counts_total_stays_at_the_ceiling_not_nodes_times_primitives(monkeypatch):
    prim = {"attributes": {"POSITION": 0}}
    doc = _shared_mesh_glb(2000, [prim] * 50)
    doc["accessors"] = [{"count": 3}]
    monkeypatch.setattr(glbio, "split_glb", lambda _data: (b"", doc, b""))
    _tris, objects, _vertices = glbimport._declared_counts(b"")
    assert glbimport.MAX_OBJECTS < objects < 1_000_000


# --- clay-16 -----------------------------------------------------------------


def _import(geom, kind: str) -> bm.Mesh:
    data = geom.export(file_type=kind)
    return meshimport.import_file(data, f".{kind}").objects[0].mesh


@pytest.mark.parametrize("kind", ["stl", "ply"])
def test_a_hard_edged_stl_cube_imports_flat_and_a_sphere_imports_smooth(kind):
    trimesh = pytest.importorskip("trimesh")
    box = _import(trimesh.creation.box(extents=(1, 1, 1)), kind)
    assert not box.smooth.any(), f"a unit box as {kind} imported smooth: {box.smooth.tolist()}"
    sphere = _import(trimesh.creation.icosphere(subdivisions=3), kind)
    assert sphere.smooth.all()
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    hinge = trimesh.Trimesh(v, [[0, 1, 2], [0, 3, 1]], process=False)
    assert not _import(hinge, kind).smooth.any()

