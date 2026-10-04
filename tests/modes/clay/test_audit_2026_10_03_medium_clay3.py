"""The 2026-10-03 audit's Medium findings clay-31 .. clay-38 (Clay's mesh
import and export: glTF loader, GLB import, STL/PLY import, OBJ export). Each
test's name is the claim."""

from __future__ import annotations

import struct
import tracemalloc

import numpy as np
import pytest

from realmspinner.kernels.geom3d import gltf
from realmspinner.kernels.geom3d.glbio import rebuild_glb
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import glbimport, meshimport, objexport, objimport
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh.elements import OpError

_TRI = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="<f4")


def _glb(doc: dict, binary: bytes = b"") -> bytes:
    binary += b"\x00" * (-len(binary) % 4)
    chunk = struct.pack("<II", len(binary), 0x004E4942) + binary
    header = struct.pack("<III", 0x46546C67, 2, 0)
    return rebuild_glb(header, doc, chunk)


def _tri_doc(nodes: list[dict], scenes: list[dict], binary_len: int = 36) -> dict:
    """One triangle mesh (mesh 0) and whatever nodes/scenes the test wants."""
    return {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": scenes,
        "nodes": nodes,
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        "buffers": [{"byteLength": binary_len}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": binary_len}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}],
    }


# --- clay-31 ----------------------------------------------------------------


def _two_scene_doc() -> bytes:
    doc = _tri_doc(
        [
            {"name": "Placed", "mesh": 0, "translation": [5, 0, 0]},
            {"name": "Elsewhere", "mesh": 0, "translation": [9, 0, 0]},
            {"name": "Orphan", "mesh": 0, "translation": [-4, 0, 0]},
        ],
        [{"nodes": [0]}, {"nodes": [1]}],
    )
    return _glb(doc, _TRI.tobytes())


def test_glb_import_skips_nodes_outside_the_active_scene() -> None:
    out = glbimport.glb_to_claydoc(_two_scene_doc())
    assert [o.name for o in out.objects] == ["Placed"]
    assert np.allclose(out.objects[0].translation, [5, 0, 0])


def test_glb_import_budgets_only_the_nodes_the_active_scene_places(monkeypatch) -> None:
    """Three nodes name the mesh but only one is placed: a ceiling of one
    object must neither refuse it on the declared count nor on the real one."""
    monkeypatch.setattr(glbimport, "MAX_OBJECTS", 1)
    out = glbimport.glb_to_claydoc(_two_scene_doc())
    assert len(out.objects) == 1


# --- clay-32 ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "count"), [("SCALAR", 9), ("VEC2", 3), ("VEC4", 3)], ids=lambda v: str(v)
)
def test_a_position_accessor_that_is_not_vec3_is_refused_by_name(kind: str, count: int) -> None:
    ncomp = {"SCALAR": 1, "VEC2": 2, "VEC4": 4}[kind]
    binary = np.arange(count * ncomp, dtype="<f4").tobytes()
    doc = _tri_doc([{"mesh": 0}], [{"nodes": [0]}], len(binary))
    doc["accessors"][0].update(type=kind, count=count)
    data = _glb(doc, binary)
    with pytest.raises(ValueError, match="POSITION"):
        gltf.load(data)
    with pytest.raises(OpError, match="POSITION"):
        glbimport.glb_to_claydoc(data)


# --- clay-33 ----------------------------------------------------------------


def test_astype_charges_the_budget_before_it_allocates() -> None:
    class _Unallocated:
        """An array that only knows how big its conversion would be."""

        size = 1 << 28
        dtype = np.dtype("u1")
        ndim = 2

        def astype(self, _dtype):
            raise AssertionError("astype allocated before the budget was charged")

    reader = gltf._Reader({}, b"")
    with pytest.raises(ValueError, match="byte budget"):
        reader._astype(_Unallocated(), "f4")  # 2**28 * 4 B = 1 GiB > the 768 MiB ceiling


# --- clay-34 ----------------------------------------------------------------


def _mat4_reader(stride: int, count: int) -> tuple[gltf._Reader, dict]:
    span = stride * (count - 1) + 64
    buffer = bytes(span + 4)
    doc = {
        "bufferViews": [{"buffer": 0, "byteLength": span, "byteStride": stride}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": count, "type": "MAT4"}],
    }
    return gltf._Reader(doc, buffer), doc


def test_interleaved_accessor_peak_allocation_stays_within_what_it_charged() -> None:
    """The gather built a (count, item) int64 index (8 bytes per output byte)
    that ``_charge`` never saw: measured 4.4x the charged total."""
    reader, _ = _mat4_reader(stride=68, count=20_000)
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        out = reader._decode_accessor(0)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert out.shape == (20_000, 16)
    assert peak <= reader._spent, f"peaked at {peak:,} B against {reader._spent:,} B charged"


def test_interleaved_gather_still_reads_the_right_rows() -> None:
    rows = np.arange(5 * 3, dtype="<f4").reshape(5, 3)
    stride = 20  # 12 bytes of element, 8 of padding
    buffer = bytearray(stride * 5)
    for i, row in enumerate(rows):
        buffer[i * stride : i * stride + 12] = row.tobytes()
    doc = {
        "bufferViews": [{"buffer": 0, "byteLength": len(buffer), "byteStride": stride}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 5, "type": "VEC3"}],
    }
    out = gltf._Reader(doc, bytes(buffer))._decode_accessor(0)
    assert np.array_equal(out, rows)


def test_a_bytestride_below_the_element_size_is_refused() -> None:
    """Stride 8 on a 12-byte VEC3 loaded as overlapping rows; stride 4 on a
    MAT4 amplified a 4-byte-per-element buffer into 64-byte elements."""
    binary = _TRI.tobytes()
    doc = _tri_doc([{"mesh": 0}], [{"nodes": [0]}], len(binary))
    doc["bufferViews"][0]["byteStride"] = 8
    with pytest.raises(ValueError, match="byteStride"):
        gltf.load(_glb(doc, binary))
    reader, _ = _mat4_reader(stride=64, count=4)
    reader.gltf["bufferViews"][0]["byteStride"] = 4
    with pytest.raises(ValueError, match="byteStride"):
        reader._decode_accessor(0)


# --- clay-35 ----------------------------------------------------------------


def test_node_light_reference_check_is_not_quadratic_in_node_count(monkeypatch) -> None:
    nodes = 60
    doc = _tri_doc([{"extensions": {"KHR_lights_punctual": {"light": 0}}}] * nodes, [])
    doc["scenes"] = [{"nodes": [0]}]
    doc["extensions"] = {"KHR_lights_punctual": {"lights": [{"type": "point"}] * 3}}
    doc["extensionsUsed"] = ["KHR_lights_punctual"]
    data = _glb(doc, _TRI.tobytes())

    calls = 0
    real = gltf._punctual

    def counting(g):
        nonlocal calls
        calls += 1
        return real(g)

    monkeypatch.setattr(gltf, "_punctual", counting)
    model = gltf.load(data)
    assert len(model.nodes) == nodes and model.nodes[0].light == 0
    # Once for the declared-count ceiling; pre-fix it was that plus one per node.
    assert calls <= 2, f"_punctual rebuilt its light list {calls} times for {nodes} nodes"


# --- clay-36 ----------------------------------------------------------------


def _instanced_big_vertex_stream(instances: int, vertices: int) -> bytes:
    """One triangle that indexes three vertices of a ``vertices``-long bufferless
    POSITION accessor, placed by ``instances`` nodes."""
    indices = np.array([0, 1, 2], dtype="<u2").tobytes() + b"\x00\x00"
    doc = _tri_doc([{"mesh": 0}] * instances, [{"nodes": list(range(instances))}], len(indices))
    doc["meshes"][0]["primitives"][0]["indices"] = 1
    doc["accessors"] = [
        {"componentType": 5126, "count": vertices, "type": "VEC3"},  # no bufferView: zeros
        {"bufferView": 0, "componentType": 5123, "count": 3, "type": "SCALAR"},
    ]
    return _glb(doc, indices)


def test_instanced_import_budgets_vertices_not_just_triangles(monkeypatch) -> None:
    monkeypatch.setattr(glbimport, "MAX_VERTICES", 3_000)
    data = _instanced_big_vertex_stream(instances=4, vertices=1_000)  # 4,000 references
    # One triangle each: the triangle and object budgets are nowhere near.
    with pytest.raises(OpError, match="vertices"):
        glbimport.glb_to_claydoc(data)
    # And refused from the JSON alone, before any accessor is decoded.
    assert glbimport._declared_vertices(data) > 3_000


def test_instanced_import_merges_each_primitive_once_not_once_per_node(monkeypatch) -> None:
    data = _instanced_big_vertex_stream(instances=8, vertices=3)
    calls = 0
    real = glbimport.topo.rebuild

    def counting(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(glbimport.topo, "rebuild", counting)
    out = glbimport.glb_to_claydoc(data)
    assert len(out.objects) == 8
    assert calls == 1, f"the vertex merge ran {calls} times for one primitive"


# --- clay-37 ----------------------------------------------------------------


def test_glb_import_refuses_non_finite_positions() -> None:
    bad = _TRI.copy()
    bad[1, 0] = np.nan
    doc = _tri_doc([{"mesh": 0}], [{"nodes": [0]}])
    with pytest.raises(OpError, match="non-finite"):
        glbimport.glb_to_claydoc(_glb(doc, bad.tobytes()))


def test_glb_import_refuses_a_non_finite_node_translation() -> None:
    doc = _tri_doc([{"mesh": 0, "translation": [float("nan"), 0, 0]}], [{"nodes": [0]}])
    with pytest.raises(OpError, match="non-finite"):
        glbimport.glb_to_claydoc(_glb(doc, _TRI.tobytes()))


def test_stl_import_refuses_non_finite_positions() -> None:
    facet = struct.pack("<12fH", 0, 0, 1, 0, 0, 0, 1, 0, 0, float("nan"), 1, 0, 0)
    data = bytes(80) + struct.pack("<I", 1) + facet
    with pytest.raises(OpError, match="non-finite"):
        meshimport.mesh_file_to_claydoc(data, ".stl")


# --- clay-38 ----------------------------------------------------------------


def test_obj_export_writes_a_name_with_a_newline_as_one_line() -> None:
    from realmspinner.kernels.geom3d import gltf as g

    doc = bd.ClayDoc()
    box = bp.box()
    material = g.Material(name="Steel\nmap_Kd evil.png\nnewmtl Evil")
    doc.objects.append(bd.Obj(uid=bd.new_uid(), name="Wall\nv 9 9 9\nf 1 1 1\no Evil", mesh=box))
    doc.materials = [material]
    obj_text, mtl_text = objexport.claydoc_to_obj(doc)

    lines = obj_text.splitlines()
    assert sum(1 for line in lines if line.startswith("o ")) == 1
    assert sum(1 for line in lines if line.startswith("v ")) == len(box.positions)
    assert "v 9 9 9" not in lines
    assert sum(1 for line in mtl_text.splitlines() if line.startswith("newmtl")) == 1
    assert "map_Kd evil.png" not in mtl_text.splitlines()
    # And the exported text is what Clay's own importer reads back as one object.
    back = objimport.obj_to_claydoc(obj_text, mtl=mtl_text)
    assert len(back.objects) == 1
