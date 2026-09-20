"""STL and PLY into a Clay document, and the ``import_file`` dispatcher."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from realmspinner.kernels.mesh import adjacency, meshimport
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh.elements import OpError


def _cube_bytes(file_type: str) -> bytes:
    """A closed, manifold unit cube, exported through trimesh itself."""
    cube = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    data = cube.export(file_type=file_type)
    return data if isinstance(data, bytes) else data.encode("utf-8")


# --- STL ------------------------------------------------------------------------


def test_binary_stl_welds_the_triangle_soup_into_one_manifold_object() -> None:
    doc = meshimport.mesh_file_to_claydoc(_cube_bytes("stl"), ".stl", "Cube")
    assert len(doc.objects) == 1
    mesh = doc.objects[0].mesh
    bm.validate(mesh)
    assert len(mesh.positions) == 8, "a raw STL cube is 36 verts; merge_vertices welds to 8"
    report = adjacency.check_manifold(mesh)
    assert report.clean, report


def test_ascii_stl_welds_the_same_way() -> None:
    doc = meshimport.mesh_file_to_claydoc(_cube_bytes("stl_ascii"), ".stl", "Cube")
    mesh = doc.objects[0].mesh
    bm.validate(mesh)
    assert len(mesh.positions) == 8
    assert adjacency.check_manifold(mesh).clean


# --- PLY --------------------------------------------------------------------------


def test_ply_imports_as_one_connected_closed_object() -> None:
    doc = meshimport.mesh_file_to_claydoc(_cube_bytes("ply"), ".ply", "Cube")
    assert len(doc.objects) == 1
    mesh = doc.objects[0].mesh
    bm.validate(mesh)
    report = adjacency.check_manifold(mesh)
    assert report.clean, report


# --- scale / up, agreeing with objimport's convention ----------------------------


def _slab_bytes(file_type: str) -> bytes:
    """An asymmetric box, so an axis permutation is actually observable in
    its bounds -- a cube's own bounds are the same under any permutation."""
    slab = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
    data = slab.export(file_type=file_type)
    return data if isinstance(data, bytes) else data.encode("utf-8")


def test_scale_and_up_apply_the_same_way_as_for_obj() -> None:
    plain = meshimport.mesh_file_to_claydoc(_slab_bytes("ply"), ".ply", "Slab")
    plain_lo, plain_hi = bm.bounds(plain.objects[0].mesh)
    plain_extent = plain_hi - plain_lo  # (1, 2, 3)

    doc = meshimport.mesh_file_to_claydoc(_slab_bytes("ply"), ".ply", "Slab", scale=2.0, up="z")
    lo, hi = bm.bounds(doc.objects[0].mesh)
    extent = hi - lo

    # (x, y, z) -> (x, z, -y): an extent of (ex, ey, ez) becomes (ex, ez, ey),
    # then doubled by ``scale`` -- the same claim objimport's own up="z" test
    # makes for a single vertex.
    expected = np.array([plain_extent[0], plain_extent[2], plain_extent[1]]) * 2.0
    assert np.allclose(extent, expected, atol=1e-4)


# --- refusals -----------------------------------------------------------------


def test_an_unsupported_suffix_names_the_supported_list() -> None:
    with pytest.raises(OpError) as excinfo:
        meshimport.mesh_file_to_claydoc(b"whatever", ".fbx", "X")
    assert ".stl" in str(excinfo.value)
    assert ".ply" in str(excinfo.value)


def test_import_file_dispatches_by_suffix_and_refuses_the_rest() -> None:
    doc = meshimport.import_file(_cube_bytes("stl"), ".stl", "Cube")
    assert len(doc.objects) == 1

    doc = meshimport.import_file(_cube_bytes("ply"), ".PLY", "Cube")  # case-insensitive
    assert len(doc.objects) == 1

    with pytest.raises(OpError) as excinfo:
        meshimport.import_file(b"whatever", ".fbx", "X")
    for suffix in meshimport.SUPPORTED_SUFFIXES:
        assert suffix in str(excinfo.value)


def test_a_corrupt_stl_is_refused_rather_than_crashing() -> None:
    with pytest.raises(OpError):
        meshimport.mesh_file_to_claydoc(b"not an stl file at all", ".stl", "X")


def _binary_stl_header(declared_triangles: int) -> bytes:
    """A binary STL whose 80-byte header and triangle count are legitimate,
    but whose facet records are not -- exactly enough for
    ``_stl_declared_triangles``'s own length cross-check to accept it, and no
    further, since the regression below must never reach a real parse."""
    import struct

    header = b"\x00" * 80 + struct.pack("<I", declared_triangles)
    return header + b"\x00" * (declared_triangles * 50)


_PLY_HEADER = (
    "ply\nformat ascii 1.0\nelement vertex 8\nproperty float x\nproperty float y\n"
    "property float z\nelement face {faces}\nproperty list uchar int vertex_indices\n"
    "end_header\n"
)


def test_the_triangle_ceiling_for_stl_and_ply_refuses_before_trimesh_parses_the_whole_file(
    monkeypatch,
) -> None:
    """The 2026-09-19 audit, finding clay-18: ``mesh_file_to_claydoc`` called
    ``trimesh.load`` -- the full parse and allocation -- before checking
    ``tri_total``/object count against the ceilings, unlike its siblings
    (objimport's text pre-pass, glbimport's ``_declared_budget``). Binary
    STL's own header and PLY's own ASCII header each declare a triangle/face
    count for free; ``trimesh.load`` is monkeypatched to raise if reached at
    all, so this proves the refusal fires before that call, not after it.
    """
    monkeypatch.setattr(meshimport, "MAX_TRIANGLES", 5)

    def _boom(*args, **kwargs):
        raise AssertionError("trimesh.load must not run once the header over-declares")

    monkeypatch.setattr(trimesh, "load", _boom)

    with pytest.raises(OpError, match="triangles in its own header"):
        meshimport.mesh_file_to_claydoc(_binary_stl_header(10), ".stl", "X")

    ply_data = (_PLY_HEADER.format(faces=10) + "0 0 0\n" * 8).encode("ascii")
    with pytest.raises(OpError, match="triangles in its own header"):
        meshimport.mesh_file_to_claydoc(ply_data, ".ply", "X")
