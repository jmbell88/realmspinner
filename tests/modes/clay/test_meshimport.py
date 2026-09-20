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
