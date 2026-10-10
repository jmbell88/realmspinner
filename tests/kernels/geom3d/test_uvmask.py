"""A painted region as a mask over the atlas: the translation is pure arithmetic,
so every claim is checkable against a brute-force reference or a hand-built quad."""

from __future__ import annotations

import numpy as np
import pytest

from realmspinner.kernels.geom3d import uvmask


def _grid(n: int):
    """An n x n grid of unit-UV quads, two triangles each: (uv, tris)."""
    uv, tris = [], []
    for j in range(n + 1):
        for i in range(n + 1):
            uv.append((i / n, j / n))
    stride = n + 1
    for j in range(n):
        for i in range(n):
            a, b = j * stride + i, j * stride + i + 1
            c, d = (j + 1) * stride + i, (j + 1) * stride + i + 1
            tris += [(a, b, d), (a, d, c)]
    return np.array(uv), np.array(tris)


def _reference(uv, tris, faces, size):
    """One triangle at a time, centre-sampling, the slow obvious way."""
    out = np.zeros((size, size), dtype=np.uint8)
    for face in faces:
        p = uv[tris[face]] * size
        for y in range(size):
            for x in range(size):
                point = np.array([x + 0.5, y + 0.5])
                a, b, c = p
                d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
                if abs(d) < 1e-12:
                    continue
                w0 = ((b[1] - c[1]) * (point[0] - c[0]) + (c[0] - b[0]) * (point[1] - c[1])) / d
                w1 = ((c[1] - a[1]) * (point[0] - c[0]) + (a[0] - c[0]) * (point[1] - c[1])) / d
                if min(w0, w1, 1 - w0 - w1) >= -1e-9:
                    out[y, x] = 255
    return out


def test_two_triangles_of_a_quad_cover_the_whole_atlas_with_no_seam():
    uv, tris = _grid(1)
    mask = uvmask.rasterize_uv(uv, tris, None, 32)
    assert (mask == 255).all(), "texels on the shared diagonal belong to both triangles"


def test_uvmask_rasterizes_only_selected_faces():
    uv, tris = _grid(2)  # four quads: bottom-left is faces 0 and 1
    mask = uvmask.rasterize_uv(uv, tris, np.array([0, 1]), 32)
    assert (mask[:16, :16] == 255).all()
    assert not mask[16:, :].any() and not mask[:, 16:].any()
    assert uvmask.coverage(mask) == pytest.approx(0.25)


def test_the_rows_follow_v_directly_with_no_flip():
    """glTF's texture origin is the image's top-left, so v = 0.25 is a quarter of
    the way down the picture -- and the mask must line up with the atlas file."""
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.25]])
    mask = uvmask.rasterize_uv(uv, np.array([[0, 1, 2]]), None, 40)
    assert mask[:10].any() and not mask[10:].any()


def test_a_vectorised_batch_matches_the_one_at_a_time_reference():
    rng = np.random.default_rng(3)
    uv = rng.random((60, 2))
    tris = rng.integers(0, 60, size=(80, 3))
    faces = np.arange(80)
    assert (uvmask.rasterize_uv(uv, tris, faces, 48) == _reference(uv, tris, faces, 48)).all()


def test_triangles_bigger_than_the_batch_ceiling_take_the_single_path_and_agree():
    uv = np.array([[0.02, 0.03], [0.97, 0.1], [0.4, 0.95]])
    tris = np.array([[0, 1, 2]])
    size = 160  # a triangle ~150 texels across, well past MAX_BATCH_SIDE
    got = uvmask.rasterize_uv(uv, tris, None, size)
    assert (got == _reference(uv, tris, np.array([0]), size)).all()


def test_a_degenerate_or_off_atlas_triangle_adds_nothing_and_does_not_wrap():
    uv = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [1.5, 1.5], [1.6, 1.5], [1.5, 1.6]])
    mask = uvmask.rasterize_uv(uv, np.array([[0, 1, 2], [3, 4, 5]]), None, 32)
    assert not mask.any()


def test_a_large_mesh_rasterises_in_batches_and_matches_a_coarse_check():
    uv, tris = _grid(150)  # 45,000 triangles, a few texels each
    mask = uvmask.rasterize_uv(uv, tris, None, 300)
    assert (mask == 255).all()
    half = uvmask.rasterize_uv(uv, tris, np.arange(len(tris) // 2), 300)
    assert 0.45 < uvmask.coverage(half) < 0.55


def test_faces_in_brush_selects_by_centre_distance():
    centres = np.array([[0.0, 0, 0], [0.5, 0, 0], [2.0, 0, 0]])
    assert uvmask.faces_in_brush(centres, [0, 0, 0], 0.6).tolist() == [0, 1]
    assert uvmask.faces_in_brush(centres, [0, 0, 0], 0.0).tolist() == []
    assert uvmask.faces_in_brush(centres[:0], [0, 0, 0], 1.0).tolist() == []


def test_feather_inside_is_exactly_zero_outside_and_one_deep_inside():
    mask = np.zeros((40, 40), np.uint8)
    mask[10:30, 10:30] = 255
    weight = uvmask.feather_inside(mask, 4)
    assert weight.dtype == np.float32
    assert (weight[mask == 0] == 0).all(), "nothing outside the region may carry weight"
    assert weight[20, 20] == pytest.approx(1.0)
    edge, near, deep = weight[20, 10], weight[20, 12], weight[20, 20]
    assert 0.2 < edge < near < deep <= 1.0


def test_feather_with_no_width_is_the_mask_itself():
    mask = np.zeros((8, 8), np.uint8)
    mask[2:4, 2:4] = 255
    assert (uvmask.feather_inside(mask, 0) == (mask > 0)).all()
    assert not uvmask.feather_inside(np.zeros((8, 8), np.uint8), 5).any()


def test_grow_dilates_by_texels_and_never_wraps():
    mask = np.zeros((9, 9), np.uint8)
    mask[0, 4] = 255
    grown = uvmask.grow(mask, 2)
    assert grown[0, 2] == 255 and grown[2, 4] == 255 and grown[1, 3] == 255
    assert grown[8].sum() == 0, "the top row's growth must not reach the bottom edge"


def test_flatten_applies_node_transforms_and_reports_missing_uvs(tmp_path):
    import trimesh

    from realmspinner.kernels.geom3d import gltf

    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    box.apply_translation([0.0, 5.0, 0.0])
    path = tmp_path / "m.glb"
    trimesh.Scene(box).export(path)
    flat = uvmask.flatten(gltf.load(path))
    assert flat.positions[:, 1].min() == pytest.approx(4.5)
    assert len(flat.tris) == 12
    assert flat.uv is None, "a mesh with no texture coordinates has nothing to repair"
    assert uvmask.centroids(flat).shape == (12, 3)
