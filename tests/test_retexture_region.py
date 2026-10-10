"""Masked repair, host side: what ``retexture.assemble(region=)`` may and may not
touch, which views a repair is worth, and the UV fingerprint that ties a mask to
the layout it was painted against. No Blender, no GPU."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from realmspinner.pipelines import retexture

SIZE = 64


def _png(path, array, mode="RGB"):
    Image.fromarray(array, mode).save(path)


def _scene(tmp_path, *, region_box=(20, 20, 44, 44), bake_colour=(220, 30, 30)):
    """One view that covers the whole atlas in ``bake_colour`` at full weight, a
    noisy base atlas, and a rectangular region mask."""
    rng = np.random.default_rng(11)
    base = rng.integers(0, 256, size=(SIZE, SIZE, 3), dtype=np.uint8)
    _png(tmp_path / "base.png", base)
    _png(tmp_path / "bake_00.png", np.full((SIZE, SIZE, 3), bake_colour, np.uint8))
    _png(tmp_path / "weight_00.png", np.full((SIZE, SIZE), 255, np.uint8), "L")
    mask = np.zeros((SIZE, SIZE), np.uint8)
    y0, x0, y1, x1 = region_box
    mask[y0:y1, x0:x1] = 255
    _png(tmp_path / "region.png", mask, "L")
    return base, mask


def test_assemble_region_leaves_unmasked_texels_byte_identical(tmp_path):
    base, mask = _scene(tmp_path)
    report = retexture.assemble(
        tmp_path, tmp_path / "base.png", tmp_path / "atlas.png",
        count=1, region=tmp_path / "region.png",
    )
    out = np.asarray(Image.open(tmp_path / "atlas.png").convert("RGB"))
    outside = mask == 0
    assert (out[outside] == base[outside]).all(), "a texel outside the region changed"
    centre = out[32, 32]
    assert centre[0] > 200 and centre[1] < 60, "deep inside, the repaint is the bake"
    assert report["region"]["texels"] == int((mask > 0).sum())
    assert report["region"]["repainted"] == pytest.approx(1.0)


def test_the_repaint_fades_in_at_the_region_edge(tmp_path):
    base, mask = _scene(tmp_path)
    retexture.assemble(
        tmp_path, tmp_path / "base.png", tmp_path / "atlas.png",
        count=1, region=tmp_path / "region.png",
    )
    out = np.asarray(Image.open(tmp_path / "atlas.png").convert("RGB")).astype(int)
    edge = out[32, 20]
    deep = out[32, 32]
    # The edge texel is a blend of the old skin and the bake, not either one.
    assert edge[1] > deep[1] + 5 or edge[0] < deep[0] - 5
    assert not np.array_equal(edge, base[32, 20])


def test_assemble_without_region_unchanged(tmp_path):
    base, _mask = _scene(tmp_path)
    report = retexture.assemble(tmp_path, tmp_path / "base.png", tmp_path / "plain.png", count=1)
    out = np.asarray(Image.open(tmp_path / "plain.png").convert("RGB"))
    assert "region" not in report
    # A full-weight single view with no occlusion test replaces the whole atlas:
    # what a re-texture always did, and the repair branch must not have touched.
    assert (out == np.array([220, 30, 30], np.uint8)).all()
    assert not np.array_equal(out, base)


def test_a_repair_with_no_base_or_a_mismatched_one_is_refused_not_guessed(tmp_path):
    _scene(tmp_path)
    assert retexture.assemble(
        tmp_path, None, tmp_path / "a.png", count=1, region=tmp_path / "region.png"
    ) is None
    _png(tmp_path / "small.png", np.zeros((8, 8, 3), np.uint8))
    small = tmp_path / "small.png"
    assert retexture.assemble(
        tmp_path, small, tmp_path / "b.png", count=1, region=tmp_path / "region.png"
    ) is None
    _png(tmp_path / "wrong_mask.png", np.full((8, 8), 255, np.uint8), "L")
    assert retexture.assemble(
        tmp_path, tmp_path / "base.png", tmp_path / "c.png", count=1,
        region=tmp_path / "wrong_mask.png",
    ) is None
    assert not (tmp_path / "a.png").exists() and not (tmp_path / "c.png").exists()


def test_texels_inside_the_region_no_view_saw_keep_their_old_colour(tmp_path):
    base, mask = _scene(tmp_path)
    _png(tmp_path / "weight_00.png", np.zeros((SIZE, SIZE), np.uint8), "L")
    report = retexture.assemble(
        tmp_path, tmp_path / "base.png", tmp_path / "atlas.png",
        count=1, region=tmp_path / "region.png",
    )
    out = np.asarray(Image.open(tmp_path / "atlas.png").convert("RGB"))
    assert np.array_equal(out, base), "nothing looked at the patch, so nothing changed"
    assert report["region"]["repainted"] == 0.0


# ---- which views ------------------------------------------------------------------


def _view(fraction, side=100):
    mask = np.zeros((side, side))
    mask[: int(round(fraction * side * side / side)), :] = 1.0
    return mask


def test_pick_region_views_takes_the_views_that_see_the_most_and_caps_them():
    shares = [0.05, 0.30, 0.0, 0.20, 0.10]
    views = [(index, _view(share)) for index, share in enumerate(shares)]
    assert retexture.pick_region_views(views) == [1, 3, 4]
    assert retexture.pick_region_views(views, cap=1) == [1]
    assert len(retexture.pick_region_views(views)) == retexture.REGION_MAX_VIEWS == 3


def test_pick_region_views_drops_a_view_that_barely_sees_the_patch():
    sliver = np.zeros((100, 100))
    sliver[0, :5] = 1.0  # 5 of 10,000 pixels
    assert retexture.pick_region_views([(0, sliver)]) == []
    assert retexture.pick_region_views([]) == []


def test_pick_region_views_breaks_ties_towards_the_lower_index():
    same = _view(0.2)
    assert retexture.pick_region_views([(5, same), (2, same), (7, same), (3, same)]) == [2, 3, 5]


def test_inpaint_mask_thresholds_the_render_and_grows_it(tmp_path):
    soft = np.zeros((40, 40), np.uint8)
    soft[18:22, 18:22] = 255
    soft[10, 10] = 100  # anti-aliased fringe, under half
    _png(tmp_path / "mask_00.png", soft, "L")
    assert retexture.inpaint_mask(tmp_path / "mask_00.png", tmp_path / "grown.png", grow_px=3)
    grown = np.asarray(Image.open(tmp_path / "grown.png"))
    assert grown[16, 20] == 255 and grown[20, 24] == 255, "grown by three pixels"
    assert grown[10, 10] == 0, "a fringe under half is not part of the region"
    assert set(np.unique(grown)) <= {0, 255}
    assert not retexture.inpaint_mask(tmp_path / "missing.png", tmp_path / "x.png")


# ---- the UV fingerprint ------------------------------------------------------------


def _textured_quad(path, uv, *, scale=1.0):
    import trimesh

    mesh = trimesh.Trimesh(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float) * scale,
        faces=[[0, 1, 2], [0, 2, 3]],
        process=False,
    )
    mesh.visual = trimesh.visual.TextureVisuals(uv=np.array(uv, float))
    trimesh.Scene(mesh).export(path)


def test_uv_digest_ignores_a_resize_and_notices_a_changed_layout(tmp_path):
    uv = [[0, 0], [1, 0], [1, 1], [0, 1]]
    _textured_quad(tmp_path / "a.glb", uv)
    _textured_quad(tmp_path / "bigger.glb", uv, scale=2.5)
    _textured_quad(tmp_path / "moved.glb", [[0, 0], [1, 0], [1, 1], [0.1, 1]])
    a = retexture.uv_digest(tmp_path / "a.glb")
    assert a and len(a) == 64
    assert retexture.uv_digest(tmp_path / "bigger.glb") == a
    assert retexture.uv_digest(tmp_path / "moved.glb") != a


def test_uv_digest_is_none_for_a_mesh_with_no_uvs_or_a_missing_file(tmp_path):
    import trimesh

    trimesh.Scene(trimesh.creation.box()).export(tmp_path / "plain.glb")
    assert retexture.uv_digest(tmp_path / "plain.glb") is None
    assert retexture.uv_digest(tmp_path / "missing.glb") is None
