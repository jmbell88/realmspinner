"""The region as each camera sees it, through real Blender.

``op_views`` renders a UV-space mask onto the mesh and saves one ``mask_NN.png``
per view. The claims are that the mask lands where the surface is (not on the
background), that it separates the painted half from the unpainted half, and that
a re-texture which was not asked for a region renders none.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from realmspinner.kernels.rig import blender_spec
from realmspinner.pipelines import blender_run

pytestmark = pytest.mark.timeout(600)


@pytest.fixture(scope="module")
def sphere_job(tmp_path_factory, gl):
    # ``gl`` is a skip, not a dependency: Blender's render ops open a GPU
    # context, and on a runner with none (Windows CI) the worker dies with an
    # access violation (exit 3221225477, "Implementation unexpectedly missing
    # WGL_ARB_extensions_string") rather than raising. Where moderngl cannot
    # get a context neither can Blender, so these skip there as ``gl`` does.
    pytest.importorskip("bpy")
    import trimesh

    job = tmp_path_factory.mktemp("region-sphere")
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=0.5)
    x, y, z = sphere.vertices.T
    # u runs around the vertical axis and v from pole to pole. The region below is
    # split on v: a camera sees only half of u's range, so a u-split would hand
    # the front view a wholly unpainted hemisphere and the back a wholly painted
    # one, but every camera sees both the northern and the southern half.
    uv = np.stack([np.arctan2(z, x) / (2 * np.pi) + 0.5, np.arcsin(y / 0.5) / np.pi + 0.5], 1)
    base = Image.new("RGB", (16, 16), (200, 200, 200))
    sphere.visual = trimesh.visual.TextureVisuals(
        uv=uv, material=trimesh.visual.material.PBRMaterial(baseColorTexture=base)
    )
    sphere.apply_translation([0.0, 0.5, 0.0])
    trimesh.Scene(sphere).export(job / "model.glb")
    mask = np.zeros((64, 64), np.uint8)
    mask[:32, :] = 255  # the top half of the atlas
    Image.fromarray(mask, "L").save(job / "region.png")
    return job


def _render(job, out, region):
    spec = blender_spec.views_spec(
        job / "model.glb", out, [(0.0, 0.0), (180.0, 0.0)], size=96, region_mask=region
    )
    return blender_run.run_worker(spec, timeout=600)


def test_each_view_gets_a_mask_that_splits_the_painted_half_from_the_rest(sphere_job, tmp_path):
    _render(sphere_job, tmp_path, sphere_job / "region.png")
    for index in (0, 1):
        colour = np.asarray(Image.open(tmp_path / f"view_{index:02d}.png").convert("RGBA"))
        mask = np.asarray(Image.open(tmp_path / f"mask_{index:02d}.png").convert("L"))
        on_surface = colour[..., 3] > 200
        assert on_surface.sum() > 500
        painted = mask[on_surface] > 127
        # One hemisphere by v is painted, so each camera sees a real mix -- and
        # the background, where nothing renders, is never painted.
        assert 0.15 < painted.mean() < 0.85, f"view {index}: {painted.mean():.2f} of the surface"
        # Fully empty pixels only: an antialiased silhouette pixel is partly the
        # surface and legitimately carries its (bright) mask.
        assert (mask[colour[..., 3] == 0] < 127).all()


def test_a_re_texture_with_no_region_renders_no_masks(sphere_job, tmp_path):
    _render(sphere_job, tmp_path, None)
    assert (tmp_path / "view_00.png").exists()
    assert not list(tmp_path.glob("mask_*.png"))


def test_the_spec_names_the_mask_only_when_one_is_asked_for(tmp_path):
    plain = blender_spec.views_spec(tmp_path / "m.glb", tmp_path, [(0.0, 0.0)], size=64)
    assert "region_mask" not in plain
    asked = blender_spec.views_spec(
        tmp_path / "m.glb", tmp_path, [(0.0, 0.0)], size=64, region_mask=tmp_path / "r.png"
    )
    assert asked["region_mask"] == str(tmp_path / "r.png")
    assert {k: v for k, v in asked.items() if k != "region_mask"} == plain
