"""The 2026-10-14 audit, finding create-viewer-01: X-ray did not turn depth writes off.

``Renderer.draw`` assigned ``ctx.depth_mask``, which moderngl's ``Context`` does
not have (only ``Framebuffer.depth_mask`` exists), so the assignment made an
inert Python attribute and a translucent object drawn first still hid every
object behind it.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from realmspinner.kernels.geom3d import gltf
from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.studio.viewer import glctx
from realmspinner.studio.viewer import scene as scenelib
from realmspinner.studio.viewer.camera import Camera
from realmspinner.studio.viewer.render import Renderer


@pytest.fixture(scope="module")
def renderer(gl):
    r = Renderer(gl)
    yield r
    r.release()


@pytest.fixture
def viewport(gl):
    vp = glctx.Viewport(gl, (96, 96))
    yield vp
    vp.release()


def _box(centre, rgba):
    mesh = trimesh.creation.box(extents=(0.6, 0.6, 0.6))
    mesh.apply_translation(centre)
    mesh.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=rgba, metallicFactor=0.0
        )
    )
    return mesh


def _centre_pixel(gl, renderer, viewport, tmp_path, camera, names, alpha):
    eye = np.asarray(camera.position, dtype="f8")
    toward = eye / np.linalg.norm(eye)
    scene = trimesh.Scene()
    # Node order matters: the nearer box is listed first, so with depth writes
    # left on it is the one that hides the far box.
    if "near" in names:
        scene.add_geometry(_box(toward * 0.9, [0, 0, 255, 255]), node_name="near", geom_name="near")
    if "far" in names:
        scene.add_geometry(_box(-toward * 0.9, [255, 0, 0, 255]), node_name="far", geom_name="far")
    path = tmp_path / ("_".join(names) + ".glb")
    scene.export(path)
    model = gltf.load(path)
    gpu = scenelib.GpuModel(gl, model)
    renderer.fit_grid(*model.bounds())
    renderer.draw(
        viewport, camera, gpu, show_grid=False, model_matrix=m3.identity(), alpha=alpha
    )
    return viewport.read_rgba()[48, 48].astype(int)


def test_xray_draws_the_far_object_through_a_nearer_one_drawn_first(
    gl, renderer, viewport, tmp_path
):
    camera = Camera()
    camera.theta, camera.phi, camera.distance = 0.0, np.pi / 2, 4.0
    near_only = _centre_pixel(gl, renderer, viewport, tmp_path, camera, ["near"], 0.4)
    both = _centre_pixel(gl, renderer, viewport, tmp_path, camera, ["near", "far"], 0.4)
    # The far box is red: seen through the translucent near one, the centre
    # pixel must gain red it does not have with the near box alone.
    assert both[0] > near_only[0] + 20, (near_only, both)


def test_xray_restores_the_depth_write_afterwards_even_when_the_draw_raises(
    gl, renderer, viewport, tmp_path, monkeypatch
):
    camera = Camera()
    camera.theta, camera.phi, camera.distance = 0.0, np.pi / 2, 4.0
    scene = trimesh.Scene()
    scene.add_geometry(_box((0, 0, 0), [0, 0, 255, 255]), node_name="a", geom_name="a")
    path = tmp_path / "a.glb"
    scene.export(path)
    gpu = scenelib.GpuModel(gl, gltf.load(path))

    def boom(*_a, **_k):
        raise RuntimeError("draw failed")

    monkeypatch.setattr(renderer, "_draw_model", boom)
    with pytest.raises(RuntimeError):
        renderer.draw(viewport, camera, gpu, show_grid=False, alpha=0.4)
    assert viewport.draw_target.depth_mask is True
