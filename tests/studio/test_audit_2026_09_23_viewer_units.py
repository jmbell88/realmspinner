"""Regression test for the 2026-09-23 audit, finding create-11.

A material with all five PBR texture slots put its occlusion map on GL unit
4 -- the same unit ``Environment.bind`` always used for the specular probe.
The rebind ``_draw_model`` did for a five-texture material restored the
probe's own physical binding at unit 4, but ``u_ao_map``'s *uniform* still
pointed at unit 4, so the occlusion sampler read the environment probe
instead of the AO map.
"""

from __future__ import annotations

import numpy as np

from realmspinner.kernels.geom3d import gltf
from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.studio.viewer import glctx
from realmspinner.studio.viewer import scene as scenelib
from realmspinner.studio.viewer.camera import Camera
from realmspinner.studio.viewer.render import Renderer


def _solid_rgba(w, h, rgba):
    px = np.zeros((h, w, 4), dtype="u1")
    px[:, :] = rgba
    return (w, h, px.tobytes())


def test_a_five_texture_material_does_not_read_the_environment_probe_through_its_occlusion_map(
    gl,
):
    # A quad facing +Z, filling the frame under an identity-ish camera --
    # mirrors the orchestrator's create-11 probe exactly.
    positions = np.array(
        [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype="f4"
    )
    indices = np.array([0, 1, 2, 0, 2, 3], dtype="u4")
    uvs = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype="f4")
    normals = np.tile(np.array([0, 0, 1], dtype="f4"), (4, 1))

    material = gltf.Material(
        base_color_factor=(1.0, 1.0, 1.0, 1.0),
        metallic_factor=0.0,
        roughness_factor=1.0,
        base_color=_solid_rgba(1, 1, (255, 255, 255, 255)),
        metallic_roughness=_solid_rgba(1, 1, (0, 255, 0, 255)),
        normal=_solid_rgba(1, 1, (128, 128, 255, 255)),
        emissive=_solid_rgba(1, 1, (0, 0, 0, 255)),
        # Occlusion's red channel is 0 -- correctly bound, `lit *= ao.r`
        # zeroes the whole pixel to black; misbound, it reads the (bright,
        # non-black) environment probe instead.
        occlusion=_solid_rgba(1, 1, (0, 0, 0, 255)),
    )
    primitive = gltf.Primitive(
        positions=positions, indices=indices, normals=normals, uvs=uvs,
        material=material,
    )
    node = gltf.Node(mesh=0)
    node.world = m3.identity()
    model = gltf.Model(nodes=[node], roots=[0], meshes=[[primitive]], skins=[])

    gpu = scenelib.GpuModel(gl, model)
    assert len(gpu.materials[0].textures) == 5

    renderer = Renderer(gl)
    viewport = glctx.Viewport(gl, (64, 64))
    camera = Camera()
    camera.view = lambda: m3.identity()
    camera.projection = lambda: m3.identity()

    try:
        renderer.draw(
            viewport, camera, gpu, show_grid=False, model_matrix=m3.identity()
        )
        pixel = viewport.read_rgba()[32, 32, :3]
        assert tuple(pixel) == (0, 0, 0), (
            "occlusion map must zero the pixel; a non-black centre means the "
            "AO sampler is reading the environment probe on a shared unit"
        )
    finally:
        viewport.release()
        renderer.release()
        gpu.release()
