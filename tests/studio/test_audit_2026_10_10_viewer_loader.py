"""Regression tests for the 2026-10-10 audit, findings create-01 .. create-05
and create-38 (the glTF loader and the viewer's draw path).
"""

from __future__ import annotations

import base64
import dataclasses
import io
import json
import struct

import numpy as np
import pytest

from realmspinner.kernels.geom3d import gltf
from realmspinner.studio.viewer import glctx
from realmspinner.studio.viewer import scene as scenelib
from realmspinner.studio.viewer.camera import Camera
from realmspinner.studio.viewer.render import Renderer


def _glb(doc: dict, binary: bytes = b"") -> bytes:
    js = json.dumps(doc).encode()
    js += b" " * (-len(js) % 4)
    binary += b"\0" * (-len(binary) % 4)
    total = 12 + 8 + len(js) + 8 + len(binary)
    return (
        struct.pack("<III", 0x46546C67, 2, total)
        + struct.pack("<II", len(js), 0x4E4F534A)
        + js
        + struct.pack("<II", len(binary), 0x004E4942)
        + binary
    )


_TRI = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="<f4")


def _tri_doc(nodes: list[dict], scenes: list[dict], n_prims: int = 1) -> tuple[dict, bytes]:
    pos = _TRI.tobytes()
    idx = np.array([0, 1, 2], dtype="<u2").tobytes() + b"\0\0"
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": scenes,
        "nodes": nodes,
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}] * n_prims}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(pos)},
            {"buffer": 0, "byteOffset": len(pos), "byteLength": 6},
        ],
        "buffers": [{"byteLength": len(pos) + len(idx)}],
    }
    return doc, pos + idx


# --- create-01: nodes x primitives was unbounded -----------------------------


def test_a_document_whose_mesh_instances_pass_the_ceiling_is_refused_before_any_instance_is_walked(
    monkeypatch,
):
    # The 2026-10-10 audit: 100 nodes naming one 5,000-primitive mesh sat under
    # every per-array ceiling and made Model.bounds() take 6.4 s on the frame
    # thread. The ceiling is lowered rather than the file built that big.
    monkeypatch.setattr(gltf, "MAX_INSTANCES", 5, raising=False)

    def _must_not_run(*args, **kwargs):
        raise AssertionError("an instance loop ran before the instance ceiling")

    monkeypatch.setattr(gltf, "_check_joint_indices", _must_not_run)
    doc, binary = _tri_doc([{"mesh": 0}] * 3, [{"nodes": [0, 1, 2]}], n_prims=2)  # 6 > 5
    with pytest.raises(ValueError, match="instances"):
        gltf.load(_glb(doc, binary))


def test_an_instanced_mesh_still_frames_exactly_as_one_box_per_instance():
    # Parity guard for the once-per-mesh walk: three nodes naming one
    # two-primitive mesh, one of them rotated, must frame as before.
    doc, binary = _tri_doc(
        [
            {"mesh": 0},
            {"mesh": 0, "translation": [10, 0, 0]},
            {"mesh": 0, "translation": [0, 0, 5], "rotation": [0, 0.7071068, 0, 0.7071068]},
        ],
        [{"nodes": [0, 1, 2]}],
        n_prims=2,
    )
    lo, hi = gltf.load(_glb(doc, binary)).bounds()
    assert np.allclose(lo, [0, 0, 0], atol=1e-6)
    assert np.allclose(hi, [11, 1, 5], atol=1e-6)


# --- create-04: a NaN/Inf POSITION blanked the whole model -------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_position_is_refused_at_load(bad):
    pos = np.array([[0, 0, 0], [2, 0, 0], [2, 2, 0]], dtype="<f4")
    pos[2, 0] = bad
    doc, binary = _tri_doc([{"mesh": 0}], [{"nodes": [0]}])
    binary = pos.tobytes() + binary[len(_TRI.tobytes()) :]
    with pytest.raises(ValueError, match="non-finite"):
        gltf.load(_glb(doc, binary))


# --- create-05: nodes outside the default scene were drawn at the origin -----


def test_a_mesh_node_outside_the_default_scene_is_not_drawn_or_framed():
    doc, binary = _tri_doc(
        [
            {"mesh": 0, "translation": [100, 0, 0]},
            {"mesh": 0, "translation": [5000, 0, 0]},
        ],
        [{"nodes": [0]}, {"nodes": [1]}],
    )
    model = gltf.load(_glb(doc, binary))
    assert model.reached == {0}
    assert len(model.mesh_instances()) == 1
    lo, hi = model.bounds()
    assert lo[0] == pytest.approx(100.0) and hi[0] == pytest.approx(101.0)


# --- create-38: one side past the GL limit passed the area ceiling -----------


def test_a_texture_with_one_side_past_the_gl_limit_is_skipped_not_uploaded():
    from PIL import Image

    # 40000 x 399 is 15.96 M pixels: under MAX_TEXTURE_PIXELS, over any GL
    # driver's MAX_TEXTURE_SIZE on one side.
    buf = io.BytesIO()
    Image.new("L", (40000, 399), 7).save(buf, "PNG")
    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    assert gltf.MAX_TEXTURE_PIXELS >= 40000 * 399
    doc, binary = _tri_doc([{"mesh": 0}], [{"nodes": [0]}])
    doc["images"] = [{"uri": uri}]
    doc["textures"] = [{"source": 0}]
    doc["materials"] = [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}]
    doc["meshes"][0]["primitives"][0]["material"] = 0
    model = gltf.load(_glb(doc, binary))
    assert len(model.meshes[0]) == 1, "the mesh still loads"
    assert model.meshes[0][0].material.base_color is None
    assert model.skipped_textures == 1


# --- create-02 / create-03: the draw path (GL 3.3) ----------------------------


@pytest.fixture(scope="module")
def _renderer(gl):
    r = Renderer(gl)
    yield r
    r.release()


def _prim(z: float, mat: gltf.Material, half: float = 5.0) -> gltf.Primitive:
    p = np.array([[-half, -half, z], [half, -half, z], [0, half, z]], dtype="f4")
    return gltf.Primitive(
        positions=p,
        indices=np.array([0, 1, 2], dtype="u4"),
        normals=np.tile(np.array([0, 0, 1], dtype="f4"), (3, 1)),
        material=mat,
    )


def _centre(gl, renderer, prims, *, flat=False, background=(0.0, 0.0, 0.0, 1.0)):
    nodes = [gltf.Node(mesh=i) for i in range(len(prims))]
    model = gltf.Model(nodes, list(range(len(prims))), [[p] for p in prims], [])
    gpu = scenelib.GpuModel(gl, model)
    vp = glctx.Viewport(gl, (64, 64))
    try:
        cam = Camera()
        cam.set_target(np.zeros(3))
        cam.set_position(np.array([0.0, 0.0, 6.0]))
        cam.far = 50.0
        renderer.draw(vp, cam, gpu, flat=flat, show_grid=False, background=background)
        depth_mask = vp.draw_target.depth_mask
        return tuple(int(v) for v in vp.read_rgba()[32, 32]), depth_mask
    finally:
        gpu.release()
        vp.release()


@pytest.mark.parametrize("flat", [False, True])
def test_a_masked_material_that_passes_its_cutoff_renders_fully_opaque(gl, _renderer, flat):
    # The 2026-10-10 audit: only OPAQUE forced alpha to 1, so a MASK material
    # that passed its cutoff kept its sampled alpha (a=0.6 -> 153) and was
    # blended; three draws it opaque. (The test used to pass ``keep_alpha`` too:
    # that blend mode served only the direction strip, deleted on 2026-10-14 --
    # the 2026-10-14 audit, finding create-viewer-03 -- and a blended 0.6 still
    # reads 0.36 -> 92 on the plain blend, so the claim is still observable.)
    mat = dataclasses.replace(
        gltf.Material(), alpha_mode="MASK", alpha_cutoff=0.5,
        base_color_factor=(1.0, 1.0, 1.0, 0.6), double_sided=True,
    )
    px, _ = _centre(gl, _renderer, [_prim(0.0, mat)], flat=flat,
                    background=(0.0, 0.0, 0.0, 0.0))
    assert px[3] == 255


def _glass(rgb, alpha=0.5):
    return dataclasses.replace(
        gltf.Material(), alpha_mode="BLEND", double_sided=True,
        base_color_factor=(*rgb, alpha),
    )


_SOLID = dataclasses.replace(
    gltf.Material(), double_sided=True, base_color_factor=(1.0, 0.1, 0.1, 1.0)
)


def test_a_blend_material_does_not_hide_an_opaque_mesh_drawn_after_it(gl, _renderer):
    glass_first, mask_on = _centre(
        gl, _renderer, [_prim(0.0, _glass((0.2, 0.4, 1.0))), _prim(-1.0, _SOLID)]
    )
    solid_first, _ = _centre(
        gl, _renderer, [_prim(-1.0, _SOLID), _prim(0.0, _glass((0.2, 0.4, 1.0)))]
    )
    assert glass_first == solid_first
    assert mask_on, "the depth mask is back on after the blended pass"


def test_blend_primitives_are_drawn_far_to_near_whatever_the_node_order(gl, _renderer):
    near, far = _glass((1.0, 0.1, 0.1)), _glass((0.1, 0.1, 1.0))
    a, _ = _centre(gl, _renderer, [_prim(1.0, near), _prim(-1.0, far)])
    b, _ = _centre(gl, _renderer, [_prim(-1.0, far), _prim(1.0, near)])
    assert a == b
    # Far (blue) is laid down first, the near red pane goes over it: red ends up
    # the stronger channel, the reverse order would leave blue on top.
    assert a[0] > a[2]
