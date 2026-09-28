"""Regression tests for the 2026-09-26 audit, findings create-viewer-02,
create-viewer-03 and create-viewer-04.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from realmspinner.kernels.geom3d import gltf
from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.kernels.geom3d.glbio import rebuild_glb
from realmspinner.studio.viewer import capture
from realmspinner.studio.viewer import scene as scenelib
from realmspinner.studio.viewer.camera import Camera


def _glb(gltf_json: dict, binary: bytes = b"") -> bytes:
    binary += b"\x00" * (-len(binary) % 4)
    chunk = struct.pack("<II", len(binary), 0x004E4942) + binary
    header = struct.pack("<III", 0x46546C67, 2, 0)
    return rebuild_glb(header, gltf_json, chunk)


# --- create-viewer-03: UNLIT_FRAG ignores alphaMode -------------------------


@pytest.fixture(scope="session")
def gl():
    moderngl = pytest.importorskip("moderngl")
    try:
        ctx = moderngl.create_context(standalone=True, require=330)
    except Exception as exc:  # no display, no driver, software-only
        pytest.skip(f"no GL 3.3 context: {exc}")
    yield ctx
    ctx.release()


def _big_triangle_glb() -> bytes:
    """One triangle covering an identity clip space, an OPAQUE material whose
    ``baseColorFactor`` carries a low alpha -- the M03-style trap this test
    is about: OPAQUE means the alpha channel is ignored, not that it happens
    to be 1.
    """
    positions = np.array([[-3, -3, 0], [3, -3, 0], [0, 3, 0]], dtype="<f4")
    binary = positions.tobytes()
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {"primitives": [{"attributes": {"POSITION": 0}, "material": 0}]}
        ],
        "materials": [
            {
                "pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, 0.2]},
                # alphaMode omitted -> OPAQUE, the glTF default.
            }
        ],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(binary)}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}
        ],
    }
    return _glb(doc, binary)


@pytest.fixture
def renderer(gl):
    from realmspinner.studio.viewer.render import Renderer

    r = Renderer(gl)
    yield r
    r.release()


@pytest.fixture
def viewport(gl):
    from realmspinner.studio.viewer.glctx import Viewport

    vp = Viewport(gl, (128, 128))
    yield vp
    vp.release()


@pytest.mark.parametrize("flat", [False, True], ids=["pbr", "unlit"])
def test_a_flat_render_of_an_opaque_material_ignores_its_base_colour_texture_alpha(
    gl, renderer, viewport, tmp_path, flat
):
    path = tmp_path / "opaque.glb"
    path.write_bytes(_big_triangle_glb())
    model = gltf.load(path)
    gpu = scenelib.GpuModel(gl, model)
    camera = Camera()
    camera.view = m3.identity
    camera.projection = m3.identity

    renderer.draw(
        viewport, camera, gpu, flat=flat, show_grid=False, background=(0.0, 0.0, 0.0, 0.0)
    )
    pixel = viewport.read_rgba()[64, 64]

    assert pixel[3] == 255, (
        f"{'unlit' if flat else 'pbr'} program rendered an OPAQUE material "
        f"translucent (alpha={pixel[3]}) instead of forcing it to fully opaque"
    )
    gpu.release()


# --- create-viewer-04: transparent-background alpha needs unpremultiplying -


def test_unpremultiply_undoes_the_msaa_resolves_own_coverage_darkening():
    """A partially covered silhouette-edge texel out of the MSAA resolve is
    ``colour * coverage`` against a transparent clear -- e.g. a pure-white
    edge at 25% coverage resolves to rgba (64, 64, 64, 64), not
    (255, 255, 255, 64). Treating that as straight alpha and compositing it
    again (an ordinary PNG viewer) darkens the fringe a second time; the
    fix must recover the original colour.
    """
    premultiplied = np.array(
        [[[64, 64, 64, 64], [10, 10, 10, 0], [200, 100, 50, 255]]], dtype=np.uint8
    )

    straight = capture.unpremultiply(premultiplied)

    assert straight[0, 0].tolist() == [255, 255, 255, 64]
    # Fully opaque pixels are already straight alpha -- a no-op.
    assert straight[0, 2].tolist() == [200, 100, 50, 255]


def test_sheet_step_unpremultiplies_its_transparent_readback():
    """``sheet.py``'s direction-strip cell renders straight into a
    transparent clear and pastes ``viewport.read_rgba()`` verbatim -- it
    must route that readback through ``capture.unpremultiply`` like every
    other alpha-carrying capture site, or its own silhouette edges keep the
    dark fringe."""
    import inspect

    from realmspinner.studio.viewer import sheet as sheet_mod

    source = inspect.getsource(sheet_mod.StripRender.step)
    assert "unpremultiply" in source, (
        "Strip.step must unpremultiply its transparent-background readback "
        "before pasting it into the direction strip"
    )


# --- create-viewer-02 (+clay-io-03): primitive() attribute validation ------


def _prim_glb(extra_accessors, extra_views, extra_blobs, attributes) -> bytes:
    """Three positions plus whatever extra attribute accessors/views/blobs
    the caller wants layered on top, wired into one primitive."""
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="<f4")
    blobs = [positions.tobytes(), *extra_blobs]
    views, binary = [], b""
    for blob in blobs:
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(blob)})
        binary += blob + b"\x00" * (-len(blob) % 4)
    accessors = [
        {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
        *extra_accessors,
    ]
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": attributes}]}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": views,
        "accessors": accessors,
    }
    return _glb(doc, binary)


def test_a_normal_accessor_with_fewer_rows_than_position_is_refused_at_load(tmp_path):
    normals = np.tile(np.array([0, 0, 1], dtype="<f4"), (2, 1))  # 2, not 3
    data = _prim_glb(
        extra_accessors=[
            {"bufferView": 1, "componentType": 5126, "count": 2, "type": "VEC3"}
        ],
        extra_views=[],
        extra_blobs=[normals.tobytes()],
        attributes={"POSITION": 0, "NORMAL": 1},
    )
    path = tmp_path / "mismatched_normal.glb"
    path.write_bytes(data)
    with pytest.raises(ValueError, match="NORMAL"):
        gltf.load(path)


def test_joints_with_no_weights_is_refused_at_load(tmp_path):
    joints = np.tile(np.array([0, 0, 0, 0], dtype="<u1"), (3, 1))
    data = _prim_glb(
        extra_accessors=[
            {"bufferView": 1, "componentType": 5121, "count": 3, "type": "VEC4"}
        ],
        extra_views=[],
        extra_blobs=[joints.tobytes()],
        attributes={"POSITION": 0, "JOINTS_0": 1},
    )
    path = tmp_path / "joints_no_weights.glb"
    path.write_bytes(data)
    with pytest.raises(ValueError, match="WEIGHTS_0"):
        gltf.load(path)


def test_an_index_past_the_last_vertex_is_refused_at_load(tmp_path):
    indices = np.array([0, 1, 5], dtype="<u4")  # 5 names a vertex that doesn't exist
    data = _prim_glb(
        extra_accessors=[
            {"bufferView": 1, "componentType": 5125, "count": 3, "type": "SCALAR"}
        ],
        extra_views=[],
        extra_blobs=[indices.tobytes()],
        attributes={"POSITION": 0},
    )
    # Wire "indices" onto the primitive by re-writing the doc directly --
    # ``_prim_glb`` only wires attributes.
    from realmspinner.kernels.geom3d import glbio

    header, gltf_json, chunk = glbio.split_glb(data)
    gltf_json["meshes"][0]["primitives"][0]["indices"] = 1
    data = rebuild_glb(header, gltf_json, chunk)
    path = tmp_path / "bad_index.glb"
    path.write_bytes(data)
    with pytest.raises(ValueError, match="vertex"):
        gltf.load(path)
