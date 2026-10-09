"""The 2026-10-09 second Clay audit's importer/exporter findings: clay-75,
clay-76, clay-112, clay-113 and clay-116. Each test's name is the claim."""

from __future__ import annotations

import hashlib
import io
import json
import math
import struct
import tracemalloc

import numpy as np
import pytest
from PIL import Image

from realmspinner.kernels.geom3d import glbio, gltf
from realmspinner.kernels.mesh import glbimport, meshimport, objimport
from realmspinner.kernels.mesh.elements import OpError


def _glb(doc: dict, binary: bytes = b"") -> bytes:
    j = json.dumps(doc).encode()
    j += b" " * (-len(j) % 4)
    body = struct.pack("<II", len(j), glbio.CHUNK_JSON) + j
    if binary:
        binary += b"\0" * (-len(binary) % 4)
        body += struct.pack("<II", len(binary), glbio.CHUNK_BIN) + binary
    return struct.pack("<III", glbio.GLB_MAGIC, 2, 12 + len(body)) + body


# --- a closed, outward-wound unit cube -----------------------------------------

_CUBE_V = np.array(
    [[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)], "<f4"
)
# Outward CCW triangles; vertex index = x*4 + y*2 + z.
_CUBE_I = np.array(
    [0, 1, 3, 0, 3, 2, 4, 6, 7, 4, 7, 5, 0, 4, 5, 0, 5, 1,
     2, 3, 7, 2, 7, 6, 0, 2, 6, 0, 6, 4, 1, 5, 7, 1, 7, 3],
    "<u2",
)  # fmt: skip


def _cube_glb(nodes: list[dict]) -> bytes:
    pos = _CUBE_V.tobytes()
    idx = _CUBE_I.tobytes()
    idx += b"\0" * (-len(idx) % 4)
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": nodes,
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 8, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "count": len(_CUBE_I), "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(pos)},
            {"buffer": 0, "byteOffset": len(pos), "byteLength": len(_CUBE_I) * 2},
        ],
        "buffers": [{"byteLength": len(pos) + len(idx)}],
    }
    return _glb(doc, pos + idx)


def _quat_z(deg: float) -> list[float]:
    half = math.radians(deg) / 2
    return [0.0, 0.0, math.sin(half), math.cos(half)]


def _signed_volume(positions: np.ndarray, loops: np.ndarray, starts: np.ndarray) -> float:
    p = positions.astype("f8")
    vol = 0.0
    for f in range(len(starts) - 1):
        c = loops[starts[f] : starts[f + 1]]
        for k in range(1, len(c) - 1):
            vol += float(np.dot(p[c[0]], np.cross(p[c[k]], p[c[k + 1]]))) / 6.0
    return vol


def _placed_volume(obj) -> float:
    """The signed volume of an imported object's mesh as the document places it."""
    from realmspinner.kernels.geom3d import math3d as m3

    world = m3.compose(obj.translation, obj.rotation, obj.scale)
    pos = (world[:3, :3] @ obj.mesh.positions.astype("f8").T).T + world[:3, 3]
    return _signed_volume(pos, obj.mesh.loops, obj.mesh.starts)


# --- clay-75 -------------------------------------------------------------------


def test_a_mirrored_node_under_a_sheared_parent_chain_imports_with_outward_faces() -> None:
    # A non-uniform-scaled parent over a rotated, mirrored child composes into a
    # matrix with shear AND a negative determinant. The 2026-10-09 second
    # audit found the shear branch baked it by hand and never reversed the
    # loops, so every face came in inside-out (signed volume -2 for a unit
    # cube scaled 1x2x1).
    data = _cube_glb(
        [
            {"scale": [1, 2, 1], "children": [1]},
            {"mesh": 0, "rotation": _quat_z(45), "scale": [-1, 1, 1]},
        ]
    )
    obj = glbimport.glb_to_claydoc(data).objects[0]
    assert _placed_volume(obj) == pytest.approx(2.0, rel=1e-5)


def test_a_sheared_node_without_a_mirror_still_imports_with_outward_faces() -> None:
    data = _cube_glb(
        [{"scale": [1, 2, 1], "children": [1]}, {"mesh": 0, "rotation": _quat_z(45)}]
    )
    obj = glbimport.glb_to_claydoc(data).objects[0]
    assert _placed_volume(obj) == pytest.approx(2.0, rel=1e-5)
    # The shear branch bakes the node's world matrix into the positions exactly
    # as it always did: float32 of the float64 product.
    world = gltf.load(data).nodes[1].world
    homo = np.c_[_CUBE_V.astype("f8"), np.ones(8)]
    want = (world @ homo.T).T[:, :3].astype("f4")
    have = obj.mesh.positions
    key = lambda a: sorted(map(tuple, a.tolist()))  # noqa: E731 - vertex order is not promised
    assert key(have) == key(want)


# sha256 over positions + loops + the object's TRS, computed on the tree
# *before* clay-75's change (a throwaway copy of the unfixed module): an
# unsheared import -- a plain mirror under a uniform parent, which is the
# object-level scale of -1 and no baking -- must not move.
_UNSHEARED_GOLDEN = "45eb8b74673f55e7cad46fc599c6e8236305150ec8ffed10ac0c97735ef57ff5"


def _unsheared_digest() -> str:
    data = _cube_glb(
        [
            {"scale": [2, 2, 2], "translation": [1, 2, 3], "children": [1, 2]},
            {"mesh": 0, "rotation": _quat_z(30), "scale": [-1, 1, 1]},
            {"mesh": 0, "translation": [0, 1, 0]},
        ]
    )
    doc = glbimport.glb_to_claydoc(data)
    h = hashlib.sha256()
    for obj in doc.objects:
        h.update(np.ascontiguousarray(obj.mesh.positions).tobytes())
        h.update(np.ascontiguousarray(obj.mesh.loops).tobytes())
        for part in (obj.translation, obj.rotation, obj.scale):
            h.update(np.ascontiguousarray(part, dtype="f8").tobytes())
    return h.hexdigest()


def test_an_unsheared_import_is_byte_identical_to_the_one_before_the_shear_fix() -> None:
    assert _unsheared_digest() == _UNSHEARED_GOLDEN


# --- clay-76 -------------------------------------------------------------------


def _bufferless_glb(accessor: dict, attribute: str) -> bytes:
    """A tiny GLB whose one primitive names a ``bufferView``-less accessor.

    Such an accessor is all zeros by spec, so the file stays a few hundred bytes
    while declaring tens of megabytes of decoded data: exactly the shape a
    charge-after-allocate bug is reachable from.
    """
    pos = np.zeros((3, 3), "<f4").tobytes()
    idx = np.array([0, 1, 2], "<u2").tobytes() + b"\0\0"
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {"primitives": [{"attributes": {"POSITION": 0, attribute: 2}, "indices": 1}]}
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
            accessor,
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(pos)},
            {"buffer": 0, "byteOffset": len(pos), "byteLength": 6},
        ],
        "buffers": [{"byteLength": len(pos) + len(idx)}],
    }
    return _glb(doc, pos + idx)


def _peak_while_loading(data: bytes) -> tuple[float, Exception | None]:
    tracemalloc.start()
    error: Exception | None = None
    try:
        gltf.load(data)
    except ValueError as exc:
        error = exc
    finally:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return peak, error


def test_a_normalized_weights_accessor_is_refused_before_its_divided_copy_is_allocated() -> None:
    # The 2026-10-09 second audit: a 588-byte GLB naming a 40M-row normalized
    # u8 WEIGHTS_0 accessor peaked at 1,373 MiB against the 768 MiB ceiling,
    # because ``/ 255.0`` allocated its copy and only then charged it.
    data = _bufferless_glb(
        {"componentType": 5121, "count": 40_000_000, "type": "VEC4", "normalized": True},
        "WEIGHTS_0",
    )
    peak, error = _peak_while_loading(data)
    assert error is not None and "byte budget" in str(error)
    assert peak <= gltf.MAX_TOTAL_BYTES, f"peaked at {peak / 2**20:.0f} MiB"


def test_a_float_weights_accessor_is_refused_before_its_renormalised_copy_is_allocated() -> None:
    # ``weights / total`` built a full-size copy that was charged afterwards. The
    # largest float accessor one view may declare (256 MiB) plus ``_astype``'s
    # copy is 512 MiB; the renormalised third copy must be refused, not built.
    data = _bufferless_glb(
        {"componentType": 5126, "count": 16_777_216, "type": "VEC4"}, "WEIGHTS_0"
    )
    peak, error = _peak_while_loading(data)
    assert error is not None and "byte budget" in str(error)
    assert peak <= gltf.MAX_TOTAL_BYTES, f"peaked at {peak / 2**20:.0f} MiB"


def test_a_signed_normalized_normal_accessor_is_refused_before_its_divided_copy_is_allocated() -> (
    None
):
    # ``decoded()``'s divide (and, for signed, the clamp on top of it) is a second
    # same-size array beyond the one ``_astype`` charged.
    data = _bufferless_glb(
        {"componentType": 5120, "count": 50_000_000, "type": "VEC3", "normalized": True},
        "NORMAL",
    )
    peak, error = _peak_while_loading(data)
    assert error is not None and "byte budget" in str(error)
    assert peak <= gltf.MAX_TOTAL_BYTES, f"peaked at {peak / 2**20:.0f} MiB"


def test_normalized_accessors_still_decode_to_the_same_values() -> None:
    # The charge moved before the allocation; the arithmetic must not have moved.
    pos = np.zeros((3, 3), "<f4").tobytes()
    idx = np.array([0, 1, 2], "<u2").tobytes() + b"\0\0"
    nrm = np.array([[-128, 0, 127]] * 3, "<i1").tobytes() + b"\0" * 3
    w = np.array([[255, 0, 0, 0], [128, 127, 0, 0], [0, 0, 0, 0]], "u1").tobytes()
    binary = pos + idx + nrm + w
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "NORMAL": 2, "WEIGHTS_0": 3, "JOINTS_0": 4},
                        "indices": 1,
                    }
                ]
            }
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
            {"bufferView": 2, "componentType": 5120, "count": 3, "type": "VEC3",
             "normalized": True},
            {"bufferView": 3, "componentType": 5121, "count": 3, "type": "VEC4",
             "normalized": True},
            {"componentType": 5121, "count": 3, "type": "VEC4"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 36},
            {"buffer": 0, "byteOffset": 36, "byteLength": 6},
            {"buffer": 0, "byteOffset": 44, "byteLength": 9},
            {"buffer": 0, "byteOffset": 56, "byteLength": 12},
        ],
        "buffers": [{"byteLength": len(binary)}],
    }  # fmt: skip
    prim = gltf.load(_glb(doc, binary)).meshes[0][0]
    assert prim.normals[0].tolist() == pytest.approx([-1.0, 0.0, 1.0])
    assert prim.weights[0].tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert prim.weights[1].tolist() == pytest.approx([128 / 255, 127 / 255, 0, 0])
    assert prim.weights[2].tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0])  # zero-sum pinned


# --- clay-112 ------------------------------------------------------------------


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGBA", (4, 4), (200, 30, 30, 255)).save(buf, "PNG")
    return buf.getvalue()


def _textured_materials(materials: list[dict], primitive_materials: list[int]) -> bytes:
    """One image (source 0) behind texture 0, *materials*, and one triangle
    primitive per entry of *primitive_materials* naming that material."""
    png = _png()
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], "<f4").tobytes()
    uv = np.array([[0, 0], [1, 0], [0, 1]], "<f4").tobytes()
    binary = pos + uv + png
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {
                "primitives": [
                    {"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "material": m}
                    for m in primitive_materials
                ]
            }
        ],
        "materials": materials,
        "textures": [{"source": 0}],
        "images": [{"bufferView": 2, "mimeType": "image/png"}],
        "accessors": [
            {"componentType": 5126, "type": "VEC3", "count": 3, "bufferView": 0},
            {"componentType": 5126, "type": "VEC2", "count": 3, "bufferView": 1},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(pos)},
            {"buffer": 0, "byteOffset": len(pos), "byteLength": len(uv)},
            {"buffer": 0, "byteOffset": len(pos) + len(uv), "byteLength": len(png)},
        ],
        "buffers": [{"byteLength": len(binary)}],
    }
    return _glb(doc, binary)


def _material(ref: dict) -> dict:
    return {"pbrMetallicRoughness": {"baseColorTexture": ref}}


_TRANSFORM = {"KHR_texture_transform": {"scale": [2, 2]}}


def test_two_materials_sharing_one_unusable_image_count_one_lost_texture_and_an_unused_material_counts_none() -> (  # noqa: E501
    None
):
    # The 2026-10-09 second audit, the fix of clay-39: the count was per material
    # reference, so three materials asking for one image on a second UV set
    # reported three lost textures when one image was lost.
    shared = [
        _material({"index": 0, "texCoord": 1}),
        _material({"index": 0, "texCoord": 1}),
        _material({"index": 0, "extensions": _TRANSFORM}),
    ]
    model = gltf.load(_textured_materials(shared, [0, 1, 2]))
    assert model.skipped_textures == 1
    assert glbimport.glb_to_claydoc_report(_textured_materials(shared, [0, 1, 2]))[1] == 1

    # A material no primitive uses loses nothing the model could have shown.
    used_fine_unused_bad = [_material({"index": 0}), _material({"index": 0, "texCoord": 1})]
    data = _textured_materials(used_fine_unused_bad, [0])
    assert gltf.load(data).skipped_textures == 0
    doc, skipped = glbimport.glb_to_claydoc_report(data)
    assert skipped == 0
    assert doc.materials[0].base_color is not None


def test_a_used_material_on_a_second_uv_set_is_still_counted_once() -> None:
    data = _textured_materials([_material({"index": 0, "texCoord": 1})], [0])
    assert gltf.load(data).skipped_textures == 1


def test_a_float_or_false_texcoord_does_not_hide_a_texture_transform() -> None:
    # ``0.0`` and ``false`` are not ints, and the old test returned before it
    # ever looked at ``extensions`` -- so the transformed map was drawn
    # untransformed with no word.
    for coord in (0.0, False):
        ref = {"index": 0, "texCoord": coord, "extensions": _TRANSFORM}
        model = gltf.load(_textured_materials([_material(ref)], [0]))
        assert model.skipped_textures == 1, repr(coord)
        assert model.meshes[0][0].material.base_color is None, repr(coord)


def test_a_float_zero_texcoord_without_a_transform_still_loads() -> None:
    for coord in (0.0, False, 0):
        ref = {"index": 0, "texCoord": coord}
        model = gltf.load(_textured_materials([_material(ref)], [0]))
        assert model.skipped_textures == 0, repr(coord)
        assert model.meshes[0][0].material.base_color is not None, repr(coord)


def test_the_lost_texture_notice_names_a_second_uv_set_and_a_texture_transform() -> None:
    notice = glbimport.skipped_textures_notice(1)
    assert notice is not None
    assert "second UV set" in notice and "texture transform" in notice
    assert glbimport.skipped_textures_notice(0) is None


# --- clay-113 ------------------------------------------------------------------


def _soup(triangles: int) -> str:
    lines = []
    for i in range(triangles):
        lines += [f"v {i} 0 0", f"v {i} 1 0", f"v {i} 0 1"]
    lines += [f"f {3 * i + 1} {3 * i + 2} {3 * i + 3}" for i in range(triangles)]
    return "\n".join(lines) + "\n"


def test_an_unwelded_obj_triangle_soup_under_the_triangle_ceiling_is_not_refused_for_vertex_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The 2026-10-09 second audit: ``MAX_VERTEX_LINES`` equalled the triangle
    # ceiling, so a 700,000-triangle unwelded soup (3 v lines per triangle,
    # every one referenced) was refused for "unreferenced" data it did not
    # have. Scaled down here: 400 triangles, a ceiling of 1,000 vertex lines.
    monkeypatch.setattr(objimport, "MAX_VERTEX_LINES", 1_000)
    doc = objimport.obj_to_claydoc(_soup(400))
    assert sum(len(o.mesh.starts) - 1 for o in doc.objects) == 400


def test_an_obj_with_vertex_lines_no_face_can_reference_is_still_refused_with_the_real_numbers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(objimport, "MAX_VERTEX_LINES", 1_000)
    text = "".join(f"v {i} 0 0\n" for i in range(1_500)) + "v 0 1 0\nv 0 0 1\nf 1 2 3\n"
    with pytest.raises(OpError) as raised:
        objimport.obj_to_claydoc(text)
    message = str(raised.value)
    # It names the real cause (lines no face can reference), not just a total.
    assert "1,502" in message and "3 corners" in message and "1,000" in message
    assert "no face" in message


# --- clay-116 ------------------------------------------------------------------


def test_import_file_report_docstring_does_not_claim_glb_scale_leaves_node_transforms_alone() -> (
    None
):
    doc = meshimport.import_file_report.__doc__ or ""
    # The code conjugates each object's placement by the same matrix as its
    # positions; the docstring used to say the opposite.
    assert "rather than composed into the object's transform" not in " ".join(doc.split())
    flat = " ".join(doc.split())
    assert "conjugated" in flat and "world placement" in flat
