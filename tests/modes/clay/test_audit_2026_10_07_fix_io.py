"""The 2026-10-07 Clay audit's import/export kernel findings: clay-07 (kernel
half), clay-27, clay-28, clay-29, clay-58, clay-59, clay-60, clay-61, clay-62
and clay-63. Each test's name is the claim."""

from __future__ import annotations

import io
import json
import struct
import zlib

import numpy as np
import pytest
from PIL import Image

from realmspinner.kernels.geom3d import glbio, glbwrite, gltf
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import glbimport, meshimport, objexport, objimport
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh.elements import OpError

# --- builders -----------------------------------------------------------------


def _glb(doc: dict, binary: bytes = b"") -> bytes:
    j = json.dumps(doc).encode()
    j += b" " * (-len(j) % 4)
    body = struct.pack("<II", len(j), glbio.CHUNK_JSON) + j
    if binary:
        binary += b"\0" * (-len(binary) % 4)
        body += struct.pack("<II", len(binary), glbio.CHUNK_BIN) + binary
    return struct.pack("<III", glbio.GLB_MAGIC, 2, 12 + len(body)) + body


def _png(side: int, colour=(200, 30, 30, 255), fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGBA", (side, side), colour).save(buf, fmt)
    return buf.getvalue()


def _textured_glb(images: list[bytes], mime: str = "image/png") -> bytes:
    """One triangle with a material whose base colour is ``images[0]``."""
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], "<f4").tobytes()
    uv = np.array([[0, 0], [1, 0], [0, 1]], "<f4").tobytes()
    blobs = [pos, uv]
    views = [
        {"buffer": 0, "byteOffset": 0, "byteLength": len(pos)},
        {"buffer": 0, "byteOffset": len(pos), "byteLength": len(uv)},
    ]
    offset = len(pos) + len(uv)
    for data in images:
        pad = -len(data) % 4
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(data)})
        blobs.append(data + b"\0" * pad)
        offset += len(data) + pad
    binary = b"".join(blobs)
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "material": 0}]}
        ],
        "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
        "textures": [{"source": i} for i in range(len(images))],
        "images": [{"bufferView": 2 + i, "mimeType": mime} for i in range(len(images))],
        "accessors": [
            {"componentType": 5126, "type": "VEC3", "count": 3, "bufferView": 0},
            {"componentType": 5126, "type": "VEC2", "count": 3, "bufferView": 1},
        ],
        "bufferViews": views,
        "buffers": [{"byteLength": len(binary)}],
    }
    return _glb(doc, binary)


def _pbr_glb(n_materials: int, side: int) -> bytes:
    """``n_materials`` materials, each with all five texture slots populated."""
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], "<f4").tobytes()
    blobs, images, textures, materials, prims = [pos], [], [], [], []
    views = [{"buffer": 0, "byteOffset": 0, "byteLength": len(pos)}]
    offset = len(pos)
    k = 0
    for m in range(n_materials):
        mat: dict = {"pbrMetallicRoughness": {}}
        for slot in (
            "baseColorTexture",
            "metallicRoughnessTexture",
            "normalTexture",
            "emissiveTexture",
            "occlusionTexture",
        ):
            png = _png(side, (k % 256, (k * 7) % 256, 3, 255))
            k += 1
            pad = -len(png) % 4
            views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(png)})
            blobs.append(png + b"\0" * pad)
            offset += len(png) + pad
            images.append({"bufferView": len(views) - 1, "mimeType": "image/png"})
            textures.append({"source": len(images) - 1})
            ref = {"index": len(textures) - 1}
            target = mat["pbrMetallicRoughness"] if slot in (
                "baseColorTexture",
                "metallicRoughnessTexture",
            ) else mat
            target[slot] = ref
        materials.append(mat)
        prims.append({"attributes": {"POSITION": 0}, "material": m})
    binary = b"".join(blobs)
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": prims}],
        "materials": materials,
        "textures": textures,
        "images": images,
        "accessors": [{"componentType": 5126, "type": "VEC3", "count": 3, "bufferView": 0}],
        "bufferViews": views,
        "buffers": [{"byteLength": len(binary)}],
    }
    return _glb(doc, binary)


def _grid_obj(n: int) -> str:
    """``n`` x ``n`` UV-mapped quads of two triangles, vt shared like any exporter."""
    lines = [f"v {i} {j} 0" for j in range(n + 1) for i in range(n + 1)]
    lines += [f"vt {i / n} {j / n}" for j in range(n + 1) for i in range(n + 1)]
    w = n + 1
    for j in range(n):
        for i in range(n):
            a = j * w + i + 1
            b, c = a + 1, a + w
            d = c + 1
            lines.append(f"f {a}/{a} {b}/{b} {d}/{d}")
            lines.append(f"f {a}/{a} {d}/{d} {c}/{c}")
    return "\n".join(lines) + "\n"


_TRI = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"


def _ply(face: bytes) -> bytes:
    return (
        b"ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\n"
        b"property float z\nelement face 1\nproperty list uchar int vertex_indices\n"
        b"end_header\n0 0 0\n1 0 0\n0 1 0\n" + face + b"\n"
    )


def _stl() -> bytes:
    return b"\0" * 80 + struct.pack("<I", 1) + struct.pack(
        "<12fH", 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0
    )


# --- clay-07 + clay-28: textures through the Clay GLB door ---------------------


def test_a_glb_texture_the_loader_skipped_is_reported_by_the_clay_import() -> None:
    corrupt = b"\x89PNG\r\n\x1a\n" + b"junk" * 10
    data = _textured_glb([corrupt])
    doc, skipped = glbimport.glb_to_claydoc_report(data, "t")
    assert skipped == 1
    assert doc.materials[0].base_color is None
    # The plain door is unchanged: it still returns just the document.
    assert glbimport.glb_to_claydoc(data, "t").materials[0].base_color is None
    # ...and a clean file reports nothing.
    _doc, none_skipped = glbimport.glb_to_claydoc_report(_textured_glb([_png(8)]), "t")
    assert none_skipped == 0


def test_a_glb_texture_over_the_clay_ceiling_is_skipped_and_counted(monkeypatch) -> None:
    monkeypatch.setattr(glbimport, "CLAY_MAX_TEXTURE_PIXELS", 16 * 16 - 1)
    doc, skipped = glbimport.glb_to_claydoc_report(_textured_glb([_png(16)]), "t")
    assert skipped == 1
    assert doc.materials[0].base_color is None


def test_a_4096_square_base_colour_texture_survives_the_clay_import() -> None:
    # 4096 x 4096 = 16,777,216 px, past the viewer's shared 16,000,000 upload
    # ceiling; a texture Clay can hold (it reads and writes 8192 squared) was
    # dropped without a word.
    data = _textured_glb([_png(4096)])
    assert gltf.load(data).skipped_textures == 1  # the shared viewer ceiling is untouched
    doc, skipped = glbimport.glb_to_claydoc_report(data, "t")
    assert skipped == 0
    base = doc.materials[0].base_color
    assert base is not None and base[0] == 4096 and base[1] == 4096


def test_a_pbr_glb_whose_extra_maps_alone_exceed_the_byte_budget_still_imports_into_clay(
    monkeypatch,
) -> None:
    # Six materials x five 1024-square maps = 120 MiB decoded; the budget is
    # lowered to 40 MiB so the fixture stays small. Base colour alone is 24 MiB.
    monkeypatch.setattr(gltf, "MAX_TOTAL_BYTES", 40 * (1 << 20))
    data = _pbr_glb(6, 1024)
    with pytest.raises(ValueError, match="byte budget"):
        gltf.load(data)  # every other caller still decodes all five slots
    doc = glbimport.glb_to_claydoc(data, "t")
    assert len(doc.materials) == 6
    assert all(m.base_color is not None for m in doc.materials)


def test_gltf_load_decodes_only_the_requested_texture_slots() -> None:
    data = _pbr_glb(1, 8)
    full = gltf.load(data)
    assert full.meshes[0][0].material.normal is not None
    only = gltf.load(data, texture_slots=("base_color",))
    material = only.meshes[0][0].material
    assert material.base_color is not None
    assert material.normal is None and material.metallic_roughness is None
    assert material.emissive is None and material.occlusion is None
    with pytest.raises(ValueError, match="texture slot"):
        gltf.load(data, texture_slots=("nonsense",))


# --- clay-60 / clay-61: the loader's texture decoder ---------------------------


def _png_with_text_bomb() -> bytes:
    def chunk(tag: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + tag + body + struct.pack(
            ">I", zlib.crc32(tag + body) & 0xFFFFFFFF
        )

    raw = _png(4)
    sig, rest = raw[:8], raw[8:]
    ihdr_len = struct.unpack(">I", rest[:4])[0]
    ihdr, tail = rest[: 12 + ihdr_len], rest[12 + ihdr_len :]
    bomb = zlib.compress(b"\0" * (8 * 1024 * 1024))
    return sig + ihdr + chunk(b"zTXt", b"k\0\0" + bomb) + tail


def test_a_texture_pillow_rejects_with_valueerror_is_skipped_not_fatal() -> None:
    model = gltf.load(_textured_glb([_png_with_text_bomb()]))
    assert model.skipped_textures == 1
    assert model.meshes[0][0].material.base_color is None


def test_the_byte_budget_still_refuses_the_whole_file() -> None:
    # The budget refusal is the loader's own and must still escape the
    # cosmetic-loss handler that now swallows Pillow's ValueErrors.
    import realmspinner.kernels.geom3d.gltf as g

    old = g.MAX_TOTAL_BYTES
    g.MAX_TOTAL_BYTES = 10
    try:
        with pytest.raises(ValueError, match="byte budget"):
            gltf.load(_textured_glb([_png(8)]))
    finally:
        g.MAX_TOTAL_BYTES = old


def test_a_glb_texture_in_a_non_gltf_image_format_is_skipped_not_decoded() -> None:
    # BMP is a format Pillow opens by content sniffing and glTF does not allow;
    # EPS (a Ghostscript child on load) and WMF are the same shape of hole.
    model = gltf.load(_textured_glb([_png(8, fmt="BMP")], mime="image/png"))
    assert model.skipped_textures == 1
    assert model.meshes[0][0].material.base_color is None
    # The formats glTF does allow still decode.
    ok = gltf.load(_textured_glb([_png(8, fmt="PNG")]))
    assert ok.skipped_textures == 0 and ok.meshes[0][0].material.base_color is not None


def test_jpeg_and_webp_textures_still_decode() -> None:
    for fmt in ("JPEG", "WEBP"):
        buf = io.BytesIO()
        Image.new("RGB", (8, 8), (10, 20, 30)).save(buf, fmt)
        model = gltf.load(_textured_glb([buf.getvalue()], mime="image/" + fmt.lower()))
        assert model.skipped_textures == 0
        assert model.meshes[0][0].material.base_color is not None


# --- clay-27 / clay-29: OBJ ----------------------------------------------------


def test_clays_own_obj_export_of_a_uv_mapped_mesh_reimports_below_the_triangle_ceiling(
    monkeypatch,
) -> None:
    doc = objimport.obj_to_claydoc(_grid_obj(10))  # 121 v + 121 vt, 200 triangles
    # The source imports under a ceiling that its own v + vt lines fit...
    monkeypatch.setattr(objimport, "MAX_VERTEX_LINES", 130)
    objimport.obj_to_claydoc(_grid_obj(10))
    obj_text, mtl = objexport.claydoc_to_obj(doc)
    # ...and a per-corner vt (600 of them) must not make Clay refuse its own file.
    back = objimport.obj_to_claydoc(obj_text, mtl=mtl)
    assert len(back.objects[0].mesh.positions) == len(doc.objects[0].mesh.positions)
    assert np.array_equal(back.objects[0].mesh.uv, doc.objects[0].mesh.uv)


def test_clays_obj_export_shares_identical_texture_coordinates_between_corners() -> None:
    doc = objimport.obj_to_claydoc(_grid_obj(4))
    obj_text, _mtl = objexport.claydoc_to_obj(doc)
    vts = [ln for ln in obj_text.splitlines() if ln.startswith("vt ")]
    assert len(vts) == len(set(vts)) == 25


def test_an_obj_with_more_texture_coordinates_than_face_corners_is_still_bounded(
    monkeypatch,
) -> None:
    monkeypatch.setattr(objimport, "MAX_VERTEX_LINES", 5)
    many_vt = "".join(f"vt {i} 0\n" for i in range(50))
    text = "v 0 0 0\nv 1 0 0\nv 0 1 0\n" + many_vt + "f 1/1 2/2 3/3\n"
    with pytest.raises(OpError, match="vertex/texture-coordinate lines"):
        objimport.obj_to_claydoc(text)


def test_an_obj_texture_coordinate_that_overflows_float32_is_refused_by_name() -> None:
    text = "v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 1e39 0\nvt 0 0\nvt 0 1\nf 1/1 2/2 3/3\n"
    with pytest.raises(OpError, match="vt"):
        objimport.obj_to_claydoc(text)
    # The v component too (the flip subtracts it from 1).
    text_v = "v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 -1e39\nvt 0 0\nvt 0 1\nf 1/1 2/2 3/3\n"
    with pytest.raises(OpError, match="vt"):
        objimport.obj_to_claydoc(text_v)


# --- clay-58 / clay-59: PLY / STL / scale --------------------------------------


@pytest.mark.parametrize("face", [b"3 0 1 7", b"3 0 1 -1", b"3 0 -1 -2"])
def test_a_ply_face_index_outside_the_vertex_list_is_refused_by_name(face: bytes) -> None:
    with pytest.raises(OpError, match="vertex index"):
        meshimport.import_file(_ply(face), ".ply", "x")


def _import(fmt: str, scale: float):
    if fmt == "obj":
        return meshimport.import_file(_TRI.encode(), ".obj", "t", scale=scale)
    if fmt == "stl":
        return meshimport.import_file(_stl(), ".stl", "t", scale=scale)
    if fmt == "ply":
        return meshimport.import_file(_ply(b"3 0 1 2"), ".ply", "t", scale=scale)
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box(), generator="box"))
    return meshimport.import_file(glbwrite.write_glb(bd.to_model(doc)), ".glb", "t", scale=scale)


@pytest.mark.parametrize("fmt", ["obj", "stl", "ply", "glb"])
def test_an_import_scale_of_zero_is_refused_for_every_format(fmt: str) -> None:
    with pytest.raises(OpError, match="scale"):
        _import(fmt, 0.0)


@pytest.mark.parametrize("fmt", ["obj", "stl", "ply", "glb"])
@pytest.mark.parametrize("scale", [float("nan"), float("inf"), -1.0])
def test_a_non_finite_or_negative_import_scale_is_refused_for_every_format(
    fmt: str, scale: float
) -> None:
    with pytest.raises(OpError, match="scale"):
        _import(fmt, scale)


# --- clay-62: the writer --------------------------------------------------------


def _tri_prim() -> gltf.Primitive:
    return gltf.Primitive(
        positions=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], "f4"),
        indices=np.array([0, 1, 2], "u4"),
    )


def test_write_glb_refuses_a_non_finite_node_translation_camera_or_light() -> None:
    node = gltf.Node(name="n", mesh=0, translation=np.array([np.nan, 0, 0]))
    with pytest.raises(ValueError, match="non-finite"):
        glbwrite.write_glb(gltf.Model([node], [0], [[_tri_prim()]], []))

    good = gltf.Node(name="n", mesh=0)
    with pytest.raises(ValueError, match="non-finite"):
        glbwrite.write_glb(
            gltf.Model([good], [0], [[_tri_prim()]], [], cameras=[gltf.Camera(yfov=float("nan"))])
        )
    light_node = gltf.Node(name="l", light=0)
    with pytest.raises(ValueError, match="non-finite"):
        glbwrite.write_glb(
            gltf.Model(
                [good, light_node],
                [0, 1],
                [[_tri_prim()]],
                [],
                lights=[gltf.Light(intensity=float("inf"))],
            )
        )
    # A finite scene still writes.
    assert glbwrite.write_glb(gltf.Model([good], [0], [[_tri_prim()]], [])).startswith(b"glTF")


def test_rebuild_glb_never_writes_a_bare_nan_token() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        glbio.rebuild_glb(struct.pack("<III", glbio.GLB_MAGIC, 2, 0), {"a": float("nan")}, b"")


# --- clay-63 --------------------------------------------------------------------


def test_a_glb_over_the_node_ceiling_is_refused_for_its_node_count_not_an_object_count() -> None:
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{}] * (gltf.MAX_NODES + 1),
    }
    with pytest.raises(OpError) as caught:
        glbimport.glb_to_claydoc(_glb(doc))
    message = str(caught.value)
    assert "nodes" in message and f"{gltf.MAX_NODES + 1:,}" in message
    assert "placing" not in message
