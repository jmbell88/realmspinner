"""The 2026-10-09 Clay audit's import/export kernel findings: clay-24, clay-25,
clay-26, clay-39, clay-40 and clay-41. Each test's name is the claim."""

from __future__ import annotations

import io
import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from realmspinner.core.safeio import atomic
from realmspinner.kernels.geom3d import glbio, gltf
from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.kernels.mesh import glbimport, meshimport, objexport, objimport
from realmspinner.kernels.mesh.elements import OpError

_ROOT = Path(__file__).resolve().parents[3]


def _glb(doc: dict, binary: bytes = b"") -> bytes:
    j = json.dumps(doc).encode()
    j += b" " * (-len(j) % 4)
    body = struct.pack("<II", len(j), glbio.CHUNK_JSON) + j
    if binary:
        binary += b"\0" * (-len(binary) % 4)
        body += struct.pack("<II", len(binary), glbio.CHUNK_BIN) + binary
    return struct.pack("<III", glbio.GLB_MAGIC, 2, 12 + len(body)) + body


_TRI = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], "<f4")


def _tri_glb(nodes: list[dict], extra: dict | None = None, pos: np.ndarray = _TRI) -> bytes:
    binary = pos.astype("<f4").tobytes() + np.array([0, 1, 2], "<u2").tobytes() + b"\0\0"
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": nodes,
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 36},
            {"buffer": 0, "byteOffset": 36, "byteLength": 6},
        ],
        "buffers": [{"byteLength": len(binary)}],
    }
    doc.update(extra or {})
    return _glb(doc, binary)


# --- clay-24 --------------------------------------------------------------------


def test_a_shear_composed_through_a_tiny_parent_scale_is_baked_not_dropped() -> None:
    big = np.array([[0, 0, 0], [1000, 0, 0], [0, 1000, 0]], "<f4")
    angle = math.radians(45)
    quat = [0, 0, math.sin(angle / 2), math.cos(angle / 2)]
    for k in (1.0, 1e-3, 1e-4):
        data = _tri_glb(
            [{"scale": [k, 3 * k, k], "children": [1]}, {"mesh": 0, "rotation": quat}], pos=big
        )
        world = gltf.load(data).nodes[1].world
        obj = glbimport.glb_to_claydoc(data).objects[0]
        placed = m3.compose(obj.translation, obj.rotation, obj.scale) @ np.c_[
            obj.mesh.positions.astype("f8"), np.ones(3)
        ].T
        want = world @ np.c_[big.astype("f8"), np.ones(3)].T
        # Vertex order is not promised by the importer: compare as point sets.
        have = sorted(map(tuple, np.round(placed[:3].T, 6)))
        expect = sorted(map(tuple, np.round(want[:3].T, 6)))
        err = np.abs(np.array(have) - np.array(expect)).max()
        assert err <= 1e-4 * np.abs(want).max(), f"k={k}"


def test_a_sheared_node_matrix_under_a_tiny_scale_is_refused_by_the_loader() -> None:
    s = 1e-4
    # column-major: the second column carries half a scale of x -- shear.
    matrix = [s, 0, 0, 0, 0.5 * s, s, 0, 0, 0, 0, s, 0, 0, 0, 0, 1]
    with pytest.raises(ValueError, match="sheared"):
        gltf.load(_tri_glb([{"mesh": 0, "matrix": matrix}]))


def test_a_shear_free_node_matrix_at_unit_scale_still_loads() -> None:
    matrix = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 0, 0, 1]
    assert gltf.load(_tri_glb([{"mesh": 0, "matrix": matrix}])).nodes[0].translation[0] == 5


# --- clay-25 --------------------------------------------------------------------


def test_an_obj_with_an_o_and_a_g_line_per_part_is_counted_by_objects_built() -> None:
    n = glbimport.MAX_OBJECTS // 2 + 10  # 2n + 1 header lines would pass the ceiling
    lines = ["v 0 0 0", "v 1 0 0", "v 0 1 0"]
    for i in range(n):
        lines += [f"o P{i}", f"g P{i}_g", "f 1 2 3"]
    doc = objimport.obj_to_claydoc("\n".join(lines))
    assert len(doc.objects) == n


# --- clay-26 --------------------------------------------------------------------


def test_import_scale_that_overflows_a_node_translation_is_refused_by_name() -> None:
    data = _tri_glb([{"mesh": 0, "translation": [1e30, 0, 0]}])
    with pytest.raises(OpError, match="scale"):
        meshimport.import_file_report(data, ".glb", scale=1e10)


def test_obj_export_refuses_a_non_finite_baked_position_by_name() -> None:
    doc = glbimport.glb_to_claydoc(_tri_glb([{"mesh": 0}]))
    doc.objects[0].translation = np.array([np.inf, 0, 0], dtype="f8")
    with pytest.raises(ValueError, match="non-finite"):
        objexport.claydoc_to_obj(doc)


# --- clay-39 --------------------------------------------------------------------


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGBA", (4, 4), (200, 30, 30, 255)).save(buf, "PNG")
    return buf.getvalue()


def _textured(ref: dict) -> bytes:
    png = _png()
    pos = _TRI.tobytes()
    uv = np.array([[0, 0], [1, 0], [0, 1]], "<f4").tobytes()
    binary = pos + uv + png
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "material": 0}]}
        ],
        "materials": [{"pbrMetallicRoughness": {"baseColorTexture": ref}}],
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


def test_a_base_colour_texture_on_texcoord_1_is_reported_as_unusable() -> None:
    model = gltf.load(_textured({"index": 0, "texCoord": 1}))
    assert model.skipped_textures == 1
    assert model.meshes[0][0].material.base_color is None


def test_a_base_colour_texture_with_a_texture_transform_is_reported_as_unusable() -> None:
    ref = {"index": 0, "extensions": {"KHR_texture_transform": {"scale": [2, 2]}}}
    model = gltf.load(_textured(ref))
    assert model.skipped_textures == 1
    assert model.meshes[0][0].material.base_color is None


def test_a_plain_base_colour_texture_on_texcoord_0_still_loads() -> None:
    for ref in ({"index": 0}, {"index": 0, "texCoord": 0}):
        model = gltf.load(_textured(ref))
        assert model.skipped_textures == 0
        assert model.meshes[0][0].material.base_color is not None


# --- clay-40 --------------------------------------------------------------------


def test_a_scenes_object_instead_of_an_array_is_refused_by_name() -> None:
    for bad in ({"a": {"nodes": [0]}}, "abc"):
        with pytest.raises(ValueError, match="scenes"):
            gltf.load(_tri_glb([{"mesh": 0}], {"scenes": bad}))


# --- clay-41 --------------------------------------------------------------------


def test_a_textured_materials_kd_keeps_its_tint_in_the_mtl() -> None:
    doc = glbimport.glb_to_claydoc(_tri_glb([{"mesh": 0}]))
    doc.materials[0] = gltf.Material(
        base_color_factor=(0.5, 0.25, 0.125, 1.0), base_color=(1, 1, bytes(4))
    )
    _obj, mtl = objexport.claydoc_to_obj(doc)
    lines = mtl.splitlines()
    assert "Kd 0.5 0.25 0.125" in lines
    assert any(ln.startswith("map_Kd ") for ln in lines)


def test_staged_set_replaces_per_file_so_an_obj_export_is_not_all_or_nothing(
    tmp_path, monkeypatch
) -> None:
    targets = {tmp_path / "a.png": b"png", tmp_path / "a.mtl": b"mtl", tmp_path / "a.obj": b"obj"}
    real = atomic.os.replace
    calls: list[str] = []

    def flaky(src, dst):
        calls.append(Path(dst).name)
        if len(calls) == 2:
            raise OSError("disk yanked")
        return real(src, dst)

    monkeypatch.setattr(atomic.os, "replace", flaky)
    with pytest.raises(OSError):
        atomic.staged_set(targets)
    assert calls[:2] == ["a.png", "a.mtl"]  # the .obj goes last
    assert (tmp_path / "a.png").exists()  # the first replace stood
    assert not (tmp_path / "a.obj").exists()
    assert not list(tmp_path.glob(".*"))  # staged temporaries are cleaned up


def test_the_manual_does_not_promise_an_obj_export_is_all_or_nothing() -> None:
    text = (_ROOT / "docs" / "manual" / "27-clay.md").read_text(encoding="utf-8")
    flat = " ".join(text.split())
    assert "together or not at all" not in flat
    assert "textured material's `Kd` is white" not in flat
