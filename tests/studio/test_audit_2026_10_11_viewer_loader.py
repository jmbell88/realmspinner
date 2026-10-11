"""Regression tests for the 2026-10-11 audit's viewer and loader findings:
create-11, create-24, create-25, create-26 (the glTF loader and the viewer's
draw path) and clay-10, clay-11, clay-30 (the loaders' resource ceilings).
create-23 (the viewer import pin) is in ``tests/test_viewer_imports.py``.
"""

from __future__ import annotations

import base64
import io
import json
import struct
import tracemalloc

import numpy as np
import pytest

from realmspinner.kernels.geom3d import gltf
from realmspinner.kernels.mesh import glbimport, objimport
from realmspinner.kernels.mesh.elements import OpError
from realmspinner.studio import viewer
from realmspinner.studio.viewer import glctx
from realmspinner.studio.viewer import scene as scenelib
from realmspinner.studio.viewer.camera import Camera
from realmspinner.studio.viewer.render import WIRE_TINT, Renderer


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


def _tri_doc(nodes: list[dict], scenes: list[dict]) -> tuple[dict, bytes]:
    pos = _TRI.tobytes()
    idx = np.array([0, 1, 2], dtype="<u2").tobytes() + b"\0\0"
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": scenes,
        "nodes": nodes,
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
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


def _png(color: tuple[int, int, int, int], size: int = 16) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGBA", (size, size), color).save(buf, "PNG")
    return buf.getvalue()


def _uri(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


# --- create-11: the scene's root list had no ceiling -------------------------


def test_a_scene_naming_one_node_a_million_times_is_refused_or_deduplicated_before_update_world_allocates():  # noqa: E501
    # The 2026-10-11 audit, finding create-11: one node named 1.5 M times in a
    # 3 MB GLB loaded in 2.0 s and 481 MiB, and every ``update_world`` after it
    # (each pose edit, each onion-ghost frame) paid 1.9 s again.
    doc, binary = _tri_doc([{"mesh": 0}], [{"nodes": [0] * (gltf.MAX_NODES + 1)}])
    try:
        model = gltf.load(_glb(doc, binary))
    except ValueError:
        return  # refused: the file is the problem, not the loader
    assert len(model.roots) <= 1, "a repeated root was kept once per repetition"


def test_a_root_list_past_the_node_ceiling_is_refused_before_it_is_walked(monkeypatch):
    monkeypatch.setattr(gltf, "MAX_NODES", 5)
    doc, binary = _tri_doc([{"mesh": 0}], [{"nodes": list(range(9))}])
    with pytest.raises(ValueError, match="root"):
        gltf.load(_glb(doc, binary))


def test_a_repeated_root_is_kept_once_in_first_seen_order():
    doc, binary = _tri_doc(
        [{"mesh": 0}, {"mesh": 0}, {"mesh": 0}], [{"nodes": [1, 0, 1, 0, 2, 2]}]
    )
    assert gltf.load(_glb(doc, binary)).roots == [1, 0, 2]


# --- create-24: only the base-colour slot checked its mapping ----------------


def _five_slot_doc(**refs: dict) -> tuple[dict, bytes]:
    """One triangle whose material carries a distinct image on each named slot.

    ``refs`` maps a slot (``mr``, ``normal``, ``emissive``, ``occlusion``,
    ``base``) to the extra keys its texture reference carries (``texCoord``,
    ``extensions``). Every slot gets its own image so ``skipped_textures``,
    which counts image sources, can tell them apart.
    """
    doc, binary = _tri_doc([{"mesh": 0}], [{"nodes": [0]}])
    colours = {"base": 10, "mr": 20, "normal": 30, "emissive": 40, "occlusion": 50}
    names = list(colours)
    doc["images"] = [{"uri": _uri(_png((colours[n], 0, 0, 255)))} for n in names]
    doc["textures"] = [{"source": i} for i in range(len(names))]
    ref = {n: {"index": i, **refs.get(n, {})} for i, n in enumerate(names)}
    doc["materials"] = [
        {
            "pbrMetallicRoughness": {
                "baseColorTexture": ref["base"],
                "metallicRoughnessTexture": ref["mr"],
            },
            "normalTexture": ref["normal"],
            "emissiveTexture": ref["emissive"],
            "occlusionTexture": ref["occlusion"],
        }
    ]
    doc["meshes"][0]["primitives"][0]["material"] = 0
    return doc, binary


def test_a_normal_or_occlusion_map_on_another_uv_set_is_counted_as_skipped():
    # The 2026-10-11 audit, finding create-24: ``_texture_needs_unsupported_
    # mapping`` covered ``baseColorTexture`` alone. These four went straight to
    # ``texture()`` and were sampled through TEXCOORD_0, untransformed, with
    # nothing in ``skipped_textures`` to say the draw was wrong.
    doc, binary = _five_slot_doc(
        mr={"texCoord": 1},
        normal={"texCoord": 1},
        emissive={"texCoord": 1},
        occlusion={"texCoord": 1},
    )
    model = gltf.load(_glb(doc, binary))
    material = model.meshes[0][0].material
    assert material.base_color is not None, "the supported slot still decodes"
    assert material.metallic_roughness is None
    assert material.normal is None
    assert material.emissive is None
    assert material.occlusion is None
    assert model.skipped_textures == 4


@pytest.mark.parametrize("slot", ["mr", "normal", "emissive", "occlusion"])
def test_a_transformed_map_in_any_slot_is_counted_as_skipped(slot):
    extensions = {"extensions": {"KHR_texture_transform": {"scale": [2, 2]}}}
    doc, binary = _five_slot_doc(**{slot: extensions})
    model = gltf.load(_glb(doc, binary))
    assert model.skipped_textures == 1
    attr = {
        "mr": "metallic_roughness",
        "normal": "normal",
        "emissive": "emissive",
        "occlusion": "occlusion",
    }[slot]
    assert getattr(model.meshes[0][0].material, attr) is None


def test_maps_on_the_first_uv_set_still_decode_in_every_slot():
    doc, binary = _five_slot_doc()
    model = gltf.load(_glb(doc, binary))
    material = model.meshes[0][0].material
    assert model.skipped_textures == 0
    for attr in ("base_color", "metallic_roughness", "normal", "emissive", "occlusion"):
        assert getattr(material, attr) is not None, attr


# --- create-25: the wire overlay took the material's maps and alpha ----------


@pytest.fixture(scope="module")
def renderer(gl):
    r = Renderer(gl)
    yield r
    r.release()


@pytest.fixture
def viewport(gl):
    # One sample: an antialiased edge is a blend of the wire and its
    # neighbour, and this test reads the wire's own pixels.
    vp = glctx.Viewport(gl, (128, 128), samples=1)
    yield vp
    vp.release()


def _textured_quad(colour: tuple[int, int, int, int], alpha_mode: str = "OPAQUE") -> bytes:
    quad = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype="<f4")
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype="<f4")
    nrm = np.tile(np.array([0, 0, 1], dtype="<f4"), (4, 1))
    idx = np.array([0, 1, 2, 0, 2, 3], dtype="<u2")
    chunks = [quad.tobytes(), nrm.tobytes(), uv.tobytes(), idx.tobytes()]
    views, offset = [], 0
    for chunk in chunks:
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(chunk)})
        offset += len(chunk) + (-len(chunk) % 4)
    binary = b"".join(c + b"\0" * (-len(c) % 4) for c in chunks)
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2},
                        "indices": 3,
                        "material": 0,
                    }
                ]
            }
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": 4, "type": "VEC2"},
            {"bufferView": 3, "componentType": 5123, "count": 6, "type": "SCALAR"},
        ],
        "bufferViews": views,
        "buffers": [{"byteLength": len(binary)}],
        "images": [{"uri": _uri(_png(colour, 4))}],
        "textures": [{"source": 0}],
        "materials": [
            {
                "alphaMode": alpha_mode,
                "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
            }
        ],
    }
    return _glb(doc, binary)


def _frame(gl, renderer, viewport, data: bytes, **kwargs) -> np.ndarray:
    model = gltf.load(data)
    gpu = scenelib.GpuModel(gl, model)
    try:
        camera = Camera()
        lo, hi = model.bounds()
        camera.frame(lo, hi)
        renderer.draw(
            viewport,
            camera,
            gpu,
            model_matrix=scenelib.placement(model),
            show_grid=False,
            **kwargs,
        )
        return viewport.read_rgba()[..., :3].astype(float)
    finally:
        gpu.release()


def _cpu_aces_srgb(linear: np.ndarray) -> np.ndarray:
    inp = np.array(
        [[0.59719, 0.35458, 0.04823], [0.07600, 0.90834, 0.01566], [0.02840, 0.13383, 0.83777]]
    )
    out = np.array(
        [
            [1.60475, -0.53108, -0.07367],
            [-0.10208, 1.10813, -0.00605],
            [-0.00327, -0.07276, 1.07602],
        ]
    )
    v = inp @ (linear / 0.6)
    v = (v * (v + 0.0245786) - 0.000090537) / (v * (0.983729 * v + 0.4329510) + 0.238081)
    v = np.clip(out @ v, 0.0, 1.0)
    return np.where(v <= 0.0031308, v * 12.92, np.power(v, 0.41666) * 1.055 - 0.055)


def test_the_wire_overlay_is_one_colour_over_a_textured_material(gl, renderer, viewport):
    # The 2026-10-11 audit, finding create-25: the overlay drew through the
    # material's own program with its base-colour map bound, so the tint was
    # multiplied by the texture and alpha was forced to 1.0 for OPAQUE -- the
    # documented 0.75 never applied on any textured mesh, which is every
    # trellis output. An edge pixel is now ``0.75 * wire + 0.25 * fill``
    # whatever the texture is.
    wire = _cpu_aces_srgb(np.array(WIRE_TINT[:3])) * 255.0
    for colour in ((250, 250, 250, 255), (230, 60, 40, 255)):
        data = _textured_quad(colour)
        fill = _frame(gl, renderer, viewport, data)
        over = _frame(gl, renderer, viewport, data, wire_overlay=True)
        edge = (np.abs(over - fill) > 0.5).any(axis=-1)
        assert edge.sum() > 20, "the overlay drew no lines at all"
        expected = WIRE_TINT[3] * wire + (1.0 - WIRE_TINT[3]) * fill[edge]
        assert np.abs(over[edge] - expected).max() <= 3.0, colour


def test_the_wire_overlay_survives_a_masked_material_with_a_transparent_texel(
    gl, renderer, viewport
):
    # MASK discarded the overlay's fragments wherever the *texture's* alpha was
    # under the cutoff, so the lines vanished over exactly the cut-out regions.
    data = _textured_quad((200, 200, 200, 0), alpha_mode="MASK")
    fill = _frame(gl, renderer, viewport, data)
    over = _frame(gl, renderer, viewport, data, wire_overlay=True)
    assert (np.abs(over - fill) > 0.5).any(axis=-1).sum() > 20


def test_the_wire_program_compiles_for_a_skinned_primitive(gl, renderer):
    # The overlay of a rigged model has to follow the pose, so the solid wire
    # program carries the SKINNED variant too.
    program = renderer.programs.get("wire", ("SKINNED",))
    assert "u_joints" in program and "u_color" in program


# --- create-26: the package docstring ----------------------------------------


def test_the_viewer_package_docstring_does_not_claim_glb_loading():
    doc = viewer.__doc__ or ""
    assert "GLB loading" not in doc.splitlines()[0]
    assert "kernels/geom3d/gltf.py" in doc


# --- clay-10: every image entry decoded and charged only its own bytes -------


def _count_image_opens(monkeypatch) -> list[int]:
    from PIL import Image

    real = Image.open
    calls = [0]

    def counting(*args, **kwargs):
        calls[0] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(Image, "open", counting)
    return calls


def test_many_image_entries_over_one_buffer_view_decode_once_or_are_refused(monkeypatch):
    # The 2026-10-11 audit, finding clay-10: the decode cache was keyed by the
    # image's *index*, so 100,000 ``images`` entries over one 16x16 PNG
    # bufferView decoded 100,000 times (7.8 s from a 9.6 MB GLB) and charged
    # about 1 KB each, never reaching ``MAX_TOTAL_BYTES``.
    n = 400
    png = _png((9, 99, 199, 255))
    doc = {
        "asset": {"version": "2.0"},
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(png)}],
        "buffers": [{"byteLength": len(png)}],
        "images": [{"bufferView": 0} for _ in range(n)],
        "textures": [{"source": i} for i in range(n)],
        "materials": [
            {"pbrMetallicRoughness": {"baseColorTexture": {"index": i}}} for i in range(n)
        ],
    }
    opens = _count_image_opens(monkeypatch)
    try:
        gltf.load(_glb(doc, png))
    except ValueError:
        return
    assert opens[0] <= 1, f"{opens[0]} decodes of one image"


def test_many_data_uri_entries_with_one_payload_decode_once(monkeypatch):
    n = 400
    uri = _uri(_png((9, 99, 199, 255)))
    doc = {
        "asset": {"version": "2.0"},
        "images": [{"uri": uri} for _ in range(n)],
        "textures": [{"source": i} for i in range(n)],
        "materials": [
            {"pbrMetallicRoughness": {"baseColorTexture": {"index": i}}} for i in range(n)
        ],
    }
    opens = _count_image_opens(monkeypatch)
    try:
        gltf.load(_glb(doc))
    except ValueError:
        return
    assert opens[0] <= 1


def test_distinct_images_past_the_decode_ceiling_are_refused_before_decoding(monkeypatch):
    monkeypatch.setattr(gltf, "MAX_IMAGE_DECODES", 3, raising=False)
    n = 8
    doc = {
        "asset": {"version": "2.0"},
        "images": [{"uri": _uri(_png((i * 10, 0, 0, 255)))} for i in range(n)],
        "textures": [{"source": i} for i in range(n)],
        "materials": [
            {"pbrMetallicRoughness": {"baseColorTexture": {"index": i}}} for i in range(n)
        ],
    }
    opens = _count_image_opens(monkeypatch)
    with pytest.raises(ValueError, match="images"):
        gltf.load(_glb(doc))
    assert opens[0] <= 3


def test_two_materials_over_one_buffer_view_share_one_decoded_texture():
    # Two image entries over one span are one pixels object, which is what the
    # GPU upload de-duplicates on (its key is the identity of that tuple).
    png = _png((1, 2, 3, 255))
    doc = {
        "asset": {"version": "2.0"},
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(png)}],
        "buffers": [{"byteLength": len(png)}],
        "images": [{"bufferView": 0}, {"bufferView": 0}],
        "textures": [{"source": 0}, {"source": 1}],
        "materials": [
            {"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}},
            {"pbrMetallicRoughness": {"baseColorTexture": {"index": 1}}},
        ],
    }
    reader = gltf._Reader(json.loads(json.dumps(doc)), png, n_lights=0)
    a = reader.material(doc["materials"][0]).base_color
    b = reader.material(doc["materials"][1]).base_color
    assert a is not None and a is b


# --- clay-30: a node's children list had no entry ceiling --------------------


def _many_children_doc(per_node: int, nodes: int = 3) -> tuple[dict, bytes]:
    doc, binary = _tri_doc(
        [{"mesh": 0, "children": [0] * per_node} for _ in range(nodes)],
        [{"nodes": [0]}],
    )
    return doc, binary


def test_a_node_naming_millions_of_children_is_refused_before_the_walk(monkeypatch):
    # The 2026-10-11 audit, finding clay-30: child references were bounded only
    # by file size (a 100 MiB file of about 50 M entries cost 9-17 s and about
    # 0.4 GB), and the existing guard only de-duplicated.
    monkeypatch.setattr(gltf, "MAX_NODES", 10)
    doc, binary = _many_children_doc(per_node=200)  # 600 refs against a cap of 10 x k

    def _must_not_run(*args, **kwargs):
        raise AssertionError("a node was decoded before the child-reference ceiling")

    monkeypatch.setattr(gltf._Reader, "node", _must_not_run)
    with pytest.raises(ValueError, match="child"):
        gltf.load(_glb(doc, binary))


def test_the_clay_glb_pre_check_refuses_the_same_child_flood(monkeypatch):
    monkeypatch.setattr(gltf, "MAX_NODES", 10)
    doc, binary = _many_children_doc(per_node=200)
    with pytest.raises(OpError, match="child"):
        glbimport._declared_counts(_glb(doc, binary))


def test_an_ordinary_hierarchy_is_far_under_the_child_reference_ceiling():
    doc, binary = _tri_doc(
        [{"mesh": 0, "children": [1, 2]}, {"mesh": 0}, {"mesh": 0}], [{"nodes": [0]}]
    )
    assert len(gltf.load(_glb(doc, binary)).reached) == 3
    glbimport._declared_counts(_glb(doc, binary))


# --- clay-11: OBJ ``v`` lines cost minutes and gigabytes inside the ceilings --


def _soup(triangles: int) -> str:
    rows = []
    for t in range(triangles):
        a, b = (t % 100) * 0.01, (t // 100) * 0.01
        rows.append(f"v {a:.2f} {b:.2f} 0\nv {a + 0.5:.2f} {b:.2f} 0\nv {a:.2f} {b + 0.5:.2f} 0\n")
    rows.extend(f"f {3 * t + 1} {3 * t + 2} {3 * t + 3}\n" for t in range(triangles))
    return "".join(rows)


def test_obj_import_at_the_vertex_ceiling_stays_under_the_memory_budget():
    # The 2026-10-11 audit, finding clay-11: every ``v`` line went through
    # ``np.array``, a matrix product, two ``isfinite`` calls and an ``errstate``
    # and was kept as a Python list of floats -- 427 B of peak per ``v`` line
    # (900 MB for a 700,000-triangle soup; the 8 M-line ceiling extrapolated to
    # 3.3 GB). Measured after the fix: 106 B per line under tracemalloc, and
    # 492 MB of process peak for the whole 2 M-triangle, 8 M-line ceiling (the
    # file itself is 115 MB), so the 2 M-line unreferenced ceiling stands.
    triangles = 12_000
    text = _soup(triangles)
    lines = 3 * triangles
    tracemalloc.start()
    try:
        doc = objimport.obj_to_claydoc(text)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(doc.objects[0].mesh.positions) == lines
    assert peak / lines < 200, f"{peak / lines:.0f} B of peak per v line"


def test_an_obj_v_line_that_overflows_float32_after_the_scale_is_still_refused_naming_it():
    text = "v 0 0 0\nv 1e30 0 0\nv 0 1 0\nf 1 2 3\n"
    with pytest.raises(OpError, match="v 1e30 0 0"):
        objimport.obj_to_claydoc(text, scale=1e10)


def test_obj_vertices_keep_first_use_order_per_object_and_the_axis_transform():
    text = (
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n"
        "o first\nf 3 1 2\n"
        "o second\nf 4 2 1\n"
    )
    doc = objimport.obj_to_claydoc(text, up="z", scale=2.0)
    first, second = (o.mesh for o in doc.objects)
    # (x, y, z) -> (x, z, -y), then x2; local order is the order of first use.
    assert first.positions.tolist() == [[0, 0, -2], [0, 0, 0], [2, 0, 0]]
    assert second.positions.tolist() == [[0, 2, 0], [2, 0, 0], [0, 0, 0]]
    assert first.loops.tolist() == [0, 1, 2]
    assert second.loops.tolist() == [0, 1, 2]
