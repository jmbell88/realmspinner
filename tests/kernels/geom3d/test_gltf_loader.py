"""The GLB loader, against a hand-built asset and (when present) a real one.

The hand-built one is the pin: it is the only way to assert what a skin
*should* decode to without a 30 MB fixture in the repo. The real-asset tests
need one particular finished, rigged job still on disk -- they are what catch a
loader that is self-consistently wrong. `assets/` is gitignored and a job is
routinely pruned, so the skip names *this asset*, not the directory: a prune
then degrades to a skip rather than to a test that silently never runs again
while claiming it only wants the main checkout.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
import trimesh

from realmspinner.kernels.geom3d import gltf
from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.kernels.geom3d.glbio import rebuild_glb

REAL_JOB = "44593039ccee"
REAL_MESH = Path(f"assets/{REAL_JOB}/model.glb")
REAL_RIG = Path(f"assets/{REAL_JOB}/rig.glb")
needs_real = pytest.mark.skipif(
    not (REAL_MESH.exists() and REAL_RIG.exists()),
    reason=f"needs the rigged asset assets/{REAL_JOB}/ (pruned, or not this checkout)",
)


def _glb(gltf_json: dict, binary: bytes) -> bytes:
    """Wrap a JSON chunk and a BIN chunk as a GLB, the way an exporter would."""
    binary += b"\x00" * (-len(binary) % 4)
    chunk = struct.pack("<II", len(binary), 0x004E4942) + binary
    header = struct.pack("<III", 0x46546C67, 2, 0)
    return rebuild_glb(header, gltf_json, chunk)


@pytest.fixture
def skinned_glb(tmp_path):
    """A two-triangle strip bound to two joints, one above the other.

    Small enough to assert on by hand: vertex 0 belongs entirely to the root
    joint, vertex 2 entirely to the tip, so rotating the tip must move exactly
    one of them.
    """
    positions = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype="<f4"
    )
    normals = np.tile(np.array([0, 0, 1], dtype="<f4"), (4, 1))
    uvs = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype="<f4")
    joints = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], dtype="<u1")
    weights = np.tile(np.array([1, 0, 0, 0], dtype="<f4"), (4, 1))
    indices = np.array([0, 1, 2, 1, 3, 2], dtype="<u4")
    # Column-major, as glTF stores them: the tip joint sits at y=1, so its
    # inverse bind matrix translates by -1.
    ibm = np.array(
        [np.eye(4, dtype="<f4").T, m3.translation(m3.vec3(0, -1, 0)).T.astype("<f4")]
    )

    blobs = [
        positions.tobytes(),
        normals.tobytes(),
        uvs.tobytes(),
        joints.tobytes(),
        weights.tobytes(),
        indices.tobytes(),
        ibm.tobytes(),
    ]
    views, binary = [], b""
    for blob in blobs:
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(blob)})
        binary += blob + b"\x00" * (-len(blob) % 4)

    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0, 1]}],
        "nodes": [
            {"name": "mesh_node", "mesh": 0, "skin": 0},
            {"name": "root", "children": [2]},
            {"name": "tip", "translation": [0, 1, 0]},
        ],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": 0,
                            "NORMAL": 1,
                            "TEXCOORD_0": 2,
                            "JOINTS_0": 3,
                            "WEIGHTS_0": 4,
                        },
                        "indices": 5,
                        "material": 0,
                    }
                ]
            }
        ],
        "skins": [{"joints": [1, 2], "inverseBindMatrices": 6}],
        "materials": [
            {
                "name": "test",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.8, 0.2, 0.1, 1.0],
                    "metallicFactor": 0.25,
                    "roughnessFactor": 0.75,
                },
                "doubleSided": True,
            }
        ],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": views,
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": 4, "type": "VEC2"},
            {"bufferView": 3, "componentType": 5121, "count": 4, "type": "VEC4"},
            {"bufferView": 4, "componentType": 5126, "count": 4, "type": "VEC4"},
            {"bufferView": 5, "componentType": 5125, "count": 6, "type": "SCALAR"},
            {"bufferView": 6, "componentType": 5126, "count": 2, "type": "MAT4"},
        ],
    }
    path = tmp_path / "skinned.glb"
    path.write_bytes(_glb(doc, binary))
    return path


# --- geometry ---------------------------------------------------------------


def test_a_plain_mesh_loads_with_its_indices_and_normals(tmp_path):
    path = tmp_path / "box.glb"
    trimesh.creation.box(extents=(1.0, 2.0, 3.0)).export(path)
    model = gltf.load(path)

    (_, prims) = model.mesh_instances()[0]
    prim = prims[0]
    assert prim.positions.shape[1] == 3
    assert prim.positions.dtype == np.float32
    assert len(prim.indices) == model.triangle_count * 3
    assert prim.indices.dtype == np.uint32


def test_triangle_count_is_memoized_like_vertex_count(tmp_path):
    """The 2026-09-18 audit, finding create-03: ``triangle_count`` summed
    every primitive's indices on every call, while ``vertex_count`` right
    beside it was memoized for B18 on the same reasoning -- the primitives
    are immutable after load, and the inspector asks for this every frame.
    Mutating ``model.meshes`` after the first read and re-reading proves the
    cache, not just the recomputed sum: an unmemoized property would notice
    the mutation and return a different count."""
    path = tmp_path / "box.glb"
    trimesh.creation.box(extents=(1.0, 2.0, 3.0)).export(path)
    model = gltf.load(path)

    first = model.triangle_count
    assert first > 0

    (_, prims) = model.mesh_instances()[0]
    model.meshes[0] = list(prims) + list(prims)  # would double the sum if unmemoized

    assert model.triangle_count == first
    assert getattr(model, "_triangle_count", None) == first


def test_a_root_transform_is_not_discarded(tmp_path):
    """The whole reason this loader exists rather than trimesh's: normalize_glb
    puts the grounding transform on a node, and trimesh drops it."""
    path = tmp_path / "box.glb"
    trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(path)
    from realmspinner.pipelines.postprocess import normalize_glb

    normalize_glb(path, 4.0)
    lo, hi = gltf.load(path).bounds()
    assert (hi - lo).max() == pytest.approx(4.0, rel=1e-4)
    # Grounded: min-Y sits at zero on every job, size or no size.
    assert lo[1] == pytest.approx(0.0, abs=1e-6)


def test_a_matrix_node_decomposes_to_the_same_transform(tmp_path):
    """A node gives either TRS or a matrix; both have to land in the same place."""
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "n", "matrix": list(m3.translation(m3.vec3(1, 2, 3)).T.flatten())}],
    }
    path = tmp_path / "m.glb"
    path.write_bytes(_glb(doc, b""))
    node = gltf.load(path).nodes[0]
    assert node.translation == pytest.approx([1, 2, 3])
    assert node.world[:3, 3] == pytest.approx([1, 2, 3])


# --- materials --------------------------------------------------------------


def test_material_factors_survive_the_load(skinned_glb):
    prim = gltf.load(skinned_glb).meshes[0][0]
    assert prim.material.base_color_factor == pytest.approx((0.8, 0.2, 0.1, 1.0))
    assert prim.material.metallic_factor == 0.25
    assert prim.material.roughness_factor == 0.75
    assert prim.material.double_sided is True


def test_a_malformed_base_color_factor_is_defaulted_rather_than_crashing_the_render_loop():
    """create-01, the 2026-09-14 audit: ``material()`` read baseColorFactor
    and emissiveFactor with a bare ``tuple(pbr.get(...))`` and no shape or
    numeric check, unlike every other numeric field this loader validates
    (see ``_trs`` for a node's TRS). A GLB whose factor has the wrong element
    count, or a non-numeric entry, used to load cleanly and then crash on the
    very next render when ``GpuMaterial.bind`` wrote it into a vec4/vec3 GL
    uniform. It must now fall back to the glTF default instead, the same
    trade a malformed camera/light field already gets.
    """
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="<f4")
    binary = positions.tobytes()
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "material": 0}]}],
        "materials": [
            {
                "pbrMetallicRoughness": {
                    # two elements, not four -- the wrong shape a hostile or
                    # buggy exporter can still produce.
                    "baseColorFactor": [1.0, 2.0],
                },
                # non-numeric, the other shape this field can go wrong.
                "emissiveFactor": "red",
            }
        ],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(binary)}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}],
    }
    material = gltf.load(_glb(doc, binary)).meshes[0][0].material
    assert material.base_color_factor == pytest.approx((1.0, 1.0, 1.0, 1.0))
    assert material.emissive_factor == pytest.approx((0.0, 0.0, 0.0))


# --- skins ------------------------------------------------------------------


def test_joint_indices_are_promoted_to_int32(skinned_glb):
    """JOINTS_0 arrives as uint8; GLSL 3.30 indexes a palette with an ivec4."""
    prim = gltf.load(skinned_glb).meshes[0][0]
    assert prim.joints.dtype == np.int32
    assert prim.weights.dtype == np.float32
    assert prim.weights.sum(axis=1) == pytest.approx(np.ones(4))


def test_the_rest_palette_is_the_identity(skinned_glb):
    """world(joint) @ inverseBind(joint) cancels at rest, by construction --
    which is what makes a rest-pose render look like the unskinned mesh."""
    model = gltf.load(skinned_glb)
    node = model.mesh_instances()[0][0]
    for mat in model.joint_palette(node):
        assert mat == pytest.approx(np.eye(4), abs=1e-6)


def test_rotating_a_joint_moves_only_its_own_vertices(skinned_glb):
    model = gltf.load(skinned_glb)
    node = model.mesh_instances()[0][0]
    assert model.set_rotation("tip", m3.quat_from_axis_angle(m3.vec3(0, 0, 1), np.pi / 2))
    model.update_world()
    root_mat, tip_mat = model.joint_palette(node)

    assert root_mat == pytest.approx(np.eye(4), abs=1e-6)
    # The tip's own origin is its pivot, so a vertex there does not move...
    assert (tip_mat @ np.array([0.0, 1.0, 0.0, 1.0]))[:3] == pytest.approx([0, 1, 0], abs=1e-6)
    # ...while one beside it swings a quarter turn about z.
    assert (tip_mat @ np.array([1.0, 1.0, 0.0, 1.0]))[:3] == pytest.approx([0, 2, 0], abs=1e-6)


def test_an_unknown_bone_is_reported_not_ignored(skinned_glb):
    model = gltf.load(skinned_glb)
    assert model.set_rotation("no_such_bone", [0, 0, 0, 1]) is False


def test_a_pose_reports_only_what_moved(skinned_glb):
    model = gltf.load(skinned_glb)
    assert model.pose() == {}
    rest = model.get_rotation("tip")
    q = m3.quat_from_axis_angle(m3.vec3(1, 0, 0), 0.3)
    model.set_rotation("tip", q)
    assert list(model.pose()) == ["tip"]
    assert model.pose()["tip"] == pytest.approx(list(q))
    # Back to rest through the same setter -- resetting wholesale is the
    # PoseEditor's job, and the Model deliberately carries no method for it.
    model.set_rotation("tip", rest)
    assert model.pose() == {}


def test_a_pose_round_trips_as_xyzw(skinned_glb):
    """The wire order is three.js's, which is the order the Blender worker
    converts from -- not the order it converts to."""
    model = gltf.load(skinned_glb)
    q = m3.quat_from_axis_angle(m3.vec3(0, 1, 0), 1.1)
    model.set_rotation("tip", q)
    assert model.get_rotation("tip")[3] == pytest.approx(np.cos(0.55))


# --- refusals ---------------------------------------------------------------


def test_a_primitive_with_no_positions_is_refused(tmp_path):
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {}}]}],
    }
    path = tmp_path / "empty.glb"
    path.write_bytes(_glb(doc, b""))
    with pytest.raises(ValueError, match="POSITION"):
        gltf.load(path)


def test_a_line_primitive_is_refused_rather_than_drawn_as_triangles(tmp_path):
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "mode": 1}]}],
        "buffers": [{"byteLength": 12}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 12}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 1, "type": "VEC3"}],
    }
    path = tmp_path / "lines.glb"
    path.write_bytes(_glb(doc, b"\x00" * 12))
    with pytest.raises(ValueError, match="mode"):
        gltf.load(path)


def test_not_a_glb_is_refused_by_the_container_reader(tmp_path):
    path = tmp_path / "bad.glb"
    path.write_bytes(b"this is not a glb at all, really")
    with pytest.raises(ValueError, match="not a GLB"):
        gltf.load(path)


def test_a_file_too_short_for_a_header_is_a_refusal_not_a_struct_error(tmp_path):
    """struct.error is not a ValueError, so it went straight past every caller
    keying on the module's own refusal channel."""
    path = tmp_path / "stub.glb"
    path.write_bytes(b"glTF")
    with pytest.raises(ValueError, match="12-byte header"):
        gltf.load(path)


def test_a_file_that_stops_before_its_first_chunk_is_a_refusal(tmp_path):
    path = tmp_path / "headeronly.glb"
    path.write_bytes(struct.pack("<III", 0x46546C67, 2, 0) + b"\x00\x00\x00")
    with pytest.raises(ValueError, match="chunk header"):
        gltf.load(path)


# --- the real thing ---------------------------------------------------------


@needs_real
def test_the_real_rigs_palette_holds_every_joint_the_skin_declares():
    """Every figure here is read off the asset, never written down.

    This asserted the literal 20 and the presence of ``neutral_bone``, both of
    which were facts about one particular pruned job rather than about the
    loader: the replacement asset has 19 joints and no synthetic one, because
    Blender only invents ``neutral_bone`` when some vertex has no group. So the
    number and the name both failed while every property actually under test
    still held.

    What a loader can get wrong is *dropping* a joint -- which silently shifts
    every index above it -- so that is what is asserted: the palette, the
    inverse-bind array and the skin's own joint list are the same length, and
    each of those nodes is one the model resolved. The synthetic joint is the
    case that motivated it and is still checked, when the asset has one.
    """
    model = gltf.load(REAL_RIG)
    node = model.mesh_instances()[0][0]
    skin = model.skins[0]
    joints = model.joint_palette(node)

    assert len(joints) > 1
    assert len(skin.joints) == len(joints)
    assert skin.inverse_bind.shape == (len(joints), 4, 4)
    # Nothing was filtered out on the way in: every declared joint is a node
    # the model kept, including any the template skeleton does not name.
    named = set(model.by_name.values())
    assert all(index in named for index in skin.joints)
    if "neutral_bone" in model.by_name:
        assert model.by_name["neutral_bone"] in skin.joints


@needs_real
def test_the_real_rig_is_skinned_with_normalized_weights():
    model = gltf.load(REAL_RIG)
    prim = model.meshes[0][0]
    node = model.mesh_instances()[0][0]
    assert prim.joints.dtype == np.int32
    # Every index addresses a joint this skin actually has -- the failure a
    # dropped synthetic joint produces, and the reason the bound is the
    # palette's own length rather than a number typed in beside it.
    assert prim.joints.max() < len(model.joint_palette(node))
    assert prim.weights.sum(axis=1) == pytest.approx(np.ones(len(prim.weights)), abs=1e-5)


@needs_real
def test_the_real_mesh_is_grounded_and_carries_both_pbr_maps():
    model = gltf.load(REAL_MESH)
    lo, _hi = model.bounds()
    # normalize_glb puts min-Y at zero on every job.
    assert lo[1] == pytest.approx(0.0, abs=1e-4)
    material = model.meshes[0][0].material
    assert material.base_color is not None
    assert material.metallic_roughness is not None


@needs_real
def test_the_real_meshs_transform_is_on_a_child_not_the_root():
    """postprocess inserts the scale/translation below the root precisely so a
    loader that drops root transforms still gets it."""
    glbio = __import__(
        "realmspinner.kernels.geom3d.glbio", fromlist=["read_glb"]
    )
    doc, _ = glbio.read_glb(REAL_MESH)
    root = doc["nodes"][doc["scenes"][doc.get("scene", 0)]["nodes"][0]]
    assert "scale" not in root and "translation" not in root
    child = doc["nodes"][root["children"][0]]
    assert "scale" in child and "translation" in child


@needs_real
@pytest.mark.perf
def test_loading_the_real_mesh_is_fast_enough_to_do_on_a_click():
    import time

    start = time.perf_counter()
    gltf.load(REAL_MESH)
    assert time.perf_counter() - start < 3.0


def test_json_only_glbs_are_still_valid(tmp_path):
    """A GLB with no BIN chunk is legal; the loader must not assume one."""
    doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": []}]}
    path = tmp_path / "empty.glb"
    path.write_bytes(rebuild_glb(struct.pack("<III", 0x46546C67, 2, 0), doc, b""))
    # The fixture really is JSON-only: no BIN chunk came back out of it.
    from realmspinner.kernels.geom3d.glbio import read_glb

    assert read_glb(path)[1] == b""

    model = gltf.load(path)
    assert model.nodes == []
    assert model.meshes == []


# --- hand-supplied GLBs: the cases an exporter never writes -------------------
#
# Everything below is about assets this pipeline does not produce. They matter
# because the loader is also how a user's own GLB gets opened, and each of
# these failed in a way that produced a *plausible* result rather than an
# error -- geometry read from the wrong place, a hierarchy flattened, a texture
# silently dropped with a message that named the wrong reason.


def _minimal(accessors, buffer_views, binary, **extra):
    doc = {
        "asset": {"version": "2.0"},
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(binary)}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        **extra,
    }
    return _glb(doc, binary)


def test_an_interleaved_accessor_at_the_tail_of_the_buffer_loads():
    """glTF requires only the elements to be present: the padding after the
    last one need not exist. Demanding stride*count bytes made this raise
    "buffer is smaller than requested size" and the model simply failed."""
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="<f4")
    stride = 16  # 12 bytes of position plus 4 of something else
    binary = b"".join(p.tobytes() + b"\x00\x00\x00\x00" for p in positions)
    # Drop the final element's padding, which is exactly what an exporter
    # packing tightly to the end of the chunk produces.
    binary = binary[: stride * (len(positions) - 1) + 12]

    data = _minimal(
        [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}],
        [{"buffer": 0, "byteOffset": 0, "byteLength": len(binary), "byteStride": stride}],
        binary,
    )
    model = gltf.load(data)
    assert np.allclose(model.meshes[0][0].positions, positions)


def test_an_interleaved_accessor_reads_the_right_rows():
    # Three rows, not two: the 2026-09-08 audit, finding create-01, refuses an
    # unindexed TRIANGLES primitive whose vertex count is not a multiple of
    # 3, and this fixture has no NORMAL/indices to exempt it from that check.
    positions = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype="<f4")
    binary = b"".join(p.tobytes() + b"\xff\xff\xff\xff" for p in positions)
    data = _minimal(
        [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}],
        [{"buffer": 0, "byteOffset": 0, "byteLength": len(binary), "byteStride": 16}],
        binary,
    )
    assert np.allclose(gltf.load(data).meshes[0][0].positions, positions)


def test_a_view_naming_a_second_buffer_is_refused_rather_than_misread():
    """read_glb returns the BIN chunk, which is buffer 0. A view naming buffer
    1 was read at the same offsets into buffer 0 -- silently wrong geometry."""
    binary = np.zeros((3, 3), dtype="<f4").tobytes()
    data = _minimal(
        [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}],
        [{"buffer": 1, "byteOffset": 0, "byteLength": len(binary)}],
        binary,
        buffers=[{"byteLength": len(binary)}, {"byteLength": 999, "uri": "extra.bin"}],
    )
    with pytest.raises(ValueError, match="binary chunk"):
        gltf.load(data)


def test_an_accessor_reading_past_the_chunk_is_refused():
    binary = np.zeros((1, 3), dtype="<f4").tobytes()
    data = _minimal(
        [{"bufferView": 0, "componentType": 5126, "count": 99, "type": "VEC3"}],
        [{"buffer": 0, "byteOffset": 0, "byteLength": len(binary)}],
        binary,
    )
    with pytest.raises(ValueError, match="past the end"):
        gltf.load(data)


def test_an_accessor_with_an_unrecognised_component_type_is_refused_with_a_value_error_not_a_key_error():  # noqa: E501
    """The 2026-09-07 audit, finding create-09: an accessor whose
    ``componentType`` is not one of the six the glTF spec defines used to
    raise a bare ``KeyError`` from the ``_COMPONENT`` dict lookup -- an
    internal detail leaking out of a boundary every sibling refusal in this
    loader states as a named ``ValueError`` (the sparse-accessor refusal, the
    byte-budget refusal, both right beside this one). A hand-supplied GLB can
    declare anything in its JSON chunk; ``check_glb`` at the import door is
    structural-only.
    """
    binary = np.zeros((3, 3), dtype="<f4").tobytes()
    data = _minimal(
        [{"bufferView": 0, "componentType": 9999, "count": 3, "type": "VEC3"}],
        [{"buffer": 0, "byteOffset": 0, "byteLength": len(binary)}],
        binary,
    )
    with pytest.raises(ValueError, match="componentType"):
        gltf.load(data)


def test_an_accessor_with_an_unrecognised_type_is_refused_with_a_value_error_not_a_key_error():
    """Same finding, the other lookup: ``type`` (SCALAR/VEC3/...) feeding
    ``_NCOMP`` raised a bare ``KeyError`` too."""
    binary = np.zeros((3, 3), dtype="<f4").tobytes()
    data = _minimal(
        [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC5"}],
        [{"buffer": 0, "byteOffset": 0, "byteLength": len(binary)}],
        binary,
    )
    with pytest.raises(ValueError, match="accessor type"):
        gltf.load(data)


def test_an_accessor_missing_count_or_componenttype_raises_the_named_valueerror():
    """The 2026-09-18 audit, finding clay-03: ``_decode_accessor`` read
    ``componentType``/``type``/``count`` by raw dict indexing, so an accessor
    with an *unrecognised* value in one of those fields raised the named
    ``ValueError`` above (create-09), but one *missing the key entirely* fell
    straight through to a bare ``KeyError`` -- the same class of leak,
    unfixed for the field-absent case. A hand-supplied GLB can omit any
    field; ``check_glb`` at the import door is structural-only."""
    binary = np.zeros((3, 3), dtype="<f4").tobytes()
    buffer_views = [{"buffer": 0, "byteOffset": 0, "byteLength": len(binary)}]

    missing_count = _minimal(
        [{"bufferView": 0, "componentType": 5126, "type": "VEC3"}],
        buffer_views,
        binary,
    )
    with pytest.raises(ValueError, match="count"):
        gltf.load(missing_count)

    missing_component_type = _minimal(
        [{"bufferView": 0, "count": 3, "type": "VEC3"}],
        buffer_views,
        binary,
    )
    with pytest.raises(ValueError, match="componentType"):
        gltf.load(missing_component_type)

    missing_type = _minimal(
        [{"bufferView": 0, "componentType": 5126, "count": 3}],
        buffer_views,
        binary,
    )
    with pytest.raises(ValueError, match="type"):
        gltf.load(missing_type)


def test_a_required_extension_this_loader_does_not_implement_is_refused():
    """KHR_mesh_quantization is the one that arrives first -- gltfpack -c
    writes it, and gltfpack is the binary this project is about to vendor. A
    quantized position stream decoded as though it were plain floats is
    geometry that looks like nothing, with nothing in the data to say why."""
    binary = np.zeros((3, 3), dtype="<f4").tobytes()
    data = _minimal(
        [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}],
        [{"buffer": 0, "byteOffset": 0, "byteLength": len(binary)}],
        binary,
        extensionsRequired=["KHR_mesh_quantization"],
    )
    with pytest.raises(ValueError, match="KHR_mesh_quantization"):
        gltf.load(data)


def test_an_extension_that_is_merely_used_is_not_a_refusal():
    """``extensionsUsed`` is advisory -- the file still reads correctly with
    the extension ignored, which is exactly what a viewer should do."""
    binary = np.zeros((3, 3), dtype="<f4").tobytes()
    data = _minimal(
        [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}],
        [{"buffer": 0, "byteOffset": 0, "byteLength": len(binary)}],
        binary,
        extensionsUsed=["KHR_materials_emissive_strength"],
    )
    assert gltf.load(data).meshes


def test_a_normalized_uv_accessor_is_decoded_rather_than_read_raw():
    """``normalized`` says the integers encode a float in 0..1. Read raw, a UV
    comes back as 65535 -- a texture tiled sixty-five thousand times, which is
    wrong in a way that looks like a broken material rather than a bad read."""
    positions = np.zeros((3, 3), dtype="<f4")
    uvs = np.array([[0, 0], [65535, 0], [0, 32768]], dtype="<u2")
    binary = positions.tobytes() + uvs.tobytes()
    doc = {
        "asset": {"version": "2.0"},
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {
                "bufferView": 1,
                "componentType": 5123,
                "count": 3,
                "type": "VEC2",
                "normalized": True,
            },
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": positions.nbytes},
            {"buffer": 0, "byteOffset": positions.nbytes, "byteLength": uvs.nbytes},
        ],
        "buffers": [{"byteLength": len(binary)}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}}]}],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
    }

    [[prim]] = gltf.load(_glb(doc, binary)).meshes

    assert np.allclose(prim.uvs, [[0.0, 0.0], [1.0, 0.0], [0.0, 0.5]], atol=1e-4)


def test_a_corrupt_texture_costs_the_texture_and_not_the_model(caplog):
    """The stated policy for images, applied to the decode as well as to the
    lookup: a texture that cannot be read is a cosmetic loss. Raising turned
    one bad map into "this file will not open" for an intact mesh."""
    import base64

    binary = np.zeros((3, 3), dtype="<f4").tobytes()
    junk = base64.b64encode(b"not a png at all").decode()
    data = _minimal(
        [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}],
        [{"buffer": 0, "byteOffset": 0, "byteLength": len(binary)}],
        binary,
        images=[{"uri": f"data:image/png;base64,{junk}"}],
        textures=[{"source": 0}],
        materials=[
            {"name": "shell", "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}
        ],
        meshes=[
            {"primitives": [{"attributes": {"POSITION": 0}, "material": 0}]}
        ],
    )

    model = gltf.load(data)

    [[prim]] = model.meshes
    assert len(prim.positions) == 3, "the mesh still loads"
    assert prim.material is not None and prim.material.name == "shell"
    assert prim.material.base_color is None, "only the unreadable map is lost"


def test_a_glb_with_no_scenes_keeps_its_hierarchy():
    """Taking every node as a root made update_world visit each child twice --
    once under its parent, then again as a root with an identity parent, which
    overwrote the matrix it had just computed. Everything ended up at its local
    transform, so anything parented rendered in the wrong place."""
    binary = np.zeros((3, 3), dtype="<f4").tobytes()
    doc = {
        "asset": {"version": "2.0"},
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(binary)}],
        "buffers": [{"byteLength": len(binary)}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        "nodes": [
            {"translation": [10.0, 0.0, 0.0], "children": [1]},
            {"translation": [0.0, 5.0, 0.0], "mesh": 0},
        ],
    }
    model = gltf.load(_glb(doc, binary))

    assert model.roots == [0]
    child = model.nodes[1]
    assert np.allclose(child.world[:3, 3], [10.0, 5.0, 0.0])


def test_a_scene_without_a_node_list_falls_back_to_the_unparented_nodes():
    binary = np.zeros((3, 3), dtype="<f4").tobytes()
    doc = {
        "asset": {"version": "2.0"},
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(binary)}],
        "buffers": [{"byteLength": len(binary)}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        "nodes": [{"children": [1]}, {"mesh": 0}],
        "scenes": [{}],
    }
    assert gltf.load(_glb(doc, binary)).roots == [0]


def test_a_texture_in_a_data_uri_is_read_rather_than_dropped():
    """A data URI is inside the file -- it just is not in the binary chunk.
    Dropping it lost a texture the asset carried, and said it was "stored
    outside the GLB", which was false."""
    import base64
    import io as _io

    from PIL import Image

    buf = _io.BytesIO()
    Image.new("RGBA", (2, 2), (10, 20, 30, 255)).save(buf, "PNG")
    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    binary = np.zeros((3, 3), dtype="<f4").tobytes()
    data = _minimal(
        [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}],
        [{"buffer": 0, "byteOffset": 0, "byteLength": len(binary)}],
        binary,
        images=[{"uri": uri}],
        textures=[{"source": 0}],
        materials=[{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
        meshes=[
            {"primitives": [{"attributes": {"POSITION": 0}, "material": 0}]}
        ],
    )
    model = gltf.load(data)
    base_color = model.meshes[0][0].material.base_color
    assert base_color is not None
    assert base_color[:2] == (2, 2)


def test_a_texture_in_a_separate_file_is_skipped_with_an_honest_message(caplog):
    binary = np.zeros((3, 3), dtype="<f4").tobytes()
    data = _minimal(
        [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}],
        [{"buffer": 0, "byteOffset": 0, "byteLength": len(binary)}],
        binary,
        images=[{"uri": "colour.png"}],
        textures=[{"source": 0}],
        materials=[{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
        meshes=[
            {"primitives": [{"attributes": {"POSITION": 0}, "material": 0}]}
        ],
    )
    with caplog.at_level("WARNING"):
        model = gltf.load(data)
    assert model.meshes[0][0].material.base_color is None
    assert "separate file" in caplog.text


# --- hostile input: ceilings and malformed graphs -----------------------------
#
# check_glb at the import door is deliberately structural-only, so a file under
# the size limit reaches this loader with whatever its JSON chunk claims -- and
# this loader runs on the frame thread.


def _graph(nodes: list[dict], roots: list[int]) -> bytes:
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": roots}],
        "nodes": nodes,
    }
    return _glb(doc, b"")


def test_a_cycle_in_the_node_graph_does_not_spin_the_frame_thread():
    model = gltf.load(_graph([{"children": [1]}, {"children": [0]}], [0]))
    model.update_world()  # the loop terminates, which is the whole assertion
    assert len(model.nodes) == 2


def test_a_child_index_that_names_no_node_is_skipped_not_raised():
    model = gltf.load(_graph([{"translation": [1.0, 0.0, 0.0], "children": [7]}], [0]))
    model.update_world()
    assert np.allclose(model.nodes[0].world[:3, 3], [1.0, 0.0, 0.0])


def test_a_glb_with_an_out_of_range_mesh_index_is_refused_with_a_value_error_not_an_index_error():
    """The 2026-09-05 audit, finding create-01. This node graph used to load
    clean -- ``load()`` never looked at ``node["mesh"]`` against how many
    meshes the file actually declares -- and only blew up later, as a bare
    ``IndexError``, the first time something read ``model.meshes[node.mesh]``
    (``Model.mesh_instances``, or ``GpuModel.__init__`` in scene.py)."""
    with pytest.raises(ValueError, match="mesh 5"):
        gltf.load(_graph([{"name": "bad", "mesh": 5}], [0]))


def test_a_skin_joint_index_past_the_node_count_is_refused_at_load():
    """The 2026-09-06 audit, finding create2-01. ``node()`` bounds-checks
    ``node.mesh``/``node.skin`` against the file's declared counts (create-01,
    2026-09-05), but nothing checked the *contents* of ``skins[i].joints``
    against ``len(nodes)``. A skin whose own ``node.skin`` index is in range
    but which names a joint past the end of ``nodes`` used to load clean and
    only raise a bare ``IndexError`` later, out of ``Model.joint_palette`` --
    called from ``GpuModel.__init__`` (scene.py) after GPU buffers/textures
    for every earlier node already exist.
    """
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="<f4")
    joints_attr = np.array([[0, 0, 0, 0]] * 3, dtype="<u1")
    weights = np.tile(np.array([1, 0, 0, 0], dtype="<f4"), (3, 1))
    indices = np.array([0, 1, 2], dtype="<u4")
    ibm = np.array([np.eye(4, dtype="<f4").T])

    blobs = [positions.tobytes(), joints_attr.tobytes(), weights.tobytes(),
             indices.tobytes(), ibm.tobytes()]
    views, binary = [], b""
    for blob in blobs:
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(blob)})
        binary += blob + b"\x00" * (-len(blob) % 4)

    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        # Only ONE node declared. node.skin (0) is in range, but the skin
        # below names joint index 5, which does not exist -- create-01's
        # check on node.skin never fires for this file.
        "nodes": [{"name": "mesh_node", "mesh": 0, "skin": 0}],
        "meshes": [{
            "primitives": [{
                "attributes": {"POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2},
                "indices": 3,
            }]
        }],
        "skins": [{"joints": [5], "inverseBindMatrices": 4}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": views,
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5121, "count": 3, "type": "VEC4"},
            {"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC4"},
            {"bufferView": 3, "componentType": 5125, "count": 3, "type": "SCALAR"},
            {"bufferView": 4, "componentType": 5126, "count": 1, "type": "MAT4"},
        ],
    }
    with pytest.raises(ValueError, match="joint .*5"):
        gltf.load(_glb(doc, binary))


def test_a_glb_with_an_out_of_range_skin_index_does_not_leak_gpu_buffers_from_earlier_valid_nodes(
    gl, tmp_path
):
    """The 2026-09-05 audit, finding create-01. Before the fix, ``load()``
    happily returned a ``Model`` for this file, so ``GpuModel.__init__``
    (scene.py) would run: it allocates real ``ctx.buffer``/``ctx.texture``
    objects for the good node first, *then* raises a bare ``IndexError`` off
    ``model.skins[node.skin]`` for the bad one. Because the constructor never
    returns, nothing is ever left holding those objects to release them --
    and ``glctx.py`` documents that this app sets no moderngl ``gc_mode``, so
    a dropped reference frees nothing either. Refusing inside ``load()``
    means ``GpuModel.__init__`` is never entered for this file at all: there
    is nothing left to leak.
    """
    from realmspinner.studio.viewer import scene as scenelib

    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="<f4")
    indices = np.array([0, 1, 2], dtype="<u4")
    binary = positions.tobytes() + indices.tobytes()
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        # Node 0 is a perfectly good, unskinned mesh node -- its GpuPrimitive
        # would be built (and a real GL buffer allocated) before node 1 is
        # even looked at. Node 1 claims skin index 7, but no skins exist.
        "scenes": [{"nodes": [0, 1]}],
        "nodes": [
            {"name": "good", "mesh": 0},
            {"name": "bad", "mesh": 0, "skin": 7},
        ],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": positions.nbytes},
            {"buffer": 0, "byteOffset": positions.nbytes, "byteLength": indices.nbytes},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5125, "count": 3, "type": "SCALAR"},
        ],
    }
    path = tmp_path / "bad_skin.glb"
    path.write_bytes(_glb(doc, binary))

    created: list[object] = []
    real_buffer = gl.buffer

    def tracking_buffer(*a, **kw):
        b = real_buffer(*a, **kw)
        created.append(b)
        return b

    gl.buffer = tracking_buffer
    try:
        with pytest.raises(ValueError, match="skin 7"):
            scenelib.GpuModel(gl, gltf.load(path))
        assert created == [], (
            "a GL buffer was allocated for the good node before the bad "
            "node's skin index was ever checked -- the refusal belongs in "
            "gltf.load(), before scene.py sees a Model at all"
        )
    finally:
        gl.buffer = real_buffer


def test_a_triangle_primitive_with_no_normals_and_a_non_multiple_of_three_index_count_is_refused_at_load():  # noqa: E501
    """The 2026-09-08 audit, finding create-01. A TRIANGLES primitive whose
    index count is not a multiple of 3 used to load clean: nothing checked
    it, and the primitive carries no NORMAL attribute here, so the first
    thing to notice was ``scene._face_normals``'s
    ``primitive.indices.reshape(-1, 3)`` -- a bare ``ValueError`` raised only
    after ``GpuModel.__init__`` had already allocated real GL buffers for any
    earlier primitives, which then leak. The refusal belongs in
    ``_Reader.primitive()``, before a ``Model`` is ever returned.
    """
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="<f4")
    # 4 indices: not a multiple of 3, and no NORMAL attribute is declared.
    indices = np.array([0, 1, 2, 0], dtype="<u4")
    binary = positions.tobytes() + indices.tobytes()
    data = _minimal(
        [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5125, "count": 4, "type": "SCALAR"},
        ],
        [
            {"buffer": 0, "byteOffset": 0, "byteLength": positions.nbytes},
            {"buffer": 0, "byteOffset": positions.nbytes, "byteLength": indices.nbytes},
        ],
        binary,
        meshes=[{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
    )
    with pytest.raises(ValueError, match="multiple of 3"):
        gltf.load(data)


def test_a_node_count_over_the_ceiling_is_refused_at_load(monkeypatch):
    monkeypatch.setattr(gltf, "MAX_NODES", 2)
    with pytest.raises(ValueError, match="more than this viewer will load"):
        gltf.load(_graph([{}, {}, {}], [0]))


def test_a_material_count_over_the_ceiling_is_refused_at_load(monkeypatch):
    """The 2026-09-07 audit, finding clay-05: ``load`` bounded ``nodes`` but
    iterated ``materials`` with no ceiling of its own, and an *unreferenced*
    material entry never reaches ``_charge`` -- reproduced with 500,000
    material entries loading in 3.7s while ``_spent`` (the byte budget) never
    moved. A material array this large is a hang before it is a scene.
    """
    monkeypatch.setattr(gltf, "MAX_MATERIALS", 2)
    doc = {"asset": {"version": "2.0"}, "materials": [{}, {}, {}]}
    with pytest.raises(ValueError, match="more than this viewer will load"):
        gltf.load(_glb(doc, b""))


def test_a_mesh_count_over_the_ceiling_is_refused_at_load(monkeypatch):
    """Same finding, the other unbounded array: a mesh with no primitives (or
    primitives that read no accessor) also never reaches ``_charge``."""
    monkeypatch.setattr(gltf, "MAX_MESHES", 2)
    doc = {
        "asset": {"version": "2.0"},
        "meshes": [{"primitives": []}, {"primitives": []}, {"primitives": []}],
    }
    with pytest.raises(ValueError, match="more than this viewer will load"):
        gltf.load(_glb(doc, b""))


def test_load_refuses_a_mesh_declaring_more_primitives_than_this_viewer_will_build(monkeypatch):
    """The 2026-09-16 audit: MAX_MESHES bounds how many *mesh* entries a file
    may declare, but nothing bounded how many *primitives* one mesh entry may
    declare. A single mesh referencing the same accessor from many primitive
    entries stays tiny in bytes (MAX_TOTAL_BYTES never trips) while still
    costing one Primitive object -- and its own allocation -- per entry;
    reproduced at 2,000,000 primitives in a 66 MB GLB, ``load`` took 6.3 s
    with every other ceiling (MAX_NODES/MAX_MATERIALS/MAX_MESHES/MAX_CAMERAS/
    MAX_LIGHTS/MAX_TOTAL_BYTES) left untripped.
    """
    monkeypatch.setattr(gltf, "MAX_PRIMITIVES", 2)
    doc = {
        "asset": {"version": "2.0"},
        "meshes": [{"primitives": [{"attributes": {}}, {"attributes": {}}, {"attributes": {}}]}],
    }
    with pytest.raises(ValueError, match="more than this viewer will load"):
        gltf.load(_glb(doc, b""))


def test_an_oversized_texture_costs_the_texture_and_not_the_model(monkeypatch, caplog):
    """Same policy as a corrupt map. The ceiling is lowered rather than a
    16-megapixel image built, which would cost the suite more than the bug."""
    import base64
    import io as _io

    from PIL import Image

    monkeypatch.setattr(gltf, "MAX_TEXTURE_PIXELS", 3)
    buf = _io.BytesIO()
    Image.new("RGBA", (2, 2), (10, 20, 30, 255)).save(buf, "PNG")
    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    binary = np.zeros((3, 3), dtype="<f4").tobytes()
    data = _minimal(
        [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}],
        [{"buffer": 0, "byteOffset": 0, "byteLength": len(binary)}],
        binary,
        images=[{"uri": uri}],
        textures=[{"source": 0}],
        materials=[{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
        meshes=[{"primitives": [{"attributes": {"POSITION": 0}, "material": 0}]}],
    )
    with caplog.at_level("WARNING"):
        model = gltf.load(data)

    assert len(model.meshes[0][0].positions) == 3, "the mesh still loads"
    assert model.meshes[0][0].material.base_color is None
    assert model.skipped_textures == 1
    assert "ceiling" in caplog.text


# --- H01: a document-wide budget, on top of the per-accessor ceiling ---------
#
# MAX_ACCESSOR_BYTES bounds one array; nothing bounded the *sum* of them, so a
# small file with many primitives -- or one accessor replayed by many
# primitives -- allocated without limit as long as each individual array
# stayed under the per-accessor cap. Measured: a 4,036-byte bufferless GLB
# with 128 primitives produced 6,144,000 bytes of geometry across 128
# separate arrays, with nothing to stop scaling the primitive count further.
#
# Every test below lowers ``MAX_TOTAL_BYTES`` before building anything, so the
# suite itself never approaches the real 768 MiB default -- proving the
# refusal without paying for the memory it exists to refuse.


def _bufferless_positions_doc(n: int, count: int) -> dict:
    """``n`` primitives, each its own zero-filled POSITION accessor with no
    bufferView -- the branch H01 was written for: nothing here reads a
    buffer, so nothing about any one primitive's cost is bounded by how small
    the file declaring it is."""
    return {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "accessors": [
            {"componentType": 5126, "count": count, "type": "VEC3"} for _ in range(n)
        ],
        "meshes": [{"primitives": [{"attributes": {"POSITION": i}} for i in range(n)]}],
    }


def test_the_aggregate_budget_refuses_many_primitives_each_under_the_per_accessor_cap(
    monkeypatch,
):
    """Each primitive here is tiny -- MAX_ACCESSOR_BYTES never fires -- and
    that is the point: only the *document-wide* ceiling can catch a file that
    scales the primitive count instead of any one accessor's size."""
    monkeypatch.setattr(gltf, "MAX_TOTAL_BYTES", 20_000)
    # 99, not 100: a multiple of 3, so create-01's unindexed-vertex-count
    # refusal (2026-09-08 audit) does not fire on these bufferless fixtures.
    small = _bufferless_positions_doc(4, 99)
    model = gltf.load(_glb(small, b""))
    assert len(model.meshes[0]) == 4

    huge = _bufferless_positions_doc(200, 99)
    with pytest.raises(ValueError, match="byte budget"):
        gltf.load(_glb(huge, b""))


def test_a_repeated_accessor_reference_is_decoded_and_charged_once(monkeypatch):
    """An instanced mesh's shared POSITION stream: fifty primitives naming
    *one* accessor must cost what one primitive costs, not fifty -- which is
    exactly what a document-wide budget cannot tell apart from fifty distinct
    accessors unless repeated references are actually deduplicated rather
    than merely bounded."""
    monkeypatch.setattr(gltf, "MAX_TOTAL_BYTES", 5_000)
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        # 99, not 100: a multiple of 3, so create-01's unindexed-vertex-count
        # refusal (2026-09-08 audit) does not fire on this fixture.
        "accessors": [{"componentType": 5126, "count": 99, "type": "VEC3"}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}} for _ in range(50)]}],
    }
    model = gltf.load(_glb(doc, b""))
    assert len(model.meshes[0]) == 50
    # Every primitive's positions really are the *same* array, not fifty
    # equal-looking copies -- the property that makes the charge above land
    # once rather than fifty times.
    first = model.meshes[0][0].positions
    assert all(p.positions is first for p in model.meshes[0])


def test_repeated_textures_over_the_document_budget_are_refused(monkeypatch):
    """Five distinct textures, each comfortably under MAX_TEXTURE_PIXELS on
    its own -- the per-texture ceiling never fires -- but together over a
    lowered document-wide budget. Texture bytes were not charged into the same
    ledger geometry was, so nothing but MAX_TEXTURE_PIXELS ever stood between
    a document and however many such textures it declared."""
    import base64
    import io as _io

    from PIL import Image

    monkeypatch.setattr(gltf, "MAX_TOTAL_BYTES", 10_000)
    binary = np.zeros((3, 3), dtype="<f4").tobytes()

    def _uri(colour):
        buf = _io.BytesIO()
        Image.new("RGBA", (32, 32), colour).save(buf, "PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    n = 5
    images = [{"uri": _uri((i, i, i, 255))} for i in range(n)]
    textures = [{"source": i} for i in range(n)]
    materials = [
        {"pbrMetallicRoughness": {"baseColorTexture": {"index": i}}} for i in range(n)
    ]
    data = _minimal(
        [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}],
        [{"buffer": 0, "byteOffset": 0, "byteLength": len(binary)}],
        binary,
        images=images,
        textures=textures,
        materials=materials,
        meshes=[
            {
                "primitives": [
                    {"attributes": {"POSITION": 0}, "material": i} for i in range(n)
                ]
            }
        ],
    )
    # 32x32 RGBA is 4,096 bytes each -- comfortably under MAX_TEXTURE_PIXELS,
    # so that ceiling never fires -- and five of them is 20,480 bytes, well
    # past the 10,000-byte document budget this test lowers.
    with pytest.raises(ValueError, match="byte budget"):
        gltf.load(data)


def test_a_large_weights_0_accessor_s_normalisation_copies_are_charged_against_the_document_budget(
    monkeypatch,
):
    """The 2026-09-13 audit, finding create-05: the WEIGHTS_0 path builds two
    or three same-size float32 arrays -- the normalising division (for a
    ubyte/ushort source) and the renormalising division that always runs --
    without charging either, so a skinned mesh's peak allocation was
    undercounted against MAX_TOTAL_BYTES. 99 vertices (a multiple of 3, so
    create-01's unindexed-vertex-count refusal does not fire) of VEC4 float32
    weights: everything already charged elsewhere (POSITION decoded and
    retyped, WEIGHTS_0 decoded and retyped) comes to 5,940 bytes, comfortably
    under the 6,500-byte budget below; the always-run renormalising
    division's own 1,584 bytes pushes the total to 7,524, over budget, once
    charged.
    """
    monkeypatch.setattr(gltf, "MAX_TOTAL_BYTES", 6_500)
    n = 99
    positions = np.zeros((n, 3), dtype="<f4")
    weights = np.tile(np.array([1, 0, 0, 0], dtype="<f4"), (n, 1))
    binary = positions.tobytes() + weights.tobytes()
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {
                "primitives": [
                    {"attributes": {"POSITION": 0, "WEIGHTS_0": 1}}
                ]
            }
        ],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": positions.nbytes},
            {"buffer": 0, "byteOffset": positions.nbytes, "byteLength": weights.nbytes},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": n, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5126, "count": n, "type": "VEC4"},
        ],
    }
    with pytest.raises(ValueError, match="byte budget"):
        gltf.load(_glb(doc, binary))


def test_a_texture_bufferview_past_the_end_of_the_bin_chunk_is_skipped_not_truncated(
    caplog,
):
    """The 2026-09-13 audit, finding create-09: an image's bufferView was
    sliced straight out of the BIN chunk with no span check, so a byteLength
    that overran the buffer was silently truncated into a corrupt-looking
    image instead of refused -- and then failed to decode and got skipped
    for the wrong reason. Same skip contract as a genuinely unreadable
    texture (clay-09's ``_check_buffer`` branch just above it): the model
    still loads, the texture is dropped, and the warning names the real
    cause.
    """
    import io as _io

    from PIL import Image

    png_buf = _io.BytesIO()
    Image.new("RGBA", (2, 2), (10, 20, 30, 255)).save(png_buf, "PNG")
    png_bytes = png_buf.getvalue()

    positions = np.zeros((3, 3), dtype="<f4")
    binary = positions.tobytes() + png_bytes

    data = _minimal(
        [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}],
        [
            {"buffer": 0, "byteOffset": 0, "byteLength": positions.nbytes},
            # Declares far more bytes than actually remain in the buffer.
            {
                "buffer": 0,
                "byteOffset": positions.nbytes,
                "byteLength": len(png_bytes) * 1000,
            },
        ],
        binary,
        images=[{"bufferView": 1}],
        textures=[{"source": 0}],
        materials=[{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
        meshes=[{"primitives": [{"attributes": {"POSITION": 0}, "material": 0}]}],
    )

    with caplog.at_level("WARNING"):
        model = gltf.load(data)

    assert len(model.meshes[0][0].positions) == 3, "the mesh still loads"
    assert model.meshes[0][0].material.base_color is None
    assert model.skipped_textures == 1
    assert "past the end" in caplog.text


# --- material and texture indices: negative and out-of-range -----------------
#
# The 2026-09-06 audit, findings clay-06 and clay-09: node.mesh, node.skin and
# skin.joints were all hardened against a bad index (create-01, create2-01),
# but a primitive's own ``material`` and a material's own texture/image/
# bufferView references were not -- and Python's negative-index wraparound
# means "not bounds-checked" is not even the same failure as "raises".


def test_a_negative_material_index_falls_back_to_default_rather_than_wrapping_to_the_last_material():  # noqa: E501
    """``prim["material"]: -1`` used to satisfy ``-1 < len(materials)`` and
    resolve through Python's own negative-index wraparound to
    ``materials[-1]`` -- the *last* entry in the palette -- rather than to the
    default ``Material()`` an out-of-range positive index already falls back
    to. No error, no log line, no toast anywhere in the call chain: the
    primitive just came out painted with a plausible but wrong material."""
    binary = np.zeros((3, 3), dtype="<f4").tobytes()
    data = _minimal(
        [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}],
        [{"buffer": 0, "byteOffset": 0, "byteLength": len(binary)}],
        binary,
        materials=[{"name": "mat0"}, {"name": "mat1"}],
        meshes=[
            {"primitives": [{"attributes": {"POSITION": 0}, "material": -1}]}
        ],
    )
    model = gltf.load(data)
    prim = model.meshes[0][0]
    assert prim.material.name == "", (
        f"expected the default material, got {prim.material.name!r} -- a "
        "negative index wrapped onto the palette instead of falling back"
    )


def test_an_out_of_range_texture_index_is_refused_with_a_value_error_not_an_index_error():
    """``glbimport.py``'s own comments say every refusal in this module is a
    ``ValueError`` and callers key on it. A material naming a texture index
    past the end of (or with no) ``textures`` array used to index straight
    into the list and raise a bare ``IndexError`` instead -- the same gap
    ``node()``/``skin()`` were hardened against for mesh/skin/joint indices."""
    binary = np.zeros((3, 3), dtype="<f4").tobytes()
    data = _minimal(
        [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}],
        [{"buffer": 0, "byteOffset": 0, "byteLength": len(binary)}],
        binary,
        # No "textures" array at all -- the probe case from the finding.
        materials=[
            {"pbrMetallicRoughness": {"baseColorTexture": {"index": 5}}}
        ],
        meshes=[
            {"primitives": [{"attributes": {"POSITION": 0}, "material": 0}]}
        ],
    )
    with pytest.raises(ValueError, match="texture 5"):
        gltf.load(data)


def test_a_texture_with_no_source_is_refused_with_a_value_error_not_a_key_error():
    """The 2026-09-08 audit's clay-04: a ``textures[]`` entry with no
    "source" key is legal per the glTF 2.0 schema (a texture may carry only a
    sampler), and indexing straight into it used to raise a bare
    ``KeyError`` -- the one shape of malformed reference this file's sibling
    boundaries (node.mesh, node.skin, skin.joints, texture/image index
    bounds, prim["material"]) had already been hardened against by the
    2026-09-05/09-06/09-07 audits. ``glbimport.glb_to_claydoc`` catches
    ``Exception`` broadly, so the unhandled shape reached the user as
    "...'source'" instead of a sentence describing what is wrong."""
    binary = np.zeros((3, 3), dtype="<f4").tobytes()
    data = _minimal(
        [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}],
        [{"buffer": 0, "byteOffset": 0, "byteLength": len(binary)}],
        binary,
        # A texture that names only a sampler, no source image.
        textures=[{"sampler": 0}],
        materials=[
            {"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}
        ],
        meshes=[
            {"primitives": [{"attributes": {"POSITION": 0}, "material": 0}]}
        ],
    )
    with pytest.raises(ValueError, match="texture 0 has no source"):
        gltf.load(data)


def test_a_negative_texture_index_is_refused_rather_than_wrapping_to_the_last_texture():
    """Same wraparound risk as clay-06, one boundary over: ``ref["index"]: -1``
    would satisfy an upper-bound-only check and silently resolve to the last
    texture in the file rather than being refused."""
    import base64
    import io as _io

    from PIL import Image

    buf = _io.BytesIO()
    Image.new("RGBA", (2, 2), (1, 2, 3, 255)).save(buf, "PNG")
    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    binary = np.zeros((3, 3), dtype="<f4").tobytes()
    data = _minimal(
        [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}],
        [{"buffer": 0, "byteOffset": 0, "byteLength": len(binary)}],
        binary,
        images=[{"uri": uri}],
        textures=[{"source": 0}],
        materials=[
            {"pbrMetallicRoughness": {"baseColorTexture": {"index": -1}}}
        ],
        meshes=[
            {"primitives": [{"attributes": {"POSITION": 0}, "material": 0}]}
        ],
    )
    with pytest.raises(ValueError, match="texture -1"):
        gltf.load(data)


def test_an_out_of_range_image_index_is_refused_with_a_value_error_not_an_index_error():
    binary = np.zeros((3, 3), dtype="<f4").tobytes()
    data = _minimal(
        [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}],
        [{"buffer": 0, "byteOffset": 0, "byteLength": len(binary)}],
        binary,
        # A texture naming an image index past the end of an empty images array.
        textures=[{"source": 3}],
        materials=[
            {"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}
        ],
        meshes=[
            {"primitives": [{"attributes": {"POSITION": 0}, "material": 0}]}
        ],
    )
    with pytest.raises(ValueError, match="image 3"):
        gltf.load(data)


def test_an_out_of_range_buffer_view_on_an_image_is_refused_with_a_value_error_not_an_index_error():
    binary = np.zeros((3, 3), dtype="<f4").tobytes()
    data = _minimal(
        [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}],
        [{"buffer": 0, "byteOffset": 0, "byteLength": len(binary)}],
        binary,
        # Only bufferView 0 exists (the geometry's own), so an image naming
        # bufferView 9 is the out-of-range case.
        images=[{"bufferView": 9}],
        textures=[{"source": 0}],
        materials=[
            {"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}
        ],
        meshes=[
            {"primitives": [{"attributes": {"POSITION": 0}, "material": 0}]}
        ],
    )
    with pytest.raises(ValueError, match="bufferView 9"):
        gltf.load(data)


# --- index-shaped fields checked for type, not just range -------------------


def test_a_non_integer_mesh_index_is_refused_with_a_value_error_not_a_type_error():
    """The 2026-09-09 audit, finding clay-04. ``node()`` bounds-checks
    ``node.mesh`` with ``0 <= mesh < n_meshes`` (create-01, 2026-09-05), but
    never checked that ``mesh`` was actually an integer first, so a string
    index reached that comparison as a bare
    ``TypeError: '<=' not supported between instances of 'int' and 'str'``
    instead of the named refusal every sibling boundary in this file gives.
    """
    with pytest.raises(ValueError, match="mesh reference must be a whole number") as exc_info:
        gltf.load(_graph([{"name": "bad", "mesh": "0"}], [0]))
    assert "'bad'" in str(exc_info.value)


def test_a_non_integer_accessor_index_is_refused_with_a_value_error_not_a_type_error():
    """The 2026-09-09 audit, finding clay-04's other operator: an
    ``attributes.POSITION`` naming a string instead of an accessor index used
    to index straight into ``self.gltf["accessors"]`` and raise
    ``TypeError: list indices must be integers or slices, not str`` rather
    than the named refusal every sibling boundary in this file gives.
    """
    binary = np.zeros((3, 3), dtype="<f4").tobytes()
    doc = {
        "asset": {"version": "2.0"},
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(binary)}],
        "buffers": [{"byteLength": len(binary)}],
        # POSITION names a non-integer accessor index.
        "meshes": [{"primitives": [{"attributes": {"POSITION": "0"}}]}],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
    }
    with pytest.raises(ValueError, match="accessor reference must be a whole number"):
        gltf.load(_glb(doc, binary))


def test_a_non_integer_child_index_is_refused_with_a_value_error_not_a_type_error():
    """The 2026-09-11 audit, finding create-01 (merged). A node's own
    ``children`` array was stored straight off the JSON with no validation at
    all, so a non-integer entry reached ``Model.update_world``'s bare
    ``0 <= index < len(self.nodes)`` comparison as an un-messaged
    ``TypeError`` instead of the named ``ValueError`` every sibling
    index-shaped field in this loader (mesh, skin, joint, texture, image
    source) was hardened to give in create-01/clay-06/clay-09/clay-04.
    """
    with pytest.raises(ValueError, match="child reference must be a whole number"):
        gltf.load(_graph([{"children": ["not-an-index"]}], [0]))


def test_a_non_integer_scene_root_index_is_refused_with_a_value_error_not_a_type_error():
    """The 2026-09-11 audit, finding create-01 (merged): the other half of
    the same gap, one level up -- a scene's own "nodes" root list."""
    with pytest.raises(ValueError, match="root node index must be a whole number"):
        gltf.load(_graph([{}], ["not-an-index"]))


def test_a_non_integer_scene_index_is_refused_with_a_value_error_not_a_type_error():
    """The 2026-09-11 audit, finding create-01 (merged): the document's own
    "scene" key reaches ``_roots``'s ``0 <= index < len(scenes)`` comparison
    unchecked when it names a scene index rather than a root list."""
    doc = {
        "asset": {"version": "2.0"},
        "scene": "0",
        "scenes": [{"nodes": [0]}],
        "nodes": [{}],
    }
    with pytest.raises(ValueError, match="scene index must be a whole number"):
        gltf.load(_glb(doc, b""))


def test_a_non_integer_image_buffer_view_is_refused_with_a_value_error_not_a_type_error():
    """The 2026-09-11 audit, finding create-01 (merged): the last of the four
    missed sites -- an image's own ``bufferView`` reaches
    ``_Reader._image_bytes``'s bare ``0 <= bv < len(buffer_views)`` comparison
    unchecked, unlike the texture/image *index* checks a few lines above it."""
    binary = np.zeros((3, 3), dtype="<f4").tobytes()
    data = _minimal(
        [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}],
        [{"buffer": 0, "byteOffset": 0, "byteLength": len(binary)}],
        binary,
        images=[{"bufferView": "0"}],
        textures=[{"source": 0}],
        materials=[{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
        meshes=[{"primitives": [{"attributes": {"POSITION": 0}, "material": 0}]}],
    )
    with pytest.raises(ValueError, match="bufferView reference must be a whole number"):
        gltf.load(data)


def test_a_glb_whose_json_chunk_is_not_an_object_is_refused_by_name_not_a_bare_attributeerror():
    """The 2026-09-11 audit, finding clay-03. glbio.split_glb never checked
    that a GLB's JSON chunk parses to an object, so a hand-built GLB whose
    JSON chunk is valid JSON but not a dict (a bare array, here) reached
    every downstream ``.get(...)`` call -- gltf.load's own among them -- as a
    bare, un-messaged AttributeError instead of the named ValueError every
    other malformed-GLB boundary in this module raises.
    """
    from realmspinner.kernels.geom3d import glbio

    header = struct.pack("<III", 0x46546C67, 2, 0)
    # rebuild_glb only promises to json.dumps whatever it is given; a list
    # serializes just as cleanly as a dict, and is exactly the shape a
    # hand-edited or hostile GLB could legally carry.
    data = rebuild_glb(header, [1, 2, 3], b"")
    with pytest.raises(ValueError, match="not a GLB file"):
        glbio.split_glb(data)
    with pytest.raises(ValueError, match="not a GLB file"):
        gltf.load(data)


def test_a_node_with_a_three_element_rotation_is_refused_by_name_not_by_unpack_error():
    """The 2026-09-09 audit, finding clay-05. ``node()`` read ``translation``/
    ``rotation``/``scale``/``matrix`` straight off the JSON with no shape
    check at all, so a three-element ``rotation`` (a quaternion with its w
    dropped) reached ``Model.update_world()`` -- called from inside
    ``Model.__init__``, itself inside ``load()`` -- and failed deep in
    ``math3d.compose``/``quat_to_mat4`` with a bare
    ``ValueError: not enough values to unpack (expected 4, got 3)`` instead
    of a refusal naming the node and the field.
    """
    with pytest.raises(ValueError, match="rotation of 3 numbers, not 4") as exc_info:
        gltf.load(_graph([{"name": "bad", "rotation": [0.0, 0.0, 0.0]}], [0]))
    assert "'bad'" in str(exc_info.value)


# --- cameras and lights ------------------------------------------------------


def _marker_glb(**extra):
    """A one-node, geometry-free document, for the two marker arrays."""
    doc = {
        "asset": {"version": "2.0"},
        "nodes": [{"name": "marker"}],
        "scenes": [{"nodes": [0]}],
        **extra,
    }
    return _glb(doc, b"")


def test_the_light_extension_is_no_longer_refused_when_a_file_requires_it():
    """This loader now reads KHR_lights_punctual, so a file that *requires* it
    is a file whose lights this build can restore. It used to be refused
    outright by the same door that refuses KHR_mesh_quantization -- which was
    right while nothing here could read a light and is wrong now that Mason
    writes them.
    """
    data = _marker_glb(
        extensionsRequired=["KHR_lights_punctual"],
        extensions={"KHR_lights_punctual": {"lights": [{"type": "point"}]}},
    )
    assert len(gltf.load(data).lights) == 1


def test_a_malformed_extensions_block_costs_the_lights_and_not_the_file():
    """Every level is read defensively rather than indexed. A light is a
    marker: losing one is a differently-lit scene, and refusing the file over
    it would lose the geometry as well -- which is the trade
    ``Model.skipped_textures`` already names for an unreadable image.
    """
    for extensions in ({"KHR_lights_punctual": []}, {"KHR_lights_punctual": {"lights": 7}}, 5):
        model = gltf.load(_marker_glb(extensions=extensions))
        assert model.lights == []
        assert model.nodes[0].name == "marker"


def test_a_node_naming_a_light_the_file_does_not_declare_is_refused_by_name():
    """The same named refusal every other index-shaped field in this loader
    gives -- an out-of-range one used to be the only kind of index here that
    could not exist at all, and now there are two more."""
    doc = {
        "asset": {"version": "2.0"},
        "nodes": [{"extensions": {"KHR_lights_punctual": {"light": 4}}}],
        "scenes": [{"nodes": [0]}],
        "extensions": {"KHR_lights_punctual": {"lights": [{"type": "point"}]}},
    }
    with pytest.raises(ValueError, match="references light"):
        gltf.load(_glb(doc, b""))


def test_a_node_naming_a_camera_the_file_does_not_declare_is_refused_by_name():
    doc = {
        "asset": {"version": "2.0"},
        "nodes": [{"camera": 2}],
        "scenes": [{"nodes": [0]}],
        "cameras": [{"type": "perspective", "perspective": {"yfov": 0.8, "znear": 0.1}}],
    }
    with pytest.raises(ValueError, match="references camera"):
        gltf.load(_glb(doc, b""))


def test_a_camera_or_light_index_that_is_not_a_whole_number_is_refused():
    """clay-04's rule, at the two boundaries it did not exist for yet: a
    string index reaching Python's own ``<=`` is a bare TypeError where every
    sibling field here raises a named ValueError."""
    for node, message in (
        ({"camera": "1"}, "camera reference"),
        ({"extensions": {"KHR_lights_punctual": {"light": 1.0}}}, "light reference"),
    ):
        doc = {
            "asset": {"version": "2.0"},
            "nodes": [node],
            "scenes": [{"nodes": [0]}],
            "cameras": [{"type": "perspective", "perspective": {"yfov": 1.0, "znear": 0.1}}],
            "extensions": {"KHR_lights_punctual": {"lights": [{"type": "point"}]}},
        }
        with pytest.raises(ValueError, match=message):
            gltf.load(_glb(doc, b""))


def test_a_non_numeric_or_infinite_camera_field_falls_back_rather_than_refusing():
    """An infinite ``yfov`` reaches a projection matrix as a frame of NaNs --
    a viewport that draws nothing with no error anywhere to say why. Falling
    back rather than raising, because a camera decodes no bytes and sizes no
    allocation: it is a marker, and a malformed one costs a frustum."""
    data = _marker_glb(
        cameras=[
            {"type": "perspective", "perspective": {"yfov": "wide", "znear": 1e400, "zfar": None}}
        ]
    )
    camera = gltf.load(data).cameras[0]
    assert camera.yfov == pytest.approx(gltf.Camera.yfov)
    assert camera.znear == pytest.approx(gltf.Camera.znear)
    assert camera.zfar == 0.0


def test_a_light_of_a_kind_this_build_has_never_heard_of_reads_as_a_point():
    """KHR_lights_punctual is explicitly extensible, so a fourth kind added to
    it later must not make a file carrying one unopenable."""
    model = gltf.load(
        _marker_glb(extensions={"KHR_lights_punctual": {"lights": [{"type": "area"}]}})
    )
    assert model.lights[0].kind == "point"


# --- an accessor's own index and bufferView, range-checked ------------------
#
# The 2026-09-12 audit, finding clay-01: every sibling index-shaped field in
# this file (node.mesh/skin/child/camera/light, a skin's joints, a material's
# texture/image index, an image's own bufferView, the document's scene index)
# is both type- and range-checked, but a primitive attribute's own accessor
# index (POSITION/NORMAL/TEXCOORD_0/JOINTS_0/WEIGHTS_0/indices, and a skin's
# inverseBindMatrices) was only type-checked (clay-04) -- never range-checked
# -- and an accessor's own "bufferView" was neither, despite a comment
# claiming otherwise.


def test_a_negative_position_accessor_index_is_refused_rather_than_silently_wrapping_onto_another_accessor():  # noqa: E501
    """``attributes.POSITION: -1`` used to satisfy Python's own negative-index
    semantics and resolve to ``accessors[-1]`` -- a real but unrelated
    accessor -- rather than being refused. No error, no log, no toast: the
    primitive loaded with geometry read from the wrong stream."""
    zeros = np.zeros((3, 3), dtype="<f4").tobytes()
    ones = np.ones((3, 3), dtype="<f4").tobytes()
    binary = zeros + ones
    data = _minimal(
        [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3"},
        ],
        [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(zeros)},
            {"buffer": 0, "byteOffset": len(zeros), "byteLength": len(ones)},
        ],
        binary,
        # -1 wraps onto accessor 1 (the "ones" accessor) instead of being
        # refused as the invalid reference it is.
        meshes=[{"primitives": [{"attributes": {"POSITION": -1}}]}],
    )
    with pytest.raises(ValueError, match="accessor reference is -1"):
        gltf.load(data)


def test_an_out_of_range_positive_accessor_index_on_a_primitive_attribute_is_refused_with_a_value_error_not_an_index_error():  # noqa: E501
    """The positive-index twin of the wraparound above: ``POSITION: 5`` with
    only one accessor declared used to index straight into ``accessors`` and
    raise a bare ``IndexError`` instead of the named refusal every sibling
    boundary in this file gives."""
    binary = np.zeros((3, 3), dtype="<f4").tobytes()
    data = _minimal(
        [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}],
        [{"buffer": 0, "byteOffset": 0, "byteLength": len(binary)}],
        binary,
        meshes=[{"primitives": [{"attributes": {"POSITION": 5}}]}],
    )
    with pytest.raises(ValueError, match="accessor reference is 5"):
        gltf.load(data)


def test_an_out_of_range_buffer_view_referenced_by_an_accessor_is_refused_with_a_value_error_not_an_index_error():  # noqa: E501
    """An accessor's own ``bufferView`` was neither type- nor range-checked at
    all -- unlike every sibling boundary in this file -- despite a comment on
    the image/bufferView branch (``_image_bytes``) claiming this one already
    raised the named refusal. It reached ``bufferViews[99]`` as a bare
    ``IndexError`` instead."""
    data = _minimal(
        [{"bufferView": 99, "componentType": 5126, "count": 1, "type": "VEC3"}],
        [],
        b"",
    )
    with pytest.raises(ValueError, match="bufferView 99"):
        gltf.load(data)


def test_a_negative_inverse_bind_matrices_accessor_index_is_refused_rather_than_wrapping_onto_another_accessor():  # noqa: E501
    """Same boundary, one accessor field over: a skin's own
    ``inverseBindMatrices`` reaches ``accessor()`` the same unchecked way a
    primitive attribute does, so ``-1`` wrapped onto whichever accessor
    happens to sit last in the file's list -- here, a MAT4 accessor that
    is not the skin's own inverse-bind data at all -- and loaded clean."""
    positions = np.zeros((3, 3), dtype="<f4").tobytes()
    decoy_ibm = np.tile(np.eye(4, dtype="<f4"), (1, 1, 1)).tobytes()
    binary = positions + decoy_ibm
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0, 1]}],
        "nodes": [{"mesh": 0, "skin": 0}, {"name": "joint"}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        # -1 wraps onto accessor 1 (a real, but unrelated, MAT4 accessor)
        # instead of being refused.
        "skins": [{"joints": [1], "inverseBindMatrices": -1}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(positions)},
            {"buffer": 0, "byteOffset": len(positions), "byteLength": len(decoy_ibm)},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5126, "count": 1, "type": "MAT4"},
        ],
    }
    with pytest.raises(ValueError, match="accessor reference is -1"):
        gltf.load(_glb(doc, binary))


def test_a_file_declaring_more_markers_than_this_build_holds_is_refused(monkeypatch):
    """finding clay-05's argument at its third and fourth name: a camera and a
    light are charged against no byte budget at all -- neither decodes an
    accessor or an image -- so a million of either is a hang with no bytes to
    refuse."""
    monkeypatch.setattr(gltf, "MAX_CAMERAS", 2)
    monkeypatch.setattr(gltf, "MAX_LIGHTS", 2)
    empty = {"type": "perspective", "perspective": {"yfov": 1.0, "znear": 0.1}}
    with pytest.raises(ValueError, match="declares 3 cameras"):
        gltf.load(_marker_glb(cameras=[empty] * 3))
    with pytest.raises(ValueError, match="declares 3 lights"):
        gltf.load(
            _marker_glb(extensions={"KHR_lights_punctual": {"lights": [{"type": "point"}] * 3}})
        )
