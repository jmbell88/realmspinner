"""The GLB writer, verified by round-tripping through the loader that ships.

A writer tested against its own idea of the format is a writer that agrees with
itself. The strongest test available here is the loader in ``viewer/gltf.py``:
it is independently written, it is what every other GLB in this project is read
by, and it decodes accessors from the bytes rather than from the intent. So
almost everything here authors a ``gltf.Model``, writes it and loads it back.

The exceptions are the assertions about the *container* -- magic, chunk lengths,
alignment, the absence of an optional attribute -- which a round trip cannot
see, because the loader is tolerant of exactly the things a strict third-party
reader is not.
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from warlock.glbio import read_glb
from warlock.studio.viewer import glbwrite, gltf
from warlock.studio.viewer import math3d as m3


def _quad(offset: float = 0.0, *, uvs: bool = False) -> gltf.Primitive:
    positions = np.array(
        [[0, 0, offset], [1, 0, offset], [0, 1, offset], [1, 1, offset]], dtype="f4"
    )
    return gltf.Primitive(
        positions=positions,
        indices=np.array([0, 1, 2, 1, 3, 2], dtype="u4"),
        normals=np.tile(np.array([0, 0, 1], dtype="f4"), (4, 1)),
        uvs=np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype="f4") if uvs else None,
    )


def _materials() -> list[gltf.Material]:
    return [
        gltf.Material(
            name="red",
            base_color_factor=(0.9, 0.1, 0.2, 1.0),
            metallic_factor=0.25,
            roughness_factor=0.75,
        ),
        gltf.Material(
            name="glow",
            base_color_factor=(0.1, 0.1, 0.1, 1.0),
            metallic_factor=0.0,
            roughness_factor=0.5,
            emissive_factor=(0.4, 0.6, 0.8),
        ),
        gltf.Material(
            name="leaf",
            base_color_factor=(0.2, 0.7, 0.3, 0.5),
            metallic_factor=1.0,
            roughness_factor=0.1,
            double_sided=True,
        ),
    ]


def _model() -> tuple[gltf.Model, list[gltf.Material]]:
    """Two nodes with distinct TRS; the first carries two primitives."""
    mats = _materials()
    a, b, c = _quad(0.0), _quad(1.0), _quad(2.0)
    a.material, b.material, c.material = mats[0], mats[1], mats[2]
    nodes = [
        gltf.Node(
            name="first",
            translation=m3.vec3(1.0, 2.0, 3.0),
            rotation=m3.quat_from_axis_angle(m3.vec3(0.0, 1.0, 0.0), 0.5),
            scale=m3.vec3(2.0, 0.5, 1.5),
            mesh=0,
        ),
        gltf.Node(
            name="second",
            translation=m3.vec3(-4.0, 0.0, 0.25),
            rotation=m3.quat_from_axis_angle(m3.vec3(1.0, 0.0, 0.0), -1.1),
            scale=m3.vec3(1.0, 1.0, 1.0),
            mesh=1,
        ),
    ]
    return gltf.Model(nodes, [0, 1], [[a, b], [c]], []), mats


def _roundtrip(model: gltf.Model) -> gltf.Model:
    return gltf.load(glbwrite.write_glb(model))


# --- geometry ----------------------------------------------------------------


def test_positions_normals_and_indices_survive_exactly() -> None:
    """Exact, not ``allclose``: everything here is already f4 on the way in, so
    any drift means a conversion happened that should not have."""
    model, _ = _model()
    out = _roundtrip(model)

    assert len(out.meshes) == 2
    assert [len(prims) for prims in out.meshes] == [2, 1]
    for before, after in zip(
        [p for prims in model.meshes for p in prims],
        [p for prims in out.meshes for p in prims],
        strict=True,
    ):
        assert np.array_equal(before.positions, after.positions)
        assert np.array_equal(before.normals, after.normals)
        assert np.array_equal(before.indices, after.indices)


def test_each_nodes_trs_survives() -> None:
    model, _ = _model()
    out = _roundtrip(model)

    assert [n.name for n in out.nodes] == ["first", "second"]
    assert out.roots == [0, 1]
    for before, after in zip(model.nodes, out.nodes, strict=True):
        assert np.allclose(before.translation, after.translation)
        assert np.allclose(before.rotation, after.rotation)  # XYZW
        assert np.allclose(before.scale, after.scale)
        assert before.mesh == after.mesh


def test_a_node_hierarchy_survives() -> None:
    parent = gltf.Node(name="parent", translation=m3.vec3(0.0, 5.0, 0.0), children=[1])
    child = gltf.Node(name="child", translation=m3.vec3(1.0, 0.0, 0.0), mesh=0)
    out = _roundtrip(gltf.Model([parent, child], [0], [[_quad()]], []))

    assert out.nodes[0].children == [1]
    assert out.roots == [0]
    # The child's world transform is its parent's composed with its own, which
    # is the property a flat re-parenting would quietly destroy.
    assert np.allclose(out.nodes[1].world[:3, 3], [1.0, 5.0, 0.0])


# --- materials ---------------------------------------------------------------


def test_every_material_factor_survives() -> None:
    model, mats = _model()
    out = _roundtrip(model)
    written = [p.material for prims in out.meshes for p in prims]

    for before, after in zip(mats, written, strict=True):
        assert after.name == before.name
        assert np.allclose(after.base_color_factor, before.base_color_factor)
        assert after.metallic_factor == pytest.approx(before.metallic_factor)
        assert after.roughness_factor == pytest.approx(before.roughness_factor)
        assert np.allclose(after.emissive_factor, before.emissive_factor)
        assert after.double_sided == before.double_sided


def test_the_primitive_to_material_assignment_survives() -> None:
    model, mats = _model()
    out = _roundtrip(model)
    assert [p.material.name for p in out.meshes[0]] == ["red", "glow"]
    assert [p.material.name for p in out.meshes[1]] == ["leaf"]
    assert [m.name for m in mats] == ["red", "glow", "leaf"]


def test_a_shared_material_is_written_once() -> None:
    """``to_model`` hands every primitive the palette entry itself, so identity
    de-duplication here is what keeps a twelve-object document at one material
    rather than twelve copies of one."""
    shared = gltf.Material(name="shared", base_color_factor=(0.3, 0.3, 0.3, 1.0))
    a, b = _quad(0.0), _quad(1.0)
    a.material = b.material = shared
    model = gltf.Model([gltf.Node(name="n", mesh=0)], [0], [[a, b]], [])

    doc, _ = read_glb(glbwrite.write_glb(model))
    assert len(doc["materials"]) == 1
    assert {p["material"] for p in doc["meshes"][0]["primitives"]} == {0}


def test_two_equal_but_distinct_materials_are_written_separately() -> None:
    """De-duplication is by identity, not by value -- the palette is a list of
    slots the user edits, and collapsing two that happen to match today would
    make editing one of them change the other."""
    a, b = _quad(0.0), _quad(1.0)
    a.material = gltf.Material(name="same")
    b.material = gltf.Material(name="same")
    model = gltf.Model([gltf.Node(name="n", mesh=0)], [0], [[a, b]], [])

    doc, _ = read_glb(glbwrite.write_glb(model))
    assert len(doc["materials"]) == 2


# --- optional attributes -----------------------------------------------------


def test_a_model_without_uvs_writes_no_texcoord_and_still_loads() -> None:
    """TEXCOORD_0 is optional in the spec, and Clay Phase 1 has no textures.
    Writing a zero-filled one would be four bytes a vertex of a lie."""
    model = gltf.Model([gltf.Node(name="n", mesh=0)], [0], [[_quad(uvs=False)]], [])
    data = glbwrite.write_glb(model)

    doc, _ = read_glb(data)
    attrs = doc["meshes"][0]["primitives"][0]["attributes"]
    assert "POSITION" in attrs
    assert "TEXCOORD_0" not in attrs

    out = gltf.load(data)
    assert out.meshes[0][0].uvs is None


def test_uvs_survive_when_they_are_there() -> None:
    model = gltf.Model([gltf.Node(name="n", mesh=0)], [0], [[_quad(uvs=True)]], [])
    out = _roundtrip(model)
    assert np.array_equal(out.meshes[0][0].uvs, _quad(uvs=True).uvs)


def test_a_primitive_without_normals_writes_no_normal_attribute() -> None:
    prim = _quad()
    prim.normals = None
    model = gltf.Model([gltf.Node(name="n", mesh=0)], [0], [[prim]], [])

    doc, _ = read_glb(glbwrite.write_glb(model))
    assert "NORMAL" not in doc["meshes"][0]["primitives"][0]["attributes"]
    assert gltf.load(glbwrite.write_glb(model)).meshes[0][0].normals is None


# --- the container -----------------------------------------------------------


def test_the_bytes_are_a_well_formed_glb() -> None:
    model, _ = _model()
    data = glbwrite.write_glb(model)

    magic, version, total = struct.unpack_from("<III", data, 0)
    assert magic == 0x46546C67  # "glTF"
    assert version == 2
    assert total == len(data)

    offset = 12
    seen = []
    while offset < len(data):
        length, kind = struct.unpack_from("<II", data, offset)
        seen.append(kind)
        assert offset + 8 + length <= len(data)
        assert length % 4 == 0
        offset += 8 + length
    assert offset == len(data)
    assert seen == [0x4E4F534A, 0x004E4942]  # JSON then BIN, in that order


def test_the_json_chunk_is_padded_with_spaces_and_parses() -> None:
    """Spaces, not zeros -- the spec says so, and a strict reader that hands the
    chunk straight to a JSON parser chokes on a trailing NUL."""
    model, _ = _model()
    data = glbwrite.write_glb(model)
    length, _kind = struct.unpack_from("<II", data, 12)
    chunk = data[20 : 20 + length]

    assert chunk.rstrip(b" ") == chunk.rstrip()
    assert json.loads(chunk.decode())["asset"]["version"] == "2.0"


def test_every_buffer_view_is_four_byte_aligned() -> None:
    """An accessor is read with ``np.frombuffer`` at the view's offset, and an
    f4 read from an odd offset is a misaligned read on the platforms that still
    care and a silently wrong number on the ones that do not."""
    model, _ = _model()
    doc, binary = read_glb(glbwrite.write_glb(model))

    for view in doc["bufferViews"]:
        assert view.get("byteOffset", 0) % 4 == 0
        assert view.get("byteOffset", 0) + view["byteLength"] <= len(binary)
    assert doc["buffers"][0]["byteLength"] == len(binary)


def test_the_position_accessor_carries_min_and_max() -> None:
    """Required by the spec, and most viewers frame the scene from them: without
    them the model loads and sits somewhere unhelpful in the frame."""
    model, _ = _model()
    doc, _ = read_glb(glbwrite.write_glb(model))

    prim = doc["meshes"][0]["primitives"][0]
    accessor = doc["accessors"][prim["attributes"]["POSITION"]]
    assert accessor["min"] == [0.0, 0.0, 0.0]
    assert accessor["max"] == [1.0, 1.0, 0.0]
    # And only there: min/max on every accessor is legal but is noise.
    assert "min" not in doc["accessors"][prim["indices"]]


def test_a_small_mesh_writes_unsigned_short_indices() -> None:
    model, _ = _model()
    doc, _ = read_glb(glbwrite.write_glb(model))
    prim = doc["meshes"][0]["primitives"][0]
    assert doc["accessors"][prim["indices"]]["componentType"] == 5123  # USHORT


def test_a_mesh_past_65535_vertices_writes_unsigned_int_indices() -> None:
    """The componentType has to agree with the bytes actually written, and the
    boundary is the one place a writer can disagree with itself."""
    n = 70000
    prim = gltf.Primitive(
        positions=np.zeros((n, 3), dtype="f4"),
        indices=np.array([0, 1, n - 1], dtype="u4"),
    )
    model = gltf.Model([gltf.Node(name="n", mesh=0)], [0], [[prim]], [])
    data = glbwrite.write_glb(model)

    doc, _ = read_glb(data)
    accessor = doc["accessors"][doc["meshes"][0]["primitives"][0]["indices"]]
    assert accessor["componentType"] == 5125  # UINT
    assert np.array_equal(gltf.load(data).meshes[0][0].indices, [0, 1, n - 1])


def test_an_empty_model_writes_no_buffer_at_all() -> None:
    """``byteLength`` has a minimum of one in the spec, so an empty buffer is
    not a degenerate file but an invalid one -- and a glTF with no buffer is
    perfectly legal. An empty scene must not be rejected for a reason that has
    nothing to do with what is in it."""
    data = glbwrite.write_glb(gltf.Model([], [], [], []))
    doc, binary = read_glb(data)
    assert "buffers" not in doc
    assert binary == b""

    out = gltf.load(data)
    assert out.nodes == []
    assert out.meshes == []


def test_a_mesh_with_no_geometry_is_dropped_rather_than_written_empty() -> None:
    """An object in the outliner with nothing in it yet is a state the document
    can be in, and neither half of it may reach the file: an accessor's count
    and a mesh's ``primitives`` both have a minimum of one in the spec, so a
    strict reader rejects the whole file over either. The node survives -- it
    is a real object with a real transform -- carrying no mesh."""
    prim = gltf.Primitive(
        positions=np.zeros((0, 3), dtype="f4"), indices=np.zeros(0, dtype="u4")
    )
    model = gltf.Model([gltf.Node(name="n", mesh=0)], [0], [[prim]], [])

    doc, _ = read_glb(glbwrite.write_glb(model))
    assert "meshes" not in doc
    assert "mesh" not in doc["nodes"][0]
    assert gltf.load(glbwrite.write_glb(model)).meshes == []


def test_dropping_an_empty_mesh_renumbers_the_ones_that_survive() -> None:
    """The drop shifts every later index, so a node pointing past it would
    otherwise render a different object's geometry."""
    empty = gltf.Primitive(
        positions=np.zeros((0, 3), dtype="f4"), indices=np.zeros(0, dtype="u4")
    )
    real = gltf.Primitive(
        positions=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="f4"),
        indices=np.array([0, 1, 2], dtype="u4"),
    )
    model = gltf.Model(
        [gltf.Node(name="gone", mesh=0), gltf.Node(name="kept", mesh=1)],
        [0, 1],
        [[empty], [real]],
        [],
    )

    doc, _ = read_glb(glbwrite.write_glb(model))
    assert len(doc["meshes"]) == 1
    assert "mesh" not in doc["nodes"][0]
    assert doc["nodes"][1]["mesh"] == 0


def test_a_model_with_a_skin_is_refused_rather_than_silently_flattened() -> None:
    """Clay has no skins. Dropping one would export a rig-shaped file with
    no rig in it, which fails a long way from here."""
    model = gltf.Model([gltf.Node(name="n")], [0], [], [gltf.Skin([0], np.eye(4)[None])])
    with pytest.raises(ValueError, match="skin"):
        glbwrite.write_glb(model)


def test_a_glb_whose_json_chunk_overruns_the_file_is_refused() -> None:
    """A body cut short mid-transfer used to reach json.loads as a partial
    document, so what came back named the JSON decoder rather than the
    truncation -- and ``_validate_glb``'s message is what a user reads when
    trellis-server dies mid-response."""
    model = gltf.Model([gltf.Node(name="n")], [0], [], [])
    data = glbwrite.write_glb(model)
    with pytest.raises(ValueError, match="truncated GLB"):
        read_glb(data[: len(data) // 2])
    # Lying in the header is the same failure without losing a byte.
    chunk_len = struct.unpack_from("<I", data, 12)[0]
    lied = data[:12] + struct.pack("<I", chunk_len + len(data)) + data[16:]
    with pytest.raises(ValueError, match="truncated GLB"):
        read_glb(lied)


# --- the writer's bytes, held still across the Mason change -------------------


def _clay_document():
    """A small, fully-determined Clay document: two primitives and a palette.

    Deliberately untextured. A texture would put Pillow's PNG encoder into the
    bytes this file pins, which is a dependency's version rather than this
    writer's behaviour -- and the claim below is about *this* writer.
    """
    from warlock.studio.clay import document as bd
    from warlock.studio.clay import mesh as bm
    from warlock.studio.clay import primitives as bp

    doc = bd.ClayDoc(
        materials=[
            gltf.Material(
                name="stone",
                base_color_factor=(0.55, 0.55, 0.6, 1.0),
                metallic_factor=0.0,
                roughness_factor=0.85,
            ),
            gltf.Material(
                name="brass",
                base_color_factor=(0.8, 0.6, 0.2, 1.0),
                metallic_factor=1.0,
                roughness_factor=0.25,
                emissive_factor=(0.05, 0.02, 0.0),
            ),
        ]
    )
    box = bd.Obj(uid=1, name="Plinth", mesh=bp.box(size=(2.0, 0.5, 2.0)))
    box.translation = m3.vec3(0.0, 0.25, 0.0)
    cyl = bd.Obj(uid=2, name="Column", mesh=bp.cylinder(radius=0.4, height=3.0, segments=12))
    cyl.translation = m3.vec3(0.0, 2.0, 0.0)
    cyl.rotation = m3.quat_from_axis_angle(m3.vec3(0.0, 1.0, 0.0), 0.25)
    # On the second palette slot, so the file carries both materials rather
    # than one: a pin that only ever exercised slot 0 would not notice a
    # change to how the other is written.
    cap = bp.cone(radius=0.5, height=0.6, segments=12)
    cap = bm.Mesh(
        positions=cap.positions,
        loops=cap.loops,
        starts=cap.starts,
        material=np.ones_like(cap.material),
        smooth=cap.smooth,
    )
    top = bd.Obj(uid=3, name="Capital", mesh=cap)
    top.translation = m3.vec3(0.0, 3.7, 0.0)
    top.scale = m3.vec3(1.2, 1.0, 1.2)
    doc.objects.extend([box, cyl, top])
    return doc


#: sha256 of ``write_glb(to_model(_clay_document()))``, recorded at 8ab32200 --
#: the commit *before* Mason's cameras-and-lights change to this writer and to
#: ``viewer/gltf.py``. See the test below for what it is claiming.
_CLAY_GLB_SHA256 = "2bffd0afac34eaeb489570a806c79e1a3411c3678daa4e6fd1de3a630922a95b"


def test_an_existing_clay_document_writes_the_same_bytes_as_before_lights_arrived() -> None:
    """Mason's one change to a shared file costs Clay, Poser and Troupe nothing.

    Cameras and lights were added to this writer and to ``viewer/gltf.py`` for
    Mason's scene export (Stage D), and all three of those modes hand this
    function models that carry neither. The rule the change was made under is
    that every new key is emitted **only when one exists**, which is a claim
    about bytes rather than about intent -- so this pins the bytes, digested
    from a run against the writer as it was before any of it landed.

    A failure here does not mean this document changed: it means the writer
    did, for a file with no camera and no light in it, which is precisely what
    the change promised not to do.
    """
    import hashlib

    from warlock.studio.clay import document as bd

    data = glbwrite.write_glb(bd.to_model(_clay_document()))
    assert hashlib.sha256(data).hexdigest() == _CLAY_GLB_SHA256


# --- cameras and lights ------------------------------------------------------


def test_a_model_with_no_light_or_camera_declares_neither_and_no_extension() -> None:
    """The emitted-only-when-one-exists rule, asserted on the JSON itself.

    The digest above says Clay's bytes did not move; this says *why* they did
    not, which is the part that survives a legitimate future change to the
    document that digest is taken from. An empty ``cameras`` array or an
    ``extensionsUsed`` naming an extension the file does not use are both
    things a strict validator complains about and some importers act on.
    """
    model, _ = _model()
    doc, _ = read_glb(glbwrite.write_glb(model))
    assert "cameras" not in doc
    assert "extensions" not in doc
    assert "extensionsUsed" not in doc
    assert all("camera" not in n and "extensions" not in n for n in doc["nodes"])


def test_a_camera_survives_the_round_trip_with_its_node() -> None:
    model = gltf.Model(
        [gltf.Node(name="eye", translation=m3.vec3(0.0, 1.6, 4.0), camera=0)],
        [0],
        [],
        [],
        cameras=[gltf.Camera(name="main", yfov=0.8, znear=0.05, zfar=250.0)],
    )
    out = _roundtrip(model)
    assert len(out.cameras) == 1
    assert out.cameras[0].name == "main"
    assert out.cameras[0].yfov == pytest.approx(0.8)
    assert out.cameras[0].znear == pytest.approx(0.05)
    assert out.cameras[0].zfar == pytest.approx(250.0)
    assert out.nodes[0].camera == 0


def test_a_camera_with_no_zfar_writes_none_rather_than_a_zero() -> None:
    """Zero means absent -- and absent is glTF's infinite perspective, where a
    literal ``zfar`` of 0 is a frustum that clips everything."""
    model = gltf.Model([gltf.Node(name="eye", camera=0)], [0], [], [], cameras=[gltf.Camera()])
    doc, _ = read_glb(glbwrite.write_glb(model))
    persp = doc["cameras"][0]["perspective"]
    assert "zfar" not in persp and "aspectRatio" not in persp
    assert doc["cameras"][0]["type"] == "perspective"
    assert _roundtrip(model).cameras[0].zfar == 0.0


def test_all_three_light_kinds_survive_the_round_trip() -> None:
    """Directional, point and spot, each with the fields its kind actually has.

    Written as one test over the three rather than three tests because the
    claim is about the set: KHR_lights_punctual has exactly these kinds, and a
    writer that handled two of them would pass two tests out of three and lose
    a third of a scene's lighting.
    """
    lights = [
        gltf.Light(name="sun", kind="directional", color=(1.0, 0.95, 0.8), intensity=3.0),
        gltf.Light(name="lamp", kind="point", color=(1.0, 0.5, 0.2), intensity=40.0, range=12.0),
        gltf.Light(
            name="spot",
            kind="spot",
            intensity=15.0,
            range=8.0,
            inner_cone_angle=0.2,
            outer_cone_angle=0.6,
        ),
    ]
    nodes = [gltf.Node(name=f"light{i}", light=i) for i in range(3)]
    out = _roundtrip(gltf.Model(nodes, [0, 1, 2], [], [], lights=lights))

    assert [light.kind for light in out.lights] == ["directional", "point", "spot"]
    assert [light.name for light in out.lights] == ["sun", "lamp", "spot"]
    assert out.lights[0].color == pytest.approx((1.0, 0.95, 0.8))
    assert out.lights[1].intensity == pytest.approx(40.0)
    assert out.lights[1].range == pytest.approx(12.0)
    assert out.lights[2].inner_cone_angle == pytest.approx(0.2)
    assert out.lights[2].outer_cone_angle == pytest.approx(0.6)
    assert [n.light for n in out.nodes] == [0, 1, 2]


def test_the_light_extension_is_declared_used_and_never_required() -> None:
    """A reader that has never heard of punctual lights still gets every mesh,
    which is the distinction between the two keys -- requiring the extension
    would turn an unlit-but-complete import into a refused file."""
    model = gltf.Model([gltf.Node(name="l", light=0)], [0], [], [], lights=[gltf.Light()])
    doc, _ = read_glb(glbwrite.write_glb(model))
    assert doc["extensionsUsed"] == ["KHR_lights_punctual"]
    assert "extensionsRequired" not in doc
    assert doc["extensions"]["KHR_lights_punctual"]["lights"][0]["type"] == "point"
    assert doc["nodes"][0]["extensions"]["KHR_lights_punctual"]["light"] == 0


def test_a_field_a_lights_kind_ignores_is_not_written_for_it() -> None:
    """``range`` on a directional light and a cone on anything but a spot are
    fields the extension says to ignore -- and several importers read them
    anyway, producing a different scene than the one exported."""
    model = gltf.Model(
        [gltf.Node(name="a", light=0), gltf.Node(name="b", light=1)],
        [0, 1],
        [],
        [],
        lights=[
            gltf.Light(kind="directional", range=50.0, outer_cone_angle=0.3),
            gltf.Light(kind="point", range=5.0, outer_cone_angle=0.3),
        ],
    )
    doc, _ = read_glb(glbwrite.write_glb(model))
    written = doc["extensions"]["KHR_lights_punctual"]["lights"]
    assert "range" not in written[0] and "spot" not in written[0]
    assert written[1]["range"] == pytest.approx(5.0) and "spot" not in written[1]


def test_a_node_naming_a_marker_the_model_does_not_carry_loses_the_reference() -> None:
    """An index past the end of an array is a file a strict reader rejects
    outright, and rejecting a whole scene over a marker is the wrong trade --
    the same answer the mesh remap gives for a mesh that was dropped."""
    model = gltf.Model(
        [gltf.Node(name="stale", camera=3, light=7)], [0], [], [], cameras=[], lights=[]
    )
    doc, _ = read_glb(glbwrite.write_glb(model))
    assert "camera" not in doc["nodes"][0]
    assert "extensions" not in doc["nodes"][0]
    assert _roundtrip(model).nodes[0].camera is None


def test_one_camera_and_one_light_shared_by_several_nodes_are_written_once() -> None:
    """Both arrays are the model's, not the nodes': three markers pointing at
    one light is one light in the file, the same way six nodes on one mesh
    index is one mesh."""
    nodes = [gltf.Node(name=f"n{i}", light=0, camera=0) for i in range(3)]
    doc, _ = read_glb(
        glbwrite.write_glb(
            gltf.Model(nodes, [0, 1, 2], [], [], cameras=[gltf.Camera()], lights=[gltf.Light()])
        )
    )
    assert len(doc["cameras"]) == 1
    assert len(doc["extensions"]["KHR_lights_punctual"]["lights"]) == 1
    assert all(n["camera"] == 0 for n in doc["nodes"])
