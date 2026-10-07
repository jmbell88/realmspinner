"""``Material.nearest`` as a glTF sampler: written crisp, read back crisp.

A texture Clay makes is meant to stay square-edged in every engine that opens
the GLB, and the only way to say so in glTF is a ``samplers`` entry with
NEAREST filters on the texture. A document with no such material must write
exactly the bytes it always did (``test_glbwrite.py`` pins the digest).
"""

from __future__ import annotations

import numpy as np

from realmspinner.kernels.geom3d import glbwrite, gltf
from realmspinner.kernels.geom3d.glbio import read_glb, rebuild_glb, split_glb

NEAREST = 9728


def other_image() -> tuple[int, int, bytes]:
    return (2, 2, bytes(reversed(range(16))))


def _prim(material: gltf.Material, offset: float = 0.0) -> gltf.Primitive:
    return gltf.Primitive(
        positions=np.array([[0, 0, offset], [1, 0, offset], [0, 1, offset]], dtype="f4"),
        indices=np.array([0, 1, 2], dtype="u4"),
        uvs=np.array([[0, 0], [1, 0], [0, 1]], dtype="f4"),
        material=material,
    )


def _write(*materials: gltf.Material) -> bytes:
    prims = [_prim(m, float(i)) for i, m in enumerate(materials)]
    nodes = [gltf.Node(mesh=i) for i in range(len(prims))]
    return glbwrite.write_glb(
        gltf.Model(nodes, list(range(len(nodes))), [[p] for p in prims], [])
    )


def _json(data: bytes) -> dict:
    return read_glb(data)[0]


def test_a_nearest_material_writes_a_nearest_sampler_on_its_texture() -> None:
    image = (2, 2, bytes(range(16)))

    doc = _json(_write(gltf.Material(base_color=image, nearest=True)))

    assert doc["samplers"] == [{"magFilter": NEAREST, "minFilter": NEAREST}]
    assert doc["textures"][0]["sampler"] == 0


def test_a_document_with_no_nearest_material_writes_no_samplers_key() -> None:
    image = (2, 2, bytes(range(16)))

    doc = _json(_write(gltf.Material(base_color=image), gltf.Material()))

    assert "samplers" not in doc
    assert all("sampler" not in texture for texture in doc["textures"])


def test_nearest_without_a_texture_writes_no_sampler() -> None:
    doc = _json(_write(gltf.Material(nearest=True)))

    assert "samplers" not in doc and "textures" not in doc


def test_one_image_shared_by_a_crisp_and_a_smooth_material_gets_two_textures_over_one_png() -> None:
    image = (2, 2, bytes(range(16)))

    doc = _json(
        _write(
            gltf.Material(name="crisp", base_color=image, nearest=True),
            gltf.Material(name="smooth", base_color=image),
        )
    )

    assert len(doc["images"]) == 1
    by_name = {m["name"]: m for m in doc["materials"]}
    crisp = by_name["crisp"]["pbrMetallicRoughness"]["baseColorTexture"]["index"]
    smooth = by_name["smooth"]["pbrMetallicRoughness"]["baseColorTexture"]["index"]
    assert crisp != smooth
    assert doc["textures"][crisp]["sampler"] == 0
    assert "sampler" not in doc["textures"][smooth]


def test_two_crisp_materials_over_one_image_still_share_one_texture() -> None:
    image = (2, 2, bytes(range(16)))

    doc = _json(
        _write(
            gltf.Material(name="a", base_color=image, nearest=True),
            gltf.Material(name="b", base_color=image, nearest=True),
        )
    )

    assert len(doc["textures"]) == 1 and len(doc["samplers"]) == 1


def test_only_the_base_colour_slot_is_sampled_crisp() -> None:
    image = (2, 2, bytes(range(16)))
    other = (2, 2, bytes(reversed(range(16))))

    doc = _json(_write(gltf.Material(base_color=image, normal=other, nearest=True)))

    material = doc["materials"][0]
    base = material["pbrMetallicRoughness"]["baseColorTexture"]["index"]
    normal = material["normalTexture"]["index"]
    assert doc["textures"][base]["sampler"] == 0
    assert "sampler" not in doc["textures"][normal]


def test_the_loader_reads_the_sampler_back_as_nearest() -> None:
    image = (2, 2, bytes(range(16)))

    model = gltf.load(
        _write(
            gltf.Material(name="crisp", base_color=image, nearest=True),
            gltf.Material(name="smooth", base_color=other_image()),
        )
    )

    by_name = {p.material.name: p.material for prims in model.meshes for p in prims}
    assert by_name["crisp"].nearest is True
    assert by_name["smooth"].nearest is False


def test_a_linear_sampler_or_a_malformed_one_loads_as_smooth() -> None:
    data = _write(gltf.Material(base_color=(2, 2, bytes(16)), nearest=True))
    header, doc, chunk = split_glb(data)
    for bad in (
        [{"magFilter": 9729, "minFilter": 9987}],  # LINEAR
        [None],
        "nope",
    ):
        doc["samplers"] = bad
        loaded = gltf.load(rebuild_glb(header, doc, chunk))
        assert loaded.meshes[0][0].material.nearest is False
    doc["samplers"] = [{"magFilter": NEAREST}]
    doc["textures"][0]["sampler"] = 7  # out of range
    assert gltf.load(rebuild_glb(header, doc, chunk)).meshes[0][0].material.nearest is False
