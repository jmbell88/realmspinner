"""Regression tests for the 2026-09-23 audit's glTF loader findings.

- clay-09: a non-dict ``accessors``/``bufferViews``/``images``/``textures``
  entry in ``kernels/geom3d/gltf.py`` raised a bare ``TypeError``/
  ``AttributeError`` instead of this loader's own named ``ValueError``.
- clay-10: ``pbrMetallicRoughness`` and ``attributes`` were used as mappings
  with no isinstance check, unlike the sibling sub-block fallback
  ``camera()``/``light()`` already use.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from realmspinner.kernels.geom3d import gltf
from realmspinner.kernels.geom3d.glbio import rebuild_glb


def _glb(gltf_json: dict, binary: bytes) -> bytes:
    """Wrap a JSON chunk and a BIN chunk as a GLB, the way an exporter would
    -- the same helper ``test_audit_2026_09_15_gltf.py`` uses."""
    binary += b"\x00" * (-len(binary) % 4)
    chunk = struct.pack("<II", len(binary), 0x004E4942) + binary
    header = struct.pack("<III", 0x46546C67, 2, 0)
    return rebuild_glb(header, gltf_json, chunk)


# --- clay-09: GLB loader entry types -------------------------------------------


def _minimal_doc(**accessor_overrides):
    n = 3
    positions = np.zeros((n, 3), dtype="<f4")
    binary = positions.tobytes()
    accessor = {"bufferView": 0, "componentType": 5126, "count": n, "type": "VEC3"}
    accessor.update(accessor_overrides)
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": positions.nbytes}],
        "accessors": [accessor],
    }
    return doc, binary


def test_a_non_dict_accessor_bufferview_image_or_texture_entry_is_refused_with_a_value_error():
    """clay-09 (2026-09-23 audit): each of these arrays used to be indexed
    straight into with no check that the entry itself was an object, so a
    GLB that is legal JSON but puts ``null``/a string/a list in one of them
    raised a bare ``TypeError``/``AttributeError`` instead of this loader's
    own named ``ValueError`` (the same rule ``_check_dict_entry`` already
    states and enforces for ``nodes``/``meshes[].primitives``/``materials``/
    ``skins`` entries)."""
    # accessors[0] is not a dict.
    doc, binary = _minimal_doc()
    doc["accessors"] = [None]
    with pytest.raises(ValueError, match="accessor 0"):
        gltf.load(_glb(doc, binary))

    # bufferViews[0], referenced by an accessor, is not a dict.
    doc, binary = _minimal_doc()
    doc["bufferViews"] = ["not a dict"]
    with pytest.raises(ValueError, match="bufferView 0"):
        gltf.load(_glb(doc, binary))

    # textures[0], referenced by a material, is not a dict.
    doc, binary = _minimal_doc()
    doc["materials"] = [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}]
    doc["meshes"][0]["primitives"][0]["material"] = 0
    doc["textures"] = [42]
    with pytest.raises(ValueError, match="texture 0"):
        gltf.load(_glb(doc, binary))

    # images[0], referenced by a texture, is not a dict.
    doc, binary = _minimal_doc()
    doc["materials"] = [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}]
    doc["meshes"][0]["primitives"][0]["material"] = 0
    doc["textures"] = [{"source": 0}]
    doc["images"] = [[]]
    with pytest.raises(ValueError, match="image 0"):
        gltf.load(_glb(doc, binary))


# --- clay-10: GLB loader sub-objects -------------------------------------------


def test_a_non_dict_pbr_or_attributes_block_is_refused_not_a_bare_typeerror():
    """clay-10 (2026-09-23 audit): ``pbrMetallicRoughness`` and
    ``attributes`` were read with ``.get(key, {})`` and used as mappings with
    no isinstance check, unlike the sibling sub-block fallback
    ``camera()``/``light()`` already use for ``perspective``/``spot``. A
    material whose "pbrMetallicRoughness" is legal JSON but not an object
    used to raise a bare ``AttributeError`` off ``pbr.get(...)``; a primitive
    whose "attributes" is a list raised a bare ``TypeError`` off
    ``attrs["POSITION"]``. Both must now behave exactly as if the field were
    simply absent."""
    # pbrMetallicRoughness is a list, not a dict -- must not raise at all,
    # the same as an absent one (falls back to {}, same as camera()/light()).
    doc, binary = _minimal_doc()
    doc["materials"] = [{"pbrMetallicRoughness": ["not", "a", "dict"]}]
    doc["meshes"][0]["primitives"][0]["material"] = 0
    model = gltf.load(_glb(doc, binary))
    assert model.meshes[0][0].material is not None

    # attributes is a list, not a dict -- POSITION is "missing" the same way
    # an ordinary primitive with no POSITION at all is refused.
    doc, binary = _minimal_doc()
    doc["meshes"][0]["primitives"][0]["attributes"] = ["POSITION"]
    with pytest.raises(ValueError, match="POSITION"):
        gltf.load(_glb(doc, binary))
