"""Regression tests for the 2026-09-15 audit's glTF loader findings.

Two rows, both in ``src/warlock/kernels/geom3d/gltf.py``:

- clay-02: ``_Reader.decoded()``'s normalized-integer branch allocated and
  divided without charging either against ``MAX_TOTAL_BYTES``, and ran again
  for every primitive that shared the accessor because the conversion sat
  ahead of ``_typed_cache`` rather than behind a cache of its own.
- clay-03: a skin with no ``"joints"`` key raised a bare ``KeyError`` instead
  of this loader's own named ``ValueError`` refusal.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from warlock.kernels.geom3d import gltf
from warlock.kernels.geom3d.glbio import rebuild_glb


def _glb(gltf_json: dict, binary: bytes) -> bytes:
    """Wrap a JSON chunk and a BIN chunk as a GLB, the way an exporter would."""
    binary += b"\x00" * (-len(binary) % 4)
    chunk = struct.pack("<II", len(binary), 0x004E4942) + binary
    header = struct.pack("<III", 0x46546C67, 2, 0)
    return rebuild_glb(header, gltf_json, chunk)


def test_a_normalized_position_or_uv_accessors_conversion_is_charged_against_the_document_byte_budget(  # noqa: E501
    monkeypatch,
):
    """clay-02 (2026-09-15 audit): ``decoded()``'s normalized branch built
    ``raw.astype("f4") / info.max`` -- two full-size float32 arrays -- with
    neither charged against ``MAX_TOTAL_BYTES``, despite the module
    docstring's promise that every ``.astype`` is charged. Reproduced
    upstream (the audit's ``clay-document-01.py``) as 1.6 MB allocated, 0
    bytes charged.

    99 vertices (a multiple of 3, so create-01's unindexed-vertex-count
    refusal does not fire) of a normalized u16 TEXCOORD_0, alongside an
    ordinary f32 POSITION. Charged with this fix in place: POSITION's raw
    read (1,188 B) + its ``_typed`` retype (1,188 B) + the unindexed arange
    indices (396 B) + TEXCOORD_0's raw read (396 B) + its normalized
    conversion (792 B astype + 792 B divide, both new with this fix) + its
    own ``_typed`` retype (792 B) = 5,148 B, over a 4,000-byte budget. Without
    the fix, the two decoded() allocations (1,584 B) are never charged, so
    the same file's charged total is only 3,564 B -- under the same budget,
    and the load that should be refused for its true ~5 KB footprint
    succeeds instead.
    """
    monkeypatch.setattr(gltf, "MAX_TOTAL_BYTES", 4_000)
    n = 99
    positions = np.zeros((n, 3), dtype="<f4")
    uvs = np.zeros((n, 2), dtype="<u2")
    binary = positions.tobytes() + uvs.tobytes()
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}}]}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": positions.nbytes},
            {"buffer": 0, "byteOffset": positions.nbytes, "byteLength": uvs.nbytes},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": n, "type": "VEC3"},
            {
                "bufferView": 1,
                "componentType": 5123,
                "count": n,
                "type": "VEC2",
                "normalized": True,
            },
        ],
    }
    with pytest.raises(ValueError, match="byte budget"):
        gltf.load(_glb(doc, binary))


def test_a_repeated_normalized_accessor_is_converted_and_charged_once(monkeypatch):
    """clay-02, the other half: an instanced mesh's shared, normalized
    TEXCOORD_0 stream must cost what one primitive costs, not fifty -- the
    same H01 dedup ``accessor()``/``_typed`` already give every other
    attribute, which the normalized branch skipped because it ran ahead of
    any cache at all.
    """
    monkeypatch.setattr(gltf, "MAX_TOTAL_BYTES", 20_000)
    n = 3
    positions = np.zeros((n, 3), dtype="<f4")
    uvs = np.array([[0, 0], [65535, 0], [0, 32768]], dtype="<u2")
    binary = positions.tobytes() + uvs.tobytes()
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {
                "primitives": [
                    {"attributes": {"POSITION": 0, "TEXCOORD_0": 1}} for _ in range(20)
                ]
            }
        ],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": positions.nbytes},
            {"buffer": 0, "byteOffset": positions.nbytes, "byteLength": uvs.nbytes},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": n, "type": "VEC3"},
            {
                "bufferView": 1,
                "componentType": 5123,
                "count": n,
                "type": "VEC2",
                "normalized": True,
            },
        ],
    }
    model = gltf.load(_glb(doc, binary))
    assert len(model.meshes[0]) == 20
    first = model.meshes[0][0].uvs
    assert all(p.uvs is first for p in model.meshes[0])
    assert np.allclose(first, [[0.0, 0.0], [1.0, 0.0], [0.0, 0.5]], atol=1e-4)


def test_a_skin_with_no_joints_key_raises_a_named_refusal_not_a_bare_keyerror():
    """clay-03 (2026-09-15 audit): ``skin()`` indexed ``skin["joints"]``
    straight off the JSON with no check, so a skin entry missing the
    (glTF-required, but ``check_glb`` at the import door is
    structural-only) ``joints`` array raised a bare ``KeyError`` instead of
    this loader's own named refusal every sibling boundary raises.
    """
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": []}],
        "nodes": [],
        "skins": [{}],
    }
    with pytest.raises(ValueError, match="joints"):
        gltf.load(_glb(doc, b""))
