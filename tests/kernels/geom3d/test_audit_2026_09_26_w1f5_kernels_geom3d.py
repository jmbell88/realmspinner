"""Findings closed from the 2026-09-26 audit, gltf-loader slice (w1f5).

create-viewer-01 (+clay-io-05): a skin with no ``inverseBindMatrices`` tiled
128 bytes per listed joint (``np.tile(np.eye(4), (len(joints), 1, 1))``) with
no charge against ``MAX_TOTAL_BYTES`` at all -- unlike the
``inverseBindMatrices``-present branch, which charges through
``self.accessor()`` like every other decode -- and with no ceiling on
``len(joints)`` either, even though every entry in it was already
bounds-checked against the node count (clay-io-05/create2-01). Reproduced by
the explorer: a 6-9 MB GLB declaring a skin with 3,000,000 joint entries (a
handful of valid node indices repeated -- nothing here requires them unique)
built a 384-432 MB array uncounted.

clay-io-06: a negative ``byteStride`` was accepted by ``_check_span``, so the
gather wrapped and either read the wrong stream or raised a bare
``IndexError`` instead of this loader's own named refusal.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from realmspinner.kernels.geom3d import gltf
from realmspinner.kernels.geom3d.glbio import rebuild_glb


def _glb(doc: dict, binary: bytes = b"") -> bytes:
    binary += b"\x00" * (-len(binary) % 4)
    chunk = struct.pack("<II", len(binary), 0x004E4942) + binary
    header = struct.pack("<III", 0x46546C67, 2, 0)
    return rebuild_glb(header, doc, chunk)


def _skin_doc(n_joints: int) -> dict:
    return {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        # One real node; every joint below repeats its (valid) index -- the
        # loader's own bounds check on individual joint values never catches
        # the size of the list, which is exactly the gap this closes.
        "nodes": [{}],
        "skins": [{"joints": [0] * n_joints}],
    }


def test_a_skin_missing_inverse_bind_matrices_charges_its_tiled_bytes(monkeypatch):
    n_joints = 5_000
    # Below MAX_SKIN_JOINTS, so only the byte charge is on trial here; a
    # ceiling one byte under what 5,000 joints at 128 bytes each costs.
    monkeypatch.setattr(gltf, "MAX_TOTAL_BYTES", n_joints * 128 - 1)
    with pytest.raises(ValueError, match="byte budget"):
        gltf.load(_glb(_skin_doc(n_joints)))


def test_a_skin_missing_inverse_bind_matrices_stays_under_budget_when_small():
    """The charge above must not refuse an ordinary rig -- 20 joints, this
    app's own convention, tiled with no ``inverseBindMatrices`` present."""
    model = gltf.load(_glb(_skin_doc(20)))
    assert len(model.skins[0].inverse_bind) == 20


def test_a_skin_with_too_many_joints_is_refused_before_it_tiles_its_bind_matrices():
    """Refused on the joint count alone, not on the byte budget -- a 6-9 MB
    file naming millions of joints must not have to also blow
    ``MAX_TOTAL_BYTES`` to be caught; the count itself is a hang and a
    memory blow-up before it is a scene, the same reasoning
    ``MAX_NODES``/``MAX_PRIMITIVES`` and their siblings already carry."""
    with pytest.raises(ValueError, match="100001 joints"):
        gltf.load(_glb(_skin_doc(gltf.MAX_SKIN_JOINTS + 1)))


def _interleaved_position_doc(byte_stride: int, binary: bytes) -> dict:
    return {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": 0,
                "byteLength": len(binary),
                "byteStride": byte_stride,
            }
        ],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}],
    }


def test_a_negative_bytestride_is_refused_with_a_value_error():
    """clay-io-06: before this refusal, a negative ``byteStride`` made the
    interleaved gather's fancy index wrap backwards through the buffer
    instead of raising -- reproduced: three VEC3 rows read back reordered
    (``[[0,0,0],[0,1,0],[1,0,0]]`` -> ``[[0,0,0],[1,0,0],[0,1,0]]``) rather
    than refused."""
    positions = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]], dtype="<f4")
    with pytest.raises(ValueError, match="byteStride"):
        gltf.load(_glb(_interleaved_position_doc(-12, positions.tobytes()), positions.tobytes()))


def test_a_bytestride_over_252_is_refused_with_a_value_error():
    """The same ceiling, the other side: glTF bounds ``byteStride`` to
    [4, 252] (accessor.schema.json), and nothing here enforced the upper
    bound either."""
    positions = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]], dtype="<f4")
    with pytest.raises(ValueError, match="byteStride"):
        gltf.load(_glb(_interleaved_position_doc(256, positions.tobytes()), positions.tobytes()))


def test_an_ordinary_interleaved_bytestride_still_loads():
    """The ceiling above must not catch a legitimate interleaved stream."""
    positions = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]], dtype="<f4")
    model = gltf.load(_glb(_interleaved_position_doc(12, positions.tobytes()), positions.tobytes()))
    assert np.allclose(model.meshes[0][0].positions, positions)
