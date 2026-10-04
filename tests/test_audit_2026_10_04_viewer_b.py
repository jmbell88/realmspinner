"""Regression tests for the 2026-10-04 audit's viewer findings create-32 and create-49.

No GL context: ``GpuModel`` is driven over a counting fake context (the idiom
``tests/studio/test_audit_2026_10_03_medium_create2.py`` uses), and the shader is
checked at the source level. A real render of the degenerate-UV triangle is
owed to the gpu lane -- the default lane has no context to compile against.
"""

from __future__ import annotations

import re
import struct

import numpy as np

from realmspinner.kernels.geom3d import gltf
from realmspinner.kernels.geom3d.glbio import rebuild_glb
from realmspinner.studio.viewer import programs as programslib
from realmspinner.studio.viewer import scene as scenelib


def _glb(doc: dict, binary: bytes = b"") -> bytes:
    binary += b"\x00" * (-len(binary) % 4)
    chunk = struct.pack("<II", len(binary), 0x004E4942) + binary
    header = struct.pack("<III", 0x46546C67, 2, 0)
    return rebuild_glb(header, doc, chunk)


def _instanced_glb(copies: int, verts: int = 3) -> bytes:
    """One mesh of ``verts`` vertices referenced by ``copies`` nodes."""
    positions = np.random.default_rng(0).random((verts, 3), dtype="f4")
    indices = np.arange(verts - verts % 3, dtype="<u4")
    binary = positions.tobytes() + indices.tobytes()
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": list(range(copies))}],
        "nodes": [{"mesh": 0} for _ in range(copies)],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": positions.nbytes},
            {"buffer": 0, "byteOffset": positions.nbytes, "byteLength": indices.nbytes},
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": verts,
                "type": "VEC3",
                "min": positions.min(0).tolist(),
                "max": positions.max(0).tolist(),
            },
            {"bufferView": 1, "componentType": 5125, "count": len(indices), "type": "SCALAR"},
        ],
    }
    return _glb(doc, binary)


class _Buf:
    def __init__(self, log: list, nbytes: int) -> None:
        self.nbytes = nbytes
        self.releases = 0
        log.append(self)

    def release(self) -> None:
        self.releases += 1


class _Ctx:
    def __init__(self) -> None:
        self.made: list[_Buf] = []

    def buffer(self, data):
        return _Buf(self.made, len(data))

    def vertex_array(self, *a, **k):  # pragma: no cover - never drawn here
        raise AssertionError("no draw in this test")


# --- create-32 ---------------------------------------------------------------


def test_a_glb_instancing_one_mesh_a_hundred_thousand_times_is_refused_or_uploads_the_mesh_once():
    """The 2026-10-04 audit found that the loader charges a shared mesh once
    against ``MAX_TOTAL_BYTES`` while ``GpuModel._build`` uploaded a vbo and an
    ibo per *node*, so a small GLB naming one big mesh from ``MAX_NODES`` nodes
    asked the frame thread for terabytes of GL buffer. Uploading once per
    distinct primitive keeps the GPU cost equal to the cost the loader charged."""
    model = gltf.load(_instanced_glb(gltf.MAX_NODES, verts=300))
    ctx = _Ctx()

    gpu = scenelib.GpuModel(ctx, model)

    # One vbo + one ibo for the one primitive, however many nodes draw it.
    assert len(ctx.made) == 2, f"{len(ctx.made)} buffers for a single shared primitive"
    # Every node still draws: sharing the upload must not drop an instance.
    assert len(gpu.draws) == gltf.MAX_NODES

    gpu.release()
    assert all(buf.releases == 1 for buf in ctx.made), (
        "a shared GL buffer must be released exactly once"
    )


def test_distinct_meshes_still_get_their_own_upload():
    """The cache is keyed on the primitive, not on the model: two different
    meshes must not be folded into one buffer."""
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="<f4")
    indices = np.array([0, 1, 2], dtype="<u4")
    binary = positions.tobytes() + indices.tobytes()
    prim = {"attributes": {"POSITION": 0}, "indices": 1}
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0, 1]}],
        "nodes": [{"mesh": 0}, {"mesh": 1}],
        "meshes": [{"primitives": [prim]}, {"primitives": [prim]}],
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
    ctx = _Ctx()
    gpu = scenelib.GpuModel(ctx, gltf.load(_glb(doc, binary)))
    assert len(ctx.made) == 4
    gpu.release()
    assert all(buf.releases == 1 for buf in ctx.made)


# --- create-49 ---------------------------------------------------------------


def _perturb_normal_source() -> str:
    src = programslib.PBR_FRAG
    start = src.index("vec3 perturbNormal(")
    end = src.index("\n}\n", start)
    return src[start:end]


def test_a_normal_mapped_triangle_with_degenerate_uvs_renders_finite_colour():
    """The 2026-10-04 audit found that ``perturbNormal`` took ``inversesqrt`` of a
    value that is exactly zero when a triangle's UVs are collapsed (the t and b
    vectors both vanish), which is Inf, and ``Inf * 0`` in the matrix product is
    NaN -- a black or flickering triangle on an imported normal-mapped GLB.
    Three r170 divides only when the determinant is non-zero; mirror that, and
    fold in the face direction the way three does.

    Source-level: the default lane has no GL context. The render check is owed
    to the gpu lane."""
    body = _perturb_normal_source()

    # inversesqrt is only reached through a zero test on the determinant.
    assert "det == 0.0" in body, "no zero-determinant guard in perturbNormal"
    assert body.index("det == 0.0") < body.index("inversesqrt("), (
        "the guard has to sit before the inversesqrt it protects"
    )
    assert not re.search(r"inversesqrt\(\s*max\(", body), (
        "inversesqrt is still fed the unguarded max(dot(t, t), dot(b, b))"
    )

    # The face direction is a factor of the tangent frame, as in three.
    assert "gl_FrontFacing" in body, "perturbNormal ignores which side of the face is shaded"

    # The all-zero sum (z == 0 texel on a degenerate triangle) has a fallback too.
    assert "return n" in body or ": n;" in body


def test_perturb_normal_is_still_the_only_normal_map_path():
    """The guard must live in the shared PBR fragment source, so every define set
    that compiles ``HAS_NORMAL_MAP`` gets it."""
    expanded = programslib._expand(programslib.PBR_FRAG, ("HAS_NORMAL_MAP",))
    assert expanded.count("vec3 perturbNormal(") == 1
    assert "perturbNormal(normal, viewDir)" in expanded
