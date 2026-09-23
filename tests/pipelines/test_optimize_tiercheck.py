"""``tiercheck.compare`` guards every gltfpack pass inside ``optimize.run``.

dev/measurements/2026-09-23-default-mesh-budget.md retires the old per-tier
corpus qualification (a chest, a sword and a rock, run once by hand and never
finished -- 0 of 20 accepted on 2026-08-13) in favour of a check on every job
instead: before ``tmp`` ever replaces ``dest``, ``optimize.run`` compares the
pass's own output to its own source with ``tiercheck.compare``. A losing
verdict must raise ``OptimizeError`` and leave ``dest`` exactly as it was; a
passing one is recorded on the returned dict under ``"tiercheck"``.

These tests write real, minimal GLBs by hand rather than reaching for a real
mesh -- ``tiercheck.survey`` parses only the glTF JSON chunk
(``glbio.read_glb``) and never touches accessor or buffer data, so a
JSON-only container with no BIN chunk is a legitimate input. ``_triangles``
is monkeypatched the way ``tests/test_optimize.py`` already does throughout,
which keeps this decoupled from trimesh and the vendored binary the same way
that file is.
"""

from __future__ import annotations

import struct
import subprocess
from pathlib import Path

import pytest

from realmspinner import winjob
from realmspinner.kernels.geom3d import glbio
from realmspinner.pipelines import optimize


def _glb_bytes(gltf: dict) -> bytes:
    """A minimal, valid GLB: the 12-byte header plus a JSON chunk, no BIN
    chunk. ``glbio.rebuild_glb`` computes the chunk length and total size;
    this only has to supply a header carrying the right magic and version.
    """
    header = struct.pack("<III", glbio.GLB_MAGIC, 2, 0)
    return glbio.rebuild_glb(header, gltf, b"")


_TEXTURED_MATERIAL = {
    "pbrMetallicRoughness": {
        "baseColorTexture": {"index": 0},
        "metallicRoughnessTexture": {"index": 0},
    },
    "normalTexture": {"index": 0},
}

_SOURCE_GLTF = {
    "materials": [_TEXTURED_MATERIAL],
    "meshes": [
        {"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "material": 0}]}
    ],
    "images": [{"uri": "tex.png"}],
}


def _fake_gltfpack(payload: dict):
    """A ``winjob.run`` stand-in that writes ``payload`` to ``-o`` instead of
    actually invoking the vendored binary -- the same shape
    ``tests/test_optimize.py``'s ``fake_run`` helpers use, except the bytes
    written are a real GLB a surveyor can read rather than an opaque marker.
    """

    def fake_run(argv, **kwargs):
        Path(argv[argv.index("-o") + 1]).write_bytes(_glb_bytes(payload))
        return subprocess.CompletedProcess(argv, 0, "", "")

    return fake_run


def _setup(tmp_path, monkeypatch, payload: dict) -> tuple[Path, Path, Path]:
    monkeypatch.setattr(winjob, "run", _fake_gltfpack(payload))
    # Triangle counting is not what these tests are about (tiercheck.survey
    # reads the JSON chunk, not the accessor data _triangles would need); a
    # source over budget and an achieved count under it are all `run` needs
    # to decide to invoke gltfpack and then accept what it wrote.
    monkeypatch.setattr(
        optimize, "_triangles", lambda p: 100_000 if p.name == "source.glb" else 50_000
    )
    exe = tmp_path / "gltfpack.exe"
    exe.write_bytes(b"")
    src = tmp_path / "source.glb"
    src.write_bytes(_glb_bytes(_SOURCE_GLTF))
    dest = tmp_path / "model.glb"
    return exe, src, dest


def test_a_tier_that_drops_uvs_is_refused_and_dest_is_untouched(tmp_path, monkeypatch):
    lossy = {
        "materials": [_TEXTURED_MATERIAL],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "material": 0}]}],
        "images": [{"uri": "tex.png"}],
    }
    exe, src, dest = _setup(tmp_path, monkeypatch, lossy)
    with pytest.raises(optimize.OptimizeError, match="UVs lost"):
        optimize.run(src, dest, target_triangles=50_000, exe=exe)
    assert not dest.exists()
    # The staging file is unlinked on this path too -- the same discipline
    # `finally` already gives every other refusal in `run`.
    assert list(tmp_path.glob(".*.opt*")) == []


def test_a_tier_that_drops_the_base_colour_texture_is_refused(tmp_path, monkeypatch):
    lossy_material = {
        "pbrMetallicRoughness": {"metallicRoughnessTexture": {"index": 0}},
        "normalTexture": {"index": 0},
    }
    lossy = {
        "materials": [lossy_material],
        "meshes": [
            {
                "primitives": [
                    {"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "material": 0}
                ]
            }
        ],
        "images": [{"uri": "tex.png"}],
    }
    exe, src, dest = _setup(tmp_path, monkeypatch, lossy)
    with pytest.raises(optimize.OptimizeError, match="base colour"):
        optimize.run(src, dest, target_triangles=50_000, exe=exe)
    assert not dest.exists()


def test_a_tier_that_keeps_everything_records_a_passing_tiercheck(tmp_path, monkeypatch):
    exe, src, dest = _setup(tmp_path, monkeypatch, _SOURCE_GLTF)
    result = optimize.run(src, dest, target_triangles=50_000, exe=exe)
    assert dest.exists()
    assert result["tiercheck"] == {"ok": True, "notes": []}
