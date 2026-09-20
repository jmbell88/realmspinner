"""``optimize.simplify_bytes`` -- gltfpack against in-memory bytes, for Clay.

Clay's decimate operator has no ``source.glb``/``dest`` pair on disk to hand
``optimize.run``; it serialises the object being edited to GLB bytes and wants
the simplified bytes straight back. This pins the same error discipline as
``run`` (missing exe, timeout, non-zero exit, unusable output) plus the two
knobs unique to this path: ``-slb`` (lock border vertices) and ``-sa``
(aggressive/topology-changing simplification), both of which the vendored
gltfpack 1.2 supports per its own ``-h`` output.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from warlock import winjob
from warlock.pipelines import optimize

VENDORED_EXE = Path("vendor/gltfpack/gltfpack.exe")
needs_real_gltfpack = pytest.mark.skipif(
    not VENDORED_EXE.is_file(), reason="needs vendor/gltfpack/gltfpack.exe (not this checkout)"
)


def test_ratio_of_one_returns_the_same_bytes_and_spawns_nothing(tmp_path, monkeypatch):
    def explode(*a, **k):
        raise AssertionError("gltfpack must not run for ratio 1.0")

    monkeypatch.setattr(winjob, "run", explode)
    data = b"some glb bytes"
    out = optimize.simplify_bytes(data, ratio=1.0, exe=tmp_path / "missing.exe")
    assert out is data


@pytest.mark.parametrize("ratio", [0.0, -0.1, 1.1, 2.0])
def test_out_of_range_ratio_raises_value_error(ratio, tmp_path):
    with pytest.raises(ValueError):
        optimize.simplify_bytes(b"glb", ratio=ratio, exe=tmp_path / "gltfpack.exe")


def test_missing_exe_raises_naming_the_path(tmp_path):
    exe = tmp_path / "nope.exe"
    with pytest.raises(optimize.OptimizeError, match=str(exe).replace("\\", "\\\\")):
        optimize.simplify_bytes(b"glb", ratio=0.5, exe=exe)


def _fake_run_writing(payload: bytes):
    def fake_run(argv, **kwargs):
        Path(argv[argv.index("-o") + 1]).write_bytes(payload)
        return subprocess.CompletedProcess(argv, 0, "", "")

    return fake_run


def test_argv_carries_slb_only_when_lock_border(tmp_path, monkeypatch):
    seen = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        Path(argv[argv.index("-o") + 1]).write_bytes(b"optimised")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(winjob, "run", fake_run)
    exe = tmp_path / "gltfpack.exe"
    exe.write_bytes(b"")

    optimize.simplify_bytes(b"glb", ratio=0.5, exe=exe, lock_border=True)
    assert "-slb" in seen["argv"]
    assert "-sa" not in seen["argv"]

    optimize.simplify_bytes(b"glb", ratio=0.5, exe=exe, lock_border=False)
    assert "-slb" not in seen["argv"]


def test_argv_carries_sa_only_when_aggressive(tmp_path, monkeypatch):
    seen = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        Path(argv[argv.index("-o") + 1]).write_bytes(b"optimised")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(winjob, "run", fake_run)
    exe = tmp_path / "gltfpack.exe"
    exe.write_bytes(b"")

    optimize.simplify_bytes(b"glb", ratio=0.5, exe=exe, aggressive=True)
    assert "-sa" in seen["argv"]
    assert "-slb" not in seen["argv"]

    optimize.simplify_bytes(b"glb", ratio=0.5, exe=exe, aggressive=False)
    assert "-sa" not in seen["argv"]


def test_argv_documented_flags_and_in_out_paths(tmp_path, monkeypatch):
    seen = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        # Read the staged input now -- simplify_bytes's tempdir is gone by
        # the time this function returns.
        seen["input"] = Path(argv[argv.index("-i") + 1]).read_bytes()
        Path(argv[argv.index("-o") + 1]).write_bytes(b"optimised")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(winjob, "run", fake_run)
    exe = tmp_path / "gltfpack.exe"
    exe.write_bytes(b"")

    out = optimize.simplify_bytes(b"the input bytes", ratio=0.25, exe=exe)
    argv = seen["argv"]
    assert argv[0] == str(exe)
    assert "-noq" in argv and "-ke" in argv and "-km" in argv
    assert argv[argv.index("-si") + 1] == "0.25"
    assert seen["input"] == b"the input bytes"
    assert out == b"optimised"


def test_timeout_raises_optimize_error(tmp_path, monkeypatch):
    def fake_run(argv, **kwargs):
        raise subprocess.TimeoutExpired(cmd=argv, timeout=kwargs.get("timeout"))

    monkeypatch.setattr(winjob, "run", fake_run)
    exe = tmp_path / "gltfpack.exe"
    exe.write_bytes(b"")
    with pytest.raises(optimize.OptimizeError, match="timed out"):
        optimize.simplify_bytes(b"glb", ratio=0.5, exe=exe, timeout=1.0)


def test_non_zero_exit_raises_with_stderr(tmp_path, monkeypatch):
    monkeypatch.setattr(
        winjob, "run", lambda argv, **k: subprocess.CompletedProcess(argv, 1, "", "boom")
    )
    exe = tmp_path / "gltfpack.exe"
    exe.write_bytes(b"")
    with pytest.raises(optimize.OptimizeError, match="boom"):
        optimize.simplify_bytes(b"glb", ratio=0.5, exe=exe)


def test_missing_output_raises(tmp_path, monkeypatch):
    # gltfpack "succeeds" (exit 0) but never writes -o -- the caller must not
    # try to read a file that was never created.
    monkeypatch.setattr(
        winjob, "run", lambda argv, **k: subprocess.CompletedProcess(argv, 0, "", "")
    )
    exe = tmp_path / "gltfpack.exe"
    exe.write_bytes(b"")
    with pytest.raises(optimize.OptimizeError, match="no output"):
        optimize.simplify_bytes(b"glb", ratio=0.5, exe=exe)


def test_empty_output_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(winjob, "run", _fake_run_writing(b""))
    exe = tmp_path / "gltfpack.exe"
    exe.write_bytes(b"")
    with pytest.raises(optimize.OptimizeError, match="empty"):
        optimize.simplify_bytes(b"glb", ratio=0.5, exe=exe)


def test_temp_dir_is_gone_after_success(tmp_path, monkeypatch):
    captured = {}

    def fake_run(argv, **kwargs):
        in_path = Path(argv[argv.index("-i") + 1])
        captured["tmpdir"] = in_path.parent
        Path(argv[argv.index("-o") + 1]).write_bytes(b"optimised")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(winjob, "run", fake_run)
    exe = tmp_path / "gltfpack.exe"
    exe.write_bytes(b"")
    optimize.simplify_bytes(b"glb", ratio=0.5, exe=exe)
    assert not captured["tmpdir"].exists()


def test_temp_dir_is_gone_after_failure(tmp_path, monkeypatch):
    captured = {}

    def fake_run(argv, **kwargs):
        in_path = Path(argv[argv.index("-i") + 1])
        captured["tmpdir"] = in_path.parent
        return subprocess.CompletedProcess(argv, 1, "", "boom")

    monkeypatch.setattr(winjob, "run", fake_run)
    exe = tmp_path / "gltfpack.exe"
    exe.write_bytes(b"")
    with pytest.raises(optimize.OptimizeError):
        optimize.simplify_bytes(b"glb", ratio=0.5, exe=exe)
    assert not captured["tmpdir"].exists()


@needs_real_gltfpack
def test_a_real_uv_sphere_simplifies_and_reparses(tmp_path, monkeypatch):
    """End to end against the vendored binary: a Clay uv_sphere written to GLB
    bytes, simplified at ratio 0.25, and read back as a Clay document with
    fewer faces than it started with.

    ``aggressive=True`` (``-sa``): measured by hand against this exact sphere
    (32 segments, 16 rings) with the vendored gltfpack 1.2, the plain
    simplifier (no ``-sa``) leaves every triangle in place at ratio 0.25 on
    this mesh -- it will not cross a UV-seam/attribute discontinuity to
    reach the target ratio, only ``-sa``'s topology-changing mode will. A
    regression here that quietly drops the flag from the argv would make
    this test's own face-count assertion fail, which is the point.
    """
    monkeypatch.setenv("WARLOCK_HOME", str(tmp_path))

    from warlock.kernels.geom3d import glbwrite
    from warlock.kernels.mesh import document as bd
    from warlock.kernels.mesh import glbimport
    from warlock.kernels.mesh import primitives as bp

    defaults, generator = bp.GENERATORS["uv_sphere"]
    params = {**defaults, "segments": 32, "rings": 16}
    mesh = generator(**params)
    doc = bd.ClayDoc()
    doc.objects.append(bd.Obj(uid=bd.new_uid(), name="Sphere", mesh=mesh))
    original_bytes = glbwrite.write_glb(bd.to_model(doc))
    original_doc = glbimport.glb_to_claydoc(original_bytes)
    original_face_count = sum(len(o.mesh.starts) for o in original_doc.objects)

    out_bytes = optimize.simplify_bytes(
        original_bytes, ratio=0.25, exe=VENDORED_EXE, aggressive=True
    )
    simplified_doc = glbimport.glb_to_claydoc(out_bytes)
    simplified_face_count = sum(len(o.mesh.starts) for o in simplified_doc.objects)

    assert len(simplified_doc.objects) >= 1
    assert simplified_face_count < original_face_count
