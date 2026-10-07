"""The 2026-10-03 audit's Low findings pipelines-19 .. 25, 28 .. 35 and 40.

Each test's name is the claim. None needs a GPU, weights or ``bpy`` (the Blender
worker is driven with stand-ins, the way the audit's Medium batch drives it).
The tests span several modules because one fixer owned the whole batch and the
house rule is one new test file per fixer.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import importlib.util
import inspect
import json
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "realmspinner"


# --- pipelines-19: the MASK cutoff comes from MASK materials the mesh uses ---


class _Nodes(list):
    def new(self, kind):
        node = SimpleNamespace(
            type=kind,
            operation=None,
            inputs=[SimpleNamespace(default_value=None) for _ in range(2)],
            outputs=[object()],
        )
        self.append(node)
        return node


class _Material(dict):
    """A Blender material stand-in: item assignment plus the attributes read."""


def _fake_atlas_material():
    alpha = SimpleNamespace(links=[SimpleNamespace(from_socket=object())], default_value=1.0)
    principled = SimpleNamespace(type="BSDF_PRINCIPLED", inputs={"Alpha": alpha})
    tree = SimpleNamespace(nodes=_Nodes([principled]), links=mock.MagicMock())
    material = _Material()
    material.node_tree = tree
    material.surface_render_method = None
    material.use_backface_culling = True
    return material


def _fake_source():
    return SimpleNamespace(
        data=SimpleNamespace(materials=[SimpleNamespace(use_backface_culling=True)])
    )


def test_repair_keeps_the_mask_cutoff_when_another_material_is_opaque():
    from realmspinner.pipelines import blender_worker as bw

    material = _fake_atlas_material()
    bw._preserve_alpha_mode(
        _fake_source(),
        material,
        [{"alphaMode": "MASK", "alphaCutoff": 0.3}, {"alphaMode": "OPAQUE"}, {}],
    )
    assert material["gltf_alpha_mode"] == "MASK"
    assert material["gltf_alpha_cutoff"] == pytest.approx(0.3)


def test_repair_ignores_a_blend_or_mask_material_no_primitive_uses(monkeypatch, tmp_path):
    """An unused MASK/BLEND material changed the mode of a mesh that never used it.

    Drives ``op_remesh`` with a document whose only referenced material is MASK
    0.3 and whose second material (BLEND) is unused; the list handed to
    ``_preserve_alpha_mode`` must be the used one alone.
    """
    from realmspinner.kernels.geom3d import glbio
    from realmspinner.pipelines import blender_worker as bw

    source = tmp_path / "source.glb"
    source.write_bytes(b"glTF")
    gltf = {
        "materials": [{"alphaMode": "MASK", "alphaCutoff": 0.3}, {"alphaMode": "BLEND"}],
        "meshes": [{"primitives": [{"material": 0}]}],
    }
    seen: list[list[dict]] = []
    bakes: list[tuple[str, ...]] = []

    monkeypatch.setattr(glbio, "read_glb", lambda _p: (gltf, b""))
    monkeypatch.setattr(bw, "_reset_scene", lambda _b: None)
    monkeypatch.setattr(bw, "_import_measured", lambda _b, _p: mock.MagicMock())
    monkeypatch.setattr(bw, "_face_stats", lambda _m: (10, 1.0))
    monkeypatch.setattr(bw, "_world_bounds", lambda _m: ([0, 0, 0], [1, 1, 1]))
    monkeypatch.setattr(bw, "weld_distance", lambda _lo, _hi: 0.0)
    monkeypatch.setattr(bw, "_remesh_object", lambda *_a, **_k: "quadriflow")
    monkeypatch.setattr(bw, "_smart_unwrap", lambda *_a, **_k: None)
    def record_bake(*_a, maps, **_k):
        bakes.append(tuple(maps))
        return mock.MagicMock(), {}

    monkeypatch.setattr(bw, "_bake_maps", record_bake)
    monkeypatch.setattr(bw, "_preserve_alpha_mode", lambda _s, _m, mats: seen.append(list(mats)))
    monkeypatch.setattr(bw, "_export", lambda *_a, **_k: None)
    monkeypatch.setattr(bw, "_tri_count", lambda _m: 10)
    monkeypatch.setattr(bw, "progress", lambda *_a, **_k: None)

    bw.op_remesh(
        mock.MagicMock(),
        {
            "source_glb": str(source),
            "out_glb": str(tmp_path / "out.glb"),
            "target_faces": 100,
            "texture_size": 64,
        },
    )
    assert seen == [[{"alphaMode": "MASK", "alphaCutoff": 0.3}]]
    assert "alpha" in bakes[0]


# --- pipelines-21: a transport ValueError is retried and keeps the tree ------


def test_a_transport_value_error_is_retried_and_keeps_the_staging_tree(monkeypatch, tmp_path):
    from realmspinner.pipelines import fetch_worker

    sleeps: list[float] = []
    monkeypatch.setattr(fetch_worker.time, "sleep", sleeps.append)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise json.JSONDecodeError("Expecting value", "", 0)

    fetch_worker._with_retries(flaky, {"retries": 3})
    assert calls["n"] == 2, "a truncated-JSON ValueError from a transport was never retried"

    # And the tree a failed fetch leaves behind is the one a retry resumes into.
    fake_hub = types.ModuleType("huggingface_hub")

    def snapshot_download(**kw):
        (Path(kw["local_dir"]) / "big.bin").write_bytes(b"x" * 64)
        raise json.JSONDecodeError("Expecting value", "", 0)

    fake_hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    dest = tmp_path / "models" / "thing"
    spec = {
        "repo_id": "acme/thing",
        "dest": str(dest),
        "filenames": [],
        "allow_patterns": [],
        "ignore_patterns": [],
        "rename": None,
        "size_gib": 0.1,
        "retries": 1,
    }
    with pytest.raises(json.JSONDecodeError):
        fetch_worker.fetch_one(spec)
    kept = list((tmp_path / "models").glob("*.fetch.part"))
    assert len(kept) == 1 and (kept[0] / "big.bin").is_file(), (
        "a transport ValueError wiped the resumable staging tree"
    )


def test_a_refusal_the_worker_raised_itself_is_still_terminal(monkeypatch):
    from realmspinner.pipelines import download, fetch_worker

    monkeypatch.setattr(fetch_worker.time, "sleep", lambda _s: None)
    calls = {"n": 0}

    def refuses():
        calls["n"] += 1
        raise download.Refusal("digest mismatch")

    with pytest.raises(ValueError, match="digest mismatch"):
        fetch_worker._with_retries(refuses, {"retries": 3})
    assert calls["n"] == 1


# --- pipelines-22: matting.available wants the weights, not just the config --


def test_available_is_false_for_a_directory_with_config_but_no_weights(tmp_path):
    from realmspinner.config import Config
    from realmspinner.pipelines import matting

    cfg = Config(t2i_model_root=tmp_path)
    root = matting.model_dir(cfg)
    root.mkdir(parents=True)
    (root / "config.json").write_text("{}", encoding="utf-8")
    assert matting.available(cfg) is False
    (root / "model.safetensors").write_bytes(b"x")
    assert matting.available(cfg) is True


# --- pipelines-23 / 30 / 28: doctor ------------------------------------------


@pytest.mark.parametrize(
    "manifest",
    [
        {"repos": ["not", "a", "mapping"]},
        {"repos": {"acme/thing": "not an object"}},
        {"repos": {"acme/thing": {"revision": ["unhashable"]}}},
    ],
)
def test_a_malformed_fetch_manifest_does_not_break_the_base_model_rows(
    monkeypatch, tmp_path, manifest
):
    from realmspinner import doctor, fetch
    from realmspinner.config import Config

    monkeypatch.setattr(fetch, "base_model_state", lambda *_a, **_k: (True, None))
    monkeypatch.setattr(fetch, "read_manifest", lambda _dest: manifest)
    checks = doctor._t2i_checks(Config(t2i_model_root=tmp_path))
    assert any(c.name for c in checks)


def test_a_fetch_manifest_revision_pin_still_reaches_the_detail(monkeypatch, tmp_path):
    from realmspinner import doctor, fetch, models
    from realmspinner.config import Config

    monkeypatch.setattr(fetch, "base_model_state", lambda *_a, **_k: (True, None))
    monkeypatch.setattr(
        fetch,
        "read_manifest",
        lambda _dest: {"repos": {"a": {"revision": "abc123"}, "b": "junk"}},
    )
    checks = doctor._t2i_checks(Config(t2i_model_root=tmp_path))
    label = next(iter(models.BASE_MODELS.values())).label
    row = next(c for c in checks if c.name.endswith(label))
    assert "revision abc123" in row.detail


def test_an_adapter_whose_weights_path_is_a_directory_is_reported_as_missing_weights(tmp_path):
    from realmspinner import doctor, models
    from realmspinner.config import Config

    config = Config(t2i_model_root=tmp_path)
    adapter = next(iter(models.IP_ADAPTERS.values()))
    root = tmp_path / adapter.dir_name
    (root / adapter.subfolder / adapter.weight_name).mkdir(parents=True)
    (root / adapter.image_encoder_dir).mkdir(parents=True)
    (root / adapter.image_encoder_dir / "config.json").write_text("{}", encoding="utf-8")
    row = next(c for c in doctor._t2i_checks(config) if f"under {root}" in c.detail)
    assert row.detail.startswith("weights not found"), row.detail


def test_doctor_matting_docstring_names_the_loader_it_probes():
    from realmspinner import doctor

    doc = inspect.getdoc(doctor._matting_checks) or ""
    assert "birefnet.load" in doc and "strict=True" in doc
    assert "runs a real CPU ``from_pretrained``" not in doc


# --- pipelines-24: has_normals reads the file --------------------------------


def _box_glb(tmp_path, *, normals):
    import trimesh

    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    if normals:
        # trimesh writes a NORMAL attribute only for normals it has cached.
        _ = mesh.vertex_normals
    path = tmp_path / "box.glb"
    trimesh.Scene(mesh).export(path)
    return path


def test_has_normals_is_false_for_a_glb_without_a_normal_attribute(tmp_path):
    from realmspinner import meshreport

    with_normals = meshreport.build(_box_glb(tmp_path, normals=True))
    assert with_normals["has_normals"] is True
    bare_dir = tmp_path / "bare"
    bare_dir.mkdir()
    without = meshreport.build(_box_glb(bare_dir, normals=False))
    assert without["has_normals"] is False


# --- pipelines-25: the installer name is a basename --------------------------


@pytest.mark.parametrize("name", ["..\\x.exe", "../x.exe", "C:\\x.exe", "sub/x.exe", ".."])
def test_update_download_refuses_an_installer_name_that_is_a_path(monkeypatch, tmp_path, name):
    from realmspinner.pipelines import download, update_worker
    from realmspinner.service import errors, updates

    def no_network(*_a, **_k):
        raise AssertionError("opened a connection for a path-shaped installer name")

    monkeypatch.setattr(download, "open_url", no_network)
    with pytest.raises(ValueError, match="not a filename"):
        update_worker.fetch(
            {
                "dest_dir": str(tmp_path / "stage"),
                "installer_name": name,
                "installer_url": "https://example.invalid/x.exe",
                "sha256": "0" * 64,
            }
        )
    assert not (tmp_path / "stage").exists() or not list((tmp_path / "stage").iterdir())

    def no_spawn(*_a, **_k):
        raise AssertionError("spawned the update worker for a path-shaped installer name")

    monkeypatch.setattr(updates, "_run_worker", no_spawn)
    with pytest.raises(errors.Invalid):
        updates.download(
            None,  # type: ignore[arg-type] -- refused before the service is touched
            {"installer_url": "https://example.invalid/x.exe", "installer_name": name,
             "sha256": "0" * 64},
        )


# --- pipelines-29: ranking gates agree with fetch.present --------------------


def test_ranking_gates_agree_with_fetch_present_for_a_partial_pickscore_directory(tmp_path):
    from realmspinner import fetch, models
    from realmspinner.bench import metrics
    from realmspinner.config import Config

    cfg = Config(t2i_model_root=tmp_path)
    spec = models.METRIC_MODELS["pickscore"]
    root = tmp_path / spec.dir_name
    root.mkdir(parents=True)
    # The half-download manual 40 warns about: the weights without the config.
    (root / "model.safetensors").write_bytes(b"x")
    assert fetch.present(cfg, "metric", spec) is False
    assert metrics.pickscore_available(cfg) is False
    (root / "config.json").write_text("{}", encoding="utf-8")
    assert fetch.present(cfg, "metric", spec) is True
    assert metrics.pickscore_available(cfg) is True


def test_ranking_gates_refuse_a_zero_byte_dino_weights_file(tmp_path):
    from realmspinner import fetch, models
    from realmspinner.bench import metrics
    from realmspinner.config import Config

    cfg = Config(t2i_model_root=tmp_path)
    spec = models.METRIC_MODELS["dinov2"]
    root = tmp_path / spec.dir_name
    root.mkdir(parents=True)
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"")
    assert fetch.suspect_files(cfg, "metric", spec)
    assert metrics.dino_available(cfg) is False
    (root / "model.safetensors").write_bytes(b"x")
    assert metrics.dino_available(cfg) is True


# --- pipelines-31: the bench docstring names its production importers --------


def _bench_metrics_importers() -> set[str]:
    found: set[str] = set()
    for path in SRC.rglob("*.py"):
        if "bench" in path.relative_to(SRC).parts[:1]:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            hit = False
            if isinstance(node, ast.ImportFrom):
                if (node.module or "").split(".")[-1:] == ["bench"] and any(
                    a.name == "metrics" for a in node.names
                ):
                    hit = True
                if (node.module or "").endswith("bench.metrics"):
                    hit = True
            elif isinstance(node, ast.Import):
                hit = any(a.name.endswith("bench.metrics") for a in node.names)
            if hit:
                found.add(path.stem)
    return found


def test_bench_docstring_lists_every_non_bench_importer():
    from realmspinner import bench

    importers = _bench_metrics_importers()
    # The walk must see the importers the audit named, or it is checking nothing.
    assert {"_q_mesh", "judge", "loras", "queue", "tools_ops"} <= importers
    doc = bench.__doc__ or ""
    missing = sorted(name for name in importers if name not in doc)
    assert not missing, f"bench/__init__.py does not name its importer(s): {missing}"
    assert "or score anything" not in doc, "the docstring still denies that anything scores"


# --- pipelines-32: comments cite tests that exist ----------------------------


def test_native_and_installer_comments_cite_tests_that_exist():
    cited = [
        ROOT / "native" / "bvh.c",
        ROOT / "native" / "rotsprite.c",
        ROOT / "native" / "realmspinnerc.h",
        SRC / "studio" / "viewer" / "picking.py",
    ]
    dead: list[str] = []
    for path in cited:
        for rel in re.findall(r"tests/[\w/]+\.py", path.read_text(encoding="utf-8")):
            if not (ROOT / rel).is_file():
                dead.append(f"{path.name}: {rel}")
    assert not dead, dead
    build = (ROOT / "installer" / "build.ps1").read_text(encoding="utf-8")
    assert "Troupe" not in build, "installer/build.ps1 still lists the removed Troupe mode"
    rebuild = (ROOT / "scripts" / "rebuild.ps1").read_text(encoding="utf-8")
    assert "123392" not in rebuild


# --- pipelines-33: make_packs downloads with a timeout -----------------------


def test_make_packs_downloads_with_a_timeout(monkeypatch, tmp_path):
    scripts = ROOT / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location("make_packs_lows", scripts / "make_packs.py")
        assert spec is not None and spec.loader is not None
        maker = importlib.util.module_from_spec(spec)
        sys.modules["make_packs_lows"] = maker
        spec.loader.exec_module(maker)
    finally:
        sys.path.remove(str(scripts))

    payload = b"wheel bytes"
    seen: dict[str, object] = {}

    class _Response:
        def __init__(self):
            self._data = [payload, b""]

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def read(self, _n=-1):
            return self._data.pop(0) if self._data else b""

    def fake_open_url(url, *, timeout=None):
        seen["timeout"] = timeout
        return _Response()

    monkeypatch.setattr(maker._download, "open_url", fake_open_url)
    source = maker.Source(
        url="https://example.invalid/pkg-1.0-py3-none-any.whl",
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    out = maker.download(source, tmp_path, offline=False)
    assert out.read_bytes() == payload
    assert seen["timeout"] is not None and float(seen["timeout"]) > 0


# --- pipelines-34 / 35: trellis log handling ---------------------------------


def _server(tmp_path, exe=None):
    from realmspinner.pipelines import trellis

    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)
    return trellis, trellis.TrellisServer(
        exe or tmp_path / "trellis-server.exe", models_dir, 59999, log_path=tmp_path / "trellis.log"
    )


def test_an_oversize_trellis_log_is_rolled_not_deleted(tmp_path):
    trellis, client = _server(tmp_path)
    log = client._log_path
    log.write_bytes(b"the crash is explained here\n" * 4)
    original = log.read_bytes()
    with mock.patch.object(trellis, "LOG_MAX_BYTES", 8):
        client._open_log()
    try:
        rolled = log.with_name(log.name + ".1")
        assert rolled.is_file() and rolled.read_bytes() == original, (
            "the oversize log was erased instead of rolled"
        )
        assert log.stat().st_size == 0
    finally:
        client._logfh.close()


def test_a_failed_spawn_closes_the_log_and_raises_runtime_error(monkeypatch, tmp_path):
    exe = tmp_path / "trellis-server.exe"
    exe.write_bytes(b"")  # a zero-byte file passes is_file() and fails in Popen
    trellis, client = _server(tmp_path, exe)
    monkeypatch.setattr(trellis, "_port_in_use", lambda _p: False)

    def boom(*_a, **_k):
        raise OSError(193, "%1 is not a valid Win32 application")

    monkeypatch.setattr(trellis.subprocess, "Popen", boom)
    with pytest.raises(RuntimeError, match="could not be started"):
        asyncio.run(client.ensure_started())
    assert client._logfh is None, "the log handle opened before Popen leaked"


# --- pipelines-40: the ABI constant has one value in both places -------------


def test_native_abi_matches_the_header_define():
    from realmspinner import native

    header = (ROOT / "native" / "realmspinnerc.h").read_text(encoding="utf-8")
    match = re.search(r"^#define\s+REALMSPINNERC_ABI\s+(\d+)\s*$", header, re.MULTILINE)
    assert match is not None, "native/realmspinnerc.h no longer #defines REALMSPINNERC_ABI"
    assert int(match.group(1)) == native.ABI
