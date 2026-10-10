"""``optimize.lod_chain``: the mesh at 100 / 50 / 25 / 10 percent, in one zip.

The drop-and-order logic is exercised against a fake ``optimize.run`` so it needs
no binary; the claims about real simplification (levels strictly smaller, still
standing on the floor) run against the vendored gltfpack and skip where it is
absent -- ``vendor/`` is gitignored, so CI does.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

from realmspinner.pipelines import optimize

GLTFPACK = Path(__file__).resolve().parents[2] / "vendor" / "gltfpack" / "gltfpack.exe"
needs_gltfpack = pytest.mark.skipif(not GLTFPACK.is_file(), reason="gltfpack is not vendored")


def _textured_sphere(path: Path, *, subdivisions: int = 5) -> None:
    """A grounded sphere with UVs and both PBR maps -- what ``tiercheck`` insists
    survive a simplification pass."""
    sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=0.5)
    x, y, z = sphere.vertices.T
    uv = np.stack([np.arctan2(z, x) / (2 * np.pi) + 0.5, np.arcsin(y / 0.5) / np.pi + 0.5], 1)
    base = Image.new("RGB", (8, 8), (180, 90, 40))
    rough = Image.new("RGB", (8, 8), (0, 128, 0))
    sphere.visual = trimesh.visual.TextureVisuals(
        uv=uv,
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=base, metallicRoughnessTexture=rough
        ),
    )
    sphere.apply_translation([0.0, 0.5, 0.0])
    trimesh.Scene(sphere).export(path)


def _names(zip_path: Path) -> list[str]:
    with zipfile.ZipFile(zip_path) as archive:
        return sorted(archive.namelist())


# ---- the ordering and dropping, with no binary ---------------------------------


class _FakeRun:
    """Stands in for ``optimize.run``: writes a stub file and reports a count."""

    def __init__(self, source_triangles: int, fail_at: set[float] | None = None, floor=None):
        self.sources: list[Path] = []
        self.budgets: list[int] = []
        self.fail_at = fail_at or set()
        self.floor = floor

    def __call__(self, source, dest, *, target_triangles, exe, timeout=0.0):
        self.sources.append(source)
        self.budgets.append(target_triangles)
        ratio = target_triangles / self.total
        if any(abs(ratio - bad) < 0.02 for bad in self.fail_at):
            raise optimize.OptimizeError("the budget lost what it must keep: UVs")
        achieved = self.floor if self.floor is not None else target_triangles
        dest.write_bytes(b"glb" * 4)
        return {"achieved": achieved}


@pytest.fixture
def fake(tmp_path, monkeypatch):
    model = tmp_path / "model.glb"
    model.write_bytes(b"model-bytes")
    monkeypatch.setattr(optimize, "_triangles", lambda path: 10_000)

    def make(**kwargs):
        runner = _FakeRun(10_000, **kwargs)
        runner.total = 10_000
        monkeypatch.setattr(optimize, "run", runner)
        return runner

    return model, make


def test_model_lods_zip_holds_decreasing_triangle_levels(tmp_path, fake):
    model, make = fake
    runner = make()
    out = tmp_path / "chain.zip"
    manifest = optimize.lod_chain(model, out, exe=GLTFPACK)
    assert _names(out) == [
        "lods.json", "model_LOD0.glb", "model_LOD1.glb", "model_LOD2.glb", "model_LOD3.glb",
    ]
    counts = [level["triangles"] for level in manifest["levels"]]
    assert counts == [10_000, 5_000, 2_500, 1_000]
    assert counts == sorted(counts, reverse=True) and len(set(counts)) == 4
    assert manifest["dropped"] == []
    assert json.loads(zipfile.ZipFile(out).read("lods.json")) == manifest
    # Every level is cut from the mesh itself, never from the one above it:
    # simplifying a simplification compounds the error.
    assert set(runner.sources) == {model}
    assert zipfile.ZipFile(out).read("model_LOD0.glb") == b"model-bytes"


def test_a_failing_level_and_every_coarser_one_are_dropped_with_the_reason(tmp_path, fake):
    model, make = fake
    make(fail_at={0.25})
    out = tmp_path / "chain.zip"
    manifest = optimize.lod_chain(model, out, exe=GLTFPACK)
    assert [level["level"] for level in manifest["levels"]] == [0, 1]
    assert [d["level"] for d in manifest["dropped"]] == [2, 3]
    assert all("lost what it must keep" in d["reason"] for d in manifest["dropped"])
    assert _names(out) == ["lods.json", "model_LOD0.glb", "model_LOD1.glb"]


def test_a_level_that_does_not_get_smaller_is_not_a_level(tmp_path, fake):
    """``optimize.run`` copies a mesh already under the budget, which would put
    two identical files in the chain."""
    model, make = fake
    make(floor=5_000)
    manifest = optimize.lod_chain(model, tmp_path / "chain.zip", exe=GLTFPACK)
    assert [level["level"] for level in manifest["levels"]] == [0, 1]
    assert "did not get smaller than LOD1" in manifest["dropped"][0]["reason"]


def test_fewer_than_two_levels_is_refused_and_leaves_nothing_behind(tmp_path, fake):
    model, make = fake
    make(fail_at={0.5, 0.25, 0.1})
    out = tmp_path / "chain.zip"
    with pytest.raises(ValueError, match="UVs"):
        optimize.lod_chain(model, out, exe=GLTFPACK)
    assert not out.exists()
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".lods")] == []


def test_the_scratch_directory_is_removed_on_success_too(tmp_path, fake):
    model, make = fake
    make()
    optimize.lod_chain(model, tmp_path / "chain.zip", exe=GLTFPACK)
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".lods")] == []


@pytest.mark.parametrize("ratios", [(), (0.5, 1.0), (1.0, 0.25, 0.5), (0.5,)])
def test_ratios_must_start_at_one_and_decrease(tmp_path, fake, ratios):
    model, make = fake
    make()
    with pytest.raises(ValueError, match="start at 1.0"):
        optimize.lod_chain(model, tmp_path / "chain.zip", exe=GLTFPACK, ratios=ratios)


# ---- real gltfpack ---------------------------------------------------------------


@needs_gltfpack
def test_a_real_chain_gets_smaller_at_every_level_and_keeps_the_pbr_maps(tmp_path):
    model = tmp_path / "model.glb"
    _textured_sphere(model)
    out = tmp_path / "chain.zip"
    manifest = optimize.lod_chain(model, out, exe=GLTFPACK)
    counts = [level["triangles"] for level in manifest["levels"]]
    assert len(counts) == 4 and manifest["dropped"] == []
    assert counts == sorted(counts, reverse=True) and len(set(counts)) == 4
    source = manifest["source_triangles"]
    for level in manifest["levels"][1:]:
        assert level["triangles"] <= source * level["ratio"] * 1.35
    with zipfile.ZipFile(out) as archive:
        archive.extractall(tmp_path / "levels")
    for level in manifest["levels"]:
        scene = trimesh.load(tmp_path / "levels" / level["file"])
        material = next(iter(scene.geometry.values())).visual.material
        assert material.baseColorTexture is not None
        assert material.metallicRoughnessTexture is not None


@needs_gltfpack
def test_lod_levels_stay_grounded(tmp_path):
    """gltfpack bakes node transforms into the geometry, so a grounded source
    stays on the floor -- but simplification can lift or shift an extremity by
    a hair, and an engine swaps LODs in place. Every level must stand on the
    floor and share LOD0's centre to within a percent of its height."""
    model = tmp_path / "model.glb"
    _textured_sphere(model)
    out = tmp_path / "chain.zip"
    manifest = optimize.lod_chain(model, out, exe=GLTFPACK)
    with zipfile.ZipFile(out) as archive:
        archive.extractall(tmp_path / "levels")
    bounds = {
        level["level"]: trimesh.load(tmp_path / "levels" / level["file"]).bounds
        for level in manifest["levels"]
    }
    height = float(bounds[0][1][1] - bounds[0][0][1])
    for index, box in bounds.items():
        assert abs(float(box[0][1])) < 0.01 * height, f"LOD{index} is off the floor"
        centre = (box[0] + box[1]) / 2.0
        reference = (bounds[0][0] + bounds[0][1]) / 2.0
        assert np.abs(centre[[0, 2]] - reference[[0, 2]]).max() < 0.01 * height
