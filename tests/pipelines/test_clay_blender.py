"""Clay's three mesh-cleanup Blender ops (retopologise, unwrap, bake high to
low) -- ``dev/CLAY-PLAN.md`` tranche 4, pipeline half.

Three layers, cheapest first:

1. ``blender_spec.clay_*_spec`` -- pure dict construction, no bpy.
2. ``pipelines.clay_blender``'s host functions with ``blender_run.run_worker``
   monkeypatched -- proves the temp-file lifecycle and the
   ``BlenderError`` -> ``ClayBlenderError`` mapping without spending a
   Blender subprocess on it.
3. A real Blender run per op, through the actual ``retopo_bytes``/
   ``unwrap_bytes``/``bake_bytes`` -- the same "nothing had ever executed
   this" argument ``test_remesh_worker.py`` makes for ``op_remesh``: a stub
   worker cannot catch a missing operator or a bake that reads background.
   Skips without ``bpy``, exactly as ``test_remesh_worker.py`` does.

Run with: uv run pytest tests/pipelines/test_clay_blender.py -n 0
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from realmspinner.kernels.geom3d import glbwrite, gltf
from realmspinner.kernels.rig import blender_spec
from realmspinner.pipelines import blender_run, clay_blender

pytestmark = pytest.mark.timeout(600)

TEXTURE_PX = 256


# --- spec construction (no bpy) ---------------------------------------------


class TestClayRetopoSpec:
    def test_defaults(self, tmp_path: Path) -> None:
        spec = blender_spec.clay_retopo_spec(
            tmp_path / "src.glb", tmp_path / "out.glb", tmp_path, target_faces=1000
        )
        assert spec == {
            "op": "clay_retopo",
            "source_glb": str(tmp_path / "src.glb"),
            "out_glb": str(tmp_path / "out.glb"),
            "result_path": str(tmp_path / ".clay_retopo_result.json"),
            "target_faces": 1000,
            "close_holes": False,
            "seed": 0,
            "keep_uvs": False,
        }

    def test_every_optional_is_carried_through(self, tmp_path: Path) -> None:
        spec = blender_spec.clay_retopo_spec(
            tmp_path / "src.glb",
            tmp_path / "out.glb",
            tmp_path,
            target_faces=5000,
            close_holes=True,
            seed=7,
            keep_uvs=True,
        )
        assert spec["close_holes"] is True
        assert spec["seed"] == 7
        assert spec["keep_uvs"] is True

    def test_target_faces_below_the_floor_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="target_faces"):
            blender_spec.clay_retopo_spec(
                tmp_path / "a", tmp_path / "b", tmp_path, target_faces=1
            )

    def test_target_faces_above_the_ceiling_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="target_faces"):
            blender_spec.clay_retopo_spec(
                tmp_path / "a", tmp_path / "b", tmp_path, target_faces=999_999
            )


class TestClayUnwrapSpec:
    def test_defaults(self, tmp_path: Path) -> None:
        spec = blender_spec.clay_unwrap_spec(tmp_path / "src.glb", tmp_path / "out.glb", tmp_path)
        assert spec == {
            "op": "clay_unwrap",
            "source_glb": str(tmp_path / "src.glb"),
            "out_glb": str(tmp_path / "out.glb"),
            "result_path": str(tmp_path / ".clay_unwrap_result.json"),
            "angle_limit": 66.0,
            "island_margin": 0.003,
        }

    def test_overrides_are_carried_through(self, tmp_path: Path) -> None:
        spec = blender_spec.clay_unwrap_spec(
            tmp_path / "src.glb",
            tmp_path / "out.glb",
            tmp_path,
            angle_limit=45.0,
            island_margin=0.01,
        )
        assert spec["angle_limit"] == 45.0
        assert spec["island_margin"] == 0.01


class TestClayBakeSpec:
    def test_defaults_to_all_three_maps(self, tmp_path: Path) -> None:
        spec = blender_spec.clay_bake_spec(
            tmp_path / "high.glb",
            tmp_path / "low.glb",
            tmp_path / "out.glb",
            tmp_path,
            texture_size=1024,
            cage_extrusion=0.02,
        )
        assert spec == {
            "op": "clay_bake",
            "high_glb": str(tmp_path / "high.glb"),
            "low_glb": str(tmp_path / "low.glb"),
            "out_glb": str(tmp_path / "out.glb"),
            "result_path": str(tmp_path / ".clay_bake_result.json"),
            "texture_size": 1024,
            "cage_extrusion": 0.02,
            "maps": ["base_color", "roughness", "normal"],
        }

    def test_a_map_subset_is_carried_through(self, tmp_path: Path) -> None:
        spec = blender_spec.clay_bake_spec(
            tmp_path / "high.glb",
            tmp_path / "low.glb",
            tmp_path / "out.glb",
            tmp_path,
            texture_size=512,
            cage_extrusion=0.01,
            maps=["normal"],
        )
        assert spec["maps"] == ["normal"]

    def test_an_unlisted_texture_size_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="texture_size"):
            blender_spec.clay_bake_spec(
                tmp_path / "h", tmp_path / "l", tmp_path / "o", tmp_path,
                texture_size=333, cage_extrusion=0.02,
            )

    def test_an_unknown_map_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="unknown bake map"):
            blender_spec.clay_bake_spec(
                tmp_path / "h", tmp_path / "l", tmp_path / "o", tmp_path,
                texture_size=512, cage_extrusion=0.02, maps=["specular"],
            )

    def test_an_empty_map_list_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="at least one"):
            blender_spec.clay_bake_spec(
                tmp_path / "h", tmp_path / "l", tmp_path / "o", tmp_path,
                texture_size=512, cage_extrusion=0.02, maps=[],
            )


# --- op registration ----------------------------------------------------------


def test_the_three_clay_ops_are_registered():
    from realmspinner.pipelines import blender_worker as bw

    assert bw.OPS["clay_retopo"] is bw.op_clay_retopo
    assert bw.OPS["clay_unwrap"] is bw.op_clay_unwrap
    assert bw.OPS["clay_bake"] is bw.op_clay_bake


# --- host functions, run_worker monkeypatched (no bpy needed) ---------------


def test_retopo_bytes_round_trips_through_a_faked_worker(monkeypatch, tmp_path):
    captured = {}

    def fake_run_worker(spec, *, timeout=None, name=None, **_kwargs):
        captured["spec"] = dict(spec)
        assert spec["op"] == "clay_retopo"
        source = Path(spec["source_glb"])
        assert source.read_bytes() == b"IN-BYTES"
        Path(spec["out_glb"]).write_bytes(b"OUT-BYTES")
        return {"ok": True, "objects": [{"name": "Hero", "method": "quadriflow"}]}

    monkeypatch.setattr(clay_blender.blender_run, "run_worker", fake_run_worker)
    out_bytes, result = clay_blender.retopo_bytes(b"IN-BYTES", target_faces=1000)
    assert out_bytes == b"OUT-BYTES"
    assert result["objects"][0]["name"] == "Hero"
    # The temp dir the spec's paths lived under is gone once the call returns.
    assert not Path(captured["spec"]["source_glb"]).parent.exists()


def test_unwrap_bytes_round_trips_through_a_faked_worker(monkeypatch):
    def fake_run_worker(spec, *, timeout=None, name=None, **_kwargs):
        assert spec["op"] == "clay_unwrap"
        Path(spec["out_glb"]).write_bytes(b"UNWRAPPED")
        return {"ok": True, "objects": []}

    monkeypatch.setattr(clay_blender.blender_run, "run_worker", fake_run_worker)
    out_bytes, result = clay_blender.unwrap_bytes(b"IN-BYTES")
    assert out_bytes == b"UNWRAPPED"
    assert result == {"ok": True, "objects": []}


def test_bake_bytes_writes_both_inputs_and_round_trips(monkeypatch, tmp_path):
    seen = {}

    def fake_run_worker(spec, *, timeout=None, name=None, **_kwargs):
        assert spec["op"] == "clay_bake"
        seen["high"] = Path(spec["high_glb"]).read_bytes()
        seen["low"] = Path(spec["low_glb"]).read_bytes()
        seen["dir"] = Path(spec["out_glb"]).parent
        Path(spec["out_glb"]).write_bytes(b"BAKED")
        return {"ok": True, "maps": ["base_color"], "texture_size": 512, "metallic": 0.0}

    monkeypatch.setattr(clay_blender.blender_run, "run_worker", fake_run_worker)
    out_bytes, result = clay_blender.bake_bytes(
        b"HIGH-BYTES", b"LOW-BYTES", texture_size=512, cage_extrusion=0.02
    )
    assert seen["high"] == b"HIGH-BYTES"
    assert seen["low"] == b"LOW-BYTES"
    assert out_bytes == b"BAKED"
    assert result["texture_size"] == 512
    assert not seen["dir"].exists()


@pytest.mark.parametrize(
    "call",
    [
        lambda: clay_blender.retopo_bytes(b"x", target_faces=1000),
        lambda: clay_blender.unwrap_bytes(b"x"),
        lambda: clay_blender.bake_bytes(b"x", b"y", texture_size=512, cage_extrusion=0.02),
    ],
    ids=["retopo", "unwrap", "bake"],
)
def test_a_worker_failure_is_raised_as_clay_blender_error(monkeypatch, call):
    def fake_run_worker(spec, *, timeout=None, name=None, **_kwargs):
        raise blender_run.BlenderError("Blender worker exited with code 3: bpy is not installed")

    monkeypatch.setattr(clay_blender.blender_run, "run_worker", fake_run_worker)
    with pytest.raises(clay_blender.ClayBlenderError, match="bpy is not installed"):
        call()


def test_a_worker_failure_cleans_up_its_temp_dir(monkeypatch):
    captured = {}

    def fake_run_worker(spec, *, timeout=None, name=None, **_kwargs):
        captured["dir"] = Path(spec["source_glb"]).parent
        raise blender_run.BlenderError("boom")

    monkeypatch.setattr(clay_blender.blender_run, "run_worker", fake_run_worker)
    with pytest.raises(clay_blender.ClayBlenderError):
        clay_blender.retopo_bytes(b"x", target_faces=1000)
    assert not captured["dir"].exists()


def test_available_reflects_a_working_blender(monkeypatch):
    from realmspinner import doctor

    monkeypatch.setattr(
        doctor, "blender_check",
        lambda probe=False: doctor.Check("Blender (rigging)", True, "bpy 5.2", fatal=False),
    )
    ok, reason = clay_blender.available()
    assert ok is True
    assert reason == ""


def test_available_names_the_rig_extra_when_blender_is_missing(monkeypatch):
    from realmspinner import doctor

    monkeypatch.setattr(
        doctor, "blender_check",
        lambda probe=False: doctor.Check("Blender (rigging)", False, "no bpy", fatal=False),
    )
    ok, reason = clay_blender.available()
    assert ok is False
    assert "rig extra" in reason


# --- real Blender, end to end -------------------------------------------------


def _box(size: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    sx, sy, sz = (s / 2 for s in size)
    positions = np.array(
        [
            [-sx, -sy, -sz], [sx, -sy, -sz], [sx, sy, -sz], [-sx, sy, -sz],
            [-sx, -sy, sz], [sx, -sy, sz], [sx, sy, sz], [-sx, sy, sz],
        ],
        dtype="<f4",
    )
    indices = np.array(
        [
            0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 4, 5, 0, 5, 1,
            1, 5, 6, 1, 6, 2, 2, 6, 7, 2, 7, 3, 3, 7, 4, 3, 4, 0,
        ],
        dtype="<u4",
    )
    return positions, indices


@pytest.fixture(scope="module")
def two_object_glb_bytes() -> bytes:
    """Two boxes, one node each -- ``Hero`` carries two materials split 6/6
    across its 12 triangles, ``Prop`` carries one. Distinctive translations
    and a non-uniform scale on ``Prop``, so a round trip that silently
    dropped or reset a transform has something to be caught by.

    Built with ``kernels.geom3d`` alone (no bpy) so this fixture never skips;
    only the tests that actually dispatch a Blender op do, via
    ``pytest.importorskip`` below.
    """
    pos_hero, indices = _box((1.0, 1.0, 1.0))
    tris = indices.reshape(-1, 3)
    red = gltf.Material(name="Red", base_color_factor=(1.0, 0.0, 0.0, 1.0), metallic_factor=0.0)
    green = gltf.Material(name="Green", base_color_factor=(0.0, 1.0, 0.0, 1.0), metallic_factor=0.0)
    prims_hero = [
        gltf.Primitive(positions=pos_hero, indices=tris[:6].reshape(-1), material=red),
        gltf.Primitive(positions=pos_hero, indices=tris[6:].reshape(-1), material=green),
    ]

    pos_prop, idx_prop = _box((0.5, 0.5, 0.5))
    blue = gltf.Material(name="Blue", base_color_factor=(0.0, 0.0, 1.0, 1.0), metallic_factor=0.25)
    prim_prop = gltf.Primitive(positions=pos_prop, indices=idx_prop, material=blue)

    node_hero = gltf.Node(
        name="Hero", translation=np.array([1.0, 2.0, 0.5]), mesh=0
    )
    node_prop = gltf.Node(
        name="Prop", translation=np.array([-1.0, 0.0, 1.5]), scale=np.array([2.0, 1.0, 1.0]), mesh=1
    )

    model = gltf.Model(
        nodes=[node_hero, node_prop],
        roots=[0, 1],
        meshes=[prims_hero, [prim_prop]],
        skins=[],
    )
    return glbwrite.write_glb(model)


@pytest.fixture(scope="module")
def require_bpy() -> None:
    """Skip every real-Blender test in this module together, the way
    ``test_remesh_worker.py``'s fixtures do -- a machine with no bpy proves
    nothing about these ops and must not report them as failing."""
    pytest.importorskip("bpy")


@pytest.fixture(scope="module")
def retopoed(require_bpy, two_object_glb_bytes) -> tuple[bytes, dict]:
    return clay_blender.retopo_bytes(two_object_glb_bytes, target_faces=200, seed=0)


def test_retopo_keeps_object_count_names_and_transforms(retopoed, two_object_glb_bytes):
    out_bytes, result = retopoed
    assert result["ok"] is True
    assert {o["name"] for o in result["objects"]} == {"Hero", "Prop"}

    before = gltf.load(two_object_glb_bytes)
    after = gltf.load(out_bytes)
    assert [n.name for n in after.nodes] == [n.name for n in before.nodes]
    for a, b in zip(after.nodes, before.nodes, strict=True):
        assert a.translation == pytest.approx(b.translation, abs=1e-5)
        assert a.rotation == pytest.approx(b.rotation, abs=1e-5)
        assert a.scale == pytest.approx(b.scale, abs=1e-5)


def test_retopo_reduces_faces_and_reports_quads(retopoed):
    _out_bytes, result = retopoed
    for obj in result["objects"]:
        assert obj["faces"] > 0
        assert obj["faces_before"] == 12  # each source box is 12 triangles
        assert obj["method"] == "quadriflow", obj
        assert obj["quads"] > 0.9


def test_retopo_preserves_each_objects_material_palette_by_nearest_face(retopoed):
    """The regression this whole helper exists for: quadriflow resets every
    face's material_index to 0, and without ``_transfer_materials_by_
    nearest_face`` the two-material ``Hero`` object would come back with
    every face repainted Red."""
    out_bytes, _result = retopoed
    after = gltf.load(out_bytes)
    materials_by_name = {}
    for prims in after.meshes:
        for p in prims:
            materials_by_name.setdefault(p.material.name, 0)
            materials_by_name[p.material.name] += len(p.indices) // 3

    assert set(materials_by_name) == {"Red", "Green", "Blue"}
    # Both halves of Hero's split survived: neither colour was swallowed by
    # the other, which is what "every face landed in slot 0" would produce.
    assert materials_by_name["Red"] > 0
    assert materials_by_name["Green"] > 0
    assert materials_by_name["Blue"] > 0


@pytest.fixture(scope="module")
def unwrapped(require_bpy, retopoed) -> tuple[bytes, dict]:
    retopo_bytes, _result = retopoed
    return clay_blender.unwrap_bytes(retopo_bytes)


def test_unwrap_adds_uvs_to_every_primitive_without_changing_geometry(unwrapped, retopoed):
    out_bytes, result = unwrapped
    retopo_out_bytes, _retopo_result = retopoed
    assert result["ok"] is True
    assert {o["name"] for o in result["objects"]} == {"Hero", "Prop"}
    for obj in result["objects"]:
        assert obj["islands"] > 0

    before = gltf.load(retopo_out_bytes)
    after = gltf.load(out_bytes)
    before_faces = sum(len(p.indices) // 3 for prims in before.meshes for p in prims)
    after_faces = sum(len(p.indices) // 3 for prims in after.meshes for p in prims)
    assert after_faces == before_faces, "unwrap must not touch geometry"
    for prims in after.meshes:
        for p in prims:
            assert p.uvs is not None


def test_bake_refuses_a_low_mesh_with_no_uvs(retopoed, two_object_glb_bytes, require_bpy):
    retopo_out_bytes, _result = retopoed
    with pytest.raises(clay_blender.ClayBlenderError, match="UVs|unwrap"):
        clay_blender.bake_bytes(
            two_object_glb_bytes, retopo_out_bytes, texture_size=TEXTURE_PX, cage_extrusion=0.02
        )


def test_bake_produces_the_requested_maps_and_a_metallic_constant(
    unwrapped, two_object_glb_bytes, require_bpy
):
    low_bytes, _unwrap_result = unwrapped
    out_bytes, result = clay_blender.bake_bytes(
        two_object_glb_bytes,
        low_bytes,
        texture_size=TEXTURE_PX,
        cage_extrusion=0.02,
        maps=["base_color", "normal"],
    )
    assert result["ok"] is True
    assert set(result["maps"]) == {"base_color", "normal"}
    assert result["texture_size"] == TEXTURE_PX
    # Red/Green (0.0) and Blue (0.25) averaged over the three source
    # materials -- see ``_metallic_constant``.
    assert result["metallic"] == pytest.approx(0.25 / 3, abs=1e-6)

    after = gltf.load(out_bytes)
    assert [n.name for n in after.nodes] == ["Hero", "Prop"]
    for prims in after.meshes:
        for p in prims:
            assert p.material.base_color is not None, "no base colour survived the bake"
            assert p.material.normal is not None, "no normal map survived the bake"
