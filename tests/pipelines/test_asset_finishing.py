"""Real Blender round trips: PBR channels, transparency, and open thin shapes."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from realmspinner.kernels.geom3d import glbio
from realmspinner.kernels.rig import blender_spec
from realmspinner.pipelines import blender_run, remesh

pytestmark = pytest.mark.timeout(600)


@pytest.fixture(scope="module")
def mixed_torus(tmp_path_factory):
    bpy = pytest.importorskip("bpy")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_torus_add(
        major_segments=64, minor_segments=16, major_radius=1.0, minor_radius=0.12
    )
    obj = bpy.context.view_layer.objects.active
    for name, metallic, roughness, opacity in (
        ("wood", 0.0, 0.75, 1.0),
        ("metal", 1.0, 0.25, 0.4),
    ):
        material = bpy.data.materials.new(name)
        material.use_nodes = True
        tree = material.node_tree
        bsdf = tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (0.6, 0.2, 0.1, 1.0)
        bsdf.inputs["Roughness"].default_value = roughness
        bsdf.inputs["Alpha"].default_value = opacity
        material.surface_render_method = "BLENDED"
        image = bpy.data.images.new(name + "_metallic", width=2, height=2)
        image.colorspace_settings.name = "Non-Color"
        image.pixels[:] = [metallic, metallic, metallic, 1.0] * 4
        image.pack()
        node = tree.nodes.new("ShaderNodeTexImage")
        node.image = image
        tree.links.new(node.outputs["Color"], bsdf.inputs["Metallic"])
        obj.data.materials.append(material)
    for polygon in obj.data.polygons:
        polygon.material_index = int(polygon.center.x > 0)
    path = tmp_path_factory.mktemp("surface") / "mixed.glb"
    bpy.ops.export_scene.gltf(filepath=str(path), export_format="GLB")
    return path


def texture(gltf, blob, texture_index):
    source = gltf["textures"][texture_index]["source"]
    image = gltf["images"][source]
    view = gltf["bufferViews"][image["bufferView"]]
    start = view.get("byteOffset", 0)
    return np.asarray(
        Image.open(io.BytesIO(blob[start : start + view["byteLength"]])).convert("RGBA")
    )


def finish(source, tmp_path, *, preserve):
    out = tmp_path / "finished.glb"
    spec = blender_spec.remesh_spec(
        source,
        out,
        tmp_path,
        target_faces=500,
        texture_size=512,
        preserve_shape=preserve,
        close_holes=not preserve,
    )
    result = blender_run.run_worker(spec)
    return out, result


def test_preserve_shape_retains_maps_transparency_and_the_thin_open_ring(mixed_torus, tmp_path):
    out, result = finish(mixed_torus, tmp_path, preserve=True)
    before, before_blob = glbio.read_glb(mixed_torus)
    after, after_blob = glbio.read_glb(out)
    assert result["method"] == "preserve_shape"
    assert 0 < result["triangles"] <= 1000
    assert len(after["materials"]) == len(before["materials"]) == 2
    for old, new in zip(before["materials"], after["materials"], strict=True):
        assert old.get("alphaMode", "OPAQUE") == new.get("alphaMode", "OPAQUE")
        old_pbr, new_pbr = old["pbrMetallicRoughness"], new["pbrMetallicRoughness"]
        assert old_pbr["baseColorFactor"] == new_pbr["baseColorFactor"]
        assert np.array_equal(
            texture(before, before_blob, old_pbr["metallicRoughnessTexture"]["index"]),
            texture(after, after_blob, new_pbr["metallicRoughnessTexture"]["index"]),
        )
    comparison = remesh.compare_geometry(mixed_torus, out)
    assert comparison["measured"] is True
    assert comparison["worst_lost_coverage"] < 0.1
    openings = [
        v["closed_openings"] for v in comparison["views"] if v["closed_openings"] is not None
    ]
    assert openings and max(openings) < 0.01


def test_repair_bakes_spatial_metallic_roughness_and_opacity(mixed_torus, tmp_path):
    out, result = finish(mixed_torus, tmp_path, preserve=False)
    gltf, blob = glbio.read_glb(out)
    material = gltf["materials"][0]
    pbr = material["pbrMetallicRoughness"]
    mr = texture(gltf, blob, pbr["metallicRoughnessTexture"]["index"])
    color = texture(gltf, blob, pbr["baseColorTexture"]["index"])
    # glTF: roughness in G, metallic in B. Both source regions must survive.
    visible = color[:, :, 0] > 30
    metallic = mr[:, :, 2][visible]
    roughness = mr[:, :, 1][visible]
    assert (metallic < 10).sum() > 100 and (metallic > 240).sum() > 100
    assert (roughness < 100).sum() > 100 and (roughness > 160).sum() > 100
    assert material["alphaMode"] == "BLEND"
    assert (color[:, :, 3][visible] < 150).sum() > 100
    assert (color[:, :, 3][visible] > 240).sum() > 100
    assert result["metallic"] == "baked"


def test_repair_preserves_mask_mode_and_cutoff(mixed_torus, tmp_path):
    source = tmp_path / "masked.glb"
    gltf, blob = glbio.read_glb(mixed_torus)
    for material in gltf["materials"]:
        material["alphaMode"] = "MASK"
        material["alphaCutoff"] = 0.3
    header, _old_gltf, rest = glbio.split_glb(mixed_torus.read_bytes())
    source.write_bytes(glbio.rebuild_glb(header, gltf, rest))
    out, _result = finish(source, tmp_path, preserve=False)
    finished, _ = glbio.read_glb(out)
    assert finished["materials"][0]["alphaMode"] == "MASK"
    assert finished["materials"][0]["alphaCutoff"] == pytest.approx(0.3)


def test_missing_geometry_is_unknown(tmp_path):
    result = remesh.compare_geometry(tmp_path / "missing.glb", tmp_path / "missing2.glb")
    assert result["measured"] is False
    assert "unknown" in " ".join(remesh.finishing_lines({"geometry": result}))
