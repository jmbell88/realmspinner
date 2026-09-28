from __future__ import annotations

import trimesh

from realmspinner import meshreport


def _write(tmp_path, mesh, name="m.glb"):
    path = tmp_path / name
    trimesh.Scene(mesh).export(path)
    return path


def test_a_clean_box_is_ready(tmp_path):
    path = _write(tmp_path, trimesh.creation.box(extents=(1.0, 1.0, 1.0)))
    report = meshreport.build(path)
    assert report["status"] in ("ready", "review")
    assert report["watertight"] is True
    assert report["triangles"] == 12
    assert report["components"] == 1
    assert report["boundary_edges"] == 0
    # A plain trimesh.creation.box carries no material at all.
    assert report["material_count"] == 0


def test_material_count_reflects_the_gltf_materials_array(tmp_path):
    """The 2026-09-26 audit, finding pipelines-mesh-04: Manual 23 has always
    claimed the mesh report records a material count; ``build()`` never did.
    Two geometries citing the same material count once; two geometries with
    their own distinct materials count twice."""
    from trimesh.visual.material import PBRMaterial

    box_a = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    box_b = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    box_b.apply_translation((2.0, 0.0, 0.0))
    shared = PBRMaterial(baseColorFactor=[255, 0, 0, 255])
    box_a.visual = trimesh.visual.TextureVisuals(material=shared)
    box_b.visual = trimesh.visual.TextureVisuals(material=shared)
    scene = trimesh.Scene()
    scene.add_geometry(box_a, node_name="a", geom_name="a")
    scene.add_geometry(box_b, node_name="b", geom_name="b")
    shared_path = tmp_path / "shared.glb"
    scene.export(shared_path)
    assert meshreport.build(shared_path)["material_count"] == 1

    box_c = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    box_d = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    box_d.apply_translation((2.0, 0.0, 0.0))
    box_c.visual = trimesh.visual.TextureVisuals(
        material=PBRMaterial(baseColorFactor=[255, 0, 0, 255])
    )
    box_d.visual = trimesh.visual.TextureVisuals(
        material=PBRMaterial(baseColorFactor=[0, 255, 0, 255])
    )
    distinct_scene = trimesh.Scene()
    distinct_scene.add_geometry(box_c, node_name="c", geom_name="c")
    distinct_scene.add_geometry(box_d, node_name="d", geom_name="d")
    distinct_path = tmp_path / "distinct.glb"
    distinct_scene.export(distinct_path)
    assert meshreport.build(distinct_path)["material_count"] == 2


def test_an_open_surface_is_not_watertight_and_is_flagged(tmp_path):
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    box.faces = box.faces[:-2]          # tear a hole
    box.remove_unreferenced_vertices()
    path = _write(tmp_path, box)
    report = meshreport.build(path)
    assert report["watertight"] is False
    assert report["boundary_edges"] > 0
    assert report["status"] == "review"
    assert any("watertight" in r for r in report["reasons"])


def _seam_split_cube():
    """A cube whose vertices have been duplicated along a UV seam.

    This is what xatlas hands back: to lay a chart flat it splits a vertex in
    two and gives each copy its own UV, leaving the *positions* identical. The
    surface is still closed -- nothing about it has a hole -- but every face on
    one side of the seam now references a different index than its neighbour,
    so an unwelded read of it counts boundary edges and calls it open.
    """
    import numpy as np
    import trimesh

    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    vertices = np.asarray(box.vertices, dtype=np.float64)
    faces = np.asarray(box.faces, dtype=np.int64).copy()
    # A seam through one corner: the faces that touch vertex 0 are split into
    # two groups and the second group is given its own copy of that position.
    touching = np.flatnonzero((faces == 0).any(axis=1))
    twin = len(vertices)
    vertices = np.vstack([vertices, vertices[0]])
    for index in touching[len(touching) // 2 :]:
        faces[index][faces[index] == 0] = twin
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def test_a_seam_split_cube_is_open_raw_and_watertight_welded(tmp_path):
    """The whole point of the welded analysis copy.

    ``build`` loads with ``process=False`` because the UV and material checks
    need the unwelded mesh -- so the raw watertight figure on a textured trellis
    reconstruction was mostly counting seam splits, not holes.
    """
    path = _write(tmp_path, _seam_split_cube())
    report = meshreport.build(path)

    assert report["watertight"] is False
    assert report["boundary_edges"] > 0
    assert report["welded_watertight"] is True
    assert report["welded_boundary_edges"] == 0
    assert report["welded_components"] == 1
    assert not any("watertight" in r for r in report["reasons"])


def test_the_watertight_reason_names_the_welded_numbers(tmp_path):
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    box.faces = box.faces[:-2]          # tear a real hole
    box.remove_unreferenced_vertices()
    report = meshreport.build(_write(tmp_path, box))

    reason = next(r for r in report["reasons"] if "watertight" in r)
    assert str(report["welded_boundary_edges"]) in reason
    assert "weld" in reason


def test_watertight_reason_does_not_claim_welding_when_weld_was_skipped(tmp_path, monkeypatch):
    """The 2026-09-14 audit, pipelines-05: ``_welded`` returns None on a
    degenerate bounding box or a weld that raises, and ``build`` correctly
    falls back to the *unwelded* topology numbers in that case -- but the
    reason string still said "after welding vertices by position" regardless,
    which claims a step that never ran. ``_welded`` is forced to return None
    here the same way it would naturally on a degenerate mesh, so the test
    does not depend on constructing one."""
    monkeypatch.setattr(meshreport, "_welded", lambda *a, **k: None)
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    box.faces = box.faces[:-2]          # tear a real hole
    box.remove_unreferenced_vertices()
    report = meshreport.build(_write(tmp_path, box))

    reason = next(r for r in report["reasons"] if "watertight" in r)
    assert "after welding" not in reason
    assert "unwelded" in reason


def _unwelded_book_fan():
    """Three triangles sharing one spine edge -- a "book" -- each with its own
    private, unwelded copy of the two spine vertices.

    Unwelded, no two triangles reference the same vertex index at all, so
    every one of their nine edges is used by exactly one face: three
    disconnected components, all boundary, zero non-manifold. Only once the
    spine's coincident positions are welded together does the shared edge
    read as what it geometrically is -- one edge used by three faces, the
    textbook non-manifold case.
    """
    import numpy as np
    import trimesh

    a = (0.0, 0.0, 0.0)
    b = (1.0, 0.0, 0.0)
    apexes = [(0.5, 1.0, 0.0), (0.5, -1.0, 0.5), (0.5, 0.0, 1.0)]
    vertices = []
    faces = []
    for i, apex in enumerate(apexes):
        base = 3 * i
        vertices.extend([a, b, apex])
        faces.append([base, base + 1, base + 2])
    return trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )


def test_a_non_manifold_edge_revealed_only_by_welding_is_reported(tmp_path):
    """The 2026-09-26 audit, finding pipelines-mesh-03: ``_topology``'s third
    return value (non-manifold edge count) on the *welded* copy used to be
    discarded with ``_``, so an edge that is only non-manifold once coincident
    seam vertices are merged was computed and then thrown away -- never
    reported, in the reasons list or anywhere in the returned dict."""
    path = _write(tmp_path, _unwelded_book_fan())
    report = meshreport.build(path)

    # Unwelded (raw): three disconnected boundary triangles, no non-manifold
    # edge at all -- this number's meaning is unchanged by the fix.
    assert report["nonmanifold_edges"] == 0
    assert report["components"] == 3

    # Welded: the spine merges into one edge shared by all three faces.
    assert report["welded_nonmanifold_edges"] == 1
    assert any("non-manifold" in r for r in report["reasons"]), report["reasons"]
    reason = next(r for r in report["reasons"] if "non-manifold" in r)
    assert "1" in reason
    assert "after welding" in reason


def test_size_and_grounding_are_measured(tmp_path):
    box = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    box.apply_translation((0.0, 1.0, 0.0))   # glTF is Y-up: min Y == 0
    path = _write(tmp_path, box)
    report = meshreport.build(path, target_size_m=2.0)
    assert report["grounded"] is True
    assert abs(report["achieved_size_m"] - 2.0) < 1e-6


def test_an_unparseable_file_is_invalid(tmp_path):
    path = tmp_path / "broken.glb"
    path.write_bytes(b"not a glb")
    report = meshreport.build(path)
    assert report["status"] == "invalid"
    assert report["reasons"]


def test_component_count_sees_separate_shells_and_lone_faces(tmp_path):
    """The count is taken off the face-adjacency graph rather than mesh.split()
    -- building a full Trimesh per shell to compute one integer is a large
    transient on a 500k-triangle reconstruction. The graph omits faces with no
    neighbour, so a floating triangle is the case that pins the `nodes` arg."""
    import numpy as np
    import trimesh

    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.4)
    box = trimesh.creation.box(extents=(0.3, 0.3, 0.3))
    box.apply_translation([2.0, 0.0, 0.0])
    lone = trimesh.Trimesh(
        vertices=np.array([[5.0, 0.0, 0.0], [5.5, 0.0, 0.0], [5.0, 0.5, 0.0]]),
        faces=np.array([[0, 1, 2]]),
    )
    path = tmp_path / "shells.glb"
    trimesh.util.concatenate([sphere, box, lone]).export(path)

    assert meshreport.build(path)["components"] == 3


def test_a_textured_uv_mapped_mesh_reports_its_uvs_and_base_colour_texture(tmp_path):
    """2026-09-18 audit (pipelines-04): ``_materials()``'s positive branch had
    no fixture at all -- every mesh above is untextured, so a report saying
    "has UVs" or "has a base-color texture" was never actually exercised, only
    the ``not has_uvs`` / ``not textures[...]`` reasons on a bare box. This
    round-trips a UV-mapped PBR material through a real GLB export and read.
    """
    import numpy as np
    from PIL import Image

    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    uv = np.random.default_rng(0).random((len(box.vertices), 2))
    image = Image.new("RGB", (4, 4), (200, 100, 50))
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=image,
        metallicRoughnessTexture=image,
        normalTexture=image,
    )
    box.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)

    report = meshreport.build(_write(tmp_path, box))

    assert report["has_uvs"] is True
    assert report["textures"] == {
        "base_color": True,
        "metallic_roughness": True,
        "normal": True,
    }
    assert not any("UV" in r or "texture" in r for r in report["reasons"])
