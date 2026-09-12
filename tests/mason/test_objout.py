"""``mason/objout.py``: the scene as a merged OBJ, and what a merged OBJ can lose.

Every test name is a claim, written to fail against the export this file is
guarding against: one whose face indices restart per object (the classic
merged-OBJ bug -- one object at the origin wearing everyone else's
triangles), one that transforms a normal by the matrix instead of its
inverse-transpose, one that writes ``vn inf inf inf`` for a node scaled to
zero on some axis, one that drops a light or an unresolved reference without
saying so, one that reads a prefab instance's template node instead of its
resolved world transform, and one whose bytes are not the same twice for the
same document.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.mason import document as doc
from warlock.studio.mason import nodes as nd
from warlock.studio.mason import objout, refs, scene
from warlock.studio.mason import terrain as tr
from warlock.studio.viewer import gltf
from warlock.studio.viewer import math3d as m3


def _prim(
    x: float = 0.0,
    *,
    name: str = "",
    normals: np.ndarray | None = None,
    uvs: np.ndarray | None = None,
    material: gltf.Material | None = None,
) -> gltf.Primitive:
    """One triangle, offset along X so two refs' geometry is distinguishable."""
    return gltf.Primitive(
        positions=np.array([[x, 0, 0], [x + 1, 0, 0], [x, 1, 0]], dtype="f4"),
        indices=np.array([0, 1, 2], dtype="u4"),
        normals=normals,
        uvs=uvs,
        material=material if material is not None else gltf.Material(name=name),
    )


class _Source:
    """A ``GeometrySource`` over a fixed table -- ``tests/mason/test_gltfout.py``'s
    own fake, copied rather than imported across test modules."""

    def __init__(self, table: dict | None = None) -> None:
        self.table = table or {}
        self.rev = 0

    def primitives(self, ref):
        return self.table.get(refs.ref_key(ref), [])

    def box(self, ref):
        prims = self.primitives(ref)
        return prims[0].box() if prims else None


def _source(*generators: str) -> _Source:
    table = {}
    for i, name in enumerate(generators):
        table[refs.ref_key(refs.primitive_ref(name, {}))] = [_prim(float(i), name=name)]
    return _Source(table)


def _mesh(name: str = "box", **params) -> nd.MeshNode:
    return nd.MeshNode(uid=nd.new_uid(), name="", ref=refs.primitive_ref(name, params))


def _obj_lines(export: objout.ObjExport) -> list[str]:
    return export.files[objout.OBJ].decode("utf-8").splitlines()


def _v_lines(lines: list[str]) -> np.ndarray:
    return np.array(
        [[float(x) for x in line.split()[1:]] for line in lines if line.startswith("v ")]
    )


# --- determinism -------------------------------------------------------------


def test_two_exports_of_an_unchanged_document_are_byte_identical():
    d = doc.MasonDoc()
    d.add_node(_mesh())
    source = _source("box")
    first = objout.obj_export(d, source)
    second = objout.obj_export(d, source)
    assert first.files == second.files


# --- the merged-OBJ bug --------------------------------------------------


def test_face_indices_are_one_based_and_monotone_across_several_objects():
    """Indices restarting per object load as one object at the origin wearing
    every other object's triangles -- this is the test that would catch it."""
    d = doc.MasonDoc()
    for _ in range(3):
        d.add_node(_mesh())
    out = objout.obj_export(d, _source("box"))
    lines = _obj_lines(out)
    faces = [line for line in lines if line.startswith("f ")]
    assert len(faces) == 3
    flat = [int(token.split("/")[0]) for face in faces for token in face[2:].split()]
    assert flat == list(range(1, 10))


def test_the_merged_geometry_bounds_match_scene_world_bounds():
    """The plan's own stated gate for this exporter."""
    d = doc.MasonDoc()
    group = nd.GroupNode(uid=nd.new_uid(), translation=m3.vec3(1.0, 2.0, -3.0))
    group.children.append(_mesh())
    d.add_node(group)
    d.add_node(_mesh())
    source = _source("box")

    out = objout.obj_export(d, source)
    verts = _v_lines(_obj_lines(out))
    expected = scene.world_bounds(d, source)
    assert expected is not None
    exp_lo, exp_hi = expected
    assert verts.min(axis=0) == pytest.approx(exp_lo, abs=1e-5)
    assert verts.max(axis=0) == pytest.approx(exp_hi, abs=1e-5)


def test_a_mesh_under_a_translated_group_lands_at_the_translated_position():
    d = doc.MasonDoc()
    group = nd.GroupNode(uid=nd.new_uid(), translation=m3.vec3(5.0, 0.0, 0.0))
    group.children.append(_mesh())
    d.add_node(group)

    out = objout.obj_export(d, _source("box"))
    first_vertex = _v_lines(_obj_lines(out))[0]
    assert first_vertex == pytest.approx([5.0, 0.0, 0.0])


# --- normals -------------------------------------------------------------


def test_normals_are_transformed_by_the_inverse_transpose_not_the_matrix():
    """A non-uniform scale is the only case that tells the two apart: the
    inverse-transpose keeps a normal perpendicular to its transformed
    surface, and a naive matrix transform generally does not."""
    d = doc.MasonDoc()
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]], dtype="f4")
    raw_normal = np.array([0.0, -1.0, 1.0], dtype="f4")
    raw_normal /= np.linalg.norm(raw_normal)
    normals = np.tile(raw_normal, (3, 1))
    mesh = nd.MeshNode(uid=nd.new_uid(), name="Panel", ref=refs.primitive_ref("panel", {}))
    mesh.scale = m3.vec3(2.0, 1.0, 4.0)
    d.add_node(mesh)
    source = _Source(
        {
            refs.ref_key(refs.primitive_ref("panel", {})): [
                gltf.Primitive(
                    positions=positions,
                    indices=np.array([0, 1, 2], dtype="u4"),
                    normals=normals,
                )
            ]
        }
    )

    out = objout.obj_export(d, source)
    lines = _obj_lines(out)
    v = _v_lines(lines)
    vn = np.array(
        [[float(x) for x in line.split()[1:]] for line in lines if line.startswith("vn ")]
    )
    assert len(vn) == 3
    edge1 = v[1] - v[0]
    edge2 = v[2] - v[0]
    for normal in vn:
        assert normal @ edge1 == pytest.approx(0.0, abs=1e-4)
        assert normal @ edge2 == pytest.approx(0.0, abs=1e-4)


def test_a_singular_scale_leaves_the_normals_out_rather_than_writing_nans():
    d = doc.MasonDoc()
    mesh = _mesh()
    mesh.name = "Flat"
    mesh.scale = m3.vec3(1.0, 0.0, 1.0)  # zero on Y: a singular 3x3
    d.add_node(mesh)
    source = _Source(
        {
            refs.ref_key(refs.primitive_ref("box", {})): [
                gltf.Primitive(
                    positions=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="f4"),
                    indices=np.array([0, 1, 2], dtype="u4"),
                    normals=np.array([[0, 0, 1]] * 3, dtype="f4"),
                )
            ]
        }
    )

    out = objout.obj_export(d, source)
    text = out.files[objout.OBJ].decode("utf-8")
    assert "vn " not in text
    assert "inf" not in text.lower()
    assert "nan" not in text.lower()
    faces = [line for line in text.splitlines() if line.startswith("f ")]
    assert faces
    for face in faces:
        for token in face[2:].split():
            assert "/" not in token  # a plain vertex index -- no normal slot at all
    assert any("Flat" in s and "singular" in s for s in out.skipped)


# --- what OBJ cannot carry -------------------------------------------------


def test_lights_and_cameras_are_reported_in_skipped_and_are_not_in_the_obj():
    d = doc.MasonDoc()
    d.add_node(nd.LightNode(uid=nd.new_uid(), name="Sun", kind="directional"))
    d.add_node(nd.CameraNode(uid=nd.new_uid(), name="Shot"))

    out = objout.obj_export(d, _source("box"))
    text = out.files[objout.OBJ].decode("utf-8")
    assert "o Sun" not in text
    assert "o Shot" not in text
    assert any("'Sun'" in s and "light" in s for s in out.skipped)
    assert any("'Shot'" in s and "camera" in s for s in out.skipped)


def test_an_unresolved_reference_is_reported_and_costs_only_its_own_geometry():
    d = doc.MasonDoc()
    d.add_node(nd.MeshNode(uid=nd.new_uid(), name="Ghost", ref=refs.LibraryRef(job_id="gone")))
    d.add_node(_mesh())

    out = objout.obj_export(d, _source("box"))
    lines = _obj_lines(out)
    o_lines = [line for line in lines if line.startswith("o ")]
    assert o_lines == ["o Mesh"]  # the other node, unnamed -> DEFAULT_NAMES["mesh"]
    assert any("Ghost" in s for s in out.skipped)


# --- names -----------------------------------------------------------------


def test_one_o_group_per_node_with_unique_names():
    d = doc.MasonDoc()
    for _ in range(3):
        node = _mesh()
        node.name = "Rock"
        d.add_node(node)

    out = objout.obj_export(d, _source("box"))
    o_lines = [line[2:] for line in _obj_lines(out) if line.startswith("o ")]
    assert o_lines == ["Rock", "Rock.001", "Rock.002"]


# --- materials and textures ------------------------------------------------


def test_usemtl_names_every_distinct_material_once_and_the_mtl_matches():
    d = doc.MasonDoc()
    plain = _mesh()
    tinted = _mesh()
    tinted.material = gltf.Material(name="mossy")
    d.add_node(plain)
    d.add_node(tinted)

    out = objout.obj_export(d, _source("box"))
    obj_text = out.files[objout.OBJ].decode("utf-8")
    mtl_text = out.files[objout.MTL].decode("utf-8")
    usemtl_names = [
        line.split(maxsplit=1)[1] for line in obj_text.splitlines() if line.startswith("usemtl ")
    ]
    assert len(usemtl_names) == len(set(usemtl_names)) == 2
    for name in usemtl_names:
        assert f"newmtl {name}" in mtl_text


def test_a_base_colour_texture_reaches_files_and_is_named_by_map_kd():
    d = doc.MasonDoc()
    material = gltf.Material(
        name="crate",
        base_color=(2, 2, bytes([255, 0, 0, 255]) * 4),
        normal=(2, 2, bytes([128, 128, 255, 255]) * 4),
    )
    mesh = nd.MeshNode(uid=nd.new_uid(), name="Crate", ref=refs.primitive_ref("crate", {}))
    d.add_node(mesh)
    source = _Source(
        {refs.ref_key(refs.primitive_ref("crate", {})): [_prim(material=material)]}
    )

    out = objout.obj_export(d, source)
    mtl_text = out.files[objout.MTL].decode("utf-8")
    assert f"map_Kd {objout.TEXTURE_DIR}/0.png" in mtl_text
    assert f"{objout.TEXTURE_DIR}/0.png" in out.files
    assert any("material 'crate' carries a normal map" in s for s in out.skipped)


# --- the refusal -------------------------------------------------------------


def test_max_obj_verts_refuses_before_formatting(monkeypatch):
    d = doc.MasonDoc()
    d.add_node(_mesh())  # one triangle: three vertices
    source = _source("box")
    monkeypatch.setattr(objout, "MAX_OBJ_VERTS", 1)

    def _boom(*args, **kwargs):
        raise AssertionError("no bytes should be formatted for a refused export")

    monkeypatch.setattr(objout, "_format", _boom)

    with pytest.raises(ValueError) as excinfo:
        objout.obj_export(d, source)
    # If _format had run instead of the refusal, the monkeypatch above would
    # have raised AssertionError, which pytest.raises(ValueError) does not
    # catch -- so this test failing that way is itself proof formatting ran.
    assert "3" in str(excinfo.value)
    assert "1" in str(excinfo.value)


# --- terrain and prefabs -----------------------------------------------


def test_the_terrain_exports_as_ordinary_geometry():
    d = doc.MasonDoc()
    d.set_terrain(
        tr.Terrain(
            heights=np.zeros((3, 3), dtype="f4"),
            size_x=4.0,
            size_z=4.0,
            material=gltf.Material(name="grass"),
        )
    )
    d.add_node(nd.TerrainNode(uid=nd.new_uid(), name="Ground"))

    out = objout.obj_export(d, _source("box"))
    lines = _obj_lines(out)
    assert "o Ground" in lines
    assert len(_v_lines(lines)) == 9  # a 3x3 grid of vertices over 2x2 cells


def test_a_prefab_instances_three_copies_each_land_at_their_own_place():
    """The test that would fail if the exporter read ``Placed.node`` (the
    template, shared by every instance and translated nowhere) instead of
    ``Placed.world`` (each instance's own resolved transform)."""
    d = doc.MasonDoc()
    template = nd.MeshNode(uid=nd.new_uid(), name="Crate", ref=refs.primitive_ref("box", {}))
    d.define_prefab("crate", template)
    for i in range(3):
        d.add_node(
            nd.PrefabNode(
                uid=nd.new_uid(),
                name=f"crate{i}",
                template="crate",
                translation=m3.vec3(float(i) * 5.0, 0.0, 0.0),
            )
        )

    out = objout.obj_export(d, _source("box"))
    text = out.files[objout.OBJ].decode("utf-8")
    bodies = text.split("\no ")[1:]
    assert len(bodies) == 3
    mins = sorted(
        min(float(line.split()[1]) for line in body.splitlines() if line.startswith("v "))
        for body in bodies
    )
    assert mins == pytest.approx([0.0, 5.0, 10.0])


# --- bytes ----------------------------------------------------------------


def test_line_endings_are_lf_only_with_no_cr_anywhere_in_the_bytes():
    d = doc.MasonDoc()
    d.add_node(_mesh())

    out = objout.obj_export(d, _source("box"))
    assert b"\r" not in out.files[objout.OBJ]
    assert b"\r" not in out.files[objout.MTL]
    assert out.files[objout.OBJ].endswith(b"\n")
    assert out.files[objout.MTL].endswith(b"\n")


def _named_mesh(node_name: str, generator: str = "box") -> nd.MeshNode:
    """A mesh node with a name of its own -- ``_mesh`` above names the
    *generator*, which is the thing a ref is keyed on, not the outliner row."""
    return nd.MeshNode(
        uid=nd.new_uid(), name=node_name, ref=refs.primitive_ref(generator, {})
    )


def test_the_obj_uses_the_shared_naming_rule_and_not_the_glbs_actual_names():
    """One rule, not one set of names -- and the difference is asserted rather
    than left to be discovered.

    The GLB names every node the walk crosses, groups and prefab instances
    included; a merged OBJ has no room for either, so the two files number
    their duplicates from different populations. A group called "Rock" holding
    a mesh called "Rock" is two nodes in the GLB and one ``o`` group here.
    Nothing needs them to agree: the GLB and its manifest address each other
    by name and are written from one shared record, while an OBJ and its MTL
    are a self-contained pair referring to nothing outside themselves. This is
    the test that keeps that a stated position rather than a surprise.
    """
    from warlock.studio.mason import gltfout

    d = doc.MasonDoc()
    group = nd.GroupNode(uid=nd.new_uid(), name="Rock")
    group.children.append(_named_mesh("Rock"))
    d.add_node(group)

    source = _source("box")
    assert [n.name for n in gltfout.scene_model(d, source).model.nodes] == ["Rock", "Rock.001"]
    out = objout.obj_export(d, source)
    assert [line[2:] for line in _obj_lines(out) if line.startswith("o ")] == ["Rock"]


def test_two_lights_with_one_name_are_reported_as_two_distinguishable_losses():
    """Every placed item claims a name, including the ones that write no
    geometry: a sentence saying "a light was left out" is worth much less than
    one saying *which*, and two lights both called "Sun" would otherwise
    report the same loss twice with nothing to tell them apart."""
    d = doc.MasonDoc()
    d.add_node(nd.LightNode(uid=nd.new_uid(), name="Sun"))
    d.add_node(nd.LightNode(uid=nd.new_uid(), name="Sun"))

    skipped = objout.obj_export(d, _source("box")).skipped
    assert len(skipped) == 2
    assert len(set(skipped)) == 2
    assert any("'Sun'" in line for line in skipped)
    assert any("'Sun.001'" in line for line in skipped)
