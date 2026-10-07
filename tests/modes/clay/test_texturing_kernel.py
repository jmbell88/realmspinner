"""Phase 3 of Clay (picoCAD-style texturing), the kernel half.

``ClayDoc.paint_faces`` and ``ClayDoc.add_texture`` are each one undo step; the
``nearest`` flag survives a save; and every import door hands Clay the reduced
material subset rather than whatever the source file carried.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from realmspinner.kernels.geom3d import glbwrite, gltf
from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import glbimport, objimport
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import serialize as ser


def _doc(*, uv: bool = True) -> tuple[bd.ClayDoc, bd.Obj]:
    """Two palette slots and one box (six faces, all on slot 0)."""
    doc = bd.ClayDoc(materials=[bd.default_material("A"), bd.default_material("B")])
    mesh = bp.box()
    if not uv:
        mesh = replace(mesh, uv=None)
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=mesh, generator="box"))
    return doc, obj


# --- paint_faces -------------------------------------------------------------


def test_paint_faces_writes_only_the_chosen_faces_and_undo_redo_restore_them() -> None:
    doc, obj = _doc()
    before = obj.mesh.material.copy()

    assert doc.paint_faces(obj.uid, [1, 4], 1) is True
    assert doc.by_uid(obj.uid).mesh.material.tolist() == [0, 1, 0, 0, 1, 0]

    assert doc.undo() is True
    assert np.array_equal(doc.by_uid(obj.uid).mesh.material, before)
    assert doc.redo() is True
    assert doc.by_uid(obj.uid).mesh.material.tolist() == [0, 1, 0, 0, 1, 0]


def test_paint_faces_is_one_undo_step_that_keeps_the_generator_claim() -> None:
    doc, obj = _doc()
    head = doc.history.head

    doc.paint_faces(obj.uid, [0, 1, 2], 1)

    assert doc.by_uid(obj.uid).generator == "box"
    doc.undo()
    assert doc.history.head == head


def test_paint_faces_that_change_nothing_push_nothing() -> None:
    doc, obj = _doc()
    head = doc.history.head

    assert doc.paint_faces(obj.uid, [], 1) is False
    assert doc.paint_faces(obj.uid, [0, 3], 0) is False  # already slot 0

    assert doc.history.head == head


def test_paint_faces_refuses_a_slot_or_a_face_that_does_not_exist() -> None:
    doc, obj = _doc()
    head = doc.history.head

    with pytest.raises(el.OpError):
        doc.paint_faces(obj.uid, [0], 2)
    with pytest.raises(el.OpError):
        doc.paint_faces(obj.uid, [0], -1)
    with pytest.raises(el.OpError):
        doc.paint_faces(obj.uid, [6], 1)
    with pytest.raises(el.OpError):
        doc.paint_faces(obj.uid, [-1], 1)

    assert doc.history.head == head


def test_paint_faces_undo_lands_on_its_object_after_a_reorder() -> None:
    doc, a = _doc()
    b = doc.add_object(bd.Obj(uid=bd.new_uid(), name="B", mesh=bp.box()))
    doc.paint_faces(a.uid, [0], 1)

    doc.objects.reverse()
    doc.undo()

    assert doc.by_uid(a.uid).mesh.material.tolist() == [0] * 6
    assert doc.by_uid(b.uid).mesh.material.tolist() == [0] * 6


def test_painted_faces_and_a_nearest_texture_survive_a_save() -> None:
    doc, obj = _doc()
    doc.paint_faces(obj.uid, [2, 5], 1)
    doc.add_texture(1, 32)

    back = ser.read_rblk(ser.rblk_bytes(doc))

    assert back.objects[0].mesh.material.tolist() == [0, 0, 1, 0, 0, 1]
    saved = back.materials[1]
    assert saved.nearest is True
    assert saved.base_color is not None and saved.base_color[:2] == (32, 32)
    assert saved.base_color[2] == doc.materials[1].base_color[2]
    assert back.materials[0].nearest is False


def test_nearest_is_omitted_from_the_scene_json_unless_it_is_set() -> None:
    plain = ser._material_json(bd.default_material())
    crisp = ser._material_json(replace(bd.default_material(), nearest=True))

    assert "nearest" not in plain
    assert crisp["nearest"] is True


def test_a_hand_written_nearest_that_is_not_true_reads_as_smooth() -> None:
    entry = ser._material_json(bd.default_material())
    entry["nearest"] = "false"

    assert ser._material_from(entry, []).nearest is False


# --- add_texture -------------------------------------------------------------


@pytest.mark.parametrize("size", [32, 64, 128])
def test_add_texture_makes_a_square_opaque_nearest_image_in_the_slots_colour(size: int) -> None:
    doc, _ = _doc()
    doc.materials[1] = replace(doc.materials[1], base_color_factor=(1.0, 0.0, 0.0, 1.0))

    assert doc.add_texture(1, size) is True

    material = doc.materials[1]
    assert material.nearest is True
    assert material.base_color is not None
    width, height, data = material.base_color
    assert (width, height, len(data)) == (size, size, size * size * 4)
    assert data[:4] == bytes([255, 0, 0, 255])
    assert data == data[:4] * (size * size)


def test_add_texture_moves_the_colour_into_the_picture_so_it_is_not_applied_twice() -> None:
    """The shader multiplies factor by texel; the picture now carries the slot's
    colour, so a factor left at that colour would render it squared (grey 0.8
    came out near 0.48)."""
    doc, _ = _doc()
    doc.materials[1] = replace(doc.materials[1], base_color_factor=(0.8, 0.4, 0.2, 0.5))

    doc.add_texture(1)

    assert doc.materials[1].base_color_factor == (1.0, 1.0, 1.0, 0.5)
    assert doc.undo()
    assert doc.materials[1].base_color_factor == (0.8, 0.4, 0.2, 0.5)


def test_add_texture_defaults_to_64() -> None:
    doc, _ = _doc()
    doc.add_texture(0)
    assert doc.materials[0].base_color[:2] == (64, 64)


def test_add_texture_refuses_a_bad_size_a_missing_slot_and_an_existing_texture() -> None:
    doc, _ = _doc()
    head = doc.history.head

    for bad in (0, 16, 48, 256, -32):
        with pytest.raises(el.OpError):
            doc.add_texture(0, bad)
    with pytest.raises(el.OpError):
        doc.add_texture(5, 32)
    assert doc.history.head == head

    doc.add_texture(0, 32)
    textured = doc.materials[0]
    head = doc.history.head
    with pytest.raises(el.OpError):
        doc.add_texture(0, 64)
    assert doc.materials[0] is textured
    assert doc.history.head == head


def test_add_texture_box_unwraps_only_the_meshes_that_have_no_uv() -> None:
    doc, bare = _doc(uv=False)
    # A layout no box projection would produce, so "left alone" is observable.
    hand_uv = np.full_like(bp.box().uv, 0.25)
    authored = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="Authored", mesh=replace(bp.box(), uv=hand_uv))
    )
    other_slot = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="Other", mesh=replace(bp.box(), uv=None))
    )
    doc.paint_faces(other_slot.uid, list(range(6)), 1)
    doc.paint_faces(authored.uid, [0], 1)  # authored uses slot 1 as well, but has uv

    doc.add_texture(0, 32)

    assert doc.by_uid(bare.uid).mesh.uv is not None
    assert np.array_equal(doc.by_uid(authored.uid).mesh.uv, hand_uv)
    assert doc.by_uid(other_slot.uid).mesh.uv is None  # no face on slot 0
    assert doc.by_uid(bare.uid).generator == "box"  # a UV change is not a topology change


def test_add_texture_is_one_undo_step_that_restores_the_material_and_the_uv() -> None:
    doc, obj = _doc(uv=False)
    before_material, before_mesh = doc.materials[0], obj.mesh
    head = doc.history.head

    doc.add_texture(0, 64)
    assert doc.materials[0].base_color is not None
    assert doc.by_uid(obj.uid).mesh.uv is not None

    assert doc.undo() is True
    assert doc.history.head == head
    assert doc.materials[0] is before_material
    assert doc.by_uid(obj.uid).mesh is before_mesh
    assert doc.by_uid(obj.uid).mesh.uv is None

    assert doc.redo() is True
    assert doc.materials[0].base_color is not None
    assert doc.by_uid(obj.uid).mesh.uv is not None


# --- reduce_material ---------------------------------------------------------


def test_reduce_material_keeps_nearest_and_still_strips_the_rest() -> None:
    image = (2, 2, bytes(16))
    heavy = gltf.Material(
        name="m",
        base_color_factor=(0.1, 0.2, 0.3, 1.0),
        metallic_factor=1.0,
        roughness_factor=0.1,
        emissive_factor=(1.0, 0.0, 0.0),
        alpha_mode="BLEND",
        alpha_cutoff=0.9,
        base_color=image,
        normal=image,
        emissive=image,
        occlusion=image,
        metallic_roughness=image,
        nearest=True,
    )

    out = bd.reduce_material(heavy)

    assert out.nearest is True
    assert out.base_color is image
    assert (out.metallic_factor, out.roughness_factor) == (0.0, 0.6)
    assert out.emissive_factor == (0.0, 0.0, 0.0)
    assert out.alpha_mode == "OPAQUE"
    assert out.normal is out.emissive is out.occlusion is out.metallic_roughness is None
    assert bd.reduce_material(out) is out


# --- the import doors strip --------------------------------------------------


def _heavy_material() -> gltf.Material:
    image = (2, 2, bytes(range(16)))
    return gltf.Material(
        name="heavy",
        base_color_factor=(0.5, 0.25, 0.75, 1.0),
        metallic_factor=1.0,
        roughness_factor=0.1,
        emissive_factor=(0.4, 0.5, 0.6),
        alpha_mode="BLEND",
        double_sided=True,
        base_color=image,
        normal=image,
        emissive=image,
        metallic_roughness=image,
        occlusion=image,
    )


def test_a_glb_import_arrives_with_the_reduced_material_subset() -> None:
    quad = gltf.Primitive(
        positions=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="f4"),
        indices=np.array([0, 1, 2], dtype="u4"),
        normals=np.tile(np.array([0, 0, 1], dtype="f4"), (3, 1)),
        uvs=np.array([[0, 0], [1, 0], [0, 1]], dtype="f4"),
        material=_heavy_material(),
    )
    node = gltf.Node(name="N", mesh=0, scale=m3.vec3(1, 1, 1))
    data = glbwrite.write_glb(gltf.Model([node], [0], [[quad]], []))

    doc = glbimport.glb_to_claydoc(data, "heavy")

    (material,) = doc.materials
    assert material.name == "heavy"
    assert material.base_color_factor == pytest.approx((0.5, 0.25, 0.75, 1.0))
    assert material.double_sided is True
    assert material.base_color is not None
    assert (material.metallic_factor, material.roughness_factor) == (0.0, 0.6)
    assert material.emissive_factor == (0.0, 0.0, 0.0)
    assert material.alpha_mode == "OPAQUE"
    assert material.normal is None
    assert material.emissive is None
    assert material.metallic_roughness is None
    assert material.occlusion is None


def test_a_nearest_glb_texture_stays_crisp_through_the_import_door() -> None:
    image = (2, 2, bytes(range(16)))
    prim = gltf.Primitive(
        positions=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="f4"),
        indices=np.array([0, 1, 2], dtype="u4"),
        uvs=np.array([[0, 0], [1, 0], [0, 1]], dtype="f4"),
        material=gltf.Material(base_color=image, nearest=True),
    )
    data = glbwrite.write_glb(gltf.Model([gltf.Node(mesh=0)], [0], [[prim]], []))

    doc = glbimport.glb_to_claydoc(data, "crisp")

    assert doc.materials[0].nearest is True


def test_an_obj_import_goes_through_reduce_material() -> None:
    obj = "mtllib m.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nusemtl Red\nf 1 2 3\n"
    mtl = "newmtl Red\nKd 1 0 0\nd 1\nNs 100\n"

    doc = objimport.obj_to_claydoc(obj, mtl=mtl)

    assert doc.materials
    for material in doc.materials:
        assert bd.reduce_material(material) is material
        assert material.base_color is None and material.nearest is False


def test_stl_and_ply_imports_carry_only_the_default_material() -> None:
    from realmspinner.kernels.mesh import meshimport

    doc = meshimport.import_file(
        b"solid s\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\n"
        b"vertex 0 1 0\nendloop\nendfacet\nendsolid s\n",
        ".stl",
    )

    assert len(doc.materials) == 1
    assert bd.reduce_material(doc.materials[0]) is doc.materials[0]



# --- the OBJ file door writes the PNG beside the OBJ ---------------------------


def test_export_mesh_file_obj_writes_each_texture_png_beside_the_obj_under_the_chosen_name(
    tmp_path, monkeypatch
) -> None:
    from realmspinner.studio import dialogs
    from realmspinner.studio.modes.clay import mode as clay_mode

    from .test_clay_tranche1_import_export import FakeCtx

    ctx = FakeCtx()
    doc, _ = _doc()
    doc.paint_faces(doc.objects[0].uid, [0, 1], 1)
    doc.add_texture(1, 32)
    tab = clay_mode.adopt(ctx, doc, title="Scene")
    out = tmp_path / "renamed.obj"
    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: out)

    clay_mode.export_mesh_file(ctx, tab, "obj")

    mtl = (tmp_path / "renamed.mtl").read_text(encoding="utf-8")
    assert "map_Kd renamed_1.png" in mtl.splitlines()
    assert "Scene_1.png" not in mtl, "the title the save dialog replaced"
    png = tmp_path / "renamed_1.png"
    assert png.is_file()
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert sorted(p.name for p in tmp_path.glob("*.png")) == ["renamed_1.png"]


def test_export_mesh_file_obj_with_no_texture_writes_no_png(tmp_path, monkeypatch) -> None:
    from realmspinner.studio import dialogs
    from realmspinner.studio.modes.clay import mode as clay_mode

    from .test_clay_tranche1_import_export import FakeCtx

    ctx = FakeCtx()
    doc, _ = _doc()
    tab = clay_mode.adopt(ctx, doc, title="Scene")
    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: tmp_path / "plain.obj")

    clay_mode.export_mesh_file(ctx, tab, "obj")

    assert (tmp_path / "plain.obj").is_file()
    assert not list(tmp_path.glob("*.png"))
    assert "map_Kd" not in (tmp_path / "plain.mtl").read_text(encoding="utf-8")
