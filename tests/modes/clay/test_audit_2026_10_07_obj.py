"""The 2026-10-07 Clay audit's OBJ findings: clay-15 (BOM), clay-16 (names),
clay-18 (float32 overflow) and clay-69 (Clay's own OBJ set round-trips)."""

from __future__ import annotations

import numpy as np
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import meshimport, objexport, objimport
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh.elements import OpError

_OBJ = "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\nusemtl Red\nf 1 2 3\nf 2 4 3\n"
_MTL = "newmtl Red\nKd 1 0 0\n"


def _imported(obj: str, mtl: str):
    doc = meshimport.import_file(obj.encode("utf-8"), ".obj", "t", mtl=mtl)
    ob = doc.objects[0]
    return (
        len(ob.mesh.positions),
        bm.face_count(ob.mesh),
        ob.mesh.positions[0].tolist(),
        tuple(doc.materials[int(ob.mesh.material[0])].base_color_factor),
    )


# --- clay-15 ---------------------------------------------------------------


def test_an_obj_and_mtl_with_a_utf8_bom_import_as_if_without_one() -> None:
    plain = _imported(_OBJ, _MTL)
    assert plain[0] == 4 and plain[1] == 2
    assert plain[3][:3] == (1.0, 0.0, 0.0)
    assert _imported("﻿" + _OBJ, _MTL) == plain
    assert _imported(_OBJ, "﻿" + _MTL) == plain
    assert _imported("﻿" + _OBJ, "﻿" + _MTL) == plain


def test_a_bom_on_the_mtl_text_does_not_hide_its_first_newmtl() -> None:
    # The app reads the sibling .mtl with ``read_text(encoding="utf-8")``, which
    # keeps a BOM, so the parser itself must tolerate one.
    assert "Red" in objimport._parse_mtl("﻿" + _MTL)


# --- clay-16 ---------------------------------------------------------------


@pytest.mark.parametrize("name", ["Crate\\", "Box #2", "Wall\\\\", "Crate\\ ", "#"])
def test_an_object_name_ending_in_a_backslash_survives_an_obj_round_trip(name: str) -> None:
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name=name, mesh=bp.box(), generator="box"))
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="Other", mesh=bp.box(), generator="box"))
    obj, mtl = objexport.claydoc_to_obj(doc, name="m")
    back = meshimport.import_file(obj.encode(), ".obj", "m", mtl=mtl)
    assert len(back.objects) == 2
    for original, reimported in zip(doc.objects, back.objects, strict=True):
        assert len(reimported.mesh.positions) == len(original.mesh.positions)
        assert bm.face_count(reimported.mesh) == bm.face_count(original.mesh)
    assert back.objects[1].name == "Other"
    # The first name keeps everything but the two characters OBJ cannot hold.
    assert "#" not in back.objects[0].name and not back.objects[0].name.endswith("\\")


def test_a_document_title_with_a_hash_gives_an_mtllib_and_texture_name_that_survive() -> None:
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box(), generator="box"))
    obj, _ = objexport.claydoc_to_obj(doc, name="my #1 model")
    mtllib = next(line for line in obj.splitlines() if line.startswith("mtllib"))
    assert "#" not in mtllib
    assert mtllib == f"mtllib {objexport.safe_name('my #1 model')}.mtl"
    assert "#" not in objexport.texture_name("my #1 model", 0)


# --- clay-18 ---------------------------------------------------------------


def test_an_obj_vertex_that_overflows_float32_is_refused_by_name() -> None:
    with pytest.raises(OpError, match="float32|too large"):
        objimport.obj_to_claydoc("v 1e39 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    # The same coordinates built by the scale knob (1e30 x 1e10).
    with pytest.raises(OpError, match="float32|too large"):
        objimport.obj_to_claydoc("v 1e30 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", scale=1e10)
    # A large but representable coordinate still imports.
    doc = objimport.obj_to_claydoc("v 1e30 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    assert np.isfinite(doc.objects[0].mesh.positions).all()


# --- clay-69 ---------------------------------------------------------------


def _two_slot_doc(*, texture: bool = False) -> bd.ClayDoc:
    unused = bd.default_material("Unused")
    wood = bd.default_material("Wood")
    wood.base_color_factor = (0.5, 0.25, 0.125, 1.0)
    stone = bd.default_material("Stone")
    stone.base_color_factor = (0.25, 0.25, 0.5, 1.0)
    doc = bd.ClayDoc(materials=[unused, wood, stone])
    mesh = bp.box()
    material = np.ones(bm.face_count(mesh), dtype="i4")
    material[::2] = 2
    mesh = bm.Mesh(
        positions=mesh.positions,
        loops=mesh.loops,
        starts=mesh.starts,
        material=material,
        smooth=mesh.smooth,
        uv=mesh.uv,
    )
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=mesh, generator=None))
    return doc


def test_an_exported_obj_reimports_without_an_unused_default_palette_slot() -> None:
    doc = _two_slot_doc()
    obj, mtl = objexport.claydoc_to_obj(doc)
    back = objimport.obj_to_claydoc(obj, mtl=mtl)
    # Slots 1 and 2 were used; slot 0 was not, and must not come back as an
    # invented grey "Material" in front of them.
    assert sorted(m.name for m in back.materials) == ["Stone", "Wood"]
    used = set(int(i) for i in back.objects[0].mesh.material)
    assert used == {0, 1}


def test_an_exported_materials_names_and_colours_come_back() -> None:
    doc = _two_slot_doc()
    obj, mtl = objexport.claydoc_to_obj(doc)
    back = objimport.obj_to_claydoc(obj, mtl=mtl)
    by_name = {m.name: tuple(round(c, 4) for c in m.base_color_factor) for m in back.materials}
    assert by_name == {"Wood": (0.5, 0.25, 0.125, 1.0), "Stone": (0.25, 0.25, 0.5, 1.0)}


def test_a_material_name_with_a_hash_comes_back_whole() -> None:
    mat = bd.default_material("Wall #2")
    doc = bd.ClayDoc(materials=[mat])
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box(), generator="box"))
    obj, mtl = objexport.claydoc_to_obj(doc)
    back = objimport.obj_to_claydoc(obj, mtl=mtl)
    assert back.materials[0].name == "Wall #2"


def test_a_file_whose_faces_all_name_a_material_has_no_implicit_slot() -> None:
    doc = objimport.obj_to_claydoc(_OBJ, mtl=_MTL)
    assert [m.name for m in doc.materials] == ["Red"]


def test_a_face_before_any_usemtl_still_gets_the_implicit_slot() -> None:
    text = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\nusemtl Red\nf 1 2 3\n"
    doc = objimport.obj_to_claydoc(text, mtl=_MTL)
    assert [m.name for m in doc.materials] == ["Material", "Red"]
    assert doc.objects[0].mesh.material.tolist() == [0, 1]


def test_a_foreign_mtl_keeps_its_newmtl_names() -> None:
    doc = objimport.obj_to_claydoc(_OBJ, mtl=_MTL)
    assert doc.materials[0].name == "Red"
