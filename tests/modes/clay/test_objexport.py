"""A Clay document out as OBJ + MTL, and the round trip back through
:mod:`realmspinner.kernels.mesh.objimport`."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import objexport, objimport
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import uv as uv_mod


def _variety_doc() -> bd.ClayDoc:
    red = bd.default_material("Red")
    red.base_color_factor = (1.0, 0.0, 0.0, 1.0)
    red.roughness_factor = 0.2
    blue = bd.default_material("Blue")
    blue.base_color_factor = (0.0, 0.0, 1.0, 0.6)
    blue.roughness_factor = 0.8
    doc = bd.ClayDoc(materials=[red, blue])

    box = bp.box()  # quads, already carries a box-unwrap uv -- every face material 0
    doc.objects.append(bd.Obj(uid=bd.new_uid(), name="Box", mesh=box, material=0))

    # ``Obj.material`` is only the default a *new* face gets; every generator
    # paints its own faces material 0, so the cylinder's own per-face array is
    # repainted here to actually put some faces on the second palette slot.
    cyl = bp.cylinder(segments=6)  # n-gon caps (a hexagon each)
    cyl = replace(cyl, material=np.ones(bm.face_count(cyl), dtype="i4"))
    doc.objects.append(bd.Obj(uid=bd.new_uid(), name="Cyl", mesh=cyl, material=1))

    planar = uv_mod.box_unwrap(bp.plane())
    doc.objects.append(bd.Obj(uid=bd.new_uid(), name="Plane", mesh=planar, material=0))
    return doc


def _material_key(material) -> tuple:
    # Colour only: roughness is fixed at Clay's default on the way in
    # (``document.reduce_material``), so a hand-set one is not a round trip.
    return (tuple(round(c, 4) for c in material.base_color_factor),)


def test_round_trip_preserves_face_counts_ngons_material_assignment_and_uv() -> None:
    doc = _variety_doc()
    obj_text, mtl_text = objexport.claydoc_to_obj(doc)
    back = objimport.obj_to_claydoc(obj_text, mtl=mtl_text)

    assert len(back.objects) == len(doc.objects)
    for orig, reimported in zip(doc.objects, back.objects, strict=True):
        om, rm = orig.mesh, reimported.mesh
        bm.validate(rm)

        assert bm.face_count(om) == bm.face_count(rm)
        assert np.array_equal(
            np.diff(om.starts), np.diff(rm.starts)
        ), "every face's corner count (n-gon or not) survives the round trip"

        # Per-face corner positions, independent of how each side happened to
        # number its own vertices internally. The 2026-09-26 audit's
        # clay-io-13 widened this writer's number format from "%.6g" (six
        # *significant* digits, losing a hair more than 1e-6 absolute near
        # this fixture's own magnitude-1-3 scale) to "%.9g", so the looser
        # ``atol`` this comment used to need is no longer -- kept anyway,
        # since an f4 position's own precision is the real floor here.
        for face in range(bm.face_count(om)):
            o_lo, o_hi = int(om.starts[face]), int(om.starts[face + 1])
            r_lo, r_hi = int(rm.starts[face]), int(rm.starts[face + 1])
            o_pts = om.positions[om.loops[o_lo:o_hi]]
            r_pts = rm.positions[rm.loops[r_lo:r_hi]]
            assert np.allclose(o_pts, r_pts, atol=1e-5)

        # Material assignment: faces that shared a material before the round
        # trip still share one (possibly renumbered) material afterward, and
        # that material's own properties survive.
        for face in range(bm.face_count(om)):
            o_mat = doc.materials[int(om.material[face])]
            r_mat = back.materials[int(rm.material[face])]
            assert _material_key(o_mat) == _material_key(r_mat)

        if om.uv is not None:
            assert rm.uv is not None
            assert np.allclose(om.uv, rm.uv, atol=1e-6)
        else:
            assert rm.uv is None


def test_material_assignment_partitions_the_same_way() -> None:
    """Two faces sharing a material before the round trip still share one
    (not necessarily the same-numbered) material afterward -- and two faces
    that did *not* share one still don't."""
    doc = _variety_doc()
    obj_text, mtl_text = objexport.claydoc_to_obj(doc)
    back = objimport.obj_to_claydoc(obj_text, mtl=mtl_text)

    box_back = back.objects[0].mesh  # every face material=0 in the source doc
    assert len(set(box_back.material.tolist())) == 1

    cyl_back = back.objects[1].mesh  # every face material=1 in the source doc
    assert len(set(cyl_back.material.tolist())) == 1

    # The box's and the cylinder's own materials were different in the
    # source document, and the round trip must not collapse them together.
    assert box_back.material[0] != cyl_back.material[0]


def test_export_is_deterministic() -> None:
    doc = _variety_doc()
    a = objexport.claydoc_to_obj(doc)
    b = objexport.claydoc_to_obj(doc)
    assert a == b


def test_hidden_objects_are_left_out_unless_asked_for() -> None:
    doc = _variety_doc()
    doc.objects[1].visible = False
    obj_text, _ = objexport.claydoc_to_obj(doc)
    assert "o Cyl" not in obj_text
    assert "o Box" in obj_text

    obj_text_all, _ = objexport.claydoc_to_obj(doc, visible_only=False)
    assert "o Cyl" in obj_text_all


def test_a_kilometre_scale_position_keeps_sub_millimetre_precision() -> None:
    """The 2026-09-26 audit, finding clay-io-13: ``%.6g`` keeps only six
    *significant* digits, so a coordinate around 1000 (a real-world scale in
    metres -- a kilometre-scale outdoor level) rounds to the nearest
    millimetre or coarser. ``%.9g`` keeps a float32 position's own ~7.2
    significant decimal digits with a full digit of headroom, so the written
    text must resolve this vertex's own sub-millimetre offset.
    """
    doc = bd.ClayDoc()
    positions = np.array(bp.box().positions)
    # A vertex at x=1234.5678901, translated by nothing further -- picked so
    # a six-significant-digit rounding (%.6g -> "1234.57") and a nine-digit
    # one (%.9g -> "1234.56789") disagree by well over a millimetre.
    # ``Mesh.__post_init__`` copies and freezes every array it is given, so
    # the value is set on this array *before* it is handed to ``replace``.
    positions[0, 0] = np.float32(1234.5678901)
    box = replace(bp.box(), positions=positions)
    doc.objects.append(bd.Obj(uid=bd.new_uid(), name="Far", mesh=box))

    obj_text, _ = objexport.claydoc_to_obj(doc)
    written = float(next(
        line.split()[1] for line in obj_text.splitlines()
        if line.startswith("v ") and line.split()[1].startswith("1234.")
    ))
    assert abs(written - float(box.positions[0, 0])) < 1e-4, (
        "the six-significant-digit writer this replaced rounds to the "
        f"nearest millimetre or coarser; got {written!r}"
    )


def test_a_material_with_a_texture_names_it_with_map_kd_not_a_silent_drop() -> None:
    from realmspinner.kernels.geom3d import gltf

    doc = bd.ClayDoc(materials=[gltf.Material(name="Tex", base_color=(2, 2, b"\x00" * 16))])
    doc.objects.append(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box(), material=0))
    obj_text, mtl_text = objexport.claydoc_to_obj(doc)
    assert "map_Kd model_0.png" in mtl_text.splitlines()


def test_ns_and_roughness_are_exact_inverses_over_the_unit_interval() -> None:
    for r in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
        ns = objimport.ns_from_roughness(r)
        assert objimport.roughness_from_ns(ns) == pytest.approx(r, abs=1e-9)


# --- textures: map_Kd and the PNG sidecar -------------------------------------


def _textured_doc() -> bd.ClayDoc:
    doc = bd.ClayDoc(materials=[bd.default_material("Plain"), bd.default_material("Painted")])
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
    doc.paint_faces(doc.objects[0].uid, [0, 1], 1)
    pixels = bytes([10, 20, 30, 255, 200, 100, 50, 255, 0, 0, 0, 255, 255, 255, 255, 128])
    doc.materials[1] = replace(doc.materials[1], base_color=(2, 2, pixels), nearest=True)
    return doc


def test_a_textured_material_names_its_png_in_the_mtl_and_an_untextured_one_does_not() -> None:
    _, mtl_text = objexport.claydoc_to_obj(_textured_doc(), name="chair")

    blocks = {b.splitlines()[0]: b for b in mtl_text.split("newmtl ")[1:]}
    assert "map_Kd chair_1.png" in blocks["Material_1"].splitlines()
    assert "map_Kd" not in blocks["Material_0"]


def test_the_png_sidecar_carries_the_textures_pixels() -> None:
    import io

    from PIL import Image

    doc = _textured_doc()

    pngs = objexport.claydoc_textures(doc)

    assert list(pngs) == [1]
    image = Image.open(io.BytesIO(pngs[1]))
    assert image.size == (2, 2)
    assert image.convert("RGBA").tobytes() == doc.materials[1].base_color[2]


def test_a_document_with_no_texture_writes_no_png_and_no_map_kd() -> None:
    doc, _ = bd.ClayDoc(), None
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))

    _, mtl_text = objexport.claydoc_to_obj(doc)

    assert objexport.claydoc_textures(doc) == {}
    assert "map_Kd" not in mtl_text


def test_a_texture_no_visible_face_uses_is_not_written() -> None:
    doc = _textured_doc()
    doc.paint_faces(doc.objects[0].uid, [0, 1], 0)  # nothing is on the textured slot now

    _, mtl_text = objexport.claydoc_to_obj(doc)

    assert objexport.claydoc_textures(doc) == {}
    assert "map_Kd" not in mtl_text


def test_the_mtl_texture_name_is_a_bare_file_name() -> None:
    _, mtl_text = objexport.claydoc_to_obj(_textured_doc(), name="a\nb")

    names = [ln.split(" ", 1)[1] for ln in mtl_text.splitlines() if ln.startswith("map_Kd")]
    assert names == [objexport.texture_name("a\nb", 1)]
    assert "\n" not in names[0] and "/" not in names[0]
