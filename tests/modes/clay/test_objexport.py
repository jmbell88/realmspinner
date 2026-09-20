"""A Clay document out as OBJ + MTL, and the round trip back through
:mod:`warlock.kernels.mesh.objimport`."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from warlock.kernels.mesh import document as bd
from warlock.kernels.mesh import mesh as bm
from warlock.kernels.mesh import objexport, objimport
from warlock.kernels.mesh import primitives as bp
from warlock.kernels.mesh import uv as uv_mod


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

    planar = uv_mod.planar_unwrap(bp.plane(), axis=1)
    doc.objects.append(bd.Obj(uid=bd.new_uid(), name="Plane", mesh=planar, material=0))
    return doc


def _material_key(material) -> tuple:
    return (
        tuple(round(c, 4) for c in material.base_color_factor),
        round(material.roughness_factor, 4),
    )


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
        # number its own vertices internally. ``atol`` a shade looser than the
        # per-corner uv check below: "%.6g" gives six *significant* digits, and
        # a position near magnitude 1-3 (this fixture's own scale) can lose a
        # hair more than 1e-6 absolute to that rounding.
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


def test_a_material_with_a_texture_gets_a_comment_not_a_silent_drop() -> None:
    from warlock.kernels.geom3d import gltf

    doc = bd.ClayDoc(materials=[gltf.Material(name="Tex", base_color=(2, 2, b"\x00" * 16))])
    doc.objects.append(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box(), material=0))
    obj_text, mtl_text = objexport.claydoc_to_obj(doc)
    assert "texture" in mtl_text


def test_ns_and_roughness_are_exact_inverses_over_the_unit_interval() -> None:
    for r in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
        ns = objimport.ns_from_roughness(r)
        assert objimport.roughness_from_ns(ns) == pytest.approx(r, abs=1e-9)
