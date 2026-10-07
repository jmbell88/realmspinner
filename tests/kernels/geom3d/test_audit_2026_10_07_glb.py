"""The 2026-10-07 Clay audit's GLB-import findings (clay-01 import half, 17, 19,
70, 71, 72): what a hand-authored or third-party GLB can put in that Clay's own
writer never would."""

from __future__ import annotations

import copy
import inspect
from dataclasses import replace

import numpy as np
import pytest

from realmspinner.kernels.geom3d import glbio, glbwrite, gltf
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import glbimport, ops, serialize
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh.elements import OpError


def _box_glb() -> bytes:
    doc = bd.ClayDoc()
    doc.objects.append(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
    return glbwrite.write_glb(bd.to_model(doc))


def _edited(data: bytes, edit) -> bytes:  # type: ignore[no-untyped-def]
    header, doc, rest = glbio.split_glb(data)
    doc = copy.deepcopy(doc)
    edit(doc)
    return glbio.rebuild_glb(header, doc, rest)


def _node(**fields: object) -> bytes:
    return _edited(_box_glb(), lambda j: j["nodes"][0].update(fields))


# --- clay-01: the import half ----------------------------------------------


@pytest.mark.parametrize("scale", [[0, 0, 0], [1e-320] * 3])
def test_a_glb_node_with_zero_scale_imports_to_a_document_that_saves_and_reopens(
    scale: list[float],
) -> None:
    """A zero-scaled node is how game assets hide a part: the import keeps it
    (at a floor scale, geometry intact) rather than breaking the document, and
    the saved ``.rblk`` the reader would otherwise refuse reopens."""
    doc = glbimport.glb_to_claydoc(_node(scale=scale))
    obj = doc.objects[0]
    assert np.any(obj.scale)
    assert np.isfinite(obj.scale).all()
    # Geometry is not collapsed: the part can be scaled back up.
    assert len(obj.mesh.positions) == 8
    back = serialize.read_rblk(serialize.rblk_bytes(doc))
    assert len(back.objects) == 1
    assert np.any(back.objects[0].scale)


def test_a_glb_node_with_one_collapsed_axis_still_imports_and_reopens() -> None:
    doc = glbimport.glb_to_claydoc(_node(scale=[0, 1, 1]))
    back = serialize.read_rblk(serialize.rblk_bytes(doc))
    assert len(back.objects) == 1


# --- clay-17 ----------------------------------------------------------------


@pytest.mark.parametrize("bad", [5, ["a"], {"a": 1}, True, 2.5])
def test_a_non_string_node_or_material_name_is_not_carried_into_the_clay_document(
    bad: object,
) -> None:
    def edit(j: dict) -> None:
        j["nodes"][0]["name"] = bad
        j["materials"][0]["name"] = bad

    data = _edited(_box_glb(), edit)
    model = gltf.load(data)
    assert all(isinstance(n.name, str) for n in model.nodes)
    mats = [p.material for prims in model.meshes for p in prims if p.material is not None]
    assert mats and all(isinstance(m.name, str) for m in mats)

    doc = glbimport.glb_to_claydoc(data)
    obj = doc.objects[0]
    assert isinstance(obj.name, str)
    assert all(isinstance(m.name, str) for m in doc.materials)
    # What Duplicate does with the name: it used to raise on an int.
    assert isinstance(ops.next_name(obj.name, {o.name for o in doc.objects}), str)
    serialize.read_rblk(serialize.rblk_bytes(doc))


# --- clay-19 ----------------------------------------------------------------


@pytest.mark.parametrize(
    "fields",
    [
        {"scale": [1e200] * 3},
        {"translation": [1e39, 0, 0]},
        {"scale": [1e39] * 3, "rotation": [5, 5, 5, 5]},
    ],
)
def test_a_glb_node_transform_that_overflows_float32_is_refused_at_import(
    fields: dict,
) -> None:
    with pytest.raises(OpError, match="float32|too large|range"):
        glbimport.glb_to_claydoc(_node(**fields))


def test_a_large_but_representable_transform_still_imports() -> None:
    doc = glbimport.glb_to_claydoc(_node(scale=[1e6] * 3, translation=[1e7, 0, 0]))
    assert np.isfinite(doc.objects[0].mesh.positions).all()


# --- clay-70 ----------------------------------------------------------------


def test_a_glb_with_a_non_finite_uv_is_refused_by_name() -> None:
    mesh = bp.box()
    uv = mesh.uv.copy()
    uv[0] = (np.nan, np.inf)
    doc = bd.ClayDoc()
    doc.objects.append(bd.Obj(uid=bd.new_uid(), name="Box", mesh=replace(mesh, uv=uv)))
    data = glbwrite.write_glb(bd.to_model(doc))
    with pytest.raises(OpError, match="UV|uv"):
        glbimport.glb_to_claydoc(data)


# --- clay-71 ----------------------------------------------------------------


def test_a_non_string_required_extension_and_a_dict_trs_field_are_refused_by_name() -> None:
    for required in ([["x"]], [{}], [5]):
        data = _edited(_box_glb(), lambda j, r=required: j.__setitem__("extensionsRequired", r))
        with pytest.raises(ValueError, match="extensionsRequired"):
            gltf.load(data)
    for key in ("translation", "rotation", "scale"):
        data = _node(**{key: {"x": 1}})
        with pytest.raises(ValueError, match=key):
            gltf.load(data)
    data = _node(scale=["a", "b", "c"])
    with pytest.raises(ValueError, match="scale"):
        gltf.load(data)


# --- clay-72 ----------------------------------------------------------------


def test_glbwrite_docstring_does_not_claim_clay_paints_no_textures() -> None:
    doc = inspect.getdoc(glbwrite.write_glb) or ""
    assert "paints none" not in doc
    assert "paints no" not in doc
