"""Regressions for the 2026-10-03 audit's mason-01 .. mason-06.

mason-07 (a material-override control) and mason-08 (template editing and an
Add-row in Properties) were built rather than left as design decisions, and
are covered in test_material_override_and_properties.py, not here.
"""

from __future__ import annotations

import numpy as np
import pytest

from realmspinner.kernels.geom3d import gltf
from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.studio import dialogs
from realmspinner.studio.modes.mason import mode as mason_mode
from realmspinner.studio.modes.mason.engine import document as md
from realmspinner.studio.modes.mason.engine import nodes as nd
from realmspinner.studio.modes.mason.engine import objout, pick
from realmspinner.studio.modes.mason.engine import refs as mrefs
from realmspinner.studio.modes.mason.engine import scene as msc
from realmspinner.studio.modes.mason.ui import view as mason_view

from .test_audit_2026_09_26_w1f8_modes_mason import _ExportCtx, _MixedSource
from .test_mason_view import RECT, _Ctx, _Source


def _group_with_child() -> tuple[md.MasonDoc, nd.GroupNode, nd.Node]:
    doc = md.MasonDoc()
    group = nd.GroupNode(uid=nd.new_uid(), name="G")
    doc.add_node(group)
    child = nd.MeshNode(uid=nd.new_uid(), name="C", ref=mrefs.primitive_ref("box", {}))
    doc.add_node(child, parent_uid=group.uid)
    other = nd.MeshNode(uid=nd.new_uid(), name="Other", ref=mrefs.primitive_ref("box", {}))
    doc.add_node(other)
    return doc, group, child


# --- mason-01 ----------------------------------------------------------------


def test_resolved_for_answers_about_an_expanded_prefab_instance_not_none():
    doc = md.MasonDoc()
    template = nd.MeshNode(uid=nd.new_uid(), name="T", ref=mrefs.primitive_ref("box", {}))
    doc.prefabs["Prop"] = template
    inst = nd.PrefabNode(uid=nd.new_uid(), template="Prop", translation=m3.vec3(1.0, 0.0, 0.0))
    doc.add_node(inst)

    found = msc.resolved_for(doc, inst.uid)

    assert found is not None
    assert found.node is inst
    assert found.world[0, 3] == pytest.approx(1.0)


# --- mason-02 ----------------------------------------------------------------


def test_isolate_of_a_nested_node_or_a_group_still_resolves_what_was_isolated():
    doc, group, child = _group_with_child()

    doc.isolate([child.uid])
    assert [p.node.uid for p in msc.resolve(doc)] == [child.uid]

    doc.show_all()
    doc.isolate([group.uid])
    assert [p.node.uid for p in msc.resolve(doc)] == [child.uid]


# --- mason-03 ----------------------------------------------------------------


def test_obj_vt_v_is_flipped_from_gltf_top_left_to_obj_bottom_left():
    doc = md.MasonDoc()
    doc.add_node(
        nd.MeshNode(uid=nd.new_uid(), name="Tri", ref=mrefs.primitive_ref("box", {}))
    )

    class Src:
        rev = 0

        def primitives(self, ref):
            return [
                gltf.Primitive(
                    positions=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="f4"),
                    indices=np.array([0, 1, 2], dtype="u4"),
                    uvs=np.array([[0, 0], [1, 0], [0, 1]], dtype="f4"),
                )
            ]

        def box(self, ref):
            return self.primitives(ref)[0].box()

    text = objout.obj_export(doc, Src()).files[objout.OBJ].decode("utf-8")
    vts = [ln.split()[1:] for ln in text.splitlines() if ln.startswith("vt ")]

    assert [[float(x) for x in vt] for vt in vts] == [[0, 1], [1, 1], [0, 0]]


# --- mason-04 ----------------------------------------------------------------


def test_a_typed_quaternion_component_is_normalized_before_it_reaches_the_node():
    doc = md.MasonDoc()
    node = nd.MeshNode(uid=nd.new_uid(), name="N", ref=mrefs.primitive_ref("box", {}))
    doc.add_node(node)

    doc.set_transform(node.uid, rotation=[0.5, 0.0, 0.0, 1.0])

    assert float(np.linalg.norm(node.rotation)) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        doc.set_transform(node.uid, rotation=[0.0, 0.0, 0.0, 0.0])


# --- mason-05 ----------------------------------------------------------------


def test_export_glb_with_a_permanently_missing_ref_writes_and_names_it_in_the_manifest(
    tmp_path, monkeypatch
):
    from realmspinner.studio.modes.mason import assets as mason_assets

    monkeypatch.setattr(mason_assets, "ensure", lambda ctx: _MixedSource())
    out_path = tmp_path / "scene.glb"
    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: str(out_path))
    ctx = _ExportCtx(None)
    doc = md.MasonDoc()
    doc.add_node(
        nd.MeshNode(
            uid=nd.new_uid(), name="Crate", ref=mrefs.LibraryRef(job_id="present", name="Crate")
        )
    )
    ghost = mrefs.LibraryRef(job_id="missing", name="Ghost")
    doc.add_node(nd.MeshNode(uid=nd.new_uid(), name="Ghost", ref=ghost))
    doc.missing.add(mrefs.ref_key(ghost))
    tab = mason_mode.adopt(ctx, doc, title="Scene")

    mason_mode.export_glb(ctx, tab)

    assert out_path.exists()
    assert "Ghost" in (tmp_path / "scene.json").read_text(encoding="utf-8")


def test_export_glb_checks_pending_refs_before_the_file_picker_opens(monkeypatch):
    from realmspinner.service.errors import NotReady
    from realmspinner.studio.modes.mason import assets as mason_assets

    from .test_audit_2026_09_26_w1f8_modes_mason import _mixed_scene

    monkeypatch.setattr(mason_assets, "ensure", lambda ctx: _MixedSource())
    opened: list[bool] = []
    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: opened.append(True))
    ctx = _ExportCtx(None)
    tab = mason_mode.adopt(ctx, _mixed_scene(), title="Scene")

    with pytest.raises(NotReady):
        mason_mode.export_glb(ctx, tab)

    assert opened == []


# --- mason-06 ----------------------------------------------------------------


def test_xray_lets_a_click_pick_the_node_behind_a_surface(gl):
    view = mason_view.MasonView(gl, _Ctx())
    try:
        view._rect = RECT
        view.camera.target = m3.vec3(0.0, 0.0, 0.0)
        view.camera.update(1.0)
        centre = (RECT[2] / 2.0, RECT[3] / 2.0)
        origin, direction = view._ray(centre)
        doc = md.MasonDoc()
        ref = mrefs.primitive_ref("box", {})
        near = nd.MeshNode(
            uid=nd.new_uid(), name="near", ref=ref,
            translation=_on_ray(origin, direction, 1.0),
        )
        far = nd.MeshNode(
            uid=nd.new_uid(), name="far", ref=ref,
            translation=_on_ray(origin, direction, -2.0),
        )
        doc.add_node(near)
        doc.add_node(far)
        doc.select([near.uid], active=near.uid)
        source = _Source()

        assert view.pick(doc, source, centre).owner == near.uid
        assert view.pick(doc, source, centre, through=True).owner == far.uid
        # nothing unselected behind: the nearest surface is still what a click gets
        doc.select([near.uid, far.uid], active=near.uid)
        assert view.pick(doc, source, centre, through=True).owner == near.uid
    finally:
        view.release()


def test_ray_scene_exclude_passes_through_listed_owners():
    placed = msc.resolve(_two_in_line())
    hit = pick.ray_scene(
        placed, _Source(), np.array([0.0, 0.0, 10.0]), np.array([0.0, 0.0, -1.0])
    )
    assert hit is not None
    behind = pick.ray_scene(
        placed,
        _Source(),
        np.array([0.0, 0.0, 10.0]),
        np.array([0.0, 0.0, -1.0]),
        exclude={hit.owner},
    )
    assert behind is not None and behind.owner != hit.owner


def _on_ray(origin, direction, plane_z: float):
    """A translation that puts the fake source's front quad (z=+0.5 local) on
    the ray at world z=plane_z."""
    t = (plane_z - float(origin[2])) / float(direction[2])
    p = origin + direction * t
    # Offset off the quad's diagonal, where a ray can slip between its two triangles.
    return np.array([p[0] + 0.2, p[1] + 0.1, plane_z - 0.5], dtype="f8")


def _two_in_line() -> md.MasonDoc:
    doc = md.MasonDoc()
    ref = mrefs.primitive_ref("box", {})
    for z in (0.0, -4.0):
        doc.add_node(
            nd.MeshNode(uid=nd.new_uid(), name=f"n{z}", ref=ref, translation=m3.vec3(0.0, 0.0, z))
        )
    return doc

