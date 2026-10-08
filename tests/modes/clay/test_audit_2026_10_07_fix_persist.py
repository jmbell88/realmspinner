"""The 2026-10-07 Clay audit, the ``.rblk`` read side, the format-3 migration and
the agent preview scratch.

clay-24, clay-25, clay-26, clay-53, clay-54, clay-55, clay-69 (read side),
clay-70 and clay-72. Each test's name is its claim; each was run against the
unfixed code first.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

import realmspinner
from realmspinner.core.safeio import npyguard
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import glbimport, legacy, scratch, separate
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import serialize as ser

SRC = Path(realmspinner.__file__).parent


def _rewrite(data: bytes, edit_scene) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as src, zipfile.ZipFile(out, "w") as dst:
        for name in src.namelist():
            raw = src.read(name)
            if name == ser.SCENE:
                scene = json.loads(raw)
                edit_scene(scene)
                raw = json.dumps(scene).encode()
            dst.writestr(name, raw)
    return out.getvalue()


def _box_doc() -> bd.ClayDoc:
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="B", mesh=bp.box()))
    return doc


def _one_triangle_many_vertices(count: int) -> bm.Mesh:
    pos = np.zeros((count, 3), "f4")
    pos[1] = (1, 0, 0)
    pos[2] = (0, 1, 0)
    return bm.Mesh(
        positions=pos,
        loops=np.array([0, 1, 2], "i4"),
        starts=np.array([0, 3], "i4"),
        material=np.zeros(1, "i4"),
        smooth=np.zeros(1, bool),
        uv=None,
    )


# --- clay-25 ------------------------------------------------------------------


def test_a_legacy_array_over_a_mesh_with_many_unused_vertices_is_dropped_not_multiplied() -> None:
    # One triangle passes every triangle ceiling at any count; 35,000 vertices
    # times the 200 an array may ask for is 7,000,000, past MAX_VERTICES.
    mesh = _one_triangle_many_vertices(35_000)
    assert glbimport.MAX_VERTICES < 35_000 * 200
    for kind in ("array", "radial-array"):
        doc = bd.ClayDoc(objects=[bd.Obj(uid=1, name="o", mesh=mesh)])
        mod = legacy._parse_modifier({"id": 1, "kind": kind, "params": {"count": 200}})
        notes = legacy.migrate(doc, {1: legacy.LegacyFields(modifiers=(mod,))})
        assert len(doc.objects[0].mesh.positions) == 35_000, kind
        assert any("vertices" in n and "dropped" in n for n in notes), (kind, notes)


def test_the_array_kernels_refuse_vertex_growth_before_building_a_copy() -> None:
    mesh = _one_triangle_many_vertices(35_000)
    with pytest.raises(el.OpError, match="vertices"):
        legacy.array_linear(mesh, {"count": 200})
    with pytest.raises(el.OpError, match="vertices"):
        legacy.array_radial(mesh, {"count": 200})


def test_the_writer_refuses_an_array_the_reader_would_refuse_to_reopen(monkeypatch) -> None:
    doc = _box_doc()
    # The reader's npyguard refuses one array over MAX_ARRAY_BYTES; the writer
    # had no such check, so it wrote a file that could never be opened again.
    monkeypatch.setattr(npyguard, "MAX_ARRAY_BYTES", 64)
    with pytest.raises(ValueError, match="B"):
        ser.snapshot_bytes(ser.snapshot(doc))
    with pytest.raises(ValueError):
        ser.read_rblk(_unguarded_bytes(doc))


def _unguarded_bytes(doc: bd.ClayDoc) -> bytes:
    # The bytes the unfixed writer produced: the same archive, no refusal.
    snap = ser.snapshot(doc)
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(ser.SCENE, snap.scene)
        for uid, mesh in snap.meshes:
            arrays = {name: getattr(mesh, name) for name in ser._MESH_FIELDS}
            zf.writestr(f"{ser.MESH_DIR}/{uid}.npz", ser._npz_bytes(arrays))
    return out.getvalue()


# --- clay-26 ------------------------------------------------------------------


def test_dropping_a_collider_under_a_denormal_scale_ancestor_keeps_the_childs_trs_finite() -> None:
    def mk(uid, name, parent=None, scale=(1.0, 1.0, 1.0)):
        return bd.Obj(uid=uid, name=name, mesh=bp.box(), scale=scale, parent=parent)

    ancestor = mk(1, "ancestor", scale=(1e-320,) * 3)
    collider = mk(2, "collider", parent=1)
    child = mk(3, "child", parent=2)
    child.translation = np.array([1.0, 2.0, 3.0])
    doc = bd.ClayDoc(objects=[ancestor, collider, child])
    legacy.migrate(doc, {2: legacy.LegacyFields(role="collider")})
    kept = doc.by_uid(3)
    assert kept.parent == 1
    assert all(np.isfinite(v).all() for v in kept.trs())


# --- clay-53 ------------------------------------------------------------------


def test_a_legacy_stack_past_the_modifier_budget_drops_the_excess_and_names_it() -> None:
    mesh = bp.grid((2.0, 2.0), 40)
    raw = [
        {"id": i, "kind": "smooth", "params": {"factor": 0.5, "iterations": 50}}
        for i in range(1, 101)
    ]
    fields = legacy.read_fields({"modifiers": raw})
    assert len(fields.modifiers) == 100
    doc = bd.ClayDoc(objects=[bd.Obj(uid=1, name="g", mesh=mesh)])
    notes = legacy.migrate(doc, {1: fields})
    dropped = [n for n in notes if "Smooth" in n and "dropped" in n]
    assert dropped, notes
    assert len(dropped) < 100  # the early ones still ran: the budget is a cap, not a ban


def test_the_v3_kernels_fixture_still_converts_without_a_budget_notice() -> None:
    data = (Path(__file__).parent / "data" / "v3_kernels.rblk").read_bytes()
    doc = ser.read_rblk(data)
    assert not any("work" in n for n in doc.notices), doc.notices


# --- clay-54 ------------------------------------------------------------------


def test_a_modifier_parameter_of_nan_is_dropped_by_name_not_clamped_to_its_maximum() -> None:
    fields = legacy.read_fields(
        {"modifiers": [{"id": 1, "kind": "weld", "params": {"distance": float("nan")}}]}
    )
    assert fields.modifiers == ()
    assert len(fields.unreadable) == 1
    assert "weld" in fields.unreadable[0] and "distance" in fields.unreadable[0]


# --- clay-24 ------------------------------------------------------------------


def test_transplant_places_a_separated_objects_pieces_where_the_preview_showed_them() -> None:
    two = legacy._concat([bp.box(), bp.box()])
    shift = np.repeat([[0, 0, 0], [5, 0, 0]], 8, axis=0).astype("f4")
    two = bm.Mesh(
        positions=two.positions + shift,
        loops=two.loops,
        starts=two.starts,
        material=two.material,
        smooth=two.smooth,
        uv=two.uv,
    )
    doc = bd.ClayDoc(
        objects=[
            bd.Obj(uid=bd.new_uid(), name="a", mesh=bp.box()),
            bd.Obj(uid=bd.new_uid(), name="b", mesh=two),
            bd.Obj(uid=bd.new_uid(), name="c", mesh=bp.box()),
        ]
    )
    sc = scratch.clone(doc)
    sc.separate(doc.objects[1].uid, separate.by_loose_parts(sc.by_uid(doc.objects[1].uid).mesh))
    preview = [o.name for o in sc.objects]
    assert preview[0] == "a" and preview[-1] == "c" and len(preview) == 4
    d = scratch.diff(doc, sc)
    assert d.order_changed
    scratch.transplant(doc, sc, d)
    assert [o.name for o in doc.objects] == preview


# --- clay-55 ------------------------------------------------------------------


def _pair() -> bd.ClayDoc:
    return bd.ClayDoc(
        objects=[
            bd.Obj(uid=bd.new_uid(), name="a", mesh=bp.box()),
            bd.Obj(uid=bd.new_uid(), name="b", mesh=bp.box()),
        ]
    )


def test_transplant_tolerates_a_uid_that_vanished_since_the_preview() -> None:
    doc = _pair()
    sc = scratch.clone(doc)
    uid = doc.objects[0].uid
    diff = scratch.PreviewDiff(parent_changed={uid: 987_654_321})
    scratch.transplant(doc, sc, diff)  # raised KeyError before
    diff = scratch.PreviewDiff(mesh_changed={uid})
    sc.by_uid(uid).mesh = bp.box((2, 2, 2))
    doc.remove_object(uid)
    scratch.transplant(doc, sc, diff)


def test_transplant_that_pushes_nothing_leaves_the_previous_steps_label_alone() -> None:
    doc = _pair()
    doc.set_props(doc.objects[0].uid, name="renamed by the user")
    label = doc.history.top.label
    sc = scratch.clone(doc)
    # The scratch's transform equals the document's, so ``set_transform`` is a
    # no-op and nothing is pushed.
    diff = scratch.PreviewDiff(transform_changed={doc.objects[1].uid})
    result = scratch.transplant(doc, sc, diff)
    assert result is False
    assert doc.history.top.label == label


# --- clay-69, the read side -----------------------------------------------------


def test_a_non_unit_rotation_is_normalised_on_read() -> None:
    data = ser.rblk_bytes(_box_doc())

    def edit(scene):
        scene["objects"][0]["rotation"] = [0.0, 0.5, 0.0, 0.5]

    doc = ser.read_rblk(_rewrite(data, edit))
    rotation = doc.objects[0].rotation
    assert float(np.linalg.norm(rotation)) == pytest.approx(1.0)
    assert np.allclose(rotation, [0.0, 2**-0.5, 0.0, 2**-0.5])


def test_a_huge_non_unit_rotation_is_normalised_on_read_not_overflowed() -> None:
    data = ser.rblk_bytes(_box_doc())

    def edit(scene):
        scene["objects"][0]["rotation"] = [0.0, 1e200, 0.0, 1e200]

    doc = ser.read_rblk(_rewrite(data, edit))
    assert float(np.linalg.norm(doc.objects[0].rotation)) == pytest.approx(1.0)


def test_a_unit_rotation_round_trips_untouched() -> None:
    doc = _box_doc()
    q = np.array([0.1, 0.2, 0.3, 0.0])
    q = q / np.linalg.norm(q)
    doc.objects[0].rotation = q
    back = ser.read_rblk(ser.rblk_bytes(doc))
    assert np.array_equal(back.objects[0].rotation, q)


# --- clay-70 --------------------------------------------------------------------


def test_read_rblk_refuses_a_default_material_past_int32() -> None:
    data = ser.rblk_bytes(_box_doc())

    def edit(scene):
        scene["objects"][0]["material"] = 5_000_000_000

    with pytest.raises(ValueError, match="material"):
        ser.read_rblk(_rewrite(data, edit))


# --- clay-72 --------------------------------------------------------------------


def test_serialize_names_the_snapshot_invariant_by_its_words_not_a_number() -> None:
    text = (SRC / "kernels" / "mesh" / "serialize.py").read_text(encoding="utf-8")
    assert "INVARIANTS 312" not in text
    assert "A mesh is an immutable CSR array" in text


# --- the object ceiling (document.add_objects refuses past it) ---------------------


def _document_at_the_object_ceiling() -> bd.ClayDoc:
    objects = [
        bd.Obj(uid=bd.new_uid(), name=f"o{i}", mesh=bp.box())
        for i in range(glbimport.MAX_OBJECTS)
    ]
    return bd.ClayDoc(objects=objects)  # the constructor does not charge the ceiling


def test_transplant_that_would_pass_the_object_ceiling_changes_nothing_and_returns_false() -> None:
    doc = _document_at_the_object_ceiling()
    sc = scratch.clone(doc)
    # The scratch is built past the ceiling by hand: its own doors refuse too.
    sc.objects.append(bd.Obj(uid=bd.new_uid(), name="extra", mesh=bp.box()))
    diff = scratch.diff(doc, sc)
    before = [o.uid for o in doc.objects]
    head = doc.history.head
    assert scratch.transplant(doc, sc, diff) is False
    assert [o.uid for o in doc.objects] == before
    assert doc.history.head == head


def test_transplant_refused_after_a_removal_rolls_the_removal_back_too() -> None:
    doc = _document_at_the_object_ceiling()
    sc = scratch.clone(doc)
    gone = sc.objects.pop(0)
    sc.objects.append(bd.Obj(uid=bd.new_uid(), name="x1", mesh=bp.box()))
    sc.objects.append(bd.Obj(uid=bd.new_uid(), name="x2", mesh=bp.box()))
    diff = scratch.diff(doc, sc)
    assert diff.removed == {gone.uid}
    before = [o.uid for o in doc.objects]
    head = doc.history.head
    assert scratch.transplant(doc, sc, diff) is False
    assert [o.uid for o in doc.objects] == before  # the removal was undone
    assert doc.history.head == head
    assert not doc.history.can_redo
