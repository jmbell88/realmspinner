"""The 2026-10-07 audit's document-layer findings: clay-04, 08, 09, 10, 11, 40, 42, 47, 56.

Each test name is the claim; each one failed against the code it was written
for. The shared fixture is a tiny hand-built hierarchy, because every finding
here is about what the document does to objects *around* the one it was asked
to change.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import regen, selection

_TINY = 1e-320  # finite and nonzero: past np.linalg.inv's singular test, inverts to inf


def _add(doc: bd.ClayDoc, name: str, parent: int | None = None, **kwargs: object) -> bd.Obj:
    obj = bd.Obj(
        uid=bd.new_uid(),
        name=name,
        mesh=bp.box(),
        parent=parent,
        **kwargs,  # type: ignore[arg-type]
    )
    doc.add_object(obj)
    return obj


def _finite(obj: bd.Obj) -> bool:
    return all(np.isfinite(np.asarray(v)).all() for v in obj.trs())


def _under_a_denormal_grandparent() -> tuple[bd.ClayDoc, bd.Obj, bd.Obj, bd.Obj]:
    """G (denormal scale) > P > C: removing P re-expresses C under G."""
    doc = bd.ClayDoc()
    g = _add(doc, "G")
    doc.set_transform(g.uid, scale=(_TINY,) * 3)
    p = _add(doc, "P", parent=g.uid)
    c = _add(doc, "C", parent=p.uid)
    return doc, g, p, c


# --- clay-04 -------------------------------------------------------------------


def test_remove_object_refuses_a_child_replacement_that_is_not_finite() -> None:
    doc, _g, p, c = _under_a_denormal_grandparent()
    before = [np.array(v, copy=True) for v in c.trs()]
    steps = len(doc.history)
    with pytest.raises(el.OpError, match="not finite"), np.errstate(all="ignore"):
        doc.remove_object(p.uid)
    assert _finite(c)
    assert all(np.array_equal(a, b) for a, b in zip(before, c.trs(), strict=True))
    assert c.parent == p.uid
    assert doc.by_uid(p.uid) is p
    assert len(doc.history) == steps


def test_remove_object_changes_nothing_when_a_later_child_is_the_one_refused() -> None:
    """All children are computed before the first is assigned."""
    doc, _g, p, c = _under_a_denormal_grandparent()
    sibling = _add(doc, "S", parent=p.uid)  # also under P; both are refused here
    with pytest.raises(el.OpError), np.errstate(all="ignore"):
        doc.remove_object(p.uid)
    assert c.parent == p.uid and sibling.parent == p.uid


def test_join_objects_refuses_a_child_replacement_that_is_not_finite() -> None:
    doc = bd.ClayDoc()
    g = _add(doc, "G")
    doc.set_transform(g.uid, scale=(_TINY,) * 3)
    target = _add(doc, "T")
    p = _add(doc, "P", parent=g.uid)
    c = _add(doc, "C", parent=p.uid)
    mesh_before = target.mesh
    steps = len(doc.history)
    with pytest.raises(el.OpError, match="not finite"), np.errstate(all="ignore"):
        doc.join_objects(target.uid, bp.box((2, 2, 2)), [p.uid])
    assert target.mesh is mesh_before
    assert _finite(c) and c.parent == p.uid
    assert len(doc.history) == steps


def test_separate_refuses_a_child_replacement_that_is_not_finite() -> None:
    doc, _g, p, c = _under_a_denormal_grandparent()
    quad = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 5, 5], [6, 5, 5], [5, 6, 5]], "f4")
    pieces = [
        bm.from_faces(quad[:3], [[0, 1, 2]]),
        bm.from_faces(quad[3:], [[0, 1, 2]]),
    ]
    count = len(doc.objects)
    with pytest.raises(el.OpError, match="not finite"), np.errstate(all="ignore"):
        doc.separate(p.uid, pieces)
    assert _finite(c) and c.parent == p.uid
    assert len(doc.objects) == count


def test_set_origin_refuses_a_child_replacement_that_is_not_finite() -> None:
    doc = bd.ClayDoc()
    p = _add(doc, "P")
    c = _add(doc, "C", parent=p.uid, translation=(1.0, 0.0, 0.0))
    doc.set_transform(p.uid, scale=(_TINY,) * 3)
    mesh_before, trs_before = p.mesh, [np.array(v, copy=True) for v in p.trs()]
    with pytest.raises((el.OpError, ValueError)), np.errstate(all="ignore"):
        doc.set_origin(p.uid, (1.0, 0.0, 0.0))
    assert _finite(c) and _finite(p)
    assert p.mesh is mesh_before
    assert all(np.array_equal(a, b) for a, b in zip(trs_before, p.trs(), strict=True))


# --- clay-08 -------------------------------------------------------------------


def _painted_and_unwrapped(mesh: bm.Mesh) -> bm.Mesh:
    material = np.zeros(bm.face_count(mesh), dtype="i4")
    material[: max(1, len(material) // 3)] = 1
    uv = np.array(mesh.uv, copy=True)
    uv[:, 0] += 0.25  # a hand-moved layout, not the generator's own
    return replace(mesh, material=material, uv=uv)


def test_a_two_key_set_params_keeps_paint_and_uv_when_the_face_count_is_unchanged() -> None:
    old = _painted_and_unwrapped(bp.cylinder(radius=0.5, height=1.0, segments=16))
    rebuilt = bp.cylinder(radius=0.6, height=2.0, segments=16)
    assert bm.face_count(old) == bm.face_count(rebuilt)

    kept = regen.carry_over(old, rebuilt, material=0, changed_keys={"radius", "height"})

    assert np.array_equal(kept.material, old.material)
    assert np.array_equal(kept.uv, old.uv)
    assert np.array_equal(kept.smooth, old.smooth)


def _two_key_changes() -> list[tuple[str, dict, dict]]:
    """For every Clay generator, a pair of parameter sets differing in 2+ keys."""
    return [
        ("box", {"size": (1, 1, 1)}, {"size": (2, 1, 3)}),
        ("plane", {"size": (1, 1)}, {"size": (3, 2)}),
        ("grid", {"size": (1, 1), "divisions": 4}, {"size": (3, 2), "divisions": 4}),
        (
            "cylinder",
            {"radius": 0.5, "height": 1.0, "segments": 16},
            {"radius": 0.6, "height": 2.0, "segments": 16},
        ),
        (
            "cone",
            {"radius": 0.5, "height": 1.0, "segments": 16},
            {"radius": 0.6, "height": 2.0, "segments": 16},
        ),
        (
            "uv_sphere",
            {"radius": 0.5, "segments": 16, "rings": 8},
            {"radius": 0.7, "segments": 16, "rings": 8},
        ),
        ("icosphere", {"radius": 0.5, "subdivisions": 2}, {"radius": 0.9, "subdivisions": 2}),
        (
            "torus",
            {"radius": 0.35, "tube": 0.15, "segments": 24, "sides": 16},
            {"radius": 0.5, "tube": 0.2, "segments": 24, "sides": 16},
        ),
        (
            "capsule",
            {"radius": 0.25, "height": 0.5, "segments": 16, "rings": 4},
            {"radius": 0.3, "height": 1.0, "segments": 16, "rings": 4},
        ),
        (
            "wedge",
            {"width": 1.0, "height": 1.0, "depth": 1.0},
            {"width": 2.0, "height": 1.5, "depth": 1.0},
        ),
        (
            "ramp",
            {"width": 1.0, "length": 1.0, "height": 0.5},
            {"width": 2.0, "length": 1.5, "height": 0.5},
        ),
        (
            "rounded_box",
            {"size": (1, 1, 1), "radius": 0.1, "segments": 4},
            {"size": (2, 1, 1), "radius": 0.2, "segments": 4},
        ),
        (
            "stairs",
            {"steps": 4, "width": 1.0, "total_height": 1.0, "total_depth": 1.0},
            {"steps": 4, "width": 2.0, "total_height": 1.5, "total_depth": 1.0},
        ),
        (
            "wall",
            {"length": 2.0, "height": 1.0, "thickness": 0.2},
            {"length": 3.0, "height": 2.0, "thickness": 0.2},
        ),
        (
            "doorway",
            {
                "wall_length": 3.0,
                "wall_height": 2.5,
                "wall_thickness": 0.2,
                "opening_width": 0.9,
                "opening_height": 2.0,
                "opening_offset": 0.0,
            },
            {
                "wall_length": 4.0,
                "wall_height": 3.0,
                "wall_thickness": 0.2,
                "opening_width": 0.9,
                "opening_height": 2.0,
                "opening_offset": 0.0,
            },
        ),
    ]


def test_every_clay_generator_keeps_paint_and_uv_through_a_size_only_multi_key_change() -> None:
    covered = set()
    for name, before, after in _two_key_changes():
        covered.add(name)
        build = bp.GENERATORS[name][1]
        old = _painted_and_unwrapped(build(**before))
        rebuilt = build(**after)
        assert bm.face_count(old) == bm.face_count(rebuilt), name
        changed = {k for k in before if before[k] != after[k]}
        assert len(changed) >= 1
        keys = changed | {"_second"}  # the rule only needs "more than one key named"
        kept = regen.carry_over(old, rebuilt, material=0, changed_keys=keys)
        assert np.array_equal(kept.material, old.material), name
        assert np.array_equal(kept.uv, old.uv), name
    assert covered == bp.CLAY_GENERATOR_NAMES


def test_swapping_two_counts_that_keep_the_face_total_still_forfeits_the_carry() -> None:
    """The only shapes whose same-count rebuilds reorder faces (swept for 2026-10-07)."""
    for name, a, b in (
        ("torus", {"segments": 8, "sides": 4}, {"segments": 4, "sides": 8}),
        ("uv_sphere", {"segments": 8, "rings": 4}, {"segments": 4, "rings": 8}),
    ):
        build = bp.GENERATORS[name][1]
        old = _painted_and_unwrapped(build(**a))
        rebuilt = build(**b)
        assert bm.face_count(old) == bm.face_count(rebuilt), name
        assert not regen.same_faces(old, rebuilt), name
        carried = regen.carry_over(old, rebuilt, material=0, changed_keys=set(a))
        assert set(np.unique(carried.material).tolist()) == {0}, name


def test_a_two_key_set_params_keeps_the_element_selection_at_an_unchanged_face_count() -> None:
    doc = bd.ClayDoc()
    obj = bd.Obj(
        uid=bd.new_uid(),
        name="Cyl",
        mesh=bp.cylinder(),
        generator="cylinder",
        params={"radius": 0.5, "height": 1.0, "segments": 16},
    )
    doc.add_object(obj)
    doc.set_element_mode("face")
    doc.set_element_sel(obj.uid, el.ElementSel(faces=np.array([0, 1, 2], dtype="i4")))

    params = {"radius": 0.6, "height": 2.0, "segments": 16}
    rebuilt = bp.cylinder(**params)
    doc.set_generator_params(obj.uid, params, rebuilt, was={"params": {
        "radius": 0.5, "height": 1.0, "segments": 16
    }})

    assert doc.element_sel_of(obj.uid).faces.tolist() == [0, 1, 2]


# --- clay-09 / clay-10 ---------------------------------------------------------


def test_group_keeps_a_selected_child_under_its_selected_parent() -> None:
    doc = bd.ClayDoc()
    p = _add(doc, "P", translation=(2.0, 0.0, 0.0))
    c = _add(doc, "C", parent=p.uid, translation=(1.0, 0.0, 0.0))
    other = _add(doc, "O")
    world_c = doc.world_matrix(c.uid).copy()

    g = doc.group([p.uid, c.uid, other.uid])

    assert p.parent == g.uid and other.parent == g.uid
    assert c.parent == p.uid
    assert np.allclose(doc.world_matrix(c.uid), world_c)


def test_group_closes_its_history_gesture_when_a_member_cannot_be_parented() -> None:
    doc = bd.ClayDoc()
    a = _add(doc, "A")
    b = _add(doc, "B")
    doc.set_transform(b.uid, scale=(1e300, 1e300, 1e300))
    count, steps = len(doc.objects), len(doc.history)

    with pytest.raises(el.OpError), np.errstate(all="ignore"):
        doc.group([a.uid, b.uid])

    assert doc.history._open_gestures == 0
    # Pre-checked: no empty was added and nothing was parented.
    assert len(doc.objects) == count and len(doc.history) == steps
    assert a.parent is None and b.parent is None


# --- clay-11 -------------------------------------------------------------------


def test_delete_selected_closes_its_history_gesture_when_remove_object_refuses() -> None:
    doc, _g, p, c = _under_a_denormal_grandparent()
    doc.select([p.uid])
    assert doc.history._open_gestures == 0

    with np.errstate(all="ignore"):
        refusals = selection.delete_selected(doc)

    assert doc.history._open_gestures == 0
    assert len(refusals) == 1 and "not finite" in refusals[0]
    assert doc.by_uid(p.uid) is p and c.parent == p.uid


def test_delete_selected_still_deletes_the_others_when_one_object_is_refused() -> None:
    doc, _g, p, _c = _under_a_denormal_grandparent()
    spare = _add(doc, "Spare")
    doc.select([p.uid, spare.uid])

    with np.errstate(all="ignore"):
        refusals = selection.delete_selected(doc)

    assert len(refusals) == 1
    assert spare.uid not in {o.uid for o in doc.objects}
    assert p.uid in {o.uid for o in doc.objects}
    assert doc.history._open_gestures == 0


# --- clay-40 -------------------------------------------------------------------


def _textured_slot(doc: bd.ClayDoc) -> int:
    index = doc.add_material()
    assert doc.add_texture(index)
    return index


def test_assign_material_of_a_textured_slot_to_a_uv_less_object_unwraps_it_in_the_same_step() -> (
    None
):
    doc = bd.ClayDoc()
    obj = _add(doc, "X")
    doc.set_mesh(obj.uid, replace(obj.mesh, uv=None), keep_generator=True)
    index = _textured_slot(doc)
    assert obj.mesh.uv is None
    steps = len(doc.history)

    assert doc.paint_faces(obj.uid, [0, 1], index)

    assert obj.mesh.uv is not None
    assert obj.mesh.material[:2].tolist() == [index, index]
    assert len(doc.history) == steps + 1
    doc.undo()
    assert obj.mesh.uv is None and obj.mesh.material[:2].tolist() == [0, 0]


def test_repainting_a_uv_less_object_with_a_textured_slot_unwraps_it_in_the_same_step() -> None:
    doc = bd.ClayDoc()
    obj = _add(doc, "X")
    doc.set_mesh(obj.uid, replace(obj.mesh, uv=None), keep_generator=True)
    index = _textured_slot(doc)
    steps = len(doc.history)

    assert doc.repaint_object(obj.uid, index)

    assert obj.mesh.uv is not None
    assert len(doc.history) == steps + 1


def test_painting_a_flat_colour_slot_or_an_already_unwrapped_object_leaves_the_uvs_alone() -> None:
    doc = bd.ClayDoc()
    flat = _add(doc, "Flat")
    doc.set_mesh(flat.uid, replace(flat.mesh, uv=None), keep_generator=True)
    plain = doc.add_material()
    doc.paint_faces(flat.uid, [0], plain)
    assert flat.mesh.uv is None  # no texture on that slot: nothing to sample

    authored = _add(doc, "Authored")
    layout = np.array(authored.mesh.uv, copy=True)
    layout[:, 1] *= 0.5
    doc.set_mesh(authored.uid, replace(authored.mesh, uv=layout), keep_generator=True)
    index = _textured_slot(doc)
    doc.paint_faces(authored.uid, [0], index)
    assert np.array_equal(authored.mesh.uv, layout)


# --- clay-42 -------------------------------------------------------------------


def test_an_undone_add_forgets_its_mesh_stamp() -> None:
    doc = bd.ClayDoc()
    obj = _add(doc, "X")
    doc.mesh_stamp(obj.uid)
    assert obj.uid in doc._mesh_stamps

    assert doc.undo()  # takes the object back out

    assert obj.uid not in doc._mesh_stamps


def test_a_redone_remove_forgets_its_mesh_stamp() -> None:
    doc = bd.ClayDoc()
    obj = _add(doc, "X")
    doc.remove_object(obj.uid)
    assert doc.undo()  # the object is back
    doc.mesh_stamp(obj.uid)
    assert obj.uid in doc._mesh_stamps

    assert doc.redo()  # and gone again

    assert obj.uid not in doc._mesh_stamps


# --- clay-47 / clay-56 ---------------------------------------------------------

_DELETED = (
    "agent_clay",
    "clay_props",
    "add_assembly",
    "LIMB_RINGS",
    "figure preset",
    "Bézier editor",
    "clay/mesh.py",
    "_forget_manifold",
)


def test_no_docstring_in_kernels_mesh_names_a_deleted_module() -> None:
    """Scoped to the files the 2026-10-07 document fixer owns; the other kernel
    files each have their own owner and sweep."""
    root = Path(inspect.getfile(bd)).parent
    offenders = []
    for name in ("document.py", "edits.py", "regen.py", "primitives.py", "selection.py"):
        text = (root / name).read_text(encoding="utf-8")
        for needle in _DELETED:
            for match in re.finditer(re.escape(needle), text):
                line = text.count("\n", 0, match.start()) + 1
                offenders.append(f"{name}:{line} names {needle!r}")
    assert not offenders, offenders
