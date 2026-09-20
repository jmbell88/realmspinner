"""Tranche 6 (integration half): seams as authoring intent, not geometry.

See ``document.py``'s own module docstring and ``ClayDoc.set_seams``'s for
the design this pins -- ``Obj.seams`` is a sorted, deduplicated set of
vertex-index pairs into the object's *base* mesh, distinct from anything
``uvtools`` can derive from a uv layout, because a seam marked before an
unwrap has ever run cannot be derived from anything else on the object.

The restriction policy a mesh-replacing door applies is the other half this
file pins, one test per door named in ``document.py``'s own docstring:
``set_mesh``/``set_generator_params`` range-check (:func:`~realmspinner.kernels.
mesh.document._restrict_seams`, the same trade :func:`~realmspinner.kernels.mesh.
elements.restrict` already makes for an element selection), while
``separate``/``join_objects``/``apply_modifiers`` drop every seam outright
because each hands the document a mesh with no known correspondence to the
one that had the seams.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import modifiers as mod
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh.elements import OpError


def _obj(name: str, mesh: bm.Mesh | None = None, **kwargs: object) -> bd.Obj:
    return bd.Obj(
        uid=bd.new_uid(),
        name=name,
        mesh=bp.box() if mesh is None else mesh,
        **kwargs,  # type: ignore[arg-type]
    )


def _merged(doc: bd.ClayDoc, target: int, others: list[int]) -> bm.Mesh:
    from realmspinner.kernels.mesh import ops as clay_ops_geom

    return clay_ops_geom.join([doc.by_uid(u) for u in [target, *others]], eps=0.0)


# --- Obj construction / normalization -----------------------------------------


def test_a_seam_pair_is_stored_ordered_a_less_than_b() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    doc.set_seams(a.uid, [(3, 1)])
    assert doc.by_uid(a.uid).seams == ((1, 3),)


def test_duplicate_and_reordered_pairs_collapse_to_one() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    doc.set_seams(a.uid, [(1, 2), (2, 1), (1, 2)])
    assert doc.by_uid(a.uid).seams == ((1, 2),)


def test_a_self_referencing_pair_is_silently_dropped_not_refused() -> None:
    """A vertex paired with itself is not an edge -- ``_normalize_seams``'s
    own tolerant answer, the same one ``_normalize_tags`` gives an empty
    string, rather than a refusal for what construction alone cannot tell
    apart from a stray duplicate index."""
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    doc.set_seams(a.uid, [(4, 4), (0, 1)])
    assert doc.by_uid(a.uid).seams == ((0, 1),)


def test_an_object_with_no_seams_marked_has_the_empty_default() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    assert doc.by_uid(a.uid).seams == ()


# --- set_seams: the door -------------------------------------------------


def test_set_seams_refuses_a_vertex_the_mesh_does_not_have() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))  # box(): 8 vertices, indices 0..7
    with pytest.raises(OpError, match="does not"):
        doc.set_seams(a.uid, [(0, 99)])
    assert doc.by_uid(a.uid).seams == ()  # nothing pushed


def test_set_seams_refuses_a_locked_object() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A", locked=True))
    with pytest.raises(OpError, match="locked"):
        doc.set_seams(a.uid, [(0, 1)])


def test_set_seams_is_one_undo_step_and_undo_restores_it() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    head = doc.history.head
    assert doc.set_seams(a.uid, [(0, 1), (2, 3)]) is True
    assert doc.history.head != head
    assert doc.undo() is True
    assert doc.by_uid(a.uid).seams == ()
    assert doc.history.head == head


def test_set_seams_with_the_same_seams_pushes_nothing() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    doc.set_seams(a.uid, [(0, 1)])
    head = doc.history.head
    assert doc.set_seams(a.uid, [(1, 0)]) is False  # same pair, reordered
    assert doc.history.head == head


# --- restriction: set_mesh (range check) --------------------------------


def test_set_mesh_keeps_seams_when_every_vertex_still_exists() -> None:
    """A shading-only change touches no position or topology -- the vertex
    count is unchanged and every old index still names the same vertex, so a
    plain range check (the honest answer ``_restrict_seams`` gives for a
    door it cannot know the caller's intent through) keeps every seam."""
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    doc.set_seams(a.uid, [(0, 1), (2, 3)])
    doc.set_shading(a.uid, faces=np.array([0]), smooth=True)
    assert doc.by_uid(a.uid).seams == ((0, 1), (2, 3))


def _tiny_mesh() -> bm.Mesh:
    """A single-vertex, faceless mesh -- every seam pair a caller marked
    against a bigger mesh now names a vertex that is gone."""
    return bm.Mesh(
        positions=np.zeros((1, 3), dtype="f4"),
        loops=np.zeros(0, dtype="i4"),
        starts=np.zeros(1, dtype="i4"),
        material=np.zeros(0, dtype="i4"),
        smooth=np.zeros(0, dtype=bool),
    )


def test_set_mesh_drops_a_seam_past_the_new_vertex_count() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))  # box(): 8 vertices
    doc.set_seams(a.uid, [(0, 1), (0, 7)])
    doc.set_mesh(a.uid, _tiny_mesh())
    assert doc.by_uid(a.uid).seams == ()


def test_set_mesh_restriction_is_folded_into_the_one_mesh_edit_step() -> None:
    """The seam restriction rides the same ``CompoundEdit`` as the mesh
    replacement -- one Ctrl+Z brings back both the old mesh and its seams,
    the same "one press, one step" shape ``set_mesh``'s own generator freeze
    already keeps."""
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))  # box(): 8 vertices
    doc.set_seams(a.uid, [(0, 7)])
    head = doc.history.head
    doc.set_mesh(a.uid, _tiny_mesh())  # (0, 7) no longer in range: dropped
    assert doc.by_uid(a.uid).seams == ()
    assert doc.history.head == head + 1  # one step, not two
    assert doc.undo()
    assert doc.history.head == head
    assert doc.by_uid(a.uid).seams == ((0, 7),)  # both halves came back together


# --- restriction: set_generator_params (range check) ----------------------


def test_generator_rebuild_drops_out_of_range_seams_and_keeps_the_rest() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(
        _obj("A", mesh=bp.cylinder(segments=16), generator="cylinder",
             params={"radius": 0.5, "height": 1.0, "segments": 16})
    )
    assert len(a.mesh.positions) == 32
    doc.set_seams(a.uid, [(0, 1), (0, 20)])  # 20 is in range at 32 vertices

    smaller = bp.cylinder(segments=4)  # 8 vertices: indices 0..7
    assert len(smaller.positions) == 8
    doc.set_generator_params(
        a.uid,
        {"radius": 0.5, "height": 1.0, "segments": 4},
        smaller,
        was={"params": {"radius": 0.5, "height": 1.0, "segments": 16}},
    )
    assert doc.by_uid(a.uid).seams == ((0, 1),)  # (0, 20) dropped, 20 >= 8


# --- restriction: separate (drop all) --------------------------------------


def test_separate_drops_every_seam_on_every_piece() -> None:
    """``separate.py``'s own ``_piece`` compacts each piece's vertex array
    (its own docstring says so), so a piece's index and the source's are, in
    general, different vertices -- ``ClayDoc.separate`` receives only the
    finished meshes, never that remap, so nothing survives."""
    box = bp.box()
    # Two materials, split down the middle -- ``by_material`` needs at least
    # two distinct slots to have something to separate.
    material = box.material.copy()
    material[: len(material) // 2] = 1
    two_toned = bm.Mesh(
        positions=box.positions, loops=box.loops, starts=box.starts,
        smooth=box.smooth, material=material,
    )
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A", mesh=two_toned))
    doc.set_seams(a.uid, [(0, 1), (2, 3)])

    from realmspinner.kernels.mesh import separate as sep

    pieces = sep.by_material(two_toned)
    new_objs = doc.separate(a.uid, pieces)
    assert len(new_objs) == 2
    assert all(o.seams == () for o in new_objs)


def test_separate_still_carries_role_and_collider_kind() -> None:
    """Every other inheritable field (parent, transform, material, modifiers,
    tags, role, collider_kind) still survives a split -- seams are the one
    exception, and this pins that the exception is scoped to seams alone."""
    box = bp.box()
    material = box.material.copy()
    material[: len(material) // 2] = 1
    two_toned = bm.Mesh(
        positions=box.positions, loops=box.loops, starts=box.starts,
        smooth=box.smooth, material=material,
    )
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A", mesh=two_toned, role="collider", collider_kind="compound"))

    from realmspinner.kernels.mesh import separate as sep

    new_objs = doc.separate(a.uid, sep.by_material(two_toned))
    assert all(o.role == "collider" and o.collider_kind == "compound" for o in new_objs)


# --- restriction: join_objects (drop all) -----------------------------------


def test_join_objects_drops_the_targets_seams() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    b = doc.add_object(_obj("B", translation=(2.0, 0.0, 0.0)))
    doc.set_seams(a.uid, [(0, 1)])
    doc.join_objects(a.uid, _merged(doc, a.uid, [b.uid]), [b.uid])
    assert doc.by_uid(a.uid).seams == ()


def test_join_objects_leaves_seams_alone_on_a_true_no_op() -> None:
    """``doomed`` can be non-empty while the target's own mesh identity does
    not change (nothing to remove that used a different mesh reference) --
    that no-op branch must not blank seams it never touched."""
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    doc.set_seams(a.uid, [(0, 1)])
    assert doc.join_objects(a.uid, a.mesh, []) is False
    assert doc.by_uid(a.uid).seams == ((0, 1),)


# --- restriction: apply_modifiers (drop all when the mesh actually bakes) --


def test_apply_modifiers_drops_seams_when_the_bake_changes_the_mesh() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A", translation=(5.0, 0.0, 0.0)))
    doc.set_seams(a.uid, [(0, 1)])
    doc.set_modifiers(a.uid, (mod.make("mirror", {"weld": 0.0}, id=1),))
    assert doc.apply_modifiers(a.uid) is True
    assert doc.by_uid(a.uid).seams == ()


def test_apply_modifiers_keeps_seams_when_the_baked_prefix_is_only_disabled() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    doc.set_seams(a.uid, [(0, 1)])
    disabled = dataclasses.replace(mod.make("mirror", {}, id=1), enabled=False)
    doc.set_modifiers(a.uid, (disabled,))
    mesh_before = doc.by_uid(a.uid).mesh
    assert doc.apply_modifiers(a.uid) is True  # still pushes: the stack shrinks
    assert doc.by_uid(a.uid).mesh is mesh_before  # nothing enabled ran
    assert doc.by_uid(a.uid).seams == ((0, 1),)


# --- .rblk v3: seams ---------------------------------------------------------


def test_rblk_round_trips_seams() -> None:
    from realmspinner.kernels.mesh import serialize as ser

    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    doc.set_seams(a.uid, [(0, 1), (3, 5)])

    out = ser.read_rblk(ser.rblk_bytes(doc))
    assert out.by_uid(a.uid).seams == ((0, 1), (3, 5))


def test_an_object_with_no_seams_writes_no_seams_key() -> None:
    from realmspinner.kernels.mesh import serialize as ser

    doc = bd.ClayDoc()
    doc.add_object(_obj("A"))
    entry = json.loads(ser.scene_json(doc))["objects"][0]
    assert "seams" not in entry


def test_a_document_with_seams_is_still_byte_identical_when_repeated() -> None:
    from realmspinner.kernels.mesh import serialize as ser

    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    doc.set_seams(a.uid, [(0, 1)])
    assert ser.rblk_bytes(doc) == ser.rblk_bytes(doc)


def test_rblk_refuses_a_seam_naming_a_vertex_the_mesh_does_not_have() -> None:
    import zipfile
    from io import BytesIO

    from realmspinner.kernels.mesh import serialize as ser

    doc = bd.ClayDoc()
    doc.add_object(_obj("A"))  # box(): 8 vertices
    data = ser.rblk_bytes(doc)
    out = BytesIO()
    with zipfile.ZipFile(BytesIO(data)) as src, zipfile.ZipFile(out, "w") as dst:
        for name in src.namelist():
            if name == ser.SCENE:
                scene = json.loads(src.read(name))
                scene["objects"][0]["seams"] = [[0, 99]]
                dst.writestr(name, json.dumps(scene))
            else:
                dst.writestr(name, src.read(name))

    with pytest.raises(ValueError, match="vertices"):
        ser.read_rblk(out.getvalue())
