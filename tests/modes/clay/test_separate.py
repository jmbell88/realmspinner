"""Tranche 3: separate -- splitting one object's mesh into several, by loose
parts, by material, or by a selection.

The kernel (:mod:`realmspinner.kernels.mesh.separate`) and the document door
(:meth:`~realmspinner.kernels.mesh.document.ClayDoc.separate`) are tested
separately, the way this package always splits geometry from bookkeeping:
the kernel answers "how would this split", the door answers "what happens to
the document when it does" -- one undo step, the same parent/transform/
modifier stack copied onto every piece, the source removed.
"""

from __future__ import annotations

import numpy as np
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import modifiers as mod
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import separate
from realmspinner.kernels.mesh.elements import OpError


def _obj(name: str, mesh: bm.Mesh | None = None, **kwargs: object) -> bd.Obj:
    return bd.Obj(
        uid=bd.new_uid(),
        name=name,
        mesh=bp.box() if mesh is None else mesh,
        **kwargs,  # type: ignore[arg-type]
    )


def _two_boxes(offset: tuple[float, float, float] = (5.0, 0.0, 0.0)) -> bm.Mesh:
    """Two boxes, sharing no vertex index -- two loose parts."""
    a = bp.box()
    b = bp.box()
    b_positions = np.asarray(b.positions, dtype="f4") + np.asarray(offset, dtype="f4")
    merged = bm.Mesh(
        positions=np.concatenate([a.positions, b_positions]),
        loops=np.concatenate([a.loops, b.loops + len(a.positions)]),
        starts=np.concatenate([a.starts, a.starts[-1] + b.starts[1:]]),
        material=np.concatenate([a.material, b.material]),
        smooth=np.concatenate([a.smooth, b.smooth]),
    )
    bm.validate(merged)
    return merged


def _two_toned_box() -> bm.Mesh:
    box = bp.box()
    material = np.zeros(bm.face_count(box), dtype="i4")
    material[: bm.face_count(box) // 2] = 1
    out = bm.Mesh(
        positions=box.positions,
        loops=box.loops,
        starts=box.starts,
        material=material,
        smooth=box.smooth,
    )
    bm.validate(out)
    return out


# --- the kernel: by_loose_parts ---------------------------------------------


def test_by_loose_parts_splits_two_disjoint_boxes() -> None:
    pieces = separate.by_loose_parts(_two_boxes())
    assert len(pieces) == 2
    for piece in pieces:
        assert len(piece.positions) == 8
        assert bm.face_count(piece) == 6


def test_by_loose_parts_refuses_a_single_connected_mesh() -> None:
    with pytest.raises(OpError, match="one connected piece"):
        separate.by_loose_parts(bp.box())


def test_by_loose_parts_compacts_positions_not_a_view_of_the_whole() -> None:
    """The module's own stated difference from ``document._submesh``: a
    piece's own vertex array holds only the vertices its own faces use."""
    pieces = separate.by_loose_parts(_two_boxes())
    total = sum(len(p.positions) for p in pieces)
    assert total == 16  # 8 + 8, not 16 + 16 (each piece holding both)


def test_by_loose_parts_keeps_material_smooth_and_uv_per_piece() -> None:
    box = bp.box()
    uv = np.zeros((len(box.loops), 2), dtype="f4")
    uv[:, 0] = np.arange(len(box.loops), dtype="f4")
    tagged = bm.Mesh(
        positions=box.positions,
        loops=box.loops,
        starts=box.starts,
        material=np.full(bm.face_count(box), 3, dtype="i4"),
        smooth=np.ones(bm.face_count(box), dtype=bool),
        uv=uv,
    )
    other = bp.box()
    merged = bm.Mesh(
        positions=np.concatenate([tagged.positions, np.asarray(other.positions) + 5.0]),
        loops=np.concatenate([tagged.loops, other.loops + len(tagged.positions)]),
        starts=np.concatenate([tagged.starts, tagged.starts[-1] + other.starts[1:]]),
        material=np.concatenate([tagged.material, other.material]),
        smooth=np.concatenate([tagged.smooth, other.smooth]),
        uv=np.concatenate([tagged.uv, np.zeros((len(other.loops), 2), dtype="f4")]),
    )
    bm.validate(merged)

    pieces = separate.by_loose_parts(merged)
    tagged_piece = next(p for p in pieces if int(p.material[0]) == 3)
    assert bool(tagged_piece.smooth[0]) is True
    assert tagged_piece.uv is not None
    assert np.array_equal(tagged_piece.uv[:, 0], uv[:, 0])


# --- the kernel: by_material -------------------------------------------------


def test_by_material_splits_by_slot() -> None:
    pieces = separate.by_material(_two_toned_box())
    assert len(pieces) == 2
    materials = sorted(int(p.material[0]) for p in pieces)
    assert materials == [0, 1]
    for piece in pieces:
        assert len(set(piece.material.tolist())) == 1


def test_by_material_refuses_a_single_material_mesh() -> None:
    with pytest.raises(OpError, match="same material"):
        separate.by_material(bp.box())


def test_by_material_refuses_a_faceless_mesh() -> None:
    with pytest.raises(OpError, match="no faces"):
        separate.by_material(bd._empty_mesh())


# --- the kernel: by_selection -------------------------------------------------


def test_by_selection_splits_selected_faces_from_the_rest() -> None:
    box = bp.box()
    n = bm.face_count(box)
    sel = el.ElementSel(faces=np.array([0, 1], dtype="i4"))
    pieces = separate.by_selection(box, sel)
    assert len(pieces) == 2
    assert bm.face_count(pieces[0]) == 2
    assert bm.face_count(pieces[1]) == n - 2


def test_by_selection_refuses_an_empty_selection() -> None:
    with pytest.raises(OpError, match="Nothing to separate"):
        separate.by_selection(bp.box(), el.empty())


def test_by_selection_refuses_selecting_everything() -> None:
    box = bp.box()
    sel = el.ElementSel(faces=np.arange(bm.face_count(box), dtype="i4"))
    with pytest.raises(OpError, match="Nothing to separate"):
        separate.by_selection(box, sel)


def test_by_selection_converts_a_vertex_selection_up_to_faces() -> None:
    """Wings3D's "all corners selected" rule, via ``elements.convert`` --
    the whole reason this takes an ``ElementSel`` rather than a bare face
    index array."""
    box = bp.box()
    corners = box.loops[box.starts[0] : box.starts[1]]
    sel = el.ElementSel(verts=np.unique(corners))
    pieces = separate.by_selection(box, sel)
    assert bm.face_count(pieces[0]) == 1


# --- the document door: ClayDoc.separate -------------------------------------


def test_document_separate_is_one_step_with_stacks_copied() -> None:
    doc = bd.ClayDoc()
    parent = doc.add_object(_obj("Parent"))
    a = doc.add_object(
        _obj(
            "A",
            mesh=_two_boxes(),
            translation=(1.0, 0.0, 0.0),
            modifiers=(mod.make("weld", id=1),),
        )
    )
    doc.set_parent(a.uid, parent.uid, keep_world=False)
    before_head = doc.history.head

    pieces = separate.by_loose_parts(a.mesh)
    new_objs = doc.separate(a.uid, pieces)

    assert len(new_objs) == 2
    assert a.uid not in [o.uid for o in doc.objects]
    for piece in new_objs:
        assert piece.parent == parent.uid
        assert np.allclose(piece.translation, [1.0, 0.0, 0.0])
        assert piece.modifiers == (mod.make("weld", id=1),)
        assert piece.generator is None

    # Names are suffixed, not identical.
    assert len({o.name for o in new_objs}) == 2

    assert doc.undo() is True
    assert [o.uid for o in doc.objects] == [parent.uid, a.uid]
    assert doc.history.head == before_head


def test_document_separate_refuses_fewer_than_two_pieces() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A"))
    with pytest.raises(OpError, match="single piece"):
        doc.separate(a.uid, [bp.box()])


def test_document_separate_selects_the_new_pieces() -> None:
    doc = bd.ClayDoc()
    a = doc.add_object(_obj("A", mesh=_two_boxes()))
    pieces = separate.by_loose_parts(a.mesh)
    new_objs = doc.separate(a.uid, pieces)
    assert doc.selection == {o.uid for o in new_objs}
