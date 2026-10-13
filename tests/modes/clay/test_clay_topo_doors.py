"""Dissolve, Bridge Loops, Fill Hole and Collapse: the doors on kernels that outlived the cut.

Tranche 1 landing 1 of the 2026-10-13 Clay rig plan. 112c69e2 cut these ops from Clay's
registry and left four of the five kernels tested but unreachable (``dissolve_verts`` and
``dissolve_edges`` were deleted outright and come back from 112c69e2^). Every claim here is
about the registry row and the doors generated from it, and each fails on the tree before the
restore: ``clay_ops.get("dissolve")`` raises ``KeyError`` and the palette has no
``clay-op:dissolve``.
"""

from __future__ import annotations

import numpy as np
import pytest

from realmspinner.kernels.mesh import adjacency as adj
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import topo
from realmspinner.studio.modes.clay import menutree
from realmspinner.studio.modes.clay import ops as clay_ops

from .test_clay_palette import _clay
from .test_clay_palette import _ctx as palette_ctx
from .test_clay_palette import _doc as palette_doc

NAMES = ("dissolve", "bridge", "fill-hole", "collapse")


class _Ctx:
    def __init__(self) -> None:
        self.toasts: list[tuple[str, str]] = []

    def toast(self, message: str, level: str = "info") -> None:
        self.toasts.append((message, level))


def _box_without(*signed_axes: tuple[int, float]) -> bm.Mesh:
    """A unit box with the faces whose normal points along each (axis, sign) removed."""
    box = bp.box()
    normals = bm.face_normals(box)
    keep = [
        f
        for f in range(bm.face_count(box))
        if not any(normals[f][axis] * sign > 0.9 for axis, sign in signed_axes)
    ]
    return topo.compact_vertices(topo.take_faces(box, np.array(keep, dtype="i8")))[0]


def _box_without_top() -> bm.Mesh:
    return _box_without((1, 1.0))


def _band() -> bm.Mesh:
    """A box with its +Y and -Y faces removed: four sides and two open loops."""
    return _box_without((1, 1.0), (1, -1.0))


def _boundary_edges(mesh: bm.Mesh) -> np.ndarray:
    a = adj.adjacency(mesh)
    return a.edge_verts[a.edge_uses == 1]


def _doc(mesh: bm.Mesh, mode: str, sel: el.ElementSel | None) -> tuple[bd.ClayDoc, int]:
    doc = bd.ClayDoc()
    uid = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Shape", mesh=mesh)).uid
    doc.select([uid])
    doc.set_element_mode(mode)
    if sel is not None:
        doc.set_element_sel(uid, sel)
    return doc, uid


def _one_edge(mesh: bm.Mesh) -> np.ndarray:
    a = adj.adjacency(mesh)
    return a.edge_verts[[int(np.flatnonzero(a.edge_uses == 2)[0])]]


def test_the_four_rows_exist_with_the_modes_their_kernels_work_in() -> None:
    assert clay_ops.get("dissolve").modes == ("vertex", "edge")
    assert clay_ops.get("bridge").modes == ("edge",)
    assert clay_ops.get("fill-hole").modes == ("edge",)
    assert clay_ops.get("collapse").modes == ("edge", "face")
    assert clay_ops.get("bridge").label == "Bridge Loops"
    assert clay_ops.get("fill-hole").label == "Fill Hole"


def test_the_four_rows_are_a_closed_topology_fold_not_the_open_mesh_card() -> None:
    # The open Mesh group is the Inspector's Shape it card; growing it pushed Delete 17 px
    # past the open dock at 1600x950 (test_colour_palette's
    # test_delete_in_the_element_state_clears_the_open_dock_at_1600x950).
    assert menutree.GROUPS["topology"].ops == NAMES
    assert not set(NAMES) & set(menutree.GROUPS["mesh"].ops)
    flat = {
        m: [op.name for run in menutree.flat(m) for op in run]
        for m in ("vertex", "edge", "face", "object")
    }
    assert {"dissolve", "bridge", "fill-hole", "collapse"} <= set(flat["edge"])
    assert flat["vertex"].count("dissolve") == 1 and "collapse" not in flat["vertex"]
    assert "collapse" in flat["face"] and "dissolve" not in flat["face"]
    assert not set(NAMES) & set(flat["object"])


@pytest.mark.parametrize("name", NAMES)
def test_each_row_is_greyed_with_a_reason_in_the_wrong_mode_and_with_nothing_picked(
    name: str,
) -> None:
    op = clay_ops.get(name)
    doc, _ = _doc(bp.box(), "object", None)
    assert not op.enabled(doc)
    assert clay_ops.reason_for(op, doc), "object mode names the mode to switch to"
    doc.set_element_mode(op.modes[0])
    assert not op.enabled(doc)
    assert "Select" in clay_ops.reason_for(op, doc)


def test_dissolving_a_picked_box_edge_merges_the_two_faces_across_it_in_one_undo_step() -> None:
    mesh = bp.box()
    doc, uid = _doc(mesh, "edge", el.ElementSel(edges=_one_edge(mesh)))
    steps, ctx = len(doc.history), _Ctx()
    assert clay_ops.run(ctx, doc, clay_ops.get("dissolve")) is True
    assert ctx.toasts == []
    assert bm.face_count(doc.by_uid(uid).mesh) == bm.face_count(mesh) - 1
    assert len(doc.history) == steps + 1
    assert doc.undo()
    assert bm.face_count(doc.by_uid(uid).mesh) == bm.face_count(mesh)


def test_dissolving_a_picked_box_corner_merges_the_three_faces_around_it() -> None:
    mesh = bp.box()
    doc, uid = _doc(mesh, "vertex", el.ElementSel(verts=[0]))
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("dissolve")) is True
    assert bm.face_count(doc.by_uid(uid).mesh) == bm.face_count(mesh) - 2
    assert len(doc.by_uid(uid).mesh.positions) == len(mesh.positions) - 1


def test_dissolving_a_boundary_edge_is_a_refusal_toast_that_changes_nothing() -> None:
    mesh = _box_without_top()
    doc, uid = _doc(mesh, "edge", el.ElementSel(edges=_boundary_edges(mesh)[:1]))
    steps, ctx = len(doc.history), _Ctx()
    assert clay_ops.run(ctx, doc, clay_ops.get("dissolve")) is False
    assert any("boundary" in message for message, _ in ctx.toasts)
    assert len(doc.history) == steps
    assert bm.face_count(doc.by_uid(uid).mesh) == bm.face_count(mesh)


def test_fill_hole_caps_a_box_missing_its_top_with_one_face() -> None:
    mesh = _box_without_top()
    assert len(_boundary_edges(mesh)) == 4
    doc, uid = _doc(mesh, "edge", el.ElementSel(edges=_boundary_edges(mesh)[:1]))
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("fill-hole")) is True
    out = doc.by_uid(uid).mesh
    assert bm.face_count(out) == bm.face_count(mesh) + 1
    assert len(_boundary_edges(out)) == 0


def test_bridge_loops_joins_the_two_open_ends_of_a_tube_into_a_closed_solid() -> None:
    band = _band()
    edges = _boundary_edges(band)
    assert len(edges) == 8, "two open loops of four edges"
    doc, uid = _doc(band, "edge", el.ElementSel(edges=edges))
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("bridge")) is True
    out = doc.by_uid(uid).mesh
    assert bm.face_count(out) == bm.face_count(band) + 4
    assert len(_boundary_edges(out)) == 0


def test_collapse_in_edge_mode_pulls_the_picked_edge_to_one_vertex() -> None:
    mesh = bp.box()
    doc, uid = _doc(mesh, "edge", el.ElementSel(edges=_one_edge(mesh)))
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("collapse")) is True
    assert len(doc.by_uid(uid).mesh.positions) == len(mesh.positions) - 1


def test_collapse_in_face_mode_pulls_the_picked_face_to_its_centre() -> None:
    mesh = bp.box()
    doc, uid = _doc(mesh, "face", el.ElementSel(faces=[0]))
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("collapse")) is True
    assert len(doc.by_uid(uid).mesh.positions) < len(mesh.positions)


@pytest.mark.parametrize("name", NAMES)
def test_ctrl_k_lists_each_row_and_it_runs_in_its_mode(name: str) -> None:
    ctx = palette_ctx(objects=1)
    palette_doc(ctx)
    row = _clay(ctx)[f"clay-op:{name}"]
    assert row.label == clay_ops.get(name).label
    assert not row.enabled(ctx) and row.why, "object mode: listed, greyed, with the reason"


def test_the_agents_clay_op_enum_names_the_four_rows() -> None:
    from realmspinner.familiar import contract

    card = contract.load_card("clay")
    for name in NAMES:
        assert f'"{name}"' in card
