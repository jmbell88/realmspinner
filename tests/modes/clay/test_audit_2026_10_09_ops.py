"""Regression tests for the 2026-10-09 Clay audit's ops findings
(clay-07, 11, 32, 33, 34, 35, 38, 50). Each test name is the claim."""

from __future__ import annotations

import dataclasses
import inspect
import re
from pathlib import Path

import numpy as np

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import drag as bdrag
from realmspinner.kernels.mesh import ops as mesh_ops
from realmspinner.kernels.mesh import ops_bevel, regen
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh.adjacency import adjacency
from realmspinner.kernels.mesh.elements import ElementSel
from realmspinner.studio.modes.clay import menutree
from realmspinner.studio.modes.clay import ops as clay_ops

ROOT = Path(__file__).resolve().parents[3]


class _Ctx:
    def __init__(self) -> None:
        self.msgs: list[str] = []

    def toast(self, message: str, level: str = "info") -> None:
        self.msgs.append(message)


def _hidden_picked_box() -> tuple[bd.ClayDoc, int]:
    doc = bd.ClayDoc()
    ob = doc.add_object(
        bd.Obj(
            uid=bd.new_uid(),
            name="Box",
            mesh=bp.box(),
            generator="box",
            params={"size": (1.0, 1.0, 1.0)},
        )
    )
    doc.set_element_mode("face")
    doc.set_element_sel(ob.uid, ElementSel(faces=[0]))
    ob.visible = False  # as a redone hide step leaves it: picks kept
    return doc, ob.uid


def _edge(mesh, p, q):
    a = adjacency(mesh)
    for u, v in a.edge_verts:
        if {tuple(np.round(mesh.positions[u], 3)), tuple(np.round(mesh.positions[v], 3))} == {p, q}:
            return (int(u), int(v))
    raise AssertionError("no such edge")


# --- clay-07 ------------------------------------------------------------------


def test_origin_to_selection_and_separate_selection_skip_a_hidden_object_that_kept_its_picks():
    doc, uid = _hidden_picked_box()
    ctx = _Ctx()
    before = tuple(doc.by_uid(uid).translation)
    assert not clay_ops.run(ctx, doc, clay_ops.get("origin-to-selection"))
    assert tuple(doc.by_uid(uid).translation) == before

    doc, uid = _hidden_picked_box()
    assert not clay_ops.run(ctx, doc, clay_ops.get("separate-selection"))
    assert len(doc.objects) == 1


def test_assign_material_on_only_hidden_objects_says_so_rather_than_already_use():
    doc, _ = _hidden_picked_box()
    ctx = _Ctx()
    assert not clay_ops.run(ctx, doc, clay_ops.get("assign-material"))
    assert ctx.msgs and "already use" not in ctx.msgs[-1]
    assert "hidden" in ctx.msgs[-1]


# --- clay-11 / clay-33 --------------------------------------------------------


def test_a_wide_bevel_never_leaves_two_vertices_at_one_point():
    m = bp.box((1, 1, 1))
    h = 0.5
    top = _edge(m, (-h, h, h), (h, h, h))
    bot = _edge(m, (-h, -h, h), (h, -h, h))
    sel = ElementSel(edges=np.array([top, bot]))
    for w in (0.1, 0.4, 0.5, 0.8, 5.0):
        out, _ = ops_bevel.bevel_edges(m, sel, width=w)
        q = np.round(out.positions.astype("f8"), 5)
        assert len(q) == len(np.unique(q, axis=0)), f"coincident vertices at width {w}"


def test_bevel_of_one_box_edge_makes_two_pentagons():
    m = bp.box((1, 1, 1))
    h = 0.5
    e = _edge(m, (-h, h, h), (h, h, h))
    out, _ = ops_bevel.bevel_edges(m, ElementSel(edges=np.array([e])), width=0.1)
    arity = np.diff(np.append(out.starts, len(out.loops)))
    assert int((arity == 5).sum()) == 2
    assert "two faces at the edge's ends" in " ".join(inspect.getdoc(ops_bevel.bevel_edges).split())


# --- clay-32 ------------------------------------------------------------------


def test_a_size_edit_keeps_the_unwrapped_uv_layout_when_the_face_count_is_unchanged():
    old = bp.box((1, 1, 1))
    marked = dataclasses.replace(old, uv=old.uv + 0.123)
    rebuilt = regen.carry_over(marked, bp.box((2, 1, 1)))
    assert np.array_equal(rebuilt.uv, marked.uv)
    doc = inspect.getdoc(clay_ops._unwrap)
    assert "correctly rebuilds it with the generator's own canonical" not in doc


# --- clay-34 / clay-35 --------------------------------------------------------


def test_ops_docstring_cross_references_resolve():
    src = inspect.getsource(mesh_ops)
    assert "viewer.math3d" not in src
    assert (ROOT / "src/realmspinner/kernels/geom3d/math3d.py").is_file()


def test_drag_input_carries_no_field_nothing_reads():
    names = {f.name for f in dataclasses.fields(bdrag.DragInput)}
    assert "_cache" not in names


# --- clay-38 ------------------------------------------------------------------


def test_the_manual_says_frame_selection_is_in_the_right_click_menu_exactly_when_flat_draws_it():
    text = (ROOT / "docs/manual/30-clay.md").read_text(encoding="utf-8")
    text = " ".join(text.split())
    drawn = all(
        any(op.name == "frame" for run in menutree.flat(mode) for op in run)
        for mode in clay_ops.ALL_MODES
    )
    says_absent = bool(
        re.search(r"every one is in the right-click menu except Frame Selection", text)
    )
    assert drawn != says_absent


# --- clay-50 ------------------------------------------------------------------


def test_bevel_width_zero_is_clamped_to_the_declared_floor_not_refused_by_the_kernel():
    param = clay_ops.get("bevel").params[0]
    assert param.low > 0.0
    seen = {}
    real = ops_bevel.bevel_edges

    def spy(mesh, sel, *, width):
        seen["width"] = width
        return real(mesh, sel, width=width)

    doc = bd.ClayDoc()
    ob = doc.add_object(
        bd.Obj(
            uid=bd.new_uid(),
            name="B",
            mesh=bp.box(),
            generator="box",
            params={"size": (1.0, 1.0, 1.0)},
        )
    )
    doc.set_element_mode("edge")
    m = ob.mesh
    edge = _edge(m, (-0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    doc.set_element_sel(ob.uid, ElementSel(edges=np.array([edge])))
    ctx = _Ctx()
    ops_bevel.bevel_edges = spy
    try:
        assert clay_ops.run(ctx, doc, clay_ops.get("bevel"), width=0.0)
    finally:
        ops_bevel.bevel_edges = real
    assert seen["width"] == param.low
