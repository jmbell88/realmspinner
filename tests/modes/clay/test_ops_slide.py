"""Edge slide, restored 2026-10-13 into ``ops_slide`` (Tranche 1 landing 4 of the Clay rig plan).

The kernel came back from 112c69e2^:``ops_model.py``, whose other ops stay cut. The first
five claims are the originals from that file's tests; the strip claims below them are new,
because the originals only asserted that a slide moved something.
"""

from __future__ import annotations

import numpy as np
import pytest

from realmspinner.kernels.mesh import adjacency as adj
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import ops_slide
from realmspinner.kernels.mesh import primitives as prim
from realmspinner.kernels.mesh.mesh import from_faces


def _uvd(mesh: bm.Mesh) -> bm.Mesh:
    n = len(mesh.loops)
    uv = np.stack([np.arange(n, dtype="f4"), np.arange(n, dtype="f4") * 2], axis=1)
    return bm.Mesh(
        positions=mesh.positions,
        loops=mesh.loops,
        starts=mesh.starts,
        material=mesh.material,
        smooth=mesh.smooth,
        uv=uv,
    )


def _strip() -> bm.Mesh:
    """Three quads side by side: vertices at x = 0, 1, 2, 3 on two rows (z = 0, 1).

    Vertex ``2 * i`` is (i, 0, 0) and ``2 * i + 1`` is (i, 0, 1), so the rung at x = 1 is
    the edge (2, 3) and its rails run to x = 0 and x = 2.
    """
    positions = [[x, 0, z] for x in range(4) for z in (0, 1)]
    faces = [[0, 2, 3, 1], [2, 4, 5, 3], [4, 6, 7, 5]]
    return from_faces(positions, faces)


def test_edge_slide_at_zero_does_not_move_anything() -> None:
    box = prim.box()
    a = adj.adjacency(box)
    sel = el.ElementSel(edges=np.array([a.edge_verts[0]], dtype="i4"))
    out, out_sel = ops_slide.edge_slide(box, sel, t=0.0)
    assert np.allclose(out.positions, box.positions)
    assert out_sel is sel


def test_edge_slide_at_extremes_reaches_its_rail_neighbours() -> None:
    box = prim.box()
    sel = el.ElementSel(edges=np.array([[4, 5]], dtype="i4"))
    plus, _ = ops_slide.edge_slide(box, sel, t=1.0)
    minus, _ = ops_slide.edge_slide(box, sel, t=-1.0)
    bm.validate(plus)
    bm.validate(minus)
    assert not np.allclose(plus.positions[4], box.positions[4])
    assert not np.allclose(minus.positions[4], box.positions[4])
    assert not np.allclose(plus.positions[4], minus.positions[4])


def test_edge_slide_with_no_selection_refuses() -> None:
    with pytest.raises(el.OpError, match="Select an edge loop"):
        ops_slide.edge_slide(prim.box(), el.empty(), t=0.5)


def test_edge_slide_an_unknown_edge_refuses() -> None:
    sel = el.ElementSel(edges=np.array([[0, 2]], dtype="i4"))  # a diagonal, no such edge
    with pytest.raises(el.OpError, match="not part of this mesh"):
        ops_slide.edge_slide(prim.box(), sel, t=0.5)


def test_edge_slide_refuses_a_selection_past_the_vertex_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 2026-09-19 audit's clay-15: a whole-mesh Select All reaches the slide with no
    seed, and the per-vertex Python loop measured 50/195/808 ms at 8k/32k/130k vertices.
    Driven with the ceiling lowered rather than with a 65,537-vertex mesh."""
    monkeypatch.setattr(ops_slide, "MAX_SLIDE_VERTICES", 1)
    sel = el.ElementSel(edges=np.array([[4, 5], [5, 6]], dtype="i4"))
    with pytest.raises(el.OpError, match="past the"):
        ops_slide.edge_slide(prim.box(), sel, t=0.3)


def test_the_real_ceiling_has_not_crept_down_onto_ordinary_use() -> None:
    assert ops_slide.MAX_SLIDE_VERTICES >= 65_536


def test_a_rung_slides_along_the_strip_and_stays_straight() -> None:
    strip = _strip()
    sel = el.ElementSel(edges=np.array([[2, 3]], dtype="i4"))
    out, _ = ops_slide.edge_slide(strip, sel, t=0.5)
    bm.validate(out)
    x = out.positions[[2, 3], 0]
    assert x[0] == pytest.approx(x[1]), "both ends of the rung move the same way"
    assert x[0] in (pytest.approx(0.5), pytest.approx(1.5))
    # Nothing but the rung moved.
    others = np.setdiff1d(np.arange(len(strip.positions)), [2, 3])
    assert np.allclose(out.positions[others], strip.positions[others])


def test_opposite_signs_slide_to_opposite_sides_and_full_travel_lands_on_the_neighbour() -> None:
    strip = _strip()
    sel = el.ElementSel(edges=np.array([[2, 3]], dtype="i4"))
    plus, _ = ops_slide.edge_slide(strip, sel, t=0.5)
    minus, _ = ops_slide.edge_slide(strip, sel, t=-0.5)
    assert {round(float(plus.positions[2, 0]), 3), round(float(minus.positions[2, 0]), 3)} == {
        0.5,
        1.5,
    }
    full, _ = ops_slide.edge_slide(strip, sel, t=1.0)
    assert float(full.positions[2, 0]) in (0.0, 2.0)


def test_a_slide_keeps_topology_and_uv_untouched() -> None:
    strip = _uvd(_strip())
    sel = el.ElementSel(edges=np.array([[2, 3]], dtype="i4"))
    out, _ = ops_slide.edge_slide(strip, sel, t=0.3)
    assert np.array_equal(out.uv, strip.uv)
    assert np.array_equal(out.loops, strip.loops) and np.array_equal(out.starts, strip.starts)


def test_a_vertex_whose_only_rails_are_selected_is_refused_by_name() -> None:
    # A lone quad's every edge sits in the selection, so no vertex has a rail outside it.
    quad = from_faces([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], [[0, 1, 2, 3]])
    sel = el.ElementSel(edges=np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype="i4"))
    with pytest.raises(el.OpError, match="no edge to slide along"):
        ops_slide.edge_slide(quad, sel, t=0.5)
