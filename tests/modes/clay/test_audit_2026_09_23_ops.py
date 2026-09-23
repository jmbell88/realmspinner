"""Regression tests for the 2026-09-23 audit's clay-01, clay-05 and clay-18.

One file, three unrelated findings, because the fixer brief for this slice
("clay-ops") owns exactly ``studio/modes/clay/ops.py``,
``kernels/mesh/uvunwrap.py`` and ``kernels/mesh/topo.py``, and every new test
in this pass goes into one new module per fixer rather than editing a shared
one two fixers might touch at once.
"""

from __future__ import annotations

import numpy as np
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import topo
from realmspinner.kernels.mesh import uvunwrap as lscm
from realmspinner.kernels.mesh.elements import OpError
from realmspinner.studio.modes.clay import ops as clay_ops


class _Ctx:
    """The minimal double ``clay_ops.run`` needs: a ``toast`` method."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.info: list[str] = []

    def toast(self, message: str, level: str = "info") -> None:
        (self.errors if level == "error" else self.info).append(message)


# --- clay-01: Bake Transform under a locked parent --------------------------


def test_bake_transform_refuses_a_child_whose_ancestor_is_locked_without_corrupting_its_mesh() -> (
    None
):
    """The 2026-09-23 audit, finding clay-01: Bake Transform on an unlocked
    child of a locked parent used to write the world-baked mesh via
    ``set_mesh`` (which checks only the object's own lock) *before*
    ``set_transform(check_ancestors=True)`` ever got a chance to refuse --
    so the parent's translation landed baked into the child's geometry, and
    was then left *still live* on the still-parented, un-reset child,
    applying it a second time on screen. The refusal must fire before any
    write, leaving both the mesh and the transform exactly as they were.
    """
    doc = bd.ClayDoc()
    parent = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="Parent", mesh=bp.box(), translation=[10.0, 0.0, 0.0])
    )
    child = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="Child", mesh=bp.box(), translation=[1.0, 0.0, 0.0])
    )
    doc.set_parent(child.uid, parent.uid, keep_world=False)
    doc.set_props(parent.uid, locked=True)
    doc.select([child.uid])

    mesh_before = doc.by_uid(child.uid).mesh
    positions_before = np.array(mesh_before.positions, copy=True)
    translation_before = np.array(doc.by_uid(child.uid).translation, copy=True)
    parent_before = doc.by_uid(child.uid).parent
    history_depth_before = len(doc.history)

    ctx = _Ctx()
    ran = clay_ops.run(ctx, doc, clay_ops.get("bake"))

    child_after = doc.by_uid(child.uid)
    assert ran is False
    assert ctx.errors  # a refusal, not a silent no-op
    # The mesh identity itself must be unchanged -- not merely equal in
    # value, but the very same object set_mesh never touched.
    assert child_after.mesh is mesh_before
    assert np.array_equal(child_after.mesh.positions, positions_before)
    assert np.array_equal(np.asarray(child_after.translation), translation_before)
    assert child_after.parent == parent_before
    assert len(doc.history) == history_depth_before


# --- clay-05: Unwrap (Seams) on many islands ---------------------------------


def _disconnected_quads(n: int) -> bm.Mesh:
    """*n* fully disconnected quads, each its own uv island with no seams
    needed at all -- no two quads share a vertex index. Well under
    ``MAX_LSCM_VERTICES`` per island (4 vertices each) but, at the count
    this test patches ``MAX_UV_ISLANDS`` down to, over the island ceiling.
    """
    positions = np.zeros((n * 4, 3), dtype="f4")
    for i in range(n):
        base = i * 4
        positions[base + 0] = (i * 2.0, 0.0, 0.0)
        positions[base + 1] = (i * 2.0 + 1.0, 0.0, 0.0)
        positions[base + 2] = (i * 2.0 + 1.0, 1.0, 0.0)
        positions[base + 3] = (i * 2.0, 1.0, 0.0)
    loops = np.arange(n * 4, dtype="i4")
    starts = np.arange(0, n * 4 + 1, 4, dtype="i4")
    material = np.zeros(n, dtype="i4")
    smooth = np.zeros(n, dtype=bool)
    return bm.Mesh(
        positions=positions, loops=loops, starts=starts, material=material, smooth=smooth
    )


def test_unwrap_lscm_refuses_the_island_count_before_solving_any_island(monkeypatch) -> None:
    """The 2026-09-23 audit, finding clay-05: ``MAX_UV_ISLANDS`` used to be
    checked only inside ``pack_islands``, at the very end of
    ``unwrap_lscm`` -- after every island had already been solved with
    LSCM (33 s at 16,000 islands, reached by clicking Unwrap (Seams) on an
    import with a lot of disconnected small parts). Proved here the same
    way clay-13 proved the neighbouring vertex-ceiling refusal: make the
    per-island solver itself an assertion failure, so the test fails loudly
    if the refusal has not moved ahead of it.
    """
    monkeypatch.setattr(lscm, "MAX_UV_ISLANDS", 8)

    def _boom(mesh, faces, tris, seams, new_uv, pins):
        raise AssertionError("_solve_island ran before the island-count refusal")

    monkeypatch.setattr(lscm, "_solve_island", _boom)

    mesh = _disconnected_quads(20)  # 20 islands, over a ceiling of 8
    with pytest.raises(OpError, match="uv islands"):
        lscm.unwrap_lscm(mesh, seams=None)


# --- clay-18: region_boundary_corners on a non-manifold edge -----------------


def _hinge_fan(n_pages: int) -> bm.Mesh:
    """*n_pages* triangles hinged on one shared edge (v0, v1) -- the
    textbook non-manifold edge, ``edge_uses == n_pages`` for that one edge
    rather than the 1 or 2 an ordinary manifold mesh ever produces. Each
    page's apex sits at its own position so the faces are non-degenerate.
    """
    positions = [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    faces = []
    for i in range(n_pages):
        angle = i * (np.pi / max(n_pages, 1))
        positions.append((1.0 + 0.1 * i, 0.5, angle))  # apex, distinct per page
        faces.append([0, 1, 2 + i])
    return bm.from_faces(np.asarray(positions, dtype="f4"), faces)


def test_region_boundary_corners_includes_a_non_manifold_edge_with_an_unselected_face_on_it() -> (
    None
):
    """The 2026-09-23 audit, finding clay-18: an edge shared by three faces
    (two selected, one not) used to be missed. The old rule compared the
    *selected* corner count on an edge against exactly 1 -- right for an
    ordinary manifold edge, where at most two faces ever share one, but
    wrong here: both selected pages touch the hinge edge, so the old count
    read 2 and the edge was treated as interior even though the third,
    unselected page means it plainly borders the selection.
    """
    mesh = _hinge_fan(3)
    # Select pages 0 and 1; leave page 2 unselected. All three share the
    # hinge edge (vertices 0 and 1).
    border = topo.region_boundary_corners(mesh, [0, 1])
    # Every corner of both selected triangles should be on the border: the
    # hinge edge because an unselected third face also touches it, and the
    # two outer edges of each triangle because nothing else touches them at
    # all (true open boundary).
    assert len(border) == 6  # 2 faces * 3 corners each, all boundary

    # The closed-off case still reads as before: with every page selected,
    # the hinge edge is used by three *selected* corners and nothing
    # unselected touches it (per_edge == edge_uses == 3, not 1), so it is
    # correctly interior. Each triangle has exactly one corner leaving along
    # the hinge edge, so 3 of the 9 corners drop out, leaving the 6 that
    # leave along an outer, still-unshared edge.
    all_border = topo.region_boundary_corners(mesh, [0, 1, 2])
    assert len(all_border) == 6
