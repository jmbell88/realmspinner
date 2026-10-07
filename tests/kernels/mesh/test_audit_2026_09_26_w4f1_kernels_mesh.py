"""Closes the findings from the 2026-09-26 audit against ``kernels/mesh/``
that still apply, owned by wave 4 fixer 1 (``clay-mesh-model-*`` /
``clay-mesh-core-*``). The ``clay-mesh-model-*`` findings (readiness, analyze,
diagnose, bisect) lived in modules cut with the picoCAD-level Clay.

- clay-mesh-core-02 -- ``Adjacency.edge_ids`` aliased a pair naming a vertex
  past the mesh onto a real edge.
- clay-mesh-core-05 -- ``elements.affected_verts``/``select.verts_of`` walked
  the selected faces in a Python loop.
- clay-mesh-core-06 -- ``selection.duplicate_selected`` left a duplicated
  child's ``parent`` naming the original parent, not its sibling copy.

Every "OLD" figure quoted in a docstring below was measured by running the
pre-fix function -- loaded from ``git show HEAD`` at the start of this fix, as
a throwaway namespace bound to the live module's own globals so its
unchanged helpers/constants still resolve -- in this fix's own scratchpad,
never inside this tree.
"""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import primitives as prim
from realmspinner.kernels.mesh import select as sl
from realmspinner.kernels.mesh import selection
from realmspinner.kernels.mesh.adjacency import adjacency
from realmspinner.kernels.mesh.mesh import Mesh, face_count


def _obj(mesh, *, translation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), parent=None, material=0):
    return bd.Obj(
        uid=bd.new_uid(), name="Obj", mesh=mesh, translation=translation, scale=scale,
        generator=None, material=material, visible=True, parent=parent,
    )


def _grounded_box(**kwargs):
    return _obj(replace(prim.box(), uv=None), translation=(0.0, 0.5, 0.0), **kwargs)


# --- clay-mesh-core-02 --------------------------------------------------------


def test_edge_ids_returns_minus_one_for_a_pair_naming_a_vertex_past_the_mesh():
    """A box has 8 vertices (0-7); vertex 10 does not exist. Unfixed,
    ``edge_ids`` packed ``lo*span+hi`` with ``span = edge_verts.max() + 1``
    (8 here) and never checked either side against it, so ``(0, 10)`` packed
    to ``0*8+10 == 10``, the exact key ``searchsorted`` also finds for the
    real edge ``(1, 2)`` (``1*8+2 == 10``) -- reporting edge 3 for a pair that
    names a vertex outside the mesh entirely."""
    a = adjacency(prim.box())
    assert a.edge_ids(np.array([[0, 10]]))[0] == -1
    # A pair fully inside range that really is an edge still resolves.
    e = a.edge_verts[3]
    assert a.edge_ids(np.array([e]))[0] == 3


# --- clay-mesh-core-05 --------------------------------------------------------


def _grid_mesh(rows: int, cols: int) -> Mesh:
    """A shared-vertex quad grid -- ``(rows-1) x (cols-1)`` faces sharing
    ``rows x cols`` vertices, the realistic shape the 2026-09-26 audit
    measured against (an all-distinct-vertex mesh makes the closing
    ``np.unique`` dominate both the old and new code equally and hides the
    fix)."""
    r_idx, c_idx = np.meshgrid(np.arange(rows - 1), np.arange(cols - 1), indexing="ij")
    a = (r_idx * cols + c_idx).astype("i4")
    corners = np.stack([a, a + 1, a + cols + 1, a + cols], axis=-1)
    loops = corners.reshape(-1).astype("i4")
    n_faces = (rows - 1) * (cols - 1)
    starts = np.arange(0, 4 * (n_faces + 1), 4, dtype="i4")
    xs, zs = np.meshgrid(np.arange(rows, dtype="f4"), np.arange(cols, dtype="f4"), indexing="ij")
    positions = np.stack(
        [xs.reshape(-1), np.zeros(rows * cols, dtype="f4"), zs.reshape(-1)], axis=-1
    )
    return Mesh(
        positions=positions, loops=loops, starts=starts,
        material=np.zeros(n_faces, dtype="i4"), smooth=np.zeros(n_faces, dtype=bool),
    )


@pytest.mark.perf
def test_affected_verts_and_verts_of_finish_well_under_the_python_loops_time():
    """The 2026-09-26 audit measured a Python loop over selected faces at
    0.51s (``elements.affected_verts``) / 0.41s (``select.verts_of``) at
    490,000 faces on a shared-vertex grid -- reproduced in this fix's own
    scratchpad against the pre-fix functions at 0.519s/0.412s. The vectorised
    ``topo.corner_spans`` gather (already used by ``ops_model.bisect``)
    measured 0.203s/0.200s for the same call -- a generous 0.35s bound is
    comfortably clear of the fix and comfortably short of the unfixed loop."""
    mesh = _grid_mesh(700, 701)
    faces = np.arange(face_count(mesh), dtype="i4")
    sel = el.ElementSel(faces=faces)

    start = time.perf_counter()
    verts_a = el.affected_verts(mesh, sel)
    elapsed_a = time.perf_counter() - start
    assert elapsed_a < 0.35, f"affected_verts took {elapsed_a:.3f}s -- still a Python loop?"

    start = time.perf_counter()
    verts_b = sl.verts_of(mesh, sel, "face")
    elapsed_b = time.perf_counter() - start
    assert elapsed_b < 0.35, f"verts_of took {elapsed_b:.3f}s -- still a Python loop?"

    assert np.array_equal(verts_a.astype("i8"), verts_b)
    assert len(verts_a) == mesh.positions.shape[0]


# --- clay-mesh-core-06 --------------------------------------------------------


def test_duplicating_a_selected_parent_and_child_reparents_the_copy_to_its_sibling_copy():
    """Unfixed, ``ops.duplicate`` copies ``parent`` verbatim and
    ``duplicate_selected`` never remapped it, so Ctrl+D on a selected parent
    *and* child left the child copy parented to the *original* parent --
    the two hierarchies tangled together instead of one independent copy of
    the whole selection."""
    parent = _grounded_box()
    child = _obj(
        replace(prim.box(), uv=None), translation=(0.0, 0.2, 0.0), scale=(0.3, 0.3, 0.3),
        parent=parent.uid,
    )
    doc = bd.ClayDoc(objects=[parent, child])
    doc.select([parent.uid, child.uid])

    new_uids = selection.duplicate_selected(doc)
    assert len(new_uids) == 2
    parent_copy_uid, child_copy_uid = new_uids  # document order: parent, then child

    child_copy = doc.by_uid(child_copy_uid)
    assert child_copy.parent == parent_copy_uid

    # A copy whose original parent was *not* part of the selection keeps
    # naming the original -- there is no sibling copy to move it to.
    lone_child = _obj(replace(prim.box(), uv=None), parent=parent.uid)
    doc.add_objects([lone_child])
    doc.select([lone_child.uid])
    (lone_copy_uid,) = selection.duplicate_selected(doc)
    assert doc.by_uid(lone_copy_uid).parent == parent.uid
