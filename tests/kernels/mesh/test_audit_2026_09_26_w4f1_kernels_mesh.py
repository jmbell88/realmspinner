"""Closes ten findings from the 2026-09-26 audit against ``kernels/mesh/``,
owned by wave 4 fixer 1 (``clay-mesh-model-*`` / ``clay-mesh-core-*``):

- clay-mesh-core-02 -- ``Adjacency.edge_ids`` aliased a pair naming a vertex
  past the mesh onto a real edge.
- clay-mesh-model-03 -- ``readiness._check_triangles`` could reach ``fail``,
  though the docstring, manual 30:819-820 and INVARIANTS all say only
  ``objects``/``uvs`` can.
- clay-mesh-model-04 -- a pair that clears the grid with no shared triangle
  cell was reported ``exact=True`` on a vertex-only distance.
- clay-mesh-model-05 -- ``MAX_TRIANGLE_PAIRS`` bounded one object pair's own
  candidate count, never the cumulative cost across one ``analyze()`` call.
- clay-mesh-model-06 -- ``diagnose.findings`` paid for ``check_manifold`` and
  an uncapped ``boundary_loops`` walk before the ``ops_clean`` ceiling ever
  refused an oversized mesh.
- clay-mesh-model-07 -- ``ops_model.bisect`` emitted a face wholly inside the
  cut plane twice under ``clear=0``.
- clay-mesh-core-05 -- ``elements.affected_verts``/``select.verts_of`` walked
  the selected faces in a Python loop.
- clay-mesh-core-06 -- ``selection.duplicate_selected`` left a duplicated
  child's ``parent`` naming the original parent, not its sibling copy.
- clay-mesh-model-08 -- one oversized visible object skipped
  ``geometry``/``normals``/``closed`` for the *whole* document, not just
  itself.

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

from realmspinner.kernels.mesh import analyze as az
from realmspinner.kernels.mesh import diagnose as dg
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import ops_clean as oc
from realmspinner.kernels.mesh import ops_model as om
from realmspinner.kernels.mesh import primitives as prim
from realmspinner.kernels.mesh import readiness, selection
from realmspinner.kernels.mesh import select as sl
from realmspinner.kernels.mesh.adjacency import adjacency
from realmspinner.kernels.mesh.mesh import Mesh, face_count


def _obj(mesh, *, translation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), parent=None, material=0):
    return bd.Obj(
        uid=bd.new_uid(), name="Obj", mesh=mesh, translation=translation, scale=scale,
        generator=None, material=material, visible=True, parent=parent,
    )


def _grounded_box(**kwargs):
    return _obj(replace(prim.box(), uv=None), translation=(0.0, 0.5, 0.0), **kwargs)


def _by_key(report: readiness.Report) -> dict[str, readiness.Check]:
    return {c.key: c for c in report.checks}


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


# --- clay-mesh-model-03 --------------------------------------------------------


def test_only_the_rows_the_manual_names_can_fail(monkeypatch):
    """The module docstring, manual 30:819-820 and INVARIANTS all say only
    ``objects`` and ``uvs`` can reach ``"fail"`` -- a budget is advice however
    far over it a document runs, since an engine imports past it regardless.
    Unfixed, ``_check_triangles`` returned ``"fail"`` once the document's
    triangle total passed ``profile.triangles_fail``, in direct contradiction
    of its own module's stated invariant."""
    low = readiness.Profile(
        key="low", label="Low",
        triangles_warn=1, triangles_fail=2,
        vertices_warn=10_000, materials_warn=8,
        texture_px_warn=4096, texture_bytes_warn=64 * 1024 * 1024,
        size_min_m=0.01, size_max_m=200.0,
    )
    monkeypatch.setitem(readiness.PROFILES, "_w4f1_low_tri", low)
    # A box triangulates to 12 triangles -- past both thresholds.
    report = readiness.validate(bd.ClayDoc(objects=[_grounded_box()]), profile="_w4f1_low_tri")
    tri = _by_key(report)["triangles"]
    assert tri.status == "warn"
    assert tri.fix == "decimate"
    # Every check but the two the manual names must never be "fail".
    for check in report.checks:
        if check.key not in ("objects", "uvs"):
            assert check.status != "fail", f"{check.key} reached fail"


# --- clay-mesh-model-04 --------------------------------------------------------


def test_a_pair_with_no_shared_grid_cell_is_not_reported_exact_with_a_vertex_distance():
    """A 10x0.1x10 slab on the ground and a unit cube hovering 2.45m above it:
    at ``near=0.5`` (so the triangle grid's cell size is 0.5m) no triangle
    from either object shares a grid cell with one from the other, so
    ``_pair_analysis`` falls to the vertex-sampled fallback -- the same
    fallback the two branches above it (grid-registration overflow,
    MAX_TRIANGLE_PAIRS overflow) both mark ``exact=False``. Unfixed, this
    third branch alone marked the same kind of approximate answer
    ``exact=True``."""
    slab = _obj(replace(prim.box(), uv=None), translation=(0.0, 0.0, 0.0), scale=(10.0, 0.1, 10.0))
    cube = _obj(replace(prim.box(), uv=None), translation=(0.0, 3.0, 0.0))

    geom_slab = az._geometry(slab)
    geom_cube = az._geometry(cube)
    closed_slab = az._is_closed(slab.mesh)
    closed_cube = az._is_closed(cube.mesh)

    row, was_truncated = az._pair_analysis(
        slab, cube, geom_slab, geom_cube, closed_slab, closed_cube,
        contact_tol=0.001, near=0.5,
        overlap_budget=[az.MAX_OVERLAP_BOOLEANS],
        triangle_pair_budget=[az.MAX_TRIANGLE_PAIRS],
    )
    assert row.exact is False
    assert row.intersects is False
    assert row.distance is not None


# --- clay-mesh-model-05 --------------------------------------------------------


def test_analyze_narrow_phase_total_candidate_pairs_are_bounded_across_object_pairs():
    """Unfixed, ``MAX_TRIANGLE_PAIRS`` only ever bounded one object pair's own
    candidate count -- the vectorised narrow phase measures ~4s per pair at
    that ceiling, and ``MAX_ANALYZE_OBJECTS`` (64) admits up to 2,016 pairs in
    one ``analyze()`` call, so nothing stopped a call from paying for the
    narrow phase thousands of times over. Three overlapping boxes give two
    pairs of 120 candidates apiece; sharing a 150-wide budget across both
    calls (still under the per-pair ceiling on each call alone) must spend it
    on the first and fall back to the honest 'unknown' on the second."""
    box = replace(prim.box(), uv=None)
    a = _obj(box, translation=(0.0, 0.0, 0.0))
    b = _obj(box, translation=(0.3, 0.3, 0.3))
    c = _obj(box, translation=(0.6, 0.6, 0.6))
    geoms = {o.uid: az._geometry(o) for o in (a, b, c)}
    closed = {o.uid: az._is_closed(o.mesh) for o in (a, b, c)}
    near, contact_tol = 1.0, 0.001

    overlap_budget = [az.MAX_OVERLAP_BOOLEANS]
    triangle_pair_budget = [150]
    row1, _ = az._pair_analysis(
        a, b, geoms[a.uid], geoms[b.uid], closed[a.uid], closed[b.uid],
        contact_tol, near, overlap_budget, triangle_pair_budget,
    )
    row2, _ = az._pair_analysis(
        a, c, geoms[a.uid], geoms[c.uid], closed[a.uid], closed[c.uid],
        contact_tol, near, overlap_budget, triangle_pair_budget,
    )
    assert row1.exact is True, "the first pair alone is well under the shared budget"
    assert row2.exact is False, "the shared budget was spent by the first pair"
    assert triangle_pair_budget[0] >= 0


# --- clay-mesh-model-06 --------------------------------------------------------


def test_findings_refuses_an_oversized_mesh_before_walking_its_boundary_loops(monkeypatch):
    """Unfixed, ``findings()`` was ``return rows_for(mesh, check_manifold(mesh))``
    -- ``check_manifold`` (and, for an open mesh, the uncapped
    ``boundary_loops`` walk inside ``rows_for``) ran unconditionally, and only
    a later ``ops_clean`` call deep inside ``rows_for`` ever refused past
    ``MAX_CLEAN_CORNERS``. A plane (4 corners, open) past a monkeypatched
    ceiling of 3 must refuse before ``check_manifold`` is ever called."""
    monkeypatch.setattr(oc, "MAX_CLEAN_CORNERS", 3)
    calls = []
    orig_check_manifold = dg.check_manifold

    def spy(mesh):
        calls.append(1)
        return orig_check_manifold(mesh)

    monkeypatch.setattr(dg, "check_manifold", spy)

    mesh = prim.plane()  # 4 corners, open -- boundary_loops would run too
    with pytest.raises(el.OpError, match="past the 3"):
        dg.findings(mesh)
    assert not calls, "check_manifold ran before the size ceiling was checked"


# --- clay-mesh-model-07 --------------------------------------------------------


def test_bisect_through_a_coplanar_face_keeps_it_once_when_nothing_is_cleared():
    """Cutting a box exactly through its own top face (``clear=0``, keep both
    sides): every corner of that face satisfies both ``s >= -eps`` and
    ``s <= eps``, so it built an identical ``f_front``/``f_back`` and,
    unfixed, was emitted into both -- one extra face out of a clean cut."""
    box = prim.box()
    faces = el.ElementSel(faces=np.arange(face_count(box), dtype="i4"))
    out, _sel = om.bisect(box, faces, point=(0.0, 0.5, 0.0), normal=(0.0, 1.0, 0.0), clear=0)
    assert face_count(out) == face_count(box), "the coplanar top face must not double"


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


# --- clay-mesh-model-08 --------------------------------------------------------


def test_one_oversized_object_does_not_skip_geometry_checks_for_the_rest_of_the_document(
    monkeypatch,
):
    """Unfixed, ``oversized`` gated ``geometry``/``normals``/``closed`` as one
    document-wide switch: *one* object past ``MAX_CLEAN_CORNERS`` skipped
    those three checks for *every* visible object, not just itself -- not the
    per-object skip the module docstring documents (and the sibling
    ``vertices`` check, which already keeps measuring the rest of the
    document, already gives)."""
    monkeypatch.setattr(oc, "MAX_CLEAN_CORNERS", 10)
    # A box (24 corners) is past the patched ceiling; a 6-corner two-triangle
    # mesh (one triangle duplicated) stays under it and is a real,
    # surveyable geometry defect on the *other* object.
    huge = _grounded_box()

    dup = Mesh(
        positions=np.array([[0.0, 0.5, 0.0], [1.0, 0.5, 0.0], [0.0, 1.5, 0.0]], dtype="f4"),
        loops=np.array([0, 1, 2, 0, 1, 2], dtype="i4"),
        starts=np.array([0, 3, 6], dtype="i4"),
        material=np.zeros(2, dtype="i4"),
        smooth=np.zeros(2, dtype=bool),
    )
    defect_obj = _obj(dup)
    assert len(dup.loops) <= 10, "the defect object must stay under the patched ceiling"

    doc = bd.ClayDoc(objects=[huge, defect_obj])
    report = readiness.validate(doc)
    checks = _by_key(report)

    geometry = checks["geometry"]
    assert geometry.status == "warn", "the small object's own defect must still be measured"
    assert defect_obj.uid in geometry.uids
    assert huge.uid not in geometry.uids
    assert "1 object(s) skipped" in geometry.message
