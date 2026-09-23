"""Regression tests for the 2026-09-23 audit (second run)'s clay-10, clay-11,
clay-12 and clay-18 -- the "clay-kernels" fixer brief, which owns
``kernels/mesh/ops_topo.py``, ``kernels/mesh/ops_spin.py``,
``kernels/mesh/colliders.py`` and ``kernels/mesh/objexport.py``.

clay-13, clay-14 (``kernels/geom3d/gltf.py``) live in
``tests/kernels/geom3d/test_audit_2026_09_23b_gltf.py`` instead, next to the
module they cover. clay-17 (``kernels/mesh/uvunwrap.py``) needed no new test:
the exact claim -- ``MAX_UV_ISLANDS`` refusing before any island is solved --
is already pinned by
``tests/modes/clay/test_audit_2026_09_23_ops.py::
test_unwrap_lscm_refuses_the_island_count_before_solving_any_island``,
written the same day for the first run's clay-05 (the same fix, found twice).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from realmspinner.kernels.mesh import colliders as cl
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import objexport
from realmspinner.kernels.mesh import ops_spin as osp
from realmspinner.kernels.mesh import ops_topo as ops
from realmspinner.kernels.mesh import primitives as prim
from realmspinner.kernels.mesh.elements import OpError

# --- clay-10: Extrude edges / vertices ---------------------------------------


def _all_boundary_edges(mesh) -> np.ndarray:
    from realmspinner.kernels.mesh import adjacency as adj

    rings, _ = adj.boundary_loops(mesh)
    return np.concatenate([np.stack([r, np.roll(r, -1)], axis=1) for r in rings])


def test_extrude_edges_refuses_past_its_own_size_ceiling(monkeypatch) -> None:
    """The 2026-09-23 audit, finding clay-10: unlike every sibling growth op
    in this module (``extrude_faces``' ``MAX_EXTRUDE_CORNERS``, ``inset``'s
    ``MAX_INSET_CORNERS``...), ``extrude_edges`` had no ceiling at all --
    2.05s at 2,000,000 boundary edges. The fix must refuse from the
    selection's own edge count, before ``_boundary_owner`` ever builds
    :func:`~.adjacency.adjacency` -- proved here by making ``adjacency``
    itself an assertion failure, so the test fails loudly if the refusal
    has not moved ahead of it.
    """
    monkeypatch.setattr(ops, "MAX_EXTRUDE_EDGE_CORNERS", 8)  # 2 edges' worth

    def _boom(mesh):
        raise AssertionError("adjacency() ran before the size refusal")

    monkeypatch.setattr(ops, "adjacency", _boom)

    mesh = prim.plane()
    edges = _all_boundary_edges(mesh)  # 4 boundary edges, over the ceiling of 2
    with pytest.raises(OpError, match="past the"):
        ops.extrude_edges(mesh, el.ElementSel(edges=edges))


def test_extrude_edges_stays_reachable_under_its_own_ceiling() -> None:
    """Ordinary use (a handful of boundary edges) must not have been caught
    by closing clay-10's hazard -- the same "still reachable" sanity every
    sibling ceiling in this module is paired with."""
    mesh = prim.plane()
    edges = _all_boundary_edges(mesh)
    assert 4 * len(edges) < ops.MAX_EXTRUDE_EDGE_CORNERS
    out, sel = ops.extrude_edges(mesh, el.ElementSel(edges=edges))
    assert len(sel.edges) == 4


# --- clay-11: Spin / Screw ----------------------------------------------------


def test_spin_refuses_a_long_profile_before_profile_order_walks_the_whole_selection(
    monkeypatch,
) -> None:
    """The 2026-09-23 audit, finding clay-11: ``_refuse_spin_size`` used to
    run only after ``_validate_profile`` -> ``_profile_order`` had already
    walked the whole selection in a plain Python loop -- 0.68s wasted on a
    500k-edge profile before the refusal ever fired. The fix refuses on the
    selection's own (cheap, exact) edge count first; proved the same way
    ``test_unwrap_lscm_refuses_the_island_count_before_solving_any_island``
    proves its own reorder, by making the walk itself an assertion failure.
    """
    monkeypatch.setattr(osp, "MAX_SPIN_QUADS", 5)

    def _boom(mesh, sel):
        raise AssertionError("_validate_profile ran before the cheap edge-count refusal")

    monkeypatch.setattr(osp, "_validate_profile", _boom)

    box = prim.box()
    # 10 edges is not a real walkable profile (duplicates, no connectivity) --
    # it does not need to be, since the refusal below must fire before
    # anything ever tries to walk it.
    edges = np.array(list(itertools.combinations(range(8), 2))[:10], dtype="i4")
    with pytest.raises(OpError, match="past the"):
        osp.spin(
            box, el.ElementSel(edges=edges), axis=1, angle=180.0, steps=1, center=(0.0, 0.0, 0.0)
        )


def test_screw_refuses_a_long_profile_before_profile_order_walks_the_whole_selection(
    monkeypatch,
) -> None:
    """Same fix, the other op in this module -- ``screw`` shares
    ``_validate_profile``/``_refuse_spin_size`` with ``spin`` and needed the
    identical reorder."""
    monkeypatch.setattr(osp, "MAX_SPIN_QUADS", 5)

    def _boom(mesh, sel):
        raise AssertionError("_validate_profile ran before the cheap edge-count refusal")

    monkeypatch.setattr(osp, "_validate_profile", _boom)

    box = prim.box()
    edges = np.array(list(itertools.combinations(range(8), 2))[:10], dtype="i4")
    with pytest.raises(OpError, match="past the"):
        osp.screw(
            box,
            el.ElementSel(edges=edges),
            axis=1,
            angle=90.0,
            steps=1,
            height=1.0,
            center=(0.0, 0.0, 0.0),
        )


def test_spin_stays_reachable_under_its_own_ceiling_after_the_reorder() -> None:
    """Ordinary use (``_PROFILE``-shaped selections, well under
    ``MAX_SPIN_QUADS``/``MAX_SPIN_BANDS``) must not have been caught by
    moving the refusal earlier -- the same "still reachable" sanity every
    ceiling in this module is paired with."""
    box = prim.box()
    profile = el.ElementSel(edges=np.array([[0, 4]], dtype="i4"))
    out, sel = osp.spin(box, profile, axis=1, angle=180.0, steps=6, center=(0.0, 0.0, 0.0))
    assert len(sel.faces) == 6


# --- clay-12: Compound collider -----------------------------------------------


def test_compound_with_face_groups_none_refuses_a_large_single_shell_mesh_before_running_face_shells(  # noqa: E501
    monkeypatch,
) -> None:
    """The 2026-09-23 audit, finding clay-12: ``MAX_COMPOUND_PARTS`` refuses
    on the *shell count* ``_face_shells`` hands back, but a single connected
    shell -- one loose part, however many faces -- always reports
    ``n_shells == 1``, so that refusal never fires, and ``_face_shells``
    itself (a plain Python BFS) still walks every face first: 0.28s at
    131,000 faces, linear, unbounded. Proved the same way clay-10 and
    clay-11 are proved: make the walk an assertion failure.
    """
    monkeypatch.setattr(cl, "MAX_COMPOUND_SHELL_FACES", 8)

    def _boom(mesh):
        raise AssertionError("_face_shells ran before the face-count refusal")

    monkeypatch.setattr(cl, "_face_shells", _boom)

    mesh = prim.uv_sphere(segments=16, rings=8)  # 128 faces, one shell -- over the ceiling of 8
    with pytest.raises(OpError, match="past the"):
        cl.compound(mesh, face_groups=None)


def test_compound_with_explicit_face_groups_is_unaffected_by_the_shell_face_ceiling() -> None:
    """The new ceiling only guards the auto-grouping (``face_groups=None``)
    path -- explicit groups never call ``_face_shells`` at all, so a caller
    naming its own parts must not be refused by a limit that exists only to
    bound a BFS it never runs."""
    mesh = prim.uv_sphere(segments=16, rings=8)
    n_faces = len(mesh.starts) - 1
    result = cl.compound(mesh, face_groups=[list(range(n_faces))])
    assert result.kind == "compound"


def test_compound_stays_reachable_under_its_own_shell_face_ceiling() -> None:
    """Ordinary use (a small kitbashed mesh) must not have been caught by
    closing clay-12's hazard."""
    mesh = prim.uv_sphere(segments=16, rings=8)
    n_faces = len(mesh.starts) - 1
    assert n_faces < cl.MAX_COMPOUND_SHELL_FACES
    result = cl.compound(mesh, face_groups=None)
    assert result.kind == "compound"


# --- clay-18: OBJ export / engine winding -------------------------------------


def _parse_obj(obj_text: str) -> tuple[np.ndarray, list[list[int]]]:
    """``(vertex positions, faces as 0-based vertex-index lists)`` from a
    written OBJ, independent of :mod:`.objimport` -- this test wants the
    file's own winding order, not what an importer does with it."""
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    for line in obj_text.splitlines():
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            faces.append([int(tok.split("/")[0]) - 1 for tok in line.split()[1:]])
    return np.array(verts, dtype="f8"), faces


def _face_normal(positions: np.ndarray, face: list[int]) -> np.ndarray:
    a, b, c = positions[face[0]], positions[face[1]], positions[face[2]]
    n = np.cross(b - a, c - a)
    return n / np.linalg.norm(n)


def test_obj_export_reverses_face_winding_for_unitys_mirrored_conversion() -> None:
    """The 2026-09-23 audit, finding clay-18: ``claydoc_to_obj``'s own
    comment claimed "none of today's [engine conversions have a] negative
    determinant" -- false the day it was written. Unity's
    (``engines._unity_obj_conversion``) negates the Z axis alone,
    determinant -1, composed with the object's world matrix before
    :func:`~.mesh.transformed` bakes it -- and that function reverses every
    face loop whenever the composed matrix's determinant is negative (see
    its own docstring). The behaviour was already correct; this pins it: a
    box exported under ``engine="unity"`` must still be outward-wound in
    the engine's own (mirrored) axes, exactly as it is under ``engine=None``
    in glTF's.
    """
    doc = bd.ClayDoc()
    doc.objects.append(bd.Obj(uid=bd.new_uid(), name="Box", mesh=prim.box()))

    obj_none, _ = objexport.claydoc_to_obj(doc, engine=None)
    obj_unity, _ = objexport.claydoc_to_obj(doc, engine="unity")

    for text in (obj_none, obj_unity):
        positions, faces = _parse_obj(text)
        centroid = positions.mean(axis=0)
        for face in faces:
            face_center = positions[face].mean(axis=0)
            normal = _face_normal(positions, face)
            # Outward: the normal and the vector from the mesh centroid to
            # the face both point away from the middle of the box.
            assert np.dot(normal, face_center - centroid) > 0, (
                f"face {face} is wound inward in {'unity' if text is obj_unity else 'none'} export"
            )
