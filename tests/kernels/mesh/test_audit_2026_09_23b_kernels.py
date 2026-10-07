"""Regression tests for the 2026-09-23 audit (second run)'s clay-10 and
clay-18 -- the "clay-kernels" fixer brief, which owns
``kernels/mesh/ops_topo.py`` and ``kernels/mesh/objexport.py``. (clay-11 and
clay-12 covered Spin/Screw and the Compound collider, both cut with the
picoCAD-level Clay.)

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

import numpy as np
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import objexport
from realmspinner.kernels.mesh import ops_topo as ops
from realmspinner.kernels.mesh import primitives as prim
from realmspinner.kernels.mesh.elements import OpError

# --- clay-10: Extrude edges / vertices ---------------------------------------


def _all_boundary_edges(mesh) -> np.ndarray:
    """Every edge used by exactly one face, as (n, 2) vertex pairs."""
    loops = mesh.loops.astype("i8")
    starts = mesh.starts.astype("i8")
    pairs = []
    for f in range(len(starts) - 1):
        ring = loops[starts[f] : starts[f + 1]]
        pairs.extend(zip(ring.tolist(), np.roll(ring, -1).tolist(), strict=True))
    keys = [tuple(sorted(p)) for p in pairs]
    return np.array([k for k in dict.fromkeys(keys) if keys.count(k) == 1], dtype="i4")


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


def test_obj_export_winds_every_face_outward() -> None:
    """The 2026-09-23 audit, finding clay-18: ``claydoc_to_obj`` bakes the
    object's world matrix through :func:`~.mesh.transformed`, which reverses
    every face loop when the composed matrix's determinant is negative. A box
    exported must still be outward-wound in glTF's axes. (The original also
    pinned Unity's mirrored engine conversion; engine profiles are gone.)
    """
    doc = bd.ClayDoc()
    doc.objects.append(bd.Obj(uid=bd.new_uid(), name="Box", mesh=prim.box()))

    obj_text, _ = objexport.claydoc_to_obj(doc)

    positions, faces = _parse_obj(obj_text)
    centroid = positions.mean(axis=0)
    for face in faces:
        face_center = positions[face].mean(axis=0)
        normal = _face_normal(positions, face)
        # Outward: the normal and the vector from the mesh centroid to
        # the face both point away from the middle of the box.
        assert np.dot(normal, face_center - centroid) > 0, f"face {face} is wound inward"
