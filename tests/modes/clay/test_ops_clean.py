"""``ops_clean``'s two ops (recalculate normals, merge by distance), pinned against
small hand-built meshes.

Every mesh below is built by hand from a :func:`primitives.box` (the one
closed, UV-unwrapped, manifold reference every other test in this file starts
from) rather than from a general-purpose fixture factory, because the point of
each test is *exactly one* defect against an otherwise clean cube -- so an
exact array comparison after an op is the proof it paid only that debt and
nothing else.
"""

from __future__ import annotations

import numpy as np
import pytest

from realmspinner.kernels.mesh import adjacency as adj
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import ops_clean as oc
from realmspinner.kernels.mesh import ops_topo
from realmspinner.kernels.mesh import primitives as prim

# --- mesh builders -----------------------------------------------------------


def _box() -> bm.Mesh:
    return prim.box()


def _with_reversed_face(mesh: bm.Mesh, face: int = 0) -> bm.Mesh:
    """*mesh* with one face's winding reversed -- a single flipped face."""
    out, _sel = ops_topo.flip_normals(mesh, el.ElementSel(faces=np.array([face], dtype="i4")))
    return out


def _fully_inverted(mesh: bm.Mesh) -> bm.Mesh:
    """*mesh* with every face reversed -- an inside-out shell, but internally
    consistent (every pair of neighbours still agrees with each other)."""
    out, _sel = ops_topo.flip_normals(mesh, el.ElementSel())
    return out


def _two_cubes_offset(mesh: bm.Mesh, eps: float = 1e-7) -> bm.Mesh:
    """Two copies of *mesh*, the second nudged by *eps* -- two shells whose
    corresponding vertices are coincident within any distance past *eps*."""
    b = bm.Mesh(
        positions=mesh.positions + eps,
        loops=mesh.loops,
        starts=mesh.starts,
        material=mesh.material,
        smooth=mesh.smooth,
        uv=mesh.uv,
    )
    n_a = len(mesh.positions)
    positions = np.vstack([mesh.positions, b.positions])
    loops = np.concatenate([mesh.loops.astype("i8"), b.loops.astype("i8") + n_a]).astype("i4")
    starts = np.concatenate(
        [mesh.starts.astype("i8")[:-1], mesh.starts.astype("i8") + len(mesh.loops)]
    ).astype("i4")
    material = np.concatenate([mesh.material, b.material])
    smooth = np.concatenate([mesh.smooth, b.smooth])
    uv = None if mesh.uv is None else np.concatenate([mesh.uv, b.uv])
    return bm.Mesh(
        positions=positions,
        loops=loops,
        starts=starts,
        material=material,
        smooth=smooth,
        uv=uv,
    )


# --- each op fixes only its own defect ---------------------------------------


def test_merge_by_distance_merges_only_the_coincident_vertices() -> None:
    box = _box()
    mesh = _two_cubes_offset(box, eps=1e-7)
    out = oc.merge_by_distance(mesh, 1e-5)
    assert len(out.positions) == len(box.positions)
    assert bm.face_count(out) == 2 * bm.face_count(box)
    assert oc.merge_by_distance(out, 1e-5) is out, "nothing is left to merge"


def test_recalc_outside_fixes_only_the_winding() -> None:
    box = _box()
    out = oc.recalc_outside(_fully_inverted(box))
    assert np.array_equal(out.positions, box.positions)
    assert np.array_equal(out.loops, box.loops)
    assert np.array_equal(out.uv, box.uv)
    assert len(adj.adjacency(out).flipped_pairs) == 0
    assert _signed_volume(out) > 0.0


# --- identity: each op is a no-op when nothing needs fixing ------------------


def test_recalc_outside_returns_the_identical_object_when_nothing_is_wrong() -> None:
    box = _box()
    assert oc.recalc_outside(box) is box


def test_merge_by_distance_returns_the_identical_object_when_nothing_merges() -> None:
    box = _box()
    assert oc.merge_by_distance(box, 1e-5) is box


@pytest.mark.parametrize("name", sorted(prim.GENERATORS))
def test_recalc_outside_on_a_clean_primitive_returns_the_same_object(name: str) -> None:
    defaults, builder = prim.GENERATORS[name]
    mesh = builder(**defaults)
    assert oc.recalc_outside(mesh) is mesh


# --- recalc_outside: the winding claims themselves ---------------------------


def test_recalc_outside_leaves_every_closed_shell_with_positive_volume() -> None:
    inverted = _fully_inverted(_box())
    assert _signed_volume(inverted) < 0.0
    fixed = oc.recalc_outside(inverted)
    assert _signed_volume(fixed) > 0.0
    assert _signed_volume(_box()) == pytest.approx(_signed_volume(fixed))


def test_recalc_outside_leaves_every_manifold_edge_traversed_oppositely() -> None:
    inverted = _fully_inverted(_box())
    fixed = oc.recalc_outside(inverted)
    a = adj.adjacency(fixed)
    two_use = a.edge_uses == 2
    assert two_use.all(), "a cube has no boundary or non-manifold edge"
    assert len(a.flipped_pairs) == 0
    assert (a.twin >= 0).all()


def _signed_volume(mesh: bm.Mesh) -> float:
    # The module-private helper is read directly to pin the geometric claim
    # itself rather than just the winding it implies.
    return float(oc._face_fan_volume(mesh).sum()) / 6.0


def test_flipping_keeps_uv_rows_attached_to_their_corners() -> None:
    """A face that gets flipped and then flipped back by ``recalc_outside``
    must come back with *exactly* the uv rows it started with -- the round
    trip only reproduces the original array if each corner's own uv followed
    it through both reversals rather than being dropped or left in place."""
    box = _box()
    assert box.uv is not None, "box() is UV-unwrapped; this claim needs real uvs"
    one_flipped = _with_reversed_face(box, face=0)
    assert not np.array_equal(one_flipped.uv, box.uv), "the flip must have moved something"
    fixed = oc.recalc_outside(one_flipped)
    assert np.array_equal(fixed.uv, box.uv)
    assert np.array_equal(fixed.loops, box.loops)


# --- the size ceiling ---------------------------------------------------------


def test_recalc_normals_refuses_a_mesh_past_the_clean_corner_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 2026-09-19 audit's clay-14: `recalc_outside` runs a BFS/adjacency/volume
    pass over every corner, and `modes/clay/ops.py`'s `_recalc_normals` (the
    **Recalculate Normals** menu item) calls it directly, so it carries the
    ceiling itself."""
    monkeypatch.setattr(oc, "MAX_CLEAN_CORNERS", 4)
    with pytest.raises(el.OpError, match="past the"):
        oc.recalc_outside(_fully_inverted(_box()))


def test_recalc_normals_runs_under_the_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(oc, "MAX_CLEAN_CORNERS", 10_000)
    box = _box()
    out = oc.recalc_outside(_fully_inverted(box))
    bm.validate(out)
    assert len(adj.adjacency(out).flipped_pairs) == 0


# --- every op's output validates ---------------------------------------------


@pytest.mark.parametrize(
    "mesh_factory",
    [
        lambda: _two_cubes_offset(_box()),
        lambda: _fully_inverted(_box()),
        lambda: _with_reversed_face(_box()),
    ],
)
def test_recalc_and_merge_outputs_always_validate(mesh_factory) -> None:
    mesh = mesh_factory()
    bm.validate(oc.recalc_outside(mesh))
    bm.validate(oc.merge_by_distance(mesh, 1e-5))


@pytest.mark.parametrize("name", sorted(prim.GENERATORS))
def test_individual_op_outputs_validate_on_every_primitive(name: str) -> None:
    defaults, builder = prim.GENERATORS[name]
    mesh = builder(**defaults)
    bm.validate(oc.merge_by_distance(mesh, 1e-5))
    bm.validate(oc.recalc_outside(mesh))
