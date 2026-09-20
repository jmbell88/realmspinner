"""Spin and screw: lathe a profile of edges into new topology."""

from __future__ import annotations

import numpy as np
import pytest

from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import ops_spin as osp
from realmspinner.kernels.mesh import primitives as prim

from .topo_asserts import assert_consistently_oriented


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


def _bare_box() -> bm.Mesh:
    """A box with no uv -- ``primitives.box`` always applies its own cube
    projection, so a caller wanting the "no uv anywhere" case has to strip
    it, the same way every other Clay op is tested against both states."""
    box = prim.box()
    return bm.Mesh(
        positions=box.positions,
        loops=box.loops,
        starts=box.starts,
        material=box.material,
        smooth=box.smooth,
    )


# A single vertical edge of a unit box, offset from the Y axis -- the
# smallest possible "profile" that already belongs to some face, since a
# Mesh stores no wire edges (see ops_topo's own module docstring). Vertex 0
# is (-.5, -.5, -.5) and vertex 4 is (-.5, .5, -.5).
_PROFILE = el.ElementSel(edges=np.array([[0, 4]], dtype="i4"))


# --- _profile_order -----------------------------------------------------


def test_profile_order_walks_an_open_chain_from_one_end() -> None:
    edges = np.array([[0, 1], [1, 2], [2, 3]], dtype="i8")
    order, closed = osp._profile_order(edges)
    assert not closed
    assert order in ([0, 1, 2, 3], [3, 2, 1, 0])


def test_profile_order_walks_a_closed_loop() -> None:
    edges = np.array([[0, 1], [1, 2], [2, 0]], dtype="i8")
    order, closed = osp._profile_order(edges)
    assert closed
    assert set(order) == {0, 1, 2}
    assert len(order) == 3


def test_profile_order_refuses_a_fork() -> None:
    edges = np.array([[0, 1], [0, 2], [0, 3]], dtype="i8")
    with pytest.raises(el.OpError, match="forks"):
        osp._profile_order(edges)


def test_profile_order_refuses_two_disjoint_runs() -> None:
    edges = np.array([[0, 1], [5, 6]], dtype="i8")
    with pytest.raises(el.OpError, match="single connected"):
        osp._profile_order(edges)


# --- spin --------------------------------------------------------------


def test_spin_with_no_edges_refuses() -> None:
    box = prim.box()
    with pytest.raises(el.OpError, match="Select the edges"):
        osp.spin(box, el.empty(), axis=1, angle=360.0, steps=8, center=(0.0, 0.0, 0.0))


def test_spin_a_bad_axis_refuses() -> None:
    box = prim.box()
    with pytest.raises(el.OpError, match="Axis"):
        osp.spin(box, _PROFILE, axis=5, angle=360.0, steps=8, center=(0.0, 0.0, 0.0))


def test_spin_zero_steps_refuses() -> None:
    box = prim.box()
    with pytest.raises(el.OpError, match="at least one step"):
        osp.spin(box, _PROFILE, axis=1, angle=360.0, steps=0, center=(0.0, 0.0, 0.0))


def test_spin_zero_angle_refuses() -> None:
    box = prim.box()
    with pytest.raises(el.OpError, match="must not be zero"):
        osp.spin(box, _PROFILE, axis=1, angle=0.0, steps=8, center=(0.0, 0.0, 0.0))


def test_spin_one_edge_a_partial_turn_builds_an_open_band_of_quads() -> None:
    box = prim.box()
    out, sel = osp.spin(box, _PROFILE, axis=1, angle=180.0, steps=6, center=(0.0, 0.0, 0.0))
    bm.validate(out)
    assert len(sel.faces) == 6, "one band per step"
    assert bm.face_count(out) == bm.face_count(box) + 6
    assert len(out.positions) == len(box.positions) + 6 * 2, "one new ring per step"


def test_spin_a_full_turn_reuses_the_first_ring_to_close_the_seam() -> None:
    box = prim.box()
    out, sel = osp.spin(box, _PROFILE, axis=1, angle=360.0, steps=8, center=(0.0, 0.0, 0.0))
    bm.validate(out)
    assert len(sel.faces) == 8, "one band per step, wrapped"
    # Eight distinct rings around, not nine -- the last band reuses ring 0.
    assert len(out.positions) == len(box.positions) + 7 * 2

    new_faces = out.loops[out.starts[bm.face_count(box)] :]
    new_verts = np.unique(new_faces)
    original_profile_verts = {0, 4}
    assert original_profile_verts & set(new_verts.tolist()) == original_profile_verts, (
        "the closing band still reads from the original profile vertices"
    )


def test_spin_a_closed_profile_includes_its_own_wrap_edge() -> None:
    """A closed loop profile spun any amount still builds a band for its own
    closing edge, not just the open-chain edges."""
    box = prim.box()
    ring = el.ElementSel(edges=np.array([[0, 1], [1, 5], [5, 4], [4, 0]], dtype="i4"))
    out, sel = osp.spin(box, ring, axis=1, angle=90.0, steps=3, center=(0.0, 0.0, -2.0))
    bm.validate(out)
    assert len(sel.faces) == 3 * 4, "4 profile edges (including the wrap) x 3 bands"


def test_spin_generates_uv_only_when_the_source_mesh_already_has_it() -> None:
    box = _bare_box()
    out, _ = osp.spin(box, _PROFILE, axis=1, angle=180.0, steps=4, center=(0.0, 0.0, 0.0))
    assert out.uv is None

    box_uv = _uvd(prim.box())
    out_uv, _ = osp.spin(box_uv, _PROFILE, axis=1, angle=180.0, steps=4, center=(0.0, 0.0, 0.0))
    assert out_uv.uv is not None
    assert out_uv.uv.shape == (len(out_uv.loops), 2)


def test_spin_refuses_before_the_ceiling_stalls_the_frame_thread_for_every_bands_pairs_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 2026-09-19 audit's clay-21: `MAX_SPIN_QUADS`'s own comment claimed
    "the cost is exactly that product" (`n_bands * n_pairs`), which this
    fix's own scratch measurement shows is false -- at the identical
    65,536-quad product, a 1-edge-profile spun 65,536 steps measured
    576-658 ms (reproducing the audit's 689-692 ms) while a 256-edge-profile
    spun 256 steps measured only 48 ms, because the per-*band* Python
    overhead (~10us/band) dominates when a band carries few profile edges,
    and the product ceiling alone cannot see that.

    Both a bands-heavy split (many steps, one profile edge) and a
    pairs-heavy split (many profile edges, one step) must refuse once
    `MAX_SPIN_BANDS` is lowered -- the first because `n_bands` alone now
    exceeds it, the second because `n_bands * n_pairs` still does.
    """
    monkeypatch.setattr(osp, "MAX_SPIN_BANDS", 4)
    box = prim.box()
    with pytest.raises(el.OpError, match="past the"):
        osp.spin(box, _PROFILE, axis=1, angle=180.0, steps=5, center=(0.0, 0.0, 0.0))


def test_spin_stays_reachable_under_the_bands_ceiling() -> None:
    """The UI's own `steps` Param caps at 256, well under `MAX_SPIN_BANDS` --
    ordinary use must not have been caught by closing the latent hazard."""
    box = prim.box()
    assert osp.MAX_SPIN_BANDS > 256
    out, sel = osp.spin(box, _PROFILE, axis=1, angle=180.0, steps=256, center=(0.0, 0.0, 0.0))
    bm.validate(out)
    assert len(sel.faces) == 256


def test_spin_new_bands_are_consistently_wound() -> None:
    box = prim.box()
    out, sel = osp.spin(box, _PROFILE, axis=1, angle=270.0, steps=5, center=(0.0, 0.0, 0.0))
    from realmspinner.kernels.mesh import topo

    new_only = topo.take_faces(out, sel.faces)
    assert_consistently_oriented(new_only)


# --- screw ---------------------------------------------------------------


def test_screw_with_no_edges_refuses() -> None:
    box = prim.box()
    with pytest.raises(el.OpError, match="Select the edges"):
        osp.screw(box, el.empty(), axis=1, angle=360.0, steps=8, height=1.0, center=(0.0, 0.0, 0.0))


def test_screw_zero_angle_and_zero_height_refuses() -> None:
    box = prim.box()
    with pytest.raises(el.OpError, match="cannot both be zero"):
        osp.screw(box, _PROFILE, axis=1, angle=0.0, steps=4, height=0.0, center=(0.0, 0.0, 0.0))


def test_screw_never_closes_even_on_a_whole_turn() -> None:
    box = prim.box()
    out, sel = osp.screw(
        box, _PROFILE, axis=1, angle=360.0, steps=8, height=1.0, center=(0.0, 0.0, 0.0)
    )
    bm.validate(out)
    assert len(sel.faces) == 8
    # Every one of the 9 rings is a distinct copy -- never wrapped, unlike spin.
    assert len(out.positions) == len(box.positions) + 8 * 2


def test_screw_lifts_each_ring_along_the_axis() -> None:
    box = prim.box()
    out, sel = osp.screw(
        box, _PROFILE, axis=1, angle=90.0, steps=4, height=2.0, center=(0.0, 0.0, 0.0)
    )
    bm.validate(out)
    new_positions = out.positions[len(box.positions) :]
    # The last ring should sit a full "height" further along Y than the start.
    last_ring_y = sorted(new_positions[-2:, 1].tolist())
    expected = sorted((0.5 + 2.0, -0.5 + 2.0))
    assert np.allclose(last_ring_y, expected, atol=1e-4)


def test_screw_a_zero_angle_pure_extrusion_still_builds_bands() -> None:
    box = prim.box()
    out, sel = osp.screw(
        box, _PROFILE, axis=1, angle=0.0, steps=3, height=3.0, center=(0.0, 0.0, 0.0)
    )
    bm.validate(out)
    assert len(sel.faces) == 3
