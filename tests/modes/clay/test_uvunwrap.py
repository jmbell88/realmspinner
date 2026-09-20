"""LSCM: the one angle-preserving unwrap in this package, from a seam plan.

:mod:`.uvtools`'s own test module covers seams/islands/pack/density/stretch
over a uv a mesh already has; everything here is about *producing* one.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.kernels.mesh import mesh as bm
from warlock.kernels.mesh import primitives as prim
from warlock.kernels.mesh import uvtools as ut
from warlock.kernels.mesh import uvunwrap as lscm
from warlock.kernels.mesh.elements import OpError


def _open_tube(segments: int = 12, radius: float = 0.5, height: float = 2.0) -> bm.Mesh:
    """A prism's lateral surface only -- no top or bottom cap -- built by
    hand rather than borrowed from ``primitives`` (another tranche's file):
    two rings of *segments* vertices joined by a band of quads. Genuinely
    developable (every side face is planar, and unrolling the whole band
    gives *segments* congruent rectangles in a row), which is what makes
    "near-zero angle *and* area distortion once flattened" a fact this test
    can assert on exactly, not just a loose bound.

    Vertex *i* and vertex *i + segments* are the same angular column, bottom
    and top -- the vertical mesh edges a caller marks a seam along.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    x, z = radius * np.cos(theta), radius * np.sin(theta)
    bottom = np.stack([x, np.full(segments, -height / 2), z], axis=1)
    top = np.stack([x, np.full(segments, height / 2), z], axis=1)
    positions = np.vstack([bottom, top])
    faces = [
        [i, segments + i, segments + (i + 1) % segments, (i + 1) % segments]
        for i in range(segments)
    ]
    return bm.from_faces(positions, faces)


def _one_vertical_seam(segments: int = 12) -> np.ndarray:
    return np.array([[0, segments]], dtype="i4")


# --- refusals ----------------------------------------------------------


def test_unwrap_lscm_of_an_empty_mesh_is_a_no_op() -> None:
    empty = bm.Mesh(
        positions=np.zeros((0, 3), dtype="f4"), loops=np.zeros(0, dtype="i4"),
        starts=np.zeros(1, dtype="i4"), material=np.zeros(0, dtype="i4"),
        smooth=np.zeros(0, dtype=bool),
    )
    assert lscm.unwrap_lscm(empty, None) is empty


def test_unwrap_lscm_refuses_a_closed_sphere_with_no_seams() -> None:
    """A full uv-sphere has no real mesh boundary anywhere, and cutting no
    edges at all leaves it exactly that way -- there is nothing for the
    solve to flatten onto a plane."""
    sphere = prim.uv_sphere(0.5, 16, 8)
    with pytest.raises(OpError, match="closed surface"):
        lscm.unwrap_lscm(sphere, np.zeros((0, 2), dtype="i4"))


def test_unwrap_lscm_refuses_an_island_past_the_vertex_ceiling(monkeypatch) -> None:
    monkeypatch.setattr(lscm, "MAX_LSCM_VERTICES", 4)
    tube = _open_tube(segments=12)  # 24 vertices, one island, over a ceiling of 4
    with pytest.raises(OpError, match="MAX_LSCM_VERTICES|vertices"):
        lscm.unwrap_lscm(tube, _one_vertical_seam())


# --- the cylinder case named in the plan -------------------------------


def test_unwrap_lscm_of_a_cylinder_with_one_seam_has_no_flipped_faces() -> None:
    tube = _open_tube()
    out = lscm.unwrap_lscm(tube, _one_vertical_seam())
    assert not ut.flipped_uv_faces(out).any()


def test_unwrap_lscm_of_a_cylinder_with_one_seam_has_near_zero_angle_distortion() -> None:
    """A prism's lateral band is exactly developable, so a correct conformal
    (in fact isometric) flattening should read as stretch ~0 on every face,
    not merely "small" -- this is the regression test for the bug this
    module's own docstring names: indexing the solve by raw vertex instead
    of by (vertex, side-of-the-seam) forced the two faces touching the cut
    to agree on a shared uv position for the cut vertex, which showed up as
    roughly 30% per-face area distortion concentrated at exactly those two
    faces. 1e-4 is solver tolerance, not a loosened bar.
    """
    tube = _open_tube()
    out = lscm.unwrap_lscm(tube, _one_vertical_seam())
    assert ut.stretch(out) == pytest.approx(np.zeros(bm.face_count(tube)), abs=1e-4)


def test_unwrap_lscm_result_fits_the_unit_square() -> None:
    tube = _open_tube()
    out = lscm.unwrap_lscm(tube, _one_vertical_seam())
    assert out.uv.min() >= -1e-6
    assert out.uv.max() <= 1.0 + 1e-6


@pytest.mark.parametrize("explicit_pins", [None, [(0, 18)], [(2, 20)]])
def test_unwrap_lscm_is_insensitive_to_which_far_apart_pins_are_chosen(explicit_pins) -> None:
    tube = _open_tube()
    out = lscm.unwrap_lscm(tube, _one_vertical_seam(), pins=explicit_pins)
    assert not ut.flipped_uv_faces(out).any()
    assert ut.stretch(out) == pytest.approx(np.zeros(bm.face_count(tube)), abs=1e-4)


def test_unwrap_lscm_gives_the_two_sides_of_a_seam_vertex_different_uv() -> None:
    """The fix in one sentence: vertex 0 sits on the cut, used by the corner
    that starts the strip (face 0) and the corner that ends it (face 11 of a
    12-segment tube) -- and a correct cut lets those two corners land in
    different places, which is the entire reason the strip can be a
    rectangle rather than a ring pinched shut at one point.
    """
    tube = _open_tube(segments=12)
    out = lscm.unwrap_lscm(tube, _one_vertical_seam(segments=12))
    starts = out.starts.astype("i8")
    face0_corner0 = out.uv[starts[0]]  # face 0's first corner is vertex 0
    face11_last = out.uv[starts[11] + 3]  # face 11's last corner is vertex 0 too
    assert np.abs(face0_corner0 - face11_last).max() > 1e-3


def test_unwrap_lscm_matches_islands_by_seams_grouping() -> None:
    """The islands the solve actually produced (read back via uv-agreement)
    are exactly the islands the seam plan asked for, up to the coincidental
    re-agreement a closed loop's two cut ends cannot avoid at the single
    seam edge itself (see ``uvtools``' own cube test for the same kind of
    coincidence) -- so this checks face count and non-triviality rather than
    exact equality."""
    tube = _open_tube()
    seams = _one_vertical_seam()
    out = lscm.unwrap_lscm(tube, seams)
    by_seam = ut.islands_by_seams(tube, seams)
    assert len(set(by_seam.tolist())) == 1  # one seam does not disconnect the ring
    # The solve did not silently fragment the strip into more pieces than
    # the seam plan called for.
    assert len(set(ut.islands(out).tolist())) <= 1
