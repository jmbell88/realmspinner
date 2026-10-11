"""The 2026-10-11 Clay audit, the clean ops and the legacy migration: clay-02,
clay-09, clay-26.

Each test's name is its claim; each was run against the unfixed code first.
(clay-09's own regression sits beside its sibling in
``test_audit_2026_10_09_r2_model.py``.)
"""

from __future__ import annotations

import numpy as np

from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import ops_clean, ops_topo
from realmspinner.kernels.mesh import primitives as bp


def _tilted(mesh: bm.Mesh, seed: int, shift: float = 0.0) -> bm.Mesh:
    """*mesh* turned by a random rotation and moved -- the plane a user rotated."""
    rng = np.random.default_rng(seed)
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    t = m3.vec3(*(rng.normal(size=3) * 3.0 + shift))
    return bm.transformed(mesh, m3.compose(t, q, m3.vec3(1, 1, 1)))


def _pillow() -> bm.Mesh:
    """One quad and its own reverse sharing the same four vertices: a closed shell
    with no thickness, so its signed volume is round-off and nothing else."""
    quad = bp.plane()
    loops = quad.loops.astype("i8")
    return bm.Mesh(
        positions=quad.positions,
        loops=np.concatenate([loops, loops[::-1]]).astype("i4"),
        starts=np.array([0, 4, 8], dtype="i4"),
        material=np.zeros(2, dtype=quad.material.dtype),
        smooth=np.zeros(2, dtype=quad.smooth.dtype),
        uv=None,
    )


def _bowl() -> bm.Mesh:
    """The lower half of a UV sphere, outward-facing: an open shell with a real
    inside and outside."""
    sph = bp.uv_sphere(0.5, 16, 8)
    centroids = np.array(
        [
            sph.positions[sph.loops[sph.starts[f] : sph.starts[f + 1]]].mean(axis=0)
            for f in range(bm.face_count(sph))
        ]
    )
    top = np.flatnonzero(centroids[:, 1] > 0.05).astype("i4")
    out, _sel = ops_topo.delete_faces(sph, el.ElementSel(faces=top))
    return out


# --- clay-02 --------------------------------------------------------------------


def test_recalc_outside_leaves_a_tilted_flat_open_sheet_exactly_as_it_was() -> None:
    # The audit's probe: a 6x6 grid under a random rotation was "repaired" in 22
    # of 40 trials, because for a flat sheet every face's normal . (face centroid
    # minus shell centroid) is float32 round-off around zero and the vote counted
    # the noise. A sheet that needs nothing comes back as the IDENTICAL object.
    sheet = bp.grid((2.0, 2.0), 6)
    flipped = [
        seed
        for seed in range(40)
        for t in [_tilted(sheet, seed, shift=50.0)]
        if ops_clean.recalc_outside(t) is not t
    ]
    assert flipped == [], f"flat sheets reversed by round-off: seeds {flipped}"


def test_recalc_outside_leaves_a_flat_closed_shell_of_no_volume_exactly_as_it_was() -> None:
    # The closed-shell half of the same defect: the volume sign of a shell with no
    # thickness is noise too.
    pillow = _pillow()
    bm.validate(pillow)
    flipped = [
        seed
        for seed in range(40)
        for t in [_tilted(pillow, seed, shift=50.0)]
        if ops_clean.recalc_outside(t) is not t
    ]
    assert flipped == [], f"zero-volume shells reversed by round-off: seeds {flipped}"


def test_recalc_outside_still_flips_a_genuinely_inward_facing_open_shell() -> None:
    bowl = _bowl()
    assert ops_clean.recalc_outside(bowl) is bowl, "an outward bowl needs nothing"
    inside_out, _sel = ops_topo.flip_normals(bowl, el.ElementSel())
    fixed = ops_clean.recalc_outside(inside_out)
    assert fixed is not inside_out
    assert np.array_equal(fixed.loops, bowl.loops), "every face reversed back"


def test_recalc_outside_still_flips_a_tilted_inward_facing_open_shell() -> None:
    # The floor is relative to the shell's extent, so it holds under any tilt
    # and offset the bowl is carried through.
    bowl = _bowl()
    inside_out, _sel = ops_topo.flip_normals(bowl, el.ElementSel())
    for seed in range(10):
        t = _tilted(inside_out, seed, shift=20.0)
        fixed = ops_clean.recalc_outside(t)
        assert fixed is not t, f"seed {seed}: an inside-out bowl was left alone"
        assert np.array_equal(fixed.loops, _tilted(bowl, seed, shift=20.0).loops)


# --- clay-26 --------------------------------------------------------------------


def test_ops_clean_ceiling_record_describes_the_box_shape_its_table_was_measured_on() -> None:
    doc = ops_clean.__doc__ or ""
    assert "300 * n" not in doc, "the table is not 300 faces a box"
    assert "6 quads" in doc and "4 corners" in doc
    # The table's own arithmetic: 2,000 boxes of 6 faces and 24 corners.
    assert 2_000 * 6 == 12_000 and 2_000 * 24 == 48_000
    assert "48,000" in doc and "12,000" in doc


def test_a_box_has_the_six_faces_and_twenty_four_corners_the_ceiling_table_charges() -> None:
    box = bp.box()
    assert bm.face_count(box) == 6
    assert len(box.loops) == 24
