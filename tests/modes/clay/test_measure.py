"""Tranche 3: measure -- distance, angle, area and volume, pure numbers with
a world matrix passed in rather than composed here.

The known-shape fixture throughout is ``primitives.box()``: a 1x1x1 cube
centred on its own origin, so its numbers are worked out by hand rather than
against another function that could share a bug.
"""

from __future__ import annotations

import numpy as np
import pytest

from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.kernels.mesh import measure
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as bp

# --- distance / angle --------------------------------------------------------


def test_distance_between_two_points() -> None:
    assert measure.distance((0.0, 0.0, 0.0), (3.0, 4.0, 0.0)) == pytest.approx(5.0)


def test_distance_is_zero_for_coincident_points() -> None:
    assert measure.distance((1.0, 1.0, 1.0), (1.0, 1.0, 1.0)) == pytest.approx(0.0)


def test_angle_at_a_right_angle_corner() -> None:
    assert measure.angle((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)) == pytest.approx(90.0)


def test_angle_for_a_straight_line_is_180() -> None:
    assert measure.angle((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (-1.0, 0.0, 0.0)) == pytest.approx(180.0)


def test_angle_is_zero_for_a_degenerate_ray() -> None:
    assert measure.angle((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)) == pytest.approx(0.0)


# --- face_area on a known cube ------------------------------------------------


def test_face_area_of_one_box_face_is_one() -> None:
    box = bp.box()
    # box() is 1x1x1, so every face -- two triangles -- has area 1.
    area = measure.face_area(box, [0])
    assert area == pytest.approx(1.0, abs=1e-5)


def test_face_area_of_the_whole_box_is_six() -> None:
    box = bp.box()
    area = measure.face_area(box, range(bm.face_count(box)))
    assert area == pytest.approx(6.0, abs=1e-5)


def test_face_area_of_an_empty_selection_is_zero() -> None:
    assert measure.face_area(bp.box(), []) == 0.0


def test_face_area_scales_with_a_world_matrix() -> None:
    box = bp.box()
    world = m3.scaling(np.array([2.0, 2.0, 2.0]))
    area = measure.face_area(box, [0], world=world)
    assert area == pytest.approx(4.0, abs=1e-5)  # (2x2 face)


# --- volume on a known cube --------------------------------------------------


def test_volume_of_a_unit_box_is_one() -> None:
    assert measure.volume(bp.box()) == pytest.approx(1.0, abs=1e-5)


def test_volume_scales_with_a_world_matrix() -> None:
    box = bp.box()
    world = m3.scaling(np.array([2.0, 1.0, 1.0]))
    assert measure.volume(box, world=world) == pytest.approx(2.0, abs=1e-5)


def test_volume_of_an_empty_mesh_is_zero() -> None:
    from realmspinner.kernels.mesh import document as bd

    assert measure.volume(bd._empty_mesh()) == 0.0


# --- edge_length ---------------------------------------------------------


def test_edge_length_of_one_box_edge() -> None:
    box = bp.box()
    # Two adjacent corners of the bottom face, one edge unit apart.
    length = measure.edge_length(box, [[0, 1]])
    assert length == pytest.approx(1.0, abs=1e-5)


def test_edge_length_sums_several_edges() -> None:
    box = bp.box()
    length = measure.edge_length(box, [[0, 1], [1, 2]])
    assert length == pytest.approx(2.0, abs=1e-5)


def test_edge_length_of_no_edges_is_zero() -> None:
    assert measure.edge_length(bp.box(), []) == 0.0


def test_edge_length_scales_with_a_world_matrix() -> None:
    box = bp.box()
    world = m3.scaling(np.array([3.0, 1.0, 1.0]))
    length = measure.edge_length(box, [[0, 1]], world=world)
    assert length >= 1.0  # stretched along X, so at least as long
