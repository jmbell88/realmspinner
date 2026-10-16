"""``kernels.mesh.score``: an order for candidate meshes, from their silhouettes alone.

Pure numpy over masks and dicts. The mesh rasteriser is ``meshaudit.view_masks``
and is used here for the two tests that need a real mesh (orientation and the
yaw sweep); everything else is arrays.
"""

from __future__ import annotations

import numpy as np
import trimesh

from realmspinner import meshaudit
from realmspinner.kernels.mesh import score
from realmspinner.service import validation
from realmspinner.vectors import VECTOR_PARAMS


def _glb(tmp_path, mesh, name="m.glb"):
    path = tmp_path / name
    mesh.export(path)
    return path


def _t_shape():
    """A bar across the top and a stem below it: not symmetric top-to-bottom."""
    bar = trimesh.creation.box(extents=(1.0, 0.25, 0.25))
    bar.apply_translation((0.0, 0.875, 0.0))
    stem = trimesh.creation.box(extents=(0.25, 1.5, 0.25))
    return trimesh.util.concatenate([bar, stem])


def _t_picture(side=200):
    """The same T as a picture: row 0 is the top, the bar is at the top."""
    # 100 px per unit, so it has the mesh's own proportions (1.0 wide, 1.75 tall).
    image = np.zeros((side, side), dtype=bool)
    image[10:35, 50:150] = True  # bar
    image[35:185, 87:112] = True  # stem
    return image


def test_score_prefers_mesh_matching_cutout_silhouette(tmp_path):
    """A T made of a bar over a stem matches a picture of that T, and a bar *under* a
    stem (the mesh upside down) does not -- which is what makes the row flip in
    ``view_masks`` load-bearing: without it the two would swap."""
    upright = _glb(tmp_path, _t_shape(), "upright.glb")
    flipped_mesh = _t_shape()
    flipped_mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, (1, 0, 0)))
    flipped = _glb(tmp_path, flipped_mesh, "flipped.glb")
    target = _t_picture()
    directions = score.view_directions()

    good, _ = score.best_view_iou(meshaudit.view_masks(upright, directions), target)
    bad, _ = score.best_view_iou(meshaudit.view_masks(flipped, directions), target)

    assert good is not None and bad is not None
    assert good > 0.85
    assert good > bad + 0.1


def test_trellis_score_is_best_over_yaw_sweep(tmp_path):
    """TRELLIS has no fixed camera-matched yaw (``bench/views.py``): a plank whose long
    side faces +X is only a good match from the side, and the sweep must find it."""
    plank = trimesh.creation.box(extents=(0.3, 0.4, 2.0))
    path = _glb(tmp_path, plank)
    picture = np.zeros((200, 200), dtype=bool)
    picture[80:120, 10:190] = True  # a long horizontal bar

    directions = score.view_directions()
    masks = meshaudit.view_masks(path, directions)
    swept, best = score.best_view_iou(masks, picture)
    front_only = score.silhouette_iou(masks[0], picture)

    assert swept is not None and front_only is not None
    assert swept > 0.85
    # Yaw 0 looks down the plank's short axis; the sweep's best is somewhere else.
    assert best not in (0, None)
    assert swept > front_only + 0.2


def test_the_sweep_is_twenty_four_yaws_by_three_elevations_and_unit_vectors():
    views = score.view_directions()
    assert len(views) == 72
    assert views[0] == (0.0, 0.0, 1.0)
    assert all(abs(sum(c * c for c in v) - 1.0) < 1e-9 for v in views)


def test_normalising_a_mask_ignores_where_it_sits_and_how_big_it_is():
    small = np.zeros((50, 50), dtype=bool)
    small[10:20, 5:15] = True
    big = np.zeros((300, 400), dtype=bool)
    big[100:200, 150:250] = True
    assert score.silhouette_iou(small, big) == 1.0


def test_normalising_does_not_stretch_a_tall_shape_into_a_wide_one():
    tall = np.zeros((100, 100), dtype=bool)
    tall[0:100, 45:55] = True
    wide = np.zeros((100, 100), dtype=bool)
    wide[45:55, 0:100] = True
    assert score.silhouette_iou(tall, wide) < 0.2


def test_an_empty_mask_is_not_measured_rather_than_zero():
    empty = np.zeros((10, 10), dtype=bool)
    full = np.ones((10, 10), dtype=bool)
    assert score.silhouette_iou(empty, full) is None
    assert score.best_view_iou([empty, empty], full) == (None, None)
    assert score.best_view_iou([full], empty) == (None, None)


def _scored(iou, **kw):
    base = {
        "iou": iou, "best_view": 3, "holes": 0.0, "hole_max": 0.05, "components": 1,
        "nonmanifold": 0, "boundary": 0, "budget_ratio": 1.0, "budget_miss": None,
    }
    base.update(kw)
    return score.build(**base)


def test_score_demotes_budget_miss_before_iou():
    """A mesh that missed its budget sorts after every one that met it, however
    well its silhouette matches. Lexicographic, not a blend."""
    missed_but_pretty = _scored(0.99, budget_miss={"requested": 5000, "achieved": 13821})
    met_but_plain = _scored(0.60)
    assert missed_but_pretty["demoted_by"] == ["budget_miss"]
    assert score.rank_key(met_but_plain) < score.rank_key(missed_but_pretty)


def test_holes_past_the_ceiling_demote():
    see_through = _scored(0.99, holes=0.2)
    assert see_through["demoted_by"] == ["holes"]
    assert score.rank_key(_scored(0.5)) < score.rank_key(see_through)
    # At the ceiling is not past it.
    assert _scored(0.9, holes=0.05)["demoted_by"] == []


def test_a_component_cap_is_off_unless_asked_for():
    """The 2026-10-15 sweep: components scale with triangles and do not separate good
    from bad, so the default must not demote on them."""
    assert _scored(0.9, components=50_000)["demoted_by"] == []
    assert _scored(0.9, components=50_000, components_max=10_000)["demoted_by"] == ["components"]


def test_among_equals_iou_decides_and_then_open_edges():
    a = _scored(0.90)
    b = _scored(0.80)
    assert score.rank_key(a) < score.rank_key(b)
    clean = _scored(0.90, nonmanifold=0, boundary=0)
    ragged = _scored(0.90, nonmanifold=10, boundary=4)
    assert score.rank_key(clean) < score.rank_key(ragged)


def test_an_unscored_mesh_sorts_after_every_scored_one():
    worst_scored = _scored(0.01, budget_miss={"requested": 1})
    assert score.rank_key(worst_scored) < score.rank_key(None)
    assert score.rank_key({"version": 999, "iou": 1.0}) == score.rank_key(None)
    assert score.rank_key(_scored(0.5)) < score.rank_key(_scored(None))


def test_mesh_score_is_derived_not_vector():
    """An output of one run, not a setting: a rerun must not inherit it, and it must
    never become a key findings group by."""
    assert "mesh_score" in validation.DERIVED_PARAMS
    assert "mesh_score" not in VECTOR_PARAMS
