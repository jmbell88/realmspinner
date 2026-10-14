"""The 2026-10-09 ``clay`` audit, texturing: clay-06, clay-21, clay-30, clay-31.

Each test's name is the claim.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import numpy as np

from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import uv as bmuv
from realmspinner.kernels.mesh import uvtools
from realmspinner.studio.modes.clay.ui.panes import uv as clay_uv

_MANUAL = Path(__file__).resolve().parents[3] / "docs" / "manual" / "27-clay.md"


def _texture_paragraph() -> str:
    text = " ".join(_MANUAL.read_text(encoding="utf-8").split())
    start = text.index("Every primitive comes with texture coordinates")
    return text[start : text.index("**Box Unwrap**", start)]


def test_the_manual_does_not_call_the_icosphere_box_projected_or_overlap_free() -> None:
    para = _texture_paragraph()
    sentences = re.split(r"(?<=\.)\s+", para)
    projected = [s for s in sentences if "box-projected" in s]
    assert projected, "the paragraph no longer says which shapes are box-projected"
    assert not any("icosphere" in s and "overlap" not in s for s in projected)
    assert not any("icosphere" in s for s in projected if "box-projected instead" in s)
    # The generator really does overlap (lat/long, seam faces translated), so the
    # manual may not file it among the shapes that sit on no other part.
    assert uvtools.overlap_faces(bp.icosphere()).any()
    free = next(s for s in sentences if "so that no two parts" in s)
    assert "icosphere" not in free


def test_every_shape_the_manual_calls_overlap_free_has_zero_overlapping_faces() -> None:
    para = _texture_paragraph()
    free = next(s for s in re.split(r"(?<=\.)\s+", para) if "so that no two parts" in s)
    names = {"cylinder": "cylinder", "cone": "cone", "sphere": "uv_sphere",
             "torus": "torus", "capsule": "capsule"}
    for word, gen in names.items():
        assert word in free
        params, fn = bp.GENERATORS[gen]
        assert not uvtools.overlap_faces(fn(**params)).any(), gen


def test_the_uv_module_docstring_does_not_claim_every_round_primitive_is_overlap_free() -> None:
    doc = " ".join((bmuv.__doc__ or "").split())
    assert "round primitives lay themselves out without overlap" not in doc


def test_a_position_only_mesh_change_reuses_the_uv_pane_measurements(monkeypatch) -> None:
    mesh = bp.box()
    moved = replace(mesh, positions=mesh.positions + np.float32(0.25))
    assert moved is not mesh
    calls = {"n": 0}
    real = uvtools.islands

    def counting(m):
        calls["n"] += 1
        return real(m)

    monkeypatch.setattr(uvtools, "islands", counting)
    state = clay_uv.UvPaneState()
    first = clay_uv._measurements(state, mesh)
    again = clay_uv._measurements(state, moved)
    assert calls["n"] == 1
    assert again is first or again[0] is first[0]


def test_a_uv_edit_still_invalidates_the_uv_pane_measurements() -> None:
    mesh = bp.box()
    state = clay_uv._measurements
    st = clay_uv.UvPaneState()
    first = state(st, mesh)
    uv2 = mesh.uv.copy()
    uv2[:, 0] *= 0.5
    second = state(st, replace(mesh, uv=uv2))
    assert second is not first


def test_a_box_inside_a_single_face_still_catches_that_islands_id() -> None:
    m = bp.plane()
    ids = uvtools.islands(m)
    assert clay_uv.islands_in_rect(m, ids, (0.2, 0.2, 0.6, 0.6)) == {0}


def test_a_box_crossing_a_face_edge_with_no_corner_inside_catches_that_island() -> None:
    m = bp.plane()
    ids = uvtools.islands(m)
    assert clay_uv.islands_in_rect(m, ids, (-0.1, 0.4, 0.3, 0.6)) == {0}
    assert clay_uv.islands_in_rect(m, ids, (0.4, -0.2, 0.6, 1.3)) == {0}  # crosses both edges
    assert clay_uv.islands_in_rect(m, ids, (1.2, 0.2, 1.5, 0.6)) == set()  # clear of it


def test_a_box_beside_a_triangular_face_diagonal_does_not_catch_it() -> None:
    # bbox overlaps but the polygon does not: the hypotenuse side of a right triangle.
    m = bp.wedge()
    uv = np.array([[0, 0], [1, 0], [0, 1]], dtype="f4")
    tri = replace(
        m, positions=m.positions[:3], loops=np.array([0, 1, 2]), starts=np.array([0, 3]),
        material=np.array([0]), smooth=np.array([False]), uv=uv,
    )
    ids = np.array([0])
    assert clay_uv.islands_in_rect(tri, ids, (0.8, 0.8, 0.95, 0.95)) == set()
    assert clay_uv.islands_in_rect(tri, ids, (0.2, 0.2, 0.3, 0.3)) == {0}


def test_the_click_that_commits_a_live_transform_does_not_start_a_marquee() -> None:
    # After the commit ``drag_mode`` is "" while the button that committed it is
    # still held; only a press that armed "move"/"box" owns an active drag.
    assert clay_uv._active_drag_applies("move")
    assert clay_uv._active_drag_applies("box")
    assert not clay_uv._active_drag_applies("")
    assert not clay_uv._active_drag_applies("refused")
    assert not clay_uv._active_drag_applies("rotate")
    assert not clay_uv._active_drag_applies("scale")
