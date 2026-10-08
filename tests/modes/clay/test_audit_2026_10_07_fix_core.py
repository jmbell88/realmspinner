"""The 2026-10-07 Clay audit's mesh-core findings: clay-14 (typed move), 17, 18, 19, 51, 52.

Each test's name is the claim, and each fails against the code before the fix.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from realmspinner.kernels.mesh import adjacency, drag, measure, select
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio.modes.clay import element_move as em

# --- clay-14 ------------------------------------------------------------------


def _doc_with_selected_box() -> tuple[bd.ClayDoc, bd.Obj]:
    doc = bd.ClayDoc()
    obj = bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box())
    doc.add_object(obj)
    doc.set_element_mode("vertex")
    doc.set_element_sel(obj.uid, el.select_all(obj.mesh, "vertex"))
    return doc, obj


def test_the_typed_drag_value_is_capped_so_a_forty_digit_number_cannot_be_entered() -> None:
    d = drag.DragInput()
    for ch in "9" * 40:
        d.key(ch)
    assert len(d.typed) <= 12
    # Backspace still works at the cap, so a mistyped digit is recoverable.
    d.key("backspace")
    assert d.key("1") is True
    assert len(d.typed) <= 12


def test_a_typed_move_too_large_for_float32_is_refused_and_leaves_the_mesh_finite() -> None:
    doc, obj = _doc_with_selected_box()
    before = doc.by_uid(obj.uid).mesh
    head = doc.history.head
    # A value no typed string can reach any more, but the door must not rely on
    # that: a gizmo, a pane field or an agent reaches it too.
    assert em.move_elements(doc, np.array([1e40, 0.0, 0.0])) is False
    after = doc.by_uid(obj.uid).mesh
    assert after is before
    assert np.isfinite(after.positions).all()
    assert doc.history.head == head


def test_commit_positions_reports_the_refusal_by_uid_for_a_beyond_float32_position() -> None:
    doc, obj = _doc_with_selected_box()
    mesh = doc.by_uid(obj.uid).mesh
    bad = np.array(mesh.positions, dtype="f8")
    bad[0, 0] = 1e40
    pushed, unchanged, refused = em.commit_positions(doc, {obj.uid: (mesh, bad)})
    assert pushed is False and unchanged == []
    assert [uid for uid, _ in refused] == [obj.uid]
    assert isinstance(refused[0][1], el.OpError)
    assert np.isfinite(doc.by_uid(obj.uid).mesh.positions).all()


def test_commit_positions_never_hands_a_beyond_float32_position_to_set_mesh() -> None:
    """The door refuses on its own, whatever the document does behind it."""

    class Doc:
        history = None
        calls: list[object] = []

        def by_uid(self, uid):
            return object()

        def element_sel_of(self, uid):
            return None

        def set_mesh(self, uid, mesh, select=None):
            self.calls.append(mesh)

    mesh = bp.box()
    bad = np.array(mesh.positions, dtype="f8")
    bad[0, 0] = 1e40
    doc = Doc()
    _pushed, _unchanged, refused = em.commit_positions(doc, {1: (mesh, bad)})
    assert doc.calls == []
    assert [uid for uid, _ in refused] == [1]


def test_commit_positions_refuses_a_non_finite_position() -> None:
    doc, obj = _doc_with_selected_box()
    mesh = doc.by_uid(obj.uid).mesh
    bad = np.array(mesh.positions, dtype="f8")
    bad[1, 1] = np.nan
    pushed, _unchanged, refused = em.commit_positions(doc, {obj.uid: (mesh, bad)})
    assert pushed is False
    assert len(refused) == 1


def test_a_large_but_representable_typed_move_still_commits() -> None:
    doc, obj = _doc_with_selected_box()
    assert em.move_elements(doc, np.array([999999999999.0, 0.0, 0.0])) is True
    assert np.isfinite(doc.by_uid(obj.uid).mesh.positions).all()


# --- clay-17 ------------------------------------------------------------------


@pytest.mark.parametrize("size", [0.6, 1.0, 0.7, 0.1])
def test_bounds_query_selects_everything_when_the_box_is_exactly_the_objects_extent_for_a_non_f4_exact_size(  # noqa: E501
    size: float,
) -> None:
    m = bp.box((size, size, size))
    lo, hi = -size / 2, size / 2
    q = select.QUERIES["bounds"].run(m, [lo] * 3, [hi] * 3)
    assert len(q.verts) == 8
    assert len(q.faces) == 6
    assert len(select.faces_in_bounds(m, [lo] * 3, [hi] * 3)) == 6


def test_a_bounds_box_still_excludes_what_is_clearly_outside() -> None:
    m = bp.box((1.0, 1.0, 1.0))
    q = select.QUERIES["bounds"].run(m, [-0.5] * 3, [0.0, 0.5, 0.5])
    assert len(q.verts) == 4
    assert len(q.faces) == 1


def test_points_in_box_is_the_one_helper_and_ignores_f4_rounding() -> None:
    pts = np.array([[0.3, 0.3, 0.3]], dtype="f4").astype("f8")
    assert bool(select.points_in_box(pts, [0.3] * 3, [0.3] * 3)[0])
    assert not bool(select.points_in_box(pts, [0.31] * 3, [1.0] * 3)[0])


# --- clay-18 ------------------------------------------------------------------


def test_face_area_triangulates_once_for_two_calls_on_the_same_mesh(monkeypatch) -> None:
    mesh = bp.uv_sphere()
    calls = {"n": 0}
    real = adjacency.triangulate

    def counting(m):
        calls["n"] += 1
        return real(m)

    monkeypatch.setattr(adjacency, "triangulate", counting)
    faces = np.arange(bm.face_count(mesh))
    a = measure.face_area(mesh, faces)
    b = measure.face_area(mesh, faces)
    assert a == b
    assert calls["n"] == 1


# --- clay-19 ------------------------------------------------------------------


def _flipped(mesh: bm.Mesh, face: int) -> bm.Mesh:
    loops = mesh.loops.copy()
    lo, hi = int(mesh.starts[face]), int(mesh.starts[face + 1])
    loops[lo:hi] = loops[lo:hi][::-1]
    return bm.Mesh(mesh.positions, loops, mesh.starts, mesh.material, mesh.smooth)


def _reference_volume_if_closed(mesh: bm.Mesh) -> float | None:
    """The pre-fix answer, spelled with the full manifold report."""
    if len(mesh.starts) <= 1:
        return None
    report = adjacency.check_manifold(mesh)
    if len(report.boundary_edges) or len(report.nonmanifold_edges):
        return None
    if len(report.flipped_edges):
        return None
    return measure.volume(mesh)


def _broken_meshes() -> list[bm.Mesh]:
    box = bp.box()
    out = [
        _flipped(box, 0),
        _flipped(bp.uv_sphere(), 3),
        bp.plane(),
        bm.from_faces(np.zeros((0, 3), dtype="f4"), []),
    ]
    # Two boxes sharing one face's edges: three faces on an edge -> non-manifold.
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, -1, 0], [0, 0, -1]], dtype="f4")
    out.append(bm.from_faces(pos, [[0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 1, 5]]))
    # Open: a box minus its last face.
    open_box = bm.Mesh(
        box.positions,
        box.loops[: box.starts[-2]],
        box.starts[:-1],
        box.material[:-1],
        box.smooth[:-1] if len(box.smooth) == bm.face_count(box) else box.smooth,
    )
    out.append(open_box)
    return out


def test_volume_if_closed_matches_the_report_answer_over_generators_and_broken_meshes() -> None:
    names = [n for _, group in bp.CLAY_GENERATORS for n in group]
    meshes = [bp.GENERATORS[name][1]() for name in names] + _broken_meshes()
    assert len(meshes) > 15
    for mesh in meshes:
        want = _reference_volume_if_closed(mesh)
        got = measure.volume_if_closed(mesh)
        if want is None:
            assert got is None
        else:
            assert got == pytest.approx(want, rel=1e-12)
        assert measure.is_closed(mesh) == (
            len(mesh.starts) > 1
            and not len(adjacency.check_manifold(mesh).boundary_edges)
            and not len(adjacency.check_manifold(mesh).nonmanifold_edges)
        )


def test_volume_if_closed_does_not_build_the_duplicate_face_report(monkeypatch) -> None:
    def boom(_mesh):
        raise AssertionError("check_manifold builds the duplicate-face report")

    monkeypatch.setattr(adjacency, "check_manifold", boom)
    monkeypatch.setattr(measure, "check_manifold", boom, raising=False)
    assert measure.volume_if_closed(bp.box((1.0, 1.0, 1.0))) == pytest.approx(1.0)
    assert measure.volume_if_closed(_flipped(bp.box(), 0)) is None


def test_the_closed_verdict_is_built_once_per_mesh(monkeypatch) -> None:
    mesh = bp.box()
    calls = {"n": 0}
    real = adjacency._build

    def counting(m):
        calls["n"] += 1
        return real(m)

    monkeypatch.setattr(adjacency, "_build", counting)
    for _ in range(3):
        measure.volume_if_closed(mesh)
        measure.is_closed(mesh)
    assert calls["n"] == 1


# --- clay-51 ------------------------------------------------------------------

_PKG = Path(bm.__file__).parent
_STALE = re.compile(
    r"\bclay_view\b|\bclay_ops\b|\bclay_mode\b|\bviewer\.gltf\b|\bviewer\.math3d\b"
    r"|(?<!studio/)viewer/\s*picking"
)


@pytest.mark.parametrize("name", ["pick.py", "mesh.py", "selection.py", "select.py"])
def test_mesh_core_docstrings_cite_no_module_from_the_old_flat_layout(name: str) -> None:
    text = (_PKG / name).read_text(encoding="utf-8")
    assert _STALE.findall(text) == []


# --- clay-52 ------------------------------------------------------------------


def test_every_public_select_verb_is_reachable_from_an_op_or_a_query() -> None:
    assert not hasattr(select, "grow")
    src = Path(select.__file__).read_text(encoding="utf-8")
    assert "def grow" not in src


# --- Duplicate against the object ceiling (verification of the clay-13 landing) --


def test_duplicate_past_the_object_ceiling_toasts_and_changes_nothing(monkeypatch) -> None:
    from realmspinner.kernels.mesh import glbimport
    from realmspinner.studio.modes.clay import ops as clay_ops

    class Ctx:
        def __init__(self) -> None:
            self.toasts: list[str] = []

        def toast(self, text: str, level: str = "info") -> None:
            self.toasts.append(text)

    doc = bd.ClayDoc()
    for name in ("A", "B"):
        doc.add_object(bd.Obj(uid=bd.new_uid(), name=name, mesh=bp.box()))
    doc.selection.update(o.uid for o in doc.objects)
    monkeypatch.setattr(glbimport, "MAX_OBJECTS", 3)
    head = doc.history.head
    ctx = Ctx()
    assert clay_ops.run(ctx, doc, clay_ops.get("duplicate")) is False
    assert len(doc.objects) == 2
    assert doc.history.head == head
    assert len(ctx.toasts) == 1 and "objects" in ctx.toasts[0]
