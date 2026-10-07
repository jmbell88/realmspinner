"""The 2026-10-07 audit's mesh-op findings clay-13, clay-25, clay-26, clay-48,
clay-49, clay-67 and clay-68 (the Clay mesh kernels: weld clustering, the UV
island passes, the concave-ring check, pack margins, box unwrap, the join weld
and the comments that cite deleted code).

Each test's name is the claim, and each one fails against the code it was
written for. They drive the kernels directly, so none of them needs a mode.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from realmspinner.kernels.mesh import document as doc
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import ops, ops_dissolve, ops_topo
from realmspinner.kernels.mesh import primitives as prim
from realmspinner.kernels.mesh import uv as uvm
from realmspinner.kernels.mesh import uvtools as ut

_MESH_DIR = Path(ops_topo.__file__).parent
_ROOT = _MESH_DIR.parents[3]

# --- clay-13: weld never fuses a pair farther apart than eps -------------------


def _widest_pair_in_a_cluster(points: np.ndarray, labels: np.ndarray) -> float:
    worst = 0.0
    for label in range(int(labels.max()) + 1):
        members = points[labels == label]
        if len(members) > 1:
            gaps = np.linalg.norm(members[:, None] - members[None], axis=2)
            worst = max(worst, float(gaps.max()))
    return worst


def _two_dimensional_cases(eps: float) -> list[np.ndarray]:
    # A leader with one point either side of it in y: each is within eps of the
    # leader, the two are 1.9 eps apart.
    straddle = np.array(
        [[0, 0, 0], [0.1 * eps, 0.95 * eps, 0], [0.1 * eps, -0.95 * eps, 0]], "f8"
    )
    grid = np.array([[i * 0.9 * eps, j * 0.9 * eps, 0.0] for i in range(6) for j in range(6)])
    rng = np.random.default_rng(7)
    cloud = rng.uniform(0.0, 6.0 * eps, size=(400, 3))
    return [straddle, grid, cloud]


def test_weld_exact_path_never_fuses_two_vertices_farther_apart_than_eps_in_two_dimensions() -> (
    None
):
    eps = 0.01
    for points in _two_dimensional_cases(eps):
        labels = ops_topo._clusters(points, eps)
        assert _widest_pair_in_a_cluster(points, labels) <= eps * (1.0 + 1e-9)


def test_weld_over_budget_pair_path_never_fuses_two_vertices_farther_apart_than_eps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The leader pass is also what the pair-budget fallback runs directly."""
    monkeypatch.setattr(ops_topo, "WELD_PAIR_BUDGET", 0)
    eps = 0.01
    for points in _two_dimensional_cases(eps):
        labels = ops_topo._clusters(points, eps)
        assert _widest_pair_in_a_cluster(points, labels) <= eps * (1.0 + 1e-9)


def test_weld_still_merges_a_close_pair_through_the_public_op() -> None:
    """The fix narrows the leader's reach; an ordinary seam pair must still fuse."""
    mesh = bm.from_faces(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 5e-5], [1, 1, 0]],
        [[0, 1, 2], [3, 1, 4]],
    )
    out, _ = ops_topo.weld(mesh, el.empty(), eps=1e-4)
    assert len(out.positions) == 4


# --- clay-25: the island passes are one grouped pass, not one pass per island ---


class _CountingIds(np.ndarray):
    """An island-id array that counts every element an ``==`` reads, so the
    test asserts how many full-corner passes were made rather than how long
    they took (wall-clock would not hold under xdist)."""

    compared = 0

    def __eq__(self, other):  # type: ignore[override]
        type(self).compared += int(self.size)
        return super().__eq__(other)

    __hash__ = None  # type: ignore[assignment]


def _patch_field(n_patches: int) -> bm.Mesh:
    """*n_patches* disjoint 2x2-quad patches, box-unwrapped: one island each."""
    positions: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []
    base = 0
    for p in range(n_patches):
        ox, oy = (p % 20) * 5.0, (p // 20) * 5.0
        for j in range(3):
            for i in range(3):
                positions.append((ox + i, oy + j, 0.0))
        for j in range(2):
            for i in range(2):
                a = base + j * 3 + i
                faces.append([a, a + 1, a + 4, a + 3])
        base += 9
    return uvm.box_unwrap(bm.from_faces(np.array(positions, dtype="f4"), faces))


def test_pack_islands_makes_a_bounded_number_of_full_corner_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = _patch_field(300)
    real = ut.islands

    def counted(m: bm.Mesh) -> np.ndarray:
        return real(m).view(_CountingIds)

    monkeypatch.setattr(ut, "islands", counted)
    _CountingIds.compared = 0
    ut.pack_islands(mesh)
    corners = len(mesh.loops)
    # One pass per island read ~300 x corners; a grouped pass reads none or one.
    assert _CountingIds.compared <= 4 * corners


def test_transform_islands_makes_a_bounded_number_of_full_corner_passes() -> None:
    mesh = _patch_field(300)
    ids = ut.islands(mesh).view(_CountingIds)
    _CountingIds.compared = 0
    ut.transform_islands(mesh, list(range(300)), rotate_deg=10.0, ids=ids)
    assert _CountingIds.compared <= 4 * len(mesh.loops)


def test_grouped_island_passes_give_the_per_island_answer() -> None:
    """Pins the numbers, so the grouped pass is a rewrite of the cost only."""
    mesh = _patch_field(12)
    ids = ut.islands(mesh)
    chosen = [1, 4, 7]
    out = ut.transform_islands(mesh, chosen, rotate_deg=30.0, scale=0.5, translate=(0.1, 0.0))
    foc = np.repeat(np.arange(bm.face_count(mesh)), np.diff(mesh.starts))
    corner_island = ids[foc]
    expected = np.array(mesh.uv, dtype="f8", copy=True)
    theta = np.radians(30.0)
    rot = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    for label in chosen:
        sel = corner_island == label
        pts = expected[sel]
        centre = (pts.min(axis=0) + pts.max(axis=0)) / 2.0
        expected[sel] = ((pts - centre) * 0.5) @ rot + centre + (0.1, 0.0)
    assert out.uv == pytest.approx(expected.astype("f4"), abs=1e-6)
    untouched = ~np.isin(corner_island, chosen)
    assert np.array_equal(out.uv[untouched], mesh.uv[untouched])


# --- clay-26: the concave-ring check copies no mesh ------------------------------


def test_refuse_concave_ring_does_not_copy_the_position_array_once_per_ring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_rings = 40
    n_verts = 10_000
    positions = np.zeros((n_verts, 3), dtype="f4")
    rings = []
    for r in range(n_rings):
        base = 4 * r
        positions[base : base + 4] = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
        rings.append(np.arange(base, base + 4, dtype="i4"))
    mesh = bm.Mesh(
        positions=positions,
        loops=np.arange(4, dtype="i4"),
        starts=np.array([0, 4], dtype="i4"),
        material=np.zeros(1, dtype="i4"),
        smooth=np.zeros(1, dtype=bool),
    )

    copied = {"vertices": 0}
    original = bm.Mesh.__post_init__

    def counting(self: bm.Mesh) -> None:
        copied["vertices"] += len(self.positions)
        original(self)

    monkeypatch.setattr(bm.Mesh, "__post_init__", counting)
    ops_dissolve._refuse_concave_ring(mesh, rings)
    # A Mesh per ring copies the whole array each time: 40 x 10,000. The ring's
    # own four points are all the question needs.
    assert copied["vertices"] <= n_verts


def test_refuse_concave_ring_still_refuses_a_concave_ring_past_the_ceiling() -> None:
    n = 2 * ops_dissolve.MAX_CONCAVE_DISSOLVE_RING + 2
    k = np.arange(n)
    radius = np.where(k % 2 == 0, 1.0, 0.5)
    angle = 2.0 * np.pi * k / n
    pts = np.stack([radius * np.cos(angle), radius * np.sin(angle), np.zeros(n)], axis=1)
    mesh = bm.Mesh(
        positions=pts.astype("f4"),
        loops=np.arange(n, dtype="i4"),
        starts=np.array([0, n], dtype="i4"),
        material=np.zeros(1, dtype="i4"),
        smooth=np.zeros(1, dtype=bool),
    )
    with pytest.raises(el.OpError, match="concave"):
        ops_dissolve._refuse_concave_ring(mesh, [np.arange(n, dtype="i4")])


# --- clay-48: the pack margin is the gap in the final square ---------------------


def _unit_squares(n: int) -> bm.Mesh:
    """*n* separate unit-uv quads: n islands, each one square."""
    positions = []
    faces = []
    uv = []
    for i in range(n):
        positions += [(3.0 * i, 0, 0), (3.0 * i + 1, 0, 0), (3.0 * i + 1, 1, 0), (3.0 * i, 1, 0)]
        faces.append([4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3])
        uv.append([(0, 0), (1, 0), (1, 1), (0, 1)])
    return bm.from_faces(np.array(positions, dtype="f4"), faces, uv)


def _island_boxes(mesh: bm.Mesh) -> list[tuple[np.ndarray, np.ndarray]]:
    boxes = []
    for f in range(bm.face_count(mesh)):
        pts = mesh.uv[mesh.starts[f] : mesh.starts[f + 1]]
        boxes.append((pts.min(axis=0), pts.max(axis=0)))
    return boxes


def test_pack_islands_leaves_the_gap_the_margin_asked_for_in_the_final_square() -> None:
    margin = 0.05
    out = ut.pack_islands(_unit_squares(4), margin=margin)
    boxes = sorted(_island_boxes(out), key=lambda b: (round(float(b[0][1]), 4), b[0][0]))
    (a, b, c, _d) = boxes
    assert float(b[0][0] - a[1][0]) == pytest.approx(margin, abs=1e-5)
    assert float(c[0][1] - a[1][1]) == pytest.approx(margin, abs=1e-5)
    # ...and the layout still fills the square rather than leaving slack.
    assert float(out.uv.max()) == pytest.approx(1.0, abs=1e-5)
    assert float(out.uv.min()) == pytest.approx(0.0, abs=1e-6)


def test_pack_islands_with_a_margin_too_big_to_fit_still_packs_inside_the_square() -> None:
    out = ut.pack_islands(_unit_squares(6), margin=0.5)
    assert float(out.uv.min()) >= -1e-6 and float(out.uv.max()) <= 1.0 + 1e-6
    boxes = _island_boxes(out)
    for i, (lo_a, hi_a) in enumerate(boxes):
        assert hi_a[0] > lo_a[0] and hi_a[1] > lo_a[1], "an island collapsed to nothing"
        for lo_b, hi_b in boxes[i + 1 :]:
            apart = (hi_a <= lo_b + 1e-6).any() or (hi_b <= lo_a + 1e-6).any()
            assert apart, "two islands overlap"


# --- clay-49: box unwrap is bounded by the vertices faces reference --------------


def test_box_unwrap_ignores_vertices_no_face_references() -> None:
    box = prim.box((2.0, 1.0, 0.5))
    stray = bm.Mesh(
        positions=np.vstack([box.positions, [[100.0, 0.0, 0.0]]]).astype("f4"),
        loops=box.loops,
        starts=box.starts,
        material=box.material,
        smooth=box.smooth,
    )
    assert np.array_equal(uvm.box_unwrap(stray).uv, uvm.box_unwrap(box).uv)


# --- clay-67: Merge Objects' weld does collapse an unmoved extrude ---------------


def test_join_docstring_names_the_unmoved_extrude_its_weld_collapses() -> None:
    """The weld is applied to the whole result, so an Extrude nobody has moved
    yet (its new ring sits on the old one) is welded flat by a later Merge
    Objects. That is documented, not changed -- the docstring must say it."""
    box = prim.box()
    cap = el.ElementSel(faces=np.array([2], dtype="i4"))
    extruded, _ = ops_topo.extrude_faces(box, cap)
    assert len(extruded.positions) > len(box.positions)
    a = doc.Obj(uid=1, name="A", mesh=extruded)
    b = doc.Obj(uid=2, name="B", mesh=prim.box(), translation=(5.0, 0.0, 0.0))
    merged = ops.join([a, b], eps=1e-4)
    # The extruded box's ten faces and twelve vertices weld back to a plain box's six and eight.
    assert bm.face_count(merged) == 12 and len(merged.positions) == 16
    text = " ".join((ops.join.__doc__ or "").split()).lower()
    assert "extrude" in text and "no op here produces two vertices at one position" not in text


# --- clay-68: no comment names a module or constant that is gone ------------------

_FILES = ["ops_topo.py", "ops_dissolve.py", "ops_boolean.py", "ops.py", "uvtools.py", "uv.py"]


def test_mesh_ops_comments_name_no_deleted_module_or_constant() -> None:
    defined: set[str] = set()
    for path in (_ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        defined.update(re.findall(r"^\s*([A-Z][A-Z0-9_]+)\s*(?::[^=\n]+)?=", text, re.M))
    files = {p.name for p in (_ROOT / "src").rglob("*.py")} | {
        p.name for p in (_ROOT / "tests").rglob("*.py")
    }
    stale: list[str] = []
    for name in _FILES:
        text = (_MESH_DIR / name).read_text(encoding="utf-8")
        for const in sorted(set(re.findall(r"\bMAX_[A-Z][A-Z0-9_]*\b", text))):
            if const not in defined:
                stale.append(f"{name}: {const}")
        for mod in sorted(set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*\.py)\b", text))):
            if mod not in files:
                stale.append(f"{name}: {mod}")
    topo_tests = (_ROOT / "tests" / "modes" / "clay" / "test_ops_topo.py").read_text(
        encoding="utf-8"
    )
    for const in sorted(set(re.findall(r"\bMAX_[A-Z][A-Z0-9_]*\b", topo_tests))):
        if const not in defined:
            stale.append(f"test_ops_topo.py: {const}")
    boolean = (_MESH_DIR / "ops_boolean.py").read_text(encoding="utf-8")
    if "the manual says which is which" in " ".join(boolean.split()):
        stale.append("ops_boolean.py: cites a manual section Clay no longer has")
    assert not stale, stale
