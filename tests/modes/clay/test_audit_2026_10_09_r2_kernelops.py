"""Regression tests for the 2026-10-09 second Clay audit's element-op kernel
findings (clay-83, 84, 88, 89, 90, 108's kernel half). Each test name is the
claim, and each was run against the unfixed tree first."""

from __future__ import annotations

import time

import numpy as np
import pytest

from realmspinner.kernels.mesh import earclip, ops, ops_dissolve, ops_topo
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh.adjacency import adjacency, boundary_ring_from, ring_is_closed
from realmspinner.kernels.mesh.mesh import _newell

# --- clay-84: Merge Faces over many small groups ------------------------------


def _pairs_mesh(n_pairs: int) -> bm.Mesh:
    """``n_pairs`` disjoint 1x2 strips: two quads sharing an edge, six own vertices."""
    pos: list[tuple[float, float, float]] = []
    loops: list[int] = []
    for i in range(n_pairs):
        b = len(pos)
        x = float(i) * 3.0
        pos += [(x, 0, 0), (x + 1, 0, 0), (x + 2, 0, 0), (x + 2, 0, 1), (x + 1, 0, 1), (x, 0, 1)]
        loops += [b, b + 1, b + 4, b + 5, b + 1, b + 2, b + 3, b + 4]
    n_faces = 2 * n_pairs
    return bm.Mesh(
        positions=np.array(pos, "f4"),
        loops=np.array(loops, "i4"),
        starts=np.arange(n_faces + 1, dtype="i4") * 4,
        material=np.zeros(n_faces, "i4"),
        smooth=np.zeros(n_faces, bool),
    )


def test_merge_faces_over_many_small_groups_stays_under_a_second_or_is_refused(monkeypatch):
    # The 2026-10-09 second audit found Merge Faces over 20,000 two-quad groups
    # froze the frame thread 1.8 s because the concavity test ran one
    # `earclip.concave_faces` call, with a dozen numpy dispatches, per group.
    # Wall clock is not asserted (xdist makes it meaningless); the cost it was
    # made of is: the number of per-ring kernel calls must not grow with the
    # group count, and the press must be either refused or done in a handful.
    n = 20_000
    m = _pairs_mesh(n)
    calls = 0
    real = earclip.concave_faces

    def counting(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(earclip, "concave_faces", counting)
    try:
        out, _ = ops_dissolve.dissolve_faces(m, el.ElementSel(faces=np.arange(bm.face_count(m))))
    except el.OpError:
        return  # refused outright is the other acceptable answer
    assert calls <= 4, f"{calls} concavity calls for {n:,} groups"
    assert bm.face_count(out) == n


def _reference_refuse_concave_ring(mesh, vertex_rings) -> str | None:
    """HEAD's per-ring `_refuse_concave_ring`, kept verbatim as the oracle."""
    budget = ops_dissolve.MAX_CONCAVE_DISSOLVE_RING * ops_dissolve.MAX_CONCAVE_DISSOLVE_RING
    for ring in vertex_rings:
        n = len(ring)
        if n < 3:
            continue
        ring_ids = np.asarray(ring, dtype="i8")
        local = np.arange(n, dtype="i8")
        points = mesh.positions[ring_ids]
        normals = _newell(points, local, np.roll(local, -1), np.zeros(1, dtype="i8"))
        is_concave = earclip.concave_faces(points, local, np.array([0, n], dtype="i8"), normals)[0]
        if not is_concave:
            continue
        if n > ops_dissolve.MAX_CONCAVE_DISSOLVE_RING:
            return "too big"
        budget -= n * n
        if budget < 0:
            return "budget"
    return None


def _outcome(mesh, rings) -> str | None:
    try:
        ops_dissolve._refuse_concave_ring(mesh, rings)
    except el.OpError as exc:
        return "too big" if "concave face with" in str(exc) else "budget"
    return None


def test_batched_concavity_test_agrees_with_the_per_ring_reference(monkeypatch):
    # The batched test must change no answer: same refusal (and the same
    # reason) as the one-ring-at-a-time original on random rings, star-shaped
    # (usually concave), convex, non-planar, degenerate and tiny.
    rng = np.random.default_rng(20261009)
    monkeypatch.setattr(ops_dissolve, "MAX_CONCAVE_DISSOLVE_RING", 40)
    seen: set[str | None] = set()
    for _ in range(300):
        n_pts = int(rng.integers(8, 200))
        kind = int(rng.integers(0, 3))
        if kind == 0:  # flat, jagged star-shaped outlines
            ang = np.sort(rng.uniform(0, 2 * np.pi, n_pts))
            rad = rng.uniform(0.2, 1.0, n_pts)
            pts = np.stack([np.cos(ang) * rad, np.zeros(n_pts), np.sin(ang) * rad], axis=1)
        elif kind == 1:  # convex circle
            ang = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
            pts = np.stack([np.cos(ang), np.sin(ang) * 0.5, np.zeros(n_pts)], axis=1)
        else:  # unstructured soup, including repeated points
            pts = rng.integers(-3, 4, size=(n_pts, 3)).astype("f8")
        mesh = bm.from_faces(pts.astype("f4"), [[0, 1, 2]])
        rings = []
        cursor = 0
        for _r in range(int(rng.integers(1, 14))):
            size = int(rng.integers(0, 60))
            if size == 0:
                rings.append(np.zeros(0, dtype="i8"))
                continue
            cursor = int(rng.integers(0, max(1, n_pts - 2)))
            ids = (np.arange(size) + cursor) % n_pts
            rings.append(ids.astype("i8"))
        want = _reference_refuse_concave_ring(mesh, rings)
        got = _outcome(mesh, rings)
        assert got == want, (kind, [len(r) for r in rings], want, got)
        seen.add(want)
    assert {None, "too big", "budget"} <= seen, f"corpus never reached {seen}"


def test_merge_faces_result_is_unchanged_by_the_batched_concavity_test():
    # An L of three quads plus a separate pair: both groups merge, and the
    # merged outline of the L (concave) still comes out as one 8-corner face.
    pos = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 0, 1), (1, 0, 1), (1, 0, 2), (0, 0, 2), (0, 0, 1)]
    m = bm.from_faces(pos, [[0, 1, 4, 7], [1, 2, 3, 4], [7, 4, 5, 6]])
    out, sel = ops_dissolve.dissolve_faces(m, el.ElementSel(faces=np.arange(3)))
    assert bm.face_count(out) == 1 and len(sel.faces) == 1
    assert len(out.loops) == 8


# --- clay-88: fill_hole on an unclosed boundary walk --------------------------


def _flipped_pair() -> bm.Mesh:
    pos = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    return bm.from_faces(pos, [[0, 1, 2], [0, 3, 2]])  # second triangle wound the same way


@pytest.mark.parametrize("seed", [[0, 1], [1, 2], [0, 3], [3, 2]])
def test_fill_hole_refuses_a_boundary_that_runs_into_a_flipped_edge_instead_of_adding_a_short_face(
    seed,
):
    # The 2026-10-09 second audit found the walk ends at the flipped edge with
    # no outgoing candidate and `fill_hole` appended the open chain (1 or 2
    # corners) as a face, which `bm.validate` then refuses to reopen.
    m = _flipped_pair()
    with pytest.raises(el.OpError):
        ops_topo.fill_hole(m, el.ElementSel(edges=np.array([seed])))


def test_ring_is_closed_reports_a_walk_that_did_not_close():
    m = _flipped_pair()
    rings, _ = boundary_ring_from(m, np.array([[0, 1]]))
    assert len(rings) == 1 and not ring_is_closed(m, rings[0])


def test_ring_is_closed_accepts_a_real_hole():
    from realmspinner.kernels.mesh import primitives as bp

    box = bp.box()
    keep = np.arange(1, bm.face_count(box))
    from realmspinner.kernels.mesh import topo

    holed = topo.take_faces(box, keep)
    edges = adjacency(holed).edge_verts[adjacency(holed).edge_uses == 1]
    rings, _ = boundary_ring_from(holed, edges[:1])
    assert len(rings) == 1 and len(rings[0]) == 4 and ring_is_closed(holed, rings[0])
    out, sel = ops_topo.fill_hole(holed, el.ElementSel(edges=edges[:1]))
    bm.validate(out)
    assert len(sel.faces) == 1


# --- clay-89: the unreached kernels' false notes ------------------------------


def _holes_mesh(n: int) -> bm.Mesh:
    """``n`` disjoint square frames, each with a 4-corner hole in the middle."""
    pos: list[tuple[float, float, float]] = []
    loops: list[int] = []
    for i in range(n):
        b = len(pos)
        x = i * 4.0
        pos += [
            (x, 0, 0), (x + 3, 0, 0), (x + 3, 0, 3), (x, 0, 3),
            (x + 1, 0, 1), (x + 2, 0, 1), (x + 2, 0, 2), (x + 1, 0, 2),
        ]  # fmt: skip
        for k in range(4):
            k2 = (k + 1) % 4
            loops += [b + k, b + k2, b + 4 + k2, b + 4 + k]
    n_faces = 4 * n
    return bm.Mesh(
        positions=np.array(pos, "f4"),
        loops=np.array(loops, "i4"),
        starts=np.arange(n_faces + 1, dtype="i4") * 4,
        material=np.zeros(n_faces, "i4"),
        smooth=np.zeros(n_faces, bool),
    )


# A 1 s budget that read 1.00 s on the 4-vCPU CI runner: wall-clock lane, serial.
@pytest.mark.perf
def test_fill_hole_at_its_seed_ceiling_finishes_in_under_a_second():
    # The 2026-10-09 second audit re-measured the note beside
    # MAX_FILL_HOLE_SEEDS: 0.83 s at 50,000 seeds was really ~4.2 s (86 us a
    # hole, linear). Timing the ceiling itself is too slow and too noisy for a
    # shared lane, so a 2,000-hole sample (best of three, which discards a
    # contended run) is scaled linearly to the ceiling.
    n = 2_000
    m = _holes_mesh(n)
    a = adjacency(m)
    inner = a.edge_verts[a.edge_uses == 1]
    inner = inner[(inner[:, 0] % 8 >= 4) & (inner[:, 1] % 8 >= 4)]
    seeds = inner[::4]  # one seed per hole
    assert len(seeds) == n
    best = float("inf")
    for _ in range(5):
        t = time.perf_counter()
        out, _sel = ops_topo.fill_hole(m, el.ElementSel(edges=seeds))
        best = min(best, time.perf_counter() - t)
    assert bm.face_count(out) == bm.face_count(m) + n
    at_ceiling = best / n * ops_topo.MAX_FILL_HOLE_SEEDS
    assert at_ceiling < 1.0, f"~{at_ceiling:.2f} s at {ops_topo.MAX_FILL_HOLE_SEEDS:,} seeds"


def test_collapse_docstring_says_faces_that_share_a_vertex_collapse_together():
    # Faces joined at a vertex are one union-find component, so they collapse to
    # ONE vertex at the mean of every vertex either face names -- not each to
    # its own centroid, which is what the docstring used to promise.
    pos = [(x, 0, z) for z in (0, 1) for x in (0, 1, 2, 3)]
    m = bm.from_faces(pos, [[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6]])
    out, _ = ops_topo.collapse(m, el.ElementSel(faces=np.array([0, 1])))
    mean = np.array([1.0, 0.0, 0.5])
    hits = np.flatnonzero(np.abs(out.positions - mean).max(axis=1) < 1e-6)
    assert len(hits) == 1  # one merged vertex ...
    assert not (np.abs(out.positions - np.array([0.5, 0.0, 0.5])).max(axis=1) < 1e-6).any()
    doc = " ".join((ops_topo.collapse.__doc__ or "").split())
    assert "share a vertex" in doc and "Faces collapse to their own centroid." not in doc


# --- clay-90: next_name -------------------------------------------------------


def test_next_name_handles_a_name_with_a_five_thousand_digit_suffix():
    # The 2026-10-09 second audit found `int()` on an unbounded digit run
    # raised ValueError (Python's 4,300-digit limit) out of Duplicate and
    # Mirror Copy for a name an imported GLB can carry verbatim.
    name = "Leaf." + "7" * 5000
    out = ops.next_name(name, [])
    assert out == name + ".001"
    assert ops.next_name(name, [out]) == name + ".002"


def test_next_name_still_counts_up_an_ordinary_suffix():
    assert ops.next_name("Box", []) == "Box.001"
    assert ops.next_name("Box.009", []) == "Box.010"
    assert ops.next_name("Box.001", ["Box.002"]) == "Box.003"
    assert ops.next_name("Box.123456789", []) == "Box.123456790"


# --- clay-108 (kernel half): snap_value ---------------------------------------


def test_snap_value_with_a_denormal_step_is_a_refusal_not_an_overflow_error():
    # The 2026-10-09 second audit found `value / 5e-324` is inf and
    # `math.floor(inf)` raised OverflowError past `run`'s OpError handler.
    with pytest.raises(el.OpError):
        ops.snap_value(1.0, 5e-324)
    with pytest.raises(el.OpError):
        ops.snap_translation([0.37, 1.2, -3.3], 5e-324)


def test_snap_value_is_unchanged_for_ordinary_steps():
    assert ops.snap_value(0.34, 0.25) == 0.25
    assert ops.snap_value(-0.34, 0.25) == -0.25
    assert ops.snap_value(0.5, 1.0) == 1.0
    assert ops.snap_value(-0.5, 1.0) == -1.0
    assert ops.snap_value(1.0, 0.0) == 1.0
    assert ops.snap_value(0.0, 1e-10) == 0.0
