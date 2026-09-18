"""The drag preview's incremental normals, held to bit-identity.

``preview_primitives`` promises byte-identity with ``to_primitives`` on the
moved mesh, so the *unit of reuse* has to be the whole face: a face's raw normal
is the reduction over that face's own corners and nothing outside it, so a face
no moved vertex touches has an identical input and an identical output. A vertex
accumulation reused the same way would not have that property -- floating-point
addition is not associative -- which is why the accumulation is still recomputed
whole and why these tests compare exact equality rather than closeness.
"""

from __future__ import annotations

import threading

import numpy as np

from warlock.kernels.mesh import document as bd
from warlock.kernels.mesh import mesh as bm
from warlock.kernels.mesh import primitives


def _mesh() -> bm.Mesh:
    """A sphere: enough faces that "only some were recomputed" means something,
    and smooth shading so the vertex accumulation is exercised."""
    return primitives.uv_sphere(segments=16, rings=12)


def _moved(mesh: bm.Mesh, which: np.ndarray, delta: float = 0.3) -> np.ndarray:
    positions = np.array(mesh.positions, dtype="f4")
    positions[which] = positions[which] + np.float32(delta)
    return positions


def test_the_incremental_pass_is_bit_identical_to_the_full_one() -> None:
    mesh = _mesh()
    layout = bm.render_layout(mesh)
    verts = np.array([0, 5, 17], dtype="i8")
    positions = _moved(mesh, verts)

    # Frame one: nothing cached, so this is the full pass and it stashes.
    first = bm.render_from_layout(layout, mesh.positions)
    cached = bm.raw_face_normals(layout)
    assert cached is not None

    incremental = bm.render_from_layout(
        layout, positions, moved=verts, previous=cached
    )
    full = bm.render_from_layout(layout, positions)

    assert np.array_equal(incremental[0], full[0])
    assert np.array_equal(incremental[1], full[1]), "normals must be bit-identical"
    assert first[1].shape == full[1].shape


def test_moving_every_vertex_still_matches() -> None:
    mesh = _mesh()
    layout = bm.render_layout(mesh)
    verts = np.arange(len(mesh.positions), dtype="i8")
    positions = _moved(mesh, verts)

    bm.render_from_layout(layout, mesh.positions)
    incremental = bm.render_from_layout(
        layout, positions, moved=verts, previous=bm.raw_face_normals(layout)
    )
    full = bm.render_from_layout(layout, positions)
    assert np.array_equal(incremental[1], full[1])


def test_moving_nothing_reuses_everything_and_still_matches() -> None:
    mesh = _mesh()
    layout = bm.render_layout(mesh)
    bm.render_from_layout(layout, mesh.positions)
    incremental = bm.render_from_layout(
        layout,
        mesh.positions,
        moved=np.zeros(0, dtype="i8"),
        previous=bm.raw_face_normals(layout),
    )
    full = bm.render_from_layout(layout, mesh.positions)
    assert np.array_equal(incremental[1], full[1])


def test_a_successive_drag_frame_matches_a_fresh_computation() -> None:
    """The shape a real drag has: one moved set, many frames, each reusing the
    last frame's raw normals."""
    mesh = _mesh()
    layout = bm.render_layout(mesh)
    verts = np.array([2, 3, 4], dtype="i8")
    bm.render_from_layout(layout, mesh.positions)

    for step in range(1, 4):
        positions = _moved(mesh, verts, delta=0.1 * step)
        incremental = bm.render_from_layout(
            layout, positions, moved=verts, previous=bm.raw_face_normals(layout)
        )
        full = bm.render_from_layout(layout, positions)
        assert np.array_equal(incremental[1], full[1]), step
        # ``full`` re-stashed, so put the incremental answer back for the next
        # round the way the drag path does.
        bm.render_from_layout(layout, positions, moved=verts, previous=bm.raw_face_normals(layout))


def test_preview_primitives_agrees_with_and_without_the_moved_hint() -> None:
    mesh = _mesh()
    verts = np.array([1, 9, 30], dtype="i8")
    positions = _moved(mesh, verts)
    materials: list = []

    bd.preview_primitives(mesh, mesh.positions, materials)
    hinted = bd.preview_primitives(mesh, positions, materials, moved=verts)
    plain = bd.preview_primitives(mesh, positions, materials)

    assert len(hinted) == len(plain)
    for a, b in zip(hinted, plain, strict=True):
        assert np.array_equal(a.positions, b.positions)
        assert np.array_equal(a.normals, b.normals)


def test_the_cache_is_stamped_on_the_layout_object_not_an_array() -> None:
    """A layout is a pure function of an immutable mesh, so the same object
    means the same topology -- where a recycled array id would mean nothing.
    A freshly built layout has no entry, however identical its contents."""
    mesh = _mesh()
    first = bm.render_layout(mesh)
    bm.render_from_layout(first, mesh.positions)
    assert bm.raw_face_normals(first) is not None

    second = bm.render_layout(mesh)
    assert second is not first
    assert bm.raw_face_normals(second) is None


def test_stashing_raw_face_normals_from_two_threads_never_corrupts_or_crashes_the_cache() -> None:
    """The 2026-09-12 audit, finding clay-04: Mason resolving a placed Clay
    primitive (``mason_assets.AssetSource._resolve_primitive``, reached from
    ``mason_mode.export_glb``/``export_obj``/``export_library`` via
    ``docmodes.start_save`` -> ``TaskRunner.submit``) calls
    ``document.to_primitives`` -> ``render_arrays`` -> ``render_from_layout``
    on a task thread, concurrently with any Clay viewport's per-frame drag
    preview on the frame thread -- both stash into the same module-level
    ``_RAW_CACHE``.

    A hammer test that races two threads and hopes for corruption would flake:
    the computation is pure, so a lost race only ever produces a cache miss,
    never a wrong answer. What must be true instead is the *mechanism* --
    that ``_stash``/``raw_face_normals`` are serialised on a lock -- so this
    test asserts that directly and deterministically: while the test thread
    holds ``_RAW_CACHE_LOCK``, a concurrent ``render_from_layout`` call (which
    stashes) must block, and it must complete and populate the cache the
    moment the lock is released.
    """
    mesh = _mesh()
    layout = bm.render_layout(mesh)

    with bm._RAW_CACHE_LOCK:
        finished = threading.Event()

        def worker() -> None:
            bm.render_from_layout(layout, mesh.positions)
            finished.set()

        thread = threading.Thread(target=worker)
        thread.start()
        # The lock is held here, so the worker's _stash must not have run yet.
        blocked_while_held = not finished.wait(timeout=0.2)

    thread.join(timeout=2)
    assert blocked_while_held, "a concurrent stash ran without waiting on the lock"
    assert finished.is_set(), "the worker never completed after the lock was released"
    assert bm.raw_face_normals(layout) is not None


def test_face_of_every_corner_covers_every_corner() -> None:
    mesh = _mesh()
    layout = bm.render_layout(mesh)
    faces = bm.face_of_every_corner(layout)
    assert len(faces) == len(layout.loops)
    assert faces.min() == 0
    assert faces.max() == layout.n_faces - 1
    # Monotone: corners are stored face by face, which is what lets the subset
    # reduction re-base its starts.
    assert np.all(np.diff(faces) >= 0)
