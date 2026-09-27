"""Regression tests for the 2026-09-26 audit's ``w1f4`` fixer brief, which
owns ``kernels/geom3d/glbwrite.py``, ``kernels/geom3d/math3d.py``,
``kernels/mesh/adjacency.py``, ``kernels/mesh/glbimport.py``,
``kernels/mesh/objimport.py`` and ``kernels/mesh/regen.py`` (plus several
``studio/modes/clay`` files covered by
``tests/modes/clay/test_audit_2026_09_26_w1f4.py`` instead, next to the code
they exercise).

clay-mesh-core-04: ``adjacency.py``'s three caches held one lock across their
own build, not just the dict access either side of it.

clay-io-02: ``glbimport._object_for`` decomposed a *composed* world matrix
that can carry shear no individual node's own TRS does.

clay-io-04: ``glbimport._declared_budget`` walked every node's every
primitive with no way to stop early.

clay-io-07: OBJ import accepted ``nan``/``inf`` that ``glbwrite.write_glb``
then had no way to refuse either.

clay-document-04: ``regen.carry_over`` carried ``smooth``/``material`` across
a same-topology generator rebuild but not ``uv``.
"""

from __future__ import annotations

import math
import threading
import time

import numpy as np
import pytest

from realmspinner.kernels.geom3d import glbwrite, gltf
from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.kernels.mesh import adjacency as adj
from realmspinner.kernels.mesh import glbimport, objimport, regen
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh.elements import OpError

# --- clay-mesh-core-04: adjacency's cache lock -------------------------------


def test_a_slow_adjacency_build_on_one_thread_does_not_block_another_meshs_cached_triangulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 2026-09-26 audit's clay-mesh-core-04: ``adjacency()``,
    ``cached_positions_f8()`` and ``cached_triangulation()`` shared one lock
    held across their own ``_build``/``triangulate`` call, not just the dict
    access either side of it -- so a slow build for one mesh (the character
    pipeline's service lane, off the frame thread) blocked every *other*
    mesh's cached lookup on the frame thread for as long as that unrelated
    build took. ``mesh_a``'s build is slowed by hand; ``mesh_b`` is a wholly
    unrelated mesh whose own cached triangulation must not wait on it.
    """
    mesh_a = bp.box()
    mesh_b = bp.cone()
    assert mesh_a is not mesh_b

    building = threading.Event()
    release = threading.Event()
    real_build = adj._build

    def slow_build(mesh: object) -> object:
        if mesh is mesh_a:
            building.set()
            release.wait(timeout=2.0)  # bounded: never hangs the suite either way
        return real_build(mesh)

    monkeypatch.setattr(adj, "_build", slow_build)

    thread = threading.Thread(target=lambda: adj.adjacency(mesh_a))
    thread.start()
    try:
        assert building.wait(timeout=2.0), "the slow build on mesh_a never started"

        start = time.monotonic()
        adj.cached_triangulation(mesh_b)
        elapsed = time.monotonic() - start
    finally:
        release.set()
        thread.join(timeout=2.0)

    assert elapsed < 0.5, (
        f"mesh_b's own cached_triangulation waited {elapsed:.2f}s on mesh_a's "
        "unrelated, still-running build -- the lock must guard only the dict "
        "get/set, not the build itself"
    )


# --- clay-io-02: composed shear -----------------------------------------------


def test_glb_import_preserves_world_geometry_under_a_nonuniform_scaled_parent() -> None:
    """The 2026-09-26 audit's clay-io-02: ``gltf.py``'s own loader already
    refuses a node whose *own* declared matrix has shear (the 2026-09-20
    audit's clay-16), by recomposing ``m3.decompose``'s answer and comparing
    it against the original -- but ``node.world`` is composed through the
    whole ancestor chain, and a non-uniform-scaled parent over a rotated
    child can compose into a matrix with shear even though neither node's
    own TRS does, which that per-node check never sees. ``_object_for`` used
    to hand a matrix like that straight to ``m3.decompose`` anyway, which
    silently drops what it cannot represent.
    """
    parent_scale = m3.scaling(m3.vec3(3.0, 1.0, 1.0))
    child_quat = m3.quat_from_axis_angle(m3.vec3(0.0, 0.0, 1.0), math.radians(45))
    world = parent_scale @ m3.quat_to_mat4(child_quat)

    # Fixture sanity: this matrix must actually carry shear, or the rest of
    # the test would pass for the wrong reason.
    t, r, s = m3.decompose(world)
    assert not np.allclose(m3.compose(t, r, s), world, atol=1e-4, rtol=1e-4)

    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype="f4"
    )
    indices = np.array([0, 1, 2, 0, 2, 3], dtype="<u4")
    prim = gltf.Primitive(positions=positions, indices=indices)
    node = gltf.Node(name="Child")
    node.world = world

    obj = glbimport._object_for(prim, node, "Imported", materials=[], palette={}, taken=set())
    assert obj is not None

    homogeneous = np.concatenate(
        [positions.astype("f8"), np.ones((len(positions), 1), dtype="f8")], axis=1
    )
    expected = (world @ homogeneous.T).T[:, :3]

    # ``_mesh_for`` merges vertices with ``np.unique`` and may reorder them,
    # so the two point sets are compared sorted rather than row for row.
    got = np.asarray(obj.mesh.positions, dtype="f8")
    got_sorted = got[np.lexsort(got.T)]
    expected_sorted = expected[np.lexsort(expected.T)]
    assert np.allclose(got_sorted, expected_sorted, atol=1e-4), (
        "the imported geometry must match the true composed-world points, "
        "not a decompose that silently dropped the shear"
    )
    assert np.allclose(obj.translation, [0.0, 0.0, 0.0])
    assert np.allclose(obj.rotation, m3.quat_identity())
    assert np.allclose(obj.scale, [1.0, 1.0, 1.0])


def test_a_rotated_child_of_an_unscaled_parent_still_decomposes_with_no_baking() -> None:
    """The ordinary case is untouched: no shear means no reason to bake, and
    the object keeps a real, editable T/R/S instead of an identity transform
    with everything folded into the mesh."""
    rotation = m3.quat_from_axis_angle(m3.vec3(0.0, 1.0, 0.0), math.radians(30))
    world = m3.compose(m3.vec3(1.0, 2.0, 3.0), rotation, m3.vec3(1.0, 1.0, 1.0))

    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype="f4"
    )
    indices = np.array([0, 1, 2, 0, 2, 3], dtype="<u4")
    prim = gltf.Primitive(positions=positions, indices=indices)
    node = gltf.Node(name="Child")
    node.world = world

    obj = glbimport._object_for(prim, node, "Imported", materials=[], palette={}, taken=set())
    assert obj is not None
    assert np.allclose(obj.translation, [1.0, 2.0, 3.0])
    assert np.allclose(obj.rotation, rotation)
    # The mesh keeps its own local positions -- nothing baked -- since there
    # is no shear to lose.
    assert np.allclose(np.sort(obj.mesh.positions.reshape(-1)), np.sort(positions.reshape(-1)))


# --- clay-io-04: the declared-budget preflight's own early exit --------------


def _instanced_multi_primitive(node_count: int, primitives_per_mesh: int) -> bytes:
    """One mesh definition with several primitives, referenced by many nodes
    -- ordinary glTF instancing (:mod:`test_glbimport`'s own ``_instanced``),
    widened so ``_declared_budget``'s inner, per-primitive loop has real work
    to do for every one of those nodes too."""
    quad = gltf.Primitive(
        positions=np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype="f4"),
        indices=np.array([0, 1, 2, 0, 2, 3], dtype="u4"),
    )
    mesh = [quad for _ in range(primitives_per_mesh)]
    nodes = [gltf.Node(name=f"n{i}", mesh=0) for i in range(node_count)]
    return glbwrite.write_glb(gltf.Model(nodes, list(range(node_count)), [mesh], []))


def test_declared_budget_refuses_an_instanced_scene_without_walking_every_node_primitive_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 2026-09-26 audit's clay-io-04: ``_declared_budget`` used to loop
    every node times every one of its primitives with no way out, even once
    the running totals were already past what the caller was about to
    refuse for -- 2.3 s of pure Python over a 2.6 MB GLB declaring 200,000
    nodes at 50 primitives each. 4,000 nodes referencing one 50-primitive
    mesh (200,000 declared node/primitive pairs) must be refused after only
    a few of them, not after walking the whole declared total.
    """
    data = _instanced_multi_primitive(node_count=4000, primitives_per_mesh=50)
    monkeypatch.setattr(glbimport, "MAX_OBJECTS", 100)

    start = time.monotonic()
    tris, objects = glbimport._declared_budget(data)
    elapsed = time.monotonic() - start

    assert objects > glbimport.MAX_OBJECTS
    # The early exit's own proof: the reported count is however many entries
    # it took to *cross* the ceiling, not the file's full declared total
    # (4,000 * 50 = 200,000) -- so it must land far below that, close to the
    # ceiling itself.
    assert objects < 1000, (
        f"the early exit did not stop the walk -- counted {objects}, most of "
        "the file's full declared total"
    )
    assert elapsed < 1.0


# --- clay-io-07: non-finite geometry and materials ---------------------------


def test_obj_import_refuses_non_finite_vertex_coordinates_and_write_glb_never_emits_nan() -> None:
    """The 2026-09-26 audit's clay-io-07: Python's ``float()`` reads
    "nan"/"inf" without raising, so a ``v`` line carrying one used to sail
    past OBJ import's own ``except ValueError`` refusal, and a non-finite
    position reaches ``glbwrite.write_glb``'s accessor ``min``/``max`` --
    JSON numbers, not buffer bytes -- as a bare ``NaN``/``Infinity`` token
    that is not valid JSON at all.
    """
    obj_text = "v 0 0 0\nv 1 0 0\nv nan 1 0\nf 1 2 3\n"
    with pytest.raises(OpError, match="non-finite"):
        objimport.obj_to_claydoc(obj_text)

    obj_text_inf = "v 0 0 0\nv 1 0 0\nv inf 1 0\nf 1 2 3\n"
    with pytest.raises(OpError, match="non-finite"):
        objimport.obj_to_claydoc(obj_text_inf)


def test_obj_import_skips_a_non_finite_mtl_factor_rather_than_writing_it_through() -> None:
    """Best-effort, like a non-numeric ``Kd``/``d``/``Ns`` line already was
    (this parser's own docstring) -- a non-finite one is skipped, not
    raised over, and the rest of the file still imports."""
    mtl_text = "newmtl M\nKd nan 0.5 0.5\nd inf\nNs 32\n"
    materials = objimport._parse_mtl(mtl_text)
    entry = materials["M"]
    assert "Kd" not in entry
    assert "d" not in entry
    assert entry["Ns"] == 32.0



# --- clay-io-07, glbwrite's own half: its backstop -----------------------


def test_write_glb_refuses_a_non_finite_vertex_bound_instead_of_emitting_nan_json() -> None:
    """``glbwrite``'s own backstop, for any caller that reaches it with
    non-finite geometry regardless of which importer let it through: a NaN
    or an Infinity in the position data reaches the POSITION accessor's
    ``min``/``max`` -- JSON numbers, not buffer bytes -- and
    ``json.dumps``'s default ``allow_nan=True`` would happily write a bare,
    invalid ``NaN`` token for it.
    """
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [float("nan"), 1.0, 0.0]], dtype="f4")
    indices = np.array([0, 1, 2], dtype="<u4")
    prim = gltf.Primitive(positions=positions, indices=indices)
    node = gltf.Node(name="n", mesh=0)
    model = gltf.Model([node], [0], [[prim]], [])

    with pytest.raises(ValueError, match="non-finite"):
        glbwrite.write_glb(model)


def test_write_glb_refuses_a_non_finite_material_factor() -> None:
    """A material factor (``baseColorFactor``, ``roughnessFactor``, ...) is
    written straight into the JSON chunk, not through an accessor at all --
    the most direct route ``Kd``/``d``/``Ns`` from a hand-edited OBJ/MTL, or
    any other caller, could reach a bare NaN/Infinity in the file."""
    material = gltf.Material(name="m", base_color_factor=(float("inf"), 0.5, 0.5, 1.0))
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype="f4")
    indices = np.array([0, 1, 2], dtype="<u4")
    prim = gltf.Primitive(positions=positions, indices=indices, material=material)
    node = gltf.Node(name="n", mesh=0)
    model = gltf.Model([node], [0], [[prim]], [])

    with pytest.raises(ValueError, match="non-finite"):
        glbwrite.write_glb(model)


def test_write_glb_still_writes_an_ordinary_finite_model() -> None:
    """The backstop must not fire on ordinary geometry -- the whole point is
    refusing only what would otherwise become invalid JSON."""
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype="f4")
    indices = np.array([0, 1, 2], dtype="<u4")
    prim = gltf.Primitive(positions=positions, indices=indices)
    node = gltf.Node(name="n", mesh=0)
    model = gltf.Model([node], [0], [[prim]], [])

    data = glbwrite.write_glb(model)
    assert len(data) > 0
    reloaded = gltf.load(data)
    assert len(reloaded.nodes) == 1


# --- clay-document-04: uv survives a same-topology generator rebuild --------


def test_a_same_topology_rebuild_keeps_the_objects_uv_layout() -> None:
    """The 2026-09-26 audit's clay-document-04: Smart Unwrap and Box Unwrap
    both call ``set_mesh(..., keep_generator=True)``, so an unwrapped
    object still answers to its generator -- and the next same-topology
    rebuild (a radius or a position edit) went through ``carry_over``'s
    verbatim branch, which carried ``smooth``/``material`` but not ``uv``,
    silently discarding the hand-made unwrap and putting back the
    generator's own flat default layout.
    """
    old = bp.box(size=(1.0, 1.0, 1.0))
    hand_unwrapped = np.arange(len(old.loops) * 2, dtype="f4").reshape(-1, 2) / len(old.loops)
    unwrapped = bm.Mesh(
        positions=old.positions,
        loops=old.loops,
        starts=old.starts,
        material=old.material,
        smooth=old.smooth,
        uv=hand_unwrapped,
    )
    rebuilt = bp.box(size=(2.0, 1.0, 1.0))  # a resize: same six faces, same order
    assert bm.face_count(rebuilt) == bm.face_count(unwrapped)

    out = regen.carry_over(unwrapped, rebuilt, material=0)
    assert out.uv is not None
    assert np.allclose(out.uv, hand_unwrapped), (
        "a same-topology rebuild must keep the object's own uv layout, the "
        "same way it already keeps smooth flags and material slots"
    )


def test_a_face_count_changing_rebuild_still_re_derives_uv_with_the_rest() -> None:
    """The other branch is unchanged: a rebuild that is not the same faces
    any more has no old uv to carry, so it re-derives (the generator's own
    default) alongside the already-established smooth/material re-derive."""
    old = bp.cylinder(segments=16)
    hand_unwrapped = np.arange(len(old.loops) * 2, dtype="f4").reshape(-1, 2) / len(old.loops)
    unwrapped = bm.Mesh(
        positions=old.positions,
        loops=old.loops,
        starts=old.starts,
        material=old.material,
        smooth=old.smooth,
        uv=hand_unwrapped,
    )
    rebuilt = bp.cylinder(segments=8)
    assert bm.face_count(rebuilt) != bm.face_count(unwrapped)

    out = regen.carry_over(unwrapped, rebuilt, material=0)
    assert out.uv is None or out.uv.shape[0] == len(rebuilt.loops)
