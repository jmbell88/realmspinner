"""Regressions for the 2026-09-23 audit's clay-02, clay-11 and clay-12
findings against ``kernels/mesh/scratch.py``, ``primitives.py``,
``adjacency.py`` and ``document.py``'s ``render_plan``.

See the audit's own records for the full write-up; this module only pins
the failure each one names.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from realmspinner.kernels.mesh import adjacency as adj
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import scratch as clay_scratch

# --- clay-02: Familiar Apply after a previewed reparent ----------------------


def test_a_transplanted_reparent_does_not_leave_the_object_under_its_old_parent_wearing_the_new_parents_local_trs():  # noqa
    """clay-02: a scratch preview that reparents a child (``keep_world=True``,
    ``clay_parent``'s own default) recomputes the child's local TRS relative
    to its *new* parent's frame. Before this fix, ``transplant`` carried only
    that recomputed TRS through ``transform_changed`` -- never the parent
    link itself, because ``parent`` was absent from ``scratch._PROP_FIELDS``
    -- so Apply landed the new-parent-relative numbers on an object still
    hanging under its *old* parent on the real document, and the object
    visibly jumped even though the preview picture showed it standing still.
    """
    doc = bd.ClayDoc()
    parent_obj = doc.add_object(
        bd.Obj(
            uid=bd.new_uid(), name="parent", mesh=bp.box(), translation=np.array([5.0, 0.0, 0.0])
        )
    )
    child_obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="child", mesh=bp.box()))

    before_world = doc.world_matrix(child_obj.uid)[:3, 3].copy()

    # Exactly what clay_parent's handler does against a Familiar preview
    # scratch: reparent with keep_world=True (the tool's own default), which
    # moves nothing on screen in the *scratch*.
    scratch = clay_scratch.clone(doc)
    scratch.set_parent(child_obj.uid, parent_obj.uid, keep_world=True)
    preview_world = scratch.world_matrix(child_obj.uid)[:3, 3].copy()
    assert np.allclose(preview_world, before_world), "the preview itself must not move the child"

    diff = clay_scratch.diff(doc, scratch)
    clay_scratch.transplant(doc, scratch, diff)

    assert doc.by_uid(child_obj.uid).parent == parent_obj.uid, (
        "Apply must carry the reparent itself, not just the recomputed local TRS"
    )
    after_world = doc.world_matrix(child_obj.uid)[:3, 3]
    assert np.allclose(after_world, before_world), (
        "the user approved a picture where the child stood still; Apply moved it to "
        f"{after_world} instead of leaving it at {before_world}"
    )


# --- clay-11: arch height clamp -----------------------------------------------


def test_clamp_params_mirrors_archs_own_height_floor():
    """clay-11: ``arch()`` raises ``height`` to at least ``width / 2`` before
    building (see its own docstring: "a height below that leaves no leg at
    all and is raised to it"), but ``clamp_params("arch")`` never mirrored
    that floor, so a saved document's ``height`` param could permanently
    disagree with the mesh the generator actually built.
    """
    params = {"width": 4.0, "height": 0.5, "thickness": 0.3}
    clamped = bp.clamp_params("arch", params)

    mesh = bp.arch(**clamped)
    # arch() centres the shape on its own origin -- the top of the bounding
    # box is height/2 above the origin regardless of what "height" claims.
    built_height = float(mesh.positions[:, 1].max() - mesh.positions[:, 1].min())

    assert clamped["height"] == 2.0, (
        f"clamp_params reported height={clamped['height']!r}, but arch() itself floors "
        "height to width/2 = 2.0 -- the saved param disagrees with the mesh"
    )
    assert np.isclose(built_height, clamped["height"])


# --- clay-12: unlocked mesh caches --------------------------------------------


def test_adjacency_cache_builds_only_once_under_concurrent_access_from_two_threads(monkeypatch):
    """clay-12: ``adjacency._CACHE`` (and its siblings ``_F8``/``_TRIS``) is a
    bare ``WeakKeyDictionary`` reached from both the frame thread and a
    Familiar scratch batch on a shared, unlocked ``Mesh``. Without a lock
    around the check-then-build-then-store sequence, two threads racing a
    cache miss for the same mesh both rebuild it; with the lock, the second
    thread blocks until the first has stored the result and gets a cache hit.
    """
    mesh = bp.box()  # a fresh instance, not already cached by anything else
    calls: list[int] = []
    real_build = adj._build

    def slow_build(m):
        calls.append(1)
        time.sleep(0.1)
        return real_build(m)

    monkeypatch.setattr(adj, "_build", slow_build)

    results: list[object | None] = [None, None]

    def worker(index: int) -> None:
        results[index] = adj.adjacency(mesh)

    t1 = threading.Thread(target=worker, args=(0,))
    t2 = threading.Thread(target=worker, args=(1,))
    t1.start()
    time.sleep(0.02)  # let t1 pass its cache-miss check and enter _build
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert len(calls) == 1, (
        f"expected exactly one _build call while a lock holds out the second thread, "
        f"got {len(calls)} -- two threads raced the same cache miss"
    )
    assert results[0] is results[1]


def test_render_plan_cache_builds_only_once_under_concurrent_access_from_two_threads(monkeypatch):
    """clay-12: ``document._PLANS`` carries the identical, unlocked-cache gap
    as ``adjacency.py``'s three tables -- reached from ``render_plan``, on a
    ``Mesh`` shared between the live document and a Familiar scratch preview.
    Same proof shape as the adjacency regression above: patch the mesh
    module's ``render_layout`` to be slow, and assert only one thread's call
    actually builds anything.
    """
    mesh = bp.box()
    calls: list[int] = []
    real_render_layout = bd.bm.render_layout

    def slow_render_layout(*args, **kwargs):
        calls.append(1)
        time.sleep(0.1)
        return real_render_layout(*args, **kwargs)

    monkeypatch.setattr(bd.bm, "render_layout", slow_render_layout)

    results: list[object | None] = [None, None]

    def worker(index: int) -> None:
        results[index] = bd.render_plan(mesh)

    t1 = threading.Thread(target=worker, args=(0,))
    t2 = threading.Thread(target=worker, args=(1,))
    t1.start()
    time.sleep(0.02)
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert len(calls) == 1, (
        f"expected exactly one render_layout build while a lock holds out the second thread, "
        f"got {len(calls)} -- two threads raced the same cache miss"
    )
    assert results[0] is results[1]
