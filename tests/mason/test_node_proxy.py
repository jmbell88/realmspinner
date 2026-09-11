"""``DrawNode`` / ``NodePool``: the per-draw proxy Mason's instancing needs
and Clay's does not.

``clay_view._composite`` does ``node.world = world`` on the cached entry's own
``gltf.Node``, and that is sound only because Clay's GPU cache is one entry
per *object*. Mason's cache is keyed on the ref (``docs/MASON-PLAN.md``,
"Six things Mason must do differently", item 1), so N placements of one
asset share one cached node -- and a composite that wrote ``.world`` on that
shared object N times a frame would leave N-1 instances drawing at wherever
the last write left it, with nothing on screen or in a log to say so. That is
the bug this file exists to gate, and :func:`test_a_shared_gltf_node_is_what_this_replaces`
demonstrates it directly, with no Mason code at all, so the reason the proxy
exists is recorded here rather than only in a comment.
"""

from __future__ import annotations

import numpy as np

from warlock.studio.mason.scene import NodePool
from warlock.studio.viewer import gltf
from warlock.studio.viewer import math3d as m3


def _translation(x: float) -> np.ndarray:
    return m3.compose(m3.vec3(x, 0.0, 0.0), m3.quat_identity(), m3.vec3(1.0, 1.0, 1.0))


def test_a_shared_gltf_node_is_what_this_replaces() -> None:
    """Writing ``node.world = w`` twice on one ``gltf.Node`` leaves one value
    -- the exact shape of the bug a shared per-ref cache entry would hit six
    times over for six instances of one asset."""
    node = gltf.Node()
    node.world = _translation(1.0)
    node.world = _translation(2.0)
    # The first write is gone. A composite built this way over six instances
    # would leave every one of them reporting the sixth (last) world.
    assert node.world[0, 3] == 2.0


def test_instances_of_one_ref_never_share_a_node() -> None:
    """The central claim: six instances of one ref, emitted through the pool,
    are six distinct objects holding six distinct worlds -- not one object
    overwritten five times.

    Written naively (one proxy reused for every ``.node()`` call, mirroring
    ``clay_view._composite``'s ``node.world = world`` on a single shared
    entry) this fails with all six worlds equal to the last one written; that
    failure was reproduced and confirmed against this exact test before
    ``DrawNode``/``NodePool`` existed in their current form.
    """
    pool = NodePool()
    pool.frame()
    worlds = [_translation(float(i)) for i in range(6)]
    proxies = [pool.node(w) for w in worlds]

    # Six distinct objects, not the same one handed back six times.
    assert len({id(p) for p in proxies}) == 6
    # Each proxy still reports the world it was given -- not the last one.
    for proxy, world in zip(proxies, worlds, strict=True):
        assert proxy.world is world
    reported = [p.world[0, 3] for p in proxies]
    assert reported == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_the_pool_reuses_its_objects_across_frames_without_growing_without_bound() -> None:
    """A thousand-item scene allocates a thousand proxies once, not once a
    frame: after the pool has grown to cover a frame's demand, an identical
    later frame reuses the same backing objects rather than growing again."""
    pool = NodePool()
    world = _translation(0.0)

    pool.frame()
    first_frame = [pool.node(world) for _ in range(1000)]

    pool.frame()
    second_frame = [pool.node(world) for _ in range(1000)]

    assert [id(p) for p in first_frame] == [id(p) for p in second_frame]


def test_the_pool_grows_when_a_later_frame_needs_more_than_the_last_one_did() -> None:
    pool = NodePool()
    world = _translation(0.0)

    pool.frame()
    small = [pool.node(world) for _ in range(3)]

    pool.frame()
    big = [pool.node(world) for _ in range(10)]

    # The first three objects of the bigger frame are the same three the
    # small frame used; the rest are freshly grown.
    assert [id(p) for p in big[:3]] == [id(p) for p in small]
    assert len({id(p) for p in big}) == 10


def test_a_proxy_from_a_previous_frame_is_not_still_live_in_this_one() -> None:
    """The subtle half of pooling: reuse across frames is exactly what a
    caller must not rely on. A proxy fetched last frame can come back this
    frame carrying a different placement's world entirely -- so holding one
    past its own frame is a bug in the caller, not a guarantee this pool owes
    it."""
    pool = NodePool()

    pool.frame()
    stale = pool.node(_translation(1.0))
    stale_id = id(stale)
    assert stale.world[0, 3] == 1.0

    pool.frame()
    fresh = pool.node(_translation(99.0))

    # Same backing object, reused -- and it no longer reports frame one's
    # value, because it was never this pool's promise to keep it.
    assert id(fresh) == stale_id
    assert stale.world[0, 3] == 99.0  # the "stale" reference sees the rewrite
