"""Ray versus scene: which owner a click lands on, and where.

**Picking is per ref, selection is per owner.** A scene of five hundred
placements is, geometrically, however many distinct assets it actually uses --
``scene.py``'s whole argument for keying the GPU cache and the BVH cache alike
on the *ref*, not the instance. So :func:`ray_scene` resolves each unique ref's
triangles once, builds one BVH over them, and re-uses that tree for every
instance sharing it -- the ray moves into each instance's own local space
instead. A hit inside an instance answers with its :attr:`~.scene.Placed
.owner`, never the template node's own uid: the outliner has a row for the
instance, not for whichever leaf of the template the ray happened to graze,
and that is exactly the "picking is per ref, selection is per owner" rule
``scene.py``'s module docstring names ``owner`` to serve.

**The cache key is the primitives list, not the ref, and that is not the same
key wearing a different name.** ``viewer.picking.cached_bvh``'s own contract
is a revision stamp: it works because a ``Mesh`` is immutable, so the same
object is the same geometry and a new object is a new edit. A :class:`~.refs
.Ref` is built to be the opposite of that -- ``refs.py`` made it frozen and
compare by value precisely so it does *not* change when the geometry behind
it does. A ``LibraryRef`` whose GLB gets re-parsed (a relink, or the user
re-exporting that asset from Clay -- the plan names this as routine, and half
the reason the document links rather than embeds) keeps the very same key
while the geometry underneath it changes, and a ref-keyed cache would keep
answering with the shape the asset used to be. So this module keeps its own
cache, one dict per ``GeometrySource`` in a ``WeakKeyDictionary`` (an entry's
lifetime follows the host, not this module), keyed inside that by
``ref_key(ref)`` -- and an entry is only good while ``source.primitives(ref)``
still hands back the *same list object* it was built from. A re-parse hands
back a new list; the identity check misses, and the entry rebuilds itself
with no invalidation call for anyone to remember. This is ``terrain
.terrain_mesh``'s own memo pattern (``cached_array is terrain.heights``), one
door over, and it is why the tree is built with ``viewer.picking.build_bvh``
directly rather than through ``cached_bvh``: holding this cache and also
routing through that one would be two caches answering the same keying
question differently, not one.

**``GeometrySource.rev`` is deliberately not what invalidates this.** ``rev``
is the viewport's redraw signal -- it moves for reasons that have nothing to
do with any one ref, an unrelated asset finishing its parse included -- where
the primitives list handed back for *this* ref is the finer, more honest
stamp: it changes exactly when this ref's own geometry does, and not a frame
sooner or later.

**Terrain gets a heightfield ray march, not a BVH.** A 256-side terrain is
``256**2`` quads -- 131,072 triangles -- rebuilt every time a brush touches it;
handing that to a triangle BVH means rebuilding the tree on every sculpt
frame for the sake of a shape a closed-form height query already answers
exactly. :func:`ray_terrain` marches the ray through ``terrain.height_at``
instead, in the terrain's own local space, and is the one piece of this module
that is not "call ``viewer.picking`` once per item".

**Hidden items are not pickable; locked items are.** ``scene.py`` and
``plotter/scene.py`` both state the lock rule the same way -- a lock stops the
user, not the document, and enforcing it here (skipping a locked item's ray
test) would make it impossible to ever select a locked prop again to unlock
it. So a locked item is hit exactly like an unlocked one, and the lock is
carried on :attr:`Hit.placed` for whoever asked to read it, never consulted by
this module to decide anything.
"""

from __future__ import annotations

import math
import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..viewer import picking
from .nodes import TerrainNode
from .refs import GeometrySource, Ref, ref_key
from .scene import Placed
from .terrain import Terrain, height_at

__all__ = ["Hit", "ray_scene", "ray_terrain"]

#: ``(primitives_list, positions, tris, box, bvh)`` -- everything one ref's
#: geometry needs for a ray test, plus the exact list object it was built
#: from, which is the only thing that makes an entry valid or stale.
_RefEntry = tuple[
    list[Any], np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray] | None, "picking.BVH | None"
]

#: One dict per host, so an entry's lifetime follows the host rather than
#: growing for the life of the process -- a ``GeometrySource`` collected
#: (a document closed, a test that built one and dropped it) takes its ref
#: entries with it instead of leaving them to leak.
_GEOMETRY_CACHE: weakref.WeakKeyDictionary[GeometrySource, dict[tuple[Any, ...], _RefEntry]] = (
    weakref.WeakKeyDictionary()
)


@dataclass(frozen=True)
class Hit:
    """One ray's answer: what it hit, and where."""

    #: What a click selects -- ``Placed.owner``, never a template leaf's uid.
    owner: int
    #: What was actually hit -- the lock, the ref and the node all live here,
    #: so this module never needs a field of its own for any of them.
    placed: Placed
    #: Along the world ray's own parameter -- comparable across every item,
    #: mesh or terrain, the same guarantee ``viewer.picking.ray_object``'s own
    #: docstring makes for two objects of different scale.
    distance: float
    #: The world-space point the ray reached: ``origin + direction * distance``.
    point: np.ndarray


def _ref_geometry(
    source: GeometrySource, ref: Ref
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray] | None, picking.BVH | None] | None:
    """One ref's combined triangles, its local box, and its BVH -- rebuilt
    only when ``source.primitives(ref)`` hands back a list this cache has not
    already seen for this ref.

    A ref can resolve to more than one primitive (a multi-material mesh), so
    every primitive's positions and indices are concatenated into one flat
    triangle set before the BVH is built over it: one tree per ref, not one
    per primitive and not one per instance. See the module docstring for why
    the cache entry is validated against the *primitives list object*, not
    the ref -- a ref is built to stay the same key exactly when its geometry
    has not, which is the wrong property for this cache to key on.

    ``None`` when nothing has resolved yet -- an asset still parsing in the
    background, say -- which is "no geometry to test against", not an error;
    the cache is left untouched rather than poisoned with an empty entry, so
    a resolution arriving one frame later is picked up the moment it does.
    """
    primitives = source.primitives(ref)
    if not primitives:
        return None
    per_source = _GEOMETRY_CACHE.setdefault(source, {})
    key = ref_key(ref)
    cached = per_source.get(key)
    if cached is not None and cached[0] is primitives:
        _cached_primitives, positions, tris, box, bvh = cached
        return positions, tris, box, bvh

    positions_parts: list[np.ndarray] = []
    tris_parts: list[np.ndarray] = []
    offset = 0
    for primitive in primitives:
        pos = np.asarray(primitive.positions, dtype="f8")
        tris = np.asarray(primitive.indices, dtype="i8").reshape(-1, 3)
        positions_parts.append(pos)
        tris_parts.append(tris + offset)
        offset += len(pos)
    positions = np.concatenate(positions_parts, axis=0)
    tris = np.concatenate(tris_parts, axis=0)
    bvh = picking.build_bvh(positions, tris)
    box = source.box(ref)
    per_source[key] = (primitives, positions, tris, box, bvh)
    return positions, tris, box, bvh


def ray_scene(
    placed: Sequence[Placed],
    source: GeometrySource,
    origin: np.ndarray,
    direction: np.ndarray,
    *,
    terrain: Terrain | None = None,
    terrain_world: np.ndarray | None = None,
) -> Hit | None:
    """The nearest thing ``origin``/``direction`` hits among ``placed`` (and
    ``terrain``, if given), or ``None``.

    ``direction`` is expected unit-length -- the same assumption every other
    ray test in ``viewer.picking`` makes -- so ``distance`` doubles as a true
    world distance, not just a comparable parameter.

    **Tie-break: an object beats the terrain at equal distance, and among
    objects the lower owner uid wins.** Exact ties are rare (a prop's base
    sitting exactly on the ground plane is the practical case) and
    ``viewer.picking.ray_triangles`` already states its own rule for the
    reason one is needed at all -- a deterministic answer written down, not
    whichever the iteration order happens to produce. An object winning
    against the ground it stands on is the more useful of the two answers a
    tie could give: it is almost always what the user meant to click.
    ``owner`` is a stable per-scene identity ``ray_scene`` did not have to
    invent, so it is what breaks a mesh-versus-mesh tie the same deterministic
    way run after run.

    Terrain is looked up in ``placed`` for its own owner, lock and visibility
    -- passed as ``terrain``/``terrain_world`` only for the heightfield and
    its placement, because a :class:`~.scene.Placed` for it already carries
    everything else the resolver worked out. A ``terrain`` with no matching
    :class:`~.nodes.TerrainNode` in ``placed`` (or no ``terrain_world``)
    contributes nothing -- there would be no owner to report a hit against.
    """
    origin = np.asarray(origin, dtype="f8")
    direction = np.asarray(direction, dtype="f8")

    cache: dict[tuple[Any, ...], Any] = {}
    best_t: float | None = None
    best_item: Placed | None = None

    for item in placed:
        if not item.visible or item.ref is None:
            continue
        key = ref_key(item.ref)
        if key not in cache:
            cache[key] = _ref_geometry(source, item.ref)
        geometry = cache[key]
        if geometry is None:
            continue
        positions, tris, bounds, bvh = geometry
        hit = picking.ray_object(origin, direction, item.world, positions, tris, bounds, bvh)
        if hit is None:
            continue
        t, _tri_index = hit
        if best_t is None or t < best_t or (t == best_t and item.owner < best_item.owner):
            best_t, best_item = t, item

    terrain_hit: tuple[float, np.ndarray, Placed] | None = None
    if terrain is not None and terrain_world is not None:
        terrain_placed = next(
            (p for p in placed if isinstance(p.node, TerrainNode) and p.visible), None
        )
        if terrain_placed is not None:
            found = ray_terrain(terrain, terrain_world, origin, direction)
            if found is not None:
                t, point = found
                terrain_hit = (t, point, terrain_placed)

    if best_item is not None and (terrain_hit is None or best_t <= terrain_hit[0]):
        assert best_t is not None
        point = origin + direction * best_t
        return Hit(owner=best_item.owner, placed=best_item, distance=best_t, point=point)
    if terrain_hit is not None:
        t, point, item = terrain_hit
        return Hit(owner=item.owner, placed=item, distance=t, point=point)
    return None


# --- terrain: a heightfield march, not a BVH ---------------------------------

#: How close the march's bisection must land on the true crossing before it
#: stops refining, in metres. Tighter than any placement snap step a settings
#: panel offers (``ops.snap_translation``'s own smallest useful step is a
#: matter of centimetres), so a sculpted hill's surface reads as exact to
#: anything that ever calls this.
TERRAIN_HIT_TOLERANCE = 1e-4

#: Bisection halvings once a crossing is bracketed. The widest a bracket can
#: ever be is the march's own two consecutive samples, and the widest *that*
#: gap gets is the whole entry-to-exit span across the terrain's AABB (a
#: near-vertical ray moves almost nothing in X/Z, so :func:`_march_step`
#: hands back one enormous step rather than many small ones) -- at
#: ``MAX_TERRAIN_SIDE``'s largest extent that span is on the order of a few
#: hundred metres. Thirty-two halvings shrink even that down to a fraction of
#: a micrometre, several orders under :data:`TERRAIN_HIT_TOLERANCE`, so this is
#: a fixed, cheap cost rather than a convergence check that could in
#: principle fail to converge in time.
_BISECT_ITERS = 32

#: A hard ceiling on how many heights one march samples, regardless of what
#: :func:`_march_step` computes. At ``MAX_TERRAIN_SIDE`` the step size is
#: never smaller than half a cell (see :func:`_march_step`), so a ray cannot
#: legitimately need more than on the order of ``2 * MAX_TERRAIN_SIDE`` steps
#: to cross the whole heightfield; this is generous headroom above that, not
#: a tight fit, and it is what makes a ray parallel to the terrain **and**
#: nearly parallel to a cell diagonal terminate instead of stepping forever.
_MAX_MARCH_STEPS = 4096


def _slab_interval(
    origin: np.ndarray, direction: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> tuple[float, float] | None:
    """``(t_enter, t_exit)`` where the ray meets the box, or ``None``.

    The same slab test ``viewer.picking.ray_aabb`` runs, restated to return
    the interval instead of a bare bool -- the march below needs the actual
    bounds to know how far to step, not just whether it should. ``t_enter`` is
    clamped to zero: a ray whose origin already sits inside the box starts the
    march from where it is, not from behind itself.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / direction
        t1 = (lo - origin) * inv
        t2 = (hi - origin) * inv
    near = np.nan_to_num(np.minimum(t1, t2), nan=-np.inf)
    far = np.nan_to_num(np.maximum(t1, t2), nan=np.inf)
    t_enter = max(float(near.max()), 0.0)
    t_exit = float(far.min())
    if t_enter > t_exit:
        return None
    return t_enter, t_exit


def _height_gap(
    terrain: Terrain, local_origin: np.ndarray, local_dir: np.ndarray, t: float
) -> float:
    """``ray.y(t) - height_at(x(t), z(t))`` -- positive above the field,
    negative or zero at or below it."""
    x = float(local_origin[0] + t * local_dir[0])
    z = float(local_origin[2] + t * local_dir[2])
    y = float(local_origin[1] + t * local_dir[1])
    return y - height_at(terrain, x, z)


def _march_step(terrain: Terrain, local_dir: np.ndarray) -> float:
    """How far (in ray parameter ``t``) one march step may safely cover.

    Half a cell's width of horizontal travel per step: any coarser and the
    ray could clear a whole ridge -- one cell tall, one cell wide -- between
    two samples without either one noticing it was ever above the field.
    ``local_dir`` is deliberately left un-normalized by the caller (matching
    ``viewer.picking.ray_object``'s own convention), so this divides by its
    *horizontal* speed to get back to metres of world travel per step, and a
    ray with almost no horizontal component (nearly vertical) is given one
    enormous step instead of a needlessly fine one -- height barely changes
    with a horizontal position that itself barely changes.
    """
    horizontal_speed = math.hypot(float(local_dir[0]), float(local_dir[2]))
    cell = min(terrain.size_x, terrain.size_z) / terrain.side
    return 0.5 * cell / max(horizontal_speed, 1e-9)


def ray_terrain(
    terrain: Terrain, world: np.ndarray, origin: np.ndarray, direction: np.ndarray
) -> tuple[float, np.ndarray] | None:
    """Where ``origin``/``direction`` first crosses ``terrain`` from above, or
    ``None``.

    The ray is carried into the terrain's local space by ``world``'s inverse
    (a singular ``world`` -- a zero scale somewhere -- answers ``None`` rather
    than raising a ``LinAlgError`` out of a mouse move, matching
    ``viewer.picking.ray_object``). ``local_dir`` is left un-normalized on
    purpose, the identical reason ``ray_object`` gives: both halves of the ray
    map linearly, so the ``t`` the march finds is the *world* ray's own
    parameter, comparable against every mesh hit ``ray_scene`` also finds,
    with no renormalization needed to make the two agree.

    **Only an above-to-below crossing counts.** A ray whose origin already
    sits under the field crosses from below to above on its way out and that
    crossing is not reported -- there is no surface there to have hit, only
    the underside of one, and a user is never standing inside the ground to
    click through it from underneath. The consequence, decided and tested
    rather than left to fall out however it likes: a ray that starts under
    the terrain and never re-emerges from above further along does not report
    a hit at all, even though it is, in another sense, "inside" the shape the
    whole time.

    **March, don't triangulate.** A :data:`~.terrain.MAX_TERRAIN_SIDE`` terrain
    is 131,072 triangles rebuilt on every sculpt frame for a BVH that would be
    stale the instant the next brush stroke lands; marching in ``O(cells along
    the ray)`` through the same ``height_at`` a "drop to ground" op already
    calls costs nothing to keep in sync. The march is bounded by the
    terrain's own local extent (via :func:`_slab_interval`) so a ray parallel
    to and entirely above the field terminates within a fixed number of
    samples rather than running forever, and a crossing found between two
    samples is refined by bisection to :data:`TERRAIN_HIT_TOLERANCE`.
    """
    origin = np.asarray(origin, dtype="f8")
    direction = np.asarray(direction, dtype="f8")
    world = np.asarray(world, dtype="f8")
    try:
        inverse = np.linalg.inv(world)
    except np.linalg.LinAlgError:
        return None
    local_origin = (inverse @ np.append(origin, 1.0))[:3]
    local_dir = inverse[:3, :3] @ direction
    if not np.isfinite(local_origin).all() or not np.isfinite(local_dir).all():
        return None

    half_x = terrain.size_x / 2.0
    half_z = terrain.size_z / 2.0
    y_lo = float(terrain.heights.min())
    y_hi = float(terrain.heights.max())
    interval = _slab_interval(
        local_origin,
        local_dir,
        np.array([-half_x, y_lo, -half_z]),
        np.array([half_x, y_hi, half_z]),
    )
    if interval is None:
        return None
    t_enter, t_exit = interval

    step = _march_step(terrain, local_dir)
    steps = max(1, min(_MAX_MARCH_STEPS, math.ceil((t_exit - t_enter) / step)))
    ts = np.linspace(t_enter, t_exit, steps + 1)

    prev_t = float(ts[0])
    prev_gap = _height_gap(terrain, local_origin, local_dir, prev_t)
    for t in ts[1:]:
        t = float(t)
        gap = _height_gap(terrain, local_origin, local_dir, t)
        # ">=" on both sides of this test, not the stricter ">"/"<" a first
        # instinct reaches for: the march's boundary samples sit *exactly* on
        # the field whenever a ray runs through the terrain's highest point
        # (the AABB's own top, by construction) or its lowest one (flat
        # ground away from any bump is exactly the AABB's own bottom), and
        # either one lands a sample at ``gap == 0`` with no room either side
        # to be strictly above or strictly below. Both are still real ground
        # contact, not a coincidence to shrug off, so a boundary sample counts
        # on whichever side of the pair it falls -- "at or above" followed by
        # "at or below" -- or a ray straight down onto flat ground would
        # report no hit at all simply because the ground it landed on was
        # exactly as low as the box around it.
        if prev_gap >= 0.0 and gap <= 0.0:
            hit_t = _bisect(terrain, local_origin, local_dir, prev_t, t)
            return hit_t, origin + direction * hit_t
        prev_t, prev_gap = t, gap
    return None


def _bisect(
    terrain: Terrain,
    local_origin: np.ndarray,
    local_dir: np.ndarray,
    lo_t: float,
    hi_t: float,
) -> float:
    """Narrow ``(lo_t, hi_t)`` -- known above/at-or-below respectively -- onto
    the crossing, to :data:`TERRAIN_HIT_TOLERANCE`."""
    for _ in range(_BISECT_ITERS):
        mid_t = (lo_t + hi_t) / 2.0
        mid_gap = _height_gap(terrain, local_origin, local_dir, mid_t)
        if mid_gap >= 0.0:
            lo_t = mid_t
        else:
            hi_t = mid_t
        if abs(mid_gap) <= TERRAIN_HIT_TOLERANCE:
            break
    return hi_t
