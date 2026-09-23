"""Collision proxies as ordinary meshes: box, sphere, capsule, convex, compound.

CLAY-PLAN.md tranche 7's pure kernel half. Everything here is a function of a
:class:`~.mesh.Mesh`'s ``positions`` (never its faces -- a collider is a fit
*over a point cloud*, and a caller with a loose set of points, not yet a valid
mesh, is exactly :func:`compound`'s own situation for each part it hulls), and
every returned :class:`Collider` is expressed in the object's own **local**
frame, matching Clay's own convention (glTF's: metres, Y up) -- there is no
separate transform to compose in; the mesh's vertex positions already are the
fitted shape, in place. The five fits share one contract: refuse nothing (a
degenerate *fit* input just produces a degenerate box/sphere/capsule -- there
is always a mathematically valid answer for those three), except
:func:`convex_hull`, which genuinely has none below four non-coplanar points
and says so through :class:`~.elements.OpError`, the same "refusal is a
sentence, not a crash" contract every other op in this package uses.

**Two things a returned collider is, and is not.** ``Collider.params`` is the
*exact* analytic fit -- a sphere's centre and radius, a capsule's axis and
half-height, a box's extents and rotation -- and it is what an exporter should
hand to an engine with a native primitive collider type (Unity's
``SphereCollider``, Godot's ``SphereShape3D``, Unreal's sphere/box/capsule
simple-collision components), because that is the shape the fit actually
proved contains every input point. ``Collider.mesh`` is a *rendered
approximation* of that same shape, built for preview inside Clay's own
viewport and as a fallback for a target with no native analytic collider (a
convex mesh is the only kind every engine here accepts in some form). For
:data:`_SPHERE_SEGMENTS`/:data:`_SPHERE_RINGS`-grade tessellation this
approximation is **inscribed**, not circumscribed -- its flat faces cut
slightly inside the true sphere or capsule cap between vertices -- so a claim
that the *polygon* contains every input point would be false at any
tessellation cheap enough to ship as a collider. The analytic params are the
honest containment claim; the mesh is not, and nothing here pretends otherwise
by silently over-inflating the tessellation to paper over it.

**Why this module imports ``primitives.py`` after all.** An earlier draft of
this file hand-copied ``box``/``uv_sphere``/``capsule``'s geometry into three
private builders, because at the time another agent's concurrent tranche
owned ``primitives.py`` and a signature or winding convention there could
have shifted mid-session. That was a reason to avoid a moving file, not a
reason to avoid that file forever, and it stopped applying the moment the
other tranche landed and ``primitives.py`` settled. Left alone, the copy was
just a second, unsynchronised description of the same sixty-odd lines of
geometry -- exactly what that module's own docstring argues against in
general ("a second copy that could drift") -- with nothing here that would
notice the day ``primitives.py``'s winding, clamping or defaults changed
under it. Building on the real generators buys the guarantee back for free:
the winding a collider ships is not merely "the same convention" any more, it
is *the same code path*, so a future fix to how ``primitives._ring`` seams a
sphere or ``primitives.capsule`` walls its poles reaches every collider built
from it without a second, silent update here. The one place the two
contracts genuinely differ is texture: every generator in ``primitives.py``
returns a UV-mapped mesh, because it is built for a placed object a user may
paint, and a collider is never textured (see :func:`_concat_meshes`) --
:func:`_untextured` strips that one field back off, the only seam this module
adds on top of what ``primitives.py`` already builds.

**Engine ceilings a hull has to fit under.** PhysX -- the physics backend
Unity and Unreal both ship -- caps a convex mesh at 255 vertices (moderate-high
confidence: this is the commonly cited PhysX convex-mesh ceiling, driven by an
8-bit vertex index internally; not independently verified against NVIDIA's
current SDK docs in this session, so treat the exact number as "in this
neighbourhood" rather than load-bearing to the byte). ``max_faces=64`` -- half
of :func:`convex_hull`'s default -- keeps every hull comfortably under that
regardless of the exact ceiling, the same "judgement with headroom" shape
``readiness.PROFILES`` states its own budgets under. Unreal's own convex
collision has no documented *minimum* vertex count this session could find a
citable source for; a claim like "8 vertices minimum" is not asserted here.

**Why :func:`convex_hull` picks farthest-point sampling over "remove the
point that changes the volume least" for its ``max_faces`` reduction.** The
latter is what the brief suggests as an example, and it is a fine algorithm --
but doing it honestly means, at every step, removing one hull vertex,
re-running quickhull on what is left, comparing the enclosed volume before and
after, and keeping whichever removal cost the least, which is one quickhull
call *per candidate, per step*. Reducing a 114-vertex sphere hull down to the
~34 vertices ``max_faces=64`` allows is tens of thousands of quickhull calls
under that scheme. Farthest-point sampling selects the target vertex count
directly from the existing hull vertices -- greedily keeping whichever
remaining point is farthest (by Euclidean distance) from everything already
kept, a standard, deterministic point-set thinning method -- and then hulls
that reduced set *once*. The tradeoff this makes plain: the result is a
well-spread, good-quality reduction, not a provably-minimum-volume-loss one,
and :func:`convex_hull`'s own docstring states the tolerance this costs on a
dense sphere rather than asserting a stronger claim than the method backs.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from . import primitives
from .adjacency import adjacency
from .elements import OpError
from .mesh import Mesh, bounds, from_faces, transformed

__all__ = [
    "COLLIDER_KINDS",
    "Collider",
    "compound",
    "convex_hull",
    "fit_box",
    "fit_capsule",
    "fit_sphere",
]


@dataclass(frozen=True)
class Collider:
    """One collision proxy: what kind it is, a preview/export mesh, and the
    exact analytic fit that produced it -- see the module docstring for which
    of the two a caller should actually hand to an engine."""

    kind: str  # "box" | "sphere" | "capsule" | "convex" | "compound"
    mesh: Mesh
    params: dict[str, Any]


# A degenerate fit (every input point coincident, or a single vertex) still
# has to produce a *valid* Mesh -- ``validate`` accepts a zero-size box or
# sphere happily, it only checks CSR structure -- but a literal zero radius or
# extent risks a NaN the moment anything downstream normalises it. This is not
# a claim about precision; it is a floor under "cannot be exactly zero".
_MIN_SIZE = 1e-6

_SPHERE_SEGMENTS = 12
_SPHERE_RINGS = 6
_CAPSULE_SEGMENTS = 12
_CAPSULE_RINGS = 3

#: The most **deduplicated** points :func:`_quickhull_core` may be asked to
#: hull at once -- covering all three of its callers: :func:`convex_hull`,
#: :func:`compound` (once per part) and :func:`fit_box`'s oriented PCA fit
#: (via :func:`_hull_points_for_fit`). The 2026-09-19 audit's clay-10: this
#: pure-Python incremental quickhull (see the module docstring: "the simpler
#: O(faces) scan per iteration is not worth an adjacency structure") had no
#: ceiling of its own -- the ``guard_limit = 20 * n + 64`` inside it is a
#: convergence valve against a numerically stuck loop, not a size refusal --
#: so an ordinary imported mesh's vertex count ran unbounded on the frame
#: thread: 2.68 s at 2,000 points, 11.16 s at 20,000 (points sitting near
#: their own eventual hull surface, which the audit's own probe picked as
#: the case that keeps the most points "outside" some face for longest, and
#: which is also what a real organic import's vertices actually look like).
#: Re-measured at merge on this same worst-case distribution: 2.52 s / 10.24
#: s at those two counts; 5,000 points (this ceiling) measures 4.44 s. That
#: is still seconds, not milliseconds -- stated honestly rather than chasing
#: the sub-second bar ``ops_dissolve.MAX_DISSOLVED_RING``/``ops_bevel.
#: MAX_BEVELED_CORNERS`` hold themselves to for a per-edit op, because
#: fitting a collider is the same "deliberate one-shot action" shape
#: ``dev/INVARIANTS.md`` already accepts a bounded multi-second stall for
#: (a large ``clay_boolean``, a whole-document ``clay_analyze``) -- the
#: fix is a bound, not an instant answer. ``_hull_points_for_fit`` already
#: treats any :class:`~.elements.OpError` out of :func:`_quickhull_core` as
#: "fall back to every vertex for the PCA", so ``fit_box(oriented=True)``
#: keeps the module docstring's "refuse nothing" contract past this ceiling
#: too -- only the hull step skips, not the box.
MAX_HULL_POINTS = 5_000


#: The most loose parts one :func:`compound` call will hull.
#:
#: The 2026-09-22 audit, clay-12: ``compound`` already bounds each part's own
#: point count via ``MAX_HULL_POINTS``/``_refuse_hull_complexity``, but not how
#: many parts ``_face_shells`` can hand it -- a kitbashed or boolean-separated
#: mesh with thousands of small loose pieces hulls every one of them in a
#: Python loop on the frame thread (``ops.py``'s ``_collider_op``), with
#: ``clay_collider`` reaching the same door. Reproduced (audit's own probe,
#: one shell each): 0.89 s at 1,000 parts, 5.3 s at 6,000. Re-measured at
#: merge on a distinct-cube-per-part mesh: 0.67 s at 700, 0.77 s at 800,
#: 0.86 s at 900 -- this ceiling keeps one call under a second.
MAX_COMPOUND_PARTS = 800


#: The most faces :func:`_face_shells` will BFS-walk on ``compound``'s
#: auto-grouping path (``face_groups=None``) before it even starts.
#:
#: The 2026-09-23 audit's clay-12: :data:`MAX_COMPOUND_PARTS` above refuses on
#: the *shell count* ``_face_shells`` hands back, but a single-shell mesh --
#: one loose part, however many faces it has -- always reports ``n_shells ==
#: 1``, so that refusal never fires no matter how large the mesh is, and
#: ``_face_shells`` itself (a plain BFS in Python, one ``deque`` pop per face)
#: still walks every face before returning: 0.28s at 131,000 faces, linear, no
#: refusal. Measured 0.28s / 131,000 faces gives roughly 2.1us/face, so the
#: "well under a second" bar every ceiling in this module holds itself to
#: lands around 470,000 faces; this sits at roughly half that, the same
#: margin :data:`MAX_COMPOUND_PARTS`'s own re-measured table keeps (700-900
#: comfortably under the point its BFS-adjacent cost stops being sub-second),
#: and the same order of magnitude as this package's other per-face ceilings
#: (``ops_subdiv.MAX_SUBDIVIDED_FACES`` is 1,000,000, but that op is a single
#: vectorised numpy pass rather than a Python BFS).
MAX_COMPOUND_SHELL_FACES = 250_000


# --- geometry, reused from primitives.py -------------------------------------


def _untextured(mesh: Mesh) -> Mesh:
    """*mesh* with any per-corner UV stripped.

    Every generator in ``primitives.py`` returns a UV-mapped mesh -- it is
    built for a placed object a user might paint -- and a collider is never
    textured (see :func:`_concat_meshes`, which drops UV for the same
    reason when it merges a compound's parts). This is the one adjustment
    :func:`fit_box`, :func:`fit_sphere` and :func:`fit_capsule` make on top
    of what ``primitives.box``/``uv_sphere``/``capsule`` already build; see
    the module docstring for why reusing them, rather than a private copy of
    their geometry, is now the right call.
    """
    return replace(mesh, uv=None) if mesh.uv is not None else mesh


def _affine(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """A 4x4 in :func:`~.mesh.transformed`'s column-vector convention."""
    m = np.eye(4, dtype="f8")
    m[:3, :3] = rotation
    m[:3, 3] = translation
    return m


# --- box ----------------------------------------------------------------------


def _pca_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(axes, mean)`` -- a right-handed orthonormal frame (columns = axes,
    widest spread first) fit to *points* by PCA, and their centroid.

    Sign is canonicalised (each axis flipped, if needed, so the points project
    onto it with a non-negative sum) so the frame is a function of the *point
    set*, not of ``eigh``'s arbitrary sign choice -- two calls on the same
    shape, even reflected numerically, land on the same axes. The third axis
    is always the cross product of the first two, never ``eigh``'s own third
    eigenvector, which is what keeps ``det(axes) == 1`` after both signs are
    canonicalised independently (flipping two of three columns of a
    right-handed frame one at a time can leave it left-handed; recomputing the
    third from the cross product cannot).
    """
    mean = points.mean(axis=0)
    centered = points - mean
    cov = centered.T @ centered
    if not np.any(cov):
        return np.eye(3, dtype="f8"), mean
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    e0 = eigvecs[:, order[0]]
    e1 = eigvecs[:, order[1]]
    if float((centered @ e0).sum()) < 0.0:
        e0 = -e0
    if float((centered @ e1).sum()) < 0.0:
        e1 = -e1
    e2 = np.cross(e0, e1)
    return np.stack([e0, e1, e2], axis=1), mean


def _hull_points_for_fit(mesh: Mesh) -> np.ndarray:
    """The mesh's own convex-hull vertices, falling back to every vertex when
    the mesh is too degenerate for :func:`_quickhull_core` to hull (a flat
    ``plane`` primitive, a line, a point) -- :func:`fit_box`'s oriented mode
    wants the extremal points for a numerically cleaner PCA, but must not
    *refuse* on an input :func:`convex_hull` itself would refuse, since an
    oriented bounding box of a flat or degenerate object is still a perfectly
    good (if thin) box.
    """
    points = mesh.positions.astype("f8")
    if len(points) < 4:
        return points
    try:
        uniq = np.unique(points, axis=0)
        if len(uniq) < 4:
            return points
        _refuse_hull_complexity(len(uniq), "oriented box fit")
        scale = float(max(np.ptp(uniq, axis=0).max(initial=0.0), 1.0))
        hull_idx, _faces = _quickhull_core(uniq, 1e-9 * scale)
        return uniq[hull_idx]
    except OpError:
        return points


def fit_box(mesh: Mesh, *, oriented: bool = False) -> Collider:
    """The tightest box: axis-aligned by default, or PCA-oriented.

    ``oriented=False`` is the object's own AABB (:func:`~.mesh.bounds`) --
    exact, and cheapest. ``oriented=True`` fits a right-handed frame by PCA
    over the mesh's own convex-hull vertices (:func:`_hull_points_for_fit`)
    rather than every vertex: an interior vertex contributes nothing to a
    bounding box's extent but does bias an unweighted PCA's covariance toward
    wherever the mesh happens to be densely tessellated, and the hull is the
    smaller, exactly-as-informative point set for this fit -- the standard
    "OBB via the convex hull's own PCA" construction (the same one
    ``pipelines.postprocess`` reaches for trimesh to do; this is the kernel
    doing the same idea without it). Not the provably-minimal oriented box
    (that is a harder, rotating-calipers-in-3D problem with no closed form);
    a good, cheap, deterministic approximation.
    """
    points = mesh.positions.astype("f8")
    if len(points) == 0:
        raise OpError("A box collider needs at least one vertex to fit.")

    if not oriented:
        lo, hi = bounds(mesh)
        lo, hi = lo.astype("f8"), hi.astype("f8")
        size = np.maximum(hi - lo, _MIN_SIZE)
        center = (lo + hi) / 2.0
        rotation = np.eye(3, dtype="f8")
    else:
        hull_points = _hull_points_for_fit(mesh)
        rotation, mean = _pca_frame(hull_points)
        local = (hull_points - mean) @ rotation
        lo, hi = local.min(axis=0), local.max(axis=0)
        size = np.maximum(hi - lo, _MIN_SIZE)
        center = mean + rotation @ ((lo + hi) / 2.0)

    out = transformed(_untextured(primitives.box(size)), _affine(rotation, center))
    params = {
        "size": tuple(float(s) for s in size),
        "center": tuple(float(c) for c in center),
        "rotation": rotation.tolist(),
        "oriented": oriented,
    }
    return Collider(kind="box", mesh=out, params=params)


# --- sphere ---------------------------------------------------------------


def _bounding_sphere(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Ritter's bounding sphere: a single deterministic pass, not the minimal
    sphere, but *provably containing every point*.

    Seeded from two farthest-apart-ish points (farthest from an arbitrary
    start, then farthest from that), the loop below then visits every point
    once in array order and grows the sphere whenever a point sits outside
    it. That growth step keeps the whole previous sphere inside the new one:
    for any point ``Q`` already inside (``|Q - C| <= r``), the new centre
    ``C'`` moves only ``R' - r`` away from ``C`` (by construction, straight
    towards the offending point), so ``|Q - C'| <= |Q - C| + |C - C'| <= r +
    (R' - r) = R'`` by the triangle inequality alone. Every point visited
    stays inside every sphere built afterwards, so the final sphere contains
    every point seen -- by induction over the pass, not by re-checking at the
    end. (Real-Time Collision Detection, Ericson, s.4.3.2 -- the standard
    citation for this construction.)

    Deterministic (array order, no randomisation), and O(n) with a Python
    loop: for a mesh past a few hundred thousand vertices, hulling first
    (:func:`convex_hull`) and fitting the sphere to *that* mesh's positions
    is exactly as correct -- a bounding sphere is decided entirely by the
    extremal points, which are exactly what a convex hull keeps -- and far
    fewer points to loop over.
    """
    p0 = points[0]
    d = np.linalg.norm(points - p0, axis=1)
    y = points[int(np.argmax(d))]
    d2 = np.linalg.norm(points - y, axis=1)
    z = points[int(np.argmax(d2))]
    center = (y + z) / 2.0
    radius = float(np.linalg.norm(z - y)) / 2.0

    for p in points:
        dd = float(np.linalg.norm(p - center))
        if dd > radius:
            new_radius = (radius + dd) / 2.0
            k = (new_radius - radius) / dd
            center = center + k * (p - center)
            radius = new_radius

    # A small safety margin against float round-trip at the boundary -- this
    # can only make containment *more* true, never less.
    radius = radius * (1.0 + 1e-7) + 1e-7
    return center, radius


def fit_sphere(
    mesh: Mesh, *, segments: int = _SPHERE_SEGMENTS, rings: int = _SPHERE_RINGS
) -> Collider:
    """The smallest sphere :func:`_bounding_sphere` finds, as a :class:`Collider`."""
    points = mesh.positions.astype("f8")
    if len(points) == 0:
        raise OpError("A sphere collider needs at least one vertex to fit.")
    center, radius = _bounding_sphere(points)
    radius = max(radius, _MIN_SIZE)
    sphere = _untextured(primitives.uv_sphere(radius, segments=segments, rings=rings))
    out = transformed(sphere, _affine(np.eye(3), center))
    params = {"radius": float(radius), "center": tuple(float(c) for c in center)}
    return Collider(kind="sphere", mesh=out, params=params)


# --- capsule ----------------------------------------------------------------


def _perp_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors completing *axis* (unit) into a right-handed frame
    with *axis* as the middle (Y) column -- ``(e_x, e_z)`` with
    ``e_x x axis == e_z``."""
    ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    e_x = np.cross(ref, axis)
    e_x = e_x / np.linalg.norm(e_x)
    e_z = np.cross(e_x, axis)
    return e_x, e_z


def fit_capsule(
    mesh: Mesh, *, segments: int = _CAPSULE_SEGMENTS, rings: int = _CAPSULE_RINGS
) -> Collider:
    """A capsule -- axis by PCA, radius and half-height covering every point.

    The axis is the largest-variance PCA direction. Given that axis and the
    centroid, the radius that makes an *infinite cylinder* around the axis
    cover every point is exactly the largest perpendicular distance to the
    axis line -- a fact independent of where the segment's endpoints end up,
    since perpendicular distance to a line does not depend on which point on
    it is used as a reference. Once that radius is fixed, each point *i* at
    axial position ``t_i`` and perpendicular distance ``d_i`` needs the near
    segment endpoint no closer than ``t_i - sqrt(r^2 - d_i^2)`` (the point on
    the axis where a sphere of that radius, centred there, just reaches it) --
    taking the max/min of that bound over every point gives the tightest
    ``[t_lo, t_hi]`` that keeps every point within ``r`` of the *segment*, not
    just the infinite line. Not the minimal capsule (that would also optimise
    the axis itself against the resulting volume); a good, cheap, always-
    containing one.
    """
    points = mesh.positions.astype("f8")
    if len(points) == 0:
        raise OpError("A capsule collider needs at least one vertex to fit.")

    mean = points.mean(axis=0)
    if len(points) == 1:
        axis = np.array([0.0, 1.0, 0.0])
    else:
        centered = points - mean
        cov = centered.T @ centered
        if not np.any(cov):
            axis = np.array([0.0, 1.0, 0.0])
        else:
            eigvals, eigvecs = np.linalg.eigh(cov)
            axis = eigvecs[:, int(np.argmax(eigvals))]
            norm = float(np.linalg.norm(axis))
            axis = axis / norm if norm > 0.0 else np.array([0.0, 1.0, 0.0])
            # Canonicalise sign the same way ``_pca_frame`` does, so the same
            # shape fit twice (or fit and re-fit after an equivalent op)
            # lands on the same axis rather than its negation.
            if float((centered @ axis).sum()) < 0.0:
                axis = -axis

    e_x, e_z = _perp_basis(axis)
    rotation = np.stack([e_x, axis, e_z], axis=1)

    local = (points - mean) @ rotation
    t = local[:, 1]
    perp = np.linalg.norm(local[:, [0, 2]], axis=1)
    radius = max(float(perp.max()) if len(perp) else 0.0, _MIN_SIZE)
    cap = np.sqrt(np.maximum(radius * radius - perp * perp, 0.0))
    t_hi = float((t - cap).max())
    t_lo = float((t + cap).min())
    if t_lo > t_hi:
        t_lo = t_hi = (t_lo + t_hi) / 2.0
    half_height = max((t_hi - t_lo) / 2.0, 0.0)
    center_t = (t_hi + t_lo) / 2.0
    center = mean + center_t * axis

    # Same safety margin as ``_bounding_sphere``, for the same reason.
    radius = radius * (1.0 + 1e-7) + 1e-7

    out = transformed(
        _untextured(primitives.capsule(radius, 2.0 * half_height, segments=segments, rings=rings)),
        _affine(rotation, center),
    )
    params = {
        "radius": float(radius),
        "half_height": float(half_height),
        "axis": tuple(float(a) for a in axis),
        "center": tuple(float(c) for c in center),
    }
    return Collider(kind="capsule", mesh=out, params=params)


# --- convex hull --------------------------------------------------------------


def _plane_of(points: np.ndarray, face: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    i, j, k = face
    pi, pj, pk = points[i], points[j], points[k]
    return np.cross(pj - pi, pk - pi), pi


def _refuse_hull_complexity(n_points: int, kind: str) -> None:
    """Refuse before :func:`_quickhull_core` runs, from the (already
    deduplicated) point count it would face -- the same shape
    :func:`~.ops_boolean._refuse_complexity` uses: a cheap count read
    before the expensive call, not a check woven into it. See
    :data:`MAX_HULL_POINTS`'s own docstring for the incident and the
    measurements behind the number.
    """
    if n_points > MAX_HULL_POINTS:
        raise OpError(
            f"This {kind} would need to hull {n_points:,} points, past the "
            f"{MAX_HULL_POINTS:,} Clay works with. Simplify the mesh first, "
            "or select fewer objects."
        )


def _quickhull_core(
    points: np.ndarray, eps: float
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """The incremental (Quickhull) 3D convex hull of *points*.

    Returns ``(hull_point_indices, faces)`` -- ``hull_point_indices`` sorted,
    into *points*; ``faces`` triangle index-triples, also into *points*
    (**not** yet remapped to ``0..len(hull_point_indices)``, which is the
    caller's job -- :func:`_hull_from_points` does it once for both the
    reduced and unreduced paths).

    The classic algorithm: seed a tetrahedron from four extremal,
    non-coplanar points, then repeatedly pick the farthest point outside any
    current face, delete every face it can see (found by a *direct scan* of
    every live face rather than a horizon-edge BFS with an adjacency table --
    a live hull here is a few dozen faces at most, not a per-frame cost, so
    the simpler O(faces) scan per iteration is not worth an adjacency
    structure to avoid), stitch a new face from that point to every boundary
    ("horizon") edge of the hole, and redistribute the deleted faces'
    outside points among the new faces. A horizon edge is a directed edge
    that belongs to exactly one visible face -- the standard trick that reads
    the boundary loop off a set difference rather than a walk, and it works
    because the visible region on a convex polytope is always one connected
    patch.
    """
    n = len(points)
    if n < 4:
        raise OpError(f"A convex hull needs at least 4 points, got {n}.")

    mins = points.argmin(axis=0)
    maxs = points.argmax(axis=0)
    cand = sorted(set(mins.tolist()) | set(maxs.tolist()))
    best_d, p0, p1 = -1.0, cand[0], cand[0]
    for ci in range(len(cand)):
        for cj in range(ci + 1, len(cand)):
            i, j = cand[ci], cand[cj]
            d = float(np.linalg.norm(points[i] - points[j]))
            if d > best_d:
                best_d, p0, p1 = d, i, j
    if best_d <= eps:
        raise OpError(
            "Every point coincides; a convex hull needs at least 4 non-coplanar points."
        )

    a, b = points[p0], points[p1]
    ab = b - a
    dist_line = np.linalg.norm(np.cross(points - a, ab), axis=1)
    p2 = int(np.argmax(dist_line))
    if dist_line[p2] <= eps * max(1.0, float(np.linalg.norm(ab))):
        raise OpError(
            "Every point is collinear; a convex hull needs at least 4 non-coplanar points."
        )

    c = points[p2]
    normal0 = np.cross(b - a, c - a)
    normal0_len = float(np.linalg.norm(normal0))
    dist_plane = (points - a) @ normal0 / normal0_len
    p3 = int(np.argmax(np.abs(dist_plane)))
    if abs(dist_plane[p3]) <= eps:
        raise OpError(
            "Every point is coplanar; a convex hull needs at least 4 non-coplanar points."
        )

    tet = (p0, p1, p2, p3)
    centroid = points[list(tet)].mean(axis=0)

    def outward(i: int, j: int, k: int) -> tuple[int, int, int]:
        pi, pj, pk = points[i], points[j], points[k]
        n_ = np.cross(pj - pi, pk - pi)
        return (i, j, k) if n_ @ (pi - centroid) >= 0.0 else (i, k, j)

    faces: dict[int, tuple[int, int, int]] = {
        idx: outward(*tri)
        for idx, tri in enumerate([(p0, p1, p2), (p0, p2, p3), (p0, p3, p1), (p1, p3, p2)])
    }
    next_id = 4

    outside: dict[int, list[int]] = {fid: [] for fid in faces}
    hull_members = set(tet)
    for idx in range(n):
        if idx in hull_members:
            continue
        for fid, f in faces.items():
            nrm, p0f = _plane_of(points, f)
            if (points[idx] - p0f) @ nrm > eps:
                outside[fid].append(idx)
                break

    guard, guard_limit = 0, 20 * n + 64
    while True:
        guard += 1
        if guard > guard_limit:
            raise OpError(
                "Convex hull construction did not converge; the input may be numerically "
                "degenerate."
            )
        fid = next((f for f, lst in outside.items() if lst), None)
        if fid is None:
            break
        lst = outside[fid]
        nrm, p0f = _plane_of(points, faces[fid])
        dists = (points[lst] - p0f) @ nrm
        eye = lst[int(np.argmax(dists))]

        visible = []
        for gid, gf in faces.items():
            gnrm, gp0 = _plane_of(points, gf)
            if (points[eye] - gp0) @ gnrm > eps:
                visible.append(gid)

        edge_owner: dict[tuple[int, int], int] = {}
        for gid in visible:
            i, j, k = faces[gid]
            edge_owner[(i, j)] = gid
            edge_owner[(j, k)] = gid
            edge_owner[(k, i)] = gid
        horizon = [(u, v) for (u, v) in edge_owner if (v, u) not in edge_owner]

        orphan: list[int] = []
        for gid in visible:
            orphan.extend(outside.pop(gid, []))
            faces.pop(gid, None)
        orphan = [p for p in orphan if p != eye]
        hull_members.add(eye)

        new_ids = []
        for u, v in horizon:
            faces[next_id] = (u, v, eye)
            outside[next_id] = []
            new_ids.append(next_id)
            next_id += 1

        for p in orphan:
            for gid in new_ids:
                nrm, p0f = _plane_of(points, faces[gid])
                if (points[p] - p0f) @ nrm > eps:
                    outside[gid].append(p)
                    break

    hull_idx = np.array(sorted(hull_members), dtype="i8")
    ordered_faces = [faces[fid] for fid in sorted(faces)]
    return hull_idx, ordered_faces


def _farthest_point_sample(points: np.ndarray, k: int) -> np.ndarray:
    """Indices of *k* points from *points*, greedily maximising the distance
    to everything already chosen. Deterministic: seeded from the point
    farthest from the centroid, every following pick the (first, on a tie)
    point with the largest current minimum distance to the chosen set."""
    n = len(points)
    if k >= n:
        return np.arange(n)
    centroid = points.mean(axis=0)
    first = int(np.argmax(np.linalg.norm(points - centroid, axis=1)))
    chosen = [first]
    min_dist = np.linalg.norm(points - points[first], axis=1)
    for _ in range(1, k):
        nxt = int(np.argmax(min_dist))
        chosen.append(nxt)
        min_dist = np.minimum(min_dist, np.linalg.norm(points - points[nxt], axis=1))
    return np.array(chosen, dtype="i8")


def _hull_from_points(
    points: np.ndarray, max_faces: int, *, kind: str = "convex hull"
) -> tuple[Mesh, dict[str, Any]]:
    pts = np.asarray(points, dtype="f8")
    uniq = np.unique(pts, axis=0)
    if len(uniq) < 4:
        raise OpError(
            f"A convex hull needs at least 4 distinct points, got {len(uniq)}."
        )
    _refuse_hull_complexity(len(uniq), kind)
    scale = float(max(np.ptp(uniq, axis=0).max(initial=0.0), 1.0))
    eps = 1e-9 * scale

    hull_idx, faces = _quickhull_core(uniq, eps)
    reduced = len(faces) > max_faces
    if reduced:
        target_v = max(4, max_faces // 2 + 2)
        sample = _farthest_point_sample(uniq[hull_idx], target_v)
        sub_points = uniq[hull_idx][sample]
        sub_hull_idx, faces = _quickhull_core(sub_points, eps)
        final_points = sub_points[sub_hull_idx]
        remap = {int(old): new for new, old in enumerate(sub_hull_idx)}
    else:
        final_points = uniq[hull_idx]
        remap = {int(old): new for new, old in enumerate(hull_idx)}
    faces = [(remap[i], remap[j], remap[k]) for i, j, k in faces]

    mesh = from_faces(final_points, [list(f) for f in faces])
    params = {
        "vertex_count": len(final_points),
        "face_count": len(faces),
        "reduced": reduced,
    }
    return mesh, params


def convex_hull(mesh: Mesh, *, max_faces: int = 64) -> Collider:
    """The convex hull of *mesh*'s vertices, triangulated, reduced (by
    :func:`_farthest_point_sample`, see the module docstring) to at most
    *max_faces* triangles.

    Refuses (:class:`~.elements.OpError`) fewer than four non-coplanar
    points -- there is no hull to build under that, not an approximate one.
    """
    hull_mesh, params = _hull_from_points(mesh.positions, max_faces)
    return Collider(kind="convex", mesh=hull_mesh, params=params)


# --- compound -----------------------------------------------------------------


def _face_shells(mesh: Mesh) -> tuple[np.ndarray, int]:
    """Which connected group of faces (crossing only a shared, 2-use edge)
    each face belongs to -- a "loose part" in the same sense ``ops_clean``'s
    winding BFS walks (see that module's own docstring), computed
    independently here (a plain BFS over :func:`~.adjacency.adjacency`'s
    ``twin`` table, not that module's private ``_shell_and_flip``) because
    this only ever needs *which* shell, never the flip bookkeeping that
    exists there for winding correction.
    """
    n_faces = len(mesh.starts) - 1
    if n_faces == 0:
        return np.zeros(0, dtype="i8"), 0

    a = adjacency(mesh)
    twin = a.twin.astype("i8")
    has_twin = twin >= 0
    c1 = np.flatnonzero(has_twin)
    c2 = twin[c1]
    f1 = a.corner_face[c1].astype("i8")
    f2 = a.corner_face[c2].astype("i8")

    all_a = np.concatenate([f1, f2])
    all_b = np.concatenate([f2, f1])
    order = np.argsort(all_a, kind="stable")
    nbr = all_b[order]
    starts = np.searchsorted(all_a[order], np.arange(n_faces + 1))

    shell = np.full(n_faces, -1, dtype="i8")
    n_shells = 0
    for seed in range(n_faces):
        if shell[seed] != -1:
            continue
        shell[seed] = n_shells
        queue: deque[int] = deque([seed])
        while queue:
            f = queue.popleft()
            for k in range(int(starts[f]), int(starts[f + 1])):
                g = int(nbr[k])
                if shell[g] == -1:
                    shell[g] = n_shells
                    queue.append(g)
        n_shells += 1
    return shell, n_shells


def _corners_of_faces(mesh: Mesh, face_idx: np.ndarray) -> np.ndarray:
    starts = mesh.starts.astype("i8")
    parts = [np.arange(starts[f], starts[f + 1]) for f in face_idx.tolist()]
    return np.concatenate(parts) if parts else np.zeros(0, dtype="i8")


def _concat_meshes(meshes: Sequence[Mesh]) -> Mesh:
    """Every mesh's faces, as one CSR -- disjoint shells in one ``Mesh``,
    the same shape ``ops_clean``'s own "many boxes in one mesh" measurement
    table already exercises. UVs are dropped: a collider is never textured.
    """
    if not meshes:
        raise OpError("Nothing to merge into a compound collider.")
    positions, loops, material, smooth = [], [], [], []
    starts = [0]
    voffset = 0
    for m in meshes:
        positions.append(m.positions)
        loops.append(m.loops.astype("i8") + voffset)
        starts.extend((m.starts[1:].astype("i8") + starts[-1]).tolist())
        material.append(m.material)
        smooth.append(m.smooth)
        voffset += len(m.positions)
    return Mesh(
        positions=np.concatenate(positions, axis=0),
        loops=np.concatenate(loops).astype("i4"),
        starts=np.array(starts, dtype="i4"),
        material=np.concatenate(material).astype("i4"),
        smooth=np.concatenate(smooth).astype("?"),
        uv=None,
    )


def compound(
    mesh: Mesh,
    face_groups: Sequence[Sequence[int] | np.ndarray] | None = None,
    *,
    max_faces: int = 64,
) -> Collider:
    """One convex hull per loose part -- a *compound* collider.

    *face_groups* names the parts explicitly (each a sequence of face
    indices into *mesh*); left as ``None``, the parts are found from
    *mesh*'s own connectivity (:func:`_face_shells`) -- the "island" a
    boolean-separated or kitbashed mesh already is, with no user-supplied
    grouping needed. ``Collider.mesh`` is every part's hull concatenated into
    one preview mesh (disjoint shells, each individually convex, the whole
    not); ``Collider.params["parts"]`` is the tuple of individual
    ``kind="convex"`` :class:`Collider` objects an exporter actually wants --
    Unreal's ``UCX_<Mesh>_00``, ``_01``... naming is written for exactly this
    shape, one indexed convex node per part (see ``engines.py``).
    """
    if face_groups is None:
        # See MAX_COMPOUND_SHELL_FACES's own comment: a single-shell mesh
        # never trips MAX_COMPOUND_PARTS below (n_shells == 1, whatever the
        # face count), so the BFS itself needs its own cheap-count refusal,
        # checked before it walks a single face.
        n_faces = len(mesh.starts) - 1
        if n_faces > MAX_COMPOUND_SHELL_FACES:
            raise OpError(
                f"Compound would scan {n_faces:,} faces to find loose parts, "
                f"past the {MAX_COMPOUND_SHELL_FACES:,} it works with before "
                "stalling the frame it runs on. Select fewer faces, or pass "
                "explicit face groups instead of grouping by connectivity."
            )
        shell, n_shells = _face_shells(mesh)
        if n_shells == 0:
            raise OpError("Compound needs at least one face to group into parts.")
        groups = [np.flatnonzero(shell == s) for s in range(n_shells)]
    else:
        if not face_groups:
            raise OpError("Compound needs at least one face group.")
        n_faces = len(mesh.starts) - 1
        groups = []
        for g in face_groups:
            arr = np.asarray(g, dtype="i8")
            # The 2026-09-20 audit, finding clay-14: an out-of-range face id
            # here used to reach ``_corners_of_faces``'s ``mesh.starts[f]``
            # indexing as a bare ``IndexError`` instead of this module's own
            # named ``OpError`` refusal every other malformed input raises,
            # and a *negative* id did not raise at all -- it wrapped through
            # numpy's own negative-index semantics onto an unrelated face at
            # the far end of the mesh, silently, with no error anywhere.
            if arr.size and (int(arr.min()) < 0 or int(arr.max()) >= n_faces):
                raise OpError(
                    f"A compound face group names a face index outside 0..{n_faces - 1}."
                )
            groups.append(arr)

    # See MAX_COMPOUND_PARTS's own comment: refuse on the part *count* before
    # any hull runs, the same "cheap count read before the expensive call"
    # shape _refuse_hull_complexity already uses per part.
    if len(groups) > MAX_COMPOUND_PARTS:
        raise OpError(
            f"Compound would hull {len(groups):,} loose parts, past the "
            f"{MAX_COMPOUND_PARTS:,} Clay works with. Merge or simplify the "
            "mesh first, or select fewer parts."
        )

    parts: list[Collider] = []
    for g in groups:
        idx = np.unique(mesh.loops[_corners_of_faces(mesh, g)])
        hull_mesh, params = _hull_from_points(mesh.positions[idx], max_faces, kind="compound part")
        parts.append(Collider(kind="convex", mesh=hull_mesh, params=params))

    merged = _concat_meshes([p.mesh for p in parts])
    params = {"parts": tuple(parts), "count": len(parts)}
    return Collider(kind="compound", mesh=merged, params=params)


# --- registry -------------------------------------------------------------


#: name -> (label, fit function, extra keyword defaults) -- the same shape
#: ``primitives.GENERATORS`` is, and for the same reason: the Clay UI/agent
#: surface this tranche hands off to derives its collider menu and its
#: parameter defaults from here rather than hard-coding a second list that
#: could drift the day a sixth collider kind is added.
COLLIDER_KINDS: dict[str, tuple[str, Any, dict[str, Any]]] = {
    "box": ("Box", fit_box, {"oriented": False}),
    "sphere": ("Sphere", fit_sphere, {}),
    "capsule": ("Capsule", fit_capsule, {}),
    "convex": ("Convex Hull", convex_hull, {"max_faces": 64}),
    "compound": ("Compound", compound, {"max_faces": 64}),
}
