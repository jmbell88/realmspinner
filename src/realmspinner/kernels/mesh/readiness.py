"""Is this document safe to hand to a game engine, and for which one.

**One pure function, called from three places with the same answer.** A
Properties-panel badge wants this on click, `clay_diagnose`-style an agent
tool wants it over the wire, and the eventual "Fix" button wants to know
*which op* addresses a given warning without re-deriving that mapping in the
UI layer. All three get exactly one function -- :func:`validate` -- because a
readiness verdict that the panel and the agent could compute differently is
worse than no verdict at all: an agent telling a user "this is fine" while
the panel shows a warning is a contradiction with nobody's name on it. Every
:class:`Check` that names a remedy names an *existing* Clay op
(:data:`FIX_OPS`), never a made-up verb, so a caller can run it rather than
just print it.

**Nothing here mutates, and only one thing refuses.** Unlike :mod:`.ops_clean`
(which fixes) or :mod:`.ops_boolean` (which refuses past a ceiling), this
module almost always only *measures*, the same doctrine :mod:`.diagnose` and
:mod:`.analyze` are written under -- see their own docstrings. A readiness
read that silently skipped an expensive object rather than refusing outright
is not new behaviour invented here either: it is the same "measure, do not
stall" answer :mod:`.analyze` gives for :data:`~.analyze.MAX_ANALYZE_TRIANGLES`,
just landing as a per-check ``"skip"`` instead of a raised
:class:`~.elements.OpError`, because a readiness report that refused outright
the moment one object in a forty-object scene was oversized would be a worse
answer than "everything else checked out; here is what did not run and why."
The one exception is :data:`MAX_VALIDATE_OBJECTS`, a document-*wide* object
count rather than one object's own size -- see its own docstring for why a
per-object skip cannot answer the case it exists for.

**Why a check fails rather than warns.** Every check here defaults to
``warn`` -- an engine budget is advice, not a correctness rule, and the
profile numbers below are informed judgement, not physical law. Exactly two
checks can reach ``fail``, because they are not budgets:

* ``objects`` -- an empty document is not "over budget," it is nothing to
  export. Every other check is marked ``skip`` rather than silently scored
  against zero objects, which would otherwise read as "nothing wrong" for a
  document that has *everything* wrong with it.
* ``uvs`` -- a textured material on an object with no UV coordinates is not a
  matter of taste. Every target engine either drops the texture or paints
  the object a single (undefined) colour; there is no "acceptable" reading of
  that combination, so it is reported as a defect, not a recommendation. The
  reverse (an untextured object with no UVs) is completely normal -- Clay's
  own primitives ship with none -- so it is a bare ``warn`` at worst.

Everything else -- triangle and vertex counts, material and texture budgets,
degenerate geometry, inconsistent winding, open edges, non-uniform scale, an
out-of-range physical size, a pivot not on the ground -- genuinely depends on
the target and the asset's role, so the worst any of them says is ``warn``.

**Cost.** The four checks built on :mod:`.ops_clean` and
:func:`~.adjacency.check_manifold` (``geometry``, ``normals``, ``closed``)
share the O(corners) BFS/adjacency cost those two already pay, cached the
same way -- see :mod:`.ops_clean`'s own measurement tables for what one
object at :data:`~.ops_clean.MAX_CLEAN_CORNERS` costs. Rather than a fourth
ceiling, this module reuses that one: **any visible object past
``ops_clean.MAX_CLEAN_CORNERS`` corners skips those three checks outright**
(status ``"skip"``, the message naming the ceiling) rather than running the
BFS on an object :mod:`.ops_clean` itself already refuses to walk. The
``vertices`` check makes the same per-object call for a cheaper reason: it
wants the *render* vertex count (:func:`~.mesh.render_arrays`, which
triangulates and computes normals), and an object past the same ceiling
instead contributes its raw position count -- a true lower bound, said as one
in the message, rather than a stall. ``triangles`` needs neither: a
face's contribution to the triangle count is ``corners - 2``, summed from
``mesh.loops``/``mesh.starts`` alone, the same O(1)-per-object arithmetic
:func:`~.analyze.analyze` already uses to refuse *before* triangulating
(see its own ``total_tris`` comment). ``scale``/``pivot`` transform every
vertex once (:func:`~.ops.world_positions`) but do no BFS, so they are not
gated by the ceiling at all -- a bare matrix multiply per vertex is the
cheapest thing this module does.

Measured on this machine, one ``uv_sphere`` object at 297,990 corners (just
under :data:`~.ops_clean.MAX_CLEAN_CORNERS`, so nothing above skips),
``godot-desktop``, ten runs after a warm-up call: :func:`validate` averages
**~490 ms**, comfortably under the ~1 s whole-document budget this module is
built to, with the first (cold-cache) call at ~600 ms. Unlike
:func:`~.ops_clean.clean`, none of this module's own per-object calls are
memoised against the mesh -- ``ops_clean.survey`` recomputes its face-defect
masks every time it is asked, and :func:`~.mesh.render_arrays` (called
directly here, not through :func:`~.document.render_plan`'s weak-keyed cache)
re-triangulates from scratch -- so this is the *uncached* cost, which is what
a one-shot panel button or agent tool call actually pays. Broken down at the
same 297,990 corners: ``survey`` alone is ~290 ms (the winding-consistency
BFS and its divergence-theorem volume pass, the same cost
:mod:`.ops_clean`'s own measurement table describes), ``render_arrays`` is
~155 ms (triangulation and normal accumulation), ``check_manifold`` is only
~45 ms because it reuses the :class:`~.adjacency.Adjacency` ``survey``
already built and cached against this mesh, and
:func:`~.ops.world_positions` is noise at ~3 ms.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ..geom3d import math3d as m3
from . import mesh as bm
from . import ops, ops_clean, primitives
from .adjacency import check_manifold
from .elements import OpError

__all__ = [
    "CHECKS",
    "DEFAULT_PROFILE",
    "FIX_OPS",
    "MAX_VALIDATE_CORNERS",
    "MAX_VALIDATE_OBJECTS",
    "PROFILES",
    "Check",
    "Profile",
    "Report",
    "validate",
]


MAX_VALIDATE_OBJECTS = 1_000
"""The most visible objects one :func:`validate` call may face at once --
:mod:`.analyze`'s :data:`~.analyze.MAX_ANALYZE_OBJECTS` shape, applied here
for the first time. The 2026-09-20 audit's clay-05 measured :func:`validate`
against documents of only ordinary-sized *objects* (not the oversized-mesh
case :data:`~.ops_clean.MAX_CLEAN_CORNERS` already guards): 522 ms at 50
objects, 3.7 s at 800, 15.1 s at 3,200 -- because every check here still
walks every object once (a ``survey`` call apiece for ``geometry``/
``normals``/``closed`` alone), and nothing before this ceiling existed
bounded how many objects there could be, only how big one of them could be.
1,000 sits close to the 800-object measurement (3.7 s) rather than the
3,200-object one (15.1 s): a single Check press tying up a worker thread for
multiple seconds is already a bad trade for a whole-document button, and this
module's own cost section states a ~1 s whole-document budget it is built to.

This is the one exception to this module's own "nothing here mutates or
refuses" doctrine (see the module docstring) -- past this many objects the
per-object work is not merely slow, it is a document whose size answers its
own readiness question before the checks even run: an asset a game engine
imports as *one thing* does not have a thousand separate objects in it.
:func:`~.mode.check_readiness` (``studio/modes/clay/mode.py``) now also runs
this off the frame thread via ``TaskRunner`` (the same audit's other half of
clay-05), so this ceiling is not a frame-stall guard the way
:data:`~.analyze.MAX_ANALYZE_OBJECTS` is -- it exists so one Check press
cannot tie up a task-pool worker for double-digit seconds on a document this
disproportionate, and so a user who somehow ends up with one is told why
rather than waiting it out."""


MAX_VALIDATE_CORNERS = 2_500_000
"""The most corners, summed across every visible object, one :func:`validate`
call may face at once -- the axis :data:`MAX_VALIDATE_OBJECTS` leaves open.
The 2026-09-20 audit's clay-05 bounded object *count*; the 2026-09-22 audit's
clay-13 found that bound alone still lets a document of objects each just
under :data:`~.ops_clean.MAX_CLEAN_CORNERS` run for minutes, because
``geometry``/``normals``/``closed`` pay a ``survey`` call per object and
nothing before this ceiling summed that cost across the document. Measured
on this machine (``godot-desktop``, objects each just under the corner
ceiling): ~465 ms/object, 9.3 s at 20 objects (5.83M corners), ~8 min
extrapolated to 1,000 objects at :data:`MAX_VALIDATE_OBJECTS`. This ceiling
sits below that 20-object, 9.3 s measurement (2.5M vs. 5.83M corners) so one
Check press stays in the low seconds even in the worst case the per-object
corner ceiling alone allows through, matching :data:`MAX_VALIDATE_OBJECTS`'s
own "sits close to the few-seconds measurement, not the double-digit one"
reasoning."""


CHECKS: tuple[str, ...] = (
    "objects",
    "triangles",
    "vertices",
    "materials",
    "textures",
    "uvs",
    "geometry",
    "normals",
    "closed",
    "transforms",
    "scale",
    "pivot",
    # Tranche 7 (export profiles): three rows about the document's own
    # collider *objects* (``Obj.role == "collider"``, see ``colliders.py``/
    # ``engines.py``), appended rather than interleaved so every existing
    # index into ``CHECKS`` a caller may have cached stays valid.
    "collider_present",
    "collider_triangles",
    "collider_convex",
)

#: Every op name a :class:`Check` may name in ``fix``, gated bidirectionally
#: by ``test_every_fix_a_check_can_return_is_in_fix_ops`` (drives a warn/fail
#: path on every check that ever returns a ``fix`` and asserts the set of
#: names it saw equals this one exactly, both directions in one comparison)
#: -- the same ``OPEN_GENERATORS``/``CONCAVE_GENERATORS`` shape
#: ``primitives.py`` polices its own registry data with, for the same reason:
#: an op name here that no check actually returns is a promise a "Fix" button
#: would offer and nothing would honour.
FIX_OPS: frozenset[str] = frozenset(
    {"clean-mesh", "recalc-normals", "decimate", "bake", "drop-to-ground", "unwrap"}
)

_LABELS: dict[str, str] = {
    "objects": "Objects",
    "triangles": "Triangles",
    "vertices": "Vertices",
    "materials": "Materials",
    "textures": "Textures",
    "uvs": "UVs",
    "geometry": "Geometry",
    "normals": "Normals",
    "closed": "Closed",
    "transforms": "Transforms",
    "scale": "Scale",
    "pivot": "Pivot",
    "collider_present": "Colliders",
    "collider_triangles": "Collider triangles",
    "collider_convex": "Collider hull size",
}


@dataclass(frozen=True)
class Profile:
    """One target's budgets. Every number below is a *recommendation* this
    module warns against crossing, not a hard engine ceiling -- see each
    entry in :data:`PROFILES` for where its numbers come from."""

    key: str
    label: str
    triangles_warn: int
    triangles_fail: int
    vertices_warn: int
    materials_warn: int
    texture_px_warn: int  # largest texture edge, in pixels
    texture_bytes_warn: int  # decoded RGBA bytes, summed over distinct textures
    size_min_m: float  # plausible largest-dimension floor for one asset
    size_max_m: float  # plausible largest-dimension ceiling for one asset
    # Tranche 7 (export profiles): a collision proxy's own budget, judgement
    # like every other *_warn field above but scoped an order of magnitude
    # below the render one on purpose -- a physics shape this dense defeats
    # the point of having a separate, cheap-to-simulate collider at all.
    # Defaulted (unlike every field above) so a ``Profile`` built by hand
    # before this tranche -- ``test_readiness.py`` builds several directly,
    # e.g. ``Profile(**{**base.__dict__, "triangles_warn": 1, ...})`` -- keeps
    # constructing without learning about colliders it never asked about.
    collider_triangles_warn: int = 2_000
    # PhysX (Unity's and Unreal's shared physics backend -- see
    # ``colliders.py``'s own module docstring for the exact confidence: "the
    # commonly cited PhysX convex-mesh ceiling... not independently verified
    # against NVIDIA's current SDK docs this session") caps a convex
    # collision mesh at 255 vertices. ``None`` where this session found no
    # citable per-engine ceiling to check against (Godot builds its own
    # ``ConvexPolygonShape3D`` from the mesh with no documented vertex limit;
    # neither WebGL engine ships a first-party physics backend at all) --
    # "no known cap", not "no cap", and stated as a skip rather than a
    # fabricated pass for exactly that reason (see ``_check_collider_convex_cap``).
    collider_convex_vertex_cap: int | None = None


# Every profile shares the same size floor and the same "judgement" caveat on
# it: nothing in any engine's own documentation states a physically plausible
# size for "one game asset" -- that is a fact about the *scene* an artist is
# building, not the engine. 0.01 m (1 cm, a coin) is this module's own
# judgement call for the floor, chosen to catch a hand-authored asset meant
# to be centimetre-scale but read as metres (a 100x-too-small import). The
# ceiling is narrower than the widest end of that same rough guide (0.01-200
# m) on purpose: a single *asset* -- one prop, one character, one vehicle --
# handed to Trellis or authored in Clay is not a whole terrain chunk (those
# are tiled from many assets, each individually well inside this ceiling), so
# desktop-class profiles cap at 50 m and mobile-class ones tighter still, both
# comfortably catching the mistake this check exists to catch: an SDXL
# reference asset imported at centimetre scale, 100x too big. Revisit with
# real measurements once a body of exported assets exists to calibrate
# against.
_SIZE_MIN_M = 0.01
_SIZE_MAX_DESKTOP_M = 50.0
_SIZE_MAX_MOBILE_M = 20.0

PROFILES: dict[str, Profile] = {
    "godot-desktop": Profile(
        key="godot-desktop",
        label="Godot (Desktop)",
        # Godot publishes no per-mesh triangle ceiling for desktop targets --
        # unlike Unity/WebGL's index-format limits below, there is no single
        # citable number here. 40k/200k is this module's own judgement call,
        # read off the "rough guide" a desktop asset budget conversation
        # keeps landing on (a hero prop comfortably fits under 40k; 200k is
        # where a single draw call starts dominating a frame on mid-range
        # desktop hardware). Revisit with measurements once Realmspinner's own
        # exported assets give a real distribution to calibrate against.
        triangles_warn=40_000,
        triangles_fail=200_000,
        vertices_warn=40_000,  # judgement, the same order as triangles_warn
        materials_warn=8,  # judgement: fewer draw calls per object
        # 4096 is the de-facto safe ceiling common practice converges on for
        # desktop: it is supported by essentially every GPU still in service
        # (a hard requirement since GL 3.0 / D3D10), where 8192+ is not
        # guaranteed on older integrated hardware. Judgement on the exact
        # number, informed by that compatibility floor.
        texture_px_warn=4096,
        texture_bytes_warn=64 * 1024 * 1024,  # one full 4096^2 RGBA texture
        size_min_m=_SIZE_MIN_M,
        size_max_m=_SIZE_MAX_DESKTOP_M,
        # Judgement, an order below this profile's own render triangle
        # budget -- see ``Profile.collider_triangles_warn``'s own comment.
        collider_triangles_warn=4_000,
        # Godot has no documented convex-hull vertex ceiling this session
        # could find (see ``Profile.collider_convex_vertex_cap``'s comment).
        collider_convex_vertex_cap=None,
    ),
    "godot-mobile": Profile(
        key="godot-mobile",
        label="Godot (Mobile)",
        # Judgement, same reasoning as godot-desktop's triangle budget, at
        # the "rough guide" mobile end: 5k-10k warn / 30k-50k fail is the
        # range a mobile-targeted real-time asset conversation keeps landing
        # on; 8k/40k sits in the middle of it.
        triangles_warn=8_000,
        triangles_fail=40_000,
        vertices_warn=8_000,
        materials_warn=4,
        # 2048 is the conservative, widely-documented mobile-safe texture
        # ceiling -- a meaningful slice of Android/iOS hardware still in the
        # field tops out there or below, where 4096 is not guaranteed.
        # Judgement on the exact number, informed by that compatibility gap.
        texture_px_warn=2048,
        texture_bytes_warn=16 * 1024 * 1024,  # one full 2048^2 RGBA texture
        size_min_m=_SIZE_MIN_M,
        size_max_m=_SIZE_MAX_MOBILE_M,
        # Tighter than the desktop profile's, at the mobile end of the same
        # judgement.
        collider_triangles_warn=1_000,
        collider_convex_vertex_cap=None,  # same reason as godot-desktop's.
    ),
    "unity": Profile(
        key="unity",
        label="Unity",
        # Judgement, desktop-comparable budget -- Unity ships on the same
        # hardware range godot-desktop targets.
        triangles_warn=50_000,
        triangles_fail=250_000,
        # Not pure judgement: Unity's default ``Mesh.indexFormat`` is
        # ``UInt16``, which caps a single mesh at 65,535 vertices before the
        # importer (or a script) has to opt into ``UInt32`` -- a real,
        # documented ceiling (Unity Manual, "Mesh.indexFormat"), not a style
        # preference. 60,000 is set as *warn* headroom below that hard
        # ceiling -- a mesh that has already crossed it is a mesh Unity's
        # default import already had to make a decision about.
        vertices_warn=60_000,
        materials_warn=8,  # judgement
        texture_px_warn=4096,  # judgement, same compatibility floor as Godot
        texture_bytes_warn=64 * 1024 * 1024,
        size_min_m=_SIZE_MIN_M,
        size_max_m=_SIZE_MAX_DESKTOP_M,
        collider_triangles_warn=4_000,
        # Unity's default physics backend is PhysX -- see
        # ``Profile.collider_convex_vertex_cap``'s own comment for the
        # confidence on 255.
        collider_convex_vertex_cap=255,
    ),
    "unreal": Profile(
        key="unreal",
        label="Unreal",
        # Judgement. Unreal's Nanite virtualized geometry removes the
        # traditional per-mesh triangle ceiling for a Nanite-enabled asset,
        # but a glTF import through Realmspinner's pipeline is an ordinary static
        # mesh unless the project opts a source asset into Nanite after
        # import -- this profile has no way to know that happened, so it
        # budgets for the traditional (non-Nanite) case, at the same
        # desktop-class numbers as Unity/Godot-Desktop.
        triangles_warn=50_000,
        triangles_fail=300_000,
        vertices_warn=50_000,
        materials_warn=8,
        texture_px_warn=4096,
        texture_bytes_warn=64 * 1024 * 1024,
        size_min_m=_SIZE_MIN_M,
        size_max_m=_SIZE_MAX_DESKTOP_M,
        collider_triangles_warn=4_000,
        # Unreal's own physics backend has historically been PhysX too --
        # ``colliders.py``'s own module docstring names Unity and Unreal as
        # "the physics backend Unity and Unreal both ship" in one breath.
        collider_convex_vertex_cap=255,
    ),
    "webgl": Profile(
        key="webgl",
        label="WebGL",
        # Judgement, at the mobile end of the "rough guide" range -- a WebGL
        # canvas runs on whatever GPU the visitor's browser has, which in
        # practice is the same wide low end a mobile target has to budget
        # for, plus the extra draw-call overhead of a browser's own GL/WebGPU
        # translation layer.
        triangles_warn=7_000,
        triangles_fail=35_000,
        # Not pure judgement: WebGL 1.0's ``drawElements`` defaults to
        # ``UNSIGNED_SHORT`` indices, a 65,535-vertex ceiling per draw call
        # unless the ``OES_element_index_uint`` extension is both present and
        # explicitly used (WebGL 1.0 spec, section 5.14.10; the extension is
        # core in WebGL 2 but not guaranteed on every WebGL-1-only device
        # this profile has to assume). 60,000 is warn headroom below that
        # ceiling, the same margin ``unity`` keeps below its own.
        vertices_warn=60_000,
        materials_warn=4,  # judgement, mobile-comparable
        # 2048, same mobile compatibility floor godot-mobile uses -- a WebGL
        # canvas inherits the visiting device's own GPU limits.
        texture_px_warn=2048,
        texture_bytes_warn=16 * 1024 * 1024,
        size_min_m=_SIZE_MIN_M,
        size_max_m=_SIZE_MAX_MOBILE_M,
        collider_triangles_warn=1_000,
        # Neither three.js nor Babylon.js ships a first-party physics
        # backend at all -- whichever a project bolts on (ammo.js, Rapier,
        # cannon-es...) sets its own ceiling this profile cannot know.
        collider_convex_vertex_cap=None,
    ),
}

DEFAULT_PROFILE = "godot-desktop"


@dataclass(frozen=True)
class Check:
    """One row of a :class:`Report`. ``fix`` names a Clay op (or ``""``),
    never advice in prose -- see :data:`FIX_OPS`."""

    key: str
    label: str
    status: str  # "pass" | "warn" | "fail" | "skip"
    message: str
    measured: float | int | None
    limit: float | int | None
    fix: str
    uids: tuple[int, ...]  # objects responsible; empty when document-wide


@dataclass(frozen=True)
class Report:
    """The whole answer for one document against one profile."""

    profile: str
    checks: tuple[Check, ...]  # one per CHECKS entry, in CHECKS order
    status: str  # worst of the checks: fail > warn > pass (skip ignored)


def _skip(key: str, message: str, *, limit: float | int | None = None) -> Check:
    return Check(
        key=key, label=_LABELS[key], status="skip", message=message,
        measured=None, limit=limit, fix="", uids=(),
    )


def _worst(checks: list[Check]) -> str:
    order = {"pass": 0, "warn": 1, "fail": 2}
    worst = "pass"
    for check in checks:
        if check.status == "skip":
            continue
        if order[check.status] > order[worst]:
            worst = check.status
    return worst


# --- per-check helpers -------------------------------------------------------


def _tri_count(mesh: Any) -> int:
    """Triangles a face-corner mesh renders as: ``corners - 2`` per face,
    summed -- the same O(1)-per-object arithmetic :func:`~.analyze.analyze`
    already uses to answer this *before* triangulating (its own ``total_tris``
    comment states the identity: an n-cornered face always triangulates to
    ``n - 2`` triangles)."""
    return len(mesh.loops) - 2 * bm.face_count(mesh)


def _over_share_uids(counts: dict[int, int]) -> tuple[int, ...]:
    """Objects whose own count exceeds an equal share of the total, sorted
    by count descending -- the "who is actually responsible" reading a flat
    "everyone over the mean" cutoff gives, so a 40-object scene where one
    prop is 90% of the triangle budget names that one prop rather than
    everything above a meaningless per-object average."""
    total = sum(counts.values())
    if total <= 0 or not counts:
        return ()
    share = total / len(counts)
    over = [uid for uid, count in counts.items() if count > share]
    over.sort(key=lambda uid: -counts[uid])
    return tuple(over)


def _used_material_indices(objects: list[Any]) -> set[int]:
    used: set[int] = set()
    for obj in objects:
        if len(obj.mesh.material):
            used.update(int(i) for i in np.unique(obj.mesh.material))
    return used


def _has_texture(material: Any) -> bool:
    return any(
        getattr(material, slot) is not None
        for slot in ("base_color", "metallic_roughness", "normal", "emissive", "occlusion")
    )


def _evaluated_world(obj: Any, doc: Any) -> Any:
    """*obj*, its mesh swapped for the evaluated one and its
    translation/rotation/scale swapped for its **world** TRS.

    Duck-typed, on purpose -- see :func:`validate`'s own docstring and
    :meth:`~.document.ClayDoc.world_matrix`'s. A parentless object
    (``getattr(obj, "parent", None) is None``, true for every object in a
    document with no parenting) skips the decompose round trip entirely
    rather than composing and immediately decomposing a matrix back to
    numbers that were already exact.
    """
    mesh = doc.evaluated(obj.uid)
    if getattr(obj, "parent", None) is None:
        return replace(obj, mesh=mesh)
    t, r, s = m3.decompose(doc.world_matrix(obj.uid))
    return replace(obj, mesh=mesh, translation=t, rotation=r, scale=s)


def _world_bounds(objects: list[Any]) -> tuple[np.ndarray, np.ndarray] | None:
    """The document's own world-space box -- every visible object's every
    vertex, via :func:`~.ops.world_positions` (the exact, not the
    conservative-under-rotation, transform: see that function's own
    docstring and :mod:`.analyze`'s for why this module wants the same
    answer that one does rather than :func:`~.ops.world_box`'s eight-corner
    upper bound)."""
    lo: np.ndarray | None = None
    hi: np.ndarray | None = None
    for obj in objects:
        pos = ops.world_positions(obj)
        if len(pos) == 0:
            continue
        obj_lo, obj_hi = pos.min(axis=0), pos.max(axis=0)
        lo = obj_lo if lo is None else np.minimum(lo, obj_lo)
        hi = obj_hi if hi is None else np.maximum(hi, obj_hi)
    return (lo, hi) if lo is not None else None


# --- the checks ---------------------------------------------------------------


def _check_objects(objects: list[Any]) -> Check:
    n = len(objects)
    if n == 0:
        return Check(
            key="objects", label=_LABELS["objects"], status="fail",
            message="No visible objects to check.", measured=0, limit=1, fix="", uids=(),
        )
    return Check(
        key="objects", label=_LABELS["objects"], status="pass",
        message=f"{n} visible object{'s' if n != 1 else ''}.",
        measured=n, limit=None, fix="", uids=(),
    )


def _check_triangles(objects: list[Any], profile: Profile) -> Check:
    counts = {obj.uid: _tri_count(obj.mesh) for obj in objects}
    total = sum(counts.values())
    if total > profile.triangles_fail:
        status, limit = "fail", profile.triangles_fail
        message = f"{total:,} triangles; this profile's hard limit is {limit:,}."
    elif total > profile.triangles_warn:
        status, limit = "warn", profile.triangles_warn
        message = f"{total:,} triangles; this profile recommends under {limit:,}."
    else:
        status, limit = "pass", None
        message = f"{total:,} triangles, within this profile's {profile.triangles_warn:,} budget."
    uids = _over_share_uids(counts) if status != "pass" else ()
    return Check(
        key="triangles", label=_LABELS["triangles"], status=status, message=message,
        measured=total, limit=limit, fix="decimate" if status != "pass" else "", uids=uids,
    )


def _check_vertices(objects: list[Any], profile: Profile) -> Check:
    total = 0
    lower_bound = False
    for obj in objects:
        if len(obj.mesh.loops) <= ops_clean.MAX_CLEAN_CORNERS:
            positions, _normals, _uvs, _indices = bm.render_arrays(obj.mesh)
            total += len(positions)
        else:
            # Past the ceiling ``render_arrays`` would triangulate and
            # compute normals over -- the raw position count instead, which
            # is always <= the render vertex count (a flat face splits a
            # vertex per corner; render never merges more than the source
            # positions already are), so this is an honest lower bound.
            total += len(obj.mesh.positions)
            lower_bound = True
    suffix = (
        " (lower bound: an oversized object could not be fully triangulated)"
        if lower_bound else ""
    )
    limit = profile.vertices_warn
    if total > limit:
        status = "warn"
        message = f"{total:,} render vertices{suffix}; this profile recommends under {limit:,}."
    else:
        status = "pass"
        message = f"{total:,} render vertices{suffix}, within this profile's {limit:,} budget."
    return Check(
        key="vertices", label=_LABELS["vertices"], status=status, message=message,
        measured=total, limit=profile.vertices_warn, fix="", uids=(),
    )


def _check_materials(objects: list[Any], profile: Profile) -> Check:
    used = _used_material_indices(objects)
    n = len(used)
    limit = profile.materials_warn
    if n > limit:
        status = "warn"
        message = f"{n} distinct material slots in use; this profile recommends {limit} or fewer."
    else:
        status = "pass"
        message = f"{n} distinct material slots in use, within this profile's {limit}."
    return Check(
        key="materials", label=_LABELS["materials"], status=status, message=message,
        measured=n, limit=profile.materials_warn, fix="", uids=(),
    )


def _check_textures(used_materials: dict[int, Any], profile: Profile) -> Check:
    seen: dict[int, tuple[int, int, bytes]] = {}
    largest_edge = 0
    for material in used_materials.values():
        for slot in ("base_color", "metallic_roughness", "normal", "emissive", "occlusion"):
            tex = getattr(material, slot)
            if tex is None:
                continue
            width, height, _data = tex
            largest_edge = max(largest_edge, int(width), int(height))
            seen[id(tex)] = tex  # dedup by identity: a shared texture counts once
    if not seen:
        return _skip("textures", "No used material has a texture.")
    total_bytes = sum(len(tex[2]) for tex in seen.values())
    over_px = largest_edge > profile.texture_px_warn
    over_bytes = total_bytes > profile.texture_bytes_warn
    mb = total_bytes / (1024 * 1024)
    if over_px or over_bytes:
        status = "warn"
        message = (
            f"Largest texture edge {largest_edge}px (recommend under {profile.texture_px_warn}px); "
            f"{mb:.1f} MB of decoded texture over {len(seen)} distinct texture(s) "
            f"(recommend under {profile.texture_bytes_warn / (1024 * 1024):.0f} MB)."
        )
    else:
        status = "pass"
        message = (
            f"Largest texture edge {largest_edge}px, {mb:.1f} MB over {len(seen)} distinct "
            f"texture(s), within this profile's budget."
        )
    return Check(
        key="textures", label=_LABELS["textures"], status=status, message=message,
        measured=largest_edge, limit=profile.texture_px_warn, fix="", uids=(),
    )


def _check_uvs(objects: list[Any], used_materials: dict[int, Any]) -> Check:
    textured_indices = {i for i, m in used_materials.items() if _has_texture(m)}
    no_uv_anywhere = all(obj.mesh.uv is None for obj in objects)
    if textured_indices:
        bad = [
            obj
            for obj in objects
            if obj.mesh.uv is None
            and len(obj.mesh.material)
            and bool(np.isin(np.unique(obj.mesh.material), list(textured_indices)).any())
        ]
        if bad:
            return Check(
                key="uvs", label=_LABELS["uvs"], status="fail",
                message=(
                    f"{len(bad)} object(s) use a textured material but have no UV "
                    "coordinates -- the texture cannot appear on them in any engine."
                ),
                measured=len(bad), limit=0, fix="unwrap",
                uids=tuple(obj.uid for obj in bad),
            )
    if no_uv_anywhere:
        return Check(
            key="uvs", label=_LABELS["uvs"], status="warn",
            message=(
                "No object in this document has UV coordinates; it can be lit "
                "but not textured."
            ),
            measured=0, limit=None, fix="unwrap", uids=(),
        )
    return Check(
        key="uvs", label=_LABELS["uvs"], status="pass",
        message="UV coverage is fine for the materials in use.",
        measured=None, limit=None, fix="", uids=(),
    )


def _check_geometry(objects: list[Any], surveys: dict[int, Any]) -> Check:
    bad: dict[int, int] = {}
    for obj in objects:
        s = surveys[obj.uid]
        total = s.degenerate_faces + s.duplicate_faces + s.loose_vertices + s.coincident_vertices
        if total:
            bad[obj.uid] = total
    if not bad:
        return Check(
            key="geometry", label=_LABELS["geometry"], status="pass",
            message="No degenerate, duplicate, loose or coincident geometry.",
            measured=0, limit=0, fix="", uids=(),
        )
    total = sum(bad.values())
    uids = tuple(sorted(bad, key=lambda uid: -bad[uid]))
    return Check(
        key="geometry", label=_LABELS["geometry"], status="warn",
        message=(
            f"{total} geometry defect(s) (degenerate, duplicate, loose or coincident) "
            f"across {len(bad)} object(s)."
        ),
        measured=total, limit=0, fix="clean-mesh", uids=uids,
    )


def _check_normals(objects: list[Any], surveys: dict[int, Any]) -> Check:
    bad: dict[int, int] = {}
    for obj in objects:
        s = surveys[obj.uid]
        total = s.flipped_faces + s.inside_out_shells
        if total:
            bad[obj.uid] = total
    if not bad:
        return Check(
            key="normals", label=_LABELS["normals"], status="pass",
            message="Every shell is consistently and outward wound.",
            measured=0, limit=0, fix="", uids=(),
        )
    total = sum(bad.values())
    uids = tuple(sorted(bad, key=lambda uid: -bad[uid]))
    return Check(
        key="normals", label=_LABELS["normals"], status="warn",
        message=f"{total} flipped face(s) or inside-out shell(s) across {len(bad)} object(s).",
        measured=total, limit=0, fix="recalc-normals", uids=uids,
    )


def _check_closed(objects: list[Any], surveys: dict[int, Any]) -> Check:
    bad: dict[int, int] = {}
    open_total = 0
    nonmanifold_total = 0
    for obj in objects:
        is_open = obj.generator in primitives.OPEN_GENERATORS if obj.generator else False
        open_edges = 0 if is_open else surveys[obj.uid].open_edges
        nonmanifold = len(check_manifold(obj.mesh).nonmanifold_edges)
        total = open_edges + nonmanifold
        if total:
            bad[obj.uid] = total
            open_total += open_edges
            nonmanifold_total += nonmanifold
    if not bad:
        return Check(
            key="closed", label=_LABELS["closed"], status="pass",
            message="Every shell is closed and manifold.",
            measured=0, limit=0, fix="", uids=(),
        )
    uids = tuple(sorted(bad, key=lambda uid: -bad[uid]))
    return Check(
        key="closed", label=_LABELS["closed"], status="warn",
        message=(
            f"{open_total} open edge(s) and {nonmanifold_total} non-manifold edge(s) "
            f"across {len(bad)} object(s)."
        ),
        measured=open_total + nonmanifold_total, limit=0, fix="clean-mesh", uids=uids,
    )


def _check_transforms(objects: list[Any]) -> Check:
    bad = []
    for obj in objects:
        scale = np.asarray(obj.scale, dtype="f8")
        negative = bool(np.any(scale < 0.0))
        mags = np.abs(scale)
        lo, hi = float(mags.min()), float(mags.max())
        nonuniform = hi > lo * 1.001 if lo > 1e-12 else hi > 1e-12
        if negative or nonuniform:
            bad.append(obj)
    if not bad:
        return Check(
            key="transforms", label=_LABELS["transforms"], status="pass",
            message="Every object has a positive, uniform scale.",
            measured=0, limit=0, fix="", uids=(),
        )
    return Check(
        key="transforms", label=_LABELS["transforms"], status="warn",
        message=(
            f"{len(bad)} object(s) have negative or non-uniform scale; engines can "
            "mishandle this on skinned or collided meshes."
        ),
        measured=len(bad), limit=0, fix="bake", uids=tuple(obj.uid for obj in bad),
    )


def _check_scale(bounds: tuple[np.ndarray, np.ndarray] | None, profile: Profile) -> Check:
    if bounds is None:
        return _skip("scale", "No geometry to measure.")
    lo, hi = bounds
    largest = float((hi - lo).max())
    if largest < profile.size_min_m:
        status = "warn"
        message = (
            f"{largest:.4g} m across its largest dimension, under this profile's "
            f"{profile.size_min_m:g} m floor. If this was modelled in metres but meant to "
            "be centimetre-scale, scale up by 100 (x0.01 read as metres)."
        )
        limit = profile.size_min_m
    elif largest > profile.size_max_m:
        status = "warn"
        message = (
            f"{largest:.4g} m across its largest dimension, over this profile's "
            f"{profile.size_max_m:g} m ceiling. If this was modelled in centimetres, "
            "scale down by 100 (x100 read as metres)."
        )
        limit = profile.size_max_m
    else:
        status = "pass"
        message = (
            f"{largest:.4g} m across its largest dimension, within "
            f"{profile.size_min_m:g}-{profile.size_max_m:g} m."
        )
        limit = None
    return Check(
        key="scale", label=_LABELS["scale"], status=status, message=message,
        measured=largest, limit=limit, fix="", uids=(),
    )


_PIVOT_TOL_M = 0.001  # 1 mm


def _check_pivot(bounds: tuple[np.ndarray, np.ndarray] | None, objects: list[Any]) -> Check:
    if bounds is None:
        return _skip("pivot", "No geometry to measure.")
    lo, _hi = bounds
    min_y = float(lo[1])
    if abs(min_y) > _PIVOT_TOL_M:
        status = "warn"
        message = (
            f"Lowest point sits at y={min_y:.4f} m; an asset should stand on "
            "its own origin (y=0)."
        )
        fix = "drop-to-ground"
        # The 2026-09-22 audit's clay-05: this check is document-wide (`bounds`
        # is the union over every visible object), but used to report no uids
        # at all, which left the panel's "Fix" button (`bridge._run_fix`)
        # nothing to select -- it ran `drop-to-ground` on whatever was already
        # selected instead, grounding an unrelated object or, with nothing
        # selected, silently doing nothing. Naming every object this check
        # measured lets the caller select them before running the fix, the
        # same shape every other check with a fix already uses.
        uids = tuple(obj.uid for obj in objects)
    else:
        status = "pass"
        tol_mm = _PIVOT_TOL_M * 1000
        message = f"Lowest point sits at y={min_y:.4f} m, within {tol_mm:.0f} mm of the ground."
        fix = ""
        uids = ()
    return Check(
        key="pivot", label=_LABELS["pivot"], status=status, message=message,
        measured=min_y, limit=0.0, fix=fix, uids=uids,
    )


# --- collider rows (tranche 7: export profiles) -----------------------------
#
# ``objects`` above has already dropped every ``role == "collider"`` object
# before any of the checks above ever run (see this module's own opening
# comment in :func:`validate`, and its "readiness triangle budget" mention in
# the module docstring) -- a collider is not part of the *rendered* asset, so
# counting its triangles against a render budget would be the wrong question.
# These three ask the *right* question of exactly the objects the checks
# above skip: not "is this too heavy to render" but "is this a sane thing to
# hand an engine's physics importer".


def _check_collider_present(colliders: list[Any]) -> Check:
    """Advisory, never a fail (the brief's own words) -- an asset with no
    collision proxy at all is not wrong the way an empty document is; it is
    only worth a heads-up, the same "soft but real" reading the ``uvs``
    check's own "no UV anywhere" branch gets (a bare Clay primitive ships
    with none either, and that is a ``warn``, not a ``fail``)."""
    if not colliders:
        return Check(
            key="collider_present", label=_LABELS["collider_present"], status="warn",
            message=(
                "No object in this document has a collider; it will not "
                "collide with anything on import."
            ),
            measured=0, limit=None, fix="", uids=(),
        )
    n = len(colliders)
    return Check(
        key="collider_present", label=_LABELS["collider_present"], status="pass",
        message=f"{n} collider object{'s' if n != 1 else ''}.",
        measured=n, limit=None, fix="", uids=(),
    )


def _check_collider_triangles(
    colliders: list[Any], meshes: dict[int, Any], profile: Profile
) -> Check:
    """Every collider's own triangle count against :attr:`Profile.
    collider_triangles_warn` -- the same "n - 2 per face, summed" arithmetic
    :func:`_tri_count` already gives the render ``triangles`` check, asked of
    the collision meshes instead."""
    if not colliders:
        return _skip("collider_triangles", "No collider to measure.")
    counts = {obj.uid: _tri_count(meshes[obj.uid]) for obj in colliders}
    limit = profile.collider_triangles_warn
    bad = {uid: n for uid, n in counts.items() if n > limit}
    if not bad:
        worst = max(counts.values())
        return Check(
            key="collider_triangles", label=_LABELS["collider_triangles"], status="pass",
            message=(
                f"Every collider is under this profile's {limit:,} triangle budget "
                f"(heaviest: {worst:,})."
            ),
            measured=worst, limit=limit, fix="", uids=(),
        )
    uids = tuple(sorted(bad, key=lambda uid: -bad[uid]))
    return Check(
        key="collider_triangles", label=_LABELS["collider_triangles"], status="warn",
        message=(
            f"{len(bad)} collider(s) exceed this profile's {limit:,} triangle budget "
            f"(heaviest: {bad[uids[0]]:,}) -- a collision proxy this dense defeats the "
            "point of having a separate, cheap-to-simulate shape."
        ),
        measured=bad[uids[0]], limit=limit, fix="", uids=uids,
    )


def _check_collider_convex_cap(
    colliders: list[Any], meshes: dict[int, Any], profile: Profile
) -> Check:
    """Whether a **convex or compound** collider's own vertex count sits
    under :attr:`Profile.collider_convex_vertex_cap`.

    Only those two kinds -- box/sphere/capsule get handed to an engine as an
    *analytic* primitive collider (:mod:`.colliders`' own module docstring:
    ``Collider.params`` is "what an exporter should hand to an engine with a
    native primitive collider type"), so no mesh vertex cap ever applies to
    them; only ``convex``/``compound`` become an actual triangle mesh a
    physics importer has to accept. A ``compound`` collider's stored mesh is
    every part's hull concatenated into one (:func:`~.colliders.compound`'s
    own docstring), so this counts the *whole* concatenation rather than each
    part individually -- a conservative reading (a compound of several
    small hulls can cross a per-*part* cap's sum without any single part
    crossing it), stated here rather than claimed away, because ``Obj`` never
    keeps the individual ``Collider.params["parts"]`` breakdown that fit it.
    """
    convex_kinds = {"convex", "compound"}
    relevant = [obj for obj in colliders if obj.collider_kind in convex_kinds]
    if not relevant:
        return _skip("collider_convex", "No convex-hull collider to measure.")
    cap = profile.collider_convex_vertex_cap
    if cap is None:
        return _skip(
            "collider_convex",
            f"{profile.label} has no documented convex-hull vertex ceiling to check against.",
        )
    counts = {obj.uid: len(meshes[obj.uid].positions) for obj in relevant}
    bad = {uid: n for uid, n in counts.items() if n > cap}
    if not bad:
        worst = max(counts.values())
        return Check(
            key="collider_convex", label=_LABELS["collider_convex"], status="pass",
            message=(
                f"Every convex-hull collider is under this profile's {cap:,}-vertex "
                f"cap (heaviest: {worst:,})."
            ),
            measured=worst, limit=cap, fix="", uids=(),
        )
    uids = tuple(sorted(bad, key=lambda uid: -bad[uid]))
    return Check(
        key="collider_convex", label=_LABELS["collider_convex"], status="warn",
        message=(
            f"{len(bad)} convex-hull collider(s) exceed this profile's {cap:,}-vertex "
            f"cap (heaviest: {bad[uids[0]]:,})."
        ),
        measured=bad[uids[0]], limit=cap, fix="", uids=uids,
    )


# --- entry point --------------------------------------------------------------


def validate(doc: Any, profile: str = DEFAULT_PROFILE, *, visible_only: bool = True) -> Report:
    """Every :data:`CHECKS` row for *doc* against *profile*, in order.

    *doc* is duck-typed on ``.objects`` (each duck-typed on ``uid``, ``mesh``,
    ``translation``, ``rotation``, ``scale``, ``generator``, ``material`` and
    ``visible`` -- :class:`~.document.Obj`'s own shape) and ``.materials``
    (each duck-typed on the five texture slots and the factor fields --
    :class:`~.geom3d.gltf.Material`'s), the same "read a document without
    importing one" contract :mod:`.analyze` and :mod:`.diagnose` state for
    their own entry points. Nothing here mutates *doc*; every ceiling below
    is a ``"skip"`` row naming why, never a refusal, except
    :data:`MAX_VALIDATE_OBJECTS` -- raised as an :class:`~.elements.OpError`
    before any per-object work runs, the same "known cheaply, refused before
    it is paid for" shape :func:`~.analyze.analyze` uses for its own object
    and triangle ceilings.

    **Every check below measures the evaluated mesh**, not the base --
    ``doc.evaluated(obj.uid)`` is swapped in for ``obj.mesh`` once, here,
    before any check runs, the same "measurement reads evaluated" rule
    :mod:`.modifiers` states in its own module docstring: an engine that
    imports this document sees the modifier stack's own output, not what the
    base mesh alone would have been, so that is what a readiness verdict has
    to be about. An object with no enabled modifiers evaluates to its own
    ``obj.mesh`` unchanged (:mod:`.modifiers`' fast path), so a document with
    none in it is checked exactly as it always was.

    **And every check measures world placement, not local** (tranche 3:
    scene structure): each object's translation/rotation/scale are likewise
    swapped for its world TRS (``doc.world_matrix(obj.uid)``, decomposed)
    before any check runs -- ``scale``/``pivot``/``transforms`` care about
    where an object actually sits once an engine imports it, not where it
    sits relative to a parent this format has no opinion about. A
    parentless object's world TRS *is* its own TRS, so a document with no
    parenting is checked exactly as it always was.
    """
    prof = PROFILES[profile]
    # A collider is not part of the rendered asset: an engine imports it as a
    # physics shape, and counting its triangles against a *rendering* budget
    # would tell a user their crate is too heavy because they gave it a box to
    # collide with. Read through ``getattr`` rather than importing
    # ``document.Obj`` -- this module is checked against duck-typed objects in
    # its own tests, and ``_evaluated_world`` already reads ``parent`` the
    # same way.
    keep = [obj for obj in doc.objects if getattr(obj, "role", "mesh") != "collider"]
    visible = [obj for obj in keep if obj.visible] if visible_only else keep

    # The 2026-09-20 audit's clay-05: checked here, against the raw count,
    # before a single ``_evaluated_world`` call runs -- every check below
    # walks every object at least once (a ``survey`` call apiece for three of
    # them), so a document already past this ceiling would otherwise pay for
    # that walk before ever being told no, the same "refuse before the cost"
    # shape :func:`~.analyze.analyze` uses for its own object and triangle
    # ceilings. See :data:`MAX_VALIDATE_OBJECTS` for the measurements behind
    # the number and why this is the one refusal in an otherwise
    # measure-only module.
    if len(visible) > MAX_VALIDATE_OBJECTS:
        raise OpError(
            f"This document has {len(visible)} visible objects, past the "
            f"{MAX_VALIDATE_OBJECTS:,} Game check works with at once. Hide "
            "or delete some before checking readiness."
        )

    objects = [_evaluated_world(obj, doc) for obj in visible]

    # The collider objects the pass above dropped, read back for the three
    # ``collider_*`` rows below -- the mirror-image filter, so a hidden
    # collider is excluded from a ``visible_only`` report the same way a
    # hidden mesh object already is. ``doc.evaluated`` rather than
    # ``obj.mesh`` for the same reason every check above reads evaluated
    # meshes (a collider carries no modifiers today, but the fast path in
    # :mod:`.modifiers` makes this free when it does not); world placement is
    # not needed here, since triangle and vertex *counts* do not depend on
    # where an object sits.
    collider_src = [obj for obj in doc.objects if getattr(obj, "role", "mesh") == "collider"]
    collider_objs = [obj for obj in collider_src if obj.visible] if visible_only else collider_src
    collider_meshes = {obj.uid: doc.evaluated(obj.uid) for obj in collider_objs}

    if not objects:
        checks = [_check_objects(objects)] + [
            _skip(key, "Skipped: no visible objects.") for key in CHECKS[1:]
        ]
        return Report(profile=prof.key, checks=tuple(checks), status=_worst(checks))

    # The 2026-09-22 audit's clay-13: `MAX_VALIDATE_OBJECTS` bounds how many
    # objects one call can face, but not how big each one is -- an object
    # just under `ops_clean.MAX_CLEAN_CORNERS` pays the full `survey` cost
    # below (geometry/normals/closed) and the ceiling above lets 1,000 of
    # them through at once. Measured on this machine (`godot-desktop`,
    # objects each just under the corner ceiling): ~465 ms/object, 9.3 s at
    # 20 objects, ~8 min extrapolated to 1,000 -- the same shape as the
    # 2026-09-20 audit's clay-05 measurement table, one axis over (total
    # corners rather than object count). Summed here, before the survey loop
    # runs, the same "known cheaply, refused before it is paid for" shape as
    # `MAX_VALIDATE_OBJECTS` above -- `len(obj.mesh.loops)` is already read
    # for free by the `oversized` computation just below.
    total_corners = sum(len(obj.mesh.loops) for obj in objects)
    if total_corners > MAX_VALIDATE_CORNERS:
        raise OpError(
            f"This document has {total_corners:,} corners across its visible "
            f"objects, past the {MAX_VALIDATE_CORNERS:,} Game check works "
            "with at once. Hide or delete some before checking readiness."
        )

    used_material_indices = _used_material_indices(objects)
    used_materials = {
        i: doc.materials[i] for i in used_material_indices if 0 <= i < len(doc.materials)
    }
    bounds = _world_bounds(objects)

    # ``geometry``/``normals``/``closed`` share this one skip decision and,
    # when it does not fire, share one ``survey`` call per object -- see the
    # module docstring's cost section.
    oversized = tuple(
        obj.uid for obj in objects if len(obj.mesh.loops) > ops_clean.MAX_CLEAN_CORNERS
    )
    if oversized:
        ceiling_msg = (
            f"{len(oversized)} object(s) exceed the {ops_clean.MAX_CLEAN_CORNERS:,} corner "
            "ceiling Clean can process without stalling; skipped rather than stall the frame."
        )
        geometry = Check(
            key="geometry", label=_LABELS["geometry"], status="skip", message=ceiling_msg,
            measured=None, limit=ops_clean.MAX_CLEAN_CORNERS, fix="", uids=oversized,
        )
        normals = Check(
            key="normals", label=_LABELS["normals"], status="skip", message=ceiling_msg,
            measured=None, limit=ops_clean.MAX_CLEAN_CORNERS, fix="", uids=oversized,
        )
        closed = Check(
            key="closed", label=_LABELS["closed"], status="skip", message=ceiling_msg,
            measured=None, limit=ops_clean.MAX_CLEAN_CORNERS, fix="", uids=oversized,
        )
    else:
        surveys = {obj.uid: ops_clean.survey(obj.mesh) for obj in objects}
        geometry = _check_geometry(objects, surveys)
        normals = _check_normals(objects, surveys)
        closed = _check_closed(objects, surveys)

    checks = [
        _check_objects(objects),
        _check_triangles(objects, prof),
        _check_vertices(objects, prof),
        _check_materials(objects, prof),
        _check_textures(used_materials, prof),
        _check_uvs(objects, used_materials),
        geometry,
        normals,
        closed,
        _check_transforms(objects),
        _check_scale(bounds, prof),
        _check_pivot(bounds, objects),
        _check_collider_present(collider_objs),
        _check_collider_triangles(collider_objs, collider_meshes, prof),
        _check_collider_convex_cap(collider_objs, collider_meshes, prof),
    ]
    return Report(profile=prof.key, checks=tuple(checks), status=_worst(checks))
