"""The fifteen shapes a user can place, and the registry the panel is built from.

Each generator is a plain function of its parameters returning a :class:`Mesh`,
and :data:`GENERATORS` maps a name to ``(defaults, builder)``. The registry is
the point of the module rather than an index over it: the properties panel is
generated from those default dictionaries, so adding a sixteenth primitive is
adding a function and one registry line, in the same spirit as "add a skeleton
by adding a JSON file, never by hardcoding bones in ``blender_worker``". A
panel that switched on a hardcoded list of shape names would be a second place
that has to know what a cylinder's parameters are, and the two would drift the
first time a parameter was renamed.

Four rules hold across all fifteen, and each of them is pinned by a test:

**Every primitive is built centred on the origin.** ``Obj`` carries the
translation, so geometry that baked its placement in would make the numeric TRS
panel lie -- a box "at the origin" would sit somewhere else, and moving it back
would leave the panel reading a position the object is not at. ``plane`` is
centred too: it lies *in* the XZ plane at y = 0, not on top of it. Every other
parameter here is an extent -- a size, a radius, a height -- and an extent
cannot express a position, so this holds for free. The cases that take actual
work are the three parameters that carry positions rather than extents --
``lathe``'s ``profile``, ``sweep``'s ``outline`` and ``tube``'s ``path`` -- and
each is re-centred by its own normaliser (:func:`_clamp_profile`,
:func:`_clamp_outline`, :func:`_clamp_path`) before a single vertex is placed,
because the coordinates in all three describe a shape rather than a place.
Re-centring in the *normaliser* and never in the generator body is what keeps
the stored parameters and the built geometry describing the same object; see
:func:`clamp_params`. Two consequences are worth stating rather than
discovering: a *measured* bounding box can still be off-centre where the
geometry is not -- a coarse ring's is, ``sweep``'s ``twist`` makes one, and a
``tube`` around a path with no point symmetry has one by construction -- and
none of that is corrected for, because it is a fact about the shape rather
than about its placement.

**Caps are n-gons, not fans.** The CSR storage exists precisely so a cylinder's
lid can be one face with sixteen corners, and it matters twice over: a fan cap
is sixteen faces where one will do, which the triangle budget notices, and a
user who clicks a cap in face mode expects to select *the cap*, not one wedge
of it. It is also why a cylinder at sixteen segments has thirty-two
vertices and not thirty-four -- there is no cap-centre vertex, because there is
no fan to radiate from one.

**Every face is wound counter-clockwise seen from outside, and convex.** The
winding is what makes the Newell normal point outward, and a flipped cap is the
single nastiest defect this module can ship: the viewport draws with back-face
culling off, so it looks perfect, and the exported GLB is inside out in every
engine with nothing in the file to explain why. Convexity is what makes
``mesh.triangulate``'s fan correct -- it fans from each face's first corner and
a fan across a reflex corner puts a triangle outside the polygon. **One shape's
cap cannot keep the second half of that claim**, and says so as registry data
rather than by weakening it: ``sweep``'s cap is an arbitrary user-supplied
outline, an L-bracket by default, and a reflex corner is exactly what "sweep"
is for. See :data:`CONCAVE_GENERATORS` for the exemption and
``clay/earclip.py``'s own docstring for why the fan-then-ear-clip triangulator
that already existed for dissolve results and imports is what makes the
exemption safe rather than merely tolerated.

That first claim takes **two** assertions, not one, and the obvious one is the
weaker of the pair. Summing ``(centroid - centre) . normal`` over the faces is
six times the enclosed volume, so it says the shell is oriented *outward* --
but reversing a single face changes it by only twice that face's own
contribution, which is small beside the whole volume. Measured on these
generators, flipping one face leaves the sum positive every time, and flipping
the cone's base leaves it at ``7e-18``: positive by float noise. So the tests
also assert that **no ordered corner pair ``(a, b)`` occurs twice across all
faces**, which is what "consistently oriented" means -- neighbouring faces
traverse their shared edge in opposite directions, so a directed edge is used
once and a flipped face immediately collides with its neighbour. The undirected
edge-use count cannot stand in for it: it keys on the sorted pair and is
orientation-blind by construction. Consistency plus a positive volume is the
whole claim; either alone is not.

**Sizes are taken as magnitudes.** Every extent -- a box's ``size``, a radius, a
height, the tube -- is passed through ``abs``, because a negative one does not
mean a mirrored primitive, it means an inside-out one: a negative height puts a
cylinder's bottom ring above its top and every side quad and both caps then
face inward, and ``validate`` would accept it happily because it checks CSR
structure and index ranges and nothing geometric. A numeric property field is
one keystroke away from a minus sign, and mirroring is ``mesh.transformed``'s
job -- it reverses the loops to keep the winding honest, which a generator
handed a negative number cannot do on the caller's behalf.

Every face comes back **flat-shaded on material zero**, and every generator
here always will: that is what makes ``clamp_params``' rebuild a pure function
of a generator's own parameters, with nothing about *placement* smuggled into
the shape itself. It used to also be the reason nothing here looked smooth --
there was no shading tool to hand a curved primitive over to, so faceted
geometry was the only honest default available. That reason expired the day
Shade Smooth, Shade Flat and auto-smooth-by-angle shipped (``clay_ops.py``),
and on 2026-09-06 the user decided what replaces it: **organic shapes insert
smooth-shaded.** The decision belongs at *insertion*, not here -- a generator
is called every time a parameter field is edited, and a generator that decided
its own shading would be deciding it again on every keystroke, silently
overwriting a Shade Flat the user had just clicked. So the two doors an object
is placed through, ``panes/clay_tools.add_primitive`` and
``panes/clay_tools.add_assembly``, apply ``clay.shading.auto_smooth`` to what a
generator hands back, and this module keeps handing back the same flat mesh it
always did -- a box "at rest" is one description regardless of where it ends
up, and what a viewport shows of it is a fact about the door it walked through,
not about the box. Axes are glTF's -- Y up, right-handed -- because that is the
space the viewer, the exporter and everything downstream of Clay already
speak.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .mesh import Mesh
from .mesh import from_faces as _mesh

# The low-end clamps. A slider dragged to zero must not be able to produce a
# mesh ``validate`` rejects, and clamping is the right shape for a control that
# is being *dragged*: raising mid-drag would have to be caught by the panel,
# which has nothing sensible to show for it. Three is the smallest ring that is
# a polygon at all; two is the smallest number of latitude bands that leaves a
# sphere with any surface between its poles.
MIN_SEGMENTS = 3
MIN_RINGS = 2

# The high-end clamps the 2026-09-11 audit added (finding clay-01): every one
# of these counts had a floor and no ceiling, so ``clamp_params("cylinder",
# {"segments": 50_000_000})`` was accepted unchanged and the generator itself
# had not returned after 30 s -- the pygame frame thread stalls inside
# ``panes/clay_props.py``'s draw-time rebuild, and the agent surface's
# ``_h_add_primitive``/``_h_set_params`` pass a number through with no refusal
# in between. Each ceiling is picked so that the *pair* of counts a two-count
# generator multiplies together (``torus``'s segments*sides, ``uv_sphere`` and
# ``capsule``'s segments*rings) still lands comfortably under
# ``glbimport.MAX_TRIANGLES`` (2,000,000) once quads become triangles, leaving
# headroom for the rest of a document sharing that budget -- these are counts
# on a *single* generated primitive, not the document ceiling itself.
MAX_SEGMENTS = 512
MAX_RINGS = 256

# A grid's own floor, deliberately *not* ``MIN_SEGMENTS``. One division is a
# legitimate grid -- it is ``plane`` -- so clamping it to three would refuse a
# shape the registry already ships, which is why the parameter is called
# ``divisions`` rather than ``segments``: the shared name would drag the shared
# clamp along with it.
MIN_DIVISIONS = 1

# ``grid``'s own ceiling, added alongside :data:`MAX_SEGMENTS` for the same
# 2026-09-11 audit finding (clay-01): a grid is ``divisions * divisions`` quads,
# so 512 divisions is 262,144 quads -- half a million triangles once
# triangulated, the same order of headroom :data:`MAX_SEGMENTS` leaves a
# two-count generator under ``glbimport.MAX_TRIANGLES``.
MAX_DIVISIONS = 512

# How far an icosphere may be subdivided from the properties panel. Each step
# quadruples the face count (20 -> 80 -> 320 -> 1280 -> 5120 -> 20480), and the
# undo stack holds two meshes per step, so an unbounded integer field is one
# keystroke away from a multi-second stall on a control that is being *typed*.
MAX_SUBDIVISIONS = 5

# The floor a *middle* profile station's radius is raised to -- see
# ``_clamp_profile``. Only the two ends of a lathe profile may be poles; a
# zero radius partway along pinches the surface to a single non-manifold
# point with no modelling meaning, which is a different failure from a
# negative extent and needs its own floor rather than an ``abs()``. Its job
# is to be *positive*, not to be a minimum anybody would model to -- the same
# reading ``arch``'s own ``d = max(abs(float(depth)), 1e-4) * 0.5`` gives its
# floor -- so it is small enough that no shape at any working scale can see
# it: a pinch point is what it exists to prevent, and a thin waist (the
# default goblet's own stem is radius 0.05) is a legitimate shape this floor
# must not visibly widen.
MIN_PROFILE_RADIUS = 1e-4

# The floor ``sweep``'s own ``taper`` is raised to. A taper of exactly zero
# collapses the far cap to a single point and every side quad touching it to
# zero area -- the same "no modelling meaning" failure :data:`MIN_PROFILE_RADIUS`
# exists to prevent for a lathe's middle stations, and the same reading: its
# job is to be *positive*, not to be a minimum anybody would model to.
MIN_TAPER = 1e-4

# ``sweep``'s own floor on how many stations it stacks along Z -- one, not
# :data:`MIN_SEGMENTS`, for the reason ``grid``'s own ``MIN_DIVISIONS`` is not
# three: a sweep's ring corner count comes from its own outline, not from this
# parameter, so one section (two rings, one band) is already the smallest
# sweep there is. The parameter is named ``sections`` rather than ``segments``
# for exactly this reason -- the shared name would drag :data:`MIN_SEGMENTS`
# in with it, the same trap ``grid``'s ``divisions`` sidesteps.
MIN_SECTIONS = 1

# ``sweep``'s own ceiling, added for the same 2026-09-11 audit finding
# (clay-01) that gives :data:`MAX_SEGMENTS`. A sweep's band count is
# ``sections`` times its outline's own corner count, which this module cannot
# bound (an outline is caller-supplied), so the ceiling is chosen the same way
# :data:`MAX_DIVISIONS` is: enough sections that even a many-cornered outline
# stays well under ``glbimport.MAX_TRIANGLES`` in the common case, not a
# guarantee for an unbounded outline.
MAX_SECTIONS = 256


def _clamp_segments(value: Any) -> int:
    """The floor and ceiling every ring-and-cap generator applies to its own
    count -- see :data:`MAX_SEGMENTS` for why the ceiling was added (the
    2026-09-11 audit, finding clay-01)."""
    return min(max(int(value), MIN_SEGMENTS), MAX_SEGMENTS)


def _clamp_rings(value: Any) -> int:
    """The floor and ceiling ``uv_sphere`` and ``capsule`` apply to their
    latitude bands -- see :data:`MAX_RINGS` (the 2026-09-11 audit, finding
    clay-01)."""
    return min(max(int(value), MIN_RINGS), MAX_RINGS)


def _clamp_divisions(value: Any) -> int:
    """``grid``'s own floor and ceiling -- one, not :data:`MIN_SEGMENTS`; see
    :data:`MIN_DIVISIONS` and :data:`MAX_DIVISIONS` (the latter added by the
    2026-09-11 audit, finding clay-01)."""
    return min(max(int(value), MIN_DIVISIONS), MAX_DIVISIONS)


def _clamp_subdivisions(value: Any) -> int:
    """``icosphere``'s own floor and ceiling."""
    return min(max(int(value), 0), MAX_SUBDIVISIONS)


def _clamp_sections(value: Any) -> int:
    """``sweep``'s own floor and ceiling -- see :data:`MIN_SECTIONS` and
    :data:`MAX_SECTIONS` (the latter added by the 2026-09-11 audit, finding
    clay-01)."""
    return min(max(int(value), MIN_SECTIONS), MAX_SECTIONS)


def _clamp_profile(value: Any) -> list[list[float]]:
    """``lathe``'s own floor on an array-valued parameter, in one place so
    ``sweep``'s ``outline`` and ``tube``'s ``path`` can each register their
    own normaliser beside it in :data:`_PROFILE_CLAMPS` rather than growing a
    second copy of this function's shape.

    Six steps, in the order ``docs/INVARIANTS.md``'s generator paragraph
    states the first four of them:

    1. Coerce to ``[radius, y]`` pairs and take ``abs()`` of every radius --
       the module's "sizes are taken as magnitudes" rule. Anything that will
       not unpack this way (the wrong shape, a non-numeric value) is treated
       as no stations at all, which step 6 turns into the default profile.
    2. Clamp ``y`` **non-decreasing**, each station raised to at least its
       predecessor's. A pair that can cross inverts a band's winding through
       two perfectly positive numbers -- the same negative-extent failure a
       negative height causes, arriving past the ``abs()`` guard in step 1,
       and ``validate`` accepts every bit of it because only the geometry is
       wrong.
    3. Drop a station that now coincides with its predecessor. A zero-area
       quad passes ``validate`` and reaches the exporter; step 2 is exactly
       what can manufacture one, by raising a station's ``y`` up to meet a
       predecessor whose radius already matched.
    4. Re-centre the survivors' ``y`` about zero, shifting every station by
       the midpoint of the range steps 2-3 left behind. A profile's ``y`` is
       *relative* -- it describes the shape a silhouette traces, not where
       that silhouette sits, and where it sits is ``Obj.translation``'s job,
       exactly the division every other generator's parameters already obey
       without having to state it: an extent cannot express a position, and a
       station can. Nothing here touches ``x``/``z`` -- a revolve is centred
       on its axis by construction.
    5. Floor a **middle** station's radius to :data:`MIN_PROFILE_RADIUS`,
       leaving only the first and last stations free to be poles -- a zero
       radius in the middle pinches the surface to a single non-manifold
       point with no modelling meaning, which a lathe's two true ends do have
       (a finial, a droplet, a chess pawn).
    6. Fall back to :data:`LATHE_DEFAULT_PROFILE` when fewer than two
       stations survive, *or* when no station has a positive radius at all.
       The second half is not the ``len == 2`` case it can only actually
       arise from today (step 5 already guarantees a positive radius at
       every *middle* station, so this can only fire when a profile has no
       middle stations to floor) -- it is stated as the general fact rather
       than that special case, because a profile with no positive radius
       anywhere is a line segment, not a solid of revolution, whatever its
       station count: two zero-radius poles and nothing between them is
       ``[[0, y0], [0, y1]]``, which ``_revolve`` would otherwise fan into
       two rings of coincident points at two positions -- every face
       zero-area, ``validate`` passing regardless, the exact failure this
       paragraph exists to close, arriving through the one arrangement the
       first four steps cannot see.
    """
    try:
        stations = [[abs(float(r)), float(y)] for r, y in value]
    except (TypeError, ValueError):
        stations = []
    for i in range(1, len(stations)):
        if stations[i][1] < stations[i - 1][1]:
            stations[i][1] = stations[i - 1][1]
    deduped: list[list[float]] = []
    for station in stations:
        if deduped and deduped[-1] == station:
            continue
        deduped.append(station)
    if deduped:
        mid = (min(y for _, y in deduped) + max(y for _, y in deduped)) / 2.0
        for station in deduped:
            station[1] -= mid
    for i in range(1, len(deduped) - 1):
        if deduped[i][0] <= 0.0:
            deduped[i][0] = MIN_PROFILE_RADIUS
    if len(deduped) < 2 or all(radius <= 0.0 for radius, _ in deduped):
        return [list(station) for station in LATHE_DEFAULT_PROFILE]
    return deduped


def _signed_area(points: list[list[float]]) -> float:
    """Twice the shoelace formula's own half -- positive for a polygon
    traversed counter-clockwise in standard (x, y) axes, which is the
    orientation :func:`sweep` builds an outward shell from. Zero for fewer
    than three points, which is what lets :func:`_clamp_outline` call this
    before it has checked the count itself.
    """
    n = len(points)
    if n < 3:
        return 0.0
    total = 0.0
    for i in range(n):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n]
        total += x0 * y1 - x1 * y0
    return total * 0.5


def _clamp_outline(value: Any) -> list[list[float]]:
    """``sweep``'s own floor on its array-valued parameter, registered beside
    :func:`_clamp_profile` in :data:`_PROFILE_CLAMPS` rather than copying that
    function's shape a second time.

    Five steps:

    1. Coerce to ``[x, y]`` float pairs. Unlike a lathe's ``profile``, neither
       coordinate is an extent -- an outline's ``x`` and ``y`` are both
       positions, exactly as a profile's ``y`` is -- so neither is taken
       through ``abs()``: a corner at a negative coordinate is not a mirrored
       corner, it is a corner. Anything that will not unpack this way is
       treated as no corners at all, which step 5 turns into the default
       outline.
    2. Drop a corner coinciding with its predecessor, *and* the last against
       the first. An outline is **closed**, which a profile is not, so this
       step has a wrap-around case ``_clamp_profile`` has no reason to check:
       an authored or agent-supplied outline that repeats its first point to
       "close the loop" would otherwise leave a zero-length edge for the
       winding and UV arc-length maths below to divide by.
    3. Reverse the outline if its own signed area is negative, so the winding
       is always the one :func:`sweep` builds an outward shell from -- the
       module's "single nastiest defect" the other way round: a clockwise
       outline is not an error, it is a shape that looks perfect in the
       viewport and is inside out in every engine, with nothing in the file
       to say why.
    4. Re-centre on the bounding-box centre, for the reason step 4 of
       :func:`_clamp_profile` re-centres a profile's ``y``: ``outline``
       carries positions, and a generator that baked its own placement in
       would make the numeric TRS panel lie.
    5. Fall back to :data:`SWEEP_DEFAULT_OUTLINE` with fewer than three
       surviving corners (not a polygon at all), or when the signed area is
       zero (every corner collinear -- a polygon with no area, the same class
       of degenerate :func:`_clamp_profile`'s own last step closes for a
       profile with no positive radius anywhere).

    **Self-intersection is not clamped here, and is not the job of this
    function.** A figure-eight outline survives every one of these five steps
    -- it has three or more corners, a well-defined (non-zero) signed area,
    and reverses cleanly -- and builds a self-intersecting solid that
    ``validate`` accepts without complaint. There is no cheap general test for
    "is this polygon simple", the same admission ``torus``'s own docstring
    makes for a tube wider than its radius: the two are legitimately
    independent right up to the point where they are not, and keeping an
    outline simple is the caller's job.
    """
    try:
        corners = [[float(x), float(y)] for x, y in value]
    except (TypeError, ValueError):
        corners = []
    deduped: list[list[float]] = []
    for corner in corners:
        if deduped and deduped[-1] == corner:
            continue
        deduped.append(corner)
    if len(deduped) > 1 and deduped[0] == deduped[-1]:
        deduped.pop()
    area = _signed_area(deduped)
    if area < 0.0:
        deduped.reverse()
    if deduped:
        xs = [c[0] for c in deduped]
        ys = [c[1] for c in deduped]
        cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
        for corner in deduped:
            corner[0] -= cx
            corner[1] -= cy
    if len(deduped) < 3 or area == 0.0:
        return [list(corner) for corner in SWEEP_DEFAULT_OUTLINE]
    return deduped


def _clamp_path(value: Any) -> list[list[float]]:
    """``tube``'s own floor on its array-valued parameter, registered beside
    :func:`_clamp_profile` and :func:`_clamp_outline` in :data:`_PROFILE_CLAMPS`
    rather than growing a third copy of either function's shape.

    Four steps:

    1. Coerce to ``[x, y, z]`` float triples. No ``abs()`` on any of the three
       -- a path carries positions, not extents, the same reasoning
       :func:`_clamp_outline` step 1 gives for its own ``x``/``y``: a point at
       a negative coordinate is not a mirrored point, it is a point. Anything
       that will not unpack this way is treated as no points at all, which
       step 4 turns into the default path.
    2. Drop a point coinciding with its predecessor. A zero-length segment has
       no tangent, so :func:`_tube_frames`'s parallel-transport has nothing to
       carry forward through it -- the same zero-area-quad failure
       :func:`_clamp_profile` step 3 and :func:`_clamp_outline` step 2 each
       exist to prevent for their own array parameter, arriving here through a
       repeated station instead of a repeated corner. **No wrap-around case**:
       unlike :func:`_clamp_outline`, a path is *open* rather than closed -- a
       cable does not join its far end back to its near one -- so there is no
       "last against first" pair to check as well.
    3. Re-centre on the bounding-box centre, for the reason step 4 of
       :func:`_clamp_profile` and step 4 of :func:`_clamp_outline` each
       re-centre their own array parameter: ``path`` carries positions, and a
       generator that baked its own placement in would make the numeric TRS
       panel lie. Re-centring the *path* does not centre the *tube* built
       around it, though -- see :data:`TUBE_DEFAULT_PATH`'s own docstring for
       why the default is chosen to make that residual vanish by construction
       rather than by a second correction applied after the mesh exists.
    4. Fall back to :data:`TUBE_DEFAULT_PATH` with fewer than two surviving
       points -- a single point has no tangent and nothing for a tube to be
       the centreline of.

    **Self-intersection is not clamped here, and it is a different admission
    from the one ``torus`` makes.** A ``radius`` wider than the path's own
    tightest turn makes the tube pass through itself, and ``validate`` accepts
    it happily -- but where ``torus``'s ``tube`` and ``radius`` are two numbers
    :func:`clamp_params` genuinely clamps against each other before a mesh is
    ever built, the limit here is a property of the *whole path*, with no
    cheap general test for it, the same admission :func:`_clamp_outline`'s own
    docstring makes for a self-crossing outline. Keeping ``radius`` inside what
    the path can hold without crossing itself is the caller's business, not
    this function's.
    """
    try:
        points = [[float(x), float(y), float(z)] for x, y, z in value]
    except (TypeError, ValueError):
        points = []
    deduped: list[list[float]] = []
    for point in points:
        if deduped and deduped[-1] == point:
            continue
        deduped.append(point)
    if deduped:
        xs = [p[0] for p in deduped]
        ys = [p[1] for p in deduped]
        zs = [p[2] for p in deduped]
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        cz = (min(zs) + max(zs)) / 2.0
        for point in deduped:
            point[0] -= cx
            point[1] -= cy
            point[2] -= cz
    if len(deduped) < 2:
        return [list(point) for point in TUBE_DEFAULT_PATH]
    return deduped


# Which key names the properties panel must clamp before calling a generator,
# and how -- see clamp_params. Keyed on parameter name rather than generator,
# because each of these floors (and, since the 2026-09-11 audit's finding
# clay-01, ceilings) is the same operation wherever the name appears
# (``torus`` clamps both its ``segments`` and its ``sides`` this way), not a
# property of any one shape.
_KEY_CLAMPS: dict[str, Callable[[Any], int]] = {
    "segments": _clamp_segments,
    "sides": _clamp_segments,
    "rings": _clamp_rings,
    "divisions": _clamp_divisions,
    "subdivisions": _clamp_subdivisions,
    "sections": _clamp_sections,
}

# The array-valued counterpart to :data:`_KEY_CLAMPS`, kept as its own table
# rather than folded into it because these normalisers return a profile, not
# an int. ``tube``'s ``path`` is the third entry, keyed on name exactly as
# ``_KEY_CLAMPS`` already is.
_PROFILE_CLAMPS: dict[str, Callable[[Any], list[list[float]]]] = {
    "profile": _clamp_profile,
    "outline": _clamp_outline,
    "path": _clamp_path,
}


def clamp_params(generator: str, params: dict[str, Any]) -> dict[str, Any]:
    """The values ``GENERATORS[generator][1]`` will actually build from ``params``.

    The 2026-09-06 audit's clay-05 finding: every generator clamps its own low
    end internally (a segment count of zero is raised to :data:`MIN_SEGMENTS`
    before a single vertex is placed) without reporting it back, so the
    properties panel was saving the number the user *typed* rather than the
    one the mesh was built from -- a saved document's params could disagree
    with its own geometry. This mirrors those floors so a caller can match
    them before the build, not discover them after it.

    It also carries ``torus``'s relational clamp -- clay-04: a ``tube`` wider
    than its ``radius`` self-intersects, which the generator's own docstring
    says is "the properties panel's business (a soft clamp on the tube
    slider)" and which, until this function, no code anywhere actually did.

    And ``column``'s own relational clamp -- the 2026-09-08 audit's clay-04:
    ``base + capital`` past :data:`COLUMN_ENDS_MAX` of the height is the
    identically shaped self-intersection this function already guards for
    torus, and this function mirrored the torus branch without noticing its
    sibling. Without it, the panel stored the raw base/capital the user typed
    while ``column`` shrank a substituted pair before ever building, so a
    saved document's params disagreed with the shaft it describes.

    And ``lathe``'s ``profile`` -- the first array-valued parameter this
    registry has: :data:`_PROFILE_CLAMPS` is :data:`_KEY_CLAMPS`'s
    counterpart for a parameter whose floor is not an int, applied the same
    way and for the same reason. ``lathe`` itself calls :func:`_clamp_profile`
    directly too, exactly as ``column`` re-applies its own base/capital
    shrink internally -- this function mirrors a generator's own floor for a
    caller that must store what the mesh was actually built from, it does not
    replace the generator refusing what it cannot represent.

    And ``sweep``'s ``taper``: the same positive floor :func:`sweep` applies
    to itself (see :data:`MIN_TAPER`), mirrored here for the reason every
    other branch above is -- a taper of exactly zero is a degenerate mesh, not
    a shape a properties panel should describe as "taper: 0". ``outline``,
    ``sections`` and ``tube``'s own ``path`` need no branch of their own: they
    are already covered by :data:`_PROFILE_CLAMPS` and :data:`_KEY_CLAMPS`
    respectively, the same generic tables ``lathe``'s ``profile`` and
    ``segments`` go through.
    """
    out = dict(params)
    for key, clamp in _KEY_CLAMPS.items():
        if key in out:
            out[key] = clamp(out[key])
    for key, normalise in _PROFILE_CLAMPS.items():
        if key in out:
            out[key] = normalise(out[key])
    if generator == "torus" and "tube" in out and "radius" in out:
        out["tube"] = min(abs(float(out["tube"])), abs(float(out["radius"])))
    if generator == "column" and "base" in out and "capital" in out and "height" in out:
        b, c, h = abs(float(out["base"])), abs(float(out["capital"])), abs(float(out["height"]))
        if b + c > h * COLUMN_ENDS_MAX:
            shrink = (h * COLUMN_ENDS_MAX) / (b + c)
            out["base"], out["capital"] = b * shrink, c * shrink
    if generator == "sweep" and "taper" in out:
        out["taper"] = max(abs(float(out["taper"])), MIN_TAPER)
    return out


# ``_mesh`` is ``mesh.from_faces`` (imported above under this name) --
# promoted there so ``agent_clay.py``'s ``clay_add_mesh`` can build a CSR
# mesh from agent-supplied corner loops without a second copy of this exact
# assembly. Kept under the original name here rather than renamed at each of
# the fifteen call sites below, which cost nothing to leave alone; see
# :func:`.mesh.from_faces` for the real docstring.


def _disc_uv(segments: int, centre: tuple[float, float], radius: float, reverse: bool = False):
    """A cap's corners on a circle in UV space, matching ``_ring``'s order.

    Caps go in their own corner of the square rather than sharing it with the
    sides: a canonical unwrap whose islands overlap is one that cannot be baked
    to, and baking is what these coordinates exist for.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    if reverse:
        theta = theta[::-1]
    return [
        (centre[0] + radius * float(np.cos(t)), centre[1] + radius * float(np.sin(t)))
        for t in theta
    ]


def _fan_uv(
    segments: int, corner: tuple[float, float], size: float, apex_first: bool
) -> list[list[tuple[float, float]]]:
    """A pole's own corner of the square -- the flat analogue of
    :func:`_disc_uv` for an end with no cap to unwrap.

    ``cone`` and ``pyramid`` already show what a fan's corners get: a
    different ``u`` per triangle rather than one shared apex coordinate, or a
    shared apex would smear the whole point of the shape into one texel. This
    is that same mapping -- base along one edge, apex along the opposite one,
    ``u`` stepping ``1/segments`` per triangle -- confined to one
    ``size``-wide square rather than the whole unit square, so it can share
    the square with the other end's cap the way two ``_disc_uv`` calls
    already do for ``column``.

    ``apex_first`` matches the corner order the triangle's own face winds in:
    a bottom pole's face is ``[pole, ring, ring + 1]`` (see ``capsule``'s
    south pole) and a top pole's is ``[ring, pole, ring + 1]`` (``cone``'s
    apex, ``capsule``'s north pole) -- the ``uv`` list has to name its
    corners in the same order as the face's loop or the texture lands on the
    wrong vertex entirely.
    """
    n = segments
    cx, cy = corner
    out: list[list[tuple[float, float]]] = []
    for i in range(n):
        ring_a = (cx + (i / n) * size, cy)
        apex = (cx + ((i + 0.5) / n) * size, cy + size)
        ring_b = (cx + ((i + 1) / n) * size, cy)
        out.append([apex, ring_a, ring_b] if apex_first else [ring_a, apex, ring_b])
    return out


def _outline_uv(
    corners: list[list[float]], origin: tuple[float, float], size: float, reverse: bool = False
) -> list[tuple[float, float]]:
    """A cap's corners taken from the outline's own bounding box, normalised
    into a ``size``-wide square at *origin* -- the polygon analogue of
    :func:`_disc_uv` for a cap that is not a circle, needed because
    ``sweep``'s cap is whatever shape the user's own outline traces.

    ``reverse`` walks the corners backwards to match a reversed cap's face
    order, exactly as :func:`_disc_uv`'s own ``reverse`` does for a circular
    one -- the near (-Z) cap's face is the outline in reverse, so its uv must
    be too, or the texture would land on the wrong corner.
    """
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    lo_x, lo_y = min(xs), min(ys)
    # The larger of the two spans, so a non-square outline (an L-bracket's own
    # bounding box, or any oblong bracket) is not stretched to fill a square
    # island -- the same texel-density reasoning ``uv.box_unwrap`` states for
    # normalising over the mesh's own largest extent rather than per island.
    span = max(max(xs) - lo_x, max(ys) - lo_y, 1e-9)
    order = list(reversed(corners)) if reverse else corners
    return [
        (origin[0] + (x - lo_x) / span * size, origin[1] + (y - lo_y) / span * size)
        for x, y in order
    ]


def _ring(radius: float, y: float, segments: int) -> np.ndarray:
    """``segments`` points on a circle in the XZ plane, at height *y*.

    Angle zero sits on +X and the sequence runs towards +Z. Seen from above --
    that is, looking down -Y -- that is *clockwise*, because Y-up
    right-handedness puts +Z towards the viewer's chin. So a ring in this order
    is the correct winding for a **bottom** cap, and a top cap is the same ring
    reversed. Getting that backwards is the flipped-cap bug, which is why it is
    written down here rather than rediscovered at each call site.

    A ring is a polygon, so a primitive's measured bounds reach its full radius
    only when a vertex happens to land on the axis -- true for any ``segments``
    divisible by four, which every default here is, and slightly short of it
    otherwise. That is honest behaviour for a polygonal approximation rather
    than something to correct for, but a properties panel showing a measured
    "size" beside a requested radius should expect the two to differ.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    return np.stack(
        [radius * np.cos(theta), np.full(segments, float(y)), radius * np.sin(theta)],
        axis=1,
    )


def _side_quads(lower: int, upper: int, segments: int) -> list[list[int]]:
    """The band of quads between two rings of ``segments`` vertices each.

    ``lower`` and ``upper`` are the first vertex index of each ring. The corner
    order -- lower *i*, upper *i*, upper *i+1*, lower *i+1* -- is the one whose
    Newell normal points away from the axis; the other traversal of the same
    four corners points into it.
    """
    return [
        [
            lower + i,
            upper + i,
            upper + (i + 1) % segments,
            lower + (i + 1) % segments,
        ]
        for i in range(segments)
    ]


def _revolve(
    profile: Sequence[Sequence[float]], segments: int
) -> tuple[np.ndarray, list[list[int]]]:
    """Revolve a bottom-to-top profile of ``(radius, y)`` stations about Y.

    ``column``'s own body, pulled out here so ``lathe`` can share it rather
    than growing a second copy of the ring-stacking loop: every station
    becomes a ring of *segments* vertices and consecutive rings are joined by
    a band of quads (:func:`_side_quads`), with an n-gon cap closing whichever
    end is not something else -- ``column`` never needed that "something
    else", because its shaft, base and capital are never zero at an end.

    **A station of zero radius at either end is a pole, not a ring of
    coincident points.** That is the one case ``column`` never exercised and
    ``lathe`` needs: a finial, a droplet, a chess pawn and a spinning top all
    come to a point, and a ring of ``segments`` vertices stacked on top of
    each other there would be non-manifold where a single vertex is exactly
    right. Only the first and last stations may do this -- see
    ``primitives.py``'s profile clamp for why a middle one may not -- so this
    function looks at exactly those two.

    The two fans are wound the way ``cone``'s apex and ``capsule``'s two
    poles already are, rather than re-derived: a bottom pole (the lowest
    station, the one before any ring) puts the pole first in its face --
    ``[pole, ring, ring + 1]``, ``capsule``'s south pole -- and a top pole
    puts it second -- ``[ring, pole, ring + 1]``, ``cone``'s apex and
    ``capsule``'s north pole.
    """
    n = int(segments)
    bottom_pole = float(profile[0][0]) == 0.0
    top_pole = float(profile[-1][0]) == 0.0
    # Two stations, both zero radius, would leave no ring for either pole to
    # fan to. ``_clamp_profile`` now refuses this exact shape -- a profile
    # with no positive radius anywhere falls back to the default rather than
    # reaching here -- so this branch is unreachable from the registry's own
    # door (``lathe`` always clamps before calling this). It stays as the
    # backstop for a caller that bypasses the clamp and hands this function
    # the raw shape directly: falling back to plain rings rather than
    # indexing past the (empty) ring list is the honest response to that,
    # which the module's rule about generators refusing gracefully asks for.
    if bottom_pole and top_pole and len(profile) <= 2:
        bottom_pole = top_pole = False
    lo = 1 if bottom_pole else 0
    hi = len(profile) - (1 if top_pole else 0)
    rings = profile[lo:hi]
    ring0 = lo

    def row(j: int) -> int:
        """First vertex index of ring *j*, counting from the bottom pole (or
        from the first station, if there is none)."""
        return ring0 + j * n

    parts: list[np.ndarray] = []
    if bottom_pole:
        parts.append(np.array([[0.0, profile[0][1], 0.0]], dtype="f8"))
    parts.extend(_ring(r, y, n) for r, y in rings)
    if top_pole:
        parts.append(np.array([[0.0, profile[-1][1], 0.0]], dtype="f8"))
    positions = np.concatenate(parts)

    pole_bottom = 0
    pole_top = ring0 + len(rings) * n

    faces: list[list[int]] = []
    if bottom_pole:
        faces.extend([pole_bottom, row(0) + i, row(0) + (i + 1) % n] for i in range(n))
    for j in range(len(rings) - 1):
        faces.extend(_side_quads(row(j), row(j + 1), n))
    if top_pole:
        faces.extend(
            [row(len(rings) - 1) + i, pole_top, row(len(rings) - 1) + (i + 1) % n]
            for i in range(n)
        )
    if not bottom_pole:
        faces.append(list(range(row(0), row(0) + n)))  # bottom cap, ring order, normal -Y
    if not top_pole:
        last = row(len(rings) - 1)
        faces.append(list(range(last + n - 1, last - 1, -1)))  # top cap, reversed, normal +Y

    return positions, faces


def _tube_frames(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One perpendicular frame per station of a centreline, by parallel
    transport -- the rotation-minimising construction :func:`tube` needs and a
    fixed world "up" cannot give it.

    Three arrays, one row per point in *points*: ``normals[i]`` is station
    *i*'s ring plane's own unit normal, and ``us[i]``/``vs[i]`` are two unit
    vectors spanning that plane, chosen right-handed with the normal exactly
    as :func:`_ring`'s own ``(X, Z, Y)`` triple is -- which is what lets
    :func:`tube` call :func:`_side_quads` in the same argument order
    :func:`_revolve` already does, rather than ``sweep``'s reversed one; see
    :func:`tube`'s own docstring for why that is the right order here and not
    an assumption carried over unchecked.

    **An interior station's ring sits in the *bisector* plane of its two
    neighbouring segments**, not either segment's own tangent alone -- a ring
    square to only the incoming segment leaves a gap on the outside of a turn
    and a pinch on the inside, visible the moment the path bends by more than
    a few degrees. The two end stations have only one neighbouring segment
    each, so that segment's own tangent is the whole answer for them.

    **The in-plane vectors are carried forward by rotation, never recomputed
    from a fixed axis.** The naive alternative -- ``cross(world_up,
    normal)`` at each station independently -- gets two things wrong at once:
    it spins the ring by whatever the bisector happens to be relative to a
    fixed axis rather than relative to its own neighbour, and it fails
    outright (a 180-degree flip, or an undefined zero vector) the moment a
    normal passes near that fixed axis, with nothing in
    :func:`~warlock.studio.clay.mesh.validate` to notice a tube gone inside
    out at exactly that station. Parallel transport has no such axis to pass
    near: ``us[i]`` is ``us[i - 1]`` rotated by the one rotation that carries
    ``normals[i - 1]`` onto ``normals[i]`` (Rodrigues' formula, since both are
    unit vectors), defined everywhere except the one case of two neighbouring
    normals turned a full 180 degrees apart -- not a path this generator's own
    convexity rule tolerates in the first place, so the branch below is a
    backstop rather than a case any default or clamp reaches. ``us[0]`` is
    perpendicular to ``normals[0]`` by construction (Gram-Schmidt against
    whichever world axis ``normals[0]`` is *least* aligned with, so the
    projection is never ill-conditioned) and is otherwise arbitrary -- there is
    no earlier twist for the first station to match, only later ones to stay
    consistent with it.
    """
    n = len(points)
    seg = np.diff(points, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    # Guarded rather than assumed: ``tube`` always calls this after its own
    # ``_clamp_path``, which has already dropped a zero-length segment, but a
    # caller that bypasses the clamp (``_revolve``'s own docstring names the
    # same case for a raw profile) gets a defined answer instead of a
    # division by zero.
    seg_len = np.where(seg_len > 0.0, seg_len, 1.0)
    tangents = seg / seg_len[:, None]

    normals = np.empty((n, 3), dtype="f8")
    normals[0] = tangents[0]
    normals[-1] = tangents[-1]
    for i in range(1, n - 1):
        bisector = tangents[i - 1] + tangents[i]
        length = np.linalg.norm(bisector)
        normals[i] = bisector / length if length > 1e-9 else tangents[i]

    axis = np.eye(3)[int(np.argmin(np.abs(normals[0])))]
    u0 = axis - np.dot(axis, normals[0]) * normals[0]
    u0 = u0 / np.linalg.norm(u0)

    us = np.empty((n, 3), dtype="f8")
    us[0] = u0
    for i in range(1, n):
        a, b = normals[i - 1], normals[i]
        cross = np.cross(a, b)
        sin_t = float(np.linalg.norm(cross))
        cos_t = float(np.dot(a, b))
        prev = us[i - 1]
        if sin_t < 1e-9:
            if cos_t > 0.0:
                us[i] = prev  # consecutive normals agree; nothing to rotate
            else:
                # 180 degrees apart -- the backstop the docstring above names.
                fallback = np.eye(3)[int(np.argmin(np.abs(b)))]
                fresh = fallback - np.dot(fallback, b) * b
                us[i] = fresh / np.linalg.norm(fresh)
        else:
            k = cross / sin_t
            rotated = prev * cos_t + np.cross(k, prev) * sin_t + k * np.dot(k, prev) * (1.0 - cos_t)
            us[i] = rotated / np.linalg.norm(rotated)

    vs = np.cross(us, normals)
    return normals, us, vs


def _tube_ring(
    center: np.ndarray, u: np.ndarray, v: np.ndarray, radius: float, sides: int
) -> np.ndarray:
    """``sides`` points on a circle of *radius* about *center*, in the plane
    spanned by *u* and *v* -- :func:`_ring`'s own construction, generalised
    from the fixed ``(X, Z)`` plane to whichever plane :func:`_tube_frames`
    built for this station. Angle zero sits on *u*, running towards *v*, for
    the same reason :func:`_ring` starts at +X and runs towards +Z: the two
    functions have to agree, because :func:`_tube_frames` chose *u* and *v* to
    make ``(u, v, normal)`` the same-handed triple ``(X, Z, Y)`` already is.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    return center + radius * (np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v)


# --- the generators ----------------------------------------------------------


def box(size: Sequence[float] = (1.0, 1.0, 1.0)) -> Mesh:
    """An axis-aligned box of the given full extents, centred on the origin.

    A negative extent is taken as its magnitude, not as a mirror -- see the
    module docstring.
    """
    hx, hy, hz = (abs(float(s)) * 0.5 for s in size)
    positions = np.array(
        [
            [-hx, -hy, -hz],
            [+hx, -hy, -hz],
            [+hx, -hy, +hz],
            [-hx, -hy, +hz],
            [-hx, +hy, -hz],
            [+hx, +hy, -hz],
            [+hx, +hy, +hz],
            [-hx, +hy, +hz],
        ],
        dtype="f4",
    )
    faces = [
        [0, 1, 2, 3],  # -Y
        [7, 6, 5, 4],  # +Y
        [3, 2, 6, 7],  # +Z
        [1, 0, 4, 5],  # -Z
        [2, 1, 5, 6],  # +X
        [0, 3, 7, 4],  # -X
    ]
    # A box's canonical unwrap *is* the cube projection, so it is that call
    # rather than a second table of the same six squares written out here.
    from .uv import box_unwrap

    return box_unwrap(_mesh(positions, faces))


def plane(size: Sequence[float] = (1.0, 1.0)) -> Mesh:
    """A single quad in the XZ plane at y = 0, facing +Y.

    The one open primitive: it has a boundary, no volume and no outward
    direction to derive, so its facing is a decision rather than a consequence.
    +Y, because a sheet a user drops into the scene is a floor.
    """
    hx, hz = (abs(float(s)) * 0.5 for s in size)
    positions = np.array(
        [[-hx, 0.0, +hz], [+hx, 0.0, +hz], [+hx, 0.0, -hz], [-hx, 0.0, -hz]],
        dtype="f4",
    )
    return _mesh(positions, [[0, 1, 2, 3]], [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]])


def cylinder(radius: float = 0.5, height: float = 1.0, segments: int = 16) -> Mesh:
    """A tube with an n-gon cap at each end, axis along Y."""
    n = _clamp_segments(segments)
    r, h = abs(float(radius)), abs(float(height)) * 0.5
    positions = np.concatenate([_ring(r, -h, n), _ring(r, +h, n)])
    faces = _side_quads(0, n, n)
    faces.append(list(range(n)))  # bottom cap, ring order, normal -Y
    faces.append(list(range(2 * n - 1, n - 1, -1)))  # top cap, reversed, normal +Y

    # The band across the top half of the square, the two caps in the bottom
    # corners. ``u`` runs to exactly 1 on the last quad rather than wrapping to
    # 0, which is the whole reason UVs are per corner.
    uv = [
        [(i / n, 0.5), (i / n, 1.0), ((i + 1) / n, 1.0), ((i + 1) / n, 0.5)]
        for i in range(n)
    ]
    uv.append(_disc_uv(n, (0.25, 0.25), 0.24))
    uv.append(_disc_uv(n, (0.75, 0.25), 0.24, reverse=True))
    return _mesh(positions, faces, uv)


def cone(radius: float = 0.5, height: float = 1.0, segments: int = 16) -> Mesh:
    """A base ring, an n-gon bottom and a fan of triangles up to a single apex.

    The apex is one vertex, not one per side: the sides meet there, and
    splitting them would produce a seam an exporter has no reason to keep. The
    faceting a shared apex would otherwise cause is not a concern here because
    every face is flat-shaded anyway -- ``render_arrays`` splits a flat face's
    corners on the way to the GPU regardless.
    """
    n = _clamp_segments(segments)
    r, h = abs(float(radius)), abs(float(height)) * 0.5
    positions = np.concatenate([_ring(r, -h, n), [[0.0, +h, 0.0]]])
    apex = n
    faces: list[list[int]] = [[i, apex, (i + 1) % n] for i in range(n)]
    faces.append(list(range(n)))  # base cap, normal -Y

    # The apex gets a *different* u per side triangle -- it is one vertex and
    # many corners, which is exactly the case a per-corner field exists for. A
    # shared apex uv would smear the whole top of the texture into one point.
    uv = [
        [(i / n, 0.5), ((i + 0.5) / n, 1.0), ((i + 1) / n, 0.5)] for i in range(n)
    ]
    uv.append(_disc_uv(n, (0.25, 0.25), 0.24))
    return _mesh(positions, faces, uv)


def uv_sphere(radius: float = 0.5, segments: int = 16, rings: int = 8) -> Mesh:
    """A latitude/longitude sphere: quads between rings, triangles at the poles.

    ``rings`` counts the *bands* from pole to pole, so there are ``rings - 1``
    intermediate latitude rings and the vertex count is
    ``(rings - 1) * segments + 2``. The poles are single vertices and the bands
    that touch them are therefore triangles, not degenerate quads with two
    coincident corners -- a degenerate quad survives ``validate`` and then
    produces a zero-area triangle in every consumer downstream of it.
    """
    n = _clamp_segments(segments)
    m = _clamp_rings(rings)
    r = abs(float(radius))

    top, bottom = 0, 1 + (m - 1) * n
    rows = []
    for j in range(1, m):
        phi = np.pi * j / m
        rows.append(_ring(r * np.sin(phi), r * np.cos(phi), n))
    positions = np.concatenate([[[0.0, +r, 0.0]], *rows, [[0.0, -r, 0.0]]])

    def row(j: int) -> int:
        """First vertex index of intermediate ring *j*, counting from the top."""
        return 1 + (j - 1) * n

    faces: list[list[int]] = [[row(1) + i, top, row(1) + (i + 1) % n] for i in range(n)]
    for j in range(1, m - 1):
        faces.extend(_side_quads(row(j + 1), row(j), n))
    faces.extend([bottom, row(m - 1) + i, row(m - 1) + (i + 1) % n] for i in range(n))

    # Equirectangular: u around, v from the south pole up. The whole square,
    # with no caps to pack, because a sphere has no faces that are not part of
    # the same surface.
    def band(j: int) -> float:
        return 1.0 - j / m

    uv: list[list[tuple[float, float]]] = [
        [(i / n, band(1)), ((i + 0.5) / n, 1.0), ((i + 1) / n, band(1))] for i in range(n)
    ]
    for j in range(1, m - 1):
        uv.extend(
            [
                (i / n, band(j + 1)),
                (i / n, band(j)),
                ((i + 1) / n, band(j)),
                ((i + 1) / n, band(j + 1)),
            ]
            for i in range(n)
        )
    uv.extend(
        [((i + 0.5) / n, 0.0), (i / n, band(m - 1)), ((i + 1) / n, band(m - 1))]
        for i in range(n)
    )
    return _mesh(positions, faces, uv)


def torus(
    radius: float = 0.35,
    tube: float = 0.15,
    segments: int = 24,
    sides: int = 12,
) -> Mesh:
    """A ring of ``segments`` tube cross-sections, each of ``sides`` corners.

    ``radius`` is to the centre of the tube and ``tube`` is the tube's own
    radius, so the outer extent is their sum -- the same pair of numbers
    Blender's torus takes, and the reason the default pair adds to 0.5: every
    primitive here defaults to fitting a one-metre box. **Not the ``tube``
    generator** (:func:`tube`, elsewhere in this registry): that is a
    different shape, a circular cross-section swept along an open path rather
    than around a closed ring, and it takes the name because "a tube" is the
    right word for what it builds; the two share a name for the same reason
    they share a shape, and :func:`tube`'s own docstring makes the same note
    back.

    **A ``tube`` larger than ``radius`` self-intersects and is not refused
    here.** The mesh stays valid, closed and outward-wound -- the tube simply
    passes through its own axis -- so there is no geometric rule to enforce, and
    the two are legitimately independent right up to the point where they are
    not. Keeping the pair sane is :func:`clamp_params`' business, which every
    caller that stores what a user typed applies *before* ``build``: this
    docstring used to send that job to the properties panel, and the panel used
    to say the generator raised on the pair, so the clamp existed in neither
    place and a self-intersecting torus was one keystroke away (the 2026-09-06
    audit, finding clay-04). The generator itself still refuses only what it
    cannot represent, which is the same division of labour as ``segments``.
    """
    n = _clamp_segments(segments)
    k = _clamp_segments(sides)
    R, r = abs(float(radius)), abs(float(tube))

    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)[:, None]
    phi = np.linspace(0.0, 2.0 * np.pi, k, endpoint=False)[None, :]
    ring_radius = R + r * np.cos(phi)
    positions = np.stack(
        [
            ring_radius * np.cos(theta),
            np.broadcast_to(r * np.sin(phi), (n, k)),
            ring_radius * np.sin(theta),
        ],
        axis=-1,
    ).reshape(-1, 3)

    # Around the tube first, then around the ring: the reverse traversal of the
    # same four corners points into the tube rather than out of it.
    faces = [
        [
            i * k + j,
            i * k + (j + 1) % k,
            ((i + 1) % n) * k + (j + 1) % k,
            ((i + 1) % n) * k + j,
        ]
        for i in range(n)
        for j in range(k)
    ]
    # Both seams run to exactly 1 rather than wrapping to 0, for the reason the
    # cylinder's does: the corners are per face, so the last ring and the last
    # tube section can each close the square instead of folding it.
    uv = [
        [
            (i / n, j / k),
            (i / n, (j + 1) / k),
            ((i + 1) / n, (j + 1) / k),
            ((i + 1) / n, j / k),
        ]
        for i in range(n)
        for j in range(k)
    ]
    return _mesh(positions, faces, uv)


def grid(size: Sequence[float] = (1.0, 1.0), divisions: int = 4) -> Mesh:
    """A subdivided sheet in the XZ plane at y = 0, facing +Y.

    ``plane`` is this at one division and stays a separate entry rather than
    being folded in: it is a single quad, which is what a decal, a billboard or
    a backdrop wants, and a user who asks for "plane" and gets sixteen faces has
    to go and find the number that undoes that. What a grid is *for* is the
    thing a single quad cannot do -- carry a displacement, a proportional edit
    or a terrain sculpt, all of which need interior vertices to move.

    Open, like ``plane``: it has a boundary and no volume, so there is no
    outward direction to derive and +Y is the same decision for the same reason.

    UV runs 0..1 across the whole sheet, so a texture fits it once however many
    divisions it is cut into -- the coordinates describe the surface, not the
    faces it happens to be made of.
    """
    hx, hz = (abs(float(s)) * 0.5 for s in size)
    n = _clamp_divisions(divisions)
    xs = np.linspace(-hx, +hx, n + 1)
    zs = np.linspace(+hz, -hz, n + 1)
    positions = np.array(
        [[x, 0.0, z] for z in zs for x in xs],
        dtype="f4",
    )

    def at(i: int, j: int) -> int:
        """Vertex index of column *i*, row *j*, rows running +Z to -Z."""
        return j * (n + 1) + i

    # The same corner order ``plane`` uses -- (-x,+z), (+x,+z), (+x,-z), (-x,-z)
    # -- which is the traversal whose Newell normal points at +Y.
    faces = [
        [at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)]
        for j in range(n)
        for i in range(n)
    ]
    uv = [
        [
            (i / n, j / n),
            ((i + 1) / n, j / n),
            ((i + 1) / n, (j + 1) / n),
            (i / n, (j + 1) / n),
        ]
        for j in range(n)
        for i in range(n)
    ]
    return _mesh(positions, faces, uv)


def capsule(
    radius: float = 0.25,
    height: float = 0.5,
    segments: int = 16,
    rings: int = 4,
) -> Mesh:
    """A cylinder with a hemisphere on each end, axis along Y.

    ``height`` is the **cylindrical** section alone, so the whole shape spans
    ``height + 2 * radius``. That is Blender's reading of the same pair and it is
    the one that keeps the two numbers independent: a capsule whose ``height``
    meant the total would silently stop being a capsule -- and start being a
    sphere, or nothing -- as the radius passed half of it, which is a shape
    change with no control the user can see it in. The defaults still add up to
    a one-metre box, as every other primitive's do.

    ``rings`` counts the latitude bands in **one** hemisphere, so the profile is
    ``rings`` bands up, one cylinder band across the middle, and ``rings`` bands
    down. There is no cap n-gon and no pole fan sharing a ring with the tube:
    the two hemisphere rings that meet the cylinder are separate rows of
    vertices at the same radius, which is what lets the cylinder band be a plain
    quad strip and keeps every band's winding the same expression.

    UV is a cylindrical unwrap whose ``v`` follows **arc length along the
    profile** rather than height, so the texel density does not collapse at the
    ends: a naive v = (y - min) / span squashes each hemisphere into a band as
    tall as its own bulge, which on a stubby capsule is most of the texture in
    a tenth of the square. ``u`` runs to exactly 1 on the last column, per
    corner, for the reason the cylinder's does.
    """
    n = _clamp_segments(segments)
    m = _clamp_rings(rings)
    r, h = abs(float(radius)), abs(float(height)) * 0.5

    # The profile, north to south, as (ring radius, y). The two poles are the
    # ends and are single vertices; everything between them is a full ring.
    profile: list[tuple[float, float]] = [(0.0, +h + r)]
    for k in range(1, m + 1):
        phi = 0.5 * np.pi * k / m
        profile.append((r * float(np.sin(phi)), +h + r * float(np.cos(phi))))
    for k in range(m):
        psi = 0.5 * np.pi * k / m
        profile.append((r * float(np.cos(psi)), -h - r * float(np.sin(psi))))
    profile.append((0.0, -h - r))

    rows = profile[1:-1]
    positions = np.concatenate(
        [[[0.0, profile[0][1], 0.0]]]
        + [_ring(rr, y, n) for rr, y in rows]
        + [[[0.0, profile[-1][1], 0.0]]]
    )
    top, bottom = 0, 1 + len(rows) * n

    def row(j: int) -> int:
        """First vertex index of intermediate ring *j*, counting from the top."""
        return 1 + j * n

    # The pole fans are wound the way ``uv_sphere``'s are, and the bands between
    # rings the way ``_side_quads`` orders them -- upper ring first, because the
    # rings run downward and "lower" there means the ring further from +Y.
    faces: list[list[int]] = [[row(0) + i, top, row(0) + (i + 1) % n] for i in range(n)]
    for j in range(len(rows) - 1):
        faces.extend(_side_quads(row(j + 1), row(j), n))
    faces.extend(
        [bottom, row(len(rows) - 1) + i, row(len(rows) - 1) + (i + 1) % n] for i in range(n)
    )

    # v by arc length along the profile, measured south-to-north so v = 0 is the
    # bottom pole -- the same direction ``uv_sphere``'s bands run.
    points = np.array(profile, dtype="f8")
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    along = np.concatenate([[0.0], np.cumsum(steps)])
    total = float(along[-1])
    v = 1.0 - along / total if total > 0.0 else np.zeros(len(along))

    uv: list[list[tuple[float, float]]] = [
        [(i / n, float(v[1])), ((i + 0.5) / n, float(v[0])), ((i + 1) / n, float(v[1]))]
        for i in range(n)
    ]
    for j in range(len(rows) - 1):
        uv.extend(
            [
                (i / n, float(v[j + 2])),
                (i / n, float(v[j + 1])),
                ((i + 1) / n, float(v[j + 1])),
                ((i + 1) / n, float(v[j + 2])),
            ]
            for i in range(n)
        )
    uv.extend(
        [
            ((i + 0.5) / n, float(v[-1])),
            (i / n, float(v[-2])),
            ((i + 1) / n, float(v[-2])),
        ]
        for i in range(n)
    )
    return _mesh(positions, faces, uv)


# The regular icosahedron, before normalisation: three mutually perpendicular
# golden rectangles. Written out rather than derived because the *face* list
# below is written out and the two have to agree about which vertex is which --
# deriving one and not the other is how a shell comes out with five faces
# reversed and passes every test that only counts things.
_ICO_T = (1.0 + 5.0**0.5) * 0.5

_ICO_VERTS = (
    (-1.0, _ICO_T, 0.0), (1.0, _ICO_T, 0.0), (-1.0, -_ICO_T, 0.0), (1.0, -_ICO_T, 0.0),
    (0.0, -1.0, _ICO_T), (0.0, 1.0, _ICO_T), (0.0, -1.0, -_ICO_T), (0.0, 1.0, -_ICO_T),
    (_ICO_T, 0.0, -1.0), (_ICO_T, 0.0, 1.0), (-_ICO_T, 0.0, -1.0), (-_ICO_T, 0.0, 1.0),
)

# Every triangle counter-clockwise seen from outside, which is what makes the
# Newell normal point away from the centre -- the module's third rule, and the
# one whose failure is invisible in the viewport.
_ICO_FACES = (
    (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
    (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
    (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
    (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
)


def _subdivide(
    verts: list[np.ndarray], faces: list[tuple[int, int, int]]
) -> tuple[list[np.ndarray], list[tuple[int, int, int]]]:
    """One Loop-style split: each triangle into four, midpoints on the sphere.

    The midpoint cache is keyed on the *sorted* vertex pair, which is what makes
    two triangles sharing an edge share the vertex on it. Without it the shell
    cracks: each face would mint its own copy, the positions would coincide, and
    the mesh would be two-manifold in appearance and open in every measurement.

    The four children are wound in the parent's own direction, so consistency is
    a property of the base list and is never re-derived.
    """
    cache: dict[tuple[int, int], int] = {}

    def middle(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        found = cache.get(key)
        if found is not None:
            return found
        point = verts[a] + verts[b]
        verts.append(point / np.linalg.norm(point))
        cache[key] = len(verts) - 1
        return cache[key]

    out: list[tuple[int, int, int]] = []
    for a, b, c in faces:
        ab, bc, ca = middle(a, b), middle(b, c), middle(c, a)
        out.extend([(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)])
    return verts, out


def _sphere_uv(
    positions: np.ndarray, faces: Sequence[Sequence[int]]
) -> list[list[tuple[float, float]]]:
    """A latitude/longitude unwrap of a sphere whose faces do not share rings.

    ``uv_sphere`` and ``capsule`` can put ``u`` on a corner directly, because
    their faces are laid out in columns and the last column simply closes at 1.
    An icosphere has no columns: its triangles straddle the seam meridian, and a
    per-*vertex* longitude therefore hands one of them the corner pair
    ``0.98, 0.02`` and wraps the whole texture backwards across it.

    So longitude is unwrapped **per face**: each corner is taken to the branch
    nearest the face's first corner, which makes the island contiguous. That can
    put it outside the square by up to its own width, and the island is then
    *translated* back in -- never scaled, so the mapping stays exact and the only
    thing paid is continuity, at the seam, where a sphere has none to keep.

    A corner on the axis has no longitude at all; it takes the mean of its
    face's other two, which is the same answer ``cone`` gives its apex and for
    the same reason.
    """
    positions = np.asarray(positions, dtype="f8")
    radius = float(np.max(np.linalg.norm(positions, axis=1))) or 1.0
    lon = (np.arctan2(positions[:, 2], positions[:, 0]) / (2.0 * np.pi)) % 1.0
    axial = np.hypot(positions[:, 0], positions[:, 2]) < 1e-9
    lat = 0.5 + np.arcsin(np.clip(positions[:, 1] / radius, -1.0, 1.0)) / np.pi

    out: list[list[tuple[float, float]]] = []
    for face in faces:
        known = [lon[i] for i in face if not axial[i]]
        anchor = known[0] if known else 0.0
        us = []
        for i in face:
            if axial[i]:
                us.append(None)
                continue
            # The branch nearest the anchor: ``round`` of the difference is the
            # number of whole turns to take off.
            us.append(float(lon[i] - round(float(lon[i]) - anchor)))
        fallback = float(np.mean([u for u in us if u is not None])) if known else 0.0
        row = [fallback if u is None else u for u in us]
        low, high = min(row), max(row)
        shift = -low if low < 0.0 else (1.0 - high if high > 1.0 else 0.0)
        out.append([(u + shift, float(lat[i])) for u, i in zip(row, face, strict=True)])
    return out


def icosphere(radius: float = 0.5, subdivisions: int = 2) -> Mesh:
    """A geodesic sphere: an icosahedron, split and pushed back out.

    The one sphere with **even triangles**. ``uv_sphere``'s poles are fans of
    slivers and its equator quads are many times their area, which is what makes
    it the wrong ball to sculpt, to bevel, or to bake to -- and the right one to
    wrap an equirectangular texture round, which is why both ship rather than
    one replacing the other.

    ``subdivisions`` is capped at :data:`MAX_SUBDIVISIONS`; see the comment
    there. Zero is the bare icosahedron, which is a legitimate shape to place.
    """
    r = abs(float(radius))
    depth = _clamp_subdivisions(subdivisions)
    verts = [np.asarray(v, dtype="f8") / np.linalg.norm(v) for v in _ICO_VERTS]
    faces = [tuple(f) for f in _ICO_FACES]
    for _ in range(depth):
        verts, faces = _subdivide(verts, faces)
    unit = np.asarray(verts, dtype="f8")
    return _mesh(unit * r, faces, _sphere_uv(unit, faces))


# How much wider a column's plinth and capital are than its shaft. A constant
# rather than a parameter because ``base`` and ``capital`` are *heights*: they
# are the two numbers a user reads off a reference photograph, and a third
# "how much wider" slider buys a shape nobody asked for at the cost of a
# control on every column ever placed. Blocks the same width as the shaft would
# not read as a column at all -- they would be invisible.
COLUMN_FLARE = 1.35

# The most of a column's height its plinth and capital may take between them.
# Without it, ``base + capital >= height`` puts the two blocks past each other:
# the shaft's two rings swap order, its band faces inward, and ``validate``
# accepts it because only the geometry is wrong -- the same failure a negative
# height would cause, arriving through two positive numbers instead.
COLUMN_ENDS_MAX = 0.8


def pyramid(base: float = 1.0, height: float = 1.0, sides: int = 4) -> Mesh:
    """An n-gon base and a fan of triangles up to a single apex, base flat to X.

    This is not ``cone`` with a small ``segments``, and the difference is the
    half-segment of rotation. ``_ring`` starts at +X, so a four-segment cone is
    a square standing on a *corner* -- its faces point along the diagonals, and
    a user who places a pyramid to be the roof of an axis-aligned box gets one
    turned 45 degrees to it. So the ring here is rotated by half a segment,
    which puts a **face** at +X instead of a vertex, and ``base`` is then the
    flat-to-flat width: exactly the box a four-sided pyramid stands on.

    That also fixes what the UV is for. A cone is unwrapped as a cone -- the
    sides are one continuous band around the apex -- whereas a pyramid's sides
    are flat plates, so they get the same band but there is no curvature for it
    to lie about. The two generators ship separately because the shapes are
    different objects to a modeller, not because the code could not be shared.
    """
    n = _clamp_segments(sides)
    # ``base`` is the flat-to-flat width, so the circumradius is the apothem
    # over ``cos(pi / n)``. At n = 4 that is the half-diagonal of the square.
    half = abs(float(base)) * 0.5
    h = abs(float(height)) * 0.5
    r = half / float(np.cos(np.pi / n))

    ring = _ring(r, -h, n)
    # A rigid rotation about Y by half a segment: it moves every vertex by the
    # same angle, so the ring's order -- and therefore the winding every face
    # below inherits from it -- is untouched.
    turn = np.pi / n
    cos_t, sin_t = float(np.cos(turn)), float(np.sin(turn))
    x, z = ring[:, 0].copy(), ring[:, 2].copy()
    ring[:, 0] = cos_t * x - sin_t * z
    ring[:, 2] = sin_t * x + cos_t * z

    positions = np.concatenate([ring, [[0.0, +h, 0.0]]])
    apex = n
    faces: list[list[int]] = [[i, apex, (i + 1) % n] for i in range(n)]
    faces.append(list(range(n)))  # base cap, ring order, normal -Y

    # The apex gets a different u per side, for the reason ``cone``'s does: it
    # is one vertex and n corners, which is the case a per-corner field is for.
    uv: list[list[tuple[float, float]]] = [
        [(i / n, 0.5), ((i + 0.5) / n, 1.0), ((i + 1) / n, 0.5)] for i in range(n)
    ]
    uv.append(_disc_uv(n, (0.25, 0.25), 0.24))
    return _mesh(positions, faces, uv)


def arch(
    width: float = 1.0,
    height: float = 1.0,
    depth: float = 0.5,
    thickness: float = 0.25,
    segments: int = 12,
) -> Mesh:
    """A swept profile: two legs and a semicircular head, extruded along Z.

    The profile is sampled at N *stations*, and each station carries two points
    -- one on the outer boundary and one on the inner -- so every face in the
    shell is a quad between two adjacent stations. That is the whole reason the
    shape is built this way rather than as a boolean of a box and a cylinder:
    convexity comes for free with quads, and the module's fan-triangulation
    rule is a promise about *every* face rather than one this generator would
    have to argue about.

    Five families of face come out of it and there are no others: the front
    plates at +Z, the back plates at -Z, the band around the outside, the band
    around the inside of the opening, and one quad capping each leg where it
    meets the ground. **The leg ends are capped**, deliberately: an uncapped
    arch would be an open shell, and the registry's sweep tests
    (``test_every_closed_generator_is_wound_outward`` and the
    each-edge-exactly-twice check) exempt only the shapes named in
    :data:`OPEN_GENERATORS` -- joining that set to avoid two quads would be
    opting out of the very tests this generator joined the registry for.

    Note what the opening is *not*: a hole through material. An arch is a
    horseshoe, open at the bottom, so the shell is a topological ball like every
    other closed primitive here and not a torus.

    ``height`` is the whole thing, ground to crown, and the head's radius is
    half the width -- so a ``height`` below that leaves no leg at all and is
    raised to it. ``thickness`` is likewise held inside the head's radius: a
    wall as thick as the arch is wide has no opening left to be an arch of.
    """
    n = _clamp_segments(segments)
    r_out = abs(float(width)) * 0.5
    # The head is a semicircle of the full half-width, so the crown is at
    # ``springline + r_out``: a height below the radius has no leg to stand on
    # and the arch would be a half-annulus hovering with its own head cut off.
    h = max(abs(float(height)), r_out)
    d = max(abs(float(depth)), 1e-4) * 0.5
    t = min(max(abs(float(thickness)), r_out * 0.02), r_out * 0.9)
    r_in = r_out - t
    springline = h - r_out

    outer: list[tuple[float, float]] = []
    inner: list[tuple[float, float]] = []
    # The legs are stations only when there are legs. At ``height == width / 2``
    # the springline is the ground and a leg-bottom station would duplicate the
    # first point of the head, which is a zero-area quad in four faces at once.
    if springline > 1e-6:
        outer.append((-r_out, 0.0))
        inner.append((-r_in, 0.0))
    for k in range(n + 1):
        angle = np.pi * (1.0 - k / n)
        cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
        outer.append((r_out * cos_a, springline + r_out * sin_a))
        inner.append((r_in * cos_a, springline + r_in * sin_a))
    if springline > 1e-6:
        outer.append((+r_out, 0.0))
        inner.append((+r_in, 0.0))

    stations = len(outer)
    # y runs 0..h in the profile and the primitive is centred, so it is shifted
    # by half the height on the way into the buffer rather than in the maths.
    def plate(points: list[tuple[float, float]], z: float) -> np.ndarray:
        return np.array([[x, y - h * 0.5, z] for x, y in points], dtype="f8")

    positions = np.concatenate(
        [plate(outer, +d), plate(inner, +d), plate(outer, -d), plate(inner, -d)]
    )
    fo, fi, bo, bi = 0, stations, 2 * stations, 3 * stations

    faces: list[list[int]] = []
    # Front, seen from +Z: outer down to inner and back along the next station
    # is counter-clockwise; the other traversal of the same four corners faces
    # into the wall. Back is that order reversed, which is the same rule.
    faces.extend([fo + i, fi + i, fi + i + 1, fo + i + 1] for i in range(stations - 1))
    faces.extend([bo + i, bo + i + 1, bi + i + 1, bi + i] for i in range(stations - 1))
    faces.extend([fo + i, fo + i + 1, bo + i + 1, bo + i] for i in range(stations - 1))
    faces.extend([fi + i, bi + i, bi + i + 1, fi + i + 1] for i in range(stations - 1))
    last = stations - 1
    # Both leg bottoms face -Y, and they are written out separately rather than
    # in a loop because the two ends traverse the same rectangle in opposite
    # directions -- the profile runs left to right, so "outer then inner" is a
    # different turn at each end.
    faces.append([fo, bo, bi, fi])
    faces.append([fo + last, fi + last, bi + last, bo + last])

    # Four islands in the four quadrants of the square plus the two small caps
    # along the bottom edge, because a canonical unwrap whose islands overlap is
    # one that cannot be baked to.
    def across(i: int) -> float:
        return (outer[i][0] + r_out) / (2.0 * r_out)

    def up(i: int, points: list[tuple[float, float]]) -> float:
        return points[i][1] / h if h > 0.0 else 0.0

    def station(i: int) -> float:
        return i / max(stations - 1, 1)

    uv: list[list[tuple[float, float]]] = []
    for i in range(stations - 1):  # front, top-left quadrant
        uv.append(
            [
                (0.5 * across(i), 0.5 + 0.5 * up(i, outer)),
                (0.5 * across(i), 0.5 + 0.5 * up(i, inner)),
                (0.5 * across(i + 1), 0.5 + 0.5 * up(i + 1, inner)),
                (0.5 * across(i + 1), 0.5 + 0.5 * up(i + 1, outer)),
            ]
        )
    for i in range(stations - 1):  # back, top-right quadrant, mirrored
        uv.append(
            [
                (1.0 - 0.5 * across(i), 0.5 + 0.5 * up(i, outer)),
                (1.0 - 0.5 * across(i + 1), 0.5 + 0.5 * up(i + 1, outer)),
                (1.0 - 0.5 * across(i + 1), 0.5 + 0.5 * up(i + 1, inner)),
                (1.0 - 0.5 * across(i), 0.5 + 0.5 * up(i, inner)),
            ]
        )
    for i in range(stations - 1):  # outer band, left of the bottom half
        uv.append(
            [
                (0.5 * station(i), 0.49),
                (0.5 * station(i + 1), 0.49),
                (0.5 * station(i + 1), 0.27),
                (0.5 * station(i), 0.27),
            ]
        )
    for i in range(stations - 1):  # inner band, right of the bottom half
        uv.append(
            [
                (0.5 + 0.5 * station(i), 0.27),
                (0.5 + 0.5 * station(i), 0.49),
                (0.5 + 0.5 * station(i + 1), 0.49),
                (0.5 + 0.5 * station(i + 1), 0.27),
            ]
        )
    uv.append([(0.01, 0.01), (0.24, 0.01), (0.24, 0.24), (0.01, 0.24)])
    uv.append([(0.51, 0.01), (0.74, 0.01), (0.74, 0.24), (0.51, 0.24)])
    return _mesh(positions, faces, uv)


def column(
    radius: float = 0.35,
    height: float = 2.0,
    segments: int = 16,
    base: float = 0.15,
    capital: float = 0.15,
) -> Mesh:
    """A lathe with a fixed shape: a shaft with a plinth at the bottom and a
    block at the top, rather than the arbitrary profile :func:`lathe` takes.
    Built as rings at varying radius, with an n-gon cap at each end, through
    the shared :func:`_revolve` -- neither end here is ever a pole, since a
    shaft, a plinth and a capital are never zero at an end, which is the one
    case ``lathe`` needed and this generator does not.

    ``base`` and ``capital`` are **heights** -- how tall the plinth at the
    bottom and the block at the top are -- and how much wider than the shaft
    they sit is :data:`COLUMN_FLARE`, a constant rather than a third slider.
    Setting either to zero removes it, and setting **both** to zero leaves the
    two rings and two caps of a plain cylinder, which is the property that says
    this generator is a superset of one rather than a shape with a cylinder
    hidden inside it.

    Between them they may take :data:`COLUMN_ENDS_MAX` of the height and no
    more. Past that the plinth's top ring would sit *above* the capital's
    bottom one, the shaft band between them would face inward, and ``validate``
    would accept every bit of it -- the inside-out failure a negative height
    causes, arriving here through two perfectly positive numbers.
    """
    n = _clamp_segments(segments)
    r = abs(float(radius))
    h = abs(float(height))
    half = h * 0.5
    b, c = abs(float(base)), abs(float(capital))
    if b + c > h * COLUMN_ENDS_MAX:
        shrink = (h * COLUMN_ENDS_MAX) / (b + c)
        b, c = b * shrink, c * shrink
    wide = r * COLUMN_FLARE

    # The profile, bottom to top, as (ring radius, y). A block is a straight
    # section and then a taper into the shaft, so it reads as a plinth rather
    # than as a step -- and the taper is what keeps the two rings that meet the
    # shaft at different heights, so no band in the lathe is zero-height.
    profile: list[tuple[float, float]] = []
    if b > 0.0:
        profile.extend([(wide, -half), (wide, -half + b * 0.5), (r, -half + b)])
    else:
        profile.append((r, -half))
    if c > 0.0:
        profile.extend([(r, +half - c), (wide, +half - c * 0.5), (wide, +half)])
    else:
        profile.append((r, +half))

    positions, faces = _revolve(profile, n)
    rings = len(profile)

    # ``v`` follows arc length along the profile rather than height, for the
    # reason ``capsule``'s does: a plinth is a tenth of the column's height and
    # a fifth of its silhouette, and height alone would squash it to the former.
    points = np.array(profile, dtype="f8")
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    along = np.concatenate([[0.0], np.cumsum(steps)])
    total = float(along[-1])
    v = 0.5 + 0.5 * (along / total if total > 0.0 else np.zeros(len(along)))

    uv: list[list[tuple[float, float]]] = []
    for j in range(rings - 1):
        uv.extend(
            [
                (i / n, float(v[j])),
                (i / n, float(v[j + 1])),
                ((i + 1) / n, float(v[j + 1])),
                ((i + 1) / n, float(v[j])),
            ]
            for i in range(n)
        )
    uv.append(_disc_uv(n, (0.25, 0.25), 0.24))
    uv.append(_disc_uv(n, (0.75, 0.25), 0.24, reverse=True))
    return _mesh(positions, faces, uv)


# A stemmed goblet, bottom to top: a pointed foot, a thin stem, a wide bowl
# and a flat-capped rim. Seven stations -- enough to read as turned rather
# than as a cone -- spanning y from -0.5 to +0.5 (symmetric about zero, for
# the centred-on-the-origin rule) with every radius under 0.32 (a 0.64 m
# diameter, inside the one-metre box every default here fits), and no two
# consecutive stations equal: the (0.05, ...) pair is a straight stem run,
# which is a real lathe feature (a constant-radius section), not a
# duplicate -- the two differ in ``y``.
LATHE_DEFAULT_PROFILE: tuple[tuple[float, float], ...] = (
    (0.00, -0.50),  # the foot's point -- a stemmed goblet's one contact
    (0.18, -0.42),  # the foot, splayed wide for balance
    (0.05, -0.28),  # the stem, thin
    (0.05, 0.05),  # the stem, still thin, up to the underside of the bowl
    (0.32, 0.22),  # the bowl, at its widest
    (0.16, 0.40),  # the bowl narrowing toward the rim
    (0.20, 0.50),  # the rim, capped flat
)


def lathe(
    profile: Sequence[Sequence[float]] = LATHE_DEFAULT_PROFILE, segments: int = 16
) -> Mesh:
    """The general case of :func:`column`: an arbitrary profile of
    ``[radius, y]`` stations, bottom to top, revolved about Y -- rather than
    the one fixed plinth/shaft/capital shape ``column``'s two extra numbers
    can reach. This is the shape a bottle, a vase, a goblet, a handle or a
    turned finial needs and none of the other twelve can give it: something
    that narrows, widens and narrows again along one axis, by however many
    stations the silhouette takes.

    The default profile (:data:`LATHE_DEFAULT_PROFILE`) reads as a stemmed
    goblet -- a pointed foot, a thin stem, a wide bowl and a flat-capped rim.

    **A station of zero radius at either end is a pole, not a degenerate
    ring.** :func:`_revolve` fans it to its neighbouring ring the way
    ``cone``'s apex and ``capsule``'s two poles already do, rather than
    stacking ``segments`` coincident points there -- which is what lets this
    generator reach a finial, a chess pawn or a spinning top as well as a
    bottle. Only the two ends may do this: a middle station whose radius
    reached zero would pinch the surface to a single non-manifold point with
    no modelling meaning, which is why :func:`_clamp_profile` floors one
    there instead.

    **The generator applies its own profile clamp**, the same division of
    labour ``torus``'s docstring states and ``column`` already practises for
    its own base/capital shrink: ``clamp_params`` mirrors
    :func:`_clamp_profile` for a caller that must store what the mesh was
    actually built from, but a document loaded from an old save or a profile
    handed in raw by an agent has had no such caller in front of it, so the
    generator refuses to trust one.

    UV follows ``column``'s own layout -- ``v`` by arc length along the whole
    profile rather than by height, so a thin stem does not eat the same share
    of the texture as the wide bowl beside it, kept to the top half of the
    square so the bottom half is free for whatever each end needs. A
    non-pole end packs an n-gon cap into its own corner with :func:`_disc_uv`,
    exactly as ``column``'s two caps do; a pole end has no disc to unwrap, so
    it gets :func:`_fan_uv`'s corner instead -- the same "different ``u`` per
    triangle" rule ``cone``'s apex already uses, confined to one quadrant
    rather than spread across the whole square.
    """
    n = _clamp_segments(segments)
    stations = _clamp_profile(profile)
    positions, faces = _revolve(stations, n)

    bottom_pole = float(stations[0][0]) == 0.0
    top_pole = float(stations[-1][0]) == 0.0
    # Mirrors ``_revolve``'s own guard exactly (dead in practice, since
    # ``_clamp_profile`` above already refuses a no-positive-radius profile
    # before this line ever runs): with no ring for either pole to fan to, it
    # falls back to plain rings, and the uv built below must agree with the
    # faces it actually produced.
    if bottom_pole and top_pole and len(stations) <= 2:
        bottom_pole = top_pole = False
    lo = 1 if bottom_pole else 0
    hi = len(stations) - (1 if top_pole else 0)
    ring0 = lo
    n_rings = hi - lo

    def row(j: int) -> int:
        return ring0 + j * n

    points = np.array(stations, dtype="f8")
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    along = np.concatenate([[0.0], np.cumsum(steps)])
    total = float(along[-1])
    v = 0.5 + 0.5 * (along / total if total > 0.0 else np.zeros(len(along)))

    uv: list[list[tuple[float, float]]] = []
    if bottom_pole:
        uv.extend(_fan_uv(n, (0.01, 0.01), 0.48, apex_first=True))
    for j in range(n_rings - 1):
        uv.extend(
            [
                (i / n, float(v[lo + j])),
                (i / n, float(v[lo + j + 1])),
                ((i + 1) / n, float(v[lo + j + 1])),
                ((i + 1) / n, float(v[lo + j])),
            ]
            for i in range(n)
        )
    if top_pole:
        uv.extend(_fan_uv(n, (0.51, 0.01), 0.48, apex_first=False))
    if not bottom_pole:
        uv.append(_disc_uv(n, (0.25, 0.25), 0.24))
    if not top_pole:
        uv.append(_disc_uv(n, (0.75, 0.25), 0.24, reverse=True))
    return _mesh(positions, faces, uv)


SWEEP_DEFAULT_OUTLINE: tuple[tuple[float, float], ...] = (
    (0.5, -0.5),
    (0.5, -0.1),
    (-0.1, -0.1),
    (-0.1, 0.5),
    (-0.5, 0.5),
    (-0.5, -0.5),
)
"""An L-bracket, deliberately concave -- the reflex corner is the inside of
the L, at ``(-0.1, -0.1)``. ``sweep`` is the one generator in the registry
whose cap cannot keep the module's "every face is convex" rule (see
:data:`CONCAVE_GENERATORS`), and a convex default would leave that exemption
untested by the very shape it exists for: the registry's parametrised tests
only ever build a generator's defaults and its clamped minimum, never an
arbitrary outline a user might type in later. Sized to fit the one-metre box
every default here fits, the same convention ``torus``'s own default pair
states.

**The starting corner is not arbitrary.** ``mesh.triangulate`` fans from a
face's *first* corner, and a fan from ``(-0.5, -0.5)`` -- the corner
diagonally opposite the notch -- happens to see every other corner of this
particular hexagon without leaving it, so a fan starting there triangulates
this shape correctly *by accident* and would not exercise
``earclip.concave_faces``'s ear-clipping path at all. Starting the list at
``(0.5, -0.5)`` instead puts the fan's first triangle across the notch: fanned
naively, it claims 1.0 square metres of area from a hexagon that measures
0.64, which is exactly the "puts a triangle outside the polygon" failure this
generator's cap exists to prove is handled --
see ``test_a_bare_fan_would_get_the_sweeps_reflex_cap_area_wrong``.
"""


def sweep(
    outline: Sequence[Sequence[float]] = SWEEP_DEFAULT_OUTLINE,
    depth: float = 1.0,
    taper: float = 1.0,
    twist: float = 0.0,
    sections: int = 1,
) -> Mesh:
    """A closed 2D ``outline`` extruded along Z -- the other family of shape a
    lathe cannot reach. Revolving a profile about an axis gives every
    *rotationally* symmetric shape; this gives everything whose cross-section
    is constant, scales, or turns along one axis instead: an L-bracket, a
    channel, an I-beam, a star, a gear blank, a picture-frame moulding, a
    keystone -- none of which any of the other thirteen primitives reach and
    none of which is a boolean of two of them.

    **``taper`` and ``twist`` are deliberately scalars, and that is the whole
    design.** The obvious alternative -- a second outline, lofting between two
    shapes -- was rejected on the same ground a variable-thickness curve sweep
    was: two array parameters, no panel affordance for either (the properties
    panel has no widget yet for even *one* array-valued parameter --
    ``lathe``'s own ``profile`` still shows as a read-only line), and a
    self-intersection between the two shapes that cannot be clamped. A
    frustum, a pedestal and a twisted column are what a loft was wanted for,
    and two numbers a human can drag reach all three: ``taper`` scales the far
    (+Z) end's outline about its own centre, and ``twist`` turns that end
    about Z, both interpolated linearly across ``sections`` stations from the
    near (-Z) end, which takes neither.

    **The outline carries positions, exactly as a lathe's ``profile`` does**,
    and :func:`_clamp_outline` re-centres it on its own bounding-box centre
    for the same reason: every other parameter in this registry is an extent,
    and an extent cannot express a position, so this generator's array
    parameter is the one place the module's "centred on the origin" rule takes
    actual work. Z is centred by construction (``-depth/2`` to ``+depth/2``).
    **``twist`` is a shape, not a placement, and is not corrected for**: a
    twisted non-symmetric outline has an off-centre *bounding box* while its
    geometry is centred -- precisely the case
    ``test_every_generator_is_centred_on_the_origin``'s own docstring already
    describes for a coarse ring, and re-centring on it here would be
    correcting for a fact about the shape rather than about its placement.

    **Winding is derived here, not borrowed from ``_side_quads``.** That
    function orders a band for a Y-axis revolve, and extruding an outline
    along Z is a different axis with its own derivation: reusing its argument
    order unchanged (the lower ring first, the upper ring second, the way
    :func:`_revolve` calls it) gives the *inward* normal for this shape --
    checked against the Newell formula and against a plain axis-aligned
    square outline, whose ``y = 0`` wall must point -Y and, built that way,
    points +Y instead. The outward order here is the ring nearer +Z first,
    the ring nearer -Z second. ``uv_sphere`` and ``capsule`` also pass the
    "later" ring first to this same function, for the opposite reason: their
    ring index runs from a pole *down*, so the later ring is the physically
    lower one there, where here it is the physically higher one -- the calls
    read the same only by coincidence. The two caps follow the same check:
    the far (+Z) cap is the outline in its own clamped (counter-clockwise)
    order, and the near (-Z) cap is that order reversed, the same "direct one
    end, reversed the other" shape ``cylinder``'s two caps already take.

    **The cap is the one face in this registry that is not convex.**
    :data:`CONCAVE_GENERATORS` names this generator for it, and the registry's
    convexity test exempts exactly that set with a replacement claim rather
    than a weaker one: ``mesh.triangulate`` ear-clips a face
    ``earclip.concave_faces`` flags instead of fanning it, so a concave cap
    still triangulates to the right *area*, even though it cannot pass the
    fan-safety check every convex face here does.

    **Self-intersection is not clamped, the same admission ``torus`` makes for
    a tube wider than its radius.** A figure-eight outline builds a
    self-intersecting solid, ``validate`` accepts it happily, and there is no
    cheap general test for "is this polygon simple" -- keeping the outline
    simple is the caller's business, the same division of labour that keeps
    ``tube`` under ``radius``.

    UV follows ``column``'s and ``lathe``'s own layout: the band across the
    top half of the square, ``u`` by **arc length around the outline's own
    perimeter** rather than by corner index -- an outline's edges are wildly
    uneven in length, and an index-parameterised ``u`` would stretch a texture
    across the bracket's short edges and squash it along its long ones --
    ``v`` across the depth; each cap unwrapped into its own bottom quadrant by
    normalising the outline's own bounding box into it with :func:`_outline_uv`,
    the same "own quadrant" rule ``cylinder``'s two circular discs already
    follow.
    """
    corners = _clamp_outline(outline)
    n = len(corners)
    m = _clamp_sections(sections)
    d = abs(float(depth))
    t = max(abs(float(taper)), MIN_TAPER)
    twist_rad = np.deg2rad(float(twist))

    xs = np.array([c[0] for c in corners], dtype="f8")
    ys = np.array([c[1] for c in corners], dtype="f8")

    rings: list[np.ndarray] = []
    for j in range(m + 1):
        frac = j / m
        z = -d * 0.5 + frac * d
        scale = 1.0 + frac * (t - 1.0)
        angle = frac * twist_rad
        cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
        rx = (xs * cos_a - ys * sin_a) * scale
        ry = (xs * sin_a + ys * cos_a) * scale
        rings.append(np.stack([rx, ry, np.full(n, z)], axis=1))
    positions = np.concatenate(rings)

    def row(j: int) -> int:
        """First vertex index of station *j*, counting from the near (-Z) end."""
        return j * n

    faces: list[list[int]] = []
    for j in range(m):
        # The far ring first, the near ring second -- see the docstring above
        # for why that is the outward order here and not ``_revolve``'s.
        faces.extend(_side_quads(row(j + 1), row(j), n))
    faces.append(list(range(row(m), row(m) + n)))  # far cap (+Z), outline order
    faces.append(list(range(row(0) + n - 1, row(0) - 1, -1)))  # near cap (-Z), reversed

    # u by arc length around the (already clamped, closed) outline, closing at
    # exactly 1 rather than wrapping to 0 -- the same per-corner reason every
    # other seam in this module closes there instead of folding back.
    pts = np.array(corners, dtype="f8")
    edge_len = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    perimeter = float(edge_len.sum())
    arc = np.concatenate([[0.0], np.cumsum(edge_len)])
    u = arc / perimeter if perimeter > 0.0 else np.linspace(0.0, 1.0, n + 1)

    uv: list[list[tuple[float, float]]] = []
    for j in range(m):
        v_lo, v_hi = 0.5 + 0.5 * (j / m), 0.5 + 0.5 * ((j + 1) / m)
        uv.extend(
            [
                (float(u[i]), v_hi),
                (float(u[i]), v_lo),
                (float(u[i + 1]), v_lo),
                (float(u[i + 1]), v_hi),
            ]
            for i in range(n)
        )
    uv.append(_outline_uv(corners, (0.01, 0.01), 0.48))
    uv.append(_outline_uv(corners, (0.51, 0.01), 0.48, reverse=True))
    return _mesh(positions, faces, uv)


TUBE_DEFAULT_PATH: tuple[tuple[float, float, float], ...] = (
    (-0.40, 0.00, 0.0),
    (-0.15, -0.12, 0.0),
    (0.15, 0.12, 0.0),
    (0.40, 0.00, 0.0),
)
"""One period of a shallow S-curve -- the centreline :func:`tube` sweeps a
circular cross-section along.

**Chosen for point symmetry about the origin, not merely for a re-centred
bounding box.** ``path`` carries positions, so :func:`_clamp_path` re-centres
its own bounding box the way :func:`_clamp_profile` and :func:`_clamp_outline`
already do for theirs -- but re-centring the *path* does not centre the
*tube* built around it. Around an arc from one flat end through an apex to
another flat end, the two ends reach no further than the path's own extremes
while the apex reaches a full ``radius`` beyond it, so the measured box comes
out asymmetric even though the path underneath it is exactly centred -- the
same class of residual :func:`test_every_generator_is_centred_on_the_origin`'s
own docstring already admits for a coarse ring, and the same line ``sweep``'s
own ``twist`` draws: correcting for it after the mesh exists would be fixing
a fact about the shape rather than about its placement. A path carried onto
itself by ``p -> -p`` sidesteps the problem instead of correcting it: every
station has a mirror station diametrically opposite the origin, built from
the same radius, so the tube around it is point-symmetric too and its
bounding box is centred by construction, not by a second pass over it.

Four points, not two straddling the middle -- a straight default would leave
:func:`_tube_frames`'s parallel-transport code untested by the registry's own
defaults sweep, the same trap :data:`SWEEP_DEFAULT_OUTLINE` avoids by being
concave rather than convex when a convex cap would have gone untested by
:data:`CONCAVE_GENERATORS`'s own gate. Sized to fit the one-metre box every
default here fits once :func:`tube`'s own default ``radius`` is added to it.
"""


def tube(
    path: Sequence[Sequence[float]] = TUBE_DEFAULT_PATH,
    radius: float = 0.1,
    sides: int = 8,
) -> Mesh:
    """A circular cross-section of ``radius``, swept along ``path`` -- the
    shape a lathe's rotational symmetry and a sweep's straight axis cannot
    reach between them: a cable, a hose, a handle, a pipe run, a bent exhaust,
    a horn, a vine, anything that *goes somewhere* rather than sitting on one
    fixed axis.

    **Not `torus`'s own ``tube`` parameter.** ``GENERATORS["torus"][0]["tube"]``
    is that shape's own tube radius, a single number; this is a generator
    name. Both exist because both are "a circular cross-section, swept" --
    one around a ring, one along an open path -- and a reader meeting either
    should be told the other exists and is a different thing entirely; see
    ``torus``'s own docstring for the same note in the other direction.

    **``radius`` is deliberately one scalar, not a per-station profile.** The
    design this replaces was a free-form variable-thickness curve sweep --
    ``path`` and a second array of per-station thickness -- and it was
    rejected on the same ground ``sweep``'s ``taper``/``twist`` already were:
    two array parameters, no panel affordance for either (the properties panel
    has no widget yet even for the one array-valued parameter every earlier
    generator here already carries), and a self-intersection between the path
    and the thickness that nothing could clamp. One array and one number is
    what survived that trade.

    **The frames are the whole difficulty, and they are parallel-transported,
    never recomputed from a fixed world "up" at each station** -- see
    :func:`_tube_frames`'s own docstring for why the naive alternative flips a
    tube inside out the moment the path's tangent passes near whatever axis
    was fixed, with nothing in ``validate`` to notice. An interior station's
    ring sits in the *bisector* plane of its two neighbouring segments, so a
    corner does not open a gap on its outside or a pinch on its inside; the
    two end stations have only one neighbouring segment each and use its
    tangent directly.

    **Winding follows :func:`_revolve`'s own argument order to
    :func:`_side_quads`, not ``sweep``'s reversed one.** That is not an
    assumption carried over -- :func:`_tube_frames` chooses each station's
    in-plane basis (``u``, ``v``) to make ``(u, v, normal)`` the same-handed
    triple :func:`_ring`'s own ``(X, Z, Y)`` already is, so "increasing
    station index, near end to far end" plays the same role here that
    :func:`_revolve`'s Y-axis stacking does, rather than the different role
    ``sweep``'s Z-axis extrusion does, which is what earned that generator's
    reversed call in the first place. The near-end cap is its ring in direct
    order, exactly as ``_revolve``'s bottom cap is; the far-end cap is that
    order reversed, exactly as its top cap is.

    **A ``radius`` wider than the path's own tightest turn self-intersects,
    and it is not clamped here** -- see :func:`_clamp_path`'s own docstring
    for why that is a different admission from the one ``torus`` makes about
    its own ``tube``.

    UV follows ``column``'s and ``lathe``'s own layout: the band across the
    top half of the square, ``v`` by arc length along the path rather than by
    station index (an authored path's segments are as uneven as a lathe
    profile's or a sweep outline's own edges), ``u`` around each ring exactly
    as :func:`cylinder`'s does. Each cap is a genuine circle in its own
    station's plane, not an arbitrary outline, so it takes :func:`_disc_uv`
    into its own quadrant exactly as ``cylinder``'s two discs do, rather than
    :func:`_outline_uv`.
    """
    points = np.array(_clamp_path(path), dtype="f8")
    n = len(points)
    k = _clamp_segments(sides)
    r = abs(float(radius))

    _, us, vs = _tube_frames(points)  # only the in-plane basis places a ring
    rings = [_tube_ring(points[i], us[i], vs[i], r, k) for i in range(n)]
    positions = np.concatenate(rings)

    def row(j: int) -> int:
        return j * k

    faces: list[list[int]] = []
    for j in range(n - 1):
        faces.extend(_side_quads(row(j), row(j + 1), k))
    faces.append(list(range(row(0), row(0) + k)))  # near cap, ring order
    last = row(n - 1)
    faces.append(list(range(last + k - 1, last - 1, -1)))  # far cap, reversed

    # v by arc length along the path, for the reason column's and lathe's own
    # v does: a path's stations are as unevenly spaced as a lathe profile's or
    # a sweep outline's own corners, and an index-parameterised v would
    # stretch a texture across a long segment and squash it across a short one.
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    along = np.concatenate([[0.0], np.cumsum(steps)])
    total = float(along[-1])
    v = 0.5 + 0.5 * (along / total if total > 0.0 else np.zeros(len(along)))

    uv: list[list[tuple[float, float]]] = []
    for j in range(n - 1):
        uv.extend(
            [
                (i / k, float(v[j])),
                (i / k, float(v[j + 1])),
                ((i + 1) / k, float(v[j + 1])),
                ((i + 1) / k, float(v[j])),
            ]
            for i in range(k)
        )
    uv.append(_disc_uv(k, (0.25, 0.25), 0.24))
    uv.append(_disc_uv(k, (0.75, 0.25), 0.24, reverse=True))
    return _mesh(positions, faces, uv)


# --- the registry ------------------------------------------------------------

GENERATORS: dict[str, tuple[dict[str, Any], Callable[..., Mesh]]] = {
    "box": ({"size": (1.0, 1.0, 1.0)}, box),
    "plane": ({"size": (1.0, 1.0)}, plane),
    "cylinder": ({"radius": 0.5, "height": 1.0, "segments": 16}, cylinder),
    "cone": ({"radius": 0.5, "height": 1.0, "segments": 16}, cone),
    "uv_sphere": ({"radius": 0.5, "segments": 16, "rings": 8}, uv_sphere),
    # ``sides`` is 16 rather than 12 for the reason ``presets.LIMB_RINGS`` is
    # 4: twelve sides puts the step around the tube at exactly 30 degrees,
    # which is ``shading.DEFAULT_ANGLE``, so a freshly placed torus came back
    # a third smooth instead of smooth throughout (measured 2026-09-06, when
    # insertion began applying the angle rule). Sixteen steps by 22.5 and
    # reads as round, for 192 more triangles.
    "torus": ({"radius": 0.35, "tube": 0.15, "segments": 24, "sides": 16}, torus),
    "grid": ({"size": (1.0, 1.0), "divisions": 4}, grid),
    "capsule": ({"radius": 0.25, "height": 0.5, "segments": 16, "rings": 4}, capsule),
    "icosphere": ({"radius": 0.5, "subdivisions": 2}, icosphere),
    "pyramid": ({"base": 1.0, "height": 1.0, "sides": 4}, pyramid),
    "arch": (
        {"width": 1.0, "height": 1.0, "depth": 0.5, "thickness": 0.25, "segments": 12},
        arch,
    ),
    "column": (
        {"radius": 0.35, "height": 2.0, "segments": 16, "base": 0.15, "capital": 0.15},
        column,
    ),
    "lathe": ({"profile": LATHE_DEFAULT_PROFILE, "segments": 16}, lathe),
    "sweep": (
        {
            "outline": SWEEP_DEFAULT_OUTLINE,
            "depth": 1.0,
            "taper": 1.0,
            "twist": 0.0,
            "sections": 1,
        },
        sweep,
    ),
    "tube": ({"path": TUBE_DEFAULT_PATH, "radius": 0.1, "sides": 8}, tube),
}
"""Name -> ``(defaults, builder)``. Every default dictionary is a complete call.

Complete because that is exactly what the properties panel does with it: splat
it into the builder to make the object the user just asked for, then bind each
key to a widget. A missing key would be a ``TypeError`` raised the first time
somebody picked that shape, which is why a test calls every entry this way.
"""

OPEN_GENERATORS: frozenset[str] = frozenset({"plane", "grid"})
"""The generators that are deliberately not closed shells.

"Open" here means the mesh has a boundary, no volume and therefore no outward
direction to derive -- ``plane`` is a single quad and ``grid`` is a subdivided
sheet, and both say so in their own docstrings. Every other generator in
:data:`GENERATORS` is a topological ball: a closed, two-manifold shell with a
consistent outward winding, and the registry's sweep tests
(``test_every_closed_generator_is_wound_outward`` and the each-directed-edge-
once check in ``tests/clay/test_primitives.py``) assert exactly that over
every generator this set does not name.

It is registry data, not test data, for the same reason :data:`CATEGORIES` is:
a thirteenth generator that is legitimately open -- a ribbon, a fan, anything
else with a boundary -- has to be able to say so where the shape is defined,
not by editing a set two directories away that the shape itself never sees.
Before this constant existed, that set was ``OPEN`` in
``tests/clay/test_primitives.py`` itself, which is the thing the tests using
it were supposed to be checking *against*.
"""

CONCAVE_GENERATORS: frozenset[str] = frozenset({"sweep"})
"""The generators whose cap is legitimately not convex.

The module's third rule -- "every face is wound counter-clockwise ... and
convex" -- does not hold for these, and the registry's convexity test
(``test_every_generators_faces_are_convex_enough_to_fan``) exempts exactly
this set with a **replacement** claim rather than dropping the check: for a
generator named here, ``mesh.triangulate``'s ``n - 2`` triangles must still
sum to the polygon's own *unsigned* area
(``test_every_concave_generators_faces_still_triangulate_to_the_right_area``),
which is the claim that ear-clipping (``clay/earclip.py``) got the shape
right even though a plain fan would not have.

It is registry data for the same reason :data:`OPEN_GENERATORS` is: a fact
about what a generator's own cap looks like belongs beside the generator, not
in a set two directories away that the shape itself never sees. Gated
**bidirectionally**, exactly as ``OPEN_GENERATORS`` is: every name here must
be a real generator (``test_concave_generators_names_only_real_generators``),
and every generator named here must really have a reflex corner at its own
defaults (``test_concave_generators_really_have_a_concave_face_at_defaults``)
-- an exemption nobody needs any more (a shape's default changed, or it was
never actually concave) should fail a test rather than linger on this set
forever, silently weakening the convexity claim for a generator that no
longer needs weakening.

``sweep``'s cap is an arbitrary user-supplied polygon and an L-bracket, a
channel or a picture-frame moulding is reflex *by definition* -- unlike every
other generator here, whose own shape is a parametrised curve or a fixed
silhouette with no way to author a reflex corner into it at all.
"""

CATEGORIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("primitives", ("box", "plane", "grid", "cylinder", "cone",
                    "uv_sphere", "icosphere", "torus", "capsule")),
    ("structures", ("pyramid", "arch", "column", "lathe", "sweep", "tube")),
)
"""The add panel's sections, in the order they are drawn.

The registry says what a shape *is*; this says where it appears, which is a
different question and one ``GENERATORS`` cannot answer -- a dict has one order
and the panel wants two headings. It lives here rather than in the pane for the
same reason the defaults do: a sixth structure should be one line in this file
and no edit to the pane at all.

It is a **partition**, asserted as one: every generator in exactly one section
and no name that is not a generator, so a primitive added to the registry and
forgotten here fails a test rather than quietly vanishing from the panel that
is generated from it.
"""
