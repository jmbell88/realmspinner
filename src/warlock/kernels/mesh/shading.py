"""The angle rule that decides which faces of a mesh render smooth.

Extracted out of ``clay_ops._shade_auto`` (the 2026-09-06 audit, the organic-
shapes decision): the manual "Shade Auto..." op and the two insertion doors
that now apply this automatically -- ``panes/clay_tools.add_primitive`` for a
shape off the grid and ``panes/clay_tools.add_assembly`` for a figure's parts
-- need the identical rule, and a rule copied into three call sites is a rule
that drifts the first time one of them is edited without the other two.
``clay_ops._shade_auto`` now delegates here; see its own (much shorter)
docstring for the object-selection plumbing this module has no opinion about.

**The bar for this extraction is byte identity.** Nothing about the maths
below changed when it moved -- ``tests/clay/test_shading.py`` asserts this
function against the pre-extraction inline computation on a sphere, a
cylinder and a box, and every one of ``_shade_auto``'s own pre-existing
callers keeps working unchanged because the wrapper still returns the same
skip-if-nothing-changed answer it always did.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from . import mesh as bm
from .adjacency import adjacency
from .mesh import Mesh

DEFAULT_ANGLE = 30.0
"""The angle every caller gets unless it names its own.

One constant rather than two literals: before this extraction, ``30.0`` was
written once as ``_shade_auto``'s own default and again as the ``Param``'s
default in ``clay_ops.py``'s registration of the "Shade Auto..." op, and
nothing tied the two together. Both now read this name.
"""


def auto_smooth(mesh: Mesh, angle: float = DEFAULT_ANGLE) -> Mesh:
    """Smooth every face whose *every* neighbour agrees with it to within *angle*.

    Per face rather than per edge, because ``smooth`` is a per-face flag and
    there is nowhere to record "smooth along this edge only" -- so a face is
    smooth only when it has no sharp edge at all. That is a real limitation and
    it is stated here because the result surprises people: **a capped cylinder
    comes out entirely flat**, since every side quad meets a cap at a right
    angle.

    That is also the *correct* answer for this renderer rather than a gap in
    the rule. A smooth face takes accumulated vertex normals, so smoothing the
    band while the caps stay flat would average the cap normals into the rim
    and round the very edge the caps are there to define. Blender avoids this
    with per-edge split normals, which is a different mesh format.

    What it does do well is exactly what it should: a sphere or a torus goes
    smooth throughout, a box stays flat, and a mesh that mixes the two gets the
    right answer per region. The measurement is the cosine between adjacent
    face normals -- the same question ``glbimport`` asks of an imported mesh's
    supplied normals -- and a boundary edge has no neighbour to disagree with,
    so it makes nothing sharp.

    Returns *mesh* itself, unchanged, when the rule leaves every face's flag
    exactly where it already was -- which lets a caller apply this
    unconditionally at every insertion door without paying for a new ``Mesh``
    object (and a fresh cache miss on it) for the shapes the rule leaves flat.
    """
    faces = bm.face_count(mesh)
    if faces == 0:
        return mesh
    limit = float(np.cos(np.radians(max(0.0, min(180.0, float(angle))))))
    # Normalised: ``face_normals`` returns Newell normals, whose length is
    # proportional to face area -- a dot product of two of those is not a
    # cosine, and comparing it against one silently called every pair sharp.
    normals = np.asarray(bm.face_normals(mesh), dtype="f8")
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 1e-12)
    counts = np.diff(np.asarray(mesh.starts, dtype="i8"))
    face_of = np.repeat(np.arange(faces, dtype="i8"), counts)
    adj = adjacency(mesh)
    twin = np.asarray(adj.twin, dtype="i8")

    paired = np.flatnonzero(twin >= 0)
    smooth = np.ones(faces, dtype=bool)
    if len(paired):
        left, right = face_of[paired], face_of[twin[paired]]
        sharp = np.einsum("ij,ij->i", normals[left], normals[right]) < limit
        smooth[left[sharp]] = False
        smooth[right[sharp]] = False
    # The 2026-09-07 audit's clay-03: a flipped-normal pair -- two corners
    # sharing an edge that wind the *same* direction, which is what leaves
    # ``twin`` at -1 for both rather than pointing at each other -- is a 180
    # degree disagreement, not merely an untested one. Before this, ``twin >=
    # 0`` skipped these corners entirely and both faces came out smooth across
    # a seam the winding itself says is broken. Unconditional, not gated on
    # ``limit``: there is no angle at which two faces facing opposite ways
    # should blend, and ``adjacency.py``'s own docstring says an imported GLB
    # routinely has exactly this.
    flipped = np.asarray(adj.flipped_pairs, dtype="i8")
    if len(flipped):
        left, right = face_of[flipped[:, 0]], face_of[flipped[:, 1]]
        smooth[left] = False
        smooth[right] = False
    # The 2026-09-11 audit's clay-05: ``twin >= 0`` is also -1 for a
    # non-manifold edge (edge_uses >= 3, three or more faces meeting there),
    # not only for a boundary edge -- and unlike a boundary edge, a
    # non-manifold one has neighbours that can genuinely disagree with it.
    # Left ungated, every face on such an edge came out smooth regardless of
    # angle, which is wrong in the opposite direction of the boundary case:
    # a boundary has nothing to compare against, but a non-manifold edge has
    # too much to reduce to one twin, so -- as with the flipped-pair branch
    # above -- it is treated as unconditionally sharp rather than silently
    # skipped. ``adjacency.py``'s own docstring says real-world GLB import
    # routinely produces these.
    corner_edge = np.asarray(adj.corner_edge, dtype="i8")
    edge_uses = np.asarray(adj.edge_uses, dtype="i8")
    nonmanifold = np.flatnonzero(edge_uses[corner_edge] >= 3)
    if len(nonmanifold):
        smooth[face_of[nonmanifold]] = False
    if np.array_equal(smooth, mesh.smooth):
        return mesh
    return replace(mesh, smooth=smooth)
