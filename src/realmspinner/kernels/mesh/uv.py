"""Texture coordinates for authored geometry: projection, not solving.

``Mesh.uv`` has been a per-corner field since the CSR layout was written and
nothing has ever filled it in -- every generator produced ``None`` and every op
carried that ``None`` faithfully forward. This is the half that produces one,
and it is deliberately the *cheap* half: a planar projection per face, chosen by
which axis the face points along most.

**Not LSCM.** A conformal solver is the right answer for an organic mesh and
the wrong shape for a modelling panel: it needs seams the user has drawn and it
fails in ways ("the solve did not converge") a panel has little to say about. A
box projection is predictable, instant, has no failure mode, and is exactly
right for the blockout geometry Clay produces -- which is why it is the only
unwrap Clay offers.

**Islands may overlap, and that is what a cube projection is.** Faces pointing
+X and -X land in the same square, because the projection is by axis *pair*
rather than by direction -- which is what Blender's Cube Projection does and
what a user asking for a box unwrap is asking for. A layout with no overlap at
all is a packing problem, and this module does not solve it: the cylinder, cone, UV sphere,
torus and capsule lay themselves out without overlap in their own generators (the
icosphere does not: its lat/long layout leaves 42 of 320 faces overlapping), and the
flat-panelled ones -- the box included, whose generator calls :func:`box_unwrap` -- keep the
shared squares. (The 2026-10-07 audit's clay-52: this paragraph used to say a
box's six faces are laid out in a cross by its own generator; it has not been
since the box began calling the cube projection.)

**v runs down the image (glTF's convention), the one convention ``Mesh.uv`` has.**
Row 0 of a texture is v = 0, which is what the viewport's sampler, the GLB
writer, the NEAREST export and the OBJ importer/exporter already read, so a wall
whose top edge is at the top of the model has the *smaller* v there. The
2026-10-07 audit's clay-03: this module and the generators in
:mod:`.primitives` wrote v as model height (v up), so a picture painted
top-red in Inker rendered top-blue on every box wall. A face seen from outside
and reading upright in a v-down image therefore has a *negative* signed uv area
in plain (u, v) axes; that sign is the convention, not a mirrored face.

Everything here is pure: numpy and :mod:`~.mesh`, nothing from ``service``,
``queue`` or ``studio``, so every rule about where a corner lands is assertable
from a hand-built mesh.
"""

from __future__ import annotations

import numpy as np

from .mesh import Mesh, face_count, face_normals

__all__ = ["box_unwrap"]

# The two axes each dominant normal projects onto, and which of them is flipped
# so the result is not mirrored when seen from outside the face. Written out
# rather than derived from a cross product: the table *is* the convention, and
# deriving it would put the sign of every island at the mercy of an argument
# order nobody would think to test.
#
# Index is the dominant axis (0 = X, 1 = Y, 2 = Z); the entry is the pair of
# axes ``(u, v)`` the face's positions are read from. A face whose normal points
# the *negative* way has its u mirrored within the object's own footprint, which
# is what keeps lettering on the far side of a box the right way round.
_AXES: tuple[tuple[int, int], ...] = (
    # clay-11: the right-handed cyclic pairs, so every face reads the same way
    # round seen from outside (the old (Z,Y) and (X,Z) pairs mirrored every X and
    # Y face); the negative way flips u below. ``v`` is written as ``1 - height``
    # (see the module docstring), which turns that handedness's signed uv area
    # negative for every face -- the sign of an upright, unmirrored v-down face.
    # clay-03, found by its fixer and closed 2026-10-08: the X walls used to read
    # u from Y and v from Z, which stands a picture on its side (Y is up, so Z is
    # a horizontal there). v has to come from Y; u from Z then runs the wrong
    # handedness for that axis alone, so X is the one axis whose *positive* side
    # is flipped (``_FLIPPED_WHEN`` below).
    (2, 1),  # X-facing: u from Z, v from Y
    (2, 0),  # Y-facing: u from Z, v from X
    (0, 1),  # Z-facing: u from X, v from Y
)

#: The sign of the normal along the dominant axis that has its u mirrored: the
#: negative side for Y and Z (their (u, v) pairs are right-handed about the axis),
#: the positive side for X (see ``_AXES``).
_FLIPPED_WHEN: tuple[float, ...] = (1.0, -1.0, -1.0)


def dominant_axis(normals: np.ndarray) -> np.ndarray:
    """Which axis each normal points along most. ``(F,)`` of 0, 1 or 2."""
    return np.argmax(np.abs(np.asarray(normals, dtype="f8")), axis=1).astype("i4")


def box_unwrap(mesh: Mesh) -> Mesh:
    """Project each face along whichever axis its normal is closest to.

    The result is normalised over the mesh's own bounding box, so a long thin
    object is not stretched into a square: one unit of UV is the *largest*
    extent of the mesh, and the shorter axes use proportionally less of the
    square. Scaling each island to fill the square independently is the obvious
    alternative and is wrong -- it makes texel density depend on how big a face
    happens to be, so a checker map comes out with different sized squares on
    every panel of one object.
    """
    if face_count(mesh) == 0:
        return mesh
    # Computed once and handed down. ``_projected`` needs the same normals to
    # decide which corners are mirrored, and asking for them again is a second
    # full pass over every loop in the mesh for an answer already in hand.
    normals = face_normals(mesh)
    return _projected(mesh, dominant_axis(normals), normals=normals)


def _projected(mesh: Mesh, axes: np.ndarray, *, normals: np.ndarray | None = None) -> Mesh:
    """One planar projection per face, into a shared normalised square.

    ``normals`` is an optimisation and never a second opinion: a caller that has
    already measured them passes them in, and one that has not leaves it None
    and gets the same array measured here.
    """
    positions = np.asarray(mesh.positions, dtype="f8")
    if len(positions) == 0:
        return mesh
    # The 2026-10-07 audit's clay-49: the extent was taken over *every* position,
    # but a GLB import keeps vertices no face references, and one such vertex
    # 100 units out shrank every referenced face's layout into a sliver of the
    # square. Only the vertices a face corner names lay anything out, so only
    # they bound it.
    referenced = positions[np.asarray(mesh.loops, dtype="i8")]
    if len(referenced) == 0:
        return mesh
    low = referenced.min(axis=0)
    span = float(np.max(referenced.max(axis=0) - low))
    # A flat or single-point mesh has no extent to normalise by. One is the
    # scale that leaves the coordinates as they are, which keeps a plane at
    # y = 0 unwrapping sensibly rather than dividing by zero.
    scale = 1.0 / span if span > 1e-12 else 1.0

    counts = np.diff(np.asarray(mesh.starts, dtype="i8"))
    face_of = np.repeat(np.arange(face_count(mesh), dtype="i8"), counts)
    corner_axis = axes[face_of]
    corner_pos = (positions[np.asarray(mesh.loops, dtype="i8")] - low) * scale

    extent = (referenced.max(axis=0) - low) * scale
    uv = np.zeros((len(mesh.loops), 2), dtype="f4")
    if normals is None:
        normals = face_normals(mesh)
    for axis in range(3):
        which = np.flatnonzero(corner_axis == axis)
        if not len(which):
            continue
        u_axis, v_axis = _AXES[axis]
        u = corner_pos[which, u_axis]
        # Mirrored within the object's own footprint rather than within the
        # unit square: the coordinates are normalised by the *largest* extent,
        # so a short axis does not reach 1 and ``1 - u`` would push that face's
        # island off its own geometry. The flip is per corner, taken from the
        # sign of its own face's normal, because opposite faces share the
        # square and without it one of the pair reads mirrored.
        facing = normals[face_of[which], axis] * _FLIPPED_WHEN[axis] > 0.0
        uv[which, 0] = np.where(facing, extent[u_axis] - u, u)
        # v down the image: the top of the model is the small v (clay-03).
        uv[which, 1] = 1.0 - corner_pos[which, v_axis]

    # The 2026-10-10 audit's clay-42: ``replace`` always built a new ``Mesh``, so
    # unwrapping an object whose layout was already this one still looked like a
    # change to ``set_mesh``'s identity check and pushed a no-op undo step. The
    # same mesh back says "nothing moved".
    if mesh.uv is not None and np.array_equal(mesh.uv, uv):
        return mesh

    from dataclasses import replace

    return replace(mesh, uv=uv)
