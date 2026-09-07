"""Figure presets: a bag of loose primitives roughed out on a rig template.

Eight assemblies, one per skeleton in ``warlock/templates/``, so the Figures
section of Clay's add panel and the rig catalogue are the same list -- no
template ships a skeleton with nothing to drive, and no body ships without a
skeleton that fits it. Each builder returns a tuple of :class:`Part`, and a Part
is a *complete generator call plus a placement*: the pane splats ``params`` into
``GENERATORS[generator][1]`` and drops the result in at ``translation`` /
``rotation`` / ``scale``. Nothing is welded, nothing is parented; what comes out
is a selection of ordinary objects the user then edits, moves and booleans like
any other. That is the point -- a preset here is a *head start*, not a rig.

``ASSEMBLIES`` maps a key to ``(label, builder)`` and deliberately not to
``(defaults, builder)`` the way :data:`~.primitives.GENERATORS` does. There is
no assembly object left after placement to hold a parameter on, and a
placement-time parameter would need a modal this UI does not have; resizing is
the scale gizmo on the multi-selection the placement leaves behind.

**Two spaces, and the swap is the whole trap.** Rig templates are authored in
Blender's frame -- ``+X`` is the subject's left, ``-Y`` is forward, ``+Z`` is
up, with ``z`` running 0 at the feet to 1 at the crown. Clay, like everything
else downstream of it, is glTF: **Y up**, right-handed. So every landmark read
off a template has to go through :func:`_to_clay` -- ``(x, y, z) -> (x, z, -y)``
-- which is the same axis conversion a glTF exporter applies on the way out of
Blender. Getting it wrong does not crash anything: it produces a figure lying on
its face, which looks like a modelling mistake rather than an axis one.

**The landmarks are copied here, not read.** This module imports ``numpy`` at
its core, because the whole clay package's claim is that it is assertable
headlessly; reaching into ``warlock/templates`` for a JSON file would buy an
outward dependency (and a file-system read) for a handful of numbers. So the
head/tail pairs below are a hard-coded copy and ``tests/clay/test_presets.py``
cross-checks every one of them against the real template -- a template edited
without editing this file is a red test rather than a body that has quietly
drifted off its skeleton. It also reaches for its two siblings, ``primitives``
and ``mesh``, and for ``viewer.math3d`` -- outward only as far as
``tests/clay/test_clay_imports.py`` already lets the rest of the package go --
because grounding (below) has to build each part's *real* mesh and place it in
world space rather than trust a bone midpoint to say where the geometry ends.

The *thicknesses* could not be read from the template even if we wanted to: a
skeleton gives joint landmarks and says nothing about how fat a limb is. They
are art direction, and they live here as constants beside the segment they
belong to.

**Grounding, and why two assemblies are exempt from it.** The 2026-09-06
audit's clay-08 finding: the eight assembly builders below followed three
different, undocumented conventions for where they sat relative to Y=0 --
measured, six sank into the grid by between 0.0009 and 0.1300 and two floated
above it, and nothing said which was intended. The six that sink
(``humanoid``, ``biped_tail``, ``quadruped``, ``bird``, ``insect``, ``blob``)
all have a part *reaching* for the ground and missing it by a small authoring
error -- a foot box modelled a hair short, a lobe roughed out approximately.
The two that float (``serpent``, ``fish``) have no ground-reaching part at
all: a serpent has no feet to miss the floor with, and both are authored
suspended on purpose -- the serpent's whole spine chain sits at a constant
template ``z = 0.30`` and the fish's at ``z = 0.50``. That is a placement
decision, not a miss, so it is left alone. :func:`build` is the door this
distinction is applied behind: every assembly not in :data:`SWIMMERS` comes
back with its lowest built vertex at exactly Y=0, computed from the real
generated geometry rather than eight hand-tuned offset constants that a preset
edited later would have to remember to re-measure.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ..viewer import math3d as m3
from .mesh import bounds, transformed
from .primitives import GENERATORS

# The smallest cylindrical section a capsule may be left with. A bone shorter
# than twice its own authored radius -- a neck, a finger-length hand -- would
# otherwise ask for a negative height, and ``capsule`` takes the magnitude of
# that, which would silently make the part *longer* the shorter the bone got.
MIN_CAPSULE_SECTION = 0.01

# Enough sides to read as round on a limb, and no more: an assembly is twenty
# parts at once, and the default sixteen would put a third again as many
# triangles into the scene for a difference nobody sees on a forearm. That
# argument still holds for ``LIMB_SEGMENTS``, around the tube, where the step
# is 360/12 = 30 degrees between faces that share a *long* edge.
LIMB_SEGMENTS = 12

# Four, not three, and the reason is arithmetic rather than taste. A capsule's
# hemisphere divides 90 degrees by its ring count, so ``rings=3`` steps by
# exactly 30 -- precisely ``shading.DEFAULT_ANGLE``, the threshold insertion
# smooths against. Landing *on* a threshold is not a margin: quad-normal
# blending tips enough of those bands fractionally past it that a limb came
# back 33% smooth, and a figure went on reading as a string of beads after the
# 2026-09-06 decision that organic shapes insert smooth-shaded was supposed to
# fix exactly that. Four rings steps by 22.5 and clears it -- measured, the
# same limb goes to 93% smooth for 48 more triangles. The silhouette does not
# move: this is tessellation density, not proportion.
LIMB_RINGS = 4


@dataclass(frozen=True, slots=True)
class Part:
    """One primitive in an assembly: what to build, and where to put it.

    ``bone`` is the template landmark the part was roughed out on, or ``None``
    for a part with no bone behind it. It is carried rather than discarded
    because it is the only thing that ties a loose box back to the skeleton it
    was shaped for -- the test cross-checks the position against it, and a
    future "fit this body to that rig" would have nothing else to go on.
    """

    name: str
    bone: str | None
    generator: str
    params: dict[str, Any] = field(default_factory=dict)
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)


Vec = tuple[float, float, float]


def _to_clay(p: Vec) -> Vec:
    """Blender Z-up (template space) -> glTF Y-up (Clay space).

    ``(x, y, z) -> (x, z, -y)``. Up becomes ``+Y``; Blender's forward ``-Y``
    becomes ``+Z``, which is the direction a glTF camera looks *from*, so a
    figure built this way faces the default view. This is the one place the
    conversion happens.
    """
    return (float(p[0]), float(p[2]), -float(p[1]))


def _align_y(direction: np.ndarray) -> tuple[float, float, float, float]:
    """The XYZW quaternion taking ``+Y`` onto *direction*.

    Every generator here is built along ``+Y`` -- that is the module rule in
    ``primitives`` -- so a bone that is not vertical is a rotation, never a
    re-authored mesh. The two degenerate cases are written out because the
    cross product vanishes for both and normalising it would divide by zero:
    parallel is the identity, and antiparallel is a half turn about ``X``,
    picked arbitrarily since every axis perpendicular to ``Y`` would do.
    """
    length = float(np.linalg.norm(direction))
    if length < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    d = np.asarray(direction, dtype="f8") / length
    dot = float(d[1])
    if dot > 1.0 - 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    if dot < -1.0 + 1e-9:
        return (1.0, 0.0, 0.0, 0.0)
    axis = np.cross(np.array([0.0, 1.0, 0.0]), d)
    s = float(np.sqrt((1.0 + dot) * 2.0))
    q = np.array([axis[0] / s, axis[1] / s, axis[2] / s, s * 0.5])
    q /= float(np.linalg.norm(q))
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def _placed(head: Vec, tail: Vec) -> tuple[Vec, tuple[float, float, float, float], float]:
    """Midpoint, rotation and length of a bone, all in Clay space."""
    a = np.array(_to_clay(head), dtype="f8")
    b = np.array(_to_clay(tail), dtype="f8")
    mid = (a + b) * 0.5
    return (
        (float(mid[0]), float(mid[1]), float(mid[2])),
        _align_y(b - a),
        float(np.linalg.norm(b - a)),
    )


def _capsule(name: str, bone: str, head: Vec, tail: Vec, radius: float) -> Part:
    """A limb: a capsule spanning the bone, ``radius`` thick."""
    translation, rotation, length = _placed(head, tail)
    return Part(
        name=name,
        bone=bone,
        generator="capsule",
        params={
            "radius": radius,
            "height": max(length - 2.0 * radius, MIN_CAPSULE_SECTION),
            "segments": LIMB_SEGMENTS,
            "rings": LIMB_RINGS,
        },
        translation=translation,
        rotation=rotation,
    )


def _box(name: str, bone: str, head: Vec, tail: Vec, width: float, depth: float) -> Part:
    """A hand, a foot, a beak: a box spanning the bone along its own Y."""
    translation, rotation, length = _placed(head, tail)
    return Part(
        name=name,
        bone=bone,
        generator="box",
        params={"size": (width, max(length, MIN_CAPSULE_SECTION), depth)},
        translation=translation,
        rotation=rotation,
    )


def _sphere(name: str, bone: str, head: Vec, tail: Vec, radius: float) -> Part:
    """A head or a lump: a UV sphere at the bone's midpoint.

    No rotation -- a sphere has no orientation to get wrong, and leaving it at
    identity keeps the numeric panel readable for the part a user is most
    likely to open it on.
    """
    translation, _, _ = _placed(head, tail)
    return Part(
        name=name,
        bone=bone,
        generator="uv_sphere",
        params={"radius": radius, "segments": 16, "rings": 8},
        translation=translation,
    )


def _ico(name: str, bone: str, head: Vec, tail: Vec, radius: float) -> Part:
    """An even-triangled ball, for the shapes that are going to be sculpted."""
    translation, _, _ = _placed(head, tail)
    return Part(
        name=name,
        bone=bone,
        generator="icosphere",
        params={"radius": radius, "subdivisions": 2},
        translation=translation,
    )


# --- the assemblies ----------------------------------------------------------
#
# Each builder is written in the template's own bone order so the two read side
# by side. The head/tail pairs are the copy the module docstring describes.


def humanoid() -> tuple[Part, ...]:
    """Nineteen parts on the nineteen-bone biped: torso, arms, legs, head."""
    return (
        _capsule("Hips", "hips", (0.00, 0.00, 0.53), (0.00, 0.00, 0.60), 0.10),
        _capsule("Spine", "spine", (0.00, 0.00, 0.60), (0.00, 0.00, 0.72), 0.11),
        _capsule("Chest", "chest", (0.00, 0.00, 0.72), (0.00, 0.00, 0.83), 0.12),
        _capsule("Neck", "neck", (0.00, 0.00, 0.83), (0.00, 0.00, 0.90), 0.040),
        _sphere("Head", "head", (0.00, 0.00, 0.90), (0.00, 0.00, 1.00), 0.090),
        _capsule("Shoulder.L", "shoulder.L", (0.03, 0.00, 0.82), (0.10, 0.00, 0.81), 0.050),
        _capsule("Upper arm.L", "upper_arm.L", (0.10, 0.00, 0.81), (0.22, 0.00, 0.66), 0.042),
        _capsule("Forearm.L", "forearm.L", (0.22, 0.00, 0.66), (0.32, 0.00, 0.52), 0.035),
        _box("Hand.L", "hand.L", (0.32, 0.00, 0.52), (0.36, 0.00, 0.47), 0.05, 0.03),
        _capsule("Shoulder.R", "shoulder.R", (-0.03, 0.00, 0.82), (-0.10, 0.00, 0.81), 0.050),
        _capsule("Upper arm.R", "upper_arm.R", (-0.10, 0.00, 0.81), (-0.22, 0.00, 0.66), 0.042),
        _capsule("Forearm.R", "forearm.R", (-0.22, 0.00, 0.66), (-0.32, 0.00, 0.52), 0.035),
        _box("Hand.R", "hand.R", (-0.32, 0.00, 0.52), (-0.36, 0.00, 0.47), 0.05, 0.03),
        _capsule("Thigh.L", "thigh.L", (0.07, 0.00, 0.53), (0.07, 0.00, 0.29), 0.055),
        _capsule("Shin.L", "shin.L", (0.07, 0.00, 0.29), (0.07, 0.00, 0.06), 0.045),
        _box("Foot.L", "foot.L", (0.07, 0.00, 0.06), (0.07, -0.10, 0.00), 0.07, 0.05),
        _capsule("Thigh.R", "thigh.R", (-0.07, 0.00, 0.53), (-0.07, 0.00, 0.29), 0.055),
        _capsule("Shin.R", "shin.R", (-0.07, 0.00, 0.29), (-0.07, 0.00, 0.06), 0.045),
        _box("Foot.R", "foot.R", (-0.07, 0.00, 0.06), (-0.07, -0.10, 0.00), 0.07, 0.05),
    )


def quadruped() -> tuple[Part, ...]:
    """Nineteen parts: a barrel along ``-Y``, four legs under it, a short tail.

    The body runs along the template's Y rather than its Z, so the torso
    capsules come out horizontal after the swap -- which is the cheapest visible
    proof the axis conversion is the right way round.
    """
    return (
        _capsule("Hips", "hips", (0.00, 0.35, 0.70), (0.00, 0.10, 0.72), 0.14),
        _capsule("Spine", "spine", (0.00, 0.10, 0.72), (0.00, -0.15, 0.72), 0.15),
        _capsule("Chest", "chest", (0.00, -0.15, 0.72), (0.00, -0.30, 0.74), 0.15),
        _capsule("Neck", "neck", (0.00, -0.30, 0.74), (0.00, -0.42, 0.90), 0.070),
        _sphere("Head", "head", (0.00, -0.42, 0.90), (0.00, -0.50, 1.00), 0.085),
        _capsule("Tail 01", "tail_01", (0.00, 0.35, 0.70), (0.00, 0.45, 0.62), 0.035),
        _capsule("Tail 02", "tail_02", (0.00, 0.45, 0.62), (0.00, 0.50, 0.50), 0.025),
        _capsule("Front upper.L", "front_upper.L", (0.12, -0.22, 0.68), (0.12, -0.20, 0.42), 0.050),
        _capsule("Front lower.L", "front_lower.L", (0.12, -0.20, 0.42), (0.12, -0.24, 0.14), 0.038),
        _box("Front foot.L", "front_foot.L", (0.12, -0.24, 0.14), (0.12, -0.32, 0.00), 0.06, 0.05),
        _capsule(
            "Front upper.R", "front_upper.R", (-0.12, -0.22, 0.68), (-0.12, -0.20, 0.42), 0.050
        ),
        _capsule(
            "Front lower.R", "front_lower.R", (-0.12, -0.20, 0.42), (-0.12, -0.24, 0.14), 0.038
        ),
        _box(
            "Front foot.R", "front_foot.R", (-0.12, -0.24, 0.14), (-0.12, -0.32, 0.00), 0.06, 0.05
        ),
        _capsule("Rear upper.L", "rear_upper.L", (0.12, 0.32, 0.68), (0.12, 0.38, 0.44), 0.055),
        _capsule("Rear lower.L", "rear_lower.L", (0.12, 0.38, 0.44), (0.12, 0.28, 0.16), 0.040),
        _box("Rear foot.L", "rear_foot.L", (0.12, 0.28, 0.16), (0.12, 0.34, 0.00), 0.06, 0.05),
        _capsule("Rear upper.R", "rear_upper.R", (-0.12, 0.32, 0.68), (-0.12, 0.38, 0.44), 0.055),
        _capsule("Rear lower.R", "rear_lower.R", (-0.12, 0.38, 0.44), (-0.12, 0.28, 0.16), 0.040),
        _box("Rear foot.R", "rear_foot.R", (-0.12, 0.28, 0.16), (-0.12, 0.34, 0.00), 0.06, 0.05),
    )


def bird() -> tuple[Part, ...]:
    """Twenty parts on the twenty-bone winged biped.

    The wings are flat plates rather than capsules -- a wing that is round in
    section is a wing that has to be flattened before it is a wing at all, and
    the box is one scale drag from a feather.
    """
    return (
        _capsule("Hips", "hips", (0.00, 0.05, 0.48), (0.00, -0.02, 0.55), 0.10),
        _capsule("Spine", "spine", (0.00, -0.02, 0.55), (0.00, -0.10, 0.64), 0.10),
        _capsule("Chest", "chest", (0.00, -0.10, 0.64), (0.00, -0.17, 0.72), 0.095),
        _capsule("Neck", "neck", (0.00, -0.17, 0.72), (0.00, -0.24, 0.84), 0.045),
        _sphere("Head", "head", (0.00, -0.24, 0.84), (0.00, -0.34, 0.92), 0.065),
        _box("Beak", "beak", (0.00, -0.34, 0.92), (0.00, -0.46, 0.90), 0.04, 0.04),
        _capsule("Tail", "tail", (0.00, 0.05, 0.48), (0.00, 0.30, 0.44), 0.055),
        _box("Tail tip", "tail_tip", (0.00, 0.30, 0.44), (0.00, 0.48, 0.40), 0.14, 0.02),
        _box("Wing base.L", "wing_base.L", (0.04, -0.08, 0.68), (0.18, -0.04, 0.70), 0.09, 0.05),
        _box("Wing mid.L", "wing_mid.L", (0.18, -0.04, 0.70), (0.34, 0.00, 0.68), 0.09, 0.03),
        _box("Wing tip.L", "wing_tip.L", (0.34, 0.00, 0.68), (0.48, 0.06, 0.64), 0.07, 0.02),
        _box("Wing base.R", "wing_base.R", (-0.04, -0.08, 0.68), (-0.18, -0.04, 0.70), 0.09, 0.05),
        _box("Wing mid.R", "wing_mid.R", (-0.18, -0.04, 0.70), (-0.34, 0.00, 0.68), 0.09, 0.03),
        _box("Wing tip.R", "wing_tip.R", (-0.34, 0.00, 0.68), (-0.48, 0.06, 0.64), 0.07, 0.02),
        _capsule("Thigh.L", "thigh.L", (0.07, 0.02, 0.46), (0.08, 0.00, 0.28), 0.040),
        _capsule("Shin.L", "shin.L", (0.08, 0.00, 0.28), (0.08, -0.02, 0.10), 0.025),
        _box("Foot.L", "foot.L", (0.08, -0.02, 0.10), (0.08, -0.16, 0.00), 0.05, 0.03),
        _capsule("Thigh.R", "thigh.R", (-0.07, 0.02, 0.46), (-0.08, 0.00, 0.28), 0.040),
        _capsule("Shin.R", "shin.R", (-0.08, 0.00, 0.28), (-0.08, -0.02, 0.10), 0.025),
        _box("Foot.R", "foot.R", (-0.08, -0.02, 0.10), (-0.08, -0.16, 0.00), 0.05, 0.03),
    )


def biped_tail() -> tuple[Part, ...]:
    """The humanoid plus a five-segment tapering tail: twenty-four parts.

    The first nineteen are ``humanoid``'s own -- the template's are identical
    bone for bone -- so they are reused rather than copied, which is also the
    only way the two bodies stay the same body when one of them is retuned.
    """
    taper = (
        _capsule("Tail 01", "tail_01", (0.00, 0.03, 0.53), (0.00, 0.12, 0.48), 0.055),
        _capsule("Tail 02", "tail_02", (0.00, 0.12, 0.48), (0.00, 0.21, 0.42), 0.046),
        _capsule("Tail 03", "tail_03", (0.00, 0.21, 0.42), (0.00, 0.30, 0.35), 0.037),
        _capsule("Tail 04", "tail_04", (0.00, 0.30, 0.35), (0.00, 0.39, 0.28), 0.028),
        _capsule("Tail 05", "tail_05", (0.00, 0.39, 0.28), (0.00, 0.48, 0.22), 0.019),
    )
    return humanoid() + taper


def serpent() -> tuple[Part, ...]:
    """Ten parts: a chain of capsules swelling at the middle and tapering out."""
    return (
        _capsule("Spine 01", "spine_01", (0.00, -0.34, 0.30), (0.00, -0.24, 0.30), 0.070),
        _capsule("Spine 02", "spine_02", (0.00, -0.24, 0.30), (0.00, -0.14, 0.30), 0.078),
        _capsule("Spine 03", "spine_03", (0.00, -0.14, 0.30), (0.00, -0.04, 0.30), 0.080),
        _capsule("Spine 04", "spine_04", (0.00, -0.04, 0.30), (0.00, 0.06, 0.30), 0.075),
        _capsule("Spine 05", "spine_05", (0.00, 0.06, 0.30), (0.00, 0.16, 0.30), 0.065),
        _capsule("Spine 06", "spine_06", (0.00, 0.16, 0.30), (0.00, 0.26, 0.30), 0.052),
        _capsule("Spine 07", "spine_07", (0.00, 0.26, 0.30), (0.00, 0.36, 0.30), 0.038),
        _capsule("Tail tip", "tail_tip", (0.00, 0.36, 0.30), (0.00, 0.48, 0.30), 0.022),
        _capsule("Neck", "neck", (0.00, -0.34, 0.30), (0.00, -0.42, 0.36), 0.060),
        _sphere("Head", "head", (0.00, -0.42, 0.36), (0.00, -0.50, 0.40), 0.065),
    )


def fish() -> tuple[Part, ...]:
    """Thirteen parts: a tapering body, three fins on the midline, two pairs.

    Every fin is a box -- a fin is a flat sheet, and a capsule would have to be
    squashed before it was one. The jaw is a box for the same reason.
    """
    return (
        _capsule("Spine 01", "spine_01", (0.00, -0.30, 0.50), (0.00, -0.14, 0.50), 0.110),
        _capsule("Spine 02", "spine_02", (0.00, -0.14, 0.50), (0.00, 0.02, 0.50), 0.100),
        _capsule("Spine 03", "spine_03", (0.00, 0.02, 0.50), (0.00, 0.18, 0.50), 0.075),
        _capsule("Spine 04", "spine_04", (0.00, 0.18, 0.50), (0.00, 0.34, 0.50), 0.045),
        _box("Tail fin", "tail_fin", (0.00, 0.34, 0.50), (0.00, 0.48, 0.50), 0.02, 0.22),
        _sphere("Head", "head", (0.00, -0.30, 0.50), (0.00, -0.46, 0.50), 0.100),
        _box("Jaw", "jaw", (0.00, -0.42, 0.46), (0.00, -0.48, 0.42), 0.08, 0.04),
        _box("Dorsal fin", "dorsal", (0.00, 0.00, 0.62), (0.00, 0.04, 0.84), 0.02, 0.16),
        _box("Ventral fin", "ventral", (0.00, 0.10, 0.38), (0.00, 0.14, 0.18), 0.02, 0.14),
        _box("Pectoral fin.L", "pectoral.L", (0.05, -0.16, 0.44), (0.26, -0.06, 0.38), 0.10, 0.02),
        _box("Pelvic fin.L", "pelvic.L", (0.04, 0.02, 0.40), (0.20, 0.10, 0.32), 0.08, 0.02),
        _box(
            "Pectoral fin.R", "pectoral.R", (-0.05, -0.16, 0.44), (-0.26, -0.06, 0.38), 0.10, 0.02
        ),
        _box("Pelvic fin.R", "pelvic.R", (-0.04, 0.02, 0.40), (-0.20, 0.10, 0.32), 0.08, 0.02),
    )


def insect() -> tuple[Part, ...]:
    """Seventeen parts: thorax, head, abdomen, two mandibles and six legs."""
    return (
        _capsule("Thorax", "thorax", (0.00, 0.00, 0.60), (0.00, -0.16, 0.62), 0.080),
        _sphere("Head", "head", (0.00, -0.16, 0.62), (0.00, -0.34, 0.62), 0.075),
        _sphere("Abdomen", "abdomen", (0.00, 0.00, 0.60), (0.00, 0.30, 0.58), 0.120),
        _capsule("Mandible.L", "mandible.L", (0.04, -0.32, 0.60), (0.10, -0.46, 0.56), 0.018),
        _capsule("Mandible.R", "mandible.R", (-0.04, -0.32, 0.60), (-0.10, -0.46, 0.56), 0.018),
        _capsule(
            "Leg A upper.L", "leg_a_upper.L", (0.06, -0.10, 0.58), (0.26, -0.22, 0.42), 0.022
        ),
        _capsule(
            "Leg A lower.L", "leg_a_lower.L", (0.26, -0.22, 0.42), (0.40, -0.28, 0.00), 0.016
        ),
        _capsule("Leg B upper.L", "leg_b_upper.L", (0.06, 0.00, 0.58), (0.28, 0.00, 0.42), 0.022),
        _capsule("Leg B lower.L", "leg_b_lower.L", (0.28, 0.00, 0.42), (0.44, 0.02, 0.00), 0.016),
        _capsule("Leg C upper.L", "leg_c_upper.L", (0.06, 0.10, 0.58), (0.26, 0.22, 0.42), 0.022),
        _capsule("Leg C lower.L", "leg_c_lower.L", (0.26, 0.22, 0.42), (0.40, 0.30, 0.00), 0.016),
        _capsule(
            "Leg A upper.R", "leg_a_upper.R", (-0.06, -0.10, 0.58), (-0.26, -0.22, 0.42), 0.022
        ),
        _capsule(
            "Leg A lower.R", "leg_a_lower.R", (-0.26, -0.22, 0.42), (-0.40, -0.28, 0.00), 0.016
        ),
        _capsule(
            "Leg B upper.R", "leg_b_upper.R", (-0.06, 0.00, 0.58), (-0.28, 0.00, 0.42), 0.022
        ),
        _capsule(
            "Leg B lower.R", "leg_b_lower.R", (-0.28, 0.00, 0.42), (-0.44, 0.02, 0.00), 0.016
        ),
        _capsule(
            "Leg C upper.R", "leg_c_upper.R", (-0.06, 0.10, 0.58), (-0.26, 0.22, 0.42), 0.022
        ),
        _capsule(
            "Leg C lower.R", "leg_c_lower.R", (-0.26, 0.22, 0.42), (-0.40, 0.30, 0.00), 0.016
        ),
    )


def blob() -> tuple[Part, ...]:
    """Eight icospheres: a vertical column of four and four lobes on the equator.

    Icospheres rather than UV spheres throughout, and that is the one shape
    decision in this file that is not about silhouette: a blob is the assembly
    most likely to be *sculpted* rather than assembled, and a UV sphere's pole
    fans are the wrong triangles to push around.
    """
    return (
        _ico("Base", "base", (0.00, 0.00, 0.00), (0.00, 0.00, 0.22), 0.240),
        _ico("Core", "core", (0.00, 0.00, 0.22), (0.00, 0.00, 0.52), 0.280),
        _ico("Crown", "crown", (0.00, 0.00, 0.52), (0.00, 0.00, 0.80), 0.220),
        _ico("Top", "top", (0.00, 0.00, 0.80), (0.00, 0.00, 1.00), 0.130),
        _ico("Lobe front", "lobe_front", (0.00, -0.10, 0.30), (0.00, -0.42, 0.40), 0.120),
        _ico("Lobe back", "lobe_back", (0.00, 0.10, 0.30), (0.00, 0.42, 0.40), 0.120),
        _ico("Lobe.L", "lobe.L", (0.10, 0.00, 0.30), (0.44, 0.00, 0.40), 0.120),
        _ico("Lobe.R", "lobe.R", (-0.10, 0.00, 0.30), (-0.44, 0.00, 0.40), 0.120),
    )


SWIMMERS: frozenset[str] = frozenset({"serpent", "fish"})
"""Assembly keys :func:`build` must never ground.

The 2026-09-06 audit's clay-08 finding drew this line: the six other figures
each have a part *reaching* for the ground and missing it by a small
authoring error (measured between 0.0009 and 0.1300 of the figure's own
height), where these two have no ground-reaching part at all and are
authored suspended on purpose -- the serpent's whole spine chain sits at a
constant template ``z = 0.30`` and the fish's at ``z = 0.50``. That is a
deliberate placement, not a miss, so it must not move. A key absent from this
set is terrestrial and :func:`build` grounds it; the point of naming the two
exceptions rather than the six defaults is that a *ninth* figure -- terrestrial
or not -- gets a rule to satisfy instead of a coin toss.
"""


def _world_min_y(part: Part) -> float:
    """The lowest world-space Y the built mesh for *part* actually reaches.

    Built from the real generator output, not the bone landmark: a landmark is
    a skeleton joint, and the geometry roughed out on it -- a foot box, a
    lobe's radius -- can end short of or past the ground the joint sits at.
    The 2026-09-06 audit's clay-08 finding is exactly that gap going
    unmeasured. ``GENERATORS[part.generator]`` is a ``(defaults, builder)``
    pair, so the defaults are splatted under the part's own params first, the
    same way ``panes/clay_tools.py`` builds the object the user ends up with.
    """
    defaults, make = GENERATORS[part.generator]
    local = make(**{**defaults, **part.params})
    matrix = m3.compose(
        np.asarray(part.translation, dtype="f8"),
        np.asarray(part.rotation, dtype="f8"),
        np.asarray(part.scale, dtype="f8"),
    )
    lo, _hi = bounds(transformed(local, matrix))
    return float(lo[1])


def _grounded(parts: tuple[Part, ...]) -> tuple[Part, ...]:
    """*parts*, shifted as one rigid body so the assembly's lowest vertex is Y=0.

    One shift for the whole assembly, not one measurement per part: a figure
    is a single body standing (or not) on the ground, and grounding a limb
    independently of its neighbours would pull the assembly apart at every
    joint it is measured on. Deriving the shift from the built meshes rather
    than hand-tuning eight offset constants is the 2026-09-06 audit's clay-08
    fix -- a preset edited later stays grounded without anyone remembering to
    re-measure it.
    """
    drop = min(_world_min_y(part) for part in parts)
    if drop == 0.0:
        return parts
    return tuple(
        replace(
            part,
            translation=(
                part.translation[0],
                part.translation[1] - drop,
                part.translation[2],
            ),
        )
        for part in parts
    )


def build(key: str) -> tuple[Part, ...]:
    """The one door: *key*'s assembly, grounded unless it is a swimmer.

    A caller gets a figure that already satisfies the grounding rule without
    knowing what the rule is or which of the eight keys is exempt from it --
    the 2026-09-06 audit's clay-08 finding was exactly that no single place
    stated the rule at all. A key this module has never seen (a test fixture,
    a future preset added straight to :data:`ASSEMBLIES`) is not in
    :data:`SWIMMERS` and is therefore grounded like any other terrestrial
    figure, which is the point of naming the exceptions rather than the
    default.

    A ``Part`` carries a generator name and a params dict, not a built
    ``Mesh`` -- so this function is not where a part's shading is decided,
    the same way it is not where a part's geometry is built. That happens once,
    downstream, in ``panes/clay_tools.add_assembly``, which calls
    :func:`GENERATORS`'s builder for each part exactly as it always did and
    now also runs the result through ``clay.shading.auto_smooth`` before the
    object is placed -- the same rule and the same unconditional application
    ``add_primitive`` uses for a lone shape off the grid, so a humanoid's box
    hands and feet get hard edges without this module or that one needing a
    list of which parts are "organic". :data:`LIMB_RINGS` is 4 rather than 3
    because of that rule and not independently of it: three rings put a
    hemisphere's latitude step at exactly the threshold and limbs came back a
    third smooth, so the figures went on reading as beads after the change
    meant to stop them doing so -- see that constant's own comment and
    ``tests/clay/test_shading.py`` for the measurement.
    """
    _label, builder = ASSEMBLIES[key]
    parts = builder()
    if key in SWIMMERS:
        return parts
    return _grounded(parts)


ASSEMBLIES: dict[str, tuple[str, Callable[[], tuple[Part, ...]]]] = {
    "humanoid": ("Humanoid (biped)", humanoid),
    "biped_tail": ("Biped with tail", biped_tail),
    "quadruped": ("Quadruped", quadruped),
    "bird": ("Bird (winged biped)", bird),
    "serpent": ("Serpent (limbless chain)", serpent),
    "insect": ("Insect / spider (six-legged)", insect),
    "fish": ("Fish (swimmer)", fish),
    "blob": ("Blob (amorphous)", blob),
}
"""Template key -> ``(label, builder)``, one entry per rig template.

The keys are the *template* keys, which ``rigging.templates`` takes from each
file's stem and enforces against the key inside it -- so ``bird`` rather than
``winged``, and a new skeleton is a body here or a failing test. The labels are
each template's own ``label`` field, copied for the same reason the landmarks
are copied and cross-checked the same way.
"""
