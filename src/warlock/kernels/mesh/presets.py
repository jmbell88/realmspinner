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
head/tail pairs below are a hard-coded copy and ``tests/modes/clay/test_presets.py``
cross-checks every one of them against the real template -- a template edited
without editing this file is a red test rather than a body that has quietly
drifted off its skeleton. It also reaches for its siblings ``primitives``,
``mesh`` and ``ops`` (the last for :func:`~.ops.align_y`, promoted out of
this module), and for ``viewer.math3d`` -- outward only as far as
``tests/modes/clay/test_clay_imports.py`` already lets the rest of the package go --
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

from ..geom3d import math3d as m3
from .mesh import bounds, transformed
from .ops import align_y
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


def _placed(head: Vec, tail: Vec) -> tuple[Vec, tuple[float, float, float, float], float]:
    """Midpoint, rotation and length of a bone, all in Clay space.

    The rotation is :func:`~.ops.align_y` -- promoted out of this module (it
    used to live here as ``_align_y``) into ``clay/ops.py`` beside
    :func:`~.ops.place_between`, its second caller: this is "point this bone
    along that direction" and ``place_between`` is "point this object along
    that line", the same quaternion for the same reason in both places.
    """
    a = np.array(_to_clay(head), dtype="f8")
    b = np.array(_to_clay(tail), dtype="f8")
    mid = (a + b) * 0.5
    return (
        (float(mid[0]), float(mid[1]), float(mid[2])),
        align_y(b - a),
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


def _mass(
    name: str,
    bone: str | None,
    head: Vec,
    tail: Vec,
    radius: float,
    *,
    width: float = 1.0,
    depth: float = 1.0,
    along: float = 1.0,
) -> Part:
    """A body mass: an ellipsoidal sphere sized by anatomy, not by bone length.

    ``_capsule`` derives its cylindrical section from the bone it spans, and
    ``max(length - 2*radius, MIN_CAPSULE_SECTION)`` means a bone shorter than
    twice its own radius collapses to a plain sphere -- not an approximation of
    one, an exact sphere. The 2026-09-06 audit's diagnosis: a humanoid torso's
    hips/spine/chest, a quadruped's whole body, and a serpent's spine chain are
    all short bones wearing fat radii, so all three collapse the same way, and
    a column of near-equal spheres is exactly what "reads as beads" measures.

    A body mass -- a pelvis, a ribcage, a barrel -- was never sized by the bone
    inside it in real anatomy; it is broader than it is deep, and its bulk
    overlaps its neighbour rather than sitting flush beside it. So this places
    a plain ``uv_sphere`` at the bone's own midpoint -- the same landmark
    :func:`_capsule` uses, so the landmark test cannot tell the difference --
    and stretches it non-uniformly in its own local frame. ``math3d.compose``
    applies ``scale`` before the bone's rotation, so the stretch is always
    relative to the bone, never to world space.

    For a bone that stands roughly along world +Y (a torso), local X is across
    the body, Y is along the bone (hip to head), and Z is front-to-back -- the
    axis a side view measures. For a bone that runs roughly horizontal (a
    quadruped's spine, a serpent's chain), the same alignment rotation that
    carries canonical +Y onto the bone direction also carries local Z onto
    world *vertical* instead of front-to-back -- so there ``depth`` shapes
    dorsoventral thickness, which is the anatomically right axis for a barrel
    chest or a snake's body to vary independently of its width. Either way the
    three knobs stay "across / along / the third axis"; only which world axis
    "the third axis" lands on changes with the bone's own orientation.
    """
    translation, rotation, _length = _placed(head, tail)
    return Part(
        name=name,
        bone=bone,
        generator="uv_sphere",
        params={"radius": radius, "segments": 16, "rings": 8},
        translation=translation,
        rotation=rotation,
        scale=(width, along, depth),
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
    """Nineteen parts on the nineteen-bone biped: torso, arms, legs, head.

    Hips, spine and chest are :func:`_mass` ellipsoids rather than capsules.
    All three bones are shorter than twice their own old radius (hips 0.07
    long at r=0.10, spine 0.12 at r=0.11, chest 0.11 at r=0.12), so under
    ``_capsule`` every one of them collapsed to a near-equal sphere -- the
    2026-09-06 audit's diagnosis for the user's "reads as overlapping beads"
    complaint. Broadening each mass laterally, flattening it front-to-back and
    sizing the three unequally (0.078/0.072/0.085) reads as pelvis / waist /
    ribcage instead of three balls on a spine; see
    ``test_a_humanoid_torso_is_one_form_rather_than_three_stacked_balls`` for
    the measurement. Limb radii are up from their pre-2026-09-06 values for the
    same complaint -- ball torsos this much broader than a 0.035-0.055 limb
    read as wire -- and the taper stays monotonic shoulder > upper arm >
    forearm > hand, thigh > shin > foot, the way a real limb narrows.
    """
    return (
        _mass(
            "Hips", "hips", (0.00, 0.00, 0.53), (0.00, 0.00, 0.60), 0.078, width=1.30, depth=0.92
        ),
        _mass(
            "Spine",
            "spine",
            (0.00, 0.00, 0.60),
            (0.00, 0.00, 0.72),
            0.072,
            width=1.28,
            depth=0.86,
            along=1.15,
        ),
        _mass(
            "Chest",
            "chest",
            (0.00, 0.00, 0.72),
            (0.00, 0.00, 0.83),
            0.085,
            width=1.42,
            depth=0.88,
            along=1.15,
        ),
        _capsule("Neck", "neck", (0.00, 0.00, 0.83), (0.00, 0.00, 0.90), 0.040),
        _sphere("Head", "head", (0.00, 0.00, 0.90), (0.00, 0.00, 1.00), 0.090),
        _capsule("Shoulder.L", "shoulder.L", (0.03, 0.00, 0.82), (0.10, 0.00, 0.81), 0.058),
        _capsule("Upper arm.L", "upper_arm.L", (0.10, 0.00, 0.81), (0.22, 0.00, 0.66), 0.050),
        _capsule("Forearm.L", "forearm.L", (0.22, 0.00, 0.66), (0.32, 0.00, 0.52), 0.040),
        _box("Hand.L", "hand.L", (0.32, 0.00, 0.52), (0.36, 0.00, 0.47), 0.055, 0.032),
        _capsule("Shoulder.R", "shoulder.R", (-0.03, 0.00, 0.82), (-0.10, 0.00, 0.81), 0.058),
        _capsule("Upper arm.R", "upper_arm.R", (-0.10, 0.00, 0.81), (-0.22, 0.00, 0.66), 0.050),
        _capsule("Forearm.R", "forearm.R", (-0.22, 0.00, 0.66), (-0.32, 0.00, 0.52), 0.040),
        _box("Hand.R", "hand.R", (-0.32, 0.00, 0.52), (-0.36, 0.00, 0.47), 0.055, 0.032),
        _capsule("Thigh.L", "thigh.L", (0.07, 0.00, 0.53), (0.07, 0.00, 0.29), 0.070),
        _capsule("Shin.L", "shin.L", (0.07, 0.00, 0.29), (0.07, 0.00, 0.06), 0.052),
        _box("Foot.L", "foot.L", (0.07, 0.00, 0.06), (0.07, -0.10, 0.00), 0.08, 0.055),
        _capsule("Thigh.R", "thigh.R", (-0.07, 0.00, 0.53), (-0.07, 0.00, 0.29), 0.070),
        _capsule("Shin.R", "shin.R", (-0.07, 0.00, 0.29), (-0.07, 0.00, 0.06), 0.052),
        _box("Foot.R", "foot.R", (-0.07, 0.00, 0.06), (-0.07, -0.10, 0.00), 0.08, 0.055),
    )


def quadruped() -> tuple[Part, ...]:
    """Nineteen parts: a barrel along ``-Y``, four legs under it, a short tail.

    The body runs along the template's Y rather than its Z, so the torso
    capsules come out horizontal after the swap -- which is the cheapest visible
    proof the axis conversion is the right way round.

    Hips, spine and chest are :func:`_mass` ellipsoids for the same reason
    they are on ``humanoid``: all three bones are shorter than twice their old
    capsule radius (hips 0.25 at r=0.14, spine 0.25 at r=0.15, chest 0.15 at
    r=0.15 -- the last pair *identical*, so spine and chest used to be the same
    sphere twice), so the barrel was three near-equal balls in a trenchcoat.
    Here the alignment rotation for a near-horizontal bone carries local Z onto
    world vertical rather than front-to-back (see :func:`_mass`'s own
    docstring), so ``depth`` below shapes the ribcage's dorsoventral thickness
    -- deep through the chest, pinched at the waist -- while ``width`` still
    shapes the animal's left-right spread and ``along`` overlaps each mass
    heavily into its neighbour so the barrel reads as one tapered form: broad
    chest, tucked waist, rounded rump, the way a running quadruped's silhouette
    actually breaks.
    """
    return (
        _mass(
            "Hips",
            "hips",
            (0.00, 0.35, 0.70),
            (0.00, 0.10, 0.72),
            0.115,
            width=1.25,
            depth=1.05,
            along=1.35,
        ),
        _mass(
            "Spine",
            "spine",
            (0.00, 0.10, 0.72),
            (0.00, -0.15, 0.72),
            0.095,
            width=1.05,
            depth=0.95,
            along=1.70,
        ),
        _mass(
            "Chest",
            "chest",
            (0.00, -0.15, 0.72),
            (0.00, -0.30, 0.74),
            0.135,
            width=1.35,
            depth=1.35,
            along=1.45,
        ),
        _capsule("Neck", "neck", (0.00, -0.30, 0.74), (0.00, -0.42, 0.90), 0.070),
        _sphere("Head", "head", (0.00, -0.42, 0.90), (0.00, -0.50, 1.00), 0.085),
        _capsule("Tail 01", "tail_01", (0.00, 0.35, 0.70), (0.00, 0.45, 0.62), 0.035),
        _capsule("Tail 02", "tail_02", (0.00, 0.45, 0.62), (0.00, 0.50, 0.50), 0.025),
        _capsule("Front upper.L", "front_upper.L", (0.12, -0.22, 0.68), (0.12, -0.20, 0.42), 0.058),
        _capsule("Front lower.L", "front_lower.L", (0.12, -0.20, 0.42), (0.12, -0.24, 0.14), 0.044),
        _box("Front foot.L", "front_foot.L", (0.12, -0.24, 0.14), (0.12, -0.32, 0.00), 0.07, 0.06),
        _capsule(
            "Front upper.R", "front_upper.R", (-0.12, -0.22, 0.68), (-0.12, -0.20, 0.42), 0.058
        ),
        _capsule(
            "Front lower.R", "front_lower.R", (-0.12, -0.20, 0.42), (-0.12, -0.24, 0.14), 0.044
        ),
        _box(
            "Front foot.R", "front_foot.R", (-0.12, -0.24, 0.14), (-0.12, -0.32, 0.00), 0.07, 0.06
        ),
        _capsule("Rear upper.L", "rear_upper.L", (0.12, 0.32, 0.68), (0.12, 0.38, 0.44), 0.065),
        _capsule("Rear lower.L", "rear_lower.L", (0.12, 0.38, 0.44), (0.12, 0.28, 0.16), 0.048),
        _box("Rear foot.L", "rear_foot.L", (0.12, 0.28, 0.16), (0.12, 0.34, 0.00), 0.07, 0.06),
        _capsule("Rear upper.R", "rear_upper.R", (-0.12, 0.32, 0.68), (-0.12, 0.38, 0.44), 0.065),
        _capsule("Rear lower.R", "rear_lower.R", (-0.12, 0.38, 0.44), (-0.12, 0.28, 0.16), 0.048),
        _box("Rear foot.R", "rear_foot.R", (-0.12, 0.28, 0.16), (-0.12, 0.34, 0.00), 0.07, 0.06),
    )


def bird() -> tuple[Part, ...]:
    """Twenty parts on the twenty-bone winged biped.

    The wings are flat plates rather than capsules -- a wing that is round in
    section is a wing that has to be flattened before it is a wing at all, and
    the box is one scale drag from a feather.

    Hips, spine and chest are the same fix as ``humanoid``'s and
    ``quadruped``'s: all three bones here are shorter than twice their old
    capsule radius too (hips 0.099 at r=0.10, spine 0.120 at r=0.10, chest
    0.106 at r=0.095), so the body was the same three-bead column under
    feathers. The bone here runs diagonally -- a hunched, forward-leaning
    perch posture -- rather than along a single world axis, but :func:`_mass`
    does not care: the ellipsoid stretches along whatever direction the bone
    itself points.
    """
    return (
        _mass(
            "Hips",
            "hips",
            (0.00, 0.05, 0.48),
            (0.00, -0.02, 0.55),
            0.095,
            width=1.20,
            depth=0.95,
            along=1.15,
        ),
        _mass(
            "Spine",
            "spine",
            (0.00, -0.02, 0.55),
            (0.00, -0.10, 0.64),
            0.092,
            width=1.15,
            depth=0.92,
            along=1.20,
        ),
        _mass(
            "Chest",
            "chest",
            (0.00, -0.10, 0.64),
            (0.00, -0.17, 0.72),
            0.088,
            width=1.30,
            depth=1.05,
            along=1.20,
        ),
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

    ``Tail 01`` is a :func:`_mass` ellipsoid rather than a capsule: at its old
    radius (0.055) it was already past half its own bone's length (0.103), so
    it collapsed to a sphere sitting at the tail's own root -- a single bead
    right where the tail meets the now much bigger hips. The other four
    segments are real capsules already (each comfortably longer than twice its
    own radius) and keep tapering as they did.
    """
    taper = (
        _mass(
            "Tail 01",
            "tail_01",
            (0.00, 0.03, 0.53),
            (0.00, 0.12, 0.48),
            0.050,
            width=1.0,
            depth=0.95,
            along=1.40,
        ),
        _capsule("Tail 02", "tail_02", (0.00, 0.12, 0.48), (0.00, 0.21, 0.42), 0.046),
        _capsule("Tail 03", "tail_03", (0.00, 0.21, 0.42), (0.00, 0.30, 0.35), 0.037),
        _capsule("Tail 04", "tail_04", (0.00, 0.30, 0.35), (0.00, 0.39, 0.28), 0.028),
        _capsule("Tail 05", "tail_05", (0.00, 0.39, 0.28), (0.00, 0.48, 0.22), 0.019),
    )
    return humanoid() + taper


def serpent() -> tuple[Part, ...]:
    """Ten parts: a chain of masses swelling at the middle and tapering out.

    ``Spine 01`` through ``Spine 06`` are :func:`_mass` ellipsoids, not
    capsules: each of those six bones is 0.10 template units long under a
    radius from 0.052 to 0.080, so ``2 * radius`` exceeds the bone every time
    and all six collapsed to near-equal spheres under ``_capsule`` -- ten
    beads was the user's literal complaint about this figure. ``along=1.55``
    stretches each one enough to overlap its neighbour by more than a bone's
    width without ballooning the taper the radii already carry; ``width`` and
    ``depth`` stay close to round (a snake's cross-section) but slightly
    flattened, since a perfectly circular tube reads as a stack of rings under
    depth shading the same way a stack of spheres does. ``Spine 07`` and
    ``Tail tip`` are left as capsules -- their radii (0.038, 0.022) are
    already under half their bone's length, so they were real tapered
    capsules before this fix and stay that way.
    """
    return (
        _mass(
            "Spine 01", "spine_01", (0.00, -0.34, 0.30), (0.00, -0.24, 0.30), 0.070,
            width=1.05, depth=0.92, along=1.55,
        ),
        _mass(
            "Spine 02", "spine_02", (0.00, -0.24, 0.30), (0.00, -0.14, 0.30), 0.078,
            width=1.05, depth=0.92, along=1.55,
        ),
        _mass(
            "Spine 03", "spine_03", (0.00, -0.14, 0.30), (0.00, -0.04, 0.30), 0.080,
            width=1.05, depth=0.92, along=1.55,
        ),
        _mass(
            "Spine 04", "spine_04", (0.00, -0.04, 0.30), (0.00, 0.06, 0.30), 0.075,
            width=1.05, depth=0.92, along=1.55,
        ),
        _mass(
            "Spine 05", "spine_05", (0.00, 0.06, 0.30), (0.00, 0.16, 0.30), 0.065,
            width=1.05, depth=0.92, along=1.55,
        ),
        _mass(
            "Spine 06", "spine_06", (0.00, 0.16, 0.30), (0.00, 0.26, 0.30), 0.052,
            width=1.05, depth=0.92, along=1.55,
        ),
        _capsule("Spine 07", "spine_07", (0.00, 0.26, 0.30), (0.00, 0.36, 0.30), 0.038),
        _capsule("Tail tip", "tail_tip", (0.00, 0.36, 0.30), (0.00, 0.48, 0.30), 0.022),
        _capsule("Neck", "neck", (0.00, -0.34, 0.30), (0.00, -0.42, 0.36), 0.060),
        _sphere("Head", "head", (0.00, -0.42, 0.36), (0.00, -0.50, 0.40), 0.065),
    )


def fish() -> tuple[Part, ...]:
    """Thirteen parts: a tapering body, three fins on the midline, two pairs.

    Every fin is a box -- a fin is a flat sheet, and a capsule would have to be
    squashed before it was one. The jaw is a box for the same reason.

    ``Spine 01`` through ``Spine 03`` are :func:`_mass` ellipsoids for the
    same reason as ``serpent``'s chain: each of those bones is 0.16 long under
    a radius from 0.075 to 0.110, so ``2 * radius`` meets or exceeds the bone
    and all three came back at or against ``MIN_CAPSULE_SECTION`` -- three
    near-spherical beads just behind the head. ``depth`` here is a fish's
    dorsoventral thickness rather than its left-right width (the same
    horizontal-bone rotation ``quadruped``'s barrel goes through), and it is
    set above ``width`` because most fish are laterally compressed -- taller
    through the body than they are wide. ``Spine 04`` stays a capsule: at
    radius 0.045 against the same 0.16 bone it was already a real tapered
    section, the fish's usual reach toward the tail fin.
    """
    return (
        _mass(
            "Spine 01", "spine_01", (0.00, -0.30, 0.50), (0.00, -0.14, 0.50), 0.100,
            width=0.85, depth=1.15, along=1.40,
        ),
        _mass(
            "Spine 02", "spine_02", (0.00, -0.14, 0.50), (0.00, 0.02, 0.50), 0.092,
            width=0.85, depth=1.10, along=1.40,
        ),
        _mass(
            "Spine 03", "spine_03", (0.00, 0.02, 0.50), (0.00, 0.18, 0.50), 0.070,
            width=0.85, depth=1.05, along=1.30,
        ),
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
    same way ``studio/modes/clay/ui/tools.py`` builds the object the user ends up with.
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
    downstream, in ``modes/clay/ui/panes/tools.add_assembly``, which calls
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
    ``tests/modes/clay/test_shading.py`` for the measurement.
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
    # 2026-09-06 audit, finding docs-14: the label read "Insect / spider
    # (six-legged)", promising an eight-legged spider the template does not
    # build -- insect.json rigs a six-legged insect. Whether a genuine spider
    # template is wanted stays a separate, open decision.
    "insect": ("Insect", insect),
    "fish": ("Fish (swimmer)", fish),
    "blob": ("Blob (amorphous)", blob),
}
"""Template key -> ``(label, builder)``, one entry per rig template.

The keys are the *template* keys, which ``templates.templates`` takes from each
file's stem and enforces against the key inside it -- so ``bird`` rather than
``winged``, and a new skeleton is a body here or a failing test. The labels are
each template's own ``label`` field, copied for the same reason the landmarks
are copied and cross-checked the same way.
"""
