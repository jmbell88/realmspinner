"""The figure presets are a copy of the rig templates, and this is what keeps
them one.

``studio/clay/presets.py`` is pure -- numpy and nothing else -- so the labels
and the landmarks it is built on are hard-coded there rather than read out of
``warlock/templates/``. That is only safe while something compares the two, and
this file is that something: the *test* may import ``warlock.rigging``, and it
fails the moment a template is renamed, a bone is renamed, or a part drifts off
the joint it was roughed out on.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.rigging import templates
from warlock.studio.clay import presets
from warlock.studio.clay.mesh import bounds, transformed, validate
from warlock.studio.clay.primitives import GENERATORS
from warlock.studio.viewer import math3d as m3

#: How far a part's centre may sit from its bone's midpoint, in the templates'
#: normalised units (the figure is one unit tall). Every part here is placed
#: *at* the midpoint, so the honest tolerance is float noise -- but a hard
#: ``1e-9`` would refuse the first deliberate art-direction nudge, and the thing
#: actually worth catching is a part wired to the wrong landmark or built
#: without the Z-up -> Y-up swap. ``0.03`` is chosen against that: the closest
#: two bone midpoints in any of the eight templates are 0.076 apart (``chest``
#: and ``shoulder.L``, in humanoid and biped_tail), and 0.03 is the round number
#: below half of that -- the largest bound that still cannot be met by the wrong
#: bone. ``test_the_tolerance_cannot_match_two_bones`` asserts that separation
#: rather than asking you to believe it, so a part inside tolerance of its bone
#: is inside tolerance of *no other bone*, and the check cannot pass by accident.
TOLERANCE = 0.03


def _midpoints(key: str) -> dict[str, np.ndarray]:
    """Every bone's midpoint in Clay space, via the module's own swap."""
    out = {}
    for bone in templates()[key].bones:
        head = np.array(presets._to_clay(bone["head"]), dtype="f8")
        tail = np.array(presets._to_clay(bone["tail"]), dtype="f8")
        out[bone["name"]] = (head + tail) * 0.5
    return out


ASSEMBLY_KEYS = sorted(presets.ASSEMBLIES)


def test_there_is_a_body_for_every_skeleton():
    """The Figures section and the rig catalogue are the same list.

    A template with no assembly is a skeleton nothing feeds; an assembly with no
    template is a body that can never be posed. Either is a half-built feature,
    and neither is visible from inside one of the two files.
    """
    assert set(presets.ASSEMBLIES) == set(templates())


@pytest.mark.parametrize("key", ASSEMBLY_KEYS)
def test_the_label_is_the_templates_own(key: str):
    """One name for one archetype. Renaming a template reddens this rather than
    leaving the add panel calling it something the rig catalogue does not."""
    assert presets.ASSEMBLIES[key][0] == templates()[key].label


@pytest.mark.parametrize("key", ASSEMBLY_KEYS)
def test_every_part_names_a_real_generator(key: str):
    for part in presets.ASSEMBLIES[key][1]():
        assert part.generator in GENERATORS, f"{key}/{part.name}: {part.generator}"


@pytest.mark.parametrize("key", ASSEMBLY_KEYS)
def test_every_part_is_a_complete_call(key: str):
    """The same claim ``test_primitives`` makes about the registry's defaults,
    made about these calls instead -- and made by *building* the mesh, because a
    parameter that exists but is nonsense (a negative height, a two-sided
    polygon) is a call that type-checks and then produces a mesh the viewer
    cannot draw. An incomplete call would be a ``TypeError`` raised the first
    time somebody placed the figure."""
    for part in presets.ASSEMBLIES[key][1]():
        builder = GENERATORS[part.generator][1]
        mesh = builder(**part.params)
        validate(mesh)
        assert len(mesh.positions) > 0, f"{key}/{part.name} built nothing"


@pytest.mark.parametrize("key", ASSEMBLY_KEYS)
def test_part_names_are_unique(key: str):
    """The outliner lists them by name and Clay's mirror selection pairs them by
    it, so two parts called the same thing is two rows a user cannot tell
    apart."""
    names = [p.name for p in presets.ASSEMBLIES[key][1]()]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("key", ASSEMBLY_KEYS)
def test_sided_parts_come_in_pairs(key: str):
    """Blender's ``.L``/``.R`` convention, which is what Clay's mirror selection
    keys on -- a lone ``.L`` is a limb the mirror tools silently skip."""
    names = {p.name for p in presets.ASSEMBLIES[key][1]()}
    for name in names:
        if name.endswith(".L"):
            assert name[:-2] + ".R" in names, f"{key}: {name} has no right side"
        if name.endswith(".R"):
            assert name[:-2] + ".L" in names, f"{key}: {name} has no left side"


@pytest.mark.parametrize("key", ASSEMBLY_KEYS)
def test_every_bone_named_by_a_part_exists(key: str):
    """One direction only, deliberately: a part per bone is the rule, but a bone
    that is a pure pivot -- a beak tip, a tail tip -- may legitimately carry no
    geometry, so the converse is not asserted."""
    known = set(_midpoints(key))
    for part in presets.ASSEMBLIES[key][1]():
        assert part.bone is None or part.bone in known, f"{key}/{part.name}: {part.bone}"


@pytest.mark.parametrize("key", ASSEMBLY_KEYS)
def test_every_part_sits_on_its_landmark(key: str):
    """The axis test, wearing a placement test's clothes.

    Templates are Blender Z-up and Clay is glTF Y-up. Forgetting the swap does
    not raise: it builds a figure lying on its face, with every part still in a
    plausible-looking place. Comparing against the landmark *after* the swap is
    what turns that into a number.
    """
    mids = _midpoints(key)
    for part in presets.ASSEMBLIES[key][1]():
        if part.bone is None:
            continue
        offset = float(np.linalg.norm(np.asarray(part.translation) - mids[part.bone]))
        assert offset <= TOLERANCE, f"{key}/{part.name} is {offset:.3f} from {part.bone}"


@pytest.mark.parametrize("key", ASSEMBLY_KEYS)
def test_the_tolerance_cannot_match_two_bones(key: str):
    """What makes the test above mean anything.

    If two bones sat within ``2 * TOLERANCE`` of each other, a part could pass
    while wired to the wrong one. So the separation is asserted rather than
    assumed, in every template -- and a future template with two coincident
    landmarks reddens *here*, where the reason is written down, instead of
    quietly weakening the check next door.
    """
    mids = list(_midpoints(key).items())
    for i, (a_name, a) in enumerate(mids):
        for b_name, b in mids[i + 1 :]:
            gap = float(np.linalg.norm(a - b))
            assert gap > 2.0 * TOLERANCE, f"{key}: {a_name} and {b_name} are {gap:.3f} apart"


@pytest.mark.parametrize("key", ASSEMBLY_KEYS)
def test_the_landmark_check_can_fail(key: str):
    """The regression test's own regression test.

    A placement assertion that cannot fail is decoration. Nudging one part by
    twice the tolerance -- roughly what dropping the axis swap does to a limb --
    must put it outside the bound.
    """
    mids = _midpoints(key)
    part = next(p for p in presets.ASSEMBLIES[key][1]() if p.bone is not None)
    moved = np.asarray(part.translation) + np.array([0.0, 2.0 * TOLERANCE, 0.0])
    assert float(np.linalg.norm(moved - mids[part.bone])) > TOLERANCE


@pytest.mark.parametrize("key", ASSEMBLY_KEYS)
def test_rotations_are_unit_quaternions(key: str):
    """XYZW, normalised, because that is what ``viewer.math3d`` expects and a
    quaternion that is merely *nearly* unit scales the part it rotates."""
    for part in presets.ASSEMBLIES[key][1]():
        assert abs(float(np.linalg.norm(part.rotation)) - 1.0) < 1e-6, part.name
        assert part.scale == (1.0, 1.0, 1.0), f"{part.name} bakes size into scale"


#: The tolerance a grounded assembly's minimum Y may miss zero by. Not
#: ``0.0`` exactly: the grounding shift and the bounds it is measured against
#: both pass through float32 mesh positions on the way to an f8 accumulator,
#: so a residue at the scale of float noise is expected and a residue at the
#: scale of the audit's smallest miss (0.0009, insect) is exactly what this
#: guards against.
GROUND_TOLERANCE = 1e-5

#: Terrestrial keys: every assembly except the two the 2026-09-06 audit's
#: clay-08 finding names as swimmers. Derived from ``presets.SWIMMERS`` rather
#: than written out, so a ninth figure added to ``ASSEMBLIES`` without being
#: added to ``SWIMMERS`` is exercised here automatically.
TERRESTRIAL_KEYS = sorted(set(presets.ASSEMBLIES) - presets.SWIMMERS)

#: The two swimmers' authored placement, measured by the 2026-09-06 audit
#: before any fix: the serpent's spine sits at a constant template
#: ``z = 0.30`` and the fish's at ``z = 0.50``, both deliberate suspensions
#: rather than a miss -- see ``presets.SWIMMERS``'s docstring. Pinned here so
#: a future edit that nudges either number has to do so on purpose.
SWIMMER_PLACEMENT = {"serpent": 0.2200, "fish": 0.1663}


def _assembly_min_y(parts: tuple[presets.Part, ...]) -> float:
    """The lowest world-space Y across every part's *built* mesh.

    Independent of ``presets._world_min_y`` on purpose -- it happens to be the
    same computation, but a regression test that called the implementation
    under test to check the implementation under test could not catch the
    implementation being wrong in a way both call sites agree on.
    """
    lows = []
    for part in parts:
        defaults, make = GENERATORS[part.generator]
        local = make(**{**defaults, **part.params})
        matrix = m3.compose(
            np.asarray(part.translation, dtype="f8"),
            np.asarray(part.rotation, dtype="f8"),
            np.asarray(part.scale, dtype="f8"),
        )
        lo, _hi = bounds(transformed(local, matrix))
        lows.append(float(lo[1]))
    return min(lows)


@pytest.mark.parametrize("key", TERRESTRIAL_KEYS)
def test_every_terrestrial_figure_preset_meets_the_ground_plane(key: str):
    """The 2026-09-06 audit's clay-08 finding: assembled world bounds measured
    that day put six of eight figures' lowest point below Y=0 (humanoid
    -0.0214, biped_tail -0.0214, quadruped -0.0124, bird -0.0122, insect
    -0.0009, blob -0.1300) with nothing in ``presets.py`` or the manual saying
    that was wrong. The user has since decided terrestrial figures sit on the
    ground exactly: ``presets.build`` must ground every key that is not a
    swimmer, to within float noise of Y=0.
    """
    min_y = _assembly_min_y(presets.build(key))
    assert abs(min_y) < GROUND_TOLERANCE, f"{key} sits at minY={min_y:.4f}, not the ground"


@pytest.mark.parametrize("key", sorted(presets.SWIMMERS))
def test_swimmers_keep_their_authored_placement(key: str):
    """The other half of the clay-08 decision: ``serpent`` and ``fish`` have no
    ground-reaching part to have missed by an authoring error -- they are
    authored floating on purpose -- so grounding must leave them exactly where
    they were measured on 2026-09-06, not pull them down to Y=0 along with
    the six terrestrial figures.
    """
    grounded = presets.build(key)
    assert grounded == presets.ASSEMBLIES[key][1](), f"{key} moved under presets.build"
    min_y = _assembly_min_y(grounded)
    assert min_y == pytest.approx(SWIMMER_PLACEMENT[key], abs=1e-4)


def test_an_upright_figures_head_is_above_its_hips():
    """The swap, stated once in the direction a human can check by eye: in Clay
    space up is ``+Y``, so the humanoid's head has the larger Y and the
    quadruped's body runs along Z rather than standing up in Y."""
    parts = {p.name: p for p in presets.humanoid()}
    assert parts["Head"].translation[1] > parts["Hips"].translation[1]

    quad = {p.name: p for p in presets.quadruped()}
    span_y = abs(quad["Head"].translation[1] - quad["Hips"].translation[1])
    span_z = abs(quad["Head"].translation[2] - quad["Hips"].translation[2])
    assert span_z > span_y, "the quadruped is standing up; the axis swap is wrong"
