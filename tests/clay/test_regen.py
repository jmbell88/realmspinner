"""``clay/regen.py``: what a generator rebuild carries over from the object it
is rebuilding, and what it cannot.

Every generator in ``primitives.py`` builds flat and grey -- ``_mesh`` stamps
a fresh ``material`` array of zeros and a fresh ``smooth`` array of falses on
every call, with no memory of the object it is rebuilding. Before
``regen.carry_over`` existed, the properties panel carried ``smooth`` back
(``clay_props._carry_shading``, now deleted) but never ``material``, and the
agent door carried neither -- so a box painted with a palette slot, or given a
hand-picked Shade Smooth in face mode, reverted to slot 0 and flat shading the
moment its numbers were touched. These tests pin the two-case rule the fix
reuses from that deleted helper's own precedent, generalised over both
attributes.
"""

from __future__ import annotations

import numpy as np

from warlock.studio.clay import mesh as bm
from warlock.studio.clay import primitives as bp
from warlock.studio.clay import regen, shading


def test_a_rebuild_that_keeps_its_face_count_keeps_the_per_face_material() -> None:
    old = bp.box(size=(1.0, 1.0, 1.0))
    painted = bm.Mesh(
        positions=old.positions,
        loops=old.loops,
        starts=old.starts,
        material=np.array([0, 1, 0, 1, 0, 1], dtype="i4"),
        smooth=old.smooth,
    )
    rebuilt = bp.box(size=(2.0, 1.0, 1.0))  # a resize: same six faces, same order
    assert bm.face_count(rebuilt) == bm.face_count(painted)

    out = regen.carry_over(painted, rebuilt, material=0)
    assert out.material.tolist() == painted.material.tolist()


def test_a_rebuild_that_keeps_its_face_count_keeps_the_per_face_smooth_flags() -> None:
    old = bp.box()
    shaded = bm.Mesh(
        positions=old.positions,
        loops=old.loops,
        starts=old.starts,
        material=old.material,
        smooth=np.array([True, False, True, False, True, False], dtype="?"),
    )
    rebuilt = bp.box(size=(1.0, 2.0, 1.0))
    assert bm.face_count(rebuilt) == bm.face_count(shaded)

    out = regen.carry_over(shaded, rebuilt, material=0)
    assert out.smooth.tolist() == shaded.smooth.tolist()


def test_a_rebuild_that_changes_its_face_count_stamps_the_objects_default_slot() -> None:
    old = bp.cylinder(segments=16)  # 18 faces: 16 sides + 2 caps
    rebuilt = bp.cylinder(segments=8)  # 10 faces: not the same faces any more
    assert bm.face_count(rebuilt) != bm.face_count(old)

    out = regen.carry_over(old, rebuilt, material=3)
    assert np.unique(out.material).tolist() == [3], (
        "a face-count change stamps the object's own default slot, not slot 0"
    )


def test_a_rebuild_that_changes_its_face_count_re_derives_shading_with_the_same_rule_as_insertion() -> None:  # noqa: E501
    old = bp.cylinder(segments=16)
    rebuilt = bp.cylinder(segments=8)
    assert bm.face_count(rebuilt) != bm.face_count(old)

    out = regen.carry_over(old, rebuilt, material=0)
    assert out.smooth.tolist() == shading.auto_smooth(rebuilt).smooth.tolist()
