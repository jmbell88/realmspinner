"""Regression test for the 2026-09-26 audit, finding poser-rig-08.

``blender_worker._apply_pose`` computed ``delta = str(space) == "delta"``,
which treats *any* unrecognised spelling as ``"node"`` instead of refusing
it -- a corrupted or hand-edited ``pose_space``/``space`` value would apply
the wrong rotation frame silently (dev/INVARIANTS.md's own "not
interchangeable" warning about the two pose spaces) rather than fail loudly.
No real Blender is exercised: the check now runs before ``_apply_pose``'s
own ``from mathutils import Quaternion``, so this needs no ``bpy`` stub, the
identical reasoning ``tests/pipelines/test_audit_2026_09_23b_blender_worker.py``
states for testing this module with no Blender installed.

The audit's other half of this finding -- "no test references"
``tests/test_charsheet.py``'s ``sheet.POSE_SPACES``/``blender_worker.POSE_SPACES``
agreement check -- is wrong: ``test_the_two_modules_name_the_same_pose_spaces``
already exists there (added in 3d7ffad5, well before this audit) and passes.
That half is not re-fixed here; see the fixer's return notes.
"""

from __future__ import annotations

import sys
import types

import pytest

from realmspinner.pipelines import blender_worker


def test_apply_pose_refuses_an_unknown_pose_space_instead_of_treating_it_as_node():
    with pytest.raises(ValueError, match="space must be one of"):
        blender_worker._apply_pose(None, {}, "world")


@pytest.mark.parametrize("space", list(blender_worker.POSE_SPACES))
def test_apply_pose_accepts_every_known_pose_space_without_raising_on_the_space_check(
    space, monkeypatch
):
    # ``_apply_pose`` imports ``mathutils`` (a Blender-bundled module)
    # unconditionally, even though an empty ``bones`` dict means the loop
    # that would actually use it never runs -- stubbed the same way
    # ``test_audit_2026_09_23b_blender_worker.py`` stubs ``bpy``, so a known
    # space still needs no real Blender to reach "the check passed".
    stub = types.ModuleType("mathutils")
    # Never instantiated -- bones is empty, so the loop that would use it
    # does not run.
    stub.Quaternion = object
    monkeypatch.setitem(sys.modules, "mathutils", stub)
    applied, unknown = blender_worker._apply_pose(None, {}, space)
    assert applied == 0
    assert unknown == []
