"""Regression tests for the 2026-09-26 audit, findings poser-rig-02 and
poser-rig-03: ``skeleton.validate_joints`` accepting input its own docstring
and every sibling refusal say it does not.
"""

from __future__ import annotations

import pytest

from realmspinner.kernels.rig import skeleton, templates


def test_validate_joints_refuses_a_non_object_bone_entry_with_a_value_error():
    """poser-rig-02: ``skeleton.validate_joints({"bones": ["hips"]})`` used
    to raise ``AttributeError`` (a bare string has no ``.get``) instead of
    the ``ValueError`` this function's docstring promises for a bad payload
    -- and ``service.rig.adjust_joints`` only catches ``(ValueError,
    KeyError, TypeError)``, so the raw ``AttributeError`` reached its caller
    unwrapped instead of becoming a field-addressed ``Invalid``."""
    template = templates.get_template("humanoid")
    with pytest.raises(ValueError):
        skeleton.validate_joints({"bones": ["hips"]}, template)


def test_validate_joints_refuses_a_bone_named_twice_instead_of_letting_the_last_win():
    """poser-rig-03: two entries naming the same bone used to both land in
    ``by_name`` with the second silently overwriting the first, dropping
    whichever correction came first with no error."""
    template = templates.get_template("humanoid")
    fitted = skeleton.fit_template(template, [-1, -1, 0], [1, 1, 2])
    bones = [{"name": b["name"], "head": b["head"], "tail": b["tail"]} for b in fitted]
    # Duplicate the first bone's entry under a different position -- if the
    # last one silently won, this payload would parse clean instead of
    # naming the repeat.
    duplicate = dict(bones[0], head=[x + 0.01 for x in bones[0]["head"]])
    bones.append(duplicate)
    with pytest.raises(ValueError, match="more than once"):
        skeleton.validate_joints({"bones": bones}, template)
