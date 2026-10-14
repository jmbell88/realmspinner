"""The 2026-10-14 audit's residue round (fixer R), finding clay-12's other half.

``clay_op``'s enum already leaves out the Pose-mode ops, but ``clay_catalog(topic="ops")``
still described them, teaching an agent four names it could only be refused for.
"""

from __future__ import annotations

from realmspinner.studio.modes.clay import ops as clay_ops
from realmspinner.studio.modes.clay.agent import schema


def test_the_ops_catalog_describes_no_pose_mode_op():
    text = schema.CATALOG_TOPICS["ops"]()

    assert clay_ops.POSE_OPS, "the guard would pass for an empty list"
    for name in clay_ops.POSE_OPS:
        assert name not in text, f"{name} is gated on Pose mode, which no agent tool enters"


def test_the_ops_catalog_still_describes_every_other_op():
    text = schema.CATALOG_TOPICS["ops"]()

    for op in clay_ops.OPS:
        if op.name not in clay_ops.POSE_OPS:
            assert op.name in text, op.name
