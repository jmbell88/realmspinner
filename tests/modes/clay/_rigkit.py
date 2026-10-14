"""Shared fixtures for the rig tests: a small document with a bone and a weighted mesh."""

from __future__ import annotations

import numpy as np

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import rig


def weighted(mesh: bm.Mesh, joint_count: int = 1) -> bm.Mesh:
    """*mesh* with every vertex weighted 1.0 to joint 0 (and a second joint, if asked, at 0)."""
    n = len(mesh.positions)
    joints = np.zeros((n, 4), dtype="i4")
    weights = np.zeros((n, 4), dtype="f4")
    weights[:, 0] = 1.0
    if joint_count > 1:
        joints[:, 1] = 1
    return bm.Mesh(
        positions=mesh.positions,
        loops=mesh.loops,
        starts=mesh.starts,
        material=mesh.material,
        smooth=mesh.smooth,
        uv=mesh.uv,
        joints=joints,
        weights=weights,
    )


def skin_for(*bone_uids: int) -> rig.Skin:
    return rig.Skin(tuple(bone_uids), np.tile(np.eye(4), (len(bone_uids), 1, 1)))


def rigged_doc() -> tuple[bd.ClayDoc, int, int]:
    """``(doc, bone_uid, body_uid)``: a marked bone and a box weighted to it."""
    doc = bd.ClayDoc()
    bone = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Hip", mesh=bp.box(), bone=True)).uid
    body = doc.add_object(
        bd.Obj(
            uid=bd.new_uid(),
            name="Body",
            mesh=weighted(bp.box()),
            skin=skin_for(bone),
        )
    ).uid
    return doc, bone, body
