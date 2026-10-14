"""Cleanup round after the 2026-10-14 audit's Clay fixes: Mirror Copy's half of
clay-mesh-core-02 (``rebind_skin``) and ``animimport``'s half of clay-rig-08.
Each test's name is its claim."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from realmspinner.kernels.geom3d import glbwrite, gltf
from realmspinner.kernels.mesh import animimport, rig
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio.modes.clay import ops as clay_ops


def _weighted(mesh: bm.Mesh) -> bm.Mesh:
    n = len(mesh.positions)
    joints = np.zeros((n, 4), dtype="i4")
    weights = np.zeros((n, 4), dtype="f4")
    weights[:, 0] = 1.0
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


def _ctx():
    return SimpleNamespace(toast=lambda *a, **k: None, inline=True)


def test_mirror_copy_binds_a_copied_weighted_mesh_to_the_copied_bone() -> None:
    doc = bd.ClayDoc()
    bone = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Hip", mesh=bp.box(), bone=True)).uid
    skin = rig.Skin((bone,), np.eye(4)[None].copy())
    body = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="Body", mesh=_weighted(bp.box()), skin=skin)
    ).uid
    doc.select([bone, body])
    assert clay_ops.run(_ctx(), doc, clay_ops.get("mirror-copy"), axis=0.0) is True
    originals = {bone, body}
    copies = [o for o in doc.objects if o.uid not in originals]
    copy_bone = next(o for o in copies if o.bone)
    copy_body = next(o for o in copies if o.skin is not None)
    assert copy_body.skin.joints == (copy_bone.uid,), "mirrored mesh names the original bone"
    assert doc.by_uid(body).skin.joints == (bone,)


def _walking_model() -> gltf.Model:
    doc = bd.ClayDoc()
    hip = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Hip", mesh=bp.box(), bone=True)).uid
    doc.add_animation("Walk")
    doc.set_key(hip, "translation", 0.0, [0, 1, 0])
    doc.set_key(hip, "translation", 1.0, [0, 2, 0])
    return gltf.load(glbwrite.write_glb(bd.to_model(doc)))


def _target() -> bd.ClayDoc:
    target = bd.ClayDoc()
    target.add_object(bd.Obj(uid=bd.new_uid(), name="Hip", mesh=bp.box(), bone=True))
    return target


def _replace(model: gltf.Model, **changes) -> None:
    for i, ch in enumerate(model.animations[0].channels):
        if ch.path == "translation":
            model.animations[0].channels[i] = dataclasses.replace(ch, **changes)


def test_import_animation_names_the_bone_when_a_key_time_is_refused() -> None:
    # clay-rig-08: ``anim.Channel``'s ValueError escaped ``animimport._retarget``
    # as a bare ValueError (which the door does not catch).
    model = _walking_model()
    _replace(model, times=np.array([-1.0, 1.0]))
    with pytest.raises(el.OpError) as info:
        animimport.import_by_name(_target(), model)
    assert "Hip" in str(info.value) and "translation" in str(info.value)


def test_import_animation_names_the_bone_when_a_smoothed_channel_is_refused() -> None:
    # clay-rig-08: ``_linear_keys`` was called without ``where=``, so its refusals
    # said "A smoothed channel" for every bone.
    model = _walking_model()
    _replace(
        model,
        times=np.array([0.0, 1.0, 1e10]),
        values=np.zeros((9, 3)),
        interpolation="CUBICSPLINE",
    )
    with pytest.raises(el.OpError) as info:
        animimport.import_by_name(_target(), model)
    assert "Hip" in str(info.value) and "A smoothed channel" not in str(info.value)


def test_the_clay_bone_name_pattern_refuses_a_trailing_newline() -> None:
    # The 2026-10-14 audit, finding poser-rig-06 (fixed in ``kernels/rig``, here for the
    # mesh kernel's copy): ``$`` matches before a final
    # newline, so ``hips\n`` was a valid bone name here though ``kernels/rig`` refuses it, and
    # ``test_clay_rig_export`` pins the two patterns equal.
    assert rig.BONE_NAME_RE.match("hips") is not None
    assert rig.BONE_NAME_RE.match("hips\n") is None
