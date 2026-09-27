"""Findings closed from the 2026-09-26 audit, viewer slice (w1f5).

create-viewer-01 (+clay-io-05): ``GpuModel.__init__`` already draws a skin
over ``MAX_JOINTS`` unskinned, at rest, and says so in the log -- but
``refresh_palettes`` never checked the same thing, so it kept calling
``Model.joint_palette`` (a Python list comprehension over every joint,
producing a second array the same size as the one ``gltf.py``'s loader was
separately found under-charging) for a node that was about to be drawn at
rest anyway, on every pose change.
"""

from __future__ import annotations

import numpy as np

from realmspinner.kernels.geom3d import gltf
from realmspinner.studio.viewer import scene as scenelib


def _over_budget_model() -> gltf.Model:
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="f4")
    indices = np.array([0, 1, 2], dtype="u4")
    prim = gltf.Primitive(positions=positions, indices=indices)
    node = gltf.Node(mesh=0, skin=0)
    n_joints = scenelib.MAX_JOINTS + 1
    skin = gltf.Skin(
        joints=[0] * n_joints, inverse_bind=np.tile(np.eye(4), (n_joints, 1, 1))
    )
    model = gltf.Model([node], [0], [[prim]], [skin])
    model.update_world()
    return model


def test_refresh_palettes_skips_building_a_palette_for_a_skin_drawn_at_rest(
    gl, monkeypatch
):
    model = _over_budget_model()

    calls: list[int] = []
    original = gltf.Model.joint_palette

    def counting(self, node):
        calls.append(1)
        return original(self, node)

    monkeypatch.setattr(gltf.Model, "joint_palette", counting)

    gpu = scenelib.GpuModel(gl, model)
    try:
        assert not calls, (
            "GpuModel.__init__ already draws this skin unskinned (over "
            "MAX_JOINTS) -- refresh_palettes must not build its full "
            f"palette anyway, but joint_palette ran {len(calls)} time(s)"
        )
        calls.clear()
        gpu.refresh_palettes()
        assert not calls, (
            "a later refresh_palettes() (e.g. after a pose change) must "
            f"also skip this skin, but joint_palette ran {len(calls)} time(s)"
        )
    finally:
        gpu.release()
