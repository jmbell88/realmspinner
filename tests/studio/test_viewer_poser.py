"""The viewer's pose half: bone lines, root translation, the render path.

Everything here runs over a hand-built skinless ``gltf.Model`` with no GL
context anywhere: ``bone_segments`` is pure, the editor is pure, and the
``Viewer`` methods under test are exercised unbound over a stub, the pattern
that keeps a GL object out of a logic test.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.kernels.geom3d.gltf import Model, Node
from realmspinner.studio.viewer import bonelines
from realmspinner.studio.viewer.pose import PoseEditor
from realmspinner.studio.viewer_embed import Viewer

BONES = ["hips", "spine", "arm.L", "arm.R"]


def _armature_model() -> Model:
    """A meshless armature: a 'rig' object node over a small joint tree --
    what a rig GLB loads as, minus the file."""
    nodes = [
        Node(name="rig", children=[1]),
        Node(name="hips", translation=m3.vec3(0.0, 0.53, 0.0), children=[2, 3, 4]),
        Node(name="spine", translation=m3.vec3(0.0, 0.07, 0.0)),
        Node(name="arm.L", translation=m3.vec3(-0.2, 0.1, 0.0)),
        Node(name="arm.R", translation=m3.vec3(0.2, 0.1, 0.0)),
    ]
    return Model(nodes, roots=[0], meshes=[], skins=[])


def _editor() -> PoseEditor:
    editor = PoseEditor()
    editor.bind(_armature_model(), BONES)
    editor.mirror_pairs = [["arm.L", "arm.R"]]
    editor.root = "hips"
    return editor


# --- bone segments ----------------------------------------------------------


def test_bone_segments_connect_a_bone_to_its_bound_parent():
    model = _armature_model()
    pairs = bonelines.bone_segments(model, BONES)
    assert pairs == [("hips", "spine"), ("hips", "arm.L"), ("hips", "arm.R")]


def test_the_root_gets_no_segment_because_its_parent_is_not_a_bone():
    """The armature object node above the root is a helper, not a joint; a
    line up to it would draw a bone that does not exist."""
    model = _armature_model()
    pairs = bonelines.bone_segments(model, BONES)
    assert all(bone != "hips" for _, bone in pairs)


def test_bone_segments_ignore_names_the_model_lacks():
    model = _armature_model()
    assert bonelines.bone_segments(model, ["hips", "tail_99"]) == []


# --- root translation -------------------------------------------------------


def test_set_root_translation_round_trips():
    editor = _editor()
    editor.set_root_translation([0.1, -0.2, 0.3])
    assert editor.root_translation() == pytest.approx([0.1, -0.2, 0.3])


def test_reset_all_restores_the_rest_translation():
    editor = _editor()
    editor.set_root_translation([0.3, 0.0, 0.0])
    editor.reset_all(dirty=False)
    assert editor.root_translation() == [0.0, 0.0, 0.0]
    assert editor.handles["hips"] == pytest.approx([0.0, 0.53, 0.0])
    assert editor.dirty is False


def test_reset_bone_on_the_root_restores_its_translation_too():
    editor = _editor()
    editor.set_root_translation([0.3, 0.0, 0.0])
    editor.reset_bone("hips")
    assert editor.root_translation() == [0.0, 0.0, 0.0]


def test_apply_preset_is_zero_translation():
    """A preset lists rotations only; applying one after a root move must not
    keep the character floating where the last pose left it."""
    editor = _editor()
    editor.set_root_translation([0.0, 0.0, 1.0])
    editor.apply_preset({"name": "idle", "bones": {"spine": [0, 0, 0, 1]}})
    assert editor.root_translation() == [0.0, 0.0, 0.0]


def test_mirror_reflects_the_root_translation():
    editor = _editor()
    editor.set_root_translation([0.3, 0.1, 0.2])
    editor.mirror()
    assert editor.root_translation() == pytest.approx([-0.3, 0.1, 0.2])


# --- the render path --------------------------------------------------------


def _overlay_stub(editor, model):
    """The attributes ``_overlays`` touches, and nothing else -- the
    unbound-method-over-a-stub pattern this file already uses, so no GL object
    goes anywhere near a logic test."""
    return SimpleNamespace(
        pose_mode=True,
        editor=editor,
        model=model,
        radius=1.0,
        placement=m3.identity(),
        _bone_pairs=[],
        bonelines=SimpleNamespace(draws=lambda *a, **k: ["lines"]),
        markers=SimpleNamespace(draws=lambda *a, **k: ["markers"]),
        # Non-None so the gizmo branch is entered at all; never called, because
        # the lookups bail before ``place``.
        _active_gizmo=lambda: object(),
    )


def test_a_selection_the_model_no_longer_carries_costs_the_gizmo_not_the_frame():
    """``_overlays`` runs on the frame thread, so a stale selection has to be a
    missing gizmo for a frame rather than a KeyError out of the render loop."""
    model = _armature_model()
    editor = _editor()
    editor.selected = "gone"

    items = Viewer._overlays(_overlay_stub(editor, model), 600)

    assert items == ["lines", "markers"]


def test_a_live_selection_still_reaches_the_gizmo():
    """The bail is the exception, not the rule: a real selection gets past both
    lookups and asks the gizmo to place itself."""
    model = _armature_model()
    editor = _editor()
    editor.selected = "spine"
    placed = []
    stub = _overlay_stub(editor, model)
    stub.translate_gizmo = object()
    stub.rotate_gizmo = SimpleNamespace(
        place=lambda *a: placed.append(a), draws=lambda: ["rings"]
    )
    stub.camera = None
    stub._active_gizmo = lambda: stub.rotate_gizmo

    assert Viewer._overlays(stub, 600) == ["lines", "markers", "rings"]
    assert placed


def test_mirror_keeps_the_pose_being_edited():
    """Mirror goes through apply, whose clearing of ``current`` is for
    *loading* -- a mirror is still the same pose, so it stays addressed and
    the next Save saves rather than silently becoming Save-as."""
    editor = _editor()
    editor.apply({"arm.L": [0.0, 0.0, 0.7071068, 0.7071068]}, pose_id="0123456789ab")
    editor.mirror()
    assert editor.current == "0123456789ab"
    assert editor.dirty is True


def test_reset_all_clears_the_pose_being_edited():
    """new_pose's half of the pair: back to rest is back to nothing-open."""
    editor = _editor()
    editor.apply({"arm.L": [0.0, 0.0, 0.7071068, 0.7071068]}, pose_id="0123456789ab")
    editor.reset_all(dirty=False)
    assert editor.current is None


def test_without_a_root_nothing_moves():
    """Nothing in production sets editor.root (Poser, the only setter, was
    removed on 2026-10-14), so a saved pose's root offset is a no-op."""
    editor = _editor()
    editor.root = None
    editor.set_root_translation([1.0, 1.0, 1.0])
    assert editor.root_translation() == [0.0, 0.0, 0.0]
    assert editor.dirty is False


def test_a_root_move_is_an_unsaved_edit():
    editor = _editor()
    editor.set_root_translation([0.1, 0.0, 0.0])
    assert editor.has_unsaved_edits()


# --- input: Alt+drag orbit (create-01) ---------------------------------------


def test_alt_drag_orbits_even_when_the_press_lands_on_a_pose_marker(monkeypatch):
    """The 2026-09-08 audit, finding create-01: ``_press`` tested a left press
    against the gizmo and every joint marker before it ever asked whether Alt
    was held, so an Alt+drag that started on top of a marker grabbed that
    joint instead of orbiting -- even though the comment further down in
    ``_press`` and ``shortcuts.py`` both promise Alt+drag orbits in every
    viewport, in every mode. Mirrors how ``test_clay_view.py`` proves the same
    contract for Clay's ``_view_drag`` handler: stub ``pygame.key.get_mods``
    to report Alt held, land the ray dead-centre on a marker, and check the
    press neither grabbed nor selected it.
    """
    import pygame

    monkeypatch.setattr(pygame.key, "get_mods", lambda: pygame.KMOD_ALT, raising=False)

    # A single marker sitting exactly on the ray the stubbed `_ray` returns,
    # so an unguarded press finds and grabs it every time.
    stub = SimpleNamespace(
        pose_mode=True,
        editor=SimpleNamespace(
            bound=True, selected=None, handles={"hips": np.array([0.0, 0.0, 0.0])}
        ),
        placement=m3.identity(),
        radius=1.0,
        _grab=None,
        _last_mouse=(0.0, 0.0),
        _ray=lambda local: (np.array([0.0, 0.0, 3.0]), np.array([0.0, 0.0, -1.0])),
        _active_gizmo=lambda: None,
    )
    # Unbound-method-over-a-stub pattern: ``_press`` calls ``self._mods()``,
    # bound here to the real implementation (``FrameOps._mods``, over the same
    # stub) so the test exercises the actual pygame read the monkeypatch above
    # sets up, not a second fake standing in for it.
    stub._mods = lambda: Viewer._mods(stub)

    consumed = Viewer._press(stub, 1, (0.0, 0.0))

    assert consumed is True
    assert stub._grab == "orbit"
    assert stub.editor.selected is None, "Alt+drag must not select the marker it started over"
