"""Rigging a mesh that arrives already rigged — the Troupe intake's real input.

Every mesh the rig path had ever been given was a TRELLIS reconstruction: no
armature, no vertex groups, no skin. That is not what the *supplied base mesh*
path receives. A humanoid a user brings to Troupe has usually been rigged by
whoever made it, and two defects lived in exactly that blind spot until
``tests/fixtures/humanoid/cesium_man.glb`` was added and this file asked.

**1. ``_skin``'s failure guard stopped working.** Bone-heat weighting reports
failure two ways -- an operator ``RuntimeError``, or a ``FINISHED`` that quietly
leaves every vertex group empty -- and ``_has_weights`` is what catches the
quiet one. It asks whether *any* vertex group carries a weight. An incoming
skin answers yes before the new armature is bound at all, so a bind that
produced nothing was reported as a clean ``automatic`` rig: the app says the
rig succeeded and the character does not deform.

**2. The old skeleton was exported beside the new one.** ``_import_glb``
returns the joined mesh and leaves the scene alone; ``_export`` writes the
*whole scene*. Two armatures in the output GLB, one with nothing weighted to it.

Both are fixed by ``_strip_incoming_rig``, and both are pinned below. Neither
could be reproduced with a synthetic fixture without first building a rigged
GLB, which is the fixture.

The fixture is CesiumMan (CC-BY 4.0, Cesium) -- see its ``ATTRIBUTION.md``. It
is a low-poly specification sample, so what it can settle is whether the
*mechanism* handles a skinned input. Whether a rig deforms well enough to ship
is an art verdict and needs the mesh the plan file asks for.

Run with: uv run pytest tests/test_rig_supplied_mesh.py -n 0
"""

from __future__ import annotations

from pathlib import Path

import pytest

#: A real bone-heat solve over 3k vertices, plus an import and an export.
pytestmark = pytest.mark.timeout(600)

FIXTURE = Path(__file__).parent / "fixtures" / "humanoid" / "cesium_man.glb"

#: What the file is known to carry, asserted so that swapping the fixture for a
#: different mesh fails here rather than silently weakening every test below.
SOURCE_BONES = 19
SOURCE_GROUPS = 19


@pytest.fixture
def imported():
    """The fixture through the real importer, on a clean scene. -> the mesh."""
    pytest.importorskip("bpy")
    import bpy

    from warlock.pipelines import blender_worker as bw

    assert FIXTURE.is_file(), f"missing fixture: {FIXTURE}"
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return bpy, bw, bw._import_glb(bpy, FIXTURE)


def test_the_fixture_is_the_rigged_textured_mesh_these_tests_assume(imported):
    """Guards every assertion below against a fixture swap.

    Also the record of what the file is: if this fails, the mesh changed and
    the defects the rest of this file pins may no longer be reachable through
    it -- which would make a green run mean nothing.
    """
    bpy, _bw, mesh = imported
    assert len(mesh.vertex_groups) == SOURCE_GROUPS
    assert len(mesh.data.materials) == 1, "the fixture must be textured"
    assert len(mesh.data.uv_layers) == 1
    armatures = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
    assert len(armatures) == 1
    assert len(armatures[0].data.bones) == SOURCE_BONES


def test_the_incoming_skin_defeats_the_weight_guard(imported):
    """The defect, stated as a property of the mesh rather than of the fix.

    This is what made defect 1 invisible: ``_has_weights`` is True on arrival,
    before anything has been bound. Asserted directly, because it is the reason
    ``_strip_incoming_rig`` has to run *before* ``_skin`` rather than inside
    its fallback chain -- and a future refactor that moves the call would pass
    every other test in this file while restoring the bug.
    """
    _bpy, bw, mesh = imported
    assert bw._has_weights(mesh) is True, (
        "the fixture is supposed to arrive skinned; without that this file "
        "cannot reach the defect it exists to pin"
    )


def test_stripping_leaves_no_skin_and_no_skeleton(imported):
    """After the strip: no groups, no armature, and the geometry untouched.

    The geometry half matters as much as the rest. ``_unbind`` also clears
    parenting and modifiers, and a strip that quietly dropped vertices would
    be a far worse bug than the one it fixes.
    """
    bpy, bw, mesh = imported
    verts_before = len(mesh.data.vertices)
    polys_before = len(mesh.data.polygons)

    removed = bw._strip_incoming_rig(bpy, mesh)

    assert removed == SOURCE_BONES
    assert len(mesh.vertex_groups) == 0
    assert not [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
    assert not [m for m in mesh.modifiers if m.type == "ARMATURE"]
    assert mesh.parent is None
    assert len(mesh.data.vertices) == verts_before, "the strip moved geometry"
    assert len(mesh.data.polygons) == polys_before
    assert len(mesh.data.materials) == 1, "the strip dropped the material"


def test_the_guard_can_detect_a_failed_bind_once_stripped(imported):
    """The fix, stated as the restored invariant rather than as a call count.

    ``_has_weights`` is only a guard while False means "nothing is weighted".
    After the strip it does, which is the whole point: a subsequent bind that
    produces no weights is now detectable, and ``_skin`` can fall through to
    envelope and *say so* instead of reporting a rig that is not there.
    """
    bpy, bw, mesh = imported
    bw._strip_incoming_rig(bpy, mesh)
    assert bw._has_weights(mesh) is False


def _live_size(mesh) -> tuple[float, float, float]:
    """The mesh's real world-space extent, read off the vertices themselves.

    ``matrix_world`` is only ever refreshed by a depsgraph update (F13: a
    strip that unparents without one leaves it stale, agreeing with a mesh
    that has since dropped to lying on its side). Forcing the update here is
    what makes this helper trustworthy rather than complicit in the same bug
    it is used to catch.
    """
    import bpy

    bpy.context.view_layer.update()
    points = [mesh.matrix_world @ v.co for v in mesh.data.vertices]
    lo = [min(p[i] for p in points) for i in range(3)]
    hi = [max(p[i] for p in points) for i in range(3)]
    return tuple(hi[i] - lo[i] for i in range(3))


def test_a_skinned_import_measures_wrong_until_it_is_stripped(imported):
    """The third consequence, and the one that would have ruined the rig quietly.

    A still-skinned mesh's ``bound_box`` does not describe the same shape its
    vertices do -- not, as this used to say, because ``matrix_world`` applies
    the Y-up -> Z-up rotation twice (see ``_strip_incoming_rig``'s docstring
    for what F13 found actually happens once the parent chain is gone).

    This is pinned as a *disagreement* rather than against fixed numbers: what
    makes it a bug is that the cached box and the actual vertices describe
    different objects, and any future change that keeps them consistent is a
    fix however the numbers land.
    """
    _bpy, bw, mesh = imported
    lo, hi = bw._world_bounds(mesh)
    reported = tuple(hi[i] - lo[i] for i in range(3))
    actual = _live_size(mesh)
    assert reported != pytest.approx(actual, abs=1e-3), (
        "a skinned mesh's bound_box now agrees with its vertices before any "
        "strip -- if so, delete this test and keep the one below"
    )


def test_the_stripped_mesh_stands_upright_and_measures_true(imported):
    """After the strip: the bounds agree with the geometry, and it is standing.

    ``_rig_bones`` fits the template to exactly this box, so a wrong one puts
    every joint in the wrong place while the stature stays plausible enough to
    pass a glance. The stored-pose spike already produced one sheet rendered
    lying down; this is the measurement that would let it happen again.
    """
    bpy, bw, mesh = imported
    bw._strip_incoming_rig(bpy, mesh)

    # F13: a stale matrix_world agrees with a mesh that has already fallen
    # over once the depsgraph catches up, so the update has to happen before
    # this test is allowed to call the two numbers a match.
    bpy.context.view_layer.update()
    lo, hi = bw._world_bounds(mesh)
    reported = tuple(hi[i] - lo[i] for i in range(3))
    assert reported == pytest.approx(_live_size(mesh), abs=1e-3)

    width, depth, height = reported
    assert height > width > depth, "this is not a standing figure in world space"
    assert 1.0 < height < 2.5, f"unexpected stature: {height:.2f} m"
    assert abs(lo[2]) < 0.1, f"feet are not near the floor: z={lo[2]:.3f}"


def test_a_supplied_rig_stays_standing_after_the_scene_re_evaluates(imported):
    """F13: CesiumMan's ``rig_qa.png`` rendered every cell lying down.

    ``_strip_incoming_rig`` used to leave the mesh's ``matrix_world`` stale --
    correct only until the next depsgraph update, which is exactly what
    ``_skin``'s ``parent_set`` triggers a moment later in ``op_rig``. This
    forces that update itself, standing in for ``_skin`` without needing a
    real bone-heat solve, and would have failed against the unfixed strip: the
    stale bounds looked upright right up until this call.
    """
    bpy, bw, mesh = imported
    bw._strip_incoming_rig(bpy, mesh)

    # Stand in for what ``_skin`` does next in ``op_rig``: parent the mesh to
    # a fresh armature, which is what forces Blender to re-evaluate
    # matrix_world off the mesh's own (now rotation-free) local transform.
    armature = bpy.data.armatures.new("probe")
    arm_obj = bpy.data.objects.new("probe", armature)
    bpy.context.scene.collection.objects.link(arm_obj)
    bpy.ops.object.select_all(action="DESELECT")
    mesh.select_set(True)
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.parent_set(type="OBJECT")
    bpy.context.view_layer.update()

    width, depth, height = _live_size(mesh)
    lo, _hi = bw._world_bounds(mesh)
    assert height > width > depth, "the re-evaluated mesh is lying down"
    assert height > 1.0, f"not standing height after re-evaluation: {height:.2f} m"
    assert abs(lo[2]) < 0.1, f"feet left the floor after re-evaluation: z={lo[2]:.3f}"


def test_a_full_rig_job_exports_a_standing_mesh(tmp_path):
    """F13, end to end: not a probe of one helper but the actual worker op a
    Troupe intake job runs, read back the way the shipped GLB would be -- by a
    library (``trimesh``) that shares no code with ``blender_worker`` and so
    cannot share its bug.

    Every test above pins one step of the mechanism; this is what would have
    caught the incident itself, since ``rig_qa.png`` rendered every cell of
    CesiumMan lying down only once the *whole* pipeline ran -- ``op_rig`` on a
    fresh scene, real bone-heat weighting, a real export.
    """
    pytest.importorskip("bpy")
    trimesh = pytest.importorskip("trimesh")
    import bpy

    from warlock.kernels.rig import blender_spec, store
    from warlock.pipelines import blender_worker

    assert FIXTURE.is_file(), f"missing fixture: {FIXTURE}"
    (tmp_path / "model.glb").write_bytes(FIXTURE.read_bytes())

    result = blender_worker.op_rig(bpy, blender_spec.rig_spec(tmp_path, "humanoid"))
    assert result["ok"] is True
    store.finalize_rig(tmp_path)

    # scene.dump bakes every node's transform into the vertices it returns --
    # the mesh's own plus, for a skinned export, whatever the skeleton nodes
    # contribute -- which is what makes this a check of the file rather than
    # a repeat of the in-process bpy measurement above.
    scene = trimesh.load(str(tmp_path / "rig.glb"), process=False)
    baked = scene.to_geometry() if hasattr(scene, "to_geometry") else scene.dump(concatenate=True)
    lo = baked.vertices.min(axis=0)
    hi = baked.vertices.max(axis=0)
    width, height, depth = hi - lo   # glTF: Y is up
    assert height > width > depth, f"CesiumMan shipped lying down: extent={hi - lo}"
    assert height > 1.0, f"unexpected stature: {height:.2f} m"
    assert abs(lo[1]) < 0.15, f"feet are not near the floor: y={lo[1]:.3f}"
