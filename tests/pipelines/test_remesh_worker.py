"""The remesh worker, run for real. The one thing ``test_remesh.py`` cannot ask.

``test_remesh.py`` fakes Blender at ``blender_run.run_worker`` -- deliberately, and
its subject is the queue's rework contract: publish over ``model.glb`` by
rename, invalidate the derived exports, never touch ``source.glb``. Every
assertion there is about the *host* half, and the child is a stub that writes
a byte string.

So on 2026-08-30, when the remesh shipped, **nothing had ever executed
``op_remesh``**. Four stages -- voxel pre-pass, quadriflow, smart UV project,
selected-to-active bake -- existed with no evidence that any of them ran, and
the failure that would show up first on a user's card (a Blender build without
the quadriflow operator, a bake that silently produces a blank atlas) is
invisible to a fake by construction.

This file is that evidence, and it is deliberately **not** in the ``gpu`` lane.
That marker means "requires local GPU + model weights"; a remesh requires
neither -- Cycles bakes here at 4 samples on whatever device Blender defaults
to, and the input is a sphere this file builds. Putting it behind ``-m gpu``
would mean it ran only when somebody opted into a lane about model weights,
which is the opposite of what a regression test is for. It skips without
``bpy`` exactly as ``test_rig_weld.py`` and ``test_rigging.py`` do.

The subject is a UV sphere and not a reconstruction. That is the honest scope:
this asks *whether the four stages run and produce a mesh with the maps they
promise*, not whether a 300k-face trellis soup survives them. The second
question needs a real reconstruction and a person looking at it, and it is
`TODO.md` P32's session rather than a test.

Run with: uv run pytest tests/pipelines/test_remesh_worker.py -n 0
"""

from __future__ import annotations

from pathlib import Path

import pytest

from realmspinner import tiercheck
from realmspinner.kernels.rig import blender_spec
from realmspinner.pipelines import blender_run, remesh

#: A real quadriflow plus three bakes is far past the suite's 120 s hang net,
#: which is sized for the default lane's ~5 s worst case. Ten minutes is still
#: a hang net rather than a budget: what it catches is a wedged child, not a
#: slow one.
pytestmark = pytest.mark.timeout(600)

#: Small on purpose. The budget only has to be one quadriflow can actually hit
#: on a sphere; a game budget here would buy minutes of runtime and no extra
#: claim.
TARGET_FACES = 500
TEXTURE_PX = 512


@pytest.fixture(scope="module")
def source_glb(tmp_path_factory) -> Path:
    """A textured sphere, exported as GLB.

    Module-scoped: it is the input to every test below and building it twice
    would only re-roll the same deterministic mesh.

    It carries a Principled material with a non-default base colour, roughness
    and metallic, because the bake is *selected-to-active from the original*.
    An untextured input would bake three blank images and the maps below would
    pass while proving nothing about whether the bake read anything.
    """
    pytest.importorskip("bpy")
    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=1.0)
    obj = bpy.context.view_layer.objects.active

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project()
    bpy.ops.object.mode_set(mode="OBJECT")

    material = bpy.data.materials.new("probe")
    material.use_nodes = True
    principled = material.node_tree.nodes["Principled BSDF"]
    principled.inputs["Base Color"].default_value = (0.8, 0.2, 0.1, 1.0)
    principled.inputs["Roughness"].default_value = 0.35
    principled.inputs["Metallic"].default_value = 0.0
    obj.data.materials.append(material)

    out = tmp_path_factory.mktemp("remesh-src") / "source.glb"
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(filepath=str(out), export_format="GLB")
    return out


@pytest.fixture(scope="module")
def remeshed(source_glb, tmp_path_factory) -> tuple[Path, dict]:
    """One real remesh. -> (the written GLB, the child's result payload)."""
    work = tmp_path_factory.mktemp("remesh-out")
    out_glb = work / "remeshed.glb"
    spec = blender_spec.remesh_spec(
        source_glb,
        out_glb,
        work,
        target_faces=TARGET_FACES,
        texture_size=TEXTURE_PX,
    )
    result = blender_run.run_worker(spec)
    return out_glb, result


def test_the_child_runs_and_writes_the_glb_it_was_asked_for(remeshed):
    """The whole contract in one assertion: spec in, result JSON and a file out.

    Nothing had ever asserted this. A missing operator, an import error inside
    bpy or a bad spec key all land here, and all of them reach the user as a
    job that goes to ``error`` after Blender has already started.
    """
    out_glb, result = remeshed
    assert result["ok"] is True
    assert out_glb.is_file(), "the worker reported ok and wrote no mesh"
    assert out_glb.stat().st_size > 0


def test_the_budget_is_respected(remeshed):
    """``faces`` lands near the budget, and the mesh is genuinely reduced."""
    _out_glb, result = remeshed
    assert result["faces"] > 0
    assert result["faces_before"] > result["faces"], "nothing was reduced"
    # Generous: quadriflow hits a budget approximately, and the exact ratio is
    # a property of the operator rather than of this code.
    assert result["faces"] <= TARGET_FACES * 3


def test_the_surface_comes_back_as_quads(remeshed):
    """Quadriflow ran, and this is a regression gate rather than a formality.

    **The first run of this file failed here**, and the defect it found is the
    reason the assertion is this strict. glTF cannot share a vertex position
    between two texture coordinates, so a GLB splits its vertices at every UV
    seam -- and quadriflow refuses non-manifold input. Every input on this path
    is a GLB, so before the weld was added to ``op_remesh`` the quadriflow
    branch *could never succeed*: every remesh silently took the decimate
    fallback, and a feature whose profiles are spelled in quads shipped
    triangles.

    Measured on this sphere: 1,106 vertices before export, 4,512 after the
    round trip, quadriflow answering "Remeshing failed"; welded back to 1,106
    it returns ~479 faces, all quads.

    ``decimate`` remains a legitimate outcome for genuinely bad geometry, which
    is why ``report_line`` still distinguishes them. It is not a legitimate
    outcome for a UV sphere, and accepting it here would let the fallback
    become the only path again without a test noticing -- which is exactly what
    happened.
    """
    _out_glb, result = remeshed
    assert result["method"] == "quadriflow", (
        "the remesh fell back to decimate on a closed, manifold sphere; if the "
        "weld in op_remesh is gone or ineffective, every remesh ships triangles"
    )
    assert result["quads"] > 0.9, f"only {result['quads']:.0%} of the faces are quads"


def test_the_report_line_never_calls_a_decimated_mesh_quads(remeshed):
    """``report_line`` against a payload that came from the real worker.

    ``test_remesh.py`` already pins this rule against hand-built dicts. The
    thing only a real run can check is that the worker's *actual* keys are the
    ones ``report_line`` reads -- a renamed key would leave the pure test green
    and the panel showing nothing.
    """
    _out_glb, result = remeshed
    line = remesh.report_line(result)
    assert line, "the worker's payload produced no report line"
    if result["method"] == "decimate":
        assert "decimated" in line and "quads" not in line
    else:
        assert "quads" in line


def test_the_bake_produced_the_maps_the_remesh_promises(remeshed):
    """Base colour **and a normal map**, which is the point of the whole step.

    A TRELLIS reconstruction ships base colour and metallic/roughness and *no*
    normal map; the remesh's argument for existing is that it bakes the
    high-resolution geometry it just threw away into a tangent-space normal
    map. Read off the written GLB through ``tiercheck.survey`` -- the same
    reader the tier qualification uses -- rather than from the child's own
    report, because a worker reporting on itself is not evidence that the
    bytes on disk carry the texture.
    """
    out_glb, _result = remeshed
    survey = tiercheck.survey(out_glb)
    assert survey is not None, "the written GLB did not parse"
    assert survey.base_color, "no base colour survived the bake"
    assert survey.normal_map, "the remesh promises a baked normal map and wrote none"


def test_the_new_surface_is_uv_unwrapped(remeshed):
    """Every bake target needs UVs, so a mesh with none is a blank atlas.

    Asserted off the file for ``test_the_bake_...``'s reason: the unwrap is a
    stage whose failure is silent -- smart UV project on a mesh it cannot lay
    out leaves the bake reading background, and the maps above would still be
    *present*.
    """
    out_glb, _result = remeshed
    survey = tiercheck.survey(out_glb)
    assert survey is not None
    assert survey.primitives > 0
    assert survey.uv_primitives == survey.primitives, (
        "a primitive came back without UVs, so its bake read background"
    )


# --- _remesh_object directly: the CANCELLED bug and the x2 bug --------------
#
# Both regressions were in the decimate fallback's own branch, which
# ``remeshed`` above never reaches -- its sphere is built in memory and never
# round-tripped, so it stays manifold and quadriflow always takes it. Two
# different non-manifold meshes are used below rather than one, because the
# two bugs need to be pinned in isolation:
#
# - The CANCELLED bug is only visible on input where ``quadriflow_remesh``
#   itself reports ``{'CANCELLED'}`` rather than raising a Python exception --
#   measured 2026-09-23 to be a mesh with inconsistent face-normal winding
#   ("The mesh needs to be manifold and have face normals that point in a
#   consistent direction"), which is exactly the failure
#   ``dev/measurements/2026-09-23-default-mesh-budget.md`` records for a real
#   trellis mesh after the voxel pass. A mesh whose non-manifold-ness instead
#   makes the operator *raise* (an edge-split boundary, measured below) hits
#   the ``except`` branch identically whether or not the outcome is checked,
#   so it cannot tell the fixed code from the unfixed code on this bug.
# - The x2 bug is in the ``except`` branch's own arithmetic, so any mesh that
#   reaches it will do -- an edge-split boundary is simpler to build and to
#   keep almost entirely quads (a GLB reimport cannot: glTF only stores
#   triangles, so a round-tripped mesh has already lost every quad before
#   ``_remesh_object`` ever sees it).
#
# Both were confirmed against a hand-written copy of the pre-fix function
# (no outcome check; ``tris = len(obj.data.polygons)``) before this file was
# written: the inconsistent-normals mesh came back reported "quadriflow" with
# its face count *unchanged*, and the edge-split mesh's achieved triangle
# count landed at ~3.2-3.75x the target rather than the fixed code's
# measured, stable 2.0x.


@pytest.fixture
def inconsistent_normal_quad_mesh():
    """A UV sphere with some faces flipped -- quads, and non-manifold in the
    way that makes ``quadriflow_remesh`` return ``{'CANCELLED'}`` rather than
    raise. Measured 2026-09-23: 512 faces, unchanged by the operator."""
    pytest.importorskip("bpy")
    import bmesh
    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=1.0)
    obj = bpy.context.view_layer.objects.active
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    for face in bm.faces[:200]:
        face.normal_flip()
    bm.to_mesh(obj.data)
    bm.free()
    return bpy, obj


@pytest.fixture
def edge_split_quad_mesh():
    """A UV sphere with a band of edges split in place -- quads, and
    non-manifold in the way that makes ``quadriflow_remesh`` raise. Measured
    2026-09-23: 512 faces (448 quads); the operator raises "Remeshing
    failed" at every target used here, so both the fixed and the unfixed
    ``_remesh_object`` reach the decimate branch identically -- what differs
    between them is only its arithmetic, which is the whole point of using
    this mesh for that test rather than the flipped-normal one above."""
    pytest.importorskip("bpy")
    import bmesh
    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=1.0)
    obj = bpy.context.view_layer.objects.active
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.edges.ensure_lookup_table()
    for edge in bm.edges:
        if edge.index % 5 == 0:
            edge.select = True
    bm.to_mesh(obj.data)
    bm.free()
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.edge_split(type="EDGE")
    bpy.ops.object.mode_set(mode="OBJECT")
    return bpy, obj


def test_a_non_manifold_mesh_falls_back_to_decimate_with_fewer_faces(
    inconsistent_normal_quad_mesh,
):
    """Pins the CANCELLED bug: ``quadriflow_remesh`` *returns* ``{'CANCELLED'}``
    on this input rather than raising, so before ``_remesh_object`` checked
    the outcome, the mesh passed through completely untouched and was
    reported as a successful ``"quadriflow"`` remesh at 0% of the budget
    applied. Every TRELLIS reconstruction is non-manifold in exactly this way
    after the voxel pre-pass (2026-09-23, measured on the raccoon: 265k faces
    at 0% quads reported as "quadriflow").
    """
    from realmspinner.pipelines import blender_worker

    bpy, obj = inconsistent_normal_quad_mesh
    faces_before = len(obj.data.polygons)
    method = blender_worker._remesh_object(
        bpy, obj, target_faces=200, seed=0, close_holes=False, diagonal=2.0,
    )
    assert method == "decimate", (
        "quadriflow_remesh returns {'CANCELLED'} rather than raising on this "
        "mesh; a report of 'quadriflow' here means the CANCELLED result went "
        "undetected and the mesh passed through unchanged"
    )
    assert len(obj.data.polygons) < faces_before, "the decimate fallback did not reduce the mesh"


def test_the_decimate_ratio_counts_triangles_not_polygons(edge_split_quad_mesh):
    """Pins the x2 bug: the decimate ratio used to be computed from
    ``len(obj.data.polygons)`` -- a *quad* count -- rather than the mesh's real
    triangle count, so on an all-quad mesh the ratio asked for twice the
    budget (measured 2026-09-23 on the raccoon: 19,996 triangles for a 10k
    target). ``target_faces`` is deliberately half the triangle budget
    (``remesh.target_faces``), so the correct fallback lands close to
    *double* ``target_faces`` -- exactly what a caller asking for
    ``target_faces`` triangles worth of quads should see once decimate has
    triangulated the result. Measured stable at ratio 2.0 on this fixture
    (100/200/300); the pre-fix formula measured 3.2-3.75x.
    """
    from realmspinner.pipelines import blender_worker

    bpy, obj = edge_split_quad_mesh
    target = 200
    method = blender_worker._remesh_object(
        bpy, obj, target_faces=target, seed=0, close_holes=False, diagonal=2.0,
    )
    assert method == "decimate"
    achieved = sum(len(p.vertices) - 2 for p in obj.data.polygons)
    # Generous band around the measured 2.0x: comfortably excludes both "the
    # ratio did nothing" (~1x) and the pre-fix ~3.2-3.75x, while tolerating
    # decimate's own approximation.
    assert target * 1.5 <= achieved <= target * 3.0, (
        f"expected roughly {target * 2} triangles (2x target_faces), got {achieved}; "
        "a count past 3x means the ratio was computed from polygon count "
        "rather than real triangle count"
    )
