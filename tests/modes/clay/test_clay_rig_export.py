"""Export to the library writes a Clay rig as ``rig.glb`` and ``rig.json``.

Tranche 2 of the 2026-10-13 Clay rig plan. The mesh row stays what it always was -- an
unskinned ``model.glb`` -- and a document with bones gains the pair every downstream door keys
on (``files.ready("rig.glb")``): a skinned GLB placed where ``import_mesh`` grounded the mesh,
and a record in Blender's axes with legal bone names, mirror pairs and a weighting. Each claim
fails on the tree before the change: nothing wrote a rig, and ``build_asset`` handed the library
a skinned model.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from realmspinner.kernels.geom3d import gltf
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import rig
from realmspinner.kernels.rig import clayrig, skeleton, store
from realmspinner.service import files as svc_files
from realmspinner.studio.modes.clay import mode as clay_mode

from ._rigkit import skin_for, weighted


def _add(doc: bd.ClayDoc, name: str, **kw) -> int:
    kw.setdefault("mesh", bp.box())
    return doc.add_object(bd.Obj(uid=bd.new_uid(), name=name, **kw)).uid


def _character() -> bd.ClayDoc:
    doc = bd.ClayDoc()
    hip = _add(doc, "Hip", bone=True, translation=(0, 1, 0))
    arm_l = _add(doc, "Arm.L", bone=True, parent=hip, translation=(0.5, 0.5, 0))
    _add(doc, "Arm.R", bone=True, parent=hip, translation=(-0.5, 0.5, 0))
    _add(doc, "Hand", parent=arm_l, translation=(0.3, 0, 0))
    _add(doc, "Prop", translation=(4, 0, 0))
    return doc


# -- the pure record ------------------------------------------------------------------------


def test_no_bones_no_record() -> None:
    doc = bd.ClayDoc()
    _add(doc, "Box")
    assert clayrig.rig_record(doc) is None


def test_the_record_is_in_blender_axes_with_heads_tails_and_a_single_named_root() -> None:
    record = clayrig.rig_record(_character())
    bones = {b["name"]: b for b in record["bones"]}
    assert list(bones) == ["Hip", "Arm.L", "Arm.R"]
    assert record["root"] == "Hip" and record["template"] == "blank"
    assert record["skeleton"] == "custom" and record["source"] == "clay"
    assert bones["Hip"]["parent"] is None and bones["Arm.L"]["parent"] == "Hip"
    # glTF (0, 1, 0) is Blender (0, 0, 1); the arm sits half a metre out and half up.
    assert bones["Hip"]["head"] == [0.0, -0.0, 1.0]
    assert bones["Arm.L"]["head"] == [0.5, -0.0, 1.5]
    # A bone with a child points at it; a leaf points up by a visible amount.
    assert bones["Hip"]["tail"] == bones["Arm.L"]["head"]
    leaf = bones["Arm.R"]
    assert leaf["tail"][2] - leaf["head"][2] > 0.01 and leaf["tail"][:2] == leaf["head"][:2]
    store.validate_rig_bones(record["bones"])
    assert skeleton.check_skeleton_structure(record["bones"]) == "Hip"


def test_mirror_pairs_weighting_and_bounds() -> None:
    record = clayrig.rig_record(_character())
    assert record["mirror_pairs"] == [["Arm.L", "Arm.R"]]
    assert record["weighting"] == "rigid"
    lo, hi = record["bounds"]["min"], record["bounds"]["max"]
    assert lo[0] < hi[0] and lo[1] <= hi[1] and lo[2] < hi[2]
    assert hi[2] > 1.4, "the arms, half a metre above the hip at a metre, are inside the box"


def test_weighting_says_heat_when_every_shape_is_weighted_and_mixed_otherwise() -> None:
    doc = bd.ClayDoc()
    bone = _add(doc, "Hip", bone=True, mesh=bd._empty_mesh())
    _add(doc, "Body", mesh=weighted(bp.box()), skin=skin_for(bone))
    assert clayrig.rig_record(doc)["weighting"] == "heat"
    _add(doc, "Plate", parent=bone)
    assert clayrig.rig_record(doc)["weighting"] == "mixed"


def test_names_are_made_legal_and_unique_for_the_record() -> None:
    doc = bd.ClayDoc()
    _add(doc, "Left Arm", bone=True)
    _add(doc, "Left Arm", bone=True)
    _add(doc, "x" * 80, bone=True)
    _add(doc, "", bone=True)
    names = list(rig.export_names(doc.objects).values())
    assert names[0] == "Left_Arm" and names[1] == "Left_Arm.001"
    assert len(names[2]) == 63 and names[3] == "Bone"
    assert all(rig.BONE_NAME_RE.match(n) for n in names) and len(set(names)) == 4


def test_the_bone_name_rule_is_the_skeleton_editors() -> None:
    assert rig.BONE_NAME_RE.pattern == skeleton.BONE_NAME_RE.pattern


@pytest.mark.parametrize(
    ("names", "pairs"),
    [
        (["Arm.L", "Arm.R"], [["Arm.L", "Arm.R"]]),
        (["Leg_L", "Leg_R", "Spine"], [["Leg_L", "Leg_R"]]),
        (["LeftArm", "RightArm"], [["LeftArm", "RightArm"]]),
        (["Hand-L", "Hand-R"], [["Hand-L", "Hand-R"]]),
        (["Arm.L", "Leg.R"], []),
        (["Spine", "Head"], []),
    ],
)
def test_mirror_pairs_are_read_off_the_names(names: list[str], pairs: list[list[str]]) -> None:
    assert rig.mirror_pairs(names) == pairs


# -- the skinned file placed like the mesh --------------------------------------------------


def test_place_moves_every_root_and_every_baked_vertex_and_inverse_bind_together() -> None:
    doc = _character()
    place = np.eye(4)
    place[:3, 3] = [10.0, 0.0, 0.0]
    plain = bd.to_model(doc)
    moved = bd.to_model(doc, place=place)
    for a, b in zip(plain.nodes, moved.nodes, strict=False):
        if a.skin is not None:
            continue
        # Only roots move: a child is placed relative to its (moved) parent.
        assert np.allclose(b.translation - a.translation, 0) or np.allclose(
            b.translation - a.translation, [10, 0, 0]
        )
    for bone, ibm in zip(rig.bones(doc.objects), moved.skins[0].inverse_bind, strict=True):
        assert np.allclose(ibm, np.linalg.inv(place @ doc.world_matrix(bone.uid)), atol=1e-9)
    skinned_plain = [n for n in plain.nodes if n.skin is not None]
    skinned_moved = [n for n in moved.nodes if n.skin is not None]
    for a, b in zip(skinned_plain, skinned_moved, strict=True):
        pa = plain.meshes[a.mesh][0].positions
        pb = moved.meshes[b.mesh][0].positions
        assert np.allclose(pb - pa, [10, 0, 0], atol=1e-5)


def test_place_with_shear_is_refused() -> None:
    place = np.eye(4)
    place[0, 1] = 0.5
    with pytest.raises(ValueError, match="no shear"):
        bd.to_model(_character(), place=place)


def test_a_static_model_has_no_skin_whatever_the_rig() -> None:
    model = bd.to_model(_character(), skinned=False)
    assert model.skins == [] and all(n.skin is None for n in model.nodes)
    assert len(model.nodes) == 5, "one node per kept object, as before"


def test_bone_names_rename_the_joints_and_their_geometry() -> None:
    doc = bd.ClayDoc()
    _add(doc, "Left Arm", bone=True)
    names = rig.export_names(doc.objects)
    model = bd.to_model(doc, bone_names=names)
    assert [n.name for n in model.nodes] == ["Left_Arm", "Left_Arm"]


# -- through the real service ---------------------------------------------------------------


def test_exporting_a_rigged_document_writes_an_unskinned_model_and_a_rig_pair(svc) -> None:
    doc = _character()
    job_id = clay_mode.build_asset(svc, doc, title="Hero")
    job_dir = svc.job_dir(job_id)

    model = gltf.load((job_dir / "model.glb").read_bytes())
    assert model.skins == [], "the library's mesh is unskinned"
    rig_model = gltf.load((job_dir / "rig.glb").read_bytes())
    assert len(rig_model.skins) == 1 and len(rig_model.skins[0].joints) == 3
    record = json.loads((job_dir / "rig.json").read_text(encoding="utf-8"))
    assert record["source"] == "clay" and record["root"] == "Hip"
    assert (job_dir / "build.rblk").exists()
    # The completion marker: rig.json exists, so the library calls it rigged.
    row = svc.store.get(job_id)
    assert svc_files.ready(row, job_dir, "rig.glb")


def test_the_rig_stands_where_the_grounded_mesh_stands(svc) -> None:
    doc = _character()
    job_id = clay_mode.build_asset(svc, doc, title="Hero")
    job_dir = svc.job_dir(job_id)
    mesh_model = gltf.load((job_dir / "model.glb").read_bytes())
    rig_model = gltf.load((job_dir / "rig.glb").read_bytes())
    mesh_model.update_world()

    def box_of(model: gltf.Model, name: str, *, skinned: bool) -> tuple[np.ndarray, np.ndarray]:
        # ``normalize_glb`` empties a root and hangs its mesh on an unnamed child, so a root's
        # geometry is on the named node itself or on the unnamed child under it.
        owners = []
        for named in (n for n in model.nodes if n.name == name):
            owners += [named, *(model.nodes[c] for c in named.children if not model.nodes[c].name)]
        node = next(n for n in owners if n.mesh is not None)
        assert (node.skin is not None) == skinned, name
        points = np.vstack([p.positions for p in model.meshes[node.mesh]]).astype("f8")
        world = np.eye(4) if skinned else node.world
        placed = (world @ np.hstack([points, np.ones((len(points), 1))]).T).T[:, :3]
        return placed.min(axis=0), placed.max(axis=0)

    # ``Prop`` lies below everything the bones move, so grounding lifted the whole scene by it:
    # a rig placed without that lift would stand half a metre low.
    for name in ("Hip", "Arm.L", "Arm.R", "Hand"):
        lo_mesh, hi_mesh = box_of(mesh_model, name, skinned=False)
        lo_rig, hi_rig = box_of(rig_model, name, skinned=True)
        assert np.allclose(lo_mesh, lo_rig, atol=1e-4), name
        assert np.allclose(hi_mesh, hi_rig, atol=1e-4), name
    # And the joints sit in the same frame: the hip's joint node is at the grounded hip.
    rig_model.update_world()
    hip_joint = next(n for n in rig_model.nodes if n.name == "Hip" and n.skin is None)
    lo, hi = box_of(mesh_model, "Hip", skinned=False)
    assert np.allclose(hip_joint.world[:3, 3], (lo + hi) / 2, atol=1e-4)


def test_a_document_without_bones_exports_without_a_rig(svc) -> None:
    doc = bd.ClayDoc()
    _add(doc, "Box")
    job_id = clay_mode.build_asset(svc, doc, title="Crate")
    job_dir = svc.job_dir(job_id)
    assert not (job_dir / "rig.glb").exists() and not (job_dir / "rig.json").exists()


def test_the_rig_door_publishes_json_last_and_leaves_no_temps(svc) -> None:
    doc = _character()
    job_id = clay_mode.build_asset(svc, doc, title="Hero")
    job_dir = svc.job_dir(job_id)
    assert (
        not (job_dir / store.RIG_GLB_TMP).exists() and not (job_dir / store.RIG_JSON_TMP).exists()
    )


def test_the_rig_door_refuses_bad_input_and_removes_its_temps(svc) -> None:
    from realmspinner.service.errors import Invalid, NotFound

    doc = _character()
    job_id = clay_mode.build_asset(svc, doc, title="Hero")
    job_dir = svc.job_dir(job_id)
    good = (job_dir / "rig.glb").read_bytes()
    with pytest.raises(NotFound):
        svc_files.save_clay_rig(svc, "0" * 12, good, {"bones": [{"name": "a"}]})
    with pytest.raises(Invalid, match="not a binary glTF"):
        svc_files.save_clay_rig(svc, job_id, b"nope", {"bones": [{"name": "a"}]})
    with pytest.raises(Invalid, match="needs bones"):
        svc_files.save_clay_rig(svc, job_id, good, {"bones": []})
    with pytest.raises(Invalid, match="no name"):
        svc_files.save_clay_rig(svc, job_id, good, {"bones": [{"parent": None}]})
    assert (
        not (job_dir / store.RIG_GLB_TMP).exists() and not (job_dir / store.RIG_JSON_TMP).exists()
    )
    assert (job_dir / "rig.glb").read_bytes() == good, "a refusal leaves the published rig alone"
