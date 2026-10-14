"""Adopting a rig that Blender weighted: ``ClayDoc.adopt_rig``.

Tranche 4 of the 2026-10-13 Clay rig plan. The "file" here is a ``ClayDoc`` built by hand to look
like what a Blender weighting run reads back as -- bones in its own frames, a weighted mesh -- so
the document-side rules are asserted without a Blender run (the run itself is in
``tests/pipelines/test_clay_rig_blender.py``). Each claim fails on the tree before the change.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from realmspinner.kernels.geom3d import gltf
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import rig

from ._rigkit import skin_for, weighted

Q90 = np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])


def _add(doc: bd.ClayDoc, name: str, **kw) -> int:
    kw.setdefault("mesh", bp.box())
    return doc.add_object(bd.Obj(uid=bd.new_uid(), name=name, **kw)).uid


def _document() -> tuple[bd.ClayDoc, dict[str, int]]:
    """A hip bone with an arm bone under it, and two plain shapes (a torso and a hand) that the
    weights will replace; the hand rides the arm."""
    doc = bd.ClayDoc()
    hip = _add(doc, "Hip", bone=True, translation=(0, 1, 0))
    arm = _add(doc, "Arm", bone=True, parent=hip, translation=(1, 0, 0))
    torso = _add(doc, "Torso", translation=(0, 1, 0))
    hand = _add(doc, "Hand", parent=arm, translation=(0.5, 0, 0))
    return doc, {"hip": hip, "arm": arm, "torso": torso, "hand": hand}


def _blender_result(materials=None, bones=("Hip", "Arm", "Hand_end")) -> bd.ClayDoc:
    """What the weighted file reads back as: bones at their own rest frames (a metre up and a
    metre out, the arm turned 90 degrees), and one body weighted to the first two."""
    imported = bd.ClayDoc()
    hip = _add(imported, bones[0], bone=True, translation=(0, 1.2, 0))
    arm = _add(imported, bones[1], bone=True, parent=hip, translation=(0, 0, 1.0), rotation=Q90)
    tip = _add(imported, bones[2], bone=True, parent=arm, translation=(0, 0.4, 0))
    body = weighted(bp.box(), joint_count=2)
    # A real blend, so a file round trip keeps it a weighted mesh and not a rigid part.
    blend = body.weights.copy()
    blend[:4, 0], blend[:4, 1] = 0.7, 0.3
    blend[4:, 0], blend[4:, 1] = 0.2, 0.8
    body = replace(body, weights=blend)
    imported.add_object(
        bd.Obj(uid=bd.new_uid(), name="Body", mesh=body, skin=skin_for(hip, arm))
    )
    if materials is not None:
        imported.materials = list(materials)
    del tip
    return imported


def _names(doc: bd.ClayDoc) -> dict[int, str]:
    return rig.export_names(doc.objects)


def _adopt(doc, ids, imported=None, to_doc=None, remove=None):
    imported = imported or _blender_result()
    return doc.adopt_rig(
        imported,
        bone_names=_names(doc),
        to_doc=np.eye(4) if to_doc is None else to_doc,
        remove=[ids["torso"], ids["hand"]] if remove is None else remove,
    )


def test_a_bone_that_matches_by_name_takes_the_imported_rest_and_its_shapes_stay_put() -> None:
    doc, ids = _document()
    hand_before = doc.world_matrix(ids["hand"]).copy()
    counts = _adopt(doc, ids, remove=[ids["torso"]])
    assert counts["matched"] == 2
    assert np.allclose(doc.world_matrix(ids["hip"])[:3, 3], [0, 1.2, 0])
    assert np.allclose(doc.world_matrix(ids["arm"])[:3, 3], [0, 1.2, 1.0])
    assert np.allclose(doc.by_uid(ids["arm"]).rotation, Q90, atol=1e-6)
    assert np.allclose(doc.world_matrix(ids["hand"]), hand_before, atol=1e-9), (
        "a shape under a bone is not moved by the bone's new frame"
    )


def test_an_imported_bone_that_matches_nothing_becomes_a_new_bone_under_its_parents_match() -> None:
    doc, ids = _document()
    _adopt(doc, ids)
    tip = next(o for o in doc.objects if o.name == "Hand_end")
    assert tip.bone and tip.parent == ids["arm"] and tip.mesh.positions.shape[0] == 0
    # 0.4 along the arm's own Y, which the arm's 90-degree turn lays along world -X.
    assert np.allclose(doc.world_matrix(tip.uid)[:3, 3], [-0.4, 1.2, 1.0], atol=1e-6)


def test_the_weighted_mesh_arrives_at_the_identity_with_a_skin_naming_our_bones() -> None:
    doc, ids = _document()
    counts = _adopt(doc, ids)
    assert counts["meshes"] == 1
    body = next(o for o in doc.objects if o.name == "Body")
    assert body.skin.joints == (ids["hip"], ids["arm"]) and body.parent is None
    assert body.mesh.joints is not None and np.allclose(body.mesh.weights.sum(axis=1), 1.0)
    # The skin's inverse bind matrices are the bones' own rest worlds, inverted.
    for joint, matrix in zip(body.skin.joints, body.skin.inverse_bind, strict=True):
        assert np.allclose(matrix @ doc.world_matrix(joint), np.eye(4), atol=1e-6)
    assert doc.rig_problems() == []


def test_the_shapes_the_weights_replace_are_removed_and_it_is_all_one_undo_step() -> None:
    doc, ids = _document()
    before = [(o.uid, o.name) for o in doc.objects]
    translations = {o.uid: np.array(o.translation, copy=True) for o in doc.objects}
    steps = len(doc.history)
    counts = _adopt(doc, ids)
    assert counts["removed"] == 2
    assert not {ids["torso"], ids["hand"]} & {o.uid for o in doc.objects}
    assert len(doc.history) == steps + 1
    doc.undo()
    assert [(o.uid, o.name) for o in doc.objects] == before
    for o in doc.objects:
        assert np.allclose(o.translation, translations[o.uid]), o.name
    assert doc.redo() and any(o.name == "Body" for o in doc.objects)


def test_the_imported_files_world_is_carried_into_the_documents_by_to_doc() -> None:
    doc, ids = _document()
    place = np.diag([2.0, 2.0, 2.0, 1.0])
    place[:3, 3] = [0.0, 0.5, 0.0]
    _adopt(doc, ids, to_doc=np.linalg.inv(place))
    # The file stood at twice the scale and half a metre up: its hip at y=1.2 is y=0.35 here.
    assert np.allclose(doc.world_matrix(ids["hip"])[:3, 3], [0, 0.35, 0], atol=1e-6)
    body = next(o for o in doc.objects if o.name == "Body")
    # A unit box spans y in -0.5..0.5 there, which is (y - 0.5) / 2 here: -0.5..0.
    assert body.mesh.positions[:, 1].max() == pytest.approx(0.0, abs=1e-6)
    assert body.mesh.positions[:, 1].min() == pytest.approx(-0.5, abs=1e-6)


def test_materials_are_matched_into_the_palette_by_name_and_colour_and_appended_otherwise() -> None:
    doc, ids = _document()
    own = list(doc.materials)
    stray = gltf.Material(name="Painted", base_color_factor=(0.2, 0.4, 0.6, 1.0))
    imported = _blender_result(materials=[own[0], stray])
    imported.by_uid(next(o.uid for o in imported.objects if o.name == "Body")).mesh = replace_faces(
        imported.objects[-1].mesh, [0, 1, 0, 1, 0, 1]
    )
    _adopt(doc, ids, imported=imported)
    assert len(doc.materials) == len(own) + 1 and doc.materials[-1].name == "Painted"
    body = next(o for o in doc.objects if o.name == "Body")
    assert set(int(i) for i in body.mesh.material) == {0, len(own)}


def replace_faces(mesh, slots):
    from dataclasses import replace

    return replace(mesh, material=np.array(slots, dtype="i4"))


def test_a_name_that_is_taken_gets_a_numbered_twin() -> None:
    doc, ids = _document()
    _add(doc, "Body")
    _adopt(doc, ids)
    assert sorted(o.name for o in doc.objects if o.name.startswith("Body")) == ["Body", "Body.001"]


def test_a_file_with_no_bones_or_no_weights_is_refused_and_changes_nothing() -> None:
    doc, ids = _document()
    steps = len(doc.history)
    bare = bd.ClayDoc()
    _add(bare, "Box")
    with pytest.raises(el.OpError, match="no bones"):
        _adopt(doc, ids, imported=bare)
    unweighted = bd.ClayDoc()
    _add(unweighted, "Hip", bone=True)
    with pytest.raises(el.OpError, match="no weights"):
        _adopt(doc, ids, imported=unweighted)
    assert len(doc.history) == steps and any(o.name == "Torso" for o in doc.objects)


def test_a_shape_to_remove_that_is_not_there_is_refused_before_anything_changes() -> None:
    doc, ids = _document()
    before = [(o.uid, o.name, o.translation.copy()) for o in doc.objects]
    steps = len(doc.history)
    with pytest.raises(KeyError):
        _adopt(doc, ids, remove=[ids["torso"], 99999])
    assert len(doc.history) == steps
    assert [(o.uid, o.name) for o in doc.objects] == [(u, n) for u, n, _ in before]


def test_a_refusal_after_the_bones_moved_takes_them_back_and_leaves_no_step(monkeypatch) -> None:
    doc, ids = _document()
    steps = len(doc.history)
    hip_before = np.array(doc.by_uid(ids["hip"]).translation, copy=True)
    names = [o.name for o in doc.objects]
    monkeypatch.setattr(doc, "remove_objects", lambda uids: ["Torso cannot go."])
    with pytest.raises(el.OpError, match="Torso cannot go"):
        _adopt(doc, ids)
    assert len(doc.history) == steps and not doc.history.can_redo
    assert np.allclose(doc.by_uid(ids["hip"]).translation, hip_before)
    assert [o.name for o in doc.objects] == names
