"""The 2026-10-14 audit, fixer F13: the rigged-GLB importer's untrusted numbers and its refusals,
Familiar's preview clone and the rig, a duplicated rig, an element Delete over a weighted object,
and Separate's cost against mesh size. One claim per test; each name is the finding it closes."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from realmspinner.kernels.geom3d import glbwrite, gltf
from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.kernels.mesh import anim, animimport, glbimport, rig, selection, separate
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import scratch as sc


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


def _skin(*bone_uids: int) -> rig.Skin:
    return rig.Skin(tuple(bone_uids), np.tile(np.eye(4), (len(bone_uids), 1, 1)))


def _rigged() -> tuple[bd.ClayDoc, int, int]:
    doc = bd.ClayDoc()
    bone = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Hip", mesh=bp.box(), bone=True)).uid
    body = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="Body", mesh=_weighted(bp.box()), skin=_skin(bone))
    ).uid
    return doc, bone, body


def _walking_glb() -> bytes:
    doc = bd.ClayDoc()
    hip = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Hip", mesh=bp.box(), bone=True)).uid
    doc.add_animation("Walk")
    doc.set_key(hip, "translation", 0.0, [0, 1, 0])
    doc.set_key(hip, "translation", 1.0, [0, 2, 0])
    doc.set_key(hip, "rotation", 0.0, [0, 0, 0, 1])
    doc.set_key(hip, "rotation", 1.0, [0, 0.7071, 0, 0.7071])
    return glbwrite.write_glb(bd.to_model(doc))


def _loading(monkeypatch: pytest.MonkeyPatch, edit) -> None:
    real = gltf.load

    def load(data, *a, **k):
        model = real(data, *a, **k)
        edit(model)
        return model

    monkeypatch.setattr(gltf, "load", load)


def _replace_channel(path: str, **changes):
    def edit(model: gltf.Model) -> None:
        for i, ch in enumerate(model.animations[0].channels):
            if ch.path == path:
                model.animations[0].channels[i] = dataclasses.replace(ch, **changes)

    return edit


def _cubic(times) -> object:
    times = np.asarray(times, dtype="f8")
    return _replace_channel(
        "translation",
        times=times,
        values=np.tile(np.array([[0.0, 1.0, 0.0]]), (3 * len(times), 1)),
        interpolation="CUBICSPLINE",
    )


# --- clay-rig-01 (clay-39): a cubic channel's time decides nothing before it is checked ----------


def test_a_cubic_channel_with_a_huge_last_time_is_refused_before_anything_is_resampled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def resample(*_a, **_k):
        calls.append(1)
        raise RuntimeError("resampled a channel whose last key is 1e10 seconds away")

    monkeypatch.setattr(anim, "resample_cubic", resample)
    _loading(monkeypatch, _cubic([0.0, 1.0, 1e10]))
    with pytest.raises(el.OpError, match="seconds"):
        glbimport.glb_to_claydoc(_walking_glb())
    assert calls == []


def test_a_cubic_channel_with_a_huge_last_time_is_refused_on_import_animation_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``animimport`` shares ``rigimport._linear_keys``, so the same door covers both."""
    monkeypatch.setattr(
        anim, "resample_cubic", lambda *a, **k: pytest.fail("resampled before checking")
    )
    data = _walking_glb()
    target = bd.ClayDoc()
    target.add_object(bd.Obj(uid=bd.new_uid(), name="Hip", mesh=bp.box(), bone=True))
    model = gltf.load(data)
    _replace_channel("translation", times=np.array([0.0, 1.0, 1e10]),
                     values=np.zeros((9, 3)), interpolation="CUBICSPLINE")(model)
    with pytest.raises(el.OpError):
        animimport.import_by_name(target, model)


# --- clay-rig-08 (clay-83): a key Channel refuses is an OpError naming the animation -------------


def test_a_negative_key_time_in_a_file_is_an_operror_naming_the_animation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _loading(monkeypatch, _replace_channel("translation", times=np.array([-1.0, 1.0])))
    with pytest.raises(el.OpError) as info:
        glbimport.glb_to_claydoc(_walking_glb())
    assert "Walk" in str(info.value) and "Hip" in str(info.value)


def test_a_zero_quaternion_key_in_a_file_is_an_operror_naming_the_animation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def zero(model: gltf.Model) -> None:
        for i, ch in enumerate(model.animations[0].channels):
            if ch.path == "rotation":
                values = ch.values.copy()
                values[0] = 0.0
                model.animations[0].channels[i] = dataclasses.replace(ch, values=values)

    _loading(monkeypatch, zero)
    with pytest.raises(el.OpError) as info:
        glbimport.glb_to_claydoc(_walking_glb())
    assert "Walk" in str(info.value)


def test_a_non_ascending_cubic_channel_is_an_operror_not_a_nan_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _loading(monkeypatch, _cubic([0.0, 1.0, 1.0]))
    with pytest.raises(el.OpError):
        glbimport.glb_to_claydoc(_walking_glb())


# --- clay-rig-09 (clay-84): a flattened joint is refused by name ---------------------------------


def test_a_joint_with_a_zero_scale_axis_is_refused_by_name_on_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc, _bone, _body = _rigged()
    data = glbwrite.write_glb(bd.to_model(doc))

    def flatten(model: gltf.Model) -> None:
        node = model.nodes[model.skins[0].joints[0]]
        node.scale = np.array([1.0, 0.0, 1.0])
        node.world = m3.compose(node.translation, node.rotation, node.scale)

    _loading(monkeypatch, flatten)
    with pytest.raises(el.OpError, match="Hip"):
        glbimport.glb_to_claydoc(data)


# --- clay-rig-10 (clay-85): a clip that moves no bone is not imported ----------------------------


def test_a_clip_that_drives_no_bone_is_not_imported_as_an_empty_animation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc, _bone, _body = _rigged()
    doc.add_animation("Walk")
    doc.set_key(_bone, "translation", 0.0, [0, 0, 0])
    data = glbwrite.write_glb(bd.to_model(doc))

    def retarget(model: gltf.Model) -> None:
        bones = {int(j) for skin in model.skins for j in skin.joints}
        other = next(i for i in range(len(model.nodes)) if i not in bones)
        for ch in model.animations[0].channels:
            ch.node = other

    _loading(monkeypatch, retarget)
    imported = glbimport.glb_to_claydoc(data)
    assert imported.rig_state.animations == ()
    assert imported.active_animation is None
    assert any("moved no bone" in n for n in imported.notices)


# --- clay-rig-12 (clay-87): the refusal for a long smoothed clip names the resampling ------------


def test_a_long_cubic_clip_refusal_names_the_resampling_not_the_file_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _loading(monkeypatch, _cubic([0.0, 1 / 30, 150.0]))
    with pytest.raises(el.OpError) as info:
        glbimport.glb_to_claydoc(_walking_glb())
    text = str(info.value)
    assert "resampl" in text
    assert "Walk" in text and "Hip" in text


# --- clay-mesh-model-01 (clay-20): a preview's rig edits apply -----------------------------------


def test_a_preview_that_marks_a_bone_or_unbinds_a_skin_or_keys_a_pose_applies_it() -> None:
    # marking a bone
    doc = bd.ClayDoc()
    a = doc.add_object(bd.Obj(uid=bd.new_uid(), name="a", mesh=bp.box())).uid
    scratch = sc.clone(doc)
    scratch.set_bone([a], True)
    delta = sc.diff(doc, scratch)
    assert not delta.empty
    assert sc.transplant(doc, scratch, delta)
    assert doc.by_uid(a).bone is True
    assert doc.undo() and doc.by_uid(a).bone is False, "one Familiar apply is one undo step"

    # unbinding a skin (the mesh keeps its shape, and the object keeps no skin)
    doc, bone, body = _rigged()
    scratch = sc.clone(doc)
    scratch.unbind_skin(body)
    delta = sc.diff(doc, scratch)
    assert sc.transplant(doc, scratch, delta)
    assert doc.by_uid(body).skin is None and doc.by_uid(body).mesh.joints is None
    assert doc.rig_problems() == []

    # keying a pose on an animation the clone inherited
    doc = bd.ClayDoc()
    hip = doc.add_bone(None, "Hip").uid
    doc.add_animation("Walk")
    scratch = sc.clone(doc)
    assert [x.name for x in scratch.rig_state.animations] == ["Walk"], "the clone carries the rig"
    assert scratch.active_animation == doc.active_animation
    scratch.set_key(hip, "translation", 0.0, [0, 1, 0])
    scratch.add_animation("Run")
    delta = sc.diff(doc, scratch)
    assert not delta.empty
    assert sc.transplant(doc, scratch, delta)
    assert [x.name for x in doc.rig_state.animations] == ["Walk", "Run"]
    assert hip in doc.rig_state.animations[0].channels
    assert doc.active_animation == scratch.active_animation


def test_a_preview_that_unmarks_a_bone_after_unbinding_its_skin_applies_in_that_order() -> None:
    doc, bone, body = _rigged()
    scratch = sc.clone(doc)
    scratch.unbind_skin(body)
    scratch.set_bone([bone], False)
    delta = sc.diff(doc, scratch)
    assert sc.transplant(doc, scratch, delta)
    assert doc.by_uid(bone).bone is False and doc.by_uid(body).skin is None


# --- clay-mesh-core-02 / clay-mesh-ops-02 (clay-07): a duplicated rig is its own rig -------------


def test_duplicating_a_bone_and_its_weighted_mesh_binds_the_copy_to_the_copied_bone() -> None:
    doc, bone, body = _rigged()
    doc.select([bone, body])
    made = selection.duplicate_selected(doc)
    by_old = dict(zip((bone, body), made, strict=True))
    copy_bone, copy_body = by_old[bone], by_old[body]
    assert doc.by_uid(copy_bone).bone
    assert doc.by_uid(copy_body).skin.joints == (copy_bone,)
    assert doc.by_uid(body).skin.joints == (bone,), "the original keeps its own skeleton"
    assert doc.rig_problems() == []


def test_duplicating_a_weighted_mesh_alone_keeps_it_on_the_original_skeleton() -> None:
    doc, bone, body = _rigged()
    doc.select([body])
    (copy,) = selection.duplicate_selected(doc)
    assert doc.by_uid(copy).skin.joints == (bone,)


def test_rebind_skin_moves_a_skin_only_when_every_joint_it_names_was_copied() -> None:
    from realmspinner.kernels.mesh import ops

    skinned = bd.Obj(uid=1, name="m", mesh=_weighted(bp.box()), skin=_skin(10, 11))
    assert ops.rebind_skin(skinned, {10: 20, 11: 21}).skin.joints == (20, 21)
    assert ops.rebind_skin(skinned, {10: 20}).skin is skinned.skin
    assert ops.rebind_skin(skinned, {}).skin is skinned.skin


# --- clay-mesh-core-01 (clay-19): element Delete refuses one object and finishes the rest --------


def _corner_faces(mesh: bm.Mesh) -> list[int]:
    """The faces that use vertex 0, so deleting them drops that vertex (a count change)."""
    counts = np.diff(mesh.starts)
    out = []
    for f in range(len(counts)):
        if 0 in mesh.loops[mesh.starts[f] : mesh.starts[f + 1]]:
            out.append(f)
    return out


@pytest.mark.parametrize("weighted_first", [True, False])
def test_element_delete_refuses_a_weighted_object_by_name_and_still_deletes_the_others(
    weighted_first: bool,
) -> None:
    doc = bd.ClayDoc()
    bone = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Hip", mesh=bp.box(), bone=True)).uid
    prop = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Prop", mesh=bp.box())).uid
    body = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="Body", mesh=_weighted(bp.box()), skin=_skin(bone))
    ).uid
    doc.set_element_mode("face")
    faces = _corner_faces(doc.by_uid(prop).mesh)
    assert len(faces) == 3
    order = (body, prop) if weighted_first else (prop, body)
    for uid in order:
        doc.set_element_sel(uid, el.ElementSel(faces=faces))
    prop_faces = bm.face_count(doc.by_uid(prop).mesh)
    body_mesh = doc.by_uid(body).mesh

    refusals = selection.delete_selected(doc)

    assert len(refusals) == 1 and "Body" in refusals[0]
    assert bm.face_count(doc.by_uid(prop).mesh) == prop_faces - 3
    assert doc.by_uid(body).mesh is body_mesh


# --- clay-mesh-model-02 (clay-21): Separate's per-piece cost does not grow with the mesh ---------


class _CountingStarts(np.ndarray):
    """A ``starts`` array that counts the full-length ``astype`` copies taken of it."""

    full_copies = 0
    threshold = 0

    def astype(self, *args, **kwargs):  # type: ignore[override]
        if self.size >= type(self).threshold:
            type(self).full_copies += 1
        return np.ndarray.astype(self, *args, **kwargs)


def test_separating_many_pieces_of_a_large_mesh_does_not_copy_the_whole_starts_array_per_piece() -> None:  # noqa: E501
    pieces = 200
    quads = 20
    per = 2 * (quads + 1)
    loops = []
    for p in range(pieces):
        for q in range(quads):
            a = p * per + 2 * q
            loops += [a, a + 1, a + 3, a + 2]
    n_faces = pieces * quads
    mesh = bm.Mesh(
        positions=np.random.default_rng(0).random((pieces * per, 3)).astype("f4"),
        loops=np.array(loops, dtype="i4"),
        starts=np.arange(n_faces + 1, dtype="i4") * 4,
        material=np.zeros(n_faces, dtype="i4"),
        smooth=np.zeros(n_faces, dtype=bool),
    )
    spy = mesh.starts.view(_CountingStarts)
    _CountingStarts.full_copies, _CountingStarts.threshold = 0, n_faces
    object.__setattr__(mesh, "starts", spy)  # ``Mesh`` copies on construction; swap it after
    assert isinstance(mesh.starts, _CountingStarts)

    result = separate.by_loose_parts(mesh)

    assert len(result) == pieces
    # one for the per-face counts, one for the corner starts; never one per piece
    assert _CountingStarts.full_copies <= 3, _CountingStarts.full_copies
