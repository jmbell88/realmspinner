"""The 2026-10-14 audit, fixer F14: ``.rblk`` reader leniencies, ``overlap_faces``'s break."""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import rig, serialize, uvtools


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


def _rigged() -> tuple[bd.ClayDoc, int, int]:
    doc = bd.ClayDoc()
    bone = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Hip", mesh=bp.box(), bone=True)).uid
    skin = rig.Skin((bone,), np.tile(np.eye(4), (1, 1, 1)))
    body = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="Body", mesh=_weighted(bp.box()), skin=skin)
    ).uid
    return doc, bone, body


def _animated() -> tuple[dict[str, bytes], dict, int, int]:
    doc, bone, body = _rigged()
    doc.add_animation("Walk")
    doc.set_key(bone, "translation", 0.0, [0, 0, 0])
    doc.set_key(bone, "translation", 1.0, [0, 1, 0])
    zin = zipfile.ZipFile(io.BytesIO(serialize.rblk_bytes(doc)))
    members = {i.filename: zin.read(i.filename) for i in zin.infolist()}
    return members, json.loads(members["scene.json"]), bone, body


def _rebuild(
    members: dict[str, bytes], scene: dict, renames: dict[str, str] | None = None
) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name, blob in members.items():
            payload = json.dumps(scene) if name == "scene.json" else blob
            z.writestr((renames or {}).get(name, name), payload)
    return out.getvalue()


@pytest.mark.parametrize("version", [3, 4])
def test_a_pre_version_5_file_with_weights_and_no_skin_is_refused_or_stripped_on_open(version):
    # The 2026-10-14 audit, findings clay-document-10 and clay-mesh-model-06: the mesh reader
    # takes joints/weights for every version while the skin is read and the rig rule run only
    # from format 5, so the file opened as a document that could not be saved or journalled.
    # The existing rule for a stray rig key in an older file is to ignore it, so the weights go.
    members, scene, _bone, _body = _animated()
    scene["version"] = version
    doc = serialize.read_rblk(_rebuild(members, scene))
    assert doc.rig_problems() == []
    assert all(obj.mesh.joints is None and obj.mesh.weights is None for obj in doc.objects)
    assert serialize.read_rblk(serialize.rblk_bytes(doc)).rig_problems() == []


def test_a_refused_clay_file_does_not_raise_the_uid_floor():
    # The 2026-10-14 audit, finding clay-document-06: reserve_uid ran per entry, before the
    # file was accepted, and the counter never moves back.
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
    zin = zipfile.ZipFile(io.BytesIO(serialize.rblk_bytes(doc)))
    members = {i.filename: zin.read(i.filename) for i in zin.infolist()}
    scene = json.loads(members["scene.json"])
    old = scene["objects"][0]["uid"]
    huge = 10**30
    scene["objects"][0]["uid"] = huge
    scene["objects"][0]["scale"] = [0, 0, 0]
    before = bd.new_uid()
    with pytest.raises(ValueError):
        serialize.read_rblk(_rebuild(members, scene, {f"meshes/{old}.npz": f"meshes/{huge}.npz"}))
    after = bd.new_uid()
    assert after < 10**12, after
    assert after > before


def test_read_rblk_refuses_a_uid_beyond_what_json_numbers_hold_exactly():
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
    zin = zipfile.ZipFile(io.BytesIO(serialize.rblk_bytes(doc)))
    members = {i.filename: zin.read(i.filename) for i in zin.infolist()}
    scene = json.loads(members["scene.json"])
    old = scene["objects"][0]["uid"]
    huge = 10**30
    scene["objects"][0]["uid"] = huge
    with pytest.raises(ValueError, match="uid"):
        serialize.read_rblk(_rebuild(members, scene, {f"meshes/{old}.npz": f"meshes/{huge}.npz"}))


def test_read_rblk_refuses_an_animation_that_declares_more_channels_than_bones():
    # The 2026-10-14 audit, finding clay-document-07: " 1" and "1 " all int() to one bone, so
    # the loop's work scaled with the file's key count, not the bone cap.
    members, scene, bone, _body = _animated()
    channels = {}
    for a in range(20):
        for b in range(20):
            channels[" " * a + str(bone) + " " * b] = ["rotation"]
    assert len(channels) > rig.MAX_BONES
    scene["rig"]["animations"][0]["channels"] = channels
    with pytest.raises(ValueError, match="channels"):
        serialize.read_rblk(_rebuild(members, scene))


def test_read_rblk_refuses_a_bone_with_more_properties_than_exist():
    members, scene, bone, _body = _animated()
    scene["rig"]["animations"][0]["channels"] = {str(bone): ["rotation"] * 50}
    with pytest.raises(ValueError, match="propert"):
        serialize.read_rblk(_rebuild(members, scene))


@pytest.mark.parametrize("field", ["bone", "visible"])
@pytest.mark.parametrize("value", ["false", 1, 0, None])
def test_read_rblk_refuses_a_bone_flag_that_is_not_a_boolean(field, value):
    # The 2026-10-14 audit, finding clay-document-09: bool("false") is True.
    members, scene, _bone, body = _animated()
    scene["objects"][1][field] = value
    with pytest.raises(ValueError, match=field):
        serialize.read_rblk(_rebuild(members, scene))


def test_read_rblk_refuses_a_loop_flag_that_is_not_a_boolean():
    members, scene, _bone, _body = _animated()
    scene["rig"]["animations"][0]["loop"] = "false"
    with pytest.raises(ValueError, match="loop"):
        serialize.read_rblk(_rebuild(members, scene))


def test_a_real_boolean_flag_still_reads():
    members, scene, _bone, _body = _animated()
    scene["objects"][1]["visible"] = False
    scene["rig"]["animations"][0]["loop"] = False
    doc = serialize.read_rblk(_rebuild(members, scene))
    assert doc.objects[1].visible is False
    assert doc.rig_state.animations[0].loop is False


def _stacked_layout() -> bm.Mesh:
    positions: list[list[float]] = []
    faces: list[list[int]] = []
    uvs: list[list[list[float]]] = []

    def add_tri(uv):
        base = len(positions)
        for _ in range(3):
            positions.append([float(len(positions)), 0.0, 0.0])
        faces.append([base, base + 1, base + 2])
        uvs.append(uv)

    for k in range(100):
        e = 0.001 * k
        add_tri([[0.0 + e, 0.0], [1.0, 0.0 + e], [0.0, 1.0]])
    for i in range(30):
        for j in range(30):
            if i >= 28 and j >= 28:
                continue
            x, y = 0.02 + i * 0.03, 0.02 + j * 0.03
            add_tri([[x, y], [x + 0.01, y], [x, y + 0.01]])
    add_tri([[0.95, 0.95], [0.99, 0.95], [0.95, 0.99]])
    add_tri([[0.95, 0.95], [0.99, 0.95], [0.95, 0.99]])
    return bm.from_faces(positions, faces, uv=uvs)


def test_overlap_faces_refuses_or_finds_every_overlap_when_the_pair_count_breaks_early():
    # The 2026-10-14 audit, finding clay-texturing-02: the early break handed the unique-pair
    # ceiling a partial list, so the two stacked triangles in a late bucket read as clean.
    mesh = _stacked_layout()
    try:
        out = uvtools.overlap_faces(mesh)
    except uvtools.OpError:
        return
    assert out[-1] and out[-2]
    assert int(out[:100].sum()) == 100


def test_overlap_faces_still_answers_a_small_layout():
    mesh = bm.from_faces(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 0, 0], [6, 0, 0], [5, 1, 0]],
        [[0, 1, 2], [3, 4, 5]],
        uv=[[[0.1, 0.1], [0.4, 0.1], [0.1, 0.4]], [[0.1, 0.1], [0.4, 0.1], [0.1, 0.4]]],
    )
    assert uvtools.overlap_faces(mesh).tolist() == [True, True]

