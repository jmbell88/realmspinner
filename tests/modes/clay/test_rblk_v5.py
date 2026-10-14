"""``.rblk`` format 5: bones and weighted skins on disk.

Tranche 2 of the 2026-10-13 Clay rig plan. Version 5 is a version of its own rather than keys
added to 4, so that a build which knows nothing of rigs *refuses* a rigged file instead of
dropping the rig; everything is omitted at its default, so an unrigged document differs from a
version 4 file in the number alone. The reader and the writer share one rig rule.
"""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import rig
from realmspinner.kernels.mesh import serialize as ser

from ._rigkit import rigged_doc, skin_for, weighted


def _scene(data: bytes) -> dict:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        return json.loads(zf.read(ser.SCENE))


def _rewrite(data: bytes, edit) -> bytes:
    """*data* with ``scene.json`` passed through *edit* (a function of the parsed dict)."""
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as src, zipfile.ZipFile(out, "w") as dst:
        for info in src.infolist():
            body = src.read(info.filename)
            if info.filename == ser.SCENE:
                scene = json.loads(body)
                edit(scene)
                body = json.dumps(scene).encode()
            dst.writestr(info, body)
    return out.getvalue()


def test_the_version_is_5_and_an_unrigged_document_is_v4_shaped_apart_from_it() -> None:
    assert ser.VERSION == 5
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box()))
    entry = _scene(ser.rblk_bytes(doc))["objects"][0]
    assert "bone" not in entry and "skin" not in entry
    with zipfile.ZipFile(io.BytesIO(ser.rblk_bytes(doc))) as zf:
        member = next(n for n in zf.namelist() if n.startswith("meshes/"))
        names = zipfile.ZipFile(io.BytesIO(zf.read(member))).namelist()
    assert "joints.npy" not in names and "weights.npy" not in names


def test_a_rig_round_trips_exactly() -> None:
    doc, bone, body = rigged_doc()
    # A non-trivial inverse bind matrix, so a reader that recomputed or reset it fails.
    ibm = np.eye(4)
    ibm[:3, 3] = [0.25, -1.5, 3.0]
    doc.by_uid(body).skin = rig.Skin((bone,), ibm[None])
    out = ser.read_rblk(ser.rblk_bytes(doc))

    assert [o.bone for o in out.objects] == [True, False]
    skin = out.by_uid(body).skin
    assert skin.joints == (bone,)
    assert np.array_equal(skin.inverse_bind[0], ibm)
    mesh = out.by_uid(body).mesh
    assert np.array_equal(mesh.joints, doc.by_uid(body).mesh.joints)
    assert np.array_equal(mesh.weights, doc.by_uid(body).mesh.weights)
    assert out.by_uid(bone).skin is None and out.by_uid(bone).mesh.joints is None
    assert out.rig_problems() == [] and out.notices == ()


def test_a_rigged_document_saves_byte_identically_twice() -> None:
    doc, bone, body = rigged_doc()
    first = ser.rblk_bytes(doc)
    assert first == ser.rblk_bytes(doc)
    assert ser.rblk_bytes(ser.read_rblk(first)) == first, (
        "a save of a reopened rig is the same file"
    )


def test_a_version_4_file_opens_with_no_notice_and_ignores_a_stray_rig_key() -> None:
    doc, bone, body = rigged_doc()
    old = _rewrite(ser.rblk_bytes(doc), lambda scene: scene.__setitem__("version", 4))
    out = ser.read_rblk(old)
    assert out.notices == ()
    assert not any(o.bone for o in out.objects) and all(o.skin is None for o in out.objects)


def test_a_newer_file_is_refused_by_this_build() -> None:
    doc, bone, body = rigged_doc()
    newer = _rewrite(ser.rblk_bytes(doc), lambda scene: scene.__setitem__("version", 6))
    with pytest.raises(ValueError, match="newer version"):
        ser.read_rblk(newer)


def _twice(skin: dict) -> None:
    skin["joints"] = skin["joints"] * 2
    skin["inverse_bind"] = skin["inverse_bind"] * 2


def _entry(scene: dict, name: str) -> dict:
    return next(e for e in scene["objects"] if e["name"] == name)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda s: _entry(s, "Hip").__setitem__("bone", False), "not a bone"),
        (lambda s: _entry(s, "Body")["skin"]["joints"].__setitem__(0, 424242), "does not exist"),
        (
            lambda s: _entry(s, "Body")["skin"].__setitem__("inverse_bind", [[0.0] * 15]),
            "unreadable",
        ),
        (lambda s: _twice(_entry(s, "Body")["skin"]), "twice"),
        (lambda s: _entry(s, "Body").__setitem__("skin", [1, 2]), "not an object"),
        (lambda s: _entry(s, "Body").pop("skin"), "weights but no skin"),
    ],
)
def test_the_reader_refuses_a_rig_that_does_not_hold_together_by_name(edit, message: str) -> None:
    doc, bone, body = rigged_doc()
    broken = _rewrite(ser.rblk_bytes(doc), edit)
    with pytest.raises(ValueError, match=message):
        ser.read_rblk(broken)


def test_the_reader_refuses_a_joint_index_past_its_skin() -> None:
    doc, bone, body = rigged_doc()
    doc.by_uid(body).mesh = weighted(bp.box(), 2)  # indexes joint 1
    doc.by_uid(body).skin = skin_for(bone, bone + 1000)
    doc.add_object(bd.Obj(uid=bone + 1000, name="Knee", mesh=bp.box(), bone=True))
    data = ser.rblk_bytes(doc)

    def drop_second_joint(scene: dict) -> None:
        skin = _entry(scene, "Body")["skin"]
        skin["joints"] = skin["joints"][:1]
        skin["inverse_bind"] = skin["inverse_bind"][:1]

    with pytest.raises(ValueError, match="does not have"):
        ser.read_rblk(_rewrite(data, drop_second_joint))


def test_the_writer_refuses_what_the_reader_would_refuse() -> None:
    doc, bone, body = rigged_doc()
    doc.by_uid(body).mesh = weighted(bp.box(), 2)
    with pytest.raises(ValueError, match="weights a joint its skin does not have"):
        ser.rblk_bytes(doc)

    doc, bone, body = rigged_doc()
    doc.by_uid(bone).bone = False
    with pytest.raises(ValueError, match="which is not a bone"):
        ser.rblk_bytes(doc)

    doc, bone, body = rigged_doc()
    doc.by_uid(body).skin = None
    with pytest.raises(ValueError, match="weights but no skin"):
        ser.rblk_bytes(doc)


def _bones_and_one_plain(count: int) -> bd.ClayDoc:
    objects = [
        bd.Obj(uid=bd.new_uid(), name=f"B{i}", mesh=bp.box(), bone=i < count) for i in range(65)
    ]
    return bd.ClayDoc(objects=objects)


def test_both_halves_refuse_more_than_the_bone_cap() -> None:
    with pytest.raises(ValueError, match="past the 64"):
        ser.rblk_bytes(_bones_and_one_plain(65))
    data = ser.rblk_bytes(_bones_and_one_plain(rig.MAX_BONES))
    assert len(rig.bones(ser.read_rblk(data).objects)) == rig.MAX_BONES
    with pytest.raises(ValueError, match="past the 64"):
        ser.read_rblk(_rewrite(data, lambda scene: _entry(scene, "B64").__setitem__("bone", True)))


def test_skin_joints_survive_a_uid_that_is_not_the_first_bone() -> None:
    doc = bd.ClayDoc()
    first = doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box(), bone=True)).uid
    second = doc.add_object(bd.Obj(uid=bd.new_uid(), name="B", mesh=bp.box(), bone=True)).uid
    body = doc.add_object(
        bd.Obj(
            uid=bd.new_uid(), name="Body", mesh=weighted(bp.box(), 2), skin=skin_for(second, first)
        )
    ).uid
    out = ser.read_rblk(ser.rblk_bytes(doc))
    assert out.by_uid(body).skin.joints == (second, first), "skin joint order is a skin's own"
