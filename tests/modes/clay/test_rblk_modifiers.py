"""``.rblk`` version 3: modifier stacks on disk.

Written the same way as ``test_serialize.py``'s own version-boundary tests --
a real document round-tripped, and an archive hand-rewritten through
``_rewrite`` to simulate a file this build did not write, because the reader
has to survive both an older build's honest output and a byte a person typed.
"""

from __future__ import annotations

import json
import zipfile
from io import BytesIO

import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import modifiers as mod
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import serialize as ser


def _doc_with_modifiers() -> bd.ClayDoc:
    doc = bd.ClayDoc()
    a = doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box()))
    b = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="B", mesh=bp.box(), translation=(3.0, 0.0, 0.0))
    )
    doc.set_modifiers(
        a.uid,
        (
            mod.make("mirror", {"axis": "Z", "weld": 0.001}, id=1),
            mod.make("array", {"count": 3, "offset_x": 2.0}, id=2),
            mod.make("boolean", {"target": b.uid, "operation": "intersection"}, id=5),
        ),
    )
    return doc


def _rewrite(data: bytes, edit) -> bytes:
    """The same archive with ``edit`` applied to its ``scene.json`` dict."""
    out = BytesIO()
    with zipfile.ZipFile(BytesIO(data)) as src, zipfile.ZipFile(out, "w") as dst:
        for name in src.namelist():
            if name == ser.SCENE:
                scene = json.loads(src.read(name))
                edit(scene)
                dst.writestr(name, json.dumps(scene))
            else:
                dst.writestr(name, src.read(name))
    return out.getvalue()


# --- the round trip ----------------------------------------------------


def test_v3_round_trip_carries_every_modifier_field() -> None:
    doc = _doc_with_modifiers()
    stack = doc.by_uid(doc.objects[0].uid).modifiers

    out = ser.read_rblk(ser.rblk_bytes(doc))
    restored = out.by_uid(doc.objects[0].uid).modifiers

    assert restored == stack
    assert [m.kind for m in restored] == ["mirror", "array", "boolean"]


def test_the_disabled_flag_round_trips() -> None:
    from dataclasses import replace

    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box()))
    doc.set_modifiers(obj.uid, (replace(mod.make("weld", id=1), enabled=False),))

    out = ser.read_rblk(ser.rblk_bytes(doc))
    assert out.objects[0].modifiers[0].enabled is False


def test_an_object_with_no_modifiers_writes_no_modifiers_key() -> None:
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box()))
    scene = json.loads(ser.scene_json(doc))
    assert "modifiers" not in scene["objects"][0]


def test_a_textureless_unmodified_document_is_byte_identical_when_repeated() -> None:
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box()))
    assert ser.rblk_bytes(doc) == ser.rblk_bytes(doc)


def test_a_modified_document_is_still_byte_identical_when_repeated() -> None:
    doc = _doc_with_modifiers()
    assert ser.rblk_bytes(doc) == ser.rblk_bytes(doc)


def test_version_is_written_as_3() -> None:
    doc = _doc_with_modifiers()
    scene = json.loads(ser.scene_json(doc))
    assert scene["version"] == 3
    assert ser.VERSION == 3


# --- backward compatibility ---------------------------------------------


def test_a_version_2_file_still_opens_with_no_modifiers() -> None:
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box()))
    data = _rewrite(ser.rblk_bytes(doc), lambda scene: scene.__setitem__("version", 2))

    out = ser.read_rblk(data)
    assert out.objects[0].modifiers == ()


# --- refusals: half-read is worse than refused -----------------------------


def test_an_unknown_modifier_kind_is_refused() -> None:
    def bad(scene: dict) -> None:
        scene["objects"][0]["modifiers"] = [
            {"id": 1, "kind": "not-a-kind", "enabled": True, "params": {}}
        ]

    with pytest.raises(ValueError, match="modifier"):
        ser.read_rblk(_rewrite(ser.rblk_bytes(_doc_with_modifiers()), bad))


def test_an_unknown_modifier_param_is_refused() -> None:
    def bad(scene: dict) -> None:
        scene["objects"][0]["modifiers"] = [
            {"id": 1, "kind": "mirror", "enabled": True, "params": {"bogus": 1}}
        ]

    with pytest.raises(ValueError, match="modifier"):
        ser.read_rblk(_rewrite(ser.rblk_bytes(_doc_with_modifiers()), bad))


def test_a_modifiers_field_that_is_not_a_list_is_refused() -> None:
    def bad(scene: dict) -> None:
        scene["objects"][0]["modifiers"] = {"not": "a list"}

    with pytest.raises(ValueError, match="not a list"):
        ser.read_rblk(_rewrite(ser.rblk_bytes(_doc_with_modifiers()), bad))


def test_two_modifiers_sharing_an_id_on_one_object_is_refused() -> None:
    def bad(scene: dict) -> None:
        scene["objects"][0]["modifiers"] = [
            {"id": 1, "kind": "weld", "enabled": True, "params": {}},
            {"id": 1, "kind": "triangulate", "enabled": True, "params": {}},
        ]

    with pytest.raises(ValueError, match="sharing id 1"):
        ser.read_rblk(_rewrite(ser.rblk_bytes(_doc_with_modifiers()), bad))


def test_the_same_id_on_two_different_objects_is_fine() -> None:
    """Ids are unique *within* a stack, not across the document -- each
    object's own modifiers start counting from 1."""
    doc = _doc_with_modifiers()
    out = ser.read_rblk(ser.rblk_bytes(doc))
    a, b = out.objects[0], out.objects[1]
    assert a.modifiers  # A carries the built stack
    assert b.modifiers == ()  # B carries none -- both still open fine


# --- what loads anyway, and evaluates to an error later --------------------


def test_a_boolean_target_naming_an_absent_uid_still_loads() -> None:
    def bad(scene: dict) -> None:
        scene["objects"][0]["modifiers"] = [
            {
                "id": 1,
                "kind": "boolean",
                "enabled": True,
                "params": {"target": 999999, "operation": 1},
            }
        ]

    out = ser.read_rblk(_rewrite(ser.rblk_bytes(_doc_with_modifiers()), bad))
    obj = out.objects[0]
    assert obj.modifiers[0].get("target") == 999999  # loaded, not refused

    ev = out.evaluation(obj.uid)
    assert ev.errors and "no longer exists" in ev.errors[0][1]


def test_an_out_of_range_param_value_loads_clamped() -> None:
    def bad(scene: dict) -> None:
        scene["objects"][0]["modifiers"] = [
            {"id": 1, "kind": "array", "enabled": True, "params": {"count": 99999}}
        ]

    out = ser.read_rblk(_rewrite(ser.rblk_bytes(_doc_with_modifiers()), bad))
    assert out.objects[0].modifiers[0].get("count") == 200  # clamped, not refused
