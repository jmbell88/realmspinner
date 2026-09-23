"""Regression tests for the 2026-09-23 audit's readiness/glTF findings.

Four rows:

- clay-07: ``readiness.validate()`` evaluated every visible object's whole
  modifier stack (``doc.evaluated`` -> ``modifiers.evaluate``) before it
  checked ``MAX_VALIDATE_CORNERS`` -- the same "evaluate before refusing"
  shape ``analyze.py`` fixed twice (clay-22, clay-19). A document already too
  big before any modifier runs must be refused before paying to evaluate one.
- clay-08: both of ``validate()``'s document-wide ceilings
  (``MAX_VALIDATE_OBJECTS``, ``MAX_VALIDATE_CORNERS``) excluded collider-role
  objects entirely, so a document made mostly or entirely of colliders sailed
  through with no refusal, however many colliders it had, even though every
  one is still evaluated and walked by the ``collider_*`` checks.
- clay-09: a non-dict ``accessors``/``bufferViews``/``images``/``textures``
  entry in ``kernels/geom3d/gltf.py`` raised a bare ``TypeError``/
  ``AttributeError`` instead of this loader's own named ``ValueError``.
- clay-10: ``pbrMetallicRoughness`` and ``attributes`` were used as mappings
  with no isinstance check, unlike the sibling sub-block fallback
  ``camera()``/``light()`` already use.
"""

from __future__ import annotations

import struct
from dataclasses import replace

import numpy as np
import pytest

from realmspinner.kernels.geom3d import gltf
from realmspinner.kernels.geom3d.glbio import rebuild_glb
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import modifiers as mods
from realmspinner.kernels.mesh import primitives as prim
from realmspinner.kernels.mesh import readiness


def _glb(gltf_json: dict, binary: bytes) -> bytes:
    """Wrap a JSON chunk and a BIN chunk as a GLB, the way an exporter would
    -- the same helper ``test_audit_2026_09_15_gltf.py`` uses."""
    binary += b"\x00" * (-len(binary) % 4)
    chunk = struct.pack("<II", len(binary), 0x004E4942) + binary
    header = struct.pack("<III", 0x46546C67, 2, 0)
    return rebuild_glb(header, gltf_json, chunk)


def _obj(mesh, *, translation=(0.0, 0.0, 0.0), **kwargs):
    return bd.Obj(uid=bd.new_uid(), name="Obj", mesh=mesh, translation=translation, **kwargs)


def _grounded_box(**kwargs):
    mesh = replace(prim.box(), uv=None)
    return _obj(mesh, translation=(0.0, 0.5, 0.0), **kwargs)


def _doc(*objects, materials=None):
    return bd.ClayDoc(objects=list(objects), materials=materials)


# --- clay-07: readiness evaluates before refusing -----------------------------


def test_readiness_validate_refuses_before_evaluating_modifier_stacks(monkeypatch):
    """The 2026-09-23 audit's clay-07, reproduced upstream by
    ``clay-mesh-model-01.py``: a single box (24 unevaluated corners) with a
    mirror modifier, ``MAX_VALIDATE_CORNERS`` patched to 1 -- below even the
    box's own *unevaluated* corner count, so an implementation that checks
    cheap, unevaluated info first refuses without ever touching the modifier
    stack. Before this fix, ``modifiers.evaluate`` ran (and the call still
    raised, only after paying for it)."""
    calls = {"n": 0}
    real_evaluate = mods.evaluate

    def _spy_evaluate(doc, uid):
        calls["n"] += 1
        return real_evaluate(doc, uid)

    monkeypatch.setattr(mods, "evaluate", _spy_evaluate)
    monkeypatch.setattr(readiness, "MAX_VALIDATE_CORNERS", 1)

    mesh = replace(prim.box(), uv=None)
    obj = _obj(
        mesh, translation=(0.0, 0.5, 0.0), modifiers=(mods.make("mirror", id=1),),
    )
    doc = _doc(obj)

    with pytest.raises(el.OpError, match=f"{readiness.MAX_VALIDATE_CORNERS:,}"):
        readiness.validate(doc)

    assert calls["n"] == 0, (
        "modifiers.evaluate() ran before the corner ceiling refused the "
        "document -- the cost the ceiling exists to avoid was paid anyway"
    )


# --- clay-08: readiness and colliders -----------------------------------------


def test_readiness_validate_refuses_a_document_past_the_ceiling_in_colliders():
    """The 2026-09-23 audit's clay-08, reproduced upstream by
    ``clay-mesh-model-02.py``: both ``MAX_VALIDATE_OBJECTS`` and
    ``MAX_VALIDATE_CORNERS`` used to count only non-collider (render)
    objects, so a document of 5x ``MAX_VALIDATE_OBJECTS`` collider objects
    (plus one ordinary render object) sailed through with no refusal at all,
    however many colliders it had -- despite every one of them being
    evaluated (``doc.evaluated``) and walked by ``collider_triangles``/
    ``collider_convex``."""
    mesh = replace(prim.box(), uv=None)
    n = readiness.MAX_VALIDATE_OBJECTS * 5
    colliders = [
        bd.Obj(
            uid=bd.new_uid(), name=f"Collider{i}", mesh=mesh,
            translation=(0.0, 0.5, 0.0), scale=(1.0, 1.0, 1.0),
            material=0, visible=True, role="collider", collider_kind="box",
        )
        for i in range(n)
    ]
    render_obj = _grounded_box()
    doc = _doc(render_obj, *colliders)

    with pytest.raises(el.OpError, match=f"{readiness.MAX_VALIDATE_OBJECTS:,}"):
        readiness.validate(doc)


def test_readiness_validate_refuses_a_document_of_only_oversized_colliders_on_corners():
    """The corner-ceiling half of clay-08: a document with few enough
    collider *objects* to clear ``MAX_VALIDATE_OBJECTS`` but whose collider
    meshes alone sum past ``MAX_VALIDATE_CORNERS`` must still refuse --
    before this fix, collider corners were never summed into that ceiling at
    all, no matter how large."""
    mesh = replace(prim.box(), uv=None)  # 24 corners
    collider = bd.Obj(
        uid=bd.new_uid(), name="BigCollider", mesh=mesh,
        translation=(0.0, 0.5, 0.0), scale=(1.0, 1.0, 1.0),
        material=0, visible=True, role="collider", collider_kind="box",
    )
    doc = _doc(collider)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(readiness, "MAX_VALIDATE_CORNERS", 10)
        with pytest.raises(el.OpError, match=f"{readiness.MAX_VALIDATE_CORNERS:,}"):
            readiness.validate(doc)


# --- clay-09: GLB loader entry types -------------------------------------------


def _minimal_doc(**accessor_overrides):
    n = 3
    positions = np.zeros((n, 3), dtype="<f4")
    binary = positions.tobytes()
    accessor = {"bufferView": 0, "componentType": 5126, "count": n, "type": "VEC3"}
    accessor.update(accessor_overrides)
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": positions.nbytes}],
        "accessors": [accessor],
    }
    return doc, binary


def test_a_non_dict_accessor_bufferview_image_or_texture_entry_is_refused_with_a_value_error():
    """clay-09 (2026-09-23 audit): each of these arrays used to be indexed
    straight into with no check that the entry itself was an object, so a
    GLB that is legal JSON but puts ``null``/a string/a list in one of them
    raised a bare ``TypeError``/``AttributeError`` instead of this loader's
    own named ``ValueError`` (the same rule ``_check_dict_entry`` already
    states and enforces for ``nodes``/``meshes[].primitives``/``materials``/
    ``skins`` entries)."""
    # accessors[0] is not a dict.
    doc, binary = _minimal_doc()
    doc["accessors"] = [None]
    with pytest.raises(ValueError, match="accessor 0"):
        gltf.load(_glb(doc, binary))

    # bufferViews[0], referenced by an accessor, is not a dict.
    doc, binary = _minimal_doc()
    doc["bufferViews"] = ["not a dict"]
    with pytest.raises(ValueError, match="bufferView 0"):
        gltf.load(_glb(doc, binary))

    # textures[0], referenced by a material, is not a dict.
    doc, binary = _minimal_doc()
    doc["materials"] = [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}]
    doc["meshes"][0]["primitives"][0]["material"] = 0
    doc["textures"] = [42]
    with pytest.raises(ValueError, match="texture 0"):
        gltf.load(_glb(doc, binary))

    # images[0], referenced by a texture, is not a dict.
    doc, binary = _minimal_doc()
    doc["materials"] = [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}]
    doc["meshes"][0]["primitives"][0]["material"] = 0
    doc["textures"] = [{"source": 0}]
    doc["images"] = [[]]
    with pytest.raises(ValueError, match="image 0"):
        gltf.load(_glb(doc, binary))


# --- clay-10: GLB loader sub-objects -------------------------------------------


def test_a_non_dict_pbr_or_attributes_block_is_refused_not_a_bare_typeerror():
    """clay-10 (2026-09-23 audit): ``pbrMetallicRoughness`` and
    ``attributes`` were read with ``.get(key, {})`` and used as mappings with
    no isinstance check, unlike the sibling sub-block fallback
    ``camera()``/``light()`` already use for ``perspective``/``spot``. A
    material whose "pbrMetallicRoughness" is legal JSON but not an object
    used to raise a bare ``AttributeError`` off ``pbr.get(...)``; a primitive
    whose "attributes" is a list raised a bare ``TypeError`` off
    ``attrs["POSITION"]``. Both must now behave exactly as if the field were
    simply absent."""
    # pbrMetallicRoughness is a list, not a dict -- must not raise at all,
    # the same as an absent one (falls back to {}, same as camera()/light()).
    doc, binary = _minimal_doc()
    doc["materials"] = [{"pbrMetallicRoughness": ["not", "a", "dict"]}]
    doc["meshes"][0]["primitives"][0]["material"] = 0
    model = gltf.load(_glb(doc, binary))
    assert model.meshes[0][0].material is not None

    # attributes is a list, not a dict -- POSITION is "missing" the same way
    # an ordinary primitive with no POSITION at all is refused.
    doc, binary = _minimal_doc()
    doc["meshes"][0]["primitives"][0]["attributes"] = ["POSITION"]
    with pytest.raises(ValueError, match="POSITION"):
        gltf.load(_glb(doc, binary))
