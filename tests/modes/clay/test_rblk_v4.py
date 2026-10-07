"""``.rblk`` version 4: the stack-free document, and a version 3 file still opening.

``data/v3_stack.rblk`` was written by the pre-change build (modifier stacks, a
collider, seams, tags, a lock, a lathe, a reduced-away material), and
``data/v3_stack_expected.npz`` holds what that build's ``ClayDoc.evaluated`` returned
for each object. The reader must bake to exactly those meshes.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from realmspinner.kernels.geom3d import gltf
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import legacy
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import serialize as ser

DATA = Path(__file__).parent / "data"
MESH_FIELDS = ("positions", "loops", "starts", "material", "smooth")


@pytest.fixture(scope="module")
def migrated() -> bd.ClayDoc:
    return ser.read_rblk((DATA / "v3_stack.rblk").read_bytes())


@pytest.fixture(scope="module")
def expected():
    return np.load(DATA / "v3_stack_expected.npz")


def _by_name(doc: bd.ClayDoc) -> dict[str, bd.Obj]:
    return {o.name: o for o in doc.objects}


def _rewrite(data: bytes, edit) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as src, zipfile.ZipFile(out, "w") as dst:
        for name in src.namelist():
            if name == ser.SCENE:
                scene = json.loads(src.read(name))
                edit(scene)
                dst.writestr(name, json.dumps(scene))
            else:
                dst.writestr(name, src.read(name))
    return out.getvalue()


# --- the format ----------------------------------------------------------


def test_the_writer_writes_version_4_and_no_dead_keys() -> None:
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box()))
    with zipfile.ZipFile(io.BytesIO(ser.rblk_bytes(doc))) as zf:
        scene = json.loads(zf.read(ser.SCENE))
    assert scene["version"] == 4
    entry = scene["objects"][0]
    for dead in ("modifiers", "locked", "tags", "seams", "role", "collider_kind"):
        assert dead not in entry


def test_a_v4_document_round_trips_with_no_notices() -> None:
    doc = bd.ClayDoc()
    parent = doc.add_object(bd.Obj(uid=bd.new_uid(), name="P", mesh=bp.box()))
    doc.add_object(
        bd.Obj(
            uid=bd.new_uid(), name="C", mesh=bp.box(), parent=parent.uid, generator="box",
            params={"size": (1.0, 1.0, 1.0)},
        )
    )
    out = ser.read_rblk(ser.rblk_bytes(doc))
    assert out.notices == ()
    assert [(o.name, o.parent, o.generator) for o in out.objects] == [
        ("P", None, None),
        ("C", parent.uid, "box"),
    ]


def test_a_document_from_a_newer_build_is_refused() -> None:
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box()))
    data = _rewrite(ser.rblk_bytes(doc), lambda scene: scene.update(version=ser.VERSION + 1))
    with pytest.raises(ValueError, match="newer version"):
        ser.read_rblk(data)


def test_the_object_no_longer_has_the_dead_fields() -> None:
    obj = bd.Obj(uid=1, name="x", mesh=bp.box())
    for dead in ("modifiers", "locked", "tags", "seams", "role", "collider_kind"):
        assert not hasattr(obj, dead)
    doc = bd.ClayDoc()
    for dead in ("evaluated", "evaluation", "set_modifiers", "apply_modifiers", "set_seams",
                 "add_collider"):
        assert not hasattr(doc, dead)


# --- a version 3 file with everything the stack held --------------------------


def test_a_v3_file_bakes_each_stack_to_the_old_evaluated_mesh(migrated, expected) -> None:
    for obj in migrated.objects:
        for field in MESH_FIELDS:
            got = np.asarray(getattr(obj.mesh, field))
            want = expected[f"{obj.name}.{field}"]
            assert got.shape == want.shape, (obj.name, field)
            assert np.array_equal(got, want), (obj.name, field)


def test_the_collider_is_dropped_and_its_child_lifted_to_the_colliders_parent(migrated) -> None:
    by = _by_name(migrated)
    assert "mirrored Box" not in by
    assert by["under_collider"].parent == by["mirrored"].uid
    # Lifted keeping its place in the world: the collider sat at the identity.
    assert np.allclose(migrated.world_matrix(by["under_collider"].uid)[:3, 3], (0.0, 0.0, 1.0))


def test_an_object_whose_mesh_the_bake_changed_is_frozen(migrated) -> None:
    by = _by_name(migrated)
    assert by["mirrored"].generator is None and by["mirrored"].params == {}
    assert by["booled"].generator is None


def test_an_object_the_bake_left_alone_keeps_its_generator(migrated) -> None:
    by = _by_name(migrated)
    assert by["disabled"].generator == "box"  # only a disabled modifier: nothing baked
    assert by["seamed"].generator == "box"
    assert by["target"].generator == "box"


def test_a_generator_clay_no_longer_offers_is_frozen_but_kept_as_a_mesh() -> None:
    def edit(scene):
        for entry in scene["objects"]:
            if entry["name"] == "lathe":
                del entry["modifiers"]  # so the freeze can only come from the generator rule

    doc = ser.read_rblk(_rewrite((DATA / "v3_stack.rblk").read_bytes(), edit))
    lathe = _by_name(doc)["lathe"]
    assert "lathe" not in bp.CLAY_GENERATOR_NAMES
    assert lathe.generator is None and lathe.params == {}
    assert len(lathe.mesh.positions) > 0
    assert any("no longer builds" in n and '"lathe"' in n for n in doc.notices)


def test_the_notices_name_each_thing_that_changed(migrated) -> None:
    text = "\n".join(migrated.notices)
    assert "modifiers" in text and '"mirrored"' in text and '"booled"' in text
    assert "collision shape" in text and '"mirrored Box"' in text
    assert "Boolean modifier could not run" in text and '"broken"' in text
    assert "Seams, tags and locks" in text and '"seamed"' in text
    assert "Materials were simplified" in text


def test_a_v4_save_of_a_migrated_document_carries_no_notices_back(migrated) -> None:
    again = ser.read_rblk(ser.rblk_bytes(migrated))
    assert again.notices == ()
    assert [o.name for o in again.objects] == [o.name for o in migrated.objects]


# --- materials -----------------------------------------------------------------


def test_materials_are_reduced_to_the_clay_subset(migrated) -> None:
    shiny = migrated.materials[1]
    assert shiny.name == "Shiny"
    assert shiny.base_color_factor == pytest.approx((0.2, 0.4, 0.6, 1.0))
    assert shiny.double_sided is True
    assert shiny.base_color is not None  # the base-colour texture survives
    assert shiny.emissive is None  # the other slots do not
    assert shiny.metallic_factor == 0.0 and shiny.roughness_factor == 0.6
    assert shiny.emissive_factor == (0.0, 0.0, 0.0)
    assert shiny.alpha_mode == "OPAQUE"  # BLEND has no place in Clay


def test_reduce_material_is_idempotent_and_keeps_identity_when_nothing_changes() -> None:
    plain = bd.default_material("Plain")
    assert bd.reduce_material(plain) is plain
    messy = gltf.Material(name="m", metallic_factor=1.0, alpha_mode="MASK")
    once = bd.reduce_material(messy)
    assert once.alpha_mode == "MASK" and once.metallic_factor == 0.0
    assert bd.reduce_material(once) is once


# --- hand-edited stacks ---------------------------------------------------------


def test_a_boolean_that_targets_itself_is_dropped_by_name_not_looped(migrated) -> None:
    def edit(scene):
        for entry in scene["objects"]:
            if entry["name"] == "mirrored":
                entry["modifiers"] = [
                    {"id": 1, "kind": "boolean", "enabled": True,
                     "params": {"target": entry["uid"], "operation": "union"}}
                ]

    data = _rewrite((DATA / "v3_stack.rblk").read_bytes(), edit)
    doc = ser.read_rblk(data)
    assert any("cycle" in n and '"mirrored"' in n for n in doc.notices)


def test_an_unknown_or_malformed_modifier_is_dropped_not_refused() -> None:
    def edit(scene):
        scene["objects"][0]["modifiers"] = [
            {"id": 1, "kind": "teleport", "enabled": True, "params": {}},
            "nonsense",
            {"id": 2, "kind": "weld", "enabled": True, "params": {"distance": None}},
        ]

    data = _rewrite((DATA / "v3_stack.rblk").read_bytes(), edit)
    doc = ser.read_rblk(data)  # must open
    text = "\n".join(doc.notices)
    assert "teleport" in text and "weld" in text


def test_a_long_boolean_target_chain_is_baked_without_recursion(monkeypatch) -> None:
    n = 3_000  # past the interpreter's recursion limit, inside MAX_OBJECTS in spirit
    doc = bd.ClayDoc()
    mesh = bp.box()
    for i in range(n):
        doc.objects.append(bd.Obj(uid=i + 1, name=f"o{i}", mesh=mesh))
    stacks = {
        i + 1: (
            legacy._Modifier(id=1, kind="boolean", params=(("operation", 1), ("target", i + 2))),
        )
        for i in range(n - 1)
    }
    monkeypatch.setattr(legacy.ops_boolean, "boolean", lambda objs, op, world=None: objs[0].mesh)
    meshes, errors = legacy._bake(doc, stacks)
    assert errors == {} and meshes == {}  # the stub returns each base unchanged
