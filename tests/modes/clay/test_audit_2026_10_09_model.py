"""The 2026-10-09 Clay audit, the document model: clay-19, clay-20, clay-36, clay-37, clay-46.

Each test's name is its claim; each was run against the unfixed code first.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import legacy, scratch
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import serialize as ser


def _rewrite(data: bytes, edit) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as src, zipfile.ZipFile(out, "w") as dst:
        for name in src.namelist():
            raw = src.read(name)
            if name == ser.SCENE:
                scene = json.loads(raw)
                edit(scene)
                raw = json.dumps(scene).encode()
            dst.writestr(name, raw)
    return out.getvalue()


# --- clay-19 --------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    ["x", None, True, float("nan"), float("inf"), [1, 2], {"a": 1}],
    ids=["string", "null", "bool", "nan", "inf", "list", "dict"],
)
def test_read_rblk_refuses_or_freezes_a_generator_param_that_is_not_a_number(bad) -> None:
    doc = bd.ClayDoc()
    obj = bd.Obj(uid=bd.new_uid(), name="C", mesh=bp.cylinder(), generator="cylinder",
                 params={"radius": 0.5, "height": 1.0, "segments": 8})
    doc.add_object(obj)
    data = ser.rblk_bytes(doc)

    def edit(scene):
        scene["objects"][0]["params"]["segments"] = bad

    try:
        back = ser.read_rblk(_rewrite(data, edit))
    except ValueError:
        return  # a named refusal is allowed
    got = back.objects[0]
    # Frozen: no live generator panel left to raise every frame.
    assert got.generator is None and got.params == {}
    assert any('"C"' in n for n in back.notices)


def test_read_rblk_keeps_a_well_formed_generator_param_set_live() -> None:
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="B", mesh=bp.box(), generator="box",
                          params={"size": (1.0, 2.0, 3.0)}))
    back = ser.read_rblk(ser.rblk_bytes(doc))
    assert back.objects[0].generator == "box"


# --- clay-20 --------------------------------------------------------------------


def test_bake_budget_charges_a_boolean_for_its_targets_triangles(monkeypatch) -> None:
    target = bp.uv_sphere(0.5, 32, 16)
    tiny = bp.box()
    doc = bd.ClayDoc()
    doc.objects.append(bd.Obj(uid=1, name="target", mesh=target))
    stacks = {}
    for i in range(6):
        doc.objects.append(bd.Obj(uid=10 + i, name=f"t{i}", mesh=tiny))
        stacks[10 + i] = (legacy._Modifier(
            id=1, kind="boolean", params=(("operation", 1), ("target", 1))),)
    ran = []
    monkeypatch.setattr(legacy.ops_boolean, "boolean",
                        lambda objs, op, world=None: ran.append(1) or objs[0].mesh)
    # Room for each cutter's own 12 triangles six times over, and for the target
    # only twice: the owner-only charge lets all six run.
    each = legacy.BAKE_WEIGHTS["boolean"] * (
        legacy._triangle_count(tiny) + legacy._triangle_count(target)
    )
    monkeypatch.setattr(legacy, "MAX_BAKE_WORK", each * 2 + 1)
    _, errors = legacy._bake(doc, stacks)
    assert len(ran) == 2
    assert sum(len(v) for v in errors.values()) == 4


# --- clay-36 --------------------------------------------------------------------


def test_migrate_does_not_bake_or_name_a_collider_objects_modifiers() -> None:
    doc = bd.ClayDoc()
    doc.objects.append(bd.Obj(uid=1, name="visible", mesh=bp.box()))
    doc.objects.append(bd.Obj(uid=2, name="hull", mesh=bp.box()))
    mirror = legacy._Modifier(id=1, kind="mirror", params=(("axis", 0), ("weld", 0.0001)))
    fields = {2: legacy.LegacyFields(modifiers=(mirror,), role="collider")}
    notices = legacy.migrate(doc, fields)
    text = "\n".join(notices)
    assert "applied" not in text  # nothing surviving was baked
    assert text.count('"hull"') == 1  # named once, in the collider-removal sentence


# --- clay-37 --------------------------------------------------------------------


def test_transplant_added_objects_do_not_alias_the_scratch_objects() -> None:
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="base", mesh=bp.box()))
    sc = scratch.clone(doc)
    extra = bd.Obj(uid=bd.new_uid(), name="new", mesh=bp.box())
    sc.add_object(extra)
    d = scratch.diff(doc, sc)
    assert scratch.transplant(doc, sc, d)
    landed = doc.by_uid(extra.uid)
    assert landed is not sc.by_uid(extra.uid)
    assert not np.shares_memory(landed.translation, sc.by_uid(extra.uid).translation)
    assert not np.shares_memory(landed.scale, sc.by_uid(extra.uid).scale)


# --- clay-46 --------------------------------------------------------------------


def test_primitives_docstring_names_clay_fifteen_and_mason_twenty_one() -> None:
    head = bp.__doc__ or ""
    assert "twenty-one shapes a user can place" not in head
    assert "fifteen" in head and "twenty-one" in head
    assert "CLAY_GENERATORS" in head
    assert len(bp.CLAY_GENERATOR_NAMES) == 15 and len(bp.GENERATORS) == 21
    source = Path(bp.__file__).read_text(encoding="utf-8")
    assert "studio/modes/clay/ui/props.py" not in source
    assert "``clay/earclip.py``" not in source
