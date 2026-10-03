"""The 2026-10-03 audit's mason-07 and mason-08, built rather than corrected.

mason-07: manual 31 and Chapter 17 promise a material override ("retint it")
that nothing in Mason's UI authored. mason-08: nothing could add a user
property or edit a prefab template, though both are documented.
"""

from __future__ import annotations

import numpy as np
import pytest
from _ui_context import imgui_context

from realmspinner.kernels.geom3d import gltf
from realmspinner.studio import probe
from realmspinner.studio.modes.mason import state as mason_state
from realmspinner.studio.modes.mason.engine import document as md
from realmspinner.studio.modes.mason.engine import nodes as nd
from realmspinner.studio.modes.mason.engine import refs as mrefs
from realmspinner.studio.modes.mason.engine import serialize as ser
from realmspinner.studio.modes.mason.engine import terrain as tr
from realmspinner.studio.modes.mason.ui.panes import props as mason_props


def _mesh(doc: md.MasonDoc, name: str = "Crate") -> nd.MeshNode:
    node = nd.MeshNode(uid=nd.new_uid(), name=name, ref=mrefs.primitive_ref("box", {}))
    doc.add_node(node)
    return node


def _labels(monkeypatch, draw) -> list[str]:
    with imgui_context(monkeypatch) as imgui:
        probe.begin_frame()
        imgui.new_frame()
        imgui.set_next_window_size((320.0, 900.0))
        imgui.begin("##host")
        draw()
        imgui.end()
        imgui.end_frame()
        return [c.label for c in probe.FRAME_CONTROLS]


def _has(labels: list[str], fragment: str) -> bool:
    return any(fragment in label for label in labels)


# --- mason-07 ----------------------------------------------------------------


def test_the_mesh_properties_block_offers_a_material_override_the_manual_promises(
    monkeypatch,
) -> None:
    doc = md.MasonDoc()
    node = _mesh(doc)
    node.material = gltf.Material(name="paint")

    labels = _labels(monkeypatch, lambda: mason_props._mesh_block(doc, node))

    assert _has(labels, "mmatcol")
    assert _has(labels, "mmatmet")
    assert _has(labels, "mmatrough")


def test_a_mesh_without_an_override_offers_only_the_switch(monkeypatch) -> None:
    doc = md.MasonDoc()
    node = _mesh(doc)

    labels = _labels(monkeypatch, lambda: mason_props._mesh_block(doc, node))

    assert not _has(labels, "mmatcol")


def test_the_terrain_block_offers_the_grounds_material(monkeypatch) -> None:
    doc = md.MasonDoc()
    heights = np.zeros((5, 5), dtype=np.float32)
    doc.set_terrain(tr.Terrain(heights=heights, size_x=4.0, size_z=4.0, material=gltf.Material()))

    labels = _labels(monkeypatch, lambda: mason_props._terrain_block(doc))

    assert _has(labels, "mmatcol")


def test_retinted_returns_a_new_material_and_leaves_the_shared_one_alone() -> None:
    shared = gltf.Material(name="shared", base_color_factor=(1.0, 1.0, 1.0, 1.0))

    out = mason_props.retinted(shared, colour=(0.2, 0.4, 2.0, 1.0), roughness=-1.0)

    assert out is not shared
    assert shared.base_color_factor == (1.0, 1.0, 1.0, 1.0)
    assert out.base_color_factor == (0.2, 0.4, 1.0, 1.0)  # clamped to 0..1
    assert out.roughness_factor == 0.0


def test_a_material_override_undoes_and_survives_a_save_and_reload() -> None:
    doc = md.MasonDoc()
    node = _mesh(doc)
    override = mason_props.retinted(
        mason_props._new_override(), colour=(0.9, 0.1, 0.1, 1.0)
    )

    assert doc.set_props(node.uid, material=override)
    reread = ser.read_rscn(ser.rscn_bytes(doc))
    kept = reread.node(node.uid)
    assert kept is not None and kept.material is not None
    assert kept.material.base_color_factor == pytest.approx((0.9, 0.1, 0.1, 1.0))

    assert doc.undo()
    assert node.material is None
    assert doc.redo()
    assert node.material is override


# --- mason-08 ----------------------------------------------------------------


def test_properties_offers_a_way_to_add_a_user_property_and_to_edit_a_template(
    monkeypatch,
) -> None:
    doc = md.MasonDoc()
    node = _mesh(doc)
    state = mason_state.MasonState()

    labels = _labels(monkeypatch, lambda: mason_props._properties(doc, node, state))

    assert _has(labels, "mpropadd")

    template = nd.MeshNode(uid=nd.new_uid(), name="T", ref=mrefs.primitive_ref("box", {}))
    template.material = gltf.Material(name="t")
    doc.prefabs["Prop"] = template
    labels = _labels(monkeypatch, lambda: mason_props._template_block(doc, "Prop", template))
    assert _has(labels, "mmatcol")


def test_set_user_property_adds_edits_removes_and_undoes_each_as_one_step() -> None:
    doc = md.MasonDoc()
    node = _mesh(doc)

    assert doc.set_user_property(node.uid, " breakable ", "yes")
    assert node.properties == {"breakable": "yes"}
    assert doc.set_user_property(node.uid, "breakable", "no")
    assert node.properties == {"breakable": "no"}
    assert not doc.set_user_property(node.uid, "breakable", "no")  # no-op pushes nothing
    assert doc.set_user_property(node.uid, "breakable", None)
    assert node.properties == {}

    assert doc.undo()
    assert node.properties == {"breakable": "no"}
    assert doc.undo()
    assert node.properties == {"breakable": "yes"}
    assert doc.undo()
    assert node.properties == {}


def test_set_user_property_refuses_a_blank_name_and_an_oversized_value() -> None:
    doc = md.MasonDoc()
    node = _mesh(doc)

    with pytest.raises(ValueError):
        doc.set_user_property(node.uid, "   ", "x")
    with pytest.raises(ValueError):
        doc.set_user_property(node.uid, "k", "v" * (md.MAX_PROPERTY_VALUE + 1))
    assert node.properties == {}


def test_a_user_property_survives_a_save_and_reload() -> None:
    doc = md.MasonDoc()
    node = _mesh(doc)
    doc.set_user_property(node.uid, "loot", "gold")

    kept = ser.read_rscn(ser.rscn_bytes(doc)).node(node.uid)

    assert kept is not None and kept.properties == {"loot": "gold"}


def _doc_with_three_instances() -> tuple[md.MasonDoc, list[nd.PrefabNode]]:
    doc = md.MasonDoc()
    source = _mesh(doc, "Lamp")
    doc.define_prefab("Lamp", source)
    instances = []
    for x in range(3):
        inst = nd.PrefabNode(uid=nd.new_uid(), name=f"Lamp{x}", template="Lamp")
        inst.translation = np.array([float(x), 0.0, 0.0])
        doc.add_node(inst)
        instances.append(inst)
    return doc, instances


def test_set_template_material_retints_every_instance_in_one_undoable_step() -> None:
    from realmspinner.studio.modes.mason.engine import scene as msc

    doc, instances = _doc_with_three_instances()
    red = gltf.Material(name="red", base_color_factor=(1.0, 0.0, 0.0, 1.0))

    assert doc.set_template_material("Lamp", red)

    mats = [
        p.material for p in msc.resolve(doc) if getattr(p, "prefab", None) == "Lamp"
    ]
    assert len(mats) == 3 and all(m is red for m in mats)
    # instances keep only their own transform: positions untouched
    assert [float(i.translation[0]) for i in instances] == [0.0, 1.0, 2.0]

    assert doc.undo()
    assert doc.prefabs["Lamp"].material is None
    assert doc.redo()
    assert doc.prefabs["Lamp"].material is red


def test_set_template_material_refuses_a_group_template_and_a_missing_one() -> None:
    doc = md.MasonDoc()
    group = nd.GroupNode(uid=nd.new_uid(), name="Wall")
    doc.add_node(group)
    doc.define_prefab("Wall", group)

    with pytest.raises(TypeError):
        doc.set_template_material("Wall", gltf.Material())
    with pytest.raises(KeyError):
        doc.set_template_material("Nope", gltf.Material())
