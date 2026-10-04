"""Regressions for the 2026-10-03 audit's mason-09 .. mason-13 and mason-22
(the Medium findings in Mason's document, serialization, glTF out and ops).

Each test's name is the claim it makes.
"""

from __future__ import annotations

import json
import time
import zipfile
from io import BytesIO
from typing import Any

import numpy as np
import pytest

from realmspinner.kernels.geom3d import gltf
from realmspinner.studio.modes.mason.engine import document as md
from realmspinner.studio.modes.mason.engine import gltfout
from realmspinner.studio.modes.mason.engine import nodes as nd
from realmspinner.studio.modes.mason.engine import ops as mops
from realmspinner.studio.modes.mason.engine import refs as mrefs
from realmspinner.studio.modes.mason.engine import scene as msc
from realmspinner.studio.modes.mason.engine import serialize as ser
from realmspinner.studio.modes.mason.engine import terrain as mterrain
from realmspinner.studio.modes.mason.ui.panes import menu as mason_menu


def _mesh(name: str = "M", **kw: Any) -> nd.MeshNode:
    return nd.MeshNode(uid=nd.new_uid(), name=name, ref=mrefs.primitive_ref("box", {}), **kw)


def _group(name: str = "G", children: list[nd.Node] | None = None) -> nd.GroupNode:
    return nd.GroupNode(uid=nd.new_uid(), name=name, children=list(children or []))


# --- mason-09 ----------------------------------------------------------------


def test_unpack_instance_refuses_a_copy_that_would_nest_past_max_depth(monkeypatch) -> None:
    monkeypatch.setattr(nd, "MAX_DEPTH", 3)
    doc = md.MasonDoc()
    # A template two levels deep (group > group > mesh): its own deepest node
    # sits at depth 2.
    doc.prefabs["tower"] = _group("T", [_group("T1", [_mesh("T2")])])
    outer = doc.add_node(_group("g0"))
    inner = doc.add_node(_group("g1"), parent_uid=outer.uid)
    inst = doc.add_node(
        nd.PrefabNode(uid=nd.new_uid(), name="I", template="tower"), parent_uid=inner.uid
    )  # depth 2; unpacking would land the template's deepest node at depth 4

    with pytest.raises(ValueError, match="deep"):
        doc.unpack_instance(inst.uid)

    # Refused before anything was detached: the instance is still in the tree.
    assert doc.node(inst.uid) is inst
    assert doc.parent_uid_of(inst.uid) == inner.uid


# --- mason-10 ----------------------------------------------------------------


def test_add_node_counts_prefab_template_nodes_against_the_ceiling_the_reader_enforces(
    monkeypatch,
) -> None:
    monkeypatch.setattr(msc, "MAX_PLACED", 20)
    doc = md.MasonDoc(roots=[_mesh(f"m{i}") for i in range(14)])
    # A 5-node template lives outside ``roots`` but ``read_rscn`` charges it
    # against the very same ceiling.
    doc.prefabs["t"] = _group("T", [_mesh(f"t{i}") for i in range(4)])

    # 14 scene + 1 + 5 template = 20: exactly the ceiling, and it reopens.
    doc.add_node(_mesh("fits"))
    ser.read_rscn(ser.rscn_bytes(doc))
    # One more is 21 > 20: the reader would refuse that file, so the writer
    # must too -- through every door.
    with pytest.raises(ValueError, match="MAX_PLACED"):
        doc.add_node(_mesh("one-too-many"))
    with pytest.raises(ValueError, match="MAX_PLACED"):
        doc.add_nodes([_mesh("one-too-many")])


# --- mason-11 ----------------------------------------------------------------


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_set_props_refuses_a_non_finite_light_or_camera_field_rather_than_saving_an_unopenable_scene(  # noqa: E501
    bad: float,
) -> None:
    doc = md.MasonDoc()
    light = doc.add_node(nd.LightNode(uid=nd.new_uid(), name="L", kind="point"))
    camera = doc.add_node(nd.CameraNode(uid=nd.new_uid(), name="C"))
    head = doc.history.head

    for field in ("intensity", "range", "inner_cone_angle", "outer_cone_angle"):
        with pytest.raises(ValueError, match="finite"):
            doc.set_props(light.uid, **{field: bad})
    for field in ("yfov", "znear", "zfar"):
        with pytest.raises(ValueError, match="finite"):
            doc.set_props(camera.uid, **{field: bad})
    with pytest.raises(ValueError, match="finite"):
        doc.set_props(light.uid, color=(1.0, bad, 0.0))
    with pytest.raises(ValueError, match="kind"):
        doc.set_props(light.uid, kind="area")

    # Nothing was written, nothing was recorded, and the scene still reopens.
    assert light.intensity == 1.0 and camera.yfov == pytest.approx(np.radians(60.0))
    assert doc.history.head == head
    ser.read_rscn(ser.rscn_bytes(doc))
    # A finite value still goes through.
    assert doc.set_props(light.uid, intensity=2.5) is True


# --- mason-12 ----------------------------------------------------------------


def _doc_with_a_texture() -> bytes:
    material = gltf.Material(name="m", base_color=(2, 2, bytes(range(16))))
    doc = md.MasonDoc(roots=[_mesh("a", material=material)], materials=[material])
    return ser.rscn_bytes(doc)


def _redeclare_textures(data: bytes, times: int) -> bytes:
    out = BytesIO()
    with zipfile.ZipFile(BytesIO(data)) as src, zipfile.ZipFile(out, "w") as dst:
        for name in src.namelist():
            if name == ser.SCENE:
                scene = json.loads(src.read(name))
                scene["textures"] = scene["textures"][:1] * times
                dst.writestr(name, json.dumps(scene))
            else:
                dst.writestr(name, src.read(name))
    return out.getvalue()


def test_read_rscn_refuses_textures_whose_decoded_bytes_pass_the_document_budget(
    monkeypatch,
) -> None:
    data = _redeclare_textures(_doc_with_a_texture(), 10)
    # A 2x2 RGBA texture decodes to 16 bytes; a budget of three of them.
    monkeypatch.setattr(ser, "MAX_TOTAL_TEXTURE_BYTES", 48, raising=False)
    with pytest.raises(ValueError, match="texture"):
        ser.read_rscn(data)
    # Within the budget the same file shape opens.
    ser.read_rscn(_redeclare_textures(_doc_with_a_texture(), 3))


def test_read_rscn_refuses_more_declared_textures_than_the_reader_cap(monkeypatch) -> None:
    data = _redeclare_textures(_doc_with_a_texture(), 10)
    monkeypatch.setattr(ser, "MAX_DECLARED_TEXTURES", 5, raising=False)
    with pytest.raises(ValueError, match="textures"):
        ser.read_rscn(data)


# --- mason-13 ----------------------------------------------------------------


class _Source:
    rev = 0

    def primitives(self, ref: Any) -> list[gltf.Primitive]:
        return [
            gltf.Primitive(
                positions=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="f4"),
                indices=np.array([0, 1, 2], dtype="u4"),
                material=gltf.Material(name=""),
            )
        ]

    def box(self, ref: Any) -> Any:
        return self.primitives(ref)[0].box()


def _naming_seconds(count: int) -> float:
    doc = md.MasonDoc(roots=[_mesh("Rock") for _ in range(count)])
    best = float("inf")
    for _ in range(2):
        start = time.perf_counter()
        export = gltfout.scene_model(doc, _Source())
        best = min(best, time.perf_counter() - start)
    names = [n.name for n in export.nodes]
    assert len(set(names)) == len(names) == count
    return best


def test_unique_name_cost_does_not_grow_quadratically_with_identical_names() -> None:
    small, large = 1500, 6000
    ratio = _naming_seconds(large) / max(_naming_seconds(small), 1e-6)
    # 4x the nodes: linear is ~4x, the old probe-from-.001 loop is ~16x.
    assert ratio < 9.0, f"naming {large} identical nodes cost {ratio:.1f}x naming {small}"


def test_unique_name_still_skips_a_name_already_taken_literally() -> None:
    taken = gltfout.TakenNames()
    taken.add("Rock.002")
    assert gltfout.unique_name("Rock", taken) == "Rock"
    assert gltfout.unique_name("Rock", taken) == "Rock.001"
    assert gltfout.unique_name("Rock", taken) == "Rock.003"  # .002 is taken
    assert gltfout.unique_name("Rock", taken) == "Rock.004"


# --- mason-22 ----------------------------------------------------------------


class _BoxSource:
    rev = 0

    def box(self, ref: Any) -> tuple[np.ndarray, np.ndarray]:
        return np.array([-0.5, -0.5, -0.5]), np.array([0.5, 0.5, 0.5])

    def primitives(self, ref: Any) -> list[gltf.Primitive]:
        return []


class _Ctx:
    mason_assets = _BoxSource()


def _doc_with_ground(heights: np.ndarray, **trs: Any) -> tuple[md.MasonDoc, nd.TerrainNode]:
    doc = md.MasonDoc()
    node = doc.add_node(nd.TerrainNode(uid=nd.new_uid(), name="Ground"))
    doc.set_terrain(
        mterrain.Terrain(
            heights=heights, size_x=8.0, size_z=8.0, material=gltf.Material(name="g")
        )
    )
    if trs:
        doc.set_transform(node.uid, **trs)
    return doc, node


def test_drop_to_ground_follows_a_moved_terrain_node() -> None:
    doc, _node = _doc_with_ground(
        np.zeros((5, 5), dtype="f4"), translation=np.array([0.0, 10.0, 0.0])
    )
    prop = doc.add_node(_mesh("P", translation=np.array([0.0, 20.0, 0.0])))
    doc.select([prop.uid])

    mason_menu._drop_to_ground(_Ctx(), type("T", (), {"doc": doc})())

    # The unit cube's bottom rests on the ground node's world height (10), so
    # its centre sits half a metre above it.
    assert msc.resolved_for(doc, prop.uid).world[1, 3] == pytest.approx(10.5)


def test_drop_to_ground_follows_a_scaled_and_offset_terrain_node_through_its_world_matrix() -> None:
    heights = np.zeros((5, 5), dtype="f4")
    heights[2, 2] = 3.0  # centre vertex: local height 3 at local (0, 0)
    doc, node = _doc_with_ground(
        heights, translation=np.array([0.0, 1.0, 0.0]), scale=np.array([1.0, 2.0, 1.0])
    )
    world = msc.resolved_for(doc, node.uid).world
    box = {7: (np.array([-0.5, 9.0, -0.5]), np.array([0.5, 10.0, 0.5]))}

    delta = mops.drop_to_ground(box, terrain=doc.terrain, terrain_world=world)

    # world height = 1 + 2 * 3 = 7
    assert box[7][0][1] + delta[7][1] == pytest.approx(7.0)
    # No terrain_world is the unchanged, origin-placed behaviour.
    flat = mops.drop_to_ground(box, terrain=doc.terrain)
    assert box[7][0][1] + flat[7][1] == pytest.approx(3.0)
