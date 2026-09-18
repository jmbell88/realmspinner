"""``mason/gltfout.py``: the scene tree as a glTF scene tree.

Every test name is a claim, written to fail against the export this file is
guarding against: one that flattens the hierarchy the way ``resolve`` does,
one that writes a node's world matrix into its local TRS, one that uploads
the same barrel five hundred times because it keyed its mesh table on the
node, one that emits two nodes called "Rock" and leaves the manifest unable to
address either, one that silently drops a prop whose library job is gone, and
one that reaches the file by a second recursive descent of its own -- which
would be the first time in this engine that the viewport and the export could
disagree about what a scene contains.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.kernels.geom3d import glbwrite, gltf
from warlock.kernels.geom3d import math3d as m3
from warlock.studio.modes.mason.engine import document as doc
from warlock.studio.modes.mason.engine import gltfout, refs, scene
from warlock.studio.modes.mason.engine import nodes as nd
from warlock.studio.modes.mason.engine import terrain as tr


def _prim(x: float = 0.0, name: str = "") -> gltf.Primitive:
    """One triangle, offset along X so two refs' geometry is distinguishable."""
    return gltf.Primitive(
        positions=np.array([[x, 0, 0], [x + 1, 0, 0], [x, 1, 0]], dtype="f4"),
        indices=np.array([0, 1, 2], dtype="u4"),
        material=gltf.Material(name=name),
    )


class _Source:
    """A ``GeometrySource`` over a fixed table, keyed the way the real one is.

    A ref that is not in the table resolves to nothing, which is exactly the
    shape of a library job that has been deleted -- so the missing-asset tests
    below need no filesystem and no service to produce one.
    """

    def __init__(self, table: dict | None = None) -> None:
        self.table = table or {}
        self.rev = 0
        self.calls: list[tuple] = []

    def primitives(self, ref):
        self.calls.append(refs.ref_key(ref))
        return self.table.get(refs.ref_key(ref), [])

    def box(self, ref):
        prims = self.primitives(ref)
        return prims[0].box() if prims else None


def _source(*generators: str) -> _Source:
    table = {}
    for i, name in enumerate(generators):
        table[refs.ref_key(refs.primitive_ref(name, {}))] = [_prim(float(i), name)]
    return _Source(table)


def _mesh(name: str = "box", **params) -> nd.MeshNode:
    return nd.MeshNode(uid=nd.new_uid(), name="", ref=refs.primitive_ref(name, params))


# --- the tree survives -------------------------------------------------------


def test_a_group_and_its_children_export_as_a_parent_and_its_children():
    """The whole reason this is ``walk`` accumulating structure rather than
    ``resolve``: a flattening export would hand the engine three roots."""
    d = doc.MasonDoc()
    group = nd.GroupNode(uid=nd.new_uid(), name="Props")
    group.children.extend([_mesh(), _mesh()])
    d.add_node(group)

    out = gltfout.scene_model(d, _source("box"))
    assert out.model.roots == [0]
    assert out.model.nodes[0].name == "Props"
    assert out.model.nodes[0].children == [1, 2]


def test_a_node_exports_its_own_transform_and_not_its_composed_world():
    """A world matrix baked into a local TRS is a scene whose every node is a
    root wearing its ancestors' transforms -- which is what a flattening
    exporter produces and what the parent chain here exists to avoid."""
    d = doc.MasonDoc()
    group = nd.GroupNode(uid=nd.new_uid(), translation=m3.vec3(10.0, 0.0, 0.0))
    leaf = _mesh()
    leaf.translation = m3.vec3(1.0, 0.0, 0.0)
    group.children.append(leaf)
    d.add_node(group)

    out = gltfout.scene_model(d, _source("box"))
    assert out.model.nodes[1].translation == pytest.approx([1.0, 0.0, 0.0])
    # And the composition still lands where the resolver puts it.
    assert out.model.nodes[1].world[:3, 3] == pytest.approx([11.0, 0.0, 0.0])


def test_every_node_the_resolver_places_is_a_node_the_export_names_at_the_same_place():
    """The claim the shared walk exists for, asserted rather than argued.

    ``resolve`` and ``scene_model`` are the same traversal with two
    accumulators, so for every placed item there is exactly one exported node
    with the same path and the same world matrix. A second traversal here --
    even a correct one on the day it was written -- is a second answer to the
    question, and this is the test that would notice the day they differed.
    """
    d = doc.MasonDoc()
    group = nd.GroupNode(
        uid=nd.new_uid(),
        translation=m3.vec3(2.0, 0.0, -1.0),
        rotation=m3.quat_from_axis_angle(m3.vec3(0.0, 1.0, 0.0), 0.7),
        scale=m3.vec3(2.0, 2.0, 2.0),
    )
    inner = nd.GroupNode(uid=nd.new_uid(), translation=m3.vec3(0.0, 3.0, 0.0))
    leaf = _mesh()
    leaf.rotation = m3.quat_from_axis_angle(m3.vec3(1.0, 0.0, 0.0), -0.4)
    inner.children.append(leaf)
    group.children.append(inner)
    d.add_node(group)

    placed = {p.path: p for p in scene.resolve(d)}
    exported = {e.path: e for e in gltfout.scene_model(d, _source("box")).nodes}
    assert set(placed) <= set(exported)
    for path, item in placed.items():
        assert exported[path].world == pytest.approx(item.world)


def test_a_hidden_node_is_not_in_the_file_at_all():
    """One flag decides drawing, picking and exporting, the way Clay's
    ``to_model`` already decides all three from ``visible``."""
    d = doc.MasonDoc()
    d.add_node(_mesh())
    hidden = _mesh()
    hidden.visible = False
    d.add_node(hidden)

    assert len(gltfout.scene_model(d, _source("box")).model.nodes) == 1
    assert len(gltfout.scene_model(d, _source("box"), include_hidden=True).model.nodes) == 2


# --- instancing --------------------------------------------------------------


def test_six_nodes_naming_one_reference_share_a_single_glb_mesh():
    """Instancing is the file format's own and falls out of keying the mesh
    table on the ref. Keyed on the node instead -- which is what Clay's
    viewport does, correctly, because in Clay every object owns its mesh --
    this is six barrels' worth of accessors for one barrel."""
    d = doc.MasonDoc()
    for _ in range(6):
        d.add_node(_mesh())

    source = _source("box")
    out = gltfout.scene_model(d, source)
    assert len(out.model.meshes) == 1
    assert [n.mesh for n in out.model.nodes] == [0] * 6
    assert source.calls.count(refs.ref_key(refs.primitive_ref("box", {}))) == 1


def test_two_references_that_differ_only_in_their_parameters_are_two_meshes():
    """``ref_key`` carries the normalized params, so a 1m box and a 2m box are
    two shapes -- and a key that dropped them would place the wrong one."""
    d = doc.MasonDoc()
    d.add_node(_mesh("box", size=1.0))
    d.add_node(_mesh("box", size=2.0))
    source = _Source(
        {
            refs.ref_key(refs.primitive_ref("box", {"size": s})): [_prim(s)]
            for s in (1.0, 2.0)
        }
    )
    assert len(gltfout.scene_model(d, source).model.meshes) == 2


def test_a_material_override_makes_a_second_mesh_over_the_same_geometry():
    """glTF hangs a material off the primitive rather than off the node, so a
    retinted copy of a shared asset is a second mesh whether we like it or
    not -- the cost the plan's open question 3 names for the GPU side, stated
    here rather than discovered later. The geometry is still shared by value:
    what doubles is the accessor set, not the source's resolve."""
    d = doc.MasonDoc()
    plain = _mesh()
    tinted = _mesh()
    tinted.material = gltf.Material(name="mossy")
    d.add_node(plain)
    d.add_node(tinted)

    out = gltfout.scene_model(d, _source("box"))
    assert len(out.model.meshes) == 2
    assert out.model.meshes[0][0].material.name == "box"  # as the source authored it
    assert out.model.meshes[1][0].material.name == "mossy"
    assert np.array_equal(
        out.model.meshes[0][0].positions, out.model.meshes[1][0].positions
    )


def test_an_override_never_writes_back_into_the_geometry_the_source_handed_over():
    """The source caches its primitives and hands the same objects to the
    viewport, so retinting one in place would retint the asset everywhere it
    is drawn -- including the five hundred placements that have no override."""
    d = doc.MasonDoc()
    tinted = _mesh()
    tinted.material = gltf.Material(name="mossy")
    d.add_node(tinted)

    source = _source("box")
    original = source.table[refs.ref_key(refs.primitive_ref("box", {}))][0]
    gltfout.scene_model(d, source)
    assert original.material.name == "box"


# --- prefabs -----------------------------------------------------------------


def _with_prefab() -> doc.MasonDoc:
    """A template of a group holding two meshes, placed three times."""
    d = doc.MasonDoc()
    template = nd.GroupNode(uid=nd.new_uid(), name="Crate")
    template.children.extend([_mesh(), _mesh()])
    d.define_prefab("crate", template)
    for i in range(3):
        d.add_node(
            nd.PrefabNode(
                uid=nd.new_uid(),
                name=f"crate{i}",
                template="crate",
                translation=m3.vec3(float(i) * 5.0, 0.0, 0.0),
            )
        )
    return d


def test_a_prefab_instance_is_a_node_of_its_own_with_the_template_under_it():
    """``scene.walk``'s ``enter`` hook exists for exactly this node: an
    expanded instance ``continue``s past ``visit``, so without the hook
    ``path[:-1]`` names a node the exporter has never seen and the instance's
    own transform is only recoverable through a matrix inverse."""
    out = gltfout.scene_model(_with_prefab(), _source("box"))
    instances = [n for n in out.nodes if n.kind == "prefab"]
    assert [n.name for n in instances] == ["crate0", "crate1", "crate2"]
    for record in instances:
        node = out.model.nodes[record.index]
        assert node.mesh is None
        assert len(node.children) == 1  # the template's root group
        assert len(out.model.nodes[node.children[0]].children) == 2


def test_three_instances_of_one_template_still_share_one_mesh():
    """The instance is authored; the *geometry* sharing is not -- it falls out
    of the ref key the same way it does for three loose placements."""
    out = gltfout.scene_model(_with_prefab(), _source("box"))
    assert len(out.model.meshes) == 1
    assert sum(n.mesh == 0 for n in out.model.nodes if n.mesh is not None) == 6


def test_each_instances_copies_land_where_that_instance_is():
    """The template's node objects are shared by identity across instances, so
    an exporter that wrote through them would put all three copies wherever
    the last one happened to land."""
    out = gltfout.scene_model(_with_prefab(), _source("box"))
    out.model.update_world()
    placed = sorted(
        float(out.model.nodes[n.index].world[0, 3])
        for n in out.nodes
        if n.kind == "mesh"
    )
    assert placed == pytest.approx([0.0, 0.0, 5.0, 5.0, 10.0, 10.0])


def test_a_node_inside_an_instance_records_the_instance_as_its_owner():
    """``Placed.owner`` carried through to the export, because the manifest
    addresses an instance's contents to the row the outliner actually has."""
    d = _with_prefab()
    out = gltfout.scene_model(d, _source("box"))
    inside = [n for n in out.nodes if n.prefab == "crate"]
    assert inside, "the template's own nodes are reached through it"
    owners = {n.owner for n in inside}
    assert owners == {n.uid for n in d.roots}


def test_an_instance_whose_template_is_gone_is_in_neither_the_file_nor_the_viewport():
    """The walk yields nothing for that branch, quietly, and the export gets
    the same nothing -- an exporter that emitted an empty node for it would
    put something in the file the viewport does not draw, which is the one
    disagreement the shared traversal exists to prevent."""
    d = doc.MasonDoc()
    d.add_node(nd.PrefabNode(uid=nd.new_uid(), name="ghost", template="missing"))
    assert gltfout.scene_model(d, _source("box")).model.nodes == []


# --- names -------------------------------------------------------------------


def test_six_nodes_called_rock_get_six_names_the_manifest_can_address():
    """Every importer's find-by-name assumes uniqueness and the manifest keys
    on it, while a scene is a place and places have six rocks in them."""
    d = doc.MasonDoc()
    for _ in range(6):
        node = _mesh()
        node.name = "Rock"
        d.add_node(node)

    names = [n.name for n in gltfout.scene_model(d, _source("box")).model.nodes]
    assert names == ["Rock", "Rock.001", "Rock.002", "Rock.003", "Rock.004", "Rock.005"]


def test_a_name_the_uniquifier_would_have_invented_is_not_handed_out_twice():
    """A document that already holds ``Rock.001`` is the case a counting
    uniquifier gets wrong: it generates the name that is already taken."""
    d = doc.MasonDoc()
    for name in ("Rock", "Rock.001", "Rock"):
        node = _mesh()
        node.name = name
        d.add_node(node)

    names = [n.name for n in gltfout.scene_model(d, _source("box")).model.nodes]
    assert names == ["Rock", "Rock.001", "Rock.002"]
    assert len(set(names)) == 3


def test_a_node_with_no_name_gets_one_named_after_its_kind():
    """A glTF node may carry no name at all, and one that does is a node the
    manifest cannot talk about."""
    d = doc.MasonDoc()
    d.add_node(nd.GroupNode(uid=nd.new_uid()))
    d.add_node(_mesh())
    d.add_node(nd.LightNode(uid=nd.new_uid()))
    d.add_node(nd.CameraNode(uid=nd.new_uid()))

    names = [n.name for n in gltfout.scene_model(d, _source("box")).model.nodes]
    assert names == ["Group", "Mesh", "Light", "Camera"]


# --- what could not be resolved ----------------------------------------------


def test_an_asset_whose_job_is_gone_keeps_its_node_and_is_reported():
    """A scene that loses a prop must not also lose *where* the prop was, or a
    relink has nothing to put back -- and a loss that is stated is not the
    same as a loss that is invisible (``Model.skipped_textures``' rule)."""
    d = doc.MasonDoc()
    gone = nd.MeshNode(
        uid=nd.new_uid(), name="Barrel", ref=refs.LibraryRef(job_id="deadbeef", name="Barrel")
    )
    d.add_node(gone)

    out = gltfout.scene_model(d, _source("box"))
    assert out.model.nodes[0].name == "Barrel"
    assert out.model.nodes[0].mesh is None
    assert [ref.job_id for _index, ref in out.unresolved] == ["deadbeef"]


# --- lights, cameras and the ground -----------------------------------------


def test_lights_and_cameras_survive_a_write_and_a_read_back():
    """Through the shared writer and the loader beside it, which is the
    strongest test available for either: Mason's ``LightNode`` fields *are*
    KHR_lights_punctual's, so this is a copy rather than a conversion and
    there is nowhere for the two to drift."""
    d = doc.MasonDoc()
    d.add_node(
        nd.LightNode(uid=nd.new_uid(), name="Sun", kind="directional", intensity=3.0)
    )
    d.add_node(
        nd.LightNode(
            uid=nd.new_uid(), name="Lamp", kind="spot", range=9.0, outer_cone_angle=0.6
        )
    )
    d.add_node(nd.CameraNode(uid=nd.new_uid(), name="Establishing", yfov=0.9, zfar=400.0))

    loaded = gltf.load(gltfout.scene_glb(d, _source("box")))
    assert [light.kind for light in loaded.lights] == ["directional", "spot"]
    assert [light.name for light in loaded.lights] == ["Sun", "Lamp"]
    assert loaded.lights[1].range == pytest.approx(9.0)
    assert loaded.cameras[0].name == "Establishing"
    assert loaded.cameras[0].zfar == pytest.approx(400.0)
    assert [n.light for n in loaded.nodes] == [0, 1, None]
    assert [n.camera for n in loaded.nodes] == [None, None, 0]


def test_the_terrain_exports_as_an_ordinary_mesh_node():
    """The node exists so the ground has an outliner row and a transform, and
    the file is where that stops being a special case: an engine importing
    this scene gets a mesh, not a format of its own to understand."""
    d = doc.MasonDoc()
    d.set_terrain(
        tr.Terrain(
            heights=np.zeros((5, 5), dtype="f4"),
            size_x=10.0,
            size_z=10.0,
            material=gltf.Material(name="grass"),
        )
    )
    d.add_node(nd.TerrainNode(uid=nd.new_uid(), name="Ground"))

    out = gltfout.scene_model(d, _source("box"))
    assert out.model.nodes[0].mesh == 0
    assert len(out.model.meshes[0][0].positions) == 25
    assert out.model.meshes[0][0].material.name == "grass"


def test_two_terrain_nodes_under_one_override_build_the_ground_once():
    """The terrain is a document singleton, so the same override is the same
    mesh -- the rule every reference in this file already follows."""
    d = doc.MasonDoc()
    d.set_terrain(
        tr.Terrain(
            heights=np.zeros((3, 3), dtype="f4"),
            size_x=4.0,
            size_z=4.0,
            material=gltf.Material(),
        )
    )
    d.add_node(nd.TerrainNode(uid=nd.new_uid()))
    d.add_node(nd.TerrainNode(uid=nd.new_uid()))

    assert len(gltfout.scene_model(d, _source("box")).model.meshes) == 1


# --- one traversal -----------------------------------------------------------


def test_the_export_reaches_every_node_through_scene_walk_and_no_other_way(monkeypatch):
    """The structural claim, asserted behaviourally rather than by reading the
    source: with the shared walk removed, this exporter must have nothing left
    to export. A recursive descent of its own would pass this test's sibling
    assertions and fail this one -- which is the point, because a second
    traversal is a second answer to "what is in this scene" and the day the
    two differed the symptom would be a file that does not match the
    viewport."""
    calls: list[int] = []

    def refuse(*args, **kwargs):
        calls.append(1)
        raise AssertionError("scene.walk is the only traversal")

    monkeypatch.setattr(gltfout.sc, "walk", refuse)
    with pytest.raises(AssertionError, match="only traversal"):
        gltfout.scene_model(_with_prefab(), _source("box"))
    assert calls == [1]


def test_the_whole_scene_survives_a_round_trip_through_the_shared_writer():
    """End to end: a hierarchy, an instanced mesh, a light, a camera and a
    ground, written and read back by the loader that ships."""
    d = _with_prefab()
    group = nd.GroupNode(uid=nd.new_uid(), name="Lighting")
    group.children.append(nd.LightNode(uid=nd.new_uid(), name="Key", kind="point"))
    d.add_node(group)
    d.add_node(nd.CameraNode(uid=nd.new_uid(), name="Shot"))

    data = gltfout.scene_glb(d, _source("box"))
    loaded = gltf.load(data)
    assert len(loaded.roots) == 5
    assert len(loaded.meshes) == 1
    assert loaded.by_name["Key"] and loaded.by_name["Shot"]
    assert data == glbwrite.write_glb(gltfout.scene_model(d, _source("box")).model)
