"""``mason/manifest.py``: the sidecar an engine import script reads.

The plan says outright that this schema is ours, that no importer exists to
test it against, and that it is therefore **verified by shape rather than by a
round trip**. So that is what these tests are: every name the manifest uses is
a name the GLB carries, every node the GLB carries is a node the manifest
describes, the counts agree, the units are stated, and every piece of
provenance the GLB has nowhere to put actually arrives. When the first import
script exists, that script becomes the real test and these stay the gate that
keeps the two files naming the same things.

Every test name is a claim, written to fail against the manifest this file is
guarding against: one whose node names are recomputed rather than taken from
the export (so the two drift the day the uniquifier changes), one that leaves
the scene unit to be guessed, one that loses a library job id, and one whose
prefab section describes nodes the GLB does not carry.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from warlock.kernels.geom3d import gltf
from warlock.kernels.geom3d import math3d as m3
from warlock.studio.modes.mason.engine import document as doc
from warlock.studio.modes.mason.engine import gltfout, manifest, refs, scene
from warlock.studio.modes.mason.engine import nodes as nd
from warlock.studio.modes.mason.engine import terrain as tr


def _prim() -> gltf.Primitive:
    return gltf.Primitive(
        positions=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="f4"),
        indices=np.array([0, 1, 2], dtype="u4"),
    )


class _Source:
    def __init__(self, resolve: bool = True) -> None:
        self.resolve = resolve
        self.rev = 0

    def primitives(self, ref):
        return [_prim()] if self.resolve else []

    def box(self, ref):
        return _prim().box() if self.resolve else None


def _mesh(name: str = "", generator: str = "box") -> nd.MeshNode:
    return nd.MeshNode(uid=nd.new_uid(), name=name, ref=refs.primitive_ref(generator, {}))


def _manifest(d: doc.MasonDoc, source: _Source | None = None) -> dict:
    source = source or _Source()
    return manifest.scene_manifest(d, gltfout.scene_model(d, source))


# --- the two files name the same things --------------------------------------


def test_every_name_the_manifest_uses_is_a_name_the_glb_carries():
    """The claim the whole design rests on. The manifest addresses nodes by
    name and nothing else, so a name it invents on its own is a line in an
    import script that silently matches nothing."""
    d = doc.MasonDoc()
    group = nd.GroupNode(uid=nd.new_uid(), name="Props")
    group.children.extend([_mesh("Rock"), _mesh("Rock"), nd.LightNode(uid=nd.new_uid())])
    d.add_node(group)

    source = _Source()
    export = gltfout.scene_model(d, source)
    entry = manifest.scene_manifest(d, export)
    in_glb = {node.name for node in export.model.nodes}
    in_manifest = {node["name"] for node in entry["nodes"]}
    assert in_manifest == in_glb
    for node in entry["nodes"]:
        if node["parent"] is not None:
            assert node["parent"] in in_glb


def test_the_names_come_from_the_export_rather_than_being_worked_out_again():
    """Uniqueness is a rule with state in it -- what is already taken -- so
    running it twice over the same document is two rules that agree only
    while nobody edits either. This is the assertion that would fail if this
    module ever grew its own uniquifier."""
    d = doc.MasonDoc()
    for _ in range(4):
        d.add_node(_mesh("Rock"))

    export = gltfout.scene_model(d, _Source())
    entry = manifest.scene_manifest(d, export)
    assert [n["name"] for n in entry["nodes"]] == [n.name for n in export.model.nodes]
    assert [n["name"] for n in entry["nodes"]] == ["Rock", "Rock.001", "Rock.002", "Rock.003"]


def test_the_parent_of_each_node_is_the_parent_it_has_in_the_file():
    d = doc.MasonDoc()
    outer = nd.GroupNode(uid=nd.new_uid(), name="Outer")
    inner = nd.GroupNode(uid=nd.new_uid(), name="Inner")
    inner.children.append(_mesh("Leaf"))
    outer.children.append(inner)
    d.add_node(outer)

    by_name = {n["name"]: n for n in _manifest(d)["nodes"]}
    assert by_name["Outer"]["parent"] is None
    assert by_name["Inner"]["parent"] == "Outer"
    assert by_name["Leaf"]["parent"] == "Inner"


def test_the_counts_agree_with_what_the_model_actually_holds():
    d = doc.MasonDoc()
    d.add_node(_mesh("A"))
    d.add_node(_mesh("B"))
    d.add_node(nd.LightNode(uid=nd.new_uid(), name="Key"))
    d.add_node(nd.CameraNode(uid=nd.new_uid(), name="Shot"))

    export = gltfout.scene_model(d, _Source())
    stats = manifest.scene_manifest(d, export)["stats"]
    assert stats["nodes"] == len(export.model.nodes) == 4
    assert stats["meshes"] == 1  # both mesh nodes name one reference
    assert stats["lights"] == 1 and stats["cameras"] == 1
    assert stats["triangles"] == 1


# --- units -------------------------------------------------------------------


def test_the_scene_unit_and_the_handedness_are_stated_outright():
    """The one thing an engine import script gets wrong, and the one thing no
    care downstream can recover: a scene imported at a hundredth of its size
    looks like a modelling mistake rather than a unit mismatch."""
    units = _manifest(doc.MasonDoc())["units"]
    assert units["length"] == "metre"
    assert units["up"] == "+Y"
    assert units["forward"] == "-Z"
    assert units["handedness"] == "right"
    assert units["rotation"] == "quaternion xyzw"
    assert units["angles"] == "radians"


def test_the_file_says_what_format_it_is_and_which_geometry_it_describes():
    """A sidecar with no name in it can only be identified by guessing, and one
    that does not name its GLB is one an import script has to be told about."""
    entry = _manifest(doc.MasonDoc())
    assert entry["format"] == "warlock-mason-scene"
    assert entry["format_version"] == manifest.VERSION
    assert entry["geometry"] == "scene.glb"


# --- the provenance the GLB cannot carry -------------------------------------


def test_a_library_asset_keeps_its_job_id_and_the_name_it_had_when_saved():
    """This is the half of the manifest that has no equivalent anywhere in the
    GLB: the file knows there are triangles and only this says whose."""
    d = doc.MasonDoc()
    d.add_node(
        nd.MeshNode(
            uid=nd.new_uid(),
            name="Barrel",
            ref=refs.LibraryRef(job_id="3f2a9c", name="Oak barrel", sha256="abc123"),
        )
    )
    source = manifest.scene_manifest(d, gltfout.scene_model(d, _Source()))["nodes"][0]["source"]
    assert source == {
        "kind": "library",
        "job_id": "3f2a9c",
        "artifact": "model.glb",
        "name": "Oak barrel",
        "sha256": "abc123",
    }


def test_a_primitive_records_the_generator_and_the_numbers_that_built_it():
    """Which is what makes re-running an import script after a parameter
    change a thing a pipeline can do, rather than a re-export by hand."""
    d = doc.MasonDoc()
    d.add_node(
        nd.MeshNode(
            uid=nd.new_uid(),
            ref=refs.primitive_ref("cylinder", {"radius": 0.5, "segments": 12}),
        )
    )
    source = _manifest(d)["nodes"][0]["source"]
    assert source["kind"] == "primitive"
    assert source["generator"] == "cylinder"
    assert source["params"] == {"radius": 0.5, "segments": 12.0}


def test_a_nested_parameter_survives_as_a_list_rather_than_as_a_json_tuple():
    """``refs._normalize`` turns every sequence into a nested tuple, and JSON
    has no tuple -- so a profile has to come back out as lists deliberately
    rather than through whatever ``json.dumps`` does with one silently."""
    d = doc.MasonDoc()
    d.add_node(
        nd.MeshNode(
            uid=nd.new_uid(),
            ref=refs.primitive_ref("lathe", {"profile": [[0.0, 0.0], [1.0, 2.0]]}),
        )
    )
    params = _manifest(d)["nodes"][0]["source"]["params"]
    assert params["profile"] == [[0.0, 0.0], [1.0, 2.0]]
    assert json.loads(json.dumps(params)) == params


def test_the_user_properties_a_designer_typed_reach_the_import_script():
    """The field the manifest exists for as much as any other: an engine does
    something with ``spawn=true`` and a GLB has nowhere to put it that every
    importer reads."""
    d = doc.MasonDoc()
    node = _mesh("Marker")
    node.properties = {"spawn": True, "faction": "red", "weight": 2.5}
    d.add_node(node)
    assert _manifest(d)["nodes"][0]["properties"] == {
        "spawn": True,
        "faction": "red",
        "weight": 2.5,
    }


def test_static_is_recorded_because_it_is_what_it_was_added_for():
    """``static`` ORs down the tree in the resolver precisely so it can reach
    this file -- an engine bakes lighting for a static object and does not for
    a moving one, and nothing in glTF carries the distinction."""
    d = doc.MasonDoc()
    group = nd.GroupNode(uid=nd.new_uid(), name="World", static=True)
    group.children.append(_mesh("Wall"))
    d.add_node(group)
    d.add_node(_mesh("Cart"))

    by_name = {n["name"]: n for n in _manifest(d)["nodes"]}
    assert by_name["Wall"]["static"] is True, "inherited from the group above it"
    assert "static" not in by_name["Cart"]


def test_a_light_carries_its_own_fields_for_a_reader_with_no_gltf_library():
    """Repeated from the GLB deliberately: the extension block is the part of
    a binary glTF a JSON-only reader is least likely to be able to reach."""
    d = doc.MasonDoc()
    d.add_node(
        nd.LightNode(
            uid=nd.new_uid(),
            name="Spot",
            kind="spot",
            color=(1.0, 0.5, 0.25),
            intensity=12.0,
            range=8.0,
            inner_cone_angle=0.1,
            outer_cone_angle=0.5,
        )
    )
    light = _manifest(d)["nodes"][0]["light"]
    assert light["type"] == "spot"
    assert light["color"] == [1.0, 0.5, 0.25]
    assert light["intensity"] == 12.0
    assert light["range"] == 8.0
    assert light["innerConeAngle"] == pytest.approx(0.1)
    assert light["outerConeAngle"] == pytest.approx(0.5)


def test_a_field_a_lights_kind_ignores_is_left_out_here_as_well_as_in_the_glb():
    """Two files describing one light must not disagree about it, and the
    extension says a directional light has no range and no cone."""
    d = doc.MasonDoc()
    d.add_node(nd.LightNode(uid=nd.new_uid(), kind="directional", range=40.0))
    light = _manifest(d)["nodes"][0]["light"]
    assert "range" not in light and "innerConeAngle" not in light


def test_the_terrain_records_its_extent_and_its_resolution():
    """The mesh is in the GLB; how big the ground is and how finely it was
    sampled is a property of the document that a triangle soup cannot state."""
    d = doc.MasonDoc()
    d.set_terrain(
        tr.Terrain(
            heights=np.zeros((9, 9), dtype="f4"),
            size_x=64.0,
            size_z=32.0,
            material=gltf.Material(),
        )
    )
    d.add_node(nd.TerrainNode(uid=nd.new_uid(), name="Ground"))
    assert _manifest(d)["nodes"][0]["terrain"] == {
        "size_x": 64.0,
        "size_z": 32.0,
        "side": 8,
    }


# --- transforms --------------------------------------------------------------


def test_both_the_authored_transform_and_the_composed_one_are_written():
    """An import script that rebuilds the hierarchy wants the local one and
    one that places a single object wants the world one, and neither should
    have to decompose a 4x4 whose handedness it would have to assume."""
    d = doc.MasonDoc()
    group = nd.GroupNode(uid=nd.new_uid(), name="Row", translation=m3.vec3(10.0, 0.0, 0.0))
    leaf = _mesh("Post")
    leaf.translation = m3.vec3(0.0, 2.0, 0.0)
    group.children.append(leaf)
    d.add_node(group)

    by_name = {n["name"]: n for n in _manifest(d)["nodes"]}
    assert by_name["Post"]["local"]["translation"] == [0.0, 2.0, 0.0]
    assert by_name["Post"]["world"]["translation"] == pytest.approx([10.0, 2.0, 0.0])


def test_a_rotation_is_written_as_four_numbers_in_the_order_the_units_block_claims():
    """The units block says ``quaternion xyzw``; a WXYZ value under that label
    is a rotation that looks plausible and is wrong on every axis."""
    d = doc.MasonDoc()
    node = _mesh("Turned")
    node.rotation = m3.quat_from_axis_angle(m3.vec3(0.0, 1.0, 0.0), np.pi / 2)
    d.add_node(node)

    written = _manifest(d)["nodes"][0]["local"]["rotation"]
    assert len(written) == 4
    assert written == pytest.approx([0.0, np.sin(np.pi / 4), 0.0, np.cos(np.pi / 4)])


# --- prefabs -----------------------------------------------------------------


def _instanced() -> doc.MasonDoc:
    d = doc.MasonDoc()
    template = nd.GroupNode(uid=nd.new_uid(), name="Crate")
    template.children.append(_mesh("Body"))
    d.define_prefab("crate", template)
    for i in range(3):
        d.add_node(
            nd.PrefabNode(
                uid=nd.new_uid(),
                name=f"crate{i}",
                template="crate",
                translation=m3.vec3(i * 4.0, 0.0, 0.0),
            )
        )
    return d


def test_each_template_lists_its_instances_and_what_each_one_became():
    """This is what lets an import script build engine prefabs instead of
    hundreds of loose meshes -- which is the difference between a scene a
    person goes on editing in their engine and one they can only look at."""
    entry = _manifest(_instanced())
    crate = entry["prefabs"]["crate"]
    assert [i["node"] for i in crate["instances"]] == ["crate0", "crate1", "crate2"]
    for instance in crate["instances"]:
        assert len(instance["nodes"]) == 2  # the template's group and its mesh


def test_an_instances_node_list_names_only_nodes_the_glb_carries():
    """A prefab section describing nodes that are not in the file is worse
    than no prefab section: the import script builds a prefab out of nothing
    and reports success."""
    d = _instanced()
    export = gltfout.scene_model(d, _Source())
    entry = manifest.scene_manifest(d, export)
    in_glb = {node.name for node in export.model.nodes}
    for block in entry["prefabs"].values():
        for instance in block["instances"]:
            assert set(instance["nodes"]) <= in_glb
            assert instance["node"] in in_glb


def test_an_instances_copies_belong_to_that_instance_and_not_to_another():
    """The template's node objects are shared by identity across instances, so
    membership can only be decided by the path that reached each copy."""
    entry = _manifest(_instanced())
    lists = [i["nodes"] for i in entry["prefabs"]["crate"]["instances"]]
    flat = [name for names in lists for name in names]
    assert len(flat) == len(set(flat)), "no node belongs to two instances"


def test_a_template_nobody_placed_is_in_no_section_of_the_file():
    """It is in the exported GLB in no form whatever, and a manifest naming it
    would describe geometry that does not exist."""
    d = doc.MasonDoc()
    d.define_prefab("unused", nd.GroupNode(uid=nd.new_uid(), name="Ghost"))
    d.add_node(_mesh("Real"))
    assert "prefabs" not in _manifest(d)


def test_a_node_inside_an_instance_points_at_the_row_the_outliner_has():
    """``Placed.owner``'s rule carried through: an import script groups an
    instance's contents the way the editor does, rather than by guessing from
    the hierarchy."""
    d = _instanced()
    export = gltfout.scene_model(d, _Source())
    entry = manifest.scene_manifest(d, export)
    owners = {n["name"]: n.get("owner_uid") for n in entry["nodes"]}
    instance_uids = {node.uid for node in d.roots}
    assert owners["crate0"] is None, "an instance is its own owner"
    inside = [name for name, owner in owners.items() if owner is not None]
    assert inside, "the template's nodes are reached through an instance"
    assert {owners[name] for name in inside} <= instance_uids


# --- what could not be exported ----------------------------------------------


def test_a_broken_link_is_named_in_the_manifest_rather_than_left_to_be_noticed():
    """The GLB shows it as a node with no mesh, which is indistinguishable
    from a node nobody has given a shape yet. This is where it is said in
    words -- ``Model.skipped_textures``' rule, one file over."""
    d = doc.MasonDoc()
    d.add_node(
        nd.MeshNode(
            uid=nd.new_uid(), name="Barrel", ref=refs.LibraryRef(job_id="gone", name="Barrel")
        )
    )
    entry = _manifest(d, _Source(resolve=False))
    assert entry["missing"] == [
        {"node": "Barrel", "kind": "library", "job_id": "gone", "artifact": "model.glb",
         "name": "Barrel"}
    ]
    assert entry["nodes"][0]["source"]["job_id"] == "gone"


def test_a_scene_with_nothing_missing_carries_no_missing_section():
    d = doc.MasonDoc()
    d.add_node(_mesh("Fine"))
    assert "missing" not in _manifest(d)


# --- the bytes ---------------------------------------------------------------


def test_the_manifest_is_sorted_and_indented_because_it_is_meant_to_be_read():
    """By the person writing the import script, and by a diff showing what
    changed between two exports of one scene. The geometry is in the file next
    to it, so there is no size argument for minifying this one."""
    d = _instanced()
    raw = manifest.manifest_bytes(d, gltfout.scene_model(d, _Source()))
    parsed = json.loads(raw)
    assert raw == json.dumps(parsed, sort_keys=True, indent=2).encode("utf-8")
    assert list(parsed) == sorted(parsed)


def test_two_manifests_of_an_unchanged_document_are_byte_identical():
    """Every other format Mason writes holds still for an unchanged document
    and this one has no excuse not to -- nothing in it is ordered by ``id()``
    or by set iteration."""
    d = _instanced()
    first = manifest.manifest_bytes(d, gltfout.scene_model(d, _Source()))
    second = manifest.manifest_bytes(d, gltfout.scene_model(d, _Source()))
    assert first == second


def test_every_node_the_resolver_places_is_described_here_too():
    """The manifest is written from the export rather than from the document,
    so a node the viewport draws and the manifest omits would mean the two had
    stopped being the same walk."""
    d = _instanced()
    d.add_node(nd.LightNode(uid=nd.new_uid(), name="Key"))
    entry = _manifest(d)
    described = {n["name"] for n in entry["nodes"]}
    export = gltfout.scene_model(d, _Source())
    for placed in scene.resolve(d):
        match = [e for e in export.nodes if e.path == placed.path]
        assert len(match) == 1
        assert match[0].name in described
