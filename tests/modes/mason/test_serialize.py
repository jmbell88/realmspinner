"""The ``.rscn`` document format: a zip of ``scene.json``, an optional
``terrain/heights.npy`` and any ``textures/<n>.png`` a material carries.

Mason stores no geometry at all -- a scene is links (a generator's parameters,
a job id) rather than triangles -- so the tests that matter here are not "does
a mesh survive" the way Clay's are, but three narrower questions this format
answers instead: does the *arrangement* survive exactly (every node kind's own
fields, the tree's shape, sibling order, which nodes share one material
object); is the file reproducible (two saves identical, a save-load-save
identical to the first save, every timestamp fixed); and does a half-broken
file get refused rather than opened wrong, except for the one kind of break
this format is designed to tolerate -- a dangling library reference, which is
a known, repairable link rather than a build step that silently failed.
"""

from __future__ import annotations

import json
import zipfile
from io import BytesIO

import numpy as np
import pytest

from realmspinner.kernels.geom3d import gltf
from realmspinner.studio.modes.mason.engine import nodes as nd
from realmspinner.studio.modes.mason.engine import serialize as ser
from realmspinner.studio.modes.mason.engine.document import MasonDoc
from realmspinner.studio.modes.mason.engine.refs import LibraryRef, ref_key
from realmspinner.studio.modes.mason.engine.terrain import Terrain


def _tex(seed: int = 0) -> tuple[int, int, bytes]:
    return (2, 2, bytes((seed + i) % 256 for i in range(2 * 2 * 4)))


def _doc() -> MasonDoc:
    """One of everything: all six node kinds, two levels of nesting, two
    prefab templates, a terrain, a texture, both ref kinds, a material shared
    by two nodes, and a saved camera -- one document every test below reuses
    rather than six almost-identical smaller ones.
    """
    palette_a = gltf.Material(name="red", base_color_factor=(0.9, 0.1, 0.2, 1.0))
    palette_b = gltf.Material(name="glow", base_color=_tex())
    shared = gltf.Material(name="shared-stone")
    ground = gltf.Material(name="ground", base_color=_tex(7))

    mesh_a = nd.MeshNode(
        uid=nd.new_uid(),
        name="RockA",
        ref=ser.primitive_ref(
            "box", {"size": (1.0, 2.0, 3.0), "profile": [[0.0, 0.0], [1.0, 2.0]]}
        ),
        material=shared,
    )
    light = nd.LightNode(
        uid=nd.new_uid(),
        name="Sun",
        kind="directional",
        color=(1.0, 0.9, 0.8),
        intensity=3.0,
        range=10.0,
        inner_cone_angle=0.1,
        outer_cone_angle=0.5,
    )
    inner_group = nd.GroupNode(uid=nd.new_uid(), name="Inner", children=[mesh_a, light])
    outer_group = nd.GroupNode(
        uid=nd.new_uid(), name="Group", children=[inner_group], properties={"note": "grouped"}
    )
    mesh_b = nd.MeshNode(
        uid=nd.new_uid(),
        name="RockB",
        ref=LibraryRef(job_id="job-123", artifact="model.glb", name="Barrel", sha256="deadbeef"),
        material=shared,
    )
    camera = nd.CameraNode(uid=nd.new_uid(), name="Main", yfov=1.2, znear=0.05, zfar=500.0)
    prefab_inst = nd.PrefabNode(uid=nd.new_uid(), name="BarrelInstance", template="barrel")
    terrain_node = nd.TerrainNode(uid=nd.new_uid(), name="Ground", material=palette_b)

    doc = MasonDoc(
        roots=[outer_group, mesh_b, camera, prefab_inst, terrain_node],
        materials=[palette_a, palette_b],
    )
    doc.terrain = Terrain(
        heights=(np.arange(25, dtype=np.float32).reshape(5, 5) * 0.1),
        size_x=10.0,
        size_z=10.0,
        material=ground,
    )
    doc.prefabs["barrel"] = nd.MeshNode(
        uid=nd.new_uid(),
        name="BarrelTemplate",
        ref=LibraryRef(job_id="job-456"),
        material=gltf.Material(name="prefab-only"),
    )
    doc.prefabs["torch"] = nd.GroupNode(
        uid=nd.new_uid(),
        name="TorchTemplate",
        children=[nd.LightNode(uid=nd.new_uid(), name="Flame", kind="point", intensity=2.0)],
    )
    doc.view = {"yaw": 0.4, "pitch": 1.1, "distance": 7.5, "target": [1.0, 2.0, 3.0]}
    return doc


def _roundtrip(doc: MasonDoc) -> MasonDoc:
    return ser.read_rscn(ser.rscn_bytes(doc))


def _rewrite(data: bytes, edit) -> bytes:
    """The same archive with ``edit`` applied to its parsed ``scene.json``."""
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


def _drop_members(data: bytes, predicate) -> bytes:
    """The same archive with every member matching ``predicate`` left out."""
    out = BytesIO()
    with zipfile.ZipFile(BytesIO(data)) as src, zipfile.ZipFile(out, "w") as dst:
        for name in src.namelist():
            if not predicate(name):
                dst.writestr(name, src.read(name))
    return out.getvalue()


def _find_node(scene: dict, name: str) -> dict:
    """The mutable node entry named ``name``, wherever it sits -- the scene
    tree or a prefab template -- so a mangle function does not have to know
    which."""

    def search(entries: list) -> dict | None:
        for entry in entries:
            if entry.get("name") == name:
                return entry
            found = search(entry.get("children", []))
            if found is not None:
                return found
        return None

    found = search(scene.get("nodes", []))
    if found is None:
        for template in scene.get("prefabs", {}).values():
            found = search([template])
            if found is not None:
                break
    if found is None:
        raise KeyError(name)
    return found


# --- each node kind, round-tripped -------------------------------------------


def _assert_group(before: nd.Node, after: nd.Node) -> None:
    assert isinstance(after, nd.GroupNode)
    assert after.properties == before.properties


def _assert_mesh(before: nd.MeshNode, after: nd.Node) -> None:
    assert isinstance(after, nd.MeshNode)
    assert after.ref == before.ref
    assert after.material is not None
    assert after.material.name == before.material.name


def _assert_light(before: nd.LightNode, after: nd.Node) -> None:
    assert isinstance(after, nd.LightNode)
    assert after.kind == before.kind
    assert after.color == pytest.approx(before.color)
    assert after.intensity == pytest.approx(before.intensity)
    assert after.range == pytest.approx(before.range)
    assert after.inner_cone_angle == pytest.approx(before.inner_cone_angle)
    assert after.outer_cone_angle == pytest.approx(before.outer_cone_angle)


def _assert_camera(before: nd.CameraNode, after: nd.Node) -> None:
    assert isinstance(after, nd.CameraNode)
    assert after.yfov == pytest.approx(before.yfov)
    assert after.znear == pytest.approx(before.znear)
    assert after.zfar == pytest.approx(before.zfar)


def _assert_prefab(before: nd.PrefabNode, after: nd.Node) -> None:
    assert isinstance(after, nd.PrefabNode)
    assert after.template == before.template


def _assert_terrain_node(before: nd.TerrainNode, after: nd.Node) -> None:
    assert isinstance(after, nd.TerrainNode)
    assert after.material is not None
    assert after.material.name == before.material.name


@pytest.mark.parametrize(
    ("kind", "make", "check"),
    [
        (
            "group",
            lambda: nd.GroupNode(uid=nd.new_uid(), name="G", properties={"a": 1}),
            _assert_group,
        ),
        (
            "mesh",
            lambda: nd.MeshNode(
                uid=nd.new_uid(),
                name="M",
                ref=ser.primitive_ref("box", {"size": (1.0, 2.0, 3.0)}),
                material=gltf.Material(name="mesh-mat"),
            ),
            _assert_mesh,
        ),
        (
            "light",
            lambda: nd.LightNode(
                uid=nd.new_uid(),
                name="L",
                kind="spot",
                color=(0.1, 0.2, 0.3),
                intensity=4.5,
                range=12.0,
                inner_cone_angle=0.2,
                outer_cone_angle=0.9,
            ),
            _assert_light,
        ),
        (
            "camera",
            lambda: nd.CameraNode(uid=nd.new_uid(), name="C", yfov=1.0, znear=0.2, zfar=250.0),
            _assert_camera,
        ),
        (
            "prefab",
            lambda: nd.PrefabNode(uid=nd.new_uid(), name="P", template="barrel"),
            _assert_prefab,
        ),
        (
            "terrain",
            lambda: nd.TerrainNode(
                uid=nd.new_uid(), name="T", material=gltf.Material(name="t-mat")
            ),
            _assert_terrain_node,
        ),
    ],
    ids=["group", "mesh", "light", "camera", "prefab", "terrain"],
)
def test_each_node_kind_round_trips_with_its_own_fields_intact(kind, make, check) -> None:
    node = make()
    doc = MasonDoc(roots=[node])
    out = _roundtrip(doc)

    assert len(out.roots) == 1
    after = out.roots[0]
    assert after.uid == node.uid
    assert after.name == node.name
    check(node, after)


# --- the tree's shape --------------------------------------------------------


def test_the_trees_shape_and_sibling_order_survive_a_round_trip() -> None:
    """Order is both export order and outliner order (``nodes.py``'s own
    module docstring), so a save that reordered anything would silently
    reorder what the outliner shows on the very next open.
    """
    doc = _doc()
    out = _roundtrip(doc)

    assert [n.name for n in out.roots] == [n.name for n in doc.roots]

    before_outer = doc.roots[0]
    after_outer = out.roots[0]
    assert [c.name for c in after_outer.children] == [c.name for c in before_outer.children]

    before_inner = before_outer.children[0]
    after_inner = after_outer.children[0]
    assert [c.name for c in after_inner.children] == [c.name for c in before_inner.children]


# --- byte identity -------------------------------------------------------------


def test_two_saves_of_an_unchanged_document_are_byte_identical() -> None:
    doc = _doc()
    assert ser.rscn_bytes(doc) == ser.rscn_bytes(doc)


def test_a_save_read_save_round_trip_is_byte_identical_to_the_first_save() -> None:
    doc = _doc()
    first = ser.rscn_bytes(doc)
    again = ser.rscn_bytes(ser.read_rscn(first))
    assert again == first


# --- the terrain ---------------------------------------------------------------


def test_a_terrains_heights_come_back_f4_same_shape_same_values_and_a_valid_terrain() -> None:
    doc = _doc()
    out = _roundtrip(doc)

    assert out.terrain is not None
    assert out.terrain.heights.dtype == np.float32
    assert out.terrain.heights.shape == doc.terrain.heights.shape
    assert np.array_equal(out.terrain.heights, doc.terrain.heights)
    assert out.terrain.size_x == pytest.approx(doc.terrain.size_x)
    assert out.terrain.size_z == pytest.approx(doc.terrain.size_z)
    assert out.terrain.material.name == doc.terrain.material.name


# --- the dangling-reference story --------------------------------------------


def test_a_library_ref_to_a_missing_job_opens_and_is_only_listed_once_a_host_says_so() -> None:
    """This format never touches the filesystem and cannot know whether a job
    id still resolves -- see ``document.missing_refs``'s own docstring, and
    the module docstring's restatement of it. The node opens with its ref
    exactly as saved; only a host adding to ``doc.missing`` makes it appear as
    a known, repairable problem.
    """
    doc = _doc()
    out = _roundtrip(doc)
    node = next(n for n in out.all_nodes() if n.name == "RockB")
    assert isinstance(node.ref, LibraryRef)

    assert out.missing == set()
    assert out.missing_refs() == []

    out.missing.add(ref_key(node.ref))
    assert out.missing_refs() == [(node, node.ref)]


# --- materials shared by identity ---------------------------------------------


def test_two_nodes_sharing_one_material_share_one_object_after_a_load() -> None:
    """``write_glb`` and the viewport's ``GpuMaterial`` cache both de-duplicate
    a material by ``id()`` -- two nodes that shared one object before a save
    sharing two after it would silently double a scene's GPU uploads and its
    glTF material count on every round trip.
    """
    doc = _doc()
    out = _roundtrip(doc)
    by_name = {n.name: n for n in out.all_nodes()}
    assert by_name["RockA"].material is by_name["RockB"].material


# --- uids ----------------------------------------------------------------------


def test_a_node_minted_after_a_load_sits_above_every_restored_uid() -> None:
    """The uid is the address every undo step is written against
    (``nodes.py``'s own docstring); a fresh node minted onto a uid a restored
    one already wears would let a later edit resolve against whichever node a
    lookup happens to find first.
    """
    high = max(nd.new_uid() for _ in range(3)) + 10_000
    doc = MasonDoc(roots=[nd.GroupNode(uid=high, name="High")])

    out = _roundtrip(doc)
    assert out.roots[0].uid == high
    assert nd.new_uid() > high


# --- the frame-thread/task-thread split ---------------------------------------


def test_snapshot_performs_no_encoding_leaving_it_to_snapshot_bytes(monkeypatch) -> None:
    """``snapshot()`` is the frame-thread half and must never pay for a PNG
    encode or a zip build -- the clay-03 finding this stage must not repeat.
    Proved by monkeypatching every encoding entry point to raise and calling
    ``snapshot()`` anyway: a timing assertion could pass by accident on a fast
    machine, but a call to a poisoned function cannot.
    """

    def boom(*_args, **_kwargs):
        raise AssertionError("snapshot() must not perform any encoding")

    monkeypatch.setattr(ser, "_png_bytes", boom)
    monkeypatch.setattr(ser.zipfile, "ZipFile", boom)
    monkeypatch.setattr(np.lib.format, "write_array", boom)

    snap = ser.snapshot(_doc())

    assert isinstance(snap, ser.RscnSnapshot)
    assert snap.heights is not None
    assert len(snap.images) >= 1


def test_snapshot_bytes_of_a_snapshot_matches_rscn_bytes() -> None:
    """``rscn_bytes`` is ``snapshot_bytes(snapshot(doc))`` -- the two must
    never drift, or a caller choosing one path over the other would write a
    different file for the same document.
    """
    doc = _doc()
    assert ser.snapshot_bytes(ser.snapshot(doc)) == ser.rscn_bytes(doc)


def test_a_snapshot_is_what_the_document_was_when_the_save_was_pressed() -> None:
    doc = _doc()
    before = ser.rscn_bytes(doc)
    snap = ser.snapshot(doc)

    doc.roots.append(nd.GroupNode(uid=nd.new_uid(), name="Late"))
    doc.materials.append(gltf.Material(name="late-material"))

    assert ser.snapshot_bytes(snap) == before


# --- readability ---------------------------------------------------------------


def test_scene_json_is_sorted_indented_and_a_person_can_read_it() -> None:
    doc = _doc()
    raw = ser.scene_json(doc)
    scene = json.loads(raw)

    assert raw == json.dumps(scene, sort_keys=True, indent=2)
    assert list(scene.keys()) == sorted(scene.keys())
    assert scene["version"] == ser.VERSION
    assert [n["name"] for n in scene["nodes"]] == [n.name for n in doc.roots]


def test_a_document_with_nothing_extra_writes_a_minimal_scene_json() -> None:
    """No terrain, no prefabs, no textures, no view: the format stays readable
    for the simple case, carrying none of the keys that only mean something
    for a scene that has them.
    """
    doc = MasonDoc(roots=[nd.GroupNode(uid=nd.new_uid(), name="Only")])
    scene = json.loads(ser.scene_json(doc))

    assert "terrain" not in scene
    assert "prefabs" not in scene
    assert "textures" not in scene
    assert "view" not in scene
    assert scene["version"] == ser.VERSION
    assert scene["materials"] == []
    assert len(scene["nodes"]) == 1


def test_an_unknown_extra_key_in_a_node_entry_is_ignored() -> None:
    def mangle(scene: dict) -> None:
        _find_node(scene, "RockA")["a_future_field_this_build_has_never_heard_of"] = 42

    out = ser.read_rscn(_rewrite(ser.rscn_bytes(_doc()), mangle))
    assert any(n.name == "RockA" for n in out.all_nodes())


# --- refusals: the archive itself --------------------------------------------


def test_bytes_that_are_not_a_zip_are_refused() -> None:
    with pytest.raises(ValueError, match="not a Realmspinner Mason scene"):
        ser.read_rscn(b"this is not a zip file at all")


def test_a_zip_without_a_scene_is_refused() -> None:
    out = BytesIO()
    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr("something.txt", "hello")
    with pytest.raises(ValueError, match="not a Realmspinner Mason scene"):
        ser.read_rscn(out.getvalue())


def test_an_archive_claiming_more_than_the_ceiling_is_refused_before_it_is_read(
    monkeypatch,
) -> None:
    monkeypatch.setattr(ser, "MAX_DECOMPRESSED_BYTES", 16)
    with pytest.raises(ValueError, match="past the 16 this build will read"):
        ser.read_rscn(ser.rscn_bytes(_doc()))


def test_a_newer_version_is_refused_naming_both_version_numbers() -> None:
    def bump(scene: dict) -> None:
        scene["version"] = ser.VERSION + 1

    with pytest.raises(ValueError, match=f"format {ser.VERSION + 1}.*reads {ser.VERSION}"):
        ser.read_rscn(_rewrite(ser.rscn_bytes(_doc()), bump))


def test_a_terrain_naming_a_missing_height_file_is_refused() -> None:
    data = ser.rscn_bytes(_doc())
    stripped = _drop_members(data, lambda name: name == ser.TERRAIN_HEIGHTS)
    with pytest.raises(ValueError, match="height data is missing"):
        ser.read_rscn(stripped)


def test_a_material_naming_a_missing_texture_is_refused() -> None:
    data = ser.rscn_bytes(_doc())
    stripped = _drop_members(data, lambda name: name.startswith(f"{ser.TEXTURE_DIR}/"))
    with pytest.raises(ValueError, match="texture the file does not carry"):
        ser.read_rscn(stripped)


def test_a_scene_with_more_nodes_than_the_ceiling_is_refused(monkeypatch) -> None:
    """``MAX_NODES`` is imported from ``scene.MAX_PLACED`` rather than a
    number of its own, so this ceiling and the one ``scene.resolve`` refuses
    at are structurally one constant -- monkeypatching this module's own name
    is what a real corrupt file would trip, without building 100,000 nodes.
    """
    monkeypatch.setattr(ser, "MAX_NODES", 2)
    with pytest.raises(ValueError, match="nodes"):
        ser.read_rscn(ser.rscn_bytes(_doc()))


def test_a_scene_nested_deeper_than_the_ceiling_is_refused(monkeypatch) -> None:
    monkeypatch.setattr(nd, "MAX_DEPTH", 1)
    with pytest.raises(ValueError, match="deep"):
        ser.read_rscn(ser.rscn_bytes(_doc()))


# --- refusals: malformed fields ------------------------------------------------


def test_a_scenes_nodes_field_that_is_not_a_list_is_refused() -> None:
    def mangle(scene: dict) -> None:
        scene["nodes"] = None

    with pytest.raises(ValueError, match="not a Realmspinner Mason scene"):
        ser.read_rscn(_rewrite(ser.rscn_bytes(_doc()), mangle))


def test_a_scenes_materials_field_that_is_not_a_list_is_refused() -> None:
    def mangle(scene: dict) -> None:
        scene["materials"] = None

    with pytest.raises(ValueError, match="not a Realmspinner Mason scene"):
        ser.read_rscn(_rewrite(ser.rscn_bytes(_doc()), mangle))


def test_a_scenes_prefabs_field_that_is_not_a_mapping_is_refused() -> None:
    def mangle(scene: dict) -> None:
        scene["prefabs"] = ["not", "a", "mapping"]

    with pytest.raises(ValueError, match="not a Realmspinner Mason scene"):
        ser.read_rscn(_rewrite(ser.rscn_bytes(_doc()), mangle))


def test_a_node_that_is_not_a_mapping_is_refused() -> None:
    def mangle(scene: dict) -> None:
        scene["nodes"].append("not a mapping")

    with pytest.raises(ValueError, match="not a mapping"):
        ser.read_rscn(_rewrite(ser.rscn_bytes(_doc()), mangle))


def test_a_node_with_an_unknown_kind_is_refused() -> None:
    def mangle(scene: dict) -> None:
        _find_node(scene, "RockA")["kind"] = "blob"

    with pytest.raises(ValueError, match="kind"):
        ser.read_rscn(_rewrite(ser.rscn_bytes(_doc()), mangle))


def test_a_node_missing_its_uid_is_refused() -> None:
    def mangle(scene: dict) -> None:
        del _find_node(scene, "RockA")["uid"]

    with pytest.raises(ValueError, match="uid"):
        ser.read_rscn(_rewrite(ser.rscn_bytes(_doc()), mangle))


def test_a_node_whose_uid_is_not_a_number_is_refused() -> None:
    def mangle(scene: dict) -> None:
        _find_node(scene, "RockA")["uid"] = "not a number"

    with pytest.raises(ValueError, match="uid"):
        ser.read_rscn(_rewrite(ser.rscn_bytes(_doc()), mangle))


@pytest.mark.parametrize(
    ("key", "bad"),
    [
        ("translation", [1.0, 2.0]),
        ("rotation", [0.0, 0.0, 0.0]),
        ("scale", [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
    ],
)
def test_a_transform_of_the_wrong_shape_is_refused_rather_than_rendered(key, bad) -> None:
    """A three-element rotation is a quaternion with its ``w`` silently
    dropped -- the document opens, nothing looks obviously wrong, and every
    transform it drives is."""

    def mangle(scene: dict) -> None:
        _find_node(scene, "RockA")[key] = bad

    with pytest.raises(ValueError, match=key):
        ser.read_rscn(_rewrite(ser.rscn_bytes(_doc()), mangle))


def test_a_transform_that_is_not_numbers_is_refused() -> None:
    def mangle(scene: dict) -> None:
        _find_node(scene, "RockA")["translation"] = ["left", "up", "out"]

    with pytest.raises(ValueError, match="translation"):
        ser.read_rscn(_rewrite(ser.rscn_bytes(_doc()), mangle))


def test_a_nodes_children_that_is_not_a_list_is_refused() -> None:
    def mangle(scene: dict) -> None:
        _find_node(scene, "Group")["children"] = "not a list"

    with pytest.raises(ValueError, match="children"):
        ser.read_rscn(_rewrite(ser.rscn_bytes(_doc()), mangle))


def test_a_nodes_properties_that_is_not_a_mapping_is_refused() -> None:
    def mangle(scene: dict) -> None:
        _find_node(scene, "RockA")["properties"] = ["oops"]

    with pytest.raises(ValueError, match="properties"):
        ser.read_rscn(_rewrite(ser.rscn_bytes(_doc()), mangle))


def test_a_lights_kind_outside_the_three_khr_kinds_is_refused() -> None:
    def mangle(scene: dict) -> None:
        _find_node(scene, "Sun")["light_kind"] = "laser"

    with pytest.raises(ValueError, match="kind"):
        ser.read_rscn(_rewrite(ser.rscn_bytes(_doc()), mangle))


def test_a_meshs_ref_that_is_not_a_mapping_is_refused() -> None:
    def mangle(scene: dict) -> None:
        _find_node(scene, "RockA")["ref"] = "not a mapping"

    with pytest.raises(ValueError, match="ref"):
        ser.read_rscn(_rewrite(ser.rscn_bytes(_doc()), mangle))


def test_a_refs_kind_that_is_neither_primitive_nor_library_is_refused() -> None:
    def mangle(scene: dict) -> None:
        _find_node(scene, "RockA")["ref"] = {"kind": "mystery"}

    with pytest.raises(ValueError, match="kind"):
        ser.read_rscn(_rewrite(ser.rscn_bytes(_doc()), mangle))


def test_a_material_index_past_the_end_of_the_palette_is_refused() -> None:
    def mangle(scene: dict) -> None:
        _find_node(scene, "RockA")["material"] = 999

    with pytest.raises(ValueError, match="material"):
        ser.read_rscn(_rewrite(ser.rscn_bytes(_doc()), mangle))


def test_a_texture_index_past_the_end_of_the_textures_is_refused() -> None:
    def mangle(scene: dict) -> None:
        for material in scene["materials"]:
            if material.get("textures"):
                slot = next(iter(material["textures"]))
                material["textures"][slot] = 999
                return
        raise AssertionError("the fixture is expected to carry a textured material")

    with pytest.raises(ValueError, match="malformed"):
        ser.read_rscn(_rewrite(ser.rscn_bytes(_doc()), mangle))


# --- the saved camera ---------------------------------------------------------


def test_the_camera_the_scene_was_left_at_comes_back_with_it() -> None:
    """``doc.view`` is opaque to every other module in this package -- this is
    the one that writes it, so it is also the one that has to say the round
    trip works. A scene that reopened looking at the world origin every time
    would be a small, constant tax on picking work back up."""
    out = ser.read_rscn(ser.rscn_bytes(_doc()))
    assert out.view == {"yaw": 0.4, "pitch": 1.1, "distance": 7.5, "target": [1.0, 2.0, 3.0]}


def test_a_camera_with_a_key_missing_costs_the_camera_and_not_the_document() -> None:
    """A malformed camera is worth one unfitted viewport on open and never a
    refused scene: it is where somebody was standing, not any part of their
    work. The pinned key set is what makes "malformed" decidable at all --
    an unpinned dict would round-trip whatever a caller once put there and
    then be silently dropped by the next build that read the table instead."""

    def mangle(scene: dict) -> None:
        del scene["view"]["distance"]

    out = ser.read_rscn(_rewrite(ser.rscn_bytes(_doc()), mangle))
    assert out.view == {}
    assert len(out.all_nodes()) == len(_doc().all_nodes()), "the scene itself is untouched"


def test_a_camera_carrying_an_infinity_is_dropped_rather_than_restored() -> None:
    """A non-finite yaw reaches a view matrix as a frame of NaNs -- a viewport
    that draws nothing with no error anywhere to say why, which is worse than
    the one unfitted frame that dropping it costs."""

    def mangle(scene: dict) -> None:
        scene["view"]["yaw"] = float("inf")

    assert ser.read_rscn(_rewrite(ser.rscn_bytes(_doc()), mangle)).view == {}


def test_a_key_the_format_does_not_carry_is_not_written_and_not_read_back() -> None:
    """The file format's own rule, not the dict's: an unrecognised key would
    round-trip once and then vanish the day a build read ``VIEW_FIELDS``
    instead of whatever the caller happened to hand over."""
    doc = _doc()
    doc.view = dict(doc.view, fov=0.9, invented="whatever")
    scene = json.loads(ser.scene_json(doc))
    assert set(scene["view"]) == set(ser.VIEW_FIELDS) | {"target"}
    assert "fov" not in ser.read_rscn(ser.rscn_bytes(doc)).view
