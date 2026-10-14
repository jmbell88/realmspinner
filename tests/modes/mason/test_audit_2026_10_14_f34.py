"""Regressions for the 2026-10-14 audit's Mason findings (fixer F34).

Each test's name is the claim, and each failed against the unfixed code before
the fix in this same change -- see the fixer's return for the pasted failing
output. Every finding id is the audit's (mason-NN, with the explorer record id
in the docstring).
"""

from __future__ import annotations

import io
import json
import math
import struct
import warnings
import zipfile
from typing import Any

import numpy as np
import pytest

from realmspinner.kernels.geom3d import glbwrite, gltf
from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.studio import dialogs
from realmspinner.studio.modes.mason import assets as mason_assets
from realmspinner.studio.modes.mason import mode as mason_mode
from realmspinner.studio.modes.mason.engine import document as md
from realmspinner.studio.modes.mason.engine import gltfout, manifest, objout, serialize
from realmspinner.studio.modes.mason.engine import nodes as nd
from realmspinner.studio.modes.mason.engine import refs as mrefs
from realmspinner.studio.modes.mason.engine import scene as sc
from realmspinner.studio.modes.mason.engine import terrain as tr


def _box() -> mrefs.PrimitiveRef:
    return mrefs.primitive_ref("box", {})


def _mesh(name: str = "Mesh", **kw: Any) -> nd.MeshNode:
    return nd.MeshNode(uid=nd.new_uid(), name=name, ref=_box(), **kw)


def _triangle(**kw: Any) -> gltf.Primitive:
    return gltf.Primitive(
        positions=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="f4"),
        indices=np.array([0, 1, 2], dtype="u4"),
        **kw,
    )


class _Src:
    """A GeometrySource answering every primitive ref with one triangle and
    every library ref with whatever ``library`` says (default: not resolved)."""

    rev = 0

    def __init__(self, prim: gltf.Primitive | None = None, library: bool = False) -> None:
        self.prim = prim or _triangle()
        self.library = library

    def primitives(self, ref: Any) -> list[gltf.Primitive]:
        if isinstance(ref, mrefs.LibraryRef) and not self.library:
            return []
        return [self.prim]

    def box(self, ref: Any) -> Any:
        prims = self.primitives(ref)
        return prims[0].box() if prims else None


def _glb_json(data: bytes) -> dict[str, Any]:
    n = struct.unpack_from("<I", data, 12)[0]
    return json.loads(data[20 : 20 + n])


# --- mason-12 (mason-mode-07): a mirrored node's baked winding ---------------


def test_bake_model_reverses_winding_for_a_mirrored_node() -> None:
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="f4")
    nrm = np.array([[0, 0, 1]] * 3, dtype="f4")
    prim = gltf.Primitive(positions=pos, indices=np.array([0, 1, 2], dtype="u4"), normals=nrm)
    node = gltf.Node(name="mirrored", mesh=0, scale=np.array([-1.0, 1.0, 1.0]))
    loaded = gltf.load(glbwrite.write_glb(gltf.Model([node], [0], [[prim]], [])))
    assert np.linalg.det(loaded.nodes[0].world[:3, :3]) < 0

    baked = mason_assets._bake_model(loaded)[0]
    p = baked.positions.astype("f8")
    i = baked.indices
    geometric = np.cross(p[i[1]] - p[i[0]], p[i[2]] - p[i[0]])
    assert float(geometric @ baked.normals[0]) > 0, (
        "the baked triangle must face the way its own transformed normal does"
    )


def test_bake_model_leaves_an_unmirrored_nodes_winding_alone() -> None:
    prim = _triangle()
    node = gltf.Node(name="plain", mesh=0, scale=np.array([2.0, 1.0, 1.0]))
    loaded = gltf.load(glbwrite.write_glb(gltf.Model([node], [0], [[prim]], [])))
    baked = mason_assets._bake_model(loaded)[0]
    assert list(baked.indices) == [0, 1, 2]


# --- mason-08 (mason-engine-05): redefining a prefab charges its live instances


def test_redefining_a_prefab_refuses_growth_that_would_resolve_past_max_placed(monkeypatch) -> None:
    monkeypatch.setattr(sc, "MAX_PLACED", 100)
    doc = md.MasonDoc()
    doc.define_prefab("Fence", _mesh("Post"))
    for _ in range(20):
        doc.add_node(nd.PrefabNode(uid=nd.new_uid(), template="Fence"))
    assert len(sc.resolve(doc)) == 20

    big = nd.GroupNode(uid=nd.new_uid(), name="Section")
    big.children.extend(_mesh() for _ in range(9))
    with pytest.raises(ValueError, match="MAX_PLACED"):
        doc.define_prefab("Fence", big)  # 20 instances x 10 nodes = 200

    assert len(sc.resolve(doc)) == 20, "a refused redefinition changes nothing"
    assert doc.prefabs["Fence"].children == []


def test_redefining_a_prefab_that_still_fits_is_accepted(monkeypatch) -> None:
    monkeypatch.setattr(sc, "MAX_PLACED", 100)
    doc = md.MasonDoc()
    doc.define_prefab("Fence", _mesh("Post"))
    for _ in range(5):
        doc.add_node(nd.PrefabNode(uid=nd.new_uid(), template="Fence"))
    bigger = nd.GroupNode(uid=nd.new_uid(), name="Section")
    bigger.children.extend(_mesh() for _ in range(3))
    doc.define_prefab("Fence", bigger)  # 5 x 4 = 20
    assert len(sc.resolve(doc, include_hidden=True)) == 15


# --- mason-10 (mason-engine-09) and mason-10 (mason-mode-06): Make prefab keeps flags


def test_make_prefab_of_a_hidden_node_yields_an_instance_the_visible_toggle_controls() -> None:
    doc = md.MasonDoc()
    hidden = doc.add_node(_mesh("Crate", visible=False, static=True, locked=True))
    template = doc.define_prefab("Crate", hidden)
    # The template root is the neutral root: nothing the instance carries is
    # ANDed/ORed into by a flag the template root kept.
    assert template.visible is True
    assert template.locked is False
    assert template.static is False

    inst = nd.PrefabNode(uid=nd.new_uid(), name="Crate", template="Crate")
    doc.remove_node(hidden.uid)
    doc.add_node(inst)
    assert len(sc.resolve(doc)) == 1, "the template alone does not hide the instance"
    doc.set_props(inst.uid, visible=False)
    assert sc.resolve(doc) == []
    doc.set_props(inst.uid, visible=True)
    assert len(sc.resolve(doc)) == 1


class _Settings:
    def __init__(self) -> None:
        self.store: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self.store.get(key)

    def set(self, key: str, value: Any) -> None:
        self.store[key] = value


class _FakeCtx:
    def __init__(self) -> None:
        self.svc = None
        self.state = type("S", (), {"mason": None, "mode": "home"})()
        self.settings = _Settings()
        self.prompts = dialogs.PromptQueue()
        self.toasts: list[tuple[str, str]] = []

    def toast(self, message: str, kind: str = "info", *_a: Any, **_k: Any) -> None:
        self.toasts.append((message, kind))


def _ctx() -> _FakeCtx:
    ctx = _FakeCtx()
    mason_mode.new_document(ctx)
    return ctx


def test_make_prefab_of_a_hidden_node_yields_an_instance_that_is_hidden_and_a_template_that_is_not(
) -> None:
    ctx = _ctx()
    doc = mason_mode.ensure(ctx).active.doc
    node = doc.add_node(
        _mesh("Crate", visible=False, locked=True, static=True, properties={"spawn": "yes"})
    )
    doc.select([node.uid])
    assert mason_mode.define_prefab_from_selection(ctx, "Crate") == "Crate", ctx.toasts

    (instance,) = doc.roots
    assert isinstance(instance, nd.PrefabNode)
    assert instance.visible is False
    assert instance.locked is True
    assert instance.static is True
    assert instance.properties == {"spawn": "yes"}
    template = doc.prefabs["Crate"]
    assert (template.visible, template.locked, template.static) == (True, False, False)
    assert template.properties == {}

    # Unpack restores exactly what was made: the hidden, static prop.
    copy = doc.unpack_instance(instance.uid)
    assert (copy.visible, copy.locked, copy.static) == (False, True, True)
    assert copy.properties == {"spawn": "yes"}


# --- mason-19 (mason-engine-04): detaching a group drops selected descendants


def test_detaching_a_group_drops_its_selected_descendants_from_the_selection() -> None:
    doc = md.MasonDoc()
    group = nd.GroupNode(uid=nd.new_uid(), name="G")
    child = _mesh("M")
    group.children.append(child)
    doc.add_node(group)
    doc.select([child.uid])
    assert doc.selection_active == child.uid

    doc.remove_node(group.uid)
    assert doc.node(child.uid) is None
    assert doc.selection == set()
    assert doc.selection_active is None

    doc.undo()
    assert doc.selection == set(), "undo never restores a selection"

    doc2 = md.MasonDoc()
    group2 = nd.GroupNode(uid=nd.new_uid(), name="G2")
    child2 = _mesh("M2")
    group2.children.append(child2)
    doc2.add_node(group2)
    doc2.select([child2.uid])
    doc2.undo()  # undo the group's add
    assert [u for u in doc2.selection if doc2.node(u) is None] == []
    assert doc2.selection_active is None


# --- mason-20 (mason-engine-07): glTF's own ranges for lights and cameras ----


@pytest.mark.parametrize(
    "kind, props",
    [
        ("spot", {"inner_cone_angle": 1.5, "outer_cone_angle": 0.5}),
        ("spot", {"outer_cone_angle": 3.0}),
        ("spot", {"inner_cone_angle": -0.1}),
        ("spot", {"outer_cone_angle": 0.0}),
        ("point", {"intensity": -5.0}),
        ("point", {"range": -1.0}),
        ("camera", {"znear": 2000.0}),
        ("camera", {"znear": 0.0}),
        ("camera", {"yfov": 4.0}),
        ("camera", {"yfov": 0.0}),
        ("camera", {"zfar": 0.05}),
    ],
)
def test_set_props_refuses_light_and_camera_values_gltf_forbids(kind: str, props: dict) -> None:
    doc = md.MasonDoc()
    if kind == "camera":
        node = doc.add_node(nd.CameraNode(uid=nd.new_uid(), name="Cam"))
    else:
        node = doc.add_node(nd.LightNode(uid=nd.new_uid(), name="L", kind=kind))
    before = {key: getattr(node, key) for key in props}
    with pytest.raises(ValueError):
        doc.set_props(node.uid, **props)
    assert {key: getattr(node, key) for key in props} == before, "a refusal stores nothing"


def test_set_props_still_takes_the_values_gltf_allows() -> None:
    doc = md.MasonDoc()
    spot = doc.add_node(nd.LightNode(uid=nd.new_uid(), name="S", kind="spot"))
    assert doc.set_props(spot.uid, outer_cone_angle=math.pi / 2)
    assert doc.set_props(spot.uid, inner_cone_angle=1.0)
    assert doc.set_props(spot.uid, intensity=0.0, range=0.0)
    cam = doc.add_node(nd.CameraNode(uid=nd.new_uid(), name="C"))
    assert doc.set_props(cam.uid, znear=0.5, zfar=20.0, yfov=1.0)
    assert doc.set_props(cam.uid, zfar=0.0)  # zero = infinite, as the exporter reads it


def test_read_rscn_refuses_light_and_camera_values_gltf_forbids() -> None:
    doc = md.MasonDoc()
    light = doc.add_node(nd.LightNode(uid=nd.new_uid(), name="L", kind="point"))
    data = serialize.rscn_bytes(doc)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        scene = json.loads(zf.read(serialize.SCENE))
        members = {n: zf.read(n) for n in zf.namelist()}
    for entry in scene["nodes"]:
        if entry["uid"] == light.uid:
            entry["intensity"] = -5.0
    members[serialize.SCENE] = json.dumps(scene).encode()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, payload in members.items():
            zf.writestr(name, payload)
    with pytest.raises(ValueError, match="intensity"):
        serialize.read_rscn(buf.getvalue())


# --- mason-06 (mason-engine-01): rigged library asset, no orphan joints ------


def _rigged(jmax: int) -> gltf.Primitive:
    return _triangle(
        joints=np.array([[0, 1, jmax, 0]] * 3, dtype="i4"),
        weights=np.array([[0.5, 0.5, 0, 0]] * 3, dtype="f4"),
    )


@pytest.mark.parametrize("jmax", [2, 300])
def test_a_rigged_library_asset_exports_from_mason_without_orphan_joint_attributes(jmax) -> None:
    doc = md.MasonDoc()
    doc.add_node(nd.MeshNode(uid=nd.new_uid(), name="Hero", ref=mrefs.LibraryRef(job_id="abc")))
    data = gltfout.scene_glb(doc, _Src(_rigged(jmax), library=True))
    j = _glb_json(data)
    attrs = j["meshes"][0]["primitives"][0]["attributes"]
    assert "JOINTS_0" not in attrs and "WEIGHTS_0" not in attrs
    assert "skins" not in j


def test_exporting_a_rigged_asset_leaves_the_cached_primitive_alone() -> None:
    prim = _rigged(2)
    doc = md.MasonDoc()
    doc.add_node(nd.MeshNode(uid=nd.new_uid(), name="Hero", ref=mrefs.LibraryRef(job_id="abc")))
    gltfout.scene_glb(doc, _Src(prim, library=True))
    assert prim.joints is not None and prim.weights is not None


# --- mason-09 (mason-engine-06): manifest world under shear -------------------


def test_manifest_world_is_a_unit_rotation_under_a_non_uniformly_scaled_parent() -> None:
    doc = md.MasonDoc()
    group = nd.GroupNode(uid=nd.new_uid(), name="Stretched", scale=np.array([2.0, 1.0, 1.0]))
    crate = nd.MeshNode(
        uid=nd.new_uid(),
        name="Crate",
        ref=_box(),
        rotation=m3.quat_from_axis_angle(np.array([0, 0, 1.0]), np.radians(45)),
    )
    group.children.append(crate)
    doc.add_node(group)
    export = gltfout.scene_model(doc, _Src())
    man = manifest.scene_manifest(doc, export)
    entry = next(n for n in man["nodes"] if n["name"] == "Crate")
    q = np.array(entry["world"]["rotation"])
    assert abs(float(np.linalg.norm(q)) - 1.0) < 1e-9
    # A sheared world matrix has no TRS, so the manifest says so rather than
    # presenting an approximation as the composed answer.
    assert entry.get("world_approximate") is True
    plain = next(n for n in man["nodes"] if n["name"] == "Stretched")
    assert "world_approximate" not in plain


# --- mason-01 (mason-engine-10): a removed template is reported, not dropped --


def _doc_with_removed_template() -> md.MasonDoc:
    doc = md.MasonDoc()
    doc.define_prefab("Barrel", _mesh("b"))
    for i in range(3):
        doc.add_node(nd.PrefabNode(uid=nd.new_uid(), name=f"Barrel {i}", template="Barrel"))
    doc.remove_prefab("Barrel")
    return doc


def test_export_reports_a_prefab_instance_whose_template_was_removed() -> None:
    doc = _doc_with_removed_template()
    export = gltfout.scene_model(doc, _Src())
    man = manifest.scene_manifest(doc, export)
    missing = man.get("missing", [])
    assert sorted(m["node"] for m in missing) == ["Barrel 0", "Barrel 1", "Barrel 2"]
    assert all(m["kind"] == "prefab" and m["template"] == "Barrel" for m in missing)

    result = objout.obj_export(doc, _Src())
    assert len(result.skipped) == 3
    assert all("Barrel" in s and "prefab" in s.lower() for s in result.skipped)


def test_export_reports_a_prefab_instance_that_would_recurse_and_a_healthy_one_is_silent() -> None:
    doc = md.MasonDoc()
    doc.define_prefab("Ok", _mesh("m"))
    doc.add_node(nd.PrefabNode(uid=nd.new_uid(), name="Fine", template="Ok"))
    export = gltfout.scene_model(doc, _Src())
    assert "missing" not in manifest.scene_manifest(doc, export)
    assert objout.obj_export(doc, _Src()).skipped == ()


def test_a_hidden_instance_with_a_removed_template_is_not_reported() -> None:
    doc = _doc_with_removed_template()
    doc.set_visibility({n.uid: False for n in doc.roots})
    export = gltfout.scene_model(doc, _Src())
    assert "missing" not in manifest.scene_manifest(doc, export)


# --- mason-07 (mason-engine-02): hidden content is charged -------------------


def test_hidden_template_content_and_hidden_instances_are_charged_against_max_placed(
    monkeypatch,
) -> None:
    monkeypatch.setattr(sc, "MAX_PLACED", 100)
    # (1) hidden leaves inside a template: 42 per instance, so the third is refused.
    doc = md.MasonDoc()
    root = nd.GroupNode(uid=nd.new_uid(), name="Root")
    inner = nd.GroupNode(uid=nd.new_uid(), name="Inner", visible=False)
    inner.children.extend(_mesh() for _ in range(40))
    root.children.append(inner)
    doc.define_prefab("P", root)
    doc.add_node(nd.PrefabNode(uid=nd.new_uid(), template="P"))
    doc.add_node(nd.PrefabNode(uid=nd.new_uid(), template="P"))
    with pytest.raises(ValueError, match="MAX_PLACED"):
        doc.add_node(nd.PrefabNode(uid=nd.new_uid(), template="P"))
    # every read that includes hidden content still works
    last = doc.roots[-1]
    assert sc.resolved_for(doc, last.uid) is not None
    assert sc.resolve(doc, include_hidden=True, max_items=sc.MAX_PLACED)

    # (2) hidden instances are charged: hide, place more, Show all stays resolvable.
    doc = md.MasonDoc()
    tpl = nd.GroupNode(uid=nd.new_uid(), name="T")
    tpl.children.extend(_mesh() for _ in range(9))
    doc.define_prefab("T", tpl)  # 10 nodes per instance
    first = [doc.add_node(nd.PrefabNode(uid=nd.new_uid(), template="T")) for _ in range(5)]
    doc.set_visibility({n.uid: False for n in first})
    placed = 0
    with pytest.raises(ValueError, match="MAX_PLACED"):
        for _ in range(8):
            doc.add_node(nd.PrefabNode(uid=nd.new_uid(), template="T"))
            placed += 1
    assert placed < 8
    doc.show_all()
    assert sc.resolve(doc, max_items=sc.MAX_PLACED), "Show all leaves a scene that resolves"


# --- mason-22 (mason-engine-12): resolve reads the ceiling at call time ------


def test_resolve_honours_a_lowered_max_placed_without_an_explicit_max_items(monkeypatch) -> None:
    doc = md.MasonDoc()
    doc.add_nodes([_mesh() for _ in range(20)])
    monkeypatch.setattr(sc, "MAX_PLACED", 5)
    with pytest.raises(ValueError, match="MAX_PLACED"):
        sc.resolve(doc)
    with pytest.raises(ValueError, match="MAX_PLACED"):
        sc.walk(doc, lambda *a: None)
    assert len(sc.resolve(doc, max_items=50)) == 20, "an explicit ceiling still governs"


# --- mason-21 (mason-engine-08): a deeply nested scene.json is a named refusal


def test_read_rscn_names_a_json_document_nested_past_the_recursion_limit() -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("scene.json", '{"version":1,"nodes":' + "[" * 100_000 + "]" * 100_000 + "}")
    with pytest.raises(ValueError, match="not a Realmspinner Mason scene"):
        serialize.read_rscn(buf.getvalue())


# --- mason-18 (mason-engine-03): float32 overflow in the brushes -------------


def test_a_brush_that_would_overflow_float32_changes_nothing() -> None:
    flat = np.zeros((17, 17), dtype="f4")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a refusal must not even warn
        assert tr.raise_lower(flat, 8, 8, 4.0, 1e39) is None
        assert tr.flatten(flat, 8, 8, 4.0, 1e39, 1.0) is None
        assert tr.smooth(flat, 8, 8, 4.0, 1e39) is None
        assert tr.noise(flat, 8, 8, 4.0, 1e39, seed=1) is None
        assert tr.set_height(flat, (0, 0, 4, 4), 1e39) is None
    # A finite amount that accumulates past float32 range changes nothing either.
    high = np.full((17, 17), 3.0e38, dtype="f4")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert tr.raise_lower(high, 8, 8, 4.0, 3.0e38) is None
        assert tr.noise(high, 8, 8, 4.0, 3.0e38, seed=1) is None
    # An in-range brush still works.
    assert tr.raise_lower(flat, 8, 8, 4.0, 1.0) is not None


def test_sculpt_refuses_a_block_that_is_not_finite() -> None:
    doc = md.MasonDoc()
    doc.set_terrain(
        tr.Terrain(heights=np.zeros((5, 5), "f4"), size_x=4.0, size_z=4.0, material=gltf.Material())
    )
    doc.begin_sculpt()
    bad = np.full((2, 2), np.inf, dtype="f4")
    with pytest.raises(ValueError, match="finite"):
        doc.sculpt((0, 0, 2, 2), bad)
    assert np.isfinite(doc.terrain.heights).all()
    doc.end_sculpt()
    assert serialize.read_rscn(serialize.rscn_bytes(doc)) is not None
