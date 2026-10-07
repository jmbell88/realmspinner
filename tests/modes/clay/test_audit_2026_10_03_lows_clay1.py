"""The 2026-10-03 audit's Low findings owned by fixer clay1 (clay/agent-tools,
clay/io, clay/mesh-ops, clay/mode, clay/panes, clay/ops-tail).

A test's name is the claim; each one is written to fail against the unfixed
code for the reason its finding gives.
"""

from __future__ import annotations

import io
import json
import struct

import numpy as np
import pytest

from realmspinner.kernels.geom3d import gltf
from realmspinner.kernels.geom3d.glbio import rebuild_glb

# ---------------------------------------------------------------------------
# clay-86: every top-level array / texture shape refuses by name
# ---------------------------------------------------------------------------


def _glb_bytes(doc: dict, binary: bytes) -> bytes:
    binary += b"\x00" * (-len(binary) % 4)
    chunk = struct.pack("<II", len(binary), 0x004E4942) + binary
    header = struct.pack("<III", 0x46546C67, 2, 0)
    return rebuild_glb(header, doc, chunk)


def _png() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGBA", (2, 2), (255, 0, 0, 255)).save(buf, "PNG")
    return buf.getvalue()


def _textured_triangle() -> tuple[dict, bytes]:
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="<f4").tobytes()
    png = _png()
    pad = b"\x00" * (-len(positions) % 4)
    binary = positions + pad + png
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "material": 0}]}],
        "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
        "textures": [{"source": 0}],
        "images": [{"bufferView": 1}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(positions)},
            {"buffer": 0, "byteOffset": len(positions) + len(pad), "byteLength": len(png)},
        ],
        "buffers": [{"byteLength": len(binary)}],
    }
    return doc, binary


def test_the_baseline_textured_triangle_loads():
    doc, binary = _textured_triangle()
    model = gltf.load(_glb_bytes(doc, binary))
    assert len(model.meshes[0]) == 1


@pytest.mark.parametrize("field", ["accessors", "bufferViews", "textures", "images", "skins"])
@pytest.mark.parametrize("bad", [None, 7, {"a": 1}, "x"])
def test_every_top_level_array_field_refuses_a_non_list_by_name(field, bad):
    doc, binary = _textured_triangle()
    doc[field] = bad
    with pytest.raises(ValueError, match=field):
        gltf.load(_glb_bytes(doc, binary))


def test_a_mesh_with_null_primitives_loads_as_an_empty_mesh_not_a_typeerror():
    # The declared-primitives pass reads ``primitives or []``; the decode loop
    # read ``primitives`` bare, so a null passed the first and raised
    # TypeError in the second.
    doc, binary = _textured_triangle()
    doc["meshes"] = [{"primitives": None}]
    doc["nodes"] = [{}]
    model = gltf.load(_glb_bytes(doc, binary))
    assert model.meshes == [[]]


@pytest.mark.parametrize("key", ["byteOffset", "byteLength"])
@pytest.mark.parametrize("bad", [4.0, "4", [4], None])
def test_an_image_bufferview_with_a_non_integer_extent_refuses_by_name(key, bad):
    doc, binary = _textured_triangle()
    doc["bufferViews"][1][key] = bad
    with pytest.raises(ValueError, match=key):
        gltf.load(_glb_bytes(doc, binary))


@pytest.mark.parametrize("slot", ["baseColorTexture", "normalTexture"])
@pytest.mark.parametrize("bad", [5, "x", [1]])
def test_a_non_object_texture_reference_refuses_by_name(slot, bad):
    doc, binary = _textured_triangle()
    mat = doc["materials"][0]
    if slot == "baseColorTexture":
        mat["pbrMetallicRoughness"]["baseColorTexture"] = bad
    else:
        mat["normalTexture"] = bad
    with pytest.raises(ValueError, match="texture"):
        gltf.load(_glb_bytes(doc, binary))


# ---------------------------------------------------------------------------
# clay-87: _declared_budget may not raise
# ---------------------------------------------------------------------------


def _raw_glb(json_bytes: bytes) -> bytes:
    json_bytes += b" " * (-len(json_bytes) % 4)
    total = 12 + 8 + len(json_bytes)
    return (
        struct.pack("<III", 0x46546C67, 2, total)
        + struct.pack("<II", len(json_bytes), 0x4E4F534A)
        + json_bytes
    )


@pytest.mark.parametrize("attrs", [[1, 2], "POSITION", 7])
def test_declared_budget_survives_a_non_object_attributes_and_a_deeply_nested_json_chunk(attrs):
    from realmspinner.kernels.mesh import glbimport

    doc = {
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": attrs}]}],
    }
    assert glbimport._declared_budget(_raw_glb(json.dumps(doc).encode())) == (0, 1)


def test_a_deeply_nested_json_chunk_reads_as_nothing_declared_and_refuses_by_name():
    from realmspinner.kernels.geom3d import glbio
    from realmspinner.kernels.mesh import glbimport
    from realmspinner.kernels.mesh.elements import OpError

    deep = _raw_glb(b"[" * 100_000 + b"]" * 100_000)
    with pytest.raises(ValueError):
        glbio.split_glb(deep)
    assert glbimport._declared_budget(deep) == (0, 0)
    with pytest.raises(OpError):
        glbimport.glb_to_claydoc(deep)


# ---------------------------------------------------------------------------
# clay-89: material-less primitives share one palette slot
# ---------------------------------------------------------------------------


def _untextured_meshes(n: int) -> bytes:
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="<f4").tobytes()
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": list(range(n))}],
        "nodes": [{"mesh": i} for i in range(n)],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]} for _ in range(n)],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(positions)}],
        "buffers": [{"byteLength": len(positions)}],
    }
    return _glb_bytes(doc, positions)


def test_material_less_primitives_share_one_palette_slot():
    from realmspinner.kernels.mesh import glbimport

    doc = glbimport.glb_to_claydoc(_untextured_meshes(40))
    assert len(doc.objects) == 40
    assert len(doc.materials) == 1
    assert {obj.material for obj in doc.objects} == {0}


def test_a_declared_material_still_gets_its_own_slot_beside_the_shared_fallback():
    from realmspinner.kernels.mesh import glbimport

    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="<f4").tobytes()
    prim = {"attributes": {"POSITION": 0}}
    gdoc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0, 1, 2]}],
        "nodes": [{"mesh": 0}, {"mesh": 1}, {"mesh": 2}],
        "meshes": [
            {"primitives": [dict(prim)]},
            {"primitives": [{**prim, "material": 0}]},
            {"primitives": [dict(prim)]},
        ],
        "materials": [{"name": "real"}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(positions)}],
        "buffers": [{"byteLength": len(positions)}],
    }
    doc = glbimport.glb_to_claydoc(_glb_bytes(gdoc, positions))
    assert len(doc.materials) == 2
    assert [o.material for o in doc.objects].count(doc.objects[0].material) == 2


# ---------------------------------------------------------------------------
# clay-93: bevel width 0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("width", [0.0, -0.1, float("nan")])
def test_bevel_refuses_a_degenerate_width(width):
    from realmspinner.kernels.mesh import elements as el
    from realmspinner.kernels.mesh import ops_bevel as ob
    from realmspinner.kernels.mesh import primitives as prim
    from realmspinner.kernels.mesh.adjacency import adjacency

    box = prim.box()
    a = adjacency(box)
    edge = a.edge_verts[a.edge_uses == 2][0]
    with pytest.raises(el.OpError, match="width"):
        ob.bevel_edges(box, el.ElementSel(edges=[edge]), width=width)


def test_bevel_still_takes_a_real_width():
    from realmspinner.kernels.mesh import elements as el
    from realmspinner.kernels.mesh import ops_bevel as ob
    from realmspinner.kernels.mesh import primitives as prim
    from realmspinner.kernels.mesh.adjacency import adjacency

    box = prim.box()
    a = adjacency(box)
    out, _ = ob.bevel_edges(
        box, el.ElementSel(edges=[a.edge_verts[a.edge_uses == 2][0]]), width=0.01
    )
    assert len(out.positions) > len(box.positions)


# ---------------------------------------------------------------------------
# clay-96: a failed save unlocks the tab it was saving
# ---------------------------------------------------------------------------


@pytest.fixture
def _no_pygame_display(monkeypatch):
    import pygame

    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0)


@pytest.mark.parametrize("name", ["clay-save", "clay-saveas", "clay-export", "clay-exportfile"])
def test_a_failed_save_unlocks_the_tab(svc, _no_pygame_display, name):
    from realmspinner.studio.modes.clay import mode as clay_mode

    from .test_clay_mode import FakeCtx, _Done, _tab

    ctx = FakeCtx(svc)
    tab = _tab(ctx)
    tab.saving = True
    clay_mode.on_task_failed(ctx, _Done(f"{name}:{tab.uid}", message="disk full"))
    assert tab.saving is False, "the failed save unlocks the tab"


# ---------------------------------------------------------------------------
# clay-102 / clay-119: manual and docstring wording
# ---------------------------------------------------------------------------


def _clay_chapter() -> str:
    from pathlib import Path

    import realmspinner

    root = Path(realmspinner.__file__).resolve().parents[2]
    return " ".join((root / "docs" / "manual" / "30-clay.md").read_text(encoding="utf-8").split())


def test_manual_describes_import_as_one_object_per_primitive():
    text = _clay_chapter()
    assert "comes in as one object per material" not in text
    assert "comes in as one object per primitive" in text


def test_glbimport_docstring_points_at_the_loader_that_exists():
    from realmspinner.kernels.mesh import glbimport

    assert "~..viewer.gltf.load" not in glbimport.__doc__
    assert "kernels/geom3d/gltf.py" in glbimport.__doc__ or "geom3d.gltf" in glbimport.__doc__


def test_manual_clay_empty_state_matches_the_bridge_panes_empty_branch():
    text = _clay_chapter()
    assert "offers the same two buttons" not in text
    assert "shows only that recent list until a document is open" in text


# ---------------------------------------------------------------------------
# clay-110: a recovered document reopens with the camera the journal stored
# ---------------------------------------------------------------------------


def test_a_recovered_clay_document_reopens_with_the_camera_the_journal_stored(
    svc, _no_pygame_display, tmp_path
):
    from realmspinner.studio.modes.clay import mode as clay_mode

    from .test_clay_mode import FakeCtx, _Done, _tab

    ctx = FakeCtx(svc)
    tab = _tab(ctx)
    tab.view.yaw, tab.view.pitch, tab.view.distance = 1.25, -0.5, 7.5
    tab.view.target = [1.0, 2.0, 3.0]
    crash_copy = tmp_path / "crash.rblk"
    crash_copy.write_bytes(clay_mode._journal_encode(tab))

    result = clay_mode._load_recovery(crash_copy, {"title": "Scene"})
    clay_mode.on_task_done(ctx, _Done(f"clay-recover:{crash_copy.name}", result))

    recovered = ctx.state.clay.active
    assert recovered is not tab
    assert (recovered.view.yaw, recovered.view.pitch, recovered.view.distance) == (
        1.25,
        -0.5,
        7.5,
    )
    assert list(recovered.view.target) == [1.0, 2.0, 3.0]
    assert recovered.view.fitted, "an auto-fit over the top would throw the stored camera away"


# ---------------------------------------------------------------------------
# clay-111: a dotted stem keeps its dot
# ---------------------------------------------------------------------------


def _picked(monkeypatch, path):
    from realmspinner.studio import dialogs

    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: path)


def test_save_as_keeps_a_dotted_stem_instead_of_replacing_it_with_the_extension(
    svc, _no_pygame_display, monkeypatch, tmp_path
):
    from realmspinner.studio.modes.clay import mode as clay_mode

    from .test_clay_mode import FakeCtx, _tab

    out = tmp_path / "out"
    out.mkdir()
    ctx = FakeCtx(svc)
    tab = _tab(ctx)
    bystander = out / "barrel.rblk"
    bystander.write_bytes(b"someone else's document")
    _picked(monkeypatch, out / "barrel.v2")
    clay_mode.save_as(ctx, tab)
    assert (out / "barrel.v2.rblk").is_file()
    assert bystander.read_bytes() == b"someone else's document"


def test_save_as_still_takes_the_extension_the_user_typed_when_it_is_right(
    svc, _no_pygame_display, monkeypatch, tmp_path
):
    from realmspinner.studio.modes.clay import mode as clay_mode

    from .test_clay_mode import FakeCtx, _tab

    out = tmp_path / "out"
    out.mkdir()
    ctx = FakeCtx(svc)
    tab = _tab(ctx)
    _picked(monkeypatch, out / "barrel.RBLK")
    clay_mode.save_as(ctx, tab)
    assert not (out / "barrel.RBLK.rblk").exists()
    assert [p.name.lower() for p in out.iterdir()] == ["barrel.rblk"]


@pytest.mark.parametrize("kind", ["glb", "obj"])
def test_export_keeps_a_dotted_stem_instead_of_replacing_it_with_the_extension(
    svc, _no_pygame_display, monkeypatch, tmp_path, kind
):
    from realmspinner.studio.modes.clay import mode as clay_mode

    from .test_clay_mode import FakeCtx, _tab

    out = tmp_path / "out"
    out.mkdir()
    ctx = FakeCtx(svc)
    tab = _tab(ctx)
    _picked(monkeypatch, out / "barrel.v2")
    clay_mode.export_mesh_file(ctx, tab, kind)
    names = sorted(p.name for p in out.iterdir())
    assert f"barrel.v2.{kind}" in names, names
    if kind == "obj":
        assert "barrel.v2.mtl" in names, names


_BLENDER_CASES = [
    ("retopo", "retopo_bytes", "_fake_retopo_bytes", {"target_faces": 1000}),
    ("smart-unwrap", "unwrap_bytes", "_fake_unwrap_bytes", {}),
]


# ---------------------------------------------------------------------------
# clay-115: Snap to Grid works in world space
# ---------------------------------------------------------------------------


def _parent_and_child(parent_at, child_local, parent_scale=1.0):
    from realmspinner.kernels.mesh import document as bd
    from realmspinner.kernels.mesh import primitives as bp

    doc = bd.ClayDoc()
    parent = doc.add_object(
        bd.Obj(
            uid=bd.new_uid(),
            name="P",
            mesh=bp.box(),
            translation=np.array(parent_at, "f8"),
            scale=np.array([parent_scale] * 3, "f8"),
        )
    )
    child = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="C", mesh=bp.box(), translation=np.array(child_local, "f8"))
    )
    doc.set_parent(child.uid, parent.uid, keep_world=False)
    return doc, parent.uid, child.uid


class _Toasts:
    def __init__(self):
        self.messages = []

    def toast(self, message, level="info"):
        self.messages.append(message)


def test_snap_to_grid_snaps_a_parented_objects_world_position():
    from realmspinner.studio.modes.clay import ops as clay_ops

    doc, parent, child = _parent_and_child((0.3, 0.0, 0.2), (1.0, 0.0, 0.0))
    doc.select([child])
    assert clay_ops.run(_Toasts(), doc, clay_ops.get("snap-to-grid"), step=1.0)
    world = np.asarray(doc.world_matrix(child))[:3, 3]
    assert np.allclose(world, (1.0, 0.0, 0.0), atol=1e-6), world


def test_snap_to_grid_snaps_parent_and_child_together_whatever_the_selection_order():
    from realmspinner.studio.modes.clay import ops as clay_ops

    # A half-scale parent: its snapped origin sits on the grid but one local
    # metre of the child is half a world metre, so a local snap cannot land.
    doc, parent, child = _parent_and_child((0.3, 0.0, 0.2), (0.8, 0.0, 0.0), parent_scale=0.5)
    doc.select([child, parent])
    assert clay_ops.run(_Toasts(), doc, clay_ops.get("snap-to-grid"), step=1.0)
    assert np.allclose(np.asarray(doc.world_matrix(parent))[:3, 3], (0.0, 0.0, 0.0), atol=1e-6)
    assert np.allclose(np.asarray(doc.world_matrix(child))[:3, 3], (0.0, 0.0, 0.0), atol=1e-6)


def test_snap_to_grid_leaves_a_roots_translation_bit_identical_to_before():
    from realmspinner.kernels.mesh import ops as geom
    from realmspinner.studio.modes.clay import ops as clay_ops

    doc, parent, _child = _parent_and_child((0.37, 0.52, -0.81), (0.0, 0.0, 0.0))
    doc.select([parent])
    expected = geom.snap_translation((0.37, 0.52, -0.81), 0.25)
    clay_ops.run(_Toasts(), doc, clay_ops.get("snap-to-grid"), step=0.25)
    assert np.array_equal(np.asarray(doc.by_uid(parent).translation, dtype="f8"), expected)


# ---------------------------------------------------------------------------
# clay-116: docstrings that name files and behaviour that moved
# ---------------------------------------------------------------------------


def test_clay_ops_docstrings_name_files_that_exist():
    import re
    from pathlib import Path

    from realmspinner.studio.modes.clay import mode as clay_mode
    from realmspinner.studio.modes.clay import ops as clay_ops

    root = Path(clay_ops.__file__).resolve().parents[5]
    missing = []
    for module in (clay_ops, clay_mode):
        source = Path(module.__file__).read_text(encoding="utf-8")
        for rel in set(re.findall(r"studio/modes/clay/[\w/]+\.py", source)):
            if not (root / "src" / "realmspinner" / rel).is_file():
                missing.append((module.__name__, rel))
        for rel in set(re.findall(r"docs/manual/\d+-[\w-]+\.md", source)):
            if not (root / rel).is_file():
                missing.append((module.__name__, rel))
    assert not missing, missing


# ---------------------------------------------------------------------------
# clay-82: the clay_uv catalogue tells an agent the type the handler accepts
# ---------------------------------------------------------------------------


def test_the_uv_action_catalogue_describes_rotate_as_the_boolean_the_handler_accepts():
    from realmspinner.studio.modes.clay.agent import schema

    text = schema.UV_ACTIONS["pack"]
    assert "rotate (boolean 0/1" not in text, "the handler refuses 0 and 1 for rotate"
    assert "rotate (boolean true/false, default false)" in text
