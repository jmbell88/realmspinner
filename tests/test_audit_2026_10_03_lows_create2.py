"""The 2026-10-03 audit's Low findings create-25..53 (fixer ``create2``).

Viewer and pose hot paths, the request document's optional ints, the engine's
provenance and Create's draft filing. Each test's name is the claim.
"""

from __future__ import annotations

import hashlib
import json
import struct
import tracemalloc
from types import SimpleNamespace

import numpy as np
import pytest

from realmspinner import generation, provenance
from realmspinner.kernels.geom3d import gltf
from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.service.errors import Invalid
from realmspinner.studio.viewer import picking, pose
from realmspinner.studio.viewer.camera import Camera

# --- create-26: ghost_handles queues a repeated child once -------------------


def test_ghost_handles_visits_a_repeated_child_reference_once():
    """``update_world`` got a ``queued`` set (clay-io-11); ``ghost_handles``,
    which claims to make the same walk, pushed every repetition on its stack --
    on each ghost, each frame. Measured by the stack's peak allocation: a million
    duplicate references were ~64 MB of tuples before, a handful of bytes now."""
    children = [1] * 1_000_000  # built before tracing: the file's own cost
    nodes = [gltf.Node(name="root", children=children), gltf.Node(name="bone")]
    model = gltf.Model(nodes, [0], [], [])
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        out = pose.ghost_handles(model, ["bone"], {})
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert set(out) == {"bone"}
    assert peak < 16 * 1024 * 1024, f"the walk's stack peaked at {peak} bytes"


# --- create-27: _skin_bones is linear ----------------------------------------


class _CountingStr(str):
    compares = 0

    def __eq__(self, other):
        type(self).compares += 1
        return str.__eq__(self, other)

    __hash__ = str.__hash__


def test_skin_bones_scales_linearly_with_the_joint_count():
    """``name not in names`` against a list was quadratic, and the loader's own
    ceiling admits 100,000 joints -- about half a minute on entering the pose
    editor. Counted in string comparisons rather than seconds so it cannot flake."""
    count = 1500
    nodes = [gltf.Node(name=_CountingStr(f"joint_{i}")) for i in range(count)]
    skin = gltf.Skin(joints=list(range(count)), inverse_bind=np.zeros((count, 4, 4)))
    model = gltf.Model(nodes, [0], [], [skin])
    _CountingStr.compares = 0
    names = pose._skin_bones(model)
    assert names == [f"joint_{i}" for i in range(count)]
    assert _CountingStr.compares < 10 * count, _CountingStr.compares


# --- create-28: stats() does not recompute the bounds each frame -------------


def _box_model() -> gltf.Model:
    positions = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="f4"
    )
    prim = gltf.Primitive(positions=positions, indices=np.array([0, 1, 2], dtype="u4"))
    return gltf.Model([gltf.Node(mesh=0)], [0], [[prim]], [])


def test_stats_does_not_recompute_bounds_when_nothing_changed(monkeypatch):
    """``Viewer.stats`` runs every frame for the inspector's size line and
    called ``Model.bounds`` -- a walk of every mesh instance -- each time."""
    from realmspinner.studio.viewer_embed import Viewer

    model = _box_model()
    calls = []
    real = gltf.Model.bounds

    def counted(self):
        calls.append(1)
        return real(self)

    monkeypatch.setattr(gltf.Model, "bounds", counted)
    viewer = SimpleNamespace(model=model, placement=m3.identity(), _bounds_memo=None)
    first = Viewer.stats(viewer)
    second = Viewer.stats(viewer)
    assert first == second
    assert len(calls) == 1
    # A pose change goes through ``update_world`` and must invalidate the memo.
    model.nodes[0].translation = m3.vec3(5, 0, 0)
    model.update_world()
    moved = Viewer.stats(viewer)
    assert len(calls) == 2
    assert moved["size"] == pytest.approx(first["size"])


# --- create-29: a zero-vertex primitive ---------------------------------------


def _glb(document: dict, binary: bytes) -> bytes:
    payload = json.dumps(document).encode()
    payload += b" " * (-len(payload) % 4)
    binary += b"\0" * (-len(binary) % 4)
    body = struct.pack("<II", len(payload), 0x4E4F534A) + payload
    body += struct.pack("<II", len(binary), 0x004E4942) + binary
    return struct.pack("<III", 0x46546C67, 2, 12 + len(body)) + body


def test_a_zero_vertex_primitive_is_refused_or_skipped_at_load_not_at_upload():
    """The loader said yes to a primitive whose POSITION accessor has no rows and
    ``GpuPrimitive._interleave`` then raised "cannot reshape array of size 0" at
    upload. It is skipped at load, as ``write_glb`` skips one."""
    triangle = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="f4").tobytes()
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(triangle)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(triangle)}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 0, "type": "VEC3"},
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
        ],
        "meshes": [
            {"primitives": [{"attributes": {"POSITION": 1}}, {"attributes": {"POSITION": 0}}]}
        ],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
    }
    model = gltf.load(_glb(document, triangle))
    counts = [len(p.positions) for prims in model.meshes for p in prims]
    assert counts == [3], "the real triangle stays and the empty primitive is gone"


# --- create-30: from_dict leaves a garbage optional int for validate ----------


def _issue_fields(request) -> set[str]:
    return {issue.field for issue in generation.validate_request(request)}


def test_from_dict_leaves_a_garbage_optional_int_for_validate_request_to_refuse():
    """``_optional_int`` folded "banana" to ``None`` ("nobody said"), so a
    request document with a garbage cell target was recorded as valid and the
    "Cell target must be a whole number." branch was unreachable from it."""
    tile = generation.GenerationRequest.from_dict(
        {"generation_type": "tileset", "prompt": "ruins", "tile": {"target_cell_px": "banana"}}
    )
    assert tile.tile.target_cell_px == "banana"
    issues = generation.validate_request(tile)
    assert any(
        i.field == "target_cell_px" and "whole number" in i.message for i in issues
    )

    sprite = generation.GenerationRequest.from_dict(
        {
            "generation_type": "sprite_sheet",
            "prompt": "a knight",
            "sprite": {"candidate_count": "banana", "frame_count": "x", "target_cell_px": "y"},
        }
    )
    fields = _issue_fields(sprite)  # must refuse, not raise TypeError on 1 <= "banana"
    assert {"sprite.candidate_count", "sprite.frame_count", "target_cell_px"} <= fields

    model = generation.GenerationRequest.from_dict({"model": {"custom_triangles": "banana"}})
    assert "model.custom_triangles" in _issue_fields(model)

    # What converts still converts and "nobody said" is still None.
    ok = generation.GenerationRequest.from_dict(
        {"tile": {"target_cell_px": "32"}, "sprite": {"candidate_count": ""}}
    )
    assert ok.tile.target_cell_px == 32 and ok.sprite.candidate_count is None


# --- create-31: load_lora_manifests is total ----------------------------------


@pytest.mark.parametrize("value", [None, 5, True, "text", {"a": 1}])
def test_load_lora_manifests_returns_nothing_for_a_non_list_manifests_value(tmp_path, value):
    """The row loop sat outside the ``try``, so ``"manifests": null`` raised out
    of every ``resolve_recipe`` and took Create's recipe column down."""
    path = tmp_path / "loras" / "manifests.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"manifests": value}), encoding="utf-8")
    config = SimpleNamespace(t2i_model_root=tmp_path)
    generation._MANIFEST_CACHE.pop(path, None)
    assert generation.load_lora_manifests(config) == []
    generation._MANIFEST_CACHE.pop(path, None)


# --- create-39: the sprite panel's seed captions ------------------------------


def test_the_sprite_panel_seed_fields_are_not_captioned_with_identifiers(monkeypatch):
    """Two user-facing controls were captioned ``seed_a`` and ``seed_b``."""
    from realmspinner.studio.panes import sprite_panel

    class FakeForm:
        def __init__(self):
            self.numbers = []

        def combo(self, field, label, value, items):
            return False, value

        def number(self, field, label, value, **kwargs):
            self.numbers.append((field, label))
            return False, value

    monkeypatch.setattr(sprite_panel.imgui, "push_id", lambda *_: None)
    monkeypatch.setattr(sprite_panel.imgui, "pop_id", lambda *_: None)
    monkeypatch.setattr(sprite_panel.controls, "small_button", lambda *a, **k: False)
    monkeypatch.setattr(sprite_panel, "_options", lambda: {})
    form_ui = FakeForm()
    form = {"sheet_type": "turnaround", "logical_size": 64, "colors": 32, "seed_a": 1, "seed_b": 2}
    sprite_panel._controls(None, form, form_ui)
    assert [field for field, _ in form_ui.numbers] == ["seed_a", "seed_b"]
    for field, label in form_ui.numbers:
        assert label != field and "_" not in label, (field, label)


# --- create-41: an orthographic gizmo's size does not depend on depth ---------


def test_an_orthographic_gizmo_has_the_same_screen_size_at_every_depth():
    """``screen_scale`` used the perspective formula unconditionally, so in an
    orthographic Poser view the gizmo grew and shrank from joint to joint."""
    camera = Camera()
    camera.orthographic = True
    toward_eye = camera.position - camera.target
    toward_eye = toward_eye / np.linalg.norm(toward_eye)
    near = picking.screen_scale(camera, camera.target + toward_eye * 0.5, 90, 900)
    at_target = picking.screen_scale(camera, camera.target, 90, 900)
    far = picking.screen_scale(camera, camera.target - toward_eye * 7.0, 90, 900)
    assert near == pytest.approx(at_target)
    assert far == pytest.approx(at_target)
    # And it is the size the orthographic frustum actually draws 90 px at.
    height = 2.0 * camera.distance * np.tan(np.radians(camera.fov * 0.5))
    assert at_target == pytest.approx(height * 90 / 900)


# --- create-42: a translucent material in the direction strip -----------------


def test_a_translucent_material_in_the_strip_keeps_its_colour_and_alpha(gl, tmp_path, monkeypatch):
    """A BLEND fragment of alpha a against a transparent clear stored colour c*a
    but alpha a*a, so ``unpremultiply`` returned c/a and the cell's alpha was a
    squared: translucent materials looked too bright and too transparent."""
    import trimesh

    from realmspinner.studio.viewer import scene as scenelib
    from realmspinner.studio.viewer import sheet as sheetlib
    from realmspinner.studio.viewer.render import Renderer

    path = tmp_path / "box.glb"
    trimesh.creation.box(extents=(1.0, 2.0, 1.0)).export(path)

    def centre(alpha: float):
        model = gltf.load(path)
        for prims in model.meshes:
            for prim in prims:
                prim.material = gltf.Material(
                    base_color_factor=(0.8, 0.4, 0.2, alpha),
                    alpha_mode="BLEND" if alpha < 1.0 else "OPAQUE",
                    double_sided=False,
                )
        gpu = scenelib.GpuModel(gl, model)
        renderer = Renderer(gl)
        try:
            img = sheetlib.strip(
                renderer, gpu, model, [0.0], model_matrix=scenelib.placement(model), flat=True
            )
        finally:
            renderer.release()
            gpu.release()
        pixels = np.array(img)
        return pixels[pixels.shape[0] // 2, pixels.shape[1] // 2].astype(int)

    opaque = centre(1.0)
    translucent = centre(0.5)
    assert opaque[3] == 255
    assert abs(translucent[3] - 128) <= 6, translucent
    assert np.abs(translucent[:3] - opaque[:3]).max() <= 8, (translucent, opaque)


# --- create-47: a request the door cannot apply is refused --------------------


def test_create_generation_request_refuses_a_model_profile_it_cannot_apply(svc):
    """``output_profile`` and ``custom_triangles`` were accepted and recorded in
    ``params["generation_request"]`` but never forwarded to ``create_job``, so
    the stored request claimed a mesh budget the job never used."""
    from realmspinner.service import jobs as svc_jobs

    for model, field in (
        ({"output_profile": "standard"}, "model.output_profile"),
        ({"output_profile": "banana"}, "model.output_profile"),
        ({"custom_triangles": 5000}, "model.custom_triangles"),
    ):
        with pytest.raises(Invalid) as excinfo:
            svc_jobs.create_generation_request(svc, {"prompt": "a barrel", "model": model})
        assert excinfo.value.field == field, model
    # The default ("nobody said") still queues.
    made = svc_jobs.create_generation_request(svc, {"prompt": "a barrel"})
    assert made["id"]


# --- create-49: the engine identity covers the whole pinned runtime -----------


def test_engine_identity_is_unverified_when_a_pinned_dll_differs(tmp_path, monkeypatch):
    """Only ``trellis-server.exe`` was compared, yet ``version_verified`` vouched
    for the whole runtime although the DLLs that decide the numerics are pinned
    beside it."""
    from realmspinner import models
    from realmspinner.config import Config

    exe = tmp_path / "trellis-server.exe"
    dll = tmp_path / "ggml-cuda.dll"
    exe.write_bytes(b"engine executable")
    dll.write_bytes(b"cuda numerics")
    pins = (
        ("ggml-cuda.dll", hashlib.sha256(dll.read_bytes()).hexdigest()),
        ("trellis-server.exe", hashlib.sha256(exe.read_bytes()).hexdigest()),
    )
    monkeypatch.setattr(models, "TRELLIS_RUNTIME_DIGESTS", pins)
    config = Config(trellis_server_exe=exe)
    good = provenance.native_engine_identity(config)
    assert good["version_verified"] is True
    assert good["engine_version"] == models.TRELLIS_RUNTIME_VERSION

    dll.write_bytes(b"a swapped, longer cuda library")
    swapped = provenance.native_engine_identity(config)
    assert swapped["version_verified"] is False
    assert swapped["engine_version"] is None
    assert swapped["engine_sha256"] == pins[1][1], "the exe's own digest is still reported"

    dll.unlink()
    assert provenance.native_engine_identity(config)["version_verified"] is False


# --- create-50: filing a draft copies nothing when nothing changed ------------


def test_save_draft_does_not_copy_an_unchanged_form(monkeypatch):
    """``session.sync`` files the draft every frame; each call deep-copied both
    forms whether or not anything moved."""
    from realmspinner.studio.modes.create.ui import session

    copies = []
    real = session.copy.deepcopy

    def counted(value, *args, **kwargs):
        copies.append(1)
        return real(value, *args, **kwargs)

    monkeypatch.setattr(session.copy, "deepcopy", counted)
    state = SimpleNamespace(
        form_2d={"prompt": "a barrel", "seed": 3},
        form_3d={"profile": "raw"},
        source_job="job-1",
        selected="job-2",
        create=SimpleNamespace(workspace="creation:abc", drafts={}),
    )
    ctx = SimpleNamespace(state=state)
    session.save_draft(ctx)
    assert len(copies) == 2
    session.save_draft(ctx)
    session.save_draft(ctx)
    assert len(copies) == 2, "an unchanged form is not copied again"
    state.form_2d["prompt"] = "a crate"
    session.save_draft(ctx)
    assert len(copies) == 4
    assert state.create.drafts["creation:abc"]["form_2d"]["prompt"] == "a crate"
    state.selected = "job-3"
    session.save_draft(ctx)
    assert state.create.drafts["creation:abc"]["selected"] == "job-3"


# --- create-51: the Pixelate sizes ---------------------------------------------


def test_pixel_sizes_offers_only_exact_divisors_no_larger_than_the_cell():
    from realmspinner.service import sheets as svc_sheets
    from realmspinner.studio.panes import sheet_panel

    assert sheet_panel._pixel_sizes(128) == [16, 32, 64, 128]
    assert sheet_panel._pixel_sizes(100) == []
    assert sheet_panel._pixel_sizes(0) == []
    for frame in (16, 48, 96, 128, 200):
        for size in sheet_panel._pixel_sizes(frame):
            assert size in svc_sheets.PIXEL_LOGICAL_SIZES
            assert frame % size == 0 and size <= frame


# --- create-53: mesh_finishing at the request-document level ------------------


def test_validate_request_refuses_an_unknown_mesh_finishing():
    request = generation.GenerationRequest.from_dict(
        {"prompt": "a barrel", "model": {"mesh_finishing": "polish"}}
    )
    issues = [i for i in generation.validate_request(request) if i.field == "model.mesh_finishing"]
    assert len(issues) == 1
    for ok in ("preserve_shape", "repair"):
        request = generation.GenerationRequest.from_dict(
            {"prompt": "a barrel", "model": {"mesh_finishing": ok}}
        )
        assert "model.mesh_finishing" not in _issue_fields(request)


def test_the_recorded_mesh_finishing_is_the_one_the_job_runs_with(svc):
    """The coercion and the pass-through: dropping either leaves every request
    recorded (and run) as ``preserve_shape``."""
    from realmspinner.service import jobs as svc_jobs

    made = svc_jobs.create_generation_request(
        svc, {"prompt": "a barrel", "model": {"mesh_finishing": "repair"}}
    )
    row = svc.store.get(made["id"])
    params = row["params"]
    assert params["generation_request"]["model"]["mesh_finishing"] == "repair"
    assert params["mesh_finishing"] == "repair"
    default = svc_jobs.create_generation_request(svc, {"prompt": "a crate"})
    assert svc.store.get(default["id"])["params"]["mesh_finishing"] == "preserve_shape"
