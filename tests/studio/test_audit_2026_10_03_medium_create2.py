"""Regression tests for the 2026-10-03 audit's Medium findings on Create's viewer
and workspace: create-09 .. create-14, create-18 .. create-21.

No GL context: ``GpuModel`` is driven over a counting fake context, and the
frame-thread methods are driven unbound over stubs, the idiom this directory
already uses for ``Viewer``.
"""

from __future__ import annotations

import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from realmspinner import models
from realmspinner.kernels.geom3d import gltf
from realmspinner.kernels.geom3d.glbio import rebuild_glb
from realmspinner.service import loras as svc_loras
from realmspinner.studio.viewer import scene as scenelib
from realmspinner.studio.viewer_embed import Viewer


def _glb(doc: dict, binary: bytes = b"") -> bytes:
    binary += b"\x00" * (-len(binary) % 4)
    chunk = struct.pack("<II", len(binary), 0x004E4942) + binary
    header = struct.pack("<III", 0x46546C67, 2, 0)
    return rebuild_glb(header, doc, chunk)


def _skinned_glb(
    *,
    node_extra: dict | None = None,
    joint_values: tuple[int, int, int, int] = (0, 0, 0, 0),
    joint_dtype: str = "<u1",
    skin_joints: tuple[int, ...] = (1, 2),
    primitives: int = 1,
    material: dict | None = None,
    skinned_node: bool = True,
) -> bytes:
    """One triangle (``primitives`` times over) bound to a two-joint skin."""
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="<f4")
    joints = np.tile(np.array(joint_values, dtype=joint_dtype), (3, 1))
    weights = np.tile(np.array([1, 0, 0, 0], dtype="<f4"), (3, 1))
    indices = np.array([0, 1, 2], dtype="<u4")
    blobs = [positions.tobytes(), joints.tobytes(), weights.tobytes(), indices.tobytes()]
    views, binary = [], b""
    for blob in blobs:
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(blob)})
        binary += blob + b"\x00" * (-len(blob) % 4)
    component = {"<u1": 5121, "<i1": 5120}[joint_dtype]
    node = {"name": "mesh_node", "mesh": 0}
    if skinned_node:
        node["skin"] = 0
    node.update(node_extra or {})
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0, 1]}],
        "nodes": [node, {"name": "root", "children": [2]}, {"name": "tip"}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2},
                        "indices": 3,
                        **({"material": 0} if material is not None else {}),
                    }
                ]
                * primitives
            }
        ],
        "skins": [{"joints": list(skin_joints)}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": views,
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": component, "count": 3, "type": "VEC4"},
            {"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC4"},
            {"bufferView": 3, "componentType": 5125, "count": 3, "type": "SCALAR"},
        ],
    }
    if material is not None:
        doc["materials"] = [material]
    return _glb(doc, binary)


class _Buf:
    def __init__(self, log: list) -> None:
        self.released = False
        log.append(self)

    def release(self) -> None:
        self.released = True


class _CountingCtx:
    """Hands out buffers and optionally raises on the Nth ``buffer`` call."""

    def __init__(self, fail_on: int | None = None) -> None:
        self.made: list[_Buf] = []
        self.calls = 0
        self.fail_on = fail_on

    def buffer(self, _data):
        self.calls += 1
        if self.calls == self.fail_on:
            raise RuntimeError("driver said no")
        return _Buf(self.made)

    def vertex_array(self, *a, **k):  # pragma: no cover - never drawn here
        raise AssertionError("no draw in this test")


# --- create-09 ---------------------------------------------------------------


def test_a_skinned_node_with_a_zero_scale_still_builds_its_gpu_model():
    model = gltf.load(_skinned_glb(node_extra={"scale": [0.0, 0.0, 0.0]}))
    ctx = _CountingCtx()

    gpu = scenelib.GpuModel(ctx, model)

    node = model.nodes[0]
    palette = gpu.palette(node)
    assert palette is not None and len(palette) > 0
    # The skin space is ignored for a degenerate node, so the palette is the
    # joint world matrices alone: the identity bind pose here.
    mats = model.joint_palette(node)
    assert mats is not None
    for matrix in mats:
        assert np.allclose(matrix, np.eye(4))


# --- create-10 ---------------------------------------------------------------


@pytest.mark.parametrize("fail_on", [2, 3, 4])
def test_a_gpu_model_that_fails_midway_releases_the_buffers_it_made(fail_on):
    # Two primitives, two buffers each: failing on the ibo of the first, the
    # vbo of the second, or the ibo of the second.
    model = gltf.load(_skinned_glb(primitives=2))
    ctx = _CountingCtx(fail_on=fail_on)

    with pytest.raises(RuntimeError, match="driver said no"):
        scenelib.GpuModel(ctx, model)

    assert len(ctx.made) == fail_on - 1
    assert all(buf.released for buf in ctx.made), (
        "a constructor that raised midway abandoned live GL buffers"
    )


def test_a_gpu_model_whose_palette_build_raises_releases_every_buffer(monkeypatch):
    model = gltf.load(_skinned_glb(primitives=2))
    ctx = _CountingCtx()

    def boom(self):
        raise RuntimeError("palette failed")

    monkeypatch.setattr(scenelib.GpuModel, "refresh_palettes", boom)
    with pytest.raises(RuntimeError, match="palette failed"):
        scenelib.GpuModel(ctx, model)

    assert len(ctx.made) == 4
    assert all(buf.released for buf in ctx.made)


def test_a_material_whose_texture_setup_raises_releases_the_texture_it_made():
    class Tex:
        released = False
        anisotropy = 1.0
        filter = None

        def build_mipmaps(self):
            raise RuntimeError("mipmaps failed")

        def release(self):
            self.released = True

    made = []

    class Ctx:
        max_anisotropy = 1.0

        def texture(self, *a, **k):
            made.append(Tex())
            return made[-1]

    material = gltf.Material()
    material.base_color = (1, 1, bytes([255]) * 4)

    with pytest.raises(RuntimeError, match="mipmaps failed"):
        scenelib.GpuMaterial(Ctx(), material, {})

    assert len(made) == 1 and made[0].released


# --- create-11 ---------------------------------------------------------------


def test_a_mouse_move_after_pose_mode_ends_mid_drag_does_not_raise():
    viewer = Viewer.__new__(Viewer)
    viewer.pose_mode = True
    viewer.pose_job_id = "a" * 12
    viewer.rotate_gizmo = SimpleNamespace(end_drag=lambda: None)
    viewer.translate_gizmo = SimpleNamespace(end_drag=lambda: None)
    viewer.editor = SimpleNamespace(
        clear=lambda: None, has_unsaved_edits=lambda: False, selected=None
    )
    viewer.on_pose_dirty = None
    viewer._pose_step = None
    viewer._bone_pairs = []
    viewer._grab = "gizmo"  # a gizmo drag is live
    viewer._last_mouse = (0.0, 0.0)
    viewer._rect = (0, 0, 100, 100)
    viewer._deselect_on_click = False
    viewer._render_dirty = False
    viewer._ray = lambda local: (None, None)

    Viewer.exit_pose_mode(viewer)  # a model was adopted mid-drag
    # The next mouse motion arrives before the button comes up.
    Viewer._motion(viewer, (3.0, 4.0))

    assert viewer._grab is None


def test_a_gizmo_grab_with_no_active_gizmo_is_dropped_by_the_motion_itself():
    viewer = Viewer.__new__(Viewer)
    viewer.pose_mode = False
    viewer.translate_gizmo = object()
    viewer._grab = "gizmo"
    viewer._last_mouse = (0.0, 0.0)
    viewer._rect = (0, 0, 100, 100)
    viewer._deselect_on_click = False
    viewer._render_dirty = False
    viewer._ray = lambda local: (None, None)

    Viewer._motion(viewer, (3.0, 4.0))

    assert viewer._grab is None


# --- create-12 ---------------------------------------------------------------


def _skins_doc(count: int) -> bytes:
    return _glb(
        {
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": []}],
            "skins": [{"joints": []}] * count,
        }
    )


def test_a_glb_declaring_more_skins_than_the_ceiling_is_refused_before_any_is_built(
    monkeypatch,
):
    built = []
    real = gltf._Reader.skin

    def spy(self, skin):
        built.append(skin)
        return real(self, skin)

    monkeypatch.setattr(gltf._Reader, "skin", spy)
    ceiling = getattr(gltf, "MAX_SKINS", 100_000)

    with pytest.raises(ValueError, match="skins"):
        gltf.load(_skins_doc(ceiling + 1))

    assert built == [], "a skin was decoded before the declared count was checked"


def test_a_skins_field_that_is_not_a_list_is_refused_by_name():
    doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": []}], "skins": 5}
    with pytest.raises(ValueError, match="skins"):
        gltf.load(_glb(doc))


# --- create-13 ---------------------------------------------------------------


@pytest.mark.parametrize("bad", [["BLEND"], {"mode": "MASK"}, "bogus", 3, None])
def test_a_non_string_alpha_mode_falls_back_to_opaque(bad):
    model = gltf.load(_skinned_glb(material={"alphaMode": bad}))
    mat = model.meshes[0][0].material

    assert mat.alpha_mode == "OPAQUE"
    # And the frame path that read it no longer trips on an unhashable value.
    uniforms = {
        "u_alpha_cutoff": SimpleNamespace(value=None),
        "u_alpha_mask": SimpleNamespace(value=None),
    }
    scenelib.GpuMaterial(SimpleNamespace(), mat, {}).bind(uniforms)
    assert uniforms["u_alpha_mask"].value == 0


def test_the_three_gltf_alpha_modes_survive():
    for mode in ("OPAQUE", "MASK", "BLEND"):
        model = gltf.load(_skinned_glb(material={"alphaMode": mode}))
        assert model.meshes[0][0].material.alpha_mode == mode


# --- create-20 ---------------------------------------------------------------


def test_a_joint_index_past_the_skins_joint_count_is_refused_at_load():
    with pytest.raises(ValueError, match="JOINTS_0"):
        gltf.load(_skinned_glb(joint_values=(200, 0, 0, 0)))
    # Between the joint count and MAX_JOINTS: it used to skin to an identity.
    with pytest.raises(ValueError, match="JOINTS_0"):
        gltf.load(_skinned_glb(joint_values=(2, 0, 0, 0)))


def test_a_negative_joint_index_is_refused_at_load():
    with pytest.raises(ValueError, match="JOINTS_0"):
        gltf.load(_skinned_glb(joint_values=(0, 0, 0, -1), joint_dtype="<i1"))


def test_in_range_joint_indices_and_an_unskinned_use_of_the_mesh_still_load():
    gltf.load(_skinned_glb(joint_values=(1, 0, 1, 0)))
    # The same stray value on a node that never skins it is never indexed.
    gltf.load(_skinned_glb(joint_values=(200, 0, 0, 0), skinned_node=False))


# --- create-14 ---------------------------------------------------------------

_ARCHIVE = (
    len(b'{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}').to_bytes(8, "little")
    + b'{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
    + b"\x00" * 4
)


def test_reimporting_a_lora_with_a_new_family_updates_the_registry_entry(svc, tmp_path):
    before = dict(models.STYLE_LORAS)
    try:
        adapter = tmp_path / "style.safetensors"
        adapter.write_bytes(_ARCHIVE)
        first = svc_loras.import_lora(
            svc, adapter, label="Cosmos", family=models.FAMILY_SDXL
        )
        key = first["key"]
        assert models.STYLE_LORAS[key].family == models.FAMILY_SDXL

        second = svc_loras.import_lora(
            svc, adapter, label="Cosmos", family=models.FAMILY_FLUX2_KLEIN
        )

        assert second["key"] == key
        assert second["family"] == models.FAMILY_FLUX2_KLEIN
        assert models.STYLE_LORAS[key].family == models.FAMILY_FLUX2_KLEIN, (
            "the registry kept the old family after the manifest was rewritten"
        )
    finally:
        models.STYLE_LORAS.clear()
        models.STYLE_LORAS.update(before)


# --- create-18 ---------------------------------------------------------------


class _FailingViewer:
    """Adoption always raises; ``clear`` empties the viewer the way the real
    one does."""

    def __init__(self) -> None:
        self.pending: Path | None = None
        self.path: Path | None = None
        self.pose_mode = False

    def adopt_model(self, model, path):
        raise RuntimeError("could not build the GPU model")

    def clear(self):
        self.path = None
        self.pending = None

    def cancel_sheet_strip(self):
        pass

    def parse_model(self, path):  # pragma: no cover
        raise AssertionError("a failed asset must not be parsed again")

    parse_reference = parse_model


def test_a_failed_adopt_is_not_retried_by_the_next_sync(tmp_path):
    from realmspinner.studio import main as main_mod

    job_dir = tmp_path / "abc"
    job = {"id": "abc", "files": ["model.glb"]}
    submitted = []
    ctx = SimpleNamespace(
        state=SimpleNamespace(
            mode="create",
            create=SimpleNamespace(stage="mesh"),
            preview={},
            comparing=None,
        ),
        job=lambda: job,
        job_dir=lambda _id: job_dir,
        toast=lambda *a, **k: None,
        submit=lambda *a, **k: submitted.append(a) or True,
        capture_thumbnail=lambda *a, **k: None,
    )
    app = main_mod.App.__new__(main_mod.App)
    app.viewer = _FailingViewer()
    app.app_ctx = ctx
    app._refresh_rig_side_data = lambda: None
    wanted = job_dir / "model.glb"
    app.viewer.pending = wanted

    app._adopt_model(SimpleNamespace(tag=wanted, result=object()))
    app._sync_viewer()

    assert app.viewer.path == wanted
    assert submitted == [], "the next sync re-parsed an asset whose adoption had failed"


# --- create-19 ---------------------------------------------------------------


def test_the_create_viewport_wireframe_toggle_draws_over_the_surface_as_the_manual_says():
    draws: list[dict] = []
    viewer = SimpleNamespace(
        _rect=None,
        wireframe=True,
        comparing=False,
        pose_mode=False,
        _frame_unchanged=lambda key: False,
        _resize=lambda *a, **k: None,
        camera=SimpleNamespace(update=lambda dt: None),
        renderer=SimpleNamespace(draw=lambda *a, **k: draws.append(k)),
        viewport=SimpleNamespace(texture="tex"),
        gpu=object(),
        placement=None,
        _overlays=lambda height: [],
    )

    Viewer.render(viewer, (0, 0, 64, 64), 0.0)

    assert len(draws) == 1
    # ``wireframe`` replaces the fill; the manual's "draw the triangles over
    # the shaded surface" is ``wire_overlay``.
    assert draws[0].get("wire_overlay") is True
    assert not draws[0].get("wireframe")


# --- create-21 ---------------------------------------------------------------


def test_vary_on_a_mesh_resets_form_keys_the_job_never_recorded(monkeypatch):
    from realmspinner.studio.modes.create.ui import stages as create_stages
    from realmspinner.studio.modes.create.ui import workspace
    from realmspinner.studio.state import DEFAULT_FORM_3D

    monkeypatch.setattr(create_stages, "go", lambda *a, **k: None)
    # The live form still holds the last session's hand-edited engine values.
    form = dict(DEFAULT_FORM_3D)
    form.update(
        {"trellis_band": 6, "size_m": 2.5, "count": 3, "rig": True, "lowpoly_triangles": 999}
    )
    ctx = SimpleNamespace(
        state=SimpleNamespace(form_3d=form, source_job=None),
        toast=lambda *a, **k: None,
    )
    job = {
        "id": "m1",
        "stage": "model",
        "parent_id": "r1",
        "params": {"lowpoly_triangles": 5000, "mesh_seed": 7},
    }

    workspace._vary(ctx, job)

    out = ctx.state.form_3d
    assert out["mesh_seed"] == 7
    assert out["lowpoly_triangles"] == 5000
    # Never recorded by the job, so the engine default -- not the live value.
    assert out["trellis_band"] == DEFAULT_FORM_3D["trellis_band"]
    assert out["size_m"] == DEFAULT_FORM_3D["size_m"]
    # How many and whether to rig are not part of a recipe: left as the user had them.
    assert out["count"] == 3 and out["rig"] is True
    assert ctx.state.source_job == "r1"
