"""Tranche 4: the three Blender-backed Clay ops -- Retopologize, Smart
Unwrap and Bake Detail, wired onto ``pipelines.clay_blender`` in decimate's
own ``prepare``/``work``/``apply`` background-op shape
(``studio/modes/clay/ops.py``'s "retopo / smart-unwrap / bake-detail" section).

Mirrors ``test_clay_tranche1_decimate.py``'s own shape: hand-built ``Ctx``
doubles drive the inline (agent) and interactive (submit-now, apply-later)
dispatch paths directly, with ``pipelines.clay_blender``'s own three
functions monkeypatched so nothing here needs a real Blender -- that
coverage belongs to ``tests/pipelines/test_clay_blender.py`` alone, per this
tranche's own brief.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from realmspinner.kernels.geom3d import glbwrite
from realmspinner.kernels.geom3d import gltf as gltf_mod
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.pipelines import blender_run, clay_blender
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay import ops as clay_ops
from realmspinner.studio.modes.clay import state as clay_state
from realmspinner.studio.modes.clay.agent import dispatch as agent_clay
from realmspinner.studio.modes.clay.agent.tools_ops import _OpCtx

# --- Ctx doubles, decimate's own shapes -------------------------------------


class _Ctx:
    """A minimal ``Ctx``: toasts recorded, nothing submitted for real."""

    def __init__(self) -> None:
        self.toasted: list[tuple[str, str]] = []

    def toast(self, message: str, level: str = "info") -> None:
        self.toasted.append((message, level))


class _InlineCtx(_Ctx):
    def __init__(self) -> None:
        super().__init__()
        self.inline = True


class _InteractiveCtx(_Ctx):
    """Records what would have been submitted, rather than running it -- so a
    test can drive the background half by hand, the way the real task runner
    would on a later frame."""

    def __init__(self, state: Any) -> None:
        super().__init__()
        self.state = SimpleNamespace(clay=state)
        self.submitted: list[tuple[str, Any, tuple, dict]] = []

    def submit(self, key: str, fn: Any, *args: Any, **kwargs: Any) -> bool:
        self.submitted.append((key, fn, args, kwargs))
        return True


class _Done:
    def __init__(self, key: str, result: Any) -> None:
        self.key = key
        self.result = result


def _tab_with(doc: bd.ClayDoc) -> tuple[clay_state.ClayState, clay_state.ClayTab]:
    state = clay_state.ClayState()
    tab = clay_state.ClayTab(doc=doc, title="Scene")
    state.add(tab)
    return state, tab


def _patch_available(monkeypatch: pytest.MonkeyPatch, ok: bool, reason: str = "") -> Any:
    monkeypatch.setattr(clay_blender, "available", lambda: (ok, reason))
    return clay_blender


# --- document fixtures -------------------------------------------------------


def _box_doc() -> tuple[bd.ClayDoc, int]:
    doc = bd.ClayDoc()
    obj = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box(), generator="box", params={"size": 1.0})
    )
    doc.select([obj.uid])
    return doc, obj.uid


def _two_object_doc() -> tuple[bd.ClayDoc, int, int]:
    """Box then Cone, both selected -- Box is added first, so it is the
    topmost object in document order and the target :func:`_bake_detail`
    (and ``_join``/``_union`` before it) reads that order for."""
    doc = bd.ClayDoc()
    box = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box(), generator="box", params={"size": 1.0})
    )
    cone = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="Cone", mesh=bp.cone(), generator="cone", params={})
    )
    doc.select([box.uid, cone.uid])
    return doc, box.uid, cone.uid


# --- fake pipeline functions --------------------------------------------------


def _fake_retopo_bytes(
    glb: bytes, *, target_faces: int, close_holes: bool = False, seed: int = 0,
    keep_uvs: bool = False, timeout: Any = None,
) -> tuple[bytes, dict]:
    del target_faces, close_holes, seed, keep_uvs, timeout
    model = gltf_mod.load(glb)
    objects = [
        {"name": n.name, "method": "quadriflow", "faces_before": 12, "faces": 12, "quads": 1.0}
        for n in model.nodes
        if n.mesh is not None
    ]
    return glb, {"ok": True, "objects": objects}


def _fake_unwrap_bytes(
    glb: bytes, *, angle_limit: float = 66.0, island_margin: float = 0.003, timeout: Any = None,
) -> tuple[bytes, dict]:
    del angle_limit, island_margin, timeout
    model = gltf_mod.load(glb)
    objects = [{"name": n.name, "islands": 1} for n in model.nodes if n.mesh is not None]
    return glb, {"ok": True, "objects": objects}


def _small_triangle_model(material: gltf_mod.Material) -> gltf_mod.Model:
    import numpy as np

    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="<f4")
    indices = np.array([0, 1, 2], dtype="<u4")
    prim = gltf_mod.Primitive(positions=positions, indices=indices, material=material)
    node = gltf_mod.Node(name="Low", mesh=0)
    return gltf_mod.Model([node], [0], [[prim]], [])


def _fake_bake_bytes(
    high_glb: bytes, low_glb: bytes, *, texture_size: int, cage_extrusion: float,
    maps: Any = clay_blender.blender_spec.CLAY_BAKE_MAPS, timeout: Any = None,
) -> tuple[bytes, dict]:
    del high_glb, low_glb, cage_extrusion, timeout
    maps = list(maps)
    material = gltf_mod.Material(
        name="wl_clay_baked",
        base_color=(2, 2, bytes(16)) if "base_color" in maps else None,
        normal=(2, 2, bytes(16)) if "normal" in maps else None,
        metallic_roughness=(2, 2, bytes(16)) if "roughness" in maps else None,
    )
    out = glbwrite.write_glb(_small_triangle_model(material))
    return out, {"ok": True, "maps": maps, "texture_size": texture_size, "metallic": 0.125}


# --- registration and gating --------------------------------------------------


def test_the_three_ops_are_registered_in_the_object_menu() -> None:
    names = {op.name for op in clay_ops.menu("object")}
    assert {"retopo", "smart-unwrap", "bake-detail"} <= names


def test_all_three_are_greyed_with_availables_own_sentence_when_blender_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reason = "Needs Blender, which is not installed (the rig extra)."
    _patch_available(monkeypatch, False, reason)

    # Two selected objects, so a missing selection can never explain the
    # greyed row this test is actually about -- only ``available()`` can.
    doc, _low, _high = _two_object_doc()
    for name in ("retopo", "smart-unwrap", "bake-detail"):
        op = clay_ops.get(name)
        assert op.enabled(doc) is False, name
        assert clay_ops.reason_for(op, doc) == reason, name


def test_the_derived_clay_op_enum_picks_up_all_three_new_ops() -> None:
    """No new agent tool -- all three arrive through ``clay_op``'s own
    ``name`` enum, built fresh from ``clay_ops.OPS`` on every ``tools()``
    call (``agent_clay.tools``'s own docstring)."""
    tools = agent_clay.tools()
    op_tool = next(t for t in tools if t.name == "clay_op")
    enum = op_tool.schema["properties"]["name"]["enum"]
    assert {"retopo", "smart-unwrap", "bake-detail"} <= set(enum)


# --- the timeout decision -----------------------------------------------------


def test_blender_timeout_reads_an_opctx_attribute_before_falling_back() -> None:
    ctx = _OpCtx(blender_timeout=42.0)
    assert clay_ops._blender_timeout(ctx) == 42.0


def test_blender_timeout_falls_back_to_clay_blenders_own_default_with_a_bare_ctx() -> None:
    assert clay_ops._blender_timeout(_Ctx()) == blender_run.BLENDER_TIMEOUT


def test_blender_timeout_reads_the_remesh_jobs_own_rig_timeout_when_reachable() -> None:
    ctx = SimpleNamespace(svc=SimpleNamespace(config=SimpleNamespace(rig_timeout=99.0)))
    assert clay_ops._blender_timeout(ctx) == 99.0


# --- the prepared GLB ----------------------------------------------------------


def test_the_prepared_glb_carries_each_objects_world_translation() -> None:
    doc, uid = _box_doc()
    doc.by_uid(uid).translation[:] = [1.0, 2.0, 3.0]
    glb, meta = clay_ops._blender_multi_prepare(doc, [uid])
    model = gltf_mod.load(glb)
    node = next(n for n in model.nodes if n.name == meta[0]["node_name"])
    assert node.translation == pytest.approx([1.0, 2.0, 3.0])


def test_a_multi_object_retopology_refuses_on_the_selections_total_before_triangulating_any_of_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """clay-40 (2026-09-19 audit, found during the fix phase, the ``ops-tail``
    reading debt): ``_blender_multi_prepare`` -- shared by retopo, unwrap and
    bake-detail -- ran ``_decimate_primitives`` (earclip triangulation plus an
    ``np.unique(key, axis=0)`` dedup) once per uid with no ceiling of any
    kind, so a selection of many legally-sized objects drove an unrefusable
    synchronous frame-thread stall. Two boxes (12 triangles each, 24 summed)
    trip a ceiling monkeypatched to 20 -- below either object alone -- so
    only the *selection's total* can be what trips it, the same shape
    ``ops_boolean._refuse_complexity``/``ops.MAX_JOINED_CORNERS`` already use
    for their own kernels.
    """
    from realmspinner.kernels.mesh.elements import OpError

    doc = bd.ClayDoc()
    a = doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box()))
    b = doc.add_object(bd.Obj(uid=bd.new_uid(), name="B", mesh=bp.box()))
    doc.select([a.uid, b.uid])

    monkeypatch.setattr(clay_ops, "MAX_PRIMITIVES_TRIANGLES", 20)
    with pytest.raises(OpError):
        clay_ops._blender_multi_prepare(doc, [a.uid, b.uid])

    # One box alone is 12 triangles -- under the ceiling -- proving this is
    # the selection's sum and not a per-object cap in different clothes.
    glb, meta = clay_ops._blender_multi_prepare(doc, [a.uid])
    assert meta


# --- retopo: inline round trip -------------------------------------------------


def test_retopo_inline_replaces_both_meshes_in_one_undo_step_and_freezes_the_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clay_blender_mod = _patch_available(monkeypatch, True)
    monkeypatch.setattr(clay_blender_mod, "retopo_bytes", _fake_retopo_bytes)

    doc, box_uid, cone_uid = _two_object_doc()
    box_mesh_before = doc.by_uid(box_uid).mesh
    cone_mesh_before = doc.by_uid(cone_uid).mesh
    depth = len(doc.history)

    ctx = _InlineCtx()
    assert clay_ops.run(ctx, doc, clay_ops.get("retopo"), target_faces=1000) is True

    assert len(doc.history) == depth + 1, "one undo step for the whole gesture"
    assert doc.history.top.label == "Retopologize"
    assert doc.by_uid(box_uid).mesh is not box_mesh_before
    assert doc.by_uid(cone_uid).mesh is not cone_mesh_before
    assert doc.by_uid(box_uid).generator is None, "retopo replaces the base mesh outright"
    assert doc.by_uid(cone_uid).generator is None

    assert doc.undo() is True
    assert doc.by_uid(box_uid).mesh is box_mesh_before
    assert doc.by_uid(cone_uid).mesh is cone_mesh_before
    assert doc.by_uid(box_uid).generator == "box", "the freeze undoes with the mesh, one step"


# --- smart-unwrap: inline round trip, generator kept ---------------------------


def test_smart_unwrap_inline_keeps_the_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    clay_blender_mod = _patch_available(monkeypatch, True)
    monkeypatch.setattr(clay_blender_mod, "unwrap_bytes", _fake_unwrap_bytes)

    doc, uid = _box_doc()
    mesh_before = doc.by_uid(uid).mesh
    depth = len(doc.history)

    ctx = _InlineCtx()
    assert clay_ops.run(ctx, doc, clay_ops.get("smart-unwrap")) is True

    assert len(doc.history) == depth + 1
    assert doc.history.top.label == "Smart Unwrap"
    assert doc.by_uid(uid).mesh is not mesh_before
    assert doc.by_uid(uid).generator == "box", "UVs are not geometry -- the generator survives"


# --- bake-detail: inline round trip, material replaced by identity -----------


def test_bake_detail_inline_replaces_the_low_objects_material_by_identity_in_one_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clay_blender_mod = _patch_available(monkeypatch, True)
    monkeypatch.setattr(clay_blender_mod, "bake_bytes", _fake_bake_bytes)

    doc, low_uid, _high_uid = _two_object_doc()
    low_mesh_before = doc.by_uid(low_uid).mesh
    material_before = doc.materials[0]
    depth = len(doc.history)

    ctx = _InlineCtx()
    assert (
        clay_ops.run(
            ctx,
            doc,
            clay_ops.get("bake-detail"),
            bake_base_color=1.0,
            bake_roughness=0.0,
            bake_normal=1.0,
        )
        is True
    )

    assert len(doc.history) == depth + 1
    assert doc.history.top.label == "Bake Detail"
    # Geometry is untouched -- only the palette entry changes.
    assert doc.by_uid(low_uid).mesh is low_mesh_before

    material_after = doc.materials[0]
    assert material_after is not material_before, "replaced by identity, never mutated in place"
    assert material_after.base_color is not None
    assert material_after.normal is not None
    assert material_after.metallic_roughness is None, "roughness was not requested"
    assert material_after.metallic_factor == pytest.approx(0.125)

    assert doc.undo() is True
    assert doc.materials[0] is material_before


# --- interactive: submit now, apply later, one object skipped ---------------


def test_a_stale_mesh_stamp_skips_only_that_object(monkeypatch: pytest.MonkeyPatch) -> None:
    clay_blender_mod = _patch_available(monkeypatch, True)
    monkeypatch.setattr(clay_blender_mod, "retopo_bytes", _fake_retopo_bytes)

    doc, box_uid, cone_uid = _two_object_doc()
    state, tab = _tab_with(doc)
    ctx = _InteractiveCtx(state)

    assert clay_ops.run(ctx, doc, clay_ops.get("retopo"), target_faces=1000) is True
    key, fn, args, kwargs = ctx.submitted[0]
    assert key == f"clay-bg:{tab.uid}"
    assert fn is clay_ops._retopo_work
    assert tab.bg_busy == "Retopologizing..."
    result = fn(*args, **kwargs)  # the "task thread" half, run inline
    assert result["kind"] == "retopo"

    # Box changes before the result lands; Cone does not.
    box_mesh_edited = bp.cone()
    doc.set_mesh(box_uid, box_mesh_edited)
    cone_mesh_before = doc.by_uid(cone_uid).mesh
    depth = len(doc.history)

    clay_mode.on_task_done(ctx, _Done(key, result))

    assert tab.bg_busy == ""
    assert doc.by_uid(box_uid).mesh is box_mesh_edited, "the stale object's edit survives untouched"
    assert doc.by_uid(cone_uid).mesh is not cone_mesh_before, "the unstale object still applied"
    assert len(doc.history) == depth + 1, "one folded step for whatever did apply"
    assert doc.history.top.label == "Retopologize"
    assert any("Box" in m and "changed" in m for m, _ in ctx.toasted), ctx.toasted


def test_bake_detail_interactive_submits_a_clay_bg_task_tagged_bake(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clay_blender_mod = _patch_available(monkeypatch, True)
    monkeypatch.setattr(clay_blender_mod, "bake_bytes", _fake_bake_bytes)

    doc, low_uid, _high_uid = _two_object_doc()
    del low_uid
    state, tab = _tab_with(doc)
    ctx = _InteractiveCtx(state)
    depth = len(doc.history)

    assert clay_ops.run(ctx, doc, clay_ops.get("bake-detail")) is True
    key, fn, args, kwargs = ctx.submitted[0]
    assert key == f"clay-bg:{tab.uid}"
    assert fn is clay_ops._bake_work
    assert tab.bg_busy == "Baking..."
    assert len(doc.history) == depth, "nothing changed yet -- a submit is not a result"

    result = fn(*args, **kwargs)
    assert result["kind"] == "bake"
    depth = len(doc.history)
    clay_mode.on_task_done(ctx, _Done(key, result))
    assert tab.bg_busy == ""
    assert len(doc.history) == depth + 1
    assert doc.history.top.label == "Bake Detail"


# --- a ClayBlenderError becomes a toast and no edit --------------------------


@pytest.mark.parametrize(
    ("op_name", "fn_name", "run_kwargs"),
    [
        ("retopo", "retopo_bytes", {"target_faces": 1000}),
        ("smart-unwrap", "unwrap_bytes", {}),
        ("bake-detail", "bake_bytes", {}),
    ],
)
def test_a_clay_blender_error_becomes_a_toast_and_no_edit(
    monkeypatch: pytest.MonkeyPatch, op_name: str, fn_name: str, run_kwargs: dict
) -> None:
    clay_blender_mod = _patch_available(monkeypatch, True)

    def _boom(*_a: Any, **_kw: Any) -> Any:
        raise clay_blender.ClayBlenderError("bpy is not installed")

    monkeypatch.setattr(clay_blender_mod, fn_name, _boom)

    if op_name == "bake-detail":
        doc, _low, _high = _two_object_doc()
    else:
        doc, _uid = _box_doc()
    depth = len(doc.history)
    materials_before = list(doc.materials)

    ctx = _InlineCtx()
    assert clay_ops.run(ctx, doc, clay_ops.get(op_name), **run_kwargs) is True
    assert len(doc.history) == depth, "the op ran (and failed), so nothing landed to undo"
    assert doc.materials == materials_before
    assert any(
        "bpy is not installed" in m for m, level in ctx.toasted if level == "error"
    ), ctx.toasted
