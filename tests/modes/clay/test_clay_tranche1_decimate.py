"""Decimate: Clay's first background op.

Every op before this one ran synchronously on the frame thread; Decimate is a
child process (gltfpack) wrapped in the ``prepare``/``work``/``apply`` split
``clay_ops`` documents in its own "decimate: Clay's first background op"
section. These tests drive all three dispatch shapes: the agent's inline
path (work and apply both run inside one call), the interactive path (a
``clay-bg:<tab uid>`` task is submitted and applied later, from
``clay_mode.on_task_done``, only if the mesh has not moved on), and the
refusals that never reach either.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from realmspinner.kernels.geom3d import gltf as gltf_mod
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import topo
from realmspinner.kernels.mesh import uv as uv_mod
from realmspinner.pipelines import optimize as optimize_mod
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay import ops as clay_ops
from realmspinner.studio.modes.clay import state as clay_state


class _Ctx:
    """A minimal ``Ctx``: toasts recorded, nothing submitted for real."""

    def __init__(self) -> None:
        self.toasted: list[tuple[str, str]] = []

    def toast(self, message: str, level: str = "info") -> None:
        self.toasted.append((message, level))


class _InlineCtx(_Ctx):
    def __init__(self, gltfpack_exe: Path) -> None:
        super().__init__()
        self.inline = True
        self.gltfpack_exe = gltfpack_exe


class _InteractiveCtx(_Ctx):
    """Records what would have been submitted, rather than running it -- so a
    test can drive the background half by hand, the way the real task runner
    would on a later frame."""

    def __init__(self, gltfpack_exe: Path, state: Any) -> None:
        super().__init__()
        self.svc = SimpleNamespace(config=SimpleNamespace(gltfpack_exe=gltfpack_exe))
        self.state = state
        self.submitted: list[tuple[str, Any, tuple, dict]] = []

    def submit(self, key: str, fn: Any, *args: Any, **kwargs: Any) -> bool:
        self.submitted.append((key, fn, args, kwargs))
        return True


class _Done:
    def __init__(self, key: str, result: Any) -> None:
        self.key = key
        self.result = result


def _two_material_box() -> tuple[bd.ClayDoc, int]:
    """A box, three quads on material 0, three on material 1 -- so a decimate
    round trip has a real palette mapping to get right rather than one slot
    trivially matching itself."""
    doc = bd.ClayDoc()
    doc.materials.append(gltf_mod.Material(name="second"))
    mesh = bp.box()
    material = mesh.material.copy()
    material[3:] = 1
    corrupted = topo.rebuild(
        mesh.positions, mesh.loops, mesh.starts, material, mesh.smooth, uv=None
    )
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=corrupted))
    doc.select([obj.uid])
    return doc, obj.uid


# --- refusals -----------------------------------------------------------------


def test_decimate_refuses_when_the_exe_is_missing(tmp_path: Path) -> None:
    doc, uid = _two_material_box()
    del uid
    ctx = _InlineCtx(tmp_path / "nope.exe")

    depth = len(doc.history)
    assert clay_ops.run(ctx, doc, clay_ops.get("decimate")) is False
    assert len(doc.history) == depth
    assert any("gltfpack" in m for m, _ in ctx.toasted)


def test_decimate_refuses_a_ratio_of_one(tmp_path: Path) -> None:
    exe = tmp_path / "gltfpack.exe"
    exe.write_bytes(b"")
    doc, uid = _two_material_box()
    del uid
    ctx = _InlineCtx(exe)

    assert clay_ops.run(ctx, doc, clay_ops.get("decimate"), ratio=1.0) is False
    assert any("nothing to decimate" in m.lower() for m, _ in ctx.toasted)


def test_a_multi_object_decimate_refuses_on_the_selections_total_before_triangulating_any_of_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """clay-40 (2026-09-19 audit, found during the fix phase, the ``ops-tail``
    reading debt): ``_decimate_prepare`` ran ``_decimate_primitives`` once per
    selected uid with no summed ceiling -- only ``_decimate``'s own
    per-object check against ``glbimport.MAX_TRIANGLES`` existed, and a
    selection of several legally-sized objects never trips a per-object cap.
    Two boxes (12 triangles each, 24 summed) trip a ceiling monkeypatched to
    20, below either alone -- proving this is the selection's total.
    """
    from realmspinner.kernels.mesh.elements import OpError

    doc = bd.ClayDoc()
    a = doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box()))
    b = doc.add_object(bd.Obj(uid=bd.new_uid(), name="B", mesh=bp.box()))

    monkeypatch.setattr(clay_ops, "MAX_PRIMITIVES_TRIANGLES", 20)
    with pytest.raises(OpError):
        clay_ops._decimate_prepare(doc, [a.uid, b.uid])

    prepared = clay_ops._decimate_prepare(doc, [a.uid])
    assert len(prepared) == 1


# --- the prepared GLB -----------------------------------------------------


def test_the_prepared_glb_has_no_normals(tmp_path: Path) -> None:
    doc, uid = _two_material_box()
    prepared = clay_ops._decimate_prepare(doc, [uid])
    assert len(prepared) == 1

    model = gltf_mod.load(prepared[0]["glb"])
    prims = [p for node in model.nodes if node.mesh is not None for p in model.meshes[node.mesh]]
    assert prims, "the prepared GLB carries no geometry"
    for prim in prims:
        assert prim.normals is None


# --- inline (the agent's sandboxed path) --------------------------------------


def test_decimate_inline_replaces_the_mesh_in_one_undo_step_and_keeps_the_palette(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exe = tmp_path / "gltfpack.exe"
    exe.write_bytes(b"")
    monkeypatch.setattr(optimize_mod, "simplify_bytes", lambda data, **_kw: data)

    doc, uid = _two_material_box()
    original_mesh = doc.by_uid(uid).mesh
    depth = len(doc.history)
    materials_before = len(doc.materials)

    ctx = _InlineCtx(exe)
    assert clay_ops.run(ctx, doc, clay_ops.get("decimate"), ratio=0.5) is True

    assert len(doc.history) == depth + 1, "one undo step for the whole gesture"
    assert doc.history.top.label == "Decimate"
    assert doc.undo() is True
    assert doc.by_uid(uid).mesh is original_mesh, "the undo restores the pre-decimate mesh"
    assert doc.redo() is True

    mesh = doc.by_uid(uid).mesh
    assert mesh is not original_mesh
    assert len(doc.materials) == materials_before, "no new palette slots"
    # Three quads on material 0 fan-triangulate to six triangles, and the
    # other three (material 1) do the same -- the monkeypatched
    # ``simplify_bytes`` is the identity, so the round trip through
    # ``_decimate_mesh_from_glb`` must land every triangle back on the slot
    # its source face carried.
    counts = {int(m): int((mesh.material == m).sum()) for m in np.unique(mesh.material)}
    assert counts == {0: 6, 1: 6}


def test_decimate_inline_toasts_the_triangle_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exe = tmp_path / "gltfpack.exe"
    exe.write_bytes(b"")
    monkeypatch.setattr(optimize_mod, "simplify_bytes", lambda data, **_kw: data)

    doc, uid = _two_material_box()
    del uid
    ctx = _InlineCtx(exe)
    clay_ops.run(ctx, doc, clay_ops.get("decimate"), ratio=0.5)

    assert any("Decimated" in m and "->" in m for m, _ in ctx.toasted)


# --- interactive: submit now, apply later, only if the mesh has not moved ----


def _tab_with(doc: bd.ClayDoc) -> tuple[clay_state.ClayState, clay_state.ClayTab]:
    state = clay_state.ClayState()
    tab = clay_state.ClayTab(doc=doc, title="Scene")
    state.add(tab)
    return state, tab


def test_decimate_interactive_submits_a_clay_bg_task(tmp_path: Path) -> None:
    exe = tmp_path / "gltfpack.exe"
    exe.write_bytes(b"")
    doc, uid = _two_material_box()
    del uid
    state, tab = _tab_with(doc)

    ctx = _InteractiveCtx(exe, SimpleNamespace(clay=state))
    assert clay_ops.run(ctx, doc, clay_ops.get("decimate"), ratio=0.5) is True

    assert len(ctx.submitted) == 1
    key, fn, _args, _kwargs = ctx.submitted[0]
    assert key == f"clay-bg:{tab.uid}"
    assert fn is clay_ops._decimate_work
    assert tab.bg_busy == "Decimating..."
    # Nothing changed yet -- a submit is not a result.
    assert len(doc.history) == 1


def test_decimate_interactive_applies_only_when_the_stamp_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exe = tmp_path / "gltfpack.exe"
    exe.write_bytes(b"")
    doc, uid = _two_material_box()
    state, tab = _tab_with(doc)
    ctx = _InteractiveCtx(exe, SimpleNamespace(clay=state))

    monkeypatch.setattr(optimize_mod, "simplify_bytes", lambda data, **_kw: data)
    assert clay_ops.run(ctx, doc, clay_ops.get("decimate"), ratio=0.5) is True
    key, fn, args, kwargs = ctx.submitted[0]
    result = fn(*args, **kwargs)  # the "task thread" half, run inline

    # The object changes before the result lands.
    cone_mesh = bp.cone()
    doc.set_mesh(uid, cone_mesh)
    depth = len(doc.history)

    clay_mode.on_task_done(ctx, _Done(key, result))

    assert tab.bg_busy == ""
    assert len(doc.history) == depth, "nothing folded in -- the stamp had moved on"
    assert doc.by_uid(uid).mesh is cone_mesh, "the edit made in the meantime survives untouched"
    assert any("changed while decimating" in m for m, _ in ctx.toasted), ctx.toasted


def test_decimate_interactive_applies_when_the_stamp_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exe = tmp_path / "gltfpack.exe"
    exe.write_bytes(b"")
    doc, uid = _two_material_box()
    state, tab = _tab_with(doc)
    ctx = _InteractiveCtx(exe, SimpleNamespace(clay=state))

    monkeypatch.setattr(optimize_mod, "simplify_bytes", lambda data, **_kw: data)
    original_mesh = doc.by_uid(uid).mesh
    assert clay_ops.run(ctx, doc, clay_ops.get("decimate"), ratio=0.5) is True
    key, fn, args, kwargs = ctx.submitted[0]
    result = fn(*args, **kwargs)
    depth = len(doc.history)

    clay_mode.on_task_done(ctx, _Done(key, result))

    assert tab.bg_busy == ""
    assert len(doc.history) == depth + 1
    assert doc.history.top.label == "Decimate"
    assert doc.by_uid(uid).mesh is not original_mesh


def test_decimate_interactive_toasts_the_optimize_error_and_clears_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exe = tmp_path / "gltfpack.exe"
    exe.write_bytes(b"")
    doc, uid = _two_material_box()
    del uid
    state, tab = _tab_with(doc)
    ctx = _InteractiveCtx(exe, SimpleNamespace(clay=state))

    def _boom(data: bytes, **_kw: Any) -> bytes:
        raise optimize_mod.OptimizeError("gltfpack exited 1: bad input")

    monkeypatch.setattr(optimize_mod, "simplify_bytes", _boom)
    assert clay_ops.run(ctx, doc, clay_ops.get("decimate"), ratio=0.5) is True
    key, fn, args, kwargs = ctx.submitted[0]
    result = fn(*args, **kwargs)
    assert isinstance(result, dict) and "error" in result

    clay_mode.on_task_done(ctx, _Done(key, result))

    assert tab.bg_busy == ""
    assert any("gltfpack exited 1" in m for m, level in ctx.toasted if level == "error")


# --- the real exe, when this checkout carries one -----------------------------

_REAL_GLTFPACK = Path("vendor/gltfpack/gltfpack.exe")


@pytest.mark.skipif(not _REAL_GLTFPACK.is_file(), reason="no vendored gltfpack in this checkout")
def test_decimate_with_the_real_gltfpack_actually_reduces_a_uv_sphere() -> None:
    """The claim behind dropping normals from the prepared GLB
    (``optimize.simplify_bytes``'s own docstring): a Clay ``uv_sphere`` at the
    exact resolution that docstring measured (segments=32, rings=16 -> 960
    triangles), box-unwrapped so it carries real per-corner UV seams, should
    lose a healthy fraction of its triangles at ratio 0.5 with no aggressive
    flag -- where the pre-fix shape (carrying normals, and so a vertex split
    per flat corner) removed nothing at all (960 -> 960, reproduced directly
    against ``document.to_primitives`` in the course of writing this test,
    against this same vendored exe).

    Measured while writing this test, against this exe:

    * The default, coarse ``uv_sphere()`` (224 triangles) is already near
      whatever quality floor this gltfpack converges to independent of the
      ``-si`` ratio asked for -- 224 -> 192 at every ratio from 0.5 down to
      0.1, with or without ``-sa``, with or without ``keep_seams``. A 30%
      target against *that* mesh would be asserting something this exe
      cannot do at any of this op's settings, not something the fix failed
      to reach -- so the higher-resolution sphere below is the one this
      assertion is against.
    * At 960 triangles, both ``keep_seams`` values reach the identical
      960 -> 556 (42%): the seam layout this box-unwrap produces is not what
      was limiting the coarse sphere above.
    """
    doc = bd.ClayDoc()
    mesh = uv_mod.box_unwrap(bp.uv_sphere(segments=32, rings=16))
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Sphere", mesh=mesh))
    doc.select([obj.uid])
    before = clay_ops._tri_count(mesh)
    assert before == 960

    ctx = _InlineCtx(_REAL_GLTFPACK)
    assert clay_ops.run(ctx, doc, clay_ops.get("decimate"), ratio=0.5, aggressive=0.0) is True

    after = clay_ops._tri_count(doc.by_uid(obj.uid).mesh)
    assert after <= before * 0.7, f"{before} -> {after} triangles, expected at least a 30% drop"


# --- clay-03 (2026-09-22 audit): a locked target must not wedge the undo stack ----


def test_decimate_apply_skips_a_locked_object_by_name_and_still_folds_and_closes_the_gesture_for_the_rest_of_the_batch(  # noqa: E501
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A locked object reaches ``_decimate_apply`` because ``has_objects``
    (the op's ``enabled`` predicate) never checks ``locked`` -- so no race
    with an interactive lock toggle is needed to hit this. Before the fix,
    ``doc.set_mesh`` raised ``OpError`` straight out of the open
    ``history.mark()``: ``collapse_since`` never ran, so
    ``UndoStack._open_gestures`` stayed at 1 forever and every later gesture
    in the document's life stopped evicting -- the 2026-09-19 audit's
    clay-16 leak shape, reproduced here for Decimate's own landing.
    """
    doc, locked_uid = _two_material_box()
    doc.by_uid(locked_uid).locked = True
    other = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Other", mesh=bp.box()))
    doc.select([locked_uid, other.uid])

    # The real glb round trip is decimate's own business, already covered
    # above; this test is only about what ``_decimate_apply`` does once it
    # has a mesh in hand, so the glb decode is stubbed to hand back a fresh
    # mesh -- a distinct object identity from what is already on each
    # target, which matters because ``set_mesh`` decides "did anything
    # happen" by identity (its own docstring), not equality.
    monkeypatch.setattr(
        clay_ops, "_decimate_mesh_from_glb", lambda data, material: bp.box()
    )

    ctx = _Ctx()
    result = {
        "items": [
            {
                "uid": locked_uid,
                "name": doc.by_uid(locked_uid).name,
                "stamp": doc.mesh_stamp(locked_uid),
                "glb_out": doc.by_uid(locked_uid).mesh,
                "material": 0,
                "before": 12,
            },
            {
                "uid": other.uid,
                "name": other.name,
                "stamp": doc.mesh_stamp(other.uid),
                "glb_out": doc.by_uid(other.uid).mesh,
                "material": 0,
                "before": 12,
            },
        ],
        "ratio": 0.5,
    }

    depth = len(doc.history)
    clay_ops._decimate_apply(ctx, doc, result)

    assert doc.history._open_gestures == 0, "the gesture must close even when one item refuses"
    assert len(doc.history) == depth + 1, "the other object's decimate still folds into one step"
    assert doc.history.top.label == "Decimate"
