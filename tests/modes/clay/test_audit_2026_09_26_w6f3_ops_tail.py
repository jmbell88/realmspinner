"""Regression tests for the 2026-09-26 audit's clay-ops-tail findings (wave 6,
fixer w6f3), all against ``src/realmspinner/studio/modes/clay/ops.py``.

Each test's name is the claim it makes about the unfixed code; see the
individual finding numbers in each docstring for the fuller incident.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from realmspinner.kernels.geom3d import gltf as gltf_mod
from realmspinner.kernels.geom3d import math3d as m3
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh.elements import OpError
from realmspinner.studio.modes.clay import ops as clay_ops


class _Toasts:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.info: list[str] = []


class _Ctx:
    """The bare toast-only double the rest of this package's ops tests use."""

    def __init__(self) -> None:
        self.toasts = _Toasts()

    def toast(self, message: str, level: str = "info") -> None:
        if level == "error":
            self.toasts.errors.append(message)
        else:
            self.toasts.info.append(message)


class _StateCtx(_Ctx):
    """A ``ctx`` that also carries ``state.clay.manifold``, the per-object
    mesh-check cache ``_forget_manifold`` prunes -- the same double
    ``test_clay_ops.py``'s own boolean-absorption test uses."""

    def __init__(self) -> None:
        super().__init__()
        self.state = SimpleNamespace(clay=SimpleNamespace(manifold={}))


def _doc() -> tuple[bd.ClayDoc, int]:
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box(), generator="box"))
    return doc, obj.uid


def _faces(doc: bd.ClayDoc, uid: int, *faces: int) -> None:
    doc.set_element_mode("face")
    doc.set_element_sel(uid, el.ElementSel(faces=list(faces)))


def _two_loose_boxes(offset: tuple[float, float, float] = (5.0, 0.0, 0.0)) -> bm.Mesh:
    """Two boxes concatenated with no shared vertex index -- two loose parts
    in one mesh. ``tests/modes/clay/test_clay_structure_ops.py``'s own
    helper, copied rather than imported: this file owns no shared fixture
    module and the audit's fixers must not touch each other's test files."""
    a = bp.box()
    b = bp.box()
    b_positions = np.asarray(b.positions, dtype="f4") + np.asarray(offset, dtype="f4")
    merged = bm.Mesh(
        positions=np.concatenate([a.positions, b_positions]),
        loops=np.concatenate([a.loops, b.loops + len(a.positions)]),
        starts=np.concatenate([a.starts, a.starts[-1] + b.starts[1:]]),
        material=np.concatenate([a.material, b.material]),
        smooth=np.concatenate([a.smooth, b.smooth]),
    )
    bm.validate(merged)
    return merged


# --- clay-ops-tail-02: _decimate_apply's undo gesture must close even when --
# --- the glb-to-mesh conversion itself raises -------------------------------


def test_decimate_apply_closes_the_gesture_even_when_the_glb_conversion_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 2026-09-26 audit's clay-ops-tail-02: ``_decimate_mesh_from_glb``
    used to run *outside* ``_decimate_apply``'s own try/finally, so a
    conversion error (its own ``OpError`` for a prim-less GLB, or anything
    ``glbimport``/``join`` could raise on a malformed result) escaped past
    ``doc.history.collapse_since(mark)`` and left ``UndoStack._open_gestures``
    stuck at 1 forever -- disabling undo eviction for the rest of the
    session, the same leak shape the neighbouring comment already fixed once
    for ``set_mesh``'s own lock refusal (the 2026-09-22 audit's clay-03).
    """
    doc, uid = _doc()

    def _boom(data: bytes, material: int) -> bm.Mesh:
        raise OpError("Decimate produced a mesh with no geometry.")

    monkeypatch.setattr(clay_ops, "_decimate_mesh_from_glb", _boom)

    ctx = _Ctx()
    result = {
        "items": [
            {
                "uid": uid,
                "name": doc.by_uid(uid).name,
                "stamp": doc.mesh_stamp(uid),
                "glb_out": b"",
                "material": 0,
                "before": 12,
            }
        ],
        "ratio": 0.5,
    }
    depth = len(doc.history)

    with pytest.raises(OpError):
        clay_ops._decimate_apply(ctx, doc, result)

    assert doc.history._open_gestures == 0, (
        "the gesture must close even when the conversion itself raises"
    )
    assert len(doc.history) == depth, "nothing was actually applied"


# --- clay-ops-tail-03: array-linear/array-radial must step world space, ----
# --- not the copy's local TRS -----------------------------------------------


def test_array_linear_steps_the_world_offset_under_a_scaled_rotated_parent() -> None:
    """The 2026-09-26 audit's clay-ops-tail-03: array-linear's own docstring
    promises a *world*-space step ("each further along one step"), but the
    unfixed op added the offset straight to the copy's *local* translation --
    correct only because a root's local frame is its own world frame. A
    child of a 2x-scaled, 90-degree-(about Y)-rotated parent, stepped by a
    world (1, 0, 0), used to land at (0, 0, -2) instead.
    """
    doc = bd.ClayDoc()
    parent = doc.add_object(
        bd.Obj(
            uid=bd.new_uid(),
            name="Parent",
            mesh=bp.box(),
            rotation=m3.quat_from_axis_angle(m3.vec3(0.0, 1.0, 0.0), math.radians(90.0)),
            scale=m3.vec3(2.0, 2.0, 2.0),
        )
    )
    child = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Child", mesh=bp.box()))
    doc.set_parent(child.uid, parent.uid, keep_world=False)
    before_world = np.array(doc.world_matrix(child.uid)[:3, 3], copy=True)
    doc.select([child.uid])

    assert clay_ops.run(_Ctx(), doc, clay_ops.get("array-linear"), count=2, x=1.0) is True

    copy = next(obj for obj in doc.objects if obj.uid not in (parent.uid, child.uid))
    after_world = doc.world_matrix(copy.uid)[:3, 3]
    assert np.allclose(after_world, before_world + [1.0, 0.0, 0.0], atol=1e-6), (
        f"expected the world-space step (1, 0, 0) applied to {before_world}, got {after_world}"
    )


def test_array_radial_spins_the_world_placement_under_a_rotated_parent() -> None:
    """The 2026-09-26 audit's clay-ops-tail-03: array-radial's own docstring
    says it spins copies "about the *world* origin", but the unfixed op spun
    the copy's *local* translation/rotation about what it treated as the
    origin -- only actually the world origin for a root. A child of a
    90-degree-(about Z)-rotated parent, spun 90 degrees about the world Y
    axis, must land where spinning its *world* position about world Y
    actually puts it, not where spinning its local position (then reading
    that through the parent's own rotation) does.
    """
    doc = bd.ClayDoc()
    parent = doc.add_object(
        bd.Obj(
            uid=bd.new_uid(),
            name="Parent",
            mesh=bp.box(),
            rotation=m3.quat_from_axis_angle(m3.vec3(0.0, 0.0, 1.0), math.radians(90.0)),
        )
    )
    child = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="Child", mesh=bp.box(), translation=m3.vec3(1.0, 0.0, 2.0))
    )
    doc.set_parent(child.uid, parent.uid, keep_world=False)
    before_world = np.array(doc.world_matrix(child.uid), copy=True)
    doc.select([child.uid])

    op = clay_ops.get("array-radial")
    assert clay_ops.run(_Ctx(), doc, op, count=2, angle=90.0, axis=1) is True

    copy = next(obj for obj in doc.objects if obj.uid not in (parent.uid, child.uid))
    after_world = doc.world_matrix(copy.uid)

    spin = m3.quat_from_axis_angle(m3.vec3(0.0, 1.0, 0.0), math.radians(90.0))
    expected_world = m3.quat_to_mat4(spin) @ before_world
    assert np.allclose(after_world, expected_world, atol=1e-6), (
        f"expected the world-space spin of {before_world[:3, 3]}, got {after_world[:3, 3]}"
    )


# --- clay-ops-tail-04: Bake Detail must not overwrite a shared material slot


def test_bake_apply_gives_the_low_object_its_own_material_slot_instead_of_the_shared_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 2026-09-26 audit's clay-ops-tail-04: ``_bake_apply`` used to call
    ``doc.set_material(index, ...)`` where ``index`` was the low object's
    *current* slot, shared with anything else on the same default material
    (the high-poly source, most often, since bake-detail leaves it in
    place) -- so baking one object silently retextured everyone else
    sharing that slot.
    """
    doc = bd.ClayDoc()
    high = doc.add_object(bd.Obj(uid=bd.new_uid(), name="High", mesh=bp.box()))
    low = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Low", mesh=bp.box()))
    assert high.material == low.material == 0, "both start on the document's shared default slot"
    original_default = doc.materials[0]

    baked_material = gltf_mod.Material(name="baked", base_color_factor=(1.0, 0.0, 0.0, 1.0))
    monkeypatch.setattr(clay_ops, "_blender_bake_material", lambda data: baked_material)

    result = {
        "meta": [{"uid": low.uid, "name": low.name, "stamp": doc.mesh_stamp(low.uid)}],
        "glb_out": b"not-empty",
        "report": {"maps": ["base_color"], "metallic": 0.5},
    }
    clay_ops._bake_apply(_Ctx(), doc, result)

    new_low_index = doc.by_uid(low.uid).material
    assert new_low_index != 0, "the low object must move off the shared default slot"
    assert doc.by_uid(high.uid).material == 0, "the high source must keep its own slot"
    assert doc.materials[0] is original_default, "the shared slot's own entry must be untouched"
    assert doc.materials[new_low_index].base_color_factor == (1.0, 0.0, 0.0, 1.0)


def test_bake_apply_leaves_the_low_objects_faces_on_the_baked_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """clay-13 (2026-10-03): the baked slot was appended and ``Obj.material``
    pointed at it, but every face kept its old slot, so the bake never showed."""
    doc = bd.ClayDoc()
    low = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Low", mesh=bp.box()))
    baked_material = gltf_mod.Material(name="baked", base_color_factor=(1.0, 0.0, 0.0, 1.0))
    monkeypatch.setattr(clay_ops, "_blender_bake_material", lambda data: baked_material)
    result = {
        "meta": [{"uid": low.uid, "name": low.name, "stamp": doc.mesh_stamp(low.uid)}],
        "glb_out": b"not-empty",
        "report": {"maps": ["base_color"], "metallic": 0.5},
    }
    clay_ops._bake_apply(_Ctx(), doc, result)
    obj = doc.by_uid(low.uid)
    assert obj.material != 0
    assert set(obj.mesh.material.tolist()) == {obj.material}
    doc.undo()
    assert set(doc.by_uid(low.uid).mesh.material.tolist()) == {0}, "one undo step"


# --- clay-ops-tail-05: face-mode Shade Smooth/Flat must skip a locked -------
# --- object rather than abort the rest of the batch -------------------------


def test_shade_in_face_mode_skips_a_locked_object_without_aborting_the_rest() -> None:
    """The 2026-09-26 audit's clay-ops-tail-05: the face-mode branch of
    ``_shade`` looped ``doc.set_shading`` with no per-object guard at all,
    unlike every other multi-object loop in this module -- a single locked
    object raised ``OpError`` and aborted shading for every object still
    queued behind it in ``doc.element_sel``.
    """
    doc = bd.ClayDoc()
    locked_obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Locked", mesh=bp.box()))
    other = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Other", mesh=bp.box()))
    _faces(doc, locked_obj.uid, 0)
    _faces(doc, other.uid, 0)
    doc.by_uid(locked_obj.uid).locked = True

    assert clay_ops.run(_Ctx(), doc, clay_ops.get("shade-smooth")) is True

    assert bool(doc.by_uid(other.uid).mesh.smooth[0]) is True, (
        "the object queued behind the locked one must still be shaded"
    )
    assert bool(doc.by_uid(locked_obj.uid).mesh.smooth[0]) is False, (
        "the locked object itself must be refused, not silently shaded"
    )


# --- clay-ops-tail-06: separate/ungroup must forget the source's manifold --
# --- cache entry -------------------------------------------------------------


def test_separate_loose_forgets_the_sources_manifold_cache_entry() -> None:
    """The 2026-09-26 audit's clay-ops-tail-06: ``ClayDoc.separate`` always
    removes the source object (its own docstring), but ``_separate_loose``
    never popped it from the properties panel's per-object mesh-check
    cache -- the same leak shape ``_delete``/the boolean ops already close
    for their own removals (the 2026-09-08 audit's clay-08), left open here.
    """
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Both", mesh=_two_loose_boxes()))
    doc.select([obj.uid])
    ctx = _StateCtx()
    ctx.state.clay.manifold[obj.uid] = object()

    assert clay_ops.run(ctx, doc, clay_ops.get("separate-loose")) is True

    assert obj.uid not in [o.uid for o in doc.objects]
    assert obj.uid not in ctx.state.clay.manifold, "left a stale cache entry for the removed source"


def test_ungroup_forgets_the_removed_emptys_manifold_cache_entry() -> None:
    """The 2026-09-26 audit's clay-ops-tail-06: ``_ungroup`` removes the
    group empty via ``doc.remove_object`` but never popped it from the
    manifold cache, the same leak shape as ``_separate_loose``'s own."""
    doc = bd.ClayDoc()
    child = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Child", mesh=bp.box()))
    empty = doc.group([child.uid])
    doc.select([empty.uid])
    ctx = _StateCtx()
    ctx.state.clay.manifold[empty.uid] = object()

    assert clay_ops.run(ctx, doc, clay_ops.get("ungroup")) is True

    assert empty.uid not in [o.uid for o in doc.objects]
    assert empty.uid not in ctx.state.clay.manifold, "left a stale cache entry for the empty"
