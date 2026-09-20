"""``clean-mesh`` and ``recalc-normals``: the two synchronous ops this tranche
adds, both named in ``kernels/mesh/readiness.py``'s ``FIX_OPS`` and so load-
bearing under those exact string names.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from warlock.kernels.mesh import document as bd
from warlock.kernels.mesh import ops_clean, topo
from warlock.kernels.mesh import primitives as bp
from warlock.studio.modes.clay import ops as clay_ops


class _Ctx:
    """Records every toast, at whatever level it was shown -- ``test_clay_ops.
    py``'s own double only keeps the error ones, and these tests need the
    plain summary toasts too."""

    def __init__(self) -> None:
        self.toasted: list[tuple[str, str]] = []

    def toast(self, message: str, level: str = "info") -> None:
        self.toasted.append((message, level))


def _flipped_and_duplicated_box() -> Any:
    """A box with face 0's winding reversed (a flipped-face defect: the shell
    is otherwise consistent, so this one disagrees with its neighbours) and
    face 1 duplicated (a second, identical face appended)."""
    mesh = bp.box()
    loops = mesh.loops.copy()
    starts = mesh.starts.copy()
    material = mesh.material.copy()
    smooth = mesh.smooth.copy()

    lo0, hi0 = int(starts[0]), int(starts[1])
    loops[lo0:hi0] = loops[lo0:hi0][::-1]

    lo1, hi1 = int(starts[1]), int(starts[2])
    dup = loops[lo1:hi1].copy()
    new_loops = np.concatenate([loops, dup])
    new_starts = np.concatenate([starts, [int(starts[-1]) + len(dup)]])
    new_material = np.concatenate([material, [int(material[1])]])
    new_smooth = np.concatenate([smooth, [bool(smooth[1])]])
    return topo.rebuild(mesh.positions, new_loops, new_starts, new_material, new_smooth, uv=None)


def _inside_out_box() -> Any:
    """Every face's winding reversed: a *uniformly* wound shell that is net
    inside-out (``Survey.inside_out_shells``), the defect ``recalc-normals``
    exists for -- distinct from a single disagreeing face."""
    mesh = bp.box()
    loops = mesh.loops.copy()
    starts = mesh.starts.astype("i8")
    for i in range(len(starts) - 1):
        lo, hi = int(starts[i]), int(starts[i + 1])
        loops[lo:hi] = loops[lo:hi][::-1]
    return topo.rebuild(mesh.positions, loops, mesh.starts, mesh.material, mesh.smooth, uv=None)


# --- clean-mesh ---------------------------------------------------------------


def test_clean_mesh_fixes_a_flip_and_a_duplicate_in_one_step_and_toasts_the_summary() -> None:
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=_flipped_and_duplicated_box()))
    doc.select([obj.uid])
    before = ops_clean.survey(obj.mesh)
    assert before.duplicate_faces >= 1
    assert before.flipped_faces >= 1

    depth = len(doc.history)
    ctx = _Ctx()
    assert clay_ops.run(ctx, doc, clay_ops.get("clean-mesh")) is True
    assert len(doc.history) == depth + 1, "one undo step for the whole clean"
    assert doc.history.top.label == "Clean Up"

    after = ops_clean.survey(doc.by_uid(obj.uid).mesh)
    assert after.duplicate_faces == 0
    assert after.flipped_faces == 0
    assert doc.by_uid(obj.uid).generator is None, "geometry changed -- frozen like any other op"

    messages = " ".join(m for m, _ in ctx.toasted)
    assert "duplicate" in messages
    assert "flipped" in messages

    assert doc.undo() is True
    restored = ops_clean.survey(doc.by_uid(obj.uid).mesh)
    assert restored.duplicate_faces >= 1 and restored.flipped_faces >= 1


def test_clean_mesh_on_a_clean_primitive_pushes_no_step_and_keeps_the_generator() -> None:
    doc = bd.ClayDoc()
    obj = doc.add_object(
        bd.Obj(
            uid=bd.new_uid(),
            name="Box",
            mesh=bp.box(),
            generator="box",
            params={"size": (1.0, 1.0, 1.0)},
        )
    )
    doc.select([obj.uid])
    depth = len(doc.history)
    ctx = _Ctx()

    assert clay_ops.run(ctx, doc, clay_ops.get("clean-mesh")) is False
    assert len(doc.history) == depth, "nothing to fix, nothing pushed"
    assert doc.by_uid(obj.uid).generator == "box"
    assert doc.by_uid(obj.uid).mesh is obj.mesh, "the identical object, not a rebuild"
    assert any("Nothing to clean" in m for m, _ in ctx.toasted)


def test_clean_mesh_is_gated_on_a_selection() -> None:
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
    op = clay_ops.get("clean-mesh")
    assert not op.enabled(doc)
    assert clay_ops.reason_for(op, doc) == "Select an object first."
    doc.select([obj.uid])
    assert op.enabled(doc)


def test_clean_mesh_refuses_a_refusal_on_one_object_without_abandoning_the_others() -> None:
    """A mesh past ``ops_clean.MAX_CLEAN_CORNERS`` refuses per object; the rest
    of the selection still gets cleaned -- ``run_object_op``'s own contract,
    exercised here rather than assumed."""
    import warlock.kernels.mesh.ops_clean as ops_clean_mod

    doc = bd.ClayDoc()
    small_mesh = _flipped_and_duplicated_box()
    huge_mesh = bp.uv_sphere(segments=16, rings=8)
    assert len(huge_mesh.loops) > len(small_mesh.loops)
    huge = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Huge", mesh=huge_mesh))
    small = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Small", mesh=small_mesh))
    doc.select([huge.uid, small.uid])

    original = ops_clean_mod.MAX_CLEAN_CORNERS
    ops_clean_mod.MAX_CLEAN_CORNERS = len(small_mesh.loops)
    try:
        ctx = _Ctx()
        assert clay_ops.run(ctx, doc, clay_ops.get("clean-mesh")) is True
    finally:
        ops_clean_mod.MAX_CLEAN_CORNERS = original

    after_small = ops_clean.survey(doc.by_uid(small.uid).mesh)
    assert after_small.duplicate_faces == 0, "the object under the ceiling still got cleaned"
    assert doc.by_uid(huge.uid).mesh is huge.mesh, "the oversized object was left alone"


# --- recalc-normals -------------------------------------------------------


def test_recalc_normals_flips_an_inside_out_cube_outward_in_one_step() -> None:
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=_inside_out_box()))
    doc.select([obj.uid])
    before = ops_clean.survey(obj.mesh)
    assert before.inside_out_shells == 1

    depth = len(doc.history)
    ctx = _Ctx()
    assert clay_ops.run(ctx, doc, clay_ops.get("recalc-normals")) is True
    assert len(doc.history) == depth + 1

    after = ops_clean.survey(doc.by_uid(obj.uid).mesh)
    assert after.inside_out_shells == 0
    assert doc.by_uid(obj.uid).generator is None, "winding is not a generator fact"

    assert doc.undo() is True
    restored = ops_clean.survey(doc.by_uid(obj.uid).mesh)
    assert restored.inside_out_shells == 1


def test_recalc_normals_on_an_untouched_primitive_is_a_no_op() -> None:
    """Every primitive generator already emits outward, consistent winding --
    see ``_recalc_normals``'s own docstring -- so this is unreachable in
    practice and the identity check is what proves it rather than merely
    asserting it in prose."""
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box(), generator="box"))
    doc.select([obj.uid])
    depth = len(doc.history)

    clay_ops.run(_Ctx(), doc, clay_ops.get("recalc-normals"))

    assert len(doc.history) == depth, "identity in, identity out -- nothing pushed"
    assert doc.by_uid(obj.uid).generator == "box"


def test_recalc_normals_is_object_mode_only() -> None:
    assert clay_ops.get("recalc-normals").modes == ("object",)


def test_recalc_normals_is_gated_on_a_selection() -> None:
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
    op = clay_ops.get("recalc-normals")
    assert not op.enabled(doc)
    doc.select([obj.uid])
    assert op.enabled(doc)


# --- both ops sweep cleanly with the rest of the registry --------------------


def test_both_new_ops_appear_in_the_object_menu() -> None:
    names = [op.name for op in clay_ops.menu("object")]
    assert "clean-mesh" in names
    assert "recalc-normals" in names
    assert "decimate" in names


def test_readiness_fix_ops_are_all_reachable_by_name() -> None:
    """``readiness.FIX_OPS`` names every op a "Fix" button may run; each one
    has to actually be registered, or the button would run ``ops.get`` on a
    name nothing answers to."""
    from warlock.kernels.mesh import readiness

    registered = {op.name for op in clay_ops.OPS}
    missing = readiness.FIX_OPS - registered
    assert not missing, f"FIX_OPS names ops the registry does not have: {missing}"
