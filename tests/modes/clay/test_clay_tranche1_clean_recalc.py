"""``recalc-normals``: the synchronous winding repair, load-bearing under that
exact string name."""

from __future__ import annotations

from typing import Any

import numpy as np

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import topo
from realmspinner.kernels.mesh.mesh import face_normals
from realmspinner.studio.modes.clay import ops as clay_ops


class _Ctx:
    """Records every toast, at whatever level it was shown."""

    def __init__(self) -> None:
        self.toasted: list[tuple[str, str]] = []

    def toast(self, message: str, level: str = "info") -> None:
        self.toasted.append((message, level))


def _inside_out_box() -> Any:
    """Every face's winding reversed: a *uniformly* wound shell that is net
    inside-out, the defect ``recalc-normals``
    exists for -- distinct from a single disagreeing face."""
    mesh = bp.box()
    loops = mesh.loops.copy()
    starts = mesh.starts.astype("i8")
    for i in range(len(starts) - 1):
        lo, hi = int(starts[i]), int(starts[i + 1])
        loops[lo:hi] = loops[lo:hi][::-1]
    return topo.rebuild(mesh.positions, loops, mesh.starts, mesh.material, mesh.smooth, uv=None)


def _faces_pointing_outward(mesh: Any) -> int:
    """How many faces' normals point away from the mesh centre -- a closed
    convex box is all outward (6 of 6) or all inward (0 of 6)."""
    normals = face_normals(mesh)
    starts = np.asarray(mesh.starts, dtype="i8")
    corners = mesh.positions[mesh.loops]
    centroids = np.add.reduceat(corners, starts[:-1], axis=0) / np.diff(starts)[:, None]
    away = centroids - mesh.positions.mean(axis=0)
    return int((np.einsum("ij,ij->i", normals, away) > 0.0).sum())


# --- recalc-normals -------------------------------------------------------


def test_recalc_normals_flips_an_inside_out_cube_outward_in_one_step() -> None:
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=_inside_out_box()))
    doc.select([obj.uid])
    assert _faces_pointing_outward(obj.mesh) == 0

    depth = len(doc.history)
    ctx = _Ctx()
    assert clay_ops.run(ctx, doc, clay_ops.get("recalc-normals")) is True
    assert len(doc.history) == depth + 1

    assert _faces_pointing_outward(doc.by_uid(obj.uid).mesh) == 6
    assert doc.by_uid(obj.uid).generator is None, "winding is not a generator fact"

    assert doc.undo() is True
    assert _faces_pointing_outward(doc.by_uid(obj.uid).mesh) == 0


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


def test_recalc_normals_appears_in_the_object_menu() -> None:
    assert "recalc-normals" in [op.name for op in clay_ops.menu("object")]
