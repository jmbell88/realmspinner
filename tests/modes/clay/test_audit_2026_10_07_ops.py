"""The 2026-10-07 Clay audit's ops-tail findings: clay-35 through clay-39."""

from __future__ import annotations

import numpy as np
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh.elements import OpError
from realmspinner.studio.modes.clay import ops as clay_ops


class _Ctx:
    def __init__(self) -> None:
        self.log: list[tuple[str, str]] = []

    def toast(self, message: str, level: str = "info") -> None:
        self.log.append((level, message))


def _box(doc: bd.ClayDoc, name: str) -> bd.Obj:
    return doc.add_object(
        bd.Obj(uid=bd.new_uid(), name=name, mesh=bp.box(), generator="box", params={})
    )


def test_mirror_copy_leaves_a_hidden_selected_object_alone() -> None:
    doc = bd.ClayDoc()
    hidden = _box(doc, "Hid")
    visible = _box(doc, "Vis")
    doc.set_visibility({hidden.uid: False})
    doc.select([hidden.uid])

    head = doc.history.head
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("mirror-copy")) is False
    assert [o.name for o in doc.objects] == ["Hid", "Vis"]
    assert doc.history.head == head

    doc.select([hidden.uid, visible.uid])
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("mirror-copy")) is True
    assert len(doc.objects) == 3
    assert sum(1 for o in doc.objects if o.name.startswith("Hid")) == 1


def test_run_refuses_a_nan_parameter_instead_of_placing_an_object_at_nan() -> None:
    nan = float("nan")
    doc = bd.ClayDoc()
    obj = _box(doc, "B")
    doc.set_transform(obj.uid, translation=[0.3, 0.4, 0.5])
    doc.select([obj.uid])
    head = doc.history.head

    for name, params in (
        ("mirror-copy", {"offset": nan}),
        ("snap-to-grid", {"step": nan}),
    ):
        ctx = _Ctx()
        assert clay_ops.run(ctx, doc, clay_ops.get(name), **params) is False
        assert ctx.log and ctx.log[0][0] == "error", name
    assert len(doc.objects) == 1
    assert doc.history.head == head
    assert np.all(np.isfinite(doc.by_uid(obj.uid).translation))

    doc.set_element_mode("face")
    from realmspinner.kernels.mesh import elements as el

    doc.set_element_sel(obj.uid, el.ElementSel(faces=[0]))
    ctx = _Ctx()
    assert clay_ops.run(ctx, doc, clay_ops.get("assign-material"), index=nan) is False
    assert ctx.log and ctx.log[0][0] == "error"


def test_snap_to_grid_with_a_zero_step_is_refused_with_a_sentence() -> None:
    doc = bd.ClayDoc()
    obj = _box(doc, "B")
    doc.set_transform(obj.uid, translation=[0.3, 0.4, 0.5])
    doc.select([obj.uid])
    head = doc.history.head
    ctx = _Ctx()

    assert clay_ops.run(ctx, doc, clay_ops.get("snap-to-grid"), step=0.0) is False
    assert doc.history.head == head
    assert ctx.log and ctx.log[0][0] == "error" and "step" in ctx.log[0][1]


def test_parent_to_last_reports_false_when_every_parenting_was_refused() -> None:
    doc = bd.ClayDoc()
    child = _box(doc, "Child")
    par = _box(doc, "Par")
    doc.set_parent(child.uid, par.uid)
    doc.select([child.uid, par.uid])
    head = doc.history.head
    ctx = _Ctx()

    # Child sits first in the outliner, so it is the target and its own
    # parent is the object asked to hang under it: a cycle, refused.
    assert clay_ops.run(ctx, doc, clay_ops.get("parent-to-last")) is False
    assert doc.history.head == head
    assert ctx.log and ctx.log[0][0] == "error"


def test_mirror_x_on_an_empty_group_pushes_no_step() -> None:
    doc = bd.ClayDoc()
    a = _box(doc, "A")
    b = _box(doc, "B")
    group = doc.group([a.uid, b.uid])
    doc.select([group.uid])
    head = doc.history.head

    clay_ops.run(_Ctx(), doc, clay_ops.get("mirror-x"))
    assert doc.history.head == head
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("mirror-x")) is False


@pytest.mark.parametrize("name", ["shade-smooth", "shade-flat"])
def test_shade_in_object_mode_reports_false_when_every_object_refused(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 2026-10-07 audit's clay-38 residual: Shade Smooth/Flat dropped
    ``run_object_op``'s result, so ``run`` said True when every object refused."""
    doc = bd.ClayDoc()
    a = _box(doc, "A")
    b = _box(doc, "B")
    doc.select([a.uid, b.uid])
    head = doc.history.head

    def refuse(uid: int, faces: object, smooth: bool) -> bool:
        raise OpError("This object cannot be shaded.")

    monkeypatch.setattr(doc, "set_shading", refuse)
    ctx = _Ctx()
    assert clay_ops.run(ctx, doc, clay_ops.get(name)) is False
    assert doc.history.head == head
    assert len(ctx.log) == 2 and ctx.log[0][0] == "error"
