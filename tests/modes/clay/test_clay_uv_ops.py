"""Pack UV Islands, the one UV row of Clay's OPS registry besides Box Unwrap
(integration half).

``kernels.mesh.uvtools`` is the kernel half and is tested on its own terms in
``test_uvtools.py``; what belongs here is the registry wiring: that the context
menu, the tools pane and the keyboard reach the row through the one ``OPS``
list, that it is gated to object mode with a reason a refused user can read,
that Pack UV Islands edits uv as one step and keeps the generator (a uv is not
geometry, the same rule Box Unwrap already states), and that the agent's derived
``clay_op`` enum picked the row up with nothing hand-listed there.
"""

from __future__ import annotations

import numpy as np
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.kernels.mesh import uv as uv_mod
from realmspinner.kernels.mesh import uvtools
from realmspinner.studio.modes.clay import ops as clay_ops


class _Toasts:
    """Only what ``Ctx`` really offers -- ``test_clay_ops.py``'s own double,
    reused here rather than reaching a ``ctx.toasts.error`` method the real
    ``Ctx`` has never had."""

    def __init__(self) -> None:
        self.errors: list[str] = []


class _Ctx:
    def __init__(self) -> None:
        self.toasts = _Toasts()

    def toast(self, message: str, level: str = "info") -> None:
        if level == "error":
            self.toasts.errors.append(message)


def _doc() -> tuple[bd.ClayDoc, int]:
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box(), generator="box"))
    return doc, obj.uid


def _unwrapped_doc() -> tuple[bd.ClayDoc, int]:
    """A box already box-unwrapped, for the rows that need a uv to run at
    all (Pack UV Islands, Normalise Texel Density)."""
    doc = bd.ClayDoc()
    mesh = uv_mod.box_unwrap(bp.box())
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=mesh, generator="box"))
    return doc, obj.uid


def _quad_no_uv() -> bm.Mesh:
    """One quad with no uv at all -- ``bp.box()`` already carries a default
    uv (game props ship pre-unwrapped), so the "needs a uv" refusal needs a
    mesh built by hand rather than any primitive in this package."""
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype="f4"
    )
    mesh = bm.Mesh(
        positions=positions,
        loops=np.array([0, 1, 2, 3], dtype="i4"),
        starts=np.array([0, 4], dtype="i4"),
        material=np.zeros(1, dtype="i4"),
        smooth=np.zeros(1, dtype=bool),
    )
    bm.validate(mesh)
    return mesh


NEW_OBJECT_ROWS = ("pack-uv",)
NEW_ROWS = NEW_OBJECT_ROWS


# --- registry wiring: menu membership and greyed reasons ---------------------


def test_every_new_row_is_registered_exactly_once() -> None:
    names = [op.name for op in clay_ops.OPS]
    for name in NEW_ROWS:
        assert names.count(name) == 1, name


@pytest.mark.parametrize("name", NEW_OBJECT_ROWS)
def test_new_object_rows_are_object_only_and_greyed_with_nothing_selected(name: str) -> None:
    doc = bd.ClayDoc()
    assert name in {op.name for op in clay_ops.menu("object")}
    for mode in ("vertex", "edge", "face"):
        assert name not in {op.name for op in clay_ops.menu(mode)}, f"{name}: leaked into {mode}"
    op = clay_ops.get(name)
    assert not op.enabled(doc)
    assert clay_ops.reason_for(op, doc)


def test_pack_uv_forwards_its_params_and_keeps_the_generator_as_one_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spies on the kernel call directly (``test_clay_model_ops.py``'s own
    pattern for a translated/forwarded param), which also proves the row
    reaches ``uvtools.pack_islands`` and not some other packer."""
    seen: list[tuple[float, bool]] = []
    real = uvtools.pack_islands

    def spy(mesh: object, *, margin: float = 0.005, rotate: bool = False) -> object:
        seen.append((margin, rotate))
        return real(mesh, margin=margin, rotate=rotate)

    monkeypatch.setattr(uvtools, "pack_islands", spy)

    ctx = _Ctx()
    doc, uid = _unwrapped_doc()
    doc.select([uid])
    depth = len(doc.history)

    assert clay_ops.run(ctx, doc, clay_ops.get("pack-uv"), margin=0.02, rotate=1.0)

    assert len(doc.history) == depth + 1
    assert seen == [(0.02, True)]
    assert doc.by_uid(uid).generator == "box", "a uv edit is not geometry"
    assert not ctx.toasts.errors


def test_pack_uv_refuses_an_object_with_no_uv_yet() -> None:
    ctx = _Ctx()
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Quad", mesh=_quad_no_uv()))
    doc.select([obj.uid])
    depth = len(doc.history)

    assert clay_ops.run(ctx, doc, clay_ops.get("pack-uv")) is False

    assert len(doc.history) == depth
    assert ctx.toasts.errors and "texture coordinates" in ctx.toasts.errors[0]


# --- the agent's derived enum -------------------------------------------------


def test_the_agent_clay_op_enum_picks_up_every_new_row_with_no_edit_there() -> None:
    """``clay_op``'s enum is built straight off ``clay_ops.OPS``
    (``agent/dispatch.py``'s own bidirectional derivation gate -- see
    ``test_agent_clay.py``'s identical claim, and ``test_clay_model_ops.py``'s
    tranche 5 copy of it). This is that same promise for Pack UV's row specifically.
    """
    from realmspinner.studio.modes.clay.agent import dispatch as agent_clay

    tools = {t.name: t for t in agent_clay.tools()}
    enum = set(tools["clay_op"].schema["properties"]["name"]["enum"])
    missing = [name for name in NEW_ROWS if name not in enum]
    assert not missing, f"clay_op's enum is missing {missing}"
