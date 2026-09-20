"""Tranche 6's OPS registry rows (integration half): mark/clear seam,
unwrap-by-seams, pack-uv and texel-density.

``kernels.mesh.uvtools``/``uvunwrap`` are the kernel half and are tested on
their own terms in ``test_uvtools.py``/``test_uvunwrap.py``; what belongs
here is the registry wiring ``dev/CLAY-PLAN.md`` tranche 6 hands to this
file -- ``ops.py``'s own module docstring: that the context menu, the tools
pane and the keyboard all reach every new row through the one ``OPS`` list,
that each is gated to the right mode with a reason a refused user can read,
that mark/clear seam round-trip ``Obj.seams`` as one step, that Unwrap
(Seams) refuses a closed surface with the kernel's own sentence and succeeds
once a seam opens it, that Pack UV Islands and Normalise Texel Density both
edit uv as one step and keep the generator (a uv is not geometry, the same
rule Box Unwrap already states), and that the agent's derived ``clay_op``
enum picked up all five rows with nothing hand-listed there.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.kernels.mesh import adjacency as adj
from warlock.kernels.mesh import document as bd
from warlock.kernels.mesh import elements as el
from warlock.kernels.mesh import mesh as bm
from warlock.kernels.mesh import primitives as bp
from warlock.kernels.mesh import uv as uv_mod
from warlock.kernels.mesh import uvtools
from warlock.studio.modes.clay import ops as clay_ops


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


def _open_tube(segments: int = 12, radius: float = 0.5, height: float = 2.0) -> bm.Mesh:
    """A prism's lateral surface only -- no top or bottom cap -- two rings of
    *segments* vertices joined by a band of quads. A single seam along one
    vertical edge (vertex *i* and vertex *i + segments* are the same angular
    column, bottom and top) is what :func:`test_unwrap_lscm_of_a_cylinder_
    with_one_seam...` in ``test_uvunwrap.py`` already proves flattens
    cleanly; reused here at the OPS layer to prove the row reaches the same
    kernel call with the object's own ``seams``.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    x, z = radius * np.cos(theta), radius * np.sin(theta)
    bottom = np.stack([x, np.full(segments, -height / 2), z], axis=1)
    top = np.stack([x, np.full(segments, height / 2), z], axis=1)
    positions = np.vstack([bottom, top])
    faces = [
        [i, segments + i, segments + (i + 1) % segments, (i + 1) % segments]
        for i in range(segments)
    ]
    return bm.from_faces(positions, faces)


NEW_EDGE_ROWS = ("mark-seam", "clear-seam")
NEW_OBJECT_ROWS = ("unwrap-seams", "pack-uv", "texel-density")
NEW_ROWS = NEW_EDGE_ROWS + NEW_OBJECT_ROWS


# --- registry wiring: menu membership and greyed reasons ---------------------


def test_every_new_row_is_registered_exactly_once() -> None:
    names = [op.name for op in clay_ops.OPS]
    for name in NEW_ROWS:
        assert names.count(name) == 1, name


@pytest.mark.parametrize("name", NEW_EDGE_ROWS)
def test_new_edge_rows_are_edge_only_and_greyed_with_nothing_selected(name: str) -> None:
    doc, uid = _doc()
    assert name in {op.name for op in clay_ops.menu("edge")}
    for mode in ("object", "vertex", "face"):
        assert name not in {op.name for op in clay_ops.menu(mode)}, f"{name}: leaked into {mode}"
    doc.set_element_mode("edge")
    op = clay_ops.get(name)
    assert not op.enabled(doc)
    assert clay_ops.reason_for(op, doc)
    del uid


@pytest.mark.parametrize("name", NEW_OBJECT_ROWS)
def test_new_object_rows_are_object_only_and_greyed_with_nothing_selected(name: str) -> None:
    doc = bd.ClayDoc()
    assert name in {op.name for op in clay_ops.menu("object")}
    for mode in ("vertex", "edge", "face"):
        assert name not in {op.name for op in clay_ops.menu(mode)}, f"{name}: leaked into {mode}"
    op = clay_ops.get(name)
    assert not op.enabled(doc)
    assert clay_ops.reason_for(op, doc)


# --- mark-seam / clear-seam: round trip as one step ---------------------------


def test_mark_seam_adds_the_selected_edges_to_the_seam_set_as_one_step() -> None:
    ctx = _Ctx()
    doc, uid = _doc()
    a = adj.adjacency(doc.by_uid(uid).mesh)
    doc.set_element_mode("edge")
    doc.set_element_sel(uid, el.ElementSel(edges=np.array([a.edge_verts[0], a.edge_verts[1]])))
    depth = len(doc.history)

    assert clay_ops.run(ctx, doc, clay_ops.get("mark-seam"))

    assert len(doc.history) == depth + 1
    wanted = {tuple(sorted(int(v) for v in a.edge_verts[i])) for i in (0, 1)}
    assert set(doc.by_uid(uid).seams) == wanted
    assert not ctx.toasts.errors


def test_clear_seam_round_trips_mark_seam_as_one_step_each_way() -> None:
    ctx = _Ctx()
    doc, uid = _doc()
    a = adj.adjacency(doc.by_uid(uid).mesh)
    doc.set_element_mode("edge")
    doc.set_element_sel(uid, el.ElementSel(edges=np.array([a.edge_verts[0]])))
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("mark-seam"))
    assert doc.by_uid(uid).seams
    depth = len(doc.history)

    assert clay_ops.run(ctx, doc, clay_ops.get("clear-seam"))

    assert len(doc.history) == depth + 1
    assert doc.by_uid(uid).seams == ()
    assert not ctx.toasts.errors


def test_mark_seam_refuses_with_no_edge_selection() -> None:
    """Edge mode with the object itself selected but nothing picked in it:
    ``doc.selection`` drops the object the moment its element selection goes
    empty (the document's own invariant), so this is really the "nothing
    selected" refusal, exercised through the seam row specifically."""
    ctx = _Ctx()
    doc, _uid = _doc()
    doc.set_element_mode("edge")
    depth = len(doc.history)

    assert clay_ops.run(ctx, doc, clay_ops.get("mark-seam")) is False

    assert len(doc.history) == depth


# --- unwrap-seams: refuses closed, succeeds once a seam opens it -------------


def test_unwrap_seams_refuses_a_closed_object_with_no_seams() -> None:
    ctx = _Ctx()
    doc, uid = _doc()  # a box: watertight, no free boundary anywhere
    doc.select([uid])
    depth = len(doc.history)

    assert clay_ops.run(ctx, doc, clay_ops.get("unwrap-seams")) is False

    assert len(doc.history) == depth, "a refusal pushes nothing"
    assert ctx.toasts.errors and "closed surface" in ctx.toasts.errors[0]


def test_unwrap_seams_succeeds_with_one_seam_and_keeps_the_generator() -> None:
    ctx = _Ctx()
    doc = bd.ClayDoc()
    obj = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="Tube", mesh=_open_tube(), generator="cylinder")
    )
    doc.set_seams(obj.uid, [(0, 12)])
    doc.select([obj.uid])
    depth = len(doc.history)

    assert clay_ops.run(ctx, doc, clay_ops.get("unwrap-seams"))

    assert len(doc.history) == depth + 1
    assert doc.by_uid(obj.uid).mesh.uv is not None
    assert doc.by_uid(obj.uid).generator == "cylinder", "a uv edit is not geometry"
    assert not ctx.toasts.errors


# --- pack-uv / texel-density: edit uv as one step, keep the generator --------


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


def test_texel_density_normalizes_to_the_requested_reading_as_one_step() -> None:
    ctx = _Ctx()
    doc, uid = _unwrapped_doc()
    doc.select([uid])
    depth = len(doc.history)

    assert clay_ops.run(ctx, doc, clay_ops.get("texel-density"), target=512.0, texture_size=1.0)

    assert len(doc.history) == depth + 1
    # texture_size choice index 1 is CLAY_TEXTURE_SIZES[1] -- read back
    # through the same kernel function the row itself calls.
    from warlock.kernels.rig import blender_spec

    px = blender_spec.CLAY_TEXTURE_SIZES[1]
    assert uvtools.texel_density(doc.by_uid(uid).mesh, texture_px=px) == pytest.approx(512.0)
    assert doc.by_uid(uid).generator == "box", "a uv edit is not geometry"
    assert not ctx.toasts.errors


def test_texel_density_refuses_an_object_with_no_uv_yet() -> None:
    ctx = _Ctx()
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Quad", mesh=_quad_no_uv()))
    doc.select([obj.uid])
    depth = len(doc.history)

    assert clay_ops.run(ctx, doc, clay_ops.get("texel-density")) is False

    assert len(doc.history) == depth
    assert ctx.toasts.errors and "texture coordinates" in ctx.toasts.errors[0]


# --- the agent's derived enum -------------------------------------------------


def test_the_agent_clay_op_enum_picks_up_every_new_row_with_no_edit_there() -> None:
    """``clay_op``'s enum is built straight off ``clay_ops.OPS``
    (``agent/dispatch.py``'s own bidirectional derivation gate -- see
    ``test_agent_clay.py``'s identical claim, and ``test_clay_model_ops.py``'s
    tranche 5 copy of it). This is that same promise for tranche 6's five new
    rows specifically.
    """
    from warlock.studio.modes.clay.agent import dispatch as agent_clay

    tools = {t.name: t for t in agent_clay.tools()}
    enum = set(tools["clay_op"].schema["properties"]["name"]["enum"])
    missing = [name for name in NEW_ROWS if name not in enum]
    assert not missing, f"clay_op's enum is missing {missing}"
