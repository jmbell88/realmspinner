"""Tranche 5's OPS registry rows: bisect, knife, edge/vertex slide, rip,
poke, triangulate, tris-to-quads, symmetrize, grid fill, spin and screw --
plus the six new "game" primitives' icons and category.

``kernels.mesh.ops_model``/``ops_spin`` are the kernel half and are tested on
their own terms elsewhere; what belongs here is the registry wiring this
tranche's integration half owns (``dev/CLAY-PLAN.md``): that the context
menu, the tools pane and (where a bare key is genuinely free) the keyboard
all reach every new row through the one list ``ops.py``'s own module
docstring describes, that each is gated to the right element mode with a
reason a refused user can read, that a representative run of each actually
edits geometry as one undo step and leaves a sensible selection, that a
refusal is a toast with no edit rather than a half-built mesh or a bare
``TypeError``, that the six new generators place from the palette carrying a
real icon, and that the agent's derived ``clay_op``/``clay_add_primitive``
enums picked up all of it with nothing hand-listed.
"""

from __future__ import annotations

import numpy as np
import pytest

from realmspinner.kernels.mesh import adjacency as adj
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio.modes.clay import ops as clay_ops


class _Toasts:
    """Only what ``Ctx`` really offers -- see ``test_clay_ops.py``'s
    identical double for the incident a ``ctx.toasts.error`` method would
    silently repeat."""

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


def _plane_doc() -> tuple[bd.ClayDoc, int]:
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Plane", mesh=bp.plane(), generator="plane"))
    return doc, obj.uid


def _single_triangle() -> bm.Mesh:
    """One isolated triangular face -- every one of its three edges is a
    boundary edge shared by nothing, which is exactly what Rip's and Grid
    Fill's own refusals need: an odd-length loop with nothing else on the
    mesh to confuse either one."""
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype="f4")
    mesh = bm.Mesh(
        positions=positions,
        loops=np.array([0, 1, 2], dtype="i4"),
        starts=np.array([0, 3], dtype="i4"),
        material=np.zeros(1, dtype="i4"),
        smooth=np.zeros(1, dtype=bool),
    )
    bm.validate(mesh)
    return mesh


def _two_quad_strip() -> bm.Mesh:
    """Two quads side by side sharing exactly one interior edge (1-4) -- the
    smallest mesh Rip can actually split. At each of that edge's two
    endpoints the two faces meet only there, so ripping it separates them
    into two shells joined at no point (unlike a single edge picked out of a
    closed, all-quad mesh like a box, where every vertex's other edges still
    keep its whole fan connected after just one edge is removed)."""
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
        ],
        dtype="f4",
    )
    mesh = bm.Mesh(
        positions=positions,
        loops=np.array([0, 1, 4, 3, 1, 2, 5, 4], dtype="i4"),
        starts=np.array([0, 4, 8], dtype="i4"),
        material=np.zeros(2, dtype="i4"),
        smooth=np.zeros(2, dtype=bool),
    )
    bm.validate(mesh)
    return mesh


def _open_box() -> bm.Mesh:
    """A box with its last face dropped -- five closed faces plus a
    4-vertex boundary hole for Grid Fill to close back up."""
    box = bp.box()
    mesh = bm.Mesh(
        positions=box.positions,
        loops=box.loops[:20],
        starts=box.starts[:6],
        material=box.material[:5],
        smooth=box.smooth[:5],
    )
    bm.validate(mesh)
    return mesh


NEW_FACE_ROWS = ("bisect", "knife", "poke", "triangulate", "tris-to-quads")
# ``grid-fill`` sits here, not in the face group its name suggests --
# ``ops_model.grid_fill`` reads ``sel.edges``, the selected boundary loop,
# exactly as Fill Hole does, so edge mode is what actually reaches it.
NEW_EDGE_ROWS = ("edge-slide", "rip", "grid-fill", "spin", "screw")
NEW_VERTEX_ROWS = ("vertex-slide",)
NEW_OBJECT_ROWS = ("symmetrize",)
NEW_ROWS = NEW_FACE_ROWS + NEW_EDGE_ROWS + NEW_VERTEX_ROWS + NEW_OBJECT_ROWS

NEW_GENERATORS = ("wedge", "ramp", "rounded_box", "stairs", "wall", "doorway")


# --- registry wiring: menu membership and greyed reasons ---------------------


def test_every_new_row_is_registered_exactly_once() -> None:
    names = [op.name for op in clay_ops.OPS]
    for name in NEW_ROWS:
        assert names.count(name) == 1, name


@pytest.mark.parametrize("name", NEW_FACE_ROWS)
def test_new_face_rows_are_face_only_and_greyed_with_nothing_selected(name: str) -> None:
    doc, _uid = _doc()
    assert name in {op.name for op in clay_ops.menu("face")}
    for mode in ("object", "vertex", "edge"):
        assert name not in {op.name for op in clay_ops.menu(mode)}, f"{name}: leaked into {mode}"
    doc.set_element_mode("face")
    op = clay_ops.get(name)
    assert not op.enabled(doc)
    assert clay_ops.reason_for(op, doc)


@pytest.mark.parametrize("name", NEW_EDGE_ROWS)
def test_new_edge_rows_are_edge_only_and_greyed_with_nothing_selected(name: str) -> None:
    doc, _uid = _doc()
    assert name in {op.name for op in clay_ops.menu("edge")}
    for mode in ("object", "vertex", "face"):
        assert name not in {op.name for op in clay_ops.menu(mode)}, f"{name}: leaked into {mode}"
    doc.set_element_mode("edge")
    op = clay_ops.get(name)
    assert not op.enabled(doc)
    assert clay_ops.reason_for(op, doc)


def test_vertex_slide_is_vertex_only_and_greyed_with_nothing_selected() -> None:
    doc, _uid = _doc()
    assert "vertex-slide" in {op.name for op in clay_ops.menu("vertex")}
    for mode in ("object", "edge", "face"):
        assert "vertex-slide" not in {op.name for op in clay_ops.menu(mode)}
    doc.set_element_mode("vertex")
    op = clay_ops.get("vertex-slide")
    assert not op.enabled(doc)
    assert clay_ops.reason_for(op, doc)


def test_symmetrize_is_object_only_and_ignores_the_element_selection_gate() -> None:
    """The table's own words: "symmetrize is object-mode and ignores the
    selection like Smooth does" -- gated on ``has_objects``, not on an
    element pick, so selecting the object (not an element) is what enables
    it."""
    doc = bd.ClayDoc()
    assert "symmetrize" in {op.name for op in clay_ops.menu("object")}
    for mode in ("vertex", "edge", "face"):
        assert "symmetrize" not in {op.name for op in clay_ops.menu(mode)}
    op = clay_ops.get("symmetrize")
    assert not op.enabled(doc)
    assert clay_ops.reason_for(op, doc)
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
    doc.select([obj.uid])
    assert op.enabled(doc), "an object selected is enough -- no element pick needed"


def test_rip_is_reachable_by_v_and_triangulate_by_t() -> None:
    """The two bare keys this tranche adds -- both param-less immediate
    actions, the same shape Extrude/Separate Selection/Delete already use for
    their own keys, and both free (checked against every existing row before
    landing; ``test_clay_ops.py``'s own
    ``test_no_two_ops_in_one_mode_claim_the_same_key`` is the standing gate
    against a future collision)."""
    assert clay_ops.by_key("edge", "V") is clay_ops.get("rip")
    assert clay_ops.by_key("face", "T") is clay_ops.get("triangulate")


# --- representative runs: one undo step, sensible geometry --------------------


def test_bisect_runs_as_one_step_and_splits_the_crossed_faces() -> None:
    ctx = _Ctx()
    doc, uid = _doc()
    doc.set_element_mode("face")
    doc.set_element_sel(uid, el.ElementSel(faces=list(range(6))))
    depth = len(doc.history)

    assert clay_ops.run(ctx, doc, clay_ops.get("bisect"), axis=0.0, clear=0.0, fill=0.0)

    assert len(doc.history) == depth + 1
    assert bm.face_count(doc.by_uid(uid).mesh) == 10, "the 4 faces crossing X=0 each split in two"
    assert doc.by_uid(uid).generator is None, "a geometry edit freezes the generator"
    assert not ctx.toasts.errors


def test_knife_runs_with_a_supplied_plane_and_refuses_with_none() -> None:
    """Firing the row with no plane -- exactly what a bare menu click or a
    tools-pane press does today, before the viewport's own click-drag gesture
    is wired in ``ui/_view_drag.py`` -- refuses with a toast rather than
    crashing on the kernel's required keywords."""
    ctx = _Ctx()
    doc, uid = _doc()
    doc.set_element_mode("face")
    doc.set_element_sel(uid, el.ElementSel(faces=list(range(6))))
    depth = len(doc.history)

    assert clay_ops.run(
        ctx, doc, clay_ops.get("knife"), point=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0)
    )
    assert len(doc.history) == depth + 1
    assert bm.face_count(doc.by_uid(uid).mesh) == 10
    assert not ctx.toasts.errors

    before = doc.by_uid(uid).mesh
    ctx2 = _Ctx()
    depth2 = len(doc.history)

    assert clay_ops.run(ctx2, doc, clay_ops.get("knife")) is False

    assert len(doc.history) == depth2, "a refusal pushes nothing"
    assert doc.by_uid(uid).mesh is before, "and touches no mesh"
    assert ctx2.toasts.errors and "knife cut" in ctx2.toasts.errors[0]


def test_edge_slide_moves_the_loops_vertices_along_its_rails() -> None:
    ctx = _Ctx()
    doc, uid = _doc()
    a = adj.adjacency(doc.by_uid(uid).mesh)
    doc.set_element_mode("edge")
    doc.set_element_sel(uid, el.ElementSel(edges=np.array([a.edge_verts[0]])))
    before = np.array(doc.by_uid(uid).mesh.positions, copy=True)
    depth = len(doc.history)

    assert clay_ops.run(ctx, doc, clay_ops.get("edge-slide"), t=0.5)

    assert len(doc.history) == depth + 1
    assert not np.allclose(before, doc.by_uid(uid).mesh.positions)
    assert bm.face_count(doc.by_uid(uid).mesh) == 6, "sliding moves vertices, not topology"
    assert not ctx.toasts.errors


def test_vertex_slide_moves_the_selected_vertex() -> None:
    ctx = _Ctx()
    doc, uid = _doc()
    doc.set_element_mode("vertex")
    doc.set_element_sel(uid, el.ElementSel(verts=np.array([0])))
    before = np.array(doc.by_uid(uid).mesh.positions, copy=True)
    depth = len(doc.history)

    assert clay_ops.run(ctx, doc, clay_ops.get("vertex-slide"), t=1.0)

    assert len(doc.history) == depth + 1
    assert not np.allclose(before, doc.by_uid(uid).mesh.positions)
    assert not ctx.toasts.errors


def test_rip_splits_two_faces_that_share_only_one_edge() -> None:
    ctx = _Ctx()
    doc = bd.ClayDoc()
    strip = _two_quad_strip()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Strip", mesh=strip))
    a = adj.adjacency(strip)
    shared = next(pair for pair in a.edge_verts.tolist() if 1 in pair and 4 in pair)
    doc.set_element_mode("edge")
    doc.set_element_sel(obj.uid, el.ElementSel(edges=np.array([shared])))
    depth = len(doc.history)

    assert clay_ops.run(ctx, doc, clay_ops.get("rip"))

    assert len(doc.history) == depth + 1
    assert len(doc.by_uid(obj.uid).mesh.positions) == 8, "both shared vertices split"
    assert not ctx.toasts.errors


def test_rip_refuses_a_boundary_edge_with_no_edit() -> None:
    ctx = _Ctx()
    doc = bd.ClayDoc()
    tri = _single_triangle()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Tri", mesh=tri))
    a = adj.adjacency(tri)
    doc.set_element_mode("edge")
    doc.set_element_sel(obj.uid, el.ElementSel(edges=np.array([a.edge_verts[0]])))
    depth = len(doc.history)

    assert clay_ops.run(ctx, doc, clay_ops.get("rip")) is False

    assert len(doc.history) == depth, "a refusal pushes nothing"
    assert doc.by_uid(obj.uid).mesh is tri
    assert ctx.toasts.errors and "boundary" in ctx.toasts.errors[0]


def test_poke_fans_the_selected_face_around_a_new_centre() -> None:
    ctx = _Ctx()
    doc, uid = _doc()
    doc.set_element_mode("face")
    doc.set_element_sel(uid, el.ElementSel(faces=[0]))
    depth = len(doc.history)

    assert clay_ops.run(ctx, doc, clay_ops.get("poke"), offset=0.1)

    assert len(doc.history) == depth + 1
    assert bm.face_count(doc.by_uid(uid).mesh) == 9, "one quad fans into four triangles"
    assert not ctx.toasts.errors


def test_triangulate_replaces_every_face_with_triangles() -> None:
    ctx = _Ctx()
    doc, uid = _doc()
    doc.set_element_mode("face")
    doc.set_element_sel(uid, el.ElementSel(faces=list(range(6))))
    depth = len(doc.history)

    assert clay_ops.run(ctx, doc, clay_ops.get("triangulate"))

    assert len(doc.history) == depth + 1
    assert bm.face_count(doc.by_uid(uid).mesh) == 12, "6 quads become 12 triangles"
    assert not ctx.toasts.errors


def test_tris_to_quads_rejoins_what_triangulate_just_split() -> None:
    ctx = _Ctx()
    doc, uid = _doc()
    doc.set_element_mode("face")
    doc.set_element_sel(uid, el.ElementSel(faces=list(range(6))))
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("triangulate"))
    tri_count = bm.face_count(doc.by_uid(uid).mesh)
    doc.set_element_sel(uid, el.ElementSel(faces=list(range(tri_count))))
    depth = len(doc.history)

    assert clay_ops.run(ctx, doc, clay_ops.get("tris-to-quads"), max_angle=40.0)

    assert len(doc.history) == depth + 1
    assert bm.face_count(doc.by_uid(uid).mesh) == 6, "the 12 triangles rejoin into 6 quads"
    assert not ctx.toasts.errors


def test_grid_fill_closes_a_four_vertex_hole() -> None:
    ctx = _Ctx()
    doc = bd.ClayDoc()
    openbox = _open_box()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="OpenBox", mesh=openbox))
    boundary = adj.check_manifold(openbox).boundary_edges
    assert len(boundary) == 4
    doc.set_element_mode("edge")
    doc.set_element_sel(obj.uid, el.ElementSel(edges=boundary))
    depth = len(doc.history)

    assert clay_ops.run(ctx, doc, clay_ops.get("grid-fill"), span=1.0)

    assert len(doc.history) == depth + 1
    assert bm.face_count(doc.by_uid(obj.uid).mesh) == 6
    assert adj.check_manifold(doc.by_uid(obj.uid).mesh).clean, "the hole is closed"
    assert not ctx.toasts.errors


def test_grid_fill_refuses_an_odd_boundary_with_no_edit() -> None:
    ctx = _Ctx()
    doc = bd.ClayDoc()
    tri = _single_triangle()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Tri", mesh=tri))
    a = adj.adjacency(tri)
    doc.set_element_mode("edge")
    doc.set_element_sel(obj.uid, el.ElementSel(edges=a.edge_verts))
    depth = len(doc.history)

    assert clay_ops.run(ctx, doc, clay_ops.get("grid-fill"), span=1.0) is False

    assert len(doc.history) == depth
    assert doc.by_uid(obj.uid).mesh is tri
    assert ctx.toasts.errors and "even number" in ctx.toasts.errors[0]


def test_spin_lathes_the_plane_boundary_into_new_bands() -> None:
    ctx = _Ctx()
    doc, uid = _plane_doc()
    a = adj.adjacency(doc.by_uid(uid).mesh)
    doc.set_element_mode("edge")
    doc.set_element_sel(uid, el.ElementSel(edges=a.edge_verts))
    depth = len(doc.history)

    assert clay_ops.run(ctx, doc, clay_ops.get("spin"), axis=1.0, angle=360.0, steps=8.0)

    assert len(doc.history) == depth + 1
    assert bm.face_count(doc.by_uid(uid).mesh) == 33, "the plane's own face, plus 8 bands of 4"
    assert not ctx.toasts.errors


def test_screw_lifts_each_band_along_the_axis() -> None:
    ctx = _Ctx()
    doc, uid = _plane_doc()
    a = adj.adjacency(doc.by_uid(uid).mesh)
    doc.set_element_mode("edge")
    doc.set_element_sel(uid, el.ElementSel(edges=a.edge_verts))
    depth = len(doc.history)

    assert clay_ops.run(
        ctx, doc, clay_ops.get("screw"), axis=1.0, angle=360.0, steps=8.0, height=2.0
    )

    assert len(doc.history) == depth + 1
    assert bm.face_count(doc.by_uid(uid).mesh) == 33
    top_y = float(doc.by_uid(uid).mesh.positions[:, 1].max())
    assert top_y > 1.5, "the last ring lifted by close to the full height, unlike Spin's"
    assert not ctx.toasts.errors


def test_symmetrize_deletes_a_side_and_mirrors_it_back() -> None:
    ctx = _Ctx()
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box(), generator="box"))
    doc.select([obj.uid])
    depth = len(doc.history)

    assert clay_ops.run(ctx, doc, clay_ops.get("symmetrize"), axis=0.0, direction=1.0)

    assert len(doc.history) == depth + 1
    assert bm.face_count(doc.by_uid(obj.uid).mesh) == 10
    assert doc.by_uid(obj.uid).generator is None, "a geometry edit freezes the generator"
    assert not ctx.toasts.errors


def test_symmetrize_direction_choice_translates_to_a_signed_kernel_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The risk :func:`clay_ops._symmetrize`'s own docstring names: a choice
    ``Param``'s stored value is always ``0`` or ``1``, never negative, so
    forwarding it unchanged the way ``clear`` is forwarded would make *both*
    choices read as "the positive side" against ``ops_model.symmetrize``'s
    own ``direction >= 0`` test. Spies on the kernel call directly, rather
    than inferring the sign from the geometry a symmetric box would produce
    either way, to prove the two choices really do translate to different
    signs.
    """
    from realmspinner.kernels.mesh import ops_model

    seen: list[float] = []
    real = ops_model.symmetrize

    def spy(mesh: object, sel: object, *, axis: int, direction: float) -> object:
        seen.append(direction)
        return real(mesh, sel, axis=axis, direction=direction)

    monkeypatch.setattr(ops_model, "symmetrize", spy)

    for choice in (0.0, 1.0):
        doc = bd.ClayDoc()
        obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
        doc.select([obj.uid])
        assert clay_ops.run(_Ctx(), doc, clay_ops.get("symmetrize"), axis=0.0, direction=choice)

    assert seen == [-1.0, 1.0], "the two choices must reach the kernel with opposite signs"


# --- the six new generators: category, icon, placement -----------------------


def test_the_six_game_generators_are_filed_in_their_own_category_in_order() -> None:
    categories = dict(bp.CATEGORIES)
    assert categories["game"] == NEW_GENERATORS


def test_the_six_game_generators_each_carry_a_real_icon() -> None:
    from realmspinner.studio import tool_palette

    for name in NEW_GENERATORS:
        assert name in tool_palette.PRIMITIVE_ICONS, name
        assert tool_palette.PRIMITIVE_ICONS[name], f"{name}: an empty icon string"


def test_the_six_game_generators_place_from_the_palette_with_the_right_tag() -> None:
    """``ui/panes/tools.py``'s ``add_primitive`` is the one door the tools
    pane and the agent's ``clay_add_primitive`` handler both call through --
    see that function's own docstring -- so placing through it here proves
    both surfaces reach these six with no further wiring."""
    from realmspinner.studio.modes.clay.ui.panes import tools as clay_tools

    doc = bd.ClayDoc()
    ctx = _Ctx()
    for name in NEW_GENERATORS:
        obj = clay_tools.add_primitive(ctx, doc, name)
        assert obj.generator == name
        assert bm.face_count(obj.mesh) > 0
        assert doc.selection == {obj.uid}


# --- the agent's derived enums -----------------------------------------------


def test_the_agent_op_and_generator_enums_pick_up_every_new_row_with_no_edit_there() -> None:
    """``clay_op``'s and ``clay_add_primitive``'s enums are built straight off
    ``clay_ops.OPS``/``primitives.GENERATORS`` (``agent/dispatch.py``'s own
    "bidirectional derivation gate" -- see ``test_agent_clay.py``'s identical
    claim, pinned there for the registries' current total). This is that same
    promise, named for tranche 5's dozen new rows and six new shapes
    specifically, so a hand-listed enum that happened to already include
    today's set would still be caught the next time either registry grows.
    """
    from realmspinner.studio.modes.clay.agent import dispatch as agent_clay

    tools = {t.name: t for t in agent_clay.tools()}
    op_enum = set(tools["clay_op"].schema["properties"]["name"]["enum"])
    missing_ops = [name for name in NEW_ROWS if name not in op_enum]
    assert not missing_ops, f"clay_op's enum is missing {missing_ops}"

    gen_enum = set(tools["clay_add_primitive"].schema["properties"]["generator"]["enum"])
    missing_gens = [name for name in NEW_GENERATORS if name not in gen_enum]
    assert not missing_gens, f"clay_add_primitive's enum is missing {missing_gens}"
