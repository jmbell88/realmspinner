"""Tranche 5's OPS registry row that Clay kept (triangulate) plus the six
"game" primitives' icons and category.

What belongs here is the registry wiring: that the context menu, the tools pane
and the keyboard reach Triangulate through the one list ``ops.py``'s own module
docstring describes, that it is gated to the face element mode with a reason a
refused user can read, that a run edits geometry as one undo step, that the six
game generators place from the palette carrying a real icon, and that the
agent's derived ``clay_op``/``clay_add_primitive`` enums picked up all of it
with nothing hand-listed.
"""

from __future__ import annotations

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


NEW_GENERATORS = ("wedge", "ramp", "rounded_box", "stairs", "wall", "doorway")


# --- registry wiring: menu membership and greyed reasons ---------------------


def test_triangulate_is_registered_exactly_once() -> None:
    names = [op.name for op in clay_ops.OPS]
    assert names.count("triangulate") == 1


def test_triangulate_is_face_only_and_greyed_with_nothing_selected() -> None:
    doc, _uid = _doc()
    assert "triangulate" in {op.name for op in clay_ops.menu("face")}
    for mode in ("object", "vertex", "edge"):
        assert "triangulate" not in {op.name for op in clay_ops.menu(mode)}, (
            f"triangulate: leaked into {mode}"
        )
    doc.set_element_mode("face")
    op = clay_ops.get("triangulate")
    assert not op.enabled(doc)
    assert clay_ops.reason_for(op, doc)


def test_triangulate_is_reachable_by_t() -> None:
    """A param-less immediate action on a bare key that is free
    (``test_clay_ops.py``'s own ``test_no_two_ops_in_one_mode_claim_the_same_key``
    is the standing gate against a future collision)."""
    assert clay_ops.by_key("face", "T") is clay_ops.get("triangulate")


# --- representative run: one undo step, sensible geometry ---------------------


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
    ``clay_ops.OPS``/``primitives.CLAY_GENERATORS`` (``agent/dispatch.py``'s
    own "bidirectional derivation gate" -- see ``test_agent_clay.py``'s
    identical claim). Named for this tranche's Triangulate row and six game
    shapes specifically, so a hand-listed enum that happened to already
    include today's set would still be caught the next time either registry
    grows.
    """
    from realmspinner.studio.modes.clay.agent import dispatch as agent_clay

    tools = {t.name: t for t in agent_clay.tools()}
    op_enum = set(tools["clay_op"].schema["properties"]["name"]["enum"])
    missing_ops = [name for name in ('triangulate',) if name not in op_enum]
    assert not missing_ops, f"clay_op's enum is missing {missing_ops}"

    gen_enum = set(tools["clay_add_primitive"].schema["properties"]["generator"]["enum"])
    missing_gens = [name for name in NEW_GENERATORS if name not in gen_enum]
    assert not missing_gens, f"clay_add_primitive's enum is missing {missing_gens}"
