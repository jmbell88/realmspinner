"""Painting a palette entry onto faces (Phase 3 of the picoCAD texturing plan).

Three doors reach ``ClayDoc.paint_faces``, and each is pinned here:

* the ``assign-material`` op (face mode, one parameter: the palette slot);
* the Material tab's swatch row -- a click paints the selected faces in face
  mode and repaints the object in object mode, exactly as the combo it
  replaced did -- and the texture block below it (Add texture, Clear, Edit in
  Inker, Take back);
* ``clay_material``'s ``faces`` argument, for an agent.

The pane is driven headless the way ``test_material_editor.py`` does it: a
real imgui frame with the one control that would take a press monkeypatched to
report one.
"""

from __future__ import annotations

import inspect
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
from _ui_context import imgui_context

from modes.clay.test_agent_clay import _Ctx as _AgentCtx
from modes.clay.test_agent_clay import _new_agent_tab, _payload
from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio.modes.clay import menutree, texture_link
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay import ops as clay_ops
from realmspinner.studio.modes.clay.agent import dispatch as agent_clay
from realmspinner.studio.modes.clay.ui.panes import props as clay_props


class _Ctx:
    """The slice of ``Ctx`` an op and the Material tab read."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.info: list[str] = []

    def toast(self, message: str, level: str = "info") -> None:
        (self.errors if level == "error" else self.info).append(message)

    def busy(self, key: str) -> bool:
        return False

    def submit(self, *args: Any, **kwargs: Any) -> bool:
        return True


class _Tab:
    def __init__(self, doc: bd.ClayDoc) -> None:
        self.uid = "t1"
        self.doc = doc


@pytest.fixture
def ui(monkeypatch):
    with imgui_context(monkeypatch) as imgui:
        yield imgui


def _doc(boxes: int = 1) -> tuple[bd.ClayDoc, list[int]]:
    """A document with *boxes* boxes and a two-entry palette (slot 1 is red)."""
    doc = bd.ClayDoc()
    uids = [
        doc.add_object(
            bd.Obj(
                uid=bd.new_uid(),
                name=f"Box{i}",
                mesh=bp.box(),
                generator="box",
                params={"size": (1.0, 1.0, 1.0)},
            )
        ).uid
        for i in range(boxes)
    ]
    doc.add_material(replace(bd.default_material("red"), base_color_factor=(1.0, 0.0, 0.0, 1.0)))
    return doc, uids


def _select_faces(doc: bd.ClayDoc, uid: int, *faces: int) -> None:
    doc.set_element_mode("face")
    doc.set_element_sel(uid, el.ElementSel(faces=list(faces)))


def _assign() -> clay_ops.Op:
    return clay_ops.get("assign-material")


# --- the op: registry, gating, refusal -----------------------------------------


def test_assign_material_is_a_face_mode_op_in_the_mesh_menu_with_one_integer_param() -> None:
    op = _assign()
    assert op.modes == ("face",)
    assert [p.name for p in op.params] == ["index"]
    assert op.params[0].integer
    assert "assign-material" in menutree.GROUPS["mesh"].ops
    names = [o.name for menu in menutree.resolve("face") for s in menu.sections for o in s.ops]
    assert "assign-material" in names
    for mode in ("vertex", "edge", "object"):
        assert op not in clay_ops.menu(mode), f"face-only, but offered in {mode} mode"


def test_assign_material_is_gated_on_face_mode_and_a_face_selection_with_a_reason() -> None:
    doc, [uid] = _doc()
    op = _assign()
    assert not op.enabled(doc)
    assert clay_ops.reason_for(op, doc) == "Switch to face mode first."

    doc.set_element_mode("face")
    assert not op.enabled(doc)
    assert clay_ops.reason_for(op, doc) == "Select something first."

    _select_faces(doc, uid, 0)
    assert op.enabled(doc)
    assert clay_ops.reason_for(op, doc) == ""


def test_assign_material_paints_only_the_selected_faces_and_leaves_the_default_slot() -> None:
    doc, [uid] = _doc()
    _select_faces(doc, uid, 1, 4)
    ctx = _Ctx()

    assert clay_ops.run(ctx, doc, _assign(), index=1)

    obj = doc.by_uid(uid)
    assert list(obj.mesh.material) == [0, 1, 0, 0, 1, 0]
    assert obj.material == 0, "the object's own default slot is not the faces' business"
    assert obj.generator == "box", "a face's material is not geometry"
    assert ctx.info == ["Painted 2 face(s) with red."]


def test_assign_material_is_one_undo_step_across_every_object_with_a_selection() -> None:
    doc, [first, second] = _doc(boxes=2)
    _select_faces(doc, first, 0)
    doc.set_element_sel(second, el.ElementSel(faces=[2, 3]))
    before = len(doc.history)

    assert clay_ops.run(_Ctx(), doc, _assign(), index=1)

    assert len(doc.history) == before + 1, "two objects painted, one Ctrl+Z"
    assert doc.history.top.label == "Assign Material"
    assert list(doc.by_uid(first).mesh.material) == [1, 0, 0, 0, 0, 0]
    assert list(doc.by_uid(second).mesh.material) == [0, 0, 1, 1, 0, 0]
    assert doc.undo()
    assert not doc.by_uid(first).mesh.material.any()
    assert not doc.by_uid(second).mesh.material.any()


@pytest.mark.parametrize("index", [2, 99])
def test_assign_material_refuses_a_slot_past_the_palette_with_a_toast_and_no_step(
    index: int,
) -> None:
    """The palette's length is the document's, so ``Param`` cannot clamp it:
    the op checks at run time and refuses rather than silently landing on the
    nearest slot."""
    doc, [uid] = _doc()
    _select_faces(doc, uid, 0)
    ctx = _Ctx()
    before = len(doc.history)

    assert clay_ops.run(ctx, doc, _assign(), index=index) is False

    assert len(doc.history) == before
    assert not doc.by_uid(uid).mesh.material.any()
    assert len(ctx.errors) == 1 and f"no palette entry {index}" in ctx.errors[0]


def test_assign_material_to_the_slot_the_faces_already_wear_says_so_and_pushes_nothing() -> None:
    doc, [uid] = _doc()
    _select_faces(doc, uid, 0)
    ctx = _Ctx()
    before = len(doc.history)

    assert clay_ops.run(ctx, doc, _assign(), index=0) is False

    assert len(doc.history) == before
    assert ctx.info and "already use" in ctx.info[0]


def test_the_agent_reaches_assign_material_through_clay_op_and_a_bad_slot_is_a_message() -> None:
    ctx = _AgentCtx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session)
    clay_op_tool = next(t for t in agent_clay.tools() if t.name == "clay_op")
    assert "assign-material" in clay_op_tool.schema["properties"]["name"]["enum"]

    for call, args in (
        ("clay_element_mode", {"mode": "face"}),
        ("clay_select_elements", {"uid": uid, "faces": [0]}),
    ):
        assert agent_clay.call(ctx, session, call, args)["isError"] is False
    result = agent_clay.call(
        ctx, session, "clay_op", {"name": "assign-material", "params": {"index": 7}}
    )
    body = _payload(result)
    assert body["ran"] is False and "no palette entry 7" in body["messages"][0]


# --- the swatch row -----------------------------------------------------------


def _frame(ui, draw) -> None:
    ui.new_frame()
    ui.begin("##host")
    draw()
    ui.end()
    ui.end_frame()


def _press_swatch(monkeypatch: pytest.MonkeyPatch, index: int) -> list[dict]:
    """Make swatch *index* report a click; -> what every swatch was drawn with."""
    drawn: list[dict] = []

    def fake(label, colour, side, **kw):
        drawn.append({"label": label, "colour": colour, **kw})
        return label == f"##matsw{index}"

    monkeypatch.setattr(clay_props, "_swatch", fake)
    return drawn


def _draw_material(ui, doc: bd.ClayDoc, uid: int, ctx: _Ctx) -> None:
    obj = doc.by_uid(uid)
    _frame(ui, lambda: clay_props._material(ctx, _Tab(doc), doc, obj))


def test_the_slot_combo_is_gone_and_there_is_one_swatch_per_palette_entry(
    monkeypatch: pytest.MonkeyPatch, ui
) -> None:
    doc, [uid] = _doc()
    drawn = _press_swatch(monkeypatch, -1)

    _draw_material(ui, doc, uid, _Ctx())

    assert [d["label"] for d in drawn] == ["##matsw0", "##matsw1"]
    assert [d["selected"] for d in drawn] == [True, False], "the object's default slot"
    assert drawn[1]["colour"] == pytest.approx((1.0, 0.0, 0.0, 1.0))
    assert "labeled_combo" not in inspect.getsource(clay_props._material), "no combo any more"


def test_a_textured_slot_is_marked_and_shows_its_first_texel() -> None:
    doc, _ = _doc()
    picture = (2, 2, bytes([10, 200, 30, 255]) * 4)
    doc.set_material(1, replace(doc.materials[1], base_color=picture))
    assert clay_props._swatch_colour(doc.materials[1]) == pytest.approx(
        (10 / 255, 200 / 255, 30 / 255, 1.0)
    )
    assert clay_props._swatch_colour(doc.materials[0]) == pytest.approx(
        tuple(doc.materials[0].base_color_factor)
    )


def test_a_swatch_click_in_object_mode_repaints_the_whole_object_as_the_combo_did(
    monkeypatch: pytest.MonkeyPatch, ui
) -> None:
    doc, [uid] = _doc()
    _press_swatch(monkeypatch, 1)
    before = len(doc.history)

    _draw_material(ui, doc, uid, _Ctx())

    obj = doc.by_uid(uid)
    assert obj.material == 1
    assert (obj.mesh.material == 1).all(), "every face, not only the default slot (clay-17)"
    assert len(doc.history) == before + 1
    assert doc.undo()
    assert doc.by_uid(uid).material == 0 and not doc.by_uid(uid).mesh.material.any()


def test_a_swatch_click_in_face_mode_paints_the_selected_faces_through_the_op(
    monkeypatch: pytest.MonkeyPatch, ui
) -> None:
    doc, [uid] = _doc()
    _select_faces(doc, uid, 2, 3)
    _press_swatch(monkeypatch, 1)
    ctx = _Ctx()
    before = len(doc.history)

    _draw_material(ui, doc, uid, ctx)

    obj = doc.by_uid(uid)
    assert list(obj.mesh.material) == [0, 0, 1, 1, 0, 0]
    assert obj.material == 0, "the default slot is left alone"
    assert len(doc.history) == before + 1
    assert ctx.info == ["Painted 2 face(s) with red."]


def test_a_swatch_click_in_face_mode_with_no_faces_selected_only_picks_the_slot(
    monkeypatch: pytest.MonkeyPatch, ui
) -> None:
    doc, [uid] = _doc()
    doc.set_element_mode("face")
    _press_swatch(monkeypatch, 1)

    _draw_material(ui, doc, uid, _Ctx())

    obj = doc.by_uid(uid)
    assert obj.material == 1, "now the slot the fields below edit"
    assert not obj.mesh.material.any(), "and nothing was repainted behind the user's back"


def test_ctrl_click_in_face_mode_picks_the_slot_without_painting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc, [uid] = _doc()
    _select_faces(doc, uid, 0, 1)
    monkeypatch.setattr(clay_props.imgui, "get_io", lambda: SimpleNamespace(key_ctrl=True))

    clay_props._pick_slot(_Ctx(), doc, doc.by_uid(uid), 1)

    obj = doc.by_uid(uid)
    assert obj.material == 1
    assert not obj.mesh.material.any()


# --- the texture block --------------------------------------------------------


def _press(monkeypatch: pytest.MonkeyPatch, *needles: str) -> None:
    """Make every small button whose label holds one of *needles* report a press."""
    monkeypatch.setattr(
        clay_props.controls,
        "small_button",
        lambda label, **kw: any(n in label for n in needles),
    )


def _draw_textures(ui, ctx: _Ctx, doc: bd.ClayDoc, index: int) -> None:
    tab = _Tab(doc)
    _frame(ui, lambda: clay_props._texture_slots(ctx, tab, doc, index, doc.materials[index]))


@pytest.mark.parametrize("size", [32, 64, 128])
def test_add_texture_makes_a_picture_of_the_chosen_size(
    monkeypatch: pytest.MonkeyPatch, ui, size: int
) -> None:
    doc, _ = _doc()
    monkeypatch.setattr(clay_props, "_TEXTURE_SIZE", size)
    _press(monkeypatch, "Add texture")
    before = len(doc.history)

    _draw_textures(ui, _Ctx(), doc, 1)

    width, height, data = doc.materials[1].base_color
    assert (width, height, len(data)) == (size, size, size * size * 4)
    assert doc.materials[1].nearest is True
    assert len(doc.history) == before + 1, "one click, one undo step"


def test_the_size_combo_sets_the_next_textures_size(monkeypatch: pytest.MonkeyPatch, ui) -> None:
    doc, _ = _doc()
    monkeypatch.setattr(clay_props, "_TEXTURE_SIZE", 64)
    monkeypatch.setattr(clay_props.controls, "combo", lambda *a, **kw: (True, "128"))

    _draw_textures(ui, _Ctx(), doc, 1)

    assert clay_props._TEXTURE_SIZE == 128
    assert doc.materials[1].base_color is None, "choosing a size makes nothing"


def test_add_texture_toasts_what_the_document_refuses(
    monkeypatch: pytest.MonkeyPatch, ui
) -> None:
    doc, _ = _doc()
    monkeypatch.setattr(clay_props, "_TEXTURE_SIZE", 48)  # not one of 32/64/128
    _press(monkeypatch, "Add texture")
    ctx = _Ctx()
    before = len(doc.history)

    _draw_textures(ui, ctx, doc, 1)

    assert len(ctx.errors) == 1 and "texture size must be" in ctx.errors[0]
    assert len(doc.history) == before
    assert doc.materials[1].base_color is None


def _textured_doc() -> tuple[bd.ClayDoc, tuple[int, int, bytes]]:
    doc, _ = _doc()
    doc.add_texture(1, 32)
    return doc, doc.materials[1].base_color


def test_clear_is_one_undo_step_that_drops_nearest_and_the_inker_link(
    monkeypatch: pytest.MonkeyPatch, ui
) -> None:
    doc, picture = _textured_doc()
    assert doc.materials[1].nearest
    unlinked: list[tuple[Any, int]] = []
    monkeypatch.setattr(texture_link, "unlink", lambda tab, index: unlinked.append((tab, index)))
    monkeypatch.setattr(
        clay_props.widgets, "disabled_button", lambda label, enabled, *a, **kw: "texclear" in label
    )
    before = len(doc.history)

    _draw_textures(ui, _Ctx(), doc, 1)

    cleared = doc.materials[1]
    assert cleared.base_color is None and cleared.nearest is False
    assert len(doc.history) == before + 1, "picture and flag in one step, not two"
    assert [index for _tab, index in unlinked] == [1]
    assert doc.undo()
    assert doc.materials[1].base_color == picture and doc.materials[1].nearest is True


def test_edit_in_inker_hands_the_slot_to_texture_link(
    monkeypatch: pytest.MonkeyPatch, ui
) -> None:
    doc, _ = _textured_doc()
    calls: list[tuple[Any, int]] = []
    monkeypatch.setattr(
        texture_link, "edit_in_inker", lambda ctx, tab, index: calls.append((tab, index))
    )
    _press(monkeypatch, "Edit texture in Inker")

    _draw_textures(ui, _Ctx(), doc, 1)

    assert [index for _tab, index in calls] == [1]


def test_take_texture_back_lists_open_inker_documents_and_lands_the_picked_one(
    monkeypatch: pytest.MonkeyPatch, ui
) -> None:
    doc, _ = _textured_doc()
    entry = SimpleNamespace(uid="ink-1", title="barrel.png", doc=object())
    monkeypatch.setattr(texture_link, "open_inker_docs", lambda ctx: [entry])
    taken: list[tuple[int, Any]] = []
    monkeypatch.setattr(
        texture_link, "take_back", lambda ctx, tab, index, inker: taken.append((index, inker))
    )
    _press(monkeypatch, "Take texture back from Inker")
    monkeypatch.setattr(
        clay_props.controls, "menu_item", lambda label, *a, **kw: "barrel.png" in label
    )

    _draw_textures(ui, _Ctx(), doc, 1)

    assert taken == [(1, entry)]


def test_a_texture_less_slot_offers_no_inker_buttons(monkeypatch: pytest.MonkeyPatch, ui) -> None:
    doc, _ = _doc()
    labels: list[str] = []

    def spy(label, **kw):
        labels.append(label)
        return False

    monkeypatch.setattr(clay_props.controls, "small_button", spy)

    _draw_textures(ui, _Ctx(), doc, 1)

    assert any("Add texture" in label for label in labels)
    assert not any("Inker" in label for label in labels)


# --- the agent: clay_material with faces --------------------------------------


def _agent_box() -> tuple[_AgentCtx, agent_clay.Session, int]:
    ctx = _AgentCtx()
    session = agent_clay.Session()
    return ctx, session, _new_agent_tab(ctx, session)


def _agent_doc(ctx: _AgentCtx, session: agent_clay.Session) -> bd.ClayDoc:
    return clay_mode.ensure(ctx).get(session.tab_uid).doc


def _history(doc: bd.ClayDoc) -> int:
    return len(doc.history.history())


def test_clay_material_with_faces_paints_only_those_faces_with_a_new_entry() -> None:
    ctx, session, uid = _agent_box()
    doc = _agent_doc(ctx, session)
    palette = len(doc.materials)
    before = _history(doc)

    result = agent_clay.call(
        ctx, session, "clay_material", {"uids": [uid], "faces": [0, 3], "color": [1.0, 0.0, 0.0]}
    )

    assert result["isError"] is False, result
    body = _payload(result)
    assert body == {"index": palette, "uids": [uid], "faces": 2, "color": [1.0, 0.0, 0.0, 1.0]}
    obj = doc.by_uid(uid)
    assert [int(m) for m in obj.mesh.material] == [palette, 0, 0, palette, 0, 0]
    assert obj.material == 0, "the object's default slot is untouched"
    assert len(doc.materials) == palette + 1
    assert _history(doc) == before + 1, "palette entry and paint are one step"
    assert doc.undo()
    assert len(doc.materials) == palette and not doc.by_uid(uid).mesh.material.any()


def test_clay_material_with_index_reuses_the_slot_and_grows_nothing() -> None:
    ctx, session, uid = _agent_box()
    doc = _agent_doc(ctx, session)
    doc.add_material(replace(bd.default_material("stone"), base_color_factor=(0.5, 0.5, 0.5, 1.0)))
    palette = len(doc.materials)
    before = _history(doc)

    result = agent_clay.call(
        ctx, session, "clay_material", {"uids": [uid], "faces": [5], "index": 1}
    )

    assert result["isError"] is False, result
    body = _payload(result)
    assert body["index"] == 1 and body["faces"] == 1
    assert body["color"] == [0.5, 0.5, 0.5, 1.0]
    assert len(doc.materials) == palette, "an existing slot adds no palette entry"
    assert [int(m) for m in doc.by_uid(uid).mesh.material] == [0, 0, 0, 0, 0, 1]
    assert _history(doc) == before + 1


def test_clay_material_with_index_and_no_faces_repaints_the_whole_object() -> None:
    ctx, session, uid = _agent_box()
    doc = _agent_doc(ctx, session)
    doc.add_material(replace(bd.default_material("stone"), base_color_factor=(0.5, 0.5, 0.5, 1.0)))

    result = agent_clay.call(ctx, session, "clay_material", {"uids": [uid], "index": 1})

    assert result["isError"] is False, result
    assert "faces" not in _payload(result)
    obj = doc.by_uid(uid)
    assert (obj.mesh.material == 1).all() and obj.material == 1


def _refusal(result: dict) -> dict:
    assert result["isError"] is True, result
    return result["structuredContent"]


def test_clay_material_refuses_faces_given_several_uids() -> None:
    ctx, session, uid = _agent_box()
    other = _payload(agent_clay.call(ctx, session, "clay_add_primitive", {"generator": "box"}))[
        "uid"
    ]
    doc = _agent_doc(ctx, session)
    before = _history(doc)

    refusal = _refusal(
        agent_clay.call(
            ctx,
            session,
            "clay_material",
            {"uids": [uid, other], "faces": [0], "color": [1.0, 0.0, 0.0]},
        )
    )

    assert refusal["field"] == "faces" and refusal["recovery"] == "fix_arguments"
    assert _history(doc) == before and len(doc.materials) == 1


@pytest.mark.parametrize(
    ("args", "field"),
    [
        ({"faces": [6], "color": [1.0, 0.0, 0.0]}, "faces"),  # a box has faces 0..5
        ({"faces": [-1], "color": [1.0, 0.0, 0.0]}, "faces"),
        ({"faces": [], "color": [1.0, 0.0, 0.0]}, "faces"),
        ({"faces": [True], "color": [1.0, 0.0, 0.0]}, "faces"),
        ({"faces": [1.5], "color": [1.0, 0.0, 0.0]}, "faces"),
        ({"faces": [0], "index": 9}, "index"),
        ({"faces": [0], "index": -1}, "index"),
        ({"faces": [0], "index": True}, "index"),
        ({"faces": [0], "index": 0, "color": [1.0, 0.0, 0.0]}, "index"),
        ({"faces": [0]}, "color"),
    ],
)
def test_clay_material_refuses_a_bad_face_or_slot_by_naming_the_field(
    args: dict, field: str
) -> None:
    ctx, session, uid = _agent_box()
    doc = _agent_doc(ctx, session)
    before = _history(doc)

    refusal = _refusal(agent_clay.call(ctx, session, "clay_material", {"uids": [uid], **args}))

    assert refusal["field"] == field
    assert _history(doc) == before and len(doc.materials) == 1, "a refusal changes nothing"
    assert not doc.by_uid(uid).mesh.material.any()


def test_clay_material_schema_declares_faces_and_index_and_keeps_closed_properties() -> None:
    tool = next(t for t in agent_clay.tools() if t.name == "clay_material")
    schema = tool.schema
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["uids"], "color is optional once index can stand in for it"
    assert schema["properties"]["faces"] == {
        "type": "array",
        "items": {"type": "integer"},
        "minItems": 1,
    }
    assert schema["properties"]["index"] == {"type": "integer", "minimum": 0}
    assert "faces" in tool.description and "index" in tool.description


def test_a_face_paint_reads_back_through_the_export_grouping() -> None:
    """``to_primitives`` is what exports: the painted faces must come out as
    their own primitive in the new colour, the rest in the old."""
    ctx, session, uid = _agent_box()
    doc = _agent_doc(ctx, session)
    agent_clay.call(
        ctx, session, "clay_material", {"uids": [uid], "faces": [0], "color": [0.0, 1.0, 0.0]}
    )
    prims = bd.to_primitives(doc.by_uid(uid), doc.materials)
    colours = sorted(tuple(p.material.base_color_factor) for p in prims)
    assert len(prims) == 2
    assert (0.0, 1.0, 0.0, 1.0) in colours


def test_paint_faces_is_what_every_door_ends_in() -> None:
    """The op, the swatch and the agent all go through ``ClayDoc.paint_faces``
    (no door rewrites ``mesh.material`` itself), so its undo-by-uid and
    generator-keeping guarantees hold for all three."""
    assert "paint_faces" in inspect.getsource(clay_ops._assign_material)
    handler = inspect.getsource(agent_clay._HANDLERS["clay_material"])
    assert "paint_faces" in handler
    assert "material[" not in handler, "no door writes the per-face array itself"
