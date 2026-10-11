"""The 2026-10-10 audit's Clay pane findings: clay-26, -27, -28, -58, -59 and -60.

Each test's name is the claim, and each one fails against the pane code as it
stood before the fix.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import elements as el
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio.modes.clay import state as clay_state
from realmspinner.studio.modes.clay.ui import _uv_texture
from realmspinner.studio.modes.clay.ui.panes import header as clay_header
from realmspinner.studio.modes.clay.ui.panes import outliner as ol
from realmspinner.studio.modes.clay.ui.panes import props as clay_props
from realmspinner.studio.tasks import Done


def _box(doc: bd.ClayDoc, name: str) -> bd.Obj:
    return doc.add_object(bd.Obj(uid=bd.new_uid(), name=name, mesh=bp.box()))


# --- clay-26 ----------------------------------------------------------------------


class _Tab:
    def __init__(self, uid: str, doc: bd.ClayDoc, *, saving: bool = False) -> None:
        self.uid = uid
        self.doc = doc
        self.saving = saving


class _State:
    def __init__(self, tab: _Tab) -> None:
        self._tab = tab

    def get(self, uid: str) -> _Tab | None:
        return self._tab if uid == self._tab.uid else None


class _Ctx:
    def __init__(self, dragging: bool = False) -> None:
        self.toasts: list[tuple[str, str]] = []
        self.clay_view = SimpleNamespace(dragging=dragging)

    def toast(self, message: str, kind: str = "info") -> None:
        self.toasts.append((message, kind))


def _landing(monkeypatch: pytest.MonkeyPatch, *, saving: bool, dragging: bool = False):
    doc = bd.ClayDoc()
    obj = _box(doc, "A")
    tab = _Tab("bd1", doc, saving=saving)
    monkeypatch.setattr(clay_props.clay_mode, "ensure", lambda ctx: _State(tab))
    ctx = _Ctx(dragging=dragging)
    done = Done(
        key=f"clay-mattex:{tab.uid}:{obj.material}:base_color",
        result={"width": 2, "height": 2, "rgba": bytes(16)},
    )
    return doc, obj, ctx, done


def test_a_texture_picked_from_a_file_does_not_land_while_the_tab_is_saving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc, obj, ctx, done = _landing(monkeypatch, saving=True)
    before = (doc.history.head, doc.materials[obj.material], doc.by_uid(obj.uid).mesh)

    clay_props.on_task_done(ctx, done)

    assert (doc.history.head, doc.materials[obj.material], doc.by_uid(obj.uid).mesh) == before
    assert len(ctx.toasts) == 1, "the pick is dropped, so the user is told to make it again"
    assert "again" in ctx.toasts[0][0] and "save" in ctx.toasts[0][0].lower()


def test_a_texture_picked_from_a_file_does_not_land_under_a_live_gesture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc, obj, ctx, done = _landing(monkeypatch, saving=False, dragging=True)
    before = doc.history.head

    clay_props.on_task_done(ctx, done)

    assert doc.history.head == before
    assert doc.materials[obj.material].base_color is None
    assert len(ctx.toasts) == 1 and "again" in ctx.toasts[0][0]


def test_a_texture_picked_from_a_file_still_lands_when_nothing_is_in_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc, obj, ctx, done = _landing(monkeypatch, saving=False)
    clay_props.on_task_done(ctx, done)
    assert doc.materials[obj.material].base_color == (2, 2, bytes(16))
    assert ctx.toasts == []


# --- clay-27 ----------------------------------------------------------------------


def _grouped_doc() -> tuple[bd.ClayDoc, dict[str, int]]:
    doc = bd.ClayDoc()
    uids = {name: _box(doc, name).uid for name in ("A", "G", "X", "Y", "C")}
    doc.set_parent(uids["X"], uids["G"])
    doc.set_parent(uids["Y"], uids["G"])
    return doc, uids


def test_shift_click_range_skips_rows_inside_a_collapsed_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc, uids = _grouped_doc()
    state = SimpleNamespace(outliner_collapsed={("t", uids["G"])}, outliner_anchor=("t", uids["A"]))
    monkeypatch.setattr(
        ol.imgui, "get_io", lambda: SimpleNamespace(key_shift=True, key_ctrl=False)
    )

    ol._click(state, doc, doc.by_uid(uids["C"]), "t")

    assert doc.selection == {uids["A"], uids["G"], uids["C"]}, (
        "the group's undrawn children X and Y are not between two drawn rows"
    )


def test_shift_click_range_skips_rows_the_name_filter_hides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc, uids = _grouped_doc()
    state = SimpleNamespace(outliner_collapsed=set(), outliner_anchor=("t", uids["A"]))
    monkeypatch.setattr(
        ol.imgui, "get_io", lambda: SimpleNamespace(key_shift=True, key_ctrl=False)
    )
    # Typing "a" or "c" in the filter box draws only A and C (lower-cased needle).
    ol._click(state, doc, doc.by_uid(uids["C"]), "t", needle="c")
    assert doc.selection == {uids["C"]}, "the anchor A is not drawn, so the click is a plain one"

    state.outliner_anchor = ("t", uids["C"])
    rows = ol._visible_rows(state, "t", doc, "c")
    assert [o.name for o, _d, _h in rows] == ["C"]


def test_range_falls_back_to_the_clicked_row_when_the_anchor_is_not_drawn() -> None:
    doc, uids = _grouped_doc()
    state = SimpleNamespace(outliner_collapsed={("t", uids["G"])})
    rows = ol._visible_rows(state, "t", doc, "")
    assert ol._range(doc, uids["X"], uids["C"], rows) == [uids["C"]]
    assert ol._range(doc, uids["A"], uids["C"], rows) == [uids["A"], uids["G"], uids["C"]]


# --- clay-28 ----------------------------------------------------------------------


def _slot_two_doc() -> tuple[bd.ClayDoc, bd.Obj, clay_state.ClayTab]:
    doc = bd.ClayDoc()
    obj = _box(doc, "Box")
    doc.add_material()
    doc.add_material()
    doc.paint_faces(obj.uid, list(range(len(obj.mesh.starts) - 1)), 2)
    doc.add_texture(2, 32)
    tab = clay_state.ClayTab.__new__(clay_state.ClayTab)
    tab.edit_slots = {}
    tab.edit_bases = {}
    return doc, doc.by_uid(obj.uid), tab


def test_uv_pane_draws_the_slot_every_face_wears_when_no_face_is_selected() -> None:
    doc, obj, tab = _slot_two_doc()
    assert obj.material == 0, "the default slot stays 0; only the faces wear slot 2"

    shown = _uv_texture.material_for_pane(doc, obj, tab)

    assert shown is doc.materials[2], "the page, dot and card edit slot 2, so the pane draws it"
    assert _uv_texture.material_for_pane(doc, obj) is doc.materials[2], (
        "and so does a caller with no tab to ask"
    )


def test_uv_pane_draws_the_slot_a_ctrl_click_chose_when_no_face_is_selected() -> None:
    doc, obj, tab = _slot_two_doc()
    doc.add_texture(1, 32)
    tab.choose_edit_slot(doc, obj, 1)
    assert tab.edit_slot(doc, obj) == 1

    assert _uv_texture.material_for_pane(doc, obj, tab) is doc.materials[1]


def test_the_slot_note_says_nothing_about_the_uv_view_when_no_face_is_selected() -> None:
    doc, obj, tab = _slot_two_doc()
    index = tab.edit_slot(doc, obj)
    assert index == 2
    assert clay_props._uv_view_slot(doc, obj, tab) == index
    note = clay_props._edit_slot_note(doc, obj, index, tab)
    assert "slot 2" in note and "UV" not in note


def test_the_slot_note_still_names_the_first_selected_faces_slot() -> None:
    doc, obj, tab = _slot_two_doc()
    doc.paint_faces(obj.uid, [0], 1)
    obj = doc.by_uid(obj.uid)
    doc.set_element_mode("face")
    doc.set_element_sel(obj.uid, el.ElementSel(faces=[0]))
    note = clay_props._edit_slot_note(doc, obj, 2, tab)
    assert "slot 1" in note and "UV" in note


# --- clay-58 ----------------------------------------------------------------------


def _history_tab() -> tuple[bd.ClayDoc, SimpleNamespace]:
    doc = bd.ClayDoc()
    _box(doc, "A")
    _box(doc, "B")
    assert doc.undo()
    return doc, SimpleNamespace(doc=doc, saving=False)


def test_header_undo_redo_grey_with_a_reason_while_a_gesture_is_live() -> None:
    _doc, tab = _history_tab()
    state = clay_state.ClayState()

    idle = {i.key: i for i in clay_header._items(state, tab, False)}
    assert idle["undo"].enabled and idle["redo"].enabled and idle["history"].enabled

    live = {i.key: i for i in clay_header._items(state, tab, True)}
    for key in ("undo", "redo", "history"):
        assert not live[key].enabled, key
        assert live[key].reason == "Finish or cancel the drag first.", key


def test_a_save_still_outranks_a_gesture_in_the_header_reason() -> None:
    _doc, tab = _history_tab()
    tab.saving = True
    live = {i.key: i for i in clay_header._items(clay_state.ClayState(), tab, True)}
    assert "written" in live["undo"].reason


def test_the_header_row_reads_the_live_gesture_from_the_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_row`` hands the items the mode's own ``_gesture_live``, not a constant."""
    _doc, tab = _history_tab()
    seen: list[bool] = []
    real = clay_header._items

    def spy(state: Any, tab: Any, gesture: bool = False) -> Any:
        seen.append(gesture)
        return real(state, tab, gesture)

    monkeypatch.setattr(clay_header, "_items", spy)
    monkeypatch.setattr(clay_header, "_fields", lambda *a, **k: [])
    monkeypatch.setattr(clay_header, "_trailing", lambda *a, **k: None)
    clay_header._row(_Ctx(dragging=True), clay_state.ClayState(), tab, None)
    clay_header._row(_Ctx(dragging=False), clay_state.ClayState(), tab, None)
    assert seen == [True, False]


def test_the_model_pages_history_block_greys_with_a_reason_while_a_gesture_is_live() -> None:
    assert clay_header.history_why(busy=False, gesture=True, idle="") == (
        "Finish or cancel the drag first."
    )
    assert clay_header.history_why(busy=True, gesture=True, idle="") == clay_header.SAVING_WHY
    assert clay_header.history_why(busy=False, gesture=False, idle="Nothing to undo yet.") == (
        "Nothing to undo yet."
    )


# --- clay-59 ----------------------------------------------------------------------


def test_outliner_rename_to_a_blank_name_keeps_the_old_name() -> None:
    for typed in ("", "   ", "\t"):
        commit, finished = ol._rename_outcome(typed, "Box", escaped=False, deactivated=True)
        assert commit is None, repr(typed)
        assert finished, "the field closes; the old name stands"


def test_outliner_rename_still_commits_a_real_name_stripped_of_nothing() -> None:
    assert ol._rename_outcome("Crate", "Box", escaped=False, deactivated=True) == ("Crate", True)


# --- clay-60 ----------------------------------------------------------------------


def test_assigning_an_oversized_png_is_refused_before_it_is_decoded(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    import numpy as np
    from PIL import Image

    from realmspinner.core.safeio import pixelguard
    from realmspinner.service.errors import ServiceError
    from realmspinner.studio import dialogs
    from realmspinner.studio.modes.clay import texture_link

    side = texture_link.MAX_TEXTURE_SIDE + 1
    path = tmp_path / "wide.png"
    Image.fromarray(np.zeros((2, side, 4), dtype=np.uint8), "RGBA").save(path)
    monkeypatch.setattr(dialogs, "open_file", lambda *a, **kw: path)

    decoded: list[Any] = []
    real = pixelguard.decode_rgba

    def spy(*a: Any, **kw: Any) -> Any:
        decoded.append(a)
        return real(*a, **kw)

    monkeypatch.setattr(pixelguard, "decode_rgba", spy)
    converted: list[str] = []
    real_convert = Image.Image.convert

    def spy_convert(self: Any, *a: Any, **kw: Any) -> Any:
        converted.append("convert")
        return real_convert(self, *a, **kw)

    monkeypatch.setattr(Image.Image, "convert", spy_convert)

    with pytest.raises(ServiceError) as caught:
        clay_props._pick_texture("base_color")

    assert not decoded and not converted, "the size is read from the header, nothing is allocated"
    assert str(side) in caught.value.message
    assert str(texture_link.MAX_TEXTURE_SIDE) in caught.value.message


def test_assigning_a_png_at_the_ceiling_still_decodes(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    import numpy as np
    from PIL import Image

    from realmspinner.studio import dialogs
    from realmspinner.studio.modes.clay import texture_link

    side = texture_link.MAX_TEXTURE_SIDE
    path = tmp_path / "edge.png"
    Image.fromarray(np.full((1, side, 4), 9, dtype=np.uint8), "RGBA").save(path)
    monkeypatch.setattr(dialogs, "open_file", lambda *a, **kw: path)
    result = clay_props._pick_texture("base_color")
    assert result is not None and result["width"] == side and result["height"] == 1
    assert len(result["rgba"]) == side * 4

