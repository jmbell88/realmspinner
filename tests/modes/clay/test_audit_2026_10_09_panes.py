"""The 2026-10-09 audit's panes findings: clay-04, 05, 16, 23, 27, 44, 47 and 48.

One block per finding id. The manual blocks read ``docs/manual/30-clay.md`` against
the constants and tables the sentences describe.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from _ui_context import imgui_context

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import selection
from realmspinner.studio.modes.clay import starter_colours as SC
from realmspinner.studio.modes.clay.state import ClayState, ClayTab
from realmspinner.studio.modes.clay.ui import inspector_actions
from realmspinner.studio.modes.clay.ui import inspector_elements as ie
from realmspinner.studio.modes.clay.ui.panes import inspector as clay_inspector
from realmspinner.studio.modes.clay.ui.panes import outliner as clay_outliner
from realmspinner.studio.modes.clay.ui.panes import props as clay_props
from realmspinner.studio.modes.clay.ui.panes import swatches
from realmspinner.studio.modes.clay.ui.panes import tools as clay_tools

_PANES = Path(clay_outliner.__file__).parent
_MANUAL = Path(__file__).resolve().parents[3] / "docs" / "manual" / "30-clay.md"


class _Ctx:
    def __init__(self, state: Any = None) -> None:
        self.toasts: list[str] = []
        self.state = state

    def toast(self, message: str, level: str = "info") -> None:
        self.toasts.append(message)


def _abc() -> tuple[bd.ClayDoc, dict[str, Any]]:
    doc, ctx = bd.ClayDoc(), _Ctx()
    objs = {}
    for name in "ABC":
        obj = clay_tools.add_primitive(ctx, doc, "box")
        obj.name = name
        objs[name] = obj
    return doc, objs


def _names(doc: bd.ClayDoc) -> str:
    return "".join(o.name for o in doc.objects)


# --- clay-04 ---------------------------------------------------------------------


def test_the_outliner_eye_deselects_the_object_it_hides(monkeypatch: pytest.MonkeyPatch) -> None:
    doc, objs = _abc()
    target = objs["A"]
    doc.select([target.uid])
    monkeypatch.setattr(
        clay_outliner.controls, "button", lambda label, *a, **k: "##vis" in label
    )
    with imgui_context(monkeypatch) as imgui:
        imgui.new_frame()
        imgui.begin("##host")
        clay_outliner._row(
            _Ctx(), ClayState(), doc, target, 0, False, filtered=False, saving=False
        )
        imgui.end()
        imgui.end_frame()
    assert not doc.by_uid(target.uid).visible, "the premise: the eye hid it"
    assert target.uid not in doc.selection, "a hidden object stayed selected"


# --- clay-05 ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dragged", "target", "zone", "expected"),
    [
        ("A", "C", "before", "BAC"),
        ("A", "B", "after", "BAC"),
        ("A", "C", "after", "BCA"),
        ("C", "A", "before", "CAB"),
        ("C", "A", "after", "ACB"),
        ("B", "A", "before", "BAC"),
        ("C", "B", "after", "ABC"),
    ],
)
def test_dropping_a_row_before_a_later_row_places_it_directly_before_it(
    dragged: str, target: str, zone: str, expected: str
) -> None:
    doc, objs = _abc()
    t = objs[target]
    clay_outliner._apply_drop(_Ctx(), doc, objs[dragged].uid, t, doc.index_of(t.uid), zone)
    assert _names(doc) == expected


def test_dropping_a_child_beside_a_later_sibling_lands_beside_it() -> None:
    doc, objs = _abc()
    parent = clay_tools.add_primitive(_Ctx(), doc, "box")
    parent.name = "P"
    doc.set_parent(objs["A"].uid, parent.uid, keep_world=True)
    doc.set_parent(objs["C"].uid, parent.uid, keep_world=True)
    # Document order is A B C P; A and C are P's children. A dropped before C.
    clay_outliner._apply_drop(
        _Ctx(), doc, objs["A"].uid, objs["C"], doc.index_of(objs["C"].uid), "before"
    )
    order = _names(doc)
    assert order.index("A") == order.index("C") - 1, order


# --- clay-16 ---------------------------------------------------------------------


def _cylinder_tab() -> tuple[ClayState, ClayTab, _Ctx, Any]:
    doc = bd.ClayDoc()
    state = ClayState()
    tab = state.add(ClayTab(doc=doc))
    ctx = _Ctx(SimpleNamespace(clay=state))
    obj = clay_tools.add_primitive(ctx, doc, "cylinder")
    SC.paint(ctx, doc, [obj.uid], SC.by_name("Red"))
    return state, tab, ctx, obj


def test_ctrl_clicking_a_slot_does_not_change_the_colour_a_later_segment_edit_gives_the_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _state, tab, ctx, obj = _cylinder_tab()
    doc = tab.doc
    red = doc.by_uid(obj.uid).material
    blue = doc.add_material(replace(bd.default_material("Blue"), base_color_factor=(0, 0, 1, 1)))
    monkeypatch.setattr(swatches, "_ctrl_held", lambda: True)

    swatches._pick_slot(ctx, doc, doc.by_uid(obj.uid), blue)

    assert tab.edit_slot(doc, doc.by_uid(obj.uid)) == blue, "the page now edits the slot"
    edited = dict(obj.params)
    edited["segments"] = int(edited.get("segments", 12)) + 4
    clay_props.apply_generator_params(doc, doc.by_uid(obj.uid), edited, ctx=ctx)
    after = doc.by_uid(obj.uid)
    assert set(after.mesh.material.tolist()) == {red}, "the shape was recoloured"
    assert after.material == red


def test_the_colours_page_edits_the_slot_a_ctrl_click_chose() -> None:
    _state, tab, _ctx, obj = _cylinder_tab()
    tab.edit_slots[obj.uid] = 0
    assert clay_props._page_slot(tab, tab.doc, tab.doc.by_uid(obj.uid)) == 0
    tab.edit_slots.clear()
    assert clay_props._page_slot(tab, tab.doc, tab.doc.by_uid(obj.uid)) == tab.doc.by_uid(
        obj.uid
    ).material


def test_a_plain_object_mode_click_repaints_and_the_page_follows_the_paint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(swatches, "_ctrl_held", lambda: False)
    _state, tab, ctx, obj = _cylinder_tab()
    doc = tab.doc
    other = doc.add_material()
    tab.edit_slots[obj.uid] = 0
    swatches._pick_slot(ctx, doc, doc.by_uid(obj.uid), other)
    assert (doc.by_uid(obj.uid).mesh.material == other).all()
    assert tab.edit_slot(doc, doc.by_uid(obj.uid)) == other


# --- clay-23 ---------------------------------------------------------------------


def test_the_element_line_does_no_per_shape_scan_while_nothing_changes_with_every_shape_picked(
) -> None:
    doc, ctx = bd.ClayDoc(), _Ctx()
    for _ in range(40):
        clay_tools.add_primitive(ctx, doc, "box")
    doc.select([o.uid for o in doc.objects])
    doc.set_element_mode("face")
    selection.select_all(doc)
    calls = 0
    real = doc.by_uid

    def counted(uid: int) -> Any:
        nonlocal calls
        calls += 1
        return real(uid)

    doc.by_uid = counted  # type: ignore[method-assign]
    first = ie.element_line(doc)
    second = ie.element_line(doc)
    assert first == second == "240 faces selected across 40 shapes"
    assert calls == 0, f"{calls} linear by_uid scans for one element line"


def test_the_element_line_follows_a_hidden_shape_after_the_memo() -> None:
    doc, ctx = bd.ClayDoc(), _Ctx()
    for _ in range(2):
        clay_tools.add_primitive(ctx, doc, "box")
    doc.select([o.uid for o in doc.objects])
    doc.set_element_mode("face")
    selection.select_all(doc)
    assert ie.element_counts(doc) == (12, 2)
    doc.set_visibility({doc.objects[0].uid: False})
    doc.select([doc.objects[1].uid])
    selection.select_all(doc)
    assert ie.element_counts(doc) == (6, 1)


# --- clay-27 and clay-47: the manual ------------------------------------------------


def _manual() -> str:
    return " ".join(_MANUAL.read_text(encoding="utf-8").split())


def test_the_manual_dock_share_matches_the_constants() -> None:
    words = {"a quarter": 0.25, "a third": 1 / 3, "two fifths": 0.4, "a fifth": 0.2,
             "a half": 0.5}
    rows = {"three": 3, "four": 4, "five": 5, "six": 6}
    found = re.search(
        r"takes a fixed share of the column \(about ([a-z ]+?), never fewer than (\w+) rows",
        _manual(),
    )
    assert found, "the dock sentence is gone or reworded: update this test with it"
    assert words[found.group(1)] == pytest.approx(clay_inspector.DOCK_SHARE)
    assert rows[found.group(2)] == clay_inspector.DOCK_MIN_ROWS


def test_the_manual_inspector_group_list_matches_action_groups() -> None:
    text = _manual()
    start = text.index("**More actions** on a selected shape lists")
    sentence = text[start : text.index("with **Duplicate**", start)]
    labels = [label for _key, label, _ops in inspector_actions.action_groups("object")]
    listed = re.split(r",\s*(?:and\s+)?|\s+and\s+", sentence.split("groups", 1)[1].strip(" :—-"))
    assert [item.strip(" —-") for item in listed if item.strip(" —-")] == labels
    # The element state draws the Colour card between Last action and the folds.
    region = text[text.index("**In Points, Edges or Faces** the heading") :]
    region = region[: region.index("## Parents and groups")]
    assert "**Colour**" in region
    row = next(r for r in text.split("| Points, edges or faces |")[1:2])
    assert "**Colour**" in row.split("|")[0]


# --- clay-44 ---------------------------------------------------------------------


def test_closing_a_tab_forgets_its_outliner_collapse_entries() -> None:
    state = ClayState()
    a = state.add(ClayTab(doc=bd.ClayDoc()))
    b = state.add(ClayTab(doc=bd.ClayDoc()))
    clay_outliner.set_collapsed(state, a.uid, 1, True)
    clay_outliner.set_collapsed(state, b.uid, 1, True)
    clay_outliner.begin_rename(state, a.uid, 1)
    state.outliner_anchor = (a.uid, 1)
    state.close(a.uid)
    state.prune_outliner_keys()
    assert state.outliner_collapsed == {(b.uid, 1)}
    assert state.renaming == 0 and state.outliner_anchor == 0


# --- clay-48 ---------------------------------------------------------------------


def test_no_clay_pane_docstring_names_a_deleted_pane_or_tab() -> None:
    stale = ("Properties' Document tab", "replaced ``bridge.py``", "and the tools pane")
    for name in ("header.py", "props.py", "menu.py", "swatches.py"):
        text = (_PANES / name).read_text(encoding="utf-8")
        for needle in stale:
            assert needle not in text, f"{name} still says {needle!r}"
    assert not hasattr(swatches, "current_slot"), "dead helper"
