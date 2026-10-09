"""Regression tests for the Add-primitive call sites the 2026-10-07 Clay audit's
clay-13 fix missed: the tool rail's quick-shape buttons and what was the menu
strip's Add menu (now the shape list ``tools.draw_add_menu`` draws, which the rail's
``+`` flyout and the Inspector's "More shapes" disclosure both host).

``ClayDoc.add_object`` refuses a shape that would pass the object or triangle
ceiling, and ``tools.place_primitive`` is the human door that turns that refusal
into a toast. ``draw_add_menu`` used it; the rail and the strip still called the
raising ``add_primitive`` bare, so a click on a full document escaped as an
exception out of the pane's draw. Neither can be drawn headlessly here, so the
imgui calls around the one click are stubbed and everything else is real. The
strip's two tests moved to the shape list when the strip was deleted: the claim
(a refused add is a toast and leaves no trace; an accepted one places, selects and
remembers the shape) is the shape list's now.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import Any

import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.mesh import glbimport
from realmspinner.kernels.mesh import primitives as bp
from realmspinner.studio.modes.clay.ui.panes import rail as pane_rail
from realmspinner.studio.modes.clay.ui.panes import tools as pane_tools


class _ToastCtx:
    def __init__(self) -> None:
        self.toasts: list[tuple[str, str]] = []

    def toast(self, message: str, level: str = "info") -> None:
        self.toasts.append((message, level))


def _full_document(monkeypatch: pytest.MonkeyPatch, objects: int = 3) -> bd.ClayDoc:
    """A document already at the ceiling, built through the constructor, which
    does not apply the refusal."""
    monkeypatch.setattr(glbimport, "MAX_OBJECTS", objects)
    return bd.ClayDoc(
        bd.Obj(uid=bd.new_uid(), name=f"o{i}", mesh=bp.box()) for i in range(objects)
    )


@contextlib.contextmanager
def _popup(*_a: Any, **_k: Any) -> Any:
    yield False


def _stub_rail(monkeypatch: pytest.MonkeyPatch) -> None:
    # The rail draws each button through ``_cell`` (a glyph over a word); what this
    # file is about is what a click on a quick shape *does*, so the cell is the
    # one stubbed seam and reports a press for every add cell.
    def cell(_glyph: str, ident: str, *_a: Any, **_k: Any) -> bool:
        return "clay-rail/add/" in ident

    fake_imgui = SimpleNamespace(
        get_item_rect_min=lambda: SimpleNamespace(x=0.0, y=0.0),
        get_item_rect_max=lambda: SimpleNamespace(x=0.0, y=0.0),
        is_popup_open=lambda *_a: False,
        open_popup=lambda *_a: None,
        get_style=lambda: SimpleNamespace(item_spacing=SimpleNamespace(y=8.0)),
        # Tall: the quick shapes are drawn only when they fit (``quick_shapes_fit``).
        get_content_region_avail=lambda: SimpleNamespace(x=100.0, y=10_000.0),
    )
    monkeypatch.setattr(pane_rail.anchors, "mark_window", lambda *_a: None)
    monkeypatch.setattr(pane_rail.widgets, "divider", lambda *_a, **_k: None)
    monkeypatch.setattr(pane_rail, "_cell", cell)
    monkeypatch.setattr(pane_rail.controls, "menu_popup", _popup)
    monkeypatch.setattr(pane_rail, "imgui", fake_imgui)


def _stub_shape_list(monkeypatch: pytest.MonkeyPatch) -> None:
    # Every row reports a press; the heading label is the only other imgui call
    # ``draw_add_menu`` makes with ``heading=False`` (the Inspector's disclosure).
    monkeypatch.setattr(pane_tools.widgets, "field_label", lambda *_a, **_k: None)
    monkeypatch.setattr(pane_tools.controls, "menu_item", lambda *_a, **_k: (True, True))


def test_a_rail_quick_shape_click_on_a_full_document_toasts_and_leaves_no_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc = _full_document(monkeypatch)
    ctx = _ToastCtx()
    state = SimpleNamespace(generator="sentinel", tool="select")
    tab = SimpleNamespace(doc=doc, saving=False)
    _stub_rail(monkeypatch)
    steps, objects = len(doc.history), len(doc.objects)

    pane_rail._body(ctx, state, tab)

    assert ctx.toasts and all(level == "error" for _m, level in ctx.toasts)
    assert len(doc.history) == steps and len(doc.objects) == objects
    assert doc.history._open_gestures == 0
    assert state.generator == "sentinel", "a refused add still moved the remembered shape"


def test_a_rail_quick_shape_click_on_a_document_with_room_still_places_and_selects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc = bd.ClayDoc()
    ctx = _ToastCtx()
    state = SimpleNamespace(generator="sentinel", tool="select")
    tab = SimpleNamespace(doc=doc, saving=False)
    _stub_rail(monkeypatch)
    pane_rail._body(ctx, state, tab)

    assert ctx.toasts == []
    assert len(doc.objects) >= 1
    assert state.generator == doc.objects[-1].generator
    assert doc.selection == {doc.objects[-1].uid}


def test_a_shape_list_click_on_a_full_document_toasts_and_leaves_no_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc = _full_document(monkeypatch)
    ctx = _ToastCtx()
    state = SimpleNamespace(generator="sentinel")
    tab = SimpleNamespace(doc=doc, saving=False)
    _stub_shape_list(monkeypatch)
    steps, objects = len(doc.history), len(doc.objects)

    pane_tools.draw_add_menu(ctx, state, tab, heading=False)

    assert ctx.toasts and all(level == "error" for _m, level in ctx.toasts)
    assert len(doc.history) == steps and len(doc.objects) == objects
    assert doc.history._open_gestures == 0
    assert state.generator == "sentinel", "a refused add still moved the remembered shape"


def test_a_shape_list_click_on_a_document_with_room_still_places_and_selects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc = bd.ClayDoc()
    ctx = _ToastCtx()
    state = SimpleNamespace(generator="sentinel")
    tab = SimpleNamespace(doc=doc, saving=False)
    _stub_shape_list(monkeypatch)
    # Every row reports a click; one shape is enough to prove the success path.
    clicks = iter([(True, True)])
    monkeypatch.setattr(
        pane_tools.controls, "menu_item", lambda *_a, **_k: next(clicks, (False, False))
    )

    pane_tools.draw_add_menu(ctx, state, tab, heading=False)

    assert ctx.toasts == []
    assert len(doc.objects) == 1
    assert state.generator == doc.objects[0].generator
    assert doc.selection == {doc.objects[0].uid}
