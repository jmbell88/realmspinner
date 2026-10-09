"""Regression tests for the two Add-primitive call sites the 2026-10-07 Clay
audit's clay-13 fix missed: the tool rail's quick-shape buttons and the menu
strip's Add menu.

``ClayDoc.add_object`` refuses a shape that would pass the object or triangle
ceiling, and ``tools.place_primitive`` is the human door that turns that refusal
into a toast. ``draw_add_menu`` used it; these two panes still called the
raising ``add_primitive`` bare, so a click on a full document escaped as an
exception out of the pane's draw. Neither pane can be drawn headlessly, so the
imgui calls around the one click are stubbed and everything else is real.
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
from realmspinner.studio.modes.clay.ui.panes import strip as pane_strip


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


@contextlib.contextmanager
def _open_menu(*_a: Any, **_k: Any) -> Any:
    yield True


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


def _stub_strip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pane_strip.controls, "menu", _open_menu)
    monkeypatch.setattr(pane_strip.controls, "menu_item", lambda *_a, **_k: (True, True))


def _primitives_section() -> Any:
    return SimpleNamespace(source="primitives")


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


def test_a_menu_strip_add_click_on_a_full_document_toasts_and_leaves_no_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc = _full_document(monkeypatch)
    ctx = _ToastCtx()
    state = SimpleNamespace(generator="sentinel")
    tab = SimpleNamespace(doc=doc, saving=False)
    _stub_strip(monkeypatch)
    steps, objects = len(doc.history), len(doc.objects)

    pane_strip._add_source(ctx, state, tab, _primitives_section())

    assert ctx.toasts and all(level == "error" for _m, level in ctx.toasts)
    assert len(doc.history) == steps and len(doc.objects) == objects
    assert doc.history._open_gestures == 0
    assert state.generator == "sentinel", "a refused add still moved the remembered shape"


def test_a_menu_strip_add_click_on_a_document_with_room_still_places_and_selects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc = bd.ClayDoc()
    ctx = _ToastCtx()
    state = SimpleNamespace(generator="sentinel")
    tab = SimpleNamespace(doc=doc, saving=False)
    _stub_strip(monkeypatch)
    # Every row reports a click; one shape is enough to prove the success path.
    clicks = iter([(True, True)])
    monkeypatch.setattr(
        pane_strip.controls, "menu_item", lambda *_a, **_k: next(clicks, (False, False))
    )

    pane_strip._add_source(ctx, state, tab, _primitives_section())

    assert ctx.toasts == []
    assert len(doc.objects) == 1
    assert state.generator == doc.objects[0].generator
    assert doc.selection == {doc.objects[0].uid}
