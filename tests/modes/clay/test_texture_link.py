"""Clay's texture round trip through Inker (``clay/texture_link.py``).

The claim under test is the pull model: a Clay palette entry follows an Inker
document, and **lands once per committed revision**, never per frame and never
when nothing moved. Each test's name is the claim.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from realmspinner.kernels.mesh import document as bd
from realmspinner.kernels.pixel import PICO8, Document
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay import texture_link as tl
from realmspinner.studio.modes.clay.state import ClayState, ClayTab
from realmspinner.studio.modes.inker import mode as inker_mode
from realmspinner.studio.modes.inker import opening as inker_open
from realmspinner.studio.modes.inker.state import InkerState
from realmspinner.studio.state import AppState
from realmspinner.studio.tasks import Done

RED = (255, 0, 77, 255)


class _Inker:
    """The slice of ``InkerState`` the link reads: ``docs`` and ``activate``."""

    def __init__(self) -> None:
        self.docs: list[Any] = []
        self.activated: list[str] = []

    def activate(self, uid: str) -> None:
        self.activated.append(uid)


class _Ctx:
    def __init__(self) -> None:
        self.state = AppState()
        self.state.clay = ClayState()
        self.state.inker = _Inker()
        self.toasts: list[tuple[str, str]] = []

    def toast(self, text: str, level: str = "info", **_kw: Any) -> None:
        self.toasts.append((text, level))


def _clay_tab(ctx: _Ctx, *, textured: bool = True) -> ClayTab:
    doc = bd.ClayDoc()
    if textured:
        assert doc.add_texture(0, 32)
    tab = ClayTab(doc=doc, title="Crate")
    ctx.state.clay.add(tab)
    return tab


def _inker_tab(ctx: _Ctx, uid: str = "id1", *, size: int = 4) -> Any:
    px = np.zeros((size, size, 4), np.uint8)
    px[..., 3] = 255
    tab = SimpleNamespace(
        uid=uid, title="Crate - Material (texture)", doc=Document.from_pixels(px, name="T")
    )
    ctx.state.inker.docs.append(tab)
    return tab


def _paint(inker_tab: Any, colour: tuple[int, int, int, int] = RED) -> None:
    """A real committed step: a flood fill over the (uniform) canvas."""
    before = inker_tab.doc.history.head
    assert inker_tab.doc.fill((0, 0), colour, thresh=0)
    assert inker_tab.doc.history.head != before


def _linked(ctx: _Ctx, tab: ClayTab, inker_tab: Any, index: int = 0) -> tl.InkerLink:
    link = tl.InkerLink(index, tab.doc.materials[index], inker_tab.uid, inker_tab.doc.history.head)
    tab.inker_links.append(link)
    return link


def _steps(tab: ClayTab) -> int:
    return len(tab.doc.history)


def test_a_pull_lands_once_per_committed_revision_as_one_undo_step():
    ctx = _Ctx()
    tab = _clay_tab(ctx)
    inker = _inker_tab(ctx)
    link = _linked(ctx, tab, inker)
    before = _steps(tab)

    _paint(inker)
    tl.pull(ctx, tab)
    tl.pull(ctx, tab)
    tl.pull(ctx, tab)

    assert _steps(tab) == before + 1
    flat = inker.doc.flatten(matte=False)
    material = tab.doc.materials[0]
    assert material.base_color == (4, 4, flat.tobytes())
    assert material.nearest is True
    assert link.material is material
    assert link.pulled_rev == inker.doc.history.head

    _paint(inker, (0, 228, 54, 255))
    tl.pull(ctx, tab)
    assert _steps(tab) == before + 2


def test_a_pull_with_an_unchanged_head_flattens_nothing_and_pushes_nothing(monkeypatch):
    ctx = _Ctx()
    tab = _clay_tab(ctx)
    inker = _inker_tab(ctx)
    _linked(ctx, tab, inker)
    calls: list[int] = []
    real = Document.flatten
    monkeypatch.setattr(
        Document, "flatten", lambda self, **kw: calls.append(1) or real(self, **kw)
    )
    before, material = _steps(tab), tab.doc.materials[0]

    for _ in range(5):
        tl.pull(ctx, tab)
        tl.pull_all(ctx)

    assert calls == []
    assert _steps(tab) == before
    assert tab.doc.materials[0] is material


def test_a_head_that_moved_but_left_the_pixels_alone_advances_the_revision_without_a_step():
    ctx = _Ctx()
    tab = _clay_tab(ctx)
    inker = _inker_tab(ctx)
    link = _linked(ctx, tab, inker)
    _paint(inker)
    tl.pull(ctx, tab)
    landed, before = tab.doc.materials[0], _steps(tab)

    # Two commits that net out to the pixels already landed.
    _paint(inker, (0, 228, 54, 255))
    _paint(inker)
    assert inker.doc.history.head != link.pulled_rev
    tl.pull(ctx, tab)

    assert _steps(tab) == before
    assert tab.doc.materials[0] is landed
    assert link.pulled_rev == inker.doc.history.head


def test_pull_all_reaches_a_tab_that_is_not_the_active_one():
    ctx = _Ctx()
    first = _clay_tab(ctx)
    second = _clay_tab(ctx)
    assert ctx.state.clay.active_uid == second.uid
    inker = _inker_tab(ctx)
    _linked(ctx, first, inker)
    _paint(inker)

    tl.pull_all(ctx)

    assert first.doc.materials[0].base_color[2] == inker.doc.flatten(matte=False).tobytes()
    assert second.inker_links == []


def test_pull_all_is_a_no_op_before_clay_or_inker_was_ever_used():
    ctx = SimpleNamespace(state=AppState(), toast=lambda *a, **k: None)
    tl.pull_all(ctx)  # no clay state
    ctx.state.clay = ClayState()
    tl.pull_all(ctx)  # no tabs
    assert tl.open_inker_docs(ctx) == []


def test_editing_the_entrys_colour_keeps_the_link_and_the_pull_keeps_the_edit():
    ctx = _Ctx()
    tab = _clay_tab(ctx)
    inker = _inker_tab(ctx)
    _linked(ctx, tab, inker)
    # A colour edit replaces the Material but not the picture it holds.
    tab.doc.set_material(0, replace(tab.doc.materials[0], base_color_factor=(0.5, 0.5, 0.5, 1.0)))
    picture = tab.doc.materials[0].base_color

    _paint(inker)
    tl.pull(ctx, tab)

    assert len(tab.inker_links) == 1
    landed = tab.doc.materials[0]
    assert landed.base_color is not picture
    assert landed.base_color_factor == (0.5, 0.5, 0.5, 1.0)


def test_a_link_whose_texture_was_replaced_is_dropped_and_never_lands_on_it():
    ctx = _Ctx()
    tab = _clay_tab(ctx)
    inker = _inker_tab(ctx)
    _linked(ctx, tab, inker)
    # Clearing the texture (or undoing a pull) puts a different picture there.
    tab.doc.set_material(0, replace(tab.doc.materials[0], base_color=None, nearest=False))
    cleared = tab.doc.materials[0]
    before = _steps(tab)

    _paint(inker)
    tl.pull(ctx, tab)

    assert tab.inker_links == []
    assert tab.doc.materials[0] is cleared
    assert _steps(tab) == before


def test_a_link_whose_slot_was_removed_below_it_is_dropped_not_shifted_onto_a_neighbour():
    ctx = _Ctx()
    tab = _clay_tab(ctx, textured=False)
    tab.doc.add_material()
    assert tab.doc.add_texture(1, 32)
    inker = _inker_tab(ctx)
    _linked(ctx, tab, inker, index=1)
    assert tab.doc.remove_material(0)
    survivor = tab.doc.materials[0]

    _paint(inker)
    tl.pull(ctx, tab)

    assert tab.inker_links == []
    assert tab.doc.materials[0] is survivor


def test_a_closed_inker_tab_drops_its_link():
    ctx = _Ctx()
    tab = _clay_tab(ctx)
    inker = _inker_tab(ctx)
    _linked(ctx, tab, inker)
    ctx.state.inker.docs.remove(inker)
    before = _steps(tab)

    tl.pull(ctx, tab)

    assert tab.inker_links == []
    assert _steps(tab) == before


def test_a_pull_of_an_absurdly_large_drawing_is_refused_and_unlinked():
    ctx = _Ctx()
    tab = _clay_tab(ctx)
    inker = _inker_tab(ctx, size=tl.MAX_TEXTURE_SIDE + 1)
    _linked(ctx, tab, inker)
    material, before = tab.doc.materials[0], _steps(tab)
    _paint(inker)

    tl.pull(ctx, tab)

    assert tab.inker_links == []
    assert tab.doc.materials[0] is material
    assert _steps(tab) == before
    assert ctx.toasts and ctx.toasts[-1][1] == "error"


def test_undoing_a_pull_restores_the_previous_texture():
    ctx = _Ctx()
    tab = _clay_tab(ctx)
    inker = _inker_tab(ctx)
    _linked(ctx, tab, inker)
    original = tab.doc.materials[0]

    _paint(inker)
    tl.pull(ctx, tab)
    assert tab.doc.materials[0].base_color != original.base_color
    tab.doc.undo()

    assert tab.doc.materials[0].base_color == original.base_color


def test_take_back_lands_the_document_and_records_a_link_that_keeps_following_it():
    ctx = _Ctx()
    tab = _clay_tab(ctx)
    inker = _inker_tab(ctx, uid="other")
    _paint(inker)
    before = _steps(tab)

    tl.take_back(ctx, tab, 0, inker)

    assert _steps(tab) == before + 1
    assert tab.doc.materials[0].base_color == (4, 4, inker.doc.flatten(matte=False).tobytes())
    assert tab.doc.materials[0].nearest is True
    (link,) = tab.inker_links
    assert (link.index, link.inker_uid) == (0, "other")
    assert link.material is tab.doc.materials[0]
    assert link.pulled_rev == inker.doc.history.head

    _paint(inker, (41, 173, 255, 255))
    tl.pull(ctx, tab)
    assert _steps(tab) == before + 2


def test_take_back_replaces_the_slots_previous_link_rather_than_stacking_two():
    ctx = _Ctx()
    tab = _clay_tab(ctx)
    first = _inker_tab(ctx, uid="a")
    second = _inker_tab(ctx, uid="b")
    _linked(ctx, tab, first)
    _paint(second)

    tl.take_back(ctx, tab, 0, second)

    assert [link.inker_uid for link in tab.inker_links] == ["b"]


def test_unlink_drops_only_that_slots_links():
    ctx = _Ctx()
    tab = _clay_tab(ctx)
    tab.inker_links[:] = [
        tl.InkerLink(0, tab.doc.materials[0], "a", 0),
        tl.InkerLink(1, tab.doc.materials[0], "b", 0),
    ]
    tl.unlink(tab, 0)
    assert [link.index for link in tab.inker_links] == [1]


def test_edit_in_inker_without_a_texture_toasts_and_opens_nothing(monkeypatch):
    ctx = _Ctx()
    tab = _clay_tab(ctx, textured=False)
    opened: list[Any] = []
    monkeypatch.setattr(inker_open, "open_pixels", lambda *a, **k: opened.append((a, k)))

    tl.edit_in_inker(ctx, tab, 0)
    tl.edit_in_inker(ctx, tab, 9)

    assert opened == []
    assert ctx.toasts[0] == ("Add a texture first.", "error")
    assert ctx.toasts[1][1] == "error"


def test_edit_in_inker_opens_pico8_with_a_distinct_title_and_its_on_open_records_the_link(
    monkeypatch,
):
    ctx = _Ctx()
    tab = _clay_tab(ctx)
    tab.doc.add_material()
    assert tab.doc.add_texture(1, 32)
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    monkeypatch.setattr(inker_open, "open_pixels", lambda *a, **k: calls.append((a, k)))

    tl.edit_in_inker(ctx, tab, 0)
    tl.edit_in_inker(ctx, tab, 1)

    assert len(calls) == 2
    (args0, kw0), (_args1, kw1) = calls
    assert args0[1].shape == (32, 32, 4) and args0[1].dtype == np.uint8
    assert kw0["palette"] == PICO8
    assert kw0["title"] != kw1["title"]
    assert "Crate" in kw0["title"]

    inker = _inker_tab(ctx, uid="opened")
    kw0["on_open"](inker)
    (link,) = tab.inker_links
    assert (link.index, link.inker_uid, link.pulled_rev) == (0, "opened", inker.doc.history.head)
    assert link.material is tab.doc.materials[0]


def test_an_on_open_that_arrives_after_the_texture_was_cleared_records_nothing(monkeypatch):
    ctx = _Ctx()
    tab = _clay_tab(ctx)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(inker_open, "open_pixels", lambda *a, **k: calls.append(k))
    tl.edit_in_inker(ctx, tab, 0)
    tab.doc.set_material(0, replace(tab.doc.materials[0], base_color=None, nearest=False))

    calls[0]["on_open"](_inker_tab(ctx))

    assert tab.inker_links == []


def test_edit_in_inker_again_focuses_the_open_document_instead_of_opening_a_second(monkeypatch):
    ctx = _Ctx()
    tab = _clay_tab(ctx)
    inker = _inker_tab(ctx)
    _linked(ctx, tab, inker)
    opened: list[Any] = []
    monkeypatch.setattr(inker_open, "open_pixels", lambda *a, **k: opened.append(a))

    tl.edit_in_inker(ctx, tab, 0)

    assert opened == []
    assert ctx.state.inker.activated == [inker.uid]
    assert ctx.state.mode == "inker"


def test_closing_a_clay_tab_releases_its_uv_texture_and_drops_its_links(monkeypatch):
    from realmspinner.studio.modes.clay.ui import _uv_texture

    ctx = _Ctx()
    ctx.settings = None
    ctx.confirms = SimpleNamespace(ask=lambda *a, **k: None)
    tab = _clay_tab(ctx)
    tab.saved_head = tab.doc.history.head
    _linked(ctx, tab, _inker_tab(ctx))
    released: list[str] = []
    monkeypatch.setattr(_uv_texture, "release_doc", lambda c, uid: released.append(uid))
    monkeypatch.setattr(clay_mode, "persist", lambda c: None)

    clay_mode.close_tab(ctx, tab.uid)

    assert released == [tab.uid]
    assert tab.inker_links == []


# --- end to end through the real Inker opening -------------------------------


class _RealCtx:
    def __init__(self) -> None:
        self.state = AppState()
        self.state.clay = ClayState()
        self.state.inker = InkerState()
        self.settings = None
        self.svc = object()
        self.keys: list[str] = []
        self.toasts: list[tuple[str, str]] = []
        self.done: Done | None = None

    def submit(self, key: str, fn: Any, *args: Any, **kwargs: Any) -> bool:
        self.keys.append(key)
        self.done = Done(key=key, result=fn(*args, **kwargs))
        return True

    def toast(self, text: str, level: str = "info", **_kw: Any) -> None:
        self.toasts.append((text, level))


@pytest.fixture
def _no_inker_settings(monkeypatch):
    monkeypatch.setattr(inker_mode, "persist", lambda ctx: None)
    monkeypatch.setattr(inker_mode, "remember_path", lambda ctx, path: None)


def test_the_real_round_trip_opens_a_palette_locked_tab_links_it_and_a_stroke_lands_in_clay(
    _no_inker_settings,
):
    ctx = _RealCtx()
    doc = bd.ClayDoc()
    assert doc.add_texture(0, 32)
    tab = ClayTab(doc=doc, title="Crate")
    ctx.state.clay.add(tab)

    tl.edit_in_inker(ctx, tab, 0)
    assert ctx.done is not None
    assert ctx.keys == ["inker-open:pixels:Crate - Material (texture)"]
    inker_mode._done_open(ctx, ctx.state.inker, ctx.done)

    inker_tab = ctx.state.inker.docs[-1]
    assert inker_tab.doc.is_palette_locked
    assert [tuple(c) for c in inker_tab.doc.palette] == list(PICO8)
    assert ctx.state.mode == "inker"
    (link,) = tab.inker_links
    assert link.inker_uid == inker_tab.uid
    assert link.pulled_rev == inker_tab.doc.history.head

    # Untouched: the palette snap itself is not a stroke to land.
    before = _steps(tab)
    tl.pull_all(ctx)
    assert _steps(tab) == before

    assert inker_tab.doc.fill((0, 0), RED, thresh=0)
    tl.pull_all(ctx)
    tl.pull_all(ctx)

    assert _steps(tab) == before + 1
    flat = inker_tab.doc.flatten(matte=False)
    assert tab.doc.materials[0].base_color == (32, 32, flat.tobytes())
    assert tuple(int(v) for v in flat[0, 0]) == RED
    assert tab.doc.materials[0].nearest is True

    # And the open document is the one a second click comes back to.
    tl.edit_in_inker(ctx, tab, 0)
    assert ctx.keys == ["inker-open:pixels:Crate - Material (texture)"]
    assert ctx.state.inker.active_uid == inker_tab.uid
