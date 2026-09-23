"""Closes findings from the 2026-09-23 audit of Packwright (second run).

packwright-01: Inker's "Add to Packwright" (``add_inker_document``) is gated
only on the Inker tab it comes from -- it never checked the *target* atlas's
``tab.busy``, unlike the sibling add doors (``add_source_paths``,
``add_job_source``, ``add_rendered_sheet``) the first run of this finding
already fixed.

packwright-02: the empty-canvas overlay's "Add sources" action calls
``ask_add_sources`` directly, ungated -- the pane's own Add button greys on
``tab.busy``, but the overlay's action reaches the door with no such check.

packwright-03: the tile-set import popup's Import button disables only on
``problem`` or ``computing``, never on the *target* tab (resolved by
``tileset_import_uid``, not whatever tab is active) being busy, so a save
elsewhere did not stop tiles being spliced in mid-encode.

packwright-04: the "From Inker" button in the sources pane greys on
``tab.busy`` with no ``reason=``, unlike its two siblings above it
(``Add an image...``, ``Add a tile set...``).

dev/TODO.md P65 item 4 (the 2026-09-23 audit, second run): every busy
refusal in ``packwright/mode.py`` used to return silently. Every one of them
now toasts the same shared sentence, ``widgets.DOCUMENT_SAVING_WHY``.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from realmspinner.studio import widgets
from realmspinner.studio.modes.packwright import mode as packwright_mode
from realmspinner.studio.modes.packwright.ui.panes import sources as packwright_sources


class FakeCtx:
    """The minimal double ``packwright_mode``'s add doors need -- trimmed
    from ``test_packwright_mode.py``'s own ``FakeCtx``, the same "this
    session does not own that module" reason the first-run audit test gave."""

    def __init__(self) -> None:
        self.state = _AppState()
        self.settings = _Settings()
        self.submitted: list[str] = []
        self.toasts: list[tuple[str, str]] = []

    def submit(self, key: str, run: Any, *args: Any) -> bool:
        self.submitted.append(key)
        return True

    def toast(self, message: str, kind: str = "info") -> None:
        self.toasts.append((message, kind))


class _AppState:
    def __init__(self) -> None:
        self.packwright = None
        self.inker = None
        self.mode = "home"
        self.preview: dict[str, Any] = {}


class _Settings:
    def __init__(self) -> None:
        self.store: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self.store.get(key)

    def set(self, key: str, value: Any) -> None:
        self.store[key] = value


def _busy_tab(ctx: FakeCtx) -> Any:
    tab = packwright_mode.new_document(ctx)
    tab.saving = True
    assert tab.busy is True
    return tab


# --- packwright-01: add_inker_document checks tab.busy ---------------------


def test_add_inker_document_refuses_while_the_target_atlas_is_saving():
    """``add_inker_document`` used to have no ``tab.busy`` check at all -- the
    one add door its own siblings (fixed by the first run of packwright-01)
    did not share. Fails against the unfixed code: frames land on a saving
    tab and no toast is raised."""
    import numpy as np

    ctx = FakeCtx()
    tab = _busy_tab(ctx)
    before = len(tab.doc.sources)
    pixels = np.zeros((4, 4, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    layer = SimpleNamespace(name="Layer 1", pixels=pixels, uid="layer-1")
    inker_tab = SimpleNamespace(
        title="Untitled",
        uid="ink-1",
        doc=SimpleNamespace(anim=None, stack=[layer], layer_meta={}),
    )

    packwright_mode.add_inker_document(ctx, inker_tab)

    assert len(tab.doc.sources) == before, "frames must not land on a saving tab"
    assert ctx.toasts, "a refused add must say why, like every other busy door"
    assert ctx.toasts[-1][0] == widgets.DOCUMENT_SAVING_WHY


# --- packwright-02: ask_add_sources checks tab.busy -------------------------


def test_ask_add_sources_refuses_while_the_atlas_is_saving():
    """``ask_add_sources`` relies on the pane's own gated button, but the
    empty-canvas overlay's action calls it directly and ungated. Fails
    against the unfixed code: a picker task is submitted for a saving tab."""
    ctx = FakeCtx()
    _busy_tab(ctx)

    packwright_mode.ask_add_sources(ctx)

    assert ctx.submitted == [], "a picker must not open onto a saving tab"
    assert ctx.toasts and ctx.toasts[-1][0] == widgets.DOCUMENT_SAVING_WHY


# --- packwright-03: import_tileset checks tab.busy --------------------------


def test_import_tileset_refuses_while_the_atlas_is_saving():
    """The popup's own Import button disables only on ``problem`` or
    ``computing``; ``import_tileset`` itself never checked whether the
    *requesting* tab (``tileset_import_uid``) was busy. Fails against the
    unfixed code: the pending sheet is sliced and spliced into a saving tab."""
    import numpy as np

    ctx = FakeCtx()
    tab = _busy_tab(ctx)
    state = packwright_mode.ensure(ctx)
    pixels = np.zeros((4, 4, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    state.tileset_import = ("sheet.png", "sheet", pixels)
    state.tileset_import_uid = tab.uid
    state.tileset_cell = (2, 2)
    before = len(tab.doc.sources)

    result = packwright_mode.import_tileset(ctx)

    assert result is False, "a busy target tab must refuse the import"
    assert len(tab.doc.sources) == before
    assert state.tileset_import is not None, (
        "the pending sheet stays parked so Import can be pressed again once "
        "the save lands"
    )
    assert ctx.toasts and ctx.toasts[-1][0] == widgets.DOCUMENT_SAVING_WHY


# --- packwright-04: the From Inker button gives a reason --------------------


def test_from_inker_button_greys_out_with_a_reason_while_saving():
    """``_from_inker`` used to disable its button with no ``reason=``, unlike
    its two siblings (``Add an image...``, ``Add a tile set...``) in the same
    pane. Checked by source, the first-run packwright-02 test's shape: imgui's
    widget stack needs a live frame this suite does not stand up."""
    source = inspect.getsource(packwright_sources._from_inker)
    assert "reason=widgets.DOCUMENT_SAVING_WHY" in source, (
        "the From Inker button must give the same reason every other greyed "
        "control in this pane already gives"
    )


# --- P65 item 4: every busy refusal in mode.py toasts the shared sentence --


def test_every_packwright_busy_refusal_says_why():
    """dev/TODO.md P65 item 4 (second run): every ``tab.busy`` refusal in
    ``packwright/mode.py`` used to return with nothing said. Exercised by
    source rather than one test per door -- the claim is that *all* of them
    share one sentence, which a per-function test would not itself prove."""
    source = inspect.getsource(packwright_mode)
    # One call per ``tab.busy``/``tab is None`` branch that refuses rather
    # than mints a document: ``add_source_paths``, ``add_job_source``,
    # ``add_rendered_sheet`` (the three the first run's packwright-01 fixed),
    # ``add_inker_document``, ``ask_add_sources``, ``import_tileset``,
    # ``remove_source``, ``rename_source`` and ``set_settings`` -- nine doors.
    assert source.count("docmodes.refuse(ctx, _BUSY_WHY)") == 9
    assert "_BUSY_WHY = widgets.DOCUMENT_SAVING_WHY" in source, (
        "one shared sentence for every busy refusal in the file, not a "
        "second string that could drift from the pane's own copy"
    )


def test_add_source_paths_still_refuses_while_saving_and_now_says_why():
    """The first run of packwright-01 already stopped the drop; this checks
    the toast P65 item 4 adds on top of that existing (silent) refusal."""
    ctx = FakeCtx()
    _busy_tab(ctx)

    packwright_mode.add_source_paths(ctx, [Path("nope.png")])

    assert ctx.submitted == []
    assert ctx.toasts and ctx.toasts[-1][0] == widgets.DOCUMENT_SAVING_WHY


def test_remove_source_still_refuses_while_saving_and_now_says_why():
    ctx = FakeCtx()
    tab = _busy_tab(ctx)
    source = tab.doc.add_source(_fake_source())

    packwright_mode.remove_source(ctx, source.uid, tab)

    assert ctx.toasts and ctx.toasts[-1][0] == widgets.DOCUMENT_SAVING_WHY


def _fake_source() -> Any:
    import numpy as np

    from realmspinner.studio.modes.packwright.engine.sources import Sprite

    pixels = np.zeros((4, 4, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    return Sprite(key="s0", name="s0", pixels=pixels)
