"""Regression tests for the 2026-09-26 audit's w2f1 fixer slice.

Three findings, three doors: Ctrl+V clobbering a fresh Copy with a stale OS
screenshot (inker-mode-02, ``mode.py``'s ``paste_from_os``), "go to next
conflict" moving the playhead by hand instead of through
``Document.set_current_frame`` (inker-mode-03, ``ops.py``'s
``_sheet_conflict_next``), and Edit > Filter raising a bare ``ValueError`` out
of the canvas pane on a tilemap layer (inker-panes-02, ``ops.py``'s "filter"
op and ``ui/panes/bridge.py``'s ``_open_filter``).
"""

from __future__ import annotations

from types import MethodType, SimpleNamespace
from typing import Any

import numpy as np

from realmspinner.kernels import pixel as inker
from realmspinner.kernels.pixel.sheetin import document_from_sheet
from realmspinner.kernels.pixel.tiles import strip
from realmspinner.studio import state as state_mod
from realmspinner.studio.modes.inker import mode as inker_mode
from realmspinner.studio.modes.inker import ops as inker_ops
from realmspinner.studio.modes.inker import state as inker_state
from realmspinner.studio.modes.inker.state import InkerDoc
from realmspinner.studio.modes.inker.ui.panes import bridge as inker_bridge

CELL = 8
FRAMES = 4


def _session(doc: Any = None):
    doc = inker.Document.blank(32, 32) if doc is None else doc
    tab = inker_state.InkerDoc(doc=doc, uid="t1", title="Untitled")
    state = inker_state.InkerState()
    state.add(tab)
    app = SimpleNamespace(inker=state, toasts=[])
    app.toast = MethodType(state_mod.AppState.toast, app)
    app.toast_once = MethodType(state_mod.AppState.toast_once, app)
    ctx = SimpleNamespace(state=app, toast=app.toast)
    return ctx, state, tab


class _PasteCtx:
    def __init__(self) -> None:
        self.toasts: list[tuple[str, str]] = []

    def toast(self, message: str, level: str = "info") -> None:
        self.toasts.append((message, level))


def _tab(size=(16, 16)) -> InkerDoc:
    doc = inker.Document.blank(*size)
    return InkerDoc(doc=doc, title="a.png", saved_head=doc.history.head)


# --- inker-mode-02: Ctrl+V vs a fresh in-app Copy ----------------------------


def test_paste_after_an_inker_copy_uses_the_copy_not_a_stale_os_image(monkeypatch):
    """The 2026-09-26 audit, finding inker-mode-02: ``paste_from_os`` pulled
    whatever sat on the OS clipboard on *every* Ctrl+V, even the same
    screenshot as last time, and stamped it straight over the app clipboard.
    Ctrl+C never touches the OS clipboard, so a Copy made after that
    screenshot was last seen was clobbered by the very next Ctrl+V re-grabbing
    the unchanged image -- Copy then Paste pasted the stale screenshot.
    """
    from PIL import Image, ImageGrab

    stale = Image.new("RGBA", (4, 4), (9, 9, 9, 255))
    monkeypatch.setattr(ImageGrab, "grabclipboard", lambda: stale)
    tab = _tab()
    ctx = _PasteCtx()

    # The first Ctrl+V of the session: nothing recorded yet to compare
    # against, so the screenshot legitimately lands.
    assert inker_mode.paste_from_os(ctx, tab) is True
    first = tab.doc.clipboard.take()
    assert first is not None and first[0][0, 0, 0] == 9

    # Copy a fresh selection inside the app. This never touches the OS
    # clipboard -- the stale screenshot above is still sitting there.
    tab.doc.stack.active.pixels[..., 3] = 255
    tab.doc.stack.active.pixels[0:2, 0:2] = (200, 0, 0, 255)
    tab.doc.select_all()
    assert tab.doc.copy()

    # Ctrl+V again: the OS clipboard has not changed since this door last
    # looked at it, so it must not clobber the copy just made.
    assert inker_mode.paste_from_os(ctx, tab) is True
    taken = tab.doc.clipboard.take()
    assert taken is not None
    assert taken[0][0, 0, 0] == 200, (
        "the Ctrl+C copy was clobbered by a re-grabbed stale screenshot"
    )


def test_paste_from_os_still_lands_a_genuinely_new_screenshot(monkeypatch):
    """The fix must not cost the ordinary case: a *different* OS image still
    lands, even after an earlier one was already pulled in."""
    from PIL import Image, ImageGrab

    first_shot = Image.new("RGBA", (4, 4), (1, 1, 1, 255))
    monkeypatch.setattr(ImageGrab, "grabclipboard", lambda: first_shot)
    tab = _tab()
    ctx = _PasteCtx()
    assert inker_mode.paste_from_os(ctx, tab) is True

    second_shot = Image.new("RGBA", (4, 4), (2, 2, 2, 255))
    monkeypatch.setattr(ImageGrab, "grabclipboard", lambda: second_shot)
    assert inker_mode.paste_from_os(ctx, tab) is True
    taken = tab.doc.clipboard.take()
    assert taken is not None and taken[0][0, 0, 0] == 2


# --- inker-mode-03: "go to next conflict" and the frame it lands on ---------


def _atlas(shade: int = 0) -> np.ndarray:
    atlas = np.zeros((CELL, FRAMES * CELL, 4), dtype=np.uint8)
    atlas[..., 3] = 255
    for i in range(FRAMES):
        atlas[:, i * CELL : (i + 1) * CELL, 0] = 30 + i * 40 + shade
    return atlas


def _sheet_cells():
    return [{"x": i * CELL, "y": 0, "w": CELL, "h": CELL} for i in range(FRAMES)]


def _sheet_doc():
    anim = {
        "tags": [{"name": "walk_front", "start": 0, "end": FRAMES - 1, "loop": True}],
        "frames": [],
    }
    doc = document_from_sheet(_atlas(), _sheet_cells(), anim, source={"job": "J", "sheet": "S"})
    doc.history.clear()
    return doc


def test_go_to_next_conflict_rebuilds_the_stack_for_the_new_frame():
    """The 2026-09-26 audit, finding inker-mode-03: this set
    ``doc.anim.current`` by hand -- never re-materialising ``stack`` for the
    new frame and never committing a floating buffer left over on the old
    one -- so a click landed strokes in the old frame's layers. Fixed to go
    through ``Document.set_current_frame``.
    """
    doc = _sheet_doc()
    doc.sheet_base.conflicts.add(doc.anim.frames[2].uid)
    ctx, state, tab = _session(doc)
    assert doc.anim.current == 0

    doc.select_all()
    assert doc.lift()
    assert doc.floating is not None, "sanity: a floating buffer is open on frame 0"

    assert inker_ops.run(ctx, inker_ops.get("sheet_conflict_next")) is True

    assert doc.anim.current == 2
    assert doc.floating is None, (
        "the floating buffer left on frame 0 must be committed before the jump"
    )
    frame2_layer = doc.anim.cels[(doc.anim.tracks[0].uid, doc.anim.frames[2].uid)]
    assert doc.stack.active is frame2_layer, (
        "the stack must be re-materialised for the frame just switched to"
    )


def test_go_to_next_conflict_refuses_while_the_document_is_busy():
    """Same finding: no busy check meant a click mid-save or mid-playback
    still moved the playhead and committed whatever was floating."""
    doc = _sheet_doc()
    doc.sheet_base.conflicts.add(doc.anim.frames[2].uid)
    ctx, state, tab = _session(doc)
    tab.playing = True

    assert inker_ops.run(ctx, inker_ops.get("sheet_conflict_next")) is False
    assert doc.anim.current == 0


# --- inker-panes-02: Filter on a tilemap layer -------------------------------


def _tilemap_doc():
    doc = inker.Document.blank(16, 16)
    tile = np.zeros((16, 16, 4), dtype=np.uint8)
    tile[..., 3] = 255
    stack = np.stack([np.zeros((16, 16, 4), dtype=np.uint8), tile], axis=0)
    slot = doc.add_tileset(strip(stack))
    doc.add_tilemap_layer(slot.uid)
    return doc


def test_open_filter_on_a_tilemap_layer_says_why_instead_of_raising():
    """The 2026-09-26 audit, finding inker-panes-02: Edit > Filter stayed lit
    on a tilemap layer, and ``begin_filter`` raises a bare ``ValueError`` past
    ``bridge.popups`` into the canvas pane's own frame. The menu row is now
    greyed with a reason, and the door itself no longer forwards the raise.
    """
    doc = _tilemap_doc()
    ctx, state, tab = _session(doc)

    op = inker_ops.get("filter")
    assert op.enabled(state, tab) is False
    assert "tilemap" in inker_ops.reason_for(op, state, tab).lower()
    assert inker_ops.run(ctx, op) is False

    # Belt: a caller that reaches the pane's own door directly (bypassing the
    # greyed menu row) must still get a toast, never a raised exception.
    inker_bridge._open_filter(ctx, tab)
    assert ctx.state.toasts, "the refusal must say why, not raise past this door"
    assert ctx.state.toasts[-1].level == "warn"
    assert "tilemap" in ctx.state.toasts[-1].text.lower()
