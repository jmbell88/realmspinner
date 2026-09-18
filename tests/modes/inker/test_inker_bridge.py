"""Landing a finished regeneration into a layer stack an encode may be walking.

``land_inpaint`` (``modes/inker/ui/panes/bridge.py``) is the frame-thread half of a
masked regenerate: the picture was decoded and resized on a task thread, and
this call blends it in through ``apply_pixels``, which autovivifies a cel and
pushes an undo step. It had no ``tab.busy`` check anywhere in its chain, while
the sibling ``_done_tileset_import`` (``inker_mode.py``) explicitly re-checks
``not target.busy`` before the analogous write, citing "``add_tileset`` pushes
a history step into a stack an encode is walking". The 2026-09-07 audit
(inker-02) reproduced the same hazard for a regeneration: with the tab saving,
the write went through anyway, leaving a ``stack.xml`` that can disagree with
its own PNG members.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from warlock.kernels import pixel as inker
from warlock.studio.modes.inker.state import InkerDoc, InkerState
from warlock.studio.modes.inker.ui.panes import bridge as inker_bridge


class _Ctx:
    """Enough of ``Ctx`` for ``land_inpaint``: the document lives on
    ``ctx.state.inker`` and refusals go out through ``ctx.toast``."""

    def __init__(self, state: InkerState) -> None:
        self.state = SimpleNamespace(inker=state)
        self.toasts: list[tuple[str, str]] = []

    def toast(self, message: str, level: str = "info") -> None:
        self.toasts.append((message, level))


def _tab() -> InkerDoc:
    doc = inker.Document.blank(8, 8)
    return InkerDoc(doc=doc, title="t", saved_head=0)


def _pending(tab: InkerDoc) -> dict:
    return {
        "tab_uid": tab.uid,
        "layer_uid": tab.doc.stack.active.uid,
        "box": (0, 0, 4, 4),
        "weight": None,
    }


def test_land_inpaint_is_refused_while_the_tab_is_saving():
    tab = _tab()
    tab.saving = True
    state = InkerState()
    state.add(tab)
    ctx = _Ctx(state)
    pending = _pending(tab)
    pixels = np.full((4, 4, 4), 255, dtype=np.uint8)
    head_before = tab.doc.history.head

    ok = inker_bridge.land_inpaint(ctx, pending, pixels)

    assert ok is False
    assert tab.doc.history.head == head_before, "busy tab: no undo step pushed"
    assert len(tab.doc.history) == 0, "busy tab: no undo step pushed"
    assert tuple(tab.doc.stack.active.pixels[0, 0]) == (0, 0, 0, 0), (
        "busy tab: the pixels must not change under a walking encode"
    )
    assert ctx.toasts and ctx.toasts[-1][1] == "warn"


def test_land_inpaint_still_lands_when_the_tab_is_not_busy():
    """The guard must refuse only the busy case -- the ordinary regeneration
    still has to land, or the fix for inker-02 would just be a new way to
    drop every regeneration."""
    tab = _tab()
    state = InkerState()
    state.add(tab)
    ctx = _Ctx(state)
    pending = _pending(tab)
    pixels = np.full((4, 4, 4), 255, dtype=np.uint8)
    assert len(tab.doc.history) == 0

    ok = inker_bridge.land_inpaint(ctx, pending, pixels)

    assert ok is True
    # ``head`` is a process-global serial (``undo._serials``), not a per-doc
    # counter, so a fresh document's step count -- not the serial's absolute
    # value -- is what proves exactly one edit landed.
    assert len(tab.doc.history) == 1
    assert tuple(tab.doc.stack.active.pixels[0, 0]) == (255, 255, 255, 255)

