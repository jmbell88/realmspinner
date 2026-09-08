"""The Convert popup also enters indexed *mode*, with a dither.

The menu row that opens it is called "Colour mode...", and what it did was
snap the pixels onto a palette while leaving the document in RGB. Meanwhile the
mode buttons entered indexed mode with ``"nearest"`` hard-coded and no way to
ask for anything else -- so the one conversion in the app that changes mode was
the one conversion with no dither.

One popup answers both now. ``convert_mode`` says which question it is asking.

**``apply_convert`` submits and lands rather than converting inline**, since
the 2026-09-08 audit (finding inker-01) found the whole-document dither
running synchronously on the button press with the pygame frame loop blocked
for as long as it took. ``_Ctx`` below is ``tests/inker/test_flourish_ops.py``'s
fake -- it runs the submitted job inline and hands the result to
``inker_mode.on_task_done`` exactly as the app's ``TaskRunner`` would, so
these tests exercise the real route (submit -> ``_done_convert`` -> land)
rather than the synchronous call that no longer exists.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from warlock.studio import inker, inker_mode
from warlock.studio.inker_state import InkerDoc, InkerState
from warlock.studio.panes import inker_bridge
from warlock.studio.tasks import Done


class _Ctx:
    """Runs a submitted job inline and lands it through ``on_task_done``,
    the same shape ``tests/inker/test_flourish_ops.py`` uses for the other
    Inker async doors."""

    def __init__(self) -> None:
        self.state = SimpleNamespace(inker=InkerState())
        self.toasts: list[tuple[str, str]] = []
        self.cache = SimpleNamespace(invalidate=lambda: None)

    def toast(self, text: str, level: str = "info", *_: Any) -> None:
        self.toasts.append((text, level))

    def submit(self, key: str, fn, *args: Any, **kwargs: Any) -> bool:
        try:
            done = Done(key=key, result=fn(*args, **kwargs))
        except Exception as exc:  # noqa: BLE001 -- the runner reports, never raises
            done = Done(key=key, error=exc)
        inker_mode.on_task_done(self, done)
        return True


def _ramp_tab(ctx: _Ctx) -> InkerDoc:
    doc = inker.Document.blank(16, 4)
    ramp = np.linspace(0, 255, 16).astype("uint8")
    doc.stack.active.pixels[:, :, :3] = ramp[None, :, None]
    doc.stack.active.pixels[:, :, 3] = 255
    doc.invalidate_all()
    tab = InkerDoc(doc=doc, title="ramp")
    ctx.state.inker.add(tab)
    return tab


def _session(ctx: _Ctx, tab: InkerDoc, *, mode: str, method: str) -> None:
    state = ctx.state.inker
    assert tab.doc.begin_convert()
    state.convert_uid = tab.uid
    state.convert_mode = mode
    state.convert_method = method
    state.convert_max = 4
    state.convert_table = tab.doc.built_palette(4)


def test_applying_a_mode_session_enters_indexed_mode():
    ctx = _Ctx()
    tab = _ramp_tab(ctx)
    _session(ctx, tab, mode="indexed", method="nearest")
    assert inker_bridge.apply_convert(ctx, tab)  # accepted for submission
    assert tab.doc.is_indexed
    assert tab.saving is False  # landed, not left locked


def test_a_mode_session_uses_the_matrix_that_was_chosen():
    ctx = _Ctx()
    tab = _ramp_tab(ctx)
    _session(ctx, tab, mode="indexed", method="floyd-steinberg")
    assert inker_bridge.apply_convert(ctx, tab)
    plane = tab.doc.composite[..., 0]
    assert not np.array_equal(plane[0], plane[1])


def test_a_plain_session_still_snaps_and_leaves_the_mode_alone():
    ctx = _Ctx()
    tab = _ramp_tab(ctx)
    _session(ctx, tab, mode="", method="nearest")
    assert inker_bridge.apply_convert(ctx, tab)
    assert not tab.doc.is_indexed
    assert len(np.unique(tab.doc.composite[..., 0])) <= 4


def test_applying_closes_the_session_either_way():
    ctx = _Ctx()
    tab = _ramp_tab(ctx)
    _session(ctx, tab, mode="indexed", method="nearest")
    inker_bridge.apply_convert(ctx, tab)
    assert ctx.state.inker.convert_uid == ""
    assert ctx.state.inker.convert_mode == ""
