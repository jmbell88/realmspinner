"""``open_pixels``' palette and ``on_open`` hooks (Clay's "Edit texture in Inker").

Each test's name is the claim. The harness runs ``submit`` synchronously and
hands the result to ``mode._done_open`` the way ``on_task_done`` does, so the
task-thread half and the frame-thread half are both exercised.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from realmspinner.kernels.pixel import PICO8
from realmspinner.kernels.pixel import palettes as pal
from realmspinner.studio.modes.inker import mode as inker_mode
from realmspinner.studio.modes.inker import opening as inker_open
from realmspinner.studio.modes.inker.state import InkerState
from realmspinner.studio.state import AppState
from realmspinner.studio.tasks import Done

_PICO_HEX = [
    *("000000", "1D2B53", "7E2553", "008751", "AB5236", "5F574F", "C2C3C7", "FFF1E8"),
    *("FF004D", "FFA300", "FFEC27", "00E436", "29ADFF", "83769C", "FF77A8", "FFCCAA"),
]


class _Ctx:
    def __init__(self) -> None:
        self.state = AppState()
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


@pytest.fixture(autouse=True)
def _no_settings(monkeypatch):
    # ``_adopt`` persists swatches and a recents entry through ``ctx.settings``;
    # neither is under test and the harness has no settings store.
    monkeypatch.setattr(inker_mode, "persist", lambda ctx: None)
    monkeypatch.setattr(inker_mode, "remember_path", lambda ctx, path: None)


def _open(ctx: _Ctx, pixels: Any, **kwargs: Any) -> Any:
    inker_open.open_pixels(ctx, pixels, **kwargs)
    assert ctx.done is not None
    inker_mode._done_open(ctx, ctx.state.inker, ctx.done)
    return ctx.state.inker.docs[-1]


def _off_palette(width: int = 4, height: int = 4) -> np.ndarray:
    px = np.zeros((height, width, 4), dtype=np.uint8)
    px[..., :3] = (200, 30, 60)  # near FF004D but not on any entry
    px[..., 3] = 255
    return px


def test_pico8_is_sixteen_distinct_opaque_colours_in_the_consoles_order():
    assert len(PICO8) == 16
    assert len(set(PICO8)) == 16
    assert [f"{r:02X}{g:02X}{b:02X}" for r, g, b, _a in PICO8] == _PICO_HEX
    assert all(len(c) == 4 and c[3] == 255 for c in PICO8)
    assert PICO8 is pal.PICO8


def test_open_pixels_with_a_palette_opens_a_palette_locked_document_on_exactly_those_colours():
    ctx = _Ctx()
    tab = _open(ctx, _off_palette(), title="Brick", palette=PICO8)
    doc = tab.doc
    assert doc.is_palette_locked
    assert [tuple(c) for c in doc.palette] == list(PICO8)
    flat = doc.flatten(matte=False)
    colours = {tuple(int(v) for v in px) for px in flat.reshape(-1, 4)}
    assert colours <= set(PICO8)
    # Every pixel moved onto the Euclidean-nearest entry, not left at (200, 30, 60).
    probe = (200, 30, 60)
    nearest = min(PICO8, key=lambda c: sum((c[i] - probe[i]) ** 2 for i in range(3)))
    assert colours == {nearest}
    assert nearest != (200, 30, 60, 255)
    assert tab.title == "Brick"


def test_the_palette_snap_leaves_the_tab_clean_so_an_untouched_pull_is_not_saved():
    ctx = _Ctx()
    tab = _open(ctx, _off_palette(), palette=PICO8)
    # The snap is a real history step (undoable), and the saved head was taken
    # after it: closing the untouched tab must not ask to save.
    assert tab.doc.history.head >= 1
    assert tab.saved_head == tab.doc.history.head


def test_the_palette_snap_keeps_transparent_and_partial_alpha_pixels_as_they_were():
    px = _off_palette(3, 1)
    px[0, 0] = (12, 34, 56, 0)  # a hole in a cutout texture
    px[0, 1] = (200, 30, 60, 128)  # a soft edge
    ctx = _Ctx()
    tab = _open(ctx, px, palette=PICO8)
    flat = tab.doc.flatten(matte=False)
    assert flat[0, 0, 3] == 0
    assert flat[0, 1, 3] == 128
    assert flat[0, 2, 3] == 255
    assert tuple(int(v) for v in flat[0, 1, :3]) in {c[:3] for c in PICO8}


def test_on_open_is_called_once_with_the_adopted_tab_on_the_frame_thread():
    ctx = _Ctx()
    seen: list[Any] = []
    tab = _open(ctx, _off_palette(), palette=PICO8, on_open=seen.append)
    assert seen == [tab]
    assert tab.uid and tab.doc is not None
    assert ctx.state.inker.active_uid == tab.uid
    assert ctx.state.mode == "inker"


def test_an_on_open_that_raises_toasts_and_does_not_kill_the_frame():
    ctx = _Ctx()

    def boom(_tab: Any) -> None:
        raise RuntimeError("clay went away")

    tab = _open(ctx, _off_palette(), on_open=boom)
    assert tab in ctx.state.inker.docs
    assert any(level == "warn" for _text, level in ctx.toasts)


def test_open_pixels_without_palette_or_hook_is_plotters_unchanged_path():
    ctx = _Ctx()
    px = _off_palette()
    tab = _open(ctx, px, title="Atlas pull")
    assert ctx.keys == ["inker-open:pixels:Atlas pull"]
    assert not tab.doc.is_palette_locked
    assert not tab.doc.palette
    assert tab.doc.history.head == 0
    assert tab.saved_head == 0
    assert "on_open" not in ctx.done.result
    assert np.array_equal(tab.doc.flatten(matte=False), px)
    assert ctx.toasts == []


def test_open_pixels_takes_palette_and_on_open_keyword_only():
    with pytest.raises(TypeError):
        inker_open.open_pixels(SimpleNamespace(), np.zeros((1, 1, 4), np.uint8), "t", PICO8)  # type: ignore[misc]
