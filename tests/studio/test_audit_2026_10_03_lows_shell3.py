"""The 2026-10-03 audit's Low findings shell-40, 48, 49, 56, 58, 60, 63, 79 and 80.

Each test's name is the claim. None needs a GL context: the drawing ones stand
imgui's calls in with recorders, and the rest read the module under test as data.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


# -- shell-40: guard.HISTORY ----------------------------------------------------


def test_an_alternating_failure_does_not_grow_history_without_bound(monkeypatch):
    """``ok()`` zeroes a non-tripped breaker on every clean draw, so a pane failing on
    alternate frames never trips -- and ``_record`` kept appending a formatted
    traceback to ``HISTORY`` for each one, for the whole session."""
    from realmspinner.studio import guard

    monkeypatch.setattr(guard, "HISTORY", [])
    monkeypatch.setattr(guard, "_BREAKERS", {})
    monkeypatch.setattr(guard, "FRAME_FAILURES", [])
    monkeypatch.setattr(guard, "_CTX", None)
    for _ in range(1000):
        try:
            raise ValueError("flickers")
        except ValueError as exc:
            guard._record("pane", "Pane", exc)
        guard.ok("pane")
    assert not guard.tripped("pane"), "the alternating pane is the case that never trips"
    assert 0 < len(guard.failures()) < 1000
    assert guard.failures()[-1].kind == "ValueError", "the newest failure is the one kept"


# -- shell-48: key releases -----------------------------------------------------


class _KeyIO:
    def __init__(self) -> None:
        self.keys: list[tuple[object, bool]] = []
        self.want_text_input = False

    def add_key_event(self, key: object, down: bool) -> None:
        self.keys.append((key, down))


def test_a_key_released_after_the_arrows_are_reserved_still_reaches_imgui(monkeypatch):
    """A nav key pressed before a surface reserved the arrows and released after it
    did was never released in imgui, which then auto-repeated its navigation."""
    import pygame

    from realmspinner.studio import imgui_backend

    io = _KeyIO()
    monkeypatch.setattr(imgui_backend.imgui, "get_io", lambda: io)
    down = SimpleNamespace(type=pygame.KEYDOWN, key=pygame.K_DOWN)
    up = SimpleNamespace(type=pygame.KEYUP, key=pygame.K_DOWN)
    try:
        imgui_backend.reserve_nav_keys(False)
        imgui_backend.process_event(down)
        assert [d for _k, d in io.keys] == [True]
        imgui_backend.reserve_nav_keys(True)
        imgui_backend.process_event(up)
        assert [d for _k, d in io.keys] == [True, False], "the release must get through"
        # And the reservation still swallows a *press*, which is its whole job.
        imgui_backend.process_event(down)
        assert [d for _k, d in io.keys] == [True, False]
    finally:
        imgui_backend.reserve_nav_keys(False)


# -- shell-49: a deeply nested settings file -------------------------------------


def test_a_deeply_nested_settings_file_is_kept_aside_and_defaults_load(tmp_path):
    """``json.loads`` raises RecursionError, not ValueError, on a deeply nested file; it
    escaped ``Settings.load`` before there is a window to draw anything."""
    from realmspinner.studio import settings

    path = tmp_path / settings.FILENAME
    path.write_text("[" * 200_000 + "]" * 200_000, encoding="utf-8")
    loaded = settings.Settings.load(tmp_path)
    assert loaded.data == {}
    assert loaded.notice and "reset to defaults" in loaded.notice
    assert not path.exists(), "the file is kept aside, not left to be re-read next launch"
    assert list(tmp_path.glob("studio_settings.corrupt-*.json"))


# -- shell-56: the Screenshot button says why it is greyed -----------------------


def test_the_screenshot_button_says_why_it_is_greyed(monkeypatch):
    from realmspinner.studio.panes import overlay

    calls: list[dict[str, Any]] = []

    def icon_button(icon, tooltip, **kwargs):
        calls.append({"icon": icon, "tooltip": tooltip, **kwargs})
        return False

    widgets = overlay.widgets
    monkeypatch.setattr(widgets, "icon_button", icon_button)
    monkeypatch.setattr(widgets, "same_line_or_wrap", lambda *_a, **_k: None)
    monkeypatch.setattr(widgets, "button_width", lambda *_a, **_k: 0.0)
    monkeypatch.setattr(overlay.manual_render, "help_button_inline", lambda *_a, **_k: None)
    monkeypatch.setattr(overlay, "offers_inker", lambda *_a: False)
    monkeypatch.setattr(overlay, "shows_tiled", lambda *_a: False)
    monkeypatch.setattr(overlay, "_has_content", lambda *_a: False)
    monkeypatch.setattr(overlay, "_texture_losses", lambda *_a: None)
    viewer = SimpleNamespace(has_model=False, reference=object(), model=None)
    ctx = SimpleNamespace(
        state=SimpleNamespace(comparing=None),
        viewer=viewer,
        job=lambda: None,
        clear_viewport=None,
    )
    overlay.toolbar(ctx)
    (shot,) = [c for c in calls if c["tooltip"].startswith("Screenshot")]
    assert shot["enabled"] is False
    assert shot.get("reason"), "a greyed control must say what it is waiting for"
    assert "mesh" in shot["reason"].lower()


# -- shell-58: the rail's budget ---------------------------------------------------


def test_the_rails_height_budget_equals_the_height_it_draws():
    """Gaps are drawn only *between* groups and captions only over labelled ones; the
    budget used to count ``len(groups)`` of each, reserving a gap and a caption row
    that nothing drew. (The helper did not exist before the fix, so this fails
    pre-fix on its absence; ``draw`` calls it for both the budget and the drawn gap.)"""
    from realmspinner.studio import modes, rail

    gaps, captions = rail.section_counts(modes.RAIL_GROUPS, modes.RAIL_GROUP_LABELS)
    drawn_gaps = sum(1 for index, _group in enumerate(modes.RAIL_GROUPS) if index)
    drawn_captions = sum(
        1 for index, _group in enumerate(modes.RAIL_GROUPS) if modes.RAIL_GROUP_LABELS[index]
    )
    assert (gaps, captions) == (drawn_gaps, drawn_captions)
    assert gaps == len(modes.RAIL_GROUPS) - 1
    assert captions < len(modes.RAIL_GROUPS), "the footer group has no caption"
    assert rail.section_counts((("a",),), ("",)) == (0, 0)
    assert rail.section_counts((), ()) == (0, 0)


# -- shell-60: staging files a killed write left behind ----------------------------


def test_snapshot_sweeps_staging_files_a_killed_write_left_behind(tmp_path):
    from realmspinner.studio import journal

    ctx = SimpleNamespace(
        svc=SimpleNamespace(config=SimpleNamespace(autosave_dir=tmp_path)),
        state=SimpleNamespace(recovery=None),
    )
    payload = tmp_path / ".Sketch-s1.rsk.tmp"
    sidecar = tmp_path / f".Sketch-s1.rsk{journal.META_SUFFIX}.tmp"
    for orphan in (payload, sidecar):
        orphan.write_bytes(b"half a document")
        old = time.time() - 3600
        os.utime(orphan, (old, old))
    # A real copy and a staging file that is *fresh* (another instance mid-write)
    # must both survive.
    kept = tmp_path / "Kept-k1.rsk"
    kept.write_bytes(b"whole")
    journal.meta_path(kept).write_text(
        json.dumps({"version": journal.VERSION, "kind": "none", "title": "Kept", "at": 1.0}),
        encoding="utf-8",
    )
    fresh = tmp_path / ".Live-l1.rsk.tmp"
    fresh.write_bytes(b"being written right now")

    journal.snapshot(ctx)

    assert not payload.exists() and not sidecar.exists()
    assert fresh.exists(), "a staging file another instance is writing is not an orphan"
    assert kept.exists() and journal.meta_path(kept).exists()


# -- shell-63: manual 23 and the collapsed Open in... header -----------------------


def test_manual_23_describes_the_collapsed_open_in_header():
    chapter = ROOT / "docs" / "manual" / "23-generating-meshes.md"
    text = " ".join(chapter.read_text("utf-8").split())
    assert "**Open in...**" in text
    pane = ROOT / "src" / "realmspinner" / "studio" / "panes" / "inspector.py"
    source = pane.read_text("utf-8")
    assert 'widgets.header("Open in..."' in source, "the chapter names the header the pane draws"
    assert "visible at all five stages" not in source


# -- shell-79: the thumbnail cache's negative cache --------------------------------


class _Tex:
    def __init__(self, size):
        self.size = size
        self.filter = None
        self.repeat_x = self.repeat_y = True

    def release(self):
        pass


class _GL:
    NEAREST, LINEAR = "n", "l"

    def texture(self, size, components, data):
        return _Tex(size)


def _png(path: Path) -> None:
    Image.new("RGBA", (8, 8), (1, 2, 3, 255)).save(path)


def test_a_transient_open_failure_is_retried_rather_than_cached_as_missing(tmp_path, monkeypatch):
    """A sharing violation while thumb.png is replaced was cached under (job, mtime...)
    and never retried, so the asset kept a blank thumbnail until its mtime changed."""
    from realmspinner.studio import textures

    path = tmp_path / "thumb.png"
    _png(path)
    real_open = Image.open
    state = {"fail": True}

    def flaky(*args, **kwargs):
        if state["fail"]:
            raise PermissionError(32, "The process cannot access the file")
        return real_open(*args, **kwargs)

    monkeypatch.setattr(Image, "open", flaky)
    cache = textures.ThumbnailCache(_GL())
    assert cache.get("job", path) is None
    state["fail"] = False
    for _ in range(textures.RETRY_AFTER_FRAMES + 1):
        cache.begin_frame()
    assert cache.get("job", path) is not None, "the failure was not transient-aware"


def test_a_file_that_cannot_be_an_image_is_still_cached_as_missing(tmp_path, monkeypatch):
    from realmspinner.studio import textures

    path = tmp_path / "thumb.png"
    path.write_bytes(b"this is not an image")
    opened: list[int] = []
    real_open = Image.open

    def counting(*args, **kwargs):
        opened.append(1)
        return real_open(*args, **kwargs)

    monkeypatch.setattr(Image, "open", counting)
    cache = textures.ThumbnailCache(_GL())
    assert cache.get("job", path) is None
    for _ in range(200):
        cache.begin_frame()
        assert cache.get("job", path) is None
    assert len(opened) == 1, "a decode that can never succeed is not retried"


# -- shell-80: a disabled control with no reason ------------------------------------


def test_a_disabled_control_with_no_reason_does_not_show_its_live_tooltip(monkeypatch):
    """``disabled_button``'s contract is that the tooltip is shown only while the control
    is live; ``_finish_item`` showed it on a greyed one too, explaining a refusal with
    the text describing the action being refused."""
    from realmspinner.studio import controls

    shown: list[str] = []
    imgui = controls.imgui
    monkeypatch.setattr(imgui, "is_item_hovered", lambda *_a, **_k: True)
    monkeypatch.setattr(imgui, "is_item_focused", lambda *_a, **_k: False)
    monkeypatch.setattr(imgui, "set_tooltip", shown.append)

    controls._finish_item(tooltip="Delete the layer", enabled=False, kind="button")
    assert shown == []

    controls._finish_item(tooltip="Delete the layer", reason="Nothing to delete", enabled=False,
                          kind="button")
    assert shown == ["Nothing to delete"]

    shown.clear()
    controls._finish_item(tooltip="Delete the layer", enabled=True, kind="button")
    assert shown == ["Delete the layer"]


@pytest.fixture(autouse=True)
def _quiet_reserve():
    from realmspinner.studio import imgui_backend

    yield
    imgui_backend.reserve_nav_keys(False)
