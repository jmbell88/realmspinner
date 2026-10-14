"""The 2026-10-14 audit, Muse's tray: order, trash, the Extend reason, the tags
bound, the wheel, and drawing only the cards in view.

Findings muse-mode-01, -02, -03, -08, -14 and -17. ``test_muse_panes_smoke``'s
stub cache is oldest-first, which the real ``JobsCache`` is not (``JobStore.list``
orders ``created_at DESC`` and ``adopt`` keeps ``top + old``), so every ordering
claim here is made over a *newest-first* cache.
"""

from __future__ import annotations

import contextlib
from typing import Any

from realmspinner.studio.modes.muse import mode as muse_mode
from realmspinner.studio.modes.muse.ui.panes import results as muse_results

from . import test_muse_panes_smoke as _smoke

_ctx = _smoke._ctx
_take = _smoke._take
#: The smoke file's bare-imgui-context fixture, re-registered under its own name.
frames = _smoke.frames


def test_the_tray_lists_newest_take_first_over_a_newest_first_job_cache(tmp_path):
    # ``cache.jobs`` is newest-first; the tray must not flip it.
    ctx = _ctx(tmp_path, [_take("new"), _take("mid"), _take("old")])
    assert [job["id"] for job in muse_results.plan_for(ctx)] == ["new", "mid", "old"]


def test_a_trashed_take_is_not_in_the_tray(tmp_path):
    trashed = _take("gone")
    trashed["deleted_at"] = "2026-10-14T10:00:00"
    ctx = _ctx(tmp_path, [_take("kept"), trashed])
    assert [job["id"] for job in muse_results.plan_for(ctx)] == ["kept"]
    only_trash = _ctx(tmp_path, [trashed])
    assert muse_results.should_draw(only_trash) is False


def test_a_disabled_extend_entry_shows_its_reason_on_hover(frames, tmp_path, monkeypatch):
    from imgui_bundle import imgui

    job = _take("long")
    job["params"] = {"duration": 300.0, "actual_duration": 300.0}
    ctx = _ctx(tmp_path, [job])
    seen: dict[str, dict[str, Any]] = {}

    def fake_menu_item(label, shortcut="", selected=False, enabled=True, **kwargs):
        seen[label] = {"enabled": enabled, **kwargs}
        return False, False

    monkeypatch.setattr(muse_results.controls, "menu_item", fake_menu_item)

    def build() -> None:
        imgui.open_popup("muse-more/long")
        muse_results._derive_menu(ctx, "long", True, "done")

    frames(build)
    extend = seen["Extend"]
    assert extend["enabled"] is False
    # imgui answers ``is_item_hovered()`` False over a disabled item, so the
    # sentence has to travel through ``controls.menu_item``'s own ``reason=``.
    assert "extend ceiling" in extend.get("reason", "")


def test_derive_edit_tags_field_allows_the_doors_prompt_length(frames, monkeypatch, tmp_path):
    from realmspinner.service.validation import MAX_PROMPT

    ctx = _ctx(tmp_path)
    captured: dict[str, int] = {}

    def fake_multiline(label, value, height, max_length, **kwargs):
        captured[label] = max_length
        return value

    monkeypatch.setattr(muse_results.widgets, "multiline", fake_multiline)
    derive = {"edit_prompt": "", "edit_lyrics": ""}
    frames(lambda: muse_results._derive_field(ctx, derive, "edit_prompt", "edit", 60.0))
    assert captured["##muse-edit_prompt"] == MAX_PROMPT


def test_the_muse_tray_pane_scrolls_with_the_wheel(frames, tmp_path, monkeypatch):
    from realmspinner.studio import layout as layout_mod
    from realmspinner.studio.modes.muse.ui import workspace as muse_workspace
    from realmspinner.studio.shell import frame as shell_frame

    recorded: dict[str, int] = {}

    @contextlib.contextmanager
    def fake_pane(pane_id, size, role=None, **kwargs):
        recorded[pane_id] = int(kwargs.get("window_flags", 0))
        yield False

    monkeypatch.setattr(layout_mod, "pane", fake_pane)
    monkeypatch.setattr(shell_frame, "_column_boundary", lambda *a, **k: None)

    class _App(muse_workspace.MuseWorkspace):
        pass

    app = _App()
    app.app_ctx = _ctx(tmp_path)
    app.layouts = None
    frames(app._muse_workspace)

    from imgui_bundle import imgui

    # With this flag a child that overflows scrolls 0.0 on the wheel, so the
    # tray's later cards were reachable only by dragging the scrollbar.
    assert recorded["muse-centre"] & imgui.WindowFlags_.no_scroll_with_mouse.value == 0


def test_muse_grid_draws_only_visible_rows(frames, tmp_path, monkeypatch):
    jobs = [_take(f"t{n:03d}") for n in range(120)]
    ctx = _ctx(tmp_path, jobs)
    drawn: list[str] = []
    # A card is stubbed to draw nothing, so give the stub the real footprint.
    from imgui_bundle import imgui

    def card(c, job, width):
        drawn.append(str(job["id"]))
        imgui.dummy((width, muse_results.sp(muse_results.CARD_H)))

    monkeypatch.setattr(muse_results, "_card", card)
    frames(lambda: muse_results._grid(ctx, jobs))
    assert 0 < len(drawn) < len(jobs) // 2, len(drawn)
    assert drawn[0] == "t000"


def test_muse_grid_never_clips_the_selected_take(frames, tmp_path, monkeypatch):
    jobs = [_take(f"t{n:03d}") for n in range(120)]
    ctx = _ctx(tmp_path, jobs)
    muse_mode.ensure(ctx).selected_job = "t110"
    drawn: list[str] = []
    from imgui_bundle import imgui

    def card(c, job, width):
        drawn.append(str(job["id"]))
        imgui.dummy((width, muse_results.sp(muse_results.CARD_H)))

    monkeypatch.setattr(muse_results, "_card", card)
    frames(lambda: muse_results._grid(ctx, jobs))
    assert "t110" in drawn


def test_muse_grid_draws_a_short_tray_whole(frames, tmp_path, monkeypatch):
    jobs = [_take(f"t{n:02d}") for n in range(12)]
    ctx = _ctx(tmp_path, jobs)
    drawn: list[str] = []
    from imgui_bundle import imgui

    def card(c, job, width):
        drawn.append(str(job["id"]))
        imgui.dummy((width, muse_results.sp(muse_results.CARD_H)))

    monkeypatch.setattr(muse_results, "_card", card)
    frames(lambda: muse_results._grid(ctx, jobs))
    assert len(drawn) == 12
