"""The 2026-10-14 audit's Muse cleanup round (fixer C5): trashed players, tray scroll."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from realmspinner.studio.modes.muse import mode as muse_mode
from realmspinner.studio.modes.muse.ui.panes import results as muse_results

from . import test_muse_panes_smoke as _smoke
from .test_muse_mode import FakeCtx
from .test_muse_player import _loaded
from .test_muse_player import device as _device

device = _device
frames = _smoke.frames
_ctx = _smoke._ctx
_take = _smoke._take


@pytest.fixture
def ctx(tmp_path):
    return FakeCtx(tmp_path)


# --- muse-02: a take in the trash has no player -----------------------------------


def test_sync_drops_the_player_of_a_take_that_has_gone_to_the_trash(ctx, device):
    _loaded(ctx, job="target")
    ctx.cache = SimpleNamespace(
        jobs=[{"id": "target", "deleted_at": "2026-10-14T10:00:00"}, {"id": "y"}], total=2
    )
    muse_mode.sync(ctx)
    assert muse_mode.player(ctx) is None


def test_sync_drops_a_trashed_takes_player_even_when_the_cache_is_only_a_window(ctx, device):
    # The row itself is the evidence, so the window need not hold the library.
    _loaded(ctx, job="target")
    ctx.cache = SimpleNamespace(
        jobs=[{"id": "target", "deleted_at": "2026-10-14T10:00:00"}], total=500
    )
    muse_mode.sync(ctx)
    assert muse_mode.player(ctx) is None


def test_sync_keeps_the_player_of_a_take_that_is_not_trashed(ctx, device):
    _loaded(ctx, job="target")
    ctx.cache = SimpleNamespace(jobs=[{"id": "target", "deleted_at": None}], total=1)
    muse_mode.sync(ctx)
    assert muse_mode.player(ctx) is not None


# --- muse-25: the tray scrolls to a keyboard selection ----------------------------


def test_the_tray_consumes_the_scroll_request_a_keyboard_selection_raises(
    frames, tmp_path, monkeypatch
):
    jobs = [_take(f"t{n:03d}") for n in range(120)]
    ctx = _ctx(tmp_path, jobs)
    state = muse_mode.ensure(ctx)
    state.selected_job = "t110"
    state.scroll_to_selected = True
    targets: list[tuple[float, float]] = []
    real = muse_results.imgui.set_scroll_from_pos_y

    def spy(local_y, ratio=0.5):
        targets.append((float(local_y), float(ratio)))
        real(local_y, ratio)

    monkeypatch.setattr(muse_results.imgui, "set_scroll_from_pos_y", spy)
    frames(lambda: muse_results._grid(ctx, jobs))

    assert len(targets) == 1 and targets[0][1] == 0.5
    assert state.scroll_to_selected is False
    # The selected card sits far down the tray; the target is in that region,
    # not at the top.
    assert targets[0][0] > muse_results.sp(muse_results.CARD_H) * 5


def test_the_scroll_request_does_not_outlive_a_selection_that_is_not_drawn(frames, tmp_path):
    jobs = [_take(f"t{n:02d}") for n in range(5)]
    ctx = _ctx(tmp_path, jobs)
    state = muse_mode.ensure(ctx)
    state.selected_job = "not-in-the-window"
    state.scroll_to_selected = True
    frames(lambda: muse_results._grid(ctx, jobs))
    assert state.scroll_to_selected is False
