"""Poser (and Troupe, folded into it before) are gone, and nothing still names them.

The 2026-10-13 Clay rig plan folded Poser into Clay and removed the mode (T5);
T5.4 removed the character-sheet kinds it rendered. A mode key surviving in one
hand-kept table is the failure this file exists for: every table below once
carried ``"poser"`` or ``"troupe"`` and a table that is not re-read is how a
retired key came back as a rail button that opens nothing, a drop refusal for a
mode that is not there, or a pack that claims to unlock it.

Each test's name is the claim. The scans read the tables as data wherever one
exists, and the source of ``shell/frame.py`` as text only for the dispatch,
which has no table to read.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from realmspinner import packs
from realmspinner.kernels.manual.targets import HELP_TARGETS
from realmspinner.studio import asset_exits, main, mode_manifest, modes
from realmspinner.studio.modes.home.ui.panes import landing
from realmspinner.studio.shell import frame as frame_mod

RETIRED_MODE_KEYS = ("poser", "troupe")


@pytest.mark.parametrize("key", RETIRED_MODE_KEYS)
def test_a_retired_mode_is_not_a_mode(key):
    assert key not in modes.KEYS
    assert key not in {entry[0] for entry in modes.MODES}
    assert key not in modes.PURPOSE


@pytest.mark.parametrize("key", RETIRED_MODE_KEYS)
def test_a_retired_mode_is_not_on_the_rail(key):
    assert all(key not in group for group in modes.RAIL_GROUPS)
    assert key not in {k for group in modes.RAIL_GROUPS for k in group}


@pytest.mark.parametrize("key", RETIRED_MODE_KEYS)
def test_a_retired_mode_is_in_none_of_the_mode_groupings(key):
    assert key not in modes.WORK_MODES
    assert key not in modes.WORKSPACE_MODES
    assert key not in modes.NAV_KEY_MODES
    assert key not in modes.NAV_KEY_SURFACES


def test_the_rail_groups_still_partition_the_modes_exactly():
    """Removing a mode from the groups but not from ``MODES`` (or the reverse) is
    how a mode becomes a button with nowhere to go."""
    on_rail = [key for group in modes.RAIL_GROUPS for key in group]
    assert sorted(on_rail) == sorted(modes.KEYS)


@pytest.mark.parametrize("key", RETIRED_MODE_KEYS)
def test_a_retired_mode_has_no_drop_refusal(key):
    assert key not in main.DROP_REFUSALS


@pytest.mark.parametrize("key", RETIRED_MODE_KEYS)
def test_a_retired_mode_is_not_a_document_mode(key):
    assert mode_manifest.by_key(key) is None
    assert key not in {entry.key for entry in mode_manifest.DOC_MODES}
    assert all(key not in module for module in mode_manifest.journal_modules())
    assert key not in mode_manifest.export_table()


@pytest.mark.parametrize("key", RETIRED_MODE_KEYS)
def test_no_asset_exit_goes_to_a_retired_mode(key):
    assert key not in asset_exits._ICON_OVERRIDES
    source = Path(asset_exits.__file__).read_text(encoding="utf-8")
    assert f'Exit("{key}"' not in source
    assert f"modes.{key}" not in source


@pytest.mark.parametrize("key", RETIRED_MODE_KEYS)
def test_the_new_menu_offers_nothing_for_a_retired_mode(key):
    assert key not in {item[0] for item in landing.NEW_ITEMS}


@pytest.mark.parametrize("key", RETIRED_MODE_KEYS)
def test_no_manual_help_target_points_at_a_retired_mode(key):
    for name, (chapter, section) in HELP_TARGETS.items():
        assert key not in name, f"{name!r} is a help key for the retired {key} mode"
        assert key not in chapter, f"{name!r} points into a {key} chapter: {chapter!r}"
        assert key not in str(section or ""), f"{name!r} points at a {key} section"


@pytest.mark.parametrize("key", RETIRED_MODE_KEYS)
def test_no_pack_claims_to_unlock_a_retired_mode(key):
    for pack in packs.PACKS:
        assert key not in pack.modes, f"pack {pack.key!r} still lists {key!r}"


@pytest.mark.parametrize("key", RETIRED_MODE_KEYS)
def test_the_frame_loop_does_not_dispatch_to_a_retired_mode(key):
    """``_build_ui``'s dispatch ends in a bare ``else``: a leftover branch for a
    deleted mode is dead code now, and a reintroduced key would draw it."""
    source = Path(frame_mod.__file__).read_text(encoding="utf-8")
    assert f'"{key}"' not in source
    assert f"{key}_viewer" not in source
    assert f"_{key}_workspace" not in source
    assert f"modes.{key}" not in source


# --- what replaced Poser's cadence clause, and what a stale journal copy does ---


def _frame_probe(monkeypatch, *, clay_playing):
    """A ``FrameMixin`` with only the Clay timeline clause of ``_frame_active``
    able to answer: input, caret, easing, toasts, jobs, tasks and every viewer
    held quiet, as ``test_studio_frame._frame_probe`` does for Mason's."""
    import types

    import pygame

    from realmspinner.studio.state import AppState

    monkeypatch.setattr(pygame.event, "peek", lambda: False)

    class _Cache:
        active = None

    class _Tasks:
        busy_keys = ()

    state = AppState()
    state.mode = "clay"
    tab = types.SimpleNamespace(doc=types.SimpleNamespace(playing=clay_playing))
    state.clay = types.SimpleNamespace(active=tab)
    obj = frame_mod.FrameMixin.__new__(frame_mod.FrameMixin)
    obj.app_ctx = types.SimpleNamespace(state=state, cache=_Cache(), tasks=_Tasks())
    obj.runtime = types.SimpleNamespace(current_job_id=None)
    obj.viewer = None
    obj.clay_view = None
    obj.mason_view = None
    return obj


def test_a_playing_clay_timeline_keeps_the_frame_at_full_cadence(monkeypatch):
    """Poser's sheet-playback clause went with Poser; the clause that took its place
    is Clay's timeline, which advances only inside the viewport's draw -- at
    ``IDLE_FPS`` a skipped frame drops playback rather than slowing it."""
    from imgui_bundle import imgui

    imgui_ctx = imgui.create_context()
    try:
        assert _frame_probe(monkeypatch, clay_playing=True)._frame_active() is True
    finally:
        imgui.destroy_context(imgui_ctx)


def test_a_paused_clay_timeline_holds_no_cadence(monkeypatch):
    from imgui_bundle import imgui

    imgui_ctx = imgui.create_context()
    try:
        assert _frame_probe(monkeypatch, clay_playing=False)._frame_active() is False
    finally:
        imgui.destroy_context(imgui_ctx)


def test_a_playing_clay_timeline_only_counts_in_clay(monkeypatch):
    from imgui_bundle import imgui

    imgui_ctx = imgui.create_context()
    try:
        probe = _frame_probe(monkeypatch, clay_playing=True)
        probe.app_ctx.state.mode = "library"
        assert probe._frame_active() is False
    finally:
        imgui.destroy_context(imgui_ctx)


def test_a_poser_journal_copy_from_an_older_session_is_skipped_without_a_word(tmp_path):
    """A ``*.pose.json`` copy of the removed Poser kind may still be in a user's
    autosave directory. It is listed by its sidecar like any copy, nothing adopts
    it (no provider is registered for the kind), nothing raises or toasts, and
    **its files stay** -- deleting work because this build cannot read it is the
    one outcome worse than not offering it."""
    import json
    from types import SimpleNamespace

    from realmspinner.studio import journal

    assert journal.provider_for("poser") is None
    payload = tmp_path / "Pose-poser.pose.json"
    payload.write_text("{}", encoding="utf-8")
    journal.meta_path(payload).write_text(
        json.dumps(
            {"version": journal.VERSION, "kind": "poser", "title": "Pose", "uid": "p", "at": 1.0}
        ),
        encoding="utf-8",
    )
    toasts: list[str] = []
    ctx = SimpleNamespace(
        svc=SimpleNamespace(config=SimpleNamespace(autosave_dir=tmp_path)),
        state=SimpleNamespace(recovery=None),
        toast=lambda text, *_a, **_k: toasts.append(text),
    )

    found = journal.recoverable(ctx)
    assert [one.kind for one in found] == ["poser"]
    assert journal.adopt(ctx, found) == 0
    assert toasts == []
    assert payload.exists() and journal.meta_path(payload).exists()
