"""The 2026-10-14 audit, fixer F55: finding shell-chrome-07, the Clay File and
history rows gated the same way as the palette row and the Inspector button."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from realmspinner.kernels.mesh import document as bd
from realmspinner.studio import menus
from realmspinner.studio.modes.clay import doc_io
from realmspinner.studio.modes.clay import mode as clay_mode


def _menu_ctx(tab: Any, *, dragging: bool = False) -> Any:
    return SimpleNamespace(
        state=SimpleNamespace(mode="clay", clay=SimpleNamespace(active=tab)),
        clay_view=SimpleNamespace(dragging=dragging),
        settings=SimpleNamespace(get=lambda key, default=None: default),
    )


def test_clay_file_and_history_rows_grey_with_the_same_reason_on_every_surface(monkeypatch):
    monkeypatch.setattr(clay_mode, "recent_paths", lambda ctx: [])
    tab = SimpleNamespace(doc=bd.ClayDoc(), saving=False, uv_view=None)

    idle = {r.identity: r for r in menus._clay_file_specs(_menu_ctx(tab), [])}
    assert idle["clay:import-mesh"].enabled and not idle["clay:import-mesh"].disabled_reason

    # The palette row and the Inspector button grey while the tab saves; the File
    # row now does too, with the same word.
    tab.saving = True
    busy = {r.identity: r for r in menus._clay_file_specs(_menu_ctx(tab), [])}
    assert not busy["clay:import-mesh"].enabled
    assert busy["clay:import-mesh"].disabled_reason == "Saving..."
    # With no document in front it is still never greyed: the import adopts a tab.
    none = {r.identity: r for r in menus._clay_file_specs(_menu_ctx(None), [])}
    assert none["clay:import-mesh"].enabled

    # Undo and Redo refuse under a save or a live gesture, and now say which.
    ctx = _menu_ctx(tab)
    assert clay_mode.history_blocked_why(ctx, tab) == "Saving..."
    tab.saving = False
    assert clay_mode.history_blocked_why(_menu_ctx(tab, dragging=True), tab) == doc_io.GESTURE_WHY
    assert clay_mode.history_blocked_why(ctx, tab) == ""
    assert clay_mode.history_blocked_why(ctx, None) == ""
