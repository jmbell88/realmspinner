"""The 2026-10-14 audit, findings shell-review-settings-05, -07 and -12 (F44).

Every case draws the real pane function with ``imgui``/``widgets``/``controls``
replaced by recorders, so what is asserted is what the pane would have put on
screen, with no GL context.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from realmspinner.studio import app_ctx
from realmspinner.studio.modes.library.ui.panes import library
from realmspinner.studio.modes.settings.ui.panes import app_settings
from realmspinner.studio.state import AppState


def _recorders(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, MagicMock, MagicMock]:
    widgets, controls, imgui = MagicMock(), MagicMock(), MagicMock()
    widgets.disabled_button.return_value = False
    widgets.destructive_button.return_value = False
    controls.button.return_value = False
    controls.small_button.return_value = False
    monkeypatch.setattr(app_settings, "widgets", widgets)
    monkeypatch.setattr(app_settings, "controls", controls)
    monkeypatch.setattr(app_settings, "imgui", imgui)
    return widgets, controls, imgui


# --- shell-review-settings-05: the trash figure follows the trash -------------


def test_settings_storage_trash_figure_is_re_measured_after_the_trash_changes(monkeypatch):
    """Settings used to ask ``measure_trash(ctx, [])``: the stamp was always ``()``, so
    the figure was taken once per session and never again."""
    widgets, _controls, _imgui = _recorders(monkeypatch)
    monkeypatch.setattr(app_settings, "_model_storage", lambda ctx: None)
    monkeypatch.setattr(app_settings, "_evidence_storage", lambda ctx: None)
    trash = {"count": 3, "bytes": 3000}
    monkeypatch.setattr(library.svc_jobs, "trash_size", lambda svc: dict(trash))

    class Ctx:
        svc = object()

        def __init__(self) -> None:
            self.state = AppState()
            self.cache = SimpleNamespace(storage_error=None, storage={}, jobs=[])

        def submit(self, key: str, fn: Any, *args: Any, **kwargs: Any) -> bool:
            tag = kwargs.pop("tag", None)
            # What shell/tasks.py does when the task lands.
            self.state.preview[library.TRASH_SIZE_SLOT] = (tag, fn(*args, **kwargs))
            return True

    ctx = Ctx()
    ctx.cache.jobs = [{"id": f"j{n}", "deleted_at": "2026-10-14"} for n in range(3)]

    def drawn() -> str:
        widgets.muted.reset_mock()
        app_settings._storage(ctx)
        lines = [str(call.args[0]) for call in widgets.muted.call_args_list if call.args]
        return next(line for line in lines if "trash" in line.lower())

    drawn()  # asks
    assert "3 assets" in drawn()

    # The user trashes thirty-seven more, and comes back to Settings.
    ctx.cache.jobs = [{"id": f"j{n}", "deleted_at": "2026-10-14"} for n in range(40)]
    trash.update(count=40, bytes=4_000_000_000)
    drawn()
    assert "40 assets" in drawn()

    # And an emptied trash reads as empty rather than as the old figure.
    ctx.cache.jobs = []
    trash.update(count=0, bytes=0)
    drawn()
    assert "0 assets" in drawn()


# --- shell-review-settings-07: a multi-pack restore shows its progress --------


def test_a_multi_pack_restore_shows_its_progress_and_cancel_on_the_packs_page(monkeypatch):
    widgets, controls, _imgui = _recorders(monkeypatch)
    rows = [
        {
            "key": key,
            "label": key.title(),
            "present": False,
            "manifest": True,
            "summary": "",
            "modes": (),
            "wheels": 0,
        }
        for key in ("rig", "music")
    ]
    restore_key = app_ctx.pack_key("rig,music")
    ctx = SimpleNamespace(
        pack_rows=rows,
        packs_to_restore=["rig", "music"],
        tasks=SimpleNamespace(
            any_busy=lambda prefix: prefix == "pack:",
            is_busy=lambda key: key == restore_key,
        ),
        progress=lambda key: (
            {"percent": 42.0, "label": "Downloading torch", "phase": "collect"}
            if key == restore_key
            else None
        ),
        toast=lambda *a, **k: None,
    )

    app_settings._packs(ctx)

    assert [call.args[0] for call in widgets.progress_bar.call_args_list] == [42.0]
    shown = [str(call.args[0]) for call in widgets.muted.call_args_list if call.args]
    assert "Downloading torch" in shown
    assert [call.args[0] for call in controls.small_button.call_args_list] == [
        "Cancel##cancel-pack"
    ]
    # The restore's own button is not offered while it runs.
    assert not any(
        "Restore packs" in str(call.args[0]) for call in widgets.disabled_button.call_args_list
    )


# --- shell-review-settings-12: wording that predates the 2026-10-14 reality ---


def test_settings_agent_help_does_not_promise_characters(monkeypatch):
    """The agent surface is Clay-only since Poser and the character pipeline went."""
    row = next(r for r in app_settings.SEARCH_INDEX if r.label.startswith("Allow AI agents"))
    assert "character" not in row.tooltip.lower()

    _recorders(monkeypatch)
    forms = MagicMock()
    form = forms.Form.return_value.__enter__.return_value
    form.switch.return_value = (False, False)
    monkeypatch.setattr(app_settings, "forms", forms)
    ctx = SimpleNamespace(settings=SimpleNamespace(get=lambda *_a: False, set=lambda *_a: None))
    app_settings._agents(ctx)
    help_text = form.switch.call_args.kwargs["help_text"]
    assert "character" not in help_text.lower()
    assert "character" not in (app_settings._agents.__doc__ or "").lower()


def test_no_settings_docstring_names_the_removed_poser_or_a_download_less_app():
    assert "Poser" not in (app_settings.pack_unlocks.__doc__ or "")
    assert "the app still has" not in (app_settings.__doc__ or "")


def test_removing_an_engine_row_says_make_3d_stops_working(monkeypatch):
    """``_confirm_removal`` gave a consequence sentence only for kind ``base``; an engine
    row (``engine:trellis_runtime`` / ``engine:trellis_gguf``) stops Make 3D."""
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        app_settings.dialogs, "ask_delete", lambda ctx, **kw: seen.update(kw)
    )
    app_settings._confirm_removal(
        object(),
        {"row_key": "engine:trellis_gguf", "label": "TRELLIS.2 weights", "kind": "engine",
         "freed_gib": 4.0},
    )
    assert "Make 3D" in seen["message"]
