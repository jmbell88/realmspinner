"""Slice B2, the studio half: the mesh grade carries a note, the generation-log
export announces where it landed (and stays quiet when the picker is cancelled),
and the clean-cutout switch's helper says the default it really has.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from realmspinner.studio import controls as real_controls
from realmspinner.studio import generation_log_export
from realmspinner.studio import settings as studio_settings
from realmspinner.studio.modes.settings.ui.panes import app_settings
from realmspinner.studio.panes import inspector
from realmspinner.studio.state import AppState


class _Spy:
    def __init__(self, answers: dict[str, Any] | None = None) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.answers = answers or {}

    def __getattr__(self, name: str):
        def call(*args: Any, **kwargs: Any) -> Any:
            self.calls.append((name, args, kwargs))
            answer = self.answers.get(name)
            return answer(*args, **kwargs) if callable(answer) else answer

        return call


_JOB = {"id": "m1", "stage": "model", "status": "done", "files": ["model.glb"]}


def _press(monkeypatch, tmp_path, *, label: str | None, full: bool = False, note: str = ""):
    """Draw the inspector's grade section with a note staged and ``label`` pressed."""
    settings = studio_settings.Settings(tmp_path / studio_settings.FILENAME)
    settings.set(inspector.FULL_SCALE_SETTING, full)
    state = AppState(mode="create")
    state.inspector_graded["m1"] = False
    if note:
        inspector.stage_note(state, "m1", note)
    ctx = SimpleNamespace(state=state, settings=settings)
    widgets = _Spy(
        {
            "header": True,
            "disabled_button": lambda text, *a, **k: bool(label) and text.startswith(label),
            "grade_buttons": lambda *a, **k: 3 if label == "grid" else None,
            "tag_toggles": None,
            "input_text": lambda text, value, **k: value,
        }
    )
    widgets.grid_width = lambda n, **k: 80.0
    controls = _Spy({"checkbox": lambda text, value, **k: (False, value)})
    controls.ButtonRole = real_controls.ButtonRole
    filed: list[tuple[int, tuple[str, ...], str | None]] = []
    monkeypatch.setattr(inspector, "widgets", widgets)
    monkeypatch.setattr(inspector, "controls", controls)
    monkeypatch.setattr(inspector, "imgui", _Spy())
    monkeypatch.setattr(
        inspector,
        "record_verdict",
        lambda c, jid, grade, tags=(), note=None: filed.append((grade, tags, note)),
    )
    inspector._verdict(ctx, _JOB)
    return filed


def test_the_note_reaches_record_verdict_from_a_quick_grade(monkeypatch, tmp_path):
    filed = _press(monkeypatch, tmp_path, label="Usable", note="stretched UVs on the back")
    assert filed == [(3, (), "stretched UVs on the back")]


def test_the_note_reaches_record_verdict_from_the_full_scale_row(monkeypatch, tmp_path):
    filed = _press(monkeypatch, tmp_path, label="grid", full=True, note="good silhouette")
    assert filed == [(3, (), "good silhouette")]


def test_no_note_staged_files_no_note(monkeypatch, tmp_path):
    assert _press(monkeypatch, tmp_path, label="Ship it") == [(5, (), None)]


def test_the_note_is_stored_with_the_grade_shown_afterwards_and_cleared_for_the_next(svc):
    job_id = svc.store.create("image", "a chest", {}, stage="model", status="done")
    ctx = SimpleNamespace(
        svc=svc,
        state=AppState(mode="create"),
        cache=SimpleNamespace(invalidate=lambda: None),
        toast=lambda *a, **k: None,
    )
    inspector.stage_note(ctx.state, job_id, "  roof is flat  ")

    assert inspector.record_verdict(
        ctx, job_id, 3, (), inspector.staged_note(ctx.state, job_id)
    )

    assert svc.store.latest_verdicts()[0]["note"] == "roof is flat"
    assert inspector.latest_note(ctx, job_id) == "roof is flat"
    assert inspector.staged_note(ctx.state, job_id) == "", "the next grade starts clean"


def test_a_note_typed_about_one_mesh_does_not_ride_onto_the_next():
    state = AppState(mode="create")
    inspector.stage_note(state, "a", "about a")
    assert inspector.staged_note(state, "b") == ""


# --- the export toast ---------------------------------------------------------------------


class _Done:
    def __init__(self, result: Any) -> None:
        self.key = generation_log_export.KEY
        self.result = result
        self.tag = None
        self.error = None


def _land(result: Any) -> list[tuple[Any, ...]]:
    from realmspinner.studio import main as main_mod

    toasts: list[tuple[Any, ...]] = []
    ctx = SimpleNamespace(toast=lambda *a, **k: toasts.append(a))
    app = SimpleNamespace(app_ctx=ctx, _unclaimed=set())
    main_mod.App._on_task_done(app, _Done(result))
    return toasts


def test_the_export_toast_names_where_the_log_was_saved():
    path = str(Path("C:/logs/realmspinner-generation-log-20261011-120000.jsonl"))
    assert _land(path) == [(f"Generation log saved to {path}.", "success")]


def test_a_cancelled_folder_picker_toasts_nothing():
    assert _land(None) == []


def test_settings_and_history_share_one_export_task_key():
    from realmspinner.studio.modes.create.ui.panes import history

    assert "generation_log_export.request" in inspect.getsource(history.draw)
    assert "generation_log_export.request" in inspect.getsource(app_settings._generation_log)
    assert generation_log_export.KEY == "generation-log-export"


def test_the_export_task_returns_none_when_the_picker_is_cancelled(monkeypatch):
    monkeypatch.setattr(generation_log_export.dialogs, "select_folder", lambda title: None)
    assert generation_log_export.task(object()) is None


def test_the_export_task_writes_the_log_into_the_picked_folder(svc, tmp_path, monkeypatch):
    out = tmp_path / "picked"
    out.mkdir()
    monkeypatch.setattr(generation_log_export.dialogs, "select_folder", lambda title: out)
    written = generation_log_export.task(svc)
    assert Path(written).parent == out and Path(written).suffix == ".jsonl"


# --- the clean-cutout helper -------------------------------------------------------------------


def _switch_helper() -> str:
    tree = ast.parse(Path(inspect.getsourcefile(app_settings)).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "switch"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "skip_clean_matte_preview"
        ):
            for kw in node.keywords:
                if kw.arg == "helper":
                    return ast.literal_eval(kw.value)
    raise AssertionError("the switch is gone")


def test_the_clean_cutout_helper_says_it_is_on_by_default():
    helper = _switch_helper()
    assert helper.startswith("On by default")
    assert "Off by default" not in helper
    # And the claim is true: the default is the settings module's, not the pane's.
    assert studio_settings.DEFAULTS["skip_clean_matte_preview"] is True


def test_the_manual_agrees_the_clean_cutout_switch_is_on_by_default():
    manual = Path(__file__).resolve().parents[2] / "docs" / "manual" / "23-generating-meshes.md"
    text = " ".join(manual.read_text(encoding="utf-8").split())
    assert "Don't ask for clean cutouts** switch, **on by default**" in text


@pytest.mark.parametrize("chapter", ["22-generating-references.md", "38-shortcuts.md"])
def test_the_manual_documents_the_history_pane_and_the_labelling_keys(chapter):
    manual = Path(__file__).resolve().parents[2] / "docs" / "manual" / chapter
    text = manual.read_text(encoding="utf-8")
    assert "Creations" not in text
    assert "History" in text
    if chapter.startswith("38"):
        assert "| G / B |" in text and "| N |" in text
