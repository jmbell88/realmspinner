"""The 2026-10-14 audit's residue round (fixer R), finding create-workspace-04.

``preview.draw`` still carried a "Model / Exported sheets" toggle and a ``"character"``
kind for the rendered character sheets that eed32549 deleted. What it draws now is a
tileset's cells and a sprite draft's atlas, and nothing else.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from realmspinner.studio.modes.create.ui import preview


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def __getattr__(self, name: str):
        def draw(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            if name == "segmented_choice":
                return False, args[2]
            if name == "combo":
                return args[1]
            return False

        return draw

    def named(self, name: str) -> list[tuple]:
        return [args for called, args, _kw in self.calls if called == name]


@pytest.fixture
def rec(monkeypatch):
    recorder = _Recorder()
    for attr in ("widgets", "controls", "imgui"):
        monkeypatch.setattr(preview, attr, recorder)
    monkeypatch.setattr(preview, "sp", lambda value: value)
    return recorder


def _ctx(job: dict, stage: str = "reference") -> SimpleNamespace:
    return SimpleNamespace(
        job=lambda: job,
        textures=None,
        state=SimpleNamespace(
            preview={}, create=SimpleNamespace(stage=stage, preview_mode="result")
        ),
    )


def _tileset_record() -> dict:
    cells = [{"x": 0, "y": 0, "w": 16, "h": 16}, {"x": 16, "y": 0, "w": 16, "h": 16}]
    return {"name": "Tileset", "path": "input.png", "columns": 2, "rows": 1, "cells": cells}


def _draft_record() -> dict:
    cells = [{"x": 0, "y": 0, "w": 32, "h": 32}]
    return {
        "id": "d1",
        "name": "walk a",
        "path": "draft.png",
        "columns": 1,
        "rows": 1,
        "cells": cells,
    }


def test_the_preview_still_draws_a_tileset(monkeypatch, rec):
    job = {"id": "t1", "status": "done", "stage": "tilesheet", "params": {"sheet": {"tile_w": 16}}}
    monkeypatch.setattr(preview, "request", lambda ctx, j: [_tileset_record()])

    assert preview.draw(_ctx(job), 400.0, 300.0) is True

    assert rec.named("segmented_choice")[0][0] == "create-atlas-view"
    assert any("Loading exported sheet" in str(args[0]) for args in rec.named("muted"))


def test_the_preview_still_draws_a_sprite_draft(monkeypatch, rec):
    job = {
        "id": "s1",
        "status": "done",
        "stage": "reference",
        "params": {"generation_type": "sprite_sheet"},
    }
    monkeypatch.setattr(preview, "request", lambda ctx, j: [_draft_record()])

    assert preview.draw(_ctx(job), 400.0, 300.0) is True

    assert [args[0] for args in rec.named("segmented_choice")] == ["create-atlas-view"]


def test_a_mesh_row_is_the_viewports_and_draws_no_exported_sheets_toggle(monkeypatch, rec):
    job = {
        "id": "m1",
        "status": "done",
        "stage": "model",
        "params": {"generation_type": "tileset", "sheet": {"tile_w": 16}},
    }
    monkeypatch.setattr(preview, "request", lambda ctx, j: [_tileset_record()])

    assert preview.draw(_ctx(job), 400.0, 300.0) is False

    assert rec.named("segmented_choice") == []


def test_a_character_row_is_not_asked_for_exported_cells(monkeypatch, rec):
    job = {
        "id": "c1",
        "status": "done",
        "stage": "reference",
        "params": {"generation_type": "character"},
    }

    def never(ctx, j):
        raise AssertionError("a character has no exported cells to load")

    monkeypatch.setattr(preview, "request", never)

    assert preview.draw(_ctx(job), 400.0, 300.0) is False
