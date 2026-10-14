"""Regression tests for the 2026-10-14 audit's Home findings shell-home-library-01
and shell-home-library-12."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from realmspinner.studio import mode_manifest
from realmspinner.studio.modes.home.ui.panes import landing
from realmspinner.studio.state import Filters


@pytest.mark.parametrize("kind", sorted(landing.KIND_OPENERS))
def test_activating_a_document_row_opens_it_in_its_own_mode(kind, tmp_path, monkeypatch):
    """Every document Resume row raised ``ModuleNotFoundError`` because
    ``_open_with`` resolved the manifest's ``modes.<x>.mode`` string relative to
    the pane's own package (``studio.modes.home.ui.panes``), not to ``studio``."""
    opener = mode_manifest._import(landing.KIND_OPENERS[kind])
    opened: list = []
    monkeypatch.setattr(opener, "open_path", lambda ctx, path: opened.append(path))
    doc = tmp_path / f"doc-{kind}.bin"
    doc.write_bytes(b"x")
    state = SimpleNamespace(mode="home", previous_mode="home", mode_observed="home")
    ctx = SimpleNamespace(state=state, settings=SimpleNamespace(), toast=lambda *a, **k: None)

    landing.open_row(ctx, landing.Row(kind=kind, key=str(doc), icon="", name=doc.name))

    assert opened == [doc]
    assert state.mode == mode_manifest.kind_mode_table()[kind]


def _click_ctx(filters: Filters, status: str = "running") -> SimpleNamespace:
    job = {"id": "q1", "name": "a chest", "status": status}
    state = SimpleNamespace(
        mode="home",
        previous_mode="home",
        mode_observed="home",
        filters=filters,
        library_scroll_to=None,
        preview={},
    )
    return SimpleNamespace(state=state, cache=SimpleNamespace(active=job), progress=None)


def test_the_queue_row_opens_a_library_that_shows_the_named_job():
    """The row set only ``Filters.status``: a leftover text query, kind,
    favourites, usable-only or the trash view still hid the job it named."""
    filters = Filters(
        text="dragon",
        kind="sprite",
        favorites_only=True,
        usable_only=True,
        trash=True,
        job_ids=frozenset({"other"}),
    )
    ctx = _click_ctx(filters)
    row = landing._queue_status(ctx)

    landing.open_status_row(ctx, row)

    job = {"id": "q1", "name": "a chest", "status": "running", "grade": 0.0}
    assert ctx.state.mode == "library"
    assert filters.matches(job)
    assert filters.job_ids is None and filters.trash is False
    assert ctx.state.library_scroll_to == "q1"
