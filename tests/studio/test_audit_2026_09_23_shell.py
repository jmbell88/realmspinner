"""Regressions for the 2026-09-23 audit's shell-layouts-tour findings.

Six records, closed together because they share owners (``layouts.py``,
``app_settings.py``, ``mode_manifest.py``, ``state.py``, ``panes/tour.py``)
rather than a theme. See "the 2026-09-23 audit, finding <id>" in each fix's
comments for the paragraph this test proves.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from realmspinner.studio import layouts as layouts_mod
from realmspinner.studio import mode_manifest
from realmspinner.studio import state as state_mod
from realmspinner.studio.modes.library.ui.panes import library as library_pane
from realmspinner.studio.panes import tour as tour_pane


class _Settings:
    """The tiny slice of ``Settings`` that ``Library`` reads and writes."""

    def __init__(self, data: dict | None = None) -> None:
        self._data = dict(data or {})

    def get(self, key, default=None):
        return self._data.get(key, default)

    def set(self, key, value):
        self._data[key] = value


# --- shell-01: Library.reset() never rewrites an unreadable layout ---------


def _library_with_future_layout(name: str = "mine") -> layouts_mod.Library:
    future_blob = {"v": layouts_mod.VERSION + 1, "workspaces": {"inker": {"future": True}}}
    settings = _Settings(
        {
            layouts_mod.LAYOUTS_KEY: {name: future_blob},
            layouts_mod.ACTIVE_KEY: name,
        }
    )
    return layouts_mod.Library(settings)


def test_reset_never_rewrites_an_unreadable_layout():
    """The 2026-09-23 audit, finding shell-01: ``Library.reset()`` had no
    ``readable`` check, so the always-enabled Settings "Reset" button
    replaced a newer build's unreadable layout with a fresh built-in one --
    the exact "older build rewrites a newer layout" case the module
    docstring's "kept verbatim" rule exists to prevent.
    """
    library = _library_with_future_layout()
    before = library.layouts["mine"]
    assert before.readable is False

    library.reset("mine")

    after = library.layouts["mine"]
    assert after.readable is False, (
        "reset() rewrote an unreadable (future-version) layout with a fresh "
        "built-in one instead of leaving it alone"
    )
    assert after.opaque == before.opaque


def test_reset_still_resets_a_readable_layout():
    """The guard above must not turn ``reset`` into a no-op for the ordinary
    case: a readable layout is still put back to the built-in arrangement."""

    settings = _Settings()
    library = layouts_mod.Library(settings)
    library.record("inker", {"left": ["x"]}, set())
    assert library.arrangement("inker").columns

    library.reset()

    assert library.current().readable is True
    assert library.arrangement("inker").columns == {}


# --- shell-04: Library.delete() never coerces an unreadable built-in -------


def test_delete_never_coerces_an_unreadable_built_in_layout():
    """The 2026-09-23 audit, finding shell-04: ``delete()``'s built-in branch
    replaced the stored layout with a fresh one with no ``readable`` check --
    the same rewrite bug as shell-01, on the sibling method. Unreachable from
    today's UI (the Delete button is disabled while a built-in is active),
    but the guard belongs on the data method, not on the one caller that
    happens to exist.
    """
    name = layouts_mod.BUILT_IN[0]
    library = _library_with_future_layout(name)
    before = library.layouts[name]
    assert before.readable is False

    result = library.delete(name)

    after = library.layouts[name]
    assert after.readable is False, (
        "delete() rewrote an unreadable built-in layout with a fresh one"
    )
    assert after.opaque == before.opaque
    assert result is False


def test_delete_still_resets_a_readable_built_in_layout():
    settings = _Settings()
    library = layouts_mod.Library(settings)
    name = layouts_mod.BUILT_IN[0]
    library.record("inker", {"left": ["x"]}, set())
    assert library.arrangement("inker").columns

    result = library.delete(name)

    assert result is True
    assert library.layouts[name].readable is True
    assert library.arrangement("inker").columns == {}


# --- shell-07: the Filters.kind comment points at KIND_OPTIONS -------------


def test_filters_kind_comment_does_not_hand_list_a_stale_option_count():
    """The 2026-09-23 audit, finding shell-07: the comment above
    ``Filters.kind`` in ``state.py`` hand-listed seven values while
    ``library.KIND_OPTIONS`` (the actual combo data) holds nine -- so it
    points at the source of truth instead of restating a count that can
    drift again.
    """
    source = Path(inspect.getfile(state_mod)).read_text(encoding="utf-8")
    # The field the comment sits above.
    idx = source.index('kind: str = "all"')
    comment_block = source[max(0, idx - 700) : idx]

    assert len(library_pane.KIND_OPTIONS) != 7, (
        "KIND_OPTIONS no longer has nine entries -- re-check this test's premise"
    )
    assert "KIND_OPTIONS" in comment_block, (
        "the Filters.kind comment does not point at library.KIND_OPTIONS, so "
        "it is free to drift from it again"
    )
    # The old hand-list, verbatim, must be gone.
    assert "all | reference | tile | model | rig | sheet | sprite" not in comment_block


# --- shell-08: the mode_manifest docstring matches the nested layout -------


def test_mode_manifest_docstring_does_not_claim_modules_live_flat():
    """The 2026-09-23 audit, finding shell-08: the ``_PACKAGE`` comment said
    document-mode modules "live flat under this package today"; every
    ``ModeManifest.module`` entry is actually nested (``"modes.<name>.mode"``),
    which the restructure this comment was written to anticipate already
    happened.
    """
    manifest_source = Path(mode_manifest.__file__).read_text(encoding="utf-8")
    assert "lives flat" not in manifest_source and "live flat" not in manifest_source, (
        "mode_manifest.py still claims modules live flat, but every module= "
        "entry is nested under studio.modes.<name>"
    )
    nested = [m.module for m in mode_manifest.DOC_MODES if "." in m.module]
    assert nested, "expected at least one nested module= entry to anchor this claim"


# --- tour-01: TourState's docstring matches what actually persists ---------


def test_tourstate_docstring_does_not_claim_a_current_method():
    """The 2026-09-23 audit, finding tour-01: ``TourState``'s docstring said
    ``index`` "comes back from settings" and pointed at a ``current()``
    method; ``TourState`` has never had a ``current`` method (``panes/tour.py``
    reads ``tour.step(state.index)`` directly) and only ``finished`` --
    never ``index`` -- is persisted, as the ``tours_finished`` setting.
    """
    doc = inspect.getdoc(state_mod.TourState) or ""
    assert ":meth:`current`" not in doc, (
        "TourState's docstring still cites a :meth:`current` method"
    )
    assert not hasattr(state_mod.TourState, "current"), (
        "TourState grew a current() method -- update the docstring's claim instead "
        "of this test if that was deliberate"
    )
    assert "comes back from settings" not in doc, (
        "TourState's docstring still claims index is settings-restored"
    )


def test_only_finished_is_persisted_not_index():
    """Anchors the claim above in the actual persistence code, not only in
    prose: ``panes/tour.py`` writes ``tours_finished`` and nothing keyed on
    ``index``."""

    tour_source = Path(inspect.getfile(tour_pane)).read_text(encoding="utf-8")
    assert 'settings.set("tours_finished"' in tour_source
    assert "tours_index" not in tour_source


# --- tour-02: _notes() is memoised on the song's edit generation -----------


def _song(*patterns):
    """A stub document with a moving ``history.head``, one push per call to
    ``bump()`` -- enough to prove the cache invalidates on a real edit
    without dragging in the real ``UndoStack``."""
    import numpy as np

    from realmspinner.studio.modes.sirens.engine import document as D

    made = []
    for rows in patterns:
        cells = np.full((len(rows), len(rows[0]), D.COLUMNS), -1, dtype="i2")
        cells[:, :, D.NOTE] = np.asarray(rows, dtype="i2")
        made.append(SimpleNamespace(cells=cells))

    history = SimpleNamespace(head=0)
    doc = SimpleNamespace(patterns=made, history=history)
    return SimpleNamespace(active=SimpleNamespace(doc=doc)), doc


def _ctx(**state):
    base = {"mode": "home", "inker": None, "sirens": None}
    base.update(state)
    return SimpleNamespace(state=SimpleNamespace(**base))


@pytest.fixture(autouse=True)
def _reset_notes_cache():
    tour_pane._notes_cache = None
    yield
    tour_pane._notes_cache = None


def test_notes_is_memoised_on_the_songs_history_head(monkeypatch):
    """The 2026-09-23 audit, finding tour-02: ``_notes()`` ran an unmemoised
    numpy pass over every pattern in the whole song on every frame a Sirens
    tour step showing "notes_at_least" was on screen. It must now answer a
    second call for the *same* ``history.head`` without re-scanning, and
    re-scan once the head moves (a real edit happened).
    """
    sirens, doc = _song([[24, -1], [-1, -1], [26, -1]])
    ctx = _ctx(sirens=sirens)

    calls = {"n": 0}
    real_asarray = __import__("numpy").asarray

    def counting_asarray(*a, **k):
        calls["n"] += 1
        return real_asarray(*a, **k)

    monkeypatch.setattr("numpy.asarray", counting_asarray)

    first = tour_pane._notes(ctx)
    scans_after_first = calls["n"]
    assert scans_after_first > 0, "the stub never even scanned once -- broken test setup"

    second = tour_pane._notes(ctx)
    assert second == first == 2
    assert calls["n"] == scans_after_first, (
        "_notes() re-scanned the song on a second call at the same history.head "
        "-- it is not memoised on the edit generation"
    )

    doc.history.head += 1  # a real edit: push/undo/redo all move head
    third = tour_pane._notes(ctx)
    assert third == 2
    assert calls["n"] > scans_after_first, (
        "_notes() did not re-scan after history.head moved -- it would serve a "
        "stale count forever after one real edit"
    )


def test_notes_still_answers_correctly_without_a_history_attribute():
    """The memoisation must not regress the documented tolerance for a
    differently-shaped document (``test_notes_at_least_survives_a_document_
    shaped_differently`` in ``test_tour_conditions.py``, which stubs a doc
    with no ``history`` at all)."""

    doc = SimpleNamespace(patterns=[])
    sirens = SimpleNamespace(active=SimpleNamespace(doc=doc))
    assert tour_pane._notes(_ctx(sirens=sirens)) == 0
