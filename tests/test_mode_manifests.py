"""P2's own claim, made twice: the manifest replaces five hand tables, and it
cannot rot in either direction.

The first half (the regression this phase exists to fix): every mode module
that defines ``persist`` is actually called at teardown. Before this phase,
``main.teardown`` hand-called five of the six -- ``sirens_mode.persist`` had
no caller anywhere, though its own docstring says it runs after every open
and save.

The second half (the ``PUBLISHERS`` shape this repo already writes down as a
bad gate): a manifest entry naming a module that does not exist, or that does
not expose what the manifest claims, must fail by name -- so the table this
phase builds cannot rot the way the tables it replaces did, in the direction
that adds a row that lies.
"""

from __future__ import annotations

import sys
from importlib import import_module
from types import SimpleNamespace
from typing import Any

import pytest
from test_studio_wiring import FakeSettings, _ctx, _teardown_app

from warlock.studio import journal, mode_manifest, modes, palette
from warlock.studio.modes.home.ui.panes import landing


@pytest.fixture
def fake_pygame(monkeypatch):
    """``teardown`` imports pygame at call time and quits it -- stubbed so a
    unit test does not tear down a display another test in the session may be
    holding. Same shape as ``test_studio_wiring``'s fixture of the same name;
    duplicated rather than imported, since importing a ``@pytest.fixture`` by
    name across modules is a redefinition ruff's F811 (rightly) does not want
    to special-case for pytest's fixture-injection magic.
    """
    stub = SimpleNamespace(quit=lambda: None, display=SimpleNamespace(set_caption=lambda _t: None))
    monkeypatch.setitem(sys.modules, "pygame", stub)
    return stub


def _all_mode_modules() -> dict[str, Any]:
    """Every document-mode module, imported once for the sweeps below."""
    return {
        entry.module: import_module(f"warlock.studio.{entry.module}")
        for entry in mode_manifest.DOC_MODES
    }


# --- the regression: every ``persist`` gets a caller ------------------------


def test_teardown_calls_persist_for_every_mode_the_manifest_reports(fake_pygame, monkeypatch):
    """The regression this phase exists to fix, run for real: every mode
    :func:`mode_manifest.persisting_modes` reports must actually be called at
    teardown.

    Against the unfixed ``main.py`` (five hand-written ``_persist_<mode>``
    methods, one per mode, Sirens omitted) this fails naming ``sirens``:
    ``main.teardown`` never called ``sirens_mode.persist`` at all, although
    the module's own docstring says it runs "after every open and save".
    """
    called: list[str] = []
    for entry in mode_manifest.persisting_modes():
        module = import_module(f"warlock.studio.{entry.module}")
        monkeypatch.setattr(module, "persist", lambda ctx, _k=entry.key: called.append(_k))

    _teardown_app(_ctx(FakeSettings())).teardown()

    expected = {entry.key for entry in mode_manifest.persisting_modes()}
    missing = expected - set(called)
    assert not missing, f"{sorted(missing)} define persist() but teardown never called it"


def test_sirens_defines_persist_and_is_covered_by_the_manifest():
    """Names the exact bug, so a passing set-difference above cannot hide a
    coincidence: Sirens defines ``persist`` and is one of the modes
    :func:`mode_manifest.persisting_modes` reports."""
    modules = _all_mode_modules()
    assert callable(modules["modes.sirens.mode"].persist)
    assert "sirens" in {entry.key for entry in mode_manifest.persisting_modes()}


def test_every_mode_module_defining_persist_is_in_persisting_modes():
    """The manifest's own derivation is exhaustive: nothing that defines
    ``persist`` is left out of what it reports, checked independently of the
    real teardown call above."""
    modules = _all_mode_modules()
    defines_persist = {
        entry.key
        for entry in mode_manifest.DOC_MODES
        if callable(getattr(modules[entry.module], "persist", None))
    }
    covered = {entry.key for entry in mode_manifest.persisting_modes()}
    assert defines_persist == covered


# --- the bidirectional gate: the manifest cannot rot either way -------------


def test_every_manifest_module_exists_and_is_importable():
    """A row naming a module this build cannot import must fail by name, not
    surface later as a mysterious ``ensure_providers`` log line."""
    for entry in mode_manifest.DOC_MODES:
        try:
            import_module(f"warlock.studio.{entry.module}")
        except ImportError as exc:  # pragma: no cover - failure path
            pytest.fail(f"{entry.key}: module {entry.module!r} does not exist ({exc})")


def test_every_manifest_module_exposes_active():
    """Every document mode's module must answer ``active(ctx)`` -- the
    palette's save/export/undo dispatch and the status bar both call it
    unconditionally for any mode this table names."""
    modules = _all_mode_modules()
    for entry in mode_manifest.DOC_MODES:
        assert callable(getattr(modules[entry.module], "active", None)), (
            f"{entry.key}: {entry.module} has no active()"
        )


def test_every_manifest_module_registers_its_claimed_journal_kind():
    """A manifest ``kind`` that disagrees with the module's own registered
    ``JOURNAL.kind`` is exactly the drift this phase exists to make
    impossible: the declared fact must match what the module says about
    itself once it is actually asked."""
    journal.ensure_providers()
    modules = _all_mode_modules()
    for entry in mode_manifest.DOC_MODES:
        provider = getattr(modules[entry.module], "JOURNAL", None)
        assert provider is not None, f"{entry.key}: {entry.module} has no JOURNAL"
        assert provider.kind == entry.kind, (
            f"{entry.key}: manifest says kind={entry.kind!r}, JOURNAL.kind={provider.kind!r}"
        )


def test_every_opener_module_is_importable_and_exposes_open_path():
    """``KIND_OPENERS``'s own bug (a ``.wsng`` row that did nothing on click)
    was exactly a module named in the table that could not do what the table
    claimed. This is that check, run over the manifest instead of the table
    it now generates."""
    for entry in mode_manifest.DOC_MODES:
        if entry.opener is None:
            continue
        module = import_module(f"warlock.studio.{entry.opener}")
        assert callable(getattr(module, "open_path", None)), (
            f"{entry.key}: opener {entry.opener!r} has no open_path()"
        )


def test_every_manifest_kind_is_unique():
    """Two modes sharing one journal/document kind would let a recovered
    crash copy of one adopt into the other."""
    kinds = [entry.kind for entry in mode_manifest.DOC_MODES]
    assert len(kinds) == len(set(kinds)), kinds


def test_every_manifest_key_is_a_real_mode():
    """A manifest row for a mode key that ``modes.MODES`` has never heard of
    would build tables the palette and the status bar could switch to
    something that does not exist."""
    for entry in mode_manifest.DOC_MODES:
        assert entry.key in modes.KEYS, f"{entry.key!r} is not in modes.KEYS"


# --- the four derived tables match what they replaced -----------------------


def test_journal_provider_modules_matches_the_manifest():
    assert mode_manifest.journal_modules() == journal._PROVIDER_MODULES


def test_palette_doc_modes_matches_the_manifest():
    assert mode_manifest.export_table() == palette._DOC_MODES


def test_landing_kind_openers_matches_the_manifest():
    assert mode_manifest.opener_table() == landing.KIND_OPENERS


def test_landing_kind_modes_matches_the_manifest():
    assert mode_manifest.kind_mode_table() == landing._KIND_MODES
