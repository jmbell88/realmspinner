"""Closes three findings from the 2026-09-23 audit of Packwright.

packwright-01: three doors that add sources to a tab -- a window drop, the
Library's "Add to Packwright" and a rendered-sheet hand-off (Poser's own
since P9, Troupe's before it) -- never checked ``tab.busy`` at all, unlike the
sources pane's own Add buttons, which the 2026-09-18 audit already greyed for
exactly this case. packwright-02: the source list's context-menu Rename and
Remove drew fully enabled while a save was writing, so a click there looked
like it worked and silently did nothing. packwright-03: ``core/safeio/
atomic.py`` and ``tests/test_atomic_writes.py`` both cited "twenty-two
``dialogs.save_file`` sites", which a grep already put at 31.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from realmspinner.studio.modes.packwright import mode as packwright_mode
from realmspinner.studio.modes.packwright.ui.panes import sources as packwright_sources


class FakeCtx:
    """The minimal double ``packwright_mode``'s add doors need. Trimmed from
    ``test_packwright_mode.py``'s own ``FakeCtx``, which this session does not
    own and so cannot import from without risking two writers touching one
    module's collection order."""

    def __init__(self) -> None:
        self.state = _AppState()
        self.settings = _Settings()
        self.submitted: list[str] = []
        self.toasts: list[tuple[str, str]] = []

    def submit(self, key: str, run: Any, *args: Any) -> bool:
        self.submitted.append(key)
        return True

    def toast(self, message: str, kind: str = "info") -> None:
        self.toasts.append((message, kind))


class _AppState:
    def __init__(self) -> None:
        self.packwright = None
        self.inker = None
        self.mode = "home"
        self.preview: dict[str, Any] = {}


class _Settings:
    """``docmodes.recents_for``'s own reader/writer, ``test_packwright_mode.
    FakeCtx``'s shape: ``new_document`` -> ``adopt`` -> ``remember_path``
    reaches this on every tab minted, including the ones minted only to prove
    the busy guard does not swallow the "no tab yet" case."""

    def __init__(self) -> None:
        self.store: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self.store.get(key)

    def set(self, key: str, value: Any) -> None:
        self.store[key] = value


def _busy_tab(ctx: FakeCtx) -> Any:
    tab = packwright_mode.new_document(ctx)
    tab.saving = True
    assert tab.busy is True
    return tab


# --- packwright-01: every add door checks tab.busy -----------------------


def test_dropping_a_file_while_saving_is_refused_like_every_other_mutating_door():
    """``add_source_paths`` (a window drop) used to reach its decode task with
    no ``tab.busy`` check at all -- unlike ``remove_source``/``rename_source``/
    ``set_settings``, which already refuse a busy tab silently. Fails against
    the unfixed code: ``ctx.submitted`` was non-empty and the drop went ahead
    on a tab mid-save."""
    ctx = FakeCtx()
    tab = _busy_tab(ctx)
    before = len(tab.doc.sources)

    packwright_mode.add_source_paths(ctx, [Path("nope.png")])

    assert ctx.submitted == [], "a drop onto a saving tab must not start a decode"
    assert len(tab.doc.sources) == before


def test_a_library_add_while_saving_is_refused_too():
    """The Library's "Add to Packwright" reaches ``add_job_source`` the same
    way a drop reaches ``add_source_paths`` -- from outside the pane, past the
    Add buttons' own ``editable`` grey-out -- and had the identical gap."""
    ctx = FakeCtx()
    _busy_tab(ctx)

    packwright_mode.add_job_source(ctx, {"id": "j1", "name": "chest"})

    assert ctx.submitted == [], "a library add onto a saving tab must not start a decode"


def test_a_rendered_sheet_handoff_while_saving_is_refused_too():
    """Poser's sheet hand-off (Troupe's before P9, 2026-09-18) is the third
    door ``add_rendered_sheet`` covers, and it had the same gap: the busy
    check has to happen before this even looks at ``job_id``/``sheet_id``, so
    a monkeypatched ``service.sheets`` is not needed to prove the refusal."""
    ctx = FakeCtx()
    _busy_tab(ctx)

    packwright_mode.add_rendered_sheet(ctx, "job1", "sheet1")

    assert ctx.submitted == [], "a sheet hand-off onto a saving tab must not start a decode"


def test_a_drop_with_no_atlas_open_still_mints_one():
    """The busy guard must not swallow the existing "no tab yet" behaviour:
    dropping with nothing open still starts an atlas, since an atlas has no
    numbers that cannot be taken back later."""
    ctx = FakeCtx()
    assert packwright_mode.active(ctx) is None

    packwright_mode.add_source_paths(ctx, [Path("a.png")])

    assert packwright_mode.active(ctx) is not None
    assert ctx.submitted != []


# --- packwright-02: the source row's context menu -------------------------


def test_the_source_row_context_menu_is_actually_disabled_while_saving():
    """The 2026-09-23 audit's packwright-02: Rename and Remove in ``_row``'s
    ``src-menu`` popup used to be drawn with ``controls.menu_item_simple(...)
    and editable`` -- fully enabled to look at and to click, with the guard
    only on the *effect* -- unlike every other greyed control in this pane
    (``widgets.disabled_button(..., reason=widgets.DOCUMENT_SAVING_WHY)``).
    Checked by source rather than by driving a popup open in a headless
    frame: imgui's own popup stack needs a live window to hold Begin/End
    matched, which a static assertion on the call shape does not.
    """
    source = inspect.getsource(packwright_sources._row)
    menu_block = source.split('begin_popup_context_item("src-menu")', 1)[1]
    menu_block = menu_block.split("imgui.end_popup()", 1)[0]

    assert "enabled=editable" in menu_block, (
        "the context menu's items must be drawn disabled, not merely have "
        "their effect gated after the fact"
    )
    assert " and editable" not in menu_block, (
        "a menu item gated only on its effect looks clickable and silently "
        "does nothing while the tab is busy"
    )
    assert menu_block.count("reason=widgets.DOCUMENT_SAVING_WHY") == 2, (
        "both Rename and Remove need the same reason every other greyed "
        "control in this pane already gives"
    )


# --- packwright-03: the stale call-site count ------------------------------


def test_the_atomic_docstrings_no_longer_cite_a_stale_save_file_count():
    """``core/safeio/atomic.py`` and ``tests/test_atomic_writes.py`` both said
    "twenty-two ``dialogs.save_file`` sites"; a grep the same day found 31.
    Rather than swap one number that will drift again for another, both
    docstrings drop the citation. Fails against the unfixed code, where both
    modules still say "twenty-two"."""
    import test_atomic_writes  # this file's sibling test module

    from realmspinner.core.safeio import atomic

    assert "twenty-two" not in (atomic.__doc__ or "")
    assert "twenty-two" not in (test_atomic_writes.__doc__ or "")
