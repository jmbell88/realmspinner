"""Regression tests for the shell findings of the 2026-09-14 audit.

Six findings, six tests, one per module: ``dialogs.ConfirmQueue`` reading
Enter as Cancel even while a body field is being edited (shell-01), the
library export popup's browse task publishing a new destination beside the
previous destination's plan (shell-02), ``status_bar.draw`` surviving its own
deletion (shell-04), a verdict recorded in Review mode never reaching the
inspector's own ``is_graded`` memo (shell-05), ``review_mode``'s reference
cache growing without bound (shell-06), and ``zipguard``'s docstring naming
only four of the modules that actually open a bounded zip (shell-08).
"""

from __future__ import annotations

import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from modes.review.test_review_mode import FakeCtx, _mesh, _scanned  # noqa: F401 -- see shell-05

from realmspinner.core.safeio import zipguard
from realmspinner.service import export as svc_export
from realmspinner.studio import dialogs, status_bar
from realmspinner.studio.modes.library.ui.panes import library
from realmspinner.studio.modes.review import mode as review_mode
from realmspinner.studio.panes import inspector

# --- shell-01: Enter always cancelled a Confirm, even mid-edit --------------


def test_pressing_enter_after_editing_the_prune_keep_count_does_not_silently_cancel_it():
    """Prune's "Keep the newest" count is a body ``input_int`` inside a
    ``Confirm``. Before this, ``ConfirmQueue.draw`` read *any* Enter as
    Cancel, so pressing Enter to commit that field closed the whole dialog
    as cancelled and pruned nothing, with no toast or error to say so.

    Driven straight at the pure decision helper (no imgui context needed),
    per the audit's own instruction to keep this testable without a window.
    """
    # Escape always cancels, whatever has focus.
    assert dialogs._confirm_cancelled_by_key(escape=True, enter=False, body_had_focus=True)
    assert dialogs._confirm_cancelled_by_key(escape=True, enter=False, body_had_focus=False)
    # Enter typed into an already-focused body field (the count box) commits
    # it -- it must not also be read as the dialog's own Cancel.
    assert not dialogs._confirm_cancelled_by_key(
        escape=False, enter=True, body_had_focus=True
    ), "Enter pressed while the Prune count field had focus silently cancelled the dialog"
    # Enter with nothing else focused still means "take the safe way out" --
    # the behaviour UX-07's fix (focus lands on the safe button) depends on.
    assert dialogs._confirm_cancelled_by_key(escape=False, enter=True, body_had_focus=False)
    # Neither key pressed: no cancel from this helper either way.
    assert not dialogs._confirm_cancelled_by_key(escape=False, enter=False, body_had_focus=True)
    assert not dialogs._confirm_cancelled_by_key(escape=False, enter=False, body_had_focus=False)


# --- shell-02: the browse task's dest/plan were two separate writes --------


def test_browse_never_shows_a_destination_paired_with_the_previous_plan(monkeypatch, tmp_path):
    """``_run_export`` (the task thread) used to write ``popup.dest`` and then
    ``popup.plan`` as two separate, unlocked assignments while the frame
    thread reads both every frame in ``_export_popup_body`` -- so a frame
    drawn between the two writes could show a freshly picked destination
    beside the *previous* destination's plan (which files exist, which get
    clobbered).

    Proven by instrumenting every assignment the browse branch makes to the
    popup: the fix (bundling dest+plan into one ``_DestPlan``) publishes
    exactly one attribute assignment; the unfixed code makes two. Verified
    against the actual pre-fix source (loaded from HEAD in a throwaway
    probe, not checked into the tree) that this records ``["dest", "plan"]``.
    """
    calls: list[str] = []
    orig_setattr = library._ExportPopup.__setattr__

    def tracking_setattr(self: Any, name: str, value: Any) -> None:
        if name in ("dest", "plan", "_dest_plan"):
            calls.append(name)
        orig_setattr(self, name, value)

    monkeypatch.setattr(library._ExportPopup, "__setattr__", tracking_setattr)

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()

    monkeypatch.setattr(library.dialogs, "select_folder", lambda *a, **k: second_dir)
    monkeypatch.setattr(
        library.svc_export,
        "plan_export",
        lambda job, dest: svc_export.ExportPlan(files=()),
    )
    # shell-01, the 2026-09-18 audit: "replace" now writes through
    # ``export_planned_to_folder`` (routed off ``popup.plan`` like "keep
    # both" always was) rather than ``export_to_folder``, which read its
    # destination off ``svc.config.export_dir`` and ignored the browsed one.
    monkeypatch.setattr(
        library.svc_export, "export_planned_to_folder", lambda *a, **k: {"ok": True}
    )

    state = SimpleNamespace(_library_export=None)
    ctx = SimpleNamespace(state=state, svc=object())

    def run() -> None:
        library._run_export(ctx, "Save to project", ["j1"], ["name"], first_dir, as_zip=False)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        deadline = time.monotonic() + 5.0
        while state._library_export is None and time.monotonic() < deadline:
            time.sleep(0.005)
        popup = state._library_export
        assert popup is not None, "the export task never published a popup"
        calls.clear()  # drop the construction's own field assignment(s)

        popup.decisions.put("browse")
        deadline = time.monotonic() + 5.0
        while popup.dest != second_dir and time.monotonic() < deadline:
            time.sleep(0.005)
        assert popup.dest == second_dir, "the browse pick never landed on the popup"

        popup.decisions.put("replace")
        thread.join(timeout=5.0)
        assert not thread.is_alive(), "the export task never finished"
    finally:
        if thread.is_alive():
            # Never leave a task thread parked on the queue for pytest's own
            # atexit machinery to hang on.
            state._library_export.decisions.put("cancel")
            thread.join(timeout=5.0)

    assert calls == ["_dest_plan"], (
        "the browse branch published dest/plan through "
        f"{calls} separate assignments instead of one atomic publish"
    )


# --- shell-04: status_bar.draw survived its own deletion --------------------


def test_status_bar_module_has_no_draw_function_once_the_menu_bar_owns_status():
    """T0 of the Familiar programme moved the per-item row into ``menus.draw``
    and the one collapsed row into what was then ``panes/bottom_pane.py``
    (replaced 2026-09-23 by ``panes/familiar_dock.py``) -- INVARIANTS, that
    module's own docstring and ``test_editor_shell.py`` have all said
    ``status_bar.draw`` was deleted since that day, but the function (and
    the ``STATUS_H`` constant only it used) was still sitting in the module,
    unreferenced.
    """
    assert not hasattr(status_bar, "draw"), (
        "status_bar.draw still exists though INVARIANTS, the Familiar dock's "
        "docstring and test_editor_shell.py all say it was deleted"
    )
    assert not hasattr(status_bar, "STATUS_H"), (
        "STATUS_H outlived the one function (draw) that read it"
    )


# --- shell-05: Review's verdict never reached the inspector's own memo -----


def test_a_verdict_recorded_in_review_mode_is_reflected_by_is_graded_in_another_host(svc):
    """``inspector.is_graded`` memoises per job id in ``state.inspector_graded``
    and ``inspector.record_verdict`` keeps that memo in step -- but Review
    mode's own ``record`` filed a verdict through a completely separate path
    (``verdicts.record_verdict`` directly) and never touched it, so
    ``is_graded`` kept answering "ungraded" for a mesh Review had just graded,
    in whatever other host asked (the inspector pane itself, reopened without
    an intervening cache-clearing refresh).
    """
    ctx = FakeCtx(svc)
    job_id = _mesh(svc, "a lamp")
    state = _scanned(ctx)
    unit = review_mode.current(state)
    assert unit is not None and unit["job_id"] == job_id

    assert inspector.is_graded(ctx, job_id) is False

    review_mode.record(ctx, 3)

    assert inspector.is_graded(ctx, job_id) is True, (
        "review_mode.record filed a verdict but inspector.is_graded's memo "
        "was never told, so another host would keep calling this mesh "
        "ungraded"
    )


# --- shell-06: the reference-path cache had no ceiling ----------------------


def test_reference_path_cache_does_not_grow_without_bound(monkeypatch, tmp_path):
    """``_REFERENCE_CACHE`` is keyed by unit dir and kept for the life of the
    process; nothing ever evicted from it, so a long session reviewing many
    sweep runs grew it without bound. ``_REFERENCE_CACHE_MAX`` and the
    ``OrderedDict`` LRU it bounds are what this pins in place -- the
    ``monkeypatch.setattr`` on ``_REFERENCE_CACHE_MAX`` below fails outright
    against the unfixed module, which carries no such constant at all.
    """
    monkeypatch.setattr(review_mode, "_REFERENCE_CACHE", type(review_mode._REFERENCE_CACHE)())
    monkeypatch.setattr(review_mode, "_REFERENCE_CACHE_MAX", 8)

    for i in range(50):
        review_mode.reference_path({"dir": str(tmp_path / f"unit{i}")})

    assert len(review_mode._REFERENCE_CACHE) <= 8, (
        f"_REFERENCE_CACHE grew to {len(review_mode._REFERENCE_CACHE)} entries "
        "past its own cap of 8"
    )


# --- shell-08: zipguard's docstring named four doors; there are more -------


def _bounded_zip_callers() -> list[str]:
    """Every module under ``realmspinner/`` that instantiates ``zipguard.BoundedZip``
    directly, named by its own directory plus filename -- derived from the
    tree by grep rather than hand-listed a second time, so this (and the
    docstring test that uses it) cannot go stale the way the docstring itself
    did.

    Scanned from the ``realmspinner`` package root, not ``studio/`` alone: P3 of
    the restructure (``dev/RESTRUCTURE.md``) moved two of these doors --
    Inker's ``ora.py`` and Clay's ``serialize.py`` -- out from under
    ``studio/`` into ``kernels/pixel/`` and ``kernels/mesh/``, so a
    ``studio/``-relative scan no longer reaches every caller. Naming each one
    by its immediate directory rather than a full path keeps the docstring
    reading as "which engine", the thing a reader of this module actually
    wants to know, rather than which package layer that engine currently
    sits in.
    """
    realmspinner_root = Path(zipguard.__file__).resolve().parent.parent.parent
    pattern = re.compile(r"zipguard\.BoundedZip\(")
    callers = []
    for path in sorted(realmspinner_root.rglob("*.py")):
        if path.name == "zipguard.py":
            continue
        if pattern.search(path.read_text(encoding="utf-8")):
            # A mode's headless package is ``modes/<mode>/engine/`` since P6;
            # "which engine" is the mode's name, not the word "engine".
            owner = path.parent.parent if path.parent.name == "engine" else path.parent
            callers.append(f"{owner.name}/{path.name}")
    return callers


def test_zipguard_docstring_names_every_module_that_opens_a_bounded_zip():
    callers = _bounded_zip_callers()
    # A sanity floor on the derivation itself, not the claim under test: if
    # this ever finds nothing, the regex (or the tree) moved, not the fix.
    assert len(callers) >= 6, callers

    doc = zipguard.__doc__ or ""
    missing = [c for c in callers if c not in doc]
    assert not missing, (
        "zipguard's module docstring does not name every module that opens a "
        f"bounded zip -- missing: {missing} (found via grep under studio/)"
    )
