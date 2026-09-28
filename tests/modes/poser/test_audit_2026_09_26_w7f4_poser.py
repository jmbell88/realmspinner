"""Regression tests for the 2026-09-26 audit's wave-7 fixes to Poser, findings
poser-mode-06 and poser-mode-07 -- the two that had no clean existing home.

poser-mode-08, poser-render-02, poser-render-03 and poser-render-04 are
covered beside the tests that already exercise the same functions
(``test_sheet_mode.py``, ``test_send_door.py`` and ``test_pixel_report_lines.py``
respectively); poser-engine-01, poser-mode-09 and poser-render-01 are
comment/doc-only and have no executable claim to pin.
"""

from __future__ import annotations

from types import SimpleNamespace

from realmspinner.studio.modes.poser import mode as poser_mode
from realmspinner.studio.modes.poser.ui.panes.clips import _no_outgoing_segment_reason

# --- poser-mode-06: a stale import report/skip list outliving its import ----


def test_switching_template_clears_a_stale_import_report_and_skip_list():
    """``_reset_for_template`` used to leave ``clip_import_reports`` and
    ``clip_import_skipped`` untouched, so switching templates and importing
    again could show a report drawn from a completely different template's
    clips, still sitting in the pane from before the switch."""
    state = poser_mode.PoserState()
    state.clip_import_reports = [{"map": "hips->Hips"}]
    state.clip_import_skipped = ["BigJump: 1200 frames exceeds the 900-frame limit"]

    poser_mode._reset_for_template(state, "quadruped")

    assert state.clip_import_reports == []
    assert state.clip_import_skipped == []


def test_a_zero_clip_import_replaces_a_stale_report_left_by_an_earlier_import():
    """``adopt_imported_clips`` used to return, before ever touching
    ``clip_import_reports``, the moment an import came back with no clips at
    all (every action skipped, or a source with nothing left to sample) -- so
    a clean-but-empty import left whatever a previous, unrelated import had
    reported still on screen next to the fresh "Imported 0 clip(s)" toast."""
    state = poser_mode.PoserState()
    state.template = "humanoid"
    state.clips = {"poses": [], "clips": []}
    ctx = SimpleNamespace(state=SimpleNamespace(poser=state, preview={}))
    state.clip_import_reports = [{"map": "hips->Hips", "clip": "old_run"}]

    poser_mode.adopt_imported_clips(
        ctx, {"template": "humanoid", "clips": [], "skipped": []}
    )

    assert state.clip_import_reports == []


# --- poser-mode-07: the last key of an open clip has no outgoing segment ----


def test_no_outgoing_segment_reason_is_empty_for_every_key_but_an_open_clips_last():
    # Three keys, two segments (an open clip): keys 0 and 1 both have an
    # outgoing segment to time.
    assert _no_outgoing_segment_reason(0, 3, False, 2) == ""
    assert _no_outgoing_segment_reason(1, 3, False, 2) == ""


def test_no_outgoing_segment_reason_disables_only_an_open_clips_last_key():
    """The 2026-09-26 audit, finding poser-mode-07: before this, the "Frames
    after this key" field silently retargeted the *previous* segment for the
    last key of an open clip instead of saying there was nothing after it."""
    assert (
        _no_outgoing_segment_reason(2, 3, False, 2)
        == "The last key of an open clip has nothing after it to time."
    )


def test_no_outgoing_segment_reason_is_empty_for_a_closed_clips_last_key():
    """A looping clip's last key loops back to the first, so ``segments`` has
    one entry per key rather than one fewer -- a real outgoing segment."""
    assert _no_outgoing_segment_reason(2, 3, True, 3) == ""


def test_no_outgoing_segment_reason_is_empty_with_no_keys_at_all():
    assert _no_outgoing_segment_reason(0, 0, False, 0) == ""
