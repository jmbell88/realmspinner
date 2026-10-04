"""Leftovers from the 2026-10-03 audit's Sirens findings (sirens-05, -15, -16).

Each is a caller the first pass of the fix missed: the export snapshot reader,
the sample picker, and the stem uniquifier's own generated suffix.
"""

from __future__ import annotations

from realmspinner.studio.modes.sirens import fileio as sirens_io
from realmspinner.studio.modes.sirens.engine import document as D
from realmspinner.studio.modes.sirens.engine import rsng

from .test_rsng import _song
from .test_sirens_mode import FakeCtx, _tab


def test_the_export_snapshot_reader_never_moves_the_shared_uid_counter(monkeypatch, tmp_path):
    raw = rsng.rsng_bytes(_song())
    # Below every uid the file carries, so a reserving read would move it.
    monkeypatch.setattr(D, "_next_uid", 0)

    sirens_io._export(raw, tmp_path)

    assert D._next_uid == 0


def test_the_sample_picker_on_a_saving_tab_is_refused_out_loud():
    ctx = FakeCtx()
    tab = _tab(ctx)
    tab.saving = True

    sirens_io.ask_sample(ctx, tab)

    assert not ctx.submitted
    assert ctx.toasts == [(sirens_io.SAMPLE_WHILE_SAVING, "error")]


def test_a_generated_stem_suffix_never_lands_on_a_name_already_typed():
    for stems in (
        ["Lead", "Lead-2", "Lead"],  # the typed -2 comes first
        ["Lead", "Lead", "Lead-2"],  # the typed -2 comes after the generated one
        ["lead-2", "Lead", "LEAD", "Lead"],  # and under the folded comparison
    ):
        out = sirens_io._unique(stems)
        assert len(out) == len(stems)
        assert len({name.casefold() for name in out}) == len(out), (stems, out)
        # A name that was not a duplicate of an earlier one is left alone.
        assert out[0] == stems[0]


def test_unique_still_numbers_plain_duplicates_from_two():
    assert sirens_io._unique(["Pulse 1", "Pulse 1", "Pulse 1"]) == [
        "Pulse 1",
        "Pulse 1-2",
        "Pulse 1-3",
    ]
