"""The 2026-09-23 (second run) audit, sirens-01/02 and P65 item 2 ("sirens-02"
in ``dev/TODO.md`` P65).

``sirens-01``: ``rsng._patterns_from`` reads the instrument column of each
pattern's cells. When the file's instrument list needed renumbering (an id
outside ``0..MAX_INSTRUMENTS-1``), a cell naming no instrument in the
renumbered map is blanked -- ``test_a_cell_naming_no_instrument_in_a_
renumbered_file_is_blanked`` in ``test_rsng.py`` already covers that. But
when the instrument list is *legal* (every uid already in range, so
``_instruments_from`` returns an empty ``remap``), the column was only
``np.clip``-ped into ``0..MAX_INSTRUMENTS-1`` -- a legal-range id that still
names no instrument this file's own list holds (a value past the real
instrument count, or a hand edit) folded onto slot 127 and played whatever
real instrument happens to sit there, instead of being blanked the way the
renumbered path already blanks the same situation.

``sirens-02``: toggling loop on an envelope whose release covers the whole
sequence (``release == len(sequence)``, i.e. no sustain body at all) put an
invisible loop marker inside the release-only tail instead of routing it
through ``marker_bounds`` the way every other loop placement does.

P65 item 2 ("sirens-02" in ``dev/TODO.md`` P65): ``SongDoc.set_song`` had no
busy check, unlike its sibling mutators, so a caller could commit a change
mid-save.
"""

from __future__ import annotations

import pytest

from realmspinner.studio.modes.sirens import mode as sirens_mode
from realmspinner.studio.modes.sirens.engine import document as D
from realmspinner.studio.modes.sirens.engine import envelope, rsng

from .test_rsng import _repack_all, _song
from .test_sirens_mode import FakeCtx, _Done, _tab

# --- sirens-01: rsng.py ------------------------------------------------------


def test_read_rsng_blanks_an_out_of_range_instrument_cell_even_when_the_instrument_list_is_legal():
    """A legal instrument list (every stored uid already in
    ``0..MAX_INSTRUMENTS-1``, the common case -- every file this build itself
    writes) takes the ``if remap:`` branch's ``else``, which used to be a bare
    ``np.clip``. A cell holding 4242 -- an id no instrument in this file's own
    (one-instrument) list actually has -- was folded onto slot 127.

    Fails against the unfixed code: with instrument 0 renamed "Lead" and no
    instrument at uid 127, this asserted ``== -1`` and got ``127`` instead --
    a value inside the legal range, and (if this build's own document ever
    minted an instrument at exactly that slot) a real, unrelated instrument's
    uid.
    """
    doc = _song()
    stored = [one.uid for one in doc.instruments]
    assert all(0 <= uid < D.MAX_INSTRUMENTS for uid in stored)  # legal, unrenumbered

    def edit(manifest, arrays):
        # No manifest edit: the instrument list is left legal on purpose, so
        # ``_instruments_from`` returns an empty ``remap`` and this exercises
        # the *other* branch from ``test_a_cell_naming_no_instrument_in_a_
        # renumbered_file_is_blanked``.
        arrays["patterns/0.npy"][0, 0, D.INSTRUMENT] = 4_242

    back = rsng.read_rsng(_repack_all(doc, edit))
    assert [one.uid for one in back.instruments] == stored  # still legal, still unrenumbered
    assert back.patterns[0].cells[0, 0, D.INSTRUMENT] == -1


def test_read_rsng_blanks_a_legal_range_id_naming_no_instrument_in_a_legal_file():
    """The sharper case the finding names by number: 127 is not "out of
    range" at all -- it is the top of the legal ``0..MAX_INSTRUMENTS-1``
    span, exactly the value ``np.clip``'s upper bound folded 4242 onto above.
    A file with one instrument (uid 0) and a cell naming 127 must still play
    nothing for that cell, not slot 127's would-be resident.
    """
    doc = _song()

    def edit(manifest, arrays):
        arrays["patterns/0.npy"][0, 0, D.INSTRUMENT] = D.MAX_INSTRUMENTS - 1

    back = rsng.read_rsng(_repack_all(doc, edit))
    assert back.patterns[0].cells[0, 0, D.INSTRUMENT] == -1


# --- sirens-02: envelope.py ---------------------------------------------------


def test_toggled_loop_never_lands_inside_a_release_only_tail():
    """``release == 0`` makes the *whole* sequence release tail (the module
    docstring: "release == 0 makes every value tail material and a held note
    silent"), so ``marker_bounds(sequence, "loop")`` answers ``(0, -1)`` --
    the same "no room" a drag into it is already refused for by ``moved``.
    ``toggled`` used to land an enabled loop at step 0 unconditionally,
    without asking ``marker_bounds`` at all -- putting a loop marker inside a
    tail the graph never draws a live loop in and the engine never reaches.

    Fails against the unfixed code: ``seq.loop`` comes back ``0`` -- inside
    the release-only tail -- instead of ``-1`` (the toggle refusing, the way
    a one-step sequence's release toggle already refuses next to it).
    """
    from realmspinner.studio.modes.sirens.engine import instruments as inst

    seq = inst.Sequence(values=(0, 4, 7, 4), loop=-1, release=0)
    toggled = envelope.toggled(seq, "loop")
    assert toggled.loop == -1


def test_toggled_loop_still_lands_at_zero_when_there_is_room():
    """The fix must not touch the ordinary case: a sequence with a live
    sustain half (``release`` at or past the end, or unset) still gets its
    loop at step 0 on toggle, exactly as before.
    """
    from realmspinner.studio.modes.sirens.engine import instruments as inst

    seq = inst.Sequence(values=(0, 4, 7, 4), loop=-1, release=-1)
    toggled = envelope.toggled(seq, "loop")
    assert toggled.loop == 0


# --- P65 item 2 ("sirens-02"): document.py -----------------------------------


def test_set_song_refuses_while_a_save_is_running(tmp_path):
    """``SongDoc.set_song`` had no busy check -- its one door, "Loop the
    song", is greyed while ``tab.saving`` (``tab.busy``) is true, but the
    setter itself took a mid-save change from any caller that reached it
    directly, not just through the greyed door.

    Fails against the unfixed code: ``doc.set_song(loop_order=0)`` succeeds
    (returns ``True``, mutates ``doc.loop_order``) while ``save_to`` has the
    tab locked and the encode already committed to disk, instead of raising.
    """
    ctx = FakeCtx()
    tab = _tab(ctx)
    doc = tab.doc
    doc.set_order([doc.patterns[0].uid, doc.patterns[0].uid])
    assert not doc.busy

    sirens_mode.save_to(ctx, tab, tmp_path / "song.rsng")
    assert tab.saving  # the door's own lock, unaffected by this fix
    assert doc.busy  # the fix: the document knows a save is running too

    with pytest.raises(ValueError, match="save is running"):
        doc.set_song(loop_order=0)

    # The lock lifts exactly when the door's own does -- landing the save...
    sirens_mode.on_task_done(ctx, _Done(f"sirens-save:{tab.uid}", ctx.result))
    assert not tab.saving
    assert not doc.busy
    assert doc.set_song(loop_order=0)  # ordinary use is untouched


def test_set_song_unlocks_after_a_failed_save_too():
    """A failed save must not leave ``set_song`` locked out forever, the same
    reason ``tab.saving`` itself is cleared on a failure rather than left set
    (``test_a_failed_save_unlocks_the_tab``, ``test_sirens_mode.py``)."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    tab.saving = True
    tab.doc.busy = True
    sirens_mode.on_task_failed(ctx, _Done(f"sirens-save:{tab.uid}", message="disk full"))
    assert not tab.saving
    assert not tab.doc.busy
    assert tab.doc.set_song(title="ok")
