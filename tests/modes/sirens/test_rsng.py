"""The ``.rsng`` container: what round-trips, and what a bad file cannot do.

The refusal tests are the substantial half. A ``.rsng`` is a file a user can be
handed -- from a bundle, a forum, a colleague -- so every door it opens is one
somebody else wrote the bytes for.
"""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest

from realmspinner.studio.modes.sirens.engine import document as D
from realmspinner.studio.modes.sirens.engine import instruments as inst
from realmspinner.studio.modes.sirens.engine import rsng, synth


def _song() -> D.SongDoc:
    doc = D.new_song()
    uid = doc.patterns[0].uid
    for row, note in enumerate((48, 52, 55, 60)):
        doc.set_cell(uid, row * 4, 0, D.NOTE, note)
        doc.set_cell(uid, row * 4, 0, D.INSTRUMENT, doc.instruments[0].uid)
    doc.set_song(title="Overworld", author="somebody", tempo=140, speed=4)
    doc.add_oneshot("jump")
    doc.set_sample("kick", np.linspace(1.0, -1.0, 600, dtype=np.float32))
    doc.update_instrument(
        doc.instruments[0].uid,
        arpeggio=inst.Sequence(values=(0, 4, 7), loop=0),
        name="Lead",
    )
    return doc


def _repack(doc: D.SongDoc, edit) -> bytes:
    """Reopen the archive, hand the manifest to ``edit``, write it back."""
    raw = rsng.rsng_bytes(doc)
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        members = {name: zf.read(name) for name in zf.namelist()}
    manifest = json.loads(members[rsng.MANIFEST])
    edit(manifest)
    members[rsng.MANIFEST] = json.dumps(manifest).encode()
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, payload in members.items():
            zf.writestr(name, payload)
    return out.getvalue()


# --- the round trip -----------------------------------------------------------


def test_everything_survives_a_round_trip():
    doc = _song()
    back = rsng.read_rsng(rsng.rsng_bytes(doc))
    assert (back.title, back.author, back.tempo, back.speed) == (
        doc.title,
        doc.author,
        doc.tempo,
        doc.speed,
    )
    assert [one.uid for one in back.channels] == [one.uid for one in doc.channels]
    assert back.order == doc.order
    assert [one.name for one in back.oneshots] == [one.name for one in doc.oneshots]
    assert np.array_equal(back.patterns[0].cells, doc.patterns[0].cells)
    assert back.instruments[0].arpeggio == doc.instruments[0].arpeggio
    assert back.instruments[0].name == "Lead"


def test_a_document_that_has_just_been_opened_is_not_unsaved():
    assert not rsng.read_rsng(rsng.rsng_bytes(_song())).dirty


def test_two_saves_of_an_unchanged_document_are_byte_identical():
    doc = _song()
    assert rsng.rsng_bytes(doc) == rsng.rsng_bytes(doc)


def test_rsng_bytes_actually_deflates_its_members():
    """sirens-01 (the 2026-09-11 audit): ``rsng_bytes`` opened the archive with
    ``ZIP_DEFLATED`` but wrote every member through a bare ``ZipInfo``, whose
    ``compress_type`` defaults to ``ZIP_STORED`` and was never overridden -- so
    every ``.rsng`` ever saved was fully uncompressed despite the archive's own
    declared intent. Asserted against the actual member metadata and size, not
    against a helper existing: a test that only checked ``_member`` was called
    would still pass if the helper itself forgot the line.
    """
    doc = _song()
    # Zeroed out for a highly compressible grid -- the common case the finding
    # measured (~30x for a sparse pattern) and the shape where STORED and
    # DEFLATED are unmistakably different sizes.
    doc.patterns[0].cells[:] = 0
    raw = rsng.rsng_bytes(doc)
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        infos = zf.infolist()
        assert infos, "the archive has no members to check"
        for info in infos:
            assert info.compress_type == zipfile.ZIP_DEFLATED, info.filename
        pattern_info = zf.getinfo(f"{rsng.PATTERN_DIR}/0.npy")
        assert pattern_info.compress_size < pattern_info.file_size // 2


def test_opening_a_file_and_saving_it_produces_the_file_that_was_opened():
    """Which is what makes a ``.rsng`` in a repository diffable, and what a
    renumbering-on-read would have destroyed."""
    raw = rsng.rsng_bytes(_song())
    assert rsng.rsng_bytes(rsng.read_rsng(raw)) == raw


def test_a_reopened_song_renders_the_same_audio():
    doc = _song()
    before, _ = synth.render(doc)
    after, _ = synth.render(rsng.read_rsng(rsng.rsng_bytes(doc)))
    assert before.tobytes() == after.tobytes()


def test_a_uid_handed_out_after_an_open_cannot_collide_with_the_file():
    """Without ``document.reserve_uid``, the first pattern added after an open
    takes a uid the file is already using and the order list starts pointing at
    two different patterns through one number."""
    doc = _song()
    D._next_uid = 0
    back = rsng.read_rsng(rsng.rsng_bytes(doc))
    taken = {one.uid for one in back.patterns} | {one.uid for one in back.channels}
    assert back.add_pattern().uid not in taken


def test_the_rendered_audio_is_not_in_the_file():
    """A ``.rsng`` is the composition. Storing a render would let the file
    disagree with the notes beside it."""
    with zipfile.ZipFile(io.BytesIO(rsng.rsng_bytes(_song()))) as zf:
        names = zf.namelist()
    assert not any(name.endswith(".wav") and name.startswith("render") for name in names)
    assert rsng.MANIFEST in names


def test_members_are_numbered_rather_than_named_after_user_text():
    """A member named after a sample key is how ``../`` gets into an archive."""
    doc = _song()
    doc.set_sample("../../etc/passwd", np.zeros(8, dtype=np.float32))
    with zipfile.ZipFile(io.BytesIO(rsng.rsng_bytes(doc))) as zf:
        assert all(".." not in name for name in zf.namelist())
    back = rsng.read_rsng(rsng.rsng_bytes(doc))
    assert "../../etc/passwd" in back.samples


# --- what a file cannot make this build do ------------------------------------


def test_something_that_is_not_a_zip_is_refused():
    with pytest.raises(ValueError, match="not a Realmspinner song"):
        rsng.read_rsng(b"nope")


def test_a_zip_with_no_manifest_is_refused():
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr("hello.txt", "hi")
    with pytest.raises(ValueError, match="not a Realmspinner song"):
        rsng.read_rsng(out.getvalue())


def test_a_file_from_a_newer_build_says_so_rather_than_guessing():
    raw = _repack(_song(), lambda m: m.__setitem__("version", rsng.VERSION + 1))
    with pytest.raises(ValueError, match="newer version"):
        rsng.read_rsng(raw)


def test_an_archive_claiming_more_than_the_ceiling_is_refused_before_any_read(
    monkeypatch,
):
    monkeypatch.setattr(rsng, "MAX_DECOMPRESSED_BYTES", 16)
    with pytest.raises(ValueError, match="past the"):
        rsng.read_rsng(rsng.rsng_bytes(_song()))


def test_a_pattern_whose_shape_disagrees_with_the_manifest_is_refused():
    """The ``.npy`` header decides the allocation and the archive's directory
    cannot see it, which is the whole reason this goes through ``npyguard``."""
    raw = _repack(_song(), lambda m: m["patterns"][0].__setitem__("rows", 999))
    with pytest.raises(ValueError, match="patterns are"):
        rsng.read_rsng(raw)


def test_a_channel_kind_this_build_cannot_play_is_named_in_the_refusal():
    raw = _repack(_song(), lambda m: m["channels"][0].__setitem__("kind", "theremin"))
    with pytest.raises(ValueError, match="theremin"):
        rsng.read_rsng(raw)


def test_a_sequence_longer_than_the_ceiling_is_refused():
    def edit(manifest):
        manifest["instruments"][0]["volume"]["values"] = list(
            range(inst.MAX_SEQUENCE_LEN + 5)
        )

    with pytest.raises(ValueError, match="past the"):
        rsng.read_rsng(_repack(_song(), edit))


def test_a_manifest_that_is_not_an_object_is_refused():
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr(rsng.MANIFEST, "[1, 2, 3]")
    with pytest.raises(ValueError, match="malformed"):
        rsng.read_rsng(out.getvalue())


def test_a_missing_pattern_member_is_named_in_the_refusal():
    doc = _song()
    raw = rsng.rsng_bytes(doc)
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        members = {n: zf.read(n) for n in zf.namelist() if not n.startswith("patterns/")}
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as zf:
        for name, payload in members.items():
            zf.writestr(name, payload)
    with pytest.raises(ValueError, match="missing patterns/"):
        rsng.read_rsng(out.getvalue())


def test_cell_values_a_hand_edited_file_could_hold_are_clipped():
    """A note of 9000 would index past the end of the frequency table at render
    time, three layers away from the file that said it."""
    doc = D.new_song()
    doc.patterns[0].cells[0, 0, D.NOTE] = 30000
    back = rsng.read_rsng(rsng.rsng_bytes(doc))
    assert back.patterns[0].cells[0, 0, D.NOTE] <= 255
    pcm, _ = synth.render(back)
    assert np.isfinite(pcm).all()


def test_an_order_entry_naming_a_pattern_that_is_gone_is_dropped_not_refused():
    """The rest of the song is intact and readable, and refusing the whole
    document over one stale number would lose all of it."""
    raw = _repack(_song(), lambda m: m["order"].append(999999))
    back = rsng.read_rsng(raw)
    assert 999999 not in back.order
    assert back.order


def test_a_loop_point_past_the_order_is_dropped():
    raw = _repack(_song(), lambda m: m.__setitem__("loop_order", 50))
    assert rsng.read_rsng(raw).loop_order == -1


def test_a_song_with_no_channels_opens_on_the_defaults():
    """A refusal here would lose a file whose channel list was truncated, and
    there is a correct answer available."""
    doc = D.new_song()
    raw = _repack(doc, lambda m: m.__setitem__("channels", []))
    assert len(rsng.read_rsng(raw).channels) == len(D.default_channels())


# --- the instrument id space --------------------------------------------------


def _repack_all(doc: D.SongDoc, edit) -> bytes:
    """``_repack``, but ``edit`` gets the pattern arrays as well as the manifest.

    Needed to forge a file an older build would have written: its instrument ids
    came from the process-global counter, so both the manifest *and* the cells
    that name them are out of the id space this build reads.
    """
    raw = rsng.rsng_bytes(doc)
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        members = {name: zf.read(name) for name in zf.namelist()}
    manifest = json.loads(members[rsng.MANIFEST])
    arrays = {
        name: np.load(io.BytesIO(payload))
        for name, payload in members.items()
        if name.endswith(".npy")
    }
    edit(manifest, arrays)
    members[rsng.MANIFEST] = json.dumps(manifest).encode()
    for name, array in arrays.items():
        buffer = io.BytesIO()
        np.lib.format.write_array(buffer, np.ascontiguousarray(array), version=(1, 0))
        members[name] = buffer.getvalue()
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, payload in members.items():
            zf.writestr(name, payload)
    return out.getvalue()


def test_an_id_outside_the_space_is_renumbered_and_its_cells_come_with_it():
    """A file from a build that minted instrument ids globally still plays the
    instruments it names: refusing it would lose the song, and renumbering the
    list without the cells would lose which instrument every note used."""
    doc = _song()
    was = doc.instruments[0].uid

    def edit(manifest, arrays):
        manifest["instruments"][0]["uid"] = 20_000
        plane = arrays["patterns/0.npy"][:, :, D.INSTRUMENT]
        plane[plane == was] = 20_000

    back = rsng.read_rsng(_repack_all(doc, edit))
    assert [one.uid for one in back.instruments] == list(range(len(back.instruments)))
    named = back.patterns[0].cells[0, 0, D.INSTRUMENT]
    assert named == 0 and back.instrument(named).name == "Lead"


def test_a_file_this_build_wrote_is_not_renumbered_at_all():
    """The map is empty when every id is already legal, so a save/open/save
    round trip stays byte-identical and a ``.rsng`` stays diffable."""
    doc = _song()
    raw = rsng.rsng_bytes(doc)
    assert rsng.rsng_bytes(rsng.read_rsng(raw)) == raw


def test_a_cell_naming_no_instrument_in_a_renumbered_file_is_blanked():
    """Left as a raw number it would land on whichever slot the renumbering
    has since put in its place -- a note playing an instrument nobody chose."""
    doc = _song()

    def edit(manifest, arrays):
        manifest["instruments"][0]["uid"] = 9_000
        arrays["patterns/0.npy"][0, 0, D.INSTRUMENT] = 4_242

    back = rsng.read_rsng(_repack_all(doc, edit))
    assert back.patterns[0].cells[0, 0, D.INSTRUMENT] == -1


def test_a_manifest_with_a_duplicate_instrument_uid_is_refused_not_silently_misattributed():
    """The 2026-09-13 audit, finding sirens-03. A manifest listing one uid for
    two instruments used to be renumbered through a last-writer-wins ``dict``
    (``{uid: index for ...}``), so cells naming the first instrument played
    the second -- or went silent, if the first uid's slot was blank -- with no
    error anywhere. ``_samples_from`` already refuses a duplicate sample *key*
    by name; this is the same refusal for an instrument's uid."""
    doc = _song()

    def edit(manifest):
        manifest["instruments"].append(dict(manifest["instruments"][0]))

    raw = _repack(doc, edit)
    with pytest.raises(ValueError, match="instrument.*twice"):
        rsng.read_rsng(raw)


def test_read_rsng_refuses_a_duplicate_pattern_uid():
    """The 2026-09-16 audit. Unlike ``_instruments_from`` (refused since the
    2026-09-13 audit) and ``_samples_from`` (refused by key since always),
    ``_patterns_from`` accepted a manifest with two patterns sharing one uid
    with no check -- and every uid-addressed lookup resolves to the first
    match, while ``_detach_pattern`` filtered by ``uid !=`` and would delete
    *every* entry sharing that uid in one call. Opening such a file and
    deleting the one pattern the user selected would silently destroy a
    second, untouched pattern with no error, no toast, and no way to tell
    from the grid."""
    doc = _song()

    def edit(manifest):
        manifest["patterns"].append(dict(manifest["patterns"][0]))

    raw = _repack(doc, edit)
    with pytest.raises(ValueError, match="pattern.*twice"):
        rsng.read_rsng(raw)


def test_read_rsng_refuses_a_duplicate_channel_uid():
    """Same shape as the pattern case, for channels."""
    doc = _song()

    def edit(manifest):
        manifest["channels"].append(dict(manifest["channels"][0]))

    raw = _repack(doc, edit)
    with pytest.raises(ValueError, match="channel.*twice"):
        rsng.read_rsng(raw)


def test_read_rsng_refuses_a_duplicate_oneshot_uid():
    """Same shape as the pattern case, for one-shots."""
    doc = _song()

    def edit(manifest):
        manifest["oneshots"].append(dict(manifest["oneshots"][0]))

    raw = _repack(doc, edit)
    with pytest.raises(ValueError, match="sound effect.*twice"):
        rsng.read_rsng(raw)


def test_a_non_numeric_channel_uid_is_reported_as_malformed_not_a_raw_valueerror():
    """sirens-02 (the 2026-09-20 audit). Four of ``rsng.py``'s five collection
    readers (``_channels_from``, ``_patterns_from``, ``_oneshots_from``,
    ``_samples_from``) parsed a manifest number with a bare ``int()`` instead
    of the module's own ``_int()`` helper, which ``_instruments_from`` already
    wraps the identical call in -- so a non-numeric uid surfaced as Python's
    own "invalid literal for int() with base 10: …" instead of the same
    "this song's manifest is malformed" every sibling refusal gives. Reproduced
    against the unfixed code: this raised a bare ``ValueError`` whose message
    was the Python exception text, not ``_MALFORMED``.
    """
    doc = _song()

    def edit(manifest):
        manifest["channels"][0]["uid"] = "not-a-number"

    raw = _repack(doc, edit)
    with pytest.raises(ValueError, match="malformed"):
        rsng.read_rsng(raw)


def test_the_reserved_uid_high_water_mark_ignores_the_instruments(monkeypatch):
    """Their ids never came out of the global counter, so reserving above one
    would walk that counter toward its own ceiling for nothing -- which is the
    walk the instrument column could not survive in the first place."""
    doc = _song()
    raw = _repack_all(
        doc, lambda m, a: m["instruments"][0].__setitem__("uid", 20_000)
    )
    # Set directly, and restored after: the counter is a process global, and a
    # test that moved it would be reaching into every test that ran later.
    monkeypatch.setattr(D, "_next_uid", 100)
    rsng.read_rsng(raw)
    assert D.new_uid() < 20_000
