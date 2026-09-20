"""Regressions for ``service/files.py`` that are not about one edit path in
isolation, but about what happens when two doors are used at once.

Created for the 2026-09-07 audit because no test module sat beside
``service/files.py`` -- the existing edit-path coverage lives in
``test_editor_service.py``, which the audit's service-misc brief does not
list as a runnable file for this fix.
"""

from __future__ import annotations

import io
import threading
from pathlib import Path

import pytest
from PIL import Image

from realmspinner.service import files as svc_files
from realmspinner.service import jobs as svc_jobs


def _png(size=(64, 64), colour=(200, 30, 30, 255)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGBA", size, colour).save(buf, "PNG")
    return buf.getvalue()


def _reference(svc, **params) -> str:
    job_id = svc_jobs.create_job(svc, kind="text", prompt="x", output="reference", **params)["id"]
    svc.job_dir(job_id).mkdir(parents=True, exist_ok=True)
    (svc.job_dir(job_id) / "input.png").write_bytes(_png())
    svc.store.set_status(job_id, "done")
    return job_id


def test_concurrent_saves_do_not_let_the_undo_anchor_become_an_edited_copy(svc, monkeypatch):
    """service-03: ``save_edited_image``'s check-then-copy-then-write took no
    lock, so two concurrent saves could make ``input.orig.png`` -- undo's only
    anchor to the generated pixels -- a copy of an *already-edited* image
    instead of the generated one, unrecoverably. Regression for the
    2026-09-07 audit, service-03.

    The interleaving is forced rather than hoped for: both saves are made to
    observe "no backup yet" before either reads the pixels to back up, and
    the second save's read is made to happen only after the first save's
    write has already landed -- exactly the ordering the finding describes.

    Hooks ``Path.read_bytes`` rather than ``shutil.copyfile``: the 2026-09-08
    audit (service-05) moved the backup write off a bare ``shutil.copyfile``
    onto the same staged-temp-then-``os.replace`` shape every other write in
    this module already uses (``dest.read_bytes()`` then ``_staged_write``),
    so the read of the source pixels is now the moment whose timing matters.
    """
    job_id = _reference(svc)
    job_dir = svc.job_dir(job_id)
    dest = job_dir / "input.png"
    original = job_dir / "input.orig.png"

    real_exists = Path.exists
    real_read_bytes = Path.read_bytes
    real_replace = svc_files.os.replace

    ident_box: dict[str, int] = {}
    a_checked = threading.Event()
    b_checked = threading.Event()
    a_write_done = threading.Event()

    def fake_exists(self):
        result = real_exists(self)
        if self == original:
            ident = threading.get_ident()
            if ident == ident_box.get("a"):
                a_checked.set()
                b_checked.wait(2)
            elif ident == ident_box.get("b"):
                b_checked.set()
                a_checked.wait(2)
        return result

    def fake_read_bytes(self):
        if self == dest and threading.get_ident() == ident_box.get("b"):
            # The second save's read of the backup source must not happen
            # until the first save's write has already landed -- otherwise
            # both reads race the still-unedited dest and the bug this test
            # targets never manifests.
            a_write_done.wait(2)
        return real_read_bytes(self)

    def fake_replace(src, dst):
        result = real_replace(src, dst)
        if Path(dst) == dest and threading.get_ident() == ident_box.get("a"):
            a_write_done.set()
        return result

    monkeypatch.setattr(Path, "exists", fake_exists)
    monkeypatch.setattr(Path, "read_bytes", fake_read_bytes)
    monkeypatch.setattr(svc_files.os, "replace", fake_replace)

    errors: list[BaseException] = []

    def run(colour):
        try:
            svc_files.save_edited_image(svc, job_id, _png(colour=colour))
        except BaseException as exc:  # noqa: BLE001 - surfaced via `errors`
            errors.append(exc)

    t_a = threading.Thread(target=run, args=((10, 20, 30, 255),))
    t_b = threading.Thread(target=run, args=((40, 50, 60, 255),))
    t_a.start()
    ident_box["a"] = t_a.ident
    t_b.start()
    ident_box["b"] = t_b.ident
    t_a.join(5)
    t_b.join(5)

    assert not errors, errors
    with Image.open(original) as im:
        anchor_pixel = im.convert("RGBA").getpixel((0, 0))
    # The anchor must still be the pixels _reference() wrote before either
    # edit -- not either racing save's colour, which is what an unlocked
    # check-then-copy-then-write produces.
    assert anchor_pixel == (200, 30, 30, 255)


def test_a_failed_backup_copy_leaves_no_truncated_undo_anchor(svc, monkeypatch):
    """service-05 (the 2026-09-08 audit): the one-time backup of the
    generated reference (``input.orig.png``, undo's only anchor to the
    pre-edit pixels) used to be written with a bare ``shutil.copyfile``
    rather than through this module's own staged-temp-then-``os.replace``
    pattern every other write here uses -- so a crash or a write failure
    partway through the copy left a truncated ``input.orig.png`` on disk.
    ``save_edited_image`` gates the backup on ``original.exists()``, so a
    truncated file is never retried: it becomes the permanent "revert to
    original" target, and a later ``revert_reference`` installs it onto the
    served ``input.png``.
    """
    job_id = _reference(svc)
    job_dir = svc.job_dir(job_id)
    original = job_dir / svc_files.ORIGINAL

    real_write_bytes = Path.write_bytes

    def flaky_write_bytes(self, data):
        if svc_files.ORIGINAL in self.name:
            # A crash partway through writing the backup: some bytes reach
            # disk, then the write is interrupted -- the failure mode the
            # finding names. Only fires on the backup's own temp sibling (its
            # name carries "input.orig.png"); the edit's own write, onto a
            # temp sibling of "input.png", is untouched.
            with open(self, "wb") as fh:
                fh.write(data[: len(data) // 2])
            raise OSError("simulated disk failure mid-copy")
        return real_write_bytes(self, data)

    monkeypatch.setattr(Path, "write_bytes", flaky_write_bytes)

    with pytest.raises(OSError):
        svc_files.save_edited_image(svc, job_id, _png(colour=(9, 9, 9, 255)))

    # No truncated file at the served backup name: the write staged to a temp
    # sibling and never replaced onto it, so the failure lands there instead.
    assert not original.exists()
    # And the temp sibling itself is gone too -- _staged_write's own
    # ``finally`` removes it whether the write succeeded or not.
    assert list(job_dir.glob(f".{svc_files.ORIGINAL}.*.tmp")) == []


def test_save_edited_image_drops_the_stale_reference_report_when_remeasurement_fails(
    svc, monkeypatch
):
    """service-02 (the 2026-09-08 audit): ``_remeasure``'s reference branch
    called ``reference.measure_file(src).as_dict()`` with no exception
    handling, unlike the parallel tile branch three lines above it (which
    catches, logs and drops the stale key) -- so a measurement failure after
    a hand edit has already changed ``input.png`` on disk used to abort the
    whole ``merge_params`` call. ``params["reference_report"]`` (already
    stale by then) was left describing pixels the user no longer has, and
    ``hand_edited`` was never recorded either, because the same call carries
    both.
    """
    from realmspinner.pipelines import reference as reference_mod

    job_id = _reference(svc)
    # A report already on the row, from the generation this edit replaces --
    # what a failed remeasurement must not leave standing.
    svc.store.merge_params(job_id, {"reference_report": {"ok": True, "reasons": []}})

    def boom(_path):
        raise ValueError("corrupt reference")

    monkeypatch.setattr(reference_mod, "measure_file", boom)

    svc_files.save_edited_image(svc, job_id, _png(colour=(1, 2, 3, 255)))

    params = svc.store.get(job_id)["params"]
    assert "reference_report" not in params
    assert params["hand_edited"] is True


# -- DERIVED_AUDIO / DERIVED_IMAGE: the allowlist rule, both directions ------
#
# MEDIA is what keeps a caller-supplied name off the filesystem (the module
# docstring's own claim); a format list that named something MEDIA does not
# carry would be underivable and unserveable, and a MEDIA row nothing else
# will ever offer is dead weight nobody notices. Both halves are worth
# pinning independently, the way ``test_derive_2d.py``'s
# ``test_every_2d_artifact_is_in_the_media_allowlist`` already pins one half
# of DERIVED_2D.


def test_every_derived_audio_name_is_in_the_media_allowlist():
    for name in svc_files.DERIVED_AUDIO:
        assert name in svc_files.MEDIA, name


def test_every_derived_image_name_is_in_the_media_allowlist():
    for name in svc_files.DERIVED_IMAGE:
        assert name in svc_files.MEDIA, name


def test_track_wav_is_a_member_of_its_own_derived_audio_list():
    # pipelines.audioout.convert's docstring explains why: it lets the
    # Library's Convert door and the Downloads grid offer WAV beside
    # FLAC/MP3/OGG/AIFF as one uniform list under one lock, rather than a
    # WAV-shaped special case in both.
    assert "track.wav" in svc_files.DERIVED_AUDIO


def test_derived_image_is_exactly_the_web_reencodings_of_input_png():
    # A name added here and nowhere else (pipelines.imageout.FORMATS,
    # artifacts.ARTIFACTS_2D/TILE/TILESHEET) is a button that would answer
    # NotReady forever -- the same argument test_every_2d_artifact_has_a_
    # derivation makes for DERIVED_2D.
    from realmspinner.pipelines import imageout

    assert set(svc_files.DERIVED_IMAGE) == set(imageout.FORMATS)


def test_asking_whether_track_wav_is_ready_does_not_recurse_forever():
    # track.wav is now a member of DERIVED_AUDIO (see its docstring), and
    # ``ready``'s ``name in DERIVED_AUDIO`` branch recurses into
    # ``ready(job, job_dir, "track.wav")`` for every other name in that tuple.
    # Without track.wav's own branch checked *first*, asking about track.wav
    # itself would recurse into that same branch forever.
    job = {"status": "done"}
    job_dir = Path("nonexistent-job-dir")
    assert svc_files.ready(job, job_dir, "track.wav") is False
