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

from PIL import Image

from warlock.service import files as svc_files
from warlock.service import jobs as svc_jobs


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
    observe "no backup yet" before either copies, and the second save's copy
    is made to happen only after the first save's write has already landed --
    exactly the ordering the finding describes.
    """
    job_id = _reference(svc)
    job_dir = svc.job_dir(job_id)
    dest = job_dir / "input.png"
    original = job_dir / "input.orig.png"

    real_exists = Path.exists
    real_copyfile = svc_files.shutil.copyfile
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

    def fake_copyfile(src, dst):
        if Path(dst) == original and threading.get_ident() == ident_box.get("b"):
            # The second save's copy must not happen until the first save's
            # write has landed -- otherwise both copies race the still-
            # unedited dest and the bug this test targets never manifests.
            a_write_done.wait(2)
        return real_copyfile(src, dst)

    def fake_replace(src, dst):
        result = real_replace(src, dst)
        if Path(dst) == dest and threading.get_ident() == ident_box.get("a"):
            a_write_done.set()
        return result

    monkeypatch.setattr(Path, "exists", fake_exists)
    monkeypatch.setattr(svc_files.shutil, "copyfile", fake_copyfile)
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
