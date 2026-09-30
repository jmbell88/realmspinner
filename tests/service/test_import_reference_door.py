"""``svc_jobs.import_reference`` as the Mesh stage's upload door.

An uploaded or dropped image becomes a finished reference row, and the ordinary
cutout-check -> ``promote_candidates`` path takes it from there.
"""

from __future__ import annotations

import io
import os

import pytest
from PIL import Image

from realmspinner.service import jobs as svc_jobs
from realmspinner.service.errors import Invalid


def _png(size=(64, 64), colour=(200, 30, 30, 255)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGBA", size, colour).save(buf, "PNG")
    return buf.getvalue()


def test_the_row_is_a_finished_reference_holding_input_png(svc):
    job_id = svc_jobs.import_reference(svc, _png(), name="hero")["id"]
    job = svc.store.get(job_id)
    assert (job["stage"], job["status"]) == ("reference", "done")
    assert job["params"]["imported"] is True
    assert (svc.job_dir(job_id) / "input.png").is_file()


def test_input_png_is_published_by_rename_and_leaves_no_temp(svc, monkeypatch):
    real = os.replace
    renamed = []

    def spy(src, dst):
        renamed.append((os.fspath(src), os.fspath(dst)))
        return real(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    job_id = svc_jobs.import_reference(svc, _png())["id"]
    served = str(svc.job_dir(job_id) / "input.png")
    assert [d for _s, d in renamed if d == served], "input.png was written in place"
    assert all(s != served for s, _d in renamed)
    assert sorted(p.name for p in svc.job_dir(job_id).iterdir()) == ["input.png"]


def _subject_png() -> bytes:
    image = Image.new("RGB", (64, 64), (230, 230, 230))
    image.paste((20, 30, 40), (16, 16, 48, 48))
    buf = io.BytesIO()
    image.save(buf, "PNG")
    return buf.getvalue()


def test_promote_to_model_accepts_the_row_without_force(svc):
    job_id = svc_jobs.import_reference(svc, _subject_png())["id"]
    report = svc.store.get(job_id)["params"]["reference_report"]
    assert report.get("ok") is not False, "fixture must be a composition the gate allows"
    out = svc_jobs.promote_to_model(svc, job_id)
    assert out["parent"] == job_id


def test_a_bad_image_is_refused_with_the_image_field(svc):
    with pytest.raises(Invalid) as exc:
        svc_jobs.import_reference(svc, b"not an image")
    assert exc.value.field == "image"
    assert not any(svc.config.data_dir.glob("*/input.png"))
