"""Presence-vs-usable regressions for the service layer's small utility
modules: ``system.py`` (health, the trellis log tail), ``verdicts.py`` (filing
a label) and ``judge.py`` (the quality probes).

Created for the 2026-09-08 audit, service-07: four sites in this segment
gated a "ready to read" decision on ``Path.exists()`` rather than
``Path.is_file()`` -- the same presence-vs-usable gap ``doctor.py``'s
suspect-file check and the audit's checklist call out by name elsewhere. A
stray directory sharing one of these names read as present and then failed
at ``open()``/``.stat()`` with an unrelated error instead of the intended
"not there yet" refusal.
"""

from __future__ import annotations

import pytest

from realmspinner.service import jobs as svc_jobs
from realmspinner.service import judge as svc_judge
from realmspinner.service import system as svc_system
from realmspinner.service import verdicts as svc_verdicts
from realmspinner.service.errors import Invalid, NotFound


def test_a_directory_named_like_the_trellis_log_is_treated_as_absent(svc):
    """``trellis_log`` gated "has a log yet" on ``Path.exists()`` rather than
    ``Path.is_file()``. A directory that happens to share the name read as
    present and would have failed inside ``path.open("rb")`` with an
    unrelated ``OSError`` (``IsADirectoryError`` on the platforms that raise
    a distinct one) instead of this refusal.
    """
    (svc.config.data_dir / "trellis.log").mkdir(parents=True, exist_ok=True)

    with pytest.raises(NotFound):
        svc_system.trellis_log(svc)


def test_a_directory_named_like_the_judged_image_is_treated_as_absent(svc):
    """The same gap, on the two sites that answer "is there a picture to
    judge here": ``verdicts.record_verdict``'s own presence check and
    ``judge._image_for``, which the quality-probe trainer and scorer both
    call. A directory named ``reference.png`` or ``input.png`` -- left behind
    by some other write, or a user's own stray folder -- used to read as a
    servable image and fail later at a read instead of "no image here".
    """
    job_id = svc_jobs.create_job(svc, kind="text", prompt="x", output="reference")["id"]
    job_dir = svc.job_dir(job_id)
    for name in svc_verdicts.IMAGE_NAMES:
        (job_dir / name).mkdir(parents=True, exist_ok=True)

    assert svc_judge._image_for(svc, job_id) is None
    with pytest.raises(Invalid):
        svc_verdicts.record_verdict(svc, job_id, verdict="accept", stage="reference")
