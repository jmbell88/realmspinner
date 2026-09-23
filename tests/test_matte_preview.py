"""The promote-time matte preview, and the alpha it hands to trellis.

Three separate claims, and they are separable on purpose:

* the preview itself is a pure function of a job's ``input.png`` -- a cutout
  composited over a checkerboard, plus the composition gate's own verdict --
  and it is cached under git's racily-clean rule, so a file written inside its
  own mtime tick is never remembered;
* a reference that carries a real (non-opaque) alpha channel promotes as an
  *approved* matte: the flag is recorded and the server is told the preserving
  mode, because ``trellis-server.exe --help`` says ``auto`` keeps a pre-matted
  image's alpha and ``birefnet`` -- today's default -- re-cuts it;
* ``matte`` is an input, so it joins the config vector and stays out of
  ``DERIVED_PARAMS``.
"""

from __future__ import annotations

import io
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from realmspinner import vectors
from realmspinner.service import jobs as svc_jobs
from realmspinner.service import matte as svc_matte
from realmspinner.service.errors import Invalid
from realmspinner.service.validation import DERIVED_PARAMS
from realmspinner.studio import matte_preview


def _png(pixels: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(pixels, "RGBA" if pixels.shape[2] == 4 else "RGB").save(buf, "PNG")
    return buf.getvalue()


def _subject_rgb(size: int = 96) -> np.ndarray:
    """A dark square on a plain light background -- what the corner fill is for."""
    arr = np.full((size, size, 3), 230, dtype=np.uint8)
    arr[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = (20, 30, 40)
    return arr


def _cutout_rgba(size: int = 96) -> np.ndarray:
    """The same subject, already matted: transparent everywhere else."""
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    arr[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = (20, 30, 40, 255)
    return arr


def _reference(svc, pixels: np.ndarray) -> str:
    result = svc_jobs.import_reference(svc, _png(pixels), name="subject")
    return result["id"]


# --- the preview ------------------------------------------------------------


def test_preview_returns_a_checkerboarded_cutout_and_the_gates_verdict(svc):
    job_id = _reference(svc, _subject_rgb())

    preview = svc_matte.preview(svc, job_id)

    assert preview.job_id == job_id
    assert preview.width > 0 and preview.height > 0
    assert len(preview.rgb) == preview.width * preview.height * 3
    # No weights on this host, so the corner fill answered -- which is a
    # measurement, not a failure.
    assert preview.source in ("alpha", "birefnet", "flood")
    assert 0.0 < preview.coverage < 1.0
    # The background is the checkerboard rather than the source's light grey.
    pixels = np.frombuffer(preview.rgb, dtype=np.uint8).reshape(
        preview.height, preview.width, 3
    )
    corner = set(int(v) for v in pixels[0, :16, 0])
    assert corner <= {svc_matte.CHECKER_LIGHT, svc_matte.CHECKER_DARK}


def test_preview_reports_the_reference_gates_warnings(svc):
    # A subject that runs to every edge: the gate refuses it, and the promote
    # preview is where the user should learn that -- before the GPU minutes.
    arr = np.full((96, 96, 3), 20, dtype=np.uint8)
    job_id = _reference(svc, arr)

    preview = svc_matte.preview(svc, job_id)

    assert preview.reasons or preview.warnings


def test_preview_refuses_a_job_with_no_image(svc, store):
    job_id = svc.store.create("text", "a crate", {}, "deadbeefcafe", stage="reference")
    with pytest.raises(Invalid):
        svc_matte.preview(svc, "deadbeefcafe")
    assert job_id is not None


def test_a_stamp_is_only_remembered_once_its_mtime_is_safely_past(svc):
    """git's racily-clean rule, exactly as ``files.attach_files`` applies it.

    A directory's -- or a file's -- mtime on Windows comes off a clock that
    ticks every 15.6 ms, so a write landing after the read but inside the
    stamped mtime's own tick is invisible to the stamp *forever*. A stamp is
    therefore stored only once it is comfortably in the past.
    """
    job_id = _reference(svc, _subject_rgb())
    cache: dict = {}

    fresh = svc_matte.preview(svc, job_id)
    fresh = svc_matte.replace_stamp(fresh, time.time_ns())
    assert svc_matte.remember(cache, fresh) is False
    assert svc_matte.cached(cache, job_id, fresh.stamp) is None

    old = svc_matte.replace_stamp(fresh, time.time_ns() - 10 * svc_matte.MTIME_RACE_NS)
    assert svc_matte.remember(cache, old) is True
    assert svc_matte.cached(cache, job_id, old.stamp) is old
    # A different stamp is a different file: the cache must not answer for it.
    assert svc_matte.cached(cache, job_id, old.stamp + 1) is None


def test_the_stamp_is_the_input_pngs_own_mtime(svc):
    job_id = _reference(svc, _subject_rgb())
    src = svc.job_dir(job_id) / "input.png"

    assert svc_matte.stamp_for(src) == src.stat().st_mtime_ns
    assert svc_matte.stamp_for(src.with_name("nothing.png")) is None


def test_pump_submits_and_reports_when_input_png_is_missing_on_first_check(svc):
    """2026-09-05 audit, finding create-03.

    ``MatteState.failed_stamp`` defaults to ``None``, and ``stamp_for`` also
    returns ``None`` for a missing ``input.png`` -- so a reference whose file
    is absent the first time ``pump`` looks made the guard
    ``failed_stamp == stamp`` true before any submit was ever tried. The
    preview task was never submitted, ``on_task_failed`` never ran, and the
    modal sat on "Cutting the subject out..." with no toast, forever.
    """
    job_id = _reference(svc, _subject_rgb())
    (svc.job_dir(job_id) / "input.png").unlink()

    class Ctx:
        def __init__(self) -> None:
            self.state = SimpleNamespace(matte=None)
            self.svc = svc
            self.submitted: list = []

        def submit(self, key, fn, *args, **kwargs):
            self.submitted.append(key)
            return True

    ctx = Ctx()
    matte_preview.open_for(ctx, job_id, {})

    state = matte_preview.pump(ctx)

    assert ctx.submitted == [matte_preview.key(job_id)]
    assert state.preview is None


def test_matte_preview_cache_stays_bounded_across_many_switches(svc):
    """2026-09-08 audit, finding create-07; narrowed by the 2026-09-23 audit,
    finding create-10.

    ``open_for``/``close`` reset the *open* preview's fields, but nothing
    removed a closed job's entry from ``MatteState.cache`` on its own -- so a
    Create session that previewed the matte on many distinct references grew
    that dict for the life of the process. The bound now lives entirely in
    ``on_task_done``'s LRU (``_MAX_CACHE_ENTRIES``): this test asserts the
    boundedness create-07 was actually about, not the specific single-entry
    eviction ``open_for`` used to do on every switch -- create-10 found that
    eviction was throwing away a preview the user had just watched finish,
    ahead of a return that should have reused it.
    """
    cap = matte_preview._MAX_CACHE_ENTRIES
    jobs = [_reference(svc, _subject_rgb()) for _ in range(cap + 3)]

    class Ctx:
        def __init__(self) -> None:
            self.state = SimpleNamespace(matte=None)
            self.svc = svc

        def submit(self, key, fn, *args, **kwargs):
            return True

    ctx = Ctx()

    def _remembered(job_id: str):
        preview = svc_matte.preview(svc, job_id)
        # Safely in the past, so ``remember`` actually stores it (git's
        # racily-clean rule -- see the module docstring above).
        return svc_matte.replace_stamp(preview, time.time_ns() - 10 * svc_matte.MTIME_RACE_NS)

    state = None
    for job_id in jobs:
        matte_preview.open_for(ctx, job_id, {})
        state = matte_preview.pump(ctx)
        matte_preview.on_task_done(
            ctx, SimpleNamespace(key=matte_preview.key(job_id), result=_remembered(job_id))
        )

    # The unfixed-before-create-07 code left every entry in the cache
    # forever -- this is the failing assertion against that.
    assert len(state.cache) <= cap
    for job_id in jobs[-cap:]:
        assert job_id in state.cache
    assert jobs[0] not in state.cache


def test_returning_to_a_job_you_just_finished_viewing_reuses_its_cached_preview(
    svc, monkeypatch
):
    """2026-09-23 audit, finding create-10.

    ``open_for`` used to pop the *outgoing* job's cache entry on every switch,
    even one that had already landed while it was open -- so the 3-entry LRU
    ``on_task_done`` maintains (``_MAX_CACHE_ENTRIES``) could only ever hold
    previews that finished *after* the user had switched away. Returning to a
    reference whose cutout had already finished re-submitted BiRefNet instead
    of reusing the cached entry.
    """
    job_a = _reference(svc, _subject_rgb())
    job_b = _reference(svc, _subject_rgb())

    class Ctx:
        def __init__(self) -> None:
            self.state = SimpleNamespace(matte=None)
            self.svc = svc
            self.submitted: list = []

        def submit(self, key, fn, *args, **kwargs):
            self.submitted.append(key)
            return True

    ctx = Ctx()

    def _land(job_id: str) -> None:
        preview = svc_matte.preview(svc, job_id)
        # Past the race window without moving the stamp off the file's real
        # mtime -- unlike ``replace_stamp``, which would make a later
        # ``pump`` for the same untouched file miss the cache for an
        # unrelated reason.
        monkeypatch.setattr(
            svc_matte.time, "time_ns", lambda: preview.stamp + svc_matte.MTIME_RACE_NS * 2
        )
        matte_preview.on_task_done(
            ctx, SimpleNamespace(key=matte_preview.key(job_id), result=preview)
        )
        monkeypatch.undo()

    matte_preview.open_for(ctx, job_a, {})
    matte_preview.pump(ctx)
    _land(job_a)  # job_a's cutout finishes while it is still the open preview

    matte_preview.open_for(ctx, job_b, {})  # switch away
    matte_preview.open_for(ctx, job_a, {})  # and back, before job_b's lands

    state = matte_preview.pump(ctx)

    assert ctx.submitted == [matte_preview.key(job_a)]
    assert state.preview is not None
    assert state.preview.job_id == job_a


def test_late_matte_preview_results_landing_after_several_more_switches_do_not_grow_the_cache_without_bound(  # noqa: E501
    svc,
):
    """The 2026-09-08 audit, finding create-03.

    ``open_for``'s own eviction (create-07, the test above) only catches the
    *immediately preceding* job at the moment of a switch. It cannot catch a
    preview that lands after the user has moved on through *several* more
    switches, whose own outgoing-job pops were each a no-op because the job
    that finally lands was never the job being switched away from. Before
    this fix, ``on_task_done`` re-inserted every one of these into
    ``state.cache`` with nothing ever removing it again -- unbounded growth
    for a session that previews many references. It must not simply prune
    everything but the current job either: a result for a job the user has
    left is still supposed to be cached, just not shown (see
    ``tests/service/test_matte_handoff.py::
    test_a_result_for_a_job_the_user_left_is_cached_but_not_shown``) -- so the
    fix is a small bound, not a single slot.
    """
    cap = matte_preview._MAX_CACHE_ENTRIES
    jobs = [_reference(svc, _subject_rgb()) for _ in range(cap + 3)]

    class Ctx:
        def __init__(self) -> None:
            self.state = SimpleNamespace(matte=None)
            self.svc = svc

        def submit(self, key, fn, *args, **kwargs):
            return True

    ctx = Ctx()

    def _remembered(job_id: str):
        preview = svc_matte.preview(svc, job_id)
        return svc_matte.replace_stamp(preview, time.time_ns() - 10 * svc_matte.MTIME_RACE_NS)

    # Open every job in turn and move on before its task ever completes --
    # each switch's own eviction is a no-op, since the outgoing job was never
    # cached yet.
    state = None
    for job_id in jobs:
        matte_preview.open_for(ctx, job_id, {})
        state = matte_preview.pump(ctx)

    # Every job's task finally lands, long after the user moved on from all
    # of them.
    for job_id in jobs:
        matte_preview.on_task_done(
            ctx, SimpleNamespace(key=matte_preview.key(job_id), result=_remembered(job_id))
        )

    # Unbounded before the fix: none of these landings was ever the
    # immediately-preceding job at a switch, so nothing ever evicted them.
    assert len(state.cache) <= cap
    # And it is the most recently landed ones that survive, not an arbitrary
    # subset -- the first job to land is the first one an LRU drops.
    for job_id in jobs[-cap:]:
        assert job_id in state.cache
    assert jobs[0] not in state.cache


# --- approved alpha ---------------------------------------------------------


def test_a_matted_reference_promotes_as_an_approved_matte(svc):
    job_id = _reference(svc, _cutout_rgba())

    result = svc_jobs.promote_to_model(svc, job_id, force=True)
    job = svc.require_job(result["id"])

    assert job["params"]["matte"] == "approved"
    # `auto` is the mode that keeps a pre-matted image's alpha; `birefnet`
    # would re-cut it, and it is what the default would otherwise have been.
    assert job["params"]["bg_removal"] == "auto"


def test_an_opaque_reference_promotes_without_a_matte_claim(svc):
    """The doors nobody approved a cutout at: an upload, a sweep, Inker's send.

    ``promote_to_model`` with no ``prepared`` copies ``input.png`` verbatim and
    claims nothing, exactly as it always did -- which is what keeps every stored
    corpus a statement about the pipeline that produced it. The *modal's* Accept
    is the case that changed, and it goes through ``promote_candidates``.
    """
    job_id = _reference(svc, _subject_rgb())

    result = svc_jobs.promote_to_model(svc, job_id, force=True)
    job = svc.require_job(result["id"])

    assert "matte" not in job["params"]


# --- the approved cutout is the reconstruction ------------------------------


def test_the_pixels_the_user_approved_are_the_pixels_trellis_reconstructs_from(svc):
    """The modal showed a cutout and the engine was handed the original.

    ``preview`` computed the host's BiRefNet cut, drew it, and dropped it;
    ``promote_to_model`` copied the untouched ``input.png``; ``matte.approve``
    found no alpha, so ``bg_removal`` stayed ``birefnet`` and the server re-cut
    the image with a *different copy* of BiRefNet under
    ``REALMSPINNER_TRELLIS_MODELS``. The picture in the modal was a claim about
    pixels nothing downstream ever saw.
    """
    job_id = _reference(svc, _subject_rgb())

    preview = svc_matte.preview(svc, job_id)
    result = svc_jobs.promote_candidates(svc, job_id, force=True)
    job = svc.require_job(result["id"])

    approved = (svc.job_dir(job_id) / svc_matte.CUTOUT).read_bytes()
    assert (svc.job_dir(result["id"]) / "input.png").read_bytes() == approved
    # And the server is told to keep them rather than cut its own.
    assert job["params"]["matte"] == "approved"
    assert job["params"]["bg_removal"] == "auto"
    assert job["params"]["approved_input"]["from"] == job_id
    assert preview.source == job["params"]["approved_input"]["source"]


def test_every_candidate_in_a_group_gets_byte_identical_approved_pixels(svc):
    """A candidate group is a controlled comparison or it is nothing.

    Cutting once and sharing the file makes "the same reference" an identity
    rather than a premise about BiRefNet being deterministic.
    """
    job_id = _reference(svc, _subject_rgb())

    result = svc_jobs.promote_candidates(svc, job_id, count=3, force=True)

    approved = (svc.job_dir(job_id) / svc_matte.CUTOUT).read_bytes()
    assert len(result["ids"]) == 3
    for made in result["ids"]:
        assert (svc.job_dir(made) / "input.png").read_bytes() == approved


def test_a_soft_painted_matte_is_not_hardened_by_approval(svc):
    """``_cut`` rebuilt alpha from ``matting.mask``, which thresholds at 8.

    So the one matte this module exists to protect -- one a person painted --
    was flattened to a hard edge by the very functions that carry it. It was
    already live in the Inker hand-off (``alpha_plane`` reopened the reference
    with a hardened rim), and it would have become destructive the moment the
    cutout started being the pixels ``input.png`` is written *from*.
    """
    arr = _cutout_rgba()
    # A feathered rim: two rows of half-transparent pixels across the subject.
    arr[24:26, 24:72] = (20, 30, 40, 128)
    job_id = _reference(svc, arr)

    plane, source = svc_matte.alpha_plane(svc, job_id)

    assert source == "alpha"
    assert int(plane[24, 40]) == 128

    # And it survives all the way onto the job that reconstructs from it.
    result = svc_jobs.promote_candidates(svc, job_id, force=True)
    with Image.open(svc.job_dir(result["id"]) / "input.png") as im:
        assert int(np.asarray(im.convert("RGBA"))[24, 40, 3]) == 128


def test_an_approved_promotion_pins_the_preserving_bg_removal_mode(svc):
    """Even against an explicit override, and that is the documented rule.

    ``birefnet`` would re-cut the cutout, which would make the approval a lie.
    """
    job_id = _reference(svc, _subject_rgb())

    result = svc_jobs.promote_candidates(svc, job_id, bg_removal="birefnet", force=True)

    assert svc.require_job(result["id"])["params"]["bg_removal"] == "auto"


def test_a_reference_edited_after_approval_is_re_prepared_not_reused(svc):
    """The Fix-matte round trip, and the revert.

    ``prepared`` is pinned to ``input.png``'s fingerprint, so a save landing
    behind the modal expires the cutout on its own -- there is no invalidation
    rule beside each writer for one of them to forget.
    """
    job_id = _reference(svc, _subject_rgb())
    first = svc_matte.prepare(svc, job_id)
    assert svc_matte.prepared(svc, job_id) is not None

    (svc.job_dir(job_id) / "input.png").write_bytes(_png(_cutout_rgba()))

    assert svc_matte.prepared(svc, job_id) is None
    again = svc_matte.ensure_prepared(svc, job_id)
    assert again.src_fingerprint != first.src_fingerprint

    # And a stale one is refused rather than reconstructed from.
    with pytest.raises(Invalid):
        svc_jobs.promote_to_model(svc, job_id, prepared=first, force=True)


def test_a_cutout_recorded_but_missing_from_disk_reads_as_absent(svc):
    """The completion gate, from the other side: params without the file."""
    job_id = _reference(svc, _subject_rgb())
    svc_matte.prepare(svc, job_id)

    (svc.job_dir(job_id) / svc_matte.CUTOUT).unlink()

    assert svc_matte.prepared(svc, job_id) is None


def test_the_cutout_record_does_not_travel_onto_a_promoted_job(svc):
    """It is about *this* row's pixels, so it is derived. Were it inherited the
    record would claim the cutout is its own source."""
    job_id = _reference(svc, _subject_rgb())

    result = svc_jobs.promote_candidates(svc, job_id, force=True)

    assert "cutout" in DERIVED_PARAMS
    assert svc_matte.CUTOUT_PARAM not in svc.require_job(result["id"])["params"]


def test_a_fully_opaque_alpha_channel_is_not_a_matte(svc):
    """Having the channel is not using it -- the ``matting._OPAQUE`` rule."""
    arr = np.dstack([_subject_rgb(), np.full((96, 96), 255, np.uint8)])
    job_id = _reference(svc, arr)

    result = svc_jobs.promote_to_model(svc, job_id, force=True)

    assert "matte" not in svc.require_job(result["id"])["params"]


def test_an_uploaded_cutout_records_the_same_approval(svc):
    result = svc_jobs.create_job(svc, kind="image", image=_png(_cutout_rgba()))
    job = svc.require_job(result["id"])

    assert job["params"]["matte"] == "approved"
    assert job["params"]["bg_removal"] == "auto"


# --- concurrent prepare -------------------------------------------------------


def test_concurrent_matte_prepare_calls_for_one_job_do_not_share_a_temp_name(svc):
    """service-02 (2026-09-11 audit): ``prepare`` staged ``cutout.png``
    through a FIXED temp name (``dest.with_name(f".{dest.name}.tmp")``) with
    no lock around it, unlike every sibling staged derivation in this segment.
    ``ensure_prepared`` has three independent doors -- the preview modal, the
    Inker hand-off, and promote/rerun -- so two of them can call ``prepare``
    for the same job at once. Reproduced directly against the real service: two
    threads racing ``prepare`` both raised a Windows ``PermissionError``
    (WinError 5/32) on ``os.replace``, and ``cutout.png`` was left MISSING
    after both finished.
    """
    job_id = _reference(svc, _subject_rgb())

    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def worker() -> None:
        try:
            barrier.wait(timeout=5)
            svc_matte.prepare(svc, job_id)
        except BaseException as exc:  # noqa: BLE001 -- the race itself is the assertion
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"prepare() raised under a race: {errors!r}"
    assert (svc.job_dir(job_id) / svc_matte.CUTOUT).exists()


def test_matte_prepare_takes_the_convert_lock_for_the_cutout_it_stages(svc, monkeypatch):
    """service-02's deterministic sibling. The racy test above only fails
    about two runs in three -- a data race is inherently probabilistic -- which
    is not a strong enough guard against someone later dropping the
    ``convert_lock`` this fix added, in a suite CLAUDE.md already flags as
    flaky under parallel load: a probabilistic guard is hard to tell apart
    from that noise. This test instead pins the *mechanism* rather than the
    symptom: it spies on ``svc.convert_lock`` and on ``os.replace`` as
    ``matte.py`` sees it, and asserts the lock keyed ``(job_id, CUTOUT)`` is
    both taken and still held at the moment the staged write lands. Remove
    the lock and this fails every time, immediately.
    """
    job_id = _reference(svc, _subject_rgb())

    real_convert_lock = svc.convert_lock
    calls: list[tuple[str, str]] = []

    def spy_convert_lock(job_id_arg: str, name_arg: str):
        calls.append((job_id_arg, name_arg))
        return real_convert_lock(job_id_arg, name_arg)

    monkeypatch.setattr(svc, "convert_lock", spy_convert_lock)

    real_replace = svc_matte.os.replace
    held_during_replace: list[bool] = []

    def spy_replace(src, dst):
        # The lock is fetched (not acquired) here purely to read .locked() --
        # prepare() itself already holds it if the fix is in place.
        held_during_replace.append(real_convert_lock(job_id, svc_matte.CUTOUT).locked())
        return real_replace(src, dst)

    monkeypatch.setattr(svc_matte.os, "replace", spy_replace)

    svc_matte.prepare(svc, job_id)

    assert (job_id, svc_matte.CUTOUT) in calls, (
        "prepare() never took svc.convert_lock(job_id, CUTOUT)"
    )
    assert held_during_replace == [True], (
        f"the lock was not held while cutout.png was being staged: {held_during_replace!r}"
    )


# --- build anyway, at the door ----------------------------------------------


def test_build_anyway_records_an_override_against_the_pixels_it_wrote(svc):
    """``force`` used to be consumed here and discarded.

    It skipped this function's own soft gate and never travelled, so the worker
    re-measured and raised the same sentences two minutes of queue later. The
    grant is pinned to the bytes actually written -- which with an approved
    cutout are not the reference's -- so the worker can fingerprint its own
    ``input.png`` and get a match.
    """
    from realmspinner import provenance
    from realmspinner.pipelines import reference

    job_id = _reference(svc, _subject_rgb())
    # A refusal the reference stage recorded: what the modal shows and what the
    # Build-anyway button is offered against.
    svc.store.merge_params(
        job_id,
        {"reference_report": {"ok": False, "reasons": ["two objects"],
                              "codes": ["multi_object"]}},
    )

    result = svc_jobs.promote_candidates(svc, job_id, force=True)
    params = svc.require_job(result["id"])["params"]

    written = provenance.file_fingerprint(svc.job_dir(result["id"]) / "input.png")
    assert reference.override_allows(params, written)
    assert params[reference.OVERRIDE_KEY]["codes"] == ["multi_object"]


def test_a_promotion_that_was_not_forced_grants_no_override(svc):
    job_id = _reference(svc, _subject_rgb())

    result = svc_jobs.promote_candidates(svc, job_id)

    from realmspinner.pipelines import reference

    assert reference.OVERRIDE_KEY not in svc.require_job(result["id"])["params"]


def test_an_unforced_promotion_of_a_refused_reference_is_still_refused(svc):
    job_id = _reference(svc, _subject_rgb())
    svc.store.merge_params(
        job_id,
        {"reference_report": {"ok": False, "reasons": ["two objects"],
                              "codes": ["multi_object"]}},
    )

    with pytest.raises(Invalid):
        svc_jobs.promote_candidates(svc, job_id)


# --- the vocabulary ---------------------------------------------------------


def test_matte_is_a_config_vector_param_and_not_a_derived_one():
    assert "matte" in vectors.VECTOR_PARAMS
    assert "matte" not in DERIVED_PARAMS
    assert vectors.config_vector({"params": {"matte": "approved"}})["matte"] == "approved"
