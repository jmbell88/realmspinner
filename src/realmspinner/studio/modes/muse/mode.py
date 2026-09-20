"""Muse's controller: the brief, the takes, playback and the bridge to Sirens.

An ordinary ``studio/`` module, and that is worth saying because the *other*
audio mode is not: ``studio/sirens/`` is a headless engine forbidden to import
imgui, moderngl, pygame, scipy or ``service``, and ``sirens_mode`` is the thin
layer that reaches those on its behalf. Muse has no engine to keep pure -- the
model lives in a subprocess two layers down -- so this module imports
``service``, ``sirens_audio`` and ``sirens_io`` freely.

**Playback is ``sirens_audio``, and what it gained is a volume, not an offset.**
``play(pcm, rate, tag=..., loops=)`` is already mode-agnostic and tag-keyed, and
it refuses anything but its ``RATE`` (44100) outright rather than resampling.
ACE-Step's vendored ``latents2audio`` *defaulted* to 48000, which is why
``read_track`` reconciles the two -- but ``REALMSPINNER 5/6`` now pins the writer to
44 100 Hz 16-bit at the call site, so that path is the fallback for a take made
by an older build rather than the normal one. See
``pipelines/acestep/ATTRIBUTION.md``.

The volume lives there because that module owns the device and its exclusivity
is the contract: one reserved channel means one ``set_volume``, so a per-mode
volume would be a control that disagrees with itself. No offset, for the
mirror-image reason -- it does not own the caller's buffer, and seeking is
slice-and-replay, so the base lives on ``MuseState.player.play_offset``.

The module is deliberately *not* renamed or relocated for having a second
caller: its file location is cosmetic and its ``pygame.mixer`` exclusivity is
the actual contract, so moving it would churn every import in Sirens to say
something the docstring already says.

**The bridge runs both ways, and only one leg opens a door.**
:func:`open_in_sirens` composes three functions that exist:
``sirens_io.import_sample`` (which submits the same ``_decode_sample`` task a
user's drag-and-drop does), ``sirens_mode.new_document``, and ``state.set_mode``
-- the one mode-switch implementation, never an assignment to the state field.
Sirens needs no change of any kind for it.

:func:`compose_from_sirens` is the mirror, and the Sirens half of it opens
nothing either -- ``rsng_bytes``, ``read_rsng``/``synth.render_marked`` and
``wavout.wav_bytes`` are already used in exactly that combination by
``sirens_play.request_render`` and ``sirens_io.export_plan``. What it does need
is one keyword on the *Muse* side, because ``create_music_job`` takes scalars
only: ``reference_wav: bytes``. Named here rather than left implicit, because
"the bridge opens no new doors" was a load-bearing claim and half of it has
stopped being true.
"""

from __future__ import annotations

import logging
from typing import Any

from ...state import set_mode
from ..sirens import audio as sirens_audio
from ..sirens import fileio as sirens_io
from ..sirens import mode as sirens_mode
from ..sirens import state as sirens_state
from . import fileio as muse_io
from .state import (  # noqa: F401
    DEFAULT_DERIVE,
    DEFAULT_FORM,
    LOOP_MEMORY,
    MuseState,
    active,
    ensure,
)
from .state import Player as MusePlayer

log = logging.getLogger(__name__)

#: The task key a take's decode runs under. Prefixed, because the app claims
#: results by prefix and a key without one is a result delivered nowhere.
LOAD_PREFIX = "muse-load:"


def generate(ctx: Any) -> bool:
    """Queue the brief. -> whether the submit was accepted.

    Through ``ctx.submit`` under the shared ``"submit"`` key, exactly as
    Create's Generate is: the key is what stops a second Ctrl+Enter queueing a
    duplicate while the first is still at the door, and it is shared because
    from the user's side there is one "am I submitting" at a time.

    Validation is the service's. There is no ``validate(form)`` sibling here on
    purpose: Create has one because its form has fifteen interacting fields and
    a disabled button needs a reason before anything is pressed, whereas every
    refusal Muse can produce comes from ``create_music_job`` and carries the
    ``field=`` that puts the ring on the right control. A second copy of those
    bounds in the pane is the thing that drifts.

    **muse-02 (2026-09-14 audit).** ``muse_brief._generate``'s draw call was
    the *only* place that checked ``model_gate.missing`` -- so it greyed the
    button correctly, but Ctrl+Enter (``handle_key``, below) and Sirens'
    "Compose in Muse" (``compose_from_sirens``) both call into a submission
    path that never asked, and landed at the service door's own refusal
    instead: a ``music_model`` toast with no field to ring, rather than the
    mode's own gate, which points at the Recipe panel /
    Settings -> Models. Checked here, once, so every entry into a music job
    -- the button, the keyboard, and the bridge -- inherits it rather than
    each needing its own copy.
    """
    from ....service import jobs as svc_jobs
    from ...panes import model_gate

    if model_gate.missing(ctx, svc_jobs.MUSIC_ROWS):
        ctx.toast("The music model is not downloaded. See the Recipe panel.", "warn")
        return False

    state = ensure(ctx)
    form = state.form

    def run():
        return svc_jobs.create_music_job(
            ctx.svc,
            prompt=str(form["prompt"]),
            lyrics=str(form["lyrics"]),
            duration=float(form["duration"]),
            count=int(form["count"]),
            seed=form["seed"],
            infer_step=int(form["infer_step"]),
            guidance_scale=float(form["guidance_scale"]),
            scheduler_type=str(form["scheduler_type"]),
            cfg_type=str(form["cfg_type"]),
            omega_scale=float(form["omega_scale"]),
        )

    # Cleared on every press: the rings from the last one describe a request
    # that no longer exists.
    ctx.state.clear_field_errors()
    if not ctx.submit("submit", run):
        ctx.toast("Still submitting the last one - try again in a moment.")
        return False
    ctx.state.remember_prompt(str(form["prompt"]))
    return True


# --- deriving ---------------------------------------------------------------


#: How far Left/Right move the playhead, in seconds, and what Shift multiplies
#: that by. One second is a beat or two at most tempos -- fine enough to place a
#: marker by ear -- and ten seconds is a phrase, which is the other scale a
#: listener works at. Two figures rather than a ladder because there are two
#: questions and no third.
NUDGE_SECONDS = 1.0
NUDGE_MULTIPLIER = 10


#: Which controls the popup draws for each task, and the only place that is
#: written down. Read as data by the pane so that adding a task is a row here
#: rather than a seventh ``elif`` in a draw function -- ``_jobs_music.TASKS``
#: is the door's half of the same list, and ``open_derive`` refuses anything
#: this table has no entry for.
DERIVE_CONTROLS: dict[str, tuple[str, ...]] = {
    "retake": ("retake_variance",),
    "extend": ("extend_left", "extend_right"),
    "repaint": ("repaint_start", "repaint_end"),
    "edit": ("edit_prompt", "edit_lyrics"),
    "loop": ("repaint_end",),
    "audio2audio": ("ref_audio_strength",),
}

#: Tasks the door refuses a count above 1 for -- read by ``muse_results`` to
#: stop the popup's "How many" slider offering a number the door would only
#: send back as a refusal.
#:
#: **service-02 (2026-09-15 audit).** Every other task varies each row through
#: ``retake_random_generators`` (see ``_q_music._task_kwargs``'s comment), but
#: audio2audio's upstream ``task`` becomes ``"audio2audio"`` inside
#: ``__call__`` itself, which never touches that generator -- so with ``seed``
#: inherited unchanged from the parent, nothing distinguishes one row of a
#: count > 1 audio2audio derive from another. A set, not a bare string check
#: at the one call site, so a second task that turns out to have no varying
#: knob is a row here rather than a second ``if``.
SINGLE_TAKE_TASKS: frozenset[str] = frozenset({"audio2audio"})


def open_derive(ctx: Any, job_id: str, task: str) -> None:
    """Point the derive popup at one take. **Frame-thread only.**

    The form is rebuilt from :data:`DEFAULT_DERIVE` on every open rather than
    carried, for the reason it is a separate dict at all: a window left over
    from the last take is a request about a piece of music the user is no
    longer looking at.
    """
    state = ensure(ctx)
    state.derive_job = job_id
    state.derive_form = dict(DEFAULT_DERIVE)
    state.derive_form["task"] = task if task in DERIVE_CONTROLS else "retake"


def close_derive(ctx: Any) -> None:
    ensure(ctx).derive_job = ""


def derive_popup_open(ctx: Any) -> bool:
    """Whether the derive popup owns the keyboard. Tolerant of a partial
    ``ctx`` (``getattr``, not attribute access), ``matte_preview.is_open``'s
    reason: ``dialogs.modal_open`` asks this on every key press (I77), for a
    caller that has never built a Muse state.

    The 2026-09-16 audit found the derive popup missing from ``modal_open``'s
    five answers -- a real ``imgui.begin_popup_modal``, but Ctrl+Enter still
    reached ``handle_key`` and queued a fresh job from the top brief while the
    popup believed it alone had the keyboard, and Space/arrows/``[``/``]``/``L``
    kept auditioning and reseeking the take behind it.
    """
    state = getattr(getattr(ctx, "state", None), "muse", None)
    return bool(state is not None and state.derive_job)


def derive(ctx: Any) -> bool:
    """Queue what the popup asks for. -> whether the submit was accepted.

    ``generate``'s shape and its reasoning verbatim, on the other door: the
    shared ``"submit"`` key, no ``validate`` sibling, and every refusal coming
    back from ``derive_music_job`` with the ``field=`` that rings the control.

    An ``edit`` sends ``None`` for a field the user left alone rather than the
    empty string, because those mean different things at that door: ``None`` is
    "keep the parent's" and ``""`` is "drop the words entirely".
    """
    from ....service import jobs as svc_jobs

    state = ensure(ctx)
    form = dict(state.derive_form)
    job_id = state.derive_job
    task = str(form["task"])
    if not job_id:
        return False

    kwargs: dict[str, Any] = {"task": task, "count": int(form["count"])}
    for name in DERIVE_CONTROLS[task]:
        value = form[name]
        if name in ("edit_prompt", "edit_lyrics"):
            kwargs[name] = str(value) if str(value).strip() else None
        else:
            kwargs[name] = float(value)
    if task == "loop":
        # The popup asks for one figure -- how much of the joint to rewrite --
        # and the door reads a window. Zero to that span *is* that figure; the
        # door then centres it on the roll. See ``derive_music_job``.
        kwargs["repaint_start"] = 0.0

    ctx.state.clear_field_errors()
    if not ctx.submit("submit", lambda: svc_jobs.derive_music_job(ctx.svc, job_id, **kwargs)):
        ctx.toast("Still submitting the last one - try again in a moment.")
        return False
    state.derive_job = ""
    return True


# --- auditioning ------------------------------------------------------------


def track_path(ctx: Any, job_id: str):
    """Where a finished take's WAV is. One spelling, three callers."""
    return ctx.svc.config.job_dir(job_id) / "track.wav"


#: ``muse_io.read_track``, under the name this module used to define.
#:
#: The function moved to ``muse_io`` with everything else that touches a take's
#: file; the alias stays because ``tests/modes/muse/test_muse_mode.py`` patches it by this
#: name, and a rename that breaks a test's patch point is a rename that hides
#: what it changed. There is one implementation.
_read_track = muse_io.read_track


def play(ctx: Any, job_id: str) -> None:
    """Audition one take, replacing whatever was playing.

    The read is on a task rather than the frame thread: four minutes of 44.1 kHz
    stereo is ~40 MB off disk, which is a visible stall in a 60 Hz loop.

    **Resume, when the take is already the one loaded.** M07: this used to
    resubmit the decode unconditionally, and ``on_task_done`` builds a brand
    new :class:`MusePlayer` for every successful load -- so pressing Play on a
    take already sitting in memory (Stop, then Play again, being the ordinary
    case) silently reset its loop region, its crossfade and its playhead.
    There is nothing to re-read here: the samples are already decoded, so this
    just puts them back on the channel from wherever they were left.
    """
    state = ensure(ctx)
    one = state.player
    if one is not None and one.job == job_id and one.pcm is not None:
        state.audition_job = job_id
        _play_from(ctx, one, one.play_offset)
        return
    path = track_path(ctx, job_id)
    if not path.exists():
        ctx.toast("that take has no audio on disk", "warn")
        return
    # The user's latest request. ``on_task_done`` checks this before adopting
    # a completed decode, which is what stops a slower decode for a take the
    # user has since moved on from landing on top of a newer one (M11).
    state.audition_job = job_id
    if not ctx.submit(f"{LOAD_PREFIX}{job_id}", _read_track, path):
        ctx.toast("still loading that take")


def on_task_done(ctx: Any, done: Any) -> None:
    """Adopt a decoded take, a set of loop points, or a precomputed loop
    cache. Routed by key prefix."""
    key, result = done.key, done.result
    if key.startswith(muse_io.CACHE_PREFIX):
        # **incident-2026-09-05b.** The one place the loop-cache pair is
        # written now, on the frame thread, which is what makes the two
        # statements below safe: nothing else ever touches them. Adopted
        # only if the key it was computed for is still the one wanted --
        # a later region change, or a different take loaded in the
        # meantime, must not have a stale-but-internally-consistent answer
        # land on top of whatever is already current.
        #
        # **muse-01 (2026-09-14 audit).** This used to check only the region
        # tuple, not the job id the key already carries (``precompute_loop``
        # submits ``f"{CACHE_PREFIX}{one.job}"``) -- unlike the LOAD_PREFIX
        # and FIND_PREFIX branches right above and below, which both check
        # the job id. ``loop_memory`` persists a region per job, so two takes
        # trimmed to the same ``(start, end, fade)`` in samples is not a rare
        # coincidence: a cache still being computed for the take the user
        # left would land on the take they switched to, and Play on B would
        # sound A.
        one = player(ctx)
        if (
            one is not None
            and result is not None
            and key[len(muse_io.CACHE_PREFIX) :] == one.job
        ):
            cache_key, buffer = result
            # **muse-02 (2026-09-18 audit).** Consumed here whether or not the
            # cache below actually installs: a ``pending_play`` waiting on a
            # region that changed since it was recorded is exactly as dead as
            # one waiting on a task for an abandoned take, and both want to
            # drop silently rather than linger for some later, unrelated
            # landing to misfire on.
            pending = getattr(one, "pending_play", None)
            one.pending_play = None
            if muse_io.loop_cache_key(one) == cache_key:
                one.loop_cache = buffer
                one.loop_cache_key = cache_key
                if (
                    pending is not None
                    and pending[0] == one.job
                    and pending[2] == cache_key
                    and ctx.state.mode == "muse"
                ):
                    # The cache this Play was waiting for just landed, still
                    # naming this take and this exact region: start it now,
                    # from the position it was asked for. ``_play_from`` finds
                    # ``loop_cache_key`` already current, so this is the O(1)
                    # cache-hit path, not the blend -- never a second wait.
                    _play_from(ctx, one, pending[1])
        return
    if key.startswith(muse_io.FIND_PREFIX):
        one = player(ctx)
        # **muse-03 (2026-09-11 audit).** ``one.finding`` guards this the way
        # the LOAD_PREFIX branch below guards on ``audition_job``: a search
        # abandoned by switching to another take and back rebuilds a fresh
        # ``Player`` (``finding`` defaults to ``False``) that happens to carry
        # the same job id, so matching on the id alone let a search nobody is
        # waiting on any more land on top of a region the user has since
        # restored or hand-set with no request of theirs behind it.
        if one is not None and one.finding and one.job == key[len(muse_io.FIND_PREFIX) :]:
            one.finding = False
            one.candidates = list(result or [])
            if one.candidates:
                # The best one adopted immediately: the finder's whole output
                # is a ranking, and making the user press a second time to hear
                # the answer it already has is a step with no decision in it.
                choose_candidate(ctx, 0)
            else:
                ctx.toast("No loop points stood out in this take.", "warn")
        return
    if key.startswith(muse_io.EXPORT_PREFIX):
        # **2026-09-16 audit.** This branch did not exist: every ``muse-export:``
        # result -- success or failure alike -- fell through to the
        # ``LOAD_PREFIX`` guard below and was discarded with no toast, so
        # "Export the loop" and "Export the track with loop points" gave the
        # user identical silence whether the write landed or not. Mirrors
        # ``sirens_mode``'s own ``EXPORT_PREFIX`` toast -- ``muse_io``'s
        # docstring calls that module "the same seam".
        #
        # ``muse_io._save``'s task returns three different things down one
        # ``str | None`` channel: the written path, ``None`` for a picker the
        # user cancelled (wants no word), and ``""`` for a write that did not
        # happen for any other reason (the region changing between muse-03's
        # upfront check and this task running, chiefly) -- which does want one.
        if result:
            ctx.toast(f"Exported to {result}")
        elif result == "":
            ctx.toast("The export did not write anything -- try again.", "warn")
        return
    if not key.startswith(LOAD_PREFIX) or not isinstance(result, dict):
        return
    job_id = key[len(LOAD_PREFIX) :]
    state = ensure(ctx)
    if job_id != state.audition_job:
        # Stale (M11): a later ``play()`` moved the user on to a different
        # take, or ``stop()`` withdrew the request outright, before this
        # decode finished. Adopting every successful load unconditionally let
        # an older, slower decode override a newer take already sounding, and
        # let a take the user had stopped waiting for start playing anyway the
        # moment its read finally landed.
        return
    # The take being replaced keeps its markers (W4), so switching back to it
    # does not mean finding its loop points again.
    remember_loop(ctx)
    # The playhead travels with the switch too, the same reason W4 gave the
    # loop markers back: auditioning a second take mid-listen and switching
    # to another used to snap the new one back to 0:00 even when the point
    # being compared was thirty seconds in. Read before the player is
    # replaced, since afterwards there is nothing left to read it from.
    previous = state.player
    # **One take at a time.** ~42 MB for four minutes, so replaced rather than
    # cached per job -- see ``MuseState.player``.
    state.player = MusePlayer(
        job=job_id,
        pcm=result["pcm"],
        rate=int(result["rate"]),
        env=result.get("env"),
        duration=float(result.get("duration", 0.0)),
    )
    remembered = state.loop_memory.get(job_id)
    if remembered is not None:
        start, end, fade = remembered
        state.player.loop_start = start
        state.player.loop_end = end
        state.player.xfade_ms = float(fade)
    if previous is not None:
        # Clamped, not carried outright: a shorter take cannot hold a
        # position the longer one reached.
        state.player.play_offset = min(previous.play_offset, state.player.duration)
    # Tagged with the job id, which is what lets a card ask "am *I* the one
    # playing" rather than only "is anything playing".
    if sirens_audio.play(result["pcm"], result["rate"], tag=job_id):
        state.playing_job = job_id
    else:
        ctx.toast(sirens_audio.unavailable_reason() or "could not play that take",
                  "warn")


def on_task_failed(ctx: Any, done: Any) -> None:
    """Clear whatever a failed Muse task would otherwise leave stuck.

    muse-03 (2026-09-07 audit): ``main.py``'s task-failure dispatcher had no
    ``muse-`` branch at all -- every other mode's own tasks routed to an
    ``on_task_failed`` here, and Muse's simply fell through to the generic
    toast. A failed loop search left ``finding`` set by ``find_loops`` (the
    only place that turns it on) with nothing left to turn it back off, so
    the strip's spinner ran forever instead of just this one search.
    """
    key = done.key
    if key.startswith(muse_io.FIND_PREFIX):
        one = player(ctx)
        if one is not None and one.job == key[len(muse_io.FIND_PREFIX) :]:
            one.finding = False
        return


def stop(ctx: Any) -> None:
    """Stop whatever is auditioning. Safe when nothing is.

    **Captures the playhead before the device forgets it (M07).** Without
    this, the next Play resumed from wherever the *last* ``_play_from`` call
    had started -- not from where the user actually stopped listening, which
    for anything but a fresh seek is a different number. ``position(ctx)`` has
    to run while the mixer still thinks it is playing, which is why this reads
    it before ``sirens_audio.stop()`` rather than after.

    **Withdraws the audition request (M11).** A decode already in flight for
    the take that was playing (or one requested and not yet decoded) has
    nothing left to land on: clearing ``audition_job`` is what ``on_task_done``
    checks to refuse it, rather than starting playback back up the moment the
    read finishes.

    **Withdraws a deferred Play too (muse-02, 2026-09-18 audit).** A Play
    pressed on a stale loop cache leaves ``one.pending_play`` set, waiting for
    ``on_task_done`` to start it once the blend lands -- Stop is the user
    taking that request back, so it must not go on to sound after all, on a
    press that never claimed to be "playing" in the first place
    (``is_playing`` below is about the *mixer*, not about an outstanding
    request).
    """
    state = active(ctx)
    if state is not None:
        one = state.player
        if one is not None:
            if is_playing(ctx, one.job):
                one.play_offset = position(ctx)
            one.pending_play = None
        state.playing_job = ""
        state.audition_job = ""
    sirens_audio.stop()


def is_playing(ctx: Any, job_id: str) -> bool:
    """Whether *this* take is the one currently sounding.

    Asked of the mixer's tag rather than of ``playing_job`` alone, so a take
    that finished on its own stops drawing as Stop without anything having to
    notice the end.
    """
    return bool(job_id) and sirens_audio.playing() and sirens_audio.tag() == job_id


def sync(ctx: Any) -> None:
    """Let a finished audition clear itself. Called once per frame by the tray."""
    state = active(ctx)
    if state is None:
        return
    if state.playing_job and not sirens_audio.playing():
        state.playing_job = ""
    if state.player is not None and state.player.job:
        # Drop the ~42 MB when its take leaves the Library. Off the cached rows
        # rather than a stat, for the tray's reason: this runs every frame.
        #
        # **muse-06 (2026-09-11 audit).** This used to build a ``set`` of
        # every job id in the cache on every one of those frames just to test
        # whether one id -- the loaded player's -- was in it, so the cost
        # scaled with the whole library's size, not Muse's rows alone, paid
        # 60 times a second for as long as the tray was drawn. A
        # short-circuiting membership scan answers the same question and
        # stops at the first match rather than visiting every row.
        jobs = getattr(ctx.cache, "jobs", []) or []
        if jobs and not any(str(job["id"]) == state.player.job for job in jobs):
            state.player = None


def separate(ctx: Any, job_id: str) -> bool:
    """Queue a split of one take into stems. -> whether the submit landed.

    A one-line controller over ``separate_job``, and it is here rather than
    inline in the pane for ``generate``'s reason: every refusal comes back from
    the door carrying the ``field=`` that rings a control, and a second copy of
    those rules in a pane is the thing that drifts.

    The refusal that matters is the missing model, and it arrives with
    ``rows=`` so the toast reaches the Download button rather than being a
    sentence about a file. Muse itself is unaffected: a take with no stems is a
    take, which is why ``check_weights`` refuses this job and never a
    generation.
    """
    from ....service import jobs as svc_jobs

    ctx.state.clear_field_errors()
    if not ctx.submit(f"muse-separate:{job_id}", svc_jobs.separate_job, ctx.svc, job_id):
        ctx.toast("Already splitting that take.")
        return False
    return True


def has_stems(ctx: Any, job: Any) -> bool:
    """Whether this take already has its stems. Off the cached row.

    ``files.LISTED`` carries them, so the answer is in ``job["files"]`` and no
    pane has to stat anything per frame -- the tray's rule for every other
    per-card question.
    """
    files = job.get("files") or []
    return any(name.startswith("stems/") for name in files)


# --- the player -------------------------------------------------------------


def player(ctx: Any):
    """The decoded take under the strip, or ``None``. Never builds one."""
    state = active(ctx)
    return None if state is None else state.player


def position(ctx: Any) -> float:
    """Where the playhead is, in seconds into the *take*.

    ``sirens_audio.position`` answers where it is in the *buffer*, and seeking
    is slice-and-replay -- so the offset the slice began at is what makes the
    two the same number. That offset lives on :class:`Player`, not in
    ``sirens_audio``: the mixer does not own the caller's buffer, and a second
    module tracking the same figure is how the two come to disagree.

    **Inside a looping region, the buffer is rotated (M10)**: it starts at
    ``loop_anchor`` rather than at ``loop_start``, so the mixer's own
    modulo-buffer-length wrap has to be unwound against the *region's* length
    before it is a position in the take -- adding the raw figure straight to
    ``play_offset``, the way this used to, is only correct for a buffer that
    starts where it plays from, which a rotated loop does not.
    """
    one = player(ctx)
    if one is None or not is_playing(ctx, one.job):
        return one.play_offset if one is not None else 0.0
    raw = sirens_audio.position()
    if (
        one.loop_anchor is not None
        and one.loop_start is not None
        and one.loop_end is not None
        and one.loop_end > one.loop_start
    ):
        length = one.loop_end - one.loop_start
        phase = one.loop_anchor - one.loop_start
        return one.loop_start + (phase + raw) % length
    return min(one.play_offset + raw, one.duration)


def seek(ctx: Any, seconds: float) -> None:
    """Move the playhead, restarting from there if a take is sounding.

    Slice-and-replay, exactly ``sirens_play.play_from_caret``: there is no
    device-side seek in the mixer this app uses, so the buffer handed to it
    *is* the remainder. Cheap because the samples are already in memory -- the
    slice is a view and ``make_sound`` copies once.

    The caller is expected to fire this on release rather than per mouse-move:
    a ``make_sound`` every frame of a drag is a ~40 MB copy per frame.
    """
    one = player(ctx)
    if one is None or one.pcm is None:
        return
    seconds = min(max(float(seconds), 0.0), one.duration)
    one.play_offset = seconds
    if not sirens_audio.playing() or sirens_audio.tag() != one.job:
        # Not sounding: move the playhead and leave it there. A seek is not a
        # play, and starting one because the user clicked the waveform to look
        # at something would be the pane deciding to make a noise.
        return
    _play_from(ctx, one, seconds)


def _play_from(ctx: Any, one: Any, seconds: float) -> None:
    """Put the take on the channel from ``seconds``, honouring the region.

    **Inside the marked region, the loop -- not a slice of it (M10).** This
    used to slice ``pcm[start:loop_end]`` and repeat *that* whenever a region
    existed at all, so seeking to a point inside a 2-8s region shrank what
    actually looped to "seek point to loop end" -- a seek to 5s left only
    three seconds repeating, not the six-second region the markers claim, and
    a seek past the region's own end still looped (the unbounded remainder to
    the end of the take, forever). The fix rotates the *whole* region's loop
    body -- :func:`muse_io.loop_body`, the crossfaded one ``export_loop``
    writes (M09) -- so it starts sounding exactly at ``seconds`` and wraps
    through the rest of the region before repeating: one lap in, it is
    indistinguishable from having started the loop at ``seconds`` and let it
    run.
    ``position()`` reads ``loop_anchor`` back to undo the rotation.

    Outside the region -- before its start, or at/after its end -- this is an
    ordinary play-through to the end of the take, never a loop: a region
    existing at all is a different question from whether *this* seek landed
    inside it, which the old unconditional ``repeat`` flag never asked.

    **Both branches toast on refusal (muse-01, 2026-09-11 audit).** Neither
    ``sirens_audio.play`` call here had an ``else`` before this: the mirror
    ``on_task_done`` already toasts ``unavailable_reason()`` on the
    first-decode landing path, but resuming an already-decoded take (the
    ordinary Stop-then-Play case), seeking while "playing", and "Play the
    loop" all route through *this* function -- so on a device-less machine
    every one of those pressed a genuinely dead button with nothing said.

    **A stale cache defers rather than refuses (muse-02, 2026-09-18 audit).**
    ``muse_io.loop_body`` used to be called unconditionally here -- and on a
    cache miss it blends the whole region from scratch, an O(n) pass this
    module's own ``precompute_loop`` docstring measures at ~100 ms on a 240 s
    take. Right on the frame thread: a marker drag's release or the crossfade
    slider's release both call ``precompute_loop`` and then return, so "Play
    right after moving a marker" is exactly the window where that task has
    not landed yet -- the stall muse-03 (2026-09-05 audit) fixed for the
    export path, just left open for the first Play after one. A first pass at
    this finding refused outright ("try again in a moment"), which made a
    user press twice for something they asked for once; ``one.pending_play``
    is what lets the *second* press be ``on_task_done``'s, not the user's --
    see its own comment below for the shape and what invalidates it.
    """
    import numpy as np

    rate = one.rate
    has_region = one.loop_start is not None and one.loop_end is not None
    if has_region and one.loop_start <= seconds < one.loop_end:
        cache_key = muse_io.loop_cache_key(one)
        if cache_key is not None and one.loop_cache_key != cache_key:
            # **muse-02** (2026-09-18 audit). Not a ``Player`` dataclass field
            # (``state.py``): deliberately the one piece of this call's own
            # request that outlives it, read back only by ``on_task_done``'s
            # ``CACHE_PREFIX`` branch and by :func:`stop`, and nowhere else --
            # ephemeral intent, not state a pane ever draws. ``(job, seconds,
            # cache_key)`` is the take, the position and the exact region this
            # was waiting for, so a landing result plays only when all three
            # still match: a different take (a fresh ``Player`` replaces this
            # one entirely, so it never carries this attribute at all), a
            # region changed since (``cache_key`` no longer equal to the
            # cache ``on_task_done`` just installed), or Stop pressed
            # (:func:`stop` clears it) each drop the request silently rather
            # than starting a loop nobody is waiting for any more.
            already_waiting = getattr(one, "pending_play", None)
            one.pending_play = (one.job, seconds, cache_key)
            precompute_loop(ctx)
            if already_waiting is None or already_waiting[2] != cache_key:
                # Progressive, not repeated: a second Play (or a seek) while
                # still waiting on the same region must not toast again --
                # ``precompute_loop`` is already idempotent for the same
                # reason.
                ctx.toast("Preparing the loop...")
            return
        body = muse_io.loop_body(one)
        length = one.loop_end - one.loop_start
        if body is None or len(body) == 0 or length <= 0:
            return
        phase = (seconds - one.loop_start) % length
        cut = min(max(int(round(phase * rate)), 0), len(body))
        buffer = np.concatenate([body[cut:], body[:cut]]) if cut else body
        if len(buffer) == 0:
            return
        if sirens_audio.play(buffer, rate, tag=one.job, loops=-1):
            one.play_offset = seconds
            one.loop_anchor = seconds
            ensure(ctx).playing_job = one.job
        else:
            ctx.toast(sirens_audio.unavailable_reason() or "could not play that take", "warn")
        return

    one.loop_anchor = None
    start = int(seconds * rate)
    tail = one.pcm[start:]
    if len(tail) == 0:
        return
    if sirens_audio.play(tail, rate, tag=one.job, loops=0):
        one.play_offset = seconds
        ensure(ctx).playing_job = one.job
    else:
        ctx.toast(sirens_audio.unavailable_reason() or "could not play that take", "warn")


def play_region(ctx: Any) -> None:
    """Play the loop region on repeat, from its start. The audition that
    matters: a seam is a thing you judge by hearing it come round again."""
    one = player(ctx)
    if one is None or one.loop_start is None:
        return
    _play_from(ctx, one, one.loop_start)


def remember_loop(ctx: Any) -> None:
    """File the current take's loop region and crossfade under its job id (W4).

    Called wherever the ``Player`` is about to be replaced. Three floats per
    job, capped at :data:`muse_state.LOOP_MEMORY` and oldest-first -- see
    ``MuseState.loop_memory`` for why this is not the PCM cache the ``Player``
    docstring refuses.
    """
    state = active(ctx)
    one = state.player if state is not None else None
    if state is None or one is None or not one.job:
        return
    state.loop_memory.pop(one.job, None)
    state.loop_memory[one.job] = (one.loop_start, one.loop_end, float(one.xfade_ms))
    while len(state.loop_memory) > LOOP_MEMORY:
        # ``dict`` keeps insertion order, so the first key is the oldest --
        # which is what makes a bounded LRU here two lines rather than a class.
        state.loop_memory.pop(next(iter(state.loop_memory)))


def set_region(ctx: Any, start: float | None, end: float | None) -> None:
    """Set or clear the loop markers, in seconds. Ordered and clamped here.

    One place, because the markers are set from four: the finder's answer, the
    two draggable grips, and the ``[``/``]`` keys. Four copies of "clamp, then
    swap if reversed" is four chances to leave a region the exporter refuses.
    """
    one = player(ctx)
    if one is None:
        return
    if start is None or end is None:
        one.loop_start = one.loop_end = None
        return
    low = min(max(float(start), 0.0), one.duration)
    high = min(max(float(end), 0.0), one.duration)
    one.loop_start, one.loop_end = min(low, high), max(low, high)


def find_loops(ctx: Any) -> None:
    """Ask for loop points. The answer lands in ``on_task_done``.

    On a task because a four-minute take is a full STFT and a Gram matrix; the
    strip draws a spinner meanwhile, which is what "being wrong costs a spinner
    rather than a frozen window" buys.
    """
    one = player(ctx)
    if one is None or one.pcm is None:
        return
    one.finding = True
    if not ctx.submit(f"{muse_io.FIND_PREFIX}{one.job}", muse_io.find_loops, one.pcm, one.rate):
        one.finding = False
        ctx.toast("Still looking for loop points.")


def choose_candidate(ctx: Any, index: int) -> None:
    """Adopt one of the finder's answers as the region.

    By index rather than by value because the strip offers them as a numbered
    list -- and the list is the point: one answer with no alternatives would
    claim a confidence the method does not have (``muse.loops``).
    """
    one = player(ctx)
    if one is None or not 0 <= index < len(one.candidates):
        return
    candidate = one.candidates[index]
    set_region(ctx, candidate.start / one.rate, candidate.end / one.rate)
    precompute_loop(ctx)


def precompute_loop(ctx: Any) -> None:
    """Recompute the crossfaded loop body on a task now that the region or
    the crossfade has settled, so the next ``_play_from`` finds it already
    cached.

    **muse-03** (2026-09-05 audit): ``_play_from`` calls ``muse_io.loop_body``
    on the frame thread, and that function computes on a cache miss -- which
    used to be exactly what happened on the very first Play after ``Find loop
    points`` or a marker drag, since the cache key is ``(start, end, fade)``
    and those are precisely the numbers that just changed. On a 240 s take
    that stall was ~100 ms, for the action (auditioning a seam while judging a
    crossfade) the strip exists for.

    Called from every place a region or a crossfade *settles* -- a candidate
    chosen, a marker drag's release, the ``[``/``]`` keys, the crossfade
    slider's release -- and nowhere a value merely changes mid-drag: doing
    this on every frame of a drag would turn the one O(n) blend the old code
    paid for into dozens.

    **The task only computes; `on_task_done` installs it (incident-2026-09-05b,
    a correction to this finding's first pass).** Submitting ``loop_body``
    itself as the task moved the blend off the frame thread but left it
    writing the cache pair -- two separate statements -- from the task
    thread, which a ``_play_from`` call for a *different*, newer key could
    interleave with, pairing one key with the other's buffer. This submits
    ``compute_loop_cache`` instead, which touches nothing on ``player``; see
    its docstring for the exact interleaving this closes.
    """
    one = player(ctx)
    if one is None:
        return
    key = muse_io.loop_cache_key(one)
    if key is None or one.loop_cache_key == key:
        return  # no region, or already current -- nothing to precompute
    ctx.submit(f"{muse_io.CACHE_PREFIX}{one.job}", muse_io.compute_loop_cache, one)


# --- keys -------------------------------------------------------------------


def select(ctx: Any, jobs: list[Any], delta: int) -> None:
    """Move the tray's selection by ``delta``, wrapping. ``jobs`` is the tray's
    own list, passed in rather than recomputed: the pane already has it, and a
    second query could disagree about what "newest" means on the frame a take
    lands."""
    if not jobs:
        return
    state = ensure(ctx)
    ids = [str(job["id"]) for job in jobs]
    here = ids.index(state.selected_job) if state.selected_job in ids else 0
    state.selected_job = ids[(here + delta) % len(ids)]


def handle_key(ctx: Any, event: Any) -> bool:
    """Muse's keyboard. -> whether the press was consumed.

    Small, and deliberately so: this mode has two verbs. Space auditions the
    selected take -- which is why ``muse`` is in ``modes.NAV_KEY_MODES``, since
    one press must not also activate whatever button imgui's focus ring is on
    -- and Ctrl+Enter presses Generate from wherever the caret is, which is the
    binding every other form in this app already carries.

    Up/Down move the tray's selection. They are bound rather than left alone
    for the reason ``poser_mode.sheet_handle_key`` states at length (Troupe's
    own ``handle_key``, before P9 2026-09-18 folded that mode into Poser):
    membership of ``NAV_KEY_MODES`` withholds those keys from imgui whether or
    not anything binds them, so an unbound one is a key taken from one
    consumer and given to none.

    The player's six are the same bargain, and cheap for the same reason.
    Left/Right nudge the playhead (Shift, ten times as far), Home returns it to
    the start, ``[`` and ``]`` set the loop's two ends *at the playhead* -- which
    is what makes the keyboard a real alternative to dragging a grip rather than
    a shortcut for the buttons -- and ``L`` runs the finder.

    Every one of them is a no-op with no player, which is the honest floor: they
    are about a decoded take, and before the first audition there is none.

    **Presses only.** Acting on ``event.key`` without looking at ``event.type``
    runs every branch twice per press, which for a play/stop toggle means
    silence.
    """
    import pygame

    if event.type != pygame.KEYDOWN:
        return False
    from .ui.panes import results as muse_results

    ctrl = bool(event.mod & pygame.KMOD_CTRL)
    if ctrl and event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
        generate(ctx)
        return True
    state = ensure(ctx)
    if event.key == pygame.K_SPACE:
        job_id = state.selected_job
        if not job_id:
            return True
        if is_playing(ctx, job_id):
            stop(ctx)
        else:
            play(ctx, job_id)
        return True
    if event.key in (pygame.K_UP, pygame.K_DOWN):
        select(ctx, muse_results.plan_for(ctx), -1 if event.key == pygame.K_UP else 1)
        return True

    one = player(ctx)
    if one is None:
        return False
    if event.key in (pygame.K_LEFT, pygame.K_RIGHT):
        step = NUDGE_SECONDS * (NUDGE_MULTIPLIER if event.mod & pygame.KMOD_SHIFT else 1)
        seek(ctx, position(ctx) + (step if event.key == pygame.K_RIGHT else -step))
        return True
    if event.key == pygame.K_HOME:
        seek(ctx, 0.0)
        return True
    if event.key == pygame.K_LEFTBRACKET:
        # At the playhead, and against whichever end already exists.
        # ``set_region`` orders the pair, so setting a start past the end is a
        # region with its two markers swapped rather than an invalid one.
        set_region(ctx, position(ctx), one.loop_end if one.loop_end is not None else one.duration)
        precompute_loop(ctx)
        return True
    if event.key == pygame.K_RIGHTBRACKET:
        set_region(ctx, one.loop_start if one.loop_start is not None else 0.0, position(ctx))
        precompute_loop(ctx)
        return True
    if event.key == pygame.K_l:
        find_loops(ctx)
        return True
    return False


# --- the bridge -------------------------------------------------------------


def open_in_sirens(ctx: Any, job_id: str) -> bool:
    """Land a take in the tracker as a sample instrument. -> whether it started.

    The one thing that makes the two audio modes a *pair* rather than two
    unrelated features: Muse writes a 44.1 kHz WAV, and Sirens' sample
    instruments read 44.1 kHz WAVs.

    Every door already existed. ``import_sample`` submits the decode and
    ``sirens_mode.adopt_sample`` adopts the result exactly as it does for a
    user's own drag-and-drop, so a generated track is not a special kind of
    sample and nothing in Sirens has to know where it came from.

    **The window moves when the take lands, not when the button is pressed.**
    The decode is a task and it can be refused; switching first meant a user
    read "this sample could not be loaded" in the tracker, with the take they
    were looking at a mode away. ``switch=True`` rides the task instead, and
    ``sirens_mode.on_task_done`` calls the one ``set_mode`` after the adopt.
    """
    path = track_path(ctx, job_id)
    if not path.exists():
        ctx.toast("that take has no audio on disk", "warn")
        return False
    tab = sirens_mode.active(ctx)
    if tab is None:
        tab = sirens_mode.new_document(ctx)
    sirens_io.import_sample(ctx, tab, path, switch=True)
    return True


def compose_from_sirens(ctx: Any, tab: Any = None) -> bool:
    """Render the open song and use it as a reference for a new take.

    ``open_in_sirens``'s mirror, and the direction the manual called
    deliberately unbuilt. **The Sirens half opens no door at all**: the document
    is serialised on the frame thread (the only thread it is safe to read on),
    rendered and encoded on a task thread, and all three of ``rsng_bytes``,
    ``read_rsng``/``synth.render`` and ``wavout.wav_bytes`` are already used in
    exactly this combination by ``sirens_play.request_render`` and
    ``sirens_io.export_plan``.

    The reference therefore carries **the loop points the user authored**,
    because ``wav_bytes`` writes them into the ``smpl`` chunk and this passes
    them. That is the headline feature meeting the round trip rather than
    fighting it.

    The rates already agree -- ``synth.SAMPLE_RATE`` is 44100 and the model's
    own loader resamples anything -- and ``tests/modes/muse/test_muse_bridge.py`` asserts
    it rather than a comment claiming it.

    One door *is* opened, on the Muse side: ``create_music_job`` takes scalars
    only, so it gains ``reference_wav``. See that function for why bytes rather
    than a path. It also takes ``ref_audio_strength`` -- *Closeness*, drawn
    beside the button that calls this, because how near to stay to the song is a
    property of this hand-off and not of the brief (W1).
    """
    from ....service import jobs as svc_jobs
    from ...panes import model_gate
    from ..sirens.engine import rsng

    # muse-02 (2026-09-14 audit): this door does not call ``generate``, so its
    # own model_gate check does not cover it -- Sirens' "Compose in Muse" used
    # to submit straight to ``create_music_job`` and get the door's own
    # ``music_model`` refusal instead of the gate that names the Recipe panel.
    if model_gate.missing(ctx, svc_jobs.MUSIC_ROWS):
        ctx.toast("The music model is not downloaded. See the Recipe panel.", "warn")
        return False

    state = ensure(ctx)
    strength = float(state.compose_strength)
    tab = tab or sirens_mode.active(ctx)
    if tab is None or not tab.doc.order:
        ctx.toast("There is nothing in the order list to compose from.", "warn")
        return False
    if not str(state.form["prompt"]).strip():
        # Refused here rather than at the door, because the door's sentence
        # would arrive in Muse pointing at a field the user is not looking at.
        # The tags are what the model is being asked *for*; the song is only
        # what it is being asked to sound like.
        ctx.toast("Describe the music you want in Muse first, then compose.", "warn")
        set_mode(ctx.state, "muse")
        return False

    # The snapshot, on the frame thread. ``request_render``'s rule and its
    # reason: this is where the document is safe to read.
    data = rsng.rsng_bytes(tab.doc)
    form = dict(state.form)

    def run():
        from ....kernels.audio import wavout
        from ....service.errors import invalid_from
        from ..sirens.engine import synth

        try:
            doc = rsng.read_rsng(data)
            samples, loop, _marks = synth.render_marked(doc)
        except ValueError as exc:
            raise invalid_from(exc, "That song did not render") from exc
        reference = wavout.wav_bytes(samples, synth.SAMPLE_RATE, loop=loop)
        return svc_jobs.create_music_job(
            ctx.svc,
            prompt=str(form["prompt"]),
            lyrics=str(form["lyrics"]),
            duration=float(form["duration"]),
            count=int(form["count"]),
            seed=form["seed"],
            infer_step=int(form["infer_step"]),
            guidance_scale=float(form["guidance_scale"]),
            scheduler_type=str(form["scheduler_type"]),
            cfg_type=str(form["cfg_type"]),
            omega_scale=float(form["omega_scale"]),
            reference_wav=reference,
            ref_audio_strength=strength,
        )

    ctx.state.clear_field_errors()
    if not ctx.submit("submit", run):
        ctx.toast("Still submitting the last one - try again in a moment.")
        return False
    ctx.state.remember_prompt(str(form["prompt"]))
    # Through the one mode-switch implementation, never a direct assignment.
    set_mode(ctx.state, "muse")
    return True


def reset_form(ctx: Any) -> None:
    """Put the brief back to its defaults. The palette's Reset."""
    state = ensure(ctx)
    state.form = dict(DEFAULT_FORM)
    # Not part of the form -- see ``MuseState.duration_custom`` -- but still a
    # view of the brief, so a Reset that left the duration control parked on
    # Custom while the number underneath it snapped back to 60 would show two
    # different answers to "how long".
    state.duration_custom = False


__all__ = [
    "DEFAULT_DERIVE",
    "DEFAULT_FORM",
    "LOAD_PREFIX",
    "MuseState",
    "DERIVE_CONTROLS",
    "SINGLE_TAKE_TASKS",
    "active",
    "close_derive",
    "compose_from_sirens",
    "derive",
    "ensure",
    "generate",
    "handle_key",
    "is_playing",
    "choose_candidate",
    "find_loops",
    "has_stems",
    "on_task_done",
    "open_derive",
    "open_in_sirens",
    "play",
    "play_region",
    "player",
    "position",
    "remember_loop",
    "reset_form",
    "seek",
    "set_region",
    "select",
    "separate",
    "stop",
    "sync",
    "track_path",
]

# Referenced by the tray's card for its "which document would this land in"
# tooltip; imported here so the pane does not need a second Sirens import.
sirens_tab_title = sirens_state.title_for
