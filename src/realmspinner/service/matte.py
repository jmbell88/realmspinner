"""The matte a mesh job will actually be reconstructed from, shown before it.

Two halves, and they answer different questions.

**The preview** is what the promote flow puts in front of the user: the host's
own BiRefNet cutout (or, with no weights, the corner fill) composited over a
checkerboard so transparency is visible as transparency, plus the composition
gate's own verdict on the same image. It exists because the matte is the single
decision that most often turns a good reference into a solid slab, and until
now it was made inside ``trellis-server.exe`` two minutes after the user
committed. Every byte of it is computed off the frame thread -- BiRefNet is
seconds of host compute -- and the result is cached by ``(job id, input.png
mtime)`` under git's racily-clean rule, exactly as ``files.attach_files``
caches its listings and for exactly the same reason: a Windows mtime comes off
a 15.6 ms clock, so a write landing inside the stamped tick would otherwise be
invisible to the stamp forever.

**Approval** is the other half, and it has two doors rather than one. A
reference that carries a real (non-opaque) alpha channel is a matte somebody
already made -- by hand in Inker, or painted. *And accepting this module's own
preview is the second*: a person looked at the cutout, at full size, over a
checkerboard, with the gate's verdict beside it, and pressed Accept. That is
the same act, and treating it as less than one is what made the modal a
decoration. So :func:`prepare` writes the cutout down and ``promote_to_model``
copies *that* file into the mesh job, where ``approve`` finds the alpha and
records it. Either way the whole point of having approved it is that it is the
matte trellis reconstructs from. ``trellis-server.exe --help`` states the
server's own rule::

    --bg-removal MODE   threshold | birefnet   (default: auto -- a pre-matted
                        image keeps its alpha; otherwise BiRefNet when its
                        model is present. ...)

So ``auto`` is the preserving mode and ``birefnet`` -- which is
``guidance.DEFAULT_BG_REMOVAL`` on a host that has the weights -- would re-cut
an approved cutout. ``approve`` therefore records ``matte: approved`` *and*
pins ``bg_removal`` to the preserving mode. An explicit override loses to it on
purpose: the user approved that cutout, and a mode that re-mattes would make
the approval a lie.

And it re-mattes with a **different model**, which is the part that made this
worth fixing rather than tidying. The host's BiRefNet lives under
``t2i_model_root / birefnet``; the server's is ``birefnet.gguf`` under
``REALMSPINNER_TRELLIS_MODELS``. Two files, two directories, two sets of weights. So
"the server will cut it the same way" was never a safe assumption to leave
standing behind a modal whose entire purpose is to show the user the cut.

Pure in the way ``pipelines/reference.py`` is: numpy and Pillow are imported
inside the functions that need them, so importing this module costs nothing on
a host with no image extra.
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..provenance import file_fingerprint
from .core import RealmspinnerService
from .errors import Invalid
from .files import MTIME_RACE_NS
from .validation import check_job_id

log = logging.getLogger(__name__)

# The approved pixels, full resolution, RGBA, written beside the reference they
# were cut from.
#
# **Deliberately not a served name.** It is a working file the way ``rig.json``
# and ``sheet.json`` are -- written by one function, read by two -- so it stays
# out of ``files.LISTED``, ``files.MIME``, ``DERIVED_2D`` and the inspector's
# list. Serving it would mean teaching ``fresh_2d`` a staleness rule it does not
# need: freshness here is a *fingerprint* comparison (see :func:`prepared`),
# which is strictly stronger than the mtime rule those names use, and a second
# weaker answer beside it is how the two come to disagree. What a user wanting
# to see the approved pixels opens is the promoted mesh job's own ``input.png``,
# which *is* this file, and is listed.
CUTOUT = "cutout.png"

# Where the record of that file lives on the reference's own row. Written
# *after* the PNG is in place, which makes it the completion gate: a crash
# between the two leaves a cutout nothing claims, and the next ``prepared``
# reads it as absent and cuts again.
CUTOUT_PARAM = "cutout"

# What ``params["matte"]`` says when the alpha on disk is a cutout somebody
# looked at. A string rather than a bool because it is a config-vector value
# (``vectors.VECTOR_PARAMS``) and a future ``matte: auto`` sits beside it.
APPROVED = "approved"

# The mode that keeps an existing alpha channel, per the exe's own help text
# quoted above. Named here rather than spelled "auto" at the call site, so the
# reason travels with the value.
PRESERVING_BG_REMOVAL = "auto"

# The checkerboard the cutout is drawn over, and its cell size in preview
# pixels. Two greys rather than the classic white/grey pair: the references
# this shows are overwhelmingly light subjects on a light background, and a
# white square behind a white rim is exactly the case the checkerboard exists
# to make visible.
CHECKER_LIGHT = 0xB4
CHECKER_DARK = 0x8C
CHECKER_CELL = 8

# The longest side the preview is scaled to. It is drawn in a modal at a few
# hundred design pixels, so a 1024-square source would be three quarters of a
# megabyte of texture spent on detail nothing can show.
PREVIEW_MAX = 384


@dataclass(frozen=True, slots=True)
class Preview:
    """One reference's matte, as a picture of :class:`Prepared`'s file.

    Transforms nothing. It no longer *writes* nothing: :func:`preview` goes
    through :func:`ensure_prepared`, so looking at a cutout is what puts it on
    disk. That is the point rather than a side effect -- the alternative is the
    modal computing one set of pixels and the promotion computing another, which
    is the defect this pair was rebuilt to remove -- but it is worth stating
    where a reader will look for it, because "the preview is pure" was true for
    as long as the preview was also a lie.
    """

    job_id: str
    # ``input.png``'s mtime when the pixels below were read, in ns. Read
    # *before* the read, so the clock the racily-clean rule compares it against
    # is unambiguously later than it.
    stamp: int | None
    width: int
    height: int
    # The cutout over the checkerboard, RGB8, ``width * height * 3`` bytes --
    # ready for a GL texture with no decode on the frame thread.
    rgb: bytes
    # Which of matting.py's three sources answered: alpha | birefnet | flood.
    source: str
    # Whether ``input.png`` already carried a matte somebody made -- i.e. whether
    # the cutout on screen is that matte rather than one the host just made.
    # It no longer decides whether promoting records ``matte: approved``:
    # accepting this preview *is* the approval, so a promotion through the modal
    # records it either way. What this still says is where the edge came from,
    # which is the difference between "kept" and "cut for you".
    approved: bool
    # The fraction of the frame the matte keeps. A cutout that kept 2% or 99%
    # of the frame is worth seeing as a number as well as a picture.
    coverage: float
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Prepared:
    """The pixels a promotion will reconstruct from, on disk and fingerprinted.

    **The object the preview and the promotion both consume**, and it exists
    because until now they consumed different things. The modal showed the
    host's BiRefNet cutout; ``promote_to_model`` copied the untouched
    ``input.png``, ``matte.approve`` found no alpha to approve, ``bg_removal``
    stayed ``birefnet``, and ``trellis-server.exe`` re-cut the image with *its
    own* ``birefnet.gguf`` under ``REALMSPINNER_TRELLIS_MODELS`` -- a different model
    file in a different directory from the one the user had just looked at. The
    picture in the modal was therefore a claim about pixels nothing downstream
    ever saw, except in the one case where the reference had already been matted
    by hand in Inker.

    So the cutout is written down. ``path`` is a real file under the reference's
    own job directory, every candidate of a promotion copies those same bytes,
    and ``src_fingerprint`` is what makes the approval expire on its own: an
    Inker save, a revert or a reroll moves ``input.png``, the fingerprints stop
    matching, and :func:`prepared` answers None rather than handing back a
    cutout of pixels that are gone. There is no second invalidation rule to
    remember and nothing to unlink beside each writer -- the same argument
    ``files.fresh_2d`` makes for asking the question at the moment somebody
    wants the answer.
    """

    job_id: str
    path: Path
    # Which of matting.py's three sources answered: alpha | birefnet | flood.
    source: str
    # ``input.png``'s fingerprint when the cut was taken. The expiry.
    src_fingerprint: str
    # The cutout's own, which is what a promoted job records and what the
    # composition override is anchored to.
    fingerprint: str
    # The fraction of the frame the matte keeps, mean alpha over 255. Soft
    # edges make this a real fraction rather than a pixel count, which is why
    # it is not simply ``mask.sum()``.
    coverage: float
    # ``reference.measure`` of the cutout itself -- not of the reference it was
    # cut from. Those differ, and deliberately: an alpha mask and a corner flood
    # fill do not find the same components, and this is the one measured against
    # the pixels that will actually be reconstructed.
    report: dict[str, Any]

    def as_params(self) -> dict[str, Any]:
        """The row's record of this file. See ``CUTOUT_PARAM``."""
        return {
            "source": self.source,
            "src_fingerprint": self.src_fingerprint,
            "fingerprint": self.fingerprint,
            "coverage": self.coverage,
            "report": self.report,
        }


def prepare(svc: RealmspinnerService, job_id: str) -> Prepared:
    """Cut ``job_id``'s reference out, write it down, record it. Off-thread.

    Seconds of host compute when BiRefNet's weights are present, so every caller
    goes through the TaskRunner -- the same rule :func:`preview` has always had,
    and the reason this is not called from a draw function.

    Staged through a dotfile and ``os.replace``d, with the ``finally`` unlink
    ``reference.prepare``, ``trellis._atomic_write`` and ``optimize.run`` all
    have: ``cutout.png`` is a served name, so it is never partially written, and
    a raising save leaves no fragment. ``merge_params`` rather than
    ``set_params`` because this runs off the frame thread while the worker may
    be writing other keys on the same row.

    **The whole body runs under** ``svc.convert_lock(job_id, CUTOUT)``, the
    same lock ``poses.py``/``rig.py`` take around their own staged writes in
    this segment. ``ensure_prepared`` has three independent doors --
    ``matte_preview.py``'s modal, ``inker_open.py``'s Inker hand-off, and
    ``_jobs_resubmit.py``'s promote/rerun -- and nothing stopped two of them
    from racing this function for the same job_id: the staging temp was a
    *fixed* name (``dest.with_name(f".{dest.name}.tmp")``, unlike
    ``files._staged_write``'s tokenized one), so two concurrent callers wrote
    and unlinked the same path under each other, and a reproduction against
    the real service measured a Windows ``PermissionError`` (WinError 5/32)
    on ``os.replace`` from both threads with ``cutout.png`` left MISSING once
    both finished (service-02, 2026-09-11 audit).

    A lock was chosen over tokenizing the temp name (the other pattern
    available here) because a token only fixes the filesystem collision --
    two callers would still both cut, both write their *own* complete file,
    and both ``merge_params`` a record, with whichever ``os.replace`` lands
    last deciding which bytes ``cutout.png`` ends up holding and which
    caller's record describes them. That is a second caller *observing* the
    first's cut only by accident (the flood-fill/BiRefNet cut of one
    ``input.png`` happens to be deterministic today). The lock instead makes
    a second caller observe the first's completed write and act after it --
    serialized, not merely non-colliding -- which is what
    ``ensure_prepared``'s callers actually need: a promote/rerun that lands
    behind an in-flight preview must see *a* finished cutout, not a torn race
    between two of them.
    """
    from PIL import Image

    from ..pipelines import reference

    src = _reference_path(svc, job_id)
    svc.require_job(job_id)

    with svc.convert_lock(job_id, CUTOUT):
        # Read before the cut, so a file that changes *during* it fingerprints
        # as the version we did not use and the record expires immediately.
        # The other order would stamp the new bytes onto the old cutout,
        # permanently.
        src_fingerprint = file_fingerprint(src)
        rgba, source, _approved = _cut(svc, src)
        coverage = float(rgba[:, :, 3].mean()) / 255.0

        image = Image.fromarray(rgba, "RGBA")
        dest = svc.job_dir(job_id) / CUTOUT
        tmp = dest.with_name(f".{dest.name}.tmp")
        try:
            image.save(tmp, format="PNG")
            os.replace(tmp, dest)
        finally:
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)

        out = Prepared(
            job_id=job_id,
            path=dest,
            source=source,
            src_fingerprint=src_fingerprint,
            fingerprint=file_fingerprint(dest),
            coverage=coverage,
            report=reference.measure(image).as_dict(),
        )
        # Last, and that ordering is the completion gate -- see CUTOUT_PARAM.
        svc.store.merge_params(job_id, {CUTOUT_PARAM: out.as_params()})
        return out


def prepared(svc: RealmspinnerService, job_id: str) -> Prepared | None:
    """The recorded cutout for exactly these reference pixels, or None.

    None means "cut it again", never an error: an absent file, a row with no
    record, a record naming a fingerprint ``input.png`` no longer has, and a
    half-written pair all mean the same thing to every caller.
    """
    job = svc.store.get(job_id)
    if job is None:
        return None
    record = (job.get("params") or {}).get(CUTOUT_PARAM)
    if not isinstance(record, dict):
        return None
    dest = svc.job_dir(job_id) / CUTOUT
    src = svc.job_dir(job_id) / "input.png"
    if not dest.exists() or not src.exists():
        return None
    if record.get("src_fingerprint") != file_fingerprint(src):
        return None
    if record.get("fingerprint") != file_fingerprint(dest):
        # The cutout itself moved under the record -- a hand edit of the file,
        # or a half-written replace. The record is about bytes that are gone.
        return None
    return Prepared(
        job_id=job_id,
        path=dest,
        source=str(record.get("source") or "flood"),
        src_fingerprint=str(record["src_fingerprint"]),
        fingerprint=str(record["fingerprint"]),
        coverage=float(record.get("coverage") or 0.0),
        report=dict(record.get("report") or {}),
    )


def ensure_prepared(svc: RealmspinnerService, job_id: str) -> Prepared:
    """:func:`prepared` if it is still current, otherwise :func:`prepare`."""
    return prepared(svc, job_id) or prepare(svc, job_id)


def stamp_for(path: Path) -> int | None:
    """``path``'s mtime in ns, or None when it is not there."""
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def cached(cache: dict, job_id: str, stamp: int | None) -> Preview | None:
    """The remembered preview for exactly this version of the file, or None."""
    hit = cache.get(job_id)
    if hit is not None and hit.stamp == stamp:
        return hit
    return None


def remember(cache: dict, preview: Preview) -> bool:
    """Store ``preview`` if its stamp is safely in the past. -> whether it was.

    git's racily-clean rule, and the whole of it. A file's mtime on Windows is
    written from the system clock, whose tick is 15.6 ms unless something has
    asked for better, so a write landing after the pixels were read but still
    inside the stamped mtime's own tick moves nothing the stamp can see -- and
    not for one tick but permanently, because every later comparison keeps
    matching. Here that is a re-save from Inker landing milliseconds after a
    preview was taken: the cutout on screen would then describe pixels that no
    longer exist, and no amount of re-checking would ever notice.

    **The clock is read here rather than in ``preview``, and that ordering is
    the proof**: the hazard needs a write later than the read yet inside the
    mtime's tick, which cannot exist if the read had already finished a tick
    after the mtime. This runs on the frame thread when the task's result is
    adopted, which is strictly after the read finished.

    A refusal costs one recomputation, which is the right way round.
    """
    if preview.stamp is None:
        return False
    if time.time_ns() - preview.stamp <= MTIME_RACE_NS:
        return False
    cache[preview.job_id] = preview
    return True


def replace_stamp(preview: Preview, stamp: int | None) -> Preview:
    """A copy wearing a different stamp -- the racily-clean rule's test hook,
    since a real one cannot be aged without sleeping through a tick."""
    return replace(preview, stamp=stamp)


def preview(svc: RealmspinnerService, job_id: str) -> Preview:
    """Cut ``job_id``'s reference out and report on it. Blocking; off-thread.

    Seconds of host compute when BiRefNet's weights are present, which is why
    every caller goes through the TaskRunner and the frame thread only adopts
    the answer.

    **A picture of :func:`prepare`'s file, never a second cut of its own.** The
    modal used to composite an in-memory RGBA that nothing else ever saw, which
    is how it came to show a cutout the reconstruction did not use. Thumbnailing
    the file on disk makes "what the user approved" and "what TRELLIS is handed"
    the same bytes by construction rather than by two functions agreeing.
    """
    import numpy as np
    from PIL import Image

    src = _reference_path(svc, job_id)
    job = svc.require_job(job_id)

    # Before the read, so ``remember``'s clock is unambiguously later than it.
    stamp = stamp_for(src)
    ready = ensure_prepared(svc, job_id)
    source, coverage = ready.source, ready.coverage
    approved = source == "alpha"

    with Image.open(ready.path) as small:
        small.load()
        small = small.convert("RGBA")
        small.thumbnail((PREVIEW_MAX, PREVIEW_MAX), Image.LANCZOS)
        composited = over_checkerboard(np.asarray(small, dtype=np.uint8))
    height, width = composited.shape[:2]

    # The gate's verdict on the *reference*, not on the cutout: these are the
    # sentences ``promote_to_model`` will refuse with, and showing the cutout's
    # own report instead would offer a Build-anyway button for a refusal that
    # was never going to fire. ``ready.report`` is the other measurement and is
    # recorded on the row for anyone comparing the two.
    report = (job.get("params") or {}).get("reference_report")
    if not isinstance(report, dict):
        from ..pipelines import reference

        report = reference.measure_file(src).as_dict()
    return Preview(
        job_id=job_id,
        stamp=stamp,
        width=int(width),
        height=int(height),
        rgb=composited.tobytes(),
        source=source,
        approved=approved,
        coverage=coverage,
        reasons=tuple(str(r) for r in (report.get("reasons") or ())),
        warnings=tuple(str(w) for w in (report.get("warnings") or ())),
    )


def alpha_plane(svc: RealmspinnerService, job_id: str) -> tuple[Any, str]:
    """-> (the full-resolution matte as a uint8 plane, which source cut it).

    What the Inker hand-off applies: 255 keeps a pixel, 0 cuts it, and anything
    between is a feathered edge that now survives (see :func:`_cut`). Read off
    :func:`prepare`'s file rather than cutting again, so the editor opens with
    the alpha the modal showed and the promotion would have used -- three
    consumers, one set of pixels. Blocking; off-thread, for ``preview``'s reason.
    """
    import numpy as np
    from PIL import Image

    _reference_path(svc, job_id)
    ready = ensure_prepared(svc, job_id)
    with Image.open(ready.path) as im:
        im.load()
        plane = np.asarray(im.convert("RGBA"), dtype=np.uint8)[:, :, 3]
    return plane, ready.source


def _reference_path(svc: RealmspinnerService, job_id: str) -> Path:
    check_job_id(job_id)
    src = svc.job_dir(job_id) / "input.png"
    if not src.exists():
        raise Invalid("this job has no reference image")
    return src


def _cut(svc: RealmspinnerService, src: Path) -> tuple[Any, str, bool]:
    """-> (RGBA with the matte as its alpha, the matte's source, approved?).

    One place, because the preview, the Inker hand-off and :func:`prepare` must
    never disagree about where the edge is: the picture the user accepted is the
    alpha the editor opens with, *and* the alpha the reconstruction runs on.

    **An alpha that is already a matte is carried through verbatim.** It used to
    be rebuilt from ``matting.mask``, which for this branch is
    ``reference.subject_mask`` -- ``alpha > 8``, a hard threshold. That threw a
    feathered edge away, and while the cutout was only ever a *picture* of the
    decision that cost nothing: promotion copied ``input.png`` byte for byte and
    the soft alpha survived anyway. Now that this RGBA becomes the pixels
    ``input.png`` is written *from*, hardening it would destroy a matte somebody
    painted, which is the one kind this module exists to protect.
    """
    import numpy as np
    from PIL import Image

    from ..pipelines import matting

    with Image.open(src) as im:
        im.load()
        approved = matting.is_cutout(im)
        rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
        if approved:
            alpha = np.asarray(im.convert("RGBA"), dtype=np.uint8)[:, :, 3]
            source = "alpha"
        else:
            mask, source = matting.mask(im, svc.config)
            alpha = np.asarray(mask, dtype=bool).astype(np.uint8) * np.uint8(255)
        rgba = np.dstack([rgb, alpha])
    return rgba, source, approved


def over_checkerboard(rgba: Any) -> Any:
    """Composite an RGBA array onto the checkerboard. -> an RGB uint8 array.

    Done here rather than by drawing a checkerboard behind the image in imgui:
    it is one numpy expression on a 384-square, it keeps the pane to a single
    texture and a single draw, and -- the reason that actually decides it -- it
    makes "what does the user see" assertable in a headless test.
    """
    import numpy as np

    h, w = rgba.shape[:2]
    ys = (np.arange(h) // CHECKER_CELL)[:, None]
    xs = (np.arange(w) // CHECKER_CELL)[None, :]
    board = np.where((ys + xs) % 2 == 0, CHECKER_LIGHT, CHECKER_DARK).astype(np.float32)
    board = np.repeat(board[:, :, None], 3, axis=2)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    out = rgba[:, :, :3].astype(np.float32) * alpha + board * (1.0 - alpha)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def approve(params: dict[str, Any], image: bytes | Path) -> bool:
    """Record an approved matte on ``params`` when ``image`` carries one.

    -> whether it did. Called from both doors onto a mesh job -- a promotion
    and an upload -- because both can arrive carrying a cutout: the promotion
    when the reference was fixed in Inker, the upload when Clay or Inker sent
    drawn pixels straight to 3D. ``files.to_png`` already preserves an alpha
    channel that was there, so nothing here has to make the alpha travel; what
    it has to do is stop the server throwing it away.
    """
    if not is_matted(image):
        return False
    params["matte"] = APPROVED
    # Last write wins over the normalized default (``birefnet`` on a host with
    # the weights), which would re-cut the cutout. See the module docstring for
    # the exe's own statement of what each mode does.
    params["bg_removal"] = PRESERVING_BG_REMOVAL
    return True


def is_matted(image: bytes | Path) -> bool:
    """Whether these pixels carry an alpha channel that is actually a matte.

    Defensive rather than strict: this decides an *addition* to params, so
    bytes that will not decode are simply not a matte -- the decode that
    matters already happened (``to_png``) or is about to (the worker's).
    """
    import io

    from PIL import Image

    from ..pipelines import matting

    try:
        source = io.BytesIO(image) if isinstance(image, bytes) else image
        with Image.open(source) as im:
            im.load()
            return matting.is_cutout(im)
    except Exception:
        return False
