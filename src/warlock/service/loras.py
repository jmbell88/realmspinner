"""Style LoRAs the user brings: importing a file, training one, removing one.

The door for ``generation.import_lora`` -- which was fully built and called
from no pane -- and for the trainer child. Everything here refuses with a
``field`` so the settings form can ring the control; the pane never
validates a second time.

A trained adapter is imported through the same call a downloaded file is, so
there is one registry path: a LoRA is a ``STYLE_LORAS`` entry with a file
under ``t2i_model_root/loras``, however it got there.
"""

from __future__ import annotations

import logging
import shutil
import uuid
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .. import generation, models
from ..pipelines import lora_train
from .core import WarlockService
from .errors import Invalid, TooLarge
from .validation import MAX_IMAGE_PIXELS, check_base_model_weights, check_pack, check_vram

log = logging.getLogger(__name__)

#: What a training image may be. Read through Pillow at the door, so a file
#: the trainer cannot open costs the request, not a queue slot and a load.
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")

#: How similar two candidate images' perceptual hashes must be for
#: ``library_training_set`` to treat them as the same picture rather than two
#: different ones. ``dev/measurements/2026-08-11-perceptual-hash-floor.md``
#: measured a lossless round trip (resize, JPEG re-encode) at a rescaled 1.0,
#: the worst real variation it tried -- a 1.4x brightness multiply -- at only
#: 0.66, and two genuinely different drawn objects at ~0.06, indistinguishable
#: from the noise floor. 0.92 sits above every "different picture" score that
#: measurement found and below only a lossless copy, so this only ever
#: collapses the same reference reused across two jobs or a plain resave --
#: never two merely similar renders, which is exactly what that document says
#: this metric cannot rank.
DUPLICATE_SIMILARITY = 0.92

#: A ceiling ``store.search_ids`` still wants, chosen so it is never the
#: limiting factor: a personal library's favourites are nowhere near this many
#: rows, and this is a background task (never the frame thread) either way.
_FAVOURITES_SCAN_LIMIT = 10_000


def catalog(svc: WarlockService) -> list[dict[str, Any]]:
    """Every style LoRA the picker offers, imported ones marked by ``source``."""
    return generation.lora_catalog(svc.config)


def imported(svc: WarlockService) -> list[dict[str, Any]]:
    """Only the adapters the user added, newest last."""
    return [asdict(m) for m in generation.load_lora_manifests(svc.config)]


def import_lora(
    svc: WarlockService,
    source: Path | str,
    *,
    label: str,
    family: str = models.FAMILY_SDXL,
    trigger_text: str = "",
    tuned_weight: float = models.DEFAULT_LORA_WEIGHT,
    commercial: bool = False,
    license: str = "",
) -> dict[str, Any]:
    """Copy a ``.safetensors`` adapter into managed storage and register it."""
    text = (label or "").strip()
    if not text:
        raise Invalid("give the style a name", field="label")
    if len(text) > lora_train.MAX_LABEL:
        raise Invalid(f"a name is at most {lora_train.MAX_LABEL} characters", field="label")
    if family not in models.FAMILIES:
        raise Invalid(f"family must be one of {list(models.FAMILIES)}", field="family")
    trigger = (trigger_text or "").strip()
    if len(trigger) > lora_train.MAX_TRIGGER:
        raise Invalid(
            f"trigger words are at most {lora_train.MAX_TRIGGER} characters", field="trigger_text"
        )
    try:
        weight = float(tuned_weight)
    except (TypeError, ValueError) as exc:
        raise Invalid("weight must be a number", field="tuned_weight") from exc
    if not 0.0 < weight <= models.LORA_WEIGHT_MAX:
        raise Invalid(
            f"weight must be between 0 and {models.LORA_WEIGHT_MAX}", field="tuned_weight"
        )
    path = Path(source)
    if path.suffix.lower() != ".safetensors":
        raise Invalid("a LoRA is a .safetensors file", field="source")
    if not path.is_file():
        raise Invalid(f"{path.name} is not a file", field="source")
    manifest = generation.import_lora(
        svc.config,
        path,
        label=text,
        family=family,
        trigger_text=trigger,
        tuned_weight=weight,
        license=license,
        commercial=bool(commercial),
        source_url="local file",
    )
    log.info("imported style LoRA %s as %s", path.name, manifest.key)
    return asdict(manifest)


def remove_lora(svc: WarlockService, key: str) -> dict[str, Any]:
    """Delete an imported adapter's file and manifest. Built-ins are refused."""
    manifest = generation.imported_lora(svc.config, key)
    if manifest is None:
        raise Invalid("that style was not imported here, so it cannot be removed", field="key")
    generation.remove_imported_lora(svc.config, key)
    return {"ok": True, "key": key}


def library_training_set(
    svc: WarlockService,
    *,
    favourites: bool = True,
    accepted_references: bool = True,
    usable_meshes: bool = True,
    dedupe: bool = True,
) -> dict[str, Any]:
    """The library's own accepted work, gathered into a LoRA training set.

    Three independent sources, each switchable and each counted on its own in
    the returned ``sources``:

    * every favourited job -- a broad "I liked this" signal;
    * every job whose latest human *reference* label is ``accept`` -- an
      explicit "this 2D asset is good";
    * the ``reference.png`` of every model job graded ``vectors.USABLE_GRADE``
      or better -- a mesh that reconstructed well is evidence its reference
      was a good blank, even though nobody labelled the picture itself.

    A job that qualifies through more than one source still contributes
    exactly one image -- it has exactly one reference file -- but ``sources``
    counts every job that qualified for *each* source, overlap and all, which
    is what the settings pane's summary line reports.

    Near-duplicates -- the same reference reused across two jobs, or a plain
    resave -- collapse to one image via ``bench.metrics.perceptual_hash`` at
    ``DUPLICATE_SIMILARITY``, keeping the EARLIER row: a style trained twice
    keeps whichever copy has been in the library longer.

    A candidate whose job row or reference file has since been pruned off
    disk -- or trashed -- is silently missing rather than an error:
    ``service.verdicts`` already documents that a verdict outlives the job it
    names, so a corpus built from a library that has since been tidied up must
    degrade, not raise.

    Refuses under ``lora_train.MIN_IMAGES`` the same way ``train_lora`` itself
    does, with ``field="images"`` so the pane can ring the same control.
    Capped at ``lora_train.MAX_IMAGES``, newest first.
    """
    from ..vectors import USABLE_GRADE
    from . import verdicts as verdicts_mod

    raw: dict[str, set[str]] = {
        "favourites": set(),
        "accepted_references": set(),
        "usable_meshes": set(),
    }
    if favourites:
        raw["favourites"] = set(
            svc.store.search_ids("", limit=_FAVOURITES_SCAN_LIMIT, favorite=True)
        )
    if accepted_references or usable_meshes:
        latest = svc.store.latest_verdicts()
        if accepted_references:
            raw["accepted_references"] = {
                v["job_id"]
                for v in latest
                if v["source"] == verdicts_mod.SOURCE_HUMAN
                and v["stage"] == "reference"
                and v["verdict"] == "accept"
            }
        if usable_meshes:
            raw["usable_meshes"] = {
                v["job_id"]
                for v in latest
                if v["source"] == verdicts_mod.SOURCE_HUMAN
                and v["stage"] == "model"
                and isinstance(v.get("grade"), int)
                and not isinstance(v.get("grade"), bool)
                and v["grade"] >= USABLE_GRADE
            }

    considered = set().union(*raw.values()) if any(raw.values()) else set()

    # (created_at, path) per candidate that is still on disk and not
    # trashed -- everything else is a pruned or trashed candidate, and is
    # counted in ``considered`` but never reaches ``paths``.
    resolved: dict[str, tuple[float, Path]] = {}
    for job_id in considered:
        job = svc.store.get(job_id)
        if job is None or job.get("deleted_at") is not None:
            continue
        job_dir = svc.job_dir(job_id)
        path = next(
            (job_dir / name for name in verdicts_mod.IMAGE_NAMES if (job_dir / name).is_file()),
            None,
        )
        if path is None:
            continue
        resolved[job_id] = (float(job["created_at"]), path)

    # Oldest first, so the dedupe pass below keeps the earlier of two
    # colliding rows and drops the later one.
    entries = sorted(
        ((job_id, created_at, path) for job_id, (created_at, path) in resolved.items()),
        key=lambda e: (e[1], e[0]),
    )

    dropped_duplicates = 0
    if dedupe:
        # A pure-stdlib-plus-numpy metric, not a bench pipeline: reusing the
        # near-duplicate hash bench/metrics.py already carries and measured is
        # the point, and this module owns none of the GPU/queue machinery
        # bench/__init__.py's "reaches the app the same way a user does" rule
        # is actually about.
        from ..bench import metrics as bench_metrics

        kept: list[tuple[str, float, Path]] = []
        kept_hashes: list[int] = []
        for job_id, created_at, path in entries:
            image_hash = bench_metrics.perceptual_hash(path)
            is_duplicate = image_hash is not None and any(
                bench_metrics.hash_similarity(image_hash, other) >= DUPLICATE_SIMILARITY
                for other in kept_hashes
            )
            if is_duplicate:
                dropped_duplicates += 1
                continue
            kept.append((job_id, created_at, path))
            if image_hash is not None:
                kept_hashes.append(image_hash)
        entries = kept

    # Newest first, then capped -- the library's own version of the ceiling
    # the folder-based door already enforces.
    entries.sort(key=lambda e: (e[1], e[0]), reverse=True)
    entries = entries[: lora_train.MAX_IMAGES]

    final_ids = {job_id for job_id, _created_at, _path in entries}
    if len(final_ids) < lora_train.MIN_IMAGES:
        raise Invalid(
            f"the library has {len(final_ids)} usable image(s) so far; a style needs at "
            f"least {lora_train.MIN_IMAGES} -- favourite more jobs, or accept more "
            "references or meshes",
            field="images",
        )

    return {
        "paths": [path for _job_id, _created_at, path in entries],
        "considered": len(considered),
        "dropped_duplicates": dropped_duplicates,
        "sources": {key: len(ids & final_ids) for key, ids in raw.items()},
    }


def train_lora(
    svc: WarlockService,
    images: Sequence[Path | str],
    *,
    label: str,
    trigger: str,
    steps: int = lora_train.DEFAULT_STEPS,
    base_model: str | None = None,
) -> dict[str, Any]:
    """Queue a training run. The images are copied into the job's directory
    at the door, so a folder the user edits later cannot change a row.

    The card is priced as exclusive (``vram.LORA_TRAIN_GIB``): the worker
    stops trellis and evicts the image pipe before it spawns the trainer.
    """
    text = (label or "").strip()
    if not text:
        raise Invalid("give the style a name", field="label")
    if len(text) > lora_train.MAX_LABEL:
        raise Invalid(f"a name is at most {lora_train.MAX_LABEL} characters", field="label")
    words = (trigger or "").strip()
    if not words:
        raise Invalid(
            "give the style trigger words -- the phrase that summons it in a prompt",
            field="trigger",
        )
    if len(words) > lora_train.MAX_TRIGGER:
        raise Invalid(
            f"trigger words are at most {lora_train.MAX_TRIGGER} characters", field="trigger"
        )
    try:
        count = lora_train.check_steps(steps)
    except ValueError as exc:
        raise Invalid(str(exc), field="steps") from exc
    paths = [Path(p) for p in images]
    if len(paths) < lora_train.MIN_IMAGES:
        raise Invalid(
            f"a style needs at least {lora_train.MIN_IMAGES} images", field="images"
        )
    if len(paths) > lora_train.MAX_IMAGES:
        raise Invalid(
            f"at most {lora_train.MAX_IMAGES} images per style", field="images"
        )
    from PIL import Image

    for path in paths:
        if path.suffix.lower() not in IMAGE_SUFFIXES or not path.is_file():
            raise Invalid(f"{path.name} is not an image file", field="images")
        try:
            with Image.open(path) as im:
                # Size before integrity: ``verify`` checks that the file is a
                # whole image, not that it is one this build will decode. These
                # are read straight from the user's folder by
                # ``lora_train_worker._load_images`` -- they never pass through
                # ``files.to_png``, which is where every *uploaded* image gets
                # this same ceiling -- so the door has to apply it here.
                width, height = im.width, im.height
                im.verify()
        except Exception as exc:
            raise Invalid(f"{path.name} could not be read: {exc}", field="images") from exc
        if width * height > MAX_IMAGE_PIXELS:
            raise TooLarge(
                f"{path.name} is {width}x{height}; the limit is "
                f"{MAX_IMAGE_PIXELS:,} pixels"
            )

    base_key = str(base_model or models.DEFAULT_BASE_MODEL)
    spec = models.BASE_MODELS.get(base_key)
    if spec is None:
        raise Invalid(f"unknown base model {base_key!r}", field="base_model")
    if spec.family != models.FAMILY_SDXL or base_key not in models.cfg_bases():
        # An SDXL checkpoint at full CFG: a distilled base (turbo, Hyper-SD)
        # shares the architecture but not the schedule the loss assumes, and
        # FLUX.2 is a different network altogether. ``cfg_bases`` is the one
        # spelling of "undistilled" the registry already has.
        usable = sorted(
            k for k in models.cfg_bases() if models.BASE_MODELS[k].family == models.FAMILY_SDXL
        )
        raise Invalid(
            f"a style is trained on an undistilled SDXL checkpoint; pick one of {usable}",
            field="base_model",
        )
    # The 2026-09-13 audit, finding service-01: this door checked weights and
    # never the pack, so a host with weights present but ``text2image``
    # removed by an upgrade queued the job and died in the worker on the SDXL
    # import instead of refusing here.
    check_pack(svc, "lora_train", {}, field="base_model")
    check_base_model_weights(svc, spec)

    params: dict[str, Any] = {
        "base_model": base_key,
        "label": text,
        "trigger": words,
        "steps": count,
        "images": len(paths),
    }
    check_vram(svc, "lora_train", "reference", params)

    new_id = uuid.uuid4().hex[:12]
    job_dir = svc.job_dir(new_id)
    train_dir = job_dir / "train"
    try:
        train_dir.mkdir(parents=True, exist_ok=True)
        for index, path in enumerate(paths):
            with Image.open(path) as im:
                im.convert("RGB").save(train_dir / f"{index:03d}.png")
        svc.store.create("lora_train", text, params, new_id)
    except Exception:
        # create_character's and create_tile_sheet's own shape (the
        # 2026-09-08 audit, service-06): this directory is minted before the
        # row, so only an insert (or copy) that never landed cleans up after
        # itself -- otherwise a failure partway through the copy (a bad
        # file, ENOSPC, a permissions error) leaves an orphaned job directory
        # with no owner, exactly what service.library.verify()'s
        # orphan_dirs finding exists to surface.
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    svc.wake_worker()
    return {"id": new_id, "images": len(paths)}
