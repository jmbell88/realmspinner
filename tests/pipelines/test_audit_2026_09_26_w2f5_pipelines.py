"""Regression tests for the 2026-09-26 audit's pipelines-layer findings closed
in pass w2f5: pipelines-children-01 (``RESUME_NAME``'s side effect on
import), pipelines-children-03 (``separation_worker``'s staged write) and
pipelines-image-01 (``reference.normalise``'s canvas margins).
"""

from __future__ import annotations

import os
import subprocess
import sys

import soundfile as sf
import torch
from PIL import Image, ImageDraw

from realmspinner.pipelines import reference, separation_worker

BG = (200, 200, 200)


def test_the_download_sweep_leaves_hf_hub_offline_at_one(tmp_path):
    """``service.downloads._is_resumable`` used to import ``RESUME_NAME``
    from ``pipelines.fetch_worker`` -- and merely *importing* that module ran
    its top-level ``os.environ["HF_HUB_OFFLINE"] = "0"`` in whichever process
    did the importing. ``_is_resumable`` runs in the *app* process, sweeping
    staging trees before every download starts, not in the spawned child
    whose job is to go online -- so one sweep left every subprocess the app
    spawned for the rest of its life with offline mode switched off, with no
    fetch ever having run.

    Run in a subprocess, like every other test in this area
    (``tests/test_fetch.py``, ``tests/test_fetch_verify.py``): importing
    ``fetch_worker`` is cached in ``sys.modules`` after the first test that
    does it anywhere in this session, which would hide the regression here.

    The child's environment is built explicitly, with ``HF_HUB_OFFLINE``
    stripped out of whatever this test's own process inherited, rather than
    left to inherit ``os.environ`` as-is. Found under ``-n 8 --dist
    loadfile``: this test passed alone but printed "0" in the full run --
    not because ``_is_resumable`` regressed, but because
    ``subprocess.run``'s default ``env=None`` inherits the *pytest worker's*
    ambient environment, and some earlier test in that same worker process
    (any of several here that import ``fetch_worker`` and restore the flag
    afterwards -- correctly, but only ever back to whatever the worker
    already had) can leave that ambient value already resolved before this
    test ever runs. A subprocess test that means to measure "does the child
    process end up offline" must control what the child starts with, the
    same way ``tests/test_offline.py`` clears the var before asserting
    ``realmspinner``'s own import sets it, rather than trust the parent's own
    environment to already be clean.
    """
    staging = tmp_path / "thing.fetch.part"
    staging.mkdir()
    (staging / ".realmspinner-resume.json").write_text("{}")

    env = dict(os.environ)
    env.pop("HF_HUB_OFFLINE", None)

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import realmspinner, os, sys; "
            "from pathlib import Path; "
            "from realmspinner.service import downloads; "
            f"downloads._is_resumable(Path({str(staging)!r})); "
            "sys.stdout.write(os.environ['HF_HUB_OFFLINE'])",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.stdout.strip() == "1", proc.stderr


def _opaque_rgba_subject(size=256, box=(64, 64, 191, 191)):
    """A reference with an alpha channel that carries no transparency at all
    -- every pixel (background and subject alike) is alpha 255. Mirrors
    ``tests/pipelines/test_reference.py``'s ``_subject(mode="RGBA")`` fixture,
    redefined here rather than imported so this file stays a self-contained
    regression record."""
    im = Image.new("RGBA", (size, size), BG + (255,))
    ImageDraw.Draw(im).rectangle(list(box), fill=(40, 90, 160, 255))
    return im


def test_normalise_of_an_opaque_rgba_reference_leaves_no_transparent_margin():
    """``normalise`` painted a fresh canvas's margin at alpha 0 whenever the
    source mode was RGBA, regardless of whether the channel ever carried real
    transparency. An opaque RGBA upload (every source pixel alpha 255) came
    back with the margins punched fully transparent -- min alpha 255 in, 0 in
    the margins out -- despite nothing in the source ever being see-through.
    The subject's own alpha must still round-trip untouched, and the mode
    must stay RGBA (``test_normalise_never_invents_or_strips_an_alpha_channel``
    already pins that half)."""
    canvas, report = reference.normalise(_opaque_rgba_subject())
    assert canvas.mode == "RGBA"
    assert canvas.getchannel("A").getextrema() == (255, 255)


def test_separation_stem_write_stages_without_losing_the_wav_format():
    """``sf.write`` with no ``format=`` infers the container from the
    filename's extension -- and the staging name ends in ``.tmp``, not
    ``.wav``, so every separation raised ``TypeError: No format specified``
    right after the model finished running, before the stem was ever renamed
    into place. ``write_stems`` is the extracted staging loop
    (``separation_worker.separate`` calls it too), pulled out so this is
    provable without loading hdemucs."""
    # Real files, in a real temp directory -- ``sf.write``'s format inference
    # is the thing under test, and a fake path object would just hide it.
    import tempfile
    from pathlib import Path

    out_dir = Path(tempfile.mkdtemp())
    stems = torch.zeros((2, 2, 10), dtype=torch.float32)
    written = separation_worker.write_stems(stems, ("vocals", "drums"), out_dir, 44100, sf=sf)

    assert written == ["vocals.wav", "drums.wav"]
    for name in written:
        path = out_dir / name
        assert path.is_file()
        assert not (out_dir / f".{name}.tmp").exists()
