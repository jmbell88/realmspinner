"""Structure hints for ControlNet: pure, deterministic, torch-free.

Same split as kernels/sheet.py -- everything decidable lives here and is
testable with a Pillow-drawn fixture and no GPU. cv2 is imported inside the
functions so this module stays importable without the text2image extra.

Canny goes through cv2 rather than Pillow's own edge filter on purpose: the
published controlnet-canny-sdxl-1.0 was trained on OpenCV Canny output, so
its model card's threshold advice only transfers if the hint comes from the
same operator.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core.safeio import atomic

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL import Image as _ImageModule

    PILImage = _ImageModule.Image
else:  # pragma: no cover - runtime alias
    PILImage = Any

DEFAULT_LOW = 100
DEFAULT_HIGH = 200

# Kinds control.hint() itself can produce. A depth hint would need a torch
# model and so would have to be a module of its own: this one must stay
# torch-free, which test_offline.py asserts exactly. No such module exists yet.
PREPROCESSORS = ("canny",)

# The range a "Start from this image" reference is fitted to, long side. The
# 2026-10-14 audit, finding pipelines-image-01: the frame used to be the
# reference's own size with no ceiling, so a phone photo made SDXL sample and
# VAE-decode at 4000x4000 with the checkpoint already resident and VRAM
# admission (which prices the spec's 1024 square) blind to it, and a 64x64
# reference sampled at 64x64, far below what SDXL draws coherently.
INIT_MAX_LONG_SIDE = 1024
INIT_MIN_LONG_SIDE = 512


def fit_init_frame(width: int, height: int) -> tuple[int, int]:
    """The ``(width, height)`` a ``width`` x ``height`` init picture is drawn at.

    Aspect is kept and the long side is clamped into
    [:data:`INIT_MIN_LONG_SIDE`, :data:`INIT_MAX_LONG_SIDE`]; both sides are
    snapped to the VAE's stride of 8 (floor 8). A picture already inside the
    range keeps its size, so Inker's ``send_size`` crops pass through. Pure, so
    the worker that draws the structure hint and the pipeline that denoises
    the picture agree on one frame (the 2026-10-14 audit, finding
    pipelines-image-03).
    """
    width, height = max(1, int(width)), max(1, int(height))
    long_side = max(width, height)
    scale = min(max(long_side, INIT_MIN_LONG_SIDE), INIT_MAX_LONG_SIDE) / long_side
    return (
        max(8, int(round(width * scale / 8)) * 8),
        max(8, int(round(height * scale / 8)) * 8),
    )


def _letterbox(image: PILImage, size: int) -> PILImage:
    """Fit ``image`` into a size x size RGB frame on black, preserving aspect.

    ControlNet wants the hint at the generation resolution exactly. Padding
    rather than cropping keeps the hint aligned with what the user supplied --
    a crop would silently promise structure that the reference never had.
    """
    from PIL import Image

    src = image.convert("RGB")
    scale = min(size / src.width, size / src.height)
    w = max(1, round(src.width * scale))
    h = max(1, round(src.height * scale))
    resized = src.resize((w, h), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(resized, ((size - w) // 2, (size - h) // 2))
    return canvas


def canny(
    image: PILImage,
    *,
    low: int = DEFAULT_LOW,
    high: int = DEFAULT_HIGH,
    size: int | tuple[int, int] | None = None,
) -> PILImage:
    """An RGB Canny edge map, optionally fitted to ``size``.

    An int ``size`` letterboxes to a ``size`` x ``size`` square, with the edges
    detected *before* padding so the black bars never register as a
    frame-shaped rectangle of structure the model then tries to honour.

    A ``(width, height)`` ``size`` is a frame the hint must equal exactly --
    the init frame, when a start picture and a ControlNet are combined. The
    picture is resized to it *first* and edges are drawn there, so the hint
    lines up with the picture being denoised: the 2026-10-14 audit, finding
    pipelines-image-03, found the square letterbox traced a distorted copy of
    a non-square reference, bars and all.
    """
    import cv2
    import numpy as np
    from PIL import Image

    rgb = image.convert("RGB")
    if isinstance(size, tuple):
        if rgb.size != size:
            rgb = rgb.resize(size, Image.LANCZOS)
        edges = cv2.Canny(np.array(rgb), low, high)
        return Image.fromarray(np.stack([edges] * 3, axis=-1), mode="RGB")
    edges = cv2.Canny(np.array(rgb), low, high)
    out = Image.fromarray(np.stack([edges] * 3, axis=-1), mode="RGB")
    return _letterbox(out, size) if size else out


def hint(
    kind: str,
    image: PILImage,
    *,
    size: int | tuple[int, int],
    low: int = DEFAULT_LOW,
    high: int = DEFAULT_HIGH,
) -> PILImage:
    """Dispatch on preprocessor name. Unknown (including "depth") raises."""
    if kind != "canny":
        raise ValueError(
            f"unsupported control preprocessor {kind!r}; "
            f"this module handles {', '.join(PREPROCESSORS)}"
        )
    return canny(image, low=low, high=high, size=size)


def edge_fraction(image: PILImage) -> float:
    """Share of pixels the detector marked as an edge.

    Recorded with every hint so "my silhouette lock did nothing" is an
    answerable question: a near-zero fraction means the thresholds found no
    structure, not that the ControlNet was ignored.
    """
    import numpy as np

    arr = np.asarray(image.convert("L"))
    if arr.size == 0:
        return 0.0
    return float((arr > 0).sum()) / float(arr.size)


def write_hint(
    src: Path,
    dest: Path,
    *,
    kind: str,
    size: int | tuple[int, int],
    low: int = DEFAULT_LOW,
    high: int = DEFAULT_HIGH,
) -> dict[str, Any]:
    """Preprocess ``src`` into ``dest`` and return the hint's provenance.

    Staged through a temp name and renamed, the same rule every other write
    onto a served path follows (optimize.staged_copy, store.finalize_rig):
    the file route may read this while a rerun is rewriting it.

    The ``finally`` is the other half of that rule, and it is the half this had
    missing: ``postprocess._staged``, ``trellis._atomic_write`` and
    ``optimize.run`` all unlink theirs. A raising ``save`` left the dotfile
    behind -- never a half-written served file, so this is consistency rather
    than correctness, but a stranded fragment is a fragment nothing will ever
    clean up.
    """
    from PIL import Image

    with Image.open(src) as im:
        im.load()
        out = hint(kind, im, size=size, low=low, high=high)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # The 2026-09-18 audit, finding pipelines-01: the staging name used to be
    # ``f".{dest.name}.tmp"``, fixed and shared by every concurrent caller of
    # this ``dest`` -- the same M03/service-02 hazard ``atomic.staged``
    # already closes elsewhere in this codebase.
    with atomic.staged(dest) as tmp:
        out.save(tmp, format="PNG")
    return {
        "kind": kind,
        "low": low,
        "high": high,
        # An int, whichever shape ``size`` took: the recorded block is read back
        # as JSON and means "the hint's long side", the same as hint_report's.
        "size": max(size) if isinstance(size, tuple) else size,
        "edge_fraction": edge_fraction(out),
    }


def hint_report(path: Path) -> dict[str, Any]:
    """The provenance block for a hint some other code already drew.

    Same shape :func:`write_hint` returns, so a job's ``control_hint`` means
    one thing whoever produced the image, and with ``edge_fraction`` measured
    over the file that will actually condition the generation -- which is the
    only reading that answers "did my guide draw anything".

    ``kind`` says ``"guide"`` rather than a preprocessor name because no
    preprocessor ran: the caller handed the ControlNet line art directly.
    """
    from PIL import Image

    with Image.open(path) as im:
        im.load()
        return {
            "kind": "guide",
            "size": max(im.size),
            "edge_fraction": edge_fraction(im),
        }
