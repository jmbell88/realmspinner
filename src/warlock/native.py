"""The optional native kernel library, and the rule for using one.

``vendor/warlockc/warlockc.dll`` is built from ``native/*.c`` by
``native/build.ps1``. It is vendored the way ``trellis-server.exe`` and
``gltfpack.exe`` are -- a local build artifact under a gitignored directory,
never downloaded -- which means the honest default is that it is *absent*, and
every caller has to work without it.

So this module is pure in the way :mod:`~warlock.vram` and :mod:`~warlock.memlog`
are pure: stdlib and numpy only, no imports from ``service``, ``queue`` or
``studio`` (and no PIL -- the RotSprite wrapper transcribes the six lines of
``Image.rotate`` it needs rather than importing them), and a missing or
unusable DLL is ``None`` rather than an exception. The call
sites read::

    if native.available():
        ...kernel...
    else:
        ...the numpy implementation...

and the numpy implementation is never deleted -- it is both the fallback and
the reference the parity tests measure the kernel against.

**The ABI check is the load-bearing part.** ``vendor/`` is gitignored, so a
working tree routinely holds a DLL built from older sources beside newer
Python. Without a version handshake that DLL would keep computing the previous
behaviour silently, which is the one failure mode a fallback path must not
have: an absent DLL is obvious, a stale one is not. ``ABI`` here must equal
``WARLOCKC_ABI`` in ``native/warlockc.h``; a mismatch is reported once and
treated as absent.

Two environment variables, both for situations that already exist elsewhere in
this project: ``WARLOCK_NATIVE=0`` forces the fallback (A/B timing, and CI on a
machine with no compiler), and ``WARLOCK_NATIVE_DLL`` relocates the file the
way ``WARLOCK_GLTFPACK`` relocates gltfpack -- a git worktree has no
``vendor/`` at all, which is why the other ``WARLOCK_*`` path overrides exist.
"""

from __future__ import annotations

import ctypes
import logging
import math
import os
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# Must match WARLOCKC_ABI in native/warlockc.h.
ABI = 11

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DLL = _PROJECT_ROOT / "vendor" / "warlockc" / "warlockc.dll"

# The probe result, cached: ``...`` means "not tried yet", None means
# unavailable. Probing is a file check plus a load, and callers ask per job.
_lib: Any = ...


def dll_path() -> Path:
    """Where the library is expected, honouring the override."""
    override = os.environ.get("WARLOCK_NATIVE_DLL")
    return Path(override) if override else _DEFAULT_DLL


def _enabled() -> bool:
    return os.environ.get("WARLOCK_NATIVE", "1").strip().lower() not in {
        "0",
        "off",
        "false",
        "no",
    }


def _bind(lib: ctypes.CDLL) -> None:
    """Declare every prototype. ctypes defaults to int-returning and
    int-sized arguments, which silently truncates a 64-bit pointer."""
    d = ctypes.POINTER(ctypes.c_double)
    f = ctypes.POINTER(ctypes.c_float)
    i64 = ctypes.c_int64

    lib.warlockc_abi.restype = ctypes.c_int32
    lib.warlockc_abi.argtypes = []

    lib.warlockc_rasterise.restype = None
    lib.warlockc_rasterise.argtypes = [
        d, d, d, d, d, d, d,  # ax ay bx by cx cy area2
        ctypes.c_int64,  # n
        ctypes.c_int32,  # resolution
        ctypes.POINTER(ctypes.c_uint8),  # covered
    ]

    lib.warlockc_over_f32.restype = None
    lib.warlockc_over_f32.argtypes = [
        f, i64,  # backdrop, row stride in floats
        f, i64,  # source
        f, i64,  # out
        i64, i64,  # h, w
        ctypes.c_float,  # opacity
        ctypes.c_int32,  # mode
    ]

    lib.warlockc_paint_colour_f32.restype = None
    lib.warlockc_paint_colour_f32.argtypes = [
        f, i64,  # before
        f, i64,  # weight, one channel
        f, i64,  # out
        i64, i64,  # h, w
        ctypes.c_float * 4,  # colour, 0..255
    ]

    lib.warlockc_stack_f32.restype = None
    lib.warlockc_stack_f32.argtypes = [
        ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),  # layer crops
        ctypes.POINTER(i64),  # their row strides, in bytes
        f,  # opacities
        ctypes.POINTER(ctypes.c_int32),  # modes
        i64,  # n
        f, i64,  # out
        i64, i64,  # h, w
        f, i64,  # base, or NULL
    ]

    lib.warlockc_to_uint8_f32.restype = None
    lib.warlockc_to_uint8_f32.argtypes = [
        f,  # pixels
        ctypes.POINTER(ctypes.c_uint8),  # out
        i64,  # count
    ]

    lib.warlockc_to_uint8_255_f32.restype = None
    lib.warlockc_to_uint8_255_f32.argtypes = [
        f,  # pixels, already 0..255
        ctypes.POINTER(ctypes.c_uint8),  # out
        i64,  # count
    ]

    u8 = ctypes.POINTER(ctypes.c_uint8)
    i32 = ctypes.POINTER(ctypes.c_int32)

    lib.warlockc_morph_u8.restype = None
    lib.warlockc_morph_u8.argtypes = [
        u8, i64,  # src, row stride in bytes
        u8,  # scratch, h * w bytes
        u8, i64,  # out, row stride in bytes
        i64, i64,  # h, w
        i64,  # radius
        ctypes.c_int32,  # 0 max (grow), 1 min (shrink)
    ]

    lib.warlockc_dither_fs.restype = None
    lib.warlockc_dither_fs.argtypes = [
        f, i64,  # work, (h, w, 3) float32, row stride in floats
        u8, i64,  # visible, (h, w) uint8, row stride in bytes
        f, i64,  # entries, (n, 3) float32
        i64, i64,  # h, w
    ]

    lib.warlockc_palette_nearest_i32.restype = None
    lib.warlockc_palette_nearest_i32.argtypes = [
        i32, i32,  # queries (n, 3) int32; palette (p, 3) int32
        i32,  # out, n int32
        i64, i64,  # n, p
    ]

    lib.warlockc_flood_u8.restype = None
    lib.warlockc_flood_u8.argtypes = [
        u8, i64,  # match, (h, w) uint8, row stride in bytes
        u8, i64,  # out, (h, w) uint8, row stride in bytes
        i32,  # scratch, h*w int32 of queue storage
        i64, i64,  # h, w
        i64, i64,  # seed x, seed y
    ]

    # ``i64`` above is the *scalar* c_int64, which is what every count and
    # stride in this header is. These two are the pointer spellings, kept
    # distinct rather than reusing the name, because ctypes will happily accept
    # a pointer where a scalar is declared and then read the wrong eight bytes.
    d = ctypes.POINTER(ctypes.c_double)
    pi64 = ctypes.POINTER(ctypes.c_int64)

    lib.warlockc_bvh_build.restype = i64
    lib.warlockc_bvh_build.argtypes = [
        d, d, d,  # tri_lo, tri_hi, centroid -- each (n, 3) float64
        i64,  # n_tris
        i64,  # leaf_size
        pi64,  # order, n int64, written by the kernel
        d, d,  # node lo, hi -- each (max_nodes, 3) float64
        pi64, pi64, pi64, pi64,  # left, right, first, count
        i64,  # max_nodes
        pi64,  # stack, 4 * max_nodes int64
    ]

    lib.warlockc_blit_cells_u8.restype = None
    lib.warlockc_blit_cells_u8.argtypes = [
        u8, i64, i64, i64,  # out, h, w, row stride in bytes
        u8, i64, i64,  # atlas, tile h, tile w
        i32,  # tile index per cell
        pi64, pi64,  # x, y per cell
        i64,  # n cells
    ]

    lib.warlockc_palette_nearest_f64.restype = None
    lib.warlockc_palette_nearest_f64.argtypes = [
        d, d,  # queries (n, 3) float64; palette (p, 3) float64
        i32,  # out, n int32
        i64, i64,  # n, p
    ]

    lib.warlockc_contours.restype = i64
    lib.warlockc_contours.argtypes = [
        u8, i64,  # mask, row stride in bytes
        i64, i64,  # h, w
        ctypes.c_uint8,  # threshold
        u8,  # scratch, one zeroed byte per lattice edge
        i32, i64,  # points, capacity in vertices
        i32, i64,  # loop lengths, capacity in loops
    ]

    lib.warlockc_rotsprite_u8.restype = ctypes.c_int32
    lib.warlockc_rotsprite_u8.argtypes = [
        u8,  # src, h * w * channels bytes, tightly packed
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,  # h, w, channels
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double,  # a0..a5
        ctypes.c_int32, ctypes.c_int32,  # out_h, out_w
        u8, ctypes.c_size_t,  # scratch, scratch_len
        u8,  # out
    ]

    lib.warlockc_smoke_blob.restype = ctypes.c_int32
    lib.warlockc_smoke_blob.argtypes = [
        f, f,  # out_rgb, out_a -- full-frame, contiguous
        ctypes.c_int32, ctypes.c_int32,  # frame_w, frame_h
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,  # y0 y1 x0 x1
        ctypes.c_float,  # scale
        ctypes.c_double, ctypes.c_double,  # px, py
        ctypes.c_double, ctypes.c_double,  # radius, rag
        ctypes.c_int64,  # fbm_seed
        ctypes.c_float, ctypes.c_float, ctypes.c_float,  # dx_f32, dy_f32, nscale_f32
        ctypes.c_float,  # alpha_mul
        ctypes.c_float, ctypes.c_float, ctypes.c_float,  # r, g, b
        f, ctypes.c_size_t,  # scratch, scratch_len (floats)
    ]


def _load() -> Any:
    if not _enabled():
        return None
    path = dll_path()
    if not path.exists():
        return None
    try:
        lib = ctypes.CDLL(str(path))
        _bind(lib)
        found = int(lib.warlockc_abi())
    except (OSError, AttributeError, ValueError) as exc:
        # A DLL built for the wrong architecture, a partial build missing an
        # export, or a file that is not a library at all. None of those should
        # stop the app: they mean "use numpy".
        log.warning("warlockc at %s is unusable (%s); using numpy fallbacks", path, exc)
        return None
    if found != ABI:
        log.warning(
            "warlockc at %s is ABI %d, this build expects %d -- rebuild with "
            "native/build.ps1; using numpy fallbacks",
            path,
            found,
            ABI,
        )
        return None
    return lib


def lib() -> Any:
    """The loaded library, or None. Probed once per process."""
    global _lib
    if _lib is ...:
        _lib = _load()
    return _lib


def available() -> bool:
    return lib() is not None


def reset() -> None:
    """Forget the cached probe.

    For tests that flip ``WARLOCK_NATIVE`` -- the whole point of the env var is
    running the same suite both ways in one process."""
    global _lib
    _lib = ...


def status() -> tuple[bool, str]:
    """(ok, detail) for the doctor row. Never raises, never loads twice."""
    path = dll_path()
    if not _enabled():
        return False, "disabled by WARLOCK_NATIVE=0 -- numpy fallbacks in use"
    if available():
        return True, f"{path} (ABI {ABI})"
    if not path.exists():
        return False, (
            f"not built at {path} -- run native\\build.ps1 "
            "(optional; numpy fallbacks in use)"
        )
    return False, f"{path} is unusable or ABI-mismatched -- rebuild with native\\build.ps1"


# --- typed views onto the kernels -------------------------------------------
#
# Callers pass numpy arrays; the conversion to pointers lives here so no call
# site has to know ctypes. Contiguity and dtype are the caller's to guarantee
# (they are cheap asserts there and would be a per-call copy here).


def _ptr(array: Any, ctype: Any) -> Any:
    return array.ctypes.data_as(ctypes.POINTER(ctype))


def rasterise(
    ax: Any, ay: Any, bx: Any, by: Any, cx: Any, cy: Any, area2: Any, covered: Any
) -> None:
    """Fill ``covered`` (uint8, square, C-contiguous) from projected triangles.

    Every array must be float64, C-contiguous and the same length; ``covered``
    is written in place and never cleared, matching the numpy path.
    """
    handle = lib()
    if handle is None:  # pragma: no cover - callers check available() first
        raise RuntimeError("warlockc is not loaded")
    c_double = ctypes.c_double
    handle.warlockc_rasterise(
        _ptr(ax, c_double),
        _ptr(ay, c_double),
        _ptr(bx, c_double),
        _ptr(by, c_double),
        _ptr(cx, c_double),
        _ptr(cy, c_double),
        _ptr(area2, c_double),
        ctypes.c_int64(len(ax)),
        ctypes.c_int32(covered.shape[0]),
        _ptr(covered, ctypes.c_uint8),
    )


def over_f32(
    backdrop: Any,
    backdrop_stride: int,
    source: Any,
    source_stride: int,
    out: Any,
    out_stride: int,
    height: int,
    width: int,
    opacity: float,
    mode: int,
) -> None:
    """Composite ``source`` onto ``backdrop`` into ``out``.

    Every array is float32 with four channels last and rows ``*_stride`` floats
    apart; ``mode`` indexes ``composite.BLEND_MODES``. The caller has already
    established all of that -- see ``composite._over_native``, which is the
    only one.
    """
    handle = lib()
    if handle is None:  # pragma: no cover - callers check available() first
        raise RuntimeError("warlockc is not loaded")
    c_float = ctypes.c_float
    handle.warlockc_over_f32(
        _ptr(backdrop, c_float),
        ctypes.c_int64(backdrop_stride),
        _ptr(source, c_float),
        ctypes.c_int64(source_stride),
        _ptr(out, c_float),
        ctypes.c_int64(out_stride),
        ctypes.c_int64(height),
        ctypes.c_int64(width),
        ctypes.c_float(opacity),
        ctypes.c_int32(mode),
    )


def paint_colour_f32(
    before: Any,
    before_stride: int,
    weight: Any,
    weight_stride: int,
    out: Any,
    out_stride: int,
    height: int,
    width: int,
    colour: tuple[int, int, int, int],
) -> None:
    """Write ``colour`` over ``before`` at ``weight`` into ``out``, 0..255."""
    handle = lib()
    if handle is None:  # pragma: no cover - callers check available() first
        raise RuntimeError("warlockc is not loaded")
    c_float = ctypes.c_float
    handle.warlockc_paint_colour_f32(
        _ptr(before, c_float),
        ctypes.c_int64(before_stride),
        _ptr(weight, c_float),
        ctypes.c_int64(weight_stride),
        _ptr(out, c_float),
        ctypes.c_int64(out_stride),
        ctypes.c_int64(height),
        ctypes.c_int64(width),
        (c_float * 4)(*(float(v) for v in colour)),
    )


def stack_f32(
    crops: list[Any],
    strides: list[int],
    opacities: list[float],
    modes: list[int],
    out: Any,
    out_stride: int,
    height: int,
    width: int,
    base: Any | None,
    base_stride: int,
) -> None:
    """Fold ``crops`` bottom-first onto ``base`` (or transparent black).

    ``crops`` are uint8 (h, w, 4) *views* into the layers' full canvases and the
    caller has to hold them alive across this call -- the pointer array below
    is built from their data addresses and ctypes keeps no reference to the
    arrays themselves.
    """
    handle = lib()
    if handle is None:  # pragma: no cover - callers check available() first
        raise RuntimeError("warlockc is not loaded")
    c_float = ctypes.c_float
    u8_ptr = ctypes.POINTER(ctypes.c_uint8)
    count = len(crops)
    handle.warlockc_stack_f32(
        (u8_ptr * count)(*(_ptr(crop, ctypes.c_uint8) for crop in crops)),
        (ctypes.c_int64 * count)(*strides),
        (c_float * count)(*opacities),
        (ctypes.c_int32 * count)(*modes),
        ctypes.c_int64(count),
        _ptr(out, c_float),
        ctypes.c_int64(out_stride),
        ctypes.c_int64(height),
        ctypes.c_int64(width),
        _ptr(base, c_float) if base is not None else None,
        ctypes.c_int64(base_stride),
    )


def to_uint8_f32(pixels: Any, out: Any, count: int) -> None:
    """Scale, round, clamp and narrow ``count`` floats into ``out``.

    Both arrays are C-contiguous and the shape is the caller's business -- this
    one is elementwise, so it sees a flat run.
    """
    handle = lib()
    if handle is None:  # pragma: no cover - callers check available() first
        raise RuntimeError("warlockc is not loaded")
    handle.warlockc_to_uint8_f32(
        _ptr(pixels, ctypes.c_float),
        _ptr(out, ctypes.c_uint8),
        ctypes.c_int64(count),
    )


def to_uint8_255_f32(pixels: Any, out: Any, count: int) -> None:
    """Round, clamp and narrow ``count`` floats already in 0..255 into ``out``.

    The unscaled sibling of :func:`to_uint8_f32`, for the call sites that do
    their arithmetic in levels rather than in fractions.
    """
    handle = lib()
    if handle is None:  # pragma: no cover - callers check available() first
        raise RuntimeError("warlockc is not loaded")
    handle.warlockc_to_uint8_255_f32(
        _ptr(pixels, ctypes.c_float),
        _ptr(out, ctypes.c_uint8),
        ctypes.c_int64(count),
    )


def morph_u8(src: Any, scratch: Any, out: Any, radius: int, op: int) -> None:
    """Grow (``op`` 0) or shrink (``op`` 1) a uint8 mask by ``radius``.

    ``src`` and ``out`` are 2-D C-contiguous uint8 of the same shape and must
    not alias; ``scratch`` is ``h * w`` bytes. ``radius`` must be at least 1 --
    the caller answers zero itself, since the reference's answer there is a
    plain copy.
    """
    handle = lib()
    if handle is None:  # pragma: no cover - callers check available() first
        raise RuntimeError("warlockc is not loaded")
    height, width = src.shape
    handle.warlockc_morph_u8(
        _ptr(src, ctypes.c_uint8),
        ctypes.c_int64(src.strides[0]),
        _ptr(scratch, ctypes.c_uint8),
        _ptr(out, ctypes.c_uint8),
        ctypes.c_int64(out.strides[0]),
        ctypes.c_int64(height),
        ctypes.c_int64(width),
        ctypes.c_int64(radius),
        ctypes.c_int32(op),
    )


def palette_nearest(queries: Any, palette: Any, out: Any) -> None:
    """Write into ``out`` the nearest ``palette`` row for each ``queries`` row.

    ``queries`` is (n, 3) **int32** and ``palette`` (p, 3) int32, both
    C-contiguous; ``out`` is (n,) int32. Int32 queries rather than uint8 is a
    correctness requirement, not a convenience: ``dither._ordered``'s second
    search is over the reflection ``2c - p1``, which spans -255..510, and a u8
    signature would silently clamp exactly the search that exists to look at the
    far side.

    Ties go to the lowest palette index -- numpy ``argmin``'s first-minimum
    rule, so duplicate entries behave identically through either path.
    """
    handle = lib()
    if handle is None:  # pragma: no cover - callers check available() first
        raise RuntimeError("warlockc is not loaded")
    handle.warlockc_palette_nearest_i32(
        _ptr(queries, ctypes.c_int32),
        _ptr(palette, ctypes.c_int32),
        _ptr(out, ctypes.c_int32),
        ctypes.c_int64(queries.shape[0]),
        ctypes.c_int64(palette.shape[0]),
    )


def flood(match: Any, out: Any, scratch: Any, seed: tuple[int, int]) -> None:
    """Four-connected flood from ``seed`` through ``match``, writing ``out``.

    ``match`` and ``out`` are (h, w) uint8 and C-contiguous; ``scratch`` is a
    flat int32 array of at least ``h * w`` entries, which is exactly enough
    because a cell is marked before it is pushed and so is pushed once.
    ``out`` is cleared by the kernel. A seed outside the array or on a
    non-matching cell reaches nothing.

    ``h * w`` must be under 2**31: the kernel's queue holds a flat cell index
    in each ``int32_t`` slot (see the ceiling note beside
    ``warlockc_flood_u8`` in ``warlockc.h``), and an array at or beyond that
    size would overflow the index rather than merely running slowly. No
    caller is near this today -- a Plotter map is orders of magnitude smaller
    -- so this is a documented ceiling, not a reachable path; the assertion
    exists so a future caller with a much larger grid fails loudly here
    instead of silently corrupting the reached set in C. Callers should keep
    this bound alongside ``available()`` in their own gate so an oversized
    grid takes the numpy fallback rather than raising.
    """
    handle = lib()
    if handle is None:  # pragma: no cover - callers check available() first
        raise RuntimeError("warlockc is not loaded")
    height, width = match.shape
    assert height * width < 2**31, "flood: h * w must be < 2**31 (int32 queue index)"
    handle.warlockc_flood_u8(
        _ptr(match, ctypes.c_uint8),
        ctypes.c_int64(match.strides[0]),
        _ptr(out, ctypes.c_uint8),
        ctypes.c_int64(out.strides[0]),
        _ptr(scratch, ctypes.c_int32),
        ctypes.c_int64(height),
        ctypes.c_int64(width),
        ctypes.c_int64(int(seed[0])),
        ctypes.c_int64(int(seed[1])),
    )


def dither_fs(work: Any, visible: Any, entries: Any) -> None:
    """Diffuse ``work`` onto ``entries`` in place, serpentine Floyd-Steinberg.

    ``work`` is (h, w, 3) float32 in the 0..255 domain and is both input and
    output; ``visible`` is the (h, w) bool/uint8 alpha mask; ``entries`` is
    (n, 3) float32 and C-contiguous. The caller does the final clamp and
    narrowing, which is vectorised and not worth crossing the seam for.
    """
    handle = lib()
    if handle is None:  # pragma: no cover - callers check available() first
        raise RuntimeError("warlockc is not loaded")
    height, width, _ = work.shape
    handle.warlockc_dither_fs(
        _ptr(work, ctypes.c_float),
        ctypes.c_int64(work.strides[0] // work.itemsize),
        _ptr(visible, ctypes.c_uint8),
        ctypes.c_int64(visible.strides[0]),
        _ptr(entries, ctypes.c_float),
        ctypes.c_int64(entries.shape[0]),
        ctypes.c_int64(height),
        ctypes.c_int64(width),
    )


def contours(
    mask: Any,
    threshold: int,
    scratch: Any,
    points: Any,
    loop_lens: Any,
) -> int:
    """Trace the closed boundary loops of ``mask >= threshold``.

    ``mask`` is a C-contiguous 2-D uint8 plane; ``scratch`` is a zeroed uint8
    buffer of ``w * (h + 1) + (w + 1) * h`` bytes; ``points`` is int32 with room
    for two values per vertex and ``loop_lens`` int32 with room for one per
    loop. Returns the number of loops, or -1 if either buffer was too small --
    the caller falls back to numpy rather than guessing a bigger one.
    """
    handle = lib()
    if handle is None:  # pragma: no cover - callers check available() first
        raise RuntimeError("warlockc is not loaded")
    height, width = mask.shape
    return int(
        handle.warlockc_contours(
            _ptr(mask, ctypes.c_uint8),
            ctypes.c_int64(mask.strides[0]),
            ctypes.c_int64(height),
            ctypes.c_int64(width),
            ctypes.c_uint8(threshold),
            _ptr(scratch, ctypes.c_uint8),
            _ptr(points, ctypes.c_int32),
            ctypes.c_int64(points.size // 2),
            _ptr(loop_lens, ctypes.c_int32),
            ctypes.c_int64(loop_lens.size),
        )
    )


def _pillow_rotate_matrix(
    w: int, h: int, degrees: float
) -> tuple[tuple[float, float, float, float, float, float], int, int]:
    """The six affine coefficients and expanded size Pillow 12.3.0's
    ``Image.rotate(degrees, expand=True)`` computes for a ``w`` by ``h``
    image, transcribed operand for operand from ``PIL/Image.py`` (the
    ``center is None`` / ``translate is None`` branch, which is the only one
    :func:`rotsprite_u8`'s caller ever uses) rather than re-derived -- see the
    comment beside ``warlockc_rotsprite_u8`` in ``native/warlockc.h`` for why
    that matters. This needs no Pillow import: it is pure trigonometry, the
    same six lines ``Image.rotate`` runs before handing the matrix to C.
    """
    angle = -math.radians(degrees % 360.0)
    cx, cy = w / 2.0, h / 2.0
    matrix = [
        round(math.cos(angle), 15),
        round(math.sin(angle), 15),
        0.0,
        round(-math.sin(angle), 15),
        round(math.cos(angle), 15),
        0.0,
    ]

    def transform(x: float, y: float, m: list[float]) -> tuple[float, float]:
        a, b, c, d, e, f = m
        return a * x + b * y + c, d * x + e * y + f

    matrix[2], matrix[5] = transform(-cx, -cy, matrix)
    matrix[2] += cx
    matrix[5] += cy

    xs = []
    ys = []
    for x, y in ((0, 0), (w, 0), (w, h), (0, h)):
        tx, ty = transform(x, y, matrix)
        xs.append(tx)
        ys.append(ty)
    nw = math.ceil(max(xs)) - math.floor(min(xs))
    nh = math.ceil(max(ys)) - math.floor(min(ys))
    matrix[2], matrix[5] = transform(-(nw - w) / 2.0, -(nh - h) / 2.0, matrix)
    return (matrix[0], matrix[1], matrix[2], matrix[3], matrix[4], matrix[5]), nw, nh


# RotSprite's scratch, kept across calls rather than allocated fresh every
# mouse-move. A fresh ``np.empty`` of this size is not free even though numpy
# never zeroes it: Windows still has to page-fault the whole thing in on first
# touch, and a free-transform drag calls this once per move -- one page-fault
# storm every ~16 ms. Grown to the largest request seen and never shrunk;
# ``rotsprite_fits``/``ROTSPRITE_MAX_PIXELS`` bound the worst case at
# 80 * 512 * 512 * 4 = 83,886,080 bytes (~80 MiB), so this never grows without
# limit. Guarded by a lock because the buffer is shared process-wide and nothing
# here stops two callers overlapping -- held for the whole kernel call, not just
# the resize, since the kernel writes into this exact array by pointer.
_rotsprite_scratch: Any = None
_rotsprite_scratch_lock = threading.Lock()


def rotsprite_u8(pixels: Any, degrees: float) -> Any | None:
    """RotSprite's three EPX rounds and Pillow-exact nearest rotation, fused.

    ``pixels`` is (h, w) uint8 (a selection mask) or (h, w, 4) uint8 (RGBA),
    need not be contiguous -- this copies. Returns the same array
    ``transform.rotsprite``'s numpy path produces before any
    ``expand=False`` cropping (the ``[4::8, 4::8]`` centre-of-block
    downsample of the rotated 8x plane), or None if the scratch buffer this
    call needs was refused -- never reached at the sizes
    ``transform.rotsprite_fits`` allows, so this is a fall-back seam and not
    a live path, exactly like :func:`contours`.

    Angles that are a multiple of 90 are Pillow fast paths (plain
    transposes) and are the caller's job to route around this kernel
    entirely -- see ``transform.rotsprite``.
    """
    handle = lib()
    if handle is None:  # pragma: no cover - callers check available() first
        raise RuntimeError("warlockc is not loaded")
    pixels = np.ascontiguousarray(pixels, dtype=np.uint8)
    h, w = int(pixels.shape[0]), int(pixels.shape[1])
    channels = 1 if pixels.ndim == 2 else int(pixels.shape[2])
    if channels not in (1, 4) or h == 0 or w == 0:
        # The kernel dispatches on "4 or not 4"; an RGB plane would be walked
        # as bytes and come back scrambled rather than refused. The numpy path
        # handles any width, so decline here and let it.
        return None
    (a0, a1, a2, a3, a4, a5), nw, nh = _pillow_rotate_matrix(8 * w, 8 * h, degrees)
    out_h = len(range(4, nh, 8))
    out_w = len(range(4, nw, 8))
    out_shape = (out_h, out_w) if channels == 1 else (out_h, out_w, channels)
    out = np.empty(out_shape, dtype=np.uint8)
    needed = 80 * h * w * channels
    global _rotsprite_scratch
    with _rotsprite_scratch_lock:
        if _rotsprite_scratch is None or _rotsprite_scratch.size < needed:
            _rotsprite_scratch = np.empty(needed, dtype=np.uint8)
        scratch = _rotsprite_scratch
        result = handle.warlockc_rotsprite_u8(
            _ptr(pixels, ctypes.c_uint8),
            ctypes.c_int32(h),
            ctypes.c_int32(w),
            ctypes.c_int32(channels),
            ctypes.c_double(a0),
            ctypes.c_double(a1),
            ctypes.c_double(a2),
            ctypes.c_double(a3),
            ctypes.c_double(a4),
            ctypes.c_double(a5),
            ctypes.c_int32(out_h),
            ctypes.c_int32(out_w),
            _ptr(scratch, ctypes.c_uint8),
            ctypes.c_size_t(scratch.nbytes),
            _ptr(out, ctypes.c_uint8),
        )
    if result != 0:
        return None
    return out


# Smoke's fbm scratch, kept across calls the same way ``_rotsprite_scratch``
# is: a smoke layer's per-blob loop calls this once per live particle (up to
# 80 in the batch 11 gate case), and a fresh ``np.empty`` big enough for the
# largest blob's window would page-fault on every one of them. Grown to the
# largest request seen and never shrunk; guarded by a lock for the same
# reason ``_rotsprite_scratch_lock`` is -- the kernel writes into this exact
# array by pointer for the whole call, not just while it is being resized.
_smoke_scratch: Any = None
_smoke_scratch_lock = threading.Lock()


def smoke_blob(
    out_rgb: Any,
    out_a: Any,
    frame_w: int,
    frame_h: int,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    scale: float,
    px: float,
    py: float,
    radius: float,
    rag: float,
    fbm_seed: int,
    dx_f32: float,
    dy_f32: float,
    nscale_f32: float,
    alpha_mul: float,
    r_rgb: float,
    g_rgb: float,
    b_rgb: float,
) -> bool:
    """One Flourish smoke blob's per-pixel work, fused: distance plane,
    optional fbm raggedness blend, coverage, and ``over_into`` onto
    ``out_rgb``/``out_a`` in place at ``[y0:y1, x0:x1]``.

    ``out_rgb`` is ``(frame_h, frame_w, 3)`` float32 and ``out_a`` is
    ``(frame_h, frame_w)`` float32, both C-contiguous -- the accumulator
    planes ``smoke.render`` builds once per layer. ``px``/``py``/``radius``/
    ``rag`` are the blob's own float64 values exactly as numpy computes them
    (see ``smoke.c``'s precision note on why that matters for bit-parity);
    ``dx_f32``/``dy_f32``/``nscale_f32``/``alpha_mul``/``r_rgb``/``g_rgb``/
    ``b_rgb`` are the float32-rounded scalars the caller already derives the
    same way the numpy path does.

    Returns ``False`` when the fbm scratch this call needed was refused --
    reachable in principle the way :func:`rotsprite_u8`'s ``None`` is, never
    at the window sizes a Flourish recipe produces -- in which case the
    caller falls back to the numpy body for that one blob.
    """
    handle = lib()
    if handle is None:  # pragma: no cover - callers check available() first
        raise RuntimeError("warlockc is not loaded")
    win_h, win_w = y1 - y0, x1 - x0
    needed = 0
    if rag > 0.0 and win_h > 0 and win_w > 0:
        s = int(scale)
        r = s // 2
        w_coarse_full = frame_w // s
        h_coarse_full = frame_h // s
        cy0, cy1 = y0 // s, -(-y1 // s)
        cx0, cx1 = x0 // s, -(-x1 // s)
        ch, cw = cy1 - cy0, cx1 - cx0
        big_h, big_w = ch * s, cw * s
        max_dim = max(big_h, big_w)
        needed = ch * cw + big_h * big_w + (max_dim + 2 * r) + (max_dim + 2 * r + 1) + big_h
        del w_coarse_full, h_coarse_full  # not needed on the Python side, kept for clarity
    global _smoke_scratch
    with _smoke_scratch_lock:
        if _smoke_scratch is None or _smoke_scratch.size < needed:
            _smoke_scratch = np.empty(max(needed, 1), dtype=np.float32)
        scratch = _smoke_scratch
        result = handle.warlockc_smoke_blob(
            _ptr(out_rgb, ctypes.c_float),
            _ptr(out_a, ctypes.c_float),
            ctypes.c_int32(frame_w),
            ctypes.c_int32(frame_h),
            ctypes.c_int32(y0),
            ctypes.c_int32(y1),
            ctypes.c_int32(x0),
            ctypes.c_int32(x1),
            ctypes.c_float(scale),
            ctypes.c_double(px),
            ctypes.c_double(py),
            ctypes.c_double(radius),
            ctypes.c_double(rag),
            ctypes.c_int64(fbm_seed),
            ctypes.c_float(dx_f32),
            ctypes.c_float(dy_f32),
            ctypes.c_float(nscale_f32),
            ctypes.c_float(alpha_mul),
            ctypes.c_float(r_rgb),
            ctypes.c_float(g_rgb),
            ctypes.c_float(b_rgb),
            _ptr(scratch, ctypes.c_float),
            ctypes.c_size_t(scratch.size),
        )
    return result == 0


def bvh_build(
    tri_lo: Any,
    tri_hi: Any,
    centroid: Any,
    leaf_size: int,
    order: Any,
    lo: Any,
    hi: Any,
    left: Any,
    right: Any,
    first: Any,
    count: Any,
    stack: Any,
) -> int:
    """Build a median-split BVH, returning the node count or -1.

    ``tri_lo``, ``tri_hi`` and ``centroid`` are C-contiguous ``(n, 3)`` float64;
    ``order`` is ``n`` int64 and is filled by the kernel. The node arrays are
    ``(max_nodes, 3)`` float64 for ``lo``/``hi`` and ``max_nodes`` int64 for the
    rest, all sized by the caller; ``stack`` is ``4 * max_nodes`` int64.

    -1 means the node arrays were too small, and nothing in them is meaningful
    -- the caller falls back to numpy rather than guessing a bigger size, which
    is the same contract :func:`contours` has.
    """
    handle = lib()
    if handle is None:  # pragma: no cover - callers check available() first
        raise RuntimeError("warlockc is not loaded")
    c_double = ctypes.c_double
    c_int64 = ctypes.c_int64
    return int(
        handle.warlockc_bvh_build(
            _ptr(tri_lo, c_double),
            _ptr(tri_hi, c_double),
            _ptr(centroid, c_double),
            ctypes.c_int64(tri_lo.shape[0]),
            ctypes.c_int64(int(leaf_size)),
            _ptr(order, c_int64),
            _ptr(lo, c_double),
            _ptr(hi, c_double),
            _ptr(left, c_int64),
            _ptr(right, c_int64),
            _ptr(first, c_int64),
            _ptr(count, c_int64),
            ctypes.c_int64(left.shape[0]),
            _ptr(stack, c_int64),
        )
    )


def blit_cells(
    out: Any, atlas: Any, tile_index: Any, xs: Any, ys: Any
) -> None:
    """Source-over ``atlas[tile_index[i]]`` onto ``out`` at each ``(x, y)``.

    ``out`` is ``(h, w, 4)`` uint8 with contiguous rows; ``atlas`` is
    ``(n_tiles, tile_h, tile_w, 4)`` uint8 and C-contiguous; ``tile_index`` is
    int32 and ``xs``/``ys`` are int64, one per cell, **in draw order**.

    Only the binary-alpha, full-opacity, normal-mode case -- the caller decides
    that, because "is this alpha binary" is one pass per distinct tile rather
    than one per cell.
    """
    handle = lib()
    if handle is None:  # pragma: no cover - callers check available() first
        raise RuntimeError("warlockc is not loaded")
    height, width = out.shape[0], out.shape[1]
    handle.warlockc_blit_cells_u8(
        _ptr(out, ctypes.c_uint8),
        ctypes.c_int64(height),
        ctypes.c_int64(width),
        ctypes.c_int64(out.strides[0]),
        _ptr(atlas, ctypes.c_uint8),
        ctypes.c_int64(atlas.shape[1]),
        ctypes.c_int64(atlas.shape[2]),
        _ptr(tile_index, ctypes.c_int32),
        _ptr(xs, ctypes.c_int64),
        _ptr(ys, ctypes.c_int64),
        ctypes.c_int64(tile_index.shape[0]),
    )


def palette_nearest_f64(queries: Any, palette: Any, out: Any) -> None:
    """Nearest ``palette`` row per ``queries`` row, Euclidean in float64.

    ``queries`` is ``(n, 3)`` float64 and ``palette`` ``(p, 3)`` float64, both
    C-contiguous; ``out`` is ``n`` int32. Ties go to the lowest index.
    """
    handle = lib()
    if handle is None:  # pragma: no cover - callers check available() first
        raise RuntimeError("warlockc is not loaded")
    handle.warlockc_palette_nearest_f64(
        _ptr(queries, ctypes.c_double),
        _ptr(palette, ctypes.c_double),
        _ptr(out, ctypes.c_int32),
        ctypes.c_int64(queries.shape[0]),
        ctypes.c_int64(palette.shape[0]),
    )


if sys.platform != "win32":  # pragma: no cover - the app is Windows-only
    # Not a hard failure: the loader simply will not find a .dll, and every
    # caller falls back. Stated here so the reason is in the module rather
    # than in a puzzled bug report.
    log.debug("warlockc is built as a Windows DLL; other platforms use numpy")
