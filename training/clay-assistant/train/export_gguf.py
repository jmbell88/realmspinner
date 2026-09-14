"""Export a finished run's merged 16-bit model to GGUF, then quantise.

Separate from ``train_a.py`` so an export failure cannot cost a run. Uses the llama.cpp
checkout Unsloth Studio installed (``~/.unsloth/llama.cpp``): its ``convert_hf_to_gguf.py``
for the bf16 GGUF, its prebuilt ``llama-quantize.exe`` for the requested quant types -- the
candidates the later integration plan chooses between. Run with Studio's Python, which has
the ``gguf`` package the converter imports::

    ~/.unsloth/studio/unsloth_studio/Scripts/python.exe train/export_gguf.py run-A
    ~/.unsloth/studio/unsloth_studio/Scripts/python.exe train/export_gguf.py run-B Q8_0
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
PKG = HERE.parent
LLAMA = pathlib.Path.home() / ".unsloth" / "llama.cpp"
CONVERT = LLAMA / "convert_hf_to_gguf.py"
QUANTIZE = LLAMA / "build" / "bin" / "Release" / "llama-quantize.exe"

DEFAULT_QUANTS = ("Q8_0",)

# The llama-quantize types this script will ask for -- an allowlist rather than an
# arbitrary string passed through to the exe. Q8_0 is the floor: run A's Q4_K_M lost 44-61
# of 232 rows to Q8_0 at every sampling setting (2026-09-13-clay-assistant-sampling.md),
# and the user settled on Q8_0 as the minimum the same day, so nothing smaller is exported.
# Refused before any subprocess runs, so a disallowed quant never costs a conversion step.
ALLOWED_QUANTS = frozenset({"Q8_0"})


def main() -> int:
    run = sys.argv[1] if len(sys.argv) > 1 else "run-A"
    quants = tuple(sys.argv[2:]) or DEFAULT_QUANTS
    unknown = [q for q in quants if q not in ALLOWED_QUANTS]
    if unknown:
        print(
            f"unknown quant type(s) {unknown} -- allowed: {sorted(ALLOWED_QUANTS)}",
            file=sys.stderr,
        )
        return 1

    out = PKG / "out" / run
    merged = out / "merged-16bit"
    if not merged.is_dir():
        print(f"no merged model at {merged}", file=sys.stderr)
        return 1
    gguf_dir = out / "gguf"
    gguf_dir.mkdir(exist_ok=True)
    bf16 = gguf_dir / f"clay-assistant-{run}-BF16.gguf"

    if not bf16.is_file():
        cmd = [
            sys.executable,
            str(CONVERT),
            str(merged),
            "--outfile",
            str(bf16),
            "--outtype",
            "bf16",
        ]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    for quant in quants:
        target = gguf_dir / f"clay-assistant-{run}-{quant}.gguf"
        if target.is_file():
            continue
        cmd = [str(QUANTIZE), str(bf16), str(target), quant]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    for p in sorted(gguf_dir.glob("*.gguf")):
        print(f"{p.stat().st_size / 2**30:6.2f} GiB  {p.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
