# The reconstruction engine leaves the installer — 2026-09-10

**Status: the asset half is measured and verified; the installer half is
pending a real build.** This document exists because `vendor/trellis/` stopped
being something the installer stages and became `models.ENGINE_MODELS
["trellis_runtime"]`, a registry row a user downloads from Settings → Models.
Every figure that decision rests on is here, because `INSTALL.md` quotes
measured numbers and `TODO.md` is deleted when it empties.

## What was bundled, and how much of the download it was

Measured against the checkout and the v0.0.40 build's `build\stage`:

| | bytes | |
|---|---:|---|
| `vendor/trellis/` | 878,218,576 | 9 files |
| `cublasLt64_13.dll` | 480,693,360 | NVIDIA CUDA redistributable |
| `ggml-cuda.dll` | 338,211,840 | |
| `cublas64_13.dll` | 51,569,776 | NVIDIA CUDA redistributable |
| `trellis-server.exe` | 2,976,768 | **the engine itself** |
| the other five | 4,766,832 | `trellis-cli.exe`, four `ggml*.dll`, `cudart64_13.dll` |
| staged install total | 1,416,016,882 | |
| `WarlockSetup-v0.0.40.exe` | 852,958,786 | |

So the engine was **62% of the installed tree** by bytes, and three DLLs were
96% of the engine. The program a user is actually here for is 3 MB of it; the
rest is linear algebra it links against. That asymmetry is the whole argument:
a machine that only ever draws pixel art in Inker downloaded 838 MB of CUDA.

## The published asset, verified

The engine has always been a third-party release rather than something this
project builds — `doctor.py` has pinned the URL and digest since it was
vendored, and `README.md` told developers to unpack it by hand. Making it a
download therefore needed no new supply chain, only proof that the pinned
artifact is what was in `vendor/`.

`https://github.com/pwilkin/trellis.cpp/releases/download/v0.6.0/trellis-cuda-windows-x64.zip`

- **728,541,568 bytes**, `sha256 4d08ab27e83094035fd8349aaf34d3460738df0466ef9c4991ddd958c0344bc2`
  — the digest `doctor.py` already pinned, confirmed against a real download of
  the whole file rather than taken on trust from the release page.
- **Nine members, flat.** No top-level directory, so the registry entry's
  `extract` prefix is `"."` and nothing is stripped. This was read off the
  archive's central directory, not guessed: a release that packed its files
  under a versioned folder would have the same filename and would publish a
  tree whose every presence probe failed.
- Unpacks to **878,218,576 bytes**, so the entry declares `size_gib=0.68` for
  the progress bar and `unpack_gib=0.82` for the disk refusal. The refusal
  budgets both because the archive and the tree it unpacks to coexist until the
  archive is deleted.
- **All nine members are CRC-identical to the copy that was vendored.** Checked
  by reading each member's CRC32 out of the archive's central directory and
  computing the same over `vendor/trellis/`; all nine matched. That is what
  licenses reusing `installer/runtime-manifest.json`'s nine sha256 values as
  `models.TRELLIS_RUNTIME_DIGESTS` — the pins did not move, they changed owner.

## Why the digests moved rather than being dropped

`installer/runtime-manifest.json` pinned those nine files because the installer
staged them, and `installer/verify_runtime.py` refuses both a missing pinned
file and any *unpinned* file under a declared root. With the engine no longer
staged, keeping the root would fail every developer checkout — which still has
`vendor/trellis/` for local runs — under the second rule. So the root and its
nine rows leave that file together, and the same digests reappear on the
registry entry, where they now pin what a user actually receives instead of
what a build machine happened to have.

## The new installer, built and weighed

`pwsh scripts\rebuild.ps1 -SkipTests -NoPrune`, on 2026-09-10, against the same
checkout every figure above was taken from.

| | before (v0.0.40) | after (v0.0.42) | |
|---|---:|---:|---|
| Installer download | 852,958,786 | **169,666,529** | −80.1% |
| Installed base runtime | 1,416,016,882 | **538,983,959** | −61.9% |

**The installer is a fifth of what it was, and that is a larger cut than the
naive arithmetic predicts.** Removing 878,218,576 bytes from a 1,416,016,882-byte
stage is a 62% reduction, and extrapolating the old installer's compression
ratio on to the smaller stage predicted about 324 MB. The real figure is half
that again, because the assumption behind the extrapolation is wrong: the
engine's three big DLLs are already-optimised native code and barely compress
— the upstream zip is 728,541,568 bytes for 878,218,576 of content, a 17%
saving — while the Python runtime beside them is source, bytecode and text that
LZMA2 eats. So the engine was not 62% of the *installer*, it was **80%** of it.
That is why this is measured rather than derived, and why the number is
recorded here rather than only in `INSTALL.md`.

What is left, and where it goes:

| | bytes | |
|---|---:|---|
| bundled Python runtime | 445,978,132 | 83% of the stage |
| pack wheels with no published build | 47,707,686 | `docopt`, `mojimoji`, `unidic-lite` — `TODO.md` P26 step 1 |
| the application | 40,474,638 | |
| `gltfpack` + `warlockc` | 3,099,562 | the two binaries still vendored |
| the manual | 1,201,487 | |

**A fresh install has no fatal rows.** Run directly against the staged runtime
with a throwaway `WARLOCK_HOME` — no models, no engine, no packs — `warlock
doctor` exits 0, reports `[SETUP] trellis-server.exe` with `Install "TRELLIS.2
engine" in Settings -> Models` ahead of the pinned manual command, and produces
**zero** `[FATAL]` rows. That is the claim this whole change rests on, checked
against the built artifact rather than only against the suite that asserts it.

## What is still not measured

A downloaded engine has not yet reconstructed anything. The engine needs a card
with 16 GB of VRAM and the development box has 8 GB, so "the download lands,
verifies, ungreys Create and then actually generates a mesh" belongs with
`TODO.md` P1's surviving half, on the machine that can close it.
