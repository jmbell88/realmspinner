# Third-party notices

Realmspinner is licensed under the GNU General Public License v3.0 or later
(see [`LICENSE`](LICENSE)). It ships, bundles or downloads the components below,
each under its own terms. Nothing here overrides those terms.

This file travels with the Windows installer: `installer/build.ps1` stages it
beside the binaries it describes, because MIT and the NVIDIA redistributable
EULA both require the notice to accompany the binary rather than merely exist in
a repository somewhere.

---

## Bundled native binaries

Pinned by SHA-256 in `installer/runtime-manifest.json` and verified at build
time. Both are redistributed unmodified.

| Component | Files | Upstream | Licence |
|---|---|---|---|
| meshoptimizer | `gltfpack.exe` | <https://github.com/zeux/meshoptimizer> | MIT |
| realmspinnerc | `realmspinnerc.dll` | this repository (`native/`) | GPL-3.0-or-later, as part of this program |

`realmspinnerc.dll` is optional: every kernel in it has a NumPy fallback and the
application runs without the DLL present.

## The reconstruction engine, which is downloaded and not redistributed

**Until 2026-09-10 the three components below were in the table above.** They
were 838 MB of every installer — more than half the installed application — so
they became a download instead: `Settings → Models` fetches
`trellis-cuda-windows-x64.zip` from trellis.cpp's own GitHub release, verifies
the SHA-256 that `src/realmspinner/models.py` pins against it, and unpacks it under
the user's Realmspinner home.

That is a change of legal posture and not only of download size, so it is worth
stating plainly: **Realmspinner no longer redistributes these files.** The bytes
travel from trellis.cpp's release page to the user, and this project ships a URL
and a digest. The notices are kept here anyway — the app fetches them on the
user's behalf and runs them, which is enough reason for the terms to be
somewhere the user can find them, and this file is installed at the application
root whether or not the engine is ever downloaded.

| Component | Files | Upstream | Licence |
|---|---|---|---|
| trellis.cpp | `trellis-server.exe`, `trellis-cli.exe` | <https://github.com/pwilkin/trellis.cpp> | MIT |
| ggml | `ggml.dll`, `ggml-base.dll`, `ggml-cpu.dll`, `ggml-cuda.dll` | <https://github.com/ggml-org/ggml> | MIT |
| NVIDIA CUDA runtime | `cudart64_13.dll`, `cublas64_13.dll`, `cublasLt64_13.dll` | NVIDIA CUDA Toolkit 12.8 redistributables | NVIDIA CUDA Toolkit EULA — redistribution permitted under the "Attachment A" redistributable list |

A source checkout is the one case where these files still arrive by hand, into
`vendor/trellis/` — see `README.md`. Nothing is redistributed there either: the
directory is gitignored and the developer downloads the same archive from the
same place.

## Familiar's runtime, which is also downloaded and not redistributed

Same shape as the reconstruction engine above, and added when Familiar did:
`Settings → Models` fetches llama.cpp's own Windows CUDA build (split across
two release zips — the server binaries, and the CUDA redistributable apart
from them) from llama.cpp's GitHub release, verifies the SHA-256
`src/realmspinner/models.py` pins, and unpacks both under
`~/.realmspinner/engine/llama/`. `docs/MODELS.md` carries the exact build tag, the
pinned revision and the download commands. Realmspinner does not redistribute any
of it.

| Component | Files | Upstream | Licence |
|---|---|---|---|
| llama.cpp | `llama-server.exe` and the DLLs it links | <https://github.com/ggml-org/llama.cpp> | MIT |
| ggml (llama.cpp's own build, distinct from trellis.cpp's) | `ggml.dll`, `ggml-base.dll`, `ggml-cpu.dll`, `ggml-cuda.dll` | <https://github.com/ggml-org/ggml> | MIT |
| NVIDIA CUDA runtime | `cublas64_12.dll`, `cublasLt64_12.dll`, `cudart64_12.dll` | NVIDIA CUDA Toolkit 12.4 redistributables | NVIDIA CUDA Toolkit EULA — redistribution permitted under the "Attachment A" redistributable list |

## Bundled Python runtime

The installer packs a CPython 3.13 runtime from
[python-build-standalone](https://github.com/astral-sh/python-build-standalone)
(PSF-2.0, plus the licences of the C libraries it embeds — OpenSSL, SQLite,
zlib, libffi, and others; see that project's own `LICENSE` files, which travel
inside the runtime tree the installer copies).

## Bundled fonts

| Font | Upstream | Licence | Notice shipped as |
|---|---|---|---|
| Inter (PUA-stripped) | <https://github.com/rsms/inter> | SIL Open Font License 1.1 | `src/realmspinner/studio/resources/fonts/LICENSE-inter.txt` |
| Lucide icons | <https://github.com/lucide-icons/lucide> | ISC | `src/realmspinner/studio/resources/fonts/LICENSE-lucide.txt` |
| Familiar sigil (U+2726 subset of Noto Sans Symbols 2) | <https://github.com/notofonts/symbols> | SIL Open Font License 1.1 | `src/realmspinner/studio/resources/fonts/LICENSE-familiar-sigil.txt` |

## Vendored source

| Component | Where | Upstream | Licence |
|---|---|---|---|
| BiRefNet modelling code | `src/realmspinner/pipelines/birefnet/` | <https://github.com/ZhengPeng7/BiRefNet> | MIT |
| ACE-Step pipeline code | `src/realmspinner/pipelines/acestep/` | <https://github.com/ace-step/ACE-Step> | Apache-2.0 |

Vendored rather than downloaded so that the application never executes Python it
fetched at runtime. The pinned commit, the SHA-256 of every original file and a
documented diff are in each directory's own ATTRIBUTION.md file:
[BiRefNet](src/realmspinner/pipelines/birefnet/ATTRIBUTION.md),
[ACE-Step](src/realmspinner/pipelines/acestep/ATTRIBUTION.md).

## Test fixtures

Not part of the application and not staged by the installer, but `/tests` is in
the source distribution's allowlist (`pyproject.toml`), so a source release
redistributes them — and CC-BY's attribution requirement travels with the file.

| Asset | Where | Upstream | Licence |
|---|---|---|---|
| CesiumMan | `tests/fixtures/humanoid/cesium_man.glb` | Cesium, via the [Khronos glTF sample models](https://github.com/KhronosGroup/glTF-Sample-Models) | CC-BY 4.0 |

**Anything published that was rendered from CesiumMan credits Cesium.** That
file's own [`ATTRIBUTION.md`](tests/fixtures/humanoid/ATTRIBUTION.md) carries the
full terms and travels beside it; this table exists because a reader looking for
what this project redistributes looks here first.

## Python dependencies

Installed from PyPI by `uv`, and packed into the installer's runtime. The
complete, exact set with resolved versions is `uv.lock`. The ones whose terms
are worth calling out:

| Package | Licence | Note |
|---|---|---|
| `bpy` (Blender as a Python module) | **GPL-3.0** | The reason this project is GPL-3.0. Only `src/realmspinner/pipelines/blender_worker.py` imports it, and only in a subprocess — but the installer distributes it inside one executable alongside this program, so the combined work is GPL-3.0. Installed by the `rig` extra. |
| `pygame-ce` | LGPL-2.1 | Used unmodified as a library. |
| `PyOpenGL`, `moderngl`, `imgui-bundle`, `trimesh`, `zstandard`, `pillow` | MIT / MIT / MIT / MIT / BSD-3 / MIT-CMU | |
| `numpy`, `scipy`, `opencv-python-headless` | BSD-3 / BSD-3 / Apache-2.0 | |
| `torch`, `diffusers`, `transformers`, `huggingface-hub`, `manifold3d` | BSD-3 / Apache-2.0 / Apache-2.0 / Apache-2.0 / Apache-2.0 | |

## Model weights

**Not bundled.** Every checkpoint is downloaded by the user, on request — the
installer ships none of them and this project redistributes none of them. Most
come from Hugging Face; the one exception is noted below. They are licensed by
their publishers, and two of them restrict commercial use of what you generate.

| Model | Publisher | Source | Licence | Commercial use of output |
|---|---|---|---|---|
| SDXL 1.0 | Stability AI | Hugging Face | OpenRAIL++-M | Permitted, subject to the use restrictions |
| SDXL-Turbo | Stability AI | Hugging Face | Stability AI Non-Commercial Research Community License | **No** — commercial use requires a paid Stability membership |
| Playground v2.5 | Playground | Hugging Face | Playground v2.5 Community License | Permitted below 1M monthly active users; requires shipping the licence and its attribution string |
| Juggernaut XL v9 | RunDiffusion | Hugging Face | OpenRAIL-M | Permitted, subject to the use restrictions |
| DreamShaper XL | Lykon | Hugging Face | OpenRAIL++-M | Permitted, subject to the use restrictions |
| FLUX.2 klein / klein-base 4B | Black Forest Labs | Hugging Face | Apache-2.0 | Permitted |
| TRELLIS.2-4B | Microsoft | Hugging Face | MIT | Permitted |
| BiRefNet weights | ZhengPeng7 | Hugging Face | MIT | Permitted |
| ACE-Step v1 3.5B | ACE-Step | Hugging Face | Apache-2.0 | Permitted |
| Qwen3-VL-4B-Instruct GGUF (Qwen's own Q8_0, a testing pin) | Qwen, requantizing their own `Qwen/Qwen3-VL-4B-Instruct` | Hugging Face | Apache-2.0 | Permitted |
| `familiar_v1.0` (reserved for a future Clay-assistant fine-tune of `Qwen/Qwen3-VL-4B-Instruct` by this project's own training pipeline; not yet published) | Realmspinner (this project), fine-tuning Qwen's `Qwen/Qwen3-VL-4B-Instruct` | Realmspinner's own download row, not a third-party Hub repo | Apache-2.0 (base); the derivative itself will ship under this project's own GPL-3.0-or-later | Not yet published |
| Hybrid Demucs (`hdemucs_high_trained.pt`) | Meta / torchaudio | `download.pytorch.org`, **not** Hugging Face | MIT code, **CC BY-NC-SA 4.0 weights** | **No** — Meta states the trained weights are for scientific purposes only; see [`docs/MODELS.md`](docs/MODELS.md) |

**`familiar_v1.0` is the one row above that will be a Realmspinner-trained derivative, not a
pass-through fetch, once it ships.** Unlike the previous base (Gemma 4 E2B, replaced 2026-09-16),
whose Gemma Terms of Use needed their own review before a fine-tune of it could ship, the current
base, `Qwen/Qwen3-VL-4B-Instruct`, is Apache-2.0 outright — a permissive licence with no
fine-tune-specific conditions to satisfy. No fine-tune exists yet; when one does, Realmspinner will
host it under its own download row
rather than pointing at a third party's Hub repo, and ship it openly under Realmspinner's own
GPL-3.0-or-later ([`LICENSE`](LICENSE)), the same as the rest of this project.

The application surfaces this per model where the registry carries it. Of the
eleven registry dataclasses in `realmspinner.models` (one per `_table()`-built
registry — `BaseModel`, `StyleLora`, `IPAdapter`, `ControlNet`, `EngineModel`,
`MetricModel`, `PoseModel`, `MusicModel`, `SeparationModel`, `MattingModel`,
`FamiliarModel`), three declare a `license` field — `BaseModel`, `MusicModel`
and `SeparationModel` — and only for those does `service/downloads.py`'s
`rows()` put a licence in the row, so only those show a licence line in the
model picker and the download confirmation. `StyleLora`, `IPAdapter`,
`ControlNet`, `EngineModel`, `MetricModel`, `PoseModel`, `MattingModel` and
`FamiliarModel` carry no `license` field, so no licence line is shown for
those entries in-app — including TRELLIS.2-4B (`EngineModel`), BiRefNet
(`MattingModel`), llama.cpp and Qwen3-VL-4B-Instruct (`FamiliarModel`), all shown by
hand in the tables above but not read from the registry.
Of those eight fieldless classes, [`docs/MODELS.md`](docs/MODELS.md) writes a
row by hand for three -- TRELLIS.2-4B (`EngineModel`), BiRefNet
(`MattingModel`), and llama.cpp and Qwen3-VL-4B-Instruct (`FamiliarModel`'s
runtime and weights), MIT or Apache-2.0. For the other five -- `StyleLora`,
`IPAdapter`, `ControlNet`, `MetricModel` (DINOv2) and `PoseModel` (ViTPose)
-- it names no row at all and says instead that they carry their own terms
on their own repository pages, not audited by this project (the 2026-09-06
and 2026-09-08 audits, both finding docs-03, narrowed this paragraph twice
already; the 2026-09-20 audit, finding docs-04, then caught the paragraph
still saying "seven... two... the other five" while listing `FamiliarModel`
among the eight fieldless classes above and never mentioning it again --
`docs/MODELS.md` had documented llama.cpp and Qwen3-VL-4B-Instruct by hand
since Familiar shipped, exactly like the two classes this sentence did
count). If you intend to sell what you generate, read the row for the model
you generated it with, or its own repository page directly if docs/MODELS.md
has none.
