# Optional models

Everything on this page is optional and independently skippable. The core setup in the
[README](../README.md) — TRELLIS.2 plus SDXL 1.0 — is enough to generate assets; what follows
widens the choices. SDXL 1.0 is the shipped default because it measured best
([docs/measurements/2026-08-11-default-base-model.md](measurements/2026-08-11-default-base-model.md))
and because its 7 GB is the one base download four registered recipes share, so most of this page
is a small adapter over weights you already have rather than another checkpoint.

One thing on the generation side is **not** on this page and never will be: Create's **Character**
type needs no weights at all. A character is built from an authored family of body plans and species
rather than reconstructed from a picture, so its mesh, its rig and its rendered sheet are CPU work
with nothing to download — which is why it is the one generation type a machine with no card can
still use ([manual/22 → Characters](manual/22-generating-references.md#characters)).

`warlock doctor` lists every entry here with the exact command to fetch it,
and **Settings → Models** inside the app downloads any of them without touching a terminal (via
the out-of-process fetch worker described in the README — the app process itself stays offline).
The same pane removes them again: a **Remove** button beside each downloaded row, showing what it
would actually free, which for a recipe sharing its weights with another is far less than the
download was.

The reference image is the single biggest lever on final mesh quality — TRELLIS can only be as
good as the picture it is handed — so the image model and an optional style LoRA are per-job
choices in the guidance panel (`manual/22-generating-references.md`). Base models are
one-resident-at-a-time (a 32 GB card holds trellis plus a single SDXL-class pipe, not two), so
switching between jobs costs a reload; style LoRAs are adapters on the resident pipe and switch
for free.

## The reconstruction engine

**Two rows, and they are the one thing Create cannot do without.** Both live under
*Reconstruction engine* in Settings → Models, and both are ordinary downloads — the second of them
only since 2026-09-10.

| Row | What it is | Size |
|---|---|---|
| **TRELLIS.2 engine** | `trellis-server.exe` and the libraries it links: four `ggml` DLLs and three NVIDIA CUDA redistributables | 0.68 GB down, 0.82 GB on disk |
| **TRELLIS.2 GGUF weights** | the model that engine loads, quantised | ~16 GB |

The engine used to ship inside the installer, where it was **838 MB of a 1.4 GB install** — more
than half of it — for a program most of whose workspaces never start it. `trellis-server.exe` is
3 MB of that; the rest is `cublasLt64_13.dll` (480 MB) and `ggml-cuda.dll` (338 MB), linear algebra
it links against. So it became a download, and a machine that only draws pixel art in Inker now
fetches none of it. The figures and the verification are in
[docs/measurements/2026-09-10-engine-as-a-download.md](measurements/2026-09-10-engine-as-a-download.md).

It is the one entry on this page that is **not a Hugging Face repository**: it is a single archive
published on trellis.cpp's own GitHub releases, so it is pinned by SHA-256 rather than by a commit
revision — which is the stronger pin of the two, since a revision names an immutable commit and a
digest *is* the artifact. The stem-separation checkpoint further down uses the same transport for
the same reason.

Nothing here needs a terminal, but the paste-able form is what `warlock doctor` prints beside the
row, as with every other entry:

```powershell
curl -L -o $HOME/.warlock/engine/trellis/trellis-cuda-windows-x64.zip `
  https://github.com/pwilkin/trellis.cpp/releases/download/v0.6.0/trellis-cuda-windows-x64.zip
# then check its sha256 is 4d08ab27e83094035fd8349aaf34d3460738df0466ef9c4991ddd958c0344bc2
# then unpack it into $HOME/.warlock/engine/trellis
```

The GGUF weights are an ordinary Hub fetch, and this is the one on this page you actually need:

```powershell
# TRELLIS.2 GGUF weights (~16 GB) -> ~/.warlock/models/trellis2-gguf/
uvx hf download ilintar/trellis2-gguf --revision a57397bd3d351599d9729fc144b3f87c3f87d65b --include "*.gguf" --exclude "q4/*" --exclude "q8/*" `
  --local-dir $HOME/.warlock/models/trellis2-gguf
```

**An NVIDIA card is required and there is no CPU build.** A source checkout keeps its own copy in
`vendor/trellis/` (README step 2); a downloaded copy wins over that one, and `WARLOCK_TRELLIS_EXE`
beats both — which is the sideload path for a machine that cannot reach GitHub.

Licences: the engine is MIT (trellis.cpp and ggml) over NVIDIA's redistributable CUDA libraries,
and because you now fetch it from upstream rather than receiving it from us, Warlock redistributes
none of it. The terms are in [`THIRD-PARTY-NOTICES.md`](../THIRD-PARTY-NOTICES.md) either way.

## Familiar

**Three rows, under their own *Familiar* heading in Settings → Models**, on the same shape as the
reconstruction engine above: a runtime (split into two rows -- see below) and the weights it
loads, in their own directories (`~/.warlock/engine/llama/`, `~/.warlock/models/familiar/`) —
never the trellis ones, because llama.cpp and trellis.cpp ship their own, differently built
`ggml*.dll`.

| Row | What it is | Size |
|---|---|---|
| **Familiar runtime** | `llama-server.exe` (llama.cpp b10948, CUDA 12.4 Windows build) and the DLLs it links, minus the CUDA redistributable | ~0.24 GB down |
| **Familiar runtime (CUDA)** | the CUDA 12.4 redistributable (`cublas64_12.dll`, `cublasLt64_12.dll`, `cudart64_12.dll`) the runtime above links against | ~0.37 GB down |
| **Familiar weights (Gemma 4 E2B)** | the model it serves, quantised | ~4.7 GB |

**Two rows for the runtime, not one, and that is upstream's shape, not this app's.** llama.cpp
publishes its CUDA Windows build as two separate release zips — the server binaries, and the CUDA
redistributable apart from them, so a machine running both a CUDA-12 and a CUDA-13 build does not
fetch the same ~370 MB of cudart twice. Warlock's `Fetch` record has no way to give one registry
row two independent URL/SHA-256 pins, so this is genuinely two rows rather than one row hiding two
downloads — both land in the same directory, pinned by SHA-256 the same way the reconstruction
engine's binaries are:

```powershell
curl -L -o $HOME/.warlock/engine/llama/llama-b10948-bin-win-cuda-12.4-x64.zip `
  https://github.com/ggml-org/llama.cpp/releases/download/b10948/llama-b10948-bin-win-cuda-12.4-x64.zip
# then check its sha256 is 9839398baa5a74fcf2447168000b2a8c659e6ee0d944f7686bb72168a0bc1e35
curl -L -o $HOME/.warlock/engine/llama/cudart-llama-bin-win-cuda-12.4-x64.zip `
  https://github.com/ggml-org/llama.cpp/releases/download/b10948/cudart-llama-bin-win-cuda-12.4-x64.zip
# then check its sha256 is 8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6
# then unpack both into $HOME/.warlock/engine/llama
```

**The weights row is a testing pin, stated as one.** It is Unsloth's own Q8_0 requantization of
the stock `google/gemma-4-E2B-it` instruct model — not the Clay-assistant fine-tune
(`training/clay-assistant/`), which a later tranche (T10) swaps in as the shipped pin. There is no
picker and no path override: this exact file, or nothing.

```powershell
# Familiar weights (~4.7 GB) -> ~/.warlock/models/familiar/
uvx hf download unsloth/gemma-4-E2B-it-GGUF --revision 0314792d7f1f7e229411f620751375812bb9faf2 `
  --include "gemma-4-E2B-it-Q8_0.gguf" --local-dir $HOME/.warlock/models/familiar
```

Licences: the runtime is MIT (llama.cpp and ggml) over NVIDIA's redistributable CUDA libraries, on
the same "fetched from upstream, nothing redistributed" footing as the reconstruction engine. The
weights repository (`unsloth/gemma-4-E2B-it-GGUF`) is published under Apache 2.0, matching the base
`google/gemma-4-E2B-it` model it requantizes — see `TODO.md` for the outstanding human review of
whether that licence still holds once the Clay-assistant fine-tune replaces this pin.

## Licences, and what you may do with the output

**Read this before you sell anything you generated.** These weights are not part
of Warlock Studio: you download them from their publishers, under their terms,
and two of them restrict commercial use of what they produce. The app shows the
row below in the model picker and in **Settings → Models**, and marks the
restricted ones; this table is the same information in full.

| Model | Licence | Output may be sold? |
|---|---|---|
| **SDXL 1.0** (the default, and the Hyper-SD / LCM / Lightning / PAG recipes over it) | OpenRAIL++-M | Yes, subject to the licence's use restrictions |
| **SDXL-Turbo** | Stability AI Non-Commercial Research Community License | **No** — commercial use requires a paid Stability AI membership |
| **Playground v2.5** | Playground v2.5 Community License | Yes below 1M monthly active users, and you must ship the licence text plus its attribution string |
| **Juggernaut XL v9** | OpenRAIL-M | Yes, subject to the use restrictions |
| **DreamShaper XL** | OpenRAIL++-M | Yes, subject to the use restrictions |
| **FLUX.2 klein / klein-base 4B** | Apache-2.0 | Yes |
| **TRELLIS.2-4B** (the reconstruction engine) | MIT | Yes |
| **llama.cpp** (Familiar's runtime) | MIT | Yes |
| **Gemma 4 E2B** (Familiar's weights, testing pin) | Apache-2.0 | Yes — pending the human licence review for the fine-tune that replaces this pin (see TODO.md) |
| **BiRefNet** (matting) | MIT | Yes |
| **ACE-Step v1 3.5B** (Muse) | Apache-2.0 | Yes |
| **Hybrid Demucs** (stem separation) | MIT code, **CC BY-NC-SA 4.0 weights** | **No** — see below |

The OpenRAIL family is commercially permissive but carries *use* restrictions —
a list of things you may not generate. They are short; read them once.

**Hybrid Demucs is the second restricted one, and it is the only optional
download in this list.** The Demucs code is MIT, but Meta has stated the trained
weights are provided for scientific purposes only, and the checkpoint here was
trained the same way with no new grant. So stems you make with it are not
cleanly licensed for a commercial release. Muse works without it — every take
still generates, plays, exports and imports into Sirens; what you lose is the
four stem files. Open-Unmix is not an escape: its code is MIT and MUSDB18-HQ is
CC BY-NC-SA. The app marks the row and warns at the moment you agree to the
download, and the decision is then yours.

Style LoRAs, ControlNet, IP-Adapter, DINOv2 and ViTPose carry their own terms on
their own repository pages. None of them is known to restrict commercial output,
but this project has not audited each one and does not warrant them.

Warlock Studio itself is GPL-3.0-or-later ([LICENSE](../LICENSE)); the components
it bundles are in [THIRD-PARTY-NOTICES.md](../THIRD-PARTY-NOTICES.md).

## Image models and style LoRAs

```powershell
# SDXL 1.0 full CFG -- the shipped default recipe, and no download at all if
# you followed the README: it is the same models/sdxl-base-1.0 weights run
# undistilled at 30 steps and CFG 7.0. Slowest of the SDXL entries and the one
# with real structural control: it takes ControlNet, and the negative prompt
# carries full weight. Everything below in this block reuses these weights.

# SDXL 1.0 + PAG -- also no download: the default recipe with two training-free
# sampling upgrades (perturbed-attention guidance at 3.0, CFG rescale at 0.7,
# which counters high-CFG washout). The opt-in arm the bench compares against
# the default before any flip.

# SDXL-Turbo (~7 GB): the 4-step fast option, at 512 px and guidance 0. Its own
# checkpoint rather than a recipe -- nothing else in the registry shares it --
# and no longer part of the core setup. Worth having when iteration speed
# matters more than fidelity, and the entry WARLOCK_T2I_DIR redirects.
uvx hf download stabilityai/sdxl-turbo --revision 71153311d3dbb46851df1931d3ca6e939de83304 --include "*.json" --include "*.txt" --include "*fp16.safetensors" `
  --exclude "sd_xl_turbo_1.0*" --local-dir $HOME/.warlock/models/sdxl-turbo

# SDXL 1.0 + Hyper-SD (787 MB on top of the base above). Style LoRAs are
# trained against full SDXL at 20-25 steps with CFG, so they land noticeably
# stronger here than on Turbo's 4 steps at guidance 0. Hyper-SD buys the step
# count back. If you skipped the README's step 4, the base line comes first:
uvx hf download stabilityai/stable-diffusion-xl-base-1.0 --revision 462165984030d82259a11f4367a4eed129e94a7b `
  --include "*.json" --include "*.txt" --include "*fp16.safetensors" --local-dir $HOME/.warlock/models/sdxl-base-1.0
uvx hf download ByteDance/Hyper-SD --revision bc08d970a87c74c71209491d64e3525845698863 Hyper-SDXL-4steps-lora.safetensors --local-dir $HOME/.warlock/models/loras

# Playground v2.5 (~7 GB): highest fidelity, ~25 steps with CFG, correspondingly slower.
uvx hf download playgroundai/playground-v2.5-1024px-aesthetic --revision 1e032f13f2fe6db2dc49947dbdbd196e753de573 `
  --include "*.json" --include "*.txt" --include "*fp16.safetensors" --local-dir $HOME/.warlock/models/playground-v2.5

# SDXL 1.0 + LCM (pixel art): the same base weights again, run at 8 steps with
# guidance 1.0 -- the recipe the pixel-art LoRA below was trained against. The
# LCM LoRA has to be renamed: loras/ is flat, and the upstream filename is
# generic enough that any other repo's default-named adapter would overwrite it.
uvx hf download latent-consistency/lcm-lora-sdxl --revision a18548dd4956b174ec5b0d78d340c8dae0a129cd `
  pytorch_lora_weights.safetensors --local-dir $HOME/.warlock/models/loras
Rename-Item $HOME/.warlock/models/loras/pytorch_lora_weights.safetensors lcm-lora-sdxl.safetensors

# SDXL 1.0 + Lightning (394 MB, reuses the sdxl-base-1.0 weights above): a
# second 4-step distillation, adversarial where Hyper-SD is trajectory-
# consistency, so the two are directly comparable with everything else fixed.
uvx hf download ByteDance/SDXL-Lightning --revision c9a24f48e1c025556787b0c58dd67a091ece2e44 `
  sdxl_lightning_4step_lora.safetensors --local-dir $HOME/.warlock/models/loras

# Juggernaut XL v9 (~6.9 GB): a photoreal SDXL finetune, DPM++ 2M Karras at 35
# steps with CFG 4.0 -- the middle of the ranges its own model card gives.
uvx hf download RunDiffusion/Juggernaut-XL-v9 --revision cf419233522daa0b9ea36c3aff98fa2cab1fb0fb `
  --include "*.json" --include "*.txt" --include "*fp16.safetensors" --local-dir $HOME/.warlock/models/juggernaut-xl-v9

# DreamShaper XL (~6.9 GB): the stylised counterpart, DEIS at 25 steps per its card.
uvx hf download Lykon/dreamshaper-xl-1-0 --revision 41e6644752a8c9aa63930e6043c4fd83c7708420 `
  --include "*.json" --include "*.txt" --include "*fp16.safetensors" --local-dir $HOME/.warlock/models/dreamshaper-xl

# FLUX.2 klein-base 4B (~16 GB): the one non-SDXL architecture -- one Qwen3 text
# encoder at 512 tokens instead of two CLIPs at 77, and a DiT instead of a UNet.
# Conditioning and seamless tiles are SDXL-only and are refused on it. Style
# LoRAs are per-architecture rather than SDXL-only: an adapter declares the
# family it was fitted to, and the picker offers a base only the ones that fit.
# The negative prompt does work here, which is why this is the undistilled
# -base variant rather than the distilled FLUX.2-klein-4B below (that one
# hardwires is_distilled=True, which switches classifier-free guidance off
# entirely).
# Streamed onto the card a submodule at a time, so it peaks near 10 GB rather
# than 16 and still coexists with trellis. The --exclude skips a redundant
# 7.75 GB single-file checkpoint the repo ships beside the diffusers layout.
uvx hf download black-forest-labs/FLUX.2-klein-base-4B --revision a3b4f4849157f664bdbc776fd7453c2783562f4d `
  --include "*.json" --include "*.txt" --include "*.jinja" --include "*.safetensors" `
  --exclude "flux-2-klein-base-4b.safetensors" --local-dir $HOME/.warlock/models/flux2-klein-base-4b

# FLUX.2 klein 4B distilled (~16 GB): the same architecture at the opposite
# recipe -- 4 steps at guidance 1.0 rather than 50 at 4.0. It registers
# is_distilled=True, so classifier-free guidance never runs and the negative
# prompt is inert on it; pick klein-base above when a negative prompt is
# wanted. It is here because the FLUX.2 pixel-art LoRA below was trained
# against it.
uvx hf download black-forest-labs/FLUX.2-klein-4B --revision e7b7dc27f91deacad38e78976d1f2b499d76a294 `
  --include "*.json" --include "*.txt" --include "*.jinja" --include "*.safetensors" `
  --exclude "flux-2-klein-4b.safetensors" --local-dir $HOME/.warlock/models/flux2-klein-4b

# Style LoRAs -> ~/.warlock/models/loras/
uvx hf download goofyai/3d_render_style_xl --revision 5ec74a57db5e244a2157173781a7b29045f88237 3d_render_style_xl.safetensors --local-dir $HOME/.warlock/models/loras
uvx hf download artificialguybr/3DRedmond-V1 --revision f4b4b980972566aea7c71af9d4e170d7fcb6c404 `
  3DRedmond-3DRenderStyle-3DRenderAF.safetensors --local-dir $HOME/.warlock/models/loras
uvx hf download artificialguybr/ps1redmond-ps1-game-graphics-lora-for-sdxl --revision 74bb3a6e2efd47ead698ff3ac2695ab63bbd2d5c `
  PS1Redmond-PS1Game-Playstation1Graphics.safetensors --local-dir $HOME/.warlock/models/loras
# Pixel art: generates on a pixel grid rather than being downscaled into one.
uvx hf download nerijs/pixel-art-xl --revision 8bf4a4d9ea283e00a51fafda8e0539f8248ea037 pixel-art-xl.safetensors --local-dir $HOME/.warlock/models/loras
# Pixel art for FLUX.2 klein -- the one non-SDXL adapter, offered only on the two
# klein entries above. Renamed on the way in because loras/ is flat and shared
# across families, and lcm-lora-sdxl ships the same generic filename.
uvx hf download Limbicnation/pixel-art-lora --revision 0ac8e5c3400af68228811edc324721e25fc26777 `
  pytorch_lora_weights.safetensors --local-dir $HOME/.warlock/models/loras
Rename-Item $HOME/.warlock/models/loras/pytorch_lora_weights.safetensors pixel-art-klein.safetensors
```

## Conditioning, matting and measurement models

Seven more registry entries, none of them required to generate anything. They lived only in
`models.py` until the download machinery started generating both lists from the same `Fetch`
records; `warlock doctor` reports each one and the Settings pane can fetch it.

```powershell
# IP-Adapter Plus (~3.5 GB): condition on a reference image's appearance. Both
# halves are needed -- the weights alone load fine and then fail at the first call.
uvx hf download h94/IP-Adapter --revision 018e402774aeeddd60609b4ecdb7e298259dc729 sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors `
  --local-dir $HOME/.warlock/models/ip-adapter
uvx hf download h94/IP-Adapter --revision 018e402774aeeddd60609b4ecdb7e298259dc729 --include "models/image_encoder/*" --local-dir $HOME/.warlock/models/ip-adapter

# ControlNet, Canny (~2.5 GB): lock the silhouette to a reference image's edges.
uvx hf download diffusers/controlnet-canny-sdxl-1.0 --revision eb115a19a10d14909256db740ed109532ab1483c `
  --include "*.json" --include "*fp16.safetensors" --local-dir $HOME/.warlock/models/controlnet-canny-sdxl

# ControlNet, Depth (~2.5 GB): anchor a re-texture's restyle passes to the
# mesh's own geometry. The hint is rendered by Blender from the mesh itself
# (never estimated from a photo), so this one is only ever used by the
# re-texture stage and does not appear in the reference-stage conditioning
# pickers.
uvx hf download diffusers/controlnet-depth-sdxl-1.0 --revision 17bb97973f29801224cd66f192c5ffacf82648b4 `
  --include "*.json" --include "*fp16.safetensors" --local-dir $HOME/.warlock/models/controlnet-depth-sdxl

# BiRefNet (~1 GB): host-side background matting for 2D exports. Without it the
# alpha comes from a corner flood fill, with visibly rougher edges. Weights only
# -- the repo's own modelling code is vendored at `pipelines/birefnet/` and
# nothing is executed out of the downloaded directory, which is why this fetch
# carries no `*.py`. That vendored code imports einops/kornia/timm/torchvision,
# so it still wants `uv sync --extra text2image`.
uvx hf download ZhengPeng7/BiRefNet --revision e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4 `
  --include "*.json" --include "*.safetensors" --local-dir $HOME/.warlock/models/birefnet

# DINOv2 base (~400 MB): the identity metric `python -m warlock.bench` scores
# with. A missing one costs a number, never a job.
uvx hf download facebook/dinov2-base --revision f9e44c814b77203eaa57a6bdbbd535f21ede1415 `
  --include "*.json" --include "*.safetensors" --local-dir $HOME/.warlock/models/dinov2-base

# PickScore v1 (~3.7 GB): a CLIP-H fine-tuned on human A/B preferences over
# generated images. With it, ranking a submit's candidates adds "which of
# these would a person pick for this prompt" beside composition and the style
# anchor; without it the ranking is exactly what it was. CPU, like the anchor.
# Two commands, deliberately: `hf download` ignores --include when a filename
# is also given, so one merged line fetches the weights and drops the configs.
uvx hf download yuvalkirstain/PickScore_v1 --revision a4e4367c6dfa7288a00c550414478f865b875800 model.safetensors `
  --local-dir $HOME/.warlock/models/pickscore-v1
uvx hf download yuvalkirstain/PickScore_v1 --revision a4e4367c6dfa7288a00c550414478f865b875800 `
  --include "*.json" --include "*.txt" --local-dir $HOME/.warlock/models/pickscore-v1
```

## Landmark-informed joint placement (rigging)

```powershell
uvx hf download usyd-community/vitpose-base-simple --revision a93ac0c67e0b7e2c55287d21d4c460c8f3c54d45 `
  --include "*.json" --include "*.safetensors" --local-dir $HOME/.warlock/models/vitpose-base
```

Without it, a humanoid rig places its joints by scaling the template onto the mesh's bounding box
— which is right when the reference is standing in a T-pose and progressively wrong as it departs
from one. With it, the *reference image the mesh was reconstructed from* is read for the subject's
actual shoulders, elbows, hips, knees and ankles, and those positions are used for the skeleton's
X and Z. Depth still comes from the template: one view cannot supply it.

It runs on the CPU, beside the resident trellis and SDXL rather than taking VRAM from them, and it
costs about a second per rig. Nothing about it is required: it engages by itself when the weights
are present, the template is `humanoid`, the job has a reference image, and the detection clears
its sanity gates — and falls back wholesale to the bounding-box fit otherwise, never partially,
since a skeleton half-measured and half-assumed is worse than either. `rig.json` records which fit
produced the joints under `fit.method` (`pose2d`, `bbox`, or `manual` after an adjust-joints
pass), and `WARLOCK_POSE_FIT=0` turns the whole thing off.

## The music model (Muse)

One entry, and the only model the Muse mode can use. It is ~8.3 GB and lives beside every
other model in the model root, so nothing new has to be configured to hold it.

```powershell
# ACE-Step v1 3.5B (~8.3 GB): text-to-music. Style tags and a lyric block in, a
# 44.1 kHz WAV out. Runs in its own subprocess on the same card as the image
# model, and needs `uv sync --extra music` -- its own extra, not `text2image`.
uvx hf download ACE-Step/ACE-Step-v1-3.5B --revision 82cd0d7b6322bd28cd4e830fe675ddb6180ce36c `
  --local-dir $HOME/.warlock/models/ace-step-v1-3.5b
```

Unlike the measuring models above there is no fallback: Muse refuses at the door and names this
download, rather than generating something worse. The pipeline code is vendored in this
application (`src/warlock/pipelines/acestep/`) with its modifications documented beside it, so
nothing is executed out of the downloaded directory — the same arrangement BiRefNet has.

### Stem separation (optional)

One entry, and the only optional model Muse has. It splits a finished take into drums, bass,
vocals and everything else.

```powershell
# Hybrid Demucs (~320 MB): stem separation for Muse. NOT on Hugging Face -- a
# single checkpoint file, pinned by digest rather than by a commit. The model
# *class* ships inside torchaudio, which `--extra music` already installs, so
# this is the trained weights and nothing else.
curl -L -o $HOME/.warlock/models/hdemucs-high/hdemucs_high_trained.pt `
  https://download.pytorch.org/torchaudio/models/hdemucs_high_trained.pt
# then check its sha256 is
#   a004b2790d73ffeaa535db458a1a79b539dfdbafbccc31f275d07e632ebd7816
```

**It is optional in a way the music model is not.** ACE-Step missing means Muse refuses at the
door, because there is no fallback and there is not supposed to be one. This missing means only
that the *Stems* button refuses; everything else about a take is unaffected.

See the licence note above before you use its output commercially — this is the one download in
this document whose weights are non-commercial.

**FLUX.1 is not offered; FLUX.2 klein is.** The two `FLUX.1` checkpoints — `dev` and `schnell` —
are not in the registry above at all, for the two reasons below. FLUX.2 klein is a different
architecture with neither problem, which is why it *is* an ordinary registry entry.

Both `dev` and `schnell` are click-through gated on Hugging Face, and 12B parameters will not
coexist with trellis on one card. To use a local FLUX copy anyway: download it yourself
(`uvx hf auth login` for the download only), point `WARLOCK_T2I_DIR` at it, and set
`WARLOCK_VRAM_EXCLUSIVE=1`. Note that `WARLOCK_T2I_DIR` only redirects *where* the built-in
`turbo` entry loads from — the redirect is pinned to that entry *by name*, so the 2026-08-11 move
of the default onto SDXL 1.0 does not affect it. It still runs at turbo's settings (512 px, 4 steps, guidance 0),
which suit schnell-like distilled checkpoints and nothing else. A model that needs different
settings wants a `models.py` entry, not this variable.
