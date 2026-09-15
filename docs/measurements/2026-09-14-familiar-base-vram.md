# Familiar on base Gemma, first real card: VRAM, thinking, and the constrained doors — 2026-09-14

Familiar T9. Up to this run every Familiar tranche (T5 chat loop, T6 router
and Manual answers, T7 character plan, T8 navigation and Create drafts) had
been tested against fake servers only. Three questions only a card answers:

1. Does the real `llama-server` accept what `pipelines/llama_client.py` sends:
   `id_slot`, `response_format` with a JSON schema, `/tokenize`, the Bearer
   key from the key file?
2. Does base Gemma 4 E2B produce replies the schema-constrained skills can
   decode?
3. What does the resident server actually cost, so `vram.FAMILIAR_GIB` (6.5,
   documented as a guess) can cite a figure?

familiar_v1.0 was deliberately not run: it is not published, and its first
real run, including the first real Clay build, belongs to T10.

## The machine

```
NVIDIA GeForce RTX 5090, 610.62, 32607 MiB
```

- llama.cpp `b10948`, `-ngl 999`, `--ctx-size 16384`, `--parallel 2`, `--jinja`.
- Weights: the base testing pin `gemma-4-E2B-it-Q8_0.gguf`, 5,048,352,864 B,
  sha256 `605d3c26…d053`. Hashed before the runs and again after, unchanged.
- Idle desktop: ~0.95 GiB used, no other compute process.
- Per-process attribution reads `[N/A]` on this driver, the same as the
  2026-09-14 Mason measurement, so every VRAM figure below is a delta of
  NVML's device-level used memory.

Harness: `tests/test_familiar_gpu.py`, run with
`uv run --no-sync pytest tests/test_familiar_gpu.py -m gpu -n 0 -v -s`. Its
server runs on a free loopback port with its key and log under a temp
directory, and nothing under `~/.warlock` is written.

## Finding 1: base Gemma thinks, and thinking starved every constrained reply

The first run passed six of eight tests. Router and constrained-door replies
came back HTTP 200 with `content=''`. A raw probe of one router request
("how do I export a GLB") showed the cause:

| Request | finish | content | completion tokens |
|---|---|---|---|
| `response_format` json_schema, max_tokens 16 | length | `''` (16 tokens of "Thinking Process: 1. **Analyze the Request:**…" in `reasoning_content`) | 16 |
| same, max_tokens 400 | stop | `{"skill": "manual"}`, after the reasoning | 305 |
| llama.cpp top-level `json_schema`, max_tokens 16 | length | `''` | 16 |
| no schema, max_tokens 64 | length | `''` | 64 |
| `response_format` plus `chat_template_kwargs: {"enable_thinking": false}`, max_tokens 16 | stop | `{"skill": "manual"}` | 12 |

Gemma 4's chat template (llama-server runs `--jinja`) opens a reasoning
channel by default, and the server's reasoning parser moves it into
`reasoning_content`. Plain chat only appeared to work because its 1,024-token
budget left room after the reasoning.

`response_format` itself works on this build: once reasoning is out of the way
the schema holds.

The request-level `enable_thinking: false` was **not sufficient**. With it on
every request, a second run still returned empty content for
"make a wooden barrel" (Clay), "generate a sprite of a fox" (Create) and the
"make me a goblin" character plan, each with `reasoning_content` populated.
At 1,200 tokens the character plan did produce `{"family": "goblin"}`, but
only after a long reasoning trace.

**Fix, verified:** the server is started with `--reasoning off
--reasoning-budget 0` (`pipelines/llama.py`). With both flags, every probed
request returned `reasoning_content=None`:
- the two router prompts that had failed;
- the character plan at its real 200-token budget, 22 tokens;
- plain chat sent with no request-level switch at all.

The request field stays in `llama_client` as a second statement of the same
intent.

**Router budget.** With reasoning off, "make a wooden barrel" returned
`{"skill": "clay_build"}` in exactly 16 tokens with `finish_reason='length'`:
complete, but with no margin. `contract.SAMPLING["router"]["max_tokens"]` is
now 32.

Run A was trained on replies with no thinking block
(`training/clay-assistant/out/run-A/sample_rendered.txt`: the model turn goes
straight to the JSON fence). But run A's own evals (`eval/run_val.py`,
`baseline/run_baseline.py`) called llama-server with `--jinja` and never
disabled thinking. Two consequences:
- base Gemma's recorded 0% on Clay builds may partly be reasoning inside the
  budget, not only a model that cannot build;
- familiar_v1.0's 74% was scored with the template's thinking default.

T10 re-measures familiar_v1.0 with reasoning off before it is trusted.

## Finding 2: VRAM

Two concurrent generations, one per slot, each with a ~5,500-token prompt and
up to ~1,500 tokens of reply; NVML sampled every 0.25 s. The figures were
identical across the three runs of the day:

| | GiB (device delta over pre-spawn) |
|---|---|
| resident, once `/health` answered 200 | **3.09** |
| peak during the two concurrent long generations | **3.20** |
| per-process | n/a (WDDM) |
| host RSS of the child | not measured (psutil is not a declared dependency) |

**3.2 GiB, not 6.5.** The guess started from the 4.70 GiB file. The likely
reason is that the "E2B" Gemma 4 variants keep their large per-layer
embedding tables in host memory rather than on the card, so the file size
overstates what `-ngl 999` puts in VRAM. That explanation is not measured
here, which is why host RSS is listed as a gap.

The 16k context's KV cache is allocated at start-up, so the resident figure
already carries most of it. The peak adds only 0.11 GiB of activation.

**Decision:** `vram.FAMILIAR_GIB = 4.0`, the measured 3.20 plus 0.8 GiB for
what one card does not show:
- a different driver or CUDA context size on another card;
- familiar_v1.0 (same architecture and quant, a file ~81 MB smaller), which
  T10 re-confirms.

`familiar_admission` still adds `HEADROOM_GIB` on top. The GPU test now
asserts `peak_gib <= vram.FAMILIAR_GIB`.

## Finding 3: the rest of what the client sends

- **Loopback and key:** the server answered on 127.0.0.1 with the Bearer key
  read from the key file.
- **Slots:** `id_slot` 0 and 1 both answered, concurrently.
- **Template margin:** raw `/tokenize` count against the completion's
  `usage.prompt_tokens`, on the frozen Clay card plus a compacted scene:
  - real overhead 15 tokens with the reasoning-on template;
  - 13 tokens with reasoning off;
  - `llama_client.TEMPLATE_MARGIN_TOKENS = 32` covers both, so it stays.
- **Clay on base Gemma:** `service.familiar.clay_build` refused with reason
  `card` before any request.
- **Stop:** the child left NVML's process list and its key file was removed.
  No `llama-server` survived any run.

## Model behaviour on base Gemma (observations, not a score)

There is no routing corpus, so these are six prompts, reasoning off, greedy:

| Prompt (mode) | Router picked |
|---|---|
| how do I export a GLB (home) | manual |
| make a wooden barrel (clay) | clay_build |
| open Mason (home) | navigate |
| make me a goblin (home) | character |
| thanks (home) | other |
| generate a sprite of a fox (create) | create |

The skill replies, also constrained:
- navigate "open Mason" → `go:mason`
- Create draft "make a reference image of a lantern" →
  `('image', 'A reference image of a lantern')`
- character plan "make me a goblin" → `{"family": "goblin"}`

All nine were the answer a person would give. Nine greedy samples say the
doors can work on base Gemma. They do not say how often.

## What would contaminate this

- Another process allocating on the card during a run. None was listed, and
  device used memory returned to the idle figure after each run.
- A running Warlock app with Familiar started. None was: no `llama-server`
  before or after, and the harness uses its own port.
- The absence of per-process attribution means a background Windows process
  allocating mid-run would be read as Familiar's. The three runs agreeing to
  the hundredth of a GiB argues against it.
