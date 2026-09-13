# Clay assistant, run A: Gemma 4 E2B-it fine-tuned on the verified dataset — 2026-09-12

**Status: tier one measured (door acceptance, machine-scored). Tier three (a person
grading renders) has not been sat; the pre-registered bar in
[`2026-09-10-clay-agent-benchmark-preregistration.md`](2026-09-10-clay-agent-benchmark-preregistration.md)
is therefore neither met nor missed yet.** Everything numeric here comes from
`training/clay-assistant/eval/run_val.py` and the files kept beside this document under
[`data/clay-assistant/run-A/`](data/clay-assistant/run-A/).

## What was trained

| | |
|---|---|
| base | `unsloth/gemma-4-E2B-it`, bf16 safetensors, snapshot `d36bdd38` |
| method | LoRA r=16 α=32, language attention+MLP, no vision layers; bf16, no 4-bit |
| data | `training/clay-assistant/dataset/` at tools_sha `70697ece…`: 1,932 train / 227 val rows, every one replayed clean through `agent_clay.call` before training |
| loss | responses only (`<|turn>model`); first row 128 of 2,212 tokens carry loss, exactly the fenced JSON |
| rows | 2,190–4,926 tokens (median 2,430); max_seq_length 8192 |
| schedule | lr 2e-4 cosine, warmup 3 %, 3 epochs, batch 2 × accum 8 = 363 steps, adamw_8bit, seed 3407 |
| runtime | 3,451 s on the RTX 5090, ~13 GB VRAM observed; Unsloth 2026.9.4, torch 2.11+cu130, transformers 5.5.0, TRL 0.23.1 |
| eval loss | 0.319 → 0.239 → 0.224 by epoch; train loss 0.28 mean, 0.10–0.22 in the last epoch |
| exports | LoRA adapter; merged 16-bit (`train/merge.py`, offline, because Unsloth's own merge re-checks the hub); GGUF BF16 8.67 GiB, Q8_0 4.63 GiB, Q4_K_M 3.19 GiB via the llama.cpp checkout Unsloth Studio installed |

Assistant turns were trained as a fenced ```` ```json ```` block, not Gemma's native tool
tokens, so any runtime that returns text can drive the door; the scorer and the later
integration parse the same fence.

## The measurement

Each of the 227 val rows and the five held-out corpus subjects was sent once (temp 1.0,
top-k 64, top-p 0.95, Google's recommended sampling) with the compact tool card as the
system turn and the prompt as the user turn (edit/query rows fold the scene turn and the
request into one user turn, as training did). The reply's fenced JSON was folded into one
`clay_batch` and replayed through `gen.verify` — the real door, the real `clay_diagnose`,
the same ground/bounds/serialize checks every training row passed. "Accepted" means all
of that held. It does **not** mean the object looks like the prompt.

| model | all (232) | build (179) | edit (40) | query (13) | corpus (5) |
|---|---|---|---|---|---|
| untuned base, same tool card | 10 (4 %) | **0** | **0** | 10 | **0** |
| run A, Q4_K_M | 132 (57 %) | 94 | 27 | 11 | 2 |
| run A, **Q8_0** | **178 (77 %)** | **132 (74 %)** | **35 (88 %)** | 11 | **4** |

Per family, run A at Q8_0: furniture 28/35, architecture 23/29, containers 26/34,
mechanical 27/35, vehicles 13/19, creatures 11/22, edits 35/40, queries 11/13, corpus 4/5.
The base's ten "accepted" are queries answered in prose, which is trivially a pass.

**Q8_0 is the candidate.** Q4_K_M loses twenty points of door acceptance and quadruples
the no-parse rate (4 → 20 of 232); the 1.4 GiB it saves is not worth that on a card that
holds SDXL and TRELLIS with 7 GiB to spare.

### Where run A still fails, Q8_0, 54 of 232

| cause | n | what it is |
|---|---|---|
| `no object named '…'` | 15 | a `$ref` to a name never created — nine of them in creatures, guessing `clay_add_figure` part names (`hound_Beak`, `t_Shank.R`, `s_Tail 01`); the rest a name typo between two calls of one batch |
| below the ground plane | 19 | a `lathe`/`sweep`/`tube` translated as if its profile were not re-centred, or a sword `blade` at y=0 — the same lesson every human author of the dataset also hit once |
| unknown params for a generator | 7 | e.g. `depth` on a `cylinder` |
| invented generator name | 3 | `plane_grid`, `wedge` and one truncated string |
| no parse | 4 | one JSON delimiter error, three replies with no fence |
| other refusals | 6 | `clay_op` given `axis` outside `params`; an edge-mode op from object mode; one `pyramid` with a list `base` (see below) |

### The five held-out subjects, Q8_0

| subject | door | objects | what the render shows |
|---|---|---|---|
| chair, slatted back | accepted | 10 | a chair: seat, four legs, two back panels; the three "slats" share one translation |
| bracket, two holes, gusset | accepted | 2 | two overlapping slabs, no holes, no gusset |
| telescope on tripod | refused | 5 of 7 | stopped at call 5: `no object named` a leg it had named differently |
| colonnade, eight columns | accepted | 11 | a slotted block, not eight columns — columns placed inside a base of the same size |
| serpentine creature | accepted | 21 | twenty-one small pieces scattered above the ground, not a body |

Renders and the exact calls are in [`data/clay-assistant/run-A/`](data/clay-assistant/run-A/)
(`<subject>.png`, `<subject>.calls.json`). By the pre-registration's own rules: four of
five are *buildable* (rule 1), the call counts (11, 6, 7, 12, 21) are reported and not
graded (rule 2), and the −5..+5 grades are a person's to give, blind to those counts
(rule 3). My reading of the pictures is that only the chair would grade at 0 or better, so
the bar of three-of-five is unlikely to be met by run A; that is a prediction, not a grade.

## What this says, and does not

- The fine-tune works as a fine-tune: door acceptance 0 → 74 % on builds and 0 → 88 % on
  edits, with a 58-minute run on ~2k rows. The envelope, the nesting, the naming
  conventions, `$ref`, materials-by-index: learned.
- Acceptance is a floor, not a score. The colonnade and serpent are accepted nonsense.
  The dataset taught *valid* geometry; it did not teach *composition* strongly enough for
  a 2B model to place eight columns in a ring from a sentence, even though the training
  set has rings of five to twelve columns (the held-out eight was excluded on purpose).
- Two dataset gaps show directly in the refusals: figure part names (creatures) and the
  re-centring rule for profile generators (below-ground). Run B's rows should target both,
  and add a scene-read turn before edits that name figure parts.

## Defects found on the way, none fixed here

- **`clay_add_primitive` crashes rather than refuses when `pyramid` is given a list for
  `base`** (`primitives.py:1411`, `float(base)` on a list): the door catches it as "failed
  unexpectedly; see the log" and prints a traceback. `_validate_number_or_vec` admits a
  list for every numeric param; scalar-only params need a scalar check.
- **`mirror-copy` does not select its copy** while `array-linear`/`array-radial` do, so
  `select → mirror-copy → select(same) → mirror-copy` re-mirrors the original and silently
  drops the fourth corner (found by the furniture author; `clay_ops.py`).
- Array and mirror copies keep the document's default material unless a later call names
  `<name>.001…`; no refusal, no diagnose finding, only a render shows it.

## Retention

Transcripts (the eval JSONs hold every reply verbatim) and the five corpus call lists are
kept under `data/clay-assistant/run-A/`; the adapter, merged model and GGUFs are under
`training/clay-assistant/out/run-A/`, gitignored and 27 GiB, and stay on this machine
until the tier-three document is written, per the pre-registration's rule 6.
