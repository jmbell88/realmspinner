# Clay assistant: sampling settings and quant, measured on run A — 2026-09-13

**Status: measured (door acceptance, machine-scored), and the decision for run B's eval is
taken from it.** No person has graded anything here; the renders are described, not graded.
Every number comes from `training/clay-assistant/eval/run_val.py` and the eval files kept
under [`data/clay-assistant/run-A/`](data/clay-assistant/run-A/).

## Why this was measured

Run A ([`2026-09-12-clay-assistant-run-A.md`](2026-09-12-clay-assistant-run-A.md)) was
scored once per row at temperature 1.0, top-k 64, top-p 0.95 (Google's recommended Gemma
sampling). One sample a row cannot tell a better model from a luckier draw, and a tool-call
model has no use for creative variety. Before run B is compared with run A, the settings
the comparison is made at had to be chosen on run A's own weights.

## Method

- **Model:** run A's Q8_0 and Q4_K_M GGUFs, unchanged, served by the llama.cpp
  `llama-server --jinja -ngl 999 -c 32768 -np 4` Unsloth Studio installed. One server at a
  time.
- **Card:** run A's exact training-time tool card (`card.txt`, sha256 `70697ece…`). The live
  card had changed since run A trained, and scoring run A against a card it never saw
  would measure the drift, not the sampler. It was recovered from `out/run-A/sample_rendered.txt`.
- **Rows:** the 227 run A val rows by id (`val-ids.txt`), plus the five held-out
  `clay-agent-v1` subjects = 232. Run B's val set is a superset, so every run B eval passes
  `--ids val-ids.txt` to stay comparable.
- **Settings:** greedy (`t0 k1`), `t0.2` and `t1.0` (both k64 p0.95), max 4,096 tokens,
  seed 0. The two sampled settings drew three generations a row (`--samples 3`). A cell
  below is the mean over the samples. The per-sample range and the count of rows whose
  outcome split between samples show how much a single run can move.
- **Scoring:** unchanged from run A. The reply's fenced JSON is replayed as one `clay_batch`
  through `gen.verify`, the real door plus `clay_diagnose` and the ground/bounds checks.

## Results

Build = the 174 non-corpus build rows; corpus = the five held-out subjects. The two
"original" rows are run A's own single-sample evals, already published, repeated for scale.

| quant | settings | all (232) | build (174) | edit (40) | corpus (5) | no parse | refused | failed check | per-sample totals | rows split |
|---|---|---|---|---|---|---|---|---|---|---|
| Q8_0 | t1.0 n1 (original) | 178 | 128 | 35 | 4 | 4 | 29 | 19 | — | — |
| Q8_0 | t0 greedy | 173 | 127 | 32 | 3 | 6 | 30 | 21 | — | — |
| Q8_0 | **t0.2 n3** | **185.0** | **140.7** | 29.7 | 3.7 | 6.0 | 23.3 | 15.7 | 183–188 | 81 |
| Q8_0 | t1.0 n3 | 177.7 | 130.0 | 33.0 | 3.3 | 5.3 | 25.7 | 21.7 | 176–180 | 96 |
| Q4_K_M | t1.0 n1 (original) | 132 | 92 | 27 | 2 | 20 | 56 | 22 | — | — |
| Q4_K_M | t0 greedy | 133 | 92 | 25 | 3 | 16 | 50 | 33 | — | — |
| Q4_K_M | t0.2 n3 | 139.0 | 97.7 | 27.0 | 2.7 | 20.7 | 49.7 | 21.3 | 133–142 | 125 |
| Q4_K_M | t1.0 n3 | 124.0 | 83.0 | 27.3 | 2.7 | 22.7 | 61.7 | 21.7 | 117–128 | 132 |

Per sample, the three draws of each n3 setting:

| quant | settings | build | edit |
|---|---|---|---|
| Q8_0 | t0.2 | 139 / 141 / 142 | 29 / 28 / 32 |
| Q8_0 | t1.0 | 130 / 129 / 131 | 32 / 35 / 32 |
| Q4_K_M | t0.2 | 101 / 90 / 102 | 27 / 28 / 26 |
| Q4_K_M | t1.0 | 87 / 81 / 81 | 27 / 32 / 23 |

## What it says

- **Low temperature helps builds, and the gap is bigger than the noise.** At Q8_0 every
  t0.2 build sample (139–142) beats every t1.0 sample (129–131). The draws within each
  setting sit two or three rows apart, and the gap between settings is about ten. Builds
  are long batches where a single bad token (a misspelled `$ref`, a wrong param key) is
  refused, so the long tail of t1.0 costs rows.
- **Edits do not follow, but they are within noise.** At Q8_0, t0.2 edit samples are 28–32
  and t1.0 are 32–35. That range overlaps on 40 rows, and the per-sample spread (4 rows at
  t0.2) is as large as the difference. It is a thing to watch on run B, not a reason to
  pick t1.0.
- **Greedy is not the best setting.** Q8_0 greedy (173) is below both sampled settings. It
  also has more refusals and failed checks than t0.2. A small amount of sampling breaks
  some repeated-token loops a greedy decode gets stuck in.
- **About a third of the rows are a coin flip.** Even at Q8_0 t0.2, 81 of 232 rows split
  between accepted and not across three samples (96 at t1.0). A single-sample difference
  of under ten rows between two models is not evidence of anything. Run B is scored n3.
- **Q4_K_M is worse under every setting,** 44–61 rows behind Q8_0 at the same settings,
  with a no-parse count (16–23) that does not fall at low temperature. The loss is in the
  quant, not the sampler. Q4_K_M is not a candidate.

## Held-out subjects at Q8_0 t0.2

Outcomes over the three samples. The renders (`t0.2/<subject>.png`, from each subject's
first sample, replayed with `eval/render_corpus.py`) are described below, not graded.

| subject | three samples | first sample's render |
|---|---|---|
| chair, slatted back | accepted ×3 | a seat on four splayed legs; the slats lie along one edge of the seat rather than standing up as a back |
| bracket, two holes, gusset | accepted, accepted, failed check | a flat plate with one cut-out and two half-cylinders at its edges; no gusset |
| telescope on tripod | failed check, accepted, failed check | a floating disc, a vertical rod through a cylinder, and two detached legs |
| colonnade, eight columns | accepted ×3 | a square base with four fused slabs standing on it, not a ring of eight columns |
| serpentine creature | accepted, failed check, accepted | an upright stack of two cylinders with four spheres at its sides; not a body along a path |

Same reading as run A's: rule 1 (buildable) is mostly met, and what gets built is not the
subject except for the chair. Lower temperature does not fix composition. That is a
dataset lesson, and run B's composition and figure rows are aimed at it.

## Decision

- **Run B is evaluated at t0.2, top-k 64, top-p 0.95, three samples a row, seed 0**, with
  `--ids val-ids.txt --corpus`. The run A comparison baseline is the Q8_0 t0.2 n3 file
  above (185.0 / 232), not run A's original single-sample 178.
- **Q8_0 is the minimum quantisation, for run B and after** (the user's decision,
  2026-09-13). Q4_K_M is worse at every setting here, so nothing smaller is exported,
  evaluated or shipped; run B exports and evaluates Q8_0 only.

## Files

`data/clay-assistant/run-A/eval-A-{q8,q4}-{t0,t0.2-n3,t1-n3}.json` hold every reply
verbatim, with the `settings` block (card hash, ids file, sampler) recorded in each.
`eval/compare.py <a> <b> --reasons` gives the transition matrix and the per-row flips
between any two of them.
