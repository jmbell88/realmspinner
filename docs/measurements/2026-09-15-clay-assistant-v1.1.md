# Clay assistant v1.1: a noise floor, a dataset winner, and a pause — 2026-09-15

**Status: three arms trained and measured (machine-scored door acceptance). None displaces
run A, and the user paused the programme the same evening it was run. `familiar_v1.0` (run
A, card `70697ece…`) stays the shipped model — nothing here changes it.** No person has
graded anything here. Tier one only, `training/clay-assistant/eval/run_val.py` at the
settings fixed in
[`2026-09-13-clay-assistant-sampling.md`](2026-09-13-clay-assistant-sampling.md): Q8_0,
temperature 0.2, top-k 64, top-p 0.95, three samples a row, seed 0, run A's 227 val ids
(`data/clay-assistant/run-A/val-ids.txt`) plus the five held-out corpus subjects (232),
each model against its own training card. Files are under
[`data/clay-assistant/run-A2/`](data/clay-assistant/run-A2/),
[`run-D1/`](data/clay-assistant/run-D1/) and [`run-D2/`](data/clay-assistant/run-D2/).

## Why this was run

[Run B](2026-09-14-clay-assistant-run-B.md) and the
[ablation](2026-09-14-clay-assistant-ablation.md) left one question the earlier work called
out explicitly: run A was trained exactly once, so part of its 185.0 lead over every other
arm might be run-to-run variance rather than anything a data or card change actually cost.
The user opened a v1.1 attempt (their own plan, not tracked in this repository) to answer
that and to try one more dataset mix. Three arms were built, all on run A's exact recipe
(LoRA r16/α32, responses-only loss, lr 2e-4 cosine, 3 epochs, batch 2×8, bf16, Q8_0 export)
— only the seed, the data, or the card vary:

| arm | dataset | card | seed |
|---|---|---|---|
| A2 | run A's 1,932/227 rows, unchanged | run A's, `70697ece…` | 4207 (not run A's 3407) |
| D1 | A's rows + 136 figure rows (no grounding, no composition), 2,068/241 | today's live card, `cfa30687…` | 3407 |
| D2 | A's rows + 138 grounding rows (no figures, no composition), 2,070/241 | `cfa30687…` | 3407 |

**A2** is the repeat the ablation asked for: same data, same card, a different seed, so any
gap from run A is recipe noise and nothing else. **D1** and **D2** each add back one of run
B's three new families the ablation never isolated on its own — B mixed grounding, figures
and composition together, and C1 paired B's whole dataset with A's card while C2 paired A's
dataset with B's card — swapping whole data and whole card as units, never splitting B's
data by family. **D3** (composition rows alone) was built as data
(`training/clay-assistant/out/run-D3/data`, 2,025/234) but never trained: after D2 the user
chose to skip D3 and treat D1 as the Phase 4 candidate, then paused the programme before a
fourth training run and kept familiar_v1.0, rather than schedule a repair-turn tier two.
Arms were built with `train/make_arm.py`; A2's data was checked byte-identical to run A's
tracked dataset at `e79b9c64` (`--expect-rev`).

## Training

| | A2 | D1 | D2 |
|---|---|---|---|
| wall time | 3,341 s | 7,921 s | 7,640 s (see below) |
| eval loss | 0.2235 | 0.2145 | 0.211 |
| export | Q8_0 | Q8_0 | Q8_0, 4,967,497,024 B |

D2's first launch (07:28) died at step 1 of 390: Windows Update restarted the machine at
07:40 (System event 1074, TrustedInstaller), not anything in the recipe. It was relaunched
at 15:25 and ran to completion; the wall time above is the completed run.

Wall time still does not track row count, the same finding the 2026-09-13/14 documents
made and did not investigate: A2 trained in 3,341 s on run A's exact 1,932 rows, while D1
and D2 took roughly twice that on datasets only about 7% larger with longer card rows. That
gap is bigger than the row-count difference explains. Not investigated here either.

## The measurement

Failure counts below were recomputed from scratch with one script over all seven eval
files (`docs/measurements/data/clay-assistant/run-{A,B,C1,C2}/eval-*.json` and
`training/clay-assistant/out/run-{A2,D1,D2}/eval-*.json`), so every run in this table is
counted the same way: accepted-mean over 232 rows; the build mean restricted to
`kind == "build"` rows excluding `family == "corpus"` (174 rows, matching the earlier
docs' "build (174)" column); and, over all non-accepted samples, how many mention "below
the ground plane" and how many mention "no object named" (split into the `creatures`
family and elsewhere). **This recomputation matches every number the 2026-09-14 documents
already published for A/B/C1/C2 exactly** — below-ground 45/61/60/33 and creature
part-names 16/6/26/20 — so there is no disagreement to report; the same accounting is
just extended to the three new arms.

| model | data | card | seed | all (232) | build (174) | edit (40) | query (13) | corpus (5) | build per-sample |
|---|---|---|---|---|---|---|---|---|---|
| A | A | A | 3407 | **185.0** | 140.7 | 29.7 | 11.0 | 3.7 | 139/141/142 |
| B | B | B | 3407 | 168.0 | 125.0 | 28.7 | 11.0 | 3.3 | 127/128/120 |
| C1 | B | A | 3407 | 169.0 | 125.7 | 29.3 | 11.0 | 3.0 | 126/129/122 |
| C2 | A | B | 3407 | 174.0 | 128.0 | 30.7 | 11.7 | 3.7 | 129/128/127 |
| A2 | A | A | **4207** | 175.3 | 130.0 | 30.7 | 11.0 | 3.7 | 124/129/137 |
| D1 | A+figures | B | 3407 | **180.3** | 133.7 | 30.7 | 11.7 | 4.3 | 132/138/131 |
| D2 | A+grounding | B | 3407 | 172.7 | 128.0 | 30.3 | 11.0 | 3.3 | 128/129/127 |

| model | below ground plane | no object named: creature part | no object named: elsewhere |
|---|---|---|---|
| A | 45 | 16 | 16 |
| B | 61 | 6 | 43 |
| C1 | 60 | 26 | 29 |
| C2 | 33 | 20 | 29 |
| A2 | 53 | 20 | 27 |
| D1 | 60 | 7 | 14 |
| D2 | 55 | 12 | 33 |

(The recomputation script also gives each run's total "no object named" count — the sum
of the last two columns — as a cross-check: A 32, B 49, C1 55, C2 49, A2 47, D1 21, D2 45,
each equal to the sum of its own row above.)

## What this says, and does not

1. **The noise floor is as large as the effect being measured.** A2 is run A's own recipe
   repeated under a different seed, on the same 1,932 rows and the same card, and it scored
   175.3 — 9.7 below run A. The spread spans 13 builds, and its low end (124) sits among
   B's, C1's and C2's samples (120–129) while its high end (137) is close to run A's
   (139–142). A2's 9.7-point gap to run A is not much smaller than B's, C1's and C2's own
   gaps to it (11–17 points). A single training run's score is a noisy draw, not a fixed
   property of its data and card, and every earlier single-draw comparison in this programme
   (A vs B, C1 vs C2, and A's own 185.0 read as a ceiling) is much less certain than it
   reads. Run A's 185.0 may simply be the top of that noisy distribution rather than a
   repeatable property of its recipe.
2. **D1's figure rows plus today's card roughly halve the creature-part "no object named"
   failures without paying grounding's or composition's cost**, and it is the best score
   since run A: 180.3. Its build mean (133.7) is above A2's 130.0, and every one of its
   build samples is above every build sample B, C1 and C2 produced (their highest is 129) —
   while still overlapping A2's own spread, so the lead over A2 is within the noise finding
   1 describes. It reproduces run B's one clean targeted gain (7 vs A2's 20, close to run
   B's own 6) while avoiding the drag the full B mix carried.
3. **D2's grounding rows alone do not survive, and do not fix the ground-plane count.** D2
   scored 172.7 — below C2's 174.0 and below every one of D1's build per-sample counts —
   despite adding exactly the rows run B's own write-up suspected of teaching a wrong
   capsule placement. Below-ground failures range from 33 to 61 with no consistent relation to
   grounding rows — runs without any (C2 33, A 45, A2 53, D1 60) span almost the whole
   range, and runs with them (D2 55, C1 60, B 61) sit inside it — so it reads as noise, not
   something grounding rows fix or break. The grounding recipe's own capsule placement was
   checked against this concern: `training/clay-assistant/drafts/_gen_grounding.py`'s
   `capsule_drop_y` places an upright capsule at `cyl_height/2 + radius`, which is correct
   for a capsule built as a cylinder of that height capped by hemispheres of that radius.
   Run B's theory that the grounding rows taught the wrong placement is not supported by
   either the code or this count.
4. **Outcome and where to pick this up.** The user chose to skip D3 (composition rows
   alone, data already built but never trained) and pause the programme the same evening,
   keeping `familiar_v1.0` (run A) as the shipped model. If the programme resumes: D1's mix
   is the best-measured starting point; given finding 1, any single arm needs at least one
   repeat before its score is trusted; D3 is unmeasured; and tier two has only been run
   once, on run A, with one attempt per subject and no repair turns; `eval/tier_two.py` has
   no repair-turn support yet, and no arm since run A has had a tier-two session.

## Corpus renders

`eval/render_corpus.py` runs headless on the CPU (`gen.headless.open_gl()` opens a
standalone moderngl context; no llama-server, no GPU inference) and ran clean for all three
arms under a throwaway `WARLOCK_HOME`, replaying each eval file's five corpus rows through
the real door: A2 (chair 12 objects, bracket 2, telescope 4, colonnade 12, serpent 7), D1
(chair 6, bracket 2, telescope 5, colonnade 7, serpent 7), D2 (chair 10, bracket 1,
telescope 6, colonnade 72, serpent 7). All five replayed `ok` for every arm; none refused
outright. The `.calls.json` and `.png` files are kept alongside each arm's other data,
matching `run-C1`/`run-C2`.

## Retention

Kept under `data/clay-assistant/run-A2/`, `run-D1/`, `run-D2/`: `arm.json`, `config.json`,
`train_result.json`, `trainer_state.json`, the eval JSON (every reply verbatim), the five
corpus `*.calls.json` files and their renders, and the training card each arm actually
trained on — A2's is a copy of run A's card (`card.txt`, sha256 `70697ece…`, verified), D1's
and D2's are a copy of run B's card (sha256 `cfa30687…`, verified). Renders are gitignored
by `docs/measurements/data/**/*.png`, same as every earlier arm. Each arm's adapter, merged
model and GGUF stay gitignored under `training/clay-assistant/out/run-A2/`,
`training/clay-assistant/out/run-D1/` and `training/clay-assistant/out/run-D2/`. D3's data
(`training/clay-assistant/out/run-D3/data`) also stays there, untrained and unretained here,
since nothing was ever run against it.
