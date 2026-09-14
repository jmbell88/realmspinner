# Clay assistant, ablation C1/C2 and a first tier-two session — 2026-09-14

**Status: tier one measured for both arms (machine-scored door acceptance), tier two run
once for run A. Neither arm displaces run A, so run A stays the candidate.** No person has
graded anything here. Every tier-one number comes from `training/clay-assistant/eval/run_val.py`
at the settings in [`2026-09-13-clay-assistant-sampling.md`](2026-09-13-clay-assistant-sampling.md).
The files are under [`data/clay-assistant/run-C1/`](data/clay-assistant/run-C1/),
[`run-C2/`](data/clay-assistant/run-C2/) and [`run-A/tier-two/`](data/clay-assistant/run-A/tier-two/).

## Why this was run

[Run B](2026-09-14-clay-assistant-run-B.md) scored 168.0/232 against run A's 185.0. It
changed the tool card and the dataset at once, so it could not say which change cost the
builds. This matters beyond the model: master's live card is run B's, and Familiar's T3
(`TODO.md` P54) has to freeze one card. Two arms each change one thing:

| arm | dataset | card |
|---|---|---|
| A | A's 2,159 rows (1,932 train / 227 val) | A's, sha `70697ece…`, 8.1k chars |
| B | B's 2,561 rows (2,299 / 262): A's rows + 152 grounding, 150 figures, 100 composition | B's, sha `cfa30687…`, 12.4k chars |
| **C1** | B's | A's |
| **C2** | A's | B's |

The arms were built by `train/make_arm.py`. Because the loss covers responses only and
every row starts with the same card, swapping the card is a rewrite of each row's system
turn. Nothing was regenerated. C2's kept records were checked against run A's tracked
dataset at `e79b9c64`: identical ids, and 0 records differing beyond
`verified.tools_sha`. Each arm's `arm.json` records its hashes and counts. The recipe is
run A's `CONFIG`, unchanged. `train_a.py` gained an optional dataset directory and now
records the card hash it trained on.

**Decision rule, fixed before any result:** an arm displaces A only if its mean total
beats 185.0 **and** all three of its per-sample build counts beat A's (139, 141, 142).

## Training

| | C1 | C2 |
|---|---|---|
| steps | 432, 4,118 s | 363, 7,163 s |
| eval loss by epoch | 0.2843 → 0.2116 → 0.1964 (B's 262 val rows; B: 0.2818 → 0.2088 → 0.1948) | 0.3136 → 0.2357 → 0.2199 (A's 227 val rows) |
| train loss | 0.2596 | 0.2771 |
| exports | merged 16-bit, Q8_0 4.63 GiB | merged 16-bit, Q8_0 4.63 GiB |

Both merges were spot-checked like run B's. Layer 0's `mlp.down_proj` differs from both A
and B, and `embed_tokens` is identical to both.

Wall time does not follow row count. C1 trained on exactly run B's rows in less than half
B's 8,504 s, and C2 on exactly run A's rows took twice A's 3,451 s. Nothing in the recipe
differs, so this is machine state. It is not investigated here.

The first launch of C2 refused to start because the chain's GPU check ran before C1's
killed llama-server had exited. The chain script now waits for the server to exit. No
training time was lost.

## The measurement

Run A's 227 val rows by id, plus the five held-out `clay-agent-v1` subjects (232). Q8_0,
temperature 0.2, top-k 64, top-p 0.95, three samples a row, seed 0, each model against its
own training card. A cell is the mean over the three samples. "No parse" here includes
"fenced or empty".

| model | data | card | all (232) | build (174) | edit (40) | query (13) | corpus (5) | refused | failed check | no parse |
|---|---|---|---|---|---|---|---|---|---|---|
| A | A | A | **185.0** | **140.7** | 29.7 | 11.0 | 3.7 | 23.3 | 15.7 | 8.0 |
| B | B | B | 168.0 | 125.0 | 28.7 | 11.0 | 3.3 | 35.7 | 20.7 | 7.7 |
| C1 | B | A | 169.0 | 125.7 | 29.3 | 11.0 | 3.0 | 35.7 | 20.7 | 6.7 |
| C2 | A | B | 174.0 | 128.0 | 30.7 | 11.7 | 3.7 | 38.3 | 11.3 | 8.3 |

Per-sample build counts (of 174):

| model | builds | edits |
|---|---|---|
| A | 139 / 141 / 142 | 29 / 28 / 32 |
| B | 127 / 128 / 120 | 29 / 28 / 29 |
| C1 | 126 / 129 / 122 | 29 / 27 / 32 |
| C2 | 129 / 128 / 127 | 30 / 31 / 31 |

No arm's best build sample reaches A's worst, so **neither arm meets the rule, and run A
stays the candidate.** Edits and queries are flat across all four models. Rows whose
outcome splits across the three samples: A 81, B 107, C1 98, C2 82.

| family | n | A | B | C1 | C2 |
|---|---|---|---|---|---|
| mechanical | 35 | 28.0 | 22.7 | 23.7 | 26.3 |
| furniture | 35 | 31.3 | 27.7 | 26.7 | 28.3 |
| vehicles | 19 | 15.3 | 12.3 | 15.7 | 12.7 |
| containers | 34 | 28.3 | 26.3 | 25.7 | 28.7 |
| architecture | 29 | 24.0 | 23.0 | 25.3 | 23.3 |
| creatures | 22 | 13.7 | 13.0 | 8.7 | 8.7 |
| edits | 40 | 29.7 | 28.7 | 29.3 | 30.7 |
| queries | 13 | 11.0 | 11.0 | 11.0 | 11.7 |
| corpus | 5 | 3.7 | 3.3 | 3.0 | 3.7 |

## Where the failures moved (all three samples)

| cause | A | B | C1 | C2 | reading |
|---|---|---|---|---|---|
| `no object named` a creature part | 16 | **6** | 26 | 20 | only B, with **both** the figure rows and the card's part catalogue, fixed part names. Either alone was worse than neither |
| `no object named` elsewhere (mostly `.00N` copy names) | 16 | 43 | 29 | 29 | each change raises it, and B's pair raises it most |
| below the ground plane | 45 | 61 | 60 | 33 | tracks **B's data**: about 60 with it, 33–45 without. This fits run B's unconfirmed suspicion that the grounding rows teach the wrong capsule placement |

## What this says, and does not

- **Each single change costs about as much as both together.** From A, B's card alone
  costs 11.0 (C2), B's data alone costs 16.0 (C1), and both together cost 17.0 (B). The
  effects are not additive. On B's data, adding B's card costs only 1.0 more (C1 → B). On
  B's card, adding B's data costs 6.0 more (C2 → B).
- **That pattern is also what a lucky run A would look like.** Every model that differs
  from A in any way lands between 168 and 174. Run A was trained once. Its seed is fixed,
  but a fixed seed with different rows is not a repeat. Until A's exact recipe is trained
  again, part of A's lead may be run-to-run variance and not anything B changed. That
  second training of A (A's data, A's card) is the one run that would separate the two.
  It is not scheduled.
- **The one mechanism the ablation isolates cleanly is the ground plane.** Below-ground
  failures follow B's data regardless of card. The grounding recipe is where to look
  before any of those rows are reused.
- **For P54's card choice:** the kept model was trained on card `70697ece`. Nothing here
  argues for training on B's card: C2 (B's card on A's data) lost 11.0. The choice stays
  the user's.
- As before, eval loss said nothing about door acceptance.

## The five held-out subjects, tier one (Q8_0, first sample rendered)

| subject | C1: three samples | C1 render | C2: three samples | C2 render |
|---|---|---|---|---|
| chair, slatted back | accepted, refused, accepted | a four-legged stool with a single rail floating above and behind the seat | accepted ×3 | a stool with a slatted top and no back |
| bracket, two holes, gusset | accepted ×3 | one flat plate with a small slot | refused, refused, accepted | two plates meeting at a right angle, no holes, no gusset |
| telescope on tripod | accepted ×3 | a tube standing above three splayed legs that do not meet it | accepted ×3 | four thin vertical rods |
| colonnade, eight columns | refused, refused, accepted | a stepped pyramid, no columns | accepted ×3 | about nine columns scattered beside a round base, not a ring |
| serpentine creature | failed check, failed check, refused | a thick coiled body | accepted, refused, refused | a long thin chain of capsules lying flat |

None is its subject. The C1 telescope and the C1 serpent are the closest to it.

## Tier two: run A drives the real app

The pre-registration's tier two had never been run. `eval/tier_two.py` is a minimal MCP
client. It spawns `warlock mcp` against a real Studio started by
`scripts/agent_bench.py --serve` in a throwaway home, and speaks newline JSON-RPC over the
bridge and pipe. For each corpus subject it sends one chat turn to run A's Q8_0 with A's
card (t0.2, seed 0, one sample), sends one `clay_batch`, and checks the result. The check
mirrors `verify.check` using `clay_scene` and `clay_diagnose`. Two rules cannot be reached
over MCP and are recorded as not reproduced: rule 3a (diagnose findings run directly on
the mesh) and rule 6 (the `.wblk` save-and-reload test). It then renders the scene lit and
clears it. There are no repair turns.

| subject | tier two | tier one, first sample | run A tier-two render (lit) |
|---|---|---|---|
| chair, slatted back | accepted, 11 calls, 10 objects | accepted | **a recognisable chair**: seat, four splayed legs, a back panel, with its slats floating beside the panel |
| bracket, two holes, gusset | accepted, 5 calls, 1 object | accepted | one L-shaped plate, no holes, no gusset |
| telescope on tripod | accepted, 6 calls, 5 objects | failed check (tube below ground) | a long rod across a disc, with three loose rods beneath |
| colonnade, eight columns | accepted, 7 calls, 11 objects | accepted | one fluted column on a stepped plinth |
| serpentine creature | failed check (body and legs below ground) | accepted | a lumpy mass on stubby legs |

- **The pipe added nothing wrong.** The recorded transcript is 30 calls, all `ok`, 163 uids
  produced. There was no framing, envelope or tab-per-connection failure.
- **The two disagreements are sampling, not a door defect.** None of the five tier-two
  replies is byte-identical to any of tier one's three samples. The server ran one slot
  (`-np 1`) against tier one's four. Tier one rates telescope and serpent at 1/3 and 2/3
  accepted, so a single draw landing either way is expected. Replaying a tier-one reply
  over the wire would be a true parity test. That was not done.
- 4 of 5 accepted, and the chair is the only render that reads as its subject. Whether any
  of the five meets the pre-registered bar is tier three (P43), a person grading renders.

## Retention

Kept under `data/clay-assistant/`: each arm's card, `arm.json`, `config.json`,
`train_result.json`, `trainer_state.json`, eval JSON (every reply verbatim) and corpus call
lists; run A's `tier-two/tier-two.json` and the recorded MCP transcript `bench.jsonl`.
Renders are gitignored by design, and the tier-two PNGs stay under
`training/clay-assistant/out/run-A/tier-two/`. Each arm's adapter, merged model and GGUFs
(about 24 GiB each) stay gitignored under `training/clay-assistant/out/run-C1/` and
`training/clay-assistant/out/run-C2/`.
