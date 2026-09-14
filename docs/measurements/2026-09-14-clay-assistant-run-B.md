# Clay assistant, run B: a negative result, and run A stays the candidate — 2026-09-14

**Status: tier one measured (door acceptance, machine-scored). Run B scored below run A,
and run A stays the candidate (the user's decision, 2026-09-14).** No person has graded
anything here. Every number comes from `training/clay-assistant/eval/run_val.py` at the
settings chosen in [`2026-09-13-clay-assistant-sampling.md`](2026-09-13-clay-assistant-sampling.md).
The files are kept under [`data/clay-assistant/run-B/`](data/clay-assistant/run-B/).

## What run B changed

Run B kept run A's recipe exactly (LoRA r=16 α=32 on bf16, responses-only loss, lr 2e-4
cosine, 3 epochs, batch 2 × accum 8, seed 3407). It changed two things at once:

| | run A | run B |
|---|---|---|
| tool card | 8.1k chars, sha `70697ece…` | 12.4k chars, sha `cfa30687…`: adds generator params, op params, and every `clay_add_figure` part name |
| dataset | 2,159 rows (1,932 train / 227 val) | 2,561 rows (2,299 train / 262 val): +152 grounding, +150 figures, +100 composition |
| row length | 2,190–4,926 tokens (median 2,430) | 4,086–7,433 tokens (median 4,313) |

The new rows aimed at run A's two largest refusal causes: figure part names, and
profile generators placed below the ground plane. Every one was replayed clean through
the door before training, and all 11 recipes regenerate byte-identically.

## Training

| | |
|---|---|
| steps | 432, 8,504 s on the RTX 5090 (run A: 363 steps, 3,451 s) |
| eval loss by epoch | 0.2818 → 0.2088 → 0.1948 (on B's 262 val rows; not comparable to A's 0.224 on 227) |
| train loss | 0.2531 mean |
| exports | merged 16-bit (`train/merge.py`); GGUF BF16 8.67 GiB, **Q8_0 4.63 GiB**. Q8_0 is the quantisation floor from 2026-09-13, so nothing smaller was made |

Getting a finished run took five attempts, and two things went wrong on the way:

- **The epoch-end eval crashed twice at step 144**, once with a llama-server sharing the
  card and once with the card to itself. Unsloth's fused cross entropy sizes its chunks
  from `torch.cuda.mem_get_info` free memory. After an epoch of these long rows the
  caching allocator had reserved about 29 of 32 GB, so free read zero and it raised "No or
  negligible GPU memory available". A reproduction (fill the card, drop the tensors
  without `empty_cache`, evaluate the eight longest val rows) failed the same way.
  Emptying the cache before eval did **not** cure it. `UNSLOTH_CE_LOSS_TARGET_GB=4` did:
  the eval passed at 0.00 GiB free. 4 is the cap the unset path already takes on this
  card, so chunking is unchanged. `train_a.py` now sets it.
- **The finished run segfaulted on interpreter exit (rc 139)**, after the adapter,
  `checkpoint-432` and the trainer state were all written. That stopped the chain before
  merge. The merge the next day confirmed the adapter was intact: B's merged weights
  differ from A's in the LoRA-targeted MLP tensors and match A's in the untrained
  embeddings. The segfault and a reported peak of 51.8 GiB on a 32 GB card are both
  unexplained. The second is probably Windows' shared-memory fallback.

## The measurement

Run A's 227 val rows by id, plus the five held-out `clay-agent-v1` subjects (232). Q8_0,
temperature 0.2, top-k 64, top-p 0.95, three samples a row, seed 0, each run against its
own training card. A cell is the mean over the three samples.

| model | all (232) | build (174) | edit (40) | query (13) | corpus (5) | refused | failed check | no parse |
|---|---|---|---|---|---|---|---|---|
| run A, Q8_0 | **185.0** | **140.7** | 29.7 | 11.0 | 3.7 | 23.3 | 15.7 | 6.0 |
| run B, Q8_0 | 168.0 | 125.0 | 28.7 | 11.0 | 3.3 | 35.7 | 20.7 | 5.7 |

Per sample, builds: A 139 / 141 / 142, B 127 / 128 / 120. The two ranges do not meet, so
the build loss is not sampling noise. Edits (A 29 / 28 / 32, B 29 / 28 / 29) and queries
are unchanged. Rows whose outcome splits across the three samples: A 81, B 107.

| family | n | run A | run B | Δ |
|---|---|---|---|---|
| mechanical | 35 | 28.0 | 22.7 | −5.3 |
| furniture | 35 | 31.3 | 27.7 | −3.7 |
| vehicles | 19 | 15.3 | 12.3 | −3.0 |
| containers | 34 | 28.3 | 26.3 | −2.0 |
| architecture | 29 | 24.0 | 23.0 | −1.0 |
| edits | 40 | 29.7 | 28.7 | −1.0 |
| creatures | 22 | 13.7 | 13.0 | −0.7 |
| corpus | 5 | 3.7 | 3.3 | −0.3 |
| queries | 13 | 11.0 | 11.0 | 0 |

Run B lost ground in every build family. It lost the least in creatures, the family its
figure rows were written for, and the most in mechanical and furniture, where no new rows
were written.

## Where the failures moved (all three samples)

| cause | run A | run B | reading |
|---|---|---|---|
| `no object named` a creature part | 16 | **6** | the figure rows did what they were for |
| `no object named` elsewhere (`leg.004`, `slat.004`, `column.001`, `flagpole_pole`) | 16 | **43** | mostly invented `.00N` copy names in furniture, mechanical and architecture. No new row names a `.00N` copy at all, so this was not copied from the new rows |
| below the ground plane | 45 | **61** | capsules 6 → 20. The grounding rows' capsule placement was not established (`capsule`'s `height` is the cylinder alone, so a grounded capsule sits at `h/2 + r`) |
| envelope errors | 1 | **12** | A: one `rotation` as a batch-entry key. B: `$ref` (4) and `rollback_on_error` (1) as batch-entry keys, and 7 references to a uid that does not exist |

One door crash appeared during scoring: a `clay_op` given a list where a number was due
raised `TypeError` at `clay_ops.run`'s clamp instead of refusing. It is `TODO.md` F9, and
it scores as a failure either way.

## The five held-out subjects, Q8_0

| subject | three samples | first accepted sample's render |
|---|---|---|
| chair, slatted back | refused, accepted, accepted | three legs under a seat, with the back lying flat and sticking out sideways |
| bracket, two holes, gusset | refused, accepted, accepted | one L-shaped solid, no holes, no gusset |
| telescope on tripod | accepted ×3 | scattered thin rods: splayed legs, a crossbar, nothing tube-like |
| colonnade, eight columns | accepted ×3 | about five columns bunched on overlapping discs, not a ring of eight |
| serpentine creature | failed check ×3 | two tapered slabs and a short rod |

None is its subject. Run A's chair was one, at this setting and the one before.
Renders are in `data/clay-assistant/run-B/<subject>.png`; the first sample of each was
replayed with `eval/render_corpus.py`.

## What this says, and does not

- **Run B is worse than run A at what matters most, one-shot builds, by about 16 of 174,
  outside noise.** Run A stays the candidate. Nothing downstream should move to run B's
  weights or card.
- **The run cannot say why.** The card grew by half and the dataset grew by a fifth at
  the same time. Either could cost a 2B model its build accuracy: a longer system turn
  to attend past on every call, or a data mix that shifts weight away from the families
  that regressed. The only targeted gain (creature part names) is consistent with the
  card's part catalogue and the figure rows helping, and does not separate the two.
- **More rows of the same kind are not the obvious next step.** Separating the causes
  would take one more training run that changes one thing: B's dataset with A's card, or
  A's dataset with B's card. That is deliberately not scheduled.
- Eval loss fell further than run A's (0.1948) but on a different val set. As with run A,
  loss said nothing about door acceptance.

## Retention

Run B's eval JSON (every reply verbatim), the corpus call lists and renders, the training
card, `config.json`, `train_result.json` and `trainer_state.json` are kept under
`data/clay-assistant/run-B/`. The adapter, merged model and GGUFs (~24 GiB) stay
gitignored under `training/clay-assistant/out/run-B/` on this machine, so an ablation can
start from them.
