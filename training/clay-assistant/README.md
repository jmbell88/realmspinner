# Clay assistant: dataset generator and verifier (Phase 1)

Phase 1 scaffold of the Clay-assistant training programme -- see the approved
plan (`we-are-going-to-cosmic-dream.md`, sections "Phase 1", "Verifier
design" and "Phase 0 result") for the programme this is one piece of: a
small Gemma, fine-tuned to drive Warlock's Clay mode over the real
`agent_clay` MCP tool surface, for a later (not-yet-built) offline "live"
assistant. **This directory is the dataset side only** -- generation,
verification, conversion to training shapes. No `src/` integration, no
registry row, no worker; those belong to a later plan.

## What this is not

Nothing here ships with the app, imports into `src/`, or runs at app
runtime. `pytest`'s own `testpaths` is `["tests"]`, so nothing under
`training/` is collected by a bare `uv run pytest`; this package has its own
tiny suite (see "Running the tests" below).

## The held-out rule

`docs/measurements/corpora/clay-agent-v1.txt` is the pre-registered tier-two
eval for this whole programme (a chair, a mechanical bracket, a telescope, a
colonnade, a serpentine creature). **No training row may be built from one
of those five subjects, or from a close paraphrase of one** -- a fine-tune
that had seen the benchmark would not be measuring what it claims to.

`gen/holdout.py`'s `is_leak(prompt, holdout)` catches three disguises:

1. an exact match once both prompts are normalised (case, punctuation,
   whitespace);
2. a token-set Jaccard similarity of 0.6 or more;
3. a shared "distinctive" bigram -- the corpus line's own content words,
   adjacent, with a short stopword list dropped first (so "a wooden chair
   with ..." and "... a wooden chair, four legs ..." are caught by the bare
   phrase `("wooden", "chair")` even when the surrounding words differ enough
   to dodge the Jaccard check).

`build.py` runs every draft prompt through this before it is ever replayed,
and refuses outright if `--drafts` is pointed anywhere under
`docs/measurements/corpora/` -- that directory is a read-only eval, never a
source of drafts.

## The record schema (`drafts/<family>.jsonl`)

One JSON object per line:

```json
{"id": "furniture-0042", "family": "furniture", "kind": "build",
 "prompt": "a low round coffee table with three splayed legs",
 "calls": [{"name": "clay_add_primitive",
            "arguments": {"generator": "box", "name": "top",
                          "params": {"size": [0.9, 0.04, 0.9]},
                          "translation": [0, 0.42, 0]}},
           "..."],
 "notes": "why these proportions"}
```

* `family` is one of the eight in `gen/schema.py::FAMILIES` (`furniture`,
  `containers`, `architecture`, `mechanical`, `vehicles`, `creatures`,
  `edits`, `queries`) -- also the `drafts/<family>.jsonl` filename and the
  `--only` argument to `build.py`.
* `kind` is `build`, `edit` or `query`.
  * `build`: `calls` is the list of tool calls that build the object. These
    are replayed as **one folded `clay_batch` call** -- the shape the
    trained model will actually emit -- so a later call in the list may
    address an earlier one by `{"$ref": "<name>"}` rather than a uid (see
    `agent_clay._resolve_batch_ref`); no training row ever needs to know a
    uid at all.
  * `edit`: also carries `prior` -- the calls that build the starting
    scene, replayed as its own folded batch first (and which must itself
    verify) -- and `calls` is the edit itself, a second folded batch against
    the same document, addressing the prior scene's objects by name via
    `$ref` or by uid read off a scene turn. An edit that does not move
    `doc.history.head` is rejected: a no-op is not a training example of an
    edit.
  * `query`: carries `prior` (what the question is about) and `answer`
    (prose, no tool call) instead of `calls`.
* `allow_below_ground` (optional, default false): opts a record out of the
  ground-plane check only -- see "What verify.check rejects" below.
* Once a record is accepted, `build.py` adds a `verified` block:
  `{"object_count", "bounds": {"size", "center"}, "diagnose_clean": true,
  "call_count", "tools_sha"}`. `call_count` here is the number of top-level
  door calls the record's own replay made (1 for `build`/`query`, 2 for
  `edit` -- the prior batch, then the edit batch) -- not the length of
  `calls` itself, which is how many sub-calls got folded into that one
  batch, a different and also-interesting number for judging how much one
  training row asks a single call to do.

## What `verify.check` rejects

Every accepted record has genuinely been replayed through the real
`agent_clay.call` door (`gen/verify.py`, a fresh `HeadlessCtx` +
`agent_clay.Session()` per record) and rejected if any of:

1. the record's own batch (or its `prior`) refused, or the batch stopped
   partway (`structuredContent["stopped_at"]` not `None`);
2. no objects exist afterwards;
3. any object has `clay_diagnose` findings -- checked both directly
   (`clay/diagnose.findings`, the function tier one's own replay grades
   against) and through the `clay_diagnose` tool itself, with no `uid`
   (every object must come back `clean: true`);
4. the document's own bounds (from `clay_scene`) are missing, or any
   dimension is under 0.005 m or over 20 m; or any single object has a size
   component at or under 1 mm;
5. any visible object's bbox sits more than 2 cm below the ground plane --
   a named check (`allow_below_ground`), so a family that legitimately
   builds below y=0 can opt out of just this one;
6. the document fails to round-trip through `serialize.wblk_bytes`/
   `read_wblk` with the same object count;
7. for an `edit`: the edit batch did not move `doc.history.head`.

## Message shapes (`--export unsloth`)

`gen/convert.to_messages` builds one ChatML/OpenAI-shaped
`{"messages": [...]}` row per accepted record:

* **`build`**: `system` (the compact tool card) / `user` (the prompt) /
  `assistant`, whose `content` is the batch call's JSON fenced as
  ` ```json ` and which *also* carries a `tool_calls` entry naming
  `clay_batch` with the same JSON as its `arguments` string -- unless
  `--inline` is passed, in which case the assistant turn is fence-only (no
  `tool_calls` key), for a template or trainer with no separate tool-call
  channel.
* **`edit`** / **`query`**: `system` / a `user` turn reading `"Here is the
  scene:\n" + <compact scene JSON>` (the `prior`'s own state, projected
  through `gen/convert.compact_scene`) / `user` (the prompt) / `assistant`
  (a batch tool call for `edit`, prose in `content` with no tool call for
  `query`).

## The compact tool card

Training on the full ~26-tool schema set (~7.4k tokens of schema alone,
`instructions()` another ~1.4k) leaves too little budget for a scene turn
inside an E2B row. `gen/convert.compact_tools()` (since Familiar T3 a re-export of
`warlock.studio.familiar.contract.derive_clay_card()`, which is where the card, the scene
compaction, the user-turn form and `parse_calls` now live, so the app and this directory
cannot drift apart; `src/` never imports `training/`) builds a smaller card
instead: the first two paragraphs of the live `agent_clay.instructions()`,
one fixed paragraph of training-only behaviour ("answer with exactly one
clay_batch call..."), then thirteen tools (`clay_batch`, `clay_scene`,
`clay_add_primitive`, `clay_add_figure`, `clay_transform`, `clay_set_params`,
`clay_material`, `clay_boolean`, `clay_select`, `clay_op`, `clay_delete`,
`clay_rename`, `clay_diagnose`) with their schemas kept **verbatim** (the
generator/op enums are what the model has to get right) and their
descriptions reduced by `gen/convert._summary()` to their first sentence
plus every sentence beginning `Known generators:`, `Known ops:` or `Parts,`
-- the three catalogue enumerations naming a generator's own param names
(`clay_add_primitive`/`clay_set_params`), an op's own param names and bounds
(`clay_op`), and a figure preset's own part names (`clay_add_figure`).

**The third sentence comes from a figure part catalogue.** The `Parts, ...` sentence is
read out of `agent_clay.tools()`'s `clay_add_figure` description. Run B trained with that
catalogue and came out negative (`docs/measurements/2026-09-14-clay-assistant-run-B.md`);
the catalogue was held back from master that day and then landed later the same day, so
this tree's card hashes to the card run B actually trained on (`tools_sha`
`cfa30687...`, kept verbatim at `docs/measurements/data/clay-assistant/run-B/card.txt`)
and `dataset/manifest.json` matches it: `build.py` regenerates `dataset/` without
`--force` again. Run A, still the candidate, was trained on the older `70697ece` card
(`docs/measurements/data/clay-assistant/run-A/card.txt`), which is why its evals pass
`--card`.

That third clause is a 2026-09-13 fix, not the original design: the prior
`_first_sentence` kept the opening line only and trusted the schema's own
enums to carry the rest, which is true for a plain string enum (a
generator's or op's *name*) but not for what a generator's or an op's own
*params* are called (`params` is one open `{string: number}` object in the
wire schema; the value shape varies per key and is never itself an enum),
or for what a figure preset's own part names are (never an argument at
all). Run A's own refusals (2026-09-13, Q8_0, 232 val+corpus rows) are
exactly the gaps this left: 15 `no object named '...'` refusals (nine of
them in the `creatures` family, guessing a generated figure's own part
names -- `hound_Beak`, `t_Shank.R`, `s_Tail 01`), 7 unknown-param refusals
for a generator (e.g. `depth` on a cylinder), and `clay_op` given `axis`
outside `params` among 6 other refusals naming an op's own arguments.
`clay_set_params` repeats `clay_add_primitive`'s own "Known generators: ..."
sentence verbatim (both are built from the same
`agent_clay._generator_catalog()`); `_summary()` keeps it only the first
time it is printed, saving 959 chars.

Built fresh from the live registry every time, never hand-copied: a new
generator, op or figure preset changes this card, and therefore
`tools_sha()` (`sha256` of the card), on the next `build.py` run. `build.py`
refuses to regenerate `dataset/` against a changed `tools_sha` unless
`--force` is passed, so a registry change forces a deliberate regeneration
instead of a dataset that silently no longer matches what it claims to
teach.

## Running `build.py`

```powershell
uv run python training/clay-assistant/gen/build.py
uv run python training/clay-assistant/gen/build.py --only furniture --gallery
uv run python training/clay-assistant/gen/build.py --export unsloth --force
uv run python training/clay-assistant/gen/build.py --promote furniture-0042 --as coffee-table
```

Walks every `drafts/*.jsonl` (or just `<family>.jsonl` with `--only`),
validates each record's shape (`gen/schema.py`), checks it against the
held-out corpus (`gen/holdout.py`), replays it through the real door
(`gen/verify.py`), and writes, sorted by id so a regeneration is
byte-identical:

* `dataset/train.jsonl` / `dataset/val.jsonl` -- accepted records, split
  10% to val per record by a sha256 of its own id (deterministic, not
  random: the same id lands in the same split on every rerun);
* `dataset/rejects.jsonl` -- `{"id", "family", "kind", "reasons"}` for every
  rejected record, `reasons` including the door's own refusal text, so a
  subagent fixing a draft can see exactly why without re-running anything;
* `dataset/manifest.json` -- counts by family and by kind, `tools_sha`,
  the held-out corpus's own sha256, and a histogram of rejection reasons.

`--export unsloth` additionally writes `dataset/unsloth_train.jsonl` /
`unsloth_val.jsonl` via `to_messages`. `--gallery` renders each accepted
record's finished document (three-quarter and front, side by side) to
`gallery/<id>.png` via `gen/render.py`, and warns once (never fails the
build) if there is no GL 3.3 context to render with. `--promote ID --as
NAME` is the one way this package writes outside `training/`: it converts
one drafts record to `tests/fixtures/agent_transcripts/NAME.jsonl` +
`.expect.json` via `gen/convert.to_transcript`, for promoting a verified
draft to a tier-one regression fixture. Not run as part of this scaffold --
only when a record is actually chosen for promotion.

## Running the tests

```powershell
uv run pytest training/clay-assistant/tests -n 0 -p no:cacheprovider
```

Not part of `uv run pytest`'s default collection (`testpaths = ["tests"]`),
and not meant to join it: these drive the real `agent_clay` door directly,
the same way the dataset build itself does, rather than through the
`tests/` tree's own fixtures.

## What is gitignored

`dataset/`, `drafts/` and `seeds/` are tracked (the reproducible input and
the verified output). Gitignored, per the repo root `.gitignore`:
`gallery/` and `runs/` (regenerable renders and training-run artefacts),
`out/`, and any `*.gguf`/`*.safetensors` anywhere under `training/` (model
weights, too large and too far from source to belong in git).

## The baseline

`baseline/run_baseline.py` ran the untuned `unsloth/gemma-4-E2B-it-GGUF`
(the `BF16.gguf` variant, full precision) against the five held-out corpus
subjects on 2026-09-11, through Unsloth's own `llama-server.exe --jinja`,
with the live `agent_clay.instructions()` text as the whole system prompt --
**no tool schemas supplied**. Five of five replies came back as a fenced
JSON object with a `calls` list; zero of five would have survived
`clay_batch`'s own door (invented tool names, wrong envelope keys,
generator params hoisted to the top level, invented fields, and geometry
that ignored the stated Y-up/centred-at-origin conventions). Full detail,
including every failure mode observed, is in the plan's own "Phase 0
result" section. The raw replies are `baseline/{chair,bracket,telescope,
colonnade,serpent}.md`; `baseline/subjects.txt` holds the five prompts in
`key | subject` form (regenerable from the corpus file;
kept as a plain copy so a rerun needs nothing beyond this directory) and
`baseline/instructions.txt` (not tracked -- regenerate with
`uv run python -c "from warlock.studio import agent_clay; print(agent_clay.instructions())"`)
is what `run_baseline.py` sends ahead of the task line.

## Authoring in parallel

Each family is emitted by a deterministic `drafts/_gen_<family>.py` (hand-authored base
builds x prompt phrasings, seeded) and the JSONL beside it is what `build.py` reads.
Regenerating a family must reproduce its JSONL byte-for-byte. While several authors verify
at once, each runs `build.py --only <family> --out <scratch dir>` so nobody races on
`dataset/`; the shared `dataset/` is written by one final full build. `dataset/unsloth_*.jsonl`
is gitignored (it repeats the system turn per row and `--export unsloth` remakes it).

Lessons every author hit, kept here so the next one does not: `lathe`/`sweep`/`tube`
re-centre their `profile`/`outline`/`path` on its own bounding box before `translation`
applies; `array-radial` spins about the *world* axis line, so the seed object must sit on
it; array and mirror copies are named `<name>.001`, `.002`... and a later `clay_material`
or `clay_boolean` must list them (or paint before duplicating), else the copies keep the
default material with no refusal and no diagnose finding -- only the gallery shows it.

## Training and evaluating (Phase 2 and 3)

`train/train_a.py` is run A's recipe, run by name with Unsloth Studio's own Python
(`~/.unsloth/studio/unsloth_studio/Scripts/python.exe train/train_a.py run-B`, not the
project env; default `run-A`, output under `out/<run>/`): bf16 LoRA, responses-only loss,
the Gemma 4 template, assistant turns as fenced JSON. `CONFIG` itself is not a per-run
knob -- run B keeps run A's recipe exactly so the comparison isolates the dataset and the
card. It saves the adapter; `train/merge.py run-B` merges it into the local base offline
(Unsloth's own merge re-checks the hub); `train/export_gguf.py run-B [quant ...]` writes a
BF16 GGUF plus `Q8_0`, the only quant it allows (Q8_0 is the floor, 2026-09-13) with
the llama.cpp checkout Studio installed. Run A's numbers and the decision they support are
in `docs/measurements/2026-09-12-clay-assistant-run-A.md`.

### Ablation arms

Run B changed the dataset (three new families) and the card at once, so run A vs run B
alone cannot say which change did what. `train/make_arm.py` builds one arm's
`unsloth_{train,val}.jsonl` by pairing a chosen card against a chosen dataset (optionally
with some families dropped back out), refusing outright rather than silently training on a
card/scene mismatch, a stale `dataset/` build, or -- with `--expect-rev` -- a "run A's data"
claim that is not byte-identical (past `verified.tools_sha`) to what git actually tracked
at that commit:

```powershell
uv run python training/clay-assistant/train/make_arm.py `
    --card training/clay-assistant/out/run-A/card.txt --out training/clay-assistant/out/run-C1/data
uv run python training/clay-assistant/train/make_arm.py `
    --card training/clay-assistant/out/run-B/card.txt --drop-families grounding,figures,composition `
    --expect-rev e79b9c64 --out training/clay-assistant/out/run-C2/data
```

`C1` = run B's data (`dataset/` as it stands) + run A's card; `C2` = run A's data
(grounding/figures/composition dropped back out of the current `dataset/`) + run B's card.
Then train each the same way as run A/B, pointing `train_a.py` at the arm's own data dir
via its optional second argument: `train/train_a.py run-C1 out/run-C1/data`. Writes
`arm.json` beside the two `unsloth_*.jsonl` files: source/card hashes, dropped families,
per-split row counts before/after, and (with `--expect-rev`) how many kept records
differed from the rev.

### `eval/run_val.py`

Against a running `llama-server --jinja`, generates on the val rows (and, with `--corpus`,
the five held-out subjects) and scores every reply through the real door; results go to
`out/<run>/eval-<tag>.json`.

```powershell
uv run python training/clay-assistant/eval/run_val.py --tag A-q8-t0 --corpus
uv run python training/clay-assistant/eval/run_val.py --tag A-q4-t0.2-n3 --corpus `
    --temperature 0.2 --top-k 64 --top-p 0.95 --samples 3
uv run python training/clay-assistant/eval/run_val.py --tag A-q8-t0-card `
    --card training/clay-assistant/out/run-A/card.txt --ids training/clay-assistant/out/run-A/val-ids.txt
```

**Tag convention:** `<run>-<quant>-t<temp>[-nN]`, e.g. `A-q8-t0` (run A, Q8_0, greedy),
`A-q4-t0.2-n3` (run A, Q4_K_M, temperature 0.2, 3 samples per row). Greedy is
`--temperature 0 --top-k 1`.

* `--temperature`/`--top-k`/`--top-p`/`--max-tokens` override the sampler (defaults are the
  script's own `SETTINGS`); `--samples N` (default 1) generates N replies per row, each with
  its own `seed` (sample *i*'s request seed is always `--seed BASE + i`, the same across
  rows, so a temperature > 0 run is reproducible) -- generation runs every (row, sample)
  pair through the thread pool in parallel, scoring stays serial (the door is one
  process-global `agent_clay`/`clay_mode` state, never meant to be shared).
* `--ids FILE` restricts the val rows to the ids listed in *FILE* (one per line, in that
  order) -- `--corpus` rows are still appended after them, and an id in *FILE* missing from
  `val.jsonl` is a refusal, not a silent skip.
* `--card FILE` reads the system prompt verbatim from *FILE* instead of the live
  `convert.compact_tools()` -- needed once the live tool card has drifted from the one a
  given run actually trained on. Run A's exact card is checked in at
  `docs/measurements/data/clay-assistant/run-A/card.txt` (sha256 `70697ece...`) and run
  B's at `docs/measurements/data/clay-assistant/run-B/card.txt` (sha256 `cfa30687...`).
* Each row's output carries `samples` (one entry per generation, with that sample's own
  `reply`/`finish`/`tokens`/`outcome`/`detail`/`sub_calls`) and `accepted_k` (how many
  samples were accepted), plus the pre-`--samples` top-level keys copied from the first
  sample so an older reader still works unchanged. The written file's own `settings` key
  records every flag above, the card's source path (or `"live"`) and its sha256, and the row
  count, so an `eval-*.json` is self-describing without its invocation.
* Refusal detail comes from `verify.refusal_reason` -- the door's own one-sentence refusal
  (e.g. "no object named 'leg_2'."), not the whole batch result serialised as JSON, which is
  what the older `verify._refusal_text` returns and what the tracked run-A `eval-A-q8.json`/
  `eval-A-q4.json` files still carry as `detail`.

### `eval/compare.py`

Compares two `eval-*.json` files row for row, restricted to the ids they have in common:
an outcome transition matrix (rows = the first file's outcome, columns = the second's), a
per-family accepted count and delta, the ids that flipped between accepted and not, and
(with `--reasons`) grouped non-accepted reason counts on both sides via `reason_key` (which
folds a quoted name to `'...'` and a number to `N` so two refusals differing only in which
object or coordinate is named land in the same bucket).

```powershell
uv run python training/clay-assistant/eval/compare.py out/run-A/eval-A-q8.json out/run-A/eval-A-q4.json --reasons
```

Handles the older, pre-`--samples` eval shape too (a bare `outcome`/`detail` per row, no
`samples` key, `detail` sometimes the truncated batch-result JSON blob) -- the two tracked
`docs/measurements/data/clay-assistant/run-A/eval-A-q8.json`/`eval-A-q4.json` files are that
shape, and are what `tests/test_scaffold.py`'s own compare regression test checks against.

### `eval/render_corpus.py`

Renders the five `corpus-1`..`corpus-5` rows of an `eval-*.json` to
`<slug>.calls.json`/`<slug>.png` (chair, bracket, telescope, colonnade, serpent -- see
`baseline/subjects.txt`), replaying the first sample's reply through the real door and
`gen/render.py`'s gallery view even when the batch refused partway through: a partial scene
is still worth seeing.

```powershell
uv run python training/clay-assistant/eval/render_corpus.py out/run-A/eval-A-q8.json out/run-A/gallery
```

