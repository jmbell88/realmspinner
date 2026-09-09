# The mesh judge probe, pre-registered before any fit is run, 2026-09-07

**Status: pre-registered, no fit taken.** A pre-registration in the shape of
[`2026-08-09-judge-threshold.md`](2026-08-09-judge-threshold.md), which
pre-registered the two *image*-stage probes and explicitly deferred this one:
"Max-versus-mean pooling is the **mesh** probe's question and is deferred:
there is no mesh corpus, and `bench/views.py`'s calibration
([`2026-08-04-view-calibration.md`](2026-08-04-view-calibration.md)) already
establishes that a single-view mesh classifier would be learning camera pose."
`judge.py`'s own module docstring says the same thing in its own words: "The
mesh probe is declared and unbuilt... it is blocked on a corpus with positives
in it, and how it pools over eight views is a decision to make with data."
Both of those are "decide later, once the data exists" — this document is the
opposite move, and deliberately: every number below is written down before the
first embedding is computed, so the corpus this repo's own rule warns about
(a stored fit is a constant a motivated reader would tune after seeing the
result) cannot happen here even by accident.

## The question

Migration 10 turned the mesh verdict from a bit into a grade
(`docs/measurements/2026-08-09-grade-scale.md`), which is what makes a
*regression* probe the right shape rather than the classifier `judge.fit`
already knows how to train — a strictly better fit for a corpus whose original
problem was the resolution of its labels, not their number. And
`2026-09-02-hole-audit-vs-grade.md` already ruled out the free feature: the one
audit scalar that exists on every finished mesh, `mesh_audit["worst"]`, scores
AUC 0.115 against `holes` on the binary corpus — backwards, because a slab has
no holes and the audit scores the dominant failure mode as perfect. A learned
probe over pixels is the only candidate left that has not already been
disqualified by a measurement.

## The corpus, checked before anything else

The pattern this repo's calibration and audit documents both open with is "see
whether there is anything to measure" before measuring it, so that came first
here too, read-only, off this machine's own `~/.warlock`:

```
store.latest_verdicts()   # filtered to stage == "model", source == "human"
```

returns **9** distinct `job_id`s. Checking each one for
`data_dir / job_id / "source.glb"` on disk finds **0**. Every one of the nine
was graded to completion inside a sweep, and `review_mode.JudgingPass`'s own
cleanup licence — "a sweep every unit of which has a verdict has its assets
removed, with a toast and no dialog" (`docs/INVARIANTS.md`, the guided-judging-
pass section) — is exactly what took them: finishing a graded pass is also
what deletes the meshes this probe would need. `2026-09-02-hole-audit-vs-
grade.md`'s own props-v1 pass graded 21 and 22 meshes in its two rounds, 43 in
total — the figure "roughly 42 graded meshes" in this document's brief is that
pass, not a live count — and that pass, like every completed sweep, was
subject to the identical cleanup the moment its last unit was filed. **The
corpus this document is defined against is not a fixed number recoverable from
a doc; it is whatever is still on disk today, and today that is zero.** That is
not a reason not to write this document — it is the reason to write it now,
while the honest answer to "is there enough" is a number nobody could have
tuned a result to.

## What will be run

**Corpus.** Every row `store.latest_verdicts()` returns with `stage == "model"`
and `source == "human"`, whose job directory still carries `source.glb` — the
reconstruction, not `model.glb`, since `model.glb` is a derived artifact a
later optimize/normalize pass can change without the mesh a human actually
graded changing with it.

**Views.** `bench.views.render_views(source_glb, out_dir)` at
`REFERENCE_ELEVATION` (20.0 degrees) — the constant `2026-08-04-view-
calibration.md` measured and left exactly where it was, because the sweep it
ran found no fixed matched *yaw* on 37 jobs (330-degree scatter) but never
touched elevation as a free variable, and nothing since has given a reason to
move it. `out_dir` is `bench_dir / "views" / job_id`; a directory that already
carries `views.json` (`views.SIDECAR`) is treated as a cache hit and is not
re-rendered, since eight views of a mesh that has not changed are the same
eight views every time this document's own fit script is re-run. The render
itself calls `rigging.run_worker` synchronously — correct for the script that
runs this fit, on the same CLI thread `bench calibrate` already blocks on
(`views.render_views`'s own docstring: "correct here and only here") — but
that licence does **not** extend to the one caller this document deliberately
does not build: a future `service.judge` path invoked from Review lives on the
frame thread, and `render_views` there would have to go through a
`TaskRunner` task exactly as every other blocking call in `studio/` does, or
Review freezes for the duration of a Blender render.

**Embedding.** `bench.metrics.cls_embedding` — the one DINOv2 loader in the
repo, `local_files_only=True`, the same call `judge.embed` already wraps for
the image stages — run once per rendered view, eight 768-d CLS vectors per
mesh.

**Pooling, declared in advance rather than chosen after looking.** Both
candidates are fitted and both are reported: **mean**-pool the eight CLS
vectors into one 768-d row, and **max**-pool them elementwise into a second.
Max is not a new idea here — it is `2026-08-04-view-calibration.md`'s own
prescription for what any future fidelity metric has to be, "take the best of
the eight turntable views rather than the first," generalised from a single
`dino_cosine` scalar to a full embedding. Mean is its natural counterpart: it
keeps information max discards (a mesh that is good from every angle looks
different, elementwise, from one that is good from exactly one) at the cost of
diluting whichever view actually shows the flaw a grade penalised. **The
winner is whichever pooling scores higher held-out Spearman ρ**, decided only
after both are fitted on the same split; the loser's held-out ρ, its bootstrap
CI and its per-subject breakdown are reported in this document's Results
section beside the winner's, not omitted once a winner exists.

**Model.** Ridge regression, closed-form (`w = (XᵀX + λI)⁻¹Xᵀy`), predicting
`grade` (−5..+5) from the pooled 768-d embedding — plain numpy, no new
dependency, the same "pure stdlib plus numpy" doctrine `judge.py` already
states for its own logistic fit, extended from gradient descent to a closed
form because ridge has one and a fixed iteration count buys nothing over it.

**Split — by `prompt_hash`, never random.** `2026-08-09-judge-threshold.md`
already established why for the image stages and the reasoning transfers
without change: several meshes reconstructed from the same subject are, for a
DINOv2 embedding, different *views* of one thing — near-identical silhouette,
palette and style — the same way eight turntable renders of one mesh are.
Assigning them to opposite sides of a random split would let the held-out
score reward recognising a subject the probe has already seen rather than
recognising what a human penalised it for, exactly the leakage a held-out set
exists to catch and exactly what a per-view (not per-subject) split would let
through unmeasured. The split is fixed once, by `prompt_hash`, before either
pooling variant is fitted.

## Success criterion, declared now

**Held-out Spearman ρ ≥ 0.5**, with a bootstrap confidence interval (percentile
bootstrap, scipy — already a project dependency outside `sirens/`/`muse/`,
which are the only two modules that ban it) over the held-out rows, excluding
0. **Per-subject ρ is also reported** — one figure per `prompt_hash` bucket
with enough held-out rows to compute one — the same discipline
`2026-08-09-judge-threshold.md` required of the image probes' false-reject
rate, because a probe that only works pooled across subjects has learned
"good X" for whichever X dominates the corpus, not "good mesh."

## Baselines

**Predict-the-mean** — the train split's mean grade, for every held-out row.
Any candidate that does not clear this by a wide margin has learned
approximately nothing, whatever its raw ρ says.

**`hole_worst`**, ranked directly against grade. Expected to run backwards, the
same direction `2026-09-02-hole-audit-vs-grade.md` already measured on the
coarser accept/reject target (AUC 0.115): a slab scores `worst = 0.0` and
audits as flawless while a human grades it anywhere from unusable to fine for
reasons the audit cannot see, so nothing about widening the target to −5..+5
gives this scalar a reason to start correlating in the right direction. It is
reported as a floor of no floor at all, not as a bar the DINOv2 probe merely
has to clear.

## Null result rule, declared now — before the corpus exists to tempt a lower one

**Minimum N per split: 8.** That is not a round number chosen for this
document — it is `judge.MIN_PER_CLASS`, the one floor this exact codebase has
already adopted for "not enough labels to learn a boundary from," restated for
a regression target: below 8 held-out points, a bootstrap CI over the sample
essentially always contains 0 regardless of the true correlation, so the test
is uninformative by construction beneath that line, the identical failure mode
`judge.fit`'s own `None` return exists to refuse rather than silently believe.
Train gets the same floor for the same reason — a ridge fit to fewer than 8
points has nothing to regularise toward.

**The rule: below N=8 in either split, or a held-out bootstrap CI containing
0, and no probe file is written.** No `probe-model.npz` (`judge.probe_path`'s
own naming, extended to the stage it does not yet cover), `service.judge.
TRAINABLE_STAGES` (currently `verdicts.IMAGE_STAGES` — `reference`, `blank`
— and not extended to include `"model"`), and `review_mode.SCORE_STAGE` stays
`"blank"`, exactly as it is today. This is not a hypothetical trigger: the
corpus section above already found N=0 on this machine, so the rule as written
would fire the moment anyone tried to run this fit right now, before a single
line of the methodology above executes. That is the point of writing the rule
down today rather than after the first attempt — a rule that only exists to be
overridden the first time it is inconvenient is not a rule.

## Authority on success — and its explicit limit

If the criterion is met, what changes is: Review units may be **sorted** by
the mesh probe's predicted grade — `by_score`'s own doctrine, "sorts and never
filters," applied to the new stage exactly as it already applies to `blank` —
and a `score_line` may name its own question the way the existing one does
(`"judge: 73% - will this reconstruct"` today; the mesh equivalent would read
something like `"mesh judge: +2.1 predicted"`, a grade rather than a
probability, since the target is a regression). **That is the entire grant.**
No `SOURCE_AI` verdict (`review_mode.SOURCE_AI = "ai:dino-probe"`) is ever
filed for the mesh stage — `docs/INVARIANTS.md`'s own line on this is exact:
"the judge is advisory — it may sort, never filter, refuse, delete or retry,"
and no AI verdict enters `findings.json` under this document, because
`2026-08-09-judge-threshold.md` already declared that feeding the aggregation
with the judge's own opinions would make the corpus self-referential and
"owes its own document" — one this document is not, and does not attempt to
be.

## What is not being built now

`judge.fit_regression` / `judge.score_regression` (`judge.fit` and
`judge.score` stay exactly what they are, a logistic fit over two classes;
these would be their ridge-and-continuous-target counterparts), any
`service.judge` view-rendering path, and `service.judge.TRAINABLE_STAGES`
gaining `"model"` are all deliberately unbuilt by this document. They follow
only if a human runs the methodology above on a real corpus and this
document's own success criterion fires — never from this document alone, and
never from a corpus assembled by waiting for the numbers to look right first.

## Results

Not yet taken. Blocked on a corpus: the check above found 0 model-stage human
verdicts on this machine still carrying `source.glb`, against a declared floor
of 8 per split. The path back is one this document does not choose: it is
whatever mix of new grading (through ordinary Create/Library review, which
`review_mode.JudgingPass`'s cleanup licence explicitly does not touch — it
"refuses `RECENT_ID`, whose units are ordinary library rows and whose removal
is prune's job") and un-swept sweep review a human decides to do next.

## Amendment, 2026-09-08 — the mechanism that emptied the corpus is gone

**Nothing in the methodology above changes, and no number in it moves.** What
changes is the sentence under Results that says the path back is whatever
grading a human does next, because it was written against a tree in which
finishing a grading pass *destroyed* what it graded, and that is no longer
true. `service.evidence` copies a judged unit's `input.png`, `cutout.png`,
`reference.png`, `source.glb` and `model.glb` — plus a `job.json` naming the
settings and the grade — out to `config.evidence_dir` before `cleanup_sweep`
or `prune_jobs` removes anything. The reclaim is unchanged; what has stopped
happening is that it also took the measurement with it.

Three things follow, and the third is the one worth being careful about:

- **The count of 0 stands as a measurement of 2026-09-07** and is not
  retroactively repaired. Those nine meshes are gone; nothing here recovers
  them, and no fit may be taken against a corpus that pretends otherwise.
- **A rejected unit now survives too**, which the retention rule deliberately
  never kept (`jobs.retained_job_ids`: a model-stage reject "is fully carried
  by its row"). That is correct for the finding it was written about and wrong
  for this document, whose declared floor is **8 per split** — a regression
  probe needs both ends of the grade scale, and until now only accepts were
  ever kept.
- **The corpus therefore accumulates from the next graded pass onward, and its
  files live outside `data_dir`.** Anything reading it must read the archive as
  well as the library; a check that walks `data_dir / job_id / "source.glb"`
  alone — which is exactly the check this document's own corpus section
  describes — will keep answering 0 for as long as the assets it wants are the
  ones a delete carried out. That is a change to *where the corpus is*, not to
  what the corpus must contain, and the success criterion above is untouched
  by it.
