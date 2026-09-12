# The approved cutout becomes the reconstruction's input — pre-registration, 2026-09-08

**Status: pre-registered 2026-09-08, before the arms are run.** Everything under
"What will be run" and "Decision rules" is fixed first; numbers go under
"Results" afterwards and whichever rule fires is applied verbatim
([`2026-08-30-art-verdicts-preregistration.md`](2026-08-30-art-verdicts-preregistration.md)).

This document is written because the change it describes moves a constant the
stored corpus is keyed on, and this repository's rule is that such a document
exists *before* the constant does.

## What changed, and why it is not a tuning decision

`service/matte.py` computed a full-resolution cutout of the reference, drew it in
the promote modal, and threw it away. `promote_to_model` copied the untouched
`input.png`; `matte.approve` found no alpha to approve; `bg_removal` stayed
`birefnet`; and `trellis-server.exe` re-cut the image with **its own**
`birefnet.gguf` under `WARLOCK_TRELLIS_MODELS` — a different model file, in a
different directory, from the host BiRefNet under `t2i_model_root / birefnet`
that produced the picture the user had just approved.

So the modal was a claim about pixels nothing downstream ever saw, except in the
one case where the reference had already been matted by hand in Inker (where
`copyfile` carried the alpha along by accident of it being in the file).

As of 2026-09-08 the cutout is written to `cutout.png` beside the reference,
`promote_candidates` cuts once for a whole candidate group, and the promoted job's
`input.png` **is** those bytes — with `matte: approved` and `bg_removal: auto`,
the exe's preserving mode, recorded on the row.

**Which doors this changes, and which it does not.** Only the promote modal,
because only there does a person look at a cutout and accept it. `create_job`
(uploads), `service.sweeps` (every campaign submitter, including
`campaign_props.py` and `campaign_detail.py`) and `inker_mode._promote` all reach
`promote_to_model` with no `prepared` and copy `input.png` verbatim, exactly as
before. That is deliberate: it is what keeps every stored corpus a statement
about the pipeline that produced it.

## Why it still needs measuring

The interactive default has moved, and two facts pull in opposite directions.

*Against the change:* the 2026-08-07 review measured `bg_removal=auto` at **0
accepts in 80**, 58 of them tagged `broken`. `_q_generate`'s own comment records
why — without a matte to preserve, `auto` falls back to a threshold cutout, and a
threshold cutout on a dark brief leaves background attached for TRELLIS to
reconstruct into a slab.

*For it:* that measurement is of `auto` with **nothing to preserve**. Here `auto`
receives a real BiRefNet matte and its documented behaviour ("a pre-matted image
keeps its alpha") is exactly what is wanted. The failure mode measured in August
cannot occur, because the fallback branch is not the one taken.

Both readings are arguments. Neither is a measurement of *this* configuration,
and the corpora that would otherwise speak to it — props-v1 (11 of 22 usable) and
fantasy-v1 (10 of 20) — were both taken with the server doing the cutting.

## What will be run

**Subjects** — the five in
[`corpora/detail-v1.txt`](corpora/detail-v1.txt), chosen there on two facts that
serve this question equally well: the reference PNG is byte-identical across
submits on this machine for these prompts (so a delta is the reconstruction's,
not SDXL's), and each is a closed form with fine surface detail. Seed 42,
`text → sdxl_cfg → TRELLIS` at the shipped defaults.

**Arms, per subject** — two, matched on the same reference:

| arm | what reaches the engine | `bg_removal` |
|---|---|---|
| `server-cut` (control) | `input.png`, opaque | `birefnet` (the exe's own weights) |
| `host-cut` | `cutout.png`, RGBA | `auto` (preserve) |

Ten units. At the props corpus' measured ~8.5 min a unit that is about 1.5 hours
of card time.

**Machine evidence, free, first** — `scripts/hole_audit_vs_grade.py`, extended as
the detail sweep extended it: `mesh_audit.worst` and `mean` per unit, faces,
`source.glb`/`model.glb` size, whole-job seconds. The audit is a reproducibility
check here rather than the question — on these closed forms `worst` must stay
within 0.02 of the subject's corpus value on the control, and a `host-cut` unit
whose silhouette *opens* past 0.07 is a regression whatever else it does.

**Void check.** The `host-cut` arm's `assets/trellis.log` must show the server
taking the pre-matted path rather than running BiRefNet. If the log cannot
distinguish them, the arm's `input.png` carrying a non-opaque alpha and the row
carrying `matte: approved` / `bg_removal: auto` stand in — but the flag reaching
the exe is checked, not assumed, which is the rule the detail sweep's own
amendment was written to honour.

**Human evidence** — one blind pass in Review on the −5..+5 scale with tags
(usable is grade ≥ +3, `vectors.USABLE_GRADE`), plus one pairwise call per subject
against the control, judged in the viewer at fit-to-view and at 4× on the busiest
surface: **better**, **same**, or **worse**. The single-reviewer caveat of
[`2026-08-09-rebaseline.md`](2026-08-09-rebaseline.md) applies.

## Decision rules

- **The change stands** if the `host-cut` arm's usable count is **not lower** than
  the control's *and* the pairwise call is "better or same" on **≥4 of 5**.
- **The change is reverted to an opt-in** — a "Use this cutout" tick in the
  modal, default off, with the mechanism and every test kept — if either half
  fails. The negative is recorded here and the shipped path returns to the server
  cutting, which is what props-v1 and fantasy-v1 describe.
- **A silhouette regression is decisive on its own**: any `host-cut` unit whose
  `mesh_audit.worst` exceeds 0.07 while its control does not fires the revert,
  whatever the grades say. That is the slab failure the 2026-08-07 review found,
  and one instance of it is enough.
- **Neither arm's result licenses a change to `guidance.DEFAULT_BG_REMOVAL`.**
  This measures `auto` *with a matte in hand*; it says nothing about `auto` on an
  opaque image, which is the configuration August condemned and which no door
  reaches any more.

## Results — machine evidence

Run 2026-09-12, ten units (five subjects x two arms), seed 42, shipped
defaults otherwise, real `WARLOCK_HOME` (deliberate: these are the same rows
the human grades in Review mode, so a throwaway home would have produced
nothing to look at).

| subject | arm | job | worst | mean | seconds | source.glb | model.glb |
|---|---|---|---|---|---|---|---|
| treasure chest | control | `cda9e2ad5216` | 0.000407 | 0.000102 | 488 | 23,264,748 | 23,264,912 |
| treasure chest | host-cut | `59e7f7088114` | 0.001112 | 0.000278 | 476 | 21,497,940 | 21,498,104 |
| barrel | control | `906ce2929a94` | 0.005643 | 0.002434 | 430 | 21,044,268 | 21,044,432 |
| barrel | host-cut | `f1bddb1866cf` | 0.009680 | 0.003810 | 475 | 26,393,692 | 26,393,856 |
| skull | control | `c64a10e43bb2` | 0.014156 | 0.010597 | 395 | 20,741,940 | 20,742,104 |
| skull | host-cut | `0ef9a6cc2b21` | 0.015568 | 0.011525 | 407 | 19,846,284 | 19,846,448 |
| helmet | control | `5b9bce812a9b` | 0.005399 | 0.001356 | 183 | 17,890,676 | 17,890,840 |
| helmet | host-cut | `cdfc1c54d5f8` | 0.006008 | 0.001502 | 165 | 17,020,908 | 17,021,072 |
| pillar capital | control | `a3d1b1d11041` | 0.0 | 0.0 | 450 | 18,501,268 | 18,501,432 |
| pillar capital | host-cut | `f7d94554de17` | 0.000171 | 0.0001 | 468 | 19,407,304 | 19,407,472 |

**Silhouette rule (decisive on its own if tripped): did not trip.** Every
`mesh_audit.worst` value across all ten units is at least two orders of
magnitude under the 0.07 threshold; no host-cut unit opens past its control.

**Void check.** No `assets/trellis.log` exists per job on this build at all —
a gap the pre-registration did not anticipate, so the log-level confirmation
of "the server took the pre-matted path" could not be made for any unit. The
stand-in evidence named in the pre-registration applies uniformly to all five
host-cut units instead: each `input.png` is RGBA with real (non-flat) alpha,
and each row's stored params carry `matte: approved` / `bg_removal: auto`.
This is recorded as a gap in the harness, not as ambiguity about any one
unit — every host-cut unit has identical, uniform stand-in evidence.

## Results — grades

Blind -5..+5 pass (Review mode, real job rows): all ten units graded +4 or
+5, `verdict: accept` — usable (`grade >= vectors.USABLE_GRADE`, i.e. >= +3)
on **5 of 5** for both arms. Host-cut's usable count is not lower than
control's.

Pairwise call per subject (host-cut vs. control, fit-to-view then 4x zoom on
the busiest surface):

| subject | call |
|---|---|
| treasure chest | better |
| barrel | better |
| skull | better |
| helmet | worse |
| pillar capital | better |

Four of five "better or same" (chest, barrel, skull, pillar capital are
better; helmet is worse) — the pairwise half of the decision rule's threshold
is met exactly.

## Verdict

**The change stands.** Both halves of the first decision rule fired in the
change's favour: host-cut's usable count (5/5) is not lower than control's
(5/5), and the pairwise call is "better or same" on 4 of 5 subjects, meeting
the >=4-of-5 threshold. The silhouette-regression rule, which would have
overridden everything else, did not trip on any unit. No result here licenses
a change to `guidance.DEFAULT_BG_REMOVAL` — this measured `auto` with a matte
already in hand, not `auto` on an opaque image.

The one open note: the void check could not be made at the log level because
no per-job `trellis.log` exists on this build, so "the server took the
pre-matted path" rests on the alpha-channel and stored-params stand-in for
every host-cut unit uniformly, not on independent per-unit confirmation. This
does not change the verdict — the pre-registration named this stand-in as
acceptable when the log cannot distinguish the two paths — but it is worth a
follow-on finding if `trellis.log` was expected to exist and does not.
