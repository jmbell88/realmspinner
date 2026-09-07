# The composition score was a constant zero — the speckle floor

*2026-09-06. A defect record, not a pre-registration: the change below fixes a
term that returned the same value on every input it was ever shown, which is a
bug rather than a tuned constant. No threshold moved and no default moved, so
there is no decision rule here to fire. What this document owes the corpus is
the before/after distribution, which is why it exists.*

## What was wrong

`rank.composition_score` starts at 1.0 and charges `COMPONENT_COST = 0.15` per
component beyond the first, then clamps at zero:

```python
score -= COMPONENT_COST * max(0, int(report.get("components") or 1) - 1)
...
return max(0.0, min(1.0, score))
```

`components` is `len(cv2.connectedComponents(...))` on the geometry mask, and
`docs/measurements/2026-08-17-reference-source-bench.md` measured its median at
**15–18 blobs per image** at this image class — antialiasing and JPEG-grade
noise around the silhouette, not objects. So the charge was **−2.25 to −2.55**
against a base of 1.0 before occupancy, warnings or edge contact took anything,
and the clamp fired on every reference:

| raw components | charge | score before clamp | reported |
| --- | --- | --- | --- |
| 1 | 0.00 | 1.00 | 1.00 |
| 7 | 0.90 | 0.10 | 0.10 |
| 8 | 1.05 | −0.05 | **0.00** |
| 15 (measured median, low) | 2.10 | −1.10 | **0.00** |
| 18 (measured median, high) | 2.55 | −1.55 | **0.00** |

Everything from 8 blobs upward is indistinguishable, and the whole measured
population sits above 8.

The consequence is not that ranking was slightly off. `COMPOSITION_WEIGHT` is
0.6, so with an anchor the blend was `0.6 * 0 + 0.4 * anchor` — candidate order
was **decided entirely by the DINOv2 style cosine**, and where there was no
anchor (the ordinary case: no `ref.png`, no PickScore download) every candidate
scored exactly 0.0 and the strip stayed in seed order. The 0.25-weighted
PickScore blend was folding a human-preference term into a constant.

A second reader had the same defect. `spritesynth._front_paste_verdict` gates on
`source_report.components > 1` to answer "is the source two objects?", so it
declined to paste the front cell on essentially every real reference and wrote
"The reference is more than one object" into the sidecar as the reason.

## What changed

`reference.measure` now reports two numbers instead of one:

- **`components`** — every connected component, unchanged. This key is recorded
  in `params["reference_report"]` on every job and rolled up by
  `vectors._refusal_metrics` into a corpus that outlives the jobs in it.
  Redefining it would have re-based a stored quantity: August's blob count and
  September's subject count would be compared, and the difference read as a
  finding. It does not move.
- **`components_major`** — the components at least `MIN_MAJOR_COMPONENT` of the
  largest one's area. This is the subject count, and it is what
  `rank.composition_score` and `spritesynth` now read.

`MIN_MAJOR_COMPONENT` is `MIN_SECOND_COMPONENT` (0.08) — the same constant, not
a second one beside it, because "is this blob big enough to be a second object?"
is one question and the `multi_object` refusal already answered it that way. The
refusal is now literally `components_major >= 2`, which is pinned by
`test_the_multi_object_refusal_is_exactly_two_major_components`; the refusal's
behaviour is unchanged in both directions, which is the point.

A report written before this change has no `components_major` key. It reads as
**1 subject — unmeasured, not floored** — the same "an absent term changes
nothing" rule the anchor and the preference blend already follow. So a stored
report re-scored today scores higher than it did when it was written. That is
the fix rather than a side effect, and it is safe precisely because `components`
did not move: nothing already recorded changed value.

## The distribution

Synthetic, on the shapes the unit tests draw — a subject plus twelve two-pixel
specks:

| | before | after |
| --- | --- | --- |
| `components` | 13 | 13 |
| `components_major` | — | 1 |
| `composition_score` at target occupancy | 0.00 | 1.00 |

**Owed: the same table over props-v1 and fantasy-v1.** The real distribution of
`components_major` on the 42 graded references, and the resulting spread of
`composition_score` across each corpus, cannot be produced from the tree — the
reports live in the job rows. It is a re-measure of stored `input.png` files
with no card time, so it is cheap, and it is what turns "the term is no longer
constant" into "the term now separates candidates". Until it is run, the claim
this document supports is the narrow one: **the score was a constant and is not
one any more.** Whether the number it now produces predicts anything is
`U9`/`2026-08-09-judge-threshold.md`'s question, and calibrating
`PREFERENCE_WEIGHT` against the blend was never meaningful before this change.

## Regression tests

- `test_speckle_does_not_floor_the_composition_score` — 18 components, 1
  subject, at the target occupancy, scores 1.0. Returns 0.0 against the unfixed
  code.
- `test_the_raw_component_count_is_not_what_is_charged_for`
- `test_a_report_written_before_the_filter_existed_is_unmeasured_not_floored`
- `test_speckle_is_counted_but_is_not_a_second_subject` — both numbers, from a
  real rasterised image through `measure`.
- `test_the_multi_object_refusal_is_exactly_two_major_components`
