# Gate patterns — what to leave behind, and what not to trust

A sweep that touches every site and adds nothing to catch the *next* miss has fixed one
occurrence of a class of bug that has already recurred four times in this tree (the
grid kind, Muse, Troupe, `_maybe_queue_rig` twice). The tree already contains three
different shapes of gate that actually held, and two that only looked like they did.
Read the working examples before writing a new one — most of what this skill needs is
already proven somewhere in `tests/`, just not generalised to the table you are about
to extend.

## Pattern 1: the two-way assertion (the tour is the model)

`src/warlock/studio/tour/` is pure data — a step names a mode, an anchor, a condition
and (usually) a manual chapter to hand the reader on to — and `tests/tour/` is four
files that check it from both directions at once, never one.

**`tests/tour/test_tour_steps.py`** is the clearest instance:

- `test_every_anchor_a_step_names_is_marked_somewhere` — every anchor a step points at
  must have a real `anchors.mark("x")` call site in the source (found by an `ast`-free
  regex sweep of every `.py` file under `studio/`, plus the rail's own derived key
  set).
- `test_every_marked_anchor_is_named_by_a_step` — the reverse: every `anchors.mark`
  call site must be pointed at by some step, or it is "a mark nobody removed when the
  step that needed it went."

Both directions matter because a **one-way** check passes while the table grows a hole
from the other side: checking only "every step's anchor exists" would never notice a
step that was deleted while its mark was left behind, and checking only "every mark is
used" would never notice a step added against a control that does not exist yet. The
same shape appears again in the same file for conditions:

- `test_every_step_waits_for_a_condition_the_vocabulary_carries` (declared → known)
- and, in `tests/tour/test_tour_conditions.py`, `test_every_declared_condition_is_answered`
  (declared → answered by `panes/tour.py`) **and**
  `test_the_evaluator_answers_nothing_it_was_not_asked_to` (answered → declared) — the
  same pair, in the other file, for the split between the headless declaration and the
  imgui-drawing evaluator that answers it.

There is also a guard on the guard: `test_the_marker_regex_actually_matches_the_call_sites`
exists because both directions above are built from one regex, and a regex that stops
matching (a rename, a keyword argument, a wrapper) would make every assertion above
pass by finding nothing on both sides at once. **When you write a two-way check, write
the guard that proves the check can still fail.**

**Apply this shape to `followup-kind`** (which the sweeps table names as having no gate
at all today): a bidirectional check that every kind in `FOLLOWUP_STAGES` is a real
kind that actually mints `params["source_job"]` rows, **and** that every kind that
mints such rows (grep `Store.create` call sites for one omitting `stage=`, or grep
`params["source_job"]` writers) has an entry in `FOLLOWUP_STAGES` *or* its own
documented special case in `route()`, is the shape this pattern buys and nothing in
the tree currently provides.

## Pattern 2: the parametrised sweep test (`tests/test_progress.py`)

`test_the_multi_pass_kinds_have_their_own_contiguous_tables` is the one stage-keyed
table in the whole `job-kind` list that got a real, if partial, gate — and it took four
failures first (the grid kind, then Muse and Troupe's own progress misses, though those
two are narrated in `docs/INVARIANTS.md` rather than in this test's own docstring,
which cites the pattern generically as "CON-02... for the fourth time"). Its shape:

```python
@pytest.mark.parametrize("kind", ["pixel_sheet", "retexture", "tile_sheet", "charsheet"])
def test_the_multi_pass_kinds_have_their_own_contiguous_tables(kind):
    table = phases_for(kind)
    assert table is not PHASES_IMAGE, f"{kind} still falls back to the image table"
    spans = sorted(table.values())
    assert spans[0][0] == 0.0
    assert spans[-1][1] == 1.0
```

Read this shape honestly rather than as a finished example: it is parametrised over a
**hand-maintained list of kind names**, not derived from `_PHASES_BY_KIND`'s own keys —
so a fifth multi-pass kind is exactly as invisible to this test as the first four were,
until someone adds its name to the list by hand. It also does not check contiguity
*between* spans, only the two endpoints — the stronger check
(`test_the_sprite_phases_cover_the_whole_bar_contiguously`) exists for exactly one
table and was never generalised. **This is what "a real gate with a hole in it" looks
like from the inside**, and it is worth naming explicitly in a sweep's report rather
than citing the test's existence as proof the site is safe.

**The rule this pattern states, and the fix its own hole implies**: a sweep run ends by
parametrising the table it just joined — either by adding the new kind's name to an
existing parametrize list (the minimum this pattern already supports), or, better, by
changing the parametrize source from a hand list to something derived (every key in
`_PHASES_BY_KIND` itself, for instance, which would make a sixth kind self-enrolling the
way `tests/test_accessibility.py`'s `sorted(tokens.PALETTES)` parametrization makes a
new palette self-enrolling). Prefer the derived form when you are already touching the
test; do not feel obliged to rewrite an existing hand list on every unrelated sweep run.

## Pattern 3: the import-pinning idiom

`tests/inker/test_inker_imports.py`, `tests/tour/test_tour_imports.py`,
`tests/test_viewer_imports.py` and `tests/test_poser_imports.py` all pin the *exact*
outward import set of a headless package, walking every module's `ast.Import`/
`ast.ImportFrom` nodes and comparing the resulting set against a literal
`OUTWARD_IMPORTS` (or, in the viewer's case, a set of banned roots plus a fixed module
count). `tests/tour/test_tour_imports.py`'s version:

```python
OUTWARD_IMPORTS: set[tuple[str, str]] = set()

def test_nothing_here_reaches_outside_the_package():
    found = {...}
    assert found == OUTWARD_IMPORTS, (
        "studio/tour's outward imports changed. A tour is data: if a step now "
        "needs something from the app to describe itself, that belongs in the "
        "pane that draws it, where it costs no headless test."
    )
```

The point of pinning the **exact set** rather than a denylist is that adding an import
becomes a deliberate, visible edit to the test file itself — a diff a reviewer sees —
rather than a silent widening nobody notices until the package can no longer be
imported without a GL context. This pattern does not apply directly to most of the
seven sweeps in this skill (none of them is "a headless package gaining an import"),
but it is the right shape to reach for if a sweep's gate step turns out to be "this
table's reader must not import `service`" or similar — `queue.py`'s own "may not
import `service`" rule (see `job-kind`'s vram/VECTOR_PARAMS discussion) is exactly that
shape and has no pinning test of its own today.

## Pattern 4: the derived table (a positive example outside `tests/`)

Not every gate is a test. `src/warlock/studio/panes/app_settings.py`'s
`_GROUPS = fetch.GROUPS` is a *design* choice that removes an entire class of miss
rather than catching it after the fact: the Settings pane heading used to be a
hand-copied second list, and the module comment states plainly why that was replaced —
"this loop only draws the kinds it lists, so an unlisted one vanished from the pane
entirely with nothing to say it existed." Deriving the heading from `fetch.KINDS`
instead means the `download-kind` sweep has one less site to visit by construction, not
because a test enforces it. **When a sweep's report finds a site that is a hand copy of
data another site already owns, the strongest gate is to delete the copy and derive it
— consider this before reaching for a new test.**

## What a bad gate looks like

Two examples in the tree, both worth citing by name rather than emulating:

- **`PUBLISHERS` in `tests/test_job_durability.py`** is a hand-written list that, by
  its own comment, "stays a list instead of a scan of its own" because the judgement
  it encodes (completion marker vs. intermediate checkpoint) cannot be automated — and
  it has covered eight of the eleven known sites, having already let six of its eight
  current rows ship the bug it exists to catch, once each, before being added. A
  hand-written list is sometimes genuinely the right shape (the judgement really can't
  be derived), but it fails **open**: a stage nobody adds to the list is invisible to
  it, not rejected by it. If you write a hand list as a sweep's gate, say so explicitly
  in the report and name the next TODO entry the way `TODO.md` P37 does, rather than
  presenting the list as complete.
- **The seventeen individually-written assertions in `tests/manual/test_manual_promises.py`
  and `test_manual_labels.py`** are the other shape of bad gate: one assertion per drift
  found, rather than one rule that would have caught all seventeen. `tests/manual/test_shortcuts.py`
  is a milder, deliberate instance of the same asymmetry — it checks the shortcut popup
  against chapter 16 in **one direction only**, because "the chapter is documented as
  the full list and the popup as a subset," which is a legitimate design (not every
  one-way check is a bug) but is exactly the shape to notice you have written when you
  meant to write pattern 1 above and stopped halfway.

## The rule

**A sweep run ends by parametrising the table it just joined**, in the tour's two-way
shape where nothing sweeps that table yet, or by adding the new name to an existing
parametrised sweep (`test_progress.py`'s kind list, `test_docs.py`'s `EXPECTED_KEYS`,
`PUBLISHERS`) where one already exists — or, failing both, by recording in the sweep's
report exactly why neither was possible this time (the judgement is irreducibly manual,
as `PUBLISHERS` is; the site is already self-gating by construction, as `download-kind`
mostly is) rather than by silence. A report that closes every site and says nothing
about the gate has done half the job this skill exists for.
