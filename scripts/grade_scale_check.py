"""Tabulate the mesh-grade corpus and check it against
``docs/measurements/2026-08-09-grade-scale.md``'s predictions.

That document introduced the -5..+5 grade scale, the +-3 backfill migration
10 wrote over every pre-existing binary verdict, and the grade >= +3 usable
cut -- and it shipped with three predictions about what the *first graded
pass* would look like, stated before the data existed so they would be
readings rather than rationalisations. This script is the reading: it opens
the job database, tabulates the latest human mesh-stage verdicts, and prints
one line per prediction saying whether the recorded corpus holds it or fails
it, plus the two facts the document's own "promote the mean above Wilson"
revisit condition is stated in terms of.

Shaped like ``scripts/hole_audit_vs_grade.py``: a reader, not a submitter. It
opens the job database through ``JobStore``, writes nothing, touches no GPU,
and reuses that script's ``tagged_jobs`` for the optional ``--tag`` scope
rather than reading the tags column a second way.

A structural note the report itself repeats: a grade of exactly +3 or -3 in
the stored corpus cannot be told apart from a migration-10 backfill or from
pressing "Accept"/"Reject" during a binary judging pass (both file +-3 through
the same door a real +3/-3 grade would). The histogram below shows every
grade; the +-3 rows are additionally called out on their own line, precisely
because they are the ones the data cannot disambiguate, and prediction 1 is
about whether grading is happening *outside* that ambiguous pair.

    uv run python scripts/grade_scale_check.py
    uv run python scripts/grade_scale_check.py --tag props-v1
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import hole_audit_vs_grade as hole_audit  # noqa: E402

from warlock import vectors  # noqa: E402
from warlock.config import Config  # noqa: E402
from warlock.db import JobStore  # noqa: E402

#: The only source and stage a mesh grade lives under -- ``hole_audit_vs_grade``'s
#: own constants, restated here rather than imported so this file reads
#: standalone.
SOURCE = "human"
STAGE = "model"

PREDICTIONS_DOC = "docs/measurements/2026-08-09-grade-scale.md"


def graded_rows(store: JobStore, tag: str | None) -> list[dict[str, Any]]:
    """Every latest human mesh-stage verdict, optionally scoped to jobs
    carrying ``tag`` (a corpus tag on the job row, not a verdict's own
    good/bad tag). ``None``/empty means the whole store.
    """
    if tag:
        jobs = hole_audit.tagged_jobs(store, tag)
        verdicts = store.verdicts_for([j["id"] for j in jobs], source=SOURCE, stage=STAGE)
        return list(verdicts.values())
    return [
        r
        for r in store.latest_verdicts()
        if r.get("source") == SOURCE and r.get("stage", "model") == STAGE
    ]


def _is_grade(value: Any) -> bool:
    """``bool`` is an ``int`` in Python, and ``True`` would land in bucket 1."""
    return isinstance(value, int) and not isinstance(value, bool)


def histogram(rows: list[dict[str, Any]]) -> dict[int, int]:
    """``{grade: count}`` over the full -5..+5 range, zero-filled so every
    bucket prints even when nothing landed in it."""
    counts = dict.fromkeys(range(vectors.GRADE_MIN, vectors.GRADE_MAX + 1), 0)
    for row in rows:
        grade = row.get("grade")
        if _is_grade(grade) and grade in counts:
            counts[grade] += 1
    return counts


def tag_tallies(rows: list[dict[str, Any]]) -> Counter[str]:
    """Every tag used, both polarities in one counter -- splitting by
    membership is ``report``'s display concern, not a counting one."""
    tags: Counter[str] = Counter()
    for row in rows:
        tags.update(row.get("reasons") or ())
    return tags


def prediction_1(hist: dict[int, int]) -> str:
    """The prediction: "the graded histogram will be left-heavy but not degenerate."

    Fails outright if every grade on record is exactly +-3 -- the two spikes
    the backfill alone produces, and the failure condition the prediction
    names by name. Otherwise holds when the mass off those two spikes leans
    negative (left-heavy), the way a ~17%-accept corpus should.
    """
    total = sum(hist.values())
    if total == 0:
        return "untestable -- no graded rows"
    cut = vectors.USABLE_GRADE
    spikes = hist[cut] + hist[-cut]
    rest = total - spikes
    if rest == 0:
        return (
            f"FAILS -- every grade is +-{cut} ({spikes}/{total}); "
            "binary grading with extra steps"
        )
    left = sum(hist[g] for g in range(vectors.GRADE_MIN, 0))
    right = sum(hist[g] for g in range(1, vectors.GRADE_MAX + 1))
    if left >= right:
        return (
            f"HOLDS -- {rest}/{total} row(s) grade outside +-{cut}, and the mass "
            f"is left-heavy ({left} negative vs {right} positive)"
        )
    return (
        f"FAILS -- {rest}/{total} row(s) grade outside +-{cut}, but the mass is "
        f"right-heavy ({right} positive vs {left} negative), not left-heavy as predicted"
    )


def prediction_2(tags: Counter[str]) -> str:
    """``sharp-detail`` and ``holes`` will be the two most-used tags."""
    top2 = [tag for tag, _count in tags.most_common(2)]
    if set(top2) == {"sharp-detail", "holes"}:
        return f"HOLDS -- top tags are {', '.join(top2)}"
    shown = ", ".join(top2) if top2 else "none used"
    return f"FAILS -- top tags are {shown}, not sharp-detail/holes"


def prediction_3(hist: dict[int, int]) -> str:
    """``0`` will be used, and if it is not... that is evidence the scale
    should be even rather than odd."""
    used = hist[0]
    if used:
        return f"HOLDS -- grade 0 filed {used} time(s)"
    return "FAILS -- grade 0 has never been filed"


def revisit_condition(rows: list[dict[str, Any]]) -> tuple[tuple[str, int] | None, bool, int]:
    """The two facts ``docs/measurements/2026-08-09-grade-scale.md``'s "promote
    the mean above Wilson" revisit condition is stated in terms of:

    - the largest configuration bucket (by n) whose grades have a standard
      deviation above zero, or ``None`` if no bucket has any spread at all;
    - whether *any* bucket -- spread or not -- reaches n >= 10.

    Buckets are keyed the same way ``service.findings`` ranks whole
    configurations: ``vectors.vector_key`` of the verdict's own stored
    vector, so this reads exactly the grouping the findings document uses
    rather than a second opinion on it.
    """
    buckets: dict[str, list[int]] = {}
    for row in rows:
        grade = row.get("grade")
        if not _is_grade(grade):
            continue
        vector = row.get("vector")
        if not isinstance(vector, dict):
            continue
        buckets.setdefault(vectors.vector_key(vector), []).append(grade)

    largest: tuple[str, int] | None = None
    any_n10 = False
    for key, grades in buckets.items():
        n = len(grades)
        if n >= 10:
            any_n10 = True
        if n >= 2 and statistics.pstdev(grades) > 0 and (largest is None or n > largest[1]):
            largest = (key, n)
    return largest, any_n10, len(buckets)


def report(rows: list[dict[str, Any]]) -> None:
    hist = histogram(rows)
    total = sum(hist.values())
    print(f"latest human model verdicts: {total}")
    print()

    print("grade histogram (-5..+5):")
    for grade in range(vectors.GRADE_MIN, vectors.GRADE_MAX + 1):
        marker = "  <- backfill cut" if abs(grade) == vectors.USABLE_GRADE else ""
        print(f"  {grade:+d}: {hist[grade]:4d} {'#' * hist[grade]}{marker}")
    print()

    cut = vectors.USABLE_GRADE
    spikes = hist[cut] + hist[-cut]
    percent = round(spikes / total * 100) if total else 0
    print(
        f"+-{cut} rows: {spikes} of {total} ({percent}%) -- indistinguishable in the"
        " data from a migration-10 backfill or a binary Accept/Reject press"
    )
    print(f"every other grade: {total - spikes} of {total}")
    print()

    tags = tag_tallies(rows)
    good = [(t, tags[t]) for t in vectors.GOOD_TAGS if tags.get(t)]
    bad = [(t, tags[t]) for t in vectors.BAD_TAGS if tags.get(t)]
    print("tags (good):")
    for tag, count in sorted(good, key=lambda kv: (-kv[1], kv[0])):
        print(f"  {tag}: {count}")
    print("tags (bad):")
    for tag, count in sorted(bad, key=lambda kv: (-kv[1], kv[0])):
        print(f"  {tag}: {count}")
    print()

    print(f"grade 0 (no opinion) used: {hist[0]} time(s)")
    print()

    print(f"predictions ({PREDICTIONS_DOC} §Predictions):")
    print(f"  1. {prediction_1(hist)}")
    print(f"  2. {prediction_2(tags)}")
    print(f"  3. {prediction_3(hist)}")
    print()

    largest, any_n10, n_buckets = revisit_condition(rows)
    print("revisit condition (promote the mean above Wilson):")
    if largest is None:
        print(f"  no configuration bucket ({n_buckets} total) has grade spread (sd > 0) yet")
    else:
        key, n = largest
        print(f"  largest bucket with grade sd > 0: {key} (n={n})")
    print(f"  any bucket reaches n >= 10: {'yes' if any_n10 else 'no'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tag",
        default=None,
        help="scope to jobs carrying this corpus tag; default is the whole store",
    )
    args = parser.parse_args()

    config = Config()
    db_path = Path(config.db_path)
    if not db_path.exists():
        print(f"no job database at {db_path}", file=sys.stderr)
        return 1
    store = JobStore(db_path)
    try:
        rows = graded_rows(store, args.tag)
    finally:
        store.close()

    if not rows:
        scope = f" tagged {args.tag!r}" if args.tag else ""
        print(f"no latest human mesh-stage verdicts{scope} in {db_path}", file=sys.stderr)
        return 1

    report(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
