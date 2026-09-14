"""Compare two ``run_val.py`` eval outputs row for row: an outcome transition matrix, a
per-family accepted delta, which ids flipped between accepted and not, and (with
``--reasons``) which refusal/failure reasons grew or shrank between the two runs.

    uv run python training/clay-assistant/eval/compare.py A.json B.json [--reasons]

Pure stdlib on purpose -- this reads two JSON files and does arithmetic on them, nothing
that needs ``gen``/``warlock`` importable, so it also works as a plain script with no
``sys.path`` setup of its own.

Handles both the current multi-sample eval shape (``row["samples"]``, ``accepted_k``) and
the older, pre-``--samples`` shape (a bare ``outcome``/``detail`` per row, no ``samples``
key at all, ``detail`` sometimes the whole batch-result JSON blob rather than a clean
sentence) -- the two tracked run-A files this module's own regression test compares are
that older shape.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys
from typing import Any

OUTCOMES: tuple[str, ...] = (
    "accepted",
    "refused",
    "built, failed check",
    "no parse",
    "fenced or empty",
)

_QUOTED = re.compile(r"'[^']*'")
_NUMBER = re.compile(r"-?\d+\.?\d*")
_WHITESPACE = re.compile(r"\s+")


def reason_key(text: str | None) -> str:
    """Normalise a refusal/failure ``detail`` string into a grouping key: a quoted name (an
    object, a material) becomes ``'...'``, a number becomes ``N``, runs of whitespace
    collapse to one space, and the result is capped to ~100 chars -- so "no object named
    'block'." and "no object named 'other_leg'." land in the same bucket, and two failures
    that differ only by a coordinate do too.
    """
    text = (text or "").strip()
    text = _QUOTED.sub("'...'", text)
    text = _NUMBER.sub("N", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text[:100]


def load(path: str | pathlib.Path) -> dict[str, Any]:
    """*path*'s eval JSON, as ``{"tag", "rows": {id: row}}`` -- rows keyed by id for the
    row-for-row lookups every comparison here needs."""
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    return {"tag": data.get("tag"), "rows": {r["id"]: r for r in data["rows"]}}


def common_ids(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    return sorted(set(a["rows"]) & set(b["rows"]))


def outcome_of(row: dict[str, Any]) -> str:
    """*row*'s first-sample outcome -- the current format's top-level ``outcome`` already
    is exactly that (copied from ``samples[0]`` by ``run_val.py``), and the older format's
    top-level ``outcome`` is its only outcome, so one accessor covers both."""
    return row.get("outcome", "?")


def accepted_mean_of(row: dict[str, Any]) -> float | None:
    """The row's per-sample acceptance rate (``accepted_k / len(samples)``), or ``None``
    for an older-format row that never carried ``samples`` at all -- callers must skip
    those rather than treat a missing rate as zero."""
    samples = row.get("samples")
    if not samples:
        return None
    accepted_k = row.get("accepted_k", sum(1 for s in samples if s.get("outcome") == "accepted"))
    return accepted_k / len(samples)


def matrix(
    a: dict[str, Any], b: dict[str, Any], ids: list[str] | None = None
) -> collections.Counter:
    """``{(a_outcome, b_outcome): count}`` over *ids* (default: the intersection)."""
    ids = ids if ids is not None else common_ids(a, b)
    counts: collections.Counter = collections.Counter()
    for i in ids:
        counts[(outcome_of(a["rows"][i]), outcome_of(b["rows"][i]))] += 1
    return counts


def print_matrix(counts: collections.Counter) -> None:
    seen = {o for pair in counts for o in pair}
    order = [o for o in OUTCOMES if o in seen] + sorted(seen - set(OUTCOMES))
    if not order:
        print("(no rows in common)")
        return
    col_w = max(6, max(len(o) for o in order))
    print(" " * 22 + "".join(f"{o:>{col_w}} " for o in order) + "total")
    col_totals = dict.fromkeys(order, 0)
    for ao in order:
        vals = [counts.get((ao, bo), 0) for bo in order]
        for bo, v in zip(order, vals, strict=True):
            col_totals[bo] += v
        print(f"{ao:<22}" + "".join(f"{v:>{col_w}} " for v in vals) + f"{sum(vals)}")
    print(
        f"{'total':<22}"
        + "".join(f"{col_totals[o]:>{col_w}} " for o in order)
        + f"{sum(col_totals.values())}"
    )


def family_accepted(
    a: dict[str, Any], b: dict[str, Any], ids: list[str] | None = None
) -> dict[str, dict[str, Any]]:
    """Per-family accepted counts for *a* vs *b*, plus their delta, over *ids* (default: the
    intersection). Adds ``a_accepted_mean``/``b_accepted_mean`` only for a family where every
    row in both files carries ``samples`` -- see :func:`accepted_mean_of`."""
    ids = ids if ids is not None else common_ids(a, b)
    families = sorted({a["rows"][i].get("family") for i in ids})
    out: dict[str, dict[str, Any]] = {}
    for fam in families:
        fam_ids = [i for i in ids if a["rows"][i].get("family") == fam]
        a_acc = sum(1 for i in fam_ids if outcome_of(a["rows"][i]) == "accepted")
        b_acc = sum(1 for i in fam_ids if outcome_of(b["rows"][i]) == "accepted")
        entry: dict[str, Any] = {
            "n": len(fam_ids),
            "a_accepted": a_acc,
            "b_accepted": b_acc,
            "delta": b_acc - a_acc,
        }
        a_means = [accepted_mean_of(a["rows"][i]) for i in fam_ids]
        b_means = [accepted_mean_of(b["rows"][i]) for i in fam_ids]
        if fam_ids and all(m is not None for m in a_means) and all(m is not None for m in b_means):
            entry["a_accepted_mean"] = sum(a_means) / len(a_means)  # type: ignore[arg-type]
            entry["b_accepted_mean"] = sum(b_means) / len(b_means)  # type: ignore[arg-type]
        out[fam] = entry
    return out


def flips(
    a: dict[str, Any], b: dict[str, Any], ids: list[str] | None = None
) -> list[dict[str, str]]:
    """Ids whose accepted-or-not status differs between *a* and *b*, with both outcomes."""
    ids = ids if ids is not None else common_ids(a, b)
    out = []
    for i in ids:
        a_outcome = outcome_of(a["rows"][i])
        b_outcome = outcome_of(b["rows"][i])
        if (a_outcome == "accepted") != (b_outcome == "accepted"):
            out.append({"id": i, "a_outcome": a_outcome, "b_outcome": b_outcome})
    return out


def reason_counts(
    data: dict[str, Any], ids: list[str], *, exclude_accepted: bool = True
) -> collections.Counter:
    """``{reason_key(detail): count}`` over *ids* -- non-accepted rows only by default."""
    counts: collections.Counter = collections.Counter()
    for i in ids:
        row = data["rows"][i]
        if exclude_accepted and outcome_of(row) == "accepted":
            continue
        counts[reason_key(row.get("detail"))] += 1
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("a", help="an eval-*.json from run_val.py")
    ap.add_argument("b", help="a second eval-*.json to compare against")
    ap.add_argument("--reasons", action="store_true", help="also print grouped reason counts")
    args = ap.parse_args(argv)

    a = load(args.a)
    b = load(args.b)
    ids = common_ids(a, b)
    print(f"comparing {len(ids)} rows in common ({a['tag']} vs {b['tag']})")
    print()

    print("outcome transition matrix (rows = A, columns = B):")
    print_matrix(matrix(a, b, ids))
    print()

    print("per-family accepted, A -> B (delta):")
    for fam, entry in family_accepted(a, b, ids).items():
        line = (
            f"  {fam:<14} {entry['a_accepted']:>4} -> {entry['b_accepted']:>4}"
            f"  (n={entry['n']}, delta {entry['delta']:+d})"
        )
        if "a_accepted_mean" in entry:
            line += f"  mean {entry['a_accepted_mean']:.2f} -> {entry['b_accepted_mean']:.2f}"
        print(line)
    print()

    flipped = flips(a, b, ids)
    print(f"flipped accepted-or-not ({len(flipped)}):")
    for f in flipped:
        print(f"  {f['id']:<24} {f['a_outcome']:<22} -> {f['b_outcome']}")

    if args.reasons:
        print()
        print("top non-accepted reasons, A vs B:")
        a_reasons = reason_counts(a, ids)
        b_reasons = reason_counts(b, ids)
        keys = sorted(
            set(a_reasons) | set(b_reasons),
            key=lambda k: -(a_reasons.get(k, 0) + b_reasons.get(k, 0)),
        )
        for key in keys[:25]:
            print(f"  A={a_reasons.get(key, 0):>3}  B={b_reasons.get(key, 0):>3}  {key}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
