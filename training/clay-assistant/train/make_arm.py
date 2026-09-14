"""Ablation arm builder: pairs a chosen dataset (optionally with some
families dropped back out) against a chosen system-prompt card, and writes
one arm's ``unsloth_{train,val}.jsonl`` under ``--out`` -- never in place in
``dataset/``, so building an arm can never clobber the tracked dataset a
parallel author might be regenerating.

Exists for the run-A/run-B ablation (see ``README.md``'s "Training and
evaluating" section): run B changed the dataset (three new families) and
the card at once, so neither of run A vs run B alone says which change
did what. Two arms isolate them:

    C1 = run B's data (dataset/ as it stands) + run A's card
    C2 = run A's data (grounding/figures/composition dropped back out) + run B's card

    uv run python training/clay-assistant/train/make_arm.py \\
        --card training/clay-assistant/out/run-A/card.txt \\
        --out training/clay-assistant/out/run-C1/data

    uv run python training/clay-assistant/train/make_arm.py \\
        --card training/clay-assistant/out/run-B/card.txt \\
        --drop-families grounding,figures,composition \\
        --expect-rev e79b9c64 \\
        --out training/clay-assistant/out/run-C2/data

Then ``train/train_a.py run-Cx out/run-Cx/data``.

Stdlib only, no project env required: this has to run before/without
``unsloth`` (or even this repo's own ``uv`` environment) ever being set up,
so every check below is deliberately paranoid rather than trusting the two
inputs (``dataset/{train,val}.jsonl`` and ``dataset/unsloth_{train,val}
.jsonl``) are still the same build they were written together as -- a
dataset/ regenerated (a new family added, a record edited) after the
unsloth export but before this script runs would otherwise silently
produce an arm that does not match either.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[3]
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]  # training/clay-assistant

DEFAULT_DATASET = _PACKAGE_ROOT / "dataset"
DEFAULT_BUILT_CARD = _PACKAGE_ROOT / "out" / "run-B" / "card.txt"
"""dataset/ as it stands today was built with run B's card (see README.md's
"The compact tool card" section) -- that is the card every row's system
turn actually carries, so it is the sensible default for ``--built-card``."""

SPLITS: tuple[str, ...] = ("train", "val")


class ArmError(Exception):
    """A refusal this script means to make, with a message meant to be read
    -- caught once in :func:`main`, never let through as a bare traceback,
    so a bad pairing reads as "refusing: ..." rather than a stack dump."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ArmError(f"{path} does not exist.")
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text:
            records.append(json.loads(text))
    return records


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    # Same json.dumps(sort_keys=True) style as build.py's own _write_jsonl/
    # _write_unsloth -- a regeneration is byte-identical, not just equal.
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def git_show(rev: str, rel_path: str, *, root: Path = _ROOT) -> str:
    """*rel_path* (posix-style, root-relative) as it read at *rev* --
    ``git show <rev>:<rel_path>``. A thin, replaceable wrapper (module-level
    function, not inlined) so a test can monkeypatch ``make_arm.git_show``
    rather than needing a real git history to exercise ``--expect-rev``."""
    result = subprocess.run(
        ["git", "show", f"{rev}:{rel_path}"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ArmError(
            f"git show {rev}:{rel_path} failed: {result.stderr.strip() or result.returncode}"
        )
    return result.stdout


def _check_alignment(
    split: str, records: list[dict[str, Any]], rows: list[dict[str, Any]], built_card: str
) -> None:
    """Refuses unless *rows* (an ``unsloth_<split>.jsonl``) is genuinely
    *records* (a ``<split>.jsonl``), row for row, in the same order: the
    system turn is *built_card* verbatim on every row, and each row's last
    user turn ends with its record's own ``prompt`` -- the one thing every
    ``kind`` (build/edit/query) puts last among its user turns (see
    ``convert.to_messages``)."""
    if len(records) != len(rows):
        raise ArmError(
            f"{split}: {len(records)} records but {len(rows)} unsloth rows -- "
            "dataset/ and dataset/unsloth_*.jsonl are not the same build "
            "(rerun `build.py --export unsloth`)."
        )
    for i, (record, row) in enumerate(zip(records, rows, strict=True)):
        rec_id = record.get("id", f"<{split}[{i}]>")
        messages = row.get("messages") or []
        if not messages or messages[0].get("role") != "system":
            raise ArmError(f"{split}[{i}] ({rec_id!r}): unsloth row has no system turn.")
        if messages[0].get("content") != built_card:
            raise ArmError(
                f"{split}[{i}] ({rec_id!r}): system turn does not match --built-card -- "
                "pass the card this dataset export was actually built with."
            )
        user_msgs = [m for m in messages if m.get("role") == "user"]
        prompt = record.get("prompt", "")
        if not user_msgs or not str(user_msgs[-1].get("content", "")).endswith(prompt):
            raise ArmError(
                f"{split}[{i}] ({rec_id!r}): last user turn does not end with the "
                "record's own prompt -- dataset/ and unsloth_*.jsonl have drifted "
                "out of the same order."
            )


def _drop_and_recard(
    records: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    drop_families: set[str],
    card: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Records (and their paired rows) whose ``family`` is not in
    *drop_families*, with every kept row's system turn replaced by *card*."""
    kept_records: list[dict[str, Any]] = []
    kept_rows: list[dict[str, Any]] = []
    for record, row in zip(records, rows, strict=True):
        if record.get("family") in drop_families:
            continue
        messages = row["messages"]
        new_row = {"messages": [dict(messages[0], content=card), *messages[1:]]}
        kept_records.append(record)
        kept_rows.append(new_row)
    return kept_records, kept_rows


def _check_expect_rev(split: str, kept_records: list[dict[str, Any]], rev: str) -> int:
    """Refuses unless *kept_records* (post family-drop) is exactly the id
    set git tracked for this split at *rev*, then returns how many of them
    differ from the rev's own JSON once ``verified.tools_sha`` is stripped
    from both sides (the one field a rebuild against a different card is
    expected to change) -- 0 means "this really is that commit's dataset,
    just re-verified against a newer tool card"."""
    rel_path = f"training/clay-assistant/dataset/{split}.jsonl"
    rev_records: dict[str, dict[str, Any]] = {}
    for line in git_show(rev, rel_path).splitlines():
        text = line.strip()
        if text:
            rec = json.loads(text)
            rev_records[rec["id"]] = rec

    kept_by_id = {r["id"]: r for r in kept_records}
    kept_ids = set(kept_by_id)
    rev_ids = set(rev_records)
    if kept_ids != rev_ids:
        missing = sorted(rev_ids - kept_ids)[:5]
        extra = sorted(kept_ids - rev_ids)[:5]
        raise ArmError(
            f"{split}: kept id set does not match {rev}:{rel_path} -- "
            f"{len(rev_ids - kept_ids)} missing from kept (e.g. {missing}), "
            f"{len(kept_ids - rev_ids)} kept but not in {rev} (e.g. {extra})."
        )

    def _without_tools_sha(rec: dict[str, Any]) -> dict[str, Any]:
        verified = dict(rec.get("verified") or {})
        verified.pop("tools_sha", None)
        return {**rec, "verified": verified}

    diffs: list[tuple[str, list[str]]] = []
    for rec_id in sorted(kept_ids):
        a = _without_tools_sha(kept_by_id[rec_id])
        b = _without_tools_sha(rev_records[rec_id])
        if a != b:
            keys = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            diffs.append((rec_id, keys))

    if diffs:
        lines = "\n".join(f"  {rec_id}: {keys}" for rec_id, keys in diffs[:5])
        raise ArmError(
            f"{split}: {len(diffs)} kept record(s) differ from {rev}:{rel_path} "
            f"beyond verified.tools_sha:\n{lines}"
        )
    return 0


def build_arm(
    *,
    card_path: Path,
    drop_families: set[str],
    expect_rev: str | None,
    out_dir: Path,
    dataset_dir: Path,
    built_card_path: Path,
) -> None:
    """Does the whole job for both splits, or raises :class:`ArmError` --
    never writes a partial ``--out`` (nothing under it is touched until
    every split has passed every check for every split)."""
    card = card_path.read_bytes().decode("utf-8")
    built_card = built_card_path.read_bytes().decode("utf-8")

    split_records: dict[str, list[dict[str, Any]]] = {}
    split_rows: dict[str, list[dict[str, Any]]] = {}
    split_kept_records: dict[str, list[dict[str, Any]]] = {}
    split_kept_rows: dict[str, list[dict[str, Any]]] = {}
    differing: dict[str, int] = {}

    for split in SPLITS:
        records = _read_jsonl(dataset_dir / f"{split}.jsonl")
        rows = _read_jsonl(dataset_dir / f"unsloth_{split}.jsonl")
        _check_alignment(split, records, rows, built_card)

        kept_records, kept_rows = _drop_and_recard(
            records, rows, drop_families=drop_families, card=card
        )

        differing[split] = _check_expect_rev(split, kept_records, expect_rev) if expect_rev else 0

        split_records[split] = records
        split_rows[split] = rows
        split_kept_records[split] = kept_records
        split_kept_rows[split] = kept_rows

    out_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        _write_jsonl(out_dir / f"unsloth_{split}.jsonl", split_kept_rows[split])

    manifest_path = dataset_dir / "manifest.json"
    arm = {
        "source_dataset": str(dataset_dir),
        "source_manifest_sha256": _sha256_file(manifest_path),
        "built_card": str(built_card_path),
        "built_card_sha256": _sha256_text(built_card),
        "card": str(card_path),
        "card_sha256": _sha256_text(card),
        "dropped_families": sorted(drop_families),
        "expect_rev": expect_rev,
        "differing_records": differing,
        "splits": {
            split: {"before": len(split_records[split]), "after": len(split_kept_records[split])}
            for split in SPLITS
        },
    }
    (out_dir / "arm.json").write_text(
        json.dumps(arm, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    counts = ", ".join(
        f"{split} {len(split_kept_records[split])}/{len(split_records[split])}" for split in SPLITS
    )
    print(f"wrote {out_dir} ({counts}); card={card_path.name}, built_card={built_card_path.name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=Path, required=True, help="the arm's system-prompt card")
    parser.add_argument(
        "--drop-families", default="", help="comma-separated family names to drop, e.g. a,b,c"
    )
    parser.add_argument(
        "--expect-rev",
        default=None,
        metavar="REV",
        help="refuse unless the kept records match this git rev's dataset/ exactly "
        "(besides verified.tools_sha)",
    )
    parser.add_argument("--out", type=Path, required=True, help="output dir, e.g. out/run-Cx/data")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="source dataset dir (default dataset/)",
    )
    parser.add_argument(
        "--built-card",
        type=Path,
        default=DEFAULT_BUILT_CARD,
        help="the card dataset/ was actually built with (default out/run-B/card.txt)",
    )
    args = parser.parse_args(argv)

    drop_families = {f for f in args.drop_families.split(",") if f}

    try:
        build_arm(
            card_path=args.card.resolve(),
            drop_families=drop_families,
            expect_rev=args.expect_rev,
            out_dir=args.out.resolve(),
            dataset_dir=args.dataset.resolve(),
            built_card_path=args.built_card.resolve(),
        )
    except ArmError as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
