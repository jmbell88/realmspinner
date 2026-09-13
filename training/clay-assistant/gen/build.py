"""The dataset CLI: ``drafts/*.jsonl`` -> verified ``dataset/{train,val}.jsonl``
(+ ``rejects.jsonl``, ``manifest.json``), driving every record through the
real ``agent_clay.call`` door via ``verify.py`` -- see the plan's Phase 1
"build.py CLI" paragraph for the full contract this implements.

    uv run python training/clay-assistant/gen/build.py
    uv run python training/clay-assistant/gen/build.py --only furniture --gallery
    uv run python training/clay-assistant/gen/build.py --export unsloth --force

Every accepted record replays clean through ``agent_clay.call`` -- that is
the proof this dataset is buildable at all, not merely shaped like one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[3]
_SRC = _ROOT / "src"
_SCRIPTS = _ROOT / "scripts"
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]  # training/clay-assistant
for _p in (_SRC, _SCRIPTS, _PACKAGE_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gen import convert, headless, holdout, render, schema, verify  # noqa: E402
from warlock.studio import agent_clay  # noqa: E402

DEFAULT_DRAFTS = _PACKAGE_ROOT / "drafts"
DATASET_DIR = _PACKAGE_ROOT / "dataset"
GALLERY_DIR = _PACKAGE_ROOT / "gallery"
FIXTURES_DIR = _ROOT / "tests" / "fixtures" / "agent_transcripts"
FORBIDDEN_DRAFTS_PARENT = _ROOT / "docs" / "measurements" / "corpora"
"""``build.py`` refuses ``--drafts`` under here -- the held-out corpus is a
read-only eval, never a place drafts are authored from or written to."""

VAL_MODULUS = 10  # 1 bucket in 10 -> 10% held for val, per family on average


def _val_bucket(rec_id: str) -> int:
    """Deterministic by the id's own sha256 -- not `random`, so a rerun (or a
    different machine) puts the same record in the same split every time."""
    return int(hashlib.sha256(rec_id.encode("utf-8")).hexdigest(), 16) % VAL_MODULUS


def _load_drafts(drafts_dir: Path, only: str | None) -> list[tuple[Path, int, dict[str, Any]]]:
    """Every record across ``drafts/*.jsonl`` (or just ``<only>.jsonl``), as
    ``(file, line_no, record)`` -- kept even for a record that fails to parse
    as JSON, so a malformed line is a rejection with a location rather than a
    silent skip."""
    rows: list[tuple[Path, int, dict[str, Any]]] = []
    pattern = f"{only}.jsonl" if only else "*.jsonl"
    for path in sorted(drafts_dir.glob(pattern)):
        for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            text = raw.strip()
            if not text:
                continue
            rows.append((path, line_no, json.loads(text)))
    return rows


def _verify_one(record: dict[str, Any]) -> tuple[list[str], verify.Replay]:
    kind = record.get("kind", "build")
    if kind == "build":
        replay = verify.replay(record["calls"])
    elif kind == "edit":
        replay = verify.replay(record["calls"], prior=record["prior"])
    else:  # query -- its own "prior" is what gets built and read, nothing else runs
        replay = verify.replay(record["prior"])
    allow_below_ground = bool(record.get("allow_below_ground", False))
    reasons = verify.check(replay, allow_below_ground=allow_below_ground)
    return reasons, replay


def _call_count_for(record: dict[str, Any]) -> int:
    """How many top-level door calls this record's own replay made -- 1 for
    a ``build``/``query`` (one folded ``clay_batch``), 2 for an ``edit``
    (the scene-setup batch, then the edit batch). Not the length of
    ``record["calls"]``: that is the sub-call count *inside* the one batch
    call the model itself would emit, a different and also-interesting
    number (see ``README.md``)."""
    return 2 if record.get("kind") == "edit" else 1


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, sort_keys=True) + "\n")


def _write_unsloth(
    records: list[dict[str, Any]], path: Path, *, tools_text: str, inline: bool
) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            prior_scene = None
            if record.get("kind") in ("edit", "query") and record.get("prior"):
                prior_replay = verify.replay(record["prior"])
                prior_scene = convert.compact_scene(verify.scene(prior_replay))
            row = convert.to_messages(
                record, tools_text=tools_text, inline=inline, prior_scene=prior_scene
            )
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def _promote(
    drafts_dir: Path, record_id: str, as_name: str | None, parser: argparse.ArgumentParser
) -> int:
    if not as_name:
        parser.error("--promote requires --as NAME")
    rows = _load_drafts(drafts_dir, None)
    match = next((r for _, _, r in rows if r.get("id") == record_id), None)
    if match is None:
        print(f"no record with id {record_id!r} under {drafts_dir}", file=sys.stderr)
        return 1

    lines, expect = convert.to_transcript(match)
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = FIXTURES_DIR / f"{as_name}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(json.dumps(line) + "\n")
    (FIXTURES_DIR / f"{as_name}.expect.json").write_text(
        json.dumps(expect, indent=2) + "\n", encoding="utf-8"
    )
    print(f"promoted {record_id!r} -> {jsonl_path} (+ .expect.json)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=schema.FAMILIES, default=None)
    parser.add_argument("--gallery", action="store_true")
    parser.add_argument("--export", choices=["unsloth"], default=None)
    parser.add_argument("--inline", action="store_true")
    parser.add_argument("--seed", type=int, default=0, help="recorded in the manifest; reserved")
    parser.add_argument("--force", action="store_true")
    # Authors verifying one family each in parallel must not race on the shared
    # dataset/ files (a Windows "file in use" or, worse, a silently clobbered
    # train.jsonl), so each can point its run at a scratch directory.
    parser.add_argument(
        "--out", type=Path, default=DATASET_DIR, help="output dir (default dataset/)"
    )
    parser.add_argument("--drafts", type=Path, default=DEFAULT_DRAFTS)
    parser.add_argument("--promote", default=None, metavar="ID", help="a drafts/ record id")
    parser.add_argument("--as", dest="as_name", default=None, metavar="NAME")
    args = parser.parse_args(argv)

    drafts_dir = args.drafts.resolve()
    try:
        drafts_dir.relative_to(FORBIDDEN_DRAFTS_PARENT.resolve())
    except ValueError:
        pass
    else:
        parser.error(
            f"--drafts may not point under {FORBIDDEN_DRAFTS_PARENT} -- "
            "that is the held-out corpus, never a source of drafts."
        )

    tools_text = convert.compact_tools()
    sha = convert.tools_sha()

    if args.promote:
        return _promote(drafts_dir, args.promote, args.as_name, parser)

    out_dir = args.out.resolve()
    manifest_path = out_dir / "manifest.json"
    if manifest_path.is_file() and not args.force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        old_sha = existing.get("tools_sha")
        if old_sha and old_sha != sha:
            print(
                "refusing to regenerate: dataset/manifest.json was built "
                f"against tools_sha {old_sha}, the live tool card is now "
                f"{sha}. Pass --force once you mean to invalidate it.",
                file=sys.stderr,
            )
            return 1

    rows = _load_drafts(drafts_dir, args.only)
    holdout_data = holdout.load_holdout()
    holdout_sha = hashlib.sha256(holdout.CORPUS_PATH.read_bytes()).hexdigest()

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    replays: dict[str, verify.Replay] = {}
    reason_histogram: dict[str, int] = {}

    def _reject(rec_id: str, family: Any, kind: Any, reasons: list[str]) -> None:
        rejected.append({"id": rec_id, "family": family, "kind": kind, "reasons": reasons})
        for reason in reasons:
            reason_histogram[reason] = reason_histogram.get(reason, 0) + 1

    for path, line_no, record in rows:
        rec_id = record.get("id") if isinstance(record, dict) else None
        rec_id = rec_id if isinstance(rec_id, str) and rec_id else f"{path.name}:{line_no}"
        family = record.get("family") if isinstance(record, dict) else None
        kind = record.get("kind") if isinstance(record, dict) else None

        errors = schema.validate_record(record)
        if errors:
            _reject(rec_id, family, kind, errors)
            continue

        leak = holdout.is_leak(record["prompt"], holdout_data)
        if leak:
            _reject(rec_id, family, kind, [f"prompt overlaps held-out corpus line: {leak!r}"])
            continue

        reasons, one_replay = _verify_one(record)
        if reasons:
            _reject(rec_id, family, kind, reasons)
            continue

        scene_payload = verify.scene(one_replay)
        bounds = scene_payload.get("bounds") or {}
        record["verified"] = {
            "object_count": len(one_replay.doc.objects),
            "bounds": {"size": bounds.get("size"), "center": bounds.get("center")},
            "diagnose_clean": True,
            "call_count": _call_count_for(record),
            "tools_sha": sha,
        }
        accepted.append(record)
        replays[rec_id] = one_replay

    accepted.sort(key=lambda r: r["id"])
    rejected.sort(key=lambda r: r["id"])

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for record in accepted:
        (val if _val_bucket(record["id"]) == 0 else train).append(record)

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "train.jsonl", train)
    _write_jsonl(out_dir / "val.jsonl", val)
    _write_jsonl(out_dir / "rejects.jsonl", rejected)

    by_family: dict[str, dict[str, int]] = {}
    by_kind: dict[str, dict[str, int]] = {}
    for record in accepted:
        by_family.setdefault(record["family"], {"accepted": 0, "rejected": 0})["accepted"] += 1
        by_kind.setdefault(record["kind"], {"accepted": 0, "rejected": 0})["accepted"] += 1
    for row in rejected:
        fam = row["family"] or "unknown"
        knd = row["kind"] or "unknown"
        by_family.setdefault(fam, {"accepted": 0, "rejected": 0})["rejected"] += 1
        by_kind.setdefault(knd, {"accepted": 0, "rejected": 0})["rejected"] += 1

    manifest = {
        "tools_sha": sha,
        "holdout_sha256": holdout_sha,
        "seed": args.seed,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "train": len(train),
        "val": len(val),
        "by_family": by_family,
        "by_kind": by_kind,
        "rejection_reasons": reason_histogram,
    }
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_path.write_text(manifest_text, encoding="utf-8")

    if args.export == "unsloth":
        _write_unsloth(
            train, out_dir / "unsloth_train.jsonl", tools_text=tools_text, inline=args.inline
        )
        _write_unsloth(
            val, out_dir / "unsloth_val.jsonl", tools_text=tools_text, inline=args.inline
        )

    if args.gallery:
        gl = headless.open_gl()
        if gl is None:
            print("warning: no GL 3.3 context -- skipping --gallery.", file=sys.stderr)
        else:
            gallery_dir = GALLERY_DIR if out_dir == DATASET_DIR.resolve() else out_dir / "gallery"
            gallery_dir.mkdir(parents=True, exist_ok=True)
            for record in accepted:
                one_replay = replays.get(record["id"])
                if one_replay is not None and one_replay.doc is not None:
                    render.render_gallery(gl, one_replay.doc, gallery_dir / f"{record['id']}.png")

    # Idempotent even when no GL was ever opened this run -- see
    # agent_clay.release()'s own docstring; harmless to call unconditionally.
    agent_clay.release()

    token_estimate = len(tools_text) // 4
    print(
        f"accepted {len(accepted)} / rejected {len(rejected)} "
        f"(train {len(train)}, val {len(val)}); tools_sha={sha}; "
        f"tool card ~{token_estimate} tokens"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
