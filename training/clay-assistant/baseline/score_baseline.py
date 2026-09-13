"""Score a baseline run through the real tool door.

Reads every ``<subject>.md`` (or ``<subject>.card.md`` with ``--card``) beside this file,
pulls the first fenced JSON block out of the raw reply, wraps it as the single
``clay_batch`` call the task asked for, and replays it through ``gen.verify`` -- the same
door and the same checks every training row must pass. Prints one row per subject and a
final tally, and writes ``scores<suffix>.json`` beside the replies so the numbers are kept
with the evidence rather than only in a chat log.

A reply that has no JSON, whose JSON is not an object with a ``calls`` list, or that
nests a ``clay_batch`` inside itself, is scored ``no parse``; one the door refuses is
scored ``refused`` with the refusal text; one that builds but fails a verify check
(diagnose findings, ground, degenerate bounds) is ``built, failed <check>``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training" / "clay-assistant"))

FENCE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.S)


def _extract(md: str) -> dict | None:
    body = md.split("## Raw reply (verbatim)", 1)[-1]
    m = FENCE.search(body)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--card", action="store_true")
    args = ap.parse_args()
    suffix = ".card" if args.card else ""

    from gen import verify

    rows = []
    for path in sorted(HERE.glob(f"*{suffix}.md")):
        if not args.card and path.name.endswith(".card.md"):
            continue
        subject = path.name[: -len(f"{suffix}.md")]
        obj = _extract(path.read_text(encoding="utf-8"))
        if obj is None or not isinstance(obj.get("calls"), list):
            rows.append({"subject": subject, "outcome": "no parse", "detail": ""})
            continue
        calls = obj["calls"]
        replay = verify.replay(calls)  # replay() folds the list into one clay_batch itself
        reasons = verify.check(replay)
        if replay.main_result.get("isError"):
            outcome, detail = "refused", verify._refusal_text(replay.main_result)
        elif reasons:
            outcome, detail = "built, failed check", "; ".join(reasons)
        else:
            outcome, detail = "accepted", ""
        rows.append(
            {
                "subject": subject,
                "outcome": outcome,
                "detail": detail[:300],
                "sub_calls": len(calls),
                "objects": len(replay.doc.objects) if replay.doc is not None else 0,
            }
        )

    for r in rows:
        print(f"{r['subject']:<10} {r['outcome']:<20} {r['detail'][:110]}")
    tally = {
        k: sum(1 for r in rows if r["outcome"] == k) for k in sorted({r["outcome"] for r in rows})
    }
    print("tally:", json.dumps(tally))
    (HERE / f"scores{suffix}.json").write_text(
        json.dumps(
            {"variant": "card" if args.card else "instructions", "rows": rows, "tally": tally},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
