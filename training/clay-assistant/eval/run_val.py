"""Tier one of the assistant eval: generate on held-out prompts, replay through the door.

Points at a running ``llama-server`` (``--jinja``, port 8081 by default) serving either the
untuned base or a run's GGUF, sends every ``dataset/val.jsonl`` record (and, with
``--corpus``, the five pre-registered subjects) as the exact prompt shape training used --
the compact tool card as the system turn, then one user turn (for edit/query rows the
scene turn and the request folded into one, as ``train_a.py`` folds them) -- and scores
each reply the only way that matters: parse the fenced JSON, fold it into one
``clay_batch``, replay through ``gen.verify`` against the record's own ``prior`` if it has
one, and apply ``verify.check``. A ``query`` row scores on "answered in prose with no tool
call and no fence".

Reports per kind and per family: parse rate, door acceptance, check-clean rate, and mean
sub-call count. Writes ``out/<run>/eval-<tag>.json`` with every row's outcome so a later
comparison can be made row for row. Loss is not reported here on purpose: it says nothing
about whether the batch is accepted.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import pathlib
import re
import sys
import time
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
PKG = HERE.parent
ROOT = PKG.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(PKG))

FENCE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.S)
SETTINGS = {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "max_tokens": 4096}


def _chat(url: str, system: str, user: str, timeout: float) -> dict:
    body = {
        "model": "x",
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        **SETTINGS,
    }
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    choice = data["choices"][0]
    return {
        "content": choice["message"].get("content") or "",
        "finish": choice.get("finish_reason"),
        "completion_tokens": data.get("usage", {}).get("completion_tokens"),
    }


def _user_turn(record: dict) -> str:
    """The user turn exactly as training saw it: scene (if any), blank line, request."""
    from gen import convert

    if record["kind"] in ("edit", "query"):
        from gen import verify

        prior_replay = verify.replay(record["prior"])
        scene = convert.compact_scene(verify.scene(prior_replay))
        return (
            "Here is the scene:\n" + json.dumps(scene, sort_keys=True) + "\n\n" + record["prompt"]
        )
    return record["prompt"]


def _score(record: dict, reply: str) -> dict:
    from gen import verify

    m = FENCE.search(reply)
    if record["kind"] == "query":
        ok = m is None and reply.strip() != ""
        return {"outcome": "accepted" if ok else "fenced or empty", "sub_calls": 0}
    if m is None:
        return {"outcome": "no parse", "detail": "no fenced json"}
    try:
        obj = json.loads(m.group(1))
    except json.JSONDecodeError as exc:
        return {"outcome": "no parse", "detail": f"json: {exc}"[:200]}
    calls = obj.get("calls") if isinstance(obj, dict) else None
    if not isinstance(calls, list) or not calls:
        return {"outcome": "no parse", "detail": "no calls list"}
    replay = verify.replay(calls, prior=record.get("prior"))
    if replay.main_result.get("isError"):
        return {
            "outcome": "refused",
            "detail": verify._refusal_text(replay.main_result)[:300],
            "sub_calls": len(calls),
        }
    reasons = verify.check(replay, allow_below_ground=bool(record.get("allow_below_ground")))
    if reasons:
        return {
            "outcome": "built, failed check",
            "detail": "; ".join(reasons)[:300],
            "sub_calls": len(calls),
        }
    return {
        "outcome": "accepted",
        "sub_calls": len(calls),
        "objects": len(replay.doc.objects) if replay.doc is not None else 0,
    }


def _corpus_records() -> list[dict]:
    from campaign_props import read_corpus

    path = ROOT / "docs" / "measurements" / "corpora" / "clay-agent-v1.txt"
    return [
        {"id": f"corpus-{i}", "family": "corpus", "kind": "build", "prompt": s.prompt}
        for i, s in enumerate(read_corpus(path), start=1)
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:8081/v1/chat/completions")
    ap.add_argument("--run", default="run-A", help="out/<run>/ to write eval-<tag>.json into")
    ap.add_argument("--tag", required=True, help="e.g. base, A-q8, A-q4")
    ap.add_argument("--corpus", action="store_true", help="also the five held-out subjects")
    ap.add_argument("--limit", type=int, default=0, help="first N val rows only (smoke test)")
    ap.add_argument("--workers", type=int, default=4, help="parallel requests (server slots)")
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    from gen import convert

    system = convert.compact_tools()
    records = [
        json.loads(line)
        for line in (PKG / "dataset" / "val.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        records = records[: args.limit]
    if args.corpus:
        records += _corpus_records()

    # Generation is parallel (the server has slots); scoring is serial, because the door
    # is one process-global agent_clay/clay_mode state and was never meant to be shared.
    t0 = time.time()
    users = {r["id"]: _user_turn(r) for r in records}

    def gen(r: dict) -> tuple[str, dict]:
        return r["id"], _chat(args.url, system, users[r["id"]], args.timeout)

    replies: dict[str, dict] = {}
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for rid, reply in pool.map(gen, records):
            replies[rid] = reply
            print(f"  generated {len(replies)}/{len(records)}", end="\r", flush=True)
    gen_s = time.time() - t0

    rows = []
    for r in records:
        reply = replies[r["id"]]
        score = _score(r, reply["content"])
        rows.append(
            {
                "id": r["id"],
                "family": r["family"],
                "kind": r["kind"],
                "prompt": r["prompt"],
                "reply": reply["content"],
                "finish": reply["finish"],
                "completion_tokens": reply["completion_tokens"],
                **score,
            }
        )

    def rate(sub: list[dict], outcome: str) -> str:
        n = len(sub)
        k = sum(1 for x in sub if x["outcome"] == outcome)
        return f"{k}/{n}"

    print(f"\ngenerated {len(rows)} replies in {gen_s:.0f}s")
    groups = {"all": rows}
    for key in ("kind", "family"):
        for value in sorted({x[key] for x in rows}):
            groups[f"{key}={value}"] = [x for x in rows if x[key] == value]
    summary = {}
    for name, sub in groups.items():
        acc = rate(sub, "accepted")
        parse_fail = rate(sub, "no parse")
        refused = rate(sub, "refused")
        failed = rate(sub, "built, failed check")
        calls = [x.get("sub_calls", 0) for x in sub if x.get("sub_calls")]
        mean_calls = sum(calls) / len(calls) if calls else 0.0
        summary[name] = {
            "accepted": acc,
            "no_parse": parse_fail,
            "refused": refused,
            "failed_check": failed,
            "mean_sub_calls": round(mean_calls, 1),
        }
        print(
            f"{name:<22} accepted {acc:>8}  no-parse {parse_fail:>8}  refused {refused:>8}  "
            f"failed-check {failed:>8}  calls {mean_calls:.1f}"
        )
    refusals: dict[str, int] = {}
    for x in rows:
        if x["outcome"] in ("refused", "built, failed check", "no parse"):
            key = (x.get("detail") or "")[:80]
            refusals[key] = refusals.get(key, 0) + 1
    print("top reasons:")
    for key, n in sorted(refusals.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {n:>3}  {key}")

    out = PKG / "out" / args.run
    out.mkdir(parents=True, exist_ok=True)
    (out / f"eval-{args.tag}.json").write_text(
        json.dumps({"tag": args.tag, "summary": summary, "rows": rows}, indent=1),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
