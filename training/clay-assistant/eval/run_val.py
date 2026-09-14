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

Reports per kind and per family: parse rate, door acceptance (first-sample and, with
``--samples`` > 1, the per-sample mean), check-clean rate, and mean sub-call count. Writes
``out/<run>/eval-<tag>.json`` with every row's outcome (and, per sample, every sample's own
outcome) so a later comparison (``compare.py``) can be made row for row. Loss is not
reported here on purpose: it says nothing about whether the batch is accepted.

Tag convention: ``<run>-<quant>-t<temp>[-nN]``, e.g. ``A-q8-t0`` (run A, Q8_0, greedy),
``A-q4-t0.2-n3`` (run A, Q4_K_M, temperature 0.2, 3 samples per row). Greedy is
``--temperature 0 --top-k 1``.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import pathlib
import sys
import time
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
PKG = HERE.parent
ROOT = PKG.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(PKG))
# So ``import compare``/``from compare import ...`` works both when this file is run as a
# script (where Python would already put HERE on sys.path[0] on its own) and when it is
# imported from a test -- an import does not get that automatic insertion.
sys.path.insert(0, str(HERE))

from compare import reason_key  # noqa: E402

from warlock.studio.familiar.contract import FENCE, parse_calls  # noqa: E402,F401

SETTINGS = {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "max_tokens": 4096}


def _chat(
    url: str,
    system: str,
    user: str,
    timeout: float,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    max_tokens: int,
    seed: int,
) -> dict:
    body = {
        "model": "x",
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "seed": seed,
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
    """The user turn exactly as training saw it: scene (if any), blank line, request.

    Deliberately *not* ``contract.user_turn``: this uses ``json.dumps(scene,
    sort_keys=True)`` -- the default separators (``", "``/``": "``) -- while
    ``contract.user_turn`` uses the compact ``separators=(",", ":")`` form
    ``train/train_a.py`` actually trains on. Run A's whole eval corpus
    (``docs/measurements/data/clay-assistant/run-A/``) was scored against
    *this* file's slightly longer encoding, so switching it over now would
    change every prompt's token count and make a new eval no longer
    comparable with those recorded numbers. See ``contract.user_turn``'s own
    docstring for the other side of this."""
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

    if record["kind"] == "query":
        ok = FENCE.search(reply) is None and reply.strip() != ""
        return {"outcome": "accepted" if ok else "fenced or empty", "sub_calls": 0}
    calls, detail = parse_calls(reply)
    if calls is None:
        return {"outcome": "no parse", "detail": detail}
    replay = verify.replay(calls, prior=record.get("prior"))
    if replay.main_result.get("isError"):
        structured = replay.main_result.get("structuredContent") or {}
        stopped_at = structured.get("stopped_at")
        result = {
            "outcome": "refused",
            # Untruncated is fine up to ~500 chars: refusal_reason is already the door's own
            # one-sentence message, not the whole batch result _refusal_text used to return.
            "detail": verify.refusal_reason(replay.main_result)[:500],
            "sub_calls": len(calls),
        }
        if stopped_at is not None:
            result["stopped_at"] = stopped_at
        return result
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


def _load_val(pkg: pathlib.Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (pkg / "dataset" / "val.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _select_by_ids(records: list[dict], ids_file: pathlib.Path) -> list[dict]:
    """*records*, restricted to the ids named in *ids_file* (one per line), in that file's
    own order -- so a rerun against a fixed subset (a smoke test, a diff against a prior
    tag) draws exactly the same rows in the same order every time."""
    wanted = [
        line.strip() for line in ids_file.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    by_id = {r["id"]: r for r in records}
    missing = [i for i in wanted if i not in by_id]
    if missing:
        raise SystemExit(f"--ids names ids not in val.jsonl: {missing}")
    return [by_id[i] for i in wanted]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:8081/v1/chat/completions")
    ap.add_argument("--run", default="run-A", help="out/<run>/ to write eval-<tag>.json into")
    ap.add_argument("--tag", required=True, help="e.g. base, A-q8, A-q4")
    ap.add_argument("--corpus", action="store_true", help="also the five held-out subjects")
    ap.add_argument("--limit", type=int, default=0, help="first N val rows only (smoke test)")
    ap.add_argument("--workers", type=int, default=4, help="parallel requests (server slots)")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--temperature", type=float, default=SETTINGS["temperature"])
    ap.add_argument("--top-k", type=int, default=SETTINGS["top_k"])
    ap.add_argument("--top-p", type=float, default=SETTINGS["top_p"])
    ap.add_argument("--max-tokens", type=int, default=SETTINGS["max_tokens"])
    ap.add_argument("--samples", type=int, default=1, help="generations per row")
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="request seed for sample i of any row is BASE+i (reproducible temp>0 runs)",
    )
    ap.add_argument(
        "--ids", type=pathlib.Path, default=None, help="one id per line; restricts val rows"
    )
    ap.add_argument(
        "--card",
        type=pathlib.Path,
        default=None,
        help="system prompt read verbatim from this file instead of the live compact_tools()",
    )
    args = ap.parse_args()

    from gen import convert

    if args.card:
        system = args.card.read_bytes().decode("utf-8")
        card_source = str(args.card)
    else:
        system = convert.compact_tools()
        card_source = "live"
    card_sha256 = hashlib.sha256(system.encode("utf-8")).hexdigest()

    val_records = _load_val(PKG)
    records = _select_by_ids(val_records, args.ids) if args.ids else val_records
    if args.limit:
        records = records[: args.limit]
    if args.corpus:
        records += _corpus_records()

    # Generation is parallel over every (row, sample) pair (the server has slots); scoring
    # stays serial, because the door is one process-global agent_clay/clay_mode state and
    # was never meant to be shared.
    t0 = time.time()
    users = {r["id"]: _user_turn(r) for r in records}
    pairs = [(r, s) for r in records for s in range(args.samples)]

    def gen(pair: tuple[dict, int]) -> tuple[tuple[str, int], dict]:
        r, s = pair
        reply = _chat(
            args.url,
            system,
            users[r["id"]],
            args.timeout,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            seed=args.seed + s,
        )
        return (r["id"], s), reply

    replies: dict[tuple[str, int], dict] = {}
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for key, reply in pool.map(gen, pairs):
            replies[key] = reply
            print(f"  generated {len(replies)}/{len(pairs)}", end="\r", flush=True)
    gen_s = time.time() - t0

    rows = []
    for r in records:
        samples = []
        for s in range(args.samples):
            reply = replies[(r["id"], s)]
            score = _score(r, reply["content"])
            samples.append(
                {
                    "reply": reply["content"],
                    "finish": reply["finish"],
                    "tokens": reply["completion_tokens"],
                    **score,
                }
            )
        first = samples[0]
        accepted_k = sum(1 for sm in samples if sm["outcome"] == "accepted")
        row = {
            "id": r["id"],
            "family": r["family"],
            "kind": r["kind"],
            "prompt": r["prompt"],
            "samples": samples,
            "accepted_k": accepted_k,
            # Top-level keys kept identical to the pre-``--samples`` shape, copied from the
            # first sample, so a reader that has never heard of "samples" still works.
            "reply": first["reply"],
            "finish": first["finish"],
            "completion_tokens": first["tokens"],
            "outcome": first["outcome"],
        }
        for key in ("detail", "sub_calls", "objects", "stopped_at"):
            if key in first:
                row[key] = first[key]
        rows.append(row)

    def rate(sub: list[dict], outcome: str) -> str:
        n = len(sub)
        k = sum(1 for x in sub if x["outcome"] == outcome)
        return f"{k}/{n}"

    def mean_rate(sub: list[dict], outcome: str) -> str:
        """Pass@1 averaged over ``args.samples`` -- the same "k/n" shape as :func:`rate`,
        but ``k`` is the mean count of samples reaching *outcome* per row rather than only
        the first sample's."""
        n = len(sub)
        total_k = sum(1 for row in sub for sm in row["samples"] if sm["outcome"] == outcome)
        mean_k = total_k / args.samples if args.samples else 0.0
        return f"{mean_k:.1f}/{n}"

    print(f"\ngenerated {len(rows)} replies in {gen_s:.0f}s")
    groups = {"all": rows}
    for key in ("kind", "family"):
        for value in sorted({x[key] for x in rows}):
            groups[f"{key}={value}"] = [x for x in rows if x[key] == value]
    summary = {}
    for name, sub in groups.items():
        acc = rate(sub, "accepted")
        acc_mean = mean_rate(sub, "accepted")
        parse_fail = rate(sub, "no parse")
        refused = rate(sub, "refused")
        failed = rate(sub, "built, failed check")
        calls = [x.get("sub_calls", 0) for x in sub if x.get("sub_calls")]
        mean_calls = sum(calls) / len(calls) if calls else 0.0
        summary[name] = {
            "accepted": acc,
            "accepted_mean": acc_mean,
            "no_parse": parse_fail,
            "refused": refused,
            "failed_check": failed,
            "mean_sub_calls": round(mean_calls, 1),
        }
        print(
            f"{name:<22} accepted {acc:>8}  mean {acc_mean:>10}  no-parse {parse_fail:>8}  "
            f"refused {refused:>8}  failed-check {failed:>8}  calls {mean_calls:.1f}"
        )
    refusals: dict[str, int] = {}
    for row in rows:
        for sample in row["samples"]:
            if sample["outcome"] in ("refused", "built, failed check", "no parse"):
                key = reason_key(sample.get("detail") or "")
                refusals[key] = refusals.get(key, 0) + 1
    print("top reasons:")
    for key, n in sorted(refusals.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {n:>3}  {key}")

    out = PKG / "out" / args.run
    out.mkdir(parents=True, exist_ok=True)
    settings = {
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "samples": args.samples,
        "seed": args.seed,
        "card": card_source,
        "card_sha256": card_sha256,
        "ids": str(args.ids) if args.ids else None,
        "rows": len(records),
        "url": args.url,
    }
    (out / f"eval-{args.tag}.json").write_text(
        json.dumps(
            {"tag": args.tag, "settings": settings, "summary": summary, "rows": rows}, indent=1
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
