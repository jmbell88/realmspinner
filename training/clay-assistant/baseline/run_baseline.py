"""Untuned Gemma 4 E2B baseline over the five held-out corpus subjects, via llama-server.

Two variants, chosen by ``--card``:

* default -- the live ``agent_clay.instructions()`` text plus the one-line task, all in
  one user turn. The model knows the conventions and generator catalogue but not the
  tool names or argument shapes. Run 2026-09-11; results ``<subject>.md``.
* ``--card`` -- the compact tool card the dataset trains on (``gen/convert.compact_tools``)
  as the system turn, the task as the user turn. The fairer "before": the same prefix a
  training row carries. Results ``<subject>.card.md``.

One fresh chat per subject. Raw replies are written verbatim; nothing is cleaned up.
Start the server first, e.g.::

    llama-server -m gemma-4-E2B-it-BF16.gguf --jinja --ctx-size 16384 --port 8081 -ngl 99
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training" / "clay-assistant"))

TASK = (
    'Answer with exactly one clay_batch tool call, as a JSON object with a "calls" list, '
    "that builds: {subject}"
)
SETTINGS = {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "max_tokens": 4096}
URL = "http://127.0.0.1:8081/v1/chat/completions"


def _subjects() -> list[tuple[str, str]]:
    out = []
    for line in (HERE / "subjects.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        key, subject = line.split("|", 1)
        out.append((key.strip(), subject.strip()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--card", action="store_true", help="system turn = the compact tool card")
    args = ap.parse_args()

    from warlock.studio import agent_clay

    if args.card:
        from gen import convert

        system = convert.compact_tools()
        variant, suffix = "compact tool card (gen/convert.compact_tools) as system turn", ".card"
    else:
        system = None
        instructions = agent_clay.instructions()
        variant, suffix = "instructions() text, no schemas, one user turn", ""

    for key, subject in _subjects():
        task = TASK.format(subject=subject)
        if system is None:
            messages = [{"role": "user", "content": instructions + "\n\n" + task}]
        else:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": task},
            ]
        body = {"model": "gemma-4-E2B-it-BF16", "messages": messages, **SETTINGS}
        headers = {"Content-Type": "application/json"}
        req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers=headers)
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=900) as resp:
            data = json.load(resp)
        dt = time.time() - t0
        choice = data["choices"][0]
        content = choice["message"].get("content") or ""
        tool_calls = choice["message"].get("tool_calls")
        usage = data.get("usage", {})
        md = [
            f"# Baseline: {key}",
            "",
            f"Subject: {subject}",
            f"Variant: {variant}",
            "Model: unsloth/gemma-4-E2B-it-GGUF, gemma-4-E2B-it-BF16.gguf, untuned",
            "Runtime: llama-server (Unsloth build 10909, commit 329b6160f), "
            "--jinja, ctx 16384, -ngl 99",
            f"Sampling: {json.dumps(SETTINGS)}",
            f"Date: {time.strftime('%Y-%m-%d')}",
            f"Tokens: prompt {usage.get('prompt_tokens')}, "
            f"completion {usage.get('completion_tokens')}; "
            f"wall {dt:.1f} s; finish_reason {choice.get('finish_reason')}",
            "",
            "## Prompt (user turn)",
            "",
            task,
            "",
            "## Raw reply (verbatim)",
            "",
            content,
        ]
        if tool_calls:
            md += [
                "",
                "## tool_calls field (verbatim)",
                "",
                "```json",
                json.dumps(tool_calls, indent=2),
                "```",
            ]
        (HERE / f"{key}{suffix}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        print(
            f"{key}: {dt:.1f}s, {usage.get('completion_tokens')} tokens, "
            f"finish={choice.get('finish_reason')}, "
            f"tool_calls={'yes' if tool_calls else 'no'}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
