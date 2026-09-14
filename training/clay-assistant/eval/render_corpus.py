"""Render the five held-out corpus rows from a ``run_val.py`` eval JSON so a human can
eyeball what a run actually built, without re-running the model: for each ``corpus-1``..
``corpus-5`` row, replay its first sample's reply through the real door
(``gen.verify.replay``) and write ``<slug>.calls.json`` (the parsed ``calls`` list, matching
``docs/measurements/data/clay-assistant/run-A/chair.calls.json``'s own format exactly) and
``<slug>.png`` (``gen.render.render_gallery``'s three-quarter/front pair).

Renders even a batch that refused partway -- a partial scene is still worth seeing -- and
skips the PNG (but still writes ``.calls.json``) only when nothing was built at all, or
there is no GL context to render with.

    uv run python training/clay-assistant/eval/render_corpus.py out/run-A/eval-A-q8.json OUT_DIR
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
PKG = HERE.parent
ROOT = PKG.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(HERE))

from run_val import parse_calls  # noqa: E402

SLUGS: tuple[str, ...] = ("chair", "bracket", "telescope", "colonnade", "serpent")
"""``corpus-1``..``corpus-5``, in ``baseline/subjects.txt``'s own order (that file's header
explains why: the corpus's own line order, carried over verbatim from the external
reviewer's five subjects)."""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("eval_json", type=pathlib.Path)
    ap.add_argument("out_dir", type=pathlib.Path)
    args = ap.parse_args()

    from gen import headless, render, verify

    from warlock.studio import agent_clay

    data = json.loads(args.eval_json.read_text(encoding="utf-8"))
    rows_by_id = {r["id"]: r for r in data["rows"]}

    gl = headless.open_gl()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    try:
        for i, slug in enumerate(SLUGS, start=1):
            row_id = f"corpus-{i}"
            row = rows_by_id.get(row_id)
            if row is None:
                print(f"{slug}: no {row_id} row in {args.eval_json}")
                continue

            reply = row["samples"][0]["reply"] if row.get("samples") else row["reply"]
            calls, detail = parse_calls(reply)
            if calls is None:
                print(f"{slug}: {detail}, 0 objects")
                continue

            (args.out_dir / f"{slug}.calls.json").write_text(
                json.dumps(calls, indent=1, sort_keys=True), encoding="utf-8"
            )

            replay = verify.replay(calls, prior=row.get("prior"))
            doc = replay.doc
            n_objects = len(doc.objects) if doc is not None else 0
            outcome = "refused" if replay.main_result.get("isError") else "ok"

            if doc is not None and n_objects and gl is not None:
                render.render_gallery(gl, doc, args.out_dir / f"{slug}.png")
            elif gl is None:
                print(f"{slug}: no GL context, skipping PNG")

            print(f"{slug}: {outcome}, {n_objects} objects")
    finally:
        agent_clay.release()

    return 0


if __name__ == "__main__":
    sys.exit(main())
