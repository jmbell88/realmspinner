"""Tier two of Clay's MCP agent benchmark: a real model, driving a real app.

``tests/test_agent_transcripts.py``'s module docstring names the split. Tier
one replays a transcript -- hand-authored, or promoted from a recorded
session -- against a bare ``ClayDoc`` double, with no model and no window,
and runs in the suite on every push. Tier two is this
file plus ``studio/agent_transcript.py`` (the recorder) and
``studio/agent_host.py`` (which calls it): the part that cannot run in CI at
all, because it needs a real MCP-speaking model, a real window and a real GL
context, driving the actual app over the actual named pipe an agent uses in
production. Tier three -- human judgement over what tier two built, against
renders rather than numbers -- is a human sitting with the pictures this
run produced, which is a person's afternoon and not a script.

Two subcommands, and only two:

``--serve``
    Isolate a throwaway home (see ``_appharness.isolate_home``), switch the
    agent bridge on in *that* home's settings, point
    ``WARLOCK_AGENT_TRANSCRIPT`` at a file, print the one command line a
    human needs to connect a real MCP client, and then run the real app --
    ``warlock.studio.main.run()``, the same entry point ``warlock`` itself
    calls, never a second copy of its loop. A human points a real model at
    the printed command, the model builds something in Clay (a corpus
    subject -- a chair, a spoked hub, whatever this run's subject is), and
    closing the app leaves a recorded transcript on disk.

``--show <path>``
    Read a transcript back in a form a person can review before promoting it
    to a fixture under ``tests/fixtures/agent_transcripts/`` -- every call,
    whether it was refused, and what it made. Cheap, and does not need the
    app, a model, or a GPU: it is a plain read of the JSON Lines file
    ``agent_transcript.record`` wrote.

**Deliberately not built: a replayer that drives a *running* app over the
pipe.** ``tests/test_agent_transcripts.py`` already replays a transcript --
faster, with no window and no GL, and with assertions against an
``.expect.json`` a promoted fixture carries. A second replayer here would
need its own copy of the remap loop (:func:`warlock.studio.agent_transcript
.remap`'s own mapping-by-position walk) to mean the same thing "replay" means
in the suite, for no gain over the one that already exists and already runs
in CI. This is a decision, not a gap left for later.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
# ``campaign_props.read_corpus`` lives beside this file; the sibling import
# below needs scripts/ on the path when this is run as a path rather than as
# a module, which is how every other script in here is run.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Set *before* the import below, not after, and that ordering is the whole
# point: isolate_home() runs at _appharness's own import time and decides
# there and then whether to register the cleanup, so a line under the import
# would be a line too late. This script is the one caller whose entire output
# -- the Library rows clay_export mints -- lives inside the throwaway home,
# and the 2026-09-15 Clay agent benchmark sitting lost both of a graded session's exports to
# that cleanup. ``setdefault`` so a human who set it to "" on purpose keeps
# the deletion.
os.environ.setdefault("WARLOCK_HARNESS_KEEP_HOME", "1")  # noqa: E402

# Imported for its side effect, not its names: _appharness.isolate_home() runs
# at *its own* import time (see that module's docstring on why advice was not
# enough) and points this process at a throwaway WARLOCK_HOME before anything
# below ever calls warlock.config.get_config(). An explicit WARLOCK_HOME (a
# human pointing this at a prepared library on purpose) is left alone either
# way -- see isolate_home's own docstring for that escape hatch.
import _appharness  # noqa: E402, F401


def _serve(transcript: Path) -> int:
    """Switch the agent bridge on in this process's (throwaway, per
    ``_appharness``) home, point the recorder at *transcript*, and hand off
    to the real app loop -- ``studio.main.run()`` itself, not a
    reimplementation of it, so this command exercises exactly the code path
    a person double-clicking the installed app does.

    The setting is written to disk, not merely held on a live ``Settings``
    instance: ``run()`` builds its own ``App``, which loads ``Settings``
    fresh from ``config.data_dir`` (``main.py``'s own comment on why that
    read happens before the window exists), so the only way this process's
    choice reaches that later, independent load is the settings file itself.
    """
    import os

    from warlock.config import get_config
    from warlock.studio import agent_host
    from warlock.studio.main import AGENT_SERVER_SETTING, run
    from warlock.studio.settings import Settings

    config = get_config()
    settings = Settings.load(config.data_dir)
    settings.set(AGENT_SERVER_SETTING, True)
    settings.flush()

    transcript_path = transcript.resolve()
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ[agent_host.TRANSCRIPT_ENV] = str(transcript_path)

    home = config.home
    print(f"WARLOCK_HOME={home}")
    print()
    print("The agent bridge is switched on for this home (the equivalent of")
    print("Settings -> Advanced -> Allow AI agents to drive the Studio).")
    print("In a shell that carries the same WARLOCK_HOME -- so `warlock mcp`")
    print("finds *this* session and not your real library -- run once:")
    print()
    print(f'    WARLOCK_HOME="{home}" claude mcp add warlock -- uv run warlock mcp')
    print()
    print("Every completed tool call will be appended, as it happens, to:")
    print(f"    {transcript_path}")
    print()
    _print_corpus()
    print("Build one of those subjects with it, then close the window when done.")
    print("Read the result back with:")
    print(f"    uv run python scripts/agent_bench.py --show {transcript_path}")
    print()
    print("Starting Warlock Studio...")

    status = run()

    # Said again on the way out, because by now the interesting thing in that
    # directory exists: every Library row clay_export minted during the
    # session, which the pre-registration's retention rule says to keep until
    # the results document is written. The home is kept rather than deleted
    # (see the WARLOCK_HARNESS_KEEP_HOME line at the top of this file), so
    # this is a path a human can still walk into -- and a directory nobody is
    # told about is a leak rather than a retention.
    print()
    print("This session's throwaway home was kept, not deleted:")
    print(f"    {home}")
    print("Its assets/ holds every GLB clay_export minted. Delete it yourself")
    print("once the results are written up.")
    return status


#: The subjects a tier-two session is run against, pre-registered in
#: ``docs/measurements/2026-09-10-clay-agent-benchmark-preregistration.md``
#: before any of them was graded. Printed by ``--serve`` rather than left as a
#: document nobody opens: a benchmark whose corpus is "whatever the operator
#: thought of that morning" is not a benchmark, and the cheapest way to keep
#: that honest is for the tool that stands the session up to read the list out.
CORPUS = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "measurements"
    / "corpora"
    / "clay-agent-v1.txt"
)


def _print_corpus() -> None:
    """Read the corpus out, through ``campaign_props.read_corpus``.

    That reader rather than a `split("|")` here: it already refuses a
    malformed line and an unknown class instead of quietly dropping either,
    and its own docstring gives the reason -- "a corpus silently one subject
    short is a corpus whose N does not mean what the writeup says". The
    three image corpora in the same directory are parsed by it too, so this
    one cannot drift into a second dialect of the same file format.
    """
    from campaign_props import read_corpus

    subjects = read_corpus(CORPUS)
    print(f"The pre-registered corpus ({CORPUS.name}), one subject per session:")
    for subject in subjects:
        print(f"    [{subject.cls}] {subject.prompt}")
    print()


def _show(path: Path) -> int:
    """Print *path* -- a tier-two transcript -- call by call, then a count.

    A plain read of the format ``agent_transcript.record`` and tier one's
    own ``_load_transcript`` both already agree on: this prints the recorded
    fields rather than recomputing any of them (there is nothing to
    recompute -- ``ok``, ``made`` and a refusal's ``error`` are exactly what
    the recorder decided they were when the call actually ran), so a reader
    sees precisely what is on disk, which is the point of a review step that
    exists to catch a recording gone wrong before it becomes a fixture.
    """
    lines = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for i, line in enumerate(lines, start=1):
        outcome = "ok" if line.get("ok") else "REFUSED"
        made = line.get("made") or []
        made_str = f"  -> made {made}" if made else ""
        args_str = json.dumps(line.get("arguments") or {}, sort_keys=True)
        print(f"{i:>4}. {outcome:<8} {line.get('tool', '?'):<28} {args_str}{made_str}")
        # The refusal's own sentence, on its own line under the call. Absent
        # from anything recorded before 2026-09-15 (see
        # ``agent_transcript.refusal_text`` for what that cost the
        # 2026-09-15 benchmark sitting), so this prints what is there and says nothing when there
        # is nothing -- an old transcript still reads exactly as it did.
        error = line.get("error")
        if error:
            print(f"      {error}")

    ok_count = sum(1 for line in lines if line.get("ok"))
    made_count = sum(len(line.get("made") or []) for line in lines)
    print()
    print(
        f"{path}: {len(lines)} call(s), {ok_count} ok, {len(lines) - ok_count} refused, "
        f"{made_count} uid(s) produced"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--serve",
        action="store_true",
        help="boot the real app with the agent bridge on, recording a transcript",
    )
    group.add_argument(
        "--show",
        type=Path,
        metavar="PATH",
        help="print a recorded transcript in human-readable form",
    )
    ap.add_argument(
        "--transcript",
        type=Path,
        default=Path("agent_bench_transcript.jsonl"),
        help="--serve only: where to record the transcript (default: %(default)s)",
    )
    args = ap.parse_args()

    if args.show is not None:
        return _show(args.show)
    return _serve(args.transcript)


if __name__ == "__main__":
    raise SystemExit(main())
