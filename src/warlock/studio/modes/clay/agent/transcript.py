"""The one definition of a Clay agent transcript -- read by tier one's replay
(``tests/modes/clay/test_agent_transcripts.py``) and written by tier two's recorder
(``studio/agent_host.py``, gated on ``WARLOCK_AGENT_TRANSCRIPT``).

Before this module existed, tier one defined all four of the functions below
privately, for its own replay. Tier two's recorder needs the *same* two
rules -- which argument names carry a uid, and which uids a result surfaced
-- because a recording and a replay that used two independently-maintained
copies of "what counts as a uid" could quietly drift apart the day a
twenty-seventh tool named one a third way, and the drift would show up as a
replay failure that named the wrong cause. So the rules live here, once, in
``src/`` where both sides can import them, and the test file imports rather
than redefines them -- the same hand-kept-duplicate drift CLAUDE.md already
refuses everywhere else in this codebase.

See ``tests/modes/clay/test_agent_transcripts.py``'s own module docstring for the
format itself (one JSON object per line, ``tool``/``arguments``/``ok``/
``made``, plus ``error`` on a refusal) and the reasoning behind every field
in it -- that specification did not move, only the functions that implement
half of it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import dispatch as agent_clay

# --- the uid-bearing key set, derived rather than hand-kept ------------------


def _all_property_names(schema: Any) -> set[str]:
    """Every key that appears as a member of some ``properties`` object,
    anywhere inside *schema* -- an object's own top level, a nested object
    (``clay_diagnose``'s ``select``), or the item schema of an array
    (``clay_batch``'s ``calls``). JSON Schema nests objects and arrays
    arbitrarily, so this recurses into every dict value and every list
    element rather than assuming ``properties`` only ever sits at the root.
    """
    names: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            props = node.get("properties")
            if isinstance(props, dict):
                names.update(props.keys())
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(schema)
    return names


def uid_keys() -> frozenset[str]:
    """Every published tool's own argument names, filtered to the ones that
    actually name a uid. Filtering on the substring "uid" rather than a
    hand-picked pair is what makes this a gate rather than a guess: a future
    tool's ``target_uid`` shows up here with no edit needed in this function,
    and ``tests/modes/clay/test_agent_transcripts.py::
    test_the_uid_bearing_argument_names_are_exactly_uid_and_uids`` is what
    turns a *new* name showing up here into a loud failure instead of a
    silent miss in :func:`remap`.
    """
    names: set[str] = set()
    for tool in agent_clay.tools():
        names |= _all_property_names(tool.schema)
    return frozenset(name for name in names if "uid" in name.lower())


UID_KEYS = uid_keys()
"""What :func:`remap` and :func:`produced_uids` treat as a uid-bearing key --
computed once, from the live schemas, at import time, rather than written
out as a ``{"uid", "uids"}`` literal: the whole point of deriving it is that
a future tool naming a uid some other way (a hypothetical ``target_uid``)
changes *this* set with no edit here, so both functions below pick it up
automatically rather than silently skipping it. The literal
``{"uid", "uids"}`` appears exactly once in this codebase, in
``tests/modes/clay/test_agent_transcripts.py``, as today's pinned expectation -- a
human's claim about what the derivation should equal, not the derivation
itself."""


# --- the one rule for "what did this result produce" ------------------------


def produced_uids(result: dict) -> list[int]:
    """Every uid a tool call's result surfaced: every scalar value under a
    key named ``uid``, and every integer member of every list value under a
    key named ``uids``, walked out of *result* in the order ``json`` already
    preserves (Python dicts and ``json.loads`` both keep insertion order, and
    every result here was built by ``dict`` literals in ``studio/modes/clay/agent/dispatch.py``
    itself, so that order is the module's own, not an accident of this
    walk).

    Run directly against the raw ``call()`` return value -- ``{"content":
    [...], "isError": ..., "structuredContent": {...}}`` -- rather than a
    pre-parsed payload: ``content``'s text block is a JSON *string*, a leaf
    this walk does not parse, so it contributes nothing and there is no risk
    of double-counting a uid that also appears, structurally, in
    ``structuredContent``. A tool whose reply carries a picture
    (``clay_render``, ``clay_reference_get``) has no ``structuredContent`` at
    all and so never produces anything here -- correct, since neither tool
    creates an object.

    One rule, used on both sides of the transcript format: tier one's replay
    (``tests/modes/clay/test_agent_transcripts.py::_replay``) and tier two's recorder
    (``studio/agent_host.py``), so the two cannot disagree about what
    "produced" means.

    Gated on :data:`UID_KEYS` rather than the two literal names directly, so
    a third derived key (see that constant's own docstring) is at least
    *noticed* here -- membership is checked before the singular/plural shape
    is -- even though extracting it correctly would still need this
    function's own edit to say whether it reads like ``uid`` or ``uids``;
    the derivation gate test in ``tests/modes/clay/test_agent_transcripts.py`` is what
    turns that "silently extracts nothing for the new key" gap into a loud,
    immediate failure instead of a shape this function would otherwise have
    to guess at.
    """
    produced: list[int] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in UID_KEYS:
                    if key == "uid" and isinstance(value, int) and not isinstance(value, bool):
                        produced.append(value)
                    elif key == "uids" and isinstance(value, list):
                        produced.extend(
                            v for v in value if isinstance(v, int) and not isinstance(v, bool)
                        )
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(result)
    return produced


class UnmappedUidError(ValueError):
    """Raised by :func:`remap` when a recorded uid names an object no
    earlier line in the transcript is known to have produced.

    A plain ``ValueError`` subclass rather than ``AssertionError``: this
    module lives in ``src/``, and an ``AssertionError`` raised as control
    flow here would run even under ``python -O`` (it is not a Python
    ``assert`` statement, so that flag would not remove it) but would still
    read, to any other caller, as "this module's own internal assumption
    broke" rather than "the input handed to it was bad" -- the distinction
    ``src/`` code is expected to keep. ``tests/modes/clay/test_agent_transcripts.py``
    catches this and re-raises it as the ``AssertionError`` its own replay
    loop has always raised, so the test-visible failure message is unchanged.
    """


def remap(
    arguments: dict, mapping: dict[int, int], transcript: str, line_no: int
) -> dict:
    """*arguments*, with every recorded uid under a :data:`UID_KEYS` key
    replaced by its live counterpart in *mapping* -- built fresh, since the
    schema declares each tool's own arguments a plain object with no fixed
    shape ``agent_clay`` will ever hand-list twice (``clay_batch``'s own
    ``arguments: {"type": "object"}`` is the extreme case: this function has
    no idea what is inside one of its entries beyond "maybe a uid, maybe a
    $ref, maybe neither").

    A ``{"$ref": "<name>"}`` dict -- ``clay_batch``'s own placeholder --
    passes through unchanged: it is not a recorded uid, and
    ``agent_clay._resolve_batch_ref`` resolves it against the live document
    by name once the call actually runs, which is a job this function has no
    business doing (and would get wrong, since a ``$ref`` names an object
    this replay may not even have remapped a uid for yet -- it was never
    given one to remap in the first place).

    A recorded uid absent from *mapping* raises :class:`UnmappedUidError`
    rather than passing the original int through: an unmapped uid sailing
    into a live call refuses for "no such object" or, worse, silently
    addresses whatever the fresh process happened to number that uid, which
    is exactly the confusing downstream refusal this whole mechanism exists
    to avoid. *transcript* and *line_no* are folded into the message so a
    failure names the file and the line, not just the number.
    """

    def remap_value(value: Any) -> Any:
        if isinstance(value, dict):
            return value  # a {"$ref": ...} placeholder -- batch-only, left alone.
        if isinstance(value, list):
            return [remap_value(item) for item in value]
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            if value not in mapping:
                raise UnmappedUidError(
                    f"{transcript}: line {line_no}: recorded uid {value} has no live "
                    "mapping -- no earlier line in this transcript produced it."
                )
            return mapping[value]
        return value

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            return {
                key: (remap_value(value) if key in UID_KEYS else walk(value))
                for key, value in node.items()
            }
        if isinstance(node, list):
            return [walk(item) for item in node]
        return node

    return walk(arguments)


# --- writing a transcript, one line at a time --------------------------------


def refusal_text(result: dict) -> str:
    """The sentence a refused call answered with, or ``""`` if it carried
    none -- the first ``text`` block of *result*'s own ``content``.

    Its own function rather than two lines inside :func:`record` because it
    is the third rule this module owns that both halves of a transcript have
    to agree on, and the 2026-09-15 Clay agent benchmark sitting is what
    earned it (``dev/measurements/2026-09-15-clay-agent-benchmark-results.md``).
    That sitting recorded twelve refusals and kept none of their messages, so
    ``2026-09-10-clay-agent-benchmark-preregistration.md``'s rule 5 -- "a
    refusal an agent could not have avoided is a defect... written up as a
    finding" -- had to be answered by replaying the file, and the replay
    could not answer it: three of the twelve refused against live state the
    transcript does not carry, and reproduced as *successes*. The one thing
    that would have settled them was in hand at record time and thrown away.

    The first text block rather than a join of all of them: every refusal
    ``agent_clay.fail`` builds carries exactly one, and ``clay_batch``'s
    envelope carries one whose text is the whole JSON reply -- which is
    precisely what the caller was told, and so precisely what a later reader
    of the transcript needs to see.
    """
    for block in result.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                return text
    return ""


def record(path: Path, tool: str, arguments: dict, result: dict) -> None:
    """Append one call to the transcript at *path*, in tier one's own line
    format -- the same shape :func:`remap`/:func:`produced_uids` above and
    ``tests/modes/clay/test_agent_transcripts.py``'s loader already agree on, so a file
    this writes is a file tier one can replay with no translation step.

    **Append-only and line-oriented, on purpose.** Every call is one
    ``json.dumps`` and one ``write`` of that line plus a trailing newline --
    never a read-modify-write of the whole file. That is the same property
    ``studio/journal.py`` leans on for crash recovery (see its own module
    docstring): a session that is killed mid-write loses at most the one
    line in flight, and every line before it is still there, complete and
    readable, because nothing already on disk is ever rewritten to make room
    for what comes after it. A transcript recorded from a real agent's own
    trajectory -- the thing this function exists for -- has no other
    completion signal to wait on; the file itself has to be safe to read at
    any point a human or a script happens to open it.

    Raises ``OSError`` on a path that cannot be written (a missing parent
    that cannot be created, a read-only directory, and so on) -- this
    function has no opinion about what a caller should do about that. The
    one caller today, ``studio/agent_host.py``'s recorder, catches it and
    logs rather than letting it reach the agent waiting on the call this
    transcript line describes: see that module's own comment for why a
    diagnostic that is off by default must never be able to break a call
    that is on.

    **A refused call also carries ``error``**, :func:`refusal_text`'s answer
    -- the fifth key, written only when the call was refused and only when
    there was a message to write. A line that succeeded has exactly the four
    keys it always had, which is what leaves every fixture already under
    ``tests/fixtures/agent_transcripts/`` valid with nothing to migrate.
    Kept because the alternative was tried: see :func:`refusal_text`.
    """
    ok = not bool(result.get("isError", False))
    entry: dict[str, Any] = {
        "tool": tool,
        "arguments": arguments,
        "ok": ok,
        "made": produced_uids(result),
    }
    # Only on a refusal, and only when there is one -- so every line an
    # earlier build wrote, and every fixture under
    # tests/fixtures/agent_transcripts/, is still exactly this format with
    # nothing to migrate: tier one's loader reads ``ok`` and ``made`` and has
    # never cared what else a line carries.
    if not ok:
        message = refusal_text(result)
        if message:
            entry["error"] = message
    line = json.dumps(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
