"""Tier one of Clay's MCP agent benchmark: replay a **transcript** -- a JSON
Lines sequence of tool calls -- against a real ``ClayDoc`` through the real
``agent_clay.call`` door, and assert what came out. See the tranche-5 plan for
the split: tier one (here) runs unattended in the suite with no model
involved; tier two is a ``scripts/`` driver plus a recorder that produces a
transcript from an actual agent's own trajectory (not yet built -- everything
under ``tests/fixtures/agent_transcripts/`` today is **hand-authored**, not
recorded, and says so below); tier three is human judgement through a
``TODO.md`` sitting. This file's job is narrower than either: a change that
makes a corpus subject unbuildable, or that silently stops a call producing
an object, fails here -- a regression gate, not a benchmark score.

**The format.** One recorded call per line, JSON, at
``tests/fixtures/agent_transcripts/<name>.jsonl``::

    {"tool": "clay_add_primitive", "arguments": {"generator": "box"}, "ok": true, "made": [12]}
    {"tool": "clay_transform", "arguments": {"uid": 12, "translation": [0, 1, 0]},
     "ok": true, "made": []}

``ok`` is whether the recorded call succeeded -- a transcript may legitimately
contain a refusal (a real model's own trajectory includes them), and replay
must reproduce the *same* outcome, not merely avoid crashing. ``made`` is the
object uids that call's result actually carried, in the order
:func:`_extract_produced_uids` below would walk them out -- not necessarily
uids that call *created*: a call that only echoes a uid it acted on (say,
``clay_set_params``'s per-object rows) reports that uid here too, because the
extraction rule is structural ("every uid this result surfaced"), not a
judgement about novelty. That is what keeps the recording and the replay
mechanically unable to disagree about what "produced" means -- there is only
one rule, used both times.

**The uid problem, and the rule that solves it.** ``document.new_uid()``
never reuses a uid for the life of a process, so a transcript recorded in one
process names uids that cannot exist in a fresh test process. The replay
therefore remaps: before each call, every value under a uid-bearing argument
key is looked up in a mapping built from every earlier line's own
``made`` list (zipped against what that line's *live* result actually
produced), and a value with no mapping fails the test loudly, naming the
transcript, the line and the uid -- never silently passed through, because a
pass-through uid would make the next call refuse for a reason that has
nothing to do with the regression this tier exists to catch. See
:func:`_remap_uid_arguments`.

There are, today, exactly two such keys: ``uid`` and ``uids``. That is not
asserted by literal -- :func:`_uid_bearing_argument_names` walks every
published tool's own argument schema (``agent_clay.tools()``, the same
registry ``tests/test_agent_clay.py``'s derivation gate reads), collects
every ``properties`` key at every depth, and keeps the ones whose name
actually contains "uid" -- and ``test_the_uid_bearing_argument_names_are_
exactly_uid_and_uids`` pins the result against ``{"uid", "uids"}``. A
twenty-seventh tool that introduces a third name (a hypothetical
``target_uid``) fails that assertion, which is the point: it forces a
decision here rather than letting the remap rule silently miss an argument
and replay against the wrong object. ``{"$ref": "<name>"}`` -- the
``clay_batch``-only placeholder ``343e4f06`` added, resolved by
``agent_clay._resolve_batch_ref`` against the document's own live names --
is not a recorded uid at all, so :func:`_remap_uid_arguments` passes it
through untouched rather than mistaking it for one; only the batch handler
itself ever learns ``$ref`` exists, exactly as that commit's own docstring
says.

**Recording and expectation are two files, on purpose.** ``<name>.jsonl`` is
what a run of the calls actually produced (machine output, whether recorded
from a real trajectory or, here, from hand-authoring one and running it once
to see what came out); ``<name>.expect.json`` is a human's claim about what
that run *should* mean -- the subject, its difficulty, and the shape the
finished document must have. Folding the two together would let a change
that quietly breaks a call rewrite its own expectation the next time someone
regenerates the recording without looking hard at the diff; kept apart, a
person has to edit the claim file on purpose for the gate to move. Every
number in ``<name>.expect.json`` is provably uid-agnostic -- object counts, a
call count, and per-object diagnose findings keyed by *position* in
``doc.objects`` (creation order, deterministic given the same transcript)
rather than by uid -- so the same file is correct however the live process
happens to have numbered its uids that run.

**Choosing the diagnose claim.** ``clay/diagnose.findings`` (imported here as
``clay_diagnose.findings``) is called directly against each finished object's
own mesh, in ``doc.objects`` order -- not through the ``clay_diagnose`` tool,
whose own JSON answer carries live uids that cannot be hard-coded into an
expectation file. Both transcripts below build nothing but primitives, moved,
sized and arrayed -- no boolean, no manual topology edit -- so every object
should come back clean: an empty finding list is the honest, *exact* claim
(stronger than "no errors" precisely because it is a real, checkable data
shape, not a string) and it is stable because nothing here has a reason to
produce a hole, a flipped edge or a stray vertex. A regression that broke a
generator's manifoldness, or an op that left a seam, would turn one of those
empty lists non-empty and fail the exact comparison -- which is the entire
reason this is asserted per object rather than as one blanket "the document
is clean" boolean.

**The ``wblk_bytes`` round trip.** Assertion 5 below is a stand-in for "the
document exports", not a proof of it: a real GLB export
(``clay_mode.build_asset``) needs ``ctx.svc`` and a Library row, neither of
which this file's ``_Ctx`` double provides (see ``tests/test_agent_clay.py``'s
own module docstring for why that double has no ``svc`` by default). Round-
tripping through ``serialize.wblk_bytes``/``serialize.read_wblk`` instead
proves the finished document's meshes, materials and scene graph survive a
real serialization format with no service layer involved -- cheap, and
enough to catch a corrupted mesh or a scene-graph cycle, but it says nothing
about the GLB path itself (optimize, normalize, ground) or about a
``service``-side failure.

**These two transcripts are hand-authored, not recorded.** A recorder that
watches a real agent's own trajectory and emits a transcript from it is tier
two, and is not built here -- describing either ``chair.jsonl`` or
``spoked-hub.jsonl`` as "recorded" would overclaim what produced them. Both
were written by reasoning about the geometry by hand, then actually run once
(through this same ``agent_clay.call`` door, with the same ``_extract_produced_uids``
this file uses to replay) to read off the real ``made`` lists and the real
call outcomes -- "authored, verified by execution," not "recorded from an
agent."
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from test_agent_clay import _Ctx  # see module docstring -- shared rather than duplicated

from warlock.studio import agent_clay, clay_mode
from warlock.studio.clay import diagnose as clay_diagnose
from warlock.studio.clay import document as bd
from warlock.studio.clay import serialize

FIXTURES = Path(__file__).parent / "fixtures" / "agent_transcripts"


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


def _uid_bearing_argument_names() -> set[str]:
    """Every published tool's own argument names, filtered to the ones that
    actually name a uid. Filtering on the substring "uid" rather than a
    hand-picked pair is what makes this a gate: a future tool's
    ``target_uid`` would show up here with no edit needed in this function,
    and ``test_the_uid_bearing_argument_names_are_exactly_uid_and_uids`` is
    what turns that into a failure instead of a silent miss in
    :func:`_remap_uid_arguments`.
    """
    names: set[str] = set()
    for tool in agent_clay.tools():
        names |= _all_property_names(tool.schema)
    return {name for name in names if "uid" in name.lower()}


UID_KEYS = frozenset(_uid_bearing_argument_names())
"""What :func:`_remap_uid_arguments` and :func:`_extract_produced_uids` treat
as a uid-bearing key -- computed once, from the live schemas, at import time,
rather than written out as a ``{"uid", "uids"}`` literal: the whole point of
deriving it is that a future tool naming a uid some other way (a hypothetical
``target_uid``) changes *this* set with no edit here, so the walkers below
pick it up automatically rather than silently skipping it. The literal
``{"uid", "uids"}`` appears exactly once in this file, in the test below, as
today's pinned expectation -- a human's claim about what the derivation
should equal, not the derivation itself."""


def test_the_uid_bearing_argument_names_are_exactly_uid_and_uids() -> None:
    """The derivation gate the whole remap rule rests on. Walks every
    argument schema :func:`agent_clay.tools` publishes today (26 tools, at
    the time this was written) rather than trusting a hand-written pair, so
    a twenty-seventh tool naming a uid some other way fails loudly here --
    :data:`UID_KEYS` would have already grown to include it, which is what
    forces a decision about :func:`_remap_uid_arguments` and
    :func:`_extract_produced_uids` instead of letting a stale mapping
    silently replay the wrong object.
    """
    assert {"uid", "uids"} == UID_KEYS


# --- the one rule for "what did this result produce" ------------------------


def _extract_produced_uids(result: dict) -> list[int]:
    """Every uid a tool call's result surfaced: every scalar value under a
    key named ``uid``, and every integer member of every list value under a
    key named ``uids``, walked out of *result* in the order ``json`` already
    preserves (Python dicts and ``json.loads`` both keep insertion order, and
    every result here was built by ``dict`` literals in ``agent_clay.py``
    itself, so that order is the module's own, not an accident of this
    walk).

    Run directly against the raw ``call()`` return value -- ``{"content":
    [...], "isError": ..., "structuredContent": {...}}`` -- rather than a
    pre-parsed payload: ``content``'s text block is a JSON *string*, a leaf
    this walk does not parse, so it contributes nothing and there is no risk
    of double-counting a uid that also appears, structurally, in
    ``structuredContent`` (see the module docstring's structured-results
    claim for why every non-picture tool duplicates its answer there). A
    tool whose reply carries a picture (``clay_render``, ``clay_reference_get``)
    has no ``structuredContent`` at all and so never produces anything here
    -- correct, since neither tool creates an object.

    One rule, used for both halves of the replay (recording-time extraction,
    when these fixtures were authored, and replay-time extraction, in
    :func:`_replay`) so the two cannot disagree about what "produced" means.

    Gated on :data:`UID_KEYS` rather than the two literal names directly, so
    a third derived key (see that constant's own docstring) is at least
    *noticed* here -- membership is checked before the singular/plural shape
    is -- even though extracting it correctly would still need this
    function's own edit to say whether it reads like ``uid`` or ``uids``;
    the derivation gate test is what turns that "silently extracts nothing
    for the new key" gap into a loud, immediate failure instead of a shape
    this function would otherwise have to guess at.
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


def _remap_uid_arguments(
    arguments: dict, mapping: dict[int, int], transcript: str, line_no: int
) -> dict:
    """*arguments*, with every recorded uid under a :data:`UID_KEYS` key
    replaced by its live counterpart in *mapping* -- built fresh, since the
    schema declares each tool's own arguments a plain object with no fixed
    shape ``agent_clay`` will ever hand-list twice (``clay_batch``'s own
    ``arguments: {"type": "object"}`` is the extreme case: this file has no
    idea what is inside one of its entries beyond "maybe a uid, maybe a
    $ref, maybe neither").

    A ``{"$ref": "<name>"}`` dict -- ``clay_batch``'s own placeholder, see
    the module docstring -- passes through unchanged: it is not a recorded
    uid, and ``agent_clay._resolve_batch_ref`` resolves it against the live
    document by name once the call actually runs, which is a job this
    function has no business doing (and would get wrong, since a ``$ref``
    names an object this replay may not even have remapped a uid for yet --
    it was never given one to remap in the first place).

    A recorded uid absent from *mapping* raises rather than passing the
    original int through: an unmapped uid sailing into a live call refuses
    for "no such object" or, worse, silently addresses whatever the fresh
    process happened to number that uid, which is exactly the confusing
    downstream refusal the task this file implements was written to avoid.
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
                raise AssertionError(
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


# --- the replay itself --------------------------------------------------------


def _load_transcript(name: str) -> tuple[list[dict], dict]:
    lines = [
        json.loads(line)
        for line in (FIXTURES / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expect = json.loads((FIXTURES / f"{name}.expect.json").read_text(encoding="utf-8"))
    return lines, expect


def _replay(name: str) -> dict[int, int]:
    calls, expect = _load_transcript(name)

    ctx = _Ctx()
    session = agent_clay.Session()
    mapping: dict[int, int] = {}

    for line_no, record in enumerate(calls, start=1):
        arguments = _remap_uid_arguments(
            record.get("arguments") or {}, mapping, name, line_no
        )
        result = agent_clay.call(ctx, session, record["tool"], arguments)

        # Assertion 1: the recorded outcome, reproduced -- a refusal names
        # which line and what the live message was, so a genuine regression
        # (a call that used to succeed now refusing, or vice versa) reads as
        # itself rather than as an assertion-count mismatch three lines down.
        assert result["isError"] is (not record["ok"]), (
            f"{name}: line {line_no} ({record['tool']}): recorded ok={record['ok']}, "
            f"replay got isError={result['isError']}: {result}"
        )

        # Assertion 2: the produced-uid count, reproduced -- a call that used
        # to make two objects and now makes one is exactly the regression
        # this tier exists to catch, reported as that rather than as a
        # confusing refusal three calls later when the second uid never
        # arrives.
        produced = _extract_produced_uids(result)
        assert len(produced) == len(record["made"]), (
            f"{name}: line {line_no} ({record['tool']}): produced {len(produced)} uids "
            f"{produced}, recording claims {len(record['made'])}: {record['made']}."
        )
        # A recorded uid that already has a live mapping must map to the
        # same uid again. It normally will -- a result lists uids in
        # ``doc.objects`` order, which is creation order and deterministic
        # for a given transcript -- so a disagreement means the replay
        # surfaced the same objects in a *different* order than the
        # recording did. That is a real divergence, and without this it
        # would be silent: the mapping would simply be overwritten, and the
        # next line would address the wrong object while every count above
        # still matched.
        for was, now in zip(record["made"], produced, strict=True):
            assert mapping.setdefault(was, now) == now, (
                f"{name}: line {line_no} ({record['tool']}): recorded uid {was} "
                f"mapped to live uid {mapping[was]} earlier and to {now} here -- "
                "the replay surfaced these objects in a different order than the "
                "recording did."
            )

    assert session.tab_uid, f"{name}: the transcript minted no document."
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    doc = tab.doc

    # Assertion 6: the expressiveness measure -- the number a later tranche
    # would want to see fall for the same subject.
    assert len(calls) == expect["call_count"]

    # Assertion 3: the finished shape.
    assert len(doc.objects) == expect["object_count"]

    # Assertion 4: clay/diagnose.findings, per object in creation order (see
    # the module docstring's "choosing the diagnose claim" paragraph for why
    # this is asserted by position rather than by uid or by name).
    findings = [
        sorted(row.kind for row in clay_diagnose.findings(obj.mesh)) for obj in doc.objects
    ]
    assert findings == expect["diagnose_findings"]

    # Assertion 5: the cheap, service-free export stand-in -- see the module
    # docstring's own paragraph on what this does and does not prove.
    restored = serialize.read_wblk(serialize.wblk_bytes(doc))
    assert len(restored.objects) == len(doc.objects)
    # Handed back so a caller can assert on the remap itself -- see
    # ``test_a_transcript_replays_even_though_its_recorded_uids_cannot_exist``,
    # which needs it to prove its own premise actually held.
    return mapping


def test_chair_transcript_builds_a_four_legged_chair() -> None:
    _replay("chair")


def test_spoked_hub_transcript_exercises_batch_ref_and_array_radial() -> None:
    _replay("spoked-hub")


def test_a_transcript_replays_even_though_its_recorded_uids_cannot_exist() -> None:
    """The claim the whole remap rule exists for, and the one the two
    replays above cannot make on their own.

    ``document.new_uid`` is a *process*-wide counter that never hands the
    same number out twice, so in a fresh worker that happens to reach this
    file first the live uids come out 1, 2, 3... -- exactly the numbers the
    hand-authored transcripts carry, and a replay that ignored the mapping
    entirely would pass. Burning a few uids first makes the recorded
    numbers ones this process can never issue, so ``chair``'s closing
    ``clay_material`` (whose ``uids`` names all six objects by their
    *recorded* numbers) reaches a live document that has none of them. It
    passes only because :func:`_remap_uid_arguments` really does rewrite
    them; without the remap it refuses with "no object with uid(s)", and
    without the loud unmapped-uid guard it would refuse for a reason that
    looks nothing like the cause.
    """
    burned = [bd.new_uid() for _ in range(5)]
    mapping = _replay("chair")
    # The premise, asserted rather than assumed: every recorded uid really
    # did come back as a different live one. Without this the test would go
    # quietly vacuous the day uid minting changed shape -- it would still
    # pass, while proving nothing about the remap at all.
    assert all(live > burned[-1] for live in mapping.values()), (burned, mapping)
    assert all(recorded != live for recorded, live in mapping.items()), mapping
