"""Tier one of Clay's MCP agent benchmark: replay a **transcript** -- a JSON
Lines sequence of tool calls -- against a real ``ClayDoc`` through the real
``agent_clay.call`` door, and assert what came out. See the tranche-5 plan for
the split: tier one (here) runs unattended in the suite with no model
involved; tier two is ``studio/agent_transcript.py``'s recorder (called from
``studio/agent_host.py``) plus ``dev/scripts/agent_bench.py``'s ``--serve``
driver, which together produce a transcript from an actual agent's own
trajectory over a real MCP connection -- something this file cannot do and
does not try to (``chair`` and ``spoked-hub`` under
``tests/fixtures/agent_transcripts/`` are **hand-authored**; the three
``p43-*`` beside them were **recorded** from a real model, and both kinds say
so below); tier three is human judgement through a ``TODO.md`` sitting.
This file's job is narrower than either: a change that makes a corpus subject
unbuildable, or that silently stops a call producing an object, fails here --
a regression gate, not a benchmark score.

**The format.** One recorded call per line, JSON, at
``tests/fixtures/agent_transcripts/<name>.jsonl``::

    {"tool": "clay_add_primitive", "arguments": {"generator": "box"}, "ok": true, "made": [12]}
    {"tool": "clay_transform", "arguments": {"uid": 12, "translation": [0, 1, 0]},
     "ok": true, "made": []}

``ok`` is whether the recorded call succeeded -- a transcript may legitimately
contain a refusal (a real model's own trajectory includes them), and replay
must reproduce the *same* outcome, not merely avoid crashing. ``made`` is the
object uids that call's result actually carried, in the order
:func:`agent_transcript.produced_uids` would walk them out -- not necessarily
uids that call *created*: a call that only echoes a uid it acted on (say,
``clay_set_params``'s per-object rows) reports that uid here too, because the
extraction rule is structural ("every uid this result surfaced"), not a
judgement about novelty. That is what keeps the recording and the replay
mechanically unable to disagree about what "produced" means -- there is only
one rule, used both times.

``error`` is the fifth key and the only optional one: the refusal's own
sentence, written by :func:`agent_transcript.refusal_text` and present only
on a line whose ``ok`` is false. **Nothing here reads it** -- the replay
asserts the recorded *outcome*, never the recorded wording, because a refusal
message is prose a later change may legitimately reword while refusing for
exactly the same reason, and a test that pinned it would fail on the
rewording and call it a regression. It is written for a human diagnosing a
tier-two session; see that function's docstring for the sitting that earned
it.

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
:func:`agent_transcript.remap`.

There are, today, exactly two such keys: ``uid`` and ``uids``. That is not
asserted by literal -- :func:`agent_transcript.uid_keys` walks every
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
is not a recorded uid at all, so :func:`agent_transcript.remap` passes it
through untouched rather than mistaking it for one; only the batch handler
itself ever learns ``$ref`` exists, exactly as that commit's own docstring
says.

**These four functions moved to ``warlock.studio.agent_transcript``.**
Tier two's recorder needs exactly the same two rules -- which argument names
carry a uid, and which uids a result surfaced -- and a second, private copy
of them here would be the hand-kept-duplicate drift CLAUDE.md refuses
elsewhere in this codebase: the day a twenty-seventh tool named a uid a third
way, only whichever copy someone remembered to update would notice. So
``uid_keys``, ``produced_uids``, ``remap`` and ``record`` are defined once,
in ``src/``, and this file imports them rather than redefining them. Only
the replay loop -- ``_load_transcript`` and ``_replay`` below -- and the two
test-only constants (``UID_KEYS``, an alias for the module's own, and the
gate test that gives it teeth) still live here, because they are about
*this* file's job (asserting a replay reproduces a recording), not about the
format itself.

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

**Two of these transcripts are hand-authored, not recorded.** Describing
either ``chair.jsonl`` or ``spoked-hub.jsonl`` as "recorded" would overclaim
what produced them -- neither ever went through a real MCP connection.
Both were written by reasoning about the geometry by hand, then actually run
once (through this same
``agent_clay.call`` door, with the same ``agent_transcript.produced_uids``
this file uses to replay) to read off the real ``made`` lists and the real
call outcomes -- "authored, verified by execution," not "recorded from an
agent."

**The three ``p43-*`` transcripts are recorded**, by tier two's recorder
during the 2026-09-15 Clay agent benchmark sitting, from a real model driving the real app --
the graded pass written up in
``dev/measurements/2026-09-15-clay-agent-benchmark-results.md``. They are
*slices* of that session's one file, and two edits were made in slicing,
both stated here so the word "recorded" does not overclaim either:
**every ``clay_render`` line is dropped** (it reads the document without
changing it, and needs the GL context ``_Ctx`` deliberately lacks), and each
slice **starts at its document's first building call**, leaving out the
refusals the operator's own closing of the previous document caused at each
handover -- a fresh session here mints its tab and would not refuse. Nothing
else is touched: arguments, outcomes and ``made`` lists are the recorder's
own. ``p43-telescope-colonnade`` is one slice because the model built both
subjects in one document. The serpent, graded −3, was not promoted: this
gate pins what a transcript builds, and nobody wants that build pinned.
"""

from __future__ import annotations

import json
from pathlib import Path

from test_agent_clay import _Ctx  # see module docstring -- shared rather than duplicated

from warlock.kernels.mesh import diagnose as clay_diagnose
from warlock.kernels.mesh import document as bd
from warlock.kernels.mesh import serialize
from warlock.studio import agent_clay, agent_transcript, clay_mode

FIXTURES = Path(__file__).parent / "fixtures" / "agent_transcripts"

UID_KEYS = agent_transcript.UID_KEYS
"""An alias for the one set ``agent_transcript`` derives -- kept as a name in
this file only because the gate test below reads better naming it directly
than spelling ``agent_transcript.UID_KEYS`` out. Not a second computation:
see that module's own docstring for the derivation itself."""


def test_the_uid_bearing_argument_names_are_exactly_uid_and_uids() -> None:
    """The derivation gate the whole remap rule rests on. Walks every
    argument schema :func:`agent_clay.tools` publishes today (28 tools, at
    the time this was written) rather than trusting a hand-written pair, so
    the next tool naming a uid some other way fails loudly here --
    :data:`UID_KEYS` would have already grown to include it, which is what
    forces a decision about :func:`agent_transcript.remap` and
    :func:`agent_transcript.produced_uids` instead of letting a stale
    mapping silently replay the wrong object.
    """
    assert {"uid", "uids"} == UID_KEYS


def test_every_program_step_kinds_uid_bearing_keys_are_within_uid_and_uids() -> None:
    """The gate above is blind to ``clay_program``'s own grammar: its wire
    schema declares ``steps`` items as bare ``{"type": "object"}`` -- and
    ``clay_batch``'s own ``calls[].arguments`` is the identical shape -- so
    :func:`agent_transcript.uid_keys`'s recursive walk of ``properties``
    never reaches whatever keys a compiled *step* actually carries a
    reference under; a step kind that started naming one a third way could
    grow silently with nothing above ever noticing.

    ``agent_program.UID_BEARING_KEYS`` is the compiler's own map of step
    kind -> which of that kind's keys (a subset of its own entry in
    ``agent_program.STEP_KINDS``) carry a reference rather than a plain
    value. Walking it directly closes the blind spot: every key it names is
    asserted to already be a member of ``UID_KEYS`` -- today just ``uid``
    and ``uids``, the same two names the wire-schema gate above already
    pins -- so the compiler and the transcript tooling cannot quietly
    disagree about what counts as a reference.
    """
    from warlock.studio import agent_program as ap

    assert set(ap.UID_BEARING_KEYS) <= set(ap.STEP_KINDS)
    for kind, keys in ap.UID_BEARING_KEYS.items():
        assert keys <= ap.STEP_KINDS[kind], kind
        assert keys <= UID_KEYS, (kind, sorted(keys - UID_KEYS))


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
        # agent_transcript.remap raises UnmappedUidError, a plain
        # ValueError -- the right shape for src/ code (see that class's own
        # docstring), but this replay's failure has always read as an
        # AssertionError, and every caller of _replay (including
        # test_a_transcript_replays_even_though_its_recorded_uids_cannot_exist,
        # which depends on this exact loop raising when it should) expects
        # that. Translated here rather than in agent_transcript itself, which
        # has no business knowing this file uses assertions for its gate.
        try:
            arguments = agent_transcript.remap(
                record.get("arguments") or {}, mapping, name, line_no
            )
        except agent_transcript.UnmappedUidError as exc:
            raise AssertionError(str(exc)) from exc
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
        produced = agent_transcript.produced_uids(result)
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


def test_recorded_p43_chair_builds_ten_parts_in_two_calls() -> None:
    _replay("p43-chair")


def test_recorded_p43_bracket_replays_to_its_three_graded_parts() -> None:
    _replay("p43-bracket")


def test_recorded_p43_telescope_and_colonnade_share_one_document() -> None:
    _replay("p43-telescope-colonnade")


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
    passes only because :func:`agent_transcript.remap` really does rewrite
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
