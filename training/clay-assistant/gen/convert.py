"""Converters from a ``drafts/`` record to the shapes training and fixture
promotion need.

* :func:`compact_tools`/:func:`tools_sha` -- the tool card every training
  row's system prompt is built from, derived from the live ``agent_clay``
  registry rather than hand-copied, so a new generator or op changes the
  dataset on the next ``build.py`` run instead of silently going stale (see
  the plan's own paragraph on why the full 26-tool schema set is too much
  prefix for an E2B row with a scene turn attached).
* :func:`compact_scene` -- the "Compact scene for edit/query rows" projection
  the Verifier design section specifies.
* :func:`to_messages` -- one ChatML row for ``--export unsloth``.
* :func:`to_transcript` -- the ``tests/fixtures/agent_transcripts/`` shape,
  for ``--promote``.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from warlock.studio import agent_clay, agent_transcript, clay_mode  # noqa: E402
from warlock.studio.clay import diagnose as clay_diagnose  # noqa: E402

from . import headless  # noqa: E402

KEEP_TOOLS: tuple[str, ...] = (
    "clay_batch",
    "clay_scene",
    "clay_add_primitive",
    "clay_add_figure",
    "clay_transform",
    "clay_set_params",
    "clay_material",
    "clay_boolean",
    "clay_select",
    "clay_op",
    "clay_delete",
    "clay_rename",
    "clay_diagnose",
)
"""The compact tool card's membership, in the order it is printed. Thirteen,
not the plan's original twelve: ``clay_add_figure`` was omitted from the
first count and folded in once figures were counted, per the plan's own
parenthetical."""

BEHAVIOUR_PARAGRAPH = (
    "You are Warlock's Clay assistant. Answer a build request with exactly "
    "one clay_batch tool call whose calls list builds the object; name "
    "every object; put generator parameters under params; place objects so "
    "they rest on the ground; reuse a material index from clay_scene when "
    "one fits, otherwise add one clay_material call at the end. Answer a "
    "question about the scene in one or two sentences with no tool call."
)
"""Verbatim from the plan's "convert.compact_tools()" paragraph -- the one
piece of this card that is prose about training behaviour rather than
derived from a live registry, because nothing in ``agent_clay`` already says
"answer with exactly one clay_batch call"; that is this dataset's own
convention, not the app's."""

_SENTENCE_RE = re.compile(r".+?\.(?=\s|$)", re.S)

_ENUMERATION_PREFIXES = ("Known generators:", "Known ops:", "Parts,")
"""The three catalogue sentences run A's own refusals show a fine-tune is
scored on (measured on 232 val+corpus rows, Q8_0): ``clay_add_primitive``/
``clay_set_params``'s "Known generators: cylinder [radius, height,
segments] ..." (7 refusals for an unknown generator param, e.g. ``depth`` on
a cylinder), ``clay_op``'s "Known ops: array-radial [count, angle, axis]
..." (``clay_op`` given ``axis`` outside ``params`` among 6 other refusals),
and ``clay_add_figure``'s "Parts, each prefixed by name_prefix: humanoid:
Hips, Spine, ..." (15 ``no object named '...'`` refusals, nine of them
creatures-family guesses at a generated figure's own part names --
``hound_Beak``, ``t_Shank.R``, ``s_Tail 01``). ``_first_sentence`` kept only
each description's opening line and trusted the schema's own enums to carry
the rest, which is true for a plain string enum (a generator's *name*, an
op's *name*) but not for what a generator's own params are called, what an
op's own params are called or bounded to, or what a figure preset's own
part names are -- none of that is expressible as a JSON Schema enum here,
because a generator's ``params``/an op's ``params`` is one open
``{string: number}`` object (the value shape varies per key) and a figure's
parts are never an argument at all. These sentences are the only place any
of that is written down."""


def _sentences(text: str) -> list[str]:
    """*text*, split into whole sentences -- the same one-period-plus-
    whitespace-or-end rule ``_first_sentence`` used to apply to the first
    sentence only, walked to the end of the string instead of stopping
    there. A catalogue sentence's own periods (a generator's ``segments=32``
    default, a part name like ``Shoulder.L``) are never followed by
    whitespace, so they never end a sentence early here -- only the period
    that actually closes the sentence, followed by a space or the string's
    end, does."""
    stripped = text.strip()
    sentences: list[str] = []
    pos = 0
    while pos < len(stripped):
        match = _SENTENCE_RE.match(stripped[pos:])
        if not match:
            sentences.append(stripped[pos:])
            break
        sentences.append(match.group(0))
        pos += match.end()
        while pos < len(stripped) and stripped[pos].isspace():
            pos += 1
    return sentences


def _summary(text: str, *, seen: set[str]) -> str:
    """*text*'s first sentence, plus every later sentence beginning
    ``Known generators:``, ``Known ops:`` or ``Parts,`` -- the enumerations
    the model is scored on (see ``_ENUMERATION_PREFIXES``). *seen* is the
    card's own running set of catalogue sentences already printed: dropped
    (a bare first sentence keeps the shorter summary) rather than fenced
    against here, so ``clay_set_params``'s "Known generators: ..." -- word
    for word ``clay_add_primitive``'s own, both built from the same
    :func:`agent_clay._generator_catalog` -- is not printed twice. Measured:
    959 of those chars, once. Anything else in ``_ENUMERATION_PREFIXES`` is
    unique per tool in ``KEEP_TOOLS`` today (``clay_op``'s ops, ``clay_add_
    figure``'s parts), so this dedupe currently only ever fires once, but it
    is a running set rather than a hand-picked "skip clay_set_params" rule
    because the next generator or op added to a second tool's description
    should not have to earn its own special case here."""
    sentences = _sentences(text)
    if not sentences:
        return text.strip()
    keep = [sentences[0]]
    for sentence in sentences[1:]:
        if not sentence.startswith(_ENUMERATION_PREFIXES):
            continue
        if sentence in seen:
            continue
        seen.add(sentence)
        keep.append(sentence)
    return " ".join(keep)


def compact_tools() -> str:
    """The system-prompt tool card every training row shares. Built fresh
    from ``agent_clay.tools()``/``instructions()`` on every call -- never
    cached at import time -- so it can never drift from what the live
    registries actually publish.

    Each tool's own summary keeps its first sentence plus its catalogue
    sentences (:func:`_summary`) rather than the first sentence alone: run A
    (2026-09-13, Q8_0, 232 val+corpus rows) showed exactly the refusals that
    dropping them causes -- 15 ``no object named '...'`` (a figure preset's
    part names, nine of them in ``creatures``), 7 unknown params for a
    generator (e.g. ``depth`` on a cylinder), and ``clay_op`` given ``axis``
    outside ``params`` among 6 other refusals naming an op's own arguments.
    See ``README.md``'s "The compact tool card" section for the fix's own
    accounting."""
    tool_map = {t.name: t for t in agent_clay.tools()}
    instructions = agent_clay.instructions()
    paragraphs = instructions.split("\n\n")

    lines = [paragraphs[0], "", paragraphs[1], "", BEHAVIOUR_PARAGRAPH, "", "Tools:"]
    seen: set[str] = set()
    for name in KEEP_TOOLS:
        tool = tool_map[name]
        summary = _summary(tool.description, seen=seen)
        schema_json = json.dumps(tool.schema, sort_keys=True, separators=(",", ":"))
        lines.append(f"- {name}: {summary}")
        lines.append(f"  schema: {schema_json}")
    return "\n".join(lines)


def tools_sha() -> str:
    """``sha256(compact_tools())``, hex -- what ``build.py`` pins in
    ``dataset/manifest.json`` and every accepted record's own ``verified``
    block, so a registry change (a new generator, say) that regenerates a
    different tool card is caught rather than silently appended past."""
    return hashlib.sha256(compact_tools().encode("utf-8")).hexdigest()


_SCENE_ROW_KEYS: tuple[str, ...] = (
    "uid",
    "name",
    "generator",
    "params",
    "translation",
    "rotation",
    "scale",
    "size",
    "material",
)
"""The Verifier design section's "Compact scene for edit/query rows"
paragraph, verbatim: ``_scene_row`` carries fifteen fields; the training
state turn keeps these nine plus the document's own ``materials``/``bounds``,
dropping ``faces``/``verts``/``stamp``/``selected``/``visible``/``bbox``/
``center`` -- the ones a token budget cannot afford and an edit prompt does
not need to answer."""


def compact_scene(structured: dict[str, Any]) -> dict[str, Any]:
    """*structured* (a ``clay_scene`` ``structuredContent`` payload),
    projected to the fields an edit/query training row's "here is the scene"
    turn actually needs."""
    objects = [
        {key: row.get(key) for key in _SCENE_ROW_KEYS} for row in structured.get("objects", [])
    ]
    return {
        "objects": objects,
        "materials": structured.get("materials", []),
        "bounds": structured.get("bounds"),
    }


def to_messages(
    record: dict[str, Any],
    *,
    tools_text: str,
    inline: bool = False,
    prior_scene: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One ChatML/OpenAI-shaped row: ``{"messages": [...]}``. See
    ``README.md``'s "Message shapes" section for the exact layout of each
    ``record["kind"]``.

    *prior_scene* is the ``compact_scene`` projection of the state
    ``record["prior"]`` built, for an ``edit``/``query`` record -- computed
    by the caller (``build.py``, from a ``verify.replay`` of ``prior``)
    rather than here, because this function is a pure formatter and has no
    business re-running a replay just to get a scene it was handed a moment
    ago by the very code that already ran one.

    *inline*, mirroring ``build.py --inline``: when true, the assistant turn
    for a ``build``/``edit`` record carries the batch call only as a fenced
    ```json``` block in ``content`` (no ``tool_calls`` key at all) -- for a
    template or a trainer that does not model a separate tool-call channel.
    The default carries both: ``tool_calls`` for a trainer that does, and the
    same JSON fenced in ``content`` besides, so either shape can be read off
    one row without regenerating the dataset.
    """
    kind = record.get("kind", "build")
    messages: list[dict[str, Any]] = [{"role": "system", "content": tools_text}]

    if kind in ("edit", "query"):
        scene_text = json.dumps(prior_scene or {}, sort_keys=True, separators=(",", ":"))
        messages.append({"role": "user", "content": "Here is the scene:\n" + scene_text})

    messages.append({"role": "user", "content": record["prompt"]})

    if kind == "query":
        messages.append({"role": "assistant", "content": record["answer"]})
    else:
        calls = record["calls"]
        args_json = json.dumps({"calls": calls}, sort_keys=True, separators=(",", ":"))
        assistant: dict[str, Any] = {
            "role": "assistant",
            "content": f"```json\n{args_json}\n```",
        }
        if not inline:
            assistant["tool_calls"] = [
                {
                    "type": "function",
                    "function": {"name": "clay_batch", "arguments": args_json},
                }
            ]
        messages.append(assistant)

    return {"messages": messages}


def to_transcript(record: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """*record*'s ``prior`` (if any) then ``calls``, replayed **one call at a
    time** -- never folded into a ``clay_batch`` -- through a fresh door, so
    the result is exactly ``tests/fixtures/agent_transcripts/<name>.jsonl``'s
    own shape: what ``--promote`` writes, and what the scaffold tests
    compare against the hand-authored ``chair``/``spoked-hub`` fixtures.

    Deliberately a second, unbatched replay rather than a reuse of
    ``verify.replay``'s batched one: ``verify`` exists to prove what the
    trained model will actually emit (one ``clay_batch`` call) survives the
    door, while this function exists to reproduce the *older*, line-per-call
    transcript format tier one already replays -- two different questions
    about the same record, and conflating them would mean promoting a fixture
    whose own call count no longer matches what a human reading it would
    expect from ``tests/fixtures/agent_transcripts/chair.jsonl``.

    A record meant for this path writes a later call's uid arguments as the
    small integers a fresh authoring run would have produced -- exactly how
    ``chair.jsonl`` already reads ("uids": [1, 2, 3, 4, 5, 6], naming the six
    boxes the six lines before it made) -- rather than as ``{"$ref": ...}``:
    that placeholder is ``_resolve_batch_ref``'s own, resolved only inside a
    ``clay_batch`` entry, and a call made one at a time through this
    function's own loop is never inside one. This function still has to
    turn those small canonical integers into whatever uids *this* run's
    process-wide counter (``document.new_uid``) actually handed out -- built
    the same way ``tests/test_agent_transcripts.py``'s own replay remaps a
    recorded transcript (:func:`agent_transcript.remap`), except the mapping
    here is built from *this* function's own canonical numbering rather than
    read off a recorded ``made`` list the input no longer carries (a
    ``drafts/`` record has none -- see ``build.py``'s own "wrapped as a
    build record" callers). The lines this returns keep each call's
    arguments exactly as given, canonical integers and all, so a promoted
    fixture reads the same way ``chair.jsonl`` already does.
    """
    calls = [*record.get("prior", []), *record.get("calls", [])]

    ctx = headless.HeadlessCtx()
    session = agent_clay.Session()

    mapping: dict[int, int] = {}  # canonical (small, sequential) uid -> this run's live uid
    reverse: dict[int, int] = {}  # the same relation, the other way
    next_canonical = 1

    lines: list[dict[str, Any]] = []
    for line_no, entry in enumerate(calls, start=1):
        canonical_arguments = entry.get("arguments") or {}
        live_arguments = agent_transcript.remap(
            canonical_arguments, mapping, record.get("id", "<record>"), line_no
        )
        result = agent_clay.call(ctx, session, entry["name"], live_arguments)

        made_canonical: list[int] = []
        for live_uid in agent_transcript.produced_uids(result):
            canonical = reverse.get(live_uid)
            if canonical is None:
                canonical = next_canonical
                next_canonical += 1
                mapping[canonical] = live_uid
                reverse[live_uid] = canonical
            made_canonical.append(canonical)

        lines.append(
            {
                "tool": entry["name"],
                "arguments": canonical_arguments,
                "ok": not bool(result.get("isError", False)),
                "made": made_canonical,
            }
        )

    doc = None
    if session.tab_uid:
        tab = clay_mode.ensure(ctx).get(session.tab_uid)
        doc = tab.doc if tab is not None else None

    object_count = len(doc.objects) if doc is not None else 0
    diagnose_findings = (
        [sorted(row.kind for row in clay_diagnose.findings(obj.mesh)) for obj in doc.objects]
        if doc is not None
        else []
    )
    expect = {
        "subject": record["prompt"],
        "difficulty": record.get("difficulty", "basic"),
        "call_count": len(lines),
        "object_count": object_count,
        "diagnose_findings": diagnose_findings,
    }
    return lines, expect
