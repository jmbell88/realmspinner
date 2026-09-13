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


def _first_sentence(text: str) -> str:
    """*text*, truncated to its first sentence -- descriptions in
    ``agent_clay.tools()`` are full paragraphs meant for a client with an
    unbounded context; the compact card keeps only the summary line and
    trusts the schema's own enums to carry the rest."""
    match = _SENTENCE_RE.match(text.strip())
    return match.group(0) if match else text.strip()


def compact_tools() -> str:
    """The system-prompt tool card every training row shares. Built fresh
    from ``agent_clay.tools()``/``instructions()`` on every call -- never
    cached at import time -- so it can never drift from what the live
    registries actually publish."""
    tool_map = {t.name: t for t in agent_clay.tools()}
    instructions = agent_clay.instructions()
    paragraphs = instructions.split("\n\n")

    lines = [paragraphs[0], "", paragraphs[1], "", BEHAVIOUR_PARAGRAPH, "", "Tools:"]
    for name in KEEP_TOOLS:
        tool = tool_map[name]
        sentence = _first_sentence(tool.description)
        schema_json = json.dumps(tool.schema, sort_keys=True, separators=(",", ":"))
        lines.append(f"- {name}: {sentence}")
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
