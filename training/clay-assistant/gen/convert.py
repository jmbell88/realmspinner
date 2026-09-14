"""Converters from a ``drafts/`` record to the shapes training and fixture
promotion need.

* :func:`compact_tools`/:func:`tools_sha` -- re-exports of
  ``warlock.studio.familiar.contract``'s :func:`~...contract.derive_clay_card`/
  :func:`~...contract.derived_card_sha` (T3 moved the tool-card contract out
  of this module so ``pipelines/llama.py`` and a test could both use it with
  no training import; this module keeps the old names so ``build.py``,
  ``eval/run_val.py``, ``drafts/_gen_queries.py`` and the training tests do
  not need to change).
* :func:`compact_scene` -- re-export of ``contract.compact_scene``, the
  "Compact scene for edit/query rows" projection the Verifier design section
  specifies.
* :func:`to_messages` -- one ChatML row for ``--export unsloth``.
* :func:`to_transcript` -- the ``tests/fixtures/agent_transcripts/`` shape,
  for ``--promote``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from warlock.studio import agent_clay, agent_transcript, clay_mode  # noqa: E402
from warlock.studio.clay import diagnose as clay_diagnose  # noqa: E402
from warlock.studio.familiar import contract  # noqa: E402

from . import headless  # noqa: E402

# Re-exports -- see the module docstring. KEEP_TOOLS/BEHAVIOUR_PARAGRAPH are
# still read by name from this module by a couple of the drafts/ generators;
# kept as aliases rather than re-derived so there is exactly one copy of each.
KEEP_TOOLS = contract.KEEP_TOOLS
BEHAVIOUR_PARAGRAPH = contract.BEHAVIOUR_PARAGRAPH
compact_tools = contract.derive_clay_card
tools_sha = contract.derived_card_sha
compact_scene = contract.compact_scene


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
