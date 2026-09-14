"""The ``drafts/*.jsonl`` record schema -- one JSON object per line. See
``README.md`` for the field-by-field description and the approved plan's
"Phase 1" section (record schema paragraph) for where this came from.

``validate_record`` checks *shape* only: types, required keys, which keys a
given ``kind`` may and may not carry. Whether a shape-valid record actually
builds something clean is ``verify.check``'s job, decided by running it
through the real tool door -- never guessed at here, because a hand-written
rule about what a call "should" do is exactly the kind of guess this whole
programme exists to replace with a real replay.
"""

from __future__ import annotations

from typing import Any

FAMILIES: tuple[str, ...] = (
    "furniture",
    "containers",
    "architecture",
    "mechanical",
    "vehicles",
    "creatures",
    "edits",
    "queries",
    "grounding",
    "figures",
    "composition",
)
"""The eight families the plan's target-count table names, in
``drafts/<family>.jsonl`` filename order. ``build.py --only`` checks a name
against this rather than against whatever files happen to exist, so a typo'd
``--only`` refuses instead of silently matching zero files."""

KINDS: tuple[str, ...] = ("build", "edit", "query")


def _call_list_errors(calls: Any, field_name: str) -> list[str]:
    """Every reason *calls* is not a well-shaped list of ``{"name",
    "arguments"}`` tool-call entries -- shared by ``calls`` and ``prior``,
    which are the same shape wearing two different names."""
    if not isinstance(calls, list) or not calls:
        return [f"{field_name!r} must be a non-empty list of tool calls"]
    errors: list[str] = []
    for i, entry in enumerate(calls):
        if not isinstance(entry, dict):
            errors.append(f"{field_name}[{i}] must be an object")
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            errors.append(f"{field_name}[{i}] is missing a string 'name'")
        arguments = entry.get("arguments", {})
        if arguments is not None and not isinstance(arguments, dict):
            errors.append(f"{field_name}[{i}]['arguments'] must be an object")
    return errors


def validate_record(record: dict[str, Any]) -> list[str]:
    """Every reason *record* is not even shaped like a ``drafts/`` row.
    Empty means the shape is fine -- not that it verifies; see
    ``verify.check`` for that."""
    if not isinstance(record, dict):
        return ["record is not a JSON object"]

    errors: list[str] = []

    rec_id = record.get("id")
    if not isinstance(rec_id, str) or not rec_id:
        errors.append("missing a non-empty string 'id'")

    family = record.get("family")
    if family not in FAMILIES:
        errors.append(f"'family' must be one of {FAMILIES}, got {family!r}")

    kind = record.get("kind")
    if kind not in KINDS:
        errors.append(f"'kind' must be one of {KINDS}, got {kind!r}")

    prompt = record.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        errors.append("missing a non-empty string 'prompt'")

    if kind == "build":
        errors += _call_list_errors(record.get("calls"), "calls")
        if "prior" in record:
            errors.append("a build record must not carry 'prior' (that is edit/query-only)")
        if "answer" in record:
            errors.append("a build record must not carry 'answer' (that is query-only)")
    elif kind == "edit":
        errors += _call_list_errors(record.get("prior"), "prior")
        errors += _call_list_errors(record.get("calls"), "calls")
        if "answer" in record:
            errors.append("an edit record must not carry 'answer' (that is query-only)")
    elif kind == "query":
        errors += _call_list_errors(record.get("prior"), "prior")
        answer = record.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            errors.append("a query record needs a non-empty string 'answer'")
        if "calls" in record:
            errors.append("a query record must not carry 'calls' (it makes no tool call)")

    return errors
