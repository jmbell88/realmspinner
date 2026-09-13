"""Scaffold tests for ``training/clay-assistant/``. Run on their own, not as
part of the main suite (``pyproject.toml``'s ``testpaths`` is ``["tests"]``,
so nothing here is collected by a bare ``uv run pytest``):

    uv run pytest training/clay-assistant/tests -n 0 -p no:cacheprovider

These exercise the real ``agent_clay.call`` door through
``gen/verify.py``/``gen/convert.py`` -- no mocks -- so a failure here is a
real defect in either the scaffold or (per the plan's own reporting
instruction) in ``src/`` itself.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[3]
_SRC = _ROOT / "src"
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]  # training/clay-assistant
for _p in (_SRC, _PACKAGE_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gen import convert, holdout, verify  # noqa: E402

from warlock.studio.clay import primitives as bp  # noqa: E402

FIXTURES = _ROOT / "tests" / "fixtures" / "agent_transcripts"


def _load_transcript(name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lines = [
        json.loads(line)
        for line in (FIXTURES / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expect = json.loads((FIXTURES / f"{name}.expect.json").read_text(encoding="utf-8"))
    return lines, expect


def _literal_calls(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """*lines*, minus ``ok``/``made`` -- the shape a ``drafts/`` record's own
    ``calls`` field takes."""
    return [{"name": line["tool"], "arguments": line["arguments"]} for line in lines]


def _ref_translate_calls(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """chair.jsonl's own literal-uid convention (predating ``$ref``),
    translated into ``{"$ref": name}`` placeholders so the result is a
    genuinely batchable build record. A real ``clay_batch`` call only
    resolves ``$ref`` -- see ``agent_clay._resolve_batch_ref`` -- so a
    literal recorded uid (which only means anything against the *specific*
    process that recorded it) has to become a name reference before it can
    be replayed as one folded batch in a fresh process with its own uid
    counter.
    """
    name_by_recorded_uid: dict[int, str] = {}
    calls: list[dict[str, Any]] = []
    for line in lines:
        arguments = dict(line["arguments"])
        if line["tool"] == "clay_add_primitive":
            calls.append({"name": line["tool"], "arguments": arguments})
            name = arguments.get("name")
            if name and line["made"]:
                name_by_recorded_uid[line["made"][0]] = name
            continue
        for key in ("uid", "uids"):
            if key not in arguments:
                continue
            value = arguments[key]
            if isinstance(value, list):
                arguments[key] = [
                    {"$ref": name_by_recorded_uid[v]} if v in name_by_recorded_uid else v
                    for v in value
                ]
            elif value in name_by_recorded_uid:
                arguments[key] = {"$ref": name_by_recorded_uid[value]}
        calls.append({"name": line["tool"], "arguments": arguments})
    return calls


# --- 1. chair.jsonl: verifies as a batch, round-trips through to_transcript --


def test_chair_fixture_verifies_and_round_trips_through_to_transcript() -> None:
    lines, expect = _load_transcript("chair")

    # The batched form -- $ref-translated, replayed as one clay_batch, which
    # is what a real build record's "calls" means (see the plan's own
    # record-schema paragraph). Uses whatever uids this process's own
    # document.new_uid() counter is at, never the fixture's recorded 1..6.
    batch_calls = _ref_translate_calls(lines)
    batch_replay = verify.replay(batch_calls)
    assert verify.check(batch_replay) == []

    # The unbatched form -- literal recorded uids, exactly chair.jsonl's own
    # shape, replayed one call at a time through convert.to_transcript. Call
    # count differs on purpose from the batched form above (1 clay_batch
    # call there, 7 individual calls here) -- see the plan's own note on why
    # only object_count/diagnose_findings are compared to the fixture, not
    # call_count.
    record = {
        "id": "fixture-chair",
        "family": "furniture",
        "kind": "build",
        "prompt": "a four-legged wooden chair with a slatted back",
        "calls": _literal_calls(lines),
    }
    got_lines, got_expect = convert.to_transcript(record)
    assert got_expect["object_count"] == expect["object_count"]
    assert got_expect["diagnose_findings"] == expect["diagnose_findings"]
    assert got_lines == lines


# --- 2. spoked-hub.jsonl: a $ref-bearing batch verifies clean ---------------


def test_spoked_hub_batch_with_ref_verifies_clean() -> None:
    lines, _expect = _load_transcript("spoked-hub")
    # lines[0] is itself a recorded clay_batch call; its own "arguments"
    # "calls" is the $ref-bearing sub-call list this test exists to exercise.
    batch_calls = lines[0]["arguments"]["calls"]
    assert any(
        entry["arguments"].get("uids") == [{"$ref": "spoke"}] for entry in batch_calls
    ), "fixture no longer carries the $ref this test means to exercise"

    # allow_below_ground: a spoked hub is a mechanical part built around its
    # own axis, not a thing that rests on a floor -- the fixture's hub and
    # spokes straddle y=0 by design, which the ground-plane check (a
    # training-data convention, not a Clay modelling rule) would otherwise
    # flag. What this test means to prove is that the $ref inside this batch
    # resolves and the built geometry is clean, not that it sits grounded.
    replay = verify.replay(batch_calls)
    assert verify.check(replay, allow_below_ground=True) == []


# --- 3. the held-out corpus and its leak check ------------------------------


def test_holdout_catches_corpus_lines_and_a_paraphrase_but_not_a_different_stool() -> None:
    data = holdout.load_holdout()

    for line in data.lines:
        assert holdout.is_leak(line, data) == line

    paraphrase = "a wooden chair with four legs and a slatted backrest"
    assert holdout.is_leak(paraphrase, data) is not None

    different_subject = "a round wooden stool with three legs"
    assert holdout.is_leak(different_subject, data) is None


# --- 4. the compact tool card -------------------------------------------------


def test_compact_tools_names_batch_and_every_generator_and_hashes_stably() -> None:
    text = convert.compact_tools()
    assert "clay_batch" in text
    for name in bp.GENERATORS:
        assert name in text
    assert convert.tools_sha() == convert.tools_sha()


# --- 5. a degenerate object is rejected, naming itself ----------------------


def test_a_near_zero_size_component_is_rejected_naming_the_object() -> None:
    calls = [
        {
            "name": "clay_add_primitive",
            "arguments": {
                "generator": "box",
                "name": "sliver",
                "params": {"size": [0.001, 1, 1]},
                "translation": [0, 0.5, 0],
            },
        }
    ]
    replay = verify.replay(calls)
    reasons = verify.check(replay)
    assert any("sliver" in reason for reason in reasons)


# --- 6. the ground check, and its named opt-out -----------------------------


def test_ground_check_rejects_and_allow_below_ground_opts_out() -> None:
    calls = [
        {
            "name": "clay_add_primitive",
            "arguments": {
                "generator": "box",
                "name": "buried",
                "params": {"size": [1, 1, 1]},
                "translation": [0, 0, 0],
            },
        }
    ]

    reasons = verify.check(verify.replay(calls))
    assert any("buried" in r and "ground" in r for r in reasons)

    reasons_allowed = verify.check(verify.replay(calls), allow_below_ground=True)
    assert reasons_allowed == []


# --- 7. a no-op edit is rejected ---------------------------------------------


def test_a_no_op_edit_is_rejected() -> None:
    prior = [
        {
            "name": "clay_add_primitive",
            "arguments": {
                "generator": "box",
                "name": "block",
                "params": {"size": [1, 1, 1]},
                "translation": [0, 0.5, 0],
            },
        }
    ]
    # clay_select never pushes an undo step (selection is not undoable by
    # design, per agent_clay.instructions()) -- a guaranteed no-op "edit"
    # regardless of anything about the object it selects.
    edit_calls = [{"name": "clay_select", "arguments": {"uids": [{"$ref": "block"}]}}]

    replay = verify.replay(edit_calls, prior=prior)
    reasons = verify.check(replay)
    assert any("history" in r.lower() or "no change" in r.lower() for r in reasons)
