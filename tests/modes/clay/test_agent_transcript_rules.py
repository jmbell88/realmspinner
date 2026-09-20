"""Unit coverage for ``realmspinner.studio.modes.clay.agent.transcript`` -- the one
definition of the transcript rules tier one's replay
(``tests/modes/clay/test_agent_transcripts.py``) and tier two's recorder
(``studio/agent_host.py``) both need. See that module's own docstring for
why it exists rather than two private copies of the same two rules.

``tests/modes/clay/test_agent_transcripts.py`` already exercises :func:`agent_transcript
.produced_uids` and :func:`agent_transcript.remap` on every call two real
transcripts make, through the real replay loop -- the coverage that file
owned before the move keeps holding, because the functions' *bodies* did not
change, only their address. This file covers what a replay never had reason
to: :func:`agent_transcript.record` (nothing in tier one writes a
transcript, only reads one) and :class:`agent_transcript.UnmappedUidError`
raised and caught directly, rather than only observed indirectly through a
transcript built to trigger it.
"""

from __future__ import annotations

import json

import pytest

from realmspinner.studio.modes.clay.agent import transcript as agent_transcript


def test_uid_keys_is_exactly_uid_and_uids() -> None:
    """The same pin ``tests/modes/clay/test_agent_transcripts.py`` carries, proven
    directly against this module: :func:`agent_transcript.uid_keys` is the
    function the whole remap rule rests on, and that should hold with no
    other test file involved."""
    assert agent_transcript.uid_keys() == {"uid", "uids"}
    assert agent_transcript.uid_keys() == agent_transcript.UID_KEYS


def test_produced_uids_reads_a_scalar_uid() -> None:
    result = {
        "content": [{"type": "text", "text": '{"uid": 7, "name": "Box"}'}],
        "isError": False,
        "structuredContent": {"uid": 7, "name": "Box"},
    }
    assert agent_transcript.produced_uids(result) == [7]


def test_produced_uids_reads_every_integer_in_a_uids_list() -> None:
    result = {"isError": False, "structuredContent": {"uids": [3, 4, 5]}}
    assert agent_transcript.produced_uids(result) == [3, 4, 5]


def test_produced_uids_ignores_a_picture_with_no_structured_content() -> None:
    """``clay_render`` and ``clay_reference_get`` create nothing, and carry
    no ``structuredContent`` at all -- correctly zero, not an error."""
    result = {
        "content": [{"type": "image", "data": "iVBORw0KGgo=", "mimeType": "image/png"}],
        "isError": False,
    }
    assert agent_transcript.produced_uids(result) == []


def test_produced_uids_does_not_double_count_the_content_text_echo() -> None:
    """``content``'s text block is a JSON *string* -- a leaf this walk does
    not parse -- so a uid appearing there too (as it always does, since
    ``agent_clay._json`` renders the same payload into both places) is
    counted exactly once, from ``structuredContent``."""
    result = {
        "content": [{"type": "text", "text": '{"uid": 7}'}],
        "isError": False,
        "structuredContent": {"uid": 7},
    }
    assert agent_transcript.produced_uids(result) == [7]


def test_remap_replaces_a_recorded_uid_with_its_live_counterpart() -> None:
    mapped = agent_transcript.remap(
        {"uid": 12, "translation": [0, 1, 0]}, {12: 99}, "chair", 2
    )
    assert mapped == {"uid": 99, "translation": [0, 1, 0]}


def test_remap_walks_a_uids_list_inside_nested_arguments() -> None:
    mapped = agent_transcript.remap({"uids": [1, 2, 3]}, {1: 10, 2: 20, 3: 30}, "t", 1)
    assert mapped == {"uids": [10, 20, 30]}


def test_remap_leaves_a_batch_ref_placeholder_untouched() -> None:
    """``clay_batch``'s own ``{"$ref": "<name>"}`` is not a recorded uid --
    ``agent_clay._resolve_batch_ref`` resolves it by name once the call
    actually runs, so :func:`agent_transcript.remap` must not mistake it for
    one and must not need a live mapping to pass it through."""
    args = {"calls": [{"tool": "clay_transform", "arguments": {"uid": {"$ref": "leg"}}}]}
    assert agent_transcript.remap(args, {}, "t", 1) == args


def test_remap_raises_unmapped_uid_error_naming_the_transcript_and_line() -> None:
    with pytest.raises(agent_transcript.UnmappedUidError) as excinfo:
        agent_transcript.remap({"uid": 12}, {}, "chair", 3)
    message = str(excinfo.value)
    assert "chair" in message
    assert "line 3" in message
    assert "12" in message


def test_unmapped_uid_error_is_a_value_error_not_an_assertion_error() -> None:
    """This module lives in ``src/`` -- see the exception's own docstring
    for why an ``AssertionError`` raised as control flow does not belong
    here, and why a ``ValueError`` subclass is what a ``src/`` module should
    raise for bad input instead."""
    assert issubclass(agent_transcript.UnmappedUidError, ValueError)
    assert not issubclass(agent_transcript.UnmappedUidError, AssertionError)


# --- record(): the write half tier one never needed --------------------------


def test_record_writes_a_line_tier_ones_own_loader_can_read_back(tmp_path) -> None:
    """The claim that actually matters: :func:`agent_transcript.record` and
    tier one's own loader (``tests/modes/clay/test_agent_transcripts.py::
    _load_transcript``, a plain ``json.loads`` per non-blank line) agree on
    the format with no translation step between them -- what makes
    "recorded" and "replayed" the same format rather than two that merely
    look alike.
    """
    path = tmp_path / "subject.jsonl"
    result = {
        "content": [{"type": "text", "text": '{"uid": 3}'}],
        "isError": False,
        "structuredContent": {"uid": 3},
    }

    agent_transcript.record(path, "clay_add_primitive", {"generator": "box"}, result)

    lines = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert lines == [
        {"tool": "clay_add_primitive", "arguments": {"generator": "box"}, "ok": True, "made": [3]}
    ]


def test_record_marks_a_refusal_ok_false_with_nothing_made(tmp_path) -> None:
    path = tmp_path / "subject.jsonl"
    refusal = {"content": [{"type": "text", "text": "no such tool"}], "isError": True}

    agent_transcript.record(path, "clay_nonexistent", {}, refusal)

    (line,) = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert line["ok"] is False
    assert line["made"] == []


def test_record_keeps_the_refusal_message_a_refused_call_answered_with(tmp_path) -> None:
    """The claim the 2026-09-15 Clay agent benchmark sitting bought at full price.

    That sitting recorded twelve refusals and kept not one of their
    messages, so the pre-registration's rule 5 -- every unavoidable refusal
    is a defect and gets written up -- had to be answered by replaying the
    file afterwards, and replay could not answer it: three of the twelve
    refused against live application state a transcript does not carry, and
    came back from the replay as *successes*. The sentence that would have
    settled each one was in hand at record time.
    """
    path = tmp_path / "subject.jsonl"
    refusal = {
        "content": [{"type": "text", "text": "an object is already named 'seat'."}],
        "isError": True,
        "structuredContent": {"changed": False, "field": "name"},
    }

    agent_transcript.record(path, "clay_add_primitive", {"name": "seat"}, refusal)

    (line,) = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert line.get("error") == "an object is already named 'seat'."


def test_record_writes_no_error_key_at_all_for_a_call_that_succeeded(tmp_path) -> None:
    """The other half, and the one that keeps every fixture already under
    ``tests/fixtures/agent_transcripts/`` valid: a successful line carries
    exactly the four keys it has always carried, so the format did not
    change for anything that was not refused.
    """
    path = tmp_path / "subject.jsonl"
    result = {
        "content": [{"type": "text", "text": '{"uid": 3}'}],
        "isError": False,
        "structuredContent": {"uid": 3},
    }

    agent_transcript.record(path, "clay_add_primitive", {"generator": "box"}, result)

    (line,) = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert "error" not in line


def test_record_appends_rather_than_overwrites(tmp_path) -> None:
    path = tmp_path / "subject.jsonl"
    ok_result = {"content": [], "isError": False}
    refused_result = {"content": [{"type": "text", "text": "no"}], "isError": True}

    agent_transcript.record(path, "clay_scene", {}, ok_result)
    agent_transcript.record(path, "clay_add_primitive", {"generator": "nope"}, refused_result)

    lines = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert [line["tool"] for line in lines] == ["clay_scene", "clay_add_primitive"]
    assert [line["ok"] for line in lines] == [True, False]


def test_record_creates_missing_parent_directories(tmp_path) -> None:
    path = tmp_path / "nested" / "deeper" / "subject.jsonl"

    agent_transcript.record(path, "clay_scene", {}, {"content": [], "isError": False})

    assert path.exists()
