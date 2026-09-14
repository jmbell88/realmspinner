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

import importlib.util
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

from warlock.studio import clay_ops  # noqa: E402
from warlock.studio.clay import presets  # noqa: E402
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
    assert any(entry["arguments"].get("uids") == [{"$ref": "spoke"}] for entry in batch_calls), (
        "fixture no longer carries the $ref this test means to exercise"
    )

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


def test_compact_tools_names_batch_every_generator_and_op_param_and_every_figure_part_and_hashes_stably() -> (  # noqa: E501
    None
):
    """The claim widened 2026-09-13: it used to be enough that a generator's
    *name* appeared in the card (the schema's own enum was trusted to carry
    the rest), but run A's own refusals (Q8_0, 232 val+corpus rows) showed
    that is not enough -- 7 unknown-param refusals for a generator (e.g.
    ``depth`` on a cylinder), ``clay_op`` given ``axis`` outside ``params``
    among 6 other refusals naming an op's own arguments, and 15 ``no object
    named '...'`` refusals (nine of them creatures-family guesses at a
    generated figure's own part names). This now asserts every generator's
    own param names, every parameterised op's own param names, and every
    figure key's own part names are in the card too -- not just the
    top-level names an enum already carries.

    Fails against the old ``_first_sentence`` card: that card kept only each
    tool's opening line, dropping the "Known generators: ...", "Known ops:
    ..." and "Parts, ..." sentences this asserts against.
    """
    text = convert.compact_tools()
    assert "clay_batch" in text

    for name, (defaults, _builder) in bp.GENERATORS.items():
        assert name in text, f"generator {name!r} missing from the card"
        for param_name in defaults:
            assert param_name in text, f"{name}'s param {param_name!r} missing from the card"

    for op in clay_ops.OPS:
        if not op.params:
            continue
        assert op.name in text, f"op {op.name!r} missing from the card"
        for param in op.params:
            assert param.name in text, f"{op.name}'s param {param.name!r} missing from the card"

    for key in sorted(presets.ASSEMBLIES):
        _label, builder = presets.ASSEMBLIES[key]
        for part in builder():
            assert part.name in text, f"{key}'s part {part.name!r} missing from the card"

    assert convert.tools_sha() == convert.tools_sha()


def test_gen_creatures_part_names_match_the_live_presets_registry() -> None:
    """``drafts/_gen_creatures.py``'s own ``PART_NAMES`` is a copy of
    :func:`presets.build`'s part names, kept as data rather than a live
    import for the reason its own module comment gives -- and that comment
    names the risk this test exists to catch: "a script that silently
    drifts if a preset is ever renamed". Imported via ``importlib`` from its
    path rather than added to ``sys.path`` and given a bare ``import`` --
    ``drafts/`` is a sibling package of authoring scripts sharing generic
    module names (``_gen_furniture.py``, ``_gen_creatures.py``, ...) across
    three other agents' own in-flight files, not a package this test should
    couple itself to by name.

    ``main()`` never runs: it is gated by ``if __name__ == "__main__"``, and
    ``importlib`` gives the loaded module a different ``__name__``.
    """
    path = _ROOT / "training" / "clay-assistant" / "drafts" / "_gen_creatures.py"
    spec = importlib.util.spec_from_file_location("_gen_creatures_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    live = {key: [part.name for part in presets.build(key)] for key in presets.ASSEMBLIES}
    assert live == module.PART_NAMES


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


# --- 8. verify.refusal_reason returns the door's own sentence -----------------


def test_refusal_reason_returns_the_doors_sentence_not_the_whole_batch_blob() -> None:
    """A folded batch that adds a box then refuses on its second call (a ``clay_transform``
    against a ``$ref`` that names nothing): ``refusal_reason`` must return the door's own
    one-sentence refusal -- naming the missing object, not starting with ``{`` -- and it
    must differ from the pre-existing ``_refusal_text``, which (pinned here so the claim
    this function exists to fix is not just asserted but shown) returns the *whole* batch
    result re-serialised as JSON.
    """
    calls = [
        {
            "name": "clay_add_primitive",
            "arguments": {
                "generator": "box",
                "name": "block",
                "params": {"size": [1, 1, 1]},
                "translation": [0, 0.5, 0],
            },
        },
        {
            "name": "clay_transform",
            "arguments": {"uid": {"$ref": "missing"}, "translation": [0, 1, 0]},
        },
    ]
    replay = verify.replay(calls)
    assert replay.main_result.get("isError")

    old = verify._refusal_text(replay.main_result)
    assert old.startswith("{"), "pinning the claim: the old accessor returns a JSON blob"
    assert '"stopped_at"' in old

    new = verify.refusal_reason(replay.main_result)
    assert not new.startswith("{")
    assert "missing" in new
    assert new != old


# --- 9. compare.py reproduces the tracked run-A q8-vs-q4 matrix ---------------


def _import_compare():
    eval_dir = _PACKAGE_ROOT / "eval"
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))
    import compare  # noqa: PLC0415 -- eval/ is not a package; path-inserted above

    return compare


def test_compare_reproduces_run_a_q8_vs_q4_matrix() -> None:
    compare = _import_compare()
    run_a = _ROOT / "docs" / "measurements" / "data" / "clay-assistant" / "run-A"

    a = compare.load(run_a / "eval-A-q8.json")
    b = compare.load(run_a / "eval-A-q4.json")
    ids = compare.common_ids(a, b)
    counts = compare.matrix(a, b, ids)

    assert counts[("accepted", "accepted")] == 113
    assert counts[("accepted", "refused")] == 35
    assert counts[("accepted", "built, failed check")] == 17
    assert counts[("accepted", "no parse")] == 13
    assert counts[("refused", "accepted")] == 11


# --- 10. reason_key groups two refusals differing only in the quoted name ----


def test_reason_key_groups_refusals_differing_only_in_quoted_name() -> None:
    compare = _import_compare()
    a = "no object named 'block'."
    b = "no object named 'other_leg'."
    assert compare.reason_key(a) == compare.reason_key(b)
