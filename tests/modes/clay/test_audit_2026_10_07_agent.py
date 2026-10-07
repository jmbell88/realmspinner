"""Regression tests for the 2026-10-07 audit's Clay agent-surface findings
(clay-33, 34, 80 through 88)."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from realmspinner.familiar import contract
from realmspinner.mcp import rpc
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay import ops as clay_ops
from realmspinner.studio.modes.clay.agent import dispatch as agent_clay
from realmspinner.studio.modes.clay.agent import schema as agent_schema

from .test_agent_clay import (
    _TETRA_MESH_ARGS,
    _add_inline_reference_raw,
    _Ctx,
    _history_len,
    _new_agent_tab,
    _payload,
)

TESTS = Path(__file__).resolve().parents[2]


def _doc(ctx: _Ctx, session: agent_clay.Session):
    return clay_mode.ensure(ctx).get(session.tab_uid).doc


# --- clay-33 ------------------------------------------------------------------


def test_a_name_longer_than_the_ceiling_is_refused_by_every_tool_that_takes_a_name() -> None:
    """``MAX_NAME_LENGTH`` reached ``clay_rename``, ``clay_checkpoint`` and
    ``clay_reference_add`` only; the four creation doors (``clay_add_primitive``,
    ``clay_add_mesh``, ``clay_material``, ``clay_group``) took any length, so one
    long name made every ``clay_scene`` page holding it exceed the frame."""
    too_long = "x" * (agent_schema.MAX_NAME_LENGTH + 1)
    exactly = "y" * agent_schema.MAX_NAME_LENGTH

    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session)
    doc = _doc(ctx, session)

    calls = {
        "clay_add_primitive": {"generator": "box", "name": too_long},
        "clay_add_mesh": {**_TETRA_MESH_ARGS, "name": too_long},
        "clay_material": {"uids": [uid], "color": [1, 0, 0], "name": too_long},
        "clay_group": {"uids": [uid], "name": too_long},
        "clay_rename": {"uid": uid, "name": too_long},
        "clay_checkpoint": {"name": too_long},
    }
    for tool, arguments in calls.items():
        history = _history_len(ctx, session)
        objects = len(doc.objects)
        palette = len(doc.materials)
        result = agent_clay.call(ctx, session, tool, arguments)
        assert result["isError"] is True, (tool, "accepted a name past the ceiling")
        assert result["structuredContent"]["field"] == "name", (tool, result)
        assert str(agent_schema.MAX_NAME_LENGTH) in result["content"][0]["text"], tool
        # Refused before the first mutation.
        assert _history_len(ctx, session) == history, tool
        assert len(doc.objects) == objects, tool
        assert len(doc.materials) == palette, tool

    refused = _add_inline_reference_raw(ctx, session, too_long)
    assert refused["isError"] is True
    assert refused["structuredContent"]["field"] == "name"

    # The ceiling is inclusive: a name exactly at it still goes through.
    for tool, arguments in (
        ("clay_add_primitive", {"generator": "box", "name": exactly}),
        ("clay_material", {"uids": [uid], "color": [1, 0, 0], "name": exactly}),
        ("clay_group", {"uids": [uid], "name": exactly[:-1] + "g"}),
    ):
        assert agent_clay.call(ctx, session, tool, arguments)["isError"] is False, tool


def test_a_name_past_the_ceiling_does_not_mint_an_empty_document() -> None:
    """Refused before the tab is resolved: a call that was always going to be
    refused must not mint an empty document to be refused against."""
    ctx = _Ctx()
    session = agent_clay.Session()
    too_long = "x" * (agent_schema.MAX_NAME_LENGTH + 1)
    result = agent_clay.call(
        ctx, session, "clay_add_primitive", {"generator": "box", "name": too_long}
    )
    assert result["isError"] is True
    assert session.tab_uid == ""


# --- clay-34 ------------------------------------------------------------------


def test_the_card_comparison_notices_a_dropped_op(monkeypatch: pytest.MonkeyPatch) -> None:
    """``tests/familiar/test_contract.py``'s card-equality test (clay-34) is only
    worth having if it fails when the surface drifts: drop one op from the live
    registry (what a Clay change does) and the derived card must stop equalling
    the frozen one. Also holds today's equality, so the drift is the only
    difference."""
    assert contract.derive_clay_card() == contract.load_card("clay")
    weld = clay_ops.get("weld")
    monkeypatch.setattr(clay_ops, "OPS", [op for op in clay_ops.OPS if op is not weld])
    assert contract.derive_clay_card() != contract.load_card("clay")


# --- clay-80 ------------------------------------------------------------------


def test_clay_scene_is_still_readable_after_the_palette_has_grown_to_its_ceiling() -> None:
    """``clay_scene`` pages ``objects`` but returns the whole palette; nothing
    bounded the palette, so enough ``clay_material`` calls (a batch takes 32 at a
    time) made every ``clay_scene`` refuse for size with a narrowing it did not
    have. ``clay_material`` now refuses a new slot past ``MAX_PALETTE``, and a
    palette at that ceiling, with the longest names an agent can give, still
    fits one reply."""
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session)
    doc = _doc(ctx, session)

    base = doc.materials[0]
    long_name = "m" * agent_schema.MAX_NAME_LENGTH
    while len(doc.materials) < agent_schema.MAX_PALETTE:
        doc.materials.append(replace(base, name=long_name))

    refused = agent_clay.call(
        ctx, session, "clay_material", {"uids": [uid], "color": [0.1, 0.2, 0.3]}
    )
    assert refused["isError"] is True
    assert refused["structuredContent"]["field"] == "color"
    assert len(doc.materials) == agent_schema.MAX_PALETTE

    # Reusing a slot is still allowed at the ceiling.
    reused = agent_clay.call(ctx, session, "clay_material", {"uids": [uid], "index": 0})
    assert reused["isError"] is False, reused

    scene = agent_clay.call(ctx, session, "clay_scene", {})
    assert scene["isError"] is False, scene["content"][0]["text"][:200]
    assert len(_payload(scene)["materials"]) == agent_schema.MAX_PALETTE


# --- clay-81 ------------------------------------------------------------------


def test_the_catalogue_ceiling_is_within_a_few_percent_of_the_measured_surface() -> None:
    """A ratchet only binds if it sits just above what it guards: the ceiling in
    ``test_agent_clay.py`` kept its pre-shrink figure (78,650) after the surface
    fell to about 51,000, leaving ~27,000 characters of silent growth room."""
    from realmspinner.studio import agent_host

    source = (TESTS / "modes" / "clay" / "test_agent_clay.py").read_text(encoding="utf-8")
    match = re.search(r"CEILING = ([\d_]+)", source)
    assert match is not None, "the catalogue ceiling went missing from test_agent_clay.py"
    ceiling = int(match.group(1).replace("_", ""))

    tools = [*agent_clay.tools(), *agent_host._transport_tools()]
    total = len(json.dumps({"tools": [rpc.tool_dict(t) for t in tools]})) + len(
        agent_clay.instructions()
    )
    assert total <= ceiling, "the surface is over its own ceiling"
    assert ceiling <= total * 1.03, (
        f"the ceiling ({ceiling}) is more than 3% above the measured surface ({total}); "
        "lower it in test_agent_clay.py and say when and why in its docstring"
    )


# --- clay-82 ------------------------------------------------------------------


def test_clay_batchs_description_names_exactly_the_batch_excluded_tools() -> None:
    """The sentence listing what cannot be batched was written out by hand
    while ``BATCH_EXCLUDED`` is the table the handler enforces and the schema's
    ``name`` enum derives from; the description now interpolates it."""
    description = {t.name: t for t in agent_clay.tools()}["clay_batch"].description
    sentence = re.search(r"((?:clay_\w+(?:, | and )?)+) cannot be batched", description)
    assert sentence is not None, description
    named = set(re.findall(r"clay_\w+", sentence.group(1)))
    assert named == set(agent_schema.BATCH_EXCLUDED)


def test_clay_batchs_description_follows_the_excluded_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Widening ``BATCH_EXCLUDED`` reaches the published sentence with no
    second edit (the test above only checks today's table)."""
    monkeypatch.setattr(
        agent_clay,
        "BATCH_EXCLUDED",
        frozenset({*agent_schema.BATCH_EXCLUDED, "clay_measure"}),
    )
    description = {t.name: t for t in agent_clay.tools()}["clay_batch"].description
    assert re.search(r"clay_measure[^.]*cannot be batched", description), description


# --- clay-83 ------------------------------------------------------------------


@pytest.mark.parametrize(
    "spell",
    [
        pytest.param(lambda uid: uid + 0.9, id="fraction"),
        pytest.param(lambda uid: str(uid), id="string"),
        pytest.param(lambda uid: True, id="bool"),
        pytest.param(lambda uid: float("inf"), id="inf"),
        pytest.param(lambda uid: float("nan"), id="nan"),
    ],
)
def test_a_fractional_or_boolean_uid_is_refused_not_truncated(spell) -> None:
    """``int()`` truncates ``9.9`` to uid 9, takes ``True`` for uid 1 and reads
    ``"1"`` as 1, so ``clay_delete uids=[9.9]`` deleted object 9. A uid is a
    whole number, as ``clay_material``'s faces already insist."""
    ctx = _Ctx()
    session = agent_clay.Session()
    target = _new_agent_tab(ctx, session)
    doc = _doc(ctx, session)
    names = [o.name for o in doc.objects]
    history = _history_len(ctx, session)
    value = spell(target)

    for tool, arguments in (
        ("clay_delete", {"uids": [value]}),
        ("clay_transform", {"uid": value, "translation": [1, 0, 0]}),
    ):
        result = agent_clay.call(ctx, session, tool, arguments)
        assert result["isError"] is True, (tool, value)
        assert [o.name for o in doc.objects] == names, (tool, value)
        assert _history_len(ctx, session) == history, (tool, value)


def test_clay_parents_parent_and_clay_measures_uid_refuse_a_fractional_value() -> None:
    """The clay-83 residual: ``clay_parent``'s ``parent`` and ``clay_measure``'s
    point uid still resolved with a bare ``int()``, so ``parent=7.9`` hung the
    child under object 7 and ``{"uid": 7.9}`` measured from object 7."""
    ctx = _Ctx()
    session = agent_clay.Session()
    child = _new_agent_tab(ctx, session)
    other = _payload(
        agent_clay.call(ctx, session, "clay_add_primitive", {"generator": "box"})
    )["uid"]
    doc = _doc(ctx, session)
    history = _history_len(ctx, session)

    refused = agent_clay.call(
        ctx, session, "clay_parent", {"uid": child, "parent": other + 0.9}
    )
    assert refused["isError"] is True, refused
    assert refused["structuredContent"]["field"] == "parent"
    assert doc.by_uid(child).parent is None
    assert _history_len(ctx, session) == history

    measured = agent_clay.call(
        ctx,
        session,
        "clay_measure",
        {"kind": "distance", "a": {"uid": other + 0.9}, "b": [0, 0, 0]},
    )
    assert measured["isError"] is True, measured
    assert measured["structuredContent"]["field"] == "a"

    vertex = agent_clay.call(
        ctx,
        session,
        "clay_measure",
        {"kind": "distance", "a": {"uid": other, "vertex": 0.9}, "b": [0, 0, 0]},
    )
    assert vertex["isError"] is True, vertex
    assert vertex["structuredContent"]["field"] == "a"

    # A whole-number float is still a uid.
    ok = agent_clay.call(
        ctx, session, "clay_parent", {"uid": child, "parent": float(other)}
    )
    assert ok["isError"] is False, ok


def test_a_whole_number_float_uid_still_resolves() -> None:
    """``3.0`` is what a JSON encoder that writes every number as a float sends;
    it is a whole number and keeps working."""
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session)
    result = agent_clay.call(
        ctx, session, "clay_transform", {"uid": float(uid), "translation": [1, 0, 0]}
    )
    assert result["isError"] is False, result


# --- clay-84 ------------------------------------------------------------------


def test_clay_select_by_in_the_wrong_element_mode_recovers_by_switching_mode() -> None:
    """The wrong-mode refusal named ``field='query'``, so ``fail`` derived
    ``recovery='fix_arguments'`` -- but the arguments are fine; the document
    is in the wrong element mode. Every other wrong-mode refusal says
    ``switch_mode``."""
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session)
    result = agent_clay.call(
        ctx, session, "clay_select_by", {"uid": uid, "query": "material", "slot": 0}
    )
    assert result["isError"] is True
    assert result["structuredContent"]["recovery"] == "switch_mode"
    assert result["structuredContent"]["field"] == "query"


# --- clay-85 ------------------------------------------------------------------


def test_instructions_name_only_ops_that_run_with_nothing_selected_as_seedless() -> None:
    """``instructions()`` called select-none seedless, but it is disabled with
    nothing selected (and entering an element mode selects nothing)."""
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session)
    doc = _doc(ctx, session)
    result = agent_clay.call(ctx, session, "clay_element_mode", {"mode": "face"})
    assert result["isError"] is False, result
    assert doc.element_mode == "face"

    selection_ops = ("select-all", "select-none", "select-invert", "select-linked")
    runnable = {name for name in selection_ops if clay_ops.get(name).enabled(doc)}

    text = agent_clay.instructions()
    match = re.search(r"seedless rows -- (.*?) --", text, re.S)
    assert match is not None, "the seedless-rows sentence went missing from instructions()"
    named = set(re.findall(r"select-[a-z]+", match.group(1)))
    assert named == runnable, (named, runnable)


# --- clay-86 ------------------------------------------------------------------


def test_clay_scene_marks_a_textured_palette_slot() -> None:
    """A slot carrying a base-colour texture read as plain white (its factor);
    the row now says ``textured`` and ``nearest`` so an agent does not
    'reuse' it as a colour or repaint over a texture it cannot see."""
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session)
    doc = _doc(ctx, session)

    plain = doc.materials[0]
    assert not plain.base_color
    textured = replace(
        plain,
        name="Crate",
        base_color_factor=(1.0, 1.0, 1.0, 1.0),
        base_color=(1, 1, bytes([200, 100, 50, 255])),
        nearest=True,
    )
    doc.materials.append(textured)

    scene = agent_clay.call(ctx, session, "clay_scene", {})
    rows = _payload(scene)["materials"]
    assert rows[0]["textured"] is False and rows[0]["nearest"] is False
    assert rows[-1]["textured"] is True and rows[-1]["nearest"] is True

    declared = {t.name: t for t in agent_clay.tools()}["clay_scene"].output_schema
    props = declared["properties"]["materials"]["items"]["properties"]
    assert {"textured", "nearest"} <= set(props)


# --- clay-87 ------------------------------------------------------------------


def test_clay_op_reports_the_param_values_it_actually_used() -> None:
    """``clay_ops.run`` clamps every declared param into its range and fills
    defaults; ``clay_op`` used to say only ``ran: true``, so an agent that asked
    for ``thickness=1000`` believed it got 1000."""
    ctx = _Ctx()
    session = agent_clay.Session()
    _new_agent_tab(ctx, session)
    agent_clay.call(ctx, session, "clay_element_mode", {"mode": "face"})
    agent_clay.call(ctx, session, "clay_op", {"name": "select-all"})

    inset = clay_ops.get("inset")
    param = inset.params[0]
    result = agent_clay.call(
        ctx, session, "clay_op", {"name": "inset", "params": {param.name: param.high * 1000}}
    )
    assert result["isError"] is False, result
    body = _payload(result)
    assert body["params"][param.name] == pytest.approx(param.high)
    assert body["clamped"] == [param.name]

    # Defaults are reported too, and an in-range value is not called clamped.
    agent_clay.call(ctx, session, "clay_op", {"name": "undo"})
    in_range = (param.low + param.high) / 2
    result = agent_clay.call(
        ctx, session, "clay_op", {"name": "inset", "params": {param.name: in_range}}
    )
    body = _payload(result)
    assert body["params"][param.name] == pytest.approx(in_range)
    assert body["clamped"] == []

    # An op with no params answers with an empty report, not a missing key.
    result = agent_clay.call(ctx, session, "clay_op", {"name": "select-all"})
    body = _payload(result)
    assert body["params"] == {} and body["clamped"] == []


# --- clay-88 ------------------------------------------------------------------


def test_the_agent_schema_gate_docstring_names_only_tools_that_exist() -> None:
    """``tests/test_agent_schemas.py``'s docstring described a surface of 47
    tools with modifiers, colliders and figures; every one of those was removed
    when Clay became a picoCAD-level modeller."""
    source = (TESTS / "test_agent_schemas.py").read_text(encoding="utf-8")
    docstring = ast.get_docstring(ast.parse(source)) or ""
    live = {t.name for t in agent_clay.tools()}
    named = set(re.findall(r"(?<![\w.])clay_[a-z_]+(?!\w|\.\w)", docstring))
    stale = sorted(named - live)
    assert not stale, f"the gate's docstring names tools that do not exist: {stale}"


def test_clay_measure_does_not_call_the_mesh_it_reads_a_base_mesh() -> None:
    """Clay has no modifier stack, so there is only the mesh; 'base mesh'
    implied a second, modified one."""
    description = {t.name: t for t in agent_clay.tools()}["clay_measure"].description
    assert "base mesh" not in description and "base-mesh" not in description
