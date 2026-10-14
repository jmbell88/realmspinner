"""The 2026-10-14 audit, fixer F22: the clay_program compiler's budgets and names, the add tools'
all-zero scale, ``clay_batch``'s published enum, the world-space live steps, and the prose an
agent reads about the rig.

Findings agents-clay-01..07, 09 (agents-15 here) and clay-agent-tools-01/-03/-04.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay.agent import dispatch as agent_clay
from realmspinner.studio.modes.clay.agent import program as ap

from .test_agent_clay import _Ctx, _new_agent_tab, _payload


def compile_err(program: dict) -> ap.ProgramError:
    with pytest.raises(ap.ProgramError) as exc_info:
        ap.compile_program(program)
    return exc_info.value


def _box(**extra) -> dict:
    return {"add": {"generator": "box", **extra}}


# --- agents-clay-01: a body that emits nothing still spends the compile budget ---------------


def test_nested_repeats_with_a_body_that_emits_no_call_are_refused_before_they_spend_seconds_compiling():  # noqa: E501
    program = {
        "steps": [
            {
                "repeat": {
                    "ranges": {"i": {"from": 0, "to": 64}},
                    "steps": [
                        {
                            "repeat": {
                                "ranges": {"j": {"from": 0, "to": 64}},
                                "steps": [
                                    {
                                        "repeat": {
                                            "ranges": {"k": {"from": 0, "to": 64}},
                                            "steps": [{"let": {"vars": {"v": 1}}}],
                                        }
                                    }
                                ],
                            }
                        }
                    ],
                }
            }
        ]
    }
    started = time.perf_counter()
    error = compile_err(program)
    elapsed = time.perf_counter() - started
    assert "PROGRAM_MAX_VISITS" in error.reason
    assert elapsed < 1.0, f"the refusal took {elapsed:.2f}s -- it must not walk 262,144 passes"


def test_a_program_inside_the_visit_budget_still_compiles():
    body = [{"let": {"vars": {"a": 1}}}, {"let": {"vars": {"b": 2}}}, _box(id="p{i}")]
    compiled = ap.compile_program(
        {"steps": [{"repeat": {"ranges": {"i": {"from": 0, "to": 64}}, "steps": body}}]}
    )
    assert compiled.expanded == 64


# --- agents-clay-02: the range-name count is capped before any range is evaluated -------------


def test_a_repeat_with_more_range_names_than_the_variable_cap_is_refused_without_evaluating_them():
    # Every value is an invalid expression: were any range evaluated first, the refusal would be
    # "bad token", not the cap.
    ranges = {f"k{i}": ["@@"] for i in range(ap.PROGRAM_MAX_VARIABLES + 1)}
    error = compile_err({"steps": [{"repeat": {"ranges": ranges, "steps": [_box()]}}]})
    assert "PROGRAM_MAX_VARIABLES" in error.reason


def test_a_huge_range_object_is_refused_in_a_blink():
    ranges = {f"k{i}": [1] for i in range(50_000)}
    started = time.perf_counter()
    error = compile_err({"steps": [{"repeat": {"ranges": ranges, "steps": [_box()]}}]})
    assert "PROGRAM_MAX_VARIABLES" in error.reason
    assert time.perf_counter() - started < 1.0


# --- agents-clay-03: a float range never reaches its exclusive upper bound ---------------------


def _range_values(rng: dict) -> list[float]:
    compiled = ap.compile_program(
        {
            "steps": [
                {
                    "repeat": {
                        "ranges": {"i": rng},
                        "steps": [_box(translation=["$i", 0, 0])],
                    }
                }
            ]
        }
    )
    return [call[1]["translation"][0] for call in compiled.calls]


def test_a_float_step_range_never_includes_its_exclusive_upper_bound():
    values = _range_values({"from": 0, "to": 1, "step": 0.1})
    assert len(values) == 10
    assert max(values) < 1.0
    assert values[-1] == pytest.approx(0.9)
    # The whole-number cases keep their meaning.
    assert _range_values({"from": 0, "to": 6, "step": 2}) == [0.0, 2.0, 4.0]
    # 1.1 / 0.1 is 11.000000000000002: the eleventh value is 1.1 itself, which is not below it.
    assert len(_range_values({"from": 0, "to": 1.1, "step": 0.1})) == 11
    assert len(_range_values({"from": 0, "to": 0.3, "step": 0.1})) == 3


def test_a_descending_float_range_stops_before_its_exclusive_lower_bound():
    values = _range_values({"from": 1, "to": 0, "step": -0.1})
    assert len(values) == 10
    assert min(values) > 0.0


# --- agents-clay-04: a group declared inside a repeat is templated like an add ---------------


def test_a_group_inside_a_repeat_templates_its_id_and_members():
    compiled = ap.compile_program(
        {
            "steps": [
                {
                    "repeat": {
                        "ranges": {"i": [1, 2]},
                        "steps": [
                            _box(id="leg_{i}"),
                            {"group": {"id": "g_{i}", "members": ["leg_{i}"]}},
                        ],
                    }
                }
            ]
        }
    )
    assert compiled.groups == {"g_1": ("leg_1",), "g_2": ("leg_2",)}


# --- agents-clay-05: an id a condition cannot name is explained when it is asserted --------------


@pytest.mark.parametrize("oid", ["front-left", "pi", "wheel.L", "1st"])
def test_an_id_that_a_condition_cannot_name_is_refused_when_it_is_declared_or_explained_when_asserted(  # noqa: E501
    oid,
):
    program = {
        "steps": [
            _box(id=oid),
            {"assert": {"condition": f"exists({oid})"}},
        ]
    }
    try:
        ap.compile_program(program)
    except ap.ProgramError as error:
        # Refused either way; the refusal has to name the id and say a condition cannot spell it.
        assert oid in error.reason
        assert "condition" in error.reason and "cannot be written" in error.reason
    else:
        pytest.fail(f"{oid!r} compiled and so would be asserted on, but a condition cannot name it")


def test_an_id_a_condition_can_name_still_asserts():
    ap.compile_program(
        {"steps": [_box(id="front_left_2"), {"assert": {"condition": "exists(front_left_2)"}}]}
    )


# --- agents-clay-06: an op that consumes objects marks them dead at compile time ----------------


@pytest.mark.parametrize("op", ["join", "union", "difference", "intersection"])
def test_an_id_joined_by_an_op_step_is_refused_at_compile_time_when_a_later_step_names_it(op):
    program = {
        "steps": [
            _box(id="a"),
            _box(id="b", translation=[2, 0, 0]),
            {"op": {"name": op, "uids": ["a", "b"]}},
            {"transform": {"uid": "b", "translation": [1, 0, 0]}},
        ]
    }
    error = compile_err(program)
    assert "'b'" in error.reason and op in error.reason
    assert "consumed" in error.reason


def test_the_survivor_of_a_join_is_the_first_made_whatever_order_the_uids_are_listed_in():
    # The outliner's topmost selected object survives, and a program's objects are made in order.
    compiled = ap.compile_program(
        {
            "steps": [
                _box(id="a"),
                _box(id="b"),
                {"op": {"name": "join", "uids": ["b", "a"]}},
                {"transform": {"uid": "a", "translation": [1, 0, 0]}},
            ]
        }
    )
    assert compiled.calls[-1][0] == "clay_transform"


def test_an_op_delete_marks_every_named_id_dead():
    error = compile_err(
        {
            "steps": [
                _box(id="a"),
                {"op": {"name": "delete", "uids": ["a"]}},
                {"transform": {"uid": "a", "translation": [1, 0, 0]}},
            ]
        }
    )
    assert "consumed" in error.reason


def test_a_join_after_a_regrouping_op_does_not_guess_the_survivor():
    # group reorders the outliner, so which object survives is no longer knowable here.
    compiled = ap.compile_program(
        {
            "steps": [
                _box(id="a"),
                _box(id="b"),
                _box(id="c"),
                {"op": {"name": "group", "uids": ["b", "c"]}},
                {"op": {"name": "join", "uids": ["a", "b"]}},
                {"transform": {"uid": "b", "translation": [1, 0, 0]}},
            ]
        }
    )
    assert compiled.calls[-1][0] == "clay_transform"


# --- agents-clay-07: an oversize range names the count that was asked for -----------------------


def test_an_oversize_array_count_is_refused_naming_the_count_asked_for():
    error = compile_err(
        {"steps": [{"array": {"count": 1000, "add": {"generator": "box"}}}]}
    )
    assert "1000" in error.reason and "PROGRAM_MAX_REPEAT" in error.reason
    assert "65" not in error.reason


def test_an_oversize_from_to_range_names_the_count_it_would_produce():
    error = compile_err(
        {"steps": [{"repeat": {"ranges": {"i": {"from": 0, "to": 500}}, "steps": [_box()]}}]}
    )
    assert "500" in error.reason and "PROGRAM_MAX_REPEAT" in error.reason


def test_an_astronomical_range_is_refused_without_a_number_or_a_hang():
    error = compile_err(
        {
            "steps": [
                {
                    "repeat": {
                        "ranges": {"i": {"from": 0, "to": 1e300, "step": 1e-300}},
                        "steps": [_box()],
                    }
                }
            ]
        }
    )
    assert "PROGRAM_MAX_REPEAT" in error.reason


# --- clay-agent-tools-04: nothing in the batch enum is refused by the batch runtime --------------


def test_no_tool_in_the_batch_name_enum_is_refused_by_the_batch_runtime():
    batch = {t.name: t for t in agent_clay.tools()}["clay_batch"]
    enum = batch.schema["properties"]["calls"]["items"]["properties"]["name"]["enum"]
    assert "clay_checkpoint" not in enum
    ctx, session = _Ctx(), agent_clay.Session()
    _new_agent_tab(ctx, session)
    for name in enum:
        result = agent_clay.call(ctx, session, "clay_batch", {"calls": [{"name": name}]})
        text = json.dumps(result.get("content"))
        assert "cannot run inside a batch" not in text, name


def test_a_clay_checkpoint_named_in_a_batch_is_refused_up_front_saying_where_it_belongs():
    ctx, session = _Ctx(), agent_clay.Session()
    _new_agent_tab(ctx, session)
    before = _state(ctx, session)
    result = agent_clay.call(
        ctx,
        session,
        "clay_batch",
        {
            "calls": [
                {"name": "clay_add_primitive", "arguments": {"generator": "box"}},
                {"name": "clay_checkpoint", "arguments": {"name": "a"}},
            ]
        },
    )
    assert result["isError"] is True
    assert "before or after the batch" in result["content"][0]["text"]
    assert result["structuredContent"]["field"] == "calls"
    assert _state(ctx, session) == before, "the refusal came before the first call ran"


def test_the_runtime_refusal_of_clay_checkpoint_in_a_fold_survives_as_a_backstop():
    from realmspinner.studio.modes.clay.agent import tools_batch

    ctx, session = _Ctx(), agent_clay.Session()
    _new_agent_tab(ctx, session)
    doc = clay_mode.ensure(ctx).get(session.tab_uid).doc
    result = tools_batch._resolve_and_call(ctx, session, doc, "clay_checkpoint", {"name": "a"})
    assert result["isError"] is True
    assert "cannot run inside a batch" in result["content"][0]["text"]


# --- clay-agent-tools-01: an all-zero scale places nothing and leaves no gesture open ------------


def _state(ctx, session) -> tuple[int, int, int]:
    doc = clay_mode.ensure(ctx).get(session.tab_uid).doc
    return len(doc.objects), len(doc.history.history()), doc.history._open_gestures


def test_add_primitive_and_add_mesh_with_an_all_zero_scale_place_nothing_and_leave_no_gesture_open():  # noqa: E501
    ctx, session = _Ctx(), agent_clay.Session()
    _new_agent_tab(ctx, session)
    before = _state(ctx, session)
    result = agent_clay.call(
        ctx, session, "clay_add_primitive", {"generator": "box", "scale": [0, 0, 0]}
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "scale"
    assert _state(ctx, session) == before
    result = agent_clay.call(
        ctx,
        session,
        "clay_add_mesh",
        {
            "positions": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            "faces": [[0, 1, 2]],
            "scale": [0, 0, 0],
        },
    )
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "scale"
    assert _state(ctx, session) == before


@pytest.mark.parametrize("tool", ["clay_add_primitive", "clay_add_mesh"])
def test_a_set_transform_refusal_after_the_add_takes_the_object_back_out(tool, monkeypatch):
    from realmspinner.kernels.mesh.elements import OpError

    ctx, session = _Ctx(), agent_clay.Session()
    _new_agent_tab(ctx, session)
    doc = clay_mode.ensure(ctx).get(session.tab_uid).doc
    before = _state(ctx, session)

    def refuse(*_a, **_k):
        raise OpError("refused after the add")

    monkeypatch.setattr(type(doc), "set_transform", refuse)
    args = (
        {"generator": "box", "translation": [1, 0, 0]}
        if tool == "clay_add_primitive"
        else {
            "positions": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            "faces": [[0, 1, 2]],
            "translation": [1, 0, 0],
        }
    )
    result = agent_clay.call(ctx, session, tool, args)
    assert result["isError"] is True
    assert result["structuredContent"]["changed"] is False
    assert _state(ctx, session) == before


# --- clay-agent-tools-03: move/turn compose in world space for a parented object ----------------


def _parented_child(call, child: int) -> int:
    parent = _payload(
        call(
            "clay_add_primitive",
            {"generator": "box", "name": "P", "rotation": [0, 0, 90], "scale": [2, 2, 2]},
        )
    )["uid"]
    result = call("clay_parent", {"uid": child, "parent": parent, "keep_world": True})
    assert not result["isError"], result
    return parent


def _world_row(call, uid: int) -> dict:
    return next(o for o in _payload(call("clay_scene"))["objects"] if o["uid"] == uid)


def test_program_turn_and_move_compose_in_world_space_for_a_parented_object():
    ctx, session = _Ctx(), agent_clay.Session()
    call = lambda name, args=None: agent_clay.call(ctx, session, name, args or {})  # noqa: E731
    child = _new_agent_tab(ctx, session)
    _parented_child(call, child)
    before = _world_row(call, child)
    assert np.allclose(before["rotation"], [0, 0, 0], atol=1e-3)

    result = call("clay_program", {"steps": [{"turn": {"uid": child, "by": [90, 0, 0]}}]})
    assert not result["isError"], result
    after = _world_row(call, child)
    # A +90 world turn about X, from a world rotation of zero, is Rx(90) -- not the Z/Y mixture a
    # turn applied in the parent's frame gives.
    assert np.allclose(after["rotation"], [90, 0, 0], atol=1e-3), after["rotation"]
    assert np.allclose(after["translation"], before["translation"], atol=1e-4)

    result = call("clay_program", {"steps": [{"move": {"uid": child, "by": [0, 1, 0]}}]})
    assert not result["isError"], result
    moved = _world_row(call, child)
    assert np.allclose(
        np.asarray(moved["translation"]) - np.asarray(after["translation"]), [0, 1, 0], atol=1e-4
    ), moved["translation"]


def test_program_turn_and_move_on_an_unparented_object_are_unchanged():
    ctx, session = _Ctx(), agent_clay.Session()
    call = lambda name, args=None: agent_clay.call(ctx, session, name, args or {})  # noqa: E731
    child = _new_agent_tab(ctx, session)
    call("clay_program", {"steps": [{"turn": {"uid": child, "by": [0, 0, 90]}}]})
    call("clay_program", {"steps": [{"move": {"uid": child, "by": [1, 2, 3]}}]})
    row = _world_row(call, child)
    assert np.allclose(row["rotation"], [0, 0, 90], atol=1e-3)
    assert np.allclose(row["translation"], [1, 2, 3], atol=1e-4)


# --- agents-clay-09 (agents-15): the prose an agent reads names the rig ---------------------------


def test_the_scene_description_and_instructions_mention_the_rig_block():
    tools = {t.name: t for t in agent_clay.tools()}
    scene = tools["clay_scene"].description.lower()
    assert "rig" in scene and "bone" in scene and "animation" in scene
    instructions = agent_clay.instructions().lower()
    for word in ("rig", "skin", "animation", "add-bone", "mark-bone", "unmark-bone", "unbind-skin"):
        assert word in instructions, word
    # The first two paragraphs and the opening sentence of each card tool are the frozen card.
    program = tools["clay_program"].description
    assert "_mirror" in program


def test_the_program_description_says_what_a_condition_can_name_and_what_a_join_consumes():
    program = {t.name: t for t in agent_clay.tools()}["clay_program"].description
    assert "PROGRAM_MAX_VISITS" not in program  # the number, not the constant's name, is published
    assert str(ap.PROGRAM_MAX_VISITS) in program
    assert "join" in program and "consume" in program
