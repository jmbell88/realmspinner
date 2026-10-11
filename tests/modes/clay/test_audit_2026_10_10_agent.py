"""Regression tests for the 2026-10-10 audit's Clay agent-door findings, first
fixer: clay-07 (whole-object paint box-unwraps), clay-08 (no prose says
``clay_select_by`` switches mode), clay-24 (an agent op is not gated by the
user's UV gesture), clay-30 (a params block read from the scene round-trips),
clay-31 (``clay_op`` reports the selection it made), clay-48 (no mutation while
the tab is saving), clay-63 (``bounds`` keeps one currency), clay-66 (argument
refusals name a field), clay-67 (a bad position row is named) and clay-69 (no
empty bracket in the uv catalogue).

Each test's name is the claim.
"""

from __future__ import annotations

import json
import re

import pytest

from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay.agent import dispatch as agent_clay
from realmspinner.studio.modes.clay.agent import schema as agent_schema

from .test_agent_clay import _Ctx, _new_agent_tab, _payload


def _doc(ctx: _Ctx, session: agent_clay.Session):
    return clay_mode.ensure(ctx).get(session.tab_uid).doc


def _text(result: dict) -> str:
    return result["content"][0]["text"]


def _call(ctx: _Ctx, session: agent_clay.Session, name: str, args: dict) -> dict:
    return agent_clay.call(ctx, session, name, args)


_TRI = {"positions": [[0, 0, 0], [1, 0, 0], [0, 1, 0]], "faces": [[0, 1, 2]]}


# --- clay-07 -----------------------------------------------------------------


def test_clay_material_box_unwraps_an_object_with_no_uv_when_the_slot_has_a_texture() -> None:
    ctx, session = _Ctx(), agent_clay.Session()
    _new_agent_tab(ctx, session, "box")
    doc = _doc(ctx, session)
    doc.add_material()
    slot = len(doc.materials) - 1
    doc.add_texture(slot)
    assert doc.materials[slot].base_color is not None

    # clay_material, whole object.
    made = _payload(_call(ctx, session, "clay_add_mesh", {**_TRI, "name": "hand"}))["uid"]
    assert doc.by_uid(made).mesh.uv is None
    result = _call(ctx, session, "clay_material", {"uids": [made], "index": slot})
    assert result["isError"] is False, result
    assert doc.by_uid(made).mesh.uv is not None

    # The ``material`` argument of clay_add_mesh and clay_add_primitive.
    placed = _call(ctx, session, "clay_add_mesh", {**_TRI, "name": "hd", "material": slot})
    assert placed["isError"] is False, placed
    assert doc.by_uid(_payload(placed)["uid"]).mesh.uv is not None

    # ... and the unwrap rides in the paint's one undo step.
    before = len(doc.history.history())
    again = _payload(_call(ctx, session, "clay_add_mesh", {**_TRI, "name": "he"}))["uid"]
    mid = len(doc.history.history())
    assert doc.by_uid(again).mesh.uv is None
    _call(ctx, session, "clay_material", {"uids": [again], "index": slot})
    assert doc.by_uid(again).mesh.uv is not None
    assert len(doc.history.history()) == mid + 1
    assert mid == before + 1


# --- clay-08 -----------------------------------------------------------------


def test_no_published_prose_says_clay_select_by_switches_the_element_mode() -> None:
    tools = {t.name: t for t in agent_clay.tools()}
    assert "mode" not in tools["clay_select_by"].schema["properties"], (
        "the premise: clay_select_by takes no mode"
    )

    published = [agent_clay.instructions()] + [t.description for t in tools.values()]
    for text in published:
        # A sentence that says a mode is switched "in the same call" may name
        # clay_select_by only to deny it.
        for sentence in re.split(r"(?<=[.;])\s+", " ".join(text.split())):
            if re.search(r"same call|in one call", sentence) and "clay_select_by" in sentence:
                assert re.search(r"cannot|takes no mode", sentence), sentence


# --- clay-24 -----------------------------------------------------------------


def test_an_agent_op_is_not_refused_by_a_uv_gesture_armed_on_the_users_active_tab() -> None:
    ctx, session = _Ctx(), agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    agent_tab = clay_mode.ensure(ctx).get(session.tab_uid)
    human = clay_mode.new_document(ctx)  # the user's own tab, now the active one
    assert clay_mode.ensure(ctx).active is human and human is not agent_tab
    human.uv_view.drag_mode = "rotate"  # the user is mid-drag in their UV pane

    assert _call(ctx, session, "clay_select", {"uids": [uid]})["isError"] is False
    result = _call(ctx, session, "clay_op", {"name": "shade-flat"})
    assert result["isError"] is False, _text(result)
    assert "Finish or cancel" not in _text(result)


# --- clay-30 -----------------------------------------------------------------


def test_a_params_block_read_from_clay_scene_round_trips_through_clay_set_params() -> None:
    ctx, session = _Ctx(), agent_clay.Session()
    placed = _call(ctx, session, "clay_add_primitive", {"generator": "stairs"})
    assert placed["isError"] is False, placed
    uid = _payload(placed)["uid"]
    echoed = _payload(placed)["params"]
    assert not any(isinstance(v, bool) for v in echoed.values()), echoed

    scene = _payload(_call(ctx, session, "clay_scene", {}))
    params = next(o for o in scene["objects"] if o["uid"] == uid)["params"]
    assert not any(isinstance(v, bool) for v in params.values()), params
    result = _call(ctx, session, "clay_set_params", {"uid": uid, "params": params})
    assert result["isError"] is False, _text(result)

    # The catalogue prints the same default the door accepts.
    catalogue = _payload(_call(ctx, session, "clay_catalog", {"topic": "primitives"}))[
        "catalogue"
    ]
    assert "True" not in catalogue and "False" not in catalogue
    stairs = next(p for p in catalogue.split("; ") if p.startswith("stairs"))
    assert "closed_underside=1" in stairs
    # ... and a value the door took (0) is echoed as 0, not as a bool.
    off = _payload(
        _call(ctx, session, "clay_set_params", {"uid": uid, "params": {"closed_underside": 0}})
    )
    assert off["params"]["closed_underside"] == 0
    assert not isinstance(off["params"]["closed_underside"], bool)


# --- clay-31 -----------------------------------------------------------------


def test_clay_op_select_all_reports_the_selection_counts_it_made() -> None:
    ctx, session = _Ctx(), agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    assert _call(ctx, session, "clay_element_mode", {"mode": "face"})["isError"] is False

    result = _call(ctx, session, "clay_op", {"name": "select-all"})
    assert result["isError"] is False, _text(result)
    body = _payload(result)
    rows = body["selection"]
    assert [r["uid"] for r in rows] == [uid]
    assert rows[0]["selected"] == {"verts": 0, "edges": 0, "faces": 6}
    assert rows[0]["stamp"] == _doc(ctx, session).mesh_stamp(uid)
    # Counts, never indices: nothing in the reply scales with the mesh.
    assert "[0, 1, 2" not in _text(result)

    cleared = _payload(_call(ctx, session, "clay_op", {"name": "select-none"}))
    assert cleared["selection"] == []


# --- clay-48 -----------------------------------------------------------------


def test_agent_mutation_is_refused_while_its_tab_is_saving() -> None:
    ctx, session = _Ctx(), agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    tab = clay_mode.ensure(ctx).get(session.tab_uid)
    doc = tab.doc
    steps = len(doc.history.history())
    tab.saving = True
    try:
        for name, args in (
            ("clay_rename", {"uid": uid, "name": "x"}),
            ("clay_transform", {"uid": uid, "translation": [1, 0, 0]}),
            ("clay_add_primitive", {"generator": "box"}),
            ("clay_op", {"name": "shade-flat"}),
            ("clay_undo", {}),
            (
                "clay_batch",
                {"calls": [{"name": "clay_rename", "arguments": {"uid": uid, "name": "y"}}]},
            ),
        ):
            result = _call(ctx, session, name, args)
            assert result["isError"] is True, name
            assert "save" in _text(result).lower(), (name, _text(result))
            assert result["structuredContent"]["changed"] is False
        assert len(doc.history.history()) == steps
        assert doc.by_uid(uid).name != "x"

        # Read-only tools stay open.
        for name, args in (
            ("clay_scene", {}),
            ("clay_elements", {}),
            ("clay_catalog", {"topic": "ops"}),
            ("clay_measure", {"kind": "distance", "a": [0, 0, 0], "b": [1, 0, 0]}),
            ("clay_reference_list", {}),
        ):
            assert _call(ctx, session, name, args)["isError"] is False, name
    finally:
        tab.saving = False
    assert _call(ctx, session, "clay_rename", {"uid": uid, "name": "after"})["isError"] is False


def test_every_read_only_tool_named_by_the_saving_gate_is_a_real_tool() -> None:
    names = {t.name for t in agent_clay.tools()}
    assert names >= agent_schema.SAVE_SAFE_TOOLS


# --- clay-63 -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "kept", "dropped"), [("vertex", "verts", "faces"), ("face", "faces", "verts")]
)
def test_select_by_bounds_stores_only_the_current_modes_kind(
    mode: str, kept: str, dropped: str
) -> None:
    ctx, session = _Ctx(), agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    assert _call(ctx, session, "clay_element_mode", {"mode": mode})["isError"] is False
    result = _call(
        ctx,
        session,
        "clay_select_by",
        {"uid": uid, "query": "bounds", "min": [-1, -1, -1], "max": [1, 1, 1]},
    )
    assert result["isError"] is False, _text(result)
    counts = _payload(result)["selected"]
    assert counts[kept] > 0
    assert counts[dropped] == 0 and counts["edges"] == 0, counts
    doc = _doc(ctx, session)
    assert len(getattr(doc.element_sel_of(uid), dropped)) == 0
    assert doc.selection_problems() == []


# --- clay-66 -----------------------------------------------------------------


def test_every_argument_refusal_names_a_field_or_a_recovery() -> None:
    ctx, session = _Ctx(), agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    refusals = [
        _call(ctx, session, "clay_transform", {"uid": uid}),
        _call(ctx, session, "clay_render", {"views": ["front"] * 12, "size": 2048}),
    ]
    for result in refusals:
        assert result["isError"] is True, result
        body = result["structuredContent"]
        assert body.get("field"), body
        assert body.get("recovery") == "fix_arguments", body


# --- clay-67 -----------------------------------------------------------------


def test_clay_add_mesh_names_the_position_row_it_refuses() -> None:
    ctx, session = _Ctx(), agent_clay.Session()
    result = _call(
        ctx,
        session,
        "clay_add_mesh",
        {"positions": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, "x"]], "faces": [[0, 1, 2]]},
    )
    assert result["isError"] is True
    assert "positions[3]" in _text(result), _text(result)
    assert result["structuredContent"]["field"] == "positions"


# --- clay-69 -----------------------------------------------------------------


def test_the_uv_action_catalogue_has_no_empty_bracket() -> None:
    assert not any(desc.rstrip().endswith("[]") for desc in agent_schema.UV_ACTIONS.values())
    ctx, session = _Ctx(), agent_clay.Session()
    text = _payload(_call(ctx, session, "clay_catalog", {"topic": "uv_actions"}))["catalogue"]
    assert "[]" not in text
    assert json.dumps(text)  # still plain prose
