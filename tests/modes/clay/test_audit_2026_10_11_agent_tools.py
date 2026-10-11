"""The 2026-10-11 audit's Clay agent-surface findings (``clay_op``, ``clay_batch``,
``clay_program``, ``clay_render``, ``clay_uv``): clay-19, -20, -47, -48, -49, -50
and -51.

Shares ``_Ctx``/``_payload``/``_new_agent_tab`` with ``test_agent_clay.py``.
"""

from __future__ import annotations

import time as real_time
from typing import Any

import pytest

from realmspinner.kernels.mesh import mesh as bm
from realmspinner.kernels.mesh import ops_simplify, uvtools
from realmspinner.kernels.mesh.elements import OpError
from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay import ops as clay_ops
from realmspinner.studio.modes.clay.agent import dispatch as agent_clay
from realmspinner.studio.modes.clay.agent import schema as agent_clay_schema
from realmspinner.studio.modes.clay.agent import tools_batch as agent_clay_tools_batch

from .test_agent_clay import _add_inline_reference, _Ctx, _new_agent_tab, _payload


def _text(result: dict) -> str:
    return result["content"][0]["text"]


def _two_spheres() -> tuple[_Ctx, agent_clay.Session, list[int]]:
    """A session whose tab holds two 128-face spheres, both selected."""
    ctx, session = _Ctx(), agent_clay.Session()
    a = _new_agent_tab(ctx, session, "uv_sphere")
    b = _payload(agent_clay.call(ctx, session, "clay_add_primitive", {"generator": "uv_sphere"}))[
        "uid"
    ]
    result = agent_clay.call(ctx, session, "clay_select", {"uids": [a, b]})
    assert result["isError"] is False, result
    return ctx, session, [a, b]


def _simplify(ctx: _Ctx, session: agent_clay.Session, **params: Any) -> dict:
    return agent_clay.call(ctx, session, "clay_op", {"name": "simplify", "params": params})


# --- clay-19 -----------------------------------------------------------------


def test_clay_op_simplify_reports_the_refused_object_when_another_object_in_the_selection_ran(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A multi-object press that pushed a step for one object and was refused for
    another answered ``ran: true, pushed: true`` with the refusal as one anonymous
    string in ``messages``; an agent read it as "every selected object was
    simplified". The refusal now also arrives as ``errors``."""
    ctx, session, _uids = _two_spheres()
    real = ops_simplify.simplify
    seen: list[Any] = []

    def refuse_the_second(mesh: Any, target: int) -> Any:
        seen.append(mesh)
        if len(seen) == 2:
            raise OpError("This object is too tangled to simplify.")
        return real(mesh, target)

    monkeypatch.setattr(ops_simplify, "simplify", refuse_the_second)
    result = _simplify(ctx, session, target=10)

    assert result["isError"] is False, _text(result)
    payload = _payload(result)
    assert payload["ran"] is True and payload["pushed"] is True
    assert payload["errors"] == ["This object is too tangled to simplify."]
    assert len(payload["changed"]) == 1


def test_clay_op_reply_carries_no_errors_key_when_nothing_was_refused() -> None:
    ctx, session, _uids = _two_spheres()
    payload = _payload(_simplify(ctx, session, target=10))
    assert payload["pushed"] is True
    assert "errors" not in payload


# --- clay-20 -----------------------------------------------------------------


def _slow_entries(monkeypatch: pytest.MonkeyPatch, seconds: float) -> None:
    original = agent_clay_tools_batch._resolve_and_call

    def slow(ctx: Any, session: Any, doc: Any, name: str, arguments: dict) -> Any:
        result = original(ctx, session, doc, name, arguments)
        real_time.sleep(seconds)  # stands in for Simplify or a Blender probe
        return result

    monkeypatch.setattr(agent_clay_tools_batch, "_resolve_and_call", slow)


def test_a_batch_of_slow_entries_stops_at_its_total_time_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each entry's own run time was added back to the idle deadline, so six
    0.2 s entries against a 0.05 s deadline all ran: the published deadline never
    fired for slow work. A second, larger cap on elapsed-since-start does."""
    ctx, session = _Ctx(), agent_clay.Session()
    _new_agent_tab(ctx, session, "box")
    monkeypatch.setattr(agent_clay_tools_batch, "BATCH_DEADLINE_S", 0.05)
    monkeypatch.setattr(agent_clay_tools_batch, "BATCH_TOTAL_S", 0.5, raising=False)
    _slow_entries(monkeypatch, 0.2)

    calls = [{"name": "clay_add_primitive", "arguments": {"generator": "cone"}}] * 6
    result = agent_clay.call(ctx, session, "clay_batch", {"calls": calls})

    assert result["isError"] is True, _text(result)
    payload = _payload(result)
    assert 1 <= payload["stopped_at"] < 6, payload
    assert "total" in _text(result)


def test_a_program_of_slow_steps_stops_at_its_total_time_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx, session = _Ctx(), agent_clay.Session()
    monkeypatch.setattr(agent_clay, "PROGRAM_DEADLINE_S", 0.05)
    monkeypatch.setattr(agent_clay, "PROGRAM_TOTAL_S", 0.5, raising=False)
    _slow_entries(monkeypatch, 0.2)

    steps = [{"add": {"generator": "box"}} for _ in range(6)]
    result = agent_clay.call(ctx, session, "clay_program", {"steps": steps})

    assert result["isError"] is True, _text(result)
    assert _payload(result)["rolled_back"] is True
    assert "total" in _text(result)


def test_the_total_ceilings_are_published_beside_the_idle_deadlines() -> None:
    tools = {t.name: t for t in agent_clay.tools()}
    assert f"{agent_clay_tools_batch.BATCH_TOTAL_S:g}s" in tools["clay_batch"].description
    assert f"{agent_clay.PROGRAM_TOTAL_S:g}s" in tools["clay_program"].description
    assert agent_clay_tools_batch.BATCH_TOTAL_S > agent_clay_tools_batch.BATCH_DEADLINE_S
    assert agent_clay.PROGRAM_TOTAL_S > agent_clay.PROGRAM_DEADLINE_S


# --- clay-47 -----------------------------------------------------------------


def test_clay_render_refuses_alpha_in_beside_mode() -> None:
    """``alpha`` is the overlay blend; in "beside" it was read and dropped, so
    ``alpha=7`` succeeded."""
    ctx, session = _Ctx(), agent_clay.Session()
    _new_agent_tab(ctx, session)
    _add_inline_reference(ctx, session, "ref1")
    for args in (
        {"compare": "ref1", "alpha": 7},
        {"compare": "ref1", "compare_mode": "beside", "alpha": 0.3},
    ):
        result = agent_clay.call(ctx, session, "clay_render", args)
        assert result["isError"] is True, (args, _text(result))
        assert result["structuredContent"]["field"] == "alpha", (args, _text(result))
        assert "overlay" in _text(result)


# --- clay-48 -----------------------------------------------------------------


def test_clay_catalog_ops_says_which_params_are_refused_not_clamped() -> None:
    catalogue = agent_clay_schema.CATALOG_TOPICS["ops"]()
    simplify = next(part for part in catalogue.split("; ") if part.startswith("simplify "))
    assert "refused (not clamped) below 4" in simplify
    snap = next(part for part in catalogue.split("; ") if part.startswith("snap-to-grid "))
    assert "refused (not clamped) below" in snap
    # A param with no refuse_below keeps its plain wording.
    assert "refused" not in next(
        part for part in catalogue.split("; ") if part.startswith("inset ")
    )


def test_clay_op_description_does_not_claim_every_out_of_range_value_is_clamped() -> None:
    desc = {t.name: t for t in agent_clay.tools()}["clay_op"].description
    assert "refused" in desc and "below its floor" in desc


def test_a_rounding_is_not_reported_as_a_clamp() -> None:
    """50.7 -> 50 is a whole-number rounding, not a range clamp; 1e308 -> 1,000,000
    is one."""
    ctx, session, _uids = _two_spheres()
    rounded = _payload(_simplify(ctx, session, target=50.7))
    assert rounded["params"]["target"] == 50
    assert rounded["clamped"] == []
    clamped = _payload(_simplify(ctx, session, target=1e308))
    assert clamped["clamped"] == ["target"]


# --- clay-49 -----------------------------------------------------------------


def test_clay_uv_pack_refusal_over_the_island_ceiling_names_a_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``action=transform`` named field "islands" for the same refusal; pack let
    the ``OpError`` escape to ``call()``'s backstop with no field or recovery."""
    ctx, session = _Ctx(), agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")
    monkeypatch.setattr(uvtools, "MAX_UV_ISLANDS", 2)
    result = agent_clay.call(ctx, session, "clay_uv", {"uid": uid, "action": "pack"})
    assert result["isError"] is True, _text(result)
    assert result["structuredContent"]["field"] == "islands", result
    assert result["structuredContent"]["recovery"] == "fix_arguments"
    assert "islands" in _text(result)


def test_clay_uv_pack_refusal_for_a_non_finite_uv_names_the_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx, session = _Ctx(), agent_clay.Session()
    uid = _new_agent_tab(ctx, session, "box")

    def refuse(mesh: Any, **_: Any) -> Any:
        raise OpError("The uv layout holds a value that is not a number.")

    monkeypatch.setattr(uvtools, "pack_islands", refuse)
    result = agent_clay.call(ctx, session, "clay_uv", {"uid": uid, "action": "pack"})
    assert result["isError"] is True, _text(result)
    assert result["structuredContent"]["field"] == "uid"


# --- clay-50 -----------------------------------------------------------------


def test_clay_op_simplify_refuses_a_target_under_four_faces_naming_params() -> None:
    ctx, session, _uids = _two_spheres()
    result = _simplify(ctx, session, target=2)
    assert result["isError"] is True, _text(result)
    assert _text(result) == "target faces must be at least 4."
    assert result["structuredContent"]["field"] == "params"


def test_clay_op_simplify_clamps_a_huge_target_to_the_ceiling_and_says_nothing_ran() -> None:
    ctx, session, _uids = _two_spheres()
    payload = _payload(_simplify(ctx, session, target=1e308))
    assert payload["params"]["target"] == 1_000_000
    assert payload["clamped"] == ["target"]
    assert payload["ran"] is False and payload["pushed"] is False


def test_clay_op_simplify_refuses_a_multi_object_selection_past_the_corner_ceiling_as_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx, session, uids = _two_spheres()
    doc = clay_mode.ensure(ctx).get(session.tab_uid).doc
    one = ops_simplify.simplify_cost(doc.by_uid(uids[0]).mesh)
    monkeypatch.setattr(ops_simplify, "MAX_SIMPLIFY_CORNERS", one + 1)
    head = doc.history.head
    result = _simplify(ctx, session, target=10)
    assert result["isError"] is True, _text(result)
    assert "Select fewer objects." in _text(result)
    assert doc.history.head == head


def test_clay_op_simplify_reply_carries_the_before_and_after_counts() -> None:
    ctx, session, uids = _two_spheres()
    doc = clay_mode.ensure(ctx).get(session.tab_uid).doc
    before = {uid: bm.face_count(doc.by_uid(uid).mesh) for uid in uids}
    payload = _payload(_simplify(ctx, session, target=10))
    assert "Simplified 256 faces to " in payload["messages"][0]
    assert "each toward 10" in payload["messages"][0]
    for row in payload["changed"]:
        assert row["faces_before"] == before[row["uid"]]
        assert row["faces"] < row["faces_before"]


# --- clay-51 -----------------------------------------------------------------


def test_clay_op_description_says_it_acts_on_the_selection() -> None:
    desc = {t.name: t for t in agent_clay.tools()}["clay_op"].description
    assert "current selection" in desc and "clay_select" in desc


def test_clay_catalog_ops_surfaces_each_ops_hint() -> None:
    """``Op.hint`` was never shown, so an agent learned that Simplify can stay
    above its target only from a reply."""
    catalogue = agent_clay_schema.CATALOG_TOPICS["ops"]()
    simplify = clay_ops.get("simplify")
    assert simplify.hint
    assert simplify.hint in catalogue
    # A bare action with no hint is still one short entry.
    assert "select-all [" in catalogue


# --- the refusal wording another fixer changed in ops.py ----------------------


def test_the_agent_surface_quotes_the_live_wrong_mode_sentence() -> None:
    """The comments and the instructions quoted "Switch to face mode first.", a
    sentence the interface no longer says (the mode switch reads Shape, Points,
    Edges, Faces)."""
    text = agent_clay.instructions()
    assert "Switch to face mode first" not in text
    assert clay_ops.wrong_mode_reason(clay_ops.get("inset")) in text
