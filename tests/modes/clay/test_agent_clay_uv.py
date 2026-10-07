"""``clay_uv`` (``studio/modes/clay/agent/tools_uv.py``): three uv actions
behind one tool (unwrap, pack, transform), and what each promises -- one undo
step, undoable, and a refusal that names a ``field`` for a missing uv and for
an unknown action.

Shares ``_Ctx``/``_payload``/``_new_agent_tab``/``_history_len`` with
``test_agent_clay.py`` rather than duplicating them -- see that module's own
fixtures; every box ``clay_add_primitive`` places already carries a uv
(``primitives.box`` calls ``box_unwrap`` on the way out), which is what makes
a bare ``_new_agent_tab`` enough setup for ``pack``/``density`` and every
refusal test below that does not need one.
"""

from __future__ import annotations

from typing import Any

import pytest

from realmspinner.studio.modes.clay import mode as clay_mode
from realmspinner.studio.modes.clay.agent import dispatch as agent_clay
from realmspinner.studio.modes.clay.agent import schema as agent_clay_schema
from realmspinner.studio.modes.clay.agent import tools_uv as agent_clay_tools_uv

from .test_agent_clay import _Ctx, _history_len, _new_agent_tab, _payload

# A hand-built mesh with no ``uv`` of its own -- ``clay_add_mesh`` never
# invents one -- so ``mesh.uv is None`` for every refusal below that needs an
# object genuinely missing one.
_TETRA_NO_UV = {
    "positions": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    "faces": [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
}


def _new_mesh_tab(ctx: _Ctx, session: agent_clay.Session) -> int:
    result = agent_clay.call(ctx, session, "clay_add_mesh", _TETRA_NO_UV)
    assert result["isError"] is False, result
    return _payload(result)["uid"]


def _doc(ctx: _Ctx, session: agent_clay.Session) -> Any:
    return clay_mode.ensure(ctx).get(session.tab_uid).doc


def _head(ctx: _Ctx, session: agent_clay.Session) -> int:
    """The document's own undo-position serial (``UndoStack.head``) --
    unlike ``_history_len`` (``len(history())``, the *combined* done+undone
    log this file's sibling module uses to prove a step was pushed), this is
    what actually moves backwards on ``clay_undo``, so it is what this
    module's own "and undoable" half of each test checks instead."""
    return _doc(ctx, session).history.head


# --- the derived enum, both ways ----------------------------------------------


def test_the_uv_action_enum_matches_uv_actions_both_ways() -> None:
    """``clay_uv``'s own schema enum, ``UV_ACTIONS``' keys and
    ``tools_uv._UV_ACTION_HANDLERS``' keys are the same three names, from three
    different files -- a same-membership pin across all three, so a fourth
    file drifting from the other two (an action added to one dispatch table
    but not the schema, or vice versa) fails here rather than at whichever
    action a caller happens to try first."""
    tool = next(t for t in agent_clay.tools() if t.name == "clay_uv")
    schema_enum = set(tool.schema["properties"]["action"]["enum"])
    assert schema_enum == set(agent_clay_schema.UV_ACTIONS)
    assert schema_enum == set(agent_clay_tools_uv._UV_ACTION_HANDLERS)


# --- one call, one undo step, and undoable ------------------------------------


def test_unwrap_is_one_undo_step_and_undoable() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_mesh_tab(ctx, session)  # a hand-built mesh: no uv until unwrapped
    assert _doc(ctx, session).by_uid(uid).mesh.uv is None
    before_len = _history_len(ctx, session)
    before_head = _head(ctx, session)

    result = agent_clay.call(ctx, session, "clay_uv", {"uid": uid, "action": "unwrap"})
    assert result["isError"] is False, result
    assert _history_len(ctx, session) == before_len + 1
    assert _doc(ctx, session).by_uid(uid).mesh.uv is not None

    undo = agent_clay.call(ctx, session, "clay_undo", {})
    assert undo["isError"] is False, undo
    assert _head(ctx, session) == before_head
    assert _doc(ctx, session).by_uid(uid).mesh.uv is None


def test_transform_is_one_undo_step_and_undoable() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session)
    before_uv = _doc(ctx, session).by_uid(uid).mesh.uv.copy()
    before_len = _history_len(ctx, session)
    before_head = _head(ctx, session)

    result = agent_clay.call(
        ctx, session, "clay_uv", {"uid": uid, "action": "transform", "translate": [0.1, 0.0]}
    )
    assert result["isError"] is False, result
    assert _history_len(ctx, session) == before_len + 1
    assert not (_doc(ctx, session).by_uid(uid).mesh.uv == before_uv).all()

    undo = agent_clay.call(ctx, session, "clay_undo", {})
    assert undo["isError"] is False, undo
    assert _head(ctx, session) == before_head
    assert (_doc(ctx, session).by_uid(uid).mesh.uv == before_uv).all()


def test_pack_is_one_undo_step_and_undoable() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session)
    before_uv = _doc(ctx, session).by_uid(uid).mesh.uv.copy()
    before_len = _history_len(ctx, session)
    before_head = _head(ctx, session)

    result = agent_clay.call(ctx, session, "clay_uv", {"uid": uid, "action": "pack"})
    assert result["isError"] is False, result
    assert _history_len(ctx, session) == before_len + 1

    undo = agent_clay.call(ctx, session, "clay_undo", {})
    assert undo["isError"] is False, undo
    assert _head(ctx, session) == before_head
    assert (_doc(ctx, session).by_uid(uid).mesh.uv == before_uv).all()


# --- refusals -------------------------------------------------------------


@pytest.mark.parametrize("action", ["pack", "transform"])
def test_pack_and_transform_refuse_an_object_with_no_uv(action: str) -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_mesh_tab(ctx, session)
    args = {"uid": uid, "action": action}
    result = agent_clay.call(ctx, session, "clay_uv", args)
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "uid"
    assert result["structuredContent"]["changed"] is False


def test_an_unknown_action_is_refused_by_name() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session)
    result = agent_clay.call(ctx, session, "clay_uv", {"uid": uid, "action": "bogus"})
    assert result["isError"] is True
    assert result["structuredContent"]["field"] == "action"


# --- clay_scene's own uv facts -------------------------------------------------


def test_clay_scene_reports_uv_facts_only_once_the_object_has_one() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session)  # a box: already has a uv
    scene = agent_clay.call(ctx, session, "clay_scene", {})
    assert scene["isError"] is False, scene
    row = next(o for o in _payload(scene)["objects"] if o["uid"] == uid)
    assert "uv" in row
    assert row["uv"]["islands"] > 0

    no_uv_uid = _new_mesh_tab(ctx, session)
    scene2 = agent_clay.call(ctx, session, "clay_scene", {})
    assert scene2["isError"] is False, scene2
    row2 = next(o for o in _payload(scene2)["objects"] if o["uid"] == no_uv_uid)
    assert "uv" not in row2


# --- clay_catalog: the uv_actions topic ----------------------------------------


def test_clay_catalog_uv_actions_matches_uv_actions_verbatim() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    result = agent_clay.call(ctx, session, "clay_catalog", {"topic": "uv_actions"})
    assert result["isError"] is False, result
    assert _payload(result)["catalogue"] == agent_clay_schema._uv_action_catalog()
    for name in agent_clay_schema.UV_ACTIONS:
        assert name in _payload(result)["catalogue"]


# --- a batch mixing two uv actions folds into one step -------------------------


def test_a_batch_mixing_two_uv_actions_folds_into_one_step() -> None:
    ctx = _Ctx()
    session = agent_clay.Session()
    uid = _new_agent_tab(ctx, session)
    before_uv = _doc(ctx, session).by_uid(uid).mesh.uv.copy()
    before_len = _history_len(ctx, session)
    before_head = _head(ctx, session)

    result = agent_clay.call(
        ctx,
        session,
        "clay_batch",
        {
            "calls": [
                {"name": "clay_uv", "arguments": {"uid": uid, "action": "unwrap"}},
                {"name": "clay_uv", "arguments": {"uid": uid, "action": "pack"}},
            ]
        },
    )
    assert result["isError"] is False, result
    assert _history_len(ctx, session) == before_len + 1

    undo = agent_clay.call(ctx, session, "clay_undo", {})
    assert undo["isError"] is False, undo
    assert _head(ctx, session) == before_head
    assert (_doc(ctx, session).by_uid(uid).mesh.uv == before_uv).all()
